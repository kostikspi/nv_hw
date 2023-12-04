import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.utils import inf_loop, MetricTracker
from hw_asr.utils.mel import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.mel = MelSpectrogram(MelSpectrogramConfig())
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        # self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "total_loss",
            "mel_loss",
            "generator_loss",
            "total_discriminator_loss",
            "feature_matching_loss_mpd",
            "feature_matching_loss_msd", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "total_loss",
            "mel_loss",
            "generator_loss",
            "total_discriminator_loss",
            "feature_matching_loss_mpd",
            "feature_matching_loss_msd", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "wave"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["total_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning_rate_generator", self.lr_scheduler["generator"].get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning_rate_discriminator", self.lr_scheduler["discriminator"].get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer["discriminator"].zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        # batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        # batch["log_probs_length"] = self.model.transform_input_lengths(
        #     batch["spectrogram_length"]
        # )
        # batch["loss"] = self.criterion(**batch)
        batch["discriminator_mpd_loss"] = self.criterion["discriminator_loss"](batch["mpd_outputs"],
                                                                               batch["mpd_real_outputs"])
        batch["discriminator_msd_loss"] = self.criterion["discriminator_loss"](batch["msd_outputs"],
                                                                               batch["msd_real_outputs"])
        batch["total_discriminator_loss"] = batch["discriminator_mpd_loss"] + batch["discriminator_msd_loss"]
        if is_train:
            batch["total_discriminator_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["discriminator"].step()
            if self.lr_scheduler is not None:
                self.lr_scheduler["discriminator"].step()
        if is_train:
            self.optimizer["generator"].zero_grad()

        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        pred_mel = self.mel(batch["gen_audio"])
        batch["mel_loss"] = self.criterion["mel_loss"](pred_mel.squeeze(1), batch["spectrogram"])
        batch["generator_loss"] = (self.criterion["generator_loss"](batch["mpd_outputs"])
                                   + self.criterion["generator_loss"](batch["msd_outputs"]))
        batch["feature_matching_loss_mpd"] = self.criterion["feature_matching_loss"](batch["mpd_real_feature_maps"],
                                                                                     batch["mpd_feature_maps"])
        batch["feature_matching_loss_msd"] = self.criterion["feature_matching_loss"](batch["msd_real_feature_maps"],
                                                                                     batch["msd_feature_maps"])
        batch["total_loss"] = (batch["mel_loss"] * 45
                               + (batch["feature_matching_loss_mpd"]
                                  + batch["feature_matching_loss_msd"]) * 2 + batch["generator_loss"])

        if is_train:
            batch["total_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["generator"].step()
            if self.lr_scheduler is not None:
                self.lr_scheduler["generator"].step()

        metrics.update("total_loss", batch["total_loss"].item())
        metrics.update("feature_matching_loss_msd", batch["feature_matching_loss_msd"].item())
        metrics.update("feature_matching_loss_mpd", batch["feature_matching_loss_mpd"].item())
        metrics.update("generator_loss", batch["generator_loss"].item())
        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("total_discriminator_loss", batch["total_discriminator_loss"].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            gen_audio,
            wave,
            audio_path,
            examples_to_log=10,
            *args,
            **kwargs,
    ):

        if self.writer is None:
            return

        tuples = list(zip(gen_audio, wave, text, audio_path))
        shuffle(tuples)
        rows = {}
        for pred, target, text, audio_path in tuples[:examples_to_log]:


            rows[Path(audio_path).name] = {
                "predicted_audio": self.writer.wandb.Audio(pred.squeeze(0).detach().cpu().numpy(),
                                                        sample_rate=self.config["preprocessing"]["sr"]),
                "target_audio": self.writer.wandb.Audio(target.squeeze(0).detach().cpu().numpy(),
                                                        sample_rate=self.config["preprocessing"]["sr"]),
                "text": text
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio_pred, audio_target):
        pred, target = random.choice(zip(audio_pred.cpu(), audio_target.cpu()))


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        a = self.model.named_parameters()
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
