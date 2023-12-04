import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    """
        Collate and pad fields in dataset items
        """

    result_batch = {"wave": [], "spectrogram": [], "text_encoded": [], "text_encoded_length": [], "text": [],
                    "spectrogram_length": [], "audio_path": []}

    for item in dataset_items:
        result_batch["wave"].append(item["audio"].squeeze(0))
        result_batch["spectrogram"].append(item["spectrogram"].squeeze(0).T)
        result_batch["text_encoded"].append(item["text_encoded"].squeeze(0))
        result_batch["spectrogram_length"].append(item["spectrogram"].shape[2])
        result_batch["text_encoded_length"].append(item["text_encoded"].shape[1])
        result_batch["text"].append(item["text"])
        result_batch["audio_path"].append(item["audio_path"])

    result_batch["wave"] = pad_sequence(result_batch["wave"], batch_first=True).unsqueeze(1)
    result_batch["spectrogram"] = pad_sequence(result_batch["spectrogram"], batch_first=True).mT
    result_batch["text_encoded"] = pad_sequence(result_batch["text_encoded"], batch_first=True)
    result_batch["spectrogram_length"] = torch.tensor(result_batch["spectrogram_length"], dtype=torch.long)
    result_batch["text_encoded_length"] = torch.tensor(result_batch["text_encoded_length"], dtype=torch.long)

    return result_batch
