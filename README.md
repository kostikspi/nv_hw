# Neural Vocoder

## Installation guide

Report

Download model checkpoint from Google Drive

```shell
pip install -r ./requirements.txt
```

```shell
curl -c ./temp.txt -s -L "https://drive.google.com/uc?export=download&id=1SjERyAnBEDFx07MwPivgrb-G4dmK3-oF" > /dev/null
curl -Lb ./temp.txt "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie.txt`&id=1bEzhZ8WbzgIzZYkjTgDdKolXghgrCy3K" -o "model_checkpoint/checkpoint-epoch45.pth"
```
Google Drive Link:
https://drive.google.com/file/d/1SjERyAnBEDFx07MwPivgrb-G4dmK3-oF/view?usp=sharing

## Reproduce

To reproduce results train 50 epoches with train_gan_kaggle.json
## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
