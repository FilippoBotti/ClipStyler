# CLIPMamba
### Environment

```
$ conda create -n CLIPstyler
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
$ pip install causal_conv1d
$ pip install mamba_ssm
$ pip install timm
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Fast Style Transfer

```
python train_fast.py --content_dir $DIV2K_DIR$ \
--name exp1 \
--text "Sketch with black pencil" --test_dir ./test_set
```
