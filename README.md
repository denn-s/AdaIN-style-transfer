# AdaIN-style-transfer

## setup 

```bash
sudo apt-get install python3-dev
sudo apt-get install python3-venv
python3 -m venv venv
```

_Ubuntu 20.04_ CUDA 10.01

```bash
source venv/bin/activate
pip install --upgrade pip
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
```

_Ubuntu 20.04_ CPU

```bash
source venv/bin/activate
pip install --upgrade pip
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
```

## train

_GTX1060_

```bash
python -m tools.train \
--content-images-dir-path data/datasets/coco/train2017 \
--style-images-dir-path data/datasets/wiki_art/train \
--output-dir-path data/checkpoints \
--encoder-model-file_path data/models/vgg19.pt \
--style-weight 10 \
--batch-size 8 \
--num-workers 4 \
--num-epochs 16 \
--log-n-iter 100 \
--image-n-iter 500 \
--save-n-epochs 1
```

## test

```bash
python -m tools.test \
--content-image-file-path  data/images/content/cat.jpg \
--style-image-file-path  data/images/style/great_wave.jpg \
--output-file-path data/output/cat_great_wave.jpg \
--model-file-path data/models/adain.pt
```
