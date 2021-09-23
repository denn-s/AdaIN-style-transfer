# AdaIN-style-transfer

## Setup 

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

## Train

_new training process_

```bash
python -m tools.train \
--content-images-dir-path ~/Data/coco/train2017 \
--style-images-dir-path ~/Data/wiki_art/train \
--output-dir-path data/checkpoints \
--encoder-model-file-path data/models/vgg19.pt \
--style-weight 1.0 \
--batch-size 4 \
--num-workers 4 \
--num-epochs 16 \
--learning-rate 1e-4 \
--lr-scheduler-gamma 0.9 \
--log-n-iter 100 \
--image-n-iter 1000 \
--save-n-epochs 1
```

_training process with pretrained decoder_

```bash
python -m tools.train \
--content-images-dir-path ~/Data/coco/train2017 \
--style-images-dir-path ~/Data/wiki_art/train \
--output-dir-path data/checkpoints \
--encoder-model-file-path data/models/vgg19.pt \
--decoder-model-file-path data/checkpoints/train_2021.06.12_19-14-41/epoch_15_decoder.pt \
--style-weight 1.0 \
--batch-size 4 \
--num-workers 4 \
--num-epochs 16 \
--learning-rate 1e-4 \
--lr-scheduler-gamma 0.9 \
--log-n-iter 100 \
--image-n-iter 1000 \
--save-n-epochs 1
```

## Test

_arbitrary image pair_

```bash
python -m tools.test \
--content-image-file-path  data/images/content/chicago_cropped.jpg \
--style-image-file-path  data/images/style/ashville_cropped.jpg \
--output-image-file-path data/images/output/chicago_style_ashville.jpg \
--encoder-model-file-path data/models/vgg19.pt \
--decoder-model-file-path data/checkpoints/train_2021.06.12_19-14-41/epoch_23_decoder.pt
```

_image pairs like those used by the paper authors_

```bash
python -m tools.test_all \
--content-image-dir-path  data/images/content \
--style-image-dir-path  data/images/style \
--output-image-dir-path data/images/output \
--encoder-model-file-path data/models/vgg19.pt \
--decoder-model-file-path data/checkpoints/train_2021.06.20_06-19-14/epoch_23_decoder.pt
```

## References

1. [Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." Proceedings of the IEEE International Conference on Computer Vision. 2017.](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
2. [Official Torch based implementation](https://github.com/xunhuang1995/AdaIN-style)
