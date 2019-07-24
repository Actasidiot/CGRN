# CGRN: Character Generation and Recognition Network

# Introduction

This is the code of our paper 'Boosting scene character recognition by learning canonical forms of glyphs' accepted by IJDAR-ICDAR Journal Track. The paper can be found here http://arxiv.org/abs/1907.05577.

# Datasets and pretrained VGG model

Download the datasets and pretrained VGG model used in our experiments in this link：
https://drive.google.com/open?id=1lVgYBIS48poHGprbYNwdXNwTjyApCcfi

After downloading these files, put 'vgg16_weights.npz' under 'pretrained_vgg', and put 'IIIT5k' and 'ICDAR03' folders under 'data'.

# Requirement
- Python 2.7
- Tensorflow >= 1.0.1

# Train
To start training, run the following command:

```sh
python train.py --experiment_dir=./experiments/IIIT5k --experiment_id=0  --batch_size=128   
                --lr=0.0001  --epoch=200 --schedule=5  --L1_penalty=100 --Lcont_penalty=100 
                --image_size=64 --fontclass_num=4 --charclass_num=62 
                --resume=0 --use_bn=1  --checkpoint_steps=192 --gpu_id=0
```

# Test on the fly
During the training, you can run a separate program to repeatedly evaluates the produced checkpoints:
```sh
python test.py --test_obj=./experiments/IIIT5k/data/test.obj 
               --model_dir=./experiments/IIIT5k/checkpoint/experiment_0_batch_128  
               --fontclass_num=4 --batch_size=256 --charclass_num=62 --use_stn=0 --use_bn=1 --gpu_id=9
```

# Train with your own data
To train CGRN with your own data, you need to papare the images as the following format:
![image sample](training_sample.png)

where a scene character image is concatenated with glyph images of different fonts along the width direction.

Name the images as 'fontclasses_charclass_imgname' (e.g., 0123_0_BadImag-img037-00009.png), where '0123' denotes that the font classes of these glyph images are 0,1,2,3, '0' denotes that the char class of this image is 0 (i.e., a).

Putting all your images into a directory, and run:
```sh
python package.py --dir=image_directory
                  --save_dir=binary_save_directory
                  --split_ratio=[0,1]
```
to pickle the images and their corresponding labels into binary format.

# Some details about the implementation
- Recently we find that the training process becomes more stable if we randomly pick glyph image of one font to generate in each step.
So we finally adopt this strategy instead of generating glyph images of all fonts in a row, which was introduced in our paper.
- During each step, we first optimize the discriminator once, then optimize the geneator twice, which is helpful for the convergence of the whole model.
- The learning rate markedly affects the recognition accuracy. Empirically, we find 0.0001 and 0.0002 are the best learning rates for IIIT5k and ICDAR03, respectively.

# Citation

If you find this project helpful for your research, please cite the following paper:
```
@article{wang2019boosting,
  title={Boosting scene character recognition by learning canonical forms of glyphs},
  author={Wang, Yizhi and Lian, Zhouhui and Tang, Yingmin and Xiao, Jianguo},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  pages={1--11},
  year={2019},
  publisher={Springer}
}
```
# Acknowledgements
Code derived from:

* [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) by [yenchenlin](https://github.com/yenchenlin)
* [Domain Transfer Network](https://github.com/yunjey/domain-transfer-network) by [yunjey](https://github.com/yunjey)
* [ac-gan](https://github.com/buriburisuri/ac-gan) by [buriburisuri](https://github.com/buriburisuri)
* [dc-gan](https://github.com/carpedm20/DCGAN-tensorflow) by [carpedm20](https://github.com/carpedm20)
* [origianl pix2pix torch code](https://github.com/phillipi/pix2pix) by [phillipi](https://github.com/phillipi)
* [zi2zi](https://github.com/kaonashi-tyc/zi2zi/) by [kaonashi-tyc](https://github.com/kaonashi-tyc)