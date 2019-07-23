# CGRN: Character Generation and Recognition Network

# Introduction

This is the code of our paper 'Boosting scene character recognition by learning canonical forms of glyphs' accepted by IJDAR-ICDAR Journal Track. The paper can be found here http://arxiv.org/abs/1907.05577.
# Datasets and pretrained VGG model

Download the datasets and pretrained VGG model used in our experiments in this link.
After download these files, 
Put 'vgg16_weights.npz' under 'pretrained_vgg'.
Put 'IIIT5k' and 'ICDAR03' under 'data'.

# Train
To start training run the following command:

```sh
python train.py --experiment_dir=./experiments/IIIT5k --experiment_id=0  --batch_size=128   
                --lr=0.0001  --epoch=200 --schedule=5  --L1_penalty=100 --Lcont_penalty=100 
                --image_size=64 --fontclass_num=4 --charclass_num=62 
                --resume=0 --use_bn=1  --checkpoint_steps=192 --gpu_id=0
```

# Test on the fly
During the training, you can run a separate program to repeatedly evaluates the produced checkpoints.
```sh
python test.py --test_obj=./experiments/IIIT5k/data/test.obj 
               --model_dir=./experiments/IIIT5k/checkpoint/experiment_0_batch_128  
               --fontclass_num=4 --batch_size=256 --charclass_num=62 --use_stn=0 --use_bn=1 --gpu_id=9
```

# Prepare your own data
To train CGRN on your own data, you need to papare the images as the following format:
![image sample](training_sample.png)
Name the image as  and run:
```sh
python package.py --dir=image_directories
                  --save_dir=binary_save_directory
                  --split_ratio=[0,1]
```
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