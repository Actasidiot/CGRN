# CGRN: Character Generation and Recognition Network

# Introcution

This is the code of our paper 'Boosting scene character recognition by learning canonical forms of glyphs' accepted by IJDAR-ICDAR Journal Track. The paper can be found here http://arxiv.org/abs/1907.05577 .

# Datasets and pretrained VGG model
To download the datasets and pretrained VGG model used in our experiments
# Train
To start training run the following command:

```sh
python train.py --experiment_dir=./experiments/IIIT5k --experiment_id=0  --batch_size=128   
                --lr=0.0001  --epoch=200 --schedule=5  --L1_penalty=100 --Lcont_penalty=100 
                --image_size=64 --fontclass_num=4 --charclass_num=62 
                --resume=0 --use_bn=1  --checkpoint_steps=192 --gpu_id=0
```

# Test on the fly
```sh
python test.py --test_obj=./experiments/IIIT5k/data/test.obj 
               --model_dir=./experiments/IIIT5k/checkpoint/experiment_0_batch_128  
               --fontclass_num=4 --batch_size=256 --charclass_num=62 --use_stn=0 --use_bn=1 --gpu_id=9
```
