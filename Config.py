import os
import torch
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

# cosineLR = True # whether use cosineLR or not
# n_channels = 3
# n_labels = 1
# epochs = 200
img_size = 256
# print_frequency = 1
# save_frequency = 5000
# vis_frequency = 10
# early_stopping_patience = 50

# pretrain = False
# task_name = 'MoNuSeg' # GlaS MoNuSeg
# # task_name = 'GlaS'
learning_rate = 0.01
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "dataset/XingB_and_LiuZ/class7/train/clip1/img"
train_lable_dir = "dataset/XingB_and_LiuZ/class7/train/clip1/label"
vali_dir = "dataset/XingB_and_LiuZ/class7/val/clip1/img"
vali_lable_dir = "dataset/XingB_and_LiuZ/class7/val/clip1/label"
# test_dir = "dataset/XingB_and_LiuZ/class7/test/clip1/img"
# test_label_dir = "dataset/XingB_and_LiuZ/class7/test/clip1/label"
test_dir = "dataset/XingB_and_LiuZ/class7/test/clip_0000/img"
test_label_dir = "dataset/XingB_and_LiuZ/class7/test/clip_0000/label"

"""
    2015
"""
train_dir_2015 = "dataset/XingB_and_LiuZ/2015/train/clip/img"
train_label_2015 = "dataset/XingB_and_LiuZ/2015/train/clip/label"
val_dir_2015 = "dataset/XingB_and_LiuZ/2015/val/clip/img"
val_label_2015 = "dataset/XingB_and_LiuZ/2015/val/clip/label"
test_dir_2015 = "dataset/XingB_and_LiuZ/2015/test/clip/img"
test_label_2015 = "dataset/XingB_and_LiuZ/2015/test/clip/label"

train_dir_2017 = "dataset/XingB_and_LiuZ/2017/train/clip/img"
train_label_2017 = "dataset/XingB_and_LiuZ/2017/train/clip/label"
val_dir_2017 = "dataset/XingB_and_LiuZ/2017/val/clip/img"
val_label_2017 = "dataset/XingB_and_LiuZ/2017/val/clip/label"
test_dir_2017 = "dataset/XingB_and_LiuZ/2017/test/clip/img"
test_label_2017 = "dataset/XingB_and_LiuZ/2017/test/clip/label"

train_dir_2019 = "dataset/XingB_and_LiuZ/2019/train/clip/img"
train_label_2019 = "dataset/XingB_and_LiuZ/2019/train/clip/label"
val_dir_2019 = "dataset/XingB_and_LiuZ/2019/val/clip/img"
val_label_2019 = "dataset/XingB_and_LiuZ/2019/val/clip/label"
test_dir_2019 = "dataset/XingB_and_LiuZ/2019/test/clip/img"
test_label_2019 = "dataset/XingB_and_LiuZ/2019/test/clip/label"

train_dir_2021 = "dataset/XingB_and_LiuZ/2021/train/clip/img"
train_label_2021 = "dataset/XingB_and_LiuZ/2021/train/clip/label"
val_dir_2021 = "dataset/XingB_and_LiuZ/2021/val/clip/img"
val_label_2021 = "dataset/XingB_and_LiuZ/2021/val/clip/label"
test_dir_2021 = "dataset/XingB_and_LiuZ/2021/test/clip/img"
test_label_2021 = "dataset/XingB_and_LiuZ/2021/test/clip/label"

test_2015 = "dataset/XingB_and_LiuZ/2015/test/clip/img"
test_2017 = "dataset/XingB_and_LiuZ/2017/clip"
test_2019 = "dataset/XingB_and_LiuZ/2019"
test_2021 = "dataset/XingB_and_LiuZ/2021/clip"

checpoint = "checpoint"
##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 24+48+64+160#960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [8, 4, 2, 1]
    #config.base_channel = 64 # base channel of U-Net
    config.n_classes = 8
    return config




# used in testing phase, copy the session name in training phase
# test_session = "Test_session_07.03_20h39"