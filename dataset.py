import os
import torch
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision import transforms
from skimage import io
from scipy.ndimage import zoom
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset,DataLoader

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h,w,3)


class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transformers = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
        self.transformers = transformers
    def __getitem__(self, idx):  #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  # 每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = io.imread(img_item_path)
        if self.transformers is not None:
            image = self.transformers(image)
        else:
            image = torch.from_numpy(image)
        # label = io.imread(label_item_path)
        #label =io.imread(label_item_path)
        label = np.load(label_item_path)
        label = torch.from_numpy(label)
        return image, label
    def __len__(self):
        return len(self.image_path)

colormap = [[0,0,0],[0,255,0],[255, 255, 0], [0, 0, 255], [255, 0, 0],[0, 255, 255],[128, 38, 205],]

classes= ['Nodata', 'suguar', 'rice', 'water', 'construction_land', 'forest', 'other_land']



#===============================================================================

if __name__ == '__main__':
    # transformers = torchvision.transforms.Compose([transforms.ToTensor(),
    #                                                transforms.Normalize(
    #                                                    [0.2842099825040938, 0.25540074293533543, 0.2792087187102787,
    #                                                     0.20640417859048207, 0.625273691422509, 0.4637834163585723,
    #                                                     0.29021676402539104, 0.2292515271910132, 0.4764320576788299,
    #                                                     0.4789521694661287, 0.4931868887867351],
    #                                                    [0.1572125568923422, 0.1525640675529944, 0.1514116953004076,
    #                                                     0.15614406329942251, 0.1626157132894107, 0.15936790867587491,
    #                                                     0.15910756855528982, 0.14939785854792284, 0.18437275994340682,
    #                                                     0.1576116121317993, 0.15955417159935628])])

    # transformers = torchvision.transforms.Compose([transforms.ToTensor(),
    #                                                transforms.Normalize([0.21403107, 0.19938979, 0.24324717,
    #                                                0.1752188, 0.6368866,0.4692611, 0.2878018,
    #                                                0.19668591, 0.4875627, 0.4896653, 0.50746346],
    #                                                [0.10008377, 0.10607897, 0.12078629, 0.124617286,
    #                                                0.15863594, 0.15149331, 0.14104126, 0.12084164,
    #                                                0.17590337, 0.11046983, 0.10874342])])

    '''
        randomcrop_1
    '''
    transformers_train = torchvision.transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([85.196976, 75.59963, 78.39488, 59.30308, 158.22722, 118.04645, 74.99491, 65.13752, 117.4489, 122.09071, 125.267746],
                                                                         [24.819115, 27.262259, 31.153456, 34.482067, 36.620674, 38.301075, 39.15172, 32.24261, 46.38067, 30.155087, 29.005043])])

    img_dir = "dataset/XingB_and_LiuZ/original_data/TrainData/randomcrop_1/img"
    label_dir = "dataset/XingB_and_LiuZ/original_data/TrainData/randomcrop_1/label"

    Xingbin_Train_dataset = MyData(img_dir, label_dir, transformers = transformers_train)
    Xingbin_Train_dataloader = DataLoader(Xingbin_Train_dataset, batch_size=8, shuffle=True, num_workers=6)

    for x,y in Xingbin_Train_dataloader:
        print(x)
        print(x.shape)
        print(y.shape)
        break

# y = voc_label_indices(Xingbin_dataset[0][1], voc_colormap2label())

