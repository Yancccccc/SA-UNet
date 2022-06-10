import matplotlib.pyplot as plt
from prettytable import PrettyTable
import Config as config
import metric
from dataset import *
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from skimage import io
from net import SA_UNet
colormap = [[255,255,255], [255,0,255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [72, 255, 72], [128, 38, 205],[160,82,45]]
class_names = ['Nodata', 'suguar', 'rice', 'water', 'construction_land', 'forest', 'other_land','bare_land']

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h, w, 3)

def mask(img):
    masking = torch.zeros(img.shape[1],img.shape[2])
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if img[1,i,j] != 0:
                masking[i,j] = 1
    return masking

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

transformers = torchvision.transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([55.193066, 51.369373, 62.4114, 45.084743, 162.25713, 119.69682, 73.51226, 50.53122, 124.01215, 125.25725, 129.83824],
                                                                    [25.581112, 27.1253, 30.857647, 31.909128, 40.36898, 38.61341, 36.058147, 30.903032, 44.86116, 28.221159, 27.749435])])
ChangXing_transform_train = torchvision.transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize([11.055669237762903, 11.304801061744158, 11.21854528266499, 34.40419996181623, 42.735463045432795, 38.92051973654884, 51.607378802291365, 34.65644746326074, 48.62429866768939, 50.762532306913364],[6.705441896500721, 8.222044822794054, 9.236548791979985, 18.967491901513185, 2.8954632178406503, 4.119911617422275, 14.51705461948123, 4.8625066740226774, 14.353340030025967, 14.757674350363498])])

transformers_2015 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([39.98135340082576, 38.7783512826099, 49.25017345571914, 43.56729661934147, 142.88135598188185, 110.54692154376903, 69.04691625041669, 45.49868291769581, 119.466705251268, 167.75940195617804, 176.32788100543948],[24.965231234410364, 25.935760307515842, 28.93568655111308, 32.799374033636695, 43.84462333501905, 42.26782425162127, 39.35267893098884, 30.278631742021666, 44.30422049354514, 32.39992776855984, 31.62088668466183])])
transformers_2017 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([79.92107939766025, 70.88585301279241, 74.34279705987088, 55.63881129680133, 146.93338202342548, 111.8844097649664, 73.2173914622164, 62.32098328613063, 129.50170305955766, 168.60956106297513, 172.5310738593263],[35.084373205808674, 35.1590432073348, 37.810388689421934, 40.06051029321953, 45.102994590979776, 44.289808957178295, 42.66778313024135, 37.43969000849209, 47.46019891141157, 35.88760521401966, 35.238889194903514])])
transformers_2021 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([49.112005585580384, 45.457424385993, 53.89288586569934, 42.91280790030649, 159.31102207196685, 109.68321806074319, 68.05436290348688, 45.75308641813284, 120.84085085607944, 127.30756595035798, 126.23945520467775],
                                                                                                [33.79766356353502, 34.213071803759576, 35.901864071386015, 38.397053918805724, 40.51438517299972, 38.87191699113998, 40.25925825910798, 36.28005575909439, 46.73744708303059, 43.096934210129795, 44.27079847112406])])
test_dataset = MyData(config.test_dir_2019, config.test_label_2019, transformers=transformers)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)



config_vit = config.get_CTranS_config()

net = SA_UNet.Unet(11,8)

net.load_state_dict(torch.load("checkponit/U_Net_ASPP_SAM_20_epoch_model.pth"))
net.eval()
acc = 0.0
predimg = []
conf_mat = np.zeros((8, 8)).astype(np.int64)
for step, (img, label) in enumerate(test_dataloader):
    img = img.float()
    #mask = torch.zeros([256,256])
    label = label.long()
    out = net(img)
    out = F.softmax(out, dim=1)
    preds = torch.argmax(out.data, dim=1)
    preds = torch.squeeze(preds)
    # preds = (preds * mask).long()
    acc += torch.sum(preds == label.data)
    predimg.append(preds)
    conf_mat += metric.confusion_matrix(preds.flatten().numpy(),label.flatten().numpy(),8)
test_acc_mat, test_acc_per_class, test_acc_cls, test_IoU, test_mean_IoU, test_kappa = metric.evaluate(conf_mat)
print(conf_mat)
table = PrettyTable(["序号", "名称", "acc", "IoU"])
for i in range(len(class_names)):
    table.add_row([i, class_names[i], test_acc_per_class[i], test_IoU[i]])
print(table)
print("train_acc:", test_acc_mat)
print("train_mean_IoU:", test_mean_IoU)
print("kappa:", test_kappa)

test_acc = acc/len(test_dataloader)/256/256
print(test_acc)

clip_dir = os.path.join("dataset/XingB_and_LiuZ/SA_UNet", "pre")
os.makedirs(clip_dir, exist_ok=True)

for i in range(72):
    plt.figure(figsize=(16, 6))
    pre = label2image(predimg[i], colormap=colormap)
    if i < 10:
        test_pre_name = "000{}.png".format(i)
        clip_image_path = os.path.join(clip_dir, test_pre_name)
        io.imsave(clip_image_path, pre)
    if i >= 10 and i < 100:
        test_pre_name = "00{}.png".format(i)
        clip_image_path = os.path.join(clip_dir, test_pre_name)
        io.imsave(clip_image_path,pre)
    if i>=100:
        test_pre_name = "0{}.png".format(i)
        clip_image_path = os.path.join(clip_dir, test_pre_name)
        io.imsave(clip_image_path,pre)




















