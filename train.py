import os
from time import time
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import nn
import Config as config
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from skimage import io
from net import SA_UNet




use_gpu = torch.cuda.is_available()

'''
clip2
'''
transformers_train = torchvision.transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([55.193066, 51.369373, 62.4114, 45.084743, 162.25713, 119.69682, 73.51226, 50.53122, 124.01215, 125.25725, 129.83824],
                                                                    [25.581112, 27.1253, 30.857647, 31.909128, 40.36898, 38.61341, 36.058147, 30.903032, 44.86116, 28.221159, 27.749435])])
transformers_2015 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([39.98135340082576, 38.7783512826099, 49.25017345571914, 43.56729661934147, 142.88135598188185, 110.54692154376903, 69.04691625041669, 45.49868291769581, 119.466705251268, 167.75940195617804, 176.32788100543948],[24.965231234410364, 25.935760307515842, 28.93568655111308, 32.799374033636695, 43.84462333501905, 42.26782425162127, 39.35267893098884, 30.278631742021666, 44.30422049354514, 32.39992776855984, 31.62088668466183])])
transformers_2017 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([79.92107939766025, 70.88585301279241, 74.34279705987088, 55.63881129680133, 146.93338202342548, 111.8844097649664, 73.2173914622164, 62.32098328613063, 129.50170305955766, 168.60956106297513, 172.5310738593263],[35.084373205808674, 35.1590432073348, 37.810388689421934, 40.06051029321953, 45.102994590979776, 44.289808957178295, 42.66778313024135, 37.43969000849209, 47.46019891141157, 35.88760521401966, 35.238889194903514])])
transformers_2021 = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize([49.112005585580384, 45.457424385993, 53.89288586569934, 42.91280790030649, 159.31102207196685, 109.68321806074319, 68.05436290348688, 45.75308641813284, 120.84085085607944, 127.30756595035798, 126.23945520467775],
                                                                                                [33.79766356353502, 34.213071803759576, 35.901864071386015, 38.397053918805724, 40.51438517299972, 38.87191699113998, 40.25925825910798, 36.28005575909439, 46.73744708303059, 43.096934210129795, 44.27079847112406])])
'''
randomcrop_1
'''
# transformers_train = torchvision.transforms.Compose([transforms.ToTensor(),
#                                                    transforms.Normalize([85.196976, 75.59963, 78.39488, 59.30308, 158.22722, 118.04645, 74.99491, 65.13752, 117.4489, 122.09071, 125.267746],
#                                                                          [24.819115, 27.262259, 31.153456, 34.482067, 36.620674, 38.301075, 39.15172, 32.24261, 46.38067, 30.155087, 29.005043])])
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
        label = np.load(label_item_path)
        #label = io.imread(label_item_path)
        label = torch.from_numpy(label)
        return image,label
    def __len__(self):
        return len(self.image_path)

#==================================================================================

def train_model(model,criterion,optimizer,scheduler,batch_size,num_epochs=25):
    since = time.time()
    writer = SummaryWriter(r"./log")
    for epoch in range(num_epochs):
        b = 0
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-' * 10)
        for phase in ['img','val']:
            running_loss = 0.0
            running_corrects = 0.0
            train_acc = 0.0
            train_loss = 0.0
            if phase == 'img':
                model.train()
                for step,(inputs, labels) in enumerate(dataloaders[phase]):
                    b = epoch*len(dataloaders[phase]) + step
                    if use_gpu:
                        inputs = inputs.float().cuda()
                        labels = labels.long().cuda()
                    inputs, labels = Variable(inputs), Variable(labels)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    out = F.softmax(outputs,dim=1)
                    preds = torch.argmax(out.data,dim=1)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=0.05,norm_type=2.0)
                    optimizer.step()
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    train_loss += loss.item()
                    train_acc += torch.sum(preds == labels.data)/(256*256)/batch_size
                    if step % 50 == 49:  # 迭代次数除以300的余数等于299，,每300轮输出一次   0，1，2，3，....299，.....
                        writer.add_scalar('loss/img', train_loss / 50, b)
                        writer.add_scalar('acc/img',train_acc / 50, b)
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, train_loss / 50))
                        train_loss = 0.0
                        print('[%d, %5d] acc: %.3f' % (epoch + 1, step + 1, train_acc / 50))
                        train_acc = 0.0
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float(running_corrects) / dataset_sizes[phase]/256/256
                print('{}Train Loss: {:.4f}Train Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('loss/epoch_train', epoch_loss, epoch + 1)
                writer.add_scalar('acc/epoch_train', epoch_acc, epoch + 1)
                if epoch > num_epochs-10:
                    #torch.save(model.state_dict(), r'.checpoint/UNet/U_Net_ASPP_SAM_class6_The_%d_epoch_model.pth' % (epoch+1))
                    torch.save(model.state_dict(), r'./checpoint/SA_UNet_%d_epoch_model.pth' % (epoch + 1))
                # 1. 记录这个epoch的loss值和准确率
                # info = {'loss': epoch_loss, 'accuracy': epoch_acc}
                # for tag, value in info.items():
                #     train_logger.scalar_summary(tag, value, epoch)
                #
                # # 2. 记录这个epoch的模型的参数和梯度
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     train_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                #     train_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

                # 3. 记录最后一个epoch的图像
                # info = {'images': inputs.cpu().numpy()}
                # for tag, images in info.items():
                #     train_logger.image_summary(tag, images, epoch)
            else:
                #取消验证阶段的梯度
                with torch.no_grad():
                    model.eval()
                    val_loss = 0.0
                    val_acc = 0.0
                    b = 0
                    for step, (inputs, labels) in enumerate(dataloaders[phase]):
                        b = epoch*len(dataloaders[phase])+step
                        # 获取输入
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.float().cuda()
                            labels = labels.long().cuda()
                        inputs, labels = Variable(inputs), Variable(labels)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        out = F.softmax(outputs, dim=1)
                        preds = torch.argmax(out.data, dim=1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        val_loss += loss.item()
                        val_acc += torch.sum(preds == labels.data) / (256 * 256)/batch_size

                        if step % 50 == 49:  # 迭代次数除以300的余数等于299，,每300轮输出一次   0，1，2，3，....299，.....
                            writer.add_scalar('loss/val', val_loss/50, b)
                            writer.add_scalar('acc/val', val_acc/50, b)
                            print('[%d, %5d] val_loss: %.3f' % (epoch + 1, step + 1, val_loss / 50))
                            val_loss = 0.0
                            print('[%d, %5d] acc: %.3f' % (epoch + 1, step + 1, val_acc / 50))
                            val_acc = 0.0
                    scheduler.step()
                    lr_exp = scheduler.get_last_lr()[0]
                    epoch_val_loss = running_loss / dataset_sizes[phase]
                    epoch_val_acc = float(running_corrects) / dataset_sizes[phase]/256/256
                    print('{}Val Loss: {:.4f}Val Acc: {:.4f}'.format(phase, epoch_val_loss, epoch_val_acc))
                    writer.add_scalar('loss/epoch_val', epoch_val_loss, epoch + 1)
                    writer.add_scalar('acc/epoch_val', epoch_val_acc, epoch + 1)
                    writer.add_scalar('lr', lr_exp, epoch + 1)
                    #info = {'loss': epoch_loss, 'accuracy': epoch_acc}

                    # for tag, value in info.items():
                    #     val_logger.scalar_summary(tag, value, epoch)
                    #
                    # for tag, value in model.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     val_logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                    #     val_logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
    writer.close()
                    # info = {'images': inputs.cpu().numpy()}
                    # for tag, images in info.items():
                    #     val_logger.image_summary(tag, images, epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    train_Data = MyData(config.train_dir, config.train_lable_dir, transformers=transformers_train)
    val_Data = MyData(config.vali_dir, config.vali_lable_dir, transformers=transformers_train)
    train_DataLoader = DataLoader(train_Data, batch_size=8, shuffle=True, drop_last=True)
    val_DataLoader = DataLoader(val_Data, batch_size=8, shuffle=True, drop_last=True)

    dataloaders = {'img': train_DataLoader, 'val': val_DataLoader}
    dataset_sizes = {'img': train_Data.__len__(), 'val': val_Data.__len__()}

    config_vit = config.get_CTranS_config()
    net = SA_UNet.Unet(11,8)
    net = net.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=None, reduction="mean")
    criterion = criterion.cuda()
    optimzer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
    #optimzer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimzer, step_size=5, gamma=0.1)
    train_model(net, criterion, optimzer, num_epochs=20, scheduler=exp_lr_scheduler, batch_size=8)

    '''
        tensorboard --logdir=log --port=6006

    '''

