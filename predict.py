import argparse
import datetime
import errno
import multiprocessing
import os
import sys
import threading
import socket
from glob import glob
import pandas as pd
import torchvision.models as models
from torch import nn
from torch.utils import data
from torchvision import transforms
from args import args
import torch
from PIL import Image
from torch.utils.data import Dataset

import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()

    # model path
    parser.add_argument('--light_model_path', default='./model/light.pth', type=str)
    parser.add_argument('--pink_model_path', default='./model/pink.pth', type=str)
    parser.add_argument('--simplate_model_path', default='./model/simplate.pth', type=str)
    parser.add_argument('--result_csv', default='./result.csv')
    parser.add_argument('--image_size', type=int, default=80)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    return args


def Socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        with open('./PortSetting.txt', 'r') as f:
            reader = f.read()

        ip = reader.split(',')[0]
        port = reader.split(',')[-1]

        print("Server address is: %s:%s" % (str(ip), str(port)))

        s.bind((ip, int(port)))

        s.listen(10)

    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print('Waiting connect')

    while 1:
        # waiting connect and accept
        conn, addr = s.accept()

        # accept data
        t = threading.Thread(target=predict, args=(conn, addr))
        t.start()


def make_model(args):
    # 加载预训练模型
    model = models.__dict__[args.arch](pretrained=True, progress=True)

    # 最后一层全连接层
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, args.num_classes)
    )
    return model


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w * ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))
        img = img.resize(self.size, self.interpolation)
        return img


class Transforms:
    def __init__(self):
        pass

    def get_test_transform(self, mean, std, size):
        return transforms.Compose([
            Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_transforms(self, test_size=224, backbone=None):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        transformations = self.get_test_transform(mean, std, test_size)
        return transformations


class load_Dataset(Dataset):
    def __init__(self, dataset_path, test_img_ids, transform=None):

        self.img_ids = test_img_ids
        self.transform = transform
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_name = self.img_ids[index]
        img_path = os.path.join(self.dataset_path, img_name)
        # index += 1
        try:
            img = Image.open(img_path)
        except:
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, img_name


def load_data(test_image_path):
    image_path = test_image_path
    tr = Transforms()
    transformations = tr.get_transforms(test_size=args.image_size)
    img_ids = glob(os.path.join(image_path, '*'))
    test_img_ids = [os.path.basename(p) for p in img_ids]
    test_set = load_Dataset(image_path, test_img_ids, transform=transformations)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return test_loader


def predict(conn, addr):
    global N
    try:
        print('Client address is: {0}'.format(addr))
        recv_order = conn.recv(1024)
        order = recv_order.decode()
        check_action = '$'
        check_action_result = check_action in order

        if not check_action_result:
            conn.send('input order not unconformity, please again input'.encode('utf-8'))
            print('waiting again input order...')
            recv_order = conn.recv(1024)
            print('order is :', recv_order.decode())

        else:
            light_model_path, pink_model_path, simplate_model_path = \
                args.light_model_path, args.pink_model_path, args.simplate_model_path
            print('action build model...')
            # load model
            light_model, pink_model, simplate_model = make_model(args), make_model(args), make_model(args)

            light_model.load_state_dict(torch.load(light_model_path))
            pink_model.load_state_dict(torch.load(pink_model_path))
            simplate_model.load_state_dict(torch.load(simplate_model_path))

            # send order sure connect success
            conn.send(recv_order)
            print('model build finish...')

            while 1:
                # again recv order
                recv_order = conn.recv(1024)
                print('recv order is:', recv_order.decode('utf-8'))

                check = ','
                check_ = '$'
                order = recv_order.decode('utf-8')
                order = order.replace('\\', '/')

                check_result = check in order
                check_result_ = check_ in order
                if check_result_:
                    break
                if not check_result:
                    conn.send('input order not unconformity, please again input'.encode('utf-8'))
                    print('waiting again input order...')
                    recv_order = conn.recv(1024)
                    print('order is :', recv_order.decode())

                N = order.split(',')[0]
                order_order = order.split(',')[1]
                origin_image_path = str(order_order).replace('\\', '/')
                model_type = 'light'

                # check origin_image_path whether or not exist
                check_dir = os.path.exists(origin_image_path)
                if not check_dir:
                    res = N + ',' + origin_image_path + ' is not exist'
                    print(res)
                    conn.send(res.encode('utf-8'))

                start_time = datetime.datetime.now()

                # 加载数据
                test_loader = load_data(origin_image_path)

                image_names = []
                y_pred = []
                with torch.no_grad():
                    # 设置成eval模式
                    light_model.eval()
                    pink_model.eval()
                    simplate_model.eval()
                    for (inputs, image_name) in test_loader:
                        image_names.extend(list(image_name))
                        inputs = torch.autograd.Variable(inputs)
                        if model_type == 'light':
                            outputs = light_model(inputs)
                        elif model_type == 'simplate':
                            outputs = simplate_model(inputs)
                        else:
                            outputs = pink_model(inputs)
                        probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
                        y_pred.extend(probability)

                    res_dict = {
                        'img_path': image_names,
                        'predict': y_pred,

                    }
                    df = pd.DataFrame(res_dict)
                    df.to_csv(args.result_csv, index=False)
                    print(f"write to {args.result_csv} succeed ")
                print('return order...')
                res = N + ',' + origin_image_path
                print(res)
                conn.send(res.encode('utf-8'))
                end_time = datetime.datetime.now()
                print('total_time:', end_time - start_time)


    except OSError as err:
        if err.errno == errno.ENOENT:
            print('*' * 30)
            print('not find dir')
            conn.send((N + ',' + 'Not find dir').encode('utf-8'))

        elif err.errno == errno.EACCES:
            print('*' * 30)
            print('Permission denied')
            conn.send((N + ',' + 'Permission denied').encode('utf-8'))
            print('*' * 30)

        else:
            conn.close()
            print('*' * 30)
            print('except connect interrupt')
            print('Waiting next process connect...')
            conn.close()
            print('*' * 30)

    finally:
        print('This process is over, Now can start next process...')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    Socket_service()
