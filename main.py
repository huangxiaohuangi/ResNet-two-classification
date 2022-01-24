import datetime
from args import args
import torch
import torch.nn as nn
import data_gen
from tqdm import tqdm
import torch.utils.data as data
from transform import get_transforms
import os
from build_net import make_model
from utils import get_optimizer, AverageMeter, save_checkpoint, accuracy, checkfiles, clear_outputs
import torchnet.meter as meter
import pandas as pd
from sklearn import metrics
import shutil
import warnings

warnings.filterwarnings('ignore')

# Use CUDA
torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
best_acc = 0


def main():
    global best_acc

    os.makedirs(args.checkpoint, exist_ok=True)

    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    train_set = data_gen.Dataset(dataset_path, root=train_txt_path, transform=transformations['val_train'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    val_set = data_gen.ValDataset(dataset_path, root=val_txt_path, transform=transformations['val_test'])
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # model
    model = make_model(args)
    if use_cuda:
        model.cuda()

    # define loss function and optimizer
    if use_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, args)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-5)

    # load checkpoint
    start_epoch = args.start_epoch
    # if args.resume:
    #     print("===> Resuming from checkpoint")
    #     assert os.path.isfile(args.resume),'Error: no checkpoint directory found'
    #     args.checkpoint = os.path.dirname(args.resume)  # 去掉文件名 返回目录
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     start_epoch = checkpoint['epoch']
    #     model.module.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # train
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, val_acc = val(val_loader, model, criterion, epoch, use_cuda)

        scheduler.step(test_loss)

        print(f'train_loss:{train_loss}\t val_loss:{test_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')

        # save_model
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            'fold': 0,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, single=True, checkpoint=args.checkpoint)

    print("best acc = ", best_acc)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()
    for (inputs, targets) in tqdm(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 梯度参数设为0
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data)
        # inputs.size(0)=32
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))

    return losses.avg, train_acc.avg


def val(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()  # 将模型设置为验证模式
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter(args.num_classes)
    for _, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        confusion_matrix.add(outputs.data.squeeze(), targets.long())
        acc1 = accuracy(outputs.data, targets.data)

        # compute accuracy by confusion matrix
        # cm_value = confusion_matrix.value()
        # acc2 = 0
        # for i in range(args.num_classes):
        #     acc2 += 100. * cm_value[i][i]/(cm_value.sum())

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        val_acc.update(acc1.item(), inputs.size(0))
    return losses.avg, val_acc.avg


def test(use_cuda):
    # data
    start_time = datetime.datetime.now()
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    test_set = data_gen.TestDataset(dataset_path, root=test_txt_path, transform=transformations['test'])
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # load model
    model = make_model(args)

    if args.model_path:
        # 加载模型
        model.load_state_dict(torch.load(args.model_path))

    if use_cuda:
        model.cuda()

    # evaluate
    y_pred = []
    y_true = []
    img_paths = []
    # temp = []
    with torch.no_grad():
        model.eval()  # 设置成eval模式
        for (inputs, targets, paths) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            img_paths.extend(list(paths))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)  # (16,2)
            # dim=1 表示按行计算 即对每一行进行softmax
            # probability = torch.nn.functional.softmax(outputs,dim=1)[:,1].tolist()
            # probability = [1 if prob >= 0.5 else 0 for prob in probability]
            # 返回最大值的索引
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            # if probability.ndim == 0:
            #     temp.append(probability)
            #     continue
            y_pred.extend(probability)

        print("y_pred=", y_pred)
        # y_pred.extend(temp)
    os.makedirs('./outputs/0/', exist_ok=True)
    os.makedirs('./outputs/1/', exist_ok=True)
    error_count = 0
    for i in range(len(img_paths)):
        if y_pred[i] == 0:
            if y_true[i] == 1:
                error_count += 1
            old_path = os.path.join(dataset_path, img_paths[i])
            new_path = os.path.join('./outputs/0/', img_paths[i])
            shutil.copy(old_path, new_path)
        else:
            if y_true[i] == 0:
                error_count += 1
            old_path = os.path.join(dataset_path, img_paths[i])
            new_path = os.path.join('./outputs/1/', img_paths[i])
            shutil.copy(old_path, new_path)
    print('error_count:', error_count)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("accuracy=", accuracy)
    # print('error_count', error_count)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("confusion_matrix=", confusion_matrix)
    print(metrics.classification_report(y_true, y_pred))
    print("roc-auc score=", metrics.roc_auc_score(y_true, y_pred))

    res_dict = {
        'img_path': img_paths,
        'label': y_true,
        'predict': y_pred,

    }
    df = pd.DataFrame(res_dict)
    if os.access(args.result_csv, os.W_OK):
        df.to_csv(args.result_csv, index=False)
    print(f"write to {args.result_csv} succeed ")
    end_time = datetime.datetime.now()
    print('use_time:', end_time - start_time)


if __name__ == "__main__":
    # main()
    # 划分数据集write
    dataset_name = 'pink'
    dataset_txt_path = './dataset/' + dataset_name + '.txt'
    train_txt_path = './dataset/' + dataset_name + '_train.txt'
    test_txt_path = './dataset/' + dataset_name + '_test.txt'
    val_txt_path = './dataset/' + dataset_name + '_val.txt'
    dataset_path = 'G:/water/code/' + dataset_name
    flag = checkfiles(dataset_name)
    clear_outputs()
    if flag is False:
        data_gen.Split_datatset(dataset_txt_path, train_txt_path, test_txt_path)
        data_gen.Split_datatset(train_txt_path, train_txt_path, val_txt_path)
    if args.mode == 'train':
        main()
    else:
        test(use_cuda)
