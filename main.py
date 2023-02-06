import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from resnet_m import resnet12
import Generator as model_g
import data as loader
import numpy as np
import random
import scipy.stats as stats
from utils import  AverageMeter, mkdir_p


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='./data/miniImageNet/',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int,
                    metavar='E', help='evaluate model on test set')
parser.add_argument('--N-way', default=5, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--N-shot', default=1, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--N-query', default=15, type=int,
                    metavar='NQUERY', help='N_query (default: 15)')
parser.add_argument('--gpu', default='0')
parser.add_argument('--pretrain_path', default='', type=str, metavar='pretrain_path',
                    help='path to latest checkpoint (default: none)')


SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def main():
    global args
    args = parser.parse_args()
    set_gpu(args.gpu)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
     # 加载backbone
    model_E = resnet12(keep_prob=1.0, avg_pool=True, num_classes=64).cuda()
    pretrain_checkpoint = torch.load("./pretrain/mini_distilled.pth")
    model_E.load_state_dict(pretrain_checkpoint['model'])

    model_G = model_g.GeneratorNet().cuda()
    
   
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([{"params":model_G.parameters()}], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    # Data loading code
    mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
    train_aug_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_train=True)

    
    val_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((84, 84)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_val=True)
    
    test_dataset = loader.ImageLoader(
        args.data,
        transforms.Compose([
            transforms.Resize((84, 84)),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ]), is_test=True)
    
    

    train_loader = torch.utils.data.DataLoader(
        train_aug_dataset, batch_size=args.N_way*(args.N_query+args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler = loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query+args.N_shot, n_class=64))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.N_way* (args.N_query + args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query + args.N_shot, n_class=16))
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.N_way* (args.N_query + args.N_shot), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=loader.GeneratorSampler(num_of_class=args.N_way, num_per_class=args.N_query + args.N_shot, n_class=20))
        
    
    if args.test:
        if args.resume:
            print('==> Resuming from generator checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(args.resume)
            model_G.load_state_dict(checkpoint['state_dict_G'])
            model_E.load_state_dict(checkpoint['state_dict_E'])
            _, test_acc, h = test(test_loader, model_E, model_G, criterion)
            print(test_acc, h)
        return 0

    train_acc, train_loss, val_acc = 0, 0, 0
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0:
            for p in optimizer.param_groups:
                p['lr'] /= 1.11111
                p['lr'] = max(1e-6, p['lr'])
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        print('phase: meta_train...')
        train_loss, train_acc = train(train_loader, model_E, model_G, criterion, optimizer, epoch)
        _, val_acc, _ = validate(val_loader, model_E, model_G, criterion)
        print('current epoch test acc: {:}'.format(val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint({
            'epoch': epoch,
            'state_dict_E': model_E.state_dict(),
            'state_dict_G': model_G.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, epoch, checkpoint=args.checkpoint)
    print('best_acc:',best_acc)
    print('best_epoch:',best_epoch)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,h


def train(train_loader, model_E, model_G, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()  
    model_E.train()
    model_G.train()
    for inter in range(1500):
        input, target = train_loader.__iter__().next()
        input = input.cuda()
        support_input = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 84, 84)
        query_input   = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 84, 84)
        support_input, _ = model_E(support_input) # (way,shot,64,10,10)
        query_input, _ = model_E(query_input)  # (way,query,64,10,10)
        _, fc, fw, fh = support_input.size()
        support_input = support_input.view(args.N_way, args.N_shot, fc, fw, fh)
        support_input = torch.mean(support_input, [1,3,4]) # (way,feature)
        query_input = query_input.view(args.N_way, args.N_query, fc, fw, fh)
        query_input = torch.mean(query_input, [3,4]) # (way,query,feature)
        predict = model_G(support_input, query_input)
        gt = np.tile(range(args.N_way), args.N_query)
        gt.sort()
        gt = torch.cuda.LongTensor(gt)
        acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
        loss = criterion(predict, gt) 
        losses.update(loss.item(), predict.size(0))
        top1.update(acc.item(), predict.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (inter+1)%200==0:
            print('meta_train:', inter+1, 'loss:', losses.avg, 'acc:', top1.avg)
    return (losses.avg, top1.avg)

def validate(val_loader, model_E, model_G, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    model_E.eval()
    model_G.eval()
    with torch.no_grad():
        accuracies = []
        for inter in range(600):
            input, target = val_loader.__iter__().next()
            input = input.cuda()
            support_input = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 84, 84)
            query_input   = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 84, 84)
            support_input, _ = model_E(support_input) # (way,shot,64,10,10)
            query_input, _ = model_E(query_input)  # (way,query,64,10,10)
            _, fc, fw, fh = support_input.size()
            support_input = support_input.view(args.N_way, args.N_shot, fc, fw, fh)
            support_input = torch.mean(support_input, [1,3,4]) # (way,feature)
            query_input = query_input.view(args.N_way, args.N_query, fc, fw, fh)
            query_input = torch.mean(query_input, [3,4]) # (way,query,feature)
            predict = model_G(support_input, query_input)
            gt = np.tile(range(args.N_way), args.N_query)
            gt.sort()
            gt = torch.cuda.LongTensor(gt)
            acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
            accuracies.append(acc.item())
            loss = criterion(predict, gt)
            losses.update(loss.item(), predict.size(0))
            top1.update(acc.item(), predict.size(0))
            if (inter+1)%100==0:
                print('test:', inter+1, 'loss:', losses.avg, 'acc:', top1.avg)
    mean, h = mean_confidence_interval(accuracies)
    return (losses.avg, top1.avg, h)

def test(val_loader, model_E, model_G, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    model_E.eval()
    model_G.eval()
    with torch.no_grad():
        accuracies = []
        for inter in range(10000):
            input, target = val_loader.__iter__().next()
            input = input.cuda()
            support_input = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 3, 84, 84)
            query_input   = input.view(args.N_way, args.N_query + args.N_shot, 3, 84, 84)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 3, 84, 84)
            support_input, _ = model_E(support_input) # (way,shot,64,10,10)
            query_input, _ = model_E(query_input)  # (way,query,64,10,10)
            _, fc, fw, fh = support_input.size()
            support_input = support_input.view(args.N_way, args.N_shot, fc, fw, fh)
            support_input = torch.mean(support_input, [1,3,4]) # (way,feature)
            query_input = query_input.view(args.N_way, args.N_query, fc, fw, fh)
            query_input = torch.mean(query_input, [3,4]) # (way,query,feature)
            predict = model_G(support_input, query_input)
            gt = np.tile(range(args.N_way), args.N_query)
            gt.sort()
            gt = torch.cuda.LongTensor(gt)
            acc = (predict.topk(1)[1].view(-1)==gt).float().sum(0)/gt.shape[0]*100.
            accuracies.append(acc.item())
            loss = criterion(predict, gt)
            losses.update(loss.item(), predict.size(0))
            top1.update(acc.item(), predict.size(0))
            if (inter+1)%100==0:
                print('test:', inter+1, 'loss:', losses.avg, 'acc:', top1.avg)
    mean, h = mean_confidence_interval(accuracies)
    return (losses.avg, top1.avg, h)

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    torch.save(state, filepath)
    print('save checkpoint success', epoch)

if __name__ == '__main__':
    main()
