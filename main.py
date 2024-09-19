import time

import numpy as np
from event_utils.utils import *
from event_utils.dataset import DVS_Lip
from event_utils.mixup import mixup_data, mixup_criterion
from event_utils.label_smooth import LSR


import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from warnings import filterwarnings
filterwarnings('ignore')

def test(args, net):
    with torch.no_grad():
        dataset = DVS_Lip('test', args, mode='test')
        # logger.info(f'Start Testing, Data Length: {len(dataset)}')
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
        # logger.info('start testing')
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}
        net.eval()
        for data in tqdm(loader):

            data = {k: data[k].cuda(non_blocking=True) for k in data}

            label = data.get('label').long()

            with autocast():
                logit = net(data)

            v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())
            label_list = label.cpu().numpy().tolist()
            pred_list = logit.argmax(-1).cpu().numpy().tolist()
            for i in range(len(label_list)):
                label_pred[label_list[i]].append(pred_list[i])

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'test acc: {:.5f}, acc part1: {:.5f}, acc part2: {:.5f}'.format(acc, acc_p1, acc_p2)
        return acc, acc_p1, acc_p2, msg

def write_parameters_to_file(model, file_handle):
    total_params = 0
    for name, parameter in model.named_parameters():
        num_params = parameter.numel()
        total_params += num_params
        file_handle.write(f"{name}: {num_params}\n")
    file_handle.write(f"Total number of parameters: {total_params}\n")


def test_train(args, net):
    with torch.no_grad():
        dataset = DVS_Lip('train', args, mode='test')
        # logger.info(f'Start Testing, Data Length: {len(dataset)}')
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

        # logger.info('start testing')
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}
        net.eval()
        for data in tqdm(loader):

            data = {k: data[k].cuda(non_blocking=True) for k in data}
            label = data.get('label').long()
            data = {
                'event_low': data.get('event_low'),
                'event_high': data.get('event_high')
            }
            with autocast():
                logit = net(data)

            v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())

            label_list = label.cpu().numpy().tolist()
            pred_list = logit.argmax(-1).cpu().numpy().tolist()
            for i in range(len(label_list)):
                label_pred[label_list[i]].append(pred_list[i])

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'test_train acc: {:.5f}, acc part1: {:.5f}, acc part2: {:.5f}'.format(acc, acc_p1, acc_p2)
        return acc, acc_p1, acc_p2, msg
def train(args, net, optimizer, log_dir, scheduler):
    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'each_acc': [],
        'finished': False
    }
    dataset = DVS_Lip('train', args, mode='train')
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

    best_acc, best_acc_p1, best_acc_p2 = 0.0, 0.0, 0.0


    scaler = GradScaler()


    # import time
    for epoch in range(args.max_epoch):
        time.sleep(0.3)
        print('epoch: {}'.format(epoch))
        time.sleep(0.3)

        if args.label_smooth:
            criterion = LSR()
        else:
            criterion = nn.CrossEntropyLoss()

        net.train()
        i_iter = -1

        k = time.time()

        for data in tqdm(loader):
            i_iter += 1
            t = time.time()
            data = {k: data[k].cuda(non_blocking=True) for k in data}

            label = data.get('label').long()

            data_low = {
                'event_low': data.get('event_low'),
                'word_boundary_low': data.get('word_boundary_low')
            }

            data_low, labels_a, labels_b, lam = mixup_data(x=data_low, y=label, alpha=args.mixup, use_cuda=True)

            input_data = {
                'event_low': data_low['event_low'],
                'word_boundary_low': data_low['word_boundary_low']
            }

            data_high = {
                'event_high': data.get('event_high'),
                'word_boundary_high': data.get('word_boundary_high')
            }

            if data_high['event_high'] is not None:
                if args.ifmixup:
                    data_high, labels_a, labels_b, lam = mixup_data(x=data_high, y=label, alpha=args.mixup, lam=lam, use_cuda=True)
                input_data['event_high'] = data_high['event_high']
                input_data['word_boundary_high'] = data_high['word_boundary_high']
            # print(i_iter, 'data time', time.time()-t)
            # print(data['event_low'].shape, data['event_high'].shape)


            # print(event_low.shape, event_high.shape)
            # torch.Size([32, 30, 1, 88, 88]) torch.Size([32, 120, 1, 88, 88])
            t_inf = time.time()
            loss = {}
            with autocast():
                logit = net(input_data)
                loss_func = mixup_criterion(labels_a, labels_b, lam)
                loss_bp = loss_func(criterion, logit)


            loss['Total'] = loss_bp
            optimizer.zero_grad()
            scaler.scale(loss_bp).backward()  
            scaler.step(optimizer)
            scaler.update()
            # print(i_iter, 'inf_time', time.time() - t_inf)
            #
            # print(i_iter, 'each time:', time.time() - k)
            k = time.time()

            # writer.add_scalar('lr', float(showLR(optimizer)), tot_iter)
            # writer.add_scalar('loss', loss_bp.item(), tot_iter)


        epoch_acc = {
            'epoch': epoch
        }

        acc, acc_p1, acc_p2, msg = test(args, net)
        epoch_acc['test_test_acc'] = '{:.5f}'.format(acc)
        epoch_acc['test_test_acc1'] = '{:.5f}'.format(acc_p1)
        epoch_acc['test_test_acc2'] = '{:.5f}'.format(acc_p2)
        print(msg)

        torch.save(net.module.state_dict(),f'{log_dir}/model_last.pth')
        if acc > best_acc:
            best_acc, best_acc_p1, best_acc_p2, best_epoch = acc, acc_p1, acc_p2, epoch
            save_name = log_dir + '/model_best.pth'
            temp = os.path.split(save_name)[0]
            if not os.path.exists(temp):
                os.makedirs(temp)
            torch.save(net.module.state_dict(), save_name)
            train_res['best_acc'] = '{:.5f}'.format(best_acc)
            train_res['best_epoch'] = epoch


        print('acc best: {}\n'.format(best_acc))
        time.sleep(0.3)
        train_res['each_acc'].append(epoch_acc)
        scheduler.step()

    train_res['finished'] = 1

    f = open(log_dir + '/train_log.json', 'w+')
    f.write(str(train_res).replace('\'', '\"'))
    f.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str,default='0', required=False)
    parser.add_argument('--lr', type=float, default=3e-4) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--n_class', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--num_workers', type=int, required=False, default=1)
    parser.add_argument('--max_epoch', type=int, required=False, default=80)
    parser.add_argument('--num_bins', type=str2list, required=False, default='2+1')  # 1+4 or 1+7
    parser.add_argument('--log_dir', type=str, required=False, default='/home/vgc/users/zhaopeng/EV-BSTCNet/result')
    parser.add_argument('--exp_name', type=str, default='log')


    parser.add_argument('--event_root', type=str, default='/home/vgc/users/zhaopeng/event-based-lip-reading/event_lip/data/DVS-Lip')

    parser.add_argument('--speech_speed_var', type=float, default=0)
    parser.add_argument('--word_boundary', type=bool, default=False)
    parser.add_argument('--mixup', type=float, default=0.4)
    parser.add_argument('--label_smooth', type=bool, default=True)

    parser.add_argument('--back_type', type=str, default='TCN')
    parser.add_argument('--se', type=str2bool, default=False)
    parser.add_argument('--base_channel', type=int, default=64)

    parser.add_argument('--ifbinary', type=bool, default=True)
    parser.add_argument('--num_layers', default=4)
    parser.add_argument('--evaluate', type=str, default=None)

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    s = ''
    for k, v in args.__dict__.items():
        s += '\t' + k + '\t' + str(v) + '\n'
    print(s)
    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_dir = os.path.join(args.log_dir,  timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f = open(log_dir + '/settings.txt', 'w+')
    f.write(s)

    if args.ifbinary:
        from model.EV_BSTCNet import EV_BSTCNet
    else:
        from model.model_v8.model import MSTP

    net = EV_BSTCNet(args).cuda()

    print(net, file=f)

    write_parameters_to_file(net, f)
    
    f.close()

    total_params = 0
    for param_name, param in net.named_parameters():
        param_params = torch.prod(torch.tensor(param.size())).item()
        total_params += param_params
        print(f"Layer: {param_name} | Parameters: {param_params}")
    print(f"Total Parameters: {total_params}")

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'AdamW':# 2% lower
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=5e-6)
    
    if args.evaluate != None:
        pretrain_dict = torch.load(args.evaluate, map_location='cpu')
        net.load_state_dict(pretrain_dict,strict=True)
        acc, acc_p1, acc_p2, msg = test(args,net.cuda())

        print(msg)
        return 0

    net = nn.DataParallel(net)

    train(args, net, optimizer, log_dir, scheduler)


if __name__ == '__main__':
    main()
