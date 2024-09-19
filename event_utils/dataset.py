import glob
import json
import os
import torch
from torch.utils.data import Dataset
from event_utils.cvtransforms import *


# https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py#L480
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """

    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    events      表示输入的事件数据，格式为n×4的np数组，n表示事件数量，4表示t-时间，xy-横纵坐标，p-事件强度，由01组成
    num_bins    min(seq_len, frame_nums) * num_bins # 当事件数据过长时，则限制到30帧，当小于30时，则不变
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():
        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)
        # print(events_torch.shape)# torch.Size([25320, 4])，所有事件，4元组为时间，横坐标，纵坐标，强度

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)

        # print(voxel_grid.shape)
        # torch.Size([30, 96, 96])
        # torch.Size([27, 96, 96])
        # torch.Size([21, 96, 96])
        # torch.Size([21, 96, 96])

        voxel_grid = voxel_grid.flatten()
        # print(voxel_grid.shape)# torch.Size([276480])

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        # print(first_stamp, last_stamp)
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        # print(ts)
        # tensor([0.0000e+00, 1.2263e-03, 1.6687e-02,  ..., 2.2929e+01, 2.2980e+01,
        #         2.3000e+01])
        # tensor([ 0.0000,  0.3733,  0.4155,  ..., 25.9858, 25.9967, 26.0000])
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        # print(pols)
        # tensor([0., 0., 0.,  ..., 1., 1., 0.])
        # tensor([0., 0., 0.,  ..., 1., 0., 0.])
        # tensor([1., 1., 1.,  ..., 1., 0., 1.])

        pols[pols == 0] = -1  # polarity should be +1 / -1
        # print(pols)
        # tensor([1., 1., 1.,  ..., 1., 1., 1.])
        # tensor([-1., -1.,  1.,  ..., -1.,  1.,  1.])
        # tensor([-1., -1.,  1.,  ...,  1., -1., -1.])

        tis = torch.floor(ts)
        # print(tis)
        # tensor([ 0.,  0.,  0.,  ..., 26., 26., 27.])
        # tensor([ 0.,  0.,  0.,  ..., 28., 28., 29.])
        # tensor([ 0.,  0.,  0.,  ..., 21., 21., 22.])
        tis_long = tis.long()
        dts = ts - tis
        # print(dts)
        # tensor([0.0000, 0.0054, 0.0063,  ..., 0.9858, 0.9869, 0.0000])
        # tensor([0.0000, 0.1347, 0.1783,  ..., 0.9774, 0.9970, 0.0000])
        # tensor([0.0000, 0.1326, 0.1707,  ..., 0.9924, 0.9946, 0.0000])

        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width + tis_long[
                                  valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width + (
                                          tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    """
    events      表示输入的事件数据，格式为n×4的np数组，n表示事件数量，4表示t-时间，xy-横纵坐标，p-事件强度，由01组成
    frame_nums  表示视频模态截取的帧的数量
    seq_len     表示预设的最终输入序列基本长度，默认为30
    num_bins    表示将最终输入的长度的扩张倍数
    """

    # 当事件数据过长时，则限制到30帧，当小于30时，则不变
    voxel_len = min(seq_len, frame_nums) * num_bins

    # 设置所有的格子
    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))

    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    # if center_pad:
    #     p = (len(voxel_grid_all) - voxel_len) // 2
    #     voxel_grid_all[p:voxel_len + p] = voxel_grid
    # else:
    voxel_grid_all[:voxel_len] = voxel_grid
    word_boundary = torch.zeros(len(voxel_grid_all))
    word_boundary[:voxel_len] = 1.0
    return voxel_grid_all, word_boundary


class DVS_Lip(Dataset):
    def __init__(self, phase, event_args, mode='train'):
        self.labels = sorted(os.listdir(os.path.join(event_args.event_root, phase)))
        self.length = event_args.seq_len
        self.phase = phase
        self.mode = mode
        self.args = event_args
        # self.center_pad = self.args.center_pad
        self.speech_speed_var = self.args.speech_speed_var

        self.file_list = sorted(glob.glob(os.path.join(event_args.event_root, phase, '*', '*.npy')))
        self.file_list = [file.replace('\\', '/') for file in self.file_list]

        with open('/'.join(event_args.event_root.split('/')[:-1]) + '/frame_nums.json', 'r') as f:
            self.frame_nums = json.load(f)

    def __getitem__(self, index):
        # load timestamps

        word = self.file_list[index].split('/')[-2]
        person = self.file_list[index].split('/')[-1][:-4]
        # phase = train or test, word = 哪一个类别的单词, person = 这个单词的第几个样本
        frame_num = self.frame_nums[self.phase][word][int(person)]  # 也可能表示视频模态中，裁切出来的帧的数量

        # load events

        events_input = np.load(self.file_list[index])
        # 加载npz文件，npz文件中存储n个四元组，用txyp索引

        # 取出中心96×96大小区域的事件
        events_input = events_input[np.where(
            (events_input['x'] >= 16) & (events_input['x'] < 112) & (events_input['y'] >= 16) & (
                        events_input['y'] < 112))]
        events_input['x'] -= 16
        events_input['y'] -= 16

        # 提取四元组，组成格式为n×4的矩阵
        t, x, y, p = events_input['t'], events_input['x'], events_input['y'], events_input['p']
        events_input = np.stack([t, x, y, p], axis=-1)
        # print(events_input.shape)# (17674, 4)

        # print(self.args.num_bins)

        event_voxel_high = None
        word_boundary_high = None
        if self.phase == 'train':
            if self.args.speech_speed_var != 0:
                d = int(frame_num * self.speech_speed_var)
                rand_len = random.randint(-d, d)
                event_voxel_low, word_boundary_low = events_to_voxel_all(events_input, int(frame_num + rand_len), self.length, self.args.num_bins[0], 96, 96, device='cpu')  # (30*num_bins[0], 96, 96)

            else:
                event_voxel_low, word_boundary_low = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[0], 96, 96, device='cpu')  # (30*num_bins[0], 96, 96)

        else:
            event_voxel_low, word_boundary_low = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[0], 96, 96, device='cpu')  # (30*num_bins[0], 96, 96)



        #     """
        #     events      表示输入的事件数据，格式为n×4的np数组，n表示事件数量，4表示t-时间，xy-横纵坐标，p-事件强度，由01组成
        #     frame_nums  表示视频模态截取的帧的数量
        #     seq_len     表示预设的最终输入序列基本长度，默认为30
        #     num_bins    表示将最终输入的长度的扩张倍数
        #     """

        # print(event_voxel_high.shape)
        # event_voxel_high = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins[1], 96, 96,
        #                                        device='cpu')  # (30*num_bins[1], 96, 96)
        # print(event_voxel_low.shape)
        # data augmentation
        # if self.mode == 'train':
        #     event_voxel_low, event_voxel_high = RandomCrop(event_voxel_low, event_voxel_high, (88, 88))
        #     event_voxel_low, event_voxel_high = HorizontalFlip(event_voxel_low, event_voxel_high)
        # else:
        #     event_voxel_low, event_voxel_high = CenterCrop(event_voxel_low, event_voxel_high, (88, 88))

        if self.mode == 'train':
            event_voxel_low, event_voxel_high = RandomCrop(event_voxel_low, event_voxel_high, (88, 88))
            event_voxel_low, event_voxel_high = HorizontalFlip(event_voxel_low, event_voxel_high)
        else:
            event_voxel_low, event_voxel_high = CenterCrop(event_voxel_low, event_voxel_high, (88, 88))

        result = {
            'event_low': torch.FloatTensor(event_voxel_low),
            'word_boundary_low': torch.FloatTensor(word_boundary_low),
            'label': self.labels.index(word)
        }
        if event_voxel_high is not None:
            result['event_high'] = torch.FloatTensor(event_voxel_high)
            result['word_boundary_high'] = torch.FloatTensor(word_boundary_high)
        # print(result['event_low'].shape, result['event_high'].shape)

        return result

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from event_utils.utils import *


    def dataset2dataloader(event_dataset, batch_size, num_workers, shuffle=True):
        data_loader = DataLoader(event_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=shuffle,
                                 drop_last=False,
                                 pin_memory=True)
        return data_loader


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, required=False)

    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--n_class', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--num_workers', type=int, required=False, default=2)


    parser.add_argument('--speech_speed_var', type=float, default=0.1)

    parser.add_argument('--num_bins', type=str2list, required=False, default='1+7')  # 1+4 or 1+7
    parser.add_argument('--test', type=str2bool, required=False, default='false')
    parser.add_argument('--log_dir', type=str, required=False, default=None)

    parser.add_argument('--event_root', type=str, default='../../data/DVS-Lip')

    args = parser.parse_args()
    train_dst = DVS_Lip('train', args, mode='train')
    test_dst = DVS_Lip('test', args, mode='test')
    num = train_dst.frame_nums
    lab = train_dst.labels

    # import matplotlib.pyplot as plt
    #
    # for i in lab:
    #     tt = num['train'][i] + num['test'][i]
    #     cls = sorted(list(set(tt)))
    #     cls = {str(k): tt.count(k) for k in cls}
    #     plt.bar(x=[k for k in cls], height=[cls[k] for k in cls])
    #     plt.savefig(r'../event_dataset/img/{}.png'.format(i))
    #     plt.cla()
    #     print(cls)
    #
    #     print(max(num['train'][i]))
    #
    # exit(0)

    train_lod = dataset2dataloader(train_dst, args.batch_size, args.num_workers)
    from tqdm import tqdm

    for i in tqdm(train_lod):
        event_low = i.get('event_low').cuda(non_blocking=True)
        event_high = i.get('event_high').cuda(non_blocking=True)
        print(event_low.shape, event_high.shape)
        # event_high = i.get('event_high').cuda(non_blocking=True)
        # label = i.get('label').cuda(non_blocking=True).long()
        # print(event_low.shape)
        # break
