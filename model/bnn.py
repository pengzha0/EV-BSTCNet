import torch
import torch.nn as nn
import torch.nn.functional as F


class Sign(nn.Module):

    def __init__(self, bound=1.2, binary=True):
        super(Sign, self).__init__()
        self.bound = bound
        self.binary = binary

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary

    def forward(self, x):
        if not self.binary:
            return x
        if not self.training:
            return torch.sign(x)

        #print('activation before sign: ', x, flush=True)
        out = torch.clamp(x, -self.bound, self.bound)
        out_forward = torch.sign(x)

        y = out_forward.detach() + out - out.detach()
        return y

class BNNCONV3D(nn.Module):
    """
        convolution with binary weights and binary activations
    """
    def __init__(self, in_chn, out_chn, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False,
            groups=1, dilation=1, bound=1.2, binary=True, R=None, sign_layer=None):
        super(BNNCONV3D, self).__init__()
        self.in_channels = in_chn
        self.out_channels = out_chn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.bound = bound
        self.binary = binary
        self.R = R

        self.sign_layer = RSign(in_chn)

        self.shape = (out_chn, in_chn // groups, kernel_size[0], kernel_size[1],kernel_size[2])
        self.weights = nn.Parameter(torch.Tensor(*self.shape))
        torch.nn.init.xavier_normal_(self.weights, gain=2.0)

        # self.o_scale = nn.Parameter(torch.ones(1, self.out_channels, 1, 1, 1), requires_grad=True)
        # self.h_scale = nn.Parameter(torch.ones(1, 1, 2000, 1, 1), requires_grad=True)
        # self.w_scale = nn.Parameter(torch.ones(1, 1, 1, 2000, 1), requires_grad=True)
        # self.z_scale = nn.Parameter(torch.ones(1, 1, 1, 1, 2000), requires_grad=True)

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary
        self.sign_layer.reset_state(bound, binary)

    def forward(self, x):

        # x = self.sign_layer(x)



        # clipped_weights = torch.clamp(self.weights, -self.bound, self.bound)

        # binary_weights_no_grad = torch.sign(self.weights).detach()

        # binary_weights = binary_weights_no_grad + \
        #         clipped_weights - clipped_weights.detach()

        # out = F.conv3d(x, binary_weights, stride=self.stride, 
        #         padding=self.padding, groups=self.groups, dilation=self.dilation)


        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=4,keepdim=True),dim=3,keepdim=True),dim=2,keepdim=True)
        # scaling_factor = torch.mean(torch.mean(torch.mean(torch.mean(abs(real_weights),dim=4,keepdim=True),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)

        # print(real_weights.shape)
        # print(scaling_factor.shape)

        # assert 0==1

        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        out = F.conv3d(x, binary_weights, stride=self.stride, 
                padding=self.padding, groups=self.groups, dilation=self.dilation)
        
        # o_scale = self.o_scale
        # h_scale = self.h_scale[:, :, :out.shape[2], :,:]
        # w_scale = self.w_scale[:, :, :, :out.shape[3],:]
        # z_scale = self.z_scale[:, :, :,:, :out.shape[4]]

        return out


"""
    RSign and RPReLU (ECCV 2020, ReActNet: Towards Precise Binary 
    Neural Network with Generalized Activation Functions)
"""
class RSign(nn.Module):

    def __init__(self, chn, bound=1.2, binary=True):
        super(RSign, self).__init__()
        self.bound = bound
        self.binary = binary

        self.beta = nn.Parameter(torch.zeros(1, chn, 1, 1, 1))

    def reset_state(self, bound, binary):
        self.bound = bound
        self.binary = binary
        
    def forward(self, x):
        if not self.binary:
            return x

        x = x + self.beta.expand_as(x)
        if not self.training:
            return torch.sign(x)

        #print('activation before sign: ', x, flush=True)
        out = torch.clamp(x, -self.bound, self.bound)
        out_forward = torch.sign(x)

        y = out_forward.detach() + out - out.detach()
        return y

class RPReLU(nn.Module):
    """
        ReAct PReLU function.
    """
    def __init__(self, chn, init=0.25):
        super(RPReLU, self).__init__()
        self.init = init
        
        self.gamma = nn.Parameter(torch.ones(1, chn, 1, 1,1) * init)
        self.beta1 = nn.Parameter(torch.zeros(1, chn, 1, 1,1))
        self.beta2 = nn.Parameter(torch.zeros(1, chn, 1, 1,1))

    def forward(self, x):
        x = x + self.beta1.expand_as(x)
        x = torch.where(x > 0, x + self.beta2.expand_as(x), x * self.gamma + self.beta2.expand_as(x))

        return x

class RPReLU1D(nn.Module):
    """
        ReAct PReLU function.
    """
    def __init__(self, chn, init=0.25):
        super(RPReLU1D, self).__init__()
        self.init = init
        self.gamma = nn.Parameter(torch.ones(1, chn,  1) * init)
        self.beta1 = nn.Parameter(torch.zeros(1, chn, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, chn, 1))

    def forward(self, x):

        x = x + self.beta1.expand_as(x)
        x = torch.where(x > 0, x + self.beta2.expand_as(x), x * self.gamma + self.beta2.expand_as(x))

        return x

class BNNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(BNNConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size
        self.shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        # binary_input_no_grad = torch.sign(x)
        # cliped_input = torch.clamp(x, -1.0, 1.0)
        # x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight.view(self.shape)
        scaling_factor = torch.mean(torch.mean(abs(real_weights),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        # real_weights = self.weight.view(self.shape)
        # binary_weights_no_grad = torch.sign(real_weights)
        # cliped_weights = torch.clamp(real_weights, -1.2, 1.2)
        # binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding,dilation=self.dilation)

        return y 