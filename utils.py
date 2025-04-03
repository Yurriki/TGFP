import math
import logging
import numpy as np
import torch
from scipy.optimize import curve_fit

## Network parameters initialization
# 在神经网络模型中找到所有的二维卷积层，使用 Xavier 均匀分布初始化它们的权重，并将偏置项初始化为常数值 0.1。
# 这样可以在训练模型之前，对模型的权重进行合适的初始化，有助于更好地训练和收敛模型。
def weights_init(m): 
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.1)

# 初始化日志记录器，将日志消息同时记录到文件和控制台中。
def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='a',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

# 五次样条插值是一种插值方法,用于在已知数据点之间拟合一个连续且光滑的函数曲线
## Quintic spline definition.
def quintic_spline(x, z, a, b, c, d, e):
    return z + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5


## Fitting the trajectory of one planning circle by quintic spline, with the current location fixed.
# bounds 参数指定了参数估计的边界。对于这个拟合过程，限制了五次项的系数为负无穷到正无穷，常数项为 y[0] 到 y[0]+1e-6。
# 函数使用 curve_fit 函数进行拟合，将自变量 x、因变量 y 以及拟合函数 quintic_spline 作为参数传递给 curve_fit。
def fitting_traj_by_qs(x, y):
    param, loss = curve_fit(quintic_spline, x, y,
        bounds=([y[0], -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [y[0]+1e-6, np.inf, np.inf, np.inf, np.inf, np.inf]))
    return param


## Custom activation for output layer (Graves, 2015)
# outputActivation 是函数的名称，它接受一个参数 x，表示网络的输出。
# 如果 displacement 为 True，则进行位移计算，将 x 中的第一个维度（行）进行累加，用于计算位移。即将原始的 mu 值转换为位移值。
# 接下来，将 x 划分为不同的部分，分别表示均值 muX 和 muY，标准差 sigX 和 sigY，以及相关性系数 rho。
# 对 sigX 和 sigY 进行指数运算，以获得正值，表示标准差的倒数（1/sigX）。
# 对 rho 进行双曲正切函数（tanh）的运算，使其值在 -1 到 1 之间。
# 最后，将 muX、muY、sigX、sigY 和 rho 沿着第二个维度进行拼接，得到输出的激活值 out。
# 通过调用 outputActivation 函数，并传入网络输出 x，可以对输出进行处理和转换，得到表示概率分布的激活值。这个函数通常在神经网络的输出层之后使用，用于处理和解释模型的预测结果。
def outputActivation(x, displacement=True):
    if displacement:
        # Then mu value denotes displacement.
        x[:, :, 0:2] = torch.stack([torch.sum(x[0:i, :, 0:2], dim=0) for i in range(1, x.shape[0] + 1)], 0)
    # Each output has 5 params to describe the gaussian distribution.
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)  # This positive value represents Reciprocal of SIGMA (1/sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)   # -1 < rho < 1
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out

# 通过调用 maskedNLL 函数，并传入预测值 y_pred、真实值 y_gt 和掩码 mask，
# 可以计算带有掩码的负对数似然损失。这个函数通常在训练阶段用于优化模型，根据预测结果与真实值之间的差异来计算损失，并进行梯度反向传播。
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho *
        torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) \
        + torch.log(torch.tensor(2 * math.pi))
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal

# 如果 use_maneuvers 为 True，则通过循环遍历 lat_pred 和 lon_pred 中的组合情况来计算损失。首先，计算权重 wts，
# 然后从 fut_pred 中提取相应的预测值 y_pred 和真实值 y_gt。接下来，按照负对数似然损失的公式计算损失值 out。将计算得到的损失值加上权重的对数，
# 并将结果保存在 acc 的相应通道中。

# 对于 use_maneuvers 为 False 的情况，直接从 fut_pred 中提取预测值 y_pred 和真实值 y_gt，然后按照负对数似然损失的公式计算损失值 out。
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                  num_lat_classes=3, num_lon_classes=2,
                  use_maneuvers=True, avg_along_time=False, separately=False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:, l] * lon_pred[:, k]
                wts = wts.repeat(len(fut_pred[0]), 1)
                y_pred = fut_pred[k * num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]
                out = -(0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
                      - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
                      + torch.log(torch.tensor(2 * math.pi)))
                acc[:, :, count] = out + torch.log(wts)
                count += 1
        acc = -logsumexp(acc, dim=2)  # Negative log-likelihood
        acc = acc * op_mask[:, :, 0]
        if avg_along_time:
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc, dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = (0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
              - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr)
              + torch.log(torch.tensor(2 * math.pi)))
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc[:, :, 0]
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts





def maskedNLLTestnointention(fut_pred, fut, op_mask,avg_along_time=False, separately=False):
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
              - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + torch.log(torch.tensor(2 * math.pi))
        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            if separately:
                lossVal = acc[:, :, 0]
                counts = op_mask[:, :, 0]
                return lossVal, counts
            else:
                lossVal = torch.sum(acc[:, :, 0], dim=1)
                counts = torch.sum(op_mask[:, :, 0], dim=1)
                return lossVal, counts
















def maskedDestMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask[-1, :, :])
    X = y_pred[:,0]
    Y = y_pred[:,1]
    x = y_gt[:, 0]
    y = y_gt[:, 1]
    out = torch.pow(x-X, 2) + torch.pow(y-Y, 2)
    acc[:,0] = out
    acc[:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal


# 计算损失值 lossVal，通过对 acc 所有元素求和，并除以掩码 mask 中为1的元素数量的总和。
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc) / torch.sum(mask)
    return lossVal

# 如果 separately 参数为 True，则返回 acc 的第一个通道和 mask 的第一个通道。
# 否则，计算损失值 lossVal 为 acc 第一个通道在第一个维度上的求和，并计算计数 counts 为 mask 第一个通道在第一个维度上的求和。
def maskedMSETest(y_pred, y_gt, mask, separately=False):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    if separately:
        return acc[:, :, 0], mask[:, :, 0]
    else:
        lossVal = torch.sum(acc[:, :, 0], dim=1)
        counts = torch.sum(mask[:, :, 0], dim=1)
        return lossVal, counts


def maskedTest(y_pred, y_gt, mask):
    
    acc     = torch.zeros_like(mask)
    acc_r   = torch.zeros_like(mask)
    
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    
    out     = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    out_r = torch.pow(out, 0.5)
    
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    
    acc_r[:, :, 0] = out_r
    acc_r[:, :, 1] = out_r
    acc_r = acc_r * mask    
    
    lossVal     = acc[:,:,0]
    lossVal_r   = acc_r[:,:,0]
    counts      = mask[:,:,0]
        
    return lossVal,lossVal_r,counts

def KL(mean, log_var):
    lossval=torch.mean(0.5 * torch.sum(mean ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)
    # lossval=-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return lossval

## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    # Get the maximal probability value from 6 full path
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    # here (inputs - s) is to compare the relative probability with the most probable behavior.
    # and then sum up all candidate behaviors.
    # s->logP(Y | m_max,X), inputs->logP(m_i,Y | X), (inputs - s)->logP(m_i | X)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs