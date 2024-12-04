import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def normalize_point_clouds(pcs):
    B,N,C = pcs.shape
    shift = torch.mean(pcs,dim=1).unsqueeze(1)
    scale = torch.std(pcs.view(B,N*C),dim=1).unsqueeze(1).unsqueeze(1)
    newpcs = (pcs - shift) / scale
    return newpcs,shift,scale


def sin(data):
    device = data.device
    b,n,c = data.shape
    alpha1 = 0.5 + 0.5 * torch.rand([1,])
    beta1 = 0.5 + 0.5 * torch.rand([1,])
    alpha2 = 0.5 + 0.5 * torch.rand([1,])
    beta2 = 0.5 + 0.5 * torch.rand([1,])
    alpha3 = 0.5 + 0.5 * torch.rand([1,])
    beta3 = 0.5 + 0.5 * torch.rand([1,])

    if torch.rand([1,])>0.5:
        k = torch.tensor(1)
    else:
        k = torch.tensor(-1)
    k = k.to(device)
    move = torch.zeros_like(data)
    newdata = torch.zeros_like(data)

    key = torch.rand([1,])
    if key < 0.5:
        move[:, :, 2] = k * alpha2.to(device) * torch.sin(beta2.to(device) * data[:, :, 2])
        newdata[:, :, 1] = data[:, :, 1]
        newdata[:, :, 2] = data[:, :, 2] + move[:, :, 2]
        newdata[:, :, 0] = data[:, :, 0]
    elif key < 0.7:
        move[:, :, 0] = k * alpha3.to(device) * torch.sin(beta3.to(device) * data[:, :, 0])
        newdata[:, :, 1] = data[:, :, 1]
        newdata[:, :, 2] = data[:, :, 2]
        newdata[:, :, 0] = data[:, :, 0] + move[:, :, 0]
    elif key < 0.9:
        move[:, :, 1] = k * alpha1.to(device) * torch.sin(beta1.to(device) * data[:, :, 1])
        newdata[:, :, 1] = data[:, :, 1] + move[:, :, 1]
        newdata[:, :, 2] = data[:, :, 2]
        newdata[:, :, 0] = data[:, :, 0]
    elif key > 0.9:
        move[:, :, 1] = k * alpha1.to(device) * torch.sin(beta1.to(device) * data[:, :, 1])
        move[:, :, 2] = k * alpha2.to(device) * torch.sin(beta2.to(device) * data[:, :, 2])
        move[:, :, 0] = k * alpha3.to(device) * torch.sin(beta3.to(device) * data[:, :, 0])
        newdata[:, :, 1] = data[:, :, 1] + move[:, :, 1]
        newdata[:, :, 2] = data[:, :, 2] + move[:, :, 2]
        newdata[:, :, 0] = data[:, :, 0] + move[:, :, 0]
    return newdata


def index_mask(mask, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = mask.device
    B = mask.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    mask[batch_indices, idx, :] = 0
    return 1-mask


def topk_point(k, data, sindata):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    mask = torch.ones_like(data)
    dist_vec = sindata - data
    dist = torch.sum(dist_vec ** 2, dim=2)
    _, idx = torch.topk(dist, int(k), dim=-1, largest=False, sorted=False)
    mask = index_mask(mask, idx)
    return mask


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, semseg = False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d() if not semseg else STNkd(k=9)
        self.conv1 = torch.nn.Conv1d(3, 64, 1) if not semseg else torch.nn.Conv1d(9, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, pointfeat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=True):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):

        x, trans, trans_feat, pointfeat = self.feat(x)
        global_feat = x
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dp1(x)
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dp2(x)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        return x


class PointNetDenseCls(nn.Module):
    def __init__(self, cat_num=16,part_num=50):
        super(PointNetDenseCls, self).__init__()
        self.cat_num = cat_num
        self.part_num = part_num
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        # classification network
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, cat_num)
        self.dropout = nn.Dropout(p=0.3)
        self.bnc1 = nn.BatchNorm1d(256)
        self.bnc2 = nn.BatchNorm1d(256)
        # segmentation network
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.dp1 = nn.Dropout(p=0.2)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.dp2 = nn.Dropout(p=0.2)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud,label):
        batchsize,_ , n_pts = point_cloud.size()
        # point_cloud_transformed
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        point_cloud_transformed = torch.bmm(point_cloud, trans)
        point_cloud_transformed = point_cloud_transformed.transpose(2, 1)
        # MLP
        out1 = F.relu(self.bn1(self.conv1(point_cloud_transformed)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        # net_transformed
        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)
        # MLP
        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = F.relu(self.bn5(self.conv5(out4)))

        #out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)
        # classification network
        net = F.relu(self.bnc1(self.fc1(out_max)))
        net = F.relu(self.bnc2(self.dropout(self.fc2(net))))
        net = self.fc3(net) # [B,16]
        # segmentation network
        out_max = torch.cat([out_max,label],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, n_pts)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net2 = F.relu(self.bns1(self.convs1(concat)))
        net2 = self.dp1(net2)
        net2 = F.relu(self.bns2(self.convs2(net2)))
        net2 = self.dp2(net2)
        net2 = F.relu(self.bns3(self.convs3(net2)))
        net2 = self.convs4(net2)
        net2 = net2.transpose(2, 1).contiguous()
        net2 = F.log_softmax(net2.view(-1, self.part_num), dim=-1)
        net2 = net2.view(batchsize, n_pts, self.part_num) # [B, N 50]

        return net, net2, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNetLoss(torch.nn.Module):
    def __init__(self, weight=1,mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.weight = weight

    def forward(self, labels_pred, label, seg_pred,seg, trans_feat):
        seg_loss = F.nll_loss(seg_pred, seg)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        label_loss = F.nll_loss(labels_pred, label)

        loss = self.weight * seg_loss + (1-self.weight) * label_loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss, seg_loss, label_loss


class PointNetSeg(nn.Module):
    def __init__(self,num_class,feature_transform=True, semseg = True):
        super(PointNetSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False,feature_transform=feature_transform, semseg = semseg)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat


def pointNet0(num_classes=40, **kwargs) -> PointNetCls:
    return PointNetCls(k=num_classes)

if __name__ == '__main__':
    point = torch.randn(8,3,1024)
    label = torch.randn(8,16)
    model = PointNetDenseCls()
    net, net2, trans_feat = model(point,label)
    print('net',net.shape)
    print('net2',net2.shape)
    print('trans_feat',trans_feat.shape)