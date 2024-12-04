import torch
import numpy as np

class SMT:
    def __init__(self, args):
        self.rand_center_num = args.rand_center_num
        self.alphamin = args.alphamin
        self.alphamax = args.alphamax
        self.sample = args.sample
        self.isCat = args.isCat
        self.shuffle = args.shuffle
        self.f = args.f

    def DA(self, data):
        """
        Args:
            data (B,N,3)
            label (B)
        """
        device = data.device
        B, N, C = data.shape
        if self.sample == "random":
            # random gen N * 3
            # idxs = torch.randperm(N)
            # idxs = idxs[0:self.rand_center_num]
            idxs = self.generate_random_permutations_batch(B,N,self.rand_center_num)
        if self.sample == "FPS":
            # FPS gen    N * 3
            idxs = self.farthest_point_sample(data, self.rand_center_num)
        dist = torch.zeros_like(data).to(device)
        for i in range(self.rand_center_num):
            center = self.index_points(data, idxs[:,i]).unsqueeze(1)
            dist = dist + data - center
        dist = dist / self.rand_center_num
        alpha = self.alphamin + (self.alphamax - self.alphamin) * torch.rand([1, 1, C])
        beta = self.alphamin + (self.alphamax - self.alphamin) * torch.rand([1, 1, C])
        f = torch.tensor(self.f)
        k = 2 * torch.rand([1, 1, C]) - 1
        move = k.to(device) * alpha.to(device) * torch.sin(f.to(device) * beta.to(device) * dist)
        newdata = data + move
        return newdata

    def DA0(self, data):
        device = data.device
        B, N, C = data.shape
        newdata = torch.zeros_like(data)
        alpha = self.alphamin + (self.alphamax - self.alphamin) * torch.rand([1, 1, C])
        beta = self.alphamin + (self.alphamax - self.alphamin) * torch.rand([1, 1, C])
        f = torch.tensor(self.f)
        move = alpha.to(device) * torch.sin(f.to(device) * beta.to(device) * data)
        newdata = data + move
        return newdata

    def sin_transform(self, data, label=[]):
        """
        Args:
            data (B,N,3)
            label (B)
        """
        B, _, _ = data.shape
        newdata, shift, scale = self.normalize_point_clouds(data)
        if self.rand_center_num == 0:
            newdata = self.DA0(newdata)
        else:
            newdata = self.DA(newdata)
        newdata = newdata * scale + shift
        label = label.unsqueeze(1)
        if self.isCat:
            newdata = torch.cat([data, newdata],dim=0)
            label = torch.cat([label, label],dim=0)
            if self.shuffle:
                idxs = torch.randperm(B*2)
                newdata = newdata[idxs,:,:]
                label = label[idxs,:]
        return newdata, label.squeeze(1)

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = self.square_distance(xyz, centroid).squeeze(2)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def generate_random_permutations_batch(self, B, N, top_k):
        # 生成B个长度为N的随机排列的批次
        all_permutations = torch.stack([torch.randperm(N) for _ in range(B)])
        # 选择每个随机排列的前top_k个位置
        selected_positions = all_permutations[:, :top_k]
        return selected_positions

    def square_distance(self,src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def topk_point(self, k, data, sindata):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        mask = torch.ones_like(data)
        dist_vec = sindata - newdata
        dist = torch.sum(dist_vec ** 2, dim=2)
        _, idx = torch.topk(dist, int(k), dim=-1, largest=False, sorted=False)
        mask = index_mask(mask, idx)
        return mask

    def normalize_point_clouds(self, pcs):
        B, N, C = pcs.shape
        shift = torch.mean(pcs, dim=1).unsqueeze(1)
        scale = torch.std(pcs.view(B, N * C), dim=1).unsqueeze(1).unsqueeze(1)
        newpcs = (pcs - shift) / scale
        return newpcs, shift, scale

    def sin(self, data):
        device = data.device
        b, n, c = data.shape
        alpha1 = 0.5 + 0.5 * torch.rand([1, ])
        beta1 = 0.5 + 0.5 * torch.rand([1, ])
        alpha2 = 0.5 + 0.5 * torch.rand([1, ])
        beta2 = 0.5 + 0.5 * torch.rand([1, ])
        alpha3 = 0.5 + 0.5 * torch.rand([1, ])
        beta3 = 0.5 + 0.5 * torch.rand([1, ])

        if torch.rand([1, ]) > 0.5:
            k = torch.tensor(1)
        else:
            k = torch.tensor(-1)
        k = k.to(device)
        move = torch.zeros_like(data)
        newdata = torch.zeros_like(data)

        ind = torch.randperm(n)

        dist = newdata - newdata[:, ind[0], :].unsqueeze(1)

        move[:, :, 1] = k * alpha1.to(device) * torch.sin(beta1.to(device) * dist[:, :, 1])
        move[:, :, 2] = k * alpha2.to(device) * torch.sin(beta2.to(device) * dist[:, :, 2])
        move[:, :, 0] = k * alpha3.to(device) * torch.sin(beta3.to(device) * dist[:, :, 0])
        newdata[:, :, 1] = data[:, :, 1] + move[:, :, 1]
        newdata[:, :, 2] = data[:, :, 2] + move[:, :, 2]
        newdata[:, :, 0] = data[:, :, 0] + move[:, :, 0]
        return newdata

    def rotate_point_cloud(self, batch_data):
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based along up direction
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float64)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def index_mask(self, mask, idx):
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
        return 1 - mask