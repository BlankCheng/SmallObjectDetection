import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Conv2d(dim_in, dim_inner, kernel_size=(1 ,1), stride=(1 ,1), padding=(0 ,0))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0 ,0))
        self.phi = nn.Conv2d(dim_in, dim_inner,  kernel_size=(1 ,1), stride=(1 ,1), padding=(0 ,0))
        self.g = nn.Conv2d(dim_in, dim_inner,  kernel_size=(1 ,1), stride=(1 ,1), padding=(0 ,0))

        self.out = nn.Conv2d(dim_inner, dim_out, kernel_size=(1 ,1), stride=(1 ,1), padding=(0 ,0))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        residual = x
        # print("residual:", residual.size())
        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view \
            (batch_size, self.dim_inner, -1)

        theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


if __name__ == '__main__':

     data = torch.randint(0, 11, (4, 3, 48, 64), dtype=torch.float32)
     net = NonLocalBlock(dim_in=3, dim_inner=128, dim_out=3)
     output = net(data)
     print(output.size())

