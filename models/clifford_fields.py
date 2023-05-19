import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder
from typing import Tuple, Union

# from cliffordlayers.nn.functional.utils import _w_assert



class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d, zero_init=False):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out)) # nn.Conv1d(d_in, d_out, 1, 1, 0)
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out)) # nn.Conv1d(d_in, c_out, 1, 1, 0)
        if zero_init:
            nn.init.constant_(layers[-1].bias, 0.0)
            nn.init.constant_(layers[-1].weight, 0.0)
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out


    def forward(self, x):
        return self.mlp(x)


class Shift(nn.Module):
    def __init__(self, shift) -> None:
        super().__init__()
        self.shift = shift


    def forward(self, x):
        return x + self.shift


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()


    def forward(self, x):
        raise NotImplementedError()


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims), nn.ReLU(), nn.Linear(2 * proj_dims, proj_dims)
        )


    @property
    def proj_dims(self):
        return self._proj_dims


    def forward(self, x):
        return self.proj(x)


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask) # 1,1,1,3 -> 1,3


    def forward(self, F, y, alpha_ratio):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1,alpha_ratio)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj


    def inverse(self, F, x, alpha_ratio):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1,alpha_ratio)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj

class MotorLayer(nn.Module):
    def __init__(self, code_sz, bias=False, motor_sz=8):
        super().__init__()
        self.g = torch.tensor((3, 0, 1))
        self.code_sz = code_sz
        self.code_proj = nn.Sequential(
            nn.Linear(code_sz, 2 * code_sz),
            nn.ReLU(),
            nn.Linear(2 * code_sz, motor_sz)
        )
        if bias:
            # TODO: learn bias as well
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        # self.reset_parameters()

    def forward(self, x, c):
        c = self.code_proj(c)
        _,  k = get_pga_kernel(c, self.g)
        output = torch.bmm(k, x.unsqueeze(-1))#  + self.bias.view(-1)
        return output.squeeze(-1)

    def inverse(self, x, c):
        print("Inverse called")
        return x
        # c = self.code_proj(c)
        # output = F.linear(x, weight, self.bias.view(-1))
        # return output


def get_pga_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an octonion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    # assert isinstance(g, torch.Tensor)
    # assert g.numel() == 3
    # w = _w_assert(w)
    assert w.shape[-1] == 8
    a =[
            w[:, 0]**2 + w[:, 6]**2 - (w[:, 4]**2 + w[:, 5]**2),
            2*w[:, 5]*w[:, 6] - 2*w[:, 0]*w[:, 4],
            2*w[:, 0]*w[:, 5] + 2*w[:, 4]*w[:, 6],
            w[:,7]*0
        ]

    k0 = torch.stack(
        [
            w[:, 0]**2 + w[:, 6]**2 - (w[:, 4]**2 + w[:, 5]**2),
            2*w[:, 5]*w[:, 6] - 2*w[:, 0]*w[:, 4],
            2*w[:, 0]*w[:, 5] + 2*w[:, 4]*w[:, 6],
            w[:,7]*0
        ],
        dim=-1,
    )
    k1 = torch.stack(
        [
            2*w[:, 0]*w[:, 4] + 2*w[:, 5]*w[:, 6],
            w[:, 0]**2 + w[:, 5]**2 - (w[:, 4]**2 + w[:, 6]**2),
            2*w[:, 4]*w[:, 5] - 2*w[:, 0]*w[:, 6],
            w[:,7]*0
        ],
        dim=-1,
    )
    k2 = torch.stack(
        [
            2*w[:, 4]*w[:, 6] - 2*w[:, 0]*w[:, 4],
            2*w[:, 0]*w[:, 6] + 2*w[:, 4]*w[:, 5],
            w[:, 0]**2 + w[:, 4]**2 - (w[:, 5]**2 + w[:, 6]**2),
            w[:, 7]*0
        ],
        dim=-1,
    )
    k3 = torch.stack(
        [
            2*w[:, 3]*w[:, 5] - 2*w[:, 2]*w[:, 4] - 2*w[:, 0]*w[:, 1],
            2*w[:, 1]*w[:, 4] - 2*w[:, 3]*w[:, 6] - 2*w[:, 0]*w[:, 2],
            2*w[:, 2]*w[:, 6] - 2*w[:, 1]*w[:, 5] - 2*w[:, 0]*w[:, 3],
            w[:, 0]**2 + w[:, 4]**2 + w[:, 5]**2 + w[:, 6]**2
        ],
        dim=-1,
    )
    k = torch.stack([k0, k1, k2, k3], dim=-1)
    return 8, k


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def euler2rot_inv(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


def euler2rot_2d(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), theta.sin()), 1),
        torch.cat((-theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def euler2rot_2dinv(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), -theta.sin()), 1),
        torch.cat((theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


class DeformField(nn.Module):
    def __init__(self,
                 d_fcode,
                 d_feature,
                 d_hidden=[512, 512],
                 ):
        super().__init__()
        self.mlp = MLP(3+d_fcode, d_feature, d_hidden)

    def forward(self, x, fcode):
        w = fcode.repeat(x.shape[0], 1)
        feats = self.mlp(torch.cat((x, w), dim=-1))
        return x, feats


# Deform
class DeformNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out_1,
                 d_out_2,
                 n_blocks,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True):
        super(DeformNetwork, self).__init__()

        self.e2cl = torch.zeros(3, 4, requires_grad=False)
        self.e2cl[:, :3] = torch.eye(3, requires_grad=False)
        self.vp = torch.zeros(4, requires_grad=False)
        self.vp[3] = 1
        self.n_blocks = n_blocks
        for i_b in range(self.n_blocks):
            mot = MotorLayer(d_feature)

            # if l == self.num_layers_a - 2:
            #     torch.nn.init.constant_(lin.bias, 0.0)
            #     torch.nn.init.constant_(lin.weight, 0.0)
            # elif multires > 0 and l == 0:
            #     torch.nn.init.constant_(lin.bias, 0.0)
            #     torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #     torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
            # elif multires > 0 and l in self.skip_in:
            #     torch.nn.init.constant_(lin.bias, 0.0)
            #     torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #     torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
            # else:
            #     torch.nn.init.constant_(lin.bias, 0.0)
            #     torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # if weight_norm and l < self.num_layers_a - 2:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "mot"+str(i_b), mot)

        self.activation = nn.Softplus(beta=100)


    def forward(self, input_pts, input_codes, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts @ self.e2cl + self.vp
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            mot = getattr(self, "mot"+str(i_b))
            x = mot(x, input_codes)
            x = self.activation(x)

        x = x @ torch.transpose(self.e2cl, 1, 0)

        return (x, input_codes)


    def inverse(self, deformation_code, input_pts, alpha_ratio):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            mot = getattr(self, "mot"+str(i_b))
            x = mot.inverse(x, input_codes)

        return (x, input_codes)


# Deform
class TopoNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 weight_norm=True):
        super(TopoNetwork, self).__init__()
        
        dims_in = d_in
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0.0, std=1e-5)
                torch.nn.init.constant_(lin.bias, bias)
            elif multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, d_in:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :d_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif multires > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims_in - d_in):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.activation = nn.Softplus(beta=100)


    def forward(self, input_pts, deformation_code, alpha_ratio):
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
        # option 1: local deformation code
        # x = torch.cat([input_pts, deformation_code], dim=-1)
        # option 2: global deformation code
        deformations_code = deformation_code.mean(dim=0)
        print(input_pts.shape, deformation_code.shape)
        x = torch.cat([input_pts, deformation_code.repeat(input_pts.shape[0],1)], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_pts], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


# Deform
class AppearanceNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_global_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature + d_global_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()


    def forward(self, global_feature, points, normals, view_dirs, feature_vectors, alpha_ratio):
        if self.embedview_fn is not None:
            # Anneal
            view_dirs = self.embedview_fn(view_dirs, alpha_ratio)

        rendering_input = None

        global_feature = global_feature.repeat(points.shape[0],1)
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors, global_feature], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors, global_feature], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors, global_feature], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in_1,
                 d_in_2,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 multires_topo=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in_1 + d_in_2] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.embed_amb_fn = None

        input_ch_1 = d_in_1
        input_ch_2 = d_in_2
        if multires > 0:
            embed_fn, input_ch_1 = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            dims[0] += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(multires_topo, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            dims[0] += (input_ch_2 - d_in_2)

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, d_in_1:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :d_in_1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if multires > 0:
                        torch.nn.init.constant_(lin.weight[:, -(dims[0] - d_in_1):-input_ch_2], 0.0)
                    if multires_topo > 0:
                        torch.nn.init.constant_(lin.weight[:, -(input_ch_2 - d_in_2):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, input_pts, topo_coord, alpha_ratio):
        input_pts = input_pts * self.scale
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
        if self.embed_amb_fn is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        inputs = torch.cat([input_pts, topo_coord], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


    # Anneal
    def sdf(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)[:, :1]


    def sdf_hidden_appearance(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)


    def gradient(self, x, topo_coord, alpha_ratio):
        x.requires_grad_(True)
        y = self.sdf(x, topo_coord, alpha_ratio)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


# This implementation is based upon IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()


    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
