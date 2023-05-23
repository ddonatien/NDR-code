import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.init.normal_(layers[-1].bias, 0.0, 1e-5)
            nn.init.normal_(layers[-1].weight, 0.0, 1e-6)
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out


    def forward(self, x):
        return self.mlp(x)


class MotorNorm(nn.Module):
    def __init__(self):
        super().__init__()
        select = torch.zeros(8, 8)
        select[0, 0] = 1
        select[4, 4] = 1
        select[5, 5] = 1
        select[6, 6] = 1
        offset = torch.zeros(8)
        offset[0] = 1
        self.register_buffer('select_mat', select)
        self.register_buffer('offset', offset)

    def forward(self, x):
        x = x + self.offset
        n = torch.linalg.norm(torch.matmul(x, self.select_mat), dim=1)
        n = torch.matmul((n-1).unsqueeze(-1).repeat(1,8),
                         self.select_mat) + 1
        n = 1 / n
        return x * n


class MotorLayer(nn.Module):
    def __init__(self, code_sz, bias=False, motor_sz=8):
        super().__init__()
        self.code_sz = code_sz
        self.code_proj = nn.Sequential(
            nn.Linear(code_sz, 2 * code_sz),
            nn.LeakyReLU(),
            nn.Linear(2 * code_sz, motor_sz),
            MotorNorm()
        )
        if bias:
            # TODO: learn bias as well
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        for l in self.code_proj:
            if isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, 0.0, 0.001)
                torch.nn.init.normal_(l.bias, 0.0, 0.0001)
                # nn.init.constant_(l.bias, 0.0)
                # nn.init.constant_(l.weight, 0.0)
        # nn.init.kaiming_uniform_(
        #     self.weight.view(self.out_channels, self.in_channels * self.n_blades),
        #     a=math.sqrt(5),
        # )
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
        #         self.weight.view(self.out_channels, self.in_channels * self.n_blades)
        #     )
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c):
        c = self.code_proj(c)
        _,  k = get_pga_kernel(c)
        output = torch.bmm(k, x.unsqueeze(-1))#  + self.bias.view(-1)
        return output.squeeze(-1)

    def inverse(self, x, c):
        print("Inverse called")
        return x
        # c = self.code_proj(c)
        # output = F.linear(x, weight, self.bias.view(-1))
        # return output


def get_pga_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]
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


class DeformField(nn.Module):
    def __init__(self,
                 d_fcode,
                 d_feature,
                 n_gcodes,
                 d_hidden=[512, 512],
                 ):
        super().__init__()
        self.mlp = MLP(3+d_fcode, d_feature, d_hidden, zero_init=True)

    def forward(self, x, fcode, gcodes):
        w = fcode.repeat(x.shape[0], 1)
        query = self.mlp(torch.cat((x, w), dim=-1))
        d_k = query.shape[-1]
        print(gcodes.shape, query.shape)
        attn = F.softmax(torch.matmul(query, gcodes) / math.sqrt(d_k), dim=-1)
        return x, torch.matmul(attn, gcodes)


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

        self.e2cl = torch.zeros(3, 4, requires_grad=False, dtype=torch.float32)
        self.e2cl[:, :3] = torch.eye(3, requires_grad=False, dtype=torch.float32)
        self.vp = torch.zeros(4, requires_grad=False, dtype=torch.float32)
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
        for i_b in range(self.n_blocks):
            x = input_pts @ self.e2cl + self.vp

            mot = getattr(self, "mot"+str(i_b))
            x = mot(x, input_codes)
            x = x @ torch.transpose(self.e2cl, 1, 0)
            x = self.activation(x)

        return x


    def inverse(self, deformation_code, input_pts, alpha_ratio):
        print("Deform network inverse called")
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):

            mot = getattr(self, "mot"+str(i_b))
            x = mot.inverse(x, input_codes)

        return x


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
        deformation_code = deformation_code.mean(dim=0)
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
