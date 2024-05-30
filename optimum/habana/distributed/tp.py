import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
import torch.distributed as dist
from torch.nn.parameter import Parameter
from typing import Optional

is_old_shard_size = None

def move(tensor, device):
    if tensor.is_meta:
        return torch.empty_like(tensor, device=device)
    else:
        return tensor.to(device, copy=True)
    
def get_accelerator():
    return torch.device("hpu")

global num_kv_heads

def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num


def get_num_kv_heads():
    global num_kv_heads
    return num_kv_heads


def get_shard_size(total_size, mp_size, name=None, rank=None):
    global num_kv_heads
    # TODO: SW-184584 remove this WA.
    global is_old_shard_size
    if is_old_shard_size is None:
        import os
        is_old_shard_size = os.environ.get("HPU_DS_OLD_SHARD_SIZE", "1").lower() in ["true", "1"]
    last_linear = ["lm_head", "embed_out"]
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce near even division
    if rank == None:
        rank = dist.get_rank()
    if num_kv_heads != None and (is_old_shard_size or (total_size % num_kv_heads == 0 and "mlp" not in str(name)
                                                       and str(name) not in last_linear)):
        my_slices = (num_kv_heads // mp_size) + (1 if rank < (num_kv_heads % mp_size) else 0)
        return total_size * my_slices // num_kv_heads
    else:
        if total_size >= 64:
            grain_size = total_size // 64
            return (grain_size // mp_size + (1 if rank < (grain_size % mp_size) else 0)) * 64
        else:
            return total_size // mp_size + (1 if rank < (total_size % mp_size) else 0)



def get_shard_size_list(total_size, mp_size, name=None):
    shard_sizes = []
    for i in range(mp_size):
        shard_sizes.append(get_shard_size(total_size, mp_size, name, i))
    return shard_sizes

class ReplaceWithTensorSlicing:
    
    def __init__(self, mp_group=None, mp_size=1, out_dim=1, in_dim=0):
        if mp_group is not None:
            self.gpu_index = dist.get_rank(group=mp_group)
        else:
            self.gpu_index = 0
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mp_size = mp_size

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, \
            'Merging tensors is not allowed here! Please use deepspeed load_checkpoint\
            for merging your checkpoints before replacing the transformer layer with\
            inference-kernels'

    def strided_copy(self,
                     dst: Optional[torch.Tensor],
                     src: Optional[torch.Tensor],
                     num_splits: int,
                     int8: bool = False,
                     allocate_tensor: bool = False):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape

        outer_dim = 0 if int8 else -1

        if allocate_tensor:
            dst = torch.empty_like(dst)

        src_split = torch.split(src.data, src.shape[outer_dim] // num_splits, dim=outer_dim)
        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[outer_dim] == dst_shape[self.out_dim]:
                try:
                    dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
                except:
                    print(dst.shape, src.shape)
                    exit()
                dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
                if hasattr(src, 'scale'):
                    dst.scale = src.scale
                return dst
            self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
            qkv_size = dst_shape[self.out_dim] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=outer_dim) for src_s in src_split]
            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=outer_dim) for i in range(len(qkv_split[0]))
            ]
            dst = dst.reshape(-1).data.copy_(weight_split[self.gpu_index].contiguous().reshape(-1)).reshape(
                weight_split[self.gpu_index].shape)
        else:
            if src_shape[0] == dst_shape[0]:
                return torch.nn.parameter.Parameter(src)
            qkv_size = dst_shape[0] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=0) for i in range(len(qkv_split[0]))]
            dst.data.copy_(bias_split[self.gpu_index].contiguous())

        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst

    def copy(self, dst, src, int8=False, allocate_tensor=False):
        if src is None:
            return src
        assert not dst.data.is_meta  # the torch.Tensor.copy_ method used below will silently fail on meta tensors
        if allocate_tensor:
            dst = torch.empty_like(dst)
        outer_dim = 0 if int8 else 1
        inner_dim = 1 if int8 else 0
        src_shape = src.shape
        dst_shape = dst.shape
        if (len(src_shape) == 2 and len(dst_shape) == 2):

            if src_shape[inner_dim] == dst_shape[self.in_dim] and src_shape[outer_dim] == dst_shape[self.out_dim]:
                dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
            else:
                if src_shape[inner_dim] != dst_shape[self.in_dim]:
                    self.merge_assert(src_shape[inner_dim], dst_shape[self.in_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim]] if inner_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim], :])
                else:
                    self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim]] if outer_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim], :])
        else:
            if src_shape[0] == dst_shape[0]:
                dst = src if src.dtype == dst.dtype else dst.data.copy_(src)
            else:
                dst.data.copy_(src[self.gpu_index * dst_shape[-1]:(self.gpu_index + 1) * dst_shape[-1]])
        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst

class LinearLayer(nn.Module):
    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            device = torch.device("hpu")
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype, device=device))
            self.bias = Parameter(torch.empty(weight_shape[0], dtype=dtype,device=device)) if bias is not None else None            
                    
    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output

class LinearAllreduce(torch.nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group    
    
    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        return output

    def all_reduce(self, input):
        if self.mp_group is not None:
            torch.distributed.all_reduce(input, group=self.mp_group)

    def post_all_reduce(self, input):
        output = input + self.bias if (self.bias is not None) else input
        return output
        
    
class LmHeadLinearAllreduce(nn.Module):    
    def __init__(
        self,
        weight,
        rank,
        world_size,
        bias=None,
        mp_group=None,
    ):
        super(LmHeadLinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group
        self.rank = rank
        self.world_size = world_size

    def forward(self, input):
        input_shard_size = get_shard_size(input.shape[-1], self.world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.world_size, "lm_head")[0:self.rank])
        output = torch.matmul(input[:, :, input_shard_offset:input_shard_offset + input_shard_size],
                              self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            torch.distributed.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output
    

def _replace(child, name, all_reduce_linears,conv_linear_layer,mp_group):
    
    if getattr(child, "replaced", False) == True:
        return
    weight_shape = child.weight.shape
    mp_size = mp_group.size()
    mp_replace = ReplaceWithTensorSlicing(mp_group=mp_group)
    if name in all_reduce_linears:
        
        # if conv_linear_layer [weight_shape[1], weight_shape[0] // mp_size]
        # else [weight_shape[0], weight_shape[1] // mp_size]
        if conv_linear_layer:
            child.weight.data = child.weight.data.transpose(-1, -2).contiguous()
        data = child.weight.data.split(get_shard_size_list(
            weight_shape[0] if conv_linear_layer else weight_shape[1], mp_size),
                                        dim=1)
        data_dc = move(data[mp_replace.gpu_index], get_accelerator()).detach()
        del data

        setattr(child, "replaced", True)
        if name == "lm_head" or name == 'embed_out':
            return LmHeadLinearAllreduce(
                torch.nn.parameter.Parameter(data_dc, requires_grad=False), dist.get_rank(), dist.get_world_size(),
                child.bias if child.bias is None else torch.nn.parameter.Parameter(
                    move(child.bias,
                            get_accelerator())), mp_group)
        return LinearAllreduce(torch.nn.parameter.Parameter(data_dc, requires_grad=False), child.bias if child.bias is None else \
                    torch.nn.parameter.Parameter(move(child.bias, get_accelerator())), mp_group)
    else:

        # if conv_linear_layer [weight_shape[1], weight_shape[0] // mp_size]
        # else [weight_shape[0] // mp_size, weight_shape[1]]
        if conv_linear_layer:
            child.weight.data = child.weight.data.transpose(-1, -2).contiguous()

        data = child.weight.data.split(get_shard_size_list(weight_shape[0], mp_size),
                                        dim=1 if conv_linear_layer else 0)
        data_dc = move(data[mp_replace.gpu_index], get_accelerator()).detach()
        del data

        if child.bias is not None:
            bias_data = child.bias.data.split(get_shard_size_list(
                weight_shape[1] if conv_linear_layer else weight_shape[0], mp_size),
                                                dim=0)
            bias_data = move(bias_data[mp_replace.gpu_index], get_accelerator())
            bias_data_dc = torch.nn.parameter.Parameter(bias_data, requires_grad=False)
            del bias_data
        else:
            bias_data_dc = None

        setattr(child, "replaced", True)
        return LinearLayer(weight=torch.nn.parameter.Parameter(data_dc, requires_grad=False), bias=bias_data_dc)

def update_mp_params(child,mp_size):
    if getattr(child, "replaced", False) == True:
        return
    for param in [
            "n_heads", "inner_dim", "num_heads", "num_kv", "num_attention_heads", "num_attn_heads",
            "all_head_size", "embed_dim", "hidden_size", "num_key_value_heads", "num_kv_heads", "kv_n_heads",
            "d_model"
    ]:
        if hasattr(child, param):
            param_val = getattr(child, param)
            setattr(child, param, get_shard_size(param_val, mp_size))
    setattr(child, "replaced", True)
        
        
def replace_layers(module,group: ProcessGroup):
    mp_size = group.size()
    all_reduce_linears = ["o_proj","down_proj","lm_head"]
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if 'lm_head' in name:                
                setattr(module, name, _replace(child, name, all_reduce_linears,False,group))
            elif 'o_proj' in name or 'down_proj' in name:
                setattr(module, name, _replace(child, name, all_reduce_linears,False,group))
            else:
                setattr(module, name, _replace(child, name, all_reduce_linears,False,group))
        else:
            update_mp_params(child,mp_size)
            replace_layers(child,group)
    return module

# Define the replacement classes      
def setup_tensor_parallel_strategy(model: nn.Module, group: ProcessGroup):
    set_num_kv_heads(model.config.num_key_value_heads)
    return replace_layers(model,group)
