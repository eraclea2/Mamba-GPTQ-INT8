import torch
import torch.nn as nn
from torch.autograd import grad
from torch import matmul
from torch.linalg import eigh

DEV=dev = torch.device('cuda:0')


def find_layers(module, layers=[nn.Linear], name=''):#nn.Conv2d,nn.Conv1d,
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(dev)
        tar = inp.clone().to(dev)
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(dev)
        tar = inp.clone().to(dev)
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def gen_conditions(_wbits, _groupsize):
    wbits = _wbits
    groupsize = _groupsize
    conditions = []
    while True:
        if wbits >= 8:
            if groupsize == -1 or groupsize == 32:
                break

        if groupsize > 32:
            groupsize /= 2
        else:
            wbits *= 2
            groupsize = _groupsize

        conditions.append((int(wbits), int(groupsize)))
    return conditions


# copy from https://github.com/openppl-public/ppq/blob/master/ppq/quantization/measure/norm.py
def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)
    
    SNR can be calcualted as following equation:
    
        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2
    
    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.
    
        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)
    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.
    Raises:
        ValueError: _description_
        ValueError: _description_
    Returns:
        torch.Tensor: _description_
    """
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')
def hessian_vector_product(vector, net, inp):
      #  grad_params = grad(net(input), net.weight, create_graph=True)
       out = net(inp)
       out.backward()
       grad_params = out.grad
       flat_grad = torch.cat([g.view(-1) for g in grad_params])
       grad_vector_product = torch.sum(flat_grad * vector)
       hvp = grad(grad_vector_product, net.weight, retain_graph=True)
       return torch.cat([g.contiguous().view(-1) for g in hvp])

def calculate_eigen_vec(net,inp):
  num_param = net.weight.numel()#sum(p.numel() for p in model.parameters())
  v = torch.rand(num_param)
  linear_operator = hessian_vector_product(v,net,inp)
  eigenvalues, eigenvectors = eigh(linear_operator)
  eigenvectors = eigenvectors.T
  return eigenvalues, eigenvectors
