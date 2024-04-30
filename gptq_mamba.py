#cell 2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gptq
from tqdm import tqdm

#cell 3
from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from gptq import *
from quantizer import *
from pack_utils import *
from utils import find_layers, get_wikitext2, get_ptb, torch_snr_error


# Constant for Mamba
dev='cuda'
quantize_config_bits=4
mc = MambaConfig()
mc.d_model = 64
mc.n_layer = 64
mc.vocab_size = 512

def skip(*args, **kwargs):
  pass

# torch.nn.init.kaiming_uniform_ = skip
# torch.nn.init.uniform_ = skip
# torch.nn.init.normal_ = skip

model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m", device='cuda',dtype=torch.float16)#,dtype=torch.float16

layers=model.backbone.layers

# print(model.backbone.embedding)# = model.embedding.to(dev)
model.backbone.embedding= model.backbone.embedding.to(dev)
# print(model.backbone.norm_f)# = model.norm.to(dev)
model.backbone.norm_f = model.backbone.norm_f.to(dev)
dtype=next(iter(model.parameters())).dtype
# print(next(iter(model.parameters())).dtype)

# Constant for GPTQ quantization
quantizers = {}
nsamples=1024+256+128+256+128
percdamp=1#0.6
groupsize=blocksize=128
act_order=True
static_groups=True
hidden_size=model.backbone.embedding.weight.shape[1]#32
trits=False
seq_len=1024#2048#1024#
sym=True
seed=0
model_id="EleutherAI/gpt-neox-20b"#"state-spaces/mamba-2.8b-hf"
dataloader, testloader=get_ptb(nsamples, seed, seq_len, model_id)#get_wikitext2

# Create quantizer



temp=0

@torch.no_grad()
def two_layer_mamba():
  inps = torch.randn((nsamples,seq_len, hidden_size), dtype=dtype, device=dev)
  outs = torch.zeros_like(inps)
  cache = {'i': 0, 'attention_mask': None}
  layers[0] = layers[0].to(dev)

  # class Catcher(nn.Module):
  #     def __init__(self, module):
  #       super().__init__()
  #       self.module = module
  #     def forward(self, inp, **kwargs):
  #       inps[cache['i']] = inp
  #       cache['i'] += 1
  #       cache['attention_mask'] = kwargs['attention_mask']
  #       raise ValueError
  # print(layers[0])
  # layers[0] = Catcher(layers[0])
  # for batch in dataloader:
  #     try:
  #       model(batch[0].to(dev))
  #     except ValueError:
  #       pass
  # layers[0] = layers[0].module
  # layers[0] = layers[0].cpu()
  torch.cuda.empty_cache()
  # attention_mask=cache['attention_mask']
  print('Ready.')

  for i in range(len(layers)):#len(layers)
      layer = layers[i].to(dev)
      layer.use_fast_path=False

      subset = find_layers(layer)
      global gptq1
      gptq1= {}
      for name in subset:
          gptq1[name] = GPTQ(subset[name])
          gptq1[name].quantizer = Quantizer()
          gptq1[name].quantizer.configure(
              quantize_config_bits, perchannel=True, mse=False, sym=sym, trits=trits
          )

      def add_batch(name,gptq1):
          def tmp(_, inp, out):
              gptq1[name].add_batch(inp[0].data, out.data)
          return tmp
      handles = []

      for name in subset:
          handles.append(subset[name].register_forward_hook(add_batch(name,gptq1)))
      errors=[]
      for j in range(nsamples):
        temp=layer(inps[j].unsqueeze(0))[0]
        errors.append(torch_snr_error(outs[j].unsqueeze(0), temp).item())
        outs[j]=temp.squeeze(0)
        
    #   print("error before quantization: ",sum(errors)/len(errors))
      del errors
     
      for h in handles:
          h.remove()
      for name in subset:
          if name not in {'mixer.x_proj', 'mixer.out_proj','mixer.conv1d'}:
            continue
          print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
          scale, zero, g_idx= gptq1[name].fasterquant(
              percdamp=percdamp, group_size=128, actorder=act_order#, static_groups=static_groups
          )
          quantizers['backbone.layers.%d.%s' % (i, name)] = (gptq1[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu())#gptq[name].quantizer
          gptq1[name].free()
          
      errors=[]
      for j in range(nsamples):
          temp=layer(inps[j].unsqueeze(0))[0]
          errors.append(torch_snr_error(outs[j].unsqueeze(0), temp).item())
          outs[j]=temp.squeeze(0)
    #   print("errors after quantization: ",sum(errors)/len(errors))
      
      layers[i] = layer.cpu()
      del layer
      del gptq1
      torch.cuda.empty_cache()

      inps, outs = outs, inps

  # model.config.use_cache = use_cache
two_layer_mamba()


# pack model
CPU='cpu'
desc_act=False
force_layer_back_to_cpu=False
pack_model(
            model=model,
            quantizers=quantizers,
            bits=quantize_config_bits,
            group_size=groupsize,
            use_triton=False,
            use_cuda_fp16=True,
            desc_act=desc_act,
            warmup_triton=False,
            force_layer_back_to_cpu=force_layer_back_to_cpu,
            use_marlin=False,
        )



# Get Wiki2 test data
model_id="EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")



# Run inference
## Get perplexity and throughput
device = "cuda:0"
model.to(device)
max_length = 1024
stride = 512
num_iterations = 10
seq_len = encodings.input_ids.size(1)
torch.set_printoptions(precision=8,threshold=10000)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

counter = 0
num_output_tokens = 0

nlls = []
current_ppls=[]
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        labels = target_ids
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        generated_tokens = torch.argmax(logits, dim=-1)
        # Count non-padding tokens in the generated sequence
        num_output_tokens += torch.sum(generated_tokens != -100).item()
    nlls.append(neg_log_likelihood)
    curr_ppl = torch.exp(torch.stack(nlls).mean())
    current_ppls.append(curr_ppl)
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

end.record()

# Waits for everything to finish running
torch.cuda.synchronize()
num_tokens = end_loc
print("RESULTS:")
print("num input tokens:", num_tokens)
print("num output tokens:", num_output_tokens)
print("time in seconds:", start.elapsed_time(end)/1000)
print("input tokens per second", num_tokens/(start.elapsed_time(end)/1000))
print("output tokens per second", num_output_tokens/(start.elapsed_time(end)/1000))
print("neg log likelihood is ", nlls) # testing
ppl = torch.exp(torch.stack(nlls).mean())
print("perplexity is: ", ppl)
# print("real perplexity is ", sum(current_ppls)/len(current_ppls))


# Get quantized model size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))