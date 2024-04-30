# Mamba-GPTQ-INT8

To run GPTQ, the following pip install methods are needed
```
pip install mamba-ssm
pip install git+https://github.com/huggingface/optimum.git
pip install einops
pip install texttable
pip install -q -U datasets
git clone https://github.com/PanQiWei/AutoGPTQ.git
pip install -vvv --no-build-isolation -e ./AutoGPTQ
```

If pip install AutoGPTQ failed to work, you can try the following found on https://github.com/PanQiWei/AutoGPTQ.git
```
python AutoGPTQ/setup.py install
```

You may need to add AutoGPTQ into your sys path by 

```{r} 
import sys
sys.path.append('/content/AutoGPTQ')
```
How to run GPTQ on Mamba?

```
python gptq_mamba.py
```

To run Mamba with LLM.int8() integration, run the Google Colab notebooks in the `colab_notebooks` directory
