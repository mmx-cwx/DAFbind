### Diffusion-couped feature attention fusion with dual language models for accurate prediction of protein-nucleic acid binding sites
***
#### Introduction
****
DAFBind, a sequence-based predictive model that enriches the semantic representation of amino acid sequences via fusing features embedded by dual language models. To fully capture inter-feature correlations, we propose feature-axis attention mechanism and couple diffusion model for reconstruction of attention matrix within it. Finally, we design a multiscale residue convolution fusion to achieve effective feature fusion. Evaluations using five-fold cross-validation and independent testing demonstrate that DAFBind not only outperforms existing sequence-based methods by a significant margin but also surpasses most state-of-the-art structure-based approaches in terms of predictive performance.
***
#### Farmework
***
![img_1.png](img_1.png)
***
#### Dataset
***
The dataset containing DNA/RNA PDB files can be downloaded from the following sources:

GraphBind: http://www.csbio.sjtu.edu.cn/bioinf/GraphBind/

GraphSite: https://github.com/biomed-AI/GraphSite
***

#### System Requirements
***
he source code developed in Python 3.10.14 using PyTorch 2.2.2 .The main python dependencies are given below.DAFbind is supported for any standard computer and operating system (Windows/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

torch 2.2.2  
Python 3.10.14  
pytorch-cuda 11.8  
pandas 2.2.1  
numpy 1.26.4  
transformers 4.37.2  

***

#### Modifying the transformers Library
***
In this project, we modify the Hugging Face transformers library to address the issue of model input length limitation. By default, models downloaded from Hugging Face support a maximum sequence length of 1024 tokens. Our modifications aim to extend this limit, allowing for handling longer input sequences efficiently.  
You can refer to the following methods for implementation. https://blog.csdn.net/weixin_40959890/article/details/128969364
***
#### Testing
***
The hyperparameters have already been modified in the code; you just need to determine the file location. You can get T5 from here https://huggingface.co/Rostlab/prot_t5_xl_uniref50.    
Run the DAFbind for DNA:  
python esm_t5_diffusion_poly_loss_esm650m_64_T600_K_4.py  

Run the DAFbind for RNA:  
python esm_t5_diffusion_poly_loss_esm650m_32_T600_K_4.py  
***