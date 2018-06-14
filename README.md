# L2X 

Code for replicating the experiments in the paper [Learning to
Explain: An Information-Theoretic Perspective on Model
Interpretation](https://arxiv.org/pdf/1802.07814.pdf) at ICML 2018, by Jianbo
Chen, Mitchell Stern, Martin J. Wainwright, Michael I. Jordan.

## Dependencies
The code for L2X runs with Python and requires Tensorflow of version 1.2.1 or higher and Keras of version 2.0 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `keras`
- `pandas`
- `nltk`

Or you may run the following and in shell to install the required packages:
```shell
git clone https://github.com/Jianbo-Lab/L2X
cd L2X
sudo pip install -r requirements.txt
```
See README.md in respective folders for details.
## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/pdf/1802.07814.pdf):
```
@arxiv{chen2018learning,
title = {Learning to Explain: An Information-Theoretic Perspective on Model Interpretation},
author = {Chen, Jianbo and Song, Le and Wainwright, Martin J and Jordan, Michael I}, 
journal={arXiv preprint arXiv:1802.07814}, 
year = {2018}  
}
```