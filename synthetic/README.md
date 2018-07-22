# L2X for Synthetic.

We provide as an example the source code to run L2X for synthetic data. 

Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/L2X
cd L2X/synthetic/
###############################################
# Train L2X on the data of a specific datatype, 
# and generate median ranks of selected features by L2X.
# Omit '--train' when using a trained model.
python explain.py --train --datatype datatype
```

The datatype can be selected from 'orange_skin', 'XOR', 'nonlinear_additive',
and 'switch'. See 'make_data.py' and the original paper for details.
