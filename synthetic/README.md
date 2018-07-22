# L2X for synthetic data.

We provide as an example the source code to run L2X for synthetic data. 

Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/L2X
cd L2X/synthetic/
###############################################
# Train L2X, 
# and generate median ranks of selected features for datatype by L2X.
# Omit '--train' when using a trained model.
python explain.py --train --datatype datatype
```

The datatype can be selected from 'orange_skin', 'XOR', 'nonlinear_additive',
and 'switch'. See 'make_data.py' and the original paper for details.
