# L2X for explaining word-based CNN on IMDB

<!-- ## Running in Docker, MacOS or Ubuntu --> We provide as an example the source code to run L2X for explaining word-based CNN on IMDB.

Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/L2X
cd L2X/imdb-word/
###############################################
# Train original models, and 
# generate predictions from the original model.
# Omit '--train' when using a trained model.
python explain.py --task original --train 
###############################################
# Train L2X, and generate data with selected words.
# Omit '--train' when using a trained model.
python explain.py --task L2X --train 
###############################################
# Evaluate the consistency between the prediction on selected 
# words and that on the original sample.
python validate_explanation.py
```

See `explain.py` and `validate_explanation.py` for details. 