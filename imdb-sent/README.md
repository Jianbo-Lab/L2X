# L2X for explaining Hierarchical LSTM on IMDB

We provide as an example the source code to run L2X for explaining Hierarchical LSTM on IMDB. First download the data:
* Download IMDB data from https://www.kaggle.com/c/word2vec-nlp-tutorial/data
* Put labeledTrainData.tsv and testData.tsv in L2X/imdb-sent/data/

Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/L2X
cd L2X/imdb-sent/
###############################################
# Train original models, and 
# generate predictions from the original model.
# Omit '--train' when using a trained model.
python explain.py --task original --train 
###############################################
# Train L2X, and generate data with selected sentences.
# Omit '--train' when using a trained model.
python explain.py --task L2X --train 
###############################################
# Evaluate the consistency between the prediction on selected 
# sentences and that on the original sample.
python validate_explanation.py
```

Note that it takes several minutes to construct data sets during the first run. See `explain.py` and `validate_explanation.py` for details. 