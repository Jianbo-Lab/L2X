# L2X for explaining Hierarchical LSTM on IMDB

<!-- ## Running in Docker, MacOS or Ubuntu -->
We provide as an example the source code to run L2X for explaining Hierarchical LSTM on IMDB. First download the data:
* Download GloVe from https://nlp.stanford.edu/projects/glove/
* Put glove.6B.100d.txt in L2X/imdb-sent/data/
* Download IMDB data from https://www.kaggle.com/c/word2vec-nlp-tutorial/data
* Put labeledTrainData.tsv in L2X/imdb-sent/data/

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

See `explain.py` and `validate_explanation.py` for details. 