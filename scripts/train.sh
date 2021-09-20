#!/bin/bash

export SEED=15270
export PYTORCH_SEED=`expr $SEED / 10`
export NUMPY_SEED=`expr $PYTORCH_SEED / 10`

# path to bert vocab and weights
export BERT_VOCAB=https://ai2-s2-research.s3-us-west-2.amazonaws.com/scibert/allennlp_files/scivocab_uncased.vocab
export BERT_WEIGHTS=https://ai2-s2-research.s3-us-west-2.amazonaws.com/scibert/allennlp_files/scibert_scivocab_uncased.tar.gz

LABEL_KEY=coarse

# path to dataset files
export TRAIN_PATH="data/Discourse/train_$LABEL_KEY.jsonl"
export DEV_PATH="data/Discourse/dev_$LABEL_KEY.jsonl"
export TEST_PATH="data/Discourse/test_$LABEL_KEY.jsonl"

# model
export USE_SEP=true  # true for our model. false for baseline
export WITH_CRF=false  # CRF only works for the baseline

# training params
export cuda_device=0
export BATCH_SIZE=4
export LR=5e-5
export TRAINING_DATA_INSTANCES=209
export NUM_EPOCHS=10

# limit number of sentneces per examples, and number of words per sentence. This is dataset dependant
export MAX_SENT_PER_EXAMPLE=20
export SENT_MAX_LEN=50

# this is for the evaluation of the summarization dataset
export SCI_SUM=false
export USE_ABSTRACT_SCORES=false
export SCI_SUM_FAKE_SCORES=false  # use fake scores for testing

CONFIG_FILE=sequential_sentence_classification/config.jsonnet

python -m allennlp.run train $CONFIG_FILE  --include-package sequential_sentence_classification -s $SERIALIZATION_DIR "$@"
