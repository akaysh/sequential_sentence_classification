# <p align=center>Sequential Sentence Classification</p>

### How to run
Create Environment
```
conda create -n allennlp python=3.8
conda activate allennlp
```

The label_keys in preprocess.py contain only `coarse` for now. Uncomment the list containing all the label types and comment/remove current one to run model on all the labels. The model will do classification one label type at a time which is controlled by the `LABEL_KEY` parameter in `train.sh` script.

```
pip install -r requirements.txt
python scripts/preprocess.py
scripts/train.sh tmp_output_dir
```

Update the `scripts/train.sh` script with the appropriate hyperparameters and datapaths. Right now the hyperparameters and data path are according to the Discourse dataset. 

