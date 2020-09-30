Official code for the paper ["Deep Contextualized Self-training for Low Resource Dependency Parsing"](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00294).\
If you use this code please cite our paper.

# Requirements
Simply run:
```
pip install -r requirements.txt
```
# Data
Preprocessed in `note` format.

## Multilingual Word Embeddings
Possible word embedding option: ['random', 'fasttext'] \
The multilingual word embedding (.vec extensions) should be placed under the `data/multilingual_word_embeddings` folder.


# Low Resource In-domain Experiments
In order to run the low resource in-domain experiments there are three steps we need to follow:
1. Running the base Biaffine parser
2. Running the sequence tagger(s)
3. Running the combined DCST parser

## Running code
If you want to run complete model then simply run bash script `test_original_dcsh.sh` otherwise
Refer to corrsoponding section in `test_original_dcsh.sh` to run corrsopnding segments.

## Running the base Biaffine Parser

Refer to corrsoponding section in `test_original_dcsh.sh`

## Running the Sequence Tagger
Once training the base parser, we can now run the Sequnece Tagger on any of the three proposed sequence tagging tasks in order to learn the syntactical contextualized word embeddings from the unlabeled data set. \

Refer to corrsoponding section in `test_original_dcsh.sh`

## Final step - Running the Combined DCST Parser
As a final step we can now run the DCST (ensemble) parser:

Refer to corrsoponding section in `test_original_dcsh.sh`

