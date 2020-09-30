## Multi-task architectures for Morphological tagging in Sanskrit
Because we are training on large number of sentences (around 50k), the end-to-end training along with bucketing and padding for each epoch can make the overall training substantially slower. To deal with this issue, we first preprocess the data using the file `preprocess_multi.py`. 

For training different models, `posTrain_outputs.py` and `posTrain_outputs_deep_shortcut.py` have been provided.

#### This directory contains code and data for 3 models:

- MultiTask with Shared Representations : **MTL-Share**

- Deep MultiTask Model : **MTL-Hierarchy**

- Monolithic Sequence Tagging : **MonSeq**

## Installation and Setup
Dependencies include Python 3.6, pytorch 0.3 etc. Please follow the following steps (for pip):

- `python3 -m venv env_tagging`

- `source env_tagging/bin/activate`

- The requirements can be installed by - 
`pip install -r requirements.txt`

- Please download data from this [google drive link](https://drive.google.com/open?id=14KA5ZV5hkB9nIHmzKl3PT8pEUlbYI7F3) and untar the file
`tar -zxf tagging_data.tar.gz`

- Run `bash setup.sh`
## Data Format and Directory details

`data/ : directory for storing the data`

`data/pkl_files/ : directory for storing preprocessed data`

`data/files/50k/ : directory for storing raw input files. Size : 50000 sentences for training`

Inside the `data/files/50k/`, the train, dev and test splits are present. `train.conll`, `dev.conll`, and `test.conll` are the corresponding files. 
As for data format, every line has single token with its morphological tags and there is a blank line after every sentence. 
```
Word <tab> POS_TAG <tab> CaseNumberGender <tab> Tensemod <tab> Case <tab> Number <tab> Gender <tab> LemmaLastCharacter
```
For multi-tasking, these tags have been separated into different files. 
Example: `train_cas.conll` is the training file with only case tag.

Since, we are using only a subset of the data, we could either restrict our label space to only those labels that are seen in training **or** use the full label space from 400k sentences. 
We decide to do the latter to facilitate testing/experimentation with unseen types. We, therefore, generate the label set from full set of training sentences and provide those labels with the data in the Google Drive Link.

We also provide the preprocessed code (i.e. the pkl files) that can be used to run MTL-Share and MTL-Hierarchy models for training with 5 tag categories - **Tense-Case-Gender-Number-LemmaLastChar** 

## Running the code
The code is still in development stage and has yet to be cleaned. 
Nevertheless, you can use `preprocess_multi.py` to preprocess the files, `posTrain_outputs.py`  to run the Multi-Task Shared representations model, and `posTrain_outputs_deep_shortcut.py`  to run the Multi-Task Layered model.

### For training MTL-Share model : 

```
python posTrain_outputs.py --prefix data/pkl_files/tense_cas_num_gen_lem/50k_ --checkpoint checkpoints_new/shared_tense_cas_num_gen_lem --output_directory outputs_new/shared_tense_cas_num_gen_lem/ --word_dim 512 --char_hidden 100 --batch_size 8 --args_file data/pkl_files/tense_cas_num_gen_lem/50k_all_data.pkl --num_tasks 5 --eva_matrix a --word_dim 200 --out_files data/files/50k/test_tense.conll data/files/50k/test_cas.conll data/files/50k/test_num.conll data/files/50k/test_gen.conll data/files/50k/test_lemma.conll
```

### For training MTL-Share model : 

```
python posTrain_outputs_deep_shortcut.py --prefix data/pkl_files/tense_cas_num_gen_lem/50k_ --args_file data/pkl_files/tense_cas_num_gen_lem/50k_all_data.pkl --order_pkl_file tense-cas-num-gen-lem --order num-gen-cas-tense-lem --num_tasks 5 --output_directory outputs_new/layered_num_gen_cas_tense_lem/ --eva_matrix a --word_dim 200 --checkpoint checkpoints_new/layered_num_gen_cas_tense_lem --word_hidden 512 --batch_size 8 --char_hidden 100 --out_files data/files/50k/test_tense.conll data/files/50k/test_cas.conll data/files/50k/test_num.conll data/files/50k/test_gen.conll data/files/50k/test_lemma.conll
```

- The `--args_file` is the path to save the arguments. The path is the same as the prefix with `all_data.pkl` as a suffix. 

- the `--order_pkl_file` is the order that preprocessing file was saved as in. Example, case-gen

- the `--order` specifies the order for the layers of the desired model. Example: gen-case : means gender at layer 1, case at layer 2.

Please email if there are any issues. 
