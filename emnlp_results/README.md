### For computing sentence level UAS and LAS:

We provide code for computing sentence level UAS and LAS (averaged over the corpus).
The predictions of dependency parser should be in conll format, with two additional columns (`pred_head` and `pred_label`)
Two files are provided here for reference.

Requirements include : `scipy numpy`

1. Run the python file `save_stats.py` to save the sentence level UAS and LAS. 
`combined_dcst_mt_2_2_run2.txt` : predictions for DCST++
`combined_dcst_3_3.txt` : predictions for Baseline DCST

```
python save_stats.py combined_dcst_mt_2_2_run2.txt combined_dcst_3_3.txt
```

- This should generate files named `results1_las.txt  results1_uas.txt  results2_las.txt  results2_uas.txt` where 1 refers to dcst++ and 2 refers to baseline DCST.


2. Run significance test for UAS results:
```
python test_significance.py results1_uas.txt results2_uas.txt 0.05
```

- This should prompt user for the type of test. For t-test, type `t-test`
- Program should print the p-value of the test
- Similarly run for LAS

`test_significance.py` has been used from [this repo](https://github.com/rtmdrr/testSignificanceNLP) with some changes to adapt the code for Python3. 

3. For getting length based results for each of the files, use `get_all_stats.py`: 

