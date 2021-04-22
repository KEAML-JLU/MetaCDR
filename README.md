# MetaCDR
Source code for MetaCDR

# Requirement

- Python 3.6.8
- PyTorch 1.4.1

# Dataset

The public dataset MovieLens-1M can be obtained from [here](https://files.grouplens.org/datasets/movielens/ml-1m.zip)  
We also provide the processed dataset as example and encode them into vectors, which can be obtained from [Google Drive](https://drive.google.com/drive/folders/1V85XUpGFmnDkVoivBHg1n90WUmjEyUEo?usp=sharing)

# Platform & Hyperparameters

Our operating environment: Ubuntu 16.04.6, GPU(Tesla V100 32G), and CPU(Intel Xeon E5-2698 v4).  
The model consumes about 25G GPU memory, appropriately reducing  ` --tasks_per_metaupdate `  can reduce memory consumption.  
Reducing the number of update steps can also speed up training and evaluating. Although experiments have shown that the number of global/local update steps has little effect on the model in a cold start environment, we recommend using the parameters we set (1/5) if conditions permit, because in most scensrio this will achieve slightly better results.

# Run

```python
python main.py
```
```python
python main.py --test
```
