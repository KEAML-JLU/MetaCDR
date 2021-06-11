# MetaCDR
Source code for MetaCDR

# Requirement

- Python == 3.6.8
- PyTorch == 1.4.1

# Dataset

The public dataset MovieLens-1M can be obtained from [here](https://files.grouplens.org/datasets/movielens/ml-1m.zip)  
We also provide the processed dataset and encode them into vectors as example, which can be obtained from [Google Drive](https://drive.google.com/drive/folders/1V85XUpGFmnDkVoivBHg1n90WUmjEyUEo?usp=sharing)

# Platform & Hyperparameters

Our operating environment: Ubuntu 16.04.6, GPU(Tesla V100 32G), and CPU(Intel Xeon E5-2698 v4).  
The model without pretrain consumes about 25G GPU memory, appropriately reducing  ` --tasks_per_metaupdate ` (default == 16) or taking a random strategy to select data sample (default Cartesian product) can reduce memory consumption, but it will damage the accuracy of the model.
Reducing the number of update steps can also speed up training and evaluating. Experiments in Sec.5 have shown that the number of global/local update steps has little effect on the model in `User-Item Cold-start` environment, but we still recommend using the parameters we set (1/5) if conditions permit, because in most scensrio this will achieve slightly better results.  
Training options can be viewed through `argslist.py`.We will also improve our code to add more options soon.

# Run

```python
python main.py
```
```python
python main.py --test
```

# Output

In `out.log`, we show the output during the training process in the `User Cold-start` scenario on the `MovieLens 1M` dataset. 

# Acknowledgement

Code: [hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU), [waterhorse1/MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch).
