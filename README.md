# MetaCDR
Source code for "Cross-Domain Meta-Learner for Cold-Start Recommendation".

# Model
![](/image/MetaCDR.png)

# Requirement

- Python == 3.6.8
- PyTorch == 1.4.1

# Dataset

The public dataset MovieLens-1M can be obtained from [here](https://files.grouplens.org/datasets/movielens/ml-1m.zip)  
We also provide the processed dataset and encode them into `torch.Tensor` as example, which can be obtained from [Google Drive](https://drive.google.com/drive/folders/1V85XUpGFmnDkVoivBHg1n90WUmjEyUEo?usp=sharing)

# Platform & Hyperparameters

Our operating environment: Ubuntu 16.04.6, GPU (Tesla V100 32G), and CPU (Intel Xeon E5-2698 v4).  

Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype:  
The model without pretraining consumes about 12G GPU memory, appropriately reducing  ` --tasks_per_metaupdate ` (default == 16) or taking a random strategy to select data sample like MetaCDR-PT in our paper (default Cartesian product) can reduce memory consumption, but it will damage the accuracy of the model.

Reducing the number of update steps can also speed up training and evaluating. Experiments in Sec.5 have shown that the number of global/local update steps has little effect on the model in `User-Item Cold-start` environment, but we still recommend using the parameters we set (1 and 5) if conditions permit, because in most scensrio this will achieve slightly better results.  

Training options can be viewed through `argslist.py`. We will also improve our code to add more options soon.

# Run

```python
python main.py
```
```python
python main.py --test
```

# Acknowledgement

Code: [drragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch), [hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU), [waterhorse1/MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch), [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)
