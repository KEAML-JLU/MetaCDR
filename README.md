# MetaCDR
Source code for "Cross-Domain Meta-Learner for Cold-Start Recommendation".

# Model
![](/image/MetaCDR.png)

# Requirement

- Python == 3.6.8
- PyTorch == 1.4.1

# Dataset

The public dataset MovieLens-1M can be obtained from [here](https://files.grouplens.org/datasets/movielens/ml-1m.zip)  

# Platform & Hyperparameters

Ubuntu 16.04.6, GPU (Tesla V100 32G), and CPU (Intel Xeon E5-2698 v4).  

# Run

```python
python main.py
```
```python
python main.py --test
```

# Acknowledgement

Code: [drragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch), [hoyeoplee/MeLU](https://github.com/hoyeoplee/MeLU), [waterhorse1/MELU_pytorch](https://github.com/waterhorse1/MELU_pytorch), [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)
