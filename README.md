Pytorch implementation of DeepMind's enformer.

This implementation is inspired by a previous pytorch [implementation](https://github.com/lucidrains/enformer-pytorch) of enformer. This implementation has more careful weight initialization, and comes with a dataloader and training scripts.


# Setup

This package has the following dependencies:

```
python==3.8.6
einops
torch==1.10
numpy
tensorflow==2.4.1
tqdm
pandas
```

see `requirements.txt`

# Download toy data

```
python data/download.py
```

# Train model with toy data

```
python bin/train.py
```

# Citation

```
@article{avsec2021nmeth,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, Ziga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwinska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature Methods},
  year={2021}
}
```