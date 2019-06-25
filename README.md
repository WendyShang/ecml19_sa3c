# Stochastic Activation Actor Critic Methods

This repository demonstrates our proposed stochastic activation actor critic methods, published at ECML-PKDD 2019. We use Qbert, BeamRider, and Seaquest to showcase sa3c, fully stochastic a3c (fa3c), and hierarchical prior sa3c (hpa3c) respectively. In addition, we provide baseline a3c and noisy-net training code as comparison. 

If you find our work and code useful, please cite our paper [[pdf](http://www-personal.umich.edu/~shangw/papers/ecml19.pdf)][[appendix](http://www-personal.umich.edu/~shangw/papers/ecml19_appendix.pdf)]:
```
@inproceedings{shang2019stochastic,
  title={Stochastic Activation Actor Critic Methods},
  author={Shang, Wenling and van Hoof, Herk and Welling, Max},
  booktitle={ECML-PKDD},
  year={2019}
}
```

## Prerequisites
```bash
conda create -n py36 python=3.6 anaconda
source activate py36
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install -c menpo opencv
pip install gym
pip install gym[atari]
pip3 install logger
```

## Baseline A3C
for Qbert
```bash
python main_atari.py --model_type baseline --save_best --game Qbert-v4 
```
for BeamRider
```bash
python main_atari.py --model_type baseline --save_best --game BeamRider-v4
```
for Seaquest
```bash
python main_atari.py --model_type baseline --save_best --game Seaquest-v4
```

## NoisyNet A3C
for Qbert
```bash
python main_atari.py --model_type nn --save_best --game Qbert-v4
```
for BeamRider
```bash
python main_atari.py --model_type nn --save_best --game BeamRider-v4 
```
for Seaquest
```bash
python main_atari.py --model_type nn --save_best --game Seaquest-v4 
```

## Stochastic Activation A3C
SA3C for Qbert
```bash
python main_atari.py --model_type sa3c --save_best --game Qbert-v4 --sig 4
```
FSA3C for BeamRider
```bash
python main_atari.py --model_type fsa3c --save_best --game BeamRider-v4 --sig 4
```
HPA3C for Seaquest
```bash
python main_atari.py --model_type hpa3c --save_best --game Seaquest-v4 --crelu
```

## Acknowledgments
We greatly appreciate the dev teams for PyTorch, Gym and ALE. Our implementation has also taken inspiration from the following excellent repositories:
 - rl_a3c_pytorch: https://github.com/dgriff777/rl_a3c_pytorch
 - DeepRL: https://github.com/ShangtongZhang/DeepRL
 - NoisyNet-A3C: https://github.com/Kaixhin/NoisyNet-A3C
 - a3c_continuous: https://github.com/dgriff777/a3c_continuous
 - ACER: https://github.com/Kaixhin/ACER
 - doom-net-pytorch: https://github.com/akolishchak/doom-net-pytorch
