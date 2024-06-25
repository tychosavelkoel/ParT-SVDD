# ParT-SVDD

## Data
Install the software as described in [JetToyHI](https://github.com/mverwe/JetToyHI/blob/master/README_ForBScStudents.md)

## Create conda environment
Before running the code, set up a conda environment. The code is written in `python 3.9`. It uses `pytorch 2.3.0` with CUDA and [weaver](https://github.com/hqucms/weaver-core/tree/main):
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install 'weaver-core>=0.4'
```


## Run code
```
cd DSVDD/src/
./ParT-SVDD.sh
```
[ParT](https://github.com/jet-universe/particle_transformer)
[Deep SVDD](https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/master)
[SVDD](https://github.com/hqucms/weaver-core/tree/main)
