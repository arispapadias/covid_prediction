# project_covid

1. Create virtual environment
```
mkdir -p ${HOME}/project_aris
cd ${HOME}/project_aris
git clone git@github.com:pvlachas/project_covid.git
python3 -m venv venv-python-3
source ${HOME}/project_aris/venv-python-3/bin/activate
pip install --upgrade pip
pip install torchdiffeq matplotlib
```


MORE useful data:

https://github.com/owid/covid-19-data/blob/master/public/data/README.md



TO READ:
1. Xavier initialization, needed for very large networks (deep learning) to alleviate the vanishing gradients problem.
https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

2. Standarization of data (SCALING). Important:
https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning



