# auto-sklearn

## Running in Docker

1. Install Docker.

2. Run:

```bash
docker pull mfeurer/auto-sklearn:master
docker run -it mfeurer/auto-sklearn:master
docker run -it -v ${PWD}:/opt/nb -p 8888:8888 mfeurer/auto-sklearn:master /bin/bash -c "mkdir -p /opt/nb && jupyter notebook --notebook-dir=/opt/nb --ip='0.0.0.0' --port=8888 --no-browser --allow-root"
```

## Running in Windows

The [documentation](https://automl.github.io/auto-sklearn/master/installation.html#windows-macos-compatibility) links to:

- <https://github.com/automl/auto-sklearn/issues/431> (2019)
- <https://github.com/automl/auto-sklearn/issues/860> (2020)

From issue 860:

1. Install Ubuntu (18.04 instead of 20.04) on Windows from the Microsoft store.

2. Install conda in the Ubuntu terminal.

3. Run the following in the Ubuntu terminal:

```bash
conda create -n ml python=3.10 -y
conda activate ml
pip install -r requirements.txt

sudo apt-get install software-properties-common -y
sudo apt-add-repository universe -y
sudo apt-get update -y
sudo apt-get install python-pip -y
sudo apt install python3-pip -y
sudo apt-get install swig -y

conda install gxx_linux-64 gcc_linux-64 swig
pip uninstall numpy -y
pip uninstall numpy -y
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
pip uninstall numpy -y
pip uninstall numpy -y

pip install -r ./src/automl/auto_sklearn_/requirements.txt
```

Or:

```bash
sudo apt-get install python-dev libxml2-dev libxslt-dev
```
