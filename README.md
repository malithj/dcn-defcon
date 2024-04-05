# DEFCON
* Python package for GPU texture memory based inference for Deformable Convolutions (PyTorch 2.x)

## Docker build
docker build -t dcn:v0.1 .
docker run -it --gpus=all dcn:v0.1

## Install python dependencies in virtual environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build As a Python Module
```
python setup.py develop
```

## Run all experiments
```
./run_all_benchmarks.sh
```
