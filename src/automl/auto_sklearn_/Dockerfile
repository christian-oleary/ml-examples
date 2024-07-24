# FROM mfeurer/auto-sklearn:master
FROM python:3.9

# Set working directory
ENV WORKDIR=/usr/src/ml/ml-examples
RUN mkdir $WORKDIR
WORKDIR $WORKDIR

# Install dependencies
RUN apt-get update
RUN apt-get install build-essential swig -y

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install Cython==0.29.36
RUN pip install numpy==1.26.4
RUN pip install scikit-learn==0.24
RUN pip install auto-sklearn

# # Install Python libraries
# COPY src/automl/auto_sklearn_/requirements.txt $WORKDIR/requirements.txt
# RUN python3 -m pip install -r $WORKDIR/requirements.txt --default-timeout=120

# COPY src/automl/auto_sklearn_/requirements.txt $WORKDIR/requirements.txt
