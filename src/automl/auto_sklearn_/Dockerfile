# FROM mfeurer/auto-sklearn:master
FROM python:3.9

# Install dependencies
RUN apt-get update
RUN apt-get install build-essential swig -y

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install Cython==0.29.36
RUN pip install numpy==1.26.4
RUN pip install scikit-learn==0.24
RUN pip install auto-sklearn
RUN pip install matplotlib

# Set working directory
ENV WORKDIR=/usr/src/ml-examples
RUN mkdir $WORKDIR
WORKDIR $WORKDIR

# # Install Python libraries
# COPY src/automl/auto_sklearn_/requirements.txt $WORKDIR/requirements.txt
# COPY src/automl/auto_sklearn_/requirements.txt $WORKDIR/requirements.txt
# RUN python3 -m pip install -r $WORKDIR/requirements.txt --default-timeout=120
