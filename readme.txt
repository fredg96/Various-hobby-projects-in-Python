This repository contains code for different hobby projects written in python. 
- "Heart diagnosis" contains a notebook where a bayesian belief network is fitted to data related to wheter the patient has a heart dissease or not. 
  
-"Semantic segmentation for Paris cityscape" showcases how we can train a deep convolutional network, U-net architecture, to segment images in this case of a road and the surroundings. 
  
-"Spam or ham" goes through how to create a spam filter by either using the frequency of words as the input data or by having complete emails and getting tf-idf vector as features (side note, it has to be run on a computer with at least 6.75GiB of memory since no feature selection is performed). 
  
-"US congress election prediction and vissualization" showcases the pipeline from aquiring data, partly by scrapeing wikipedia, to cleaning and preparing it, and finally applying simple machine learning algorithms to try and predict how a US congressional district will vote, a fairly thorough exploration of the data is also performed. 
  
-"Variational autoencoder to generate faces" trains a deep CNN encoder and decoder and connects them to form a variational autoencoder. This notebook showcases how we can use a deep learning model to generate new data from "random noise" and furthermore it deals with the problem of having a dataset which we can not fit in memory, instead streaming from the disk. It is highly recommended to run this notebook with gpu support. 
