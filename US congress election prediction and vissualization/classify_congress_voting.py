# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 2020
Machine learning part of predicting voting of us congressional district.
Explaination of code is in the notebook "machieLearning.ipynb".

@author: fredg96
"""

import pandas as pd
import numpy as np
import os

from sklearn import ensemble, preprocessing, model_selection, metrics, decomposition 


def random_forest(data_train, target_train, depth = 10, jobs = -1):
    """
    Random forest classifier

    Parameters
    ----------
    data_train : numpy array
        array with training data.
    target_train : numpy array
        array with labels.
    depth : int
        depth of tree. The default is 10.
    nr_trees : int
        number of trees. The default is 100.
    jobs : int
        number of parallel jobs. The default is -1 to use all

    Returns
    -------
    model : scikit.model
        a random forest classifier.

    """
    model = ensemble.RandomForestClassifier(max_depth = depth, n_jobs = jobs)
    model.fit(data_train, target_train)
    return model

def preprocess_data(data, split_percentage, normalize, pca, variance_explained):
    """
    Function to drop some columns from the input data. The data is then split into training and testing data
    and finally normalized using min max scaling.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing the data to use.
    split_percentage : int
        how high percentage of data to keep as test data, interval [0, 1].
    normalize : int
        if applying, 1, min max scaling or not, 0.
    pca : int
        if performing, 1, pca on data or not, 0.
    variance_explained : int
        percentage of variance which the pca transformation should be able to explain [0, 1].

    Returns
    -------
    data_train : numpy array
        training data either scaled or not.
    data_test : numpy array
        test data either scaled or not.
    targets_train : numpy array
        array containing labels for training.
    targets_test : numpy array
        array with labels for testing.

    """
    data = data.drop(columns = ['Unnamed: 0', 'District', 'Member', 'Prior experience', 'Education', 
                               'Assumed office', 'Residence', 'Born', 'Spouse', 'Childrens', 'Years in office',
                               'Age when elected', 'Race Total population','Birth Total Population', 'State'])
    data['Party'] = data['Party'].replace({'Republican':0,'Democratic':1,'Libertarian':2})

    data = pd.concat([data,pd.get_dummies(data['Most Employees'], prefix='employer')],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Largest Payroll'], prefix='paying')],axis=1)
    data = pd.concat([data,pd.get_dummies(data['Most Establishments'], prefix='established')],axis=1)
    data = data.drop(columns = ['Most Employees', 'Largest Payroll', 'Most Establishments'])
    
    target = data.pop('Party')
    
    data_train, data_test, targets_train, targets_test = model_selection.train_test_split(data, target, 
                                                                               test_size = split_percentage, random_state=42)
    if normalize == 1:
        min_max_scaler = preprocessing.MinMaxScaler()
        data_train = min_max_scaler.fit_transform(data_train)
        data_test = min_max_scaler.transform(data_test)
    
    if pca == 1:
        pca = decomposition.PCA(variance_explained)
        data_train = pca.fit_transform(data_train)
        data_test = pca.transform(data_test)
        
    return data_train, data_test, targets_train, targets_test


def evaluate_model(model, data_train, data_test, target_train, target_test, detailed, folds):
    """
    Function to evaluate a classifier.

    Parameters
    ----------
    model : sklearn.model
        Classifier to evaluate.
    data_train : numpy array
        array with training data for k-fold cv.
    data_test : numpy array
        array with testing data.
    target_train : numpy array
        array with training labels for k-fold cv.
    target_test : numpy array
        array with labels for testing.
    detailed : int
        returns a confussion matrix and precision and recall.
    folds : int
        number of folds for k-fold cv.

    Returns
    -------
    results : list
        list with resulting metrics.

    """
    results = []
    
    if folds != 1:
        cross_validation_score = model_selection.cross_val_score(model, data_train, target_train, cv = folds)
        results.append('cv score: ')
        results.append(cross_validation_score)
        results.append('mean cv score: ')
        results.append(np.mean(cross_validation_score))
    
    prediction = model.predict(data_test)
    accuracy = metrics.accuracy_score(target_test, prediction)
    results.append('acc: ')
    results.append(accuracy)

    if detailed == 1:
        print(metrics.confusion_matrix(target_test, prediction))

        results.append('Precision: ')
        results.append(metrics.precision_score(y_true = target_test, y_pred = prediction, average = 'weighted', zero_division = 0))
        results.append('Recall: ')
        results.append(metrics.recall_score(y_true = target_test, y_pred = prediction, average = 'weighted'))
    
    return results

def machine_learning(path, details, parameters, jobs, normalize, pca, variance_explained, folds):
    """
    Pipeline function for machine learning

    Parameters
    ----------
    path : str
        path to data file.
    details : int
        wheter to report detailed metrics.
    parameters : int
        depth of each tree.
    jobs : int
        wheter to leverage multicore cpus.
    normalize : int
        normalize data or not.
    pca : int
        perform pca or not.
    variance_explained : int
        variance of data which pca should still be able to explain [0,1].
    folds : int
        number of folds for k-fold cv.

    Returns
    -------
    None.

    """
    current_folder = os.getcwd()
    data = pd.read_csv(current_folder + path)
    
    x_train, x_test, y_train, y_test = preprocess_data(data, 0.9, normalize, pca, variance_explained) 
   
    model = random_forest(x_train, y_train, parameters[0], jobs = jobs)
    
    results = evaluate_model(model, x_train, x_test, y_train, y_test, details, folds)
    print(results)
    return None

def main():
    """
    Driver function

    Returns
    -------
    None.

    """
    data_path = '/data/resultingData/merged_data.csv'
    details = 1
    parameters = [18]
    jobs = 1
    normalize = 1
    pca = 0
    variance_explained = 0.99
    folds = 10
    
    machine_learning(data_path, details, parameters, jobs, normalize, pca, variance_explained, folds)
    
    return 0

main()
