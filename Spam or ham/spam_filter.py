# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:05:28 2020

@author: Asus
"""
import pandas as pd
import sys

def read_file(path, file):
    """
    Function to read a datafile and drop any rows with na

    Parameters
    ----------
    path : str
        path to datafile.
    file : int
        integer to decide which file to decide what column to pop.

    Returns
    -------
    data : pandas.DataFrame
        dataframe containing the data in the specified file.
    target : pandas.Series
        series with the label column
    """
    data = pd.read_csv(path)
    data.dropna()
    #since the two different files have different name for the labels we have to separate them 
    if file == 0:
        target = data.pop('Prediction')
        data = data.drop(data.columns[0], axis = 1) #first column an id in the form of mailX which we don't want
    
    elif file == 1:
        target = data.pop('Class')
    
    else:
        print('Invalid file either 0 for word count or 1 for csv with texts')
        sys.exit()
    
    return data, target

def word_lemmatizer(words):
    """
    Helper function to clean data frame

    Parameters
    ----------
    words : list 
        list of words to lemmatize.

    Returns
    -------
    list
        the lematized words.

    """
    lemmatizer = nltk.stem.WordNetLemmatizer() 
    
    return [lemmatizer.lemmatize(o) for o in words]

def cleaning_text(data, s_words = ['th', 'en', 'ct', 'hou', 'ect']):
    """
    Function to clean a dataframe with text data by removing stopwords and lemmatize words.

    Parameters
    ----------
    data : pandas.Series
        pandas column containing data.
    stop_words : list
        additional stop words.

    Returns
    -------
    data : list
        the cleaned data.

    """
    stop_words = nltk.corpus.stopwords.words('english') + s_words
    data = data.apply(lambda x: [item for item in x.split() if item not in stop_words]) #filter out stopwords
    data = data.apply(word_lemmatizer)
    data = [" ".join(item) for item in data_train['Text'].values]

    return data

def cleaning_frequency(data_train, data_test, s_words = ['th', 'en', 'ct', 'hou', 'ect']):
    stop_words = nltk.corpus.stopwords.words('english') + s_words

    for i in range(len(stop_words)):
        if stop_words[i] in data_train.columns:
            data_train =  data_train.drop(columns = stop_words[i])
        if stop_words[i] in data_test.columns:
            data_test =  data_test.drop(columns = stop_words[i])
    
    sum = data_train.sum(axis = 0)
    to_remove = []
    for i in range(len(sum)):
        if data_train.iloc[:,i].sum(axis = 0) > 3200:
            to_remove.append(i)
    data_train = data_train.drop(data_train.columns[to_remove], axis = 1)
    data_test = data_test.drop(data_test.columns[to_remove], axis = 1)

    return data_train, data_test 


def tfid_features(data, vector, training):
    """
    Calculate tf-idf features for dataframe

    Parameters
    ----------
    data : dataframe
        dataframe consisting of dicuments.
    vector : sklearn.feature_extraction.text.TfidfVectorizer
        tf-idf vectorizer to calculate the tf-idf features.
    training : int
        wheter or not the data is training, 1, or testing, 0, data.

    Returns
    -------
    features : numpy.ndarray
        array with the features.

    """
    vector = vector
   
    if training == 1:
         features = vector.fit_transform(data) #calculate features
    
    if training == 0:
        features = vector.transform(data)
    
    return features.toarray(), vector
        
def confusion_matrix_plot(matrix, classes, normalize):
    """
    

    Parameters
    ----------
    matrix : sklearn.metrics.confussion_matrix
        a confussion matrix.
    classes : list
        list of names of classes.
    normalize : int
        wheter to normalize numbers in plotted confussion matrix.

    Returns
    -------
    None.

    """
    if normalize == 1:
        matrix = matrix.astype('float') / matrix.sum(axis = 1)[:, np.newaxis]
    
    plt.imshow(matrix, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.title('Confussion matrix')
    plt.colorbar()
    marks = np.arange(len(classes))
    plt.xticks(marks, classes, rotation=45)
    plt.yticks(marks, classes)

    fmt = '.2f' if normalize else 'd'
    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def bayes(data, target):
    """
    nanive bayes classifier

    Parameters
    ----------
    data : numpy.ndarray
        array with trainig data.
    target : pandas.Series
        series containig target data.

    Returns
    -------
    model : sklearn.mode
        trained classifier.

    """
    model = naive_bayes.GaussianNB()
    model.fit(data, target)
    
    return model

def random_forest(data, target):
    """
    nanive bayes classifier

    Parameters
    ----------
    data : numpy.ndarray
        array with trainig data.
    target : pandas.Series
        series containig target data.

    Returns
    -------
    model : sklearn.mode
        trained classifier.
    """
    model = ensemble.RandomForestClassifier()
    model.fit(data, target)
    
    return model

def linear_regression(data, target):
    """
    nanive bayes classifier

    Parameters
    ----------
    data : numpy.ndarray
        array with trainig data.
    target : pandas.Series
        series containig target data.

    Returns
    -------
    model : sklearn.mode
        trained classifier.
    """
    model = linear_model.LogisticRegression()
    model.fit(data, target)
    
    return model

def evaluate_model(model, train_data, test_data, train_target, test_target, visualize):
    """
    Evaluates a classifier

    Parameters
    ----------
    model : sklearn.model
        classifier to evaluate.
    train_data : numpy.ndarray
        array with training data.
    test_data : numpy.ndarray
        array with testing data.
    train_targets : numpy.ndarray
        array with training labels.
    test_targets : numpy.ndarray
         array with testing labels.
    visualize : int
        plot confussion matrix or not.

    Returns
    -------
    None.

    """
    train_score = model.score(train_data, train_target)
    test_score = model.score(test_data, test_targets)
    print("Training accuracy: ", train_score)
    print("Test accuracy: ",  test_score)
    
    scores = model_selection.cross_val_score(model, train_data, train_target, cv = 10) #Do cross validation
    print("\n Cross validation score: \n")
    print(scores) #Report scores of cross validation
    print("\n Mean cv score: ", np.mean(scores))
    
    if visualize == 1:
        prediction = model.predict(test_data)
        confusion_matrix_plot(metrics.confusion_matrix(test_target, target_predict), ['Ham', 'Spam'], 1)
        tn, fp, fn, tp = metrics.confusion_matrix(test_target, target_predict).ravel()
        print('Precision: ', tp / (tp + fp))
        print('Recall: ', tp / (tp + fn))
   
    return None

def pipeline(data_set, word_data, visualize, classifier, stop_words = ['th', 'en', 'ct', 'hou', 'ect']):
    data, target = read_file(data_set, word_data)
    train_data, test_data, train_target, test_target = model_selection.train_test_split(data, target, test_size=0.2, random_state=42)
    
    
    if word_data == 1:
        train_data = cleaning_text(train_data, stop_words)
        test_data = cleaning_text(test_data, stop_words)
        
        tf_idf_vector = feature_extraction.text.TfidfVectorizer()
        train_features = tfid_features(train_data, tf_idf_vector, 1)
        test_features = tfid_features(test_data, tf_idf_vector 0)
    
    elif word_data == 0:
        train_features, test_features = cleaning_frequency(train_data, test_data)
   
    else:
        print('Invalid data type: word_data should be set to 1 if dataset has complete words or 0 if dataset has columns with word frequencies for each mail.')
        return 1
   
    if classifier == 'bayes':
        model = bayes(train_features, train_targets)
    
    elif classifier == 'random forest':
        model = random_forest(train_features, train_targets)
    
    elif classifier == 'linear regression'
        model == linear_regression(train_features, train_targets)
    
    else:
        print('Invalid classifier: classifiers supported \'bayes\', \'random forest\', and \'linear regression\'.')
        return 1
    
    evaluate_model(model, train_features, test_features, train_targets, test_targets, visualize)
  
    
def main():
    nltk.download('stopwords')
    nltk.download('wordnet')
    current_folder = os.getcwd()

    word_path = current_folder + '/data/fraud_email.csv'
    frequency_path = current_folder + '/data/emails.csv'
    word_data = 1
    visualize = 1
    pipeline(word_path, word_data, visualize)
      
    return 0
      
main()
      
      
      
      
      
      
      
      
      
      