# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 2020
Program to merge several csv files detailing statistics about each US states congressional district
The same code can be found in the notebook "mergeCensusData.ipynb"

@author: fredrg96
"""
#for handling data
import pandas as pd

#for multithreading
import multiprocessing as mp
from joblib import Parallel, delayed 

#other
import os
import glob
import sys


def process_csv(file, subjects, titles):
    """
    Code to process, extract feature names and data, of a single csv file.

    Parameters
    ----------
    file : str
        path to a csv file.
    subjects : list
        list of subjects to keep.
    titles : list
        list of titles to keep.

    Returns
    -------
    district_frame : pandas.DataFrame
        a dataframe holding the data for each congress district in the csv.

    """
    state_name = os.path.basename(file).split('_District', 1)[0] #get state name
    state_name = state_name.replace("_"," " )
    
    frame = pd.read_csv(file, low_memory = False) #read csv
    frame = frame[frame.columns.drop(list(frame.filter(regex = 'MOE')))] #remove MOE columns
    
    headers = frame.head(0)
    districts = [state_name + " " + col for col in headers if 'Estimate' in col] #Add the state to each district and save as a list i.e. getting it on form State District 01 Estimate .... State District 02 Estimate....
    district_frame = pd.DataFrame(districts, columns = ['District'])
    district_frame['District'] = district_frame['District'].apply(lambda x: x.split('Estimate')[0]) #remove the word Estimate
    data_row = [] #save each combination of subject + title in a row
    for i in range(len(subjects)):
        for j in range(len(titles)):
            data = frame[(frame['Subject'] == subjects[i]) & (frame['Title'] == titles[j])] #extract data
            data_row.append(data) 
    
    data_frame = pd.concat(data_row) #create a frame from the data
    data_list = data_frame.iloc[:,3:].values.tolist()
    
    features = [] 
    for i in range(29):
        feature = data_frame.iloc[i,0] + ": " + data_frame.iloc[i,1] + " " + data_frame.iloc[i,2] #get the name of each feature in form Topic + subject + Title
        features.append(feature)
    
    for i in range(len(features)):
        district_frame[features[i]] = data_list[i] #Add feature and corresponding data to our local dataFram

    #in data set there are three special features related to largest buisness area, we will get which sector employs most people, which sector has the largest payroll, and which sector has most establishments
    largest_sector_employd = frame[(frame['Subject'] == 'Paid employees for pay period including March 12')].iloc[1:]
    largest_sector_employd = largest_sector_employd.iloc[:,3:].replace(['S','O', 'D', 'G', 'H','J','N','X'],0).astype(int) #remove special signs
    largest_sector_by_paid_employees = []
    
    most_valued = frame[(frame['Subject'] == 'Annual payroll ($1,000)')].iloc[1:]
    most_valued = most_valued.iloc[:,3:].replace(['S','O', 'D', 'G', 'H','J','N','X'],0).astype(int)
    highest_payroll = []
    
    most_establishment = frame[(frame['Subject'] == 'Total Establishments')].iloc[1:]
    most_establishment = most_establishment.iloc[:,3:].replace(['S','O', 'D', 'G', 'H','J','N','X'],0).astype(int)
    most_establish = []
    
    indexes = largest_sector_employd.idxmax().values.tolist() #get the index of the row with largest value, corresponding to the industry
    for idx in range(len(indexes)):
        vocation = frame.iloc[indexes[idx],:].values.tolist()[2] #get the name
        largest_sector_by_paid_employees.append(vocation)
    
    indexes = most_valued.idxmax().values.tolist()
    for idx in range(len(indexes)):
        vocation = frame.iloc[indexes[idx],:].values.tolist()[2]
        highest_payroll.append(vocation)
        
    indexes = most_establishment.idxmax().values.tolist()
    for idx in range(len(indexes)):
        vocation = frame.iloc[indexes[idx],:].values.tolist()[2]
        most_establish.append(vocation)
        
    district_frame['Most Employees'] = largest_sector_by_paid_employees #add the special cases
    district_frame['Largest Payroll'] = highest_payroll
    district_frame['Most Establishments'] = most_establish
    
    return district_frame

def merge_csvs(subjects, titles, data_path, save_path, parallel):
    """
    Helper code to merge csv files either in a serial or parallel way.

    Parameters
    ----------
    subjects : list
        list of subjects from csv files.
    titles : list
        list of titles to keep more granular than subjects.
    data_path : str
        path to folder where data is located assumed that program is in root folder.
    save_path : path to where to save the merged data
        DESCRIPTION.
    parallel : int
        if processing csvs in parallel, 1, or not, 0.

    Returns
    -------
    None.

    """
    current_folder = os.getcwd()
    directory_path = current_folder + data_path
    
    if len(os.listdir(directory_path)) == 0:
        print('Missing data files, exiting')
        sys.exit()
        
    if parallel == 0:
        result = []
        for state in glob.glob(directory_path + '*.csv'):
            state_result = process_csv(state, subjects, titles)
            result.append(state_result)
        result = pd.concat(result)
        
    elif parallel == 1:
        num_cores = mp.cpu_count() #Use all cores
        input = glob.glob(directory_path + '*.csv') 
        if __name__ == "__main__": #Has to be used in windows to not execture main module on each child
                results = Parallel(n_jobs=num_cores)(delayed(process_csv)(link, subjects, titles) for link in input)
                result = pd.concat(results)
    
    result.to_csv(os.getcwd() + save_path) #Print the results to a csv file.   

    return None

def main():
    """
    Main code to run the merging of csv files.

    Returns
    -------
    None

    """
    load_path = '/data/sourceData/' #folder containing individual datafiles
    save_path = '/data/resultingData/congress_merged.csv' #location and name for where to save 
    subjects = ['Sex and Age', 'Race', 'Hispanic or Latino and Race', 'Place of Birth', 'Employmen Status', 'Occupation', #subjects of interest 
                'Value', 'Income and Benefits (In 2018 inflation-adjusted dollars)', 
                'Percentage of Families and People Whose Income in the Past 12 Months is Below the Poverty Level',
                'Educational Attainment']
    
    titles = ['Total population', 'Male', 'Female', 'Median age (years)','White', 'Black or African American', #titles of interest, more granular control over what to save
              'American Indian and Alaska Native', 'Asian','Native Hawaiian and Other Pacific Islander','Some other race',
              'Hispanic or Latino (of any race)','Native','Foreign born', 'Armed Forces', 'Unemployment Rate',
              'Management, business, science, and arts occupations', 'Service occupations', 'Sales and office occupations',
              'Natural resources, construction, and maintenance occupations', 'Production, transportation, and material moving occupations',
              'Median (dollars)','Total households', 'Less than $10,000', '$200,000 or more', 'Median household income (dollars)',
              'Mean household income (dollars)', 'All people', 'Percent high school graduate or higher', 'Percent bachelor\'s degree or higher',
              ]
    
    merge_csvs(subjects, titles, load_path, save_path, 1)
    
    return 0

main()
