# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 2020
Program to clean data code is explained in more detail in notebook
"dataCleaning.ipynb"

@author: fredg96
"""
import pandas as pd
import  numpy as np
import os

def clean_member(path, general_level_of_education):
    """
    Functoion which cleans the members data, for reasoning behind decissions see notebook.

    Parameters
    ----------
    path : str
        path to datafile containing information about congress members.
    
    general_level_of_education : str
        what level of education which is assumed if nothing else is stated.
    Returns
    -------
    data : pandas.DataFrame
        cleaned members data.

    """
    data = pd.read_csv(path)
    data = data.drop(columns = ['Unnamed: 0','Party']) #drop unwanted columns
    data = data.rename(columns = {'Party.1':'Party'}) 
    
    #the following lines add data which couldn't be joined due to the name being different 
    data = data.iloc[:435, :]
    data.loc[23, 'Spouse'] = 'Yes'
    data.loc[23, 'Childrens'] = 6
    data.loc[27, 'Spouse'] = 'Yes'
    data.loc[27, 'Childrens'] = 1
    data.loc[141, 'Spouse'] = 'Yes'
    data.loc[141, 'Childrens'] = 2
    data.loc[387, 'Spouse'] = 'Yes'
    data.loc[387, 'Childrens'] = 3
    data.loc[84, 'Spouse'] = 'Yes'
    data.loc[84, 'Childrens'] = 3
    data.loc[128, 'Spouse'] = 'Yes'
    data.loc[128, 'Childrens'] = 4
    data.loc[133, 'Spouse'] = 'No'
    data.loc[133, 'Childrens'] = 3
    data.loc[138, 'Spouse'] = 'Yes'
    data.loc[138, 'Childrens'] = 3
    data.loc[187, 'Spouse'] = 'Yes'
    data.loc[187, 'Childrens'] = 3
    data.loc[199, 'Spouse'] = 'Yes'
    data.loc[199, 'Childrens'] = 1
    data.loc[244, 'Spouse'] = 'Yes'
    data.loc[244, 'Childrens'] = 2
    data.loc[254, 'Spouse'] = 'Yes'
    data.loc[254, 'Childrens'] = 3
    data.loc[261, 'Spouse'] = 'Yes'
    data.loc[261, 'Childrens'] = 2
    data.loc[269, 'Spouse'] = 'Yes'
    data.loc[269, 'Childrens'] = 1
    data.loc[274, 'Spouse'] = 'No'
    data.loc[274, 'Childrens'] = 5
    data.loc[284, 'Spouse'] = 'Yes'
    data.loc[284, 'Childrens'] = 3
    data.loc[277, 'Spouse'] = 'Yes'
    data.loc[277, 'Childrens'] = 3
    data.loc[312, 'Spouse'] = 'Yes'
    data.loc[312, 'Childrens'] = 0
    data.loc[344, 'Spouse'] = 'No'
    data.loc[344, 'Childrens'] = 0
    data.loc[359, 'Spouse'] = 'Yes'
    data.loc[359, 'Childrens'] = 2
    data.loc[370, 'Spouse'] = 'Yes'
    data.loc[370, 'Childrens'] = 5
    data.loc[386, 'Spouse'] = 'Yes'
    data.loc[386, 'Childrens'] = 3
    data.loc[394, 'Spouse'] = 'Yes'
    data.loc[394, 'Childrens'] = 0
    
    #the following code standardize the level attained by the congress members, assuming a masters degree if nothing else was stated.
    data = data[data.Member != 'Vacant']
    data = data.reset_index(drop=True)
    data.loc[data['Education'].str.contains('JD'),
             'Education'] = 'JD'
    data.loc[data['Education'].str.contains('PhD'),
             'Education'] = 'PhD'
    data.loc[data['Education'].str.contains('MD|DDS|DVM'), 
             'Education'] = 'Profesional Doctorate'
    data.loc[data['Education'].str.contains('MBA|MS|MPA|MPP|MFA|MA|MPhil|MUP|MSN|MPH|LLM|MHS|THM'),
             'Education'] = 'Master'
    data.loc[data['Education'].str.contains('BA|BS'),
             'Education'] = 'Bachelor'
    data.loc[data['Education'].str.contains('Bachelor|Master|PhD|Profesional Doctorate|JD') == False, 
                   'Education'] = general_level_of_education    

    #fill in if having prior public experience
    data.loc[data['Prior experience'].str.contains('Board of|City Council|House of Representatives|Judge'
                  '|Mayor|Senate|Assembly|Sheriff|State Director|Commissioner|Deputy Secretary of the Interior'
                  '|Police Department|Ambassador|Secretary of Health and Human Services|Treasurer|City-County Council'
                  '|Judge-Executive|Restoration Authority|County Executive|Governor|Secretary of Defense for International Security Affairs'
                  '|Presidential Task Force|Municipal Council|New Mexico Democratic Party|Town Council'
                  '|Puerto Rican Community Affairs|Supreme Court|Republican Committee|Secretary of State|Commission'
                  '|Texas Justice of the Peace|House of Delegates'),'Prior experience'] = 'Yes'
    data.loc[data['Prior experience'].str.contains('Yes') == False,'Prior experience'] = 'No'
    data.rename(columns = {'Prior experience':'Prior Public Experience'})

    data['Assumed office'] = data['Assumed office'].apply(lambda x: x.split('(special)')[0]) #date when elected

    #calculate how long the person has been a member of congress and at what age they were elected
    data = data.astype({'Assumed office': int, 'Born':int})
    years_in_office = []
    age_when_elected = []
    for i in range(431):
        years = 2020-data.loc[i,'Assumed office']
        age = data.loc[i,'Assumed office']-data.loc[i,'Born']
        years_in_office.append(years)
        age_when_elected.append(age)
    data['Years in office'] = years_in_office
    data['Age when elected'] = age_when_elected
    
    #fill in number of childrens
    data.loc[26, 'Childrens'] = '1'
    data.loc[32, 'Childrens'] = '5'
    data.loc[57, 'Childrens'] = '4'
    data.loc[110, 'Childrens'] = '1'
    data.loc[117, 'Childrens'] = '1'
    data.loc[138, 'Childrens'] = '2'
    data.loc[139, 'Childrens'] = '2'
    data.loc[154, 'Childrens'] = '4'
    data.loc[152, 'Childrens'] = '0'
    data.loc[153, 'Childrens'] = '4'
    data.loc[167, 'Childrens'] = '1'
    data.loc[168, 'Childrens'] = '1'
    data.loc[177, 'Childrens'] = '3'
    data.loc[178, 'Childrens'] = '3'
    data.loc[193, 'Childrens'] = '1'
    data.loc[194, 'Childrens'] = '1'
    data.loc[203, 'Childrens'] = '2'
    data.loc[211, 'Childrens'] = '4'
    data.loc[214, 'Childrens'] = '3'
    data.loc[215, 'Childrens'] = '3'
    data.loc[225, 'Childrens'] = '1'
    data.loc[226, 'Childrens'] = '1'
    data.loc[284, 'Childrens'] = '3'
    data.loc[314, 'Childrens'] = '1'
    data.loc[317, 'Childrens'] = '1'
    data.loc[338, 'Childrens'] = '0'
    data.loc[342, 'Childrens'] = '4'
    data.loc[346, 'Childrens'] = '3'
    data.loc[363, 'Childrens'] = '0'
    data.loc[373, 'Childrens'] = '2'
    data.loc[380, 'Childrens'] = '4'
    data.loc[381, 'Childrens'] = '2'
    data.loc[392, 'Childrens'] = '5'
    data.loc[424, 'Childrens'] = '2'
    data.loc[2, 'Childrens'] = '0'
    data = data.replace(r'^\s*$', np.nan, regex=True).fillna('0') #Replace empty with first NaN and then with '0'
       
    #fill in marital status
    data.loc[7, 'Spouse'] = 'Yes'
    data.loc[14, 'Spouse'] = 'No'
    data.loc[31, 'Spouse'] = 'No'
    data.loc[34, 'Spouse'] = 'Yes'
    data.loc[38, 'Spouse'] = 'No'
    data.loc[52, 'Spouse'] = 'No'
    data.loc[63, 'Spouse'] = 'Yes'
    data.loc[63, 'Spouse'] = 'No'
    data.loc[76, 'Spouse'] = 'No'
    data.loc[79, 'Spouse'] = 'Yes'
    data.loc[85, 'Spouse'] = 'No'
    data.loc[98, 'Spouse'] = 'No'
    data.loc[109, 'Spouse'] = 'No'
    data.loc[117, 'Spouse'] = 'No'
    data.loc[128, 'Spouse'] = 'Yes'
    data.loc[131, 'Spouse'] = 'Yes'
    data.loc[139, 'Spouse'] = 'Yes'
    data.loc[170, 'Spouse'] = 'Yes'
    data.loc[174, 'Spouse'] = 'Yes'
    data.loc[178, 'Spouse'] = 'No'
    data.loc[184, 'Spouse'] = 'No'
    data.loc[208, 'Spouse'] = 'No'
    data.loc[209, 'Spouse'] = 'No'
    data.loc[222, 'Spouse'] = 'No'
    data.loc[223, 'Spouse'] = 'No'
    data.loc[228, 'Spouse'] = 'No'
    data.loc[267, 'Spouse'] = 'No'
    data.loc[282, 'Spouse'] = 'No'
    data.loc[306, 'Spouse'] = 'No'
    data.loc[320, 'Spouse'] = 'Yes'
    data.loc[346, 'Spouse'] = 'No'
    data.loc[349, 'Spouse'] = 'Yes'
    data.loc[350, 'Spouse'] = 'Yes'
    data.loc[352, 'Spouse'] = 'Yes'
    data.loc[397, 'Spouse'] = 'Yes'
    data.loc[412, 'Spouse'] = 'Yes'
    data.loc[26, 'Spouse'] = 'Yes'
    data.loc[141, 'Spouse'] = 'Yes'
    data.loc[171, 'Spouse'] = 'No'
    data.loc[187, 'Spouse'] = 'Yes'
    data.loc[337, 'Spouse'] = 'Yes'
    data.loc[7, 'Spouse'] = 'Yes'
    data.loc[data['Spouse'].str.contains('none|0'), 'Spouse'] = 'No'
    data.loc[data['Spouse'].str.contains('Yes|No') == False, 'Spouse'] = 'Yes'
   
    data['District'] = data['District'].str.split().str.join(' ')
     
    return data

def clean_district(path):
    """
    Function to clean up the us congress fistricts data

    Parameters
    ----------
    path : str
        path to datfile.

    Returns
    -------
    data : pandas.DataFrame
        cleaned dataframe.

    """
    data = pd.read_csv(path)
    data = data.drop(columns = ['Unnamed: 0'])
    
    states = data['District'].apply(lambda x: x.split('District')[0]) #Remove the word 'Estimate' from first column

    #change the names in the 'District columns
    data['District'] = data['District'].replace(' District', '', regex = True)
    data['District'] = data['District'].replace('\(At Large\)', 'at-large', regex = True)
    data['District'] = data['District'].replace('01', '1', regex = True)
    data['District'] = data['District'].replace('02', '2', regex = True)
    data['District'] = data['District'].replace('03', '3', regex = True)
    data['District'] = data['District'].replace('04', '4', regex = True)
    data['District'] = data['District'].replace('05', '5', regex = True)
    data['District'] = data['District'].replace('06', '6', regex = True)
    data['District'] = data['District'].replace('07', '7', regex = True)
    data['District'] = data['District'].replace('08', '8', regex = True)
    data['District'] = data['District'].replace('09', '9', regex = True)
    data['District'] = data['District'].str.strip() #Remove any trailing whitespace

    
    #calculate different fractions
    data['People: Sex and Age Male'] = data['People: Sex and Age Male'] / data['People: Sex and Age Total population']
    data['People: Sex and Age Female'] = data['People: Sex and Age Female'] / data['People: Sex and Age Total population']
    data['People: Race White'] = data['People: Race White'] / data['People: Race Total population']
    data['People: Race Black or African American'] = data['People: Race Black or African American'] / data['People: Race Total population']
    data['People: Race American Indian and Alaska Native'] = data['People: Race American Indian and Alaska Native'] / data['People: Race Total population']
    data['People: Race Asian'] = data['People: Race Asian'] / data['People: Race Total population']
    data['People: Race Native Hawaiian and Other Pacific Islander'] = data['People: Race Native Hawaiian and Other Pacific Islander'] / data['People: Race Total population']
    data['People: Race Some other race'] = data['People: Race Some other race'] / data['People: Race Total population']
    data['People: Hispanic or Latino and Race Hispanic or Latino (of any race)'] = data['People: Hispanic or Latino and Race Hispanic or Latino (of any race)'] / data['People: Race Total population']
    data['People: Place of Birth Native'] = data['People: Place of Birth Native'] / data['People: Race Total population']
    data['People: Place of Birth Foreign born'] = data['People: Place of Birth Foreign born'] / data['People: Race Total population']
    data['Total Occupation'] = data['Workers: Occupation Management, business, science, and arts occupations'] + data['Workers: Occupation Service occupations'] + data['Workers: Occupation Sales and office occupations'] + data['Workers: Occupation Natural resources, construction, and maintenance occupations'] + data['Workers: Occupation Production, transportation, and material moving occupations']
    data['Workers: Occupation Management, business, science, and arts occupations'] = data['Workers: Occupation Management, business, science, and arts occupations'] / data['Total Occupation']
    data['Workers: Occupation Sales and office occupations'] = data['Workers: Occupation Sales and office occupations'] / data['Total Occupation']
    data['Workers: Occupation Service occupations'] = data['Workers: Occupation Service occupations'] / data['Total Occupation']
    data['Workers: Occupation Natural resources, construction, and maintenance occupations'] = data['Workers: Occupation Natural resources, construction, and maintenance occupations'] / data['Total Occupation']
    data['Workers: Occupation Production, transportation, and material moving occupations'] = data['Workers: Occupation Production, transportation, and material moving occupations'] / data['Total Occupation']
    data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) $200,000 or more'] = data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) $200,000 or more'] / data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Total households']
    data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Less than $10,000'] = data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Less than $10,000'] / data['Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Total households']
    data['Socioeconomic: Percentage of Families and People Whose Income in the Past 12 Months is Below the Poverty Level All people'] = data['Socioeconomic: Percentage of Families and People Whose Income in the Past 12 Months is Below the Poverty Level All people'] / 100
    data['Education: Educational Attainment Percent high school graduate or higher'] = data['Education: Educational Attainment Percent high school graduate or higher'] / 100

    #rename some features
    data = data.rename(columns = {'People: Sex and Age Total population':'Total Population',
                                             'People: Sex and Age Male': 'Percentage Male', 'People: Sex and Age Female': 'Percentage Female',
                                             'People: Sex and Age Median age (years)':'Median age(years)',
                                             'People: Race Total population': 'Race Total population',
                                             'People: Race White':'Fraction White',
                                             'People: Race Black or African American':'Fraction Black or A.American',
                                             'People: Race American Indian and Alaska Native':'Fraction Indian or Alaskan Native',
                                             'People: Race Asian':'Fraction Asian',
                                             'People: Race Native Hawaiian and Other Pacific Islande': 'Fraction Hawaiian and Pacific Islande',
                                             'People: Race Some other race': 'Fraction Other',
                                             'People: Hispanic or Latino and Race Hispanic or Latino (of any race)':'Fraction Hispanic or Latino',
                                             'People: Place of Birth Total population':'Birth Total Population',
                                             'People: Place of Birth Native':'Fraction Native',
                                             'People: Place of Birth Foreign born':'Fraction Foreigner',
                                             'Workers: Occupation Management, business, science, and arts occupations':'Fraction Management, business, science, and arts occupation',
                                             'Workers: Occupation Service occupations':'Fraction Service occupations',
                                             'Workers: Occupation Sales and office occupations':'Fraction Sales and office occupations',
                                             'Workers: Occupation Natural resources, construction, and maintenance occupations':'Fraction Natural resources, construction, and maintenance occupations',
                                             'Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Total households':'Total households',
                                             'Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Less than $10,000':'Fraction of households with income below $10,000',
                                             'Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) $200,000 or more':'Fraction of households with income at $200,000 or more',
                                             'Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Median household income (dollars)':'Median household income',
                                             'Socioeconomic: Income and Benefits (In 2018 inflation-adjusted dollars) Mean household income (dollars)':'Mean household income',
                                             'Socioeconomic: Percentage of Families and People Whose Income in the Past 12 Months is Below the Poverty Level All people':'Fraction of all people below poverty level',
                                             'Education: Educational Attainment Percent high school graduate or higher': 'Fraction attaining at least high school graduation',
                                             'Workers: Occupation Production, transportation, and material moving occupations':'Fraction Production, transportation, and material moving occupations'})

    data['State'] = states
    
    return data

def clean_data(d_path, m_path, save_path, level_of_education):
    """
    Function to merge the two cleaned datasets

    Parameters
    ----------
    d_path : str
        path to congress district data file.
    m_path : str
        path to member data file.
    save_path : str
        location and name of merged and cleaned data.
    level_of_education : str
        level of education attained by members if nothing else is stated.

    Returns
    -------
    None.

    """
    current_folder = os.getcwd()
    district_path = current_folder + d_path
    member_path = current_folder + m_path
    
    #get the two cleaned dataframes
    member_data_clean = clean_member(member_path, level_of_education)
    district_data_clean = clean_district(district_path)
    
    data = pd.merge(member_data_clean, district_data_clean, on = 'District')
    data['State'] = data['State'].str.strip()

    data.to_csv(current_folder + save_path)

    return None

def main():
    district_path = '/data/resultingData/congress_merged.csv'
    member_path = '/data/resultingData/congress_members.csv'
    save_path = '/data/resultingData/merged_data.csv'
    level_of_education = 'Master'
    
    clean_data(district_path, member_path, save_path, level_of_education)
    
    return 0

main()
