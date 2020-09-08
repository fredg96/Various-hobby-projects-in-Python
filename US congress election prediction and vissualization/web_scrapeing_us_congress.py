# -*- coding: utf-8 -*-
"""
Created on Thursday Jul 23 2020

@author: fredg96

This program is the first part of a project about classifying how  us congress districts will vote.
This file scrapes a wikipedia page for data about each us congress district and from it extracts 
personal data, from wikipedia, about each congress member. The code in here is als presented in the
notebook scrapeData.ipynb 
"""
#For requesting the webpages which we will srape
import requests
from bs4 import BeautifulSoup as bs 

#To fetch data in parallel
import multiprocessing as mp
from joblib import Parallel, delayed 

#To create datasets
import pandas as pd
import numpy as np

#Other
import time 
import os
import sys


def get_table(url, type, nr):
    """
    Get specific table type and number from URL
    
    input: url, webpage address. type, type of table e.g. "wikitable sortable". nr, what number does the specifi table type have.
    output: returns the specific table as a BeautifulSoup object
    """
    response = requests.get(url) #Requested webpage
    soup = bs(response.content, 'lxml') #Parsed webpage 
    table = soup.find_all("table", class_ = type)[nr] #the nr:th specific tabletype in parsed format as a bs object 
    return table

def get_links(table, nr_col):
    """
    Extracts links from the specific BeautifulSoup table
    
    input: table, BeautifulSoup table containing the links. nr_col, number of columns in table to know how long each row is.
    output: extracted_links, the links as a list
    """
    extracted_links = []
    for row in table.findAll('tr'):
        cells = row.findAll('td')
        if len(cells) == nr_col:
            links = cells[1].findAll('a')
            if links != []:
                link = links[1].get('href')
                extracted_links.append('https://en.wikipedia.org' + link)
            else:
                continue
                
    return extracted_links
            
def extract_wikibox_data(url, politeness):
    """
    Get data from a single wikibox 
    
    input: url, URL to the webpage with wikibox. politeness, parameter to determine how long to wait between requests.
    output: name, congress member name. spouse, name of congress member spouse(if any). children, number or name of children/s(if any).
    """
    cname = " "
    bname = " "
    spouse = "none"
    children = " "
    response = requests.get(url, params={'action': 'raw'})
    page = response.text
    
    for line in page.splitlines():
        #Different sections where the name might be found
        if line.startswith('| birth_name'):
            bname = line.partition('=')[-1].strip()
        elif line.startswith('|birth_name'):
            bname = line.partition('=')[-1].strip()
        elif line.startswith('|name'):
            cname = line.partition('=')[-1].strip()
        elif line.startswith('| name'):
            cname = line.partition('=')[-1].strip()
        elif line.startswith('|Name'):
            cname = line.partition('=')[-1].strip()
        elif line.startswith('| Name'):
            cname = line.partition('=')[-1].strip()
        #Spouse are most likelt found under spouse or Spouse
        elif line.startswith('|spouse'):
            spouse = line.partition('=')[-1].strip()
        elif line.startswith('|Spouse'):
            spouse = line.partition('=')[-1].strip()
        elif line.startswith('| Spouse'):
            spouse = line.partition('=')[-1].strip()
        elif line.startswith('| spouse'):
            spouse = line.partition('=')[-1].strip()
        #Different names where the number of childrens might be found
        elif line.startswith('| children'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('| Children'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('|children'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('|Children'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('|Childrens'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('| Childrens'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('| childrens'):
            children = line.partition('=')[-1].strip()
        elif line.startswith('| childrens'):
            children = line.partition('=')[-1].strip()
        #Website URL is most likely the last thing so we can stop looking then
        elif line.startswith('|website'):  
            break 
        elif line.startswith('| website'):  
            break
        if cname != " ": #We will prefere their called name which should correspond better between tables
            name = cname
        elif bname != " ": #If we only find their birth name we will use that instead to make manual pairing easier when cleaning data
            name = bname 
        else: #If we do not find any name we will fill it in as blank
            name = " "
            
        time.sleep(politeness) #Wait politeness seconds before next request
        
    return name, spouse, children
        
def extract_wikibox(links, politeness, parallel):
    """
    Function to allow us to either extract the personal data in serial or parallel
    
    input: links, list of URLs. politeness, time to wait between requests. parallel, if doing it in parallel or not.
    output: names, list (numpy array if done in parallel) of congress members names. spouses, list(numpy array if done in paralle) of spouses.
            childrens, list(numpy array if done in paralle) of number of childrens
    """
    if parallel == 0:
        names = [] #List to keep the names used as keys.
        spouses = [] #List to keep name of spouses
        childrens = [] #List for number or names of childrens
        for link in links:
            name, spouse, children = extract_wikibox_data(link, politeness)
            names.append(name)
            spouses.append(spouse)
            childrens.append(children)
            
    elif parallel == 1:
        num_cores = mp.cpu_count() #Use all cores
        input = links 
        if __name__ == "__main__": #Has to be used in windows to not execture main module on each child
                results = np.array(Parallel(n_jobs=num_cores)(delayed(extract_wikibox_data)(link, politeness) for link in input))
                #Split the result to three arrays
                names = results[:,0]
                spouses = results[:,1]
                childrens = results[:,2]  
    else:
        print('WARNING: parallel must be 0 or 1, exiting')
        sys.exit()
                        
    return names, spouses, childrens

def extract_data(save_dir, url, table_type, table_nr, nr_col, politeness, parallel):
    """
    Pipeline to extract data from wikipedia page

    Parameters
    ----------
    save_dir : str
        directory and file name to save data to
    url : str
        URL to webpage
    table_type : str
        type of table to scrape
    table_nr : int
        table types number e.g. 1 or 2 if we want second or third table of type table type
    nr_col : int
        number of columns in table
    politeness : int
        seconds to wait between each request to webpage
    parallel : int
        if doing scrapeing in parallel, 1, or not, 0.

    Returns
    -------
    None.

    """
    
    member_table = get_table(url, table_type, table_nr)
    member_links = get_links(member_table, nr_col)
    member_name, member_spouse, member_childrens = extract_wikibox(member_links, politeness, parallel)
    
    member_personal_data = pd.DataFrame(member_name, columns = ['Member'])
    member_personal_data['Spouse'] = member_spouse
    member_personal_data['Childrens'] = member_childrens

    member_frame = pd.read_html(url)[6] #Index specifies which table to put into a fram
    
    result = pd.merge(member_frame, member_personal_data, how = 'outer', on = 'Member')
    result.to_csv(os.getcwd() + save_dir) #Print the results to a csv file.

    return None

def main():
    """
    Main function to call extract_data() with chosen parameters.

    Returns
    -------
    None.

    """
    
    url = "https://en.wikipedia.org/wiki/List_of_current_members_of_the_United_States_House_of_Representatives" #Url to wikipedia page
    table_type = 'wikitable sortable' #type of table to scrape
    table_nr = 2 #The table in question is the 2nd sortable
    nr_col = 9 #Table has 9 columns so this helps to find correct rows
    
    politeness = 0.0 #seconds to wait before each request to Wikipedia
    parallel = 1 #extract personal information about members in parallel or not has good parallel scaling
    
    save_dir = '/data/resultingData/congress_members.csv' #folder to save data in
    extract_data(save_dir, url, table_type, table_nr, nr_col, politeness, parallel)
    
    return 0

main()
