import pandas as pd
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



def run_linear_regression(for_fig,name_col0,name_col1):
    X = for_fig.iloc[:, for_fig.columns.get_loc(name_col0)].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = for_fig.iloc[:, for_fig.columns.get_loc(name_col1)].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    print('Regression for:'+str(name_col1))

    print('Coefficients: \n', linear_regressor.coef_)
    # The mean squared error
    print("Mean squared error:"+str(mean_squared_error(Y, Y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score:'+str(r2_score(X, Y_pred)))

    print('R squared:'+str(r2_score(Y, Y_pred)))

    print('linear_regressor.intercept_:'+str(linear_regressor.intercept_))
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()


def assign_to_closest_minor(version):
    version = str(version)
    parts = version.split('.')
    release=parts[0]+'.'+parts[1]
    return  parts[0]+'.'+parts[1]

def trans_to_datetime(dd,columns):
    for col in columns:
        dd[col] = pd.to_datetime(dd[col], utc =True)
        dd[col]=dd[col].dt.tz_localize(None)
    return dd

def is_larger_release(rel1,rel2):
    if rel1 is None or rel2 is None:
        return False
    rel1 = str(rel1)
    rel2 = str(rel2)
    rel1 = tuple(int(i) for i in rel1.split('.'))
    rel2 = tuple(int(i) for i in rel2.split('.'))
    if rel1[0]>rel2[0]:
        return True
    elif rel1[0]<rel2[0]:
        return False

    if rel1[1]>rel2[1]:
        return True
    return False

def is_smaller_release(rel1,rel2):
    if rel1 is None or rel2 is None:
        return False
    rel1 = str(rel1)
    rel2 = str(rel2)
    rel1 = tuple(int(i) for i in rel1.split('.'))
    rel2 = tuple(int(i) for i in rel2.split('.'))
    if rel1[0]<rel2[0]:
        return True
    elif rel1[0]>rel2[0]:
        return False

    if rel1[1]<rel2[1]:
        return True
    return False

import numpy as np
def get_tap(value):
    value = str(value)
    tap = tuple(int(i) for i in value.split('.'))
    return tap

def get_string(value):
    return str(value[0])+'.'+str(value[1])

def sort_df(df, column_idx):
    '''Takes dataframe, column index and custom function for sorting,
    returns dataframe sorted by this column using this function'''

    col = df.loc[:,column_idx]
    temp = np.array(col.values.tolist())
    values = col.values.tolist()
    values = [get_tap(x) for x in values]
    values = sorted(values)
    values = [get_string(x) for x in values]
    order = values
    df = df.set_index(column_idx)
    df = df.loc[order]
    df = df.reset_index()
    return df

def fetch_minimal_columns(df):
    df = df.rename(index=str,columns={'product':'Product'})
    df = df[['id','Product','version','resolution','status','severity','creation_time','priority']]
    df['creation_time'] = pd.to_datetime(df['creation_time'])
    return df
def fetch_maximal_columns(df):
    df = df.rename(index=str,columns={'product':'Product'})
    df = df[['id','Product','version','resolution','status','severity','creation_time','priority','target_milestone','deadline']]
    df['creation_time'] = pd.to_datetime(df['creation_time'])
    return df



def get_release_dates():
    relase_creation_ts_all = {
        "3.0" : "2004-06-25 00:00:00",
        "3.1" : "2005-06-28 00:00:00",
        "3.2" : "2006-06-29 00:00:00", #Callisto
        "3.3" : "2007-06-28 00:00:00", #Europa
        "3.4" : "2008-06-25 00:00:00", #Ganymede
        "3.5" : "2009-06-24 00:00:00", #Galileo
        "3.6" : "2010-06-23 00:00:00", #Helios
        "3.7" : "2011-06-22 00:00:00", #Indigo
        "3.8" : "2012-06-27 00:00:00", #
    #    "4.1" : "2011-06-22",
        "4.2" : "2012-06-27 00:00:00", #Juno
        "4.3" : "2013-06-26 20:00:00", #Kepler
        "4.4" : "2014-06-25 12:15:00", #Luna
        "4.5" : "2015-06-24 20:00:00", #Mars
        "4.6" : "2016-06-22 11:00:00", #Neon
        "4.7" : "2017-06-28 09:50:00", #Oxygen
        "4.8" : "2018-06-27 00:00:00", #Photon
        "4.9"  : "2018-09-19 00:00:00",
        "4.10" : "2018-12-19 00:00:00",#,
       "4.11" : "2019-03-20 00:00:00",
       "4.12" : "2019-06-19 00:00:00",
        "4.13" : "2019-09-18 00:00:00",
        "4.14" : "2019-12-18 00:00:00",
          "4.15" : "2020-03-18 00:00:00"
       # "4.16" : "2020-06-17 00:00:00"     
    }
    return relase_creation_ts_all
def get_release_dates_edited():
    relase_creation_ts_all = {
        "2.0" : "2003-06-25 00:00:00",
        "3.0" : "2004-06-25 00:00:00",
        "3.1" : "2005-06-28 00:00:00",
        "3.2" : "2006-06-29 00:00:00", #Callisto
        "3.3" : "2007-06-28 00:00:00", #Europa
        "3.4" : "2008-06-25 00:00:00", #Ganymede
        "3.5" : "2009-06-24 00:00:00", #Galileo
        "3.6" : "2010-06-23 00:00:00", #Helios
        "3.7" : "2011-06-22 00:00:00", #Indigo
        "3.8" : "2012-06-27 00:00:00", #
    #    "4.1" : "2011-06-22",
        "4.2" : "2012-06-27 00:00:00", #Juno
        "4.3" : "2013-06-26 20:00:00", #Kepler
        "4.4" : "2014-06-25 12:15:00", #Luna
        "4.5" : "2015-06-24 20:00:00", #Mars
        "4.6" : "2016-06-22 11:00:00", #Neon
        "4.7" : "2017-06-28 09:50:00", #Oxygen
        "4.8" : "2018-06-27 00:00:00", #Photon
        "4.9"  : "2018-09-19 00:00:00",
        "4.10" : "2018-12-19 00:00:00",#,
       "4.11" : "2019-03-20 00:00:00",
       "4.12" : "2019-06-19 00:00:00",
        "4.13" : "2019-09-18 00:00:00",
        "4.14" : "2019-12-18 00:00:00",
         "4.15" : "2020-03-18 00:00:00",
        "4.16" : "2020-06-17 00:00:00"
    }
    return relase_creation_ts_all
def get_yearly_releases():
    yearly_releases = ["3.0",
    "3.1",
    "3.2", #Callisto
    "3.3", #Europa
    "3.4", #Ganymede
    "3.5", #Galileo
    "3.6", #Helios
    "3.7", #Indigo
    "4.2", #Juno

    "4.3", #Kepler
    "4.4", #Luna
    "4.5", #Mars
    "4.6", #Neon
    "4.7", #Oxygen
    "4.8" #Photon
    ]
    return yearly_releases

def get_rolling_releases():
    relase_creation_ts_rolling = ["4.9","4.10","4.11","4.12","4.13","4.14","4.15"]
    return relase_creation_ts_rolling

def addYears(d, years):
    try:
        #Return same day of the current year
        return d.replace(year = d.year + years)
    except ValueError:
        #If not same day, it will return other, i.e.  February 29 to March 1 etc.
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

def get_x_axis_tick_placement():
    places = [0,1,2,3,4,5,6,20/3,22/3,24/3,26/3,28/3,10,32/3]
    return places
def get_all_x_axis_tick_placement():
    places = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,47/3,49/3,51/3,53/3,55/3,57/3,59/3]
    return places
def successor(yearly_releases,release):
    if release=='3.8':
        return '4.2'
    return yearly_releases[yearly_releases.index(release)+1]

def predecessor(yearly_releases,release):
    if release=='4.2':
        return '3.7'
    return str(yearly_releases[yearly_releases.index(release)-1])


def attach_severity_priority_to_dataframe(df):
    sev_info = pd.read_csv('.'+os.sep+'data'+os.sep+'bugs_full_all.csv',index_col=False,
                      dtype={'version':str})
    sev_info = sev_info[['id','severity','priority']]

    df = pd.merge(df,sev_info,on=['id'],how='left')
    return df

