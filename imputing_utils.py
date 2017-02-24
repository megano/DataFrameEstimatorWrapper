# import common_pandas_broadcast_functions as cpbf
import math as m
import pandas as pd
from enum import Enum
import numpy as np


def delete_rows_with_value(df, value = [None]) :
    df = df[~df.isin(value).any(axis = 1)]
    return df

def delete_columns_with_value(df, value = [None]) :
    df = df[df.columns[~df.isin(value).any(axis = 0)]]
    return df

##########################

class Replace(Enum):
     MEAN = 1
     MODE = 2
     VALUE = 3

#dependent : enum Replace
def replace_given_values(ser, replaceVal, value = [None, float('nan')]) :
    ser = ser.replace(value, np.nan)
    if(type(replaceVal) == Replace) :
        if(replaceVal == Replace.MEAN) :
            ser = ser.fillna(ser.mean(skipna = True, axis = 0))
        elif(replaceVal == Replace.MODE) :
            mode =  ser.mode()
            if mode.empty == False:
                if (type(mode[0]) not in [float, int]) or (m.isnan(mode[0]) == False):
                    ser = ser.fillna(mode[0])
                else:
                    ser = ser.fillna(0.0)
            else :
                ser = ser.fillna(ser.mean(skipna = True, axis = 0))
    elif (type(replaceVal) in (tuple, list)): #Problem with python :( one extra tab this becomes the else for another (Wrong) if condition
        ser = ser.fillna(dict(enumerate(replaceVal)))
    else : #Problem with python :( one extra tab this becomes the else for another (Wrong) if condition
        ser = ser.fillna(replaceVal)
    return ser

#dependent : enum Replace
#dependent : replace_given_values
def replace_given_values_With_Mean(df, value = [None, float('nan')]) :
    return replace_given_values(df, Replace.MEAN, value=value)

#dependent : enum Replace
#dependent : replace_given_values
def replace_given_values_With_Mode(df, value = [None, float('nan')]) :
    return replace_given_values(df, Replace.MODE, value=value)

##########################

#fill na for a listof na values while loading data frame'''
''' na_values = ['NO CLUE', 'N/A', '0'] requests = pd.read_csv('../data/311-service-requests.csv', na_values=na_values, dtype={'Incident Zip': str})'''
#Delete rows with na values
'''df = df.dropna(axis = 0)'''
#Delete rows with na values
'''df = df.dropna(axis = 1)'''
#map a dictionary
'''Series.map(arg, na_action=None)'''

'''
class
    df
    mode_columns
    mean_columns
    ordinal_columns
    categorical columns
    file_name
'''

class impute_class :

    def __init__(self, df) :
        self.df = df

        #set of columns to treat as categorical or ordinal
        self.categ_columns = 'All'
        self.ordinal_columns = None

        #coluns to impute with mean/mode
        self.mean_column = None
        self.mode_column = 'All'

        self.save_file = False
        self.save_file_name = ""

        #if None then replace only nan values
        self.dic_to_impute_with_Mean = None
        self.dic_to_impute_with_Mode = None

    #mean_column : list
    def set_columns_to_impute_with_Mean(self, mean_column, dic_to_impute = None) :
        self.mean_column = mean_column
        self.dic_to_impute_with_Mean = dic_to_impute

    #mode_column : list
    def set_columns_to_impute_with_Mode(self, mode_column, dic_to_impute = None) :
        self.mode_column = mode_column
        self.dic_to_impute_with_Mode = dic_to_impute
    #columns : list
    def set_columns_as_categ(self, columns) :
        self.categ_columns = columns

    #columns : list
    def set_columns_as_ordinal(self, columns, ord_col_map) :
        self.ordinal_columns = columns
        self.ord_col_map = ord_col_map

    #file name to save
    def set_file_name_to_save(self, file_path) :
        self.save_file = True
        self.save_file_name = file_path

    def _impute(self) :
        self._impute_mean()
        self._impute_mode()

    def _impute_mean(self) :
        #if columns is all, then impute with mean all columns except for the mode columns
        if (self.mean_column) and (type(self.mean_column) == str) and (self.mean_column == 'All') :
            columns_to_impute_with_mean = [col for col in self.df.columns if (not self.mode_column) or (col not in self.mode_column)]
        else :
            columns_to_impute_with_mean = self.mean_column

        if(columns_to_impute_with_mean) :
            print "Impute Means"
            for col in columns_to_impute_with_mean :
                emp_val = self.dic_to_impute_with_Mean[col] if (self.dic_to_impute_with_Mean) and (col in self.dic_to_impute_with_Mean) else [None]
                self.df[col+"_imp"] = self.df[col].map(lambda x : 1 if ((not x) | m.isnan(x) | (x in emp_val)) else 0)
                self.df[col] = replace_given_values_With_Mean(self.df[col] ,
                    value = emp_val)



    def _impute_mode(self) :
        #if columns is all, then impute with mean all columns except for the mode columns
        cond = (self.mode_column)
        cond = (type(self.mode_column) == str) if cond else False

        cond = (self.mode_column == 'All') if cond == True else False
        if cond == True :
            columns_to_impute_with_mode = [col for col in self.df.columns if (not self.mean_column) or (col not in self.mean_column)]
        else :
            columns_to_impute_with_mode = self.mode_column

        #replace given values with mode
        if(columns_to_impute_with_mode) :
            tot_len = len(columns_to_impute_with_mode)
            i = 1
            for col in columns_to_impute_with_mode :
                print "%s - %d out of %d columns" %(col, i, tot_len)
                i += 1
                self.df[col] = replace_given_values_With_Mode(self.df[col] ,
                    value = self.dic_to_impute_with_Mode[col] if (self.dic_to_impute_with_Mode) and (col in self.dic_to_impute_with_Mode) else [None])

    def _ordinal_manipulation(self) :
        if(self.ordinal_columns) :
            print "Ordinal manipulation"
            for col in self.ordinal_columns:
                self.df[col] = self.df[col].map(lambda x : x.lower() if (type(x) == str) else x)
                self.df[col] = self.df[col].map(lambda x : self.ord_col_map[x] if x in self.ord_col_map else float('nan'))

    def _dummy_columns(self):
        if(self.categ_columns) :
            print "category manipulation"
            self.df = pd.merge(self.df,pd.get_dummies(self.df[self.categ_columns], drop_first=False), right_index = True, left_index = True)
            self.df.drop(self.categ_columns, inplace=True, axis = 1)


    #TODO normalize data
    #TODO columns to deleted if any

    def run(self):
        self._ordinal_manipulation()

        #impute
        print "Start : Impute"
        self._impute()
        print "End : Impute"

        #categ things
        self._dummy_columns()
        #delete columns whose all values are na
        self.df.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)

        if(self.save_file == True) :
            self.df.to_csv(self.save_file_name)
        print "return"
        return self.df
