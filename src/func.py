#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 00:30:57 2017

@author: roman
"""

import pandas as pd
import numpy as np
import itertools as it

def toCategorical(df):
    """
    This function change object datatype in pandas.DataFrame into category datatype.
    Parameters
    ----------
    df : pandas.DataFrame with train or test DataFrame.
    
    Returns
    -------
    df : pandas.DataFrame with new datatypes.
    
    """
    columns=['availability','group','content','unit','pharmForm',
             'campaignIndex','salesIndex', 'category', 'manufacturer']
    for col in columns:
        if col in df.columns:
            df[col]=df[col].astype('category')
    return df
    
def solveNA(df,df2,coef,flag):
    """
    This function fills some missing data in pandas.DataFrame.
    Parameters
    ----------
    df   : pandas.DataFrame with train or test DataFrame.
    df2  : pandas.DataFrame with train or test DataFrame. (Same as df if it's 
           all table instead just items)
    coef : mean of the relationship between variables competitorPrice and rrp
    flag : signal of which variant of function we use
    Returns
    -------
    df : pandas.DataFrame with solved some NA.
    
    """
    if flag==1:  
        df['pharmForm'] = df['pharmForm'].fillna('no_pharmForm')   
        df['category'] = df['category'].fillna(410)
        df['campaignIndex'] = df['campaignIndex'].fillna('D')
    else:  
        df['competitorPrice'] = df['competitorPrice'].fillna(df2['rrp']*coef)
        if 'pharmForm' in df.columns:
            df['pharmForm'] = df['pharmForm'].cat.add_categories(['no_pharmForm'])
            df['pharmForm'] = df['pharmForm'].fillna('no_pharmForm')
        if 'category' in df.columns:
            df['category'] = df['category'].cat.add_categories([410])
            df['category'] = df['category'].fillna(410)
        if 'campaignIndex' in df.columns:
            df['campaignIndex'] = df['campaignIndex'].cat.add_categories(['D'])
            df['campaignIndex'] = df['campaignIndex'].fillna('D')
    columns2=['category', 'manufacturer']
    for col2 in columns2:
        if col2 in df.columns:
            df[col2]=df[col2].astype('int')
            df[col2]=df[col2].astype('category')
    return df
    
def Dummies(df):
    """
    This function creates new columns in pandas.DataFrame uses existing columns 
    with category datatype as input and pd.get_dummies for creation.
    Parameters
    ----------
    df : pandas.DataFrame with train or test data.
    
    Returns
    -------
    df : pandas.DataFrame with new float64 columns.
    
    """
    columns=['availability','unit','salesIndex','campaignIndex']
    dumm=pd.get_dummies(df[columns])
    df=pd.concat([df, dumm], axis=1)
    return df
    
def solveCategorical(c1,df1,df2,flag):
    """
    This function creates new columns in items DataFrame uses existing columns 
    with category datatype and high cardinality as input and pd.get_dummies for creation.
    Attention high time of execution, because using unefficient tuple
    Parameters
    ----------
    c1   : string with name of column which we try to describe
    df1  : pandas.DataFrame with train data.
    df2  : pandas.DataFrame with items.
    flag : signal that we should generate count of examples which represent our mean
    
    Returns
    -------
    df2 : pandas.DataFrame with new float64 columns.
    
    """
    def f(x):
        tuples = [tuple(i) for i in x.values]
        tuples.reverse()
        res=[]
        for i in xrange(len(tuples)):
            try:
                res.append(t2[tuples.pop()])
            except:
                res.append(np.nan)
        return res
    columns=['group','content','pharmForm','category','manufacturer']
    for L in range(1, 4):
        for col in it.combinations(columns, L):
            print col
            t1=df1.groupby(list(col))
            t2=dict(t1[c1].mean()) 
            str1='_'.join(col)
            if len(col)==1:
                df2[c1+'_'+str1+'_mean']=df2[col[0]].map(t2)
            else:
                df2[c1+'_'+str1+'_mean']=f(df2[list(col)])
            if flag==1:
                t2=dict(t1[c1].count())
                if len(col)==1:
                    df2[c1+'_'+str1+'_count']=df2[col[0]].map(t2)
                else:
                    df2[c1+'_'+str1+'_count']=np.array(f(df2[list(col)]))/2756003.0
    return df2


def moreFeautures(df):
    """
    This function creates new columns in pandas.DataFrame uses existing columns 
    Parameters
    ----------
    df : pandas.DataFrame with train or test data.
    
    Returns
    -------
    df : pandas.DataFrame with new float64 columns.
    
    """
    df['day_of_week']=df['day']%7
    df['discount']=df['price']/df['rrp']
    df['compDiscount']=df['competitorPrice']/df['price']
    return df
    
