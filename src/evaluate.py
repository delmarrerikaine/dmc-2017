#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 22:38:55 2017

@author: roman
"""

import pandas as pd
import numpy as np
import os

#chanhe this manualy
path = '/media/roman/Main/Programing/contest/dmc2017/dmc-2017/'
os.chdir(path)

mysubm = pd.read_csv('data/Uni_Polytechnic_Lviv_1.csv',sep='|')
realsubm = pd.read_csv('data/raw/realclass.csv',sep='|')


def mse(actual, predicted):
    return np.sqrt(((np.array(actual) - np.array(predicted))
    *(np.array(actual) - np.array(predicted))).sum())
    
print mse(realsubm['revenue'].values,mysubm['revenue'].values)
print mse(realsubm['revenue'].values,np.zeros((1210767,)))
