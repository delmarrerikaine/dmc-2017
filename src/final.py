#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:48:38 2017

@author: roman
"""

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from func import toCategorical, solveNA, Dummies, solveCategorical, moreFeautures
import os

#chanhe this manualy
path = '/media/roman/Main/Programing/contest/dmc2017/dmc-2017/'
os.chdir(path)

items = pd.read_csv('data/raw/items.csv',sep='|')
train = pd.read_csv('data/raw/train.csv',sep='|')



train_items=pd.merge(train,items,on='pid')
del train
gc.collect
train_items=toCategorical(train_items)


t1=train_items['rrp'].as_matrix()#18.309065356605213
t2=train_items['competitorPrice'].as_matrix()#12.772879581940531
t2_div_t1=t2/t1
t2_div_t1=t2_div_t1[np.logical_not(np.isnan(t2_div_t1))]
coef_competitorPrice_to_rrp=t2_div_t1.mean()
items=solveNA(items,train_items,coef_competitorPrice_to_rrp,1)
train_items=solveNA(train_items,train_items,coef_competitorPrice_to_rrp,0)
train_items=Dummies(train_items)
train_items=moreFeautures(train_items)

#items = solveCategorical('revenue',train_items,items,1)
#items = solveCategorical('click',train_items,items,0)
#items = solveCategorical('basket',train_items,items,0)
#items = solveCategorical('order',train_items,items,0)
#items.to_csv('items_vx')


items = pd.read_csv('data/interim/items_v1.csv')

items_pred=list(items.columns)
t1=['pid']
for p in items_pred:
    if 'revenue' in p:
        t1.append(p)
items_pred=t1
train_items=pd.merge(train_items,items[items_pred],on='pid')


predictors=[ #'lineID',
             #'day',
             #'pid',
             'adFlag',
             #'availability',
             #'competitorPrice',
             #'click',
             #'basket',
             #'order',
             #'price',
             #'revenue',
             #'manufacturer',
             #'group',
             #'content',
             #'unit',
             #'pharmForm',
             'genericProduct',
             #'salesIndex',
             #'category',
             #'campaignIndex',
             'rrp',
             'availability_1',
             'availability_2',
             'availability_3',
             'availability_4',
             'unit_CM',
             'unit_G',
             'unit_KG',
             'unit_L',
             'unit_M',
             'unit_ML',
             'unit_P',
             'unit_ST',
             'salesIndex_40',
             'salesIndex_44',
             'salesIndex_52',
             'salesIndex_53',
             'campaignIndex_A',
             'campaignIndex_B',
             'campaignIndex_C',
             'day_of_week',
             'discount',
             'compDiscount',
             'revenue_group_mean',
             'revenue_group_count',
             'revenue_content_mean',
             'revenue_content_count',
             'revenue_pharmForm_mean',
             'revenue_pharmForm_count',
             'revenue_category_mean',
             'revenue_category_count',
             'revenue_manufacturer_mean',
             'revenue_manufacturer_count',
             'revenue_group_content_mean',
             'revenue_group_content_count',
             'revenue_group_pharmForm_mean',
             'revenue_group_pharmForm_count',
             'revenue_group_category_mean',
             'revenue_group_category_count',
             'revenue_group_manufacturer_mean',
             'revenue_group_manufacturer_count',
             'revenue_content_pharmForm_mean',
             'revenue_content_pharmForm_count',
             'revenue_content_category_mean',
             'revenue_content_category_count',
             'revenue_content_manufacturer_mean',
             'revenue_content_manufacturer_count',
             'revenue_pharmForm_category_mean',
             'revenue_pharmForm_category_count',
             'revenue_pharmForm_manufacturer_mean',
             'revenue_pharmForm_manufacturer_count',
             'revenue_category_manufacturer_mean',
             'revenue_category_manufacturer_count',
             'revenue_group_content_pharmForm_mean',
             'revenue_group_content_pharmForm_count',
             'revenue_group_content_category_mean',
             'revenue_group_content_category_count',
             'revenue_group_content_manufacturer_mean',
             'revenue_group_content_manufacturer_count',
             'revenue_group_pharmForm_category_mean',
             'revenue_group_pharmForm_category_count',
             'revenue_group_pharmForm_manufacturer_mean',
             'revenue_group_pharmForm_manufacturer_count',
             'revenue_group_category_manufacturer_mean',
             'revenue_group_category_manufacturer_count',
             'revenue_content_pharmForm_category_mean',
             'revenue_content_pharmForm_category_count',
             'revenue_content_pharmForm_manufacturer_mean',
             'revenue_content_pharmForm_manufacturer_count',
             'revenue_content_category_manufacturer_mean',
             'revenue_content_category_manufacturer_count',
             'revenue_pharmForm_category_manufacturer_mean',
             'revenue_pharmForm_category_manufacturer_count']
             
y_train = train_items['revenue']
x_train = train_items[predictors]

del train_items
gc.collect

scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)

model_lasso = linear_model.LassoCV(alphas = [1, 0.16, 0.1, 0.001, 0.0005]).fit(x_train, y_train)

model_ridge = linear_model.Ridge(alpha=6, fit_intercept=True, max_iter=10000)
model_ridge.fit(x_train, y_train)

del y_train,x_train
gc.collect

items = pd.read_csv('data/raw/items.csv',sep='|')
clas = pd.read_csv('data/raw/class.csv',sep='|')
clas_items=pd.merge(clas,items,on='pid')
clas_items=toCategorical(clas_items)
clas_items=solveNA(clas_items,clas_items,coef_competitorPrice_to_rrp,0)
clas_items=Dummies(clas_items)
clas_items=moreFeautures(clas_items)
items = pd.read_csv('data/interim/items_v1.csv')
clas_items=pd.merge(clas_items,items[items_pred],on='pid')

submission = pd.DataFrame({
        "lineID": clas_items["lineID"],
        "revenue": np.zeros(shape=(1210767,))
})

x_test = clas_items[predictors]
del clas_items,clas,items
gc.collect

x_test = x_test.fillna(x_test.mean())
x_test = scaler.fit_transform(x_test)

lasso_preds = model_lasso.predict(x_test)
ridge_preds = model_ridge.predict(x_test)

predictions = pd.DataFrame({"ridge":ridge_preds, "lasso":lasso_preds})
predictions[predictions['ridge']<0]=0
predictions[predictions['lasso']<0]=0

lasso_preds = predictions['lasso']
ridge_preds = predictions['ridge']
com_pred = (lasso_preds + ridge_preds) / 2.0

submission['revenue']=com_pred
submission_sorted=submission.sort_values("lineID");
submission_sorted.to_csv(
"data/Uni_Polytechnic_Lviv_1.csv", index=False, sep='|')
