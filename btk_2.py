# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:59:55 2022

@author: MCGirgin
"""

#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#KODLAR
#veri yükleme


veriler = pd.read_csv("eksikveriler.csv")


#---------------------------------------------------------


# #veri on işleme

boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy", "kilo"]]
print(boykilo)


#----------------------------------------------------------


#class tanımı
class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10

ali = insan()
print(ali.boy)
print(ali.kosmak(150))

#liste
l = [1,3,4]


# -----------------------------------------------------------


# Eksik veriler

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:,1:4].values #2 ve 3. kolonu alma

print(Yas)
imputer = imputer.fit(Yas[:,1:4]) #ortalama değer öğrenme
Yas[:,1:4] = imputer.transform(Yas[:,1:4]) #uygulanma
print(Yas)


# -----------------------------------------------------------


# metinsel verileri sayısal verilere çevirme

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


#-----------------------------------------------------------


# geçen iki bölümü birleştirme

sonuc = pd.DataFrame(data=ulke, index = range(22), columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns=["cinsiyet"])
print(sonuc3)

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)


# -----------------------------------------------------------


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


# -----------------------------------------------------------


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print(X_train)
print("")
print(X_test)

# -----------------------------------------------------------


















