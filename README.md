[TOC]

# Desórdenes del sueño, ocupaciones y actividad física.

En el Notebook denominado **Analisis.ipynb** se presenta un análisis de los datos para un dataset existente tomado de kaggle, el cual esta alojado en la siguiente URL:

https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
Licencia: CC0: Public Domain

Dicho Dataset, esta relacionado con la temática de estilo de vida y hábitos saludables; brinda información de un grupo de individuos discriminado su ocupación, algunas métricas de salud y actividad física.


#### Descripción de las variables

* Person ID: Es un identificador único para cada persona.
* Gender: Hace referencia al género (Femenino/Masculino).
* Age: Es la edad de cada persona.
* Occupation: Profesión u ocupación de la persona.
* Sleep Duration: Corresponde al número de horas que la persona duerme.
* Quality of Sleep: Se refiere a la calidad del sueño de cada persona, teniendo en cuenta una escala de 1-10.
* Physical Activity Level: Se refiere a los minutos que cada persona realiza actividad física.
* Stress Level: Representa el nivel de estrés que experimenta cada persona en una escala de 1-10.
* BMI Category: Hace referencia a la clasificación según su indice de masa corporal.
* Blood Pressure (systolic/diastolic): Es la presión sanguinea.
* Heart Rate (bpm): Son las pulsaciones por minuto, frecuencia cardiaca.
* Daily Steps: Número de pasos que una persona realiza a diario.
* Sleep Disorder: Son los desórdenes del sueño tales como Apnea del Sueño e insomnio. En su gran mayoria no se registran desórdenes.

#### Librerias Utilizadas

+  Librerías para el análisis y la visualización de los datos
   + import numpy as np 
   + import pandas as pd
   + import matplotlib.pyplot as plt
   + %matplotlib inline
   + import seaborn as sns
   + import altair as alt
   + from ydata_profiling import ProfileReport

+  Librerias Feature Engineering.
   + from sklearn.model_selection import train_test_split
   
+ Librerias Preprocesamiento, pipeline y transformacion.
   + from sklearn.preprocessing import OneHotEncoder
   + from sklearn.preprocessing import LabelEncoder
   + from sklearn.preprocessing import MaxAbsScaler
   + from sklearn.preprocessing import OrdinalEncoder
   + from sklearn.compose import ColumnTransformer
   + from sklearn.pipeline import FeatureUnion, Pipeline

+ Librerias model training.
   + from sklearn.base import clone

+ Librerias utilizadas para emplear Algoritmos de ML y joblib.
   + from sklearn.ensemble import RandomForestClassifier
   + from sklearn.linear_model import LogisticRegression
   + from sklearn.svm import SVC
   + from sklearn.tree import DecisionTreeClassifier
   + from sklearn.neighbors import KNeighborsClassifier
   + from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
   + from joblib import dump
   
#### Archivos Generados 
* df_clean.csv
* inference_pipeline.joblib
