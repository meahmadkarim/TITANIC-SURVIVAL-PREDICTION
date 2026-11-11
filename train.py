import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
p=argparse.ArgumentParser(); p.add_argument('--data',default='tested.csv'); p.add_argument('--allow-synthetic-labels',action='store_true'); args=p.parse_args()
df=pd.read_csv(args.data)
if (df.Survived.eq(1)==df.Sex.eq('female')).all():
    if not args.allow_synthetic_labels: raise SystemExit('Labels exactly match Sex=female. This file cannot validate survival prediction. Supply verified training labels, or use --allow-synthetic-labels for a clearly labelled pipeline demonstration.')
    print('SYNTHETIC-LABEL PIPELINE DEMONSTRATION: scores are not evidence of survival prediction.')
numeric=['Age','Fare','SibSp','Parch']; categorical=['Pclass','Sex','Embarked']
X_train,X_test,y_train,y_test=train_test_split(df[numeric+categorical],df.Survived,test_size=.3,random_state=42,stratify=df.Survived)
pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
from sklearn.svm import SVC
model=Pipeline([('preprocess',pre),('model',SVC(kernel='linear'))]).fit(X_train,y_train)
print(classification_report(y_test,model.predict(X_test),zero_division=0))
