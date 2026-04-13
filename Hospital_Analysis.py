import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_curve,auc

#-------------------------------------------------------------------------------------------------
#data importing
#-------------------------------------------------------------------------------------------------
df = pd.read_csv('csv/hospital data analysis.csv')
print(df.head(5))

#-------------------------------------------------------------------------------------------------
#data cleaning
#-------------------------------------------------------------------------------------------------
print(df.isnull().sum())
print(df.nunique())
print(df.describe())
print(df.info())

#------------------------------------------------------------------------------------------------- 
#data visualization
#-------------------------------------------------------------------------------------------------
sb.scatterplot(x='Age',y='Length_of_Stay',hue='Gender',data=df)
plt.title('age vs Length of Stay')
plt.show()

count = df['Age'].value_counts().plot(kind='line')
plt.show()

#patient condition on cost
plt.bar('Condition','Cost',data=df)
plt.show()

Redm= df['Readmission'].value_counts().plot(kind='bar')
plt.show()

df['Readmission'] = df['Readmission'].str.lower().map({'yes':1,'no':0})
df['Outcome'] = df['Outcome'].str.lower().map({'Recovered':1,'Stable':0})

grpby = df.groupby('Procedure')['Satisfaction'].mean().plot(kind='barh')
plt.show()

grpby = df.groupby('Condition')['Length_of_Stay'].mean().plot(kind='barh')
plt.show()

grpby = df.groupby('Procedure')['Cost'].sum().plot(kind='barh')
plt.show()

sb.pairplot(df,vars=['Satisfaction','Length_of_Stay','Cost','Age'],hue='Gender')
plt.show()

corr = df[['Satisfaction','Length_of_Stay','Cost','Age']].corr()
sb.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()

rt = df['Condition'].unique()
print(rt)

tu = df['Procedure'].unique()
print(tu)
#-------------------------------------------------------------------------------------------------
#logistic regression
#-------------------------------------------------------------------------------------------------

LE = LabelEncoder()
df['Gender']   = LE.fit_transform(df['Gender'])
df['Condition']=LE.fit_transform(df['Condition'])
df['Procedure']=LE.fit_transform(df['Procedure'])
df['Readmission']=LE.fit_transform(df['Readmission'])
df['Outcome']=LE.fit_transform(df['Outcome'])


#creating Variable to train
X = df.drop(['Patient_ID'],axis=1)
y = df['Readmission']

print(X,y)

# model Train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
print('train',X_train)
X_test = scaler.transform(X_test)
print('test',X_test)

#build logistic regression model
model =LogisticRegression(max_iter=983)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

accuracy =accuracy_score(y_test,y_pred)
print('accuracy_score::>',accuracy)

cm = confusion_matrix(y_test,y_pred) 
sb.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual data')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test,y_pred))

fpr,tpr,threshold = roc_curve(y_test,y_prob)

roc_curve = auc(fpr,tpr)
print(tpr,fpr)

coefficients = pd.DataFrame({
    'Feature':X.columns,'coefficient':model.coef_[0]
})

new_columns = pd.DataFrame({
    'Age':[45],'Gender':[1],'Condition':[6],'Procedure':[10],'Cost':[12000],'Length_of_Stay':[20],'Readmission':[1],'Outcome':[0],
    'Satisfaction':[3]
})

new_Redmission_scale = scaler.transform(new_columns)
print(new_Redmission_scale)
 
predication = model.predict(new_Redmission_scale)
probability = model.predict_proba(new_Redmission_scale)

print('predicted class::',predication[0])
print('Redmission',probability[0][1])