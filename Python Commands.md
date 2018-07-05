

```python
#Include necessary packages
from sklearn import datasets


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

```


```python
# Load the dataset
data=pd.read_csv("creditcard.csv")
print(data.columns)
```

    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')
    


```python
#Getting familier to the data
print(data.shape)
```

    (284807, 31)
    


```python
print(data.head())
print(data.describe())
```

       Time        V1        V2        V3        V4        V5        V6        V7  \
    0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
    1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
    2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
    3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
    4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   
    
             V8        V9  ...         V21       V22       V23       V24  \
    0  0.098698  0.363787  ...   -0.018307  0.277838 -0.110474  0.066928   
    1  0.085102 -0.255425  ...   -0.225775 -0.638672  0.101288 -0.339846   
    2  0.247676 -1.514654  ...    0.247998  0.771679  0.909412 -0.689281   
    3  0.377436 -1.387024  ...   -0.108300  0.005274 -0.190321 -1.175575   
    4 -0.270533  0.817739  ...   -0.009431  0.798278 -0.137458  0.141267   
    
            V25       V26       V27       V28  Amount  Class  
    0  0.128539 -0.189115  0.133558 -0.021053  149.62      0  
    1  0.167170  0.125895 -0.008983  0.014724    2.69      0  
    2 -0.327642 -0.139097 -0.055353 -0.059752  378.66      0  
    3  0.647376 -0.221929  0.062723  0.061458  123.50      0  
    4 -0.206010  0.502292  0.219422  0.215153   69.99      0  
    
    [5 rows x 31 columns]
                    Time            V1            V2            V3            V4  \
    count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean    94813.859575  3.919560e-15  5.688174e-16 -8.769071e-15  2.782312e-15   
    std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   
    min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   
    25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   
    50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   
    75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   
    max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   
    
                     V5            V6            V7            V8            V9  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean  -1.552563e-15  2.010663e-15 -1.694249e-15 -1.927028e-16 -3.137024e-15   
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   
    
               ...                 V21           V22           V23           V24  \
    count      ...        2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean       ...        1.537294e-16  7.959909e-16  5.367590e-16  4.458112e-15   
    std        ...        7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   
    min        ...       -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   
    25%        ...       -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   
    50%        ...       -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   
    75%        ...        1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   
    max        ...        2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   
    
                    V25           V26           V27           V28         Amount  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   
    mean   1.453003e-15  1.699104e-15 -3.660161e-16 -1.206049e-16      88.349619   
    std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   
    min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   
    25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   
    50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   
    75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   
    max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   
    
                   Class  
    count  284807.000000  
    mean        0.001727  
    std         0.041527  
    min         0.000000  
    25%         0.000000  
    50%         0.000000  
    75%         0.000000  
    max         1.000000  
    
    [8 rows x 31 columns]
    


```python
#Determining null values
data.isnull().sum(axis=0)
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
# Treating missing values with mean
# data = data.groupby(data.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
# Not needed here
```


```python
#Plot histogram of each parameter
data.hist(figsize=(20,20))
plt.show()
```


![png](output_6_0.png)



```python
#Determin number of fraud cases in dataset
Fraud=data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud Cases: {}'. format(len(Fraud)))
print('Valid Cases: {}'. format(len(Valid)))
```

    0.0017304750013189597
    Fraud Cases: 492
    Valid Cases: 284315
    


```python
# Correlation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x197681ed198>




![png](output_8_1.png)



```python
# Resampling - Underscore
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix, classification_report

# Get all the coulmns frm the dataframes
columns=data.columns.tolist()

#Filter the columns to remove data we do not want
columns=[c for c in columns if c not in ["Class"]]

#Store the variable we will be prediction on
target="Class"


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)


# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud*19, replace = False)

random_normal_indices = np.array(random_normal_indices)


# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

 # Get all the coulmns frm the dataframes
    #Creading Columns
X=under_sample_data[columns]
Y=under_sample_data[target]

X_train_data,X_test_data,Y_train_data,Y_test_data = train_test_split(X,Y,test_size=0.3,random_state=8)
    
```


```python
#Standardization
sc=StandardScaler()
sc.fit(X_train_data)
X_train_std=sc.transform(X_train_data)
X_test_std=sc.transform(X_test_data)
```


```python
print(X_test_std)
```

    [[ 0.915104   -0.82529022  0.15132653 ... -0.64043155  0.38206136
      -0.36776407]
     [ 0.67709012 -0.01645393  0.37765914 ... -0.09218086 -0.27855947
      -0.25770831]
     [-1.09162567 -0.03958797  0.5311543  ... -0.14418883 -0.04980606
      -0.36370938]
     ...
     [ 1.26490304 -0.44260358 -0.1266738  ... -0.71801585  0.03725984
      -0.36776407]
     [-0.9126685  -0.28943486 -0.62504862 ... -1.17578892 -1.08798272
       0.15702811]
     [-0.49716914  0.50309614  0.07100962 ...  0.05712547  0.09253052
      -0.26867251]]
    


```python
# Logistic Regression - Sigmoid function


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 



    
for i in (.01, 0.1,1,10,100,1000):
    for j in (0.025,0.05,0.075,0.1,0.125,0.150,0.175,0.2):
        lr= LogisticRegression (C=i, random_state=0)
        lr.fit(X_train_std,Y_train_data)
        probs=(lr.predict_proba(X_test_std)[:,1])
        Y_pred=(lr.predict(X_test_std))
        probs2=(lr.predict_proba(X_test_std)[:,1].reshape(1,-1))

        Y_pred_new=binarize(probs2, j)[0]

        print("======================")

        print("Event Rate=0.05","    ;  C Parameter=",i,"   ;  Classification Probability=",j)
        print("")
        print(confusion_matrix(Y_test_data,Y_pred_new))
        print(classification_report(Y_test_data,Y_pred_new))
         #print(metrics.roc_auc_score(Y_test_data,Y_pred_new))    

    print("======================")
    print("")
    print("Event Rate=0.05","    ;  C Parameter=",i,"   ;  Default Classification Probability)=",0.5)
    print("")
    print(confusion_matrix(Y_test_data,Y_pred))
    print(classification_report(Y_test_data,Y_pred))

    print("**********************")

print("..............................")
 

```

    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.025
    
    [[  54 2743]
     [   0  155]]
                 precision    recall  f1-score   support
    
              0       1.00      0.02      0.04      2797
              1       0.05      1.00      0.10       155
    
    avg / total       0.95      0.07      0.04      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.05
    
    [[1999  798]
     [   6  149]]
                 precision    recall  f1-score   support
    
              0       1.00      0.71      0.83      2797
              1       0.16      0.96      0.27       155
    
    avg / total       0.95      0.73      0.80      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.075
    
    [[2647  150]
     [  11  144]]
                 precision    recall  f1-score   support
    
              0       1.00      0.95      0.97      2797
              1       0.49      0.93      0.64       155
    
    avg / total       0.97      0.95      0.95      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.1
    
    [[2739   58]
     [  15  140]]
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.99      2797
              1       0.71      0.90      0.79       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.125
    
    [[2763   34]
     [  15  140]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.80      0.90      0.85       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.15
    
    [[2773   24]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.85      0.89      0.87       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.175
    
    [[2779   18]
     [  18  137]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.88      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Classification Probability= 0.2
    
    [[2785   12]
     [  20  135]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      2797
              1       0.92      0.87      0.89       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 0.01    ;  Default Classification Probability)= 0.5
    
    [[2796    1]
     [  26  129]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.99      0.83      0.91       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.025
    
    [[2539  258]
     [   7  148]]
                 precision    recall  f1-score   support
    
              0       1.00      0.91      0.95      2797
              1       0.36      0.95      0.53       155
    
    avg / total       0.96      0.91      0.93      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.05
    
    [[2714   83]
     [  13  142]]
                 precision    recall  f1-score   support
    
              0       1.00      0.97      0.98      2797
              1       0.63      0.92      0.75       155
    
    avg / total       0.98      0.97      0.97      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.075
    
    [[2758   39]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.78      0.91      0.84       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.1
    
    [[2770   27]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.84      0.91      0.87       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.125
    
    [[2775   22]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.87      0.91      0.89       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.15
    
    [[2778   19]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.90      0.89       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.175
    
    [[2783   14]
     [  18  137]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.91      0.88      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Classification Probability= 0.2
    
    [[2785   12]
     [  19  136]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      2797
              1       0.92      0.88      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 0.1    ;  Default Classification Probability)= 0.5
    
    [[2796    1]
     [  26  129]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.99      0.83      0.91       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.025
    
    [[2626  171]
     [  10  145]]
                 precision    recall  f1-score   support
    
              0       1.00      0.94      0.97      2797
              1       0.46      0.94      0.62       155
    
    avg / total       0.97      0.94      0.95      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.05
    
    [[2730   67]
     [  13  142]]
                 precision    recall  f1-score   support
    
              0       1.00      0.98      0.99      2797
              1       0.68      0.92      0.78       155
    
    avg / total       0.98      0.97      0.97      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.075
    
    [[2754   43]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.99      2797
              1       0.77      0.91      0.83       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.1
    
    [[2770   27]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.84      0.91      0.87       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.125
    
    [[2772   25]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.85      0.91      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.15
    
    [[2778   19]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.91      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.175
    
    [[2781   16]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.90      0.90      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1    ;  Classification Probability= 0.2
    
    [[2782   15]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.90      0.90      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 1    ;  Default Classification Probability)= 0.5
    
    [[2795    2]
     [  26  129]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.98      0.83      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.025
    
    [[2632  165]
     [  11  144]]
                 precision    recall  f1-score   support
    
              0       1.00      0.94      0.97      2797
              1       0.47      0.93      0.62       155
    
    avg / total       0.97      0.94      0.95      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.05
    
    [[2724   73]
     [  13  142]]
                 precision    recall  f1-score   support
    
              0       1.00      0.97      0.98      2797
              1       0.66      0.92      0.77       155
    
    avg / total       0.98      0.97      0.97      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.075
    
    [[2757   40]
     [  13  142]]
                 precision    recall  f1-score   support
    
              0       1.00      0.99      0.99      2797
              1       0.78      0.92      0.84       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.1
    
    [[2764   33]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.81      0.91      0.86       155
    
    avg / total       0.99      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.125
    
    [[2772   25]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.85      0.91      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.15
    
    [[2778   19]
     [  15  140]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.90      0.89       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.175
    
    [[2782   15]
     [  15  140]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.90      0.90      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 10    ;  Classification Probability= 0.2
    
    [[2784   13]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      2797
              1       0.91      0.89      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 10    ;  Default Classification Probability)= 0.5
    
    [[2795    2]
     [  25  130]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.98      0.84      0.91       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.025
    
    [[2642  155]
     [  11  144]]
                 precision    recall  f1-score   support
    
              0       1.00      0.94      0.97      2797
              1       0.48      0.93      0.63       155
    
    avg / total       0.97      0.94      0.95      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.05
    
    [[2724   73]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.97      0.98      2797
              1       0.66      0.91      0.76       155
    
    avg / total       0.98      0.97      0.97      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.075
    
    [[2753   44]
     [  14  141]]
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.99      2797
              1       0.76      0.91      0.83       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.1
    
    [[2770   27]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.84      0.90      0.87       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.125
    
    [[2776   21]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.87      0.89      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.15
    
    [[2778   19]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.89      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.175
    
    [[2782   15]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.90      0.89      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 100    ;  Classification Probability= 0.2
    
    [[2783   14]
     [  18  137]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.91      0.88      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 100    ;  Default Classification Probability)= 0.5
    
    [[2796    1]
     [  26  129]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.99      0.83      0.91       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.025
    
    [[2640  157]
     [  10  145]]
                 precision    recall  f1-score   support
    
              0       1.00      0.94      0.97      2797
              1       0.48      0.94      0.63       155
    
    avg / total       0.97      0.94      0.95      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.05
    
    [[2727   70]
     [  13  142]]
                 precision    recall  f1-score   support
    
              0       1.00      0.97      0.99      2797
              1       0.67      0.92      0.77       155
    
    avg / total       0.98      0.97      0.97      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.075
    
    [[2754   43]
     [  15  140]]
                 precision    recall  f1-score   support
    
              0       0.99      0.98      0.99      2797
              1       0.77      0.90      0.83       155
    
    avg / total       0.98      0.98      0.98      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.1
    
    [[2769   28]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.83      0.90      0.86       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.125
    
    [[2774   23]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.86      0.90      0.88       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.15
    
    [[2778   19]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.88      0.90      0.89       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.175
    
    [[2781   16]
     [  16  139]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.90      0.90      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    Event Rate=0.05     ;  C Parameter= 1000    ;  Classification Probability= 0.2
    
    [[2783   14]
     [  17  138]]
                 precision    recall  f1-score   support
    
              0       0.99      0.99      0.99      2797
              1       0.91      0.89      0.90       155
    
    avg / total       0.99      0.99      0.99      2952
    
    ======================
    
    Event Rate=0.05     ;  C Parameter= 1000    ;  Default Classification Probability)= 0.5
    
    [[2796    1]
     [  25  130]]
                 precision    recall  f1-score   support
    
              0       0.99      1.00      1.00      2797
              1       0.99      0.84      0.91       155
    
    avg / total       0.99      0.99      0.99      2952
    
    **********************
    ..............................
    


```python
#ROC AUC Curve

from sklearn import metrics
fpr,tpr, thresholds = metrics.roc_curve(Y_test_data,probs)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Fraud classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


roc_auc = auc(fpr, tpr)
roc_auc

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from ggplot2 import *
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')



```


![png](output_13_0.png)



    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-24-8eb09966bb62> in <module>()
         27 
         28 
    ---> 29 from ggplot2 import *
         30 df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
         31 ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')
    

    ModuleNotFoundError: No module named 'ggplot2'

