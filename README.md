# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: n.magesh 
RegisterNumber:  212222040091



import chardet
file = "spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()

data.info()

data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy



*/
```

## Output:
Result output

![Screenshot 2023-11-03 163438](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/b8d6c70c-468c-4dee-9b43-a42b3d817ef3)

data.head()

![Screenshot 2023-11-03 163446](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/c475ab9b-8313-4570-b027-3d9a78d1ff38)

data.info()

![Screenshot 2023-11-03 163501](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/e3122181-9af6-44da-ac62-5afc3d80252b)

data.isnull().sum()

![Screenshot 2023-11-03 163508](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/c6611f61-dabc-4041-b331-585065a8a46b)

Y_prediction value

![Screenshot 2023-11-03 163523](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/b86c9fcf-06ed-4b03-a271-fffdd386360e)

Accuracy value

![Screenshot 2023-11-03 163529](https://github.com/22008496/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119476113/d6d7d415-09fe-410d-a690-8ed7b14f2119)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
