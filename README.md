# hoax_detection_model
### Python version 3.7 or above

Python library requirement:
```
pip3 install pickle
pip3 install pandas
pip3 install sklearn
pip3 install Sastrawi
```
The main file is in: ``nnq.py``

If you want to predict a set of text document, please change the following code into your csv filename. Store different text data separated by each line. 

```
X_test = pd.read_csv('testdata.csv', error_bad_lines=False, encoding='latin1')
```
then change the following code: 
```
X_tf1 = tf1_new.fit_transform([inputdata])
```
into:
```
X_tf1 = tf1_new.fit_transform(X_test)
```

if you just want to use one input, store the text data into variable ```inputdata```
