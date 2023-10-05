import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set(style = "darkgrid")
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import neighbors, metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import onlyYOUcanPREPROCESSsmoke as PreProcess




def main():
    df = pd.read_csv('messy_wildfire_train.csv')
    #df_test = pd.read_csv('wildfire_test.csv')
    
    models = PreProcess.cleanAndEncode(df)
    
    # Creates the basis for our models based on the type of imputing used
    simple_model = MLModel("simple", models[0])
    iterative_model = MLModel("iterative", models[1])
    knn_model = MLModel("knn", models[2])
    
    for model in models:
        print()
        print()
        print("_-_-_-_-_-_-_-_-_-_-_-_ " + model.name + " MODEL _-_-_-_-_-_-_-_-_-_-_-_ ")
        model.findKNN()
        model.findRFC()
        model.findCLF()
        model.findHistGBC()
    return




# Definitly not an MLM. Think of it more as a reverse funnel
class MLModel:
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        
        self.knn_pred = None
        self.knn_acc = None
        
        self.rfc_pred = None
        self.rfc_acc = None
        
        self.clf_pred = None
        self.clf_acc = None
        
        self.hgbc_pred = None
        self.hgbc_acc = None
        
        self.predictors()
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
        
        self.scaler(self.X_train, self.X_valid)
        
    # Sets "size_class" as our predictor
    def predictors(self):
        self.X = self.df.drop("size_class", axis=1)
        self.y = self.df["size_class"]
        return
    
    # Scales the data so it isn't so wonky
    def scaler(self, train, valid):
        sc = StandardScaler()
        self.X_train = sc.fit_transform(train)
        self.X_valid = sc.transform(valid)
        return
    
    # Where are my nearest neighbours?!
    def findKNN(self):
        knn = neighbors.KNeighborsClassifier(n_neighbors=20, weights="uniform")
        knn.fit(self.X_train, self.y_train) #Thats it, thats the training, all there is to it, you have a model now
        self.knn_pred = knn.predict(self.X_valid) #.predict() takes our X_test and returns an array of predicted labels
        self.knn_acc = metrics.accuracy_score(self.y_valid, self.knn_pred) #.accuracy_score() returns a number reflecting how accurately our model predicted our testing data
        self.knn_acc
        
        print()
        print("-------------------------------- K N N --------------------------------")
        print(classification_report(self.y_valid, self.knn_pred))
        print(confusion_matrix(self.y_valid, self.knn_pred))
        return
    
    
    # I love finding forest. LOL I'm so random rawr xD
    def findRFC(self):
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(self.X_train, self.y_train)
        self.rfc_pred = rfc.predict(self.X_valid)
        self.rfc_acc = metrics.accuracy_score(self.y_valid, self.rfc_pred)
        self.rfc_acc
        
        print()
        print("-------------------------------- R F C --------------------------------")
        print(classification_report(self.y_valid, self.rfc_pred))
        print(confusion_matrix(self.y_valid, self.rfc_pred))
        return


    # I don't know enough about this one to make a decent joke. 
    # You must be this tall ------- to divide the data.
    # (It was a stretch, I know. Shut up.)
    def findCLF(self):
        clf = SVC()
        clf.fit(self.X_train, self.y_train)
        self.clf_pred = clf.predict(self.X_valid)
        self.clf_acc = metrics.accuracy_score(self.y_valid, self.clf_pred)
        self.clf_acc
        
        print()
        print("-------------------------------- C L F --------------------------------")
        print(classification_report(self.y_valid, self.clf_pred))
        print(confusion_matrix(self.y_valid, self.clf_pred))
        return
    
    def findHistGBC(self):
        hgbc = HistGradientBoostingClassifier()
        hgbc.fit(self.X_train, self.y_train)
        
        self.hgbc_pred = hgbc.predict(self.X_valid)
        self.hgbc_acc = metrics.accuracy_score(self.y_valid, self.hgbc_pred)
         
        print()
        print("-------------------------------- H G B C--------------------------------")
        print(classification_report(self.y_valid, self.hgbc_pred))
        print(confusion_matrix(self.y_valid, self.hgbc_pred))
        return

main()