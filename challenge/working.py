import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score

from multiprocessing import cpu_count, Process, Manager


def load_data():
    data=np.load('embeddings_1.npy')
    data=np.concatenate((data, np.load('embeddings_2.npy')), axis=0)
    labels=[]
    for file_name in ['icd_codes_1.txt', 'icd_codes_2.txt']:
        with open(file_name) as file:
            for line in file:
                labels.append(line.strip().split(';'))
    multihot=MultiLabelBinarizer(sparse_output=True)
    labels=multihot.fit_transform(labels)
    return data, labels, multihot




class MultiLogReg(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimators= Manager().dict()
        self.estimator=estimator
        self._is_fitted=False
        self.n_outputs=0
    
    # def fit_individual(self, X, y, estimator):
    #     print(id(X))
    #     try:
    #         estimator.fit(X, y)
    #     except ValueError:
    #         return

    def fit(self, X, y):
        print(self.fit.__name__)
        self.estimators=Manager().dict()
        self.estimators['data']=X
        self.n_outputs=y.shape[1]
        processes=[]
        for i in range(self.n_outputs):
            self.estimators['labels']=y[:, i].toarray().flatten()
            new_estimator=clone(self.estimator)
            self.estimators[i]=new_estimator
            p=Process(target=fit_individual, args=(i, self.estimators))
            processes.append(p)
            if __name__ == '__main__':
                print(1)
                p.start()
            # fit_individual(X, y[:, i].toarray().flatten(), new_estimator)
        for p in processes:
            if __name__ == '__main__':
                p.join()

        self._is_fitted=True
        return self
    
    def predict(self, X):
        predictions=[]
        for i in range(self.n_outputs):
            pred=None
            try:
                pred=self.estimators[i].predict(X)
            except AttributeError:
                pred=np.zeros(X.shape[0])
            predictions.append( sparse.csr_matrix(pred) )
        
        return sparse.vstack(predictions).T

    def __sklearn_is_fitted__(self):
        return self._is_fitted

def fit_individual(i, estimators):
    print(id(estimators['data']))
    try:
        estimators[i]=estimators[i].fit(estimators['data'], estimators['labels'])
    except ValueError:
        return


if __name__=='__main__': 
    X, y, multihot=load_data()
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)


    n=2
    clsf=MultiLogReg(LogisticRegression())
    clsf.fit(X_test, y_test[:, :n])

    print(type(clsf.estimators[0]))

    # predictions=clsf.predict(X_train)
    # print(fbeta_score(y_train[:, :n], predictions, beta=2, average='micro'))

    # predictions=clsf.predict(X_test)
    # print(fbeta_score(y_test[:, :n], predictions, beta=2, average='micro'))

# from multiprocessing import Process, cpu_count
# import os

# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())

# def f(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()

#     print(cpu_count())