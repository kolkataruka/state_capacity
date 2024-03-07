from json import load
import pandas as pd
import tensorflow as tf
from preprocess import load_data, data_split
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def linear_regressors(X_train, y_train):
    '''
    Look at initial coefficients assigned using OLS, Lasso, and Ridge
    '''
    trained_ols = sm.OLS(y_train, X_train).fit()
    lassoreg = make_pipeline(StandardScaler(with_mean=False), Lasso())
    alphas=np.linspace(1e-6, 1, num=50)
    params = {'lasso__alpha':alphas}
    gslasso = GridSearchCV(lassoreg, params, n_jobs=-1, cv=10)
    gslasso.fit(X_train, y_train)
    lasso_alpha = list(gslasso.best_params_.values())[0]
    ridgereg = make_pipeline(StandardScaler(with_mean=False), Ridge())
    alphas=np.linspace(1e-6, 1, num=50)
    ridgeparams = {'ridge__alpha':alphas * X_train.shape[0]}
    gsridge = GridSearchCV(ridgereg, ridgeparams, n_jobs=-1, cv=10)
    gsridge.fit(X_train, y_train)
    ridge_alpha = list(gsridge.best_params_.values())[0] / X_train.shape[0]
    lassoReg = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=lasso_alpha))
    lassoReg.fit(X_train, y_train)
    ridgeReg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=ridge_alpha * X_train.shape[0]))
    ridgeReg.fit(X_train, y_train)
    coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist(), 'val_lasso':lassoReg['lasso'].coef_, 'var_ridge':ridgeReg['ridge'].coef_})
    print(coef_comp)

def main():
    init_data = load_data()
    init_data.describe().to_csv('../outputs/descriptive_stats.csv')
    X_train, X_test, y_train, y_test = data_split(init_data)
    linear_regressors(X_train, y_train)
    #coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist()})
    



if __name__ == '__main__':
    main()