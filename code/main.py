from json import load
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_data, data_split
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np



def linear_regressors(X_train, X_test, y_train, y_test):
    '''
    Evaluating initial coefficients assigned using OLS, Lasso, and Ridge
    '''
    trained_ols = sm.OLS(y_train, X_train).fit()
    ols_pred = trained_ols.predict(X_test)
    ols_mse = mean_squared_error(y_test, ols_pred)
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
    lasso_pred = lassoReg.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)

    ridgeReg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=ridge_alpha * X_train.shape[0]))
    ridgeReg.fit(X_train, y_train)
    ridge_pred = ridgeReg.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist(), 'val_lasso':lassoReg['lasso'].coef_, 'var_ridge':ridgeReg['ridge'].coef_})


    mse = min(ols_mse, lasso_mse, ridge_mse)
    if mse == ols_mse:
        coef = 'val_ols'
    elif mse == lasso_mse:
        coef = 'val_lasso'
    elif mse == ridge_mse:
        coef = 'val_ridge'
    print(coef)
    
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    plt.scatter(coef_comp[coef], X_train.columns, )
    plt.errorbar(coef_comp[coef], X_train.columns, xerr=mse, fmt="o", ecolor="red")
    plt.grid(True)
    ax.set_facecolor("whitesmoke")
    plt.ylabel('Coefficient', fontweight="bold")
    plt.xlabel('Value', fontweight="bold")
    plt.title('Coefficient Plot', fontweight="bold")
    
    plt.savefig('../outputs/coefficients.jpeg')
    plt.show()

    
    print(coef_comp)

def clusters():
    pass

def multilinear():
    pass

def main():
    init_data = load_data()
    init_data.describe().to_csv('../outputs/descriptive_stats.csv')
    X_train, X_test, y_train, y_test = data_split(init_data)
    linear_regressors(X_train, X_test, y_train, y_test)
    #coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist()})
    



if __name__ == '__main__':
    main()