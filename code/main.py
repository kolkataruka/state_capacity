from json import load
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import load_data, data_split
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from linearmodels import PanelOLS
import numpy as np
import seaborn as sns



def linear_regressors(X_train, X_test, y_train, y_test):
    '''
    Evaluating initial coefficients assigned using OLS and Lasso
    '''

    #OLS
    trained_ols = sm.OLS(y_train, X_train).fit()
    ols_pred = trained_ols.predict(X_test)
    ols_mse = mean_squared_error(y_test, ols_pred)
    print(trained_ols.summary())

    #Lasso
    lassoReg = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=1e-06))
    lassoReg.fit(X_train, y_train)
    lasso_pred = lassoReg.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    print(lassoReg.score(X_test, y_test))


    coef_comp=pd.DataFrame({'var':X_train.columns, 'val_ols':trained_ols.params.tolist(), 'val_lasso':lassoReg['lasso'].coef_})

    #Lasso Coefficients visualization
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    plt.scatter(coef_comp['val_lasso'][:8], X_train.columns[:8], )
    plt.errorbar(coef_comp['val_lasso'][:8], X_train.columns[:8], xerr=lasso_mse, fmt="o", ecolor="red")
    plt.grid(True)
    ax.set_facecolor("whitesmoke")
    plt.ylabel('Coefficient', fontweight="bold")
    plt.xlabel('Value', fontweight="bold")
    plt.title('Coefficient Plot', fontweight="bold")
    
    plt.savefig('../outputs/coefficients.jpeg')
    #plt.show()

    cols = X_train.columns[:8]
    for col in cols:
        visualise(X_test, lasso_pred, col)

    #Printing to LateX
    print(coef_comp)
    coef_comp.to_latex('../outputs/lasso_coef.tex')

def pca(df):
    '''Conducting Principal Component Analysis'''


    df = df.dropna()
    #Getting Dummies for fixed effects 
    year_dummies = pd.get_dummies(df['Year'], drop_first=True, dtype=int)
    country_dummies = pd.get_dummies(df['Code'], drop_first=True, dtype=int)
    new_df = pd.concat([df, year_dummies, country_dummies], axis=1)
    new_df = new_df.drop(columns=['Code', 'Year'])

    #Splitting data and generating PCA
    y = new_df[['Human Development Index']]
    X = new_df.drop(columns=['Human Development Index'])
    pca = PCA(n_components=1).fit_transform(X[['rigor_admin', 'rule_of_law', 'state_capacity', 'taxation', 'territory_control']])
    X['pc1'] = pca[:,0] 
    new_df['pc1'] = X['pc1']

    #Normalizing data
    new_df_scaled = normalize(X) 
    new_X = pd.DataFrame(new_df_scaled, columns=X.columns, index=X.index)
    

    #Splitting into training and testing, training the model
    X_train, X_test, y_train, y_test = train_test_split(new_X.drop(columns=['rigor_admin', 'rule_of_law', 'state_capacity', 'taxation', 'territory_control']), y, test_size=0.2, random_state=1680)
    ols = sm.OLS(y_train, X_train)
   
    trained_reg = ols.fit()
    ols_pred = trained_reg.predict(X_test)
    ols_mse = mean_squared_error(y_test, ols_pred)
    summary = summary_col([trained_reg], stars=True, float_format='%0.2f', model_names=['OLS FE'], regressor_order=['pc1', 'civil_liberties', 'corruption', 'years_colonized'], drop_omitted=True).as_latex()
    
    #Printing to LateX
    print(trained_reg.summary())
    print(f'OLS MSE:{ols_mse}')
    with open('../outputs/pca_table.tex', 'w') as file:
        file.write(summary)
    

    #Visualizing results of the model
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    params = trained_reg.params.tolist()
    plt.scatter(params[:3] + params[-1:], X_train[['civil_liberties', 'corruption', 'years_colonized', 'pc1']].columns)
    plt.errorbar(params[:3] + params[-1:], X_train[['civil_liberties', 'corruption', 'years_colonized', 'pc1']].columns, xerr=ols_mse, fmt="o", ecolor="red")
    plt.grid(True)
    ax.set_facecolor("whitesmoke")
    plt.ylabel('Coefficient', fontweight="bold")
    plt.xlabel('Value', fontweight="bold")
    plt.title('Coefficient Plot', fontweight="bold")
    
    plt.savefig('../outputs/coefficients_pca.jpeg')
    
    plt.figure(figsize=(12, 8))
    ax1 = plt.axes()
    sns.regplot(x=X_test['pc1'], y=ols_pred, scatter_kws={"color": "#20b2aa"}, line_kws={"color": "red"})
    plt.grid(True)
    ax1.set_facecolor("whitesmoke")
    plt.title('HDI vs the Principal Component')
    plt.xlabel('PCA')
    plt.ylabel('Human Development Index')
    plt.savefig('../outputs/pca_graph.jpeg')
    plt.show()  



def visualise(xdata, y, col):
    '''Visualising relationships between predicted HDI and independent variables'''
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    sns.regplot(x=xdata[col], y=y,scatter_kws={"color": "#20b2aa"}, line_kws={"color": "red"})
    plt.grid(True)
    ax.set_facecolor("whitesmoke")
    plt.title('HDI vs ' + col)
    plt.xlabel(col)
    plt.ylabel('Predicted Human Development Index')
    plt.savefig('../outputs/' + col + '_graph.jpeg')
    plt.show()  
    

def main():
    '''Main function to load, preprocess, and train the data'''
    init_data = load_data()[['Human Development Index', 'rigor_admin','rule_of_law','civil_liberties','corruption','years_colonized','state_capacity','taxation','territory_control', 'Code', 'Year']]
    init_data.describe().to_latex('../outputs/descriptive_stats.tex')
    X_train, X_test, y_train, y_test = data_split(init_data)
    linear_regressors(X_train, X_test, y_train, y_test)
    pca(init_data)
   
    



if __name__ == '__main__':
    main()





#Ignore please

# def multilayer(X_train, X_test, y_train, y_test):
#     '''Training the multilayer neural network'''

#     scaler=StandardScaler() 
#     scaler.fit(X_train) 
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     print(y_train)
#     #Initializing a multilayer perceptron classifer with defined hidden layers, optimizer, and learning rate
#     MLP = MLPRegressor(
#     random_state=1680,
#                         activation='relu', solver='adam', 
#                         max_iter =500,
#                         learning_rate_init=0.01) 
#     #Training the MLP model using the training data
#     MLP.fit(X_train,y_train)
#     #Printing accuracy of train and test predictions

#     print(MLP.score(X_train,y_train))
#     print("mlp test accuracy:")
#     print(MLP.score(X_test, y_test))