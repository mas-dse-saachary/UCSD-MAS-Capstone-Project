import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import seaborn as sb
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.compat import lzip
import matplotlib.cm as cm

def descriptive_stats(X,varname):
    print 'Skew for original variable: ', X[varname].skew()
    print 'Kurtosis for original variable: ', X[varname].kurtosis()
    print(X[varname].describe())
    
    
def plot_var_distrib(X,varname):
    
    ax = plt.axes()
    sb.distplot(X[varname].fillna(X[varname].mean()),ax=ax)
    ax.set_title('Distribution of %s'%(varname))
    plt.show()

def plot_residuals_distribution(y,predictions):
    residuals = y - predictions
    
    #Residuals Distribution
    sb.distplot(residuals)
    
def plot_residuals(X,y,predictions):
    residuals = y - predictions
    
    #Residuals Plot
    plt.scatter(X.index, residuals, alpha=0.12)
    plt.xlabel('Listing')
    plt.ylabel('Price Residual')
    plt.show()
    
def plot_fitted_residuals(y,predictions):
    residuals = y - predictions
    plt.plot(predictions, residuals, 'x')
    plt.ylabel('Residuals')
    plt.xlabel('Fitted Values')
    plt.show()
    
def plot_predictions(y,predictions):
    
    #to do: sort predictions by different variables such as zipcode, etc.
    
    plt.scatter(predictions, y, alpha=0.1)
    plt.ylabel('Price')
    plt.xlabel('Predictions')
    plt.show()
    
def plot_partial_residuals(model_results,X,y,predictions):
    #Partial Residuals
    indx = [i for i in range(0,len(X.columns))]
    residuals = y - predictions
    
    fig, axes = plt.subplots(nrows = len(X.columns), ncols = 3, sharex = False, sharey= False, figsize = (18,len(X.columns)*2.5))
    axes_list = [item for sublist in axes for item in sublist] 
    
    for i,val in enumerate(X.columns):
        ax = axes_list.pop(0)
        partial_residual = residuals + X[val]*model_results.coef_[i]
        ax.plot(X[val], partial_residual, 'o', alpha=0.1)
        ax.set_title(str(val))
        ax.set_ylabel('Partial Residuals')
    
    for ax in axes_list:
        ax.remove()
        
def plot_outliers(X):
    plt.figure(figsize=(25,8))
    stand = StandardScaler(with_mean=True, with_std=True)
    Xs = stand.fit_transform(X)
    boxplot = plt.boxplot(Xs)
    plt.show()
    
def map_variable(var):
    fig, ax = plt.subplots(figsize=(19,8))
    plt.scatter(listings['longitude'],listings['latitude'],c=var, cmap='RdBu', alpha=.5,vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()
    
def make_visualizations(X,y,predictions,model_results):
    
    #detect_outliers(X)
    plot_residuals_distribution(y,predictions)
    plot_residuals(X,y,predictions)
    plot_fitted_residuals(y,predictions)
    plot_predictions(y,predictions)
    plot_partial_residuals(model_results,X,y,predictions)
    plot_outliers(X)
    
    residuals = y - predictions
    map_variable(residuals)
    
def scale_data(X):
    return pd.DataFrame(preprocessing.scale(X),columns = X.columns)

def r2_est(X,y):
    linear_regression = linear_model.LinearRegression(normalize=True, fit_intercept=True)
    return r2_score(y, linear_regression.fit(X,y).predict(X))

def r2_sq_est(X,y):
    linear_regression = linear_model.LinearRegression(normalize=True, fit_intercept=True)
    quad = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    quadratic_predictor = make_pipeline(quad, linear_regression)
    return r2_score(y, quadratic_predictor.fit(X,target).predict(X))

def bivar_reg_linear(X,y):
    R2s = []
    for col in X.columns:
        linear_regression = linear_model.LinearRegression(normalize=True, fit_intercept=True)
        regression_results = linear_regression.fit(X[col].values.reshape(-1, 1),y)
        predictions = cross_val_predict(regression_results, X[col].values.reshape(-1, 1), y, cv=10)
        R2s.append((col, r2_score(y, predictions)))
    return R2s

def bivar_reg_poly(X,y,degree):
    R2s = []
    for col in X.columns:
        model = Pipeline([('poly', PolynomialFeatures(degree=degree, interaction_only=False)),('linear', linear_model.LinearRegression(normalize=True,fit_intercept=True))])
        model = model.fit(X[col].values.reshape(-1, 1),y)
        predictions = cross_val_predict(model, X[col].values.reshape(-1, 1), y, cv=10)
        
        R2s.append((col, r2_score(y, predictions)))
        
    return R2s

def Ridge_reg(X,y,alpha):
    #modify this to use gridsearch cv this weekend: 
    #https://stackoverflow.com/questions/45857274/interpreting-ridge-regression-in-gridsearchcv
    estimator = linear_model.Ridge(alpha=alpha)
    estimator.fit(X, y)
    
    return estimator.predict(X), estimator.coef_

def Lasso_reg(X,y,alpha):
    #modify this to use gridsearch cv this weekend: 
    #https://stackoverflow.com/questions/45857274/interpreting-ridge-regression-in-gridsearchcv
    estimator = linear_model.Lasso(alpha=alpha)
    estimator.fit(X, y)
    
    return estimator.predict(X), estimator.coef_

def Random_Lasso_reg(X,y,alpha):
    #modify this to use gridsearch cv this weekend: 
    #https://stackoverflow.com/questions/45857274/interpreting-ridge-regression-in-gridsearchcv
    estimator = linear_model.RandomizedLasso(alpha=alpha)
    estimator.fit(X, y)
    
    return estimator.scores_

def RFECV_eval(X,y):
    estimator = linear_model.LinearRegression()
    selector = RFECV(estimator, step=1, cv=3)
    selector.fit(X, y)
    return selector.ranking_

def RF_reg(X,y):
    estimator = RandomForestRegressor(max_depth=2, random_state=0)
    estimator.fit(X, y)
    return estimator.predict(X), estimator.feature_importances_

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def linear_reg(X, y):
    
    model = linear_model.LinearRegression(normalize=True, fit_intercept=True)
    scores = cross_validate(model, X, y, cv=10, scoring=('r2', 'neg_mean_squared_error','neg_mean_absolute_error'))
    predictions = cross_val_predict(model, X, y, cv=10)
    model_results = model.fit(X,y)
    
    r2 = scores['test_r2'].mean()
    mse = -scores['test_neg_mean_squared_error'].mean()
    mae =  -scores['test_neg_mean_absolute_error'].mean()
    rmse =  np.sqrt(-scores['test_neg_mean_absolute_error'].mean())
    
    return model, predictions, model_results, r2, mse, mae, rmse

def nonlinear_reg(X,y, deg):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model = Pipeline([('poly', PolynomialFeatures(degree = deg, interaction_only=False)),
                       ('linear', linear_model.LinearRegression(normalize=True, fit_intercept=True))])
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_test)
       
        
    r2 = r2_score(predictions, y_test)
    mse = mean_squared_error(predictions, y_test)
    mae =  mean_absolute_error(predictions, y_test)
    rmse =  np.sqrt(mse)
    
    return model, predictions, r2, mse, mae, rmse

def detect_feature_importance(X,y):
    
    names = X.columns
    ranks = {}
    
    model, predictions, model_results, r2, mse, mae, rmse = linear_reg(X, y)
    ranks["Linear_Reg"] = rank_to_dict(np.abs(model.coef_), names)
    
    model, predictions, r2, mse, mae, rmse = nonlinear_reg(X, y,2)
    ranks["Quad_Reg"] = rank_to_dict(np.abs(model.named_steps['linear'].coef_), names)
    
    #model, predictions, model_results, r2, mse, mae, rmse = nonlinear_reg(X, y,3)
    #ranks["Cub_Reg"] = rank_to_dict(np.abs(model.coef_), names)
    
    pred, coef = Ridge_reg(X,y,7)
    ranks["Ridge"] = rank_to_dict(np.abs(coef), names)

    pred, coef = Lasso_reg(X,y,0.05)
    ranks["Lasso"] = rank_to_dict(np.abs(coef), names)

    scores = Random_Lasso_reg(X,y,0.05)
    ranks["Stability"] = rank_to_dict(np.abs(scores), names)

    ranking = RFECV_eval(X,y)
    ranks["RFECV"] = rank_to_dict(map(float, ranking), names, order=-1)

    pred, imp = RF_reg(X,y)
    ranks["RF"] = rank_to_dict(imp, names)

    #f, pval  = f_regression(X, y, center=True)
    #ranks["Corr"] = rank_to_dict(f, names)
    
    return ranks







def compute_bivar_r2s(X,y,y_log):
    linear_R2s = bivar_reg_linear(X,y)
    linear_R2s_log = bivar_reg_linear(X,y_log)
    quad_R2s = bivar_reg_poly(X,y,2)
    quad_R2s_log = bivar_reg_poly(X,y_log,2)
    cub_R2s = bivar_reg_poly(X,y,3)
    cub_R2s_log = bivar_reg_poly(X,y_log,3)
    
    idx = [i[0] for i in linear_R2s]
    lin_r2 = [i[1] for i in linear_R2s]
    lin_r2_log = [i[1] for i in linear_R2s_log]
    quad_r2 = [i[1] for i in quad_R2s]
    quad_r2_log = [i[1] for i in quad_R2s_log]
    cub_r2 = [i[1] for i in cub_R2s]
    cub_r2_log = [i[1] for i in cub_R2s_log]

    bivar_r2s = pd.DataFrame({'Feature':idx, 'R2_linear': lin_r2, 'R2_linear_logy':lin_r2_log,
                       'R2_quad':quad_r2, 'R2_quad_logy': quad_r2_log, 'R2_cub':cub_r2, 'R2_cub_logy':cub_r2_log})
    
    return bivar_r2s

def summarize_differences(bivar_r2s, line, title):    
   
        plt.figure(figsize=(28,15))
        plt.ylabel('Difference')
        plt.legend(loc='upper left', framealpha=0.2, prop={'size':'small'})
        plt.xlabel('Features')
        plt.title(title)
        plt.scatter(bivar_r2s.index, line, alpha=0.7)

        for label, x, y in zip(bivar_r2s.Feature, bivar_r2s.index, line):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
        plt.show()
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    