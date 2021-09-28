stocks = ["dollar", "netherlands", "australia", "japan", "france", "spain", "usa", "switzerland", "china", "europe",
          "hongkong", "india", "poland", "germany", "uk"]

currencies = ["usdChf", "gbpUsd", "usdJpy", "ethUsd", "eurUsd", "btcUsd"]

tweets = ["stockmarketTotalCount", "stockmarketNegCount", "stockmarketNeuCount", "stockmarketPosCount",
          "stockmarketNegOverTotal", "stockmarketNeuOverTotal", "stockmarketPosOverTotal", "stockmarketCompoundAverage",
          "dayTotalCount", "dayNegCount", "dayNeuCount", "dayPosCount", "dayNegOverTotal", "dayNeuOverTotal",
          "dayPosOverTotal", "dayCompoundAverage", "forexTotalCount", "forexNegCount", "forexNeuCount", "forexPosCount",
          "forexNegOverTotal", "forexNeuOverTotal", "forexPosOverTotal", "forexCompoundAverage", "currencyTotalCount",
          "currencyNegCount", "currencyNeuCount", "currencyPosCount", "currencyNegOverTotal", "currencyNeuOverTotal",
          "currencyPosOverTotal", "currencyCompoundAverage", "moneyTotalCount", "moneyNegCount", "moneyNeuCount",
          "moneyPosCount", "moneyNegOverTotal", "moneyNeuOverTotal", "moneyPosOverTotal", "moneyCompoundAverage",
          "exchangeTotalCount", "exchangeNegCount", "exchangeNeuCount", "exchangePosCount", "exchangeNegOverTotal",
          "exchangeNeuOverTotal", "exchangePosOverTotal", "exchangeCompoundAverage", "marketTotalCount",
          "marketNegCount", "marketNeuCount", "marketPosCount", "marketNegOverTotal", "marketNeuOverTotal",
          "marketPosOverTotal", "marketCompoundAverage", "tradingTotalCount", "tradingNegCount", "tradingNeuCount",
          "tradingPosCount", "tradingNegOverTotal", "tradingNeuOverTotal", "tradingPosOverTotal",
          "tradingCompoundAverage", "investingTotalCount", "investingNegCount", "investingNeuCount",
          "investingPosCount", "investingNegOverTotal", "investingNeuOverTotal", "investingPosOverTotal",
          "investingCompoundAverage", "dollarTotalCount", "dollarNegCount", "dollarNeuCount", "dollarPosCount",
          "dollarNegOverTotal", "dollarNeuOverTotal", "dollarPosOverTotal", "dollarCompoundAverage", "obamaTotalCount",
          "obamaNegCount", "obamaNeuCount", "obamaPosCount", "obamaNegOverTotal", "obamaNeuOverTotal",
          "obamaPosOverTotal", "obamaCompoundAverage", "trumpTotalCount", "trumpNegCount", "trumpNeuCount",
          "trumpPosCount", "trumpNegOverTotal", "trumpNeuOverTotal", "trumpPosOverTotal", "trumpCompoundAverage",
          "dolarTotalCount", "dolarNegCount", "dolarNeuCount", "dolarPosCount", "dolarNegOverTotal",
          "dolarNeuOverTotal", "dolarPosOverTotal", "dolarCompoundAverage", "economyTotalCount", "economyNegCount",
          "economyNeuCount", "economyPosCount", "economyNegOverTotal", "economyNeuOverTotal", "economyPosOverTotal",
          "economyCompoundAverage", "breakingTotalCount", "breakingNegCount", "breakingNeuCount", "breakingPosCount",
          "breakingNegOverTotal", "breakingNeuOverTotal", "breakingPosOverTotal", "breakingCompoundAverage",
          "todayTotalCount", "todayNegCount", "todayNeuCount", "todayPosCount", "todayNegOverTotal",
          "todayNeuOverTotal", "todayPosOverTotal", "todayCompoundAverage", "stocksTotalCount", "stocksNegCount",
          "stocksNeuCount", "stocksPosCount", "stocksNegOverTotal", "stocksNeuOverTotal", "stocksPosOverTotal",
          "stocksCompoundAverage", "donaldtrumpTotalCount", "donaldtrumpNegCount", "donaldtrumpNeuCount",
          "donaldtrumpPosCount", "donaldtrumpNegOverTotal", "donaldtrumpNeuOverTotal", "donaldtrumpPosOverTotal",
          "donaldtrumpCompoundAverage", "euroTotalCount", "euroNegCount", "euroNeuCount", "euroPosCount",
          "euroNegOverTotal", "euroNeuOverTotal", "euroPosOverTotal", "euroCompoundAverage", "usaTotalCount",
          "usaNegCount", "usaNeuCount", "usaPosCount", "usaNegOverTotal", "usaNeuOverTotal", "usaPosOverTotal",
          "usaCompoundAverage", "breakingnewsTotalCount", "breakingnewsNegCount", "breakingnewsNeuCount",
          "breakingnewsPosCount", "breakingnewsNegOverTotal", "breakingnewsNeuOverTotal", "breakingnewsPosOverTotal",
          "breakingnewsCompoundAverage", "stockTotalCount", "stockNegCount", "stockNeuCount", "stockPosCount",
          "stockNegOverTotal", "stockNeuOverTotal", "stockPosOverTotal", "stockCompoundAverage", "turkeyTotalCount",
          "turkeyNegCount", "turkeyNeuCount", "turkeyPosCount", "turkeyNegOverTotal", "turkeyNeuOverTotal",
          "turkeyPosOverTotal", "turkeyCompoundAverage", "usdtryTotalCount", "usdtryNegCount", "usdtryNeuCount",
          "usdtryPosCount", "usdtryNegOverTotal", "usdtryNeuOverTotal", "usdtryPosOverTotal", "usdtryCompoundAverage"]

p_values = [1, 2]
power_values = [0, 1, 2, 3]
bool_values = [True, False]
c_values = [1, 2, 3, 300, 500]
selection_values = ['cyclic', 'random']
n_neighbors_values = [1, 2, 3, 5, 10, 15, 25, 50, 100, 200, 500]
# max_iter_values = [100, 1000, 10000, 100000]
leaf_size_values = [1, 2, 3, 5, 10, 15, 25, 50, 75, 100, 150, 250]
tol_values = [1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
epsilon_values = [1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]
validation_fraction_values = [1, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]
lambda_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_leaf_values = [1, 2, 3, 4, 5]
random_state_values = [0, 1, None]
degree_values = [1, 2, 3, 4, 5]
alpha_values = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000000001]
gamma_values = ['scale', 'auto']
penalty_values = ['l1', 'l2', 'elasticnet', 'none']
solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'lbfgs']
multi_class_values = ['auto', 'ovr', 'multinomial']
weight_values = ['uniform', 'distance']
algorithm_values = ['auto', 'ball_tree', 'kd_tree', 'brute']

logisticRegressionParameters = [
    {
        # 'tol': tol_values,
        #  'C': c_values,
        # 'fit_intercept': bool_values,
        'solver': solver_values,
        # 'multi_class': multi_class_values,
        # 'warm_start': bool_values,
        # 'max_iter': [2000],
        'random_state': random_state_values
    }]

kNeighborsParameters = [
    {
        # 'n_neighbors': n_neighbors_values,
        'weights': weight_values,
        'algorithm': algorithm_values,
        # 'leaf_size': leaf_size_values
    }]

decisionTreeParameters = [
    {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': max_depth_values,
        # 'min_samples_split': min_samples_split_values,
        # 'min_samples_leaf': min_samples_leaf_values,
        # 'ccp_alpha': alpha_values,
        'random_state': random_state_values}]

randomForestParameters = [
    {'criterion': ['gini', 'entropy'],
     # 'max_depth': max_depth_values,
     # 'min_samples_split': min_samples_split_values,
     # 'min_samples_leaf': min_samples_leaf_values,
     'bootstrap': bool_values,
     'oob_score': bool_values,
     'warm_start': bool_values,
     # 'ccp_alpha': alpha_values,
     'random_state': random_state_values}]

gradientBoostingParameters = [
    {
        'criterion': ['friedman_mse', 'mse', 'mae'],
        # 'loss': ['deviance', 'exponential'],
        # 'max_depth': max_depth_values,
        # 'min_samples_split': min_samples_split_values,
        # 'min_samples_leaf': min_samples_leaf_values,
        'random_state': random_state_values,
        # 'ccp_alpha': alpha_values,
        # 'warm_start': bool_values
    }]

svcParameters = [
    {
        # 'kernel': kernel_values,
        # 'degree': degree_values,
        # 'gamma': gamma_values,
        'shrinking': bool_values,
        # 'probability': bool_values,
        'break_ties': bool_values,
        'random_state': random_state_values,
        # 'decision_function_shape': ['ovo', 'ovr'],
        # 'C': c_values
    }]

gaussianNbParameters = [{'var_smoothing': tol_values}]

multinomialNbParameters = [
    {'alpha': alpha_values,
     'fit_prior': bool_values}]

complementNbParameters = [
    {'alpha': alpha_values,
     'fit_prior': bool_values,
     'norm': bool_values}]

bernoulliNbParameters = [
    {'alpha': alpha_values,
     'fit_prior': bool_values}]

linearDiscriminantParameters = [
    {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': ['auto', None],
        'store_covariance': bool_values,
        'tol': tol_values
    }]

adaBoostParameters = [
    {
        # 'n_estimators': n_neighbors_values,
        # 'learning_rate': alpha_values,
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': random_state_values
    }]

mlpParameters = [{
    'max_iter': [200, 500],
    # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
    # 'solver': ['lbfgs', 'sgd', 'adam'],
    # 'alpha': alpha_values,
    # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
    # 'shuffle': bool_values,
    # 'random_state': random_state_values,
    # 'tol': tol_values,
    # 'learning_rate_init': lambda_values,
    # 'warm_start': bool_values,
    # 'momentum': alpha_values,
    # 'nesterovs_momentum': bool_values,
    # 'early_stopping': bool_values,
    # 'validation_fraction': alpha_values
}]

linearRegressionParameters = [
    {'fit_intercept': bool_values,
     'normalize': bool_values,
     'copy_X': bool_values}]

ridgeParameters = [
    {
        # 'alpha': alpha_values,
        'fit_intercept': bool_values,
        'normalize': bool_values,
        'copy_X': bool_values,
        'tol': tol_values
    }]

bayesianRidgeParameters = [
    {
        'compute_score': bool_values,
        'fit_intercept': bool_values,
        'normalize': bool_values,
        'copy_X': bool_values,
        # 'tol': tol_values
    }]

lassoParameters = [
    {
        # 'alpha': alpha_values,
     'fit_intercept': bool_values,
     'normalize': bool_values,
     'precompute': bool_values,
     'copy_X': bool_values,
     # 'tol': tol_values,
     'warm_start': bool_values,
     'positive': bool_values,
     'selection': selection_values}]

lassoLarsParameters = [
    {
        # 'alpha': alpha_values,
     'fit_intercept': bool_values,
     'normalize': bool_values,
     'copy_X': bool_values,
     'fit_path': bool_values,
     'positive': bool_values}]

tweedieParameters = [
    {
        # 'power': power_values,
     # 'alpha': alpha_values,
     'fit_intercept': bool_values,
     # 'tol': tol_values,
     'warm_start': bool_values}]

svrParameters = [
    {
        # 'tol': tol_values,
        'epsilon': epsilon_values,
        # 'C': c_values,
        'shrinking': bool_values}]

sgdParameters = [
    {
        # 'alpha': alpha_values,
        # 'epsilon': epsilon_values,
        'fit_intercept': bool_values,
        'shuffle': bool_values,
        'warm_start': bool_values,
        'average': bool_values,
        'early_stopping': bool_values}]

kNeighborsRegressionParameters = [
    {
        'p': p_values,
        # 'leaf_size': leaf_size_values,
        'n_neighbors': n_neighbors_values
    }]

decisionTreeRegressionParameters = [
    {'criterion': ["mse", "friedman_mse", "mae"],
     'splitter': ['best', 'random'],
     # 'max_depth': max_depth_values,
     # 'min_samples_split': min_samples_split_values,
     # 'min_samples_leaf': min_samples_leaf_values,
     'ccp_alpha': alpha_values,
     'random_state': random_state_values}]

gaussianProcessorParameters = [
    {
        # 'alpha': alpha_values,
     'normalize_y': bool_values,
     'copy_X_train': bool_values}]
