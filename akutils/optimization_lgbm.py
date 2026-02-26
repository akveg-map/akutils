# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Optimization for LightGBM
# Author: Timm Nawrocki
# Last Updated: 2024-11-24
# Usage: Must be executed in an Anaconda Python 3.12+ distribution.
# Description: "Optimization for LightGBM" is a set of functions that perform Bayesian optimization on either a LightGBM classifier or regressor.
# ---------------------------------------------------------------------------

# Define a function to calculate the cross validated balanced accuracy score for the classifier
def cross_val_bacc_classifier(estimator, data, all_variables, predictor_all, target_field, stratify_field, group_field):
    # Import packages
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score
    from akutils import determine_optimal_threshold

    # Create inner cv splits
    inner_cv_splits = StratifiedGroupKFold(n_splits=5)
    inner_split = ['inner_split_n']

    # Create an empty data frame to store the inner cross validation splits
    inner_train = pd.DataFrame(columns=all_variables + inner_split)
    inner_test = pd.DataFrame(columns=all_variables + inner_split)

    # Create an empty data frame to store the inner test results
    inner_results = pd.DataFrame(columns=all_variables + inner_split + ['y_abs', 'y_pres'])

    # Create inner cross validation splits
    count = 1
    for train_index, test_index in inner_cv_splits.split(data,
                                                         data[stratify_field[0]].astype('int32'),
                                                         data[group_field[0]].astype('int32')):
        # Split the data into train and test partitions
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        # Insert iteration to train
        train = train.assign(inner_split_n=count)
        # Insert iteration to test
        test = test.assign(inner_split_n=count)
        # Append to data frames
        inner_train = pd.concat([inner_train if not inner_train.empty else None,
                                 train],
                                axis=0)
        inner_test = pd.concat([inner_test if not inner_test.empty else None,
                                test],
                               axis=0)
        # Increase counter
        count += 1
    inner_cv_length = count - 1

    # Reset indices
    inner_train = inner_train.reset_index()
    inner_test = inner_test.reset_index()

    # Iterate through inner cross validation splits
    inner_cv_i = 1
    while inner_cv_i <= inner_cv_length:
        inner_train_iteration = inner_train[inner_train[inner_split[0]] == inner_cv_i].copy()
        inner_test_iteration = inner_test[inner_test[inner_split[0]] == inner_cv_i].copy()

        # Identify X and y inner train and test splits
        X_class_inner = inner_train_iteration[predictor_all].astype(float).copy()
        y_class_inner = inner_train_iteration[target_field[0]].astype('int32').copy()
        X_test_inner = inner_test_iteration[predictor_all].astype(float).copy()

        # Train regressor on the inner train data
        estimator.fit(X_class_inner, y_class_inner)

        # Predict inner test data
        probability_inner = estimator.predict_proba(X_test_inner)

        # Assign predicted values to inner test data frame
        inner_test_iteration = inner_test_iteration.assign(y_abs=probability_inner[:, 0])
        inner_test_iteration = inner_test_iteration.assign(y_pres=probability_inner[:, 1])

        # Add the test results to output data frame
        inner_results = pd.concat([inner_results if not inner_results.empty else None,
                                   inner_test_iteration],
                                  axis=0)

        # Increase n value
        inner_cv_i += 1

    # Calculate the optimal threshold and performance of the presence-absence classification
    threshold, sensitivity, specificity, auc, accuracy = determine_optimal_threshold(
        inner_results['y_pres'],
        inner_results[target_field[0]]
    )

    # Convert probability to presence-absence
    presence_zeros = np.zeros(inner_results['y_pres'].shape)
    presence_zeros[inner_results['y_pres'] >= threshold] = 1

    # Assign binary prediction values to test data frame
    inner_results = inner_results.assign(y_pred=presence_zeros)

    # Calculate negative mean squared error
    y_class_observed = inner_results[target_field[0]].astype('int32').copy()
    y_class_predicted = inner_results['y_pred'].astype('int32').copy()
    bacc = balanced_accuracy_score(y_class_observed, y_class_predicted)

    return bacc


# Define a function to calculate the cross validated negative mean squared error for the regressor
def cross_val_nmse_regressor(estimator, data, all_variables, predictor_all, target_field, stratify_field, group_field):
    # Import packages
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import mean_squared_error

    # Limit data to valid abundance observations
    regress_inner = data[data[target_field[0]] >= 0].copy()

    # Create inner cv splits
    inner_cv_splits = StratifiedGroupKFold(n_splits=5)
    inner_split = ['inner_split_n']

    # Create an empty data frame to store the inner cross validation splits
    inner_train = pd.DataFrame(columns=all_variables + inner_split)
    inner_test = pd.DataFrame(columns=all_variables + inner_split)

    # Create an empty data frame to store the inner test results
    inner_results = pd.DataFrame(columns=all_variables + inner_split + ['y_pred'])

    # Create inner cross validation splits
    count = 1
    for train_index, test_index in inner_cv_splits.split(regress_inner,
                                                         regress_inner[stratify_field[0]].astype('int32'),
                                                         regress_inner[group_field[0]].astype('int32')):
        # Split the data into train and test partitions
        train = regress_inner.iloc[train_index]
        test = regress_inner.iloc[test_index]
        # Insert iteration to train
        train = train.assign(inner_split_n=count)
        # Insert iteration to test
        test = test.assign(inner_split_n=count)
        # Append to data frames
        inner_train = pd.concat([inner_train if not inner_train.empty else None,
                                 train],
                                axis=0)
        inner_test = pd.concat([inner_test if not inner_test.empty else None,
                                test],
                               axis=0)
        # Increase counter
        count += 1
    inner_cv_length = count - 1

    # Reset indices
    inner_train = inner_train.reset_index()
    inner_test = inner_test.reset_index()

    # Iterate through inner cross validation splits
    inner_cv_i = 1
    while inner_cv_i <= inner_cv_length:
        inner_train_iteration = inner_train[inner_train[inner_split[0]] == inner_cv_i].copy()
        inner_test_iteration = inner_test[inner_test[inner_split[0]] == inner_cv_i].copy()

        # Identify X and y inner train and test splits
        X_regress_inner = inner_train_iteration[predictor_all].astype(float).copy()
        y_regress_inner = inner_train_iteration[target_field[0]].astype(float).copy()
        X_test_inner = inner_test_iteration[predictor_all].astype(float).copy()

        # Train regressor on the inner train data
        estimator.fit(X_regress_inner, y_regress_inner)

        # Predict inner test data
        prediction_inner = estimator.predict(X_test_inner)

        # Assign predicted values to inner test data frame
        inner_test_iteration = inner_test_iteration.assign(y_pred=prediction_inner)

        # Add the test results to output data frame
        inner_results = pd.concat([inner_results if not inner_results.empty else None,
                                   inner_test_iteration],
                                  axis=0)

        # Increase n value
        inner_cv_i += 1

    # Calculate negative mean squared error
    y_regress_observed = inner_results[target_field[0]].astype(float).copy()
    y_regress_predicted = inner_results['y_pred'].astype(float).copy()
    nmse = -(mean_squared_error(y_regress_observed, y_regress_predicted))

    return nmse


# Define a function to conduct a cross validation iteration for a LightGBM classifier
def lgbmclassifier_cv(num_leaves, max_depth, learning_rate, n_estimators,
                      min_split_gain, min_child_weight, min_child_samples,
                      subsample, colsample_bytree, reg_alpha, reg_lambda,
                      data, all_variables, predictor_all, target_field, stratify_field, group_field):
    """
    Description: conducts cross validation of a LightGBM regressor with a particular set of hyperparameter values
    Inputs: 'data' -- the covariate data to conduct the model training and validation
            'targets' -- the response data to conduct the model training and validation
            'groups' -- the group data for the cross validation method
            All other inputs are set by other functions
    Returned Value: Returns the cross validation score
    Preconditions: requires pre-processed X and y data
    """

    # Import packages
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    # Define cross validation
    cv_splits = StratifiedGroupKFold(n_splits=5, shuffle=True)

    # Define estimator
    estimator = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        objective='binary',
        class_weight='balanced',
        min_split_gain=min_split_gain,
        min_child_weight=min_child_weight,
        min_child_samples=int(min_child_samples),
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=2,
        importance_type='gain',
        verbosity=-1
    )

    # Define cross validation
    bacc = cross_val_bacc_classifier(estimator,
                                     data,
                                     all_variables,
                                     predictor_all,
                                     target_field,
                                     stratify_field,
                                     group_field)

    return bacc


# Define a function to conduct a cross validation iteration for a LightGBM regressor
def lgbmregressor_cv(num_leaves, max_depth, learning_rate, n_estimators,
                     min_split_gain, min_child_weight, min_child_samples,
                     subsample, colsample_bytree, reg_alpha, reg_lambda,
                     data, all_variables, predictor_all, target_field, stratify_field, group_field):
    """
    Description: conducts cross validation of a LightGBM regressor with a particular set of hyperparameter values
    Inputs: 'data' -- the covariate data to conduct the model training and validation
            'targets' -- the response data to conduct the model training and validation
            'groups' -- the group data for the cross validation method
            All other inputs are set by other functions
    Returned Value: Returns the cross validation score
    Preconditions: requires pre-processed X and y data
    """

    # Import packages
    from lightgbm import LGBMRegressor

    # Define estimator
    estimator = LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        objective='regression',
        min_split_gain=min_split_gain,
        min_child_weight=min_child_weight,
        min_child_samples=int(min_child_samples),
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=2,
        importance_type='gain',
        verbosity=-1
    )

    # Define cross validation
    nmse = cross_val_nmse_regressor(estimator,
                                    data,
                                    all_variables,
                                    predictor_all,
                                    target_field,
                                    stratify_field,
                                    group_field)

    # Return mean score across all cross validation partitions
    return nmse


# Define a function to optimize hyperparameters for a LightGBM classifier
def optimize_lgbmclassifier(init_points, n_iter, data, all_variables, predictor_all, target_field, stratify_field, group_field):
    """
    Description: applies Bayesian optimization to the hyperparameters of a LightGBM classifier
    Inputs: 'data' -- the covariate data to conduct the model training and validation
            'targets' -- the response data to conduct the model training and validation
            'groups' -- the group data for the cross validation method
            'init_points' -- the number of random search iterations to perform initially
            'n_iter' -- the number of Bayesian search iterations to perform
    Returned Value: Returns the hyperparameters from the iteration with the best cross validation performance
    Preconditions: requires pre-processed X and y data
    """

    # Import packages
    from bayes_opt import BayesianOptimization

    # Define a function to return hyperparameters from an optimization iteration
    def lgbmclassifier_params(num_leaves, max_depth, learning_rate, n_estimators,
                              min_split_gain, min_child_weight, min_child_samples,
                              subsample, colsample_bytree, reg_alpha, reg_lambda):
        '''
        Description: returns the hyperparameter values from a cross validation set
        Inputs: All inputs are set by other functions
        Returned Value: Returns a set of hyperparameters
        Preconditions: this function wraps lgbmclassifier_cv
        '''

        return lgbmclassifier_cv(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            data=data,
            all_variables=all_variables,
            predictor_all=predictor_all,
            target_field=target_field,
            stratify_field=stratify_field,
            group_field=group_field
        )

    optimizer = BayesianOptimization(
        f=lgbmclassifier_params,
        pbounds={
            'num_leaves': (5, 200),
            'max_depth': (3, 12),
            'learning_rate': (0.001, 0.2),
            'n_estimators': (50, 100), # UPDATE MAX TO 1000
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (0.001, 1),
            'min_child_samples': (1, 200),
            'subsample': (0.3, 0.9),
            'colsample_bytree': (0.3, 0.9),
            'reg_alpha': (0, 5),
            'reg_lambda': (0, 5)
        },
        random_state=314,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return optimizer.max['params']


# Define a function to optimize hyperparameters for a LightGBM regressor
def optimize_lgbmregressor(init_points, n_iter, data, all_variables, predictor_all, target_field, stratify_field,
                           group_field):
    """
    Description: applies Bayesian optimization to the hyperparameters of a LightGBM regressor
    Inputs: 'data' -- the covariate data to conduct the model training and validation
            'targets' -- the response data to conduct the model training and validation
            'groups' -- the group data for the cross validation method
            'init_points' -- the number of random search iterations to perform initially
            'n_iter' -- the number of Bayesian search iterations to perform
    Returned Value: Returns the hyperparameters from the iteration with the best cross validation performance
    Preconditions: requires pre-processed X and y data
    """

    # Import packages
    from bayes_opt import BayesianOptimization

    # Define a function to return hyperparameters from an optimization iteration
    def lgbmregressor_params(num_leaves, max_depth, learning_rate, n_estimators,
                             min_split_gain, min_child_weight, min_child_samples,
                             subsample, colsample_bytree, reg_alpha, reg_lambda):
        '''
        Description: returns the hyperparameter values from a cross validation set
        Inputs: All inputs are set by other functions
        Returned Value: Returns a set of hyperparameters
        Preconditions: this function wraps lgbmregressor_cv
        '''

        return lgbmregressor_cv(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            data=data,
            all_variables=all_variables,
            predictor_all=predictor_all,
            target_field=target_field,
            stratify_field=stratify_field,
            group_field=group_field
        )

    optimizer = BayesianOptimization(
        f=lgbmregressor_params,
        pbounds={
            'num_leaves': (5, 200),
            'max_depth': (3, 12),
            'learning_rate': (0.001, 0.2),
            'n_estimators': (50, 100), # UPDATE MAX TO 1000
            'min_split_gain': (0.001, 0.1),
            'min_child_weight': (0.001, 1),
            'min_child_samples': (1, 200),
            'subsample': (0.3, 0.9),
            'colsample_bytree': (0.3, 0.9),
            'reg_alpha': (0, 5),
            'reg_lambda': (0, 5)
        },
        random_state=314,
        verbose=2
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return optimizer.max['params']
