model = CatBoostRegressor(n_estimators = 3000,
                       learning_rate = 0.01,
                       rsm = 0.4, ## Analogous to colsample_bytree
                       random_state=2054,
                       )

fit_params = {'verbose': 200, 'early_stopping_rounds': 200}

cb_oofs, cb_preds, fi = run_gradient_boosting(model, fit_params, train, test, cat_num_cols)

cb_preds_t = np.expm1(cb_preds)
download_preds(cb_preds_t, file_name = 's8_cb1.csv')
