model = XGBRegressor(n_estimators = 1000,
                    max_depth = 6,
                    learning_rate = 0.05,
                    colsample_bytree = 0.5,
                    random_state=1452,
                    )

fit_params = {'verbose': 200, 'early_stopping_rounds': 200}

xgb_oofs, xgb_preds, fi = run_gradient_boosting(model, fit_params, train, test, cat_num_cols)

xgb_preds_t = np.expm1(xgb_preds)
download_preds(xgb_preds_t, file_name = 's9_xgb1.csv')
