#Load Data
test = pd.read_csv('/content/MyDrive/My Drive/Data Science/Analytics Vidhya/Hacklive 2 Guided Community Hackathon/Data/test.csv')
train = pd.read_csv('/content/MyDrive/My Drive/Data Science/Analytics Vidhya/Hacklive 2 Guided Community Hackathon/Data/train.csv')

#Define Values for Model
train_new = train[[ID_COL, TARGET_COL]]
train_new[TARGET_COL] = np.log1p(train_new[TARGET_COL])

test_new = test[[ID_COL]]

train_new['lgb'] = lgb_oofs
test_new['lgb'] = lgb_preds

train_new['cb'] = cb_oofs
test_new['cb'] = cb_preds

train_new['xgb'] = xgb_oofs
test_new['xgb'] = xgb_preds

#Define Features
features = [c for c in train_new.columns if c not in [ID_COL, TARGET_COL]]

#LGBM Model
model = LGBMRegressor(n_estimators = 5000,
                        learning_rate = 0.05,
                        colsample_bytree = 0.65,
                        metric = 'None',
                        )
fit_params = {'verbose': 300, 'early_stopping_rounds': 200, 'eval_metric': 'rmse'}

ens_oofs, ens_preds, fi = run_gradient_boosting(model, fit_params, train_new, test_new, features)

ens_preds_t = np.expm1(ens_preds)
download_preds(ens_preds_t, file_name = 's13_ens1.csv')
