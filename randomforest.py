import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor
import sklearn.metrics as mt
data = pd.read_csv("C:/Users/ilhan/Downloads/Advertising.csv")
veri = data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

dtmodel=DecisionTreeRegressor(random_state=0,max_leaf_nodes=19,min_samples_split=9)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)


bgmodel=BaggingRegressor(random_state=0,n_estimators=18)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)


rfmodel=RandomForestRegressor(random_state=0,max_depth=8,max_features=4,n_estimators=19)
rfmodel.fit(X_train,y_train)
rftahmin=rfmodel.predict(X_test)

r2dt = mt.r2_score(y_test,dttahmin)
r2bg = mt.r2_score(y_test,bgtahmin)
r2rf = mt.r2_score(y_test,rftahmin)

rmsedt = mt.mean_squared_error(y_test,dttahmin,squared=False)
rmsebg = mt.mean_squared_error(y_test,bgtahmin,squared = False)
rmserf = mt.mean_squared_error(y_test,rftahmin,squared = False)


print("Karar agaci Modeli R2: {} Rmse: {}".format(r2dt,rmsedt))
print("Bagging Modeli R2: {} Rmse: {}".format(r2bg,rmsebg))
print("Random Forest Modeli: R2 {} Rmse: {}".format(r2rf,rmserf))


"""
dtparametreler={"min_samples_split":range(2,20),"max_leaf_nodes":range(2,20)}
dtgrid = GridSearchCV(estimator=dtmodel,param_grid=dtparametreler,cv=10)
dtgrid.fit(X_train,y_train)
print(dtgrid.best_params_)

bgparametreler = {"n_estimators":range(2,20)}
bggrid = GridSearchCV(estimator=bgmodel,param_grid=bgparametreler,cv=10)
bggrid.fit(X_train,y_train)
print(bggrid.best_params_)

rfparametreler={"max_depth":range(2,20),"max_features":range(2,20),"n_estimators":range(2,20)}
rfgrid = GridSearchCV(estimator=rfmodel,param_grid=rfparametreler,cv=10)
rfgrid.fit(X_train,y_train)
print(rfgrid.best_params_)


"""

#'max_leaf_nodes': 19, 'min_samples_split': 9
#n_estimators': 18
#max_depth': 8, 'max_features': 4, 'n_estimators': 19