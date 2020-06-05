import os
import pandas as pd

FILE_PATH = "datasets/diamonds"
FILE_NAME = "diamonds.csv"


def load_diamonds_data(file_path= FILE_PATH):
    csv_path = os.path.join(file_path, FILE_NAME)
    return pd.read_csv(csv_path)


diamonds = load_diamonds_data()
diamonds.drop('Unnamed: 0', inplace=True, axis=1)

# price: price in US dollars
# carat: weight of diamond
# cut: quality of the cut
# color: diamond color J(worst) to D(best)
# clarity: measurement of how clear diamond is (I1(worst), SI2, SI1, VS2, VS1, VVS2, VVS1,
# 1F(best))
# x: length in mm
# y: width
# z: depth
# depth: total depth percentage (z/mean(x,y)) = 2 * x/(x+y)(43--79)
# table: width of top of diamond relative to widest point (43--95)


import numpy as np
np.random.seed(42)

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

diamonds_with_id = diamonds.reset_index()
train_set, test_set = split_train_test_by_id(diamonds_with_id, 0.2, "index")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(diamonds, test_size=0.2, random_state=42)

#print(len(train_set), "train +", len(test_set), "test")

import numpy as np
np.random.seed(42)
diamonds['carat_cat'] = pd.cut(diamonds['carat'],
                              bins=[0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, np.inf],
                              labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# diamonds['carat_cat'].value_counts()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(diamonds, diamonds["carat_cat"]):
    strat_train_set = diamonds.loc[train_index]
    strat_test_set = diamonds.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["carat_cat"], axis=1, inplace=True)

#create copy of train set
diamonds = strat_train_set.copy()

diamonds['volume'] = diamonds['x'] * diamonds['y'] * diamonds['z']
corr_matrix = diamonds.corr()
corr_matrix['price'].sort_values(ascending=False)

#create copy of clean training set without labels
#we don't want to apply transformations to target values
diamonds = strat_train_set.drop("price", axis=1)
diamond_labels = strat_train_set["price"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
diamonds_num = diamonds.drop(['color', 'cut', 'clarity'], axis=1)
imputer.fit(diamonds_num)

X = imputer.transform(diamonds_num)
diamonds_tr = pd.DataFrame(X, columns=diamonds_num.columns, index=diamonds.index)

diamonds_tr = pd.DataFrame(X, columns=diamonds_num.columns, index=diamonds_num.index)
# print(diamonds_tr.head())

#transform the training set
X = imputer.transform(diamonds_num)
diamonds_tr = pd.DataFrame(X, columns=diamonds_num.columns, index=diamonds.index)

# handling text and categorical attributes
# use Scikit-Learn's transformer LabelEncoder to convert text labels to numbers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
diamonds_cut_cat = diamonds['cut']
diamonds_cut_cat_encoded = encoder.fit_transform(diamonds_cut_cat)
# print(diamonds_cut_cat_encoded)

# transform text categories to integer categories then from integer categories to one-hot vectors
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
diamonds_cut_cat_1hot = encoder.fit_transform(diamonds_cut_cat)

diamonds_cut_cat = diamonds[['cut']]

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
diamonds_cut_cat_encoded = ordinal_encoder.fit_transform(diamonds_cut_cat)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

cut_cat_encoder = OneHotEncoder()
diamonds_cut_cat_1hot = cut_cat_encoder.fit_transform(diamonds_cut_cat)

diamonds_cut_cat_1hot.toarray()

diamonds_cut_cat = diamonds[["cut"]]
diamonds_cut_cat_encoded = ordinal_encoder.fit_transform(diamonds_cut_cat)
cut_cat_encoder = OneHotEncoder()
diamonds_cut_cat_1hot = cut_cat_encoder.fit_transform(diamonds_cut_cat)
diamonds_cut_cat_1hot.toarray()

ordinal_encoder = OrdinalEncoder()
diamonds_color_cat = diamonds[["color"]]
diamonds_color_cat_encoded = ordinal_encoder.fit_transform(diamonds_color_cat)
diamonds_color_cat_encoded[:10]
color_cat_encoder = OneHotEncoder()
diamonds_color_cat_1hot = color_cat_encoder.fit_transform(diamonds_color_cat)
diamonds_color_cat_1hot.toarray()

diamonds_clarity_cat = diamonds[["clarity"]]
diamonds_clarity_cat_encoded = ordinal_encoder.fit_transform(diamonds_clarity_cat)
clarity_cat_encoder = OneHotEncoder()
diamonds_clarity_cat_1hot = clarity_cat_encoder.fit_transform(diamonds_clarity_cat)
diamonds_clarity_cat_1hot.toarray()

#scikit-learn's FunctionTransformer class creates a transformer based on a transformation function
from sklearn.base import BaseEstimator, TransformerMixin

x_ix, y_ix, z_ix = [
    list(diamonds.columns).index(col)
    for col in ("x", "y", "z")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_volume=True):
        self.add_volume = add_volume
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        volume = X[:, x_ix] * X[:, y_ix] * X[:, z_ix]
        return np.c_[X, volume]

# Create a class to select numerical or categorical columns
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.preprocessing import FunctionTransformer


def add_extra_features(X, add_volume=True):
    if add_volume:
        volume = X[:, x_ix] * X[:, y_ix] * X[:, z_ix]
    return np.c_[X, volume]


attr_adder = FunctionTransformer(add_extra_features, validate=False, kw_args={"add_volume": True})
diamonds_extra_attribs = attr_adder.fit_transform(diamonds.values)

diamonds_extra_attribs = pd.DataFrame(
    diamonds_extra_attribs,
    columns=list(diamonds.columns) + ['volume'],
    index=diamonds.index)
# print(diamonds_extra_attribs.head())

#build pipeline for processing numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

x_ix, y_ix, z_ix = [
    list(diamonds_num.columns).index(col)
    for col in ("x", "y", "z")]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
])
diamonds_num_tr = num_pipeline.fit_transform(diamonds_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(diamonds_num)
cat_attribs = ["color", "cut", "clarity"]

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs),
])
diamonds_prepared = full_pipeline.fit_transform(diamonds)

diamonds_num = diamonds.drop(['color', 'cut', 'clarity'], axis=1)
num_attribs = list(diamonds_num)
cat_attribs = ['color', 'cut', 'clarity']

old_num_pipeline = Pipeline([
    ('selector', OldDataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder(add_volume=True)),
    ('std_scaler', StandardScaler()),
])
old_cat_pipeline = Pipeline([
    ('selector', OldDataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])

from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', old_num_pipeline),
    ('cat_pipeline', old_cat_pipeline),
])
old_diamonds_prepared = old_full_pipeline.fit_transform(diamonds)

np.allclose(diamonds_prepared, old_diamonds_prepared)

#Select and train a model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(diamonds_prepared, diamond_labels)

# try full preprocessing pipeline on a few training instances
some_data = diamonds.iloc[:5]
some_labels = diamond_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

from sklearn.metrics import mean_squared_error

diamonds_predictions = lin_reg.predict(diamonds_prepared)
lin_mse = mean_squared_error(diamond_labels, diamonds_predictions)
lin_rmse = np.sqrt(lin_mse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(diamond_labels, diamonds_predictions)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(diamonds_prepared, diamond_labels)

diamonds_predictions = tree_reg.predict(diamonds_prepared)
tree_mse = mean_squared_error(diamond_labels, diamonds_predictions)
tree_rmse = np.sqrt(tree_mse)

# use cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, diamonds_prepared, diamond_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("scores:", scores)
    print("mean:", scores.mean())
    print("standard deviation:", scores.std())

lin_scores = cross_val_score(lin_reg, diamonds_prepared, diamond_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(diamonds_prepared, diamond_labels)

diamonds_predictions = forest_reg.predict(diamonds_prepared)
forest_mse = mean_squared_error(diamond_labels, diamonds_predictions)
forest_rmse = np.sqrt(forest_mse)

forest_scores = cross_val_score(forest_reg, diamonds_prepared, diamond_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

scores = cross_val_score(lin_reg, diamonds_prepared, diamond_labels, scoring="neg_mean_squared_error", cv=10)
# pd.Series(np.sqrt(-scores)).describe()

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                          return_train_score=True)
grid_search.fit(diamonds_prepared, diamond_labels)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=100),
    'max_features': randint(low=1, high=6),
}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                               n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42)
rnd_search.fit(diamonds_prepared, diamond_labels)

feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ['volume']
cat_encoder = full_pipeline.named_transformers_["cat"]
cut_cat_one_hot_attribs = list(cut_cat_encoder.categories_[0])
color_cat_one_hot_attribs = list(color_cat_encoder.categories_[0])
clarity_cat_one_hot_attribs = list(clarity_cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cut_cat_one_hot_attribs + color_cat_one_hot_attribs + clarity_cat_one_hot_attribs

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('price', axis=1)
y_test = strat_test_set['price'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# compute 95% confidence interval for test RMSE
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m-1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))

def output_prediction(carat, cut, color, clarity, depth, table, x, y, z):
    # create array from user data
    user_data = {'carat': [carat],
                 'cut': [cut],
                 'color': [color],
                 'clarity': [clarity],
                 'depth': [depth],
                 'table': [table],
                 'x': [x],
                 'y': [y],
                 'z': [z]}

    user_df = pd.DataFrame(user_data, columns=['carat', 'cut', 'color', 'clarity', 'depth',
                                               'table', 'x', 'y', 'z'])

    user_test_prepared = full_pipeline.transform(user_df)
    user_prediction = final_model.predict(user_test_prepared)
    return user_prediction[0]

print(output_prediction(0.21, 'Premium', 'E', 'SI1', 59.8, 61, 3.89, 3.84, 2.31))