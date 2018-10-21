import pandas as pd
import numpy as np
# medium.com/i-like-big-data-and-i-cannot-lie/how-i-scored-in-the-top-9-of-kaggles-titanic-machine-learning-challenge-243b5f45c8e9
pd.set_option('display.max_colwidth', 6000)
pd.set_option('display.max_rows', 60000)
pd.set_option('display.max_columns', 6000)
output = r'C:\Users\capco\Dropbox\My-Kaggle\titanic\titanic3'
train = pd.read_csv(r'C:\Users\capcom\Dropbox\02-Data\Kaggle_Titanic\train_titanic.csv')
test = pd.read_csv(r'C:\Users\capcom\Dropbox\02-Data\Kaggle_Titanic\test_titanic.csv')
# print(train.columns)
# print(test.columns)

y = train.pop('Survived')
x = pd.concat([train,test], axis=0, ignore_index=True)

# Drop PassengerId 
x = x.drop(['PassengerId'], axis=1)

# Name ###########################################
# Method 1
# x['Title_extract'] = x['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# print(x['Title_extract'].unique())
# titles1 = {'Mr': 1,'Mrs': 2,'Miss': 3,'Master': 4,'Don': 5,'Rev':6,'Dr': 7,'Mme': 8,'Ms': 9,'Major': 10,\
# 			'Lady': 11,'Sir': 12,'Mlle': 13,'Col': 14,'Capt': 15,'Countess': 16,'Jonkheer': 17, 'Dona': 18}
# x['Title_extract'] = x['Title_extract'].map(titles1)
# Method 2: medium top9% solution
x['Title_lambda'] = x.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
# print(x['Title_lambda'].unique())
normalized_titles_example = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}
normalized_titles1 = {
    "Capt":       "1",
    "Col":        "1",
    "Major":      "1",
    "Jonkheer":   "2",
    "Don":        "2",
    "Sir" :       "2",
    "Dr":         "3",
    "Rev":        "4",
    "the Countess":"2",
    "Dona":       "2",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}
x['Title_lambda'] = x['Title_lambda'].map(normalized_titles_example)

del x['Name']
## Sex #########################################################
x['Sex'] = x['Sex'].map({'male': 1, 'female': 0})

## Age: youtube vid ############################################
# x['Age'] = x['Age'].fillna(x['Age'].median()).astype(int)

# Method 2: medium top9% solution
grouped = x.groupby(['Sex','Pclass', 'Title_lambda'])  
print(grouped.Age.median())
x['Age'] = grouped.Age.apply(lambda z: z.fillna(z.median()).astype(int))

# Method 3: medium score 95% solution 
# for dataset in x:
#     mean = x["Age"].mean()
#     std = x["Age"].std()
#     is_null = dataset["Age"].isnull().sum()
#     # compute random numbers between the mean, std and is_null
#     rand_age = np.random.randint(mean - std, mean + std, size = is_null)
#     # fill NaN values in Age column with random values generated
#     age_slice = dataset["Age"].copy()
#     age_slice[np.isnan(age_slice)] = rand_age
#     dataset["Age"] = age_slice
#     dataset["Age"] = x["Age"].astype(int)

## Embarked: medium score 95% solution #########################
# embarked_mode = train_df['Embarked'].mode()
# data = [train_df, test_df]
# for dataset in data:
#     dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)

## Ticket ######################################################
# x['Ticket_0'] = x['Ticket'].astype(str).str.split(' ').get(0)
# print(x['Ticket'].value_counts())
# print(x.head())

print(x.head())

## Dummies ###################################################


# Models #####################################################
def randon_forests():
	model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
	model.fit(X_train, y_train)
	get_feature_importances_series(model, X_train)
	print(model.oob_score_)
	
	y_oob_score = model.oob_score_
	y_oob_prediction = model.oob_prediction_

