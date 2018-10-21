import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
pd.set_option('display.max_colwidth', 6000)
pd.set_option('display.max_colwidth', 6000)
pd.set_option('display.max_rows', 60000)
pd.set_option('display.max_columns', 6000)

	# print(x[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	# print(x[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	# print(x[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	# print(x[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	# # g = sns.FacetGrid(x, col='Survived')
	# # g.map(plt.hist, 'Age', bins=20)

	# # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
	# grid = sns.FacetGrid(x, col='Survived', row='Pclass', height=2.2, aspect=1.6)
	# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
	# grid.add_legend()

	# # grid = sns.FacetGrid(train_df, col='Embarked')
	# grid = sns.FacetGrid(x, row='Embarked', height=2.2, aspect=1.6)
	# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
	# grid.add_legend()

	# # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
	# grid = sns.FacetGrid(x, row='Embarked', col='Survived', height=2.2, aspect=1.6)
	# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
	# grid.add_legend()
	# plt.show()

def f(df):
	for sex in df['Sex']:
		if sex == 1:
			df['Age'].fillna(sex.median(), inplace=True)
		if sex == 0:
			df['Age'].fillna(sex.median(), inplace=True)


if __name__ == '__main__':
	np.random.seed(10)
	output = r'C:\Users\capco\Dropbox\My-Kaggle\titanic\titanic1'
	train_df = pd.read_csv(r'C:\Users\capcom\Dropbox\02-Data\Kaggle_Titanic\train_titanic.csv')
	test_df = pd.read_csv(r'C:\Users\capcom\Dropbox\02-Data\Kaggle_Titanic\test_titanic.csv')
	x = pd.concat([train_df, test_df], axis=0, sort=True)
	y = x.pop('Survived')

	# Extract title from names.
	x['Title'] = x['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
	titles = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Don': 5, 'Rev':6,\
				 'Dr': 7, 'Mme': 8, 'Ms': 9, 'Major': 10, 'Lady': 11, 'Sir': 12,\
				 'Mlle': 13, 'Col': 14, 'Capt': 15, 'Countess': 16, 'Jonkheer': 17, 'Dona': 18}
	x['Title'] = x['Title'].map(titles)
	# print(x['Title'].value_counts(dropna=False))

	# del x['PassengerId']
	# del x['Name']
	x = x.drop(['PassengerId','Name'], axis=1)
	# Binarize sex feature.
	gender = {'male': 0, 'female': 1}
	x['Sex'] = x['Sex'].map(gender)

	# Engineer Embarked Col.
	# print(x['Embarked'].value_counts(dropna=False))
	x['Embarked'] = x['Embarked'].fillna('S')
	embarked = {'S': 1, 'C': 2, 'Q': 3}
	x['Embarked'] = x['Embarked'].map(embarked).astype(int)

	# Engineer Pclass
	# x['Pclass'] = x['Pclass']

	# Age Col
	avg_age = x['Age'].mean()
	print(f'avg age {avg_age}')
	med_age = x['Age'].median()
	print(f'Median age {med_age}')	
	print('-' * 40)

	avg_age_female = x[x['Sex'] == 1]
	fem_avg = avg_age_female['Age'].mean()
	print(f'Female Average age: {fem_avg}')
	fem_med = avg_age_female['Age'].median()
	print(f'Female median age: {fem_med}')
	print('-' * 40)

	avg_age_male = x[x['Sex'] == 0]
	fem_avg = avg_age_male['Age'].mean()
	print(f'male Average age: {fem_avg}')
	fem_med = avg_age_female['Age'].median()
	print(f'male median age: {fem_med}')
	print('-' * 40)
	# Amount of Nans: this is from kaggle.com/omarelgabry/a-journey-through-titanic
	# count_nan_age_test = x["Age"].isnull().sum()
	# print(count_nan_age_test)

	x['Age'] = x['Age'].fillna(27).astype(int)

	# Family Col
	x['Family'] = x['Parch'] + x['SibSp']
	x['Family'].loc[x['Family'] > 0] = 1
	x['Family'].loc[x['Family'] == 0] = 0
	# x.to_csv('test3.csv', index=False)
	x = x.drop(['Parch','SibSp'], axis=1)
	print(x.head())



