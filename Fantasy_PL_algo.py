#!/usr/bin/env python
# coding: utf-8

# ## FPL Analysis
#
# This is going to be a mix of descriptive analytics and predictive analytics to help me pick an FPL team for the 2021/2021 Premier League season. This model should work better than how i have been normally picking my teams, with my heart. This little project also serves as a little break from a difficult job search to use the skills I have learnt in class on cases that i enjoy doing.

# ## Part 1: Data Exploration
# <h3> Package imports, peaking into data and checking for missing values
#

# In[1]:


# Importing libraries
import pandas as pd  # Data science essentials
import matplotlib.pyplot as plt  # Essential graphical output
import seaborn as sns  # Enhanced graphical output
import numpy as np  # Mathematical essentials
import statsmodels.formula.api as smf  # Regression modeling
from os import listdir  # Look inside file directory
from sklearn.model_selection import train_test_split  # Split data into training and testing data
import gender_guesser.detector as gender  # Guess gender based on (given) name
from sklearn.linear_model import LinearRegression  # OLS Regression
import sklearn.linear_model  # Linear models
from sklearn.neighbors import KNeighborsRegressor  # KNN for Regression
from sklearn.preprocessing import StandardScaler  # standard scaler

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Files
all_fpl = '/Users/simbamariwande/Desktop/Python ML- FPL/EPL_Data_Combined.xlsx'
data_20_21_season = '/Users/simbamariwande/Desktop/Python ML- FPL/2021_union.xlsx'
data_19_21_season = '/Users/simbamariwande/Desktop/Python ML- FPL/1921_union.xlsx'

# Importing the dataset
fpl_total = pd.read_excel(io=all_fpl)
fpl_20_21 = pd.read_excel(io=data_20_21_season)
fpl_19_21 = pd.read_excel(io=data_19_21_season)

# <h3> Data Sizes </h3>

# In[2]:


# formatting and printing the dimensions of the dataset
print(f"""
Size of Original Dataset All Data
-----------------------------------------
Observations: {fpl_total.shape[0]}
Features:     {fpl_total.shape[1]}

There are {fpl_total.isnull().any().sum()} missing values 
""")

# formatting and printing the dimensions of the dataset
print(f"""
Size of Original Dataset 20/21 Data
-----------------------------------------
Observations: {fpl_20_21.shape[0]}
Features:     {fpl_20_21.shape[1]}

There are {fpl_20_21.isnull().any().sum()} missing values 
""")

# formatting and printing the dimensions of the dataset
print(f"""
Size of Original Dataset 19/21 Data
-----------------------------------------
Observations: {fpl_19_21.shape[0]}
Features:     {fpl_19_21.shape[1]}

There are {fpl_19_21.isnull().any().sum()} missing values 
""")

# <h2> Descriptive Analytics </h2>
#
#
# <h3> Histograms </h3>

# In[3]:


# Histogram to check distribution of the response variable
sns.displot(data=fpl_19_21,
            x='minutes',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# In[4]:


# Histogram to check distribution of the response variable
sns.displot(data=fpl_19_21,
            x='goals_scored',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# In[5]:


# Histogram to check distribution of the response variable
sns.displot(data=fpl_19_21,
            x='total_points',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# All histograms seem to show non normlaity violating Regression rules
#

# In[6]:


# Filtering Outlier values to get uniformity within the data set

ml_fpl = fpl_19_21[fpl_19_21['total_points'] > 0]  # Subsetting points scorers

# Histogram to check distribution of the response variable
sns.displot(data=ml_fpl,
            x='total_points',
            height=5,
            aspect=2)

# displaying the histogram
plt.show()

# In[7]:


ml_fpl.describe()

# In[8]:


ml_fpl = fpl_19_21.copy()

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# ## Part 2: Trend Based Features
# <h3>Checking the Continuous Data</h3>

# In[9]:


########################
# Visual EDA (Scatterplots)
########################

# setting figure size
fig, ax = plt.subplots(figsize=(10, 8))

# developing a scatterplot
plt.subplot(2, 2, 1)
sns.scatterplot(x=ml_fpl['goals_scored'],
                y=ml_fpl['total_points'],
                color='g')

# adding labels but not adding title
plt.xlabel(xlabel='Goals Scored')
plt.ylabel(ylabel='Total Points')

########################


# developing a scatterplot
plt.subplot(2, 2, 2)
sns.scatterplot(x=ml_fpl['bps'],
                y=ml_fpl['total_points'],
                color='r')

# adding labels but not adding title
plt.xlabel(xlabel='BPS')
plt.ylabel(ylabel='Total Points')

########################


# developing a scatterplot
plt.subplot(2, 2, 3)
sns.scatterplot(x=ml_fpl['creativity'],
                y=ml_fpl['total_points'],
                color='b')

# adding labels but not adding title
plt.xlabel(xlabel='Creativity')
plt.ylabel(ylabel='Total Points')

########################


# developing a scatterplot
plt.subplot(2, 2, 4)
sns.scatterplot(x=ml_fpl['influence'],
                y=ml_fpl['total_points'],
                color='orange')

# adding labels but not adding title
plt.xlabel(xlabel='Influence')
plt.ylabel(ylabel='Total Points')

# cleaning up the layout and displaying the results
plt.tight_layout()
plt.show()

# <ul>Independence is obeyed  </ul>
# <ul>Linearity Law is obeyed  </ul>
# <ul>Normality is better </ul>

# <h2> Top Players in the Past 3 Years </h2>

# In[10]:


# Scatter of players

ml_fpl_top = ml_fpl[ml_fpl['total_points'] > 190]

# Create figure
plt.figure(figsize=(20, 20))

# Plots
ax = sns.scatterplot(data=ml_fpl_top,
                     x='influence',
                     y='total_points',
                     s=20)

# For each point, we add a text inside the bubble
for line in range(0, ml_fpl_top.shape[0]):
    ax.text(ml_fpl_top.influence.iloc[line], ml_fpl_top.total_points.iloc[line] + 2,
            ml_fpl_top.full_name.iloc[line], horizontalalignment='center',
            size='large', color='black', weight='semibold', fontsize=17)

# adding labels but not adding title
plt.xlabel(xlabel='Influence')
plt.ylabel(ylabel='Total Points')

plt.show()

# If you watch the prem, you would know that Eden, Harry, Kevin, Bruno and Mo have been ruling the league, so this graph makes sense. It should also be noted that the graph confirms that the more influencial a player the higher the points they will score, this means that influencial players, even in smaller teams might be very important to the FPL Squad chosen.

# In[11]:


# instantiating a correlation matrix
df_corr = fpl_19_21.corr().round(2)

# setting figure size
fig, ax = plt.subplots(figsize=(15, 15))

# visualizing the correlation matrix
sns.heatmap(df_corr,
            cmap='coolwarm',
            square=True,
            annot=True,
            linecolor='black',
            linewidths=0.5)

# saving and displaying the correlation matrix
plt.tight_layout()
plt.show()

# The correlation matrix is to help with corroborating some of the values obtained in the linear regression. It will serve as a way to check if the coefficients make sense and help with diagnostics later on for this case.

# In[12]:


ml_fpl.columns

# <hr style="height:.9px;border:none;color:#333;background-color:#333;" /><br>
#
# ## Part 3: Model Testing
# <br>

# In[13]:


# making a copy of housing
fpl_explanatory = ml_fpl.copy()

# dropping SalePrice and Order from the explanatory variable set
fpl_explanatory = fpl_explanatory.drop(['first_name', 'second_name', 'full_name', 'Season'], axis=1)

# formatting each explanatory variable for statsmodels
for val in fpl_explanatory:
    print(val, '+')

# In[14]:


# Step 1: build a model
lm_best = smf.ols(formula="""total_points ~ minutes +
                                        goals_scored +
                                        assists +
                                        goals_conceded +
                                        clean_sheets +
                                        red_cards +
                                        yellow_cards +
                                        creativity +
                                        influence +
                                        bps +
                                        ict_index +
                                        bonus  """,
                  data=ml_fpl)

# Step 2: fit the model based on the data
results = lm_best.fit()

# Step 3: analyze the summary output
print(results.summary())

# ## Linear Regression Model

# In[15]:


# preparing explanatory variable data

x_variables = ['minutes', 'goals_scored', 'assists',
               'goals_conceded', 'clean_sheets', 'red_cards',
               'yellow_cards', 'creativity', 'influence',
               'bps', 'ict_index', 'bonus']

fpl_data = ml_fpl[x_variables]

# preparing the target variable
fpl_target = ml_fpl.loc[:, 'total_points']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    fpl_data,
    fpl_target,
    test_size=0.25,
    random_state=219)

# INSTANTIATING a model object
lr = LinearRegression()

# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)

# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)

# SCORING the results
print('OLS Training Score :', lr.score(X_train, y_train).round(4))  # using R-square
print('OLS Testing Score  :', lr.score(X_test, y_test).round(4))  # using R-square

lr_train_score = lr.score(X_train, y_train).round(4)
lr_test_score = lr.score(X_test, y_test).round(4)

# displaying and saving the gap between training and testing
print('OLS Train-Test Gap :', abs(lr_train_score - lr_test_score).round(4))
lr_test_gap = abs(lr_train_score - lr_test_score).round(4)

# zipping each feature name to its coefficient
lr_model_values = zip(fpl_data.columns,
                      lr_fit.coef_.round(decimals=4))

# setting up a placeholder list to store model features
lr_model_lst = [('intercept', lr_fit.intercept_.round(decimals=4))]

# printing out each feature-coefficient pair one by one
for val in lr_model_values:
    lr_model_lst.append(val)

# Making the list a data frame to print later
lr_model_lst = pd.DataFrame(lr_model_lst)

# Naming the Columns
lr_model_lst.columns = ['Variables', 'Coefficients']

# Removing indices for print
lr_model_lst_no_indices = lr_model_lst.to_string(index=False)

# In[16]:


#  Selected Model
print(f"""

The Model selected is OLS Regression 

Model      Train Score      Test Score      Train-Test Gap     Model Size
-----      -----------      ----------      ---------------    ----------
OLS        {lr_train_score}            {lr_test_score}          {lr_test_gap}               {len(lr_model_lst)}


Model Coefficients
----------------------

{lr_model_lst_no_indices}


""")

# ## Points Predictions

# In[17]:


len(ml_fpl.full_name)

# Data prep (player names db )
player_names = ml_fpl.loc[:, 'full_name']

player_names = pd.DataFrame(player_names)

player_names = player_names.drop_duplicates(subset=['full_name'])

len(player_names.full_name)

player_names.loc[:, 'player_id'] = 0

for i, j in player_names.iterrows():
    player_names.loc[i, 'player_id'] = i

# In[18]:


# Pivot tables to get data for inputing
player_averages = pd.pivot_table(data=fpl_19_21, index=['full_name'])

#####Getting points estimates using LR and Ave


# for i, j in player_averages.iterrows():

player_averages.loc[:, 'points_average'] = player_averages.total_points

player_averages.loc[:,
'points_prediction'] = 1.6613 + player_averages.minutes * 0.0192 + player_averages.goals_scored * 2.8559 + player_averages.assists * 2.444 + player_averages.goals_conceded * (
    -0.2364) + player_averages.clean_sheets * 1.6156 - player_averages.red_cards * 3.5394 - player_averages.yellow_cards * 1.5007 - player_averages.creativity * 0.0106 - player_averages.influence * 0.0113 + player_averages.bps * 0.0755 + player_averages.ict_index * 0.0367 + player_averages.bonus * 1.0546

player_averages.head()

# ## Final List

# In[19]:


player_averages = player_averages.sort_values('points_prediction', ascending=False, inplace=False)

top_players = player_averages.iloc[0:100, :]

top_players

# <h3> Imports to perform table joins for positions </h3>

# In[20]:


file = '/Users/simbamariwande/Desktop/Python ML- FPL/clubs_positions.xlsx'
file_2 = '/Users/simbamariwande/Desktop/Python ML- FPL/player_values.xlsx'

clubs_players = pd.read_excel(io=file)
player_values = pd.read_excel(io=file_2)

clubs_players.head()

# <h3> Final DataFrame Upon which a team can be built </h3>

# In[21]:


final_df = pd.merge(top_players, clubs_players, on='full_name')

final_df = final_df.drop(['First_Name', 'Second_Name'], axis=1)

final_df.head()

# <h2> Conclusion </h2>
#
# The Final DF made above is a strong basis to build a team upon. As of last week these players are still in the league.
#
# Be careful to note:
# -The DF only has players who have been in the PL playing in this seasons premier league and doesnt take into account new players.
# -However that may not be an issue, one of the methods this final DF can. be used is to populate with current FPL player values then use a solver to maximise the number of points the team it will choose will make whilst using the position numbers as constraints. This will help with optimising where in the. past 3 years the best positional allocations will be for your budget then from there you could use intuition with a combination of the influences that were seen earlier to be important.
#
#
# That is one way, but since some of my FPL competition is on this timeline i will not say which method i chose for the sake of competition.

