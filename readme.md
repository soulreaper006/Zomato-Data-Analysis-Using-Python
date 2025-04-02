# Zomato Data Analysis

## Overview
This script performs an exploratory data analysis (EDA) on Zomato restaurant data. It utilizes Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn to analyze and visualize restaurant ratings, votes, online ordering trends, and cost distributions.

## Dependencies
The script requires the following Python libraries:
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib.pyplot` for plotting
- `seaborn` for statistical data visualization

## Code Breakdown

### 1. Load the Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("Zomato data .csv")
print(dataframe.head())
```
The dataset is read from a CSV file into a Pandas DataFrame and the first few rows are displayed.

### 2. Handle Ratings
```python
def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)

dataframe['rate']=dataframe['rate'].apply(handleRate)
print(dataframe.head())
```
- The `rate` column contains ratings in a fraction format (e.g., "4.5/5").
- This function extracts and converts the rating to a float.

### 3. Data Information
```python
dataframe.info()
```
Displays an overview of the dataset, including column types and non-null counts.

### 4. Restaurant Type Distribution
```python
sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")
```
Creates a count plot showing the distribution of different types of restaurants.

### 5. Votes Analysis
```python
grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c='green', marker='o')
plt.xlabel('Type of restaurant', c='red', size=20)
plt.ylabel('Votes', c='red', size=20)
```
- Aggregates votes by restaurant type.
- Plots the total votes received by each category.

### 6. Identify Most Voted Restaurant
```python
max_votes = dataframe['votes'].max()
restaurant_with_max_votes = dataframe.loc[dataframe['votes'] == max_votes, 'name']

print('Restaurant(s) with the maximum votes:')
print(restaurant_with_max_votes)
```
Finds and prints the restaurant(s) with the highest number of votes.

### 7. Online Ordering Trends
```python
sns.countplot(x=dataframe['online_order'])
```
Creates a count plot to show how many restaurants offer online ordering.

### 8. Ratings Distribution
```python
plt.hist(dataframe['rate'], bins=5)
plt.title('Ratings Distribution')
plt.show()
```
Plots a histogram to visualize the distribution of restaurant ratings.

### 9. Approximate Cost Analysis
```python
couple_data=dataframe['approx_cost(for two people)']
sns.countplot(x=couple_data)
```
Displays the distribution of approximate cost for two people using a count plot.

### 10. Boxplot of Online Orders vs Ratings
```python
plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = dataframe)
```
Creates a boxplot to analyze the relationship between online orders and ratings.

### 11. Heatmap of Restaurant Type vs Online Ordering
```python
pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap')
plt.xlabel('Online Order')
plt.ylabel('Listed In (Type)')
plt.show()
```
- Creates a pivot table to analyze how many restaurants of each type offer online ordering.
- Visualizes the relationship using a heatmap.

## Conclusion
This script provides insights into Zomato restaurant data, including:
- Distribution of restaurant types.
- Analysis of ratings and votes.
- Popularity of online ordering.
- Cost and rating distribution.
- Heatmap for online ordering trends.

These analyses help in understanding restaurant trends and customer preferences.
