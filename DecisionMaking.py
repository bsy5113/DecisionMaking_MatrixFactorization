# Decision Making With Matrices


#%% Input people's preference
"""
On a scale of 1-10
'distance': 10 being the closest
'novelty': 10 being looking for something new and creative
'cost': 10 being the most important
'rating': 10 being how important the rating is to the person
'vegetarian': 10 being the most important factor
"""
people = {'Jane': {'distance':2,
                  'novelty':4,
                  'cost':5,
                  'rating':8,
                  'vegetarian':1},
        'David': {'distance':7,
                  'novelty':8,
                  'cost':9,
                  'rating':3,
                  'vegetarian':2},
        'Rachel': {'distance':2,
                  'novelty':3,
                  'cost':8,
                  'rating':3,
                  'vegetarian':2},
        'Bo': {'distance':6,
                  'novelty':6,
                  'cost':3,
                  'rating':8,
                  'vegetarian': 6},
        'Lucas': {'distance':9,
                  'novelty':8,
                  'cost':1,
                  'rating':8,
                  'vegetarian': 6},
        'Katelyn': {'distance':2,
                  'novelty':1,
                  'cost':1,
                  'rating':4,
                  'vegetarian':1},
        'Emma': {'distance':3,
                  'novelty':2,
                  'cost':5,
                  'rating':9,
                  'vegetarian':9},
        'Sophia': {'distance':1,
                  'novelty':1,
                  'cost':9,
                  'rating':2,
                  'vegetarian':1},
        'Nate': {'distance':7,
                  'novelty':7,
                  'cost':6,
                  'rating':7,
                  'vegetarian':5},
        'Sam': {'distance':3,
                  'novelty':4,
                  'cost':3,
                  'rating':4,
                  'vegetarian':3}
                }
                 
#%%
# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
import numpy as np
preference_P = ['distance','novelty','cost','rating','vegetarian']
M_people=np.array([[people[i][preference_P] for preference_P in people[i]] for i in people])

# Resulting numpy array as below
"""
M_people = 
array([[2, 4, 5, 8, 1],
       [7, 8, 9, 3, 2],
       [2, 3, 8, 3, 2],
       [6, 6, 3, 8, 6],
       [9, 8, 1, 8, 6],
       [2, 1, 1, 4, 1],
       [3, 2, 5, 9, 9],
       [1, 1, 9, 2, 1],
       [7, 7, 6, 7, 5],
       [3, 4, 3, 4, 3]])
"""
#%%
# Next you collected data from an internet website. You got the following information.
"""
On a scale of 1-10
'distance': 10 being the closest
'novelty': 10 being new, creative
'cost': 10 being cheaptest
'rating': 10 being the best restaurant overall
'vegetarian': 10 being having many vegetarian options
"""
restaurants  = {'Flacos':{'distance': 3, 
                        'novelty': 3,
                        'cost': 9,
                        'rating': 4,
                        'vegetarian': 9},
                'Texas Roadhouse':{'distance': 2, 
                        'novelty': 5,
                        'cost': 7,
                        'rating': 7,
                        'vegetarian': 3},
                'Torchys taco':{'distance': 9, 
                        'novelty': 3,
                        'cost': 9,
                        'rating': 4,
                        'vegetarian': 6},
                'Rudys':{'distance': 3, 
                        'novelty': 4,
                        'cost': 8,
                        'rating': 8,
                        'vegetarian': 5},
                'Sushi Masa':{'distance': 1, 
                        'novelty': 7,
                        'cost': 6,
                        'rating': 9,
                        'vegetarian': 7},
                'Seoul Garden':{'distance': 3, 
                        'novelty': 7,
                        'cost': 5,
                        'rating': 8,
                        'vegetarian': 7},
                'McDonalds':{'distance': 7, 
                        'novelty': 2,
                        'cost': 9,
                        'rating': 4,
                        'vegetarian': 2},
                'Chickflia':{'distance': 6, 
                        'novelty': 4,
                        'cost': 8,
                        'rating': 8,
                        'vegetarian': 4},
                'BlackBear':{'distance': 5, 
                        'novelty': 4,
                        'cost': 6,
                        'rating': 8,
                        'vegetarian': 2},
                'TacoBell':{'distance': 4, 
                        'novelty': 1,
                        'cost': 9,
                        'rating': 3,
                        'vegetarian': 3}}

#%%
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
preference_R = ['distance', 'novelty', 'cost', 'rating', 'vegetarian']
M_restaurants=np.array([[restaurants[i][preference_R] for preference_R in restaurants[i]] for i in restaurants])

# Resulting numpy array for restaurants
"""
M_restaurant = 
array([[3, 3, 9, 4, 9],
       [2, 5, 7, 7, 3],
       [9, 3, 9, 4, 6],
       [3, 4, 8, 8, 5],
       [1, 7, 6, 9, 7],
       [3, 7, 5, 8, 7],
       [7, 2, 9, 4, 2],
       [6, 4, 8, 8, 4],
       [5, 4, 6, 8, 2],
       [4, 1, 9, 3, 3]])
"""

#%%
# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.
"""
Linear combination is multiplying two vectors to get a resultant vector. Using this resultant vector,
we can draw insightful information such as each restaurant preference by per person as in this practice. 
By summing columns we can determine the restaurant preference as a group also. 
"""

#%%
# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 
# Jane 

Jane_ranking=np.dot(M_restaurants,M_people[0].transpose())
"""
The resulting linear combination of the two matrices : array([104, 118, 113, 131, 139, 130, 101, 136, 122,  84])
=> Each entry means the score for each restaurant for Jane specifically. 
"""


#%%
# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 
M_usr_x_rest=np.dot(M_restaurants,M_people.transpose())
"""
M_usr_x_rest=
array([[104, 118, 113, 131, 139, 130, 101, 136, 122,  84],
       [156, 144, 192, 159, 158, 160, 162, 178, 149, 132],
       [117, 102, 123, 116, 112, 105, 108, 120,  98,  98],
       [149, 137, 167, 160, 180, 181, 125, 172, 148,  99],
       [146, 139, 182, 161, 185, 194, 132, 182, 159,  95],
       [ 43,  47,  52,  55,  58,  57,  43,  60,  54,  33],
       [177, 141, 168, 174, 191, 183, 124, 174, 143, 113],
       [104,  87, 107, 100,  87,  78, 100, 102,  81,  95],
       [169, 155, 196, 178, 190, 191, 155, 194, 165, 125],
       [ 91,  84, 100,  96, 106, 105,  78, 102,  87,  64]])

=> Here each row represents a restaurant and each column represents an individual person. 
Thus each entry in columns represents scores of each restaurant by each person. 

"""

#%%
# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?
score_sum=np.sum(M_usr_x_rest, axis=1)
"""

array([1256, 1154, 1400, 1330, 1406, 1384, 1128, 1420, 1206,  938])
=> The entries represent the total sum of scores for each restaurant for all the people. 

"""


#%%
from scipy.stats import rankdata


# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  
M_usr_x_rest_rank=M_usr_x_rest.argsort(axis=0).argsort(axis=0) + 1

# Optimal restaurant choice for all people
rank_sum=np.sum(M_usr_x_rest_rank, axis=1)

# Rank based on summing rankings
rank_ranking=len(rank_sum)-rankdata(rank_sum,method='ordinal')+1

# Rank based on summing scores
score_ranking=len(score_sum)-rankdata(score_sum,method='ordinal')+1

"""
Score aggregation method => array([ 6,  8,  3,  5,  2,  4,  9,  1,  7, 10])
Ranking aggregation method => array([ 6,  9,  3,  5,  2,  4,  8,  1,  7, 10])
*** Here 10 being the lowest rank, 1 being the highest rank
"""

#%%
# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?
"""
It's the difference between score and ranking aggregation methods. Score aggregation methods takes into account
the magnitude of difference among scores, however, ranking only shows the index of the sorted final ranking. 
Thus ranking aggregation method is NOT fair in the sense that it ignores the difference of scores in each person's
restaurant preference. In the real world problems, ranking aggregation is mostly not fit in the same sense. 
"""

#%%
# How should you preprocess your data to remove this problem. 
"""
if we normalize the score to 0 and 1 before we multiply the two vectors M_people and M_restaurant,
we can minimize the impact of the magnitude difference among scores to the resultant vector and thus ranking. 
"""

#%%
# Find user profiles that are problematic, explain why?
"""
Here, I am going to calculate the variance of each person and see how much variant their own 
scoring is and we can use this to compare with the rest of the group members. 
This should give an idea on how certain individual's scoring is more extreme than the others, 
affect the ranking and thus be problematic. 

"""
variance_distribution=list(np.var(M_usr_x_rest,axis=0))
people_list=list(people.keys())

variance_list=dict(zip(people_list, variance_distribution))

"""
variance_list=
{'Jane': 275.96, 'David': 252.4, 'Rachel': 73.88999999999999, 'Bo': 610.1600000000001, 
'Lucas': 833.45, 'Katelyn': 65.36, 'Emma': 637.5600000000001, 'Sophia': 92.89, 
'Nate': 462.56000000000006, 'Sam': 161.01}
As seen in the dictionary above, Lucas and Emma had the most variance while Rachel and Katelyn 
had the least variance. This means Lucas and Emma's input might be problematic 
"""


#%%
# Think of two metrics to compute the disatistifaction with the group.  
"""
Measure of score separations for each restaurant indicates disatisfication among the group. 
One way to do this is to calculate standard deviation. Another way to do this is to calculate interquartile range.  
The bigger these two numbers, the more disatisfaction there are within the group. 
"""
STDEV=np.std(M_usr_x_rest,axis=1)

q75, q25 = np.percentile(M_usr_x_rest,[25,75],axis=1)
IQR = abs(q75 - q25)

"""
RESULT:

STDEV=array([39.29427439, 32.74507597, 45.35195696, 38.37968212, 45.29944812,47.6491343 , 33.5463858 , 42.38867773, 36.26348025, 27.44011662])
IQR=array([50.25, 49.75, 70.  , 56.75, 76.25, 77.5 , 30.  , 70.5 , 59.  ,22.75])
"""

rest_list=list(restaurants.keys())
STDEV_rest=dict(zip(rest_list,STDEV.tolist()))
IQR_rest=dict(zip(rest_list,IQR.tolist()))
import operator

# Finding the restaurant with the highest standard deviation and IQR number
STDEV_result=max(STDEV_rest.items(), key=operator.itemgetter(1))[0]
IQR_result=max(IQR_rest.items(), key=operator.itemgetter(1))[0]

print("Standard deviation shows that {} restaurant will result the most disatisfaction".format(STDEV_result))
print("IQR shows that {} restaurant will result the most disatisfaction".format(IQR_result))


#%%
# Should you split in two groups today? 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

"""
Here, I will transform the people matrix into 2D matrix using PCA so we can conduct 
K means clustering based on people's preference on restaurants.
"""
pca=PCA(n_components=2)
pca_people=pca.fit_transform(M_people)
kmeans=KMeans(n_clusters=2).fit(pca_people)
kmean_group=kmeans.predict(pca_people)
centers=kmeans.cluster_centers_
labels=kmeans.labels_

plt.scatter(pca_people[:, 0], pca_people[:, 1], c=kmean_group, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# After trial and error, group of 3 seems to be the ideal choice for clustering visually
Group0=M_people[kmean_group==0]
Group1=M_people[kmean_group==1]
Group2=M_people[kmean_group==2]

#%%
# FOR GROUP 0
M_usr_x_rest_0=np.dot(M_restaurants,Group0.transpose())
M_usr_x_rest_1=np.dot(M_restaurants,Group1.transpose())
M_usr_x_rest_2=np.dot(M_restaurants,Group2.transpose())

STDEV_0=np.std(M_usr_x_rest_0,axis=1)
q75_0, q25_0 = np.percentile(M_usr_x_rest_0,[25,75],axis=1)
IQR_0 = abs(q75_0 - q25_0)

rest_list=list(restaurants.keys())
STDEV_0_rest=dict(zip(rest_list,STDEV_0.tolist()))
IQR_0_rest=dict(zip(rest_list,IQR_0.tolist()))
import operator

# Finding the restaurant with the highest standard deviation and IQR number
STDEV_0_key=min(STDEV_0_rest.items(), key=operator.itemgetter(1))[0]
STDEV_0_value=min(STDEV_0_rest.items(), key=operator.itemgetter(1))[1]
IQR_0_key=min(IQR_0_rest.items(), key=operator.itemgetter(1))[0]
IQR_0_value=min(IQR_0_rest.items(), key=operator.itemgetter(1))[1]

print("Standard deviation shows that {} restaurant will be the best choice for group 0 with STD of {}".format(STDEV_0_key, STDEV_0_value))
print("IQR shows that {} restaurant will be the best choice for group 0 with IQR of {}".format(IQR_0_key,IQR_0_value))

#%%
# FOR GROUP 1
STDEV_1=np.std(M_usr_x_rest_1,axis=1)
q75_1, q25_1 = np.percentile(M_usr_x_rest_1,[25,75],axis=1)
IQR_1 = abs(q75_1 - q25_1)

rest_list=list(restaurants.keys())
STDEV_1_rest=dict(zip(rest_list,STDEV_1.tolist()))
IQR_1_rest=dict(zip(rest_list,IQR_1.tolist()))
import operator

# Finding the restaurant with the highest standard deviation and IQR number
STDEV_1_key=min(STDEV_1_rest.items(), key=operator.itemgetter(1))[0]
STDEV_1_value=min(STDEV_1_rest.items(), key=operator.itemgetter(1))[1]
IQR_1_key=min(IQR_1_rest.items(), key=operator.itemgetter(1))[0]
IQR_1_value=min(IQR_1_rest.items(), key=operator.itemgetter(1))[1]

print("Standard deviation shows that {} restaurant will be the best choice for group 0 with STD of {}".format(STDEV_1_key, STDEV_1_value))
print("IQR shows that {} restaurant will be the best choice for group 0 with IQR of {}".format(IQR_1_key,IQR_1_value))


#%%
# FOR GROUP 2 (Just one individual)
STDEV_2=np.std(M_usr_x_rest_2,axis=1)
q75_2, q25_2 = np.percentile(M_usr_x_rest_2,[25,75],axis=1)
IQR_2 = abs(q75_2 - q25_2)

rest_list=list(restaurants.keys())
STDEV_2_rest=dict(zip(rest_list,STDEV_2.tolist()))
IQR_2_rest=dict(zip(rest_list,IQR_2.tolist()))
import operator

# Finding the restaurant with the highest standard deviation and IQR number
STDEV_2_key=min(STDEV_2_rest.items(), key=operator.itemgetter(1))[0]
STDEV_2_value=min(STDEV_2_rest.items(), key=operator.itemgetter(1))[1]
IQR_2_key=min(IQR_2_rest.items(), key=operator.itemgetter(1))[0]
IQR_2_value=min(IQR_2_rest.items(), key=operator.itemgetter(1))[1]

print("Standard deviation shows that {} restaurant will be the best choice for group 0 with STD of {}".format(STDEV_2_key, STDEV_2_value))
print("IQR shows that {} restaurant will be the best choice for group 0 with IQR of {}".format(IQR_2_key,IQR_2_value))

#%%
# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?
"""
Here, we are making all the cost in the people matrix zero because cost is not a concern anymore.
"""
people_ZeroCost=people
for k,v in people_ZeroCost.items():
        v['cost']=0

preference_P = ['distance','novelty','cost','rating','vegetarian']
M_people_ZeroCost=np.array([[people_ZeroCost[i][preference_P] for preference_P in people_ZeroCost[i]] for i in people_ZeroCost])


M_usr_x_rest_ZeroCost=np.dot(M_restaurants,M_people_ZeroCost.transpose())
"""
M_usr_x_rest_ZeroCost=
array([[ 59,  75,  45, 122, 137,  34, 132,  23, 115,  64],
       [ 83,  81,  46, 116, 132,  40, 106,  24, 113,  63],
       [ 68, 111,  51, 140, 173,  43, 123,  26, 142,  73],
       [ 91,  87,  52, 136, 153,  47, 134,  28, 130,  72],
       [109, 104,  64, 162, 179,  52, 161,  33, 154,  88],
       [105, 115,  65, 166, 189,  52, 158,  33, 161,  90],
       [ 56,  81,  36,  98, 123,  34,  79,  19, 101,  51],
       [ 96, 106,  56, 148, 174,  52, 134,  30, 146,  78],
       [ 92,  95,  50, 130, 153,  48, 113,  27, 129,  69],
       [ 39,  51,  26,  72,  86,  24,  68,  14,  71,  37]])
"""

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?
score_sum_ZeroCost=np.sum(M_usr_x_rest_ZeroCost, axis=1)
"""
array([ 806,  804,  950,  930, 1106, 1134,  678, 1020,  906,  488])
"""

# Rank based on summing scores
score_ranking_ZeroCost=len(score_sum_ZeroCost)-rankdata(score_sum_ZeroCost,method='ordinal')+1
"""
Ranking based on boss is paying => array([ 7,  8,  4,  5,  2,  1,  9,  3,  6, 10])
Previous ranking with cost considered => array([ 6,  8,  3,  5,  2,  4,  9,  1,  7, 10])

*** Here 10 being the lowest rank, 1 being the highest rank
"""

#%%
# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 
""" 
No we can't calculate a weight matrix for the new team with just optimal ordering/ranking for restauraunts. 
We need the raw score of the restaurants to be able to calculate the weight matrix. 
It ties back to the question of comparing score aggregation vs rank aggregation methods. 
Score aggregation is more appropriate since it tells how close the final ranking is determined, 
whereas ranking indexes ignores the magnitude of differences among restaurants scores. 
"""