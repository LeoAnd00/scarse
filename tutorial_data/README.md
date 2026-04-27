# Tutorial

The regression and classification training and testing data are random subsets of the POLG_PESV dataset.<br>
The "score" column is the original endpoint of POLG_PESV.<br>
The "score_2" column is just a copie of the "score" column but with gaussian noise added with a std=0.1.<br>
The "score_binary_class" column consist of when dividing the data into "low" (bottom 50% of data) and "high" (top 50% of data).<br>
The "score_binary_class_random" column consist of fully randomly assigned classes.<br>
The "score_multi_class" column consist of when dividing the data into "low" (bottom 25% of data), "medium" (between 25-75% of data) and "high" (top 25% of data).