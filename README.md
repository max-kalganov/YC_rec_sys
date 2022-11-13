# Competition task solving attempt

Competition - [Yandex Cup 2022 - ML RecSys](https://contest.yandex.ru/yacup/contest/41618)

## Short task description

There is a music hosting service where users can mark tracks as 
`liked`.
The task is to predict the most possible next `liked` track id, 
using previous **N** `liked` tracks.

### Input
**_Train file_**

A row file where each line represents a single user. The line
contains tracks ids separated by the space symbol. All this
tracks are `liked` by the user in the same time order as
presented in the line.

**_Test file_**

Has the same format as train, but it's smaller. 

### Output

The result has to be a file where each line has not more than 100
possible track ids as next `liked` track id for the corresponding
user in the same line in **test file**.

## Suggested solution
### See in `exp` folder... (baseline_4)
Calculate statistics for the nearest users tracks / the nearest
users aritsts, etc. Trying to predict next liked track. 
One of the ideas is to get mean tracks popularity over all
users and sort prediction tracks for each user with the following key:
```commandline
(sum_of_user_tracks_popularity + current_prediction_track)/(num_of_user_tracks + 1) - dataset_mean_users_tracks_popularity
```


## Didn't have enough time to implement and debug the following solution:

### Dataset
2 datasets:
1. For tracks embeddings pretraining
Tracks triplets, where positives - same artist,
and negative - different artists.

2. For users embeddings training
Tracks triplets, where positives - liked tracks,
and negatives - different user liked tracks (with coefficient of how
many intersected tracks with the chosen user) 

### Model
2 models
1. For tracks embeddings pretraining
Simple model with a single embedding layer

2. For users embeddings training
Input as frozen track embedding + FFN. Output - same size as
tracks embedding.
