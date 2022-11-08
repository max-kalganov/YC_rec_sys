# Description for exp folder content

### baseline...

- baseline.py - was given with the train/test file 
(uses most popular tracks for predicting liked track)
- improved_baseline.py - just slightly refactored baseline.py
- baseline2.py - improved baseline version
(uses not only most popular tracks but most popular among users with
the same liked tracks)


### first version directory
Contains base for preprocessing scripts. 

The idea was to download tracks (or at least 30s parts of them)
and use this info for making a recommendation system. 

#### Idea in short
1. Make a simple embedding for each track using librosa lib (like tempo, rhythm, tonality, etc.)
2. Train LSTM to predict for each user it's next liked track embedding. Use LSTM like `seq2seq` model to train on each liked track.
3. Train such a model and store as pretrained model.
4. Overfit model for each user on inference from pretrained model on N tracks to predict the last N+1 track. Then just select the nearest 100 tracks from pool of all tracks embeddings.

Didn't work well, because music tracks are too large to be loaded fast.