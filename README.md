## Twitter Sentiment Analysis: Comparing Classification Tasks to Predict Netflix's Trending US Movies

Can tweets predict the popularity of Netflix movies? 

This study aims to predict the popularity of movies by using the sentiment analysis model VADER (Valence Aware Dictionary for
Sentiment Reasoning) to assign sentiment scores to tweets referencing movies streaming on
Netflix. 

It then compares the performance of machine learning classification models Logistic
Regression and Bernoulli Naïve Bayes in predicting which movies are ranked top ten at least
once in Netflix's weekly top ten streamed movies list.



## Documentation

[Code](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/Code_movie_prediction.ipynb)

[Data - Movies List](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/movies_list.csv)

[Data - Tweets](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/tweets.csv)

[Research Analysis](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/Research_Summary.pdf)



## Data

#### Movie List - 106 movies 

The online movie repository "What's on Netflix" posts a monthly list of movies streaming on Netflix and is the source of the list of 106 movie names manually extracted for a period of three months.

Netflix releases a weekly list of top ten ranked movies used to label the chosen movies with the binary labels, 1 for movies in the top 10 chart at least once and 0 for movies never in the top 10 chart. 

![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Dataframe_1_Movie_List.png)

#### Tweets - 46,270 tweets

Tweepy, a Python-based API is used to access Twitter's backend server to collect its users' public 
tweets.

Querying for mentions of hashtags containing one of the chosen streaming movies creates a corpus of 46,270 tweets to perform sentiment analysis. 



## Methodology

### Model
![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Methods_Classification_Model.png)


### Pre-processing
The data cleaning includes:
- Converting text to lower-case, 
- Removing URLs and HTTP links, 
- Removing usernames beginning with an @, 
- Removing hashtags beginning with a #, 
- Removing special characters except for exclamation points.


### Sentiment analysis
This study uses VADER (Valence Aware Dictionary for Sentiment Reasoning), a lexicon-based sentiment analysis tool specifically tailored to microblogs yet effective in multi-disciplinary corpora.

Sentiment scores reflect sentiment polarity, positive to negative, and sentiment intensity on a scale from -4 to +4, where good has a positive 
valance of 1.9 while great is 3.4 (Hutto and Gilbert 2014, 220). 

##### Sentiment Analysis Dataframe
![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Dataframe_2_Sentiment_Analysis.png)


### Final Data Features
Once the model produces the sentiment score for all tweets, calculating 
[a count of tweets] & [an average sentiment score for each movie] 
provides the features to train the classification models.

![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Dataframe_3_Classification_Data.png)


### Classification models

  #### Bernoulli Naive Bayes
  
  The classification algorithm Bernoulli Naïve Bayes, based on Bayes’ Theorem, measures 
  the independent probability of events A (label) and B (features) as:
  ![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Methods_Formula_Naive_Bayes.png)
  
  The goal of Bernoulli Naïve Bayes is to calculate the conditional 
  probability of the features, i.e., mean sentiment score or number of tweets given a binary class or label, 
  i.e., top ten (1), or not (0) for a particular movie (Esposito and Esposito 2020, chap.14).

  #### Logistic Regression

  Logistic Regression estimates the probability that an input belongs to a particular class, where a probability greater than 50% predicts that the input belongs to 
  the desired class, labeled 1; otherwise, it does not belong, labeled 0 (Geron 2019, chap. 4). 
  
  This binary classifier weighs the sum of the input features plus a bias term. 
  
  It then measures the logistic of the result by using a sigmoid function shown below, which outputs a number between 0 and 1 (Geron 2019, chap. 4):  
  
  ![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Methods_Formula_Logistic_Regression.png)
  
  Source: (Geron 2019) https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch04.html#idm45022189757752



## Results

Results show that the Bernoulli Naïve Bayes classifier slightly outperforms the Logistic Regression model with F1-scores of 0.53 and 0.49, respectively. 
However, the Confusion Matrix shows a bias toward movies not in the top 10, that is the majority. The ROC curve indicates a low accuracy score, one that is slightly better than a guess. 

#### F1-Score
Bernoulli Naïve Bayes slightly outperforms Logistic Regression with F1-scores of 53% and 49%, respectively.

![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Results_F1_Score.png)

#### Confusion Matrix
![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Results_Confusion_Matrix.png)

#### ROC Curve
![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Results_ROC_Curve.png)



## Analysis

### Limited corpus size

#### The Twitter API standard access level limits tweet querying to the past seven days and caps the number of tweets per session and per month. 

- Given the constant addition of new streaming movies on Netflix, tweets about older movies become scarce as new movies become a popular topic.

- The two weeks following the release of a movie are pivotal to building a comprehensive corpus of tweets for that movie. Missing this time frame or capping the number   of tweets affects the training model due to an incomplete dataset. 

- Due to these constraints, in this study, only 28% of movies queried have 100 tweets or more from which to assign an accurate mean sentiment score.


### Training methodology
 
#### This study's methodology differs from the conventional literature on analyzing tweets for sentiment analysis and opinion prediction.

- This study trains two classification models on the average sentiment score per movie and the total number of tweets generated per movie.

- Conventional methods train on the actual tweets and their corresponding sentiment scores.

- This study also includes neutral sentiments (tweets that do not include opinion word markers and therefore have a score of 0). 
  
  As shown in the chart below, many tweets are neutral, diminishing the average sentiment score; consequently, movies with more negative sentiment scores appear less 
  negative, and movies with more positive sentiment scores appear less positive when averaged.

  ##### Count of Tweets per Sentiment Type
  ![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Results_Tweets_Sentiment_Type.png)
    
    
### Restrictive twitter demographics

- A Pew Research Center study (Shah, Remy, and Smith, 2020) found that 92% of tweets come from 10% of their users, meaning that most users engage by re-tweeting,         liking, or commenting on original posts. 

- Another report found that 59% of Twitter users are between the ages of 25 and 49, male users outnumber females, and theplatform only comprises 8% of social media       users (Dean 2022). 

    #### Ex: Sentiment Scores Top Ranked (1) vs Non-Top Ranked (0)
    A small corpus and an underrepresented demographic resulted in more positive tweets for non-top-ranked movies listed as 0 than top-ranked movies listed as 1.
    
    ![App Screenshot](https://github.com/patsarmient/py_NLP-Classification-Twitter-Sentiment-Analysis/blob/main/z_Results_Tweets_Top_Rank_Per_Sentiment_Type.png)
