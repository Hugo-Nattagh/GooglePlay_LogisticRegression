# Google Play Store - Logistic Regression

### Classification of Apps on Spark with Scala

Spark being one of the hottest technologies in Big Data, better be ready by mastering some Scala!  
This case is simply practice, no crazy breakthrough.

#### The Data

![](https://raw.githubusercontent.com/Hugo-Nattagh/ScrapGod/master/Images/Logo.png)

[My dashboard on Tableau](https://public.tableau.com/profile/hugo.nattagh#!/vizhome/Classeur1_15513265005410/Tableaudebord1)

This dataset I got from [kaggle](https://www.kaggle.com/lava18/google-play-store-apps) contains the info of the apps of the Google Play Store.  
Interesting stuff!

#### What's going on

The goal was to predict if an app would be free or paid when installed.  
I took out the «Price» feature because that would have bring no challenge whatsoever.  
I did some typical data cleaning, feature selection, feature engineering, data processing...  
At the end, I reached a RMSE of 0.276.