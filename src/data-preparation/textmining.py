import pandas as pd
from textblob import TextBlob
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #as analyser
from langdetect import detect, DetectorFactory


data = pd.read_csv('../../gen/data-preparation/temp/parsed-data.csv', sep = '\t')
data.head()

DetectorFactory.seed = 0
analyser = SentimentIntensityAnalyzer()
good_words = ['spectacular', 'good', 'great', 'best', 'goat', 'incredible', 'amazing', 'crazy', 'insane', 'fire']
delete_words = ['$', '%', '=', '»', '«', '@', '  ', '£', '§', '€', '*']

for i, j in data.iterrows():
    print(i)
    time=0
    date = str(j['created_at'])
    
    blob = TextBlob(str(j['text'.lower()]))
    for d in delete_words:
            blob = blob.replace(d, '') 
            
    if 'RT' in str(j['text']):
            data.loc[i, 'retweet'] = True
    else:
        data.loc[i, 'retweet'] = False
            
    try:
        date = date.split(' ')
        hour = date[3].split(':')
        time += float(hour[0]) + float(hour[1])/60

        data.loc[i, 'hour'] = time
        data.loc[i, 'language'] = detect(str(j['text']))
        data.loc[i, 'polarity'] = blob.sentiment.polarity
        data.loc[i, 'subjectivity'] = blob.sentiment.subjectivity
        data.loc[i, 'nwords'] = len(blob.words)

        data.loc[i, 'goodwords'] = 0
        
        for word in good_words: 
            data.loc[i, 'goodwords'] += blob.words.count(word)
                
        
    except:
        data.loc[i, 'polarity'] = ''
        data.loc[i, 'subjectivity'] = ''
        data.loc[i, 'nwords'] = ''
        data.loc[i, 'language'] = ''
        data.loc[i, 'goodwords'] = ''    
        
    data.loc[i, 'score'] = str(analyser.polarity_scores(str(j['text'])))
    data.loc[i, 'negative_score'] = str(analyser.polarity_scores(str(j['text']))['neg'])
    data.loc[i, 'positive_score'] = str(analyser.polarity_scores(str(j['text']))['pos'])
    data.loc[i, 'neutral_score'] = str(analyser.polarity_scores(str(j['text']))['neu'])
    data.loc[i, 'compound_score'] = str(analyser.polarity_scores(str(j['text']))['compound'])  

    if time < 14:
        data.loc[i, 'period'] = str('before')
    if time >= 14 and time <= 14.33:
        data.loc[i, 'period'] = str('during')
    if time >=  14.33:
        data.loc[i, 'period'] = str('after')
        
            
        
data.head()

os.makedirs('../../gen/data-preparation/output/', exist_ok=True)

data.to_csv('../../gen/data-preparation/output/dataset.csv', index = False)



print('done.')
