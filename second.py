import operator

import numpy as np
import tweepy
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

client_key = 'XKMAYbDKkS5ovqwA706QJdRmE'
client_secret = '67qP9BIHDczo1RCSG0TIlUjetp32zAYBI0A8sWNEnhYtcrV3h0'

access_token = '227343221-61lnYTkCNJjhUls9a2LPp24eGN93e4Vfekm1fhDZ'
access_token_secret = 'bFFrJNzxgXNwgOez5KPxw4fe15qWYvh2LOfsdyQpVye8C'

auth = tweepy.OAuthHandler(client_key, client_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)

# Step 2 - Prepare query features

# List of candidates to French Republicans Primary Elections
candidates_names = ['Sarkozy', 'Kosciusko', 'Cope', 'Juppe', 'Fillon', 'Le Maire', 'Poisson']
# Hashtag related to the debate
name_of_debate = "PrimaireLeDebat"
# Date of the debate : October 13th
since_date = "2016-10-13"
until_date = "2016-10-14"


# Step 2b - Function of labelisation of analysis
def get_label(analysis, threshold=0):
    if analysis.sentiment[0] > threshold:
        return 'Positive'
    else:
        return 'Negative'


# Step 3 - Retrieve Tweets and Save Them
all_polarities = dict()
for candidate in candidates_names:
    this_candidate_polarities = []
    # Get the tweets about the debate and the candidate between the dates
    this_candidate_tweets = api.search(q=[name_of_debate, candidate], count=100, since=since_date, until=until_date)
    # Save the tweets in csv
    with open('%s_tweets.csv' % candidate, 'wb') as this_candidate_file:
        this_candidate_file.write('tweet,sentiment_label\n')
        for tweet in this_candidate_tweets:
            analysis = TextBlob(tweet.text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            # Get the label corresponding to the sentiment analysis
            this_candidate_polarities.append(analysis.sentiment[0])
            this_candidate_file.write('%s,%s\n' % (tweet.text.encode('utf8'), get_label(analysis)))
    # Save the mean for final results
    all_polarities[candidate] = np.mean(this_candidate_polarities)

# Step bonus - Print a Result
sorted_analysis = sorted(all_polarities.items(), key=operator.itemgetter(1), reverse=True)
print
'Mean Sentiment Polarity in descending order :'
for candidate, polarity in sorted_analysis:
    print('%s : %0.3f' % (candidate, polarity))
