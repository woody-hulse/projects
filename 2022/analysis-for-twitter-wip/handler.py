import tweepy
import config
import random
import json
import re
from textblob import TextBlob
from datetime import datetime

bearerToken = "AAAAAAAAAAAAAAAAAAAAAOI1ZQEAAAAAMTo3YqUjKBPzTNYsZurd4OL4NBY%3DS0xx6iLnBrbQ1ZnEBdQJdJH1IzHkBn0soCcOxg5RbJ585cy16O"
apiKey = "mu8XHLHGhjeuH93iC21MuYn9T"
apiKeySecret = "E06ecHLZaRVL8e7CgiYiizaFxw8kr1ej14mg77ve8Bju9wyOFr"
accessToken = "1368903126426791939-IS5ZP4VETX9xBhdvrNxxIyq6e2w4IG"
accessTokenSecret = "7BimTcMAIAaAVcuLFXMSkpJNPdAGvF0UaP2VQzaKPzn8o"

def getClient():
    client = tweepy.Client(bearer_token=bearerToken,
                           consumer_key=apiKey,
                           consumer_secret=apiKeySecret,
                           access_token=accessToken,
                           access_token_secret=accessTokenSecret)

    return client

def getAPI():
    auth = tweepy.OAuthHandler(consumer_key=apiKey,
                               consumer_secret=apiKeySecret,
                               access_token=accessToken,
                               access_token_secret=accessTokenSecret)

    return tweepy.API(auth)

def searchTweets(query, hashtags, maximum=100):
    client = getClient()
    api = getAPI()
    
    tweets = [list(tweepy.Paginator(client.search_recent_tweets, query=query,
                                    tweet_fields=['context_annotations', 'created_at'],
                                    max_results=100).flatten(limit=maximum))]

    for hashtag in hashtags:
        search = hashtag + " " + query

        tweets.append(list(tweepy.Paginator(client.search_recent_tweets, query=search,
                                            tweet_fields=['context_annotations', 'created_at'],
                                            max_results=100).flatten(limit=maximum)))

    '''
    print(len(tweets[0]), len(tweets[1]), len(tweets[2]), len(tweets[3]))
    print(len(tweets[0]), tweets[1], tweets[2], tweets[3])
    '''

    return tweets

def removeBlankHashtags(hashtags):
    for hashtagIndex in range(len(hashtags)):

        if hashtagIndex >= len(hashtags):
            continue

        if hashtags[hashtagIndex] == "" or hashtags[hashtagIndex] == " " or hashtags[hashtagIndex] == "#":
            replaceValue = ""
            for hashtagShiftIndex in range(len(hashtags) - 1, hashtagIndex - 1, -1):
                tempReplaceValue = hashtags[hashtagShiftIndex]
                hashtags[hashtagShiftIndex] = replaceValue
                replaceValue = tempReplaceValue
            hashtags = hashtags[:len(hashtags) - 1]
    
    hashtags.append("#")
            
    return hashtags

def correctHashtags(hashtags):
    correct = []
    for hashtag in hashtags:
        if hashtag != "":
            if hashtag[0] == "#" and hashtag[1:].isalnum():
                correct.append(hashtag)
                
    return correct

def getTweetTimestamp(ID):
    offset = 1288834974657
    timestamp = (ID >> 22) + offset
    utc = datetime.utcfromtimestamp(timestamp/1000)
    
    return utc


"""
    tweets = client.search_recent_tweets(query=query, max_results=10)

    tweetData = tweets.data

    results = []
    if not tweetData is None and len(tweetData) > 0:
        for tweet in tweetData:
            print(tweet)
            obj = {'id': tweet.id, 'text': tweet.text}
            results.append(obj)
        return results
    else:
        return []
    """
