import tweepy
import config
import random
import json
import re
from textblob import TextBlob

import streamlit as st

'''
Bearer token: AAAAAAAAAAAAAAAAAAAAAOI1ZQEAAAAAMTo3YqUjKBPzTNYsZurd4OL4NBY%3DS0xx6iLnBrbQ1ZnEBdQJdJH1IzHkBn0soCcOxg5RbJ585cy16O
API Key: mu8XHLHGhjeuH93iC21MuYn9T
    API Key Secret: E06ecHLZaRVL8e7CgiYiizaFxw8kr1ej14mg77ve8Bju9wyOFr
Access Token: 1368903126426791939-IS5ZP4VETX9xBhdvrNxxIyq6e2w4IG
    Access Token Secret: 7BimTcMAIAaAVcuLFXMSkpJNPdAGvF0UaP2VQzaKPzn8o
    
JSON Request in terminal:
curl --request GET 'https://api.twitter.com/2/tweets/search/recent?query=from:twitterdev' --header 'Authorization: Bearer AAAAAAAAAAAAAAAAAAAAAOI1ZQEAAAAAMTo3YqUjKBPzTNYsZurd4OL4NBY%3DS0xx6iLnBrbQ1ZnEBdQJdJH1IzHkBn0soCcOxg5RbJ585cy16O'

export 'BEARER_TOKEN'='AAAAAAAAAAAAAAAAAAAAAOI1ZQEAAAAAMTo3YqUjKBPzTNYsZurd4OL4NBY%3DS0xx6iLnBrbQ1ZnEBdQJdJH1IzHkBn0soCcOxg5RbJ585cy16O'
'''

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

def searchTweets(query, hashtags, maximum=100):
    client = getClient()

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


tweets = searchTweets("lol")

# get_tweet_sentiment