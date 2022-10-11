import numpy as np
import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
from PIL import Image
from matplotlib import pyplot as plt
import altair as alt

import handler
import linearprediction as lp

import os

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly.express as px

path = os.path.dirname(__file__)

# streamlit run "/Users/woodyhulse/Documents/edie/projectCode/webapp.py"

# define title
st.markdown(
"""
    <style>
    .title-style {
        font-size:40px;
        font-family:sans-serif;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# define header
st.markdown(
    """
    <style>
    .header-style {
        font-size:25px;
        font-family:sans-serif;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# define subheader
st.markdown(
    """
    <style>
    .subheader-style {
        font-size = 20px;
        font-family:sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# define normal
st.markdown(
    """
    <style>
    .body-style {
        font-size = 20px;
        font-family:sans-serif;
        font-weight:lighter;
    }
    </style>
    """,
    unsafe_allow_html=True
)


############################################################


# Sidebar visual information
st.sidebar.markdown(
    '<p class="header-style">Search Parameters</p>',
    unsafe_allow_html=True
)

st.sidebar.markdown("***")
query = st.sidebar.text_input("Search Query", max_chars=20)

st.sidebar.markdown("***")

# Hashtags
numHashtags = 3 # st.sidebar.number_input("Number of Hashtags", min_value=1, value=1)
hashtags = []
for i in range(numHashtags):
    hashtags.append(st.sidebar.text_input("Hashtag " + str(i + 1), value="#"))

st.sidebar.markdown("***")
desiredNumTweets = st.sidebar.slider("Breadth of Search", min_value=10, max_value=1000, value=100)


############################################################


# Main Page
logo, _, title = st.columns([1, 3, 20])
with logo:
    path = os.path.dirname(__file__)
    logoPath = path + "/analysislogo-transparent.png"
    image = Image.open(logoPath)
    st.image(image, width=150)
with title:
    st.markdown(
        '<p class="title-style"><br><i>Analysis</i> for Twitter</p>',
        unsafe_allow_html=True
    )

st.markdown("***")

if not query == "":
    # get tweets from handler
    tweets = handler.searchTweets(query, handler.correctHashtags(hashtags), desiredNumTweets)

    numTweets = 0
    valid = []
    text = []
    for results in tweets:
        for tweet in results:
            # text.append(tweet)
            numTweets += len(tweet)

    st.subheader(str(min(numTweets, desiredNumTweets)) + " of " + str(numTweets) + " Results for \"" + query + "\"")

    for hashtagIndex in range(len(hashtags)):
        hashtag = hashtags[hashtagIndex]
        if hashtag in handler.correctHashtags(hashtags):
            valid.append(hashtagIndex)

            if hashtagIndex == 0:
                # yellow (1st)
                st.warning(hashtag)
            elif hashtagIndex == 1:
                # green (2nd)
                st.success(hashtag)
            elif hashtagIndex == 2:
                # blue (3rd)
                st.info(hashtag)

        else:
            if not hashtag == "" and not hashtag == "#":
                st.error("Hashtag " + str(hashtagIndex + 1) + " is invalid")

    # assemble dataframe
    dfs = []
    timedfs = pd.DataFrame(range(1, 1000))

    categories = handler.correctHashtags(hashtags) ## ["All"] + handler.correctHashtags(hashtags)

    for i in valid:
        tweetContent = []
        tweetID = []
        tweetTimeStamp = []

        for tweet in list(tweets[i]):

            tweetContent.append(tweet.text)
            tweetID.append(tweet.id)
            tweetTimeStamp.append(handler.getTweetTimestamp(tweet.id))

        df = pd.DataFrame(np.array([tweetContent, tweetID, tweetTimeStamp]).transpose(),
                          columns=["content", "id", "timestamp"])
        dfs.append(df)

        timedf = pd.to_datetime(df['timestamp'], errors='coerce')
        timedf = timedf.groupby(timedf.dt.hour).count()

        timedfs[hashtags[i]] = timedf

    # create linechart
    timedfs = timedfs[timedfs.isnull().sum(axis=1) <= len(valid) - 1]
    timedfs = timedfs.fillna(0)
    timedfs[0] -= len(timedfs[0])
    # timedfs = timedfs.melt(id_vars="Date", var_name="Hashtag", value_name="Frequency")

    relativeFrequencies = [0 for i in range(len(categories))]
    numTotal = 0
    for i, row in timedfs.iterrows():
        if i > len(timedfs) / 2:
            for cat in range(len(categories)):
                relativeFrequencies[cat] += timedfs[categories[cat]][i]
                numTotal += timedfs[categories[cat]][i]

    relativeFrequencies = np.array(relativeFrequencies) / numTotal

    st.markdown(
        '<p class="header-style"><center>frequency over time (mo.)</center></p>',
        unsafe_allow_html=True
    )

    st.line_chart(pd.DataFrame(timedfs, columns=categories))


    # sentiment analysis

    sentiment = []

    stopwords = set(nltk.corpus.stopwords.words("english"))
    for df in dfs:
        words = []
        for i in range(len(df["content"])):
            for word in df["content"][i].split():
                word = ''.join(ch.lower() for ch in word if ch.isalnum())
                if word not in stopwords and not word == "":
                    words.append(word)
        # stemmed = [PorterStemmer().stem(word) for word in words]
        words = [WordNetLemmatizer().lemmatize(word) for word in words]

        analyzer = SentimentIntensityAnalyzer()
        # analyzer.polarity_scores(words)

        result = {'pos': 0, 'neg': 0, 'neu': 0}
        for word in words:
            score = analyzer.polarity_scores(word)
            if score['compound'] > 0.05:
                result['pos'] += 1
            elif score['compound'] < -0.05:
                result['neg'] += 1
            else:
                result['neu'] += 1
        if result['pos'] + result['neg'] > 0:
            ratio = result['pos'] / (result['pos'] + result['neg'])
            sentiment.append(ratio)
        else:
            sentiment.append(0.5)

    data = [relativeFrequencies, sentiment]



    st.markdown(
        '<p class="header-style"><center>sentiment analyses</center></p>',
        unsafe_allow_html=True
    )

    df = pd.DataFrame(
        [[categories[i], relativeFrequencies[i], sentiment[i]] for i in range(len(categories))],
        columns=["hashtag", "relative frequency", "sentiment"]
    )

    fig = px.bar(df, x="hashtag", y=["relative frequency", "sentiment"], barmode='group', height=400, range_y=[0, 1])
    st.plotly_chart(fig)















