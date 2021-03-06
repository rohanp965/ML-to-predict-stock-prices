# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:43:45 2017

@author: Dell
"""

from tkinter import *
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from PIL import Image, ImageTk 


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 't6jrDb1AuccUSRd2LUPYDMerA'
        consumer_secret = 'qPOamfTQWyC9PL2tgQH1oXpRNMC1k8GFIZ8yu1SngKw82JwHcs'
        access_token = '3108132087-vwNPOuBOjId4p1T17E9H6NNgsh8ywQ39XIoUOG5'
        access_token_secret = 'qfghkhmjEPNyRBBlAPwYp6ntrUc0rV1n3ESla0xKM16vg'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
 
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
 
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
 
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
 
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    
    master = Tk()
    master.wm_title("Sentiment Analyzer")
    L1 = Label(master, text="Enter a string to search:",justify=LEFT)
    L1.pack()
    L1.config(anchor=NW)
    e = Entry(master)
    e.pack()
    e.focus_set()
    
    def callback():
        tweets = api.get_tweets(query = e.get(), count = 1000)
        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        t.delete('1.0', END)
        t.insert(INSERT,"\n Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        t.insert(INSERT,"\n Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
        # percentage of neutral tweets
        t.insert(INSERT,"\n Neutral tweets percentage: {} %".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
        
        # printing first 5 positive tweets
        t.insert(INSERT,"\n\nPositive tweets:")
        for tweet in ptweets[:10]:
            t.insert(INSERT,"\n "+tweet['text'])
            
        # printing first 5 negative tweets
        t.insert(INSERT,"\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            t.insert(INSERT,"\n"+tweet['text'])
        
    b = Button(master, text="Send", width=10, command=callback)
    b.pack()
    w = Canvas(master, width=100, height=20)
    w.pack()
    t = Text(master,height=100,width=100,borderwidth=3, relief="raised",background="lightblue")
    t.tag_add('highlightline', '5.0', '6.0')
    t.pack()
    scrollb = Scrollbar(master, command=t.yview)
    t['yscrollcommand'] = scrollb.set
    mainloop()
    # calling function to get tweets
    #tweets = api.get_tweets(query = "Reliance Industries", count = 1000)
    #tweets = callback()
    
    
if __name__ == "__main__":
    # calling main function
    main()