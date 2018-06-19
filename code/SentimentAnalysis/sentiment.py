import os
import os.path
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import datetime
import json


# Decode utf-8 file for processing
def decode_file(file):
    lines = []
    lines_inter = file.readlines()
    for line in lines_inter:
        lines.append(line.decode('utf-8'))
    return lines


# Get indices of where the date begins
def get_date_indices(lines):
    indices = []
    for line in range(len(lines)):
        if lines[line].startswith('##########'):
            indices.append(line)
    return indices


# Create a dict with date as keys and all the tweets on that particular day as a value list
def get_date_tweets(lines, indices):
    tweet_dict = {}
    for index in range(len(indices)):
        date = lines[indices[index]][10:20]
        start = indices[index] + 1
        end = indices[index] + 50
        # if index < len(indices) - 1:
        #     end = indices[index + 1]
        #     tweet_dict[date] = lines[start:end]
        # else:
        #     tweet_dict[date] = lines[start:]
        tweet_dict[date] = lines[start:end]
    return tweet_dict


# remove the new lines in the converged tweets
def remove_new_lines(string):
    char = '\n'
    while char in string:
        x = string.find(char)
        if x == -1:
            continue
        else:
            string = string[:x] + string[x + 1:]
    return string


# converge all tweets of a day to a single string
def converge_tweets(tweet_dict):
    converged_tweet_dict = {}
    for date in tweet_dict.keys():
        joined_string = ' '.join(tweet_dict[date])
        new_string = remove_new_lines(joined_string)
        converged_tweet_dict[date] = new_string
    return converged_tweet_dict


# Getting sentiment for converged tweets
def get_tweet_sentiments(tweet_dict):
    sentiment_dict = {}
    for date in tweet_dict.keys():
        value = sentiment_NaiveBayes(tweet_dict[date])
        sentiment_dict[date] = value
    return sentiment_dict


# Get sentiment for tweets without converging
def get_tweet_sentiments(tweet_dict):
    sentiment_dict = {}
    for date in tweet_dict.keys():
        sum = 0.0
        length = len(tweet_dict[date])
        for tweet in tweet_dict[date]:
            sum += sentiment_NaiveBayes(tweet)
        value = sum/float(length)
        print tweet_dict[date]
        print(datetime.datetime.now())
        print value
        sentiment_dict[date] = value
    print sentiment_dict
    return sentiment_dict


# Get sentiment of a particular string. Returns p_pos(it's positivity) of a string.
def sentiment_NaiveBayes(sentence):
    if sentence == "":
        sentiment = 0.5
    else:
        sentiment = (TextBlob(sentence, analyzer = NaiveBayesAnalyzer()).sentiment).p_pos
    return sentiment


path = os.getcwd() + '/dataset/test_data'
writepath = os.getcwd() + '/dataset/tweet_data.txt'
sentiment_dict = {}
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        file = os.path.join(root, name)
        f = open(file,'r')
        decoded_file = decode_file(f)
        date_indices = get_date_indices(decoded_file)
        date_tweets = get_date_tweets(decoded_file, date_indices)
        # converged_tweets = converge_tweets(date_tweets)
        tweet_sentiments = get_tweet_sentiments(date_tweets)
        sentiment_dict.update(tweet_sentiments)
        f.close()
print sentiment_dict
with open(writepath, 'w') as file:
    file.write(json.dumps(sentiment_dict))


# path = os.getcwd() + '\\dataset\\2014\\Output0'
# f = open(path,'r')
# decoded_file = decode_file(f)
# date_indices = get_date_indices(decoded_file)
# print "Got date indices"
# date_tweets = get_date_tweets(decoded_file, date_indices)
# print "Got date wise tweets"
# c = converge_tweets(date_tweets)
# print len(c)
# #print c
# tweet_sentiments = get_tweet_sentiments(c)
# print tweet_sentiments
