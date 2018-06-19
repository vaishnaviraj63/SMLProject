import sys
from datetime import timedelta, date
import format_tweets

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main():

    start_date = date(2014, 2, 19)
    end_date = date(2015, 1, 1)
    fileNamePrefix = "Output"
    i=1
    for single_date in daterange(start_date, end_date):
        start =  single_date.strftime("%Y-%m-%d")
        end = (single_date + timedelta(1)).strftime("%Y-%m-%d")
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch('bitcoin').setTopTweets(True).setSince(start).setUntil(end).setMaxTweets(1000)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        writeToFile(fileNamePrefix+str(i/10), start, tweets)
        print "written tweets for date - " + start
        i+=1

def writeToFile(fileName, date, tweets):
    file = open(fileName, 'a')
    file.write("##########"+date+"\n")
    for tweet in tweets:
        file.write(tweet.text.encode('utf-8')+"\n")
    file.close()


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def printTweet(descr, t):
    print(descr)
    print("Date: %s" % t.date)
    print("favorites: %s" % t.favorites)
    print("geo: %s" % t.geo)
    print("id: %s" % t.id)
    print("Username: %s" % t.username)
    print("Retweets: %d" % t.retweets)
    print("Text: %s" % t.text)
    print("Mentions: %s" % t.mentions)
    print("Hashtags: %s\n" % t.hashtags)

if __name__ == '__main__':
	main()
