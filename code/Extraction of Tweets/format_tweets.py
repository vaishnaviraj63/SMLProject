'''
format_tweets.py
Includes methods for processing raw tweet data

Created by Miles Luders
'''
from string import ascii_lowercase
from nltk.corpus import words as english_words, stopwords
from multiprocessing.dummy import Pool as ThreadPool
import sys


def remove_excess_whitespace(text):
    return ' '.join(text.split())


def convert_to_lowercase(text):
    return text.lower()


def remove_non_alpha_chars(text):
    T = list(text)
    i = 0
    while i < len(T):
        if T[i] not in ascii_lowercase and T[i] != ' ':
            del T[i]
        else:
            i += 1

    return ''.join(T)


def format_syntax(text):
    a = convert_to_lowercase(text)
    b = remove_non_alpha_chars(a)
    c = remove_excess_whitespace(b)
    return c


def remove_non_english_words(text, english):
    T = text.split(' ') # ["hello", "world"]

    i = 0
    while i < len(T):
        if T[i] not in english:
            del T[i]
        else:
            i += 1

    return ' '.join(T)


def remove_stopwords(text, stop):
    T = text.split(' ')

    i = 0
    while i < len(T):
        if T[i] in stop:
            del T[i]
        else:
            i += 1

    return ' '.join(T)


def format_semantic(text):
    english = set(w.lower() for w in english_words.words())
    stop = set(w.lower() for w in stopwords.words())
    a = remove_non_english_words(text, english)
    b = remove_stopwords(a, stop)

    return b

def format_text(text):
    if text.startswith("##########"):
        return text
    return format_semantic(format_syntax(text))

def format_files(start, end):
    fileNamePrefix = "Output"

    for i in xrange(start, end):
        fileName = fileNamePrefix+str(i)
        OutputFileName = "Formatted"+fileName
        outfile = open(OutputFileName,"w")
        file = open(fileName, "r")
        # format_file(file, outfile)
        for line in file:
            outfile.write(format_text(line)+"\n")


        outfile.close()
        file.close()
        print "completed file Output"+str(i)

def format_file(input_file, output_file):
    # make the Pool of workers
    pool = ThreadPool(8)
    lines = input_file.read().splitlines()

    # open the urls in their own threads
    # and return the results
    results = pool.map(format_text, lines)
    # close the pool and wait for the work to finish
    pool.close()
    pool.join()

    for line in results:
        output_file.write(line+"\n")





if __name__ == '__main__':
    '''print("hello")
    T = collect_tweets.get_latest_tweets('bitcoin', 2)
    for t in T:
        print(remove_excess_whitespace(t[1])+"\n")'''

    #print(format_syntax("# ILoveNY bcuz    $ $  money"))
    # print(format_semantic("hello to the world"))
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    format_files(start, end)

