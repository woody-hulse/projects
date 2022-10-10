from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import pandas_datareader
import math

#   number of positive words
num_positive_words = 0
#   number of pejorative words
num_pejorative_words = 0

#   list of stock tickers
ticker = 'BABA'
# tickers = to_list('tickers.txt')

#   dictionary of tickers to company
tickers_dictionary = {
    "AAPL": "apple",
    "BAC": "bank of america",
    "AMZN": "amazon",
    "T": "at&t",
    "GOOG": "google",
    "MO": "altria",
    "DAL": "delta",
    "AA": "alcoa",
    "AXP": "american express",
    "BABA": "alibaba",
    "ABT": "abbott",
    "MSFT": "microsoft"
}

#   path to chromedriver
path = '/usr/local/bin/chromedriver'

driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
# driver = webdriver.Chrome(executable_path=r'C:/usr/local/bin/chromedriver')

#   url parts (yahoo)
yahoo_URL = ['https://finance.yahoo.com/quote/',
             '/community?p=']

google_URL = ['https://www.google.com/search?q=',
              '&sxsrf=ALeKk03gfDkLplQEOn-FBjbujz-W5vTGVQ:1592274953373&source=lnms&tbm=nws&sa=X&ved'
              '=2ahUKEwig3Ou3poXqAhVrlXIEHTSXDnEQ_AUoAXoECB0QAw&biw=1920&bih=1001']


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


#   separate each comment into list of words
def break_comment(sentence):
    list_of_words = []
    word_in_sentence = ''

    for char in str(sentence):
        if char == ' ':
            list_of_words.append(word_in_sentence)
            word_in_sentence = ''
        else:
            word_in_sentence += char

    if len(word_in_sentence) > 0:
        list_of_words.append(word_in_sentence)

    return list_of_words


#   match each word in comment to either positive or pejorative connotation
def match_word(find_word):
    #   lowercase find_word
    find_word = find_word.lower()

    for test in positive_words:
        if find_word == test:
            return 1

    for test in negative_words:
        if find_word == test:
            return -1

    return 0


# count the number of keywords from the given list of comments
def count_keywords_from_list(comments):
    global num_pejorative_words, num_positive_words

    #   iterate through comments
    for comment in comments:

        comment = str(comment.text)

        #   break comment into list of words
        words = break_comment(comment)

        for word in words:

            #   determine connotation fo word
            connotation = match_word(word)

            if connotation == 1:
                num_positive_words += 1
            if connotation == -1:
                num_pejorative_words += 1


#   create a visual representation on command line
def create_visual(num):
    #   total length of scale
    length = 51

    for i in range(0, length):
        if i <= num * length:
            print('■', end='')
        else:
            print('□', end='')
    print('')

    #   draw midpoint
    for i in range(0, math.ceil(length / 2)):
        if i == math.ceil(length / 2) - 1:
            print('|')
        else:
            print(' ', end='')


#   determines average rate of change for stock price
def avg_rate_of_change(input_values):

    interval = 60

    last_value = len(input_values) - 1

    return (input_values[last_value] - input_values[last_value - interval]) / 60


#   compares this stock's growth to the s&p 500
def compare_to_market():
    #   stock data of company
    start_day = '2016-01-01'
    end_day = '2020-06-14'
    df = pandas_datareader.DataReader(ticker, data_source='yahoo', start=start_day, end=end_day)
    dataset = df.filter(['Close']).values

    #   market average
    spy = pandas_datareader.DataReader('SPY', data_source='yahoo', start=start_day, end=end_day)
    spy_dataset = spy.filter(['Close']).values

    #   create a percentage of spy: y = -0.5^x + 1 to center at 0.5 adjust for infinitely exceeding spy
    performance = -1 * pow(0.5, avg_rate_of_change(dataset) / avg_rate_of_change(spy_dataset)) + 1

    print(ticker + " performance against market: " + str(performance))


#   convert positive and negative words into list of strings
positive_words = to_list('positive_words.txt')
negative_words = to_list('negative_words.txt')


#   determines the raw growth rate of the stock
def growth():
    #   stock data of company
    start_day = '2016-01-01'
    end_day = '2020-06-14'
    df = pandas_datareader.DataReader(ticker, data_source='yahoo', start=start_day, end=end_day)
    dataset = df.filter(['Close']).values

    performance = -1 * pow(0.5, avg_rate_of_change(dataset)) + 1

    print(ticker + " total growth: " + str(performance))


#   yahoo search
def search_yahoo():
    try:
        driver.get(yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)
    except:
        print('invalid URL: ' + yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)

    #   press 'Show More' a few times
    for i in range(0, 10):
        driver.find_element_by_xpath("""//*[@id="canvass-0-CanvassApplet"]/div/button""").click()

    #   find comments
    comments = driver.find_elements_by_class_name("comments-body")

    #   add to positive and pejorative lists from comments
    count_keywords_from_list(comments)


#   google news search
def search_google():

    try:
        driver.get(google_URL[0] + tickers_dictionary[ticker] + google_URL[1])
    except:
        print('invalid URL: ' + google_URL[0] + tickers_dictionary[ticker] + google_URL[1])

    #   press the "next page" button a few times
    for i in range(0, 30):

        if i > 0:
            try:
                driver.find_element_by_xpath("""//*[@id="pnnext"]""").click()
            except:
                print("google search stopped at " + str(i) + " pages")
                break

        #   find comments
        comments = driver.find_elements_by_class_name("med")

        #   add to positive and pejorative lists from comments
        count_keywords_from_list(comments)


#   determine and graph final results from scrapes
def results():
    if num_positive_words + num_pejorative_words != 0:

        #   score between 0 and 1 of positivity of comments
        score = num_positive_words / (num_positive_words + num_pejorative_words)

        print(ticker + ': ' + str(score))

        #   create a visual percentage bar to represent public opinion
        create_visual(score)

        print('     positive: ' + str(num_positive_words))
        print('     pejorative: ' + str(num_pejorative_words))
    else:
        print(ticker + ': no keywords found')


def main():
    compare_to_market()

    growth()

    search_google()

    # search_yahoo()

    driver.quit()

    results()


main()
