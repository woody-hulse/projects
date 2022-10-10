"""
CRITERIA:

DIVIDEND/SHARE: 0.5
REVENUE: 1.5
P/E RATIO: 0.7
BETA:1.1
ANALYST BUY/SELL/HOLD: 1.2
PRICE HISTORY: 1
QUARTERLY EARNINGS GROWTH: 0.3
FAIR VALUE: 0.7
"""

import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import pandas_datareader
import math
from time import sleep
import pandas_datareader

#   if new statistics should be written to 'company_statistics.txt
new_statistics = True

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
    "MSFT": "microsoft",
    "AEG": "aegon nv",
    "APD": "air products and chemicals inc",
    "ALXN": "alexion pharmaceuticals",
    "ABC": "amerisourcebergen",
    "BCS": "barclays plc",
    "BMRN": "biomarin pharmaceutical",
    "BLK": "blackrock",
    "AVGO": "broadcom",
    "CRS": "carpenter technology corporation",
    "CBOE": "Cboe Global Markets",
    "CNC": "Centene",
    "CRL": "charles river laboratories international",
    "CHTR": "charter communications inc",
    "BYND": "beyond meat",
    "CLW": "clearwater paper",
    "CLF": "cleveland-cliffs",
    "DG": "dollar general",
    "GD": "general dynamics",
    "G": "genpact limited",
    "HXL": "hexcel",
    "KBH": "kb home",
    "LMT": "lockheed martin",
    "LITE": "lumentum holdings",
    "MDT": "medtronic plc",
    "MRK": "merck and co",
    "NEWR": "new relic",
    "NOC": "northrop grumman",
    "PEP": "pepsi",
    "PLD": "prologis",
    "RSG": "republic services",
    "SRE": "sempra energy",
    "SWKS": "skyworks solutions",
    "TMUS": "t mobile",
    "TECK": "teck resources",
    "TMO": "thermo fisher",
    "TOT": "total sa",
    "TSN": "tyson foods",
    "WMT": "walmart",
    "ZTS": "zoetis",
}

#   path to chromedriver
path = '/usr/local/bin/chromedriver'

driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())

#   url parts
yahoo_URL = ['https://finance.yahoo.com/quote/', '/key-statistics?p=']

yahoo_comments_URL = ['https://finance.yahoo.com/quote/',
                      '/community?p=']

google_URL = ['https://www.google.com/search?q=',
              '&sxsrf=ALeKk03gfDkLplQEOn-FBjbujz-W5vTGVQ:1592274953373&source=lnms&tbm=nws&sa=X&ved'
              '=2ahUKEwig3Ou3poXqAhVrlXIEHTSXDnEQ_AUoAXoECB0QAw&biw=1920&bih=1001']

#   get 30 day sma SPY statistics
spy = pandas_datareader.DataReader(
    'SPY', data_source='yahoo', start='2014-10-01', end='2020-7-30'
).filter(['Close']).rolling(window=30).mean().values

#   spy 0, 60, 365, and 1825 days ago
spy_0 = spy[len(spy) - 1, 0]
spy_growth_60 = spy_0 / spy[len(spy) - 42, 0]
spy_growth_365 = spy_0 / spy[len(spy) - 254, 0]
spy_growth_1825 = spy_0 / spy[len(spy) - 1270, 0]

#   for scraping comments
num_positive_words = 0
num_pejorative_words = 0


#   write statistics to text file
def append_to_text_file(data):

    #   reopen file and append information
    file = open('company_statistics.txt', 'a')
    file.write(data)
    file.write("\n")
    file.close()


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


#   separate a sentence into a list of words
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


#   match each word in a comment to either positive or pejorative connotation
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


#   gets the last number from a string (statistic)
def get_last_number(string):
    #   split string into list and return last element
    global strings
    for i in range(0, 15):

        try:
            strings = string.split(' ')
        except AttributeError:
            continue

        if len(strings) > 0:
            break

        if i == 9:
            strings = []

    number = strings[len(strings) - 1]

    #   get rid of various symbols
    number = number.replace(',', '')
    number = number.replace('%', '')

    try:
        return float(number)
    except ValueError:
        return 0


#   gets the first number from a string (statistic)
def get_first_number(string):
    #   split string into list and return last element
    global strings
    for i in range(0, 15):

        try:
            strings = string.split(' ')
        except AttributeError:
            continue

        if len(strings) > 0:
            break

        if i == 9:
            strings = []

    number = strings[0]

    #   get rid of various symbols
    number = number.replace(',', '')
    number = number.replace('%', '')

    try:
        return float(number)
    except ValueError:
        return 0


def get_price(price):

    if price.find('-') >= 0:
        price_num = price[0 : price.find('-')]
    else:
        price_num = price[0 : price.find('+')]

    return price_num


#   ORDER: [trailing_pe, beta, dividend_rate, revenue_growth, diluted_eps, earnings_growth,
#           fair_value, recommendation_rating, share_price_growth, public opinion
def extract_relevant_statistics(statistics, attempt):
    criteria = []

    #   convert statistics to strings
    try:
        statistics = statistics[0].text.splitlines()
    except IndexError:
        return criteria

    for statistic in statistics:

        #   find relevant statistics
        #       annual dividend rate
        if statistic.find('Trailing Annual Dividend Yield') >= 0:
            dividend_rate = get_last_number(statistic)
            criteria.append(dividend_rate)

        #       year-over-year revenue growth
        if statistic.find('Quarterly Revenue Growth') >= 0:
            revenue_growth = get_last_number(statistic)
            criteria.append(revenue_growth)

        #       current P/E ratio
        if statistic.find('Trailing P/E') >= 0:
            trailing_pe = get_first_number(statistics[statistics.index(statistic) + 1])
            #   if no p/e found, set to default
            if trailing_pe == 0:
                if statistic.find('Forward P/E') >= 0:
                    trailing_pe = get_first_number(statistics[statistics.index(statistic) + 1])
            if trailing_pe == 0:
                trailing_pe = 23.16
            criteria.append(trailing_pe)

        #       five year beta
        if statistic.find('Beta') >= 0:
            beta = get_last_number(statistic)
            #   if no beta found, set to a default
            if beta == 0:
                beta = 1
            criteria.append(beta)

        #       diluted earnings per share
        if statistic.find('Diluted EPS') >= 0:
            deps = get_last_number(statistic)
            criteria.append(deps)

        #       year over year earnings growth (with expenditures)
        if statistic.find('Quarterly Earnings Growth') >= 0:
            earnings_growth = get_last_number(statistic)
            criteria.append(earnings_growth)

    if attempt == 5:
        return []

    if len(criteria) != 6:
        extract_relevant_statistics(statistics, attempt + 1)

    return criteria


#   get analysts' opinions
def get_fair_value():
    global average_fair_value
    global current_price

    #   navigate to "Analysis" page
    for x in range(0, 4):
        try:
            driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[9]/a""").click()
            str_error = None
        except Exception as str_error:
            pass

        if str_error:
            sleep(1)
        else:
            break

    #   scroll to bottom of page
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    #   get current share price and average fair value
    for x in range(0, 4):

        try:
            current_price = driver.find_elements_by_xpath(
                """//*[@id="Col2-5-QuoteModule-Proxy"]/div/section/div/div[1]/div[3]"""
            )

            average_fair_value = driver.find_elements_by_xpath(
                """//*[@id="Col2-5-QuoteModule-Proxy"]/div/section/div/div[1]/div[4]"""
            )
        except:
            continue

        if len(current_price) == 1 and len(average_fair_value) == 1:
            current_price = current_price[0].text
            average_fair_value = average_fair_value[0].text

            break

        sleep(1)

    try:
        return get_last_number(average_fair_value) / get_last_number(current_price)
    except ZeroDivisionError:
        return 1


#   gets the 1 - 5 recommendation rating (must be called after get_fair_value)
def get_recommendation_rating():
    global rating

    #   scroll to bottom of page
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    #   get rating
    for x in range(0, 3):

        try:
            rating = driver.find_elements_by_xpath(
                """//*[@id="Col2-4-QuoteModule-Proxy"]/div/section/div/div/div[1]"""
            )
        except:
            continue

        if len(rating) == 1:
            rating = rating[0].text

            break

        if x == 2:
            rating = '3'

        sleep(1)

    #   bind rating to a 0 - 1 scale and return
    return (5 - get_last_number(rating)) / 4


#   compare stock moving average growth to SPY over past 60 days, 1 year, 5 years
def compare_growth_to_spy(ticker):
    #   get 30 day sma SPY statistics
    prices = pandas_datareader.DataReader(
        ticker, data_source='yahoo', start='2014-10-01', end='2020-7-30'
    ).filter(['Close']).rolling(window=30).mean().values

    if len(prices) > 1280:
        #   get values from 0, 60, 365, and 1825 days ago
        price_0 = prices[len(prices) - 1, 0]
        price_growth_60 = price_0 / prices[len(prices) - 42, 0]
        price_growth_365 = price_0 / prices[len(prices) - 254, 0]
        price_growth_1825 = price_0 / prices[len(prices) - 1270, 0]

        #   compare these values to spy
        adjusted_performance = price_growth_60 / spy_growth_60 / 3 + \
                               price_growth_365 / spy_growth_365 / 3 + \
                               price_growth_1825 / spy_growth_1825 / 3
    elif len(prices) > 260:
        #   get values from 0, 60, 365 days ago
        price_0 = prices[len(prices) - 1, 0]
        price_growth_60 = price_0 / prices[len(prices) - 42, 0]
        price_growth_365 = price_0 / prices[len(prices) - 254, 0]

        #   compare these values to spy
        adjusted_performance = price_growth_60 / spy_growth_60 / 2 + \
                               price_growth_365 / spy_growth_365 / 2 * 0.95
    else:
        #   get values from 0, 60 days ago
        price_0 = prices[len(prices) - 1, 0]
        price_growth_60 = price_0 / prices[len(prices) - 42, 0]

        #   compare these values to spy
        adjusted_performance = price_growth_60 / spy_growth_60 * 0.8

    return adjusted_performance


# ticker_list = to_list('buy_list.txt')

# ticker_list = to_list('tickers.txt')

ticker_list = to_list('all_tickers.txt')

# ticker_list = ['AAPL']

#   convert positive and negative words into list of strings
positive_words = to_list('positive_words.txt')
negative_words = to_list('negative_words.txt')


#   yahoo search
def search_yahoo_comments(ticker):
    try:
        driver.get(yahoo_comments_URL[0] + ticker + yahoo_comments_URL[1] + ticker)
    except:
        print('invalid URL: ' + yahoo_comments_URL[0] + ticker + yahoo_comments_URL[1] + ticker)

    #   press 'Show More' a few times
    for i in range(0, 4):
        try:
            driver.find_element_by_xpath("""//*[@id="canvass-0-CanvassApplet"]/div/button""").click()
        except:
            continue

    #   find comments
    comments = driver.find_elements_by_class_name("comments-body")

    #   add to positive and pejorative lists from comments
    count_keywords_from_list(comments)


#   google news search
def search_google(ticker):

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
                break

        #   find comments
        comments = driver.find_elements_by_class_name("med")

        #   add to positive and pejorative lists from comments
        count_keywords_from_list(comments)


#   yahoo search
def search_yahoo():

    passes = 0

    for ticker in ticker_list:

        global statistics, num_positive_words, num_pejorative_words, price, driver

        passes += 1

        if passes > 19:
            driver.quit()
            sleep(1)
            driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())
            sleep(1)
            passes = 0

        try:
            try:
                driver.get(yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)
            except:
                print('invalid URL: ' + yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)

            sleep(1)
            #   scroll to bottom of page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            sleep(1)
            driver.execute_script("window.scrollTo(0, 0)")

            #   check if penny stock
            for x in range(0, 6):

                try:
                    price = driver.find_elements_by_id("Lead-3-QuoteHeader-Proxy")
                except:
                    continue

                if len(price) == 1:
                    break

                sleep(1)

            price = price[0].text.splitlines()

            price = get_price(price[4])

            if float(price) < 5:
                continue

            #   get all statistics
            for x in range(0, 6):

                try:
                    statistics = driver.find_elements_by_id("Col1-0-KeyStatistics-Proxy")
                except:
                    continue

                if len(statistics) == 1:
                    break

                sleep(1)

            #   reset positive and pejorative word counts
            num_positive_words = 0
            num_pejorative_words = 0

            criteria = extract_relevant_statistics(statistics, 0)

            if len(criteria) == 0:
                continue

            #   get and append fair value / current value
            criteria.append(get_fair_value())

            #   get and append analyst recommendation
            criteria.append(get_recommendation_rating())

            #   get and append share performance against spy
            criteria.append(compare_growth_to_spy(ticker))

            #   gauge public opinion and append score
            # search_yahoo_comments(ticker)
            # search_google(ticker)
            # criteria.append(num_positive_words / (num_pejorative_words + num_positive_words))
            criteria.append(0)

            #   reset positive and pejorative word counts
            num_positive_words = 0
            num_pejorative_words = 0

            #   reformat list into string and enter statistics to text file if logging new statistics
            if new_statistics:
                info = ticker
                for a in criteria:
                    info = info + ' ' + str(a)

                append_to_text_file(info)
        except:
            continue


def main():

    #   clear the file of existing contents if overwriting
    # if new_statistics:
    #    open('company_statistics.txt', 'w').close()

    search_yahoo()

    driver.quit()


main()
