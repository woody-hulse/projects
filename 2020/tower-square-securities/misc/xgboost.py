from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

path = '/usr/local/bin/chromedriver'

driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())

#   url parts
yahoo_URL = ['https://finance.yahoo.com/quote/', '/analysis?p=']


#   transform text file into list of strings
def to_list(file):
    file = open(file, 'r')
    list_of_words = []

    for line in file:
        stripped_line = line.strip()
        list_of_words.append(stripped_line)

    file.close()
    return list_of_words


#   gets the last number from a string (statistic)
def get_last_number(string):
    #   split string into list and return last element
    strings = string.split(' ')
    number = strings[len(strings) - 1]

    #   get rid of various symbols
    number = number.replace(',', '')
    number = number.replace('%', '')

    try:
        return float(number)
    except ValueError:
        return 0


def get_recommendation_rating(ticker):
    global rating

    #   get rating
    try:
        driver.get(yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)
    except:
        print('invalid URL: ' + yahoo_URL[0] + ticker + yahoo_URL[1] + ticker)

    #   scroll to bottom of page
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")

    #   get rating
    for x in range(0, 15):

        try:
            rating = driver.find_elements_by_xpath(
                """//*[@id="Col2-4-QuoteModule-Proxy"]/div/section/div/div/div[1]"""
            )
        except:
            continue

        if len(rating) == 1:
            rating = rating[0].text
            break

        if x == 14:
            rating = '3'

        sleep(2)

    #   bind rating to a 0 - 1 scale and return
    return (5 - float(rating)) / 4


ticker_list = to_list('buy_list.txt')

for ticker in ticker_list:
    print(ticker + ': ' + str(get_recommendation_rating(ticker)))

driver.quit()
