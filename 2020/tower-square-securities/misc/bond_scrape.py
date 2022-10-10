from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import pandas_datareader
import math
from time import sleep
import pandas_datareader

#   if new statistics should be written to 'company_statistics.txt
new_statistics = True

#   path to chromedriver
path = '/usr/local/bin/chromedriver'

driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())

#   url parts
yahoo_URL = ['https://finance.yahoo.com/quote/', '?p=']


#   write statistics to text file
def append_to_text_file(data):
    #   reopen file and append information
    file = open('bond_statistics.txt', 'a')
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

    #   check if its a star rating (★)
    if number[0:1] == '★':
        return len(string) / 5

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

    #   check if its a star rating (★)
    if number[0:1] == '★':
        return len(string) / 5

    #   get rid of various symbols
    number = number.replace(',', '')
    number = number.replace('%', '')

    try:
        return float(number)
    except ValueError:
        return 0


#                   0
#   ORDER: [YTD Return 1.5, Expense Ratio 0.5, Beta 1.2, Yield 1,
#           Holdings Turnover 0.5,
#                   4
#           Morningstar Rating 1.5
#                   1
#           Cash -0.5 (line after), Maturity 1.5 <5 (line after),
#           Duration 1.2 (short term) (line after),
#                   2
#           Morningstar Return Rating 1.5 (line after), 5-Year Average Return 1.5 (line after)
#           Number of Years Up 0.5 (LA), Number of Years Down <- (LA),
#                   3
#           Morningstar Risk Rating 1.2 (LA), Alpha 1.5 (LA)
#
#   page 0 = summary, page 1 = holdings, page 2 = performance, page 3 = risk, page 4 = holdings
def get_statistics(stats, attempt, page):
    criteria = []

    #   convert statistics to strings
    try:
        stats = stats[0].text.splitlines()
    except IndexError:
        return criteria

    for statistic in stats:

        # print(statistic)

        #   summary page
        if page == 0:
            if statistic.find('YTD Return') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Expense Ratio') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Beta') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Yield') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Holdings Turnover') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

        #   Profile Page
        elif page == 4:
            if statistic.find('Morningstar Rating') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

        #   Holdings page
        elif page == 1:
            if statistic.find('Cash') >= 0 and statistic.find('Cashflow') < 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Maturity') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Duration') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

        #   Performance Page
        elif page == 2:
            if statistic.find('Morningstar Return Rating') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('5-Year Average Return') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Number of Years Up') >= 0:
                s1 = get_first_number(stats[stats.index(statistic) + 1])
                s2 = get_first_number(stats[stats.index(statistic) + 3])
                criteria.append(float(s1) / (float(s1) + float(s2)))

        #   Risk Page
        elif page == 3:
            if statistic.find('Morningstar Risk Rating') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Alpha') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

    return criteria


# ticker_list = to_list('bond_funds.txt')

# ticker_list = ['PIMIX']

ticker_list = to_list('bonds.txt')


#   yahoo search
def search_yahoo():
    for ticker in ticker_list:

        global statistics

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

            #       get all statistics

            #   Summary page
            for x in range(0, 3):
                sleep(2)
                try:
                    statistics = driver.find_elements_by_id("YDC-Col1")
                except:
                    continue

                if len(statistics) == 1:
                    break

            criteria = get_statistics(statistics, 0, 0)

            #   Profile Page
            for x in range(0, 3):
                try:
                    driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[5]/a""").click()
                    break
                except:
                    sleep(2)

            sleep(1)

            for x in range(0, 3):
                try:
                    statistics = driver.find_elements_by_id("Col1-0-Profile-Proxy")
                except:
                    sleep(2)
                    continue

                if len(statistics) == 1:
                    break

            #   get statistics and add them to criteria
            holdings_statistics = get_statistics(statistics, 0, 4)

            for statistic in holdings_statistics:
                criteria.append(statistic)

            #   Holdings page
            for x in range(0, 3):
                try:
                    driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[6]/a""").click()
                    break
                except:
                    sleep(2)

            sleep(1)
            #   scroll to bottom of page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            sleep(1)
            driver.execute_script("window.scrollTo(0, 0)")

            for x in range(0, 3):
                try:
                    statistics = driver.find_elements_by_id("Col1-0-Holdings-Proxy")
                except:
                    sleep(2)
                    continue

                if len(statistics) == 1:
                    break

            #   get statistics and add them to criteria
            holdings_statistics = get_statistics(statistics, 0, 1)

            for statistic in holdings_statistics:
                criteria.append(statistic)

            #   Performance Page
            for x in range(0, 3):
                try:
                    driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[7]/a""").click()
                    break
                except:
                    sleep(2)

            sleep(1)
            #   scroll to bottom of page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            sleep(1)
            driver.execute_script("window.scrollTo(0, 0)")

            for x in range(0, 3):
                try:
                    statistics = driver.find_elements_by_id("Col1-0-Performance-Proxy")
                except:
                    sleep(2)
                    continue

                if len(statistics) == 1:
                    break

            #   get statistics and add them to criteria
            performance_statistics = get_statistics(statistics, 0, 2)

            for statistic in performance_statistics:
                criteria.append(statistic)

            #   Risk page
            for x in range(0, 3):
                try:
                    driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[8]/a""").click()
                    break
                except:
                    sleep(2)

            sleep(1)
            #   scroll to bottom of page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            sleep(1)
            driver.execute_script("window.scrollTo(0, 0)")

            for x in range(0, 3):

                try:
                    statistics = driver.find_elements_by_id("Col1-0-Risk-Proxy")
                except:
                    sleep(2)
                    continue

                if len(statistics) == 1:
                    break

            #   get statistics and add them to criteria
            risk_statistics = get_statistics(statistics, 0, 3)

            for statistic in risk_statistics:
                criteria.append(statistic)

            # print(criteria)

            if len(criteria) != 14:
                continue

            #   reformat list into string and enter statistics to text file if logging new statistics
            if new_statistics:
                info = ticker
                for a in criteria:
                    info = info + ' ' + str(a)

                append_to_text_file(info)
        except:
            print(Exception)
            continue


def main():
    #   clear the file of existing contents if overwriting

    # if new_statistics:
    #     open('bond_statistics.txt', 'w').close()

    search_yahoo()

    driver.quit()


main()
