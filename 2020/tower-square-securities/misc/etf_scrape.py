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
    file = open('etf_statistics.txt', 'a')
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


#   Yield, Beta, Expense Ratio, NAV
#   1-Year Daily Total Return, 3-Year Daily Total Return,
#   Alpha, Mean Annual Return, Sharpe Ratio, Treynor Ratio
def get_statistics(stats, attempt, page):
    criteria = []

    #   convert statistics to strings
    try:
        stats = stats[0].text.splitlines()
    except IndexError:
        return criteria

    for statistic in stats:

        #   Summary page
        if page == 0:
            if statistic.find('NAV') >= 0:
                stat1 = get_last_number(statistic)
                stat2 = get_last_number(stats[stats.index(statistic) - 9])
                criteria.append(stat2/stat1)

            if statistic.find('Yield') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Beta') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

            if statistic.find('Expense Ratio') >= 0:
                stat = get_last_number(statistic)
                criteria.append(stat)

        #   Performance page
        elif page == 1:
            if statistic.find('1-Year Daily Total Return') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('3-Year Daily Total Return') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

        #   Risk page
        elif page == 2:
            if statistic.find('Alpha') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Mean Annual Return') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Sharpe Ratio') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

            if statistic.find('Treynor Ratio') >= 0:
                s = get_first_number(stats[stats.index(statistic) + 1])
                criteria.append(s)

    return criteria


# ticker_list = to_list('bond_funds.txt')

# ticker_list = ['VTV']

# ticker_list = to_list('tickers.txt')

ticker_list = to_list("etfs.txt")


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

            #   Performance Page
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
                    statistics = driver.find_elements_by_id("Col1-0-Performance-Proxy")
                except:
                    sleep(2)
                    continue

                if len(statistics) == 1:
                    break

            #   get statistics and add them to criteria
            performance_statistics = get_statistics(statistics, 0, 1)

            for statistic in performance_statistics:
                criteria.append(statistic)

            #   Risk page
            for x in range(0, 3):
                try:
                    driver.find_element_by_xpath("""//*[@id="quote-nav"]/ul/li[9]/a""").click()
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
            risk_statistics = get_statistics(statistics, 0, 2)

            for statistic in risk_statistics:
                criteria.append(statistic)

            if len(criteria) != 10:
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
    #     open('etf_statistics.txt', 'w').close()

    search_yahoo()

    driver.quit()


main()
