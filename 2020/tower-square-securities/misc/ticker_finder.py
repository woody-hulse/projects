from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

#   path to chromedriver
path = '/usr/local/bin/chromedriver'

driver = webdriver.Chrome(executable_path=ChromeDriverManager().install())

#   url parts
url = 'https://www.nyse.com/listings_directory/etf'


#   write statistics to text file
def append_to_text_file(data):
    #   reopen file and append information
    file = open('etfs.txt', 'a')
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


#   gets the first number from a string (statistic)
def get_first_word(string):
    #   split string into list and return last element
    global strings
    for i in range(0, 10):

        try:
            strings = string.split(' ')
        except AttributeError:
            continue

        if len(strings) > 0:
            break

        if i == 9:
            strings = []

    word = strings[0]

    if len(word) > 1 and not word == 'NEXT ›' and not word == 'LAST »' and not word == 'Symbol' and not word == 'NEXT'\
            and not word == 'LAST' and word.find('.') < 0:

        try:
            word = int(word)
        except:
            return word


def is_ticker(data):
    try:
        data = data[0].text.splitlines()
    except IndexError:
        return

    for item in data:
        word = get_first_word(item)

        try:
            if 1 < len(word) < 5:
                append_to_text_file(word)
        except TypeError:
            continue


def search():

    global data

    try:
        driver.get(url)
    except:
        print('invalid URL: ' + url)

    sleep(1)

    #   number of pages (about 10 tickers each)
    for i in range(0, 242):

        try:

            #   get data
            for x in range(0, 3):
                sleep(1)
                try:
                    data = driver.find_elements_by_class_name('true-grid-8')
                except:
                    continue

                if len(data) == 1:
                    break

            is_ticker(data)

            #   next page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/5)")

            for x in range(0, 4):
                try:
                    driver.find_element_by_xpath(
                        """//*[@id="content-4d8fb6e3-0970-4079-89ea-21d445e50709"]/div/div[2]/div[2]/div/ul/li[8]/a"""
                    ).click()
                    break
                except:
                    sleep(1)

        except:
            print('Exception: page ' + str(i))
            continue


def main():

    search()

    driver.quit()

main()
