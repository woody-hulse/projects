file = open("instagram_data.txt", 'r')

line = file.read().replace("\n", " ")
file.close()

words = line.split(',')

open('instagram_messages.txt', 'w').close()


#   write statistics to text file
def append_to_text_file(data):
    #   reopen file and append information
    write_file = open('instagram_messages.txt', 'a')
    write_file.write(data)
    write_file.write("\n")
    write_file.close()


for i in range(0, len(words)):
    try:
        if words[i].index("\"text\"") != -1:
            append_to_text_file(words[i - 2].replace("{\"sender\":", ' ') +
                                ': ' + words[i].replace("\"text\":", ' '))
    except ValueError:
        continue
