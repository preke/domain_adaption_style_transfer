import random


list_of_sent = []

with open('../data/pos.txt', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        list_of_sent.append(new_line)


with open('../data/neg.txt', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        list_of_sent.append(new_line)




random.shuffle(list_of_sent)


writer = open('../data/amazon_small_train.tsv', 'w')
for line in list_of_sent[:10000]:
    writer.write(line)
writer.close()
writer = open('../data/amazon_test.tsv', 'w')
for line in list_of_sent[-1000:]:
    writer.write(line)
writer.close()

