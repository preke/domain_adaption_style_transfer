from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
# logging
import logging
import logging.config
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)



list_pos = []
list_neg = []
pos_sentence_list = []
neg_sentence_list = []
stop_words = stopwords.words('english')

logger.info('Read in data...\n')
with open('../data/pos.txt', 'r') as reader:
    for line in reader:
        pos_sentence_list.append(line)
        new_line = line.lower().split()
        new_line = [w for w in new_line if w not in stop_words]
        list_pos.append(new_line)

with open('../data/neg.txt', 'r') as reader:
    for line in reader:
        neg_sentence_list.append(line)
        new_line = line.lower().split()
        # new_line = [w for w in new_line if w not in stop_words]
        list_neg.append(new_line)

logger.info('Train word2vec model...\n')
# train w2v model
# list_pos = list_pos[:1000]
# list_neg = list_neg[:1000]
# pos_sentence_list = pos_sentence_list[:1000]
# neg_sentence_list = neg_sentence_list[:1000]


list_all = list_pos + list_neg
w2v_model = Word2Vec(list_all, min_count=5)

sim_matrix = []
for i in range(10):
    sim_matrix.append([])
    for j in range(list_neg[:10000]):
        sim_matrix[i].append(w2v_model.wmdistance(list_pos[i], list_neg[j]))

logger.info('sim_matrix_shape: %d, %d\n' %(len(sim_matrix), len(sim_matrix[0])))

with open('../data/wmd_result', 'w') as writer:
    for i in range(10):
        writer.write('Pos: %s\n' %list_pos[i])    
        #print sim_matrix[i]
        neg_index = int(np.argmax(np.array(sim_matrix[i])))
        #print neg_index
        writer.write('Neg: %s\n' %list_neg[neg_index]) 
        writer.write('\n')    


# w2v_model.wmdistance


# word mover's distance pairing

