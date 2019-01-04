import re





def preprocess(in_path, pos_output_paths, neg_output_paths):
    '''
    Remove labels and split data into 2 files(.pos and .neg)
    '''
    print('Preprocess Begin!')
    pos_writer = open(pos_output_paths, 'w')
    neg_writer = open(neg_output_paths, 'w')
    with open(in_path, 'r') as reader:
        for line in reader:
            if line[9] == '1': # negative
                text = line[11:]
                neg_writer.write(text.split(': ')[0])
                neg_writer.write('\t')
                neg_writer.write(text.split(': ')[1])
            if line[9] == '2': # positive
                text = line[11:]
                pos_writer.write(text.split(': ')[0])
                pos_writer.write('\t')
                pos_writer.write(text.split(': ')[1])
    pos_writer.close()
    neg_writer.close()
    print('Preprocess is done!')