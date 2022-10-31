import pandas as pd

list_of_train_classifier = []
list_of_train_auto_encoder = []
list_of_test_auto_encoder  = []

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()

for line in Lines:

    content = line.strip()
    train_auto_encoder_line = 'train_autoencoder/' + content + '.pkl'
    test_auto_encoder_line  = 'test_autoencoder/'  + content + '.pkl'
    train_classifier_line = 'test_classifier/' + content + '.pkl'
    list_of_train_classifier.append(train_classifier_line)
    list_of_train_auto_encoder.append(train_auto_encoder_line)
    list_of_test_auto_encoder.append(test_auto_encoder_line)

with open('train_classifier_pickle.txt', 'w') as fp:
    for item in list_of_train_classifier:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done list_of_train_classifier')

with open('train_autoencoder_pickle.txt', 'w') as fp:
    for item in list_of_train_auto_encoder:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done list_of_train_auto_encoder')

with open('test_auto_encoder_pickle.txt', 'w') as fp:
    for item in list_of_test_auto_encoder:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done list_of_test_auto_encoder')