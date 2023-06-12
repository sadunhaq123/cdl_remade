import pandas as pd

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()
list_of_rows = []
list_of_cves = []
one_minute = 600
three_minute = 1800
four_minute = 2400
train_autoencoder_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder/'
#train_autoencoder_path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/train_autoencoder/'
#df = pd.DataFrame()
df2 = pd.DataFrame()
list_of_values = [1,2,3,4]
normal = 1
anomaly = -1
list_of_all = []
path_to_pickle=''
for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_values:
        path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content +'/' + content + '-' + str(k) +'_freqvector_full.csv'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        list_of_cves.append(path)

cve_four_counter=1
for cve_path in list_of_cves:
    file2 = open(cve_path, 'r')
    lines2 = file2.readlines()
    cve = cve_path.split('/')[7]
    #print(cve)
    #break

    line_number =1
    for l2 in lines2:
        content = l2.strip()

        if line_number == 1:
            line_number +=1
            continue

        elif line_number > one_minute + 1 and line_number <= three_minute+1:
            #print(cve_path,content)
            data = content.split(",")[0:]
            data.append(normal)
            #print(data)
            #exit()

            list_of_rows.append(data)
            list_of_all.append(data)

        line_number += 1

    df = pd.DataFrame(list_of_rows)
    list_of_rows.clear()
    path_to_pickle = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder_with_timestamp/' + cve + '-' + str(cve_four_counter%4 + 1) + '.pkl'
    df.to_pickle(path_to_pickle)
    cve_four_counter += 1



df3 = pd.DataFrame(list_of_all)
df3.to_pickle('C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder_with_timestamp/' + 'train_autoencoder_all.pkl')



