import pandas as pd

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()
list_of_rows = []
list_of_cves = []
one_minute = 600
three_minute = 1800
four_minute = 2400
#train_autoencoder_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder/'
train_autoencoder_path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/train_autoencoder/'
#df = pd.DataFrame()
df2 = pd.DataFrame()
list_of_values = [1,2,3,4]
for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_values:
        #path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content +'/' + content + '-' + str(k) +'_freqvector_full.csv'
        path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(
            k) + '_freqvector_full.csv'
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
            data = content.split(",")[1:]

            list_of_rows.append(data)
        line_number += 1

    if cve_four_counter % 4 ==0:

        df = pd.DataFrame(list_of_rows)

        list_of_rows.clear()
        path_to_pickle = train_autoencoder_path + cve + '.pkl'
        #print(df)
        #break
        df.to_pickle(path_to_pickle)

    cve_four_counter +=1


