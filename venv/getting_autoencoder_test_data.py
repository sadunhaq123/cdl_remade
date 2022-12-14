import pandas as pd

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()
list_of_rows = []
list_of_cves = []
one_minute = 600
three_minute = 1800
four_minute = 2400
seven_minute = 4200
test_autoencoder_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/test_autoencoder/'
#train_autoencoder_path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/train_autoencoder/'
#df = pd.DataFrame()
df2 = pd.DataFrame()
list_of_values = [1,2,3,4]
list_of_path_timings = []
list_of_timings = []
normal = 1
anomaly = 0
list_of_all = []
for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_values:
        path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content +'/' + content + '-' + str(k) +'_freqvector_full.csv'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        list_of_cves.append(path)

    path_timing = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/timing.csv'
    list_of_path_timings.append(path_timing)

for paths in list_of_path_timings:
    file3 = open(paths, 'r')
    lines3 = file3.readlines()
    line_number_3=1
    for lines in lines3:
        content = lines.strip()
        print(content)
        if line_number_3 > 1:
            split_content = content.split(',')
            for i in range(len(split_content)):
                if i >0:
                    list_of_timings.append(split_content[i])
        line_number_3 += 1


cve_four_counter=1
for cve_path in list_of_cves:
    file2 = open(cve_path, 'r')
    lines2 = file2.readlines()
    cve = cve_path.split('/')[7]
    #break
    begin = int(list_of_timings.pop(0))
    end = int(list_of_timings.pop(0))
    #print(cve)
    #print(begin)
    #print(end)

    line_number =1
    for l2 in lines2:
        content = l2.strip()

        if line_number == 1:
            line_number +=1
            continue

        elif line_number > three_minute + 1 and line_number <= seven_minute+1:
            #print(cve_path,content)
            data = content.split(",")[1:]
            #print(cve)
            if line_number >= begin and line_number <= end:
                data.append(anomaly)
                #print("ANOMALY")
            else:
                data.append(normal)

            list_of_rows.append(data)
            list_of_all.append(data)
        line_number += 1

    if cve_four_counter % 4 ==0:

        df = pd.DataFrame(list_of_rows)

        list_of_rows.clear()
        path_to_pickle = test_autoencoder_path + cve + '.pkl'
        #print(df)
        #break
        df.to_pickle(path_to_pickle)

    cve_four_counter +=1

df3 = pd.DataFrame(list_of_all)
df3.to_pickle(test_autoencoder_path + 'train_autoencoder_all.pkl')
