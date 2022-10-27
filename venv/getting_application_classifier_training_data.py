import pandas as pd

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()
list_of_rows = []
list_of_cves = []
one_minute = 600
four_minute = 2400
train_classifier_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_classifier/'
#df = pd.DataFrame()
df2 = pd.DataFrame()
for line in Lines:

    content = line.strip()
    #print(content)
    path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content +'/' + content + '-1_freqvector_full.csv'
    #print(path)
    list_of_cves.append(path)


for cve_path in list_of_cves:
    file2 = open(cve_path, 'r')
    lines2 = file2.readlines()
    cve = cve_path.split('/')[7]
    #print(cve)
    #break

    line_number =1
    for line in lines2:
        content = line.strip()

        if line_number == 1:
            line_number +=1
            continue

        elif line_number <= one_minute + 1:
            #print(cve_path,content)
            data = content.split(",")[1:]


            #print(data,content)
            #print(type(data))
            list_of_rows.append(data)
            #df = pd.DataFrame(data)
            #df2.append(df)
            #break
        line_number +=1
    df = pd.DataFrame(list_of_rows)
    list_of_rows.clear()
    path_to_pickle = train_classifier_path + cve + '.pkl'
    df.to_pickle(path_to_pickle)


#print("NEW")
#df = pd.DataFrame(list_of_rows)
#print(df)