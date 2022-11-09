import pandas as pd

file1 = open('apps-cdl.txt', 'r')
Lines = file1.readlines()
list_of_rows = []
list_of_cves = []
cve_and_application_mapping = {}
one_minute = 600
four_minute = 2400
three_minute = 1800
seven_minute =4200
list_of_all = []
test_classifier_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/test_classifier/'
#test_classifier_path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/test_classifier/'
#df = pd.DataFrame()
df2 = pd.DataFrame()
list_of_values = [1,2,3,4]
for line in Lines:

    content = line.strip()
    #print(content)
    for k in list_of_values:
        path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/data-CDL/' + content +'/' + content + '-' + str(k) +'_freqvector_full.csv'
        #path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
        print(path)
        list_of_cves.append(path)




file2 = open('cve_and_application.txt', 'r')
Line2 = file2.readlines()

for line in Line2:
    content = line.strip()
    content = content.split('/')
    #print(content)
    cve_and_application_mapping.setdefault(content[1], [])
    cve_and_application_mapping[content[1]].append(content[0])
    #if content[0] not in cve_and_application_mapping.keys():
    #    cve_and_application_mapping[content[0]] = content[1]
    #else:
    #    cve_and_application_mapping[content[0]].append(content[1])


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


        elif line_number > three_minute + 1 and line_number <= seven_minute+1:
            #print(cve_path,content)
            data = content.split(",")[1:]
            #print(data)
            #print(cve_and_application_mapping[cve][0])
            data.append(cve_and_application_mapping[cve][0])
            #print(data)

            list_of_rows.append(data)
            list_of_all.append(data)
        line_number += 1

    if cve_four_counter % 4 ==0:

        df = pd.DataFrame(list_of_rows)

        list_of_rows.clear()
        path_to_pickle = test_classifier_path + cve + '.pkl'
        #print(df)
        #break
        df.to_pickle(path_to_pickle)

    cve_four_counter +=1

df_all = pd.DataFrame(list_of_all)
path_all_to_pickle = test_classifier_path + 'all_combined.pkl'
df_all.to_pickle(path_all_to_pickle)