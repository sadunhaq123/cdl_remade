import pandas as pd
import os
import numpy as np


cve_and_application_mapping = {}

file1 = open('cve_and_application.txt', 'r')
Lines = file1.readlines()

for line in Lines:
    content = line.strip()
    content = content.split('/')
    #print(content)
    cve_and_application_mapping.setdefault(content[0], [])
    cve_and_application_mapping[content[0]].append(content[1])
    #if content[0] not in cve_and_application_mapping.keys():
    #    cve_and_application_mapping[content[0]] = content[1]
    #else:
    #    cve_and_application_mapping[content[0]].append(content[1])

c=0
for i, j in cve_and_application_mapping.items():
    print(i, j)
    c1 = len(j)
    c +=c1

print(c)
