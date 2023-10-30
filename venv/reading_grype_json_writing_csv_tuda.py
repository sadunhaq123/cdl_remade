import json
import xlsxwriter
import os
import csv

#file_list = open('list_of_files_snyk_json.txt', 'r')

#file_list = open('list_of_grype_images_tuda.txt', 'r')
#reading_file_lists = file_list.readlines()

reading_file_lists = ['2014-0050.json']
#file_list = open('debug.txt', 'r')



for first_line in reading_file_lists:
    content_first = first_line.strip()
    image_name_and_tag=content_first
    #print(content_first)
    #index_of_last_underscore = content_first.rfind('_')
    #json_file_name_with_underscore = content_first.split('_')
    #json_file_name = content_first[:index_of_last_underscore]
    #image_name_and_tag = json_file_name + 't'
    print(image_name_and_tag)

    #workbook = xlsxwriter.Workbook('trivy_officials_first_json/trivy_' +json_file_name +'_from_json_errors.xlsx')
    #worksheet = workbook.add_worksheet()

    row = 0
    list_without_headings = []
    flag_found_json_bracket = False
    flag_file_empty_errors = False
    first_name_and_target = True

    name_and_tag_list = []
    cvssScore_list = []
    unapproved_list_for_concatenation = []
    id_list = []
    identifiers_list = []
    nvdSeverity_list = []
    packageManager_list = []
    description_list = []
    packageName_list = []
    patches_list = []
    severity_list = []
    title_list = []
    from_list_list = []
    name_list = []
    version_list = []
    fixed_version_list = []
    image_name_and_tag_list = []
    modified_time_list = []
    assigner_list = []
    
    dummy_name = image_name_and_tag
    find_substring = dummy_name.find('_errors')
    json_file_name = image_name_and_tag
    print("JSON:", json_file_name)

    file1 = open('tuda_tests/Grype/' +json_file_name,'r', encoding='utf-8')
    #print('clair_official_with_json/clair_official_digests_tag_upto_2/' +content_first)
    #Lines = file1.readlines()

    #for line in Lines:
    data = json.load(file1)

    #print(data)
    #print(type(data))
    
    try:
        vulnerabilities = data['matches']
        #vulnerabilities = content_first[1]
        #print(vulnerabilities[0])
        #exit()
    except KeyError:
        continue
    
    
    
    
    for i in range(len(vulnerabilities)):
        first_dict = vulnerabilities[i]['vulnerability']
        print(first_dict)
        exit()
        
    
    
        try:
            id = vulnerabilities[i]['id']
            print(id)
            exit()
            #print(id)
        except KeyError:
            #print("id not found")
            id = None
            
        
        try:
            cvssScore = vulnerabilities[i]['metrics']
            print(cvssScore)
            exit()
        except KeyError:
            #print("cvssScore not found")
            cvssScore = None

        try:
            description = vulnerabilities[i]['description']
            #print(description)
        except KeyError:
            #print("description not found")
            description = None

        

        try:
            identifiers_all = vulnerabilities[i]['identifiers']
            try:
                identifiers = identifiers_all['CVE'][0]
            except IndexError:
                identifiers = None
            #print(identifiers)
            #print(identifiers)
        except KeyError:
            #print("identifiers not found")
            identifiers = None

        try:
            nvdSeverity = vulnerabilities[i]['nvdSeverity']
            #print(nvdSeverity)
        except KeyError:
            #print("nvdSeverity not found")
            nvdSeverity = None

        try:
            package_type = vulnerabilities[i]['packageManager']
            #print(packageManager)
        except KeyError:
            #print("packageManager not found")
            package_type = None

        try:
            package_name = vulnerabilities[i]['packageName']
            #print(packageName)
        except KeyError:
            #print("packageName not found")
            package_name = None

        try:
            patches = vulnerabilities[i]['patches']
            #print(patches)
        except KeyError:
            #print("patches not found")
            patches = None

        try:
            severity = vulnerabilities[i]['severity']
            #print(severity)
        except KeyError:
            #print("severity not found")
            severity = None

        try:
            title = vulnerabilities[i]['title']
            #print(title)
        except KeyError:
            #print("title not found")
            title = None

        try:
            from_list = vulnerabilities[i]['from']
            #print(from_list)
        except KeyError:
            #print("from_list not found")
            from_list = None

        try:
            name = vulnerabilities[i]['name']
            #print(name)
        except KeyError:
            #print("name not found")
            name = None

        try:
            version = vulnerabilities[i]['version']
            #print(version)
        except KeyError:
            #print("version not found")
            version = None

        try:
            modification_time = vulnerabilities[i]["modificationTime"]
        
        except KeyError:
            #print("version not found")
            modification_time = None
        
        
        try:
            cvss_details = vulnerabilities[i]["cvssDetails"]
            try:
                list_of_cvss = cvss_details[0]
                
                try:
                    assigner = list_of_cvss["assigner"]
            
                except KeyError:
                    assigner = None
            
            except IndexError:
            
                assigner = None
            
            
                
        except KeyError:
            assigner=None
        #print(assigner)
            
        
        try:
            fixed_version = vulnerabilities[i]['nearestFixedInVersion']
            #print(version)
        except KeyError:
            #print("version not found")
            fixed_version = None



        image_name_and_tag_list.append(image_name_and_tag)
        cvssScore_list.append(cvssScore)
        id_list.append(id)
        identifiers_list.append(identifiers)
        nvdSeverity_list.append(nvdSeverity)
        packageManager_list.append(package_type)
        description_list.append(description)
        packageName_list.append(package_name)
        patches_list.append(patches)
        severity_list.append(severity)
        title_list.append(title)
        from_list_list.append(from_list)
        name_list.append(name)
        version_list.append(version)
        fixed_version_list.append(fixed_version)
        modified_time_list.append(modification_time)
        assigner_list.append(assigner)



    with open('grype_combined_result_tuda.csv', 'a') as result_file:
        writer = csv.writer(result_file, dialect='excel')

        for w in range(len(image_name_and_tag_list)):
            writer.writerow([image_name_and_tag_list[w]
                            ,id_list[w]
                            ,severity_list[w]
                            ,identifiers_list[w]
                            ,cvssScore_list[w]
                            ,packageManager_list[w]
                            ,description_list[w]
                            ,packageName_list[w]
                            ,patches_list[w]
                            ,nvdSeverity_list[w]
                            ,assigner_list[w]
                            ,modified_time_list[w]
                            ,title_list[w]
                            ,from_list_list[w]
                            ,name_list[w]
                            ,version_list[w]
                            ,fixed_version_list[w]])

    result_file.close()





#print(data["Target"])
