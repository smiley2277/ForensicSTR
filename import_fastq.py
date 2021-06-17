#!/usr/bin/env python
# coding: utf-8


import os
fastq_list = []
gz_list = []
fileDir = r"/pytorch-forensic-str"
items = os.listdir(".")

def bwa_samtools():
    #if times == 1:
    #    print(times)
    #else:
    #    os.system('bwa index Homo_sapiens_assembly38.fasta')
    #    times+=1
    #    print(times)
    os.system('bwa mem Homo_sapiens_assembly38.fasta ${FILE}".fastq" > ${FILE}".sam"')
    os.system('./samtools view -bS -h ${FILE}".sam" > ${FILE}".bam"')
    os.system('./samtools sort -@ 20 ${FILE}".bam" > ${FILE}"_sort.bam"')
    os.system('./samtools index ${FILE}"_sort.bam"')

def model_predict():
    os.system('python3 fq_processing_v2.py --flanking 10bp.xlsx --input ${FILE}"_sort.bam" --filename ${FILE}"_001"')
    os.system('mv ${FILE}".sam" ./output')
    os.system('mv ${FILE}".bam" ./output')
    os.system('mv ${FILE}"_sort.bam" ./output')
    os.system('mv ${FILE}"_001.csv" ./testset')
    os.system('python3 main_v2.py --mode test --test_file ./testset/${FILE}"_001.csv"')
    os.system('python3 organize.py -- input ${FILE}"_001"')

def gzip():
    os.system('chmod +x '+gz_list[index])
    os.system('gzip -d ${FILE}".fastq.gz"')

for names in items:
    if names.endswith(".fastq"):
        fastq_list.append(names)
    if names.endswith(".fastq.gz"):
        gz_list.append(names)
print(fastq_list)
'''
for index in range(0,len(gz_list)-1):
    file = os.path.splitext(gz_list[index])[0]
    print("===========================")
    print("Input file :",gz_list[index])
    print("file : ",file)
    print("---------------------------")
    os.system('FILE="'+file+'"')
    os.system('echo ${FILE}')
    print("----------------------------------")
    os.system('chmod +x '+gz_list[index])
    os.system('gzip -d ${FILE}".fastq.gz"')
    bwa_samtools()
    print(gz_list[index],"finish alignmened and done with",file+"_sort.bam")
for index in range(0,len(fastq_list)-1):
    file = os.path.splitext(fastq_list[index])[0]
    print("===========================")
    print("Input file :",fastq_list[index])
    print("file : ",file)
    print("---------------------------")
    os.system('FILE=\"' + file + '\"')
    os.system('echo $FILE')
    print("-----------------------------------")
    os.system('chmod +x '+fastq_list[index])
    bwa_samtools()
    print(fastq_list[index],"finish alignmened and done with",file+"_sort.bam")
'''
