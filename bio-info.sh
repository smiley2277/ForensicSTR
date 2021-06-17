#!/bin/bash

#make a directory for output
if [ ! -d ./output ] ; then
    mkdir output
fi

BAMFILE_BAI=$(find ./ -name '*sort.bam.bai' -type f)
echo $BAMFILE_BAI
#BAMFILE=$(find ./ -name '*sort.bam' -type f)
for element in $BAMFILE_BAI ; do
    chmod +x $element
    FILEPATH_1=${element##*/}
    FILEPATH_2=${FILEPATH_1%%.*}
    FILENAME=${FILEPATH_2%_*}
    echo "==========> data :"$FILEPATH_1
    echo "==========> filename :"$FILENAME
    mkdir -m 755 ./output/$FILENAME"_data"
    if [ ! -f ./output/$FILENAME"_data"/$FILENAME".bam.bai" ]; then
        samtools index $FILENAME".bam"
        if [ -s ./output/$FILENAME"_data"/$FILENAME".bam.bai" ]; then
            samtools index $FILENAME".bam"
        fi
    fi
    python3 fq_processing_v3.py --flanking 10bp.xlsx --input $FILENAME"_sort.bam" --filename $FILENAME"_001"
    mv $FILENAME"_sort.bam" ./output/$FILENAME"_data"
    mv $FILENAME"_sort.bam.bai" ./output/$FILENAME"_data"
    mv $FILENAME"_001.csv" ./output/$FILENAME"_data" #./testset
    main_v2.exe --mode test --test_file ./testset/$FILENAME"_001.csv"
    organize.exe --input $FILENAME"_001"
    done


#search for the fastq in certain directory 
FILEPATH=$(find ./ -name '*.fastq.gz' -type f)
echo $FILEPATH
for element in $FILEPATH ; do
    chmod +x $element
    FILEPATH_1=${element##*/} #including .fastq.gz
    FILENAME=${FILEPATH_1%%.*} #only filename, excluding .fastq.gz
    gzip -d $FILENAME.fastq.gz
    echo $FILEPATH $FILEPATH_1 : "exist and unpressed"
    echo "==========> data :"$FILENAME
    #bwa index Homo_sapiens_assembly38.fasta
    chmod +x $FILENAME".fastq"
    mkdir -m 755 ./output/$FILENAME"_data"

    if [ ! -f ./output/$FILENAME"_data"/$FILENAME".sam" ]; then
        bwa mem Homo_sapiens_assembly38.fasta $FILENAME".fastq" > $FILENAME".sam"
        if [ -s ./output/$FILENAME"_data"/$FILENAME".sam" ]; then
            bwa mem Homo_sapiens_assembly38.fasta $FILENAME".fastq" > $FILENAME".sam"
        fi
    fi
    if [ ! -f ./output/$FILENAME"_data"/$FILENAME".bam" ]; then
        samtools view -bS -h $FILENAME".sam" > $FILENAME".bam"
        if [ -s ./output/$FILENAME"_data"/$FILENAME".bam" ]; then
            samtools view -bS -h $FILENAME".sam" > $FILENAME".bam"
        fi
    fi
    if [ ! -f ./output/$FILENAME"_data"/$FILENAME"_sort.bam" ]; then
        samtools sort -@ 20 $FILENAME".bam" > $FILENAME"_sort.bam"
        if [ -s ./output/$FILENAME"_data"/$FILENAME"_sort.bam" ]; then
            samtools sort -@ 20 $FILENAME".bam" > $FILENAME"_sort.bam"
        fi
    fi
    if [ ! -f ./output/$FILENAME"_data"/$FILENAME"_sort.bam.bai" ]; then
        samtools index $FILENAME"_sort.bam"
        if [ -s ./output/$FILENAME"_data"/$FILENAME"_sort.bam.bai" ]; then
            samtools index $FILENAME"_sort.bam"
        fi
    fi
    python3 fq_processing_v3.py --flanking 10bp.xlsx --input $FILENAME"_sort.bam" --filename $FILENAME"_001"
    mv $FILENAME".fastq" ./output/$FILENAME"_data"
    mv $FILENAME".sam" ./output/$FILENAME"_data"
    mv $FILENAME".bam" ./output/$FILENAME"_data"
    mv $FILENAME"_sort.bam" ./output/$FILENAME"_data"
    mv $FILENAME"_sort.bam.bai" ./output/$FILENAME"_data"
    mv $FILENAME"_001.csv" ./output/$FILENAME"_data" #./testset
    main_v2.exe --mode test --test_file ./testset/$FILENAME"_001.csv"
    organize.exe --input $FILENAME"_001"
    done
     

FASTQPATH=$(find ./ -name '*.fastq' -type f)
echo $FASTQPATH
#file_array=$(python3 import_fastq.py)
for element in $FASTQPATH ; do
    chmod +x $element
    FASTQPATH_1=${element##*/}
    FASTQNAME=${FASTQPATH_1%%.*}
    echo $FASTQNAME : "exist"
    echo "==========> data :"$FASTQNAME
    #bwa index Homo_sapiens_assembly38.fasta
    chmod +x $FASTQNAME".fastq"
    mkdir -m 755 ./output/$FASTQNAME"_data"
    if [ ! -f ./output/$FASTQNAME"_data"/$FASTQNAME".sam" ]; then
        bwa mem Homo_sapiens_assembly38.fasta $FASTQNAME".fastq" > $FASTQNAME".sam"
        if [ -s ./output/$FASTQNAME"_data"/$FASTQNAME".sam" ]; then
            bwa mem Homo_sapiens_assembly38.fasta $FASTQNAME".fastq" > $FASTQNAME".sam"
        fi
    fi
    if [ ! -f ./output/$FASTQNAME"_data"/$FASTQNAME".bam" ]; then
        samtools view -bS -h $FASTQNAME".sam" > $FASTQNAME".bam"
        if [ -s ./output/$FASTQNAME"_data"/$FASTQNAME".bam" ]; then
            samtools view -bS -h $FASTQNAME".sam" > $FASTQNAME".bam"
        fi
    fi
    if [ ! -f ./output/$FASTQNAME"_data"/$FASTQNAME"_sort.bam" ]; then
        samtools sort -@ 20 $FASTQNAME".bam" > $FASTQNAME"_sort.bam"
        if [ -s ./output/$FASTQNAME"_data"/$FASTQNAME"_sort.bam" ]; then
            samtools sort -@ 20 $FASTQNAME".bam" > $FASTQNAME"_sort.bam"
        fi
    fi
    if [ ! -f ./output/$FASTQNAME"_data"/$FASTQNAME"_sort.bam.bai" ]; then
        samtools index $FASTQNAME"_sort.bam"
        if [ -s ./output/$FASTQNAME"_data"/$FASTQNAME"_sort.bam.bai" ]; then
            samtools index $FASTQNAME"_sort.bam"
        fi
    fi
    python3 fq_processing_v3.py --flanking 10bp.xlsx --input $FASTQNAME"_sort.bam" --filename $FASTQNAME"_001"
    mv $FASTQNAME".fastq" /output/$FASTQNAME"_data"
    mv $FASTQNAME".sam" ./output/$FASTQNAME"_data"
    mv $FASTQNAME".bam" ./output/$FASTQNAME"_data"
    mv $FASTQNAME"_sort.bam" ./output/$FASTQNAME"_data"
    mv $FASTQNAME"_001.csv" ./output/$FASTQNAME"_data" #./testset
    main_v2.exe --mode test --test_file ./testset/$FASTQNAME"_001.csv"
    organize.exe --input $FASTQNAME"_001"
    done
#file_array=$(python3 import_fastq.py)
#echo $file_array

