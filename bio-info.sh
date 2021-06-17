#!/bin/bash
FILENAME="B5_R1"

#bwa mem Homo_sapiens_assembly38.fasta $FILENAME.fastq > $FILENAME.sam
#samtools view -bS -h $FILENAME.sam > $FILENAME.bam
#samtools sort -@ 20 $FILENAME.bam -o $FILENAME"_sort".bam 
#samtools index $FILENAME"_sort".bam 
python3 fq_processing_v2.py --flanking 10bp.xlsx --input $FILENAME"_sort".bam --filename $FILENAME"_001"


