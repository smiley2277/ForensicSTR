# ForensicSTR
# system requirement
- python 3.7
  - pysam 
  - openpyxl
  - re
  - csv
  - collections
  - sys
  - bwa

# How to use
- set up all the enviroment
- make sure "Homo_sapiens_assembly38.dict" in folder "dist".
  - if not, please type "bwa index Homo_sapiens_assembly38.fasta" at command line
- put the .fastq / .bam / .fastq.gz into the folder "dist".
- open the command line and type "./bio-info.sh", it might be awhile, depends on how big your fastq file.
- the report will generate at folder named "output" which will be under folder "dist".
--------------------------------------------------
