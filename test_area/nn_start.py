import csv
from tqdm import tqdm
from time import sleep
# first, let's find out the total number of lines in the file
with open('data.csv') as f:
    total_lines = sum(1 for line in f)

# now, let's read the file and display the progress
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in tqdm(reader, total=total_lines):
        sleep(1)
        # process your row here
        pass
