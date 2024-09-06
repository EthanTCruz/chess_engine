#temporary script for turning endgame tablebase into zips less than 100mb so github will take them

import os
import zipfile

def create_zip_archive(file_paths, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))

def get_size_in_mb(file_paths):
    total_size = sum(os.path.getsize(file) for file in file_paths)
    return total_size / (1024 * 1024)  # Convert bytes to MB

source_dir = "C:/Users/ethan/git/Full_Chess_App/Chess_Model/src/model/data/EndgameTblCopy"
output_dir = "C:/Users/ethan/git/Full_Chess_App/Chess_Model/src/model/data/EndgameZips" 
max_size_mb = 100  # Size limit for each ZIP file in MB

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
current_zip_files = []
zip_index = 1

for file in files:
    current_zip_files.append(file)
    if get_size_in_mb(current_zip_files) > max_size_mb:
        current_zip_files.pop()  # Remove the last file; it will be added to the next ZIP
        create_zip_archive(current_zip_files, os.path.join(output_dir, f'archive_{zip_index}.zip'))
        zip_index += 1
        current_zip_files = [file]  # Start new ZIP with the last file

# Create the last ZIP with any remaining files
if current_zip_files:
    create_zip_archive(current_zip_files, os.path.join(output_dir, f'archive_{zip_index}.zip'))

print(f"Created {zip_index} ZIP file(s).")