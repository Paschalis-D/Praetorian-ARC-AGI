import os 
import shutil

DIR = os.getcwd()
COMB_DIR = os.path.join(DIR, "arc-all")
os.makedirs(COMB_DIR, exist_ok=True)

datasets = []
print("Checking available datasets...")
print("Datasets: ")
for dataset in os.listdir(os.path.join(DIR, "arc-dataset-collection")):
    print(dataset)
    datasets.append(dataset)

# List all files in the directory and subdirectories with the specified extension and copy them all to a new directory.
for dataset in datasets:
    dataset_dir = os.path.join(DIR, F"arc-dataset-collection/{dataset}")   
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                shutil.copy(file_path, COMB_DIR)





