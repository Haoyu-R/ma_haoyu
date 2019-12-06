import os

if __name__ == r"__main__":
    path = r'..\concatenate_data\not_processed_1'
    csv_files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                a = os.path.join(root, file)
                print(os.path.basename(a))
