import csv

def read_csv_file(file_name):
    try:
        with open(f"{file_name}", 'r') as file:
          csvreader = csv.DictReader(file)
          for row in csvreader:
            print(row)
        return csvreader
    except Exception as e:
        pass

def main(file_name=None):
    if file_name:
        dict_data = read_csv_file(file_name)
        print("Process completed")
    else:
        print("Invalid file path")