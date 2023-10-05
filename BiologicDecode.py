import pandas as pd
import re
import datetime as dt

# Extension of the DataFrame
class BiologicDataframe(pd.DataFrame):
    def __init__(self, data = pd.DataFrame(), name = None, tech = None, timestamp = None, channel = None):
        super().__init__(data)
        self.name = name
        self.tech = tech
        self.timestamp = timestamp
        self.channel = channel

    def name(self):         # Return the file name
        return self.name

    def tech(self):         # Return the technique used (cyclic voltammetry, EIS, etc...)
        return self.tech
    
    def timestamp(self):    # Returns the time at which the experiment was conducted
        return self.timestamp
    
    def channel(self):      # Returns the channel in which the experiment was conducted
        return self.channel

# Function to clean the lines before turning them in the dataframe rows
def line_clean(line: str):
    return line.replace(",", ".").rstrip().split("\t")

def extract_simple(file_path: str):
    with open(file_path, "r") as f:
        lines_list = f.readlines()

    # Finding the header list containing the columns of the dataframe
    header_line_id= re.search(r"\d+", lines_list[1]).group()
    header_line_id = int(header_line_id) - 1

    data_rows = list(map(line_clean, lines_list[(header_line_id + 1):]))
    column_names = line_clean(lines_list[header_line_id])

    # Finding the name
    for i, line in enumerate(lines_list):
        if "Saved on :" in line:
            line = lines_list[i + 1]
            name = line.split(" : ")[1].rstrip()
            break
        else:
            continue

    # Finding the technique used
    technique = str(lines_list[3]).rstrip()

    # Finding the date
    for line in lines_list:
        if "Acquisition started on :" in line:
            raw_date = line.split(" : ")[1].rstrip()

            date_line = raw_date.split()[0]
            time_line = raw_date.split()[1]

            day = int(date_line.split("/")[1])
            month = int(date_line.split("/")[0])
            year = int(date_line.split("/")[2])
            hour = int(time_line.split(":")[0])
            minute = int(time_line.split(":")[1])
            second = int(time_line.split(":")[2].split(".")[0])

            timestamp = dt.datetime(year, month, day, hour, minute, second)
            break
        else:
            continue

    # Finding the channel
    for line in lines_list:
        if "Run on channel :" in line:
            channel = line.split(" : ")[1].rstrip()
            break
        else:
            continue

    # The data itself
    df = pd.DataFrame(
        data = data_rows,
        columns = column_names,
        dtype = float)
    
    # The final object to return with the potentiostat information
    final_df = BiologicDataframe(
        data = df,
        name = name,
        timestamp = timestamp,
        tech = technique,
        channel = channel)

    return final_df