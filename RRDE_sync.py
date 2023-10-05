import os
import BiologicDecode as bd
import re
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np

# Options
data_folder = "examplefolder"
correct_fft = True

# Column names
pot = "Ewe/V"
curr = "<I>/mA"
time = "time/s"
disc_curr = "<I>/mA_disc"
ring_curr = "<I>/mA_ring"
cycle = "cycle number"
rotation_column = "Analog OUT/V"
rot = "Rotation/rpm"
water = "H2O"
branch = "IsCathodic"
id = "ID"
orig_file = "file"

# Function that syncronizes the two dataframes
# If correct_ring_fft is True, the data from the ring will be cleared with a low-pass FFT filter

def ring_disc_sync(disc_df: bd.BiologicDataframe, ring_df: bd.BiologicDataframe, correct_ring_fft = False):
    # LSV need the "cycle" to be manually defined
    if disc_df.tech == "Linear Sweep Voltammetry":
        disc_df[cycle] = 1
    
    for i in disc_df[cycle].unique():
        disc_sub_df = disc_df[disc_df[cycle] == i]

        time_begin = disc_sub_df[time].min()
        time_end = disc_sub_df[time].max()

        ring_sub_df = ring_df[(ring_df[time] >= time_begin) & (ring_df[time] <= time_end)].copy()
        ring_sub_df = ring_sub_df.reset_index(drop = True)

        # FFT filter
        if correct_ring_fft:
            rotation_Hz = int(disc_df[rotation_column].mean()) / 60 # Converting RPM to Hz

            if rotation_Hz == 0:
                continue

            dt = (ring_sub_df[time] - ring_sub_df[time].shift(1)).mean()

            nyq = 1 / (2 * dt)  # Nyquist frequency
            cutoff_freq = 0.5 * rotation_Hz / nyq   # The filter frequency is set to be half of the rotation frequency

            b, a = butter(2, cutoff_freq, btype = "lowpass", analog = False)
            new_curr = filtfilt(b, a, np.array(ring_sub_df[curr]))

            ring_sub_df[curr] = new_curr

        # The ring datasets have a larger amount of points because of the requirements from FFT filters.
        # Therefore, only the values corresponding to the disc points times are selected
        # However, since most of the time the exact times will not be shared, the data will be linearly interpolated from the two closest

        timepoints = pd.concat([disc_sub_df[time], ring_sub_df[time]]).unique()

        temp_df = pd.DataFrame({time: timepoints})
        temp_df = pd.merge(temp_df, disc_sub_df[[curr, time]], on = time, how = "outer")
        temp_df = pd.merge(temp_df, ring_sub_df[[curr, time]], on = time, how = "outer", suffixes=("_disc", "_ring"))

        temp_df = temp_df.sort_values(by = time)
        temp_df = temp_df.set_index(time)

        temp_df[ring_curr] = temp_df[ring_curr].interpolate(method = "index")

        temp_df = temp_df[temp_df[disc_curr].notna()]

        disc_df.loc[disc_df[cycle] == i, ring_curr] = temp_df[ring_curr].values

    disc_df = disc_df.rename(columns = {rotation_column: rot, curr: disc_curr})
    disc_df = disc_df[disc_df[ring_curr].notna()]
    disc_df = disc_df.sort_values(by = time)
    disc_df = disc_df.reset_index(drop = True)

    return disc_df

# The first task is to find a disc and ring file pair
# They can be recognized by their file name
# If a pair is found, it's stored in pair_list
pair_list = []

for ring_file in os.listdir(data_folder):
    if ".mpt" not in ring_file or "_CA_" not in ring_file:  # Ring files are only CA
        continue

    # This is the base name shared by the two files
    base_name = ring_file.split("_")[:-3]

    for disc_file in os.listdir(data_folder):
        if ".mpt" not in disc_file or ("_CV_" not in disc_file and "_LSV_" not in disc_file):
            continue # Disc files are either CV or LSV

        if disc_file.split("_")[:-3] == base_name:
            pair_list.append((disc_file, ring_file))

final_df = pd.DataFrame() # Dataframe where data will be finally stored
unique_id = 0   # Unique ID of the specific file/cycle pair, useful for code built upon this data

for (disc_file, ring_file) in pair_list:
    # To keep things neat, all the data is stored in a separate folder
    disc_path = os.path.join(data_folder, disc_file)
    ring_path = os.path.join(data_folder, ring_file)

    # Getting the dataframes from the paths
    disc_df = bd.extract_simple(disc_path)
    ring_df = bd.extract_simple(ring_path)

    pair_df = ring_disc_sync(disc_df, ring_df, correct_ring_fft = correct_fft)

    # If H2O content is specified in the filename, then is added as a column
    if re.search(r"(\d+)H2O", disc_file) == None:
        pair_df[water] = np.nan
    else:
        pair_df[water] = int(re.search(r"(\d+)H2O", disc_file).group(1))

    # Storing the original file as column
    pair_df[orig_file] = disc_file

    # The ID should uniquely identify a file/cycle pair, therefore the last id used is stored in unique_id at each cycle
    pair_df[id] = pair_df[cycle] + unique_id
    unique_id = unique_id + pair_df[cycle].max()

    # Cleaning the rotation data
    pair_df[water] = int(pair_df[water].mean())

    final_df = pd.concat([final_df, pair_df], ignore_index = True)

# Dividing in cathodic and anodic branch, useful for data analysis later
for i in final_df[id].unique():
    sub_df = final_df[final_df[id] == i]
    cathodic_extreme = sub_df[pot].idxmin()
    change_time = sub_df.loc[cathodic_extreme, time]

    final_df.loc[(final_df[id] == i) & (final_df[time] <= change_time), branch] = True
    final_df.loc[(final_df[id] == i) & (final_df[time] > change_time), branch] = False

final_df.to_csv(data_folder + ".csv", index = False)