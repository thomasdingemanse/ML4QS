##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
from datetime import datetime

# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/phyphox/')
RESULT_PATH = Path('./intermediate_datafiles/own_data')
RESULT_FNAME = 'chapter2_result_own_data.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [60000, 6000, 1000, 250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

DIRS = ['Fietsen', 'Lopen', 'Op tafel', 'Rennen', 'Springen']
FILES = ['Accelerometer.csv', 'Barometer.csv', 'Gyroscope.csv', 'Linear Accelerometer.csv', 'Location.csv', 'Magnetometer.csv']

labels_filename = os.path.join(os.getcwd(), DATASET_PATH, 'labels.csv')
with open(labels_filename, 'w') as labels_file:
    labels_file.write("sensor_type,device_type,label,label_start,label_start_datetime,label_end,label_end_datetime\n")
    for file in FILES:
        initial_timestamp = 1654618487.0
        output_filename = os.path.join(os.getcwd(), DATASET_PATH, file.lower().replace(" ", "_"))
        if os.path.exists(output_filename):
            os.remove(output_filename)
        with open(output_filename, 'a') as output_file:
            for dir in DIRS:
                timestamp = None
                with open(os.path.join(os.getcwd(), DATASET_PATH, dir, file)) as input_file:
                    lines = input_file.readlines()
                    lines[0] = lines[0].replace(';', ',')
                    if dir == 'Fietsen':
                        output_file.write(lines[0])
                    for line in lines[1:]:
                        values = [value.replace(',', '.') for value in line.split(';')]
                        time = float(values[0])
                        timestamp = (initial_timestamp + time)
                        timestamp_ns = timestamp * float(10 ** 9)
                        values[0] = str(int(timestamp_ns))
                        line = ",".join(values)
                        output_file.write(line)
                start_datetime = datetime.fromtimestamp(initial_timestamp)
                end_datetime = datetime.fromtimestamp(timestamp)
                
                if file == 'Accelerometer.csv':
                    labels_file.write(f"interval_label,smartphone,{dir},{int(initial_timestamp*float(10**9))},{start_datetime},{int(timestamp * float(10**9))},{end_datetime}\n")
                    print(dir, initial_timestamp, "-", timestamp)

                initial_timestamp = timestamp + 1.0

print('Please wait, this will take a while to run!')

# datasets = []
# DataViz = VisualizeDataset(__file__)
# for milliseconds_per_instance in GRANULARITIES:
#     for experiment in DIRS:
#         print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

#         # Create an initial dataset object with the base directory for our data and a granularity
#         dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

#         # Add the selected measurements to it.

#         # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
#         # and aggregate the values per timestep by averaging the values
#         dataset.add_numerical_dataset(experiment + '/Accelerometer_timestamps.csv', 'Time (s)', ['X (m/s^2)','Y (m/s^2)','Z (m/s^2)'], 'avg', 'acc_phone_')
        

#         # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
#         # and aggregate the values per timestep by averaging the values
#         dataset.add_numerical_dataset(experiment + '/Gyroscope_timestamps.csv', 'Time (s)', ['X (rad/s)','Y (rad/s)','Z (rad/s)'], 'avg', 'gyr_phone_')

#         # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
#         # dataset.add_numerical_dataset('light_phone.csv', 'timestamps', ['lux'], 'avg', 'light_phone_')

#         # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
#         # and aggregate the values per timestep by averaging the values
#         dataset.add_numerical_dataset(experiment + '/Magnetometer_timestamps.csv', 'Time (s)', ['X (µT)','Y (µT)','Z (µT)'], 'avg', 'mag_phone_')

#         # Get the resulting pandas data table
#         dataset = dataset.data_table

#         # Plot the data
#         # DataViz = VisualizeDataset(__file__)

#         # Boxplot
#         DataViz.plot_dataset_boxplot(dataset, ['acc_phone_X (m/s^2)','acc_phone_Y (m/s^2)','acc_phone_Z (m/s^2)'])

#         # Plot all data
#         DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'mag_'],
#                                     ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
#                                     ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

#         # And print a summary of the dataset.
#         util.print_statistics(dataset)
#         datasets.append(copy.deepcopy(dataset))

#         # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
#         dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')


# Make a table like the one shown in the book, comparing the two datasets produced.
# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
# dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through

print('The code has run through successfully!')