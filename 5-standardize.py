import time
import siuts
from os.path import isfile, join
import numpy as np
import math
import pickle
import json


# Based on http://www.johndcook.com/standard_deviation.html
class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())

def main():
    siuts.create_dir(siuts.standardized_dataset_dir)
    start_time = time.time()
    current_file = 0
    rs = RunningStats()
    
    while isfile("{0}{1}-training_{2}.pickle".format(siuts.dataset_dir, siuts.species_list[0], current_file)):
        print()
        for specimen in siuts.species_list:
            path = "{0}{1}-training_{2}.pickle".format(siuts.dataset_dir, specimen, current_file)
            print("{} | {}".format(current_file, specimen))
            segments = siuts.load(path)
            for segment in segments:
                for row in segment:
                    for pixel in row:
                        rs.push(float(pixel))
        current_file += 1
    print("Finding stats took " + str(time.time() - start_time) + " seconds")
    mean = rs.mean()
    std = rs.standard_deviation()

    stats =  {"mean": mean, "std": std}
    
    print("mean={}".format(mean))
    print("std={}".format(std))
    with open(siuts.data_dir + "stats.json", 'w') as f:
        json.dump(stats, f)
    nr_of_files = current_file
    current_file = 0
    
    while current_file < nr_of_files:
        print()
        for specimen in siuts.species_list:
            path = "{0}{1}-training_{2}.pickle".format(siuts.dataset_dir, specimen, current_file)
            print("{} | {}".format(current_file, specimen))
            segments = siuts.load(path)
            standardized_segments = []
            for segment in segments:
                standardized_segment = []
                for row in segment:
                    standardized_row = []
                    for x in row:
                        standardized_row.append((x-mean)/std)
                    standardized_segment.append(standardized_row)
                standardized_segments.append(standardized_segment)
    	    with open("{0}{1}-training_{2}.pickle".format(siuts.standardized_dataset_dir, specimen, current_file), 'wb') as f:
                pickle.dump(np.array(standardized_segments, dtype=np.float16), f, protocol=-1)
        current_file += 1
    
    print("Standardizing testing data")
    standardize_pickled_segments("testing", mean, std)

    print("Standardizing validation data")
    standardize_pickled_segments("validation", mean, std)
    print("Everything took " + str(time.time() - start_time) + " seconds")


def standardize_pickled_segments(dataset_name, mean, std):
    path = "{}{}_data.pickle".format(siuts.dataset_dir, dataset_name)
    segments = siuts.load(path)
    standardized_segments = []
    for segment in segments:
        standardized_segment = []
        for row in segment:
            standardized_row = []
            for x in row:
                standardized_row.append((x-mean)/std)
            standardized_segment.append(standardized_row)
        standardized_segments.append(standardized_segment)

    with open("{}{}_data.pickle".format(siuts.standardized_dataset_dir, dataset_name), 'wb') as f:
        pickle.dump(np.array(standardized_segments, dtype=np.float16), f, protocol=-1)

    data = siuts.load("{}{}_labels.pickle".format(siuts.dataset_dir, dataset_name))
    with open("{}{}_labels.pickle".format(siuts.standardized_dataset_dir, dataset_name), 'wb') as f:
        pickle.dump(np.array(data, dtype=np.float16), f, protocol=-1)

    data = siuts.load("{}{}_rec_ids.pickle".format(siuts.dataset_dir, dataset_name))
    with open("{}{}_rec_ids.pickle".format(siuts.standardized_dataset_dir, dataset_name), 'wb') as f:
        pickle.dump(np.array(data, dtype=np.float16), f, protocol=-1)


if __name__ == "__main__":
    main()
