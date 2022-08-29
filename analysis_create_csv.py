import os
import sys
import pandas
import numpy

def main(data_path):

    csv_path = data_path + '.csv'

    filenames = os.walk(data_path).__next__()[2]
    filenames.sort()
    df = pandas.DataFrame(filenames, columns=['filename'])
    df.to_csv(csv_path)

if __name__ == '__main__':

    main('/Users/hojinjang/Desktop/Datasets/FaceScrub')
