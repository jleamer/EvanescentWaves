import os
import numpy as np
import pandas as pd

########################################################
#
#   File for cleaning headers from COMSOL results for evanescent waves
#
########################################################


def process_files(path, line, cols):
    """
    Function that reads all files in a directory and fixes the headers using the line number and columns
    :param path: directory of files
    :param line: the line number the header should be
    :param cols: the string for the column headings
    :return:
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if ".csv" in name:
                # Read from file and replace specified line with columns heading
                filename = os.path.join(root, name)
                f = open(filename, 'r')
                content = f.readlines()
                f.close()
                content[line-1] = cols+'\n'

                # convert the list of strings into 1 string
                content = ''.join([str(elem) for elem in content])

                # open the file again and overwrite with modified content
                f = open(filename, 'w')
                f.write(content)
                f.close()

    return


def fix_complex(path):
    """
    Function for changing i to j in Comsol data to match Python conventions for complex numbers
    :param path: directory of files
    :return:
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if ".csv" in name:
                # Read the data into dataframe object
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep=',', header=8)

                # Can ignore X and Y because they're stored as real numbers
                # Turn Ex, Ey, Ez into arrays
                ex = df['Ex'].to_numpy()
                if isinstance(ex[0], str):
                    ex = np.array([ex[n].replace('i', 'j') for n in range(ex.size)], dtype=complex)

                ey = df['Ey'].to_numpy()
                if isinstance(ey[0], str):
                    ey = np.array([ey[n].replace('i', 'j') for n in range(ey.size)], dtype=complex)

                ez = df['Ez'].to_numpy()
                if isinstance(ez[0], str):
                    ez = np.array([ez[n].replace('i', 'j') for n in range(ez.size)], dtype=complex)

                # Create new dataframe using the fixed data
                dict = {'X': df['X'].to_numpy(), 'Y': df['Y'].to_numpy(), 'Ex': ex, 'Ey': ey, 'Ez': ez, 'd': df['d'].to_numpy()}
                cols = ['X', 'Y', 'Ex', 'Ey', 'Ez', 'd']
                out = pd.DataFrame(dict, columns=cols)
                out.to_csv(filename)
    return



# specify paths for directories of results
ypath = "YPol"
zpath = "ZPol"

# specify header line and columns
header_line = 9
cols = "X,Y,Ex,Ey,Ez,d"

# process files
process_files(ypath, header_line, cols)
process_files(zpath, header_line, cols)

# fix complex numbers
fix_complex(ypath)
fix_complex(zpath)
