import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import numpy as np
import os

########################################################
#
#   File for plotting results from COMSOL evanescent wave simulations
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


def get_data(df, rows):
    """
    Function for getting data from a DataFrame along rows
    :param df:      DataFrame object to pull from
    :param rows:    list of rows to pull data from
    :return:        arrays of data at rows in df - we want x, y, Ex, Ey, Ez
    """
    x = np.array([df.at[row, 'X'] for row in rows])
    y = np.array([df.at[row, 'Y'] for row in rows])
    ex = np.array([df.at[row, 'Ex'] for row in rows], dtype=complex)
    ey = np.array([df.at[row, 'Ey'] for row in rows], dtype=complex)
    ez = np.array([df.at[row, 'Ez'] for row in rows], dtype=complex)
    return x, y, ex, ey, ez


def compare_pols(pol1, pol2, save, tol=1e-4):
    """
    Function for comparing the intensity output from COMSOL data for 2 different polarizations
    :param pol1: directory for data from first polarization simulation
    :param pol2: directory for data from second polarization simulation
    :param save: directory where figures where will be saved
    :param tol:  tolerance in selecting the boundary to compare over
    :return:
    """
    # Get filenames
    pol1_files = []
    for root, dirs, files in os.walk(pol1):
        for name in files:
            if '.csv' in name:
                pol1_files.append(os.path.join(root, name))

    pol2_files = []
    for root, dirs, files in os.walk(pol2):
        for name in files:
            if '.csv' in name:
                pol2_files.append(os.path.join(root, name))

    pol1_files.sort()
    pol2_files.sort()
    print(pol1_files)
    print(pol2_files)
    for i in range(len(pol1_files)):
        # Create dataframes from files
        pol1_df = pd.read_csv(pol1_files[i], sep=',', header=0)
        pol2_df = pd.read_csv(pol2_files[i], sep=',', header=0)

        max_x1 = pol1_df['X'].to_numpy().max()
        max_x2 = pol2_df['X'].to_numpy().max()

        rows1 = pol1_df.query('X-@max_x1 < @tol & @max_x1-X < @tol').index.to_numpy()
        rows2 = pol2_df.query('X-@max_x2 < @tol & @max_x2-X < @tol').index.to_numpy()

        # Create arrays for storing data and grab data
        y1 = np.zeros(rows1.size)
        ex1 = np.zeros(rows1.size, dtype=complex)
        ey1 = np.zeros(rows1.size, dtype=complex)
        ez1 = np.zeros(rows1.size, dtype=complex)
        for n in range(rows1.size):
            y1[n] = pol1_df.at[rows1[n], 'Y']
            ex1[n] = pol1_df.at[rows1[n], 'Ex']
            ey1[n] = pol1_df.at[rows1[n], 'Ey']
            ez1[n] = pol1_df.at[rows1[n], 'Ez']

        y2 = np.zeros(rows2.size)
        ex2 = np.zeros(rows2.size, dtype=complex)
        ey2 = np.zeros(rows2.size, dtype=complex)
        ez2 = np.zeros(rows2.size, dtype=complex)
        for n in range(rows2.size):
            y2[n] = pol2_df.at[rows2[n], 'Y']
            ex2[n] = pol2_df.at[rows2[n], 'Ex']
            ey2[n] = pol2_df.at[rows2[n], 'Ey']
            ez2[n] = pol2_df.at[rows2[n], 'Ez']

        # Get intensities
        ix1 = ex1 * ex1.conj()
        iy1 = ey1 * ey1.conj()
        iz1 = ez1 * ez1.conj()

        ix2 = ex2 * ex2.conj()
        iy2 = ey2 * ey2.conj()
        iz2 = ez2 * ez2.conj()

        plt.figure(i)
        plt.title(pol1_files[i] + pol2_files[i])
        plt.plot(y1, ix1, label="X Pol 1")
        plt.plot(y1, iy1, label='Y Pol 1')
        plt.plot(y1, iz1, label='Z Pol 1')
        plt.plot(y2, ix2, '--', label='X Pol 2')
        plt.plot(y2, iy2, '--', label='Y Pol 2')
        plt.plot(y2, iz2, '--', label='Z Pol 2')
        plt.legend(numpoints=1)

    plt.show()
    return


def transmitted_edge(filename, tol=1e-4, numpoints=600):
    """
    Function for obtaining field values on the transmitted boundary from COMSOL data files
    :param filename:    name of data file to get
    :param tol:         tolerance in x value for getting field values on boundary
    :param numpoints:   number of points to use in interpolation
    :return:            y values and polarization vector for field
    """
    # Get rows that are on boundary
    df = pd.read_csv(filename, sep=',', header=0)
    max_X = df['X'].to_numpy().max()
    rows = df.query('X-@max_X <= @tol & @max_X-X <= @ @tol').index.to_numpy()

    # Read out relevant data
    y = np.zeros(rows.size, dtype=complex)
    Ey = np.zeros(rows.size, dtype=complex)
    Ez = np.zeros(rows.size, dtype=complex)

    for n in range(rows.size):
        y[n] = df.at[rows[n], 'Y']
        Ey[n] = df.at[rows[n], 'Ey']
        Ez[n] = df.at[rows[n], 'Ez']

    # Interpolate Ey, Ez so that the final array has size numpoints
    fy = interp1d(y, Ey)
    fz = interp1d(y, Ez)

    # Create new array of y values and use interpolation functions
    ymin = df['Y'].to_numpy().min()
    ymax = df['Y'].to_numpy().max()
    iny = np.linspace(ymin, ymax, numpoints)
    inEy = fy(iny)
    inEz = fz(iny)

    E = np.zeros((numpoints, 2), dtype=complex)
    for n in range(numpoints):
        E[n][0] = inEy[n]
        E[n][1] = inEz[n]

    return iny, E


def get_J(yE, zE, alpha, beta, numpoints=600):
    """
    function to construct the coherency matrices from yE, zE and get DOP
    :param yE:      the data from initially y-polarized field
    :param zE:      the data from initially z-polarized field
    :param alpha:   normalization constant for yE (1 + alpha^2 = beta^2)
    :param beta:    normalization constant for zE (1 + alpha^2 = beta^2)
    :return:        the coherency matrices
    """
    j = np.zeros((numpoints, 2, 2), dtype=complex)
    for n in range(numpoints):
        j[n] = alpha ** 2 * np.outer(yE[n], yE[n].conj().T) + beta ** 2 * np.outer(zE[n], zE[n].conj().T)

    return j


def get_DOP(j, numpoints=600):
    """
    function to get DOP from coherency matrices in j
    :param j:           array of coherency matrices
    :param numpoints:   number of points to use in array for DOP
    :return:            array of DOP values
    """
    dop = np.zeros(numpoints, dtype=complex)
    for n in range(numpoints):
        s0 = j[n][0][0] + j[n][1][1]
        s1 = j[n][0][0] - j[n][1][1]
        s2 = j[n][0][1] + j[n][1][0]
        s3 = -1j*(j[n][0][1] - j[n][1][0])
        dop[n] = np.sqrt(s1**2 + s2**2 + s3**2)/s0

    return dop

ypath = "YPol"
zpath = "ZPol"

test = "YPol/ypol_thickness_1.1E-7.csv"
test2 = "ZPol/zpol_thickness_1.1E-7.csv"
y, yE = transmitted_edge(test)
y, zE = transmitted_edge(test2)
alpha = 1/np.sqrt(2)
beta = 1/np.sqrt(2)
j = get_J(yE, zE, alpha, beta)
dop = get_DOP(j)

plt.plot(y, dop)
plt.ylabel("DOP")
plt.xlabel("y(um)")
plt.show()
