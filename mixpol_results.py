import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import numpy as np
import os

########################################################
#
#   File for plotting results from COMSOL evanescent wave simulations with mixed polarization
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


def fix_complex(filename):
    """
    Function for changing i to j in Comsol data to match Python conventions for complex numbers
    :param filename: name of file to be corrected
    :return:
    """
    dict = {}
    df = pd.read_csv(filename, sep=',', header=8)
    cols = df.columns
    for col in cols:
        temp = df[col].to_numpy()
        if isinstance(temp[0], str):
            temp = np.array([temp[n].replace('i', 'j') for n in range(temp.size)])
        dict[col] = temp

    outfile = pd.DataFrame(dict, columns=cols)
    outfile.to_csv("MixPol_theta/pd_out.csv")
    return


def incident_edge(filename, theta, tol=1e-4, numpoints=600):
    """
    Function for obtaining field values on the incident boundary from COMSOL data files
    :param filename:    name of data file to get
    :param theta:       angle of polarization that we want data for
    :param tol:         tolerance in x value for getting field values on boundary
    :param numpoints:   number of points to use in interpolation
    :return:            y values and polarization vector for field
    """
    # Get rows that are on boundary
    df = pd.read_csv(filename, sep=',', header=0)
    min_X = df['X'].to_numpy().min()
    rows = df.query('X-@min_X <= @tol & -(@min_X-X) <= @ @tol').index.to_numpy()

    # Read out relevant data
    y = np.zeros(rows.size-1, dtype=complex)
    Ey = np.zeros(rows.size-1, dtype=complex)
    Ez = np.zeros(rows.size-1, dtype=complex)

    for n in range(rows.size-1):
        y[n] = df.at[rows[n], 'Y']
        Ey[n] = df.at[rows[n], 'Ey[' + str(theta) + ']']
        Ez[n] = df.at[rows[n], 'Ez[' + str(theta) + ']']

    # Interpolate Ey, Ez so that the final array has size numpoints
    fy = interp1d(y, Ey, kind='linear')
    fz = interp1d(y, Ez, kind='linear')

    # Create new array of y values and use interpolation functions
    ymin = y.min()  # df['Y'].to_numpy().min()
    ymax = y.max()  # df['Y'].to_numpy().max()
    iny = np.linspace(ymin, ymax, numpoints)
    inEy = fy(iny)
    inEz = fz(iny)

    E = np.zeros((numpoints, 2), dtype=complex)
    for n in range(numpoints):
        E[n][0] = inEy[n]
        E[n][1] = inEz[n]

    return iny, E


def transmitted_edge(filename, theta, tol=1e-4, numpoints=600):
    """
    Function for obtaining field values on the transmitted boundary from COMSOL data files
    :param filename:    name of data file to get
    :param theta:       angle of polarization
    :param tol:         tolerance in x value for getting field values on boundary
    :param numpoints:   number of points to use in interpolation
    :return:            y values and polarization vector for field
    """
    # Get rows that are on boundary
    df = pd.read_csv(filename, sep=',', header=0)
    max_X = df['X'].to_numpy().max()
    rows = df.query('X-@max_X <= @tol & @max_X-X <= @ @tol').index.to_numpy()

    #Read out relevant data
    y = np.zeros(rows.size, dtype=complex)
    Ey = np.zeros(rows.size, dtype=complex)
    Ez = np.zeros(rows.size, dtype=complex)

    for n in range(rows.size):
        y[n] = df.at[rows[n], 'Y']
        Ey[n] = df.at[rows[n], 'Ey[' + str(theta) + ']']
        Ez[n] = df.at[rows[n], 'Ez[' + str(theta) + ']']


    # Interpolate Ey, Ez so that the final array has size numpoints
    fy = interp1d(y, Ey)
    fz = interp1d(y, Ez)

    # Create new array of y values and use interpolation functions
    ymin = 0  #df['Y'].to_numpy().min()
    ymax = 10 #df['Y'].to_numpy().max()
    iny = np.linspace(ymin, ymax, numpoints)
    inEy = fy(iny)
    inEz = fz(iny)

    E = np.zeros((numpoints, 2), dtype=complex)
    for n in range(numpoints):
        E[n][0] = inEy[n]
        E[n][1] = inEz[n]

    return iny, E


def reflected_edge(filename, theta, tol=1e-4, numpoints=600):
    """
    function for obtaining the field values from the COMSOL simulation along reflected edge
    :param filename:    name of the file data is stored in
    :param theta:       angle of polarization
    :param tol:         tolerance to use when grabbing the appropriate rows
    :param numpoints:   number of points to use in interpolation
    :return:            array of x values and vector field
    """
    # Get rows that are on the boundary
    df = pd.read_csv(filename, sep=',', header=0)
    max_Y = df['Y'].to_numpy().max()
    rows = df.query("Y-@max_Y <= @tol & @max_Y-Y <= @tol").index.to_numpy()

    # Arrays for storing data
    x = np.zeros(rows.size, dtype=complex)
    Ex = np.zeros(rows.size, dtype=complex)
    Ez = np.zeros(rows.size, dtype=complex)

    # Store data
    for i in range(rows.size):
        x[i] = df.at[rows[i], 'X']
        Ex[i] = df.at[rows[i], 'Ex[' + str(theta) + ']']
        Ez[i] = df.at[rows[i], 'Ez[' + str(theta) + ']']

    # Interpolate to make sure data are the right-size
    fx = interp1d(x, Ex)
    fz = interp1d(x, Ez)

    # Create new arrays using interpolation functions
    xmin = 0
    xmax = 10
    x = np.linspace(xmin, xmax, numpoints)
    inEx = fx(x)
    inEz = fz(x)

    # Construct vector to return
    E = np.zeros((numpoints, 2), dtype=complex)
    for i in range(numpoints):
        E[i][0] = inEx[i]
        E[i][1] = inEz[i]

    return x, E


def get_J(E, numpoints=600):
    """
    function to construct the coherency matrices from yE, zE and get DOP
    :param E:       the electric field vector along boundary
    :return:        the coherency matrices
    """
    j = np.zeros((numpoints, 2, 2), dtype=complex)
    for n in range(numpoints):
        #j[n] = alpha ** 2 * np.outer(yE[n], yE[n].conj().T) + beta ** 2 * np.outer(zE[n], zE[n].conj().T)
        #A = alpha * yE + beta * zE
        #j[n] = np.outer(A, A.conj().T)
        j[n] = np.outer(E[n], E[n].conj().T)

    return j


def get_DOP(j, x, numpoints=600):
    """
    function to get DOP from coherency matrices in j
    :param j:           array of coherency matrices
    :param x:           the position values
    :param numpoints:   number of points to use in array for DOP
    :return:            array of DOP values
    """
    s0 = j[:, 0, 0] + j[:, 1, 1]
    s1 = j[:, 0, 0] - j[:, 1, 1]
    s2 = j[:, 0, 1] + j[:, 1, 0]
    s3 = -1j*(j[:, 0, 1] - j[:, 1, 0])
    dop = np.sqrt(s1**2 + s2**2 + s3**2)/s0
    power = integrate.trapezoid(s0, x)
    return dop, s0, power


if __name__ == "__main__":
    # Define theta and alpha values that will be used
    theta = np.arange(0, 195, 15)
    alpha = np.cos(theta*np.pi/180)

    # Code to process files - only necessary if COMSOL data hasn't been processed yet
    process = False
    if process:
        cols = "X,Y"
        for t in theta:
            cols += ',Ex[' + str(t) + ']'
            cols += ',Ey[' + str(t) + ']'
            cols += ',Ez[' + str(t) + ']'

        path = "MixPol_theta"
        process_files(path, 9, cols)

        fix_complex(path + "/mixpol.csv")

    filename = "MixPol_theta/pd_out.csv"
    tp = np.zeros(theta.size, dtype=complex)
    rp = np.zeros(theta.size, dtype=complex)
    ip = np.zeros(theta.size, dtype=complex)
    for i in range(theta.size):
        yt, Et = transmitted_edge(filename, theta[i])
        jt = get_J(Et)
        tp[i] = get_DOP(jt, yt)[2]

        yi, Ei = incident_edge(filename, theta[i])
        ji = get_J(Ei)
        ip[i] = get_DOP(ji, yi)[2]

        xr, Er = reflected_edge(filename, theta[i])
        jr = get_J(Er)
        rp[i] = get_DOP(jr, xr)[2]

    plt.plot(alpha, tp, label="Transmitted")
    plt.plot(alpha, rp, label="Reflected")
    plt.plot(alpha, ip, label="Incident")
    plt.plot(alpha, tp+rp, 'o', label="T+R")
    plt.title("Power vs Cos(theta)")
    plt.xlabel("cos(theta)")
    plt.ylabel("Power")
    plt.legend(numpoints=1)
    plt.show()