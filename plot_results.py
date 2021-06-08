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


#TODO: Figure out a better way of plotting this - currently causes seg faults
def surface(results, save):
    """
    File for plotting and saving 2D surface plots for each polarization
    :param results: location of the results
    :param save:    location where images should be stored
    :return:
    """
    i = 0
    for root, dirs, files in os.walk(results):
        for name in files:
            if ".csv" in name:
                # Read file using pandas
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep=',', header=8)

                # Get data for plotting
                x = df['X'].to_numpy()
                y = df['Y'].to_numpy()
                ex = df['Ex'].to_numpy()
                if isinstance(ex[0], str):
                    ex = np.array([ex[n].replace("i", "j") for n in range(ex.size)], dtype=complex)

                ey = df['Ey'].to_numpy()
                if isinstance(ey[0], str):
                    ey = np.array([ey[n].replace("i", "j") for n in range(ey.size)], dtype=complex)

                ez = df['Ez'].to_numpy()
                if isinstance(ez[0], str):
                    ez = np.array([ez[n].replace("i", "j") for n in range(ez.size)], dtype=complex)

                # contourf requires z values to be MxN matrix, so we need to interpolate
                # Create the mesh size using minimum and maximum values of x,y
                test_dict = {'X':x, 'Y':y, 'Ex':ex, 'Ey':ey.real, 'Ez':ez.real}
                test = pd.DataFrame(test_dict, columns=['X','Y','Ex','Ey','Ez'])

                if i == 0:
                    #test.to_csv('test.csv')
                    numpoints = x.size
                    X = np.linspace(x.min(), x.max(), numpoints)
                    Y = np.linspace(y.min(), y.max(), numpoints)
                    X, Y = np.meshgrid(X, Y)
                    interp_x = LinearNDInterpolator(list(zip(x, y)), ex)
                    EX = interp_x(X, Y)
                    plt.figure(1)
                    plt.contourf(X, Y, EX)
                    plt.colorbar()
                    plt.show()
                    print(ex.size)
                    print(ey.size)
                    print(ez.size)
                    print(x.size, y.size)
                    i += 1
    return

def x_line(results, save, x=0, tol=1e-4):
    """
    Function for plotting intensities along lines from data obtained using COMSOL
    :param results: filename for data from COMSOL
    :param x:       x value of the boundary to consider data from
    :param save:    destination for saving image files
    :return:
    """
    i = 0
    for root, dirs, files in os.walk(results):
        for name in files:
            if ".csv" in name:
                # Read in data from file
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep=',', header=0)
                max_x = df['X'].to_numpy().max()

                # Run query to get rows where X value is within x +/- tol
                rows = df.query('X-@max_x <= @tol & @max_x-X <= @tol').index.to_numpy()
                y = np.zeros(rows.size, dtype=complex)
                ex = np.zeros(rows.size, dtype=complex)
                ey = np.zeros(rows.size, dtype=complex)
                ez = np.zeros(rows.size, dtype=complex)
                for n in range(rows.size):
                    y[n] = df.at[rows[n], 'Y']
                    ex[n] = df.at[rows[n], 'Ex']
                    ey[n] = df.at[rows[n], 'Ey']
                    ez[n] = df.at[rows[n], 'Ez']

                # Turn field values into intensities
                ix = ex * ex.conj()
                iy = ey * ey.conj()
                iz = ez * ez.conj()

                plt.figure(i)
                plt.title(filename)
                plt.plot(y, ix, label='X Pol.')
                plt.plot(y, iy, label='Y Pol.')
                plt.plot(y, iz, label='Z Pol.')
                plt.ylabel("I")
                plt.xlabel("y(um)")
                plt.legend(numpoints=1)

                i += 1
    return

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

def reflectance(results, save, tol=1e-4):
    """
    Function for calculating the reflectance and transmittance curves for light at the outputs of our prism arrangement
    using COMSOL simulation data
    :param results: directory for the data
    :param save:    location to store figure
    :param tol:     tolerance for picking data along boundaries
    :return:
    """
    # Create lists for peak input intensity and peak output intensity along either boundary
    peak_input = []
    peak_right = []
    peak_top = []
    reflected = []
    transmittance = []
    i = 0
    for root, dirs, files in os.walk(results):
        for name in files:
            if ".csv" in name:
                # Read in data from file
                filename = os.path.join(root, name)
                df = pd.read_csv(filename, sep=',', header=0)

                # Get x and y values for the top and right output boundaries and input
                max_x = df['X'].to_numpy().max()
                max_y = df['Y'].to_numpy().max()
                min_x = df['X'].to_numpy().min()

                # Filter data along boundaries
                right_rows = df.query('X-@max_x < @tol & @max_x-X < @tol').index.to_numpy()
                top_rows = df.query('Y-@max_y < @tol & @max_y-Y < @tol').index.to_numpy()
                left_rows = df.query('X-@min_x < @tol & @min_x-X < @tol').index.to_numpy()

                # Input polarizations and get amplitude
                in_x, in_y, in_ex, in_ey, in_ez = get_data(df, left_rows)
                #in_amp = in_ex * in_ex.conj() + in_ey * in_ey.conj() + in_ez * in_ez.conj()

                # Right output polarizations and get amplitude
                rout_x, rout_y, rout_ex, rout_ey, rout_ez = get_data(df, right_rows)
                #r_amp = rout_ex * rout_ex.conj() + rout_ey * rout_ey.conj() + rout_ez * rout_ez.conj()

                # Top output polarizations and get amplitude
                tout_x, tout_y, tout_ex, tout_ey, tout_ez = get_data(df, top_rows)
                #t_amp = tout_ex * tout_ex.conj() + tout_ey * tout_ey.conj() + tout_ez * tout_ez.conj()

                # Interpolate data so that every array has the same size
                numpoints = 300
                Y = np.linspace(in_y.min(), in_y.max(), numpoints)
                in_Y = np.linspace(in_y.min(), in_y.max(), in_y.size)
                in_interx = interp1d(in_Y, in_ex, kind='cubic')
                in_exn = in_interx(Y)
                in_intery = interp1d(in_Y, in_ey, kind='cubic')
                in_eyn = in_intery(Y)
                in_interz = interp1d(in_Y, in_ez, kind='cubic')
                in_ezn = in_interz(Y)

                X = np.linspace(tout_x.min(), tout_x.max(), numpoints)
                tout_X = np.linspace(tout_x.min(), tout_x.max(), tout_x.size)
                tout_interx = interp1d(tout_X, tout_ex, kind='cubic')
                tout_exn = tout_interx(X)
                tout_intery = interp1d(tout_X, tout_ey, kind='cubic')
                tout_eyn = tout_intery(X)
                tout_interz = interp1d(tout_X, tout_ez, kind='cubic')
                tout_ezn = tout_interz(X)

                rout_Y = np.linspace(rout_y.min(), rout_y.max(), rout_y.size)
                #Y = np.linspace(rout_y.min(), rout_y.max(), numpoints)
                rout_interx = interp1d(rout_Y, rout_ex, kind='cubic')
                rout_exn = rout_interx(Y)
                rout_intery = interp1d(rout_Y, rout_ey, kind='cubic')
                rout_eyn = rout_intery(Y)
                rout_interz = interp1d(rout_Y, rout_ez, kind='cubic')
                rout_ezn = rout_interz(Y)

                refly = tout_eyn.max()/in_eyn.max()
                reflected.append(refly)
                trany = rout_eyn.max()/in_eyn.max()
                transmittance.append(trany)

                #plt.figure(1)
                #plt.plot(Y, in_eyn, label="Interp")
                #plt.plot(in_y, in_ey, '*', label="Data")

                #plt.plot(Y, rout_eyn, label="Reflected Y Pol")
                #plt.plot(rout_y, rout_ey, '*-', label="Data")

                #plt.plot(X, tout_eyn, label="Transmitted Y Pol")
                #plt.plot(tout_x, tout_ey, '*-', label="Data")
                #plt.legend(numpoints=1)
                #plt.show()
                #return


    plt.figure(1)
    plt.plot(reflected, label='Reflectance')
    plt.plot(transmittance, label='Transmittance')
    plt.legend(numpoints=1)
    plt.xlabel("Run #")
    plt.ylabel("Ratio")

    plt.figure(2)
    plt.plot(transmittance, label='Transmittance')
    plt.plot(1-np.array(reflected), label='Theoretical')
    plt.legend(numpoints=1)
    plt.show()

    return





ypath = "YPol"
zpath = "ZPol"
#x_line(ypath, '', x=10, tol=1e-3)
#x_line(zpath, '', x=10, tol=1e-3)
#compare_pols(ypath, zpath, '')
#surface(ypath, '')
reflectance(ypath, '', tol=1e-4)