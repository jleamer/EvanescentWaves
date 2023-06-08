import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from scipy.integrate import simps
import time
from multiprocessing import Pool
from itertools import product, chain
from tqdm import tqdm


def fix_complex(filename, output):
    """
    Function for changing i to j in Comsol data to match Python conventions for complex numbers
    :param filename: name of file to be corrected
    :param output:   name of file to output to
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
    outfile.to_csv(output)
    return


def grab_data(path, cols):
    """
    function for getting the data exported by COMSOL
    :param path:    the path to the file
    :param cols:    the columns to use for headings of pandas dataframe
    :return:        the dataframe
    """
    df = pd.read_csv(path, sep=',', header=None, skiprows=1, names=cols)
    return df


def get_intensity(beam_df, phase_diff, tol, int_from_vis):
    # Get incident and transmitted slices of beam df data
    beam_df_i = beam_df.loc[beam_df.X < beam_df.X.min() + tol]
    print(beam_df_i)
    beam_df_t = beam_df.loc[beam_df.X > beam_df.X.max() - tol]
    print(beam_df_t)

    # Get components of fields
    Ex_i = beam_df_i['Ex @ ' + str(phase_diff)].to_numpy(dtype=complex)
    Ex_t = beam_df_t['Ex @ ' + str(phase_diff)].to_numpy(dtype=complex)
    Ey_i = beam_df_i['Ey @ ' + str(phase_diff)].to_numpy(dtype=complex)
    Ey_t = beam_df_t['Ey @ ' + str(phase_diff)].to_numpy(dtype=complex)
    Ez_i = beam_df_i['Ez @ ' + str(phase_diff)].to_numpy(dtype=complex)
    Ez_t = beam_df_t['Ez @ ' + str(phase_diff)].to_numpy(dtype=complex)

    # Add together with phase and sqrt(imin) coefficient
    comb_Ex_i = Ex_i + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ex_i
    comb_Ex_t = Ex_t + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ex_t
    comb_Ey_i = Ey_i + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ey_i
    comb_Ey_t = Ey_t + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ey_t
    comb_Ez_i = Ez_i + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ez_i
    comb_Ez_t = Ez_t + np.sqrt(int_from_vis) * np.exp(1j * phase_diff) * Ez_t

    # Calculate dot product manually
    dot_Ex_i = comb_Ex_i * comb_Ex_i.conj()
    dot_Ex_t = comb_Ex_t * comb_Ex_t.conj()
    dot_Ey_i = comb_Ey_i * comb_Ey_i.conj()
    dot_Ey_t = comb_Ey_t * comb_Ey_t.conj()
    dot_Ez_i = comb_Ez_i * comb_Ez_i.conj()
    dot_Ez_t = comb_Ez_t * comb_Ez_t.conj()
    I_i = dot_Ex_i + dot_Ey_i + dot_Ez_i
    I_t = dot_Ex_t + dot_Ey_t + dot_Ez_t

    return I_i, I_t


def get_power(arg):
    """
    function to combine fields and calculate power
    :param beam_field:      list of x, y, z components for beam field
    :param pos:             y, z linspace arrays used for interpolation of beam fields
    :param phase_diff:      difference of phase between beams
    :param int_from_vis:    intensity coefficient for beam 2 from visibility
    :return:                power through surface
    """
    #beam_field, pos, phase_diff, int_from_vis = args
    beam1_field, beam2_field, pos, phase_diff, beam1_int, beam2_int = arg
    # Add together beam components
    comb_Ex = np.sqrt(beam1_int) * beam1_field[0] + np.sqrt(beam2_int) * np.exp(1j * phase_diff) * beam2_field[0]
    comb_Ey = np.sqrt(beam1_int) * beam1_field[1] + np.sqrt(beam2_int) * np.exp(1j * phase_diff) * beam2_field[1]
    comb_Ez = np.sqrt(beam1_int) * beam1_field[2] + np.sqrt(beam2_int) * np.exp(1j * phase_diff) * beam2_field[2]

    # Perform dot product to get intensity
    I = comb_Ex * comb_Ex.conj() + comb_Ey * comb_Ey.conj() + comb_Ez * comb_Ez.conj()

    # Get power
    power = 0.5 * 3 * 8.854e-4 * simps(simps(I, pos[0]*1e-6), pos[1]*1e-6)
    return power


if __name__ == "__main__":
    # Set start time
    start = time.time()
    # First we want to do a sanity check to make sure that the two beam simulations from comsol are the same as adding
    # two 1 beam simulations together for each phase
    # Define filenames
    """
    onebeam_file = "2 beam/3D/Visibility/one_beam_solution,t=4um.csv"
    onebeam_out = "2 beam/3D/Visibility/one_beam_out.csv"
    twobeam_file = "2 beam/3D/Visibility/two_beam_solution,t=4um.csv"
    twobeam_out = "2 beam/3D/Visibility/two_beam_out.csv"

    # Define column names to use
    #   Psi will be changed later - needs to be set to 9 elements here to pull from data
    cols = ['X', 'Y', 'Z']
    psi = np.linspace(0, 2*np.pi, 9)
    for i in range(psi.shape[0]):
        cols.append('Ex @ ' + str(psi[i]))
        cols.append('Ey @ ' + str(psi[i]))
        cols.append('Ez @ ' + str(psi[i]))

    fix = False
    if fix:
        fix_complex(onebeam_file, onebeam_out)
        fix_complex(twobeam_file, twobeam_out)

    onebeam_df = grab_data(onebeam_out, cols)
    twobeam_df = grab_data(twobeam_out, cols)

    # Condition to check for linearity or not
    check_linearity = False
    if check_linearity:
        print("Linearity check: ", assert_linear_comb(onebeam_df, twobeam_df, psi))
    """
    # Define imax as the intensity peak at center of beam 1
    w0 = 10
    n = 1.5
    lda0 = 4
    xr = np.pi*w0**2*n/lda0
    wx = w0*np.sqrt(1 + 1**2/xr**2)
    E0 = 1
    i1 = np.array([np.abs(E0)**2 * w0**2/wx**2])

    #i2 = np.linspace(0, i1, 100)
    i2 = np.array([0.5757161])

    """
    # Check that intensity from adding two beams is the same as one two beam simulation
    tol = 1e-4
    check_intensity_linearity = False
    if check_intensity_linearity:
        inc_check, tra_check = assert_intensity_linear_comb(onebeam_df, twobeam_df, psi, tol)
        print("Incident: ", inc_check)
        print("Transmitted: ", tra_check)
    """""

    # Load interpolated fields for beams 1 and 2
    beam1_fields = dict(np.load("gap=1to2lda0,y=3.96um,z=4um,zpolarized,flipped,interp_fields,check.npz"))
    beam1_ifield = beam1_fields['ifield']
    ipos = beam1_fields['ipos']
    beam1_tfield = beam1_fields['tfield']
    tpos = beam1_fields['tpos']
    beam2_fields = dict(np.load("gap=1to2lda0,y=4um,z=4um,zpolarized,flipped,interp_fields,check.npz"))
    beam2_ifield = beam2_fields['ifield']
    beam2_ipos = beam2_fields['ipos']
    beam2_tfield = beam2_fields['tfield']
    beam2_tpos = beam2_fields['tpos']


    #for i in range(imin.shape[0]):
    #    for j in range(psi.shape[0]):
    #        ips[i][j] = get_power(ifield, ipos, psi[j], imin[i])
    #        tps[i][j] = get_power(tfield, tpos, psi[j], imin[i])

    #ips = []
    #tps = []

    #ips.append([pool.apply(get_power, args=(ifield, ipos, _psi, imin[0])) for _psi in psi])
    #ips.append([pool.apply(get_power, args=(ifield, ipos, _psi, imin[1])) for _psi in psi])
    #ips.append([pool.apply(get_power, args=(ifield, ipos, _psi, imin[2])) for _psi in psi])
    #ips = np.array(ips)

    #tps.append([pool.apply(get_power, args=(tfield, tpos, _psi, imin[0])) for _psi in psi])
    #tps.append([pool.apply(get_power, args=(tfield, tpos, _psi, imin[1])) for _psi in psi])
    #tps.append([pool.apply(get_power, args=(tfield, tpos, _psi, imin[2])) for _psi in psi])
    #tps = np.array(tps)

    # Redefine psis for calculations
    psi = np.linspace(0, 2*np.pi, 100)

    """
    print([_psi for _psi in psi])
    ips = np.array([
        [get_power(ifield, ipos, _psi, _imin) for _psi in psi]
        for _imin in imin
    ])
    tps = np.array([
        [get_power(tfield, tpos, _psi, _imin) for _psi in psi]
        for _imin in imin
    ])
    """
    with Pool() as pool:
        #args = ((ifield, ipos, _psi, _imin) for _psi, _imin in product(psi, imin))
        args = product([beam1_ifield], [beam2_ifield], [ipos], psi, i1, i2)
        ips = pool.map(get_power, args)
        ips = np.array(ips).reshape((psi.size, i2.size))

        #args = ((ifield, ipos, _psi, _imin) for _psi, _imin in product(psi, imin))
        args = product([beam1_tfield], [beam2_tfield], [tpos], psi, i1, i2)
        tps = pool.map(get_power, args)
        tps = np.array(tps).reshape((psi.size, i2.size))


    print("IP: ", ips)
    print("TP: ", tps)

    # Print elapsed time
    print("Time Elapsed: ", time.time()-start)

    # Export data to file
    incident_power_file = "2 beam/3D/Visibility/incident_vis,changing_gap_width,y1=3.96um,y2=4um,zpolarized,check,1.csv"
    transmitted_power_file = "2 beam/3D/Visibility/transmitted_vis,changing_gap_width,y1=3.96um,y2=4um,zpolarized,check,1.csv"
    out_cols = ["Phase (rad)"]
    ip_dict = {out_cols[0]: psi}
    tp_dict = {out_cols[0]: psi}
    for i in range(i2.shape[0]):
        out_cols.append(str(i2[i]))
        ip_dict[out_cols[i+1]] = ips.T[i]
        tp_dict[out_cols[i+1]] = tps.T[i]

    ip_df = pd.DataFrame(ip_dict, columns=out_cols)
    ip_df.to_csv(incident_power_file)
    tp_df = pd.DataFrame(tp_dict, columns=out_cols)
    tp_df.to_csv(transmitted_power_file)
