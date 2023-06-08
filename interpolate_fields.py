import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.integrate import simps
import time
from multiprocessing import Pool
from itertools import product, chain


def griddata_(_):
    return griddata(*_, method='nearest')


def interp_field(beam_df, phase_diff, tol):
    # Get incident and transmitted slices of beam df data
    beam_df_i = beam_df.loc[beam_df.X < beam_df.X.min() + tol]
    beam_df_t = beam_df.loc[beam_df.X > beam_df.X.max() - tol]

    # Get min and max Y and Z to create meshgrids
    iymin = beam_df_i.Y.min()
    iymax = beam_df_i.Y.max()
    izmin = beam_df_i.Z.min()
    izmax = beam_df_i.Z.max()
    iy, iz = np.linspace(iymin, iymax, 10000), np.linspace(izmin, izmax, 10000)
    iY, iZ = np.meshgrid(iy, iz)

    tymin = beam_df_t.Y.min()
    tymax = beam_df_t.Y.max()
    tzmin = beam_df_t.Z.min()
    tzmax = beam_df_t.Z.max()
    ty, tz = np.linspace(tymin, tymax, 10000), np.linspace(tzmin, tzmax, 10000)
    tY, tZ = np.meshgrid(ty, tz)

    # Get data for interpolation from dataframes
    i_Ex = beam_df_i['Ex @ ' + str(phase_diff[0])].to_numpy(dtype=complex)
    t_Ex = beam_df_t['Ex @ ' + str(phase_diff[0])].to_numpy(dtype=complex)
    i_Ey = beam_df_i['Ey @ ' + str(phase_diff[0])].to_numpy(dtype=complex)
    t_Ey = beam_df_t['Ey @ ' + str(phase_diff[0])].to_numpy(dtype=complex)
    i_Ez = beam_df_i['Ez @ ' + str(phase_diff[0])].to_numpy(dtype=complex)
    t_Ez = beam_df_t['Ez @ ' + str(phase_diff[0])].to_numpy(dtype=complex)

    # Get interpolations using griddata
    interp_i_Ex = griddata((beam_df_i.Y.to_numpy(), beam_df_i.Z.to_numpy()), i_Ex, (iY, iZ), method='nearest')
    interp_i_Ey = griddata((beam_df_i.Y.to_numpy(), beam_df_i.Z.to_numpy()), i_Ey, (iY, iZ), method='nearest')
    interp_i_Ez = griddata((beam_df_i.Y.to_numpy(), beam_df_i.Z.to_numpy()), i_Ez, (iY, iZ), method='nearest')

    interp_t_Ex = griddata((beam_df_t.Y.to_numpy(), beam_df_t.Z.to_numpy()), t_Ex, (tY, tZ), method='nearest')
    interp_t_Ey = griddata((beam_df_t.Y.to_numpy(), beam_df_t.Z.to_numpy()), t_Ey, (tY, tZ), method='nearest')
    interp_t_Ez = griddata((beam_df_t.Y.to_numpy(), beam_df_t.Z.to_numpy()), t_Ez, (tY, tZ), method='nearest')

    # Combine results to make processing easier
    ifield = [interp_i_Ex, interp_i_Ey, interp_i_Ez]
    ipos = [iy, iz]
    tfield = [interp_t_Ex, interp_t_Ey, interp_t_Ez]
    tpos = [ty, tz]

    return ifield, ipos, tfield, tpos


if __name__ == "__main__":
    # Specify files
    beam1_file = "comsol_output/onebeam_ypol_gap=1to2lda0_y=4um_z=4um.csv"
    beam2_file = "comsol_output/onebeam_ypol_gap=1to2lda0_y=3.96um_z=4um.csv"

    # Create columns for dataframe
    cols = ['X', 'Y', 'Z']
    psi = 0
    cols.append('Ex @ ' + str(psi))
    cols.append('Ey @ ' + str(psi))
    cols.append('Ez @ ' + str(psi))

    # Create dataframes (need to use df.apply to correct formatting for numpy later)
    beam1_df = pd.read_csv(beam1_file, sep=',', header=None, skiprows=9, names=cols)
    beam1_df = beam1_df.apply(lambda _: _.str.replace('i', 'j'), axis=1)

    beam2_df = pd.read_csv(beam2_file, sep=',', header=None, skiprows=1, names=cols)
    beam2_df = beam2_df.apply(lambda _: _.str.replace('i', 'j'), axis=1)

    # Interpolate fields for the incident and transmitted surfaces and save to file
    beam1_outfile = "interpolated_fields/beam1_ypol_gap=1to2lda0_y=4um_z=4um_interp_fields.npz"
    beam2_outfile = "interpolated_fields/beam2_ypol_gap=1to2lda0_y=3.96um_z=4um_interp_fields.npz"

    psi = np.linspace(0, 2 * np.pi, 10)
    beam1_ifield, beam1_ipos, beam1_tfield, beam1_tpos = interp_field(beam1_df, [0], 1e-4)

    np.savez(
        beam1_file,
        ifield=beam1_ifield,
        ipos=beam1_ipos,
        tfield=beam1_tfield,
        tpos=beam1_tpos,
    )

    beam2_ifield, beam2_ipos, beam2_tfield, beam2_tpos = interp_field(beam2_df, [0], 1e-4)
    np.savez(
        beam2_file,
        ifield=beam2_ifield,
        ipos=beam2_ipos,
        tfield=beam2_tfield,
        tpos=beam2_tpos,
    )