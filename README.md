# EvanescentWaves
## Transmittance vs Phase with different visibilities
The procedure for calculating the transmittance as the visibility between beams changes is as follows:
1. Create and rum simulations using COMSOL for the two beams separately.  Once they are done, export the fields at every point to .csv files using COMSOL's export options
2. COMSOL exports complex numbers as x + iy, but numpy uses x + jy.  To correct this, run fix_data.py on the files you exported.
3. Run interpolate_fields.py on the fixed files to obtain the interpolation of the electric field.  This ensures that the comparison between the fields of the two beams will match up on the grid
4. Run visibility.py to obtain the to combine the beams with different visibilities and calculate the transmittance as the phase difference between the beams is varied.  This can be done to get a range of visibilities or specific ones.  Note that the default behavior assumes that the intensity of the second beam is changed to alter the visibility.
