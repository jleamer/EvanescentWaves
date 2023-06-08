import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    # Create variable for directory name
    directory_name = "comsol_output"

    # Loop over files in the specified directory
    for filename in os.listdir(directory_name):
        # Load file with pandas
        path = directory_name + "/" + filename
        df = pd.read_csv(path, delimiter=',', skiprows=9)

        # Use df.apply to replace 'i's with 'j's for use with numpy
        df = df.apply(lambda _: _.str.replace('i', 'j'), axis=1)

        # Convert to numpy array
        data = df.to_numpy(dtype=complex)

        # Save as compressed file
        out_path = directory_name + "/" + filename[:len(filename)-4]
        np.save(out_path, data)

