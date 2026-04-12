import os
import numpy as np
import mne
import h5py


path_eeg = '/media/ric23/Extreme SSD/Multimodal_dataset_hdf5'
file_path = os.path.join(path_eeg, 'sub-0047.h5')
print('path to eeg file =', file_path)

with h5py.File(file_path, 'r') as f:
    eeg = f['eeg_data'][:]

print(eeg.shape, eeg.shape[1]/500, ' seconds')
print(f"avg amp = {np.mean(eeg, axis=0)}")

sampling_rate = 500
all_ch_names_in_file = ['FC3', 'FC4', 'CP3', 'CP4', 'FT7', 'FT8', 'TP7', 'TP8']
info = mne.create_info(all_ch_names_in_file, sampling_rate, 'eeg')
raw = mne.io.RawArray(eeg, info, verbose=False)
raw.plot(block=True)

