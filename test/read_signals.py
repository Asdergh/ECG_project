import torch as th
import matplotlib.pyplot as plt
import h5py as h5
from torchaudio.transforms import Spectrogram
from scipy.signal import spectrogram
import numpy as np
import cv2
from utils import convert_to_spec


file = h5.File("C:\\Users\\1\\Downloads\\exams_part0\\exams_part0.hdf5", "r")

data = file["tracings"]
print(data.shape, th.Tensor(data).size())
signal_s = th.Tensor(data[0, :, 0])
signals_s = th.Tensor(data[0, :, :])
# signals_s = signals_s.max(dim=0)[0]

specs = convert_to_spec(signal_s)
specs_s = convert_to_spec(signals_s)
specs_s = specs_s.mean(dim=0)
print(specs.size(), specs_s.size())
# f, t, Sxx1 = spectrogram(th.Tensor(data[0, :, 0]))
# Sxx1 = Sxx1 / np.max(Sxx1)
# print(f.shape, t.shape, np.max(Sxx1))
# _, _, Sxx2 = spectrogram(th.Tensor(data[0, :, 1]))
# _, _, Sxx3 = spectrogram(th.Tensor(data[0, :, 2]))
# Sxx1 = th.Tensor(cv2.resize(Sxx1, (215, 215)))
# Sxx2 = cv2.resize(Sxx2, (129, 129))
# Sxx3 = cv2.resize(Sxx3, (215, 215))

# image_fft = np.stack((Sxx1, Sxx2, Sxx3), axis=2)
# print(image_fft.shape)
_, axi = plt.subplots(nrows=3)
axi[0].imshow(specs, cmap="jet")
axi[1].imshow(specs_s, cmap="jet")
# axi[2].imshow(Sxx3, cmap="jet")
# axi[2].imshow(image_fft)

plt.show()
#  
# print(signals.shape)