import pydicom
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def getIMA(path="/Users/zane/PycharmProjects/ddpm_CTLDR/tmpdatasets"):
    IMAfiles = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.ima') or filename.endswith('.IMA'):
                IMAfiles += [os.path.join(path, filename)]
    else:
        IMAfiles += [path]

    for IMAfile in IMAfiles:
        dcm = pydicom.dcmread(IMAfile)
        image_data = dcm.pixel_array

        plt.imshow(image_data, cmap='gray')
        plt.show()

        # 归一化到0-255 uint8
        min_value = np.min(image_data)
        max_value = np.max(image_data)
        normalized_image_data = (image_data - min_value) / (max_value - min_value) * 255
        normalized_image_data = normalized_image_data.astype(np.uint8)

        Image.fromarray(normalized_image_data).save(IMAfile.split(".")[0] + '.png')


if __name__ == "__main__":
    getIMA()
