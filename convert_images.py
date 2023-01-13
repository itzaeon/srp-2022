# Credits to documentation https://github.com/carsales/pyheif

import os
import pyheif
from PIL import Image


for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('.HEIC'):    
            print(os.path.join(root, file))

            heif_file = pyheif.read(os.path.join(root, file))
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
                )
            image.save(file[:-5] + ".jpg", "JPEG")
