# import cv2
# import numpy as np
# import os
# from craft_text_detector import empty_cuda_cache
# from craft import CRAFTTextDetector

# image_dir = 'vietsignboard'
# output_dir = 'vietsignboard/craft_output'
# bound_types = ['polys', 'boxes']


# def get_craft_regions():
#     for img in os.listdir(image_dir):
#         image_path = os.path.join(image_dir, img)
#         print(image_path)
#         for bound in bound_types:
#             out_path = os.path.join(output_dir, os.path.splitext(img)[0], bound)
#             craft_run = CRAFTTextDetector(image_path=image_path, output_dir=out_path,boundary_type=bound)
#             craft_run.run()

#     empty_cuda_cache()

# get_craft_regions()

import cv2
import numpy as np
import os
from craft_text_detector import empty_cuda_cache
from craft import CRAFTTextDetector

image_dir = 'vietsignboard'
output_dir = 'vietsignboard/craft_output'
bound_types = ['polys', 'boxes']

def get_craft_regions():
    img = 'baker-2.jpg'  # Specify only this image
    image_path = os.path.join(image_dir, img)
    print(image_path)
    for bound in bound_types:
        out_path = os.path.join(output_dir, os.path.splitext(img)[0], bound)
        craft_run = CRAFTTextDetector(image_path=image_path, output_dir=out_path, boundary_type=bound)
        craft_run.run()

    empty_cuda_cache()

get_craft_regions()
