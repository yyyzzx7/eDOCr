# Importing packages
import os
from eDOCr import tools
import cv2
import string
from skimage import io

# Loading image and destination file
# dest_DIR = 'tests/test_Results'
# file_path = 'tests/test_samples/Candle_holder.jpg'
dest_DIR = './Results'
file_path = './drawing.png'
filename = os.path.splitext(os.path.basename(file_path))[0]
img = cv2.imread(file_path)

# Selecting alphabet and model (Note that alphabet and alphabet model need to match)
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
model_dimensions = 'eDOCr/keras_ocr_models/models/recognizer_dimensions.h5'
alphabet_infoblock = string.digits + string.ascii_letters + ',.:-/'
model_infoblock = 'eDOCr/keras_ocr_models/models/recognizer_infoblock.h5'
alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols
model_gdts = 'eDOCr/keras_ocr_models/models/recognizer_gdts.h5'

# Selecting personalized color palette and cluster setting
color_palette = {
    'infoblock': (180, 220, 250),
    'gdts': (94, 204, 243),
    'dimensions': (93, 206, 175),
    'frame': (167, 234, 82),
    'flag': (241, 65, 36)
}
cluster_t = 20

# eDOCr functions
class_list, img_boxes = tools.box_tree.findrect(img)
boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
io.imsave(os.path.join(dest_DIR, filename + '_process.jpg'), process_img)

infoblock_dict = tools.pipeline_infoblock.read_infoblocks(boxes_infoblock, img, alphabet_infoblock, model_infoblock)
gdt_dict = tools.pipeline_gdts.read_gdtbox1(gdt_boxes, alphabet_gdts, model_gdts, alphabet_dimensions, model_dimensions)

process_img = os.path.join(dest_DIR, filename + '_process.jpg')

dimension_dict = tools.pipeline_dimensions.read_dimensions(process_img, alphabet_dimensions, model_dimensions,
                                                           cluster_t)
mask_img = tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict, cl_frame, color_palette)

# Record the results
io.imsave(os.path.join(dest_DIR, filename + '_boxes.jpg'), img_boxes)
io.imsave(os.path.join(dest_DIR, filename + '_mask.jpg'), mask_img)
tools.output.record_data(dest_DIR, filename, infoblock_dict, gdt_dict, dimension_dict)