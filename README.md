# eDOCr

eDOCr is a packaged version of [keras-ocr](https://github.com/faustomorales/keras-ocr) that facilitates end-to-end digitization of mechanical EDs. Developed for Windows OS and using Python as the primary programming language. The implementation is discussed in the researh paper [Optical character recognition on engineering drawings to achieve automation in production quality control](https://www.frontiersin.org/articles/10.3389/fmtec.2023.1154132/full) 

## Getting Started

### Installation

The test environment I used is Python = 3.9 and TensorFlow == 2.13.0.


```bash
conda create -n edocr python=3.9 -y 
conda activate edocr

# To install from Source
cd path/to/your/folder
git clone https://github.com/yyyzzx7/eDOCr.git
cd eDOCr
pip install -r requirements.txt
pip install .
```

### Using 

There are two ways of using eDOCr: from terminal and from your own python file.

#### From Terminal:

```bash
python PATH/TO/YOUR/FOLDER/eDOCr/ocr_it.py PATH/TO/YOUR/DRAWING/my_drawing.pdf
```
Exp: (Note that the drawing.pdf is in the root folder)
```bash
python eDOCr/ocr_it.py drawing.pdf
```

#### From your own python file

More customization is possible using your own python file, such as selecting a different model, alphabet or changing colors.

```bash
# Importing packages
import os
from eDOCr import tools
import cv2
import string
from skimage import io

# Loading image and destination file
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
alphabet_infoblock = string.digits+string.ascii_letters+',.:-/'
model_infoblock = 'eDOCr/keras_ocr_models/models/recognizer_infoblock.h5'
alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols
model_gdts = 'eDOCr/keras_ocr_models/models/recognizer_gdts.h5'

# Selecting personalized color palette and cluster setting
color_palette = {'infoblock': (180, 220, 250), 'gdts': (94, 204, 243), 'dimensions': (93, 206, 175), 'frame': (167, 234, 82), 'flag': (241, 65, 36)}
cluster_t = 20

# eDOCr functions
class_list, img_boxes = tools.box_tree.findrect(img)
boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
io.imsave(os.path.join(dest_DIR, filename + '_process.jpg'), process_img)

infoblock_dict = tools.pipeline_infoblock.read_infoblocks(boxes_infoblock, img, alphabet_infoblock, model_infoblock)
gdt_dict = tools.pipeline_gdts.read_gdtbox1(gdt_boxes, alphabet_gdts, model_gdts, alphabet_dimensions, model_dimensions)
 
process_img = os.path.join(dest_DIR, filename + '_process.jpg')

dimension_dict = tools.pipeline_dimensions.read_dimensions(process_img, alphabet_dimensions, model_dimensions, cluster_t)
mask_img = tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict, cl_frame, color_palette)

# Record the results
io.imsave(os.path.join(dest_DIR, filename + '_boxes.jpg'), img_boxes)
io.imsave(os.path.join(dest_DIR, filename + '_mask.jpg'), mask_img)
tools.output.record_data(dest_DIR, filename, infoblock_dict, gdt_dict, dimension_dict)
```
