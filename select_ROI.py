import numpy as np
import cv2
import array as arr
import dicom
import pydicom
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
import plotly.express as px
from PIL import Image, ImageOps

#reading the image

dicom_file = dicom.read_file('assets/dicomdirectory/imatgeprova.dcm') ## original dicom File
#### a dicom monochrome-2 file has pixel value between approx -2000 and +2000, opencv doesn't work with it#####
#### in a first step we transform those pixel values in (R,G,B)
### to have gray in RGB, simply give the same values for R,G, and B,
####(0,0,0) will be black, (255,255,255) will be white,

## the threeshold to be automized with a proper quartile function of the pixel distribution
black_threeshold=0###pixel value below 0 will be black,
white_threeshold=1400###pixel value above 1400 will be white
wt=white_threeshold
bt=black_threeshold

###### function to transform a dicom to RGB for the use of opencv,
##to be strongly improved, as it takes to much time to run,
## and the linear process should be replaced with an adapted weighted arctan or an adapted spline interpolation.
def DicomtoRGB(dicomfile,bt,wt):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((dicomfile.Rows, dicomfile.Columns, 3), np.uint8)
    #loops on image height and width
    i=0
    j=0
    while i<dicomfile.Rows:
        j=0
        while j<dicomfile.Columns:
            color = yaxpb(dicom_file.pixel_array[i][j],bt,wt) #linear transformation to be adapted
            image[i][j] = (color,color,color)## same R,G, B value to obtain greyscale
            j=j+1
        i=i+1
    return image
##linear transformation : from [bt < pxvalue < wt] linear to [0<pyvalue<255]: loss of information...
def yaxpb(pxvalue,bt,wt):
    if pxvalue < bt:
        y=0
    elif pxvalue > wt:
        y=255
    else:
        y=pxvalue*255/(wt-bt)-255*bt/(wt-bt)
    return y



image=DicomtoRGB(dicom_file,bt=0,wt=1400)
## loading the RGB in a proper opencv format
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("assets/gray.jpg", gray)
## look at the gray file
"""
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyWindow("gray")
"""
# Create image
img = gray
class ExtractImageWidget(object):
    def __init__(self):
        self.original_image = img

        # Resize image, remove if you want raw image size
        self.original_image = cv2.resize(self.original_image, (640, 556))
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.angle = 0
        self.extract = False
        self.selected_ROI = False

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False

            self.selected_ROI = True
            self.crop_ROI()

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click and reset angle
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.angle = 0
            self.selected_ROI = False

    def show_image(self):
        return self.clone

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.clone.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]

            self.cropped_image = self.cropped_image[y1:y2, x1:x2]

            print('Cropped image: {} {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
        else:
            print('Select ROI to crop before cropping')


    def show_cropped_ROI(self):
        cv2.imshow('cropped image', self.cropped_image)
        cv2.imwrite('crop.jpg', self.cropped_image)
        print('Saved!')

if __name__ == '__main__':
    extract_image_widget = ExtractImageWidget()
    while True:
        cv2.imshow('image', extract_image_widget.show_image())
        key = cv2.waitKey(1)

        # Rotate clockwise 5 degrees
        if key == ord('r'):
            extract_image_widget.rotate_image(5)

        # Rotate counter clockwise 5 degrees
        if key == ord('e'):
            extract_image_widget.rotate_image(-5)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

        # Crop image
        if key == ord('c'):
            cropped_image= extract_image_widget.show_cropped_ROI()

###### Here the roi has already been selected and saved, this is a gray image
