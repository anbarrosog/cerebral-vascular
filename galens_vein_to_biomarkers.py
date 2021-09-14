
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage as sc
import skimage
import cv2
from skimage.morphology import skeletonize
from skimage import data, morphology
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy as np
from scipy import misc,ndimage
import scipy.ndimage as ndi
from mahotas.morph import hitmiss as hit_or_miss
from skimage.measure import label, regionprops, regionprops_table
import matplotlib.patches as mpatches
from PIL import Image
import math
import pandas as pd
from scipy.cluster.vq import vq
from itertools import product
import math
from geopy.distance import geodesic
from skan import skeleton_to_csgraph
from skan import Skeleton, summarize
from skan import draw
#-----------------first we convert the image to gray-------------------------------------
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

image = rgb2gray(cv2.imread('assets/segmented.jpg'))

#-------------------------------convert the gray image to binary-----------------
thresh = threshold_otsu(image)
binary = image > thresh


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()


ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(binary, cmap=plt.cm.gray)
ax[1].set_title('Thresholded')
ax[1].axis('off')

fig.tight_layout()
plt.show()


#----------I fill gaps and make closures to finish joining nearby pixels---------------

imfill= ndimage.binary_fill_holes(binary)

open_img = ndimage.binary_opening(imfill)
close_img = ndimage.binary_closing(open_img)
clg = close_img.astype(np.int)


# --------------------------------------------perform skeletonization--------------------


skeleton = skeletonize(clg)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(binary, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Binary Image', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()

#----------------next step is to remove branches to keep the vein of galeno---------------

def find_branch_points(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    X1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    #T like
    T=[]
    #T0 contains X0
    T0=np.array([[2, 1, 2],
                 [1, 1, 1],
                 [2, 2, 2]])

    T1=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [1, 2, 2]])  # contains X1

    T2=np.array([[2, 1, 2],
                 [1, 1, 2],
                 [2, 1, 2]])

    T3=np.array([[1, 2, 2],
                 [2, 1, 2],
                 [1, 2, 1]])

    T4=np.array([[2, 2, 2],
                 [1, 1, 1],
                 [2, 1, 2]])

    T5=np.array([[2, 2, 1],
                 [2, 1, 2],
                 [1, 2, 1]])

    T6=np.array([[2, 1, 2],
                 [2, 1, 1],
                 [2, 1, 2]])

    T7=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1],
                 [0, 1, 0],
                 [2, 1, 2]])

    Y1=np.array([[0, 1, 0],
                 [1, 1, 2],
                 [0, 2, 1]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y3=np.array([[0, 2, 1],
                 [1, 1, 2],
                 [0, 1, 0]])

    Y4=np.array([[2, 1, 2],
                 [0, 1, 0],
                 [1, 0, 1]])
    Y5=np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + hit_or_miss(skel,x)
    for y in Y:
        bp = bp + hit_or_miss(skel,y)
    for t in T:
        bp = bp + hit_or_miss(skel,t)

    return bp

def find_end_points(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])

    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=hit_or_miss(skel,endpoint1)
    ep2=hit_or_miss(skel,endpoint2)
    ep3=hit_or_miss(skel,endpoint3)
    ep4=hit_or_miss(skel,endpoint4)
    ep5=hit_or_miss(skel,endpoint5)
    ep6=hit_or_miss(skel,endpoint6)
    ep7=hit_or_miss(skel,endpoint7)
    ep8=hit_or_miss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


branch_pts = find_branch_points(skeleton)


#plt.imshow(skeleton + branch_pts, cmap='gray', interpolation='nearest'); plt.show()

end_pts = find_end_points(skeleton)

detected= branch_pts+end_pts+skeleton
plt.imshow(detected, cmap='gray', interpolation='nearest'); plt.show()


#The following function deletes the segments with an area larger than the imposed area, so that we can keep only the segment of the galeno vein.
label_image = sc.measure.label(detected)
segmentList = sc.measure.regionprops(label_image)
#print(len(segmentList))


correct_labels = [segment.label for segment in segmentList if
                  (
                      (segment.area>=70) #or 40 to see the landmark
                  )
                 ]
finalMask = np.zeros_like(detected) # create the final mask image, empty at the moment

for label in correct_labels: # loop through the list of filtered labels
    obj = segmentList[label-1] # assign the coming fiber to 'obj' (need to be label-1 because it start at 0)
    for r,c in obj.coords:
        finalMask[r,c] = True # Assign label value

plt.imshow(finalMask, cmap='gray', interpolation='nearest'); plt.show()

#the result of this is to eliminate all small segments, the next step is by orientation and centroid.

branch_pts1 = find_branch_points(finalMask)
#plt.imshow(branch_pts1, cmap='gray', interpolation='nearest'); plt.show()

pixels = np.asarray(branch_pts1)
coords = np.column_stack(np.where(pixels == 1))

print(coords)

landmark=coords[2]

print(landmark)

props = regionprops(finalMask)

centroid=props[0]['Centroid']

print(centroid)

image = cv2.imread ("assets/crop1.jpg")

height = image.shape[0]
width = image.shape[1]

cv2.line(image, (landmark[1],0), (landmark[1],height), (255,0,0), 1)

plt.imshow(image, cmap='gray', interpolation='nearest'); plt.show()

def imCrop(x):
    height,width,depth = x.shape
    return [x[: , :landmark[1]] , x[:, (width-landmark[1]):]]

firsthem=imCrop(image)[0]
secondhem=imCrop(image)[1]


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(firsthem, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Gray image of right hemisphere', fontsize=10)
ax[1].imshow(secondhem, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Gray image of left hemisphere', fontsize=10)
fig.tight_layout()
plt.show()

grayhem1=rgb2gray(firsthem)
grayhem2=rgb2gray(secondhem)

thresh1 = threshold_otsu(grayhem1)
binary1 = grayhem1 > thresh1

thresh2 = threshold_otsu(grayhem2)
binary2 = grayhem2 > thresh2

imfillhem1= ndimage.binary_fill_holes(binary1)
imfillhem2= ndimage.binary_fill_holes(binary2)


open_img1 = ndimage.binary_opening(imfillhem1)
close_img1 = ndimage.binary_closing(open_img1)
clg1 = close_img1.astype(np.int)

open_img2 = ndimage.binary_opening(imfillhem2)
close_img2 = ndimage.binary_closing(open_img2)
clg2 = close_img2.astype(np.int)


skeletonhem1 = skeletonize(clg1)
skeletonhem2 = skeletonize(clg2)

#plt.imshow(skeletonhem1, cmap='gray', interpolation='nearest'); plt.show()
#plt.imshow(skeletonhem2, cmap='gray', interpolation='nearest'); plt.show(); plt.set_title('Skeleton left hemisphere with small segments to remove', fontsize=7)
fig = plt.figure()
cax = plt.imshow(skeletonhem2,cmap='gray', interpolation='nearest')
plt.title('Skeleton left hemisphere with small segments to remove', fontsize=7)


#so far I have obtained the two hemispheres and the skeletonisation.
#The next step is to obtain the biomarkers, which are:
#1.accounting for branchpoints and enpoints in each hemisphere. Ideally for symmetrical brains this percentage should be around 50%.



#-------------------------------extraction of the first biomarker--->percentage of branchpoints and enpoints------------------------------------------------------------------

#as we can see in the left hemisphere there are very small segments that have to be eliminated because they alter the measurement.

label_image = sc.measure.label(skeletonhem2)
segmentList = sc.measure.regionprops(label_image)
#print(len(segmentList))

correct_labels = [segment.label for segment in segmentList if
                  (
                      (segment.area>=20) #or 40 to see the landmark
                  )
                 ]
newskel2 = np.zeros_like(skeletonhem2) # create the final mask image, empty at the moment

for label in correct_labels: # loop through the list of filtered labels
    obj = segmentList[label-1] # assign the coming fiber to 'obj' (need to be label-1 because it start at 0)
    for r,c in obj.coords:
        newskel2[r,c] = True # Assign label value


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(skeletonhem1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Skeleton structure right hemisphere', fontsize=10)
ax[1].imshow(newskel2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Skeleton structure left hemisphere', fontsize=10)
fig.tight_layout()
plt.show()
#------------the next step is to find branch and enpoints------------
branchpointshem1= find_branch_points(skeletonhem1)
branchpointshem2= find_branch_points(newskel2)
#plt.imshow(branchpointshem1+skeletonhem1, cmap='gray', interpolation='nearest'); plt.show()
#plt.imshow(branchpointshem2+newskel2, cmap='gray', interpolation='nearest'); plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow((branchpointshem1+skeletonhem1), cmap='gray')
ax[0].axis('off')
ax[0].set_title('1st BIOMARKER: Branch_points right hemisphere', fontsize=10)
ax[1].imshow((branchpointshem2+newskel2), cmap='gray')
ax[1].axis('off')
ax[1].set_title('1st BIOMARKER: Branch_points left hemisphere', fontsize=10)
fig.tight_layout()
plt.show()


endpointshem1= find_end_points(skeletonhem1)
endpointshem2=find_end_points(newskel2)
#plt.imshow(endpointshem1+skeletonhem1, cmap='gray', interpolation='nearest'); plt.show()
#plt.imshow(endpointshem2+newskel2, cmap='gray', interpolation='nearest'); plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow((endpointshem1+skeletonhem1), cmap='gray')
ax[0].axis('off')
ax[0].set_title('1st BIOMARKER: End_points right hemisphere', fontsize=10)
ax[1].imshow((endpointshem2+newskel2), cmap='gray')
ax[1].axis('off')
ax[1].set_title('1st BIOMARKER: End_points left hemisphere', fontsize=10)
fig.tight_layout()
plt.show()


#to do the branchpoint and enpoint count we do
number_of_white_pix_hem1_branchpts = np.sum (branchpointshem1 == 1) # extracting only white pixels
print ("Number of branch_points in right hemisphere:",number_of_white_pix_hem1_branchpts)

number_of_white_pix_hem2_branchpts = np.sum (branchpointshem2 == 1)
print ("Number of branch_points in left hemisphere:",number_of_white_pix_hem2_branchpts)

number_of_white_pix_hem1_endpoints = np.sum (endpointshem1 == 1)
print ("Number of end_points in right hemisphere:",number_of_white_pix_hem1_endpoints)

number_of_white_pix_hem2_endpoints = np.sum (endpointshem2 == 1)
print ("Number of end_points in left hemisphere:",number_of_white_pix_hem2_endpoints)

#---------------calculation of percentage of branchpoints and enpoints in each hemisphere-----------------

calcperbranchhem1=((number_of_white_pix_hem1_branchpts*100)/(number_of_white_pix_hem1_branchpts+number_of_white_pix_hem2_branchpts))
print ("Percentage of branch_points right hemisphere as a biomarker:",calcperbranchhem1)
print("Percentage of branch_points left hemisphere as a biomarker:",100-calcperbranchhem1)

calcperendhem1=((number_of_white_pix_hem1_endpoints*100)/(number_of_white_pix_hem1_endpoints+number_of_white_pix_hem2_endpoints))
print ("Percentage of end_points right hemisphere as a biomarker:",calcperendhem1)
print("Percentage of end_points left hemisphere as a biomarker:",100-calcperendhem1)

#WE OBSERVE HOW THE PERCENTAGE IS CLOSE TO THE PERCENTAGE DETERMINED AS IDEAL, WHICH IS 50%.
#THUS LOOKING AT THIS FIRST BIOMARKER THIS IMAGE WOULD SEEM SYMMETRICAL-----

#2.Calculation of the average and total length of the veins----------------------------
#---------------------------------extraction of the second biomarker--->geodetic distance----------------------------------------------------------------
#Euclidean distance at the end but it also stabilises pixels.


endbranch1= branchpointshem1 + endpointshem1 + skeletonhem1
endbranch2=branchpointshem2 + endpointshem2 + newskel2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(endbranch1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('2nd BIOMARKER: Long segments branchpts endpts right hemisphere', fontsize=7)
ax[1].imshow(endbranch2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('2nd BIOMARKER: Long segments branchpts endpts left hemisphere', fontsize=7)
fig.tight_layout()
plt.show()

pixel_graph, coordinates, degrees = skeleton_to_csgraph(endbranch1)
branch_data1 = summarize(Skeleton(endbranch1))
#print(branch_data.head())
#print(branch_data1['euclidean-distance']) #como se observan son los 14 segmentos de branchpts to endpts en el hemisferio derecho
#calculamos la longitud media de éste primer hemisferio, es decir sumamos los segmentos y los dividimos entre el número total que haya
long_med_hem_dret= ((branch_data1['euclidean-distance'].sum())/len(branch_data1['euclidean-distance']))
#print(long_med_hem_dret)


pixel_graph, coordinates, degrees = skeleton_to_csgraph(endbranch2)
branch_data2 = summarize(Skeleton(endbranch2))
#print(branch_data.head())
#print(branch_data2['euclidean-distance']) #as you can see are the 16 segments from branchpts to endpts in the left hemisphere.
##we calculate the average length of this second hemisphere, i.e. we add up the segments and divide them by the total number that there are
long_med_hem_esq= ((branch_data2['euclidean-distance'].sum())/len(branch_data2['euclidean-distance']))
#print(long_med_hem_esq)

dif_long_med=(long_med_hem_esq/long_med_hem_dret) #if the ratio that we have calculated ideally gives 1, this would mean that the two hemispheres have the same average length.

print("Diferencia longitud mitjana ambos hemisferis:",dif_long_med)

#----to obtain the total length instead of taking the median we simply add up the lengths of the segments without dividing them by the number of segments.
long_tot_hem_dret= ((branch_data1['euclidean-distance'].sum()))
long_tot_hem_esq= ((branch_data2['euclidean-distance'].sum()))

print("Diferencia longitud total ambos hemisferis:",long_tot_hem_dret/long_tot_hem_esq)

#ratio proper to 1 means that both hemispheres have the same total length of veins which means that the total blood supply of both hemispheres is very similar.
#and this is an indicator of symmetry
#https://jni.github.io/skan/getting_started.html#measuring-the-length-of-skeleton-branches



####----------------------- the third and last biomarker is the measurement of axial asymmetry -----------------------------
#to make it mirror image of the left hemisphere and then apply dilation to the skeletons of the hemispheres to obtain more greyness.
#as there are more pixels, there is a higher probability of coincidence.

esp_hem2 = cv2.flip(grayhem2, 1) #with this we obtain the specular image of the left hemisphere.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(grayhem1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Image right hemisphere', fontsize=10)
ax[1].imshow(esp_hem2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Image left hemisphere flipped', fontsize=10)
fig.tight_layout()
plt.show()

thresh21 = threshold_otsu(esp_hem2)
binary21 = esp_hem2 > thresh21
imfillhem21= ndimage.binary_fill_holes(binary21)
open_img21 = ndimage.binary_opening(imfillhem21)
close_img21 = ndimage.binary_closing(open_img21)
clg21 = close_img21.astype(np.int)
skeleton_esp_hem2 = skeletonize(clg21)

#remove small segments

label_image = sc.measure.label(skeleton_esp_hem2)
segmentList = sc.measure.regionprops(label_image)
#print(len(segmentList))

correct_labels = [segment.label for segment in segmentList if
                  (
                      (segment.area>=30) #o 40 para ver el landmark
                  )
                 ]
new_esp_skel2 = np.zeros_like(skeleton_esp_hem2) # create the final mask image, empty at the moment

for label in correct_labels: # loop through the list of filtered labels
    obj = segmentList[label-1] # assign the coming fiber to 'obj' (need to be label-1 because it start at 0)
    for r,c in obj.coords:
        new_esp_skel2[r,c] = True # Assign label value


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(skeletonhem1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Skeleton right hemisphere', fontsize=10)
ax[1].imshow(new_esp_skel2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Skeleton left hemisphere flipped', fontsize=10)
fig.tight_layout()
plt.show()

#expansion of the two skeletons

imfill1= ndimage.binary_fill_holes(skeletonhem1)
dil_img1 = ndimage.binary_dilation(imfill1)
dil1 = dil_img1.astype(np.int)

imfill2= ndimage.binary_fill_holes(new_esp_skel2)
dil_img2 = ndimage.binary_dilation(imfill2)
dil2 = dil_img2.astype(np.int)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(dil1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Skeleton right hemisphere after dilatation', fontsize=10)
ax[1].imshow(dil2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Skeleton left hemisphere flipped after dilatation', fontsize=10)
fig.tight_layout()
plt.show()

def xor(a, b):
    "Same as a ^ b."
    return a ^ b

Isim= xor(dil1, dil2)

ratiNC=((np.sum (Isim == 1))/(np.sum (dil1 == 1)+np.sum (dil2 == 1)))

#The ratio of non-coincidence is obtained by dividing the pixels of equal value that the xor function will return 1 (therefore they coincide) by the total pixels of value 1 of the sum of the two hemispheres.

print("Diferencia asimetria axial entre la imatge de l’hemisferi dret amb la imatge especular de l’hemisferi esquerre:", ratiNC)
