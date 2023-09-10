import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.exposure
from rembg import remove

#ROI
#original_image = cv2.imread("A5.jpg")
original_image = cv2.imread("Original/valve 5.jpg")
original_image = cv2.resize(original_image,(600,650))
A=original_image
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
roi = original_image
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Histogram ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
#cv2.imshow("roi_hist",roi_hist)
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)
# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 150, cv2.THRESH_BINARY)
mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_or(original_image, mask)
#cv2.imshow("Mask", mask)
#cv2.imshow("Original image", original_image)
#cv2.imshow("Result", result)
cv2.imshow("Roi", hsv_original)
cv2.imwrite("result.jpg",result)
#Enhance
img = result
#converting the image to gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#applying histogram equalization
equalized_img = cv2.equalizeHist(gray_img)
#applying gamma correction
gamma = 1.2
gamma_corrected = np.array(255*(equalized_img / 255) ** gamma, dtype = 'uint8')
#displaying the enhanced image
cv2.imwrite('Enhanced_Image.jpg', gamma_corrected)
#Sharpen
gaussian_blur=cv2.GaussianBlur(img,(7,7),2)
#sharpen1=cv2.addWeighted(img,1.5,gaussian_blur,-0.5,0)
#cv2.imshow("sharpen",sharpen1)
#FINALL
image = gamma_corrected
output = remove(image)
original_image=output

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 2)
edged = cv2.Canny(blurred, 10, 100)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

dilate = cv2.dilate(edged, kernel, iterations=1)

contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
image_copy = image.copy()
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
#largest_item= sorted_contours[0]
cnt = sorted_contours[0]

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.drawContours(img,[cnt],0,(255,255,0),2)
#cv2.imshow('Con',img)
img = cv2.rectangle(A,(x,y),(x+w,y+h),(0,255,255),2)

# compute rotated rectangle (minimum area)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

# draw minimum area rectangle (rotated rectangle)
#img = cv2.drawContours(A,[box],0,(0,255,255),2)
#print(math.dist(box[0],box[1]))
#print(math.dist(box[1],box[2]))

wcm0=(math.dist(box[0],box[1])*0.0264583333)
lcm0=(math.dist(box[1],box[2])*0.0264583333)
lcm=round(lcm0,3)
wcm=round(wcm0,3)

print(x)
print(y)

if(x>=0 or x<=260):
    x=60
    y=30

cv2.putText(img, "Width : {} cm".format(round(wcm, 2)), (int(x - 50), int(y + 350)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 1)
cv2.putText(img, "Lenght : {} cm".format(round(lcm, 2)), (int(x - 50), int(y +380)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 1)

#cv2.imwrite("result-lampbase.jpg",img)

cv2.imwrite('OUTPUT/valve_5/Enhanced.jpg',gamma_corrected)
cv2.imwrite("OUTPUT/valve_5/out.jpg",output)
cv2.imwrite("OUTPUT/valve_5/RESULT.jpg",img)
cv2.imwrite("OUTPUT/valve_5/Roi.jpg", hsv_original)

#cv2.imshow('Enhanced',gamma_corrected)
#cv2.imshow("out",output)
#cv2.imshow("result",img)


cv2.waitKey(0)
cv2.destroyAllWindows()