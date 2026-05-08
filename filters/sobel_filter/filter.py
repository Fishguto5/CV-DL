import cv2 as cv
import numpy as np

img = cv.imread('/home/gustavo-fernandes/CVS/filters/assets/letras_alfabeto.jpg', cv.IMREAD_GRAYSCALE)

sobelx = cv.Sobel(img,cv.CV_64F,1,0,None,3)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, None, 3)

gradient_magnitude = cv.magnitude(sobelx,sobely) #combine the both matrixes to into one (like a G² = Gx² + Gy²)
gradient_magnitude = cv.convertScaleAbs(gradient_magnitude) #here the image is in float64 format, we need to convert it to unit8

imagex = cv.convertScaleAbs(sobelx)
imagey = cv.convertScaleAbs(sobely)

cv.imshow("Cross Matrix - XY", gradient_magnitude)
cv.imshow("Unique Matrix - X", imagex)
cv.imshow("Unique Matrix - Y", imagey)
cv.waitKey(0)
cv.destroyAllWindows()
