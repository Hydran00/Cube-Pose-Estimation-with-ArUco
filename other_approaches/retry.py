import cv2
import numpy as np
import math


def rot_3D(x,y,z,theta):
    mat=[[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,0]]
    vect=[x,y,z]
    new_values=np.matmul(mat,vect)
    new_values=np.int_(new_values)
    return new_values

def line_coeff(x1,x2,y1,y2):
    
    #this finction has the obective of applying the 45 degrees for the two points in order to find the third point

    v1=(x1-x2)
    v2=(y1-y2)
    print(v1,v2)
    X1=(v1)*math.cos(math.pi/4)-math.sin(math.pi/4)*(v2)
    X1=round(X1)
    Y1=(v1)*math.sin(math.pi/4)-math.cos(math.pi/4)*(v2)
    Y1=round(Y1)
    
    v1=(x2-x1)
    v2=(y2-y1)

    print(v1,v2)
    X2=(v1)*math.cos(-math.pi/4)-math.sin(-math.pi/4)*(v2)
    X2=round(X2)
    Y2=(v1)*math.sin(-math.pi/4)-math.cos(-math.pi/4)*(v2)
    Y2=round(Y2)
    print(X1,Y2)



    return X1,Y1,X2,Y2
    

def isolateRed(image):
    #convert image into hsv
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #caluculate first mask
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    #calculate second mask
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0 + mask1
    #apply mask
    output_img = image.copy()
    output_img[np.where(mask==0)] = 0
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0

    #convert to bgr
    out_rgb = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
    #convert to grayscale
    image_gray= cv2.cvtColor(out_rgb,cv2.COLOR_BGR2GRAY)
    #blur image
    blurred=cv2.GaussianBlur(image_gray,(5,5),0)

    #create parameters for canny filter
    sigma = np.std(blurred)
    mean = np.mean(blurred)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))

    canny = cv2.Canny(blurred, lower, 180)
    canny_cp=canny

    kernel = np.ones((5,5),np.uint8)
    #apply erode function
    canny_cp = cv2.erode(canny_cp,kernel,iterations = 2)
   
    #find contours and maximize area
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    #create approximation of polygon
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    
    cv2.drawContours(canny_cp,[approx],-1, (255,255,0), thickness=2)
    
    #apply HoughLines to find vertical lines
    linesP = cv2.HoughLinesP(canny_cp, rho=1, theta=np.pi/180, threshold=80, minLineLength=70, maxLineGap=1)
    
    #if the distance between two lines is <10 then remove the line
    if linesP is not None:
        flag=False
        for i in range(0,len(linesP)):
            if(i>=len(linesP)):
                    break
            for j in range(0,len(linesP)):
                if(j>=len(linesP)):
                    break
                if((0<abs(linesP[i][0][0]-linesP[j][0][0])<10)):
                    linesP=np.delete(linesP, obj=j,axis=0)
                    flag=True

    #draw lines on image           
    for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)

    #only keep the ones with the highes start point
    y_1=min(linesP[0][0][1],linesP[0][0][3])
    if(y_1==linesP[0][0][1]):
        x_1=linesP[0][0][0]
    else:
        x_1=linesP[0][0][2]

    x_2=min(linesP[1][0][0],linesP[1][0][2])
    y_2=min(linesP[1][0][1],linesP[1][0][3])
    if(y_2==linesP[1][0][1]):
        x_2=linesP[1][0][0]
    else:
        x_2=linesP[1][0][2]

    #draw the points corresponding to the lines
    cv2.circle(image,center=(x_1,y_1), radius=3 , color=(255,0,0) , thickness=2)
    cv2.circle(image,center=(x_2,y_2), radius=3 , color=(255,0,0) , thickness=2)
    #apply line_coeff
    X2,Y2,X1,Y1=line_coeff(x_1, x_2,y_1,y_2)
    
    cv2.circle(image,center=(X2+x_2,Y2+y_2), radius=5 , color=(255,150,0) , thickness=2)
    cv2.circle(image,center=(X1+x_1,Y1+y_1), radius=5 , color=(255,0,150) , thickness=2)

if __name__== "__main__":
    image=cv2.imread("wrap2.jpg")
    isolateRed(image)
    cv2.imshow("original", image)

    cv2.waitKey(0)
