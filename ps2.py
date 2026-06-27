"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """


    
    original_image = img_in.copy()

    img_in = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    	
    hsvImage = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    maskgreen = cv2.inRange(hsvImage, (36, 0, 255), (70, 255,255))


    maskyellow = cv2.inRange(hsvImage, (15,0,255), (36, 255, 255))

    maskred = cv2.inRange(hsvImage, (0,50,255), (5,255,255))

    target_red = cv2.bitwise_and(original_image,original_image, mask=maskred)
    target_yellow = cv2.bitwise_and(original_image,original_image, mask=maskyellow)
    target_green = cv2.bitwise_and(original_image,original_image, mask=maskgreen)


    x_cord = 0
    y_cord = 0
    traffic_light_colour = ''

    img_green = cv2.cvtColor(target_green,cv2.COLOR_BGR2GRAY)
    img_green = cv2.medianBlur(img_green, 5)

    green_circles = cv2.HoughCircles(img_green,cv2.HOUGH_GRADIENT,1,70,param1=50, param2=10, minRadius=10, maxRadius=30)
    if green_circles is not None:
        traffic_light_colour = 'green'

    img_yellow = cv2.cvtColor(target_yellow,cv2.COLOR_BGR2GRAY)
    img_yellow = cv2.medianBlur(img_yellow, 5)

    yellow_circles = cv2.HoughCircles(img_yellow,cv2.HOUGH_GRADIENT,1,70,param1=50, param2=10, minRadius=10, maxRadius=30)
    if yellow_circles is not None:
        traffic_light_colour = 'yellow'

    img_red = cv2.cvtColor(target_red,cv2.COLOR_BGR2GRAY)
    img_red = cv2.medianBlur(img_red, 5)

    red_circles = cv2.HoughCircles(img_red,cv2.HOUGH_GRADIENT,1,70,param1=60, param2=10, minRadius=10, maxRadius=30)
    if red_circles is not None:
        dn_x,dn_y = do_not_enter_sign_detection(original_image)
        traffic_light_colour = 'red'
        red_circles = np.uint16(np.around(red_circles))
        
        for i in red_circles[0,:]:
            if(dn_x-10<=i[0]<= dn_x+10 and dn_y-10<= i[1]<= dn_y+10):
                print('this is do not enter sign')
        else:
            cv2.circle(original_image,(i[0],i[1]),i[2],(0,255,0),3)


    mask_middle_yellow = cv2.inRange(hsvImage, (15,0,0), (36, 255, 255))
    target_middle_yellow = cv2.bitwise_and(original_image,original_image, mask=mask_middle_yellow)

    img_in = cv2.cvtColor(target_middle_yellow,cv2.COLOR_BGR2GRAY)
    img_in = cv2.medianBlur(img_in, 5)

    circles = cv2.HoughCircles(img_in,cv2.HOUGH_GRADIENT,1.03,80,param1=65, param2=13, minRadius=8, maxRadius=30)
   

    small_circle = 30
    if circles is not None:
        
        vector_circles = np.uint16(np.around(circles))
        
        if(len(vector_circles[0,:])>1):
            for i in vector_circles[0,:]:
                if(i[2] < small_circle):
                    small_circle = i[2]
        else:
            for i in vector_circles[0,:]:
                small_circle = i[2]
            
                
   

        for i in vector_circles[0,:]:
            
            if(i[2] == small_circle):
                cv2.circle(original_image,(i[0],i[1]),i[2],(0,255,0),3)
                x_cord = i[0]
                y_cord = i[1]
 

    return (x_cord,y_cord), traffic_light_colour


    
    #raise NotImplementedError


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    """         
    if(dne_x >0 and dne_y > 0):
        cmask = np.ones(original_gray.shape,np.uint8) # snp.ones(ys_original.shape, dtype=np.uint8)
        cmask_circle = cv2.circle(cmask, (dne_x, dne_y), 50, 255, -1)
        cmask_circle = 255-cmask_circle



        original_gray = original_gray & cmask_circle 

    """
        


    ys_original = img_in.copy()
    
    original_gray = cv2.cvtColor(ys_original,cv2.COLOR_BGR2GRAY)
    dne_x,dne_y = do_not_enter_sign_detection(img_in)


    ys_hsvImage = cv2.cvtColor(ys_original, cv2.COLOR_BGR2HSV)

    mask_ys_red = cv2.inRange(ys_hsvImage,(0,150,255), (3,255,255))
    target_ys_red = cv2.bitwise_and(ys_original,ys_original, mask=mask_ys_red)
    target_ys_gray = cv2.cvtColor(target_ys_red,cv2.COLOR_BGR2GRAY)


    kernel = np.ones((4,4), np.uint8)
    closing = cv2.morphologyEx(target_ys_gray, cv2.MORPH_CLOSE, kernel)

    
    coord = cv2.findNonZero(target_ys_gray)

    x,y,w,h = cv2.boundingRect(coord)
    target_ys_red = cv2.rectangle(target_ys_red,(x,y),(x+w,y+h),(0,255,0),2)

    
    img = cv2.medianBlur(target_ys_red,11)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(closing, (3, 3))
    edges = cv2.Canny(gray,100,300,apertureSize = 3)



    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, 2, 20, 10)
    
    xmax= 0
    xmin= 0
    
    ymax= 0
    ymin= 0

    line_coords_list = []
    if linesP is not None:

        if(len(linesP)>=4):

            
            for i in range(0, len(linesP)):

                

                l = linesP[i][0]
                #print(linesP[i][0])

                if((l[0] > (dne_x+10) or l[0] < (dne_x-10)) and (l[1] > (dne_y+10) or l[1] < (dne_y-10))): 
                

                    if (l[2] >= l[0] and l[2] >= xmax):
                        xmax = l[2]
                    if (xmin == 0):
                        xmin = l[0]
                    if (l[0] <= xmin):
                        xmin = l[0]

                    if (l[3] >= l[1] and l[3] >= ymax):
                        ymax = l[3]
                    if (ymin == 0):
                        ymin = l[1]
                    if (l[1] <= ymin):
                        ymin = l[1]
                        
                        
                    
                    
                    
                    cv2.line(ys_original, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)
            



    edges = cv2.cvtColor(ys_original, cv2.COLOR_BGR2GRAY)


    return (int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/3))


    #raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
 


    ss_original = img_in.copy()
    ss_hsvImage = cv2.cvtColor(ss_original, cv2.COLOR_BGR2HSV)

    mask_ss_red = cv2.inRange(ss_hsvImage, (0,50,20), (5,255,220))

    target_ss_red = cv2.bitwise_and(ss_original,ss_original, mask=mask_ss_red)

    img = cv2.medianBlur(target_ss_red,11)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200)


    
    lines = cv2.HoughLines(edges,1,np.pi/180,32,np.array([]))

    if lines is not None:

        for line in lines:
            
            rho,theta = line[0]

            a,b = np.cos(theta), np.sin(theta)
            x0,y0 = a*rho, b*rho
            x1,y1= int(x0 + 1000*(-b)),int(y0 + 1000*(a))
            x2,y2 = int(x0 - 1000*(-b)),int(y0 - 1000*(a))

            cv2.line(ss_original,(x1,y1),(x2,y2),(0,255,0),2)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 32, None, 20, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)
        


    edges = cv2.cvtColor(cdstP, cv2.COLOR_BGR2GRAY)

    coord = cv2.findNonZero(edges)

    x,y,w,h = cv2.boundingRect(coord)
    cdstP = cv2.rectangle(cdstP,(x,y),(x+w,y+h),(0,255,0),2)




    
    return (x+int(w/2),y+int(h/2))
    #raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    wr_original = img_in.copy()
    wr_hsvImage = cv2.cvtColor(wr_original, cv2.COLOR_BGR2HSV)
    mask_wr_yellow = cv2.inRange(wr_hsvImage, (20, 100, 255), (40, 255, 255))
    target_wr_yellow = cv2.bitwise_and(wr_original,wr_original, mask=mask_wr_yellow)



    img = cv2.medianBlur(target_wr_yellow,11)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 32, None, 20, 10)

    xmax= 0
    xmin= 0
    
    ymax= 0
    ymin= 0
    if linesP is not None:
        for i in range(0, len(linesP)):
            
            
            l = linesP[i][0]
            
            
            if (l[2] >= l[0] and l[2] >= xmax):
                xmax = l[2]
            if (xmin == 0):
                xmin = l[0]
            if (l[0] <= xmin):
                xmin = l[0]

            if (l[3] >= l[1] and l[3] >= ymax):
                ymax = l[3]
            if (ymin == 0):
                ymin = l[1]
            if (l[1] <= ymin):
                ymin = l[1]
                
                
            
            
            
            cv2.line(wr_original, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)
        



    edges = cv2.cvtColor(wr_original, cv2.COLOR_BGR2GRAY)

    return (int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2))
    #raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    cs_original = img_in.copy()
    cs_hsvImage = cv2.cvtColor(cs_original, cv2.COLOR_BGR2HSV)
    mask_cs_orange = cv2.inRange(cs_hsvImage, (15, 60, 255), (15,255,255))
    target_cs_orange = cv2.bitwise_and(cs_original,cs_original, mask=mask_cs_orange)

    img_cs_orange = cv2.cvtColor(target_cs_orange, cv2.COLOR_BGR2GRAY)

    coord = cv2.findNonZero(img_cs_orange)


    x,y,w,h = cv2.boundingRect(coord)
    target_cs_orange = cv2.rectangle(target_cs_orange,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    return (x+int(w/2),y+int(h/2))
    #raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    dne_original = img_in.copy()

    cv2.waitKey(0)
    dne_hsvImage = cv2.cvtColor(dne_original, cv2.COLOR_BGR2HSV)
    cmask = np.zeros(dne_original.shape, dtype=np.uint8)

    mask_dne_red = cv2.inRange(dne_hsvImage, (0,150,225), (5,255,255))

    target_dne_red = cv2.bitwise_and(dne_original,dne_original, mask=mask_dne_red)

    x = 0
    y = 0

    img_dne_red = cv2.cvtColor(target_dne_red, cv2.COLOR_BGR2GRAY)


    img_dne_red = cv2.medianBlur(img_dne_red, 7)
   
    dne_red_circles = cv2.HoughCircles(img_dne_red,cv2.HOUGH_GRADIENT,1,150,param1=30, param2=15, minRadius=25, maxRadius=50)

    if dne_red_circles is not None:
        dne_red_circles = np.uint16(np.around(dne_red_circles))

        for i in dne_red_circles[0,:]:
            # draw the outer circle
            
            cv2.circle(dne_original,(i[0],i[1]),i[2],(0,255,0),2)
            x = i[0]
            y = i[1]
 
    
    return (x,y)
    #raise NotImplementedError


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    sign_type = []
    center = []
    dne_x = 0
    dne_y = 0
    dne_x,dne_y = do_not_enter_sign_detection(img_in)

    ss_x = 0
    ss_y = 0
    ss_x,ss_y = stop_sign_detection(img_in)

    cs_x = 0
    cs_y = 0
    cs_x,cs_y = construction_sign_detection(img_in)

    ws_x = 0
    ws_y = 0
    ws_x,ws_y = warning_sign_detection(img_in)

    ys_x = 0
    ys_y = 0
    ys_x,ys_y = yield_sign_detection(img_in)

    radii_range = range(10, 30, 1)
    tl_x = 0
    tl_y = 0
    tl_coord, tl_state = traffic_light_detection(img_in,radii_range)
 
    tl_x = tl_coord[0]
    tl_y = tl_coord[1]






    if(dne_x > 0 and dne_y > 0):
        sign_type.append('no_entry')
        center.append((dne_x,dne_y))
    if(ss_x >0 and ss_y >0):
        sign_type.append('stop')
        center.append((ss_x,ss_y))
    if(cs_x >0 and cs_y >0):
        sign_type.append('construction')
        center.append((cs_x,cs_y))
    if(ws_x >0 and ws_y >0):
        sign_type.append('warning')
        center.append((ws_x,ws_y))
    if(ys_x >0 and ys_y >0):
        sign_type.append('yield')
        center.append((ys_x,ys_y))
    if(tl_x >0 and tl_y >0):
        sign_type.append('traffic_light')
        center.append((tl_x,tl_y)) 


    final = {}
    for i in range(0, len(sign_type)):
        final[sign_type[i]] = center[i]


    return final
    #raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    n_sign_type = []
    n_center = []

    dne_img_in = cv2.GaussianBlur(img_in,(5,5),cv2.BORDER_DEFAULT)
    
    n_dne_x = 0
    n_dne_y = 0
    n_dne_x,n_dne_y = do_not_enter_sign_detection(img_in)

    n_ss_x = 0
    n_ss_y = 0
    n_ss_x,n_ss_y = stop_sign_detection(img_in)

    n_cs_x = 0
    n_cs_y = 0
    n_cs_x,n_cs_y = construction_sign_detection(img_in)

    n_ws_x = 0
    n_ws_y = 0
    n_ws_x,n_ws_y = warning_sign_detection(img_in)

    n_ys_x = 0
    n_ys_y = 0
    n_ys_x,n_ys_y = yield_sign_detection(img_in)

    radii_range = range(10, 30, 1)
    n_tl_x = 0
    n_tl_y = 0
    n_tl_coord, n_tl_state = traffic_light_detection(img_in,radii_range)
 
    n_tl_x = n_tl_coord[0]
    n_tl_y = n_tl_coord[1]






    if(n_dne_x > 0 and n_dne_y > 0):
        n_sign_type.append('no_entry')
        n_center.append((n_dne_x,n_dne_y))
    if(n_ss_x >0 and n_ss_y >0):
        n_sign_type.append('stop')
        n_center.append((n_ss_x,n_ss_y))
    if(n_cs_x >0 and n_cs_y >0):
        n_sign_type.append('construction')
        n_center.append((n_cs_x,n_cs_y))
    if(n_ws_x >0 and n_ws_y >0):
        n_sign_type.append('warning')
        n_center.append((n_ws_x,n_ws_y))
    if(n_ys_x >0 and n_ys_y >0):
        n_sign_type.append('yield')
        n_center.append((n_ys_x,n_ys_y))
    if(n_tl_x >0 and n_tl_y >0):
        n_sign_type.append('traffic_light')
        n_center.append((n_tl_x,n_tl_y)) 


    final_n = {}
    for i in range(0, len(n_sign_type)):
        final_n[n_sign_type[i]] = n_center[i]


    return final_n


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
      
    chg_original = img_in.copy()
    
    ss_hsvImage = cv2.cvtColor(chg_original, cv2.COLOR_BGR2HSV)

    mask_ss_red = cv2.inRange(ss_hsvImage, (0,100,70), (5,255,225))

    target_ss_red = cv2.bitwise_and(chg_original,chg_original, mask=mask_ss_red)

    gray = cv2.cvtColor(target_ss_red,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200)
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    slinesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 10, 20)
    
    sxmax= 0
    sxmin= 0
    
    symax= 0
    symin= 0
    if slinesP is not None:
        
        for i in range(0, len(slinesP)):
            
            
            l = slinesP[i][0]
            
            
            if (l[2] >= l[0] and l[2] >= sxmax):
                sxmax = l[2]
            if (sxmin == 0):
                sxmin = l[0]
            if (l[0] <= sxmin):
                sxmin = l[0]

            if (l[3] >= l[1] and l[3] >= symax):
                symax = l[3]
            if (symin == 0):
                symin = l[1]
            if (l[1] <= symin):
                symin = l[1]
                
            cv2.line(chg_original, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)
        




 #   return {'stop': (int(sxmin+(sxmax-sxmin)/2), int(symin+(symax-symin)/2))}
    

    cs_original = img_in.copy()
    cs_hsvImage = cv2.cvtColor(cs_original, cv2.COLOR_BGR2HSV)
    mask_cs_orange = cv2.inRange(cs_hsvImage, (5, 100, 20), (25,255,200))
    target_cs_orange = cv2.bitwise_and(cs_original,cs_original, mask=mask_cs_orange)



    gray = cv2.cvtColor(target_cs_orange,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,200)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, None, 120, 5)
    
    xmax= 0
    xmin= 0
    
    ymax= 0
    ymin= 0
    if linesP is not None:
        
        for i in range(0, len(linesP)):
            
            
            l = linesP[i][0]
            
            
            if (l[2] >= l[0] and l[2] >= xmax):
                xmax = l[2]
            if (xmin == 0):
                xmin = l[0]
            if (l[0] <= xmin):
                xmin = l[0]

            if (l[3] >= l[1] and l[3] >= ymax):
                ymax = l[3]
            if (ymin == 0):
                ymin = l[1]
            if (l[1] <= ymin):
                ymin = l[1]
                
            cv2.line(cs_original, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA)

    #cv2.imshow('cs_original',cs_original)
    #cv2.waitKey(0)

    if(len(slinesP) > len(linesP)):
        xmin = 0
        xmax = 0
    else:
        sxmin = 0
        sxmax = 0

    if(sxmin >0):
        return {'stop': (int(sxmin+(sxmax-sxmin)/2), int(symin+(symax-symin)/2))}
    else:
        return {'construction':(int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2))}
        

    
 #   return {'stop': (int(sxmin+(sxmax-sxmin)/2), int(symin+(symax-symin)/2)),'construction':(int(xmin+(xmax-xmin)/2), int(ymin+(ymax-ymin)/2))}
#    raise NotImplementedError


################################ CHANGE BELOW FOR MORE CUSTOMIZATION #######################
""" The functions below are used for each individual part of the report section.

Feel free to change the return statements but ensure that the return type remains the same 
for the autograder. 

"""

# Part 2 outputs
def ps2_2_a_1(img_in):
    return do_not_enter_sign_detection(img_in)

def ps2_2_a_2(img_in):
    return stop_sign_detection(img_in)

def ps2_2_a_3(img_in):
    return construction_sign_detection(img_in)

def ps2_2_a_4(img_in):
    return warning_sign_detection(img_in)

def ps2_2_a_5(img_in):
    return yield_sign_detection(img_in)


# Part 3 outputs
def ps2_3_a_1(img_in):
    return traffic_sign_detection(img_in)

def ps2_3_a_2(img_in):
    return traffic_sign_detection(img_in)



# Part 4 outputs
def ps2_4_a_1(img_in):
    return traffic_sign_detection_noisy(img_in)

def ps2_4_a_2(img_in):
    return traffic_sign_detection_noisy(img_in)

# Part 5 outputs
def ps2_5_a(img_in):
    return traffic_sign_detection_challenge(img_in)

