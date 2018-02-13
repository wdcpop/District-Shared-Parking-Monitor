# District-Shared-Parking-Monitor
# Marker detection code

import math
import numpy as np 
import cv2
import os
import json
import time
import array

# Return a new RGB shade of a color
def rgb_shade(color, brightness):
    return [int(min(c * brightness, 255.0)) for c in color]

# Normalize Photoshop HSV (360/100/100) to OpenCV HSV (180/255/255)
def norm_hsv(color):
    hsv = list(color)
    hsv[0] = int(color[0] / 2)
    hsv[1] = int(color[1] * 255 / 100)
    hsv[2] = int(color[2] * 255 / 100)
    return hsv

def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

# Compute the distance between two vectors (p1/p2 each have 2 coords)
def distance(p1, p2):
    return math.sqrt(   math.pow(p1[0]-p2[0], 2) + 
                        math.pow(p1[1]-p2[1], 2) )

# Calculate the CCW angle in degrees between p1 and p2, where 
# 0deg points from p1 in the -y direction
# (used for determining marker orientation, bounding box, etc)
def calc_angle(p1, p2, d):
    angle = 0.0
    # x1 < x2, y1 < y2
    if p1[0] <= p2[0] and p1[1] <= p2[1]:
        return 360.0 - (180.0 - math.degrees(math.asin((p2[0]-p1[0]) / d)))
    # x1 > x2, y1 < y2
    elif p1[0] > p2[0] and p1[1] <= p2[1]:
        return 360 - (180.0 + math.degrees(math.asin((p1[0]-p2[0]) / d)))
    # x1 > x2, y1 > y2
    elif p1[0] > p2[0] and p1[1] > p2[1]:
        return 360 - (270.0 + math.degrees( math.asin((p1[1]-p2[1]) / d)))
    # x1 < x2, y1 > y2
    elif p1[0] <= p2[0] and p1[1] > p2[1]:
        return 360 - (math.degrees( math.asin((p2[0]-p1[0]) / d)))
    else:
        return -1.0

# Sort the skews in m
# Assume m.skews[0] is "on the left", and m.skews[0] is "on the right"
# Swap the skews if they're not correct
def sortSkews(m):
    xL = m.skews[0].location[0]
    xR = m.skews[1].location[0]
    yL = m.skews[0].location[1]
    yR = m.skews[1].location[1]

    # Correct if: xL < xR, yL > yR
    if m.component_angle < 90.0 and m.component_angle >= 0.0:
        if not( xL < xR and yL > yR):
            # Swap
            tmp = m.skews[0]
            m.skews[0] = m.skews[1]
            m.skews[1] = tmp
    # Correct if: xL < xR, yL < yR
    elif m.component_angle < 360.0 and m.component_angle >= 270.0:
        if not( xL < xR and yL < yR):
            # Swap
            tmp = m.skews[0]
            m.skews[0] = m.skews[1]
            m.skews[1] = tmp
    # Correct if: xL > xR, yL > yR
    elif m.component_angle < 180.0 and m.component_angle >= 90.0:
        if not( xL > xR and yL > yR):
            # Swap
            tmp = m.skews[0]
            m.skews[0] = m.skews[1]
            m.skews[1] = tmp
    # Correct if: xL > xR, yL < yR
    elif m.component_angle < 270.0 and m.component_angle >= 180.0:
        if not( xL > xR and yL < yR):
            # Swap
            tmp = m.skews[0]
            m.skews[0] = m.skews[1]
            m.skews[1] = tmp
    return m

# Remove all but the two closest skew components from association with
# the provided marker
def pruneSkews(m):
    if len(m.skews) > 2:
        smallest = [m.skews[0], m.skews[1]]
        for i in range(2, len(m.skews)):
            maxIndex = smallest.index(
                max([distance(smallest[0].location,
                        m.location), distance(
                        smallest[1].location,
                        m.location)]))
            if distance(m.skews[i].location, m.location) < \
            distance(smallest[maxIndex].location, m.location):
                smallest[maxIndex] = m.skews[i]
    return m

# Contour data structures
class contour():
    def __init__(self, location, area):
        self.location   = location
        self.area       = area

    def __repr__(self):
        return "\"location\": [" + str(self.location[0]) + ", " + \
            str(self.location[1]) + "], \"area\": " + str(self.area)

# Define a marker class, that contains the location of the marker,
# as well as its constituent parts
class marker():
    def __init__(self, location, tri, trap, component_angle, 
        skews_angle = None, skews = None, number = None, area = None):
        self.location   = location
        self.tri        = tri
        self.trap       = trap
        self.component_angle = component_angle
        self.skews_angle = skews_angle
        self.skews      = skews
        self.number     = number
        self.area       = area

    def __repr__(self):
        return "\"" + str(id(self)) + "\": { \"location\": [" + \
        str(self.location[0]) + ", " + \
        str(self.location[1]) + "], " + \
        "\"tri\": {" + str(self.tri) + "}, " + \
        "\"trap\": {" + str(self.trap) + "}, " + \
        "\"comp_angle\": " + str(self.component_angle) + ", " + \
        "\"skews_angle\": " + str(self.skews_angle) + ", " + \
        "\"skews\": {" + str(self.skews)[1:-1] + "}, " + \
        "\"number\": {" + str(self.number) + "}, " + \
        "\"area\": {" + str(self.area) + "}}"

# JSON reading example:
# Keys() returns top-level keys in order
# file = open("json-test.txt", "r")
# test = json.load(file)
# el0 = test[test.keys()[0]]

# Run image processing on a file
# Returns locations and other data on detected markers
def process_image(f, debug = False):
    CURRENT_DIR = os.getcwd()
    IMAGE_NAME = f
    IMAGE_NAME_EXT = f + ".jpg"
    IMAGE_DIR = IMAGE_NAME + str("-output")
    OUTPUT_DIR = CURRENT_DIR + "/received_photos/" + IMAGE_NAME + "/" + IMAGE_DIR

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img_bgr = cv2.imread("./received_photos/" + IMAGE_NAME + "/" + IMAGE_NAME_EXT)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]

    if debug:
        print "Operating on image size (h, w): " + str(img_bgr.shape[:2])
        print "Output directory @: " + OUTPUT_DIR

    # -------------------- FILTERING --------------------

    # Bright/faded blue (PS HSV): [196, 100, 94] / [197, 87, 79]
    # Bright/faded pink (PS HSV): [324, 100, 93] / [325, 89, 79]
    # Bright/faded green (PS HSV): [119, 92, 87] / [120, 100, 62]
    trap_hsv = [(200, 100, 100), (150, 40, 40)]
    tri_hsv = [(350, 100, 100), (315, 40, 40)]
    skew_hsv = [(100, 100, 100), (85, 40, 40)]

    # Normalize the PS HSV values to CV HSV
    trap_hsv = [norm_hsv(c) for c in trap_hsv]
    tri_hsv = [norm_hsv(c) for c in tri_hsv]
    skew_hsv = [norm_hsv(c) for c in skew_hsv]

    trap_lower = np.array(trap_hsv[1])
    trap_upper = np.array(trap_hsv[0])
    tri_lower = np.array(tri_hsv[1])
    tri_upper = np.array(tri_hsv[0])
    skew_lower = np.array(skew_hsv[1])
    skew_upper = np.array(skew_hsv[0])

    # Output boundary swatches
    swatches_zeroes = np.zeros((300, 200, 3), np.uint8)
    swatches = cv2.cvtColor(swatches_zeroes, cv2.COLOR_BGR2HSV)
    swatches[0:100, 0:100]      = trap_hsv[1]
    swatches[0:100, 100:200]    = trap_hsv[0]
    swatches[100:200, 0:100]    = tri_hsv[1]
    swatches[100:200, 100:200]  = tri_hsv[0]
    swatches[200:300, 0:100]    = skew_hsv[1]
    swatches[200:300, 100:200]  = skew_hsv[0]
    swatches = cv2.cvtColor(swatches, cv2.COLOR_HSV2BGR)
    cv2.imwrite(OUTPUT_DIR + '/boundary-swatch.png', swatches)

    # Filter trapezoid color only
    trap_mask   = cv2.inRange(img_hsv, trap_lower, trap_upper)
    trap_output = cv2.bitwise_and(img_hsv, img_hsv, mask = trap_mask)

    # Filter triangle color only
    tri_mask   = cv2.inRange(img_hsv, tri_lower, tri_upper)
    tri_output = cv2.bitwise_and(img_hsv, img_hsv, mask = tri_mask)

    # Filter skew detect color only
    skew_mask  = cv2.inRange(img_hsv, skew_lower, skew_upper)
    skew_output = cv2.bitwise_and(img_hsv, img_hsv, mask = skew_mask)

    # Filter both
    total_mask      = cv2.bitwise_or(trap_mask, tri_mask)
    total_mask      = cv2.bitwise_or(total_mask, skew_mask)
    total_mask_output    = cv2.bitwise_and(img_hsv, img_hsv, mask = total_mask)
    total_mask_output    = hsv2bgr(total_mask_output)
    total_edges     = cv2.Canny(total_mask, 50, 100)

    # Output results (convert to BGR since your computer likely won't
    # display HSV by default - don't need to worry about the masks 
    # because they're binary)
    cv2.imwrite(OUTPUT_DIR + "/trap-mask.png",   trap_mask)
    cv2.imwrite(OUTPUT_DIR + "/trap-output.png", hsv2bgr(trap_output))
    cv2.imwrite(OUTPUT_DIR + "/tri-mask.png",    tri_mask)
    cv2.imwrite(OUTPUT_DIR + "/tri-output.png",  hsv2bgr(tri_output))
    cv2.imwrite(OUTPUT_DIR + "/skew-mask.png", skew_mask)
    cv2.imwrite(OUTPUT_DIR + "/skew-output.png", hsv2bgr(skew_output))
    cv2.imwrite(OUTPUT_DIR + "/total-mask.png",  total_mask)
    cv2.imwrite(OUTPUT_DIR + "/total-edges.png", total_edges)

    # -------------------- TRACING --------------------

    total_output = np.zeros((h, w, 3), np.uint8)

    # The allowable error when matching a contour to a polygon
    POLY_EPSILON = 0.03
    # The minimum pixel area to consider when finding markers
    AREA_LIMIT = 200.0
    # The maximum separation distance when colocating marker tris/traps
    COLOC_LIMIT_T = 50.0
    # The maximum separation distance when colocating 
    COLOC_LIMIT_S = 50.0

    # The maximum separation distance between a trapezoid component and a 
    # triangle component relative to the pixel area of the triangle
    COLOC_LIMIT_T_REL = 0.045

    # The maximum separation distance between a skew component and it's 
    # respective marker relative to the pixel area of the marker's triangle
    COLOC_LIMIT_S_REL = 0.04

    # Lower Canny edge detection limit
    CANNY_LOWER = 25
    # Upper Canny edge detection limit
    CANNY_UPPER = 50

    # Idea: add contours to a data structure/array that we can then
    # iterate over to check that we've not detected a similar-sized
    # contour in a similar area (ie. if |cy2-cy1| < 5px, |cx2-cx1| < 5px)

    tris   = []
    traps  = []
    skews  = []

    # Triangle contours
    tri_edges = cv2.Canny(tri_mask, CANNY_LOWER, CANNY_UPPER)

    # Find the Laplacian derivitaves as an alternative to Canny
    tri_laplacian  = cv2.Laplacian(tri_mask, cv2.CV_64F)
    tri_lap_abs    = np.absolute(tri_laplacian)
    tri_lap_8b     = np.uint8(tri_lap_abs)

    im_tris, contours_tris, h_tris = cv2.findContours(tri_lap_8b,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_tris:
        approx = cv2.approxPolyDP(cnt,
            POLY_EPSILON * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            # True flag indicates signed area of the contour;
            # objects and holes have different sign and we only need one
            # (we'll pick positive, should work either way though)
            area = cv2.contourArea(cnt, True)

            # ignore negligible contours
            if area > AREA_LIMIT: 
                if debug:
                    print "triangle found"
                    print "contour area: " + str(area)

                # Calculate the center of mass from the moments
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if debug:
                    print "center location: " + str(cx) + " " + str(cy)

                cv2.circle(total_output, (cx, cy), 5, (0, 0, 255), -1)
                
                # Add contour to the collection of contours,
                tri = contour((cx, cy), area)
                tris.append(tri)
    if debug:
        print "tris found: " + str(len(tris))
        print " -------- "

    # Trapezoid contours
    trap_edges = cv2.Canny(trap_mask, CANNY_LOWER, CANNY_UPPER)

    # Find the Laplacian derivatives as an alternative to Canny
    trap_laplacian  = cv2.Laplacian(trap_mask, cv2.CV_64F)
    trap_lap_abs    = np.absolute(trap_laplacian)
    trap_lap_8b     = np.uint8(trap_lap_abs)

    im_traps, contours_traps, h_traps = cv2.findContours(trap_lap_8b,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_traps:
        approx = cv2.approxPolyDP(cnt, 
            POLY_EPSILON * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            # True flag indicates signed area of the contour;
            # objects and holes have different sign and we only need one
            # (we'll pick positive, should work either way though)
            area = cv2.contourArea(cnt, True)

            # ignore negligible contours
            if area > AREA_LIMIT:
                if debug:
                    print "trapezoid found"
                    print "contour area: " + str(area)

                # Calculate center of mass from moments
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if debug:
                    print "center location: " + str(cx) + " " + str(cy)    
                cv2.circle(total_output, (cx, cy), 5, (0, 0, 255), -1)
                
                # Add contour to the collection of contours,
                trap = contour((cx, cy), area)    
                traps.append(trap)
    if debug:
        print "traps found: " + str(len(traps))
        print " -------- "

    # Skew contours
    skew_edges = cv2.Canny(skew_mask, CANNY_LOWER, CANNY_UPPER)

    # Find the Laplacian derivatives as an alternative to Cany
    skew_laplacian = cv2.Laplacian(skew_mask, cv2.CV_64F)
    skew_lap_abs = np.absolute(skew_laplacian)
    skew_lap_8b = np.uint8(skew_lap_abs)

    im_skews, contours_skews, h_skews = cv2.findContours(skew_lap_8b,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_skews:
        approx = cv2.approxPolyDP(cnt, 
            POLY_EPSILON * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            # True flag indicates signed area of the contour;
            # objects and holes have different sign and we only need one
            # (we'll pick positive, should work either way though)
            area = cv2.contourArea(cnt, True)

            # ignore negligible contours
            if area > AREA_LIMIT:
                if debug:
                    print "skew found"
                    print "contour area: " + str(area)

                # Calculate center of mass from moments
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if debug:
                    print "center location: " + str(cx) + " " + str(cy)    
                cv2.circle(total_output, (cx, cy), 5, (0, 0, 255), -1)
                
                # Add contour to the collection of contours,
                skew = contour((cx, cy), area)    
                skews.append(skew)

    if debug:
        print "skews found: " + str(len(skews))

    # Rectangle PS HSV: [129/92/100]
    bounding_box_limits = [norm_hsv((190, 90, 97)), 
                           norm_hsv((194, 95, 105))]

    # -------- COLOCATION --------

    markers = []
    marker_count = 1

    # Colocate tris and traps, determine marker rotation
    for pt in tris:
        for bt in traps:
            if distance(bt.location, pt.location) <= COLOC_LIMIT_T_REL * pt.area:
                # Found a pair, draw a line!
                cv2.line(total_output, bt.location, pt.location,
                         (0, 0, 255), 2)

                # Find the orientation of the marker
                angle = calc_angle(bt.location, pt.location, 
                    distance(bt.location, pt.location))
                cv2.putText(total_output, 
                    "theta1: " + str(round(angle, 2)) + " deg",
                    pt.location,
                    cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (0, 255, 255))

                marker_center = (
                    (pt.location[0] + bt.location[0]) / 2, 
                    (pt.location[1] + bt.location[1]) / 2
                )

                m = marker(marker_center, pt, bt, angle, 
                    None, [], marker_count, pt.area + bt.area)
                markers.append(m)
                cv2.putText(total_output,
                    str(m.number), m.location,
                    cv2.FONT_HERSHEY_TRIPLEX, 5, (0, 255, 255), 2)

                # Draw the marker component number on the image
                cv2.putText(total_output,
                    "T", 
                    (m.tri.location[0], m.tri.location[1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.4, (0, 255, 255))
                cv2.putText(total_output,
                    "B",
                    (m.trap.location[0], m.trap.location[1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.4, (0, 255, 255))

                marker_count = marker_count + 1

    # Colocate skews with markers
    for m in markers:
        # Idea: colocate skews to markers, then for each marker, remove
        # all but the two closest skews
        for s in skews:
            if distance(s.location, m.location) < COLOC_LIMIT_S_REL * m.tri.area:
                m.skews.append(s)     
        m = pruneSkews(m)

    # Sort the skews of each marker, draw skew info, calculate skew angle
    for m in markers:
        if debug:
            print "Marker has " + str(len(m.skews)) + " skews associated"
        # If there are two skews (as there should be), reorganize them so
        # skews[0] is "on the left" relative to the marker orientation
        if len(m.skews) == 2:
            cv2.line(total_output,
                m.skews[0].location, m.skews[1].location,
                (0, 0, 255), 2)
            
            # Sort the skews (make sure "left" skew is on the "left")
            m = sortSkews(m)

            # Draw the skew marker number on the image
            cv2.putText(total_output,
                "L", 
                (m.skews[0].location[0], m.skews[0].location[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4, (0, 255, 255))
            cv2.putText(total_output,
                "R",
                (m.skews[1].location[0], m.skews[1].location[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4, (0, 255, 255))

            # Calculate the angle between left skew marker and right
            # skew marker relative to the -y direction
            m.skews_angle = calc_angle(m.skews[0].location,
                m.skews[1].location, 
                distance(m.skews[0].location, m.skews[1].location))

            # Draw the angle result on the final image
            cv2.putText(total_output,
                "theta2: " + str(round(m.skews_angle, 2)) + " deg",
                m.skews[0].location,
                cv2.FONT_HERSHEY_DUPLEX, 0.4,
                (0, 255, 255))
            
    # NOTE: ALL WARPAFFINE CALLS EXPECT A DESTINATION WITH REVERSED
    # SIZE FROM THE STANDARD. (HEIGHT, WIDTH) NOT (WIDTH, HEIGHT)!!!

    # Draw the parking space masks
    img_cnt = 0
    # Padding to use to ensure affine transformations don't translate/
    # rotate/skew image data off of the image
    padding = 1000
    total_output_padded = 0
    for m in markers:
        marker_center = m.location    

        # note: need to negate theta when calculating psi because 
        # theta is measured CCW...?
        tan_psi = math.tan(-1.0 * math.radians(m.component_angle - 90.0))
        if debug:
            print "tan psi = " + str(tan_psi)

        # Skew and rotate the parking space mask, then overlay
        rect = np.zeros((h, w, 3), np.uint8)
        cv2.line(rect, (0, m.location[1]), (w, m.location[1]), (0, 0, 255), 2)

        skewed_center = (marker_center[0], marker_center[1] + int(marker_center[0]*tan_psi))
        if debug:
            print "skew center = " + str(skewed_center)
            print "mark center = " + str(marker_center)

        skewed_center_padded = (marker_center[0] + (padding/2), 
            marker_center[1] + (padding/2) + int((marker_center[0]+(padding/2))*tan_psi)) 

        skew_y_delta = marker_center[1] - skewed_center[1]
        if debug:
            print "skew y delta = " + str(skew_y_delta)

        # Construct the boundary box
        cv2.rectangle(rect,
            (marker_center[0] - 60, marker_center[1] + 120),
            (marker_center[0] + 60, marker_center[1] - 120),
            (255, 255, 255),
            2)
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-base.png", rect)

        # Put the constructed rect on the padded image before affines
        rect_padded = np.zeros((h + padding, w + padding, 3), np.uint8)
        rect_padded[(padding/2):(padding/2)+h, (padding/2):(padding/2)+w] = rect
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-padded.png", rect_padded)

        M_rot_center = (skewed_center[0], skewed_center[1] + skew_y_delta)
        M_rot_center_padded = (skewed_center_padded[0], skewed_center_padded[1] + skew_y_delta)

        # Generate the affine matrices
        M_skew = np.array([[1.0, 0.0, 0.0], [tan_psi, 1.0, 0.0]])
        M_rot = cv2.getRotationMatrix2D(
            (skewed_center[0], skewed_center[1] + skew_y_delta),
            -1.0 * round(m.component_angle, 2) * -1.0, 1)
        M_trans = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(2 * skew_y_delta)]])

        # Generate the affine rotation matrix to account for a padded image
        M_rot_padded = cv2.getRotationMatrix2D(
            (skewed_center_padded[0], skewed_center_padded[1] + skew_y_delta),
            -1.0 * round(m.component_angle, 2) * -1.0, 1)

        # Translate the boundary box to account for skew
        rect = cv2.warpAffine(rect, M_trans, (rect.shape[1], rect.shape[0]))
        rect_padded = cv2.warpAffine(rect_padded, M_trans, 
            (rect_padded.shape[1], rect_padded.shape[0]))

        # Mark a circle on the boundary box center
        # cv2.circle(rect, 
        #     (skewed_center[0], skewed_center[1] + skew_y_delta),
        #     3, (0, 0, 255), -1)
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-t.png", rect)
        # cv2.circle(rect_padded,
        #     (skewed_center[0] + (padding/2), skewed_center[1] + (padding/2)),
        #     3, (0, 0, 255), -1)
        #cv2.rectangle(rect_padded, (500, 500), (500 + w, 500 + h), (255, 255, 255))
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-p-t.png", rect_padded)

        # Skew the boundary box
        rect = cv2.warpAffine(rect, M_skew, (rect.shape[1], rect.shape[0]))
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-ts.png", rect)
        rect_padded = cv2.warpAffine(rect_padded, M_skew, 
            (rect_padded.shape[1], rect_padded.shape[0]))
        cv2.rectangle(rect_padded, (500, 500), (500 + w, 500 + h), (255, 255, 255))
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-p-ts.png", rect_padded)
        
        # Draw debug markers
        cv2.circle(rect, M_rot_center, 3, (0, 0, 255), -1)
        cv2.circle(rect_padded, M_rot_center_padded, 3, (0, 0, 255), -1)

        # Rotate the boundary box to correct orientation
        rect = cv2.warpAffine(rect, M_rot, (rect.shape[1], rect.shape[0]))
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-tsr.png", rect)
        rect_padded = cv2.warpAffine(rect_padded, M_rot_padded, 
            (rect_padded.shape[1], rect_padded.shape[0]))
        cv2.imwrite(OUTPUT_DIR + "/rect-" + str(img_cnt) + "-p-tsr.png", 
            rect_padded)
        
        img_cnt = img_cnt + 1
        total_output = cv2.bitwise_or(rect, total_output)

        # Crop padded affine results and OR onto total output
        rect_padded = rect_padded[(padding/2):(h+padding)-(padding/2), (padding/2):(w+padding)-(padding/2)]
        total_output_padded = cv2.bitwise_or(rect_padded, total_output_padded)


    if debug:
        print " -------- "
        print " markers (" + str(len(markers)) + "): " + str(markers)
        print " -------- "
    # -------- OUTPUT --------

    # markers_json = json.loads("{" + str(markers)[1:-1] + "}")

    # file = open("json_test.txt", "w+")
    # json.dump(markers_json, file)
    # file.close()

    # Output image results
    cv2.imwrite(OUTPUT_DIR + "/trap-edges.png",     trap_edges)
    cv2.imwrite(OUTPUT_DIR + "/tri-edges.png",      tri_edges)
    cv2.imwrite(OUTPUT_DIR + "/skew-edges.png",     skew_edges)

    cv2.imwrite(OUTPUT_DIR + "/trap-laplacian.png", trap_laplacian)
    cv2.imwrite(OUTPUT_DIR + "/tri-laplacian.png",  tri_laplacian)
    cv2.imwrite(OUTPUT_DIR + "/skew-laplacian.png", skew_laplacian)

    cv2.imwrite(OUTPUT_DIR + "/trap-laplacian-abs.png", trap_lap_8b)
    cv2.imwrite(OUTPUT_DIR + "/tri-laplacian-abs.png",  tri_lap_8b)
    cv2.imwrite(OUTPUT_DIR + "/skew-laplacian-abs.png", skew_lap_8b)

    cv2.imwrite(OUTPUT_DIR + "/total-output.png",   total_output)
    cv2.imwrite(OUTPUT_DIR + "/total-output-padded.png", total_output_padded)

    total_output_gray = cv2.cvtColor(total_output, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(total_output_gray, 25, 255, 0)
    cv2.imwrite(OUTPUT_DIR + "/total-threshold.png", thresh)

    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    final = cv2.bitwise_or(thresh_bgr, img_bgr)
    cv2.imwrite(OUTPUT_DIR + "/final.png", final)

    return [markers, final]
