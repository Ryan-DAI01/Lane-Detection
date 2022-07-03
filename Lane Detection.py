import numpy as np
import cv2
import os
global_leftline = [0, 0, 0, 0]
global_rightline = [0, 0, 0, 0]


def region(image):
    height, width = image.shape
    # isolate the gradients that correspond to the lane lines
    triangle = np.array([[(0, 360),
                          (640, 0),
                          (120, 0),
                          (640, 360),
                          (640, 470),
                          (0, 470)]],
                          dtype=np.int32)
    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if abs(y2-y1)/abs(x2-x1) <= 0.2:
            continue
        # fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        # lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    # takes average among all the columns (column0: slope, column1: y_int)
    if len(right) == 0:
        right_avg = np.array([0, 0])
    else:
        right_avg = np.average(right, axis=0)
        global global_rightline
        global_rightline = make_points(image, right_avg)
    if len(left) == 0:
        left_avg = np.array([0, 0])
    else:
        left_avg = np.average(left, axis=0)
        global global_leftline
        global_leftline = make_points(image, left_avg)
    # create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def make_points(image, average):
    slope, y_int = average
    if slope==0 and y_int==0:
        x1=x2=y1=y2=0
        return np.array([x1, y1, x2, y2])
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (2/5))
    # determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            global global_rightline
            if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                x1, y1, x2, y2 = global_rightline
                cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
                continue
            if abs(x1) >= 1000 or abs(x2) >= 1000 or abs(y1) >= 1000 or abs(y2) >= 1000:
                x1, y1, x2, y2 = global_rightline
                cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
                continue
            # draw lines on a black image
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    return lines_image


if __name__ == "__main__":
    file_list = os.listdir('D://LanePicture')
    for i in file_list:
        image = cv2.imread('D://LanePicture//'+i)
        copy = np.copy(image)
        gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur_gray, 50, 150)
        isolated = region(edges)
        # cv2.imshow("edges", edges)
        # cv2.imshow("iso", isolated)
        # cv2.waitKey(0)

        # DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
        lines = cv2.HoughLinesP(isolated, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average(copy, lines)
        black_lines = display_lines(copy, averaged_lines)
        # taking wighted sum of original image and lane lines image
        lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
        # cv2.imshow("lanes", lanes)
        # cv2.waitKey(0)
        cv2.imwrite('D://Result//'+i, lanes)
