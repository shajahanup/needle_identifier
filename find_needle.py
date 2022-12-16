import cv2
import numpy as np

def find_biggest_contour_area(contours):
    area = 0
    area_indx = 0
    for indx, contour in enumerate(contours):
        if indx == 1:
            continue
        temp_area = cv2.contourArea(contour)
        if(temp_area > area):
            area = temp_area
            area_indx = indx
        temp_img = cv2.imread(img_name)
        print("area = ", temp_area)
        print("top area = ", area, "area index =", area_indx)
        # cv2.drawContours(temp_img, contour, -1, (0, 255, 0), 2)
        # cv2.imshow(str(indx), temp_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    return area_indx


def dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    # cv2.imshow('binary', img)
    # cv2.waitKey(0)
    return dilation;


def erode(img):
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.erode(img, kernel, iterations=1)
    # cv2.imshow('erode', img)
    # cv2.waitKey(0)
    return dilation;


# def flood_fill(img):
#     h, w = img.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#     cv2.floodFill(img, mask, (0, 0), 255)
#     cv2.imshow('flood filled', img)
#     cv2.waitKey(0)


if __name__ == "__main__":
    img_name='60.png'
    img = cv2.imread(img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying binary threshold
    ret, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('thrsh', thresh)
    cv2.waitKey(0)

    # dilate(thresh)
    # #applying canny
    canny_result_img = cv2.Canny(thresh, 50, 200)
    # flood_fill(canny_result_img)
    # find contours
    contours, _ = cv2.findContours(dilate(canny_result_img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[find_biggest_contour_area(contours)]

    # approximate the contours
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    # approximate the contour
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    lower_mask = np.array([10])  # example value
    upper_mask = np.array([15])  # example value
    #
    mask0 = cv2.inRange(img_gray, lower_mask, upper_mask)
    #
    # needle_coordinate_1, needle_coordinate_2 = findXAndY(mask0)
    # cv2.line(mask0, needle_coordinate_1, needle_coordinate_2, (255, 255, 255), 1)

    cv2.imshow("img", img)
    # cv2.imshow('Binary image', thresh)
    # cv2.imshow("img_mask", mask0)
    cv2.imshow("canny", canny_result_img)
    # cv2.imshow("img", img)
    # cv2.imshow("canny", canny_result_img)
    cv2.waitKey()