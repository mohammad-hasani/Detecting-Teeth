import numpy as np
import cv2

img = cv2.imread('teeth00.jpg')

clicked = []


def m_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(img[y, x])
        clicked.append(hsv[y, x])


def k_means(image):
    img = cv2.imread(image)
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('image')
cv2.setMouseCallback('image', m_event)
while (1):
    cv2.imshow('image', hsv)
    if cv2.waitKey(20) & 0xFF == 27:
        break

clicked = np.array(clicked)

min_r = np.min(clicked[:, 0])
max_r = np.max(clicked[:, 0])

min_g = np.min(clicked[:, 1])
max_g = np.max(clicked[:, 1])

min_b = np.min(clicked[:, 2])
max_b = np.max(clicked[:, 2])

lower = [min_r, min_g, min_b]
upper = [max_r, max_g, max_b]

lower = np.array(lower)
upper = np.array(upper)

mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

confirmed_pixels = set()
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j] == 255:
            for k in range(i - 1, i + 3, 1):
                for kk in range(j - 3, j + 3, 1):
                    confirmed_pixels.add((k, kk))

new_image = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        tmp = (i, j)
        if tmp in confirmed_pixels:
            new_image[i, j] = img[i, j]
        else:
            new_image[i, j] = 0

# cv2.imshow('after', new_image)

new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', new_image)

kernel = [[1,1,1],
          [1,-2,1],
          [1,1,1]]
kernel = np.array(kernel)

edges = cv2.filter2D(new_image, -1, kernel)


im2, contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('edges', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
