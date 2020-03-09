import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial


k = 1.414
sigma = 1.0


def LoG(sigma):
    n = np.ceil(sigma*6)
    y, x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter


def LoG_convolve(img):
    log_images = []
    for i in range(0, 9):
        y = np.power(k, i)
        sigma_1 = sigma*y
        filter_log = LoG(sigma_1)
        image = cv2.filter2D(img, -1, filter_log)
        image = np.pad(image, ((1, 1), (1, 1)), 'constant')
        image = np.square(image)
        log_images.append(image)

    log_image_np = np.array([i for i in log_images])
    return log_image_np


def blob_overlap(blob1, blob2):
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)
    # print(n_dim)

    # radius of two blobs
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))

    # no overlap between two blobs
    if d > r1 + r2:
        return 0
    # one blob is inside the other, the smaller blob must die
    elif d <= abs(r1 - r2):
        return 1
    else:
        # computing the area of overlap between blobs
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d)))
        return area / (math.pi * (min(r1, r2) ** 2))


def redundancy(blobs_array, overlap):
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blob_overlap(blob1, blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])


def detect_blob(log_image_np, img):
    co_ordinates = []
    (h, w) = img.shape
    for i in range(1,h):
        for j in range(1,w):
            slice_img = log_image_np[:, i-1:i+2, j-1:j+2]
            result = np.amax(slice_img)
            # result_1 = np.amin(slice_img)
            if result >= 0.03:
                z, x, y = np.unravel_index(slice_img.argmax(), slice_img.shape)
                co_ordinates.append((i+x-1, j+y-1, k**z*sigma))
    return co_ordinates


def sort_points(pointsOrg, axis="x"):
    """ Bubble Sort
    points: list of points to sort
    ax: "x" or "y" which axis should be used to sort list default "x"
    returns sorted by axis"""

    points = pointsOrg.copy()
    if not isinstance(points, list):
        points = points.tolist()

    sortAxis = -1
    if axis == "x":
        sortAxis = 0
    elif axis == "y":
        sortAxis = 1

    issorted = False
    while not issorted:
        issorted = True
        for i in range(len(points)-1):
            if points[i][sortAxis] > points[i+1][sortAxis]:
                points[i], points[i+1] = points[i+1], points[i]
                issorted = False
    return points


def exchangeAxis(points):
    return [[elem[1], elem[0]] for elem in points]


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


org = cv2.imread("skeleton4.png", 1)
img2 = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
img2 = img2/255.0

# cv2.imshow("Okno", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

log_image_np = LoG_convolve(img2)


co_ordinates = list(set(detect_blob(log_image_np, img2)))
co_ordinates = np.array(co_ordinates)
co_ordinates = redundancy(co_ordinates, 0.01)

# Convert list of floats to integers
blob_centers = co_ordinates[:, 0:2].astype(int)

blob_centers = exchangeAxis(blob_centers)

# Draw all blobs
for blob in blob_centers:
    x, y = blob
    cv2.circle(org, (x, y), 6, (0, 0, 255), cv2.FILLED)

# Sort blobs by height
sortedCentersH = sort_points(blob_centers, axis="y")
# Sort blobs by width
sortedCentersW = sort_points(blob_centers, axis="x")

# Max width of body
maxleft = tuple(sortedCentersW[0])
maxright = tuple(sortedCentersW[-1])
maxwidth = dist(maxleft, maxright)

# Display corner points and line
cv2.line(org, maxleft, maxright, (0, 255, 255))
cv2.circle(org, maxleft, 6, (0, 255, 255), cv2.FILLED)
cv2.circle(org, maxright, 6, (0, 255, 255), cv2.FILLED)
cv2.putText(org, "Odl: {:.2f} px".format(maxwidth), (maxright[0]+15, maxright[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

# Highest point is head
head = tuple(sortedCentersH[0])

# Four lowest points are feets
feets = sortedCentersH[len(sortedCentersH)-4:]

# Calculate average of feets position
sumX = 0
sumY = 0

for feet in feets:
    sumX += feet[0]
    sumY += feet[1]

footsAVG = (sumX//4, sumY//4)
approxHeight = dist(head, footsAVG)

# Display corner points and line
cv2.circle(org, head, 6, (255, 255, 0), cv2.FILLED)
cv2.circle(org, footsAVG, 6, (255, 255, 0), cv2.FILLED)
cv2.line(org, head, footsAVG, (255, 255, 0))
cv2.putText(org, "Odl: {:.2f} px".format(approxHeight), (head[0] + 15, head[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

# Shoulders width
shoulders = [tuple(sortedCentersH[2]), tuple(sortedCentersH[3])]
shoulderWidth = dist(shoulders[0], shoulders[1])

if shoulders[0][0] < shoulders[1][0]:
    shoulderRight = 1
else:
    shoulderRight = 0

cv2.line(org, shoulders[0], shoulders[1], (255, 255, 0))
cv2.putText(org, "Odl: {:.2f} px".format(shoulderWidth), (shoulders[shoulderRight][0] + 15, shoulders[shoulderRight][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))


policzone2 = dist(sortedCentersH[-9], sortedCentersH[-10])
cv2.line(org, tuple(sortedCentersH[-9]), tuple(sortedCentersH[-11]), (255, 255, 0), 1)
cv2.putText(org, "Odl: {:.2f} px".format(policzone2), (sortedCentersH[-9][0]+15, sortedCentersH[-9][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

height, width, _ = org.shape
print(org.shape)
result = org.copy()
whitebox = 250*np.ones((50, width, 3))
result = np.vstack([org, whitebox])
person = "Aleksandra Zmijewska"

cv2.putText(result, "Osoba: {}".format(person), (20, height+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imshow("Okno", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
