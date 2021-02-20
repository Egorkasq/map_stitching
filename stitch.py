import os
import glob
import cv2
import numpy as np


def take_center_of_image(image):
    image = image[int(image.shape[0] * 0.25): int(image.shape[0] * 0.75),
                  int(image.shape[1] * 0.25): int(image.shape[1] * 0.75)]
    return image


def take_image_without_black_line(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 0, cv2.THRESH_TOZERO)
    print(image)
    print(image[1].shape)
    cv2.imshow('123', cv2.resize(image[1], (1000, 500)))
    cv2.waitKey(0)
    return image

def create_panorama(image_path, expansion, resize_image=False, gaussian_blur=False):
    """
    :param image_path: folder with images
    :param expansion: expansion of the searched files
    :param resize_image: used for reduce original images (helpful, when ultimate map is big)
    :param add_gaussian_blur: used for keypoint detect (use 3, 6, 9, 11 ....)
    :return:
    """

    searchstr = os.path.join(image_path, expansion)
    list_of_images = glob.glob(searchstr)
    list_of_images.sort()
    print('find {} images for stitching:{}'.format(len(list_of_images), list_of_images))
    base_image = cv2.imread(list_of_images[0], 0)
    list_of_images.pop(0)

    for file in list_of_images:

        next_image = cv2.imread(file, 0)
        # next_image = take_center_of_image(next_image)

        if gaussian_blur is not False:
            base_image = cv2.GaussianBlur(base_image, (gaussian_blur, gaussian_blur), 0)
            next_image = cv2.GaussianBlur(next_image, (gaussian_blur, gaussian_blur), 0)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(base_image, None)
        kp2, des2 = orb.detectAndCompute(next_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.2)]
        # cv2.imshow('12', cv2.resize(cv2.drawMatches(base_image, kp1, next_image, kp2, matches, base_image), (1000, 500)))
        # cv2.waitKey(0)
        assert len(matches) > 10

        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h1, w1 = base_image.shape[:2]
        h2, w2 = next_image.shape[:2]
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, M)
        pts = np.concatenate((pts1, pts2), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        result = cv2.warpPerspective(next_image, Ht.dot(M), (xmax - xmin, ymax - ymin))
        image_mask = take_image_without_black_line(base_image)
        cv2.imshow('res', cv2.resize(result, (1000, 500)))
        cv2.waitKey()

        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image_mask[1]

        base_image = result

    print('map created {}'.format(result.shape[:2]))
    return result


if __name__ == '__main__':
    img_path = '/home/error/Documents/map_stitch/image'
    res_path = './total'
    img_path = os.path.join(img_path)
    print(img_path)
    img = create_panorama(img_path, '*.JPG')

    cv2.imwrite('{}'.format('123.JPG'), img)
