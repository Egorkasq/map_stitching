import cv2
import os
import glob


def rotate_image_from_folder(image_path):
    images_list = os.path.join(image_path)
    list_of_images = glob.glob(images_list)
    print(list_of_images)
    for i in list_of_images:
        print(i)
        image = cv2.imread(i, 1)
        if image.shape[0] > image.shape[1]:
            print(i)
            temp = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite('./rotate_image/' + str(i), temp)


if __name__ == "__main__":
    image_path = './map_stitch/image'
    rotate_image_from_folder(image_path)
