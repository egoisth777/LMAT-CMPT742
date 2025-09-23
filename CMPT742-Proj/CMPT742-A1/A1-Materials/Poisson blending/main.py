import cv2
import numpy as np

from align_target import align_target


def poisson_blend(source_image, target_image, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image
    raise NotImplementedError
if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)