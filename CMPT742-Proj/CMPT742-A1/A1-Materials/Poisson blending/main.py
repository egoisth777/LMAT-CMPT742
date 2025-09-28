import cv2
import numpy as np

from align_target_1 import align_target
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

def generate_laplacian(height, width, mask):
    s = height * width
    A = lil_matrix((s, s))

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if mask[y, x]:  # inside the mask
                A[idx, idx] = 4
                if x > 0: A[idx, idx - 1] = -1  # left
                if x < width - 1: A[idx, idx + 1] = -1  # right
                if y > 0: A[idx, idx - width] = -1  # top
                if y < height - 1: A[idx, idx + width] = -1  # bottom
            else:
                A[idx, idx] = 1  # outside the mask

    return A

def getCoefficientMatrix(indexes, mask):
    pass    
def getIndexes(srcImg):
    pass

def poisson_blend(source_image, target_image, target_mask):
    source = source_image.astype(np.float64)
    target = target_image.astype(np.float64)
    mask = target_mask > 0
    
    height, width, channels = target.shape
    A = generate_laplacian(height, width, mask)
    res = source_image.copy()
    
    for layer in range(channels):
        s = source[:,:,layer].flatten()
        t = target[:,:,layer].flatten()
        
        b = np.zeros_like(s)
        
        for j in range(height):
            for i in range(width):
                curr_index = j * width + i
                if mask[j, i]:  # inside the mask
                    # laplacian of the source patch
                    b[curr_index] = (
                        4 * s[curr_index]
                        - (s[curr_index - 1] if i > 0 else 0)
                        - (s[curr_index + 1] if i < width - 1 else 0)
                        - (s[curr_index - width] if j > 0 else 0)
                        - (s[curr_index + width] if j < height - 1 else 0)
                    )
                else:  # outside the mask
                    b[curr_index] = t[curr_index]
                    

        x = spsolve(A.tocsc(), b)
        
        # Show the difference between the solver and the gradient
        print("|Av - b| layer#" + str(layer) + ": ", np.linalg.norm(A.dot(x) - b)) 
        x = x.reshape((height, width))
        x = np.clip(x, 0, 255).astype(np.uint8)

        res[:, :, layer] = x
    return res


def erode_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask

    
if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)
    mask = erode_mask(mask, 5)
    
    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)
    
    cv2.imshow('blend result', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()