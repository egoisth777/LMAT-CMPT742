import numpy as np
import cv2
import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.signal as signal
from scipy.sparse import lil_matrix




def getIndexes(srcImg):
    '''
    Constructs an Index Matrix as the same shape with the original image size H x W,
    Entries are labeled with 1 ... N ... N + 1 ... 2N ... N x (N - 1) ... N x N

    Args:
        srcImg: The image that we are about to reconstruct
    
    Return:
        Indexes: Index Matrix as the same shape with the original image size H x W, 
    '''
    h, w = srcImg.shape
    # Create an array with values from 1 to H*W and reshape to match image dimensions
    indexes = np.arange(0, h * w).reshape(h, w)
    
    return indexes

def generateLaplacian(srcImg):
    height, width = srcImg.shape
    s = height * width
    matrix = lil_matrix((s, s))

    for i in range(width):
        for j in range(height):
            curr_index = j * width + i
            # corners
            if (i == 0 or i == width - 1) and (j == 0 or j == height - 1):
                    matrix[curr_index, curr_index] = 1
            # top & bottom
            elif (i == 0 or i == width - 1):
                matrix[curr_index, curr_index] = 2
                matrix[curr_index, curr_index - width] = -1
                matrix[curr_index, curr_index + width] = -1
            # left & right
            elif (j == 0 or j == height - 1):
                matrix[curr_index, curr_index] = 2
                matrix[curr_index, curr_index - 1] = -1
                matrix[curr_index, curr_index + 1] = -1

            else:
                matrix[curr_index, curr_index] = 4
                matrix[curr_index, curr_index - 1] = -1
                matrix[curr_index, curr_index + 1] = -1
                matrix[curr_index, curr_index - width] = -1
                matrix[curr_index, curr_index + width] = -1
    return matrix
    
def getCoefficientMatrix(indexes):
    '''
    Constructs the Coefficient Matrix A as A x = b
    
    Args:
        indexes: The indexes Matrix as the same shape as the source image, but entries
        are the indices from 1 ... h x w

    Returns:
        A: The coefficient Matrix represents "Laplacian Operator" on every pixel of the image
    '''
    imgH, imgW = indexes.shape
    N = imgH * imgW
    A = np.zeros((N, N))
    
    # Create masks
    cornerMask = np.zeros((imgH, imgW), dtype=bool)
    cornerMask[0, 0] = cornerMask[-1, -1] = cornerMask[-1, 0] = cornerMask[0, -1] = True
    cornerIdxs = indexes[cornerMask]

    horizontalBorderMask = np.zeros((imgH, imgW), dtype=bool)
    horizontalBorderMask[0, :] = horizontalBorderMask[-1, :] = True
    horizontalBorderMask[cornerMask] = False

    verticalBorderMask = np.zeros((imgH, imgW), dtype=bool)
    verticalBorderMask[:, 0] = verticalBorderMask[:, -1] = True
    verticalBorderMask[cornerMask] = False
    
    interiorMask = ~cornerMask & ~horizontalBorderMask & ~verticalBorderMask
    
    diag = np.zeros(N)
    diag[indexes[interiorMask]] = 4
    diag[indexes[horizontalBorderMask]] = 2
    diag[indexes[verticalBorderMask]] = 2
    diag[cornerIdxs] = 1
    A[np.arange(N), np.arange(N)] = diag


    y_int, x_int = np.where(interiorMask)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y_int + dy, x_int + dx
        orig_indices = indexes[y_int, x_int]
        neighbor_indices = indexes[ny, nx]
        A[orig_indices, neighbor_indices] = 1

    y_h, x_h = np.where(horizontalBorderMask)
    for dy, dx in [(0, -1), (0, 1)]:
        ny, nx = y_h + dy, x_h + dx
        # Check boundaries
        valid_mask = (nx >= 0) & (nx < imgW)
        valid_y = y_h[valid_mask]
        valid_x = x_h[valid_mask]
        valid_ny = ny[valid_mask]
        valid_nx = nx[valid_mask]
        
        orig_indices = indexes[valid_y, valid_x]
        neighbor_indices = indexes[valid_ny, valid_nx]
        A[orig_indices, neighbor_indices] = -1

    y_v, x_v = np.where(verticalBorderMask)
    for dy, dx in [(-1, 0), (1, 0)]:
        ny, nx = y_v + dy, x_v + dx
        # Check boundaries
        valid_mask = (ny >= 0) & (ny < imgH)
        valid_y = y_v[valid_mask]
        valid_x = x_v[valid_mask]
        valid_ny = ny[valid_mask]
        valid_nx = nx[valid_mask]
        
        orig_indices = indexes[valid_y, valid_x]
        neighbor_indices = indexes[valid_ny, valid_nx]
        A[orig_indices, neighbor_indices] = -1
    
    # For corner pixels, ensure the rest of the row is 0
    A[cornerIdxs, :] = 0
    
    return sparse.csr_matrix(A)
    

def getSolutionVector(srcImg, indexes):
    '''
    Constructs the solution vector b for the linear system Av = b
    
    Args:
        srcImg: The source image (grayscale) that we want to reconstruct
        indexes: The indexes Matrix, used to identify corner pixel locations.
        
    Returns:
        b: The solution vector of size N (where N = height * width)
    '''
    
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    b_matrix = signal.convolve2d(srcImg, laplacian_kernel, 'same')
    
    b_normalized = np.clip(b_matrix, 0, 255)
    b_normalized = b_normalized.astype(np.uint8)
    
    cv2.imshow('edge image', b_normalized)
    
    b_matrix[0, 0] = srcImg[0, 0]
    b_matrix[0, -1] = srcImg[0, -1]
    b_matrix[-1, 0] = srcImg[-1, 0]
    b_matrix[-1, -1] = srcImg[-1, -1]

    # @TODO 
    print(b_matrix.shape)
    # @TODO
        
    b = b_matrix.flatten()
    
    return b

def reconstructImg(srcImg):
    indexes = getIndexes(srcImg)
    
    # @TODO
    H, W = srcImg.shape
    A = generateLaplacian(srcImg)
    A = A.tocsr()
    # @TODO
    
    # A = getCoefficientMatrix(indexes)
    b = getSolutionVector(srcImg, indexes)
    v = sparse.linalg.spsolve(A, b)
    print(v)
    reconstructed_img = v.reshape(srcImg.shape)
    
    reconstructed_img = np.clip(reconstructed_img, 0, 255)
    reconstructed_img = reconstructed_img.astype(np.uint8)

    cv2.imshow('Source Image', srcImg)
    cv2.imshow('Reconstructed Image', reconstructed_img)
    print("Press any key to close the image windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('reconstructed_image.jpg', reconstructImg)
    
    

if __name__ == '__main__':
    # Load the Source Image as GrayScale Image
    srcImg = cv2.imread('large.jpg', cv2.IMREAD_GRAYSCALE)
    reconstructImg(srcImg)