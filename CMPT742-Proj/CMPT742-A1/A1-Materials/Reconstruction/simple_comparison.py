import numpy as np
import scipy.sparse as sparse
from scipy.sparse import lil_matrix

# Simplified versions of the functions to avoid OpenCV dependency
def getIndexes_simple(h, w):
    '''Create indexes matrix without needing actual image'''
    return np.arange(0, h * w).reshape(h, w)

def generateLaplacian_simple(height, width):
    '''Generate Laplacian matrix - simplified version'''
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

def getCoefficientMatrix_simple(indexes):
    '''Simplified version of getCoefficientMatrix'''
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
    
    # Set diagonal elements
    diag = np.zeros(N)
    diag[indexes[interiorMask]] = 4
    diag[indexes[horizontalBorderMask]] = 2
    diag[indexes[verticalBorderMask]] = 2
    diag[cornerIdxs] = 1
    A[np.arange(N), np.arange(N)] = diag

    # Set off-diagonal elements for interior pixels
    y_int, x_int = np.where(interiorMask)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y_int + dy, x_int + dx
        orig_indices = indexes[y_int, x_int]
        neighbor_indices = indexes[ny, nx]
        A[orig_indices, neighbor_indices] = -1

    # Set off-diagonal elements for horizontal border pixels
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

    # Set off-diagonal elements for vertical border pixels  
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
    # Reset diagonal for corners
    A[cornerIdxs, cornerIdxs] = 1
    
    return sparse.csr_matrix(A)

def compare_matrices_detailed(mat1, mat2, name1, name2):
    '''Compare two matrices in detail'''
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    # Convert to CSR format for comparison
    if not sparse.issparse(mat1):
        mat1 = sparse.csr_matrix(mat1)
    if not sparse.issparse(mat2):
        mat2 = sparse.csr_matrix(mat2)
        
    mat1_csr = mat1.tocsr()
    mat2_csr = mat2.tocsr()
    
    print(f"Shape: {name1} = {mat1_csr.shape}, {name2} = {mat2_csr.shape}")
    print(f"Non-zero elements: {name1} = {mat1_csr.nnz}, {name2} = {mat2_csr.nnz}")
    
    if mat1_csr.shape == mat2_csr.shape:
        diff = mat1_csr - mat2_csr
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        print(f"Maximum absolute difference: {max_diff}")
        
        if max_diff < 1e-10:
            print("✓ MATRICES ARE IDENTICAL!")
            return True
        else:
            print("✗ MATRICES DIFFER")
            
            # Show differences
            diff_coo = diff.tocoo()
            print(f"Number of different elements: {diff_coo.nnz}")
            
            if diff_coo.nnz > 0:
                print("First 10 differences:")
                for i in range(min(10, diff_coo.nnz)):
                    row, col = diff_coo.row[i], diff_coo.col[i]
                    val1 = mat1_csr[row, col]
                    val2 = mat2_csr[row, col]
                    print(f"  Position ({row:2d}, {col:2d}): {name1} = {val1:6.1f}, {name2} = {val2:6.1f}")
            return False
    else:
        print("✗ MATRICES HAVE DIFFERENT SHAPES!")
        return False

def test_functions():
    '''Test both functions with different image sizes'''
    
    test_cases = [
        (2, 2, "2x2"),
        (3, 3, "3x3"),
        (2, 3, "2x3"), 
        (4, 3, "4x3")
    ]
    
    print("Testing if getCoefficientMatrix and generateLaplacian produce the same results")
    print("=" * 80)
    
    all_identical = True
    
    for height, width, description in test_cases:
        print(f"\nTesting {description} image:")
        print("-" * 40)
        
        # Create indexes
        indexes = getIndexes_simple(height, width)
        print(f"Indexes matrix:\n{indexes}")
        
        # Generate matrices using both functions
        coeff_matrix = getCoefficientMatrix_simple(indexes)
        laplacian_matrix = generateLaplacian_simple(height, width)
        
        # Compare the matrices
        identical = compare_matrices_detailed(
            coeff_matrix, laplacian_matrix, 
            "getCoefficientMatrix", "generateLaplacian"
        )
        
        if not identical:
            all_identical = False
            
            # Print dense versions for small matrices to see the differences
            if height * width <= 16:  # Only for small matrices
                print(f"\ngetCoefficientMatrix (dense):\n{coeff_matrix.toarray()}")
                print(f"\ngenerateLaplacian (dense):\n{laplacian_matrix.toarray()}")
    
    print("\n" + "=" * 80)
    if all_identical:
        print("✓ CONCLUSION: Both functions generate IDENTICAL matrices for all test cases!")
    else:
        print("✗ CONCLUSION: The functions generate DIFFERENT matrices!")
    print("=" * 80)

if __name__ == "__main__":
    test_functions()