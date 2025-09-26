import numpy as np
import scipy.sparse as sparse
from scipy.sparse import lil_matrix
from reconstruction import getCoefficientMatrix, getIndexes
import warnings

warnings.filterwarnings("ignore")

def generateLaplacian_fixed(srcImg):
    """
    Fixed version of generateLaplacian that takes srcImg parameter correctly
    """
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

def compare_sparse_matrices(matrix1, matrix2, name1="Matrix1", name2="Matrix2"):
    """
    Compare two sparse matrices and print detailed analysis
    """
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    # Convert to same format for comparison
    if hasattr(matrix1, 'tocsr'):
        mat1_csr = matrix1.tocsr()
    else:
        mat1_csr = sparse.csr_matrix(matrix1)
    
    if hasattr(matrix2, 'tocsr'):
        mat2_csr = matrix2.tocsr()
    else:
        mat2_csr = sparse.csr_matrix(matrix2)
    
    # Basic properties
    print(f"Shape: {name1} = {mat1_csr.shape}, {name2} = {mat2_csr.shape}")
    print(f"Non-zero elements: {name1} = {mat1_csr.nnz}, {name2} = {mat2_csr.nnz}")
    print(f"Density: {name1} = {mat1_csr.nnz / (mat1_csr.shape[0] * mat1_csr.shape[1]):.6f}, {name2} = {mat2_csr.nnz / (mat2_csr.shape[0] * mat2_csr.shape[1]):.6f}")
    
    # Check if matrices are identical
    if mat1_csr.shape == mat2_csr.shape:
        diff_matrix = mat1_csr - mat2_csr
        max_diff = np.abs(diff_matrix.data).max() if diff_matrix.nnz > 0 else 0
        print(f"Maximum absolute difference: {max_diff}")
        
        if max_diff < 1e-10:
            print("✓ MATRICES ARE IDENTICAL!")
        else:
            print("✗ Matrices differ")
            
            # Show some examples of differences
            diff_coo = diff_matrix.tocoo()
            if diff_coo.nnz > 0:
                print("First few differences:")
                for i in range(min(5, diff_coo.nnz)):
                    row, col = diff_coo.row[i], diff_coo.col[i]
                    val1 = mat1_csr[row, col]
                    val2 = mat2_csr[row, col]
                    print(f"  Position ({row}, {col}): {name1} = {val1}, {name2} = {val2}")
    else:
        print("✗ Matrices have different shapes!")
    
    return mat1_csr, mat2_csr

def analyze_matrix_structure(matrix, name):
    """
    Analyze the structure of a sparse matrix
    """
    print(f"\n=== Structure Analysis: {name} ===")
    if hasattr(matrix, 'tocsr'):
        mat_csr = matrix.tocsr()
    else:
        mat_csr = sparse.csr_matrix(matrix)
    
    # Sample some diagonal elements
    diag_elements = mat_csr.diagonal()
    unique_diag, counts_diag = np.unique(diag_elements, return_counts=True)
    print(f"Diagonal elements: {dict(zip(unique_diag, counts_diag))}")
    
    # Sample some off-diagonal elements
    mat_coo = mat_csr.tocoo()
    off_diag_mask = mat_coo.row != mat_coo.col
    off_diag_data = mat_coo.data[off_diag_mask]
    if len(off_diag_data) > 0:
        unique_off_diag, counts_off_diag = np.unique(off_diag_data, return_counts=True)
        print(f"Off-diagonal elements: {dict(zip(unique_off_diag, counts_off_diag))}")
    
    return mat_csr

def test_small_image():
    """
    Test with a small image to easily verify results
    """
    print("=" * 60)
    print("Testing with small 3x3 image")
    print("=" * 60)
    
    # Create a small test image
    small_img = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]], dtype=np.uint8)
    
    print(f"Test image shape: {small_img.shape}")
    print(f"Test image:\n{small_img}")
    
    # Get indexes
    indexes = getIndexes(small_img)
    print(f"Indexes:\n{indexes}")
    
    # Test both functions
    coeff_matrix = getCoefficientMatrix(indexes)
    laplacian_matrix = generateLaplacian_fixed(small_img)
    
    # Convert to CSR for consistent comparison
    coeff_csr = coeff_matrix.tocsr() if hasattr(coeff_matrix, 'tocsr') else sparse.csr_matrix(coeff_matrix)
    laplacian_csr = laplacian_matrix.tocsr() if hasattr(laplacian_matrix, 'tocsr') else sparse.csr_matrix(laplacian_matrix)
    
    # Analyze individual matrices
    analyze_matrix_structure(coeff_csr, "getCoefficientMatrix")
    analyze_matrix_structure(laplacian_csr, "generateLaplacian")
    
    # Compare matrices
    compare_sparse_matrices(coeff_csr, laplacian_csr, "getCoefficientMatrix", "generateLaplacian")
    
    # Print dense versions for small matrices (for visual inspection)
    print(f"\ngetCoefficientMatrix (dense):\n{coeff_csr.toarray()}")
    print(f"\ngenerateLaplacian (dense):\n{laplacian_csr.toarray()}")

def test_medium_image():
    """
    Test with a medium-sized image
    """
    print("\n" + "=" * 60)
    print("Testing with medium 5x4 image")
    print("=" * 60)
    
    # Create a medium test image
    medium_img = np.random.randint(0, 255, (5, 4), dtype=np.uint8)
    
    print(f"Test image shape: {medium_img.shape}")
    
    # Get indexes
    indexes = getIndexes(medium_img)
    
    # Test both functions
    coeff_matrix = getCoefficientMatrix(indexes)
    laplacian_matrix = generateLaplacian_fixed(medium_img)
    
    # Analyze and compare
    analyze_matrix_structure(coeff_matrix, "getCoefficientMatrix")
    analyze_matrix_structure(laplacian_matrix, "generateLaplacian")
    compare_sparse_matrices(coeff_matrix, laplacian_matrix, "getCoefficientMatrix", "generateLaplacian")

if __name__ == "__main__":
    print("Testing sparse matrix equivalence between getCoefficientMatrix and generateLaplacian")
    
    # Run tests
    test_small_image()
    test_medium_image()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)