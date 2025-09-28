import numpy as np
from reconstruction import getSolutionVector, getIndexes

# Test with a small 3x3 image
test_img = np.array([
    [1, 2, 3],
    [4, 5, 6], 
    [7, 8, 9]
], dtype=np.float64)

print("Test image:")
print(test_img)
print()

indexes = getIndexes(test_img)
print("Indexes:")
print(indexes)
print()

b = getSolutionVector(test_img, indexes)
print("Solution vector b:")
print(b)
print(f"Shape: {b.shape}")
print()

# Reshape back to 2D to visualize
b_2d = b.reshape(test_img.shape)
print("Solution vector as 2D:")
print(b_2d)