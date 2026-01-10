import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def normalize_points(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_distance = np.mean(distances)
    scale = np.sqrt(2) / mean_distance

    T = np.array([[scale, 0, -scale * centroid[0]],
                    [0, scale, -scale * centroid[1]],
                    [0, 0, 1]])
    normalized_points = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return normalized_points[:, :2], T

def FindFundamentalMatrix(pts1, pts2):
        #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Normalize the points
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)


    #todo: Form the matrix A
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x, y = pts1_normalized[i]
        x_prime, y_prime = pts2_normalized[i]
        A[i] = [x_prime * x, x_prime * y, x_prime, y_prime * x, y_prime * y, y_prime, x, y, 1]

    #todo: Find the fundamental matrix
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0 
    F = U @ np.diag(S) @ Vt
    F = T2.T @ F @ T1

    return F

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Run RANSAC and find the best fundamental matrix
    best_F = None
    max_inliers = 0
    num_points = pts1.shape[0]
    
    for _ in range(num_trials):
        sample_indices = np.random.choice(num_points, 8, replace=False)
        sampled_pts1 = pts1[sample_indices]
        sampled_pts2 = pts2[sample_indices]

        try:
            F_candidate = FindFundamentalMatrix(sampled_pts1, sampled_pts2)
        except np.linalg.LinAlgError:
            continue

        ones = np.ones((num_points, 1))
        pts1_h = np.hstack((pts1, ones))
        pts2_h = np.hstack((pts2, ones))

        Fx1 = F_candidate @ pts1_h.T  # (3, N)
        Fx2 = F_candidate.T @ pts2_h.T  # (3, N)
        errors = (np.sum(pts2_h * (F_candidate @ pts1_h.T).T, axis=1) ** 2) / (
            Fx1[0, :]**2 + Fx1[1, :]**2 + Fx2[0, :]**2 + Fx2[1, :]**2
        )

        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F_candidate

    return best_F

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = False

    #Load images
    image1_path = os.path.join(data_path, 'mount_rushmore_1.jpg')
    image2_path = os.path.join(data_path, 'mount_rushmore_2.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]

    print(f"image 1's path: {image1_path}")
    print(f"image 2's path: {image2_path}")
    
    # Printing the comparison
    F_true /= F_true[-1, -1] # normalize by the last element
    print("Below is F_True from OpenCV CV2")
    print(F_true)

    F_8points = FindFundamentalMatrix(pts1, pts2)
    F_8points /= F_8points[-1, -1]
    print("Below is the F found from 8 points algorithm")
    print(F_8points)

    F_ransac = FindFundamentalMatrixRansac(pts1,pts2)
    F_ransac /= F_ransac[-1, -1]
    print("Below is the implmentation result from RANSAC")
    print(F_ransac)
    

    # Set matplotlib style for better aesthetics
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.titleweight'] = 'bold'

    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_8points)
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_ransac)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(image1, image2, lines2, pts1, pts2)

    lines3 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_true)
    lines3 = lines3.reshape(-1, 3)
    img5, img6 = drawlines(image1, image2, lines3, pts1, pts2)

    # Create figure with improved spacing and layout
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Epipolar Lines: Points from Image 2 → Lines on Image 1',
                 fontweight='bold', fontsize=16, y=0.98)

    # Add subtle background color
    fig.patch.set_facecolor('#f8f9fa')

    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.15,
                          left=0.05, right=0.95, top=0.94, bottom=0.02)

    # 8-Point Algorithm
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img1)
    ax1.set_title('8-Point Algorithm', pad=10, fontsize=13, fontweight='bold', color='#2c3e50')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img2)
    ax2.set_title('Corresponding Points (Image 2)', pad=10, fontsize=13, fontweight='bold', color='#2c3e50')
    ax2.axis('off')

    # RANSAC Algorithm
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(img3)
    ax3.set_title('RANSAC Algorithm', pad=10, fontsize=13, fontweight='bold', color='#16a085')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(img4)
    ax4.set_title('Corresponding Points (Image 2)', pad=10, fontsize=13, fontweight='bold', color='#16a085')
    ax4.axis('off')

    # OpenCV Reference
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(img5)
    ax5.set_title('OpenCV Reference', pad=10, fontsize=13, fontweight='bold', color='#c0392b')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(img6)
    ax6.set_title('Corresponding Points (Image 2)', pad=10, fontsize=13, fontweight='bold', color='#c0392b')
    ax6.axis('off')

    # Add separation lines
    # Vertical line between columns (at x = 0.5 in figure coordinates)
    line_v = Line2D([0.5, 0.5], [0.02, 0.94], transform=fig.transFigure,
                    color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig.add_artist(line_v)

    # Horizontal line between row 1 and 2
    line_h1 = Line2D([0.05, 0.95], [0.645, 0.645], transform=fig.transFigure,
                     color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig.add_artist(line_h1)

    # Horizontal line between row 2 and 3
    line_h2 = Line2D([0.05, 0.95], [0.32, 0.32], transform=fig.transFigure,
                     color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig.add_artist(line_h2)

    plt.show()

    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines4 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_8points)
    lines4 = lines4.reshape(-1, 3)
    img7, img8 = drawlines(image2, image1, lines4, pts2, pts1)

    lines5 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_ransac)
    lines5 = lines5.reshape(-1, 3)
    img9, img10 = drawlines(image2, image1, lines5, pts2, pts1)

    lines6 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_true)
    lines6 = lines6.reshape(-1, 3)
    img11, img12 = drawlines(image2, image1, lines6, pts2, pts1)

    # Create second figure with improved layout
    fig2 = plt.figure(figsize=(14, 12))
    fig2.suptitle('Epipolar Lines: Points from Image 1 → Lines on Image 2',
                  fontweight='bold', fontsize=16, y=0.98)

    fig2.patch.set_facecolor('#f8f9fa')

    gs2 = fig2.add_gridspec(3, 2, hspace=0.25, wspace=0.15,
                            left=0.05, right=0.95, top=0.94, bottom=0.02)

    # 8-Point Algorithm
    ax7 = fig2.add_subplot(gs2[0, 0])
    ax7.imshow(img7)
    ax7.set_title('8-Point Algorithm', pad=10, fontsize=13, fontweight='bold', color='#2c3e50')
    ax7.axis('off')

    ax8 = fig2.add_subplot(gs2[0, 1])
    ax8.imshow(img8)
    ax8.set_title('Corresponding Points (Image 1)', pad=10, fontsize=13, fontweight='bold', color='#2c3e50')
    ax8.axis('off')

    # RANSAC Algorithm
    ax9 = fig2.add_subplot(gs2[1, 0])
    ax9.imshow(img9)
    ax9.set_title('RANSAC Algorithm', pad=10, fontsize=13, fontweight='bold', color='#16a085')
    ax9.axis('off')

    ax10 = fig2.add_subplot(gs2[1, 1])
    ax10.imshow(img10)
    ax10.set_title('Corresponding Points (Image 1)', pad=10, fontsize=13, fontweight='bold', color='#16a085')
    ax10.axis('off')

    # OpenCV Reference
    ax11 = fig2.add_subplot(gs2[2, 0])
    ax11.imshow(img11)
    ax11.set_title('OpenCV Reference', pad=10, fontsize=13, fontweight='bold', color='#c0392b')
    ax11.axis('off')

    ax12 = fig2.add_subplot(gs2[2, 1])
    ax12.imshow(img12)
    ax12.set_title('Corresponding Points (Image 1)', pad=10, fontsize=13, fontweight='bold', color='#c0392b')
    ax12.axis('off')

    # Add separation lines
    # Vertical line between columns (at x = 0.5 in figure coordinates)
    line_v2 = Line2D([0.5, 0.5], [0.02, 0.94], transform=fig2.transFigure,
                     color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig2.add_artist(line_v2)

    # Horizontal line between row 1 and 2
    line_h3 = Line2D([0.05, 0.95], [0.645, 0.645], transform=fig2.transFigure,
                     color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig2.add_artist(line_h3)

    # Horizontal line between row 2 and 3
    line_h4 = Line2D([0.05, 0.95], [0.32, 0.32], transform=fig2.transFigure,
                     color='#bdc3c7', linewidth=2, linestyle='-', alpha=0.7)
    fig2.add_artist(line_h4)

    plt.show()





