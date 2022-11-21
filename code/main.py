import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # TODO ...
    """
    svd(A) : Gets U, S, V^T that satisfies A = U S V^T
    S : Eigenvalue of A^T*A
    The eigenvector when the eigenvalue is 0 can be calculated by V_T[-1, :]
    """
    n = len(p1) - 1
    A = []  # 2N x 9 matrix
    # Build A 
    while n > 0:
        A.append([p2[n][0], p2[n][1], 1, 0, 0, 0, -p1[n][0] * p2[n][0], -p1[n][0] * p2[n][1], -p1[n][0]])
        A.append([0, 0, 0, p2[n][0], p2[n][1], 1, -p1[n][1] * p2[n][0], -p1[n][1] * p2[n][1], -p1[n][1]])
        n -= 1
    A = np.asarray(A)

    # SVD
    U, S, V_T = np.linalg.svd(A)
    h = V_T[-1, :]
    H = np.reshape(h, (3, 3))
    # Normalize
    H /= H[2][2]

    return H

def compute_h_norm(p1, p2):
    # TODO ...
    # Transformation matrix
    # (x, y) -> (x/col, y/row)
    T = [[1/400, 0, 0], [0, 1/302, 0], [0, 0, 1]]
    T = np.asarray(T)
    inv_T = np.linalg.inv(T)

    p1_norm, p2_norm = p1, p2
    # Compute normalized coordinates
    for i in range(len(p1)):
        p1_coord = np.matmul(T, [p1_norm[i][0], p1_norm[i][1], 1])
        p2_coord = np.matmul(T, [p2_norm[i][0], p2_norm[i][1], 1])
        p1_norm[i][0], p1_norm[i][1] = p1_coord[0], p1_coord[1]
        p2_norm[i][0], p2_norm[i][1] = p2_coord[0], p2_coord[1]
        
    H_norm = compute_h(p1_norm, p2_norm)

    # H = T^-1 * H_norm * T
    H = np.matmul(np.matmul(inv_T, H_norm), T)
    # Floating point problem
    H /= H[2][2]

    return H

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    inv_H = np.linalg.inv(H)    # Inverse warping
    in_y, in_x, _ = igs_in.shape
    ref_y, ref_x, _ = igs_ref.shape

    # Corners of igs_in & igs_ref
    in_corners = [(0, 0), (in_y, in_x), (0, in_x), (in_y, 0)]
    ref_corners = [(0, 0), (ref_y, ref_x), (0, ref_x), (ref_y, 0)]

    # Maximum and minimum values of x & y in igs_warp
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    # Compute min, max
    for i, j in in_corners:
        # Homogeneous coordinate
        x, y, w = np.matmul(H, [j, i, 1])
        x = x/w
        y = y/w

        if x > max_x:
            max_x = int(x)
        if x < min_x:
            min_x = int(x)
        if y > max_y:
            max_y = int(y)
        if y < min_y:
            min_y = int(y)

    igs_warp = np.zeros((max_y - min_y, max_x - min_x, 3))

    # Compute igs_warp
    for i in range(0, max_x - min_x):
        for j in range(0, max_y - min_y):
            # Homogeneous coordinate
            x, y, w = np.matmul(inv_H, [i + min_x, j + min_y, 1])
            # interpolate
            x = int(x/w)
            y = int(y/w)
            # Colors
            r, g, b = 0, 0, 0
            if not (y < 0 or y >= in_y or x < 0 or x >= in_x):
                r, g, b = igs_in[y, x, :]
            igs_warp[j, i, :] = [r, g, b]

    # old : min & max of igs_warp
    old_min_x = min_x
    old_min_y = min_y
    old_max_x = max_x
    old_max_y = max_y

    # Compute min, max of igs_merge
    for i, j in ref_corners:
        if j > max_x:
            max_x = int(j)
        if j < min_x:
            min_x = int(j)
        if i > max_y:
            max_y = int(i)
        if i < min_y:
            min_y = int(i)
    
    igs_merge = np.zeros(((max_y - min_y), (max_x - min_x), 3))

    # Compute igs_merge
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            r, g, b = 0, 0, 0
            # igs_warp
            if not (j < old_min_y or j >= old_max_y or i < old_min_x or i >= old_max_x):
                r, g, b = igs_warp[j - old_min_y, i - old_min_x, :]
                if r * g * b == 0.0:
                    if not (j < 0 or j >= ref_y or i < 0 or i >= ref_x):
                        r, g, b = igs_ref[j, i, :]
            # igs_ref
            else:
                if not (j < 0 or j >= ref_y or i < 0 or i >= ref_x):
                    r, g, b = igs_ref[j, i, :]                  
            igs_merge[j - min_y, i - min_x, :] = [r, g, b] 

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...
    # Get homography matrix of p1->p2
    H = compute_h(p2, p1)
    # Use warp_image to get igs_rec
    igs_rec, _ = warp_image(igs, np.zeros(igs.shape), H)

    return igs_rec

def set_cor_mosaic():
    # TODO ...
    """
    p_in : N x 2 matrices of corresponded (x, y)^T coordinates in igs_in
    p_ref : N x 2 matrices of corresponded (x, y)^T coordinates in igs_ref
    N=13 point corresondences are chosen.
    Points that I chose are mentioned in the writeup.
    """
    p_in = [
        [370, 244],
        [155, 162],
        [214, 190],
        [164, 125],
        [339, 92],
        [173, 187],
        [217, 161],
        [88, 174],
        [223, 103],
        [384, 137],
        [359, 194],
        [357, 110],
        [325, 288]
    ]

    p_ref = [
        [111, 98],
        [347, 68],
        [272, 74],
        [387, 97],
        [259, 253],
        [305, 58],
        [298, 90],
        [376, 34],
        [368, 149],
        [161, 209],
        [146, 135],
        [218, 232],
        [130, 58]
    ]
    
    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    """
    p_in : N x 2 matrices of corresponded (x, y)^T coordinates in igs_in
    p_ref : N x 2 matrices of corresponded (x, y)^T coordinates in igs_ref
    N=4 point corresondences are chosen.
    Points that I chose are mentioned in the writeup.
    """
    c_in = [
        [163, 14],
        [260, 25],
        [163, 258],
        [260, 245]
    ]

    c_ref = [
        [200, 70],
        [260, 1],
        [150, 252],
        [258, 250]
    ]

    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/wdc1.png').convert('RGB')
    img_ref = Image.open('data/wdc2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('wdc1_warped.png')
    img_merge.save('wdc_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
