import cv2
import sys
import numpy as np
import random
import my_sift

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None

    max_inlier = []
    kp_length = len(list_pairs_matched_keypoints) - 1  # for random purpose

    if(len(list_pairs_matched_keypoints) < 4):  # need at least 4 correspondances
        print ("Not enough matched keypoints to build homography!")
        sys.exit(1)

    for i in range(max_num_trial):
        p1 = list_pairs_matched_keypoints[random.randint(0, kp_length)]
        p2 = list_pairs_matched_keypoints[random.randint(0, kp_length)]
        p3 = list_pairs_matched_keypoints[random.randint(0, kp_length)]
        p4 = list_pairs_matched_keypoints[random.randint(0, kp_length)]

        four_matches = []
        four_matches.append(p1)
        four_matches.append(p2)
        four_matches.append(p3)
        four_matches.append(p4)

        matrix_A = []
        for m in four_matches:  # build matrix A
            v1 = [m[0][0], m[0][1], 1]
            v2 = [m[1][0], m[1][1], 1]
            A1_row = [-v1[0], -v1[1], -1, 0, 0, 0, v1[0]*v2[0], v1[1]*v2[0], v2[0]]
            A2_row = [0, 0, 0, -v1[0], -v1[1], -1, v1[0]*v2[1], v1[1]*v2[1], v2[1]]
            matrix_A.append(A1_row)
            matrix_A.append(A2_row)
        u, s, v = np.linalg.svd(matrix_A)
        Homo = np.reshape(v[8], (3, 3))  # build Homography
        Homo = (Homo/Homo.item(8))  # normalize
        #print(Homo)
        
        num_inliers = []  # number of inliers for this homography
        for j in list_pairs_matched_keypoints:  # calculate eulidean distance
            src = [j[0][0], j[0][1], 1]
            est = np.dot(Homo, src)
            est = est/est[2] 
            dest = [j[1][0], j[1][1], 1]
            error = np.sqrt(np.sum((dest - est) **2))
            if error < threshold_reprojtion_error:
                num_inliers.append(j)
        # end for

        if len(num_inliers) > len(max_inlier):  # update current best inlier
            max_inlier = num_inliers
            alter_H = Homo
            if (len(max_inlier) / len(list_pairs_matched_keypoints) ) > threshold_ratio_inliers:
                best_H = Homo
    # end 1000 iterations for loop

    #print ("max inlier length = ", len(max_inlier))
    #print ("all match length = ", len(list_pairs_matched_keypoints))
    if best_H is None:
        print ("Cannot find best Homography matrix based on ratio = ", threshold_ratio_inliers)
        print ("An alternative Homography is used based on ratio = ", len(max_inlier) / len(list_pairs_matched_keypoints))
        best_H = alter_H

    # re-compute H with all inliers
    matrix_A = []
    for m in max_inlier:
        v1 = [m[0][0], m[0][1], 1]
        v2 = [m[1][0], m[1][1], 1]
        A1_row = [-v1[0], -v1[1], -1, 0, 0, 0, v1[0]*v2[0], v1[1]*v2[0], v2[0]]
        A2_row = [0, 0, 0, -v1[0], -v1[1], -1, v1[0]*v2[1], v1[1]*v2[1], v2[1]]
        matrix_A.append(A1_row)
        matrix_A.append(A2_row)
    u, s, v = np.linalg.svd(matrix_A)
    Homo = np.reshape(v[8], (3, 3))  # build Homography
    Homo = (Homo/Homo.item(8))  # normalize Homo
    best_H = Homo

    return best_H



def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================

    aa = my_sift.mysift(img_1)
    kp1, descriptor1 = aa.my_detect_and_compute()
    bb = my_sift.mysift(img_2)
    kp2, descriptor2 = bb.my_detect_and_compute()

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    matches = []
    first_dist = -1.0
    second_dist = -1.0
    first_idx = -1
    second_idx = -1

    # use eulidean distance to find first and second nearest point
    for i in range(len(descriptor1)):
        for j in range(len(descriptor2)):
            distance = np.sqrt(np.sum((descriptor1[i] - descriptor2[j]) **2))
            if first_dist == -1.0:  # first time compare
                first_dist = distance
                first_idx = j
            elif distance < first_dist:
                second_dist = first_dist
                second_idx = first_idx
                first_dist = distance
                first_idx = j
            elif second_dist == -1.0:
                second_dist = distance
                second_idx = j    
            elif distance < second_dist:
                second_dist = distance
                second_idx = j
        #end j for
        matches.append([i, first_dist, first_idx, second_dist, second_idx])  # [i(img-1), 100(first_closest), 10(index), 101(second closest), 50]
        first_dist = -1.0
        second_dist = -1.0
        first_idx = -1
        second_idx = -1           
    #end i for

    for i in range(len(matches)):  # apply ratio test to select the set of robust matched points
        if (matches[i][1] / matches[i][3]) < ratio_robustness:
            m_index = matches[i][2]
            list_pairs_matched_keypoints.append( [ [kp1[i][0], kp1[i][1]], [kp2[m_index][0], kp2[m_index][1]] ] )

    return list_pairs_matched_keypoints



def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...
    H_inverse = np.linalg.inv(H_1)
    # f11 - f14 : 4 ending points corresponding to warped img_1
    f11 = np.dot(H_1, [0, 0, 1])
    f11 = f11/f11[2]
    f12 = np.dot(H_1, [0, img_1.shape[0], 1])
    f12 = f12/f12[2]
    f13 = np.dot(H_1, [img_1.shape[1], 0, 1])
    f13 = f13/f13[2]
    f14 = np.dot(H_1, [img_1.shape[1], img_1.shape[0], 1])
    f14 = f14/f14[2]
    hh = [f11[1], f12[1], f13[1], f14[1]]
    ww = [f11[0], f12[0], f13[0], f14[0]]
    
    h0 = int(abs(np.floor(min(hh))))
    h1 = int(np.floor(min(hh)))  # min height
    h2 = int(np.ceil(max(hh)))   # max height
    w0 = int(abs(np.floor(min(ww))))
    w1 = int(np.floor(min(ww)))  # min width
    w2 = int(np.ceil(max(ww)))   # max width
    right_w = img_2.shape[1] - w2
    
    warp = np.zeros([abs(h1) + abs(h2), abs(w1) + abs(w2) + right_w, 3], dtype=np.float32)

    for y in range(h1, h2):
        for x in range(w1, w2 + right_w):
            # for bilinear interpolation
            x_y = np.dot(H_inverse, [x, y, 1])
            x_y = x_y/x_y[-1]
            i = int(np.floor(x_y[0]))
            j = int(np.floor(x_y[1]))
            a = x_y[0] - i
            b = x_y[1] - j
            
            #out of bound
            if i < 0 or j < 0 or i >= img_1.shape[1]-1 or j >= img_1.shape[0]-1:
                # stitch img2 to img1
                if y >= 0 and y < img_2.shape[0] and x >= 0 and x < img_2.shape[1]:
                    result = img_2[y, x]
                else:
                    result = [0, 0, 0]  # black

            # bilinear interpolation for warping img_1
            else:
                bi1 = ((1 - a) * (1 - b)) * (img_1[j, i])
                bi2 = (a * (1 - b)) * (img_1[j, i + 1])
                bi3 = (a * b) * (img_1[j + 1, i + 1])
                bi4 = ((1 - a) * b) * (img_1[j + 1, i])
                result = bi1 + bi2 + bi3 + bi4
                if x >= 0 and x < img_2.shape[1] and y >= 0 and y < img_2.shape[0]:
                    r2 = img_2[y, x]
                    result = (result + r2) / 2.0

            warp[y + h0, x + w0] = result

    img_panorama = warp
    return img_panorama



def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)

    return img_panorama



if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)
    
    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))
