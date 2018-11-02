import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.1


def alignImages(im1, im2):
    # Convert images to grayscale
    # im1: Color image to be aligned
    # im2: Reference image
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(100)
    ny = 4
    nx = 3

    keypoints1 = getKeypoints(orb, im1Gray, ny, nx)
    keypoints2 = getKeypoints(orb, im2Gray, ny, nx)

    # compute the descriptors with ORB
    keypoints1, descriptors1 = orb.compute(im1Gray, keypoints1)
    keypoints2, descriptors2 = orb.compute(im2Gray, keypoints2)

    # Match features.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("Images/matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.LMEDS)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

def getKeypoints(orb, image, ny, nx):
    keypoints = []
    for j in range(ny):
        for i in range(nx):
            img_patch = getPatch(image, ny, nx, j, i)
            kp = orb.detect(img_patch, None)
            if kp:
                keypoints += kp
    return keypoints

def getPatch(image, ny, nx, j, i):
    sh = image.shape
    y_low = int(sh[0]/ny*j)
    y_high = int(sh[0]/ny*(j+1))
    x_low = int(sh[1] / nx * i)
    x_high = int(sh[1] / nx * (i + 1))
    new_image = np.zeros(sh, dtype=image.dtype)
    new_image[y_low:y_high, x_low:x_high] = image[y_low:y_high, x_low:x_high]
    return new_image

def deWarp(im_target, im_reference):
    # Convert images to grayscale
    im_target_gray = cv2.cvtColor(im_target, cv2.COLOR_BGR2GRAY)
    im_reference_gray = cv2.cvtColor(im_reference, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(im_reference_gray, im_target_gray, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=1,
                                        poly_n=5, poly_sigma=1.1,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return warp_flow(im_target, flow)

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    # flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    # Read reference image
    refFilename = "Images/2018-03-24_20-00MEZ_Mond.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "Images/2018-03-24_21-01MEZ_Mond.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "Images/2018-03-24_21-01MEZ_Mond_aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)

    imDewarped = deWarp(imReg, imReference)
    # Write de-warped image to disk.
    outFilename = "Images/2018-03-24_21-01MEZ_Mond_dewarped.jpg"
    print("Saving de-warped image : ", outFilename)
    cv2.imwrite(outFilename, imDewarped)