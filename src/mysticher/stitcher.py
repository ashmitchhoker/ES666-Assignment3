import cv2
import numpy as np
import glob 
import os

class PanaromaStitcher():
    def __init__(self):
        pass

    def homography_matrix(self,src_pts, dst_pts): 
        A=[]
        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0], src_pts[i][1]
            x2, y2 = dst_pts[i][0], dst_pts[i][1]
            z1=z2=1
            A_i = [  [0, 0, 0, -z2*x1, -z2*y1, -z2*z1, y2*x1, y2*y1, y2*z1],
                    [z2*x1, z2*y1, z2*z1, 0, 0, 0, -x2*x1, -x2*y1, -x2*z1]  ]
            A.extend(A_i)

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        return H / H[2, 2]  # Normalize H

    def RANSAC(self,src_pts,dst_pts,max_iters=1000,sigma=1):
            threshold = np.sqrt(5.99) * sigma
            best_H = None
            best_inliers = []
            max_inliers = 0

            for _ in range(max_iters):
                # Step 1: Randomly sample 4 points
                random_idxs = np.random.choice(src_pts.shape[0], 4, replace=False)
                src_sample = src_pts[random_idxs]
                dst_sample = dst_pts[random_idxs]

                # Step 2: calculate H
                H = self.homography_matrix(src_sample, dst_sample)

                src_pts_homogeneous = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])  # Convert to homogeneous coordinates
                projected_pts = np.dot(H, src_pts_homogeneous.T).T  # Apply homography
                projected_pts = projected_pts[:, :2] / projected_pts[:, 2].reshape(-1, 1)
                
                # Step 4: Calculate the error (distance between projected points and destination points)
                errors = np.linalg.norm(dst_pts - projected_pts, axis=1)
                
                # Step 5: Find inliers where the error is below the threshold
                inliers = np.where(errors < threshold)[0]
                
                # Step 6: If the number of inliers is larger than the previous best, update the best model
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_H = H
                    best_inliers = inliers
                
            if len(best_inliers) > 4:
                best_H = self.homography_matrix(src_pts[best_inliers], dst_pts[best_inliers])
            
            return best_H, best_inliers

    def cylindrical_projection(self,img, f):
        h, w = img.shape[:2]
        cylindrical_img = np.zeros_like(img)
        center_x, center_y = w // 2, h // 2

        for x in range(w):
            for y in range(h):
                theta = (x - center_x) / f
                h_ = (y - center_y) / f
                X, Y, Z = np.sin(theta), h_, np.cos(theta)
                x_img, y_img = int(f * X / Z + center_x), int(f * Y / Z + center_y)

                if 0 <= x_img < w and 0 <= y_img < h:
                    cylindrical_img[y, x] = img[y_img, x_img]
        
        return cylindrical_img

    def compute_homography(self,image1, image2):
        detector = cv2.SIFT_create()
        keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(image2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 4:
            print("Not enough matches found.")
            return None

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        H, inliers = self.RANSAC(src_pts, dst_pts)
        return H

    def warp_image(self,image, H, target_shape):
        return cv2.warpPerspective(image, H, target_shape)

    def blend_images(self,image1, image2):
        mask = image2 > 0
        image1[mask] = image2[mask]
        return image1

    def crop_black_borders(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return image[y:y+h, x:x+w]
        return image
    
    def calculate_output_size(self, H_matrices, anchor_shape):
        # Compute the new dimensions based on the homography matrices
        # Initialize min and max coordinates to find the bounding box
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for H in H_matrices:
            # Calculate the corners of the original image
            corners = np.array([[0, 0, 1], [anchor_shape[1], 0, 1],
                                [anchor_shape[1], anchor_shape[0], 1], [0, anchor_shape[0], 1]])
            transformed_corners = H @ corners.T
            transformed_corners /= transformed_corners[2, :]  # Normalize by the third row

            # Find new bounding box
            min_x = min(min_x, transformed_corners[0, :].min())
            max_x = max(max_x, transformed_corners[0, :].max())
            min_y = min(min_y, transformed_corners[1, :].min())
            max_y = max(max_y, transformed_corners[1, :].max())

        # Calculate output size
        width = int(max_x - min_x)
        height = int(max_y - min_y)
        return (width, height)


    def make_panaroma_for_images_in(self,path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        images = [cv2.imread(img) for img in all_images]

        anchor_index = len(images) // 2
        f = images[anchor_index].shape[1] / (2 * np.pi) * 7
        images = [self.cylindrical_projection(img, f) for img in images]
        anchor_image = images[anchor_index]
        H_matrices = [None] * len(images)
        H_matrices[anchor_index] = np.eye(3)

        for i in range(anchor_index - 1, -1, -1):
            H_matrices[i] = self.compute_homography(images[i], images[i + 1]) @ H_matrices[i + 1]

        for i in range(anchor_index + 1, len(images)):
            H_matrices[i] = np.linalg.inv(self.compute_homography(images[i - 1], images[i])) @ H_matrices[i - 1]

        output_size = self.calculate_output_size(H_matrices, anchor_image.shape)
        panorama = self.warp_image(anchor_image, H_matrices[anchor_index], output_size)
        for i in range(len(images)):
            if i != anchor_index:
                warped_image = self.warp_image(images[i], H_matrices[i], output_size)
                panorama = self.blend_images(panorama, warped_image)

        return self.crop_black_borders(panorama),H_matrices


