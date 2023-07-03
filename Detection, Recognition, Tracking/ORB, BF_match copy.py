import cv2

img1 = cv2.imread('./data/mang1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/mang2.jpg', cv2.IMREAD_GRAYSCALE)

# ORB

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(img1, None) # descriptors1.shape = (500, 32)임
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# BF matching

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2) # len(matches) = 500임
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matches", result)

num_matches = len(matches)
num_good_matches = sum(1 for m in matches if m.distance < 50)
matching_percent = (num_good_matches / num_matches) * 100
print("Matching Percent: %.2f%%" % matching_percent)

#

cv2.waitKey(0)
cv2.destroyAllWindows()