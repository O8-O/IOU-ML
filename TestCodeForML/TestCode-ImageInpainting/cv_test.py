import numpy as np
import cv2

ITERATE_NUM = 500

img = cv2.imread('./Image/many_step/interior_deleted.png')

mask = cv2.imread('./Image/many_step/interior_deleted_mask.png', 0)

next_picture = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

for _ in range(0, ITERATE_NUM):
    next_picture = cv2.inpaint(next_picture, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Changed', next_picture)
cv2.waitKey(0)
cv2.destroyAllWindows()