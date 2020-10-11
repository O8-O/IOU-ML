import numpy as np
import cv2

ITERATE_NUM = 500

img = cv2.imread('./Image/many_step/interior_deleted.png')

mask = cv2.imread('./Image/many_step/interior_deleted_mask.png', 0)

'''
mask_half1 = cv2.imread('./Image/many_step/interior_deleted_mask_001.png', 0)
mask_half2 = cv2.imread('./Image/many_step/interior_deleted_mask_002.png', 0)


next_picture = cv2.inpaint(img, mask_half2, 3, cv2.INPAINT_TELEA)
next_picture = cv2.inpaint(next_picture, mask_half1, 3, cv2.INPAINT_TELEA)

for _ in range(0, ITERATE_NUM):
    next_picture = cv2.inpaint(img, mask_half1, 3, cv2.INPAINT_TELEA)
    next_picture = cv2.inpaint(next_picture, mask_half2, 3, cv2.INPAINT_TELEA)
'''

'''
devided_mask = [
    cv2.imread('./Image/many_step/interior_deleted_mask_01.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_02.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_03.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_04.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_05.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_06.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_07.png', 0),
    cv2.imread('./Image/many_step/interior_deleted_mask_08.png', 0),
]

next_picture = cv2.inpaint(img, devided_mask[0], 20, cv2.INPAINT_TELEA)
for dm in range(1, len(devided_mask)):
    next_picture = cv2.inpaint(next_picture, devided_mask[dm], 3, cv2.INPAINT_TELEA)

for _ in range(0, ITERATE_NUM):
    for _ in range(0, len(devided_mask)):
        next_picture = cv2.inpaint(next_picture, devided_mask[dm], 3, cv2.INPAINT_TELEA)

'''

next_picture = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

for _ in range(0, ITERATE_NUM):
    next_picture = cv2.inpaint(next_picture, mask, 3, cv2.INPAINT_TELEA)

cv2.imshow('Changed', next_picture)
cv2.waitKey(0)
cv2.destroyAllWindows()