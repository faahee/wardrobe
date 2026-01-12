import numpy as np, cv2
h,w=200,200
left=np.full((h,w//2,3),(0,0,128),np.uint8)
right=np.full((h,w-w//2,3),(245,245,220),np.uint8)
img=np.concatenate([left,right],axis=1)
cv2.imwrite('test_half.png', img[:,:,::-1])
print('created test_half.png')
