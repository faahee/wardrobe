import cv2, numpy as np
from sklearn.cluster import KMeans
from collections import Counter

img=cv2.imread('imgg.png', cv2.IMREAD_UNCHANGED)
print('shape', img.shape, 'dtype', img.dtype)
if img.shape[2]==4:
    print('has alpha channel')
    alpha=img[:,:,3]
    print('alpha unique', np.unique(alpha)[:10])
    img=img[:,:,:3]
# convert to RGB
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2]==3 else img
print('top-left pixel', img_rgb[0,0].tolist())
print('top-right pixel', img_rgb[0,-1].tolist())
# sample a few pixels
h,w,_=img_rgb.shape
pixels=img_rgb.reshape(-1,3).astype(np.float32)
print('mean RGB', pixels.mean(axis=0).tolist())
# most frequent colors
cnt=Counter([tuple(p) for p in pixels.astype(int)])
most=cnt.most_common(10)
print('most common colors (top5):', most[:5])
# run kmeans
k=2
kmeans=KMeans(n_clusters=k,random_state=0,n_init=10).fit(pixels)
centers=np.round(kmeans.cluster_centers_).astype(int)
labels=kmeans.labels_
unique,counts=np.unique(labels,return_counts=True)
print('centers', centers.tolist())
print('counts', dict(zip(unique,counts)))
# show mapping to simple palette
PALETTE={'navy':(0,0,128),'beige':(245,245,220),'charcoal':(54,69,79),'white':(255,255,255),'black':(0,0,0)}
for i,c in enumerate(centers):
    dists={name:sum((int(c[j])-v[j])**2 for j in range(3)) for name,v in PALETTE.items()}
    nearest=min(dists.items(),key=lambda x:x[1])
    print('center',i,'rgb',c.tolist(),'nearest simple',nearest)
