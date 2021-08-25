import cv2
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
# from skimage.color import rgb2lab, deltaE_cie76
import matplotlib.pyplot as plt

image_path = 'img_cap.jpg'
number_of_colors = 4

# Helper function for image display
def cv2plt(img):
    plt.figure(figsize=(8,8))        # To change the size of figure
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)  
    plt.show()
# Helper function to convert color to color HEX
def RGB2HEX(color):
    return "#{:2x}{:2x}{:2x}".format(int(color[0]), int(color[1]), int(color[2]))

image = cv2.imread(image_path)

img_crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img_crop_array = img_crop.reshape(img_crop.shape[0] * img_crop.shape[1], 3)

clf = KMeans(n_clusters = 2)

labels = clf.fit_predict(img_crop_array)
counts = Counter(labels)

center_colors = clf.cluster_centers_
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

plt.figure(figsize = (8, 6))
plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
plt.show()