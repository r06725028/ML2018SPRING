import numpy as np 
from numpy import linalg as la 
import skimage
from skimage import io
import os
import sys

img_path = sys.argv[1]
img_name = sys.argv[2]

output_size = (600,600,3)
top_k = 4

def read_data1(img_path):
	data = []
	index_dict = {}

	for dirPath, dirNames, fileNames in os.walk(img_path):#遍歷資料夾下每個jpg
		for i,file in enumerate(fileNames):
			index_dict[file] = i
			img = io.imread(os.path.join(img_path,file))
			data.append(img.flatten())
	return np.array(data),index_dict

def show_recon(img_path,img_name,output_size,top_k):
	data,index_dict = read_data1(img_path)
	
	mean = np.mean(data,axis=0)

	data = data-mean
	U,sigma,VT=la.svd(data.T,full_matrices=False)

	y = data[index_dict[img_name],]#1xM
	weight = np.dot(y,U[:,:top_k])#(1xM) * (Mxk) = (1xk)
	print('weight shape ',weight.shape)
	y_re = np.dot(U[:,:top_k],weight.T)+mean.T#(Mxk) * (kx1) = (Mx1)

	y_re -= np.min(y_re,axis=0)
	y_re /= np.max(y_re,axis=0)
	y_re = (y_re*255).astype(np.uint8)
	io.imsave('reconstruction.jpg', y_re.reshape(output_size))

show_recon(img_path,img_name,output_size,top_k)






