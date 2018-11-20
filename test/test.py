import os
import numpy as np
import caffe
import sys
from pylab import *
import re
import random
import time
import copy
import matplotlib.pyplot as plt
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime

def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
		print (templateFile)
		outFile = 'DeployT.prototxt'
		with open(outFile, 'w') as fd:
			fd.write(template.format(height=height,width=width))

def split_image(image, grid_size):

	height = image.shape[0]
	width = image.shape[1]

	height_div = int(height / grid_size[0])
	width_div = int(width / grid_size[1])

	height_idxs = []
	idx = 0
	while idx < height:
		idx += height_div
		height_idxs.append(idx)

	width_idxs = []
	idx = 0
	while idx < width:
		idx += width_div
		width_idxs.append(idx)

	# print ('height_idxs:', height_idxs)
	# print ('width_idxs:', width_idxs)

	grids = []
	for h in height_idxs:
		h = int(h)
		for w in width_idxs:
			w = int(w)
			grids.append(image[h-height_div:h, w-width_div:w])

	# print (len(grids))
	# for grid in grids:
	# 	print (grid.shape)

	return grids

def stitch_image(splits, grid_size):

	# idx = 0
	stitched_until_now = np.concatenate(splits[:grid_size[1]], 1)
	idx = grid_size[1]
	# print ('stitched_until_now', stitched_until_now.shape)
	for i in range(grid_size[0]-1):
		stitched_row = np.concatenate(splits[idx:idx+grid_size[1]], 1)
		idx += grid_size[1]
		# print ('stitched_row', stitched_row.shape)
		stitched_until_now = np.concatenate([stitched_until_now, stitched_row])
		# print ('stitched_until_now', stitched_until_now.shape)

	return stitched_until_now

def test(stitch):
	caffe.set_mode_gpu()
	caffe.set_device(0)
	# caffe.set_mode_cpu()

	info = os.listdir('../data/img');
	imagesnum=0;
	for line in info:
		reg = re.compile(r'(.*?).jpg');
		all = reg.findall(line)
		if (all != []):
			imagename = str(all[0]);
			if (os.path.isfile(r'../data/img/%s.jpg' % imagename) == False):
				continue;
			else:
				imagesnum = imagesnum + 1;
				npstore = caffe.io.load_image('../data/img/%s.jpg'% imagename)
				print('Image shape:', npstore.shape)

				if stitch:
					grid_size = (4, 4)
					splits = split_image(npstore, grid_size)
					out_splits = []
					for split in splits:
						height = split.shape[0]
						width = split.shape[1]

						templateFile = 'test_template.prototxt'
						EditFcnProto(templateFile, height, width)


						model='../AOD_Net.caffemodel';

						net = caffe.Net('./DeployT.prototxt', model, caffe.TEST);
						batchdata = []
						data = split

						split = split.transpose((2, 0, 1))
						batchdata.append(split)
						net.blobs['data'].data[...] = batchdata;

						net.forward();

						split = net.blobs['sum'].data[0];
						split = split.transpose((1, 2, 0));
						split = split[:, :, ::-1]

						out_splits.append(split)

					# print (len(out_splits))

					stitched_out = stitch_image(out_splits, grid_size)

					savepath = '../data/result/' + imagename + '_AOD-Net_stitched.jpg'
					cv2.imwrite(savepath, stitched_out * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])

				else:
					npstore = caffe.io.resize_image(npstore, (2177,2902))
					height = npstore.shape[0]
					width = npstore.shape[1]

					templateFile = 'test_template.prototxt'
					EditFcnProto(templateFile, height, width)

					model='../AOD_Net.caffemodel';

					net = caffe.Net('./DeployT.prototxt', model, caffe.TEST);
					batchdata = []
					data = npstore

					data = data.transpose((2, 0, 1))
					batchdata.append(data)
					net.blobs['data'].data[...] = batchdata;

					net.forward();

					data = net.blobs['sum'].data[0];
					data = data.transpose((1, 2, 0));
					data = data[:, :, ::-1]

					savepath = '../data/result/' + imagename + '_AOD-Net_resized.jpg'
					cv2.imwrite(savepath, data * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])

				print (imagename)

		print ('image numbers:',imagesnum)

def main():
	stitch = False
	test(stitch)


if __name__ == '__main__':
	main();


