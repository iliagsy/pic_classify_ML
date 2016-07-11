# coding:utf-8
# 预处理
# 找到每类min{nri,nci}，把该类所有图片中心截一个正方形
# 纪录每个label的训练样例个数，供preprocessing使用

import matplotlib.image as mpimg
import numpy as np

label_names = ['glass','city','earth','plane','dog','flowers','gold','car','gun','worldcup','fruits','sky','fireworks']
label_names = {k:v  for k,v in enumerate( label_names )}
not_truncate_labels = {'glass', 'gun'}

nClass = 13

def find_min_r_c():
	min_r_c = []
	pic_num = []
	for label in range( nClass ):
		file_name_ = ('./%d/' % label) + label_names[ label ] + '_'
		
		min_r_c_ = None
	
		for pic_No in range(200):
			file_name = file_name_ + '%04d.jpg' % pic_No
			try:
				img = mpimg.imread( file_name )
			except:
				pic_num.append( pic_No )
				break
			
			if min_r_c_ is None:
				min_r_c_ = min(img.shape[:1])
			elif min(img.shape[:1]) < min_r_c_:
				min_r_c_ = min(img.shape[:1])
				
		min_r_c.append( min_r_c_ )
		
	return pic_num, min_r_c

pic_num_lst, min_r_c_lst = find_min_r_c()

for label in range( nClass ):
	file_name_ = ('./%d/' % label) + label_names[ label ] + '_'
	
	min_r_c = min_r_c_lst[ label ]
	
	for pic_No in range(200):
		file_name = file_name_ + '%04d.jpg' % pic_No
		try:
			in_img = mpimg.imread( file_name )
		except:
			break
		
		if label_names[ label ] not in not_truncate_labels:
			# truncate
			nRow = in_img.shape[0]
			nCol = in_img.shape[1]
			in_img = in_img[ nRow/2 - min_r_c/2.0 : nRow/2 + min_r_c/2, 
				nCol/2 - min_r_c/2 : nCol/2 + min_r_c/2, ... ]
			
		# save
		mpimg.imsave( file_name, in_img, None,None,None, 'jpg')