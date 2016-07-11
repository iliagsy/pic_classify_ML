# coding:utf-8
# 预处理

# 找出各类数目最大值，每个图片集按照这个数目replicate一定的倍数 (ndarray method?)
# （数据集变大，先只遍历一遍）

import matplotlib.image as mpimg
import numpy as np
from truncate import pic_num_lst, nClass, label_names
from compress import dim_vals, dim_val

data_set = []

max_pic_num = max( pic_num_lst )

for label in range( nClass ):
	file_name_ = ('./%d/' % label) + label_names[ label ] + '_'
	
	data_set_ = []
	data_set__ = []
	
	for pic_No in range(200):
		file_name = file_name_ + '%04d.jpg' % pic_No
		try:
			in_img = mpimg.imread( file_name )
		except:
			break
		
		img = in_img.transpose(2, 0, 1)
		data_set_.append( (img, label) )
		
	# replicate dataset
	for i in range( max_pic_num / pic_num_lst[ label ] ):
		data_set__ += data_set_
	data_set__ += data_set_[:max_pic_num % pic_num_lst[ label ]]
	data_set.append( data_set__ )
	
dim_vals = [dim_vals[2], dim_vals[0], dim_vals[1]]
print 'preprocessing done.'	
