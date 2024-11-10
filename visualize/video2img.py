import cv2
import os
import numpy as np

video_path = ''  
output_path = video_path.replace(video_path.split('/')[-1], 'img/')  
if not os.path.exists(output_path):
	os.makedirs(output_path)
interval = 10  

if __name__ == '__main__':
	num = 1
	vid = cv2.VideoCapture(video_path)
	while vid.isOpened():
		is_read, frame = vid.read()
		if is_read:
			if num % interval == 1:
				b_channel, g_channel, r_channel = cv2.split(frame)
				alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
				frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
				file_name = '%06d' % num
				for i in range(frame.shape[0]):
					for j in range(frame.shape[1]):
						# print(frame[i][j])
						if np.linalg.norm(frame[i][j][0:3]-[255,255,255]) < 125:
							frame[i][j][3] = 0
				cv2.imwrite(output_path + str(file_name) + '.png', frame)
				cv2.waitKey(1)
			num += 1

		else:
			break