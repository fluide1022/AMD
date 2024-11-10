import sys, os
sys.path.append('..')
sys.path.append('../T2M/')
import numpy as np
import torch.distributed.launch

from os.path import join as pjoin

# import T2M.utils.paramUtil as paramUtil
from T2M.options.evaluate_options import TestOptions
from torch.utils.data import DataLoader

from T2M.networks.modules import *
from T2M.data.dataset import RawTextDataset, SingleTextDataset, BatchTextDataset
from T2M.utils.word_vectorizer import WordVectorizer, POS_enumerator

class LengthEstimator():
	def __init__(self,  checkpoints_name, checkpoints_choice = 'finest.tar'):
		dataset_name = 't2m'
		self.unit_length = 4
		self.device = torch.device("cuda:2")
		torch.autograd.set_detect_anomaly(True)

		self.w_vectorizer = WordVectorizer('./glove', 'our_vab')

		self.mean = np.load(pjoin('../T2M/checkpoints', dataset_name, checkpoints_name, 'meta', 'mean.npy'))
		self.std = np.load(pjoin('../T2M/checkpoints', dataset_name, checkpoints_name, 'meta',  'std.npy'))

		dim_word = 300
		dim_pos_ohot = len(POS_enumerator)
		num_classes = 200 // self.unit_length
		self.estimator = MotionLenEstimatorBiGRU(dim_word, dim_pos_ohot, 512, num_classes)
		checkpoints = torch.load(pjoin('../T2M/checkpoints', dataset_name, checkpoints_name, 'model', checkpoints_choice))
		self.estimator.load_state_dict(checkpoints['estimator'])
		self.estimator.to(self.device)
		self.estimator.eval()

	def t2l_gen(self, text):
		dataset = SingleTextDataset(20, self.mean, self.std, text, self.w_vectorizer)

		data_loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=1)
		with torch.no_grad():
			for i, data in enumerate(data_loader):
				# print('%02d_%03d'%(i, len(data_loader)))
				word_emb, pos_ohot, caption, cap_lens = data
				item_dict = {'caption': caption}
				# print(caption)

				word_emb, pos_ohot, caption, cap_lens = data
				word_emb = word_emb.detach().to(self.device).float()
				pos_ohot = pos_ohot.detach().to(self.device).float()

				pred_dis = self.estimator(word_emb, pos_ohot, cap_lens)
				pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

				# length = torch.multinomial(pred_dis, 1)
				min_mov_length = 10
				length = torch.multinomial(pred_dis, 1, replacement=True)
				if length < min_mov_length:
					length = torch.multinomial(pred_dis, 1, replacement=True)
				if length < min_mov_length:
					length = torch.multinomial(pred_dis, 1, replacement=True)

				m_lens = length * self.unit_length
				# print(m_lens)
		return m_lens[0]

	def t2l_eval(self, texts):
		dataset = BatchTextDataset(20, self.mean, self.std, texts, self.w_vectorizer)

		data_loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=1)
		res = []
		with torch.no_grad():
			for i, data in enumerate(data_loader):
				# print('%02d_%03d'%(i, len(data_loader)))
				word_emb, pos_ohot, caption, cap_lens = data
				item_dict = {'caption': caption}
				# print(caption)

				word_emb, pos_ohot, caption, cap_lens = data
				word_emb = word_emb.detach().to(self.device).float()
				pos_ohot = pos_ohot.detach().to(self.device).float()

				pred_dis = self.estimator(word_emb, pos_ohot, cap_lens)
				pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

				# length = torch.multinomial(pred_dis, 1)
				min_mov_length = 10
				length = torch.multinomial(pred_dis, 1, replacement=True)
				if length < min_mov_length:
					length = torch.multinomial(pred_dis, 1, replacement=True)
				if length < min_mov_length:
					length = torch.multinomial(pred_dis, 1, replacement=True)

				m_lens = length * self.unit_length
				res.append(m_lens[0].item())
				# print(m_lens)
		return np.array(res)