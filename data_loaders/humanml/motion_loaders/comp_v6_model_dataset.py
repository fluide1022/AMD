import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
from text2length import LengthEstimator
from copy import deepcopy

def lengths_to_mask(lengths, max_len):
	# max_len = max(lengths)
	mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
	return mask

def SlerpAutoreg(sample, len_0, len_1):
	start_idx, end_idx = len_0-3, len_0+3
	for i in range(start_idx, end_idx):
		u = (i-start_idx)/(end_idx - start_idx)
		sample[:,:,i] = (1.0-u)*sample[:,:,start_idx] + u*sample[:,:,end_idx]
	return sample

def Slerp(sample, len_0, len_1):
	prefix_end = 0.90
	suffix_start = 0.10
	start_idx, end_idx = int(prefix_end * len_0), int(len_0 + suffix_start * len_1)
	for i in range(start_idx, end_idx):
		u = (i-start_idx)/(end_idx - start_idx)
		sample[:,:,i] = (1.0-u)*sample[:,:,start_idx] + u*sample[:,:,end_idx]
	return sample

def build_models(opt):
	if opt.text_enc_mod == 'bigru':
		text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
										pos_size=opt.dim_pos_ohot,
										hidden_size=opt.dim_text_hidden,
										device=opt.device)
		text_size = opt.dim_text_hidden * 2
	else:
		raise Exception("Text Encoder Mode not Recognized!!!")

	seq_prior = TextDecoder(text_size=text_size,
							input_size=opt.dim_att_vec + opt.dim_movement_latent,
							output_size=opt.dim_z,
							hidden_size=opt.dim_pri_hidden,
							n_layers=opt.n_layers_pri)


	seq_decoder = TextVAEDecoder(text_size=text_size,
								 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
								 output_size=opt.dim_movement_latent,
								 hidden_size=opt.dim_dec_hidden,
								 n_layers=opt.n_layers_dec)

	att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
						 key_dim=text_size,
						 value_dim=opt.dim_att_vec)

	movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
	movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

	len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

	# latent_dis = LatentDis(input_size=opt.dim_z * 2)
	checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
	len_estimator.load_state_dict(checkpoints['estimator'])
	len_estimator.to(opt.device)
	len_estimator.eval()

	# return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
	return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

	def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
		assert mm_num_samples < len(dataset)
		print(opt.model_dir)

		dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
		text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
		trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
		epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
		generated_motion = []
		mm_generated_motions = []
		mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
		mm_idxs = np.sort(mm_idxs)
		min_mov_length = 10 if opt.dataset_name == 't2m' else 6
		# print(mm_idxs)

		print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
		trainer.eval_mode()
		trainer.to(opt.device)
		with torch.no_grad():
			for i, data in tqdm(enumerate(dataloader)):
				word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
				tokens = tokens[0].split('_')
				word_emb = word_emb.detach().to(opt.device).float()
				pos_ohot = pos_ohot.detach().to(opt.device).float()

				pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
				pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

				mm_num_now = len(mm_generated_motions)
				is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):
					mov_length = torch.multinomial(pred_dis, 1, replacement=True)
					if mov_length < min_mov_length:
						mov_length = torch.multinomial(pred_dis, 1, replacement=True)
					if mov_length < min_mov_length:
						mov_length = torch.multinomial(pred_dis, 1, replacement=True)

					m_lens = mov_length * opt.unit_length
					pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
														  m_lens[0]//opt.unit_length, opt.dim_pose)
					if t == 0:
						# print(m_lens)
						# print(text_data)
						sub_dict = {'motion': pred_motions[0].cpu().numpy(),
									'length': m_lens[0].item(),
									'cap_len': cap_lens[0].item(),
									'caption': caption[0],
									'tokens': tokens}
						generated_motion.append(sub_dict)

					if is_mm:
						mm_motions.append({
							'motion': pred_motions[0].cpu().numpy(),
							'length': m_lens[0].item()
						})
				if is_mm:
					mm_generated_motions.append({'caption': caption[0],
												 'tokens': tokens,
												 'cap_len': cap_lens[0].item(),
												 'mm_motions': mm_motions})

		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.opt = opt
		self.w_vectorizer = w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		if m_length < self.opt.max_motion_length:
			motion = np.concatenate([motion,
									 np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
									 ], axis=0)
		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

	def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		self.dataloader = dataloader
		self.dataset = dataloader.dataset
		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()


		with torch.no_grad():
			for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
															device=dist_util.dev()) * scale

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = i in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):

					sample = sample_fn(
						model,
						motion.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
						# when experimenting guidance_scale we want to nutrileze the effect of noise on generation
					)

					if t == 0:
						sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
									# 'length': 196,
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
										'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
										# 'length': 196,
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]


		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompT2LMDMGeneratedDatasetV2(Dataset):

	def __init__(self, checkpoints_name, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		self.dataloader = dataloader
		self.dataset = dataloader.dataset

		T2LModel = LengthEstimator(checkpoints_name)

		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()


		with torch.no_grad():
			for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
				# motion duration prediction
				motion_lens = T2LModel.t2l_eval(model_kwargs['y']['text'])
				print('Pred motion len: ',motion_lens[0])
				print('True motion len: ',model_kwargs['y']['lengths'].cpu().numpy()[0])
				print('Sample motion len: ',motion.shape[3])
				# print(196)

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
															device=dist_util.dev()) * scale

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = i in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):

					sample = sample_fn(
						model,
						motion.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
						# when experimenting guidance_scale we want to nutrileze the effect of noise on generation
					)

					if t == 0:
						sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': motion_lens[bs_i],
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
										'length': motion_lens[bs_i],
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]


		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

# new MDMGenDataset with T2L single batch
class CompT2LMDMGeneratedDatasetV1(Dataset):

	def __init__(self, checkpoints_name, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		self.dataloader = dataloader
		self.dataset = dataloader.dataset

		T2LModel = LengthEstimator(checkpoints_name)

		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()


		with torch.no_grad():
			for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
				# motion duration prediction
				motion_len = T2LModel.t2l(model_kwargs['y']['text'][0])
				# print(motion_len.item())
				# print(model_kwargs['y']['lengths'][0].item())
				
				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
															device=dist_util.dev()) * scale

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = i in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []

				sample_shape = motion.numpy().shape[:3] + (motion_len.item(),)
				# print(sample_shape)
				for t in range(repeat_times):

					sample = sample_fn(
						model,
						sample_shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
						# when experimenting guidance_scale we want to nutrileze the effect of noise on generation
					)
					# print(sample.shape)

					if t == 0:
						sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
										'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]


		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']
		max_motion_length = 196
		if m_length < max_motion_length:
			motion = np.concatenate([motion,
									 np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
									 ], axis=0)

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompAutoregMDMGeneratedDataset(Dataset):

	def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		print("Generate AUTOREG")
		self.dataloader = dataloader
		self.dataset = dataloader.dataset
		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()

		with torch.no_grad():
			for j, (motion_0, cond_0, motion_1, cond_1) in tqdm(enumerate(dataloader)):
				# if j>0:
				# 	break

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				# add CFG scale to batch
				if scale != 1.:
					cond_0['y']['scale'] = torch.ones(motion_0.shape[0], device=dist_util.dev()) * scale
					cond_1['y']['scale'] = cond_0['y']['scale']

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = j in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):

					sample_0 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
						motion_0.shape,
						clip_denoised=clip_denoised,
						model_kwargs=cond_0,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)
					cond_1['y']['motion_prev'] = sample_0
					sample_1 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[1]),
						motion_1.shape,
						clip_denoised=clip_denoised,
						model_kwargs=cond_1,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)

					tokens_concat = []
					for i in range(len(cond_0['y']['tokens_curr'])):
						tokens_0 = cond_0['y']['tokens_curr'][i].split('_')
						tokens_1 = cond_1['y']['tokens_curr'][i].split('_')
						tokens_concat.append(tokens_0 + tokens_1)

					text_concat = deepcopy(cond_0['y']['text_curr'])
					for i in range(len(text_concat)):
						text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]

					length_0 = deepcopy(cond_0['y']['lengths_curr'].cpu().numpy())
					length_1 = deepcopy(cond_1['y']['lengths_curr'].cpu().numpy())

					sample_0 = sample_0.cpu().numpy()
					sample_1 = sample_1.cpu().numpy()
					sample_concat = []
					for i in range(sample_0.shape[0]):
						s_tmp_0 = sample_0[i][:,:,0:length_0[i]]
						s_tmp_1 = sample_1[i][:,:,0:length_1[i]]
						len_tmp_concat = length_0[i]+length_1[i]
						s_tmp_concat = np.concatenate((s_tmp_0, s_tmp_1, np.zeros((s_tmp_0.shape[0], s_tmp_0.shape[1], 196*2 - len_tmp_concat))), axis = 2)
						s_tmp_concat = SlerpAutoreg(s_tmp_concat, length_0[i], length_1[i])
						sample_concat.append(s_tmp_concat)
					sample_concat = torch.tensor(np.array(sample_concat))

					# sample_concat = torch.tensor(np.random.random((motion_0.shape[0], motion_0.shape[1], motion_0.shape[2], motion_0.shape[3]*2))).float()
					if t == 0:
						sub_dicts = [{'motion': sample_concat[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': length_0[bs_i] + length_1[bs_i],
									'caption': text_concat[bs_i],
									'tokens': tokens_concat[bs_i],
									'cap_len': len(tokens_concat[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample_concat[bs_i].squeeze().permute(1,0).cpu().numpy(),
										'length': length_0[bs_i] + length_1[bs_i],
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': text_concat[bs_i],
									'tokens': tokens_concat[bs_i],
									'cap_len': len(tokens_concat[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]


		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompJointMDMGeneratedDataset(Dataset):

	def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		print("Generate JOINT")
		self.dataloader = dataloader
		self.dataset = dataloader.dataset
		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()

		with torch.no_grad():
			for j, (motion_0, cond_0, motion_1, cond_1) in tqdm(enumerate(dataloader)):
				# if j>0:
				# 	break

				model_kwargs = {'y':{}}

				text_concat = deepcopy(cond_0['y']['text_curr'])
				for i in range(len(text_concat)):
					text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]
				model_kwargs['y']['text'] = text_concat
				# print(model_kwargs['y']['text'])

				length_concat = deepcopy(cond_0['y']['lengths_curr'])
				for i in range(len(length_concat)):
					length_concat[i] = length_concat[i] + cond_1['y']['lengths_curr'][i]
				model_kwargs['y']['lengths'] = length_concat
				# print(model_kwargs['y']['lengths'])

				model_kwargs['y']['mask'] = lengths_to_mask(model_kwargs['y']['lengths'], max_motion_length).unsqueeze(1).unsqueeze(1)
				# print(model_kwargs['y']['mask'] .shape)

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs['y']['scale'] = torch.ones(motion_0.shape[0], device=dist_util.dev()) * scale

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = j in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []

				for t in range(repeat_times):

					sample = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
						(motion_0.shape[0], motion_0.shape[1], motion_0.shape[2], motion_0.shape[3]*2),
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)
					# sample = torch.tensor(np.random.random((motion_0.shape[0], motion_0.shape[1], motion_0.shape[2], motion_0.shape[3]*2))).float()

					tokens = []
					for i in range(len(cond_0['y']['tokens_curr'])):
						tokens_0 = cond_0['y']['tokens_curr'][i].split('_')
						tokens_1 = cond_1['y']['tokens_curr'][i].split('_')
						tokens.append(tokens_0 + tokens_1)

					if t == 0:
						sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
									# 'length': 196,
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
										'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
										# 'length': 196,
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]

		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompSlerpMDMGeneratedDataset(Dataset):

	def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		print("Generate SLERP")
		self.dataloader = dataloader
		self.dataset = dataloader.dataset
		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()

		with torch.no_grad():
			for j, (motion_0, cond_0, motion_1, cond_1) in tqdm(enumerate(dataloader)):
				# if j>0:
				# 	break

				model_kwargs_0 = {'y':{}}
				model_kwargs_0['y']['text'] = deepcopy(cond_0['y']['text_curr'])
				model_kwargs_0['y']['lengths'] = deepcopy(cond_0['y']['lengths_curr'])
				model_kwargs_0['y']['mask'] = deepcopy(cond_0['y']['mask_curr'])
				
				model_kwargs_1 = {'y':{}}
				model_kwargs_1['y']['text'] = deepcopy(cond_1['y']['text_curr'])
				model_kwargs_1['y']['lengths'] = deepcopy(cond_1['y']['lengths_curr'])
				model_kwargs_1['y']['mask'] = deepcopy(cond_1['y']['mask_curr'])

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs_0['y']['scale'] = torch.ones(motion_0.shape[0], device=dist_util.dev()) * scale
					model_kwargs_1['y']['scale'] = model_kwargs_0['y']['scale']

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = j in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):

					sample_0 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
						motion_0.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs_0,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)

					sample_1 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[1]),
						motion_1.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs_1,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)
					# sample_concat = torch.tensor(np.random.random((motion_0.shape[0], motion_0.shape[1], motion_0.shape[2], motion_0.shape[3]*2))).float()
					
					tokens_concat = []
					for i in range(len(cond_0['y']['tokens_curr'])):
						tokens_0 = cond_0['y']['tokens_curr'][i].split('_')
						tokens_1 = cond_1['y']['tokens_curr'][i].split('_')
						tokens_concat.append(tokens_0 + tokens_1)

					text_concat = deepcopy(cond_0['y']['text_curr'])
					for i in range(len(text_concat)):
						text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]

					length_0 = deepcopy(cond_0['y']['lengths_curr'].cpu().numpy())
					length_1 = deepcopy(cond_1['y']['lengths_curr'].cpu().numpy())

					sample_0 = sample_0.cpu().numpy()
					sample_1 = sample_1.cpu().numpy()
					sample_concat = []
					for i in range(sample_0.shape[0]):
						s_tmp_0 = sample_0[i][:,:,0:length_0[i]]
						s_tmp_1 = sample_1[i][:,:,0:length_1[i]]
						len_tmp_concat = length_0[i]+length_1[i]
						s_tmp_concat = np.concatenate((s_tmp_0, s_tmp_1, np.zeros((s_tmp_0.shape[0], s_tmp_0.shape[1], 196*2 - len_tmp_concat))), axis = 2)
						s_tmp_concat = Slerp(s_tmp_concat, length_0[i], length_1[i])
						sample_concat.append(s_tmp_concat)
					sample_concat = torch.tensor(np.array(sample_concat))


					if t == 0:
						sub_dicts = [{'motion': sample_concat[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': length_0[bs_i] + length_1[bs_i],
									'caption': text_concat[bs_i],
									'tokens': tokens_concat[bs_i],
									'cap_len': len(tokens_concat[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample_concat[bs_i].squeeze().permute(1,0).cpu().numpy(),
										'length': length_0[bs_i] + length_1[bs_i],
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': text_concat[bs_i],
									'tokens': tokens_concat[bs_i],
									'cap_len': len(tokens_concat[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]


		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompFillMDMGeneratedDataset(Dataset):

	def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
		print("Generate FILL")
		self.dataloader = dataloader
		self.dataset = dataloader.dataset
		assert mm_num_samples < len(dataloader.dataset)
		use_ddim = False  # FIXME - hardcoded
		clip_denoised = False  # FIXME - hardcoded
		self.max_motion_length = max_motion_length
		sample_fn = (
			diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
		)

		real_num_batches = len(dataloader)
		if num_samples_limit is not None:
			real_num_batches = num_samples_limit // dataloader.batch_size + 1
		print('real_num_batches', real_num_batches)

		generated_motion = []
		mm_generated_motions = []
		if mm_num_samples > 0:
			mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
			mm_idxs = np.sort(mm_idxs)
		else:
			mm_idxs = []
		print('mm_idxs', mm_idxs)

		model.eval()

		with torch.no_grad():
			for j, (motion_0, cond_0, motion_1, cond_1) in tqdm(enumerate(dataloader)):
				# if j>0:
				# 	break

				model_kwargs_0 = {'y':{}}
				model_kwargs_0['y']['text'] = deepcopy(cond_0['y']['text_curr'])
				model_kwargs_0['y']['lengths'] = deepcopy(cond_0['y']['lengths_curr'])
				model_kwargs_0['y']['mask'] = deepcopy(cond_0['y']['mask_curr'])
				
				model_kwargs_1 = {'y':{}}
				model_kwargs_1['y']['text'] = deepcopy(cond_1['y']['text_curr'])
				model_kwargs_1['y']['lengths'] = deepcopy(cond_1['y']['lengths_curr'])
				model_kwargs_1['y']['mask'] = deepcopy(cond_1['y']['mask_curr'])

				model_kwargs = {'y':{}}
				text_concat = deepcopy(cond_0['y']['text_curr'])
				for i in range(len(text_concat)):
					text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]
				model_kwargs['y']['text'] = text_concat
				# print(model_kwargs['y']['text'])

				model_kwargs['y']['lengths'] =  deepcopy(cond_0['y']['lengths_curr'])

				if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
					break

				# add CFG scale to batch
				if scale != 1.:
					model_kwargs_0['y']['scale'] = torch.ones(motion_0.shape[0], device=dist_util.dev()) * scale
					model_kwargs_1['y']['scale'] = model_kwargs_0['y']['scale']

				mm_num_now = len(mm_generated_motions) // dataloader.batch_size
				is_mm = j in mm_idxs
				repeat_times = mm_num_repeats if is_mm else 1
				mm_motions = []
				for t in range(repeat_times):

					sample_0 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
						motion_0.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs_0,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)

					sample_1 = sample_fn(
						model,
						# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[1]),
						motion_1.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs_1,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)
					max_frames = max_motion_length // 2
					length_0 = deepcopy(model_kwargs_0['y']['lengths'].cpu().numpy())
					length_1 = deepcopy(model_kwargs_1['y']['lengths'].cpu().numpy())
					sample_0 = sample_0.cpu().numpy()
					sample_1 = sample_1.cpu().numpy()
					input_motions = []
					for i in range(sample_0.shape[0]):
						clip_start = max(0, length_0[i]-max_frames//2)
						s_tmp_0 = sample_0[i][:,:,clip_start:length_0[i]]
						s_tmp_1 = sample_1[i][:,:,0: max_frames-length_0[i]+clip_start]
						s_tmp_concat = np.concatenate((s_tmp_0, s_tmp_1), axis = 2)
						
						model_kwargs['y']['lengths'][i] = min(length_1[i] + length_0[i] - clip_start, max_frames)
						input_motions.append(s_tmp_concat)
					input_motions = torch.tensor(np.array(input_motions)).float()
					input_motions = input_motions.to(dist_util.dev())
					# print('input_motions: ',input_motions.shape)

					# add inpainting mask according to args
					model_kwargs['y']['mask'] = lengths_to_mask(model_kwargs['y']['lengths'], max_frames).unsqueeze(1).unsqueeze(1)
					model_kwargs['y']['inpainted_motion'] = input_motions
					model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
																		   device=input_motions.device)  # True means use gt motion
					for i in range(sample_0.shape[0]):
						clip_start = max(0, length_0[i]-max_frames//2)
						start_idx, end_idx = int(0.90 * length_0[i]), int(length_0[i] + 0.10 * length_1[i])
						model_kwargs['y']['inpainting_mask'][i, :, :,
						start_idx-clip_start: end_idx-clip_start] = False  # do inpainting in those frames
					model_kwargs['y']['scale'] = torch.ones(motion_0.shape[0], device=dist_util.dev()) * scale

					sample = sample_fn(
						model,
						motion_0.shape,
						clip_denoised=clip_denoised,
						model_kwargs=model_kwargs,
						skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
						init_image=None,
						progress=False,
						dump_steps=None,
						noise=None,
						const_noise=False,
					)

					tokens = []
					for i in range(len(cond_0['y']['tokens_curr'])):
						tokens_0 = cond_0['y']['tokens_curr'][i].split('_')
						tokens_1 = cond_1['y']['tokens_curr'][i].split('_')
						tokens.append(tokens_0 + tokens_1)

					sample_concat = []
					for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
						clip_start = max(0, length_0[i]-max_frames//2)
						s_tmp_0 = sample_0[i][:,:,0:clip_start]
						s_tmp_1 = sample_1[i][:,:, min(max_frames-length_0[i]+clip_start, length_1[i]):length_1[i]]
						model_kwargs['y']['lengths'][i] = s_tmp_0.shape[2] + length + s_tmp_1.shape[2]
						s_tmp_concat = np.concatenate((s_tmp_0, sample[i][:,:,:length].cpu().numpy(), s_tmp_1, np.zeros((s_tmp_0.shape[0], s_tmp_0.shape[1], max_frames*2 - model_kwargs['y']['lengths'][i]))), axis = 2)
						# print(s_tmp_concat.shape)
						sample_concat.append(s_tmp_concat)
					sample_concat = torch.tensor(np.array(sample_concat)).float()
					sample_concat = sample_concat.to(dist_util.dev())
					# sample_concat = torch.tensor(np.random.random((motion_0.shape[0], motion_0.shape[1], motion_0.shape[2], motion_0.shape[3]*2))).float()
					if t == 0:
						sub_dicts = [{'motion': sample_concat[bs_i].squeeze().permute(1,0).cpu().numpy(),
									'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
									# 'length': 196,
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									} for bs_i in range(dataloader.batch_size)]
						generated_motion += sub_dicts

					if is_mm:
						mm_motions += [{'motion': sample_concat[bs_i].squeeze().permute(1, 0).cpu().numpy(),
										'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
										# 'length': 196,
										} for bs_i in range(dataloader.batch_size)]

				if is_mm:
					mm_generated_motions += [{
									'caption': model_kwargs['y']['text'][bs_i],
									'tokens': tokens[bs_i],
									'cap_len': len(tokens[bs_i]),
									'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
									} for bs_i in range(dataloader.batch_size)]



		self.generated_motion = generated_motion
		self.mm_generated_motion = mm_generated_motions
		self.w_vectorizer = dataloader.dataset.w_vectorizer


	def __len__(self):
		return len(self.generated_motion)


	def __getitem__(self, item):
		data = self.generated_motion[item]
		motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
		sent_len = data['cap_len']

		if self.dataset.mode == 'eval':
			normed_motion = motion
			denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
			renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
			motion = renormed_motion
			# This step is needed because T2M evaluators expect their norm convention

		pos_one_hots = []
		word_embeddings = []
		for token in tokens:
			word_emb, pos_oh = self.w_vectorizer[token]
			pos_one_hots.append(pos_oh[None, :])
			word_embeddings.append(word_emb[None, :])
		pos_one_hots = np.concatenate(pos_one_hots, axis=0)
		word_embeddings = np.concatenate(word_embeddings, axis=0)

		return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)