# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import custom_generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate, collate_autoreg
from text2length import LengthEstimator
from copy import deepcopy

import time

def Slerp(sample, len_0, len_1):
	start_idx, end_idx = len_0-3, len_0+3
	for i in range(start_idx, end_idx):
		u = (i-start_idx)/(end_idx - start_idx)
		sample[:,:,i] = (1.0-u)*sample[:,:,start_idx] + u*sample[:,:,end_idx]
	return sample

def ProcessSample(model, data, sample_0, cond_0, sample_1, cond_1, lenEstimator = None):
	text_concat = deepcopy(cond_0['y']['text_curr'])
	for i in range(len(text_concat)):
		text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]

	if lenEstimator == None:
		# len_res = model_kwargs['y']['lengths'].cpu().numpy()
		length_0 = deepcopy(cond_0['y']['lengths_curr'].cpu().numpy())
		length_1 = deepcopy(cond_1['y']['lengths_curr'].cpu().numpy())
	else:
		length_0 = []
		length_1 = []
		for text in cond_0['y']['text_curr']:
			motion_len = lenEstimator.t2l_gen(text)
			length_0.append(motion_len.cpu().numpy())
		for text in cond_1['y']['text_curr']:
			motion_len = lenEstimator.t2l_gen(text)
			length_1.append(motion_len.cpu().numpy())

	# print(sample_0.shape)
	# print(sample_1.shape)
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


	n_joints = 22 if sample_concat.shape[1] == 263 else 21
	sample_concat = data.dataset.t2m_dataset.inv_transform(sample_concat.permute(0, 2, 3, 1)).float()
	sample_concat = recover_from_ric(sample_concat, n_joints)
	sample_concat = sample_concat.view(-1, *sample_concat.shape[2:]).permute(0, 2, 3, 1)

	sample_concat = model.rot2xyz(x=sample_concat, mask=None, pose_rep='xyz', glob=True, translation=True,
						   jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
						   get_rotations_back=False)
	

	motion_concat = deepcopy(sample_concat.cpu().numpy())

	return text_concat, motion_concat, length_0, length_1

def Visualize(args, total_num_samples, fps, all_text, all_motions, all_lengths_0, all_lengths_1, gt_frames_per_sample, out_path):

	all_motions = np.concatenate(all_motions, axis=0)
	all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
	all_text = all_text[:total_num_samples]
	all_lengths_0 = np.concatenate(all_lengths_0, axis=0)[:total_num_samples]
	all_lengths_1 = np.concatenate(all_lengths_1, axis=0)[:total_num_samples]

	if os.path.exists(out_path):
		shutil.rmtree(out_path)
	os.makedirs(out_path)

	npy_path = os.path.join(out_path, 'results.npy')
	print(f"saving results file to [{npy_path}]")
	np.save(npy_path,
			{'motion': all_motions, 'text': all_text, 'lengths_0': all_lengths_0, 'lengths_1': all_lengths_1,
			 'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
	with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
		fw.write('\n'.join(all_text))
	with open(npy_path.replace('.npy', '_len_0.txt'), 'w') as fw:
		fw.write('\n'.join([str(l) for l in all_lengths_0]))
	with open(npy_path.replace('.npy', '_len_1.txt'), 'w') as fw:
		fw.write('\n'.join([str(l) for l in all_lengths_1]))

	print(f"saving visualizations to [{out_path}]...")
	skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

	sample_files = []
	num_samples_in_out_file = 7
	for sample_i in range(args.num_samples):
		rep_files = []
		for rep_i in range(args.num_repetitions):
			# caption = all_text[rep_i*args.batch_size + sample_i]
			caption = ""
			length_0 = all_lengths_0[rep_i*args.batch_size + sample_i]
			length_1 = all_lengths_1[rep_i*args.batch_size + sample_i]
			# print(length_0)
			# print(length_1)
			m_tmp = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)
			# print(m_tmp.shape)
			# motion = np.concatenate((m_tmp[:length_0], m_tmp[196:196+length_1]), axis = 0)
			motion = m_tmp[:length_0+length_1]
			# print(motion.shape)
			save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
			animation_save_path = os.path.join(out_path, save_file)
			print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
			# plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
			plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
							dataset=args.dataset, fps=fps, vis_mode='in_between',
							gt_frames=gt_frames_per_sample.get(sample_i, []))
			# Credit for visualization: https://github.com/EricGuo5513/text-to-motion

			rep_files.append(animation_save_path)
		all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
		ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
		hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
		ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
		os.system(ffmpeg_rep_cmd)
		print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
		sample_files.append(all_rep_save_file)

		if (sample_i+1) % num_samples_in_out_file == 0 or sample_i+1 == args.num_samples:
			all_sample_save_file = os.path.join(out_path, f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4')
			ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
			vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
			ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_file}'
			os.system(ffmpeg_rep_cmd)
			print(f'[(samples {(sample_i - len(sample_files) + 1):02d} to {sample_i:02d}) | all repetitions | -> {all_sample_save_file}]')
			sample_files = []


	abs_path = os.path.abspath(out_path)
	print(f'[Done] Results are at [{abs_path}]')

def Generate(args, lenEstimator):
	fixseed(args.seed)
	out_path = args.output_dir
	name = os.path.basename(os.path.dirname(args.model_path))
	niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
	max_frames = 196 if args.dataset in ['kit', 'humanml', 'autoreg'] else 60
	fps = 12.5 if args.dataset == 'kit' else 20

	# np_len_list = np.expand_dims(np.array(motion_len_list), axis = 0)
	# np_max_frames = np.expand_dims(np.full(len(motion_len_list), max_frames), axis = 0)
	# n_frames_arr = np.min(np.concatenate((np_len_list, np_max_frames)), axis = 0)
	# print(n_frames_arr)

	is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
	dist_util.setup_dist(args.device)
	if out_path == '':
		out_path = os.path.join(os.path.dirname(args.model_path),
								'samples_{}_{}_seed{}'.format(name, niter, args.seed))
		gt_path = os.path.join(os.path.dirname(args.model_path),
								'GT_{}_{}_seed{}'.format(name, niter, args.seed))
	
	#=============================================================================
	# TODO: text prompt for self-defined text
		if args.text_prompt != '':
			out_path += '_' + args.text_prompt[0].replace(' ', '_').replace('.', '')
	
	# this block must be called BEFORE the dataset is loaded
	if args.text_prompt != '':
		texts = args.text_prompt
		args.num_samples = 1
	#=============================================================================
	
	assert args.num_samples <= args.batch_size, \
		f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
	args.batch_size = args.num_samples

	print('Loading dataset...')
	# data = get_dataset_loader(name=args.dataset,
	# 						  batch_size=args.batch_size,
	# 						  data_root=args.data_dir,
	# 						  num_frames=max_frames,
	# 						  split='test',
	# 						  hml_mode='text_only')

	data = get_dataset_loader(name=args.dataset,
							  batch_size=args.batch_size,
							  data_root=args.data_dir,
							  num_frames=max_frames,
							  split='val',
							  hml_mode='train')
	# data.dataset.t2m_dataset.fixed_length_arr = n_frames_arr
	total_num_samples = args.num_samples * args.num_repetitions

	print("Creating model and diffusion...")
	model, diffusion = create_model_and_diffusion(args, data)

	print(f"Loading checkpoints from [{args.model_path}]...")
	state_dict = torch.load(args.model_path, map_location='cpu')
	load_model_wo_clip(model, state_dict)

	if args.guidance_param != 1:
		model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
	model.to(dist_util.dev())
	model.eval()  # disable random masking

	if is_using_data:
		iterator = iter(data)
		motion_0, cond_0, motion_1, cond_1 = next(iterator)
		# motion_0, cond_0, motion_1, cond_1 = next(iterator)
	else:
		#=============================================================================
		# TODO: text prompt for self-defined text
		# collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples

		collate_args = [{	'text_0': texts[0],
							'motion_0': torch.zeros(max_frames), # [seqlen, J] -> [J, 1, seqlen]
							'lengths_0': max_frames,
							'tokens_0': None,
							'text_1': texts[1],
							'motion_1': torch.zeros(max_frames), # [seqlen, J] -> [J, 1, seqlen]
							'lengths_1': max_frames,
							'tokens_1': None,
						}] * args.num_samples
		_, cond_0, _, cond_1 = collate_autoreg(collate_args)
		#=============================================================================

	gt_frames_per_sample = {}

	all_motions = []
	all_lengths_0 = []
	all_lengths_1 = []
	all_text = []
	# gt_all_motions = []
	# gt_all_lengths_0 = []
	# gt_all_lengths_1 = []
	# gt_all_text = []
	for rep_i in range(args.num_repetitions):
		print(f'### Sampling [repetitions #{rep_i}]')

		# add CFG scale to batch
		if args.guidance_param != 1:
			cond_0['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
			cond_1['y']['scale'] = cond_0['y']['scale']

		sample_fn = diffusion.p_sample_loop

		sample_0 = sample_fn(
			model,
			# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
			(args.batch_size, model.njoints, model.nfeats, max_frames),
			clip_denoised=False,
			model_kwargs=cond_0,
			skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
			init_image=None,
			progress=True,
			dump_steps=None,
			noise=None,
			const_noise=False,
		)
		cond_1['y']['motion_prev'] = sample_0
		sample_1 = sample_fn(
			model,
			# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[1]),
			(args.batch_size, model.njoints, model.nfeats, max_frames),
			clip_denoised=False,
			model_kwargs=cond_1,
			skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
			init_image=None,
			progress=True,
			dump_steps=None,
			noise=None,
			const_noise=False,
		)

		# gt_text_concat, gt_motion_concat, gt_length_0, gt_length_1 = ProcessSample(model, data, motion_0, cond_0, motion_1, cond_1)
		# gt_all_text += gt_text_concat
		# gt_all_motions.append(gt_motion_concat)
		# gt_all_lengths_0.append(gt_length_0)
		# gt_all_lengths_1.append(gt_length_1)
		# print(f"created {len(gt_all_motions) * args.batch_size} gts")

		# text_concat, motion_concat, length_0, length_1 = ProcessSample(model, data, sample_0, cond_0, sample_1, cond_1)
		text_concat, motion_concat, length_0, length_1 = ProcessSample(model, data, sample_0, cond_0, sample_1, cond_1, lenEstimator)
		for i in range(len(length_0)):
			gt_frames_per_sample[i] = list(range(0, length_0[i]))

		all_text += text_concat
		all_motions.append(motion_concat)
		all_lengths_0.append(length_0)
		all_lengths_1.append(length_1)

		print(f"created {len(all_motions) * args.batch_size} samples")

	# Visualize(args, total_num_samples, fps, gt_all_text, gt_all_motions, gt_all_lengths_0, gt_all_lengths_1, gt_frames_per_sample, gt_path)
	Visualize(args, total_num_samples, fps, all_text, all_motions, all_lengths_0, all_lengths_1, gt_frames_per_sample, out_path)
	
if __name__ == "__main__":
	model_path = ''
	data_root = ''
	t2l_checkpoints_name = ''
	text_list = ['someone performs spin the body, which is a martial art action,', 'there is a man doing knee bending and high lifting']


	# m_len_list = [120, 80]
	# m_len_list = []
	# for text in text_list:
	# 	m_len = LengthEstimator(t2l_checkpoints_name).t2l_gen(text)
	# 	print('caption: ', text)
	# 	print('motion_len: ', m_len)
	# 	m_len_list.appand(m_len)
	
	start_time = time.time()
	lenEstimator = LengthEstimator(t2l_checkpoints_name)
	args = custom_generate_args(model_path)
	args.text_prompt = text_list
	args.output_dir = ""
	args.data_dir = data_root
	args.num_repetitions = 1
	args.num_samples = 1
	Generate(args, lenEstimator)
	end_time = time.time()
	print('generation time',end_time-start_time)
