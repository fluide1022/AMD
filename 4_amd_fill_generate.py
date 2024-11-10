from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import custom_edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from text2length import LengthEstimator
from copy import deepcopy

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def ProcessSmp(model, data, sample, model_kwargs):
	# Recover XYZ *positions* from HumanML3D vector representation
	if model.data_rep == 'hml_vec':
		n_joints = 22 if sample.shape[1] == 263 else 21
		sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
		sample = recover_from_ric(sample, n_joints)
		sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

	rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
	rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
	sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
						   jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
						   get_rotations_back=False)
	text_smp = deepcopy(model_kwargs['y']['text'])
	motion_smp = deepcopy(sample.cpu().numpy())
	length_smp = deepcopy(model_kwargs['y']['lengths'].cpu().numpy())
	return text_smp, motion_smp, length_smp

def ProcessGT(model, data, sample_0, cond_0, sample_1, cond_1):
	text_concat = deepcopy(cond_0['y']['text_curr'])
	for i in range(len(text_concat)):
		text_concat[i] = text_concat[i].replace('.', ', ') + cond_1['y']['text_curr'][i]

	length_0 = deepcopy(cond_0['y']['lengths_curr'].cpu().numpy())
	length_1 = deepcopy(cond_1['y']['lengths_curr'].cpu().numpy())

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

def VisualizeSmp(args, total_num_samples, fps, all_text, all_motions, all_lengths, gt_frames_per_sample, out_path):

	all_motions = np.concatenate(all_motions, axis=0)
	all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
	all_text = all_text[:total_num_samples]
	all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

	if os.path.exists(out_path):
		shutil.rmtree(out_path)
	os.makedirs(out_path)

	npy_path = os.path.join(out_path, 'results.npy')
	print(f"saving results file to [{npy_path}]")
	np.save(npy_path,
			{'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
			 'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
	with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
		fw.write('\n'.join(all_text))
	with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
		fw.write('\n'.join([str(l) for l in all_lengths]))

	print(f"saving visualizations to [{out_path}]...")
	skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

	sample_files = []
	num_samples_in_out_file = 7
	for sample_i in range(args.num_samples):
		rep_files = []
		for rep_i in range(args.num_repetitions):
			# caption = all_text[rep_i*args.batch_size + sample_i]
			caption = ""
			length = all_lengths[rep_i*args.batch_size + sample_i]
			motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
			save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
			animation_save_path = os.path.join(out_path, save_file)
			print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
			# plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
			plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
						   dataset=args.dataset, fps=fps, vis_mode=args.edit_mode,
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


def VisualizeGT(args, total_num_samples, fps, all_text, all_motions, all_lengths_0, all_lengths_1, out_path):

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
			plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
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

def Generate(args, motion_len_list):
	fixseed(args.seed)
	out_path = args.output_dir
	name = os.path.basename(os.path.dirname(args.model_path))
	niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
	max_frames = 196 if args.dataset in ['kit', 'humanml', 'autoreg'] else 60
	fps = 12.5 if args.dataset == 'kit' else 20

	np_len_list = np.expand_dims(np.array(motion_len_list), axis = 0)
	np_max_frames = np.expand_dims(np.full(len(motion_len_list), max_frames), axis = 0)
	n_frames_arr = np.min(np.concatenate((np_len_list, np_max_frames)), axis = 0)
	print(n_frames_arr)

	dist_util.setup_dist(args.device)
	if out_path == '':
		out_path = os.path.join(os.path.dirname(args.model_path), '4_FILL',
								'samples_{}_{}_seed{}'.format(name, niter, args.seed))
		gt_path = os.path.join(os.path.dirname(args.model_path), '4_FILL',
								'GT_{}_{}_seed{}'.format(name, niter, args.seed))
	
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

	data = get_dataset_loader(name='autoreg',
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

	model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
	model.to(dist_util.dev())
	model.eval()  # disable random masking

	iterator = iter(data)
	motion_0, cond_0, motion_1, cond_1 = next(iterator)
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
	print(model_kwargs['y']['text'])

	model_kwargs['y']['lengths'] =  deepcopy(cond_0['y']['lengths_curr'])

	# model_kwargs['y']['mask'] = lengths_to_mask(model_kwargs['y']['lengths'], max_frames*2).unsqueeze(1).unsqueeze(1)
	# print(model_kwargs['y']['mask'] .shape)

	if args.text_condition == '':
		args.guidance_param = 0.  # Force unconditioned generation

	gt_frames_per_sample = {}
	for i in range(cond_0['y']['lengths_curr'].shape[0]):
		gt_frames_per_sample[i] = list(range(0, cond_0['y']['lengths_curr'][i]))

	all_motions = []
	all_lengths = []
	all_text = []
	gt_all_motions = []
	gt_all_lengths_0 = []
	gt_all_lengths_1 = []
	gt_all_text = []
	for rep_i in range(args.num_repetitions):
		print(f'### Sampling [repetitions #{rep_i}]')

		# add CFG scale to batch
		if args.guidance_param != 1:
			model_kwargs_0['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
			model_kwargs_1['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

		sample_fn = diffusion.p_sample_loop

		sample_0 = sample_fn(
			model,
			# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
			(args.batch_size, model.njoints, model.nfeats, max_frames),
			clip_denoised=False,
			model_kwargs=model_kwargs_0,
			skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
			init_image=None,
			progress=True,
			dump_steps=None,
			noise=None,
			const_noise=False,
		)

		sample_1 = sample_fn(
			model,
			# (args.batch_size, model.njoints, model.nfeats, n_frames_arr[0]),
			(args.batch_size, model.njoints, model.nfeats, max_frames),
			clip_denoised=False,
			model_kwargs=model_kwargs_1,
			skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
			init_image=None,
			progress=True,
			dump_steps=None,
			noise=None,
			const_noise=False,
		)

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
		print('input_motions: ',input_motions.shape)

		# add inpainting mask according to args
		model_kwargs['y']['mask'] = lengths_to_mask(model_kwargs['y']['lengths'], max_frames).unsqueeze(1).unsqueeze(1)
		model_kwargs['y']['inpainted_motion'] = input_motions
		model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
															   device=input_motions.device)  # True means use gt motion
		for i in range(sample_0.shape[0]):
			clip_start = max(0, length_0[i]-max_frames//2)
			start_idx, end_idx = int(args.prefix_end * length_0[i]), int(length_0[i] + args.suffix_start * length_1[i])
			# gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames*2))
			model_kwargs['y']['inpainting_mask'][i, :, :,
			start_idx-clip_start: end_idx-clip_start] = False  # do inpainting in those frames
		model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

		sample = sample_fn(
			model,
			(args.batch_size, model.njoints, model.nfeats, max_frames),
			clip_denoised=False,
			model_kwargs=model_kwargs,
			skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
			init_image=None,
			progress=True,
			dump_steps=None,
			noise=None,
			const_noise=False,
		)

		# gt_text_concat, gt_motion_concat, gt_length_0, gt_length_1 = ProcessGT(model, data, motion_0, cond_0, motion_1, cond_1)
		# gt_all_text += gt_text_concat
		# gt_all_motions.append(gt_motion_concat)
		# gt_all_lengths_0.append(gt_length_0)
		# gt_all_lengths_1.append(gt_length_1)
		# print(f"created {len(gt_all_motions) * args.batch_size} gts")

		sample_concat = []
		for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
			clip_start = max(0, length_0[i]-max_frames//2)
			s_tmp_0 = sample_0[i][:,:,0:clip_start]
			s_tmp_1 = sample_1[i][:,:, min(max_frames-length_0[i]+clip_start, length_1[i]):length_1[i]]
			model_kwargs['y']['lengths'][i] = s_tmp_0.shape[2] + length + s_tmp_1.shape[2]
			s_tmp_concat = np.concatenate((s_tmp_0, sample[i][:,:,:length].cpu().numpy(), s_tmp_1, np.zeros((s_tmp_0.shape[0], s_tmp_0.shape[1], max_frames*2 - model_kwargs['y']['lengths'][i]))), axis = 2)
			print(s_tmp_concat.shape)
			sample_concat.append(s_tmp_concat)
		sample_concat = torch.tensor(np.array(sample_concat)).float()
		sample_concat = sample_concat.to(dist_util.dev())
		text_smp, motion_smp, length_smp = ProcessSmp(model, data, sample_concat, model_kwargs)
		all_text += text_smp
		all_motions.append(motion_smp)
		all_lengths.append(length_smp)
		print(f"created {len(all_motions) * args.batch_size} samples")

	# VisualizeGT(args, total_num_samples, fps, gt_all_text, gt_all_motions, gt_all_lengths_0, gt_all_lengths_1, gt_path)
	VisualizeSmp(args, total_num_samples, fps, all_text, all_motions, all_lengths, gt_frames_per_sample, out_path)
	
if __name__ == "__main__":
	model_path = ''
	data_root = ''
	m_len_list = [120, 80]
	# m_len_list = []
	# for text in text_list:
	# 	m_len = LengthEstimator(t2l_checkpoints_name).t2l_gen(text)
	# 	print('caption: ', text)
	# 	print('motion_len: ', m_len)
	# 	m_len_list.appand(m_len)
		
	args = custom_edit_args(model_path)
	# args.text_prompt = text
	args.data_dir = data_root
	args.num_repetitions = 1
	args.prefix_end = 0.90
	args.suffix_start = 0.10
	Generate(args, m_len_list)
