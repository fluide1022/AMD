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
from data_loaders.tensors import collate
from text2length import LengthEstimator

import time

def ProcessSample(model, data, sample, model_kwargs, lenEstimator = None):
	if model.data_rep == 'hml_vec':
		n_joints = 22 if sample.shape[1] == 263 else 21
		sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
		sample = recover_from_ric(sample, n_joints)
		sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

	rot2xyz_pose_rep = 'xyz'
	rot2xyz_mask = None
	sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
						   jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
						   get_rotations_back=False)

	if lenEstimator == None:
		len_res = model_kwargs['y']['lengths'].cpu().numpy()
	else:
		len_res = []
		for text in model_kwargs['y']['text']:
			motion_len = lenEstimator.t2l_gen(text)
			len_res.append(motion_len.cpu().numpy())

	return model_kwargs['y']['text'], sample.cpu().numpy(), len_res

def Visulize(args, total_num_samples, fps, all_text, all_motions, all_lengths, out_path):
	all_motions = np.concatenate(all_motions, axis=0)
	all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
	all_text = all_text[:total_num_samples]
	all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

	if not os.path.exists(out_path):
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


def Generate(args, lenEstimator):
	fixseed(args.seed)
	out_path = args.output_dir
	name = os.path.basename(os.path.dirname(args.model_path))
	niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
	max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
	fps = 12.5 if args.dataset == 'kit' else 20
	# n_frames = min(max_frames, motion_len)
	is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
	dist_util.setup_dist(args.device)
	if out_path == '':
		out_path = os.path.join(os.path.dirname(args.model_path), '1_SINGLE',
								'samples_{}_{}_seed{}'.format(name, niter, args.seed))
		gt_out_path = os.path.join(os.path.dirname(args.model_path), '1_SINGLE',
								'GT_{}_{}_seed{}'.format(name, niter, args.seed))
		if args.text_prompt != '':
			out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
		elif args.input_text != '':
			out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

	# this block must be called BEFORE the dataset is loaded
	if args.text_prompt != '':
		texts = [args.text_prompt]
		args.num_samples = 1
	elif args.input_text != '':
		assert os.path.exists(args.input_text)
		with open(args.input_text, 'r') as fr:
			texts = fr.readlines()
		texts = [s.replace('\n', '') for s in texts]
		args.num_samples = len(texts)
	elif args.action_name:
		action_text = [args.action_name]
		args.num_samples = 1
	elif args.action_file != '':
		assert os.path.exists(args.action_file)
		with open(args.action_file, 'r') as fr:
			action_text = fr.readlines()
		action_text = [s.replace('\n', '') for s in action_text]
		args.num_samples = len(action_text)

	assert args.num_samples <= args.batch_size, \
		f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
	
	# If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
	# If it doesn't, and you still want to sample more prompts, run this script with different seeds
	# (specify through the --seed flag)
	args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

	# -----data load
	print('Loading dataset...')
	data = load_dataset(args, max_frames)
	total_num_samples = args.num_samples * args.num_repetitions

	# -----model and diffusion
	print("Creating model and diffusion...")
	model, diffusion = create_model_and_diffusion(args, data)

	# ------load clip model
	print(f"Loading checkpoints from [{args.model_path}]...")
	state_dict = torch.load(args.model_path, map_location='cpu')
	load_model_wo_clip(model, state_dict)

	if args.guidance_param != 1:
		model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
	model.to(dist_util.dev())
	model.eval()  # disable random masking

	if is_using_data:
		iterator = iter(data)	#torch.Size([1, 263, 1, 196]) cond(y,{mask(torch.Size([1, 1, 1, 196])),lengths(1),text(1),tokens(1)})
		gt_motion, model_kwargs = next(iterator)
		print("model_kwargs-==>")
		print(model_kwargs)
	else:
		# collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
		is_t2m = any([args.input_text, args.text_prompt])
		print("is_t2m==>",is_t2m)
		if is_t2m:
			# t2m
			collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
		else:
			# a2m
			action = data.dataset.action_name_to_action(action_text)
			collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
							arg, one_action, one_action_text in zip(collate_args, action, action_text)]
		_, model_kwargs = collate(collate_args)

	all_motions = []
	all_lengths = []
	all_text = []
	# gt_all_motions = []
	# gt_all_lengths = []
	# gt_all_text = []
	for rep_i in range(args.num_repetitions):
		print(f'### Sampling [repetitions #{rep_i}]')

		# add CFG scale to batch
		if args.guidance_param != 1:
			model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
		print(args.guidance_param)
		print(model_kwargs)
		# diffusion sampling
		sample_fn = diffusion.p_sample_loop

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
		# sample = torch.randn((args.batch_size, model.njoints, model.nfeats, max_frames), device=dist_util.dev())

		# text_gt, motion_gt, length_gt = ProcessSample(model, data, gt_motion, model_kwargs)
		# text_gt, motion_gt, length_gt = ProcessSample(model, data, gt_motion, model_kwargs, lenEstimator)
		# gt_all_text += text_gt
		# gt_all_motions.append(motion_gt)
		# gt_all_lengths.append(length_gt)
		# model，sample，lenEstimator 
		text_res, motion_res, length_res = ProcessSample(model, data, sample, model_kwargs)
		text_res, motion_res, length_res = ProcessSample(model, data, sample, model_kwargs, lenEstimator)
		all_text += text_res
		all_motions.append(motion_res)
		all_lengths.append(length_res)

		print(f"created {len(all_motions) * args.batch_size} samples")

	# Visulize(args, total_num_samples, fps, gt_all_text, gt_all_motions, gt_all_lengths, gt_out_path)
	Visulize(args, total_num_samples, fps, all_text, all_motions, all_lengths, out_path)

def load_dataset(args, max_frames):
	# data = get_dataset_loader(name=args.dataset,
	# 						  batch_size=args.batch_size,
	# 						  data_root=args.data_dir,
	# 						  num_frames=max_frames,
	# 						  split='train',
	# 						  hml_mode='text_only')
	# data.dataset.t2m_dataset.fixed_length = n_frames
	data = get_dataset_loader(name=args.dataset,
							  batch_size=args.batch_size,
							  # batch_size=1,
							  data_root=args.data_dir,
							  num_frames=max_frames,
							  split='val',
							  hml_mode='train')
	# data.dataset.t2m_dataset.fixed_length = n_frames
	return data


if __name__ == "__main__":

	model_path = ''

	# single motion
	data_root = ''

	t2l_checkpoints_name = ''

	# text = 'someone is taking 3 steps backwards'
	# text = 'someone is doing qixing boxing, then he performs double sanda'

	start_time = time.time()

	# motion duration estimation
	lenEstimator = LengthEstimator(t2l_checkpoints_name)
	# motion_len = LengthEstimator(t2l_checkpoints_name).t2l_gen(text)
	# print('motion_len: ', motion_len)
	# motion_len = 110
	args = custom_generate_args(model_path)
	# args.text_prompt = "the action he is doing is called swing leg"
	args.output_dir = ""
	args.data_dir = data_root
	args.num_repetitions = 3
	args.num_samples = 64

	# motion generation
	Generate(args, lenEstimator)
	end_time = time.time()
	print('generation time:',end_time-start_time)
