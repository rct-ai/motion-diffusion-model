# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/12/30 3:13 下午
==================================="""
# a flask app to return a mp4 file
import os
import shutil
import time
from os.path import join as pjoin

import numpy as np
import torch
from flask import Flask, request, jsonify, Response

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from model.cfg_sampler import ClassifierFreeSampleModel
from model.mdm import MDM
from sample.generate import construct_template_variables
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import get_model_args_simple, create_gaussian_diffusion, load_model_wo_clip
from utils.parser_util import generate_args

app = Flask(__name__)


# @app.route('/t2video', methods=['POST'])
def t2video():
    # params = request.get_json()
    # text = params.get('text', None)
    text = 'the person walked forward and is picking up his toolbox.'
    if text is None:
        return jsonify({'status': 'error', 'msg': 'text is None'}), 400
    texts = [text]
    args.num_samples = 1
    args.batch_size = args.num_samples

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples

    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)
    all_motions = []
    all_lengths = []
    all_text = []

    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop

    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    if model.data_rep == 'hml_vec':  # true
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = sample.cpu().permute(0, 2, 3, 1) * std * mean
        sample = recover_from_ric(sample, n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size,
                                                                                            n_frames).bool()
    sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                           jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                           get_rotations_back=False)

    if args.unconstrained:  # false
        all_text += ['unconstrained'] * args.num_samples
    else:
        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]

    all_motions.append(sample.cpu().numpy())
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

    print(f"created {len(all_motions) * args.batch_size} samples")

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

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []

        caption = all_text[0]
        length = all_lengths[0]
        motion = all_motions[0].transpose(2, 0, 1)[:length]
        save_file = sample_file_template.format(sample_i, 0)  #  'sample00_rep00.mp4'
        print(sample_print_template.format(caption, sample_i, 0, save_file))
        animation_save_path = os.path.join(out_path, save_file)  # './save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10/sample00_rep00.mp4'
        plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
        # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
        rep_files.append(animation_save_path)

    abs_path = os.path.abspath(out_path)
    # caculate the run time of the program
    start_time = time.time()

    render_mesh_cmd = f'/home/rct/miniconda3/envs/mdm/bin/python3 -m visualize.render_mesh --input_path {os.path.join(os.path.abspath(out_path), sample_file_template.format(0,0))} '
    os.system(render_mesh_cmd)
    end_time = time.time()
    print("The run time of the render mesh is: ", (end_time - start_time), "s")
    print(f'[Done] Results are at [{abs_path}]')



if __name__ == '__main__':
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))  # ./save/humanml_trans_enc_512/model000200000.pt
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60  # 最大帧数 196
    fps = 12.5 if args.dataset == 'kit' else 20  # config fps 20
    n_frames = min(max_frames, int(args.motion_length * fps))  # 120

    dist_util.setup_dist(args.device)
    # 创建输出文件夹
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    args_ = get_model_args_simple(args, dataset='humanml')
    model = MDM(**args_)
    diffusion = create_gaussian_diffusion(args)

    # 加载模型
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    # load
    mean = np.load(pjoin('/data/cll/motion-diffusion-model/dataset/HumanML3D', 'Mean.npy'))
    std = np.load(pjoin('/data/cll/motion-diffusion-model/dataset/HumanML3D', 'Std.npy'))
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    total_num_samples = args.num_samples * args.num_repetitions
    args.train = False

    t2video()