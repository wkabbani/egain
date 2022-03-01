from argparse import ArgumentParser
import os
import json
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from criteria.lpips.lpips import LPIPS
from datasets.gt_res_dataset import GTResDataset
from criteria.image_similarity import calculate_vifp, calculate_msssim, calculate_sam, calculate_rase, calculate_scc, calculate_ergas, calculate_uqi, calculate_rmse, calculate_mse, calculate_fsim, calculate_issm, calculate_sre, calculate_uiq, calculate_ssim, calculate_psnr

def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--mode', type=str, default='lpips')
	parser.add_argument('--data_path', type=str, default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):

	transform = transforms.Compose([transforms.Resize((256, 256)),
									transforms.ToTensor(),
									transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	print('Loading dataset')
	dataset = GTResDataset(root_path=args.data_path,
	                       gt_dir=args.gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
	                        batch_size=args.batch_size,
	                        shuffle=False,
	                        num_workers=int(args.workers),
	                        drop_last=True)

	if args.mode == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif args.mode == 'l2':
		loss_func = torch.nn.MSELoss()
	elif args.mode == 'ssim':
		loss_func = calculate_ssim
	elif args.mode == 'psnr':
		loss_func = calculate_psnr
	elif args.mode == 'vifp':
		loss_func = calculate_vifp
	elif args.mode == 'msssim':
		loss_func = calculate_msssim
	elif args.mode == 'sam':
		loss_func = calculate_sam
	elif args.mode == 'rase':
		loss_func = calculate_rase
	elif args.mode == 'scc':
		loss_func = calculate_scc
	elif args.mode == 'ergas':
		loss_func = calculate_ergas
	elif args.mode == 'uqi':
		loss_func = calculate_uqi
	elif args.mode == 'rmse':
		loss_func = calculate_rmse
	elif args.mode == 'fsim':
		loss_func = calculate_fsim
	elif args.mode == 'issm':
		loss_func = calculate_issm
	elif args.mode == 'sre':
		loss_func = calculate_sre
	elif args.mode == 'uiq':
		loss_func = calculate_uiq
	else:
		raise Exception('Not a valid mode!')

	if args.mode == 'l2' or args.mode == 'lpips':
		loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		for i in range(args.batch_size):
			if args.mode == 'l2' or args.mode == 'lpips':
				loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			else:
				assert args.batch_size == 1
				loss = float(loss_func(result_batch[i:i+1][0], gt_batch[i:i+1][0]))
			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			scores_dict[os.path.basename(im_path)] = loss
			global_i += 1

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = 'Average loss is {:.2f}+-{:.2f}'.format(mean, std)
	print('Finished with ', args.data_path)
	print(result_str)

	out_path = os.path.join(os.path.dirname(args.data_path), 'inference_metrics')
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	with open(os.path.join(out_path, 'stat_{}.txt'.format(args.mode)), 'w') as f:
		f.write(result_str)
	with open(os.path.join(out_path, 'scores_{}.json'.format(args.mode)), 'w') as f:
		json.dump(scores_dict, f)


if __name__ == '__main__':
	args = parse_args()
	run(args)
