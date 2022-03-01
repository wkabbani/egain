import matplotlib
matplotlib.use('Agg')
import math
import pprint
import numpy as np
from PIL import Image, ImageChops

import torch
from torch import nn
import torchvision.transforms as transforms
from models.encoders import psp_encoders
from configs.paths_config import model_paths
from utils.common import tensor2im
from models.stylegan2.model import Generator as StyleGenerator
from models.stylegan2.swagan import Generator as SwaganGenerator

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()
		self.set_opts(opts)
		pprint.pprint(self.opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2

		# Define architecture
		self.encoder = self.set_encoder()
		self.decoder = self.set_decoder()
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		self.residue =  psp_encoders.ResidualEncoder()
		self.fusion = psp_encoders.SimpleFusionModule()

		# Load weights if needed
		self.load_encoder_weights()
		self.load_decoder_weights()
		self.load_residue_weights()
		self.load_fusion_weights()


	def set_decoder(self):
		if self.opts.decoder_type == 'StyleGanGenerator':
			decoder = StyleGenerator(self.opts.output_size, 512, 8)
		elif self.opts.decoder_type == 'SwaganGenerator':
			decoder = SwaganGenerator(self.opts.output_size, 512, 8)
		else:
			raise Exception('{} is not a valid decoder'.format(self.opts.decoder_type))
		return decoder

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_encoder_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
		else:
			print('Loading encoders weights from irse50!')
			encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			if self.opts.label_nc != 0:
				encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			self.encoder.load_state_dict(encoder_ckpt, strict=False)

	def load_decoder_weights(self):
		if self.opts.decoder_type == 'StyleGanGenerator':
			print('loading stylegan encoder')
			decoder_ckpt = torch.load(self.opts.stylegan_weights)
		elif self.opts.decoder_type == 'SwaganGenerator':
			print('loading swagan encoder')
			decoder_ckpt = torch.load(self.opts.swagan_weights)

		self.decoder.load_state_dict(decoder_ckpt['g_ema'], strict=False)
		self.decoder = self.decoder.to(self.opts.device)
		alternative_avg = self.decoder.mean_latent(int(1e5))[0].detach()

		if self.opts.learn_in_w:
			self.__load_latent_avg(decoder_ckpt, alternative_avg, repeat=1)
		else:
			self.__load_latent_avg(decoder_ckpt, alternative_avg, repeat=self.opts.n_styles)

	def load_residue_weights(self):
		if self.opts.egain_checkpoint_path is not None:
			ckpt = torch.load(self.opts.egain_checkpoint_path, map_location='cpu')
			self.residue.load_state_dict(get_keys(ckpt, 'residue'), strict=True)

	def load_fusion_weights(self):
		if self.opts.egain_checkpoint_path is not None:
			ckpt = torch.load(self.opts.egain_checkpoint_path, map_location='cpu')
			self.fusion.load_state_dict(get_keys(ckpt, 'fusion'), strict=True)


	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes], None,
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		# external fusion >>
		diff_results = self.img_difference(x, images)
		diff_codes = self.encoder(diff_results)
		codes = self.fusion(codes, diff_codes)
		# external fusion <<

		# imgs_ = torch.nn.functional.interpolate(torch.clamp(images, -1., 1.), size=(256,256) , mode='bilinear')
		# res_gt = (x - imgs_ ).detach()
		# res = res_gt.to(self.opts.device)
		# conditions = self.residue(res)
		# if conditions is not None:

		images, result_latent = self.decoder([codes], None,
											input_is_latent=input_is_latent,
											randomize_noise=randomize_noise,
											return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images

	def img_difference(self, original_images, generated_images):
		batch_size = original_images.shape[0]
		diff_results = []
		for i in range(batch_size):
			org_image = tensor2im(original_images[i])
			gen_image = tensor2im(generated_images[i])

			org_image = Image.fromarray(np.array(org_image))
			gen_image = Image.fromarray(np.array(gen_image))

			gen_image = gen_image.resize((256, 256))

			diff = ImageChops.difference(org_image, gen_image)

			img_transforms = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

			diff = img_transforms(diff)
			diff_results.append(diff)

		diff_results = torch.stack(diff_results)
		diff_results = diff_results.to("cuda").float()

		return diff_results

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, alternative_avg, repeat=None):
		if 'latent_avg' in ckpt:
			print('Using ckpt avg latent')
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
		else:
			print('Using alternative avg latent')
			self.latent_avg = alternative_avg.to(self.opts.device)

		if repeat is not None:
			self.latent_avg = self.latent_avg.repeat(repeat, 1)
