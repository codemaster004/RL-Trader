import torch
from torch import nn

from lab.models.modules import DownConv, UpConv


class VAE(nn.Module):
	def __init__(self, latent_dim=32, base_channels=128, channel_multipliers=(1, 2, 3, 4), input_res=128, device="cpu"):
		super().__init__()
		
		self.latent_dim = latent_dim
		self.base_channels = base_channels
		self.multipliers = channel_multipliers
		self.out_res = int(input_res / 2 ** len(channel_multipliers))
		self.device = device
		
		# Input convolution
		
		self.input_conv = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=3, stride=1, padding=1)
		
		# Encoder
		
		encoder_modules = []
		prev_channels = base_channels
		for multiplier in channel_multipliers:
			encoder_modules.append(DownConv(prev_channels, base_channels * multiplier))
			prev_channels = base_channels * multiplier
		
		self.encoder = nn.ModuleList(encoder_modules)
		
		self.extract_mu = nn.Linear(in_features=prev_channels * self.out_res * self.out_res, out_features=latent_dim)
		self.extract_logvar = nn.Linear(in_features=prev_channels * self.out_res * self.out_res, out_features=latent_dim)
		
		# Decoder
		
		self.projection = nn.Linear(in_features=self.latent_dim, out_features=prev_channels * self.out_res * self.out_res)
	
		decoder_modules = []
		for multiplier in channel_multipliers[-2::-1]:
			decoder_modules.append(UpConv(prev_channels, base_channels * multiplier))
			prev_channels = base_channels * multiplier
		decoder_modules.append(UpConv(prev_channels, base_channels))
		self.decoder = nn.ModuleList(decoder_modules)
		
		# Output convolution
		
		self.output_conv = nn.Conv2d(in_channels=base_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
	
	def forward(self, x):
		x = self.input_conv(x)
		for module in self.encoder:
			x = module(x)
		x = torch.flatten(x, 1)
		
		mu = self.extract_mu(x)
		logvar = self.extract_logvar(x)
		z = self.reparameterize(mu, logvar)
		
		return z, mu, logvar
	
	def reparameterize(self, mu, logvar):
		z = mu + torch.exp(0.5 * logvar) * torch.randn(self.latent_dim, device=self.device, requires_grad=False)
		return z
	
	def decode(self, z):
		x = self.projection(z)
		x = torch.reshape(x, (z.shape[0], self.base_channels * self.multipliers[-1], self.out_res, self.out_res))
		for module in self.decoder:
			x = module(x)
		x = self.output_conv(x)
		
		return x
