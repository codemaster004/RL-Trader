import torch
from torch import nn

from lab.models.modules import DownConv, UpConv


class VAE(nn.Module):
	def __init__(self, latent_dim=32, base_channels=128, channel_multiplier=(1, 2, 3, 4), input_res=128):
		super().__init__()
		
		self.latent_dim = latent_dim
		self.out_res = input_res / 2 ** len(channel_multiplier)
		
		# Input convolution
		
		self.input_conv = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=3, stride=1, padding=1)
		
		# Encoder
		
		encoder_modules = []
		prev_channels = base_channels
		for multiplier in channel_multiplier:
			encoder_modules.append(DownConv(prev_channels, base_channels * multiplier))
			prev_channels = base_channels * multiplier
		
		self.encoder = nn.ModuleList(encoder_modules)
		
		self.extract_mu = nn.Linear(in_features=prev_channels, out_features=latent_dim)
		self.extract_logvar = nn.Linear(in_features=prev_channels, out_features=latent_dim)
		
		# Decoder
	
		decoder_modules = []
		for multiplier in channel_multiplier[:-2:-1]:
			decoder_modules.append(UpConv(prev_channels, base_channels * multiplier))
			prev_channels = base_channels * multiplier
		self.decoder = nn.ModuleList(decoder_modules)
		
		# Output convolution
		
		self.output_conv = nn.Conv2d(in_channels=prev_channels, out_channels=3, kernel_size=3, stride=1, padding=1)
	
	def forward(self, x):
		x = self.input_conv(x)
		x = self.encoder(x)
		x = torch.flatten(x, 1)
		
		mu = self.extract_mu(x)
		logvar = self.extract_logvar(x)
		z = self.reparameterize(mu, logvar)
		
		return z, mu, logvar
	
	def reparameterize(self, mu, logvar):
		z = mu + torch.exp(0.5 * logvar) * torch.randn(self.latent_dim)
		return z
	
	def decode(self, z):
		x = torch.reshape(z, (z.shape[0], self.out_res, self.out_res))
		x = self.decoder(x)
		x = self.output_conv(x)
		
		
