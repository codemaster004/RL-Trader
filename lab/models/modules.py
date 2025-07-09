from torch import nn


class DownConv(nn.Module):
	def __init__(self, in_channels, out_channels, residual_block=False):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
		)
	
	def forward(self, x):
		return self.conv(x)


class UpConv(nn.Module):
	def __init__(self, in_channels, out_channels, residual_block=False):
		super().__init__()
		self.conv = nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
		)
	
	def forward(self, x):
		return self.conv(x)
