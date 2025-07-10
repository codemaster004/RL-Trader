import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from os.path import join as pjoin

from torchvision.transforms import transforms

from lab.models.VAE import VAE


def train(dataloader, model: VAE, criterion, optimizer, epochs, device):
	for epoch in range(epochs):
		for batch, _ in dataloader:
			batch = batch.to(device)

			z, mu, logvar = model(batch)
			reconstruction = model.decode(z)

			# Reconstruction loss
			recon_loss = criterion(reconstruction, batch)

			# KL Divergence loss
			kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
			kl_loss /= batch.size(0)

			loss = recon_loss + kl_loss

			# --- 3. Backpropagation ---
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")


def main():
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor()
	])
	dataset = ImageFolder(root=pjoin("data", "images"), transform=transform)
	loader = DataLoader(dataset, batch_size=64, shuffle=True)
	
	device = 'cuda'
	model = VAE(latent_dim=8, input_res=256, device=device).to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	
	criterion = torch.nn.MSELoss()
	
	train(loader, model, criterion, optimizer, 10, device=device)


if __name__ == '__main__':
	main()
