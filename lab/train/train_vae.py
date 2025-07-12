import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from os import makedirs
from os.path import join as pjoin
import logging as log

from lab.models.VAE import VAE

log.basicConfig(level=log.INFO)


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

		log.info(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
	
	return model


def main():
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor()
	])
	dataset = ImageFolder(root=pjoin("data", "images"), transform=transform)
	loader = DataLoader(dataset, batch_size=8, shuffle=True)
	
	makedirs(pjoin("models", "vae"), exist_ok=True)
	
	experiments = [
		(24, 256),
		(16, 256),
		(8, 256),
		(4, 256),
		(24, 128),
		(16, 128),
		(8, 128),
		(4, 128),
	]
	
	device = 'mps'
	for i, e in enumerate(experiments):
		model = VAE(latent_dim=e[0], base_channels=e[1], input_res=256, device=device).to(device)
		
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		criterion = torch.nn.MSELoss()
		
		log.info(f"Experiment {str(i)}: ({str(e[0])}, {str(e[1])})")
		model = train(loader, model, criterion, optimizer, 50, device=device)
		torch.save(model.state_dict(), pjoin("models", "vae", f"vae_{str(i)}.pt"))


if __name__ == '__main__':
	main()
