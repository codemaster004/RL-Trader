import torch
from PIL import Image
from torchvision.transforms import transforms

from lab.models.VAE import VAE


@torch.no_grad()
def main():
	device = 'mps'
	model = VAE(latent_dim=8, base_channels=128, input_res=256, device=device).to(device)
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor()
	])
	to_pil = transforms.ToPILImage()
	
	model.load_state_dict(torch.load('saves/vae/vae_0.pt', map_location=device))
	model.to(device)
	model.eval()

	n_rows = 1
	n_columns = 2
	resolution = 256

	bgc = Image.new("RGB", (resolution * n_columns, resolution * n_rows), color=(255, 255, 255)).convert("RGB")
	for i, img_path in enumerate(["data/images/AAPL/AAPL-0.jpg"]):
		image = Image.open(img_path).convert('RGB')
		tensor = transform(image).unsqueeze(0).to(device)
		
		z, mu, logvar = model(tensor)
		print(z.detach().cpu().numpy())
		print(mu.detach().cpu().numpy())
		print(logvar.detach().cpu().numpy())
		
		reconstruction = model.decode(z)

		bgc.paste(to_pil(tensor.squeeze(0)), (resolution * i, 0))
		bgc.paste(to_pil(reconstruction.squeeze(0)), (resolution * i, resolution))

	bgc.show()


if __name__ == "__main__":
	main()
