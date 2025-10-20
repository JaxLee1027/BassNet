from model import UNet

model = UNet(in_channels=1, out_channels=1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total trainable parameters in the model: {total_params}')
print(f'Total trainable parameters (in millions): {total_params / 1e6:.2f}M')