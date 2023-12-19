import torch
import torch.nn as nn
import config

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X N X C
        key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C x N
        energy = torch.bmm(query, key)  # Batch matrix multiplication, B X N X N
        attention_map = torch.softmax(energy, dim=-1)  # Softmax over the last dimension to create attention map
        value = self.value_conv(x).view(batch_size, -1, width * height)  # B X C X N
        out = torch.bmm(value, attention_map.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x  # Apply attention gamma and add input
        return out, attention_map

class ConvVAE(nn.Module):
    def __init__(self, latent_space_dim):
        super(ConvVAE, self).__init__()
        
        self.latent_space_dim = latent_space_dim
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 3x224x224 -> 32x112x112
        self.att1 = SelfAttention(in_dim=32) # 32x112x112 -> 32x112x112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x112x112 -> 64x56x56
        self.att2 = SelfAttention(in_dim=64) # 64x56x56 -> 64x56x56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 64x56x56 -> 128x28x28
        self.fc_mu = nn.Linear(128 * 28 * 28, latent_space_dim) # 128x28x28 -> 256
        self.fc_logvar = nn.Linear(128 * 28 * 28, latent_space_dim) # 128x28x28 -> 256
        
        # Decoder
        self.fc_decode = nn.Linear(latent_space_dim, 128 * 28 * 28) # 256 -> 128x28x28
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # 128x28x28 -> 64x56x56
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 64x56x56 -> 32x112x112
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1) # 32x112x112 -> 3x224x224
        
        self.relu = nn.ReLU() # ReLU activation function
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x, attention_map1 = self.att1(x)
        x = self.relu(self.conv2(x))
        x, attention_map2 = self.att2(x)
        x = self.relu(self.conv3(x))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the convolutional layer output
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, attention_map1, attention_map2
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.relu(self.fc_decode(z))
        x = x.view(-1, 128, 28, 28)  # Unflatten to prepare for transposed convolutions
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x

    def forward(self, x):
        mu, logvar, _, _ = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
if __name__ == '__main__':
    # Now, let's instantiate and view the model to ensure it's structured correctly.
    conv_vae = ConvVAE(latent_space_dim=256)
    print(conv_vae)

