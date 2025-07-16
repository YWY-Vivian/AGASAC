import torch
import torch.nn as nn
import torch.nn.functional as F

import random
class GAN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GAN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out

class AGANet(nn.Module): 
	
	def __init__(self, input_dim, output_dim, blocks=5, batch_norm=True, separate_weights=True):
		
		super(AGANet, self).__init__()
		
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.p_in = nn.Conv2d(self.input_dim, 128, 1, 1, 0)

		self.res_blocks = []
		self.separate_probs = separate_weights
		self.batch_norm = batch_norm
		
		for i in range(0, blocks):  
			if batch_norm:
				self.res_blocks.append((
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),
				))
			else:
				self.res_blocks.append((
                    nn.Conv2d(128, 128, 1, 1, 0),
                    nn.Conv2d(128, 128, 1, 1, 0),
                ))

		for i, r in enumerate(self.res_blocks):
			super(AGANet, self).add_module(str(i) + 's0', r[0])
			super(AGANet, self).add_module(str(i) + 's1', r[1])
			if batch_norm:
				super(AGANet, self).add_module(str(i) + 's2', r[2])
				super(AGANet, self).add_module(str(i) + 's3', r[3])

		self.p_out =  nn.Conv2d(128, output_dim, 1, 1, 0)
		if self.separate_probs:
			self.p_out2 = nn.Conv2d(128, output_dim, 1, 1, 0)

		self.gan = GAN_Block(128)

	def forward(self, inputs):
		
		batch_size = inputs.size(0)
		inputs_ = torch.transpose(inputs, 1, 2).unsqueeze(-1)

		x = inputs_[:, 0:self.input_dim]
		w = self.p_in(x)
		x = F.relu(w)
		
		for r in self.res_blocks:
			res = x
			if self.batch_norm:
				x = F.relu(r[1](F.instance_norm(r[0](x))))
				x = F.relu(r[3](F.instance_norm(r[2](x))))
			else:
				x = F.relu(F.instance_norm(r[0](x)))
				x = F.relu(F.instance_norm(r[1](x)))
			x = x + res

		out = x
		weights = self.p_out(x).view(batch_size, -1)
		out_g = self.gan(out, weights)
		out = out_g + out

		log_probs = F.logsigmoid(self.p_out(out))
		log_ng = torch.transpose(log_probs, 1, 2)

		normalizer = torch.logsumexp(log_ng, dim=1, keepdim=True)
		log_ng = log_ng - normalizer

		if self.separate_probs: 
			log_ng2 = F.logsigmoid(self.p_out2(out))
			log_ng2 = torch.transpose(log_ng2, 1, 2)
			normalizer = torch.logsumexp(log_ng2, dim=-2, keepdim=True)
			log_ng2 = log_ng2 - normalizer
		else:
			log_ng2 = log_ng

		return log_ng, log_ng2
