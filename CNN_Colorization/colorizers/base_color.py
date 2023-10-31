
import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__() # 初始化参数

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):# L标准化
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l): # L去标准化
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):# AB标准化
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab): # AB去标准化
		return in_ab*self.ab_norm

