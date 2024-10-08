#implementation from : https://github.com/kylemin/S3D

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class S3D(nn.Module):
    def __init__(self):
        super(S3D, self).__init__()

        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0)),
            Mixed_5b(),
            Mixed_5c(),
        )

    def forward(self, x):
        y = self.base(x)
        y = F.avg_pool3d(y, (2, y.size(3), y.size(4)), stride=1)

        y = y.view(y.size(0), y.size(1), y.size(2))
        visual_features = torch.mean(y, 2)

        return visual_features

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #print(x.shape)
        return x

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        #print(x.shape)
        return x

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class SA_Block(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(SA_Block, self).__init__()

    self.multihead_attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first = True)
    self.norm1 = nn.LayerNorm(embed_dim)

  def forward(self, x):

    attn_output = self.norm1(self.multihead_attn(x, x, x, need_weights=False)[0])
    x = x + attn_output

    return x

class SA_Linear_Block(nn.Module):
  def __init__(self, embed_dim, num_heads, linear_dims_in, linear_dims_out):
    super(SA_Linear_Block, self).__init__()

    self.multihead_attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, batch_first = True)
    self.linear = nn.Sequential(nn.Linear(linear_dims_in, linear_dims_out), nn.ReLU(inplace = True))
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(linear_dims_out)

  def forward(self, x):

    attn_output = self.norm1(self.multihead_attn(x, x, x, need_weights=False)[0])
    x = x + attn_output
    linear_output = self.linear(x)

    return self.norm2(linear_output)

class AudioEncoder(nn.Module):
  def __init__(self, device  = "cpu"):
    super(AudioEncoder, self).__init__()

    self.device = device
    print("AudioEncoder using device: ", device)
    self.batch_norm = nn.BatchNorm1d(13)
    self.input_linear = nn.Linear(2*4*13, 128)
    self.position_embedding_table = nn.Embedding(25, 4*13)

    self.audio_stage1 = nn.Sequential(SA_Block(128,4),SA_Linear_Block(128,4,128,256))
    self.audio_stage2 = nn.Sequential(SA_Block(256,8),SA_Linear_Block(256,8,256,512))
    self.audio_stage3 = nn.Sequential(SA_Block(512,8),SA_Linear_Block(512,8,512,512))

  def forward(self,x):
    
    norm_features = self.batch_norm(x).permute(0,2,1).contiguous().view(-1,25,4*13)
    
    pos_embed = self.position_embedding_table(torch.arange(25, device = self.device)).unsqueeze(0)
    input_sequence = torch.cat((norm_features,pos_embed.repeat(x.shape[0],1,1)), dim = 2)

    input_sequence = self.input_linear(input_sequence)
    audio_embeddings_1 = self.audio_stage1(input_sequence)
    audio_embeddings_2 = self.audio_stage2(audio_embeddings_1)
    audio_embeddings_3 = self.audio_stage3(audio_embeddings_2)

    return audio_embeddings_1,audio_embeddings_2,audio_embeddings_3

class ASD_model2(nn.Module):

  def __init__(self, num_heads = 8, load_state = False, use_cuda = True):
      super(ASD_model2, self).__init__()
      
      if use_cuda:
        device = torch.device("cuda:0")
      else:
        device = torch.device("cpu")  
      
      self.visual_extractor = S3D()
      if not load_state:
        self.load_pretrained_s3d(use_cuda)

      self.s3d_stage1 = self.visual_extractor.base[:8]
      self.s3d_stage2 = self.visual_extractor.base[8:14]
      self.s3d_stage3 = self.visual_extractor.base[14:]

      self.audio_extractor = AudioEncoder(device = device)

      self.cross_attn1 = nn.MultiheadAttention(480, num_heads = num_heads, dropout = 0.1, kdim = 256, vdim = 256, batch_first = True)
      self.ln1 = nn.LayerNorm(480)
      self.cross_attn2 = nn.MultiheadAttention(832, num_heads = num_heads, dropout = 0.1, kdim = 512, vdim = 512, batch_first = True)
      self.ln2 = nn.LayerNorm(832)
      self.cross_attn3 = nn.MultiheadAttention(1024, num_heads = num_heads, dropout = 0.1, kdim = 512, vdim = 512, batch_first = True)
      self.ln3 = nn.LayerNorm(1024)

      self.classifier_head = nn.Sequential(
        nn.Linear(in_features = 1024, out_features = 2048),
        nn.ReLU6(),
        nn.Linear(2048, 512),
        nn.ReLU6(),
        nn.Linear(512,1)
      )

  def forward(self,audio_input,visual_input):

      audio_features = self.audio_extractor(audio_input)

      audio_embeddings_1 = audio_features[0]
      audio_embeddings_2 = audio_features[1]
      audio_embeddings_3 = audio_features[2]

      visual_features = self.s3d_stage1(visual_input)

      #cross attention 1
      va_stage1 = visual_features.view(-1,480,7*8*8).permute(0,2,1).contiguous()

      cross_attn1 = self.ln1(self.cross_attn1(va_stage1, audio_embeddings_1, audio_embeddings_1, need_weights = False)[0]).permute(0,2,1).contiguous().view(-1,480,7,8,8)
      visual_features = visual_features + cross_attn1
      
      #---

      visual_features = self.s3d_stage2(visual_features)

      #cross attention 2
      va_stage2 = visual_features.view(-1,832,3*4*4).permute(0,2,1).contiguous()

      cross_attn2 = self.ln2(self.cross_attn2(va_stage2, audio_embeddings_2, audio_embeddings_2, need_weights = False)[0]).permute(0,2,1).contiguous().view(-1,832,3,4,4)
      visual_features = visual_features + cross_attn2

      #---

      visual_features = self.s3d_stage3(visual_features)

      #cross attention 3

      va_stage3 = visual_features.view(-1,1024,3*4*4).permute(0,2,1).contiguous()

      cross_attn3 = self.ln3(self.cross_attn3(va_stage3, audio_embeddings_3, audio_embeddings_3, need_weights = False)[0]).permute(0,2,1).contiguous().view(-1,1024,3,4,4)
      visual_features = visual_features + cross_attn3
      
      #---

      visual_features = F.avg_pool3d(visual_features, (2, visual_features.size(3), visual_features.size(4)), stride=1)

      visual_features = visual_features.view(visual_features.size(0), visual_features.size(1), visual_features.size(2))
      visual_features = torch.mean(visual_features, 2)
      output = self.classifier_head(visual_features)

      return output

  def load_pretrained_s3d(self, use_cuda):

    if use_cuda:
        weight_dict = torch.load("drive/MyDrive/ASD/S3D_kinetics400.pt")
    else:
        weight_dict = torch.load("drive/MyDrive/ASD/S3D_kinetics400.pt", map_location = ("cpu"))

    model_dict = self.visual_extractor.state_dict()

    for name, param in weight_dict.items():
      if 'module' in name:
        name = '.'.join(name.split('.')[1:])
      if name in model_dict:
        if param.size() == model_dict[name].size():
            model_dict[name].copy_(param)
        else:
            print (' size? ' + name, param.size(), model_dict[name].size())
            pass
      else:
        print (' name? ' + name)
        pass

  def extract_features(self, audio_input, visual_input):

    audio_features = self.audio_extractor(audio_input)
    visual_features = self.visual_extractor(visual_input)

    return torch.cat((audio_features,visual_features), dim = 1)