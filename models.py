import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BasicConv2d_Ins(nn.Module):
    '''
    BasicConv2d module with InstanceNorm
    '''
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding):
        super(BasicConv2d_Ins, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernal_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x


class block32_Ins(nn.Module):
    def __init__(self, scale=1.0):
        super(block32_Ins, self).__init__()

        self.scale = scale

        self.branch0 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0)
        )

        self.branch1 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(48, 64, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class encoder(nn.Module):
    '''
    encoder structure: Inception + Instance Normalization
    '''
    def __init__(self, latent_dim=3, GRAY=True):
        super(encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        if GRAY:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(1, 32, kernal_size=5, stride=1, padding=2))
        else:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(3, 32, kernal_size=5, stride=1, padding=2))

        self.conv2 = nn.Sequential(BasicConv2d_Ins(32, 64, kernal_size=5, stride=1, padding=2))
        self.repeat = nn.Sequential(
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            # block32_Ins(scale=0.17)
        )
        self.conv3 = nn.Sequential(BasicConv2d_Ins(64, 128, kernal_size=5, stride=1, padding=2))
        self.conv4 = nn.Sequential(BasicConv2d_Ins(128, 128, kernal_size=5, stride=1, padding=2))

        self.fc = nn.Sequential(
        	nn.Dropout(p=0.5), 
        	nn.Linear(8 * 8 * 128, 1024),
        	nn.ReLU(), 

        	nn.Dropout(p=0.5), 
        	nn.Linear(1024, 128),
        	nn.ReLU(), 

        	nn.Linear(128, latent_dim)
        )

    def forward(self, x_in):
        # in_chanx128x128 -> 32x128x128
        self.conv1_out = self.conv1(x_in)
        # 32x128x128 -> 32x64x64
        self.ds1_out = self.maxpool(self.conv1_out)
        # 32x64x64 -> 64x64x64
        self.conv2_out = self.conv2(self.ds1_out)
        # 64x64x64 -> 64x32x32
        self.ds2_out = self.maxpool(self.conv2_out)
        # 64x32x32 -> 64x32x32
        self.incep_out = self.repeat(self.ds2_out)
        # 64x32x32 -> 128x32x32
        self.conv3_out = self.conv3(self.incep_out)
        # 128x32x32 -> 128x16x16
        self.ds3_out = self.maxpool(self.conv3_out)
        # 128x16x16 -> 128x16x16
        self.conv4_out = self.conv4(self.ds3_out)
        # 128x16x16 -> 128x8x8
        self.ds4_out = self.maxpool(self.conv4_out)

        # 128x8x8 -> 3
        self.fc_out = self.fc(self.ds4_out.view(self.ds4_out.size(0), -1))

        return self.fc_out


class resblock(nn.Module):
    '''
    residual block
    '''
    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out


class decoder(nn.Module):
    def __init__(self, latent_dim=3, Nb=3, Nc=128, GRAY=False):
        '''
        decoder to generate an image
        :param Nz: dimension of noises
        :param Nb: number of blocks
        :param Nc: channel number
        '''
        super(decoder, self).__init__()

        # upsampling layer
        self.fc_up = nn.Sequential(
        	nn.Linear(latent_dim, 128),
        	nn.Linear(128, 1024),
        	nn.Linear(1024, 128 * 8 * 8)
        )
        self.emb1 = nn.Sequential(*[
            nn.Conv2d(128, Nc, 3, 1, 1),
            nn.ReLU(),
        ])
        self.emb2 = self._make_layer(resblock, Nb, Nc)

        # decoding layers
        self.us1 = nn.Sequential(*[
            nn.ConvTranspose2d(Nc, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
        ])
        self.us2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ])
        self.us3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ])
        self.us4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ])
        if GRAY:
            self.us5 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            self.us5 = nn.Sequential(*[
                nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])

    def _make_layer(self, block, num_blocks, n_chan):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(n_chan))
        return nn.Sequential(*layers)

    def forward(self, fea_in):
        
        self.fc_out = self.fc_up(fea_in)
        self.emb1_out = self.emb1(self.fc_out.view(-1, 128, 8, 8))
        # bsxNcx8x8 -> bsxNcx8x8
        self.emb2_out = self.emb2(self.emb1_out)

        # decoding:
        # bsxNcx8x8 -> bsx512x16x16
        self.us1_out = self.us1(self.emb2_out)
        # bsx512x16x16 -> bsx256x32x32
        self.us2_out = self.us2(self.us1_out)
        # bsx256x32x32 -> bsx128x64x64
        self.us3_out = self.us3(self.us2_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us4_out = self.us4(self.us3_out)
        # bsx64x128x128 -> bsxout_chanx128x128
        self.img = self.us5(self.us4_out)

        return self.img


class classifier(nn.Module):
    '''
    classifier head
    '''
    def __init__(self, latent_dim=3, n_class=7):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(), 

            nn.Linear(256, n_class)
        )

        self.logit = nn.Softmax(dim=1)

    def forward(self, fea_in):
        self.fc_out = self.fc(fea_in)
        self.logit_out = self.logit(self.fc_out)
        return self.logit_out


class AE(nn.Module):
    '''
    the class of auto-encoder
    '''
    def __init__(self, latent_dim=3, n_class=7, Nb=3, radius=1.0, GRAY=False):
        super(AE, self).__init__()
        # encoder and decoder
        self.radius = radius
        self.encoder = encoder(latent_dim=latent_dim, GRAY=GRAY)
        self.decoder = decoder(latent_dim=latent_dim, Nb=Nb, GRAY=GRAY)
        self.classifier = classifier(latent_dim=latent_dim, n_class=n_class)

    def reparameterize(self, z_mean, radius):
        gaussian_smooth = torch.distributions.normal.Normal(z_mean, torch.ones_like(z_mean)*radius)
        return gaussian_smooth

    def embed(self, img):
        fea = self.encoder(img)
        return fea

    def gen_img(self, img):
        fea = self.encoder(img)
        rec_img = self.decoder(fea)
        return rec_img

    def gen_img_from_embedding(self, fea):
        rec_img = self.decoder(fea)
        return rec_img

    def forward(self, img):
        fea = self.encoder(img)
        gaussian_smooth = self.reparameterize(fea, self.radius)
        new_fea = gaussian_smooth.rsample()

        class_img = self.classifier(new_fea)
        rec_img = self.decoder(new_fea)
        return rec_img, class_img


class dualAE(nn.Module):
    '''
    the class of auto-encoder
    '''
    def __init__(self, latent_dim=3, n_class_emo=7, n_class_id=10, Nb=3, radius=1.0, GRAY=False):
        super(dualAE, self).__init__()
        # encoder and decoder
        self.radius = radius
        self.encoder_emo = encoder(latent_dim=latent_dim, GRAY=GRAY)
        self.encoder_id = encoder(latent_dim=latent_dim, GRAY=GRAY)

        self.decoder = decoder(latent_dim=latent_dim*2, Nb=Nb, GRAY=GRAY)

        self.classifier_emo = classifier(latent_dim=latent_dim, n_class=n_class_emo)
        self.classifier_id = classifier(latent_dim=latent_dim, n_class=n_class_id)

    def reparameterize(self, z_mean, radius):
        gaussian_smooth = torch.distributions.normal.Normal(z_mean, torch.ones_like(z_mean)*radius)
        return gaussian_smooth

    def embed(self, img):
        fea_emo = self.encoder_emo(img)
        fea_id = self.encoder_id(img)
        return fea_emo, fea_id

    def gen_img(self, img):
        fea_emo = self.encoder_emo(img)
        fea_id = self.encoder_id(img)
        rec_img = self.decoder(torch.cat((fea_emo, fea_id), 1))
        return rec_img

    def gen_img_from_embedding(self, fea_emo, fea_id):
        rec_img = self.decoder(torch.cat((fea_emo, fea_id), 1))
        return rec_img

    def forward(self, img):
        fea_emo = self.encoder_emo(img)
        fea_id = self.encoder_id(img)

        gaussian_smooth_emo = self.reparameterize(fea_emo, self.radius)
        gaussian_smooth_id = self.reparameterize(fea_id, self.radius)
        new_fea_emo = gaussian_smooth_emo.rsample()
        new_fea_id = gaussian_smooth_id.rsample()

        class_emo = self.classifier_emo(new_fea_emo)
        class_id = self.classifier_id(new_fea_id)
        new_fea = torch.cat((new_fea_emo, new_fea_id), 1)
        rec_img = self.decoder(new_fea)
        return rec_img, class_emo, class_id






















