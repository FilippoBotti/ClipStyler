import argparse
from pathlib import Path
import utility.function as func
import torch
import csv
import os
import mamba
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import fast_stylenet 
from utility.load_pretrained import load_pretrained
from sampler import InfiniteSamplerWrapper
import clip
from utility.template import imagenet_templates
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel


def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def hr_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', type=str, default ='./test_set_small_new') 
parser.add_argument('--save_dir', type=str, default ='./test_output') 

parser.add_argument('--text', default='Fire', help='text condition')
parser.add_argument('--hr_dir', type=str)     
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
# training options
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--num_test', type=int, default=16)

parser.add_argument('--layer_enc_c_mamba',type=int, default=3)
parser.add_argument('--layer_enc_s_mamba',type=int, default=0)
parser.add_argument('--layer_dec_mamba',type=int, default=3)
parser.add_argument('--vssm', type=int, default=4)
parser.add_argument('--patch', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--decoder_path', type=str, default='./model_fast/decoder_iter_1000.pth')
parser.add_argument('--mamba_path', type=str, default='./model_fast/mamba_iter_1000.pth')
parser.add_argument('--embedding_path', type=str, default='./model_fast/embedding_iter_1000.pth')
parser.add_argument('--mlp_path', type=str, default='./model_fast/mlp_iter_1000.pth')
parser.add_argument('--addtext', type=str, default='')

parser.add_argument('--inference', type=int, default=0)
args = parser.parse_args()
file_scores = "scores_weights_test.csv"

device = torch.device('cuda')


if args.inference != 0:
    vgg = fast_stylenet.vgg
    vgg = nn.Sequential(*list(vgg.children())[:31])

    embedding = fast_stylenet.PatchEmbed(patch_size=args.patch)
    if args.patch == 8:
        decoder = fast_stylenet.decoder
    elif args.patch == 16:
        decoder = fast_stylenet.decoder16
    elif args.patch == 4:
        decoder = fast_stylenet.decoder4
    else:
        print("Size patch only 4, 8 or 16")
        exit()
    mamba_net = mamba.Mamba(args=fast_stylenet.Args(args.layer_enc_s_mamba, args.layer_enc_c_mamba,args.layer_dec_mamba, args.vssm), d_model = 512)
    mlp_style = fast_stylenet.mlp
    network = fast_stylenet.Net(vgg, mamba_net, decoder, mlp_style, embedding)

    network.eval()
    network.to(device)
    st = time.time()


    for t in range(80):
        img = torch.rand(4, 3, 224, 224, device=device)
        text_feature = torch.rand(1, 512, device=device)
        #print(img.shape, text_feature.shape)
        _, _ = network( img, text_feature)


    elapsed_time = time.time() - st
    print("Time inference: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    exit()
else:
    network = load_pretrained(args)
    network.eval()
    network.to(device)

output_dir=args.save_dir
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

clip_model, _ = clip.load('ViT-B/32', device, jit=False)

model_ae = AestheticMeanPredictionLinearModel(512)
model_ae.load_state_dict(
    torch.load("models/sac_public_2022_06_29_vit_b_32_linear.pth")
)
model_ae = model_ae.to(device)

test_tf = test_transform()
test_dataset = FlatFolderDataset(args.test_dir, test_tf)
test_iter = iter(data.DataLoader(
    test_dataset, batch_size=args.num_test,
    num_workers=args.n_threads))

test_images1 = next(test_iter)
test_images1 = test_images1.cuda()

if args.hr_dir is not None:
    hr_tf = hr_transform()
    hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)
    hr_iter = iter(data.DataLoader(
    hr_dataset, batch_size=1,
    num_workers=args.n_threads))

    hr_images = next(hr_iter)
    hr_images = hr_images.cuda()




with torch.no_grad():
    template_text = func.compose_text_with_templates(args.text, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)                                ##torch.Size([79, 77])
    text_features = clip_model.encode_text(tokens).detach()                         ##torch.Size([79, 512])
    text_features = text_features.mean(axis=0, keepdim=True)                        ##torch.Size([1, 512])
    text_features /= text_features.norm(dim=-1, keepdim=True)    

with torch.no_grad():
    _, test_out1 = network( test_images1, text_features)
    elapsed_time = time.time() - st
    clip_score = func.get_clip_score(clip_model, test_out1, args.text, device)
    ssim_val = func.get_ssim(test_out1, test_images1, device)
    aesth_val = func.get_aesthetic_score(clip_model, model_ae, test_out1, device)
    
    test_out1 = adjust_contrast(test_out1,1.5)
    output_test = torch.cat([test_images1,test_out1],dim=0)
    output_name = output_dir + '/test_'+args.text+'_'+str(args.addtext)+'.png'
    save_image(output_test, str(output_name),nrow=test_out1.size(0),normalize=True,scale_each=True)
    print("Saved image: " + output_name)
    with open(file_scores, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([args.text, args.addtext,
                                     str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))),
                                     func.truncate(clip_score.item()), func.truncate(ssim_val.item()), func.truncate(aesth_val.item())])
                    
    if args.hr_dir is not None:
        _, test_out = network(hr_images, text_features)
        clip_score = func.get_clip_score(clip_model, test_out1, args.text, device)
        ssim_val = func.get_ssim(test_out1, test_images1, device)
        aesth_val = func.get_aesthetic_score(clip_model, model_ae, test_out1, device)
        test_out = adjust_contrast(test_out,1.5)
        output_name = output_dir + '/hr_'+args.text+'_'+str(args.addtext)+'.png'
        save_image(test_out, str(output_name),nrow=test_out.size(0),normalize=True,scale_each=True)
        print("Saved image: " + output_name)   
