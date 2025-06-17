import argparse
from pathlib import Path
from matplotlib import pyplot
import math

import csv
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from template import imagenet_templates
import fast_stylenet
from sampler import InfiniteSamplerWrapper
import clip
import time
from template import imagenet_templates
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
from torchmetrics.image import StructuralSimilarityIndexMeasure
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel


cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
st = time.time()
file_scores = "scores_weights.csv"

def truncate(f):
    n=3
    return math.trunc(f * 10**n) / 10**n

def train_transform(crop_size=224):
    transform_list = [
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),                     ##add antialias=True
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def hr_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

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

def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def clip_normalize(image):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def reverse_normalize(image):
    mean=torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).to(device)
    std=torch.tensor([1./0.229, 1./0.224, 1./0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image
def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def get_clip_score(clip_model, img, text):
    #out1, out2 = clip_model.encode(clip_normalize(img), clip.tokenize(args.text).to(device))
    img_feature = clip_model.encode_image(clip_normalize(img))
    tx_feature = clip_model.encode_text(clip.tokenize(text).to(device))
    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
    tx_feature = tx_feature / tx_feature.norm(dim=1, keepdim=True)
    score = img_feature @ tx_feature.t()
    
    score = score.mean()
    #print("clip score " + str(score.item()))
    return score

def get_ssim(preds, target):
    ssim = StructuralSimilarityIndexMeasure().to(device)
    score = ssim(preds, target)
    #print("SSIM " + str(score.item()))
    return score

def get_aesthetic_score(clip_model, model_ae, img):
    img_emb = clip_model.encode_image(clip_normalize(img))
    score = model_ae(img_emb)
    score = score.mean()
    #print("Aesthetic score " + str(score.item()))
    return score

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default ='./train_set_small')
parser.add_argument('--test_dir', type=str, default ='./test_set_small_new') 
parser.add_argument('--hr_dir', type=str)  
parser.add_argument('--img_dir', type=str, default ='./test_set_small_new')     
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./model_fast',
                    help='Directory to save the model')

parser.add_argument('--text', default='Fire',
                    help='text condition')
parser.add_argument('--name', default='none',
                    help='name')

parser.add_argument('--layer_enc_c_mamba',type=int, default=1)
parser.add_argument('--layer_enc_s_mamba',type=int, default=0)
parser.add_argument('--layer_dec_mamba',type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--content_weight', type=float, default=1.0)           
parser.add_argument('--clip_weight', type=float, default=10.0)             
parser.add_argument('--tv_weight', type=float, default=1e-4)               
parser.add_argument('--glob_weight', type=float, default=1.0)              
parser.add_argument('--n_threads', type=int, default=16)                   
parser.add_argument('--num_test', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=200)
parser.add_argument('--save_img_interval', type=int, default=50)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--thresh', type=float, default=0.7)
parser.add_argument('--decoder', type=str, default='./models/decoder.pth')
parser.add_argument('--linear', type=int, default=0)
parser.add_argument('--mlp', type=int, default=1)
parser.add_argument('--vssm', type=int, default=4)
parser.add_argument('--addtext', type=str, default='')

args = parser.parse_args()

str_weights = str(args.content_weight) + '-' + str(args.clip_weight) + '-' + str(args.tv_weight) + '-' + str(args.glob_weight)
str_encdec = str(args.layer_enc_s_mamba) + '-' + str(args.layer_enc_c_mamba) + '-' + str(args.layer_dec_mamba)

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

vgg = fast_stylenet.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

network = fast_stylenet.Net(vgg, args.layer_enc_s_mamba, args.layer_enc_c_mamba, args.layer_dec_mamba, args.linear, args.mlp, args.vssm)
network.train()
network.to(device)
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

model_ae = AestheticMeanPredictionLinearModel(512)
model_ae.load_state_dict(
    torch.load("models/sac_public_2022_06_29_vit_b_32_linear.pth")
)
model_ae = model_ae.to(device)

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

source = "a Photo"


content_tf = train_transform(args.crop_size)
hr_tf = hr_transform()
test_tf = test_transform()


content_dataset = FlatFolderDataset(args.content_dir, content_tf)
test_dataset = FlatFolderDataset(args.test_dir, test_tf)


augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
test_iter = iter(data.DataLoader(
    test_dataset, batch_size=args.num_test,
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(list(network.decoder.parameters()) +
                             list(network.mamba.parameters()),
                             lr=args.lr)

test_images1 = next(test_iter)
test_images1 = test_images1.cuda()

if args.hr_dir is not None:
    hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)
    
    hr_iter = iter(data.DataLoader(
        hr_dataset, batch_size=1,
        num_workers=args.n_threads))
    hr_images = next(hr_iter)
    hr_images = hr_images.cuda()

##prompt = "Futuristic painting"
print("\nStyle:", args.text)
print("Train set:", args.content_dir)
print("Test set:", args.test_dir)
print("lineare:", args.linear, ", mlp:", args.mlp)
print("batch size:", args.batch_size)
##print("Number direction of vssm: ", args.vssm)

#used to plot loss
loss_content_values = []
loss_patch_values = []
loss_dir_values = []
loss_tv_values = []
loss_values = []

#used to plot scores
clip_score_epoch = []
clip_scores = []
ssim_values = []
aesth_values = []

best_ssim = -1
best_clip_score = -1
best_aesth = -1
bc_int = -1

higher_ssim = -1
hs_aesth = -1
hs_clip = -1
ha_int = -1

higher_aesth = -1
ha_clip = -1
ha_ssim = -1
hs_int = -1

img_out = []
output_name = ''

with torch.no_grad():
    template_text = compose_text_with_templates(args.text, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)                                ##torch.Size([79, 77])
    text_features = clip_model.encode_text(tokens).detach()                         ##torch.Size([79, 512])
    text_features = text_features.mean(axis=0, keepdim=True)                        ##torch.Size([1, 512])
    text_features /= text_features.norm(dim=-1, keepdim=True)
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
        
for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)                  ##torch.Size([4, 3, 224, 224])

    istest = False

    loss_c, out_img = network(content_images, text_features)        ##out_img->torch.Size([4, 3, 224, 224])
    loss_patch = 0

    aug_img = []
    for it in range(16):
        out_aug = augment_trans(out_img)                            ##torch.Size([4, 3, 224, 224])
        aug_img.append(out_aug)
    aug_img = torch.cat(aug_img,dim=0)                              ##torch.Size([64, 3, 224, 224])
    source_features = clip_model.encode_image(clip_normalize(content_images))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
    
    image_features = clip_model.encode_image(clip_normalize(aug_img))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
    img_direction = (image_features-source_features.repeat(16,1))
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    
    text_direction = (text_features-text_source)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))#.mean()

    loss_temp[loss_temp<args.thresh] =0

    loss_patch+=loss_temp.mean()
    glob_features = clip_model.encode_image(clip_normalize(out_img))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    
    loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()
    
    loss_c = args.content_weight * loss_c
    reg_tv = args.tv_weight*get_image_prior_losses(out_img)
    
    loss = loss_c + args.clip_weight*loss_patch + args.glob_weight*loss_glob + reg_tv

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_content_values.append(loss_c.cpu().detach())
    loss_patch_values.append(loss_patch.cpu().detach())
    loss_dir_values.append(loss_glob.cpu().detach())
    loss_tv_values.append(reg_tv.cpu().detach())
    loss_values.append(loss.cpu().detach())

    if (i+1)%100==0:
        print('loss_content:' + str(loss_c.item()))
        print('loss_patch:' + str(loss_patch.item()))   
        print('loss_dir:' + str(loss_glob.item())) 
        print('loss_tv:' + str(reg_tv.item()))


    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = fast_stylenet.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'clip_decoder_iter_{:d}.pth.tar'.format(i + 1))

    if (i + 1) % args.save_img_interval ==0 :
        with torch.no_grad():            
            
            ##RESULT WITH IMAGES OF TEST SET
            _, test_out1 = network( test_images1, text_features)       #torch.cuda.FloatTensor - torch.Size([4, 3, 512, 512])
            #print("test_images1" + str(test_images1.shape))
            
            ##CALCULATION CLIP SCORE, SSIM, AESTHETIC SCORE
            score = get_clip_score(clip_model, test_out1, args.text)
            clip_scores.append(score.cpu().detach())
            clip_score_epoch.append(i)

            ssim_val = get_ssim(test_out1, test_images1)
            ssim_values.append(ssim_val.cpu().detach())

            aesth_val = get_aesthetic_score(clip_model, model_ae, test_out1)
            aesth_values.append(aesth_val.cpu().detach())

            if (i+1)%100==0:
                elapsed_time = time.time() - st
                print('execution time:' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
                print('clip score:' + str(score.item()))
                print('SSIM:' + str(ssim_val.item()))
                print('aesthetic value:' + str(aesth_val.item()))

            if (ssim_val > higher_ssim):
                higher_ssim = ssim_val
                hs_clip = score
                hs_int = i
                hs_aesth = aesth_val
            
            if (aesth_val > higher_aesth):
                higher_aesth = aesth_val
                ha_clip = score
                ha_ssim = ssim_val
                ha_int = i

            if (score > best_clip_score):
                best_clip_score = score
                best_ssim = ssim_val
                best_aesth = aesth_val
                bc_int = i
                liv_string = ''
                if args.linear == 1:
                    liv_string = liv_string + 'lin'
                if args.mlp == 1:
                    liv_string = liv_string + 'mlp'
                
                #FILE NAME: test_(num_layer_encoder-decoder_mamba)_(weights)_(options)_(num_iteration).png
                output_name = './output_fast/' + args.text + '_'+ str_encdec + '_'+ str_weights + '_' + str(args.addtext) + '_' + str(i+1)+'.png'
                img_out = test_out1
                print("partial clip score:" + str(best_clip_score.item()))
                print("output file:" + output_name)
            
            if (i+1) == args.max_iter:
                img_out = adjust_contrast(img_out,1.5)
                output_test = torch.cat([test_images1,img_out],dim=0)
                save_image(output_test, str(output_name),nrow=img_out.size(0),normalize=True,scale_each=True)
                print("best metrics: clip score " + str(best_clip_score.item()) + ", SSIM " + str(best_ssim.item()) + ", aesthetic val " + str(best_aesth.item()))
                print("higher ssim " + str(higher_ssim.item()) + ", higher aesth " + str(higher_aesth.item()))
                print("output file:" + output_name)
                
                ##write on file .csv all scores
                with open(file_scores, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([args.text, str(args.layer_enc_s_mamba), str(args.layer_enc_c_mamba), str(args.layer_dec_mamba),
                                     str(args.content_weight), str(args.clip_weight), str(args.tv_weight), str(args.glob_weight), 
                                     str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))),
                                     truncate(best_clip_score.item()), truncate(best_ssim.item()), truncate(best_aesth.item()), bc_int,
                                     truncate(higher_ssim.item()), hs_int, truncate(higher_aesth.item()), ha_int])

            
            
            if args.hr_dir is not None:
                _, test_out = network(hr_images, text_features)
                test_out = adjust_contrast(test_out,1.5)
                output_name = './output_fast/hr2_'+ args.name +'_'+ args.text +'_'+ str(i+1)+'.png'
                save_image(test_out, str(output_name),nrow=test_out.size(0),normalize=True,scale_each=True)


fig, (ax1, ax2, ax3, ax4, ax5) = pyplot.subplots(5, 1 )
fig.set_size_inches(20, 40)
fig.suptitle('Total Loss and Losses')
ax1.plot(loss_values, label='train_loss', color = 'red')
ax1.legend()
ax2.plot(loss_content_values, label='loss_content', color = 'blue')
ax2.plot(loss_patch_values, label='loss_patch', color = 'green')
ax2.plot(loss_dir_values, label='loss_dir', color = 'yellow')
ax2.plot(loss_tv_values, label='loss_tv', color = 'pink')
ax2.legend()
ax3.plot(clip_score_epoch, clip_scores, label='CLIP score', color = 'black')
ax3.legend()
ax4.plot(clip_score_epoch, ssim_values, label='SSIM', color = 'red')
ax4.legend()
ax5.plot(clip_score_epoch, aesth_values, label='Aesthetic score', color = 'green')
ax5.legend()
fig.savefig('./plot/' + args.text + '_' + str_encdec + '_' + str_weights + '_' + str(args.addtext) + '.png')        
