import torch
import torch.nn as nn
import fast_stylenet 
import mamba
from collections import OrderedDict

def load_pretrained(args):
    vgg = fast_stylenet.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

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
    mlp_style = fast_stylenet.mlp.half()

    embedding = fast_stylenet.PatchEmbed()

    decoder_path = args.decoder_path
    mamba_path = args.mamba_path
    embedding_path = args.embedding_path
    mlp_path = args.mlp_path

    decoder.eval()
    mamba_net.eval()
    vgg.eval()
    mlp_style.eval()
    
    new_state_dict = OrderedDict()
    state_dict = torch.load(decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(mamba_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    mamba_net.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(mlp_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    mlp_style.load_state_dict(new_state_dict)

    with torch.no_grad():
        network = fast_stylenet.Net(vgg, mamba_net, decoder, mlp_style, embedding)

    print(f"Loaded Embedding checkpoints from {embedding_path}")
    print(f"Loaded Mamba checkpoints from {mamba_path}")
    print(f"Loaded CNN decoder checkpoints from {decoder_path}")
    print(f"Loaded Style mlp checkpoints from {mlp_path}")
    return network