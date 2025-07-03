#!/usr/bin/env bash
##prompts=('Watercolor painting with purple brush' 'a monet style painting' 'the great wave of kanagawa by Hokusai' 'a sketch with black pencil' 'Wheatfield by Vincent van Gogh' 'stone wall' 'neon light' 'underwater' 'pop art' 'white wool')

prompts=('Watercolor painting with purple brush' 'underwater' 'a monet style painting' 'stone wall' 'Wheatfield by Vincent van Gogh' 'a sketch with black pencil')

for i in "${prompts[@]}"
do 
    python3 train_fast.py --text "$i"

    #examples with other options
    #python3 train_fast.py --layer_enc_s_mamba 0 --layer_enc_c_mamba 1 --layer_dec_mamba 1 --text "$i" --vssm 2 --addtext "vssm2"
    #python3 train_fast.py --layer_enc_s_mamba 0 --layer_enc_c_mamba 1 --layer_dec_mamba 1 --text "$i" --patch 4 --addtext "patch4"

    #to test inference or load models
    
    #text_model="${i// /_}"
    #python3 test_fast.py --text "$text_model" --decoder_path './model_fast/decoder_Wheatfield_by_Vincent_van_Gogh_iter_500.pth' --mamba_path './model_fast/mamba_Wheatfield_by_Vincent_van_Gogh_iter_500.pth' --embedding_path './model_fast/embedding_Wheatfield_by_Vincent_van_Gogh_iter_500.pth' --mlp_path './model_fast/style_mlp_Wheatfield_by_Vincent_van_Gogh_iter_500.pth'
    #python3 test_fast.py --inference 1


done
