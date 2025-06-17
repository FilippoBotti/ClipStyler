#!/usr/bin/env bash
##prompts=('Watercolor painting with purple brush' 'a monet style painting' 'the great wave of kanagawa by Hokusai' 'a sketch with black pencil' 'Wheatfield by Vincent van Gogh' 'stone wall' 'neon light' 'underwater' 'pop art' 'white wool')
##prompts=('stone wall' 'underwater')

#prompts=('Watercolor painting with purple brush' 'monet painting' 'stone wall' 'a sketch with black pencil' 'underwater' 'Wheatfield by Vincent van Gogh' 'white wool' 'fire')
#prompts=('a sketch with black pencil' 'underwater' 'Wheatfield by Vincent van Gogh' 'neon light' 'pop art' 'white wool')
prompts=('stone wall' 'a sketch with black pencil' 'underwater' 'Wheatfield by Vincent van Gogh' 'white wool' 'fire')
for i in "${prompts[@]}"
do
    
    #python3 train_fast.py --layer_enc_c_mamba 3 --layer_dec_mamba 3 --max_iter 1000 --save_img_interval 25 --text "$i"
    #python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 3 --max_iter 1000 --save_img_interval 25 --text "$i"
    #python3 train_fast.py --layer_enc_c_mamba 3 --layer_dec_mamba 1 --max_iter 1000 --save_img_interval 25 --text "$i"
    python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --max_iter 1000 --save_img_interval 25 --text "$i"
    python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --max_iter 1000 --addtext 'batch8' --batch_size 8 --save_img_interval 25 --text "$i"
    python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --max_iter 1000 --clip_weight 20.0 --content_weight 1.0 --save_img_interval 25 --text "$i"
    python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --max_iter 1000 --clip_weight 5.0 --content_weight 10.0 --save_img_interval 25 --text "$i"
    python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --max_iter 1000 --addtext 'thresh1' --thresh 1 --save_img_interval 25 --text "$i"
    #python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 0 --max_iter 1000 --save_img_interval 25 --text "$i"
    #python3 train_fast.py --layer_enc_c_mamba 0 --layer_dec_mamba 1 --max_iter 1000 --save_img_interval 25 --text "$i"
done

##python3 train_fast.py --layer_enc_c_mamba 3 --layer_dec_mamba 3 --max_iter 1000 --save_img_interval 100 --text 'a sketch with black pencil'
#python3 train_fast.py --layer_enc_c_mamba 1 --layer_dec_mamba 1 --linear 1 --max_iter 1000 --save_img_interval 25 --text 'stone wall'

