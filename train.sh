cd projects/CLIPstyler
CUDA_VISIBLE_DEVICES=1
python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
--name exp1 \
--text "Acrylic painting" --test_dir ./test_set_small --max_iter 2000 --num_test 6
# python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
# --name exp1 \
# --text "Desert sand" --test_dir ./test_set_small --max_iter 2000 --num_test 6
# python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
# --name exp1 \
# --text "Stone wall" --test_dir ./test_set_small --max_iter 2000 --num_test 6
# python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
# --name exp1 \
# --text "Watercolor painting with purple brush" --test_dir ./test_set_small --max_iter 2000 --num_test 6
# python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
# --name exp1 \
# --text "Oil painting with blue and red brushes" --test_dir ./test_set_small --max_iter 2000 --num_test 6
# python train_fast.py --content_dir /home/filippo/datasets/DIV2K_train_HR \
# --name exp1 \
# --text "Ink wash painting with green brush" --test_dir ./test_set_small --max_iter 2000 --num_test 6