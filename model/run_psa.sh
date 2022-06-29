CUDA_VISIBLE_DEVICES=0 python model/psa_meantext_sum.py --dataset_name Twitter15 --model_name event_sep_meantext_psa_sum --epochs 16 --iters 20 
CUDA_VISIBLE_DEVICES=0 python model/psa_meantext_sum.py --dataset_name Twitter16 --model_name event_sep_meantext_psa_sum --epochs 30 --iters 20
CUDA_VISIBLE_DEVICES=0 python model/psa_roottext_mean.py --dataset_name pheme_veracity_t10 --model_name event_sep_roottext_psa_mean --epochs 30 --iters 20
