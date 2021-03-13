# ML
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir matrix_lang-ar_msa_ea_trfs_crf_lr0.1_b32_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --early_stop 10

# EL
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir embedded_lang-arz_msa_ea_trfs_crf_lr0.1_b32_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --early_stop 10

# LINEAR
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir linear_msa_ea_trfs_crf_lr0.1_b32_arz_ar_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --early_stop 10

# CONCAT
CUDA_VISIBLE_DEVICES=0 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir concat_msa_ea_trfs_crf_lr0.1_b32_arz_ar_stop10 --batch_size 32 --mode concat --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --early_stop 10

# ATTN_SUM
CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_stop10 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --early_stop 10

# ATTN_SUM, BPE, CHAR
CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_bpe_char_stop10 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --bpe_lang_list arz ar --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=1 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_bpe_char_stop10_2 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --bpe_lang_list arz ar --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_bpe_char_stop10_3 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --bpe_lang_list arz ar --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_bpe_char_stop10_4 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --bpe_lang_list arz ar --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/cc.arz.300_lince_ner_msa_ea.vec embedding/cc.ar.300_lince_ner_msa_ea.vec --cuda --use_crf --model_dir attn_sum_msa_ea_trfs_crf_lr0.1_b32_arz_ar_bpe_char_stop10_5 --batch_size 32 --mode attn_sum --train_file data/lince_ner_msa_ea/train.conll --valid_file data/lince_ner_msa_ea/dev.conll --test_file data/lince_ner_msa_ea/test.conll --lr 0.1 --bpe_lang_list arz ar --add_char_emb --early_stop 10