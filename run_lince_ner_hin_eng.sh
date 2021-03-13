# ML
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/cc.hi.300_lince_ner_hin_eng.vec --cuda --use_crf --model_dir matrix_lang-hi_hin_eng_trfs_crf_lr0.1_b32_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --early_stop 10

# EL
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec --cuda --use_crf --model_dir embedded_lang-en_hin_eng_trfs_crf_lr0.1_b32_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --early_stop 10

# LINEAR
CUDA_VISIBLE_DEVICES=1 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir linear_hin_eng_trfs_crf_lr0.1_b32_hi_en_stop10 --batch_size 32 --mode linear --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --early_stop 10

# CONCAT
CUDA_VISIBLE_DEVICES=0 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir concat_hin_eng_trfs_crf_lr0.1_b32_hi_en_stop10 --batch_size 32 --mode concat --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --early_stop 10

# ATTN_SUM
CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_stop10 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --early_stop 10

# ATTN_SUM, BPE, CHAR
CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_bpe_char_stop10 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --bpe_lang_list hi en --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=1 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_bpe_char_stop10_2 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --bpe_lang_list hi en --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_bpe_char_stop10_3 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --bpe_lang_list hi en --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=3 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_bpe_char_stop10_4 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --bpe_lang_list hi en --add_char_emb --early_stop 10

CUDA_VISIBLE_DEVICES=2 python train.py --emb_list embedding/crawl-300d-2M-subword_lince_ner_hin_eng.vec embedding/cc.hi.300_lince_ner_hin_eng.vec embedding/glove.840B.300d.txt --cuda --use_crf --model_dir attn_hin_eng_trfs_crf_lr0.1_b32_hi_en_bpe_char_stop10_5 --batch_size 32 --mode attn_sum --train_file data/lince_ner_hin_eng/train.conll --valid_file data/lince_ner_hin_eng/dev.conll --test_file data/lince_ner_hin_eng/test.conll --lr 0.1 --bpe_lang_list hi en --add_char_emb --early_stop 10