## nmt-data-map

Trying to get ["data cartography"](https://arxiv.org/abs/2009.10795)-like maps for translation to do better [early dynamics data pruning](https://arxiv.org/abs/2405.19462). 

Follow fairseq's installation instructions and the de-en training [example](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#iwslt14-german-to-english-transformer) to prepare the data. We just added a new criterion `per_example_loss.py`. Run with

```bash
rm -rf checkpoints; rm *.csv; CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion per_example_loss --label-smoothing 0.1 \
    --max-tokens 4096 \
    --disable-validation \
    --report-accuracy \
    --max-epoch 20
```
