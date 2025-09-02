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

Preliminary results in `viz.ipynb`. For example, with metrics logged in epochs 1, 3, and 5:

<img width="1189" height="489" alt="1_3_5" src="https://github.com/user-attachments/assets/47eb5e7d-6a74-44d9-b99b-273f7f597f62" />

and for taking into account the first 20 epochs:

<img width="1189" height="489" alt="20" src="https://github.com/user-attachments/assets/cc2874b6-c895-4f8c-92d6-23ca8c329993" />

Using per-token accuracy doesn't seem to yield difficulty regions like in the cartography paper.

TODO try other correctness metrics
