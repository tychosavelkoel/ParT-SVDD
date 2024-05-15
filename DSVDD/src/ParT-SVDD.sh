#!/bin/bash

modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"


python main.py \
    mnist ParT ../log/mnist_test ../data --objective one-class\
     --lr 0.0001 --n_epochs 2 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 \
     --ae_lr 0.0001 --ae_n_epochs 2 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3\
      --normal_class 3 \
     --data-train "/data/alice/wvolgering/soft/JetToyHI/Data_Train.root" \
     --data-test "/data/alice/wvolgering/soft/JetToyHI/Data_Test.root" \
     --network-config $modelopts --data-config ../data/Test/test_kin.yaml \
     --log logs/Test/Test_${model}_{auto}.log --model-prefix training/Test/${model}/{auto}/net \
     --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
     --samples-per-epoch 1600 --samples-per-epoch-val 200 \
     --min-epochs 0 --max-epochs 1 --gpus 0 \
     --predict-output predict/pred.root --optimizer ranger \
     --tensorboard Quenched_${FEATURE_TYPE}_${model}