# Hinton distillation

This experiment consists of distillating a trained resnet18 (e.g. from the cross-entropy experiment) into a same network. Distillation is done using the [Hinton's approach](https://arxiv.org/abs/1503.02531), but without additional term for the hard labels.

Full results can be found in [this report](https://wandb.ai/binpord/kd-cifar100-resnet18/reports/Hinton-distillation--VmlldzoxODM2ODcy?accessToken=qicmxvfmse1p2mmpy3jf68z8lqfos4cpxg4gmsv96v9tf7myt00jdxnqrit3xg41).

**TL;DR**: distilling the 50% cross-entropy model yields model with 52.7% accuracy. Further distilling it yields 53.8%, and distilling that model yet again - 54.1%. The fourth run of self-distillation, however, yield a model with lower (but still impressive) test accuracy of 53.6%. Training accuracy steadily declines after every round. Training is done with regularization (i.e. `AdamW`), which means that results here are very consistent with [this paper](https://arxiv.org/abs/2002.05715).
