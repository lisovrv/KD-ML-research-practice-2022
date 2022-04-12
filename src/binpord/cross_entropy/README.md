# Cross entropy

This experiment tries to achieve best possible validation accuracy on CIFAR100 dataset using the resnet18 architecture.

I trained model for 50 epochs using the `AdamW` optimizer with `OneCycleLR` scheduler. Learning rate and weight decay coefficients were tuned using the bayesian optimization. Results of the sweep can be found [in this report](https://wandb.ai/binpord/cifar100_resnet18_cross_entropy/reports/Cross-Entropy-sweep--VmlldzoxODI4NjQ2?accessToken=97zlf7no78xt2jhe5amj1be33tzspjgbi4ihirk83tr00a9bw6zzxi47f4xfh8fh).

**TL;DR**: best possible accuracy with this approach is around 51% with learning rate of 0.04 and weight decay of 0.2. Final model was trained (earlier) with learning rate of 0.05 and weight decay of 0.1 and showed 50% val accuracy.
