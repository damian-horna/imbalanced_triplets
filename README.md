# Imbalanced triplets
Improving the classification of multi-class imbalanced data is more difficult than its two-
class counterpart. In this repo, we use deep neural networks to train new representations of
tabular multi-class data. Unlike the typically developed re-sampling pre-processing meth-
ods, our proposal modifies the distribution of features, i.e. the positions of examples in the
learned embedded representation, and it does not modify the class sizes. In order to learn
such embedded representations we introduced various definitions of triplet loss functions:
the simplest one uses weights related to the degree of class imbalance, while the next pro-
posals are intended for more complex distributions of examples and aim to generate a safe
neighborhood of minority examples. Similarly to the resampling approaches, after applying
such preprocessing, different classifiers can be trained on new representations. Experiments
with popular multi-class imbalanced benchmark data sets and three classifiers showed the
advantage of the proposed approach over popular pre-processing methods as well as basic
versions of neural networks with classical loss function formulations.

![triplets-idea](https://github.com/damian-horna/imbalanced_triplets/assets/26741353/c679448d-79e2-4781-ad77-9b847ddfae85)
