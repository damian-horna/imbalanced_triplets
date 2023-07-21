# Imbalanced triplets

## Abstract
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

## The idea
Imbalanced data often forms very complicated structures in the feature space. Additional obstacles
in their classification are caused by the data difficulty factors (e.g., overlapping, noise, decompo-
sition of the minority class into many rare sub-concepts) often associated with such datasets. 

In order to mitigate those issues, we came up with an idea to transform a feature space into
an easier one with triplet networks.

The idea is to take the original dataset, feed it to a triplet-based neural network, transform it
to an easier representation and then train the classifier on this easier representation where these
data difficulty factors are hopefully reduced. The learned representation is taken from the last
layer of the network and can be used with any independent classifier (including ensembles). We
visualize this idea below:

![triplets-idea](https://github.com/damian-horna/imbalanced_triplets/assets/26741353/c679448d-79e2-4781-ad77-9b847ddfae85)

Different variants of the proposed approach have been experimentally
evaluated using 17 diversified datasets. Our experiments show that learning a new representation
of multi-class imbalanced data with similarity learning methods and then training classifiers on
such representation can significantly improve their performance on most datasets in comparison to
training on original representation or using well known pre-processing methods.
