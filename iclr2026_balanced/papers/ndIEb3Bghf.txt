# FORMATTING INSTRUCTIONS FOR ICLR 2026 CONFERENCE SUBMISSIONS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Regularization plays a crucial role in neural network training by preventing overfitting and improving generalization. In this paper, we introduce a novel regularization technique grounded in the properties of characteristic functions, leveraging assumptions from decomposable distributions and the central limit theorem.
Rather than replacing traditional regularization methods such as L2 or dropout,
our approach is designed to supplement them, providing a contextual delta of
generalization. We demonstrate that integrating this method into standard architectures improves performance on benchmark datasets by preserving essential distributional properties and mitigating the risk of overfitting. This characteristic function-based regularization offers a new perspective in the direction of
distribution-aware learning in machine learning models.


1 INTRODUCTION


Regularization remains a cornerstone of modern machine learning, enabling models, especially deep
neural networks with vast parameter spaces, to generalize effectively and avoid overfitting. Traditional norm-based penalties, such as L2 (ridge) (Hoerl & Kennard, 1970), L1 (lasso) (Tibshirani,
1996), and Elastic Net, have long been foundational for controlling model complexity and promoting sparsity. Meanwhile, stochastic regularization techniques like dropout (Srivastava et al., 2014)
and its variants mitigate co-adaptation among units, and implicit methods including early stopping
(Prechelt, 1998), data augmentation, and label perturbation have further enriched this landscape.
Structured (Vapnik, 2006) and group-wise regularizers exploit dependencies among parameters to
yield more interpretable and computationally efficient models. Collectively, these methods form a
rich, well-studied toolkit that underpins much of contemporary machine learning.


In parallel, a growing body of work in probability and statistics has explored characteristic function–based methods, leveraging the Fourier transform of probability distributions to overcome challenges posed by intractable or unknown density functions (Warr, 2014). These techniques have
established rigorous computability results (Mori et al., 2009), advanced numerical methods for quantile estimation (Junike, 2025), inversion methods for density estimation (Lunde et al., 2018), enabled
efficient simulation of stochastic processes (Boyarchenko & Levendorskii, 2023), and led to practical software tools supporting broad applicability (Witkovsky, 2024). Applications of characteristic
function–based approaches span jump-diffusions, semi-Markov models, L´evy processes, and more.


Despite these advances, characteristic function–based methods have received limited attention
within the machine learning community, particularly in the context of regularization and model
generalization. While classical regularization methods primarily operate in the parameter or output
space, characteristic function–based inference offers a complementary perspective rooted in distributional structure and transform-domain analysis. This suggests potentially intriguing parallels that
remain largely unexplored.


This paper seeks to explore this gap by investigating regularization through the lens of characteristic
function–based methods. We explore how characteristic functions can serve as regularizers that
implicitly incorporate distributional information, providing new ways to control model behavior and
improve generalization. Our approach offers a principled yet simple framework that complements
existing regularization techniques by exploiting the rich representational power of characteristic
functions.


1


Our main contribution is the establishment of a novel regularization framework that leverages characteristic function–based methods to impose distributionally informed constraints on model training.
By integrating characteristic function perspectives with classical regularization principles, we open
new avenues for incorporating probabilistic structure into machine learning models. This work lies
at the intersection of transform-based inference and regularization, with the goal to enhance model
robustness and performance.


2 RELATED WORK


Traditional approaches such as L1 and L2 regularization (Tibshirani, 1996; Hoerl & Kennard, 1970)
constrain model complexity by penalizing the magnitude or sparsity of weights. In deep learning, stochastic methods like dropout (Srivastava et al., 2014) and data-augmentation-based techniques like mixup (Zhang et al., 2017) inject noise or synthetic variability to guide learning toward
more robust solutions. Other strategies operate implicitly, such as early stopping (Prechelt, 2002)
and sharpness-aware minimization (Foret et al., 2020), which regularize via optimization dynamics.
However, these methods primarily affect the learning process in parameter space, offering limited
interpretability in terms of output distributional behaviour.


An emerging line of work studies the statistical properties of a model’s _output_ _distribution_ as a
lens for regularization and evaluation. Generative models frequently use divergences between empirical and target distributions e.g., Kullback-Leibler, Jensen-Shannon, or Wasserstein metrics, as
objectives or diagnostics (Goodfellow et al., 2014; Arjovsky et al., 2017). Other approaches employ
_kernel-based distances_ like Maximum Mean Discrepancy (MMD) (Gretton et al., 2012) or energy
distances (Sz´ekely et al., 2004), which implicitly rely on moment-matching or characteristic function alignment. Despite these advances, the explicit use of _central limit theorem (CLT)_ assumptions
to model and regularize output distributions is rare. While CLT-based arguments have been invoked
to explain model behavior in wide neural networks (Lee et al., 2017; Matthews et al., 2018), they
are seldom used as direct regularization tools.


Our methodology rests on the premise that the final layer outputs of the network can be modeled
as a collection of Bernoulli random variables, whose aggregate behaviour, by virtue of the Lyapunov Central Limit Theorem (Lyapunov, 1900), approaches a Gaussian distribution in the limit of
large dimensionality. Rather than enforcing normality on parameters or intermediate activations, we
focus on the distribution of these output sums, leveraging their probabilistic structure to define a
natural frequency-domain regularizer. Deviations from the predicted Gaussian characteristic function serve as an informative metric quantifying the network’s departure from the asymptotic regime,
thus reflecting potential overfitting or instability in representation learning. This framework imposes
a functional prior that guides gradient updates toward outputs exhibiting asymptotic normality, effectively harnessing classical limit theorems within the modern paradigm of deep learning. To our
knowledge, this represents a novel instantiation of Lyapunov CLT-based constraints as a regularizing
force acting directly on neural network training.


3 REGULARIZATION METHODOLOGY


We now propose a regularization framework that leverages distributional convergence to enforce
structured behavior in neural network outputs.


3.1 MOTIVATION AND HIGH-LEVEL INTUITION


Specifically, we aim to softly constrain the model’s output representations by encouraging them to
relax toward a target distribution, achieved by modeling the output layer as a function of Bernoulli
random variables. This modeling is motivated by the observation that, in many classification settings, the output of a neural network (especially under sigmoid or softmax activation) can be interpreted as a sequence of Bernoulli trials. Each output unit reflects a probabilistic decision, and
collectively, these outputs can be visualized as a chain of Bernoulli outcomes. By slightly reformulating this output structure, we construct a random variable as described in Definition 3, which
allows us to treat the final layer’s activations as a sample from an approximate sum of independent
Bernoulli variables.


2


Under this formulation, and by invoking the Lyapunov Central Limit Theorem, we establish that
this random variable converges in distribution to the standard normal distribution as the number of
Bernoulli components increases and certain conditions are met. This theoretical insight forms the
basis for a new regularization term that penalizes deviations from this target Gaussian distribution.
Practically, this is implemented by comparing the characteristic function of the empirical representation with that of the standard normal, and incorporating the resulting distance into the loss function.
Because the construction of _ℵ_ is flexible, the regularizer can be generalized to different layers or
structural parts of the network. In this work, however, we focus on applying the regularization at the
output layer to keep the formulation both interpretable and practically feasible, while demonstrating
strong generalization improvements.


We now proceed to derive the regularization. The starting point is the Lyapunov Central Limit Theorem, which establishes that suitably normalized sums of independent random variables converge in
distribution to a Gaussian under mild moment conditions.


3.2 FORMALISING THE REGULARIZATION APPROACH


**Definition 1** (Lyapunov Central Limit Theorem) **.** Suppose we have a sequence of independent random variables, _{Y_ 1 _, Y_ 2 _, . . ., Yn}_, each with finite expected value _µi_ and variance _σi_ [2][.]


If we define the following sum of variances:


where _Xi_ _∼_ _Bern_ ( _pi_ ) = _⇒_ _µi_ = E[ _Xi_ ] = _pi_ and _s_ [2] _n_ [=][ �] _i_ _[n]_ =1 _[V ar]_ [[] _[X][i]_ [] =][ �] _i_ _[n]_ =1 _[p][i]_ [(1] _[ −]_ _[p][i]_ [)][.]

1We deviate from common practice and use _ϑ_ instead of _i_ to define the imaginary unit in a bid to reduce
confusion as the letter _i_ is used for indexing in much of the later proofs and writing


3


_s_ [2] _n_ [=]


If _∃δ_ _>_ 0, such that Lyapunov’s condition:


_n_

- _σi_ [2] _[.]_ (1)

_i_ =1


1
lim
_n→∞_ _s_ [2+] _n_ _[δ]_


_n_

- E - _|Yi −_ _µi|_ [2+] _[δ]_ [�] = 0 _,_ (2)


_i_ =1


is satisfied = _⇒_ the sum of the normalized variables _[Y][i]_ _s_ _[−]_ _n_ _[µ][i]_ converges in distribution to a standard

normal random variable as _n →∞_ :


1
_sn_


_n_
�( _Yi −_ _µi_ ) _→_ _N._ (3)


_i_ =1


where _N_ _∼N_ (0 _,_ 1)
(Lyapunov, 1900)


We model each data point as being generated from a random variable _ℵ_, which is represented as an
approximation of linear combination of Bernoulli variables _Xi_ _∼_ Bern( _p_ ). We will now demonstrate
that properly formulating _ℵ_ allows for convergence to _N_ (0 _,_ 1) as the number of Bernoulli variables
increases sufficiently.

**Axiom 1.** Let us establish the following fundamental assumption that will underpin our framework.
Given the true data distribution, _D_ (which can be conceptualized as the ”Population Distribution” in
statistical terms), we assert that the data points (the ”Samples” we usually have in our finite dataset)
are generated from a random variable _ℵ_ such that:


_ℵ∼D_ (4)


**Definition 2.** The characteristic function _ϕ_ ( _u_ ) of a random variable _Y_ is defined as:


_ϕ_ ( _u_ ) = E[ _e_ _[ϑuY]_ ] _._ [1] (5)


**Definition 3.** We model the random variable _ℵ_, as a linear combination of Bernoulli Random Variables, defined as follows:


_ℵ_ = [1]

_sn_


_n_


( _Xi −_ _µi_ ) _,_ (6)

_i_ =1


3.3 CHARACTERIZING THE OUTPUT DISTRIBUTION


**Proposition 1.** The characteristic function for _ℵ_ can be computed as follows from 5 and 6:


**Corollary** **1.** By the Lyapunov Central Limit Theorem, as _n_ _→∞_, the characteristic function
converges to that of the characteristic function of the normal distribution:
∵ _D_ _→N_ (0 _,_ 1) = _⇒_ ∴ _ϕD_ ( _u_ ) _→_ _ϕN_ (0 _,_ 1)( _u_ ) _._ (17)


Note this is true because rate of growth of the moments is contrained as per the Lyapunov condition,
described in detail by proof outlined in the appendix A.1. Some graphics from numerical simulation
of this effect is also attached in the appendix 7, along with some helper graphs to visualise some
transformations of characteristic functions as it is difficult to find some and there aren’t really many
online.


4


~~_√_~~ ( _ϑu_ (1 _−pi_ ))

 - _n_
_i_ =1 [(] _[pi][∗]_ [(1] _[−][pi]_ [))2] ( _pi_ ) (7)


~~_√_~~ ( _−ϑupi_ )

 - _n_
_i_ =1 [(] _[pi][∗]_ [(1] _[−][pi]_ [))2] (1 _−_ _pi_ ) +


_n_

- _e_


_i_ =1


_n_


_ϕD_ ( _u_ ) =


_n_

- _e_


_i_ =1


_n_


_Proof._


_ϕD_ ( _u_ ) = E[ _e_ _[ϑu][ℵ]_ ] = E - _e_ _[ϑu]_ _sn_ [1] - _ni_ =1 [(] _[X][i][−][µ][i]_ [)][�] _._ (8)


Using the product law of exponents ( _a_ _[n]_ _∗_ _a_ _[m]_ = _a_ [(] _[n]_ [+] _[m]_ [)] ), we rewrite the characteristic function:


    
[1]

_sn_ [(] _[X][i][−][µ][i]_ [)]


_._ (9)


= E


- _n_


- _e_ _[ϑu]_ [1]


_i_ =1


Next, we separate it into the following by linearity of the expectation:


- _n_

 


_sn_ [(] _[X][i][−][µ][i]_ [)] I _{Xi_ = 0 _}_


+ E


_sn_ [(] _[X][i][−][µ][i]_ [)] I _{Xi_ = 1 _}_


= E


- [1]

_e_ _[ϑu]_ _sn_

_i_ =1


- _n_


- _e_ _[ϑu]_ [1]


_i_ =1


By properties of the Bernoulli Random Variable this is more precisely:


_sn_ [(1] _[−][p][i]_ [)] I _{Xi_ = 1 _}_


- _n_

 


_sn_ [(0] _[−][p][i]_ [)] I _{Xi_ = 0 _}_


+ E


_._ (10)


_._ (11)


- _n_


- _e_ _[ϑu]_ [1]


_i_ =1


= E


- _e_ _[ϑu]_ [1]


_i_ =1


Using linearity of expectation, we have:


_sn_ [(1] _[−][p][i]_ [)] E[I _{Xi_ = 1 _}_ ] _._ (12)


_n_


_sn_ [(] _[−][p][i]_ [)] E[I _{Xi_ = 0 _}_ ] +


- _e_ _[ϑu]_ [1]


_i_ =1


_n_


- _e_ _[ϑu]_ [1]


_i_ =1


=


By definition of Expectation of Indicator Function, we have:


_sn_ [(] _[−][p][i]_ [)] P( _Xi_ = 0) +


_sn_ [(1] _[−][p][i]_ [)] P( _Xi_ = 1) _._ (13)


_n_


- _e_ _[ϑu]_ [1]


_i_ =1


=


_n_


- [1]

_e_ _[ϑu]_ _sn_

_i_ =1


By properties of the Bernoulli Random Variable, this can be re-expressed as:


_sn_ [(1] _[−][p][i]_ [)] _pi._ (14)


_n_


_sn_ [(0] _[−][p][i]_ [)] (1 _−_ _pi_ ) +


- _e_ _[ϑu]_ [1]


_i_ =1


_n_


- [1]

_e_ _[ϑu]_ _sn_

_i_ =1


=


Reformulating the _s_ [2] _n_ [:]


∵ _s_ [2] _n_ [=]


Thus, we conclude:


~~�~~

- _n_
��� ( _pi_ (1 _−_ _pi_ )) [2] (15)


1


_n_

- _σi_ [2] [=] _[⇒]_ [∴] _[s][n]_ [=]

1


~~�~~

- _n_
��� _σi_ [2] [=]

1


_n_

- _e_


_i_ =1


~~_√_~~ ( _−ϑupi_ )

 - _n_
_i_ =1 [(] _[pi]_ [(1] _[−][pi]_ [))2] (1 _−_ _pi_ ) +


~~_√_~~ ( _ϑu_ (1 _−pi_ ))

 - _n_
_i_ =1 [(] _[pi]_ [(1] _[−][pi]_ [))2] ( _pi_ ) (16)


= _⇒_ ∴ _ϕD_ ( _u_ ) =


_n_

- _e_


_i_ =1


3.4 FORMULATING THE REGULARIZATION OBJECTIVE


**Definition 4** (Regularization) **.** Regularization is a technique used to prevent overfitting by adding a
penalty term to the loss function. The regularized loss function is typically expressed as:


where ˆ _yi_ = _f_ ( _xi_ ) is the predicted output, _L_ is the loss function, _R_ ( _f_ ) is the regularization term, and
_λ_ is a hyperparameter that controls the trade-off between model fit and complexity.


If we interpret _ϕN_ (0 _,_ 1) as a relaxed fit that our function can be adjusted towards, we can establish
a regularization term _R_ ( _f_ ), which levies a penalty on the complexity of model _f_, by adding a
constraint through examining the difference between _ϕD_ and _ϕN_ (0 _,_ 1). This can be achieved by
measuring the distance between the signals using:


_R_ ( _f_ ) = _d_ ( _ϕD, ϕN_ (0 _,_ 1)) (19)


The choice of the distance metric, _d_ ( _·_ ), is up to the practitioner but we briefly mention the ones we
used for evaluation in the appendix B for reference.


3.5 EXTENSION TO BROADER NETWORK STRUCTURES


For greater generality, the regularization approach can be extended to different components of the
network or to various classes of networks by appropriately redefining the construction of the random
variable introduced in Definition 3. This, in turn, accordingly allows for a reformulation of the
equation in Proposition 1. In this work, we focus specifically on the output layer formulation as
presented and will evaluate its effectiveness on real-world datasets in the following section.


4 EXPERIMENTS


For evaluating the effectiveness of our regularization framework, we begin by considering five main
cases: no regularization (None), classical _L_ [1], _L_ [2], and our proposed _ψ_ 1 and _ψ_ 2 regularization terms
(that are described in detail in Appendix B), which are designed to reflect underlying signal structure. For clarity of exposition, and to focus on the most representative behaviors, we simplify this
comparison to **three main regimes** . The first is **ElasticNet** (Zou & Hastie, 2005), which interpolates
between _L_ [1] and _L_ [2] regularization and serves as a standard mixed-norm baseline. The second is our
mixed-norm approach called **SpectralNet** ( _ψ_ spec) (see Appendix B), which adopts the same formal
structure as ElasticNet, but replaces the classical norms with _ψ_ -based distances. The third regime
corresponds to the **unregularized** case, included as the baseline for context.


For our experiments, we evaluate the proposed regularization method across a diverse set of multiclass classification datasets, spanning various domains including tabular data, text, audio, and images. Table 1 summarizes the key characteristics of these datasets along with the representative
models employed, where the output layer size corresponds to the number of classes. The datasets
vary widely in scale and complexity, ranging from small tabular datasets like PhiUSIIL and Wine,
to large-scale image recognition benchmarks such as ImageNet-1K and ImageNet-21K, as well as
audio and text datasets with thousands of classes.


This broad selection allows us to comprehensively assess the generalization capability and robustness of our regularization approach across different modalities and model architectures, including
logistic regression, multi-layer perceptrons (MLPs), convolutional neural networks (CNNs), transformers, and state-of-the-art models like BERT and Vision Transformers. Detailed results and
analyses on these datasets are provided in the following sections, evaluating the effectiveness of
the proposed regularization method in diverse learning scenarios and data modalities in real-world,
multi-class classification settings.


Additionally, for all models, core hyperparameters such as regularization parameter _λ_, learning
rate were optimized using Optuna (Akiba et al., 2019). Model-specific parameters, such as the


5


min


_n_

- _L_ (ˆ _yi, yi_ ) + _λR_ ( _f_ ) (18)


_i_ =1


Table 1: Multi-Class Classification Datasets and Representative Models used for Experiments


**# Classes** **Dataset Name** **Domain** **Accuracy Metric** **Model (Output Layer = # of Classes)** **Notes**
2 PhiUSIIL (Phishing URL) Text Top-1 MLP Binary classification of phishing vs. benign URLs
3 Waveform Audio Top-1 Logistic Regression Synthetic waveforms in three categories
5 BBC News Text Top-1 Multinomial Logistic Regression News articles in five categories
10 MNIST Image Top-1 LeNet5 Handwritten digits (0–9)
15 Human Action Recognition (HAR) Image Top-1 CNN Image/video-based human activity recognition
20 20 Newsgroups Text Top-1 BERT Text classification of 20 news categories
35 Google Speech Commands (GSC) v2 Audio Top-1 RNN Voice command recognition from audio clips
43 German Traffic Sign Recognition Benchmark (GTSRB) Image Top-1 VGG19 Classification of traffic sign images
50 ESC-50 (Environmental Sound Classification) Audio Top-1 CNN Environmental audio sounds (e.g., dog bark, siren)
100 CIFAR-100 Image Top-1 ResNet18 Tiny natural images across 100 fine-grained classes
196 Stanford Cars Dataset Image Top-1 EfficientNetB7 Car classification
251 FoodX-251 Image Top-1 MobileNetV3 Food image classification with many categories
500 Oxford-BBC Lip Reading in the Wild (LRW) Dataset Image Top-1 ResNet-18 + Bi-GRU Lipreading Dataset
1,000 CAS-VSR-W1k Image Top-1 ResNet-18 + Bi-GRU Lipreading Dataset extended from LRW Dataset
5,000 WebVision 2.0 Image Top-5 ResNet152 Large-scale noisy web data
10,000 iNaturalist2021 Image Top-5 EfficientNetB1 Species classification from real-world observations


number of hidden layers, number of hidden units per layer, convolutional kernel size and stride (for
CNNs), number of attention heads and transformer layers (for Transformers), activation functions,
and optimizer type, were also tuned where applicable, ensuring that the reported results, presented
in Table 2, reflect the best performance achievable under each configuration.


Table 2: Comparison of accuracies and generalization metrics across datasets and model architectures. Bold: best, Italic: second-best.


**Dataset** **Metric** None ElasticNet _ψ_ spec


4.1 GENERALIZATION METRICS


4.1.1 ∆ TRAIN-VAL, ∆ VAL-TEST AND ∆ TRAIN-TEST


To quantify the amount of generalization achieved by the learned function, we report three primary
gap metrics in addition to standard Train, Validation, and Test accuracies. The Intermediate Generalization Gap (∆ Train-Val) measures the difference between training and validation accuracy, while
the True Generalization Gap (∆ Train-Test) measures the difference between training and fully unseen test data. We also present the Validation-Test Gap (∆ Val-Test), which captures the difference
between validation and test accuracy. Smaller values in all these gaps indicate better generalization,
as the model maintains consistent performance out-of-sample and avoids overfitting. Conceptually,
the ∆ Train-Val provides an estimate of how well the model generalizes during training and hyperparameter tuning, the ∆ Train-Test evaluates true generalization on completely unseen data, and the
∆ Val-Test reflects the reliability of validation performance as a proxy for final test accuracy. A


6


**Dataset** **Metric** None Elastic Net _ψ_ spec


**Dataset** **Metric** None Elastic Net _ψ_ spec


PhilUSIL


Waveform


BBC News


MNIST


HAR


20 Newsgroups


Train Acc (%) _96,60_ 96,04 **97,28**
Val Acc (%) 94,92 _94,98_ **95,81**
Test Acc (%) 93,73 **94,36** _94,27_
∆ Train-Val (%) 1,68 **1,06** _1,47_
∆ Val-Test (%) _1,19_ **0,62** 1,54
∆ Train-Test (%) _2,87_ **1,68** 3,01
GES _0.0000_ **0.1605** -0.1164
GenScore 0.0000 **1.0000** _0.0213_

Train Acc (%) _88,80_ 87,43 **89,74**
Val Acc (%) **86,80** _85,07_ _85,07_
Test Acc (%) **86,27** _84,67_ _84,67_
∆ Train-Val (%) **2,00** _2,36_ 4,67
∆ Val-Test (%) _0,53_ **0,40** **0,40**
∆ Train-Test (%) **2,53** _2,76_ 5,07
GES **0.0000** _-0.0143_ -0.1577
GenScore 0.0000 **1.0000** _0.1590_

Train Acc (%) _95,17_ 95,03 **95,91**
Val Acc (%) **94,84** _94,57_ **94,84**
Test Acc (%) _93,21_ _93,21_ **93,48**
∆ Train-Val (%) **0,33** _0,46_ 1,07
∆ Val-Test (%) _1,63_ **1,36** **1,36**
∆ Train-Test (%) _1,96_ **1,82** 2,43
GES _0.0000_ **0.1321** -0.4448
GenScore 0.0000 **1.0000** _0.1689_

Train Acc (%) **97,87** 96,72 _97,45_
Val Acc (%) 95,36 _96,44_ **97,38**
Test Acc (%) 95,22 **96,03** _95,58_
∆ Train-Val (%) 2,51 _0,28_ **0,07**
∆ Val-Test (%) **0,14** _0,41_ 1,80
∆ Train-Test (%) 2,65 **0,69** _1,87_
GES 0.0000 _0.1254_ **0.9573**
GenScore 0.0000 **1.0000** _0.3023_

Train Acc (%) **92,87** 90,89 _92,68_
Val Acc (%) _90,79_ 90,50 **91,11**
Test Acc (%) 89,00 **90,33** _89,32_
∆ Train-Val (%) 2,08 **0,39** _1,57_
∆ Val-Test (%) _1,79_ **0,17** _1,79_
∆ Train-Test (%) 3,87 **0,56** _3,36_
GES 0.0000 _0.0251_ **0.4238**
GenScore 0.0000 **1.0000** _0.2636_

Train Acc (%) **94,91** 93,39 _94,51_
Val Acc (%) 88,98 _90,55_ **93,75**
Test Acc (%) 88,47 _88,65_ **88,74**
∆ Train-Val (%) 5,93 _2,84_ **0,76**
∆ Val-Test (%) **0,51** _1,90_ 5,01
∆ Train-Test (%) 6,44 **4,74** _5,77_
GES 0.0000 _0.9549_ **2.6193**
GenScore 0.0000 **1.0000** _0.3956_


GSC v2


GTSRB


ESC-50


CIFAR100


Stanford Cars


Train Acc (%) **88,23** _87,42_ 87,20
Val Acc (%) **87,59** _86,99_ 86,79
Test Acc (%) _84,96_ **86,33** 84,40
∆ Train-Val (%) 0,64 _0,43_ **0,41**
∆ Val-Test (%) 2,63 **0,66** _2,39_
∆ Train-Test (%) 3,27 **1,09** _2,80_
GES 0.0000 _0.2951_ **0.8156**
GenScore 0.0000 **1.0000** _0.6781_

Train Acc (%) **92,20** 87,83 _91,78_
Val Acc (%) **92,07** 86,48 _89,21_
Test Acc (%) **87,51** 85,68 _87,40_
∆ Train-Val (%) **0,13** _1,35_ 2,57
∆ Val-Test (%) 4,56 **0,80** _1,81_
∆ Train-Test (%) 4,69 **2,15** _4,38_
GES 0.0000 **0.3394** _0.2163_
GenScore 0.0000 **1.0000** _0.1265_

Train Acc (%) _78,07_ **78,21** 76,36
Val Acc (%) **78,00** 75,33 _76,00_
Test Acc (%) 73,67 **74,33** _74,00_
∆ Train-Val (%) **0,07** 2,88 _0,36_
∆ Val-Test (%) 4,33 **1,00** _2,00_
∆ Train-Test (%) 4,40 _3,88_ **2,36**
GES 0.0000 _0.1192_ **1.8629**
GenScore 0.0000 _0.3729_ **1.0000**

Train Acc (%) _75,47_ **78,04** 73,40
Val Acc (%) _74,85_ **75,00** 72,54
Test Acc (%) 69,77 **72,66** _70,46_
∆ Train-Val (%) **0,62** 3,04 _0,86_
∆ Val-Test (%) 5,08 _2,34_ **2,08**
∆ Train-Test (%) 5,70 _5,38_ **2,94**
GES 0.0000 _0.3201_ **2.1156**
GenScore 0.0000 _0.2180_ **1.0000**

Train Acc (%) 83,72 **86,84** _85,53_
Val Acc (%) 82,79 **85,24** _84,44_
Test Acc (%) 77,82 **84,53** _81,46_
∆ Train-Val (%) **0,93** 1,60 _1,09_
∆ Val-Test (%) 4,97 **0,71** _2,98_
∆ Train-Test (%) 5,90 **2,31** _4,07_
GES 0.0000 _0.3332_ **2.8833**
GenScore 0.0000 _0.0964_ **1.0000**


FoodX-251


LWR


CAS-VSR-W1k


WebVision2


iNaturalist2021


Train Acc (%) **69,53** _67,41_ 65,19
Val Acc (%) **65,39** _64,51_ 63,44
Test Acc (%) _62,99_ **63,80** 62,81
∆ Train-Val (%) 4,14 _2,90_ **1,75**
∆ Val-Test (%) 2,40 _0,71_ **0,63**
∆ Train-Test (%) 6,54 _3,61_ **2,38**
GES 0.0000 _0.2287_ **0.2517**
GenScore 0.0000 _0.7660_ **1.0000**

Train Acc (%) _80,71_ **81,89** 79,18
Val Acc (%) _79,61_ **80,13** 79,11
Test Acc (%) 76,85 **78,57** _77,37_
∆ Train-Val (%) _1,10_ 1,76 **0,07**
∆ Val-Test (%) 2,76 **1,56** _1,74_
∆ Train-Test (%) 3,86 _3,32_ **1,81**
GES 0.0000 _0.3481_ **1.6188**
GenScore 0.0000 _0.5765_ **1.0000**

Train Acc (%) _49,96_ **50,49** 48,22
Val Acc (%) _44,37_ 39,48 **46,08**
Test Acc (%) _35,29_ **36,29** 34,53
∆ Train-Val (%) _5,59_ 11,01 **2,14**
∆ Val-Test (%) _9,08_ **3,19** 11,55
∆ Train-Test (%) 14,67 _14,20_ **13,69**
GES 0.0000 _0.3353_ **8.7198**
GenScore 0.0000 _0.4892_ **1.0000**

Train Acc (%) **62,91** 59,79 _61,96_
Val Acc (%) _61,06_ 56,98 **61,43**
Test Acc (%) 51,26 _56,51_ **57,65**
∆ Train-Val (%) _1,85_ 2,81 **0,53**
∆ Val-Test (%) 9,80 **0,47** _3,78_
∆ Train-Test (%) 11,65 **3,28** _4,31_
GES 0.0000 _0.1750_ **10.1245**
GenScore 0.0000 _0.5882_ **1.0000**

Train Acc (%) _69,24_ **70,43** 68,68
Val Acc (%) _64,15_ 60,14 **64,31**
Test Acc (%) 57,09 **59,37** _59,24_
∆ Train-Val (%) _5,09_ 10,29 **4,37**
∆ Val-Test (%) 7,06 **0,77** _5,07_
∆ Train-Test (%) 12,15 _11,06_ **9,44**
GES 0.0000 _0.0553_ **5.9493**
GenScore 0.0000 _0.3107_ **1.0000**


large ∆ Val-Test may suggest overfitting to the validation set or poor alignment between validation
and test distributions.


4.1.2 GENERALIZATION EFFICIENCY SCORE


To more precisely quantify generalization quality beyond raw gap reductions (i.e., the simple difference between training and test accuracy), we introduce the Generalization Efficiency Score (GES).
This metric jointly captures the extent to which a model reduces its generalization gap and retains
validation and test accuracy, relative to an unregularized baseline. GES thus penalizes models that
trivially reduce overfitting by underfitting, and favors those that achieve meaningful generalization
improvements while maintaining high performance on unseen data. A detailed derivation and motivation of GES is provided in Appendix E.


4.1.3 GENSCORE


To better quantify generalization, we additionally report the Generalization Score (GenScore), a
variance-normalized metric that assesses the smoothness of accuracy degradation from training
through validation to test. Unlike simple accuracy ratios, GenScore adaptively weights performance
gaps based on their empirical stability across models. Detailed construction and properties of GenScore are provided in Appendix F.


4.2 SUMMARY OF REGULARIZATION PERFORMANCE


Figure 1: Heatmap of the ranking function _r_ : _M × D_ _→{_ 1 _,_ 2 _,_ 3 _}_ for the regularization methods
_M_ = _{_ UnRegularised _,_ ElasticNet _,_ SpectralNet _}_ across the ordered datasets _D_ = _{Di}_ [16] _i_ =1 [.] [Each]
entry _r_ ( _m, Di_ ) denotes the discrete rank of method _m_ on dataset _Di_, where _r_ ( _m, Di_ ) = 1 indicates
the best-performing method and = 3 the worst. These rankings are derived by applying a hard
threshold to the GenScore values in Table 2, converting them into discrete ranks to enable a clear
comparative evaluation.


From Figure 1, we can see that _ψ_ spec consistently attains the best rank on large class size datasets,
underscoring its usefulness in big class size settings. ElasticNet, by contrast, beats it on datasets with
fewer classes. The Unregularised baseline uniformly ranks last, which is expected and validates the
experimental setup, since setting the regularization parameter _λ_ = 0 recovers this method in both
cases. For a clearer view of these trends, we refer to Appendix G, where Figure 8 shows the trend
in GenScore for _ψ_ spec and ElasticNet as the number of classes grows and Figure 9 presents a more
granular analysis of the effects of different regularisers _L_ [1], _L_ [2], and our proposed _ψ_ 1 and _ψ_ 2 by
examining the _α_ values. These results collectively demonstrate that _ψ_ spec offers a new and useful
regularization approach, particularly in large class size settings, beyond the regime where traditional
methods remain competitive.


From Figure 2, the moving average of the GES, reveals distinct performance profiles for each regularization method. The unregularised baseline shows a moving average that is stable at 0. This
outcome is not unexpected, as the Unregularised method serves as the baseline ( _G_ 0), and its GES
is therefore fixed at 0 by construction. This provides a crucial and stable point of reference, validating the integrity of our experimental setup. The ElasticNet method maintains a moving average


7


Figure 2: Comparison of the 3-point moving average of the Generalization Efficiency Score (GES)
for Unregularised, ElasticNet, and _ψ_ spec methods. GES is a composite metric that rewards models
for simultaneously reducing the generalization gap, retaining high accuracy, and maintaining stable
validation-test performance as detailed in Appendix E. This smoothed visualization clearly reveals
the distinct performance profiles of each regularization method as dataset class size (indexed from 1
to 16 as per Table 1 and as detailed in Figure 1’s caption, i.e same Y axis) increases.


GES that remains consistently above baseline. This suggests that while it may satisfy some of the
GES’s constituent criteria, its overall contribution is limited by something, i.e it struggles to consistently achieve a substantial positive score implying that it either does not sufficiently reduce the
generalization gap or fails to retain accuracy in a manner that yields a meaningful product across
all three GES factors. In contrast, _ψ_ spec demonstrates a compelling and non-trivial performance. It
has a poor early performance detailed by it’s negative value but its moving average GES exhibits a
pronounced upward trajectory, particularly as we transition to datasets of higher complexity. This
behaviour is likely directly attributable to its regularization mechanism. The method’s effectiveness
becomes increasingly evident as the datasets grow in number of classes. For further details, refer to
Figure 10, in Appendix G, which provides a point-by-point view of the GES scores on a logarithmic
scale against class size, highlighting the marked fluctuations that characterize the performance of
each method prior to smoothing. In these scenarios, the ability of _ψ_ spec to simultaneously reduce the
generalization gap, retain high accuracy, and ensure stable performance on held-out data becomes a
powerful asset, which the adaptive components of the GES metric keenly reward. This improvement
provides strong empirical support for our hypothesis that characteristic function based regularization
is a potent tool for achieving reliable generalization in challenging, large-scale settings.


5 CONCLUSION


In this work, we introduced a new class of regularization based on the characteristic function. We
demonstrated a specific implementation of this framework by modeling the output layer of a neural
network as a function of random variables and provided the necessary theoretical proofs to validate
its properties. Our empirical experiments on a diverse selection of real-world datasets yielded results
that directly align with our theoretical claims: the empirical data suggests that our method has a
robust scaling behaviour, where the it’s benefit becomes increasingly pronounced as the class size
grows. These compelling results demonstrate that our characteristic function-based regularization
is a promising method for future exploration and use, especially given the trend toward large-scale,
high-parameter models, where its ability to promote another layer of robust generalization could
prove to be greatly useful.


8


REFERENCES


Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna:
A next-generation hyperparameter optimization framework. In _Proceedings_ _of_ _the_ _25th_ _ACM_
_SIGKDD international conference on knowledge discovery & data mining_, pp. 2623–2631, 2019.


Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein generative adversarial networks.
In _International conference on machine learning_, pp. 214–223. PMLR, 2017.


Olivier Bournez. lix.polytechnique.fr. [https://www.lix.polytechnique.fr/](https://www.lix.polytechnique.fr/~bournez/load/MPRI/Cours-2024-MPRI-partie-I-goodMPRI.pdf)
˜ [[bournez/load/MPRI/Cours-2024-MPRI-partie-I-goodMPRI.pdf][, 2024.]](https://www.lix.polytechnique.fr/~bournez/load/MPRI/Cours-2024-MPRI-partie-I-goodMPRI.pdf) [[Ac-]
cessed 02-10-2024].


Svetlana Boyarchenko and Sergei Levendorskii. Simulation of a l _\_ ’evy process, its extremum, and
hitting time of the extremum via characteristic functions. _arXiv preprint arXiv:2312.03929_, 2023.


G Cantor. Uber eine elementare frage der mannigfaltigkeitslehre, jahresbericht der deutschen
mathematiker-vereiningung 1: 75–78, 1932.


Rogero Cotes. Logometria auctore rogero cotes, trin. coll. cantab. soc. astr. and ph. exp. professore
plumiano, and r. s. s. _Philosophical Transactions (1683-1775)_, 29:5–45, 1714. ISSN 02607085.
[URL http://www.jstor.org/stable/103030.](http://www.jstor.org/stable/103030)


Richard Dedekind. _Essays on the Theory of Numbers_ . Courier Corporation, 2012.


Leonhard Euler. Introductio in analysin infinitorum, volume 2. _Lausanæ: Apud Marcum-Mich ælem_
_Bousquet and Socios_, 1748.


Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. _arXiv preprint arXiv:2010.01412_, 2020.


Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio. Generative adversarial nets. _Advances in neural information_
_processing systems_, 27, 2014.


Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Sch¨olkopf, and Alexander Smola.
A kernel two-sample test. _The journal of machine learning research_, 13(1):723–773, 2012.


Arthur E Hoerl and Robert W Kennard. Ridge regression: Biased estimation for nonorthogonal
problems. _Technometrics_, 12(1):55–67, 1970.


Gero Junike. Precise quantile function estimation from the characteristic function. _Statistics_ _&_
_Probability Letters_, 222:110395, 2025.


Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S Schoenholz, Jeffrey Pennington, and Jascha
Sohl-Dickstein. Deep neural networks as gaussian processes. _arXiv preprint arXiv:1711.00165_,
2017.


Berent AS Lunde, Tore S Kleppe, and Hans J Skaug. Saddlepoint-adjusted inversion of characteristic [˚]
functions. _arXiv preprint arXiv:1811.05678_, 2018.


Alexandre Lyapunov. Sur une proposition de la th´eorie des probabilit´es. _Izvestiya_ _Rossiiskoi_
_Akademii Nauk. Seriya Matematicheskaya_, 13(4):359–386, 1900.


Alexander G de G Matthews, Mark Rowland, Jiri Hron, Richard E Turner, and Zoubin Ghahramani.
Gaussian process behaviour in wide deep neural networks. _arXiv_ _preprint_ _arXiv:1804.11271_,
2018.


Takakazu Mori, Yoshiki Tsujii, and Mariko Yasugi. Computability of probability distributions and
distribution functions. In _6th International Conference on Computability and Complexity in Anal-_
_ysis (CCA’09)(2009)_, pp. 185–196. Schloss Dagstuhl–Leibniz-Zentrum f¨ur Informatik, 2009.


Lutz Prechelt. Automatic early stopping using cross validation: quantifying the criteria. _Neural_
_networks_, 11(4):761–767, 1998.


9


Lutz Prechelt. Early stopping-but when? In _Neural_ _Networks:_ _Tricks_ _of_ _the_ _trade_, pp. 55–69.
Springer, 2002.


Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. _The_ _journal_ _of_ _machine_
_learning research_, 15(1):1929–1958, 2014.


G´abor J Sz´ekely, Maria L Rizzo, et al. Testing for equal distributions in high dimension. _InterStat_,
5(16.10):1249–1272, 2004.


Robert Tibshirani. Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical_
_Society Series B: Statistical Methodology_, 58(1):267–288, 1996.


Vladimir Vapnik. The method of structural minimization of risk. In _Estimation_ _of_ _Dependences_
_Based on Empirical Data_, pp. 232–266. Springer, 2006.


Richard L Warr. Numerical approximation of probability mass functions via the inverse discrete
fourier transform. _Methodology and Computing in Applied Probability_, 16(4):1025–1038, 2014.


Klaus Weihrauch. _Computable_ _analysis:_ _an_ _introduction_ . Springer Science & Business Media,
2012.


Viktor Witkovsky. Charfuntool: The characteristic functions toolbox. _Matlab_ _R_ _File_ _Ex-_
_change:_ _Script cfX_ ~~_P_~~ _DF. m for Exponential Distributions in the CF_ ~~_R_~~ _epository. Available online:_
_https://www._ _mathworks._ _com/matlabcentral/fileexchange/64400-charfuntool–the-characteristic-_
_functions-toolbox (accessed on 19 March 2018)_, 2024.


Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical
risk minimization. _arXiv preprint arXiv:1710.09412_, 2017.


Hui Zou and Trevor Hastie. Regularization and variable selection via the elastic net. _Journal of the_
_Royal Statistical Society Series B: Statistical Methodology_, 67(2):301–320, 2005.


A APPENDIX


A.1 SATISFACTION OF LYAPUNOV CONDITION


1
lim
_n→∞_ _s_ [2+] _n_ _[δ]_


Without Loss of Generality, let _δ_ = 2:


1
lim
_n→∞_ _s_ [4] _n_


**Definition 5.** Suppose there exists a sequence of independent random variables _{Y_ 1 _, Y_ 2 _, ...Yn}_, with
finite mean and variance, we can expect that the growth of the moments are limited by the Lyapunov
Condition.


1
lim
_n→∞_ _s_ [2+] _n_ _[δ]_


_n_


- 
E _|Yi −_ _µi|_ [2+] _[δ]_ [�] = 0 (20)

_i_ =1


**Definition 6.** For some sequence of independent Bernoulli random variables _{X_ 1 _, X_ 2 _, ..Xn}_, such
that
_Xi_ _∼_ _Bernoulli_ ( _pi_ ) (21)


P( _Xi_ = 1) = _p_, 0 _≤_ _p ≤_ 1, _E_ ( _Xi_ ) = _p_, _V ar_ ( _Xi_ ) = _pi_ (1 _−_ _pi_ )


**Proposition** **2.** Under most conditions, the Lyapunov CLT condition holds for Bernoulli Random
Variables.


_Proof._


_n_

- _E_ [( _|Xi −_ _µi|_ [2+] _[δ]_ )] (22)


_i_ =1


_n_

- _E_ [( _|Xi −_ _µi|_ [4] )] (23)


_i_ =1


10


By replacing _µi_ with _E_ ( _Xi_ ):


_||_ **x** _||p_ =


1
= lim
_n→∞_ _s_ [4] _n_


_n_

- _E_ [( _|Xi −_ _E_ ( _Xi_ ) _|_ [4] )] (24)


_i_ =1


By the Law of the Unconscious Statistician (LOTUS):


_Xi_ =1

- [( _|Xi −_ _E_ ( _Xi_ ) _|_ [4] )] (25)


_Xi_ =0


1
= lim
_n→∞_ _s_ [4] _n_


By definition of Bernoulli distribution:


_n_


_i_ =1


1
= lim
_n→∞_ _s_ [4] _n_


With reference to equation 2:


_n_
�(0 _−_ _pi_ ) [4] (1 _−_ _pi_ ) + (1 _−_ _pi_ ) [4] ( _pi_ ) (26)


_i_ =1


1
= lim
_n→∞_ ( [�] _[n]_ _i_ =1 _[σ]_ [2][)][2]


_n_

- _p_ [4] _i_ [(1] _[ −]_ _[p][i]_ [) + (1] _[ −]_ _[p][i]_ [)][4][(] _[p][i]_ [)] (27)

_i_ =1


By the Variance described for Bernoulli Random Variables, _σ_ [2] = _pi_ (1 _−_ _pi_ ):


1
= lim
_n→∞_ ( [�] _[n]_ _i_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [)))][2]


_n_

- _p_ [4] _i_ [(1] _[ −]_ _[p][i]_ [) + (1] _[ −]_ _[p][i]_ [)][4][(] _[p][i]_ [)] (28)

_i_ =1


Since parameter 0 _≤_ _p ≤_ 1, we can claim _p_ [4] _i_ _[≤]_ _[p][i]_ [and][ (1] _[ −]_ _[p][i]_ [)][4] _[≤]_ [(1] _[ −]_ _[p][i]_ [)][:]


1
_≤_ lim
_n→∞_ ( [�] _[n]_ _i_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [)))][2]


_n_

- _pi_ (1 _−_ _pi_ ) + (1 _−_ _pi_ )( _pi_ ) (29)


_i_ =1


1
= lim
_n→∞_ ( [�] _[n]_ _i_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [)))][2]


_n_

- 2 _pi_ (1 _−_ _pi_ ) (30)


_i_ =1


By Linearity of the Sum,


As _n →∞_,


= lim 2 [�] _i_ _[n]_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [))] (31)
_n→∞_ ( [�] _[n]_ _i_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [)))][2]

2
= _n_ lim _→∞_ - _n_ (32)
_i_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [)]


∵


_n_
�( _pi_ (1 _−_ _pi_ )) _→∞_ (33)


_i_ =1


We have
2
_n_ lim _→∞_               - _ni_ =1 [(] _[p][i]_ [(1] _[ −]_ _[p][i]_ [))] [= 0] (34)

as desired.


B DISTANCE MEASURES


In this section, we extend the concept of _Lp_ norms to measure the differences between the distributions _ϕD_ and _ϕN_ (0 _,_ 1). We define the distance function _d_ ( _ϕD, ϕN_ (0 _,_ 1)) by calculating the pointwise
differences between the two distributions and applying the _Lp_ norms.


We start with the general definition of the _Lp_ norm for a vector **x** = ( _x_ 1 _, x_ 2 _, . . ., xn_ ):


   - _p_ [1]
��
_|xk|_ _[p]_ _,_ _p ≥_ 1 _._ (35)


11


Extending the definition of the standard _L_ 1 norm, which provides a measure based on the absolute
differences:


Finally, we define a convex combination of _ψ_ 1 and _ψ_ 2, analogous to the ElasticNet regularization.
This formulation, we call SpectralNet [2], allows interpolation between the _ψ_ 1 and _ψ_ 2 :


_ψ_ spec = _d_ spec( _ϕD, ϕN_ (0 _,_ 1)) = _α ψ_ 1 + (1 _−_ _α_ ) _ψ_ 2 _,_ _α ∈_ [0 _,_ 1] _._ (38)


Here, _α_ governs the trade-off: _α_ = 1 recovers the pure _ψ_ 1 distance, while _α_ = 0 corresponds to the
_ψ_ 2 distance. Intermediate values yield a hybrid measure, capturing both distributional discrepancies.


While geometric distance measures could potentially yield greater performance, we have chosen to
focus on these straightforward metrics to provide a gentle introduction to the topic and methodology
discussed in this paper.


C CHARACTERISTIC FUNCTION SIMULATION FIGURES


C.1 CHARACTERISTIC FUNCTION OF NORMAL AND BERNOULLI DISTRIBUTION


Figure 3: Plot of Normal and Bernoulli Characteristic Functions (Only Real Part)


The figure 3 shows the plot of real part of the Normal and Bernoulli Distribution. We thought this
would be apt to add as this is to give a visual intuition for the reader for how these functions look
when graphed as there is not much literature regarding visualising them.


12


_ψ_ 1 = _d_ 1( _ϕD, ϕN_ (0 _,_ 1)) = _||ϕD −_ _ϕN_ (0 _,_ 1) _||_ 1 =


_∞_

- _|ϕD_ ( _uk_ ) _−_ _ϕN_ (0 _,_ 1)( _uk_ ) _|._ (36)


_k_ = _−∞_


Next, we extend the _L_ 2 norm, which measures the Euclidean distance between the pointwise differences:


_ψ_ 2 = _d_ 2( _ϕD, ϕN_ (0 _,_ 1)) = _||ϕD −_ _ϕN_ (0 _,_ 1) _||_ 2 =


~~�~~ - _∞_
�� - _|ϕD_ ( _uk_ ) _−_ _ϕN_ (0 _,_ 1)( _uk_ ) _|_ [2] _._ (37)

_k_ = _−∞_


Figure 4: Extended Plot of Normal and Bernoulli Characteristic Function (Includes Imaginary Part)


C.2 IMAGINARY PART INCLUSIVE CHARACTERISTIC FUNCTION OF NORMAL AND
BERNOULLI DISTRIBUTION


The figure 4 shows the plot of the Normal and Bernoulli Distribution inclusive of the imaginary part.
It is interesting to note the imaginary part is on the zero line for the Normal. As for the Bernoulli
we can see a ”phase” difference between the Imaginary and the Real Part.


If one would like to explore why, they can derive insight using the following as a starting point:


**Definition 7.** Euler’s formula (Euler, 1748) (Cotes, 1714) states that for any real number _x_ :


_e_ _[ϑx]_ = cos( _x_ ) + _ϑ_ sin( _x_ ) (39)


This formula can be used to express complex exponentials in terms of trigonometric functions.


**Definition 8.** Using equation 5 and definition 7, the characteristic function of a random variable _X_
is defined as:


_ϕ_ ( _u_ ) = E[ _e_ _[ϑuX]_ ] = E[cos( _uX_ )] + _ϑ_ E[sin( _uX_ )] (40)


where the real and imaginary parts of the characteristic function are:


Re( _ϕ_ ( _tu_ ) = E[cos( _uX_ )] (41)


Im( _ϕ_ ( _u_ )) = E[sin( _uX_ )] (42)


C.3 ZOOMED OUT VIEW TO OBSERVE PERIODICITY


The figure 5 shows that the Normal Characteristic Function does not seem to periodic unlike the
Bernoulli Characteristic Function which seems to have a defined _π_ -periodic structure It also shows
that the Normal Characteristic Function is concentrated within the _−_ 2 _π_ to 2 _π_ region. (Which motivated our choice in the numerics section D).


13


Figure 5: Plot of Normal and Bernoulli


Figure 6: Numerical Simulation Plot of the Convergence


C.4 BEHAVIOUR OF THE CHARACTERISTIC FUNCTION WHEN JUST ADDING BERNOULLI
VARIABLES TOGETHER MINDLESSLY


The figure 6 is generated random generated _pi_ values for a [�] _i_ _[N]_ =1 [Bernoulli] [Distributions.] [It] [is]
interesting to note how just adding the Bernouli’s will result in it resulting in a convergence towards
the zero line.


14


Figure 7: Numerical Simulation Plot of just adding Bernoullis


C.5 NUMERICAL SIMULATION OF CONVERGENCE DESCRIBED IN PROPOSITION 1


The figure 7 is generated random generated _pi_ values for a linear combination _N_ Bernoulli Distributions which are added according to the _ℵ_ model described in definition 3.


D NUMERICAL CONSIDERATIONS


The characteristic function is generally a considered a ”pure mathematical tool” whereby it’s continuous nature presents significant challenges when implemented in discrete computational environments. Modern computers rely on finite precision arithmetic, which inherently restricts the exact
representation of continuous functions, including characteristic functions. This means that we have
to formulate discretizations based on some assumptions to integrate the characteristic function into
a practical regularization algorithm for machine learning, enabling it to operate and execute within
finite time.


Specifically, we will have to restrict it’s domain to a ”good enough” range since _t_ _∈_ R and the
associated infinite nature of the reals. In other words, abstractly the problem is then as follows (with
the help of some informal proof sketches for the sake of brevity due to page limit and for the reader’s
sanity) :

**Proposition 3.** The set of real numbers R is uncountably infinite.

**Informal Proof Sketch 1.** This can be shown using Cantor’s famous Diagonal Argument. Assume
for contradiction that R is countable. Then we can list all real numbers in the interval [0 _,_ 1] as
_r_ 1 _, r_ 2 _, r_ 3 _, . . ._ . We can then construct a new real number _r_ by taking the diagonal of this list and
changing each digit, ensuring that _r_ differs from each _rn_ at the _n_ -th digit. Therefore, _r_ cannot be
in our original list, contradicting the assumption that we had listed all real numbers. Thus, R is
uncountably infinite. (Cantor, 1932)


2We refer to this as ”Spectral” since the distance is constructed from characteristic functions, which are
Fourier transforms of probability distributions. This naturally evokes the idea of ”spectral concepts” into play
especially in light of the term ”eigen” (German for ”characteristic”) and hence motivates the name.


15


**Proposition 4.** The set of real numbers R is complete.

**Informal Proof Sketch 2.** The completeness of R can be demonstrated using Dedekind’s cuts. A
Dedekind cut partitions the rational numbers into two non-empty sets _A_ and _B_, where all elements
of _A_ are less than all elements of _B_ . For any non-empty set of rationals that is bounded above, there
exists a least upper bound (supremum) in R. This property ensures that every Cauchy sequence of
real numbers converges to a real number, establishing the completeness of R. (Dedekind, 2012)

**Proposition 5.** The set of computable real numbers is countably infinite.

**Informal Proof Sketch 3.** The set of computable real numbers can be described as those numbers
for which there exists a finite algorithm (Turing machine) that can produce their digits. Since the
set of all finite algorithms is countable, it follows that the set of computable real numbers is also
countable.(Bournez, 2024)(Weihrauch, 2012)

**Proposition 6.** The set of computable real numbers is not complete.

**Informal** **Proof** **Sketch** **4.** To see this, consider the sequence of computable numbers defined
by _rn_ = _n_ 1 [,] [which] [converges] [to] [0][.] [Although] [0] [is] [a] [limit] [point] [of] [the] [sequence,] [it] [is] [not] [com-]
putable because there is no finite algorithm that can output the exact value of 0. This demonstrates that there exist Cauchy sequences of computable real numbers that do not converge to a computable limit, thereby showing that the set of computable real numbers is not complete. (Bournez,
2024)(Weihrauch, 2012)


It becomes evident that propositions 3, 4, 5, and 6 present significant challenges in computing the
desired function _ϕ_ ( _t_ ) especially on a Discrete Dynamical System like the modern computer we use.
To address this, we have adopted a strategy of restricting to a finite domain of _t ∈_ [ _−_ 2 _π,_ 2 _π_ ] where
we discretize this interval into _n_ = 1000 finite segments, which can be easily accomplished using
a linear space function such as numpy.linspace in python or similar methods on any modern
programming languages.


The rationale for selecting the interval [ _−_ 2 _π,_ 2 _π_ ] is motivated by the analysis of the figures 3 and 5 in
the appendix of the characteristic function for the standard normal distribution, as well as the set of
convergence graphs for the _ℵ_ -modelled linear combinations of Bernoulli random variables observed
in 7. The region of primary interest lies within this interval, and while any variations outside this
interval may be potentially significant under certain circumstances, we can effectively treat them as
an acceptable level of statistical noise, we are willing to quantified by some _ϵ_ . This allows us to
disregard this noise in the context of testing viability, though it may come at the expense of some
regularization ”performance”.


There is no universally ”correct” range or sample size ( _n_ ); however, for the purposes of our experimentation, we consider this choice to be sufficient.


E CRAFTING A COMPOSITE SCALAR MEASURE TO BETTER REFLECT
GENERALIZATION GAP


E.1 MOTIVATION AND OVERVIEW


We introduce the **Generalization** **Efficiency** **Score** **(GES)**, a scalar metric designed to evaluate
how effectively a model generalizes beyond the training distribution. Traditional metrics such as
the raw generalization gap, defined as the difference between training and test accuracy, can be
misleading. For instance, underfitting models often show small generalization gaps despite poor test
performance. Likewise, some overfit models may achieve high accuracy on held-out data despite
exhibiting larger gaps.


GES addresses this by incorporating three key factors:


    - Relative reduction in generalization gap compared to a baseline,

    - Retention of accuracy on both validation and test sets, and

    - A penalty for test-validation disagreement.


This composite metric provides a more holistic view of generalization by rewarding models that
achieve smaller generalization gaps _and_ maintain stable, high performance on unseen data.


16


E.2 FORMAL DEFINITION


Let _Mi_ denote a trained model indexed by _i ∈{_ 0 _,_ 1 _, . . ., N_ _}_, evaluated on the following datasets:


    - _D_ train: training set,

    - _D_ val: validation set,

    - _D_ test: test set.


We hence get accuracies [3] of model _Mi_ on these sets:

Acc [(] train _[i]_ [)] _[,]_ Acc [(] val _[i]_ [)] _[,]_ Acc [(] test _[i]_ [)] _[.][ ∈]_ [[0] _[,]_ [ 100]]


We define the generalization gap of model _Mi_ as:

_Gi_ := Acc [(] train _[i]_ [)] _[−]_ [Acc] test [(] _[i]_ [)] _[.]_


Let _M_ 0 be a fixed _baseline model_ (typically unregularized), with:

_G_ 0 := Acc [(0)] train _[−]_ [Acc] test [(0)] _[.]_


We now define the three components of GES:


**Gap Factor:** Measures improvement in generalization gap:

GapFactor _i_ := _[G]_ [0] _G_ _[ −]_ 0 _[G][i]_ _._


**Accuracy Retention Factor:** Measures retained accuracy relative to the baseline:

AccFactor _i_ := _α ·_ [Acc] val [(] _[i]_ [)] + (1 _−_ _α_ ) _·_ [Acc] test [(] _[i]_ [)] _._
Acc [(0)] val Acc [(0)] test


Here, _α ∈_ [0 _,_ 1] is a weight that balances validation and test accuracy. We compute it automatically
based on the variance of test-validation discrepancies across models:


Var( _d_ 1 _, . . ., dN_ ) ( _i_ )
_α_ := 0 _._ 5 _−_ 0 _._ 5 _·_ Var( _d_ 1 _, . . ., dN_ ) + _ϵ_ [with] _[ d][i]_ [:=] ���Acctest _[−]_ [Acc][(] val _[i]_ [)] ��� _,_

where _R_ val and _R_ test are vectors of accuracies across models and _ϵ_ = 10 _[−]_ [6] is a small constant for
numerical stability.


Intuitively, when the difference between test and validation accuracies exhibits low variance,
the model performances are consistent, and we assign roughly equal weight to both accuracies
( _α_ _≈_ 0 _._ 5). Conversely, if the discrepancy varies widely, indicating that validation accuracy may
be unstable or less reliable, _α_ shifts closer to 0, placing greater emphasis on the test accuracy, which
is considered a more robust indicator of generalization. This adaptive weighting mechanism ensures the retention factor prioritizes the most trustworthy signal, yielding a more reliable measure of
accuracy retention.


**Penalty Factor:** Penalizes disagreement between validation and test performance:


( _i_ )                                 - _k_
Penalty _i_ := ����Acctest _[−]_ [Acc][(] val _[i]_ [)] ��� _,_


where the exponent _k_ is determined from the variance in discrepancies across all models:


Var( _d_ 1 _, . . ., dN_ ) ( _i_ )
_k_ := 1 + Var( _d_ 1 _, . . ., dN_ ) + _ϵ_ _[,]_ with _di_ := ���Acctest _[−]_ [Acc][(] val _[i]_ [)] ��� _._


3All accuracy values are originally in the range [0 _,_ 1], but are rescaled to the [0 _,_ 100] range for metric
computation. This scaling mitigates unintended numerical effects arising from exponentiation of values less
than one, which can otherwise lead to disproportionately small quantities and instability.


17


The penalty factor quantifies the disagreement between validation and test accuracies for each
model, raising this discrepancy to the power _k_ to modulate its influence. The exponent _k_ is adaptively determined based on the variance of these discrepancies across all models. When the variance
is low, indicating consistent agreement between validation and test accuracies, _k_ remains close to
1, applying a moderate penalty. However, when the variance is high, reflecting unstable or divergent performance between validation and test sets, _k_ increases towards 2, amplifying the penalty on
large discrepancies. This adaptive exponent ensures that models exhibiting greater inconsistency are
penalized more heavily, thereby promoting reliability and stability in the overall scoring.


E.3 FINAL METRIC


Combining the components, the Generalization Efficiency Score for model _Mi_ is given by:


GES2 = ( _−_ 0 _._ 1884) _·_ 0 _._ 9984 _÷_ 0 _._ 2916 _≈−_ 0 _._ 645


A negative score reflects worse generalization gap and marginal validation/test mismatch compared
to the baseline.


18


(43)


E.4 INTERPRETATION AND PROPERTIES


This construction ensures that:


    - GES0 = 0 by design (the baseline model),

    - GES _i_ _>_ 0 indicates an improvement in generalization _and_ accuracy,

    - GES _i_ _<_ 0 indicates worse generalization, worse accuracy, or unstable validation-test behavior,


The Gap Imporvement Factor quantifies improvement in overfitting, the Accuracy Retention Factor
encourages retention of predictive power, and the Penalty discourages large discrepancies between
validation and test accuracy, which could signal instability or poor generalization.


E.5 EXAMPLE CALCULATION


Consider a baseline model _M_ 0 with:


Acc [(0)] train [= 0] _[.]_ [9185] _[,]_ Acc [(0)] val [= 0] _[.]_ [9061] _[,]_ Acc [(0)] test [= 0] _[.]_ [8893] _[.]_


This gives _G_ 0 = 0 _._ 9185 _−_ 0 _._ 8893 = 0 _._ 0292.


Now consider a second model _M_ 2 with:


Acc [(2)] train [= 0] _[.]_ [9300] _[,]_ Acc [(2)] val [= 0] _[.]_ [9007] _[,]_ Acc [(2)] test [= 0] _[.]_ [8953] _[,]_
_G_ 2 = 0 _._ 9300 _−_ 0 _._ 8953 = 0 _._ 0347 _,_ _d_ 2 = 100 _· |_ 0 _._ 8953 _−_ 0 _._ 9007 _|_ = 0 _._ 54 _._


Assuming _α_ = 0 _._ 6 and _k_ = 2 (for illustration), we compute:


GapFactor2 = [0] _[.]_ [0292] 0 _._ 0292 _[ −]_ [0] _[.]_ [0347] _≈−_ 0 _._ 1884


AccFactor2 = 0 _._ 6 _·_ [0] _[.]_ [9007]


[0] _[.]_ [9007] [0] _[.]_ [8953]

0 _._ 9061 [+ 0] _[.]_ [4] _[ ·]_ 0 _._ 8893


0 _._ 8893 _[≈]_ [0] _[.]_ [5956 + 0] _[.]_ [4028 = 0] _[.]_ [9984]


Penalty2 = (0 _._ 54) [2] = 0 _._ 2916


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


E.6 APPLICATION IN OUR STUDY


We use the Generalization Efficiency Score as a diagnostic in ablation and regularization studies. By
choosing the unregularized model as a reference, we can meaningfully assess which techniques truly
enhance generalization, rather than simply reducing capacity or trivially lowering the generalization
gap. GES helps us focus on models that maintain accuracy while improving robustness on unseen
data.


F VARIANCE-NORMALIZED METRIC FOR GENERALIZATION PERFORMANCE


In assessing model generalization, prevailing metrics often reduce the intricate geometry of performance degradation across data splits to a scalar ratio, typically, test accuracy divided by training
accuracy. While intuitive, such metrics fail to account for the asymmetric roles played by different
evaluation regimes (training, validation, test), and more critically, for the _relative stabilities_ of these
regimes across candidate models or hyperparameter configurations.


We propose a variance-normalized generalization score, denoted as **GenScore**, that measures the
smoothness and consistency of performance degradation from training to validation to test. Unlike
traditional metrics, GenScore respects the empirical variance structure of model behavior across
evaluation splits and adapts its penalization accordingly.


Let each model instance _m_ be associated with a triplet of accuracies as follows:


**a** [(] _[m]_ [)] = (Acctrain _,_ Accval _,_ Acctest) _,_


with natural deltas defined as:

∆ [(] tv _[m]_ [)] = Acctrain _−_ Accval _,_ ∆ [(] vt _[m]_ [)] = Accval _−_ Acctest _,_ ∆ [(] tt _[m]_ [)] = Acctrain _−_ Acctest _._


These quantities encode distinct aspects of generalization: ∆tv captures overfitting to training data,
∆vt reflects sensitivity to unseen but proximate data, and ∆tt represents end-to-end generalization
collapse.


To define a principled aggregation of these deltas into a scalar score, we take a data-adaptive approach. Let _σ_ tv [2] _[, σ]_ vt [2] _[, σ]_ tt [2] [denote] [the] [sample] [variances] [of] [these] [deltas] [across] [a] [fixed] [set] [of] [model]
configurations _M_, e.g., top-performing runs or a representative validation batch. We define inversevariance weights:


which serve as soft attention coefficients over the deltas, allocating greater influence to more stable
differences. These weights reflect a natural statistical heuristic that stable gaps are more trustworthy
indicators of systemic generalization behaviour.


The raw generalization penalty for model _m_ is then:

_L_ [(] gen _[m]_ [)] [=] _[ w]_ [tv] _[·]_ [ (∆][(] tv _[m]_ [)][)][2][ +] _[ w]_ [vt] _[·]_ [ (∆][(] vt _[m]_ [)][)][2][ +] _[ w]_ [tt] _[·]_ [ (∆][(] tt _[m]_ [)][)][2] _[,]_


which is minimized when performance degrades smoothly and uniformly. Finally, we normalize
this penalty across _M_ to obtain a bounded generalization score between [0,1] as follows:

_L_ [(] gen _[m]_ [)] _[−]_ [min] _m_ _[′]_ _∈M_ _[L]_ [(] gen _[m][′]_ [)]
GenScore [(] _[m]_ [)] = 1 _−_ _,_
max _m_ _[′]_ _∈M L_ [(] gen _[m][′]_ [)] _[−]_ [min] _m_ _[′]_ _∈M_ _[L]_ [(] gen _[m][′]_ [)] [+] _[ ε]_

with _ε_ = 10 _[−]_ [6] ensuring numerical stability. This final score lies in the interval [0 _,_ 1], with higher
values indicating smoother and more robust generalization. Models with high training accuracy but
erratic validation or test behavior are sharply penalized, while those that exhibit graceful degradation
are rewarded.


We find that GenScore correlates more reliably with downstream robustness metrics (e.g., performance under distribution shift) than na¨ıve ratios or accuracy gaps. Its variance-normalized structure
allows it to adapt to the particular generalization geometry of the task, architecture, and dataset
under study.


19


_wi_ =


1

- _σi_ [2] 1 _,_ _i ∈{_ tv _,_ vt _,_ tt _},_

_j_ _σj_ [2]


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


G FURTHER FIGURES


The figures in this section provide additional miscellaneous insight into the performance and internal
behaviour of the different regularization strategies explored in the main text.


Figure 8: **GenScore** **over** **number** **of** **classes** **(log** **scale)** . This plot visualises the GenScore performance of three regularization methods (UnRegularised, ElasticNet, SpectralNet) across datasets
ordered by increasing number of classes (x-axis in log scale). SpectralNet exhibits a clear upward
trend, with GenScore increasing sharply and saturating at 1 as class cardinality increases. ElasticNet
achieves the highest GenScore on low-class datasets, but its performance degrades on high-class settings. The Unregularised baseline remains consistently near zero across all datasets. This separation
supports the conclusion that SpectralNet is uniquely effective in high-class regimes.


Figure 8 plots the GenScore values for Unregularised, ElasticNet, and SpectralNet methods across
datasets ordered by increasing class cardinality (x-axis on a log scale). The results reveal a clear
separation in behaviour, whereby ElasticNet dominates in low-class-count regimes, but its performance deteriorates as the number of classes increases. In contrast, SpectralNet exhibits a monotonic
increase in GenScore, approaching a saturation point near 1 (implying it is the best performing after
normalization). This suggests that SpectralNet is particularly well-suited for high-class scenarios.
The Unregularised baseline remains at zero throughout, as expected. This plot underpins the discrete
rankings shown in Figure 1 by offering the continuous underlying signal prior to thresholding and
shows the second degree of the ”variability” of the performance too where we can see ”how much
better or worse” each method is performing relative to each other and the baseline in the space of
the GenScore as defined in Appendix F.


Figure 9: **Discrete** **ranking** **of** **regularization** **methods** **using** **learned** _α_ **values.** This figure displays the ranking function _r_ ( _m, Di_ ) for five regularization strategies across datasets of increasing
class complexity. The methods include Baseline ( _λ_ = 0), _L_ [1], _L_ [2], and the proposed _ψ_ 1, _ψ_ 2 regularisers. Rankings are derived from the relative magnitude of the learned regularization weight _α_
assigned to each method within the same model. This granular perspective, derived from interpolation factor _α_, offers an extra insight into the different regimes.


To complement the performance-based perspective, Figure 9 provides a more granular look at how
different regularization terms influence the model internally. Here, we report the ranking of regularisers according to the linear interpolant coefficient _α_ associated with each term, within a shared
architecture. Specifically, we compare the standard _L_ [1] and _L_ [2] penalties with our proposed spectral
regularisers _ψ_ 1 and _ψ_ 2. The results show that there is a skew towards _L_ [1] and _L_ [2] in simpler low class
size tasks, while _ψ_ 1 and _ψ_ 2 emerge randomly dominant in datasets with many classes. This rank

20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


Figure 10: A point-by-point view of the Generalization Efficiency Score (GES) for Unregularised,
ElasticNet, and SpectralNet methods. The GES is a composite metric that rewards models for
simultaneously reducing the generalization gap, retaining high accuracy, and maintaining stable
validation-test performance. This visualization highlights the distinct performance of each regularization method as dataset complexity, indexed by the number of classes, increases on a logarithmic
scale.


ing reflects the model’s own inductive bias, revealing which ”flavour” of regularization it favours in
different dataset settings. We caution, however, that these rankings should not be over-interpreted
in isolation: in many cases, the top-ranked methods had _α_ values close to others (e.g., near 0.5), indicating that the model viewed several regularisers as comparably useful. The ranking here reflects
relative ordering rather than the magnitude of preference.


Figure 10 provides a greater analysis of the GES scores as a function of dataset complexity, which is
indexed by the number of classes on a logarithmic scale. The Unregularised baseline, serving as our
reference, demonstrates a GES that remains consistently at zero, obviously indicating a persistent
lack of generalization across all class sizes as per definition of GES in Appendix E. The ElasticNet
method exhibits a similar behavior, maintaining a GES that largely hovers above zero, with only minor fluctuations, suggesting a limited capacity to leverage its regularization for a more meaningful
improvement in generalization as class sizes changes. In contrast, the performance of SpectralNet is
markedly distinct. Its GES score is initially negative, indicating a worse performance than the baseline on datasets with a small number of classes, probably a consequence of the central limit theorem,
i.e in this case we have not hit a sufficiently large enough sample in the ”class size space”. Thus,
in ultra-low complexity settings, where simpler regularization is sufficient, our approach imposes
an extremely poor form of regularization, leading to a negative GES [4] . However, its GES demonstrates a dramatic increase as the class count rises [5] . This trend underscores the unique efficacy of
characteristic function-based regularization, whose ability to promote stability and reduce the generalization gap becomes increasingly pronounced and effective in high-dimensional settings where
the sufficiently large class count satisfies the conditions for the Central Limit Theorem and allows
the regularization to kick into gear more effectively.


4The Figure 7 in Appendix C can give a somewhat visual idea that explains in some way the discrepancy
that happens in the ultra-low class size setting.
5Given the nature of the data, the observed increase in performance on datasets with a larger number of
classes is likely due to the regularization’s mechanism agreeing more favourably with the Central Limit Theorem assumption as the number of classes tends towards infinity.


21