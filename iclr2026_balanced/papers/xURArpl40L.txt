# IS THE PREDICTION SET SIZE WELL-CALIBRATED? A CLOSER LOOK AT UNCERTAINTY IN CONFORMAL PREDICTION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Given its flexibility and low computation, conformal prediction (CP) has become
one of the most popular uncertainty quantification methods in recent years. In
deep classifiers, CP will generate a prediction set for a test sample that satisfies
the (1 _−_ _α_ ) coverage guarantee. The prediction set size (PSS) is then considered
a reflection of the predictive uncertainty. However, it is unknown whether the
predictive uncertainty of CP is aligned with its predictive correctness, which is
an imperative property for predictive uncertainty. This work answers this open
question by investigating the uncertainty calibration of CP in deep classifiers. We
first give a definition for the uncertainty calibration of CP by building a connection
between PSS and prediction accuracy and then propose a calibration target for CP
based on a theoretical analysis of the predictive distributions. Given this defined
CP calibration, we present an empirical study on several classification datasets and
reveal their weak calibration of CP. To strengthen the calibration of CP, we propose
CP-aware calibration (CPAC), a bi-level optimization algorithm, and demonstrate
the effectiveness of CPAC on several standard classification datasets by testing
models including ResNet, Vision Transformer and GPT-2.


1 INTRODUCTION


Due to its low computational overhead and distribution-free assumption, conformal prediction (CP)
Shafer & Vovk (2008); Romano et al. (2020); Angelopoulos et al. (2021) has become a dominant
approach to uncertainty quantification (UQ). CP has been successfully adopted in various machinelearning applications, including object detection Timans et al. (2024), pose estimation Yang & Pavone
(2023), pixel-level image understanding Brunekreef et al. (2024) and natural language understanding
Quach et al. (2024); Gui et al. (2024); Mohri & Hashimoto (2024). The predictive uncertainty from
CP stems from the _frequentist_ _approach_ to uncertainty, i.e., producing a confidence interval will
contain the true value with a specified probability (e.g., 90% or 95%). In a classification task, CP will
produce a prediction set _S_ for a test sample that is theoretically guaranteed to contain the ground-truth
class label with a high probability (e.g., 90%). Although it is desirable to have a prediction set with
a coverage guarantee, as a probabilistic forecast model, it is also important that the uncertainty is
consistent with the decision’s reliability Mincer & Zarnowitz (1969); Kochenderfer (2015), known as
_calibration_ Zadrozny & Elkan (2001); Gneiting et al. (2007).


For a multi-class classification model, the confidence score of the predicted label from a predictive
distribution has traditionally been used as a measure of uncertainty. In that case, model calibration
aims to reduce the gap between a model’s predicted confidence score and the actual observed
predictive correctness, measuring the model’s ability to estimate its predictive reliability. Whilst
the confidence-based calibration of modern large-scale machine learning models has been actively
investigated in recent years Guo et al. (2017); Minderer et al. (2021a); Achiam et al. (2023), calibration
through the lens of prediction set size (PSS), aiming for a tight coupling between PSS and expected
accuracy is under-investigated. Although the coverage of PSS is guaranteed, its alignment with
prediction accuracy is also essential for making risk-aware decisions and makes conformal inference
more reliable. As illustrated in Figure 1, point prediction calibration has already been thoroughly
investigated, but it outputs a single prediction for each query thus cannot guarantee the coverage
of ground truth in its prediction. Conformal prediction, a well-established approach for achieving


1


coverage guarantees, conveys uncertainty through the size of the prediction set Angelopoulos et al.
(2021). Yet, it remains unclear whether this set size truly aligns with prediction correctness.


between the PSS and accuracy. Meanwhile, CP

Figure 1: **(a)** and **(b)** compare CP and point pre
calibration was studied systematically on regres
diction in coverage and calibration. **(c)** Two key

sion tasks van der Laan & Alaa (2024), but our

contributions: _CP’s calibration target in multino-_

paper is the first attempt to systematically in
_mial sampling_ and _CP-aware calibration_ .

vestigate the calibration of CP on classification
tasks. Note that the fundamental difference between van der Laan & Alaa (2024) and our work is
that van der Laan & Alaa (2024) treats point prediction and prediction interval independently, while
we aim to calibrate a classification model so that the CP’s prediction set is calibrated and has valid
coverage at the same time. It is important to note that calibration for PSS is _fundamentally different_
from conditional coverage Gibbs et al. (2025). While conditional coverage ensures subgroup-level
validity, PSS calibration evaluates whether smaller sets consistently correspond to higher per-instance
reliability, a property not captured by conditional guarantees. Also note that, studying whether the
uncertainty CP conveys in practice (via set size) is trustworthy does not suggest CP should replace
probabilistic calibration such as entropy or confidence.


Developing calibration for conformal prediction through PSS has three key challenges. First, while
CP produces a set of plausible labels, it does not directly yield a point prediction and thus accuracy.
Second, the function between PSS and prediction accuracy is not straightforward, compared to the
linear function in traditional confidence-based uncertainty. Third, it is unclear how to effectively
calibrate a model to ensure that smaller prediction sets consistently correspond to higher accuracy.
To address these challenges, we first use multinomial sampling with temperature to generate a
point prediction from a prediction set, enabling us to obtain the accuracy. Then, we introduce a
calibration target function based on the predictive distribution that maps PSS to expected accuracy,
capturing the relationship between uncertainty and reliability in CP. An empirical study on the
calibration target functions reveals weak calibrations of conformalized models, highlighting the
need for correction. To this end, we propose a CP-aware calibration algorithm based on bi-level
optimization as a _pre-processing_ step before the quantile computation in the CP framework. Our
contributions are three-fold:


    - We establish a connection between the PSS and accuracy by sampling a label from the
prediction set with the predictive distribution. An empirical study on the alignment of PSS
and accuracy demonstrates the weak calibration of PSS.


    - We propose a calibration target function motivated by both our empirical study and a
theoretical analysis of the predictive distribution. It can handle prediction sampling with
different temperatures and has a lower calibration error on average compared with other
alternative target functions.


    - We propose a CP-aware calibration algorithm as a pre-processing step of CP to improve the
calibration of CP. The effectiveness in classification tasks is validated using three benchmark
datasets in computer vision and natural language understanding with state-of-the-art models,
including vision transformers and GPT-2.

2 RELATED WORK


**Conformal Prediction and Uncertainty Quantification** . Different from existing methods in the
frequentist’s approach to prediction uncertainty, CP is distribution-free and can be applied to any
black-box machine learning model as long as the data exchangeability assumption is satisfied Vovk
et al. (1999); Shafer & Vovk (2008). The original version of CP needs to train a model multiple times
but is later improved by Vovk et al. (2005) as the split conformal prediction that can be used in any
black-box model, leading to its popularity in many applications Romano et al. (2020); Angelopoulos
et al. (2021). Existing research mainly aims to improve the coverage validity Gibbs & Candes (2021)
and efficiency Angelopoulos et al. (2021); Ghosh et al. (2023a;b); Liu et al. (2024), as well as extend


2


Figure 1: **(a)** and **(b)** compare CP and point prediction in coverage and calibration. **(c)** Two key
contributions: _CP’s calibration target in multino-_
_mial sampling_ and _CP-aware calibration_ .


(a) 𝑟!"#=0.1 (b) 𝑟!"#=0.2 (c) 𝑟!"#=0.4 (d) 𝑟!"#=1.0


Figure 2: Reliability diagrams of ResNet50 trained on CIFAR100. The first row shows plots for
varying training data sizes ( _rsub_ is the subsampling ratio) with a pre-trained (on ImageNet-1k) initial
model, while the second row shows plots for a randomly initialized model. The **red** curve is the
observed one while the **blue** curve is the target curve.
CP to non-exchangeable data Barber et al. (2023) and achieve conditional coverage Gibbs et al.
(2023). More recently, a random set method with improved calibration is proposed Manchingal et al.
(2025), but it cannot be readily plugged into a pre-trained model as it is developed for Bayesian neural
networks. In contrast, our work systematically investigates the calibration of quantified uncertainty
by CP in classification tasks. This topic has been studied by Lu et al. (2023) and van der Laan &
Alaa (2024), but Lu et al. (2023) proposes federated CP for distributed users and does not solely
focus on the calibration, and van der Laan & Alaa (2024) only considers regression tasks. The
proposed conformalizing Venn-Abers calibration van der Laan & Alaa (2024) for regression models
cannot be applied to our classification problem as they produce the calibration multi-prediction and
prediction interval separately, but we aim to calibrate the prediction interval/set directly. The effect of
temperature scaling in CP is also investigated Xi et al. (2024); Dabah & Tirer (2024), but they only
consider the coverage and efficiency of PSS instead of reducing the gap between PSS and its accuracy


**Model Calibration** . Model calibration focuses on adjusting predictive models to ensure that the
predictive uncertainty accurately reflects the true likelihood Vaicenavicius et al. (2019). Model
calibration is crucial in safety-critic applications Huang et al. (2020) where decision-making relies on
well-calibrated probabilities. Deep neural networks have been found to be weakly calibrated can be
fixed by traditional methods like Platt scaling Guo et al. (2017), which fits a temperature scalar to a
classifier’s scores. Recent studies Minderer et al. (2021b) have shown that the large-scale pre-trained
models are more calibrated, in particular for the convolution architecture. With the huge impact of
large language models (LLMs), their calibration are also actively investigated Achiam et al. (2023);
Xiong et al. (2024). However, the model calibration mainly focuses on the heuristic uncertainty such
as confidence scores in classification. Our work aims to unveil the calibration of uncertainty when
CP is used in state-of-art models including both vision transformer models for vision tasks and an
LLM for language understanding tasks.


3 PRELIMINARIES
We introduce the necessary mathematical annotations and background in this section.


**Notations** . We split the dataset into three subsets, i.e., training set _Dtr_ = _{_ ( _**x**_ _i, yi_ ) _}_ _[N]_ _i_ =1 _[tr]_ [, calibration]
set _Dcal_ = _{_ ( _**x**_ _i, yi_ ) _}_ _[N]_ _i_ =1 _[cal]_ [and] [test] [set] _[D][te]_ [=] _[{]_ [(] _**[x]**_ _[i][, y][i]_ [)] _[}][N]_ _i_ =1 _[te]_ [.] [A] [classification] [model] [is] [trained] [on]
_Dtr_, and conformalization including Platt scaling is performed using the calibration set, and then the
conformalized model is evaluated on the test set. Each data point ( _**x**_ _, y_ ) is sampled from a distribution
over the data space _X_ _× Y_ . As we only investigate the classification task in this paper, the label space
_Y_ = _{_ 1 _, · · ·_ _, K}_ denoted as [ _K_ ] for simplicity. After training a deep classification model _fθ_ ( _**x**_ ) with
parameters _θ_, the model produces a logit vector _**l**_ _i_ _∈_ R _[K]_ for a test sample _**x**_ _i_, where the arg max _j_ _**l**_ _ij_
is the predicted label. The predictive distribution _**p**_ _i_ is the output of the softmax function when the
input is _**l**_ _i_ .


**Conformal Prediction** . Conformal prediction ensures population-level coverage guarantees without
distributional assumptions and applies to both regression and classification. This study focuses on
classification, where a conformalized classification model generates a prediction set _Si_ _∈_ 2 [[] _[K]_ []] for a
test sample _**x**_ _i_ so that the coverage guarantee is ensured
_P_ ( _y_ _∈S_ ( _**x**_ )) _≥_ 1 _−_ _α,_ (1)


3


(a) 𝜎=0 (b) 𝜎=0.1 (c) 𝜎=0.2 (d) 𝜎=0.4 (e) 𝜎=0.8


Figure 3: Reliability diagrams of three classification models on ImageNet when there is input noise
sampled from Gaussian distribution with a standard deviation of _σ_ .
where 1 _−_ _α_ is a confidence level such as 90%, indicating that the prediction set will contain the
ground-truth label with 90% confidence at the population level. The original CP needs to train a
model multiple times to obtain such a guarantee, while our paper uses the _split conformal prediction_
approach that does not need to train multiple times and can be plugged into any pre-trained black-box
classifier Papadopoulos et al. (2002); Lei et al. (2018).


In this study, we use the APS (Adaptive Prediction Sets) method Romano et al. (2020) to perform
the conformalization if not specified otherwise. There are two stages in APS, both done on _Dcal_
(a) temperature scaling Guo et al. (2017) and (b) computing the (1 _−_ _α_ )-quantile of conformity
scores. The temperature scaling aims to make the confidence score more calibrated by find an optimal
temperature to scale the logic vectors such that the likelihood of _Dcal_ is maximized. After scaling the
logits, conformity scores on _Dcal_ are computed and their (1 _−_ _α_ )-quantile can be found. We give a
description on the process of computing the conformity scores and the quantile in Appendix B.


4 CALIBRATION OF CONFORMAL PREDICTION


We now introduce a definition of CP calibration and describe how to use multinomial sampling to
obtain accuracy. This is followed by a theoretical analysis on the proposed calibration target function.
Moreover, we propose a calibration algorithm for CP on classification tasks (Algorithm 1).


4.1 CALIBRATION OF A CONFORMALIZED MODEL

Calibration for the uncertainty expressed by confidence scores is straightforward, as prediction
accuracy with a confidence score is easily obtained. However, as the uncertainty in CP is measured
by the PSS, the connection between a prediction set and its prediction quality cannot be immediately
obtained. In other words, a prediction set is not directly comparable to the ground-truth class. To
build a connection between a prediction set and its prediction accuracy, we propose a multinomial
sampling strategy to produce a prediction from the prediction set and take the average accuracy of
the sampled labels as the prediction correctness. Denote the normalized predictive probability in the
prediction set _Si_ as _**p**_ ˜ _i_ = [˜ _pi_ 1 _, · · ·_ _,_ ˜ _pi|Si|_ ] where [�] _j_ _[p]_ [˜] _[ij]_ [= 1][, and the multinomial distribution is a]

function of the predictive probability


where _t_ _∈_ [0 _,_ + _∞_ ) is the exponent for the sampling. When _t_ = 0, we use uniform sampling
to produce the predictive label. When _t_ approaches + _∞_, the sampling is equivalent to Top-1
accuracy using the maximum confidence. With the sampled accuracy, we give a definition for the CP
calibration.


**Definition 4.1.** A classifier is conformally calibrated if the conditional expectation of accuracy using
multinomial sampling with the temperature _t_ decreases with the prediction set size _k_, i.e.,
E[ _Acct_ ( _**x**_ ) _|S_ ( _**x**_ ) = _k_ ] = _f_ ( _k_ ) _,_ (3)
where _f_ ( _k_ ) is a monotonically decreasing function and _S_ ( _·_ ) maps an input sample into its PSS, the
condition means for the expectation is computed on all _**x**_ with _S_ ( _**x**_ ) = _k_, and
_Acct_ ( _**x**_ ) = E _**q**_ ( _t_ )( _**x**_ ) **1** (ˆ _y_ = _y_ ) _._ (4)


4


_**q**_ _i_ [(] _[t]_ [)] = [˜ _p_ _[t]_ _i_ 1 _[,][ · · ·]_ _[,]_ [ ˜] _[p][t]_ _i|Si|_ []] _[/]_


_|Si|_

- _p_ ˜ _[t]_ _ij_ _[,]_ (2)


_j_


We define the following two metrics for the calibration error for a conformalized model.


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|~~Acc=54.7~~|~~0%~~||||
||||||
||||||


|Acc=71.69|%|Col3|
|---|---|---|
||||
||||
||||
||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|~~Acc=35.7~~|~~%~~|||
|||||
|||||


standard deviation over five seeds. where [�] _j_ _[|S]_ =1 _[i][|]_ _[q]_ _ij_ [(] _[t]_ [)] = 1 _,_ [�] _j_ _[|S]_ =1 _[i][|]_ _[p]_ [˜] _[ij]_ [=] [1][.] [Note] [that] [the]

binary correctness in Equ. 4 becomes a probabilistic
one. It is straightforward to obtain that if _qij_ = 1 _/|Si|_ for every _j_, then the expected accuracy
is 1 _/|Si|_ . Fig. 4 shows that the curve of accuracy versus PSS fits well with the power function
_f_ ( _k_ ) = 1 _/k_, validating the assumption of the true class distribution.


**General cases.** Inspired by the success of the power function in the uniformly weighted sampling,
we propose to use a power function _f_ ( _k_ ) = 1 _/k_ _[τ]_ as the calibration target where _τ_ _∈_ [0 _,_ 1], to
accommodate the improvement in accuracy when using non-uniformly weighted sampling. Intuitively,
when we use a low temperature in the multinomial sampling, the probability distribution gets sparse
and the effective set size decreases. To account for such a set size decrease in the calibration function,
we use _τ_ _<_ 1 to decrease it from _|Si|_ to _|Si|_ _[τ]_ . Two alternative functions are the exponential decay
exp( _−τ_ ( _k −_ 1)) and the logarithmic scaling function 1 _/_ (1 + _τ_ log( _k_ )), where _τ_ _>_ 0. However, our
empirical study in Sec. 5 shows that the exponential decay function has a much faster decreasing
rate than the power function, while the logarithmic scaling function cannot fit the curve of uniform
sampling well.


We give a theorem on the relationship between the expected accuracy and the target calibration
function by assuming that both _**p**_ and _**q**_ are sampled from two Dirichlet distributions with the same
underlying shape.
**Theorem 4.2** (Expected Accuracy and Prediction Set Size) **.** _Let_ _K_ _≥_ 2 _be a dimension,_ _and let_
_**a**_ = ( _a_ 1 _, . . ., aK_ ) _be_ _a_ _vector_ _with_ _aj_ _≥_ 0 _and_ [�] _j_ _[K]_ =1 _[a][j]_ [=] [1] _[.]_ _[Suppose]_ _[α]_ [0] _[>]_ [0] _[and]_ _[β]_ [0] _[>]_ [0] _[are]_


5


_K_


_k_ =1


_Nte_ [(] _[k]_ [)] _|Acct_ ( _**x**_ _|S_ ( _**x**_ ) = _k_ ) _−_ _f_ ( _k_ ) _|,_ Uniform CP-ECE =
_Nte_


Standard CP-ECE =


_K_


_k_ =1


1

_|Acct_ ( _**x**_ _|S_ ( _**x**_ ) = _k_ ) _−_ _f_ ( _k_ ) _|._ (5)
_K_


The standard CP Expected Calibration Error (CP-ECE) is weighted by the proportion of samples
with PSS equal to _k_ relative to the entire test set. In contrast, the uniform CP-ECE corresponds to
the curve-fitting error measured by the absolute distance. We use the uniform CP-ECE since we
want to measure the curve fitting performance in the reliability diagram without considering the
number of samples in each bin. Moreover, we believe it is an important metric in practice as well,
because it gives the same weight to different groups to prevent discrimination towards minor groups
Mehrabi et al. (2021). It is also similar to the unweighted accuracy that is often used to measure the
performance of a model as a complement to the standard weighted accuracy. While both our work
and Huang et al. (2024) define calibration, note that they are fundamentally different, as we focus
on PSS calibration within conformal prediction and propose an optimization algorithm to reduce
calibration error, whereas Huang et al. (2024) neither considers conformal prediction nor minimizes
calibration error.
4.2 CALIBRATION TARGET FUNCTION
Another challenge for CP calibration is the target calibration curve _f_ ( _k_ ). In confidence calibration,
the identity _f_ ( _c_ ) = _c_ is the target curve as the perfect calibration Guo et al. (2017) is defined by
P( _Y_ [ˆ] = _Y |P_ [ˆ] = _p_ ) = _p, ∀p ∈_ [0 _,_ 1] _._ (6)


(a) Uniform Sampling (b) Multinomial Sampling (c) Top-1 Sampling


Figure 4: Sampling accuracy versus prediction set size (PSS) on three datasets. As
the temperature in multinomial sampling
decreases (uniform _⇒_ multinomial _⇒_ Top-1),
accuracy increases but the calibration of
PSS worsens. The shaded area shows the
standard deviation over five seeds.


However, the PSS from CP does not have a straightforward relationship with the prediction correctness
except for the monotonically decreasing property. We
propose a calibration target in this work, motivated by
the heuristic generalization of the uniform sampling
curve but also derived from a theoretical analysis based
on a Dirichlet distribution assumption on the predictive
probability.


We start from a simple case, where we use a uniform
weight to sample from the predictive probability, i.e.,
the temperature in multinomial sampling is zero. We
assume that the re-normalized predictive probability
_p_ ˜ _ij_ in the prediction set _Si_ is the probability of _j_ th
class being the true class, then the expected accuracy
of using multinomial sampling with weight _**q**_ _i_ is
_Acct_ ( _**x**_ _i_ ) =       - _qij_ [(] _[t]_ [)] _[p]_ [˜] _[ij][,]_ (7)

_j∈Si_


(1) Clean (2) Typo 10 (3) Typo 20 (4) Typo 30 (5) Typo 40 (6) Typo 50


(7) Gaussian 0.10 (8) Gaussian 0.11 (9) Gaussian 0.12 (10) Gaussian 0.13 (11) Gaussian 0.14 (12) Gaussian 0.15


Figure 5: Reliability diagrams of GPT-2 on a topic classification dataset when there is Gaussian noise
added on the embeddings or typos in textual input.


**Algorithm 1** CP-Aware Calibration

**Require:** Calibration dataset _D_ cal, logits from a pre-trained deep neural network _{_ _**l**_ _i}_ _[N]_ _i_ _[cal]_, regularization parameter _λ_, target calibration parameter _τ_, learning rate _η_, sampling temperature _t >_ 0,
batch size _B_, optimization round _Mopt_, mis-coverage level _α_
1: **for** _e ←_ 1 : _Mopt_ **do**
2: Run _g_ ( _**W**_ _,_ _**b**_ ) = _ν_, i.e., Compute conformality scores _{Ei}_ _[N]_ _i_ =1 _[cal]_ [and find the][ (1] _[ −]_ _[α]_ [)][-quantile]
_ν_
3: Randomly divide _Dcal_ into _T_ batches _{Bit}_ _[T]_ _it_ =1 [with batch size] _[ B]_
4: **for** _it ←_ 1 : _T_ **do**
5: Compute _∇_ _**W**_ _,_ _**b**_ _Lps_ + _λ_ ( _∥_ _**W**_ _−_ _**I**_ _∥_ [2] 2 [+] _[ ∥]_ _**[b]**_ _[∥]_ 2 [2][)][ for] _[ i][ ∈B][t]_ [, denoted as][ (∆] _**[W]**_ _[,]_ [ ∆] _**[b]**_ [)]
6: ( _**W**_ _it,_ _**b**_ _it_ ) _←_ ( _**W**_ _it−_ 1 _,_ _**b**_ _it−_ 1) _−_ _η_ ∆( _**W**_ _,_ _**b**_ )
7: **end for**
8: **end for**
9: **return** The calibration parameter ( _**W**_ _,_ _**b**_ )


_positive scalars, and define Dirichlet parameters_
_α_ = ( _α_ 1 _, . . ., αK_ ) _,_ _β_ = ( _β_ 1 _, . . ., βK_ )
_where αj_ = _α_ 0 _aj_ _and βj_ = _β_ 0 _aj._ _Let_ _**p**_ _∼_ Dir( _α_ ) _and_ _**q**_ _∼_ Dir( _β_ ) _be independent draws._ _We have_

1 log( [�] _[K]_ _j_ =1 _[a]_ _j_ [2][)]
E[ _**q**_ _·_ _**p**_ ] = _K_ _[τ]_ _[,]_ _where_ _τ_ = _−_ log _K_ _._


**Remark.** This theorem assumes that _**p**_ and _**q**_ are drawn from two Dirichlet distributions with the same
underlying mass distribution vector _**a**_ . This assumption is due to the proposed multinomial sampling,
where both sampling accuracy _**q**_ [(] _[t]_ [)] and correctness probability _**p**_ ˜ follow the same mass distribution.
The difference between Dir( _α_ ) and Dir( _β_ ) is the degree of concentration, which corresponds to
multinomial sampling with temperature. The exponent _τ_ is determined by the shape of the vector
_**a**_ . When _**a**_ is close to uniform, the expected accuracy is 1 _/K_ . When _**a**_ is highly peaked at a
single dimension, the expected accuracy will be one. We report the calibration error of more models
on three datasets in Sec. 5 using this target calibration function. Note that we derived a similar
decay function when the Dirichlet distribution does not hold and the prediction is sampled from a
logistic-normal distribution with ordered variances, see Appendix A. Thus, both the Dirichlet and
logistic-normal distributions serve as illustrative instantiations of our target behavior rather than
restrictive assumptions.


4.3 CONFORMAL-PREDICTION-CALIBRATION WITH BI-LEVEL OPTIMIZATION


As the calibration error of CP is not satisfactory, particularly on challenging datasets such as ImageNet
as our experiments show, we propose to post-process the model prediction so that the uncertainty
from CP is better aligned with the accuracy.


Denote the logit vector of _i_ th sample in the calibration set as _**l**_ _i_, the model is calibrated by optimizing
the weight matrix _**W**_ and bias _**b**_ similar to Platt scaling, but with our proposed calibration target. The
correctness probability _**p**_ ˜ and sampling probability _**q**_ [(] _[t]_ [)] function is


_**l**_ ˜ _ij_ = �[ _**W**_ _T_ _**l**_ _i_ + _**b**_ ] _j,_ if _j_ _∈Si_ _,_ _**p**_ ˜ _i_ = softmax( [˜] _li_ ) _,_ _**q**_ _i_ = [˜ _p_ _[t]_ _i_ 1 _[,][ · · ·]_ _[,]_ [ ˜] _[p][t]_ _ik_ []] _[/]_
_−∞,_ if _j_ _∈S/_ _i_


6


_|Si|_

- _p_ ˜ _[t]_ _ij_ (8)


_j_


|Col1|Col2|Std CP-ECE Uni CP-ECE Acc. Cov. PSS|
|---|---|---|
|Clean|PS<br>PS-Full<br>CPAC|6.88(0.06)<br>11.02(0.30)<br>81.98(0.08)<br>94.52(0.13)<br>18.47(0.64)<br>6.38(0.06)<br>8.40(0.50)<br>81.98(0.08)<br>93.88(0.07)<br>13.70(0.25)<br>6.74(0.09)<br>6.74(0.47)<br>80.17(0.16)<br>92.39(0.08)<br>10.04(0.10)|
|Norm-0.1|PS<br>PS-Full<br>CPAC|7.52(0.05)<br>10.70(0.28)<br>80.39(0.11)<br>94.22(0.14)<br>20.71(0.77)<br>6.90(0.04)<br>9.10(0.18)<br>80.39(0.11)<br>93.55(0.11)<br>15.00(0.42)<br>7.21(0.14)<br>7.39(1.12)<br>78.73(0.16)<br>91.83(0.19)<br>10.62(0.33)|
|Norm-0.2|PS<br>PS-Full<br>CPAC|8.30(0.11)<br>10.37(0.21)<br>76.98(0.07)<br>93.85(0.18)<br>27.49(1.15)<br>7.67(0.13)<br>10.54(0.48)<br>76.98(0.07)<br>93.04(0.17)<br>20.03(0.55)<br>7.99(0.12)<br>7.99(0.84)<br>75.33(0.11)<br>91.31(0.19)<br>14.90(0.45)|
|Norm-0.4|PS<br>PS-Full<br>CPAC|9.73(0.16)<br>8.82(0.17)<br>65.69(0.03)<br>92.85(0.20)<br>58.44(1.48)<br>9.34(0.07)<br>8.69(0.28)<br>65.69(0.03)<br>91.93(0.14)<br>48.13(1.01)<br>9.02(0.11)<br>8.30(0.64)<br>64.00(0.15)<br>89.72(0.17)<br>38.30(1.28)|
|Norm-0.8|PS<br>PS-Full<br>CPAC|8.09(0.10)<br>5.30(0.08)<br>28.34(0.09)<br>90.60(0.21)<br>239.18(2.99)<br>8.29(0.20)<br>5.43(0.05)<br>28.34(0.09)<br>90.37(0.27)<br>258.33(4.55)<br>7.26(0.13)<br>4.94(0.08)<br>28.17(0.19)<br>88.95(0.16)<br>239.63(2.55)|
|Blur-3|PS<br>PS-Full<br>CPAC|7.88(0.10)<br>11.04(0.56)<br>79.03(0.07)<br>93.92(0.14)<br>23.30(0.89)<br>7.50(0.05)<br>11.50(0.49)<br>79.03(0.07)<br>93.52(0.10)<br>19.37(0.51)<br>7.57(0.15)<br>8.38(0.56)<br>77.62(0.15)<br>92.15(0.05)<br>14.67(0.30)|
|Blur-5|PS<br>PS-Full<br>CPAC|8.40(0.07)<br>10.64(0.28)<br>77.94(0.05)<br>93.74(0.18)<br>25.92(1.19)<br>8.01(0.09)<br>10.61(0.50)<br>77.94(0.05)<br>93.39(0.11)<br>22.50(0.58)<br>8.02(0.19)<br>8.51(0.22)<br>76.41(0.20)<br>92.05(0.09)<br>17.24(0.57)|
|Blur-7|PS<br>PS-Full<br>CPAC|8.59(0.06)<br>10.27(0.21)<br>77.51(0.05)<br>93.67(0.23)<br>27.08(1.22)<br>8.24(0.08)<br>10.93(0.31)<br>77.51(0.05)<br>93.37(0.13)<br>23.83(0.51)<br>8.12(0.18)<br>8.96(0.44)<br>76.25(0.15)<br>92.10(0.07)<br>18.44(0.59)|
|Drop-1|PS<br>PS-Full<br>CPAC|9.69(0.06)<br>10.99(0.46)<br>74.50(0.13)<br>93.55(0.19)<br>40.62(2.10)<br>9.00(0.10)<br>11.38(0.45)<br>74.50(0.13)<br>92.54(0.13)<br>29.33(0.82)<br>8.71(0.22)<br>9.52(0.32)<br>73.68(0.19)<br>91.32(0.18)<br>23.93(0.67)|
|Drop-3|PS<br>PS-Full<br>CPAC|11.05(0.13)<br>7.70(0.25)<br>60.83(0.11)<br>92.46(0.12)<br>93.09(1.16)<br>11.37(0.12)<br>8.71(0.31)<br>60.83(0.11)<br>91.15(0.16)<br>79.22(1.73)<br>10.09(0.21)<br>8.28(0.24)<br>61.99(0.09)<br>91.82(0.19)<br>70.12(1.43)|
|Drop-5|PS<br>PS-Full<br>CPAC|10.29(0.03)<br>6.30(0.15)<br>44.36(0.12)<br>91.77(0.12)<br>161.52(1.21)<br>10.77(0.10)<br>6.61(0.08)<br>44.36(0.12)<br>90.69(0.08)<br>169.74(1.20)<br>9.67(0.09)<br>6.18(0.21)<br>45.01(0.20)<br>88.98(0.13)<br>126.75(1.51)|
|Drop-7|PS<br>PS-Full<br>CPAC|6.73(0.11)<br>5.62(0.20)<br>19.69(0.05)<br>90.48(0.32)<br>306.40(4.44)<br>7.71(0.15)<br>5.21(0.09)<br>19.69(0.05)<br>90.18(0.27)<br>392.71(6.12)<br>6.93(0.11)<br>4.86(0.16)<br>21.77(0.10)<br>88.84(0.40)<br>300.52(8.65)|


|Col1|Col2|Std CP-ECE Uni CP-ECE Acc. Cov. PSS|
|---|---|---|
|Clean|PS<br>PS-Full<br>CPAC|7.16(0.10)<br>10.01(0.27)<br>80.35(0.09)<br>94.18(0.17)<br>18.05(0.90)<br>6.65(0.07)<br>9.36(0.44)<br>80.35(0.09)<br>93.75(0.14)<br>14.44(0.54)<br>7.45(0.11)<br>7.09(0.54)<br>77.68(0.14)<br>91.29(0.09)<br>10.43(0.53)|
|Norm-0.1|PS<br>PS-Full<br>CPAC|7.78(0.09)<br>9.16(0.30)<br>78.67(0.06)<br>93.98(0.22)<br>20.78(1.19)<br>7.21(0.10)<br>9.12(0.47)<br>78.67(0.06)<br>93.43(0.18)<br>16.19(0.58)<br>7.72(0.11)<br>7.17(0.58)<br>75.45(0.32)<br>90.75(0.11)<br>11.37(0.26)|
|Norm-0.2|PS<br>PS-Full<br>CPAC|8.53(0.10)<br>9.98(0.35)<br>74.13(0.05)<br>93.57(0.12)<br>31.27(0.76)<br>7.97(0.07)<br>10.11(0.29)<br>74.13(0.05)<br>92.93(0.06)<br>24.89(0.29)<br>8.46(0.16)<br>7.98(0.36)<br>70.57(0.41)<br>89.66(0.31)<br>17.11(0.40)|
|Norm-0.4|PS<br>PS-Full<br>CPAC|9.75(0.14)<br>7.21(0.16)<br>57.19(0.16)<br>92.05(0.07)<br>82.51(0.40)<br>9.59(0.10)<br>7.54(0.17)<br>57.19(0.16)<br>91.29(0.10)<br>75.44(0.78)<br>8.67(0.10)<br>7.24(0.36)<br>53.43(0.28)<br>87.77(0.41)<br>63.31(3.75)|
|Norm-0.8|PS<br>PS-Full<br>CPAC|5.06(0.12)<br>7.25(0.27)<br>12.30(0.06)<br>90.00(0.30)<br>429.42(5.39)<br>5.32(0.07)<br>5.10(0.16)<br>12.30(0.06)<br>89.84(0.28)<br>469.92(5.28)<br>4.39(0.23)<br>4.72(0.27)<br>12.50(0.21)<br>89.09(0.19)<br>437.94(3.66)|
|Blur-3|PS<br>PS-Full<br>CPAC|8.17(0.10)<br>10.13(0.46)<br>77.06(0.10)<br>93.61(0.19)<br>24.61(0.90)<br>7.79(0.09)<br>10.68(0.41)<br>77.06(0.10)<br>93.27(0.16)<br>21.24(0.78)<br>8.02(0.07)<br>7.79(0.39)<br>75.19(0.32)<br>91.25(0.10)<br>15.18(0.47)|
|Blur-5|PS<br>PS-Full<br>CPAC|8.76(0.05)<br>10.20(0.44)<br>75.24(0.09)<br>93.50(0.14)<br>28.73(0.70)<br>8.42(0.10)<br>10.08(0.34)<br>75.24(0.09)<br>93.21(0.17)<br>25.44(0.84)<br>8.39(0.16)<br>8.07(0.15)<br>73.09(0.10)<br>91.01(0.21)<br>18.73(0.50)|
|Blur-7|PS<br>PS-Full<br>CPAC|8.96(0.16)<br>9.74(0.27)<br>74.39(0.06)<br>93.40(0.16)<br>31.02(0.88)<br>8.67(0.12)<br>9.91(0.51)<br>74.39(0.06)<br>93.09(0.19)<br>27.65(0.97)<br>8.54(0.14)<br>8.11(0.51)<br>72.12(0.16)<br>90.82(0.21)<br>20.55(0.56)|
|Drop-1|PS<br>PS-Full<br>CPAC|9.76(0.15)<br>9.46(0.47)<br>71.04(0.07)<br>93.10(0.18)<br>39.13(1.03)<br>9.33(0.14)<br>9.87(0.50)<br>71.04(0.07)<br>92.41(0.24)<br>31.90(0.94)<br>9.04(0.12)<br>8.22(0.69)<br>68.26(0.14)<br>89.72(0.18)<br>25.09(0.33)|
|Drop-3|PS<br>PS-Full<br>CPAC|10.31(0.15)<br>6.60(0.09)<br>54.91(0.11)<br>91.93(0.24)<br>93.11(2.37)<br>10.62(0.16)<br>7.29(0.17)<br>54.91(0.11)<br>91.00(0.32)<br>86.53(2.79)<br>8.09(0.46)<br>8.33(0.82)<br>50.49(1.14)<br>88.08(0.20)<br>84.83(10.46)|
|Drop-5|PS<br>PS-Full<br>CPAC|8.75(0.08)<br>5.42(0.20)<br>35.38(0.07)<br>91.03(0.23)<br>187.77(3.32)<br>9.18(0.05)<br>5.63(0.05)<br>35.38(0.07)<br>90.48(0.28)<br>211.86(3.96)<br>6.81(0.54)<br>5.35(0.09)<br>32.77(0.53)<br>88.62(0.36)<br>215.71(6.65)|
|Drop-7|PS<br>PS-Full<br>CPAC|5.10(0.07)<br>7.16(0.11)<br>11.97(0.09)<br>89.97(0.12)<br>423.58(2.71)<br>5.84(0.10)<br>5.26(0.16)<br>11.97(0.09)<br>89.90(0.26)<br>490.89(5.01)<br>4.78(0.17)<br>4.84(0.21)<br>12.73(0.22)<br>88.95(0.14)<br>427.64(2.94)|


Table 1: Result of ViT-Large (left) and ViT-Base (right) on ImageNet-1k. _Norm-σ_ means Gaussian
noise with a std _σ_, _Blur_ - _n_ means Gaussian blur with kernel size _n_ and _Drop_ - _r_ means randomly drop
pixels with ratio _r_ .
The optimization problem is
min �(    - _p_ ˜ _ij_ ( _**W**_ _,_ _**b**_ _, ν_ ) _qij_ [(] _[t]_ [)][(] _**[W]**_ _[,]_ _**[ b]**_ _[, ν]_ [)] _[ −]_ _[f][τ]_ [(] _[|S][i][|]_ [))][2] _[,]_ _s.t._ _ν −_ _g_ ( _**W**_ _,_ _**b**_ ) = 0 _,_ (9)
_**W**_ _,_ _**b**_

_i_ _j∈Si_

where the _fτ_ ( _·_ ) function is the target calibration curve with _τ_ as the exponent. To optimize ( _**W**_ _,_ _**b**_ ),
we need first to obtain the prediction set _Si_ for each sample by finding the empirical (1 _−_ _α_ )-quantile
of conformity scores in the calibration set. We denote the target as _ν_ and the searching function as
_g_ . Thus, we formulate the CP calibration problem as a bi-level optimization problem, where the
prediction set is produced from solving the lower-level conformity scores’ (1 _−_ _α_ )-quantile. As this
objective function will lead to a zero gradient for samples of PSS=1, we use the cross-entropy loss
for samples with PPS=1.


To solve this bi-level optimization problem, we adopt the alternative optimization approach as shown
in Alg. 1 by assuming the _ν_ variable does not change drastically during optimization ( _**W**_ _,_ _**b**_ ). Note
that we add a regularization term to constrain the distance between ( _**W**_ _,_ _**b**_ ) and the initialization ( _**I**_ _,_ **0** )
to prevent overfitting on the calibration set. Note that we choose to optimize the full weight matrix
instead of a temperature parameter as in Guo et al. (2017) as our pilot empirical study shows that a
single scalar does not affect the calibration significantly as it fails to re-rank the class probabilities.


5 EXPERIMENTAL RESULTS

This section first describes the experimental details and then reports our empirical study on the
calibration of CP on the three datasets.


5.1 EXPERIMENTAL SETTINGS

We conducted the experiment on three datasets using seven models, including two for image classification and one for topic classification. The experimental details are reported below.


**CIFAR100 Krizhevsky et al. (2009).** The dataset comprises 100 categories, each containing 600
images where 500 of them are used for training and 100 are used for test. We use 20% of the original
test data as the calibration set and the rest 80% as the test set. The model we use is ResNet50 He et al.
(2016), pre-trained on ImageNet Deng et al. (2009) or randomly initialized He et al. (2015). We train
the model for 60 epochs and decay the learning rate by dividing it by 10 at 30th and 50th epoch. The
initial learning rate is 0.1 in all the CIFAR100 training trials.


**ImageNet-1k Deng et al. (2009).** The dataset consists of approximately 1.28 million training images
and 50,000 validation images, categorized into 1,000 classes. We utilize three models—ResNet101
He et al. (2016), ViT-B, and ViT-L Dosovitskiy et al. (2021)—with parameter sizes of 44.5M, 86M,
and 307M, respectively. All images are resized to 224 _×_ 224, and the patch sizes in ViT-B and ViT-L
are 16 _×_ 16. These pre-trained models, officially released and trained on ImageNet-1K, are used
without further modifications. We evaluate the models under three types of image perturbations:
Gaussian noise, Gaussian blur, and pixel dropout. ( **1** ) Gaussian Noise: We apply Gaussian noise
with four different sigma values ( _σ_, the square root of variance): 0.1, 0.2, 0.4, and 0.8. Each sigma
value is uniformly applied across all test images. Additionally, we test a range of sigma values (0


7


(  
_i_ _j∈S_


- _p_ ˜ _ij_ ( _**W**_ _,_ _**b**_ _, ν_ ) _qij_ [(] _[t]_ [)][(] _**[W]**_ _[,]_ _**[ b]**_ _[, ν]_ [)] _[ −]_ _[f][τ]_ [(] _[|S][i][|]_ [))][2] _[,]_ _s.t._ _ν −_ _g_ ( _**W**_ _,_ _**b**_ ) = 0 _,_ (9)

_j∈Si_


to 0.8), where a random sigma is sampled to each image individually. ( **2** ) Gaussian Blur: Images
are perturbed using Gaussian blur with kernel sizes of 3×3, 5×5, and 7×7. ( **3** ) Pixel Dropout: We
randomly drop pixels from images at varying ratios: 0.1, 0.3, 0.5, and 0.7.


**Topic Classification** [1] **.** The dataset includes 22,500 (a) Entropy of 𝑝" (b) Entropy of 𝑝
pieces of text which are categorized into 120 topics. We split the dataset with 80% as the training set and 20% as the test set to fine-tune a GPT2 Small model (137M) Radford et al. (2019) for
the topic classification task. In evaluation, we test
the calibration performance of different methods
on the clean and two perturbed datasets. The perturbation strategies are: ( **1** ) Gaussian Noise: We Figure 6: Different behaviour of predictive
add a norm distribution noise _ϵ_ _∼_ _N_ ( _µ, σ_ [2] ) on distributions within the prediction set and the
the text embeddings. We set _µ_ to 0, and vary whole dimension when using ImageNet and
the perturbation strength by assigning _σ_ values in ResNet101.
the range [0.10, 0.15], incremented by 0.01; and ( **2** ) Typo: We randomly insert English
keystroke typo, which is a mixture of insertion, deletion, substitution, and transposition Kukich (1992), to the test data to simulate the practical scenario. The perturbation rate, i.e., the
portion of typo words relative to the total words, ranges from 10% to 50%, stepping by 10%.


|Col1|Col2|Std CP-ECE Uni CP-ECE Acc. Cov. PSS|
|---|---|---|
|Clean|PS<br>PS-Full<br>CPAC|7.87(0.58)<br>6.01(0.53)<br>60.46(0.36)<br>91.08(1.00)<br>9.07(0.77)<br>7.33(0.45)<br>6.21(0.37)<br>60.46(0.36)<br>91.58(0.98)<br>9.06(0.63)<br>7.02(0.55)<br>5.50(0.46)<br>60.40(0.38)<br>91.71(0.81)<br>9.18(0.59)|
|Norm-0.10|PS<br>PS-Full<br>CPAC|6.36(0.81)<br>5.29(0.75)<br>56.13(0.28)<br>90.64(1.25)<br>12.10(1.47)<br>6.48(0.56)<br>5.72(0.71)<br>56.13(0.28)<br>90.84(1.26)<br>11.88(1.37)<br>5.91(0.50)<br>5.08(0.63)<br>56.07(0.27)<br>91.14(0.93)<br>11.93(1.05)|
|Norm-0.11|PS<br>PS-Full<br>CPAC|5.69(0.21)<br>4.96(0.42)<br>53.88(0.21)<br>90.66(0.57)<br>13.62(0.70)<br>5.24(0.38)<br>4.59(0.43)<br>53.88(0.21)<br>90.90(0.63)<br>13.22(0.65)<br>5.32(0.43)<br>4.59(0.49)<br>53.88(0.25)<br>90.35(0.92)<br>12.62(0.94)|
|Norm-0.12|PS<br>PS-Full<br>CPAC|5.28(0.05)<br>5.22(0.32)<br>50.90(0.29)<br>90.89(0.29)<br>17.06(0.49)<br>5.06(0.51)<br>5.15(0.59)<br>50.90(0.29)<br>91.03(0.52)<br>16.47(1.00)<br>5.02(0.56)<br>4.87(0.68)<br>50.96(0.29)<br>91.07(0.92)<br>16.61(1.77)|
|Norm-0.13|PS<br>PS-Full<br>CPAC|5.51(0.42)<br>5.42(0.33)<br>46.02(0.24)<br>89.72(0.17)<br>20.11(0.32)<br>5.37(0.31)<br>5.43(0.22)<br>46.02(0.24)<br>89.77(0.33)<br>19.64(0.69)<br>5.24(0.18)<br>5.19(0.30)<br>46.05(0.24)<br>89.49(0.35)<br>18.98(0.54)|
|Norm-0.14|PS<br>PS-Full<br>CPAC|5.31(0.43)<br>6.04(0.57)<br>39.46(0.70)<br>90.13(1.43)<br>30.86(3.72)<br>4.95(0.42)<br>5.56(0.56)<br>39.46(0.70)<br>90.10(1.47)<br>30.45(3.77)<br>5.10(0.36)<br>5.87(0.62)<br>39.37(0.78)<br>89.86(1.79)<br>30.01(4.72)|
|Norm-0.15|PS<br>PS-Full<br>CPAC|5.49(0.14)<br>6.18(0.32)<br>31.06(0.28)<br>89.85(0.67)<br>40.83(1.73)<br>5.21(0.56)<br>6.01(0.62)<br>31.06(0.28)<br>89.89(0.64)<br>40.99(1.83)<br>4.69(0.32)<br>5.46(0.22)<br>31.10(0.35)<br>89.99(0.68)<br>41.12(2.01)|
|Typo-10|PS<br>PS-Full<br>CPAC|7.69(0.46)<br>6.74(0.20)<br>54.35(0.27)<br>91.02(0.58)<br>15.17(0.75)<br>7.01(0.37)<br>6.07(0.17)<br>54.35(0.27)<br>90.92(0.59)<br>14.42(0.71)<br>6.48(0.40)<br>5.40(0.24)<br>54.66(0.47)<br>91.27(0.44)<br>14.78(0.64)|
|Typo-20|PS<br>PS-Full<br>CPAC|7.79(0.53)<br>6.36(0.34)<br>53.09(0.17)<br>90.37(0.53)<br>15.38(0.65)<br>6.84(0.34)<br>5.87(0.26)<br>53.09(0.17)<br>90.42(0.57)<br>14.73(0.61)<br>6.48(0.48)<br>5.73(0.50)<br>53.41(0.44)<br>90.99(0.21)<br>15.65(0.31)|
|Typo-30|PS<br>PS-Full<br>CPAC|7.24(0.48)<br>6.63(0.67)<br>51.54(0.40)<br>90.35(0.51)<br>17.62(0.62)<br>6.10(0.62)<br>5.75(0.85)<br>51.54(0.40)<br>90.66(0.52)<br>17.12(0.80)<br>6.48(0.35)<br>6.35(0.48)<br>51.61(0.39)<br>90.95(0.33)<br>17.33(0.76)|
|Typo-40|PS<br>PS-Full<br>CPAC|6.36(0.53)<br>5.94(0.49)<br>48.14(0.31)<br>90.59(0.45)<br>19.66(0.79)<br>5.83(0.39)<br>5.67(0.34)<br>48.14(0.31)<br>90.75(0.49)<br>19.33(1.09)<br>5.97(0.54)<br>5.67(0.32)<br>48.10(0.47)<br>90.16(0.60)<br>18.25(1.00)|
|Typo-50|PS<br>PS-Full<br>CPAC|5.34(0.41)<br>5.28(0.53)<br>44.44(0.26)<br>89.85(0.80)<br>23.55(1.85)<br>5.08(0.63)<br>5.13(0.76)<br>44.44(0.26)<br>89.96(0.65)<br>23.31(1.60)<br>5.00(0.30)<br>5.14(0.39)<br>44.47(0.15)<br>90.17(0.41)<br>23.48(0.70)|


5.2 TARGET CALIBRATION FUNCTION


We compare the curve fitting error of using the proposed target function and two alternatives,
exponential function exp( _−τ_ ( _k_ _−_ 1)) and logarithmic function 1 _/_ (1 + _τ_ log( _k_ )) in Tab. 3. The
logarithmic function is better than the power function in Multinomial and Top-1 sampling, but it fails
to fit the simple curve of uniform sampling. Therefore, we still use the power function in our paper
but the logarithmic function can also be used in low-temperature sampling.


1https://huggingface.co/datasets/valurank/Topic_Classification


8


(a) Entropy of 𝑝" (b) Entropy of 𝑝


Figure 6: Different behaviour of predictive
distributions within the prediction set and the
whole dimension when using ImageNet and
ResNet101.


**CPAC details.** We use 20% of the original test
set as the calibration set _Dcal_ in our experiment.
Based on our preliminary experiment, the sampling temperature _t_ is 3, the round of optimization _Mopt_ is 4, the regularization hyperparameter _λ_ = 1 _e−_ 4 and the batch size is 1024. We use
grid search to choose the optimal learning rate
from _{_ 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1 _}_
and _τ_ from _{_ 0 _._ 1 _,_ 0 _._ 2 _, · · ·_ _,_ 0 _._ 6 _}_ . The CPAC is
performed on samples with low PSS (PSS _<_ 400
on ImageNet and PSS _<_ 70 on Topic Cls. data)
as we only need to cover (1 _−_ _α_ ) of all samples
in CP and we choose to optimize those low-PSS
samples. We run the experiment five times in
each setting by using five random seeds when
splitting the original test set and report the average of the CP-ECE, accuracy, coverage and
PSS. We use _CPAC_ to denote our method and
_PS_ to denote the standard confidence calibration
method in APS. All experiments were run on
NVIDIA GeForce RTX 3090.


Table 2: Result of GPT-2 on Topic Classification.
Norm means adding Normal distribution noise on
embedding vectors and Typo means mixing typos
with original text.


As high-temperature sampling tends to have a with original text.
good calibration error, we mainly investigate the
most ill-behaved sampling strategy, i.e., Top-1 sampling, in our experiment. All figures and tables
show the result of Top-1 accuracy, if not specified otherwise. During the test stage, we use grid
search to find the optimal _τ_ to compute the standard and uniform CP-ECE respectively. In reliability
diagrams (Accuracy versus PSS), we only visualize the result of one random seed following the
convention in Guo et al. (2017). Note that either sampling or expectation is possible to report but we
use the sampling notion to approximate the real-world decision-making process in this paper. We
exclude the empty PSS case in our implementation by setting the minimum PSS to be 1.


5.3 FACTORS THAT AFFECT CP CALIBRATION

**Pre-Trained Model vs.** **Random Initialization.** Fig. 2.1 and 2.2 compare the reliability diagram
of with and without pre-trained weights on CIFAR100. In all cases, the standard CP-ECE of using
pre-trained weights is worse than using random initialization, despite its improved accuracy. In terms
of the target function, when there is sufficient data, i.e., subsampling ratio is 0.2, 0.4 or 0.8, _τ_ of
pre-trained weights is larger than random initialization, meaning that the predictive distribution of
using pre-trained weights is more uniform than that of random initialization. However, when there
is only limited data, i.e., subsampling ratio is 0.1, _τ_ is less peaked in random initialization than in
pre-trained weights with very low accuracy. This corroborates the effectiveness of a pre-trained
model in the low-data regime, but also shows its weakness in CP calibration.


**Subsampling.** Fig. 2.a-d show the change of reliability diagrams when more training data is used.
An increasing trend in accuracy from left to right is observed, but in most cases, standard CP-ECE
goes up. This indicates that training with more data does not necessarily improve the CP calibration.


**Noisy Environment.** Fig. 3 shows the CP reliability diagrams when input images are perturbed with
Gaussian noise. Both ViT models are not as robust to Gaussian noise as the ResNet model, but the
ResNet model has the worse standard CP-ECE compared with the other two. In particular, Fig. 3.1
shows that when there is more noise, _τ_ will decrease, indicating that the probability shape within the
prediction set gets more peaked. This finding is validated by the result in Fig. 6.a, where the entropy
of _**p**_ ˜ within a prediction set when _σ_ = 0 _._ 8 is smaller than that when test images are clean. However,
there is an opposite trend in the entropy of the original probability _**p**_ as shown in Fig. 6.b. The CP
reliability diagrams when Gaussian noise is added to the embeddings or there are typos in the text
when using GPT-2 are shown in Fig. 5. The _τ_ of clean input is slightly higher than that when there is
input noise, but the change of _τ_ ’s is minor in GPT-2 compared with that in vision models.


5.4 PERFORMANCE OF CPAC

The previous subsection shows that calibration is an independent dimension of CP and needs to be
optimized. We present our empirical results on ImageNet and Topic Classification in this subsection,
as the calibration error on CIFAR100 is not high, we focus on calibrating CP.


topic classification task. On almost all the settings,

|Col1|Power|Exponential|Logarithmic|
|---|---|---|---|
|Uniform|0.29(0.03)|2.73(0.12)|2.04(0.07)|
|Multinomial|6.99(0.24)|17.23(0.20)|5.95(0.22)|
|Top-1|11.02(0.30)|18.40(0.37)|10.41(0.26)|

CPAC reduces the Uniform CP-ECE without sacrific- Table 3: Curve fitting error (Uni. CP-ECE) of
ing Std CP-ECE or even improve it, and meanwhile ViT-L on ImageNet using different sampling
maintains the accuracy and decreases the PSS. The strategies.
decreased PSS can be attributed to the CPAC on samples with low PSS. We also observe that CPAC
mainly reduces the Uni. CP-ECE, especially when there are many classes, i.e., on ImageNet. This
is due to the fact that the loss of high-PSS samples in CPAC is often larger than low-PSS samples,
so CPAC tends to focus on high-PSS samples and leads to low curve fitting error. To compare the
PSS when the coverage is fixed, we select the non-conformity score threshold so that the coverage is
controlled to be the 90% and report the result in Appendix C. When the coverage is fixed, our method
enlarges the PSS compared with the baseline (split CP with Platt scaling). However, the coverage
control experiment is not doable in practice as the test set is unknown. The increased PSS won’t
diminish our major contributions, i.e., the concept and method of PSS calibration, as the table still
shows the improvement of calibration error when using CPAC.


6 CONCLUSION


This work presents a systematic research into the uncertainty calibration in CP for classification,
where the uncertainty is measured by the prediction set size. We first give a definition and metrics for
the calibration of CP, then propose a target calibration function for PSS which is validated by both
empirical results and our theoretical analysis. Finally, we propose a bi-level optimization algorithm
that performs CP-aware calibration, and show its effectiveness on three classification tasks with
state-of-the-art models. This work will inspire future research into the uncertainty calibration of CP,
which is largely neglected by the community. One weakness of this work is that the convergence and
generalization of the bi-level optimization problem are only validated empirically but not analyzed in
theory, which will be addressed by our future work.


9


Table 3: Curve fitting error (Uni. CP-ECE) of
ViT-L on ImageNet using different sampling
strategies.


REFERENCES


Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.
_arXiv preprint arXiv:2303.08774_, 2023.


Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan, and Jitendra Malik. Uncertainty
sets for image classifiers using conformal prediction. In _International Conference on Learning_
_Representations_, 2021.


Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas, and Ryan J Tibshirani. Conformal
prediction beyond exchangeability. _The Annals of Statistics_, 51(2):816–845, 2023.


Joren Brunekreef, Eric Marcus, Ray Sheombarsing, Jan-Jakob Sonke, and Jonas Teuwen. Kandinsky
conformal prediction: Efficient calibration of image segmentation algorithms. In _Proceedings of_
_the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 4135–4143, 2024.


Lahav Dabah and Tom Tirer. On temperature scaling and conformal prediction of deep classifiers.
_arXiv preprint arXiv:2402.05806_, 2024.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pp. 248–255. Ieee, 2009.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit,
and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale.
In _International Conference on Learning Representations_, 2021. [URL https://openreview.](https://openreview.net/forum?id=YicbFdNTTy)
[net/forum?id=YicbFdNTTy.](https://openreview.net/forum?id=YicbFdNTTy)


Subhankar Ghosh, Taha Belkhouja, Yan Yan, and Janardhan Rao Doppa. Improving uncertainty
quantification of deep classifiers via neighborhood conformal prediction: Novel algorithm and
theoretical analysis. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 37,
pp. 7722–7730, 2023a.


Subhankar Ghosh, Yuanjie Shi, Taha Belkhouja, Yan Yan, Jana Doppa, and Brian Jones. Probabilistically robust conformal prediction. In _Uncertainty in Artificial Intelligence_, pp. 681–690. PMLR,
2023b.


Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift. _Advances_
_in Neural Information Processing Systems_, 34:1660–1672, 2021.


Isaac Gibbs, John J Cherian, and Emmanuel J Candès. Conformal prediction with conditional
guarantees. _arXiv preprint arXiv:2305.12616_, 2023.


Isaac Gibbs, John J Cherian, and Emmanuel J Candès. Conformal prediction with conditional
guarantees. _Journal of the Royal Statistical Society Series B: Statistical Methodology_, pp. qkaf008,
2025.


Tilmann Gneiting, Fadoua Balabdaoui, and Adrian E Raftery. Probabilistic forecasts, calibration
and sharpness. _Journal of the Royal Statistical Society Series B: Statistical Methodology_, 69(2):
243–268, 2007.


Yu Gui, Ying Jin, and Zhimei Ren. Conformal alignment: Knowing when to trust foundation models
with guarantees. In _The_ _Thirty-eighth_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_
_Systems_, 2024. [URL https://openreview.net/forum?id=YzyCEJlV9Z.](https://openreview.net/forum?id=YzyCEJlV9Z)


Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural
networks. In _International conference on machine learning_, pp. 1321–1330. PMLR, 2017.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification. In _Proceedings of the IEEE international_
_conference on computer vision_, pp. 1026–1034, 2015.


10


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,
pp. 770–778, 2016.


Xinmeng Huang, Shuo Li, Mengxin Yu, Matteo Sesia, Hamed Hassani, Insup Lee, Osbert Bastani,
and Edgar Dobriban. Uncertainty in language models: Assessment through rank-calibration. In
Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), _Proceedings of the 2024 Conference_
_on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_, pp. 284–312, Miami, Florida, USA,
November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
18. [URL https://aclanthology.org/2024.emnlp-main.18/.](https://aclanthology.org/2024.emnlp-main.18/)


Yingxiang Huang, Wentao Li, Fima Macheret, Rodney A Gabriel, and Lucila Ohno-Machado. A
tutorial on calibration measurements and calibration models for clinical prediction models. _Journal_
_of the American Medical Informatics Association_, 27(4):621–633, 2020.


Mykel J Kochenderfer. _Decision making under uncertainty:_ _theory and application_ . MIT press,
2015.


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.


Karen Kukich. Techniques for automatically correcting words in text. _ACM_ _computing_ _surveys_
_(CSUR)_, 24(4):377–439, 1992.


Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J Tibshirani, and Larry Wasserman. Distribution-free
predictive inference for regression. _Journal of the American Statistical Association_, 113(523):
1094–1111, 2018.


Ziquan Liu, Yufei CUI, Yan Yan, Yi Xu, Xiangyang Ji, Xue Liu, and Antoni B. Chan. The pitfalls and
promise of conformal inference under adversarial attacks. In _Forty-first International Conference on_
_Machine Learning_, 2024. [URL https://openreview.net/forum?id=2xLyc5TkFl.](https://openreview.net/forum?id=2xLyc5TkFl)


Charles Lu, Yaodong Yu, Sai Praneeth Karimireddy, Michael Jordan, and Ramesh Raskar. Federated
conformal predictors for distributed uncertainty quantification. In _International Conference on_
_Machine Learning_, pp. 22942–22964. PMLR, 2023.


Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, and
Fabio Cuzzolin. Random-set neural networks. In _The Thirteenth International Conference on Learn-_
_ing Representations_, 2025. [URL https://openreview.net/forum?id=pdjkikvCch.](https://openreview.net/forum?id=pdjkikvCch)


Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey
on bias and fairness in machine learning. _ACM computing surveys (CSUR)_, 54(6):1–35, 2021.


Jacob A Mincer and Victor Zarnowitz. The evaluation of economic forecasts. In _Economic forecasts_
_and expectations:_ _Analysis of forecasting behavior and performance_, pp. 3–46. NBER, 1969.


Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby,
Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), _Advances_ _in_
_Neural Information Processing Systems_, volume 34, pp. 15682–15694. Curran Associates, Inc.,
2021a. URL [https://proceedings.neurips.cc/paper_files/paper/2021/](https://proceedings.neurips.cc/paper_files/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf)
[file/8420d359404024567b5aefda1231af24-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2021/file/8420d359404024567b5aefda1231af24-Paper.pdf)


Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby,
Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. _Advances in_
_Neural Information Processing Systems_, 34:15682–15694, 2021b.


Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees. In _Forty-first_ _International_ _Conference_ _on_ _Machine_ _Learning_, 2024. URL [https:](https://openreview.net/forum?id=uYISs2tpwP)
[//openreview.net/forum?id=uYISs2tpwP.](https://openreview.net/forum?id=uYISs2tpwP)


Harris Papadopoulos, Kostas Proedrou, Volodya Vovk, and Alex Gammerman. Inductive confidence
machines for regression. In _Machine learning: ECML 2002: 13th European conference on machine_
_learning Helsinki, Finland, August 19–23, 2002 proceedings 13_, pp. 345–356. Springer, 2002.


11


Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S Jaakkola, and Regina
Barzilay. Conformal language modeling. In _The Twelfth International Conference on Learning_
_Representations_, 2024.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.


Yaniv Romano, Matteo Sesia, and Emmanuel Candes. Classification with valid and adaptive coverage.
_Advances in Neural Information Processing Systems_, 33:3581–3591, 2020.


Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. _Journal of Machine Learning_
_Research_, 9(3), 2008.


Alexander Timans, Christoph-Nikolas Straehle, Kaspar Sakmann, and Eric Nalisnick. Adaptive
bounding box uncertainties via two-step conformal prediction. In _European_ _Conference_ _on_
_Computer Vision_, pp. 363–398. Springer, 2024.


Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten, Jacob Roll, and Thomas
Schön. Evaluating model calibration in classification. In _The 22nd international conference on_
_artificial intelligence and statistics_, pp. 3459–3467. PMLR, 2019.


Lars van der Laan and Ahmed Alaa. Self-calibrating conformal prediction. In _The_ _Thirty-_
_eighth_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_, 2024. URL [https:](https://openreview.net/forum?id=BJ6HkT7qIk)
[//openreview.net/forum?id=BJ6HkT7qIk.](https://openreview.net/forum?id=BJ6HkT7qIk)


Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. _Algorithmic learning in a random world_,
volume 29. Springer, 2005.


Volodya Vovk, Alexander Gammerman, and Craig Saunders. Machine-learning applications of
algorithmic randomness. 1999.


Huajun Xi, Jianguo Huang, Kangdao Liu, Lei Feng, and Hongxin Wei. Delving into temperature
scaling for adaptive conformal prediction. _arXiv preprint arXiv:2402.04344_, 2024.


Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can
LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs.
In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2024. URL [https:](https://openreview.net/forum?id=gjeQKFxFpZ)
[//openreview.net/forum?id=gjeQKFxFpZ.](https://openreview.net/forum?id=gjeQKFxFpZ)


Heng Yang and Marco Pavone. Object pose estimation with statistical guarantees: Conformal
keypoint detection and geometric uncertainty propagation. In _Proceedings_ _of_ _the_ _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pp. 8947–8958, 2023.


Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees
and naive bayesian classifiers. In _Icml_, volume 1, pp. 609–616, 2001.


12


A PROOF


_Proof of Theorem 4.2._ By the definition of the Dirichlet distribution, for each coordinate _j_ we have


Deterministic means: _µ_ ¯ _t_ :=


E[ _pj_ ] = _[α][j]_


[0] _[ a][j]_

= _aj,_ and similarly E[ _qj_ ] = _[β][j]_
_α_ 0 _β_ 0


_[α][j]_ = _[α]_ [0] _[ a][j]_

_α_ 0 _α_ 0


= _aj._
_β_ 0


Since _p_ and _q_ are independent,
E[ _pj qj_ ] = E[ _pj_ ] E[ _qj_ ] = _a_ [2] _j_ _[.]_
Summing over _j_ = 1 _, . . ., K_ gives


�� _[K]_    E[ _p · q_ ] = E _pj qj_ =


_j_ =1


_K_


E[ _pj qj_ ] =

_j_ =1


_K_

- _a_ [2] _j_ _[,]_


_j_ =1


proving the exact mean. The bounds follow from [�] _j_ _[K]_ =1 _[a][j]_ [=] [1][ and the fact that] _[ a][j]_ _[≥]_ [0][, and the]
power-law exponent _τ_ is obtained by taking the negative logarithm base _K_ :


�� _K_   ln _j_ =1 _[a]_ _j_ [2]
_τ_ = _−_ = _⇒_

ln _K_


_K_

- _a_ [2] _j_ [=] _[ K]_ _[−][τ]_ _[.]_

_j_ =1


Thus the theorem is established.
**Theorem A.1** (Expected accuracy under heterogeneous logistic–normal) **.** _Let Zj_ _∼N_ (0 _, σj_ [2][)] _[ be]_
_independent Gaussian logits with σ_ 1 [2] _[≥]_ _[σ]_ 2 [2] _[≥· · · ≥]_ _[σ]_ _k_ [2] _[>]_ [ 0] _[.]_ _[If the variances satisfy the Lindeberg–]_
_Feller bounds_


_σ_ 1 [2] _[≤]_ _[C]_ _[<][ ∞][,]_


_then for every fixed exponent t >_ 1


_k_

- _σj_ [2] [=] _[ O]_ [(] _[k]_ [)] _[,]_ (A.1)

_j_ =1


_where_


_Ct,hetero_ :=


_Acct_ = _[C][t,][hetero]_ + _O_ - _k_ _[−]_ [3] _[/]_ [2][�] ( _k_ _→∞_ ) _,_

_k_


1      - _k_

- 1 - _k_ _k_ _j_ =1�� _[µ]_ 1 _[t,j]_ - _k_ - _,_ _µr,j_ = exp� 12 _[r]_ [2] _[σ]_ _j_ [2] - _._
_k_ _j_ =1 _[µ][t][−]_ [1] _[,j]_ _k_ _j_ =1 _[µ]_ [1] _[,j]_


The expectation of sampling accuracy with temperature _t_ is defined, as in Equation (7) of the main
paper, by





_Acct_ = E




_k_

- _Pj_ _[t][−]_ [1]

_j_ =1


_k_

- _Pj_ _[t]_


_j_ =1





_Tj_
_,_ _Pj_ =  - _k_ _,_ _Tj_ = _e_ _[Z][j]_ _._
_ℓ_ =1 _[T][ℓ]_



_k_


**Assumptions and notations.**
Latent logits: _Zj_ _∼N_ (0 _, σj_ [2][)] _[,]_ 1 _≤_ _j_ _≤_ _k,_ _σ_ 1 [2] _[≥]_ _[σ]_ 2 [2] _[≥· · · ≥]_ _[σ]_ _k_ [2] _[>]_ [ 0] _[.]_


Log–normals: _Tj_ := _e_ _[Z][j]_ _,_ _µr,j_ := E[ _Tj_ _[r]_ [] = exp]   - 12 _[r]_ [2] _[σ]_ _j_ [2]   - _,_

_ςr,j_ [2] [:= Var(] _[T]_ _j_ _[ r]_ [) =]                     - _e_ _[r]_ [2] _[σ]_ _j_ [2] _−_ 1� _µ_ [2] _r,j_ _[.]_


Softmax probabilities: _Pj_ = _[T][j]_ _,_ _T_ Σ :=

_T_ Σ


_k_

- _Tj._


_j_ =1


_k_

- _Tj_ _[t][−]_ [1] _._

_j_ =1


_k_

- _µt−_ 1 _,j,_ _µ_ ¯1 :=


_j_ =1


_k_

- _µ_ 1 _,j._


_j_ =1


Power sums: _Nk_ :=


_k_

- _Tj_ _[t][,]_ _Dk_ :=

_j_ =1


_k_

- _µt,j,_ _µ_ ¯ _t−_ 1 :=


_j_ =1


13


**Law of Large Numbers (Lindeberg–Feller).** Fix _r_ _∈_ [2 _, t_ ]. Define the centred summand _ξk,j_ [(] _[r]_ [)] [:=]

_Tj_ _[r]_ _[−]_ _[µ][r,j]_ [and total variance] _[ V]_ _k,r_ [2] [:=][ �] _j_ _[k]_ =1 _[ς]_ _r,j_ [2] [.] [If]

max _r,j_ [=] _[ O]_ [(1)] = _⇒_ _Vk,r_ [2] _[≍]_ _[k,]_
1 _≤j≤k_ _[ς]_ [2]

and the Lindeberg condition holds for every _ε >_ 0,


It has input _x_, _u ∈_ [0 _,_ 1], _π_, and _ν_ which can be seen as a generalized inverse of Equation 10.


14


1
_Vk,r_ [2]


_k_

- E� _ξk,j_ [(] _[r]_ [)2] **[1]** _[{|][ξ]_ _k,j_ [(] _[r]_ [)] _[|][ > εV][k,r][}]_ - _−−−−→k→∞_ [0] _[,]_

_j_ =1


then, since log–normals have super–polynomially decaying tails,


_k_

- _ξk,j_ [(] _[r]_ [)] [=] _[ O][p]_ [(] _[k][−]_ [1] _[/]_ [2][)] _[.]_

_j_ =1


1

~~�~~ _Vk,r_ [2]


_k_

- _ξk,j_ [(] _[r]_ [)] _[⇒]_ _[N]_ [(0] _[,]_ [ 1)] = _⇒_ 1

_k_
_j_ =1


Hence
_Nk −_ _µ_ ¯ _t_


_µ_ ¯ _t−_ 1 _T_ Σ _−_ _µ_ ¯1

_,_
_k_ _k_


_−_ _µ_ ¯ _t_ _Dk −_ _µ_ ¯ _t−_ 1

_,_
_k_ _k_


1 = _Op_   - _k_ _[−]_ [1] _[/]_ [2][�] _._

_k_


A sufficient explicit condition is again (A.1). For descending variances _σj_ [2] [=] _[ σ]_ 1 [2] _[j][−][β]_ [with any] _[ β]_ _[>]_ [ 0][,]
A.1 is satisfied.


**Fraction expansion with heterogeneous means.** Let
_Xk_ = _Nk −_ _µ_ ¯ _t,_ _Yk_ = _Dk −_ _µ_ ¯ _t−_ 1 _,_ _Zk_ = _T_ Σ _−_ _µ_ ¯1 _._
Then
_Nk_ _µ_ ¯ _t_ 1 + _Xk/µ_ ¯ _t_
= _·_
_DkT_ Σ _µ_ ¯ _t−_ 1 _µ_ ¯1 (1 + _Yk/µ_ ¯ _t−_ 1)(1 + _Zk/µ_ ¯1) _[.]_


A second–order Taylor expansion yields
_Nk_ _µ_ ¯ _t_
=
_DkT_ Σ _µ_ ¯ _t−_ 1 _µ_ ¯1


- 1 + _Op_ ( _k_ _[−]_ [1] _[/]_ [2] ) _._


**Expectation and scaling law.** Taking expectations cancels linear terms:


_µ_ ¯ _t_
_Acct_ =
_µ_ ¯ _t−_ 1 _µ_ ¯1


�1 + _O_ - _k_ _[−]_ [1] _[/]_ [2][��] _._


Since _µ_ ¯ _r_ = [�] _j_ _[k]_ =1 _[µ][r,j]_ [=] _[ k][µ]_ [ˆ] _[r]_ [, the ratio of averages is][ Θ(1)][ and]

_kµ_ ˆ _t_
_Acct_ = _[C][t,]_ [hetero] + _O_ ( _k_ _[−]_ [3] _[/]_ [2] ) _,_
( _kµ_ ˆ _t−_ 1)( _kµ_ ˆ1) [+] _[ O]_ [(] _[k][−]_ [3] _[/]_ [2][) =] _k_


with


_Ct,_ hetero :=


1      - _k_
_k_ _j_ =1 _[µ][t,j]_
_._

- 1 - _k_ �� 1 - _k_ _k_ _j_ =1 _[µ][t][−]_ [1] _[,j]_ _k_ _j_ =1 _[µ]_ [1] _[,j]_


B ADAPTIVE PREDICTION SETS (ROMANO ET AL., 2020)


Here is a description of the adaptive prediction sets (APS) method used in our paper. Suppose we
have the prediction distribution _**p**_ ( _x_ ) = _fθ_ ( _x_ ) and order this probability vector with the descending
order _p_ (1)( _x_ ) _≥_ _p_ (2)( _x_ ) _≥_ _. . . ≥_ _p_ ( _K_ )( _x_ ). The generalized conditional quantile function is defined
as,
_Q_ ( _x_ ; _p, ν_ ) = min _{k_ _∈{_ 1 _, . . ., K}_ : _p_ (1)( _x_ ) + _p_ (2)( _x_ ) + _. . ._ + _p_ ( _k_ )( _x_ ) _≥_ _ν},_ (10)
which produces the class index with the generalized quantile _ν_ _∈_ [0 _,_ 1]. The function _S_ can be
defined as

_S_ ( _x, u_ ; _p, ν_ ) =  - ‘ _y_ ’ indices of the _Q_ ( _x_ ; _p, ν_ ) _−_ 1 largest _py_ ( _x_ ) _,_ if _u ≤_ _U_ ( _x_ ; _p, ν_ ) _,_ (11)
‘ _y_ ’ indices of the _Q_ ( _x_ ; _p, ν_ ) largest _py_ ( _x_ ) _,_ otherwise _,_


where








 _._


1
_U_ ( _x_ ; _p, ν_ ) =
_p_ ( _Q_ ( _x_ ; _p,ν_ ))( _x_ )


_Q_ ( _x_ ; _p,ν_ )

 





_p_ ( _k_ )( _x_ ) _−_ _ν_

_k_ =1


On the calibration set _Dcal_, we compute a generalized inverse quantile conformity score using the
following function,
_E_ ( _x, y, u_ ; _p_ ) = min _{ν_ _∈_ [0 _,_ 1] : _y_ _∈S_ ( _x, u_ ; _p, ν_ ) _},_ (12)
which is the smallest quantile to ensure that the ground-truth class is contained in the prediction set
_S_ ( _x, u_ ; _p, ν_ ). With the conformity scores on calibration set _{Ei}_ _[N]_ _i_ =1 _[cal]_ [, we compute the] _[ ⌈]_ [(1] _[ −]_ _[α]_ [)(1 +]
_Ncal_ ) _⌉_ th largest value in the score set as _ν_ ˆcal. During inference, the prediction set is generated with
_S_ ( _**x**_ _[∗]_ _, u_ ; _p_ _[∗]_ _,_ ˆ _ν_ cal) for a test sample _**x**_ _[∗]_ .


C MORE EXPERIMENT


Fig. 7 and 8 shows the reliability diagrams of PS and CPAC on ImageNet and ViT-L when uniform
CP-ECE is used as the metric. The calibration error of CPAC is qualitatively better than that of PS.
Similarly, we visualize the uniform CP-ECE comparison of PS and CPAC for ViT-B on ImageNet in
Fig. 11 and 12. The results of standard CP-ECE are shown in Fig. 9 and 10 for ViT-L and Fig. 13
and 14 for ViT-B. Finally, we report the result of using PS and CPAC on ImageNet in Tab. 5 when
ResNet101 is used, which demonstrates the strength of CPAC in reducing uniform CP-ECE.


D THE USE OF LLM


The use of LLMs is restricted to language refinement, including grammar correction, sentence rephrasing, and improving the clarity of writing. No LLMs were used to generate research ideas, design
methodology, conduct experiments, or create results. All technical contributions, implementations,
and analyses presented in this paper are solely the work of the authors.


Std CP-ECE Uni CP-ECE Acc. Cov. PSS


Clean PS 8.68(0.01) 7.34(0.71) 80.35(0.09) 90.00(0.01) 6.20(0.10)
CPAC 7.93(0.14) 6.37(0.52) 77.68(0.14) 90.00(0.02) 7.81(0.29)
Norm-0.1 PS 9.09(0.07) 7.85(0.59) 78.67(0.06) 90.00(0.01) 7.93(0.23)
CPAC 8.46(0.09) 6.25(0.28) 78.67(0.06) 90.00(0.01) 7.13(0.18)
Norm-0.2 PS 9.85(0.12) 8.85(0.37) 74.13(0.05) 89.99(0.04) 14.82(0.15)
CPAC 8.37(0.09) 7.78(0.61) 70.58(0.41) 90.00(0.03) 18.17(0.97)
Norm-0.4 PS 10.74(0.09) 6.72(0.21) 57.19(0.16) 90.00(0.03) 62.37(0.49)
CPAC 7.58(0.33) 7.68(0.44) 51.97(0.27) 89.97(0.05) 93.87(2.89)
Norm-0.8 PS 5.02(0.12) 7.23(0.21) 12.30(0.06) 90.01(0.02) 429.82(3.04)
CPAC 4.37(0.12) 4.90(0.11) 12.50(0.21) 89.97(0.04) 456.89(3.74)
Blur-3 PS 9.27(0.13) 9.60(0.36) 77.06(0.10) 90.00(0.02) 10.52(0.13)
CPAC 8.46(0.14) 7.24(0.68) 75.19(0.32) 90.02(0.01) 11.76(0.44)
Blur-5 PS 9.81(0.06) 8.91(0.45) 75.24(0.09) 90.00(0.03) 13.39(0.14)
CPAC 8.73(0.19) 7.51(0.23) 73.09(0.10) 90.00(0.02) 15.48(0.46)
Blur-7 PS 9.98(0.04) 8.45(0.39) 74.39(0.06) 89.99(0.02) 15.08(0.22)
CPAC 8.65(0.19) 7.93(0.26) 71.12(0.20) 89.98(0.03) 19.14(0.84)
Drop-1 PS 10.93(0.07) 8.67(0.39) 71.04(0.07) 90.00(0.05) 21.45(0.34)
CPAC 8.96(0.11) 8.52(0.52) 68.26(0.14) 89.99(0.02) 26.39(0.70)
Drop-3 PS 11.42(0.04) 6.52(0.17) 54.91(0.11) 90.01(0.02) 72.49(0.24)
CPAC 8.22(0.39) 7.60(0.47) 52.37(0.76) 90.00(0.03) 89.10(5.19)
Drop-5 PS 8.95(0.14) 5.00(0.10) 35.38(0.07) 90.00(0.05) 171.88(1.52)
CPAC 6.63(0.53) 5.52(0.27) 32.77(0.53) 90.02(0.08) 241.31(10.71)
Drop-7 PS 5.09(0.08) 7.23(0.10) 11.97(0.09) 90.00(0.01) 424.25(0.65)
CPAC 4.72(0.17) 5.01(0.19) 12.73(0.22) 90.02(0.06) 450.48(3.68)


Table 4: Performance metrics of ViT-Base on ImageNet when the coverage is controlled to be 90%.


15


Figure 10: Reliability diagrams of ViT-L on ImageNet under different types of noise with standard
CP-ECE after CPAC.


16


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(1) Clean, Acc=81.94%, =0.22, Uni ECE=10.84%


0(7) Blur-5, Acc=77.92%, =0.20, Uni ECE=10.73%20 40 PSS (k)60 80 100


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(2) Norm-0.1, Acc=80.31%, =0.20, Uni ECE=10.37%


0(8) Blur-7, Acc=77.47%, =0.21, Uni ECE=10.37%20 40 PSS (k)60 80 100 120


0 50 PSS (k)100 150


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=76.99%, =0.21, Uni ECE=10.42%


(9) Drop-0.1, Acc=74.61%, =0.19, Uni ECE=11.83%0 50 PSS (k)100 150 200


0 50 100 150PSS (k)200 250 300


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=65.70%, =0.24, Uni ECE=9.14%


(10) Drop-0.3, Acc=60.93%, =0.22, Uni ECE=7.99%0 50 100 150PSS (k)200 250 300 350


0 100 200PSS (k) 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=28.33%, =0.34, Uni ECE=5.24%


(11) Drop-0.5, Acc=44.54%, =0.27, Uni ECE=6.14%0 100 200 PSS (k)300 400 500 600


0 100 200 PSS (k)300 400 500


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=79.00%, =0.20, Uni ECE=11.50%


(12) Drop-0.7, Acc=19.68%, =0.36, Uni ECE=5.62%0 25 50 PSS (k)75 100 125 150


0 100 200 PSS (k)300 400 500


Figure 7: Reliability diagrams of ViT-L on ImageNet under different types of noise with uniform
CP-ECE.


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=80.08%, =0.37, Uni ECE=5.92%


0 (7) Blur-5, Acc=76.41%, =0.28, Uni ECE=8.90%5 10 15PSS (k)20 25 30 35


0 20 40 PSS (k)60 80 100


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(2) Norm-0.1, Acc=78.66%, =0.34, Uni ECE=6.35%


0 (8) Blur-7, Acc=76.34%, =0.26, Uni ECE=9.20%10 PSS (k)20 30 40


0 20 40 PSS (k)60 80 100 120


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=75.24%, =0.32, Uni ECE=9.11%


(9) Drop-0.1, Acc=73.62%, =0.27, Uni ECE=10.08%0 20 PSS (k)40 60 80


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=64.12%, =0.31, Uni ECE=8.31%


(10) Drop-0.3, Acc=62.07%, =0.26, Uni ECE=8.55%0 50 100PSS (k)150 200 250


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=27.98%, =0.36, Uni ECE=4.93%


(11) Drop-0.5, Acc=45.02%, =0.31, Uni ECE=6.04%0 100 200 300PSS (k)400 500 600


0 100 200 PSS (k)300 400 500


1.0


0.9


0.8


0.7


0.6


0.5


0.4


0.3


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=77.57%, =0.28, Uni ECE=8.05%


(12) Drop-0.7, Acc=21.78%, =0.39, Uni ECE=4.74%0 20 PSS (k)40 60 80


0 100 200 300PSS (k)400 500 600


Figure 8: Reliability diagrams of ViT-L on ImageNet under different types of noise with uniform
CP-ECE after CPAC.


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(1) Clean, Acc=81.94%, =0.42, Std ECE=6.87%


0(7) Blur-5, Acc=77.92%, =0.41, Std ECE=8.38%20 40 PSS (k)60 80 100


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(2) Norm-0.1, Acc=80.31%, =0.49, Std ECE=7.51%


0(8) Blur-7, Acc=77.47%, =0.38, Std ECE=8.53%20 40 PSS (k)60 80 100 120


0 50 PSS (k)100 150


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=76.99%, =0.41, Std ECE=8.31%


(9) Drop-0.1, Acc=74.61%, =0.28, Std ECE=9.63%0 50 PSS (k)100 150 200


0 50 100 150PSS (k)200 250 300


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=65.70%, =0.33, Std ECE=10.04%


(10) Drop-0.3, Acc=60.93%, =0.26, Std ECE=11.31%0 50 100 150PSS (k)200 250 300 350


0 100 200PSS (k) 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=28.33%, =0.37, Std ECE=7.93%


(11) Drop-0.5, Acc=44.54%, =0.29, Std ECE=10.18%0 100 200 PSS (k)300 400 500 600


0 100 200 PSS (k)300 400 500


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=79.00%, =0.41, Std ECE=7.80%


(12) Drop-0.7, Acc=19.68%, =0.38, Std ECE=6.90%0 25 50 PSS (k)75 100 125 150


0 100 200 PSS (k)300 400 500


Figure 9: Reliability diagrams of ViT-L on ImageNet under different types of noise with standard
CP-ECE.


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=80.08%, =0.60, Std ECE=6.83%


0 (7) Blur-5, Acc=76.41%, =0.51, Std ECE=8.09%5 10 15PSS (k)20 25 30 35


0 20 40 PSS (k)60 80 100


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(2) Norm-0.1, Acc=78.66%, =0.60, Std ECE=7.32%


0 (8) Blur-7, Acc=76.34%, =0.52, Std ECE=8.28%10 PSS (k)20 30 40


0 20 40 PSS (k)60 80 100 120


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=75.24%, =0.50, Std ECE=8.05%


0(9) Drop-0.1, Acc=73.62%, =0.48, Std ECE=8.41%20 PSS (k)40 60 80


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=64.12%, =0.46, Std ECE=9.14%


(10) Drop-0.3, Acc=62.07%, =0.38, Std ECE=10.49%0 50 100PSS (k)150 200 250


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=27.98%, =0.41, Std ECE=7.37%


(11) Drop-0.5, Acc=45.02%, =0.37, Std ECE=9.73%0 100 200 300PSS (k)400 500 600


0 100 200 PSS (k)300 400 500


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=77.57%, =0.50, Std ECE=7.50%


(12) Drop-0.7, Acc=21.78%, =0.42, Std ECE=6.97%0 20 PSS (k)40 60 80


0 100 200 300PSS (k)400 500 600


Figure 12: Reliability diagrams of ViT-B on ImageNet under different types of noise with uniform
CP-ECE after CPAC.


17


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=80.42%, =0.21, Uni ECE=9.49%


0(7) Blur-5, Acc=77.17%, =0.21, Uni ECE=10.11%20 40PSS (k) 60 80 100


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(2) Norm-0.1, Acc=78.71%, =0.22, Uni ECE=8.84%


0(8) Blur-7, Acc=77.17%, =0.21, Uni ECE=10.11%20 40 PSS (k)60 80 100 120


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=74.13%, =0.22, Uni ECE=10.04%


(9) Drop-0.1, Acc=71.08%, =0.22, Uni ECE=9.55%0 50 100PSS (k) 150 200


0 50 100 PSS (k)150 200 250


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


(4) Norm-0.4, Acc=57.39%, =0.26, Uni ECE=7.00%


(10) Drop-0.3, Acc=55.10%, =0.25, Uni ECE=6.68%0 100 PSS (k)200 300 400


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=12.34%, =0.35, Uni ECE=7.28%


(11) Drop-0.5, Acc=35.44%, =0.31, Uni ECE=5.54%0 100 200 PSS (k)300 400 500 600


0 100 200PSS (k)300 400 500


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=77.17%, =0.21, Uni ECE=10.11%


(12) Drop-0.7, Acc=11.81%, =0.35, Uni ECE=7.29%0 25 50 75PSS (k)100 125 150 175


0 100 200 PSS (k)300 400 500 600


Figure 11: Reliability diagrams of ViT-B on ImageNet under different types of noise with uniform
CP-ECE.


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=77.62%, =0.34, Uni ECE=7.23%


0 (7) Blur-5, Acc=73.08%, =0.29, Uni ECE=7.97%10 PSS (k)20 30 40


0 20 40 PSS (k)60 80 100 120


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(2) Norm-0.1, Acc=75.48%, =0.36, Uni ECE=7.52%


0 (8) Blur-7, Acc=72.09%, =0.29, Uni ECE=7.84%10 20 PSS (k)30 40 50


0 20 40 60PSS (k)80 100 120


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=70.76%, =0.32, Uni ECE=7.51%


(9) Drop-0.1, Acc=68.41%, =0.29, Uni ECE=8.28%0 20 40 PSS (k)60 80 100


0 25 50 PSS (k)75 100 125 150


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=53.31%, =0.30, Uni ECE=7.81%


(10) Drop-0.3, Acc=51.35%, =0.28, Uni ECE=8.35%0 100 PSS (k)200 300 400


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=12.64%, =0.41, Uni ECE=4.72%


(11) Drop-0.5, Acc=33.22%, =0.31, Uni ECE=5.38%0 200 PSS (k)400 600


0 100 200 PSS (k)300 400 500 600


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=75.37%, =0.30, Uni ECE=7.72%


(12) Drop-0.7, Acc=12.51%, =0.41, Uni ECE=4.69%0 20 PSS (k)40 60 80


0 100 200 300PSS (k)400 500 600 700


Figure 14: Reliability diagrams of ViT-B on ImageNet under different types of noise with standard
CP-ECE after CPAC.


18


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=80.42%, =0.42, Std ECE=7.12%


0(7) Blur-5, Acc=77.17%, =0.42, Std ECE=8.10%20 40PSS (k) 60 80 100


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(2) Norm-0.1, Acc=78.71%, =0.43, Std ECE=7.79%


0(8) Blur-7, Acc=77.17%, =0.42, Std ECE=8.10%20 40 PSS (k)60 80 100 120


0 25 50 75PSS (k)100 125 150 175


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=74.13%, =0.37, Std ECE=8.45%


(9) Drop-0.1, Acc=71.08%, =0.32, Std ECE=9.97%0 50 100PSS (k) 150 200


0 50 100 PSS (k)150 200 250


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


(4) Norm-0.4, Acc=57.39%, =0.32, Std ECE=9.89%


(10) Drop-0.3, Acc=55.10%, =0.28, Std ECE=10.30%0 100 PSS (k)200 300 400


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=12.34%, =0.42, Std ECE=5.17%


(11) Drop-0.5, Acc=35.44%, =0.32, Std ECE=8.83%0 100 200 PSS (k)300 400 500 600


0 100 200PSS (k)300 400 500


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=77.17%, =0.42, Std ECE=8.10%


(12) Drop-0.7, Acc=11.81%, =0.42, Std ECE=5.09%0 25 50 75PSS (k)100 125 150 175


0 100 200 PSS (k)300 400 500 600


Figure 13: Reliability diagrams of ViT-B on ImageNet under different types of noise with standard
CP-ECE.


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


(1) Clean, Acc=77.62%, =0.61, Std ECE=7.46%


0 (7) Blur-5, Acc=73.08%, =0.46, Std ECE=8.21%10 PSS (k)20 30 40


0 20 40 PSS (k)60 80 100 120


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(2) Norm-0.1, Acc=75.48%, =0.50, Std ECE=7.86%


0 (8) Blur-7, Acc=72.09%, =0.47, Std ECE=8.30%10 20 PSS (k)30 40 50


0 20 40 60PSS (k)80 100 120


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(3) Norm-0.2, Acc=70.76%, =0.54, Std ECE=8.25%


(9) Drop-0.1, Acc=68.41%, =0.52, Std ECE=9.27%0 20 40 PSS (k)60 80 100


0 25 50 PSS (k)75 100 125 150


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(4) Norm-0.4, Acc=53.31%, =0.40, Std ECE=8.58%


(10) Drop-0.3, Acc=51.35%, =0.37, Std ECE=8.40%0 100 PSS (k)200 300 400


0 100 PSS (k)200 300 400


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


(5) Norm-0.8, Acc=12.64%, =0.45, Std ECE=4.38%


(11) Drop-0.5, Acc=33.22%, =0.34, Std ECE=6.26%0 200 PSS (k)400 600


0 100 200 PSS (k)300 400 500 600


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


(6) Blur-3, Acc=75.37%, =0.50, Std ECE=7.94%


(12) Drop-0.7, Acc=12.51%, =0.44, Std ECE=4.69%0 20 PSS (k)40 60 80


0 100 200 300PSS (k)400 500 600 700


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


|Col1|Col2|Std CP-ECE Uni CP-ECE Acc. Cov. PSS|
|---|---|---|
|Clean|PS<br>PS-Full<br>CPAC|7.01(0.14)<br>11.33(0.92)<br>81.93(0.13)<br>93.46(0.29)<br>16.04(1.59)<br>9.20(0.15)<br>7.85(0.23)<br>81.93(0.13)<br>94.77(0.21)<br>29.53(1.46)<br>9.16(0.13)<br>7.82(0.25)<br>81.93(0.13)<br>94.75(0.18)<br>29.57(1.18)|
|Norm-0.1|PS<br>PS-Full<br>CPAC|7.44(0.17)<br>10.87(1.41)<br>80.76(0.12)<br>93.36(0.33)<br>17.73(2.05)<br>9.12(0.19)<br>8.07(0.20)<br>80.76(0.12)<br>94.65(0.26)<br>29.99(1.62)<br>9.11(0.15)<br>7.91(0.24)<br>80.76(0.12)<br>94.63(0.20)<br>30.18(1.38)|
|Norm-0.2|PS<br>PS-Full<br>CPAC|7.90(0.14)<br>11.24(0.72)<br>79.18(0.13)<br>93.13(0.38)<br>19.89(2.27)<br>9.47(0.13)<br>7.66(0.19)<br>79.18(0.13)<br>94.29(0.20)<br>30.22(1.33)<br>9.40(0.19)<br>7.67(0.44)<br>79.20(0.14)<br>94.39(0.23)<br>30.92(1.42)|
|Norm-0.4|PS<br>PS-Full<br>CPAC|8.74(0.17)<br>9.58(0.38)<br>74.40(0.07)<br>92.63(0.31)<br>28.04(2.16)<br>9.41(0.14)<br>7.31(0.21)<br>74.40(0.07)<br>93.76(0.24)<br>35.76(1.57)<br>9.39(0.18)<br>7.08(0.21)<br>74.40(0.07)<br>93.83(0.34)<br>36.53(2.33)|
|Norm-0.8|PS<br>PS-Full<br>CPAC|9.23(0.06)<br>6.65(0.16)<br>58.57(0.11)<br>91.53(0.25)<br>67.89(2.20)<br>8.44(0.10)<br>5.51(0.11)<br>58.57(0.11)<br>91.97(0.11)<br>63.14(0.94)<br>8.36(0.07)<br>5.33(0.18)<br>58.63(0.09)<br>91.94(0.20)<br>62.27(1.74)|
|Blur-3|PS<br>PS-Full<br>CPAC|7.47(0.15)<br>10.71(0.36)<br>79.88(0.14)<br>93.31(0.28)<br>18.56(1.43)<br>9.31(0.18)<br>7.99(0.13)<br>79.88(0.14)<br>94.60(0.18)<br>30.69(1.47)<br>9.29(0.20)<br>7.82(0.43)<br>79.88(0.16)<br>94.59(0.10)<br>30.64(0.84)|
|Blur-5|PS<br>PS-Full<br>CPAC|8.04(0.18)<br>10.66(0.35)<br>78.17(0.11)<br>93.16(0.25)<br>22.38(1.51)<br>9.41(0.14)<br>7.42(0.36)<br>78.17(0.11)<br>94.44(0.22)<br>33.23(1.57)<br>9.39(0.15)<br>7.42(0.40)<br>78.17(0.11)<br>94.44(0.15)<br>33.15(1.09)|
|Blur-7|PS<br>PS-Full<br>CPAC|8.26(0.21)<br>10.23(0.38)<br>77.45(0.12)<br>93.06(0.25)<br>23.94(1.53)<br>9.38(0.20)<br>7.40(0.28)<br>77.45(0.12)<br>94.34(0.22)<br>34.34(1.59)<br>9.35(0.17)<br>7.27(0.17)<br>77.45(0.12)<br>94.34(0.17)<br>34.14(1.25)|
|Drop-1|PS<br>PS-Full<br>CPAC|10.92(0.17)<br>8.48(0.23)<br>67.03(0.13)<br>91.81(0.29)<br>55.23(2.38)<br>10.36(0.23)<br>6.63(0.30)<br>67.03(0.13)<br>92.59(0.25)<br>55.07(1.98)<br>9.97(0.24)<br>6.40(0.19)<br>67.37(0.13)<br>92.79(0.24)<br>54.24(1.85)|
|Drop-3|PS<br>PS-Full<br>CPAC|10.72(0.31)<br>6.36(0.18)<br>52.05(0.12)<br>90.64(0.34)<br>108.84(4.37)<br>9.87(0.31)<br>5.69(0.25)<br>52.05(0.12)<br>90.88(0.35)<br>98.10(3.90)<br>9.10(0.23)<br>5.42(0.28)<br>53.11(0.12)<br>91.11(0.36)<br>87.01(3.53)|
|Drop-5|PS<br>PS-Full<br>CPAC|8.25(0.20)<br>5.03(0.17)<br>37.59(0.07)<br>90.29(0.37)<br>182.23(6.52)<br>8.02(0.12)<br>4.89(0.20)<br>37.59(0.07)<br>90.29(0.33)<br>176.38(5.28)<br>7.31(0.13)<br>4.71(0.09)<br>39.03(0.10)<br>90.35(0.30)<br>142.69(3.28)|
|Drop-7|PS<br>PS-Full<br>CPAC|4.75(0.14)<br>5.55(0.07)<br>19.41(0.06)<br>89.94(0.27)<br>328.26(6.81)<br>5.82(0.17)<br>5.48(0.08)<br>19.41(0.06)<br>90.00(0.31)<br>359.11(7.00)<br>4.77(0.26)<br>5.09(0.28)<br>21.14(0.21)<br>89.47(0.28)<br>300.94(5.65)|


Table 5: Result of ResNet101 on ImageNet-1k.


19