# LEARNING INTERPRETABLE MODELS USING UNCER## TAINTY ORACLES


**Anonymous authors**
Paper under double-blind review


ABSTRACT


A desirable property of interpretable models is small size, so that they are easily
understandable by humans. This leads to the following challenges: (a) small sizes
typically lead to diminished accuracy, and, (b) different techniques offer bespoke
levers, e.g., L1 regularization, for making this size-accuracy trade-off that might
be insufficient to reach the desired balance.

We address these challenges here. Earlier work has shown that learning the training distribution creates accurate small models. Our contribution is a new technique that exploits this idea. The training distribution is modeled as a Dirichlet Process for flexibility in representation. Its parameters are learned using
Bayesian Optimization; a design choice that makes the technique applicable to
non-differentiable loss functions. To avoid challenges with high data dimensionality, the data is first projected down to one-dimension using uncertainty scores of
a separate probabilistic model, that we refer to as the uncertainty oracle.

Based on exhaustive experiments we show that this technique possesses multiple merits: (1) it significantly enhances small model accuracies, (2) is versatile:
it may be applied to different model families with varying notions of size, e.g.,
depth of a decision tree, non-zero coefficients in a linear model, simultaneously
the maximum depth of a tree and number of trees in Gradient Boosted Models, (3)
is practically convenient because it needs only one hyperparameter to be set and
works with non-differentiable losses, (4) works across different feature spaces between the uncertainty oracle and the interpretable model, e.g., a Gated Recurrent
Unit trained using character sequences may be used as an oracle for a Decision
Tree that uses character n-grams, and, (5) may augment the accuracies of fairly old
techniques to be competitive with recent task-specialized techniques, e.g., CART
Decision Tree (1984) vs Iterative Mistake Minimization (2020), on the task of
cluster explanation.


1 INTRODUCTION


In recent years, Machine Learning (ML) models have become increasingly pervasive in various real
world systems. This has led to a growing emphasis on models to be _understandable_, especially
in high human-impact domains, e.g., medicine and healthcare (Caruana et al., 2015; Mienye et al.,
2024), defence applications (Gunning, 2016; Moustafa et al., 2023), law enforcement (Angwin et al.,
2016; Hall et al., 2022; Herrewijnen et al., 2024).


An important aspect of model interpretability is its size (smaller is better); this has been established
through user studies (Feldman, 2000; Kulesza et al., 2013; Piltaver et al., 2016; Lage et al., 2019;
Poursabzi-Sangdeh et al., 2021), and is also evidenced by its popularity as an algorithm design
criteria (Tibshirani, 1996; Ribeiro et al., 2016; Herman, 2017; Lipton, 2018; Murdoch et al., 2019;
Lakkaraju et al., 2016; Good et al., 2023). However, smaller sizes typically imply relatively lower
capacity and thus, lower accuracy. A practitioner may control this size-accuracy trade-off using
bespoke levers offered by a training algorithm, e.g., early stopping in Decision Trees (DT), L1
regularization in linear models. However, this presents certain challenges: (1) one needs to be
intimately aware of how various hyperparameters (hence referred to as _hyperparams_ ) interact, and
(2) the desired trade-off might not even be achievable within its hyperparam search space.


1


Figure 1: Application of our technique is shown on the toy dataset in (a). Learning a DT _constrained_
to a depth of 5 using the CART (Breiman et al., 1984) algorithm produces the regions shown in (b).
Additionally learning the training distribution using our technique produces the regions in (c). For
both (b) and (c) the F1-macro scores on a held-out set are reported.


Here we propose a model-agnostic [1] technique that often produces better accuracies for small-sized
models on classification problems. The underlying strategy is to learn a distribution over training
instances, that represents their informational value for learning, and sample a new training set accordingly; models thus constructed have been shown to possess favorable size-accuracy trade-offs
(Ghose & Ravindran, 2020). Our technique is an implementation of this principle.


The distribution used is a mixture model based on the _Dirichlet_ _Process_ - picked for its flexibility
of representation and maturity within the Bayesian nonparametrics community. Its parameters are
learned using _Bayesian_ _Optimization_, so as to accommodate non-differentiable losses, e.g., many
DT and rules learners. To make this process computationally efficient, we avoid directly learning
the distribution over the input space which may have high dimensionality. Instead we first project
instances down to a single dimension, using an auxiliary model’s prediction uncertainty scores. We
refer to this model as the _uncertainty oracle_ .


As an illustration, consider the toy dataset in Figure 1(a). Figure 1(b) visualizes class regions learned
by a DT of _depth_ = 5 using the CART (Breiman et al., 1984) algorithm. The F1-macro score on
a held-out set is 0 _._ 63. When the training distribution is also learned using our technique, we obtain
the regions in Figure 1(c) and a F1-macro score of 0 _._ 77, for the same tree depth. The oracle used is
a Gradient Boosted Model (GBM) Friedman (2001).


**Our primary contribution** is a model-agnostic technique that produces small accurate models. It
is also agnostic to the notion of model _size_, e.g., number of terms of with non-zero coefficients in a
linear model or depth of a DT, _both_ the number of trees and depth per tree in GBMs. We show that
this produces relative improvements of _∼_ 100% in some cases. It is convenient to use as it works
with with non-differentiable losses, and only one hyperparam needs to be set.


**Additionally**, we show that: (1) it is more accurate than its predecessor (Ghose & Ravindran, 2020),
(2) it can elevate the performance of fairly old techniques to be competitive with relatively new
ones, and (3) can use an uncertainty oracle that is trained on a different feature space than what the
target model uses. The last property allows for a broad choice of oracles, e.g., in the case of text
classification, the oracle might be a _Gated Recurrent Unit (GRU)_ that is learned using a sequence of
characters while the target model might be a DT over _n-grams_ .


The rest of the paper is organized as follows: we first review related work in §2. We then detail
our technique in §3. We follow that up with rigorous empirical validations in §4.A side effect of
allowing non-differentiable losses is high running times; we discuss this limitation, and a mitigation
in §5. Finally, we conclude with a discussion on future work in §6.


1We use the term to mean agnostic to the model _family_, as is accepted usage in the area of XAI, e.g., SHAP
(Lundberg & Lee, 2017), LIME (Ribeiro et al., 2016).


2


toy dataset


when _depth=5_ . **F1=0.63** .


_depth=5_ using our technique. **F1=0.77** .


Figure 2: Overview of our technique. _Left_ : Training instances are characterized by their proximity
to class boundaries. As a proxy for this quantity, we use the prediction uncertainty scores of a probabilistic oracle (these may also be seen as an 1D _projection_ ): higher uncertainty indicates proximity
to a boundary. These scores are calculated _once_ . _Right_ : The size-constrained model is learned iteratively. A sampling distribution, parameterized by Φ, over the uncertainty values (shown in Step
1) is used to sample training instances (as in Step 2), which is used to train a size-constrained
model (shown in Step 3). Its accuracy on a held-out set - Step 4 - is used to modify Φ. This
loop, Steps 1-4, is executed by a BayesOpt algorithm.


2 RELATED WORK


The concept of using a different training distribution relative to test is common in the case of class
imbalance, e.g., undersample the majority class data (Japkowicz & Stephen, 2002; Chawla et al.,
2002; He et al., 2008; Santhiappan et al., 2018), but it was shown to be a general strategy for
improving accuracy in Ghose & Ravindran (2020). Their technique relies on a specialized DT,
called _density_ _tree_, that encodes the geometric placement of training data. We believe that using
these trees - which are primarily learned using the CART algorithm - inherently limits the accuracy
of their technique. This work may be seen as a non-trivial extension: since the uncertainty oracle
can come from an arbitrary model family, it provides greater flexibility and accuracy.


The interaction of two models - the oracle and the interpretable model - suggests an overlap with
the area of _Knowledge Distillation_ Gou et al. (2021). But there is a critical difference: _in theory, we_
_don’t require the oracle model_ ; here it happens to be a convenient _tool for dimensionality reduction_ .
Indeed, there are other ways to achieve a similar outcome, e.g., within _Active Learning_, it is common
to infer proximity to a class boundary by noting the labels of an instance’s neighbors (Margatina
et al., 2021; Chen et al., 2023); these setups might be thought of as rudimentary _k-Nearest Neighbor_
_(kNN)_ models. With our use of the oracle, we avoid having to worry about neighborhood-related
hyperparams, such as neighbor distance.This should not be seen as distillation for the same reason
as we don’t consider using such _kNN_ s as effecting distillation. The oracle’s peripheral role is also
underscored by the fact that its labels are ignored. This lack of fidelity wrt the oracle is also why
our technique shouldn’t be seen an explanation technique, i.e., _XAI_, such as _TREPAN_ (Craven &
Shavlik, 1995) or _LIME_ (Ribeiro et al., 2016).


3 METHODOLOGY


We begin describing our technique with an overview. This is then used as a foundation for introducing details.


3.1 OVERVIEW


Our technique is visualized in Figure 2. Instead of learning the training distribution directly, which
might be expensive because of the dimensionality of the data, we first project the data down to _one_


3


One-time


Sizes indicate
weights. Grey
circles indicate
instances that
were dropped.


1D projection of training data
based on how hard is it to


proximity to boundary


proximity to boundary


_(measured using the oracle's prediction_
_uncertainty - higher scores indicate_
_closeness to boundary)_


dimension. This is done just once, and is shown in the left panel in Figure 2. Since we are solving for
classification, we want this dimension to correspond to the “classifiability” of an in instance, or how
close is an instance to a class boundary. As a tractable proxy for this property, we train a separate
highly accurate probabilistic _oracle_ model on the training data [2], and use its prediction _uncertainty_
score as the projected value; high uncertainty scores typically denote proximity to class boundaries
Lewis & Catlett (1994).


The distribution is modeled as an _Infinite_ _Beta_ _Mixture_ _Model_ using a _Dirichlet_ _Process_, which is
_iteratively_ learned. Step 1 on the right panel in Figure 2 shows the current distribution, based on
which training data is sampled (Step 2). The size-constrained model of interest is then trained on
this sample - Step 3 - and its accuracy on a held-out set is calculated - Step 4. This score is
used as a feedback for the optimizer which repeats the process to learn better distribution parameters.
We use _Bayesian Optimization (BayesOpt)_ (Shahriari et al., 2016; Garnett, 2023) to accommodate
models with _non-differentiable_ loss functions. Note that we can’t just pick highly uncertain points,
because that has not been shown to consistently work well (Ghose & Nguyen, 2024).


3.2 TERMINOLOGY AND NOTATION


We introduce some nomenclature before discussing our algorithm.


1. A dataset is denoted as a set of instance-label pairs, _D_ = _{_ ( _x_ 1 _, y_ 1) _,_ ( _x_ 2 _, y_ 2) _, ...,_ ( _xN_ _, yN_ ) _}_ .
A joint distribution over a dataset is denoted by _p_ ( _X, Y_ ).


2. To distinguish between the distribution we are given (in form of the dataset) and the one
we learn, we refer to the former as the _original_ distribution. In _all_ experiments here, the
test and held-out data follow the original distribution; for the training data, we learn a new
distribution.


3. We let _acc_ ( _M, p_ ) denote some classification accuracy metric for model _M_ on data represented _p_ ( _X, Y_ ).


4. _trainF_ _,f_ ( _p, η_ ) is understood to produce a model of size _η_ (for some pre-decided notion of
size) from the model family _F_ using a specific training algorithm _f_ .
For instance, _F_ might represent DTs and _f_ might be the CART algorithm, and _η_ = 5 might
denote a DT of _depth_ = 5. We let _η_ = _∗_ denote unbounded size.


Let us state our objective using this notation. Typically, a model is trained on the same distribution as
the test (on which it is evaluated), i.e., we evaluate _acc_ ( _trainF_ _,f_ ( _p, η_ ) _, p_ ). In this work, the training
distribution is allowed to be different relative to the test. In other words, we seek _p_ _[′]_ such that:


arg max _acc_ ( _trainF_ _,f_ ( _p_ _[′]_ _, η_ ) _, p_ ) (1)
_p_ _[′]_


3.3 ALGORITHM


Referring to the high-level flow in Figure 2, we note that the proposed technique relies on a few
important ingredients. These are described below, while a more comprehensive discussion may be
found in §A.4:


1. **Uncertainty** **score** : This is needed for the one-time projection using the oracle. There
are multiple ways to measure prediction uncertainty; here we choose _margin_ _uncertainty_
(Scheffer et al., 2001), since (a) it accounts for prediction probabilities of different classes,
(b) while also producing high scores even with two dominant predicted classes in a setting
with more classes. The uncertainty score for _x_, as provided by model _M_, is denoted by
_uM_ ( _x_ ) _∈_ [0 _,_ 1]. The margin uncertainty is calculated as:


_uM_ ( _x_ ) _←_ 1 _−_ ( _pC_ 1 _−_ _pC_ 2) (2)


Here, _pC_ 1 and _pC_ 2 denote the probabilities of the most confident and next most confident
classes. See §A.1 for further details.


2Performed using cross-validation or using a random held-out set, to avoid overfitting.


4


2. **Density** **model** : Since we want to _learn_ a distribution, we want the representation to be
flexible. We encode the density as a _mixture model_ of _Beta_ distributions. We use the latter
since (a) their support matches the range of uncertainty scores, i.e., _uM_ ( _x_ ) _∈_ [0 _,_ 1], and
(b) a _Beta_ _mixture_ _model_ can approximate any distribution in [0 _,_ 1] arbitrarily well (Diaconis & Ylvisaker, 1983). Further, in the interest of flexibility, we refrain from explicitly
dictating the number of _Beta_ components, and thus, we use an _Infinite_ Beta Mixture Model
(IBMM), where the component assignments are decided by a standard _Dirichlet_ _Process_
_(DP)_ (Ferguson, 1973). This is a popular tool in the area of _Bayesian Nonparametrics_ Yee
Whye Teh & Blei (2006); Wang et al. (2011). Another advantage of this formulation is that
it leads to a fixed number of parameters irrespective of the number of active components
which makes it easy to pick an optimizer. We note here that Ghose & Ravindran (2020)
also use a DP-based IBMM, but for modeling the height of density trees.
Two sets of parameters are required to describe this density model:

(a) The shape parameters _Ai, Bi_ of the _i_ _[th]_ _Beta_ component. These are separately sampled
from prior distributions that are themselves _Beta_ distributions, with shape parameters
_a, b_ and _a_ _[′]_ _, b_ _[′]_ respectively. Since naively doing this would restrict _Ai_ or _Bi_ to _Beta’s_
support, i.e., [0 _,_ 1], we also multiply the sampled value by a variable _scale_, that we
set to be large enough to cover the family of component distributions we require [3] .
Effectively then, _Ai_ _∼_ _scale × Beta_ ( _a, b_ ) and _Bi_ _∼_ _scale × Beta_ ( _a_ _[′]_ _, b_ _[′]_ ).
(b) The DP needs just a _concentration_ parameter _α_ _∈_ R _>_ 0 that decides the number of
active components, i.e., ones with instances assigned to them [4] .
In all, the density model requires _five_ parameters, which we denote as Ψ = _{α, a, b, a_ _[′]_ _, b_ _[′]_ _}_ .
To sample _Ns_ instances given Ψ, we first determine the number of instances per component
using a standard technique like the _Chinese_ _Restaurant_ _Process_ (Aldous, 1985) and then
sample component-wise. Please see §A.2 for details.

3. **Optimization** : As mentioned earlier, we use BayesOpt to accommodate non-differentiable
losses. It is also resilient to noise, which is relevant due to factors such as randomized
initialization of model parameters, different dataset splits across trials, etc. Specifically,
we use the _hyperopt_ library (Bergstra et al., 2013), which implements the _Tree Structured_
_Parzen Estimator (TPE)_ algorithm Bergstra et al. (2011). _Because there is no tight coupling_
_between our formulation and the optimizer, it is possible to use a different BayesOpt library_ .
This can be a crucial practical consideration, and is discussed in §5.
For optimization, in addition to Ψ, we retain the following parameters, originally introduced in Ghose & Ravindran (2020):


(a) _Ns_ : This is the sample size - this is also learned.
(b) _po_ _∈_ [0 _,_ 1]: Proportion of the new training set that is uniformly sampled from the
original training data. This serves two purposes: (1) it acts as a “shortcut” for the
optimizer to mix in the original distribution as needed, and (2) it serves as a “probe
variable”, i.e., it shows how much of the original distribution is actually needed for
good accuracies.

Accounting for these, we now have a total of _seven_ optimization variables: Ψ =
_{α, a, b, a_ _[′]_ _, b_ _[′]_ _}, Ns, po_, which are iteratively optimized, till the **budgeted** **number** **of** **it-**
**erations**, _T_, are exhausted. These variables are collectively denoted as Φ = _{_ Ψ _, Ns, po}_ .
Algorithm 1 outlines the overall technique; here the interpretable and oracle model families are denoted by _I_ and _O_, and the respective training algorithms are denoted by _h_ and _g_
respectively. §A.4 provides additional details around model selection, robust estimation of
_acc_, etc.


**Optimization** **variables** **and** **parameters** : The task of the optimizer is to find Φ that maximizes
the held-out accuracy (line 11 in Algorithm 1) within _T_ iterations. The optimizer here accepts _box_
_constraints_, and as such their lower/upper bounds, which need to be set by the user, are _parameters_
(along with _T_ ) of the technique. We discuss in §A.3 that reasonable default bounds exist for parameters Φ, e.g., its easy to see _po_ _∈_ [0 _,_ 1]. So, **in practice,** _T_ **is the only parameter that a user needs**
**to set** .


3 **NOTE** : This is fixed at a value of 10000 and not learned; hence it isn’t counted as a parameter.
4Of course, in theory, there are an infinite number of components, but the number of _active_ components
grows with data.


5


**Algorithm 1:** Learning interpretable model using oracle

**Data:** Dataset _D_, model size _η_, _trainO,h_ (), _trainI,g_ (), iterations _T_
**Result:** Optimal parameters Φ _[∗]_, test set accuracy _stest_ at Φ _[∗]_, and interpretable model _M_ _[∗]_ at Φ _[∗]_

**1** Create splits _Dtrain, Dval, Dtest_ from _D_, stratified wrt labels. Here
_|Dtrain|_ : _|Dval|_ : _|Dtest|_ :: 60 : 20 : 20.

**2** _MO_ _←_ _trainO,h_ ( _Dtrain, ∗_ )

**3** **for** _t ←_ 1 **to** _T_ **do**

**4** Φ _t_ _←_ _suggest_ ( _s_ 0 _, s_ 1 _, ...st−_ 1 _,_ Φ0 _,_ Φ1 _, ...,_ Φ _t−_ 1) // _s_ 0 _,_ Φ0 initialized at _t_ = 0,
see text. Note: Φ _t_ = _{_ Ψ _t, Ns,t, po,t}_ where Ψ _t_ = _{αt, at, bt, a_ _[′]_ _t_ _[, b]_ _t_ _[′]_ _[}]_ [.]

**5** _No_ _←_ _po,t × Ns,t_

**6** _Nu_ _←_ _Ns_ _t −_ _No_

**7** _Do_ _←_ uniformly sample with replacement _No_ points from _Dtrain_

**8** _Du_ _←_ sample _Nu_ points from _Dtrain_ using the DP-based IBMM given current values for
_Nu, MO, Dtrain,_ Ψ _t_ // see Algorithm A.2 for details

**9** _Ds_ _←_ _Do ⊎_ _Du_ // _Do_, _Du_ are assumed to be multisets

**10** _Mt_ _←_ _trainI,g_ ( _Ds, η_ )

**11** _st_ _←_ _acc_ ( _Mt, Dval_ )

**12** **end**


**Smoothing** : A final practical consideration is the smoothness of the optimization landscape. Uncertainty scores over the training data may often result in a density that isn’t smooth, making it difficult
to learn a good distribution. We redress this by explicitly smoothing the density. We detail this in
§A.5.


This concludes our discussion of algorithmic details; next, we look at empirical validation.


4 EXPERIMENTS


We have performed extensive empirical investigations to validate the utility of our technique. These
may be grouped in the following manner:


1. Those that establish the _effectiveness_ of the technique in various settings, i.e., different
datasets, interpretable models and oracles, across different model sizes. This is our key
result.


2. _Benchmarking_ against the density tree approach.


3. _Competitiveness_ : Even if our technique produces significant improvements, it leaves open
the question of these gains being competitive with task-specific techniques, e.g., clusterexplanation trees and prototype-based classifiers. These set of experiments affirmatively
answer this question.


4. _Additional properties_    - while these are not as rigorous as the previous groups, they highlight
some interesting properties: (a) model size can be multivariate, and (b) it is possible to have
different feature spaces between the oracle and the target models.


All experiments were performed on an Intel i7-7700HQ machine with 32 GB RAM.


Due to space constraints, only the key result - point 1 above - is presented in detail in the main paper
(some aspects are relegated to the Appendix), while other findings are only summarized here, with
details being provided in the Appendix.


6


**13** _t_ _[∗]_ _←_ arg max _t {s_ 1 _, s_ 2 _, ..., sT −_ 1 _, sT }_


**14** Φ _[∗]_ _←_ Φ _t∗_


**15** _M_ _[∗]_ _←_ _Mt∗_


**16** _stest_ _←_ _acc_ ( _M_ _[∗]_ _, Dtest_ )


**17** **return** Φ _[∗]_, _stest_, _M_ _[∗]_


4.1 EFFECTIVENESS OF OUR TECHNIQUE


We begin describing this set of experiments with the various settings.


4.1.1 EXPERIMENT SETTINGS


We tested our technique across the following configurations:


1. Datasets: We use the following 13 publicly available standard classification datasets for
our experiments: _cod-rna,_ _ijcnn1,_ _higgs,_ _covtype.binary,_ _phishing,_ _a1a,_ _pendigits,_ _letter,_
_Sensorless,_ _senseit_ ~~_a_~~ _co,_ _senseit_ ~~_s_~~ _ei,_ _covtype,_ _connect-4_ . These were obtained from the
LIBSVM website (Chang & Lin, 2011a). For details, such as number of classes and extent
of imbalance, please see §A.7.
10000 instances from each dataset are used. The split ratio used in Algorithm 1 is _|Dtrain|_ :
_|Dval|_ : _|Dtest|_ :: 60 : 20 : 20, where the splits are stratified wrt labels.

2. Interpretable model families: we use _Linear Probability Models (LPM)_ [5] and the DTs (produced by the CART algorithm). These were picked as they are commonly considered
interpretable R¨az (2024).
The notion of model size for LPMs is the number of non-zero coefficients, and sizes _η_ _∈_
_{_ 1 _,_ 2 _, ...,_ 15 _}_ are explored (except for _cod-rna_, that has 8 features, and so we cannot have a
sizes greater than 8).

For DTs, the notion of size is depth. For a dataset, we first learn a tree (with no size
constraints) with the highest _F1-macro_ score using standard 5 _−_ fold cross-validation. We
refer to this as the optimal tree _Topt_, and its depth as _depth_ ( _Topt_ ). We then explore model
sizes _η_ _∈{_ 1 _,_ 2 _, ..., min_ ( _depth_ ( _Topt_ ) _,_ 15) _}_ . Stopping early makes sense since the model
is saturated in its learning at _depth_ ( _Topt_ ); changing the input distribution is not helpful
beyond this point.


3. Oracle families: As oracles we use _Random_ _Forests_ _(RF)_ (Breiman, 2001) and _GBMs_
(Friedman, 2001). They were learned using cross-validation or using a held-out set, to
avoid overfitting, and were calibrated (Platt, 1999; Niculescu-Mizil & Caruana, 2005) for
reliable probability estimates.


4. Optimization budget: For DTs, we use _T_ = 3000, while for LPMs _T_ = 1000 is used.
These values were determined based on limited search. The budget for LPMs is lower
since for multi-class datasets (7 of 13 here) we construct one-vs-all models which makes
training LPMs time-consuming.


4.1.2 METRICS


For various combinations of models and oracles, i.e., _{LPM, DT_ _} × {GBM, RF_ _}_, we measure
the percentage relative improvement in the _F1-macro_ score (on the test set _Dtest_ ) in terms of the
baseline score _F_ 1 _[base]_ _test_ [and the one produced by our model,] _[ F]_ [1] _test_ _[∗]_ [:]


_test_ _[−]_ _[F]_ [1] _test_ _[base]_ [)]
_δF_ 1 _test_ = [100] _[ ×]_ [ (] _[F]_ [1] _[∗]_ (3)
_F_ 1 _[base]_ _test_


We use the macro score since its not impacted by class imbalance.


In the interest of robustness we run **five** **trials** **per** **configuration**, i.e., a combination of dataset,
oracle family, model family and size, and utilize the validation set to accept the model produced
by our technique _M_ _[∗]_ . Specifically: indexing trials with _i_, we conduct an independent _t-test_ on
_{F_ 1 _[∗]_ _val_ _[}]_ [1] _[≤][i][≤]_ [5][ and] _[ {][F]_ [1] _[base]_ _val_ _[}]_ [1] _[≤][i][≤]_ [5][.] [The null hypothesis is that] _[ M][ ∗]_ [doesn’t produce results different]
_M_ _[base]_ . If we can reject the null at a significance of _p_ = 0 _._ 1, we report _δF_ 1 _test_ as in Equation 3 [6],
else we report _δF_ 1 _test_ = 0, i.e., we reject _M_ _[∗]_ . Here _δF_ 1 _test_ _∈_ ( _−∞, ∞_ ); negative values are
possible since we pick a model based on _Dval_ while we report based on _Dtest_ .


5We have not used the more common _Logistic_ _Regression_ because: (1) LPMs are considered more interpretable (Mood, 2010), and (2) LPMs results are indicative of behavior of linear models in general.
6The test scores from different trials are averaged first.


7


4.1.3 OBSERVATIONS


Table 1 shows a portion of the results in the interest of space - for complete results, and analysis of
statistical significance (using the _Wilcoxon signed-rank_ test (Wilcoxon, 1945)), please see §A.8.


Table 1: This table shows the average improvements, _δF_ 1, over **five runs** for the combinations **model=** _**{LPM,**_
_**DT}**_ and **oracle=** _**GBM**_, for different model sizes. The improvements are measured relative to the model at the
first iteration. Here, _δF_ 1 _∈_ ( _−∞, ∞_ ). Negative improvements are shown in underlined. **Complete results,**
**including analysis of statistical significance, are presented in §A.8** .


dataset model ~~o~~ ra size=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15


cod-rna lpm ~~g~~ bm 1.39 12.53 14.76 15.73 14.97 12.00 0.00 0.08  -  -  -  -  -  -  dt ~~g~~ bm 0.00 0.00 0.00 1.26 0.00 0.00 0.00 0.00 -0.28 0.08           -           -           -           -           

ijcnn1 lpm ~~g~~ bm -0.16 3.36 3.93 0.00 5.19 4.18 3.85 3.79 3.69 2.99 2.97 3.21 3.11 3.26 3.02
dt ~~g~~ bm 1.96 12.00 10.15 11.37 10.63 7.18 3.63 4.52 2.91 1.78 1.93 2.29 1.47 2.26 0.00


higgs lpm ~~g~~ bm 29.29 17.80 11.40 6.56 3.06 2.68 3.16 2.90 2.67 2.82 2.65 1.79 2.62 2.19 1.63
dt ~~g~~ bm 0.00 0.00 1.86 0.26 0.93 0.45           -           -           -           -           -           -           -           -           

covtype.binary lpm ~~g~~ bm 76.52 66.39 29.17 12.51 9.18 5.28 4.94 4.56 3.92 3.56 3.62 3.31 2.59 2.83 2.39
dt ~~g~~ bm 0.00 0.00 2.35 1.27 1.18 1.11 0.00 0.00 0.00           -           -           -           -           -           

phishing lpm ~~g~~ bm 0.00 1.88 2.88 3.05 3.22 3.25 2.99 1.69 1.42 1.45 1.29 0.00 0.00 0.00 0.00
dt ~~g~~ bm 0.00 0.00 0.00 0.07 0.39 0.00 0.28 0.22 0.44 0.23 0.00 0.00 0.00 0.00 0.00


a1a lpm ~~g~~ bm 0.00 2.55 7.58 8.98 8.40 8.03 8.90 8.23 8.17 7.90 5.96 7.10 6.97 6.18 5.73
dt ~~g~~ bm 0.00 5.54 2.39 3.84 3.55 2.55 1.51 2.25 4.87           -           -           -           -           -           

pendigits lpm ~~g~~ bm 51.39 23.44 16.18 8.95 8.84 6.63 4.86 1.83 2.27 2.16 2.44 2.16 3.33 2.97 2.73
dt ~~g~~ bm 14.02 6.72 5.11 13.14 6.42 4.20 2.46 1.09 0.98 0.16 -0.26 0.00 0.00 0.00 0.00


letter lpm ~~g~~ bm 57.06 48.48 59.85 29.76 36.09 19.27 20.37 16.08 17.55 15.16 17.26 16.51 18.46 17.19 15.55
dt ~~g~~ bm 0.00 13.98 25.05 33.96 32.05 15.49 11.17 0.00 4.26 3.50 1.99 0.00 0.00 0.00 0.00


Sensorless lpm ~~g~~ bm 216.47 257.56 178.31 117.01 90.70 83.90 73.50 65.95 61.57 57.97 56.54 57.15 55.45 66.24 68.24
dt ~~g~~ bm -0.01 42.42 68.13 44.38 17.39 10.32 1.82 1.44 0.79 0.64 0.41 0.12 0.00 -0.02 0.34


senseit ~~a~~ co lpm ~~g~~ bm 173.71 170.68 63.95 44.20 33.49 22.99 19.14 13.50 10.29 7.59 6.26 5.92 5.30 4.89 4.32
dt ~~g~~ bm 14.89 0.00 3.71 2.32 4.85 0.81 0.00           -           -           -           -           -           -           -           

senseit ~~s~~ ei lpm ~~g~~ bm 160.59 65.27 23.44 10.48 6.76 4.86 4.82 4.46 4.79 4.12 4.54 5.17 3.91 4.21 4.46
dt ~~g~~ bm 2.66 1.01 3.49 2.29 0.95 1.30 1.37 0.00           -           -           -           -           -           -           

covtype lpm ~~g~~ bm 36.87 49.24 12.78 11.21 7.84 7.15 7.15 8.07 7.70 8.25 10.94 8.35 4.37 8.77 5.84
dt ~~g~~ bm 342.27 92.85 43.23 20.04 8.14 8.05 5.67 3.26 4.92 3.52 2.72 0.00 0.00 0.00 1.74


connect-4 lpm ~~g~~ bm 37.62 11.66 12.01 6.84 5.68 6.82 4.58 2.10 3.82 3.21 3.02 3.64 2.32 2.97 3.40
dt ~~g~~ bm 89.33 29.23 20.20 12.10 9.73 9.88 7.82 7.43 0.57 4.61 1.08 3.35 2.23 1.15 1.55


We highlight some interesting trends:


1. The incidence of negative improvements is fairly low. Of course, this result set is incomplete, but referring to the complete set in §A.8, we note that only 13 of 690 non-null
observations, or 1 _._ 88%, are negative. The average negative improvement is _−_ 0 _._ 24%.

2. As model size increases (left to right in Table 1), positive improvements (which can be high
for small sizes, e.g., _>_ 100%) tend to reduce. This makes intuitive sense since beyond a
certain model size, when all informative patterns in the data have been captured, modifying
the training distribution should not have much/any effect.

3. For DTs, the drop in improvements happen earlier than for LPMs. An intuitive explanation
for this is that an unit increase in size for the LPM and DT do not lead to identical increase
in capacity. DTs are non-linear models to begin with, and then, increasing their depth by
one leads to a much larger increment in capacity, e.g., it doubles the number of leaves for a
binary tree.


4.2 SUMMARY OF OTHER FINDINGS


In the interest of space, we summarize our other findings below:


1. Benchmarking against the density tree approach: we perform this comparison since it is
the closest to ours in terms of methodology (see §2). The experiment settings are identical
to the previous section, §4.1. While we present a detailed discussion in §A.9, the salient
observations are:

(a) We report the _scaled_ difference in test _F1-macro_ score improvement ( _δF_ 1 _[ora]_ _−_
_δF_ 1 _[den]_ ) _/_ max _{δF_ 1 _[den]_ _, δF_ 1 _[ora]_ _}_ . The denominator ensures a range of [ _−_ 1 _,_ 1],
where a positive value is desiredAveraged over model sizes and datasets this value
is 0 _._ 37 and 0 _._ 31 for LPMs and DTs respectively.


8


(b) An aggregate score like the above might be influenced by outlier improvement scores;
so we also report the _percentage of times_ we produce a better score. This is 81 _._ 38%
and 67 _._ 30% for LPMs and DTs respectively.

For additional details, please see §A.9.

2. Competitiveness: We compare against techniques specialized for certain tasks, to see if our
technique can elevate the performance of older techniques to be competitive:


(a) On the task of _cluster_ _explanation_, decision trees are constructed whose leaves represent clusters. Some specialized algorithms in the area are _ExShallow_ Laber et al.
(2021) and _Iterative_ _Mistake_ _Minimization_ _(IMM)_ Moshkovitz et al. (2020). While
these are recent algorithms, we show that CART-based Breiman et al. (1984) DTs
obtained by our technique _outperform_ the more recent IMM.
(b) We consider _prototype-based_ _classification_ where, in the interest of interpretability,
we want a small number of prototypes. Here the notion of size is the number of
prototypes. We show that using our technique improves the performance of a simple _Radial_ _Basis_ _Function_ _Network_ _(RBFN)_ Broomhead & Lowe (1988) to perform
similar to _Stochastic Neighbor Compression (SNC)_ Kusner et al. (2014).

The _Mean Rank_ is used as the primary metric, while the _Friedman_ (Friedman, 1937) and
_Wilcoxon_ _signed-rank_ tests are used to measure statistical significance. Please see §A.10
for details.

3. We also conducted these experiments to highlight some interesting properties:


(a) It may be applied even in cases when model size is defined by more than one attribute,
e.g., _max_ _depth_ and _num_ ~~_t_~~ _rees_ in the case of GBMs. This is because Algorithm 1
delegates size enforcement to _trainI,g_ . See §A.12 for details.
(b) The technique works even if the oracle and the target model use different feature
representations. This is because all that is required of the oracle are uncertainty scores,
irrespective of how it arrives at them. We demonstrate this via a text classification
task of predicting nationalities from surnames (Rao & McMahan, 2019). A _Gated_
_Recurrent_ _Unit_ _(GRU)_ (Cho et al., 2014) is used as the oracle. This is trained on a
sequence of characters. The interpretable target model is a DT that uses character
_n-grams_ as input. See §A.13 for additional details.


5 LIMITATION: RUNTIMES


The cost of catering to non-differentiable loss functions, i.e., no gradient information, is high running
times. Our experiments used _hyperopt_ on account of its popularity and maturity, but this leads to high
runtimes, e.g., for the _a1a_ “dt ~~g~~ bm” setting in Table 1, at our budget of _T_ = 3000, the optimizer runs
for close to an **hour** . But with a different surrogate model representation, e.g., _Gaussian Processes_,
and with a judiciously picked acquisition function such as a noise-resilient version of _logEI_ (Ament
et al., 2023), the runtime can be reduced to _∼_ **2** **min** . These preliminary results are presented in
§A.11. Our takeaway is that there exists a path to improving the runtime in future work.


6 CONCLUSION AND FUTURE WORK


In this work we presented a model-agnostic technique that obtains good size-accuracy trade-offs.
This was empirically shown to perform well in diverse settings. Conveniently, there is only one
hyperparameter to set (the number of iterations). Further, it can accommodate multivariate model
sizes and can be used with differing feature spaces between the oracle and the interpretable models.


For future work, we think the following themes are meaningful: (a) Extension to differentiable models/losses for faster learning. Techniques such as _bilevel optimization_, e.g., Pedregosa (2016), might
be useful here to learn instance weights directly, instead of a distribution. (b) We noted that improvements diminish with increasing model size (Section 4.1.2). It would be interesting to explore
whether applying the technique separately to smaller models obtained from decomposing a larger
model, e.g., subtrees within a DT, delays this effect. (c) Finally, exploring newer BayesOpt algorithms would be a good way to improve the running time for our algorithm - our current experiments
(mentioned in §5) already indicate this to be a fruitful direction of study.


9


ETHICS STATEMENT


[We acknowledge that we have read, understood and adhere to the code of ethics provided at https:](https://iclr.cc/public/CodeOfEthics)
[//iclr.cc/public/CodeOfEthics. We declare that this paper faithfully represents research](https://iclr.cc/public/CodeOfEthics)
that was performed with rigor and integrity, and the claims presented here are substantiated by our
experiments, which have been presented in detail in either the main paper or the Appendix.


We further declare that _Large Language Models (LLMs)_ were not used in this work.


10


REPRODUCIBILITY STATEMENT


We have uploaded our code as a supplementary material - this contains implementations of all ideas
presented in paper.


The datasets used are publicly available and the versions available at the source mentioned in Section
4.1.1 were directly used without any modifications..


11


REFERENCES


Mart´ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S.
Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew
Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath
Kudlur, Josh Levenberg, Dandelion Man´e, Rajat Monga, Sherry Moore, Derek Murray, Chris
Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vi´egas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL [https://www.tensorflow.org/.](https://www.tensorflow.org/) Software
available from tensorflow.org.


David J. Aldous. Exchangeability and related topics. In P. L. Hennequin (ed.), _Ecole_ _[´]_ _d’ Et´e_ _[´]_ _de_
_Probabilit´es_ _de_ _Saint-Flour_ _XIII_ _–_ _1983_, pp. 1–198, Berlin, Heidelberg, 1985. Springer Berlin
Heidelberg. ISBN 978-3-540-39316-0.


Fevzi Alimoglu and Ethem Alpaydin. Methods of combining multiple classifiers based on different
representations for pen-based handwritten digit recognition. In _Proceedings of the Fifth Turkish_
_Artificial Intelligence and Artificial Neural Networks Symposium (TAINN 96_, 1996.


Sebastian Ament, Sam Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Unexpected improvements to expected improvement for bayesian optimization. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_, 2023. [URL https://openreview.](https://openreview.net/forum?id=1vyAG6j9PE)
[net/forum?id=1vyAG6j9PE.](https://openreview.net/forum?id=1vyAG6j9PE)


Fabrizio Angiulli. Fast condensed nearest neighbor rule. In _Proceedings of the 22nd International_
_Conference on Machine Learning_, ICML ’05, pp. 25–32, New York, NY, USA, 2005. Association
for Computing Machinery. ISBN 1595931805. doi: 10.1145/1102351.1102355. [URL https:](https://doi.org/10.1145/1102351.1102355)
[//doi.org/10.1145/1102351.1102355.](https://doi.org/10.1145/1102351.1102355)


Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine Bias. [https://www.propublica.org/article/](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
[machine-bias-risk-assessments-in-criminal-sentencing, May 2016.](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)


Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham, Andrew Gordon Wilson, and Eytan Bakshy. BoTorch: A Framework for Efficient Monte-Carlo
Bayesian Optimization. In _Advances in Neural Information Processing Systems 33_, 2020. URL
[http://arxiv.org/abs/1910.06403.](http://arxiv.org/abs/1910.06403)


P. Baldi, P. Sadowski, and D. Whiteson. Searching for exotic particles in high-energy physics
with deep learning. _Nature_ _Communications_, 5(1):4308, 2014. ISSN 2041-1723. doi:
10.1038/ncomms5308. [URL https://doi.org/10.1038/ncomms5308.](https://doi.org/10.1038/ncomms5308)


Alessio Benavoli, Giorgio Corani, and Francesca Mangili. Should we really use post-hoc tests
based on mean-ranks? _Journal of Machine Learning Research_, 17(5):1–10, 2016. [URL http:](http://jmlr.org/papers/v17/benavoli16a.html)
[//jmlr.org/papers/v17/benavoli16a.html.](http://jmlr.org/papers/v17/benavoli16a.html)


J. Bergstra, D. Yamins, and D. D. Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In _Proceedings of the 30th Interna-_
_tional_ _Conference_ _on_ _International_ _Conference_ _on_ _Machine_ _Learning_ _-_ _Volume_ _28_, ICML’13,
pp. I–115–I–123. JMLR.org, 2013. URL [http://dl.acm.org/citation.cfm?id=](http://dl.acm.org/citation.cfm?id=3042817.3042832)
[3042817.3042832.](http://dl.acm.org/citation.cfm?id=3042817.3042832)


James Bergstra, R´emi Bardenet, Yoshua Bengio, and Bal´azs K´egl. Algorithms for hyper-parameter
optimization. In _Proceedings_ _of_ _the_ _24th_ _International_ _Conference_ _on_ _Neural_ _Information_ _Pro-_
_cessing_ _Systems_, NIPS’11, pp. 2546–2554, USA, 2011. Curran Associates Inc. ISBN 978-161839-599-3. [URL http://dl.acm.org/citation.cfm?id=2986459.2986743.](http://dl.acm.org/citation.cfm?id=2986459.2986743)


Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe LlinaresL´opez, Fabian Pedregosa, and Jean-Philippe Vert. Efficient and modular implicit differentiation.
_arXiv preprint arXiv:2105.15183_, 2021.


12


G. E. P. Box and D. R. Cox. An analysis of transformations. _Journal_ _of_ _the_ _Royal_ _Statistical_
_Society._ _Series_ _B_ _(Methodological)_, 26(2):211–252, 1964. ISSN 00359246. URL [http://](http://www.jstor.org/stable/2984418)
[www.jstor.org/stable/2984418.](http://www.jstor.org/stable/2984418)


Leo Breiman. Random forests. _Machine Learning_, 45(1):5–32, Oct 2001. ISSN 1573-0565. doi:
10.1023/A:1010933404324. [URL https://doi.org/10.1023/A:1010933404324.](https://doi.org/10.1023/A:1010933404324)


Leo Breiman et al. _Classification and Regression Trees_ . Chapman & Hall, New York, 1984. ISBN
0-412-04841-8.


Eric Brochu, Vlad M. Cora, and Nando de Freitas. A tutorial on bayesian optimization of expensive
cost functions, with application to active user modeling and hierarchical reinforcement learning.
_CoRR_, abs/1012.2599, 2010.


D.S. Broomhead and D. Lowe. Multivariable functional interpolation and adaptive networks. _Com-_
_plex Systems_, 2:321–355, 1988.


Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In _Pro-_
_ceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data_
_Mining_, KDD ’15, pp. 1721–1730, New York, NY, USA, 2015. ACM. ISBN 978-1-4503-36642. doi: 10.1145/2783258.2788613. URL [http://doi.acm.org/10.1145/2783258.](http://doi.acm.org/10.1145/2783258.2788613)
[2788613.](http://doi.acm.org/10.1145/2783258.2788613)


Chih-Chung Chang and Chih-Jen Lin. Ijcnn 2001 challenge: Generalization ability and text decoding. In _In Proceedings of IJCNN. IEEE_, pp. 1031–1036, 2001.


Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. _ACM_
_Transactions on Intelligent Systems and Technology_, 2:27:1–27:27, 2011a. Software available at
[http://www.csie.ntu.edu.tw/˜cjlin/libsvm.](http://www.csie.ntu.edu.tw/~cjlin/libsvm)


Chih-Chung Chang and Chih-Jen Lin. LIBSVM: A library for support vector machines. _ACM_
_Transactions_ _on_ _Intelligent_ _Systems_ _and_ _Technology_, 2:27:1–27:27, 2011b. Software avail[able at http://www.csie.ntu.edu.tw/˜cjlin/libsvm, datasets at https://www.](http://www.csie.ntu.edu.tw/~cjlin/libsvm)
[csie.ntu.edu.tw/˜cjlin/libsvmtools/datasets/.](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)


Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer. Smote: Synthetic minority over-sampling technique. _J._ _Artif._ _Int._ _Res._, 16(1):321–357, June 2002. ISSN
1076-9757.


Cheng Chen, Yong Wang, Lizi Liao, Yueguo Chen, and Xiaoyong Du. Real: A representative
error-driven approach for active learning. In Danai Koutra, Claudia Plant, Manuel Gomez Rodriguez, Elena Baralis, and Francesco Bonchi (eds.), _Machine Learning and Knowledge Discov-_
_ery in Databases:_ _Research Track_, pp. 20–37, Cham, 2023. Springer Nature Switzerland. ISBN
978-3-031-43412-9.


Kyunghyun Cho, Bart van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares,
Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder–
decoder for statistical machine translation. In _Proceedings_ _of_ _the_ _2014_ _Conference_ _on_ _Em-_
_pirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_ _(EMNLP)_, pp. 1724–1734, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1179. URL
[https://www.aclweb.org/anthology/D14-1179.](https://www.aclweb.org/anthology/D14-1179)


Ronan Collobert, Samy Bengio, and Yoshua Bengio. A parallel mixture of
svms for very large scale problems. In T. G. Dietterich, S. Becker, and
Z. Ghahramani (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _14_,
pp. 633–640. MIT Press, 2002. URL [http://papers.nips.cc/paper/](http://papers.nips.cc/paper/1949-a-parallel-mixture-of-svms-for-very-large-scale-problems.pdf)
[1949-a-parallel-mixture-of-svms-for-very-large-scale-problems.](http://papers.nips.cc/paper/1949-a-parallel-mixture-of-svms-for-very-large-scale-problems.pdf)
[pdf.](http://papers.nips.cc/paper/1949-a-parallel-mixture-of-svms-for-very-large-scale-problems.pdf)


Mark Craven and Jude Shavlik. Extracting tree-structured representations of trained networks. In
D. Touretzky, M.C. Mozer, and M. Hasselmo (eds.), _Advances_ _in_ _Neural_ _Information_ _Process-_
_ing_ _Systems_, volume 8. MIT Press, 1995. URL [https://proceedings.neurips.cc/](https://proceedings.neurips.cc/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf)
[paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf.](https://proceedings.neurips.cc/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf)


13


Denis J. Dean and Jock A. Blackard. Comparison of neural networks and discriminant analysis in
predicting forest cover types. 1998.


Janez Demˇsar. Statistical comparisons of classifiers over multiple data sets. _Journal_ _of_ _Machine_
_Learning Research_ [, 7(1):1–30, 2006. URL http://jmlr.org/papers/v7/demsar06a.](http://jmlr.org/papers/v7/demsar06a.html)
[html.](http://jmlr.org/papers/v7/demsar06a.html)


Don Kurian Dennis, Yash Gaurkar, Sridhar Gopinath, Sachin Goyal, Chirag Gupta, Moksh Jain,
Shikhar Jaiswal, Ashish Kumar, Aditya Kusupati, Chris Lovett, Shishir G Patil, Oindrila Saha,
and Harsha Vardhan Simhadri. EdgeML: Machine Learning for resource-constrained edge devices, 2021. [URL https://github.com/Microsoft/EdgeML.](https://github.com/Microsoft/EdgeML)


Persi Diaconis and Donald Ylvisaker. Quantifying prior opinion. Technical Report EFS ~~N~~ SF ~~2~~ 07,
1983. [URL https://purl.stanford.edu/ck231kf8763.](https://purl.stanford.edu/ck231kf8763)


Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. [URL http://archive.](http://archive.ics.uci.edu/ml)
[ics.uci.edu/ml.](http://archive.ics.uci.edu/ml)


Marco F. Duarte and Yu Hen Hu. Vehicle classification in distributed sensor networks. _J. Parallel_
_Distrib. Comput._, 64(7):826–838, July 2004. ISSN 0743-7315. doi: 10.1016/j.jpdc.2004.03.020.
[URL https://doi.org/10.1016/j.jpdc.2004.03.020.](https://doi.org/10.1016/j.jpdc.2004.03.020)


Jacob Feldman. Minimization of boolean complexity in human concept learning. _Nature_, 407:
630–3, 11 2000. doi: 10.1038/35036586.


Thomas S. Ferguson. A Bayesian Analysis of Some Nonparametric Problems. _The_ _Annals_ _of_
_Statistics_, 1(2):209  - 230, 1973. doi: 10.1214/aos/1176342360. URL [https://doi.org/](https://doi.org/10.1214/aos/1176342360)
[10.1214/aos/1176342360.](https://doi.org/10.1214/aos/1176342360)


Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. _The_ _Annals_
_of_ _Statistics_, 29(5):1189  - 1232, 2001. doi: 10.1214/aos/1013203451. URL [https://doi.](https://doi.org/10.1214/aos/1013203451)
[org/10.1214/aos/1013203451.](https://doi.org/10.1214/aos/1013203451)


Milton Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis
of variance. _Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 32(200):675–701, 1937. ISSN
01621459. [URL http://www.jstor.org/stable/2279372.](http://www.jstor.org/stable/2279372)


Nave Frost, Michal Moshkovitz, and Cyrus Rashtchian. Exkmc: Expanding explainable _k_ -means
clustering. _arXiv preprint arXiv:2006.02399_, 2020.


Roman Garnett. _Bayesian Optimization_ . Cambridge University Press, 2023.


Abhishek Ghose and Emma Thuong Nguyen. On the fragility of active learners for text classification. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), _Proceedings_ _of_ _the_
_2024 Conference on Empirical Methods in Natural Language Processing_, pp. 22217–22233, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/
v1/2024.emnlp-main.1240. URL [https://aclanthology.org/2024.emnlp-main.](https://aclanthology.org/2024.emnlp-main.1240/)
[1240/.](https://aclanthology.org/2024.emnlp-main.1240/)


Abhishek Ghose and Balaraman Ravindran. Interpretability with accurate small models. _Frontiers_
_in_ _Artificial_ _Intelligence_, 3:3, 2020. ISSN 2624-8212. doi: 10.3389/frai.2020.00003. URL
[https://www.frontiersin.org/article/10.3389/frai.2020.00003.](https://www.frontiersin.org/article/10.3389/frai.2020.00003)


Jack Henry Good, Torin Kovach, Kyle Miller, and Artur Dubrawski. Feature learning for interpretable, performant decision trees. In _Thirty-seventh_ _Conference_ _on_ _Neural_ _Information_ _Pro-_
_cessing Systems_, 2023. [URL https://openreview.net/forum?id=PYEgC56flW.](https://openreview.net/forum?id=PYEgC56flW)


Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation:
A survey. _International_ _Journal_ _of_ _Computer_ _Vision_, 129(6):1789–1819, Jun 2021. ISSN
1573-1405. doi: 10.1007/s11263-021-01453-z. URL [https://doi.org/10.1007/](https://doi.org/10.1007/s11263-021-01453-z)
[s11263-021-01453-z.](https://doi.org/10.1007/s11263-021-01453-z)


David Gunning. Explainable Artificial Intelligence. [https://www.darpa.mil/program/](https://www.darpa.mil/program/explainable-artificial-intelligence)
[explainable-artificial-intelligence, July 2016.](https://www.darpa.mil/program/explainable-artificial-intelligence)


14


Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural
networks. In _Proceedings_ _of_ _the_ _34th_ _International_ _Conference_ _on_ _Machine_ _Learning_ _-_ _Volume_
_70_, ICML’17, pp. 1321–1330. JMLR.org, 2017.


Chirag Gupta, Arun Sai Suggala, Ankit Goyal, Harsha Vardhan Simhadri, Bhargavi Paranjape,
Ashish Kumar, Saurabh Goyal, Raghavendra Udupa, Manik Varma, and Prateek Jain. ProtoNN:
Compressed and accurate kNN for resource-scarce devices. In Doina Precup and Yee Whye Teh
(eds.), _Proceedings_ _of_ _the_ _34th_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume 70 of
_Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 1331–1340. PMLR, 06–11 Aug 2017. URL
[https://proceedings.mlr.press/v70/gupta17a.html.](https://proceedings.mlr.press/v70/gupta17a.html)


Stuart W. Hall, Amin Sakzad, and Kim-Kwang Raymond Choo. Explainable artificial intelligence for digital forensics. _WIREs_ _Forensic_ _Science_, 4(2):e1434, 2022. doi: https://doi.org/10.
1002/wfs2.1434. [URL https://wires.onlinelibrary.wiley.com/doi/abs/10.](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wfs2.1434)
[1002/wfs2.1434.](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wfs2.1434)


Haibo He, Yang Bai, Edwardo A. Garcia, and Shutao Li. Adasyn: Adaptive synthetic sampling
approach for imbalanced learning. In _2008_ _IEEE_ _International_ _Joint_ _Conference_ _on_ _Neural_
_Networks_ _(IEEE_ _World_ _Congress_ _on_ _Computational_ _Intelligence)_, pp. 1322–1328, 2008. doi:
10.1109/IJCNN.2008.4633969.


Bernease Herman. The promise and peril of human evaluation for model interpretability. Presented
at NIPS 2017 Symposium on Interpretable Machine Learning. Available at: [https://arxiv.](https://arxiv.org/abs/1711.09889v3)
[org/abs/1711.09889v3, 2017.](https://arxiv.org/abs/1711.09889v3) [URL http://arxiv.org/abs/1711.07414.](http://arxiv.org/abs/1711.07414)


Elize Herrewijnen, Meagan B. Loerakker, Marloes Vredenborg, and Paweł W. Wo´zniak. Requirements and attitudes towards explainable ai in law enforcement. In _Proceedings of the 2024 ACM_
_Designing Interactive Systems Conference_, DIS ’24, pp. 995–1009, New York, NY, USA, 2024.
Association for Computing Machinery. ISBN 9798400705830. doi: 10.1145/3643834.3661629.
[URL https://doi.org/10.1145/3643834.3661629.](https://doi.org/10.1145/3643834.3661629)


Chih-Wei Hsu and Chih-Jen Lin. A comparison of methods for multiclass support vector machines.
_IEEE transactions on neural networks / a publication of the IEEE Neural Networks Council_, 13:
415–25, 02 2002. doi: 10.1109/72.991427.


Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by
reducing internal covariate shift. In _Proceedings_ _of_ _the_ _32nd_ _International_ _Conference_ _on_ _In-_
_ternational_ _Conference_ _on_ _Machine_ _Learning_ _-_ _Volume_ _37_, ICML’15, pp. 448–456. JMLR.org,
2015.


Nathalie Japkowicz and Mohak Shah. _Evaluating Learning Algorithms:_ _A Classification Perspec-_
_tive_ . Cambridge University Press, 2011. doi: 10.1017/CBO9780511921803.


Nathalie Japkowicz and Shaju Stephen. The class imbalance problem: A systematic study. _Intell._
_Data_ _Anal._, 6(5):429–449, October 2002. ISSN 1088-467X. URL [http://dl.acm.org/](http://dl.acm.org/citation.cfm?id=1293951.1293954)
[citation.cfm?id=1293951.1293954.](http://dl.acm.org/citation.cfm?id=1293951.1293954)


Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. Field-aware factorization machines for ctr prediction. In _Proceedings_ _of_ _the_ _10th_ _ACM_ _Conference_ _on_ _Recommender_ _Sys-_
_tems_, RecSys ’16, pp. 43–50, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450340359. doi: 10.1145/2959100.2959134. [URL https://doi.org/10.](https://doi.org/10.1145/2959100.2959134)
[1145/2959100.2959134.](https://doi.org/10.1145/2959100.2959134)


Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua
Bengio and Yann LeCun (eds.), _3rd International Conference on Learning Representations, ICLR_
_2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings_, 2015. [URL http:](http://arxiv.org/abs/1412.6980)
[//arxiv.org/abs/1412.6980.](http://arxiv.org/abs/1412.6980)


Todd Kulesza, Simone Stumpf, Margaret Burnett, Sherry Yang, Irwin Kwan, and Weng-Keen Wong.
Too much, too little, or just right? ways explanations impact end users’ mental models. In _2013_
_IEEE_ _Symposium_ _on_ _Visual_ _Languages_ _and_ _Human_ _Centric_ _Computing_, pp. 3–10, 2013. doi:
10.1109/VLHCC.2013.6645235.


15


Matt Kusner, Stephen Tyree, Kilian Weinberger, and Kunal Agrawal. Stochastic neighbor compression. In Eric P. Xing and Tony Jebara (eds.), _Proceedings_ _of_ _the_ _31st_ _International_ _Conference_
_on_ _Machine_ _Learning_, volume 32 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 622–630,
Bejing, China, 22–24 Jun 2014. PMLR. [URL https://proceedings.mlr.press/v32/](https://proceedings.mlr.press/v32/kusner14.html)
[kusner14.html.](https://proceedings.mlr.press/v32/kusner14.html)


Eduardo Sany Laber, Lucas Murtinho, and Felipe Oliveira. Shallow decision trees for explainable
k-means clustering. _CoRR_ [, abs/2112.14718, 2021. URL https://arxiv.org/abs/2112.](https://arxiv.org/abs/2112.14718)
[14718.](https://arxiv.org/abs/2112.14718)


Isaac Lage, Emily Chen, Jeffrey He, Menaka Narayanan, Been Kim, Samuel J. Gershman, and
Finale Doshi-Velez. Human evaluation of models built for interpretability. _Proceedings_ _of_ _the_
_AAAI_ _Conference_ _on_ _Human_ _Computation_ _and_ _Crowdsourcing_, 7(1):59–67, Oct. 2019. URL
[https://ojs.aaai.org/index.php/HCOMP/article/view/5280.](https://ojs.aaai.org/index.php/HCOMP/article/view/5280)


Himabindu Lakkaraju, Stephen H. Bach, and Jure Leskovec. Interpretable decision sets: A joint
framework for description and prediction. In _Proceedings_ _of_ _the_ _22Nd_ _ACM_ _SIGKDD_ _Interna-_
_tional_ _Conference_ _on_ _Knowledge_ _Discovery_ _and_ _Data_ _Mining_, KDD ’16, pp. 1675–1684, New
York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/2939672.2939874. URL
[http://doi.acm.org/10.1145/2939672.2939874.](http://doi.acm.org/10.1145/2939672.2939874)


David D. Lewis and Jason Catlett. Heterogeneous uncertainty sampling for supervised learning.
In _In Proceedings of the Eleventh International Conference on Machine Learning_, pp. 148–156.
Morgan Kaufmann, 1994.


Oscar Li, Hao Liu, Chaofan Chen, and Cynthia Rudin. Deep learning for case-based reasoning
through prototypes: A neural network that explains its predictions. In _Proceedings of the Thirty-_
_Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Arti-_
_ficial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial_
_Intelligence_, AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018. ISBN 978-1-57735-800-8.


Zachary C. Lipton. The mythos of model interpretability. _Queue_, 16(3):30:31–30:57, June 2018.
[ISSN 1542-7730. doi: 10.1145/3236386.3241340. URL http://doi.acm.org/10.1145/](http://doi.acm.org/10.1145/3236386.3241340)
[3236386.3241340.](http://doi.acm.org/10.1145/3236386.3241340)


Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _30_, pp.
4765–4774. Curran Associates, Inc., 2017. URL [http://papers.nips.cc/paper/](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)
[7062-a-unified-approach-to-interpreting-model-predictions.pdf.](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)


Konstantin Makarychev and Liren Shan. Explainable k-means: Don’t be greedy, plant bigger
trees! In _Proceedings_ _of_ _the_ _54th_ _Annual_ _ACM_ _SIGACT_ _Symposium_ _on_ _Theory_ _of_ _Computing_,
STOC 2022, pp. 1629–1642, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392648. doi: 10.1145/3519935.3520056. [URL https://doi.org/10.](https://doi.org/10.1145/3519935.3520056)
[1145/3519935.3520056.](https://doi.org/10.1145/3519935.3520056)


Katerina Margatina, Giorgos Vernikos, Lo¨ıc Barrault, and Nikolaos Aletras. Active learning by
acquiring contrastive examples. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and
Scott Wen-tau Yih (eds.), _Proceedings of the 2021 Conference on Empirical Methods in Natural_
_Language_ _Processing_, pp. 650–663, Online and Punta Cana, Dominican Republic, November
2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.51. URL
[https://aclanthology.org/2021.emnlp-main.51/.](https://aclanthology.org/2021.emnlp-main.51/)


Donald Michie, D. J. Spiegelhalter, C. C. Taylor, and John Campbell (eds.). _Machine_ _Learning,_
_Neural and Statistical Classification_ . Ellis Horwood, USA, 1995. ISBN 013106360X.


Ibomoiye Domor Mienye, George Obaido, Nobert Jere, Ebikella Mienye, Kehinde Aruleba,
Ikiomoye Douglas Emmanuel, and Blessing Ogbuokiri. A survey of explainable artificial intelligence in healthcare: Concepts, applications, and challenges. _Informatics_ _in_
_Medicine_ _Unlocked_, 51:101587, 2024. ISSN 2352-9148. doi: https://doi.org/10.1016/j.imu.
2024.101587. URL [https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/S2352914824001448)
[S2352914824001448.](https://www.sciencedirect.com/science/article/pii/S2352914824001448)


16


R. M. Mohammad, F. Thabtah, and L. McCluskey. An assessment of features related to phishing
websites using an automated technique. In _2012 International Conference for Internet Technology_
_and Secured Transactions_, pp. 492–497, Dec 2012.


Carina Mood. Logistic regression : Why we cannot do what we think we can do, and what we can
do about it. _European Sociological Review_, 26(1):67–82, 2010. doi: 10.1093/esr/jcp006.


Michal Moshkovitz, Sanjoy Dasgupta, Cyrus Rashtchian, and Nave Frost. Explainable k-means
and k-medians clustering. In Hal Daum´e III and Aarti Singh (eds.), _Proceedings_ _of_ _the_ _37th_
_International Conference on Machine Learning_, volume 119 of _Proceedings of Machine Learning_
_Research_, pp. 7055–7065. PMLR, 13–18 Jul 2020. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v119/moshkovitz20a.html)
[press/v119/moshkovitz20a.html.](https://proceedings.mlr.press/v119/moshkovitz20a.html)


Nour Moustafa, Nickolaos Koroniotis, Marwa Keshk, Albert Y. Zomaya, and Zahir Tari. Explainable intrusion detection for cyber defences in the internet of things: Opportunities and solutions.
_IEEE Communications Surveys & Tutorials_, 25(3):1775–1807, 2023. doi: 10.1109/COMST.2023.
3280465.


W. James Murdoch, Chandan Singh, Karl Kumbier, Reza Abbasi-Asl, and Bin Yu. Definitions, methods, and applications in interpretable machine learning. _Proceedings_ _of_ _the_ _Na-_
_tional Academy of Sciences_, 116(44):22071–22080, 2019. ISSN 0027-8424. doi: 10.1073/pnas.
1900654116. [URL https://www.pnas.org/content/116/44/22071.](https://www.pnas.org/content/116/44/22071)


Meike Nauta, Annemarie Jutte, Jesper Provoost, and Christin Seifert. This looks like that, because
... explaining prototypes for interpretable image recognition. In Michael Kamp, Irena Koprinska,
Adrien Bibal, Tassadit Bouadi, Benoˆıt Fr´enay, Luis Gal´arraga, Jos´e Oramas, Linara Adilova,
Yamuna Krishnamurthy, Bo Kang, Christine Largeron, Jefrey Lijffijt, Tiphaine Viard, Pascal
Welke, Massimiliano Ruocco, Erlend Aune, Claudio Gallicchio, Gregor Schiele, Franz Pernkopf,
Michaela Blott, Holger Fr¨oning, G¨unther Schindler, Riccardo Guidotti, Anna Monreale, Salvatore
Rinzivillo, Przemyslaw Biecek, Eirini Ntoutsi, Mykola Pechenizkiy, Bodo Rosenhahn, Christopher Buckley, Daniela Cialfi, Pablo Lanillos, Maxwell Ramstead, Tim Verbelen, Pedro M. Ferreira, Giuseppina Andresini, Donato Malerba, Ib´eria Medeiros, Philippe Fournier-Viger, M. Saqib
Nawaz, Sebastian Ventura, Meng Sun, Min Zhou, Valerio Bitetta, Ilaria Bordino, Andrea Ferretti,
Francesco Gullo, Giovanni Ponti, Lorenzo Severini, Rita Ribeiro, Jo˜ao Gama, Ricard Gavald`a,
Lee Cooper, Naghmeh Ghazaleh, Jonas Richiardi, Damian Roqueiro, Diego Saldana Miranda,
Konstantinos Sechidis, and Guilherme Grac¸a (eds.), _Machine Learning and Principles and Prac-_
_tice_ _of_ _Knowledge_ _Discovery_ _in_ _Databases_, pp. 441–456, Cham, 2021. Springer International
Publishing. ISBN 978-3-030-93736-2.


Alexandru Niculescu-Mizil and Rich Caruana. Obtaining calibrated probabilities from boosting.
In _Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence_, UAI’05,
pp. 413–420, Arlington, Virginia, United States, 2005. AUAI Press. ISBN 0-9749039-1-4. URL
[http://dl.acm.org/citation.cfm?id=3020336.3020388.](http://dl.acm.org/citation.cfm?id=3020336.3020388)


D. I. Ohlssen, L. D. Sharples, and D. J. Spiegelhalter. Flexible random-effects models using bayesian
semi-parametric models: applications to institutional comparisons. _Statistics in Medicine_, 26(9):
2088–2112, 2007. doi: 10.1002/sim.2666. [URL https://onlinelibrary.wiley.com/](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2666)
[doi/abs/10.1002/sim.2666.](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2666)


Fabian Paschke, Christian Bayer, Martyna Bator, Uwe M¨onks, Alexander Dicks, Olaf EngeRosenblatt, and Volker Lohweg. Sensorlose zustands¨uberwachung an synchronmotoren. In _Pro-_
_ceedings of Computational Intelligence Workshop_, 12 2013.


F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and
E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_,
12:2825–2830, 2011.


Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In _Proceedings of the_
_33rd_ _International_ _Conference_ _on_ _International_ _Conference_ _on_ _Machine_ _Learning_ _-_ _Volume_ _48_,
ICML’16, pp. 737–746. JMLR.org, 2016.


17


Rok Piltaver, Mitja Luˇstrek, Matjaˇz Gams, and Sanda Martinˇci´c-Ipˇsi´c. What makes classification
trees comprehensible? _Expert Systems with Applications_, 62:333–346, 2016. ISSN 0957-4174.
doi: https://doi.org/10.1016/j.eswa.2016.06.009. URL [https://www.sciencedirect.](https://www.sciencedirect.com/science/article/pii/S0957417416302901)
[com/science/article/pii/S0957417416302901.](https://www.sciencedirect.com/science/article/pii/S0957417416302901)


John Platt. Fast training of support vector machines using sequential minimal optimization. In _Advances_ _in_ _Kernel_ _Methods_ _-_ _Support_ _Vector_ _Learning_ . MIT Press, January
1998. URL [https://www.microsoft.com/en-us/research/publication/](https://www.microsoft.com/en-us/research/publication/fast-training-of-support-vector-machines-using-sequential-minimal-optimization/)
[fast-training-of-support-vector-machines-using-sequential-minimal-optimization/.](https://www.microsoft.com/en-us/research/publication/fast-training-of-support-vector-machines-using-sequential-minimal-optimization/)


John C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized
likelihood methods. In _ADVANCES IN LARGE MARGIN CLASSIFIERS_, pp. 61–74. MIT Press,
1999.


Forough Poursabzi-Sangdeh, Dan Goldstein, Jake Hofman, Jennifer Wortman Vaughan, and
Hanna Wallach. Manipulating and measuring model interpretability. In _CHI_ _2021_,
May 2021. [URL https://www.microsoft.com/en-us/research/publication/](https://www.microsoft.com/en-us/research/publication/manipulating-and-measuring-model-interpretability/)
[manipulating-and-measuring-model-interpretability/.](https://www.microsoft.com/en-us/research/publication/manipulating-and-measuring-model-interpretability/)


Danil Prokhorov. IJCNN 2001 Neural Network Competition. [http://www.geocities.ws/](http://www.geocities.ws/ijcnn/nnc_ijcnn01.pdf)
[ijcnn/nnc_ijcnn01.pdf, 2001.](http://www.geocities.ws/ijcnn/nnc_ijcnn01.pdf)


Delip Rao and Brian McMahan. _Natural_ _Language_ _Processing_ _with_ _PyTorch_ .
O’Reilly, 2019. ISBN 978-1491978238. [https://www.amazon.com/](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236/)
[Natural-Language-Processing-PyTorch-Applications/dp/1491978236/](https://www.amazon.com/Natural-Language-Processing-PyTorch-Applications/dp/1491978236/)
[and https://github.com/joosthub/PyTorchNLPBook.](https://github.com/joosthub/PyTorchNLPBook)


Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “why should i trust you?”: Explaining the
predictions of any classifier. In _Proceedings of the 22Nd ACM SIGKDD International Conference_
_on_ _Knowledge_ _Discovery_ _and_ _Data_ _Mining_, KDD ’16, pp. 1135–1144, New York, NY, USA,
2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/2939672.2939778. [URL http://doi.](http://doi.acm.org/10.1145/2939672.2939778)
[acm.org/10.1145/2939672.2939778.](http://doi.acm.org/10.1145/2939672.2939778)


Tim R¨az. Ml interpretability: Simple isn’t easy. _Studies_ _in_ _History_ _and_ _Philosophy_
_of_ _Science_, 103:159–167, 2024. ISSN 0039-3681. doi: https://doi.org/10.1016/j.shpsa.
2023.12.007. URL [https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/S0039368123001723)
[S0039368123001723.](https://www.sciencedirect.com/science/article/pii/S0039368123001723)


Sudarsun Santhiappan, Jeshuren Chelladurai, and Balaraman Ravindran. A novel topic modeling based weighting framework for class imbalance learning. In _Proceedings_ _of_ _the_ _ACM_ _In-_
_dia_ _Joint_ _International_ _Conference_ _on_ _Data_ _Science_ _and_ _Management_ _of_ _Data_, CoDS-COMAD
’18, pp. 20–29, New York, NY, USA, 2018. Association for Computing Machinery. ISBN
9781450363419. doi: 10.1145/3152494.3152496. URL [https://doi.org/10.1145/](https://doi.org/10.1145/3152494.3152496)
[3152494.3152496.](https://doi.org/10.1145/3152494.3152496)


Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How does batch
normalization help optimization? In S. Bengio, H. Wallach, H. Larochelle, K. Grauman,
N. Cesa-Bianchi, and R. Garnett (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_,
volume 31. Curran Associates, Inc., 2018. URL [https://proceedings.neurips.cc/](https://proceedings.neurips.cc/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf)
[paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf.](https://proceedings.neurips.cc/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf)


Tobias Scheffer, Christian Decomain, and Stefan Wrobel. Active hidden markov models for information extraction. In _Proceedings of the 4th International Conference on Advances in Intelligent_
_Data_ _Analysis_, IDA ’01, pp. 309–318, London, UK, UK, 2001. Springer-Verlag. ISBN 3-54042581-0. [URL http://dl.acm.org/citation.cfm?id=647967.741626.](http://dl.acm.org/citation.cfm?id=647967.741626)


Burr Settles. Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009. URL [http://axon.cs.byu.edu/˜martinez/](http://axon.cs.byu.edu/~martinez/classes/778/Papers/settles.activelearning.pdf)
[classes/778/Papers/settles.activelearning.pdf.](http://axon.cs.byu.edu/~martinez/classes/778/Papers/settles.activelearning.pdf)


B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas. Taking the human out of the
loop: A review of bayesian optimization. _Proceedings_ _of_ _the_ _IEEE_, 104(1):148–175, Jan 2016.
ISSN 0018-9219. doi: 10.1109/JPROC.2015.2494218.


18


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


Jiaming Song, Lantao Yu, Willie Neiswanger, and Stefano Ermon. A general recipe for likelihoodfree Bayesian optimization. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari,
Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine_
_Learning_, volume 162 of _Proceedings of Machine Learning Research_, pp. 20384–20404. PMLR,
17–23 Jul 2022. [URL https://proceedings.mlr.press/v162/song22b.html.](https://proceedings.mlr.press/v162/song22b.html)


Herbert A. Sturges. The choice of a class interval. _Journal of the American Statistical Association_,
21(153):65–66, 1926. doi: 10.1080/01621459.1926.10502161. URL [https://doi.org/](https://doi.org/10.1080/01621459.1926.10502161)
[10.1080/01621459.1926.10502161.](https://doi.org/10.1080/01621459.1926.10502161)


Robert Tibshirani. Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical_
_Society. Series B (Methodological)_ [, 58(1):267–288, 1996. ISSN 00359246. URL http://www.](http://www.jstor.org/stable/2346178)
[jstor.org/stable/2346178.](http://www.jstor.org/stable/2346178)


Andrew V. Uzilov, Joshua M. Keegan, and David H. Mathews. Detection of non-coding rnas on
the basis of predicted secondary structure formation free energy change. _BMC bioinformatics_, 7:
173–173, Mar 2006. ISSN 1471-2105. doi: 10.1186/1471-2105-7-173. [URL https://www.](https://www.ncbi.nlm.nih.gov/pubmed/16566836)
[ncbi.nlm.nih.gov/pubmed/16566836.](https://www.ncbi.nlm.nih.gov/pubmed/16566836) 16566836[pmid].


Chien-Chih Wang, Kent Loong Tan, Chun-Ting Chen, Yu-Hsiang Lin, S. Sathiya Keerthi, Dhruv
Mahajan, S. Sundararajan, and Chih-Jen Lin. Distributed newton methods for deep neural
networks. _Neural_ _Comput._, 30(6):1673–1724, June 2018. ISSN 0899-7667. doi: 10.1162/
neco ~~a~~ ~~0~~ 1088. [URL https://doi.org/10.1162/neco_a_01088.](https://doi.org/10.1162/neco_a_01088)


Chong Wang, John Paisley, and David M. Blei. Online variational inference for the hierarchical
dirichlet process. In Geoffrey Gordon, David Dunson, and Miroslav Dud´ık (eds.), _Proceedings_
_of the Fourteenth International Conference on Artificial Intelligence and Statistics_, volume 15 of
_Proceedings of Machine Learning Research_, pp. 752–760, Fort Lauderdale, FL, USA, 11–13 Apr
2011. PMLR. [URL https://proceedings.mlr.press/v15/wang11a.html.](https://proceedings.mlr.press/v15/wang11a.html)


Frank Wilcoxon. Individual comparisons by ranking methods. _Biometrics_ _Bulletin_, 1(6):80–83,
1945. ISSN 00994987. [URL http://www.jstor.org/stable/3001968.](http://www.jstor.org/stable/3001968)


Matthew J Beal Yee Whye Teh, Michael I Jordan and David M Blei. Hierarchical dirichlet processes.
_Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 101(476):1566–1581, 2006. doi: 10.1198/
016214506000000302. [URL https://doi.org/10.1198/016214506000000302.](https://doi.org/10.1198/016214506000000302)


In-Kwon Yeo and Richard A. Johnson. A new family of power transformations to improve normality
or symmetry. _Biometrika_ [, 87(4):954–959, 2000. ISSN 00063444. URL http://www.jstor.](http://www.jstor.org/stable/2673623)
[org/stable/2673623.](http://www.jstor.org/stable/2673623)


Wei Zhang, Xiaohui Chen, Yueqi Liu, and Qian Xi. A distributed storage and computation k-nearest
neighbor algorithm based cloud-edge computing for cyber-physical-social systems. _IEEE Access_,
8:50118–50130, 2020. doi: 10.1109/ACCESS.2020.2974764.


19


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


A APPENDIX


A.1 1. UNCERTAINTY METRICS


Some other popular uncertainty metrics are:


1. **Least confident** : we calculate the extent of uncertainty w.r.t. the class we are most confident about:
_uM_ ( _x_ ) = 1 _−_ max (4)
_yi∈{_ 1 _,_ 2 _,...,C}_ _[M]_ [(] _[y][i][|][x]_ [)]


Here, we have _C_ classes, and _M_ ( _yi|x_ ) is the probability score produced by the model [7] .


2. **Entropy** : this is the standard Shannon entropy measure calculated over class prediction
confidences:
_uM_ ( _x_ ) =          - _−M_ ( _yi|x_ ) log _M_ ( _yi|x_ ) (5)


_yi∈{_ 1 _,_ 2 _,...,C}_


We do not use the _least confident_ metric since it completely ignores confidence distribution across
labels. While _entropy_ is quite popular, and does take into account the confidence distribution, we
do not use it since it reaches its maximum for only points for which the classifier must be equally
ambiguous about _all_ labels; for datasets with many labels (one of our experiments uses a dataset
with 26 labels - see Table 3) we may never reach this maximum.


Fig 3 visually shows what uncertainty values look like for the different metrics. Panel (a) displays
a dataset with 4 labels. A probabilistic _linear_ _Support_ _Vector_ _Machine_ _(SVM)_ is learned on this,
and uncertainty scores corresponding to the metrics “margin”, “least confident” and “entropy” are
visualized in panels (b), (c) and (d) respectively. Darker shades of gray correspond to high uncertainty. Observe that only the “margin” metric in panel (b) achieves scores close to 1 at the two-label
boundaries.


There is no best uncertainty metric in general, and the choice is usually application specific (Settles,
2009).


A.2 SAMPLING FROM THE IBMM


Given our representation, the procedure to sample _Ns_ points, from a dataset _D_, using an oracle _MO_
is shown in Algorithm 2. We also explain the steps below:


1. Determine partitioning over the _Ns_ points induced by the _DP_ . We use the _Chinese_
_Restaurant_ _Process_ Aldous (1985) for this. Let’s assume this step produces _k_ partitions
_{c_ 1 _, c_ 2 _, ..., ck}_ and quantities _ni_ _∈_ N where [�] _i_ _[k]_ =1 _[n][i]_ [=] _[N]_ [.] [Here,] _[ n][i]_ [denotes the number]
of points that belong to partition _ci_ .


2. We determine the _Beta_ ( _Ai, Bi_ ) component for each _ci_ by sampling from the priors, i.e.,
_Ai_ _∼_ _scale × Beta_ ( _a, b_ ) and _Bi_ _∼_ _scale × Beta_ ( _a_ _[′]_ _, b_ _[′]_ ).


3. Repeat for each _ci_ : for each instance-label pair ( _xj, yj_ ) in our training dataset, we calculate
the oracle uncertainty score, _uMO_ ( _xj_ ). We then calculate _pj_ = _c · Beta_ ( _uMO_ ( _xj_ ) _|Ai, Bi_ ).
_c_ is a normalizing constant that scales the probabilities across instances to sum to 1. The
quantities _pj_ are used as sampling probabilities for various ( _xj, yj_ ), and _ni_ points are sampled with replacement based on them.


A.3 DEFAULT PARAMETERS


The optimizer we use, TPE, requires _box_ _constraints_ . Here we specify our search space for the
optimization variables, Φ in Algorithm 1:


7The possibly confusing name “least confident” for this idea originated within the context of
uncertainty sampling, where we are interested in sampling the most uncertain point, _x_ _[∗]_ =
arg min _x_ [max _yi∈{_ 1 _,_ 2 _,...,C} M_ ( _yi|x_ )], which may be considered to be the instance with the “least most confident label”.


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


Figure 3: Visualizations of different uncertainty metrics. (a) shows a 4-label dataset on which linear
SVM is learned. (b), (c), (d) visualize uncertainty scores based on different metrics, as per the linear
SVM, where darker shades imply higher scores.


**Algorithm 2:** Sample based on uncertainties and Ψ

**Data:** Sample size _Ns_, oracle _MO_, dataset _D_ = _{_ ( _xi, yi_ ) _}_ _[N]_ _i_ =1 [, IBMM parameters]
Ψ = _{α, a, b, a_ _[′]_ _, b_ _[′]_ _}_
**Result:** Sample _D_ _[′]_, where _|D_ _[′]_ _|_ = _Ns_

**1** _D_ _[′]_ = _{}_ // assumed to be a multiset

**2** _{_ ( _c_ 1 _, n_ 1) _,_ ( _c_ 2 _, n_ 2) _, ...,_ ( _ck, nk_ ) _} ←_ partition _Ns_ using the _DP_ // Here [�] _i_ _[k]_ =1 _[n][i]_ [=] _[ N][s]_ [.]

**3** **for** _i ←_ 1 **to** _k_ **do**

**4** _Ai_ _∼_ _scale × Beta_ ( _a, b_ )

**5** _Bi_ _∼_ _scale × Beta_ ( _a_ _[′]_ _, b_ _[′]_ )

**6** **for** _j_ _←_ 1 **to** _N_ **do**

**7** _pj_ _←_ _c · Beta_ ( _uMO_ ( _xj_ ); _Ai, Bi_ ) // _c_ is a normalizing constant s.t.

   - _Ni_ _[c][ ·][ p][j]_ [= 1][.]

**8** **end**

**9** _temp ←_ sample with replacement _ni_ instance-label pairs based on _pj_

**10** _D_ _[′]_ _←_ _D_ _[′]_ _⊎_ _temp_ // _⊎_ is the multiset sum

**11** **end**

**12** **return** _D_ _[′]_


1. _po_ : We want to allow the algorithm to pick an arbitrary fraction of samples from the original
data; we set _po_ _∈_ [0 _,_ 1].

2. _Ns_ : We set _Ns_ _∈_ [400 _,_ 10000]. The lower bound ensures we have statistically significant
results. The upper bound is set to a reasonably large value.

3. _{a, b, a_ _[′]_ _, b_ _[′]_ _}_ : Each of these parameters are allowed a range [0 _._ 1 _,_ 10] to allow for a wide
range of shapes for the component _Beta_ distributions.


4. _scale_ : We fix _scale_ = 10000 for our experiments, to allow for _Ai_ and _Bi_ to model skewed
distributions where shape parameter large values might be required. For small values, the
algorithm adapts by learning the appropriate _{a, b, a_ _[′]_ _, b_ _[′]_ _}_ .


5. _α_ : For a DP, _α ∈_ R _>_ 0. We use a lower bound of 0 _._ 1.


21


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


**Algorithm 3:** Learning interpretable model using oracle - reproduction of Algorithm 1.

**Data:** Dataset _D_, model size _η_, _trainO,h_ (), _trainI,g_ (), iterations _T_
**Result:** Optimal parameters Φ _[∗]_, test set accuracy _stest_ at Φ _[∗]_, and interpretable model _M_ _[∗]_ at Φ _[∗]_

**1** Create splits _Dtrain, Dval, Dtest_ from _D_, stratified wrt labels. Here
_|Dtrain|_ : _|Dval|_ : _|Dtest|_ :: 60 : 20 : 20.

**2** _MO_ _←_ _trainO,h_ ( _Dtrain, ∗_ )

**3** **for** _t ←_ 1 **to** _T_ **do**

**4** Φ _t_ _←_ _suggest_ ( _s_ 0 _, s_ 1 _, ...st−_ 1 _,_ Φ0 _,_ Φ1 _, ...,_ Φ _t−_ 1) // _s_ 0 _,_ Φ0 initialized at _t_ = 0,
see text. Note: Φ _t_ = _{_ Ψ _t, Ns,t, po,t}_ where Ψ _t_ = _{αt, at, bt, a_ _[′]_ _t_ _[, b]_ _t_ _[′]_ _[}]_ [.]

**5** _No_ _←_ _po,t × Ns,t_

**6** _Nu_ _←_ _Ns_ _t −_ _No_

**7** _Do_ _←_ uniformly sample with replacement _No_ points from _Dtrain_

**8** _Du_ _←_ sample _Nu_ points from _Dtrain_ using the DP-based IBMM given current values for
_Nu, MO, Dtrain,_ Ψ _t_ // see Algorithm A.2 for details

**9** _Ds_ _←_ _Do ⊎_ _Du_ // _Do_, _Du_ are assumed to be multisets

**10** _Mt_ _←_ _trainI,g_ ( _Ds, η_ )

**11** _st_ _←_ _acc_ ( _Mt, Dval_ )

**12** **end**


To determine the upper bound, we rely on the following empirical relationship (Ohlssen
et al., 2007) between the number of components _k_ and _α_ :


_E_ [ _k|α_ ] _≈_ 5 _α_ + 2 (6)


We empirically estimated a fairly inclusive upper bound on the number of components to
be 500, which provides us the _α_ upper bound of 99 _._ 6. Thus, we use _α ∈_ [0 _._ 1 _,_ 99 _._ 6].


A.4 NOTES ON THE MAIN ALGORITHM


We provide some additional details in reference to the main algorithm - Algorithm 1 - in the paper.
For convenience, we reproduce the algorithm here, as Algorithm 3. Our notes follow:


1. We will consider the initialization to happen at _t_ = 0, while the iterations range from 1 to
_T_ . Φ0 is set to: _α_ = 0 _._ 1 _, a_ = 1 _, b_ = 1 _, a_ _[′]_ = 1 _, b_ _[′]_ = 1 _, Ns_ = _|Dtrain|, po_ = 1. A model
is constructed based on Φ0 and a score _s_ 0 is recorded. (Φ0 _, s_ 0) serve as the history for the
iteration at _t_ = 1. The values for _α, a, b, a_ _[′]_ _, b_ _[′]_ carry no significance and are arbitrary, since
setting _po_ _→_ 1 forces sampling only from the original distribution. Combined with _Ns_ =
_|Dtrain|_, this setting mimics the baseline, i.e., training the interpretable model without our
algorithm, thus providing the optimizer with a good initial reference point in its search
space.

2. The optimizer is represented by the function call _suggest_ () which takes as input all past
parameter values and validation scores. _suggest_ () denotes a generic optimizer; not all
optimizers require this extent of historical information.

3. While the training algorithm for the oracle, _trainO,h_ () is taken as input, a pre-constructed
oracle _MO_ may also be used. This would eliminate the oracle training step in line 2.

4. _acc_ () on the validation data, _Dval_, serves as both the objective and fitness function.
5. Evaluation on the test set, _Dtest_ is done only once, in line 16, with the model that produces
the best validation score.

6. Since we sample with replacement, both temporary datasets _Do_ and _Du_, procured from
uniformly sampling the original training data and sampling based on uncertainties respectively, are multisets. Accordingly, line 9 uses the multiset sum operator _⊎_ to combine them.


22


**13** _t_ _[∗]_ _←_ arg max _t {s_ 1 _, s_ 2 _, ..., sT −_ 1 _, sT }_


**14** Φ _[∗]_ _←_ Φ _t∗_


**15** _M_ _[∗]_ _←_ _Mt∗_


**16** _stest_ _←_ _acc_ ( _M_ _[∗]_ _, Dtest_ )


**17** **return** Φ _[∗]_, _stest_, _M_ _[∗]_


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


7. _Mt_ is created (line 10) with limited or no hyperparameter search using simple random
validation, i.e., a stratified (by labels) random sample of size 0 _._ 2 _Ns,t_ is used as the validation set. A restricted search is performed because often hyperparameters are correlated
with model size, and setting them to particular values would fail to produce a model of
the required size _η_ . As an example, consider DTs: setting a high threshold for the number
of instances in a node for it be split (hyperparameter _min_ ~~_s_~~ _amples_ ~~_s_~~ _plit_ in _scikit-learn’s_
(Pedregosa et al., 2011) implementation) would produce only short trees.
We don’t use cross-validation since at small values of _Ns,t_, the amount of training data, i.e.,
( _[k][−]_ _k_ [1] [)] _[N][s,t]_ [for] _[k]_ [-folds,] [may] [become] [too] [small] [to] [obtain] [a] [good] [model.] [For] [example,] [for]

3-folds, the training data size is 0 _._ 67 _Ns,t_ . The data shortage problem can be addressed by
increasing the number of folds, but that also increases the running time per iteration owing
to the larger number of models that now need to be trained. As a practical compromise, we
perform simple validation _thrice_ and average the outcomes. This number is configurable,
and may be decreased for models that are expensive to train.

8. Since the validation score _st_ (line 11) needs to be reliable, in our implementation we repeat
lines 7-10 _thrice_ and use the averaged validation score as _st_ .
9. Class imbalance is accounted for in our implementation when training model _Mt_ in line 10.
We either balance the data by sampling (this is the case with a _Linear Probability Model_ ),
or an appropriate cost function is used to simulate balanced classes (this is the case with
DTs and GBMs).


It is important to note here that _Dval_ _and Dtest_ _are not modified by our algorithm in any way_, and
therefore _st_ and _stest_ measure the accuracy on the original distribution.


A.5 SMOOTHING THE OPTIMIZATION LANDSCAPE


A practical consideration in our implementation is if we might facilitate finding the maxima Φ _[∗]_ in
Algorithm 1?


Since BayesOpt algorithms model the response surface of the actual objective function using a finite
number of evaluations ( _st_ in Algorithm 1), a certain degree of _smoothness_ is assumed (Shahriari
et al., 2016; Brochu et al., 2010). Here, the optimization variables Φ influence the objective value
_s_ via this indirect chain: Φ _t_ _→_ _Ds_ _→_ _Mt_ _→_ _st_ (symbols as in Algorithm 1), and for BayesOpt to
work well, it is required that small changes in Φ _t_ result in small changes in _st_ .


However, we have noticed that an oracle might produce uncertainty score distributions that are
“spiky” or “jagged” - as an example, see the curve labelled “original” in Figure 4(a); which leads
us to hypothesize that this principle is violated in general. A spiky distribution implies that small
shifts Φ _t_ + ∆Φ _t_ may lead to sampling of instances with very different uncertainties; and since such
instances may occur in regions far from those indicated by Φ _t_, they produce models with different
class prediction behavior. This indirectly causes a disproportionate shift in _st_ . While, in theory, a
good BayesOpt algorithm should adapt to such problem characteristics, in practice they make the
optimization problem harder, especially when the optimization budget is small.


To address this, we “flatten” the distribution [8] within [0 _,_ 1]. Our transformation is simple: we divide
the interval [0 _,_ 1] into _B_ bins, and map approximately _|Dtrain|/B_ uncertainty scores to each bin,
while maintaining order between the original and mapped scores. Within a bin, the mapped scores
are linearly spread across its range. This distributes the mapped scores approximately uniformly in
the range [0 _,_ 1]. The algorithm is detailed in Algorithm 4.


Figure 4 visualizes the process of flattening. The original and modified uncertainty distributions for
the datasets Sensorless and covtype.binary are shown in Figure 4(a) and 4(b) respectively.


While Sensorless appears to have a non-smooth distribution, and flattening here might help, this
seems redundant for covtype.binary. _However,_ _since_ _this_ _step_ _is_ _computationally_ _cheap,_ _we_
_perform this for all our experiments, saving us the effort of assessing its need._


8Distribution transformations have a long history in statistics, e.g., _power transforms_ like the _Box-Cox_ (Box
& Cox, 1964) and _Yeo-Johnson_ (Yeo & Johnson, 2000) transforms. Within ML, _Batch_ _Normalization_ (Ioffe
& Szegedy, 2015) is a popular example of a distribution transformation applied to a loss landscape (Santurkar
et al., 2018).


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


**Algorithm 4:** Flatten distribution of uncertainty scores _{u_ ( _x_ 1) _, u_ ( _x_ 2) _, ..., u_ ( _xN_ ) _}_
**Data:** _{u_ ( _x_ 1) _, u_ ( _x_ 2) _, ..., u_ ( _xN_ ) _},_ number of bins _B_
**Result:** _{u_ _[′]_ ( _x_ 1) _, u_ _[′]_ ( _x_ 2) _, ..., u_ _[′]_ ( _xN_ ) _}_

**1** _bin_ ~~_s_~~ _ize ←⌈N/B⌉, bin_ ~~_r_~~ _ange ←_ 1 _/B_

**2** _bin_ ~~_m_~~ _in ←_ [ ] _, bin_ ~~_m_~~ _ax ←_ [ ]

**3** Let _sortedIndex_ ( _i_ ) _∈{_ 1 _,_ 2 _, ..., N_ _}_ be the index of _u_ ( _xi_ ) in the sequence of scores ordered by
non-decreasing values.

**4** **for** _j_ _←_ 1 **to** _B_ **do**

**5** _bin_ ~~_m_~~ _in_ [ _j_ ] _←_ min _{u_ ( _xi_ ) _|i ∈{_ 1 _,_ 2 _, ..., N_ _} ∧_ _sortedIndex_ ( _i_ ) = _j}_

**6** _bin_ ~~_m_~~ _ax_ [ _j_ ] _←_ max _{u_ ( _xi_ ) _|i ∈{_ 1 _,_ 2 _, ..., N_ _} ∧_ _sortedIndex_ ( _i_ ) = _j}_

**7** **end**

**8** **for** _i ←_ 1 **to** _N_ **do**

**9** _j_ _←_ _sortedIndex_ ( _i_ )

**10** _bin_ ~~_n_~~ _um ←⌈j/bin_ ~~_s_~~ _ize⌉_

**11** _boundary_ _low_ _←_ ( _bin_ ~~_n_~~ _um −_ 1) _× bin_ ~~_r_~~ _ange_ + _δ_

**12** _boundary_ _high ←_ _bin_ _num × bin_ ~~_r_~~ _ange −_ _δ_

_u_ ( _xi_ ) _−bin_ ~~_m_~~ _in_ [ _j_ ]
**13** _u_ _[′]_ ( _xi_ ) _←_ _low_ + _bin_ _max_ [ _j_ ] _−bin_ _min_ [ _j_ ] _[×]_ [ (] _[boundary]_ ~~_[h]_~~ _[igh][ −]_ _[boundary]_ ~~_[l]_~~ _[ow]_ [)]

**14** **end**

**15** **return** _{u_ _[′]_ ( _x_ 1) _, u_ _[′]_ ( _x_ 2) _, ..., u_ _[′]_ ( _xN_ ) _}_


(a) Sensorless, GBM (b) covtype.binary, GBM


Figure 4: Example of curve-flattening, for datasets (a) Sensorless and (b) covtype.binary.
The uncertainty scores shown are obtained using the _GBM_ oracle.


Our transformation is invertible, which is useful in analyzing the observations from our experiments.
Note however, it is not differentiable because of the discontinuities at the bin-boundaries; we also
don’t require this property.


The transformation affects line 7 in Algorithm 2. Instead of sampling based on the actual oracle
uncertainty scores:


_pj_ _←_ _Beta_ ( _uMO_ ( _xj_ ); _Ai, Bi_ ) (7)
we sample based on the transformed uncertainty scores, _u_ _[′]_ _MO_ [(] _[x][j]_ [)][:]


_pj_ _←_ _Beta_ ( _u_ _[′]_ _MO_ [(] _[x][j]_ [);] _[ A][i][, B][i]_ [)] (8)
In §A.6 we show that smoothing indeed has a positive effect.


A.6 EFFECT OF SMOOTHING


We first consider the question: does flattening (§A.5) help? Table 2 contrasts _improved_ _F_ 1 scores
obtained without (rows denoted as “original”) and with (denoted “flattened”) flattening the uncer

24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Table 2: Improved scores averaged over three trials, shown for different parameter settings, with
and without flattening. Here, Setting 1 is _{max_ ~~_c_~~ _omponents_ = 500 _, scale_ = 10000 _}_ and Setting
2 is _{max_ ~~_c_~~ _omponents_ = 50 _, scale_ = 10 _}_ . “curr.” signifies this is the current setting for our
experiments in the main paper, while “low” signifies lower values of parameters. Highlighted cells
indicate positive effect of fattening.


Setting 1 (curr.) Setting 2 (low)


dataset dist. 1 2 3 1 2 3


tainty distribution. This is shown for the datasets Sensorless and covtype.binary, for
model _size_ _∈{_ 1 _,_ 2 _,_ 3 _}_, with _model_ = _LPM_ and _oracle_ = _GBM_ . Two different parameter settings are used: (a) In Setting 1, maximum allowed _Beta_ components are 500 and _scale_ = 10000
(b) Setting 2 looks at much lower values of these parameters where maximum allowed components
is 50 and _scale_ = 10. The scores presented are the average over three trials.


We observe that while flattening influences results, other parameters determine the magnitude of its
effect. At Setting 1, Sensorless is affected at _size_ = 1 (flattening is better), but at higher sizes
the differences seem to be from random variations across trials. At Setting 2 however, the differences
are seen for _size ∈{_ 1 _,_ 2 _,_ 3 _}_ (flattening is better). For covtype.binary only _size_ = 2 seems to
be affected in either setting.


Recall we had noted in Figure 4 that the datasets Sensorless and covtype.binary have
non-smooth and smooth uncertainty distributions respectively. The observations in Table 2 align
well with the expectation that Sensorless is positively affected by the transformation, while
results for covtype.binary remain mostly unchanged.


Based on these tests, we hypothesize that for non-smooth uncertainty distributions, flattening makes
our technique robust across parameter settings. It does not affect smooth distributions in a significant
way. Of course, rigorous and extensive tests are required to conclusively establish this effect.


A.7 DATASETS


Table 3 provides details about the various datasets used in the experiments in §4. All of these are
publicly available on the _LIBSVM_ website (Chang & Lin, 2011a).


The “Label Entropy” column indicates how balanced a dataset is wrt its classes. For a dataset with
_N_ instances and _C_ labels, this is calculated as:


Label Entropy =             - _−pj_ log _C pj_ (9)

_j∈{_ 1 _,_ 2 _,...,C}_

[=] _[ j][}|]_
Here, _pj_ = _[|{][x][i][|][y][i]_

_N_


Label Entropy _∈_ [0 _,_ 1], where values close to 1 denote the dataset is nearly balanced, and values
close to 0 represent relative imbalance.


A.8 VALIDATION RESULTS


An extended version of the results shown in Table 1 are presented here in Table 4. This shows results
for all combinations of models and oracles: _{LPM, DT_ _} × {GBM, RF_ _}_ .


We also perform a _Wilcoxon signed-rank test_ (Wilcoxon, 1945) to measure statistical significance.
We use this test as it has been shown to be useful in comparing classifiers (Demˇsar, 2006; Benavoli
et al., 2016; Japkowicz & Shah, 2011). Results are shown in figure 5 for the following test setup:


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


Table 3: We use the following datasets available on the LIBSVM website (Chang & Lin, 2011a).
Their original source is mentioned in the “Description” column. 10000 instances from each dataset
are used. A _train_ : _val_ : _test_ split ratio of 60 : 20 : 20 is used for _Dtrain, Dval_ and _Dtest_ in
Algorithm 1. The splits are stratified wrt labels.


S.No. Dataset Dimensions # Classes Label Entropy Description


1 cod-rna 8 2 0.92 Predict presence of non-coding RNA common to a pair
of RNA sequences, based on individual sequence properties and their similarity (Uzilov et al., 2006).
2 ijcnn1 22 2 0.46 Time series data produced by an internal combustion
engine is used to predict normal engine firings vs misfirings (Prokhorov, 2001). Transformations as in Chang
& Lin (2001).
3 higgs 28 2 1.00 Predict if a particle collision produces Higgs bosons or
not, based on collision properties (Baldi et al., 2014).
4 covtype.binary 54 2 1.00 Modification of the _covtype_ dataset (see row 12), where
classes are divided into two groups (Collobert et al.,
2002).
5 phishing 68 2 0.99 Various website features are used to predict if the website is a _phishing_ website (Mohammad et al., 2012).
Transformations used as in Juan et al. (2016)
6 a1a 123 2 0.80 Predict whether a person makes over 50K a year, based
on census data variables (Dua & Graff, 2017). Transformations as in Platt (1998).
7 pendigits 16 10 1.00 Classify handwritten digit samples into the digits 0-9
(Alimoglu & Alpaydin, 1996; Dua & Graff, 2017).
8 letter 16 26 1.00 Images of the capital letters A-Z were produced by random distortion of these characters from 20 fonts. The
task is to classify these character images as one of the
original letters (Michie et al., 1995). Transformations
as in Hsu & Lin (2002).
9 Sensorless 48 11 1.00 Based on phase current measurements of an electric motor, predict different error conditions (Paschke et al.,
2013). We use the transformations from Wang et al.
(2018).
10 senseit ~~a~~ co 50 3 0.95 Predict vehicle type using acoustic data gathered by a
sensor network (Duarte & Hu, 2004).
11 senseit ~~s~~ ei 50 3 0.94 Predict vehicle type using seismic data gathered by a
sensor network (Duarte & Hu, 2004).
12 covtype 54 7 0.62 Predict forest cover type from cartographic variables
(Dean & Blackard, 1998; Dua & Graff, 2017).
13 connect-4 126 3 0.77 Predict if the first player wins, loses or draws, based on
board positions of the board game _Connect_ _Four_ (Dua
& Graff, 2017).


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


Table 4: This table shows the average improvements, _δF_ 1, over five runs for different combinations of models
and oracles: _{LPM, DT_ _} × {GBM, RF_ _}_ . This is an extended version of the results in Table 1. The improvements are measured relative to the model at the first iteration.The best improvement for a model size and
oracle is indicated in bold. Here, _δF_ 1 _∈_ ( _−∞, ∞_ ). Negative improvements are shown in underlined.


dataset model ~~o~~ ra 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15


cod-rna lpm ~~g~~ bm 1.39 12.53 **14.76** **15.73** 14.97 12.00 0.00 **0.08**  -  -  -  -  -  -  lpm ~~r~~ f **2.66** **13.91** 14.69 15.34 **16.06** **12.49** **8.30** 0.00        -        -        -        -        -        -        
dt ~~g~~ bm **0.00** **0.00** 0.00 1.26 0.00 **0.00** 0.00 0.00 -0.28 0.08           -           -           -           -           dt ~~r~~ f 0.00 0.00 **1.78** **2.28** **0.39** -0.02 **0.17** **0.47** **0.00** **0.72**           -           -           -           -           

ijcnn1 lpm ~~g~~ bm -0.16 **3.36** **3.93** 0.00 **5.19** **4.18** **3.85** **3.79** **3.69** 2.99 **2.97** 3.21 3.11 3.26 3.02
lpm ~~r~~ f **0.19** 2.80 3.36 **3.65** 3.33 1.94 3.58 3.30 3.46 **3.81** 2.66 **4.65** **3.99** **3.82** **4.85**

dt ~~g~~ bm 1.96 12.00 **10.15** **11.37** **10.63** 7.18 3.63 **4.52** **2.91** **1.78** **1.93** **2.29** 1.47 **2.26** 0.00
dt ~~r~~ f **4.06** **12.10** 8.95 10.75 10.13 **8.25** **5.38** 2.46 2.63 1.25 1.46 1.37 **1.91** 0.00 **1.38**


higgs lpm ~~g~~ bm **29.29** **17.80** 11.40 6.56 3.06 2.68 **3.16** **2.90** **2.67** **2.82** 2.65 1.79 2.62 2.19 1.63
lpm ~~r~~ f 26.71 17.29 **15.06** **10.60** **5.35** **4.04** 2.35 2.03 1.66 1.89 **2.91** **2.94** **3.31** **2.58** **2.22**

dt ~~g~~ bm 0.00 0.00 **1.86** 0.26 0.93 0.45           -           -           -           -           -           -           -           -           dt ~~r~~ f **4.04** **1.26** 1.74 **1.32** **1.54** **0.91**           -           -           -           -           -           -           -           -           

covtype.binary lpm ~~g~~ bm 76.52 **66.39** **29.17** **12.51** **9.18** **5.28** **4.94** **4.56** **3.92** **3.56** **3.62** **3.31** **2.59** **2.83** **2.39**
lpm ~~r~~ f **96.77** 63.38 14.36 9.61 6.79 3.94 2.93 2.81 2.96 2.84 2.31 2.26 2.00 2.43 2.22

dt ~~g~~ bm **0.00** **0.00** **2.35** 1.27 1.18 1.11 0.00 0.00 0.00           -           -           -           -           -           dt ~~r~~ f 0.00 0.00 2.10 **2.33** **2.44** **2.39** **1.84** **2.19** **1.65** 0.70           - 0.89           -           -           

phishing lpm ~~g~~ bm **0.00** 1.88 2.88 3.05 3.22 3.25 2.99 1.69 1.42 **1.45** **1.29** 0.00 0.00 0.00 0.00
lpm ~~r~~ f 0.00 **2.14** **3.29** **3.22** **3.59** **3.79** **3.29** **2.05** **1.42** 1.44 1.24 **1.23** **1.16** **1.26** **1.02**

dt ~~g~~ bm **0.00** 0.00 **0.00** 0.07 **0.39** **0.00** **0.28** 0.22 **0.44** **0.23** 0.00 **0.00** **0.00** 0.00 **0.00**
dt ~~r~~ f 0.00 **0.72** 0.00 **0.57** 0.00 -0.17 0.13 **0.48** 0.13 0.05 **0.03** -0.03 -0.28 **0.00** -0.16


a1a lpm ~~g~~ bm **0.00** 2.55 7.58 8.98 8.40 8.03 8.90 8.23 8.17 7.90 5.96 7.10 6.97 6.18 5.73
lpm ~~r~~ f 0.00 **4.17** **8.81** **9.92** **9.88** **9.47** **8.99** **9.31** **9.19** **9.26** **9.33** **8.25** **7.15** **7.55** **7.98**

dt ~~g~~ bm **0.00** 5.54 2.39 3.84 **3.55** 2.55 1.51 2.25 4.87           -           -           -           -           -           dt ~~r~~ f 0.00 **6.44** **3.36** **5.60** 3.40 **5.94** **6.06** **4.97** **4.89** 4.01 4.73 5.21           -           - 4.53


pendigits lpm ~~g~~ bm **51.39** **23.44** 16.18 **8.95** **8.84** **6.63** 4.86 **1.83** 2.27 2.16 **2.44** 2.16 **3.33** 2.97 **2.73**
lpm ~~r~~ f 46.28 22.74 **21.72** 8.80 8.47 6.29 **6.48** 1.69 **3.03** **2.79** 2.34 **2.68** 2.70 **3.02** 0.00

dt ~~g~~ bm 14.02 **6.72** 5.11 13.14 6.42 4.20 2.46 **1.09** **0.98** **0.16** -0.26 **0.00** **0.00** **0.00** **0.00**
dt ~~r~~ f **21.46** 4.18 **5.22** **14.51** **7.36** **4.55** **2.86** 0.00 0.00 0.00 **0.00** 0.00 0.00 0.00 0.00


letter lpm ~~g~~ bm 57.06 48.48 59.85 **29.76** **36.09** 19.27 20.37 16.08 17.55 15.16 17.26 16.51 18.46 17.19 15.55
lpm ~~r~~ f **61.06** **65.34** **64.26** 23.69 35.20 **26.15** **22.10** **20.74** **20.91** **20.31** **19.28** **21.40** **20.77** **19.39** **18.18**

dt ~~g~~ bm **0.00** **13.98** 25.05 **33.96** 32.05 15.49 **11.17** **0.00** **4.26** **3.50** **1.99** 0.00 0.00 0.00 **0.00**
dt ~~r~~ f 0.00 12.21 **28.67** 33.47 **33.51** **18.41** 8.10 0.00 1.84 1.21 1.31 **0.67** **0.61** **0.11** -0.08


Sensorless lpm ~~g~~ bm 216.47 **257.56** **178.31** **117.01** **90.70** **83.90** **73.50** **65.95** 61.57 57.97 56.54 57.15 55.45 66.24 68.24
lpm ~~r~~ f **224.18** 210.28 134.44 115.00 85.85 74.96 66.77 61.10 **66.88** **64.65** **69.00** **70.09** **72.91** **80.14** **82.15**

dt ~~g~~ bm -0.01 42.42 **68.13** 44.38 **17.39** **10.32** 1.82 **1.44** **0.79** **0.64** 0.41 0.12 **0.00** -0.02 **0.34**
dt ~~r~~ f **0.00** **52.54** 57.10 **44.61** 16.63 6.19 **2.19** 0.96 0.51 0.00 **0.48** **0.33** 0.00 **0.00** 0.10


senseit ~~a~~ co lpm ~~g~~ bm 173.71 170.68 63.95 **44.20** 33.49 22.99 19.14 13.50 10.29 7.59 6.26 5.92 **5.30** 4.89 4.32
lpm ~~r~~ f **177.67** **181.26** **79.86** 42.86 **37.60** **28.80** **23.75** **19.06** **13.91** **10.74** **8.48** **6.09** 5.20 **5.32** **4.62**

dt ~~g~~ bm 14.89 0.00 **3.71** 2.32 **4.85** 0.81 **0.00**           -           -           -           -           -           -           -           dt ~~r~~ f **20.03** **2.54** 3.64 **5.91** 3.34 **2.63** 0.00 0.00           -           -           -           -           -           -           

senseit ~~s~~ ei lpm ~~g~~ bm 160.59 **65.27** 23.44 10.48 6.76 4.86 4.82 4.46 4.79 4.12 4.54 **5.17** 3.91 4.21 **4.46**
lpm ~~r~~ f **165.98** 63.72 **31.58** **14.94** **9.07** **5.79** **4.95** **5.07** **5.24** **4.70** **4.60** 3.74 **4.30** **4.35** 4.35

dt ~~g~~ bm **2.66** **1.01** **3.49** **2.29** **0.95** **1.30** **1.37** 0.00           -           -           -           -           -           -           dt ~~r~~ f 2.33 0.00 3.36 1.65 0.87 0.00 -1.23           -           -           -           -           -           -           -           

covtype lpm ~~g~~ bm **36.87** **49.24** **12.78** **11.21** 7.84 7.15 7.15 8.07 7.70 8.25 **10.94** **8.35** 4.37 8.77 5.84
lpm ~~r~~ f 32.15 39.49 10.49 8.53 **8.11** **8.59** **9.61** **11.99** **11.22** **9.91** 8.47 8.16 **10.34** **13.76** **12.92**

dt ~~g~~ bm 342.27 92.85 43.23 **20.04** 8.14 **8.05** **5.67** 3.26 **4.92** **3.52** **2.72** 0.00 **0.00** **0.00** **1.74**
dt ~~r~~ f **354.45** **98.94** **50.87** 14.10 **9.46** 7.38 4.76 **4.20** 0.94 1.81 2.30 **0.71** -0.37 0.00 0.00


connect-4 lpm ~~g~~ bm **37.62** 11.66 12.01 6.84 5.68 6.82 4.58 2.10 3.82 3.21 **3.02** **3.64** **2.32** **2.97** **3.40**
lpm ~~r~~ f 33.77 **12.99** **17.60** **14.66** **15.91** **10.73** **6.38** **5.35** **7.07** **6.98** 2.84 3.14 2.09 2.52 2.46

dt ~~g~~ bm 89.33 **29.23** 20.20 **12.10** 9.73 9.88 7.82 7.43 0.57 4.61 1.08 **3.35** 2.23 **1.15** **1.55**
dt ~~r~~ f **113.71** 21.91 **20.52** 11.23 **16.86** **10.96** **10.64** **9.11** **6.51** **5.88** **6.76** 2.16 **2.97** 0.61 0.00


1. We divide the analysis by model size. This is because size strongly influences _δF_ 1 (as in
Table 4).


2. Normalized model sizes are used. Binning of model sizes is done using _Sturges_ _rule_
(Sturges, 1926).


3. The _one-sided_ version of the _paired_ test is performed for each bin, where pairs of scores
_F_ 1 _[base]_ and _F_ 1 _[∗]_ for a dataset, for models with sizes assigned to the bin, are compared. In
cases were where multiple model sizes for a dataset fall within the same bin, _F_ 1 _[base]_ and
_F_ 1 _[∗]_ are first averaged and then compared.


4. The following hypotheses are tested:


       - **H0**, null hypothesis: accuracies of models produced by our technique are not better.

       - **H1**, alternate hypothesis: accuracies of models trained using the oracle are better.


_p-values_ are shown for each bin. Small _p-values_ favor **H1**, i.e., our algorithm.


27


0.0010


0.0008


0.0006


0.0004


0.0002


0.0000


0.04


0.03


0.02


0.01


0.00


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


(a) model=LPM, oracle=GBM


0.0010


0.0008


Figure 5: These plots show the _p-values_ for the Wilcoxon signed-rank test, with the null hypothesis
_H_ 0: using the oracle does _not_ produce better F1 test scores. The bin boundaries are selected using
the _Sturges_ rule (Sturges, 1926). Low _p-values_ favor our algorithm.


5. Scores of _δF_ 1 = 0 are split equally between positive and negative ranks [9] .


A.9 COMPARISON WITH THE APPROACH BASED ON DENSITY TREES


As mentioned in §4.2 we benchmark against the density tree technique Ghose & Ravindran (2020)
because that’s the closest in terms of methodology. Their metric is slightly different from ours.
Instead of reporting results for _F_ 1 _[∗]_, they report them for _max_ ( _F_ 1 _[∗]_ _, F_ 1 _[base]_ ). This is an “outcomecentric” view [10], where you can’t do worse than your best model. For this case, _δF_ 1 _test_ _∈_ [0 _, ∞_ ).
We also follow this scoring scheme in this section to match their reporting.


We report two scores for comparison ( _den_ and _ora_ denote density trees and our oracle based technique respectively):


1. To compare improvements, we use the _Scaled Difference in Improvement (SDI)_ :


�( _δF_ 1 _ora −_ _δF_ 1 _den_ ) _/H,_ if _H_ _>_ 0
_SDI_ = (10)
0 _,_ if _H_ = 0


where _H_ = max _{δF_ 1 _[den]_ _, δF_ 1 _[ora]_ _}_


Here _δF_ 1 _[ora]_ and _δF_ 1 _[den]_ are the improvements from our technique and by using density
trees, respectively. The scaling wrt _H_ ensures that _SDI_ _∈_ [ _−_ 1 _,_ 1] making it convenient to
interpret. Note that _H_ _≥_ 0 since both _δF_ 1 _[ora]_ _≥_ 0 and _δF_ 1 _[den]_ _≥_ 0 in the current scoring


9The zplit option in [https://numpy.org/doc/stable/reference/generated/numpy.](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html)
[histogram_bin_edges.html is used.](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html)
10Another reason provided is that with a sufficient budget the optimizer will eventually learn to set _po_ = 1,
thus emulating _M_ _[base]_ exactly, if _M_ _[base]_ is indeed the best possible model. In this case _δF_ 1 = 0 as per Equation
3.


28


(b) model=LPM, oracle=RF


2 4 6 8
model size bins

(d) model=DT, oracle=RF


2 4 6 8
model size bins


0.0006


0.0004


0.0002


0.0000


0.025


0.020


0.015


0.010


0.005


0.000


2 4 6 8
model size bins

(c) model=DT, oracle=GBM


2 4 6 8
model size bins


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


scheme. For brevity, we average the _SDI_ scores at the level of a dataset, across model
sizes, for a given model and oracle. This averaged score is denoted by _SDI_, and this is
what we report.


2. Since _SDI_ is aggregated over model sizes, we also report the percentage of times
_δF_ 1 _[ora]_ _> δF_ 1 _[den]_ across these model sizes. This is denoted as _pct_ ~~_b_~~ _etter_


All _δF_ 1 _[ora]_ and _δF_ 1 _[den]_ scores used are the _averaged over five runs_ .


We consider our approach to be better if _SDI_ _**>**_ **0** _and pct_ ~~_b_~~ _etter_ _**>**_ **50** %. These scores are shown
in Table 5. Since the density trees approach lacks a notion of an oracle, we present results for GBMs
and RFs separately. Numbers that represent superior performance by density trees are underlined.
Note also the two special groupings:


    - **ANY** : For each model size, the _SDI_ score considered is the higher of the ones obtained
from using the _GBM_ or _RF_ as oracles. The _SDI_ and _pct_ ~~_b_~~ _etter_ scores are computed
based on these scores. This grouping represents the ideal way to use our technique in
practice: try multiple oracles and pick the best.


    - **OVERALL** : This averages results across datasets, to provide an aggregated view.


The cells identified by **OVERALL** _and_ **ANY** provide comparison numbers aggregated over datasets,
model sizes and oracles.


Table 5: LPM, DT compared to the Density Tree approach. All _δF_ 1 _[ora]_ and _δF_ 1 _[den]_ scores used are the
_average over five runs_ . Cases where density trees fare better are underlined. The line in the middle separates
binary class datasets (top) from multi-class ones (bottom).


LPM DT


dataset GBM RF **ANY** GBM RF **ANY**


cod-rna -0.38, 0.00% -0.45, 0.00% -0.33, 0.00% 0.51, 60.00% 0.50, 70.00% 0.65, 80.00%
ijcnn1 0.06, 66.67% 0.11, 80.00% 0.20, 93.33% 0.23, 53.33% 0.68, 100.00% 0.68, 100.00%
higgs -0.07, 40.00% -0.07, 40.00% 0.04, 46.67% 0.23, 50.00% 0.61, 83.33% 0.61, 83.33%
covtype.binary -0.16, 40.00% -0.33, 13.33% -0.15, 40.00% 0.23, 66.67% 0.26, 72.73% 0.38, 81.82%
phishing 0.30, 80.00% 0.37, 86.67% 0.38, 86.67% 0.11, 26.67% -0.00, 26.67% 0.23, 46.67%
a1a -0.03, 60.00% 0.13, 66.67% 0.13, 66.67% -0.06, 44.44% 0.43, 75.00% 0.52, 83.33%


pendigits 0.59, 100.00% 0.59, 93.33% 0.62, 100.00% 0.23, 60.00% 0.16, 46.67% 0.25, 60.00%
letter 0.79, 100.00% 0.81, 100.00% 0.81, 100.00% 0.02, 33.33% -0.34, 13.33% 0.06, 40.00%
Sensorless 0.64, 100.00% 0.65, 100.00% 0.66, 100.00% -0.23, 20.00% -0.39, 20.00% -0.23, 20.00%
senseit ~~a~~ co 0.55, 100.00% 0.63, 100.00% 0.63, 100.00% 0.50, 85.71% 0.37, 75.00% 0.39, 75.00%
senseit ~~s~~ ei 0.61, 100.00% 0.66, 100.00% 0.67, 100.00% -0.25, 42.86% 0.51, 100.00% 0.51, 100.00%
covtype 0.20, 80.00% 0.39, 93.33% 0.43, 100.00% 0.26, 66.67% 0.16, 66.67% 0.40, 80.00%
connect-4 0.23, 73.33% 0.24, 66.67% 0.38, 86.67% -0.23, 33.33% -0.13, 53.33% 0.08, 66.67%
**OVERALL** 0.28, 75.00% 0.32, 75.00% 0.37, 81.38% 0.10, 47.06% 0.16, 57.23% 0.31, 67.30%


The predominance of non-underlined values indicate that our technique performs better in most
settings. In both cases, the **OVERALL** + **ANY** entries indicate that our technique works better on
average - in terms of both the extent of improvement _SDI_ and its frequency _pct_ ~~_b_~~ _etter_ .


29


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


Model families


|Col1|Col2|Col3|Col4|Ca<br>wi|
|---|---|---|---|---|
||||||
||||||
|||||<br>|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||


Figure 6: (a) shows a standard measurement scheme with datasets in rows and model families in
columns. Statistical tests are performed on the column values. Row-wise ranks are first computed for
calculating the mean rank. (b) To account for model sizes, we allow rows to represent combinations
of datasets and model sizes. See text for details.


A.10 EVALUATING COMPETITIVENESS


As mentioned in the main paper in §4.2, we consider the following tasks for evaluating competitiveness:


1. Building cluster explanation trees.


2. Prototype-based classification.


For evaluation on each of these tasks, we follow a common theme: (a) first, we show that a traditional
technique is almost always not as good as newer and specialized techniques, and, (b) then we show
that its performance may be radically improved by learning the training distribution. Collectively,
these evaluations show that the strategy of learning the training distribution is both _general_ - may
be applied to different tasks, models, notions of model sizes - and _effective_ - results in competitive
performance. We first detail our measurement strategy.


A.10.1 MEASUREMENT


While each task-specific section contains a detailed discussion on the experiment setup, we discuss
some common aspects here:


1. To compare model families _F_ 1 _, F_ 2 _, F_ 3, each of which is, say, used to construct models
for different sizes _η_ _∈{_ 2 _,_ 3 _}_, for datasets _D_ 1 _, D_ 2 _, D_ 3, we use the _mean rank_, and support
our conclusions with statistical tests such as the _Friedman_ (Friedman, 1937) and _Wilcoxon_
_signed-rank_ (Wilcoxon, 1945) tests [11] .

Typically mean rank is used to compare model families based on their accuracies across
datasets - which, ignoring model sizes, may be visualized as a 3 _×_ 3 table here, with rows
representing datasets, and columns denoting model families     - see Figure 6(a). An entry
such as “ _D_ 2 _, F_ 3” represents the accuracy (or some other metric) of a model from family
_F_ 3 on dataset _D_ 2. Models are ranked on a per-dataset basis, i.e., row-wise, and the average
ranks (computed per family, i.e., column-wise) are reported (lower is better). For statistical
tests, the column values are directly used.


11The _Wilcoxon signed-rank_ test was used here since it has been advocated by various studies for measuring
classification performance (Demˇsar, 2006; Benavoli et al., 2016; Japkowicz & Shah, 2011).


30


Model families


Compare column values
for _**statistical tests**_ .


[  ...  ]


Calculate
column-wise
_**mean ranks**_ .


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


However, we have an additional factor here - the model size. To avoid inventing a custom
metric, we assimilate it in the previous scheme by using the combination of datasets _and_
model sizes as a row - see Figure 6(b). We think of such combinations as “pseudo-dataset”
entries, i.e., now we have a 6 _×_ 3 table, with rows for _D_ 1 [2] _[, D]_ 1 [3] _[, D]_ 2 [2] _[, D]_ 2 [3] _[, D]_ 3 [2] _[, D]_ 3 [3][, and same]
columns as before. The entry for “ _D_ 1 [2] _[,][ F]_ [3][” indicates the accuracy of a model of size][ 2][ from]
family _F_ 3 on dataset _D_ 1.
Effectively, now the comparisons automatically account for model size since we use
pseudo-datasets instead of datasets. **Note** that no new datasets are being created     - we are
merely defining a convention to include model size in the familiar dataset-model crossproduct table.

2. For _each_ model family, model size and dataset combination (essentially a cell in this crossproduct table), models are constructed multiple times (we refer to these as multiple “trials”),
and their scores are averaged. Five trials were used in our experiments.


A.10.2 EXPLAINABLE CLUSTERING


The first task we investigate is the problem of _Explainable_ _Clustering_ . Introduced by Moshkovitz
et al. (2020), the goal is to explain cluster allocations as discovered by techniques such _k-means_
or _k-medians_ . This is achieved by constructing axis-aligned decision trees with leaves that either
exactly correspond to clusters, e.g., _Iterative Mistake Minimization (IMM)_ Moshkovitz et al. (2020),
or are proper subsets, e.g., _Expanding Explainable k-Means Clustering (ExKMC)_ Frost et al. (2020).
We consider the former case here, i.e., a tree must possess exactly _k_ leaves to explain _k_ clusters.


For a specific clustering _C_, let _C_ ( _xi_ ) denote the assigned cluster for an instance _xi, i_ = 1 _...N_,
where _C_ ( _xi_ ) _∈{_ 1 _,_ 2 _, ..., k}_, and the cluster centroids by _µj, j_ = 1 _, ..., k_ . The cost of clustering _J_
is then given by:


In the case of an explanation trees with _k_ leaves, _µj_ are centroids of leaves. Cluster explanation
techniques attempt to minimize this cost.


The price of explainability maybe measured as the _cost ratio_ [12] :

cost ratio = _[J][Ex]_ (12)

_JKM_


Here _JEx_ is the cost achieved by an explanation tree, and _JKM_ is the cost obtained by a standard
k-means algorithm. It assumes values in the range [1 _, ∞_ ], where the lowest cost is obtained when
using k-means, i.e., _JEx_ and _JKM_ are the same.


One may also indirectly minimize the cost in the following manner: use k-means to produce a
clustering, use the cluster allocations of instances as their labels, and then learn a standard decision
tree for classification, e.g., CART. This approach has been shown to be often outperformed by tree
construction algorithms that directly minimize the cost in Equation 11, e.g., IMM.


_⋄_ **Algorithms and Hyperparameters**
The algorithms we compare and their hyperparameter settings are as follows:


1. **Iterative** **Mistake** **Minimization** **(IMM)** Moshkovitz et al. (2020): This generates a decision tree via greedy partitioning using a criterion that minimizes number of mistakes
at each split (the number of points separated from their corresponding reference cluster
center). There are no parameters to tune. We used the implementation available here:
[https://github.com/navefr/ExKMC, which internally uses the reference imple-](https://github.com/navefr/ExKMC)
mentation for IMM.

2. **ExShallow** Laber et al. (2021): Here, the decision tree construction explicitly accounts for
minimizing explanation complexity while targeting a low cost ratio. The trade-off between


12This is referred to as the _cost ratio_ in Frost et al. (2020), _price of explainability_ in Moshkovitz et al. (2020)
and _competitive ratio_ in Makarychev & Shan (2022).


31


 - _||xi −_ _µj||_ [2] 2 (11)

_{xi|C_ ( _xi_ )= _j}_


_J_ = [1]

_N_


_k_


_j_ =1


**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


(a) mice-protein (f) Mean Rank


(b) Sensorless (c) covtype (d) covtype.binary (e) avila


Figure 7: Comparisons over explainable clustering algorithms are shown. (a) shows the comparison
for a specific dataset _mice-protein_ . (b), (c), (d) and (e) show comparisons over other datasets miniaturized to fit the page. (f) shows mean ranks of these techniques over five datasets across model
sizes; the Friedman test is conducted over the **top three** techniques only, with _p_ = 6 _._ 688 _×_ 10 _[−]_ [6] .
.


clustering cost and explanation size is controlled via a parameter _λ_ . This is set as _λ_ = 0 _._ 03
in our experiments; this value is used in the original paper for various experiments. We used
the reference implementation available here: [https://github.com/lmurtinho/](https://github.com/lmurtinho/ShallowTree)
[ShallowTree.](https://github.com/lmurtinho/ShallowTree)


3. **CART** : We use CART Breiman et al. (1984) as the traditional model to compare, and
maximize the classification accuracy for predicting clusters, as measured by the F1-macro
score. The implementation in _scikit_ Pedregosa et al. (2011) is used. During training, we
set the following parameters: (a) the maximum number of leaves (this represents _model_
_size η_ here) is set to the number of clusters _k_, and (b) the parameter _class_ ~~_w_~~ _eight_ is set to
_“balanced”_ for robustness to disparate cluster sizes. Results for CART are denoted with
label **CART** . We then apply our technique to CART; these results are denoted as **c** ~~**C**~~ **ART** .
We set _T_ = 2000, and use default settings for other parameters, e.g., _Ns_ _∈_ [400 _, |Xtrain|_ ].
Since we are explaining clusters (and not predicting on unseen data), the training, validation
and test sets are identical.


_⋄_ **Experiment Setup**
The comparison is performed over five datasets (limited to 1000 instances), and for each dataset,
_k_ = 2 _,_ 3 _, ...,_ 10 clusters are produced. Results for the cost ratio (Equation 12) are reported over _five_
trials. Evaluations are performed over the following publicly available datasets: _avila_, _covtype_, _cov-_
_type.binary_, _Sensorless_ Chang & Lin (2011b) and _mice-protein_ Dua & Graff (2017). We _specifically_
_picked_ these datasets since CART is known to perform poorly on them Frost et al. (2020); Laber
et al. (2021), and thus these provide a good opportunity to showcase the power of this technique.


_⋄_ **Observations**
Figure 7 presents our results. Figure 7(a) shows the plot for the _mice-protein_ dataset: the 95% confidence interval, in addition to cost ratio, is shown [13] . Plots for other datasets are shown miniaturized

- (b), (c), (d), (e) in the interest of space. The cost for k-means is shown for reference a blue horizontal line at _y_ = 1. Figure 7(f) shows the _mean_ _ranks_ of the various techniques (lower is better)
across datasets and number of clusters (as described in §A.10.1, trials scores are averaged), and its
title shows the _p-value_ = 6 _._ 688 _×_ 10 _[−]_ [6] of a _Friedman test_ conducted over _the top three techniques_ :
we restrict the test to top candidates since otherwise it would be very easy to obtain a low score


13It might come as a surprise that the cost ratio increases with increasing _k_, but this seems to be a transient
phenomenon; at even higher values of _k_ we do observe that cost ratios collectively decrease


32


**1728**

**1729**


**1730**

**1731**

**1732**

**1733**

**1734**

**1735**


**1736**

**1737**

**1738**

**1739**

**1740**

**1741**


**1742**

**1743**

**1744**

**1745**

**1746**


**1747**

**1748**

**1749**

**1750**

**1751**

**1752**


**1753**

**1754**

**1755**

**1756**

**1757**

**1758**


**1759**

**1760**

**1761**

**1762**

**1763**

**1764**


**1765**

**1766**

**1767**

**1768**

**1769**


**1770**

**1771**

**1772**

**1773**

**1774**

**1775**


**1776**

**1777**

**1778**

**1779**

**1780**

**1781**


favorable to us, due to the high cost ratios for CART. The low score indicates with high confidence
that ExShallow, IMM and c ~~C~~ ART do not produce the same outcomes.


From the plot of mean ranks in Figure 7(f), we observe that although CART performs quite poorly,
the application of our technique drastically improves its performance, to the extent that it competes
favorably with techniques like IMM and ExShallow; its mean rank places it between them. This is
especially surprising given that it doesn’t explicitly minimize the cost in Equation 11. We also note
the following _p-values_ from _Wilcoxon signed-rank_ tests:


    - CART vs c ~~C~~ ART: _p_ = 1 _._ 4783 _×_ 10 _[−]_ [6] . The low value indicates that using our technique
indeed significantly changes the accuracy of CART.

    - IMM vs c ~~C~~ ART: _p_ = 0 _._ 0155. The relatively high value indicates that the performance of
c ~~C~~ ART is competitive with IMM.


Here, both the Friedman and Wilcoxon tests are performed for combinations of datasets and _k_ - a
“pseudo-dataset”, as discussed in §A.10.1.


A.10.3 PROTOYPE-BASED CLASSIFICATION


(a) adult (f) Mean Rank


(b) covtype.binary (c) senseit-sei (d) senseit-aco (e) phishing


Figure 8: Various prototype-based classifiers are compared. (a) shows comparison for the dataset
_adult_ . Number of prototypes are shown as percentage of the training data on the _x-axis_, and is referred to as “compression”. (b), (c), (d) and (e) shows plots for other datasets - these are miniaturized
to fit the page. (f) shows the mean ranks of techniques based on five datasets; the Friedman test is
conducted over the **top four** techniques only, with _p_ = 3 _._ 5025 _×_ 10 _[−]_ [8] .
.


Next, we consider prototype-based classification. At training time, such techniques identify “prototypes” (actual training instances or generated instances), that maybe used to classify a test instance
based on their similarity to them. A popular technique in this family is the _k-Nearest_ _Neighbor_
_(kNN)_ . These are simple to interpret, and if a small but effective set of protoypes maybe identified,
they can be convenient to deploy on edge devices Gupta et al. (2017); Zhang et al. (2020). Prototypes also serve as minimal “look-alike” examples for explaining models (Li et al., 2018; Nauta
et al., 2021). Research in this area has focused on minimizing the number of prototypes that need to
be retained while minimally trading off accuracy.


We define some notation first. The number of prototypes we want is an input to our experiments, and
is denoted by _Np_ . We will also use _Kγ_ ( _xi, xj_ ) = _e_ _[−][γ][||][x][i][−][x][j]_ _[||]_ 2 [2] to denote the _Radial Basis Function_
_(RBF) kernel_, parameterized by the kernel bandwidth _γ_ .


_⋄_ **Algorithms and Hyperparameters**
These are the algorithms we compare:


33


**1782**

**1783**


**1784**

**1785**

**1786**

**1787**

**1788**

**1789**


**1790**

**1791**

**1792**

**1793**

**1794**

**1795**


**1796**

**1797**

**1798**

**1799**

**1800**


**1801**

**1802**

**1803**

**1804**

**1805**

**1806**


**1807**

**1808**

**1809**

**1810**

**1811**

**1812**


**1813**

**1814**

**1815**

**1816**

**1817**

**1818**


**1819**

**1820**

**1821**

**1822**

**1823**


**1824**

**1825**

**1826**

**1827**

**1828**

**1829**


**1830**

**1831**

**1832**

**1833**

**1834**

**1835**


1. **ProtoNN** Gupta et al. (2017): This technique uses a RBF kernel to aggregate influence
of prototypes. Synthetic prototypes are learned and additionally a “score” is learned for
each of them that designates their contribution towards _each_ class. The prediction function
sums the influence of neighbors using the RBF kernel, weighing contribution towards each
class using the learned score values; the class with the highest total score is predicted. The
method also allows for reducing dimensionality, but we don’t use this aspect [14] . The various
parameters are learned via gradient based optimization.

We use the _EdgeML_ library (Dennis et al., 2021), which contains the reference implementation for ProtoNN. For optimization, the implementation uses the version of
_ADAM_ Kingma & Ba (2015) implemented in _TensorFlow_ Abadi et al. (2015); we set
_num_ ~~_e_~~ _pochs_ = 200, _learning_ ~~_r_~~ _ate_ = 0 _._ 05, while using the defaults for other parameters. The _num_ ~~_e_~~ _pochs_ and _learning_ ~~_r_~~ _ate_ values are picked based on a limited search
among values _{_ 100 _,_ 200 _,_ 300 _}_ and _{_ 0 _._ 01 _,_ 0 _._ 05 _}_ respectively. The search space explored
for _γ_ is [0 _._ 001 _,_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _,_ 10]. Defaults are used for the other ProtoNN hyperparameters.


2. **Stochastic Neighbor Compression (SNC)** Kusner et al. (2014): This also uses a RBF kernel to aggregate influence of prototypes, but unlike ProtoNN, the prediction is performed
via the _1-NN rule_, i.e., prediction uses only the nearest prototype. The technique bootstraps
with randomly sampled _Np_ prototypes (and corresponding labels) from the training data,
and then modifies their coordinates for greater accuracy using gradient based optimization; the labels of the prototypes stay unchanged in this process. This is another difference
compared to ProtoNN, where in the latter, each prototype contributes to all labels to varying extents. The technique maybe extended to reduce the dimensionality of the data (and
prototypes); we don’t use this aspect.

We were unable to locate the reference implementation mentioned in the paper, so we
implemented our own version, with the help of the _JAXopt_ library Blondel et al. (2021).
For optimization, gradient descent with _backtracking_ _line_ _search_ is used. A total of 100
iterations for the gradient search is used (based on a limited search among these values:
_{_ 100 _,_ 200 _,_ 300 _}_ ), and each backtracking search is allowed up to 50 iterations. A grid search
over the following values of _γ_ is performed: [0 _._ 001 _,_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _,_ 10].


3. **Fast** **Condensed** **Nearest** **Neighbor** **Rule** Angiulli (2005): Learns a “consistent subset”
for the training data: a subset such that for any point in the training set (say with label _l_ ),
the closest point in this subset also has a label _l_ . Of the multiple variations of this technique
proposed in Angiulli (2005), we use **FCNN1**, which uses the _1-NN_ rule for prediction.
There are no parameters to tune. We used our own implementation.

A challenge in benchmarking this technique is it _does not_ accept _Np_ as a parameter; instead
it iteratively produces expanding subsets of prototypes until a stopping criteria is met, e.g.,
if prototype subsets _Vi_ and _Vi_ +1 are produced at iterations _i_ and _i_ + 1 respectively, then
they satisfy the relationship _Vi_ _⊂_ _Vi_ +1. For comparison, we consider the performance at
iteration _i_ to be the result of _Np_ prototypes where _Np_ is defined to be _|Vi|_, i.e., instead of
setting _Np_, we use the value the algorithm produces at each iteration.


4. **RBFN** : For the traditional model, we use _Radial Basis Function Networks (RBFN)_ Broomhead & Lowe (1988). For a binary classification problem with classes _{−_ 1 _,_ 1 _}_, given prototypes _xi, i_ = 1 _,_ 2 _, ..., p_, the label of a test instance _x_ is predicted as _sgn_ ( [�] _[p]_ _i_ _[w][i][K][γ]_ [(] _[x, x][i]_ [))]
(a score of 0 is set to a label of 1). Weights _wi_ are learned using linear regression. A onevs-rest setup is used for multiclass problems. For our baseline, we use cluster centres of a
_k-means_ clustering as our prototypes, where _k_ is set to _Np_ . These results are denoted using
the term **KM** ~~**R**~~ **BFN** . In our version, denoted by **c** ~~**R**~~ **BFN**, the _Np_ prototypes are sampled
from the training data. _Np_ represents _model size η_ here.
Note that the standard RBFN, and therefore the variants used here KM ~~R~~ BFN and c ~~R~~ BFN,
don’t provide a way to reduce dimensionality; this is the reason why this aspect of ProtoNN
and SNC wasn’t used (for fair comparison). We set _T_ = 1000 and _Ns_ was set to _Np_ to get
the desired number of prototypes.


14The implementation provides no way to switch off learning a projection, so we set the dimensionality of the
projection to be equal to the original number of dimensions. This setting might however learn a transformation
of the data to space within the same number of dimensions, e.g., translation, rotation.


34


**1836**

**1837**


**1838**

**1839**

**1840**

**1841**

**1842**

**1843**


**1844**

**1845**

**1846**

**1847**

**1848**

**1849**


**1850**

**1851**

**1852**

**1853**

**1854**


**1855**

**1856**

**1857**

**1858**

**1859**

**1860**


**1861**

**1862**

**1863**

**1864**

**1865**

**1866**


**1867**

**1868**

**1869**

**1870**

**1871**

**1872**


**1873**

**1874**

**1875**

**1876**

**1877**


**1878**

**1879**

**1880**

**1881**

**1882**

**1883**


**1884**

**1885**

**1886**

**1887**

**1888**

**1889**


Although all the above techniques use prototypes for classification, it is interesting to note variations
in their design: ProtoNN, SNC, KM ~~R~~ BFN use synthetic prototypes, i.e., they are not part of the
training data, while c ~~R~~ BFN and FCNN1 select _Np_ instances from the training data. The prediction
logic also differs: ProtoNN, KM ~~R~~ BFN, c ~~R~~ BFN derive a label from some function of the influence
by all prototypes, while SNC and FCNN1 use the 1-NN rule.


_⋄_ **Experiment Setup**
As before, we evaluate these techniques over five standard datasets: _adult, covtype.binary, senseit-_
_sei,_ _senseit-aco,_ _phishing_ Chang & Lin (2011b). 1000 training points are used, with _Np_ _∈_
_{_ 20 _,_ 40 _,_ 60 _,_ 80 _,_ 100 _,_ 140 _,_ 160 _,_ 180 _,_ 200 _}_ . Results are reported over five trials. The score reported
is the F1-macro score.


_⋄_ **Observations**
Results are shown in Figure 8. (a) shows the plot for the _adult_ dataset. The number of prototypes
are shown on the _x-axis_ as _percentages_ of the training data. Plots for other datasets are shown in (b),
(c), (d) and (e); these have been miniaturized to fit the page. Figure 8(f) shows the mean rank (lower
is better) across datasets and number of prototypes (as described in §A.10.1, trials are aggregated
over). The p-value of the Friedman test is reported, _p_ = 3 _._ 5025 _×_ 10 _[−]_ [8] . Here too, we do not
consider the worst performing candidate, FCNN1 - so as to not bias the Friedman test in our favor.


We observe in Figure 8(f) that while both ProtoNN and SNC outperform c ~~R~~ BFN, the performance
of SNC and c RBFN are close. We also observe that FCNN1 performs poorly; this matches the
observations in Kusner et al. (2014).


We also consider the following _p-values_ from _Wilcoxon signed-rank_ tests:


1. KM ~~R~~ BFN vs c ~~R~~ BFN: _p_ = 1 _._ 699 _×_ 10 _[−]_ [4] . The low value indicates that our technique
significantly improves upon the baseline KM ~~R~~ BFN.


2. SNC vs c ~~R~~ BFN: _p_ = 0 _._ 1260. The relatively high value here indicates that c ~~R~~ BFN is
competitive with SNC; in fact, at a confidence threshold of 0 _._ 1, their outcomes would not
be interpreted as significantly different.


As discussed in §A.10.1, these statistical tests are conducted over a combination of dataset and
model size.


A.11 RUNTIMES


For our experiments in the main paper, we used the _hyperopt_ library on account of its popularity and
maturity. Its acquisition function approximates the _Probability of Improvement (PI)_ utility function
(Song et al., 2022), which can exhibit greedy behavior (Garnett, 2023). In contrast, we might use a
different utility function, such as _Expected Improvement (EI)_, which is relatively more exploratory,
and thus, is likely to find better extrema.


We present some initial results around this line of thought. Instead of using the naive EI, we use
a numerically stable version called _LogEI_ (Ament et al., 2023) from the _BoTorch_ (Balandat et al.,
2020) package. We also note that we might use an acquisition function that can explicitly account
for noise [15], thus bypassing the need for estimating _st_ in Algorithm 3 via averaging (see notes in
§A.4).


Table 6 shows results for datasets _a1a_ and _ijcnn1_, where the interpretable model is a DT and the
oracle is a GBM. We assume _homoscedastic_ noise with variance of 0 _._ 5. _hyperopt_ was provided
a budget of _T_ = 3000 evaluations as in the main paper, while _BoTorch_ was allowed _T_ = 200
iterations. We observe that significant speedups are obtained without mostly noticeable change in
the quality of results - the only exception seems to be for _ijcnn1_, for _depth_ = 2. To take an example,
for the dataset _a1a_, for DT _depth_ = 1, the time taken by _hyperopt_ is 3193 _._ 27 seconds or **53 minutes**,
while _BoTorch_ offers of speedup of 21 _._ 57x; this is a runtime of 3193 _._ 27 _/_ 21 _._ 57 = 148 _._ 04 seconds
or _∼_ **2 minutes** .


15We use this particular function: [https://botorch.readthedocs.io/en/latest/](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.analytic.LogNoisyExpectedImprovement)
[acquisition.html#botorch.acquisition.analytic.LogNoisyExpectedImprovement.](https://botorch.readthedocs.io/en/latest/acquisition.html#botorch.acquisition.analytic.LogNoisyExpectedImprovement)


35


**1890**

**1891**


**1892**

**1893**

**1894**

**1895**

**1896**

**1897**


**1898**

**1899**

**1900**

**1901**

**1902**

**1903**


**1904**

**1905**

**1906**

**1907**

**1908**


**1909**

**1910**

**1911**

**1912**

**1913**

**1914**


**1915**

**1916**

**1917**

**1918**

**1919**

**1920**


**1921**

**1922**

**1923**

**1924**

**1925**

**1926**


**1927**

**1928**

**1929**

**1930**

**1931**


**1932**

**1933**

**1934**

**1935**

**1936**

**1937**


**1938**

**1939**

**1940**

**1941**

**1942**

**1943**


Table 6: Difference between using _BoTorch_ with the noisy _LogEI_ acquisition function, and _hyperopt_ .
The table shows: (a) _hyperopt_ runtimes (in seconds), (b) percentage point (pp) difference between
the %-age improvements seen between _BoTorch_ and _hyperopt_, and (c) the speedup in wallclock
runtime with _BoTorch_ . _BoTorch_ and _hyperopt_ were run for 200 and 3000 iterations respectively.
For these examples, _BoTorch_ runs significantly faster. Aside from one case - _ijcnn1_, _depth_ = 2,the
performance degradation is reasonable. In some cases, it seems to perform better. **Results** **are**
**averaged over three runs.**


dataset tree depth = 1 2 3 4 5


Our technique is applicable even when the model size has more than one attribute. This is because
Algorithm 1 delegates size enforcement to _trainI,g_ . Consider GBMs, where we might consider a
bivariate size, _η_ = [ _max_ ~~_d_~~ _epth, num_ ~~_t_~~ _rees_ ]; here the quantities respectively denote the maximum
depth allowed for each constituent DT in a GBM, and the number of DTs in the GBM. In Figure
9, we show how improvements for GBMs vary when 1 _≤_ _max_ ~~_d_~~ _epth_ _≤_ 5 ( _x_ -axis) and 1 _≤_
_num_ ~~_t_~~ _rees_ _≤_ 5 ( _y_ -axis); the oracle used is a GBM as well (unconstrained in size). Results are
averaged over _three_ runs for these datasets: (a) _senseit-sei_ (b) _higgs_ (c) _cod-rna_ and (d) _senseit-aco_
here. We continue to observe pattern that as model sizes increase, in terms of both _max_ ~~_d_~~ _epth_ and
_num_ ~~_t_~~ _rees_, improvements decrease.


A.13 DIFFERENT FEATURE SPACES


In our validation experiments in §4.1.2, the feature vector representation was identical for the oracle
and the interpretable model. This is also what Algorithm 1 implicitly assumes. Here, we consider
the possibility of going a step further and using different feature vectors. If _fO_ and _fI_ are the
feature vector creation functions for the oracle and the interpretable model respectively, and _xi_ is a
“raw data” instance, then:


1. The oracle is trained on instances _fO_ ( _xi_ ), and provides uncertainties _uO_ ( _fO_ ( _xi_ )).


2. The interpretable model is provided with data _fI_ ( _xi_ ), but the uncertainty scores available
to it are _uO_ ( _fO_ ( _xi_ )).


The motivation for using different feature spaces is that the combination ( _O, fO_ ) may be known to
work well together and/or a pre-trained oracle might be available only for this combination.


We illustrate this application with the example of predicting nationalities from surnames of individuals. Our dataset (Rao & McMahan, 2019) contains examples from 18 nationalities: _Arabic,_
_Chinese,_ _Czech,_ _Dutch,_ _English,_ _French,_ _German,_ _Greek,_ _Irish,_ _Italian,_ _Japanese,_ _Korean,_ _Polish,_
_Portuguese, Russian, Scottish, Spanish, Vietnamese_ . The representations and models are as follows:


1. The oracle model is a _Gated_ _Recurrent_ _Unit_ _(GRU)_ (Cho et al., 2014), that is learned on
the sequence of characters in a surname. The GRU is calibrated with _temperature scaling_
(Guo et al., 2017).


2. The interpretable model is a DT, where the features are character n-grams, _n ∈_ 1 _,_ 2 _,_ 3. The
entire training set is initially scanned to construct an n-gram vocabulary, which is then used
to create a sparse binary vector per surname - 1s and 0s indicating the presence and absence
of an n-gram respectively.


36


timehyp = 3193 _._ 27s,
a1a pp = +0 _._ 19,
speedup = 21 _._ 57x


4221 _._ 34s,
ijcnn1 _−_ 2 _._ 39,
28 _._ 37x


A.12 MULTIVARIATE MODEL SIZES


4228 _._ 81s,
+1 _._ 87
27 _._ 02x


3902 _._ 53s,

_−_ 7 _._ 84
24 _._ 62x


3867 _._ 64s,
+3 _._ 22,
25 _._ 23x


4613 _._ 63s,

_−_ 3 _._ 47,
29 _._ 01x


4843 _._ 54s,

_−_ 0 _._ 43,
34 _._ 50x


4362 _._ 36s,

_−_ 1 _._ 36,
27 _._ 20x


3610 _._ 96s,
+4 _._ 04,
27 _._ 86x


4962 _._ 23s,
+0 _._ 20,
29 _._ 59x


(b) dataset: higgs


1 2 3 4 5
max_depth

(d) dataset: senseit_aco


1 2 3 4 5
max_depth


**1944**

**1945**


**1946**

**1947**

**1948**

**1949**

**1950**

**1951**


**1952**

**1953**

**1954**

**1955**

**1956**

**1957**


**1958**

**1959**

**1960**

**1961**

**1962**


**1963**

**1964**

**1965**

**1966**

**1967**

**1968**


**1969**

**1970**

**1971**

**1972**

**1973**

**1974**


**1975**

**1976**

**1977**

**1978**

**1979**

**1980**


**1981**

**1982**

**1983**

**1984**

**1985**


**1986**

**1987**

**1988**

**1989**

**1990**

**1991**


**1992**

**1993**

**1994**

**1995**

**1996**

**1997**


(a) dataset: senseit_sei


Figure 9: Improvements in test _F_ 1-macro for multiple datasets for different sizes of _GBM_ models
are shown. (a) Top-left: _senseit-sei_ (b) Top-right: _higgs_ (c) Bottom-left: _cod-rna_ and (d) Bottomright: _senseit-aco_ . Here, model size is the combination of _max_ ~~_d_~~ _epth_ and _number_ _of_ _trees_ in the
_GBM_ model. Greater improvements are seen at lower sizes.


Figure 10 shows a schematic of the setup.


The n-gram representation leads to a vocabulary of _∼_ 5000 terms, that is reduced to 600 terms
based on a _χ_ [2] -test in the interest of lower running time. DTs of different _depth_ _≤_ 15 were trained.
A budget of _T_ = 3000 iterations was used, and the relative improvement in the _F_ 1 macro score (as
in Equation 3) is reported, averaged over three runs. Figure 11 shows the results.


We see large improvements at small depths, that peak with _δF_ 1 = 83 _._ 04% at _depth_ = 3, and then
again at slightly larger depths, which peak at _depth_ = 9 with _δF_ 1 = 12 _._ 34%.


To obtain a qualitative idea of the changes in the DT using a oracle produces, we look at the
prediction rules for _Polish_ surnames, when DT _depth_ = 3. For each rule, we also present examples
of true and false positives.


**Baseline rules** - _precision_ = 2 _._ 99% _, recall_ = 85 _._ 71% _, F_ 1 = 5 _._ 77%:


Rule 1. _k ∧_ _ski ∧¬v_


       - True Positives: _jaskolski, rudawski_

       - False Positives: _skipper (English), babutski (Russian)_

Rule 2. _k ∧¬ski ∧¬v_


       - True Positives: _wawrzaszek, koziol_

       - False Positives: _konda (Japanese), jagujinsky (Russian)_


37


70


60


50


40


30


20


10


200


175


150


125


100


75


50


25


1 2 3 4 5
max_depth

(c) dataset: cod-rna


1 2 3 4 5
max_depth


175


150


125


100


75


50


25


120


100


80


60


40


20


**1998**

**1999**


**2000**

**2001**

**2002**

**2003**

**2004**

**2005**


**2006**

**2007**

**2008**

**2009**

**2010**

**2011**


**2012**

**2013**

**2014**

**2015**

**2016**


**2017**

**2018**

**2019**

**2020**

**2021**

**2022**


**2023**

**2024**

**2025**

**2026**

**2027**

**2028**


**2029**

**2030**

**2031**

**2032**

**2033**

**2034**


**2035**

**2036**

**2037**

**2038**

**2039**


**2040**

**2041**

**2042**

**2043**

**2044**

**2045**


**2046**

**2047**

**2048**

**2049**

**2050**

**2051**


Figure 10: The feature representations for the oracle and the interpretable model may be different.
Consider the name “Amy”: the GRU is provided its letters, one at a time, in sequence, while the DT
is given an n-gram representation of the name.


Figure 11: Improvements _δF_ 1 are shown for different depths of the DT.


**Oracle-based DT rules** - _precision_ = 25 _._ 00% _, recall_ = 21 _._ 43% _, F_ 1 = 23 _._ 08%:


Rule 1. _ski ∧¬_ ( _b ∨_ _kin_ )


       - True Positives: _jaskolski, rudawski_

       - False Positives: _skipper (English), aivazovski (Russian)_


We note that the baseline rules are in conflict w.r.t. the literal “ski”, and taken together, they simplify
to _k ∧¬v_ . This makes them extremely permissive, especially _Rule 2_, which requires the literal “k”
while needing “ski” and “v” to be absent. Not surprisingly, these rules have high recall (= 85 _._ 71%)
but poor precision (= 2 _._ 99%), leading to _F_ 1 = 5 _._ 77%.


In the case of the oracle-based DT, now we have only one rule, that requires the atypical trigram
“ski”. This improves precision (= 25%), trading off recall (= 21 _._ 43%), for a significantly improved
_F_ 1 = 23 _._ 08%.


The difference in rules may also be visualized by comparing the distribution of nationalities represented in their false positives, as in Figure 12. We see that the baseline DT rules, especially _Rule 2_,
predict many nationalities, but in the case of the DT learned using the oracle, the model confusion


38


**2052**

**2053**


**2054**

**2055**

**2056**

**2057**

**2058**

**2059**


**2060**

**2061**

**2062**

**2063**

**2064**

**2065**


**2066**

**2067**

**2068**

**2069**

**2070**


**2071**

**2072**

**2073**

**2074**

**2075**

**2076**


**2077**

**2078**

**2079**

**2080**

**2081**

**2082**


**2083**

**2084**

**2085**

**2086**

**2087**

**2088**


**2089**

**2090**

**2091**

**2092**

**2093**


**2094**

**2095**

**2096**

**2097**

**2098**

**2099**


**2100**

**2101**

**2102**

**2103**

**2104**

**2105**


Figure 12: The distribution of nationalities in false positive predictions for the baseline and oracle
based models, shown for predicting _Polish_ names. Only nationalities with non-zero counts are
shown.


is concentrated around _Russian_ names, which is reasonable given the shared _Slavic_ origin of many
_Polish_ and _Russian_ names.


We believe this is a particularly powerful and exciting application of our technique, and opens up a
wide range of possibilities for translating information between models of varied capabilities.


39