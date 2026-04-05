# UNCOVAER: ESTIMATING CAUSAL CONCEPT EFFECTS UNDER VISUAL LATENT CONFOUNDING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Estimating the effect of human-interpretable concepts on model predictions is crucial for explaining and auditing machine learning systems, as well as for mitigating their reliance on spurious correlations. Most existing approaches assume
complete concept annotations, but in practice some concepts may remain unobserved and act as confounders, biasing causal effect estimates. We introduce **Un-**
**CoVAEr** (Unobserved Confounding Variational AutoEncoder), a latent-variable
model that partitions image latent representations into confounder-related and
non-confounding residual components. This allows us to (i) identify which observed concepts are confounded, (ii) obtain corrected unbiased effect estimates
via backdoor adjustment, and (iii) learn confounder-proxy variables that align with
underlying latent factors. On a controlled semi-synthetic MorphoMNIST benchmark, we show that UnCoVAEr yields substantially less biased effect estimates
than prior methods, providing practitioners with a practical tool for trustworthy
concept-level causal inference in partially annotated image datasets.


1 INTRODUCTION


Human-interpretable visual concepts are being increasingly used to explain, audit and control the
behavior of machine learning models. Concept-based explanations enable domain experts to ask
and answer targeted causal questions such as “how much does hippocampal atrophy, as seen on an
MRI, contribute to an Alzheimer’s diagnosis?” (Castro et al., 2020) or “which facial attributes drive
perceived attractiveness in our annotated dataset?” (Lingenfelter et al., 2022). In practice, however,
causal statements at the concept level are only as valid as the assumptions that underlie them. Most
existing concept effect estimators implicitly assume that we have measured all relevant visual factors
that confound concept-label relations. When important factors are missing from the annotation set,
naive observational estimates can be severely biased and lead practitioners to mistaken conclusions
and harmful interventions.


Consider a scenario in medical imaging where an interpretable concept-based model concludes that
hippocampal atrophy is a dominant predictor of Alzheimer’s disease. However, this association may
be inflated by confounders such as scanner hardware or acquisition protocols: different scanners
alter image appearance in ways that affect how atrophy is manifested and measured, while also
correlating with hospital site and diagnostic practices that influence the diagnosis. As a result,
a naive estimate of the effect of hippocampal atrophy on diagnosis may capture site- or devicespecific artifacts rather than a genuine biological causal effect, potentially misleading auditors and
downstream clinical decisions. Next, consider a facial-attribute dataset where a set of annotated
concepts such as smiling, makeup or age are used to predict attractiveness. Yet, unannotated factors
like skin tone, lighting, or demographic imbalance in the dataset can act as confounders, inflating
the estimated influence of certain attributes and masking annotator prejudice (Lingenfelter et al.,
2022). Obtaining unbiased causal effect estimates enables practitioners and researchers to estimate
bias in datasets (Madras et al., 2019; Di Stefano et al., 2020), decide if they need to collect additional
metadata or reweight training examples (Zhao et al., 2023). Moreover, obtaining corrected causal
effect estimates can be useful to improve the performance of domain generalization methods by
penalizing reliance on spurious attributes (Kumar et al., 2023).


Concept-based models are not new: they can be understood as a principled evolution of feature
engineering in which model decisions are expressed in terms of semantically meaningful factors


1


(a)


(c)


Figure 1: **(a)** The causal graph shows a case where one of our observed concepts _C_ ( _intensity_ ) and
our label of interest _Y_ are caused by an unobserved confounder ( _thickness_ ). **(b)** As a result, when
computing the effect of _intensity_ on _Y_ without taking this confounding into account, we get a wrong
estimate. **(c)** Our method _Unobserved_ _Confounding_ _Variational_ _AutoEncoder_ _(UnCoVAEr)_ estimates the correct causal effects and finds which observed concepts are confounded. After training
our model, we provide _C_, _Y_, and the image _X_ to the encoder, which outputs a confounder-related
latent _ZC_ and a non-confounding residual _ZS_ used only for image reconstruction. We then perform
backdoor-adjustment using the learned proxy _ZC_ to debias the Average Treatment Effect (ATE) estimation. Additionally, we can intervene on the confounder proxy and interpret its effect.


rather than opaque input dimensions. Early work assumed a fixed, complete set of predefined concepts (Koh et al., 2020); subsequent methods allow concepts to be learned post-hoc or discovered
from images (Yuksekgonul et al., 2023; Oikarinen et al., 2023; Sawada & Nakamura, 2022; Shang
et al., 2024; Rao et al., 2024), and a growing body of research takes a causal formulation on concept
models Dominici et al. (2025b;a); Moreira et al. (2024). Separately, a line of work that builds on
proximal causal inference (Tchetgen et al., 2020) studies causal question in the presence of latent
confounders that manifest through proxies, estimating treatment effects originally in tabular settings
(Louizos et al., 2017; Wu & Fukumizu, 2022; Zhang et al., 2020; Miao et al., 2018; Wang & Blei,
2021) and extending to high-dimensional data such as images (Kaddour et al., 2021; Kompa et al.,
2022; Israel et al., 2023; Jerzak et al., 2023; Schulte et al., 2025). Our work sits at the intersection
of these two threads: we incorporate a deep latent-variable method from the proximal causal inference literature to robustly estimate causal quantities when unobserved visual concepts confound
both observed concepts and the label.


We propose **UnCoVAEr**, a latent-variable model inspired by Causal Effect Variational AutoEncoder (CEVAE) that explicitly accounts for concept incompleteness by partitioning the latent space
into two parts: a confounder-related component that explains variation shared between concepts and
label, and a non-confounding block that captures residual image variation. This structured decom

2


(b)


position lets us (i) recover proxies for the confounders, (ii) use them to estimate concept effects
via backdoor-adjustment, and (iii) identify which observed concepts are substantially confounded
by latent visual factors. We validate our approach on a controlled semi-synthetic Morpho-MNIST
benchmark. UnCoVAEr reduces bias in concept-effect estimates compared to prior concept-based
and latent-variable baselines.


Our contributions are as follows: (1) we formalize concept incompleteness as a latent confounding
problem in image datasets and introduce partitioned-latent representations as an effective inductive
bias; (2) we propose a principled criterion to distinguish confounded from unconfounded concepts
and correct their effect estimates via backdoor adjustment; and (3) we provide empirical evidence
that UnCoVAEr reduces bias in causal effect estimates compared to strong baselines and learns
proxy variables that correlate with underlying latent confounders on a controlled semi-synthetic
MorphoMNIST dataset.


2 RELATED WORK


**Latent-variable** **proximal** **causal** **inference** UnCoVAEr builds on a line of work that utilizes
deep latent-variable models to estimate causal quantities like ATE in the presence of unobserved
confounders. CEVAE (Louizos et al., 2017) assumes a causal graph where latent confounders are
also causes of observed proxies and uses a Variational AutoEncoder (VAE) formulation to model the
data-generating process and estimate ATE with backdoor adjustment. While CEVAE has demonstrated promising empirical performance, its reliance on variational inference raises concerns about
identifiability. Rissanen & Marttinen (2021) provide an extensive critique, showing analytically and
experimentally that CEVAE can fail when the latent space is misspecified or when the data distribution is complex, despite working in simple synthetic setups, while they also provide simple
experiments on digit images. Follow-up works provide identification under limited overlap assumptions (Wu & Fukumizu, 2022) and disentangle instrumental, risk, and confounding factors to better
isolate causal effects (Zhang et al., 2020), while Madras et al. (2019) utilize CEVAE to improve
causal effect estimates between sensitive attributes and outcome in a fairness setting. At the same
time, proximal causal inference literature (Tchetgen et al., 2020) and related proxy-variable identification results provide formal conditions (completeness / rank) under which proxies identify causal
effects (Miao et al., 2018; Wang & Blei, 2021). A number of recent works tackles causal effect
estimation assuming that images or image-derived features act as proxies for latent confounders.
Some approaches apply standard adjustment ideas by learning a model that extracts confounding information from the image via propensity score matching (Jerzak et al., 2023) or by extracting image
features (Xu et al., 2021; Schulte et al., 2025). Others develop neural methods that directly learn the
necessary adjustment functions from high-dimensional proxies (Kompa et al., 2022). Kumar et al.
(2023) also use the image directly to perform backdoor adjustment and use the estimated causal
effects of the attributes to regularize classifiers for domain generalization.


**Concept-based** **explanations** Concept-based explanation methods such as TCAV (Kim et al.,
2018) or Concept Bottleneck Model (CBM) (Koh et al., 2020) treat concepts as interpretable primitives for explaining image classification, enabling interventions and attributing predictions directly
to concepts. Follow-up work has extended CBMs to incorporate concepts not predefined in the
concept set (Yuksekgonul et al., 2023; Oikarinen et al., 2023; Sawada & Nakamura, 2022; Shang
et al., 2024; Rao et al., 2024), while also revealing important limitations, such as whether the learned
concepts truly correspond to human-understandable semantics or instead capture spurious shortcuts
(Mahinpei et al., 2021; Margeloiu et al., 2021; Havasi et al., 2022). In this line of work, Bahadori &
Heckerman (2021) address biases in concept-based explanations arising from confounding information. They propose a two-stage regression technique, inspired by instrumental variable methods, to
remove the impact of confounders and noise. Their approach also considers the completeness of the
concept set (Yeh et al., 2020), demonstrating effectiveness even when the set is incomplete. Goyal
et al. (2020) introduce the notion of _CaCE (Causal Concept Effect)_, defining it as the effect of the
presence or absence of a human-interpretable concept on a deep neural network’s prediction. They
train a conditional VAE to generate counterfactuals by disentangling and intervening on the concept
of interest. While they highlight the importance of causality for concept explanations, they rely
on the assumption that unobserved confounders do not significantly impact the observed concepts.
Gao & Chen (2024) explicitly tackle concept incompleteness by constructing pseudo-concepts or

3


thogonal to the observed ones and using a linear predictor to capture residual bias. However, their
orthogonality assumption does not allow for confounding.


Figure 2: Our causal graph assumption. We assume that the image _X_ is jointly caused by a set
of observed concepts _C_, the unobserved confounder _ZC_ and some _ZS_ (e.g. writing style, point of
view) that is irrelevant for _C_ and the outcome _Y_ . _C_ cause _Y_, while the unobserved confounder _ZC_
causes both _C_ and _Y_ .


3 PRELIMINARIES AND PROBLEM SETUP


We observe i.i.d. samples ( _X, C, Y_ ) _∼D_, where _X_ _∈X_ is an image, _C_ = ( _C_ 1 _, . . ., CM_ ) _∈_
_{_ 0 _,_ 1 _}_ _[M]_ are observed binary concept annotations, and _Y_ _∈{_ 0 _,_ 1 _}_ is a binary outcome of interest.
We are interested in quantifying how changes to a single concept _Ci_ causally affect _Y_ . Two common estimands for this are the individual (or conditional) treatment effect (ITE) and the population
Average Treatment Effect (ATE):

ITE _i_ ( _x_ ) := E� _Y_ _| do_ ( _Ci_ = 1) _, X_ = _x_     - _−_ E� _Y_ _| do_ ( _Ci_ = 0) _, X_ = _x_     
(1)
ATE _i_ = E� _Y_ _| do_ ( _Ci_ = 1)     - _−_ E� _Y_ _| do_ ( _Ci_ = 0)     - _._


Concept-based explanation methods such as CBMs (Koh et al., 2020) perform _concept-interventions_
by editing intermediate concept values and re-evaluating the outcome classifier. These types of
interventions and the treatment effects they yield are similar to those obtained by meta-learners in
the causal inference literature (K¨unzel et al., 2019). For example, the S-learner, which is one of the
simplest methods for treatment effect estimation, performs the same operation: it fits a predictor
of _Y_ given covariates and treatment (concepts in this case) and then estimates treatment effects by
changing the treatment value in the input. These estimators coincide with the _do_ -intervention only
under the ignorability assumption (no unobserved confounding) and can work well enough only in
this setting. However, in the presence of unobserved confounders, the estimates will be biased.


**Confounding** **and** **backdoor** **adjustment** Let _V_ denote an observed variable that jointly causes
some observed concepts _C_ and the outcome _Y_ . If _V_ blocks all backdoor paths from _Ci_ to _Y_, the
interventional mean is given by the backdoor formula:


                E[ _Y_ _| do_ ( _Ci_ = _c_ )] = E[ _Y_ _| Ci_ = _c, V_ = _v_ ] _p_ ( _v_ ) _dv._ (2)

_V_

Thus, if _V_ is observed, then Eq. 2 gives an unbiased estimand for the ATE as an immediate consequence of the backdoor-criterion Pearl (1993).


**Proxy learning from images** In our setup, instead of an observed _V_, we have a set of unobserved
variables _U_ . We assume that the unobserved confounders of interest manifest themselves in the
image _X_ (e.g. scanner type, lighting or a separate attribute that affects both annotations and labels).
Thus _X_ serves as a high-dimensional proxy for _U_ . Because _X_ also contains many features irrelevant
to the causal relation between _C_ and _Y_, we aim to learn a lower-dimensional proxy latent _ZC_ from
( _X, C, Y_ ) with the following operational properties:


4


(P1) _Adjustment sufficiency:_ conditioning on _ZC_ blocks the backdoor paths between _Ci_ and _Y_
(so _ZC_ plays the role of _V_ in Eq. 2);
(P2) _Parsimony:_ _ZC_ is low-dimensional and amenable to downstream estimation and marginalization.

(P3) _Interpretability:_ We model _ZC_ as a binary variable to fit well in our setup of binary concepts and outcomes


We assume the causal structure shown in Figure 2. As we want _ZC_ to contain only the confounding
information for adjustment, we decompose the image into two parts: a discrete confounder-specific
latent _ZC_ _∈{_ 0 _,_ 1 _}_ _[K]_ and a continuous residual latent _ZS_ _∈_ R _[d]_ . Intuitively, _ZC_ captures the visual
variation that (partially) explains both some observed concepts _C_ and the label _Y_, while _ZS_ contains
the remaining image variation that is irrelevant for the causal relationship between _C_ and _Y_, but
necessary to model _X_ accurately.


Under (P1), we have:

_p_ ( _Y_ _| X, do_ ( _Ci_ = _c_ )) =       - _p_ ( _Y_ _| X, Ci_ = _c, ZC_ ) _p_ ( _ZC_ _| X_ ) _._ (3)


_ZC_

Thus, to estimate ITEi(x) = E� _Y_ _|_ _X_ = _x, do_ ( _Ci_ = _c_ )� and ATEi = E� _ITEi_ ( _x_ )� we need to
learn the conditional _p_ ( _Y_ _| X, C, ZC_ ) [1] and the posterior _p_ ( _ZC_ _| X_ ).


**Identification** **requirements** **and** **limitations** Identification of causal effects from observational
( _X, C, Y_ ) rests on standard proxy-type and overlap assumptions. First, the observed image _X_ must
carry _sufficient_ _proxy_ _signal_ of the unobserved confounder: if the confounder leaves no detectable
trace in pixels, then no method can recover its effect. Second, the support of relevant predictive
distributions must overlap (positivity), so that the requisite conditional expectations are well defined.
Additionally, we assume that there are no _unobserved colliders_ : no latent variables caused by both
the outcome _Y_ and a concept _Ci_ . If such a collider exists, conditioning on it opens a spurious path.
Under these conditions, and assuming a sufficiently expressive latent-variable model, our adjusted
estimator using _ZC_ is consistent in principle. We offer three clarifications: (i) we do not require
recovery of the true confounder, only that the learned proxy _ZC_ suffices for valid adjustment; (ii)
if the confounder leaves no observable imprint, identification is impossible for _any_ method; and
(iii) while practical estimation is subject to approximation error, our results show that UnCoVAEr
recovers unbiased ATEs whenever these identification assumptions hold.


4 UNCOVAER: UNOBSERVED CONFOUNDING VARIATIONAL
AUTOENCODER


We now introduce _UnCoVAEr_, a variational autoencoder designed to recover causal concept effects
under unobserved confounding.


**Generative** **model** The assumed causal graph of Figure 2 leads to the following factorization of
the joint distribution:
_p_ ( _X, C, Y, ZC, ZS_ ) = _p_ ( _ZC_ ) _p_ ( _ZS_ ) _p_ ( _C_ _| ZC_ ) _p_ ( _X_ _| C, ZC, ZS_ ) _p_ ( _Y_ _| C, ZC_ ) _._ (4)


Our model parameterizes three decoders: _pθx_ ( _X_ _|_ _C, ZC, ZS_ ), _pθc_ ( _C_ _|_ _ZC_ ), and _pθy_ ( _Y_ _|_ _C, ZC_ ).
Because the exact posterior _p_ ( _ZC, ZS_ _| X, C, Y_ ) is intractable, we introduce a variational encoder:
_qϕc,ϕs_ ( _ZC, ZS_ _| X, C, Y_ ) = _qϕc_ ( _ZC_ _| X, C, Y_ ) _qϕs_ ( _ZS_ _| X, C, Y_ ) _._
We implement a shared backbone with separate output heads: logits for the discrete confounderrelated latent _ZC_ (parameters _ϕc_ ) and Gaussian parameters ( _µϕs_ _, σϕs_ ) for the continuous residual
latent _ZS_ . _ZC_ is sampled with the Gumbel–Softmax relaxation during training (Jang et al., 2017). To
reduce information leakage between the two blocks, we additionally include a mutual-information
regularizer that minimizes MI( _ZC, ZS_ ) using the CLUB estimator with parameters _ψ_ (Cheng et al.,
2020). This encourages _ZC_ to capture confounder-related variation distinct from the residual information in _ZS_ .


1Note that under the assumed causal graph of Figure 2 _Y_ is independent of _X_ given _C_ and _ZC_, so _p_ ( _Y_ _|_
_C, ZC_ ) suffices.


5


**Training objective** We maximize the evidence lower bound (ELBO):
_L_ ELBO = E _qϕc,ϕs_ ( _ZC_ _,ZS_ _|X,C,Y_ )� log _pθx_ ( _X_ _| C, ZC, ZS_ ) + log _pθc_ ( _C_ _| ZC_ ) + log _pθy_ ( _Y_ _| C, ZC_ )�

_−_ KL� _qϕc_ ( _ZC_ _| X, C, Y_ ) _∥_ _p_ ( _ZC_ )� _−_ KL� _qϕs_ ( _ZS_ _| X, C, Y_ ) _∥_ _p_ ( _ZS_ )� _,_ (5)

using the following priors: _p_ ( _ZC_ ) = [�] _j_ _[K]_ =1 [Bern(] _[π]_ [= 0] _[.]_ [5)][ and] _[ p]_ [(] _[Z][S]_ [) =][ �] _[d]_ _j_ =1 _[N]_ - _ZSj_ _|_ 0 _,_ 1�.

We augment _L_ ELBO with two auxiliary discriminative losses, implemented as small classification
heads (following Louizos et al. (2017)):
_L_ aux _,C_ = _−_ E _p_ data( _x,c_ )� log _qξC_ ( _C_ _| X_ )� _,_ _L_ aux _,Y_ = _−_ E _p_ data( _x,c,y_ )� log _qξY_ ( _Y_ _| X, C_ )� _._


The auxiliary losses serve two roles: (i) they are used during inference, providing predictors for _C_
and _Y_, and (ii) they encourage representations that capture task-relevant information, sharpening the
posterior and improving the quality of learned _ZC_ as a confounder proxy. Adding the CLUB-based
mutual-information estimate _L_ MI = MI [�] _ψ_ ( _ZC, ZS_ ), The overall training objective is therefore
_L_ train = _−L_ ELBO + _λCL_ aux _,C_ + _λY L_ aux _,Y_ + _λ_ MI _L_ MI _,_ (6)
where _λC, λY, λ_ MI _≥_ 0 balance the auxiliary and independence terms. In our experiments, setting
( _λC, λY, λ_ MI) = (1 _._ 0 _,_ 1 _._ 0 _,_ 0 _._ 1) yielded the best performance.


The auxiliary _qξC_ ( _C_ _|_ _X_ ) is analogous to the concept layer in concept-bottleneck models (it provides an image-to-concept mapping), while _qξY_ ( _Y_ _|_ _X, C_ ) functions similar to an outcome layer
with a residual connection (Yuksekgonul et al., 2023).


While the original CEVAE utilizes a TARNET-style architecture (Shalit et al., 2017) that fits separate
outcome heads per treatment, our model shares decoders and conditions on _C_, since the networks
would scale exponentially with the number of concepts. In our experiments we also explore a variant
that allocates an independent discrete latent _ZCi_ for each concept _Ci_ (i.e., separate confounder
proxies per concept). This allows us to estimate confounder proxies separately per concept and
better interpret their relation with the observed proxies.


We use KL-annealing for the latent KL terms (gradually increasing their weight from 0 to 1 during early epochs) to avoid posterior collapse (Bowman et al., 2016) and temperature annealing for
the Gumbel-Softmax relaxation of _ZC_ (start at _τ_ 0 and reduce to _τ_ min) to transition from smooth
relaxation to near-discrete samples (Jang et al., 2017).


**ATE** **estimation** After training, we estimate interventional means by marginalizing over the aggregated posterior of the confounder-latent _ZC_ . Concretely, for each test image _x_ we draw samples
( _c, y, z_ ) with _c_ _∼_ _qξC_ ( _C_ _|_ _x_ ), _y_ _∼_ _qξY_ ( _Y_ _|_ _x, c_ ), and _z_ _∼_ _qϕc_ ( _ZC_ _|_ _x, c, y_ ). This yields approximate draws from _qϕc_ ( _ZC_ _|_ _x_ ), analogous to the marginalization strategy in Louizos et al. (2017).
For each target concept _Ci_, we intervene by setting it to _c_ _∈{_ 0 _,_ 1 _}_ while leaving the remaining
concepts _C−i_ at their sampled values, and evaluate


where _M_ (100 in our experiments) is the number of posterior samples per image. For each _Ci_,
the ATE is the difference in predicted outcomes under interventions _Ci_ = 1 and _Ci_ = 0. To detect confounding, we compare the above ATE with the estimated difference in conditional means:
ATEnaive = E[ _Y_ _|_ _Ci_ = 1] _−_ E[ _Y_ _|_ _Ci_ = 0], which would . We flag a concept as confounded
when the computed ATE significantly and systematically differs from ATEnaive. For this, we employ a bootstrap test, in which we resample and recompute ATEs per batch and flag a concept as
confounded if the 95% confidence intervals of the ATE do not overlap.


5 EXPERIMENTAL SETUP


5.1 DATASET


We evaluate UnCoVAEr on a controlled semi-synthetic benchmark derived from Morpho-MNIST
(Castro et al., 2019), where digit images are systematically modified along interpretable morphological axes. All experiments use 5 random seeds; for each seed we select a different digit class (0–4).


6


_M_

- 
E _θy_ _Y_ _| Ci_ = _c,_ _C−i_ = _c_ [(] _−_ _[m]_ _i_ [)] _[,]_ _[Z][C]_ [=] _[ z]_ [(] _[m]_ [)][�] _,_ (7)
_m_ =1


E�[ _Y_ _| do_ ( _Ci_ = _c_ )] _≈_ _N_ [1]


_N_


_n_ =1


1

_M_


This design minimizes variation due to digit identity and isolates causal effects arising purely from
morphology.


We focus on four pixel-level morphological attributes as binary concepts: _thickness_, _intensity_, _slant_,
and _width_ . Continuous values for each concept are sampled conditionally from _N_ (0 _._ 25 _,_ 0 _._ 01) when
_Ci_ = 0 and _N_ (0 _._ 75 _,_ 0 _._ 01) when _Ci_ = 1. The values are then scaled according to the attribute.


The outcome _Y_ is a synthetic label constructed as a logical rule over the concepts:


_Y_ = **1** _{_ thickness + slant + width _≥_ 2 _},_


i.e., _Y_ = 1 if at least two of these three concepts are active, with _intensity_ not causing _Y_ .


We design three dataset variants to probe distinct confounding patterns:


(i) **Single confounder:** observed concepts: _{intensity, slant, width}_ ; unobserved: _{thickness}_ .
Thickness causally influences intensity, making it the only observed concept affected by an
unobserved confounder.


(ii) **Common confounder:** observed: _{_
emphintensity, slant, width _}_ ; unobserved: _{thickness}_ . Here thickness jointly drives both
intensity and slant, acting as a shared confounder across multiple observed concepts.


(iii) **Multiple confounders:** observed: _{intensity, width}_ ; unobserved: _{thickness, slant}_ . Both
thickness and slant affect intensity through a non-linear XOR causal mechanism, so a single
observed concept is influenced by two distinct unobserved confounders.


Across all variants we control the _confounding strength α_ . For a causal link _Ci_ _→_ _Cj_, the label of
_Cj_ is set equal to _Ci_ with probability _α_ . We evaluate under two regimes: an _in-distribution_ (ID)
test set with strong confounding ( _α_ = 0 _._ 9) and an _out-of-distribution_ (OOD) test set with much
weaker confounding ( _α_ = 0 _._ 6), enabling us to assess robustness of ATE estimation under shifts in
the confounding mechanism.


Lastly, to ensure that our method correctly adjusts for observed confounders, we construct an additional experimental setup by modifying the _Multiple confounders_ variant. We assume that thickness
is now observed (slant remains unobserved).


5.2 BASELINES AND ABLATIONS


We benchmark UnCoVAEr against latent-variable, concept-based, and feature-adjustment methods.


**CEVAE** (Louizos et al., 2017) is adapted for the image domain with convolutional encoders/decoders. Its difference from our method is that it does not partition the latent space but
uses a single continuous latent.


**CaCE** (Goyal et al., 2020) estimates causal concept effects via counterfactual generation. Its original
formulation refers to estimating effect on a classifier, rather than the true causal effect. For fair
comparison we use the same architecture for encoder/decoder and we train an auxiliary predictor
_qξ_ ( _Y_ _| X_ ), which we use to assess change in outcome.


**Image-adjustment** (Jerzak et al., 2023) conditions directly on image embeddings by fitting a
propensity score model _e_ ˆ _i_ ( _x_ ) _≈_ _p_ ( _Ci_ = 1 _|_ _X_ = _x_ ) and applying inverse-probability weighting (IPW) to estimate _E_ [ _Y_ _| do_ ( _Ci_ )].


**Concept Bottleneck Model (CBM)** (Koh et al., 2020) predicts _Y_ through an intermediate concept
layer _C_ [ˆ] = _f_ ( _X_ ) and enables interventions by editing _C_ [ˆ] _i_ .


**Residual CBM (Res-CBM)** augments standard CBM by explicitly modeling variation unexplained
by observed concepts. Predictions are of the form _Y_ [ˆ] = _g_ ( _C, r_ [ˆ] ( _X_ )), where _r_ ( _X_ ) is a residual representation. During training the concept layer remains fixed and we discretize _r_ ( _X_ ) with GumbelSoftmax. To estimate causal effects, we use IPW with ˆ _ei_ ( _Ci_ _| r_ ( _X_ )).


Finally, we include two meta-learners in the style of S-learners: (i) a **Naive** **Estimator**, which
conditions only on _C_ (biased under unobserved confounding); and (ii) an **Oracle Estimator**, which
additionally conditions on the true latent confounder(s), providing an empirical upper bound.


7


To assess the contribution of each component of UnCoVAEr, we perform the following ablations:
(i) removing the image reconstruction term _pθx_ ( _X_ _|_ _C, ZC, ZS_ ), (ii) using only a shared discrete
latent _ZC_, (iii) the default model with shared _ZC_ and residual _ZS_ ; and (iv) a variant with separate
per-concept confounder proxies _ZCi_ .


6 RESULTS


Table 1 reports ATE estimation error across methods, datasets, and test regimes. Several consistent
patterns emerge. First, in the _single_ _confounder_ setting, UnCoVAEr substantially outperforms all
baselines, apart from the oracle which has access to the true confounder. The closest competitor
is CEVAE, which itself can be seen as a restricted instance of our model using only a continuous
latent.


Second, in the _common confounder_ scenario, UnCoVAEr again improves upon feature-adjustment
and CBM-based approaches. Interestingly, CaCE performs competitively here. Counterfactual concept editing remains effective when a single latent factor drives multiple observed pathways. Nevertheless, UnCoVAEr maintains strong performance, especially in-distribution.


Third, the _multiple_ _confounders_ variant exposes an interesting case. Since intensity is caused by
the logical XOR of two latent factors, naive conditioning and CBMs manage to directly learn and
exploit the _intenisty_ - _Y_ relation without accounting for the unobserved confounder at all, performing
unexpectedly close to the oracle. Image-based methods, by contrast, are misled by this non-linear
dependence. Among them, UnCoVAEr provides the lowest error, though the per-concept _ZCi_ variant
proves unstable in this regime. This suggests that while our structured latent partition is generally
robust, learning disentangled proxies remains challenging under interacting confounders.


Finally, across all scenarios, UnCoVAEr shows improved out-of-distribution robustness: errors remain consistently lower than baselines when the strength of confounding shifts from _α_ = 0 _._ 9 to
_α_ = 0 _._ 6. This supports our central claim that learning an explicit confounder proxy yields more
stable causal effect estimates under distributional change.


**Ablations and Diagnostics** Table 1 further shows the effectiveness of the partitioned latent space
design of UnCoVAEr. When the same bottleneck latent is used for reconstruction and for recovering
the confounder proxy, the method underperforms. Moreover, our hypothesis that the proxy should
guide image reconstruction is validated, as is evident in the performance drop. Finally, per-concept
latents _ZCi_ provide marginal gains, but become unstable under complex confounding. Figure 3
indicates that our confounding-detection criterion is generally effective, especially in the common
confounder scenario, where it correctly characterizes both observed concepts. We report occasional
false positives in the single case and a significant deterioration in the multiple-confounder case,
where the naive estimators approximate the true ATE more closely than our estimands.


Table 1: Mean ATE estimation error (MAE, lower is better) across methods, datasets, and test
regimes (averaged across concepts). Results are reported as mean ± std over 5 seeds. ID: indistribution test set ( _α_ = 0 _._ 9); OOD: out-of-distribution test set ( _α_ = 0 _._ 6). Best non-oracle baseline
per column is in **bold** .


Single confounder Common confounder Multiple confounders


Method ID OOD ID OOD ID OOD


Naive .131 _±_ .18 .135 _±_ .19 .163 _±_ .11 .213 _±_ .04 _._ _± ._ **01** _._ _± ._ **01**
Oracle .003 _±_ .01 .002 _±_ .00 .002 _±_ .01 .009 _±_ .02 .001 _±_ .01 .001 _±_ .01


Image-adjustment .168 _±_ .16 .133 _±_ .18 .440 _±_ .24 .183 _±_ .14 .109 _±_ .14 .117 _±_ .15
CBM .136 _±_ .18 .136 _±_ .18 .163 _±_ .11 .214 _±_ .04 .011 _±_ .01 .012 _±_ .01
Res-CBM .331 _±_ .20 .418 _±_ .09 .171 _±_ .17 .560 _±_ .40 .287 _±_ .24 .253 _±_ .23
CaCE .114 _±_ .14 .087 _±_ .09 .058 _±_ .05 _._ _± ._ **03** .157 _±_ .13 .166 _±_ .07
CEVAE .058 _±_ .06 .049 _±_ .05 .079 _±_ .08 .112 _±_ .05 .106 _±_ .09 .096 _±_ .06


UnCoVAEr (no _p_ ( _X_ )) .113 _±_ .07 .112 _±_ .06 .064 _±_ .06 .119 _±_ .07 .210 _±_ .14 .209 _±_ .14
UnCoVAEr (only _ZC_ ) .070 _±_ .11 .089 _±_ .11 .098 _±_ .10 .172 _±_ .08 .080 _±_ .10 .077 _±_ .10
UnCoVAEr ( _ZS_ + _ZC_ ) _._ _± ._ **04** .040 _±_ .03 .055 _±_ .07 .097 _±_ .06 .070 _±_ .05 .065 _±_ .05
UnCoVAEr ( _ZS_ + _ZCi/Ci_ ) .041 _±_ .04 _._ _± ._ **04** _._ _± ._ **03** .105 _±_ .04 .136 _±_ .11 .138 _±_ .09


8


Figure 3: Empirical rate at which each concept was detected as confounded across random seeds
( _confounding_ _detection_ _rate_ ). Results are shown for all three MorphoMNIST variants, plus the
multiple-confounders setting where one of the confounders (thickness) is observed.


7 DISCUSSION


Our work addresses a critical gap in concept-based model interpretation: the presence of unobserved
visual confounders that bias causal effect estimates. While concept-based methods have gained
traction for their interpretability, our results demonstrate that ignoring latent confounding can lead
to substantially biased conclusions about which concepts truly drive model predictions.


Our experiments reveal interesting nuances in different confounding scenarios. While UnCoVAEr
excels with single or shared confounders, performance degrades when confounders interact in nonlinear ways (e.g., XOR). In such cases, direct statistical associations remain easier to capture than
the underlying more complex causal structure, and all tested image-based methods fail. Handling
complex, interacting confounders remains an open challenge requiring further methodological development. Still, the OOD evaluations are encouraging: robustness to shifts in confounding strength
indicates that _ZC_ captures meaningful causal signals rather than overfitting correlations. This robustness is essential for real-world applications where confounding patterns may vary across datasets or
deployment contexts.


**Limitations** **and** **Future** **Work** UnCoVAEr’s primary limitation is its reliance on the assumption
that confounders manifest visually in the image. Our experiments also highlight that complex causal
structures or interactions remain challenging for current latent-variable approaches. However, the
most critical challenge—and the most important direction for future work—is validating UnCoVAEr on complex real-world datasets. The true test of our model’s practical utility lies in its ability to
perform robustly in settings like medical imaging or model auditing, where concepts interact in unpredictable ways and the ground-truth confounding variables are fundamentally unknown. Successfully demonstrating effectiveness in these noisy, high-stakes environments is essential for moving
from a theoretical proof of concept to a reliable tool for causal interpretability in applied domains.


8 CONCLUSIONS


We introduced UnCoVAEr, a deep latent-variable model for estimating causal concept effects under
visual latent confounding. By partitioning the latent space into confounder-related and residual components, our method recovers proxy variables that enable valid backdoor adjustment even when key
visual concepts remain unannotated. On controlled benchmarks, UnCoVAEr substantially reduces
bias in causal effect estimates compared to existing concept-based and latent-variable approaches,
while maintaining robustness under distribution shift. Our work highlights a critical consideration for practitioners that rely on concept-based explanations: incomplete concept annotations can
severely bias causal conclusions. UnCoVAEr provides a practical tool for detecting and correcting
such biases, enabling more trustworthy concept-level causal inference in partially annotated image
datasets.


9


REFERENCES


Mohammad Taha Bahadori and David Heckerman. Debiasing concept-based explanations with
causal analysis. In _International Conference on Learning Representations_, 2021. [URL https:](https://openreview.net/forum?id=6puUoArESGp)
[//openreview.net/forum?id=6puUoArESGp.](https://openreview.net/forum?id=6puUoArESGp)


Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio.
Generating sentences from a continuous space. In _Proceedings_ _of_ _the_ _20th_ _SIGNLL_ _conference_
_on computational natural language learning_, pp. 10–21, 2016.


Daniel C Castro, Jeremy Tan, Bernhard Kainz, Ender Konukoglu, and Ben Glocker. Morpho-mnist:
Quantitative assessment and diagnostics for representation learning. _Journal of Machine Learning_
_Research_, 20(178):1–29, 2019.


Daniel C Castro, Ian Walker, and Ben Glocker. Causality matters in medical imaging. _Nature_
_Communications_, 11(1):3673, 2020.


Pengyu Cheng, Zhijie Hao, Wenyue Dai, Xinyuan Liu, and Weinan Chen. Club: A contrastive
log-ratio upper bound of mutual information. In _International Conference on Machine Learning_
_(ICML)_, 2020.


Pietro G Di Stefano, James M Hickey, and Vlasios Vasileiou. Counterfactual fairness: removing
direct effects through regularization. _arXiv preprint arXiv:2002.10774_, 2020.


Gabriele Dominici, Pietro Barbiero, Francesco Giannini, Martin Gjoreski, Giuseppe Marra, and
Marc Langheinrich. Counterfactual concept bottleneck models. In _The_ _Thirteenth_ _Interna-_
_tional Conference on Learning Representations_, 2025a. [URL https://openreview.net/](https://openreview.net/forum?id=w7pMjyjsKN)
[forum?id=w7pMjyjsKN.](https://openreview.net/forum?id=w7pMjyjsKN)


Gabriele Dominici, Pietro Barbiero, Mateo Espinosa Zarlenga, Alberto Termine, Martin Gjoreski,
Giuseppe Marra, and Marc Langheinrich. Causal concept graph models: Beyond causal opacity in
deep learning. In _The Thirteenth International Conference on Learning Representations_, 2025b.
[URL https://openreview.net/forum?id=lmKJ1b6PaL.](https://openreview.net/forum?id=lmKJ1b6PaL)


Jifan Gao and Guanhua Chen. Mcce: Missingness-aware causal concept explainer, 2024. URL
[https://arxiv.org/abs/2411.09639.](https://arxiv.org/abs/2411.09639)


Yash Goyal, Amir Feder, Uri Shalit, and Been Kim. Explaining classifiers with causal concept effect
(cace), 2020. [URL https://arxiv.org/abs/1907.07165.](https://arxiv.org/abs/1907.07165)


Marton Havasi, Sonali Parbhoo, and Finale Doshi-Velez. Addressing leakage in concept bottleneck
models. _Advances in Neural Information Processing Systems_, 35:23386–23397, 2022.


Daniel Israel, Aditya Grover, and Guy Van den Broeck. High dimensional causal inference with
variational backdoor adjustment, 2023. [URL https://arxiv.org/abs/2310.06100.](https://arxiv.org/abs/2310.06100)


Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In
_International Conference on Learning Representations_, 2017. [URL https://openreview.](https://openreview.net/forum?id=rkE3y85ee)
[net/forum?id=rkE3y85ee.](https://openreview.net/forum?id=rkE3y85ee)


Connor T. Jerzak, Fredrik Johansson, and Adel Daoud. Estimating causal effects under image confounding bias with an application to poverty in africa, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2206.06410)
[abs/2206.06410.](https://arxiv.org/abs/2206.06410)


Jean Kaddour, Yuchen Zhu, Qi Liu, Matt Kusner, and Ricardo Silva. Causal effect inference for
structured treatments. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.),
_Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 2021. URL [https://openreview.](https://openreview.net/forum?id=0v9EPJGc10)
[net/forum?id=0v9EPJGc10.](https://openreview.net/forum?id=0v9EPJGc10)


Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory
sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation
vectors (TCAV). In Jennifer Dy and Andreas Krause (eds.), _Proceedings of the 35th International_
_Conference on Machine Learning_, volume 80 of _Proceedings of Machine Learning Research_, pp.
2668–2677. PMLR, 10–15 Jul 2018. URL [https://proceedings.mlr.press/v80/](https://proceedings.mlr.press/v80/kim18d.html)
[kim18d.html.](https://proceedings.mlr.press/v80/kim18d.html)


10


Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and
Percy Liang. Concept bottleneck models. In _International conference on machine learning_, pp.
5338–5348. PMLR, 2020.


Benjamin Kompa, David Bellamy, Tom Kolokotrones, james m robins, and Andrew Beam. Deep
learning methods for proximal inference via maximum moment restriction. In S. Koyejo,
S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), _Advances_ _in_ _Neu-_
_ral_ _Information_ _Processing_ _Systems_, volume 35, pp. 11189–11201. Curran Associates, Inc.,
2022. URL [https://proceedings.neurips.cc/paper_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/487c9d6ef55e73aa9dfd4b48fe3713a6-Paper-Conference.pdf)
[file/487c9d6ef55e73aa9dfd4b48fe3713a6-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2022/file/487c9d6ef55e73aa9dfd4b48fe3713a6-Paper-Conference.pdf)


Abhinav Kumar, Amit Deshpande, and Amit Sharma. Causal effect regularization: Automated detection and removal of spurious correlations. _Advances in Neural Information Processing Systems_,
36:20942–20984, 2023.


S¨oren R. K¨unzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. _Proceedings_ _of_ _the_ _National_ _Academy_ _of_
_Sciences_, 116(10):4156–4165, 2019. doi: 10.1073/pnas.1804597116. URL [https://www.](https://www.pnas.org/doi/abs/10.1073/pnas.1804597116)
[pnas.org/doi/abs/10.1073/pnas.1804597116.](https://www.pnas.org/doi/abs/10.1073/pnas.1804597116)


Bryson Lingenfelter, Sara R. Davis, and Emily M. Hand. A quantitative analysis of labeling issues
in the celeba dataset. In George Bebis, Bo Li, Angela Yao, Yang Liu, Ye Duan, Manfred Lau,
Rajiv Khadka, Ana Crisan, and Remco Chang (eds.), _Advances_ _in_ _Visual_ _Computing_, pp. 129–
141, Cham, 2022. Springer International Publishing. ISBN 978-3-031-20713-6.


Christos Louizos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling.
Causal effect inference with deep latent-variable models. _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing Systems_, 30, 2017.


David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel. Fairness through causal awareness: Learning causal latent-variable models for biased data. In _Proceedings of the Conference on_
_Fairness, Accountability, and Transparency_, FAT* ’19, pp. 349–358, New York, NY, USA, 2019.
Association for Computing Machinery. ISBN 9781450361255. doi: 10.1145/3287560.3287564.
[URL https://doi.org/10.1145/3287560.3287564.](https://doi.org/10.1145/3287560.3287564)


Anita Mahinpei, Justin Clark, Isaac Lage, Finale Doshi-Velez, and Weiwei Pan. Promises and
pitfalls of black-box concept learning models. _arXiv preprint arXiv:2106.13314_, 2021.


Andrei Margeloiu, Matthew Ashman, Umang Bhatt, Yanzhi Chen, Mateja Jamnik, and Adrian
Weller. Do concept bottleneck models learn as intended? _arXiv_ _preprint_ _arXiv:2105.04289_,
2021.


Wang Miao, Zhi Geng, and Eric J Tchetgen Tchetgen. Identifying causal effects with proxy variables
of an unmeasured confounder. _Biometrika_, 105(4):987–993, 2018.


Ricardo Miguel de Oliveira Moreira, Jacopo Bono, M´ario Cardoso, Pedro Saleiro, M´ario A. T.
Figueiredo, and Pedro Bizarro. Diconstruct: Causal concept-based explanations through blackbox distillation. In Francesco Locatello and Vanessa Didelez (eds.), _Proceedings_ _of_ _the_ _Third_
_Conference on Causal Learning and Reasoning_, volume 236 of _Proceedings of Machine Learn-_
_ing Research_, pp. 740–768. PMLR, 01–03 Apr 2024. [URL https://proceedings.mlr.](https://proceedings.mlr.press/v236/moreira24a.html)
[press/v236/moreira24a.html.](https://proceedings.mlr.press/v236/moreira24a.html)


Tuomas Oikarinen, Subhro Das, Lam M Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. In _The Eleventh International Conference on Learning Representations_, 2023.


Judea Pearl. [bayesian analysis in expert systems]: Comment: Graphical models, causality and
intervention. _Statistical_ _Science_, 8(3):266–269, 1993. ISSN 08834237. URL [http://www.](http://www.jstor.org/stable/2245965)
[jstor.org/stable/2245965.](http://www.jstor.org/stable/2245965)


Sukrut Rao, Sweta Mahajan, Moritz B¨ohle, and Bernt Schiele. Discover-then-name: Taskagnostic concept bottlenecks via automated concept discovery. In _Computer_ _Vision_ _–_ _ECCV_


11


_2024:_ _18th_ _European_ _Conference,_ _Milan,_ _Italy,_ _September_ _29–October_ _4,_ _2024,_ _Proceed-_
_ings,_ _Part_ _LXXVII_, pp. 444–461, Berlin, Heidelberg, 2024. Springer-Verlag. ISBN 978-3031-72979-9. doi: 10.1007/978-3-031-72980-5 ~~2~~ 6. URL [https://doi.org/10.1007/](https://doi.org/10.1007/978-3-031-72980-5_26)
[978-3-031-72980-5_26.](https://doi.org/10.1007/978-3-031-72980-5_26)


Severi Rissanen and Pekka Marttinen. A critical look at the consistency of causal estimation
with deep latent variable models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman
Vaughan (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 2021. URL [https:](https://openreview.net/forum?id=vU96vWPrWL)
[//openreview.net/forum?id=vU96vWPrWL.](https://openreview.net/forum?id=vU96vWPrWL)


Yoshihide Sawada and Keigo Nakamura. Concept bottleneck model with additional unsupervised
concepts. _IEEE Access_, 10:41758–41765, 2022.


Rickmer Schulte, David R¨ugamer, and Thomas Nagler. Adjustment for confounding using pretrained representations. In _Forty-second_ _International_ _Conference_ _on_ _Machine_ _Learning_, 2025.
[URL https://openreview.net/forum?id=D2cDJzotb8.](https://openreview.net/forum?id=D2cDJzotb8)


Uri Shalit, Fredrik D Johansson, and David Sontag. Estimating individual treatment effect: generalization bounds and algorithms. In _International conference on machine learning_, pp. 3076–3085.
PMLR, 2017.


Chenming Shang, Shiji Zhou, Hengyuan Zhang, Xinzhe Ni, Yujiu Yang, and Yuwang Wang. Incremental residual concept bottleneck models. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_
_Computer Vision and Pattern Recognition_, pp. 11030–11040, 2024.


Eric J Tchetgen Tchetgen, Andrew Ying, Yifan Cui, Xu Shi, and Wang Miao. An introduction to
proximal causal learning. _arXiv preprint arXiv:2009.10982_, 2020.


Yixin Wang and David Blei. A proxy variable view of shared confounding. In _International Con-_
_ference on Machine Learning_, pp. 10697–10707. PMLR, 2021.


Pengzhou Abel Wu and Kenji Fukumizu. $ _\_ beta$-intact-VAE: Identifying and estimating causal
effects under limited overlap. In _International_ _Conference_ _on_ _Learning_ _Representations_, 2022.
[URL https://openreview.net/forum?id=q7n2RngwOM.](https://openreview.net/forum?id=q7n2RngwOM)


Liyuan Xu, Heishiro Kanagawa, and Arthur Gretton. Deep proxy causal learning and its application
to confounded bandit policy evaluation. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman
Vaughan (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 2021. URL [https://](https://openreview.net/forum?id=0FDxsIEv9G)
[openreview.net/forum?id=0FDxsIEv9G.](https://openreview.net/forum?id=0FDxsIEv9G)


Chih-Kuan Yeh, Been Kim, Sercan Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar.
On completeness-aware concept-based explanations in deep neural networks. _Advances in neural_
_information processing systems_, 33:20554–20565, 2020.


Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck models. In
_The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2023. URL [https:](https://openreview.net/forum?id=nA5AZ8CEyow)
[//openreview.net/forum?id=nA5AZ8CEyow.](https://openreview.net/forum?id=nA5AZ8CEyow)


Weijia Zhang, Lin Liu, and Jiuyong Li. Treatment effect estimation with disentangled latent factors. In _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_, 2020. URL [https://api.](https://api.semanticscholar.org/CorpusID:210943075)
[semanticscholar.org/CorpusID:210943075.](https://api.semanticscholar.org/CorpusID:210943075)


Xuan Zhao, Klaus Broelemann, Salvatore Ruggieri, and Gjergji Kasneci. Causal fairness-guided
dataset reweighting using neural networks. In _2023 IEEE International Conference on Big Data_
_(BigData)_, pp. 1386–1394. IEEE, 2023.


12


A CODE AND IMPLEMENTATION DETAILS


All code, configuration files, and instructions required to reproduce our experiments are available
at [https://anonymous.4open.science/r/causal-residual-concepts-346E/](https://anonymous.4open.science/r/causal-residual-concepts-346E/README.md)
[README.md.](https://anonymous.4open.science/r/causal-residual-concepts-346E/README.md) The repository includes full implementations of UnCoVAEr and all baselines, as well
as scripts for dataset preparation, training, and evaluation. We provide detailed configuration files
specifying model architectures, optimizer settings, training schedules, and hyperparameter choices.
Additional results, including json files and qualitative figures (e.g., counterfactual visualizations for
benchmarked methods), are also included in the repository.


**Reproducibility checklist**


    - **Datasets:** MorphoMNIST variants described in Section 5, with generation scripts included
in the repository.


    - **Evaluation metrics:** mean absolute error (MAE) of ATE estimates, bootstrap uncertainty
test for confounding assessment, as described in Section 6.


    - **Code** **availability:** full training/evaluation code and pre-trained model checkpoints are
provided.


    - **Hyperparameters:** all hyperparameters (learning rate, optimizer type, batch size, KLannealing schedules, Gumbel-Softmax temperature annealing) are specified in configuration files.


    - **Compute:** experiments were run on a single NVIDIA A10 GPU (24GB memory); training
a model typically takes around 15 minutes.


    - **Randomness:** results are averaged over 5 seeds, with random seeds fixed and logged for
reproducibility.


B USE OF LARGE LANGUAGE MODELS


Large Language Models (LLMs) were used as assistive tools during the preparation of this paper.
Their role was limited to improving readability and presentation: for example, rephrasing paragraphs for smoother academic flow, standardizing LaTeX formatting, and polishing grammar. In
some cases, LLMs were also used to suggest more concise ways of summarizing experimental findings. They were not involved in research ideation, experimental design, implementation, or interpretation of results. All scientific contributions are the sole responsibility of the authors, who take
full responsibility for the final content.


13