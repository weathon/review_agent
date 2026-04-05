# RETHINKING DIFFUSION MODEL IN HIGH DIMENSION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


**Curse** **of** **Dimensionality** is an unavoidable challenge in statistical probability
models, yet diffusion models seem to overcome this limitation, achieving impressive results in high-dimensional data generation. Diffusion models assume that
they can learn the statistical quantities of the underlying probability distribution,
enabling sampling from this distribution to generate realistic samples. But is this
really how they work? We argue not, based on the following observations 1)
In high-dimensional sparse scenarios, the fitting target of the diffusion model’s
objective function degrades from a **weighted sum of multiple samples** to a **single**
**sample**, which we believe hinders the model’s ability to effectively learn essential
statistical quantities such as posterior, score, or velocity field. 2) Most inference
methods can **be unified within a simple framework** which involves no statistical
concepts, aligns with the degraded objective function, and provides an novel and
intuitive perspective on the inference process. _Code is available at Supplementary_
_Material._


1 INTRODUCTION


Diffusion models exhibit remarkable competitiveness in high-dimensional data generation scenarios,
particularly in image generation Rombach et al. (2022). Beyond delivering outstanding performance,
diffusion models also possess various elegant mathematical formulations. Sohl-Dickstein et al.
(2015) first introduced the diffusion model approach, which utilizes a Markov chain to transform
complex data distributions into simple normal distributions and then learns the posterior probability
distribution corresponding to the transformation process. Ho et al. (2020) refined the objective
function of diffusion models and introduced Denoised DPM. Song et al. (2020b) generalized the noise
addition process of diffusion models from discrete to continuous and formulated it as a stochastic
differential equation (SDE). Lipman et al. (2022) proposed a new optimization perspective based on
flow matching, enabling the model to directly learn the velocity field of probability flows.


All three of the aforementioned models assume that the diffusion model can learn the statistical
quantities of the data distribution. In the Markov Chain formulation, it is assumed that the model can
learn the posterior probability distribution. In the SDE formulation, it is assumed that the model can
learn the score of the marginal distribution. In the flow matching approach, it is assumed that the
model can learn the velocity field. However, this assumption contradicts conventional understandings.
Traditionally, it is believed that in high-dimensional sparse scenarios, machine learning models cannot
effectively learn complex hidden probability distributions and their essential statistical quantities.
This discrepancy prompts a fundamental inquiry: **This discrepancy raises a fundamental question:**
**Do diffusion models truly learn these complex distributions and their statistical quantities as**
**theoretically assumed?** **If not, why are they still able to generate high-quality samples?** **Could it**
**be that diffusion models operate via a different underlying mechanism?**


**We argue that diffusion models do not learn these statistical quantities; instead, they operate**
**via a different mechanism** .


To support this conclusion, this paper provides a detailed analysis of the objective function and
inference methods of diffusion models.


Section 3 focuses on the objective function. We identify a phenomenon that emerges in highdimensional spaces: due to data sparsity, the fitting target of the diffusion model’s objective function
**degrades** **from** **a** **weighted** **sum** **to** **a** **single** **sample** . Under such conditions, we argue that the


1


model cannot effectively learn the essential statistical quantities of the underlying data distribution,
including the posterior, score, and velocity field.


Section 4 focuses on the inference method. We propose a novel inference framework that not
only aligns with the degraded objective function but also unifies most existing inference methods,
including DDPM Ancestral Sampling, DDIM (Song et al., 2020a), Euler, DPM-Solver (Lu et al.,
2022a), DPM-Solver++ (Lu et al., 2022b), and DEIS (Zhang & Chen, 2022). Furthermore, this
framework provides an entirely new and intuitive perspective for understanding the inference process,
without relying on any statistical concepts.


This work makes the following key contributions:


    - We present the **first rigorous analysis** of the diffusion model objective in high-dimensional
sparse scenarios, demonstrating that its fitting target **degrades from a weighted sum of**
**multiple samples to a single sample** . This degradation prevents the model from effectively
capturing the underlying data distribution and its associated statistical quantities (posterior,
score, velocity field).


    - We further introduce a novel inference framework that **unifies** **most** **existing** **inference**
**methods**, encompassing both stochastic and deterministic approaches. This framework provides an entirely new way of understanding the inference process—free from any reliance on
statistical concepts—while remaining fully consistent with the degraded objective function.


    - Taken together, these contributions offer a **complete and fundamentally new perspective**
on high-dimensional diffusion models, covering both their training objectives and inference
mechanisms. This perspective is simple, intuitive, and free from statistical concepts, opening
up a promising new direction for advancing diffusion models in high-dimensional settings.


2 BACKGROUND


Given a batch of sampled data _X_ 0 [0] _[, X]_ 0 [1] _[, . . ., X]_ 0 _[N]_ [from the random variable] _[ X]_ [0][, the diffusion model]
mixes the data with random noise in different proportions, forming a sequence of new variables
_X_ 1 _, X_ 2 _, · · ·_ _, XT_ . The signal-to-noise ratio (SNR), which represents the ratio of data to noise,
gradually decreases, and by the final variable _XT_, it almost consists entirely of random noise.


For the original diffusion model and VP SDE, they mix in the following way:


1 _−_ _α_ ¯ _t_

- �� _const_ = _C_ 0


_√_
_Xt_ = _[√]_ _α_ ¯ _t · X_ 0 +


1 _−_ _α_ ¯ _t · ε_ (1)


where _α_ ¯ _t_ gradually decreases from 1 to 0, and _t_ takes discrete values from 1 to _T_ .


For flow matching, it mixes in the following way:


_Xt_ = (1 _−_ _σt_ ) _· X_ 0 + _σt · ε_ (2)


where _σt_ also gradually increases from 0 to 1, and in practice, _σt_ = _t_ is often set. _t_ takes continuous
values, _t ∈_ [0 _,_ 1].


**Markov** **Chain-based** **diffusion** **model** For the Markov Chain-based diffusion model, its core
lies in learning the conditional posterior probability _p_ ( _xt−_ 1 _|xt_ ). Since the posterior probability is
approximately a Gaussian function, and its variance is relatively fixed, we can focus on learning
the mean of the conditional posterior probability E _p_ ( _xt−_ 1 _|xt_ )[ _xt−_ 1]. According to the Total Law of
Expectation (Ross, 2010), the mean can be expressed in another form:


               E _p_ ( _xt−_ 1 _|xt_ )[ _xt−_ 1] = _p_ ( _x_ 0 _|xt_ ) _Ep_ ( _xt−_ 1 _|x_ 0 _,xt_ ) ( _xt−_ 1) _dx_ 0 (3)


As seen from equation (7) in Ho et al. (2020), the mean of _p_ ( _xt−_ 1 _|x_ 0 _, xt_ ) can be expressed as a linear
combination of _x_ 0 and _xt_, i.e.


_√_
_α_ ¯ _t−_ 1 _βt_
E _p_ ( _xt−_ 1 _|x_ 0 _,xt_ )[ _xt−_ 1] =


_√_
_·x_ 0 + _αt_ (1 _−_ _α_ ¯ _t−_ 1)


1 _−_ _α_ ¯ _t_

- �� _const_ = _Ct_


_·xt_ (4)


2


Thus, the mean of _p_ ( _xt−_ 1 _|xt_ ) can be further expressed as


         E _p_ ( _xt−_ 1 _|xt_ )[ _xt−_ 1] = _p_ ( _x_ 0 _|xt_ ) ( _C_ 0 _· x_ 0 + _Ct · xt_ ) _dx_ 0 = _C_ 0


The specific proof can be found in Appendix A.1. The integrals above cannot be computed exactly
and are typically approximated using Monte Carlo integration. In practice, the required samples are
typically obtained via **Ancestral Sampling** . The detailed procedure is as follows: sample _X_ 0 from
_p_ ( _x_ 0), and then sample _Xt_ from _p_ ( _xt|x_ 0). The pair ( _X_ 0 _, Xt_ ) follows the joint distribution _p_ ( _x_ 0 _, xt_ ),
and the individual _Xt_ follows _p_ ( _xt_ ), and the individual _X_ 0 follows _p_ ( _x_ 0 _|xt_ = _Xt_ ).


3


_p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 + _Ct · xt_ (5)


Therefore, the objective of Markov Chain-based diffusion model can be considered as learning the
mean of _p_ ( _x_ 0 _|xt_ ), i.e.


min
_θ_


- - 2
_p_ ( _xt_ ) _fθ_ ( _xt_ ) _−_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 _dxt_ (6)
���� ����


where _fθ_ ( _xt_ ) is a learnable neural network function, with input _xt_ .


**Score-based diffusion model** For the score-based diffusion model, its core lies in learning the
score of the marginal distribution _p_ ( _xt_ ) - _∂_ log _p_ ( _xt_ ) �. Similar to the Markov chain-based diffusion


score of the marginal distribution _p_ ( _xt_ ) - _∂_ log _∂x pt_ ( _xt_ ) �. Similar to the Markov chain-based diffusion

model, by introducing another variable _X_ 0, the score can be expressed in another form:


_∂xt_


_p_ ( _xt_ )  
= _p_ ( _x_ 0 _|xt_ ) _[∂]_ [log] _[ p]_ [(] _[x][t][|][x]_ [0][)]
_∂xt_ _∂xt_


_∂_ log _p_ ( _xt_ )


_dx_ 0 (7)
_∂xt_


The proof of this relationship can be found in Appendix A.2.


Since _p_ ( _xt|x_ 0) _∼N_ - _xt_ ; _[√]_ _α_ ¯ _tx_ 0 _,_ _[√]_ 1 _−_ _α_ ¯ _t_ �, the score of _p_ ( _xt|x_ 0) can be expressed as


_−_ 1
_·x_ 0 +
1 _−_ _α_ ¯ _t_

   - ���
_const_ = _St_


_p_ ( _xt|x_ 0)

= _−_ _[x][t][ −]_ _[√][α]_ [¯] _[t][x]_ [0]
_∂xt_ 1 _−_ _α_ ¯ _t_


_∂_ log _p_ ( _xt|x_ 0)


_√_

_[√][α]_ [¯] _[t][x]_ [0] _α_ ¯ _t_

=
1 _−_ _α_ ¯ _t_ 1 _−_ _α_ ¯ _t_

      - ���
_const_ = _S_ 0


_·xt_ (8)


Thus, the score of _p_ ( _xt_ ) can be expressed as


_∂_ log _p_ ( _xt_ )    
= _p_ ( _x_ 0 _|xt_ )( _S_ 0 _· x_ 0 + _St · xt_ ) _dx_ 0 = _S_ 0
_∂xt_


_p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 + _St · xt_ (9)


Therefore, the objective of score-based diffusion model can also be considered as learning the mean
of _p_ ( _x_ 0 _|xt_ ).


**Flow Matching-based diffusion model** The core of the flow matching-based diffusion model lies
in learning the velocity field of the probability flow. According to Theorem 1 in Lipman et al. (2022),
the velocity field _u_ ( _xt_ ) can be expressed as a weighted sum of the conditional velocity field _u_ ( _xt|x_ 0),
i.e.


               _u_ ( _xt_ ) = _p_ ( _x_ 0 _|xt_ ) _u_ ( _xt|x_ 0) _dx_ 0 (10)


From equation 2, we know that the conditional velocity field _u_ ( _xt|x_ 0) is

_u_ ( _xt|x_ 0) ≜ [d] _[x][t]_ (11)

d _t_ [=] _[ ε][ −]_ _[x]_ [0]

Thus, the velocity field _u_ ( _xt_ ) can be expressed as


           -           _u_ ( _xt_ ) = _p_ ( _x_ 0 _|xt_ )( _ε −_ _x_ 0) _dx_ 0 = _ε −_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 (12)


Therefore, the objective of flow matching-based diffusion model can also be considered as learning
the mean of _p_ ( _x_ 0 _|xt_ ).


**Equivalent to predicting** _X_ 0 Fitting the mean of _p_ ( _x_ 0 _|xt_ ) is equivalent to **predicting** _X_ 0, i.e.


min
_θ_


- - 2
_p_ ( _xt_ ) _fθ_ ( _xt_ ) _−_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 _dxt_ _⇐⇒_ min
���� ���� _θ_


��
_p_ ( _x_ 0 _, xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _dxt_


3 IMPACT OF SPARSITY ON THE OBJECTIVE FUNCTION


We first show the form of the posterior probability distribution _p_ ( _x_ 0 _|xt_ ).


3.1 FORM OF THE POSTERIOR _p_ ( _x_ 0 _|xt_ )


For convenience, we use a unified form to represent the two mixing ways in equation 1 and equation 2
as follows: _xt_ = _c_ 0 _· x_ 0 + _c_ 1 _· ε_ . When _c_ [2] 0 [+] _[ c]_ 1 [2] [=] [1][,] [this represents the mixing way of Markov]
Chain-based and Score-based diffusion model. When _c_ 0 + _c_ 1 = 1, it represents the Flow Matching
mixing way. Under this representation, _p_ ( _xt|x_ 0) _∼N_ ( _xt_ ; _c_ 0 _x_ 0 _, c_ [2] 1 [)][.]


From the analysis in Appendix A.3, the posterior _p_ ( _x_ 0 _|xt_ ) has the following form:


The mean of _p_ ( _x_ 0 _|xt_ ) is a weighted sum of all _X_ 0 _[i]_ [samples, and the weight is inversely proportional]
to the distance between _X_ 0 _[i]_ [and] _[ µ]_ [.] [If one sample is much closer than all others, the weighted sum]
degrades to that single sample. This is more likely with sparse data.


Figure 1 presents an example with sparse data ( _X_ 0, blue), and small noise std (green circle). In
this case, most of _Xt_ remain near its origin data sample. This make _p_ ( _x_ 0 _|xt_ ) highly peaked at the
closest _X_ 0, causing its mean to degrade from a weighted sum to that single sample. We call this
phenomenon **weighted sum degradation** and argue it potentially hinders the model learning the true
data distribution.


Next, we analyze _weighted sum degradation_ for conditional ImageNet-256 and ImageNet-512(Deng
et al., 2009). Both datasets have high pixel dims (196608 and 786432) and retain high latent
dims (4096 and 16480) after VAE(Kingma et al., 2013; Rombach et al., 2022) compression. As
compression is typical, we will only consider the compressed case below.


We calculate the proportion of degradation. First, we sample _Xt_ as in training (first randomly select
an _X_ 0, then sample _Xt_ from _p_ ( _xt|x_ 0 = _X_ 0)). Then, we determine whether _p_ ( _x_ 0 _|xt_ = _Xt_ ) is
degraded. If there exists an _X_ 0 _[′]_ [such that] _[ p]_ [(] _[x]_ [0] [=] _[X]_ 0 _[′]_ _[|][x][t]_ [=] _[X][t]_ [)] _[>]_ [0] _[.]_ [9][, then we consider] _[ weighted]_
_sum degradation_ to be present; if _X_ 0 _[′]_ [=] _[ X]_ [0][, it is called] _[ weighted sum degradation to][ X]_ [0][.]


Since noise level also affects weighted sum degradation, we calculate degradation rates separately for
different _t_ . We calculate the proportions of both _weighted sum degradation_ and _degradation to X_ 0
under two noise mixing schemes: VP (equation 1) and Flow Matching (equation 2).


Tables 1 and 2 present statistics for both datasets, showing several clear patterns:


4


       _p_ ( _x_ 0 _|xt_ ) = Normalize exp _[−]_ [(] _[x]_ [0] _[ −]_ _[µ]_ [)][2]


_[x][t]_ _σ_ = _[c]_ [1]

_c_ 0 _c_ 0


    
[0] _[ −]_ _[µ]_ [)][2]

_p_ ( _x_ 0) where _µ_ = _[x][t]_
2 _σ_ [2] _c_ 0


(13)
_c_ 0


Here, _p_ ( _x_ 0) is the hidden data distribution, which is unknown and cannot be sampled directly.
It can only be randomly selected from the existing samples _{X_ 0 [0] _[, X]_ 0 [1] _[, . . ., X]_ 0 _[N]_ _[}]_ [(] _[X]_ 0 _[i]_ _[∼]_ _[p]_ [(] _[x]_ [0][)][).]
The selection process can be considered as sampling from the following mixed Dirac delta
distribution: _p_ ( _x_ 0) = _N_ [1] - _Ni_ =0 _[δ]_ - _x_ 0 _−_ _X_ 0 _[i]_ �. Substituting this into equation 13, we get:


[1] exp _[−]_ [(] _[x]_ [0] _[ −]_ _[µ]_ [)][2]

_Zc_ 2 _σ_ [2]


_N_

- _δ_ - _x_ 0 _−_ _X_ 0 _[i]_ - (14)


_i_ =0


_p_ ( _x_ 0 _|xt_ ) = [1]


2 _σ_ [2]


Here, _µ_ = _[x][t]_


_[x]_ _c_ 0 _[t]_ [,] _[ σ]_ [=] _[c]_ _c_ [1] 0


Here, _µ_ = _c_ 0 _[t]_ [,] _[ σ]_ [=] _c_ [1] 0 [, and] _[ Z][c]_ [is normalization factor.] [It can be seen that when] _[ p]_ [(] _[x]_ [0][)][ is discrete,]

_p_ ( _x_ 0 _|xt_ ) is also discrete, and the probability of each discrete value _X_ 0 _[i]_ [is] **[ inversely]** [ proportional to]
**the distance between** _X_ 0 _[i]_ **[and]** _[ µ]_ [.] [A similar conclusion is also presented in Appendix B of Karras]
et al. (2022), although the derivation method differ.


3.2 WEIGHTED SUM DEGRADATION PHENOMENON


We further analyze the characteristics of the mean of _p_ ( _x_ 0 _|xt_ ). According to the definition of
expectation, the mean of _p_ ( _x_ 0 _|xt_ ) can be expressed as:


_N_

         _x_ 0 _p_ ( _x_ 0 _|xt_ ) _dx_ 0 = [1]          - _X_ 0 _[i]_ [exp] _[−]_ [(] _[X]_ 0 _[i]_ _[−]_ _[µ]_ [)][2] (15)


_N_


(15)
2 _σ_ [2]


_Zc_


- _X_ 0 _[i]_ [exp] _[−]_ [(] _[X]_ 0 _[i]_ _[−]_ _[µ]_ [)][2]

2 _σ_ [2]

_i_ =0


Table 1: Statistics of ImageNet-256(weighted sum degradation / weighted sum degradation to _X_ 0)


**merging\time** **vp** 1.00/1.00 1.00/1.00 1.00/0.98 0.91/0.57 0.41/0.01 0.02/0.00 0.00/0.00 0.00/0.00
**flow** 1.00/1.00 1.00/1.00 1.00/1.00 1.00/1.00 1.00/0.95 0.97/0.69 0.76/0.15 0.09/0.00


Table 2: Statistics of ImageNet-512(weighted sum degradation / weighted sum degradation to _X_ 0)


**merging\time** **vp** 1.00/1.00 1.00/1.00 1.00/0.98 0.98/0.57 0.87/0.08 0.50/0.00 0.03/0.00 0.00/0.00
**flow** 1.00/1.00 1.00/1.00 1.00/1.00 1.00/1.00 1.00/0.94 0.99/0.67 0.95/0.20 0.71/0.01


    - As _t_ decreases, the _weighted sum degradation_ phenomenon becomes more pronounced.

    - The degradation rate of Flow Matching is higher than that of VP.

    - The higher the dimension, the greater the proportion of degradation.


Besides, we observe severe degradation in both datasets for both VP and Flow Matching, especially
for _t <_ 600. Furthermore, due to limited sampling during training, each _p_ ( _x_ 0 _|xt_ = _Xt_ ) cannot be
sufficiently sampled, so **the actual degradation ratio should be higher than the statistics show** .


In high dimensions, each _p_ ( _x_ 0 _|xt_ = _Xt_ ) should be complex. When _weighted_ _sum_ _degradation_
occurs, it is equivalent to using a single sample as an estimator of the mean, which typically have
large error. If we cannot provide an accurate fitting target, we argue that the model is unlikely to learn
the ideal target accurately. Therefore, **it is necessary to reconsider if diffusion models can truly**
**learn the hidden probability distribution and how they work** .


3.3 A SIMPLE WAY TO UNDERSTAND THE OBJECTIVE FUNCTION


As shown previously, weighted sum degradation is significant in high dimensions, which reduces the
fitting target to the original data sample ( _X_ 0). Therefore, we can understand the objective in a simple
way: **predict the original data sample (** _X_ 0 **) from the noise-mixed sample (** _Xt_ **)** .


From the perspective of the frequency spectrum, we can further understand the principle (Dieleman,
2024).


As seen in Figure 2, natural image spectra concentrate energy in low frequencies (bright centrally, dark
peripherally), while noise have a uniform spectrum. Thus, when mixed with noise, high frequencies
always have lower SNR (signal-noise-ratio) than low frequencies. As noise grows, high frequencies
are submerged first, then low frequencies (Figure 3).


5


Figure 1: Impact of data sparsity on posterior
probability distribution


Figure 2: Left: Natural image and its
spectrum. Right: noise and its spectrum


When training a model to predict _X_ 0 from noise-mixed samples, the model prioritizes frequencies
based on their SNR. It easily predicts non-submerged frequencies (likely copying them).For submerged frequencies, it prioritizes predicting the lower-frequency components, as they have relatively
higher SNR and larger amplitudes (giving them more weight in the Euclidean loss).


Thus, the objective can be further understood as **filtering higher-frequency components – com-**
**pleting the filtered frequency components** (Figure 4). At large _t_, even some low frequencies are
submerged, so the model prioritizes predicting low frequencies. At small _t_, only high frequencies are
submerged, and the model works on predicting these details. This frequency-dependent process is
confirmed during inference: early steps (large _t_ ) generate contours, while later steps (small _t_ ) add
details. Since the model compensates for the submerged frequencies, it can also be regarded as an
**information enhancement operator** .


Figure 3: Image and noise frequency spectrum Figure 4: New perspective of training object
function


4 A UNIFIED INFERENCE FRAMEWORK-NATURAL INFERENCE


We know that the inference methods of diffusion models rely on an assumption that the model
can learn the hidden probability distributions or statistical quantities. However, as pointed out in
the Section 3.2, in high-dimensional spaces the degradation phenomenon prevents the model from
effectively learning these quantities. Therefore, it is necessary to attempt to reinterpret existing
inference methods from a new perspective. Moreover, we have also seen in the Section 3.3 that the
degraded objective function can be understood in a simple way - predicting the original image _x_ 0
from a noisy image _xt_ . Based on the principle of **train-test matching**, this naturally leads us to ask:
can current inference methods also be understood in a similar, simpler way?


The answer is yes. Below, we will reveal that most of inference methods can be unified into a
simple framework based on predicting _x_ 0, including Ancestor Sampling, DDIM, Euler, DPMsolver,
DPMSolver++, DEIS, and Flow Matching solvers, among others.


We first introduce a class of key operations contained in the new framework.


4.1 SELF GUIDENCE


Following the concept of Classifier Free Guidance (Ho & Salimans, 2022), we introduce a new
operation called Self Guidance. The principle of Classifier Free Guidance can be summarized as
follows:


_Iout_ = _Ibad_ + _λ ·_ ( _Igood −_ _Ibad_ ) (16)


where _Ibad_ is the output of a less capable model, _Igood_ is the output of a more capable model, and
both models share the same input. _λ_ controls the degree of guidance.


In fact, Classifier Free Guidance is somewhat similar to Unsharp Masking algorithm in traditional
image enhancing processing(Gonzalez & Woods, 2017; scikit-image Development Team, 2013).
In Unsharp Masking algorithm, _Igood_ is the original image, and _Ibad_ is the image after Gaussian


6


blur. The term ( _Igood_ _−_ _Ibad_ ) provides the edge information, which, when added to the original
image _Igood_, results in an image with sharper edges. Therefore, Classifier Free Guidance can also be
considered as an **image enhancement** operation.


In the diffusion model inference process, a series of predicted _x_ 0 are generated, where the quality of
_x_ 0 starts poor and improves over time. If an earlier predicted _x_ 0 is used as _Ibad_ and a later predicted
_x_ 0 is used as _Igood_, then in this paper, we refer to this operation as Self Guidance, because both _Ibad_
and _Igood_ are outputs of the same model, and no additional model is needed.


Based on the value of _λ_, we further classify Self Guidance as follows:


    - When _λ >_ 1, it is called **Fore Self Guidance**, where the output improves the quality. See
Fig. 6(c).


    - When 0 _< λ <_ 1, it is called **Mid Self Guidance**, where the output is a linear interpolation
between _Ibad_ and _Igood_, with a quality worse than _Igood_ but better than _Ibad_ . See Fig. 6(d).


    - When _λ <_ 0, it is called **Back Self Guidance**, where the output is not only worse than _Igood_,
but also worse than _Ibad_ . See Fig. 6(e).


As shown in Appendix B, **the linear combination of any two model outputs can be viewed as a**
**single Self Guidance, while the linear combination of multiple model outputs can be viewed as**
**a composition of multiple Self Guidances** .


4.2 NATURAL INFERENCE


Figure 5: A new inference framework - Natural Inference


The new inference framework is illustrated in Figure 5, with the core ideas summarized as follows:


    - It consists of _T_ models that predict _X_ 0;


    - Each model takes two part inputs: signal (image) and noise;


    - The image signal is a linear combination of outputs from previous models, while the noise
is a linear combination of previous noise and newly added noise;

    - At time _t_, the sum of the coefficients corresponding to the image signal ( [�] _[T]_ _i_ = _t_ +1 _[c]_ _t_ _[i]_ [) equals]
_√α_ ¯ _t_, and the square root of the sum of the squared noise coefficients (�� _Ti_ = _t_ [(] _[b]_ _t_ _[i]_ [)][2][) equals]


7


_√_
1 _−_ _α_ ¯ _t_ . This means that **the magnitudes of the signal and noise remain consistent with**
**those used during the training phase** .


As shown in Section 4.1, **the** **linear** **combination** **of** **image** **signals** **can** **be** **interpreted** **as** **a**
**composition of multiple Self Guidance operations** . The linear combination of independent noise is
still noise (Taboga, 2021). Since the input signal of each model depends only on the output signals of
previous models, the inference framework exhibits an **autoregressive** structure.

In this paper, we refer to _[√]_ _α_ ¯ _t_ as the **marginal signal coefficient**, and _[√]_ 1 _−_ _α_ ¯ _t_ as the **marginal**
**noise coefficient** . The term [�] _i_ _[T]_ = _t_ +1 _[c]_ _t_ _[i]_ [is referred to as the] **[ equivalent marginal signal coefficient]** [,]

and ~~�~~ - _Ti_ = _t_ [(] _[b]_ _t_ _[i]_ [)][2] [is called the] **[ equivalent marginal noise coefficient]** [.] [For clarity, all coefficients]
are organized into matrix form, as shown in the lower part of Figure 5. Due to the autoregressive
property, the signal coefficient matrix has a lower triangular structure.


4.3 REPRESENT SAMPLING METHODS WITH NATURAL INFERENCE FRAMEWORK


This section briefly demonstrates how various sampling methods can be reformulated within the
Natural Inference Framework. For more detailed explanations, please refer to Appendix C.


For first-order sampling methods (including DDPM, DDIM, ODE Euler, SDE Euler, and Flow
Matching ODE Euler), their iterative procedures can all be expressed in the following form:
_yt_ = _ft_ ( _xt_ ) (17)


_xt−_ 1 = _dt−_ 1 _· xt_ + _et−_ 1 _· yt_ + _gt−_ 1 _· ϵt−_ 1 (18)
Here, _ft_ is the model function predicting _x_ 0 at step _t_ . _xt_, _yt_, and _ϵt−_ 1 are vectors, while _dt−_ 1, _et−_ 1,
and _gt−_ 1 are fixed scalars. For deterministic methods, _gt−_ 1 is zero.


Starting from _xT_, we can iterate according to the above equation to further determine the expressions
of _xT −_ 1, _xT −_ 2, _· · ·_, _x_ 1, and _x_ 0. Each _xt_ can be represented as two components: one is a linear
combination of _{yi}_ _[T]_ _i_ = _t_ +1 [,] [and] [the] [other] [is] [a] [linear] [combination] [of] _[{][ε][i][}][T]_ _i_ = _t_ [.] [Since] _[d][t][−]_ [1][,] _[e][t][−]_ [1][,]
and _gt−_ 1 are all known constants, the weights for each element in _{yi}_ _[T]_ _i_ = _t_ +1 [and] _[{][ε][i][}][T]_ _i_ = _t_ [can] [be]
calculated.

The calculation results show that the sum of the coefficients corresponding to _{yi}_ _[T]_ _i_ = _t_ +1 [is] [ap-]
proximatelyapproximatelyequ _[√]_ al1 _−_ to _α_ ¯ _[√]_ _t_ . _α_ ¯Moreover, the approximation error decreases as the number of sampling _t_, and the square root of the sum of squared coefficients for _{εi}_ _[T]_ _i_ = _t_ [is]
steps increases (see Figures 7-9 and Figures 13-14). Therefore, these sampling methods can be
represented in the form of the Natural Inference framework.


The above computation can be quite complex, especially when the number of sampling steps is
large. Therefore, it is necessary to seek more efficient computation methods. Symbolic computation
software (Team, 2013) offers a promising solution. With minor modifications to the original algorithm
code, it can automatically compute the expression for each _xt_ . For more detailed information, please
refer to the accompanying code.


For higher-order sampling methods, their iteration rules are relatively complex, but the expression for
_xt_ can also be quickly calculated with the help of symbolic computation software. The calculation
results indicate that DPMSOLVER, DPMSOLVER++, and DEIS yield results similar to those of
first-order sampling methods (see Figures 10-12).


Appendix C.6 also provides a simple and intuitive example that represent the five-step Euler Inference
method with the form of Natural Inference.


4.4 ADVANTAGES OF THE NATURAL INFERENCE FRAMEWORK


Thus, we have used a completely new perspective to explain high-dimensional diffusion models,
including the objective function during training and the inference algorithm during testing. This new
perspective has several advantages:


    - The new perspective maintains **training-testing consistency**, where the goal during training
is to predict _x_ 0, and the goal during testing is also to predict _x_ 0.


8


- The new perspective divides the inference process into a series of operations for predicting
_x_ 0, each of which has clear input image signals and output image signals. This makes the
inference process **more visual and interpretable**, providing significant help for debugging
and problem analysis. Figures 15 and 16 provide a visualization of the complete inference
process.


    - As discussed in Section 3.3, predicting _x_ 0 can be regarded as an information enhancement
operator. Similarly, Section 4.1 shows that classifier-free guidance can also be viewed
as an information enhancement operator. Therefore, the entire inference process can be
understood as a progressive enhancement of information, a process that does not require any
statistical knowledge.


    - From this new perspective, existing sampling algorithms are merely specific parameter
configurations within the Natural Inference framework. Within this framework, other,
potentially more optimal parameter configurations may exist that can generate higher-quality
samples. Exploring these possibilities could be a direction for future work.


5 CONCLUSION


This paper investigates the operational principles of high-dimensional diffusion models. We first
analyze the objective function and explore the impact of data sparsity in high-dimensional settings,
demonstrating that, due to such sparsity, these models cannot effectively learn the underlying
probability distributions or their key statistical quantities. Building on this insight, we propose a
novel perspective for interpreting the objective function. In addition, we introduce a new inference
framework that not only unifies most inference methods but also aligns with the degraded objective
function. This framework offers an intuitive understanding of the inference process without relying
on any statistical concepts. We hope that this work will encourage the community to rethink the
operational principles of high-dimensional diffusion models and further enhance their training and
inference methodologies.


REFERENCES


Brian DO Anderson. Reverse-time diffusion equation models. _Stochastic_ _Processes_ _and_ _their_
_Applications_, 12(3):313–326, 1982.


Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum,
Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without
noise. _Advances in Neural Information Processing Systems_, 36:41259–41282, 2023.


Fan Bao, Chongxuan Li, Jiacheng Sun, and Jun Zhu. Why are conditional generative models better
than unconditional ones? _arXiv preprint arXiv:2212.00362_, 2022a.


Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal covariance
with imperfect mean in diffusion probabilistic models. _arXiv preprint arXiv:2206.07309_, 2022b.


Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal
reverse variance in diffusion probabilistic models. _arXiv preprint arXiv:2201.06503_, 2022c.


Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth
words: A vit backbone for diffusion models. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _conference_ _on_
_computer vision and pattern recognition_, pp. 22669–22679, 2023.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pp. 248–255. Ieee, 2009.


Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. _Advances_
_in neural information processing systems_, 34:8780–8794, 2021.


Sander Dieleman. Perspectives on diffusion, 2023. [URL https://sander.ai/2023/07/20/](https://sander.ai/2023/07/20/perspectives.html)
[perspectives.html.](https://sander.ai/2023/07/20/perspectives.html)


9


[Sander Dieleman. Diffusion is spectral autoregression, 2024. URL https://sander.ai/2024/](https://sander.ai/2024/09/02/spectral-autoregression.html)
[09/02/spectral-autoregression.html.](https://sander.ai/2024/09/02/spectral-autoregression.html)


Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for
high-resolution image synthesis. In _Forty-first international conference on machine learning_, 2024.


Rafael Gonzalez and Richard Woods. _Digital image processing, 4th Edition_ . Pearson education india,
2017.


Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv preprint arXiv:2207.12598_,
2022.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in_
_neural information processing systems_, 33:6840–6851, 2020.


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _Advances in neural information processing systems_, 35:26565–26577,
2022.


Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.


Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching
for generative modeling. _arXiv preprint arXiv:2210.02747_, 2022.


Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow. _arXiv preprint arXiv:2209.03003_, 2022.


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast
ode solver for diffusion probabilistic model sampling in around 10 steps. _Advances in Neural_
_Information Processing Systems_, 35:5775–5787, 2022a.


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast
solver for guided sampling of diffusion probabilistic models. _arXiv preprint arXiv:2211.01095_,
2022b.


Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.
In _International conference on machine learning_, pp. 8162–8171. PMLR, 2021.


William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of_
_the IEEE/CVF international conference on computer vision_, pp. 4195–4205, 2023.


Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image
synthesis. _arXiv preprint arXiv:2307.01952_, 2023.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In _International conference on machine learning_, pp.
8748–8763. PmLR, 2021.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF confer-_
_ence on computer vision and pattern recognition_, pp. 10684–10695, 2022.


Sheldon Ross. A first course in probability. 2010.


scikit-image Development Team. Unsharp masking, 2013. URL [https://scikit-image.](https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_unsharp_mask.html)
[org/docs/0.25.x/auto_examples/filters/plot_unsharp_mask.html.](https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_unsharp_mask.html)


Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. In _International conference on machine learning_,
pp. 2256–2265. pmlr, 2015.


10


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv_
_preprint arXiv:2010.02502_, 2020a.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint_
_arXiv:2011.13456_, 2020b.


Marco Taboga. Linear combinations of normal random variables, 2021.
URL [https://www.statlect.com/probability-distributions/](https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations)
[normal-distribution-linear-combinations.](https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations)


SymPy Development Team. simpy documentation, 2013. URL [https://](https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html#what-is-symbolic-computation)
[docs.sympy.org/latest/tutorials/intro-tutorial/intro.html#](https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html#what-is-symbolic-computation)
[what-is-symbolic-computation.](https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html#what-is-symbolic-computation)


Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural computa-_
_tion_, 23(7):1661–1674, 2011.


Li Yang and Abdallah Shami. On hyperparameter optimization of machine learning algorithms:
Theory and practice. _Neurocomputing_, 415:295–316, 2020.


Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator.
_arXiv preprint arXiv:2204.13902_, 2022.


Zhenxin Zheng. The art of dpm, 2023. [URL https://github.com/blairstar/The_Art_](https://github.com/blairstar/The_Art_of_DPM)
[of_DPM.](https://github.com/blairstar/The_Art_of_DPM)


Zhenxin Zheng. Understanding diffusion probability model interactively, 2024. URL [https:](https://huggingface.co/spaces/blairzheng/DPMInteractive)
[//huggingface.co/spaces/blairzheng/DPMInteractive.](https://huggingface.co/spaces/blairzheng/DPMInteractive)


11


A ADDITIONAL PROOFS


A.1 PREDICTING POSTERIOR MEAN IS EQUIVALENT TO PREDICTING _X_ 0


In the following, we will prove that the following two objective functions are equivalent:


2
_dxt_ _⇐⇒_ min

���� _θ_


min
_θ_


- _p_ ( _xt_ ) _fθ_ ( _xt_ ) _−_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0
����


��
_p_ ( _x_ 0 _, xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _dxt_


where, _fθ_ ( _xt_ ) is a neural network model.


Proof:

For �� _fθ_ ( _xt_ ) _−_ - _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0��2, the following relation holds:


    _fθ_ ( _xt_ ) _−_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0
����


2
(19)

����


         -          = _∥fθ_ ( _xt_ ) _∥_ [2] _−_ 2 _fθ_ ( _xt_ ) _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 + _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0
����


2
(20)

����


     -      = _p_ ( _x_ 0 _|xt_ ) _∥fθ_ ( _xt_ ) _∥_ [2] _dx_ 0 _−_ 2 _fθ_ ( _xt_ ) _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0 + _C_ 1 (21)


     -      = _p_ ( _x_ 0 _|xt_ )    - _∥fθ_ ( _xt_ ) _∥_ [2] _−_ 2 _fθ_ ( _xt_ ) _x_ 0 + _x_ [2] 0� _dx_ 0 _−_ _p_ ( _x_ 0 _|xt_ ) _x_ [2] 0 _[dx]_ [0] [+] _[ C]_ [1] (22)


     = _p_ ( _x_ 0 _|xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _−_ _C_ 2 + _C_ 1 (23)


Where _C_ 1 and _C_ 2 are constants that do not depend on _θ_ . In equation 21, we apply _∥fθ_ ( _xt_ ) _∥_ [2] =
_∥fθ_ ( _xt_ ) _∥_ [2][ �] _p_ ( _x_ 0 _|xt_ ) _dx_ 0 = - _p_ ( _x_ 0 _|xt_ ) _∥fθ_ ( _xt_ ) _∥_ [2] _dx_ 0.


Substituting the above relation into the objective function for predicting posterior mean, we get:


- _p_ ( _xt_ ) _fθ_ ( _xt_ ) _−_ _p_ ( _x_ 0 _|xt_ ) _x_ 0 _dx_ 0
����


2
_dxt_ (24)

����


 - ��
= _p_ ( _xt_ ) _p_ ( _x_ 0 _|xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _−_ _C_ 2 + _C_ 1


_dxt_ (25)


��       = _p_ ( _x_ 0 _, xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _dxt_ + _p_ ( _xt_ ) ( _C_ 1 _−_ _C_ 2) _dxt_ (26)


��
= _p_ ( _x_ 0 _, xt_ ) _∥fθ_ ( _xt_ ) _−_ _x_ 0 _∥_ [2] _dx_ 0 _dxt_ + _C_ 3 (27)


That is, the two objective functions differ only by a constant that does not depend on the optimization
parameters. Therefore, the two objective functions are equivalent.


A.2 CONDITIONAL SCORE


Below is the proof of the following relation:


_p_ ( _xt_ )  
= _p_ ( _x_ 0 _|xt_ ) _[∂]_ [log] _[ p]_ [(] _[x][t][|][x]_ [0][)]
_∂xt_ _∂xt_


_∂_ log _p_ ( _xt_ )


_dx_ 0 (28)
_∂xt_


Proof:


12


In the above derivation, due to the presence of the normalization operator, we can ignore the factor
~~_√_~~ 1

2 _πc_ [2] 1 [.]


13


_∂_ log _p_ ( _xt_ )


(29)
_∂xt_


_p_ ( _xt_ ) 1 _∂p_ ( _xt_ )

=
_∂xt_ _p_ ( _xt_ ) _∂xt_


1 _∂_ �� _p_ ( _x_ 0) _p_ ( _xt|x_ 0) _dx_ 0�
= (30)
_p_ ( _xt_ ) _∂xt_


 - _p_ ( _x_ 0) _∂p_ ( _xt|x_ 0)
= _dx_ 0 (31)
_p_ ( _xt_ ) _∂xt_


 - _p_ ( _x_ 0 _, xt_ ) _/p_ ( _xt_ ) _∂p_ ( _xt|x_ 0)
= _dx_ 0 (32)
_p_ ( _x_ 0 _, xt_ ) _/p_ ( _x_ 0) _∂xt_


 - _p_ ( _x_ 0 _|xt_ ) _∂p_ ( _xt|x_ 0)
= _dx_ 0 (33)
_p_ ( _xt|x_ 0) _∂xt_


 = _p_ ( _x_ 0 _|xt_ ) _[∂]_ [log] _[ p]_ [(] _[x][t][|][x]_ [0][)] _dx_ 0 (34)

_∂xt_


A.3 FORM OF THE POSTERIOR PROBABILITY


The following derivation is based on Zheng (2023) and Zheng (2024).


Assume that _xt_ has the following form:


_xt_ = _c_ 0 _· x_ 0 + _c_ 1 _· ϵ_ where _c_ 0 and _c_ 1 is constant (35)


Then we have:


_p_ ( _xt|x_ 0) _∼N_ ( _xt_ ; _c_ 0 _x_ 0 _, c_ [2] 1 [)] (36)


According to Bayes’ theorem, we have

_p_ ( _x_ 0 _|xt_ ) = _[p]_ [(] _[x][t][|][x]_ [0][)] _[p]_ [(] _[x]_ [0][)] (37)

_p_ ( _xt_ )

_p_ ( _xt|x_ 0) _p_ ( _x_ 0)
=           - _p_ ( _xt|x_ 0) _p_ ( _x_ 0) _dx_ 0 (38)

= Normalize� _p_ ( _xt|x_ 0) _p_ ( _x_ 0)� (39)


where _Normalize_ represents the normalization operator, and the normalization divisor is

- _p_ ( _xt|x_ 0) _p_ ( _x_ 0) _dx_ 0.


Substituting Equation equation 36 into this, we get:


_p_ ( _x_ 0 _|xt_ ) = Normalize


= Normalize


1
exp _[−]_ [(] _[x][t][ −]_ _[c]_ [0] _[x]_ [0][)][2]

~~�~~ 2 _πc_ [2] 1 2 _c_ [2] 1


 ~~�~~ 21 _πc_ [2] 1 exp _−_ ( _x_ 02 _−_ _[c]_ 1 [2][2] _[x]_ _c_ 0 _[t]_


1
2 _[c]_ [2]
_c_ [2] 0


_p_ ( _x_ 0)
2 _c_ [2] 1


(40)





_[x]_ _c_ 0 _[t]_ [)][2]





_p_ ( _x_ 0) (41)


    -    = Normalize exp _[−]_ [(] _[x]_ [0] _[ −]_ _[µ]_ [)][2] _p_ ( _x_ 0) (42)

2 _σ_ [2]


where _µ_ = _[x][t]_


_[x][t]_ _σ_ = _[c]_ [1]

_c_ 0 _c_ 0


(43)
_c_ 0


(a) (b) (c) (d) (e)


Figure 6: (a) Model output on t=540 ( _Ibad_ ) (b) Model output on t=500 ( _Igood_ ) (c) Output of
Fore Self Guidence (d) Output of Mid Self Guidence (e) Output of Back Self Guidence


B SELF GUIDANCE AND ITS COMPOSITION


In this section, we show that the linear combination of any two model outputs(prediting _x_ 0) can be
viewed as a Self Guidance operation.


As described in Section 3.2, the self guidance is defined as follows:


_Iout_ = _Ibad_ + _λ ·_ ( _Igood −_ _Ibad_ ) (44)


This equation can be further written as


_Iout_ = _λ · Igood_ + (1 _−_ _λ_ ) _· Ibad_ (45)


= _ηgood · Igood_ + _ηbad · Ibad_ (46)


where _ηgood,_ _ηbad_ _∈_ _real_ _ηgood_ + _ηbad_ = 1 (47)


As shown above, the coefficients of _Ibad_ and _Igood_ can take any value, but the sum of _Ibad_ and _Igood_
must equal 1. For **Fore Self Guidence**, _ηgood_ _>_ 0, _ηbad_ _<_ 0; for **Mid Self Guidence**, _ηgood_ _>_ 0,
_ηbad_ _>_ 0; for **Back Self Guidence**, _ηgood_ _<_ 0, _ηbad_ _>_ 0.


For the linear combination of any two model outputs, it can be written as:


_a_ _b_
_Iout_ = _a · Igood_ + _b · Ibad_ = ( _a_ + _b_ ) _·_ ( (48)
_a_ + _b_ _[·][ I][good]_ [ +] _a_ + _b_ _[·][ I][bad]_ [)]


Since the sum of the two coefficients equals 1, the operation inside the parentheses is a Self Guidence
operation.


Thus, **the linear combination of any two** _Ibad_ **and** _Igood_ **can be represented as Self Guidence**
**with a scaling factor** .


For the linear combination of multiple model outputs, it can be written as:


_Iout_ = _a · Ia_ + _b · Ib_ + _c · Ic_ (49)

_a_ _b_
= ( _a_ + _b_ ) _·_ ( (50)
_a_ + _b_ _[·][ I][a]_ [ +] _a_ + _b_ _[·][ I][b]_ [) +] _[ c][ ·][ I][c]_

= ( _a_ + _b_ + _c_ ) _·_        - _a_ + _b_ _a_ _b_ _c_        - (51)
_a_ + _b_ + _c_ _[·]_ [ (] _a_ + _b_ _[I][a]_ [ +] _a_ + _b_ _[I][b]_ [) +] _a_ + _b_ + _c_ _[·][ I][c]_


Thus, **the linear combination of multiple model outputs can be viewed as a composition of Self**
**Guidences** .


14


C REPRESENT SAMPLING METHODS WITH NATURAL INFERENCE

FRAMEWORK


C.1 REPRESENT DDPM ANCESTRAL SAMPLING WITH NATURAL INFERENCE FRAMEWORK


This subsection will demonstrate that the DDPM Ancestor Sampling can be reformulated within the
Natural Inference framework. The iterative process of the Ancestor Sampling is as follows:


_yt_ = _ft_ ( _xt_ )


= ( _dT −_ 2 _eT −_ 1 _yT_ + _eT −_ 2 _yT_ _−_ 1) + ( _dT −_ 2 _dT −_ 1 _gT_ _ϵT_ + _dT −_ 2 _gT −_ 1 _ϵT_ _−_ 1 + _gT −_ 2 _ϵT_ _−_ 2)
(54)


Based on the expression of _xT −_ 2, the expression of _xT −_ 3 can be further written as


_xT −_ 3 = _dT −_ 3 _· xT −_ 2 + _eT −_ 3 _· yT_ _−_ 2 + _gT −_ 3 _· ϵT_ _−_ 3


= ( _dT −_ 3 _dT −_ 2 _eT −_ 1 _yT_ + _dT −_ 3 _eT −_ 2 _yT_ _−_ 1 + _eT −_ 3 _yT_ _−_ 2)


+ ( _dT −_ 3 _dT −_ 2 _dT −_ 1 _gT_ _ϵT_ + _dT −_ 3 _dT −_ 2 _gT −_ 1 _ϵT_ _−_ 1 + _dT −_ 3 _gT −_ 2 _ϵT_ _−_ 2 + _gT −_ 3 _ϵT_ _−_ 3)
(55)


Similarly, each _xt_ can be recursively written in a similar form. It can be observed that each _xt_ can
be decomposed into two parts: one part is a weighted sum of past predictions of _x_ 0 (i.e., _yt_ ), and
the other part is a weighted sum of past noise and newly added noise. Since _dt_, _et_, and _gt_ are all
known constants, the equivalent signal coefficient and equivalent noise coefficient for each _xt_ can be
accurately computed.

The computation results show that the equivalent signaland the equivalent noise coefficient is approximately _[√]_ coeff1 _−_ _α_ ¯ _t_ c.ient of eachMoreover, the slight error diminishes _xt_ is almost equal to _[√]_ _α_ ¯ _t_,
as the number of sampling steps _T_ increases. Specifically, Figure 7 illustrates the results for 18 steps,
100 steps, and 500 steps.


Table 3 presents the complete coefficients of each _xt_ with respect to _yt_ in matrix form, where each
row corresponds to an _xt_ . Table 4 provides the complete coefficients of each _xt_ with respect to _ϵt_ in
matrix form. It can be seen that the noise coefficient matrix differs slightly from the signal coefficient
matrix, with an additional nonzero coefficient appearing to the right of the diagonal elements. This
indicates that a small amount of new noise is introduced at each step, causing the overall noise _pattern_
to change at a slow rate.


At this point, we have successfully demonstrated that the DDPM Ancestral Sampling process can be
represented using the Natural Inference framework.


15


_xt−_ 1 = _dt−_ 1 _· xt_ + _et−_ 1 _· yt_ + _gt−_ 1 _· ϵt−_ 1


(52)


_√_
_αt_ (1 _−_ _α_ ¯ _t−_ 1)
where _dt−_ 1 =


_√_

(1 _−_ _α_ ¯ _t−_ 1) _α_ ¯ _t−_ 1 _βt_

_et−_ 1 =
1 _−_ _α_ ¯ _t_ 1 _−_ _α_ ¯ _t_


_α_ ¯ _t−_ 1 _βt_ - 1 _−_ _α_ ¯ _t−_ 1

_gt−_ 1 =
1 _−_ _α_ ¯ _t_ 1 _−_ _α_ ¯ _t_


_βt_
1 _−_ _α_ ¯ _t_


Here, _ft_ is the model function at the _t_ -th step. In this case, we assume the model predicts _x_ 0, but
other forms of prediction models (such as predict _ε_ or predict _v_ ) can be transformed into the form of
predicting _x_ 0. _yt_ is the output of _ft_, which is the predicted _x_ 0 at the _t_ -th step.


According to the above iterative algorithm, _xT −_ 1 can be expressed as


_xT_ = _gT_ _· ϵT_ where _gT_ = 1


_xT −_ 1 = _dT −_ 1 _· xT_ + _eT −_ 1 _· yT_ + _gT −_ 1 _· ϵT_ _−_ 1


= _eT −_ 1 _yT_ + ( _dT −_ 1 _gT_ _ϵT_ + _gT −_ 1 _ϵT_ _−_ 1)


Based on the expression of _xT −_ 1, the expression of _xT −_ 2 can be written as


_xT −_ 2 = _dT −_ 2 _· xT −_ 1 + _eT −_ 2 _· yT_ _−_ 1 + _gT −_ 2 _· ϵT_ _−_ 2


(53)


(a) (b) (c)


Figure 7: DDPM equivalent marginal coefficients and ideal margingal coefficients (a) 18 step (b)
100 step (c) 500 step


C.2 REPRESENT DDIM WITH NATURAL INFERENCE FRAMEWORK


The iterative rule of the DDIM can be expressed in the following form:


It can be seen that the form of DDIM is slightly different from DDPM. Since DDIM does not
introduce new noise at each step, there is only one noise term.

The compequalcoefficient is almost equal toto _[√]_ utat _α_ ¯ _t_,ion results show that the equivalent signal coefficients of eachand the equiv _[√]_ alent1 _−_ _α_ ¯no _t_ .iseFigure 8 illustrates the results for 18 steps, 100 steps, and 500coefficient contains only the term _x_ related _t_ are approximatelyto _ϵT_, whose
steps, respectively. It can be observed that the errors in the equivalent coefficients are minimal and
almost indistinguishable. Table 5 presents the complete signal coefficient matrix for 18 steps.


Therefore, the sampling process of DDIM can also be represented using the Natural Inference
framework.


C.3 REPRESENT FLOW MATCHING EULER SAMPLING WITH NATURAL INFERENCE
FRAMEWORK


The noise mixing method of Flow Matching is shown in Equation equation 2. When using Euler
discretized integral sampling, its iterative rule can be expressed as follows:


16


_yt_ = _ft_ ( _xt_ )

_xt−_ 1 = _[√]_ _α_ ¯ _t−_ 1 _· xt_ + ~~�~~ 1 _−_ _α_ ¯ _t−_ 1 _·_ _[x][t]_ ~~_√_~~ _[ −]_ _[√][α]_ [¯] _[t][y][t]_ _· yt_
1 _−_ _α_ ¯ _t_

= _dt−_ 1 _· xt_ + _et−_ 1 _· yt_

_√_ _√_
where _dt−_ 1 = ~~_√_~~ 1 _−_ _α_ ¯ _t−_ 1 _et−_ 1 = ( _[√]_ _α_ ¯ _t−_ 1 _−_ ~~_√_~~ 1 _−_ _α_ ¯ ~~_√_~~ _t−_ 1 )
1 _−_ _α_ ¯ _t_ 1 _−_ _α_ ¯ _t_ _α_ ¯ _t_


(56)


It can be seen that the iterative rule of DDIM is similar to those of DDPM Ancestral Sampling, except
that the term _gt−_ 1 _· ϵt−_ 1 is missing, meaning that no new noise is added at each step. Following the
recursive way of DDPM Ancestral Sampling, each _xt_ corresponding to _t_ can also be written in a
similar form, as follows:


_xT_ = _gT_ _· ϵT_ where _gT_ = 1


_xT −_ 1 = _eT −_ 1 _yT_ + _dT −_ 1 _gT_ _ϵT_

_xT −_ 2 = ( _dT −_ 2 _eT −_ 1 _yT_ + _eT −_ 2 _yT_ _−_ 1) + _dT −_ 2 _dT −_ 1 _gT_ _ϵT_

_xT −_ 3 = ( _dT −_ 3 _dT −_ 2 _eT −_ 1 _yT_ + _dT −_ 3 _eT −_ 2 _yT_ _−_ 1 + _eT −_ 3 _yT_ _−_ 2)


+ _dT −_ 3 _dT −_ 2 _dT −_ 1 _gT_ _ϵT_


(57)


(a) (b) (c)


Figure 8: DDIM equivalent marginal coefficients and ideal marginal coefficients (a) 18 step (b)
100 step (c) 500 step


(a) (b) (c)


Figure 9: Flow matching euler sampler equivalent marginal coefficients and ideal marginal coefficients
(a) 18 step (b) 100 step (c) 500 step


_yi_ = _fi_ ( _xi_ )


_xi−_ 1 = _xi_ + ( _ti−_ 1 _−_ _ti_ )( _−yi_ + _ϵ_ )


where _fi_ is the model predicting _x_ 0, and _yi_ is the output of the model _fi_ corresponding to the discrete
time point _ti_ .


It can be observed that the iterative rule of the Euler algorithm in Flow Matching is similar to that of
DDIM, so each _xi_ can also be expressed in a similar form.


The computation results show that for each discrete point _ti_, the equivalent signal coefficient of _xi_ is
**exactly equal to** 1 _−_ _ti_, and the equivalent noise coefficient has only the _ϵN_ term, whose coefficient
is **exactly equal to** _ti_ . The specific results can be seen in Figure 9, which shows the results for 18
steps, 200 steps, and 500 steps, respectively. Table 6 presents the signal coefficient matrix for 18
steps.


C.4 REPRESENT HIGH ORDER SAMPLERS WITH NATURAL INFERENCE FRAMEWORK


In the previous sections, most first-order sampling algorithms have already been represented using
the Natural Inference framework. For second-order and higher-order sampling algorithms, since the
update rules of _xi_ are more complex, it is quite difficult to directly compute the expression of each
_xi_ . Therefore, alternative solutions must be sought. Symbolic computation tools provide a suitable
solution to this challenge, as they can automatically analyze complex mathematical expressions. With


17


= _xi_ + ( _ti−_ 1 _−_ _ti_ ) _[x][i][ −]_ _[y][i]_

_ti_
= _di−_ 1 _· xi_ + _ei−_ 1 _· yi_


(58)


where _di−_ 1 = _[t][i][−]_ [1]


)
_tti_


_[i][−]_ [1]

_tti_ _ei−_ 1 = (1 _−_ _[t]_ _t_ _[i][−]_ _ti_ [1]


(a) (b) (c)


Figure 10: DEIS equivalent marginal coefficients and ideal marginal coefficients (a) 18 step (b)
100 step (c) 500 step


(a) (b) (c)


Figure 11: dpmsolver3s equivalent marginal coefficients and ideal marginal coefficients (a) 18 step
(b) 99 step (c) 201 step


slight modifications to the original algorithm code, they can automatically compute the coefficients
of each _yi_ term and _ϵi_ term. The toolkit used in this paper is SymPyTeam (2013), and For specific
details, please refer to the code attached in this paper.


The compuation results show that DEIS, DPMSolver, and DPMSolver++ yield the same conclusion as
DDIM: each _xi_ can be decomposed into two parts, with its equivalent signal coeffcient approximately
equal to _[√]_ _α_ ¯ _i_ and its equivalent noise coefficient approximately equal to _[√]_ 1 _−_ _α_ ¯ _i_ .


Figure 10 shows the results for the DEIS(tab3) algorithm, Figure 11 presents the results for the
third-order DPMSolver, and Figure 12 illustrates the results for the second-order DPMSolver++. It
can be observed that these higher-order sampling algorithms exhibit the same properties and can also
be represented using the Natural Inference framework.


Table 7 provides the coefficient matrix for the third-order DEIS algorithm (18 steps). Table 8 and
Table 9 present the coefficient matrices for the second-order and third-order DPMSolver algorithms
(18 steps). Table 10 and Table 11 provide the coefficient matrices for the second-order and third-order
DPMSolver++ algorithms (18 steps).


C.5 REPRESENT SDE EULER AND ODE EULER WITH NATURAL INFERENCE FRAMEWORK


For SDE Euler and ODE Euler, the expressions for each _xt_ can also be automatically computed using
SymPy. The compuation results indicate that these two algorithms yield results similar to previous
algorithms, but they suffer from relatively larger errors, especially when the number of steps is small.
For details, see Figure 13 and Figure 14.


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


(a) (b) (c)


Figure 12: dpmsolver++2s equivalent marginal coefficients and ideal marginal coefficients (a) 18
step (b) 99 step (c) 201 step


(a) (b) (c)


Figure 13: SDE Euler equivalent marginal coefficients and ideal marginal coefficients (a) 18 step
(b) 50 step (c) 200 step


(a) (b) (c)


Figure 14: ODE Euler equivalent marginal coefficients and ideal marginal coefficients (a) 18 step
(b) 50 step (c) 200 step


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


C.6 A TOY EXAMPLE - REPRESENT FLOW MATCHING’S EULER SAMPLING WITH NATURAL
INFERENCE


For the inference method of Flow Matching, it is equivalent to solving the following ODE:
_d_ **x** _t_

**x** _t_ = (1 _−_ _t_ ) _·_ **x** 0 + _t · ε._ (59)
_dt_ [=] _[ µ][θ]_ [(] **[x]** _[t][, t]_ [)] _[,]_


Here, _µθ_ ( **x** _t, t_ ) is a neural network model, which is trained to fit the degenerated target ( _ε −_ **x** 0).


Considering that **x** _t_ can be written in the form of Eq. (2), and that _µθ_ ( **x** _t, t_ ) is a model predicting
( _ε −_ **x** 0), we have
**x** _t_ = (1 _−_ _t_ ) _·_ **x** 0 + _t · ε_ = **x** 0 + _t ·_ ( _ε −_ **x** 0) _._ (60)


Then, **x** _t −_ _t · µθ_ ( **x** _t, t_ ) is a function that predicts **x** 0, denoted as _fθ_ **[x]** [0][(] **[x]** _[t][, t]_ [)][, i.e.,]
_fθ_ **[x]** [0][(] **[x]** _[t][, t]_ [) =] **[ x]** _[t][ −]_ _[t][ ·][ µ][θ]_ [(] **[x]** _[t][, t]_ [)] _[.]_ (61)

It can be observed that _fθ_ **[x]** [0][(] **[x]** _[t][, t]_ [)][ is also a function of] _[ t]_ [ and] **[ x]** _[t]_ [.] [For convenience of notation, we]
abbreviate _fθ_ **[x]** [0][(] **[x]** _[t][, t]_ [)][ as] _[ f]_ _t_ **[ x]** [0][.]


C.6.1 A SIMPLE EULER INFERENCE EXAMPLE


The Euler method is the most basic approach for solving ODEs. Below we illustrate it with a 5-step example. We discretize the continuous interval _t ∈_ [1 _,_ 0] into 6 time points: [1 _,_ 0 _._ 8 _,_ 0 _._ 6 _,_ 0 _._ 4 _,_ 0 _._ 2 _,_ 0 _._ 0].


First, we randomly initialize a noise sample (denoted as _εs_ ) as the starting point of integration(i.e.
_x_ ˆ1 _._ 0 = _εs_ ). We substitute _t_ = 1 _._ 0 and _xt_ = _x_ ˆ1 _._ 0 into the model function _µθ_ ( _xt, t_ ), run the model,
and obtain the **velocity field** at _t_ = 1 _._ 0, denoted by _µθ_ (ˆ _x_ 1 _._ 0 _,_ 1 _._ 0).


We then perform **the first** update:
**x** ˆ0 _._ 8 = **x** ˆ1 _._ 0 + (0 _._ 8 _−_ 1 _._ 0) _· µθ_ (ˆ **x** 1 _._ 0 _,_ 1 _._ 0) (62)

= **x** ˆ1 _._ 0 _−_ 0 _._ 2 _·_ **[x]** [ˆ][1] _[.]_ [0] _[ −]_ _[f]_ 1 **[ x]** _._ [0] 0 (63)
1 _._ 0

= 0 _._ 8 _· εs_ + 0 _._ 2 _· f_ 1 **[x]** _._ [0] 0 _[.]_ (64)


From equation 64, we can see that _x_ ˆ0 _._ 8 consists of two components: the **initial random noise** and
the **model output predicting** _x_ 0, with coefficients 0 _._ 8 and 0 _._ 2, respectively. Moreover, we know that
during training, the constructed _x_ 0 _._ 8 satisfies
_x_ 0 _._ 8 = 0 _._ 8 _· ε_ + 0 _._ 2 _· x_ 0 _._ (65)


Comparing the two expressions, we observe that _x_ ˆ0 _._ 8 **and** _x_ 0 _._ 8 **have very similar structures:** **both**
**are composed of noise and signal with identical proportions; the only difference is that** _**predicted**_
_x_ 0 **replaces** _**original**_ _x_ 0 **.**


We now perform **the second** update:


Substitute _t_ = 0 _._ 8 and **x** _t_ = **x** ˆ0 _._ 8 into the model function _µθ_ ( **x** _t, t_ ), run the model, and obtain the
velocity field at _t_ = 0 _._ 8, denoted as _µθ_ (ˆ **x** 0 _._ 8 _,_ 0 _._ 8).


**x** ˆ0 _._ 6 = **x** ˆ0 _._ 8 + (0 _._ 6 _−_ 0 _._ 8) _· µθ_ (ˆ **x** 0 _._ 8 _,_ 0 _._ 8) (66)

= **x** ˆ0 _._ 8 _−_ 0 _._ 2 _·_ **[x]** [ˆ][0] _[.]_ [8] _[ −]_ _[f]_ 0 **[ x]** _._ [0] 8 (67)
0 _._ 8

= 0 _._ 75 _·_ ˆ **x** 0 _._ 8 + 0 _._ 25 _· f_ 0 **[x]** _._ [0] 8 (68)


Replacing **x** ˆ0 _._ 8 in equation 68 using equation 64, we obtain:
**x** ˆ0 _._ 6 = 0 _._ 75 _·_ ˆ **x** 0 _._ 8 + 0 _._ 25 _· f_ 0 **[x]** _._ [0] 8 (69)
= 0 _._ 75 _·_ (0 _._ 8 _· εs_ + 0 _._ 2 _· f_ 1 **[x]** _._ [0] 0 [) + 0] _[.]_ [25] _[ ·][ f]_ 0 **[ x]** _._ [0] 8 (70)


     - 0 _._ 15
= 0 _._ 6 _· εs_ + 0 _._ 4 _·_ 1 _._ 0 [+] [0] _[.]_ [25] 0 _._ 8
0 _._ 4 _[·][ f]_ **[ x]** [0] 0 _._ 4 _[·][ f]_ **[ x]** [0]


20


(71)


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


The expression inside the parentheses in equation 71 can be viewed as a **linear interpolation of** _f_ 1 _[x]_ [ˆ] _._ [0] 0
**and** _f_ 0 _[x]_ [ˆ] _._ [0] 8 [, which can also be interpreted as a form of Self Guidance.]


Again, we observe that **the structure of** ˆ _x_ 0 _._ 6 **is identical to that of** _x_ 0 _._ 6 **constructed during training:**
**both consist of noise and signal with the same proportions; the only difference is that** _x_ 0 _._ 6 **uses**
**the** _**original**_ _x_ 0 **,** **while** _x_ ˆ0 _._ 6 **uses** **a** **linear** **weighted** **sum** **of** _**previous**_ _**predicted**_ _x_ 0 **as** **its** **signal**
**component.**


We now proceed to **the third** update:


**x** ˆ0 _._ 4 = **x** ˆ0 _._ 6 + (0 _._ 4 _−_ 0 _._ 6) _· µθ_ (ˆ **x** 0 _._ 6 _,_ 0 _._ 6) (72)

= **x** ˆ0 _._ 6 _−_ 0 _._ 2 _·_ **[x]** [ˆ][0] _[.]_ [6] _[ −]_ _[f]_ 0 **[ x]** _._ [0] 6 (73)
0 _._ 6


We now proceed to **the fifth** update:


**x** ˆ0 _._ 0 = **x** ˆ0 _._ 2 + (0 _._ 4 _−_ 0 _._ 2) _· µθ_ (ˆ **x** 0 _._ 2 _,_ 0 _._ 2) (84)

= **x** ˆ0 _._ 2 _−_ 0 _._ 2 _·_ **[x]** [ˆ][0] _[.]_ [2] _[ −]_ _[f]_ 0 **[ x]** _._ [0] 2 (85)
0 _._ 2

= 0 _._ 0 _· εs_ + 1 _._ 0 _· f_ 0 **[x]** _._ [0] 2 (86)


It can be seen that the compositions of **x** ˆ0 _._ 4, **x** ˆ0 _._ 2, and **x** ˆ0 _._ 0 all share the same structural properties.


Up to this point, we have expressed the Euler inference method in the form of Natural Inference.


D COEFFICIENT MATRIXES


D.1 DDPM COEFFICIENT MATRIX


21


[2] [1]

3 _[·]_ [ ˆ] **[x]** [0] _[.]_ [6][ +] 3


= [2]

3

= [2]


3 _[·][ f]_ 0 **[ x]** _._ [0] 6 (75)


3 _[·][ f]_ 0 **[ x]** _._ [0] 6 (74)


3 [2] [(0] _[.]_ [6] _[ ·][ ε][s]_ [ + 0] _[.]_ [15] _[ ·][ f]_ 1 **[ x]** _._ [0] 0 [+ 0] _[.]_ [25] _[ ·][ f]_ 0 **[ x]** _._ [0] 8 [) +] [1] 3


[3] 1 _._ 0 [+] [5]

30 _[·][ f]_ **[ x]** [0]


(76)
30 _[·][ f]_ [0] _[.]_ [6]


= 0 _._ 4 _· εs_ + [3]


[5] 0 _._ 8 [+] [10]

30 _[·][ f]_ **[ x]** [0] 30


= 0 _._ 4 _· εs_ + 0 _._ 6 _·_ - 3 1 _._ 0 [+] [5]
18 _[·][ f]_ **[ x]** [0]


18 _[·][ f]_ [0] _[.]_ [6]


[5] 0 _._ 8 [+] [10]

18 _[·][ f]_ **[ x]** [0] 18


(77)


We now proceed to **the fourth** update:


**x** ˆ0 _._ 2 = **x** ˆ0 _._ 4 + (0 _._ 2 _−_ 0 _._ 4) _· µθ_ (ˆ **x** 0 _._ 4 _,_ 0 _._ 4) (78)

= **x** ˆ0 _._ 4 _−_ 0 _._ 2 _·_ **[x]** [ˆ][0] _[.]_ [4] _[ −]_ _[f]_ 0 **[ x]** _._ [0] 4 (79)
0 _._ 4


= [1]

2

= [1]


[1] [1]

2 _[·]_ [ ˆ] **[x]** [0] _[.]_ [4][ +] 2


[3] 1 _._ 0 [+] [5]

30 _[·][ f]_ **[ x]** [0] 30


2 _[·][ f]_ 0 **[ x]** _._ [0] 4 (80)


 
[1] 0 _._ 4 _· εs_ + [3]

2 _[·]_ 30


[5] 0 _._ 8 [+] [10]

30 _[·][ f]_ **[ x]** [0] 30


30 _[·][ f]_ [0] _[.]_ [6]


+ [1] 2 _[·][ f]_ 0 **[ x]** _._ [0] 4 (81)


+ [1]


= 0 _._ 2 _· εs_ + [3]


[3] 1 _._ 0 [+] [5]

60 _[·][ f]_ **[ x]** [0]


[5] 0 _._ 8 [+] [10]

60 _[·][ f]_ **[ x]** [0] 60


[10] [30]

60 _[·][ f]_ [0] _[.]_ [6][ +] 60


60 _[·][ f]_ 0 **[ x]** _._ [0] 4 (82)


     - 3
= 0 _._ 2 _· εs_ + 0 _._ 8 _·_ 1 _._ 0 [+] [5]
48 _[·][ f]_ **[ x]** [0]


[10] [30]

48 _[·][ f]_ [0] _[.]_ [6][ +] 48


[5] 0 _._ 8 [+] [10]

48 _[·][ f]_ **[ x]** [0] 48


48 _[·][ f]_ 0 **[ x]** _._ [0] 4


(83)


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


Table 3: DDPM’s signal coefficient matrix on Natural Inference framework


**time** **sum**


0.008 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.008
0.005 0.013 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.017
0.003 0.008 0.02 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.031
0.002 0.005 0.013 0.032 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.051
0.001 0.003 0.008 0.02 0.047 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.079
0.001 0.002 0.005 0.013 0.031 0.067 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.119
0.0 0.001 0.004 0.009 0.021 0.046 0.09 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.172
0.0 0.001 0.003 0.006 0.015 0.032 0.062 0.12 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.24
0.0 0.001 0.002 0.005 0.01 0.022 0.044 0.085 0.154 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.323
0.0 0.0 0.001 0.003 0.007 0.016 0.031 0.06 0.109 0.192 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.42
0.0 0.0 0.001 0.002 0.005 0.011 0.022 0.042 0.076 0.135 0.232 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.526
0.0 0.0 0.001 0.002 0.003 0.007 0.015 0.028 0.051 0.091 0.156 0.284 0.0 0.0 0.0 0.0 0.0 0.0 0.639
0.0 0.0 0.0 0.001 0.002 0.005 0.009 0.018 0.033 0.057 0.099 0.18 0.345 0.0 0.0 0.0 0.0 0.0 0.749
0.0 0.0 0.0 0.001 0.001 0.003 0.005 0.01 0.018 0.032 0.056 0.101 0.195 0.426 0.0 0.0 0.0 0.0 0.849
0.0 0.0 0.0 0.0 0.001 0.001 0.002 0.005 0.008 0.015 0.026 0.047 0.09 0.196 0.536 0.0 0.0 0.0 0.927
0.0 0.0 0.0 0.0 0.0 0.0 0.001 0.001 0.002 0.004 0.007 0.013 0.024 0.053 0.145 0.728 0.0 0.0 0.98
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.002 0.998 0.0 1.0
**-01** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0


Table 4: DDPM’s noise coefficient matrix on Natural Inference framework


**time** **-01** **norm**


0.561 0.828 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.326 0.481 0.814 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.197 0.292 0.494 0.795 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.999
0.123 0.181 0.307 0.494 0.782 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.999
0.079 0.117 0.197 0.318 0.502 0.763 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.997
0.052 0.077 0.131 0.211 0.333 0.506 0.741 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.993
0.036 0.053 0.09 0.144 0.228 0.347 0.508 0.712 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.985
0.025 0.037 0.062 0.1 0.159 0.241 0.353 0.496 0.687 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.971
0.018 0.026 0.044 0.071 0.112 0.17 0.249 0.349 0.485 0.653 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.946
0.012 0.018 0.031 0.05 0.079 0.12 0.176 0.247 0.342 0.462 0.613 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.907
0.009 0.013 0.022 0.035 0.056 0.084 0.123 0.173 0.24 0.324 0.43 0.564 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.85
0.006 0.009 0.015 0.024 0.037 0.057 0.083 0.117 0.162 0.218 0.29 0.38 0.513 0.0 0.0 0.0 0.0 0.0 0.0 0.769
0.004 0.005 0.009 0.015 0.024 0.036 0.053 0.074 0.102 0.138 0.183 0.24 0.324 0.449 0.0 0.0 0.0 0.0 0.0 0.662
0.002 0.003 0.005 0.008 0.013 0.02 0.03 0.042 0.058 0.078 0.103 0.135 0.183 0.253 0.375 0.0 0.0 0.0 0.0 0.529
0.001 0.001 0.002 0.004 0.006 0.009 0.014 0.019 0.027 0.036 0.048 0.062 0.084 0.117 0.173 0.285 0.0 0.0 0.0 0.375
0.0 0.0 0.001 0.001 0.002 0.003 0.004 0.005 0.007 0.01 0.013 0.017 0.023 0.032 0.047 0.077 0.173 0.0 0.0 0.201
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.01 0.0 0.01
**-01** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.00


D.2 DDIM COEFFICIENT MATRIX


Table 5: DDIM’s signal coefficient matrix on the Natural Inference framework


**time** **sum**


0.005 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.005
0.005 0.008 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.013
0.005 0.008 0.013 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.026
0.005 0.008 0.013 0.019 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.045
0.005 0.008 0.013 0.019 0.028 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.074
0.005 0.008 0.013 0.019 0.028 0.04 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.113
0.005 0.008 0.012 0.019 0.028 0.04 0.053 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.166
0.005 0.008 0.012 0.019 0.028 0.039 0.052 0.07 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.234
0.005 0.008 0.012 0.018 0.027 0.038 0.051 0.069 0.089 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.317
0.005 0.007 0.011 0.018 0.026 0.037 0.049 0.066 0.086 0.111 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.415
0.004 0.007 0.011 0.017 0.024 0.034 0.046 0.062 0.08 0.104 0.132 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.521
0.004 0.006 0.01 0.015 0.022 0.031 0.041 0.056 0.073 0.094 0.12 0.163 0.0 0.0 0.0 0.0 0.0 0.0 0.634
0.003 0.005 0.008 0.013 0.019 0.027 0.036 0.048 0.063 0.081 0.103 0.14 0.199 0.0 0.0 0.0 0.0 0.0 0.745
0.003 0.004 0.007 0.01 0.015 0.021 0.029 0.038 0.05 0.065 0.082 0.112 0.159 0.25 0.0 0.0 0.0 0.0 0.845
0.002 0.003 0.005 0.007 0.011 0.015 0.02 0.027 0.035 0.046 0.058 0.08 0.113 0.177 0.325 0.0 0.0 0.0 0.924
0.001 0.002 0.003 0.004 0.006 0.008 0.011 0.015 0.019 0.025 0.031 0.043 0.06 0.095 0.174 0.483 0.0 0.0 0.978
0.0 0.0 0.0 0.0 0.0 0.0 0.001 0.001 0.001 0.001 0.002 0.002 0.003 0.005 0.009 0.024 0.951 0.0 1.0
**-01** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0


D.3 FLOW MATCHING COEFFICIENT MATRIX


22


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


Table 6: Flow Matching Euler sampler’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.944** **0.889** **0.833** **0.778** **0.722** **0.667** **0.611** **0.556** **0.500** **0.444** **0.389** **0.333** **0.278** **0.222** **0.167** **0.111** **0.056** **sum**


**0.944** 0.056 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.056
**0.889** 0.052 0.059 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.111
**0.833** 0.049 0.055 0.062 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.167
**0.778** 0.046 0.051 0.058 0.067 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.222
**0.722** 0.042 0.048 0.054 0.062 0.071 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.278
**0.667** 0.039 0.044 0.05 0.057 0.066 0.077 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.333
**0.611** 0.036 0.04 0.046 0.052 0.06 0.071 0.083 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.389
**0.556** 0.033 0.037 0.042 0.048 0.055 0.064 0.076 0.091 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.444
**0.500** 0.029 0.033 0.038 0.043 0.049 0.058 0.068 0.082 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5
**0.444** 0.026 0.029 0.033 0.038 0.044 0.051 0.061 0.073 0.089 0.111 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.556
**0.389** 0.023 0.026 0.029 0.033 0.038 0.045 0.053 0.064 0.078 0.097 0.125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.611
**0.333** 0.02 0.022 0.025 0.029 0.033 0.038 0.045 0.055 0.067 0.083 0.107 0.143 0.0 0.0 0.0 0.0 0.0 0.0 0.667
**0.278** 0.016 0.018 0.021 0.024 0.027 0.032 0.038 0.045 0.056 0.069 0.089 0.119 0.167 0.0 0.0 0.0 0.0 0.0 0.722
**0.222** 0.013 0.015 0.017 0.019 0.022 0.026 0.03 0.036 0.044 0.056 0.071 0.095 0.133 0.2 0.0 0.0 0.0 0.0 0.778
**0.167** 0.01 0.011 0.012 0.014 0.016 0.019 0.023 0.027 0.033 0.042 0.054 0.071 0.1 0.15 0.25 0.0 0.0 0.0 0.833
**0.111** 0.007 0.007 0.008 0.01 0.011 0.013 0.015 0.018 0.022 0.028 0.036 0.048 0.067 0.1 0.167 0.333 0.0 0.0 0.889
**0.056** 0.003 0.004 0.004 0.005 0.005 0.006 0.008 0.009 0.011 0.014 0.018 0.024 0.033 0.05 0.083 0.167 0.5 0.0 0.944
**0.000** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0


D.4 DEIS COEFFICIENT MATRIX


Table 7: DEIS sampler’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.895** **0.796** **0.703** **0.616** **0.534** **0.459** **0.389** **0.324** **0.266** **0.213** **0.167** **0.126** **0.090** **0.061** **0.037** **0.019** **0.007** **sum**


**0.895** 0.011 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.011
**0.796** 0.002 0.033 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.034
**0.703** 0.014 -0.01 0.072 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.076
**0.616** -0.005 0.058 -0.043 0.13 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.14
**0.534** 0.014 -0.013 0.09 -0.046 0.183 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.229
**0.459** -0.004 0.054 -0.037 0.135 -0.046 0.235 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.337
**0.389** 0.011 -0.005 0.069 -0.02 0.165 -0.046 0.283 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.457
**0.324** -0.001 0.038 -0.015 0.093 -0.004 0.19 -0.047 0.324 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.577
**0.266** 0.007 0.004 0.041 0.004 0.105 0.009 0.209 -0.053 0.363 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.689
**0.213** 0.001 0.023 -0.001 0.055 0.017 0.113 0.016 0.223 -0.063 0.401 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.785
**0.167** 0.004 0.006 0.022 0.012 0.06 0.025 0.116 0.015 0.234 -0.076 0.441 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.86
**0.126** 0.001 0.013 0.003 0.03 0.018 0.062 0.026 0.117 0.009 0.245 -0.094 0.487 0.0 0.0 0.0 0.0 0.0 0.0 0.916
**0.090** 0.002 0.005 0.011 0.01 0.032 0.02 0.06 0.021 0.115 -0.003 0.257 -0.115 0.541 0.0 0.0 0.0 0.0 0.0 0.954
**0.061** 0.001 0.006 0.002 0.015 0.011 0.03 0.016 0.056 0.012 0.114 -0.02 0.271 -0.141 0.606 0.0 0.0 0.0 0.0 0.977
**0.037** 0.001 0.002 0.005 0.004 0.014 0.009 0.027 0.01 0.051 -0.0 0.112 -0.042 0.284 -0.173 0.687 0.0 0.0 0.0 0.99
**0.019** 0.0 0.002 0.001 0.006 0.004 0.012 0.005 0.022 0.002 0.045 -0.014 0.11 -0.066 0.292 -0.208 0.785 0.0 0.0 0.997
**0.007** 0.0 0.0 0.002 0.001 0.004 0.002 0.008 0.001 0.017 -0.005 0.039 -0.027 0.103 -0.088 0.285 -0.244 0.902 0.0 0.999
**0.001** -0.0 0.0 -0.0 0.001 -0.0 0.002 -0.001 0.005 -0.003 0.012 -0.012 0.033 -0.039 0.09 -0.111 0.262 -0.319 1.078 1.0


D.5 DPMSOLVER COEFFICIENT MATRIX


Table 8: DPMSolver2S’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.946** **0.889** **0.835** **0.778** **0.724** **0.667** **0.614** **0.556** **0.502** **0.445** **0.390** **0.334** **0.277** **0.223** **0.161** **0.112** **0.016** **sum**


**0.946** 0.005 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.005
**0.889** -0.008 0.021 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012
**0.835** -0.008 0.021 0.011 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.023
**0.778** -0.008 0.021 -0.017 0.045 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.041
**0.724** -0.008 0.021 -0.017 0.045 0.024 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.064
**0.667** -0.008 0.02 -0.017 0.045 -0.029 0.088 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.099
**0.614** -0.008 0.02 -0.017 0.045 -0.029 0.087 0.044 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.143
**0.556** -0.008 0.02 -0.017 0.045 -0.029 0.086 -0.044 0.149 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.203
**0.502** -0.008 0.02 -0.016 0.044 -0.028 0.085 -0.043 0.146 0.073 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.272
**0.445** -0.008 0.019 -0.016 0.042 -0.027 0.082 -0.042 0.142 -0.059 0.225 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.359
**0.390** -0.007 0.018 -0.015 0.04 -0.026 0.078 -0.04 0.135 -0.056 0.215 0.112 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.454
**0.334** -0.007 0.017 -0.014 0.038 -0.024 0.073 -0.037 0.126 -0.052 0.2 -0.077 0.318 0.0 0.0 0.0 0.0 0.0 0.0 0.559
**0.277** -0.006 0.015 -0.012 0.034 -0.022 0.065 -0.033 0.112 -0.047 0.179 -0.069 0.285 0.168 0.0 0.0 0.0 0.0 0.0 0.669
**0.223** -0.005 0.013 -0.011 0.029 -0.019 0.056 -0.028 0.097 -0.04 0.154 -0.059 0.245 -0.112 0.45 0.0 0.0 0.0 0.0 0.768
**0.161** -0.004 0.01 -0.008 0.022 -0.014 0.043 -0.022 0.074 -0.031 0.118 -0.046 0.188 -0.086 0.346 0.278 0.0 0.0 0.0 0.869
**0.112** -0.003 0.007 -0.006 0.016 -0.01 0.031 -0.016 0.054 -0.023 0.086 -0.033 0.137 -0.063 0.252 -0.235 0.735 0.0 0.0 0.932
**0.016** -0.001 0.001 -0.001 0.003 -0.002 0.006 -0.003 0.01 -0.004 0.015 -0.006 0.024 -0.011 0.045 -0.042 0.13 0.833 0.0 0.998
**0.001** -0.0 0.0 -0.0 0.0 -0.0 0.001 -0.0 0.002 -0.001 0.003 -0.001 0.004 -0.002 0.007 -0.007 0.022 -4.895 5.867 1.0


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


Table 9: DPMSolver3S’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.948** **0.892** **0.834** **0.782** **0.727** **0.667** **0.615** **0.560** **0.500** **0.447** **0.391** **0.334** **0.273** **0.217** **0.167** **0.044** **0.009** **sum**


**0.948** 0.004 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.004
**0.892** -0.004 0.016 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012
**0.834** 0.019 -0.033 0.037 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.024
**0.782** 0.019 -0.033 0.037 0.016 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.039
**0.727** 0.019 -0.033 0.037 -0.012 0.052 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.063
**0.667** 0.019 -0.033 0.037 0.049 -0.078 0.104 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.099
**0.615** 0.019 -0.033 0.037 0.049 -0.077 0.104 0.042 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.141
**0.560** 0.019 -0.032 0.036 0.048 -0.076 0.103 -0.024 0.125 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.198
**0.500** 0.019 -0.032 0.036 0.047 -0.075 0.101 0.093 -0.134 0.219 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.274
**0.447** 0.018 -0.031 0.035 0.046 -0.073 0.098 0.09 -0.13 0.213 0.089 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.356
**0.391** 0.017 -0.029 0.033 0.044 -0.069 0.093 0.086 -0.124 0.203 -0.04 0.238 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.452
**0.334** 0.016 -0.027 0.031 0.041 -0.064 0.087 0.08 -0.115 0.188 0.147 -0.191 0.368 0.0 0.0 0.0 0.0 0.0 0.0 0.559
**0.273** 0.014 -0.024 0.027 0.036 -0.057 0.077 0.071 -0.102 0.167 0.131 -0.17 0.327 0.178 0.0 0.0 0.0 0.0 0.0 0.675
**0.217** 0.012 -0.02 0.023 0.031 -0.049 0.065 0.06 -0.087 0.142 0.111 -0.144 0.277 -0.078 0.435 0.0 0.0 0.0 0.0 0.778
**0.167** 0.01 -0.017 0.019 0.025 -0.039 0.053 0.049 -0.07 0.115 0.09 -0.117 0.225 0.248 -0.336 0.605 0.0 0.0 0.0 0.859
**0.044** 0.003 -0.005 0.006 0.007 -0.012 0.016 0.015 -0.021 0.035 0.027 -0.035 0.068 0.074 -0.101 0.181 0.73 0.0 0.0 0.987
**0.009** 0.001 -0.001 0.001 0.002 -0.003 0.004 0.004 -0.006 0.009 0.007 -0.009 0.018 0.02 -0.027 0.048 -1.201 2.132 0.0 0.999
**0.001** 0.0 -0.0 0.0 0.001 -0.001 0.001 0.001 -0.001 0.002 0.002 -0.002 0.005 0.005 -0.007 0.013 6.607 -10.588 4.963 1.0


D.6 DPMSOLVER++ COEFFICIENT MATRIX


Table 10: DPMSolverpp2S’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.946** **0.889** **0.835** **0.778** **0.724** **0.667** **0.614** **0.556** **0.502** **0.445** **0.390** **0.334** **0.277** **0.223** **0.161** **0.112** **0.016** **sum**


**0.946** 0.005 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.005
**0.889** 0.0 0.012 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012
**0.835** 0.0 0.012 0.011 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.023
**0.778** 0.0 0.012 0.0 0.029 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.041
**0.724** 0.0 0.012 0.0 0.029 0.024 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.064
**0.667** 0.0 0.012 0.0 0.028 0.0 0.059 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.099
**0.614** 0.0 0.012 0.0 0.028 0.0 0.058 0.044 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.143
**0.556** 0.0 0.012 0.0 0.028 0.0 0.058 0.0 0.105 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.203
**0.502** 0.0 0.012 0.0 0.028 0.0 0.057 0.0 0.103 0.073 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.272
**0.445** 0.0 0.011 0.0 0.027 0.0 0.055 0.0 0.1 0.0 0.166 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.359
**0.390** 0.0 0.011 0.0 0.025 0.0 0.052 0.0 0.095 0.0 0.159 0.112 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.454
**0.334** 0.0 0.01 0.0 0.024 0.0 0.049 0.0 0.089 0.0 0.147 0.0 0.241 0.0 0.0 0.0 0.0 0.0 0.0 0.559
**0.277** 0.0 0.009 0.0 0.021 0.0 0.044 0.0 0.079 0.0 0.132 0.0 0.216 0.168 0.0 0.0 0.0 0.0 0.0 0.669
**0.223** 0.0 0.008 0.0 0.018 0.0 0.037 0.0 0.068 0.0 0.113 0.0 0.185 0.0 0.338 0.0 0.0 0.0 0.0 0.768
**0.161** 0.0 0.006 0.0 0.014 0.0 0.029 0.0 0.052 0.0 0.087 0.0 0.143 0.0 0.26 0.278 0.0 0.0 0.0 0.869
**0.112** 0.0 0.004 0.0 0.01 0.0 0.021 0.0 0.038 0.0 0.064 0.0 0.104 0.0 0.189 0.0 0.501 0.0 0.0 0.932
**0.016** 0.0 0.001 0.0 0.002 0.0 0.004 0.0 0.007 0.0 0.011 0.0 0.018 0.0 0.034 0.0 0.089 0.833 0.0 0.998
**0.001** 0.0 0.0 0.0 0.0 0.0 0.001 0.0 0.001 0.0 0.002 0.0 0.003 0.0 0.006 0.0 0.015 0.0 0.972 1.0


Table 11: DPMSolverpp3S’s signal coefficient matrix on Natural Inference framework


**time** **1.000** **0.948** **0.892** **0.834** **0.782** **0.727** **0.667** **0.615** **0.560** **0.500** **0.447** **0.391** **0.334** **0.273** **0.217** **0.167** **0.044** **0.009** **sum**


**0.948** 0.004 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.004
**0.892** 0.025 -0.014 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.012
**0.834** 0.046 0.0 -0.022 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.024
**0.782** 0.046 0.0 -0.022 0.016 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.039
**0.727** 0.046 0.0 -0.022 0.085 -0.045 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.063
**0.667** 0.046 0.0 -0.022 0.144 0.0 -0.068 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.099
**0.615** 0.045 0.0 -0.022 0.143 0.0 -0.068 0.042 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.141
**0.560** 0.045 0.0 -0.022 0.142 0.0 -0.067 0.211 -0.111 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.198
**0.500** 0.044 0.0 -0.021 0.139 0.0 -0.066 0.334 0.0 -0.156 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.274
**0.447** 0.043 0.0 -0.021 0.135 0.0 -0.064 0.325 0.0 -0.151 0.089 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.356
**0.391** 0.041 0.0 -0.02 0.129 0.0 -0.061 0.31 0.0 -0.144 0.415 -0.217 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.452
**0.334** 0.038 0.0 -0.018 0.119 0.0 -0.057 0.288 0.0 -0.134 0.6 0.0 -0.277 0.0 0.0 0.0 0.0 0.0 0.0 0.559
**0.273** 0.034 0.0 -0.016 0.106 0.0 -0.05 0.255 0.0 -0.119 0.533 0.0 -0.246 0.178 0.0 0.0 0.0 0.0 0.0 0.675
**0.217** 0.029 0.0 -0.014 0.09 0.0 -0.043 0.217 0.0 -0.101 0.452 0.0 -0.209 0.749 -0.393 0.0 0.0 0.0 0.0 0.778
**0.167** 0.023 0.0 -0.011 0.073 0.0 -0.035 0.176 0.0 -0.082 0.368 0.0 -0.17 0.962 0.0 -0.445 0.0 0.0 0.0 0.859
**0.044** 0.007 0.0 -0.003 0.022 0.0 -0.01 0.053 0.0 -0.025 0.11 0.0 -0.051 0.288 0.0 -0.133 0.73 0.0 0.0 0.987
**0.009** 0.002 0.0 -0.001 0.006 0.0 -0.003 0.014 0.0 -0.007 0.029 0.0 -0.013 0.076 0.0 -0.035 2.235 -1.304 0.0 0.999
**0.001** 0.0 0.0 -0.0 0.002 0.0 -0.001 0.004 0.0 -0.002 0.008 0.0 -0.004 0.02 0.0 -0.009 2.116 0.0 -1.134 1.0


E SD3’S COEFFICIENT MATRIX AND INFERENCE PROCESS VISUALIZATION


E.1 COEFFICIENT MATRIX AND ITS CORRESPONDING OUTPUTS


Note that, for readability, the coefficients in Table 12 are the original coefficients multiplied by 100.
When using them, they should be normalized to the corresponding Marginal Coefficient for each step.


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


Table 12: SD3’s signal coefficient matrix for Flow Matching Euler sampling


**time** **1.00** **0.99** **0.97** **0.96** **0.95** **0.93** **0.91** **0.90** **0.88** **0.86** **0.84** **0.81** **0.79** **0.76** **0.74** **0.71** **0.68** **0.64** **0.60** **0.56** **0.52** **0.46** **0.41** **0.35** **0.28** **0.20** **0.11** **0.01**


**0.99** 1.26 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.97** 1.26 1.33 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.96** 1.26 1.33 1.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.95** 1.26 1.33 1.4 1.47 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.93** 1.26 1.33 1.4 1.47 1.56 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.91** 1.26 1.33 1.4 1.47 1.56 1.65 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.90** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.88** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.86** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.84** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.81** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.79** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.76** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.74** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.71** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.68** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.64** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.60** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.56** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.52** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.46** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 0.0 0.0 0.0 0.0 0.0 0.0 0.0
**0.41** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 0.0 0.0 0.0 0.0 0.0 0.0
**0.35** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 0.0 0.0 0.0 0.0 0.0
**0.28** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 0.0 0.0 0.0 0.0
**0.20** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 0.0 0.0 0.0
**0.11** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 0.0 0.0
**0.01** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 10.2 0.0
**0.00** 1.26 1.33 1.4 1.47 1.56 1.65 1.74 1.85 1.97 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 10.2 0.89


For example, the first row should be normalized to 0.0126, and the second row should be normalized
to 0.0259. The usage of Table 13 is the same.


Table 13: SD3’s signal coefficient matrix with more sharpness


**time** **1.00** **0.99** **0.97** **0.96** **0.95** **0.93** **0.91** **0.90** **0.88** **0.86** **0.84** **0.81** **0.79** **0.76** **0.74** **0.71** **0.68** **0.64** **0.60** **0.56** **0.52** **0.46** **0.41** **0.35** **0.28** **0.20** **0.11** **0.01**


**0.99** 1.26 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.97** 1.26 1.33 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.96** 1.26 1.33 1.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.95** 0.0 1.33 1.4 1.47 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.93** 0.0 1.33 1.4 1.47 1.56 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.91** 0.0 0.0 1.44 1.56 1.56 1.65 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.90** 0.0 0.0 0.0 0.0 1.56 1.65 1.74 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.88** 0.0 0.0 0.0 0.0 0.0 1.65 1.74 1.85 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.86** 0.0 0.0 0.0 0.0 0.0 1.65 1.74 1.85 1.97 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.84** 0.0 0.0 0.0 0.0 0.0 0.0 1.74 1.85 1.97 2.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.81** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.85 1.97 2.1 2.24 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.79** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.97 2.1 2.24 2.4 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.76** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.74** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 2.76 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.71** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 2.76 2.98 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.68** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 2.76 2.98 3.22 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.64** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.60** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.1 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.56** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.52** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.46** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 0.0 0.0 0.0 0.0 0.0 0.0 0
**0.41** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.24 2.4 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 0.0 0.0 0.0 0.0 0.0 0
**0.35** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 0.0 0.0 0.0 0.0 0
**0.28** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 0.0 0.0 0.0 0
**0.20** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 0.0 0.0 0
**0.11** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.57 2.76 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 0.0 0
**0.01** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 10.2 0
**0.00** 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.98 3.22 3.49 3.8 4.15 4.56 5.02 5.56 6.19 6.93 7.82 8.89 10.2 15


E.2 INFERENCE PROCESS VISUALIZATION


Figures 15 and 16 provide a visualization of the complete inference process. The left half shows the
inference process using the coefficient matrix from Table 12, and the right half shows the inference
process using the coefficient matrix from Table 13. The first column shows the result of Self Guidance,
which is also the input image signal to the model. The second column shows the model output without
conditioning, the third column shows the conditioned model output, and the fourth column shows the
result of Classifier Free Guidance. For each model operation, there is a clear image signal input and
image signal output, which greatly enhances intuitive understanding of the operation’s purpose and
facilitates efficient debugging and problem analysis.


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


Figure 15: Inference process visualization: first half.


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


Figure 16: Inference process visualization: second half