# LEARNING ROBUST DIFFUSION MODELS FROM IMPRECISE SUPERVISION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Conditional diffusion models have achieved remarkable success in various generative tasks recently, but their training typically relies on large-scale datasets that
inevitably contain imprecise information in conditional inputs. Such supervision,
often stemming from noisy, ambiguous, or incomplete labels, will cause condition
mismatch and degrade generation quality. To address this challenge, we propose
_DMIS_, a unified framework for training robust Diffusion Models from Imprecise
Supervision, which is the first systematic study within diffusion models. Our
framework is derived from likelihood maximization and decomposes the objective
into generative and classification components: the generative component models
imprecise-label distributions, while the classification component leverages a diffusion classifier to infer class-posterior probabilities, with its efficiency further
improved by an optimized timestep sampling strategy. Extensive experiments
on diverse forms of imprecise supervision, covering tasks of image generation,
weakly supervised learning, and noisy dataset condensation demonstrate that _DMIS_
consistently produces high-quality and class-discriminative samples.


1 INTRODUCTION


Diffusion models (DMs) (Ho et al., 2020; Song et al., 2020; Karras et al., 2022) have emerged as
powerful generative frameworks that have unprecedented capabilities in generating realistic data
(He et al., 2025; Yang et al., 2024; Ho et al., 2022). With the classifier guidance (Ho & Salimans,
2022; Dhariwal & Nichol, 2021), conditional diffusion models (CDMs) extended the capabilities of
DMs by conditioning the generation process on additional information, such as text descriptions or
class labels. These models have demonstrated remarkable performance in various tasks, including
text-to-image synthesis (Rombach et al., 2022; Saharia et al., 2022), image inpainting (Zhao et al.,
2024; Corneanu et al., 2024), and super-resolution (Esser et al., 2024; Xie et al., 2025).


Unfortunately, the conditioning information required by CDMs is often imprecise in real-world
scenarios. When sourced from the internet or obtained through crowdsourcing, such information
can be affected by factors such as privacy constraints or limited annotator expertise, leading to
various imperfections. In particular, the conditioning data may contain noise, exhibit ambiguity, or
suffer from missing and incomplete annotations. We refer to such cases collectively as imprecise
supervision (Chen et al., 2024a), where the provided conditioning information is not fully aligned
with the true underlying labels. This includes scenarios such as noisy-label data (Li et al., 2017; Wei
et al., 2021), partial-label data (Wang et al., 2025b;a), and supplementary-unlabeled data (He et al.,
2023). These forms of imprecise supervision can introduce incorrect inductive biases during training
and severely affect the reliability and generalization of CDMs.


To address this, several recent studies have proposed adaptations of diffusion models to handle
imprecise supervision, such as noise-robust diffusion models (Na et al., 2024; Li et al., 2024) and
positive-unlabeled diffusion models (Takahashi et al., 2025). However, these approaches often focus
on specific types of imprecise supervision. Moreover, many of them rely on strong external priors to
guide the learning process. For example, Na et al. (2024) estimated a noise transition matrix using
external noisy-label learning methods, and Li et al. (2024) required risk confidence scores associated
with noisy samples. These diffusion-based methods not only rely on prior knowledge from data
or previous techniques, but are also designed with task-specific architectures for particular types
of supervision. Such reliance and structural complexity limit their applicability and efficiency in


1


practice. There remains a need for a unified framework that can robustly train CDMs under diverse
forms of imprecise supervision without requiring strong prior assumptions.


In this paper, to train a robust CDM in a unified manner, we first formulate the overall learning
objective as a likelihood maximization problem (Section 4.1). Then we decompose this objective
into a generative term that models the imprecise data distribution (Section 4.2) and a classification
term that infers posterior label probabilities from imprecise supervision (Section 4.3). During
generative modeling, we show that the imprecise-label conditional score can be expressed as a linear
combination of clean-label conditional scores, weighted by the corresponding posterior probabilities.
Building on this insight, we propose a weighted denoising score matching objective, which enables
the model to achieve label-conditioned learning without requiring clean annotations. Finally, to
reduce the time complexity of posterior inference, we further introduce an efficient timestep sampling
strategy (Section 5). Extensive experiments across multiple tasks, including image generation, weakly
supervised learning, and noisy dataset condensation show that CDMs trained with our framework
not only achieve strong generative quality but also produce class-discriminative samples. Our
contributions are summarized as follows:


    - We propose a unified diffusion framework for training CDMs under diverse forms of
imprecise supervision, which is the first exploration in the diffusion model field.


    - To improve efficiency, we develop an optimized timestep sampling strategy for diffusion
classifiers that greatly reduces the computation cost without compromising performance.


    - Building on this framework, we pioneer the study of noisy dataset condensation, a practical
yet previously unexplored setting, and establish a solid baseline for future research.


    - Extensive experiments on image generation, weakly supervised learning, and noisy dataset
condensation demonstrate the effectiveness and versatility of our unified framework.


2 RELATED WORK


2.1 ROBUST DIFFUSION MODELS


Training conditional diffusion models under limited or imperfect supervision is still relatively underexplored. Recent work has begun to address specific forms of weak or noisy information, such as
noise-robust diffusion models (Na et al., 2024; Li et al., 2024) that mitigate corrupted labels, and
positive-unlabeled diffusion models (Takahashi et al., 2025) that combine positive samples with
large unlabeled corpora to approximate conditional distributions. We take a different perspective:
rather than tailoring objectives to a single type of imperfect label, we formulate a unified conditional
score-learning framework that can be instantiated under multiple imprecise-label regimes.


2.2 IMPRECISE LABEL LEARNING


Imprecise label learning studies supervision that is incomplete, ambiguous, or corrupted relative to
clean ground-truth labels. Canonical settings include partial-label learning (Feng et al., 2020; Wu
et al., 2022; Tian et al., 2023; Lv et al., 2020; Wang et al., 2025b), where each instance is associated
with a candidate label set containing the true label; semi-supervised learning (Berthelot et al., 2019b;
Zhang et al., 2021a; Yang et al., 2022; Wang et al., 2022c), where only a subset of samples are
labeled; and noisy-label learning (Han et al., 2018; Wei et al., 2021; Han et al., 2020), where observed
labels are corrupted versions of the true labels. Beyond these settings, mixture imprecise-label
learning (Chen et al., 2024a; Zhang et al., 2020; Wei et al., 2023; Shukla et al., 2023; Xie et al., 2024)
combines several forms of imprecision in a single framework. Our work can be viewed as lifting
these ideas from discriminative prediction to conditional score modeling, providing a generative view
of learning with heterogeneous imprecise supervision.


2.3 DATASET CONDENSATION


Dataset distillation (DD) (Wang et al., 2018) compresses a large labeled dataset into a compact
synthetic set that preserves task-relevant information, thereby reducing training cost while maintaining
competitive accuracy. Bi-level optimization methods learn synthetic data whose training signals


2


match those of the original data via gradient, trajectory, or meta-model matching (Zhao et al., 2021;
Kim et al., 2022; Cazenavette et al., 2022a; Cui et al., 2023; Wang et al., 2018; Loo et al., 2022), often
achieving high fidelity at nontrivial computational cost. Distribution-matching approaches instead
align statistics in pixel, feature, or kernel space (Wang et al., 2022b; Sajedi et al., 2023; Xue et al.,
2025; Yin et al., 2024; Sun et al., 2024; Shao et al., 2024; Yin & Shen, 2024), enabling more scalable
DD. While DD primarily targets data efficiency under clean labels, our framework instead focuses on
robustness to imprecise supervision.


3 BACKGROUND


**Diffusion Models.** Let _X_ _⊆_ R _[d]_ denote the _d_ -dimensional input space. Given a clean input **x** := **x** 0
from the real data distribution with density _q_ ( **x** 0), the forward diffusion process corrupts the data
into a sequence of noisy samples _{_ **x** _t}_ _[T]_ _t_ =11 by gradually adding Gaussian noise with a fixed scaling
schedule _{αt}_ _[T]_ _t_ =1 [and a fixed noise schedule] _[ {][σ][t][}]_ _t_ _[T]_ =1 [, as defined by]

_q_ ( **x** _t |_ **x** 0) = _N_ ( **x** _t_ ; _αt_ **x** 0 _, σt_ [2] **[I]** [)] _[,]_ (1)

where **I** denotes the identity matrix and _N_ ( **x** ; _**µ**_ _,_ **Σ** ) denotes the Gaussian density with mean _**µ**_
and covariance matrix **Σ** . Assuming that the signal-to-noise ratio SNR( _t_ ) = _αt_ [2] _[/σ]_ _t_ [2] [decreases]
monotonically over time, the sample **x** _t_ becomes increasingly noisier during the forward process.
The scaling and noise schedules are prescribed such that **x** _T_ nearly follows an isotropic Gaussian
distribution. The reverse process for Eq. (1) is defined as a Markov chain, which aims to approximate
_q_ ( **x** 0) by gradually denoising from the standard Gaussian distribution _p_ ( **x** _T_ ) = _N_ ( **x** _T_ ; **0** _,_ **I** ):

_pθ_ ( **x** 0: _T_ ) = _p_ ( **x** _T_ )� _T_ (2)

_t_ =1 _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[ |]_ **[x]** _[t]_ [)] _[,]_

_pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ) = _N_              - **x** _t−_ 1; _**µ**_ _θ_ ( **x** _t, t_ ) _,_ _σ_ ˜ _t_ [2] **[I]**              - _,_ (3)


where _**µ**_ _θ_ is generally parameterized by a time-conditioned score prediction network **s** _θ_ ( **x** _t, t_ ) (Song
et al., 2020; 2021; Song & Ermon, 2019; 2020):


where E denotes the expectation, _wt_ = _σ_ 2 _t_ [2] [(] _σσtt_ [2][2] _−_ _[α]_ 1 _t_ [2] _−_ _[α]_ _t_ [2] 1 _[−]_ [1)][,] [and] _[C]_ [1] [is] [a] [constant] [that] [is] [typically]
small and can be dropped (Song et al., 2020). The expectation term is called the _score matching_
_loss_ (Kingma et al., 2021), where _∇_ log _qt_ ( **x** _t_ ) is the gradient of data density at **x** _t_ in data space.


The above definition can be reformulated to match other commonly used diffusion models, such as
those in Ho et al. (2020), Karras et al. (2022) and Song et al. (2020). The corresponding conversions
are detailed in Appendix B.1. For clarity, we adopt the the elucidated diffusion model (EDM) (Karras
et al., 2022) as the default diffusion model throughout this paper, as it offers a unified structure and
well-optimized parameterization.


**Imprecise Supervision.** Imprecise-label data typically refers to settings where the true label is
not directly available, and instead only imprecise label information is provided. Let _Y_ = [ _c_ ] :=
_{_ 1 _, . . ., c}_ represent the label space with _c_ distinct classes. In this work, we primarily focus on three
representative forms of imprecise supervision that have been widely studied in the literature:


- _Partial-label data_, where each instance _X_ is associated with a candidate label set _S_ _⊂_ [ _c_ ] that is
guaranteed to contain the true label _Y_, i.e., _p_ ( _Y_ _∈_ _S |_ _X, S_ ) = 1. This setting is widely studied in
partial-label learning (Tian et al., 2023).

- _Supplementary-unlabeled data_, consisting of a small labeled subset ( _X_ [l] _, Y_ [l] ) together with a large
number of unlabeled samples ( _X_ [u] _, ∅_ ). This scenario is the focus of semi-supervised learning (Yang
et al., 2022), which aims to exploit unlabeled data to improve generalization.


1We use the subscript _t_ of the sample **x** to denote the noisy version of the sample at timestep _t_ .


3


_**µ**_ _θ_ ( **x** _t, t_ ) = _[α][t][−]_ [1]

_αt_


- **x** _t_ + - _σt_ [2] _[−]_ _αt_ [2] _σt_ [2] _−_ 1� **s** _θ_ ( **x** _t, t_ )� _._ (4)
_αt_ [2] _−_ 1


The reverse process can be learned by optimizing the variational lower bound on log-likelihood as


        - 2
log _pθ_ ( **x** ) _≥−_ E _t_ _wt_ �� **s** _θ_ ( **x** _t, t_ ) _−∇_ **x** _t_ log _qt_ ( **x** _t_ )��2


+ _C_ 1 _,_ (5)


- _Noisy-label data_, where the observed label _Y_ [ˆ] is a corrupted version of the underlying true label
_Y_, modeled by a conditional distribution _p_ ( _Y_ [ˆ] _|_ _X, Y_ ). This gives rise to noisy-label learning (Han
et al., 2020), which seeks to build models robust to label corruption.


4 METHODOLOGY


In this section, we first introduce the unified learning objective that integrates generative and classification components. Then we elaborate on the formulation and optimization of these components.


4.1 UNIFIED LEARNING OBJECTIVE


To robustly learn a diffusion model with learnable parameters _θ_ under imprecise supervision (denoted
as _Z_ _⊆Y_ ), we treat the true label _Y_ as a latent variable and maximize the likelihood of the joint
distribution of the input _X_ and _Z_ . By the maximum likelihood principle, our objective is to find


_θ_ _[∗]_ = arg max _θ_ log _pθ_ ( _X, Z_ ) = arg max _θ_ log        - _Y_ _[p][θ]_ [(] _[X, Y, Z]_ [)] _[,]_ (6)

where _θ_ _[∗]_ denotes the optimal parameter. Eq. (6) involves the log of the marginalization over latent
variables and cannot generally be solved in closed form. To circumvent this intractability, we instead
maximize a variational lower bound on the marginal log-likelihood:


where _C_ 2 is another constant. Directly optimizing the score network with this objective on impreciselabel data would lead it to converge to the score of the imprecise conditional distribution.

**Remark 1.** Let _θ_ [ˆ] denote the parameters obtained by maximizing the lower bound in Eq. (8) using denoising score matching. In this case, the learned score function satisfies **s** _θ_ ˆ( **x** _t, z, t_ ) = _∇_ **x** _t_ log _qt_ ( **x** _t |_
_z_ ) for all **x** _t_ _∈X_, _z_ _⊆Y_, and _t ∈_ [ _T_ ]. However, since _qt_ ( **x** _t |_ _z_ ) corresponds to the imprecise-label
density, the resulting generation is biased and thus fails to fully recover the true data distribution. The
derivation and visualization of this bias is deferred to Appendix B.5.


Therefore, to align the learned score with the clean-label conditional score, we propose modifying
the objective to correct the gradient signal from score matching (Kingma et al., 2021). Building on
the linear relationship between clean- and noisy-label conditional scores modeled by Na et al. (2024),
we further derive an explicit relationship that connects imprecise-label conditional scores to their
clean-label counterparts.


4


_θ_ _[n]_ = arg max _θ_ E _pϕ_ ( _Y |X,Z_ )� log _pθ_ ( _X, Y, Z_ )�


= arg max
_θ_


- log _pθ_ ( _X |_ _Z_ ) + E _pϕ_ ( _Y |X,Z_ )� log _pθ_ ( _Y_ _|_ _X, Z_ )� [�] _,_ (7)


where _θ_ _[n]_ denotes the _n_ -th estimate of _θ_, and _ϕ_ is instantiated as the exponential moving average
(EMA) of _θ_ over its 1st through ( _n_ _−_ 1) iterates. A complete derivation of this variational lower
bound is provided in Appendix B.3. From Eq. (7), we can observe that maximizing the marginal
likelihood can be performed from generative and classification perspectives. The former focuses on
modeling the data distribution conditioned on the imprecise supervision, while the latter aims to infer
the posterior distribution based on the feature and the imprecise label. In this paper, we adopt the
commonly used class-conditional setting, where the generation of the imprecise label _Z_ is assumed
to be independent of the input _X_ given the true label _Y_ (Yao et al., 2020; Wen et al., 2021).


4.2 GENERATIVE OBJECTIVE: MODELING THE IMPRECISE DATA DISTRIBUTION


Since samples are assumed to be independent of each other, we present the analysis in this and the
following subsections using a single sample ( **x** _, z_ ) for notational clarity, with the final objective
computed over the entire dataset. Following the standard formulation of diffusion models in Eq. (5),
we parameterize the conditional generative process _pθ_ ( **x** _|_ _z_ ) using a score network **s** _θ_ ( **x** _t, z, t_ ). The
corresponding variational lower bound on the conditional log-likelihood is given by


         - 2
log _pθ_ ( **x** 0 _|_ _z_ ) _≥−_ E _t_ _wt_ �� **s** _θ_ ( **x** _t, z, t_ ) _−∇_ **x** _t_ log _qt|_ 0( **x** _t |_ **x** 0 _, z_ )��2


+ _C_ 2 _,_ (8)


**Theorem 1.** _Under the class-conditional setting, for all_ **x** _t_ _∈X_ _, z_ _⊆Y, and t ∈_ [ _T_ ] _,_

_∇_ **x** _t_ log _qt_ ( **x** _t |_ _z_ ) =      - _cy_ =1 _[p]_ [(] _[y]_ _[|]_ **[x]** _[t][, z]_ [)] _[ ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][ |]_ _[y]_ [)] _[.]_ (9)


The formal proof is in Appendix B.6. Since _p_ ( _y |_ **x** _t, z_ ) _≥_ 0 and [�] _y_ _[c]_ =1 _[p]_ [(] _[y][ |]_ **[ x]** _[t][, z]_ [) = 1][, Theorem 1]
implies that the imprecise-label conditional score can be expressed as a convex combination of the
clean-label conditional scores, weighted by _p_ ( _y_ _|_ **x** _t, z_ ). These weights represent the model’s posterior
probability over labels given **x** _t_ and _z_, implicitly requiring the model to perform classification during
training. To our knowledge, this is the first work to explicitly reveal and exploit the classification
capability of diffusion models within the training process under imprecise supervision.


According to Remark 1, directly optimizing the denoising score matching objective in Eq. (8) drives
the score network to approximate the imprecise-label conditional score. However, Theorem 1 shows
that this score can be decomposed as a convex combination of clean-label conditional scores, weighted
by the posterior probability _p_ ( _y_ _|_ **x** _t, z_ ). Motivated by this insight, we propose a new training objective
that supervises the clean-label score network **s** _θ_ ( **x** _t, y, t_ ) through a reweighted aggregation of its
posterior outputs. The resulting weighted denoising score matching loss is


This diffusion classifier can be extended to non-uniform priors by incorporating _p_ ( _y_ ) into the logits
of class _y_, where _p_ ( _y_ ) is estimated from the training set (Luo et al., 2024; Wang et al., 2022a), as
detailed in Appendix C.2. As training proceeds, the conditional ELBO converges towards the true
distribution _qt_ ( **x** _t |_ _y_ ), thereby yielding increasingly accurate posterior estimates. For convenience,
we denote the class probability of a noisy input **x** _t_ with the diffusion classifier as _f_ ( **x** _t_ ).


To derive the classification loss, we transform the maximization problem of the classification term in
Eq. (7) into the minimization of the negative log-likelihood. We show that the resulting objective,
i.e., _−_ [�] _Y_ _[p][ϕ]_ [(] _[Y]_ _[|]_ _[X, Z]_ [) log] _[ p][θ]_ [(] _[Y]_ _[|]_ _[X, Z]_ [)][, naturally aligns closely with prior work (Lv et al., 2020;]
Tarvainen & Valpola, 2017; Liu et al., 2020) and has been shown to be effective in practice.


2As specified in the EDM (Karras et al., 2022), we use _σ_ data = 0 _._ 5, _P_ mean = _−_ 1 _._ 2 and _P_ std = 1 _._ 2.


5


_t_ - _wt_ - _c_

���


2


_L_ Gen( _θ_ ) = E _t_


_cy_ =1 _[p]_ [(] _[y]_ _[|]_ **[x]** _[t][, z]_ [)] **[ s]** _[θ]_ [(] **[x]** _[t][, y, t]_ [)] _[ −∇]_ **[x]** _[t]_ [ log] _[ q][t][|]_ [0][(] **[x]** _[t][ |]_ **[x]** [0] _[, z]_ [)] ���22


_._ (10)


This loss encourages the weighted aggregation of clean-label scores to approximate the imprecise
score derived from data, thereby enabling label-conditioned learning without the need for explicit
clean annotations. The following Proposition 1, with proof provided in Appendix B.7, guarantees
that the optimal solution recovers the clean-label conditional scores:

**Proposition 1.** _Let θ_ Gen _[∗]_ [=] [arg min] _[θ][ L]_ [Gen][(] _[θ]_ [)] _[ be the minimizer of Eq. (10).]_ _[Then, for all]_ **[ x]** _[t]_ _[∈X]_ _[,]_
_z_ _⊆Y, and t ∈_ [ _T_ ] _, the learned score function satisfies_ **s** _θ_ Gen _[∗]_ [(] **[x]** _[t][, y, t]_ [) =] _[ ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][ |]_ _[y]_ [)] _[.]_


4.3 CLASSIFICATION OBJECTIVE: INFERRING LABELS FROM IMPRECISE SIGNALS


We assume the class prior to be uniform, i.e., _p_ ( _y_ ) = 1 _/c_ . To infer the class-posterior probability
_pθ_ ( _y_ _|_ **x** _t_ ), we adopt a diffusion-based approximation as defined below:


**Definition 1** (Approximated Posterior Noised Diffusion Classifier (Chen et al., 2024b)) **.** Assuming
the uniform prior _p_ ( _y_ ), the class-posterior probability for a noisy input **x** _t_ under a conditional
diffusion model can be derived using Bayes’ rule, as follows:


_pθ_ ( **x** _t |_ _y_ ) exp _{_ log _pθ_ ( **x** _t |_ _y_ ) _}_

   _y_ _[′][ p][θ]_ [(] **[x]** _[t][ |]_ _[y][′]_ [)] [=] _y_ _[′]_ [ exp] _[{]_ [log] _[ p][θ]_ [(] **[x]** _[t][ |]_ _[y]_


_pθ_ ( **x** _t |_ _y_ )
_pθ_ ( _y_ _|_ **x** _t_ ) = 


(11)
_y_ _[′]_ [ exp] _[{]_ [log] _[ p][θ]_ [(] **[x]** _[t][ |]_ _[y][′]_ [)] _[}]_ _[.]_


Here, following Chen et al. (2024b), the conditional likelihood log _pθ_ ( **x** _t |_ _y_ ) is approximated by the
conditional evidence lower bound (ELBO), given by


log _pθ_ ( **x** _t |_ _y_ ) _≈−_ - _Tτ_ = _−t_ 1+1 _[w][τ]_ [E] _[q]_ [(] **[x]** _[τ][ |]_ **[h]** _[θ]_ [(] **[x]** _[t][,y,t]_ [))] ��� **h** _θ_ ( **x** _τ_ _, y, τ_ ) _−_ **x** 0��22


_,_ (12)


_α_ **[x]** _[τ]_ _τ_ [+] _α_ _[σ]_ _τ_ [2] _τ_ **[s]** _[θ]_ [(] **[x]** _[τ]_ _[, y, τ]_ [)][ and] _[ w][τ]_ [=] _[σ]_ _στ_ [2] _τ_ [2][+] _σ_ _[σ]_ data [2] data [2] _[·]_ _σPτ_ ~~_√_~~ std _[−]_ 2 [1]


. [2]


where **h** _θ_ ( **x** _τ_ _, y, τ_ ) = **[x]** _[τ]_


[1] 
2 _π_ [exp] _−_ [(log] _[ σ]_ 2 _[τ][ −]_ _P_ std [2] _[P]_ [mean][)][2]


2 _P_ std [2]


**Partial-label data.** For partial-label data, the imprecise label _Z_ is given as a candidate set _S_ that is
guaranteed to include the true label. In this case, the posterior distribution _pθ_ ( _Y |X, S_ ) is restricted
to assign non-zero probability only to labels within the candidate set. Accordingly, for each sample
( **x** _, s_ ), we compute the classification loss from Eq. (7) as


where **y** ˆ denotes the one-hot vector of the noisy label _y_ ˆ, sg( _·_ ) is the stop-gradient operator [3], _⊙_ is the
Hadamard product, and _⟨·, ·⟩_ denotes the inner product. This formulation inherits the core principle
of _ELR_, stabilizing training through soft pseudo-targets derived from the EMA model. It effectively
amplifies the gradient contribution of cleanly labeled samples while suppressing the influence of
mislabeled ones, which we further analyze in detail in Appendix C.1.


5 TIME COMPLEXITY REDUCTION


The oracle diffusion classifier requires repeated calculations of the conditional ELBO across all classes
to make a prediction, resulting in a substantial computation cost. To address this issue, Chen et al.
(2024b) showed that when estimating ELBO with Monte Carlo sampling, reusing the same **x** _τ_ across
classes and selecting timesteps at uniform intervals is sufficient for effective classification. However,
our experiments reveal that this strategy is empirically suboptimal as illustrated in Figure 1(a). We
identify the core reason to be the model’s varying discriminative ability across different timesteps,
with notable disparities in performance, as shown in Figure 1(b) where the accuracy is evaluated
using only a single timestep. Specifically, when the timestep _τ_ is small, the added noise is negligible,
leading to reconstructions with low label sensitivity. Conversely, when the timestep _τ_ is large, the
input becomes overwhelmed by noise, rendering the predictions highly unreliable.


To this end, we aim to identify a compact subset of timesteps that enables efficient ELBO estimation
while maintaining sufficient classification performance. Let _p_ ( _τ_ ) be a probability density function


3The stop-gradient operator sg( _·_ ) returns its input but blocks gradient flow, i.e., _∇_ **x** sg( **r** ( **x** )) = 0.


6


_L_ [PL] Cls [(] **[x]** [) =] _[ −]_ 


- _pϕ_ ( _y_ _|_ **x** _, s_ ) log _pθ_ ( _y_ _|_ **x** _, s_ ) = _−_ 
_y∈Y_ _y∈s_


- _f_ ˜ _ϕ_ [PL][(] **[x]** [)] _[y]_ [log] _[ f][θ]_ [(] **[x]** [)] _[y][,]_ (13)

_y∈s_


where _f_ [˜] _ϕ_ [PL][(] **[x]** [)][ denotes the normalized probability over] _[ s]_ [ such that][ �] _y∈s_ _[f]_ [˜] _ϕ_ [ PL][(] **[x]** [)] _[y]_ [= 1][ and] _[f]_ [˜] _ϕ_ [ PL][(] **[x]** [)] _[y]_ [=]

0 for all _y_ _∈/_ _s_ . Eq. (13) can be interpreted as an EMA-stabilized variant of the method called
progressive identification ( _PRODEN_ ) (Lv et al., 2020), where EMA predictions serve as soft pseudotargets.


**Supplementary-unlabeled data.** In this scenario, the training set consists of a small portion of
labeled data and a larger number of unlabeled data. This setting can be regarded as a special case of
the partial-label formulation: labeled instances are assigned singleton candidate sets containing the
ground-truth label, while unlabeled instances are associated with the full label space. Accordingly,
the classification loss for each instance is defined as


_L_ [SU] Cls [(] **[x]** [) =] _[ −]_ 


- _pϕ_ ( _y_ _|_ **x** _, z_ ) log _pθ_ ( _y_ _|_ **x** _, z_ ) = _−_ 

_y∈Y_ _y∈Y_


- _f_ ˜ _ϕ_ [SU][(] **[x]** [)] _[y]_ [log] _[ f][θ]_ [(] **[x]** [)] _[y][,]_ (14)

_y∈Y_


where _f_ [˜] _ϕ_ [SU][(] **[x]** [)][ denotes the pseudo-target distribution:] [for labeled samples, it reduces to a one-hot]
vector of the ground-truth label, while for unlabeled samples, it corresponds to the EMA model’s
prediction over the entire label set. This loss can thus be viewed as an EMA-stabilized self-training
objective (Tarvainen & Valpola, 2017), a widely used strategy in semi-supervised learning that
leverages unlabeled data through soft pseudo-labels.


**Noisy-label** **data.** In practice, accurately distinguishing clean labels from noisy ones is often
difficult, making it challenging to retain reliable supervision while applying self-training for label
refinement. To mitigate this, we leverage the memorization effect in noisy-label learning, where
neural networks typically fit clean labels before overfitting to noise (Han et al., 2020). Drawing
inspiration from the noisy-label learning method called early learning regularization ( _ELR_ ) (Liu
et al., 2020), we propose a simpler yet effective loss function that retains its core idea, defined as


- sg( **r** ( **x** )) _y_ log _fθ_ ( **x** ) _y,_ **r** ( **x** ) = **ˆy** _−_ _[f][θ]_ [(] **[x]** [)] _[ ⊙]_ - _⟨fθ_ ( **x** ) _, fϕ_ ( **x** ) _⟩_ **1** _−_ _fϕ_ ( **x** )�

1 _−⟨fθ_ ( **x** ) _, fϕ_ ( **x** ) _⟩_

_y∈Y_


_L_ [NL] Cls [(] **[x]** [) =] _[ −]_ 


_,_ (15)
1 _−⟨fθ_ ( **x** ) _, fϕ_ ( **x** ) _⟩_


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||Chen et al.|t<br>0|Ourst<br>0|
|||||<br>Chen et al.<br>Chen et al.|t=0.5<br>t=1.0|<br><br>Ourst=0.5<br>Ourst=1.0|


where ℏ( _τ, y_ ) = _wτ_ E **x** _τ_ [ _∥_ **h** _θ_ ( **x** _τ_ _, y, τ_ ) _−_ **x** 0 _∥_ [2] 2 []][.] [Eq. (16) formalizes the goal of finding a representa-]
tive range where the expected reconstruction error closely matches that of the full distribution. To
strike a compromise between signal and noise within the selected subinterval, we propose choosing it
around the median of _p_ ( _τ_ ), so that signal-dominant early timesteps and noise-dominant later timesteps
complement each other. This strategy yields a more stable and representative approximation, especially when _p_ ( _τ_ ) is skewed. Therefore, we provide a formal characterization of the subinterval
construction for the EDM by the following theorem.


**Theorem 2.** _Consider an EDM where τ_ _is sampled from a log-normal distribution, i.e.,_ ln( _τ_ ) _∼_
_N_ ( _τ_ ; _P_ mean _,_ _P_ std [2] [)] _[, where][ P]_ [mean] _[∈]_ [R] _[ and][ P]_ [std] _[>]_ [ 0] _[.]_ _[Given a fixed subinterval length]_ [ ∆] _[, a sampling]_
_range centered around the median of p_ ( _τ_ ) _can be constructed by solving the following equation for_
_the left boundary l:_


_l_ = Solve _τ_ ( _F_ ( _τ_ ) + _F_ ( _τ_ + ∆) _−_ 1 = 0) _,_ _r_ = _l_ + ∆ _,_


_where_ Solve _τ_ ( _·_ ) _denotes a numerical root-finding algorithm over τ_ _, such as the Brent method (Brent,_
_2013), and F_ ( _·_ ) _is the cumulative distribution function of p_ ( _τ_ ) _._


The proof of Theorem 2 as well as a similar conclusion for denoising diffusion probabilistic model
(DDPM) (Ho et al., 2020) are provided in Appendix B.8. Notably, our finding aligns with the
effective timestep hypothesis proposed in Li et al. (2023) for the DDPM setting. Furthermore, based
on Eq. (16), we can derive a necessary condition that any theoretically optimal subinterval must
satisfy, as formalized in the following theorem:

**Theorem 3** (Necessary Condition for Optimal Subinterval) **.** _Given_ ( _l_ _[∗]_ _, r_ _[∗]_ ) _be an optimal subinterval_
_of the support of p_ ( _τ_ ) _, a necessary condition for attaining the theoretical minimum of the squared_
_error objective in Eq. (16) is_

ERR( _l_ _[∗]_ _, r_ _[∗]_ _, y_ ) = E _τ_ _∼p_ ( _τ_ ) _|τ_ _∈_ [ _l∗,r∗_ ][ℏ( _τ, y_ )] _−_ [ℏ][(] _[l][∗][, y]_ [) +] 2 [ ℏ][(] _[r][∗][, y]_ [)] = 0 _._ (17)


The proof of Theorem 3 can be found in Appendix B.9. Based on Theorem 3, we empirically present
the class-wise distribution of ERR( _·, ·, y_ ) across samples in Figure 1(c), where the errors are generally
concentrated around zero, supporting the effectiveness of our proposed time complexity reduction
strategy. Notably, when the subinterval is reduced to a single sampling point, choosing the median of
_p_ ( _τ_ ) (i.e., _e_ _[P]_ [mean] ) yields the best classification performance as shown in Figure 1(b). This observation
is consistent with our earlier hypothesis regarding the informativeness of the median timestep. In
practical posterior inference, we combine timestep subinterval reduction strategy with **x** _τ_ reuse
technique (Chen et al., 2024c) to further improve inference efficiency.


7


90

80

70

60

50

40

30

20


0.5 1.1 1.7 2.3 2.9 3.5
Timestep used


2 4 8 16 32 64
T


(a) Results across different _T_ and _t_ .


(b) Results with a single timestep.


75


65


55


45


35


0.02


0.01


0.00


0.01


0.02


0 1 2 3 4 5 6 7 8 9
y (Class)


(c) Class-wise ERR distribution.


Figure 1: (a): Test accuracy (%) comparison on CIFAR-10 dataset under time complexity reduction
technique from Chen et al. (2024c) and ours. (b): Test accuracy (%) on CIFAR-10 dataset evaluated
with only a single timestep per class. (c): Violin plot of class-wise ERR( _·, ·, y_ ) computed across
samples using a fixed subinterval length ∆. Wider regions of the violin indicate higher density.


over the interval _τ_ _∈_ (0 _,_ + _∞_ ), satisfying �0+ _∞_ _p_ ( _τ_ ) d _τ_ = 1. Our objective is to select a subinterval
_τ_ _∈_ [ _l, r_ ] such that


minimize
0 _≤l≤r_


2
��E _τ_ _∼p_ ( _τ_ _|τ_ _∈_ [ _l,r_ ])[ℏ( _τ, y_ )] _−_ E _τ_ _∼p_ ( _τ_ )[ℏ( _τ, y_ )]��2 _[,]_ (16)


Table 1: Generative results on CIFAR-10 and ImageNette under various settings. ‘uncond’ and ‘cond’
indicate unconditional and conditional metrics. **Bold** numbers indicate better performance.


Noisy-label supervision Partial-label supervision Suppl-unlabeled supervision
Metric Clean
Sym-40% Asym-40% Random Class-50% Random-1% Random-10%


_Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_


FID ( _↓_ ) **3.33** 3.47 3.23 **3.10** 7.76 **2.26** 11.75 **2.77** 3.16 **3.12** 2.93 **2.89** 2.05
IS ( _↑_ ) 9.56 **9.68** 9.02 **9.73** 9.09 **9.80** 9.62 **9.68** 10.03 **10.57** 9.80 **9.83** 10.61
Density ( _↑_ ) 101.39 **109.75** 100.06 **109.69** 103.21 **106.49** 108.76 **109.06** 97.19 **108.18** 99.96 **108.87** 112.59
Coverage ( _↑_ ) 81.12 **81.21** 80.71 **81.30** 68.45 **82.69** 64.90 **81.52** 78.44 **81.00** 81.85 **82.00** 83.27

CW-FID ( _↓_ ) 29.84 **13.85** 14.70 **13.24** 27.18 **10.65** 32.44 **11.56** 16.25 **16.12** 11.84 **11.77** 9.83
CW-Density ( _↑_ ) 72.98 **107.23** 90.85 **107.07** 102.04 **105.75** 102.43 **108.66** 89.99 **100.73** 96.29 **107.94** 111.70
CW-Coverage ( _↑_ ) 73.39 **80.11** 79.63 **79.65** 65.45 **82.09** 61.45 **81.24** 75.03 **76.84** 80.80 **81.12** 83.91


FID ( _↓_ ) 14.11 **13.44** 13.93 **13.91** 79.13 **72.62** 91.28 **79.12** 23.88 **19.26** 14.32 **12.84** 11.52
IS ( _↑_ ) 12.69 **13.21** 12.51 **13.73** 9.19 **9.40** **9.27** 9.11 12.23 **13.72** 12.80 **13.16** 13.81
Density ( _↑_ ) 109.31 **112.52** **111.66** 106.78 95.33 **99.83** 94.29 **102.58** 115.94 **125.68** 105.27 **109.23** 117.23
Coverage ( _↑_ ) 76.62 **76.81** 78.32 **79.81** 21.44 **32.48** 16.69 **22.30** 53.53 **55.39** 73.79 **75.55** 80.12

CW-FID ( _↓_ ) 80.31 **60.12** 62.26 **58.20** 157.76 **63.58** 163.45 **67.92** 71.66 **70.27** 49.22 **44.31** 40.20
CW-Density ( _↑_ ) 73.99 **81.12** 93.53 **94.58** 93.38 **95.83** 91.50 **95.21** 115.90 **118.69** 103.41 **115.67** 120.35
CW-Coverage ( _↑_ ) 67.89 **71.94** 74.18 **75.82** 19.76 **24.35** 15.88 **18.93** 51.73 **52.15** 72.61 **74.85** 78.48


6 EXPERIMENTS


We present experiments on three tasks including image generation, weakly supervised learning,
and dataset condensation to demonstrate the utility and versatility of our method. Evaluations
are performed on three benchmark datasets widely used for both generation and classification,
covering image resolutions from 28 _×_ 28 (Fashion-MNIST (Xiao et al., 2017)) and 32 _×_ 32 (CIFAR10 (Krizhevsky et al., 2009)) to 64 _×_ 64 (ImageNette (Deng et al., 2009)). As a baseline, we refer
to the model trained with the generative objective in Eq. (8) as the _Vanilla_ method. The training
hyperparameters are kept consistent with those used in the EDM model (Karras et al., 2022).


**Dataset construction** . For partial-label data, we generate synthetic candidate label sets using both
class-dependent (Wen et al., 2021) and random generation models (Feng et al., 2020). In the classdependent setting, we construct a transition matrix that maps each true label to a set of semantically
similar labels, where each similar label is included in the candidate set with probability 50%. In
contrast, the random setting assign each incorrect label an equal probability 50% of being included in
the candidate set. For supplementary-unlabeled data, we follow a standard semi-supervised setup by
randomly selecting 10% and 1% of the training data classwise as labeled samples, and treating the
remaining data as unlabeled. For noisy-label data, we consider both symmetric and asymmetric noise.
In the symmetric case, labels are uniformly flipped to any incorrect class, whereas in the asymmetric
case, they are flipped to semantically similar classes according to a predefined mapping. In both
cases, the corruption probability is referred to as the noise rate, which is set to 40%.


6.1 TASK1: IMAGE GENERATION


**Setup** . We evaluate the trained CDMs using four unconditional metrics, including Frechet Inception´
Distance (FID) (Heusel et al., 2017), Inception Score (IS) (Salimans et al., 2016), Density, and
Coverage (Naeem et al., 2020), as well as three conditional metrics, namely CW-FID, CW-Density,
and CW-Coverage (Chao et al., 2022). The Class-Wise (CW) metrics are computed per class and
then averaged. Detailed descriptions of these metrics are provided in the Appendix E.1.


**Results** . Table 10 reports the generative performance of the _Vanilla_ model and our proposed _DMIS_
model on various settings. It can be seen that our model outperforms the baseline across almost all
cases with respect to both unconditional and conditional metrics. The performance gap is especially


(a) Fashion-MNIST (b) CIFAR-10 (c) ImageNette


Figure 2: Comparison of conditionally generated images from _Vanilla_ (top) and our _DMIS_ model
(bottom), each trained with 40% symmetric noise on Fashion-MNIST, CIFAR-10, and ImageNette.


**Dress**


**Coat**


**Bag**


**Dress**


**Coat**


**Bag**


**Deer**


**Dog**


**Frog**


**Deer**


**Dog**


**Frog**


**Tench**


**Truck**


**Ball**


**Tench**


**Truck**


**Ball**


(a) Fashion-MNIST


(b) CIFAR-10


8


Table 2: Classification results (test accuracy, %) on Fashion-MNIST, CIFAR-10, and ImageNette
datasets under various types of imprecise supervision ( _♠_ : partial-label, _♡_ : supplementary-unlabeled,
_♣_ : noisy-label). **Bold** numbers indicate the best performance. [5]


Dataset _[♠]_ Type _PRODEN_ _IDGP_ _PiCO_ _CRDPLL_ _DIRK_ _Vanilla_ _DMIS_ _[CE]_ _DMIS_


Random 93.31 _±_ 0.07 92.26 _±_ 0.25 93.32 _±_ 0.12 94.03 _±_ 0.14 94.11 _±_ 0.22 80.20 _±_ 1.29 84.24 _±_ 0.37 **94.27** _±_ **0.55**
F-MNIST Class-50% 93.44 _±_ 0.21 93.07 _±_ 0.16 93.32 _±_ 0.33 93.80 _±_ 0.23 93.99 _±_ 0.24 66.03 _±_ 1.43 78.45 _±_ 0.46 **94.20** _±_ **0.15**


Random 90.02 _±_ 0.22 89.65 _±_ 0.53 86.40 _±_ 0.89 92.74 _±_ 0.26 93.48 _±_ 0.14 60.25 _±_ 0.17 91.47 _±_ 0.15 **94.70** _±_ **0.49**
CIFAR-10 Class-50% 90.44 _±_ 0.44 90.83 _±_ 0.34 87.51 _±_ 0.66 92.89 _±_ 0.27 93.22 _±_ 0.37 56.34 _±_ 0.50 90.52 _±_ 0.35 **93.53** _±_ **0.12**


Random 84.75 _±_ 0.13 84.07 _±_ 0.26 82.15 _±_ 0.23 84.31 _±_ 0.25 87.90 _±_ 0.11 56.04 _±_ 0.61 84.49 _±_ 0.05 **89.31** _±_ **0.21**
ImageNette Class-50% 83.50 _±_ 0.60 82.18 _±_ 0.13 84.41 _±_ 0.93 88.08 _±_ 0.34 87.47 _±_ 0.17 59.47 _±_ 0.51 82.34 _±_ 0.27 **88.42** _±_ **0.43**


Dataset _[♡]_ Type _Dash_ _CoMatch_ _FlexMatch_ _SimMatch_ _SoftMatch_ _Vanilla_ _DMIS_ _[CE]_ _DMIS_


Random-1% 84.73 _±_ 0.09 85.31 _±_ 0.29 84.43 _±_ 0.30 84.69 _±_ 0.17 84.72 _±_ 0.23 78.37 _±_ 0.72 82.92 _±_ 0.17 **85.92** _±_ **0.13**
F-MNIST Random-10% 91.16 _±_ 0.20 90.52 _±_ 0.12 90.69 _±_ 0.03 91.18 _±_ 0.13 91.22 _±_ 0.11 90.50 _±_ 1.00 91.07 _±_ 0.18 **92.97** _±_ **0.21**


Random-1% 70.14 _±_ 0.69 61.45 _±_ 1.46 70.72 _±_ 0.93 73.33 _±_ 1.02 73.74 _±_ 0.82 53.49 _±_ 0.15 75.30 _±_ 0.17 **76.40** _±_ **0.54**
CIFAR-10 Random-10% 81.50 _±_ 0.68 77.79 _±_ 0.53 81.35 _±_ 0.48 82.90 _±_ 0.43 88.66 _±_ 0.60 85.13 _±_ 0.12 89.85 _±_ 0.08 **92.47** _±_ **0.39**


Random-1% 57.68 _±_ 2.19 63.88 _±_ 0.78 61.39 _±_ 0.70 58.12 _±_ 2.66 58.50 _±_ 2.31 49.55 _±_ 0.99 62.64 _±_ 0.24 **68.23** _±_ **0.19**
ImageNette Random-10% 74.66 _±_ 0.81 73.20 _±_ 0.46 73.08 _±_ 0.13 76.12 _±_ 0.45 75.75 _±_ 0.25 74.70 _±_ 0.53 71.39 _±_ 0.45 **77.30** _±_ **0.15**


Dataset _[♣]_ Type _CE_ _Mixup_ _Coteaching_ _ELR_ _PENCIL_ _Vanilla_ _DMIS_ _[CE]_ _DMIS_


Sym-40% 76.18 _±_ 0.26 92.21 _±_ 0.03 92.17 _±_ 0.34 93.13 _±_ 0.13 90.85 _±_ 0.58 90.11 _±_ 1.24 87.76 _±_ 0.57 **93.40** _±_ **0.40**
F-MNIST Asym-40% 82.01 _±_ 0.06 92.01 _±_ 1.02 92.78 _±_ 0.25 92.82 _±_ 0.09 91.77 _±_ 0.69 85.41 _±_ 0.96 83.39 _±_ 0.24 **93.20** _±_ **0.30**


Sym-40% 67.22 _±_ 0.26 84.26 _±_ 0.64 86.54 _±_ 0.57 85.68 _±_ 0.13 85.91 _±_ 0.26 80.22 _±_ 0.10 84.75 _±_ 0.36 **88.63** _±_ **0.12**
CIFAR-10 Asym-40% 76.98 _±_ 0.42 83.21 _±_ 0.85 79.38 _±_ 0.39 81.32 _±_ 0.31 84.89 _±_ 0.49 86.31 _±_ 0.10 84.21 _±_ 0.18 **88.83** _±_ **0.33**


Sym-40% 58.43 _±_ 0.77 76.65 _±_ 1.62 66.55 _±_ 1.00 84.33 _±_ 2.86 81.94 _±_ 1.26 55.86 _±_ 1.95 80.47 _±_ 0.56 **84.12** _±_ **0.18**
ImageNette Asym-40% 71.81 _±_ 0.38 77.16 _±_ 0.71 75.12 _±_ 0.50 73.51 _±_ 0.31 77.20 _±_ 1.15 53.91 _±_ 1.07 77.21 _±_ 0.19 **79.30** _±_ **0.27**


pronounced under partial-label supervision. These results indicate that _DMIS_ not only enhances
the quality of samples but also produces generative distributions that more closely align with the
true data distribution. Furthermore, Figure 2 compares conditionally generated samples from the
_Vanilla_ and _DMIS_ models across different datasets. Compared to the _Vanilla_ model which often
produces samples misaligned with the class, our model produces images of higher visual fidelity and
class-conditional generations that more accurately reflect the intended semantic categories.


6.2 TASK2: WEAKLY SUPERVISED LEARNING


**Setup** . We evaluate our method under three weakly supervised scenarios. In partial-label learning,
we compare against approaches including _PRODEN_ (Lv et al., 2020), _IDGP_ (Qiao et al., 2023),
_PiCO_ (Wang et al., 2023), _CRDPLL_ (Wu et al., 2022) and _DIRK_ (Wu et al., 2024). For semisupervised learning, we adopt _Dash_ (Xu et al., 2021), _CoMatch_ (Li et al., 2021a), _FlexMatch_ (Zhang
et al., 2021a), _SimMatch_ (Zheng et al., 2022) and _SoftMatch_ (Chen et al., 2023) as comparison
methods. For noisy-label learning, we compare with _Coteaching_ (Han et al., 2018), _ELR_ (Liu
et al., 2020), _PENCIL_ (Yi & Wu, 2019), as well as standard normal cross-entropy _(CE_ ) training and
_Mixup_ (Zhang et al., 2018). To ensure a fair comparison, the discriminative classifier is implemented
as Wide-ResNet-40-10 with 55.84M parameters, while our generative model contains 55.73M
parameters, and all models are trained from scratch without pre-training.


**Results** . The classification results for weakly supervised learning are reported in Table 2. Overall,
our method _DMIS_, evaluated via a diffusion classifier, achieves the best performance, demonstrating
the stronger generalization capability of diffusion models over prior discriminative approaches. Interestingly, the _Vanilla_ method still outperforms several baselines, particularly in the noisy-label setting,
suggesting that the vanilla denoising score matching objective still acts as an implicit regularizer
against label noise. Moreover, compared to standard _CE_ training, the regenerate-classification variant
_DMIS_ _[CE]_ improves accuracy by up to 11.58%, 17.53%, and 22.13% on Fashion-MNIST, CIFAR-10,
and ImageNette dataset, respectively, showing that the regenerated dataset effectively mitigates label
imprecision and yields cleaner supervision for downstream discriminative training.


6.3 TASK3: NOISY DATASET CONDENSATION


While the task of dataset condensation has achieved remarkable progress recently, existing methods
are typically developed under the assumption of clean labels. However, label noise is inevitable and
cannot be fully eliminated in practice. Therefore, exploring how to condense a clean dataset from


5 _DMISCE_ denotes regenerate-classification results, i.e., we regenerate datasets of the same size under
conditional sampling and then train a discriminative model on them using standard _CE_ loss.


9


Table 3: Classification results (test accuracy, %) on noisy-label Fashion-MNIST, CIFAR-10, and
ImageNette datasets. ‘IPC’ indicates the number of images per class in the condensed dataset. **Bold**
numbers indicate the best performance.


Dataset Type IPC _Random_ _DC_ _DSA_ _DM_ _MTT_ _RDED_ _SRE2L_ _DMIS_


10 34.42 _±_ 0.69 22.85 _±_ 1.69 42.07 _±_ 2.49 57.06 _±_ 1.52 9.03 _±_ 3.81 18.57 _±_ 1.06 15.80 _±_ 0.38 **70.18** _±_ **0.37**
Sym-40% 50 52.36 _±_ 0.60 35.64 _±_ 2.26 55.22 _±_ 1.51 68.23 _±_ 0.47 10.91 _±_ 0.82 23.19 _±_ 0.74 19.51 _±_ 0.96 **80.73** _±_ **0.07**
100 55.14 _±_ 0.06 30.46 _±_ 1.74 41.30 _±_ 0.85 73.21 _±_ 0.69 13.73 _±_ 3.96 25.43 _±_ 0.21 19.66 _±_ 1.91 **84.26** _±_ **0.02**


10 48.28 _±_ 0.34 53.17 _±_ 1.59 57.15 _±_ 2.37 63.27 _±_ 1.60 8.75 _±_ 0.82 18.42 _±_ 1.62 16.45 _±_ 1.96 **65.02** _±_ **1.85**
Asym-40% 50 69.44 _±_ 0.17 49.21 _±_ 0.69 77.20 _±_ 0.34 76.39 _±_ 0.57 8.76 _±_ 2.11 22.31 _±_ 0.67 27.07 _±_ 0.35 **79.65** _±_ **0.63**
100 70.80 _±_ 0.91 36.95 _±_ 0.57 80.24 _±_ 0.54 78.43 _±_ 0.63 12.59 _±_ 1.22 24.03 _±_ 0.97 26.52 _±_ 1.46 **83.22** _±_ **0.33**


10 16.30 _±_ 0.96 18.11 _±_ 1.02 18.06 _±_ 1.72 23.71 _±_ 0.40 12.06 _±_ 0.46 19.85 _±_ 0.88 13.12 _±_ 1.04 **27.83** _±_ **0.98**
Sym-40% 50 26.59 _±_ 0.70 20.63 _±_ 0.22 28.76 _±_ 0.57 29.50 _±_ 0.56 17.96 _±_ 2.10 34.64 _±_ 0.58 14.23 _±_ 1.67 **46.47** _±_ **0.41**
100 31.19 _±_ 0.74 19.91 _±_ 0.54 29.45 _±_ 0.34 32.26 _±_ 0.75 18.04 _±_ 3.55 44.03 _±_ 0.21 14.21 _±_ 0.93 **56.53** _±_ **0.03**


10 24.89 _±_ 1.65 18.51 _±_ 1.35 22.23 _±_ 1.80 26.53 _±_ 0.07 9.62 _±_ 1.45 23.48 _±_ 0.65 14.64 _±_ 1.03 **24.94** _±_ **0.49**
Asym-40% 50 40.95 _±_ 0.59 25.97 _±_ 0.97 40.81 _±_ 0.29 43.09 _±_ 0.76 16.54 _±_ 1.88 39.12 _±_ 0.13 16.03 _±_ 0.21 **47.77** _±_ **0.78**
100 47.49 _±_ 0.64 27.76 _±_ 0.72 42.96 _±_ 0.84 51.61 _±_ 0.60 17.67 _±_ 2.53 44.45 _±_ 0.19 17.55 _±_ 0.91 **55.89** _±_ **0.39**


10 23.09 _±_ 0.19 15.89 _±_ 0.73 27.70 _±_ 1.25 28.83 _±_ 0.73 33.60 _±_ 0.53 21.15 _±_ 1.05 25.03 _±_ 1.17 **34.36** _±_ **1.05**
Sym-40% 50 33.83 _±_ 0.28 24.62 _±_ 0.73 32.07 _±_ 1.01 42.66 _±_ 1.27 38.39 _±_ 1.67 35.87 _±_ 0.39 35.37 _±_ 0.82 **44.93** _±_ **0.28**
100 40.04 _±_ 0.71 22.81 _±_ 1.22 36.05 _±_ 1.76 43.25 _±_ 2.13 39.61 _±_ 1.52 35.87 _±_ 0.39 41.74 _±_ 1.37 **56.23** _±_ **0.84**


10 26.54 _±_ 0.88 19.26 _±_ 0.98 30.62 _±_ 2.09 33.40 _±_ 0.48 33.65 _±_ 1.29 26.23 _±_ 0.06 25.74 _±_ 2.21 **37.09** _±_ **0.29**
Asym-40% 50 47.91 _±_ 0.61 31.68 _±_ 2.15 43.41 _±_ 1.24 50.97 _±_ 1.61 38.71 _±_ 1.24 32.75 _±_ 0.43 35.29 _±_ 0.14 **55.20** _±_ **0.46**
100 59.10 _±_ 1.41 29.19 _±_ 0.21 53.79 _±_ 0.84 60.70 _±_ 1.88 37.69 _±_ 1.29 35.48 _±_ 0.22 42.37 _±_ 0.34 **68.97** _±_ **0.12**


noisy-label data is natural and meaningful. To the best of our knowledge, this is the first work to
investigate dataset condensation under noisy supervision, which we term _noisy dataset condensation_ .


**Setup** . During condensation, we employ our trained CDMs to synthesize images according to the specified IPC. For evaluation, we compare against both hard-label-based methods, including _DC_ (Zhao
et al., 2021), _DSA_ (Zhao & Bilen, 2021), _DM_ (Zhao & Bilen, 2023), and _MTT_ (Cazenavette et al.,
2022b), as well as soft-label-based methods, namely _RDED_ (Sun et al., 2024) and _SRE2L_ (Yin et al.,
2024). Following common protocols (Sun et al., 2024; Yin et al., 2024), we adopt ResNet-18 as the
backbone during condensation and evaluate the condensed datasets on a test set using ResNet-34.


**Results** . Table 3 presents the results of noisy dataset condensation, with qualitative visualizations
provided in Appendix E.5. Our method consistently surpasses prior approaches across datasets and
noise types. These results highlight the advantage of generative condensation: rather than memorizing
noisy labels, _DMIS_ implicitly denoises them during generation, leading to cleaner condensed datasets.
Notably, unlike the trends observed in clean dataset condensation, distribution-matching methods
(e.g., _DM_ ) achieve the second-best results in this noisy setting, suggesting that distribution alignment
helps regularize the effect of label noise. Moreover, instance-selection methods generally outperform
synthetic-generation methods (e.g., _Random_ vs. _DC_ / _DSA_ / _MTT_ and _RDED_ vs. _SRE2L_ ), indicating
that discarding noisy samples during condensation is also an effective strategy to mitigate label noise.
Collectively, these findings not only demonstrate the superiority of our approach but also provide
useful insights for future work on noisy dataset condensation.


7 CONCLUSION


In this paper, we addressed the challenge of training CDMs under imprecise supervision, a setting
that frequently arises in real-world applications. We introduced a unified framework that formulates
the learning problem as likelihood maximization and decomposes it into generative and classification
components. Based on this formulation, we proposed a weighted denoising score matching objective
that enables label-conditioned learning without clean annotations, and developed an efficient timestep
sampling strategy to reduce the computational cost of posterior inference. Extensive experiments
across image generation, weakly supervised learning, and noisy dataset condensation verified the
effectiveness and versatility of our approach. Beyond establishing strong baselines, our work also
pioneers the study of noisy dataset condensation, opening new opportunities for future exploration in
robust and scalable diffusion modeling under weak supervision.


REFERENCES


David Berthelot, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, and
Colin Raffel. Remixmatch: Semi-supervised learning with distribution matching and augmentation
anchoring. In _International Conference on Learning Representations_, 2019a.


10


David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A
Raffel. Mixmatch: A holistic approach to semi-supervised learning. In _Advances_ _in_ _Neural_
_Information Processing Systems_, pp. 5050–5060, 2019b.


Richard P Brent. _Algorithms for minimization without derivatives_ . Courier Corporation, 2013.


George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu. Dataset
distillation by matching training trajectories. In _IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 4750–4759, 2022a.


George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu. Dataset
distillation by matching training trajectories. In _IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 4750–4759, 2022b.


Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Yi-Chen Lo, Chia-Che Chang, Yu-Lun Liu, Yu-Lin
Chang, Chia-Ping Chen, and Chun-Yi Lee. Denoising likelihood score matching for conditional
score-based data generation. In _International Conference on Learning Representations_, 2022.


Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj,
and Marios Savvides. Softmatch: Addressing the quantity-quality tradeoff in semi-supervised
learning. In _International Conference on Learning Representations_, 2023.


Hao Chen, Ankit Shah, Jindong Wang, Ran Tao, Yidong Wang, Xiang Li, Xing Xie, Masashi
Sugiyama, Rita Singh, and Bhiksha Raj. Imprecise label learning: A unified framework for learning
with various imprecise label configurations. In _Advances in Neural Information Processing Systems_,
pp. 59621–59654, 2024a.


Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, and Jun Zhu.
Diffusion models are certifiably robust classifiers. In _Advances in Neural Information Processing_
_Systems_, pp. 50062–50097, 2024b.


Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, and Jun Zhu.
Robust classification via a single diffusion model. In _Proceedings of the International Conference_
_on Machine Learning_, pp. 6643–6665, 2024c.


Ciprian Corneanu, Raghudeep Gadde, and Aleix M Martinez. Latentpaint: Image inpainting in latent
space with diffusion models. In _Proceedings of the IEEE/CVF winter conference on applications_
_of computer vision_, pp. 4334–4343, 2024.


Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet-1k
with constant memory. In _Proceedings of the International Conference on Machine Learning_, pp.
6565–6590, 2023.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_,
pp. 248–255, 2009.


Prafulla Dhariwal and AlexanderQuinn Nichol. Diffusion models beat gans on image synthesis. In
_Advances in Neural Information Processing Systems_, pp. 8780–8794, 2021.


Nicolas Dufour, Victor Besnier, Vicky Kalogeiton, and David Picard. Don’t drop your samples!
coherence-aware training benefits conditional diffusion. In _IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, 2024.


Bradley Efron. Tweedie’s formula and selection bias. _Journal of the American Statistical Association_,
pp. 1602–1614, 2011.


Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers
for high-resolution image synthesis. In _Proceedings of the International Conference on Machine_
_Learning_, pp. 12606–12633, 2024.


11


Lei Feng, Jiaqi Lv, Bo Han, Miao Xu, Gang Niu, Xin Geng, Bo An, and Masashi Sugiyama. Provably
consistent partial-label learning. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, pp.
10948–10960, 2020.


Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi
Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In
_Advances in Neural Information Processing Systems_, pp. 8536–8546, 2018.


Bo Han, Quanming Yao, Tongliang Liu, Gang Niu, Ivor W Tsang, James T Kwok, and Masashi
Sugiyama. A survey of label-noise representation learning: Past, present and future. _arXiv preprint_
_arXiv:2011.04406_, 2020.


Chunming He, Chengyu Fang, Yulun Zhang, Longxiang Tang, Jinfa Huang, Kai Li, Zhenhua Guo,
Xiu Li, and Sina Farsiu. Reti-diff: Illumination degradation image restoration with retinex-based
latent diffusion model. In _International Conference on Learning Representations_, 2025.


Wei He, Kai Han, Ying Nie, Chengcheng Wang, and Yunhe Wang. Species196: A one-million
semi-supervised dataset for fine-grained species recognition. In _Advances in Neural Information_
_Processing Systems_, pp. 44957–44975, 2023.


Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium. In _Advances in Neural_
_Information Processing Systems_, pp. 6626–6637, 2017.


Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In _NeurIPS 2021 Workshop on_
_Deep Generative Models and Downstream Applications_, 2022.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In _Advances in_
_Neural Information Processing Systems_, pp. 6840–6851, 2020.


Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J
Fleet. Video diffusion models. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, pp.
8633–8646, 2022.


Takuhiro Kaneko, Yoshitaka Ushiku, and Tatsuya Harada. Label-noise robust generative adversarial
networks. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 2467–2476,
2019.


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In _Advances in Neural Information Processing Systems_, pp. 26565–26577,
2022.


Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, JungWoo Ha, and Hyun Oh Song. Dataset condensation via efficient synthetic-data parameterization.
In _Proceedings of the International Conference on Machine Learning_, pp. 11102–11118, 2022.


Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In
_Advances in Neural Information Processing Systems_, pp. 21696–21707, 2021.


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.


Dong-Hyun Lee et al. Pseudo-label: The simple and efficient semi-supervised learning method for
deep neural networks. In _Workshop on challenges in representation learning, ICML_, pp. 896, 2013.


Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion
model is secretly a zero-shot classifier. In _Proceedings of the IEEE/CVF International Conference_
_on Computer Vision_, pp. 2206–2217, 2023.


Junnan Li, Caiming Xiong, and Steven CH Hoi. Comatch: Semi-supervised learning with contrastive
graph regularization. In _Proceedings of the IEEE/CVF International Conference on Computer_
_Vision_, pp. 9475–9484, 2021a.


Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, and Luc Van Gool. Webvision database: Visual
learning and understanding from web data. _arXiv preprint arXiv:1708.02862_, 2017.


12


Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, and Masashi Sugiyama. Provably end-to-end labelnoise learning without anchor points. In _Proceedings of the International Conference on Machine_
_Learning_, pp. 6403–6413, 2021b.


Yangming Li, Max Ruiz Luyten, and Mihaela van der Schaar. Risk-sensitive diffusion: Robustly
optimizing diffusion models with noisy samples. In _International Conference on Learning Repre-_
_sentations_, 2024.


Sheng Liu, Jonathan Niles-Weed, Narges Razavian, and Carlos Fernandez-Granda. Early-learning regularization prevents memorization of noisy labels. In _Advances in Neural Information Processing_
_Systems_, pp. 20331–20342, 2020.


Noel Loo, Ramin Hasani, Alexander Amini, and Daniela Rus. Efficient dataset distillation using
random feature approximation. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, pp.
13877–13891, 2022.


Calvin Luo. Understanding diffusion models: A unified perspective. _arXiv preprint arXiv:2208.11970_,
2022.


Wenshui Luo, Shuo Chen, Tongliang Liu, Bo Han, Gang Niu, Masashi Sugiyama, Dacheng Tao, and
Chen Gong. Estimating per-class statistics for label noise learning. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence_, 47(1):305–322, 2024.


Jiaqi Lv, Miao Xu, Lei Feng, Gang Niu, Xin Geng, and Masashi Sugiyama. Progressive identification
of true labels for partial-label learning. In _Proceedings of the International Conference on Machine_
_Learning_, pp. 6500–6510, 2020.


Eran Malach and Shai Shalev-Shwartz. Decoupling” when to update” from” how to update”. In
_Advances in Neural Information Processing Systems_, pp. 960–970, 2017.


Takeru Miyato and Masanori Koyama. cgans with projection discriminator. In _International_
_Conference on Learning Representations_, 2018.


Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a
regularization method for supervised and semi-supervised learning. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence_, 41(8):1979–1993, 2018.


Byeonghu Na, Yeongmin Kim, HeeSun Bae, Jung Hyun Lee, Se Jung Kwon, Wanmo Kang, and
Il-Chul Moon. Label-noise robust diffusion models. In _International Conference on Learning_
_Representations_, 2024.


Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. Reliable
fidelity and diversity metrics for generative models. In _Proceedings of the International Conference_
_on Machine Learning_, pp. 7176–7185, 2020.


Zhuoshi Pan, Yuguang Yao, Gaowen Liu, Bingquan Shen, H Vicky Zhao, Ramana Rao Kompella,
and Sijia Liu. From trojan horses to castle walls: Unveiling bilateral backdoor effects in diffusion
models. In _NeurIPS 2023 Workshop on Backdoors in Deep Learning-The Good, the Bad, and the_
_Ugly_, 2023.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style,
high-performance deep learning library. In _Advances in Neural Information Processing Systems_,
2019.


Congyu Qiao, Ning Xu, and Xin Geng. Decompositional generation process for instance-dependent
partial label learning. In _International Conference on Learning Representations_, 2023.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. Highresolution image synthesis with latent diffusion models. In _IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition_, pp. 23464–23473, 2022.


13


Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic
text-to-image diffusion models with deep language understanding. pp. 36479–36494, 2022.


Ahmad Sajedi, Samir Khaki, Ehsan Amjadian, Lucy Z. Liu, Yuri A. Lawryshyn, and Konstantinos N.
Plataniotis. Datadam: Efficient dataset distillation with attention matching. In _Proceedings of the_
_IEEE/CVF International Conference on Computer Vision_, pp. 17097–17107, 2023.


Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In
_International Conference on Learning Representations_, 2022.


Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen.
Improved techniques for training gans. In _Advances in Neural Information Processing Systems_, pp.
2226–2234, 2016.


Shitong Shao, Zikai Zhou, Huanran Chen, and Zhiqiang Shen. Elucidating the design space of dataset
condensation. In _Advances in Neural Information Processing Systems_, pp. 99161–99201, 2024.


Vinay Shukla, Zhe Zeng, Kareem Ahmed, and Guy Van den Broeck. A unified approach to countbased weakly supervised learning. In _Advances in Neural Information Processing Systems_, pp.
38709–38722, 2023.


Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning
with consistency and confidence. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, pp.
596–608, 2020.


Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
In _Advances in Neural Information Processing Systems_, pp. 11895–11907, 2019.


Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In
_Advances in Neural Information Processing Systems_, pp. 12438–12448, 2020.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In _International_
_Conference on Learning Representations_, 2020.


Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. In _Advances in Neural Information Processing Systems_, pp. 1415–1428,
2021.


Peng Sun, Bei Shi, Daiwei Yu, and Tao Lin. On the diversity and realism of distilled dataset: An
efficient dataset distillation paradigm. In _IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pp. 9390–9399, 2024.


Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking
the inception architecture for computer vision. In _IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 2818–2826, 2016.


Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Yuuki Yamanaka, and Tomoya Yamashita.
Positive-unlabeled diffusion models for preventing sensitive data generation. In _International_
_Conference on Learning Representations_, 2025.


Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In _Advances in Neural Information_
_Processing Systems_, pp. 1195–1204, 2017.


Yingjie Tian, Xiaotong Yu, and Saiji Fu. Partial label learning: Taxonomy, analysis and outlook.
_Neural Networks_, 161:708–734, 2023.


Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural computa-_
_tion_, 23(7):1661–1674, 2011.


14


Haobo Wang, Mingxuan Xia, Yixuan Li, Yuren Mao, Lei Feng, Gang Chen, and Junbo Zhao. Solar:
Sinkhorn label refinery for imbalanced partial-label learning. In _Advances in Neural Information_
_Processing Systems_, pp. 8104–8117, 2022a.


Haobo Wang, Ruixuan Xiao, Yixuan Li, Lei Feng, Gang Niu, Gang Chen, and Junbo Zhao. Pico+:
Contrastive label disambiguation for robust partial label learning. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence_, pp. 3183–3198, 2023.


Hsiu-Hsuan Wang, Tan-Ha Mai, Nai-Xuan Ye, Wei-I Lin, and Hsuan-Tien Lin. Climage: Humanannotated datasets for complementary-label learning. _Transactions on Machine Learning Research_,
2025, 2025a.


Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Shuo Yang, Shuo Wang, Guan Huang, Hakan
Bilen, Xinchao Wang, and Yang You. Cafe: Learning to condense dataset by aligning features. In
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 12196–12205, 2022b.


Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A Efros. Dataset distillation. _arXiv_
_preprint arXiv:1811.10959_, 2018.


Wei Wang, Dong-Dong Wu, Jindong Wang, Gang Niu, Min-Ling Zhang, and Masashi Sugiyama.
Realistic evaluation of deep partial-label learning algorithms. In _International_ _Conference_ _on_
_Learning Representations_, 2025b.


Yidong Wang, Hao Chen, Yue Fan, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, Linyi Yang, Zhi
Zhou, Lan-Zhe Guo, et al. Usb: A unified semi-supervised learning benchmark for classification.
In _Advances in Neural Information Processing Systems_, pp. 3938–3961, 2022c.


Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios
Savvides, Takahiro Shinozaki, Bhiksha Raj, et al. Freematch: Self-adaptive thresholding for
semi-supervised learning. In _International Conference on Learning Representations_, 2022d.


Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross
entropy for robust learning with noisy labels. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _International_
_Conference on Computer Vision_, pp. 322–330, 2019.


Hongxin Wei, Lei Feng, Xiangyu Chen, and Bo An. Combating noisy labels by agreement: A joint
training method with co-regularization. In _IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pp. 13726–13735, 2020.


Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning with
noisy labels revisited: A study using real-world human annotations. In _International Conference_
_on Learning Representations_, 2021.


Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning with
noisy labels revisited: A study using real-world human annotations. In _International Conference_
_on Learning Representations_, 2022.


Zixi Wei, Lei Feng, Bo Han, Tongliang Liu, Gang Niu, Xiaofeng Zhu, and Heng Tao Shen. A
universal unbiased method for classification from aggregate observations. In _Proceedings of the_
_International Conference on Machine Learning_, pp. 36804–36820, 2023.


Hongwei Wen, Jingyi Cui, Hanyuan Hang, Jiabin Liu, Yisen Wang, and Zhouchen Lin. Leveraged
weighted loss for partial label learning. In _Proceedings of the International Conference on Machine_
_Learning_, pp. 11091–11100, 2021.


Dong-Dong Wu, Deng-Bao Wang, and Min-Ling Zhang. Revisiting consistency regularization for
deep partial label learning. In _Proceedings of the International Conference on Machine Learning_,
pp. 24212–24225, 2022.


Dong-Dong Wu, Deng-Bao Wang, and Min-Ling Zhang. Distilling reliable knowledge for instancedependent partial label learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
pp. 15888–15896, 2024.


15


Shiyu Xia, Jiaqi Lv, Ning Xu, and Xin Geng. Ambiguity-induced contrastive learning for instancedependent partial label learning. In _Proceedings of the International Joint Conference on Artificial_
_Intelligence_, pp. 3615–3621, 2022.


Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking
machine learning algorithms. _arXiv preprint arXiv:1708.07747_, 2017.


Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li,
Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion
transformers. In _International Conference on Learning Representations_, 2025.


Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. Unsupervised data augmentation
for consistency training. In _Advances in Neural Information Processing Systems_, pp. 6256–6268,
2020.


Zheng Xie, Yu Liu, Hao-Yuan He, Ming Li, and Zhi-Hua Zhou. Weakly supervised auc optimization:
A unified partial auc approach. _IEEE Transactions on Pattern Analysis and Machine Intelligence_,
46(7):4780–4795, 2024.


Ning Xu, Biao Liu, Jiaqi Lv, Congyu Qiao, and Xin Geng. Progressive purification for instancedependent partial label learning. In _Proceedings_ _of_ _the_ _International_ _Conference_ _on_ _Machine_
_Learning_, pp. 38551–38565, 2023.


Yi Xu, Lei Shang, Jinxing Ye, Qi Qian, Yu-Feng Li, Baigui Sun, Hao Li, and Rong Jin. Dash: Semisupervised learning with dynamic thresholding. In _Proceedings of the International Conference on_
_Machine Learning_, pp. 11525–11536, 2021.


Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, and Haohan Wang. Towards adversarially robust dataset distillation by curvature regularization. In _Proceedings of the AAAI Conference_
_on Artificial Intelligence_, pp. 9041–9049, 2025.


Muqiao Yang, Chunlei Zhang, Yong Xu, Zhongweiyang Xu, Heming Wang, Bhiksha Raj, and Dong
Yu. Usee: Unified speech enhancement and editing with conditional diffusion models. In _IEEE_
_International Conference on Acoustics, Speech and Signal Processing_, pp. 7125–7129, 2024.


Xiangli Yang, Zixing Song, Irwin King, and Zenglin Xu. A survey on deep semi-supervised learning.
_IEEE transactions on knowledge and data engineering_, 35(9):8934–8954, 2022.


Yu Yao, Tongliang Liu, Bo Han, Mingming Gong, Jiankang Deng, Gang Niu, and Masashi Sugiyama.
Dual t: Reducing estimation error for transition matrix in label-noise learning. pp. 7260–7271,
2020.


Kun Yi and Jianxin Wu. Probabilistic end-to-end noise correction for learning with noisy labels. In
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 7017–7025, 2019.


Zeyuan Yin and Zhiqiang Shen. Dataset distillation via curriculum data synthesis in large data era.
_Transactions on Machine Learning Research_, 2024.


Zeyuan Yin, Eric Xing, and Zhiqiang Shen. Squeeze, recover and relabel: Dataset condensation at
imagenet scale from a new perspective. In _Advances in Neural Information Processing Systems_,
pp. 73582–73603, 2024.


Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor Tsang, and Masashi Sugiyama. How does
disagreement help generalization against label corruption? In _Proceedings of the International_
_Conference on Machine Learning_, pp. 7164–7173, 2019.


Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro
Shinozaki. Flexmatch: Boosting semi-supervised learning with curriculum pseudo labeling. In
_Advances in Neural Information Processing Systems_, pp. 18408–18419, 2021a.


Fei Zhang, Lei Feng, Bo Han, Tongliang Liu, Gang Niu, Tao Qin, and Masashi Sugiyama. Exploiting class activation value for partial-label learning. In _International Conference on Learning_
_Representations_, 2021b.


16


Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical
risk minimization. In _International Conference on Learning Representations_, 2018.


Yivan Zhang, Nontawat Charoenphakdee, Zhenguo Wu, and Masashi Sugiyama. Learning from
aggregate observations. In _Advances in Neural Information Processing Systems_, pp. 7993–8005,
2020.


Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks
with noisy labels. In _Advances in Neural Information Processing Systems_, pp. 8792–8802, 2018.


Bo Zhao and Hakan Bilen. Dataset condensation with differentiable siamese augmentation. In
_Proceedings of the International Conference on Machine Learning_, pp. 12674–12685, 2021.


Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In _Proceedings of the_
_IEEE/CVF winter conference on applications of computer vision_, pp. 6514–6523, 2023.


Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching. In
_International Conference on Learning Representations_, 2021.


Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information
interaction with frequency diffusion adjustment for underwater image restoration. In _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pp. 8281–8291, 2024.


Mingkai Zheng, Shan You, Lang Huang, Fei Wang, Chen Qian, and Chang Xu. Simmatch: Semisupervised learning with similarity matching. In _IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 14471–14481, 2022.


17


CONTENTS


**1** **Introduction** **1**


**2** **Related Work** **2**


2.1 Robust Diffusion Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2


2.2 Imprecise Label Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2


2.3 Dataset Condensation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2


**3** **Background** **3**


**4** **Methodology** **4**


4.1 Unified Learning Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4


4.2 Generative Objective: Modeling the Imprecise Data Distribution . . . . . . . . . . 4


4.3 Classification Objective: Inferring Labels from Imprecise Signals . . . . . . . . . . 5


**5** **Time Complexity Reduction** **6**


**6** **Experiments** **8**


6.1 Task1: Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8


6.2 Task2: Weakly Supervised Learning . . . . . . . . . . . . . . . . . . . . . . . . . 9


6.3 Task3: Noisy Dataset Condensation . . . . . . . . . . . . . . . . . . . . . . . . . 9


**7** **Conclusion** **10**


**A** **Notation and Definitions** **20**


**B** **Proof** **20**


B.1 Connections among Different Diffusion Models. . . . . . . . . . . . . . . . . . . . 20


B.2 Derivation of Eq. (5) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


B.3 Derivation of Varitional Lower Bound Eq. (7) . . . . . . . . . . . . . . . . . . . . 23


B.4 Derivation of Conditional ELBO in Eq. (8) . . . . . . . . . . . . . . . . . . . . . . 24


B.5 Derivation of Remark 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


B.6 Proof of Theorem 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


B.7 Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


B.8 Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


B.9 Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28


**C** **Disccusion** **29**


C.1 Analysis of Early-Learning Regularization in Eq. (15) . . . . . . . . . . . . . . . . 29


C.2 Class-Prior Estimation in Imprecise-Label Datasets . . . . . . . . . . . . . . . . . 30


**D** **Implementation Details** **31**


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


**E** **Experiments** **32**


E.1 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32


E.2 Full Results in Weakly Supervised Learning . . . . . . . . . . . . . . . . . . . . . 32


E.3 Integration with Existing Imprecise-Label Correctors . . . . . . . . . . . . . . . . 34


E.4 Comparison of Accuracy Curves between _DMIS_ and _Vanilla_ . . . . . . . . . . . . 34


E.5 Visualization of Noisy Condensed Datasets . . . . . . . . . . . . . . . . . . . . . 34


E.6 Additional Results on Dataset Condensation under Different Forms of Imprecise
Supervision . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35


**F** **Additional Experiments Results** **37**


F.1 The Full Rsults of Table 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37


F.2 Comparison against Other Noise-robust Diffusion Methods. . . . . . . . . . . . . . 37


F.3 Top- _k_ truncation for large label spaces . . . . . . . . . . . . . . . . . . . . . . . . 38


F.4 Experiments beyond synthetic class-conditional noise . . . . . . . . . . . . . . . . 38


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


**The Use of Large Language Models (LLMs).** LLMs were only used for language polishing and
proofreading. No part of the technical content, experiments, or analysis was generated by LLMs.


A NOTATION AND DEFINITIONS


We present the notation table for each symbol used in this paper in Table 4.


Table 4: List of common mathematical symbols used in this paper.

**Symbol** **Definition**


**x** A sample of training data
_z_ Imprecise label associated with a sample
_s_ Candidate label set for a sample
_y_ Class index label
_c_ Total number of classes
_X_ Input space from which **x** is drawn
_Y_ Label space from which _y_ is drawn
_X_ Random variable for training instances
_Y_ Random variable for true labels
_Z_ Random variable for imprecise labels
_S_ Random variable for partial labels
_Y_ ˆ Random variable for noisy labels
_X_ [l] Set of labeled data instances
_X_ [u] Set of unlabeled data instances
_Y_ [l] Set of labels corresponding to _X_ [l]
_∅_ Empty label set
_θ_ Parameters of the diffusion model to be optimized
_ϕ_ Exponential moving average of _θ_ over training iteration
**0** Zero vector
**I** Identity matrix
**x** _t_ Noisy version of the sample at timestep _t_
_τ_ Continuous timestep variable
_αt_ Scaling factor at timestep _t_
_σt_ Noise scale at timestep _t_
_l_ Left boundary of a subsampled timestep interval
_r_ Right boundary of a subsampled timestep interval
∆ Length of a subsampled timestep interval
_q_ ( _·_ ) Real Data distribution
_q_ ( _· | ·_ ) Real conditional data distribution
_p_ ( _·_ ) Marginal probability distribution
_p_ ( _· | ·_ ) Model-infered conditional distribution
_F_ ( _·_ ) Cumulative distribution function of _p_ ( _·_ )
_f_ ( _·_ ) Diffusion classifier
**s** ( _·, ·_ ) Time-conditioned score prediction network
_N_ ( _·, ·_ ) Gaussian distribution


B PROOF


B.1 CONNECTIONS AMONG DIFFERENT DIFFUSION MODELS.


The diffusion model we define in this paper can be reformulated to align with other common diffusion
frameworks, such as DDPM (Ho et al., 2020), SMLD (Song & Ermon, 2019), VE-SDE (Song et al.,
2020) and VP-SDE (Song et al., 2020), as well as with approaches like x-prediction (Ho et al., 2020),
v-prediction (Salimans & Ho, 2022), and _ϵ_ -prediction (Ho et al., 2020). This demonstrates that
our formulation is compatible with diverse diffusion paradigms while facilitating unified theoretical
analysis. To better demonstrate this transformation, we present the following pseudocodes.


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


**Algorithm 1** Our models to EDM
**Require:** A score network **s** _θ_, a noisy input **x** _t_, noise level _t_, linear schedule _{αi}_ _[T]_ _i_ =1 [and] _[ {][σ][i][}]_ _i_ _[T]_ =1 [.]
1: Calculate the denoised image **x** 0 using **s** _θ_ : **x** 0 = ( **x** _t_ + _σt_ [2] **[s]** _[θ]_ [(] **[x]** _[t][/α][t][, σ][t][/α][t]_ [))] _[/α][t]_
2: **if** performing **x** 0-prediction **then**
3: **return x** 0.
4: **end if**
5: Calculate the noise component _**ϵ**_ : _**ϵ**_ = **[x]** _[t][−]_ _σ_ _[α]_ _t_ _[t]_ **[x]** [0]

6: **if** performing _**ϵ**_ -prediction **then**
7: **return** _**ϵ**_ .
8: **end if**
9: Calculate the noise component _**v**_ : _**v**_ = _αt_ _**ϵ**_ _−_ _σt_ **x** 0
10: **if** performing _**v**_ -prediction **then**
11: **return** _**v**_ .
12: **end if**


**DDPM.** DDPM define a sequence _{βt}_ _[T]_ _t_ =0 [and] **[ x]** _[t]_ [=] �� _ti_ =0 [(1] _[ −]_ _[β][i]_ [)] **[x]** [0][ +] �1 _−_ [�] _[t]_ _i_ =0 [(1] _[ −]_ _[β][i]_ [)] _**[ϵ]**_ [,]

which can be seen as a special case of Eq. (1) where we can set _αt_ = �� _ti_ =0 [(1] _[ −]_ _[β][i]_ [)][ and] _[ σ][t]_ [=]

1 _−_ [�] _[t]_ _i_ =0 [(1] _[ −]_ _[β][i]_ [)][.]


**SMLD.** SMLD defines a noise schedule _σ_ ( _t_ ) _[T]_ _t_ =0 [and] **[ x]** _[t]_ [=] **[x]** [0] [+] _[ σ]_ [(] _[t]_ [)] _**[ϵ]**_ [, with] _[ σ]_ [(1)] _[<]_ _[σ]_ [(2)] _[<]_ _[· · ·]_ _[<]_
_σ_ ( _T_ ). In this setup, Eq. (1) reduces to _αt_ = 1, _σt_ = _σ_ ( _t_ ).


**VP-SDE.** VP-SDE is the continuous case of DDPM, which define a stochastic differential equation
(SDE) as


_dXt_ = _−_ [1]          - _β_ ( _t_ ) _dWt,_ _t ∈_ [0 _,_ 1] _,_

2 _[β]_ [(] _[t]_ [)] _[X][t][dt]_ [ +]


         where _β_ ( _t_ ) = _βt·T_ _· T_ . In this setup, _αt_ = exp - _−_ �0 _t_ _[β]_ [(] _[s]_ [)] _[ds]_ �, _σt_ = 1 _−_ exp - _−_ �0 _t_ _[β]_ [(] _[s]_ [)] _[ds]_ �.


**VE-SDE.** VE-SDE is the continuous case of SMLD, whose forward process of VE-SDE is defined as


In this setup, _αt_ = 1 and _σt_ = ~~�~~ _σ_ [2] ( _t_ ) _−_ _σ_ [2] (0).


While the models above each define their own specific frameworks for the diffusion process,
EDM (Karras et al., 2022) proposes a unified structure and optimizes the parameters choice within
the diffusion process, making it both robust and adaptable. Therefore, for our implementation, we
adopt EDM as the foundational diffusion model. In EDM, the scaling and noise schedules are a
special case of VE-SDE, where the variance of the noise is given by _σ_ ( _t_ ) = _t_ . Accordingly, we use
**s** _θ_ ( **x** _/αt, σt/αt_ ) to obtain the predicted score, as shown in Algorithm 1.


B.2 DERIVATION OF EQ. (5)


Maximizing the variational lower bound, or equivalently evidence lower bound (ELBO), to optimize
the diffusion model is a common approach. To avoid redundant proofs, we directly use the conclusion
from Eq. (58) in Luo (2022) as below:


        log _pθ_ ( **x** ) _≥_ E _q_ [ _−D_ KL( _q_ ( **x** _T |_ **x** 0) _∥p_ ( **x** _T_ ))+log _pθ_ ( **x** 0 _|_ **x** 1) _−_ _D_ KL( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _∥pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ))]


_t>_ 1

Although each KL divergence term _D_ KL( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _∥pθ_ ( **x** _t−_ 1 _|_ **x** _t_ )) is difficult to minimize for
arbitrary posteriors, we can leverage the Gaussian transition assumption to make optimization
tractable. By Bayes rule, we have:

_q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) = _[q]_ [(] **[x]** _[t][|]_ **[x]** _[t][−]_ [1] _[,]_ **[ x]** [0][)] _[q]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** [0][)]

_q_ ( **x** _t|_ **x** 0)


21


_dXt_ =


~~�~~ _dσ_ ( _t_ ) [2]

_dWt._
_dt_


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


As we already know that _q_ ( **x** _t|_ **x** 0) and _q_ ( **x** _t−_ 1 _|_ **x** 0) from Eq. (1), _q_ ( **x** _t|_ **x** _t−_ 1 _,_ **x** 0) can be derived from
its equivalent form _q_ ( **x** _t|_ **x** _t−_ 1) as follows:


**x** _t_ = _αt_ **x** 0 + _σt_ _**ϵ**_ 0

= _αt_ ( **[x]** _[t][−]_ [1] _[ −]_ _[σ][t][−]_ [1] _**[ϵ]**_ 0 _[∗]_ ) + _σt_ _**ϵ**_ 0
_αt−_ 1

_αt_ _αt_
= _αt−_ 1 **x** _t−_ 1 + _σt_ _**ϵ**_ 0 _−_ _αt−_ 1 _σt−_ 1 _**ϵ**_ _[∗]_ 0


where in the fourth Equation, _C_ ( **x** _t,_ **x** 0) is a constant term with respect to **x** _t−_ 1 computed as a
combination of only **x** _t_, **x** 0, and _α_ values. We have therefore shown that at each step, **x** _t−_ 1 _∼_


22


_αt_
= **x** _t−_ 1 +
_αt−_ 1


~~�~~

_σt_ [2] _[−]_ _αt_ [2] _σt_ [2] _−_ 1 _**[ϵ]**_ _[t][−]_ [1]
_αt_ [2] _−_ 1


= _N_ ( **x** _t_ ; _[α][t]_ **x** _t−_ 1 _, σt_ [2] _[−]_ _αt_ [2] _σt_ [2] _−_ 1 **[I]** [)] _[.]_

_αt−_ 1 _αt_ [2] _−_ 1


Now, knowing the forms of _q_ ( **x** _t|_ **x** _t−_ 1 _,_ **x** 0), we can proceed to calculate the form of _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0)
by substituting into the Bayes rule expansion:

_q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) = _[q]_ [(] **[x]** _[t][|]_ **[x]** _[t][−]_ [1] _[,]_ **[ x]** [0][)] _[q]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** [0][)]

_q_ ( **x** _t|_ **x** 0)


    _N_ ( **x** _t_ ; _ααt−t_ 1 **[x]** _[t][−]_ [1] _[,]_ _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 **[I]** [)] _[N]_ [(] **[x]** _[t][−]_ [1][;] _[ α][t][−]_ [1] **[x]** [0] _[, σ][t][−]_ [1] **[I]** [)]
=

_N_ ( **x** _t_ ; _αt_ **x** 0 _, σt_ **I** )





_t_
_αt−_ 1 **[x]** _[t][−]_ [1][)]

2( [(] **[x]** _σ_ _[t][ −]_ _t_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] + [(] **[x]** _[t][−]_ [1] _[ −]_ 2 _σ_ _[α]_ _t_ [2] _−_ _[t]_ 1 _[−]_ [1] **[x]** [0][)][2]


_[ −]_ _[α][t][−]_ [1] **[x]** [0][)][2]

_−_ [(] **[x]** _[t][ −]_ _[α][t]_ **[x]** [0][)][2]
2 _σt_ [2] _−_ 1 2 _σt_ [2]


2 _σt_ [2]






_∝_ exp


= exp


_∝_ exp


= exp


= exp


= exp


= exp





 _[−]_


_−_ [1]


_αt_
_αt−_ 1 **[x]** _[t][−]_ [1][)][2]
 [(] **[x]** _[t][ −]_ [2]


















_t_ **[x]**

**[x]** _[t]_ **[x]** _[t][−]_ [1][ + (] _αt−_ 1 [)] _t−_ 1 _t−_ 1 _[−]_ [2] _[α][t][−]_ [1] **[x]** _[t][−]_ [1] **[x]** [0]

+ **[x]** [2]
_σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _−_ 1


+ _C_ ( **x** _t,_ **x** 0)
_σt_ [2] _−_ 1











2

 _[−]_ [1]









2


_αt_ _αt_
_αt−_ 1 **[x]** _[t]_ **[x]** _[t][−]_ [1][ + (] _αt−_ 1 [)][2] **[x]** _t_ [2] _−_ 1
 _[−]_ [2] [2]


















_αt−_ 1 _[t]_ + _[α][t][−]_ [1] **[x]** [0]

_σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _−_ 1


_σt_ [2] _−_ 1








 **x** [2] _t−_ 1 _[−]_ [2]


_αt_
_αt−_ 1 **[x]** _[t]_





 **x** _t−_ 1

















2

 _[−]_ [1]


_αt−_ 1 + 1

_σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _−_ 1


2





( _ααt−t_ 1 [)][2]
 [2]






_σt_ [2] _−_ 1








2

 _[−]_ [1]








_αt−_ 1 _[t]_ + _[α][t][−]_ [1] **[x]** [0]

_σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _−_ 1


_αt_
_αt−_ 1 **[x]** _[t]_











 **x** _t−_ 1


2


 _σt_ [2] _−_ 1 [(] _ααt−t_ 1 [)][2][ + (] _[σ]_ _t_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)]

 [2]


( _σtt_ [2] 1 _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[σ]_ _tt_ [2] _−−_ 11 **x** [2] _t−_ 1 _[−]_ [2]






_αt−_ 1 _[t]_ + _[α][t][−]_ [1] **[x]** [0]

_σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _−_ 1


_σt_ [2] _−_ 1




















 **x** _t−_ 1





2

 _[−]_ [1]


_αt_
_αt−_ 1 **[x]** _[t]_


2




 ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _tσ_ 1 _t_ [2] _[σ]_ _t_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1 **x** [2] _t−_ 1 _[−]_ [2]






_σt_ [2]

( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _σt_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1


_αt_
_αt−_ 1 **[x]** _[t]_













 **x** _t−_ 1



















 




**x** [2] _t−_ 1 _[−]_ [2] 



 


_σt_ [2] _[−]_ _ααtα−_ [2] _t−_ [2] _t_ 11 _σ_ _[t]_ _t_ [2] _−_ 1 + _[α]_ _σ_ _[t][−]_ _t_ [2] _−_ [1] **[x]** 1 [0]


_σt_ [2] _−_ 1


_−_ [1]




_σt_ [2]
 ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1









2







  **x** [2] _t−_ 1 _[−]_ [2]












_αt_ _α_ [2] _t_
_αt−_ 1 **[x]** _[t][σ]_ _t_ [2] _−_ 1 [+ (] _[σ]_ _t_ [2] _[−]_ _α_ [2] _t−_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[α][t][−]_ [1] **[x]** [0]

_σt_ [2]





 **x** _t−_ 1









 1

 ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _σt_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1

_σt_ [2]











_∝N_ ( **x** _t−_ 1;


2


_[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[α][t][−]_ [1] **[x]** [0] ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1

_,_
_σt_ [2] _σt_ [2]


_αt_ _α_ [2] _t_
_αt−_ 1 **[x]** _[t][σ]_ _t_ [2] _−_ 1 [+ (] _[σ]_ _t_ [2] _[−]_ _α_ [2] _t−_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[α][t][−]_ [1] **[x]** [0]


**I** )
_σt_ [2]


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


_q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) is normally distributed, with mean _**µ**_ _q_ ( **x** _t,_ **x** 0) that is a function of **x** _t_ and **x** 0, and
variance **Σ** _q_ ( _t_ ) as a function of _α_ and _σ_ coefficients. These coefficients are known and fixed at each
timestep; they are either set permanently when modeled as hyperparameters, or treated as the current
inference output of a network that seeks to model them.


We can then set the variances of the two Gaussians to match exactly, optimizing the KL Divergence
term reduces to minimizing the difference between the means of the two distributions:


arg min _D_ KL( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _∥pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ))
_θ_


_[X, Y, Z]_ [)] 
_dY_ _−_ _Q_ ( _Y_ ) log _[p][θ]_ [(] _[Y][ |][X, Z]_ [)]
_Q_ ( _Y_ ) _Q_ ( _Y_ )


= arg min
_θ_


1
2 _σq_ [2] ( _t_ )


- _∥_ _**µ**_ _θ_ ( **x** _t, t_ ) _−_ _**µ**_ _q_ ( **x** _t,_ **x** 0) _∥_ [2] 2 _,_ (18)


where _σq_ [2][(] _[t]_ [) =] ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _σtσ_ [2] _t_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1, the derivation is the same as in Eq. (92) in Luo (2022), so we skip

the derivation here. To derive the score matching funciton, we appeal to Tweedie’s Formula Efron
(2011), which states E[ _**µ**_ _z|z_ ] = _z_ + **Σ** _z∇z_ log _q_ ( _z_ ) for a given Gausssion variable _z_ _∼N_ ( _z_ ; _**µ**_ _z,_ **Σ** _z_ ).
In this case, we apply it to predict the true posterior mean of **x** _t_ given its samples. We can obtain:

E[ _**µ**_ **x** _t|_ **x** _t_ ] = **x** _t_ + _σt_ [2] _[∇]_ **[x]** _t_ [log] _[ q]_ [(] **[x]** _[t]_ [) =] _[ α][t]_ **[x]** [0]

∴ **x** 0 = **[x]** _[t]_ [ +] _[ σ]_ _t_ [2] _[∇]_ **[x]** _t_ [log] _[ q]_ [(] **[x]** _[t]_ [)] (19)

_αt_

Then, we can plug Eq. (19) into our ground-truth denoising transition mean _**µ**_ _q_ ( **x** _t,_ **x** 0) once again
and derive a new form:


_**µ**_ _q_ ( **x** _t,_ **x** 0) =


=


_ααt−t_ 1 _[σ]_ _t_ [2] _−_ 1 **[x]** _[t]_ [+ (] _[σ]_ _t_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[α][t][−]_ [1] _[·]_ **[x]** _[t]_ [+] _[σ]_ _t_ [2] _[∇]_ **[x]** _αtt_ [log] _[ q]_ [(] **[x]** _[t]_ [)]

_σt_ [2]


_ααt−t_ 1 _[σ]_ _t_ [2] _−_ 1 _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1
_σt_ [2] **x** _t_ + _σt_ [2] _ααt_


_αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _[σ]_ _t_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _[∇]_ **[x]** _t_ [log] _[ q]_ [(] **[x]** _[t]_ [)]

_σt_ [2] _ααt−t_ 1 **x** _t_ + _σt_ [2] _ααt−t_ 1


_σt_ [2] _ααt−t_ 1


    - _αt−_ 1 _αt_

= _[α][t][−]_ [1] **x** _t_ + _σt_ [2] _[−]_ _σt_ [2] _−_ 1

_αt_ _αt_ _αt−_ 1


_∇_ **x** _t_ log _q_ ( **x** _t_ )


= _[α][t][−]_ [1]

_αt_


- **x** _t_ + - _σt_ [2] _[−]_ _αt_ [2] _σt_ [2] _−_ 1
_αt_ [2] _−_ 1


- **s** _θ_ ( **x** _t, t_ ) (20)


Finally, we plug Eq. (20) into our optimization function Eq. (18), and we can get:


arg min _D_ KL( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _∥pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ))
_θ_


= arg min
_θ_


= arg min
_θ_


1
2 _σq_ [2] ( _t_ )


1

2 ( _σt_ [2] _[−]_ _αα_ [2] _t−_ [2] _t_ 1 _σt_ [2] _−_ 1 [)] _[σ]_ _t_ [2] _−_ 1

_σt_ [2]


- _∥_ _**µ**_ _θ_ ( **x** _t, t_ ) _−_ _**µ**_ _q_ ( **x** _t,_ **x** 0) _∥_ [2] 2


_·_ ( _[α][t][−]_ [1] ) [2] _·_ ( _σt_ [2] _[−]_ _αt_ [2] _σt_ [2] _−_ 1 [)][2] _[∥]_ **[s]** _[θ]_ [(] **[x]** _[t][, t]_ [)] _[ −∇]_ **[x]** _t_ [log] _[ q]_ [(] **[x]** _[t]_ [)] _[∥]_ 2 [2]
_αt_ _αt_ [2] _−_ 1


_t_ _t_ _[α]_ _t_ [2] _−_ 1
= _[σ]_ [2] _−_ 1) _∥_ **s** _θ_ ( **x** _t, t_ ) _−∇_ **x** _t_ log _q_ ( **x** _t_ ) _∥_ [2] 2
2 [(] _[σ]_ _σt_ [2][2] _−_ 1 _[α]_ _t_ [2]


B.3 DERIVATION OF VARITIONAL LOWER BOUND EQ. (7)


To model log _pθ_ ( _X, Z_ ), we introduce an auxiliary distribution _Q_ ( _Y_ ) over the latent variable _Y_ :


          log _pθ_ ( _X, Z_ ) = _Q_ ( _Y_ ) log _pθ_ ( _X, Z_ ) _dY_


          = _Q_ ( _Y_ ) log _pθ_ ( _X, Z_ ) _[p][θ]_ [(] _[Y][ |][X, Z]_ [)]

_pθ_ ( _Y |X, Z_ ) _[dY]_


 = _Q_ ( _Y_ ) log _[p][θ]_ [(] _[X, Y, Z]_ [)]


_dY,_
_Q_ ( _Y_ )


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


where the first term is the ELBO and the second term is the KL divergence _D_ KL( _Q_ ( _Y_ ) _||pθ_ ( _Y |X, Z_ )).
Since the KL divergence is non-negative, maximizing the ELBO provides a valid surrogate for
maximizing log _pθ_ ( _X, Z_ ). Replacing _Q_ ( _Y_ ) with _pϕ_ ( _Y |X, Z_ ) at each iteration will obtain as follows:

_θ_ _[∗]_ =arg max log _pθ_ ( _X, Z_ )
_θ_

=arg max E _pϕ_ ( _Y |X,Z_ )[log _pθ_ ( _X, Y, Z_ )]
_θ_

=arg max E _pϕ_ ( _Y |X,Z_ )[log _pθ_ ( _X|Z_ ) + log _pθ_ ( _Y |X, Z_ ) + log _pθ_ ( _Z_ )]
_θ_

=arg max E _pϕ_ ( _Y |X,Z_ )[log _pθ_ ( _X|Z_ )] + E _pϕ_ ( _Y |X,Z_ )[log _pθ_ ( _Y |X, Z_ )]
_θ_

=arg max log _pθ_ ( _X|Z_ ) + E _pϕ_ ( _Y |X,Z_ )[log _pθ_ ( _Y |X, Z_ )] _._
_θ_


which is exactly the variational lower bound presented in Eq. (7).


B.4 DERIVATION OF CONDITIONAL ELBO IN EQ. (8)


We provide a derivation of conditional ELBO in the following, which is similar to the unconditional
ELBO in Ho et al. (2020).

log _pθ_ ( **x** 0 _|_ _z_ )

   - _pθ_ ( **x** 0: _T |_ _z_ ) _q_ ( **x** 1: _T |_ **x** 0 _, z_ )
= log _d_ **x** 1: _T_
_q_ ( **x** 1: _T |_ **x** 0 _, z_ )


    -     = _−_ E _t_ _wt ∥_ **s** _θ_ ( **x** _t, z, t_ ) _−∇_ log _q_ ( **x** _t |_ **x** 0 _, z_ ) _∥_ [2] 2 + _C_ 2 _._


We get the result of Eq. (8).


B.5 DERIVATION OF REMARK 1


Although this result follows directly from prior studies (Vincent, 2011; Song & Ermon, 2019), we provide a brief derivation here for completeness. Let _L_ DSM( _θ_ ; _q_ ( _X, Y_ )) and _L_ ESM( _θ_ ; _q_ ( _X, Y_ )) denote


24


= log E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


- _pθ_ ( **x** _T |_ _z_ ) _pθ_ ( **x** 0: _T −_ 1 _|_ **x** _T, z_ ) _q_ ( **x** 1: _T |_ **x** 0 _, z_ )


_≥_ E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


=E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


=E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


=E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


=E _q_ ( **x** 1: _T |_ **x** 0 _,z_ )


log _[p][θ]_ [ (] **[x]** _[T][ |]_ _[z]_ [)] _[ p][θ]_ [ (] **[x]** [0:] _[T][ −]_ [1] _[ |]_ **[x]** _[T][, z]_ [)]

_q_ ( **x** 1: _T |_ **x** 0 _, z_ )


log _[p][θ]_ [ (] **[x]** _[T][ |]_ _[z]_ [)] _[ p][θ]_ [ (] **[x]** [0:] _[T][ −]_ [1] _[ |]_ **[x]** _[T][, z]_ [)]


 - _T −_ 1
_i_ =0 _[p][θ]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[, z]_ [)]
log _−_ log _[q]_ [ (] **[x]** _[T][ |]_ **[x]** [0] _[, z]_ [)]

 - _Ti_ =0 _−_ 1 _[q]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[,]_ **[ x]** [0] _[, z]_ [)] _pθ_ ( **x** _T |_ _z_ )


_i_ =0 _[p][θ]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[, z]_ [)]
log _[p][θ]_ [ (] **[x]** _[T][ |]_ _[z]_ [)][ �] _[T][ −]_ [1]

  - _T −_ 1
_i_ =0 _[q]_ [ (] **[x]** _[i]_ [+1] _[ |]_ **[x]** _[i][,]_ **[ x]** [0] _[, z]_ [)]





_i_ =0 _[p][θ]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[, z]_ [)]
log _[p]_ - _[θ]_ [ (] _T_ **[x]** _−_ _[T]_ 1 _[ |]_ _q_ _[z]_ ( [)] **x** [ �] _i_ +1 _[T]_ _|_ **x** _[ −]_ 0 [1] _,z_ ) _q_ ( **x** _i|_ **x** _i_ +1 _,_ **x** 0 _,z_ )
_i_ =0 _q_ ( **x** _|_ **x** _,z_ )








_q_ ( **x** _i|_ **x** 0 _,z_ )


                    
_i_ =0 _[p][θ]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[, z]_ [)]
log _[p][θ]_ [ (] **[x]** - _[T]_ _T_ _[ |]_ _−_ _[z]_ [)] 1 [ �] _[T][ −]_ [1] _−_ log _q_ ( **x** _T |_ **x** 0 _, z_ )
_i_ =0 _[q]_ [ (] **[x]** _[i][ |]_ **[x]** _[i]_ [+1] _[,]_ **[ x]** [0] _[, z]_ [)]


_−_ _D_ KL ( _q_ ( **x** _T |_ **x** 0 _, z_ ) _∥pθ_ ( **x** _T |_ _z_ ))


=


=


_T −_ 1


E _q_ ( **x** _i,_ **x** _i_ +1 _|_ **x** 0 _,z_ )

_i_ =0


_T −_ 1


E _q_ ( **x** _i_ +1 _|_ **x** 0 _,z_ )E _q_ ( **x** _i|_ **x** _i_ +1 _,_ **x** 0 _,z_ )

_i_ =0


- _pθ_ ( **x** _i |_ **x** _i_ +1 _, z_ )
log
_q_ ( **x** _i |_ **x** _i_ +1 _,_ **x** 0 _, z_ )


- _pθ_ ( **x** _i |_ **x** _i_ +1 _, z_ )
log
_q_ ( **x** _i |_ **x** _i_ +1 _,_ **x** 0 _, z_ )


_−_ _D_ KL ( _q_ ( **x** _T |_ **x** 0 _, z_ ) _∥pθ_ ( **x** _T |_ _z_ ))


= _C_ 3 _−_


_T −_ 1


E _q_ ( **x** _i_ +1 _|_ **x** 0 _,z_ ) [ _D_ KL ( _q_ ( **x** _i |_ **x** _i_ +1 _,_ **x** 0 _, z_ ) _∥pθ_ ( **x** _i |_ **x** _i_ +1 _, z_ ))]

_i_ =1


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


**T-shirt**


**Trouser**


**Pullover**


**Dress**


**Coat**


**Sandal**


**Shirt**


**Sneaker**


**Bag**


**Ankle boot**

(a) Partial-label supervision (b) Suppl-unlabeled supervision (c) Noisy-label supervision


Figure 3: Examples of randomly generated Fashion-MNIST images from _Vanilla_ models trained
under different types of imprecise supervision.


**Airplane**


**Automobile**


**Bird**


**Cat**


**Deer**


**Dog**


**Frog**


**Horse**


**Ship**


**Truck**

(a) Partial-label supervision (b) Suppl-unlabeled supervision (c) Noisy-label supervision


Figure 4: Examples of randomly generated CIFAR-10 images from _Vanilla_ models trained under
different types of imprecise supervision.


**Tench**


**English springer**


**Cassette player**


**Chain saw**


**Church**


**French horn**


**Garbage truck**


**Gas pump**


**Golf ball**


**Parachute**

(a) Partial-label supervision (b) Suppl-unlabeled supervision (c) Noisy-label supervision


Figure 5: Examples of randomly generated ImageNette images from _Vanilla_ models trained under
different types of imprecise supervision.


the denoising score matching (DSM) and explicit score matching (ESM) objectives, respectively:


It has been established (Vincent, 2011; Song & Ermon, 2019) that these two formulations differ only
by an additive constant independent of _θ_ :


_L_ ESM( _θ_ ; _q_ ( _X, Y_ )) = _L_ DSM( _θ_ ; _q_ ( _X, Y_ )) + _C_ 3 _,_

where _C_ 3 does not depend on _θ_ . Hence, both objectives admit the same minimizer.

Applying this result to an imprecise-label dataset by identifying _Y_ = _Z_, let _θ_ ESM _[∗]_ :=
arg min _θ L_ ESM( _θ_ ; _q_ ( _X, Z_ )). Then the optimal score function satisfies **s** _θ_ ESM _[∗]_ [(] **[x]** _[t][, z, t]_ [)] =
_∇_ **x** _t_ log _qt_ ( **x** _t | Z_ = _z_ ). Since the same conclusion holds for _L_ DSM, we obtain **s** _θ_ ESM _[∗]_ [=] **[s]** _[θ]_ DSM _[∗]_ [=]
_∇_ **x** _t_ log _qt_ ( **x** _t |_ _z_ ), which is precisely the statement of Remark 1.


25


           - 2
_L_ DSM( _θ_ ; _q_ ( _X, Y_ )):= E _t_ _λ_ ( _t_ )E _y∼q_ ( _Y_ )E **x** _t∼qt|_ 0( **x** _t|_ **x** _,y_ )�� **s** _θ_ ( **x** _t, y, t_ ) _−∇_ **x** _t_ log _qt|_ 0( **x** _t |_ **x** _, Y_ = _y_ )��2


_,_


           - 2
_L_ ESM( _θ_ ; _q_ ( _X, Y_ )):= E _t_ _λ_ ( _t_ )E _y∼q_ ( _Y_ )E **x** _t∼qt_ ( **x** _t|y_ )�� **s** _θ_ ( **x** _t, y, t_ ) _−∇_ **x** _t_ log _qt_ ( **x** _t |_ _Y_ = _y_ )��2


_._


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


To directly illustrate this bias, we train CDMs under different forms of imprecise supervision by
applying Eq. (8) directly, a baseline we refer to as _Vanilla_ . We then visualize the images generated by
these biased models, as shown in the figures below. The results reveal the following patterns:


    - **Partial-label supervision:** The generated images often lack diversity and typically capture
only the dominant object. This effect is particularly pronounced on the ImageNette dataset,
where samples within the same class appear highly similar. Interestingly, the generated
categories generally align with the ground-truth labels, suggesting that diffusion models can
still extract correct class information under partial-label supervision. However, the inherent
label ambiguity prevents the model from capturing intra-class variation.


    - **Noisy-label supervision:** The generated samples tend to contain visual noise. Although
the model is able to capture class diversity, corrupted labels cause mismatches between
generated samples and their true categories.


    - **Supplementary-unlabeled supervision:** The generated images are often both less diverse
and noisier. This phenomenon combines the limitations of partial-label supervision with
the challenge of abundant unlabeled samples. Because the model has limited access to
labeled examples, it relies on averaging confidence across all classes, which reduces its
discriminative boundaries and introduces noise.


B.6 PROOF OF THEOREM 1


The derivation here is analogous to that of Theorem 1 in Na et al. (2024), and we provide the full
proof below for completeness. First, for all _t_, the perturbed distribution _qt_ ( **x** _t|z_ ) satisified:


=


=


=


=


=


=


=


=


_qt_ ( **x** _t|z_ ) =


_c_

- _p_ ( _y|z_ ) _qt_ ( **x** _t|y_ ) _∀_ **x** _t_ _∈X_ _, z_ _⊂Y._


_y_ =1


This implies that the transition from imprecise labels to clean labels is independent of the timesteps.
Consequently, Eq. (9) can be derived as follows,


_∇_ **x** _t_ log _qt_ ( **x** _t|z_ )

= _[∇]_ **[x]** _[t][q][t]_ [(] **[x]** _[t][|][z]_ [)]

_qt_ ( **x** _t|z_ )


- _c_
_y_ =1 _[p]_ [(] _[y][|][z]_ [)] _[∇]_ **[x]** _t_ _[q][t]_ [(] **[x]** _[t][|][y]_ [)]

_qt_ ( **x** _t|z_ )


_c_

- _p_ ( _y|_ **x** _t, z_ ) _∇_ **x** _t_ log _qt_ ( **x** _t|y_ )

_y_ =1


_c_


_y_ =1


_c_


_y_ =1


_c_


_[p]_ [(] _[z]_ [)] _[p]_ [(] _[y][|]_ **[x]** _[t]_ [)]

_p_ ( _y_ ) _[·]_ _p_ ( _z|_ **x** _t_ )


_p_ ( _y|z_ ) _qt_ ( **x** _t|y_ ) _·_ _[∇]_ **[x]** _[t][q][t]_ [(] **[x]** _[t][|][y]_ [)]

_qt_ ( **x** _t|z_ ) _qt_ ( **x** _t|y_ )


_p_ ( _y|z_ ) _qt_ ( **x** _t|y_ ) _· ∇_ **x** _t_ log _qt_ ( **x** _t|y_ )

_qt_ ( **x** _t|z_ )


_qt_ ( **x** _t_ ) _[· ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][|][y]_ [)]


- _p_ ( _y|z_ ) _·_ _[p]_ [(] _[z]_ [)]

_p_ ( _y_ )

_y_ =1


_[p]_ [(] _[y][|]_ **[x]** _[t]_ [)] _[q][t]_ [(] **[x]** _[t]_ [)]

_p_ ( _z|_ **x** _t_ ) _[·]_ _qt_ ( **x** _t_ )


_c_


_p_ ( _z|_ **x** _t_ ) _[· ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][|][y]_ [)]


- _p_ ( _z|y_ ) _·_ _[p]_ [(] _[y][|]_ **[x]** _[t]_ [)]

_p_ ( _z|_ **x** _t_ )

_y_ =1


_c_


(Conditional indep. of _z_ and **x** _t_ given _y_ .)
_p_ ( _z|_ **x** _t_ ) _[· ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][|][y]_ [)]


- _p_ ( _z|y,_ **x** _t_ ) _·_ _[p]_ [(] _[y][|]_ **[x]** _[t]_ [)]

_p_ ( _z|_ **x** _t_ )

_y_ =1


_c_


_y_ =1


_p_ ( _z|y,_ **x** _t_ ) _p_ ( _y|_ **x** _t_ ) _· ∇_ **x** _t_ log _qt_ ( **x** _t|y_ )

_p_ ( _z|_ **x** _t_ )


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


B.7 PROOF OF PROPOSITION 1


By Remark 1 and Theorem 1, the optimal solution _θ_ Gen _[∗]_ [to Eq. (10) satisfies]


where erf( _x_ ) denotes the error function.


The median of this distribution _τ_ mid is the value at which the CDF equals 0 _._ 5, i.e., _F_ ( _τ_ mid) = 0 _._ 5.


To ensure that the selected subinterval allows signal-dominant early timesteps and noise-dominant
later timesteps to complement each other, we require the cumulative probability mass on either side
of the median to be equal. Formally, for subinterval boundaries ( _l, r_ ) with _r_ = _l_ + ∆, we enforce the
following symmetry condition:


_F_ ( _r_ ) _−_ _F_ ( _τ_ mid) = _F_ ( _τ_ mid) _−_ _F_ ( _l_ ) _._


Rewriting this with _r_ = _l_ + ∆ gives:


_F_ ( _l_ + ∆) + _F_ ( _l_ ) = 2 _F_ ( _τ_ mid) = 1 _._


This implicit equation defines the subinterval ( _l, l_ + ∆) such that the cumulative probability mass is
centered around the median of _p_ ( _τ_ ). To compute the left boundary _l_, we solve:


_l_ = Solve _τ_ ( _F_ ( _τ_ ) + _F_ ( _τ_ + ∆) _−_ 1 = 0) _,_ (21)


and set _r_ = _l_ + ∆. The solution can be obtained using any standard root-finding algorithm, such as
the Brent method (Brent, 2013).


27


_c_

 - _p_ ( _y_ _|_ **x** _t, z_ ) **s** _θ_ Gen _[∗]_ [(] **[x]** _[t][, y, t]_ [) =] _[ ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][ |]_ _[z]_ [) =]

_y_ =1


for all **x** _t_ _∈X_, _z_ _⊆Y_, and _t ∈_ [ _T_ ].


Next, recall the weighted denoising score matching loss:


_c_

- _p_ ( _y_ _|_ **x** _t, z_ ) _∇_ **x** _t_ log _qt_ ( **x** _t |_ _y_ ) _,_

_y_ =1


_c_


_wt_
���


2

- _p_ ( _y_ _|_ **x** _t, z_ ) _∇_ **x** _t_ log _qt_ ( **x** _t |_ _y_ )���2

_y_ =1


_._


_c_

- _p_ ( _y_ _|_ **x** _t, z_ ) **s** _θ_ ( **x** _t, y, t_ ) _−_


_y_ =1


_c_


2


_L_ Gen( _θ_ ) = E _t_


Differentiating with respect to **s** _θ_ ( **x** _t, y, t_ ) and setting the derivative to zero yields

_∂_      - **s** _θ_ Gen _[∗]_ [(] **[x]** _[t][, y, t]_ [)] _[ −∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][ |]_ _[y]_ [)]      - = 0 _._
_∂_ **s** _θ_ ( **x** _t, y, t_ ) _[L]_ [Gen][(] _[θ]_ [) = 2] _[w][t][ p]_ [(] _[y]_ _[|]_ **[x]** _[t][, z]_ [)]


Since _wt_ _>_ 0, for any _y_ such that _p_ ( _y_ _|_ **x** _t, z_ ) _>_ 0, the optimality condition implies


**s** _θ_ Gen _[∗]_ [(] **[x]** _[t][, y, t]_ [) =] _[ ∇]_ **[x]** _[t]_ [ log] _[ q][t]_ [(] **[x]** _[t][ |]_ _[y]_ [)] _[.]_


In particular, under the partial-label learning setting, if _p_ ( _y_ _|_ **x** _t, z_ ) = 0, the loss does not depend on
**s** _θ_ ( **x** _t, y, t_ ), and the equality can be established without loss of generality. This completes the proof.


B.8 PROOF OF THEOREM 2


We first consider the case where the timestep _τ_ is sampled from a log-normal distribution, as defined
in the EDM framework. Specifically,

ln( _τ_ ) _∼N_ ( _P_ mean _, P_ std [2] [)] _[,]_


where the parameters are set to _P_ mean = 1 _._ 2 and _P_ std = _−_ 1 _._ 2. Accordingly, the probability density
function of _τ_ is given by


1
_p_ ( _τ_ ) = ~~_√_~~
_τ P_ std


  - _[−]_ _[P]_ [mean][)][2]   
_−_ [(ln] _[ τ]_ _,_ _τ_ _>_ 0 _._
2 _π_ [exp] 2 _P_ std [2]


The corresponding cumulative distribution function (CDF) is denoted as:


_F_ ( _τ_ ) = [1]

2


�1 + erf - ln _τ_ _−_ _P_ ~~_√_~~ mean �� _,_
_P_ std 2


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


We then consider the DDPM setting, where the timestep _τ_ is uniformly sampled from a fixed interval.
Specifically, we assume _τ_ _∼U_ (0 _,_ 1), whose CDF is given by


_F_ ( _τ_ ) = _τ_


Under this distribution, the symmetry condition in Eq. (21) simplifies to


_l_ = Solve _τ_ ( _F_ ( _τ_ ) + _F_ ( _τ_ + ∆) _−_ 1 = 0)
= Solve _τ_ ( _τ_ + _τ_ + ∆ _−_ 1 = 0)

= [1] _[ −]_ [∆] _,_

2


Differentiating _L_ w.r.t. _l_ and _r_ gives


_∂L_ _p_ ( _l_ )

_∂l_ [= 2(] _[µ][′][ −]_ _[µ][′′]_ [)] _[ ·]_ _Z_ ( _l, r_ ) [(] _[µ][′][ −]_ [ℏ][(] _[l]_ [))] _[ −]_ _[λp]_ [(] _[l]_ [)] _[,]_


_∂L_ _p_ ( _r_ )

_∂r_ [= 2(] _[µ][′][ −]_ _[µ][′′]_ [)] _[ ·]_ _Z_ ( _l, r_ ) [(][ℏ][(] _[r]_ [)] _[ −]_ _[µ][′]_ [) +] _[ λp]_ [(] _[r]_ [)] _[.]_


Setting both derivatives to zero yields the necessary conditions


2( _µ_ _[′]_ _−_ _µ_ _[′′]_ )( _µ_ _[′]_ _−_ ℏ( _l_ )) = _λZ_ ( _l, r_ ) _,_ 2( _µ_ _[′]_ _−_ _µ_ _[′′]_ )(ℏ( _r_ ) _−_ _µ_ _[′]_ ) = _λZ_ ( _l, r_ ) _._


Equating the two expressions gives

_µ_ _[′]_ _−_ ℏ( _l_ ) = ℏ( _r_ ) _−_ _µ_ _[′]_ = _⇒_ _µ_ _[′]_ = 2 [1] �ℏ( _l_ ) + ℏ( _r_ )� _._


28


and thus _r_ = _l_ + ∆= [1] _[−]_ 2 [∆] [.] [This result implies that the optimal subinterval is symmetric around the]

midpoint of the distribution. In the special case where only a single timestep is used (i.e., ∆ _→_ 0),
the best estimate of the conditional ELBO occurs exactly at the median. As the sampled timestep
deviates further from the midpoint, classification accuracy tends to degrade. This observation aligns
with the empirical findings of Li et al. (2023), who reported that classification accuracy is maximized
near the median and declines towards the edges. Their use of evenly spaced timesteps centered
around the median further supports our strategy.


B.9 PROOF OF THEOREM 3


For clarity, we abbreviate ℏ( _τ, y_ ) as ℏ( _τ_ ), since the proof does not depend explicitly on _y_ . Define the
weighted integral of ℏ and the normalization factor over an interval [ _l, r_ ] as


    - _r_
_A_ ( _l, r_ ) :=


_r_ - _r_

ℏ( _τ_ ) _p_ ( _τ_ ) d _τ,_ _Z_ ( _l, r_ ) :=
_l_ _l_


_p_ ( _τ_ ) d _τ,_
_l_


so that the local expectation can be written as _µ_ _[′]_ = _A_ ( _l, r_ ) _/Z_ ( _l, r_ ). Let _µ_ _[′′]_ = E _τ_ _∼p_ ( _τ_ )[ℏ( _τ_ )] denote
the global expectation. The squared error objective in Eq. (16) then becomes


_g_ ( _l, r_ ) := ( _µ_ _[′]_ _−_ _µ_ _[′′]_ ) [2] _,_


subject to the probability-mass constraint _Z_ ( _l, r_ ) = _α_ .


We apply the method of Lagrange multipliers with


_r_

                  - [�]                   _L_ ( _l, r, λ_ ) := ( _µ_ _[′]_ _−_ _µ_ _[′′]_ ) [2] + _λ_ _p_ ( _τ_ ) d _τ_ _−_ _α_ _._

_l_


By the Leibniz rule, the derivatives of _A_ ( _l, r_ ) and _Z_ ( _l, r_ ) with respect to the interval boundaries are


_∂A_ ( _l, r_ )


( _l, r_ ) _∂A_ ( _l, r_ )

= _−_ ℏ( _l_ ) _p_ ( _l_ ) _,_
_∂l_ _∂r_


( _l, r_ ) _∂Z_ ( _l, r_ )

= ℏ( _r_ ) _p_ ( _r_ ) _,_
_∂r_ _∂l_


( _l, r_ ) _∂Z_ ( _l, r_ )

= _−p_ ( _l_ ) _,_
_∂l_ _∂r_


= _p_ ( _r_ ) _._
_∂r_


Hence, the derivatives of _µ_ _[′]_ = _A/Z_ are


_∂µ_ _[′]_ _p_ ( _l_ ) _∂µ_ _[′]_

_∂l_ [=] _Z_ ( _l, r_ ) [(] _[µ][′][ −]_ [ℏ][(] _[l]_ [))] _[,]_ _∂r_


_∂µ_ _[′]_


_∂µ_ _p_ ( _r_ )

_∂r_ [=] _Z_ ( _l, r_ ) [(][ℏ][(] _[r]_ [)] _[ −]_ _[µ][′]_ [)] _[.]_


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


_Proof._ For any _i ∈{_ 1 _, . . ., n}_, let us first verify that **r** [[] _[i]_ []] sums to 1. With

                        - _δ_ [[] _[i]_ []] **1** _−_ _fϕ_ ( **x** [[] _[i]_ []] )�
**r** [[] _[i]_ []] = **ˆy** [[] _[i]_ []] _−_ _λ_ _[f][θ]_ [(] **[x]** [[] _[i]_ []][)] _[ ⊙]_ _,_

1 _−_ _δ_ [[] _[i]_ []]

we sum over classes and using _⟨fθ_ ( **x** [[] _[i]_ []] ) _,_ **1** _⟩_ = 1 yields

**1** _[⊤]_ **r** [[] _[i]_ []] = 1 _−_ _λ_ _[δ]_ [[] _[i]_ []] _[ −⟨][f][θ]_ [(] **[x]** [[] _[i]_ []][)] _[, f][ϕ]_ [(] **[x]** [[] _[i]_ []][)] _[⟩]_ = 1 _,_

1 _−_ _δ_ [[] _[i]_ []]

so **r** [[] _[i]_ []] lies on the simplex (hence Eq. (22) is an ordinary cross-entropy with a fixed target). Let
_ψ_ [[] _[i]_ []] = _ψθ_ ( **x** [[] _[i]_ []] ) be the logits and recall _[∂]_ [logsoftmax(] _∂ψ_ _[ψ]_ [)] = _I_ _−_ softmax( _ψ_ ) **1** _[⊤]_ _._ For the per-sample

loss _ℓ_ [[] _[i]_ []] = _−⟨_ sg( **r** [[] _[i]_ []] ) _,_ log softmax( _ψ_ [[] _[i]_ []] ) _⟩,_ the derivative w.r.t. logits is

_∂ℓ_ [[] _[i]_ []]

_∂ψ_ [[] _[i]_ []] [= softmax(] _[ψ]_ [[] _[i]_ []][)] _[ −]_ [sg(] **[r]** [[] _[i]_ []][) =] _[ f][θ]_ [(] **[x]** [[] _[i]_ []][)] _[ −]_ [sg(] **[r]** [[] _[i]_ []][)] _[,]_

which is Eq. (23). Applying the chain rule and averaging over _i_ gives Eq. (24). Replacing sg( **r** [[] _[i]_ []] ) by
its explicit form produces Eq. (25).


29


Substituting back, we obtain

    - _r_


_[l, r]_ [)] �ℏ( _l, y_ ) + ℏ( _r, y_ )� _,_ _Z_ ( _l, r_ ) := - _r_

2


_p_ ( _τ_ ) ℏ( _τ, y_ ) d _τ_ = _[Z]_ [(] _[l, r]_ [)]
_l_ 2


_p_ ( _τ_ ) d _τ._
_l_


Equivalently, the necessary optimality condition is ERR( _l_ _[∗]_ _, r_ _[∗]_ _, y_ ) = 0. Since _Z_ ( _l_ _[∗]_ _, r_ _[∗]_ ) _>_ 0, this is
also equivalent to


ERR( _l_ _[∗]_ _, r_ _[∗]_ _, y_ ) := E _τ_ _∼p_ ( _τ_ _|τ_ _∈_ [ _l∗,r∗_ ])[ℏ( _τ, y_ )] _−_ 12 �ℏ( _l_ _[∗]_ _, y_ ) + ℏ( _r_ _[∗]_ _, y_ )� = 0 _._


This establishes the necessary condition for an optimal subinterval.


C DISCCUSION


C.1 ANALYSIS OF EARLY-LEARNING REGULARIZATION IN EQ. (15)


The effectiveness of Eq. (15) can be better understood by examining the form of its gradient. For
clarity, we restate the loss with the following notation: given a noisy-labeled input ( **x** _,_ ˆ _y_ ), we denote
the model’s output probabilities as _fθ_ ( **x** ) and the corresponding EMA target as _fϕ_ ( **x** ).


Let **ˆy** _∈_ R _[c]_ be the one-hot vector corresponding to the noisy label _y_ ˆ. Then the loss over the whole
dataset _D_ = _{_ ( **x** [[] _[i]_ []] _,_ **ˆy** [[] _[i]_ []] ) _}_ _[n]_ _i_ =1 [can be computed according to Eq. (15) as]


_L_ [NL] Cls [(] _[D]_ [) =] _[ −]_ [1]

_n_


_n_


_i_ =1


- sg( **r** [[] _[i]_ []] ) _,_ log _fθ_ ( **x** [[] _[i]_ []] )� _,_ **r** [[] _[i]_ []] = **ˆy** [[] _[i]_ []] _−_ _λ_ _[f][θ]_ [(] **[x]** [[] _[i]_ []][)] _[ ⊙]_ - _δ_ [[] _[i]_ []] **1** _−_ _fϕ_ ( **x** [[] _[i]_ []] )� _,_ (22)

1 _−_ _δ_ [[] _[i]_ []]


where _δ_ [[] _[i]_ []] = _⟨fθ_ ( **x** [[] _[i]_ []] ) _, fϕ_ ( **x** [[] _[i]_ []] ) _⟩_, sg( _·_ ) denotes the stop-gradient operator, and _⊙_ is the Hadamard
product. By construction **r** [[] _[i]_ []] is treated as a _constant_ w.r.t. _θ_ due to the stop-gradient.
**Lemma 1.** _Let ψθ_ ( **x** ) _denote the pre-softmax logits such that fθ_ ( **x** ) = softmax( _ψθ_ ( **x** )) _._ _For the loss_
_in Eq. (15), the gradients are_
_∂L_ [NL] Cls [(] **[x]** [[] _[i]_ []][)]       - **r** [[] _[i]_ []][�] _,_ _for each i_ = 1 _, . . ., n,_ (23)
_∂ψθ_ ( **x** [[] _[i]_ []] ) [=] _[ f][θ]_ [(] **[x]** [[] _[i]_ []][)] _[ −]_ [sg]

_and, by the chain rule,_


_∇θL_ [NL] Cls [(] _[D]_ [) =] [1]

_n_


_∇θL_ [NL] Cls [(] _[D]_ [) =] [1]


_n_

- _J_ **z** _θ_ ( **x** [[] _[i]_ []] ) _[⊤]_ [�] _fθ_ ( **x** [[] _[i]_ []] ) _−_ sg� **r** [[] _[i]_ []][��] _,_ (24)


_i_ =1


_where J_ **z** _θ_ ( **x** ) = _∂_ **z** _θ_ ( **x** ) _/∂θ is the Jacobian of the logits w.r.t. the parameters._ _Equivalently, expand-_
_ing_ **r** [[] _[i]_ []] _gives_


- [�]

_._ (25)


_∇θL_ [NL] Cls [(] _[D]_ [) =] [1]

_n_


_n_ 
- _J_ **z** _θ_ ( **x** [[] _[i]_ []] ) _[⊤]_


_i_ =1


        - _fθ_ ( **x** [ _i_ ]) _⊙_        - _δ_ [[] _[i]_ []] **1** _−_ _fϕ_ ( **x** [[] _[i]_ []] )�
_fθ_ ( **x** [[] _[i]_ []] ) _−_ **ˆy** [[] _[i]_ []] + _λ_ sg

1 _−_ _δ_ [[] _[i]_ []]


        - _fθ_ ( **x** [ _i_ ]) _⊙_        - _δ_ [[] _[i]_ []] **1** _−_ _fϕ_ ( **x** [[] _[i]_ []] )�
_fθ_ ( **x** [[] _[i]_ []] ) _−_ **ˆy** [[] _[i]_ []] + _λ_ sg


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


**Remark.** Eq. (25) shows that _L_ [NL] Cls [behaves like the standard cross-entropy gradient plus an ELR-like]
corrective term. This term amplifies gradients on clean samples and counteracts gradients on noisy
samples. Specifically, we expand this ELR-like corrective term into:


where _µ_ _∈_ [0 _,_ 1] is a momentum parameter, _Si_ is the candidate label set for sample _xi_, and _fj_ ( _xi_ )
denotes the model prediction for class _j_ . This rule progressively refines **r** as the model improves,
leading to more accurate and stable class-prior estimates.


C.2.2 CLASS-PRIOR ESTIMATION IN SUPPLEMENTARY-UNLABELED DATASETS


In the case of supplementary-unlabeled datasets, which also is called semi-supervised datasets,
the estimation of class-prior is relatively straightforward. We assume that the distribution of the
labeled dataset is consistent with that of the unlabeled dataset. Therefore, the class-prior can be
directly obtained by counting the class distribution over the labeled dataset, which serves as a reliable
approximation of the overall data distribution.


C.2.3 CLASS-PRIOR ESTIMATION IN NOISY-LABEL DATASETS


We consider the widely adopted class-dependent label noise setting (Yao et al., 2020), where the
observed noisy label of each **x** _∈X_ depends only on its underlying clean label. Formally, the
transition probability from class _i_ to class _j_ is defined as


_P_ ( _Y_ [�] = _ej |_ _Y_ = _ei, X_ = **x** ) = _P_ ( _Y_ [�] = _ej |_ _Y_ = _ei_ ) = _Tij,_ _∀i, j_ _∈_ [[ _c_ ]] _,_

where **T** = [ _Tij_ ] _∈_ [0 _,_ 1] _[c][×][c]_ is the noise transition matrix. To make the estimation of **T** feasible, we
follow prior work and impose the following assumptions.


30


_fθ_ ( **x** [[] _[i]_ []] )
**g** _y_ [[] _[i]_ []] [:=]
1 _−⟨fθ_ ( **x** [[] _[i]_ []] ) _, fϕ_ ( **x** [[] _[i]_ []] ) _⟩_


_c_
�( _fϕ_ ( **x** [[] _[i]_ []] ) _k −_ _fϕ_ ( **x** [[] _[i]_ []] ) _y_ ) _fθ_ ( **x** [[] _[i]_ []] ) _k._ (26)


_k_ =1


If _y_ _[∗]_ is the true class, then the _y_ _[∗]_ th entry of _fϕ_ ( **x** [[] _[i]_ []] ) tends to be dominant during early-learning. In
that case, the _y_ _[∗]_ th entry of **g** [[] _[i]_ []] is negative. This is useful both for examples with clean labels and for
examples with noisy labels. For examples with clean labels, the cross-entropy term _fθ_ ( **x** [[] _[i]_ []] ) _−_ **ˆy** [[] _[i]_ []]

tends to vanish after the early-learning stage because _fθ_ ( **x** [[] _[i]_ []] ) is very close to **ˆy** [[] _[i]_ []], allowing examples
with wrong labels to dominant the gradient. Adding **g** [[] _[i]_ []] counteracts this effect by ensuring that
the magnitudes of the coefficients on examples with clean labels remain large. Thus, **g** [[] _[i]_ []] fulfils the
two desired properties that boosting the gradient of examples with clean labels, and neutralizing the
gradient of the examples with false labels.


C.2 CLASS-PRIOR ESTIMATION IN IMPRECISE-LABEL DATASETS


When the class priors _p_ ( _y_ ) (here we slightly abuse notation and denote them as _πy_ ) are not directly
accessible to the learning algorithm, they can be estimated using off-the-shelf estimation methods (Luo
et al., 2024; Wang et al., 2022a). In this section, we present the problem formulation and outline how
class priors can be estimated in practice.


C.2.1 CLASS-PRIOR ESTIMATION IN PARTIAL-LABEL DATASETS


In partial-label learning, each instance is associated with a candidate label set rather than a single
ground-truth label. This label ambiguity makes it difficult to estimate the class prior distribution,
since simply counting training samples per class is no longer feasible. To address this issue, we adopt
an iterative estimation strategy that updates the class prior in a moving-average manner.


We use the model’s predicted labels as a proxy for class prior estimation. Since predictions in the
early stage of training are often unreliable, we design a moving-average update rule that gradually
stabilizes the estimated distribution. The update starts from a uniform prior **r** = [1 _/c, . . .,_ 1 _/c_ ], and
is refined at each training epoch as


**r** _←_ _µ_ **r** + (1 _−_ _µ_ ) **z** _,_ **z** _j_ = [1]

_n_


_n_ - 

I _j_ = arg max _,_ (27)
_y∈Si_ _[f][j]_ [(] _[x][i]_ [)]
_i_ =1


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


**Assumption 1** (Sufficiently Scattered Assumption (Li et al., 2021b)) **.** The clean class posterior _P_ ( _Y_ _|_
_X_ ) = [ _P_ ( _Y_ = _e_ 1 _| X_ ) _, . . ., P_ ( _Y_ = _ec | X_ )] _[⊤]_ _∈_ [0 _,_ 1] _[c]_ is said to be sufficiently scattered if there
exists a set _H_ = _{_ **x** 1 _, . . .,_ **x** _m}_ such that the matrix **H** = [ _P_ ( _Y_ _| X_ = **x** 1) _, . . ., P_ ( _Y_ _| X_ = **x** _m_ )]
satisfies: (i) _Q_ _⊆_ cone _{_ **H** _}_, where _Q_ = _{_ **v** _∈_ R _[c]_ _|_ **v** _[⊤]_ **1** _≥_ _[√]_ _c −_ 1 _∥_ **v** _∥_ 2 _}_, and cone _{_ **H** _}_ denotes
the convex cone generated by the columns of **H** ; (ii) cone _{_ **H** _}_ ⊈ cone _{_ **U** _}_ for any unitary matrix
**U** _∈_ R _[c][×][c]_ that is not a permutation matrix.


**Assumption 2** (Nonsingular **T** ) **.** The transition matrix **T** is nonsingular, i.e., Rank( **T** ) = _c_ .


Assumption 1 ensures that the clean posteriors are sufficiently scattered so that the ground-truth **T**
can be identified, while Assumption 2 guarantees the invertibility of **T** .

Let _ϵ_ denote the noise rate. For symmetric label noise, we have _Tii_ = 1 _−_ _ϵ_ and _Tij_ = _c−ϵ_ 1 [with]
_j_ = _i._ In practice, the transition matrix can be estimated by solving the following optimization
problem (Li et al., 2021b):


each row maps a true label to a candidate set of labels with varying probabilities, and _q_ is set
to 0 _._ 5. For noisy-label datasets with asymmetric noise (40% flip probability), we adopt the
following mappings: _Fashion-MNIST:_ ‘Pullover’ _→_ ‘Sneaker’, ‘Dress’ _→_ ‘Bag’, ‘Sandal’ _→_ ‘Shirt’,
‘Shirt’ _→_ ‘Sandal’. _CIFAR-10:_ ‘Truck’ _→_ ‘Automobile’, ‘Bird’ _→_ ‘Airplane’, ‘Deer’ _→_ ‘Horse’,
‘Cat’ _→_ ‘Dog’, ‘Dog’ _→_ ‘Cat’. _ImageNette:_ ‘Tench’ _→_ ‘English springer’, ‘Cassette player’ _→_ ‘Garbage
truck’, ‘Chain saw’ _→_ ‘Church’, ‘Golf ball’ _→_ ‘Parachute’, ‘Parachute’ _→_ ‘Golf ball’.


**Model setup.** The overall diffusion framework follows EDM (Karras et al., 2022), and the training
hyperparameters are kept consistent with those reported therein. For all experiments, we adopt the
DDPM++ network architecture with a U-Net backbone. Specifically, we employ the Adam optimizer


31


min _L_ ( _θ,_ **T** ) = [1]
_θ,_ **T** [�] [�] _n_


_n_

- _ℓ_ �� **T** _⊤hθ_ ( **x** _i_ ) _,_ - _yi_ - + _λ ·_ log det( **T** [�] ) _,_ (28)

_i_ =1


where _ℓ_ is a loss function (typically cross-entropy), _hθ_ ( _·_ ) is the output of a neural network parameterized by _θ_, and the regularization term log det( **T** [�] ) encourages the estimated transition matrix to have
minimal simplex volume. Here _λ >_ 0 is a trade-off hyperparameter. By Assumption 1, the solution
**T** - converges to the true **T** given sufficient noisy data (Theorem 1 in (Li et al., 2021b)).

Once the transition matrix **T** is estimated, the clean class prior _π_ = [ _π_ 1 _, . . ., πc_ ] _[⊤]_ can be obtained by
solving the following system of linear equations:
 _ππ_ ��21 == _T T_ 1112 _ππ_ 11 + + _T T_ 2122 _ππ_ 22 + + _· · · · · ·_ + + _T Tcc_ 12 _ππcc_


_,_ (29)





_π_ �1 = _T_ 11 _π_ 1 + _T_ 21 _π_ 2 + _· · ·_ + _Tc_ 1 _πc_
_π_ �2 = _T_ 12 _π_ 1 + _T_ 22 _π_ 2 + _· · ·_ + _Tc_ 2 _πc_
...
_π_ - _c_ = _T_ 1 _cπ_ 1 + _T_ 2 _cπ_ 2 + _· · ·_ + _Tccπc_


where � _πi_ = _P_ ( _Y_ [�] = _ei_ ) is the noisy class prior of the _i_ -th class. The empirical estimate of � _πi_ can be
computed as


_π_ �� _i_ = _n_ [1]


_n_

- **1** _{y_ - _j_ = _ei},_ _∀i ∈_ [[ _c_ ]] _._ (30)

_j_ =1


Solving this system yields the clean class prior _π_, which is then used in subsequent modeling.


D IMPLEMENTATION DETAILS


Our implementation is based on PyTorch 1.12 (Paszke et al., 2019), and all experiments were
conducted on NVIDIA Tesla A100 GPUs with CUDA 12.4.


**Imprecise-label** **construction.** For all class-dependent partial-label datasets, we construct a








10 _×_ 10 circulant transition matrix





1 _q_ +0 _._ 2 _q_ _q_ _−_ 0 _._ 2 _· · ·_ _q_ +0 _._ 2 _q_ _q_ _−_ 0 _._ 2
_q_ _−_ 0 _._ 2 1 _q_ +0 _._ 2 _q_ _· · ·_ _q_ _q_ _−_ 0 _._ 2 _q_ +0 _._ 2
_q_ _q_ _−_ 0 _._ 2 1 _q_ +0 _._ 2 _· · ·_ _q_ _−_ 0 _._ 2 _q_ +0 _._ 2 _q_
... ... ... ... ... ... ... ...
_q_ +0 _._ 2 _q_ _q_ _−_ 0 _._ 2 1 _· · ·_ _q_ _q_ _−_ 0 _._ 2 1


, where


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


with a learning rate of 1e _−_ 3, parameters ( _β_ 1 _, β_ 2) = (0 _._ 9 _,_ 0 _._ 999), and _ϵ_ = 1e _−_ 8. The EMA decay is
set to 0 _._ 5. We use a batch size of 128 for Fashion-MNIST, 64 for CIFAR-10, and 16 for ImageNette.
For the diffusion classifier, we set the timestep interval length ∆ to 6.4. All models are trained from
scratch for 200k iterations.


E EXPERIMENTS


E.1 EVALUATION METRICS


We evaluate the trained CDMs using four unconditional metrics, including Frechet Inception Dis-´
tance (FID) (Heusel et al., 2017), Inception Score (IS) (Salimans et al., 2016), Density, and Coverage (Naeem et al., 2020), and three conditional metrics, namely CW-FID, CW-Density, CWCoverage (Chao et al., 2022). All metrics are computed using the official implementation of
DLSM (Chao et al., 2022). Although these metrics have been introduced in related work (Na
et al., 2024), we briefly recap them here for completeness and clarity.


**Unconditional metrics.** Unconditional metrics evaluate generated samples without reference to class
labels. In our experiments, images are first generated conditionally per class but then pooled without
labels when computing the metrics. This evaluation protocol is consistent with prior studies (Kaneko
et al., 2019; Chao et al., 2022).


    - FID measures the distance between real and generated image distributions in the pre-trained
feature space (Szegedy et al., 2016), indicating the fidelity and diversity of generated images.

    - IS evaluates whether generated images belong to distinct classes and whether each image is
class-consistent, reflecting the realism and class separability of generated images.

    - Density and Coverage are reliable versions of Precision and Recall (Naeem et al., 2020),
respectively. Density measures how well generated samples cover real data distribution,
while Coverage assesses how well real samples are represented by generated ones.


**Conditional** **metrics.** To measure conditional consistency, we adopt class-wise (CW) variants
of the above metrics, which compute each metric separately within each class and then average
across classes. Notably, CW-FID (also called intra-FID) is widely used in conditional generative
modeling (Miyato & Koyama, 2018; Kaneko et al., 2019), and has been highlighted as a key measure
of conditional distribution quality.


**Remark** : It should be noted that the Fashion-MNIST dataset is not suitable for evaluation using these
metrics, so we do not perform evaluation on the Fahsion-MNIST dataset.


E.2 FULL RESULTS IN WEAKLY SUPERVISED LEARNING


Building on the experiments presented in the main text, we further provide an extended comparison
with a broader set of methods to ensure a comprehensive evaluation. The details are summarized as


**Partial-label learning.** We compare against ten representative baselines: _PRODEN_ (Lv et al., 2020),
_CAVL_ (Zhang et al., 2021b), _POP_ (Xu et al., 2023), _CC_ (Feng et al., 2020), _LWS_ (Wen et al., 2021),
_IDGP_ (Qiao et al., 2023), _PiCO_ (Wang et al., 2023), _ABLE_ (Xia et al., 2022), _CRDPLL_ (Wu et al.,
2022), and _DIRK_ (Wu et al., 2024). For a fair comparison, we follow the hyperparameter settings
used in _PLENCH_ (Wang et al., 2025b). The complete results are reported in Table 5.


**Semi-supervised learning.** We follow the training and evaluation protocols of _USB_ (Wang et al.,
2022c), a widely adopted benchmark for fair and unified SSL comparisons. Our baselines cover a
broad spectrum of recent approaches. First, we include confidence-thresholding methods such as
_FixMatch_ (Sohn et al., 2020), _FlexMatch_ (Zhang et al., 2021a), _FreeMatch_ (Wang et al., 2022d),
_ReMixMatch_ (Berthelot et al., 2019a), _Dash_ (Xu et al., 2021) and _UDA_ (Xie et al., 2020). Second,
we consider contrastive-learning based and pseudo-label based methods, including _CoMatch_ (Li
et al., 2021a), _SoftMatch_ (Chen et al., 2023) and _SimMatch_ (Zheng et al., 2022). Finally, we add
several classical and widely studied SSL approaches, including _Pseudo-Labeling_ (Lee et al., 2013),
_VAT_ (Miyato et al., 2018) and _Mean Teacher_ (Tarvainen & Valpola, 2017). This diverse collection of
baselines allows us to rigorously examine whether our framework remains competitive against both
state-of-the-art and classical SSL methods under consistent experimental setups.


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


Table 5: Classification results on Fashion-MNIST, CIFAR-10, and ImageNette datasets under various
types of partial-label supervision. **Bold** numbers indicate the best performance.


Fashion-MNIST CIFAR-10 ImageNette
Method

Random Class-50% Random Class-50% Random Class-50%


_PRODEN_ 93.31 _±_ 0.07 93.44 _±_ 0.21 90.02 _±_ 0.22 90.44 _±_ 0.44 84.75 _±_ 0.13 83.50 _±_ 0.60
_CAVL_ 93.09 _±_ 0.17 92.67 _±_ 0.25 87.28 _±_ 0.64 87.16 _±_ 0.58 41.69 _±_ 4.12 46.46 _±_ 7.15
_POP_ 93.59 _±_ 0.17 93.57 _±_ 0.19 89.13 _±_ 0.22 90.19 _±_ 0.10 84.65 _±_ 0.55 84.29 _±_ 0.17
_CC_ 93.17 _±_ 0.32 92.65 _±_ 0.29 88.40 _±_ 0.24 89.12 _±_ 0.23 81.11 _±_ 0.50 80.74 _±_ 0.68
_IDGP_ 92.26 _±_ 1.25 93.07 _±_ 0.16 89.65 _±_ 0.53 90.83 _±_ 0.34 84.07 _±_ 0.26 82.18 _±_ 0.13
_PiCO_ 93.32 _±_ 0.12 93.32 _±_ 0.33 86.40 _±_ 0.89 87.51 _±_ 0.66 82.15 _±_ 0.23 84.41 _±_ 0.93
_ABLE_ 93.02 _±_ 0.26 93.20 _±_ 0.16 90.77 _±_ 0.33 90.74 _±_ 0.48 71.81 _±_ 2.46 76.53 _±_ 1.28
_CRDPLL_ 94.03 _±_ 0.14 93.80 _±_ 0.23 92.74 _±_ 0.26 92.89 _±_ 0.27 84.31 _±_ 0.25 88.08 _±_ 0.34
_DIRK_ 94.11 _±_ 0.22 93.99 _±_ 0.24 93.48 _±_ 0.14 93.22 _±_ 0.37 87.90 _±_ 0.11 87.47 _±_ 0.17
_Vanilla_ 80.20 _±_ 1.29 66.03 _±_ 1.43 60.25 _±_ 0.17 56.34 _±_ 0.50 56.04 _±_ 0.61 59.47 _±_ 0.51
_DMIS_ _[CE]_ 84.24 _±_ 0.37 78.45 _±_ 0.46 91.47 _±_ 0.15 90.52 _±_ 0.35 84.49 _±_ 0.05 82.34 _±_ 0.27
_DMIS_ **94.27** _±_ **0.55** **94.20** _±_ **0.15** **94.70** _±_ **0.49** **93.53** _±_ **0.12** **89.31** _±_ **0.21** **88.42** _±_ **0.43**


Table 6: Classification results on Fashion-MNIST, CIFAR-10, and ImageNette datasets under various
types of supplementary-unlabeled supervision. **Bold** numbers indicate the best performance.


Fashion-MNIST CIFAR-10 ImageNette
Method

Random-1% Random-10% Random-1% Random-10% Random-1% Random-10%


_Pseudo-Labeling_ 83.53 _±_ 0.46 89.59 _±_ 0.23 50.10 _±_ 0.95 72.92 _±_ 0.17 43.00 _±_ 0.82 68.03 _±_ 0.32
_Mean Teacher_ 82.34 _±_ 0.09 89.91 _±_ 0.15 47.69 _±_ 0.27 73.01 _±_ 0.78 40.53 _±_ 1.56 65.72 _±_ 0.55
_VAT_ 83.31 _±_ 0.61 89.35 _±_ 0.12 49.64 _±_ 0.90 71.07 _±_ 1.27 38.63 _±_ 8.39 63.93 _±_ 5.18
_UDA_ 84.28 _±_ 0.41 90.83 _±_ 0.34 69.20 _±_ 1.41 80.50 _±_ 0.55 50.52 _±_ 3.79 72.53 _±_ 1.17
_FixMatch_ 84.32 _±_ 0.33 90.76 _±_ 0.38 67.48 _±_ 1.42 80.00 _±_ 0.63 50.41 _±_ 4.43 71.32 _±_ 1.93
_Dash_ 84.73 _±_ 0.09 91.16 _±_ 0.20 70.14 _±_ 0.69 81.50 _±_ 0.68 57.68 _±_ 2.19 74.66 _±_ 0.81
_CoMatch_ 85.31 _±_ 0.29 90.52 _±_ 0.12 61.45 _±_ 1.46 77.79 _±_ 0.53 63.88 _±_ 0.78 73.20 _±_ 0.46
_FlexMatch_ 84.43 _±_ 0.30 90.69 _±_ 0.03 70.72 _±_ 0.93 81.35 _±_ 0.48 61.39 _±_ 0.70 73.08 _±_ 0.13
_FreeMatch_ 84.30 _±_ 0.37 90.92 _±_ 0.24 70.15 _±_ 0.44 80.99 _±_ 0.56 60.37 _±_ 1.11 73.14 _±_ 1.03
_SimMatch_ 84.69 _±_ 0.17 91.18 _±_ 0.13 73.33 _±_ 1.02 82.90 _±_ 0.43 58.12 _±_ 2.66 76.12 _±_ 0.45
_SoftMatch_ 84.72 _±_ 0.23 91.22 _±_ 0.11 73.24 _±_ 0.82 88.66 _±_ 0.60 58.50 _±_ 2.31 75.75 _±_ 0.25
_Vanilla_ 78.37 _±_ 3.72 90.50 _±_ 1.00 53.49 _±_ 0.15 85.13 _±_ 0.12 49.55 _±_ 0.99 74.70 _±_ 0.53
_DMIS_ _[CE]_ 82.92 _±_ 0.17 91.07 _±_ 0.18 75.40 _±_ 0.54 89.85 _±_ 0.08 62.64 _±_ 0.24 71.39 _±_ 0.45
_DMIS_ **85.92** _±_ **0.13** **92.97** _±_ **0.21** **76.30** _±_ **0.17** **92.47** _±_ **0.39** **68.23** _±_ **0.19** **77.30** _±_ **0.15**


Table 7: Classification results on Fashion-MNIST, CIFAR-10, and ImageNette datasets under various
types of noisy-label supervision. **Bold** numbers indicate the best performance.


Fashion-MNIST CIFAR-10 ImageNette
Method

Sym-40% Asym-40% Sym-40% Asym-40% Sym-40% Asym-40%


_CE_ 76.18 _±_ 0.26 82.01 _±_ 0.06 67.22 _±_ 0.26 76.98 _±_ 0.42 58.43 _±_ 0.77 71.81 _±_ 0.38
_Co-learning_ 90.85 _±_ 0.63 84.10 _±_ 2.01 84.97 _±_ 0.53 80.36 _±_ 1.09 76.16 _±_ 0.96 75.37 _±_ 0.49
_Co-teaching_ 92.17 _±_ 0.34 92.78 _±_ 0.25 86.54 _±_ 0.57 79.38 _±_ 0.39 66.55 _±_ 1.00 75.12 _±_ 0.50
_Co-teaching+_ 91.05 _±_ 0.06 91.62 _±_ 0.20 67.28 _±_ 1.85 79.43 _±_ 0.47 75.79 _±_ 0.79 75.17 _±_ 0.40
_SCE_ 93.62 _±_ 0.22 88.60 _±_ 0.20 82.82 _±_ 0.40 81.54 _±_ 0.64 77.99 _±_ 0.39 74.81 _±_ 1.04
_GCE_ 93.64 _±_ 0.03 87.48 _±_ 0.09 85.00 _±_ 0.27 77.97 _±_ 3.69 81.18 _±_ 0.35 72.61 _±_ 1.14
_Decoupling_ 92.24 _±_ 0.23 92.10 _±_ 0.44 82.24 _±_ 0.28 79.89 _±_ 0.58 75.53 _±_ 0.69 78.24 _±_ 0.21
_ELR_ 93.13 _±_ 0.13 92.82 _±_ 0.09 85.68 _±_ 0.13 81.32 _±_ 0.31 84.03 _±_ 2.86 73.51 _±_ 0.31
_JoCoR_ 84.05 _±_ 1.11 89.45 _±_ 4.43 77.92 _±_ 3.92 78.68 _±_ 0.07 67.82 _±_ 1.97 74.67 _±_ 0.43
_Mixup_ 92.21 _±_ 0.03 92.01 _±_ 1.02 84.26 _±_ 0.64 83.21 _±_ 0.85 76.65 _±_ 1.62 77.16 _±_ 0.71
_PENCIL_ 90.85 _±_ 0.58 91.77 _±_ 0.69 85.91 _±_ 0.26 84.89 _±_ 1.49 81.94 _±_ 1.26 77.20 _±_ 1.15
_Vanilla_ 90.11 _±_ 1.24 85.41 _±_ 0.96 80.22 _±_ 0.10 86.31 _±_ 0.10 55.86 _±_ 1.95 53.91 _±_ 1.07
_DMIS_ _[CE]_ 82.76 _±_ 0.57 83.39 _±_ 0.24 84.75 _±_ 0.36 84.21 _±_ 0.18 80.47 _±_ 0.56 77.21 _±_ 0.19
_DMIS_ **93.40** _±_ **0.40** **93.20** _±_ **0.30** **88.63** _±_ **0.12** **88.83** _±_ **0.33** **84.12** _±_ **0.18** **79.30** _±_ **0.27**


**Noisy-label** **learning.** We further benchmark our method against nine widely used approaches:
_Coteaching_ (Han et al., 2018), _Coteaching+_ (Yu et al., 2019), _SCE_ (Wang et al., 2019), _GCE_ (Zhang
& Sabuncu, 2018), _Decoupling_ (Malach & Shalev-Shwartz, 2017), _ELR_ (Liu et al., 2020), and
_JoCoR_ (Wei et al., 2020). These methods cover a range of strategies, from sample selection and
reweighting to robust loss design, thus providing a diverse and rigorous benchmark.


Across all three weakly supervised scenarios, our method consistently achieves the best performance
compared to existing baselines, reinforcing both its robustness and versatility under different forms
of imprecise supervision.


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


E.3 INTEGRATION WITH EXISTING IMPRECISE-LABEL CORRECTORS


Existing weakly supervised learning methods often rely on pseudo-labeling strategies that aim to
correct imprecise labels by assigning refined labels to training samples. From this perspective, our
approach is orthogonal to such methods: while pseudo-labeling seeks to approximate the true labels
as closely as possible, our framework focuses on robustly learning from the remaining label noise. In
practice, pseudo-labeling methods inevitably produce imperfect corrections. While most samples
may be relabeled correctly, a non-negligible portion of instances still receive erroneous pseudo-labels
because no classifier is perfect. As a result, imprecise supervision is effectively transformed into a
noisy-label supervision.


This naturally complements our framework: by combining a pseudo-label corrector with _DMIS_, one
can first reduce label uncertainty through correction and then leverage the robustness of diffusion
models to learn from the residual noise. To validate this premise, we conduct a case study where a
noisy-label learning method trained on CIFAR-10 with 40% symmetric noise achieves a pseudo-label
accuracy of 80% on the training set. Using this pseudo-labeled dataset as input, our _DMIS_ framework
further improves the classification performance. As illustrated in Figure 6, integrating pseudo-label
correction with _DMIS_ consistently improves the performance across all datasets. Thus, we believe
that our framework addresses the challenge of imprecise labels through the lens of diffusion model,
offering a complementary perspective to conventional noisy-label approaches.


E.5 VISUALIZATION OF NOISY CONDENSED DATASETS


We visualize the condensed images on CIFAR-10 and Fashion-MNIST in Figure 8 and Figure 9,
respectively. It is evident that datasets generated by our method exhibit both higher diversity and
stronger realism compared to other approaches. In particular, for the condensed Fashion-MNIST
images, methods such as _DC_ and _DM_ often produce samples that do not faithfully correspond to
their assigned class, resulting in condensed datasets that still contain noisy labels and thus degrade
performance. By contrast, our proposed _DMIS_ generates class-consistent and visually recognizable
samples across categories, yielding condensed datasets that better preserve label fidelity and semantic


34


100


90


80


70


100


90


80


70


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


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
|||||~~Vanill~~<br>|
|||||~~DMIS~~|


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
|||||~~Vanilla~~<br>|
|||||~~DMIS~~|


Figure 7: Test accuracy curves of the _Vanilla_ and _DMIS_ models on CIFAR-10 under different forms
of imprecise supervision, including noisy-label learning (NLL), partial-label learning (PLL), and
semi-supervised learning (SSL).


alignment. These visualizations further support the quantitative results, highlighting the advantage of
generative condensation under noisy supervision.


E.6 ADDITIONAL RESULTS ON DATASET CONDENSATION UNDER DIFFERENT FORMS OF
IMPRECISE SUPERVISION


To illustrate the extreme case of noisy dataset condensation, we report the results when the IPC is set to
1. As shown in Table 8, _DMIS_ consistently achieves the best performance across all datasets and noise
types, even under the extreme case of IPC = 1. Notably, while most existing condensation methods
collapse under severe supervision noise, our method maintains a clear advantage, outperforming the
strongest baselines by a large margin. These results further demonstrate the robustness of _DMIS_ in
distilling informative representations despite highly limited and imprecisely labeled data.


Table 8: Classification results (test accuracy, %) on noisy-label Fashion-MNIST, CIFAR-10, and
ImageNette datasets. ‘IPC’ indicates the number of images per class in the condensed dataset. **Bold**
numbers indicate the best performance.


Dataset Type IPC _DC_ _DSA_ _DM_ _MTT_ _RDED_ _SRE2L_ _DMIS_


Sym-40% 1 15.21 _±_ 0.75 19.55 _±_ 0.58 15.56 _±_ 0.20 10.86 _±_ 1.90 18.07 _±_ 3.33 14.33 _±_ 1.20 **33.18** _±_ **2.15**
F-MNIST

Asym-40% 1 20.17 _±_ 0.29 17.61 _±_ 0.89 23.91 _±_ 0.36 7.39 _±_ 0.84 13.20 _±_ 0.83 13.13 _±_ 0.21 **25.78** _±_ **0.70**


Sym-40% 1 8.99 _±_ 1.59 10.00 _±_ 0.00 **14.41** _±_ **1.03** 9.99 _±_ 0.00 11.20 _±_ 0.41 11.06 _±_ 0.83 11.81 _±_ 1.04
CIFAR-10

Asym-40% 1 11.88 _±_ 1.55 10.00 _±_ 0.00 10.00 _±_ 0.00 9.96 _±_ 0.05 13.96 _±_ 1.38 15.49 _±_ 0.34 **15.88** _±_ **0.57**


Sym-40% 1 9.87 _±_ 0.00 9.87 _±_ 0.00 9.87 _±_ 0.00 19.17 _±_ 2.35 12.98 _±_ 1.16 11.90 _±_ 0.78 **19.32** _±_ **0.84**
ImageNette

Asym-40% 1 9.87 _±_ 0.00 9.87 _±_ 0.00 9.87 _±_ 0.00 17.36 _±_ 0.10 12.98 _±_ 0.42 18.55 _±_ 2.19 **21.13** _±_ **0.95**


Furthermore, even when samples are imprecisely annotated with candidate labels, our method is still
able to perform effective condensation on partial-label datasets. In contrast, most existing dataset
condensation methods rely on the assumption of having single labels for each instance and therefore
fail under this type of supervision. The only exception lies in decoupled condensation approaches
such as _RDED_ and _SRE2L_, where a teacher model can still be trained on partial-label data. We
present the corresponding results under partial-label supervision below in Table 9.


35


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(b) SSL, Random-1%.


90

80

70

60

50

40

30

20

10

0


90

80

70

60

50

40

30

20

10

0


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(a) PLL, Class-5%.


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(c) PLL, Random.


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(c) SSL, Random-10%.


90

80

70

60

50

40

30

20

10

0


90

80

70

60

50

40

30

20

10

0


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(a) NLL, Sym-40%.


0k 25k 50k 75k 100k 125k 150k 175k 200k
Training Steps


(b) NLL, Sym-40%.


90

80

70

60

50

40

30

20

10

0


90

80

70

60

50

40

30

20

10

0


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


As shown in the table, our method consistently achieves substantial improvements across both
Random and Class-50% candidate set generation strategies, and under all IPC settings. In particular,
on Fashion-MNIST, our approach yields dramatic performance gains, reaching above **87%** accuracy
even with partial label supervision, whereas both _RDED_ and _SRE2L_ fail to exceed 16% under the
same setting. On the more challenging CIFAR-10 benchmark, our method also demonstrates strong
robustness, especially under larger IPCs where the gap over baseline methods becomes increasingly
pronounced (e.g., over **20%** absolute improvement at IPC = 100). These results highlight that our
condensation strategy can effectively leverage weak supervision and generate compact yet highly
informative synthetic datasets, even when label noise is introduced by the partial-label scenario.


Table 9: Classification results (test accuracy, %) on partial-label Fashion-MNIST and CIFAR-10
datasets under different IPCs. ‘Random’ and ‘Class-50%’ denote two candidate set generation
strategies. **Bold** numbers indicate the best performance.


|Random Class-50%<br>Dataset<br>IPC RDED SRe2L Ours IPC RDED SRe2L Ours|Random|Col3|Class-50%|Col5|
|---|---|---|---|---|
|Dataset<br>**Random**<br>**Class-50%**<br>IPC<br>_RDED_<br>_SRe2L_<br>_Ours_<br>IPC<br>_RDED_<br>_SRe2L_<br>_Ours_|IPC|_RDED_<br>_SRe2L_<br>_Ours_|IPC|_RDED_<br>_SRe2L_<br>_Ours_|


Figure 8: Visualization of condensed CIFAR-10 images generated by _DC_, _DM_, _SRE2L_, and our
method _**DMIS**_ .


36


F-MNIST


CIFAR-10


**Airplane**


**Automobile**


**Bird**


**Cat**


**Deer**


**Dog**


**Frog**


**Horse**


**Ship**


**Truck**


**Airplane**


**Automobile**


**Bird**


**Cat**


**Deer**


**Dog**


**Frog**


**Horse**


**Ship**


**Truck**


1 10.48 _±_ 0.82 9.72 _±_ 1.02 **44.06** _±_ **1.64** 1 10.73 _±_ 0.78 10.93 _±_ 0.76 **33.99** _±_ **2.48**
10 13.17 _±_ 4.66 8.80 _±_ 0.70 **72.02** _±_ **0.77** 10 14.58 _±_ 1.19 10.83 _±_ 0.35 **70.46** _±_ **0.26**
50 13.06 _±_ 2.36 10.30 _±_ 0.40 **83.98** _±_ **0.12** 50 15.55 _±_ 0.55 11.33 _±_ 0.76 **79.67** _±_ **0.13**
100 13.39 _±_ 2.40 9.34 _±_ 1.77 **87.30** _±_ **0.31** 100 11.90 _±_ 2.67 11.05 _±_ 0.77 **81.42** _±_ **0.25**


1 15.32 _±_ 1.72 **20.69** _±_ **0.88** 16.31 _±_ 1.54 1 14.52 _±_ 1.06 15.55 _±_ 0.91 **16.61** _±_ **1.14**
10 26.28 _±_ 0.29 19.45 _±_ 1.12 **30.50** _±_ **0.27** 10 10.00 _±_ 0.00 18.56 _±_ 0.05 **25.00** _±_ **0.70**
50 34.96 _±_ 0.92 20.56 _±_ 0.71 **44.39** _±_ **0.65** 50 25.59 _±_ 1.32 19.39 _±_ 1.09 **45.94** _±_ **1.27**
100 28.88 _±_ 2.56 19.65 _±_ 1.09 **58.46** _±_ **0.64** 100 25.81 _±_ 1.64 18.65 _±_ 1.51 **58.06** _±_ **0.32**


(a) DC (b) DM


(c) SRE2L (d) **DMIS**


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


**T-shirt**


**Trouser**


**Pullover**


**Dress**


**Coat**


**Sandal**


**Shirt**


**Sneaker**


**Bag**


**Ankle boot**

(a) DC (b) DM


**T-shirt**


**Trouser**


**Pullover**


**Dress**


**Coat**


**Sandal**


**Shirt**


**Sneaker**


**Bag**


**Ankle boot**

(c) SRE2L (d) **DMIS**


Figure 9: Visualization of condensed Fashion-MNIST images generated by _DC_, _DM_, _SRE2L_, and
our method _**DMIS**_ .


F ADDITIONAL EXPERIMENTS RESULTS


F.1 THE FULL RSULTS OF TABLE 1.


Table 10: Generative results on CIFAR-10 and ImageNette under various settings. ‘uncond’ and
‘cond’ indicate unconditional and conditional metrics. **Bold** numbers indicate better performance.


Metric Noisy-label supervision Partial-label supervision Suppl-unlabeled supervision Clean
Sym-40% Asym-40% Random Class-50% Random-1% Random-10%


_Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_ _DMIS_ _Vanilla_

|CIFAR-10 uncond cond|FID (↓) 3.33±0.06 3.47±0.11 3.23±0.07 3.10±0.11 7.76±0.25 2.26±0.08 11.75±0.42 2.77±0.09 3.16±0.07 3.12±0.05 2.93±0.11 2.89±0.08 2.05±0.05<br>IS (↑) 9.56±0.08 9.68±0.05 9.02±0.12 9.73±0.06 9.09±0.15 9.80±0.04 9.62±0.11 9.68±0.07 10.03±0.09 10.57±0.06 9.80±0.08 9.83±0.05 10.61±0.04<br>Density (↑) 101.39±0.85 109.75±0.62 100.06±0.91 109.69±0.55 103.21±1.12 106.49±0.48 108.76±0.75 109.06±0.52 97.19±1.05 108.18±0.66 99.96±0.88 108.87±0.59 112.59±0.45<br>Coverage (↑) 81.12±0.35 81.21±0.28 80.71±0.41 81.30±0.32 68.45±0.65 82.69±0.25 64.90±0.72 81.52±0.30 78.44±0.45 81.00±0.29 81.85±0.38 82.00±0.26 83.27±0.22<br>CW-FID (↓) 29.84±1.15 13.85±0.45 14.70±0.52 13.24±0.38 27.18±0.95 10.65±0.35 32.44±1.22 11.56±0.41 16.25±0.65 16.12±0.55 11.84±0.48 11.77±0.42 9.83±0.35<br>CW-Density (↑) 72.98±0.82 107.23±0.65 90.85±0.75 107.07±0.58 102.04±0.92 105.75±0.61 102.43±0.88 108.66±0.55 89.99±0.78 100.73±0.68 96.29±0.72 107.94±0.62 111.70±0.52<br>CW-Coverage (↑) 73.39±0.45 80.11±0.35 79.63±0.42 79.65±0.38 65.45±0.68 82.09±0.32 61.45±0.75 81.24±0.36 75.03±0.52 76.84±0.41 80.80±0.45 81.12±0.39 83.91±0.30|
|---|---|
|ImageNette<br>uncond<br>cond|FID<br>(_↓_)<br>14.11_±_0.55<br>**13.44**_±_0.48<br>13.93_±_0.52<br>**13.91**_±_0.45<br>79.13_±_2.15<br>**72.62**_±_1.85<br>91.28_±_2.45<br>**79.12**_±_1.95<br>23.88_±_0.85<br>**19.26**_±_0.72<br>14.32_±_0.58<br>**12.84**_±_0.46<br>11.52_±_0.42<br>IS<br>(_↑_)<br>12.69_±_0.15<br>**13.21**_±_0.12<br>12.51_±_0.14<br>**13.73**_±_0.11<br>9.19_±_0.25<br>**9.40**_±_0.18<br>**9.27**_±_0.22<br>9.11_±_0.24<br>12.23_±_0.16<br>**13.72**_±_0.13<br>12.80_±_0.15<br>**13.16**_±_0.12<br>13.81_±_0.10<br>Density<br>(_↑_)<br>109.31_±_0.95<br>**112.52**_±_0.82<br>**111.66**_±_0.88<br>106.78_±_0.92<br>95.33_±_1.25<br>**99.83**_±_0.75<br>94.29_±_1.32<br>**102.58**_±_0.85<br>115.94_±_1.15<br>**125.68**_±_0.95<br>105.27_±_0.85<br>**109.23**_±_0.78<br>117.23_±_0.65<br>Coverage<br>(_↑_)<br>76.62_±_0.42<br>**76.81**_±_0.38<br>78.32_±_0.45<br>**79.81**_±_0.35<br>21.44_±_0.85<br>**32.48**_±_0.55<br>16.69_±_0.92<br>**22.30**_±_0.65<br>53.53_±_0.72<br>**55.39**_±_0.58<br>73.79_±_0.48<br>**75.55**_±_0.42<br>80.12_±_0.35<br>CW-FID<br>(_↓_)<br>80.31_±_2.55<br>**60.12**_±_1.85<br>62.26_±_1.95<br>**58.20**_±_1.65<br>157.76_±_3.55<br>**63.58**_±_2.15<br>163.45_±_3.85<br>**67.92**_±_2.25<br>71.66_±_2.35<br>**70.27**_±_2.10<br>49.22_±_1.55<br>**44.31**_±_1.45<br>40.20_±_1.25<br>CW-Density<br>(_↑_)<br>73.99_±_0.85<br>**81.12**_±_0.65<br>93.53_±_0.95<br>**94.58**_±_0.72<br>93.38_±_0.98<br>**95.83**_±_0.68<br>91.50_±_1.05<br>**95.21**_±_0.75<br>115.90_±_1.15<br>**118.69**_±_0.85<br>103.41_±_0.92<br>**115.67**_±_0.78<br>120.35_±_0.65<br>CW-Coverage<br>(_↑_)<br>67.89_±_0.55<br>**71.94**_±_0.45<br>74.18_±_0.52<br>**75.82**_±_0.48<br>19.76_±_0.85<br>**24.35**_±_0.55<br>15.88_±_0.92<br>**18.93**_±_0.62<br>51.73_±_0.75<br>**52.15**_±_0.65<br>72.61_±_0.58<br>**74.85**_±_0.52<br>78.48_±_0.45|


F.2 COMPARISON AGAINST OTHER NOISE-ROBUST DIFFUSION METHODS.


We have also shown comparisons against noise-robust diffusion methods (Na et al., 2024; Dufour
et al., 2024) that are designed to handle noisy-label data. For image generation task, we compare them
and our DMIS model on CIFAR-10 with 40% symmetric and asymmetric label noise, reporting FID,
IS under the same architecture and training budget. For noisy-label learning task, we compare them
and our DMIS as data generators for downstream classification. We use each model to synthesize
the same number of labeled samples and then train a Wide-ResNet-40-10 classifier on top of these


37


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


Sym-40% Asym-40%
Metric

FID IS Accuracy FID IS Accuracy


DMIS (Ours) **3.47** **9.80** **88.63** **3.10** 9.73 **88.83**
CAD (Dufour et al., 2024) 4.10 9.68 81.75 3.87 9.16 82.33
TDSM (Na et al., 2024) 3.85 9.40 66.40 3.96 **10.12** 72.32


Table 11: Results under 40% symmetric and asymmetric noise.


synthetic datasets. Importantly, both two compared methods assume access to additional prior
information, which can give them an advantage in this setting. Despite this, our method still achieves
the best overall performance under the same backbone and training budget. This suggests that our
approach is competitive while relying on strictly weaker assumptions about the available supervision.


F.3 TOP- _k_ TRUNCATION FOR LARGE LABEL SPACES


When extending to datasets with many classes, a straightforward implementation becomes expensive
because it requires estimating and backpropagating a per-class objective at every step. To reduce this
cost in practice, we restrict gradients to classes that carry non-negligible probability mass.


Concretely, we apply a top- _k_ strategy to both the diffusion posterior and the pseudo-label distribution:
only the _k_ largest entries are retained, while all remaining entries are zero-masked and do not
contribute to the gradient. In this way, the effective complexity scales with the number of active
classes _k_ per example, rather than with the total number of classes.


To assess the impact of this approximation, we conduct an experiment on the Caltech-15 dataset (Pan
et al., 2023) with 40% symmetric label noise and set _k_ = 10. As shown below, the top- _k_ variant
achieves performance comparable to the full model while reducing computational cost, indicating
that this strategy is a practical mechanism for scaling our method to larger label spaces.


Generation Metric Classification Metric


FID IS Density Coverage Test accuracy (%)


DMIS (Ours) 4.25 12.39 103.83 96.20 78.92


Table 12: Performance of DMIS under Caltech-15 dataset with 40% symmetric noise.


F.4 EXPERIMENTS BEYOND SYNTHETIC CLASS-CONDITIONAL NOISE


We primarily use synthetic noisy labels to obtain a controlled setting that supports our theory, where
both the noise rate and the noise type (e.g., symmetric, class-dependent) can be precisely specified.


Starting from such controlled synthetic-noise regimes is a necessary first step to validate both the
theoretical predictions and the basic empirical behavior of our method. To further demonstrate its
practicality under more realistic supervision, we also evaluate DMIS on real noisy-label and partiallabel benchmarks. Specifically, we report results on the real noisy-label dataset CIFAR-10N (Wei
et al., 2022) and the real partial-label dataset PLCIFAR-10 (Wang et al., 2025b), whose labels are
provided by human annotators.


In addition, we consider instance-dependent label noise on CIFAR-10, following standard instancedependent noise protocols in the noisy-label literature. The results below show that DMIS maintains
strong generative quality and competitive classification accuracy under these more complex and
realistic noise conditions.


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


Dataset FID IS Test accuracy (%)


CIFAR10-N 3.22 9.66 93.21
Instance-dependent CIFAR10 4.85 9.21 81.32
PLCIFAR10 2.95 9.82 93.65


Table 13: Performance of DMIS under more complex imprecise supervision datasets.


39