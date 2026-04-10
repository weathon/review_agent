# wd1 : WEIGHTED POLICY OPTIMIZATION FOR REASONING IN DIFFUSION LANGUAGE MODELS


**Xiaohang Tang** [1] _[,]_ [2] _[∗]_ **Rares Dolga** [2] _[,]_ [5] _[∗]_ **Sangwoong Yoon** [3] _[†]_ **Ilija Bogunovic** [4] _[,]_ [2] _[†]_


1Department of Statistical Science, University College London, United Kingdom
2UCL AI Centre, University College London, United Kingdom
3Graduate School of AI, Ulsan National Institute of Science and Technology, South Korea
4Department of Mathematics and Computer Science, Universität Basel, Switzerland
5UIPath


ABSTRACT


Improving the reasoning capabilities of diffusion-based large language models
(dLLMs) through reinforcement learning (RL) remains an open problem. The
intractability of dLLMs likelihood function necessitates approximating the current,
old, and reference policy likelihoods at each policy optimization step. This reliance
introduces additional computational overhead, and can lead to large variance
and estimation error in RL objective – particularly in computing the policy ratio
for importance sampling. To mitigate these issues, we introduce _wd1_, a novel
ratio-free policy optimization approach that reformulates the RL objective as a
weighted log-likelihood, requiring only a single approximation for the current
parametrized policy likelihood. We formally show that our proposed method can
be interpreted as energy-guided discrete diffusion training combined with negative
sample unlearning, thereby confirming its theoretical soundness. In experiments on
LLaDA-8B model, _wd1_ outperforms diffusion-based GRPO ( _d1_ ) while requiring
lower computational cost, achieving up to a +59% improvement in accuracy.
Furthermore, we extend _wd1_ to denoising-stepwise weighted policy optimization
( _wd1_ ++), achieving state-of-the-art math performance of 44 _._ 2% on MATH500 and
84 _._ 5% on GSM8K with only 20 RL training steps.


1 INTRODUCTION


Diffusion-based large language models (dLLMs) have recently gained attention as promising
alternatives to autoregressive (AR) models for language modelling tasks (Nie et al., 2025b; Ou
et al., 2025; Yang et al., 2025). Unlike AR models, which generate tokens sequentially, dLLMs
iteratively refine entire response sequences through a denoising process. A primary advantage of
this approach is the significantly improved inference efficiency. Notably, recent closed models such
as Mercury (Labs et al., 2025) and Gemini Diffusion achieve over an order of magnitude speed-up in
generation compared to AR models, while maintaining comparable generation quality. Furthermore,
open-weight dLLMs demonstrate competitive performance on standard language benchmarks, with
smaller models (Lou et al., 2024; Ou et al., 2025; Nie et al., 2024) achieving parity with equivalently
sized AR baselines, and larger-scale models such as LLaDA-8B (Zhu et al., 2025a) and Dream-7B
(Ye et al., 2025) extending this trend at scale. While dLLMs demonstrate strong performance in
text generation, it remains an open and important question how best to fine-tune dLLMs using RL

- a paradigm that has proven highly effective in alignment and improving reasoning capabilities of
AR models (Ouyang et al., 2022; Shao et al., 2024).


A key challenge in applying reinforcement learning (RL) to dLLMs is the intractability of their
likelihood functions (Zhao et al., 2025; Yang et al., 2025), which necessitates approximation for
policy optimization. Applying approximated log-likelihood for diffusion-based GRPO (Shao et al.,
2024; Zhao et al., 2025) can exponentially amplify the approximation error and lead to large variance


_∗_ Equal contribution. Code: [https://github.com/xiaohangt/wd1](https://github.com/xiaohangt/wd1)

_†_ Corresponding authors (ilija.bogunovic@unibas.ch, swyoon@unist.ac.kr).


1


when computing the policy ratio for importance sampling. Moreover, GRPO requires the estimated
likelihoods of the current, old, and reference policies at every training step, leading to significant
computational overhead. These issues can be further exacerbated as the completion length and the
number of diffusion steps increase.


To address these challenges, we propose _wd1_, a policy optimization approach with **w** eighted loglikelihood objective for **d** LLMs. Crucially, this objective dispenses with explicit policy ratios and
relies on a single likelihood approximation, thereby avoiding the potentially large bias and variance
in policy ratio, and reducing the computational overhead. Our principal contributions are:


- We propose a novel reinforcement learning method for dLLMs, termed _wd1_, which formulates the
objective as a weighted log-likelihood of outcome sequence, derived from reverse-KL regularized
policy optimization. The weight, defined as ( _−w_ [+] + _w_ _[−]_ ) and dependent on the advantage _A_,
balances two terms: _w_ [+] _∝_ exp( _A_ ) increases the probability of higher-advantage completions,
while _w_ _[−]_ _∝_ exp( _−A_ ) decreases the probability of lower-advantage ones. Together, this mechanism
amplifies beneficial outcomes meanwhile actively reducing detrimental ones.


- We prove that our proposed RL method for dLLMs can be interpreted as jointly training an energyguided discrete diffusion model—guided by the advantage function—and unlearning low-advantage
data, thereby steering generation toward higher-advantage completions.


- We conduct experiment with LLaDA-8B-Instruct model (Nie et al., 2025a). Compared to the
baseline method _d1_ (Zhao et al., 2025), our method _wd1_ achieves **76.4% on Sudoku (Arel, 2025)**
**(+58.8% over** _**d1**_ **)** and **51.2% on Countdown (Pan et al., 2025) (+16% over** _**d1**_ **)**, without requiring
supervised fine-tuning (SFT), and with significantly less computational burden in RL training.


- We further extend our method to leverage intermediate completions generated in the decoding
process, which we call _wd1_ ++. The extended method surpasses several concurrent RL for dLLMs
methods, achieving state-of-the-art performance **44.2% on MATH500** and **84.5% on GSM8K**
with only 20 training steps, and 10 _×_ fewer rollouts compared to the baseline methods.


2 PRELIMINARIES


We denote the generation policy of diffusion-based Large Language Models (dLLMs) by _πθ_ . Denote
prompt _q_ _∈D_, and completions _o ∈O_ . Notably, the reward function denoted by _R_ ( _q, o_ ) in this work
is not limited to verifiers. We use superscript _k_ to indicate the _k_ -th token of completion: _o_ _[k]_ or _x_ _[k]_ 0 [.]


2.1 DIFFUSION LARGE LANGUAGE MODELS


The most promising discrete diffusion for language modeling is masked diffusion, which gradually
corrupts the text sequence with a mask token (Sahoo et al., 2024; Ou et al., 2025; Shi et al., 2025;
Lou et al., 2024). Let _t_ _∈_ [0 _,_ 1] denote the diffusion timestep, and _xt_ as the masked sequence at
step _t_ . The fully denoised sequence (i.e., the completion _o_ ) is represented by _x_ 0, and the forward
process ( _pt|_ 0( _xt_ _| x_ 0)) is formulated as a continuous-time Markov chain. The transition kernel **Q** _t_
is absorbing (Austin et al., 2023; Campbell et al., 2022), meaning that at time _t_, **Q** _t_ = _σ_ ( _t_ ) **Q** [absorb],
where _σ_ is a decreasing scalar noise schedule and **Q** [absorb] is a constant matrix (See Definition 2).


This work aims to apply reinforcement learning to fine-tune masked discrete diffusion models such
as LLaDA (Ou et al., 2025; Zhu et al., 2025a), which models the clean data distribution conditional
on masked sequence as _πθ_ ( _x_ _[k]_ 0 _[|][ x][t]_ [)][.] [The negative Evidence Lower Bound (ELBO) is reduced to a]
simple objective called Denoising Cross Entropy (DCE) (Ou et al., 2025): _∀x_ 0 _∼_ _p_ data,


- **1** ( _x_ _[k]_ _t_ [= [][mask][]) log] _[ π][θ]_ [(] _[x]_ 0 _[k]_ _[|][ x][t]_ [)]

_k_ =1


_,_ (1)


_L_ DCE( _x_ 0) = _−_ E _t∼U_ [0 _,_ 1] _,_ _xt∼pt|_ 0( _xt|x_ 0)


1


_t_


_K_


where _K_ is the length of the sequence and _x_ _[k]_ 0 [is the] _[ k]_ [-th token of] _[ x]_ [0][.] [Specifically, intermediate steps]
_t_ are sampled from uniform distribution, and masked sequence is sampled following the predefined
forward process _pt|_ 0( _xt_ _| x_ 0). DCE can be used to approximate the marginal likelihood log _πθ_ ( _x_ 0)
for supervised fine-tuning and reinforcement learning (Nie et al., 2025a; Yang et al., 2025).


2


2.2 EXISTING POLICY OPTIMIZATION METHODS


The base method of current prevailing RL fine-tuning algorithms is Trust Region Policy Optimization
(TRPO) (Schulman et al., 2015), in which _forward_ KL divergence is applied to restrict the update:


                 -                 max _θ_ E _q∼D,_ _o∼πθ_ ( _·|q_ ) _A_ _[π]_ [old] ( _q, o_ ) _−_ _λD_ KL( _π_ old( _·|q_ ) _∥_ _πθ_ ( _·|q_ ) ) _,_ (2)


where _A_ _[π]_ [old] is the advantage function, _q_ and _o_ are denoted as the prompt and (clean) response,
respectively. Proposition 1 (Appendix A) demonstrates the monotonic policy improvement of TRPO.


PPO then extends the soft constraint (KL penalty) to clipping policy ratio _πθ_ ( _·|q_ ) _/π_ old( _·|q_ ) and
employing pessimism for policy update, further employed in fine-tuning (Ouyang et al., 2022)
with additional reverse-KL regularization w.r.t. the reference policy _π_ ref. Group Relative Policy
Optimization (GRPO) (Shao et al., 2024) simplifies PPO by sampling a group of completions _{oi}_ _[G]_ _i_ =1
and approximating their advantage with their normalized rewards. This advantage is corrected by
subtracting the mean reward across the group (Liu et al., 2025): _A_ [ˆ] _i_ = _R_ ( _q, oi_ ) _−_ mean� _R_ ( _q, o_ 1: _G_ )� _,_
which we refer to as the _group-relative advantage_ .


2.3 POLICY OPTIMIZATION FOR DLLMS


Adapting GRPO to diffusion-based large language models (dLLMs) presents notable challenges,
since dLLMs generate outputs via a non-autoregressive, iterative denoising process, making the
computation of log _πθ_ ( _o|q_ ) intractable and necessitating approximation for policy optimization.


Existing works by Nie et al. (2025a); Yang et al. (2025) employ ELBO for per-token log-likelihood
approximation following DCE: _ϕ_ _[π]_ ( _x_ _[k]_ 0 [) =][ E] _t∈U_ [0 _,_ 1] [[] _[w][ ·]_ **[ 1]** [[] _[x][k]_ _t_ [=] `[ mask]` [] log] _[ π]_ [(] _[x]_ 0 _[k][|][x][t][, q]_ [)]][, where] _[ w]_ [=]
1 _/t_ in DCE and _w_ = 1 in UniGRPO (Yang et al., 2025). However, an accurate estimation requires
a large sample size of _t_, resulting in inefficiency for online RL. A biased but efficient method is
introduced in _d1_ (Zhao et al., 2025), requiring only sample at _t_ = 1: _ϕ_ _[π]_ ( _x_ _[k]_ 0 [)] [=] [log] _[ π]_ [(] _[x][k]_ 0 _[|][x]_ [1] _[, q][′]_ [)][,]
where prompt _q_ _[′]_ is randomly masked, _x_ 1 is fully masked response.


In diffusion-based GRPO (Zhao et al., 2025; Yang et al., 2025), policy ratio is then computed
using the approximated log-likelihoods: _ri_ _[k]_ [(] _[θ]_ [) =] _[ π][θ]_ [(] _[o]_ _i_ _[k]_ [)] _[/π]_ [old][(] _[o]_ _i_ _[k]_ [)] _[ ≈]_ [exp] - _ϕ_ _[π][θ]_ ( _o_ _[k]_ _i_ [)] _[ −]_ _[ϕ][π]_ [old] [(] _[o]_ _i_ _[k]_ [)] - for
importance sampling in estimating the objective of GRPO:


_K_

- min - _ri_ _[k]_ [(] _[θ]_ [) ˆ] _[A][i][,]_ [ clip][(] _[r]_ _i_ _[k]_ [(] _[θ]_ [)] _[,]_ [ 1] _[ ±][ ϵ]_ [) ˆ] _[A][i]_ - _−_ _βD_ KL� _πθ_ ( _·_ ) _∥_ _π_ ref( _·_ )� [�] _._ (3)

_k_ =1


_G_


_i_ =1


E _q∼D,_

_o_ 1: _G∼π_ old( _·|q_ )


- 1

_GK_


However, existing approaches are hampered by their reliance on extensive likelihood approximation to compute
the policy ratio. In current diffusion-based GRPO methods,
the ratio is computed as _ri_ _[k]_ _[≈]_ [exp] - _ϕ_ _[π][θ]_ ( _o_ _[k]_ _i_ [)] _[ −]_ _[ϕ][π]_ [old][(] _[o]_ _i_ _[k]_ [)] 
so approximation errors in likelihood can be _exponentially_
amplified. As formally shown in Appendix A.1, the resulting error in the estimated RL objective becomes more
severe when less accurate log-likelihood approximations
are used, such as in _d1_, or ELBO used in DCE and UniGRPO when the Monte Carlo sample size _t_ is small.


Although increasing _t_ in the ELBO estimator can reduce
approximation error, the induced ratio estimates can still
exhibit high variance, as illustrated in Figure 1. Although
alternative approximator such as that in _d1_ can improve
efficiency, but yields a biased ratio that can differ substantially from the ELBO-based ratio, thereby introducing a
systematic bias into the RL training objective. Finally,
GRPO requires applying the approximation function _ϕ_
separately to three policies— _πθ_, _π_ old, and _π_ ref—which
further increases computational overhead.


3


Sample Size of t


Figure 1: Example policy ratio value
_ri_ _[k]_ [computed using ELBO and approxi-]
mated likelihood in _d1_ on GSM8K after
a policy update. Ratio’s unclipped interval is [1 _−_ _ϵ,_ 1 + _ϵ_ ], where _ϵ_ = 0 _._ 5.
ELBO-based likelihood approximation
yields high-variance ratio estimates; _d1_
induces a biased ratio that can deviate
substantially from ELBO. Both methods suffer from efficiently and accurately
compute ratios.


3 _wd1_ : WEIGHTED POLICY OPTIMIZATION FOR DLLMS


In this section, we introduce _wd1_, a novel RL algorithm that eliminates the need for approximating
the likelihood (policy) ratios for importance sampling, aiming to reduce the computational burden,
and the variance and approximation error in computing the RL objective. We further extend our
method to _wd1_ ++ by applying denoising-stepwise policy optimization.


3.1 REINFORCEMENT LEARNING AS WEIGHTED LOG-LIKELIHOOD MAXIMIZATION


Prevailing RL methods are based on constrained policy optimization (Belousov & Peters, 2017),
penalizing the deviation of current policy _πθ_ ( _·|q_ ) from the old policy _π_ old( _·|q_ ). TRPO objective
(Equation (2)) applies a forward-KL penalty. We instead adopt reverse-KL penalty augmented with
the reference policy regularization _D_ KL( _πθ_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ )):


         -         max _θ_ E _q∈D,o∼πθ_ ( _·|q_ ) _A_ _[π]_ [old] ( _q, o_ ) _−_ _λD_ KL _πθ_ ( _·|q_ ) _∥_ _π_ old( _·|q_ )


- - ��

_−_ _βD_ KL _πθ_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ ) _._


(4)


Note that the monotonic improvement guarantee still holds when using reverse-KL penalty, as we
show in Theorem 2. From the method of Lagrange multipliers, the solution to Equation (4) has the
following form (Peng et al., 2019; Rafailov et al., 2023):


              - _Aπ_ old( _q, ·_ )
_π_ _[∗]_ ( _·|q_ ) _∝_ _π_ old( _·|q_ ) _[λ/]_ [(] _[λ]_ [+] _[β]_ [)] _· π_ ref( _·|q_ ) _[β/]_ [(] _[λ]_ [+] _[β]_ [)] _·_ exp
_λ_ + _β_


_._ (5)


As the analytic form of the optimal policy _π_ _[∗]_ is known, we can train our policy by directly minimizing
_D_ KL( _π_ _[∗]_ ( _·|q_ ) _∥_ _πθ_ ( _·|q_ )). This minimization can be expressed as the following weighted log-likelihood
(WLL) loss, where the weights _∝_ exp - _ψA_ _[π]_ [old] [�], _ψ_ = _λ_ +1 _η_ [and the samples are obtained from the]
geometric mixture policy _π_ old [ref] [(] _[·|][q]_ [)] _[ ∝]_ _[π]_ [old][(] _[·|][q]_ [)] _[λ/]_ [(] _[λ]_ [+] _[β]_ [)] _[ ·]_ _[π]_ [ref][(] _[·|][q]_ [)] _[β/]_ [(] _[λ]_ [+] _[β]_ [)][ (See Proposition 2):] _[ ∀][q]_ _[∼D]_ [,]

_L_ WLL( _θ_ ) = E _o∼π_ oldref [(] _[·|][q]_ [)]     - _−_ exp     - _ψA_ _[π]_ [old] ( _q, o_ )� _·_ log _πθ_ ( _o|q_ )� (6)


- _G_ _−_ exp - _ψA_ [ˆ] _i_ 
_i_ =1 - _Gj_ =1 [exp] - _ψ_


- _Gj_ =1 [exp] - _ψA_ [ˆ] _j_ - log _πθ_ ( _oi|q_ )


_._ (7)


_≈_ E _{oi}Gi_ =1 _[∼][π]_ old [ref] [(] _[·|][q]_ [)]


1

_G_


_G_


As shown in Equation (7), we approximate the advantage function using the group-relative advantage _A_ [ˆ] and normalize the weights, thereby limiting their magnitude and reducing variance in loss
computation. Notably, dividing by the normalization constant does not affect the solution, since it
is independent of the completions. The resulting objective does not involve ratio _πθ_ ( _·|q_ ) _/π_ old( _·|q_ )
for importance sampling or _πθ_ ( _·|q_ ) _/π_ ref( _·|q_ ) for regularization, successfully avoiding the potential
amplification of log-likelihood approximation error and large variance in diffusion GRPO.


Although the objective _L_ WLL( _θ_ ) in Equation (7) avoids the likelihood ratio estimation, it has two
limitations. First, the algorithm is not fully utilizing all the completions. Due to the exponential form
of the weighting, completions with relatively low advantage – referred to as _negative_ samples – may
receive vanishingly small weights, and do not contribute to learning. Second, due to the likelihoodmaximization property of WLL, the likelihoods of all sampled completions are increased, even for
negative samples. This issue is exacerbated in scenarios where all completions attain identical but
low rewards (e.g. 0), thus all weights become equal and the likelihoods of these suboptimal samples
are nonetheless reinforced.


3.2 _wd1_ : FULLY UTILIZING COMPLETIONS


We propose _wd1_, an improved weighted log-likelihood objective that explicitly reinforces positive
samples and penalizes negative samples:


- _−_ _w_ [+] ( _q, oi_ ) + _w_ _[−]_ ( _q, oi_ )� _·_ log _πθ_ ( _oi|q_ )� _,_ (8)


4


              - 1
_Lwd1_ ( _θ_ ) = E _q∼D,{oi}Gi_ =1 _[∼][π]_ old [ref] [(] _[·|][q]_ [)] _G_


_G_


_i_ =1


where the weights are based on group-relative (GRPO) advantage and are further normalized to avoid
overly imbalanced weight _A_ [ˆ] _i_ = _R_ ( _q, oi_ ) _−_ mean( _R_ ( _q, o_ 1: _G_ )):


exp  - _ψA_ [ˆ] _i_  - exp  - _−_ _ψA_ [ˆ] _i_  
- _Gj_ =1 [exp] - _ψA_ [ˆ] _j_ - _,_ _w_ _[−]_ ( _q, oi_ ) = - _Gj_ =1 [exp] - _−_ _ψ_


exp       - _ψA_ [ˆ] _i_       _w_ [+] ( _q, oi_ ) =


- _Gj_ =1 [exp] - _−_ _ψA_ [ˆ] _j_ - _._ (9)


_wd1_ objective balances positive and negative samples through a complementary penalty term,
_w_ _[−]_ ( _q, oi_ ) log _πθ_ ( _oi|q_ ), which minimizes the likelihood of low-advantage completions. This penalty
induces negative gradients, thereby accelerating divergence from undesirable completions. Moreover,
in the extreme case where all completions exhibit identical advantages, the optimization naturally
halts since _w_ [+] = _w_ _[−]_, thereby addressing the concern on increasing likelihood of negative samples
proposed in Sec 3.1. We demonstrate the effectiveness of this combination via ablations in C.2.


Our method _wd1_, a simple ratio-free policy optimization based on **w** eighted log-likelihood objective
for **d** LLMs, is formally presented in Algorithm 1. We first obtain _G_ completions _{o}_ _[G]_ _i_ =1 [sampled]
from geometric mixture _π_ old [ref] [(] _[·|][q]_ [)] _[∝]_ _[π]_ [old][(] _[·|][q]_ [)] _[λ/]_ [(] _[λ]_ [+] _[β]_ [)] _[·]_ _[π]_ [ref][(] _[·|][q]_ [)] _[β/]_ [(] _[λ]_ [+] _[β]_ [)] [(line] [5).] [Since] [the] [base]
model LLaDA parametrizes the clean token prediction _π_ old [ref] [(] _[x]_ 0 _[k][|][x][t][, q]_ [)][ for denoising, we approximate]
log _π_ old [ref] [(] _[x]_ 0 _[k][|][x][t][, q]_ [)] _[ ≈]_ _[λ]_ [ log] _[ π]_ [old][(] _[x][k]_ 0 _[|][x][t][, q]_ [) +] _[ β]_ [ log] _[ π]_ [ref][(] _[x][k]_ 0 _[|][x][t][, q]_ [)][ as the logits of the denoising distribu-]
tion at each step _t_ . We then use the samples to compute weights in Equation (9) (line 6). In weights
computing, we leverage completions from all groups to estimate the normalization constant, in order
to restrict the the gradient norm and stabilize the training. Finally in line 8, we approximate the
log-likelihood of completions, and compute objectives for policy update. Likelihood approximation
in _d1_ (Zhao et al., 2025) is directly applicable to _wd1_ : log _πθ_ ( _x_ 0 _|q_ ) _≈_ [�] _k_ [log] _[ π][θ]_ [(] _[x]_ 0 _[k][|][x]_ [1] _[, q][′]_ [)][, where]
_q_ _[′]_ is randomly masked from prompt _q_ at every gradient step.


3.3 _wd1_ ++: STEPWISE WEIGHTED POLICY OPTIMIZATION


The decoding process in dLLMs relies on confidence-based remasking (Wang et al., 2025b). At each
denoising step _l ∈{_ 1 _, · · ·_ _, L}_ in decoding, clean data is predicted conditional on the masked sequence
_xl_ and then tokens with low-confidence are re-masked for further denoising, which construct a refinement process. Since current diffusion RL methods only use the final predicted clean completion for
training, there are bunch of clean completions in the intermediate denoising steps remaining _unused_ .


To leverage intermediate clean completions, we extend our weighted log-likelihood objective to
a step-wise formulation based on DCE, which we term _wd1_ ++. In _wd1_ (as well as in GRPO), a
group of completions _{oi}_ _[G]_ _i_ =1 [is sampled for policy optimization.] [In] _[ wd1]_ [++, we expand this group]
to _{Oi}_ _[G]_ _i_ =1 [, where] _[ O][i]_ [=] _[{][x]_ [0] _[|][l]_ _[|]_ _[x]_ [0] _[|][l]_ _[∼]_ _[π]_ old [ref] [(] _[·]_ _[|]_ _[x][t][, q]_ [)] _[,]_ _[x]_ [0] _[|][L]_ [=] _[o][i][,]_ _[l]_ [=] [1] _[, . . ., L][}][,]_ [ which contains]
all generated completions during the decoding process, including intermediate ones. The expanded
group of completions is then used to estimate both the advantage function and the corresponding
weights. The resulting loss objective is defined as:


_x_ 0 _|l∈Oi_


- _−_ _w_ [+] ( _q, x_ 0 _|l_ ) + _w_ _[−]_ ( _q, x_ 0 _|l_ )� _·_ log _πθ_ ( _x_ 0 _|l|xl, q_ )� _._


(10)


_G_


_i_ =1


_Lwd1_ ++( _θ_ ) = E _q∼D,_
_{Oi}_ _[G]_ _i_ =1 _[∼][π]_ old [ref] [(] _[·|][q]_ [)]
_l∈_ Unif _{_ 1 _,···,L}_


- _L_
_Gl_


4 THEORETICAL INSIGHTS: ENERGY-GUIDED DIFFUSION SAMPLING


In this section, we present a novel theoretical interpretation of policy optimization for _dLLMs_ . We
prove that the advantage-weighted log-likelihood objective ( _wd1_ ) for dLLMs can be viewed as
energy-guided discrete diffusion training combined with negative sample unlearning.


Sampling from the solution policy of the reverse-KL policy optimization, as described in Equation (5),
can be interpreted as energy-guided sampling, where the energy function _E_ ( _q, ·_ ) = _−A_ _[π]_ [old] ( _q, ·_ ).
Equation (5) defines the marginal distribution of the clean data ( _x_ 0 = _o_ ) which we denote as _p_ _[∗]_ 0 [(] _[x]_ [0][)][1][.]
To obtain the guidance at intermediate time steps _t >_ 0, we define the forward diffusion process for
the target diffusion policy _π_ _[∗]_ as following.
**Definition** **1.** _The_ _forward_ _diffusion_ _process_ _of_ _the_ _target_ _policy_ _(π_ _[∗]_ _)_ _satisfies_ _p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] [=]
_pt|_ 0( _xt|x_ 0) _, where pt|_ 0 _is the forward process of old diffusion policy πold._


1To adapt to the setting of diffusion, we use _xt_ to denote the (masked) completions, and omit the prompt _q_ .


5


**Algorithm 1** _wd1_ : **W** eighted Policy Optimization for **d** LLMs
**Require:** Reference model _π_ ref, prompt distribution _D_, group size _G_, reward function _R_, dLLM _πθ_,
regularization hyperparameters _λ_ and _β_

1: Initialize _πθ_ _←_ _π_ ref
2: **while** not converged **do**
3: _π_ old _←_ _πθ_
4: Sample prompt _q_ _∼D_ and _G_ completions _oi_ _∼_ _π_ old( _· | q_ ) _, ∀i ∈_ [ _G_ ]
5: Compute advantage _A_ [ˆ] _i_ = _R_ ( _q, oi_ ) _−_ mean( _R_ ( _q, o_ 1: _G_ )), _∀i ∈_ [ _G_ ]
6: Compute weights _w_ [+] and _w_ _[−]_ in Equation (9), _∀i ∈_ [ _G_ ]
7: **for** gradient update iterations _n_ = 1 _,_ 2 _, . . ., µ_ **do**
8: Compute approximated log-likelihood log _πθ_ ( _oi|q_ )
9: Compute objective _Lwd1_ ( _θ_ ) in Equation (8) or Equation (10) and update _θ_
10: **end for**
11: **end while**
12: **return** _πθ_


Since the reference diffusion policy is the initial policy, three policies have _identical forward diffusion_
_process_, being _p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][) =] _[ p][t][|]_ [0][(] _[x][t][|][x]_ [0][) =] _[ p]_ [ref] _t|_ 0 [(] _[x][t][|][x]_ [0][)][, and thus,] _[ p]_ _t_ _[∗]_ _|_ 0 [(] _[x][t][|][x]_ [0][) =] _[ p][′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)][, where]
_p_ _[′]_ is the geometric mixture diffusion _p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[∝]_ _[p][t][|]_ [0][(] _[x][t][|][x]_ [0][)] _[λ][p]_ [ref] _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[β]_ [.] [We can then obtain]
the energy guidance at all time step _t_ .

**Lemma 1** (Intermediate Energy Guidance on Discrete Diffusion) **.** _The marginal probability distribu-_
_tion of the masked responses (xt) in the diffusion process satisfies p_ _[∗]_ _t_ [(] _[x][t]_ [) =] _[ p]_ _t_ _[′]_ [(] _[x][t]_ [)] _[·]_ [exp] - _At_ ( _xt_ )� _/Zt,_
_which induces an energy-guided discrete diffusion:_


_p_ _[∗]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[ ∝]_ _[p][′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[ ·]_ [ exp(] _[A]_ [(] _[x]_ [0][)] _[ −]_ _[A][t]_ [(] _[x][t]_ [))] _[,]_ (11)


_where −At_ ( _xt_ ) = _−_ log E _x_ 0 _∼p_ _[′]_ 0 _|t_ [(] _[·|][x][t]_ [)][[exp] - _A_ ( _x_ 0)�] _is intermediate energy function for t >_ 0 _, and_
_A_ ( _·_ ) _is advantage function (Proof in Appendix A.3)._


The guidance provided in Lemma 1 demonstrates that it directs the sampling process toward
generating completions that exhibit higher advantage values. However, conducting training-free
guided sampling following Equation (11) requires estimating the posterior mean of the exponential
of advantage (Lu et al., 2023). Rather than relying on such estimation, we instead aim to find the
training objective to directly approximate the target guided diffusion model.


Since existing masked dLLMs parametrize the concrete score (Meng et al., 2022), to apply the energy
guidance, we aim to directly approximate target guided concrete score. Denote _xt_ = ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)]
and _x_ ˆ _t_ is identical to _xt_ except the _i_ -th token is unmasked (i.e. _x_ _[i]_ _t_ [= [] _[M]_ []][ and] _[x]_ [ˆ] _[i]_ _t_ [= [] _[M]_ []][).] [Concrete]
score is defined as the marginal probability ratio between _x_ ˆ _t_ and _xt_ :

def _t_ _[,][ · · ·]_ _[,]_ [ ˆ] _[x][i]_ _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)]
_s_ ( _xt, t_ ) = _[p]_ [(] _[x]_ [1] (12)
_p_ ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][i]_ _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)] _[.]_


We prove that the training objective to approximate the guided concrete score can be simplified as a
weighted Denoising Concrete Score Matching (D-CSM) (Meng et al., 2022):

**Theorem 1.** _The model sθ_ _approximates the concrete score of the energy-guided discrete diffusion_
_p_ _[∗]_ _when the following loss objective is minimized._ _This objective is in a form of_ _**advantage-weighted**_
_Denoising Concrete Score Matching, which we call AW-D-CSM:_


_·_ E _t∼_ [0 _,T_ ] _,p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)][[] _[∥][s][θ]_ [(] _[x][t][, t]_ [)] _[ −]_ _[p]_ 0 _[′]_ [(ˆ] _[x][t][|][x]_ [0][)] 2 []]
_p_ _[′]_ 0 [(] _[x][t][|][x]_ [0][)] _[∥]_ [2]

 - �� _LD-CSM_ ( _x_ 0)


_._ (13)


_LAW-D-CSM_ =E _x_ 0 _∼p_ _[′]_ 0 [(] _[·]_ [)]


exp - _A_ ( _x_ 0)�


 - ��  _Advantage Weight_


We provide the proof in Appendix A.3. Additionally, D-CSM is an approximation of CSM (Meng
et al., 2022), which is equivalent to Denoising score entropy (DSE) (Lou et al., 2024). For all _x_ 0,
it is satisfied up to multiplying a constant that _L_ D-CSM( _x_ 0) _⇔L_ CSM( _x_ 0) _⇔L_ DSE( _x_ 0) _⇔L_ DCE( _x_ 0)


6


(Ou et al., 2025). Therefore, AW-D-CSM can then be applied for both SEDD (Lou et al., 2024) and
RADD (Ou et al., 2025) model such as LLaDA. Denote _pθ_ as the concrete score reparametrized
model, AW-D-CSM can be converted to a weighted denoising cross-entropy loss (AW-DCE):


[1] _t_ [log] _[ p][θ]_ [(] _[x]_ 0 _[i]_ _[|][x]_ [UM] _t_ )� [�] _._ (14)


_L_ AW-DCE =E _x_ 0 _∼p_ _[′]_ 0 [(] _[·]_ [)]


exp - _A_ ( _x_ 0)� _·_ E _t∼_ [0 _,T_ ] _,p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] - 


 - _−_ [1]

_t_

_x_ _[i]_ _t_ [=[][mask][]]


DSE and DCE objectives both can be used for likelihood approximation in fine-tuning (Ou et al.,
2025; Nie et al., 2025a; Yang et al., 2025) since they can serve as negative ELBO (Lou et al., 2024;
Shi et al., 2025). Thus, the advantage-weighted objective AW-DCE (or AW-DSE) used to learn
energy-guided score is in a weighted log-likelihood form as in _wd1_ with only _w_ [+] (i.e. WLL loss in
Equation (6)), which contributes to our main theoretical findings:


**Remark 1.** _In the context of applying RL to masked discrete diffusion, the advantage-weighted log-_
_likelihood (WLL) objective (Equation_ (6) _) induced by reverse-KL policy optimization, is equivalent to_
_the objective of training energy-guided diffusion models, where the energy function is the negative_
_advantage._ _Formally, LWLL_ _⇔LAW-DCE when DCE is used for likelihood approximation._


**Remark 2.** _Additionally, based on DCE likelihood, the additional penalty term on negative samples_
_used_ _to_ _extend_ _WLL_ _to_ _wd1_ _loss_ _can_ _be_ _viewed_ _as_ _applying_ _data_ _unlearning_ _by_ _minimizing_ _the_
_ELBO (Alberti et al., 2025), where the data {x_ _[−]_ 0 _[}][ (negative samples) has probability distribution]_
_pdata_ ( _x_ _[−]_ 0 [)] _[∝]_ _[p][′]_ 0 [(] _[x][−]_ 0 [) exp(] _[−][A]_ [(] _[q, x]_ 0 _[−]_ [))] _[, which corresponds to a Boltzmann distribution that places]_
_higher probability mass on regions with lower advantage values (more details in Appendix D.1)._


5 EXPERIMENTS


In this section, we empirically validate the following key advantages of our approach:


i) Improved reasoning capabilities than existing methods on popular reasoning benchmarks;
ii) reduced computational burden, as reflected by decreased runtime, lower FLOPs and numbers of
function evaluations (NFEs) per training step, number of training steps and rollouts; and
iii) marked performance gains attributable to the incorporation of samples with low-advantage.


To evaluate our approach, we next detail the experimental setup and implementation.


**Experimental Setup.** We perform reinforcement learning (RL) fine-tuning on the LLaDA-8B-Instruct
model (Nie et al., 2025a) with Low-Rank Adaptation (LoRA) on: GSM8k (Cobbe et al., 2021), MATH
(Lightman et al., 2023), Sudoku (Arel, 2025), and Countdown (Pan et al., 2025). As for decoding,
we follow the default strategy Mounier & Idehpour (2025); Arriola et al. (2025); Wang et al. (2025b).
Our main baseline is _d1_ (Zhao et al., 2025), the _first_ RL method developed for masked diffusion LLMs
(dLLMs). We reproduce the baseline methods _Diffu_ -GRPO, which applies diffusion-based GRPO
training directly to the LLaDA base model, and _d1_, which performs SFT before applying _Diffu_ -GRPO.
We use s1K (Muennighoff et al., 2025) data for SFT in _d1_ . We also compare with SDPO (Han et al.,
2025), TCR (Wang et al., 2025d), and MDPO (He et al., 2025) on benchmarks GSM8K and MATH500.
MDPO is reproduced based on the official implementation and the training dataset (He et al., 2025).


**Implementation.** As for _wd1_, we conduct training on the same dataset as in _d1_ (Zhao et al., 2025):
training splits on GSM8k and MATH, and the dataset splits provided by Zhao et al. (2025) on Sudoku
and Countdown. In our implementation of _wd1_, we apply the same likelihood approximation method
as _d1_ . The hyperparameters used in our method and our reproduction of _d1_ are listed in Table 6 and
Table 5. As for _wd1_ ++, we train on dataset provided by (He et al., 2025), which is sampled from
OpenR1 dataset (Face, 2025). Since previous works (Yu et al., 2025) have demonstrated that the
reference policy is empirically unnecessary, we set _β_ = 0 and _λ_ = 1 to eliminate _π_ ref in practice. We
report results using _zero-shot_ and pass@1 evaluation on sequence lengths of 256 and 512 tokens.


5.1 MAIN RESULTS


**Superior Reasoning Ability.** In Table 1, we observe that _wd1_, even without supervised fine-tuning or
using any supervised data, consistently outperforms our reproduced implementation of _d1_ . Notably,


2In the technical report version of this work, our method achieved scores of 25.2 and 24.2 on Sudoku after
5K training steps. In this paper, we extend the training to 12.5K steps, and _wd1_ results in improved performance.


7


Table 1: Test Accuracy (%) of _**wd1**_ and _d1_ . We reproduce _d1_ and vary completion length. Our
approach without SFT, demonstrates particularly higher accuracy on Sudoku [2] and Countdown.


**Sudoku** **Countdown** **GSM8K** **MATH500**


**Model** / **Gen Len**


256 512 256 512 256 512 256 512


LLaDA-8B-Instruct 6.7 5.5 19.5 16.0 76.7 78.2 32.4 36.2
+ _diffu_ -GRPO 16.1 11.7 27.0 34.0 80.7 79.1 **34.4** **39.0**
+ SFT + _diffu_ -GRPO ( _d1_ ) 17.6 16.2 25.8 35.2 78.2 82.0 **34.4** 38.0


+ _**wd1**_ **76.4** **62.8** **51.2** **46.1** **80.8** **82.3** **34.4** **39.0**


Table 2: Comparison of Training Cost on 4 _×_ A100. We show SFT cost, average training time, FLOPs
evaluated by DeepSpeed Flops Profiler, and theoretical NFEs per training step which includes _µ_ = 8
gradient steps. _wd1_ removes SFT and has less cost per-step in RL than _d1_ .


**SFT** **RL Training**
**Method**

**Time Cost** **Time Cost** **FLOPs** **NFEs for Likelihood**


_d1_ 2.01 hrs 103.5 sec/step 9 _._ 922 _×_ 10 [15] /step ( _µ_ + 2)/step
_**wd1**_ 0 hrs 81.16 sec/step 8 _._ 887 _×_ 10 [15] /step _µ_ /step


_wd1_ surpasses _d1_ by 43% in test accuracy on the Sudoku task, and achieves up to a 25% improvement
on Countdown with maximum length 256. Relative to the base LLaDA model, the performance gain
reaches as high as 54% on Sudoku and 42% on Countdown. On math problem-solving benchmarks
GSM8K and MATH500, _wd1_ attains slightly higher accuracy. Nevertheless, the extended method
_wd1_ ++ obtains significantly better accuracy. In Table 3 (left), we further compare with concurrent
baselines released in recent months. _wd1_ ++ outperforms the baselines including strong one MDPO.


**Reduced Training Cost.** Table 2 demonstrates that the training cost required by _wd1_ is substantially
lower than that of _d1_ . Unlike _d1_, _wd1_ does not require a SFT stage, which alone accounts for
approximately two hours of training in _d1_ . _wd1_ achieves additional speedup during the RL phase,
where runtime is measured by averaging over _µ_ = 8 inner gradient steps per global step. Notably, the
time efficiency gap is expected to widen further under settings with larger maximum sequence lengths
and more diffusion steps. This efficiency gain is further supported by a reduced FLOPs and number
of function evaluations (NFEs) per step, as _wd1_ bypasses the need to approximate the likelihood
of the old policy. We exclude NFEs associated with sampling, since both methods share identical
sampling costs as _wd1_ removes the reference policy regularization.


In Table 3 (right), we report the training cost required to obtain the best post-trained models on
GSM8K and MATH500, measured in terms of the number of training steps and rollouts. _wd1_ ++
requires 10 _×_ fewer rollouts to achieve superior performance, clearly demonstrating the efficiency of
our method. This rapid convergence arises primarily from the _exponential_ advantage weights applied
to the log-likelihood in _wd1_ . In contrast, standard RL methods such as GRPO and PPO weight the
log-likelihood (or policy ratio) terms directly by the advantage function.


5.2 ABLATION STUDY


We present an ablation study in Figure 4. Notably, we observe that supervised fine-tuning (SFT)
yields only marginal improvements within our approach, with a slight gain in the Sudoku task. This
contrasts with _d1_, where SFT plays a significant role in improving performance. These findings
indicate that _wd1_ can eliminate the need for an SFT phase, thereby simplifying the training pipeline
and substantially reducing computational cost. Additionally, we evaluate the impact of removing the
negative-weighted term by setting _w_ _[−]_ = 0, thus relying solely on the positive advantage weights _w_ [+] .
We provide further ablation on the combined method between _w_ [+] and _w_ _[−]_ in Table 9. The results
highlight the importance of explicitly penalizing the likelihood of low-advantage completions, thereby
reinforcing the role of negative samples, and emphasize the critical balance between the two weights.


3To facilitate efficient ablation studies, we restrict our comparisons to checkpoints saved prior to 5000 steps.


8


Table 3: **Left:** Extended method _wd1++_ compared to concurrent RL methods to fine-tune LLaDA8B-Instruct. Methods denoted by “(full)” perform full fine-tuning. **Right:** Training cost to obtain the
best model on GSM8K and MATH500. We count the total number of steps of policy iteration (model
weights update), and the number of rollouts used for training (see Table 8 for details on counting).


**Model** **GSM8K** **MATH500**


LLaDA-8B-Instruct 78.2 36.2
+ _diffu_ -GRPO (Zhao et al., 2025) 80.7 39.0
+ _d1_ (Zhao et al., 2025) 82.0 38.0
+ SDPO (Han et al., 2025) (full) 81.2 + TCR (Wang et al., 2025d) 83.0 41.4
+ MDPO (He et al., 2025) (full) 83.4 43.4


+ _**wd1**_ 82.3 39.0
+ _**wd1**_ (full) 82.7 43.6


wd1++ MDPO d1


Table 4: Ablation on SFT and Negative Samples Weight ( _w_ _[−]_ ). We conduct _wd1_ training after SFT
( _wd1_ -SFT) and with only _w_ [+] (namely _wd1_ -P or WLL defined in Equation (6)) [3] . Results show
that _wd1_ performs better without SFT on planning and math tasks. Removing negative sample
reinforcement ( _w_ _[−]_ ) significantly hurts performance, highlighting its importance.


**Sudoku** **Countdown** **GSM8K** **MATH500**


**Model** / **Gen Len**


256 512 256 512 256 512 256 512


_wd1_ -P (WLL) 6.69 6.84 13.67 4.69 65.66 78.17 29.40 22.80


We further assess sensitivity to the relative weighting of positive and negative samples. The combined weight (cw) corresponds to _λ_ in the mix- 0.2
ture _−λw_ [+] + (1 _−_ _λ_ ) _w_ _[−]_, which scales the loglikelihood term in _wd1_ . Training on negative samples alone (cw= 0 _._ 0) yields a pronounced dete- 0.1
rioration in performance relative to our default
setting (cw= 0 _._ 5). The results reinforce our argu
|Col1|Col2|Col3|cw=0.0<br>cw=0.2|
|---|---|---|---|
||||cw=0.5<br>cw=0.8|
|||||
|||||


negative weights is most effective. In the absence Training Steps
of positive samples, the reinforcement-learning
signal collapses and optimisation becomes largely

Figure 2: Training rewards of _wd1_

ineffective. A large emphasis on positive samples

ent combined weights on Sudoku.

(cw= 0 _._ 8) causes performance to deteriorate more
rapidly, highlighting the critical role of negative samples in weighted log-likelihood methods.


0.2


0.1


0.0


Training Steps


Figure 2: Training rewards of _wd1_ under different combined weights on Sudoku.


6 RELATED WORK


**RL for Diffusion-based LLM.** RL for discrete diffusion models has been explored through several
approaches. One line of work, exemplified by DRAKES (Wang et al., 2024), leverages reward backpropagation along the denoising trajectory. This approach requires computing a critic and propagating
gradients through each denoising step, which is computationally intensive and prone to vanishing
gradients. Alternatively, methods such as MMaDA (Yang et al., 2025) and _d1_ (Zhao et al., 2025) adopt
direct RL formulations like GRPO, approximating missing diffusion components—such as per-token
likelihoods—for policy optimization. Zhu et al. (2025a) applies Direct Preference Optimization (DPO)
to fine-tune the LLaDA base model (Nie et al., 2025a), achieving notable gains in reasoning tasks.
However, these approaches all depend on likelihood ratios, which can introduce bias and instability
due to likelihood approximation errors. In contrast, our method derives a weighted policy optimiza

9


tion approach that eliminates the need for explicit policy ratios. Importantly, similar to prior works,
our method directly optimizes the predictive distribution over clean data. A complementary line of
research formulates policy optimization in terms of concrete scores (Lou et al., 2024; Meng et al.,
2022). SEPO (Zekri & Boullé, 2025), for instance, introduces a policy optimization objective that only
depends on concrete score estimation, thereby circumventing likelihood approximation altogether.


**RL for AR Models.** The connection between GRPO and weighted regression has recently been
explored in the context of RL with verifier reward (Mroueh, 2025), where binary rewards simplify
policy optimization into likelihood-based objectives. Other closely related approaches are Rejection
Sampling Fine-Tuning (RAFT), which maximizes the likelihood of positive-reward samples (Xiong
et al., 2025). Extensions of this idea incorporate negative samples to actively penalize the likelihood
of negative-reward completions while enhancing that of high-reward ones (Zhu et al., 2025b; Chen
et al., 2025). Other works introduce negative penalization through contrastive methods, such as
Noise Contrastive Estimation (NCE) (Gutmann & Hyvärinen, 2012; van den Oord et al., 2019; Chen
et al., 2024). Beyond binary rewards, preference-based learning has been widely studied using the
Bradley–Terry model (Bradley & Terry, 1952; Ouyang et al., 2022; Rafailov et al., 2024; Azar et al.,
2023; Ethayarajh et al., 2024; Wang et al., 2023; Hong et al., 2024). In contrast to these approaches,
our method accommodates general reward signals and can be interpreted as a form of soft rejection
sampling, enabling efficient and stable policy optimization for dLLMs.


**RL** **via** **Weighted** **Regression.** RL via weighted regression has been explored in earlier works
advantage-weighted regression (AWR) (Peng et al., 2019; Peters et al., 2010), and more recently in the
context of continuous control with diffusion policies (Ding et al., 2024; Zhang et al., 2025). Weighted
likelihood-based approaches have also been proposed for fine-tuning autoregressive (AR) language
models using general reward functions (Du et al., 2025; Baheti et al., 2024; Zhu et al., 2023). However,
for AR models, where likelihoods are tractable, the necessity of such approaches remains unclear.
In contrast, dLLMs suffer from intractable likelihoods, making weighted likelihood formulations
particularly advantageous by reducing the number of required likelihood approximations. As such,
RL via weighted likelihood provides a natural and efficient fit for optimizing dLLMs. In addition,
we demonstrate in ablation study that merely optimizing policy with AWR ( _wd1_ -P) is ineffective.


**"Ratio-Free" Policy Optimization.** If a policy optimization objective requires neither importance
sampling nor regularization with respect to a reference model, then the objective is ratio-free.
Consequently, _on-policy_ algorithms such as vanilla policy gradient methods (e.g., REINFORCE
(Williams, 1992)) and their variants (e.g., RLOO (Kool et al., 2019)) are inherently ratio-free. This
property is particularly valuable for dLLMs, where errors in log-likelihood approximation can
accumulate and propagate through ratio-based computations. Concurrent work, such as SPG (Wang
et al., 2025a), adopts a policy-gradient formulation and develops an objective tailored specifically
for diffusion language models. Another _on-policy_ optimization approach, d2 (Wang et al., 2025c),
removes both the ratios and the likelihood terms from the RL objective for dLLMs, offering a more
fundamental solution. However, our method _wd1_, similar to AWR (Peng et al., 2019), is inherently
an _off-policy_ loss, which is more general.


7 CONCLUSION


We introduce _wd1_, a weighted policy optimization method for reasoning with dLLMs. _wd1_ is designed
to minimize reliance on likelihood approximation, thereby mitigating the potentially substantial bias
that can arise from approximation errors in policy ratios. Our method is grounded in a weighted loglikelihood objective, derived to approximate the closed-form solution to the reverse-KL-constrained
policy optimization. Empirically, we show that _wd1_, even without supervised fine-tuning, surpasses
the existing method _d1_ by up to 16% in accuracy on reasoning benchmarks, while also delivering
notable improvements in computational efficiency during RL training. These results highlight the
effectiveness of _wd1_ and establish it as a more scalable and efficient approach for fine-tuning dLLMs.


8 ETHICS AND REPRODUCIBILITY STATEMENT


This work raises no question or concern regarding the Code of Ethics. As for reproducibility of our
results, we provide details of implementations in Section 5, in Experimental Setup and Implementation
subsections. Additional details including dataset, reward functions, and hyperparameters are provided
in Appendix B. All the theoretical results are proved in Appendix A.


10


9 ACKNOWLEDGMENTS


Ilija Bogunovic was supported by the EPSRC New Investigator Award EP/X03917X/1; the Engineering and Physical Sciences Research Council EP/S021566/1. Sangwoong Yoon was supported
by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant
funded by the Korea government (MSIT) (No. RS-2020-II201336, Artificial Intelligence Graduate
School Program (UNIST)), the National Research Foundation of Korea(NRF) grant funded by the
Korea government(MSIT) (No. RS-2024-00408003), and the Center for Advanced Computation at
Korea Institute for Advanced Study. Xiaohang Tang was supported by the Engineering and Physical
Sciences Research Council [grant number EP/T517793/1, EP/W524335/1]. Rares Dolga is supported
by EPSRC, grant reference number EP/S021566/1.


The authors would like to thank UIPath and Che Liu (Imperial College London) for providing
computing resources that supported our experiments, and Prof. David Barber, Yiming Yang, Xiaoyuan
Cheng, and Keyue Jiang from University College London for valuable discussions during the early
stages of this work.


REFERENCES


Silas Alberti, Kenan Hasanaliyev, Manav Shah, and Stefano Ermon. Data unlearning in diffusion
models. _arXiv preprint arXiv:2503.01034_, 2025.


[Arel. Arel’s sudoku generator. https://www.ocf.berkeley.edu/~arel/sudoku/main.](https://www.ocf.berkeley.edu/~arel/sudoku/main.html)
[html, 2025.](https://www.ocf.berkeley.edu/~arel/sudoku/main.html) Accessed: 2025-04-08.


Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. _arXiv preprint arXiv:2503.09573_, 2025.


Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured
Denoising Diffusion Models in Discrete State-Spaces, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2107.03006)
[abs/2107.03006.](https://arxiv.org/abs/2107.03006)


Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal
Valko, and Rémi Munos. A General Theoretical Paradigm to Understand Learning from Human
Preferences, 2023. [URL https://arxiv.org/abs/2310.12036.](https://arxiv.org/abs/2310.12036)


Ashutosh Baheti, Ximing Lu, Faeze Brahman, Ronan Le Bras, Maarten Sap, and Mark Riedl. Leftover
Lunch: Advantage-Based Offline Reinforcement Learning for Language Models, 2024. URL
[https://arxiv.org/abs/2305.14718.](https://arxiv.org/abs/2305.14718)


Boris Belousov and Jan Peters. f-divergence constrained policy improvement. _arXiv_ _preprint_
_arXiv:1801.00056_, 2017.


Ralph Allan Bradley and Milton E Terry. Rank Analysis of Incomplete Block Designs: I. The Method
of Paired Comparisons. _Biometrika_, 39(3/4):324–345, 1952.


Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and
Arnaud Doucet. A Continuous Time Framework for Discrete Denoising Models. _Advances in_
_Neural Information Processing Systems_, 35:28266–28279, 2022.


Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, and Jun Zhu. Noise Contrastive
Alignment of Language Models with Explicit Rewards, 2024. [URL https://arxiv.org/](https://arxiv.org/abs/2402.05369)
[abs/2402.05369.](https://arxiv.org/abs/2402.05369)


Huayu Chen, Kaiwen Zheng, Qinsheng Zhang, Ganqu Cui, Yin Cui, Haotian Ye, Tsung-Yi Lin,
Ming-Yu Liu, Jun Zhu, and Haoxiang Wang. Bridging Supervised Learning and Reinforcement
Learning in Math Reasoning. _arXiv preprint arXiv:2505.18116_, 2025.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems. _arXiv preprint arXiv:2110.14168_, 2021.


11


Shutong Ding, Ke Hu, Zhenhao Zhang, Kan Ren, Weinan Zhang, Jingyi Yu, Jingya Wang, and Ye Shi.
Diffusion-Based Reinforcement Learning via Q-Weighted Variational Policy Optimization. _arXiv_
_preprint arXiv:2405.16173_, 2024.


Yuhao Du, Zhuo Li, Pengyu Cheng, Zhihong Chen, Yuejiao Xie, Xiang Wan, and Anningzhe
Gao. Simplify RLHF as Reward-Weighted SFT: A Variational Method, 2025. URL [https:](https://arxiv.org/abs/2502.11026)
[//arxiv.org/abs/2502.11026.](https://arxiv.org/abs/2502.11026)


Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. KTO: Model
Alignment as Prospect Theoretic Optimization, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.01306)
[2402.01306.](https://arxiv.org/abs/2402.01306)


Hugging Face. Open r1: A fully open reproduction of deepseek-r1, 2025.


Aditya Golatkar, Alessandro Achille, and Stefano Soatto. Eternal sunshine of the spotless net:
Selective forgetting in deep networks. In _Proceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pp. 9304–9312, 2020.


Michael U. Gutmann and Aapo Hyvärinen. Noise-Contrastive Estimation of Unnormalized Statistical
Models, with Applications to Natural Image Statistics. _Journal of Machine Learning Research_, 13
(11):307–361, 2012. [URL http://jmlr.org/papers/v13/gutmann12a.html.](http://jmlr.org/papers/v13/gutmann12a.html)


Jiaqi Han, Austin Wang, Minkai Xu, Wenda Chu, Meihua Dang, Yisong Yue, and Stefano Ermon. Discrete diffusion trajectory alignment via stepwise decomposition. _arXiv preprint arXiv:2507.04832_,
2025.


Haoyu He, Katrin Renz, Yong Cao, and Andreas Geiger. Mdpo: Overcoming the training-inference
divide of masked diffusion language models. _arXiv preprint arXiv:2508.13148_, 2025.


Jiwoo Hong, Noah Lee, and James Thorne. ORPO: Monolithic Preference Optimization without
Reference Model, 2024. [URL https://arxiv.org/abs/2403.07691.](https://arxiv.org/abs/2403.07691)


Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. [URL https:](https://github.com/huggingface/open-r1)
[//github.com/huggingface/open-r1.](https://github.com/huggingface/open-r1)


Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In
_Proceedings of the nineteenth international conference on machine learning_, pp. 267–274, 2002.


Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free!
2019.


Hynek Kydlíˇcek. Math-Verify: Math Verification Library. URL [https://github.com/](https://github.com/huggingface/math-verify)
[huggingface/math-verify.](https://github.com/huggingface/math-verify)


Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer
Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and
Volodymyr Kuleshov. Mercury: Ultra-Fast Language Models Based on Diffusion, 2025. URL
[https://arxiv.org/abs/2506.17298.](https://arxiv.org/abs/2506.17298)


Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth_
_International Conference on Learning Representations_, 2023.


Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee,
and Min Lin. Understanding R1-Zero-Like Training: A Critical Perspective. _arXiv_ _preprint_
_arXiv:2503.20783_, 2025.


Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete Diffusion Modeling by Estimating the
Ratios of the Data Distribution, 2024. [URL https://arxiv.org/abs/2310.16834.](https://arxiv.org/abs/2310.16834)


Cheng Lu, Huayu Chen, Jianfei Chen, Hang Su, Chongxuan Li, and Jun Zhu. Contrastive Energy
Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning. In
_International Conference on Machine Learning_, pp. 22825–22855. PMLR, 2023.


12


Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete Score Matching: Generalized Score Matching for Discrete Data. _Advances in Neural Information Processing Systems_, 35:
34532–34545, 2022.


Nikita Mounier and Parsa Idehpour. Review, remask, refine (r3): Process-guided block diffusion for
text generation. _arXiv preprint arXiv:2507.08018_, 2025.


Youssef Mroueh. Reinforcement Learning with Verifiable Rewards: GRPO’s Effective Loss, Dynamics, and Success Amplification. _arXiv preprint arXiv:2503.06639_, 2025.


Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time
scaling, 2025. [URL https://arxiv.org/abs/2501.19393.](https://arxiv.org/abs/2501.19393)


Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan
Li. Scaling Up Masked Diffusion Models on Text. _arXiv preprint arXiv:2410.18514_, 2024.


Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong
Wen, and Chongxuan Li. Large Language Diffusion Models. _arXiv preprint arXiv:2502.09992_,
2025a.


Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin,
Ji-Rong Wen, and Chongxuan Li. Large Language Diffusion Models, 2025b. URL [https:](https://arxiv.org/abs/2502.09992)
[//arxiv.org/abs/2502.09992.](https://arxiv.org/abs/2502.09992)


Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li.
Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data,
2025. [URL https://arxiv.org/abs/2406.03736.](https://arxiv.org/abs/2406.03736)


Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training Language Models to Follow
Instructions with Human Feedback. _Advances_ _in_ _neural_ _information_ _processing_ _systems_, 35:
27730–27744, 2022.


Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero.
https://github.com/Jiayi-Pan/TinyZero, 2025. Accessed: 2025-01-24.


Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-Weighted Regression:
Simple and Scalable Off-Policy Reinforcement Learning. _arXiv preprint arXiv:1910.00177_, 2019.


Jan Peters, Katharina Mulling, and Yasemin Altun. Relative Entropy Policy Search. In _Proceedings_
_of the AAAI Conference on Artificial Intelligence_, volume 24, pp. 1607–1612, 2010.


David Pollard. Asymptopia: An Exposition of Statistical Asymptotic Theory. _URL http://www. stat._
_yale. edu/pollard/Books/Asymptopia_, 2000.


Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
_Advances in Neural Information Processing Systems_, 36:53728–53741, 2023.


Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea
Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2024.
[URL https://arxiv.org/abs/2305.18290.](https://arxiv.org/abs/2305.18290)


Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In _Proceedings_
_of the 26th ACM SIGKDD international conference on knowledge discovery & data mining_, pp.
3505–3506, 2020.


Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T
Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models, 2024. [URL https://arxiv.org/abs/2406.07524.](https://arxiv.org/abs/2406.07524)


13


John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust Region
Policy Optimization. In _International conference on machine learning_, pp. 1889–1897. PMLR,
2015.


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the Limits of Mathematical
Reasoning in Open Language Models. _arXiv preprint arXiv:2402.03300_, 2024.


Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and
Generalized Masked Diffusion for Discrete Data, 2025. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2406.04329)
[2406.04329.](https://arxiv.org/abs/2406.04329)


Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive
Predictive Coding, 2019. [URL https://arxiv.org/abs/1807.03748.](https://arxiv.org/abs/1807.03748)


Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond Reverse KL:
Generalizing Direct Preference Optimization with Diverse Divergence Constraints, 2023. URL
[https://arxiv.org/abs/2309.16240.](https://arxiv.org/abs/2309.16240)


Chengyu Wang, Paria Rashidinejad, DiJia Su, Song Jiang, Sid Wang, Siyan Zhao, Cai Zhou, Shannon Zejiang Shen, Feiyu Chen, Tommi Jaakkola, et al. Spg: Sandwiched policy gradient for
masked diffusion language models. _arXiv preprint arXiv:2510.09541_, 2025a.


Chenyu Wang, Masatoshi Uehara, Yichun He, Amy Wang, Tommaso Biancalani, Avantika Lal,
Tommi Jaakkola, Sergey Levine, Hanchen Wang, and Aviv Regev. Fine-Tuning Discrete Diffusion
Models via Reward Optimization with Applications to DNA and Protein Design. _arXiv preprint_
_arXiv:2410.13643_, 2024.


Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete
diffusion models with inference-time scaling. _arXiv preprint arXiv:2503.00307_, 2025b.


Guanghan Wang, Yair Schiff, Gilad Turok, and Volodymyr Kuleshov. d2: Improved techniques for
training reasoning diffusion language models. _arXiv preprint arXiv:2509.21474_, 2025c.


Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang,
Hao Chen, and Chunhua Shen. Time is a feature: Exploiting temporal dynamics in diffusion
language models. _arXiv preprint arXiv:2508.09138_, 2025d.


Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement
learning. _Machine learning_, 8(3):229–256, 1992.


Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong
Zhang, Caiming Xiong, and Hanze Dong. A Minimalist Approach to LLM Reasoning: From
Rejection Sampling to Reinforce, 2025. [URL https://arxiv.org/abs/2504.11343.](https://arxiv.org/abs/2504.11343)


Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada:
Multimodal Large Diffusion Language Models. _arXiv preprint arXiv:2505.15809_, 2025.


Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng
Kong. Dream 7B, 2025. [URL https://hkunlp.github.io/blog/2025/dream.](https://hkunlp.github.io/blog/2025/dream)


Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian
Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at
scale. _arXiv preprint arXiv:2503.14476_, 2025.


Oussama Zekri and Nicolas Boullé. Fine-Tuning Discrete Diffusion Models with Policy Gradient
Methods. _arXiv preprint arXiv:2502.01384_, 2025.


Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, and Wenhu Chen. Acecoder:
Acing coder rl via automated test-case synthesis. _arXiv preprint arXiv:2502.01718_, 2025.


Shiyuan Zhang, Weitong Zhang, and Quanquan Gu. Energy-Weighted Flow Matching for Offline
Reinforcement Learning. _arXiv preprint arXiv:2503.04975_, 2025.


14


Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling Reasoning in Diffusion
Large Language Models via Reinforcement Learning. _arXiv preprint arXiv:2504.12216_, 2025.


Banghua Zhu, Hiteshi Sharma, Felipe Vieira Frujeri, Shi Dong, Chenguang Zhu, Michael I Jordan,
and Jiantao Jiao. Fine-Tuning Language Models with Advantage-Induced Policy Alignment. _arXiv_
_preprint arXiv:2306.02231_, 2023.


Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei
Chen, Yankai Lin, Ji-Rong Wen, et al. LLaDA 1.5: Variance-Reduced Preference Optimization for
Large Language Diffusion Models. _arXiv preprint arXiv:2505.19223_, 2025a.


Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The Surprising
Effectiveness of Negative Reinforcement in LLM Reasoning. _arXiv preprint arXiv:2506.01347_,
2025b.


15


CONTENTS


**1** **Introduction** **1**


**2** **Preliminaries** **2**


2.1 Diffusion Large Language Models . . . . . . . . . . . . . . . . . . . . . . . . . . 2


2.2 Existing Policy Optimization Methods . . . . . . . . . . . . . . . . . . . . . . . . 3


2.3 Policy Optimization for dLLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . 3


**3** _**wd1**_ **:** **Weighted Policy Optimization for dLLMs** **4**


3.1 Reinforcement Learning as Weighted Log-Likelihood Maximization . . . . . . . . 4


3.2 _wd1_ : Fully Utilizing Completions . . . . . . . . . . . . . . . . . . . . . . . . . . 4


3.3 _wd1_ ++: Stepwise Weighted Policy Optimization . . . . . . . . . . . . . . . . . . . 5


**4** **Theoretical Insights:** **Energy-Guided Diffusion Sampling** **5**


**5** **Experiments** **7**


5.1 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7


5.2 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8


**6** **Related Work** **9**


**7** **Conclusion** **10**


**8** **Ethics and Reproducibility Statement** **10**


**9** **Acknowledgments** **11**


**A** **Proofs and Additional Theory** **18**


A.1 Objective Estimation Error due to Likelihood Approximation . . . . . . . . . . . . 18


A.2 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18


A.3 Masked Discrete Diffusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


**B** **Additional Experiment Setup Details** **23**


B.1 Dataset, Training and Evaluation Protocol . . . . . . . . . . . . . . . . . . . . . . 23


B.2 Reward Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


B.3 Sampling from Geometric Mixture . . . . . . . . . . . . . . . . . . . . . . . . . . 24


B.4 Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


B.5 Training Cost Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25


B.6 Computing Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25


**C** **Additional Experiments** **26**


C.1 Summary of _wd1_ Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


16


C.2 Additional Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


C.3 Coding Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28


C.4 Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28


**D** **Limitations** **28**


D.1 Additional Analysis on Unlearning . . . . . . . . . . . . . . . . . . . . . . . . . . 29


17


A PROOFS AND ADDITIONAL THEORY


A.1 OBJECTIVE ESTIMATION ERROR DUE TO LIKELIHOOD APPROXIMATION


In this section, we aim to show that _diffu_ -GRPO amplify the log-likelihood approximation error. Denote the approximator by _ϕ_ such that _∥ϕ_ _[π][θ]_ ( _q, o_ ) _−_ log _πθ_ ( _o|q_ ) _∥≤_ _ϵ_ and _∥ϕ_ _[π]_ [old] ( _q, o_ ) _−_ log _π_ old( _o|q_ ) _∥≤_
_ϵ_ _[′]_ . Then the objective _diffu_ -GRPO in the worst case suffers from exponential error. We discuss the
case without ratio clipping and omit the regularization for convenience. Denote _L_ GRPO as the ground
truth objective without likelihood approximation:


_∥Ldiffu_ -GRPO _−L_ GRPO _∥_


            
- exp _ϕ_ _[π][θ]_ ( _o_ _[k]_ _i_ [)] _[/]_ [ exp] _[ ϕ][π]_ [old] [(] _[o]_ _i_ _[k]_ [)] - _A_ ˆ _i_ 

       


_G_


_i_ =1


1
_|oi|_


_|oi|_


_k_ =1


= _||_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


       
_−_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


1
_G_


1
_G_


_G_


_i_ =1


1
_|oi|_


_|oi|_


_k_ =1


- _πθ_ ( _o_ _[k]_ _i_ [)] _[/π]_ [old][(] _[o]_ _i_ _[k]_ [)] - _A_ ˆ _i_


_||_


_|oi|_ 
- exp - _ϕ_ _[π][θ]_ ( _o_ _[k]_ _i_ [)] _[ −]_ _[ϕ][π]_ [old][(] _[o]_ _i_ _[k]_ [)] - _A_ ˆ _i_ 
_k_ =1


_G_


_i_ =1


1
_|oi|_


= _||_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


       
_−_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


1
_G_


1
_G_


_G_


_i_ =1


1
_|oi|_


_|oi|_


_k_ =1


        
- _πθ_ ( _o_ _[k]_ _i_ [)] _[/π]_ [old][(] _[o]_ _i_ _[k]_ [)] - _A_ ˆ _i_ - _||_


_G_


_i_ =1


1
_|oi|_


_|oi|_ 
- exp - log _πθ_ ( _o_ _[k]_ _i_ [)] _[ −]_ [log] _[ π]_ [old][(] _[o]_ _i_ _[k]_ [) + (] _[ϵ]_ [ +] _[ ϵ][′]_ [)] - _A_ ˆ _i_ 
_k_ =1


_≤||_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


       
_−_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


1
_G_


_|oi|_


_k_ =1


        
- _πθ_ ( _o_ _[k]_ _i_ [)] _[/π]_ [old][(] _[o]_ _i_ _[k]_ [)] - _A_ ˆ _i_ - _||_


1
_G_


_G_


_i_ =1


1
_|oi|_


_G_


_i_ =1


1
_|oi|_


_|oi|_ 
- exp - _ϵ_ + _ϵ_ _[′]_ [�] _A_ [ˆ] _i_ 

_k_ =1


_≤_ _C_ exp - _ϵ_ + _ϵ_ _[′]_ [�] _,_ (15)


= _||_ E _q∼D, o_ 1: _G∼π_ old( _·|q_ )


1
_G_


where _C_ is a constant independent to _ϵ_ and _ϵ_ _[′]_ . In contrast _wd1_ has only linear approximation error.
Denote the objective computed using approximated log-likelihood as _Lϕ_


_G_

                 -                 _∥Lϕ −Lwd1∥_ = _∥_ E _q∼D,{oi}Gi_ =1 _[∼][π]_ old [ref] [(] _[·|][q]_ [)]

_i_ =1


- - - - [�]

_−_ _w_ [+] ( _q, oi_ ) + _w_ _[−]_ ( _q, oi_ ) _·_ _ϕ_ ( _q, oi_ ) _−_ log _πθ_ ( _oi|q_ ) _∥_


_≤C_ _[′]_ _ϵ._ (16)


A.2 REINFORCEMENT LEARNING


**Reinforcement Learning Formulation.** We first introduce the reinforcement learning notations and
then extend it to the setting of LLM post-training. Denote _τ_ as a trajectory ( _τ_ = ( _s_ 0 _, a_ 0 _, s_ 1 _, . . ._ ) _∼_ _π_ )
sampled following policy _π_ . Specifically, _s_ 0 _∼_ _µ_, _at_ _∼_ _π_ ( _·|st_ ), _st_ +1 _∼_ _P_ ( _·|qt, at_ ). The objective of
Reinforcement Learning aims to find policy _π_, which maximizes a discounted total return,

_η_ ( _π_ ) = E _τ_ _∼π_             -             - _[∞]_ _γ_ _[t]_ _r_ ( _st, at, st_ +1)� _._


_t_ =0


Let the discounted return of a trajectory be _R_ ( _τ_ ) = [�] _t_ _[∞]_ =0 _[γ][t][r]_ [(] _[s][t][, a][t][, s][t]_ [+1][)][. The advantage function is]
defined as _A_ _[π]_ ( _s, a_ ) = _Q_ _[π]_ ( _s, a_ ) _−V_ _[π]_ ( _s_ ), where _V_ _[π]_ ( _s_ ) = E _τ_ _∼π_ [ _R_ ( _τ_ ) _|s_ 0 = _s_ ] is state value function,
and _Q_ _[π]_ ( _s, a_ ) = E _τ_ _∼π_ [ _R_ ( _τ_ ) _|s_ 0 = _s, a_ 0 = _a_ ] is state-action value function. Denote _ρπ_ old as the
marginal state distribution. Denote the total variation of two discrete probability distributions _a, b_ by
_DT V_ ( _a, b_ ) := [1] - _i_ _[|][a][i][ −]_ _[b][i][|]_ [ and] _[ D][T V]_ [ (] _[a, b]_ [)][2] _[≤]_ _[D]_ [KL][(] _[a][ ∥]_ _[b]_ [)][ (Pollard, 2000; Schulman et al., 2015).]


_DT V_ ( _a, b_ ) := [1] 2 - _i_ _[|][a][i][ −]_ _[b][i][|]_ [ and] _[ D][T V]_ [ (] _[a, b]_ [)][2] _[≤]_ _[D]_ [KL][(] _[a][ ∥]_ _[b]_ [)][ (Pollard, 2000; Schulman et al., 2015).]

When _a_ and _b_ are conditional probability distribution, denote _DT V_ [max][(] _[a, b]_ [) = max] _[q][ D]_ [TV][(] _[a]_ [(] _[·|][q]_ [)] _[∥][b]_ [(] _[·|][q]_ [))]
and _D_ KL [max][(] _[a][ ∥]_ _[b]_ [) = max] _[q][ D]_ [KL][(] _[a]_ [(] _[·|][q]_ [)] _[∥][b]_ [(] _[·|][q]_ [))][.]


[1] 
2


18


We then extend RL for LLM post-training. In this paper we only consider the sequence-level reward
and loss objective, so we directly replace _s_ with _q_ and _a_ with completion _o_ . Then the horizon of the
RL for post-training becomes only 1. The following theorem provides a monotonic (non-decreasing)
guarantee of existing prevailing RL methods.

**Proposition** **1** (Policy Improvement Bound (Kakade & Langford, 2002; Schulman et al.,
2015)) **.** _Let_ _surrogate_ _objective_ _Lπold_ ( _π_ ) = _η_ ( _πold_ ) + E _s∼ρπold_ ( _·_ ) _,_ _a∼π_ ( _·|s_ )� _A_ _[π][old]_ ( _s, a_ )� _,_ _and_ _C_ =
4 max _s,a,π |A_ _[π]_ ( _s, a_ ) _|γ/_ (1 _−_ _γ_ ) [2] _, then ∀k_ _∈_ N _:_

_η_ ( _π_ _[∗]_ ) _≥_ _Lπold_ ( _π_ _[∗]_ ) _−_ _CDTV_ [max][(] _[π][old][, π][∗]_ [)][2] _[.]_


**Remark** **3.** _Based_ _on_ _Proposition_ _1,_ _due_ _to_ _DTV_ [max][(] _[a][||][b]_ [)][2] _[≤]_ _[D]_ _KL_ [max][(] _[a][||][b]_ [)] _[(Pollard,]_ _[2000;]_ _[Schul-]_
_man_ _et_ _al.,_ _2015),_ _TRPO_ _and_ _PPO_ _with_ _fixed_ _forward_ _KL_ _regularization_ _have_ _the_ _monotonic_ _im-_
_provement_ _guarantees._ _In_ _other_ _words,_ _η_ ( _π_ _[∗]_ ) _≥_ _Lπold_ ( _π_ _[∗]_ ) _−_ _CDTV_ [max][(] _[π][old][, π][∗]_ [)][2] _[≥]_ _[L][π]_ _old_ [(] _[π][∗]_ [)] _[ −]_
_C_ E[ _DKL_ ( _πold∥π_ _[∗]_ )] _≥_ _Lπold_ ( _πold_ ) = _η_ ( _πold_ ) _._
**Proposition 2.** _Minimizing DKL_ ( _π_ _[∗]_ ( _·|q_ ) _∥_ _πθ_ ( _·|q_ )) _w.r.t._ _θ_ _is equivalent to optimize the following_
_loss objective:_

_LWLL_ ( _θ_ ) =E _q∼D,o∼πoldref_ [(] _[·|][q]_ [)]    - _−_ exp    - _ψA_ _[π][old]_ ( _q, o_ )� _·_ log _πθ_ ( _oi|q_ )� (17)


          - 1
_≈−_ E _{oi}_ G _i_ =1 _[∼][π]_ _old_ _[ref]_ [(] _[o][|][q]_ [)] _G_


G


_i_ =1


exp  - _ψA_ _[π][old]_ ( _q, oi_ )�  
�G _j_ =1 [[exp] - _ψA_ _[π][old]_ ( _q, oj_ )�] _·_ log _πθ_ ( _oi|q_ ) _._ (18)


_Proof._ To obtain the practical objective in Equation (18), we first start from the cross-entropy loss,
and obtain the following. _∀q_ _∈D_ :

_D_ KL( _π_ _[∗]_ ( _·|q_ ) _∥_ _πθ_ ( _·|q_ ))


               -               = _−_ E _o∼π∗_ ( _·|q_ ) log _πθ_ ( _o|q_ ) (19)


��     = _−_ _π_ _[∗]_ ( _o|q_ ) _·_ log _πθ_ ( _o|q_ ) (20)


_o_


��
= _−_


_o_


old  - (21)

_o_ _[′][ π]_ old [ref] [(] _[o][′][|][q]_ [)[exp] - _ψA_ _[π]_ [old] ( _q, o_ _[′]_ )�] _[·]_ [ log] _[ π][θ]_ [(] _[o][|][q]_ [)]


_π_ old [ref] [(] _[o][|][q]_ [) exp] - _ψA_ _[π]_ [old] ( _q, o_ )�


       - exp        - _ψA_ _[π]_ [old] ( _q, o_ )�        = _−_ E _o∼π_ oldref [(] _[o][|][q]_ [)] E _o′∼π_ oldref [[exp] - _ψA_ _[π]_ [old] ( _q, o_ _[′]_ )�] _[·]_ [ log] _[ π][θ]_ [(] _[o][|][q]_ [)] (22)


Since the normalization constant E _o′∼π_ oldref [[exp] - _ψA_ _[π]_ [old] ( _q, o_ _[′]_ )�] is independent to _o_, we can convert the
objective to a weighted log-likelihood, and approximate it with samples from the group and weight
normalization to obtain:

_L_ WLL( _θ_ ) = _−_ E _o∼π_ oldref [(] _[o][|][q]_ [)] �exp   - _ψA_ _[π]_ [old] ( _q, o_ )� _·_ log _πθ_ ( _o|q_ )� (23)


          - 1
_≈−_ E _{oi}_ G _i_ =1 _[∼][π]_ old [ref] [(] _[o][|][q]_ [)] _G_


G


_i_ =1


exp  - _ψA_ _[π]_ [old] ( _q, oi_ )�  
�G _j_ =1 [[exp] - _ψA_ _[π]_ [old] ( _q, oj_ )�] _·_ log _πθ_ ( _oi|q_ ) _._ (24)


We derive Equation (21) from Equation (20) by simply using the known form of the optimal policy
_π_ _[∗]_ ( _·|q_ ) _∝_ _π_ old [ref] [(] _[·|][q]_ [)] _[ ·]_ [ exp] - _ψA_ [ˆ] ( _q, ·_ )�. We derive Equation (22) from Equation (21) by using the
definition of expectation and from Equation (22) to Equation (24) by approximating through _G_
samples _{oi}_ [G] _i_ =1 _[∼]_ _[π]_ old [ref] [(] _[o][|][q]_ [)][.]


**Theorem** **2.** _Reverse-KL-regularized_ _Policy_ _Optimization_ _defined_ _in_ _the_ _following_ _objective_ _has_
_monotonic_ _improvement_ _guarantees._ _Specifically,_ _denote_ _regularized_ _objective_ _η_ _[′]_ ( _π_ ) = _η_ ( _π_ ) _−_
E _q∈D_ - _βDKL_ - _π_ ( _·|q_ ) _∥_ _πref_ ( _·|q_ )�� _and denote_


               -               - ��
_M_ ( _π_ ) = _L_ ( _π_ ) _−_ E _q∈D_ _λDKL_ ( _π_ ( _·|q_ ) _∥πold_ ( _·|q_ )) + _βDKL_ _π_ ( _·|q_ ) _∥_ _πref_ ( _·|q_ ) _,_ (25)


19


_where_ _L_ ( _π_ ) = _η_ ( _πold_ ) + E _q∼D,_ _o∼π_ ( _·|q_ )� _A_ _[π][old]_ ( _q, o_ )� _._ _Let_ _θ_ _[∗]_ _be_ _the_ _solution_ _to_ _the_ _objective_
max _θ M_ ( _πθ_ ) _:_


             -              - ��
_θ_ _[∗]_ = arg max _θ_ E _q∈D,o∼πθ_ ( _·|q_ ) _A_ _[π][old]_ ( _q, o_ ) _−_ _λDKL_ ( _πθ_ ( _·|q_ ) _∥_ _πold_ ( _·|q_ ) ) _−_ _βDKL_ _πθ_ ( _·|q_ ) _∥_ _πref_ ( _·|q_ )

(26)

_then η_ _[′]_ ( _π_ _[∗]_ ) _≥_ _η_ _[′]_ ( _πold_ ) _._


_Proof._ Based on Proposition 1, we have


             -             - ��
_η_ _[′]_ ( _π_ _[∗]_ ) = _η_ ( _π_ _[∗]_ ) _−_ E _q∈D_ _βD_ KL _π_ _[∗]_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ )


                      -                       - ��
_≥L_ ( _π_ _[∗]_ ) _−_ _CD_ TV [max][(] _[π]_ [old] _[, π][∗]_ [)][2] _[ −]_ [E] _[q][∈D]_ _βD_ KL _π_ _[∗]_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (27)


                      -                      - ��
_≥L_ ( _π_ _[∗]_ ) _−_ _CD_ KL [max][(] _[π][∗][∥][π]_ [old][)] _[ −]_ [E] _[q][∈D]_ _βD_ KL _π_ _[∗]_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (28)


             -             - ��
_≥L_ ( _π_ _[∗]_ ) _−_ E _q∈D_ _λD_ KL( _π_ _[∗]_ ( _·|q_ ) _∥π_ old( _·|q_ )) + _βD_ KL _π_ _[∗]_ ( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (29)


= _M_ ( _π_ _[∗]_ ) (30)
_≥M_ ( _π_ old) (31)


             -              - ��
= _L_ ( _π_ old) _−_ E _q∈D_ _λD_ KL( _π_ old( _·|q_ ) _∥π_ old( _·|q_ )) + _βD_ KL _π_ old( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (32)


             -              - ��
_≥L_ ( _π_ old) _−_ E _q∈D_ _βD_ KL _π_ old( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (33)


             -              - ��
= _η_ ( _π_ old) _−_ E _q∈D_ _βD_ KL _π_ old( _·|q_ ) _∥_ _π_ ref( _·|q_ ) (34)

= _η_ _[′]_ ( _π_ old) (35)

Equation (27) holds due to Proposition 1. Equation (28) holds due to _D_ TV [max][(] _[p][||][q]_ [)][2] _[≤]_ _[D]_ KL [max][(] _[p][||][q]_ [)][ (Pol-]
lard, 2000). Equation (29) holds due to the definition of _D_ KL [max][.] [Equation (30) is according to the defini-]
tion of _M_ ( _·_ ). The key inequality Equation (31) holds since _π_ _[∗]_ is the maximizer of function _L_ ( _·_ ). Equation (32) holds due to the definition of _M_ ( _·_ ). Equation (33) holds since _D_ KL( _π_ old( _·|q_ ) _∥π_ old( _·|q_ )) = 0.
Equation (34) holds since _L_ ( _π_ old) = _η_ ( _π_ old)+E _q∼D,_ _o∼π_ old( _·|q_ )� _A_ _[π]_ [old] ( _q, o_ )� = _η_ ( _π_ old). Equation (35)
is from the definition of _η_ _[′]_ .


A.3 MASKED DISCRETE DIFFUSION


In this section, we show how our objective learns a distribution for which all marginals at time _t_
satisfy intermediate energy guidance as per Lu et al. (2023).
**Definition 2.** _The absorbing transition kernel is defined as Qt_ = _σ_ ( _t_ ) _Q_ _[absorb]_ _, where_








_._



_Q_ _[absorb]_ =





_−_ 1 0 _· · ·_ 0 1
0 _−_ 1 _· · ·_ 0 1
_..._ _..._ _..._ _..._ _..._
0 0 _· · ·_ _−_ 1 1
0 0 _· · ·_ 0 0


**Definition 3** (Concrete Score) **.** _Denote xt_ = ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)] _[ and]_ _[x]_ [ˆ] _[t]_ _[is identical to][ x][t]_ _[except the][ i][-th]_
_token_ _is_ _unmasked_ _(i.e._ _x_ _[i]_ _t_ [=] [[] _[M]_ []] _[and]_ _[x]_ [ˆ] _[i]_ _t_ [=] [[] _[M]_ []] _[).]_ _[Concrete]_ _[score]_ _[is]_ _[defined]_ _[as]_ _[the]_ _[marginal]_
_probability ratio between_ _x_ ˆ _t and xt:_

_def_ _t_ _[,][ · · ·]_ _[,]_ [ ˆ] _[x][i]_ _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)]
_s_ ( _xt, t_ ) = _[p]_ [(] _[x]_ [1] (36)
_p_ ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][i]_ _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)] _[.]_

**Proposition 3** ( **Marginal Distribution** (Ou et al., 2025)) **.** _Denote {xt} as a continuous time Markov_
_chain with transition matrix_ **Q** _t_ = _σ_ ( _t_ ) **Q** _[absorb]_ _._ _Assume d_ 1 _tokens in xt_ = ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)] _[ are masked]_
_tokens_ [ **M** ] _, and d_ 2 = _d −_ _d_ 1 _tokens are unmasked, the marginal distribution pt_ ( _xt_ ) _satisfies_

_pt_ ( _xt_ ) = �1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [1][�] _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [2] _p_ 0( _x_ _[UM]_ _t_ ) _,_ (37)

_where_ _σ_ ¯( _t_ ) = �0 _t_ _[σ]_ [(] _[s]_ [)] _[ds][, and][ x]_ _t_ _[UM]_ _is the set of unmasked tokens in xt._


20


The following theorem provides the foundation of directly modeling the clean data distribution.
**Proposition 4** ( **Analytic Concrete Score** (Ou et al., 2025)) **.** _Denote xt_ = ( _x_ [1] _t_ _[,][ · · ·]_ _[, x][d]_ _t_ [)] _[ and]_ _[x]_ [ˆ] _[t]_ _[is]_
_identical to xt_ _except the i-th token is unmasked (i.e._ _x_ _[i]_ _t_ [= [] _[M]_ []] _[ and]_ _[x]_ [ˆ] _[i]_ _t_ [= [] _[M]_ []] _[).]_ _[Then the concrete]_
_score at time t can be expressed by the conditional probability of predicting this unmasked token._


_pt_ ( _x_ [1] _t_ _[. . .]_ [ ˆ] _[x]_ _t_ _[i]_ _[. . . x]_ _t_ _[d]_ [)] _e_ _[−][σ]_ [¯][(] _[t]_ [)]
_pt_ ( _x_ [1] _t_ _[. . . x]_ _t_ _[i]_ _[. . . x]_ _t_ _[d]_ [)] [=] 1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)] _[p]_ [0][(ˆ] _[x]_ _t_ _[i]_ _[|][ x]_ _t_ _[UM]_ )

_._

**Lemma** ( **1** ) **.** _The_ _marginal_ _probability_ _distribution_ _of_ _the_ _masked_ _responses_ _(xt)_ _in_ _the_ _diffusion_
_process_ _satisfies_ _p_ _[∗]_ _t_ [(] _[x][t]_ [)] [=] _[p][′]_ _t_ [(] _[x][t]_ [)] _[·]_ [exp] - _At_ ( _xt_ )� _/Zt,_ _which_ _induces_ _an_ _energy-guided_ _discrete_
_diffusion:_

_p_ _[∗]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[ ∝]_ _[p][′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[ ·]_ [ exp(] _[A]_ [(] _[x]_ [0][)] _[ −]_ _[A][t]_ [(] _[x][t]_ [))] _[,]_ (38)

_where intermediate energy function is defined as At_ ( _x_ 0) = log E _x_ 0 _∼p_ _[′]_ 0 _|t_ [(] _[·|][x][t]_ [)][[exp] - _A_ ( _x_ 0)�] _for t >_ 0 _,_
_and A_ 0( _x_ 0) = _A_ ( _x_ 0) _, A_ ( _·_ ) _is advantage function, Zt is the normalization constant._


_Proof._ The theorem and proof mainly extend from theory developed in continuous setting (Lu et al.,
2023). According to the marginal likelihood of clean data distribution _p_ _[∗]_ 0 [(] _[x]_ [0][) =] _[ p][′]_ 0 [(] _[x]_ [0][)] _[e][A]_ _Z_ [(] _[x]_ [0)], and

identical forward process, we can rewrite the marginal likelihood of masked data:


         -         _p_ _[∗]_ _t_ [(] _[x][t]_ [) =] _p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[∗]_ [(] _[x]_ [0][) d] _[x]_ [0] [=] _p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][)] _[e][ψA]_ [(] _[x]_ [0][)] d _x_ 0

_Z_


         = _p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][)] _[e][ψA]_ [(] _[x]_ [0][)] d _x_ 0 _._

_Z_

Applying Bayesian rule we know that _p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][) =] _[ p][′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[p]_ _t_ _[′]_ [(] _[x][t]_ [)][, hence we can further]
rewrite


    _p_ _[∗]_ _t_ [(] _[x][t]_ [) =] _p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][)] _[e][ψA]_ [(] _[x]_ [0][)]


d _x_ 0
_Z_


[(] _[x]_ [0][)]   
d _x_ 0 = _p_ _[′]_ _t_ [(] _[x][t]_ [)] _p_ _[′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[e][ψA]_ [(] _[x]_ [0][)]
_Z_ _Z_


_p_ _[′]_ _t_ [(] _[x][t]_ [)][ E] _p_ _[′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)]  - _e_ _[ψA]_ [(] _[x]_ [0][)][�]
=


[0] _[|][x][t]_ [)] _e_ [0]

_t_ [(] _[x][t]_ [)] _[ e][ψA][t]_ [(] _[x][t]_ [)]
= _[p][′]_
_Z_ _Zt_


_Z_ _Zt_

Therefore, the marginal likelihood of masked sequence satisfies: _p_ _[∗]_ _t_ [(] _[x][t]_ [) =] _[ p]_ _t_ _[′]_ [(] _[x][t]_ [)] _[ ·]_ [ exp] - _At_ ( _xt_ )� _/Zt_ .
Since _p_ _[∗]_ _t|_ 0 [=] _[p][′]_ _t|_ 0 [,] [based] [on] [the] [marginal] [likelihood] [of] [clean] [data] [distribution] [satisfies] _[p][∗]_ 0 [(] _[x]_ [0][)] [=]

_p_ _[′]_ 0 [(] _[x]_ [0][)] _[e][A]_ _Z_ [(] _[x]_ [0)], we can further applying Bayesian rule to obtain the energy-guided discrete diffusion

model:


_p_ _[′]_ 0 [(] _[x]_ [0][)] _[e][A]_ _Z_ [(] _[x]_ [0)]


_p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[∗]_ [(] _[x]_ [0][)]
_p_ _[∗]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [) =] (39)

_p_ _[∗]_ _t_ [(] _[x][t]_ [)]


_p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][)] _[e][A]_ _Z_ [(] _[x]_ [0)]
= (40)

_p_ _[∗]_ _t_ [(] _[x][t]_ [)]


_p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ 0 _[′]_ [(] _[x]_ [0][)] _[e][A]_ _Z_ [(] _[x]_ [0)]
=


0 _Z_

_p_ _[′]_ _t_ [(] _[x][t]_ [)] _[ e][ψAt]_ [(] _[xt]_ [)]


(41)

_[ e][ψAt]_ [(] _[xt]_ [)]

_Zt_


_∝_ _p_ _[′]_ 0 _|t_ [(] _[x]_ [0] _[|][x][t]_ [)] _[ ·]_ [ exp(] _[A]_ [(] _[x]_ [0][)] _[ −]_ _[A][t]_ [(] _[x][t]_ [))] _[,]_ (42)


**Lemma 2.** _According to Definition 1, due to the identical forward process_

_p_ _[∗]_ _t|_ 0 [(] _[x][t][|][x]_ [0][) =] _[ p][′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][) =] _[ p][ref]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)] _[,]_ (43)


_based_ _on Lemma_ _1 and_ _Proposition 3,_ _we have_ _the_ _marginal_ _probability of_ _the unmasked_ _tokens_
_satisfies that for all step t,_

_p_ _[∗]_ 0 [(] _[x][UM]_ _t_ _|q_ ) = _p_ 0( _x_ _[UM]_ _t_ _|q_ ) _[λ]_ _· p_ _[ref]_ 0 [(] _[x]_ _t_ _[UM]_ _|q_ ) _[β]_ _·_ E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp]      - _A_ ( _q, x_ 0)�] _/Z,_ (44)

_where Z_ _is the normalization constant._


21


_Proof._ According to the identical forward distribution of three diffusion process (new, old, and
reference), based on Equation (37), we have _∀t_ :


_pt_ ( _xt|q_ ) = �1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [1][�] _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [2] _p_ 0( _x_ [UM] _t_ _|q_ ) (45)

_p_ _[∗]_ _t_ [(] _[x][t][|][q]_ [) =] �1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [1][�] _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [2] _p_ _[∗]_ 0 [(] _[x]_ [UM] _t_ _|q_ ) (46)

_p_ [ref] _t_ [(] _[x][t][|][q]_ [) =] �1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [1][�] _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [2] _p_ [ref] 0 [(] _[x]_ _t_ [UM] _|q_ ) (47)


Then rewrite Equation (46) in the residual energy-based form defined in Equation (38), we have

�1 _−_ _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [1][�] _e_ _[−][σ]_ [¯][(] _[t]_ [)][�] _[d]_ [2] _p_ _[∗]_ 0 [(] _[x]_ [UM] _t_ _|q_ ) = _p_ _[∗]_ _t_ [(] _[x][t][|][q]_ [) =] _[p]_ _t_ _[′]_ [(] _[x][t][|][q]_ [)] _[ ·]_ [ exp]     - _At_ ( _q, xt_ )� _/Z._ (48)


By plugging _p_ _[′]_ _t_ [(] _[x][t][|][q]_ [)] [=] _[p][t]_ [(] _[x][t][|][q]_ [)] _[λ]_ _[·][ p]_ [ref] _t_ [(] _[x][t][|][q]_ [)] _[β]_ [and] [Equation] [(45)] [and] [Equation] [(47)] [into] [Equa-]
tion (48), we have that the clean data distribution of the unmask tokens at diffusion time _t_ satisfies:


_p_ _[∗]_ 0 [(] _[x]_ [UM] _t_ _|q_ ) = _p_ 0( _x_ [UM] _t_ _|q_ ) _[λ]_ _· p_ [ref] 0 [(] _[x]_ _t_ [UM] _|q_ ) _[β]_ _·_ exp      - _At_ ( _q, xt_ )� _/Z_ (49)

= _p_ 0( _x_ [UM] _t_ _|q_ ) _[λ]_ _· p_ [ref] 0 [(] _[x]_ _t_ [UM] _|q_ ) _[β]_ _·_ E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp]       - _A_ ( _q, x_ 0)�] _/Z._ (50)


**Proposition** **5.** _The_ _marginal_ _likelihood_ _of_ _the_ _target_ _diffusion_ _model_ _p_ _[∗]_ _satisfies_ _Equation_ (38) _._
_Consequently, the concrete score of the target diffusion model, denoted by s_ _[∗]_ _, can be expressed by the_
_score of the mixture diffusion p_ _[′]_ _and the posterior mean of the advantage:_

_s_ _[∗]_ ( _xt, t_ ) = _s_ _[′]_ ( _xt, t_ ) _·_ [E] E _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)][[exp][[exp] �� _AA_ (( _xx_ 00))��]] _//ZZ_ [ˆ] _[,]_ (51)


_and equivalently_

_p_ 0(ˆ _x_ _[i]_ _t_ _[|][x]_ _t_ _[UM]_ _, q_ ) _[λ]_ _· p_ _[ref]_ 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ _[UM]_ _, q_ ) _[β]_ _·_ [E] E _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)][[exp][[exp] �� _AA_ (( _q, xq, x_ 00))��]] _//ZZ_ [ˆ] _[.]_ (52)


_Proof._ According to Lemma 2


_p_ _[∗]_ 0 [(] _[x]_ _t_ [UM] _,_ ˆ _x_ _[i]_ _t_ _[|][q]_ [)] _t_ _,_ ˆ _x_ _[i]_ _t_ _[|][q]_ [)] _[λ]_
= _[p]_ [0][(] _[x]_ [UM]
_p_ _[∗]_ 0 [(] _[x]_ _t_ [UM] _|q_ ) _p_ 0( _x_ [UM] _t_ _|q_ ) _[λ]_


[(] _[x]_ _t_ [UM] _,_ ˆ _x_ _[i]_ _t_ _[|][q]_ [)] _[β]_ _·_ [E] _[p]_ 0 _[′]_ [(] _[x]_ [0] _[|][x]_ [ˆ] _[t]_ [)][[exp]   - _A_ ( _q, x_ 0)�] _/Z_ [ˆ]

_p_ [ref] 0 [(] _[x]_ _t_ [UM] _|q_ ) _[β]_ E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp] - _A_ ( _q, x_ 0)�] _/Z_


[(] _[x]_ [UM] _t_ _,_ ˆ _x_ _[i]_ _t_ _[|][q]_ [)] _[λ]_ 0 [(] _[x]_ _t_ [UM] _,_ ˆ _x_ _[i]_ _t_ _[|][q]_ [)] _[β]_

_·_ _[p]_ [ref]
_p_ 0( _x_ [UM] _t_ _|q_ ) _[λ]_ _p_ [ref] 0 [(] _[x]_ _t_ [UM] _|q_ ) _[β]_


0 (53)

E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp] - _A_ ( _q, x_ 0)�] _/Z_


_p_ _[∗]_ 0 [(ˆ] _[x][i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) = _p_ 0(ˆ _x_ _[i]_ _t_ _[|][x]_ _t_ [UM] _, q_ ) _[λ]_ _· p_ [ref] 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ [UM] _, q_ ) _[β]_ _·_ [E] E _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)][[exp][[exp] �� _AA_ (( _q, xq, x_ 00))��]] _//ZZ_ [ˆ] _[.]_ (54)


_e_ _[−·]_ _σ_ (¯ _t_ )
Both sides in Equation (52) multiply _C_ ( _t_ ) = 1 _−eσ_ (¯ _t_ ) [and based on the analytic form of concrete]

score introduced in Proposition 4, _C_ ( _t_ ) _· p_ 0(ˆ _x_ _[i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) = _s_ ( _x_ _[i]_ _t_ _[, t]_ [)][.] [Thus, we have]


_C_ ( _t_ ) _· p_ _[∗]_ 0 [(ˆ] _[x][i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) = _C_ ( _t_ ) _· p_ 0(ˆ _x_ _[i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) _[λ]_ _· p_ [ref] 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ [UM] _t_ _, q_ ) _[β]_ _·_ [E] E _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)][[exp][[exp] �� _AA_ (( _q, xq, x_ 00))��]] _//ZZ_ [ˆ]

(55)


_s_ _[∗]_ ( _xt, t_ ) = _s_ _[′]_ ( _xt, t_ ) _·_ [E] E _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)][[exp][[exp] �� _AA_ (( _q, xq, x_ 00))��]] _//ZZ_ [ˆ] _[.]_ (56)


**Lemma 3.** _The normalization constant Z_ = [�] _xt_ _[p][′]_ [(] _[x][t][|][q]_ [)] _[ ·]_ [ E] _[x]_ 0 _[|][x]_ _t_ [[exp] _[ A]_ [(] _[q, x]_ [0][)]] _[ is independent to]_

_the_ _masked_ _response_ _xt._ _In_ _other_ _words,_ _Z_ = _Z_ [ˆ] := [�] _x_ ˆ _t_ _[p][′]_ [(ˆ] _[x][t][|][q]_ [)] _[ ·]_ [ E] _[x]_ 0 _[|][x]_ [ˆ] _t_ [[exp] _[ A]_ [(] _[q, x]_ [0][)]] _[for]_ _[any]_


**Lemma 3.** _The normalization constant Z_ = [�]


_the_ _masked_ _response_ _xt._ _In_ _other_ _words,_ _Z_ = _Z_ [ˆ] := [�] _x_ ˆ _t_ _[p][′]_ [(ˆ] _[x][t][|][q]_ [)] _[ ·]_ [ E] _[x]_ 0 _[|][x]_ [ˆ] _t_ [[exp] _[ A]_ [(] _[q, x]_ [0][)]] _[for]_ _[any]_

_x_ ˆ _t_ = _xt._


22


_Proof._


     _p_ _[′]_ ( _xt|q_ ) _·_ E _x_ 0 _|xt_ [exp _A_ ( _q, x_ 0)] =
_xt_ _xt_


_Z_ = 


- _p_ _[′]_ ( _xt|q_ ) _·_ 

_xt_ _x_ 0


_p_ _[′]_ ( _x_ 0 _|xt_ ) _·_ exp _A_ ( _q, x_ 0) (57)

_x_ 0


= 

_xt_


- _p_ _[′]_ ( _x_ 0 _, xt|q_ ) _·_ exp _A_ ( _q, x_ 0) = 

_x_ 0 _x_ 0


_x_ 0


- _p_ _[′]_ ( _x_ 0 _, xt|q_ ) _·_ exp _A_ ( _q, x_ 0) (58)


_xt_


=   - _p_ _[′]_ ( _x_ 0 _|q_ ) _·_ exp _A_ ( _q, x_ 0) (59)


_x_ 0


Thus _Z_ becomes independent to _xt_, leading to that


      _Z_ = _Z_ [ˆ] := _p_ _[′]_ (ˆ _xt|q_ ) _·_ E _x_ 0 _|x_ ˆ _t_ [exp _A_ ( _q, x_ 0)] (60)

_x_ ˆ _t_


**Theorem** ( **1** ) **.** _The score model sθ_ = _s_ _[∗]_ _defined in Equation_ (52) _is satisfied when the following loss_
_objective is minimized._ _This objective is in a form of_ _**advantage-weighted**_ _Denoising Concrete Score_
_Matching (D-CSM), which we call AW-D-CSM:_


_LAW-D-CSM_ =E _p_ _[′]_ 0 [(] _[x]_ [0][)][[exp] - _A_ ( _q, x_ 0)�

          - ��          _Advantage Weight_


_·_ E _t∼_ [0 _,T_ ] _,p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)][[] _[∥][s][θ]_ [(] _[x][t][, t]_ [)] _[ −]_ _[p]_ 0 _[′]_ [(ˆ] _[x][t][|][x]_ [0][)] 2 []] ] _._ (61)
_p_ _[′]_ 0 [(] _[x][t][|][x]_ [0][)] _[∥]_ [2]

 - �� _LD-CSM_ ( _x_ 0)


_e_ _[−][σ]_ [¯][(] _[t]_ [)]
_Proof._ Denote _sθ_ ( _xt, t_ ) = 1 _−e_ _[−][σ]_ [¯][(] _[t]_ [)] _[p][θ]_ [(ˆ] _[x][i]_ _t_ _[|][ x]_ _t_ [UM] ) is the concrete score model induced by _pθ_ . Accord
ing to Lemma 3, _Z_ [ˆ] = _Z_ . Then according to Proposition 5, Equation (52) is equivalent to

_p_ _[∗]_ 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ [UM] _, q_ ) _·_ E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp]         - _A_ ( _q, x_ 0)�]

= _p_ 0(ˆ _x_ _[i]_ _t_ _[|][x]_ _t_ [UM] _, q_ ) _[λ]_ _· p_ [ref] 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ [UM] _, q_ ) _[β]_ _·_ E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x]_ [ˆ] _[t]_ [)][[exp]     - _A_ ( _q, x_ 0)�] (62)


We aim to update _pθ_ (ˆ _x_ _[i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) _→_ _p_ _[∗]_ 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ [UM] _t_ _, q_ ) to satisfy Equation (62), thus we can construct a
loss function objective by replacing _p_ _[∗]_ with _pθ_ and construct a _L_ [2] norm loss

E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp]  - _A_ ( _q, x_ 0)� _· ∥pθ_ (ˆ _x_ _[i]_ _t_ _[|][x]_ _t_ [UM] _, q_ ) _−_ _[p]_ _p_ 0 _[′][′]_ 0 [(][(] _[x][x]_ [0][0] _[|][|][x][x]_ [ˆ] _[t][t]_ [)][)] _[·][ p]_ [0][(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ [UM] _, q_ ) _[λ]_ _p_ [ref] 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ _t_ [UM] _, q_ ) _[β]_ _∥_ [2] 2 []]

(63)

=E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp] - _A_ ( _q, x_ 0)� _· ∥pθ_ (ˆ _x_ _[i]_ _t_ _[|][x]_ _t_ [UM] _, q_ ) _−_ _[p]_ 0 _[′]_ [(] _[x]_ [0] _[|][x]_ [ˆ] _[t]_ [)] 0 [(ˆ] _[x]_ _t_ _[i][|][x]_ [UM] _t_ _, q_ ) _∥_ [2] 2 []] (64)
_p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)] _[·][ p][′]_

=E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp] - _A_ ( _q, x_ 0)� _· ∥pθ_ (ˆ _x_ _[i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) _−_ _[p]_ 0 _[′]_ [(ˆ] _[x][t][|][x]_ [0][)] _[p]_ _t_ _[′]_ [(] _[x][t]_ [)] 0 [(ˆ] _[x][i]_ _t_ _[|][x]_ [UM] _t_ _, q_ ) _∥_ [2] 2 []] (65)
_p_ _[′]_ 0 [(] _[x][t][|][x]_ [0][)] _[p]_ _t_ _[′]_ [(ˆ] _[x][t]_ [)] _[·][ p][′]_

=E _p_ _[′]_ 0 [(] _[x]_ [0] _[|][x][t]_ [)][[exp] - _A_ ( _q, x_ 0)� _· ∥sθ_ ( _xt, t_ ) _−_ _[p]_ 0 _[′]_ [(ˆ] _[x][t][|][x]_ [0][)] 2 []] _[.]_ (66)
_p_ _[′]_ 0 [(] _[x][t][|][x]_ [0][)] _[∥]_ [2]


B ADDITIONAL EXPERIMENT SETUP DETAILS


B.1 DATASET, TRAINING AND EVALUATION PROTOCOL


As for _wd1_ and _d1_, we reproduce _d1_ by running the official code [4] without and change, and train
our method _wd1_ evaluated for accuracy of the test datasets at steps 1000, 2500, 5000, 7500 in both
GSM8k and MATH; at steps 1000, 2500, 4000, 5000, 12500 in Sudoku; and at 1000, 2500, 4000
in Countdown. We evaluate less checkpoints compared to _d1_ . On the GSM8K, we train models on


[4https://github.com/dllm-reasoning/d1](https://github.com/dllm-reasoning/d1)


23


the train split [5] and evaluate on the test split. On Countdown, we train on the 3-number subset of the
dataset [6] from TinyZero (Pan et al., 2025), and evaluate on 256 synthetic 3-number questions provided
by Zhao et al. (2025). On Sudoku we use the 4×4 dataset [7] generated by Arel (2025). We train on 1M
unique puzzles and evaluate on 256 synthetic ones provided by Zhao et al. (2025). On MATH500, we
train models on the train split [8] .


To train _wd1_ ++ for evaluating on MATH500, we use dataset provided by (He et al., 2025), which is
subsampled from OpenR1 dataset Face (2025). To evaluate on GSM8k, we leverage its train split
to conduct _wd1_ ++ training. Notably, we leverage a more effective system prompt and Math-Verify
(Kydlíˇcek) to parse the answers for full-parameter fine-tuning of _wd1_, _wd1_ ++ and MDPO.


B.2 REWARD FUNCTION


To train _wd1_ and reproduce _d1_, we use the reward function defined in (Zhao et al., 2025). For
completion, we provide the details as following.


**GSM8K.** Following the Unsloth reward setup [9], we apply five addtive components: XML Structure
Reward: +0.125 per correct tag; small penalties for extra content post-tags. Soft Format Reward:
+0.5 for matching the pattern <reasoning>...</reasoning><answer>...</answer> _._
Strict Format Reward: +0.5 for exact formatting with correct line breaks. Integer Answer Reward:
+0.5 if the answer is a valid integer. Correctness Reward: +2.0 if the answer matches ground truth.


**Countdown.** We include three cases: +1.0 if the expression reaches the target using the exact
numbers. +0.1 if numbers are correct but target is missed. 0 otherwise.


**Sudoku.** The reward is the fraction of correctly filled empty cells, focusing on solving rather than
copying.


**MATH500.** We include two additive subrewards. Format Reward is +1.00 for _<_ answer _>_ with
\boxed inside; +0.75 for _<_ answer _>_ without \boxed; +0.50 for \boxed only. +0.25 for neither.
Correctness Reward: +2.0 if the correct answer is in \boxed{}.


To train _wd1_ ++, we leverage Math-Verify (Kydlíˇcek), constructing a simple verifier reward function
to evalaute on GSM8K and MATH500.


B.3 SAMPLING FROM GEOMETRIC MIXTURE


Although the sampling strategy eliminates the need to approximate the reference policy’s likelihood,
it incurs computational overhead, as generating a full completion requires multiple forward passes
through the dLLM—compared to a single pass for likelihood estimation. An alternative is to sample
from _π_ old and shift the advantage to _A_ [ˆ] _i_ = _A_ _[π]_ [old] ( _q, oi_ ) + _β_ log _π_ ˆref _/_ ( _λ_ + _β_ ), which reintroduces the
need for reference policy likelihood approximation. However, policy ratio has been removed, and the
reference model can be reused when conducting multiple gradient updates with the same batch of
rollouts (off-policy). The increased computational burden is slight.


B.4 HYPERPARAMETERS


We provide the hyperparameters of SFT in Table 5 and for _wd1_ in Table 6.


**bacth_size** **max_length** **learning_rate** **grad_accum_steps**


**Value** 1 4096 1e-5 4


Table 5: Hyperparameters of SFT in _d1_ reproduction.


[5https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
[6https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
[7https://github.com/Black-Phoenix/4x4-Sudoku-Dataset](https://github.com/Black-Phoenix/4x4-Sudoku-Dataset)
[8https://huggingface.co/datasets/ankner/math-500](https://huggingface.co/datasets/ankner/math-500)
[9https://unsloth.ai/blog/r1-reasoning](https://unsloth.ai/blog/r1-reasoning)


24


**Parameter** _**wd1**_ _**d1**_


**Model and Precision**
use_peft true true
torch_dtype bfloat16 bfloat16
load_in_4bit true true
attn_implementation flash_attention_2 flash_attention_2
lora_r 128 128
lora_alpha 64 64
lora_dropout 0.05 0.05
peft_task_type CAUSAL_LM CAUSAL_LM


**Training Configuration**
seed 42 42
bf16 true true
sync_ref_model True True
ref_model_sync_steps 64 64
adam_beta1 0.9 0.9
adam_beta2 0.99 0.99
weight_decay 0.1 0.1
_ψ_ (Equation (9)) 1.0
max_grad_norm 0.2 0.2
warmup_ratio 0.0001 0.0001
learning_rate 3e-6 3e-6
lr_scheduler_type constant_with_warmup constant_with_warmup


**Batching and Evaluation**
per_device_train_batch_size 6 6
per_device_eval_batch_size 1 1
gradient_accumulation_steps 2 2


**RL**
num_generations 6 6
max_completion_length 256 256
max_prompt_length 200 200
block_length 32 32
diffusion_steps 128 128
generation_batch_size 6 6
remasking low_confidence low_confidence
random_masking True True
p_mask_prompt 0.15 0.15
beta 0.00 0.04
epsilon - 0.5
num_iterations 12 12


Table 6: Comparison of hyperparameters between _wd1_ and _d1_ .


B.5 TRAINING COST ESTIMATION


For the runtime measurements reported in Table 2, we set _µ_ = 8 and train for a total of 6 global
steps, corresponding to 48 gradient update steps. We use a batch size of 4 and the rest of the
hyperparameters are the same as in Table 6. To estimate the number of function evaluations (NFEs)
involved in computing likelihood approximations, we count only the forward passes, as the number
of backward passes remains consistent across methods. The additional NFEs observed in the _d1_
model arise from evaluating the likelihood under both the old and reference models, which are used
for regularization. These extra evaluations are required only when new samples are drawn, as their
outputs can be cached and reused across all gradient updates for _µ_ . We additionally report the number
of floating-point operations (FLOPs) per global training step, measured using the Flops Profiler from
Rasley et al. (2020).


B.6 COMPUTING RESOURCES


For both _wd1_ and _d1_, RL training is conducted on four NVIDIA A100 GPUs (80GB), and SFT
is performed on four A6000 GPUs (48GB). For _wd1_ ++ and MDPO, RL training is conducted on
8 _×_ A800 (80GB).


25


C ADDITIONAL EXPERIMENTS


We additionally report results for comparison to the results of the baseline _d1_ reported in the paper
(Zhao et al., 2025). As shown in Table 7, our method _wd1_ evaluated and selected from less
checkpoints, can outperform _d1_ with a large margin in Sudoku and Countdown, achieving comparable
performance in math problem-solving tasks.


C.1 SUMMARY OF _wd1_ RESULTS


Table 7: Test accuracy across different tasks. Our method demonstrates higher accuracy, especially
significant in Sudoku and Countdown. The shaded area indicates where our method outperforms.


**Sudoku** **Countdown** **GSM8K** **MATH500**
**Model**

256 512 256 512 256 512 256 512


LLaDA-8B-Instruct 6.7 5.5 19.5 16.0 76.7 78.2 32.4 36.2
+ diffu-GRPO _(reported)_ 12.9 11.0 31.3 37.1 79.8 81.9 37.2 39.2
+ diffu-GRPO _(reproduced)_ 16.1 11.7 27.0 34.0 80.7 79.1 34.4 39.0
_d1 (reported)_ 16.7 9.5 32.0 42.2 **81.1** 82.1 **38.6** **40.2**
_d1 (reproduced)_ 17.6 16.2 25.8 35.2 78.2 82.0 34.4 38.0


_**wd1**_ **76.4** **62.8** **51.2** **46.1** 80.8 **82.3** 34.4 39.0


Table 8: **Training cost.** The training steps to obtained the best post-trained model of three methods are
20, 150, and 7500. To compute the total rollouts, we need to compute the average rollouts in a single
training step. Gradient steps per rollout batch represents the number of gradient descent conducted
with a single batch of rollouts. In other words, 1 represents it is a pure on-policy RL training, and for
any value _>_ 12, off-policy RL is executed. Total Batch Size is computed by multiplying per-device
batch size, gradient accumulation and the number of gpus. Therefore, the average number of rollouts
used for single step gradient descent should be computed by total batch size divided by gradient steps
per rollout batch.


**Hyperparameter** _**wd1++**_ **MDPO** _**d1**_


Training step of the best checkpoint 20 150 7500


Training Steps per Rollout batch 1 1 12


Per-Device Batch Size 4 1 6
Gradient Accumulation 2 16 2
GPUs used for training 8 8 4
Total Batch Size 64 128 48


Avg. Rollouts per Step 64 128 4
Total Rollouts 1280 19200 30000


We additionally provide reward dynamics in comparison to _wd1_ -SFT in training. In Sudoku and
Countdown, directly training with _wd1_ without SFT shows significantly more efficient and stable
learning process. In GSM8k and MATH500, the difference is negligible.


C.2 ADDITIONAL ABLATION STUDY


We provide additional ablation study on the combined weight to confirm our analysis that the positive
and negative samples terms in the loss function should be assigned equal proportion, due to the side
case of a batch of all-negative generated responses (see the paragraph below Equation (9)). Assigning
equal proportions to positive and negative weights is not arbitrary but rather the most robust design.
This can be understood through two critical failure modes that arise from imbalanced proportions:


26


0.4


0.3


0.3


0.2


0.1


0.0


0 2000 4000 6000
Steps


0 2000 4000 6000
Steps


0.1


0 2000 4000 6000
Steps


1.2


-0.3


1.4


0.8

0 2000 4000 6000
Steps


Figure 3: Reward Dynamics. _wd1_ without SFT demonstrates better rewards in Sudoku and Countdown.


Table 9: Ablation on weight _λ_ to combine positive _w_ [+] and negative weights _w_ _[−]_ in _wd1_ on Sudoku.
Specifically, the final weight assigned to log-likelihood is computed as _−λw_ [+] + (1 _−_ _λ_ ) _w_ _[−]_ .


**Combined Weight** **Accuracy** **Effective Tokens**


0.5 25.63% 326.97
0.4 11.77% 240.04
0.6 14.11% 220.13


    - When positive weight has larger proportion: In scenarios where all sampled completions
have uniformly low rewards, a larger proportion of positive weights would paradoxically
increase the log-likelihood of negative samples during wd1 optimization, which is clearly
undesirable and contradicts the learning objective.


    - When negative weight has larger proportion: Conversely, when all generated completions
achieve uniformly high rewards, an insufficient proportion of positive weights would result
in unlearning high-quality samples.


To empirically validate this analysis, we conducted experiments on the Sudoku dataset with varying
mixing proportions. The results, presented in the table below, confirm our theoretical predictions.


Training Steps


2.0


1.5


1.0


1.5


1.4


1.3


1.2


|Col1|Col2|
|---|---|
||~~= 0.1~~|
||<br>= 1.0<br>= 5.0<br>|
||~~= 10.0~~|


Training Steps


Figure 4: **Left:** Ablation study on _ψ_ in the weights (Equation (9)) on GSM8K. **Right:** _wd1_ training
on MATH with a random seed different from the seed used in our main experiments. The abrupt
decrease of the rewards in the early training (see Figure 3 MATH500) disappears.


In all benchmark evaluations, we fix the hyperparameter _ψ_ = 1, which controls the scale of the
exponential weighting in _wd1_ . To validate this choice, we provide an ablation study on the coefficient
_ψ_ = 1 in the exponential weight of _wd1_ (Equation 9) below. Larger values _ψ_ leads to more extreme
weight assigned to the samples. According to Figure 4, the training of applying different _ψ_ converges
to similar rewards if _ψ_ is small. Overly large value (e.g. 10) can cause performance drop, implying
that extreme weight assignment is detrimental.


27


160


170


240


260


240


210


|Col1|Col2|Col3|
|---|---|---|
|||wd1<br>d1|


0 2000 4000 6000
Steps


70


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||


|Col1|Col2|Col3|
|---|---|---|
||||


0 2000 4000 6000
Steps


0 2000 4000 6000
Steps


0 2000 4000 6000
Steps


100


220


Figure 5: Completion lengths dynamics of _wd1_ and _d1_ . In math problem-solving tasks (GSM8K and
MATH500), our method demonstrates smaller completion lengths and better token efficiency.


C.3 CODING BENCHMARKS


We conduct 200 steps of _wd1_ training on AceCode-87K (Zeng et al., 2025) following the implementation of Open-R1 (Hugging Face, 2025). Our method achieves consistent improvements over the
base model.


Table 10: Comparative performance of _wd1_ improvements compared to base model LLaDA-8BInstruct. We present the results of _wd1_ fine-tuned with AceCode dataset (Zeng et al., 2025).


Task Gen Length Steps Block Size _wd1_ LLaDA


HumanEval 256 128 32 **34.76** **(+3.66)** 31.10
HumanEval 256 256 32 **39.02** **(+1.82)** 37.20
HumanEval 512 512 32 **36.59** **(+0.61)** 35.98
MBPP 128 128 32 **39.2** **(+2.40)** 36.8
MBPP 256 256 32 **36.6** **(+1.00)** 35.6
MBPP 512 512 32 **36.8** **(+0.40)** 36.4


C.4 TRAINING DYNAMICS


Figure 3 presents the reward dynamics over gradient steps during training. _wd1_ exhibits a notably
faster reward increase compared to _d1_, highlighting its superior sample efficiency–effectively leveraging the reward signal to accelerate policy optimization. In addition, Figure 5 shows the average
length of generated completions during training. On math reasoning benchmarks such as GSM8K and
MATH500, _wd1_ converges to shorter output sequences than _d1_, suggesting improved token efficiency
while maintaining or improving performance.


D LIMITATIONS


Similar to other RL-based approaches, _wd1_ may lose effectiveness when all generations within a
sampled group receive identical rewards. This situation can occur under several conditions—for
example, when the training dataset is either too simple or too challenging for the base model.
Nonetheless, such cases can be mitigated through careful reward design and the incorporation of
curriculum learning strategies.


An additional limitation of this work is that the current _wd1_ framework is restricted to text-based
reasoning. Extending it to multimodal reasoning or unified diffusion-based models (e.g., (Yang et al.,
2025)) represents a valuable direction for future research.


A final limitation concerns the likelihood approximation used in _wd1_ . Our approach relies on the
_d1_ -based approximation, which is computationally efficient but introduces bias. Although some prior
works employ ELBO-based estimators (e.g., DCE), they require additional computational overhead
(Zhao et al., 2025) often exhibit high variance, as demonstrated in Figure 1. This trade-off highlights
an important area for further exploration.


28


D.1 ADDITIONAL ANALYSIS ON UNLEARNING


We provide extended demonstrations for Remark Remark 2, focusing specifically on the theoretical
insights underlying the interpretation of the negative-sample reinforcement term in _wd1_ as a form of
data unlearning. Under the DCE likelihood approximation, the negative-sample reinforcement term
in _wd1_ becomes


   -    
E _o∼p_ _[′]_ 0 [(] _[·]_ [)] _w_ _[−]_ ( _q, o_ ) _·_ log _πθ_ ( _o|q_ ) = E _o∼p_ _[′]_ 0 [(] _[·]_ [)]


- exp( _−A_ ( _x_ 0)) _·_ log _πθ_ ( _o|q_ ) (67)


exp  - _−_ _A_ ( _x_ 0)� _·_ E _t∼_ [0 _,T_ ] _,p_ _[′]_ _t|_ 0 [(] _[x][t][|][x]_ [0][)]  -  


_t_ [log] _[ p][θ]_ [(] _[x]_ 0 _[i]_ _[|][x]_ [UM] _t_ )�


(68)


=E _x_ 0 _∼p_ _[′]_ 0 [(] _[·]_ [)]


 - _−_ [1]

_t_

_x_ _[i]_ _t_ [=[][mask][]]


- �� _L_ DCE


 E _t∼_ [0 _,T_ ] _,p′t|_ 0 [(] _[x][t][|][x]_ 0 _[−]_ [)] - 


0 _[|][x]_ _t_ [UM] )�
_t_ [log] _[ p][θ]_ [(] _[x][−][,i]_


_,_ (69)


=E _x−_ 0 _[∼][p]_ [data]


 - _−_ [1]

_t_

_x_ _[i]_ _t_ [=[][mask][]]


          - ��          _L_ DCE _⇔_ ELBO


exp( _−A_ ( _x_ _[−]_ 0 [))]
where _p_ data( _x_ _[−]_ 0 [) =] _[ p]_ 0 _[′]_ [(] _[x][−]_ 0 [)] - _x_ _[−]_ 0 _p_ _[′]_ 0 [(] _[x][−]_ 0 [) exp(] _[−][A]_ [(] _[x]_ 0 _[−]_ [))] [.] [Equation (69) holds by simply applying impor-]

tance sampling.


Since DCE is equivalent to the evidence lower bound (ELBO) of masked discrete diffusion models,
we draw an analogy between the final objective in Equation (69) and data unlearning in diffusion
models (Alberti et al., 2025). Equation (69) can be viewed as a direct masked discrete–diffusion
extension of NegGrad (Golatkar et al., 2020), which aims to minimize the evidence lower bound of
the log-likelihood on samples with lower advantage.


29