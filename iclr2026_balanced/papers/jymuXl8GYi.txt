# REINFORCING DIFFUSION MODELS BY DIRECT GROUP PREFERENCE OPTIMIZATION


**Yihong Luo** [1] **Tianyang Hu** [2] **Jing Tang** [3] _[,]_ [1] _[∗]_

1 HKUST 2 CUHK (SZ) 3 HKUST (GZ)


1.30


1.0


0.9


0.8


0.7


0.6


|DGPO (|0.97)|Col3|Flow-G|Col5|RPO (0.95|
|---|---|---|---|---|---|
||DGPO is ~30×|Faster than Fl|<br>   ow-GRPO|<br>   ow-GRPO||
||||G<br>|G<br>|PT-4o (0.84<br>|
||||Flow<br>DGP|Flow<br>DGP|-GRPO<br> (Ours)|
||||Flow<br>DGP|Flow<br>DGP||
||||SD3.5-M Bas|SD3.5-M Bas|eline (0.63|
|||||||
|||||||


Training Time (GPU Hours)


GenEval Performance


1.25


1.20


1.15


1.10


1.05


1.00


0.95


strong performance on other out-of-domain metrics (Right Figure).


ABSTRACT


1 INTRODUCTION

|Image Quality Preference Score|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|
|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|~~SD3.5-M~~|||||||||||
|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>|Flow-GRPO<br>||||||||||||
|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)|DGPO (Ours)||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|||||||||||||||||||||
|Aest<br>    ng<br>t is<br>lati<br>   ua<br>lar<br>    ler<br>    in<br>de<br>     e p<br>    hm<br>ect<br>    wit<br>, u<br>Ex<br>ate<br>    ou<br> uo|h<br>    c<br> <br>v<br>   g<br>, <br>    s<br>    e<br>l<br>     r<br> <br>l<br>    h<br>n<br>te<br>-<br>    t<br>|et<br> <br>    o<br> a<br>e <br>   e<br> <br>    a<br>    f<br>-a<br>     op<br>    th<br>y <br>    in<br>l<br>n<br>o<br>    -<br> -|ic <br>O<br>    m<br>c<br> <br>    M<br>G<br>    r<br>    fc<br>g<br>     o<br>    a<br> f<br> <br>o<br>s<br>f-<br>    of<br> Y|D<br>ut<br>    p<br>hi<br>Pr<br> <br>R<br>    e<br>    i<br>n<br>     s<br>    t<br>ro<br>    gr<br>ck<br>iv<br>t<br>    -<br> i|eQ<br>-o<br>    a<br>e<br>e<br>    od<br>P<br>     b<br>    en<br>o<br>     e<br>     d<br>m<br>    o<br>i<br>e<br>he<br>    d<br> h|A <br>f-<br>    re<br>ve<br>fe<br>    e<br>O<br>     as<br>    t<br>st<br>      D<br>     is<br> <br>    u<br>n<br> r<br>-<br>    o<br> o|I<br>D<br>    d<br>d<br>r<br>    l<br> <br>     e<br> <br>ic<br>      i<br>     p<br> g<br>    p<br>g <br> e<br>a<br>    m<br> n|m<br>o<br>     t<br> <br>e<br>    s,<br>d<br>     d<br>    S<br> <br>      re<br>     e<br>r<br>    s.<br> t<br> s<br>rt <br>    ai<br> g|gR<br>ma<br>     o<br>w<br>nc<br>    a<br>em<br>     o<br>    D<br>G<br>      c<br>     ns<br>ou<br>  <br>he<br> ul<br> m<br>    n<br> /|w<br>i<br>     F<br>h<br>e <br>    d<br><br>     n<br>    E-<br>a<br>      t<br>     e<br>p<br>Th<br> <br> ts<br><br>     r<br> D|d<br>n<br>     l<br>il<br> <br>    a<br>an<br>      d<br>    b<br>u<br>      G<br>     s<br>-<br>i<br>u<br> s<br>et<br>     e<br> G|Pic<br> M<br>     o<br>e <br>O<br>    pt<br>d<br>      e<br>    a<br>ss<br>      r<br>     w<br>le<br>s<br>se<br> h<br>h<br>     w<br> P|k<br> et<br>     w<br> m<br>pt<br>    in<br>s <br>      te<br>    se<br>ia<br>      ou<br>     i<br>v<br> d<br> <br> o<br>o<br>     a<br> O|Sc<br> ri<br>     -<br>a<br>i-<br>    g<br> a<br>      r-<br>    d<br>n<br>      p<br>     th<br>el<br> e-<br>of<br> w<br>ds<br>     rd<br> .|. <br> cs<br>     G<br>i<br><br> <br><br> <br> <br><br> <br> <br><br> <br><br> <br><br>|n<br> <br>     R<br>nt|iR<br>     P<br>ai|w<br>     O<br>n|d<br>      o<br>in|


Reinforcement Learning (RL) has become a cornerstone for the post-training of Large Language
Models (LLMs), significantly enhancing their capabilities (Ziegler et al., 2019; Ouyang et al., 2022;
Bai et al., 2022). In particular, methods like Group Relative Policy Optimization (GRPO) (Shao
et al., 2024) have demonstrated remarkable success in substantially improving the complex reasoning abilities of LLMs (DeepSeek-AI, 2025). However, progress in applying RL for post-training
diffusion models has lagged considerably behind that of language models, leaving a significant gap
in methods for aligning generative models with human preferences and complex quality metrics.


A central obstacle is the mismatch between GRPO’s policy gradient-based framework (short for
policy framework in the following) and the mechanics of diffusion generation. GRPO requires access to a stochastic policy to enable effective training and exploration. This requirement is naturally


_∗_ Corresponding Author: Jing Tang


1


A photo of a
sandwich
**left of** a bear


A photo of a girl
with red dress **right**
**of** a green cat


A photo of a blue
burger **on the**
**head** of a red dog


A photo of
**five** cats


A photo of
**four** cakes


SD 3.5M


SD 3.5M
w/ FlowGRPO


**SD 3.5M**
**w/ DGPO**
**(Ours)**


Figure 2: Qualitative comparisons of DGPO against competing methods. It can be seen that our
proposed DGPO not only accurately follows the instructions, but also keeps a strong visual quality.
All images are generated by the same initial noise.


met by LLMs, which inherently output a probability distribution over a vocabulary. In contrast,
diffusion models predominantly rely on deterministic ODE-based samplers to strike a better balance
between sample quality and cost (Song et al., 2020; Luo et al., 2025c), and thus do not naturally
provide a stochastic policy. To bridge this gap, prior work has resorted to a forced adaptation: using
stochastic SDE-based sampling to induce a conditional Gaussian policy suitable for GRPO’s policy framework (Liu et al., 2025; Xue et al., 2025). This workaround, however, introduces severe
negative consequences: (1) SDE-based rollouts are less efficient than their ODE counterparts and
produce lower-quality samples under a fixed computational budget (Lu et al., 2022a;b; Bao et al.,
2022; Song et al., 2020); (2) The policy’s stochasticity comes from model-agnostic Gaussian noise,
which provides a weak learning signal and results in slow convergence; and (3) Training is performed over the entire sampling trajectory, making each iteration computationally expensive and
time-consuming.


We argue that the practical success of GRPO stems less from its policy-gradient formulation, and
more from its ability to utilize fine-grained relative preference information within group. Based on
the insight, an ideal RL method for diffusion models should be capable of leveraging this powerful
group-level information while dispensing with the need for a stochastic policy and its associated
negative effects. To this end, we introduce **D** irect **G** roup **P** reference **O** ptimization ( **DGPO** ), a new
online RL method tailored to diffusion models. DGPO circumvents the policy-gradient framework
entirely, instead optimizing the model by directly learning from the group-level preference between
a set of “good” samples and a set of “bad” samples. Concretely, for each prompt, we generate
_G_ samples using efficient ODE-based rollouts, partition them into positive and negative groups,
and directly optimize the model by maximizing the likelihood of these group-wise preferences.
Conceptually, DGPO can be understood as a natural extension of Direct Preference Optimization
(DPO) (Wallace et al., 2024) that incorporates group-wise information, and as a diffusion-native
re-imagination of GRPO.


This proposed methodology allows us to bypass the dependency on a stochastic policy, which yields
several benefits: (1) **Efficient** **Sampling** **and** **Learning** : by using high-fidelity ODE samplers,
DGPO learns from higher-quality rollouts, leading to more effective learning. (2) **Efficient** **Con-**
**vergence** : optimization is directly guided by group-level preferences rather than inefficient modelagnostic random exploration, leading to faster convergence. (3) **Efficient** **Training** : our approach


2


avoids training on the entire sampling trajectory, notably reducing the computational cost of each
training iteration. Together, these advantages establish DGPO as a highly efficient and powerful
online RL algorithm for diffusion models. Our extensive experiments show that DGPO achieves
around 20× faster training than prior state-of-the-art Flow-GRPO (Liu et al., 2025), while delivering
superior performance on both in-domain and out-of-domain metrics. Most notably, on the challenging GenEval benchmark (Ghosh et al., 2023), DGPO trains nearly 30× faster than Flow-GRPO and
boosts the base model’s performance from 63% to 97% (Fig. 1). These compelling results demonstrate DGPO’s potential as a powerful technique for aligning diffusion models.


2 PRELIMINARIES


**Diffusion Models (DMs)** DMs (Sohl-Dickstein et al., 2015; Ho et al., 2020) define a forward diffusion mechanism that progressively introduces Gaussian noise to input data **x** across _T_ sequential
timesteps. The forward process follows the distribution _q_ ( **x** _t|_ **x** ) ≜ _N_ ( **x** _t_ ; _αt_ **x** _, σt_ [2] **[I]** [)][, where the hy-]
perparameters _αt_ and _σt_ control the noise scheduling strategy. At each timestep, noisy samples are
obtained by **x** _t_ = _αt_ **x** + _σtϵ_, where _ϵ_ _∼N_ ( **0** _,_ **I** ). The parameterized reversed diffusion process is
defined by: _pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ) ≜ _N_ ( **x** _t−_ 1; _µθ_ ( **x** _t, t_ ) _, ηt_ [2] **[I]** [)][.] [The model’s neural network] _[ f][θ]_ [is learned by]
denoising E **x** _,ϵ,tλt||fθ_ ( **x** _t, t_ ) _−_ **x** _||_ [2] 2 [.] [We note that the flow matching and DMs are equivalent in the]
context of diffusing by Gaussian noise (Gao et al., 2025).


**Reward** **Modeling** Given ranked pairs generated from certain conditioning **x** _[w]_ 0 _[≻]_ **[x]** 0 _[l]_ _[|]_ **[c]** [,] [where]
**x** _[w]_ 0 [and] **[ x]** 0 _[l]_ [denote the “better” and “worse” samples.] [The Bradley-Terry (BT) model formulates the]
preferences as:
_p_ BT( **x** _[w]_ 0 _[≻]_ **[x]** 0 _[l]_ _[|]_ **[c]** [) =] _[ σ]_ [(] _[r]_ [(] **[c]** _[,]_ **[ x]** 0 _[w]_ [)] _[ −]_ _[r]_ [(] **[c]** _[,]_ **[ x]** 0 _[l]_ [))] (1)


where _σ_ ( _·_ ) denotes the sigmoid function. A network _rϕ_ that models reward can be trained by
maximum likelihood as follows:


_L_ BT( _ϕ_ ) = _−_ E **c** _,_ **x** _w_ 0 _[,]_ **[x]** 0 _[l]_ �log _σ_       - _rϕ_ ( **c** _,_ **x** _[w]_ 0 [)] _[ −]_ _[r][ϕ]_ [(] **[c]** _[,]_ **[ x]** 0 _[l]_ [)] �� (2)


**RLHF** RLHF typically aims to optimize a conditional density _pθ_ ( **x** 0 _|_ **c** ) to maximize a underlying
reward _r_ ( **c** _,_ **x** 0) while staying close to a reference distribution _p_ ref via KL regularization, i.e.,


max _pθ_ [E] **[c]** _[,]_ **[x]** [0] _[∼][p][θ]_ [(] **[x]** [0] _[|]_ **[c]** [)][ [] _[r]_ [(] **[c]** _[,]_ **[ x]** [0][)]] _[ −]_ _[β]_ [KL [] _[p][θ]_ [(] **[x]** [0] _[|]_ **[c]** [)] _[∥][p]_ [ref][(] **[x]** [0] _[|]_ **[c]** [)]] (3)


where the hyperparameter _β_ controls strength of regularization.


**GRPO Objective (Shao et al., 2024)** The RLHF objective in Eq. (3) can be optimized by policybased learning (we omit the KL term and clip term hereinafter for brevity):


max _pθ_ [E][(] **[x]** [0] _[,]_ **[x]** [1] _[,][···][,]_ **[x]** _[T]_ [ )] _[∼][p][θ]_ [old] [(] _[·|]_ **[c]** [)]


_T_


_k_ =0


_pθ_ ( **x** _k_ +1 _|_ **x** _k,_ **c** )
(4)
_pθ_ old( **x** _k_ +1 _|_ **x** _k,_ **c** ) _[A]_ [(] **[x]** _[k]_ [+1][)] _[,]_


where _A_ ( **x** _k_ +1) denotes the advantage of **x** _k_ +1, which can be directly computed by reward _r_ ( **x** _k_ +1),
or introduce additional value model for reducing variance.


The GRPO proposes to sample a group of outputs for each prompt **c** from the old policy,
then compute the advantage of each sample by normalization among groups, i.e., _Ai_ = ( _ri_ _−_
mean( _{r_ 1 _, r_ 2 _, · · ·_ _, rG}_ )) _/_ std( _{r_ 1 _, r_ 2 _, · · ·_ _, rG}_ ). The policy learning requires that the transition between **x** _k_ and **x** _k_ +1 follows a stochastic distribution. To meet this requirement for a stochastic
policy, recent works (Liu et al., 2025; Xue et al., 2025) employ a stochastic SDE for sampling,
rather than the more efficient deterministic ODE. However, the SDE itself is less effective in sampling high-quality samples with insufficient steps. Besides, the policy-based method requires performing training on the whole trajectory, which further leads to slow training. More importantly,
unlike LLMs, which directly output distribution, _the stochasticity in DM’s policy comes from model-_
_agnostic Gaussian Noise. This makes the stochastic exploration rely on the model-agnostic Gaussian_
_noise, which is extremely inefficient in high-dimensional space._


3


**DPO Objective (Rafailov et al., 2024)** The unique global optimal density _p_ _[∗]_ _θ_ [of the RLHF objec-]
tive (Eq. (3)) is given by:

_p_ _[∗]_ _θ_ [(] **[x]** [0] _[|]_ **[c]** [) =] _[ p]_ [ref][(] **[x]** [0] _[|]_ **[c]** [) exp (] _[r]_ [(] **[c]** _[,]_ **[ x]** [0][)] _[/β]_ [)] _[ /Z]_ [(] **[c]** [)] (5)

where _Z_ ( **c** ) = [�] **x** 0 _[p]_ [ref][(] **[x]** [0] _[|]_ **[c]** [) exp (] _[r]_ [(] **[c]** _[,]_ **[ x]** [0][)] _[/β]_ [)][ is a intractable partition function.] [We can compute]

the reward function as follows:

_θ_ [(] **[x]** [0] _[|]_ **[c]** [)]
_r_ ( **c** _,_ **x** 0) = _β_ log _[p][∗]_ (6)
_p_ ref( **x** 0 _|_ **c** ) [+] _[ β]_ [ log] _[ Z]_ [(] **[c]** [)]


After obtaining the parameterization of the reward function, the DPO optimizes the models by the
reward learning objective in Eq. (2):


_L_ DPO( _θ_ )= _−_ E **c** _,_ **x** _w_ 0 _[,]_ **[x]** 0 _[l]_


- 0 _[|]_ **[c]** [)] 0 _[|]_ **[c]** [)]
log _σ_ _β_ log _[p][θ]_ [(] **[x]** _[w]_ _[p][θ]_ [(] **[x]** _[l]_
_p_ ref( **x** _[w]_ 0 _[|]_ **[c]** [)] _[−]_ _[β]_ [ log] _p_ ref( **x** _[l]_ 0 _[|]_ **[c]** [)]


��
(7)


Diffusion DPO (Wallace et al., 2024) has adapted DPO to Diffusion models by defining reward over
the diffusion paths **x** 0: _T_, which does not require a stochastic policy. _However,_ _it_ _strictly_ _relies_ _on_
_pairwise samples for optimization due to the intrinsic restriction of the intractable partition Z_ ( **c** ) _,_
_preventing the use of the fine-grained preference information of each sample._


3 METHOD


We believe that the key to GRPO’s success lies in its ability to utilize fine-grained relative preference
information within groups. However, existing GRPO-style methods (Liu et al., 2025; Xue et al.,
2025) require using an inefficient stochastic policy. Although existing DPO-style methods (Wallace
et al., 2024) provide a framework without the need for a stochastic policy, they require performing
training on pairwise samples to eliminate the intractable partition _Z_ ( **c** ). To this end, we propose
Direct Group Preference Optimization (DGPO), which eliminates the inefficient stochastic policy
and allows us to directly optimize inter-group preferences without concerning ourselves with an
intractable partition, leveraging fine-grained reward information to significantly improve training
efficiency. The pseudo code of DGPO is summarized in Algorithm 1.


**Problem** **Setup** Let _p_ ref denote a pre-trained reference diffusion model with parameter _θ_ ref. We
have a dataset of conditions _Dc_ = _{_ **c** _[N]_ _i_ =1 _[}]_ [ and a reward function] _[ r][ϕ]_ [(] _[·][,][ ·]_ [) :] _[ X × C]_ _[→]_ [R][ that evaluates]
the quality of generated samples **x** _∈X_ given condition **c** _∈C_ . Our goal is to enhance a diffusion
model _pθ_, initialized from _p_ ref, according to the reward signal. At each training iteration, we use an
online model _pθ−_ to generate a group of samples conditioned on _c_ _∈Dc_, where _θ_ _[−]_ can be set as
the current parameters _θ_ or an exponential moving average (EMA) version of previous _θ_ ’s. These
generated samples form a dataset _D_ = _{_ ( _Gi_ = _{_ **x** _[G]_ _k_ =1 _[}][,]_ **[ c]** _[i]_ [)] _[|]_ **[x]** _[k]_ _[∼]_ _[p][θ][−]_ [(] _[·|]_ **[c]** _[i]_ [)] _[}]_ [,] [which] [are] [then]
evaluated by the reward function to provide reward signals for splitting positive or negative groups.


3.1 DIRECT GROUP PREFERENCE OPTIMIZATION


In order to leverage relative information within groups, we propose directly learn the group-level
preferences using the Bradley-Terry model via maximum likelihood:

max E( _G_ + _,G−,c_ ) _∼D_ log _p_ ( _G_ [+] _≻G_ _[−]_ _|_ **c** ) = E( _G_ + _,G−,c_ ) _∼D_ log _σ_ ( _Rθ_ ( _G_ [+] _|_ **c** ) _−_ _Rθ_ ( _G_ _[−]_ _|_ **c** )) (8)
_θ_


where _G_ [+] and _G_ _[−]_ represent positive and negative groups respectively, with _G_ [+] _∪G_ _[−]_ = _G_, and
_G_ = _{_ **x** [1] 0 _[,][ · · ·]_ _[,]_ **[ x]** _[G]_ 0 _[}]_ [ being the complete group of samples for conditioning] **[ c]** [.] [Intuitively, the objec-]
tive in Eq. (8) can leverage fine-grained preference information of each sample within groups with
appropriate parameterization.


Therefore, we propose parameterizing the group-level reward as a weighted sum of rewards
_rθ_ ( **c** _,_ **x** 0) for each sample within the group:


_Rθ_ ( _G|_ **c** ) =            - _w_ ( **x** 0) _· rθ_ ( **c** _,_ **x** 0) _,_ (9)


**x** 0 _∈G_


where _ω_ controls the “importance level” of the sample within the group. The parameterization of
group-level reward can reflect the fine-grained information of each sample. And the reward of single


4


**Algorithm 1** Direct Group Preference Optimization (DGPO)
**Require:** Diffusion model _fθ_, Reference model _f_ ref, Reward model _rϕ_, Group size _G_, Hyperparameter _β_, Minimum training timestep _t_ min, Learning rate _η_, Iterations _N_, EMA decay _µ_ (optional).
**Ensure:** Optimized model _fθ_ .

1: **for** _n ←_ 1 **to** _N_ **do**
2: # Sample conditioning and generate group
3: Sample conditioning **c** _∼Dc_
4: Generate group _G_ = _{_ **x** [1] 0 _[, ...,]_ **[ x]** 0 _[G][}]_ [ by sampling from] _[ p]_ _θ_ _[−]_ [(] _[·|]_ **[c]** [)]
5: # Compute advantages
6: _{ri} ←{rϕ_ ( **c** _,_ **x** _[i]_ 0 [)] _[}][G]_ _i_ =1
7: _Ai_ _←_ _[r][i][−]_ std [mean] ( _{r_ [(] _j_ _[{]_ _}_ _[r]_ ) _[j]_ _[}]_ [)] for all _i_

8: # Partition into positive and negative groups
9: _G_ [+] _←{_ **x** _[i]_ 0 [:] _[ A][i]_ _[>]_ [ 0] _[}]_ [ and] _[ G][−]_ _[←{]_ **[x]** _[i]_ 0 [:] _[ A][i]_ _[≤]_ [0] _[}]_
10: # Compute DGPO loss
11: Sample _t ∼U_ [ _t_ min _, T_ ], _ϵ ∼N_ (0 _, I_ )
12: **x** _[i]_ _t_ _[←]_ _[α][t]_ **[x]** 0 _[i]_ [+] _[ σ][t][ϵ]_ [ for all] _[ i]_
13: Compute _L_ DGPO by Eq. (17)
14: Update _θ_ _←_ _θ −_ _η∇θL_ DGPO
15: Update _θ_ _[−]_ _←_ _θ_ or _θ_ _[−]_ _←_ _µθ_ _[−]_ + (1 _−_ _µ_ ) _θ_
16: **end for**


sample can be parameterized by following Eq. (6) and Diffusion-DPO (Wallace et al., 2024):


_rθ_ ( **c** _,_ **x** 0) = _β_ E _pθ_ ( **x** 1: _T |_ **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]


_p_ ref( **x** 0: _T |_ **c** ) [+] _[ β]_ [ log] _[ Z]_ [(] **[c]** [)] _[,]_


(10)

_[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c** ) [+] _[ β]_ [ log] _[ Z]_ [(] **[c]** [)]


_≈_ _β_ E _q_ ( **x** 1: _T |_ **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]


where _Z_ ( **c** ) is a intractable partition function. We note that since sampling from the inversion chain
_pθ_ ( **x** 1: _T |x_ 0) is expensive, the forward diffusion _q_ ( **x** 1: _T |_ **x** 0) has been utilized as an approximation in
practice (Wallace et al., 2024).


By combining Eq. (9) and Eq. (8), we can derive the desired training objective:


E _q_ ( **x** 1: _T |_ **x** 0) _βw_ ( **x** 0)[log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c**

**x** 0 _∈G_ [+]


     _L_ ( _θ_ ) = _−_ E( _G_ + _,G−,c_ ) _∼D_ log _σ_ (


_p_ ref( **x** 0: _T |_ **c** ) [+] _[ Z]_ [(] **[c]** [)]]


E _q_ ( **x** 1: _T |_ **x** 0) _βw_ ( **x** 0) _·_ [log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c**

**x** 0 _∈G_ _[−]_


_−_ 


_p_ ref( **x** 0: _T |_ **c** ) [+] _[ Z]_ [(] **[c]** [)]])]


- _w_ ( **x** 0) _·_ log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c** )

**x** 0 _∈G_ [+]


      = _−_ E( _G_ + _,G−,c_ ) _∼D_ log _σ_ ( _β_ [E _q_ ( **x** 1: _T |_ **x** 0)


_p_ ref( **x** 0: _T |_ **c** )


E _q_ ( **x** 1: _T |_ **x** 0) _w_ ( **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c**

**x** 0 _∈G_ _[−]_


- _w_ ( **x** 0) _Z_ ( **c** ) _−_ 

**x** 0 _∈G_ [+] **x** 0 _∈G_


_−_ 


_[θ]_ [0:] _[T]_   
_p_ ref( **x** 0: _T |_ **c** ) [+]


_w_ ( **x** 0) _Z_ ( **c** )])

**x** 0 _∈G_ _[−]_


(11)


A remaining crucial challenge is that the partition function _Z_ ( **c** ) is intractable for training. We have
to carefully select an appropriate weighting _wi_ for each sample to eliminate the intractable partition
function _Z_ ( **c** ). Generally speaking, a good weighting strategy should satisfy the following:


- Larger weights correspond to better samples in _G_ [+] and worse samples in _G_ _[−]_ .

- The weights satisfy: [�] **x** 0 _∈G_ [+] _[ w]_ [(] **[x]** [0][) =][ �] **x** 0 _∈G_ _[−]_ _[w]_ [(] **[x]** [0][)][, such that][ �] **x** 0 _∈G_ [+]


**x** 0 _∈G_ [+] _[ w]_ [(] **[x]** [0][) =][ �]


**x** 0 _∈G_ _[−]_ _[w]_ [(] **[x]** [0][)][, such that][ �]


The weights satisfy: [�] **x** 0 _∈G_ [+] _[ w]_ [(] **[x]** [0][) =][ �] **x** 0 _∈G_ _[−]_ _[w]_ [(] **[x]** [0][)][, such that][ �] **x** 0 _∈G_ [+] _[ w]_ [(] **[x]** [0][)] _[Z]_ [(] **[c]** [)] _[ −]_

**x** 0 _∈G_ _[−]_ _[w]_ [(] **[x]** [0][)] _[Z]_ [(] **[c]** [)) = 0][ for eliminating the intractable] _[ Z]_ [(] **[c]** [)][.]


3.2 ADVANTAGE-BASED WEIGHT DESIGN


We propose using advantage-based weights derived from GRPO-style normalization to address the
aforementioned issues. Given a group _G_ = **x** [1] 0 _[,]_ **[ x]** [2] 0 _[, ...,]_ **[ x]** _[G]_ 0 [with corresponding rewards] _[ r]_ [1] _[, r]_ [2] _[, ..., r][G]_ [,]


5


we compute advantages:

_[−]_ [mean][(] _[{][r][j][}][G]_ _j_ =1 [)]
_A_ ( **x** _[i]_ 0 [) =] _[r][i]_ (12)
std( _{rj}_ _[G]_ _j_ =1 [)]


We then partition the group based on advantages:


_G_ [+] = _{_ **x** _[i]_ 0 [:] _[ A]_ [(] **[x]** 0 _[i]_ [)] _[ >]_ [ 0] _[}][,]_ _G_ _[−]_ = _{_ **x** _[i]_ 0 [:] _[ A]_ [(] **[x]** 0 _[i]_ [)] _[ ≤]_ [0] _[}][.]_ (13)


And we set weights as:
_w_ ( **x** 0) = _|A_ ( **x** 0) _|_ (14)

This choice ensures [�] **x** 0 _∈G_ [+] _[ w]_ [(] **[x]** [0][)] [=] [�] **x** 0 _∈G_ _[−]_ _[w]_ [(] **[x]** [0][)] [due] [to] [the] [zero-mean] [property] [of] [the] [nor-]

malized advantages. It also dynamically assigns larger weights to samples that deviate more from
the average, which enables the model to more effectively learn relative preference relationships.
More importantly, this weighting turns the objective in Eq. (11) to:


E _q_ ( **x** 1: _T |_ **x** 0) _w_ ( **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c** )

**x** 0 _∈G_ [+]


     _L_ ( _θ_ ) = _−_ E( _G_ + _,G−,c_ ) _∼D_ log _σ_ ( _β_ [


_p_ ref( **x** 0: _T |_ **c** )


_w_ ( **x** 0)E _q_ ( **x** 1: _T |_ **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c**

**x** 0 _∈G_ _[−]_


_−_ 


(15)

_[p][θ]_ [(] **[x]** [0:] _[T][ |]_ **[c]** [)]

_p_ ref( **x** 0: _T |_ **c** ) [])] _[.]_


By using Jensen’s inequality and the convexity of _−_ log _σ_, we can move the expectation outside:


_w_ ( **x** 0) _βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ _[,]_ _,_ **[ c]** **c** [)] )
**x** 0 _∈G_ [+]


      _L_ ( _θ_ ) _≤−_ E( _G_ + _,G−,c_ ) _∼D_ E _t,q_ ( **x** _t|_ **x** 0) log _σ_ (


_p_ ref( **x** _t−_ 1 _|_ **x** _t,_ **c** )


_w_ ( **x** 0) _βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ _[,]_ _,_ **[ c]** **c** [)]
**x** 0 _∈G_ _[−]_


_−_ 


(16)

_[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t][,]_ **[ c]** [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t,_ **c** ) [)] _[,]_


where _G_ = _G_ [+] _∪G_ _[−]_ . We note that to reduce the variance, the sampled noise _ϵ_ is shared among
samples within the same complete groups. By some simplification, we obtain our final training
objective of the proposed DGPO:


_L_ DGPO( _θ_ ) ≜ _−_ E( _G_ + _,G−,c_ ) _∼D_ E _t,q_ ( **x** _t|_ **x** ) log _σ_ ( _−λtβT_ ( - _w_ ( **x** )[ _L_ _[θ]_ dsm [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [)] _[ −]_ _[L][θ]_ dsm [ref] [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [)]]

**x** _∈G_ [+]

_−_    - _w_ ( **x** )[ _L_ _[θ]_ dsm [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [)] _[ −]_ _[L][θ]_ dsm [ref] [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [)]))] _[,]_

**x** _∈G_ _[−]_

(17)
where _L_ _[θ]_ dsm [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [) =] _[ ||][f][θ]_ [(] **[x]** _[t][, t, c]_ [)] _[−]_ **[x]** _[||]_ 2 [2][,] _[ L][θ]_ dsm [ref] [(] **[x]** _[,]_ **[ x]** _[t][,]_ **[ c]** [) =] _[ ||][f][θ]_ ref [(] **[x]** _[t][, t, c]_ [)] _[−]_ **[x]** _[||]_ [2] 2 [,] _[ λ][t]_ [is a weighting]
function and the constant _T_ can be factored into _β_ . We defer the derivation from Eq. (15) to Eq. (17)
in the Appendix C. The advantage of the derived DGPO objective is fourfold: 1) _**Leverages Relative**_
_**information**_ : It directly learns preferences between groups of samples, which leverages the finegrained relative preference information of individual samples within groups. 2) _**Enhances training**_
_**efficiency**_ : It does not require training on the entire sampling trajectory, which notably reduces
the computational cost per iteration. 3) _**Enables**_ _**effective**_ _**learning**_ : It sidesteps the need for an
inefficient stochastic policy, thus avoiding inefficient model-agnostic exploration and allowing the
model to learn more effectively and directly from the preference data. 4) _**Efficient**_ _**Sampling**_ _**and**_
_**Learning**_ : It allows the usage of deterministic ODE sampling for rollouts. This yields higher-quality
training samples compared to inefficient SDE sampling, all while using the same inference budget.


**Timestep Clip Strategy** The considered online setting requires generating samples from the online model which might be expensive; thus, we take a few steps (e.g., 10) for generating samples
to reduce the inference cost following Flow-GRPO (Liu et al., 2025). However, naively performing
DGPO’s training on these samples generated by few steps would lead to serious performance degradation due to the poor sample quality. To mitigate this, we propose the simple yet effective _Timestep_
_Clip Strategy_ : during training, we only sample timesteps from the range [ _t_ min _, T_ ] with a chosen minimum timestep _t_ min _>_ 0. This could effectively prevent the model from overfitting specific artifacts
(e.g., blurriness) of the generated samples by few steps (see ablation in Fig. 4).


6


Table 1: **GenEval Result.** We **highlight** the best scores. Obj.: Object; Attr.: Attribution.


**Model** **Overall** **Single Obj.** **Two Obj.** **Counting** **Colors** **Position** **Attr.** **Binding**


_**Autoregressive Models:**_


Show-o (Xie et al., 2024) 0.53 0.95 0.52 0.49 0.82 0.11 0.28
Emu3-Gen (Wang et al., 2024a) 0.54 0.98 0.71 0.34 0.81 0.17 0.21
JanusFlow (Ma et al., 2025) 0.63 0.97 0.59 0.45 0.83 0.53 0.42
Janus-Pro-7B (Chen et al., 2025b) 0.80 0.99 0.89 0.59 0.90 0.79 0.66
GPT-4o (Hurst et al., 2024) 0.84 0.99 0.92 0.85 0.92 0.75 0.61

|Diffusion Models:|Col2|Col3|
|---|---|---|
|LDM (Rombach et al., 2022)<br>SD1.5 (Rombach et al., 2022)<br>SD2.1 (Rombach et al., 2022)<br>SD-XL (Podell et al., 2023)<br>DALLE-2 (OpenAI, 2023)<br>DALLE-3 (Betker et al., 2023)<br>FLUX.1 Dev (Labs, 2024)<br>SD3.5-L (Esser et al., 2024)<br>SANA-1.5 4.8B (Xie et al., 2025)|0.37<br>0.43<br>0.50<br>0.55<br>0.52<br>0.67<br>0.66<br>0.71<br>0.81|0.92<br>0.29<br>0.23<br>0.70<br>0.02<br>0.05<br>0.97<br>0.38<br>0.35<br>0.76<br>0.04<br>0.06<br>0.98<br>0.51<br>0.44<br>0.85<br>0.07<br>0.17<br>0.98<br>0.74<br>0.39<br>0.85<br>0.15<br>0.23<br>0.94<br>0.66<br>0.49<br>0.77<br>0.10<br>0.19<br>0.96<br>0.87<br>0.47<br>0.83<br>0.43<br>0.45<br>0.98<br>0.81<br>0.74<br>0.79<br>0.22<br>0.45<br>0.98<br>0.89<br>0.73<br>0.83<br>0.34<br>0.47<br>0.99<br>0.93<br>0.86<br>0.84<br>0.59<br>0.65|
|SD3.5-M (Esser et al., 2024)<br>w/ Flow-GRPO (Liu et al., 2025)|0.63<br>0.95|0.98<br>0.78<br>0.50<br>0.81<br>0.24<br>0.52<br>**1.00**<br>0.99<br>0.95<br>0.92<br>**0.99**<br>0.86|
|**SD3.5-M w/ DGPO (Ours)**|**0.97**|**1.00**<br>**0.99**<br>**0.97**<br>**0.95**<br>**0.99**<br>**0.91**|


Table 2: **Performance** **on** **Compositional** **Image** **Generation,** **Visual** **Text** **Rendering,** **and** **Hu-**
**man Preference** benchmarks. ImgRwd: ImageReward; UniRwd: UnifedReward.


**Task Metric** **Image Quality** **Preference Score**
**Model**

**GenEval** **OCR Acc.** **PickScore** **Aesthetic** **DeQA** **ImgRwd** **PickScore** **UniRwd**


SD3.5-M 0.63 0.59 21.72 5.39 4.07 0.87 22.34 3.33


_**Compositional Image Generation:**_


Flow-GRPO 0.95  -  - 5.25 4.01 1.03 22.37 3.51
**DGPO (Ours)** 0.97  -  - 5.31 4.03 1.08 22.41 3.60


_**Visual Text Rendering:**_


Flow-GRPO  - 0.92  - 5.32 4.06 0.95 22.44 3.42
**DGPO (Ours)**  - 0.96  - 5.37 4.09 1.02 22.52 3.48


_**Human Preference Alignment:**_


Flow-GRPO  -  - 23.31 5.92 4.22 1.28 23.53 3.66
**DGPO (Ours)**  -  - 23.89 6.08 4.40 1.32 23.91 3.74


4 EXPERIMENTS


In this section, we comprehensively evaluate the proposed DGPO. Specifically, we benchmark improvements on three tasks—compositional image generation, visual text rendering, and human preference alignment (Tables 1 and 2). We also present qualitative comparisons and training efficiency
(Figs. 2 and 3). We further conduct ablations on key components (Figs. 4 and 5).


4.1 EXPERIMENTAL SETUP


**Evaluation Tasks** We evaluate the DGPO on post-training the SD3.5-M (Esser et al., 2024) across
three distinct valuable tasks: 1) compositional image generation, using GenEval to test object counting, spatial relations, and attribute binding; 2) visual text rendering (Gong et al., 2025), measuring
accuracy of rendering text in generated images, and 3) human preference alignment, using PickScore
to assess visual quality and text-image alignment. Details are provided in the Section E.


**Out-of-Domain Evaluation Metrics** To fairly evaluate model performance and guard against reward hacking—where models may overfit to training rewards signal while compromising actual
image quality—we employ four independent image quality metrics not used during training as outof-domain evaluations: Aesthetic Score (Schuhmann et al., 2022), DeQA (You et al., 2025), ImageReward (Xu et al., 2023), and UnifiedReward (Wang et al., 2025). We compute these metrics on
DrawBench (Saharia et al., 2022), a comprehensive benchmark featuring diverse prompts.


7


1.0


0.9


0.8


0.7


0.6


(a) OCR Accuracy Comparison

|Col1|Col2|Col3|~1|9× Faste|r|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||||~~~1~~|||||||
|||||||||||
|||||||||||
|||||||||low-GR<br>~~DGPO (O~~|O<br>~~ urs)~~|


0 25 50 75 100 125 150 175 200
Training Time (Hours)


24.0


23.5


23.0


22.5


22.0


21.5


(b) Pick Score Comparison

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||~1|~1|× Faster||||
|||~1|~1|× Faster|× Faster|× Faster|× Faster|
|||||||||
|||||||||
||||||||~~low-GRPO~~<br>DGPO (Ours)|


0 50 100 150 200 250 300
Training Time (Hours)


Figure 3: Compare the training speed of Flow-GRPO and our proposed DGPO.


A panda holds the
sign says " **Timestep**
**Clip Matters** "


A panda holds the
sign says " **Timestep**
**Clip Matters** "


A coffee shop
with sign " **DGPO** "


A dragon with
Subtitles " **Ultra-**
**Fast Training** "


A coffee shop
with sign " **DGPO** "


A dragon with
Subtitles " **Ultra-**
**Fast Training** "


DGPO w/o Timestep Clip Strategy **DGPO (Ours)**


Figure 4: Visual comparisons among variants. It can be seen that without the proposed timestep clip
strategy, although it can still accurately follow the instruction, the visual quality notably degrades


4.2 MAIN RESULTS


**Quantitative** **Results** Table 1 shows that DGPO achieves state-of-the-art performance on
GenEval, notably surpassing prior SOTA methods such as GPT-4o and Flow-GRPO. This improvement is achieved while maintaining performance across various out-of-domain metrics (such as
AeS, DeQA, and Image Reward), as indicated by Table 2. Beyond compositional image generation,
Table 2 provides detailed evaluation results on visual text rendering and human preference tasks,
where DGPO similarly demonstrates significant improvements in the target optimization metrics
while maintaining performance across various out-of-domain metrics.


**Qualitative** **Comparison** We present the qualitative comparisons of methods trained with
GenEval’s signal in Fig. 2. It is clear that the proposed DGPO can follow the instructions more
accurately compared to the base diffusion model and also the Flow-GRPO. Although Flow-GRPO
also shows accurate instruction following, its image quality degrades seriously, while our method
shows notably better visual quality. We present additional visual samples in Appendix F.


**Training** **Cost** The overall training of the proposed DGPO is quick, since we do not require the
inefficient stochastic policy for training. Besides, the training of the DGPO is efficient per iteration,
since we do not perform training on the whole trajectory. Benefit from these points, the overall
training of DGPO for reinforcement post-training is much faster than prior SOTA Flow-GRPO. _**As**_
_**shown in Figs. 1 and 3, the overall training of DGPO is generally around 20**_ _×_ _**faster than Flow-**_
_**GRPO.**_


4.3 ABLATION STUDY


**Effect of Timestep Clip Strategy** We found that without the proposed timestep clip strategy, the
reward metric slightly degrades from 0.96 to 0.95 regarding OCR Accuracy, while the visual quality
seriously degrades as shown in Fig. 4.


**ODE** **Rollout** **vs.** **SDE** **Rollout** A core advantage of our work compared to prior GRPO-style
works (Liu et al., 2025) is the ability to use the efficient ODE solvers for generating samples. This
can deliver samples with better quality and rewards. Results in Fig. 5 show that ODE rollout notably
outperforms SDE rollout in both convergence speed and ultimate metrics. This suggests that the use


8


0.90


0.85


0.80


0.75


0.70


0.65


0.60


0.55


(a) Offline Methods

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||O<br>O|O<br>O|||
||||O<br>O|O<br>O|ffline DPO<br>fline DGPO (|urs)|


0 200 400 600 800 1000
Training Steps


1.0


0.9


0.8


0.7


0.6


(b) Online Methods

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
|||||O<br>DG<br>~~D~~|line DPO<br>PO w/ SDE (<br>~~PO (Ours)~~|Ours)|


0 200 400 600 800 1000
Training Steps


Figure 5: Comparison of visual text rendering across variants.


of SDE in prior works may have been a requirement of the policy gradient framework rather than
providing more diverse samples for training.


**Offline DGPO** Our work can be easily adapted to the offline setting by using the reference model
_p_ ref for generating the training dataset. Results in Fig. 5 show that our offline DGPO can reasonably
boost the performance over the baseline, but its performance is notably worse than the online setting.


**Compared** **to** **Diffusion** **DPO** Diffusion DPO can also avoid the need for the stochastic policy,
however, it cannot leverage the fine-grained reward signals of each sample. Results in Fig. 5 show
that our DGPO notably outperforms the DPO in both online and offline settings, indicating the
effectiveness of our proposed DGPO in leveraging fine-grained group relative information.


5 RELATED WORKS


Recent research has focused extensively on aligning DMs with human preferences through three
primary approaches. The first involves fine-tuning diffusion models on carefully curated imageprompt datasets (Dai et al., 2023; Podell et al., 2023). The second approach maximizes explicit
reward functions, either by evaluating multi-step diffusion generation outputs (Prabhudesai et al.,
2023; Clark et al., 2023; Lee et al., 2023; Ho et al., 2022; Luo et al., 2025a;b) or through policy
gradient-based learning (Fan et al., 2024; Black et al., 2023; Ye et al., 2024). The third category
employs implicit reward maximization, as demonstrated by Diffusion-DPO (Wallace et al., 2024)
and Diffusion-KTO (Yang et al., 2024), which directly leverage raw preference data. A concurrent work (Chen et al., 2025a) explores utilizing the group information in DPO by enumerating
all pairwise comparisons within the group. In contrast, our work defines a single group-level reward and reinforces the DMs with maximum-likelihood learning on that group-level reward. Recent
works have also adapted GRPO to DMs (Liu et al., 2025; Xue et al., 2025) under policy-gradient
framework, demonstrating promising scalability and impressive performance improvements. However, a notable drawback of existing GRPO-style approaches is their reliance on a stochastic policy,
which requires inefficient SDE-based rollouts during training. Our work identifies group relative
information as the critical component of GRPO and introduces DGPO to directly optimize group
preferences, thereby exploiting fine-grained group relative information without requiring stochastic
policies. As a result, DGPO achieves significantly faster training and superior performance on both
in-domain and out-of-domain reward benchmarks compared to prior GRPO-style methods.


6 CONCLUSION


In this work, we introduce Direct Group Preference Optimization (DGPO), a novel online reinforcement learning method specifically designed for post-training diffusion models. Our approach
addresses the fundamental mismatch between policy gradient methods like GRPO and the inherent mechanics of diffusion generation. By recognizing that GRPO’s effectiveness stems primarily
from its utilization of group relative preference information within the group rather than its policygradient nature, we developed a method that preserves this key strength while eliminating the need
for stochastic policies. DGPO’s direct optimization approach offers substantial practical advantages over existing methods. By enabling the use of efficient ODE-based samplers, eliminating
reliance on model-agnostic noise for exploration, and avoiding expensive trajectory-based training,
DGPO achieves around 20× speedup in overall training time compared to Flow-GRPO. More importantly, our experiments demonstrate that this efficiency gain comes with superior performance, as


9


DGPO consistently outperforms baseline methods across both in-domain and out-of-domain evaluation metrics.


ETHICS STATEMENT


This work did not involve human or animal subjects, sensitive data, or any other elements that would
necessitate an ethical review. We have identified no potential for misuse or negative societal impact.


REPRODUCIBILITY STATEMENT


To ensure the reproducibility of our findings, our complete code and experimental setup, which
builds upon the open-source Flow-GRPO codebase (Liu et al., 2025), will be made publicly available
[at https://github.com/Luo-Yihong/DGPO.](https://github.com/Luo-Yihong/DGPO)


ACKNOWLEDGMENTS


Jing Tang’s work is partially supported by National Key R&D Program of China under Grant No.
2024YFA1012700, by the National Natural Science Foundation of China (NSFC) under Grant No.
62402410, and by Guangdong Provincial Project (No. 2023QN10X025).


REFERENCES


Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless
assistant with reinforcement learning from human feedback. _arXiv_ _preprint_ _arXiv:2204.05862_,
2022.


Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal
reverse variance in diffusion probabilistic models. _arXiv preprint arXiv:2201.06503_, 2022.


James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang
Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. _Computer_
_Science. https://cdn. openai. com/papers/dall-e-3. pdf_, 2023.


Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion
models with reinforcement learning. _arXiv preprint arXiv:2305.13301_, 2023.


Onur Celik, Zechu Li, Denis Blessing, Ge Li, Daniel Palenicek, Jan Peters, Georgia Chalvatzaki,
and Gerhard Neumann. Dime: Diffusion-based maximum entropy reinforcement learning. _arXiv_
_preprint arXiv:2502.02316_, 2025.


Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser:
Diffusion models as text painters. _Advances in Neural Information Processing Systems_, 36:9353–
9387, 2023.


Renjie Chen, Wenfeng Lin, Yichen Zhang, Jiangchuan Wei, Boyuan Liu, Chao Feng, Jiao Ran, and
Mingyu Guo. Towards self-improvement of diffusion models via group preference optimization,
2025a. [URL https://arxiv.org/abs/2505.11070.](https://arxiv.org/abs/2505.11070)


Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and
Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model
scaling. _arXiv preprint arXiv:2501.17811_, 2025b.


Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models
on differentiable rewards. _arXiv preprint arXiv:2309.17400_, 2023.


Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon
Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation
models using photogenic needles in a haystack. _arXiv preprint arXiv:2309.15807_, 2023.


10


DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,
2025. [URL https://arxiv.org/abs/2501.12948.](https://arxiv.org/abs/2501.12948)


Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow
transformers for high-resolution image synthesis, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2403.03206)
[2403.03206.](https://arxiv.org/abs/2403.03206)


Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel,
Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for finetuning text-to-image diffusion models. _Advances in Neural Information Processing Systems_, 36,
2024.


Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin Patrick Murphy, and
Tim Salimans. Diffusion models and gaussian flow matching: Two sides of the same coin. In
_The Fourth Blogpost Track at ICLR 2025_ [, 2025. URL https://openreview.net/forum?](https://openreview.net/forum?id=C8Yyg9wy0s)
[id=C8Yyg9wy0s.](https://openreview.net/forum?id=C8Yyg9wy0s)


Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework
for evaluating text-to-image alignment. _Advances in Neural Information Processing Systems_, 36:
52132–52152, 2023.


Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu,
Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation
foundation model. _arXiv preprint arXiv:2503.07703_, 2025.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in_
_Neural Information Processing Systems_, 33:6840–6851, 2020.


Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P
Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition
video generation with diffusion models. _arXiv preprint arXiv:2210.02303_, 2022.


Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. _arXiv_ _preprint_
_arXiv:2410.21276_, 2024.


Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. _Advances_ _in_ _neural_
_information processing systems_, 36:36652–36663, 2023.


Black Forest Labs. Flux. [https://github.com/black-forest-labs/flux, 2024.](https://github.com/black-forest-labs/flux)


Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel,
Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human
feedback. _arXiv preprint arXiv:2302.12192_, 2023.


Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan,
Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. _arXiv_
_preprint arXiv:2505.05470_, 2025.


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A
fast ode solver for diffusion probabilistic model sampling in around 10 steps. _arXiv_ _preprint_
_arXiv:2206.00927_, 2022a.


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast
solver for guided sampling of diffusion probabilistic models. _arXiv preprint arXiv:2211.01095_,
2022b.


Yihong Luo, Tianyang Hu, Weijian Luo, Kenji Kawaguchi, and Jing Tang. Rewardinstruct: A reward-centric approach to fast photo-realistic image generation. _arXiv_ _preprint_
_arXiv:2503.13070_, 2025a.


11


Yihong Luo, Tianyang Hu, Yifan Song, Jiacheng Sun, Zhenguo Li, and Jing Tang. Adding additional
control to one-step diffusion with joint distribution matching. _arXiv preprint arXiv:2503.06652_,
2025b.


Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion
models by trajectory distribution matching. _arXiv preprint arXiv:2503.06674_, 2025c.


Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan,
Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In _Proceedings of the Computer_
_Vision and Pattern Recognition Conference_, pp. 7739–7751, 2025.


OpenAI. Dalle-2, 2023. [URL https://openai.com/dall-e-2.](https://openai.com/dall-e-2)


Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. _Advances in neural information processing systems_, 35:
27730–27744, 2022.


Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image
synthesis. _arXiv preprint arXiv:2307.01952_, 2023.


Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-toimage diffusion models with reward backpropagation. _arXiv preprint arXiv:2310.03739_, 2023.


Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. Learning a diffusion model policy
from rewards via q-score matching. _arXiv preprint arXiv:2312.11752_, 2023.


Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model,
2024. [URL https://arxiv.org/abs/2305.18290.](https://arxiv.org/abs/2305.18290)


Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. _arXiv preprint arXiv:2409.00588_, 2024.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition_, pp. 10684–10695, 2022.


Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic
text-to-image diffusion models with deep language understanding. _Advances in neural informa-_
_tion processing systems_, 35:36479–36494, 2022.


Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi
Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An
open large-scale dataset for training next generation image-text models. _Advances_ _in_ _Neural_
_Information Processing Systems_, 35:25278–25294, 2022.


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_, 2024.


Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. In _International Conference on Machine Learn-_
_ing_, pp. 2256–2265. PMLR, 2015.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv_
_preprint arXiv:2010.02502_, 2020.


12


Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam,
Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using
direct preference optimization. In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_, pp. 8228–8238, 2024.


Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan
Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need.
_arXiv preprint arXiv:2409.18869_, 2024a.


Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. _arXiv preprint arXiv:2503.05236_, 2025.


Yinuo Wang, Likun Wang, Yuxuan Jiang, Wenjun Zou, Tong Liu, Xujie Song, Wenxuan Wang,
Liming Xiao, Jiang Wu, Jingliang Duan, et al. Diffusion actor-critic with entropy regulator.
_Advances in Neural Information Processing Systems_, 37:54183–54204, 2024b.


Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai
Zhang, Muyang Li, Junyu Chen, et al. Sana 1.5: Efficient scaling of training-time and inferencetime compute in linear diffusion transformer. _arXiv preprint arXiv:2501.18427_, 2025.


Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin,
Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer
to unify multimodal understanding and generation. _arXiv preprint arXiv:2408.12528_, 2024.


Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao
Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In
_Proceedings of the 37th International Conference on Neural Information Processing Systems_, pp.
15903–15935, 2023.


Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei
Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. _arXiv_
_preprint arXiv:2505.07818_, 2025.


Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li.
Using human feedback to fine-tune diffusion models without any reward model. In _Proceedings_
_of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 8941–8951, 2024.


Long Yang, Zhixiong Huang, Fenghao Lei, Yucun Zhong, Yiming Yang, Cong Fang, Shiting Wen,
Binbin Zhou, and Zhouchen Lin. Policy representation via diffusion probability model for reinforcement learning. _arXiv preprint arXiv:2305.13122_, 2023.


Zilyu Ye, Zhiyang Chen, Tiancheng Li, Zemin Huang, Weijian Luo, and Guo-Jun Qi. Schedule
on the fly: Diffusion time prediction for faster and better image generation. _arXiv_ _preprint_
_arXiv:2412.01243_, 2024.


Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, and Chao Dong. Teaching large language models
to regress accurate image quality scores using score distribution. In _Proceedings of the Computer_
_Vision and Pattern Recognition Conference_, pp. 14483–14494, 2025.


Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul
Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. _arXiv_
_preprint arXiv:1909.08593_, 2019.


A USE OF LARGE LANGUAGE MODELS


The use of large language models (LLMs) was strictly limited to language refinement and minor
editorial tasks. The authors affirm that LLMs played no part in the substantive phases of the research,
which include the ideation, experimental design, data analysis, and the interpretation of results. All
scientific content, methodologies, and conclusions presented herein were conceived and developed
exclusively by the authors.


13


B ADDITIONAL RELATED WORKS


**Diffusion-Based Policies in RL.** Diffusion policies have gained significant traction in recent online RL research. Existing methods have investigated optimizing these policies via different approaches. The first uses reparameterized policy gradients for optimization (Wang et al., 2024b; Ren
et al., 2024; Celik et al., 2025); The second explores optimizing via variants of score matching.
There are also works exploring optimizing via weighted schemes (Psenka et al., 2023; Yang et al.,
2023). These works focus on using diffusion policy for action learning. In contrast, our work explores reinforcing diffusion models in a more diffusion-native way for achieving better performance
in the image generation domain.


C DERIVATION


C.1 PRELIMINARY: JENSEN’S INEQUALITY.


Let _ϕ_ : R _→_ R be a convex function. For a random variable _X_, Jensen’s inequality states that the
function of the expectation is less than or equal to the expectation of the function:


_ϕ_ (E[ _X_ ]) _≤_ E[ _ϕ_ ( _X_ )] _._ (18)


C.2 DERIVATION OF EQ. 16


For simplicity and without loss of generality, we omit the condition **c** in the derivation. The Eq. 15
can be rewritten as:


E _q_ ( **x** 1: _T |_ **x** 0) _w_ ( **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T]_ [ )]

_p_ ref( **x** 0: _T_ )

**x** 0 _∈G_ [+]


     _L_ ( _θ_ ) = _−_ E( _G_ + _,G−_ ) _∼D_ log _σ_ ( _β_ [


_p_ ref( **x** 0: _T_ )


_w_ ( **x** 0)E _q_ ( **x** 1: _T |_ **x** 0) log _[p][θ]_ [(] **[x]** [0:] _[T]_ [ )]

_p_ ref( **x** 0: _T_

**x** 0 _∈G_ _[−]_


_−_ 


_p_ ref( **x** 0: _T_ ) [])]


      = _−_ E( _G_ + _,G−_ ) _∼D_ log _σ_ ( _β_ E _q_ ( **x** 1: _T |_ **x** 0) _,_ **x** 0 _∈G_ [ _w_ ( **x** 0)

**x** 0 _∈G_ [+]


_T_


- log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )

_t_ =1


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) [])]


_−_ - _w_ ( **x** 0)


**x** 0 _∈G_ _[−]_


_T_


- log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )

_t_ =1


_w_ ( **x** 0) _T_ E _t_ log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )

**x** 0 _∈G_ [+]


      = _−_ E( _G_ + _,G−_ ) _∼D_ log _σ_ ( _β_ E _q_ ( **x** 1: _T |_ **x** 0) _,_ **x** 0 _∈G_ [


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )


_w_ ( **x** 0) _T_ E _t_ log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_

**x** 0 _∈G_ _[−]_


_−_ 


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) [])]


- _w_ ( **x** 0) log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )

**x** 0 _∈G_ [+]


        = _−_ E( _G_ + _,G−_ ) _∼D_ log _σ_ ( _βT_ E _t_ E _q_ ( **x** _t|_ **x** 0) _,q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _,_ **x** 0 _∈G_ [


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )


- _w_ ( **x** 0) log _[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_

**x** 0 _∈G_ _[−]_


_−_ 


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) [])]


(19)
By Jensen’s inequality (Section C.1) and the convexity of _−_ log _σ_, we have:


_w_ ( **x** 0) _· βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ [)] )
**x** 0 _∈G_ [+]


      _L_ ( _θ_ ) _≤−_ E( _G_ + _,G−_ ) _∼D_ E _t,q_ ( **x** _t|_ **x** 0) log _σ_ (


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )


_w_ ( **x** 0) _· βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ [)]
**x** 0 _∈G_ _[−]_


_−_ 


(20)

_[p][θ]_ [(] **[x]** _[t][−]_ [1] _[|]_ **[x]** _[t]_ [)]

_p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) [)] _[,]_


14


C.3 DERIVATION OF EQ. 17


For simplicity and without loss of generality, we omit the condition **c** in the derivation. The DGPO
objective in Eq. 16 can be rewritten to:


_w_ ( **x** 0) _· βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ [)] )
**x** 0 _∈G_ [+]


      _L_ ( _θ_ ) _≤−_ E( _G_ + _,G−_ ) _∼D_ E _t,q_ ( **x** _t|_ **x** 0) log _σ_ (


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ )


_w_ ( **x** 0) _· βT_ E _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) log _p_ _[p]_ ref _[θ]_ [(] ( **[x]** **x** _[t]_ _t_ _[−]_ _−_ [1] 1 _[|]_ _|_ **[x]** **x** _[t]_ _t_ [)]
**x** 0 _∈G_ _[−]_


_−_ 


_p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) [)] _[,]_


      = _−_ E( _G_ + _,G−_ ) _∼D_ E _t,q_ ( **x** _t|_ **x** 0) log _σ_ ( _−βT_ _{_ _w_ ( **x** 0) _·_ [ _KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ))

**x** 0 _∈G_ [+]


_−_ _KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||p_ ref( **x** _t−_ 1 _|_ **x** _t_ ))] _−_ - _w_ ( **x** 0) _·_ [ _KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ))

**x** 0 _∈G_ _[−]_


_−_ _KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||p_ ref( **x** _t−_ 1 _|_ **x** _t_ ))] _}_ )
(21)
With the Gaussian parameterization (Song et al., 2020) of the posterior _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) ≜

_N_ ( _αt−_ 1 **x** 0 + ~~�~~ _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1 **x** _t−σαtt_ **x** 0 _, ηt_ [2] _−_ 1 _[I]_ [)] and reverse sampling _pθ_ ( **x** _t−_ 1 _|_ **x** _t_ ) ≜

_N_ ( _αt−_ 1 _fθ_ ( **x** _t, t_ ) + ~~�~~ _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1 **x** _t−αtσftθ_ ( **x** _t,t_ ) _, ηt_ [2] _−_ 1 _[I]_ [)][,] the KL divergence term

_KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||pθ_ ( **x** _t−_ 1 _|_ **x** _t_ )) can be computed by:


_KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||pθ_ ( **x** _t−_ 1 _|_ **x** _t_ )


1   - **x** _t −_ _αt_ **x** 0
= _||αt−_ 1 **x** 0 + _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1
2 _ηt_ [2] _−_ 1 _σt_


_αt_ **x** 0 - **x** _t −_ _αtfθ_ ( **x** _t, t_ )

_−_ ( _αt−_ 1 _fθ_ ( **x** _t, t_ ) + _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1
_σt_ _σt_


_θ_ _t_

_σt_ ) _||_ [2] 2


�2
_||_ **x** 0 _−_ _fθ_ ( **x** _t, t_ ) _||_ [2] 2


1
=
2 _ηt_ [2] _−_ 1


- - _αt_
_αt−_ 1 _−_ _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1
_σt_


�2
_._


1
= _λt||_ **x** 0 _−_ _fθ_ ( **x** _t, t_ ) _||_ [2] 2 _[,]_ where _λt_ =
2 _ηt_ [2] _−_ 1


- - _αt_
_αt−_ 1 _−_ _σt_ [2] _−_ 1 _[−]_ _[η]_ _t_ [2] _−_ 1
_σt_


(22)
Similarly, we have _KL_ ( _q_ ( **x** _t−_ 1 _|_ **x** _t,_ **x** 0) _||p_ ref( **x** _t−_ 1 _|_ **x** _t_ ) = _λt||_ **x** 0 _−_ _fθ_ [ref][(] **[x]** _[t][, t]_ [)] _[||]_ 2 [2][.] [Substituting] [the]
computed KL divergence into Eq. 21, we can obtain Eq. 17.


D VISUALIZATION OF REWARD HACKING


We set _β_ to be smaller (e.g., _β_ = 10) than its normal value (e.g., _β_ = 100) and extended the
training iterations to make the model over-optimize the rewards. We visualize some failure modes
of over-optimizing rewards in Section D.


E EXPERIMENT DETAILS


**Compositional Image Generation.** We evaluate text-to-image models on complex compositional
prompts using GenEval (Ghosh et al., 2023), which tests six challenging compositional generation
tasks including object counting, spatial relations, and attribute binding.


**Visual Text Rendering.** Following the methodology in TextDiffuser (Chen et al., 2023) and FlowGRPO’s experimental setup, we evaluate models’ ability to accurately render text within generated
images. Each prompt follows the template structure “A sign that says ‘text”’, where ‘text’ represents
the exact string to be rendered in the image. We measure text fidelity (Gong et al., 2025) as follows:


_r_ = max(1 _−_ _Ne/N_ ref _,_ 0)


where _Ne_ denotes the minimum edit distance between rendered and target text, and _N_ ref represents
the character count within the prompt’s quotation marks.


15


w/ OCR Acc


Overoptimized
DGPO


DGPO


w/ HPS


A photo of a panda holding a sign says ‘Over optimize Test’ A photo of a panda holding a sign says ‘DGPO’ in forest

Visualization of reward hacking. Over-optimizing the rule-based reward (i.e., OCR Acc) preserves
text accuracy but degrades image quality. In contrast, over-optimizing the model-based reward (i.e.,
HPS) introduces specific artifacts, such as repeated objects in the background.


**Human Preference Alignment.** To align text-to-image models with human preferences, we employ PickScore (Kirstain et al., 2023) as the reward signal. The PickScore model, trained on largescale human preference data, evaluates both visual quality and text-image alignment, providing a
comprehensive assessment of generation quality from a human-centric perspective.


**Setup** **Details.** We generate 24 samples for each group for training. We adopt the Flow-DPMSolver (Xie et al., 2025) with steps of 10 for rollout during training. We adopt LoRA fine-tuning
with a rank of 32. The _β_ is set to be 100 by default. Defaultly, We update _θ_ _[−]_ by identity mapping,
i.e., _θ_ _[−]_ _←_ _θ_ within 200 steps, and update _θ_ _[−]_ by EMA with decay of 0.3 in the remaining training.
Experiments are performed over 512 resolution. We use a probability of 0.05 to drop text during
training. The experiments are performed over A100. The reported GPU hours are A100 hours.


**Details of the out-of-domain evaluation metrics** We outline the specific out-of-domain metrics
used to assess quality: The aesthetic score (Schuhmann et al., 2022) employs a linear regression model based on CLIP to evaluate the visual appeal of generated images; For assessing image
quality degradation, we utilize the DeQA score (You et al., 2025). This metric leverages a multimodal large language model architecture to measure the impact of various imperfections—including
distortions, textural degradation, and low-level visual artifacts—on the overall perceived quality of
images; ImageReward (Xu et al., 2023) serves as a comprehensive human preference model for
text-to-image generation tasks. This reward function evaluates multiple dimensions including the
coherence between textual descriptions and visual content, the fidelity of generated visuals; Finally,
UnifiedReward (Wang et al., 2025) represents the latest advancement in this area. This integrated reward framework can evaluate both multimodal understanding and generation tasks, and has
demonstrated superior performance compared to existing methods on the human preference assessment leaderboard.


F ADDITIONAL QUALITATIVE COMPARISON


We present additional visual samples in Figs. 6 and 7.


G LIMITATIONS AND FUTURE WORKS


Our work focuses on text-to-image synthesis; however, it also has the potential to be adapted to
enhance text-to-video synthesis. Exploring the extension would be an interesting future work.


16


A photo of a
coffee shop with
sign "OCR Test"


A photo of a
book with title
"Diffusion RL"


A photo of a
cat holds a sign
"DGPO SOTA"


A photo of a
worn road
sign "Danger"


A vibrant urban alley with
a graffiti wall prominently
spray-painted "Street


SD 3.5M


SD 3.5M
w/ FlowGRPO


**SD 3.5M**
**w/ DGPO**
**(Ours)**


Figure 6: Qualitative comparisons of DGPO against competing methods. The training signal is
given by OCR Accuracy. All images are generated by the same initial noise.


A cute baby
playing with toys
in the snow


A photo of an
Asia girl with
sunglass


A small cactus with
a happy face in the
Sahara desert


A photo of a
woman on top
of a horse


A photo of a
monkey making
latte


SD 3.5M


SD 3.5M
w/ FlowGRPO


**SD 3.5M**
**w/ DGPO**
**(Ours)**


Figure 7: Qualitative comparisons of DGPO against competing methods. The training signal is
given by PickScore. All images are generated by the same initial noise.


17