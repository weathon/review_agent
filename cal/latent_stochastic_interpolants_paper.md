# LATENT STOCHASTIC INTERPOLANTS


**Saurabh Singh** _[∗]_
Poetiq AI
saurabh@poetiq.ai


**Dmitry Lagun**
Google DeepMind
dlagun@google.com


ABSTRACT


Stochastic Interpolants (SI) is a powerful framework for generative modeling,
capable of flexibly transforming between two probability distributions. However,
its use in jointly optimized latent variable models remains unexplored as it requires
direct access to the samples from the two distributions. This work presents Latent
Stochastic Interpolants (LSI) enabling joint learning in a latent space with endto-end optimized encoder, decoder and latent SI models. We achieve this by
developing a principled Evidence Lower Bound (ELBO) objective derived directly
in continuous time. The joint optimization allows LSI to learn effective latent
representations along with a generative process that transforms an arbitrary prior
distribution into the encoder-defined aggregated posterior. LSI sidesteps the simple
priors of the normal diffusion models and mitigates the computational demands
of applying SI directly in high-dimensional observation spaces, while preserving
the generative flexibility of the SI framework. We demonstrate the efficacy of
LSI through comprehensive experiments on the standard large scale ImageNet
generation benchmark.


1 INTRODUCTION


Diffusion models have achieved remarkable success in modeling complex, high-dimensional data
distributions across various domains. These models learn to transform a simple “prior” distribution
_p_ 0, such as a standard Gaussian, into a complex data distribution _p_ 1. While early formulations were
constrained to use specific prior distributions that are Lévy Stable, recent advancements, particularly
Stochastic Interpolants (SI) (Albergo et al., 2023) offer a powerful, unifying framework capable
of bridging arbitrary probability distributions. However, SI assumes that both the prior _p_ 0 and the
target _p_ 1 distributions are fixed and the samples from both are directly _observed_ . This requirement
limits their use in jointly learned latent variable models where the generative model is learned, along
with an encoder and a decoder, in a latent unobserved space. Further, the latent space, often lower
dimensional, evolves as the encoder and decoder are jointly optimized. Lack of support for joint
optimization implies that arbitrary fixed latent representations may not be optimally aligned with the
generative process resulting in inefficiencies.


To address this, we present Latent Stochastic Interpolants (LSI), a novel framework for end-to-end
learning of a generative model in an _unobserved_ latent space. Our key innovation lies in deriving a
principled, flexible and scalable training objective as an Evidence Lower Bound (ELBO) directly in
continuous time. This objective, like SI, provides data log-likelihood control, while enabling scalable
end-to-end training of the three components: an encoder mapping high-dimensional observations
to a latent space, a decoder reconstructing observations from latent representations, and a latent
SI model operating entirely within the learned latent space. Our approach allows transforming
arbitrary prior distributions into the encoder-defined aggregated posterior, simultaneously aligning
data representations with a high-fidelity generative process using that representation.


LSI’s single ELBO objective provides a unified, scalable framework that avoids the need for simple
priors of the normal diffusion models, mitigates the computational demands of applying SI directly
in high-dimensional observation spaces and offers an alternative to ad-hoc multi-stage training. Our
formulation admits simulation-free training analogous to observation-space diffusion and SI models,
while preserving the flexibility of SI framework. We empirically validate LSI’s strengths through


_∗_ Work done while at DeepMind.


1


comprehensive experiments on the challenging ImageNet generation benchmark, demonstrating
competitive generative performance and highlighting its advantages in efficiency.


Our key contributions are: 1) **Latent stochastic interpolants (LSI):** a novel and flexible framework
for scalable training of a latent variable generative model with continuous time dynamic latent
variables, where the encoder, decoder and latent generative model are jointly trained, 2) **Unifying**
**perspective:** a novel perspective on integrating flexible continuous-time formulation of SI within
latent variable models, leveraging insights from continuous time stochastic processes, 3) **Principled**
**ELBO objective:** a new ELBO as a principled training objective that retains strengths of SI – simple
simulation free training and flexible prior choice – while enabling the benefits of joint training in a
latent space.


2 BACKGROUND


**Notation.** We use small letters _x, y, t_ etc. to represent scalar and vector variables, _f, g_ etc. to
represent functions, Greek letters _β, θ_ etc. to represent (hyper-)parameters. Lower case letters _x_
are used to represent both the random variable and a particular value _x ∼_ _p_ ( _x_ ). Dependence on an
argument _t_ is indicated as a subscript _ut_ or argument _u_ ( _t_ ) interchangeably.


Our work builds upon two key results briefly reviewed below. The first result (Li et al., 2020;
Theodorou, 2015) states an Evidence Lower Bound (ELBO) for models using continuous time
dynamic latent variables. We state a more general form than the original to aid the discussion of the
prior distributions. The second result is a well known method for constructing a stochastic mapping
between two distributions. We exploit it to construct a variational approximation in the latent space.


2.1 VARIATIONAL LOWER BOUND USING DYNAMIC LATENT VARIABLES


Consider two SDEs, starting with the same starting point _z_ ˜0 = _z_ 0 at _t_ = 0, sharing the same
dispersion coefficient _σ_ ( _zt, t_ ) but potentially different initial distributions — _z_ 0 _∼_ _p_ 0( _z_ 0) for the
model, with path measure P _θ_, and _z_ 0 _∼_ _q_ 0( _z_ 0) for the variational posterior, with path measure Q:


_dz_ ˜ _t_ = _hθ_ (˜ _zt, t_ ) _dt_ + _σ_ (˜ _zt, t_ ) _d_ ˜ _wt,_ (model, path measure P _θ_ ) (1)
_dzt_ = _hϕ_ ( _zt, t_ ) _dt_ + _σ_ ( _zt, t_ ) _dwt,_ (variational posterior, path measure Q) (2)


Where _w_ ˜ _t_ and _wt_ are Wiener processes under corresponding path measures. The first equation can
be viewed as the latent dynamics under the model _hθ_ we are interested in learning and the second
as the latent dynamics under some variational approximation to the posterior that can be used to
produce samples _zt_ . Further, let _xti_ be observations at time _ti_ that are assumed to only depend on the
corresponding unobserved latent state _zti_, then the ELBO can be written as


_[q]_ [0][(] _[z]_ [0][)]

_p_ 0( _z_ 0) _[−]_ [1] 2


- _T_


ln _pθ_ ( _xt_ 1 _, . . ., xtn_ ) _≥_ EQ


= EQ


Where _u_ satisfies


- _n_


- ln _pθ_ ( _xti|zti_ ) _−_ ln _p_ _[q]_ [0] 0 [(] ( _[z]_ _z_ [0] 0 [)] )

_i_ =1


2


_∥u_ ( _zt, t_ ) _∥_ [2] _dt_
0


(3)


- _n_


ln _pθ_ ( _xti|zti_ )

_i_ =1


_−_ KL(Q _∥_ P _θ_ ) (4)


_σ_ ( _z, t_ ) _u_ ( _z, t_ ) = _hϕ_ ( _z, t_ ) _−_ _hθ_ ( _z, t_ ) (5)


We provide the proof of the above general form in Section A. Similar to the ELBO for the VAEs
(Kingma et al., 2013), the first term in eq. (4) explains observations given the latent path and the
second term penalizes the mismatch between the variational and model path distributions. In the
following, we focus on the case of _q_ 0 = _p_ 0 and draw attention to the general case when needed.


2.2 DIFFUSION BRIDGE


Given two arbitrary points _z_ 0 and _z_ 1, a diffusion bridge between the two is a random process
constrained to start and end at the two given end points. A diffusion bridge can be used to specify the
stochastic dynamics of a particle that starts at _z_ 0 at _t_ = 0 and is constrained to land at _z_ 1 at _t_ = 1.


2


Consider a stochastic process starting at _z_ 0 with the dynamics specified by eq. (2). Using Doob’s
h-transform, the SDE for the end point conditioned diffusion bridge, constrained to end at _z_ 1 at time
_t_ = 1 can be written as
_dzt_ = [ _hϕ_ ( _zt, t_ ) + _σ_ ( _zt, t_ ) _σ_ ( _zt, t_ ) _[T]_ _∇zt_ ln _p_ ( _z_ 1 _|zt_ )] _dt_ + _σ_ ( _zt, t_ ) _dwt_ (6)
where _p_ ( _z_ 1 _|zt_ ) is the conditional density for _z_ 1 under the original dynamics in eq. (2) and depends
on _hϕ_ . Note that a Brownian bridge is a special case of a Diffusion bridge where the dynamics are
specified by the standard Brownian motion. Diffusion bridges can be used to construct a stochastic
mapping between two distributions by considering the end points _z_ 0 _∼_ _p_ 0( _z_ 0) and _z_ 1 _∼_ _p_ 1( _z_ 1) to be
sampled from the two distributions of interest.


3 LATENT STOCHASTIC INTERPOLANTS


**Stochastic Interpolants (SI) and their limitation:** SI (Albergo et al., 2023) is a powerful framework for generative modeling, capable of learning a model that can flexibly transform between two
probability distributions. Let _x_ 1 _∼_ _p_ ( _x_ 1) be an observation from the data distribution _p_ ( _x_ 1) that we
want to sample from. In SI framework, another distribution _p_ 0( _x_ 0) is chosen as a prior with samples
_x_ 0 _∼_ _p_ 0( _x_ 0). Typically, _p_ 0 is easy to sample from, e.g. a Gaussian distribution. A stochastic interpolant _xt_ is then constructed with the requirement that the marginal distribution _pt_ ( _xt_ ) of _xt_ equals _p_ 0
at _t_ = 0 and _p_ 1 at _t_ = 1. For example, the interpolant _xt_ = (1 _−t_ ) _x_ 0 + _tx_ 1 +� _t_ (1 _−_ _t_ ) _ϵ, ϵ ∼_ _N_ (0 _, I_ )
satisfies this requirement. The velocity field and the score function for the generative model are then
estimated as solutions to particular least squares problems. The learned velocity field and the score
function can then be used to transform a sample from _p_ 0 to produce a sample from _p_ 1. SI requires
that the samples _x_ 0 and _x_ 1 are observed, though _x_ 1 could be an output of a _fixed_ model, hence still
observed. We use the term observation space SI to emphasize this.


However, we are interested in jointly learning a generative model in a latent space to leverage
efficiency of low dimensional representations while also aligning the latents with the generative
process. Therefore, we want to jointly optimize an encoder _pθ_ ( _z_ 1 _|x_ 1) that represents high dimensional
observations in the latent space and a decoder _pθ_ ( _x_ 1 _|z_ 1) that maps a given latent representation
to the observation space, along with the generative model in latent space. To use SI, we need
to interpolate between a fixed prior _p_ 0( _z_ 0) in the latent space and the true marginal posterior
_p_ 1( _z_ 1) _≡_ - _p_ ( _z_ 1 _|x_ 1) _dx_ 1. However, we only have access to the posterior model _pθ_ ( _z_ 1 _|x_ 1) that is
optimized concurrently and is an approximation to the true intractable posterior. Consequently, we
can not directly construct an interpolant in the latent space that satisfies the requirements of SI. In the
following, we address this issue by deriving Latent Stochastic Interpolants (LSI), though from an
entirely different perspective than is considered by SI.


**Generative model with dynamic latent variables:** Since we want to jointly learn the generative
model in a latent space, we propose a latent variable model where the unobserved latent variables
are assumed to evolve in continuous time according to the dynamics specified by an SDE of the
form in eq. (1). Let _pθ_ ( _x_ 1 _|z_ 1) be a parameterized stochastic decoder and _hθ_ parameterized drift for
eq. (1). Then, the generation process using our model is as following – first a sample _z_ 0 _∼_ _p_ 0( _z_ 0)
is produced from a prior _p_ 0( _z_ 0), then _z_ 0 evolves according to the dynamics specified by eq. (1)
using _hθ_ from _t_ = 0 to _t_ = 1 to yield a _z_ 1, and finally an observation space sample is produced
using the decoder _pθ_ ( _x_ 1 _|z_ 1). In theory, we can now utilize the ELBO presented in section 2.1 to
train this model. Note that, although the ELBO in eq. (3) supports arbitrary number of observations
_xti_ at arbitrary times _ti_, in this paper we focus on a single observation _x_ 1 at _t_ = 1. The ELBO in
eq. (3) needs a variational approximation to the posterior _pθ_ ( _zt|x_ 1) which can be used to sample
_zt_ . This approximation is constructed as another dynamical model specified by the SDE in eq. (2).
Unfortunately, for a general variational approximation specified by an arbitrary _hϕ_, simulating eq. (2)
would lead to significant computational burden for large problems during each training iteration and
open the door to additional issues resulting from approximations needed for simulation of the SDE.
Instead, we explicitly construct the drift _hϕ_ in eq. (2) such that _zt_ can be sampled directly without
simulation for any time _t_ . Our scheme provides a scalable alternative that allows simulation free
efficient training, as is common in the observation space diffusion models.


**Variational posterior with simulation free samples:** Next we construct a variational posterior
approximation, that enables easy sampling of _zt_ without requiring the simulation of the SDE in


3


eq. (2). Let _z_ 1 _∼_ _pθ_ ( _z_ 1 _|x_ 1) be a stochastic encoding of the observation _x_ 1 providing direct access to
_z_ 1 at _t_ = 1. Next, using the Diffusion Bridge specified by eq. (6) we construct a stochastic mapping
between the prior _p_ 0( _z_ 0) and the aggregated approximate posterior - _pθ_ ( _z_ 1 _|x_ 1) _dx_ 1 at _t_ = 1. The
diffusion bridge, coupled with the encoder _pθ_ ( _z_ 1 _|x_ 1) yields our approximate posterior _pθ_ ( _zt|x_ 1).
However, _p_ ( _z_ 1 _|zt_ ) is unknown in general. If we additionally assume that _hϕ_ ( _zt, t_ ) _≡_ _htzt_ and
_σ_ ( _zt, t_ ) _≡_ _σt_, then the original SDE in eq. (2) becomes linear with additive noise


_dzt_ = _htztdt_ + _σtdwt_ (7)


It is well known that for linear SDEs of the above form, the transition density _p_ ( _zt|zs_ ) _, t_ _>_ _s_
is gaussian _N_ ( _zt_ ; _astzs, bstI_ ) (see section G) for some functions _ast, bst_ that depend on _ht, σt_ .
Consequently, we can compute _∇zt_ ln _p_ ( _z_ 1 _|zt_ ) for a given _zt_ as

_∇zt_ ln _p_ ( _z_ 1 _|zt_ ) = _[a][t]_ [1][(] _[z]_ [1] _[ −]_ _[a][t]_ [1] _[z][t]_ [)] (8)

_bt_ 1


The transformed SDE in terms of the simplified drift and dispersion coefficients can be expressed as


_dzt_ = [ _htzt_ + _σt_ [2] _[∇][z]_ _t_ [ln] _[ p]_ [(] _[z]_ [1] _[|][z][t]_ [)]] _[dt]_ [ +] _[ σ][t][dw][t]_ (9)


Further, if we condition on the starting point _z_ 0, then the conditional density _p_ ( _zt|z_ 1 _, z_ 0) can be
expressed as following using the Bayes rule


_[t][, z]_ [0][)] _[p]_ [(] _[z][t][|][z]_ [0][)]

= _[p]_ [(] _[z]_ [1] _[|][z][t]_ [)] _[p]_ [(] _[z][t][|][z]_ [0][)]
_p_ ( _z_ 1 _|z_ 0) _p_ ( _z_ 1 _|z_ 0)


_p_ ( _zt|z_ 1 _, z_ 0) = _[p]_ [(] _[z]_ [1] _[|][z][t][, z]_ [0][)] _[p]_ [(] _[z][t][|][z]_ [0][)]


(10)
_p_ ( _z_ 1 _|z_ 0)


where _p_ ( _z_ 1 _|zt, z_ 0) = _p_ ( _z_ 1 _|zt_ ) because of the Markov independence assumption inherent in eq. (2).
Note that all the factors on the right are gaussian. It can be shown that the conditional density
_p_ ( _zt|z_ 1 _, z_ 0) is also gaussian if the transition densities are gaussian and takes the following form


_−_ [1]


_b_ 01

[1]

2 _b_ 0 _tbt_ 1


����2 [�]


    - 1 _b_ 01
_p_ ( _zt|z_ 1 _, z_ 0) =
2 _π_ _b_ 0 _tbt_ 1


- _[d]_ 2
exp


_b_ 0 _tat_ 1 _z_ 1 + _bt_ 1 _a_ 0 _tz_ 0
_zt −_
���� _b_ 01


(11)


Where _a_ ( _·_ ) _, b_ ( _·_ ) are constant or time dependent scalars and _d_ is the dimensionality of _zt_ . Their
specific forms depends on the choice of _ht, σt_ . Refer to section G for details. _zt_ can now be directly
sampled without simulating the SDE, given a sample _z_ 0 and the encoded observation _z_ 1. Note that
the assumptions made for eq. (7), while restrictive, do not limit the empirical performance.


**Latent stochastic interpolants:** We can now define latent stochastic interpolants using reparameterization trick in conjuction with eq. (11) to parameterize _zt_ as


_zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) (12)


For some functions _ηt, κt, νt_ that depend on _a_ ( _·_ ) _, b_ ( _·_ ). Note that _η_ 0 = _η_ 1 = 0 _, κ_ 0 = _ν_ 1 = 0 _, κ_ 1 =
_ν_ 0 = 1 since _zt_ is sampled from a diffusion bridge with the two end points fixed at _z_ 0 _, z_ 1. Equation (12) specifies a general stochastic interpolant, akin to the proposal in (Albergo et al., 2023), but
now in the latent space. If we choose the encoder and decoder to be identity functions, then above can
be viewed as an alternative way to construct stochastic interpolants in the observation space. Instead
of choosing _ht, σt_ first, we can instead choose _κt, νt_ and infer the corresponding _ht, σt_ . For example,
choosing _κt_ = _t, νt_ = 1 _−_ _t_ leads to _σt_ = _σ_, a constant, and we arrive at the following


_zt_ = _σ_             - _t_ (1 _−_ _t_ ) _ϵ_ + _tz_ 1 + (1 _−_ _t_ ) _z_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) (13)


See section J for a detailed derivation. We use the above form for all the experiments in the
paper. Further, if _p_ 0( _z_ 0) is chosen to be a standard gaussian then the interpolant simplifies to
_zt_ = _tz_ 1 + �(1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _z_ 0 (section M). With the above interpolants, we can now define
the ELBO and optimize it efficiently with _√_ simulation free samples _zt_ . We also derive the expressions
for variance preserving choices of _κt_ = _t, ηt_ [2] [+] _[ ν]_ _t_ [2] [= 1] _[ −]_ _[t]_ [ in section K, however we do not explore]

this interpolant empirically.


4


**Constructing training objective using ELBO (eq. (3)):** We first define _u_ ( _zt, t_ ) using eq. (9) as

_u_ ( _zt, t_ ) = _σt_ _[−]_ [1] [ _htzt_ + _σt_ [2] _[∇][z]_ _t_ [ln] _[ p]_ [(] _[z]_ [1] _[|][z][t]_ [)] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]] (14)

For the general latent stochastic interpolant _zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0 (eq. (12)), we show that _u_ ( _zt, t_ )
takes the following form


_u_ ( _zt, t_ ) = _σt_ _[−]_ [1] �� _dηdtt_ _[−]_ 2 _[σ]_ _ηt_ [2] _t_


_ϵ_ + _[dκ][t]_


       
_[t]_

(15)
_dt_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


_[t]_

_[dν][t]_
_dt_ _[z]_ [1][ +] _dt_


See section H for the proof. This _u_ ( _zt, t_ ) can be substituted into the ELBO in eq. (3) to construct a
training objective. For example, with the choices _κt_ = _t, νt_ = 1 _−_ _t_, we get


      _u_ ( _zt, t_ ) = _σ_ _[−]_ [1] _−σ_


~~�~~ _t_ 1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


See section J for details. We write a generalized loss based on the ELBO as


           

E _p_ ( _t_ ) _p_ ( _x_ 1 _,z_ 0) _pθ_ ( _z_ 1 _|x_ 1) _p_ ( _zt|z_ 1 _,z_ 0)


_−_ ln _pθ_ ( _x_ 1 _|z_ 1) + _[β][t]_

2


_σ_
����


(16)


(17)


~~�~~ _t_ 2 [�]
1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)] ����


_−_ ln _pθ_ ( _x_ 1 _|z_ 1) + _[β][t]_


Where _βt_ (discussed further in section 4) is a relative weighting term, similar in spirit to _β_ -VAE
(Higgins et al., 2017; Alemi et al., 2018), allowing empirical re-balancing for metrics of interest, e.g.
FID. Above loss is reminiscent of the SI training objective, but with an additional reconstruction term
and the interpolants _zt_ arising from the variational posterior. We use this training objective for all the
experiments, and optimize it using stochastic gradient descent to jointly train all three components –
encoder _pθ_ ( _z_ 1 _|x_ 1), decoder _pθ_ ( _x_ 1 _|z_ 1) and latent SI model _hθ_ ( _zt, t_ ). Note that we choose _pθ_ ( _x_ 1 _|z_ 1)
to be a conditional gaussian in all experiments, resulting in a simple _L_ 2 decoder loss.


**Observation-space stochastic interpolants:** To elucidate the connection with observation-space
SI (Albergo et al., 2023) we derive the corresponding training objective in our framework, yielding:


_σ_
����


~~�~~ _t_ 2 [�]
1 _−_ _t_ _[ϵ]_ [ +] _[ x]_ [1] _[ −]_ _[x]_ [0] _[ −]_ _[h][θ]_ [(] _[x][t][, t]_ [)] ����


E _p_ ( _t_ ) _p_ ( _x_ 1 _,x_ 0) _p_ ( _xt|x_ 1 _,x_ 0)


_βt_

2


(18)


where _βt_ has the same interpretation as in eq. (17), with _βt_ = _σ_ _[−]_ [2] corresponding to exact ELBO. See
Section B for detailed proof. Comparing with the LSI loss (eq. (17)), the observation-space ELBO is
precisely the LSI objective with the reconstruction term _−_ ln _pθ_ ( _x_ 1 _|z_ 1) removed and _z_ replaced by _x_ .
LSI recovers observation-space stochastic interpolants when the encoder and decoder are identity
functions. All parameterizations (Section 4) and sampling procedures (Section 5) apply directly with
_z_ replaced by _x_ . Lastly, the likelihood control property of the above objective is trivially established

- the objective corresponds to KL(Q _∥_ P _θ_ ) for _βt_ = _σ_ _[−]_ [2] and KL( _p_ 1 _∥pθ_ ) _≤_ KL(Q _∥_ P _θ_ ) (eq. (41)),
where _p_ 1 is the true data distribution and _pθ_ is the data likelihood under the model.


**Learnable priors:** When the prior _p_ 0 is parameterized (e.g., _pθ_ ( _z_ 0) = _N_ ( _µθ,_ Σ _θ_ )), the default
construction above uses the same learnable prior for both processes ( _q_ 0 = _pθ_ ), so KL( _q_ 0 _∥p_ 0) =
0 and the ELBO retains the same form. The prior parameters are still learned: they affect the
distribution of _z_ 0 in the path integral EQ[� _∥u∥_ [2] _dt_ ], and gradients flow through _z_ 0 _∼_ _pθ_ ( _z_ 0) via the
reparameterization trick. Alternatively, if the variational process uses a fixed reference _q_ 0 = _pθ_, the
KL( _q_ 0 _∥pθ_ ) term appears as an additional regularizer penalizing deviation from the reference. Same
carries over to the observation-space stochastic interpolants as well.


4 PARAMETERIZATION


Directhe _[√]_ tly us1 _−_ _t_ in the denominator of the second term.ing the loss in eq. (17) leads to high variance in gradients and unreliable training due toConsequently, we consider several alternative
parameterizations for the second term, including denoising and noise prediction (see section C for
details). Among the alternatives considered, we found the following parameterization, referred to as
InterpFlow, to reliably lead to better results and we use it in all our experiments.
_βt_ _√_ _√_ _√_ 2


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) +


2
_tzt −_ _h_ [ˆ] _θ_ ( _zt, t_ )��� (19)


2


_√_
_−σ_
���


_√_
_tϵ_ +


5


_√_
Where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ _tzt_ + _[√]_ 1 _−_ _thθ_ ( _zt, t_ ) and _βt_ _≡_ _β/_ (1 _−_ _t_ ) is a time _t_ dependent weighting term,

with _β_ a constant. Instead of explicitly using the weights _βt_, due to 1 _−_ _t_ in the denominator, we
consider a change of variable for _t_ with the parametric family _t_ ( _s_ ) = 1 _−_ (1 _−_ _s_ ) _[c]_ with _s ∼U_ [0 _,_ 1]
1
uniformly sampled. It can be shown that _p_ ( _t_ ) _∝_ (1 _−_ _t_ ) _c_ _[−]_ [1], therefore the change of variable provides
the reweighting and we simply set _βt_ = _β_, a constant. Empirically, we found that a value of _c_ = 1
(i.e. a uniform schedule) works the best for all parameterizations during training and sampling, except
for NoisePred and Denoising, which preferred _c_ _≈_ 2 during sampling. _c_ _<_ 1 led to degradation
in FID. Figure 4 in appendix visualizes _t_ ( _s_ ) for various values of _c_ . While the ELBO suggests
using _β_ = 1 _/σ_ [2], we compute the two terms in eq. (17) as averages and experiment with different
weightings. When used with optimizers like Adam or AdamW, _β_ can be interpreted as the relative
weighting of the gradients from the two terms for the encoder _pθ_ ( _z_ 1 _|x_ 1). A lower value of _β_ leads
the encoder to focus purely on the reconstruction and is akin to using a pre-trained encoder-decoder
pair as _β_ _→_ 0. A higher value of _β_ forces the encoder to adapt its representation for the second term
as well. We empirically study the effect of _β_ in the experiments.


5 SAMPLING


For the InterpFlow parameterization, the learned drift _√_ _h_ [ˆ] _θ_ ( _zt, t_ ) is related to the original drift _hθ_ ( _zt, t_ )
as _hθ_ ( _zt, t_ ) = ( _h_ [ˆ] ( _zt, t_ ) _−_ _tzt_ ) _/_ _[√]_ 1 _−_ _t_ (see section F.2). We can sample from the model by

discretizing the SDE in eq. (1), where _σt_ = _σ_ for the choices of _κt_ = _t, νt_ = 1 _−_ _t_ . However,
to derive a flexible family of samplers where we can independently tune the dispersion _σ_ without
retraining, we exploit Corollary 1 from Singh & Fischer (2024) to introduce a family of SDEs with
the same marginal distributions as that for eq. (1)


          -           _t_ [)] _[σ]_ [2]
_dzt_ = _hθ_ ( _zt, t_ ) _−_ [(1] _[ −]_ _[γ]_ [2] _∇zt_ ln _pt_ ( _zt_ ) _dt_ + _γtσdwt_ (20)

2


Where _γt_ _≥_ 0 can be chosen to control the amount of stochasticity introduced into sampling. For
example, setting _γt_ = 0 yields the probability flow ODE for deterministic sampling. In general,
to use eq. (20) for _γt_ = 1, the score function _∇zt_ ln _pt_ ( _zt_ ) is needed as well. For the interpolant
_zt_ = _σ_ - _t_ (1 _−_ _t_ ) _ϵ_ + _tz_ 1 + (1 _−_ _t_ ) _z_ 0, the score can be estimated using

_∇zt_ ln _pt_ ( _zt_ ) = _−_ _σ_ ~~�~~ E _t_ [(1 _ϵ|z −t_ ] _t_ ) (21)


See section E for the proof. However, for Gaussian _z_ 0, score can be computed from the drift _hθ_ ( _zt, t_ )
(Singh & Fischer, 2024) as following (see section D for details)


_∇x_ ln _pt_ ( _zt_ ) = _−zt_ + _thθ_ ( _zt, t_ ) (22)


Section F provides detailed derivation of samplers for various parameterizations. For classifier free
guided sampling (Ho & Salimans, 2022; Xie et al., 2024; Dao et al., 2023; Zheng et al., 2023; Singh
& Fischer, 2024), we define the guided drift as a linear combination of the conditional drift _hθ_ ( _zt, t, c_ )
and the unconditional drift _hθ_ ( _zt, t, c_ = ∅) as


_h_ [cfg] ( _zt, t, c_ ) _≡_ (1 + _λ_ ) _hθ_ ( _zt, t, c_ ) _−_ _λhθ_ ( _zt, t, c_ = ∅) (23)


where _λ_ is the relative weight of the guidance, _c_ is the conditioning information and _c_ = ∅ denotes
no conditioning. Note that _λ_ = _−_ 1 corresponds to unconditional sampling, _λ_ = 0 corresponds to
conditional sampling and _λ >_ 0 further biases towards the modes of the conditional distribution.


6 EXPERIMENTS


We evaluate LSI on the standard ImageNet (2012) dataset (Deng et al., 2009; Russakovsky et al.,
2015). We train models at various image resolutions and compare their sample quality using the
Frechet Inception Distance (FID) metric (Heusel et al., 2017) for class conditional samples. All
models were trained for 1000 epochs, except for the comparison in table 1 which reports FID at
2000 epochs. All results use deterministic sampler, using _γt_ = 0, unless otherwise specified. A key
implementation detail to note is that the encoder uses normalization and tanh to bound the scale of
the latents. See sections O and P for additional details.


6


|blished as a conference paper at ICLR 2026|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|lished as a conference paper at ICLR 2026|lished as a conference paper at ICLR 2026|lished as a conference paper at ICLR 2026|lished as a conference paper at ICLR 2026|lished as a conference paper at ICLR 2026|lished as a conference paper at ICLR 2026|
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID||||||
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID||||Fix<br>|ed_ c_<br>|
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|||Lea|Lea|rned_ c_|
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID|||Lea|Lea||
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID||||||
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID||||||
|le 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achiev<br> similar to observation space models of comparable size. However, the latent space model <br> fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of th<br>ameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder i<br>d only once, while the latent model L is run repeatedly, once for each sampling step. Therefor<br>OP savings from a computationally cheaper latent model accumulate with sampling steps.<br>FID @ 2K epochs<br># Params (M)<br>Flops (G)<br>Resolution<br>Latent<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>Latent (E/D/L)<br>Observ.<br>64_ ×_ 64<br>2.62<br>2.57<br>392 (5/5/382)<br>398<br>15/15/161<br>201<br>128_ ×_ 128<br>3.12<br>3.46<br>392 (5/5/382)<br>400<br>59/59/327<br>466<br>256_ ×_ 256<br>3.91<br>3.87<br>393 (5/5/383)<br>405<br>240/240/450<br>1288<br><br><br><br><br><br><br><br>0_ ←β_<br>40<br>41<br>42<br>PSNR<br>17<br>18<br><br>4<br>5<br>6<br>Fixed_ c_<br>Learned_ c_<br>FID||||||


0 2 4 6


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|0_ ←β_||||
|||||
|||||


_β_


Encoder Scale _c_


_·_ 10 _[−]_ [2]


Figure 1: **Effect of loss trade-off** _**β**_ **and encoder noise scale** _**c**_ **:** In the left panel, we evaluate the
effect of loss trade-off weight _β_ for 128 _×_ 128 models and observe that FID improves with _β_, until
the degradation in reconstruction quality (PSNR) starts degrading FID. In the right panel, we evaluate
the effect of encoder noise scale on FID. We also plot the FID for a model with learned scale as
dashed line. A deterministic encoder performs the worst ( _c_ = 0), with FID improving with _c_ until it
degrades again. Encoder with learned _c_ (dashed line) is outperformed by fixed _c_ in our experiments.


**LSI** **enables** **joint** **learning** **for** **SI** **:** While SI doesn’t allow latent variables, LSI enables joint
learning of Encoder (E), Decoder (D), and Latent SI models (L). In table 1 we compare FID across
various resolutions for LSI models against SI models trained directly in observation (pixel) space. LSI
models achieve FIDs similar to the observation space models indicating on par performance in terms
of the final FID. Models for both were chosen with similar architecture and number of parameters
and trained for 2000 epochs. Reference comparison with other methods is provided in section R.


**LSI enables computationally cheaper sampling:** In table 1 we also report the parameter counts
(in millions) as well as FLOPs (in Giga) for the observation space SI model as well as E, D and
L models for the LSI. For the latent L model, FLOPs are reported for a single forward pass. First
note that the parameters in LSI are partitioned across the encoder E, the decoder D and the latent L
models. At sampling time, encoder is not used, decoder is used only once, while the latent model is
run multiple times, once for each step of sampling. Therefore, while the overall FLOP count for LSI
and Observation space SI models is similar for a single forward pass, sampling with multiple steps
becomes significantly cheaper. For example, sampling with 100 steps leads to 73 _._ 6% reduction in
FLOPs for sampling 128 _×_ 128 images and 48 _._ 6% for 256 _×_ 256 images.


**Joint learning is beneficial:** In fig. 1(left panel) we plot the FID as the weighting term _β_ is varied
(eq. (19)). A higher _β_ forces the encoder to adapt the latents more for the second term of the loss. We
observe that FID improves as _β_ increases, going from 4 _._ 53 (for _β_ _→_ 0) to 3 _._ 75 ( _≈_ 17% improvement)
for _β_ = 0 _._ 0001, indicating that this adaptation is beneficial for the overall performance. Eventually,
FID worsens as _β_ is increased further. We also plot the reconstruction PSNR for each of these models


7


Table 2: **Joint training helps mitigate capacity shift:** We evaluate the effect of moving first _k_ and
last _k_ convolutional blocks from the latent model L to encoder and decoder respectively, for 128 _×_ 128
resolution models. This results in the overall parameter count staying roughly the same, but the
number of FLOPs required for sampling changing significantly. We observe that the model trained
with _β_ _>_ 0 perform better and maintains FID well, in comparison to the independently trained model
( _β_ _→_ 0), even when capacity is shifted away from the latent model L, resulting in 8 _._ 5% reduction in
FLOPs for sampling from _k_ = 0 to _k_ = 6.


_k_ FID ( _β_ _>_ 0) FID ( _β_ _→_ 0) #Params. (E/D/L) FLOPs (E/D/L)


0 3.76 4.31 392 (5/5/382) 59/59/327
3 3.91 4.55 389 (9/8/372) 68/66/313
6 3.96 4.87 387 (13/12/362) 75/73/299
9 4.61 4.98 383 (16/16/351) 82/80/284


in orange and observe that increasing _β_ essentially trades-off reconstruction quality with generative
performance. For too large a _β_, poor reconstruction quality leads to worsening FID. The dashed line
indicates the performance when the encoder-decoder are trained independently of the latent model,
limit of _β_ _→_ 0. We implement it as a stop gradient operation in implementation, where the gradients
from the second term of the loss are not backpropagated into _z_ 1. To further assess the benefits of joint
training, in table 2 we compare the FIDs between jointly trained model ( _β_ _>_ 0) and independently
trained model ( _β_ _→_ 0) as parameters are shifted from the latent model L to the encoder E and decoder
D models, by moving first _k_ and last _k_ convolutional blocks from the latent model to the encoder and
the decoder respectively. While this keeps the total parameter count roughly the same, the number
of FLOPs required for sampling changes significantly. The jointly trained model performs better
and maintains FID well even when capacity shifts away from the latent model, resulting in 8 _._ 5%
reduction in FLOPs required for sampling from _k_ = 0 to _k_ = 6.


**Encoder** **noise** **scale** **affects** **performance:** The stochasticity of the encoder _pθ_ ( _z_ 1 _|x_ ) has a
significant impact on the performance. We parameterize the encoder as a conditional Gaussian
_N_ ( _z_ 1; _µθ_ ( _x_ ) _,_ Σ _θ_ ( _x_ )) where Σ( _x_ ) is assumed to be diagonal. We experimented with a purely deterministic encoder (Σ _θ_ ( _x_ ) = 0), learned Σ _θ_ ( _x_ ) and constant noise Σ _θ_ ( _x_ ) = _cI_ . In fig. 1(right panel)
we plot FID as the encoder output stochasticity _c_ is varied. Dashed line indicates performance with
learned Σ _θ_ ( _x_ ). A deterministic encoder ( _c_ = 0) performs poorly. FID improves as the noise scale _c_
is increased, until eventually it degrades again. While learned Σ _θ_ ( _x_ ) (dashed line) performs well,
fixed _c_ models achieved higher FID.


**InterpFlow** **parameterization** **performs** **better** **than** **alternatives:** In table 3 we compare different parameterizations discussed in section 4 and section C. The InterpFlow parameterization
consistently led to better FID. Both OrigFlow and NoisePred parameterizations exhibited higher
variance gradients and noisy optimization. While Denoising parameterization resulted in less noisy
training, InterpFlow parameterization led to fastest improvement in FID.


**LSI supports diverse** _p_ 0 **:** In table 4 we report FID achieved by LSI using different prior _p_ 0( _z_ 0)
distributions. While Gaussian _p_ 0 performs the best, other choices for _p_ 0 yield competitive results
indicating that LSI retains one of the key strengths of SI – support for diverse _p_ 0 distributions. See
section N for additional details. To allow flexible sampling using eq. (20), we modified latent SI
model to output extra output channels and augmented the loss with another term to estimate E[ _ϵ|zt_ ].
Equation (21) was used to compute the score and sample with the deterministic sampler using _γt_ = 0.


**LSI supports flexible sampling:** In fig. 2 and fig. 3 we qualitatively demonstrate flexible sampling
with LSI model for popular use cases. Figure 2 demonstrates compatibility of classifier free guidance
(CFG) with LSI, using eq. (22). Increasing guidance weight _λ_ results in more typical samples. First
_z_ 0 is sampled from _p_ 0( _z_ 0), Gaussian in this example, following which eq. (20) is simulated forward
in time, using class conditional drift with different guidance weights _λ_ . In fig. 3 a given ‘Original’
image (shown leftmost) is first encoded to yield it’s representation _z_ 1, which is then inverted by
simulating probability flow ODE (setting _γt_ = 0 in eq. (20)) backward in time from _t_ = 1 to _t_ = 0,
yielding _z_ 0 (similar to DDIM inversion (Song et al., 2020a)). Using this _z_ 0 as starting point, eq. (20)


8


Table 3: **Effect of parameterization:** We compare various parameterization schemes at 128 _×_
128 resolution. InterpFlow parameterization
performs better against the alternatives.


Parameterization FID @1K epochs


OrigFlow 4.56
NoisePred 4.73
Denoising 4.28
InterpFlow 3.76


Table 4: **LSI supports diverse** _**p**_ **0** **:** LSI retains
one of the key strengths of SI – support for arbitrary _p_ 0 distribution. Different _p_ 0 achieve competetive FID for 128 _×_ 128 resolution model.


_p_ 0 FID @1K epochs


Uniform 4.81
Laplacian 4.45
Gaussian 3.76


Gaussian Mixture 4 _._ 26


_λ_ = 0 _._ _λ_ = 1 _._ _λ_ = 3 _._ _λ_ = 5 _._ _λ_ = 0 _._ _λ_ = 1 _._ _λ_ = 3 _._ _λ_ = 5 _._


Figure 2: **LSI supports CFG sampling.** Class conditional samples are visualized with increasing
guidance weight _λ_ leading to more typical samples for the class. See text for details.


is simulated forward is time using _γt_ _≡_ _γ_ (1 _−_ _t_ ) for different values of _γ_ . We show three samples
for each value of _γ_ and observe increasing diversity with increasing _γ_ . See section Q for additional
details and results.


7 RELATED WORK


Latent Stochastic Interpolants (LSI) draw from insights in diffusion models, latent variable models,
and continuous-time generative processes. We discuss key works from these areas in the following.


**Diffusion Models:** Diffusion models, originating from foundational work on score matching (Vincent,
2011; Song & Ermon, 2019) and early variational formulation (Sohl-Dickstein et al., 2015), gained
prominence with Denoising Diffusion Probabilistic Models (DDPMs) (Ho et al., 2020). Subsequent
improvements focused on architectural choices and learned variances (Nichol & Dhariwal, 2021),
faster sampling via Denoising Diffusion Implicit Models (DDIMs) (Song et al., 2020a), progressive
distillation (Salimans & Ho, 2022), and powerful conditional generation through techniques like
classifier-free guidance (Ho & Salimans, 2022). Further exploration of the design space (Karras et al.,
2022; 2024) has lead to highly performant models. More recently, diffusion inspired consistency
models (Song et al., 2023) have emerged, offering efficient generation. LSI complements these with
a flexible method for jointly learning in a latent space using richer prior distributions.


**Latent Variable Models and Expressive Priors:** Variational Autoencoders (VAEs) (Kingma et al.,
2013; Rezende et al., 2014) learn a compressed representation _z_ of data _x_, but are limited by the
expressiveness of the prior _p_ ( _z_ ) (NVAE (Vahdat & Kautz, 2020), LSGM (Vahdat et al., 2021)), as
they typically use simple priors (e.g., isotropic Gaussian). LSI addresses this by jointly learning a
flexible generative process in the latent space, enabling powerful transformations of the simple prior.
Early work (Sohl-Dickstein et al., 2015) derived ELBO for discrete time diffusion models, while
Variational Diffusion Models (VDM) (Kingma et al., 2021) interpret diffusion models as a specific
type of VAE with Gaussian noising process. In contrast, while LSI also optimizes an ELBO, it allows
for a broader choice of the prior _p_ ( _z_ 0) and the transforms mapping the prior to the learned aggregated
posterior. Our work is similar in spirit to models like NVAE, which employed deep hierarchical latent
representations, and LSGM, which proposed training score-based models in the latent space of a
VAE, but offers a flexible framework similar to SI allowing a rich family of priors and latent space


9


Original _γ_ = 0 _._ 25 _γ_ = 0 _._ 5 _γ_ = 1 _._ 0


Figure 3: **LSI supports flexible sampling.** We demonstrate inversion of an ‘Original’ image, using
reverse probability flow ODE (similar to DDIM inversion), followed by forward stochastic sampling
to yield samples similar to it, with diversity increasing with _γ_ (eq. (20)). See text for details.


dynamics. Note that LDM (Rombach et al., 2022) train a diffusion generative model in the latent
space of a _fixed_ encoder-decoder pair – making their latents actually _observed_ from the point of view
of generative modeling.


**Continuous-Time Generative Processes:** While diffusion models have been formulated and studied
using continuous time dynamics (Song et al., 2020b;a; Kingma et al., 2021; Vahdat et al., 2021), their
relation to Continuous Normalizing Flows (CNFs) (Chen et al., 2018; Grathwohl et al., 2019) offers
another perspective on continuous-time transformations. Early training challenges with the CNFs
have been addressed by newer methods like Flow Matching (FM) (Lipman et al., 2022; Xu et al.,
2022), Conditional Flow Matching (CFM) (Neklyudov et al., 2023; Tong et al., 2023), and Rectified
Flow (Liu et al., 2022). These approaches propose simulation-free training by regressing vector
fields of fixed conditional probability paths. However, likelihood control is typically not possible
(Albergo et al., 2023), consequently extension to jointly learning in latent space is ill-specified. In
contrast, LSI optimizes an ELBO, offering likelihood control along with joint learning in a latent
space. Stochastic Interpolants (SI) (Albergo et al., 2023) provides a unifying perspective on generative
modeling, capable of bridging _any_ two probability distributions via a continuous-time stochastic
process, encompassing aspects of both flow-based and diffusion-based methods. While SI formulates
learning the velocity field and score function directly in the observation space using pre-specified
stochastic interpolants, LSI arrives at a similar objective in the latent space, as part of the ELBO, from
the specific choices of the approximate variational posterior. LSI reduces to SI when encoder and
decoder are chosen to be Identity functions. SI is related to the Optimal Transport and the Schrödinger
Bridge problem (SBP) which have been explored as a basis for generative modeling (De Bortoli et al.,
2021; Wang et al., 2021; Shi et al., 2023). While LSI learns a transport, its primary objective is data
log-likelihood maximization via the ELBO, rather than solving a specific OT or SBP.


8 CONCLUSION


In this paper, we introduced Latent Stochastic Interpolants (LSI), generalizing Stochastic Interpolants
to enable joint end-to-end training of an encoder, a decoder, and a generative model operating
entirely within the learned latent space. LSI overcomes the limitation of simple priors of the normal
diffusion models and mitigates the computational demands of applying SI directly in high-dimensional
observation spaces, while preserving the generative flexibility of the SI framework. LSI leverage
SDE-based Evidence Lower Bound to offer a principled approach for optimizing the entire model.
We validate the proposed approach with comprehensive experimental studies on standard ImageNet
benchmark. Our method offers scalability along with a unifying perspective on continuous-time
generative models with dynamic latent variables. However, to achieve scalable training, our approach
makes simplifying assumptions for the variational posterior approximation. While restrictive, and
common with other methods, these assumptions do not seem to limit the empirical performance.


ACKNOWLEDGMENTS


We would like to thank Kevin J. Shih and Ian Fischer for proofreading early drafts of this manuscript
and providing valuable feedback.


10


REPRODUCIBILITY STATEMENT


We have included detailed proofs of all the key theoretical results in the appendix. Sections 6 and O
provide key training and evaluation setup details. Section P provides the necessary architecture
details to reproduce the models used in the experiments. Section Q provides additional sampling
setup details.


REFERENCES


Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying
framework for flows and diffusions. _arXiv preprint arXiv:2303.08797_, 2023.


Alexander Alemi, Ben Poole, Ian Fischer, Joshua Dillon, Rif A Saurous, and Kevin Murphy. Fixing a
broken elbo. In _International conference on machine learning_, pp. 159–168. PMLR, 2018.


Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary
differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and
R. Garnett (eds.), _Advances in Neural Information Processing Systems_, volume 31. Curran Associates, Inc., 2018. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)
[2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)


Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow matching in latent space. _arXiv preprint_
_arXiv:2307.08698_, 2023.


Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger
bridge with applications to score-based generative modeling. _Advances in Neural Information_
_Processing Systems_, 34:17695–17709, 2021.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pp. 248–255. Ieee, 2009.


Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. _Advances_
_in neural information processing systems_, 34:8780–8794, 2021.


Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. Scalable reversible
generative models with free-form continuous dynamics. In _International Conference on Learning_
_Representations_, 2019. [URL https://openreview.net/forum?id=rJxgknCcK7.](https://openreview.net/forum?id=rJxgknCcK7)


Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in neural_
_information processing systems_, 30, 2017.


Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick,
Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a
constrained variational framework. In _International conference on learning representations_, 2017.


Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv preprint arXiv:2207.12598_,
2022.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in_
_neural information processing systems_, 33:6840–6851, 2020.


Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for
high resolution images. In _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 13213–13232.
PMLR, 2023.


Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans.
Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. _arXiv_ _preprint_
_arXiv:2410.19324_, 2024.


Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation.
_arXiv preprint arXiv:2212.11972_, 2022.


11


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _Advances in Neural Information Processing Systems_, 35:26565–26577,
2022.


Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing
and improving the training dynamics of diffusion models. In _Proceedings_ _of_ _the_ _IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition_, pp. 24174–24184, 2024.


Patrick Kidger, James Foster, Xuechen Chen Li, and Terry Lyons. Efficient and accurate gradients
for neural sdes. _Advances in Neural Information Processing Systems_, 34:18747–18761, 2021.


Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka,
Yuki Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a one-step generator from
a low-resolution diffusion teacher. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 37:
19167–19208, 2024.


Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data
augmentation. _Advances in Neural Information Processing Systems_, 36:65484–65516, 2023.


Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. _Advances_
_in neural information processing systems_, 34:21696–21707, 2021.


Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv preprint_
_arXiv:1412.6980_, 2014.


Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.


Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients
for stochastic differential equations. In _International Conference on Artificial Intelligence and_
_Statistics_, pp. 3870–3882. PMLR, 2020.


Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching
for generative modeling. _arXiv preprint arXiv:2210.02747_, 2022.


Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow. _arXiv preprint arXiv:2209.03003_, 2022.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv_ _preprint_
_arXiv:1711.05101_, 2017.


Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning
stochastic dynamics from samples. In _International conference on machine learning_, pp. 25858–
25889. PMLR, 2023.


Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.
In _International conference on machine learning_, pp. 8162–8171. PMLR, 2021.


Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and
approximate inference in deep generative models. In _International conference on machine learning_,
pp. 1278–1286. PMLR, 2014.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Confer-_
_ence on Computer Vision and Pattern Recognition (CVPR)_, pp. 10684–10695, June 2022.


Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet
Large Scale Visual Recognition Challenge. _International Journal of Computer Vision (IJCV)_, 115
(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y.


Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. _arXiv_
_preprint arXiv:2202.00512_, 2022.


Simo Särkkä and Arno Solin. _Applied_ _stochastic_ _differential_ _equations_, volume 10. Cambridge
University Press, 2019.


12


Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger
bridge matching. _Advances in Neural Information Processing Systems_, 36:62183–62223, 2023.


Saurabh Singh and Ian Fischer. Stochastic sampling from deterministic flow models. _arXiv preprint_
_arXiv:2410.02217_, 2024.


Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. In _International conference on machine learning_,
pp. 2256–2265. PMLR, 2015.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv_
_preprint arXiv:2010.02502_, 2020a.


Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
_Advances in neural information processing systems_, 32, 2019.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint_
_arXiv:2011.13456_, 2020b.


Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.


Evangelos A Theodorou. Nonlinear stochastic control and information theoretic dualities: Connections, interdependencies and thermodynamic interpretations. _Entropy_, 17(5):3352–3375, 2015.


Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian
Fatras, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models
with minibatch optimal transport. _arXiv preprint arXiv:2302.00482_, 2023.


Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. _Advances in neural_
_information processing systems_, 33:19667–19679, 2020.


Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space.
_Advances in neural information processing systems_, 34:11287–11302, 2021.


Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural computa-_
_tion_, 23(7):1661–1674, 2011.


Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via
schrödinger bridge. In _International conference on machine learning_, pp. 10794–10804. PMLR,
2021.


Tianyu Xie, Yu Zhu, Longlin Yu, Tong Yang, Ziheng Cheng, Shiyue Zhang, Xiangyu Zhang, and
Cheng Zhang. Reflected flow matching. _arXiv preprint arXiv:2405.16577_, 2024.


Yilun Xu, Ziming Liu, Max Tegmark, and Tommi Jaakkola. Poisson flow generative models.
_Advances in Neural Information Processing Systems_, 35:16782–16795, 2022.


Yilun Xu, Gabriele Corso, Tommi Jaakkola, Arash Vahdat, and Karsten Kreis. Disco-diff: Enhancing
continuous diffusion models with discrete latents. _arXiv preprint arXiv:2407.03300_, 2024.


Qinqing Zheng, Matt Le, Neta Shaul, Yaron Lipman, Aditya Grover, and Ricky TQ Chen. Guided
flows for generative modeling and decision making. _arXiv preprint arXiv:2311.13443_, 2023.


Juntang Zhuang, Nicha C Dvornek, Sekhar Tatikonda, and James S Duncan. Mali: A memory
efficient and reverse accurate integrator for neural odes. _arXiv preprint arXiv:2102.04668_, 2021.


13


APPENDIX


A PROOF OF ELBO FOR DYNAMIC LATENT VARIABLES (EQ. (3))


We provide a self-contained proof of the variational lower bound stated in eq. (3). The proof is based
on the approach of Li et al. (2020), but in a more general form. We first establish the path-space
KL divergence between two diffusion processes via Girsanov’s theorem (Theorem 1), then use it to
derive the ELBO (Theorem 2). Let P _θ_ be the path measure under the model as in eq. (1) and let Q be
the path measure under the variational posterior process approximation as in eq. (2).

**Theorem 1 (Path-space KL via Girsanov’s theorem):** Consider two SDEs, starting at the same
starting point _z_ ˜0 = _z_ 0, sharing the same dispersion coefficient _σ_ ( _zt, t_ ) but potentially different initial
distributions — _z_ 0 _∼_ _q_ 0( _z_ 0) for Q and _z_ 0 _∼_ _p_ 0( _z_ 0) for P _θ_ :


_dz_ ˜ _t_ = _hθ_ (˜ _zt, t_ ) _dt_ + _σ_ (˜ _zt, t_ ) _dwt,_ (model, path measure P _θ_ ) (eq. 1)
_dzt_ = _hϕ_ ( _zt, t_ ) _dt_ + _σ_ ( _zt, t_ ) _dwt,_ (variational, path measure Q) (eq. 2)


Define _u_ ( _zt, t_ ) by _σ_ ( _zt, t_ ) _u_ ( _zt, t_ ) = _hϕ_ ( _zt, t_ ) _−_ _hθ_ ( _zt, t_ ) (eq. (5)). Then:


KL(Q _∥_ P _θ_ ) = KL( _q_ 0 _∥p_ 0) + [1] 2 [E][Q]


�� 1 
_∥u_ ( _zt, t_ ) _∥_ [2] _dt_ (24)
0


When _q_ 0 = _p_ 0 (both processes share the same initial distribution, potentially learnable), the initial
KL vanishes and the path-space KL reduces to the dynamics mismatch alone.


_Proof._ The full path measure factorizes as the initial distribution times the conditional path measure
given the initial state. The Radon-Nikodym derivative therefore decomposes as


_[q]_ [0][(] _[z]_ [0][)] _[d]_ [Q] _[z]_ [0]

_p_ 0( _z_ 0) _[·]_ _d_ P _[z]_ [0]


_d_ Q
( _Z_ ) = _[q]_ [0][(] _[z]_ [0][)]
_d_ P _θ_ _p_ 0( _z_ 0)


( _Z_ ) (25)
_d_ P _[z]_ _θ_ [0]


where Q _[z]_ [0] and P _[z]_ _θ_ [0] [denote] [the] [conditional] [path] [measures] [given] [the] [initial] [state] _[z]_ [0][.] [Next] [we] [use]
Girsanov’s theorem to evaluate the second factor. Under Q, the process satisfies _dzt_ = _hϕ dt_ + _σ dwt_ [Q]
where _wt_ [Q] [is a standard Brownian motion.] [Define the][ P] _[θ]_ [-Brownian motion via] _[ dw]_ _t_ [P] _[θ]_ = _σ_ _[−]_ [1] ( _dzt −_
_hθ dt_ ). Substituting the Q-dynamics for _dzt_ :


_dwt_ [P] _[θ]_ = _dwt_ [Q] [+] _[ u]_ [(] _[z][t][, t]_ [)] _[ dt]_ (26)


That is, under Q, the process _wt_ [P] _[θ]_ acquires a drift _ut_ . By Girsanov’s theorem:


_d_ Q _[z]_ [0] �� 1

= exp
_d_ P _[z]_ _θ_ [0] 0


_d_ Q _[z]_ [0]


_u_ _[T]_ _t_ _[dw]_ _t_ [P] _[θ]_ _−_ [1]
0 2


2


- 1 
_∥ut∥_ [2] _dt_ (27)
0


Substituting _dwt_ [P] _[θ]_ = _dwt_ [Q] [+] _[ u][t]_ _[dt]_ [ and combining with the initial density ratio:]


ln _[d]_ [Q]


_[d]_ [Q] = ln _[q]_ [0][(] _[z]_ [0][)]

_d_ P _θ_ _p_ 0( _z_ 0)


_[q]_ [0][(] _[z]_ [0][)] - 1

_p_ 0( _z_ 0) [+] 0


_u_ _[T]_ _t_ _[dw]_ _t_ [Q] [+] [1]
0 2


2


- 1

_∥ut∥_ [2] _dt_ (28)
0


Taking the expectation under Q:


+ [1]

2 [E][Q]


+ [1]


       
KL(Q _∥_ P _θ_ ) = E _q_ 0 ln _p_ _[q]_ [0] 0 [(] ( _[z]_ _z_ [0] 0 [)] )


�� 1


�� 1 
_∥ut∥_ [2] _dt_ (29)
0


+ EQ


�� 1

_u_ _[T]_ _t_ _[dw]_ _t_ [Q]
0


- �� KL( _q_ 0 _∥p_ 0)


- �� = 0


The Itô integral �01 _[u]_ _t_ _[T]_ _[dw]_ _t_ [Q] [is] [a] [martingale] [under] [Q] [(under] [the] [standard] [integrability] [condition]
EQ[�01 _[∥][u][t][∥]_ [2] _[ dt]_ []] _[ <][ ∞]_ [), and the expectation of a martingale starting at zero is zero.]


**Remark 1:** Theorem 1 applies to any two diffusion processes sharing the same dispersion, regardless
of whether the state space represents latent variables or observations. In particular, it applies directly
in observation space (with _z_ replaced by _x_ ).


14


**Remark 2 (Learnable prior):** When the prior _p_ 0 is parameterized (e.g., _pθ_ ( _z_ 0) = _N_ ( _µθ,_ Σ _θ_ )), the
natural construction uses the same learnable prior for both processes ( _q_ 0 = _pθ_ ), so KL( _q_ 0 _∥p_ 0) =
0 and the ELBO retains the same form. The prior parameters are still learned: they affect the
distribution of _z_ 0 in the path integral EQ[� _∥u∥_ [2] _dt_ ], and gradients flow through _z_ 0 _∼_ _pθ_ ( _z_ 0) via the
reparameterization trick. Alternatively, if the variational process uses a fixed reference _q_ 0 = _pθ_, the
KL( _q_ 0 _∥pθ_ ) term appears as an additional regularizer penalizing deviation from the reference.
**Theorem 2 (ELBO for dynamic latent variables, eqs. (3) and (5)):** Under the setup of Theorem 1,
with potentially learnable initial distributions _p_ 0 and _q_ 0, let _xti_ be observations at times _ti_ _∈_ [0 _,_ 1],
_i_ = 1 _, . . ., n_, assumed to depend on the latent state only through _zti_, i.e., _pθ_ ( _xti|Z_ ) = _pθ_ ( _xti|zti_ ).
Then:


_[q]_ [0][(] _[z]_ [0][)]

_p_ 0( _z_ 0) _[−]_ [1] 2


- 1


ln _pθ_ ( _xt_ 1 _, . . ., xtn_ ) _≥_ EQ


= EQ


- _n_


- ln _pθ_ ( _xti|zti_ ) _−_ ln _p_ _[q]_ [0] 0 [(] ( _[z]_ _z_ [0] 0 [)] )

_i_ =1


ln _pθ_ ( _xti|zti_ )

_i_ =1


2


_∥u_ ( _zt, t_ ) _∥_ [2] _dt_
0


(30)


- _n_


_−_ KL(Q _∥_ P _θ_ ) (31)


When _q_ 0 = _p_ 0, the second term on the right vanishes and we recover the special case of above as
stated in Li et al. (2020).


_Proof._ Under the model, the latent path evolves according to P _θ_ (eq. (1)) and observations are
generated conditionally at each time _ti_ . The marginal likelihood is obtained by integrating over all
latent paths:

_n_

                 
        _pθ_ ( _xt_ 1 _, . . ., xtn_ ) = _pθ_ ( _xti|zti_ ) _d_ P _θ_ ( _Z_ ) (32)


_i_ =1

Since both P _θ_ and Q share the same dispersion and initial distribution, they are mutually absolutely
continuous (by Girsanov’s theorem, under standard regularity). We can therefore re-express the
integral using the variational path measure Q (eq. (2)) as a proposal:


- _n_

 


_d_ Q [(] _[Z]_ [)]


_pθ_ ( _xt_ 1 _, . . ., xtn_ ) = EQ


- _pθ_ ( _xti|zti_ ) _·_ _[d]_ [P] _[θ]_

_d_ Q

_i_ =1


Taking the logarithm of both sides and using Jensen’s inequality:


_d_ Q [(] _[Z]_ [)]


(33)


(34)


ln _pθ_ ( _xt_ 1 _, . . ., xtn_ ) _≥_ EQ


- _n_


- ln _pθ_ ( _xti|zti_ ) + ln _[d]_ [P] _[θ]_

_d_ Q

_i_ =1


From the proof of Theorem 1 (with _q_ 0 = _p_ 0, so the initial density ratio cancels), the log RadonNikodym derivative expressed in terms of the Q-Brownian motion is:


[P] _[θ]_

_[q]_ [0][(] _[z]_ [0][)]
_d_ Q [(] _[Z]_ [) =] _[ −]_ [ln] _p_ 0( _z_ 0)


_u_ _[T]_ _t_ _[dw]_ _t_ [Q] _[−]_ [1]
0 2


2


ln _[d]_ [P] _[θ]_


_[q]_ [0][(] _[z]_ [0][)] - 1

_p_ 0( _z_ 0) _[−]_ 0


- 1

_∥ut∥_ [2] _dt_ (35)
0


Substituting the above we get:


_[q]_ [0][(] _[z]_ [0][)] - 1

_p_ 0( _z_ 0) _[−]_ 0


_u_ _[T]_ _t_ _[dw]_ _t_ [Q] _[−]_ [1]
0 2


- 1


(36)


(37)


ln _pθ_ ( _xt_ 1 _, . . ., xtn_ ) _≥_ EQ


= EQ


= EQ


- _n_


- ln _pθ_ ( _xti|zti_ ) _−_ ln _p_ _[q]_ [0] 0 [(] ( _[z]_ _z_ [0] 0 [)] )

_i_ =1


_n_

- ln _pθ_ ( _xti|zti_ ) _−_ ln _p_ _[q]_ [0] 0 [(] ( _[z]_ _z_ [0] 0 [)] )

_i_ =1


ln _pθ_ ( _xti|zti_ )

_i_ =1


_∥ut∥_ [2] _dt_
0


- _n_


_[q]_ [0][(] _[z]_ [0][)]

_p_ 0( _z_ 0) _[−]_ [1] 2


2


- 1


2


_∥ut∥_ [2] _dt_
0


- _n_


_−_ KL(Q _∥_ P _θ_ ) (38)


Where the Itô integral �01 _[u]_ _t_ _[T]_ _[dw]_ _t_ [Q] [vanishes under][ E][Q] [(as established in Theorem 1).]


**Remark** **3:** The bound has a natural interpretation: the first term is a reconstruction likelihood
(how well the model explains observations given the latent path) and the second term penalizes the
mismatch between the variational and model path distributions. [1] 2 �01 _[∥][u][t][∥]_ [2][ can also be seen as the]

control cost required to steer the model process P _θ_ to match the variational process Q (Theodorou,
2015). The bound is tight when Q = P _θ_ ( _· | xt_ 1 _, . . ., xtn_ ), i.e., when the variational process equals
the true posterior process.


15


B OBSERVATION-SPACE STOCHASTIC INTERPOLANTS


The LSI framework (Section 3) jointly trains an encoder, decoder, and latent generative model. Here
we consider the special case where the generative process operates directly in observation space,
without an encoder or decoder. This corresponds to setting _zt_ _≡_ _xt_ for all _t_, making the latent process
identical to the observation process.


**Setup:** The generative model is an SDE directly in observation space,


_dx_ ˜ _t_ = _hθ_ (˜ _xt, t_ ) _dt_ + _σt d_ ˜ _wt,_ _x_ ˜0 _∼_ _p_ 0( _x_ 0) (39)

i.e., eq. (1) with _z_ _→_ _x_ . The prior _p_ 0( _x_ 0) may be fixed or learnable (e.g., _pθ_ ( _x_ 0) = _N_ ( _µθ,_ Σ _θ_ )). The
marginal at _t_ = 1 defines the model distribution _pθ_ ( _x_ 1).


The variational process Q is constructed using the same diffusion bridge machinery as in Section 2.2,
now bridging _p_ 0( _x_ 0) and _p_ 1( _x_ 1) = _p_ data( _x_ 1) directly in observation space. As in the previous section,
both Q and P _θ_ can use potentially different initial distributions _q_ 0 and _p_ 0. Starting from the linear
SDE _dxt_ = _htxt dt_ + _σt dwt_ (cf. eq. (7)) with _z_ _→_ _x_ ) and applying Doob’s h-transform to condition
on ending at _x_ 1 _∼_ _p_ 1, the drift is (cf. eq. (9)):

_hϕ_ ( _xt, t_ ) = _htxt_ + _σt_ [2] _[∇][x]_ _t_ [ln] _[ p]_ [(] _[x]_ [1] _[|][ x][t]_ [)] (40)

By construction, Q has marginal _p_ data = _p_ 1 at _t_ = 1, while P _θ_ has marginal _pθ_ ( _x_ 1).


**ELBO:** Since _p_ data = _p_ 1 and _x_ 1 is a deterministic function of the path _X_ _∼_ Q, the data processing
inequality gives


KL( _p_ 1 _∥pθ_ ) _≤_ KL(Q _∥_ P _θ_ ) (41)


Expanding the left side and rearranging yields the evidence lower bound:


E _p_ 1[ln _pθ_ ( _x_ 1)] _≥_ E _p_ 1[ln _p_ 1( _x_ 1)] _−_ KL(Q _∥_ P _θ_ ) (42)
= _−_ H[ _p_ 1] _−_ KL(Q _∥_ P _θ_ ) (43)


Where H[ _p_ 1] is the entropy of the data distribution _p_ 1. Using Theorem 1, the path-space KL can be
evaluated via Girsanov’s theorem as (with _z_ _→_ _x_ ):


KL(Q _∥_ P _θ_ ) = KL( _q_ 0 _∥p_ 0) + [1] 2 [E][Q]


�� 1 
_∥u_ ( _xt, t_ ) _∥_ [2] _dt_ (44)
0


where _σt u_ ( _xt, t_ ) = _hϕ_ ( _xt, t_ ) _−_ _hθ_ ( _xt, t_ ), as in eq. (5). If _q_ 0 = _p_ 0, the first term on the right vanishes,
leaving only the dynamics cost. As in main text, we assume _q_ 0 = _p_ 0 in the following.


**Simulation-free** **training:** All the simulation-free machinery from section 3 carries over with
_z_ _→_ _x_ . Using the observation-space interpolant (cf. eq. (12)):


_xt_ = _ηt ϵ_ + _κt x_ 1 + _νt x_ 0 _,_ _ϵ ∼N_ (0 _, I_ ) (45)


_u_ ( _xt, t_ ) takes the following form – similar to the result in section 3 (cf. eq. (15)):


_u_ ( _xt, t_ ) = _σt_ _[−]_ [1] �� _dηdtt_ _[−]_ 2 _[σ]_ _ηt_ [2] _t_


_ϵ_ + _[dκ][t]_


       
_[t]_

(46)
_dt_ _[x]_ [0] _[ −]_ _[h][θ]_ [(] _[x][t][, t]_ [)]


_[t]_

_[dν][t]_
_dt_ _[x]_ [1][ +] _dt_


**Final objective:** Substituting (44) and (46) into (43), the observation-space ELBO is:

_−_ H[ _p_ 1] _−_ [1] 2 [E] _[t][∼U]_ [[0] _[,]_ [1]] _[, x]_ [1] _[∼][p]_ [1] _[, x]_ [0] _[∼][p]_ [0] _[, ϵ][∼N]_ [ (0] _[,I]_ [)]      - _∥u_ ( _xt, t_ ) _∥_ [2][�] (47)

where the entropy H[ _p_ 1] of the data distribution is a constant and is independent of the model
parameters. For the linear choice _κt_ = _t_, _νt_ = 1 _−_ _t_, this specializes to the following loss to be
minimized (cf. eq. (17)):





�����


2 []


_βt_


2 [E]


_σ_
�����





- _t_
1 _−_ _t_ _[ϵ]_ [ +] _[ x]_ [1] _[ −]_ _[x]_ [0] _[ −]_ _[h][θ]_ [(] _[x][t][, t]_ [)]


 (48)


where _βt_ has the same interpretation as in eq. (17), of a generalized weighting term, and the constant
term has been dropped.


16


**Remark 4:** Comparing with the LSI loss (eq. (17)), the observation-space ELBO is precisely the
LSI objective with the reconstruction term _−_ ln _pθ_ ( _x_ 1 _|_ _z_ 1) removed and _z_ _→_ _x_ throughout. This
confirms the consistency of the framework: LSI reduces to observation-space stochastic interpolants
when the encoder and decoder are identity functions. All parameterizations (Section 4) and sampling
procedures (Section 5) apply directly with _z_ _→_ _x_ .


**Remark** **5** **(Learnable** **prior):** The ELBO in eq. (47) supports a learnable prior _pθ_ ( _x_ 0) without
modification. If both Q and P _θ_ start from the same _pθ_ ( _x_ 0), the initial KL vanishes regardless
of the prior’s form (Theorem 1). The prior parameters are still learned through the interpolant
_xt_ = _ηtϵ_ + _κtx_ 1 + _νtx_ 0 and the target _[dν]_ _dt_ _[t]_ _[x]_ [0][ in eq. (46), providing gradients via the reparameterization]

trick. Note that the interpolant coefficients _ηt, κt, νt_ depend only on the base SDE parameters _ht, σt_,
not on _p_ 0, so changing the prior affects only the sampling distribution of _x_ 0 — not the interpolant
structure.


C PARAMETERIZATIONS


For the linear choice of _κt_ = _t, νt_ = 1 _−_ _t_ (section J) used for experiments in this paper, the loss
term with _u_ ( _zt, t_ ) is


�����


2


1
E _t∼U_ [0 _,_ 1]E _p_ ( _x_ 1 _,z_ 0 _,z_ 1)E _p_ ( _zt|z_ 1 _,z_ 0) 2 _σ_ [2]


_−σ_
�����


- _t_
1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


(49)


Where _ϵ_ _∼_ _N_ (0 _, I_ ). If _z_ 0 is also Gaussian, _z_ 0 _∼_ _N_ (0 _, I_ ), we can combine _ϵ, z_ 0 to yield _zt_ =
_tz_ 1 + ~~�~~ (1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _z_ 0 and rewrite the above as


1
E _t∼U_ [0 _,_ 1]E _p_ ( _x_ 1 _,z_ 0 _,z_ 1)E _p_ ( _zt|z_ 1 _,z_ 0) 2 _z_ 1 _−_

�����


- _σ_ [2] _t_ + 1 _−_ _t_


�����


2


(50)


_z_ 0 _−_ _hθ_ ( _zt, t_ )
1 _−_ _t_


Directly using aboNaNs due to the _[√]_ ve for1 _−_ _t_ in the denominator.ms leads to high variance in gradients and unreliable training with frequentConsequently, we consider alternative parameterizations
as discussed in the following. Two of the parameterizations OrigFlow and InterpFlow are applicable
for arbitrary _p_ 0, while the remaining two Denoising and NoisePred are applicable when _z_ 0 is
Gaussian. For each of these parameterizations, we also derive the corresponding sampler in section F


C.1 OrigFlow


With straightforward manipulation of the term inside the expectation we arrive at


2
_tϵ −_ _h_ [ˆ] _θ_ ( _zt, t_ ) (51)
���


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) _−_ _σ_


1 1
2 _σ_ [2] 1 _−_ _t_


_√_
���


where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ _[√]_ 1 _−_ _thθ_ ( _zt, t_ ). We rewrite above in terms of a time dependent weighting
_βt_ _≡_ _σ_ [2] (11 _−t_ ) [as following.]


2
_tϵ −_ _h_ [ˆ] _θ_ ( _zt, t_ ) (52)
���


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) _−_ _σ_


_βt_

2


_√_
���


When _z_ 0 is Gaussian, we can rewrite as


2

  1 _−_ _tz_ 1 _−_ _σ_ [2] _t_ + 1 _−_ _tz_ 0 _−_ _h_ [ˆ] _θ_ ( _zt, t_ )��� (53)


_βt_

2


_√_
���


_√_
This objective can be viewed as estimating _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ E[ _[√]_ 1 _−_ _tz_ 1 _−_ _σ_ [2] _t_ + 1 _−_ _tz_ 0 _|zt_ ] with a

time _t_ dependent weighting _βt_ .


17


C.2 InterpFlow


Again, starting with the loss term with _u_ ( _zt, t_ ) and straightforward manipulations we arrive at the
parameterization


�����


2


1
2 _σ_ [2]


_−σ_
�����


- _t_
1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


(54)


(55)


_−σ_
�����


- _t_
1 _−_ _t_ _[z][t][ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


�����


2


~~�~~ _t_
1 _−_ _t_ _[z][t][ −]_


1
=
2 _σ_ [2]


- _t_
1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0][ +]


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) +


2
_tzt −_ _h_ [ˆ] _θ_ ( _zt, t_ )��� (56)


= _[β][t]_


2


_√_
_−σ_
���


_√_
_tϵ_ +


_√_
Where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_


_√_
Where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ _tzt_ + _[√]_ 1 _−_ _thθ_ ( _zt, t_ ) and _βt_ _≡_ _σ_ [2] (11 _−t_ ) [.] [To gain insights into this parameteri-]

zation, let’s consider the term inside the norm and substitute _zt_


_√_

_−_ _σ_


_√_
_tϵ_ +


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) +


_√_
1 _−_ _t_ ( _z_ 1 _−_ _z_ 0) +


_√_
= _−σ_


_t_ ( _tz_ 1 + (1 _−_ _t_ ) _z_ 0 + _σ_ ~~�~~ _t_ (1 _−_ _t_ ) _ϵ_ ) (58)


_√_
1 _−_ _t_ ) _z_ 0 + _σ_ ( _t_


_√_
_t_ ) _z_ 1 + (


_tzt_ (57)


_√_
1 _−_ _t_ + _t_


_√_
_t_ (1 _−_ _t_ ) _−_


_t_ ) _ϵ_ (59)


_√_
= (


_√_
_tϵ_ +


_√_
1 _−_ _t −_


Leading to


_√_
_t_ (1 _−_ _t_ ) _−_


_√_
1 _−_ _t_ + _t_ _t_


_√_
_t_ ) _z_ 1 + ( _t_


_√_
1 _−_ _t_ ) _z_ 0 + _σ_ ( _t_


_√_ 2
1 _−_ _t −_ _t_ ) _ϵ −_ _h_ [ˆ] _θ_ ( _zt, t_ ) (60)

���


_βt_

2


_√_
(
���


_√_
_t_ ) _z_ 1 + (


_√_
The term ( _[√]_ 1 _−_ _t_ + _t_


_√_
_t_ (1 _−_ _t_ ) _−_ _[√]_ 1 _−_ _t_ ) _z_ 0 + _σ_ ( _t_ _[√]_ 1 _−_ _t −_


The term ( 1 _−_ _t_ + _t_ _t_ ) _z_ 1 + ( _t_ (1 _−_ _t_ ) _−_ 1 _−_ _t_ ) _z_ 0 + _σ_ ( _t_ 1 _−_ _t −_ _t_ ) _ϵ_ reduces to _z_ 1 _−_ _z_ 0 at

_t_ = 0 and _z_ 1 _−_ _σϵ_ at _t_ = 1. Since this term appears to interpolate between the two, we refer to this
parameterization as InterpFlow. When _z_ 0 is also Gaussian, we can combine _ϵ, z_ 0 and rewrite as


2
_t_ ) _z_ 1 + (� _t_ (1 _−_ _t_ ) _−_ 1)� _σ_ [2] _t_ + 1 _−_ _tz_ 0 _−_ _h_ [ˆ] _θ_ ( _zt, t_ )��� (61)


_βt_

2


_√_
(
���


_√_
1 _−_ _t_ + _t_


_√_
Observe that, with _σ_ = 1, the term ( _[√]_ 1 _−_ _t_ + _t_ _t_


Observe that, with _σ_ = 1, the term ( 1 _−_ _t_ + _t_ _t_ ) _z_ 1 + (� _t_ (1 _−_ _t_ ) _−_ 1) _z_ 0 reduces to _z_ 1 _−_ _z_ 0 both

at _t_ = 0 and _t_ = 1.


C.3 Denoising


This parameterization is applicable only when _z_ 0 is Gaussian. Starting with the loss term with _u_ ( _zt, t_ )
and using the fact that _zt_ = _tz_ 1 + ~~�~~ (1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _z_ 0, we can manipulate the objective as
following


�����


2


1

2


_z_ 1 _−_
�����


~~�~~ _σ_ [2] _t_ + 1 _−_ _t_


_z_ 0 _−_ _hθ_ ( _zt, t_ )
1 _−_ _t_


(62)


(63)


�����


2


= [1]

2


_z_ 1 _−_
�����


- _σ_ [2] _t_ + 1 _−_ _t_


+ 1 _−_ _t_ _zt −_ _tz_ 1

1 _−_ _t_ �(1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _−_ _hθ_ ( _zt, t_ )


= [1]

2


2

_zt −_ _tz_ 1
_z_ 1 _−_ _−_ _hθ_ ( _zt, t_ ) (64)
���� 1 _−_ _t_ ����


1

= [1] 2 (1 _−_ _t_ ) [2] _[∥][z]_ [1] _[ −]_ _[z][t][ −]_ [(1] _[ −]_ _[t]_ [)] _[h][θ]_ [(] _[z][t][, t]_ [)] _[∥]_ [2] (65)


1

= [1]

2 (1 _−_ _t_ ) [2]


2
_z_ 1 _−_ _h_ ˆ _θ_ ( _zt, t_ ) (66)
��� ���


= _[β][t]_

2


2
_z_ 1 _−_ _h_ ˆ _θ_ ( _zt, t_ ) (67)
��� ���


where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ _zt_ + (1 _−_ _t_ ) _hθ_ ( _zt, t_ ) and _βt_ _≡_ 1 _/_ (1 _−_ _t_ ) [2] . In this form, _h_ [ˆ] can be viewed as a
denoiser.


18


C.4 NoisePred


This parameterization is applicable only when _z_ 0 is Gaussian. Similar to the previous section, we can construct the noise prediction parameterization by substituting _z_ 1 using _zt_ =
_tz_ 1 + ~~�~~ (1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _z_ 0.


�����


2


1

2


_z_ 1 _−_
�����


~~�~~ _σ_ [2] _t_ + 1 _−_ _t_


_z_ 0 _−_ _hθ_ ( _zt, t_ )
1 _−_ _t_


�����


2


= [1]

2


_zt_ _−_ ~~�~~ (1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ ) _z_ 0 _−_

_t_

�����


- _σ_ [2] _t_ + 1 _−_ _t_


_z_ 0 _−_ _hθ_ ( _zt, t_ )
1 _−_ _t_


_√_
_σ_ [2] _t_ + 1 _−_ _tz_ 0 _−_ _t_


_t_ + 1 _−_ _tz_ 0 _−_ _t_ _σ_ [2] _t_ + 1 _−_ _tz_ 0

_−_ _hθ_ ( _zt, t_ )
_t_ ~~_[√]_~~ 1 _−_ _t_


�����


(68)


(69)


(70)


(71)


2


= [1]

2


_√_ _√_
1 _−_ _tzt −_ (1 _−_ _t_ )


�����


2


= [1]

2


�����

�����


_√_ _√_
1 _−_ _tzt −_


_σ_ [2] _t_ + 1 _−_ _tz_ 0 _−_ _hθ_ ( _zt, t_ )
_t_ ~~_[√]_~~ 1 _−_ _t_


1

= [1]

2 _t_ [2] (1 _−_ _t_ )


_√_
���


  - _√_
1 _−_ _tzt −_ _σ_ [2] _t_ + 1 _−_ _tz_ 0 _−_ _t_


2
1 _−_ _thθ_ ( _zt, t_ ) (72)
���


[1] _σ_ [2] _t_ + 1 _−_ _t_

2 _t_ [2] (1 _−_ _t_ )


_√_ _√_
1 _−_ _tzt −_ _t_ 1 _−_ _thθ_ ( _zt, t_ )
_z_ 0 _−_ ~~_√_~~
���� _σ_ [2] _t_ + 1 _−_ _t_


2

(73)

����


= [1]


_t_ [2] (1 _−_ _t_ )


= _[β][t]_


2


2
_z_ 0 _−_ _h_ ˆ _θ_ ( _zt, t_ ) (74)
��� ���


_√_
where _h_ [ˆ] _θ_ ( _zt, t_ ) _≡_ ( _[√]_ 1 _−_ _tzt −_ _t_ _[√]_ 1 _−_ _thθ_ ( _zt, t_ )) _/_


_σ_ [2] _t_ + 1 _−_ _t_ and _βt_ _≡_ 1 _/_ ( _t_ [2] (1 _−_ _t_ )).


D LATENT SCORE FUNCTION WITH GAUSSIAN _p_ 0


When _p_ 0( _z_ 0) is gaussian, _z_ 0 _∼_ _N_ (0 _, I_ ), we can compute the score function estimate _∇zt_ ln _pt_ ( _zt_ )
from the learned drift _hθ_ (Singh & Fischer, 2024). When _z_ 0 is gaussian, the transition density _p_ ( _zt|z_ 1)
is Gaussian. With _zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0, we can reparameterize as _zt_ = _κtz_ 1 + ~~�~~ _νt_ [2] [+] _[ η]_ _t_ [2] _[z]_ [0] _[, z]_ [0] _[∼]_
_N_ (0 _, I_ ).

_p_ ( _zt|z_ 1) = _N_ ( _zt_ ; _κtz_ 1 _,_ ( _νt_ [2] [+] _[ η]_ _t_ [2][)] _[I]_ [)] (75)


From Singh & Fischer (2024)(eq. 41, Appendix B) we have


(76)


_∇zt_ ln _pt_ ( _zt_ ) = E _pt_ ( _z_ 1 _|zt_ )


- _−zt_ + _µ_ ( _z_ 1 _, t_ )
_σ_ ( _z_ 1 _, t_ ) [2]


Substituting


(77)


_∇zt_ ln _pt_ ( _zt_ ) = E _pt_ ( _z_ 1 _|zt_ )


- _−zt_ + _κtz_ 1
_νt_ [2] [+] _[ η]_ _t_ [2]


= _[−][z][t]_ [ +] _[ κ][t]_ [E][[] _[z]_ [1] _[|][z][t]_ []] (78)

_νt_ [2] [+] _[ η]_ _t_ [2]


Since the interpolation relates _z_ 0 _, z_ 1 _, zt_ as _zt_ = _κtz_ 1 + ~~�~~ _νt_ [2] [+] _[ η]_ _t_ [2] _[z]_ [0][,] [we] [can] [rewrite] [the] [above]
expression in terms of _z_ 0 as following

_∇zt_ ln _pt_ ( _zt_ ) = _−_               - [E] _ν_ [[] _[z]_ _t_ [2][0][+] _[|][z][ η][t]_ []] _t_ [2] (79)


E LATENT SCORE FUNCTION WITH GENERAL _p_ 0


For a general distribution _p_ 0( _z_ 0), it may not be possible to estimate the score function _∇zt_ ln _pt_ ( _zt_ )
from the learned drift _hθ_ ( _zt, t_ ) alone. Here we derive the expression for estimating the score function


19


for a general distribution _p_ 0( _z_ 0). Recall from eq. (10) that _p_ ( _zt|z_ 0 _, z_ 1) is Gaussian. From Denoising
Score Matching (Vincent, 2011), we can write


_∂_ ln _pt_ ( _zt|z_ 0 _, z_ 1)
_∇zt_ ln _pt_ ( _zt_ ) = E _pt_ ( _z_ 0 _,z_ 1 _|zt_ ) (80)

_∂zt_


where we have conditioned on both variables _x_ 0 _, x_ 1. Since _p_ ( _zt|z_ 0 _, z_ 1) is Gaussian, as in the previous
section, we can write


- _−zt_ + _µ_ ( _z_ 0 _, z_ 1 _, t_ )
_σ_ ( _z_ 0 _, z_ 1 _, t_ ) [2]


_∇zt_ ln _pt_ ( _zt_ ) = E _pt_ ( _z_ 0 _,z_ 1 _|zt_ )


(81)


Now, for _zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0, we have _p_ ( _zt|z_ 0 _, z_ 1) = _N_ ( _zt_ ; _κtz_ 1 + _νtz_ 0 _, ηt_ [2] _[I]_ [)][.] [Substituting]


- _−zt_ + _κtz_ 1 + _νtz_ 0
_ηt_ [2]


_∇zt_ ln _pt_ ( _zt_ ) = E _pt_ ( _z_ 0 _,z_ 1 _|zt_ )


(82)


(83)


= E _pt_ ( _ϵ|zt_ )


- _−ηtϵ_
_ηt_ [2]


= _−_ [E] _[p][t]_ [(] _[ϵ][|][z][t]_ [)][[] _[ϵ]_ []]


(84)
_ηt_


_[ϵ][|][z][t]_ [)][[] _[ϵ]_ []]

_≡−_ [E][[] _[ϵ][|][z][t]_ []]
_ηt_ _ηt_


Note that this result mirrors the one for SI(Theorem 2.8, (Albergo et al., 2023)), though our derivation
is straightforward and follows directly from Denoising Score Matching (Vincent, 2011).


F DETAILED DERIVATION OF SAMPLING


For an SDE of the form


_dzt_ = _hθ_ ( _zt, t_ ) _dt_ + _σtdwt_ (85)


Singh & Fischer (2024) (Corollary 1) derives a flexible family of samplers as following


          -           _t_ [)] _[σ]_ _t_ [2]
_dzt_ = _hθ_ ( _zt, t_ ) _−_ [(1] _[ −]_ _[γ]_ [2] _∇zt_ ln _pt_ ( _zt_ ) _dt_ + _γtσtdwt_ (86)
2


where _γt_ is a time dependent weighting that can be chosen to control the amount of stochasticity
injected into the sampling. Note that choosing _γt_ = 0 yields the probability flow ODE (Song et al.,
2020b) and results in a deterministic sampler. This general form of sampler requires both the drift
_hθ_ ( _zt, t_ ) and the score function _∇zt_ ln _pt_ ( _zt_ ). In general, the score function needs to be separately
estimated. See section E for an estimator. We can also set _γt_ = 1, leading to direct discretization of
the original SDE in eq. (85). However, for the special case of Gaussian _z_ 0, we can infer the score
function from the learned drift _hθ_ (section D). For this special case, we use the general form above to
derive a family of samplers for various parameterizations discussed in section C. Recall that for the
choice of _κt_ = _t, νt_ = 1 _−_ _t_ used in this paper, the loss term is specified by eq. (50). Without any
reparameterization, we have

_hθ_ ( _zt, t_ ) = [E][[] _[z]_ [1] _[|][z][t]_ []] _[ −]_ _[z][t]_ (87)

1 _−_ _t_

E[ _z_ 1 _|zt_ ] = _zt_ + (1 _−_ _t_ ) _hθ_ ( _zt, t_ ) (88)


We can use the above to determine the expression for the score function

_∇x_ ln _pt_ ( _zt_ ) = _[−][z][t]_ [ +] _[ th][θ]_ [(] _[z][t][, t]_ [)] (89)

_σ_ [2] _t_ + 1 _−_ _t_


Above expressions for the score _∇x_ ln _pt_ ( _zt_ ) can then be plugged into eq. (86) to derive a sampler
for the original formulation


   _t_ [)] _[σ]_ [2]
_dzt_ = _hθ_ ( _zt, t_ ) _−_ [(1] _[ −]_ _[γ]_ [2]


_σ_ [2] _t_ + 1 _−_ _t_


_[γ]_ _t_ [2][)] _[σ]_ [2] _−zt_ + _thθ_ ( _zt, t_ )

2 _σ_ [2] _t_ + 1 _−_ _t_


_dt_ + _γtσdwt_ (90)


For each of the following parameterizations, we calculate the expression for the drift _hθ_ and the score
function _∇x_ ln _pt_ ( _zt_ ). These expressions can then be plugged into eq. (86) to derive the sampler.


20


F.1 SAMPLER FOR OrigFlow


For the OrigFlow parameterization, we have

_hθ_ ( _zt, t_ ) = _h_ ˆ ~~_√_~~ _θ_ ( _zt, t_ ) (91)
1 _−_ _t_


For Gaussian _z_ 0, we can now substitute into the expression for the score function


_∇x_ ln _pt_ ( _zt_ ) = _[−][z][t]_ [ +] _[ th][θ]_ [(] _[z][t][, t]_ [)]


(92)
_σ_ [2] _t_ + 1 _−_ _t_


= _[−]_ ~~_√_~~ _[√]_ [1] _[ −]_ _[tz][t]_ [ +] _[ t][h]_ [ˆ] _[θ]_ [(] _[z][t][, t]_ [)] (93)
1 _−_ _t_ ( _σ_ 2 _t_ + 1 _−_ _t_ )


The drift _hθ_ and the score function _∇x_ ln _pt_ ( _zt_ ) can now be plugged into eq. (86) to derive the
sampler.


F.2 SAMPLER FOR InterpFlow


For the InterpFlow parameterization, we have


_√_
_hθ_ ( _zt, t_ ) = _h_ ˆ( _zt_ ~~_√_~~ _, t_ ) _−_


~~_√_~~ _, t_ ) _−_ _tzt_ (94)
1 _−_ _t_


For Gaussian _z_ 0, we can now substitute into the expression for the score function


_∇x_ ln _pt_ ( _zt_ ) = _[−][z][t]_ [ +] _[ th][θ]_ [(] _[z][t][, t]_ [)]


(95)
_σ_ [2] _t_ + 1 _−_ _t_


_√_
= _[−][√]_ [1] _[ −]_ ~~_√_~~ _[tz][t]_ [ +] _[ t][h]_ [ˆ] _[θ]_ [(] _[z][t][, t]_ [)] _[ −]_ _[t]_


~~_√_~~ _[tz][t]_ [ +] _[ t][h][θ]_ [(] _[z][t][, t]_ [)] _[ −]_ _[t]_ _tzt_ (96)
1 _−_ _t_ ( _σ_ 2 _t_ + 1 _−_ _t_ )

_√_

_[ −]_ _[t]_ [ +] _[ t]_ _t_ ) _zt_ + _th_ [ˆ] _θ_ ( _zt, t_ )

~~_√_~~ (97)
1 _−_ _t_ ( _σ_ 2 _t_ + 1 _−_ _t_ )


_√_
= _[−]_ [(] _[√]_ [1] ~~_√_~~ _[ −]_ _[t]_ [ +] _[ t]_


The drift _hθ_ and the score function _∇x_ ln _pt_ ( _zt_ ) can now be plugged into eq. (86) to derive the
sampler.


F.3 SAMPLER FOR Denoising


For the Denoising parameterization, we have

_h_ ˆ _θ_ ( _zt, t_ ) _−_ _zt_
_hθ_ ( _zt, t_ ) = (98)

1 _−_ _t_

For Gaussian _z_ 0, substituting into the expression for the score function

_∇x_ ln _pt_ ( _zt_ ) = _[−][z][t]_ [ +] _[ th][θ]_ [(] _[z][t][, t]_ [)] (99)

_σ_ [2] _t_ + 1 _−_ _t_

= _[−]_ [(1] _[ −]_ _[t]_ [)] _[z][t]_ [ +] _[ t][h]_ [ˆ] _[θ]_ [(] _[z][t][, t]_ [)] _[ −]_ _[tz][t]_ (100)

(1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ )

_−zt_ + _th_ [ˆ] _θ_ ( _zt, t_ )
= (101)
(1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ )


The drift _hθ_ and the score function _∇x_ ln _pt_ ( _zt_ ) can now be plugged into eq. (86) to derive the
sampler.


F.4 SAMPLER FOR NoisePred


Again, we have


_√_
_hθ_ ( _zt, t_ ) = _[−]_


_σ_ [2] _t_ + 1 _−_ _th_ [ˆ] _θ_ ( _zt, t_ ) + _[√]_ 1 _−_ _tzt_

(102)
_t_ ~~_[√]_~~ 1 _−_ _t_


21


For Gaussian _z_ 0, substituting into the expression for the score function


_∇x_ ln _pt_ ( _zt_ ) = _[−][z][t]_ [ +] _[ th][θ]_ [(] _[z][t][, t]_ [)]


(103)
_σ_ [2] _t_ + 1 _−_ _t_


_√_
= _[−][√]_ [1] _[ −]_ _[tz][t][ −]_ ~~_√_~~


_√_

~~_√_~~ _σ_ [2] _t_ + 1 _−_ _th_ [ˆ] _θ_ ( _zt, t_ ) + _[√]_ 1 _−_ _tzt_ (104)
1 _−_ _t_ ( _σ_ 2 _t_ + 1 _−_ _t_ )


_−h_ [ˆ] _θ_ ( _zt, t_ )
= (105)
�(1 _−_ _t_ )( _σ_ [2] _t_ + 1 _−_ _t_ )


The drift _hθ_ and the score function _∇x_ ln _pt_ ( _zt_ ) can now be plugged into eq. (86) to derive the
sampler.


G GAUSSIANITY OF CONDITIONAL DENSITY


We have

_p_ ( _zt|z_ 1 _, z_ 0) = _[p]_ [(] _[z]_ [1] _[|][z][t]_ [)] _[p]_ [(] _[z][t][|][z]_ [0][)] (106)

_p_ ( _z_ 1 _|z_ 0)


Further, for the SDE in eq. (7), using results from section L, we have that the transition density
_p_ ( _xt|xs_ ) is normal with


_p_ ( _xt|xs_ ) = _N_ ( _xt_ ; _µst,_ Σ _st_ ) (107)

�� _t_             _µst_ = _µs_ exp _h_ ( _τ_ ) _dτ_ _≡_ _µsast_ (108)

_s_


   - _t_
Σ _st_ = _I_


_t_ - - _t_

_σ_ ( _τ_ ) [2] exp 2
_s_ _τ_


   _h_ ( _u_ ) _du_ _dτ_ ) _≡_ _Ibst_ (109)
_τ_


Then, the conditional density _p_ ( _zt|z_ 1 _, z_ 0) is also normal _N_ ( _zt_ ; _µ_ ( _z_ 0 _, z_ 1 _, t_ ) _,_ Σ( _z_ 0 _, z_ 1 _, t_ )) with


**Proof:** First note that


Next


_µ_ ( _z_ 0 _, z_ 1 _, t_ ) = _[b]_ [0] _[t][a][t]_ [1] _[z]_ [1][ +] _[ b][t]_ [1] _[a]_ [0] _[t][z]_ [0] (110)

_b_ 01

Σ( _z_ 0 _, z_ 1 _, t_ ) = _[b]_ [0] _[t][b][t]_ [1] _I_ (111)

_b_ 01


_a_ 01 = _a_ 0 _tat_ 1 (112)


_ast_ = _[a]_ [0] _[t]_


_[a]_ [0] _[t]_ = _[a][s]_ [1]

_a_ 0 _s_ _at_ 1


(113)
_at_ 1


   - _t_
_bst_ = _σ_ ( _v_ ) [2] _a_ [2] _vt_ _[dv]_ (114)

_s_


   - 1
_b_ 01 = _σ_ ( _v_ ) [2] _a_ [2] _v_ 1 _[dv]_ (115)

0


 - _t_
=


_t_  - 1

_σ_ ( _v_ ) [2] _a_ [2] _v_ 1 _[dv]_ [ +]
0 _t_


_σ_ ( _v_ ) [2] _a_ [2] _v_ 1 _[dv]_ (116)
_t_


 - _t_
= _σ_ ( _v_ ) [2] _a_ [2] _vt_ _[a]_ [2] _t_ 1 _[dv]_ [ +] _[ b][t]_ [1] (117)

0


- _t_

_σ_ ( _v_ ) [2] _a_ [2] _vt_ _[dv]_ [ +] _[ b][t]_ [1] (118)
0


- _t_


= _a_ [2] _t_ 1


= _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [+] _[ b][t]_ [1] (119)


22


Now


_at_ 1 _zt|_ 2

+ _[|][z][t][ −]_ _[a]_ [0] _[t][z]_ [0] _[|]_ [2]
_bt_ 1 _b_ 0 _t_


_[a]_ [0] _[t][z]_ [0] _[|]_ [2]

_−_ _[|][z]_ [1] _[ −]_ _[a]_ [01] _[z]_ [0] _[|]_ [2]
_b_ 0 _t_ _b_ 01


_b_ 01


��


(120)


(121)


    - 1 _b_ 01
_p_ ( _zt|z_ 1 _, z_ 0) =
2 _π_ _bt_ 1 _b_ 0 _t_


- _[n]_ 2 exp _−_ [1]

2


- _|z_ 1 _−_ _at_ 1 _zt|_ 2


Using the identities _a_ 01 = _a_ 0 _tat_ 1 _, b_ 01 = _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [+] _[ b][t]_ [1] [and completing the squares we get]


_−_ [1]


_b_ 01

[1]

2 _b_ 0 _tbt_ 1


����


2 [�]


    - 1 _b_ 01
_p_ ( _zt|z_ 1 _, z_ 0) =
2 _π_ _b_ 0 _tbt_ 1


- _[n]_ 2
exp


_b_ 0 _tat_ 1 _z_ 1 + _bt_ 1 _a_ 0 _tz_ 0
_zt −_
���� _b_ 01


We can therefore parameterize _zt_ as following using the reparameterization trick.


_z_ 1 + _[b][t]_ [1] _[a]_ [0] _[t]_

_b_ 01
���   _νt_


_z_ 1 + _[b][t]_ [1] _[a]_ [0] _[t]_


_z_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) (122)


_zt_ =


~~�~~ _b_ 0 _tbt_ 1

_b_ 01

- �� _ηt_


_ϵ_ + _[b]_ [0] _[t][a][t]_ [1]

_b_ 01
���  _κt_


_ϵ_ + _[b]_ [0] _[t][a][t]_ [1]


we can succinctly rewrite the above as


_zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) (123)


Where _ηt, κt, νt_ are appropriate scalar functions of time _t_ .


H GENERAL TRAINING OBJECTIVE


Here we derive the form of the general training objective. The first term in the objective is the
reconstruction term and remains as is. The second term of the training objective uses _u_ ( _zt, t_ ), let’s
recall it’s expression

_u_ ( _zt, t_ ) = _σt_ _[−]_ [1] [ _htzt_ + _σt_ [2] _[∇][z]_ _t_ [ln] _[ p]_ [(] _[z]_ [1] _[|][z][t]_ [)] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]] (124)


The first two terms in the above serve as the target for _hθ_ . Next, we rewrite them in terms of existing
variables. Let _ξ_ ( _t_ ) denote these two terms and substitute eq. (8) as following


_ξ_ ( _t_ ) = _htzt_ + _σt_ [2] _[∇][z]_ _t_ [ln] _[ p]_ [(] _[z]_ [1] _[|][z][t]_ [)] (125)

_t_ _[a][t]_ [1][(] _[z]_ [1] _[−]_ _[a][t]_ [1] _[z][t]_ [)]
= _htzt_ + _[σ]_ [2] (126)

_bt_ 1


 _t_ _[a]_ _t_ [2] 1
= _ht −_ _[σ]_ [2]
_bt_ 1


_t_ _[a][t]_ [1] _[z]_ [1]
_zt_ + _[σ]_ [2] (127)

_bt_ 1


_t_ _[a][t]_ [1] _[z]_ [1]
_zt_ + _[σ]_ [2]


Next, recall the stochastic interpolant and the expressions for _ast_ and _bst_ from section G


_zt_ = _ηtϵ_ + _κtz_ 1 + _νtz_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) (128)


[0] _[t][a][t]_ [1]

_,_ _νt_ = _[b][t]_ [1] _[a]_ [0] _[t]_
_b_ 01 _b_ 01


_,_ (129)
_b_ 01


_ηt_ =


~~�~~ _b_ 0 _tbt_ 1


_tbt_ 1

_,_ _κt_ = _[b]_ [0] _[t][a][t]_ [1]
_b_ 01 _b_ 01


�� _t_
_ast_ = exp


_t_  -  - _t_

_h_ ( _τ_ ) _dτ_ _,_ _bst_ =
_s_ _s_


_σ_ ( _v_ ) [2] _a_ [2] _vt_ _[dv]_ (130)
_s_


(131)


Intuitively, we expect the drift _hθ_ to be related to the velocity field. Therefore, we compute the time
derivatives of _κt, νt_ and _ηt_ next


_dt_ _[a][t]_ [1]


_dκt_


_dκt_

[1]
_dt_ [=] _b_


_b_ 01


- _dat_ 1
_b_ 0 _t_


_dt_ _[a]_ [0] _[t]_


_dνt_


_dνt_

[1]
_dt_ [=] _b_


(132)


(133)


_b_ 01


- _da_ 0 _t_
_bt_ 1


_t_ 1

+ _[db]_ [0] _[t]_
_dt_ _dt_


0 _t_

+ _[db][t]_ [1]
_dt_ _dt_


_dt_ _[b][t]_ [1]


_dηt_ 1

_dt_ [=] 2 _ηtb_ 01


- _dbt_ 1
_b_ 0 _t_


_t_ 1

_[db]_ [0] _[t]_
_dt_ [+] _dt_


(134)


23


From the expression for _ast_, using differentiation under the integral sign, we have
_da_ 0 _t_ _dat_ 1


0 _t_ _dat_ 1

= _a_ 0 _tht,_
_dt_ _dt_


= _−at_ 1 _ht_ (135)
_dt_


Similarly, from the expression for _bst_
_db_ 0 _t_ [2] [2]


_σ_ ( _v_ ) [2] _a_ [2] _vt_ _[h][t][dv]_ [=] _[ σ]_ _t_ [2] [+ 2] _[b]_ [0] _[t][h][t]_ (136)
0


      - _t_

0 _t_

= _σt_ [2] _[a]_ _tt_ [2] [+ 2]
_dt_


_dbt_ 1

= _−σt_ [2] _[a]_ _t_ [2] 1 (137)
_dt_


Since _att_ = 1. Substituting back into the equations for the derivatives of _κt_ and _νt_
_dκt_

[1] [2] [1] [2]


_dκt_

[1]
_dt_ [=] _b_


_b_ 01


- _−b_ 0 _tat_ 1 _ht_ + ( _σt_ [2] [+ 2] _[b]_ [0] _[t][h][t]_ [)] _[a][t]_ [1] - = [1]


_b_ 01


- _σt_ [2] _[a][t]_ [1] [+] _[ b]_ [0] _[t][a][t]_ [1] _[h][t]_ - (138)


_t_ _[a][t]_ [1]
= _[σ]_ [2] + _κtht_ (139)
_b_ 01


_dνt_


_dνt_

[1]
_dt_ [=] _b_


_b_ 01


- _bt_ 1 _a_ 0 _tht −_ _σt_ [2] _[a]_ _t_ [2] 1 _[a]_ [0] _[t]_ - = _νtht −_ _[σ]_ _t_ [2] _[a]_ _t_ [2] 1 _[a]_ [0] _[t]_ (140)
_b_ 01


  _t_ _[a]_ _t_ [2] 1

= _νt_ _ht −_ _[σ]_ [2]

_bt_ 1


(141)


_dηt_ 1

_dt_ [=] 2 _ηtb_ 01

1
=
2 _ηtb_ 01

1
=
2 _ηtb_ 01


1
=
2 _ηtb_ 01


1
=
2 _ηtb_ 01


_t_ _[b]_ [01]
_b_ 01 _σt_ [2] [+] [2] _[η]_ [2]
_νt_


- _−σt_ [2] _[a]_ _t_ [2] 1 _[b]_ [0] _[t]_ [+ (] _[σ]_ _t_ [2] [+ 2] _[b]_ [0] _[t][h][t]_ [)] _[b][t]_ [1] - (142)


�( _bt_ 1 _−_ _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [)] _[σ]_ _t_ [2] [+ 2] _[b]_ [0] _[t][b][t]_ [1] _[h][t]_ - (143)


- _bt_ 1
_νt_


( _bt_ 1 _−_ _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [)] _[σ]_ _t_ [2] [+ 2] _[b]_ [0] _[t]_


_dνt_

_t_ _[a]_ _t_ [2] 1
_dt_ [+] _[ σ]_ [2]


_dνt_


��
(144)


�( _bt_ 1 + _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [)] _[σ]_ _t_ [2] [+] [2] _[b]_ [0] _[t][b][t]_ [1]

_νt_


�( _bt_ 1 + _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [)] _[σ]_ _t_ [2] [+] [2] _[b]_ [0] _[t][b][t]_ [1]


(145)


_dνt_

_dt_


_dνt_ 
(146)

_dt_


= _[σ]_ _t_ [2] + _[η][t]_
2 _ηt_ _νt_


_dνt_

(147)
_dt_


Where we have used the identity _b_ 01 = _bt_ 1 + _a_ [2] _t_ 1 _[b]_ [0] _[t]_ [from eq. (119).] [Further, we can relate] _[dκ]_ _dt_ _[t]_


Where we have used the identity _b_ 01 = _bt_ 1 + _at_ 1 _[b]_ [0] _[t]_ [from eq. (119).] [Further, we can relate] _dt_ _[t]_ [and]

_dνt_


_t_

_dt_ [by eliminating] _[ h][t]_ [ as following]


_dκt_ _t_ _[a][t]_ [1] - 1

_[σ]_ [2] + _κt_
_dt_ [=] _b_ 01 _νt_


_dνt_ _[σ]_ _t_ [2] _[a]_ _t_ [2] 1

_dt_ [+] _bt_ 1


= _[κ][t]_

_νt_


_dνt_ _[σ]_ _t_ [2] _[a][t]_ [1] + _κt_ _σt_ [2] _[a]_ _t_ [2] 1 (148)

_dt_ [+] _b_ 01 _bt_ 1


_dνt_ _[σ]_ _t_ [2] _[a][t]_ [1] + _κt_ _σt_ [2] _[a]_ _t_ [2] 1 (148)

_dt_ [+] _b_ 01 _bt_ 1

_dνt_ _t_ _[a][t]_ [1][(] _[b][t]_ [1] [+] _[ b]_ [0] _[t][a]_ [2] _t_ 1 [)]

_[σ]_ [2] (149)
_dt_ [+] _b_ 01 _bt_ 1


_σt_ [2] _[a]_ _t_ [2] 1
= _[κ][t]_
_bt_ 1 _νt_


_t_ _t_ _[a][t]_ [1][(] _[b][t]_ [1] [+] _[ b]_ [0] _[t][a]_ [2] _t_ 1 [)]

_[σ]_ [2] (149)
_dt_ [+] _b_ 01 _bt_ 1


= _[κ][t]_

_νt_


_dνt_ _t_ _[a][t]_ [1]

_[σ]_ [2] + _[b]_ [0] _[t][a][t]_ [1]
_dt_ [+] _b_ 01 _b_ 01


_b_ 01


_dνt_


= _[κ][t]_

_νt_


_dνt_ _t_ _[a][t]_ [1] _[b]_ [01]

_[σ]_ [2] (150)
_dt_ [+] _b_ 01 _bt_ 1


= _[κ][t]_

_νt_


_dνt_ _t_ _[a][t]_ [1] _[b]_ [01]

_[σ]_ [2] (150)
_dt_ [+] _b_ 01 _bt_ 1

_dνt_ _t_ _[a][t]_ [1]

_[σ]_ [2] (151)
_dt_ [+] _bt_ 1


_dνt_ _t_ _[a][t]_ [1]

_[σ]_ [2] (151)
_dt_ [+] _bt_ 1


We can now substitute into the expression for _ξ_ ( _t_ ) in eq. (127)


_ξ_ ( _t_ ) = [1]

_νt_


_dνt_ _t_ _[a][t]_ [1] _[z]_ [1]

_[σ]_ [2]
_dt_ _[z][t]_ [ +] _bt_ 1


_dνt_


(152)
_bt_ 1


= [1]

_νt_


_dνt_ - _dκt_

_dt_ [(] _[η][t][ϵ]_ [ +] _[ κ][t][z]_ [1][ +] _[ ν][t][z]_ [0][) +] _dt_ _[−]_ _[κ]_ _νt_ _[t]_


_dνt_


_νt_


_z_ 1 (153)


_dνt_

_dt_


= _[η][t]_

_νt_


_dνt_


_dνt_

_[dκ][t]_
_dt_ _[ϵ]_ [ +] _dt_


_[t]_

_[dν][t]_
_dt_ _[z]_ [1][ +] _dt_


(154)
_dt_ _[z]_ [0]


= - _dηt_ _t_
_dt_ _[−]_ 2 _[σ]_ _η_ [2] _t_


_ϵ_ + _[dκ][t]_


(155)
_dt_ _[z]_ [0]


_[t]_

_[dν][t]_
_dt_ _[z]_ [1][ +] _dt_


24


Substituting back into the expression for _u_ ( _zt, t_ ) we can write the general form as following


       
_[t]_

(156)
_dt_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


_u_ ( _zt, t_ ) = _σt_ _[−]_ [1]


�� _dηt_ _t_
_dt_ _[−]_ 2 _[σ]_ _η_ [2] _t_


_ϵ_ + _[dκ][t]_


_[t]_

_[dν][t]_
_dt_ _[z]_ [1][ +] _dt_


With the _u_ ( _zt, t_ ) above, the ELBO can be written using eq. (3).


I DRIFT _ht_, DISPERSION _σt_ AND STOCHASTICITY _ηt_ FROM _κt, νt_


Often, specifying the interpolant coefficients _κt, νt_ is intuitively easier than specifying _ht, σt_ directly.
Here we derive expressions for _ht_ and _σt_ given _κt_ and _νt_ . We have

_dκt_ _t_ _[a][t]_ [1]

_[σ]_ [2] (157)
_dt_ [=] _[ κ][t][h][t]_ [ +] _b_ 01

_dνt_ _t_ _[a]_ _t_ [2] 1

_νt_ (158)
_dt_ [=] _[ h][t][ν][t][ −]_ _[σ]_ _b_ [2] _t_ 1

Multiplying first equation by _νt_ and second by _κt_ and then subtracting the second from the first


_t_ _dνt_

_dt_ _[−]_ _[κ][t]_ _dt_


_dκt_
_νt_


_dνt_ _σt_ [2] _[a][t]_ [1] _σt_ [2] _[a]_ _t_ [2] 1

+ _κt_ _νt_ (159)
_dt_ [=] _[ ν][t]_ _b_ 01 _bt_ 1


             - _σt_ [2] _[a][t]_ [1] _σt_ [2] _[a]_ _t_ [2] 1
= _νt_ + _κt_ _νt_
_b_ 01 _bt_ 1

Substituting in the definitions of _κt_ and _νt_ in RHS and simplifying


(160)


_dκt_
_νt_


_dκt_ _dνt_

_dt_ _[−]_ _[κ][t]_ _dt_


_dνt_ - _bt_ 1 _a_ 01 _σt_ 2 _t_ _[a]_ _t_ [2] 1 _[a]_ [01]

+ _[b]_ [0] _[t][σ]_ [2]
_dt_ [=] _b_ [2] 01 _b_ [2] 01


(161)


= _[a]_ [01] _[σ]_ _t_ [2]
_b_ [2] 01


- _bt_ 1 + _b_ 0 _ta_ [2] _t_ 1� = _[a]_ [01] _b_ [2] 01 _[σ]_ _t_ [2] _b_ 01 (162)


= _[a]_ [01] _[σ]_ _t_ [2] (163)
_b_ 01

where we have used _a_ 01 = _a_ 0 _tat_ 1 and _b_ 01 = _bt_ 1 + _b_ 0 _ta_ [2] _t_ 1 [.] [Therefore]


_dt_


(164)


_σt_ [2] [=] _[b]_ [01]

_a_ 01


- _dκt_
_νt_


_t_ _dνt_

_dt_ _[−]_ _[κ][t]_ _dt_


Where _b_ 01 _>_ 0 _, a_ 01 _>_ 0 are time _t_ independent constants that can’t be determined by _κt, νt_ alone. In
this paper, we assume _a_ 01 = 2 and _b_ 01 = _a_ 01 _σ_ [2], where _σ_ is a hyper-parameter. Next, to derive the
expression for _ht_, we eliminate _σt_ [2] [from eqs. (157) and (158).]


- _dκt_
_dt_ _[−]_ _[κ][t][h][t]_


_−_ [1]

_νt_


_−_ [1]


_dνt_

_dt_ [+] _[ h][t]_


(165)


_b_ 01


_ht_


= _[b][t]_ [1]

_at_ 1


= _[b][t]_ [1]


- _dκt_
= _b_ 01


_dνt_


_b_ 01 _κt_ + _[b][t]_ [1]

_at_ 1


_t_

_[b][t]_ [1]
_dt_ [+] _at_ 1


_t_

_[b][t]_ [1]
_dt_ [+] _at_ 1


_at_ 1


1
_νt_


_dνt_

(166)
_dt_


_at_ 1


_b_ 01
_bt_ 1 _a_ 0 _t_


_dνt_

(167)
_dt_


_ht_


- _at_ 1 _b_ 01 _κt_ + _bt_ 1
_at_ 1


- _dκt_
= _b_ 01


_t_

_[dν][t]_
_dt_ [+] _dt_


(168)
_dt_


_ht_


_a_ 0 _tat_ 1 _κt_ + _[a]_ [0] _[t][b][t]_ [1]

_b_ 01


_a_ 0 _tat_ 1 _κt_ + _[a]_ [0] _[t][b][t]_ [1]


- _dκt_
= _a_ 0 _tat_ 1


_dκt_

_[dν][t]_
_dt_ [+] _dt_


_dκt_
_ht_ ( _a_ 01 _κt_ + _νt_ ) = _a_ 01


(169)
_dt_


_ht_ = _[a]_ [01] _[dκ]_ _dt_ _[t]_


_[dκ]_ _dt_ _[t]_ [+] _[dν]_ _dt_ _[t]_


_dt_ _dt_ (170)

_a_ 01 _κt_ + _νt_


As before, _a_ 01 _>_ 0 is a time independent constant that can’t be determined from the choice of _κt, νt_
alone. Finally, to express _ηt_ in terms of given _κt, νt_, note that


_b_ 01

_ηt_ [2] [=] _[b]_ [0] _[t][b][t]_ [1] =

_b_ 01 _a_ 0 _tat_ 1


_b_ 0 _tat_ 1

_b_ 01


25


_bt_ 1 _a_ 0 _t_


1 _a_ 0 _t_

= _[b]_ [01]
_b_ 01 _a_ 01


_κtνt_ (171)
_a_ 01


where we have used the identity _a_ 01 = _a_ 0 _tat_ 1. In the following, we derive the formulation for
the linear _κt, νt_ schedule used in experiments in this paper. This schedule also corresponds to the
choice used in Stochastic Interpolants(Albergo et al., 2023). Note that similar choice is made by the
Rectified Flow (Liu et al., 2022), however the missing _η_ term implies that they do not have a bound
on the likelihood, as also observed by Albergo et al. (2023). We also provide the derivation for the
variance preserving schedule as it is quite commonly used for diffusion models. However, we do not
empirically explore it.


J FORMULATION FOR LINEAR _κt, νt_


For linear choice _κt_ = _t, νt_ = 1 _−_ _t_ . Further, we assume _a_ 01 = 2 _, b_ 01 = _a_ 01 _σ_ [2] . Therefore,


_dκt_


(172)
_dt_ [=] _[ −]_ [1]


_t_ _dνt_

_dt_ [= 1] _[,]_ _dt_


We can write the expressions for _ht_ and _σt_ [2] [directly, using eqs. (164) and (170), as]


1
_ht_ = _σt_ [2] [=] _[ σ]_ [2] (173)
1 + _t_ _[,]_


To express the latent stochastic interpolant, we can calculate the coefficient _ηt_ for _ϵ_


_ηt_ =


- _b_ 01 _κtνt_ = _σ_ - _t_ (1 _−_ _t_ ) (174)
_a_ 01


We can now write the expression for the latent stochastic interpolant


_zt_ = _σ_ ~~�~~ _t_ (1 _−_ _t_ ) _ϵ_ + _tz_ 1 + (1 _−_ _t_ ) _z_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) _._ (175)


Finally, to express _u_ ( _zt, t_ ) first we calculate


_dηt_ _t_ = _[σ]_ [(1] _[ −]_ _[t][ −]_ _[t]_ [)]

_dt_ _[−]_ 2 _[σ]_ _η_ [2] _t_ 2 ~~�~~ _t_ (1 _−_ _t_ )


_σ_ [2]

= _[σ]_ [2][(1] _[ −]_ [2] _[t]_ [)] _[ −]_ _[σ]_ [2]
2 _σ_ - _t_ (1 _−_ _t_ ) 2 _σ_ - _t_ (1 _−_ _t_ )


= _−σ_
2 _σ_ - _t_ (1 _−_ _t_ )


_dηt_


[(1] _[ −]_ _[t][ −]_ _[t]_ [)] _σ_ [2]

_−_
2 ~~�~~ _t_ (1 _−_ _t_ ) 2 _σ_ - _t_


- _t_
(176)
1 _−_ _t_


(177)


leading to


      _u_ ( _zt, t_ ) = _σ_ _[−]_ [1] _−σ_


~~�~~ _t_ 1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


K FORMULATION FOR VARIANCE PRESERVING _κt, νt_


_√_
For the variance preserving formulation, we set _κt_ =


For the variance preserving formulation, we set _κt_ = _t_ and _ηt_ [2] [+] _[ν]_ _t_ [2] [=] [1] _[−]_ _√_ _[t]_ [.] [Not][e] [that] [if]

_z_ 0 _∼_ _N_ (0 _, I_ ) is Gaussian, this setting leads to the latent stochastic interpolant _zt_ = _tz_ 1 + _[√]_ 1 _−_ _tz_ 0.


_t_ _t_ _t_ _√_

_z_ 0 _∼_ _N_ (0 _, I_ ) is Gaussian, this setting leads to the latent stochastic interpolant _zt_ = _tz_ 1 + _[√]_ 1 _−_ _tz_ 0.

Here _ϵ_ and _z_ 0 have been combined since they both are Gaussian. Let _b_ 01 _/a_ 01 = _C_, then


_√_
_ηt_ [2] [=] _[ C]_


_tνt_ = 1 _−_ _t −_ _νt_ [2] (178)


_√_
= _⇒_ _νt_ = _[−][C]_


_t_ + �( _C_ [2] _−_ 4) _t_ + 4

(179)
2


_t_ + �( _C_ [2] _−_ 4) _t_ + 4


Using above, the expressions for _ht_ and _σt_ [2] [can be derived as]


_a_ ~~_√_~~ 01


_C_ [2] _−_ 4
_t_ [+] 2 ~~_[√]_~~ ( _C_ [2] _−_


(180)
_t_ + ~~�~~ ( _C_ [2] _−_ 4) _t_ + 4


_ht_ =


_t_ 2 _t_ 2 ( _C_ [2] _−_ 4) _t_ +4

~~_√_~~ ~~_√_~~
2 _a_ 01 _t −_ _C_ _t_ + ~~�~~ ( _C_ [2] _−_ 4)


_C_ ~~_√_~~
_t_ _[−]_ 2


~~_√_~~
_t −_ _C_


_C_
_σt_ [2] [=] ~~_√_~~ _t_ ~~�~~ ( _C_ [2] _−_ 4) _t_ + 4 (181)


Choosing _a_ 01 = 1 and _C_ = 2 yields


_√_

_ht_ = 0 _,_ _σt_ [2] [=] ~~_√_~~ [1] _νt_ = 1 _−_

_t_ _[,]_


26


_t_ (182)


The coefficient _ηt_ for _ϵ_ can be calculated as


_ηt_ =


- _b_ 01 - ~~_√_~~ ~~_√_~~
_κtνt_ = 2 _t_ (1 _−_
_a_ 01


_t_ ) (183)


We can now write the expression for the latent stochastic interpolant


 - ~~_√_~~
_zt_ = 2


~~_√_~~
_t_ (1 _−_


_√_
_t_ ) _ϵ_ +


_√_
_tz_ 1 + (1 _−_


_t_ ) _z_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) _._ (184)


Finally, to express _u_ ( _zt, t_ ) first we calculate


_dηt_ _t_ = _−_ 1

    - ~~_√_~~
_dt_ _[−]_ 2 _[σ]_ _η_ [2] _t_ 2 _t_ (1


(185)
_t_ )


_dηt_


~~_√_~~
_t_ (1 _−_


with
_dκt_ ~~_√_~~ 1

_dt_ [=] 2


with

_dκt_


_dνt_
_t_ _[,]_ _dt_


_t_ ~~_√_~~

_dt_ [=] _[ −]_ 2 [1]


~~_√_~~
2


(186)
_t_





we arrive at




1

_u_ ( _zt, t_ ) = _σ_ _[−]_ [1]  _−_ ~~�~~ ~~_√_~~ ~~_√_~~

2 _t_ (1 _−_


1
_ϵ_ + ~~_√_~~
2
_t_ )


1
~~_√_~~
_t_ _[z]_ [1] _[ −]_ 2


 (187)

_t_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


_t_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


Note that above expression is for a particular choice of _a_ 01 = 1 and the ratio _b_ 01 _/a_ 01 = 2, which we
chose for relative simplicity of the final expression above. Other choices can be made, leading to
different expressions.


L GAUSSIAN TRANSITION DENSITIES


Let’s consider a linear SDE of the form


_dzt_ = _htztdt_ + _utdt_ + _σtdwt_ (188)


When the SDE is linear with additive noise, we know that the transition densities are gaussian and
are therefore fully specified by their mean and covariance. From Särkkä & Solin (2019) (Eq 6.2)
these are specified by the following differential equations


_dµt_

(189)
_dt_ [=] _[ h][t][µ][t]_ [ +] _[ u][t]_

_d_ Σ _t_

_t_ _[I]_ (190)
_dt_ [= 2] _[h][t]_ [Σ] _[t]_ [ +] _[ σ]_ [2]


The solution to these is given by (eq. 6.3, 6.4, Särkkä & Solin (2019))


               - _t_
_µt_ = Ψ( _t, t_ 0) _µt_ 0 + Ψ( _t, τ_ ) _u_ ( _τ_ ) _dτ_ (191)

_t_ 0

                  - _t_
Σ _t_ = Ψ( _t, t_ 0)Σ _t_ 0Ψ( _t, t_ 0) _[T]_ + _σ_ ( _τ_ ) [2] Ψ( _t, τ_ )Ψ( _t, τ_ ) _[T]_ _dτ_ (192)

_t_ 0


Where Ψ( _s, t_ ) is the transition matrix. For our specific case of linear SDEs, we have


�� _s_                Ψ( _s, t_ ) = exp _h_ ( _τ_ ) _dτ_ (193)

_t_


Substituting, we get


�� _t_
_µt_ = _µt_ 0 exp


_t_  -  - _t_

_h_ ( _τ_ ) _dτ_ +
_t_ 0 _t_ 0


_t_ �� _t_

exp
_t_ 0 _τ_


   _h_ ( _s_ ) _ds_ _u_ ( _τ_ ) _dτ_ (194)
_τ_


    -     - _t_
Σ _t_ = Σ _t_ 0 exp 2


_t_  -  - _t_

_h_ ( _τ_ ) _dτ_ + _I_
_t_ 0 _t_ 0


_t_  -  - _t_

_σ_ ( _τ_ ) [2] exp 2
_t_ 0 _τ_


   _h_ ( _s_ ) _ds_ _dτ_ (195)
_τ_


27


M GAUSSIAN _z_ 0


For the interpolant (section J)


_zt_ = _σ_ ~~�~~ _t_ (1 _−_ _t_ ) _ϵ_ + _tz_ 1 + (1 _−_ _t_ ) _z_ 0 _,_ _ϵ ∼_ _N_ (0 _, I_ ) _,_ (196)


if _z_ 0 is gaussian, we can replace the linear combination of two normal random variables _ϵ, z_ 0 with a
single random variable _z_ ˆ0 _∼_ _N_ (ˆ _µ,_ Σ) [ˆ] . Assuming _z_ 0 _∼_ _N_ (0 _, I_ ), the mean _µ_ ˆ = 0 and covariance Σ [ˆ]
can be computed as


ˆΣ =                - _σ_ [2] _t_ (1 _−_ _t_ ) + (1 _−_ _t_ ) [2][�] _I_ (197)

= (1 _−_ _t_ )( _tσ_ [2] + (1 _−_ _t_ )) _I_ (198)


Using the reparameterization trick, we can express _z_ ˆ0 in terms of _z_ 0 and write

_zt_ = _tz_ 1 + �(1 _−_ _t_ )( _tσ_ [2] + (1 _−_ _t_ )) _z_ 0 _,_ _z_ 0 _∼_ _N_ (0 _, I_ ) (199)


Note that


_√_
_zt_ = _tz_ 1 +


1 _−_ _tz_ 0 _,_ if _σ_ [2] = 1 (200)


_zt_ = _tz_ 1 + (1 _−_ _t_ ) _z_ 0 _,_ if _σ_ [2] = 0 (201)


Similarly, recall the expression for _u_ ( _zt, t_ ) from section J


      _u_ ( _zt, t_ ) = _σ_ _[−]_ [1]


~~�~~ _t_ 1 _−_ _t_ _[ϵ]_ [ +] _[ z]_ [1] _[ −]_ _[z]_ [0] _[ −]_ _[h][θ]_ [(] _[z][t][, t]_ [)]


_−σ_


If _z_ 0 _∼_ _N_ (0 _, I_ ) is also gaussian, we can combine _ϵ, z_ 0 and write


      _u_ ( _zt, t_ ) = _σ_ _[−]_ [1] _z_ 1 _−_


_z_ 0 _−_ _hθ_ ( _zt, t_ )
1 _−_ _t_


(202)


(203)


- 1 + ( _σ_ [2] _−_ 1) _t_


if we choose _σ_ [2] = 1, then the expression simplifies to


1
_u_ ( _zt, t_ ) = _z_ 1 _−_ ~~_√_~~ 0 _−_ _hθ_ ( _zt, t_ ) (204)
1 _−_ _tz_


Finally, we would like to reiterate that we arrive at the above by assuming _z_ 0 is gaussian. The general
form derived in other sections make no assumptions about the distribution of _z_ 0.


N CHOICE OF PRIOR _p_ 0


The Gaussian distribution, along with a small set of other distributions, enjoys the special privilege of
being Lévy stable. That is, a linear combination of two Gaussian random variables is still a Gaussian
random variable. Lévy stability is the main property behind the original formulation of the simulation
free training of the Gaussian diffusion models, e.g. as in DDPM. In contrast, Laplacian, Uniform and
Gaussian Mixture are not Lévy stable, and thus our experiment with those provides strong evidence
for the general nature of the proposed method. The Gaussian mixture used in our experiment was
constructed by having a component for each training image. Consequently, it is a mixture with a very
large number of components. The current estimate of the encoder being learned was used to encode
the training images, yielding the means of the corresponding components. Standard deviation for
each dimension was fixed to 0 _._ 1. In practice, we simply shuffled the encoding of the training images,
added noise, and used a `stop` _ `gradient` operation to prevent the flow of gradient through the prior.
Since the encoder is also evolving during training, this experiment required _∼_ 3 _×_ more steps to yield
the reported FID. Without `stop` _ `gradient`, the experiment became unstable.


O IMAGENET TRAINING AND EVALUATION DETAILS


We trained our models using the entire ImageNet training dataset, consisting of approximately 1.2
million images. Models are trained with Stochastic Gradient Descent (SGD) with the AdamW


28


_t_ ( _s_ ) = 1 _−_ (1 _−_ _s_ ) _[c]_


1


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0

|Col1|Col2|Col3|
|---|---|---|
||||
||||
||_c_ = 0_._<br>_c_ = 0_._|_c_ = 0_._<br>_c_ = 0_._|
||_c_ = 0_._<br>_c_ = 0_._|_c_ = 0_._<br>_c_ = 0_._|
|||<br>_c_ = 0_._<br>_c_ = 1<br>_c_ = 2|
||<br>_c_ = 5<br>_c_ = 1|<br>_c_ = 5<br>_c_ = 1|

0 0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1

_s_


Figure 4: **Schedule for** _t_ **.** A visualization of the schedule for _t_ ( _s_ ) with _s ∈_ [0 _,_ 1] as _c_ is varied. As _c_
increases, larger _t_ values are favored, thereby sampling interpolants closer to _t_ = 1 more frequently.

.


optimizer (Kingma & Ba, 2014; Loshchilov & Hutter, 2017), using _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 99 _, ϵ_ = 10 _[−]_ [12] .
All models are trained for 1000 epochs using a batch size of 2048, except for the ones reported
in table 1 where they were trained for 2000 epochs. Only center crops were used after resizing
the images to the have the smaller side match the target resolution. For data augmentation, only
horizontal (left-right) flips were used. Pixel values for an image _I_ were scaled to the range [ _−_ 1 _,_ 1] by
computing 2( _I/_ 255) _−_ 1 before feeding to the model. For evaluation, a exponential moving average
of the model’s parameters was used using a decay rate of 0 _._ 9999. The FIDs were computed over the
training dataset, with reference statistics derived from center-cropped images, without any further
augmentation. All FIDs are reported with class conditioned samples. To compute PSNR, sampled
image pixel values were scaled back to the range [0 _,_ 255] and quantized to integer values. Figure 4
visualizes the change of variables discussed in section 4. All reported results use _c_ = 1, resulting
in uniform schedule, for both training and sampling, except for NoisePred and Denoising both of
which resulted in slightly better FID values for _c_ = 2 during sampling.


Each model was trained on Google Cloud TPU v3 with 8 _×_ 8 configuration. For 2000 epochs, the
64 _×_ 64 model took 2 days to train, 128 _×_ 128 took 4 days to train and 256 _×_ 256 took 7 days to
train. For 1000 epochs, the training times were roughly the half of that for 2000 epochs. The training
times for the models reported in table 1 are roughly similar for similarly sized models. Note that our
training setup is not maximally optimized for training throughput.


P ARCHITECTURE DETAILS


The base architecture of our model is adapted from the work described by Hoogeboom et al. (2023)
and modified to separate out Encoder, Decoder and Latent SI models. In the adapted base architecture
feature maps are processed using groups of convolution blocks and downsampled spatially after
each group, to yield the lowest feature map resolution at 16 _×_ 16. A sequence of Self-Attention
Transformer blocks then operates on the 16 _×_ 16 feature map. Note that the transformer blocks in
our adapted architecture operate only at 16 _×_ 16 resolution. Consequently, for a 64 _×_ 64 resolution
input image, two downsamplings are performed, for 128 _×_ 128 resolution, three downsamplings
are performed and for 256 _×_ 256 four downsamplings are performed. All convolutional groups
have the same number of convolutional blocks. The observation space SI models used in this paper
are constructed using this adapted base architecture. To construct Encoder, Decoder and Latent SI


29


|Col1|Col2|4|Col4|4|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|128_ ×_ 128_ ×_ 3|Dense|128_ ×_ 128_ ×_ 6|Conv (x3)|128_ ×_ 128_ ×_ 6|Downsampling|64_ ×_ 64_ ×_ 128|Conv (x3)|64_ ×_ 64_ ×_ 128|Downsampling|32_ ×_ 32_ ×_ 256|Dense|32_ ×_ 32_ ×_ 16|Normalization|32_ ×_ 32_ ×_ 16|Tanh|32_ ×_ 32_ ×_ 16|
||||||||||||||||||


(a) Encoder architecture


32 _×_ 32 _×_ 512


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|28|Col10|28|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|32_ ×_ 32_ ×_ 16|Dense|32_ ×_ 32_ ×_ 512|Upsampling|64_ ×_ 64_ ×_ 256|Conv (x3)|64_ ×_ 64_ ×_ 256|Upsampling|128_ ×_ 128_ ×_ 12|Conv (x3)|128_ ×_ 128_ ×_ 12|Dense|128_ ×_ 128_ ×_ 3|
||||||||||||||


(b) Decoder architecture


|Downsampling2|Col2|
|---|---|
|Downsampling2||


|Conv (x9)|Col2|
|---|---|
|Conv (x9)||


(c) Latent stochastic interpolant model architecture. The blocks shown with dashed boundaries are optional
across different resolutions.


Figure 5: An overview of the architecture of various components for 128 _×_ 128 resolution model.
The architecture for 64 _×_ 64 and 256 _×_ 256 resolutions is similar, except for the difference in the
spatial feature map sizes. See section P for details.


models, we simply partition the base model into three parts. The first part contains two groups of
convolutional blocks, each followed by downsampling, and forms the encoder. An extra dense layer
is added to reduce the number of channels. Further, the output is normalized to have zero mean
and unit standard deviation followed by tanh activation to limit the range to [ _−_ 1 _,_ 1]. Similarly, the
last part contains two groups of convolutional blocks, each followed by upsampling, and forms the
decoder. An extra dense layer is added at the beginning to increase the number of channels. The
remaining middle portion forms the Latent SI model, where two extra dense layers are added, one at
beginning and one at end to increase and decrease the feature map sizes respectively. We show an
overview of the architecture for various components in the fig. 5.


Note that the tanh activation or other forms of scale control, such as normalization, play a crucial
role in preventing the encoder from learning arbitrarily large embeddings and allowing it to achieve
better FID. Without this constraint, the model makes the encoder outputs have large scale to make
denoising easier at later timesteps. This is an important implementation detail that ensures stable
training. Empirically, encoder output normalization yielded more stable training and better FID, than
without anything, at the same number of steps. Addition of tanh further improved the FID.


For different resolutions, the Encoder and Decoder models are fully convolutional and have the same
architecture. The architecture of Latent SI models differs in the presence/absence of the optional
downsampling and upsampling blocks (shown as blocks with dashed boundaries). The 64 _×_ 64 Latent
SI model does not contain any downsampling/upsampling blocks as the encoder output is already
16 _×_ 16. The 128 _×_ 128 model does not contain "Downsampling1" and "Upsampling2" blocks. The
256 _×_ 256 model contains all blocks. All models contain 16 Self-Attention Transformer blocks. To
increase/decrease number of parameters to match model capacities, only the number of convolutional
blocks in groups immediately before and after the Self-Attention Transformer blocks is changed.


All models operate with a 3 _×_ smaller latent dimensionality that the observations. We focused on this
dimensionality ratio to ensure fair comparison with observation-space baselines while maintaining
reasonable latent dimensionality for effective modeling. In earlier experiments we tried other compression ratios including 2 _×_ and 4 _×_, before settling on 3 _×_ . The primary effect of the dimensionality
ratio is on the reconstruction performance. Higher the dimensionality ratio, the harder it is for the
decoder to achieve a high PSNR at the same number of training steps, resulting in worse sample
quality (FID) and longer training times. Lower the dimensionality ratio, less the computational
advantage.


30


Table 5: Comparison with state-of-the-art FID results on ImageNet 128 _×_ 128. Note that these models
have differing sizes, FLOPs and NFEs. The comparison is provided purely for reference.


Method FID


Ours 3.12


SiD2 (Hoogeboom et al., 2024) 1.26
PaGoDA (Kim et al., 2024) 1.48
DisCo-Diff (Xu et al., 2024) 1.73
VDM++ (Kingma & Gao, 2023) 1.75
SiD (Hoogeboom et al., 2023) 1.94
RIN (Jabri et al., 2022) 2.75
CDM (Ho & Salimans, 2022) 3.52
ADM (Dhariwal & Nichol, 2021) 5.91


Q ADDITIONAL SAMPLING DETAILS AND RESULTS


All the results reported in the paper use the deterministic sampler with 300 steps, setting _γt_ = 0 in
eq. (86), except when otherwise stated. fig. 3 and fig. 6 use stochastic sampling with _γt_ _≡_ _γ_ (1 _−_ _t_ ),
where _γ_ is a specified constant. We use Euler (for probability flow ODE) and Euler-Maruyama (for
SDE) discretization for all results, except for qualitative inversion results in fig. 3 and fig. 6. For
the inversion results we experimented with two reversible samplers: 1) Reversible Heun (Kidger
et al., 2021) and, 2) Asynchronous Leapfrog Integrator (Zhuang et al., 2021). While both exhibited
instability and failed to invert some of the images, we found Asynchronous Leapfrog Integrator to be
more stable in our experiments and used it for results in fig. 3 and fig. 6. Figure 7 provides additional
samples for qualitative assessment, complementing fig. 2 in the main paper.


Sampling speed (with 100 steps) for pixel space models is roughly 2.2 images/sec/core for 64x64,
0.95 images/sec/core for 128x128 and 0.21 images/sec/core for 256x256. LSI achieves 2.65 images/sec/core for 64x64, 1.30 images/sec/core, and 0.53 images/sec/core for 256x256. We would like
to emphasize that these numbers exhibit high variance, are highly hardware dependent and can be
significantly impacted by hardware specific optimizations that are not the focus of this paper.


R COMPARISON WITH OTHER METHODS


While the primary focus of this paper is on the theoretical results and their empirical validation,
in table 5 we present comparison with other image generation methods for completeness. We
provide this table purely for reference as these methods are not directly comparable due to differing
model sizes, FLOPs and NFEs. While our best result is comparable, techniques in these works are
complementary to our method. We leave it as future work to explore this direction.


S USE OF LLM


LLMs were used to help create some of the figures in the paper.


31


Original _γ_ = 0 _._ 25 _γ_ = 0 _._ 5 _γ_ = 1 _._ 0


Figure 6: LSI supports flexible sampling.


32


_λ_ = 0 _._ _λ_ = 1 _._ _λ_ = 3 _._ _λ_ = 5 _._ _λ_ = 0 _._ _λ_ = 1 _._ _λ_ = 3 _._ _λ_ = 5 _._


Figure 7: LSI supports CFG sampling.


33