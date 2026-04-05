# OPERATOR FLOW MATCHING FOR TIMESERIES FORE
## CASTING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Forecasting high-dimensional, PDE-governed dynamics remains a core challenge
for generative modeling. Existing autoregressive and diffusion-based approaches
often suffer cumulative errors and discretisation artifacts that limit long, physically consistent forecasts. Flow matching offers a natural alternative, enabling
efficient, deterministic sampling. We propose TempO, a time-conditioned latent
flow matching method mimicking classical PDE evolution operators. We introduce an attention-based multiscale autoencoder, a latent Fourier vector field regressor, and decoupled spatial and temporal processing for temporally coherent
and accurate rollouts. We prove an upper bound on FNO approximation error and
empirically show that TempO outperforms state-of-the-art baselines across three
benchmark PDE datasets, and spectral analysis further demonstrates superior recovery of multi-scale dynamics, while efficiency studies highlight its parameterand memory-light design compared to attention-based or convolutional regressors.


1 INTRODUCTION


Generative artificial intelligence has brought unparalleled creative and scientific potential, with models capable of producing images (Hatamizadeh et al., 2025), video (Bar-Tal et al., 2024), audio (Ju
et al., 2024), and text (Grattafiori et al., 2024) that rival human quality. From autoregressive transformers to diffusion models and energy-based approaches, the landscape of generative AI is rich and
diverse, offering multiple pathways to model complex data distributions. At the core of this revolution are probabilistic generative models, which learn to sample from complex, high-dimensional
distributions. Among these, flow matching models have emerged as a class of generative models
which learn to transform a simple prior distribution to a more complex data distribution as a continuous transformation. This direct, simulation-free approach enables both efficiency and precision,
offering a new lens on modeling complex systems (Lipman et al., 2023).


Despite recent advances, forecasting high-dimensional temporal dynamics remains challenging.
Deep learning models are computationally expensive and often fail catastrophically after a few dozen
timesteps due to compounding errors in autoregressive predictions, particularly in stiff or chaotic
systems (Ansari et al., 2024; Raissi et al., 2019). Even with the advent of large language models
and their remarkable ability to generate, models that attempt to leverage them for forecasting introduce truncation and quantisation errors due to tokenised representations which further exacerbate
cumulative errors (Ansari et al., 2024), offering little practical benefit relative to their computational
cost (Tan et al., 2024). Modern generative models have been proven capable of generating visually compelling and coherent videos (John et al., 2024), but critically lack the fine-grained control
required to be used in scientific and engineering contexts.


Recent foundation models for forecasting include GenCast for weather (Price et al., 2025) and
Chronos for general time series (Ansari et al., 2024), demonstrate the promise of large-scale pretraining. These models leverage massive datasets across multiple domains resulting in strong zero- and
few-shot transferability. Chronos captures coarse, long-range correlations remarkably long timespans; however, the granularity, i.e. prediction length still falls at an average of 22 across 55 datasets,
with only 7 tasks exceeding 30 steps (Ansari et al., 2024). GenCast, likewise, can generate 15-day
global weather forecasts, but at a granularity of 12 hours, around 30 steps. True progress requires
models capable of deterministic yet flexible generation, able to explore plausible trajectories while
respecting physical constraints to then select precise forecasts out of the space of plausible predic

1


tions (Guo et al., 2025). Although the short to mid term range is a popular horizon to explore (Lim
et al., 2025b), the goal is to generate long-horizon predictions on the order of 30 timesteps or more,
generating trajectories that are not just plausible, but physically consistent.


Fundamentally, models relying on discretisation or tokenisation are not ideal for continuous, Partial
Differential Equation (PDE)-governed dynamics. Demonstrating smooth trajectories in state space
which generalise to long forecasting horizons would show greater fidelity to the underlying physics.
Other existing efforts which leverage diffusion (Molinaro et al., 2025; Yao et al., 2025; Huang et al.,
2024) move toward more natural representations and, in some settings, benefit from the stochasticity inherent to diffusion, e.g. for uncertainty quantification or modeling ensembles of plausible
trajectories in chaotic regimes (Lippe et al., 2023). However, their stochastic sampling procedures
can introduce computational overhead and variance when the goal is to learn a sharp, deterministic
operator. Flow matching provides a complementary alternative: vector field regression aligns naturally with learning PDE operators, which themselves describe deterministic time derivatives, and
yields efficient Ordinary Differential Equation (ODE)-based sampling without iterative denoising.
Existing flow matching methods have individually worked toward video generation (Davtyan et al.,
2023; Jin et al., 2025; Holzschuh et al., 2025; Serrano et al., 2024) and PDE single-step prediction (Kerrigan et al., 2023), but thus far have not been thoroughly tested for long-horizon temporal
forecasting and do not design for the deterministic and stable rollouts required for such tasks.


In this work, we propose Temporal Operator flow matching (TempO), the first principled integration
of time-conditioned latent flow matching with neural operators mimicking classical PDE evolution operators, offering an efficient, stable and spectrally accurate alternative to autoregressive and
diffusion-based PDE forecasting. TempO is built around four key innovations:


1. We design a multi-headed attention–enhanced autoencoder that learns multiscale latent
representations, allowing the flow field to jointly capture global structures and fine local
features.


2. We construct time-conditioned latent spectral embeddings that encode PDE states in a
frequency-aware latent space, enabling smooth flow interpolation and preserving highfrequency dynamics.


3. Motivated by the structure of PDE evolution operators, TempO explicitly decouples spatial
and temporal processing via channel folding, using operator layers for spatial modes and
latent flow dynamics for temporal evolution.


4. We incorporate sparse conditioning, providing computational and data efficiency by conditioning the flow only on two prior timesteps without degrading accuracy.


We derive theoretical error bounds that characterize the efficiency and expressivity of TempO, and
showcase its performance on PDE benchmarking datasets accompanied with a spectral analysis
showing a distinct advantage in capturing the essential dynamics required for forecasting. Empirically, we see a 16% lower error when predicting vorticity of 2D incompressible Navier Stokes, with
Pearson correlations remaining above 0.95 for a 40 step forecasting horizon, demonstrating its stable
temporal forecasting and high quality generation capability.


2 RELEVANT WORKS


Application-specific models for scientific data have also seen development: GenCFD (Molinaro
et al., 2025) proposes a conditional diffusion model to generate the underlying distributions of high
fidelity flow fields. Kerrigan et al. (2023) propose the first extension of Fourier Neural Operators
(FNOs) to flow matching tasks and predicts plausible fluid dynamic fields, Functional Flow Matching. Shi et al. (2025) builds on this concept and extends it to learning stochastic process priors on
function spaces. Similarly, Lim et al. (2025a) extend denoising diffusion models to function space,
introducing Denoising Diffusion Operators for unconditional prior generation; Yao et al. (2025) extends this to conditional posterior sampling under observation constraints and sees state-of-the-art
performance for multi-resolution PDE tasks, as compared to its competitor DiffusionPDE Huang
et al. (2024) which originally demonstrated strong performance in solving PDEs with partial observations. Such methods have thus far focused on static prediction, i.e., generating diverse samples of
plausible PDE solutions, rather than deterministic temporal rollouts.


2


Models designed to predict sequences of future states include the aforementioned large-scale
Chronos and GenCast (Ansari et al., 2024; Price et al., 2025). In addition, pyramidal flow matching (Jin et al., 2025) produces state-of-the-art video generation compared to leading models (Zheng
et al., 2024), representing a successful flow matching foundation model. (Tamir et al., 2024) present
conditional flow matching for time series, succeeding in long 1D trajectories where neural ODEs
fail, but has not scaled to 2D spatiotemporal data. Physics-Based Flow Matching (Baldan et al.,
2025) adds a PINN-style loss (Raissi et al., 2019) to flow matching for surrogate model. Kollovieh
et al. (2024) extends this with Gaussian processes for forecasting tasks outside of scientific machine
learning. We focus instead on models that fall between these two categories, scaling reasonably to
2D data to match common PDE settings. Li et al. (2025) further leverage latent space modeling and
coarsely sampled diffusion for PDE generation on irregular grids.


TempO occupies a distinct methodological position in this landscape: it is conditional and deterministic, built to learn an operator-valued transport that maps an initial PDE state to its evolved state,
rather than to sample diverse static solutions. TempO also shares the same spectral motivation as
prior works, e.g. Functional Flow Matching, but differentiates itself in a crucial way: instead of
using an FNO as a denoiser inside a sampler, TempO uses an FNO to parameterize a latent-space
velocity field that is time-conditioned and integrated as an ODE. This design enables deterministic,
operator-valued rollouts rather than stochastic sample generation, and directly targets long-horizon
forecasting stability.


3 METHOD


We begin by developing the background which is then used to construct our method. Flow matching
learns a time-dependent velocity field _vθ_ ( _z, t_ ) defining an ODE in the latent space:
_dz_ ( _t_ )

= _vθ_ ( _z_ ( _t_ ) _, t_ ) _,_ _z_ (0) _∼_ _π_ 0 _,_ (1)
_dt_

where _π_ 0 is a simple prior (e.g., Gaussian). Integrating this ODE transports samples to the latent
data distribution _π_ 1, see Appendix B. Training reduces to a regression objective that matches the
model velocity field to a target velocity along interpolating probability paths (Lipman et al., 2023).
This enables simulation-free sampling from complex distributions.


Table 1: Representative Path Choices in Flow Matching Models.


Path _at_ _bt_ _ct_ Parameter definitions

Affine-OT [1] _t_ 0 (1 _−_ (1 _−_ _ϵ_ min) _t_ ) [2] _ϵ_ min _≥_ 0: min. noise level
RIVER [2] (1 _−_ (1 _−_ _σ_ min) _t_ ) _t_ _σ_ [2] _σ_ _≥_ 0: noise scale, _σ_ min _≥_ 0: min. noise
SLP [3] (1 _−_ _t_ ) _t_ _σ_ min [2] [+] _[ σ]_ [2] _[t]_ [(1] _[ −]_ _[t]_ [)] _σ, σ_ min _≥_ 0: variance parameters
VE-diff [4] 1 0 _σt_ [2] _σt_ : geometric schedule, _σ_ min _, σ_ max _>_ 0
VP-diff [4] exp( _−_ [1] 2 _[T]_ [(1] _[ −]_ _[t]_ [))] 0 1 _−_ exp( _−T_ (1 _−_ _t_ )) _β_ min _, β_ max _>_ 0, _T_ ( _t_ ) = �0 _t_ _[β]_ [(] _[s]_ [)] _[ ds]_

1 (Lipman et al., 2023), 2 (Davtyan et al., 2023), 3(Lim et al., 2025b),4 (Ryzhakov et al., 2024)


A key component of flow matching is the choice of the probability density path _pt_ interpolating
between the reference distribution _π_ 0 and the target _π_ 1. We focus on Gaussian conditional paths
with closed-form velocity fields:

_pt_            - _Z_ _|_ _Z_ [˜] := ( _Z_ 0 _, Z_ 1)� = _N_            - _Z_ �� _atZ_ 0 + _btZ_ 1 _,_ _c_ 2 _t_ _[I]_            - _,_


where _at, bt, ct_ define the path (Table 1). This pair-conditional path is defined for a specific transition ( _Z_ 0 _, Z_ 1), and the marginal interpolant is obtained by averaging over all pairs: _pt_ ( _Z_ ) =
E( _Z_ 0 _,Z_ 1)[ _pt_ ( _Z_ _|_ _Z_ 0 _, Z_ 1)]. While _π_ 0 is typically a standard Gaussian, intermediate densities _pt_
can follow diffusion-inspired, optimal transport, or other custom schedules.


To parameterize _vθ_, we modify FNOs, which approximate mappings between functions via spectral
convolution layers. Given input _u_, the FNO parameterizes an operator as _Gθ_ : _u_ _�→_ _u,_ ˜ _u_ ˜ : _D_ _→_
R _[c]_ [out], that maps _u_ to an output function _u_ ˜. Iterative Fourier layers perform spectral transformations
of the input ˆ _u_ ( _k_ ) = _F_ [ _u_ ]( _k_ ) _,_ _u_ ˆ˜( _k_ ) = _Rθ_ ( _k_ ) _·_ ˆ _u_ ( _k_ ) _,_ followed by an inverse Fourier transform back
to the spatial domain; _u_ ˜( _x_ ) = _F_ _[−]_ [1] [ _u_ [ˆ] ˜]( _x_ ) _,_ with _Rθ_ ( _k_ ) being learnable Fourier-mode weights and _F_
denoting the Fourier transform. This spectral representation allows the FNO to efficiently capture
long-range dependencies and global correlations in the data.


3


3.1 TEMPORAL OPERATOR FLOW MATCHING (TEMPO)


Using an FNO-inspired regressor to learn the vector field of a flow matching model has a number of
benefits, namely, the added expressivity that the Fourier representation provides at a low computational cost thanks to highly optimised Fast Fourier Transform (FFT) operations. Building on prior
analysis of FNOs for solving PDEs (Kovachki et al., 2021), we show that an FNO-inspired regressor
can achieve an upper bound on approximation error for flow matching models and we provide a
lower bound on the accuracy achievable by sampler-based methods (e.g., Transformer or U-Net) in
relation to their number of parameters.

**Theorem 3.1** (FNO regressor constructive upper bound) **.** _Let_ T _[d]_ _be the d–torus._ _Fix s, s_ _[′]_ _≥_ 0 _and_
_let U_ _⊂_ _H_ _[s]_ (T _[d]_ ) _be compact._ _Suppose G_ : _U_ _→_ _H_ _[s][′]_ (T _[d]_ ) _is continuous and satisfies |G_ [�] ( _u_ )( _k_ ) _|_ _≤_
_Cλ_ (1 + _|k|_ ) _[−][p]_ _for all u_ _∈U, k_ _∈_ Z _[d]_ _, with constants Cλ_ _>_ 0 _, p_ _>_ 0 _._ _If p_ _>_ _s_ _[′]_ + _[d]_ _[and we define]_


_P_ F _NO_ ( _ε_ ) ≲ _ε_ _[−]_ _[d/α]_ _,_


_such that_ sup _u∈U ∥G_ ( _u_ ) _−Gθ_ ( _u_ ) _∥H_ _s_ _[′]_ _≤_ _ε._ _The hidden constants depend only on d, s, s_ _[′]_ _, U, Cλ and_
_mild/logarithmic factors._


This result is in line with the estimates and arguments made in (Kovachki et al., 2021).


_Sketch of proof of Theorem 3.1._ (Spectral truncation.) The Fourier decay assumption implies that
high-frequency modes of _G_ ( _u_ ) contribute at most _O_ ( _K_ _[−]_ [2] _[α]_ ) to the _H_ _[s][′]_ -error. Choosing _K_ _≍_ _ε_ _[−]_ [1] _[/α]_
makes this truncation error _≤_ _ε/_ 2.


(Finite-dimensional reduction.) For this cutoff _K_, the operator _GK_ is determined by _O_ ( _K_ _[d]_ ) Fourier
coefficients, and inputs can likewise be restricted to finitely many low modes without significant
loss of accuracy. Thus the problem reduces to approximating a continuous map between compact
subsets of R _[m]_ [in] and R _[m]_ [out], with _m_ out _≍_ _K_ _[d]_ .


(Approximation by networks.) Standard universal approximation results (or the constructive FNO
design in (Kovachki et al., 2021)) ensure that such a finite map can be uniformly approximated by a
network with _O_ ( _K_ _[d]_ ) parameters, up to mild logarithmic factors.


(Conclusion.) Combining these errors yields an overall accuracy _ε_ with parameter count _P_ ≲ _K_ _[d]_ _≍_
_ε_ _[−][d/α]_, proving the claim.


**Proposition** **3.2** (Transformer/UNet Sampler-based lower bound) **.** _Under the_ _assumptions_ _of_ _The-_
_orem_ _3.1,_ _consider_ _any_ _learner_ _that_ _observes_ _each_ _u_ _∈U_ _only_ _through_ _n_ _fixed_ _point_ _evaluations_
_and applies a parametric map with P_ _real parameters, required in the worst case to reconstruct all_
_Fourier modes up to radius K_ _≍_ _ε_ _[−]_ [1] _[/α]_ _._ _Then necessarily_


_n_ ≳ _ε_ _[−]_ _[d/α]_ _,_ _P_ sampler( _ε_ ) ≳ _ε_ _[−]_ _[βd/α]_ _,_


_for some architecture–dependent β_ _≥_ 1 _(optimistically β_ = 1 _when only diagonal mode-wise maps_
_are_ _needed,_ _generically_ _β_ = 2 _for_ _arbitrary_ _dense_ _linear_ _maps)._ _These_ _bounds_ _are_ _information-_
_theoretic and asymptotic, up to constants and mild/logarithmic factors._


_Sketch of proof of Proposition 3.2._ (Sampling necessity.) The _K_ –mode subspace _VK_ has dimension
_DK_ _≍_ _K_ _[d]_ . Sampling at _n_ points defines a linear map _S_ : _VK_ _→_ C _[n]_ . For _S_ to be injective on _VK_,
its matrix must have rank _DK_, hence _n ≥_ _DK_ _≍_ _K_ _[d]_ .


(Parameter complexity.) After sampling, the learner implements a parametric map _M_ : C _[n]_ _→_ C _[m]_ .
To represent arbitrary linear maps on the _DK_ -dimensional coefficient space (e.g. arbitrary diagonal
multipliers), the parameter family must have at least _P_ ≳ _DK_ degrees of freedom. For fully general
dense linear maps one needs _P_ ≳ _DK_ [2] [.]


(Conversion.) Substituting _K_ _≍_ _ε_ _[−]_ [1] _[/α]_ (from the theorem) gives _n_ ≳ _ε_ _[−][d/α]_ and _P_ ≳ _ε_ _[−][βd/α]_ with
_β_ = 1 (optimistic) or _β_ = 2 (dense case), establishing the lower bound, see Appendix A for the
extended proof.


4


_Cλ_ (1 + _|k|_ ) _for all u_ _∈U, k_ _∈_ Z _, with constants Cλ_ _>_ 0 _, p_ _>_ 0 _._ _If p_ _>_ _s_ + 2 _[and we define]_

_α_ := _p −_ _s_ _[′]_ _−_ _[d]_ _[>]_ [ 0] _[, then for every][ ε >]_ [ 0] _[ there exists a Fourier Neural Operator][ G][θ]_ _[with]_


2 _[>]_ [ 0] _[, then for every][ ε >]_ [ 0] _[ there exists a Fourier Neural Operator][ G][θ]_ _[with]_


**Corollary 3.3** (FNO vs sampler scaling) **.** _From Theorem 3.1 and Proposition 3.2 one has_


_P_ F _NO_ ( _ε_ ) ≲ _ε_ _[−]_ _[d/α]_ _,_ _P_ sampler( _ε_ ) ≳ _ε_ _[−]_ _[βd/α]_ _._


_Hence, whenever β_ _>_ 1 _, FNOs achieve the same accuracy ε with asymptotically fewer parameters_
_than sampler–based learners._


**TempO** Consequently, we propose TempO, a framework capable of long rollout PDE forecasting via a multiscale attention-based autoencoder, time-conditioned FNO vector field regressor, and
channel folding for both efficiency and enhanced temporal coherency, reflecting the PDE evolutionoperator perspective that motivates our decoupling of spatial and temporal processing. Together
with temporal conditioning, these define a novel, end-to-end trainable model for predicting latent
dynamics.


Let _fϕ_ : R _[X]_ _→_ R _[Z]_ denote an encoder mapping data points _x_ to latent embeddings _z_ = _fϕ_ ( _x_ ).
We use a _multi-headed_ _attention-enhanced_ _autoencoder_ for multiscale embeddings that are wellsuited for subsequent temporal conditioning and flow-based evolution. The use of attention layers
and residual blocks preserves multi-scale spatial correlations while compressing the data for more
efficient processing. We can then define a latent-space velocity field described by 1 where _vθ_ is
parameterized by an FNO, which provides a spectral inductive bias for learning PDE-consistent
dynamics.


Under the standard flow-matching assumption that the learned drift has bounded Lipschitz constant,
the global forecasting error grows no faster than exp��01 _[L][t][ dt]_ - _ε_ yielding polynomial accumulation,
consistent with classical ODE stability. In contrast, autoregressive models accumulate errors through
the product of stepwise Jacobians, [�] _t_ _[∥][J][t][∥]_ [,] [which] [can] [increase] [much] [more] [rapidly.] [We] [further]

mitigate the expected error accumulation via sparse conditioning.


_Sparse conditioning_ provides computational and data efficiency by leveraging the locality and temporal coherence inherent to PDEs, and additionally allows the model to condition on fixed points
during rollout rather than recursively using its own predictions as inputs (Davtyan et al., 2023; Lim
et al., 2025b). For some discrete-time sequence _{xt}_ _[N]_ _t_ =1 [with] _[x][t]_ _[∈X]_ [,] [its] [latent] [representation] [is]
given by _{zt}_ _[N]_ _t_ =1 [, where] _[ z][t]_ [=] _[ f][ϕ]_ [(] _[x][t]_ [)][.] [For a prediction horizon] _[ T]_ _[∈{][L, . . ., N]_ _[−]_ [1] _[}]_ [ with sequence]
length _L_, the objective is to predict the next latent embedding _zT_ +1. We define a reference embedding to be _zT_, corresponding to the most recent observation prior to the prediction target, and a
conditioning embedding as some observation selected at a timestep _τ_ _∈{T_ _−_ _L, . . ., T_ _−_ 1 _}_ . These
two embeddings are concatenated with the temporal offset, defined as ∆= _T −τ_, which is the extent
of temporal data the model is provided to predict the next-step embedding, _z_ ˆ _T_ +1 = _fθ_ ( _zT, zτ_ _,_ ∆).
In practice, we fix the conditioning to one of the provided initial steps, effectively pinning the generation against a known correct solution and incrementing the temporal offset and updating the
reference embedding: this results in significantly more stable rollouts.


We then propose _channel folding_, a key architectural contribution that allows 2D FNO layers to process time-varying latent fields without temporal blurring. This folding preserves a clean separation
between spatial operator structure and temporal latent dynamics, reflecting the semigroup property
of PDE evolution operators. We collapse the batch and channel axes into a single “effective batch”
dimension _u_ _[′]_ _∈_ R [(] _[B][·][C]_ [)] _[×][T][ ×][H][×][W]_ as input to the FNO. This folding operation effectively treats each
channel of each sample as an independent element within the extended batch. As a consequence,
the FNO is applied identically across all channels but without cross-channel mixing at this stage.
This disentangles _t_ (within the flow matching integration step) from _τ_ (the PDE timestep), which is
explicitly provided as part of the conditioning tuple _C_ = _{zτ_ _,_ ∆ _}_ .


This _time-conditioned_ _FNO_ then operates over latent temporal embeddings as functions on their
spatial domain _vθ_ ( _z, t_ ) = _Gθ_ ( _z_ ) to learn the time-dependent vector field that transports a reference
latent distribution _π_ 0 to the latent data distribution _π_ 1. This differs from prior FNO-based generative
models, which operate either unconditionally or autoregressively in input space rather than in a timeconditioned latent ODE. By leveraging the spectral inductive bias of FNOs, the learned velocity field
models cross-scale correlations, thereby stabilizing flow matching across long horizons. Combined
with spectral latent embeddings, this produces temporally coherent latent trajectories that maintain
high-frequency fidelity during transport.


5


4 EXPERIMENTS


The TempO is evaluated with the goal of assessing its ability to learn accurate stochastic latent-space
dynamics and forecast high-dimensional solution fields over medium to long time horizons. We test
our method over PDE datasets which pose challenging spatio-temporal correlations and multiscale
features, making them a natural testbed for latent flow-based modeling.


Our proposed TempO was set against five key methods. The state-of-the-art video generation method
based on a U-Net shaped Vision Transformer (ViT) and modified optimal transport path Random
frame conditioned flow Integration for VidEo pRediction (RIVER) proposed by Davtyan et al.
(2023) matches or surpasses common video prediction benchmarks using 10x fewer computational
resources (Davtyan et al., 2023). We also include the baseline conditional flow matching Lipman
et al. (2023) which implements a U-Net trained using a theoretically optimal affine optimal transport (Affine-OT) path. The stochastic linear path (SLP) was proposed by Lim et al. (2025b), tested
with a ViT to directly address the challenges of spatiotemporal forecasting for PDE datasets. The
Transformer-based latent space flow matching method with Affine-OT proposed by Dao et al. (2023)
further demonstrates competitive performance in image generation using latent flow matching compared against both flow matching models and diffusion models (Phung et al., 2023; Ho et al., 2020)
among others. We also evaluate both variance preserving diffusion (VP-diff) and variance exploding
diffusion (VE-diff) paths which generalise the Denoising Diffusion Probabilistic noise perturbation
model and a score-based model to flow matching paths, respectively (Ho et al., 2020; Song et al.,
2021). Ryzhakov et al. (2024) establishes strong theoretical backing for both paths.


We then ablate the specific implementation of the methods (consisting of a specific architecture and
a specific probability path). In summary, the choice of regressor includes our proposed TempO regressor, and additionally implement a ViT regressor (Davtyan et al., 2023; Lim et al., 2025b) and a
classic U-Net regressor (Lipman et al., 2023). We pretrain a convolutional autoencoder with residual
and attention blocks to obtain a compressed latent representation of the dynamics, see Appendix D.
We additionally compare against baseline, non-flow matching models FNO2D, FNO3D, Wavelet
Neural Operator (WNO)2D, WNO3D, and a U-Net to contextualize performance, with further details in Appendix F (Li et al., 2021; Tripura & Chakraborty, 2022).


All flow matching methods were conditioned using sparse conditioning, and baseline 2D methods
mapping 10 timesteps to predict the following, and 3D methods directly mapping the block of 10
timesteps to directly predict the remaining 40. Flow matching models are then supervised by each
probability density paths described in Table 1. Further details in Appendix E. The Adam optimiser
with a learning rate of 1e-4 was used for the FNO, and 5e-5 for the ViT and U-Net regressors.
Models are trained on an 80/10/10 training to validation to test data splits.


We evaluate our models on three spatiotemporal PDE datasets: the shallow water equation (SWE),
which simulate 2D free-surface flows; 2D reaction diffusion (RD-2D) exhibiting multiscale nonlinear patterns; and 2D incompressible Navier-Stokes vorticity (NS- _ω_ ) dataset capturing chaotic
turbulent dynamics. During training, models are sparsely conditioned on the first 15 frames and
tasked with predicting the subsequent frame at resolutions of 1 _×_ 128 _×_ 128 (shallow water equation
(SWE)), 2 _×_ 128 _×_ 128 (2D reaction diffusion (RD-2D)), and 1 _×_ 64 _×_ 64 (2D incompressible
Navier-Stokes vorticity (NS- _ω_ )), see Appendix G.


5 RESULTS


Overall, TempO outperforms the methods proposed by Lim et al. (2025b); Song et al. (2022); Lipman et al. (2023) and Davtyan et al. (2023) as well as the ablated flow-matching methods. For
results predicting NS- _ω_ in Table 2, we observe a 16% improvement in MSE and an 11.4% lower
spectral MSE, producing spatially and spectrally accurate next steps. Its lower RFNE indicates
reduced scale-independent error, while SSIM shows improved fidelity in local features, critical for
the localized vorticity patterns where small spatial distortions significantly affect downstream evolution (Majda & Bertozzi, 2001). PSNR and Pearson see lower normalised ranges in values, indicating
that large scale features like the vorticity intensity and global structure agreement, respectively, are
more easily captured across all models, with a clear advantage by TempO; additional visualisations
in Appendix I. We observe that the baseline FNO models outperform across the board for next step,
but critically fail at longer rollouts evidenced by the 58.1% worse MSE/time of FNO-3D versus


6


Table 2: NS- _ω_ Results: Comparison of TempO, U-Net, and ViT models.


Regressor Path MSE _↓_ SpectralMSE _↓_ RFNE _↓_ PSNR _↑_ Pearson _↑_ SSIM _↑_ MSE/time _↓_


1 (Dao et al., 2023), 2(Lim et al., 2025b; Song & Ermon, 2020), 3(Davtyan et al., 2023), 4(Lim et al., 2025b); results reported from original paper trained on same dataset., 5(Lipman et al., 2023)


TempO. The orders of magnitude better next step prediction can be seen to blur significantly in
rollout visualisations, provided for top next-step model FNO-2D in Appendix N


We select top performing comparisons for SWE and RD-2D, Table 3), where TempO maintains
superior performance. In SWE, it achieves a 28.8% lower SpectralMSE and higher PSNR, indicating
faithful amplitude, spectral content, and structural coherence with sharp boundaries preserved, see
Appendix J for additional visualisations and ablated comparisons. Overall MSE is reduced by 30.8


In RD-2D, U-NetAffine-OT competes closely with TempO, benefiting from translation-equivariant convolutional layers that capture multi-scale dynamics and repeating local structures (Cohen & Welling,
2016). Both TempO and the U-Net have nearly matched PSNR, Pearon, and SSIM scores, with an
improvement of 11% in SpectralMSE from the TempO. By contrast, the next best ViT regressor
model is 95.6% drop in SpectralMSE, where attention might emphasize low-frequency global structures (Wang et al., 2022; Piao et al., 2024); see visual comparison in Appendix K.


The timeseries forecasting task, see Fig. 1, evaluates how well models capture the underlying PDE.
Models follow the inference protocol used by (Davtyan et al., 2023; Li et al., 2021): a short lead-up
of 9 initial frames is provided, with the baseline FNO using all 9; the sparsely conditioned flowmatching models use only the last two frames in the lead-up as conditioning and reference frames.
The conditioning frame is then pinned and the temporal offset vector is incremented while the reference frame is set to timestep _t −_ 1 to generate timestep _t_ . TempO maintains Pearson correlation
above 0.98 over 40 forecasted timesteps, indicating stable amplitude and phase tracking. The ViT
regressor holds above 0.95 for 20 steps before degrading, while the flow matching baseline (Lipman
et al., 2023) shows steady decline. This suggests TempO effectively mimics the dynamics without
significant error accumulation. This is further demonstrated by visualisations of the vorticity field at
key timesteps in Fig. 1 (right), where _t_ = 35 most clearly shows TempO’s faithful capture of turbulent eddies in comparison to the ViT regressor, which fails to predict the small vortical structure.


7


TempO


ViT


U-Net


Baselines


RIVER **5.63e-02** **3.84e-02** **2.50e-01** **25.19** **0.969** 0.786 **2.67e-02**
Affine-OT 5.77e-02 3.98e-02 2.54e-01 25.08 0.968 **0.789** 2.91e-02
VP-diff 8.10e-02 5.34e-02 2.85e-01 23.61 0.955 0.731 2.29e-01
VE-diff 2.96e-01 1.73e-01 5.60e-01 17.98 0.821 0.373 5.02e-01


Affine-OT [1] 6.75e-02 4.38e-02 2.72e-01 24.40 0.962 0.758 8.71e-02
RIVER [2] 6.88e-02 4.33e-02 2.73e-01 24.32 0.962 0.750 3.85e-02
VP-diff [3] 7.77e-02 4.95e-02 2.89e-01 23.79 0.956 0.729 6.65e-02
VE-diff [3] 1.63e+00 9.27e-01 1.35e+00 10.67 0.118 0.024 1.67e+00


VP-diff [4] 4.05e-01 3.26e-01 6.71e-01 16.62 0.756 0.323 2.66e-01
RIVER 4.08e-01 3.28e-01 6.74e-01 16.59 0.752 0.321 2.79e-01
Affine-OT [5] 4.10e-01 3.42e-01 6.76e-01 16.57 0.751 0.324 2.82e-01
VE-diff [4] 5.02e-01 3.70e-01 7.48e-01 15.68 0.694 0.263 2.92e-01


FNO-2D _6.09e-04_ _4.27e-04_ _2.54e-02_ _44.85_ _1.000_ _0.992_ 1.92e-01
FNO-3D 1.06e-01 7.34e-02 3.37e-01 22.46 0.945 0.645 6.37e-02
WNO-2D 3.72e-03 2.83e-03 6.06e-02 36.99 0.998 0.966 5.19e-01
WNO-3D 2.23e-01 1.19e-01 4.97e-01 19.209 0.868 0.452 2.05e-01
U-Net 2.47e-03 1.92e-03 4.83e-02 38.772 0.999 0.976 1.66e+00


1 (Dao et al., 2023), 2(Davtyan et al., 2023), 3 (Lim et al., 2025b; Song & Ermon, 2020), 4(Ryzhakov et al., 2024), 5(Lipman et al., 2023)


Table 3: SWE and RD-2D Results: Comparison of TempO, U-Net, and ViT models.


Dataset Method MSE _↓_ SpectralMSE _↓_ RFNE _↓_ PSNR _↑_ Pearson _↑_ SSIM _↑_ MSE/time _↓_


SWE


RD-2D


TempOAffine-OT **6.64e-05** **5.65e-05** **7.64e-03** **46.5** **0.998** **0.997** **1.60e-03**
ViTAffine-OT [1] 9.59e-05 7.93e-05 9.06e-03 44.9 0.997 0.995 7.02e-03
ViTVP-diff [2] 1.30e-04 8.81e-05 1.05e-02 43.6 0.996 0.993 1.61e-03
ViTRIVER [3] 2.99e-04 1.67e-04 1.63e-02 40.0 0.992 0.981 6.96e-03
ViTSLP [4] 6.60e-04 - 1.28e-01 36.1 - 0.93 

TempOAffine-OT **2.76e-05** **2.18e-05** **3.29e-02** **65.7** **1.000** **0.999** **1.89e-02**
U-NetAffine-OT [5] 3.09e-05 2.45e-05 3.57e-02 65.2 0.999 **0.999** 1.95e-02
ViTAffine-OT 6.30e-04 4.40e-04 1.67e-01 52.2 0.987 0.986 2.04e-02
ViTSLP [4] 3.56e-04 - 1.16e-01 34.3 - 0.90 

Figure 1: **Prediction performance comparison for NS-** _ω_ **.** _Left:_ _Pearson correlation across fore-_
_casted timesteps._ Forty timesteps are predicted by TempO, ViT, U-Net, and top performing baseline
FNO-3D averaged over 20 initial conditions on the withheld test set plotted as means and standard
deviations. The Pearson correlation coefficient shows significant degradation for the U-Net, oscillatory behavior and degradation for the ViT, and consistently stable values above 0 _._ 95 for TempO.
The FNO-3D baseline exhibits a high variance in the first timesteps, attributed to the direct time
convolution _Right:_ _Predicted_ _vorticity_ _fields._ True data (a), TempO (b), ViTAffine-OT (c), U-NetVP-diff,
(d), and baseline FNO3D (e). At timesteps 5, 15, and 35 (c), (d), and (e) clearly diverge, with (d)
regressing to a noisy diagonal and (e) losing detail, while TempO maintains excellent accuracy.


5.1 SPECTRAL ANALYSIS


To assess whether TempO’s forecasting performance depends strongly on the spectral bandwidth of
the latent FNO, we conduct an ablation in which the number of retained Fourier modes is varied
(Appendix M). We find that the model is remarkably stable across a wide range of truncation levels:
reducing from 16 to 8 or even 4 modes yields nearly identical MSE, spectral error, and long-horizon
stability. This aligns with the spectral energy distribution observed in Fig. 2, where NS- _ω_ exhibits


8


The spectral analysis of TempO versus the top alternative ViTAffine-OT and the baseline UNetAffine-OT (Lipman et al., 2023) in Fig. 2 examines the scale-resolved error via the energy per
wavenumber _k_, or at the scale of _k_ [1] [.] [This provides scale-resolved context to the SpectralMSE, which]

averages the MSE of the Fourier coefficients to a single metric. For NS- _ω_, the first 8 modes which
cumulatively make up 99% of the total energy, beyond which the modes have negligible contributions to overall flow dynamics, see Appendix H. TempO closely follows the true spectrum compared
to both ViTAffine-OT and U-NetAffine-OT, though all three methods diverge past _k_ = 8. We observe from the
inset of Fig. 2 that TempO exhibits a small residual which fluctuates about 0 whereas the ViTAffine-OT
has a negative and increasing error: the ViT regressor tends to capture the lower wavenumbers well,
but then underestimates the higher wavelengths notably after _k_ = 4.


We observe also that the number of modes retained during the

Modes SpectralMSE

FFT of TempO in Fig. 3 follows the observation of a close

1 8.57e-02 spectral match up until _k_ = 8, where the SpectralMSE sees the
2 4.10e-02 most improvement; however, from 8 modes to 16 modes, the
4 3.98e-02 performance appears to saturate. Fig. 3 demonstrates that up to
8 3.79e-02 8 modes capture the essential dynamics, while the fundamen16 3.74e-02 tal frequency alone is insufficient and likely under-represents

necessary higher frequency components; adding more than 8

Figure 3: Ablation: Fourier mode

modes yields diminishing returns, matching the true spectral

cutoffs with TempO.

analysis; extended metrics support this trend in Appendix L. This empirical saturation beyond 8
modes is consistent with the theoretical expectation in Theorem 3.3, where FNOs are shown to
achieve accuracy with asymptotically fewer parameters by leveraging only the most informative
spectral modes.


Modes SpectralMSE


1 8.57e-02
2 4.10e-02
4 3.98e-02
8 3.79e-02
16 3.74e-02


Figure 3: Ablation: Fourier mode
cutoffs with TempO.


Figure 2: **Spectral graphs for NS-** _ω_ **.** Comparison of energy spectra for TempO, a ViT-based model,
and the U-Net baseline (Lipman et al., 2023). The first eight Fourier modes capture 99% of the
energy, with higher modes negligible. TempO aligns closely, while the ViT underestimates energy
beyond _k_ = 4. The inset bar plot shows TempO oscillating tightly around zero with small deviations,
the ViT producing larger negative deviations, and the U-Net performing markedly worse.


weak high-frequency content. Only highly restrictive truncation, e.g., 2 or 1 mode, produces noticeable degradation, indicating that the autoencoder sufficiently captures the multiscale features and
further confirming that TempO is robust to spectral compression in latent space.


5.2 EFFICIENCY


Finally, we also train the models over varying sequence lengths and measuring next-step prediction
error (MSE) and 40-step forecast error (MSE/time), shown in Table 5. MSE is lowest for shorter
sequences, as the model learns from fewer choices of indices for sparse conditioning during training.
Conversely, MSE/time decreases with longer sequences, reflecting better long-horizon performance.
Notably, TempO’s MSE/time drops faster and plateaus lower than the ViT, indicating better data
efficiency to extrapolate from the same available sequence length.


Model Params FLOPs Mem (MB) NFEs


TempO 0.49M 208M _∼_ 50 560
ViT 3.39M 10M _∼_ 80 942
U-Net 14.0M 555M _∼_ 68 728

Table 4: Model Complexity and Efficiency: number of function evaluationss (NFEs) are averaged
from sampling performed for Table 2 for adaptive
solver dopri5 and tolerances of 1e-5.


TempO is the most lightweight model among the
three choices of regressors, with 7x [˜] fewer parameters than the ViT and 28x [˜] fewer than the U-Net. In
addition, it sees a significantly lower memory usage
compared to the ViT where attention has higher demands and the U-Net where skip-connections hold
onto additional memory.


9


Method Seq MSE MSE/time


TempO


ViT


(Affine-OT)


2 4.92e-02 2.70e-01
5 4.75e-02 3.41e-01
10 5.04e-02 4.94e-02
15 5.61e-02 3.83e-02
25 6.26e-02 4.22e-02


2 6.75e-02 2.71e-01
5 5.43e-02 3.59e-01
10 6.01e-02 1.49e-01
15 6.70e-02 4.53e-02
25 7.68e-02 8.56e-02


Table 5: Ablation: Performance comparison
scaling with sequence length tested on NS- _ω_ .


While TempO has a moderate number of FLoating Point OPerations (FLOPs), landing between
the ViT and U-Net, this may be offsetted by the NFEs seen during the ODE integration where
TempO takes only 560 evaluations to meet the same tolerances. Beyond these empirical measures,
TempO further benefits from its shared spatial Fourier layers. By folding the channel dimension and
truncating higher modes, the spectral convolution scales as _O_ ( _N_ [2] log _N_ ), in contrast to the naive
_O_ ( _N_ [3] log _N_ ) cost of a full 3D FFT. Also for reference, a ViT layer can scale as _O_ ( _N_ [4] ) in 2D 3.2,
higher than the quasi-quadratic cost of the FNO.


6 LIMITATIONS


Flow matching models struggle with extreme data sparsity which can distort the distributions being learned, whereas hybrid models or models with explicitly defined conservations can fall back
on injected physical knowledge. Additionally, similar to other generative models, adaptations, e.g.
architectural modifications, would be necessary to extend the method towards a foundational model
framework. Finally, while our stable and accurate 40-step forecasting represents the longer end time
horizons, it remains an open question on how to forecast for much longer timeframes. Critical applications in science and engineering would require further study both experimentally and theoretically
to establish statistically reliable forecasting.


7 CONCLUSIONS AND FURTHER WORK


In this work, we addressed the challenge of long-horizon PDE forecasting via our proposed method
TempO. TempO consistently outperformed state-of-the-art baselines across three benchmark PDE
datasets and achieves stable long-horizon 40 step forecasts with remarkable accuracy to the true
trajectories as well as superior spectral fidelity. The modified time-conditioned FNO is parameterefficient while improving the capture of both local and global spectral modes, resulting in improvements in both data- and compute- efficiency. Additionally, we establish that FNO can achieve an
upper bound on approximation error that sampler-based architectures cannot reach without significantly more parameters, Corollary 3.3. These results highlight the importance of architectures that
align with the continuous nature of PDE dynamics, enabling not only improved predictive accuracy
but also physically consistent, long-horizon trajectories.


Consequently, TempO poses significant opportunity for further work in this field. Under typical
real-world environments, PDE observations may come from irregularly sampled domains; since
our method already demonstrates state-of-the-art generations using a simple autoencoder (AE) and
the latent time-conditioned FNO which no longer relies on a regular grid as is a limitation of the
original FNO (Li et al., 2021), one extension of our work is to then extend our method to real-world
settings to forecast PDE over irregular domains and irregularly sampled domains. In addition, a
detailed sensitivity analysis quantifying how TempO’s effective Lipschitz constants and integration
error evolve over long horizons would complement our empirical findings.


REFERENCES


Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen,
Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor,
Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the
Language of Time Series, November 2024.


Giacomo Baldan, Qiang Liu, Alberto Guardone, and Nils Thuerey. Flow Matching Meets PDEs: A
Unified Framework for Physics-Constrained Generation, November 2025.


Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat,
Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver
Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A Space-Time Diffusion Model
for Video Generation. In _SIGGRAPH_ _Asia_ _2024_ _Conference_ _Papers_, pp. 1–11, Tokyo Japan,
December 2024. ACM. ISBN 979-8-4007-1131-2. doi: 10.1145/3680528.3687614.


10


Taco Cohen and Max Welling. Group equivariant convolutional networks. In _International Confer-_
_ence on Machine Learning_, pp. 2990–2999. PMLR, 2016.


Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow Matching in Latent Space, July 2023.


Aram Davtyan, Sepehr Sameni, and Paolo Favaro. Efficient Video Prediction via Sparsely Conditioned Flow Matching, August 2023.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan,
Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava
Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux,
Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret,
Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,
Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary,
Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab
AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco
Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra,
Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu,
Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala,
Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid
El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren
Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin,
Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi,
Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew
Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C¸ elebi, Patrick Alrassy, Pengchuan
Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan,
Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan
Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell,
Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng
Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer
Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman,
Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor
Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei
Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang
Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning
Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh,
Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria,
Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein,
Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel,
Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu
Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao
Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia
Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide
Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le,


11


Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily
Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni,
Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia
Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan,
Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj,
Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James
Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang,
Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy
Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang,
Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell,
Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa,
Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias
Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L.
Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike
Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari,
Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan
Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong,
Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent,
Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar,
Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy,
Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin
Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon,
Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha,
Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal,
Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj
Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo
Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook
Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar,
Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li,
Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu,
Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi,
Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen
Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen,
Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The Llama 3 Herd of Models, November 2024.


Xingzhuo Guo, Yu Zhang, Baixu Chen, Haoran Xu, Jianmin Wang, and Mingsheng Long. Dynamical Diffusion: Learning Temporal Dynamics with Diffusion Models, March 2025.


Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, and Arash Vahdat. DiffiT: Diffusion Vision
Transformers for Image Generation. In Aleˇs Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and G¨ul Varol (eds.), _Computer Vision_ _– ECCV_ _2024_, volume 15066,
pp. 37–55. Springer Nature Switzerland, Cham, 2025. ISBN 978-3-031-73241-6 978-3-03173242-3. doi: 10.1007/978-3-031-73242-3 ~~3~~ .


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models, December
2020.


Benjamin Holzschuh, Qiang Liu, Georg Kohl, and Nils Thuerey. PDE-Transformer: Efficient and
Versatile Transformers for Physics Simulations. https://arxiv.org/abs/2505.24717v1, May 2025.


Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. DiffusionPDE: Generative PDESolving Under Partial Observation, November 2024.


12


Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang,
Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal Flow Matching for Efficient Video Generative Modeling, March 2025.


Richard John, Lukas Herron, and Pratyush Tiwary. A survey of probabilistic generative frameworks
for molecular simulations, November 2024.


Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong
Leng, Kaitao Song, Siliang Tang, Zhizheng Wu, Tao Qin, Xiang-Yang Li, Wei Ye, Shikun Zhang,
Jiang Bian, Lei He, Jinyu Li, and Sheng Zhao. NaturalSpeech 3: Zero-Shot Speech Synthesis
with Factorized Codec and Diffusion Models, April 2024.


Gavin Kerrigan, Giosue Migliorini, and Padhraic Smyth. Functional Flow Matching, December
2023.


Marcel Kollovieh, Marten Lienen, David L¨udke, Leo Schwinn, and Stephan G¨unnemann. Flow
Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting. In _The_ _Thir-_
_teenth International Conference on Learning Representations_, October 2024.


Nikola Kovachki, Samuel Lanthaler, and Siddhartha Mishra. On Universal Approximation and
Error Bounds for Fourier Neural Operators. _Journal_ _of_ _Machine_ _Learning_ _Research_, 22(290):
1–76, 2021. ISSN 1533-7928.


Zijie Li, Anthony Zhou, and Amir Barati Farimani. Generative Latent Neural PDE Solver using
Flow Matching, March 2025.


Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential
Equations, May 2021.


Jae Hyun Lim, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal,
Arash Vahdat, and Anima Anandkumar. Score-based Diffusion Models in Function Space, January 2025a.


Soon Hoe Lim, Yijin Wang, Annan Yu, Emma Hart, Michael W. Mahoney, Xiaoye S. Li, and
N. Benjamin Erichson. Elucidating the Design Choice of Probability Paths in Flow Matching
for Forecasting, July 2025b.


Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching
for Generative Modeling, February 2023.


Phillip Lippe, Bastiaan S. Veeling, Paris Perdikaris, Richard E. Turner, and Johannes Brandstetter.
PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers, October 2023.


Andrew J. Majda and Andrea L. Bertozzi. _Vorticity and Incompressible Flow_ . Cambridge Texts in
Applied Mathematics. Cambridge University Press, Cambridge, 2001. ISBN 978-0-521-63057-3.
doi: 10.1017/CBO9780511613203.


Roberto Molinaro, Samuel Lanthaler, Bogdan Raoni´c, Tobias Rohner, Victor Armegioiu, Stephan
Simonis, Dana Grund, Yannick Ramic, Zhong Yi Wan, Fei Sha, Siddhartha Mishra, and Leonardo
Zepeda-N´u˜nez. Generative AI for fast and accurate statistical computation of fluids, February
2025.


Hao Phung, Quan Dao, and Anh Tran. Wavelet Diffusion Models are fast and scalable Image
Generators, March 2023.


Xihao Piao, Zheng Chen, Taichi Murayama, Yasuko Matsubara, and Yasushi Sakurai. Fredformer:
Frequency Debiased Transformer for Time Series Forecasting. In _Proceedings of the 30th ACM_
_SIGKDD Conference on Knowledge Discovery and Data Mining_, pp. 2400–2410, August 2024.
doi: 10.1145/3637528.3671928.


13


Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, and
Matthew Willson. Probabilistic weather forecasting with machine learning. _Nature_, 637(8044):
84–90, January 2025. ISSN 1476-4687. doi: 10.1038/s41586-024-08252-9.


M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential
equations. _Journal_ _of_ _Computational_ _Physics_, 378:686–707, February 2019. ISSN 0021-9991.
doi: 10.1016/j.jcp.2018.10.045.


Gleb Ryzhakov, Svetlana Pavlova, Egor Sevriugov, and Ivan Oseledets. Explicit Flow Matching:
On The Theory of Flow Matching Algorithms with Applications. In _ICOMP_ _2024_ . arXiv, July
2024. doi: 10.48550/arXiv.2402.03232.


Louis Serrano, Thomas X. Wang, Etienne Le Naour, Jean-No¨el Vittaut, and Patrick Gallinari.
AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields, October 2024.


Yaozhong Shi, Zachary E. Ross, Domniki Asimaki, and Kamyar Azizzadenesheli. Stochastic Process Learning via Operator Flow Matching, October 2025.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models, October
2022.


Yang Song and Stefano Ermon. Generative Modeling by Estimating Gradients of the Data Distribution, October 2020.


Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-Based Generative Modeling through Stochastic Differential Equations, February
2021.


Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk
Pfl¨uger, and Mathias Niepert. PDEBENCH: An Extensive Benchmark for Scientific Machine
Learning, October 2022.


Ella Tamir, Najwa Laabid, Markus Heinonen, Vikas Garg, and Arno Solin. Conditional Flow Matching for Time Series Modelling. In _ICML_ _2024_ _Workshop_ _on_ _Structured_ _Probabilistic_ _Inference_
_{\&} Generative Modeling_, July 2024.


Mingtian Tan, Mike A. Merrill, Vinayak Gupta, Tim Althoff, and Thomas Hartvigsen. Are Language
Models Actually Useful for Time Series Forecasting?, October 2024.


Tapas Tripura and Souvik Chakraborty. Wavelet neural operator: A neural operator for parametric
partial differential equations, May 2022.


Peihao Wang, Wenqing Zheng, Tianlong Chen, and Zhangyang Wang. Anti-Oversmoothing in Deep
Vision Transformers via the Fourier Domain Analysis: From Theory to Practice, March 2022.


Jiachen Yao, Abbas Mammadov, Julius Berner, Gavin Kerrigan, Jong Chul Ye, Kamyar Azizzadenesheli, and Anima Anandkumar. Guided Diffusion Sampling on Function Spaces with Applications to PDEs, May 2025.


Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun
Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing Efficient Video Production for All,
December 2024.


14


A PROOFS


_Proof of Theorem 3.1._ _Step 1:_ _Spectral truncation._ By assumption the Fourier coefficients of _G_ ( _u_ )
satisfy
�� _G_ �( _u_ )( _k_ )�� _≤_ _Cλ_ (1 + _|k|_ ) _−p,_ _∀u ∈U,_ _k_ _∈_ Z _[d]_ _._


If we keep only the modes _|k| ≤_ _K_ and set


A standard counting argument (comparing the lattice sum with a radial integral) shows this tail is
≲ _K_ _[−]_ [2] _[α]_, with
_α_ := _p −_ _s_ _[′]_ _−_ _[d]_ 2 _[>]_ [0] _[.]_

This is exactly the pseudo-spectral tail estimate also used in (Kovachki et al., 2021, Thm. 40). Hence
choosing
_K_ _≍_ _ε_ _[−]_ [1] _[/α]_


ensures _∥G −GK∥H_ _s_ _[′]_ _≤_ _ε/_ 2.


_Step 2:_ _Reduction to a finite-dimensional map._ The truncated operator _GK_ is determined by finitely
many Fourier coefficients _{G_ [�] ( _u_ )( _k_ ) _}|k|≤K_, with output dimension _m_ out _≍_ _K_ _[d]_ . To apply a neural
network, we also restrict the input to finitely many low modes. By compactness of _U_ _⊂_ _H_ _[s]_ and
continuity of the projection _PM_, there exists _M_ such that


_∥GK_ ( _u_ ) _−GK_ ( _PM_ _u_ ) _∥H_ _s_ _[′]_ _≤_ _ε/_ 6 _∀u ∈U._


This is the same finite-dimensional reduction used in the universal approximation argument of (Kovachki et al., 2021, Thm. 15). Thus it suffices to approximate the finite-dimensional continuous
map
_F_ : ( _u_ �( _k_ )) _|k|≤M_ _�−→_ ( _G_ [�] ( _u_ )( _k_ )) _|k|≤K,_

between compact subsets of Euclidean spaces.


_Step 3:_ _Approximation of the finite map._ Classical universal approximation theorems (and the constructive Ψ–FNO realization in (Kovachki et al., 2021, Def. 11, Thm. 15)) ensure that for any desired
accuracy _δ_ _>_ 0, one can build a neural network (or FNO block) approximating _F_ uniformly to error
_δ_ on each retained coefficient. To control the _H_ _[s][′]_ –norm it suffices to achieve coefficient accuracy


_ε_
_δ_ ≲ _K_ _[s][′]_ [+] _[d/]_ [2] _[.]_


This choice ensures _∥PKG_ ( _u_ ) _−_ _G_ [�] _θ_ ( _u_ ) _∥H_ _s_ _[′]_ _≤_ _ε/_ 3. Constructive approximation bounds then give a
parameter count
_P_ ≲ _K_ _[d]_ _·_ polylog(1 _/ε_ ) _,_


where the extra logarithmic factor reflects standard overheads in coefficient quantization and network approximation (Kovachki et al., 2021, Remark 22).


_Step 4:_ _Assemble errors and conclude._ Adding the contributions: - spectral truncation error _≤_ _ε/_ 2
(Step 1), - input-projection error _≤_ _ε/_ 6 (Step 2), - finite-map approximation error _≤_ _ε/_ 3 (Step 3),


15


_GK_ ( _u_ )( _x_ ) :=           

_|k|≤K_


then the error lives in the high modes:


_G_ �( _u_ )( _k_ ) _e_ _[ik][·][x]_ _,_


_∥G_ ( _u_ ) _−GK_ ( _u_ ) _∥_ [2] _H_ _[s][′]_ [=] - (1 + _|k|_ [2] ) _[s][′]_ [��] _G_ [�] ( _u_ )( _k_ )��2 _._

_|k|>K_


Using the decay bound gives


_∥G_ ( _u_ ) _−GK_ ( _u_ ) _∥_ [2] _H_ _[s][′]_ _[≤]_ _[C]_ _λ_ [2] - (1 + _|k|_ ) [2(] _[s][′][−][p]_ [)] _._

_|k|>K_


we obtain
sup _∥G_ ( _u_ ) _−Gθ_ ( _u_ ) _∥H_ _s_ _[′]_ _≤_ _ε._
_u∈U_

Substituting _K_ _≍_ _ε_ _[−]_ [1] _[/α]_ into the parameter bound gives


_P_ F _NO_ ( _ε_ ) ≲ _ε_ _[−][d/α]_ _,_


up to the mild logarithmic factors discussed above.


_Proof of Proposition 3.2._ _Step 1: Finite-dimensional subspace and sampling._ Consider the _K_ -mode
Fourier subspace


_VK_ := span _{e_ _[ik][·][x]_ : _|k| ≤_ _K} ⊂_ _L_ [2] (T _[d]_ ) _,_ dim _VK_ =: _DK_ _≍_ _K_ _[d]_ _._


Any sampler-based learner observes an input _u_ _∈_ _VK_ only through _n_ fixed points
( _u_ ( _x_ 1) _, . . ., u_ ( _xn_ )). This defines a linear map

_S_ : _VK_ _→_ C _[n]_ _,_ _S_ ( _u_ ) = ( _u_ ( _x_ 1) _, . . ., u_ ( _xn_ )) _._


_Step 2: Nyquist / injectivity argument._ To reconstruct all Fourier modes up to radius _K_, the sampling
map _S_ must be injective on _VK_ . In matrix terms, _S_ is represented by an _n × DK_ Vandermonde-like
matrix. To have full rank _DK_, we require


_n_ _≥_ _DK_ _≍_ _K_ _[d]_ _._


If _n_ _<_ _DK_, there exists a nonzero _u_ _∈_ _VK_ vanishing on all sample points, so the learner cannot
distinguish it from zero. This is the standard Nyquist/dimension-counting requirement: at least as
many samples as degrees of freedom.


_Step 3:_ _Parameter lower bound._ After sampling, the learner applies a parametric map _M_ : C _[n]_ _→_
C _[m]_ (e.g., a neural network) to produce either output samples or coefficients. To implement arbitrary
linear transformations on the _DK_ retained modes (e.g., arbitrary Fourier multipliers), the parametric
map must have at least _DK_ free parameters. For fully general dense linear maps (no structural
constraints), one needs
_P_ ≳ _DK_ [2] _[≍]_ _[K]_ [2] _[d][.]_


_Step_ _4:_ _Conversion_ _to_ _accuracy_ _ε._ From the FNO upper bound analysis, achieving accuracy _ε_
requires
_K_ _≍_ _ε_ _[−]_ [1] _[/α]_ _,_ _α_ = _p −_ _s_ _[′]_ _−_ _d/_ 2 _>_ 0 _._

Substituting this into the previous bounds gives the scaling


_n_ ≳ _ε_ _[−]_ _[d/α]_ _,_ _P_ sampler( _ε_ ) ≳ _ε_ _[−]_ _[βd/α]_ _,_


with _β_ = 1 for minimal mode-wise maps and _β_ = 2 for fully dense maps.


_Step_ _5:_ _Conclusion._ Hence any sampler-based architecture that must reconstruct all modes up to
radius _K_ requires asymptotically more parameters than an FNO whenever _β_ _>_ 1, justifying the
lower bound in the proposition.


B FLOW MATCHING BACKGROUND


**Flow matching** The core idea of flow matching is to learn a time-dependent velocity field, _vθ_ ( _z, t_ ),
which defines an ODE in the latent space:


_dz_ ( _t_ )

= _vθ_ ( _z_ ( _t_ ) _, t_ ) _,_ _z_ (0) _∼_ _π_ 0 _,_ (2)
_dt_


where _π_ 0 is a simple reference distribution (e.g., Gaussian). Integrating this ODE transports samples to the latent data distribution _π_ 1, such that _z_ (1) _∼_ _π_ 1 and _p_ 1( _z_ ) _≈_ _fϕ_ # _D_ data, where _fϕ_ # _µ_
denotes the pushforward measure of a distribution _µ_ under _fϕ_, i.e., ( _fϕ_ # _µ_ )( _A_ ) = _µ_ ( _fϕ_ _[−]_ [1][(] _[A]_ [))][ for]


16


measurable sets _A_ . The corresponding time-dependent probability density, _pt_ ( _z_ ), evolves according
to the continuity equation:
_∂pt_ ( _z_ ) + _∇z_ _·_              - _pt_ ( _z_ ) _vθ_ ( _z, t_ )� = 0 _._ (3)

_∂t_

In practice, the target velocity field _u_ ( _t, z_ ) and the full marginal density _pt_ ( _z_ ) are generally unknown
and intractable. Flow matching sidesteps this issue by directly supervising the model to match the
instantaneous vector field along interpolating paths between the reference _π_ 0 and the target _π_ 1,
allowing for deterministic, efficient sampling. Different choices of paths lead to different training
dynamics and inductive biases, as they implicitly define the target velocity field _u_ ( _t, z_ ) that the
model regresses against.


Integrating this ODE from _t_ = 0 to _t_ = 1 transports the reference distribution _π_ 0 to the latent data
distribution _π_ 1, so that _z_ (1) _∼_ _π_ 1 and _p_ 1( _z_ ) _≈_ _fϕ_ # _D_ data.


**Latent** **Flow** **Matching.** We now instantiate the general flow matching framework in the latent
space. Let _zτ_ = _fϕ_ ( _xτ_ ) for _τ_ = 1 _, . . ., m_, where _fϕ_ is a pretrained encoder mapping from the
data space to the lower-dimensional latent space. Our objective is to approximate the ground-truth
latent distribution _q_ ( _zτ_ _|_ _x_ 1 _, . . ., xτ_ _−_ 1) by a parametric distribution _p_ ( _zτ_ _|_ _zτ_ _−_ 1), which can later
be decoded back to the data space via _xτ_ = _gψ_ ( _zτ_ ) using a decoder _gψ_ .


The latent dynamics can be expressed by the ODE:

_z_ ˙ _t_ = _ut_ ( _zt_ ) _,_ (4)

where _ut_ denotes the (true) time-dependent velocity field. Learning these dynamics amounts to
approximating _ut_ with a neural parameterization. Following the flow matching framework, we
introduce a model velocity field _vθ_ : [0 _,_ 1] _×_ R _[Z]_ _→_ R _[Z]_ and consider the ODE
_ϕ_ ˙ _t_ ( _z_ ) = _vθ_ ( _ϕt_ ( _z_ ) _, t_ ) _,_ _ϕ_ 0( _z_ ) = _z,_ (5)
which defines a time-dependent diffeomorphism _ϕt_ pushing forward an initial reference distribution
_p_ 0 (often chosen as _N_ (0 _, I_ )) to a target distribution _p_ 1 _≈_ _q_ along the density path _pt_ :
_pt_ = ( _ϕt_ )# _p_ 0 _,_ (6)
where ( _·_ )# denotes the pushforward. In other words, the goal of flow matching is to learn a deterministic coupling between _p_ 0 and _q_ by training _vθ_ so that the solution satisfies _z_ 0 _∼_ _p_ 0 and _z_ 1 _∼_ _q_ .


Given a probability path _pt_ and its associated velocity field _ut_, flow matching reduces to a leastsquares regression problem:
_L_ FM( _θ_ ) = E _t∼U_ [0 _,_ 1] _, z∼pt_ _ω_ ( _t_ ) _∥vθ_ ( _z, t_ ) _−_ _ut_ ( _z_ ) _∥_ [2] 2 _[,]_ (7)


where _ω_ ( _t_ ) _>_ 0 is a weighting function, often taken as _ω_ ( _t_ ) = 1 (Lipman et al., 2022). This
formulation ensures that the learned velocity field aligns with the target field _ut_ at all times, thereby
generating the desired marginal probability path.


C FOURIER NEURAL OPERATOR BACKGROUND


An FNO is designed to learn a mapping between function spaces, rather than between finitedimensional vectors. Consider a function _u_ : R _[d]_ _→_ R _[c]_ representing data, for example in R _[X]_,
with samples _x ∈_ R _[X]_ . Then, an FNO parameterizes an operator as
_Gθ_ : _u �→_ _u,_ ˜ _u_ ˜ : _D_ _→_ R _[c]_ [out] _,_

that maps _u_ to an output function _u_ ˜ (e.g., a solution field of a PDE or a transformed spatial signal).


This mapping is implemented via iterative Fourier layers which perform spectral transformations of
the input:
_u_ ˆ( _k_ ) = _F_ [ _u_ ]( _k_ ) _,_ _u_ ˆ˜( _k_ ) = _Rθ_ ( _k_ ) _·_ ˆ _u_ ( _k_ ) _,_ (8)
followed by an inverse Fourier transform back to the spatial domain:

_u_ ˜( _x_ ) = _F_ _[−]_ [1] [ _u_ [ˆ] ˜]( _x_ ) _,_ (9)

with _Rθ_ ( _k_ ) being learnable Fourier-mode weights and _F_ denoting the Fourier transform. This
spectral representation allows the FNO to efficiently capture long-range dependencies and global
correlations in the data.


17


D AUTOENCODER DETAILS


Residual blocks throughout the architecture consist of two 3 _×_ 3 convolutions with ReLU activation and group normalization (8 groups) in between, with the input added back to the output. Attention blocks are implemented using PyTorch’s nn.MultiheadAttention, with embeddings
reshaped from [ _B, C, H, W_ ] to [ _B, HW, C_ ].


The autoencoder is initialised with a depth of _d_ = 2 resulting in a factor 2 _[d]_ = 4 compression for all
datasets.


E MODEL HYPERPARAMETERS


We initialised the probability paths with the following hyperparameters. RIVER was defined with
variance parameters _σ_ = 0 _._ 1 and _σ_ min = 10 _[−]_ [7] . SLP used _σ_ = 0 _._ 1 and _σ_ min = 0 _._ 01. We further
considered the VE-diff path with _σ_ min = 0 _._ 01 and _σ_ max = 0 _._ 1 and the VP-diff path initialized with
_β_ min = 0 _._ 1 and _β_ max = 20 _._ 0 per (Lim et al., 2025b).


We provide details for the vector field regressors’ width and depth hyperparameters as per Table 6.


**Model** **Parameter** **Value**


Table 6: Descriptions of hyperparameters across TempO, U-Net, and ViT architectures.


F TRAINING AND INFERENCE SETUP FOR BASELINE MODELS


We evaluate three classes of baselines: Fourier Neural Operators (FNO) (Li et al., 2021), Wavelet
Neural Operators (WNO) (Tripura & Chakraborty, 2022), and a standard U-Net backbone.


**2D** **Autoregressive** **Models** **(FNO-2D,** **WNO-2D,** **U-Net).** The 2D variants of FNO, WNO, and
U-Net operate purely on spatial fields _ω_ ( _·, t_ ) _∈_ R [64] _[×]_ [64] and treat time autoregressively. Each model
is provided a dense block of _n_ input frames. These frames are concatenated and mapped to the
next-time-step prediction _ω_ ˆ( _·, t_ + 1). During training, models minimize an _ℓ_ 2 regression loss on
the next-step vorticity. During inference, the prediction is appended to the input sequence, and the
model is rolled forward autoregressively for the desired horizon, replacing the oldest frame at each
step. This _2D+RNN_ structure allows propagation to arbitrarily long forecast windows using a fixed
temporal stride.


**3D Convolutional Models (FNO-3D, WNO-3D).** The 3D variants of FNO and WNO treat time
as an additional convolutional dimension and directly process space–time blocks ( _x, y, t_ ) as 3D
volumes. Following Li et al. (2021), the model receives the dense history of the first _n_ timesteps and
performs a 3D convolutional operator mapping


R _[n][×][x][×][y]_ _−→_ R _[T]_ [pred] _[×][x][×][y]_ _,_


18


TempO


U-Net


ViT


_n_ modes 20
Hidden channels 64
Projection channels 64
Depth 4


Hidden channels 64
Attention resolutions (1, 2, 2)
Channel multiplier (1, 2, 4)
Depth 3


Hidden channels 256
Depth 4
Mid-depth 5
Output normalization LayerNorm


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


producing the entire future trajectory segment _T_ at once. Unlike the 2D autoregressive setting, the
3D operator does _not_ iterate forward in time: it learns a direct operator from the initial block to the
entire forecast window. This makes the method better conditioned and more expressive at the cost
of requiring a fixed temporal window during training (Li et al., 2021).


Both 2D and 3D WNOs follow the same temporal training structure as their FNO counterparts. The
only architectural change is the replacement of Fourier transforms with multi-resolution wavelet
transforms, but the data flow (input–output tensor shapes, roll-out strategy, and supervision) is identical.


**Training** **Details.** Models were trained using the Adam optimiser with weight decay of 0.0001.
FNO and WNO learning rate of 0.001 except the U-Net, which was trained with a learning rate of
0.0001.


**Model** **Dim** **Layers** **Width** **Modes / Wavelet** **Level** **Input Ch.**


FNO-3D 3D 4 64 (20 _,_ 20 _,_ 16) (Fourier)    - 1
FNO-2D 2D 4 64 (20 _,_ 20) (Fourier)    - _T_ in
UNet-2D 2D 4 (down/up) 128 base    -    - _T_ in
WNO-3D 3D 4 40 db6 (wavelet) 2 ( _T_ in + 3)
WNO-2D 2D 4 64 db6 (wavelet) 2 ( _T_ in + 2)


Table 7: Summary of hyperparameters for Fourier and Wavelet neural operators and the UNet baseline. Channels refers to hidden width. Modes applies only to FNO; wavelet type/level applies only
to WNO.


**Inference** **Protocol.** For comparability with prior operator-learning works, we evaluate autoregressive models by providing the first _n_ = 10 frames as context and generating the next 40 timesteps
using a single-step temporal stride. For 3D models, the first _n_ frames are mapped directly to a 40step output block without iterative rollout. This setup follows the conventions established in (Li
et al., 2021; Tripura & Chakraborty, 2022).


G DATASET DETAILS


Table 8: Dataset sizes and trajectory lengths used in evaluation.


Dataset # Trajectories Timeseries Length


SWE 1000 100
RD-2D 1000 100
NS- _ω_ 5000 50


_Shallow water equation (SWE)_


The SWEs are derived from the compressible Navier–Stokes equations and model free-surface flow
problems in 2D. The system of hyperbolic PDEs is given by:


_∂th_ + _∂x_ ( _hu_ ) + _∂y_ ( _hv_ ) = 0 _,_ (10)


            _∂t_ ( _hu_ ) + _∂x_ _u_ [2] _h_ + [1] + _∂y_ ( _uvh_ ) = _−grh ∂xb,_ (11)

2 _[g][r][h]_ [2][�]


            _∂t_ ( _hv_ ) + _∂y_ _v_ [2] _h_ + [1] + _∂x_ ( _uvh_ ) = _−grh ∂yb,_ (12)

2 _[g][r][h]_ [2][�]


where _u, v_ are the horizontal and vertical velocities, _h_ is the water height, _b_ represents spatially
varying bathymetry, and _gr_ is gravitational acceleration. The quantities _hu_ and _hv_ correspond to
directional momentum components.


The dataset ( (Takamoto et al., 2022)) simulates a 2D radial dam break scenario on a square domain
Ω= [ _−_ 2 _._ 5 _,_ 2 _._ 5] [2] . The initial water height is a circular bump in the center of the domain:


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


_h_ ( _t_ = 0 _, x, y_ ) = �2 _._ 0 _,_ if _r_ _< r_ 0 _,_ _r_ = ~~�~~ _x_ [2] + _y_ [2] _,_ _r_ 0 _∼U_ (0 _._ 3 _,_ 0 _._ 7) _._
1 _._ 0 _,_ if _r_ _≥_ _r_ 0 _,_


_2D reaction diffusion (RD-2D)_


The RD-2D dataset models two non-linearly coupled variables: the activator _u_ = _u_ ( _t, x, y_ ) and the
inhibitor _v_ = _v_ ( _t, x, y_ ). The system of PDEs is:


_∂tu_ = _Du ∂xxu_ + _Du ∂yyu_ + _Ru_ ( _u, v_ ) _,_ (13)
_∂tv_ = _Dv ∂xxv_ + _Dv ∂yyv_ + _Rv_ ( _u, v_ ) _,_ (14)


where _Du_ and _Dv_ are diffusion coefficients, and _Ru_ ( _u, v_ ), _Rv_ ( _u, v_ ) are the reaction functions.
Specifically, the FitzHugh–Nagumo model defines the reactions as:


_Ru_ ( _u, v_ ) = _u −_ _u_ [3] _−_ _k −_ _v,_ (15)
_Rv_ ( _u, v_ ) = _u −_ _v,_ (16)


with _k_ = 5 _×_ 10 _[−]_ [3], _Du_ = 1 _×_ 10 _[−]_ [3], and _Dv_ = 5 _×_ 10 _[−]_ [3] .


The dataset ( (Takamoto et al., 2022)) uses a simulation domain _x, y_ _∈_ ( _−_ 1 _,_ 1) and _t_ _∈_ (0 _,_ 5] with
initial condition set as standard normal random noise: _u_ (0 _, x, y_ ) _∼N_ (0 _,_ 1 _._ 0).


_2D incompressible Navier-Stokes vorticity (NS-ω)_


The NS- _ω_ ( (Li et al., 2021)) models 2D incompressible fluid flow on the unit torus. The system of
equations is:


_∂tw_ ( _x, t_ ) + _u_ ( _x, t_ ) _· ∇w_ ( _x, t_ ) = _ν_ ∆ _w_ ( _x, t_ ) + _f_ ( _x_ ) _,_ _x ∈_ (0 _,_ 1) [2] _,_ _t ∈_ (0 _, T_ ] _,_ (17)
_∇· u_ ( _x, t_ ) = 0 _,_ (18)

_w_ ( _x,_ 0) = _w_ 0( _x_ ) _,_ (19)


where _w_ ( _x, t_ ) is the vorticity, _u_ ( _x, t_ ) is the velocity field, _ν_ is viscosity, and _f_ ( _x_ ) is a fixed forcing
term:


            -            _f_ ( _x_ ) = 0 _._ 1 sin(2 _π_ ( _x_ 1 + _x_ 2)) + cos(2 _π_ ( _x_ 1 + _x_ 2)) _._


The initial condition is sampled from a Gaussian measure:


_w_ 0 _∼_ _µ,_ _µ_ = _N_ �0 _,_        - _−_ ∆+ 49 _I_        - _−_ 2 _._ 5 73 _/_ 2 [�] _,_


with periodic boundary conditions.


H SPECTRAL ANALYSIS OF GROUND TRUTH NS- _ω_


Fig. 4 shows how the quality of spectral truncations of the true Navier–Stokes vorticity field depends
on the cutoff wavenumber _k_ cut. Given the full Fourier spectrum _ω_ ˆ( _kx, ky_ ), we apply a mask that
retains only modes with _|kx|_ + _|ky| ≤_ _k_ cut, reconstruct the signal by inverse FFT, and compute three
quantities as functions of _k_ cut:


1. Reconstruction MSE: the mean squared error between the original and truncated fields in
physical space.

2. Spectral MSE: the mean squared error in Fourier space, quantifying lost spectral content.


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


Figure 4: **Spectral Analysis of True Vorticity** : Reconstruction MSE, spectral MSE, and cumulative
enstrophy fraction of true Navier–Stokes vorticity data as functions of cutoff wavenumber _k_ cut.


3. Cumulative energy fraction: the fraction of total energy [�] _|ω_ ˆ _|_ [2] retained by the truncated
spectrum.


As _k_ cut increases, both reconstruction and spectral errors decrease, while the retained energy approaches unity.


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


I EXTENDED RESULTS FOR NAVIER–STOKES VORTICITY


Figure 5: **Navier–Stokes Vorticity (Original).** Ground-truth timeseries across 40 timesteps.


Figure 6: **Navier–Stokes Vorticity (U-Net).** Forecasted timeseries across 40 timesteps.


Figure 7: **Navier–Stokes Vorticity (ViT).** Forecasted timeseries across 40 timesteps.


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


Figure 8: **Navier–Stokes Vorticity (TempO).** Forecasted timeseries across 40 timesteps.


J EXTENDED RESULTS FOR SHALLOW WATER EQUATION


Regressor Path MSE _↓_ SpectralMSE _↓_ RFNE _↓_ PSNR _↑_ Pearson _↑_ SSIM _↑_


VP-diff 1.37e-02 8.26e-03 1.10e-01 23.4 0.546 0.627
U-Net RIVER 1.61e-02 1.00e-02 1.20e-01 22.7 0.437 0.610
Affine-OT 1.68e-02 1.01e-02 1.22e-01 22.5 0.435 0.593


Table 9: Comparison of TempO, U-Net, and ViT models under different probability paths for the
SWE. The best value for each metric is highlighted in bold.


Figure 9: **SWE (Original).** Ground-truth timeseries across 40 timesteps.


23


TempO


ViT


Affine-OT **6.64e-05** **5.65e-05** **7.64e-03** **46.5** **0.998** **0.997**
RIVER 4.04e-04 2.33e-04 1.89e-02 38.7 0.989 0.976
VE-diff 9.37e-04 8.22e-04 2.89e-02 35.2 0.994 0.977
VP-diff 4.41e-03 2.51e-03 4.31e-02 28.3 0.872 0.857


Affine-OT 9.59e-05 7.93e-05 9.06e-03 44.9 0.997 0.995
VP-diff 1.30e-04 8.81e-05 1.05e-02 43.6 0.996 0.993
RIVER 2.99e-04 1.67e-04 1.63e-02 40.0 0.992 0.981
SLP [1] 6.60e-04 - 1.28e-01 36.1 - 0.93
VE-diff 1.28e-03 1.01e-03 3.38e-02 33.7 0.985 0.960


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


Figure 10: **SWE (U-Net).** Forecasted timeseries across 40 timesteps.


Figure 11: **SWE (ViT).** Forecasted timeseries across 40 timesteps.


Figure 12: **SWE (TempO).** Forecasted timeseries across 40 timesteps.


K EXTENDED RESULTS FOR 2D REACTION DIFFUSION


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


Regressor Path MSE _↓_ SpectralMSE _↓_ RFNE _↓_ PSNR _↑_ Pearson _↑_ SSIM _↑_


Table 10: Comparison of TempO, U-Net, and ViT models under different probability paths for the
RD-2D. The best value for each metric is highlighted in bold.


Figure 13: **Reaction** **Diffusion** **(Original).** Ground-truth end sample, from initial conditions of
randomly sampled noise.


Figure 14: **Reaction Diffusion (U-Net).** Forecasted end sample, from initial conditions of randomly
sampled noise.


Figure 15: **Reaction Diffusion (ViT).** Forecasted end sample, from initial conditions of randomly
sampled noise.


Figure 16: **Reaction** **Diffusion** **(TempO).** Forecasted end sample, from initial conditions of randomly sampled noise.


25


TempO


ViT


U-Net


Affine-OT **2.76e-05** **2.18e-05** **3.29e-02** **65.7** **1.000** **0.999**
RIVER 9.36e-04 5.47e-04 2.08e-01 50.4 0.975 0.978
VE-diff 1.58e-03 1.38e-03 2.70e-01 48.2 0.990 0.977
VP-diff 1.24e-02 1.01e-02 4.95e-01 39.2 0.714 0.862


Affine-OT 6.30e-04 4.40e-04 1.67e-01 52.2 0.987 0.986
SLP [2] 3.56e-04 - 1.16e-01 34.3 - 0.90
RIVER 1.00e-03 5.89e-04 2.16e-01 50.1 0.973 0.977
VE-diff 3.54e-03 2.23e-03 4.06e-01 44.7 0.915 0.946


Affine-OT 3.09e-05 2.45e-05 3.57e-02 65.2 0.999 **0.999**
RIVER 1.02e-03 5.49e-04 2.17e-01 50.1 0.972 0.976
VE-diff 9.03e-03 6.07e-03 6.42e-01 40.6 0.820 0.860
VP-diff 2.09e-02 1.66e-02 6.81e-01 37.0 0.574 0.792


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


L EXTENDED ABLATION RESULTS


Table 11: Ablation over different training sequence lengths on the NS- _ω_ dataset. TempO and the top
performing alternative are trained while varying sequence lengths and evaluated on 10 timesteps to
predict the next step.


Method Seq. Len. MSE DensityMSE SpectralMSE RFNE PSNR Pearson SSIM NFE


Table 12: Ablation of the TempO model on the NS- _ω_ dataset by varying the number of modes.
Models are trained with different numbers of Fourier modes and evaluated on 10 timesteps to predict
the next step.


Modes MSE DensityMSE SpectralMSE RFNE PSNR Pearson SSIM NFE


1 1.409e-01 1.075e-04 8.566e-02 3.947e-01 21.204 0.921 0.588 5798
2 6.103e-02 8.928e-05 4.096e-02 2.596e-01 24.837 0.966 0.765 1688
4 5.789e-02 8.361e-05 3.978e-02 2.538e-01 25.066 0.968 0.776 1058
8 5.528e-02 8.498e-05 3.788e-02 2.481e-01 25.267 0.969 0.788 800
16 5.471e-02 8.757e-05 3.742e-02 2.467e-01 25.312 0.970 0.787 884


M ABLATION ON FOURIER-MODE TRUNCATION


**Setup** To assess the sensitivity of TempO to the number of retained Fourier modes, we perform an
ablation in which the spectral truncation level of the underlying FNO blocks is varied while keeping
all other architectural and training settings fixed. Specifically, we evaluate truncation levels of


_m ∈{_ 16 _,_ 8 _,_ 4 _,_ 2 _,_ 1 _},_


where _m_ denotes the number of Fourier modes kept per spatial dimension. The case _m_ = 1 represents the most extreme truncation and serves as a lower-bound sanity check for TempO under highly
restricted spectral capacity.


**Method** All models are trained with identical optimisation hyperparameters and dataset splits on
the NS- _ω_ dataset. For each truncation level, we measure ..., following the evaluation protocol used
in the main experiments.


**Results** Table 13 reports the aggregated metrics. We observe that performance remains stable for
moderate truncations ( _m_ = 8 and _m_ = 16), with only mild degradation for _m_ = 4. As expected,
substantial deterioration appears only when the spectral capacity is collapsed to _m_ = 2 or _m_ = 1,
reflecting the loss of essential mid-frequency components.


**Discussion** The study shows that TempO’s performance is largely stable for moderate truncations
(4–16 modes), and interestingly at 4 modes (though k=8 captures 99% of the energy, k=4 converges
upon almost 97% already) the rollout is marginally more stable than 8 modes. The extreme truncation to 2 modes and 1 mode produces a noticeable drop in accuracy, as expected, highlighting
the importance of retaining a sufficient number of spectral modes to capture the essential dynamics.


26


TempO


ViT
(Affine-OT)


3 4.924e-02 7.685e-05 3.531e-02 2.328e-01 25.769 0.973 0.803 74
6 4.753e-02 1.133e-04 3.394e-02 2.276e-01 25.923 0.974 0.800 608
11 5.036e-02 1.055e-04 3.620e-02 2.352e-01 25.672 0.972 0.800 842
16 5.607e-02 1.282e-04 3.821e-02 2.497e-01 25.205 0.969 0.786 938
26 6.255e-02 7.487e-05 3.726e-02 2.541e-01 24.730 0.968 0.765 1070


3 6.748e-02 1.414e-04 4.652e-02 2.678e-01 24.401 0.963 0.766 116
6 5.434e-02 1.239e-04 3.727e-02 2.416e-01 25.341 0.970 0.783 1766
11 6.014e-02 1.376e-04 4.067e-02 2.546e-01 24.901 0.967 0.777 1712
16 6.701e-02 1.093e-04 4.428e-02 2.680e-01 24.431 0.963 0.764 1622
26 7.682e-02 8.104e-05 4.468e-02 2.778e-01 23.838 0.960 0.741 1100


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


Table 13: Ablation on Fourier mode truncation in TempO.


Modes MSE SpectralMSE PSNR Pearson MSE/time


1 5.34e-02 3.74e-2 25.416 0.970 3.950e-01
2 3.16e-02 3.79e-2 27.695 0.982 1.668e-01
4 2.72e-02 4.00e-2 28.346 0.984 2.715e-02
8 2.91e-02 4.10e-2 28.056 0.983 3.244e-02
16 2.80e-02 8.57e-2 28.215 0.984 3.084e-02


Figure 17: Ablation of varying Fourier mode truncation to evaluate sensitivity to the truncation level.


Overall, this confirms that TempO is robust to Fourier-mode selection within a reasonable range,
though the best performing model is at 16 modes.


Crucially, truncation from 16 to 8 modes does not materially affect accuracy, consistent with the
observation that the datasets considered have weak energy content in the highest spectral bands
(Appendix G). Only extremely aggressive truncation (e.g. _m ≤_ 2) produces meaningful degradation,
as such settings remove both high- and mid-frequency modes required to represent the system’s
dynamics. Any truncation for computational requirements is also mitigated by the fact that we
operate in latent space, which caps the maximum number of modes according to the resolution of
the latent space.


N EXTENDED RESULTS FO FNO-2D ON NAVIER STOKES VORTICITY


O USE OF LARGE LANGUAGE MODELS (LLMS)


We acknowledge the use of ChatGPT to make suggestions on how to polish the text, correct grammar, and ensure clarity in writing. No results, code, or data were created or altered by the model.


27


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


Figure 18: **Navier–Stokes Vorticity (Ground Truth).** Forecasted timeseries across 40 timesteps.


Figure 19: **Navier–Stokes Vorticity (FNO-2D).** Forecasted timeseries across 40 timesteps.


Figure 20: **Navier–Stokes Vorticity (TempO).** Forecasted timeseries across 40 timesteps.


28