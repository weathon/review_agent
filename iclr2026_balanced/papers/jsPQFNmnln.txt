# NEURAL POSTERIOR ESTIMATION WITH LATENT BASIS EXPANSIONS


**Declan McNamara, Yicun Duan, Jeffrey Regier**
Department of Statistics
University of Michigan
Ann Arbor, MI 48109, USA
{declan, pduan, regier}@umich.edu


ABSTRACT


Neural posterior estimation (NPE) is a likelihood-free amortized variational inference method that approximates projections of the posterior distribution. To
date, NPE variational families have been either simple and interpretable (such as
the Gaussian family) or highly flexible but black-box and potentially difficult to
optimize (such as normalizing flows). In this work, we parameterize variational
families via basis expansions of the latent variables. The log density of our variational distribution is a linear combination of latent basis functions (LBFs), which
may be fixed a priori or adapted to the problem class of interest. Our training
and inference procedures are computationally efficient even for problems with
high-dimensional latent spaces, provided only a low-dimensional projection of the
posterior is of interest, owing to NPE’s automatic marginalization capabilities. In
numerous inference problems, the proposed variational family outperforms existing variational families used with NPE, including mixtures of Gaussians (mixture
density networks) and normalizing flows, and also outperforms an existing basis
expansion method for variational inference.


1 INTRODUCTION


Neural Posterior Estimation (NPE) is an increasingly popular approach to Bayesian inference (Papamakarios & Murray, 2016; Cranmer et al., 2020; Dax et al., 2021; Ward et al., 2022). In NPE, a
neural network is trained exclusively on synthetic data—latent variables drawn from the prior paired
with observations—to learn the inverse mapping from observations to latent variables. Once trained,
this network can produce posterior approximations for real data in a single forward pass. In contrast
to traditional (ELBO-based) variational inference, NPE does not require likelihood evaluations.
Furthermore, when the generative model contains both parameters of interest and nuisance variables,
NPE can automatically marginalize over the nuisance parameters during training: by simulating
complete data and then discarding the nuisance variables to create training pairs, the method infers
posterior projections for the parameters of interest (Ambrogioni et al., 2019).


Despite these advantages, NPE shares with traditional ELBO-based variational inference a fundamental trade-off between the flexibility of the variational family and the tractability of optimization.
Simple variational families, such as Gaussians, enable stable optimization but often lack the expressiveness needed for complex posterior geometries. More flexible alternatives, such as Gaussian
mixture models and normalizing flows, offer greater flexibility but can create difficult optimization
landscapes with shallow local optima. Recent theoretical work (McNamara et al., 2024a) has established conditions for global convergence in NPE that hold for simple Gaussian variational families,
but these results do not extend to the more expressive families commonly deployed in practice.


In this work, we propose a variational family specialized for NPE that leverages NPE’s automatic
marginalization and likelihood-free nature. Unlike standard variational inference, where nuisance
variables must be modeled in the variational distribution, NPE applications typically require posteriors
over just a few scientifically relevant parameters: high-dimensional posterior samples do not directly
aid in interpretation but must instead be post-processed to estimate low-dimensional interpretable
quantities. Because NPE does not require likelihood evaluations, this post-processing can often be


1


incorporated directly into the Bayesian model, resulting in a low-dimensional latent space of interest.
In these low-dimensional settings, even numerical integration is computationally feasible, freeing us
from the usual requirement of closed-form normalization.


We leverage this freedom by parameterizing the log density of variational distributions through
latent basis expansions: a neural network processes observations to produce coefficients for linear
combinations of basis functions over the latent space. This approach yields distributions in the exponential family—among the most flexible classes available (Pacchiardi & Dutta, 2022; Khemakhem
et al., 2020; Sriperumbudur et al., 2017)—while maintaining favorable optimization properties. The
resulting method, which we refer to as Latent Basis Function NPE (LBF-NPE), optimizes over the
class of all exponential families of a fixed dimension _K_ by adaptively fitting the basis functions,
denoted _sψ_ ( _z_ ) _∈_ R _[K]_ . Simultaneously, amortization is performed by fitting a separate network
_fϕ_ ( _x_ ) _∈_ R _[K]_ that maps observations _x_ to coefficients of these basis functions (Section 3).


In Section 4, we introduce and analyze several variants of LBF-NPE. For a variant with fixed basis
functions, such as B-splines or wavelets, optimization is convex despite the log-normalizer, providing
stable training that has proven elusive for more complex variational families, and ensuring stable
convergence to global optima under the conditions presented by McNamara et al. (2024a). Alternatively, both coefficients _fϕ_ ( _x_ ) and basis functions _sψ_ ( _z_ ) can be fitted jointly through alternating
optimization that exploits marginal convexity in each component. We employ stereographic projection reparameterization to address identifiability issues in this adaptive setting, constraining outputs
to the unit hypersphere and stabilizing training.


We demonstrate superior performance of LBF-NPE across diverse inference problems, from synthetic
benchmarks to real scientific applications (Section 6). LBF-NPE consistently converges to global
optima on multimodal problems where mixture density networks converge to shallow local minima.
LBF-NPE with just 20 basis functions achieves order-of-magnitude improvements in KL divergence
over both MDNs and normalizing flows on complex 2D posteriors. The method successfully captures
multimodal posteriors in astronomical object detection and substantially outperforms MDN baselines
on cosmological redshift estimation using the LSST DESC DC2 survey dataset.


2 BACKGROUND


2.1 ELBO-BASED VARIATIONAL INFERENCE


In variational inference (VI), numerical optimization is used to select an approximation _q_ ( _z_ ) of
the posterior distribution of some model _p_ ( _z, x_ ) on observables _x_ and latent variables _z_ . The most
common variational objective, the evidence lower bound (ELBO), targets minimization of the reverse
KL divergence (Blei et al., 2017; Zhang et al., 2019; Kingma & Welling, 2019) by constructing the
variational quantity


       - _p_ ( _z, x_ )
ELBO ( _η_ ) := E _q_ ( _z_ ; _η_ ) log
_q_ ( _z_ ; _η_ )


_≤_ log _p_ ( _x_ ) (1)


and performing maximization in _η_ for a fixed choice of _x_ . Optimizing the ELBO remains a longstanding challenge in VI. Firstly, even simple variational families, such as the Gaussian, can exhibit
problematic optimization landscapes without careful parameterization. Targeting the ELBO, even in
the non-amortized setting, is generally a nonconvex problem (Liu et al., 2023; Domke, 2020; Domke
et al., 2023). Secondly, and most significantly in our setting, ELBO-based variational inference does
not provide a way to marginalize over nuisance latent variables _ξ_ . For a model _p_ ( _z, ξ, x_ ) on latent
variables _{z, ξ}_ and observed variables _x_, inference on _z_ conditional on _x_ is a common target of
interest. Yet ELBO-based methods typically cannot compute the quantity _p_ ( _z, x_ ) = - _p_ ( _z, ξ, x_ ) _dξ_
required to target the objective function (1), and instead typically require explicitly modeling the
nuisance variables.


2.2 NEURAL POSTERIOR ESTIMATION


In amortized variational inference, the shared parameters of a deep neural network define variational
approximations for arbitrary observations _x_ (Ganguly et al., 2024; McNamara et al., 2024a). Precisely,
a variational approximation for latent variable _z_ is given by _q_ ( _z_ ; _η_ ) with parameters _η_ . Rather than
fitting these parameters separately for each _x_, the amortized approach sets _η_ = _fϕ_ ( _x_ ) for a neural


2


network _fϕ_ with parameters _ϕ_ . This inference network, once fit, yields the variational posterior
_q_ ( _z_ ; _fϕ_ ( _x_ )) for arbitrary _x_ by a single forward pass (Kingma & Welling, 2019; Ambrogioni et al.,
2019). Neural posterior estimation (NPE) (Papamakarios & Murray, 2016) targets an expectation of
the forward KL divergence for amortized VI:


_L_ ˜NPE( _ϕ_ ) = E _p_ ( _x_ )KL ( _p_ ( _z_ _| x_ ) _|| q_ ( _z_ _| x_ )) _._ (2)


Here, the integral over _p_ ( _x_ ), the marginal of the model _p_ ( _z, ξ, x_ ), indicates that the NPE objective
averages over all possible draws from the generative process instead of averaging over observations _x_
from some finite training set. The objective is equivalent (up to a constant) to


_L_ NPE( _ϕ_ ) = _−_ E _p_ ( _z,x_ ) log _q_ ( _z_ ; _fϕ_ ( _x_ )) _,_ (3)


where _qϕ_ ( _z_ ; _η_ ) _≡_ _q_ ( _z_ ; _fϕ_ ( _x_ )) for any _x_ . Equation (3) admits unbiased estimation of its gradient with
stochastic draws ( _z, x_ ) _∼_ _p_ ( _z, ξ, x_ ) from the joint model, readily obtained by ancestral sampling, even
in the presence of nuisance latent variables _ξ_ . For example, in a hierarchical model we can sample
( _z, x_ ) _∼_ _p_ ( _z, x_ ) by simulating an entire sequence _p_ ( _z_ ) _p_ ( _ξ_ 1 _| z_ ) _p_ ( _ξ_ 2 _| ξ_ 1) _· · · p_ ( _ξL_ _| ξL−_ 1) _p_ ( _x | ξL_ )
and discarding variables that are not the target for inference. The expected forward KL objective
has been independently derived and analyzed in several related works (Bornschein & Bengio, 2015;
Ambrogioni et al., 2019).


3 LBF-NPE: BASIS EXPANSIONS FOR AMORTIZED LOG DENSITY
ESTIMATION


We propose an amortized method to fit complex multimodal variational distributions, called Latent
Basis Function NPE (LBF-NPE). For latent variables _z_ in latent space _I_, the method fits a basis
function network _sψ_ : _I_ _�→_ R _[K]_, which evaluates _K_ basis functions [ _s_ [(1)] _ψ_ _[, . . ., s]_ _ψ_ [(] _[K]_ [)] ] at any point _z_ .
The inference network _fϕ_ : _x_ _�→_ _η_ _∈_ R _[K]_ maps observations to coefficients of the basis functions.
The number of basis functions _K_ is fixed ahead of time, and larger _K_ may be used to increase
expressivity (see Appendix E). The variational parameters to be fit are the neural network weights _ϕ_
and _ψ_ for the networks _fϕ_ and _sψ_ . Below, we first construct an exponential family defined by the
basis functions (Section 3.1) and then give the fitting routine for _fϕ_ and _sψ_ (Section 3.2).


3.1 THE VARIATIONAL FAMILY


Fix _x ∈X_ . Let _I_ _⊆_ R _[d]_ denote the latent space and let _{s_ [(] _ψ_ _[i]_ [)][(] _[z]_ [)] _[}]_ _i_ _[K]_ =1 [be a collection of basis functions]
defined by parameters _ψ_ . Selecting more basis functions (larger _K_ ) leads to a more expressive
variational distribution, but also a higher-dimensional optimization problem. For a value _z_ _∈_ _I_, let
_sψ_ ( _z_ ) = [ _s_ [(1)] _ψ_ _[, . . ., s]_ _ψ_ [(] _[K]_ [)] ] _[⊤]_ _∈_ R _[K]_ . Our variational family is parameterized by coefficients _η_ _∈_ R _[K]_,
and has the density function

_q_ ( _z_ ; _η_ ) _∝_ _h_ ( _z_ ) exp                - _η_ _[⊤]_ _sψ_ ( _z_ )� _._ (4)


We aim to select _η_ and _ψ_ such that _q_ ( _z_ ; _η_ ) _≈_ _p_ ( _z_ _| x_ ). The log density is


log _q_ ( _z_ ; _η_ ) = log _h_ ( _z_ ) + _η_ _[⊤]_ _sψ_ ( _z_ ) _−_ _C,_ (5)


where _C_ is the log of the normalizing constant. This variational family (Equation 4) is an exponential
family (Wainwright & Jordan, 2008; Srivastava et al., 2014). The vector of basis functions _sψ_ ( _z_ ) is
the sufficient statistic, _η_ is the natural parameter vector, and _h_ ( _z_ ) is any finite base measure on the
latent space _I_ (i.e., - _I_ _[h]_ [(] _[z]_ [)] _[dz]_ [ is finite).]


We represent the _K_ -dimensional quantity _sψ_ ( _z_ ) as the output of a deep neural network with input
_z_ . Because the number and form of the basis functions _sψ_ ( _z_ ) are arbitrary, the expressivity of this
variational family is far greater than that of “classical” exponential families, such as the Gaussian
family. As _K_ _→∞_, the set of _all_ exponential family distributions is arbitrarily rich: any distribution
can be represented as an infinite-dimensional exponential family distribution (Khemakhem et al.,
2020; Sriperumbudur et al., 2017).


3


3.2 THE AMORTIZED VARIATIONAL OBJECTIVE & GRADIENT ESTIMATOR


The formulation of Section 3.1 is _non-amortized_ : it requires selecting a single _η_ for _q_ ( _z_ ; _η_ ) to
approximate the posterior for a single _x_ . In NPE, we consider the amortized problem, where we
define the posterior for an arbitrary _x_ . We set _η_ = _fϕ_ ( _x_ ), and thus require fitting two separate
networks, _fϕ_ and _sψ_ . We fit the variational parameters, _ϕ_ and _ψ_, by minimizing the NPE objective
function, as given in general in Section 2.2. Our specific variational objective (up to constants) is thus


_L_ LBF-NPE( _ϕ, ψ_ ) = _−_ E _p_ ( _z,x_ )


- �� ��
_fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ) _−_ log exp  - _fϕ_ ( _x_ ) _[⊤]_ _sψ_ (˜ _z_ )� _h_ (˜ _z_ ) _dz_ ˜ _,_ (6)


where the log-normalizer takes the form of the log of an integral with respect to the base measure
_h_ . The log-normalizer cannot be estimated without bias by Monte Carlo sampling due to the
Jensen gap (Adil Khan et al., 2015). For training, we only require stochastic gradients, which
can be computed using importance sampling. Focusing on the log-normalizer for now, we let
_kϕ,ψ_ ( _x, z_ ) = _fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ). We suppress the dependence on _x_ for now, as it is fixed in the integral.
Let _J_ ( _ϕ, ψ_ ) := - exp _kϕ,ψ_ (˜ _z_ ) _dh_ (˜ _z_ ). The gradient can be computed by estimating an expectation
with respect to an exponentially tilted transformation of _h_ . Let _∇ϕ,ψ_ denote the gradient with respect
to either _ϕ_ or _ψ_ . Then, we have


1
_∇ϕ,ψ_ log _J_ ( _ϕ, ψ_ ) = (7)
_J_ ( _ϕ, ψ_ ) _[· ∇][ϕ,ψ][J]_ [(] _[ϕ, ψ]_ [)]

1                = [ _∇ϕ,ψkϕ,ψ_ (˜ _z_ )] _·_ exp _kϕ,ψ_ (˜ _z_ ) _h_ (˜ _z_ ) _dz_ ˜ (8)
_J_ ( _ϕ, ψ_ ) _[·]_


 -  - exp _kϕ,ψ_ (˜ _z_ ) _h_ (˜ _z_ )
= [ _∇ϕ,ψkϕ,ψ_ (˜ _z_ )] _·_ - exp _kϕ,ψ_ ( _z_ _[′]_ ) _h_ ( _z_ _[′]_ ) _dz_ _[′]_


_dz_ ˜ (9)


             =: [ _∇ϕ,ψkϕ,ψ_ (˜ _z_ )] _· qϕ,ψ_ (˜ _z_ ) _dz_ ˜ (10)


where we recognize an expectation with respect to _qϕ,ψ_, an exponentially tilted density that depends
on the current values of _ϕ, ψ_ . This integral can be estimated by the use of self-normalized importance
sampling (SNIS) with a proposal distribution _r_ (˜ _z_ ) (Owen, 2013):


[ _∇ϕ,ψkϕ,ψ_ (˜ _zj_ )] _w_ (˜ _zj_ )
_,_

 - _P_
_k_ =1 _[w]_ [(˜] _[z][k]_ [)]


[ _∇ϕ,ψkϕ,ψ_ (˜ _z_ )] _qϕ,ψ_ (˜ _z_ ) _dz_ ˜ _≈_


_P_


_j_ =1


where _w_ ( _z_ ) = [exp(] _[k][ϕ,ψ]_ _r_ ( _z_ [(] ) _[z]_ [))] _[h]_ [(] _[z]_ [)] and ˜ _z_ 1 _, . . .,_ ˜ _zP_ _iid∼_ _r_ . The gradient estimator is thus biased for the true

gradient, similar to other gradient estimators targeting the forward KL from the family of “wake-sleep”
algorithms, but consistent as _P_ _→∞_ (Le et al., 2019; Bornschein & Bengio, 2015; McNamara et al.,
2024b).


Algorithm 1 details our gradient computation procedure.


4 VARIANTS & PROPERTIES OF LBF-NPE


We now motivate the construction of the LBF-NPE variational family by examining aspects of its
optimization routine. As a result of parameterizing and targeting the log-density (cf. 5, 7), both
our construction and optimization routine depend entirely on the inner product _fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ). We
elaborate on some key properties and variants of LBF-NPE that stem from this observation.


4.1 AFFINE GRADIENTS


In Section 3.2, we showed that the gradient of the objective for LBF-NPE depends only on the
gradient of _kψ,ϕ_ ( _x, z_ ) = _fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ), via the relation


_∇ϕ,ψL_ LBF-NPE( _ϕ, ψ_ ) = _−_ E _p_ ( _z,x_ )    - _∇ϕ,ψkϕ,ψ_ ( _x, z_ ) _−_ E _qϕ,ψ_ (˜ _z_ ) [ _∇ϕ,ψkϕ,ψ_ ( _x,_ ˜ _z_ )]� (11)


4


**Algorithm 1:** Gradient Computation for LBF-NPE

**Inputs:** Sampling model _p_ ( _z, x_ ); networks _fϕ_ and _sψ_ ; proposal distribution _r_ ( _z_ ).

_iid_
Sample batch _{_ ( _zi, xi_ ) _}_ _[B]_ _i_ =1 _∼_ _p_ ( _z, x_ )
/* Gradient for log-normalizer */
Propose _z_ ˜1 _, . . .,_ ˜ _zP_ _∼_ _r_
Compute _kj_ _[i]_ [:=] _[ k][ϕ,ψ]_ [(] _[x][i][,]_ [ ˜] _[z][j]_ [) =] _[ f][ϕ]_ [(] _[x][i]_ [)] _[⊤][s][ψ]_ [(˜] _[z][j]_ [)] _[,]_ _i ∈_ [ _B_ ] _, j_ _∈_ [ _P_ ]
Compute unnormalized weights _wj_ _[i]_ [= exp] - _kj_ _[i]_ - _· h_ (˜ _zj_ ) _,_ _i ∈_ [ _B_ ] _, j_ _∈_ [ _P_ ]

Compute _Uϕ,ψ_ _[i]_ [=][ �] _j_ _[P]_ =1 _w_ - _j_ _[i]_ _[∇]_ _j_ _[′][ϕ,ψ][w]_ _j_ _[i][k][′]_ _j_ _[i]_ _[,]_ _i ∈_ [ _B_ ]

/* Gradient non-tilted inner product */
Compute _Vϕ,ψ_ _[i]_ [=] _[ ∇][ϕ,ψ][k][ϕ,ψ]_ [(] _[x][i][, z][i]_ [) =] _[ ∇][ϕ,ψ]_ - _fϕ_ ( _xi_ ) _[⊤]_ _sψ_ ( _zi_ )�

/* Compute combined gradient */

Return _∇_ [ˆ] _ϕ,ψ_ = _−_ _B_ [1] - _Bi_ =1 - _Vϕ,ψ_ _[i]_ _[−]_ _[U]_ _ϕ,ψ_ _[ i]_ 

where _qϕ,ψ_ denotes the exponentially tilted density constructed in Section 3.2. Accordingly, the form
of gradient updates for the LBF-NPE procedure is extremely simple; in fact, holding _ψ_ constant,
the gradient with respect to _ϕ_ is that of an affine function of the network outputs _fϕ_ . The same
relationship holds when taking gradients for _ψ_ holding _ϕ_ constant. Targeting such simple functions for
optimization, besides being simple to implement, benefits from a convex formulation (see Section 4.2
below). The invariance of the inner product _kψ,ϕ_ under arbitrary rescalings of _f_ and _s_, on the other
hand, complicates optimization: this motivates a variant of LBF-NPE that reparameterizes outputs to
unit norm (see Section 4.4).


4.2 CONVEXITY


McNamara et al. (2024a) show that neural posterior estimation (NPE) optimizes a convex functional
objective function provided that the variational family is log-concave in _f_, the inference network.
This ensures the forward KL objective of NPE (cf. Equation 3) is convex in _f_ . Recent advances in
the study of wide networks via the neural tangent kernel (NTK) (Jacot et al., 2018) have shown that
fitting network parameters to minimize convex loss functionals (e.g., mean squared error) follows
kernel gradient descent dynamics to a global optimum in the infinite-width limit (Jacot et al., 2018;
McNamara et al., 2024a).


The amortized forward KL objective function that we target (cf. Equation 6) benefits from these same
properties. For an arbitrary collection of basis functions, the objective function _L_ LBF-NPE remains a
convex functional in _f_ . Likewise, for fixed _f_, _L_ LBF-NPE is a convex functional in _s_ . We formalize this
in the proposition below.

**Proposition 1.** _The functional_


_L_ ( _f, s_ ) = _−_ E _p_ ( _z,x_ )


- �� ��
_f_ ( _x_ ) _[⊤]_ _s_ ( _z_ ) _−_ log exp  - _f_ ( _x_ ) _[⊤]_ _s_ (˜ _z_ )� _h_ (˜ _z_ ) _dz_ ˜


_is marginally convex in the arguments f_ _and s, respectively._


A proof and additional discussion are provided in Appendix B. Proposition 1 shows that in the case
where either _f_ or _s_ is fixed, the resulting objective function is fully convex in the remaining argument.
This observation motivates a variant of LBF-NPE where the basis functions are fixed a priori (see
Section 4.3).


4.3 FIXED BASIS FUNCTIONS


Rather than adaptively fitting basis functions _sψ_, the practitioner may simply use a fixed basis defined
ahead of time. This approach is motivated by the convexity of the resulting functional in _f_, as
well as the approaches of related work based on basis expansion parameterizations, which use fixed
orthonormal eigenfunctions (cf. Section 5). In this variant of LBF-NPE, the objective function
_L_ ( _ϕ, ψ_ ) collapses to the marginal _L_ ( _ϕ_ ) for optimization. As we elaborate in Section 4.2, LBF-NPE


5


has a convex formulation in this setting, which empirically results in advantageous optimization
trajectories relative to competing methods (we demonstrate this in Section 6.1).


Several choices of basis may be of interest to practitioners. EigenVI, a related basis-expansion method
for VI (cf. Section 5), utilizes a (truncated) orthonormal basis of eigenfunctions, such as Bernstein,
Legendre, or Hermite polynomials (Cai et al., 2024). Selecting a large _K_ improves faithfulness to the
complete basis, but doing so increases the dimension of the optimization problem, exponentially so
in multiple dimensions. Further, as generally such basis functions are _global_ (i.e., nonzero on all of
the latent space _I_ ), in this design _every_ basis function contributes to the density value _q_ ( _z_ ) at every
point _z_ ; this may make it difficult to control the local behavior of the fitted density.


An alternative approach is to model log _q_ ( _z_ ; _η_ ) = log _q_ ( _z_ ; _fϕ_ ( _x_ )) via a _local_ basis expansion. We
specialize to B-splines (Appendix A.1) and wavelets (Appendix A.2) in our experiments, two rich
families that we recommend for practitioners. In this framework, each basis function is nonzero only
in a small neighborhood of the latent support. Locality of the basis functions simplifies optimization
by inducing a sparser problem than a set of _global_ basis functions would. For a single Monte Carlo
draw ( _z_ _[∗]_ _, x_ _[∗]_ ) _∼_ _p_ ( _z, x_ ), the gradient _−∇η_ log _q_ ( _z_ _[∗]_ ; _η_ ) _|η_ = _fϕ_ ( _x∗_ ) is nonzero at only a few indices
because many basis functions are zero at any given _z_ _[∗]_ .


4.4 REDUCING DEGENERACY THROUGH STEREOGRAPHIC PROJECTION


As noted in Section 4.1, both gradients and the log-density itself only depend on the inner product
_fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ). Adaptively learning both the inference network _fϕ_ and basis function network _sψ_
thus suffers from an inherent lack of identifiability: different rescalings or rotations of the vectors
defined by _fϕ_ and _sψ_ can lead to the same loss function values, since the loss function (Equation 6)
only depends on the inner product. To mitigate this degeneracy, we propose a variant of LBF-NPE
that uses stereographic projection reparameterization to normalize the output tensor onto the unit
hypersphere. This resolves the rescaling degeneracy, but rotational degeneracy persists even after
the projection. Specifically, for a _K_ -dimensional coefficient or basis function vector, we construct
a neural network that outputs a ( _K_ _−_ 1)-dimensional vector _u_ . We then apply the stereographic
projection reparameterization: _y_ = - 1+2 _∥uu∥_ [2] _[,]_ [1] 1+ _[−∥]_ _∥u_ _[u]_ _∥_ _[∥]_ [2][2] 

which maps _u_ _∈_ R _[K][−]_ [1] to a unit vector _y_ _∈_ R _[K]_ such that _∥y∥_ = 1. This normalization mitigates
identifiability issues, and the reparameterization yields strong results in our experiments (see Appendix D for additional discussion). Our loss function takes the following form when we apply this
reparameterization:


_L_ ˆLBF-NPE( _ϕ, ψ_ ) = E _p_ ( _z,x_ )


- �� - - ��
_−wf_ [ˆ] _ϕ_ ( _x_ ) _[⊤]_ _s_ ˆ _ψ_ ( _z_ ) + log exp _wf_ [ˆ] _ϕ_ ( _x_ ) _[⊤]_ _s_ ˆ _ψ_ (˜ _z_ ) _h_ (˜ _z_ ) _dz_ ˜ _,_ (12)


where _f_ [ˆ] _ϕ_ ( _·_ ) and ˆ _sψ_ ( _·_ ) are reparameterized network outputs, and _w_ is a fixed scaling hyperparameter.


5 RELATED WORK


Exponential family distributions are a common class of distributions for both traditional variational
inference and NPE. In the simplest cases, Gaussian, Bernoulli, and other “simple” exponential
families are used (Liu et al., 2023; Cranmer et al., 2020; Blei et al., 2017). Typically, however, these
distributions are not parameterized in canonical form (where _η_ = _fϕ_ ( _x_ ) is the natural parameter of the
family). However, for NPE, McNamara et al. (2024a) recommend using the canonical parameterization, even for simple families such as the Gaussian, to benefit from convex loss (Section 4.2). General
exponential families parameterized by neural networks were first proposed in Pacchiardi & Dutta
(2022) to represent the _likelihood_ function for likelihood-free settings. Akin to Approximate Bayesian
Computation (ABC) methods, this approach aims to learn low-dimensional summary statistics of _x_
to represent the likelihood, and subsequently performs inference with potentially expensive MCMC
or ABC routines. We compare to this approach in Appendix E.6. To our knowledge, LBF-NPE is
the first method to utilize neural exponential families to represent the posterior distribution and to
use this family within the amortized inference setting. LBF-NPE is also unique in exploiting the low
dimensionality of the posterior projections of interest.


6


Parameterizing variational distributions via basis expansions is a relatively new line of research; a
recent non-amortized approach, EigenVI (Cai et al., 2024), presents an algorithm for optimizing a
score-based divergence with a variational family parameterized via a linear combination of orthogonal
eigenfunctions. Key limitations of this approach are i) the lack of amortization and ii) the necessity of
utilizing orthogonal, fixed eigenfunctions as the basis: truncation of these bases necessarily introduces
approximation error. In our case, _sψ_ is unrestricted: the basis functions can be arbitrary, and so a
fixed number _K_ may be sufficient for some classes of posteriors (cf. Section 6.2).


Mixtures of Gaussians and normalizing flows are other common choices of variational families for
NPE (Gershman et al., 2012; Papamakarios & Murray, 2016; Papamakarios et al., 2021; Rezende
& Mohamed, 2015). Although more flexible than simple exponential families, these parameterizations may suffer from convergence to shallow local optima during optimization (cf. Section 6.1).
We compare LBF-NPE to mixtures of Gaussians, normalizing flows, as well as EigenVI in our
experiments.


6 EXPERIMENTS


In numerical experiments, we fit a variety of complex posterior distributions using LBF-NPE. In
Sections 6.1 and 6.4, we infer one-dimensional posterior projections using the variant of our method
with fixed basis functions, whereas in Sections 6.2 and 6.3, we infer two-dimensional posterior
projections using adaptive basis functions. Additional details about each of these experiments appear
in Appendix D.


NPE with various alternative variational families serves as our primary benchmarks; we can compare
to these methods quantitatively by assessing the NPE objective with each choice of variational
distribution. We benchmark NPE with variational families based on mixture density networks
(MDNs), RealNVP, and neural spline flows (Papamakarios & Murray, 2016; Durkan et al., 2019). In
addition to the results in this section, additional results appear in Appendix E, including results from
comparisons to two non-NPE-based variational inference methods: EigenVI (Appendix E.5) and a
score-matching neural-likelihood-based method for likelihood-free inference (Appendix E.6).


6.1 TOY EXAMPLE: SINUSOIDAL LIKELIHOOD


Figure 1: Negative log likelihood of our method and the MDN. Each model is trained with 20
different random seeds, and the records are smoothed using a Gaussian filter with _σ_ = 20 _._ 0.


We first exhibit the advantages of LBF-NPE’s convex variational objective by demonstrating consistent
convergence on a highly multimodal problem with fixed basis functions (cf. Section 4.3). The model
draws an angle _z_ _∼_ Unif[0 _,_ 2 _π_ ] followed by _x_ _|_ _z_ _∼N_ �sin(2 _z_ ) _, σ_ [2][�] for fixed _σ_ [2] = 1. The
exact posteriors _p_ ( _z_ _|_ _x_ ) have up to four modes, depending on the realization _x_ . We compare
LBF-NPE, using a fixed collection of 14 B-spline functions of degree two on a mesh of [0 _,_ 2 _π_ ] (see
Appendix A.1 for additional detail on B-splines), and a mixture density network (MDN) (Bishop,
1994; Papamakarios & Murray, 2016), using a mixture of five Gaussian distributions. Both variational
distributions have 14 distributional parameters per observation _x_ (there are four mixture weight
parameters for the MDN due to the simplex constraint). Additional experimental details are given
in Appendix D. Figure 1 shows that for 20 different runs of the optimization routine, LBF-NPE


7


Table 1: Forward KL divergence (FKL), reverse KL divergence (RKL), and negative log-likelihood
(NLL) of LBF-NPE (ours), NSF (Neural Spline Flow), RealNVP, and MDN on three 2D test cases.
Lower values indicate better posterior approximation.


**Forward KL Divergence**


**LBF-NPE** **NSF** **RealNVP** **MDN**


**Bands** **0.0048 (** _±_ **0.0003)** 0.016 ( _±_ 0.003) 0.015 ( _±_ 0.005) 0.182 ( _±_ 0.01)
**Ring** **0.0054 (** _±_ **0.0005)** 0.017 ( _±_ 0.004) 0.024 ( _±_ 0.005) 0.205 ( _±_ 0.02)
**Spiral** **0.187 (** _±_ **0.004)** 0.201 ( _±_ 0.01) 0.545 ( _±_ 0.07) 0.948 ( _±_ 0.09)


**Reverse KL Divergence**


**LBF-NPE** **NSF** **RealNVP** **MDN**


**Bands** **0.0014 (** _±_ **0.0004)** 0.0099 ( _±_ 0.001) 0.011 ( _±_ 0.007) 0.156 ( _±_ 0.02)
**Ring** **0.0027 (** _±_ **0.0003)** 0.013 ( _±_ 0.003) 0.014 ( _±_ 0.003) 0.204 ( _±_ 0.01)
**Spiral** **0.188 (** _±_ **0.005)** 0.322 ( _±_ 0.04) 0.666 ( _±_ 0.09) 1.973 ( _±_ 0.14)


**Negative Log-Likelihood**


**LBF-NPE** **NSF** **RealNVP** **MDN**


**Bands** **-0.060 (** _±_ **0.07)** 0.151 ( _±_ 0.23) 0.157 ( _±_ 0.22) 1.389 ( _±_ 0.41)
**Ring** **0.030 (** _±_ **0.03)** 0.621 ( _±_ 0.24) 0.733 ( _±_ 0.11) 1.031 ( _±_ 0.18)
**Spiral** 0.838 ( _±_ 0.13) **0.727 (** _±_ **0.25)** 0.859 ( _±_ 0.32) 2.788 ( _±_ 0.31)


Figure 2: Example posteriors of three problems in two dimensions. NSF refers to Neural Spline Flow.


consistently converges to the same solution, whereas the MDN sometimes converges to a suboptimal
local optimum. Visualizations of posterior approximations from both methods are provided in
Appendix E.


6.2 COMPLEX MULTIVARIATE REPRESENTATIONS IN 2D


We showcase LBF-NPE on three test problems in two dimensions, named “banded”, “ring”, and
“spiral,” and visualized in the left column of Figure 2. Each model consists of a two-dimensional
latent variable _z_ _∈_ R [2] and an observation _x_ _∈_ R, and in some cases nuisance latent variables as
well. Further details of the generative processes for these three problems are provided in Appendix D
(Sections D.2 to D.4).


LBF-NPE is able to approximate these complex posteriors nearly perfectly using only 20 adaptive
basis functions _sψ_ . Both the amortization function _fϕ_ and the basis functions _sψ_ are parameterized


8


using deep neural networks. For additional implementation details, we refer the reader to Appendix D.
We follow Algorithm 1 and evaluate both the variational posterior _q_ ( _z_ ; _fϕ_ ( _x_ ) _, sψ_ ( _z_ )) and the exact
posterior _p_ ( _z_ _|_ _x_ ) on a fine mesh grid that covers the support of the posterior. Qualitative results
appear in Figure 2 and quantitative results appear in Table 1 for held-out test points. LBF-NPE
outperforms the MDN and multiple types of normalizing flow in nearly all cases. We provide
additional visualizations of the variational approximations found by LBF-NPE and its competitors in
Appendix E.


6.3 OBJECT DETECTION


We apply LBF-NPE with adaptive basis functions to
the problem of object detection in astronomical images.
We use a generative model resembling the scientific
model of Liu et al. (2023). In brief, this generative process first independently samples star locations _l_ 1 _, l_ 2 _∼_
Unif ([0 _,_ 16] _×_ [0 _,_ 16]) and star fluxes _f_ 1 _, f_ 2 _∼N_ ( _µ, σ_ [2] )
for two objects. Afterward, a latent noise-free image _I_ is
rendered by convolving these point sources with a pointspread function (PSF). Finally, given _I_, the intensity of
each pixel ( _j_, _k_ ), for _j, k_ _∈{_ 1 _, . . .,_ 16 _}_ is independently
drawn as _xjk_ _∼_ Poisson( _Ijk_ ), reflecting Poisson shot
noise.


Figure 3 shows two examples of the noisy observations
_x_, along with the posterior approximations for the locations of each. The posterior distribution for this problem
is multimodal with a high degree of separation between
modes. LBF-NPE parameterizes this shape effectively. Figure 3: Two example posteriors (left)
Inference on locations of objects would initially appear to conditional on the observed images _x_
be difficult for LBF-NPE, as our method does not directly (right). In each case, LBF-NPE correctly
parameterize location parameters: instead, the learned recovers the locations of the two objects.
basis functions must be expressive enough to parameterize arbitrary pairs of separated modes. In our ablations, we vary the number of basis functions
_K_ = 9 _,_ 20 _,_ 36 _,_ 64 for LBF-NPE and provide visualizations in Appendix E.3. Examining the form
of the fitted basis functions across varying _K_ for this problem is particularly illustrative of the
expressivity of the adaptive approach to fitting the basis functions compared to a fixed basis function
framework.


6.4 CASE STUDY: REDSHIFT ESTIMATION


The redshift of galaxies is a key quantity of Table 2: Total held-out NLL of the true redshift _z_ .
interest as it characterizes their distances
from Earth. Redshift measures the ex
**LBF-NPE** **NSF** **MDN**

tent to which electromagnetic waves are
“stretched” to redder wavelengths as objects **NLL** **-57,220 (** _±_ **152)** -55,389 ( _±_ 379) -50,648 ( _±_ 322)
move away from Earth. The distribution of
redshifts across many objects is a powerful
probe of cosmology (Malz & Hogg, 2022). Redshift estimation from photometric data (images) is
referred to as photo- _z_ estimation. We extend the methodology of the Bayesian Light Source Separator
(BLISS) package (Liu et al., 2023), a state-of-the-art package for probabilistic object detection in
astronomical images, for this task, by adding a redshift density estimation “head” to the existing
BLISS network. To each detected object, we associate a redshift probability density function fitted
by LBF-NPE with a fixed B-spline basis family. LSST DESC DC2 Simulated Sky Survey dataset
(LSST Dark Energy Science Collaboration et al., 2021) serves as the generative model, providing
simulated ( _z, x_ ) pairs, where _z_ denotes redshift and _x_ are the astronomical images. This highly
realistic dataset consists of mock catalogs of astronomical images generated by directly modeling
known physical quantities of the universe using empirical priors and physics-informed modeling.
Appendix D provides further details of the experimental setting.


Table 2: Total held-out NLL of the true redshift _z_ .


**LBF-NPE** **NSF** **MDN**


**NLL** **-57,220 (** _±_ **152)** -55,389 ( _±_ 379) -50,648 ( _±_ 322)


9


We compare LBF-NPE with variational families based on neural spline flow (NSF) and a mixture
density network (MDN), all embedded with the BLISS framework. The MDN uses five Gaussian
components, in keeping with state-of-the-art work on photo- _z_ estimation (Merz et al., 2025); adding
more components did not improve the quality of fit. The only difference between the two approaches
is the choice of parameterization for the variational family. We compute the negative log-likelihood
(NLL) of a held-out test set of 153,000 astronomical objects. Table 2 shows that LBF-NPE with the
B-spline variational family parameterization outperforms both the MDN and the NSF.


7 DISCUSSION & LIMITATIONS


LBF-NPE models the log density of the variational distribution as a linear combination of expressive
basis functions, which is beneficial for several reasons. First, log-space modeling results in a
_multiplicative_ influence of different basis functions. Regions of latent space can effectively be
“zeroed-out” more easily in this context compared to performing the modeling in density space
directly. Second, our model of the log density results in an unconstrained optimization problem in
_fϕ_ and _sψ_ : the coefficients and basis functions may be either positive or negative, whereas other
density estimation methods may require nonnegativity or other constraints on the coefficients or
basis functions to obtain a valid density that integrates to unity (Cai et al., 2024; Koo & Kim,
1996; Kirkby et al., 2023). Third, unlike other methods, LBF-NPE may be applied to the growing
class of “likelihood-free inference” problems (Cranmer et al., 2020; Thomas et al., 2022) as fitting
the variational posteriors only requires samples from the joint model _p_ ( _z, x_ ). The advantageous
marginalization properties of the forward KL criterion pair nicely with this setting: the generative
model can have arbitrary nuisance latent variables that are implicitly marginalized out (Ambrogioni
et al., 2019). We provide an additional example marginalizing over the parameters of a Bayesian
neural network (BNN) in Appendix E.8.


Using basis expansions to parameterize variational distributions is a recent and exciting innovation
in variational inference. Relative to EigenVI, LBF-NPE performs better with fewer basis functions
(Appendix E.5). This may be due to removing orthogonality constraints and adaptively fitting basis
functions: by averaging across posteriors for arbitrary _x_, our approach implicitly regularizes the basis
functions, preventing overfitting to any single instance. Appendix E contains additional discussion
and visualization of fitted basis functions _sψ_ .


The main limitation of LBF-NPE is the difficulty of sampling from the variational distribution.
LBF-NPE directly fits the log density of the variational distribution, but samples from this density are
typically needed for inference. The low dimensionality of many NPE targets ensures that sampling
is straightforward: inverse transform sampling is readily applicable (cf. Appendix C). For higherdimensional targets, importance sampling may be adequate to estimate functionals with respect to the
variational distribution.


Despite this sampling challenge, our approach demonstrates that basis expansion methods offer a
compelling middle ground between optimization simplicity and expressivity for NPE. Future work
could explore new bilevel optimization approaches (Xiao & Chen, 2025) to jointly learn adaptive basis
functions alongside their coefficients. Additionally, extending our approach beyond low-dimensional
targets to high-dimensional targets with simplifying structure—such as known or assumed conditional
independencies—could broaden the applicability of basis expansion methods.


10


ACKNOWLEDGMENTS


We thank the reviewers for their helpful comments and suggestions. This material is based on work
supported by the National Science Foundation under Grant Nos. 2209720 (OAC) and 2241144
(DGE), and the U.S. Department of Energy, Office of Science, Office of High Energy Physics under
Award Number DE-SC0023714.


REFERENCES


M. Adil Khan, G. Ali Khan, T. Ali, and A. Kilicman. On the refinement of Jensen’s inequality.
_Applied Mathematics and Computation_, 262:128–135, 2015.


Luca Ambrogioni, Umut Güçlü, Julia Berezutskaya, Eva van den Borne, Yagmur Güçlütürk, Maxˇ
Hinne, Eric Maris, and Marcel van Gerven. Forward amortized inference for likelihood-free
variational marginalization. In _International Conference on Artificial Intelligence and Statistics_,
2019.


Jason Ansel, Edward Yang, Horace He, et al. PyTorch 2: Faster Machine Learning Through Dynamic
Python Bytecode Transformation and Graph Compilation. In _ACM International Conference on_
_Architectural Support for Programming Languages and Operating Systems_, 2024.


Francis Bach. Breaking the curse of dimensionality with convex neural networks. _Journal of Machine_
_Learning Research_, 18(19):1–53, 2017.


Yoshua Bengio, Nicolas Roux, Pascal Vincent, Olivier Delalleau, and Patrice Marcotte. Convex
neural networks. In _Conference on Neural Information Processing Systems_, 2005.


Christopher M. Bishop. Mixture density networks. Technical Report NCRG/94/004, Aston University,
Birmingham, UK, 1994.


David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational inference: A review for statisticians.
_Journal of the American Statistical Association_, 112(518):859–877, 2017.


Jörg Bornschein and Yoshua Bengio. Reweighted wake-sleep. In _International_ _Conference_ _on_
_Learning Representations_, 2015.


James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang.
JAX: composable transformations of Python+NumPy programs, 2018.


Diana Cai, Chirag Modi, Charles C. Margossian, Robert M. Gower, David M. Blei, and Lawrence K.
Saul. EigenVI: score-based variational inference with orthogonal function expansions. In _Confer-_
_ence on Neural Information Processing Systems_, 2024.


Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In _International Conference on Machine Learning_,
2020.


Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference.
_Proceedings of the National Academy of Sciences_, 117(48):30055–30062, 2020.


Carl d. Boor. _A Practical Guide to Splines_ . Springer Verlag, New York, 1978.


Maximilian Dax, Stephen R Green, Jonathan Gair, Jakob H Macke, Alessandra Buonanno, and
Bernhard Schölkopf. Real-time gravitational wave science with neural posterior estimation.
_Physical review letters_, 127(24):241103, 2021.


Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. ArcFace: Additive angular margin
loss for deep face recognition. In _Conference on Computer Vision and Pattern Recognition_, 2019.


Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. In
_International Conference on Learning Representations_, 2017.


11


Justin Domke. Provable smoothness guarantees for black-box variational inference. In _International_
_Conference on Machine Learning_, 2020.


Justin Domke, Robert M. Gower, and Guillaume Garrigos. Provable convergence guarantees for
black-box variational inference. In _Conference on Neural Information Processing Systems_, 2023.


Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In
_Conference on Neural Information Processing Systems_, 2019.


Paul H. C. Eilers and Brian D. Marx. Flexible smoothing with B-splines and penalties. _Statistical_
_Science_, 11(2):89–121, 1996.


Ankush Ganguly, Sanjana Jain, and Ukrit Watchareeruetai. Amortized variational inference: A
systematic review. _Journal of Artificial Intelligence Research_, 78:1–49, 1 2024.


Samuel Gershman, Matt Hoffman, and David Blei. Nonparametric variational inference. In _Interna-_
_tional Conference on Machine Learning_, 2012.


Derek Hansen, Ismael Mendoza, Runjing Liu, Ziteng Pang, Zhe Zhao, Camille Avestruz, and Jeffrey
Regier. Scalable Bayesian inference for detection and deblending in astronomical images. In _ICML_
_Workshop on Machine Learning for Astrophysics_, 2022.


Elad Hoffer and Nir Ailon. Deep metric learning using triplet network. In _Similarity-Based Pattern_
_Recognition_, 2015.


Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. In _Conference on Neural Information Processing Systems_, 2018.


Glenn Jocher et al. YOLOv5. [https://github.com/ultralytics/yolov5, 2020.](https://github.com/ultralytics/yolov5)


Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvärinen. Variational autoencoders
and nonlinear ICA: A unifying framework. In _International Conference on Artificial Intelligence_
_and Statistics_, 2020.


Patrick Kidger and Cristian Garcia. Equinox: neural networks in JAX via callable PyTrees and
filtered transformations. _Differentiable Programming Workshop at Neural Information Processing_
_Systems_, 2021.


Diederik P. Kingma and Max Welling. An introduction to variational autoencoders. _Foundations and_
_Trends in Machine Learning_, 12(4):307–392, 2019.


J. Lars Kirkby, Álvaro Leitao, and Duy Nguyen. Spline local basis methods for nonparametric density
estimation. _Statistics Surveys_, 17:75–118, 2023.


Ja-Yong Koo and Woo-Chul Kim. Wavelet density estimation by approximation of log-densities.
_Statistics & Probability Letters_, 26(3):271–278, 1996.


Tuan Anh Le, Adam R. Kosiorek, N. Siddharth, Yee Whye Teh, and Frank Wood. Revisiting
reweighted wake-sleep for models with stochastic control flow. In _Conference on Uncertainty in_
_Artificial Intelligence_, 2019.


Jongmin Lee, Joo Young Choi, Ernest K Ryu, and Albert No. Neural tangent kernel analysis of deep
narrow neural networks. In _International Conference on Machine Learning_, 2022.


Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and LSST Dark Energy Science Collaboration.
Variational inference for deblending crowded starfields. _Journal of Machine Learning Research_,
24(179):1–36, 2023.


Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. SphereFace: Deep
hypersphere embedding for face recognition. In _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition_, 2017.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Confer-_
_ence on Learning Representations_, 2019.


12


LSST Dark Energy Science Collaboration, Bela Abolfathi, David Alonso, Robert Armstrong, Éric
Aubourg, et al. The LSST DESC DC2 simulated sky survey. _The Astrophysical Journal Supplement_
_Series_, 253(1):31, 2021.


LSST Dark Energy Science Collaboration, Bela Abolfathi, Robert Armstrong, Humna Awan, Yadu N.
Babuji, et al. DESC DC2 data release note, 2022.


David J. C. MacKay. A practical Bayesian framework for backpropagation networks. _Neural_
_Computation_, 4(3):448–472, 1992a.


David J. C. MacKay. Bayesian interpolation. _Neural Computation_, 4(3):415–447, 1992b.


Alex I. Malz and David W. Hogg. How to obtain the redshift distribution from probabilistic redshift
estimates. _The Astrophysical Journal_, 928(2):127, 2022.


Declan McNamara, Jackson Loper, and Jeffrey Regier. Globally convergent variational inference. In
_Conference on Neural Information Processing Systems_, 2024a.


Declan McNamara, Jackson Loper, and Jeffrey Regier. Sequential Monte Carlo for inclusive
KL minimization in amortized variational inference. In _International Conference on Artificial_
_Intelligence and Statistics_, 2024b.


Grant Merz, Xin Liu, Samuel Schmidt, Alex I. Malz, Tianqing Zhang, et al. DeepDISC-photoz: Deep
learning-based photometric redshift estimation for Rubin LSST. _The Open Journal of Astrophysics_,
8, 2025.


Radford M. Neal. _Bayesian Learning for Neural Networks_ . Springer-Verlag, Berlin, Heidelberg,
1996.


Art B. Owen. _Monte_ _Carlo_ _Theory,_ _Methods_ _and_ _Examples_, chapter 9, pp. 265–294. Stanford
University, 2013. Chapter on Importance Sampling. Available at [https://artowen.su.](https://artowen.su.domains/mc/)
[domains/mc/.](https://artowen.su.domains/mc/)


Lorenzo Pacchiardi and Ritabrata Dutta. Score matched neural exponential families for likelihood-free
inference. _Journal of Machine Learning Research_, 23(38):1–71, 2022.


George Papamakarios and Iain Murray. Fast _ϵ_ -free inference of simulation models with Bayesian
conditional density estimation. In _Conference on Neural Information Processing Systems_, 2016.


George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji
Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. _Journal_ _of_
_Machine Learning Research_, 22(57):1–64, 2021.


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, et al. PyTorch: An imperative style, high-performance deep learning library. In _Conference_
_on Neural Information Processing Systems_, 2019.


Aakash Patel, Tianqing Zhang, Camille Avestruz, Jeffrey Regier, and The LSST Dark Energy Science
Collaboration. Neural posterior estimation for cataloging astronomical images with spatially
varying backgrounds and point spread functions. _The Astronomical Journal_, 170(3):155, 2025.


Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In _International_
_Conference on Machine Learning_, 2015.


Kihyuk Sohn. Improved deep metric learning with multi-class N-pair loss objective. In _Conference_
_on Neural Information Processing Systems_, 2016.


Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In _International_
_Conference on Learning Representations_, 2021.


Bharath Sriperumbudur, Kenji Fukumizu, Arthur Gretton, Aapo Hyvärinen, and Revant Kumar.
Density estimation in infinite dimensional exponential families. _Journal of Machine Learning_
_Research_, 18(57):1–59, 2017.


13


Manoj Kumar Srivastava, Abdul Hamid Khan, and Namita Srivastava. _Statistical Inference:_ _Theory_
_of Estimation_ . PHI Learning, 2014.


J. Michael Steele. _Stochastic calculus and financial applications_ . Springer, New York, 2010.


Owen Thomas, Ritabrata Dutta, Jukka Corander, Samuel Kaski, and Michael U. Gutmann. Likelihoodfree inference by ratio estimation. _Bayesian Analysis_, 17(1):1–31, 3 2022.


Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In _European_
_Conference on Computer Vision_, 2020.


Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational
inference. _Foundations and Trends in Machine Learning_, 1(1-2):1–305, 2008.


Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei
Liu. CosFace: Large margin cosine loss for deep face recognition. In _IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, 2018.


Xun Wang, Xintong Han, Weilin Huang, Dengke Dong, and Matthew R. Scott. Multi-similarity
loss with general pair weighting for deep metric learning. In _Conference on Computer Vision and_
_Pattern Recognition_, 2019.


Daniel Ward, Patrick Cannon, Mark Beaumont, Matteo Fasiolo, and Sebastian Schmon. Robust
neural posterior estimation and statistical model criticism. _Conference on Neural Information_
_Processing Systems_, 2022.


Stephan Wojtowytsch. On the convergence of gradient descent training for two-layer ReLU-networks
in the mean field regime, 2020.


Quan Xiao and Tianyi Chen. Unlocking global optimality in bilevel optimization: A pilot study. In
_International Conference on Learning Representations_, 2025.


Mang Ye, Xu Zhang, Pong C. Yuen, and Shih-Fu Chang. Unsupervised embedding learning via invariant and spreading instance feature. In _Conference on Computer Vision and Pattern Recognition_,
2019.


Cheng Zhang, Judith Bütepage, Hedvig Kjellström, and Stephan Mandt. Advances in variational
inference. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 41(8):2008–2026,
2019.


Dingyi Zhang, Yingming Li, and Zhongfei Zhang. Deep metric learning with spherical embedding.
In _Conference on Neural Information Processing Systems_, 2020.


Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed
gradients. In _Conference on Neural Information Processing Systems_, 2020.


14


A B-SPLINE & WAVELET BASIS FUNCTIONS


We give examples of two classes of local basis functions defined on the unit interval [0 _,_ 1]: B-splines
and wavelets. Without loss of generality, these definitions can be extended to construct the family
on any interval _I_ = [ _a, b_ ] for _a, b_ _∈_ R. These families are potential candidates for a local basis
parameterization of the variational posterior.


A.1 B-SPLINES


For a choice of degree _d ≥_ 1 and a uniformly spaced set of points _{ti}_ _[K]_ _i_ =1 [, the B-splines are a set of]
local basis functions _{b_ [(] _i_ _[d]_ [)] ( _z_ ) _}_ _[K]_ _i_ =1 [that are defined recursively as follows (d. Boor, 1978; Eilers &]
Marx, 1996):


_b_ [(] _i_ _[d]_ [)] ( _z_ ) = 1 on [ _ti, ti_ +1) o.w. 0 for all _i, z_, if _d_ = 0 (13)

_b_ [(] _i_ _[d]_ [)] ( _z_ ) _z −_ _ti_ _·_ _[b]_ _i_ [(] _[d][−]_ [1)] ( _z_ ) + _ti −_ _z_ _·_ _[b]_ _i_ [(] +1 _[d][−]_ [1)] ( _z_ ) _d ≥_ 1 _._ (14)
( _ti_ + _d −_ _ti_ ) [=] _ti_ + _d−_ 1 _−_ _ti_ _ti_ + _d−_ 1 _−_ _ti_ _ti_ + _d −_ _ti_ _ti_ + _d −_ _ti_


The B-spline basis functions _b_ [(] _i_ _[d]_ [)] ( _z_ ) are thus individually piecewise polynomial splines of degree _d_ .
While each function _b_ [(] _i_ _[d]_ [)] is symmetric about the _i_ th knot _ti_ (or a midpoint of two knots), it is nonzero
for a range of only 2 _d_ knots, in accordance with the _locality_ aspect described above (see Figure 4).


Figure 4: Example visualizations of B-Spline basis functions of degree 1 (left) and 2 (right). For
brevity, we show only a subset of the basis functions.


A.2 WAVELETS


A collection of wavelet local basis functions on [0 _,_ 1] is defined relative to a “mother wavelet” function
denoted _H_ . For ease of exposition, we consider the Haar wavelet (Koo & Kim, 1996; Steele, 2010)
given by


_H_ ( _z_ ) =






1 _,_ 0 _≤_ _z_ _<_ [1] 2

_−_ 1 _,_ [1] _[≤]_ _[z]_ _[≤]_


_−_ 1 _,_ 2 _[≤]_ _[z]_ _[≤]_ [1] (15)

0 _,_ otherwise





Thereafter, the set of local basis functions _bi_ is defined recursively as follows: writing each _i_ uniquely
as _i_ = 2 _[j]_ + _k_, _j_ _≥_ 0 _,_ 0 _≤_ _k_ _<_ 2 _[j]_, we have

_bi_ ( _z_ ) = 2 _[j/]_ [2] _H_ (2 _[j]_ _· z −_ _k_ ) _._ (16)


The local basis functions are thus defined as shifted and scaled versions of the mother wavelet _H_ .
One can check that _bi_ ( _z_ ) is nonzero only on the interval [ _k ·_ 2 _[−][j]_ _,_ ( _k_ + 1) _·_ 2 _[−][j]_ ] for _i_ = 2 _[j]_ + _k_, so
the basis functions become nonzero on increasingly local regions even for moderate values of _i_ (say,
_i_ = 200). Wavelets are often described as being local with respect to both space _and_ frequency as _bi_
becomes increasingly “spiky” as well due to the coefficient 2 _[j/]_ [2] (Steele, 2010).


15


B CONVEXITY


B.1 CONVEXITY & CONVERGENCE OF LBF-NPE WITH FIXED BASIS FUNCTIONS


As referenced in Section 4.2, in the setting where the basis functions _sψ_ are fixed ahead of time, the
objective function of LBF-NPE becomes a convex functional of the amortization network _f_ . In this
case, our setting can be shown to be globally convergent under suitable regularity conditions on the
network architecture, in the asymptotic limit as the network width tends arbitrarily large. We restate
this result, proven in McNamara et al. (2024a), below.

**Proposition.** _Let X_ _⊆_ R _[d]_ _and Y_ _⊆_ R _[K]_ _._ _Let fϕ_ = _f_ ( _·_ ; _ϕ_ ) : _X_ _→_ R _[K]_ _be parameterized as a scaled_
_two-layer ReLU network of width p, i.e._ _fi_ ( _x_ ; _ϕ_ ) = ~~_√_~~ 1 _p_ - _pj_ =1 _[a][ij][σ]_ [(] _[x][⊤][w][j]_ [)] _[ for][ i]_ [ = 1] _[, . . ., K][.]_ _[Define]_
_the loss functional_

_ℓ_ ( _x, η_ ) = KL( _p_ ( _z_ _| x_ ) _|| q_ ( _z_ ; _η_ )) _,_


_and allow parameters ϕ_ = _{aij, wj}, i_ = 1 _, . . ., K, j_ = 1 _, . . . p to evolve via the gradient flow ODE_
_ϕ_ ˙( _t_ ) = _−∇ϕ_ E _p_ ( _x_ ) _ℓ_ ( _x, f_ ( _x_ ; _ϕ_ ( _t_ ))) _._ _Then we have the following results (under regularity conditions_
_(A1)–(A6)):_


_a)_ _LLBF-NPE_ ( _ϕ_ ( _t_ )) _is_ _precisely_ E _p_ ( _x_ ) _ℓ_ ( _x, f_ ( _x_ ; _ϕ_ ( _t_ )) _,_ _optimized_ _by_ _the_ _gradient_ _flow_ _above._
_Further, the functional MLBF-NPE_ ( _f_ ) : _f_ _�→_ E _p_ ( _x_ ) _ℓ_ ( _x, f_ ( _x_ )) _is a convex functional in f_ _, with_
_a global optimum f_ _[∗]_ _._


_b)_ _For the parameterization above, with ϕ following the gradient flow ODE, we have that there_
_exists T_ _>_ 0 _such that_


lim
_p→∞_ _[M][LBF-NPE]_ [(] _[f][T]_ [ )] _[ ≤]_ _[M][LBF-NPE]_ [(] _[f][ ∗]_ [) +] _[ ϵ,]_


_where fT_ = _f_ ( _·_ ; _ϕ_ ( _T_ )) _._


The proposition above, proven in (McNamara et al., 2024a), states that gradient descent on the convex
functional E _p_ ( _x_ ) _ℓ_ ( _x, f_ ( _x_ ; _ϕ_ )) converges arbitrarily close to its optimum in the infinite-width limit,
relying on universality results of shallow networks and NTK theory (Jacot et al., 2018; Lee et al.,
2022). Our parameterization of the variational distribution _q_ as an exponential family in LBF-NPE
falls into this setting when the basis functions are fixed, allowing us to directly apply results from
(McNamara et al., 2024a). We refer to the proofs therein rather than a restatement here.


Regularity conditions sufficient for the above to hold are provided below. They assume a well-behaved
functional _M_, a particular initialization of the width- _p_ network, and uniform boundedness conditions
on gradients along the optimization trajectory. Although we emphasize that NTK-style results only
approximately explain the success of our method or other neural posterior estimation methods in
practice (the infinite-width limit can only be approximated by finite networks, and the continuous
gradient flow ODE is approximated by stochastic gradient descent), results like these prove that
LBF-NPE benefits from an advantageous optimization landscape asymptotically.


(A1) The data space _X_ is compact.


_a.s._ _iid_
(A2) Weights are initialized as _aij_ = 0, _wj_ _∼N_ (0 _, Id_ ).


(A3) The neural tangent kernel at initialization, _Kϕ_ ( _x, x_ _[′]_ ) = _Jf_ ( _x_ ; _ϕ_ ) _Jf_ ( _x_ _[′]_ ; _ϕ_ ) _[⊤]_ at _ϕ_ = _ϕ_ (0),
is dominated by some integrable random variable _G_, uniformly over _x, x_ _[′]_ .


(A4) The gradient _∇ηℓ_ ( _x, η_ ) is uniformly bounded for all _x, η_ .


(A5) The limiting NTK _K∞_ = lim _p→∞_ _Kϕ_ is positive-definite (we note that the existence of the
limit is guaranteed).


(A6) The functional _M_ LBF-NPE is bounded below; and its minimizer _f_ _[∗]_ has finite norm with
respect to the RKHS norm of the limiting NTK _K∞_ .


(A7) The function _ℓ_ ( _x, η_ ) is _C_ -smooth in _η_ for some _C_ _< ∞_ .


16


B.2 CONVEXITY OF LBF-NPE


In this subsection, we prove the convexity of the joint functional _L_ ( _f, s_ ) in Proposition 1. Neural
network theory and NTK-based analysis of this objective in _s_ are beyond the scope of this work. We
prove marginal convexity in _s_, and appeal to previous NTK-based results (such as the above), which
motivate our empirical results: optimization of convex functionals of neural network outputs is advantageous compared to the optimization of nonconvex functionals due to the preferable optimization
landscape of the former (Bach, 2017; Bengio et al., 2005; Jacot et al., 2018; Wojtowytsch, 2020).


We first present Hölder’s inequality, which we’ll use in the proof.


**Lemma 1** (Hölder) **.** _If S_ _is a measurable subset of_ R _[n]_ _(with respect to Lebesgue measure), and f_
_and g are measurable real-valued functions on S, then Hölder’s inequality is_


��
_|f_ ( _x_ ) _g_ ( _x_ ) _|_ d _x ≤_
_S_


    - _q_ [1]
_|g_ ( _x_ ) _|_ _[q]_ d _x_ _._
_S_


    - _p_ [1]

[��]
_|f_ ( _x_ ) _|_ _[p]_ d _x_
_S_


_for any p, q satisfying_ [1]


[1] [1]

_p_ [+] _q_


_q_ [= 1] _[.]_


Hölder’s inequality will be used to prove Proposition 1, restated below.


**Proposition 1.** _The functional_


_L_ ( _f, s_ ) = _−_ E _p_ ( _z,x_ )


- �� ��
_f_ ( _x_ ) _[⊤]_ _s_ ( _z_ ) _−_ log exp  - _f_ ( _x_ ) _[⊤]_ _s_ (˜ _z_ )� _h_ (˜ _z_ ) _dz_ ˜


_is marginally convex in the arguments f_ _and s, respectively._


_Proof._ Ignore the outer expectation for now, and consider a fixed _x, z_ drawn from _p_ ( _z, x_ ). The
inner product _−f_ ( _x_ ) _[⊤]_ _s_ ( _z_ ) is clearly convex in _s_ . We now turn to the more complicated expression,
log �� exp - _f_ ( _x_ ) _[⊤]_ _s_ ( _z_ _[′]_ )� _dz_ _[′]_ [�] . Note that the integral is over _z_ _[′]_ this expression and doesn’t depend
on the realization _z_ . It still depends on _s_, however.


We prove this function is convex as follows. Let _α ∈_ (0 _,_ 1), and consider functions _s_ 1 _, s_ 2. Then


��       log exp         - _f_ ( _x_ ) _[⊤]_ [ _αs_ 1( _z_ _[′]_ ) + (1 _−_ _α_ ) _s_ 2( _z_ _[′]_ )]� _dh_ ( _z_ _[′]_ )


��       = log exp        - _αf_ ( _x_ ) _[⊤]_ _s_ 1( _z_ _[′]_ )� exp �(1 _−_ _α_ ) _f_ ( _x_ ) _[⊤]_ _s_ 2( _z_ _[′]_ )� _dh_ ( _z_ _[′]_ )


= log ���exp        - _f_ ( _x_ ) _[⊤]_ _s_ 1( _z_ _[′]_ )�� _α_ �exp        - _f_ ( _x_ ) _[⊤]_ _s_ 2( _z_ _[′]_ )��1 _−α dh_ ( _z′_ )�


��       = log [ _u_ ( _z_ _[′]_ )] _[α]_ [ _v_ ( _z_ _[′]_ )] [1] _[−][α]_ _dh_ ( _z_ _[′]_ )


where we have defined _u_ ( _z_ _[′]_ ) = exp( _f_ ( _x_ ) _[⊤]_ _s_ 1( _z_ _[′]_ )) and _v_ ( _z_ _[′]_ ) = exp( _f_ ( _x_ ) _[⊤]_ _s_ 2( _z_ _[′]_ )). Consider the
integral above, and apply Hölder’s inequality with 1 _/p_ = _α_ and 1 _/q_ = 1 _−_ _α_ . These sum to one as
required. We take _f_ to be [ _u_ ( _z_ _[′]_ )] _[α]_ and _g_ to be [ _v_ ( _z_ _[′]_ )] [(1] _[−][α]_ [)] .


Continuing, we have


17


��    = log [ _u_ ( _z_ _[′]_ )] _[α]_ _·_ [ _v_ ( _z_ _[′]_ )] [(1] _[−][α]_ [)] _dh_ ( _z_ _[′]_ ) [this line repeated for clarity]


[Hölder]


_≤_ log


���� - _α_ ��� �1 _−α_ [�]

[ _u_ ( _z_ _[′]_ )] _[α]_ [�][1] _[/α]_ _dh_ ( _z_ _[′]_ ) _·_ [ _v_ ( _z_ _[′]_ )] [(1] _[−][α]_ [)][�][1] _[/]_ [(1] _[−][α]_ [)] _dh_ ( _z_ _[′]_ )


��     - ��     = _α_ log _u_ ( _z_ _[′]_ ) _dh_ ( _z_ _[′]_ ) + (1 _−_ _α_ ) log _v_ ( _z_ _[′]_ ) _dh_ ( _z_ _[′]_ )


��     - ��     = _α_ log exp( _f_ ( _x_ ) _[⊤]_ _s_ 1( _z_ _[′]_ )) _dh_ ( _z_ _[′]_ ) + (1 _−_ _α_ ) log exp( _f_ ( _x_ ) _[⊤]_ _s_ 2( _z_ _[′]_ )) _dh_ ( _z_ _[′]_ ) _._


This was all that is required to show that the mapping _s �→_ log �� exp - _f_ ( _x_ ) _[⊤]_ _s_ ( _z_ _[′]_ )� _dz_ _[′]_ [�] is convex.
As the sum of two convex functions is convex, we’ve shown convexity of the integrand for any fixed
draw _z, x_ _∼_ _p_ ( _z, x_ ). To conclude, we observe that by linearity of integration, this holds for the
integral as well. The argument for _f_ is identical by symmetry of the inner product.


C SAMPLING


Similar to EigenVI (Cai et al., 2024), LBF-NPE does not easily admit sampling from the fitted
variational density. This is one limitation of the nonparametric nature of the density model in both of
these approaches to variational inference.


In the low-dimensional case, sampling can be performed by inverse transform sampling. In this
approach, one uses the cumulative distribution function _Q_ of _q_, defined as


                    - _z_ _[∗]_
_Q_ ( _z_ _[∗]_ ) = _Pq_ ( _z_ _< z_ _[∗]_ ) = _q_ ( _z_ ) _dz_

_−∞_


where _q_ ( _z_ ) is the fitted variational density (say, conditional on some datum _x_ of interest). The
function _Q_ is invertible. Sampling can be performed by drawing _U_ _∼_ Unif[0 _,_ 1], and thereafter
computing
_Z_ = _Q_ _[−]_ [1] ( _U_ ) _._


The result of this procedure is a draw _Z_ _∼_ _q_ ( _z_ ). Computing and inverting cumulative distribution functions in low dimensions is fairly straightforward. As outlined in the main text, this
low-dimensional setting is one we commonly find to be of use to practitioners, especially for the
types of problems we consider in our experiments.


To sample from high-dimensional posteriors, several different approaches are available. One approach,
as outlined in (Cai et al., 2024), is sequential sampling, whereby one samples


_q_ ( _z_ 1) _, q_ ( _z_ 2 _| z_ 1) _, q_ ( _z_ 3 _| z_ 1 _, z_ 2) _, . . ., q_ ( _zd_ _| z_ 1 _, . . ., zd−_ 1)


in order. Each individual density above can be sampled using the inverse transform sampling approach
outlined above; conditioning within the exponential family parameterization is easily accomplished
by freezing the already sampled indices and changing the variables of integration in the log integral.
More generally, one could also utilize rejection sampling or other Monte Carlo sampling algorithms.
As the unnormalized variational density has an extremely simple form, i.e., _q_ ( _z_ ) _∝_ exp( _η_ _[⊤]_ _b_ ( _z_ )),
Markov chain Monte Carlo algorithms could also be an efficient way to sample from the distribution
defined by the fitted density. We present some selected results of Langevin sampling and inverse
transform sampling in Appendix C.1 to illustrate the utility of these approaches.


C.1 RESULTS OF LANGEVIN DYNAMICS AND INVERSE TRANSFORM SAMPLING


We present the sampling results obtained via Langevin dynamics (Song et al., 2021) and inverse
transform sampling for the three 2D case studies: bands, ring, and spiral. In addition to
the inverse transform sampling described in the preceding section, we explore the use of Langevin


18


dynamics, a widely adopted method for generating samples in score-based generative models. This
approach iteratively updates a set of particles according to:


_√_
_dzt_ = _ϵ∇z_ log _p_ ( _zt_ _| x_ ) _dt_ +


2 _ϵdWt,_


where _ϵ_ = 0 _._ 001, _dWt_ _∼N_ (0 _,_ 1), and _t_ _∈{_ 1 _,_ 2 _, . . .,_ 1000 _}_ . Since our model provides a differentiable approximation of log _p_ ( _zt_ _|_ _x_ ) through _fϕ_ ( _x_ ) _[⊤]_ _sψ_ ( _z_ ), the gradient _∇z_ log _p_ ( _zt_ _|_ _x_ ) can be
directly estimated. This enables us to apply Langevin dynamics for posterior sampling.


For each of the bands, ring, and spiral case studies, we generate 10,000 samples and visualize
both the samples and their marginal densities in Figures 5 to 10. As illustrated in these figures, both
Langevin dynamics and inverse transform sampling yield samples that closely match the estimated
posterior distributions.


Figure 5: Sampling results for bands. “LD” and
“Inv” denote Langevin dynamics and inverse transform sampling, respectively.


Figure 7: Sampling results for ring. “LD” and
“Inv” denote Langevin dynamics and inverse transform sampling, respectively.


Figure 6: Marginal density of samples for bands.
The green line indicates the estimated posterior
density.


Figure 8: Marginal density of samples for ring.
The green line indicates the estimated posterior
density.


19


Figure 9: Sampling results for spiral. “LD”
and “Inv” denote Langevin dynamics and inverse
transform sampling, respectively.


D EXPERIMENTAL DETAILS


Figure 10: Marginal density of samples for
spiral. The green line indicates the estimated
posterior density.


Code to reproduce results is provided at [https://github.com/YicunDuanUMich/](https://github.com/YicunDuanUMich/LBF-NPE)
[LBF-NPE.](https://github.com/YicunDuanUMich/LBF-NPE) We use PyTorch (Paszke et al., 2019; Ansel et al., 2024) and JAX (Bradbury et al.,
2018). We also use the equinox library for deep learning in JAX (Kidger & Garcia, 2021). For
Section 6.4, the DC2 Simulated Sky Survey data is publicly available from the LSST Dark Energy
Science Collaboration (LSST DESC) (LSST Dark Energy Science Collaboration et al., 2021; 2022).
All experiments are conducted on an Ubuntu 22.04 server equipped with an NVIDIA RTX 2080 Ti
GPU.


D.1 TOY EXAMPLE: SINUSOIDAL LIKELIHOOD


In Section 6.1, we design a simple Bayesian model to evaluate the convergence behavior of LBF-NPE
and MDN. This statistical model is

_z_ _∼_ Unif[0 _,_ 2 _π_ ] _,_

_x | z_ _∼N_ (sin(2 _z_ ) _, σ_ [2] ) _,_


where _σ_ [2] = 1. This model induces a multimodal posterior:
_P_ ( _z_ _| x_ ) _∝_ exp       - _−_ (sin(2 _z_ ) _−_ _x_ ) [2] _/_ (2 _σ_ [2] )� _×_ I(0 _≤_ _z_ _≤_ 2 _π_ ) _,_

which exhibits two modes when _x ≥_ 1 or _x ≤−_ 1, and four modes otherwise.


For LBF-NPE, we construct a multilayer perceptron (MLP) to predict the coefficient vector _η_ = _fϕ_ ( _x_ ),
and the sufficient statistics _b_ ( _z_ ) _∈_ R _[K]_ are computed using B-spline basis functions with _K_ = 14. The
MLP architecture consists of an input layer, four hidden layers, and an output layer, mapping _x ∈_ R
to _η_ _∈_ R _[K]_ . Each hidden layer includes a full connection layer of 128 units, layer normalization, and a
ReLU activation. The output layer is linear. The B-spline basis comprises 14 degree-2 basis functions,
with knots at [0 _,_ 0 _,_ linspace(0 _,_ 2 _π,_ num = _K_ ) _,_ 2 _π,_ 2 _π_ ]. Although the B-spline basis _b_ ( _z_ ) can be
evaluated recursively as described in Section A.1, we precompute it on a grid to avoid redundant
computation during training. Specifically, we pick 1000 uniformly spaced points in the interval [0 _,_ 2 _π_ ]
and approximate the integral - exp( _ηi_ _[⊤][b]_ [(] _[z]_ [))] _[ dz]_ [ using the trapezoidal sum.] [For the term] _[ f][ϕ]_ [(] _[x]_ [)] _[⊤][b]_ [(] _[z]_ [)][,]
we use the basis vector corresponding to the grid point closest to the true latent variable _z_ true.


For MDN, we use the same MLP architecture, with the output adapted to represent the parameters of
a mixture of 5 Gaussian distributions. The output vector has 10 parameters for means and variances,
and 4 additional parameters for the mixture weights. The MDN objective is
_L_ MDN( _γ_ ) = _−_ E _p_ ( _x,z_ ) log _q_ ( _z_ ; _tγ_ ( _x_ )) _,_ (17)


20


where _γ_ are the neural network parameters, _tγ_ ( _x_ ) denotes the predicted distribution parameters, and
_q_ ( _z_ ; _·_ ) is the corresponding density.


The training procedures for LBF-NPE and MDN are identical, except for the loss function. At
each step, we sample 1024 latent–observed pairs ( _z, x_ ) from the generative model and update model
parameters using the AdaBelief optimizer (Zhuang et al., 2020) with a learning rate of 0.001. Training
proceeds for 50,000 steps and completes within one hour for both methods. Peak GPU memory usage
is approximately 8300MB. We hold out 1000 ( _z, x_ ) pairs and track their negative log-likelihood
(NLL) over training, as shown in Figure 1. We apply Gaussian smoothing to the NLL curves with
standard deviation _σ_ = 20. This results in a smoothing kernel of size 161 = 4 _×_ 20 _×_ 2 + 1, with
weights given by _Gi_ = exp( _−i_ [2] _/_ (2 _σ_ [2] )) for _i ∈{−_ 80 _, . . .,_ 80 _}_ . With the normalization constants
for _Gi_ omitted, the smoothed NLL at step _j_ is computed as


NLLsmooth _,j_ =


D.2 2D CASE STUDIES: BANDS


80

- NLL _j_ + _i · Gi._

_i_ = _−_ 80


The statistical model for the bands test case, as introduced in Section 6.2, is


_z_ 1 _, z_ 2 _∼_ Unif[ _−_ 1 _,_ 1] _,_
_z_ = ( _z_ 1 _, z_ 2) _,_

_x | z_ _∼N_ ( _|z_ 1 _−_ _z_ 2 _|, σ_ [2] ) _,_


where _σ_ [2] = 10 _[−]_ [2] . The resulting posterior forms two elongated bands in the 2D latent space
_P_ ( _z_ _| x_ ) _∝_ exp - _−_ ( _|z_ 1 _−_ _z_ 2 _| −_ _x_ ) [2] _/_ (2 _σ_ [2] )� _·_ I( _−_ 1 _≤_ _z_ 1 _, z_ 2 _≤_ 1), with its maxima occurring along
the lines where _|z_ 1 _−_ _z_ 2 _|_ = _x_ .


As the latent variable _z_ is now two-dimensional, LBF-NPE encounters increased complexity due
to the larger number of basis functions required. In our LBF-NPE framework, both the coefficient
network _fϕ_ and the sufficient statistic network _sψ_ are implemented as multilayer perceptrons (MLPs)
with four hidden layers, each containing 128 units. All layers use layer normalization to stabilize
optimization and ReLU activations. The network _fϕ_ maps input _x_ _∈_ R to a coefficient vector in
R _[K]_, while _sψ_ maps _z_ _∈_ R [2] to sufficient statistics in R _[K]_ . We set _K_ = 20 for consistency with
other 2D case studies, though even _K_ = 2 suffices to capture the posterior structure in this example
(see Appendix E.3). The loss function for LBF-NPE follows Algorithm 1, where the integral term

- exp( _ηi_ _[⊤][s][ψ]_ [(] _[z]_ [))] _[ dz]_ [ is approximated using a trapezoidal sum over a][ 100] _[×]_ [100][ uniform grid spanning]

[ _−_ 1 _,_ 1] [2] . During training, we alternate between updating _fϕ_ and _sψ_ : we train _fϕ_ for 1000 steps
while holding _sψ_ fixed, then train _sψ_ for 1000 steps with _fϕ_ fixed, and repeat this process until the
total training budget is exhausted. The choice of 1000 steps per phase is empirical; we observe
diminishing returns in the loss reduction beyond 1000 steps, indicating that each sub-network has
reached a near-optimal solution given the other is fixed. In addition, we use stereographic projection
to reparameterize the output of _fϕ_ and _sψ_ .


For the MDN baseline, we use an MLP with the same architecture as _fϕ_, except that it outputs a
50-dimensional vector representing the parameters of a mixture of 10 Gaussian components. Each
component is parameterized by five values: two for the mean, two for the (diagonal) variance
(assuming zero covariance), and one for the mixture weight. The loss function is identical to that
described in Appendix D.1.


For the normalizing flow baseline, we adapt the classic coupling flow from (Dinh et al., 2017) to
model the conditional posterior _p_ ( _z_ _|_ _x_ ). Each coupling layer includes translation and scaling
subnetworks conditioned on _x_ _∈_ R. These sub-networks are implemented as MLPs, each taking
as input the concatenation of the masked latent variable _z_ and the conditioning variable _x_ . Each
MLP consists of a single hidden layer with 128 units. We use 10 coupling layers to ensure sufficient
expressiveness. The resulting conditional density is given by:


_q_ ( _z_ _| x_ ) = _qN_ ( _hν_ ( _z_ ; _x_ )) _· |_ det _J|,_


where _qN_ ( _·_ ) denotes the standard Gaussian density, _hν_ ( _z_ ; _x_ ) is the transformed latent variable via
the flow, and det _J_ is the product of the Jacobian determinants from each flow transformation.


21


We train LBF-NPE, MDN, and the normalizing flow using the AdamW optimizer (Loshchilov &
Hutter, 2019) with a learning rate of 10 _[−]_ [5] for 50,000 steps. The batch size is set to 1024, and
training completes in approximately 2 hours. Maximum GPU memory usage is around 8400MB. For
evaluation, we use a held-out set of 1000 ( _z, x_ ) pairs to compute the average forward and reverse
KL divergences. For each test observation _x_, LBF-NPE, MDN, and the normalizing flow estimate
the density _q_ ( _z_ _|_ _x_ ) over a 100 _×_ 100 uniform grid on [ _−_ 1 _,_ 1] [2] . These estimated posteriors are
normalized such that their integral over the grid equals 1. The true posterior _p_ ( _z_ _| x_ ) is computed
analytically over the same grid, enabling pointwise comparison. We then calculate the forward and
reverse KL divergences between the estimated and true posteriors and report the average over all
1000 test cases in Table 1. For the illustrative posterior plots shown in Figure 2, we fix _x_ = 0 _._ 7 and
visualize the estimated density _q_ ( _z_ _| x_ ) from each method over the same 100 _×_ 100 grid.


D.3 2D CASE STUDIES: RING


The statistical model for the ring case study in Section 6.2 is defined as:


_z_ 1 _, z_ 2 _∼_ Unif[ _−_ 1 _,_ 1] _,_
_z_ = ( _z_ 1 _, z_ 2) _,_

_x | z_ _∼N_ ( _∥z∥_ [2] _, σ_ [2] ) _,_


where _σ_ [2] = 10 _[−]_ [2] . The resulting posterior, _P_ ( _z_ _|_ _x_ ) _∝_ exp - _−_ ( _∥z∥_ [2] _−_ _x_ ) [2] _/_ (2 _σ_ [2] )� _·_ I( _−_ 1 _≤_
_z_ 1 _, z_ 2 _≤_ 1), forms a ring-shaped distribution in the latent space, with radius approximately _[√]_ _x_ .


The network architectures and training configurations used in this case are identical to those described
in Appendix D.2. An example posterior _q_ ( _z_ _| x_ = 0 _._ 7) is visualized in Figure 2.


D.4 2D CASE STUDIES: SPIRAL


The spiral model is defined as follows:


_b ∼_ Unif[0 _._ 1 _,_ 0 _._ 5]
_d ∼_ Unif[0 _._ 0 _, sb_ (2 _π_ )]

_θ_ = _s_ _[−]_ _b_ [1][(] _[d]_ [)]
_r_ = _bθ_

_z_ 1 = _r_ cos( _θ_ )
_z_ 2 = _r_ sin( _θ_ )
_z_ = ( _z_ 1 _, z_ 2)

_x | z_ _∼N_ ( _b, σ_ [2] )


_√_
where _σ_ [2] = 10 _[−]_ [4], and _sb_ ( _θ_ ) = 2 _b_ [(] _[θ]_


where _σ_ [2] = 10 _[−]_ [4], and _sb_ ( _θ_ ) = 2 _b_ [(] _[θ]_ 1 + _θ_ [2] + sinh _[−]_ [1] ( _θ_ )). The posterior is _P_ ( _z_ _|_ _x_ ) _∝_

exp - _−_ ( _θ_ _[r]_ _[−]_ _[x]_ [)][2] _[/]_ [(2] _[σ]_ [2][)] - _·_ I(0 _≤_ _θ_ _≤_ 2 _π,_ 0 _._ 1 _θ_ _≤_ _r_ _≤_ 0 _._ 5 _θ_ ).


Most training settings follow those in Appendix D.2, except that we increase the number of coupling
layers in normalizing flow to 16. We observe minimal performance gain beyond this depth. For
visualization in Figure 2, we display the estimated posterior _q_ ( _z_ _| x_ = 0 _._ 35) over the 100 _×_ 100 grid.


D.5 OBJECT DETECTION


We define the image generative model as follows:


_l_ 1 _, l_ 2 _∼_ Unif([0 _,_ 16] _×_ [0 _,_ 16]) _,_

_f_ 1 _, f_ 2 _∼N_ ( _µ, σ_ [2] ) _,_
_I_ = Image( _{l_ 1 _, l_ 2 _}, {f_ 1 _, f_ 2 _},_ PSF) _,_
_xj,k_ _∼_ Poisson( _Ij,k_ ) _,_


22


_θ_ _[r]_ _[−]_ _[x]_ [)][2] _[/]_ [(2] _[σ]_ [2][)] - _·_ I(0 _≤_ _θ_ _≤_ 2 _π,_ 0 _._ 1 _θ_ _≤_ _r_ _≤_ 0 _._ 5 _θ_ ).


where _µ_ = 2000, _σ_ [2] = 400 [2], and Image( _·_ ) and PSF( _·_ ) are defined below. Note that in our
implementation, flux values are constrained to be positive.


**Algorithm 2:** Image
**Inputs:** list of source locations _{l_ 1 _, l_ 2 _}_ ; list
of source fluxes _{f_ 1 _, f_ 2 _}_ ;
point-spread function PSF.
Initialize pixel location matrix _pl_
**for** _li, fi_ in zip( _{l_ 1 _, l_ 2 _}, {f_ 1 _, f_ 2 _}_ ) **do**

Compute relative location _rli_ = _pl −_ _li_
Compute PSF density _di_ = PSF( _rli_ )
Compute _Ii_ = _fi × di_
**end**
Compute _I_ = _I_ 1 + _I_ 2
Return _I_ .


**Algorithm 3:** PSF
**Inputs:** relative position matrix _rli_ .
Compute
_di_ = _−_ ( _rli_ [ _. . .,_ 0] [2] + _rli_ [ _. . .,_ 1] [2] ) _/_ (2 _σ_ PSF [2] [)]


Compute _di_ = exp( _di_ ) _/_ sum(exp( _di_ ))
Return _di_ .


The pixel location matrix _pl_ is a mesh grid of shape ( _H, W,_ 2) defined over [0 _._ 5 _,_ 1 _._ 5 _, . . ., H −_ 0 _._ 5] _×_

[0 _._ 5 _,_ 1 _._ 5 _, . . ., W_ _−_ 0 _._ 5], where _H_ and _W_ are the height and width of the image, respectively. Each
source location _li_ is a 2D vector, and the term _rli_ [ _. . .,_ 0] [2] + _rli_ [ _. . .,_ 1] [2] is a matrix of shape ( _H, W_ ).
We use _σ_ PSF [2] [=] [1][.] [Each source flux] _[f][i]_ [is a scalar.] [Before passing the image to the network,] [we]
normalize it using min-max scaling: _x_ = ( _x −_ min( _x_ )) _/_ (max( _x_ ) _−_ min( _x_ )).


Since the input is a 16 _×_ 16 image, we employ a convolutional layer in our network to reduce
computational cost. The first layer of the model, _fϕ_, is a 2D convolutional layer with a kernel size of
4, increasing the channel dimension from 1 to 3. This is followed by a 2D max pooling layer and a
ReLU activation. The output is then flattened and passed through four MLP layers, each with 128
hidden units, layer normalization, and ReLU activations. Another model, _sψ_, is an MLP with four
hidden layers, each with 128 units, layer normalization, and ReLU activations. The outputs of _fϕ_ and
_sψ_ are reparameterized via stereographic projection.


As each image contains two astronomical sources, we compute the loss separately for each source.
For the source located at _li_, the first term in the loss is _−wf_ [ˆ] _ϕ_ ( _x_ ) _[⊤]_ _s_ ˆ _ψ_ ( _li_ ). Only _s_ ˆ _ψ_ ( _li_ ) needs to be
evaluated per source; shared terms such as _f_ [ˆ] _ϕ_ ( _x_ ) and the integral term can be reused across both. We
approximate the integral using Monte Carlo integration with 22,500 random samples. The final loss is
the sum of the losses for both sources. For alternating optimization, we train one of _fϕ_ or _sψ_ for 300
steps at a time (shorter than the 1000-step updates used in the 2D case studies; see Appendix D.2)
since convergence is typically achieved more rapidly in this setting. Optimization is performed using
the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 0.001. The total number
of training steps is 45,000, with overall training time under two hours.


For posterior visualization, we adopt the same procedure as in the previous 2D case studies (see
Appendix D.2) but evaluate the posterior over a 200 _×_ 200 grid on the domain [0 _,_ 16] [2] . The estimated
posterior for a certain image is shown in Figure 3. To generate this posterior, we leverage a model
trained with _K_ = 64. For results with other values of _K_ (e.g., 9, 20, 36), we provide further
discussion in Appendix E.3.


D.6 REDSHIFT ESTIMATION


Our redshift experiment extends the methodology of the Bayesian Light Source Separator (BLISS)
(Liu et al., 2023; Hansen et al., 2022; Patel et al., 2025). For a given generative model of astronomical
images and latent quantities (locations; fluxes; type of object; redshift; etc.), BLISS utilizes neural
posterior estimation (Papamakarios & Murray, 2016) to perform amortized variational inference. The
network architecture for BLISS operates on _tiles_ of images, returning distributional parameters for
each object detected per tile. The architecture is thus convolutional, except for several additional
image normalizations and other design choices suitable for astronomical image processing.


For samples of the generative model, we use images from two tracts of the LSST DESC DC2
Simulated Sky Survey (LSST Dark Energy Science Collaboration et al., 2021; 2022), numbers 3828


23


and 3829. LBF-NPE does not sample the generative model on-the-fly in this setting, but only has
access to a finite number of draws from the training sets.


We use the BLISS preprocessing routines to produce training, validation, and test image sets, along
with ground-truth catalogs. Images, each of size 80 _×_ 80, are processed in batches of 64 by the
BLISS inference network, which further splits these into 4 _×_ 4-pixel tiles. The network is fit to the
training set to minimize the forward KL divergence using a learning rate of 0.001. All nuisance latent
variables are marginalized over, and we only score redshift variational posteriors; BLISS also allows
easy addition of posteriors on other latent quantities to the computed NLL loss, should the user desire
to perform inference on them.


We adapt the neural network architecture from BLISS for redshift estimation. The complete architectural details and parameter configurations are provided in Table 3. As shown in the table, the input
and output shapes of each layer are expressed as tuples, e.g., (bands, h, w) or (64, h, w),
where bands denotes the number of bands in the input astronomical images. In the DC2 dataset,
there are six bands: _u, g, r, i, z, y_ . The variables h and w represent the image’s height and width,
respectively, and are both set to 80 in our experiments. The model is composed primarily of three
types of layers: Conv2DBlock, C3Block, and Upsample. A Conv2DBlock is a composite
module consisting of a 2D convolution, group normalization, and a SiLU activation function. The
C3Block is adapted from the YOLOv5 architecture (Jocher et al., 2020). It comprises three convolutional layers with kernel size 1 and includes skip connections implemented via multiple bottleneck
blocks (parameterized by n). The Upsample layer performs spatial upscaling of the input tensor
by a specified factor. The architecture follows a U-shaped design with four downsampling steps
followed by two upsampling steps. To denote skip connections and input dependencies between
layers, we use the “Input From” column. For example, the entry “[-1, 9]” indicates that the current
layer takes as input the concatenation of the outputs from the previous layer and layer 9. The final
layer is a convolutional module with kernel size 1, producing an output of shape (n_coeff, h/4,
w/4), where n_coeff is the number of predicted coefficients per tile.


The forward KL divergence framework prescribes that predictions are only scored for true objects.
Accordingly, for each ground-truth redshift in the training set, we score the predicted NLL computed
from the variational distribution for the 4x4 pixel tile containing that object. BLISS makes this
transdimensional inference problem (a result of the number of objects per tile being unknown _a_
_priori_ ) tractable by sharing parameters across objects within the same 4 _×_ 4-pixel tile, at the cost
of bias introduced by this approximation. For both the MDN and B-spline parameterizations, we fit
to the training data for 30 epochs and use the model weights with the lowest held-out NLL on the
validation set to compute the test-set NLL. Training the inference network _fϕ_ takes approximately
12 hours on a single NVIDIA GeForce RTX 2080 Ti GPU. We note that, due to the approximations
involved in using a finite training set rather than true “simulated” draws, we can easily overfit to the
training and validation sets. The procedure outlined above aims to mitigate these issues to the extent
possible.


24


**Layer #** **Input From** **Input Shape** **Layer Type** **Config** **Output Shape**


in_ch=bands; out_ch=64;
1 -1 (bands, h, w) Conv2DBlock (64, h, w)

kernel_size=5


in_ch=64; out_ch=64;
2 -1 (64, h, w) Conv2DBlock (64, h, w)

kernel_size=5


Sequence of
3 -1 (64, h, w)

Conv2DBlock


in_ch=64; out_ch=64;

(64, h, w)
kernel_size=5; sequence_len=3


in_ch=64; out_ch=64;
4 -1 (64, h, w) Conv2DBlock (64, h/2, w/2)

kernel_size=3; stride=2


in_ch=64; out_ch=64;
5 -1 (64, h/2, w/2) C3Block (64, h/2, w/2)

n=3


in_ch=64; out_ch=128;
6 -1 (64, h/2, w/2) Conv2DBlock (128, h/4, w/4)

kernel_size=3; stride=2


in_ch=128; out_ch=128;
7 -1 (128, h/4, w/4) C3Block (128, h/4, w/4)

n=3


in_ch=128; out_ch=256;
8 -1 (128, h/4, w/4) Conv2DBlock (256, h/8, w/8)

kernel_size=3; stride=2


in_ch=256; out_ch=256;
9 -1 (256, h/8, w/8) C3Block (256, h/8, w/8)

n=3


in_ch=256; out_ch=512;
10 -1 (256, h/8, w/8) Conv2DBlock (512, h/16, w/16)

kernel_size=3; stride=2


in_ch=512; out_ch=256;
11 -1 (512, h/16, w/16) C3Block (256, h/16, w/16)

n=3


scale=2;
12 -1 (256, h/16, w/16) Upsample (256, h/8, w/8)

mode="nearest"


in_ch=512; out_ch=256;
13 [-1, 9] (512, h/8, w/8) C3Block (256, h/8, w/8)

n=3


scale=2;
14 -1 (256, h/8, w/8) Upsample (256, h/4, w/4)

mode="nearest"


in_ch=384; out_ch=256;
15 [-1, 6] (384, h/4, w/4) C3Block (256, h/4, w/4)

n=3


in_ch=256; out_ch=n_coeff;
16 -1 (256, h/4, w/4) Conv2D (n_coeff, h/4, w/4)

kernel_size=1


Table 3: Neural network architecture for redshift estimation.


25


D.7 ANGULAR DISTANCE OPTIMIZATION


As discussed in Section 4.4, our method can be interpreted as performing angular-distance optimization, but with a loss and gradient derived from a probabilistic space. This interpretation
becomes evident if we decouple the magnitude and directional components of the output tensors
_fϕ_ ( _x_ ) _, sψ_ ( _z_ ) _∈_ R _[K]_ through normalization techniques such as L2 normalization or stereographic projection reparameterization. Angular distance optimization is a common objective in modern machine
learning pipelines, improving performance and ensuring consistent alignment between training and
testing metrics. Several widely-used loss functions, including the triplet loss (Hoffer & Ailon, 2015),
N-pair loss (Sohn, 2016), and multi-similarity loss (Wang et al., 2019), incorporate angular distance
in their formulation. Cosine-based softmax loss functions are widely used for face recognition (Liu
et al., 2017; Wang et al., 2018; Deng et al., 2019), and many contrastive learning algorithms (Chen
et al., 2020; Tian et al., 2020; Ye et al., 2019) employ angular objectives to maximize the cosine
similarity between embeddings from positive pairs.


Our variational objective in Equation (6) is related to cosine-based softmax loss, whose general form
is








 _,_ (18)


_L_ = _−wSi,yi_ + log


exp( _wSi,yi_ ) + 


exp( _wSi,j_ )

_j_ = _yi_


as described in Section 4.4, and suggesting a more general form for our variational objective, namely


_L_ ˆLBF-NPE( _ϕ, ψ_ ) = E _p_ ( _z,x_ )


- �� - - ��
_−wf_ [ˆ] _ϕ_ ( _x_ ) _[⊤]_ _s_ ˆ _ψ_ ( _z_ ) + log exp _wf_ [ˆ] _ϕ_ ( _x_ ) _[⊤]_ _s_ ˆ _ψ_ ( _z_ _[′]_ ) _dz_ _[′]_ _,_ (19)


where again _f_ [ˆ] _ϕ_ ( _·_ ) and ˆ _sψ_ ( _·_ ) are normalized outputs of neural networks (i.e., with unit norm) and _w_
is a scaling factor.


The key differences from the cosine-based softmax formulation are: (1) the summation is replaced by
an integral over the continuous latent space, and (2) the angular distance is computed between coefficient vectors and basis functions, rather than between learned embeddings. This connection offers
two main advantages. First, our theoretical guarantees on convexity and convergence may extend to
angular distance optimization problems, suggesting broader applicability. Second, our method can
leverage off-the-shelf improvements developed for angular optimization, such as SEC (Zhang et al.,
2020), which regularizes gradient updates to stabilize and accelerate training. Given that even simple
stereographic normalization already yields strong empirical results, we leave the integration of these
enhancements to future work.


In our experiments, we use stereographic reparameterization to project the output tensor onto the unit
hypersphere. Precisely, _u ∈_ R _[K][−]_ [1] is transformed to R _[K]_ via


  - 2 _u_
_y_ = [1] _[ −∥][u][∥]_ [2]
1 + _∥u∥_ [2] _[,]_ 1 + _∥u∥_ [2]


_,_ (20)


ensuring that _∥y∥_ = 1. This projection serves as a smooth and bijective transformation from Euclidean
space R _[K][−]_ [1] onto the _K_ _−_ 1-sphere _S_ _[K][−]_ [1] = _{x ∈_ R _[K]_ : _∥x∥_ = 1 _}_ . Although this transformation
changes the form of the variational objective in the neural network outputs, and thus violates some
assumptions of our NTK framing from the perspective of convexity, strong empirical results suggest
the benefits of reparameterization, and also the importance of future work in understanding the
success of neural posterior estimation (NPE) techniques under arbitrary parameterizations. We
hypothesize that the advantageous properties of this normalization stem from the smooth gradient
trajectories and mapping to a compact space, discussed in more detail below.


We illustrate the stereographic normalization process in a 2D case, as shown in Figure 11. In this
setting, a scalar input _u ∈_ R [1] is projected onto a vector _y_ _∈_ R [2] lying on the 1-sphere (i.e., the unit
circle). For any given _u_, there exists a unique line connecting the point ( _u,_ 0) and the north pole
_N_ = (0 _,_ 1). This line intersects the 1-sphere at a single point, which serves as the projection of _u_ . By
drawing a vector from the origin to this intersection point, we obtain a unit vector _y_ on the 1-sphere.
Notably, the location of the intersection reflects the magnitude of _u_ : if _∥u∥_ _>_ 1, the intersection lies
on the upper half of the circle; if _∥u∥_ _<_ 1, it falls on the lower half.


26


Figure 11: Visualization of stereographic projection in 2D. A scalar _u_ is mapped to a point on the
unit circle via intersection of the line connecting ( _u,_ 0) and the north pole _N_ = (0 _,_ 1).


This reparameterization offers several advantages. First, the stereographic projection is differentiable
everywhere and provides well-behaved gradients throughout the domain. Second, the projection
naturally enforces unit-norm constraints without requiring additional normalization layers or manual
clipping, thus making training more stable and efficient.


E ADDITIONAL EXPERIMENTAL RESULTS


E.1 EFFECT OF NORMALIZATION


We compare the neural network’s output basis functions with and without stereographic projection
normalization, demonstrating that normalization helps the network learn clearer boundaries between
regions of the parameter space. Figure 12 and Figure 13 show the values of the 20-dimensional basis
functions (i.e., [ _sψ_ ( _z_ )] _i_, where _i ∈{_ 1 _,_ 2 _, . . .,_ 20 _}_ ) evaluated over the plane _z_ _∈_ [ _−_ 3 _,_ 3] _×_ [ _−_ 3 _,_ 3] for
the spiral case study. It is evident that the model with normalization exhibits more distinguishable and
structured partition boundaries in _z_ -space, while the model without normalization suffers from blurry
transitions and over-exposure artifacts, as seen in Figure 13. This highlights a key drawback of the
non-normalized approach: it struggles to effectively disentangle the parameter space. Normalization
also enhances interpretability by promoting better separation among basis functions. The estimated
posterior density is expressed as a weighted linear combination of these basis function densities. For
a given target spiral, the neural network increases the weights (i.e., [ _fϕ_ ( _x_ )] _i_ ) for dimensions whose
corresponding basis functions have high overlap with the target density, and decreases weights for
dimensions with low overlap.


27


Figure 12: Density plot of 20-dim basis function (w/ stereographic projection normalization) over
plane _z_ . Each subplot represents the density plot of a certain dimension.


Figure 13: Density plot of 20-dim basis function (w/o stereographic projection normalization) over
plane _z_ . Each subplot represents the density plot of a certain dimension.


28


E.2 POSTERIOR COMPARISON


Figure 14: Visual comparison of posterior densities estimated by different methods (LBF-NPE,
Normalizing Flow, MDN) against ground truth for ten representative observations.


29


Figure 15: Visual comparison of posterior densities estimated by different methods (LBF-NPE,
Normalizing Flow, MDN) against ground truth for ten representative observations.


30


Figure 16: Visual comparison of posterior densities estimated by different methods (LBF-NPE,
Normalizing Flow, MDN) against ground truth for ten representative observations.


31


E.3 DIMENSIONS OF BASIS FUNCTIONS & FLEXIBILITY


The flexibility of our method is positively correlated with the dimensionality of the basis functions. We
demonstrate this by analyzing both the forward and reverse KL divergences, as well as basis-function
density plots, for models using basis functions of varying dimensions in the object detection case
study. To quantify this, we compute the forward and reverse KL divergence between the estimated
posterior distribution and a target mixture of Gaussians defined by the true underlying locations _l_ 1 _, l_ 2:


0 _._ 5 _N_ ( _l_ 1 _,_ 0 _._ 1 [2] ) + 0 _._ 5 _N_ ( _l_ 2 _,_ 0 _._ 1 [2] ) _._ (21)


The results, shown in Table 4, indicate that both forward and reverse KL divergences decrease as
the dimensionality of the basis functions increases. For example, the 64-dimensional basis functions
achieve the lowest forward KL divergence (1.524), while the 9-dimensional basis functions result in
the highest (3.397). A similar trend holds for reverse KL divergence. However, the marginal gain
from increasing dimensionality diminishes as the number of basis functions grows. Increasing from 9
to 20 dimensions yields a significant improvement in forward/reverse KL divergence (1.187/0.511),
but the improvement from 36 to 64 dimensions is relatively minor (0.246/0.287). This suggests that,
for a complex task like object detection, a basis function dimensionality under 100 is sufficient to
achieve near-optimal performance.


**9-dim** **20-dim** **36-dim** **64-dim**


**Forward KL Divergence** 3.397 2.210 1.770 **1.524**
∆ **Forward KL Divergence**      - **-1.187** -0.440 -0.246


**Reverse KL Divergence** 2.380 1.869 1.360 **1.073**
∆ **Reverse KL Divergence**      - **-0.511** -0.509 -0.287


Table 4: Object detection: forward/reverse KL divergence for models of different basis function
dimensions.


The basis function density plots provide further intuition for this trend. As shown in Figures 17
to 20, the 64- and 36-dimensional basis functions can partition the _z_ -space into fine-grained regions,
capturing detailed structure. In contrast, 20- and 9-dimensional basis functions fail to do so, resulting
in coarser approximations and reduced representational capacity.


Figure 17: Object detection: density plot of 9-dim basis function over plane _z_ . Each subplot represents
the density plot of a certain dimension.


32


Figure 18: Object detection: density plot of 20-dim basis function over plane _z_ .


Figure 19: Object detection: density plot of 36-dim basis function over plane _z_ . For brevity, we only
show the first 20 dimensions.


33


Figure 20: Object detection: density plot of 64-dim basis function over plane _z_ . For brevity, we only
show the first 20 dimensions.


Figure 21: Object detection: 9/20/36/64-dim basis functions and the corresponding estimated posterior
density.


34


Interestingly, in the ring case study, we observe that even a 2-dimensional basis function is sufficient
to recover the ring-shaped posterior. As shown in Figure 25, the estimated posterior using a 2dimensional basis function is visually nearly indistinguishable from the true posterior, with only
minor artifacts appearing when the observation _x_ approaches extreme values (e.g., _x_ = 1 _._ 95). This
observation is quantitatively supported by the KL divergence results in Table 5, where both forward
and reverse KL values are low (0.032/0.031) for the 2-dimensional case. These results demonstrate
that for simpler posterior structures, our method can achieve accurate inference with remarkably
low-dimensional basis functions.


**2-dim** **4-dim** **9-dim** **20-dim**


**Forward KL Divergence** 0.032 0.0057 0.0056 **0.0054**
∆ **Forward KL Divergence**      - **-0.0263** -0.0001 -0.0002


**Reverse KL Divergence** 0.031 0.0032 0.0028 **0.0027**
∆ **Reverse KL Divergence**      - **-0.0278** -0.0004 -0.0001


Table 5: Ring: forward/reverse KL divergence for models of different basis function dimensions.


Figure 22: Ring: density plot of 2-dim basis function over plane _z_ . Each subplot represents the
density plot of a certain dimension.


Figure 23: Ring: density plot of 4-dim basis function over plane _z_ .


Figure 24: Ring: density plot of 9-dim basis function over plane _z_ .


35


Figure 25: Ring: density plot of 20-dim basis function over plane _z_ .


Figure 26: Ring: 2/4/9/20-dim basis functions and the corresponding estimated posterior density.


36


E.4 TOWARDS HIGH DIMENSION


To evaluate the capability of our model in predicting high-dimensional posteriors, we construct a
50-dimensional annulus model defined as:


_z_ 1 _, z_ 2 _, . . ., z_ 50 _∼_ Unif[0 _,_ 1] _,_
_z_ = ( _z_ 1 _, . . ., z_ 50) _,_

_x | z_ _∼N_ ( _∥z∥_ [2] _, σ_ [2] ) _,_


where _σ_ [2] = 10 _[−]_ [2] .


In Figures 27 and 28, we randomly select two pairs of dimensions (3, 14) and (43, 47), and visualize
the estimated posterior over these subspaces, i.e., _q_ ( _z_ 3 _, z_ 14 _|_ _x_ ) and _q_ ( _z_ 43 _, z_ 47 _|_ _x_ ). We discretize
each pair onto a 100 _×_ 100 grid and perform Monte Carlo integration over the remaining 48 dimensions
to obtain an estimate of the posterior in the chosen 2D subspace. The results show that the estimated
posteriors closely match the true posteriors, with minor discrepancies likely due to Monte Carlo
integration variance. The final two columns of each plot show the marginal densities, demonstrating
that our model captures the true marginal posterior distributions. This good performance can also be
verified by quantitative metrics. Our model attains 0.018 forward KL divergence and 0.022 reverse
KL divergence, on average, across 50 dimensions.


Figure 27: True and estimated posterior density over dimensions 3 and 14. The y-axis in the last two
columns represents the marginal density.


37


Figure 28: True and estimated posterior density over dimensions 43 and 47. The y-axis in the last
two columns represents the marginal density.


38


E.5 EIGENVI ON 2D CASE STUDIES


We evaluate EigenVI (Cai et al., 2024) on three two-dimensional targets with thin or curved posterior
density patterns. We use a tensor product expansion with _K_ = 16 basis functions per axis ( _K_ [2] = 256
coefficients) and 50 _,_ 000 importance samples for fitting. The reconstructions capture only coarse
structure: for the diagonal bands, EigenVI recovers orientation, but the two ridges are blurry; for
the ring, it fills the central hole and collapses mass inward; for the spiral, it loses the manifold and
yields blurry lobes. These failures arise from the spectral bias of orthogonal expansions, which
under-represent the high-frequency content required by a thin or strongly curved posterior. While
increasing _K_ can help, computational and statistical costs scale as _K_ _[d]_ (here _d_ = 2), making adequate
resolution impractical. In sum, with practical _K_, EigenVI is adequate for smooth densities but
inadequate for multimodal or topologically nontrivial two-dimensional targets.


Figure 29: EigenVI results for three 2D test cases. Along each axis, we fit 16 basis functions (i.e.,
_K_ = 16 in their original paper); For their importance sampling, we draw 50,000 samples.


39


E.6 SCORE MATCHED NEURAL EXPONENTIAL FAMILIES ON 2D CASE STUDIES


We reproduce the method proposed in Score Matched Neural Exponential Families for LikelihoodFree Inference (Pacchiardi & Dutta, 2022) and evaluate its sampling quality on three two-dimensional
case studies. Their approach estimates the unnormalized probability _p_ ( _x | z_ ), rather than our focus
on _p_ ( _z_ _| x_ ), and employs Exchange MCMC to draw posterior samples. As illustrated in the figure
below, the Exchange MCMC samples are suboptimal, often overdispersed and misaligned with the
true density. In the bands case study, the samples fail to align with the ridges and instead spread into
low-density regions. In the ring case study, many samples are scattered inside the ring rather than
concentrating on its boundary, where the density peaks. In the spiral case study, the bias is most
evident, with samples deviating substantially from the high-density spiral structure. We also apply
inverse transform sampling, which was not considered in the original paper. This approach produces
samples that more closely follow the true density across all three case studies, though in the spiral
example, a residual bias remains.


Figure 30: Sampling results of Score Matched Neural Exponential Families (SMNEF) for LikelihoodFree Inference on three 2D case studies: SMNEF uses Exchange MCMC as its default sampling
setting. SMNEF+ITS is our variant of SMNEF, customized for low-dimensional settings. For our
method, we use inverse transform sampling.


40


E.7 COMPUTATIONAL COST


Table 6 compares the computational efficiency of several baseline models with the proposed LBFNPE across the 2D case studies, object detection, and redshift estimation. While LBF-NPE shows
slightly higher per-step runtime and memory usage than some baselines, its principal advantage is
its substantially faster convergence, requiring markedly fewer training steps, such as 8k in the 2D
case and 48k in the redshift task. This accelerated convergence results in competitive or superior total
training time across tasks, indicating that LBF-NPE achieves an effective optimization trajectory. A
notable detail is that, in the redshift experiments, the computational costs of all three methods appear
very similar. This is largely due to the dominant overhead from the convolutional U-shape network
used for image processing, which outweighs differences in the loss computation. Another point of
clarification is that the GPU memory usage reported here is lower than the values in Section D.2 (e.g.,
the peak usage is approximately 8400 MB). This discrepancy arises because, for the computational
cost evaluation, we disable GPU memory preallocation in JAX, which otherwise reserves roughly
75% of the available GPU memory.


**Time per step** **GPU Memory** **Converge at**
**Case study** **Method** **Time until converge (s)**
**(s/step)** **(MB)** **(steps)**


**Sinusoidal likelihood** LBF-NPE 0.038 312 **4k** **(batch size:** **1024)** MDN **0.021** 10k 210


**2D case studies**
**(batch size:** **1024)**


LBF-NPE 0.127 970 **8k** 1016
NSF 0.084 780 29k 2436
RealNVP 0.082 690 25k 2050
MDN **0.044** 13k **Object detection**
LBF-NPE 0.143 2230 15k 2145
**(batch size:** **1024)**


**Redshift**
**(batch size:** **32)**


LBF-NPE 0.28 7319 **48k** **13440**
NSF 0.28 7012 80k 22400
MDN **0.26** **6988** 54k 14040


Table 6: Computational cost: For 2D case studies, we only report the computational cost for the
spiral case study, because the other two case studies have similar computational costs. The
"Converge at (steps)" refers to the maximum number of training steps required to achieve the
performance reported in the paper. NSF is the abbreviation of Neural Spline Flow.


41


E.8 BAYESIAN NEURAL NETWORKS AS GENERATIVE MODELS


A Bayesian neural network (BNN) is a neural network in which each weight (and bias) is treated
as a probability distribution rather than a fixed value (MacKay, 1992a;b; Neal, 1996). When
making predictions, it marginalizes over these distributions to produce not just a prediction, but
also an estimate of uncertainty. We consider BNNs as generative models to illustrate that LBF-NPE
can perform posterior predictive inference while implicitly marginalizing over a high-dimensional
parameter space. In particular, we consider a two-layer fully connected BNN:


BNN _θ_ ( _x_ ) = Linear _θ_ 2�ReLU(Linear _θ_ 1( _x_ ))� _,_ _θ_ 1 _, θ_ 2 _∼N_ ( **0** _,_ **I** ) _,_ (22)


where _θ_ 1 and _θ_ 2 are the weight matrices of the first and second linear layers, respectively, and
ReLU( _·_ ) denotes the rectified linear unit activation. The network takes a one-dimensional input and
produces a one-dimensional output through a hidden layer of width 16. Throughout this experiment
we restrict attention to _x ∈_ [0 _,_ 10] and _y_ _∈_ [ _−_ 8 _,_ 8].


Let _θ_ = ( _θ_ 1 _, θ_ 2) and let the dataset be

_D_ = _{_ ( _xi, yi_ ) _}_ _[n]_ _i_ =1 _[,]_ _yi_ = BNN _θ_ ( _xi_ ) + _ϵi,_ _n_ = 5 _,_


for a fixed draw of _θ_ from _N_ ( **0** _,_ **I** ), and noise _ϵi_ _∼N_ (0 _,_ 1). The posterior predictive distribution for
a new pair ( _x_ _[′]_ _, y_ _[′]_ ) is

                _p_ ( _y_ _[′]_ _| x_ _[′]_ _, D_ ) = _p_ ( _y_ _[′]_ _| x_ _[′]_ _, θ_ ) _p_ ( _θ_ _| D_ ) d _θ,_ (23)


which involves integration over the high-dimensional weight vector _θ_ . Our goal in this section is to
show that LBF-NPE can approximate the conditional density in (23) without ever explicitly sampling
or optimizing over _θ_ .


In our implementation, LBF-NPE parameterizes the conditional density via a basis-function network
_sψ_ and a coefficient network _fϕ_ . The coefficient network _fϕ_ takes as input the query variate _x_ _[′]_
together with the conditioning set _D_, while the basis-function network _sψ_ takes _y_ _[′]_ as input. Together
they define
LBF-NPE( _y_ _[′]_ _, x_ _[′]_ _, D_ ) _≈_ _p_ ( _y_ _[′]_ _| x_ _[′]_ _, D_ ) _,_

and are trained using Algorithm 1. Both _sψ_ and _fϕ_ are implemented as four-layer multilayer
perceptrons with 128 hidden units per layer; the resulting basis-function and coefficient vectors have
dimension 20. We optimize the parameters ( _ψ, ϕ_ ) with the AdamW optimizer (Loshchilov & Hutter,
2019), using a learning rate of 10 _[−]_ [3] for 10 _,_ 000 gradient steps.


The qualitative behavior of the learned posterior predictive distributions is shown in Figure 31. Across
the four panels, the underlying BNN functions (orange curves) exhibit markedly different slopes,
curvatures, and ranges, yet LBF-NPE recovers their overall shape from only _n_ = 5 observations.
Moreover, the posterior highest-density interval (HDI) is wide in regions with sparse observations
and narrow in regions with many data points. For example, when no observation is available near
_x ∈_ [0 _,_ 2] (top-left panel), the HDI is wide, whereas in regions densely populated with observations
(e.g., _x_ _∈_ [2 _._ 5 _,_ 5 _._ 0] in the bottom-left panel) the HDI becomes narrow. This pattern indicates that
LBF-NPE successfully captures the epistemic uncertainty induced by marginalization over the BNN
weights.


42


Figure 31: Posterior predictive distributions obtained from LBF-NPE on four synthetic regression
problems generated by a Bayesian neural network (BNN). Each panel corresponds to a different
draw of the BNN weights. The orange curve is the true mapping _x �→_ BNN _θ_ ( _x_ ), and the red dots
denote the _n_ = 5 observed data points used for inference. The solid purple curve shows the pointwise
mode of the posterior predictive density produced by LBF-NPE, and the shaded region indicates the
associated 90% highest–density interval (HDI).


43