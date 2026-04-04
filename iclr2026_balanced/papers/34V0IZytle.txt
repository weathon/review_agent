# WHEN SCORES LEARN GEOMETRY: RATE SEPARATIONS UNDER THE MANIFOLD HYPOTHESIS


**Xiang Li, Zebang Shen, Ya-Ping Hsieh & Niao He**
Department of Computer Science
ETH Zurich
8092 Zurich, Switzerland
_{_ xiang.li,zebang.shen,yaping.hsieh,niao.he _}_ @inf.ethz.ch


ABSTRACT


Score-based methods, such as diffusion models and Bayesian inverse problems,
are often interpreted as learning the **data** **distribution** in the low-noise limit
( _σ_ _→_ 0). In this work, we propose an alternative perspective: their success
arises from implicitly learning the **data** **manifold** rather than the full distribution. Our claim is based on a novel analysis of scores in the small- _σ_ regime
that reveals a sharp **separation** **of** **scales** : _information_ _about_ _the_ _data_ _manifold_
_is_ Θ( _σ_ _[−]_ [2] ) _stronger_ _than_ _information_ _about_ _the_ _distribution._ We argue that this
insight suggests a paradigm shift from the less practical goal of distributional
learning to the more attainable task of **geometric learning**, which provably tolerates _O_ ( _σ_ _[−]_ [2] ) larger errors in score approximation. We illustrate this perspective
through three consequences: i) in diffusion models, concentration on data support
can be achieved with a score error of _o_ ( _σ_ _[−]_ [2] ), whereas recovering the specific data
distribution requires a much stricter _o_ (1) error; ii) more surprisingly, learning the
**uniform** **distribution** on the manifold—an especially structured and useful object—is also _O_ ( _σ_ _[−]_ [2] ) easier; and iii) in Bayesian inverse problems, the **maximum**
**entropy prior** is _O_ ( _σ_ _[−]_ [2] ) more robust to score errors than generic priors. Finally,
we validate our theoretical findings with preliminary experiments on large-scale
models, including Stable Diffusion.


1 INTRODUCTION


_Score learning_ has emerged as a particularly powerful paradigm for modeling complex probabilistic
distributions, driving breakthroughs in generative modeling, Bayesian inverse problems, and sampling (Laumont et al., 2022; Saremi et al., 2023; Ho et al., 2020; Song & Ermon, 2019; Song et al.,
2021). Let _µ_ data be a data measure over R _[d]_ and define a Gaussian-smoothed measure as

��          _µσ_ := law ( _X_ + _σZ_ ) or _µσ_ := law 1 _−_ _σ_ [2] _X_ + _σZ_ _,_ where _X_ _∼_ _µ_ data _, Z_ _∼N_ (0 _, I_ ) _._ (1)


Let _pσ_ be its density function w.r.t. the Lebesgue measure over R _[d]_ . A key step in the score learning
framework is to approximate the score function _∇_ log _pσ_ and to sample from the target distribution
_µσ_, possibly across a spectrum of different _σ_ values (Vincent, 2011; Hyv¨arinen & Dayan, 2005).


A central challenge in this framework is understanding the _low-temperature limit_, i.e., learning the
score of _µσ_ as _σ_ _→_ 0, which encodes the most detailed information about the data distribution. Empirically, this regime is also the most valuable: low-temperature scores underpin many probabilistic
learning frameworks (Laumont et al., 2022; Saremi et al., 2023; Janati et al., 2024; Kadkhodaie
& Simoncelli, 2020), including the influential diffusion model framework (Ho et al., 2020; Song
et al., 2020; Karras et al., 2022), whose noise schedules are specifically designed to emphasize low
temperatures and often require substantial post-training engineering to stabilize the learned scores.


Despite its importance, accurately estimating the score function in the low- _σ_ regime remains notoriously difficult (Song et al., 2021; Karras et al., 2022; Arts et al., 2023; Raja et al., 2025; Stanczuk
et al., 2024). Motivated by this challenge, this paper establishes a new qualitative phenomenon under
the widely adopted _manifold_ _hypothesis_, which posits that the data distribution _µ_ data is supported
on a low-dimensional manifold _M_ embedded in a high-dimensional ambient space.


1


(a) Existing Paradigm


learning the


uniform sampling in


Bayesian inverse problems
with maximum entropy prior


learning difficulty


(b) New Paradigm


learning difficulty


Figure 1: Toy examples illustrating recovered distributions under different regimes, with the manifold represented as a one-dimensional circle embedded in R [2] .


Our key finding, formalized in Theorem 3.1, is that in the small- _σ_ regime of score learning there
is a **sharp** **separation** **of** **scales** : _geometric_ _information_ _about_ _the_ _data_ _manifold_ _appears_ _at_ _order_
Θ( _σ_ _[−]_ [2] ) _, whereas density information of µ_ data _emerges only at order_ Θ(1). As shown in Section 3,
this implies that distribution learning of _µσ_ (e.g., in diffusion models) **necessarily** first recovers the
support of the data distribution before any information about the density can be learned. This perspective naturally separates score learning into two fundamental tasks: _geometric_ _learning_, which
targets the manifold geometry, and _density learning_, which targets the specific data density on that
manifold, with the latter being order of magnitude more difficult. It also suggests that the practical
success of score-based models (e.g., diffusion models) stems from constraining generated samples
to the manifold, thereby producing realistic data even without fully recovering the underlying distribution. According to our analysis, to achieve this, a score error even as large as _o_ ( _σ_ _[−]_ [2] ) is sufficient.


However, our analysis reveals a critical limitation: unless the score is learned to a stringent accuracy
that is beyond _O_ (1), attempts to recover the data distribution may yield _arbitrary_ densities supported
on the manifold. This amounts to only a partial recovery of geometry and can compromise the reliability of downstream tasks and analyses. Such an observation motivates us to pursue _full geometric_
_learning_ —that is, learning to sample _uniformly_ with respect to the manifold’s intrinsic (Riemannian)
volume measure, as it is well-known that uniform samples can best support tasks that depend solely
on the underlying geometry (e.g., Laplace–Beltrami and heat-kernel approximation, geodesic and
diffusion distances) (Coifman & Lafon, 2006; Belkin & Niyogi, 2008; Jost, 2005). In addition, they
also facilitate principled manifold exploration, yielding diverse samples while mitigating potential
biases present in _µ_ data (De Santi et al., 2025).


In this light, a central contribution of this work is to show that a simple, one-line modification
to standard algorithms can _provably_ generate the _uniform_ _distribution_ on the manifold—requiring
only _o_ ( _σ_ _[−]_ [2] ) score accuracy, in stark contrast to the _o_ (1) accuracy needed for exact distributional
recovery. In summary, we advocate a paradigm shift: from the demanding goal of _distributional_
_learning_ toward the more practical and robust objective of _geometric learning_ .


We substantiate the aforementioned rate separation phenomenon by three key results (see also Figure 1):


- Theorem 4.1 shows that, in existing frameworks, the score accuracy required to force concentration on the data manifold is _O_ ( _σ_ _[−]_ [2] ) weaker than that needed to exactly recover _µ_ data. Nevertheless, the resulting distribution can still be _arbitrary_ .


- In contrast, Theorems 5.1 to 5.2 establish a new paradigm centered on extracting precise _geometric_
information of the data manifold by producing the _uniform distribution_ . Notably, we show that a
simple one-line modification of a widely used sampling algorithm suffices to obtain samples from
the uniform distribution under the relaxed score error condition _o_ ( _σ_ _[−]_ [2] ), substantially weaker than
the _o_ (1) required for full recovery of _µ_ data.


- In the context of Bayesian inverse problems (Venkatakrishnan et al., 2013), Theorem 6.1 establishes a rate separation in posterior sampling depending on the choice of prior. When the prior is
uniform, posterior sampling requires only _o_ ( _σ_ _[−]_ [2] ) score accuracy. By contrast, when the prior is
taken to be the commonly used data distribution _µ_ data, substantially stronger accuracy guarantees
are needed to ensure provable success in existing works (Laumont et al., 2022; Pesme et al., 2025).


2


We validate these theoretical results with preliminary experiments on both synthetic and real-world
data, including an application of our algorithm to a large-scale image generation model (Stable
Diffusion 1.5 (Rombach et al., 2022)).


1.1 RELATED WORK


**Diffusion models for distribution learning.** Prior theory shows that diffusion/score-based samplers converge to the target law when the learned score is accurate, with error bounds that scale
directly with the score mismatch (De Bortoli, 2022; Chen et al., 2023; Lee et al., 2023); related
works study other factors such as dimension dependence (Azangulov et al., 2024; Tang & Yang,
2024). However, these results do not separate geometry from density in the score error but instead
consider them together, therefore they do not imply any scale separation.


**Diffusion** **models detect** **data** **manifold.** There is a growing body of work probing whether diffusion models learn the full data distribution or primarily the underlying low-dimensional manifold. A number of studies suggest that these models often capture the data support while missing
fine-grained distributional structure. However, these results are obtained under restricted settings:
Stanczuk et al. (2024) focuses on estimating the intrinsic dimension of the data manifold; Ventura
et al. (2024) analyzes only linear manifolds (linear subspaces); and Pavlova & Wei (2025) provides
primarily empirical evidence. Pidstrigach (2022) establishes sufficient regularity conditions under
which high-accuracy scores concentrate mass near the manifold, but does not address how approximation errors scale with _σ_ and therefore does not reveal a separation of scales. By contrast, our
analysis quantifies how inaccuracies in the learned score propagate differently to geometry versus
distribution learning, exhibiting distinct error rates that lead to a sharp scale separation in the small- _σ_
regime. Furthermore, prior work does not address full geometric recovery via uniform sampling.


**Asymptotic behavior of the score.** It is established that under the manifold hypothesis, the score
function develops a singularity in the small-noise regime, becoming orthogonal to the data manifold.
Recent works characterize this behavior mathematically, showing that the score effectively acts as a
geometric projection operator onto the manifold (Lu et al., 2023; Lyu et al., 2025; Liu et al., 2025).
This aligns with the leading-order term in our expansion (Equation (6)), which governs geometric
concentration. However, these analyses generally subsume the distributional information into a
generic bounded remainder term. Crucially, they do not explicitly isolate the higher-order terms
involving _p_ data and thus do not characterize the separation between geometry and density. Our
analysis reveals that these missing terms are not merely residuals but are essential for establishing
the rate separation between recovering the manifold support and learning the underlying density.


**Uniform** **sampling** **on** **manifolds.** Classical approaches achieve uniform-on-manifold sampling
via graph-based normalizations that cancel the sampling density so that the limiting operator is
the Laplace–Beltrami operator (Coifman & Lafon, 2006; Hein et al., 2007). While foundational,
these methods are designed to approximate geometric operators from neighborhood graphs and do
not readily scale to high-dimensional, large-scale generative modeling. Recently, De Santi et al.
(2025) proposed fine-tuning diffusion models to produce uniform samples. In contrast, our approach
operates entirely at inference time, achieving uniform sampling without the cost of fine-tuning.


2 PRELIMINARIES AND NOTATION


In this work, we adopt the manifold assumption (Song & Ermon, 2019; De Bortoli, 2022; LoaizaGanem et al., 2024) as follows:


**Assumption** **2.1** (The Manifold Hypothesis) **.** _We_ _assume_ _that_ _the_ _data_ _distribution_ _µ_ data _is_ _sup-_
_ported on a compact, boundaryless C_ [4] _embedded submanifold M ⊂_ R _[d]_ _, with_ dim( _M_ ) = _n._


**Local** **coordinates** **and** **manifold** **geometry.** Under the manifold hypothesis, the _n_ -dimensional
manifold _M_ can be described locally using coordinates from a flat, Euclidean space. This is done
via a set of smooth mappings, or charts, Φ : _U_ _→M_, where each chart maps an open set of
parameters _U_ _⊂_ R _[n]_ to a patch on the manifold. For notational simplicity, we will work with a single
chart, where _u ∈_ _U_ represents the local coordinates of a point Φ( _u_ ) on _M_ . The manifold’s intrinsic,


3


and generally non-Euclidean, geometry is captured by the Riemannian metric tensor, _g_ ( _u_ ). This
tensor provides the means to measure lengths and angles on the curved surface. The metric gives
rise to the Riemannian volume measure, _dM_ ( _x_ ), which is the natural way to integrate a function
_f_ : _M →_ R over the manifold. In local coordinates, this integral is expressed as - _M_ _[f]_ [(] _[x]_ [)] _[ d][M]_ [(] _[x]_ [) =]

- _U_ _[f]_ [(Φ(] _[u]_ [))] �det( _g_ ( _u_ )) _du_, w.r.t. the Lebesgue measure on _U_ . Here, the term �det( _g_ ( _u_ )) is the

volume correction factor. While we use a single chart for clarity, integration over the entire compact
manifold is handled by stitching together multiple charts via a partition of unity. The set of points in
R _[d]_ that are sufficiently close to the manifold forms the tubular neighborhood: _TM_ ( _ϵ_ ) := _{x ∈_ R _[d]_ :
dist( _x, M_ ) _<_ _ϵ}_ . For any point _x_ within this neighborhood, there exists a unique closest point on
the manifold, given by the _PM_ ( _x_ ) : _TM_ ( _ϵ_ ) _→M_ . This projection allows us to define the squared
distance function to the manifold, a quantity of central importance to our analysis:


1

_dM_ ( _x_ ) := [1] [min] (2)

2 [dist][2][(] _[x,][ M]_ [) =] _x_ ¯ _∈M_ 2 _[∥][x][ −]_ _[x]_ [¯] _[∥]_ [2] _[.]_


Further details and notations regarding the manifold hypothesis are provided in Appendix A.


2.1 THE GAUSSIAN SMOOTHED MEASURE AND CONNECTION TO DIFFUSION MODELS


With Assumption 2.1, we define the corresponding density _p_ data of _µ_ data with respect to the
Lebesgue measure on _U_ : _p_ data( _u_ ) := _d_ (Φ _[∗]_ _duµ_ data) ( _u_ ), where Φ _[∗]_ _µ_ data( _S_ ) := _µ_ data(Φ( _S_ )) for

_S_ _⊆_ _U_, and assume the following regularity assumption:
**Assumption** **2.2** (Regularity and Converage of _p_ data) **.** _The_ _probability_ _density_ _p_ data : _U_ _→_ R
_defined w.r.t._ _the Lebesgue measure on U_ _is C_ [1] ( _U_ ) _and strictly positive._


Recall the two Gaussian–smoothed measures _µσ_ introduced in Equation (1). We follow the naming
convention of Song et al. (2021) and denote by _µ_ [VE] _σ_ the variance–exploding (VE) smoothing and by
_µ_ [VP] _σ_ the variance–preserving (VP) smoothing. Their densities w.r.t. the Lebesgue measure on R _[d]_ is


    _pσ_ ( _x_ ) :=


_p_ data( _u_ ) d _u,_ (3)


_M_


1   
_−_ _[∥][x][ −]_ _[γ]_ [(] _[σ]_ [)Φ(] _[u]_ [)] _[∥]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


2 _σ_ [2]


_√_
where the densities are denoted _p_ [VE] _σ_ for VE with _γ_ ( _σ_ ) = 1 and _p_ [VP] _σ_ for VP with _γ_ ( _σ_ ) =


where the densities are denoted _p_ [VE] _σ_ for VE with _γ_ ( _σ_ ) = 1 and _p_ [VP] _σ_ for VP with _γ_ ( _σ_ ) = 1 _−_ _σ_ [2] .

We take _p_ data to be the true population density rather than a finite-sample empirical approximation.


These smoothed distributions correspond to the marginals of the forward noising processes used in
diffusion and score-based generative modeling. In SMLD or VE-SDE (Song et al., 2021), Gaussian
noise with variance _σ_ [2] ( _t_ ) : R+ _→_ R+ is added to the data at time _t_, a model is trained to progressively denoise, and in the reverse process the objective is to sample from _p_ [VE] _σ_ ( _t_ ) [, recovering] _[ p]_ [data] [as]
_t_ _→_ 0 (equivalently, _σ_ ( _t_ ) _→_ 0). Similarly, DDPM or VP-SDE (Ho et al., 2020; Song et al., 2021)
corresponds to the VP density _p_ [VP] _σ_ ( _t_ ) [,] [again] [with] [the] [goal] [of] [recovering] _[p]_ [data] [in] [the] [limit] _[t]_ _[→]_ [0][.]
Beyond the reverse process, one may also directly use the learned score to run a Langevin sampler
targeting _p_ [VE] _σ_ (Song & Ermon, 2019) or _p_ [VP] _σ_ [, or combine Langevin sampling with the reverse pro-]
cess, as in the Predictor–Corrector algorithm (Song et al., 2021). Since our results apply to both VE
and VP settings, we adopt the unified notation _pσ_ whenever no ambiguity arises.


2.2 BAYESIAN INVERSE PROBLEMS


Another important algorithmic implication of our results concerns Plug-and-Play (PnP) methods
for Bayesian inverse problems (Venkatakrishnan et al., 2013). Let _x_ _∈_ R _[d]_ be the latent signal and
_y_ _∈Y_ _⊆_ R _[m]_ the observation _y_ = _A_ ( _x_ )+ _ξ_, where _A_ : R _[d]_ _→_ R _[m]_ is the measurement map and _ξ_ _∈_
R _[m]_ is noise. Under standard assumptions on _A_ and _ξ_ (e.g., _A_ linear, _ξ_ _∼N_ (0 _, s_ [2] _I_ )), the likelihood
admits a density _p_ ( _y_ _|_ _x_ ) _∝_ exp� _−_ _v_ ( _x_ ; _y_ )� (for the Gaussian case, _v_ ( _x_ ; _y_ ) = 21 _s_ [2] _[∥][A]_ [(] _[x]_ [)] _[ −]_ _[y][∥]_ [2][).]
In the Bayesian framework we endow _x_ with a prior _p_ prior. Inference is cast as sampling from the
posterior _p_ ( _x | y_ ) = _p_ ( _y_ _| x_ ) _p_ prior( _x_ ) _/_ - _p_ ( _y_ _|_ _x_ ¯) _p_ prior(¯ _x_ ) _dx_ ¯.


**Plug-and-Play (PnP).** PnP methods address the case where the prior is (i) known up to a normalizing constant, e.g. a Gibbs measure or (ii) only accessible via samples (common in ML). A unifying
sampling paradigm is posterior Langevin with a (possibly learned) prior score ˆ _s ≃∇_ log _p_ prior,


_√_
d _Xt_ = _−∇xv_ ( _Xt_ ; _y_ ) d _t_ + _s_ ˆ( _Xt_ ) d _t_ +


2 d _Wt._ (4)


4


In case (ii), _s_ ˆ is a score estimator obtained, e.g., by score matching on prior samples. A common
choice of _p_ prior would be the density _pσ_ defined in eq. (3) with small _σ_ . In this context, to ensure update (4) yields samples matching the target posterior distribution, existing works require the learned
score ˆ _s_ to be at least _o_ (1) accurate (Laumont et al., 2022), or even exact (Pesme et al., 2025).


2.3 STATIONARY DISTRIBUTION FOR NON-REVERSIBLE DYNAMICS


In score learning, one typically learns a score function _s_ ( _x, ϵ_ ) for a target density and then runs
Langevin dynamics (equivalently, the corrector step in the Predictor–Corrector algorithm for diffusion models (Song et al., 2021)) until near stationarity to sample from that density:


_√_
_dXt_ = _s_ ( _Xt, ϵ_ ) _dt_ +


2 _dWt._


If _s_ ( _x, ϵ_ ) = _−∇fϵ_ ( _x_ ), the stationary distribution is proportional to exp( _−fϵ_ ( _x_ )). In practice, however, the score is often produced by a parameterized model and need not be a gradient field (this is
also the case for our proposed algorithms). The resulting Langevin dynamics is then generally _non-_
_reversible_, and its stationary distribution need not admit a closed form—an open problem studied
in, e.g., (Graham & T´el, 1984; Maes et al., 2009; Rey-Bellet & Spiliopoulos, 2015).


Several works have sought to characterize the stationary distribution of non-reversible SDEs. Notably, Matkowsky & Schuss (1977); Maier & Stein (1997); Graham & T´el (1984); Bouchet &
Reygner (2016) employ the WKB ansatz (Wentzel, 1926; Kramers, 1926; Brillouin, 1926), which
is commonly used in matched asymptotic expansions (Holmes, 2012). This approach posits that the
stationary density takes the form


  -  exp _−_ _[V]_ [ (] _ϵ_ _[x]_ [)] _cϵ_ ( _x_ ) _,_ with _cϵ_ ( _x_ ) =


_k_

- _ci_ ( _x_ ) _ϵ_ _[i]_ _,_ (5)


_i_ =0


for some _k_ _∈_ N. The functions _V_ and _{ci}_ are then identified by inserting (5) into the stationary
Fokker–Planck equation and balancing terms order by order in _ϵ_ . Importantly, prior analyses typically focus on low-dimensional special examples or on drifts with a _single_ stable point. The difficulty
of removing such restrictions turn out to be central to our analysis; see Section 5 for details.


3 CENTRAL INSIGHT: GAUSSIAN SMOOTHING RECOVERS GEOMETRY
BEFORE DISTRIBUTION


This section presents the central insight of the paper: While the proofs of our later main results
are technically involved, they are all guided by a common intuition that is transparent and can be
understood through a simple Taylor expansion of log _pσ_ at _σ_ = 0:
**Theorem 3.1** (Informal Theorem B.2) **.** _Assume Assumptions 2.1 and 2.2 holds. For any x ∈_ _TM_ ( _ϵ_ ) _,_


log _pσ_ ( _x_ ) = _−_ [1]


2 _[n]_ log(2 _πσ_ [2] ) + _H_ ( _x_ ) + _o_ (1) _,_ (6)


_σ_ [2] _[d][M]_ [(] _[x]_ [) + log] _[ p]_ [data][(Φ] _[−]_ [1][(P] _[M]_ [(] _[x]_ [)))] _[ −]_ _[d][−]_ 2 _[n]_


_where_ _H_ ( _x_ ) _contains_ _the_ _curvature_ _information_ _of_ _the_ _manifold_ _and_ _ϵ_ _is_ _some_ _sufficiently_ _small_
_constant; both of them are independent of σ._ _The small o_ (1) _term is uniform for x ∈_ _TM_ ( _ϵ_ ) _._


From Equation (6), it follows immediately that the scaled log-density recovers the distance function
to the manifold in the small _σ_ limit:


lim [=] _[−][d][M]_ [(] _[x]_ [)] uniformly for all _x ∈_ _TM_ ( _ϵ_ ) _._ (7)
_σ→_ 0 _[σ]_ [2][ log] _[ p][σ]_ [(] _[x]_ [)]


The appearance of _dM_ ( _x_ ) under the manifold hypothesis should not come as a surprise; indeed, as
_pσ_ _→_ _p_ data when _σ_ _→_ 0, and since _p_ data is supported entirely on _M_, any point _x_ with _dM_ ( _x_ ) _>_ 0
must be assigned zero probability, which explains the divergent scaling factor _σ_ _[−]_ [2] in the coefficient.
What is more surprising is that _only dM_ ( _x_ ) appears at leading order, with _no dependence on p_ data:
Information about _p_ data enters only at the higher-order terms of Θ(1).


This reveals a fundamental _rate separation_ : for _any_ distribution supported on _M_, one must first recover _dM_ ( _x_ ) _exactly_ before learning anything about _p_ data, as any inaccuracy in _dM_ ( _x_ ) gets blown


5


up by the diverging factor _σ_ _[−]_ [2] . Moreover, coefficients encoding _p_ data appear at order _O_ ( _σ_ _[−]_ [2] )
higher, meaning that extracting information about _p_ data requires a level of accuracy orders of magnitude stricter than that needed to recover the manifold geometry, i.e., the distance function _dM_ .


As demonstrated in Sections 4 to 5, this observation entails several significant consequences for
machine learning. Each of these can be understood as a manifestation of the fundamental rate
separation between geometric recovery _vs_ . distributional learning established in Theorem 3.1.


4 SCALE SEPARATION IN EXISTING GENERATIVE LEARNING: GEOMETRY
VERSUS DISTRIBUTION


In this section, we study the paradigm of existing generative learning where algorithms target to
learn the Gaussian-smoothed measure _µσ_, such as the diffusion models discussed in Section 2.1.
We denote the corresponding perfect score function by _s_ _[∗]_ ( _x, σ_ ) := _∇_ log _pσ_ ( _x_ ) _._


In practice, however, the generated samples may follow a different distribution due to imperfections
such as errors in training or discretization of the reverse differential equation. We therefore let
_πσ_ ( _x_ ) : R _[d]_ _→_ R denote the density of the distribution actually produced by an empirical algorithm,
and define its associated score as _sπσ_ ( _x_ ) := _∇_ log _πσ_ ( _x_ ) _._ Our analysis focuses on _πσ_ in terms of
discrepancies between _sπσ_ ( _x_ ) and the ideal score _s_ _[∗]_ ( _x, σ_ ).


Before presenting our result, we impose the following assumption on the recovered distribution.

**Assumption 4.1.** _We denote the log-density of the recovered distribution as −fσ_ := log _πσ_ ( _x_ ) _, and_
_assume that fσ_ _is C_ [1] ( _K_ ) _._ _Furthermore, we impose the following conditions:_


_1._ _There exists a compact set K_ _⊂_ R _[d]_ _with TM_ ( _ϵ_ ) _⊂_ _K_ _such that the density concentrates on K_ _as_
_σ_ _→_ 0 _, i.e.,_ lim _σ→_ 0  - _K_ _[π][σ]_ [(] _[x]_ [)] _[ dx]_ [ = 1] _[.]_


_2._ _K is uniformly rectifiably path-connected, meaning that for any two points x, y_ _∈_ _K, there exists_
_a path in K_ _connecting x and y whose length is uniformly bounded for all x, y_ _∈_ _K._

_Remark_ 4.1 _._ We believe our assumptions are already reflected in practice: Since _πσ_ represents the
effective distribution of the generated samples, it can incorporate standard constraints such as data
clipping (e.g., to [ _−_ 1 _,_ 1]) used in many diffusion models (Ho et al., 2020; Saharia et al., 2022).
This ensures the generated density concentrates on a compact set _K_ as required. Furthermore, such
regular sets are naturally uniformly rectifiably path-connected.


We are ready to state our main result in this section; see Appendix B.3 for the proof.

**Theorem 4.1.** _Suppose Assumptions 2.1, 2.2 and 4.1 hold._ _Denote the score error as_


_Eσ_ := _∥sπσ_ _−_ _s_ _[∗]_ ( _·, σ_ ) _∥L∞_ ( _K_ ) _._


_1._ _**Concentration on Manifold.**_ _If we have that_ _**Eσ**_ **=** _**o**_ **(** _**σ**_ _**[−]**_ **[2]** **)** _, then πσ_ _concentrates on M, i.e.,_


           


lim
_σ→_ 0


_πσ_ ( _x_ ) _dx_ = 0 _for any_ _δ_ _>_ 0 _._
dist( _x,M_ ) _>δ_


_2._ _**Arbitrary Distribution Recovery.**_ _For any distribution_ _π_ ˆ _supported on M with C_ [1] _density,_ _one_
_can construct fσ_ _such that_ _**Eσ**_ **= Ω(1)** _as σ_ _→_ 0 _, and πσ_ _converges weakly to_ _π_ ˆ _._


_3._ _**Recovering**_ _p_ data _**.**_ _If we have that_ _**Eσ**_ **=** _**o**_ **(1)** _as σ_ _→_ 0 _, then πσ_ _converges weakly to p_ data _._


This result formalizes the intuitive fact that recovering _p_ data requires _∇_ log _πσ_ to match the true
score to within _o_ (1) accuracy as _σ_ _→_ 0. The reason is clear from the expansion (6): the distribution
_p_ data only appears in the Θ(1) term, and any larger error would overwhelm this information. In
practice, however, achieving such accuracy is extremely challenging, particularly in the small- _σ_
regime. However, recovering the manifold is simple—only _o_ (1 _/σ_ [2] ) accuracy is required such that
as _σ_ _→_ 0, the density will concentrate on _M_ —a shape separation from recovering _p_ data.


**Implications to Diffusion Models.** As we mentioned before, the paradigmatic example to which
our results can be applied is diffusion models. Our Theorem 4.1 then reveals a sharp scale separation


6


in terms of the score error: _well before the true distribution p_ data _is fully recovered, one can already_
_recover a distribution supported on the same data manifold_ . In practice, this often suffices, as what
truly matters is capturing the _structural features_ of the manifold—realistic images, plausible protein
conformations, or meaningful material geometries. This insight provides a potential new explanation
for the remarkable success of diffusion models.


5 NEW PARADIGM OF GEOMETRIC LEARNING: RECOVER UNIFORM
DISTRIBUTIONS WITH _o_ ( _σ_ _[−]_ [2] ) SCORE ERROR


As shown in Theorem 4.1, while concentration on the manifold is orders of magnitude simpler,
the recovered distribution can still be **arbitrary** unless the score is learned with _o_ (1) accuracy. In
contrast, we show in this section the striking fact that even with score errors as large as _o_ ( _σ_ _[−]_ [2] ),
with a simple modification of the existing algorithm, one can recover the _uniform distribution on the_
_manifold_ —a fundamental distribution that plays a key role in scientific discovery and encodes rich
geometric information about the manifold (De Santi et al., 2025; Belkin & Niyogi, 2008).


Unlike in Section 4, where we compared errors by evaluating a learned _distribution πσ_ against the
ideal _pσ_ through their score functions, in this section we assume direct access to an estimated _score_
_oracle s_ ( _·, σ_ ), such as those learned via score matching in diffusion models. Given access to such
an oracle, our proposed algorithm consists of running the following SDE for some _α >_ 0:


_√_
_dXt_ = _σ_ _[α]_ _s_ ( _Xt, σ_ ) _dt_ +


2 _dWt,_ (8)


which we refer to as the _Tempered_ _Score_ (TS) Langevin dynamics. We claim that, under mild
error assumptions, the stationary distribution of this SDE, denoted _π_ ˜ _σ_, converges to the uniform
distribution on the manifold as _σ_ _→_ 0.


Our analysis proceeds in two steps. First, we establish the result in a simplified setting where the
score oracle _s_ ( _·, σ_ ) is guaranteed to be a gradient field, with a proof analogous to Section 4. Second,
we tackle the substantially more challenging case in which no _a priori_ gradient structure is assumed.
Full proofs are provided in Appendix B.5.


**Warm-up:** **Score Oracle is a Gradient Field.** We use the same notation as in Section 4, namely
_s_ ( _x, σ_ ) = _−∇fσ_ ( _x_ ). In this case, the stationary distribution of Equation (8) admits the explicit form


_π_ ˜ _σ_ ( _x_ ) _∝_ exp( _−σ_ _[α]_ _fσ_ ( _x_ )) _._


We then obtain the following result, using a proof technique similar to that of Theorem 4.1.


**Theorem 5.1.** _Assume Assumptions 2.1, 2.2 and 4.1 hold, with πσ_ _replaced by_ _π_ ˜ _σ._ _Suppose_


_∥s_ ( _·, σ_ ) _−_ _s_ _[∗]_ ( _·, σ_ ) _∥L∞_ ( _K_ ) = _o_          - _σ_ _[β]_ [�] _for some β_ _> −_ 2 _._ (9)


_Then_ _for_ _any_ max _{−β,_ 0 _}_ _<_ _α_ _<_ 2 _,_ _as_ _σ_ _→_ 0 _,_ _π_ ˜ _σ_ _converges_ _weakly_ _to_ _the_ _**uniform**_ _**distribu-**_
_**tion**_ _on the manifold M with respect to the intrinsic volume measure._ _More precisely,_ _the limiting_
_distribution_ _π_ ˜ _with respect to the Lebesgue measure on U_ _satisfies_

_π_ ˜( _u_ ) _∝_ _[d][M]_

_du_ [(] _[u]_ [)] _[,]_


_where_ ( _dM/du_ )( _u_ ) = �det( _g_ ( _u_ )) _is the Riemannian volume element on M._


**General** **Non-Gradient** **Score** **Oracle.** While theorem 5.1 already illustrates the rate separation
phenomenon we wish to emphasize, it relies on the highly impractical assumption that the estimated
scores _s_ ( _·, σ_ ) are exact gradient fields. To enhance the applicability of our framework, it is crucial
to relax this stringent assumption.


As discussed in Section 2.3, existing approaches to non-gradient scores (and hence non-reversible
dynamics) typically assume the existence of a unique point _x_ _[∗]_ such that lim _σ→_ 0 _σ_ _[α]_ _s_ ( _x_ _[∗]_ _, σ_ ) = 0,
with the key consequence of collapsing the prefactor _c_ 0 in (5) to a normalization constant _c_ 0( _x_ _[∗]_ ).
Our framework, however, explicitly violates this assumption: we require that lim _σ→_ 0 _σ_ _[α]_ _s_ ( _·, σ_ ) stabilizes to a _manifold_ rather than a singleton. Under this setting, the limiting behavior of _c_ 0 is far
from obvious, and the resolution of this issue turns out to be highly nontrivial.


7


To this end, a central part of our proof is devoted to showing that _c_ 0 nevertheless remains constant,
albeit for an entirely different reason: we prove that the higher-order terms in the Fokker–Planck
expansion enforce _c_ 0 to satisfy a _parabolic_ _PDE_ on the manifold, and by the strong maximum
principle (Gilbarg et al., 1977), the only solutions on a compact manifold are constants.


With these techniques, we obtain the same conclusion as Theorem 5.1:
**Theorem** **5.2.** _Assume_ _Assumptions_ _2.1_ _and_ _2.2_ _and_ _eq._ (9) _hold,_ _and_ _further_ _suppose_ _p_ data _∈_
_C_ [2] ( _U_ ) _._ _For_ _any_ max _{−β,_ 0 _}_ _<_ _α_ _<_ 2 _,_ _assume_ _that_ _the_ _SDE_ _admits_ _a_ _unique_ _stationary_ _distri-_
_bution, denoted_ _π_ ˜ _σ, which locally admits a WKB form (Assumption B.2 with θ_ = _σ_ [2] _[−][α]_ _)._ _Then the_
_conclusion of Theorem 5.1 holds._


Setting _α_ = 0 in eq. (8) recovers the standard Langevin sampler or the “Corrector” step commonly
used in diffusion-based sampling (Song et al., 2021). Our results in Theorems 5.1 and 5.2 therefore
imply that a simple, one-line modification of these standard schemes is enough to recover the uniform distribution on the data manifold _from samples of p_ data, even when the score error is as large as

_o_ ( _σ_ _[−]_ [2] )—a substantially weaker requirement than the _o_ (1) accuracy needed to recover _p_ data itself.
_Remark_ 5.1 _._ In Appendix D, we provide further discussion on the convergence (mixing time) of TS
Langevin compared to standard Langevin dynamics. While characterizing the general convergence
rate is a non-trivial problem left for future work, our analysis indicates that TS Langevin maintains
comparable algorithmic efficiency. In fact, by analyzing the Poincar´e constant, we identify concrete
examples where TS Langevin converges provably exponentially faster than standard, untempered
Langevin dynamics.


6 UNIFORM PRIOR IS MORE ROBUST BAYESIAN INVERSE PROBLEMS


In Bayesian learning, one often sets the prior _p_ prior to the Gaussian-smoothed data distribution
_pσ_ defined in Equation (3) with some small smoothing parameter _σ_ . To ensure asymptotically
correct posterior samples under this choice, the learned score typically must be exact (Pesme et al.,
2025), _s_ ˆ = _∇_ log _pσ_, or achieve vanishing error, _∥s_ ˆ _−∇_ log _pσ∥L∞_ = _o_ (1) (Laumont et al., 2022,
Proposition 3.3 and H2). In contrast, under our framework, if one adopts the manifold volume
measure (i.e., the uniform distribution on _M_ ) as the prior, then correct posterior sampling can be
attained under a substantially weaker requirement: it suffices that the score error scales as _o_ ( _σ_ _[−]_ [2] ).
The precise statement is given in the theorem below.
**Theorem 6.1.** _Under the same assumptions as in Theorem 5.2, and suppose v_ : R _[d]_ _→_ R _is bounded_
_on_ R _[d]_ _, and C_ [1] _on TM_ ( _ϵ_ ) _._ _Then, as σ_ _→_ 0 _, the stationary distribution of the SDE_


_√_
_dxt_ = _−∇v_ ( _xt_ ) _dt −_ _σ_ _[α]_ _∇fσ_ ( _xt_ ) _dt_ +


2 _dWt,_ (10)


_converges weakly to a distribution supported on M with density ∝_ exp� _−v_ (Φ( _u_ ))� _dduM_ [(] _[u]_ [)] _[.]_


**Diffusion** **Models** **with** **Classifier-Free** **Guidance.** The above result can also be applied to diffusion models. The drift term in Equation (10) represents the effective score of a diffusion model with
classifier-free guidance (Ho & Salimans, 2022). In this formulation, _−∇fσ_ denotes the unconditional score estimate, while the guidance term _−∇v_ equals the guidance scale _w_ times the difference
between the conditional and unconditional score estimates. Our tempered score can be applied directly to CFG diffusion models with a Predictor–Corrector sampler: in the corrector (Langevin) step,
replace the score by its tempered version according to Equation (10) (i.e., scale the unconditional
score by _σ_ _[α]_ ). We will demonstrate the effectiveness of this modification empirically in Section 7.2.


7 EXPERIMENTS


To empirically validate our theory, we present preliminary experiments on both simple synthetic
manifolds and a real-world image–generation setting with diffusion models. On synthetic manifolds,
we directly verify the claims of Section 5, demonstrating recovery of the uniform distribution on the
manifold. In the image domain, we show that our proposed algorithm yields samples that are both
more diverse and high-quality. Further experimental details are provided in Appendix C.


7.1 NUMERICAL SIMULATIONS ON ELLIPSE


In this subsection, we illustrate our theoretical results with numerical simulations. We consider
a simple manifold given by an ellipse embedded in the two-dimensional Euclidean space, _M_ =


8


|Col1|p|
|---|---|
||data|


(c) L (circle)


(a) L (ellipse)


(d) TS-1 (circle)


(b) TS-1 (ellipse)


Figure 2: Comparison of stationary sample distributions generated with standard Langevin dynamics
(L) versus our Tempered Score Langevin dynamics Equation (8) with _α_ = 1 (TS-1). The circle and
ellipse correspond to manifolds with ( _a, b_ ) = (1 _,_ 1) and ( _a, b_ ) = (1 _,_ 2), respectively.


**Prompt** **Furniture** **Car** **Architecture**


**Method** P-sim _↑_ I-sim _↓_ P-sim I-sim P-sim I-sim


DDPM 29.56 80.78 26.23 87.30 **27.36** 81.53
PC 29.40 81.24 26.30 87.20 27.13 81.03
_TS (ours)_ **30.20** **80.76** **26.62** **87.14** 27.32 **80.76**


Table 1: Comparison of images generated by DDPM, PC, and TS. The prompts used are “Creative
furniture,” “An innovative car design,” and “A creative architecture.” For PC and TS, the number of
corrector steps and _α_ (for TS) are tuned.


_{_ ( _x, y_ ) _∈_ R [2] _|_ ( _x/a_ ) [2] +( _y/b_ ) [2] = 1 _},_ _a, b >_ 0, and _p_ data is chosen to be a von Mises distribution
supported on the angular parameterization of the ellipse. The score function is parameterized using
a transformer-based neural network, trained with the loss function introduced in (Song & Ermon,
2019). After training, we evaluate the learned score function with _σ_ = 10 _[−]_ [2] and perform Langevin
dynamics until convergence. Training hyperparameters are tuned to minimize the test loss.


As shown in Figure 2, the stationary distribution produced by standard Langevin dynamics deviates
substantially from _p_ data, even in this simple elliptical setting, highlighting the difficulty of accurately
learning the score function at small _σ_ . In contrast, our TS Langevin dynamics reliably recovers the
uniform distribution on the manifold, in agreement with Theorem 5.2.


7.2 IMAGE GENERATION WITH DIFFUSION MODELS


To validate our theoretical findings in a practical, large-scale setting, we conducted experiments
on image generation. We demonstrate that a one-line modification to the widely-used PredictorCorrector (PC) sampling algorithm (Song et al., 2021) can enhance both the quality and diversity of
images generated by a pre-trained diffusion model. These experiments serve as a proof of concept,
applying our proposed Tempered Score (TS) method to off-the-shelf diffusion models. Our modification targets the corrector step of the PC algorithm, which uses Langevin dynamics to refine the
sample at each stage of the reverse process. In our TS method, we scale the unconditioned score
prediction by a factor of _σ_ _[α]_, as motivated by our analysis and discussion in Section 6. The standard
classifier-free guidance term, i.e., _∇v_ in Equation (10), remains unchanged. Specifically, we compare Stable Diffusion 1.5 (Rombach et al., 2022) with a DDPM sampler (Ho et al., 2020), DDPM
with PC sampler, and DDPM with our TS sampler.


We evaluate the performance using two metrics derived from CLIP scores (Hessel et al., 2021),
which measure the cosine similarity between feature embeddings. **Quality** : We use the CLIP
Prompt Similarity (P-sim), defined as the average CLIP score between the generated images and
their corresponding text prompt. A higher P-sim value indicates better alignment with the prompt
and thus higher image quality. **Diversity** : We use the CLIP Inter-Image Similarity (I-sim), which is
the average pairwise CLIP score between all images generated with the same prompt. A lower I-sim
value means greater diversity among the samples.


The experimental results in Table 1 and Table 2 provide empirical validation of our theoretical
framework. Our proposed TS method consistently generates more diverse images than the DDPM
and standard PC baselines across three distinct prompts, while maintaining very high image quality.
In particular, Table 2 shows that, for all numbers of corrector steps considered, TS outperforms


9


**Num.** **Corrector Steps** **5** **10** **15** **20** **30**


**Prompt** **Method** P-sim _↑_ I-sim _↓_ P-sim I-sim P-sim I-sim P-sim I-sim P-sim I-sim


**Furniture** PC 29.40 81.34 29.30 81.24 29.32 81.64 28.98 81.72 28.67 82.33
_TS (ours)_ **29.54** **81.11** **29.58** **80.95** **29.68** **81.34** **29.52** **81.15** **29.43** **81.87**


**Car** PC 26.20 87.20 26.30 87.57 26.24 87.98 26.26 **88.06** 26.17 87.94
_TS (ours)_ **26.23** **87.14** **26.37** **87.42** **26.32** **87.88** **26.28** 88.07 **26.20** **87.87**


**Architect.** PC 27.13 81.83 27.13 81.81 26.92 81.64 26.87 81.60 26.60 81.03
_TS (ours)_ **27.23** **81.58** **27.27** **81.57** **27.14** **81.54** **27.06** **80.97** **26.84** **80.76**


Table 2: Comparison of images generated by PC and TS across different numbers of corrector steps.
For TS, _α_ = 1 is used without further tuning. The prompts are the same as in Table 1.


Figure 3: Top row: PC. Bottom row: _TS (ours)_ . Samples in the same column are generated using
the same prompt, the same number of corrector steps, and the same random seed. As shown, TS
produces samples that appear more authentic and contain richer details.


standard PC in nearly every case. Crucially, these improvements are robust to the choice of _α_ and
are not merely the result of a larger tuning budget; as demonstrated in Table 2, simply setting _α_ = 1
without further tuning is sufficient to consistently enhance both quality and diversity compared to
the baseline. Examples of the generated images by PC and TS are shown in Figure 3.


8 CONCLUSION


This paper advocates for a paradigm shift in score-based learning, moving from the difficult goal of
full distributional recovery to a more robust, geometry-first approach. We demonstrate a fundamental rate separation in the low-noise limit, where information about the data manifold is encoded at
a significantly stronger scale (Θ( _σ_ _[−]_ [2] )) than details about the on-manifold distribution (Θ(1)). This
finding explains why models often succeed at capturing the data support even with imperfect score
estimates. Building on this insight, we introduce Tempered Score (TS) Langevin dynamics, a simple
one-line modification that robustly targets the uniform volume measure on the manifold, tolerating
score errors up to _o_ ( _σ_ _[−]_ [2] ). This geometric approach not only provides a more stable foundation for
Bayesian inverse problems but also, as shown in our experiments with models like Stable Diffusion,
empirically improves the diversity and fidelity of generated samples.


**Limitations** **and** **future** **work.** Key limitations and future directions include: a) The implications
for diffusion models are presently limited: we do not track cumulative error along the sampling
trajectory; instead, we analyze a simplified setting that assumes access to the error of the final
generated distribution. b) Our _L_ _[∞]_ score–error assumption could potentially be relaxed to an _L_ [2]
bound, thereby aligning our theoretical framework with practical training objectives like denoising
score matching (Fisher divergence) that minimize _L_ [2] error. c) It remains to generalize the rate
separation in score estimation into corresponding results on statistical sample complexity. d) Our
analyses on the uniform sampling are in continuous time; we do not quantify discretization error
arising in practical implementations. e) Our experiments are preliminary; we have not conducted a
large-scale study with state-of-the-art diffusion models.


ACKNOWLEDGMENT


The work is supported by ETH research grant, Swiss National Science Foundation (SNSF) Project
Funding No. 200021-207343, and SNSF Starting Grant.


10


REFERENCES


Marloes Arts, Victor Garcia Satorras, Chin-Wei Huang, Daniel Zugner, Marco Federici, Cecilia
Clementi, Frank No´e, Robert Pinsler, and Rianne van den Berg. Two for one: Diffusion models and force fields for coarse-grained molecular dynamics. _Journal_ _of_ _Chemical_ _Theory_ _and_
_Computation_, 19(18):6151–6159, 2023.


Iskander Azangulov, George Deligiannidis, and Judith Rousseau. Convergence of diffusion models
under the manifold hypothesis in high-dimensions. _arXiv preprint arXiv:2409.18804_, 2024.


Mikhail Belkin and Partha Niyogi. Towards a theoretical foundation for laplacian-based manifold
methods. _Journal of Computer and System Sciences_, 74(8):1289–1308, 2008.


Thibault Bonnemain and Denis Ullmo. Mean field games in the weak noise limit: A wkb approach to
the fokker–planck equation. _Physica A: Statistical Mechanics and its Applications_, 523:310–325,
2019.


Freddy Bouchet and Julien Reygner. Generalisation of the eyring–kramers transition rate formula to
irreversible diffusion processes. In _Annales Henri Poincar´e_, volume 17, pp. 3499–3532. Springer,
2016.


L´eon Brillouin. La m´ecanique ondulatoire de schr¨odinger; une m´ethode g´en´erale de r´esolution par
approximations successives. _CR Acad. Sci_, 183(11):24–26, 1926.


Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy
as learning the score: theory for diffusion models with minimal data assumptions. In _International_
_Conference on Learning Representations_, 2023.


Ronald R Coifman and St´ephane Lafon. Diffusion maps. _Applied_ _and_ _computational_ _harmonic_
_analysis_, 21(1):5–30, 2006.


Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis.
_Transactions on Machine Learning Research_, 2022.


Riccardo De Santi, Marin Vlastelica, Ya-Ping Hsieh, Zebang Shen, Niao He, and Andreas
Krause. Provable maximum entropy manifold exploration via diffusion models. _arXiv_ _preprint_
_arXiv:2506.15385_, 2025.


David Gilbarg, Neil S Trudinger, David Gilbarg, and NS Trudinger. _Elliptic_ _partial_ _differential_
_equations of second order_, volume 224. Springer, 1977.


Yun Gong, Niao He, and Zebang Shen. Poincare inequality for local log-polyak- _\_ l ojasiewicz measures: Non-asymptotic analysis in low-temperature regime. _arXiv_ _preprint_ _arXiv:2501.00429_,
2024.


R Graham and T T´el. On the weak-noise limit of fokker-planck models. _Journal_ _of_ _statistical_
_physics_, 35(5):729–748, 1984.


Matthias Hein, Jean-Yves Audibert, and Ulrike von Luxburg. Graph laplacians and their convergence on random neighborhood graphs. _Journal of Machine Learning Research_, 8(6), 2007.


Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A
reference-free evaluation metric for image captioning. _arXiv preprint arXiv:2104.08718_, 2021.


Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. _arXiv_ _preprint_
_arXiv:2207.12598_, 2022.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _Advances in_
_neural information processing systems_, 33:6840–6851, 2020.


Richard Holley and Daniel Stroock. Logarithmic sobolev inequalities and stochastic ising models.
_Journal of Statistical Physics_, 46(5-6):1159–1194, 1987.


Mark H Holmes. _Introduction to perturbation methods_, volume 20. Springer Science & Business
Media, 2012.


11


Chii-Ruey Hwang. Laplace’s method revisited: weak convergence of probability measures. _The_
_Annals of Probability_, pp. 1177–1182, 1980.


Aapo Hyv¨arinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. _Journal of Machine Learning Research_, 6(4), 2005.


Yazid Janati, Badr Moufad, Alain Durmus, Eric Moulines, and Jimmy Olsson. Divide-and-conquer
posterior sampling for denoising diffusion priors. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 37:97408–97444, 2024.


J¨urgen Jost. _Riemannian geometry and geometric analysis_ . Springer, 2005.


Zahra Kadkhodaie and Eero P Simoncelli. Solving linear inverse problems using the prior implicit
in a denoiser. _arXiv preprint arXiv:2007.13640_, 2020.


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _Advances in neural information processing systems_, 35:26565–26577,
2022.


Hendrik Anthony Kramers. Wellenmechanik und halbzahlige quantisierung. _Zeitschrift f¨ur Physik_,
39(10):828–840, 1926.


Tomasz M Łapi´nski. Multivariate laplace’s approximation with estimated error and application to
limit theorems. _Journal of Approximation Theory_, 248:105305, 2019.


R´emi Laumont, Valentin De Bortoli, Andr´es Almansa, Julie Delon, Alain Durmus, and Marcelo
Pereyra. Bayesian imaging using plug & play priors: when langevin meets tweedie. _SIAM Journal_
_on Imaging Sciences_, 15(2):701–737, 2022.


Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence of score-based generative modeling for
general data distributions. In _International_ _Conference_ _on_ _Algorithmic_ _Learning_ _Theory_, pp.
946–985. PMLR, 2023.


Gunther Leobacher and Alexander Steinicke. Existence, uniqueness and regularity of the projection
onto differentiable manifolds. _Annals of global analysis and geometry_, 60(3):559–587, 2021.


Zichen Liu, Wei Zhang, and Tiejun Li. Improving the euclidean diffusion generation of manifold
data by mitigating score function singularity. _arXiv preprint arXiv:2505.09922_, 2025.


Gabriel Loaiza-Ganem, Brendan Leigh Ross, Rasa Hosseinzadeh, Anthony L Caterini, and Jesse C
Cresswell. Deep generative models through the lens of the manifold hypothesis: A survey and
new connections. _Transactions on Machine Learning Research_, 2024.


Yubin Lu, Zhongjian Wang, and Guillaume Bal. Mathematical analysis of singularities in the diffusion model under the submanifold assumption. _arXiv preprint arXiv:2301.07882_, 2023.


Yang Lyu, Tan Minh Nguyen, Yuchun Qian, and Xin T Tong. Resolving memorization in empirical
diffusion model for manifold data in high-dimensional spaces. _arXiv preprint arXiv:2505.02508_,
2025.


Christian Maes, Karel Netoˇcn`y, and Bidzina M Shergelashvili. Nonequilibrium relation between potential and stationary distribution for driven diffusion. _Physical Review E—Statistical, Nonlinear,_
_and Soft Matter Physics_, 80(1):011121, 2009.


Robert S Maier and Daniel L Stein. Limiting exit location distributions in the stochastic exit problem. _SIAM Journal on Applied Mathematics_, 57(3):752–790, 1997.


Piotr Majerski. Simple error bounds for the multivariate laplace approximation under weak local
assumptions. _arXiv preprint arXiv:1511.00302_, 2015.


Bernard J Matkowsky and Zeev Schuss. The exit problem for randomly perturbed dynamical systems. _SIAM Journal on Applied Mathematics_, 33(2):365–382, 1977.


Georg Menz and Andr´e Schlichting. Poincar´e and logarithmic sobolev inequalities by decomposition of the energy landscape. _The Annals of Probability_, 42(5):1809, 2014.


12


John Willard Milnor and James D Stasheff. _Characteristic classes_ . Number 76. Princeton university
press, 1974.


James Raymond Munkres. _Topology_ . Prentice Hall, 2nd edition, 2000.


Elizabeth Pavlova and Xue-Xin Wei. Diffusion models under low-noise regime. _arXiv_ _preprint_
_arXiv:2506.07841_, 2025.


Scott Pesme, Giacomo Meanti, Michael Arbel, and Julien Mairal. Map estimation with denoisers:
Convergence rates and guarantees. _arXiv preprint arXiv:2507.15397_, 2025.


Jakiw Pidstrigach. Score-based generative models detect manifolds. _Advances in Neural Information_
_Processing Systems_, 35:35852–35865, 2022.


Sanjeev Raja, Martin S´ıpka, [ˇ] Michael Psenka, Tobias Kreiman, Michal Pavelka, and Aditi S Krishnapriyan. Action-minimization meets generative modeling: Efficient transition path sampling
with the onsager-machlup functional. _arXiv preprint arXiv:2504.18506_, 2025.


Luc Rey-Bellet and Konstantinos Spiliopoulos. Irreversible langevin samplers and variance reduction: a large deviations approach. _Nonlinearity_, 28(7):2081, 2015.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition (CVPR)_, pp. 10684–10695, June 2022.


Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar
Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic
text-to-image diffusion models with deep language understanding. _Advances in neural informa-_
_tion processing systems_, 35:36479–36494, 2022.


Saeed Saremi, Rupesh Kumar Srivastava, and Francis Bach. Universal smoothed score functions for
generative modeling. _arXiv preprint arXiv:2303.11669_, 2023.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. _arXiv_
_preprint arXiv:2010.02502_, 2020.


Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
_Advances in neural information processing systems_, 32, 2019.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In _ICLR_, 2021.


Jan Pawel Stanczuk, Georgios Batzolis, Teo Deveney, and Carola-Bibiane Sch¨onlieb. Diffusion
models encode the intrinsic dimension of data manifolds. In _Forty-first International Conference_
_on Machine Learning_, 2024.


Rong Tang and Yun Yang. Adaptivity of diffusion models to manifold structures. In _International_
_Conference on Artificial Intelligence and Statistics_, pp. 1648–1656. PMLR, 2024.


Singanallur V Venkatakrishnan, Charles A Bouman, and Brendt Wohlberg. Plug-and-play priors for
model based reconstruction. In _2013 IEEE global conference on signal and information process-_
_ing_, pp. 945–948. IEEE, 2013.


Enrico Ventura, Beatrice Achilli, Gianluigi Silvestri, Carlo Lucibello, and Luca Ambrogioni. Manifolds, random matrices and spectral gaps: The geometric phases of generative diffusion. _arXiv_
_preprint arXiv:2410.05898_, 2024.


Pascal Vincent. A connection between score matching and denoising autoencoders. _Neural compu-_
_tation_, 23(7):1661–1674, 2011.


Gregor Wentzel. Eine verallgemeinerung der quantenbedingungen f¨ur die zwecke der wellenmechanik. _Zeitschrift f¨ur Physik_, 38(6):518–529, 1926.


Hermann Weyl. On the volume of tubes. _American Journal of Mathematics_, 61(2):461–472, 1939.


Stephen Willard. _General topology_ . Courier Corporation, 2012.


13


A ADDITIONAL NOTATION AND PRELIMINARIES


In this section, we provide some notation and preliminaries complementary to Section 2.


We denote by _Wt_ a standard Brownian motion, with its dimension clear from context. The Gaussian
density with mean _µ_ and covariance Σ, evaluated at _x_, is written as _N_ ( _x_ _|_ _µ,_ Σ). The symbol _∗_
denotes the convolution operator. We use _∝_ to indicate proportionality, i.e., that the left-hand side
and right-hand side are equal up to a constant factor. For a set _S_, we write _S_ for its closure, _∂S_ for
its boundary, and _S_ _[c]_ for its complement. Throughout the paper, by the term _limiting_ _distribution_
or by convergence of a distribution/density function, we mean convergence of the corresponding
measures in the weak sense.


A.1 THE MANIFOLD HYPOTHESIS


We outline few notations and standard results from differential geometry. By the tubular neighborhood theorem (Milnor & Stasheff, 1974; Weyl, 1939), there exists _ϵ_ _>_ 0 such that the normal
tube
_TM_ ( _ϵ_ ) := _{x ∈_ R _[d]_ : dist( _x, M_ ) _< ϵ}._
admits local _C_ [4] coordinate

Φ : _U_ _× R →_ _TM_ ( _ϵ_ ) _,_ where _U_ _⊂_ R _[n]_ _, R_ := _{r_ _∈_ R _[d][−][n]_ : _∥r∥_ _< ϵ},_


such that Φ is a diffeomorphism mapping from local coordinates to ambient Euclidean space. With
this result, we can then work with local coordinates to describe the manifold. For notational simplicity, we work with a single chart and suppress indices: _u_ _∈_ _U_ denote tangential coordinates
and _r_ _∈_ _R_ denote normal coordinates. The slice _r_ = 0 corresponds to points on _M_, and we
write Φ( _u_ ) := Φ( _u,_ 0). Let _J_ ( _u, r_ ) denote the Jacobian of Φ( _u, r_ ) with respect to ( _u, r_ ), i.e.,
_J_ ( _u, r_ ) = _∂_ Φ( _u, r_ ) _/∂_ ( _u, r_ ). Furthermore, let _g_ ( _u_ ) denote the Riemannian metric tensor of the
manifold _M_, defined as _g_ ( _u_ ) := _J_ ( _u,_ 0) _[⊤]_ _J_ ( _u,_ 0). Intuitively, the Riemannian metric tensor gives a
way to measure lengths and angles of the manifold geometry.


B PROOFS OF MAIN THEOREMS


In this section, we prove the main theorems of the paper. We begin by developing a general framework for characterizing the limiting distribution when the density admits a specific form. This
framework will then be applied to establish the results in Section 4, where such a density form was
assumed.


The results in Section 5 require a different approach, since no explicit form of the density is available.
In this case, we employ the WKB approximation to obtain an approximate stationary distribution,
which we then substitute into the general framework to derive the limiting distribution.


B.1 A GENERAL FRAMEWORK FOR THE CONVERGENCE OF THE LIMITING DISTRIBUTION


In this subsection, we will establish a general framework for the limiting distribution of density
proportional to


exp ( _−_ ( _fθ_ ( _x_ )) _/θ_ ) _,_ with _fθ_ ( _x_ ) = _f_ 0( _x_ ) + _θf_ 1( _x_ ) + _f_ [ˆ] ( _x, θ_ ) _,_ (11)


where _f_ 0’s minimizer is on the manifold _M_ and _f_ [ˆ] ( _x, θ_ ) is a perturbation that is uniformly _o_ ( _θ_ ) so
that it does not affect the limiting distribution. This general result is stated in Theorem B.1. Our main
results fall into this framework by letting _θ_ = _σ_ [2] for Theorem 4.1 and _θ_ = _σ_ [2] _[−][α]_ for Theorem 5.2.


In all cases the theorems we will prove later, the density will concentrate on the tubular neighborhood of _M_, i.e., _TM_ ( _ϵ_ ). Therefore, we will discuss the lemmas and intermediate results in such
a neighborhood and use local coordinates ( _u, r_ ). The notations used can be found in Section 2.
When we use local coordinates, we assume the discussion is in the closure of _TM_ ( _ϵ_ ). We define
the local coordinate versions of the functions: _fθ_ ( _u, r_ ) := _fθ_ (Φ( _u, r_ )), _f_ 0( _u, r_ ) := _f_ 0(Φ( _u, r_ )),
_f_ 1( _u, r_ ) := _f_ 1(Φ( _u, r_ )), and _f_ [ˆ] ( _u, r, θ_ ) := _f_ [ˆ] (Φ( _u, r_ ) _, θ_ ).


Our assumptions are stated as follows.


14


**Assumption B.1.** _We assume that_


_1._ _M ⊂_ R _[d]_ _is a compact C_ [4] _manifold without boundary with dimension n < d._


_2._ _M_ = arg min _x∈TM_ ( _ϵ_ ) _f_ 0( _x_ ) _._ _In_ _addition,_ _we_ _assume_ _that_ _there_ _exists_ 0 _<_ _ϵ_ ˆ _<_ _ϵ_ _such_ _that_
inf _x∈TM_ ( _ϵ_ ) _\TM_ (ˆ _ϵ_ ) _f_ 0( _x_ ) _−_ min _x∈TM_ ( _ϵ_ ) _f_ 0( _x_ ) _is bounded away from zero._


_3._ _The absolution value of_ _f_ [ˆ] ( _u, r, θ_ ) _is o_ ( _θ_ ) _as θ_ _→_ 0 _uniformly for all u ∈_ _U_ _and ∥r∥_ _< ϵ._


_4._ _f_ 0 _≥_ 0 _is C_ [3] _, f_ 1 _is C_ [1] _, and fθ_ _is continuous on coordinates_ ( _u, r_ ) _for all u_ _∈_ _U_ _and ∥r∥≤_ _ϵ,_
_i.e., in the closure of TM_ ( _ϵ_ ) _._


_5._ _Further,_ _we_ _assume_ _that_ _the_ _smallest_ _eigenvalue_ _of_ _[∂]_ _∂r_ [2] _[f]_ [2][0] [(] _[u, r]_ [)] _[is]_ _[uniformly]_ _[bounded]_ _[away]_ _[from]_

_zero for all u ∈_ _U_ _and ∥r∥_ _< ϵ._


_Remark_ B.1 (Compactness of the manifold implies boundedness of gradients.) _._ Consider _f_ _∈_
_C_ _[k]_ ( _TM_ ( _ϵ_ )). In local coordinates ( _u, r_ ) induced by a tubular atlas, we write _f_ ( _u, r_ ) := _f_ (Φ( _u, r_ )).
Since _M_ is compact, one can choose a finite atlas with precompact coordinate domains. Let the
cover be _{Ui}_ . By the Shrinking Lemma (Munkres (2000, Theorem 32.3) combined with Willard
(2012, Theorem 15.10)), there exist open subsets _{Vi}_ with _Vi_ _⊂_ _Ui_ such that _{Vi}_ still forms a
cover. We use these _{Vi}_ as the new atlas. The transition maps Φ and their derivatives are then
bounded on these sets (since _Vi_ is compact), and by the chain rule the same holds for _f_ ( _u, r_ ) and its
derivatives up to order _k_ . Thus, throughout our arguments we may freely assume uniform boundedness of such derivatives without loss of generality. The same reasoning applies to _p_ data, we can use
the same constructed atlas such that _p_ data is uniformly lower and upper bounded, and gradients of
_p_ data are uniformly upper bounded.


During our proofs, we will frequently use Laplace’s method for integrals. We adapt the error estimate
from Łapi´nski (2019) as follows.

**Corollary B.1** (Theorem 2 of Łapi´nski (2019)) **.** _Let_ Ω _⊂_ R _[m]_ _be an open set and let_ Ω _[′]_ _⊂_ Ω _be a_
_closed ball._ _Let c_ 1 := Vol(Ω _[′]_ ) _._ _Let F, g_ : Ω _→_ R _with the following assumptions:_


_1._ _F_ _|_ Ω _′_ _∈_ _C_ [3] (Ω _[′]_ ) _and F_ _≥_ 0 _on_ Ω _._ _There is a unique minimizer x_ _[∗]_ _∈_ int(Ω _[′]_ ) _of F_ _on_ Ω _._ _Define_


_m_ 1 := inf   - _F_ ( _x_ ) _−_ _F_ ( _x_ _[∗]_ )� _>_ 0 _,_ _m_ 2 := inf   - _∇_ [2] _F_ ( _x_ )� _>_ 0 _._
_x∈_ Ω _\_ Ω _[′]_ _x∈_ Ω _[′][ λ]_ [min]


_Let_
_c_ 2 := sup _c_ 3 := sup
_x∈_ Ω _[′][ ∥∇]_ [2] _[F]_ [(] _[x]_ [)] _[∥][,]_ _x∈_ Ω _[′][ ∥∇]_ [3] _[F]_ [(] _[x]_ [)] _[∥][.]_


_2._ _g|_ Ω _′_ _∈_ _C_ [1] (Ω _[′]_ ) _and_ 


Ω _[|][g]_ [(] _[x]_ [)] _[|][ dx <][ ∞][.]_ _[Let]_


                      _c_ 4 := sup _c_ 5 := sup _c_ 6 := _|g_ ( _x_ ) _| dx._
_x∈_ Ω _[′][ |][g]_ [(] _[x]_ [)] _[|][,]_ _x∈_ Ω _[′][ ∥∇][g]_ [(] _[x]_ [)] _[∥][,]_ Ω


_Then, for every θ_ _>_ 0 _,_

     - (2 _πθ_ ) _[m/]_ [2]

_g_ ( _x_ ) _e_ _[−][F]_ [ (] _[x]_ [)] _[/θ]_ _dx_ = exp( _−F_ ( _x_ _[∗]_ ) _/θ_ ) ( _g_ ( _x_ _[∗]_ ) + _h_ ( _θ_ )) _,_
Ω ~~�~~ _|∇_ [2] _F_ ( _x_ _[∗]_ ) _|_


_√_
_where |h_ ( _θ_ ) _| can be √upper bounded by a function of_ ( _c_ 1 _, . . ., c_ 6 _, m_ 1 _, m_ 2) _. Moreover, h_ ( _θ_ ) = _O_ ( _θ_ )

_as θ_ _→_ 0 _._ _The O_ ( _θ_ ) _is_ uniform _over any class of pairs_ ( _F, g_ ) _for which c_ 1 _, . . ., c_ 6 _are bounded_

_above and m_ 1 _, m_ 2 _are bounded below by strictly positive constants uniformly over the class._


_Proof._ The result follows directly from Łapi´nski (2019, Theorem 2).


To show the convergence of the distribution to a distribution on the manifold, a key step is to integrate
out the normal direction so as to obtain a distribution on _u_, such as what Hwang (1980) did. The
following lemma proves Laplace’s type of result for integrating out _r_ .


15


**Lemma** **B.1.** _Assume_ _Assumption_ _B.1,_ _and_ _let_ _h_ ( _x_ ) : R _[d]_ _→_ R _be_ _C_ [1] _and_ _uniformly_ _bounded_ _in_
_TM_ ( _ϵ_ ) _._ _Define h_ ( _u, r_ ) := _h_ (Φ( _u, r_ )) _._ _Then we have_

      -       -       


   exp _−_ _[f][θ]_ [(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


_θ_


_h_ ( _u, r_ ) _dr_


exp ( _−f_ 1( _u,_ 0)) �� [(2] �� _[πθ]_ _∂∂r_ 2 [)] _f_ [2][(] 0 _[d]_ [(] _[−][u,][n]_ [)][ 0)] _[/]_ [2] ��� ( _h_ ( _u,_ 0) + _o_ (1)) _,_


exp ( _−f_ 1( _u,_ 0)) [(2] _[πθ]_ [)][(] _[d][−][n]_ [)] _[/]_ [2]


  = exp _−_ _[f]_ [0][(] _[u,]_ [ 0)]

_θ_


  = exp _−_ _[f]_ [0][(] _[u,]_ [ 0)]


_where the o_ (1) _term is uniform for u._


_Proof._ We have that

    


   exp _−_ _[f][θ]_ [(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


   exp _−_ _[f]_ [0][(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


   exp _−_ _[f]_ [0][(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


   exp _−_ _[f]_ [0][(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


_θ_


_h_ ( _u, r_ ) _dr_


_f_ ˆ ( _u, r, θ_ )

_−_


_θ_


��


 =


_θ_


_dr._


exp ( _−f_ 1( _u, r_ )) _h_ ( _u, r_ )


exp


 =


_θ_


exp ( _−f_ 1( _u, r_ )) _h_ ( _u, r_ ) _dr_ +


_f_ ˆ ( _u, r, θ_ )

_−_


_θ_


_dr_


 
_−_ 1


_θ_


exp ( _−f_ 1( _u, r_ )) _h_ ( _u, r_ )


exp


For the first term, we can directly apply Corollary B.1 with _F_ ( _r_ ) = _f_ 0( _u, r_ ), _g_ ( _r_ ) =
exp( _−f_ 1( _u, r_ )) _h_ ( _u, r_ ), and Ω _[′]_ being the ball _{r_ _| ∥r∥≤_ _ϵ_ ˆ _}_ . Define


exp ( _−f_ 1( _u,_ 0)) �� [(2] �� _[πθ]_ _∂∂r_ 2 [)] _f_ [2][(] 0 _[d]_ [(] _[−][u,][n]_ [)][ 0)] _[/]_ [2] ��� _._


   _J_ = exp _−_ _[f]_ [0][(] _[u,]_ [ 0)]

_θ_


   _J_ = exp _−_ _[f]_ [0][(] _[u,]_ [ 0)]


The first term can be approximated as _J_ ( _h_ ( _u,_ 0) + _o_ (1)). The boundedness of the quantities in
Corollary B.1 will be discussed later. The second term can be upper bounded by


sup exp
_r_ _[|][h]_ [(] _[u, r]_ [)] _[| ·]_ [ sup] _r_ �����


_f_ ˆ( _u, r, θ_ )

_−_


_−_ 1


exp ( _−f_ 1( _u, r_ )) _dr_


�����


   exp _−_ _[f]_ [0][(] _[u, r]_ [)]
_∥r∥<ϵ_ _θ_


_θ_


_θ_


= _o_ (1) _J_ (1 + _o_ (1)) = _o_ (1) _J,_


where we used Corollary B.1 for the integral. The lower bound can be obtained similarly. The result
follows.

Regarding the uniform boundedness of the quantities in Corollary B.1, _{c}_ [5] 1 [is uniformly bounded]
by the compactness of the manifold. The constant _c_ 6 is uniformly bounded by our assumption on _h_ .
The uniform lower bounds of _m_ 1 and _m_ 2 is guaranteed by Assumption B.1.


Next, we will prove that the support of the limiting distribution will concentrate on the minimizers
of the leading term. Previously, we considered _fθ_ consisting of _f_ 0 + Θ( _θ_ ) + _o_ ( _θ_ ). Next, we will
show that as long as _fθ_ is _f_ 0 + _o_ (1), the concentration on _f_ 0’s minimizers will happen.

**Lemma** **B.2.** _Let_ _fθ_ ( _x_ ) = _f_ 0( _x_ ) + _f_ [˜] ( _x, θ_ ) _,_ _such_ _that_ exp( _−fθ_ ( _x_ ) _/θ_ ) _is_ _a_ _normalized_ _density_
_function_ _on_ R _[d]_ _._ _Suppose_ _M is_ _a_ _connected and compact C_ [4] _manifold without boundary._ _Assume_
_that:_


_1._ _f_ 0( _x_ ) _is continuous with_ arg min _x∈TM_ ( _ϵ_ ) _f_ 0( _x_ ) = _M and_ min _x∈x∈TM_ ( _ϵ_ ) _f_ 0( _x_ ) = 0 _._


_2._ _f_ [˜] ( _x, θ_ ) _is continuous and uniformly o_ (1) _as θ_ _→_ 0 _for all x ∈_ _TM_ ( _ϵ_ ) _._


_3._ _The density concentrates in TM_ ( _ϵ_ ) _, i.e.,_


_dx_ = 1 _._


lim
_θ→_ 0


   exp _−_ _[f][θ]_ [(] _[x]_ [)]
_TM_ ( _ϵ_ ) _θ_


_θ_


16


_For any η_ _>_ 0 _, define the set Cη_ = _{x | f_ 0( _x_ ) _> η}._ _Then,_

         
_as_ _θ_ _→_ 0 _._
_Cη∪TM_ ( _ϵ_ ) _[c]_ [ exp(] _[−][f][θ]_ [(] _[x]_ [)] _[/θ]_ [)] _[ dx][ →]_ [0]


_If_ _in_ _addition,_ exp( _−fθ_ ( _x_ ) _/θ_ ) _converges_ _weakly_ _to_ _a_ _distribution_ _as_ _θ_ _→_ 0 _,_ _the_ _support_ _of_ _the_
_limiting distribution is contained in M._


_Proof._ Since we have that 


_Proof._ Since we have that - _TM_ ( _ϵ_ ) [exp(] _[−][f][θ]_ [(] _[x]_ [)] _[/θ]_ [)] _[dx]_ _[→]_ [1][,] [for] [the] [first] [result,] [it] [suffices] [to] [show]

that - [exp(] _[−][f][θ]_ [(] _[x]_ [)] _[/θ]_ [)] _[dx][ →]_ [0][.] [According to the assumptions, we have that for any] _[ δ]_ [,] _[ ∃][θ]_ [0][,]


_TM_ ( _ϵ_ ) _∩Cη_ [exp(] _[−][f][θ]_ [(] _[x]_ [)] _[/θ]_ [)] _[dx][ →]_ [0][.] [According to the assumptions, we have that for any] _[ δ]_ [,] _[ ∃][θ]_ [0][,]


such that _∀θ_ _< θ_ 0, _|f_ [˜] ( _x, θ_ ) _| < δ_ . Therefore, we have

- 


            exp( _−fθ_ ( _x_ ) _/θ_ ) _dx ≤_
_TM_ ( _ϵ_ ) _∩Cη_


exp(( _−η_ + _δ_ ) _/θ_ ) _dx ≤_ Vol( _TM_ ( _ϵ_ )) exp(( _−η_ + _δ_ ) _/θ_ ) _._
_TM_ ( _ϵ_ ) _∩Cη_


We choose _δ_ = _η/_ 2, then the right-hand side goes to zero as _θ_ _→_ 0.


Let the limiting measure be _P_, and _Pθ_ be the probability measure corresponding to the density
exp( _−fθ_ ( _x_ ) _/θ_ ). Since _Cη_ is an open set, we have that


_P_ ( _Cη_ ) _≤_ lim inf
_θ→_ 0 _[P][θ]_ [(] _[C][η]_ [) = 0] _[.]_


We also have that


      - ~~_c_~~ [�]       - ~~_c_~~ [�]
_P_ _TM_ ( _ϵ_ ) _≤_ lim inf _TM_ ( _ϵ_ ) _≤_ lim inf
_θ→_ 0 _[P][θ]_ _θ→_ 0 _[P][θ]_ [ (] _[T][M]_ [(] _[ϵ]_ [)] _[c]_ [) = 0] _[.]_


~~_c_~~
Denote _C_ := _M_ _[c]_ . We have that _C_ = _∪_ _[∞]_ _m_ =1 _[C]_ 1 _/m_ _[∪]_ _[T][M]_ [(] _[ϵ]_ [)] . Then we have


_P_ ( _C_ ) _≤_


_∞_

- _P_ ( _C_ 1 _/m_ ) + _P_ - _TM_ ( _ϵ_ ) ~~_c_~~ [�] = 0 _._


_m_ =1


which concludes the proof.


**Theorem B.1.** _Assume Assumption B.1._ _Define_


              _πθ_ ( _x_ ) _∝_ exp _−_ _[f][θ]_ [(] _[x]_ [)]

_θ_


_,_


_Assume_ _that_ 1 _−_ - _x∈TM_ ( _ϵ_ ) _[π][θ]_ [(] _[x]_ [)] _[dx]_ _[→]_ [0] _[as]_ _[θ]_ _[→]_ [0] _[.]_ _[Then]_ _[we]_ _[have]_ _[that]_ _[as]_ _[θ]_ _[→]_ [0] _[,]_ _[π][θ]_ _[converges]_

_weakly to the following distribution:_


_∂_ 2 _f_ 0( _u,_ 0) _−_ 1 _/_ 2
exp( _−f_ 1( _u,_ 0))��� _∂r_ [2] ��� _dM_ ( _u_ ) _/du_
_π_ ( _u_ ) =


_,_
_M_ [exp(] _[−][f]_ [1][(] _[u,]_ [ 0))] ��� _∂_ 2 _f∂r_ 0( [2] _u,_ 0) ��� _−_ 1 _/_ 2 _dM_ ( _u_ ) _/du_


_where dM is the intrinsic measure on the manifold M, i.e., dM_ ( _u_ ) = _|g_ ( _u_ ) _|_ [1] _[/]_ [2] _du, and du is the_
_Lebesgue measure on the local parameterization domain U_ _._


_Proof._ The proof follows the same as the proof in Hwang (1980, Theorem 3.1). The only difference
is that we replace the estimate of Hwang (1980, Equation (3.2)) with our Lemma B.1. Note that
the Q in Hwang (1980, Theorem 3.1) is assumed as a probability measure, thus _f_ (in his notation)
integrates to one. However, the proof technique of Hwang (1980, Theorem 3.1) remains valid even
if _f_ is not a probability density, so applying to our case.


B.2 PROOF FOR THEOREM 3.1


The remaining of the proof is to expand the true log-density w.r.t. _σ_, analyze the error of the learned
log-density, and then to plug in the result obtained from Appendix B.1.


17


**Theorem B.2.** _Assume Assumptions 2.1 and 2.2 holds._ _Suppose x ∈_ _TM_ ( _ϵ_ ) _._ _Then we have that_


[1]

2 _σ_ [2] _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2][ + log] _[ p]_ [data][(Φ] _[−]_ [1][(P] _[M]_ [(] _[x]_ [)))] _[ −]_ _[d][ −]_ 2 _[n]_


log _p_ [VE] _σ_ [(] _[x]_ [) =] _[ −]_ [1]


log(2 _πσ_ [2] ) _−_
2


log


~~�~~
��� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )��� + _p_ ˆVE( _x, σ_ ) _,_


[1]

2 _σ_ [2] _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2][ + log] _[ p]_ [data][(Φ] _[−]_ [1][(P] _[M]_ [(] _[x]_ [)))] _[ −]_ _[d][ −]_ 2 _[n]_


log _p_ [VP] _σ_ [(] _[x]_ [) =] _[ −]_ [1]


log(2 _πσ_ [2] ) _−_
2


log


~~�~~

  -  - 1
�� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )�� _−_ 2 _[⟨]_ [P] _[M]_ [(] _[x]_ [)] _[, x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[⟩]_ [+] _[p]_ [ˆ][VP][(] _[x, σ]_ [)] _[,]_


_where_ _p_ ˆ [VE] ( _x, σ_ ) _and_ _p_ ˆ [VP] ( _x, σ_ ) _are functions that are o_ (1) _uniformly for x_ _∈_ _TM_ ( _ϵ_ ) _._ _The matrix_
_H_ ˆ ( _u, x_ ) _is such that_


    - _∂_ 2Φ( _u_ )    -    - _∂_ Φ( _u_ )    _H_ ˆ ( _u, x_ ) _i,j_ = _,_ Φ( _u_ ) _−_ _x_ + _,_ _[∂]_ [Φ(] _[u]_ [)] _._
_∂ui∂uj_ _∂ui_ _∂uj_


_Proof._ We can apply Corollary B.1 as an error estimate for Laplace’s method, to the integral in _pσ_ .
The minimizer of _F_ ( _u_ ) is Φ _[−]_ [1] (P _M_ ( _x_ )) for both VE and VP.

We first consider the case of VE. By letting _F_ ( _u_ ) = _∥x −_ Φ( _u_ ) _∥_ [2] _/_ 2, _g_ ( _u_ ) = _p_ data( _u_ ) and _θ_ = _σ_ [2]
we can obtain that


- _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))) + _h_ ( _σ_ [2] )�


(12)


    _pσ_ ( _x_ ) = exp _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

2 _σ_ [2]


- �2 _πσ_ [2][�][(] _[n][−][d]_ [)] _[/]_ [2]

��  �� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )��


where _|h_ ( _σ_ [2] ) _|_ is _O_ ( _σ_ ). Now we take logarithmic and use the fact that log( _A_ + _B_ ) = log( _A_ ) +
log(1 + _B/A_ ), we obtain


log _pσ_ ( _x_ )


= _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]


_[ −]_ _[d]_ log(2 _πσ_ [2] ) + log _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ ) _−_ 1 _/_ 2+

2 ��� ���


[P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

+ _[n][ −]_ _[d]_
2 _σ_ [2] 2


log - _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))) + _h_ ( _σ_ [2] )�


= _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]


log(2 _πσ_ [2] ) + log _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ )))+
2


[P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

+ _[n][ −]_ _[d]_
2 _σ_ [2] 2


log _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ ) _−_ 1 _/_ 2 + log �1 + _h_ ( _σ_ [2] )        - _._
��� ��� _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ )))


Therefore, we have


    - _h_ ( _σ_ [2] )
_p_ ˆ( _x, σ_ ) = log 1 +
_p_ data(Φ _[−]_ [1] (P _M_ ( _x_ )))


The remaining is to show that _h_ ( _σ_ [2] ) _/p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))) is uniformly _o_ (1) for all _x_ _∈_ _TM_ ( _ϵ_ ).
Since the manifold is compact, _p_ data( _u_ ) is uniformly bounded away from zero (see Remark B.1).
The remaining is to find a suitable Ω _[′]_ and upper and lower bound the constants in Corollary B.1. We
will discuss this later.


Now let us look at the case of VP. The only difference is in the exponential, we changed from
_∥x −_ Φ( _u_ ) _∥_ [2] to


_∥x −_ ~~�~~ 1 _−_ _σ_ [2] Φ( _u_ ) _∥_ [2] = _∥x −_ Φ( _u_ ) + �1 _−_ �1 _−_ _σ_ [2]       - Φ( _u_ ) _∥_ [2] _._


_√_
If we do a Taylor expansion of 1 _−_


1 _−_ _σ_ [2] :


~~�~~
1 _−_ 1 _−_ _σ_ [2] = [1]

2 _[σ]_ [2][ +] _[ o]_ [(] _[σ]_ [2][)] _[.]_


18


Using this expansion, we have that


_∥x −_ Φ( _u_ ) + �1 _−_ ~~�~~ 1 _−_ _σ_ [2]          - Φ( _u_ ) _∥_ [2]


= _∥x −_ Φ( _u_ ) _∥_ [2] + _σ_ [2] _⟨_ Φ( _u_ ) _, x −_ Φ( _u_ ) _⟩_ + _o_ ( _σ_ [2] ) _⟨x,_ Φ( _u_ ) _⟩._


Then we can use the same argument as in the proof Lemma B.1 to show that the _o_ ( _σ_ [2] ) does not
affect the approximation. Specifically, let


   _J_ := exp _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

2 _σ_ [2]


- �2 _πσ_ [2][�][(] _[n][−][d]_ [)] _[/]_ [2]
_,_

~~�~~

   -    �� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )��


and


1     _K_ := (2 _πσ_ [2] ) _[d/]_ [2] [exp] _−_ _[∥][x][ −]_ 2 [Φ(] _σ_ [2] _[u]_ [)] _[∥]_ [2]


- - exp _−_ [1] _p_ data( _u_ ) _._

2 _[⟨]_ [Φ(] _[u]_ [)] _[, x][ −]_ [Φ(] _[u]_ [)] _[⟩]_


- exp _−_ [1]


We have

      

_M_


_p_ data( _u_ ) _du_


1
(2 _πσ_ [2] ) _[d/]_ [2] [exp]


- _√_

_−_ _[∥][x][ −]_


_√_

_−_ _[∥][x][ −]_


1 _−_ _σ_ [2] Φ( _u_ ) _∥_ [2]

2 _σ_ [2]


 =


    _Kdu_ +
_M_

    _Kdu_ +
_M_


_K_ (exp ( _o_ (1) _⟨_ Φ( _u_ ) _, x⟩_ ) _−_ 1) _du_
_M_


 _≤_


_Ko_ (1) _du_
_M_


  -  -  -  _≤_ _J_ _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))) exp _−_ [1] + _o_ (1) _._

2 _[⟨]_ [P] _[M]_ [(] _[x]_ [)] _[, x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[⟩]_


The rest of the proof follows similarly to the proof of the VE case.


Then we need to discuss the upper and lower bounds in Corollary B.1. For the upper bounds,
since the manifold is compact, there exists such uniform upper bounds for _{ci}_ [6] 1 [(see Remark B.1).]


For the lower bounds we first consider _λ_ min - _H_ ˆ ( _u, x_ )�. The part _[∂]_ [Φ(] _[u]_ [)]


⊺
_∂_ Φ( _u_ )


For the lower bounds we first consider _λ_ min _H_ ( _u, x_ ) . The part _∂u_ _∂u_ is positive def
inite and uniformly bounded away from zero for all _u_ . The eigenvalues of other part, i.e.,

- _∂u∂_ Φ( _i∂uu_ ) _j_ _[,]_ [ Φ(] _[u]_ [)] _[ −]_ _[x]_ �, may be negative. However, as long as its eigenvalues are small enough, by

Weyl’s inequality, we can still lower bound the smallest eigenvalue of _H_ [ˆ] ( _u, x_ ). The eigenvalues of

- _∂u∂_ Φ( _i∂uu_ ) _j_ _[,]_ [ Φ(] _[u]_ [)] _[ −]_ _[x]_ �, can then be bounded by _∥∇_ [2] Φ( _u_ ) _∥∥_ Φ( _u_ ) _−_ _x∥_ . Therefore, as long as the

tubular neighborhood and the set Ω _[′]_ is small enough, we can lower bound _λ_ min - _H_ ˆ ( _u, x_ )�. For


_∂u_


mally, let _G_ _>_ 0 be the lower bound of the smallest eigenvalue of _[∂]_ [Φ(] _[u]_ [)]


⊺
_∂_ Φ( _u_ )


mally, let _G_ _>_ 0 be the lower bound of the smallest eigenvalue of _∂u_ _∂u_ [.] [Let] _[C]_ [2] [be] [the]

uniform upper bound of _∥∇_ [2] Φ( _u_ ) _∥_, and _C_ 1 be that of _∥∇_ Φ( _u_ ) _∥_ . Those constants are uniform for
a fixed finite atlas since the manifold is compact. Let the radius of Ω _[′]_ be _r_ 0. We have that in Ω _[′]_,
_λ_ min - _H_ ˆ ( _u, x_ )� _≥_ _G −_ _C_ 2( _∥_ Φ( _u_ ) _−_ P _M_ ( _x_ )) _∥_ + _∥_ P _M_ ( _x_ ) _−_ _x∥_ ) _≥_ _G −_ _C_ 2( _C_ 1 _r_ 0 + _ϵ_ ). There
fore, we can choose _r_ 0 and _ϵ_ small enough (but away from zero) such that _λ_ min - _H_ ˆ ( _u, x_ )� _≥_ _G/_ 2,

e.g., _ϵ_ is the minimum of _G/_ (4 _C_ 2) and the original _ϵ_ in the tubular neighborhood definition, and
_r_ 0 = _G/_ (4 _C_ 1 _C_ 2). This way, _m_ 1 can be lower bounded by _Gr_ 0 [2] _[/]_ [2][.]


B.3 PROOFS FOR SECTION 4


The results in Appendices B.1 and B.3 consider only points in _TM_ ( _ϵ_ ). Therefore, to use the results,
we need first show that the density outside the tubular neighborhood becomes negligible as _σ_ _→_ 0.
In the following two lemmas, we will show the concentration of the density for _pσ_ and exp( _−fσ_ ).


_∂u_


**Lemma B.3.** _Assume Assumptions 2.1 and 2.2 holds._ _We have that_ lim _σ→_ 0 - _x∈TM_ ( _ϵ_ ) _[p][σ]_ [(] _[x]_ [)] _[dx]_ [ = 1] _[.]_


19


_Proof._ We have that

      
_pσ_ ( _x_ ) _dx_
_x∈_ R _[d]_ _/TM_ ( _ϵ_ )


 =

_x∈_ R _[d]_ _/TM_ ( _ϵ_ )


_u∈M_


1   
_−_ _[∥][x][ −]_ [Φ(] _[u]_ [)] _[∥]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


_p_ data( _u_ ) _dudx_


 =


     _p_ data( _u_ )
_u∈M_


     _p_ data( _u_ )
_u∈M_


_x∈_ R _[d]_ _/TM_ ( _ϵ_ )


_dxdu_


_dxdu,_


 _≤_


_∥x−_ Φ( _u_ ) _∥≥ϵ_


1   
_−_ _[∥][x][ −]_ [Φ(] _[u]_ [)] _[∥]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


1   
_−_ _[∥][x][ −]_ [Φ(] _[u]_ [)] _[∥]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


where the exchange of the integral is justified by Tonelli’s theorem with the non-negativity of the
integrand. The last inequality holds since any point in R _[d]_ _/TM_ ( _ϵ_ ) is at least _ϵ_ away from any point
on the manifold. Now the inner integral is the integral of a Gaussian density with distance to the
origin at least _ϵ_ . It will decay exponentially fast as _σ_ _→_ 0. Let _Z_ be a standard Gaussian random
variable of dimension _d_, and then the above integral is equivalent to

         


      _p_ data( _u_ ) _P_ _∥Z∥≥_ _[ϵ]_
_u∈M_ _σ_


_σ_


- _du_ = _P_ _∥Z∥≥_ _[ϵ]_


_σ_


_._


The RHS can be shown to decay exponentially fast by the Gaussian concentrations.


**Lemma B.4.** _Assume Assumptions 2.1 and 2.2 holds._ _Further assume that_


sup _∥∇fσ_ ( _x_ ) + _∇_ log _pσ_ ( _x_ ) _∥_ = _o_            - _σ_ _[−]_ [2][�]
_x∈K_


_We have that_


             


lim
_σ→_ 0


exp( _−fσ_ ( _x_ )) _dx_ = 0 _._
_x∈K\TM_ ( _ϵ_ )


_Proof._ For _x_ _∈/_ _TM_ ( _ϵ_ ), the points are at least _ϵ_ away from the manifold. Therefore, we have that


- 1 _p_ data( _u_ ) _du_ = _−_ _[ϵ]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


    _pσ_ ( _x_ ) _≤_

_u∈M_


1   
_−_ _[ϵ]_ [2]
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2]


_,_


as _p_ data is a density function. Therefore, we have that

exp( _−fσ_ ( _x_ )) _≤_ 1          - _−_ _[ϵ]_ [2]          - _σ_ _[−]_ [2][��] _,_
(2 _πσ_ [2] ) _[d/]_ [2] [exp] 2 _σ_ [2] [+] _[ o]_


There exists _σ_ 0, such that for all _σ_ _<_ _σ_ 0, the _o_ ( _σ_ _[−]_ [2] ) term is upper bounded by _ϵ_ [2] _/_ 4 _σ_ [2] . Then we
have that

      -      - [2]      


1                  exp( _−fσ_ ( _x_ )) _dx ≤_ Vol( _K_ ) _−_ _[ϵ]_ [2]
_x∈K\TM_ ( _ϵ_ ) (2 _πσ_ [2] ) _[d/]_ [2] [exp] 4 _σ_


4 _σ_ [2]


_._


The RHS goes to zero as _σ_ _→_ 0 as _p_ data is bounded.


Now we are ready to prove our main theorems.


_Proof of Theorem 4.1._ First, since both _fσ_ and log _pσ_ are _C_ [1] functions on _K_, we have the that _L_ _[∞]_
norm of their gradients is the same as the supremum. First we will show that for any _η_ _≥−_ 2,


sup _∥∇fσ_ ( _x_ ) + _∇_ log _pσ_ ( _x_ ) _∥_ = _o_ ( _σ_ _[η]_ ) as _σ_ _→_ 0 _,_
_x∈K_


implies that


sup _|fσ_ ( _x_ ) + log _pσ_ ( _x_ ) _|_ = _o_ ( _σ_ _[η]_ ) as _σ_ _→_ 0 _._
_x∈K_


20


Given our assumption, for any two points _x, y_ _∈_ _K_, there exists a finite length path, say _γx,y_ ( _·_ ) :

[0 _,_ 1] _→_ _K_ with and _∥γ_ _[′]_ _∥_ being upper bounded uniformly. Consider an arbitrary point _x_ 0 _∈_ _K_,
then we have


∆ _σ_ ( _x_ ) := _−fσ_ ( _x_ ) _−_ log _pσ_ ( _x_ )

               - 1
= _−fσ_ ( _x_ 0) _−_ log _pσ_ ( _x_ 0) + ( _−∇fσ_ ( _γ_ ( _t_ )) _−∇_ log _pσ_ ( _γ_ ( _t_ ))) _· γ_ _[′]_ ( _t_ ) _dt_

0

= ∆ _σ_ ( _x_ 0) + _g_ ( _x, σ_ ) _,_


where sup _x|g_ ( _x, σ_ ) _|_ is _o_ ( _σ_ _[η]_ ) uniformly for _x_ _∈_ _K_ according to the assumption. Further, we have
that

      -       


         exp( _−fσ_ ( _x_ )) _dx_ =
_x∈K_


_pσ_ ( _x_ ) exp(∆ _σ_ ( _x_ )) _dx_
_x∈K_


 = _pσ_ ( _x_ ) exp(∆ _σ_ ( _x_ 0) + _g_ ( _x, σ_ )) _dx,_

_x∈K_


which then imply that


      ∆ _σ_ ( _x_ 0) _≥_ log


          exp( _−fσ_ ( _x_ )) _dx −_ log
_x∈K_


_pσ_ ( _x_ ) _dx −_ sup _|g_ ( _x, σ_ ) _|,_
_x∈K_ _x∈K_


and


      ∆ _σ_ ( _x_ 0) _≤_ log


          exp( _−fσ_ ( _x_ )) _dx −_ log
_x∈K_


_pσ_ ( _x_ ) _dx_ + sup _|g_ ( _x, σ_ ) _|._
_x∈K_ _x∈K_


The first two terms on the right-hand side is _o_ (1) as _σ_ _→_ 0 as our assumption about _fσ_ and

- 
_K_ _[p][σ]_ [(] _[x]_ [)] _[dx][ ≥]_ _TM_ ( _ϵ_ ) _[p][σ]_ [(] _[x]_ [)] _[dx][ →]_ [1][ according to Lemma B.3. Thus,] _[ |]_ [∆] _[σ]_ [(] _[x]_ [0][)] _[|]_ [ is] _[ o]_ [(] _[σ][η]_ [)][. Therefore,]

_|_ ∆ _σ_ ( _x_ ) _|_ is _o_ ( _σ_ _[η]_ ) uniformly for _x_ _∈_ _K_ . Further we can apply Lemma B.4 to conclude that the
density of exp( _−fσ_ ) concentrates in _TM_ ( _ϵ_ ) as _σ_ _→_ 0.


Then, we can prove the first conclusion that the support is on the manifold. By the expansion of
log _pσ_ in Theorem B.2, we have that

_fσ_ ( _x_ ) = 1 �1 _/σ_ [2][�] _._
2 _σ_ [2] _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2][ +] _[ o]_

Then we can apply Lemma B.2 with _fθ_ ( _x_ ) = _σ_ [2] _fσ_ ( _x_ ), _θ_ = _σ_ [2] and _η_ = _δ_ [2] _/_ 2 to conclude the claim.


To prove that the limiting distribution is _p_ data on the manifold, we have


1
_fσ_ ( _x_ ) =
2 _σ_ [2] _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2] _[ −]_ [log] _[ p]_ [data][(Φ] _[−]_ [1][(P] _[M]_ [(] _[x]_ [)))+]


log


  -  - _d −_ _n_
�� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )�� + 2 log(2 _πσ_ [2] ) + _o_ (1) _._


Then we can apply Theorem B.1. Then the _f_ 0 becomes the distance function (changed to local

~~�~~
coordinates), and _f_ 1 is _−_ log _p_ data + log ��� _H_ ˆ ( _u_ ) _,_ Φ( _u, r_ ))���, In addition, we note that for _r_ = 0,


  -  �� _H_ ˆ ( _u_ ) _,_ Φ( _u, r_ ))�� = _dM_ ( _u_ ) _/du_, and therefore, we recover _p_ data. The ( _d −_ _n_ ) log(2 _πσ_ 2) term


is simply a constant and does not affect the result after normalization. One can replace _fσ_ with
_fσ_ + _[d][−]_ 2 _[n]_ log(2 _πσ_ [2] ) and then apply Theorem B.1, and this does not change the distribution after

normalization.


What remains is to ensure Assumption B.1 holds, especially the second condition, i.e., to ensure
that the Hessian of _∥_ Φ( _u, r_ ) _−_ Φ( _u_ ) _∥_ [2] _/_ 2 w.r.t. _r_ is uniformly bounded away from zero. We can
write Φ( _u, r_ ) as Φ( _u_ ) + _N_ ( _u_ ) _r_, where _N_ ( _u_ ) is the normal vector field on the manifold _M_ at point
Φ( _u_ ) (Weyl, 1939). We have that


_∂_ _∥_ Φ( _u, r_ ) _−_ Φ( _u_ ) _∥_ [2]

_∂r_ 2


_∂r_


_−_ Φ( _u_ ) _∥_

= _[∂]_ [Φ(] _[u, r]_ [)]
2 _∂r_


⊺
(Φ( _u, r_ ) _−_ Φ( _u_ )) = _N_ ( _u_ ) [⊺] _N_ ( _u_ ) _r_ = _r,_


21


since the columns of _N_ ( _u_ ) are orthonormal. Therefore, the Hessian of _∥_ Φ( _u, r_ ) _−_ Φ( _u_ ) _∥_ [2] _/_ 2 w.r.t.
_r_ is simply the identity matrix, which satisfies the assumption.


To construct a _s_ ( _σ, x_ ) such that the limiting distribution is arbitrarily, say ˆ _π_, we let _s_ ( _σ, x_ ) being the
gradient of


_−_ [1]

2 _σ_ [2] _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2][ + log ˆ] _[π]_ [(Φ] _[−]_ [1][(P] _[M]_ [(] _[x]_ [)))] _[ −]_ [log]


  -  �� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )�� + _o_ (1) _._


The difference between _fσ_ and log _pσ_ is then Ω(1).


B.4 MANIFOLD WKB ANALYSIS OF THE STATIONARY DISTRIBUTION


A key difference between our theorem in Section 5 and the results in Section 4 is that, in the former, the density does not admit an explicit form. When _s_ ( _x, σ_ ) is a gradient field, a closed-form
expression for the density is readily available; however, this property is not guaranteed for most
parameterized models, such as neural networks. We therefore resort to the WKB approximation
to approximate the stationary distribution. Similarly to Appendix B.1, we first present a general
framework and then apply it to our specific setting. We will show that SDE with the following form
admits a stationary distribution of the form Equation (11). Interested readers may refer to Bouchet
& Reygner (2016); Bonnemain & Ullmo (2019) for more details on WKB applied on Fokker-Planck
equation.


We consider the following SDE:


_√_
_dxt_ = _bθ_ ( _xt_ ) _dt_ +


2 _θdWt,_ with _bθ_ ( _x_ ) = _−∇f_ 0( _x_ ) _−_ _θ∇f_ 1( _x_ ) + [ˆ] _b_ ( _x, θ_ ) _,_


or the following SDE with the same stationary distribution,


_√_

_dxt_ = _[b][θ]_ [(] _[x][t]_ [)] _dt_ +

_θ_


2 _dWt._ (13)


We assume that [ˆ] _b_ ( _x, θ_ ) is uniformly _o_ ( _θ_ ) in _TM_ ( _ϵ_ ) as _θ_ _→_ 0. Also, we have arg min _f_ 0( _x_ ) = _M_ .
This framework is general enough to cover the cases of Theorems 5.2 and 6.1. We will see later
that in these two cases, the function _f_ 0 is the distance function to the manifold, and _θ_ will be chosen
differently in different cases. We make the following assumptions about the SDE.


Let _πθ_ ( _x_ ) be the stationary distribution of the SDE Equation (13). First we assume the WKB ansatz:
**Assumption B.2** (Local WKB ansatz) **.** _We assume that_ lim _θ→_ 0 - _TM_ ( _ϵ_ ) _[π][θ]_ [(] _[x]_ [)] _[dx]_ [ = 1] _[, and that][ π][θ]_ [(] _[x]_ [)]

_admits a local WKB form within compact set TM_ ( _ϵ_ ) _:_


    _πθ_ ( _x_ ) _∝_ exp _−_ _[V]_ [ (] _[x]_ [)]

_θ_


_cθ_ ( _x_ ) _with_ _cθ_ ( _x_ ) = _c_ 0( _x_ ) + ˆ _c_ ( _x, θ_ ) _,_


_where_ _c_ 0 _∈_ _C_ [2] ( _TM_ ( _ϵ_ )) _is_ _positive,_ _and_ _cθ_ _→_ _c_ 0 _in_ _C_ [2] ( _TM_ ( _ϵ_ )) _._ _We_ _further_ _assume_ _that_ _V_ _∈_
_C_ [3] ( _TM_ ( _ϵ_ )) _admits a unique solution._


The normalization constant can be explicitly written as

        -         


    exp _−_ _[V]_ [ (] _[x]_ [)]
_x∈TM_ ( _ϵ_ ) _θ_


_cθ_ ( _x_ ) _dx,_


       _πθ_ ( _x_ ) _dx/_
_x∈TM_ ( _ϵ_ )


_θ_


since we have for _x ∈_ _TM_ ( _ϵ_ ),


_πθ_ ( _x_ ) = _πθ_ ( _x_ ) _·_ **1** _TM_ ( _ϵ_ )( _x_ ) = _πθ_ ( _x | x ∈_ _TM_ ( _ϵ_ )) _πθ_ ( _TM_ ( _ϵ_ ))


       _cθ_ ( _x_ ) exp _−_ _[V]_ [ (] _θ_ _[x]_ [)]
=


_[x]_ [)]  
_θ_


_θ_ _[x]_ [)] - _dx_ _πθ_ ( _TM_ ( _ϵ_ )) _._


       _x∈TM_ ( _ϵ_ ) _[c][θ]_ [(] _[x]_ [) exp] _−_ _[V]_ [ (] _θ_ _[x]_ [)]


Our goal would be to solve for _V_ ( _x_ ) and _c_ 0( _x_ ) with the Fokker-Planck equation. Once solved, to
study the limit of _πθ_, we can use results in Appendix B.1 as


    _πθ_ ( _x_ ) _∝_ exp _−_ _[V]_ [ (] _[x]_ [)] _[ −]_ _[θ]_ [ log] _[ c]_ [0][(] _[x]_ [) +] _[ o]_ [(] _[θ]_ [)]

_θ_


22


_._


**Theorem B.3.** _Consider the SDE described in Equation_ (13) _._ _Assume Assumption B.2 holds._ _Then_
_we have that_


_V_ ( _x_ ) = _f_ 0( _x_ ) _,_ _c_ 0( _x_ ) = _C_ exp( _−f_ 1( _x_ )) _,_


_for some constant C._


_Proof._ By Fokker-Planck equation for the stationary distribution, we have that


           0 = div _−bθ_ ( _x_ ) _πθ_ ( _x_ ) + _θ_ _[∂π][θ]_ [(] _[x]_ [)]

_∂x_


By plugging in the WKB ansatz, we have that


(14)


_._


- _−_ 2� _∂cθ_
_∂x_ _[, ][∂V]_ _∂x_


    
_−_ div( _bθ_ ) _cθ −_ _bθ,_ _[∂c][θ]_


[1] _∂V_

_θ_ _∂x_


_[∂c][θ]_

_∂x_ _[−]_ [1] _θ_


_∂x_ _[c][θ]_


- - _∂_ 2 _cθ_
+ _θ_ Tr
_∂x_ [2]


  - _∂_ 2 _V_

_−_ Tr
_∂x_ [2]


_cθ_ + [1]

_θ_


_∂V_
���� _∂x_


_∂V_
���� _∂x_


2
_cθ_ = 0 _._

����


_cθ_ + [1]


Next by the method of WKB, we will equate different orders of _θ_ in the above equation to solve
for _V_ ( _x_ ) and _c_ 0( _x_ ), starting from the lowest order _θ_ _[−]_ [1] . It is easier to show a function is constant,
therefore, for _c_ 0, we will define ˜ _c_ 0( _x_ ) = exp( _f_ 1( _x_ )) _c_ 0( _x_ ), and try to show that it is constant.


**Order** _θ_ _[−]_ [1] In this order, we have that

          - _∂f_ 0
_∂x_ _[, ][∂V]_ _∂x_


- _∂V_
=
���� _∂x_


2
_._

����


This corresponds to the Hamilton-Jacobi equation typically appears in the WKB approximation. The
equation gives the solution for _V_ ( _x_ ) as _V_ ( _x_ ) = _f_ 0( _x_ ). Plugging this solution into Equation (14),
we can get


[2] _[f]_ [1]

_[∂]_ [ˆ] _[b]_
_∂x_ [2] [+] _∂x_


  _cθ −_ _bθ,_ _[∂c][θ]_

_∂x_


_∂x_


_∂f_ 0

[1]

_θ_ _∂x_


- + _−θ_ _[∂f]_ [1]


_[∂f]_ [1] [1]

_∂x_ [+ ˆ] _[b,]_ _θ_


_∂x_ _[c][θ]_


_−_ Tr


_−θ_ _[∂]_ [2] _[f]_ [1]


- - _∂_ 2 _cθ_
+ _θ_ Tr
_∂x_ [2]


_−_ 2� _∂cθ_
_∂x_ _[, ][∂f]_ _∂x_ [0]


= 0 _._


We will work with this equation for equating the higher orders.


**Order** _θ_ [0] In this order, we have that

        - _∂f_ 1
_∂x_ _[, ][∂f]_ _∂x_ [0]


- - _∂c_ 0
_c_ 0 +
_∂x_ _[, ][∂f]_ _∂x_ [0]


= 0 _._


This is known as the transport equation (Bouchet & Reygner, 2016). It shows how _c_ 0 changes along
the gradient of _f_ 0. Next, we express the equation in terms of ˜ _c_ 0:

           - _∂c_ ˜0           
= 0 _._ (15)

_∂x_ _[, ][∂f]_ _∂x_ [0]


This implies that along the gradient of _f_ 0, ˜ _c_ 0 is constant. Since the manifold _M_ consists of the minimizers of _f_ 0, for any point _x_ in _K_, the value of ˜ _c_ 0( _x_ ) is the same as the value at the corresponding
minimizer _y_ on _M_ following the gradient flow of _f_ 0. Formally, we have


_c_ ˜0( _x_ ) = ˜ _c_ 0( _ψ_ _[x]_ (+ _∞_ )) _,_


where _ψ_ _[x]_ ( _t_ ) follows _dψ_ _[x]_ ( _t_ ) _/dt_ = _−∇f_ 0( _ψ_ _[x]_ ( _t_ )) with _ψ_ _[x]_ (0) = _x_ given the initial condition
_ψ_ _[x]_ (0) = _x_ . Therefore, we see that to solve for _c_ ˜0, we need to know the value of it on _M_ . We
find that the next order equation will help us to solve for ˜ _c_ 0 on _M_ .


23


_∂x_


= 0 _._ (15)


**Order** _θ_ [1] In this order, if we directly find all terms in Equation (14) that are of order _θ_ [1], we will
find that it includes higher order terms, e.g., ˆ _c_ ( _x, θ_ ). However, since we only care about the solution
on _M_, we evaluate the equation on _M_ and interestingly find that it does not include such higher
order terms, as crucially the factor _∂f_ 0 _/∂x_ becomes 0 at _M_ . Specifically, for _x ∈M_, we have that


 - _∂_ 2 _f_ 1
Tr
_∂x_ [2]


- - _∂f_ 1
_c_ 0 +
_∂x_ _[, ][∂c]_ _∂x_ [0]


- - _∂_ 2 _c_ 0
+ Tr
_∂x_ [2]


= 0 _._


Replacing _c_ 0 with ˜ _c_ 0 exp( _−f_ 1), we have that


 - _∂_ 2 _c_ ˜0  -  - _∂c_ ˜0
Tr _−_
_∂x_ [2] _∂x_ _[, ][∂f]_ _∂x_ [1]


= 0 _._


Our goal here would be to solve for _c_ ˜0 on _M_, and apparently it would be helpful to convert the
equation to the local coordinates and establish a PDE for the manifold chart coordinate _u_ .


**Local coordinates** We convert the above order _θ_ [1] equation about ˜ _c_ 0 to the local coordinates _z_ =
( _u, r_ ) and get that for _r_ = 0, i.e., points on _M_,


- _−_ - _∂c_ ˜0 _[∂f]_ [1]
_∂z_ _[, G][−]_ [1] _∂z_


0 = [1]

_|J|_ [div] _[z]_


- _|J|G_ _[−]_ [1] _[∂][c]_ [˜][0]

_∂z_


- - _[∂]_ [2] _[c]_ [˜][0]
+ Tr _|J|G_ _[−]_ [1]

_∂z_ [2]


- - _[∂]_ [2] _[c]_ [˜][0]
+ Tr _|J|G_ _[−]_ [1]


�� _−_ - _∂c_ ˜0 _[∂f]_ [1]
_∂z_ _[, G][−]_ [1] _∂z_


(16)

_,_


= [1]

_|J|_


��
div _z_  - _|J|G_ _[−]_ [1][�] _,_ _[∂][c]_ [˜][0]

_∂z_


where _G_ = _J_ [⊺] _J_ and the divergence of a matrix is understood as the divergence of the column
vectors. Note that we cannot simply conclude from the above equation that ˜ _c_ 0 is constant, by say, the
strong maximum principle, since the gradients of ˜ _c_ 0 include not only the manifold chart coordinate
_u_ but also the normal coordinate _r_ . Therefore, we have to further derive a PDE about _u_ and any
gradients of _c_ ˜0 w.r.t. _r_ should be replaced by known functions. Fortunately those gradients can be
solved by the equation we obtain at order _θ_ [0] .


First, let us derive from Equation (16) a PDE about _u_ :


**Lemma B.5.** _From Equation_ (16) _, we have that for r_ = 0 _,_


    - _∂c_ ˜0
∆ _Mc_ ˜0( _u_ ) _−_
_∂u_ _[, g][−]_ [1] _[ ∂f]_ _∂u_ [1]


- 1
+

 - _|g|_


- _∂|J|_
_∂r_ _[, ][∂]_ _∂r_ _[c]_ [˜][0]


- _−_ - _∂c_ ˜0
_∂r_ _[, ][∂f]_ _∂r_ [1]


- - _∂_ 2 _c_ ˜0
+ Tr
_∂r_ [2]


= 0 _,_ (17)


_where_ ∆ _M is the Laplace-Beltrami operator on M._


_Proof._ Let the index _i, j_ when showing at _∂_ be derivatives w.r.t. the _i_ or _j_ -th coordinate of _u_, and
let _p, q_ be the derivatives w.r.t. _r_ respectively. Any index variable that is not explicitly defined is
understood to be summed over. From Equation (16), by carefully expanding the divergence, the
inner product term becomes
�div   - _|J|G_ _[−]_ [1][�] _, ∇c_ ˜0��� _r_ =0
= ~~�~~ _|g|∂j_ - _g_ _[−]_ [1][�] _[∂][i][c]_ [˜][0][ +] - _g_ _[−]_ [1][�] _[∂][j]_ - _|g|∂ic_ ˜0 _−_ ~~�~~ _|g|_ - _g_ _[−]_ [1][�] _[∂][p][d][p,k][|]_ _[∂][i][c]_ [˜][0][ +] _[ ∂][p][|][J][|][∂][p][c]_ [˜][0] _[.]_


_i,j_ _[∂][j]_ - _|g|∂ic_ ˜0 _−_ ~~�~~ _|g|_ - _g_ _[−]_ [1][�]


     - _g_ _[−]_ [1][�]
_i,j_ _[∂][i][c]_ [˜][0][ +]


_i,k_ _[∂][p][d][p,k][|]_ _r_ =0 _[∂][i][c]_ [˜][0][ +] _[ ∂][p][|][J][|][∂][p][c]_ [˜][0] _[.]_


For the trace term, we have

Tr        - _|J|G_ _[−]_ [1] _∇_ [2] _c_ ˜0��� _r_ =0 [=] ~~�~~ _|g|_        - _g_ _[−]_ [1][�] _i,j_ _[∂][j,i][c]_ [˜][0][ +] ~~�~~ _|g|∂p,pc_ ˜0 _._

Now we look at Equation (17). From the definition of Laplace-Beltrami operator, we have

∆ _Mc_ ˆ0( _u_ ) = ~~�~~ 1 _∂i_ �� _|g|_   - _g_ _[−]_ [1][�] _i,j_ _[∂][j][c]_ [˜][0]   _|g|_


= ~~�~~ 1 _|g|_ _∂i_ - _|g|_ - _g_ _[−]_ [1][�]


      - _g_ _[−]_ [1][�]
_i,j_ _[∂][j][c]_ [˜][0][ +] _[ ∂][i]_


     - _g_ _[−]_ [1][�]
_i,j_ _[∂][j][c]_ [˜][0][ +]


_i,j_ _[∂][i,j][c]_ [˜][0] _[.]_


             - _g_ _[−]_ [1] 0
Since _G_ _[−]_ [1] evaluated at _r_ = 0 is
0 _I_


�, the term _−_ - _∂c_ ˜0


_[∂f]_ _∂z_ [1] - in Equation (16) matches


_c_ ˜0 _[∂f]_ [1]

_∂z_ _[, G][−]_ [1] _∂z_


_−_ - _∂c_ ˜0


_−_ - _∂∂uc_ ˜0 _[, g][−]_ [1] _[∂f]_ _∂u_ [1] - _−_ - _∂∂rc_ ˜0 _[,]_ _[∂f]_ _∂r_ [1] - in Equation (17). Now compare the terms of Equation (17) and

Equation (16), the only remaining term is

                  - _g_ _[−]_ [1][�] _[∂][p][d][p,k][|]_ _[∂][i][c]_ [˜][0] _[,]_


_[∂f]_ _∂u_ [1] - _−_ - _∂∂rc_ ˜0


_c_ ˜0 _[∂f]_ [1]

_∂u_ _[, g][−]_ [1] _∂u_


_c_ ˜0 _[∂f]_ [1]

_∂r_ _[,]_ _∂r_


_i,k_ _[∂][p][d][p,k][|]_ _r_ =0 _[∂][i][c]_ [˜][0] _[,]_


24


which we will prove is 0. We will show that [�] _p_ _[∂][p][d][p,k]_ ��� _r_ =0 [= 0][.]


Since the columns of _N_ are orthonormal, we have for any _p_, [�] _i_ [(] _[N][i,p]_ [)][2] [= 1][.] [Taking derivative for]

both sides to _uj_, we have for any _p, j_, [�] _i_ _[N][i,p][∂][j][N][i,p]_ [=] [0][.] [We also have by definition that for any]


Since the columns of _N_ are orthonormal, we have for any _p_, [�]


both sides to _uj_, we have for any _p, j_, [�] _i_ _[N][i,p][∂][j][N][i,p]_ [=] [0][.] [We also have by definition that for any]

_p, j_,

                 - _N_ [⊺] _∇N_ _r_                 - [=] _[ N][i,p][∂][j][N][i,l][r][l][.]_


_p,j_ [=] _[ N][i,p][∂][j][N][i,l][r][l][.]_


Using the above two results, we have for any _j_,

   -    


_Ni,p∂jNi,p_ = 0 _._

_p_


_∂pdp,j_ =  
_p_ _p_


_∂p_ ( _Ni,p∂jNi,lrl_ ) =  
_p_ _p_


From Equation (17), we see that it contains gradients of ˜ _c_ 0 w.r.t. _r_, which we will solve by the order
_θ_ [0] equation.


**Lemma B.6.** _From Equation_ (15) _, we have that on the manifold M,_
_∂c_ ˜0       - _∂_ 2 _c_ ˜0       -       


_[c]_ [˜][0]  
_,_
_∂u_ [(] _[u,]_ [ 0)]


_c_ ˜0 - _∂_ 2 _c_ ˜0

_and_ Tr
_∂r_ [(] _[u,]_ [ 0) = 0] _[,]_ _∂r_ [2]


 - ( _u,_ 0) = _h_ ( _u_ ) _,_ _[∂][c]_ [˜][0]


_where h_ ( _u_ ) _does not contain the unknown function_ ˜ _c_ 0 _._


_Proof._ Since we care about the evaluation of the equation on _M_, we start by changing the coordinates to the local coordinates _z_ = ( _u, r_ ) from Equation (15) to get that

          - _∂c_ ˜0 _[∂f]_ [0]           
= 0 _._

_∂z_ _[, G][−]_ [1] _∂z_


_∂z_


= 0 _._


Next, we compute the gradient w.r.t. _z_ :


- _T_ - _∂f_ 0
_∂z_ _[⊗]_ _[∂]_ _∂z_ _[c]_ [˜][0]


_∂z_ [+]


_∂_ [2] _c_ ˜0


_[ ∂f]_ [0]

_[∂]_ [2] _[f]_ [0]
_∂z_ [+] _∂z_ [2]


_∂_ _c_ ˜0

_∂z_ [2] _[G][−]_ [1] _[ ∂f]_ _∂z_ [0]


_[f]_ [0]

_[∂][c]_ [˜][0]
_∂z_ [2] _[G][−]_ [1] _∂z_


- _∂_ vec - _G_ _[−]_ [1][�]


_∂z_


= 0 _,_ (18)


where _⊗_ is the Kronecker product. When we evaluate this equation at _r_ = 0, the factor _∂f_ 0 _/∂r_


          - _g_ _[−]_ [1] 0
becomes 0, _G_ _[−]_ [1] ( _u,_ 0) =
0 _I_


_∂z_ [2] _[f]_ [2][0] [(] _[u,]_ [ 0) =] �00 _∂_ [2] _f_ 0 _/∂r_ 0 [2] ( _u,_ 0)


and _[∂]_ [2] _[f]_ [2][0]


_._ Then we have


_∂_ [2] _f_ 0


_∂r_ [(] _[u,]_ [ 0) = 0] _[.]_


_f_ 0

_∂r_ [2] [(] _[u,]_ [ 0)] _[∂]_ _∂r_ _[c]_ [˜][0]


Since _[∂]_ [2] _[f]_ [2][0]


[0]

_∂r_ [(] _[u,]_ [ 0) = 0][.]


_∂r_ _[f]_ [2][0] [(] _[u,]_ [ 0)][ is full-rank, we have that] _[∂]_ _∂r_ _[c]_ [˜][0]


Next, we compute gradient again for Equation (18), and evaluate at _r_ = 0. Ignoring _∂f_ 0 _/∂z_ which
is 0, we have the _i, j_ -th element of the matrix is

     - _∂_ 2 _c_ ˜0 _[∂]_ [2] _[f]_ [0]      -      - _∂_ 2 _f_ 0 _[∂]_ [2] _[c]_ [˜][0]      - _∂_ [3] _f_ 0      - _[∂][c]_ [˜][0]      _∂z_ [2] _[G][−]_ [1] _∂z_ [2] + _∂z_ [2] _[G][−]_ [1] _∂z_ [2] + _∂zi∂zk∂zj_ _G_ _[−]_ [1] _∂z_


- _[∂][c]_ [˜][0]
_G_ _[−]_ [1]


_∂z_ [2]


  - _∂_ 2 _f_ 0 _[∂]_ [2] _[c]_ [˜][0]
_i,j_ + _∂z_ [2] _[G][−]_ [1] _∂z_ [2]


_∂z_ [2]


_∂_ [3] _f_ 0
+
_i,j_ _∂zi∂zk∂zj_


_∂z_


_k_


(19)


_∂_ [2] _f_ 0
= 0 _,_
_∂zp∂zj_


+ _[∂]_ [2] _[f]_ [0]

_∂zi∂zk_


+ _[∂]_ [2] _[f]_ [0]


_∂G_ _[−]_ _k,p_ [1]
_∂zj_


_∂c_ ˜0
+ _[∂][c]_ [˜][0]
_∂zp_ _∂zk_


_∂c_ ˜0
+ _[∂][c]_ [˜][0]
_∂zp_ _∂zk_


_∂G_ _[−]_ _k,p_ [1]
_∂zi_


where _∂c_ ˜0 _/∂r_ is 0. The first two terms have nice structure when evaluated at _r_ = 0, as


_∂_ [2] _f_ 0
and


_∂_ [2] _f_ 0 _[∂]_ [2] _[c]_ [˜][0]

_∂z_ [2] _[G][−]_ [1] _∂z_ [2]


_∂_ [2] _c_ ˜0


_∂_ [2] _c_ ˜0 _[∂]_ [2] _[f]_ [0]

_∂z_ [2] _[G][−]_ [1] _∂z_ [2]


_∂z_ [2] [=]


_[∂]_ [2] _[c]_ [˜][0] - 0 0

_∂z_ [2] [=] _∂_ [2] _f_ [2] 0 _∂_ [2] _c_ ˜0 _∂_ [2] _f_ [2] 0


[2] _f_ 0 _∂_ [2] _c_ ˜0 _∂_ [2] _f_ 0

_∂r_ [2] _∂r∂u_ _∂r_ [2]


_∂r_ [2] _f_ [2] 0 _[∂]_ _∂r_ [2] _[c]_ [˜][2][0]


_∂r_ [2]


�0 _∂u∂r∂_ [2] _c_ ˜0 _∂∂r_ [2] _f_ [2] 0


_∂u∂r_ _∂r_ [2]

0 _∂_ [2] _c_ ˜ [2] 0 _∂_ [2] _f_ [2] 0


[2] _c_ ˜0 _∂_ [2] _f_ 0

_∂r_ [2] _∂r_ [2]


_∂r_ [2]


_._


�0 0


We then multiply Equation (19) by matrix


0 - _∂_ [2] _f_ [2] 0


0  
_∂_ [2] _f_ 0 - _−_ 1

_∂r_ [2]


from the left, and get


�0 0


[2] _f_ 0  - _−_ 1 _∂_ 2 _c_ ˜0 _∂_ [2] _f_ 0

_∂r_ [2] _∂r_ [2] _∂r_ [2]


+ remaining terms = 0 _._


0 - _∂_ [2] _f_ [2] 0


_∂r_ [2]


- - 0 0

+ _∂_ [2] _c_ ˜0 _∂_ [2] _c_ ˜0
_∂r∂u_ _∂r_ [2]


Since _∂c_ ˜0 _/∂r_ is 0, the element of the remaining terms all have one and only one factor of _∂c_ ˜0 _/∂ui_
for some _i_ . Taking the trace of the above equation, and we have proved the second statement.


25


Now we plug in Lemma B.6 to Lemma B.5, and obtain a PDE about ˜ _c_ 0( _·,_ 0) on _u_ whose second order
derivatives are the Laplace-Beltrami operator, and the zero-th order term, i.e., the term that includes
the function value ˜ _c_ 0( _·,_ 0), is 0. Therefore, we can conclude by strong maximum principle (Gilbarg
et al., 1977, Theorem 3.5) that ˜ _c_ 0( _·,_ 0) is a constant. According to the equation at order _θ_ [0], we obtain
that ˜ _c_ 0 off-manifold is the same constant.


B.5 PROOF FOR SECTION 5


We will first prove Theorem 5.1, which follows similar proof technique as Theorem 4.1, and then
turn to the harder case of Theorem 5.2.


_Proof of Theorem 5.1._ The proof follows the same as Theorem 4.1, except that now we use Theorem B.1 with _θ_ = _σ_ [2] _[−][α]_ . In this case, _f_ 0( _x_ ) = _∥x −_ P _M_ ( _x_ ) _∥_ [2] _/_ 2, _f_ 1 _≡_ 0 and all other terms are
asymptotically small compared to _σ_ [2] _[−][α]_ . According to the proof of Theorem 4.1, the determinant
of the Hessian of _f_ 0 in the normal direction is the same for all _u_, therefore, we recover the uniform
distribution on the manifold.


The only thing remains to verify is to ensure


  
lim _π_ ˜ _σ_ ( _x_ ) _dx_ = lim
_σ→_ 0 R _[d]_ _\TM_ ( _ϵ_ ) _σ→_ 0


R _[d]_ _\TM_ ( _ϵ_ ) [exp(] _[−][σ][α][f][σ]_ [(] _[x]_ [))] _[dx]_


= 0 _._
R _[d]_ [ exp(] _[−][σ][α][f][σ]_ [(] _[x]_ [))] _[dx]_


Since we have lim _σ→_ 0 


Since we have lim _σ→_ 0 - _K_ _[π]_ [˜][(] _[x]_ [)] _[dx][ →]_ [1][, we only need to consider within] _[ K]_ [.] [For the numerator, we]

can do similarly as Lemma B.4 to obtain

  -  -  - _σ_ _[α]_  - [2]


 - _ϵ_ [2]
exp _−_


4 _σϵ_ [2][2] _[−][α]_ [+] _[ o]_ - _σ_ _[α]_ [+] _[β]_ [��] _,_


           - 1
exp( _−σ_ _[α]_ _fσ_ ( _x_ )) _dx ≤_ Vol( _K_ )
_K\TM_ ( _ϵ_ ) (2 _πσ_ [2] ) _[d/]_ [2]


- _σ_ _[α]_


where 2 _−_ _α_ _>_ 0 and _α_ + _β_ _>_ 0. There exists _σ_ 0, such that for all _σ_ _<_ _σ_ 0, the _o_ ( _σ_ _[α]_ [+] _[β]_ ) term is
upper bounded by _ϵ_ [2] _/_ 8 _σ_ [2] _[−][α]_ . Then we have the numerator upper bounded by


   - 1   - _σ_ _[α]_
Vol( _K_ )
(2 _πσ_ [2] ) _[d/]_ [2]


 - _ϵ_ [2]
exp _−_

8 _σ_ [2] _[−][α]_


_._


For the denominator, it is lower bounded by

       -       -       - _σ_


  exp _−_ _[∥][x][ −]_ [Φ(] _[x]_ [)] _[∥]_ [2] + _o_ - _σ_ _[α]_ [+] _[β]_ [��] _dx_

2 _σ_ [2] _[−][α]_


- _σ_ _[α]_


_TM_ ( _ϵ/_ 2)


 _≥_

_TM_ ( _ϵ/_ 2)


- 1
(2 _πσ_ [2] ) _[d/]_ [2]


- 1
(2 _πσ_ [2] ) _[d/]_ [2]


- _σ_ _[α]_


exp - _−_ 8 _σϵ_ [2][2] _[−][α]_ [+] _[ o]_ - _σ_ _[α]_ [+] _[β]_ [��] _dx._


There exists _σ_ 1, such that for all _σ_ _<_ _σ_ 1, the _o_ ( _σ_ _[α]_ [+] _[β]_ ) term is lower bounded by _ϵ_ [2] _/_ 16 _σ_ [2] _[−][α]_ . Then
the denominator is lower bounded by


 - _ϵ_ [2]
exp _−_

16 _σ_ [2] _[−][α]_


_._


           - 1
Vol( _TM_ ( _ϵ/_ 2))
(2 _πσ_ [2] ) _[d/]_ [2]


Therefore, the ratio is upper bounded by


- _σ_ _[α]_


Vol( _K_ )  - _ϵ_ [2]

_−_
Vol( _TM_ ( _ϵ/_ 2)) [exp] 16 _σ_ [2] _[−][α]_


Vol( _K_ )  - _ϵ_ [2]

_−_
Vol( _TM_ ( _ϵ/_ 2)) [exp] 16 _σ_ [2]


_,_


which goes to zero as _σ_ _→_ 0.


Next, for Theorem 5.2, we use results in Appendix B.4 to find an approximate stationary distribution
of the SDEs considered in Section 5, and then use results in Appendix B.1 to prove the main theorem.


_Proof of Theorem 5.2._ The SDE we consider can be also written as


_√_

_dXt_ = _[σ]_ [2] _[s]_ [(] _[X][t][, σ]_ [)] _dt_ +

_σ_ [2] _[−][α]_


26


2 _dWt,_


Therefore, we want to apply Theorem B.3 with _θ_ = _σ_ [2] _[−][α]_ and _bθ_ = _σ_ [2] _s_ ( _Xt, σ_ ). We assert that
under our assumption of Theorem 5.2, we can write

_bθ_ ( _x_ ) = _−_ _[∂][∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2] _[/]_ [2] + _o_              - _σ_ [2] _[−][α]_ [�] _,_

_∂x_

meaning that _f_ 0 = _∥x −_ P _M_ ( _x_ ) _∥_ [2] _/_ 2 and _f_ 1 _≡_ 0. We will discuss the proof of this later. If we have
the above, by Theorem B.3, the stationary distribution in _TM_ ( _ϵ_ ) is given by


           -            _πσ_ ( _x_ ) _∝_ exp _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [))] _[∥]_ [2] _[/]_ [2] + _o_ (1) _,_

_σ_ [2] _[−][α]_


where the error in the prefactor is equivalent to the error in the exponent. The remaining proof
follows the same as Theorem 5.1.


It remains to prove the assertion about _bθ_ . A sufficient condition is that


1 _∂∥x −_ P _M_ ( _x_ ) _∥_ [2] _/_ 2

sup _∇_ log _pσ_ ( _x_ ) +
_x∈TM_ ( _ϵ_ )���� _σ_ [2] _∂x_


= _O_ (1) _._ (20)
����


Because if Equation (20) holds, we have uniformly for any _x ∈_ _TM_ ( _ϵ_ ),

_∂∥x −_ P _M_ ( _x_ ) _∥_ 2 _/_ 2
_bθ_ ( _x_ ) +
���� _∂x_ ����


= _σ_ 2 _s_ ( _x, σ_ ) + _∂∥x −_ P _M_ ( _x_ ) _∥_ 2 _/_ 2
���� _∂x_


����


= _σ_ 2 _s_ ( _x, σ_ ) _−_ _σ_ 2 _∇_ log _pσ_ ( _x_ ) + _σ_ 2 _∇_ log _pσ_ ( _x_ ) + _∂∥x −_ P _M_ ( _x_ ) _∥_ 2 _/_ 2
���� _∂x_


����


_≤_ �� _σ_ 2 _s_ ( _x, σ_ ) _−_ _σ_ 2 _∇_ log _pσ_ ( _x_ )�� + _σ_ 2 _∇_ log _pσ_ ( _x_ ) + _∂∥x −_ P _M_ ( _x_ ) _∥_ 2 _/_ 2
���� _∂x_

= _o_ ( _σ_ [2+] _[β]_ ) + _O_ ( _σ_ [2] )

= _o_ ( _σ_ [2] _[−][α]_ ) _,_


����


where the last inequality holds because _α >_ max _{−β,_ 0 _}_ . In the theorem, we assumed _L_ _[∞]_ ( _TM_ ( _ϵ_ ))
norm, which is the same as sup _x∈TM_ ( _ϵ_ ) since _s_ ( _x, σ_ ) and _∇_ log _pσ_ ( _x_ ) are continuous.


Therefore, it remains to prove Equation (20). We will prove for the case of VE, and the case of VP
holds with similar argument. The gradient of the distance function can be written as:


�⊺ [�]


_∂∥x −_ P _M_ ( _x_ ) _∥_ [2] _/_ 2

=
_∂x_


- - _∂_ P _M_ ( _x_ )

_I_ _−_
_∂x_


( _x −_ P _M_ ( _x_ )) = _x −_ P _M_ ( _x_ ) _,_


where the last equality holds because _x −_ P _M_ ( _x_ ) is orthogonal to the manifold and the image of
_∂_ P _∂xM_ ( _x_ ) is in the tangent space of the manifold (Leobacher & Steinicke, 2021). Then note that


[Φ(] _[u]_ [)] _[−][x]_
_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _σ_ [2]


_[N]_ [(] _[x]_ [;] _[ u, σ]_ _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _σ_ [2] _du_

- [2]


_∇_ log _pσ_ ( _x_ ) = _[∇][p][σ]_ [(] _[x]_ [)] =

_pσ_ ( _x_ )


_._
_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _[du]_


For the denominator, follow the same as in the proof of Theorem B.2 to obtain that


    _pσ_ ( _x_ ) = exp _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

2 _σ_ [2]


- [�] 2 _πσ_ [2][�][(] _[n][−][d]_ [)] _[/]_ [2] _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))
(1 + _O_ ( _σ_ )) _,_
��   �� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )��


since Equation (12) holds and _p_ data is uniformly bounded away from zero. We could do the same
for the numerator, however, the _O_ ( _σ_ ) error is not enough here. Intuitively, the numerator would be


 exp _−_ _[∥][x][ −]_ [P] _[M]_ [(] _[x]_ [)] _[∥]_ [2]

2 _σ_ [2]


- [�] 2 _πσ_ [2][�][(] _[n][−][d]_ [)] _[/]_ [2] _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))

~~�~~

       -       �� _H_ ˆ (Φ _−_ 1(P _M_ ( _x_ )) _, x_ )��


27


- P _M_ ( _x_ ) _−_ _x_ + _O_ (1 _/σ_ ) _._
_σ_ [2]


Apparently, the error term is not enough to prove Equation (20).


Therefore, we turn to stronger Laplace’s method result that has an error term of _O_ - _σ_ [2][�], i.e., the _h_ ( _θ_ )
_√_
term in Corollary B.1 could be improved to _O_ ( _θ_ ) instead of _O_ ( _θ_ ). However, such result should

have the cost of requiring the function _F_ (as the notation use in Corollary B.1) to be _C_ [4] and _g_ to be
_C_ [2], a stronger condition [1] . Formally, we have that


_σ_ [2] _∇_ log _pσ_ ( _x_ ) + ( _x −_ P _M_ ( _x_ ))


=


_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [) (Φ(] _[u]_ [)] _[ −]_ [P] _[M]_ [(] _[x]_ [))] _[ du]_

    - _,_

_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _[du]_


and we want to prove its _L_ _[∞]_ ( _TM_ ( _ϵ_ )) norm is _O_ (1). For any _x_ _∈_ _TM_ ( _ϵ_ ) and _v_ _∈{v_ _|_ _∥v∥_ = 1 _}_,
we have that

_v_ [⊺] [�] _σ_ [2] _∇_ log _pσ_ ( _x_ ) + ( _x −_ P _M_ ( _x_ ))�


_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _[v]_ [⊺] [(Φ(] _[u]_ [)] _[ −]_ [P] _[M]_ [(] _[x]_ [))] _[ du]_

    
_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _[du]_

_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [) (] _[v]_ [⊺] [(Φ(] _[u]_ [)] _[ −]_ [P] _[M]_ [(] _[x]_ [)) + 1)] _[ du]_

      - _−_ 1 _._

_M_ _[N]_ [(] _[x]_ [;] _[ u, σ]_ [2] _[I]_ [)] _[p]_ [data][(] _[u]_ [)] _[du]_


=


=


The last step where we add 1 is a simple trick because the Laplace’s method we will use does
not allow the prefactor to be 0 at the minimizer. Next, we multiply the numerator and de


P _M_ ( _x_ ) _∥_ [2] - ~~[�]~~ _|H_ [ˆ] (Φ _[−]_ [1] (P _M_ ( _x_ )) _,x_ ) _|_

2 _σ_ [2] (2 _πσ_ [2] ) [(] _[n][−][d]_ [)] _[/]_ [2]


nominator by exp - _∥x−_ P _M_ [2] ( _x_ ) _∥_ [2]


, so that their limit does not diminishing to
(2 _πσ_ [2] ) [(] _[n][−][d]_ [)] _[/]_ [2]


0. For the numerator, we apply Majerski (2015, Theorem 2.4) with their _n_ = 1 _/σ_ [2], _t_ = _u_,
_f_ ( _u_ ) = _∥x −_ Φ( _u_ ) _∥_ [2] _/_ 2, _α_ = 2 ( _f_ ( _u_ ) is _C_ [4] since Φ( _u_ ) is _C_ [4] ), _Bδ_ can be selected the same
as in the proof of Theorem B.2, _g_ ( _u_ ) = _p_ data( _u_ )( _v_ [⊺] (Φ( _u_ ) _−_ P _M_ ( _x_ )) + 1) (g(u) is _C_ [2] since
_p_ data( _u_ ) is _C_ [2] ), and the minimizer is Φ _[−]_ [1] (P _M_ ( _x_ )). The upper boundedness of the constants can
be easily verified by compactness and one can show that they are uniform for _x_ and _v_ . Crucially,
_g_ (Φ _[−]_ [1] (P _M_ ( _x_ ))) = _p_ data(Φ _[−]_ [1] (P _M_ ( _x_ ))) is uniformly lower bounded. The lower boundedness of
_λ_ min can be reasoned in the same way as in the proof of Theorem B.2. Therefore, we have

_v_ [⊺] [�] _σ_ [2] _∇_ log _pσ_ ( _x_ ) + ( _x −_ P _M_ ( _x_ ))� = [1 +] _[ O]_ [(] _[σ]_ [2][)]

1 + _O_ ( _σ_ [2] ) _[−]_ [1 =] _[ O]_ [(] _[σ]_ [2][)] _[.]_


Since the bound is uniformly for _x_ and _∥v∥_ = 1, we have that


sup �� _σ_ 2 _∇_ log _pσ_ ( _x_ ) + ( _x −_ P _M_ ( _x_ ))��
_x∈TM_ ( _ϵ_ )


_≤_ sup sup _v_ [⊺] [�] _σ_ [2] _∇_ log _pσ_ ( _x_ ) + ( _x −_ P _M_ ( _x_ ))� = _O_ ( _σ_ [2] ) _,_
_x∈TM_ ( _ϵ_ ) _∥v∥_ =1


which proves Equation (20).


B.6 PROOF FOR SECTION 6


_Proof of Theorem 6.1._ The proof follows the same as Theorem 5.2, except that now we have _f_ 1 = _v_
when applying Theorem B.3 and Theorem B.1.


C EXPERIMENTAL DETAILS AND FURTHER EXPERIMENTS


C.1 NUMERICAL SIMULATIONS ON ELLIPSE


**Loss function.** In our experiments, we train the score network to predict


_s_ ˆ( _x, σ_ ) := _σ_ [2] _s_ ( _x, σ_ ) _,_


1Weaker condition such as _C_ 1 _,_ 1 is also possible, see Majerski (2015, Theorem 2.4).


28


instead of _s_ ( _x, σ_ ) directly. This formulation is more stable across noise levels, since the leading
term in the score expansion is of order 1 _/σ_ [2], making _s_ ˆ( _x, σ_ ) an _O_ (1) target. With this choice, the
training objective becomes


1
2 [E] _[u][∼][p]_ [data][E] _[x][∼N]_ [ (Φ(] _[u]_ [)] _[,σ]_ [2] _[I]_ [)]


- 2 [�]
_σ_ [2][ ��] - _s_ ( _x, σ_ ) + _[x][−]_ _σ_ [Φ(][2] _[u]_ [)] ���


= 2 [1] [E] _[u][∼][p]_ [data][E] _[x][∼N]_ [ (Φ(] _[u]_ [)] _[,σ]_ [2] _[I]_ [)]      - _σ_ 1 [2] _[∥][s]_ [ˆ][(] _[x, σ]_ [) +] _[ x][ −]_ [Φ(] _[u]_ [)] _[∥]_ [2][�] _._

The score function _s_ is parameterized by a neural network consisting of four transformer blocks,
each with hidden dimension 128.


**Data and noise.** Training data is generated from a von Mises distribution with parameter _κ_ = 1.
The injected Gaussian noise variance _σ_ [2] is sampled from a range _σ_ _∈_ [0 _._ 01 _,_ 50].


**Optimization.** We use AdamW with weight decay 1 _×_ 10 _[−]_ [4] and global gradient clipping at norm
1 _._ 0. The initial learning rate is 3 _×_ 10 _[−]_ [3], decayed cosine-schedule over 4 _×_ 10 _[−]_ [4] steps down to
1% of its initial value, after which training continues with a constant learning rate of 4 _×_ 10 _[−]_ [4] . The
batch size is set to 1024.


**Sampling.** For sampling, we run Langevin dynamics


           _dxt_ = _s_ ˆ( _xt, σ_ min) _dt_ + 2 _σ_ min [2] _[dW][t][,]_


with _σ_ min = 0 _._ 01. This process has the same stationary distribution as


_√_
_dxt_ = _s_ ( _xt, σ_ min) _dt_ +


2 _dWt._


            For the TS Langevin dynamics, the diffusion coefficient is 2 _σ_ min [2] _[−][α]_ [instead of] �2 _σ_ min [2] [. We employ]
the Euler–Maruyama scheme with a step size of 0 _._ 1, running 10 _,_ 000 steps with 10 _,_ 000 runs.


C.2 IMAGE GENERATION WITH DIFFUSION MODELS


**Algorithm** **details.** We use a pre-trained Stable Diffusion 1.5 model with a DDPM sampler in a
predictor–corrector (PC) scheme. The pre-trained network provides a denoiser _ϵ_ ( _x, t, y_ ), and the
corresponding classifier-free guidance (CFG) score at time _t_ is


_st_ ( _x, y_ ) = _∇x_ log _pt_ ( _x_ )

     - ��      unconditional score


+ _w_ - _∇x_ log _pt_ ( _x | y_ ) _−∇x_ log _pt_ ( _x_ )�


   - ��   conditional increment


= _−_ [1]


_σt_


- _ϵ_ ( _x, t, ∅_ ) + _w_ - _ϵ_ ( _x, t, y_ ) _−_ _ϵ_ ( _x, t, ∅_ )�� _,_


where _y_ is the conditioning input (prompt embedding), _w_ is the guidance scale, _σt_ = _[√]_ 1 _−_ _α_ ¯ _t_, and
_α_ ¯ _t_ is as in Ho et al. (2020). Our tempered-score framework applies to this PC sampler by modifying
only the unconditional component while leaving the guided increment unchanged:


_s_ ˜ _t_ ( _x, y_ ) = _−_ [1]

_σt_


- _σt_ _[α]_ _[ϵ]_ [(] _[x, t,][ ∅]_ [)] [+] _[w]_ - _ϵ_ ( _x, t, y_ ) _−_ _ϵ_ ( _x, t, ∅_ )�� _,_


which is consistent with Equation (8). Let _{ti}_ denote the discrete reverse-time schedule. After each
DDPM predictor update at level _ti_, we perform _n_ corr _._ _corrector_ steps of Langevin dynamics with
the tempered score:


_xk_ +1 = _xk_ + _δi_ ˜ _sti_ ( _xk, y_ ) + �2 _δi ξk,_ _ξk_ _∼N_ (0 _, I_ ) _,_


where the step size _δi_ follows Song et al. (2021, Algorithm 5). After the entire reverse process, we
apply an additional _n_ corr _._ deterministic projection steps using the unconditional score (no guidance,
no noise) to further project onto the data manifold:


_dxτ_ = _∇_ log _pt_ 0( _xτ_ ) _dτ._


We use the same number of projection steps for both the original PC baseline and our TS to ensure
a fair comparison.


29


0.8


0.6


0.4


0.2


0.0


0.8


0.6


0.4


0.2


0.0


|pdata|Col2|
|---|---|
||3<br><br>2|


Angle (rad)


Angle (rad)


~~2~~


(a) Distribution generated by Diffusion Model


(b) Distribution generated by TS-1


Figure 4: Comparison of distributions generated with VE diffusion model versus our TS Langevin
dynamics Equation (8) with _α_ = 1.


**Hyperparameter** **setting.** We adopt the default configuration of Stable Diffusion 1.5 [(https:](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[//huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) Unless otherwise noted, all results in Section 7.2 use guidance scale _w_ = 7 _._ 5 and 30 inference steps.
For the best-results reported in Table 1, we perform a grid search over the number of corrector steps
in _{_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 30 _}_ and _α_ _∈{_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}_ . The original PC baseline is tuned over the same
set of numbers of corrector step for fairness. For CLIP evaluations, we generate 512 images per
setting and downscale each to 256 _×_ 256 before computing the scores.


C.3 CONTROLLED EXPERIMENT WITH GROUND TRUTH SCORES


To empirically validate the rate separation results in Theorems 4.1 and 5.1, we designed a controlled
experiment using synthetic data where the manifold and ground truth scores are known analytically.


We consider the unit circle manifold _M_ = _{x_ _∈_ R [2] _|_ _∥x∥_ = 1 _}_ with a Von Mises distribution
_p_ data( _θ_ ) _∝_ exp( _κ_ cos( _θ −_ _θ_ 0)), where we used _κ_ = 4 and _θ_ 0 = _π_ . This setup allows us to compute
the analytic ground truth score _s_ _[∗]_ ( _x, σ_ ). We then inject a deterministic error field _e_ ( _x_ ) into the true
score:


�1
_x −_
���� 0


�����


4 [�]


_s_ ˆ( _x, σ_ ) = _s_ _[∗]_ ( _x, σ_ ) + _e_ ( _x_ ) _,_ with _e_ ( _x_ ) = _−∇_


The magnitude of this error term _e_ ( _x_ ) is _O_ (1) with respect to _σ_ .


1

2


_._


We compare the performance of the standard reverse diffusion process against our proposed TS
Langevin dynamics using this corrupted score _s_ ˆ. As shown in Figure 4, the standard reverse diffusion process using _s_ ˆ produces samples that deviate significantly from the ground truth _p_ data, confirming that _O_ (1) score errors are sufficient to corrupt distributional recovery, while the TS Langevin
dynamics with _α_ = 1 robustly recovers the uniform distribution on the circle.


C.4 SENSITIVITY ANALYSIS OF HYPERPARAMETER _α_


To evaluate the sensitivity of the hyperparameter _α_, we performed an ablation study using the
Stable Diffusion 1.5 model, under the same setting as in Section 7.2 of our paper. We tested
_α_ _∈{_ 0 _,_ 0 _._ 1 _,_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}_ across three prompt categories, with the number of corrector steps fixed
at 10 and 20. Note that _α_ = 0 corresponds to the standard predictor-corrector baseline.


As shown in Tables 3 and 4, our method yields consistent improvements over the baseline ( _α_ = 0)
once _α_ is sufficiently large ( _α_ _≥_ 0 _._ 5), demonstrating that the performance gains are robust and
not limited to a narrow hyperparameter setting. The performance is particularly stable for _α_ _∈_

[1 _._ 0 _,_ 1 _._ 5], which aligns well with our theoretical framework (Theorems 5.1 and 5.2) that guarantees
convergence to the uniform distribution for any _α_ _<_ 2. While we utilized _α_ = 1 in Table 2 for
simplicity, these results suggest that slightly more aggressive tempering ( _α_ = 1 _._ 5) can provide
further gains in diversity and quality.


30


_α_ = 0 _α_ = 0 _._ 1 _α_ = 0 _._ 5 _α_ = 1 _._ 0 _α_ = 1 _._ 5


**Prompt** P-sim _↑_ I-sim _↓_ P-sim I-sim P-sim I-sim P-sim I-sim P-sim I-sim


**Architecture** 27.13 81.81 27.12 81.73 27.14 81.67 27.27 81.57 **27.32** **81.52**
**Furniture** 29.30 81.24 29.32 81.37 29.33 81.06 29.58 80.95 **30.16** **80.76**
**Car** 26.30 87.57 26.30 87.58 26.31 87.44 26.37 87.42 **26.50** **87.34**


Table 3: Ablation of _α_ for 10 corrector steps.


_α_ = 0 _α_ = 0 _._ 1 _α_ = 0 _._ 5 _α_ = 1 _._ 0 _α_ = 1 _._ 5


**Prompt** P-sim _↑_ I-sim _↓_ P-sim I-sim P-sim I-sim P-sim I-sim P-sim I-sim


**Architecture** 26.87 81.60 26.85 81.56 26.97 81.49 27.06 **80.97** **27.10** 81.13
**Furniture** 28.98 81.72 28.99 81.65 29.07 81.40 29.52 **81.15** **30.20** 81.39
**Car** 26.26 88.06 26.26 88.09 26.25 87.95 26.28 88.07 **26.62** **87.70**


Table 4: Ablation of _α_ 20 corrector steps.


D CONVERGENCE OF TS LANGEVIN


In this section, we deduce the mixing time analysis, i.e. the convergence analysis for a stochastic
process, of the TS Langevin to the estimation of the Poincar´e constant. The goal is to show that
TS Langevin is not necessarily slower—and can in fact be significantly faster—than the standard
Langevin dynamics in terms of mixing time. To carry out such an analysis, we assume that the score
network is a gradient field, i.e. _s_ ( _·, σ_ ) = _∇_ log _pθ_ for some parameterized density function. WLOG,
we assume _pθ_ is normalized as the normalizing factor does not affect the velocity field _s_ .


D.1 CONVERGENCE ANALYSIS OF LANGEVIN DYNAMICS USING FUNCTIONAL INEQUALITY


To analyze the convergence of Langevin dynamics, it is customary to use a functional inequality
satisfied by the invariant measure _p∞_ of the Langevin dynamics (Here, _pt_ denotes the density of
the process at time _t_, and _p∞_ is its stationary distribution. This notation differs from _pθ_, and the
distinction should be clear from context). In this response, we focus on the Poincar´e inequality (PI):
We say _p∞_ satisfies PI( _C_ PI) if for all _f_ _∈_ _H_ [1] ( _p∞_ ) (Sobolev space weighted by _p∞_ ),
�� _f_ _−_        - _fdp∞_ �2 _dp∞_ _≤_ 1 ��� _∇f_ ��2 _dp∞,_
_C_ PI

where we call _C_ PI _>_ 0 is the Poincar´e constant.


Consider the overdamped Langevin dynamics with potential _Uσ_ : R _[d]_ _→_ R:


_√_
_dX_ ( _t_ ) = _−∇Uσ_ ( _X_ ( _t_ )) _dt_ +


2 _dW_ ( _t_ ) _,_


and let _pt_ = Law( _X_ ( _t_ )). Under mild assumptions, _p∞_ _∝_ exp( _−Uσ_ ) is the unique invariance
measure of the above dynamics. If _p∞_ _∝_ exp( _−Uσ_ ) satisfies PI( _C_ PI), then

_χ_ [2] ( _pt, p∞_ ) _≤_ _e_ _[−][C]_ [PI] _[t]_ _χ_ [2] ( _p_ 0 _, p∞_ ) _,_

where _χ_ [2] denotes the _χ_ [2] -divergence. In particular, to ensure _χ_ [2] ( _pt, p∞_ ) _≤_ _η_ for some target accuracy _η_ _>_ 0, it suffices to take _t_ = _O_ ( _C_ 1PI [log] _η_ [1] [)][.] _[Thus, the larger the Poincar´e constant, the faster]_

_the convergence._


D.2 ANALYZING THE EFFECT OF DRIFT SCALING TO THE POINCARE´ CONSTANT.


Under the assumptions of our paper, the comparison between the mixing of standard Langevin and
TS Langevin therefore reduces to comparing their Poincar´e constants. We illustrate how drift scaling
affects the Poincar´e constant in the simple case where the data manifold is the unit circle:


_M_ = _{x ∈_ R _[d]_ : _∥x∥_ = 1 _}._


31


In this case, the squared distance function can be computed in a closed form:


_d_ ( _x_ ) = [1]


[1] _x_ [=] [1]

2 _[∥][x][ −]_ _∥x∥_ _[∥]_ [2] 2


[1] [1]

2 [dist][2][(] _[x,][ M]_ [) =] 2


2 [(] _[∥][x][∥−]_ [1)][2] _[.]_


Following section 5 of our paper, we assume the score error is _O_ ( _σ_ _[β]_ ) for some _−_ 2 _< β_ _<_ 0. Recall
that we assume the learned score is a gradient field, i.e. _s_ ( _·, σ_ ) = _∇_ log _pθ_ . Let us further suppose
that the problem dimension is _d_ = 2, i.e. _x ∈_ R [2], and the density function _pθ_ (corresponding to the
learned score _s_ ( _·, σ_ )) has the following form

_−_ log _pθ_ = [1]

_σ_ [2] _[d]_ [(] _[x]_ [) +] _[ σ][β][ϕ]_ [(] _[x]_ [)] _[,]_ [ where] _[ ϕ]_ [(] _[x]_ [) = (] _[|][x]_ [1] _[| −]_ [1)][2] _[,]_


where _x_ 1 denotes the first coordinate of _x_ . Clearly, this function satisfies all requirement in our
paper. Crucially, such a construction ensures that the score error is _O_ ( _σ_ _[β]_ ).


**Standard Langevin dynamics.** We restate the standard Langevin dynamics for the ease of reference: _√_
_dX_ ( _t_ ) = _∇_ log _pθ_ ( _X_ ( _t_ )) _dt_ + 2 _dW_ ( _t_ ) _._

Without temperature scaling, the error function _ϕ_ ( _x_ ) introduces two separated modes ( _−_ 1 _,_ 0) and
(+1 _,_ 0). For such a multimodal measure, classical Eyring-Kramers law or the large deviation principle results imply that the Poincar´e constant can scale as


_C_ PI [LD] [=] _[ O]_ [(exp]           - _−σ_ _[β]_ [�] ) _._


Consequently, the mixing time of the original Langevin dynamics can become _exponentially large_
as _σ_ _→_ 0.


**TS Langevin.** We restate the standard Langevin dynamics for the ease of reference:


_√_
_dX_ ( _t_ ) = _σ_ _[α]_ _∇_ log _pθ_ ( _X_ ( _t_ )) _dt_ +


_√_
2 _dW_ ( _t_ ) = _∇_ log _p_ _[σ]_ _θ_ _[α]_ [(] _[X]_ [(] _[t]_ [))] _[dt]_ [ +]


2 _dW_ ( _t_ ) _._


Under mild conditions, the unique equilibrium measure is _p_ _[σ]_ _θ_ _[α]_ [.] [We] [show] [that,] [under] [our] [standing]
assumptions and _α_ _>_ _−β_, that its Poincar´e constant, denoted as _C_ PI [TS][,] [is] _[ uniformly bounded away]_
_from zero_, _independent of σ_ for sufficiently small _σ_ . Here we summarize the main steps:


- Recall the Holley–Stroock perturbation principle (Holley & Stroock, 1987): Let _U_ and _U_ [˜]
be two potential functions defined on R _[d]_ . Suppose that the corresponding Gibbs measures
_p∞_ _∝_ exp( _−U_ ) and _p_ ˜ _∞_ _∝_ exp( _−U_ [˜] ) satisfy Poincar´e inequality with constants _C_ PI and _C_ [˜] PI
respectively. One has
_C_ ˜PI _≥_ exp( _−osc_ ( ˜ _U, U_ )) _C_ PI _,_

where the oscillation between _U_ and _U_ [˜] is defined as

_osc_ ( _U, U_ [˜] ) := sup _[−]_ _[U]_ [)] _[ −]_ [inf] _[−]_ _[U]_ [)] _[.]_
_x∈_ R _[d]_ [( ˜] _[U]_ _x∈_ R _[d]_ [( ˜] _[U]_

Since 2 _>_ _α_ _>_ _−β_, a Holley–Stroock perturbation argument implies that the PI constant of _p_ _[σ]_ _θ_ _[α]_
is comparable (up to a fixed factor) to that of the measure _µd_ _∝_ exp( _−d_ ( _x_ ) _/σ_ [2] _[−][α]_ ) for small _σ_ .
We denote the Poincar´e constant of this ideal potential as _C_ PI [dist][.]
A short proof for the above statement: Pick

_U_ ˜ = log _p_ _[σ]_ _θ_ _[α]_ and _U_ = _d_ ( _x_ ) _/σ_ [2] _[−][α]_ _._

One can bound _osc_ ( _U, U_ [˜] ) using Theorem 3.1 of our submission. Apply the above principle to
yield
_C_ PI [TS] _[≥]_ [exp(] _[−][O]_ [(] _[σ][α]_ [+] _[β]_ [))] _[C]_ PI [dist] _≥_ exp( _−_ 1) _C_ PI [dist] _[,]_
for a sufficiently small _σ_ .

- We note that the distance function _d_ ( _x_ ) is locally Polyak–Łojasiewicz, and hence one can expect
the recent results (Gong et al., 2024) on the temperature-independent Poincar´e constant for locally
log-PL measure can be applied. The only requirement in (Gong et al., 2024) that is not satisfied
by _µd_ is that it is not _C_ [2] at _x_ = 0.


32


- We therefore introduce a smoothed potential


_Vc_ ( _x_ ) := _[∥][x][∥]_ [2]


 - _∥x∥_ [2] + _c_ [2] _,_
2 _[−]_


_[∥]_

+ [1]
2 2


and apply Holley–Stroock again to compare the PI constant of _µd_ with that of _µc_ _∝_
exp( _−Vc/σ_ [2] _[−][α]_ ). Choosing _c_ = _σ_ [3] _[−][α]_, we can verify that _Vc_ satisfies the assumptions of the
log-PL result (Gong et al., 2024), which implies that the corresponding Poincar´e constant (denoted as _C_ PI [smooth] ) is _independent of σ_ .
A short proof to bound _C_ PI [dist] with _C_ PI [smooth] : Pick

_U_ ˜ ( _x_ ) = _d_ ( _x_ ) _/σ_ [2] _[−][α]_ and _U_ ( _x_ ) = _Vc_ ( _x_ ) _/σ_ [2] _[−][α]_ _._


To bound _osc_ ( _U, U_ [˜] ), notice that

_|d_ ( _x_ ) _−_ _Vc_ ( _x_ ) _|_ = _|∥x∥−_       - _∥x∥_ [2] + _c_ [2] _|_ = _∥x∥_ +       - _c∥_ [2] _x∥_ [2] + _c_ [2] _[≤]_ _[c]_ [ =] _[ σ]_ [3] _[−][α][.]_


Apply the perturbation principle to yield


_C_ PI [dist] _≥_ exp( _−O_ ( _σ_ )) _C_ PI [smooth] _≥_ exp( _−_ 1) _C_ PI [smooth] _,_


for a sufficiently small _σ_ .

- Combining these comparisons shows that the Poincar´e constant of _p_ _[σ]_ _θ_ _[α]_ [,] [i.e.,] _[C]_ PI [TS][,] [differs] [from]
_C_ PI [dist] and _C_ PI [smooth] only by a constant factor.

- In this point, we discuss on proving _C_ PI [smooth] is independent of _σ_ . First, we note that directly
apply the result in (Gong et al., 2024) on the potential _Vc_ already yields that the Poincar´e constant
_C_ PI [smooth] is of order Ω( _c_ ): It is easy to verify the assumptions in (Gong et al., 2024), i.e. local PL,
non-saddle point, growth condition beyond a compact set, and the boundedness of _|_ ∆ _Vc|_, i.e. the
absolution value of the Laplacian of _Vc_ within a compact set. We can hence directly use Theorem 2
in (Gong et al., 2024). However, the quantity _|_ ∆ _Vc|_ is of order [1] _c_ [in this vanilla analysis and hence]

we would yield that the Poincar´e constant _C_ PI [smooth] is of order Ω( _c_ ). It turns out that by exploiting
the particular structure of _Vc_, we can further improve this result: We note that _|_ ∆ _Vc|_ does _not_ need
to hold in the neighborhood of the local maximum set and their analysis still goes through. We
hence pick this neighborhood as a ball centered around the local maximum _x_ = 0 with radius 0 _._ 1.
One can see that outside of this neighborhood but within a compact set, _|_ ∆ _Vc|_ is bounded by a
_σ_ -independent constant. Then _C_ PI [smooth] could be proved to be Ω(1). We highlight that even the
vanilla Ω( _c_ ) bound already establishes the exponential difference between _C_ PI [TS] [(lower] [bounded]
by a polynomial in _σ_ ) and _C_ PI [LD] [(upper bounded by exponential of] _[ −]_ [1] _[/poly]_ [(] _[σ]_ [)][).] [Of course,] [the]
Ω(1) one leads to even bigger separation.


Putting these estimates together, we see that, at least in this unit-circle example, _TS Langevin mixes_
_strictly_ _faster_ than the original Langevin dynamics in the small- _σ_ regime. This illustrates that
temperature-scaled Langevin is not necessarily slower—and can in fact be significantly faster—than
the standard Langevin dynamics in terms of mixing time.


D.3 A REFINED ANALYSIS FOR _C_ PI [smooth]


Directly applying the result in (Gong et al., 2024), we have that _C_ PI [smooth] = Ω( _σ_ [1]


Directly applying the result in (Gong et al., 2024), we have that _C_ PI = Ω( _σ_ [)][ for a sufficiently]

small _σ_ . In this subsection, we show that this can be improved to _C_ PI [smooth] = Ω(1) with a small
modification to the analysis of the Lyapunov function in (Gong et al., 2024).

**Proposition D.1.** _(Menz & Schlichting, 2014, Theorem 3.8) Consider the Langevin dynamics_


_√_
_dX_ ( _t_ ) = _−∇V_ ( _X_ ( _t_ )) _dt_ +


2 _ϵdW_ ( _t_ ) _._


_Define the associated infinitesimal generator L as_


_L_ := _−∇V_ _· ∇_ + _ϵ_ ∆ (21)


_A function W_ : R _[d]_ _→_ [1 _, ∞_ ) _is a_ Lyapunov function _for L if there exists U_ _⊆_ R _[d]_ _, b_ _>_ 0 _, σ_ _>_ 0 _,_
_such that_
_∀x ∈_ R _[d]_ _,_ _ϵ_ _[−]_ [1] _LW_ ( _x_ ) _≤−σW_ ( _x_ ) + _b_ 1 _U_ ( _x_ ) _._ (22)


33


_Given_ _the_ _existence_ _of_ _such_ _a_ _Lyapunov_ _function_ _W,_ _if_ _one_ _further_ _has_ _that_ _the_ _truncated_ _Gibbs_
_measure µϵ,U_ _satisfies PI with constant_ PI _ϵ,U_ _>_ 0 _, the Gibbs measure µϵ_ _satisfies PI with constant_


_σ_
_ρϵ_ _≥_ _ρϵ,U_ _._ (23)
_b_ + _ρϵ,U_


In the context of this section, _ϵ_ = _σ_ [2] _[−][α]_ and _V_ = _Vc_ . In (Gong et al., 2024), the Lyapunov function
is chosen to be _W_ = exp( 2 _[V]_ _ϵ_ [)][ and eq. (22) can be simplified to]


_LW_


_[V]_

2 _ϵ_ _[−]_ _[|∇]_ 4 _[V]_ _ϵ_ [2] _[ |]_ [2]


_LW_ [∆] _[V]_

_ϵW_ [=] 2 _ϵ_


_≤−_ _σ_ + _b_ 1 _U_ _._ (24)
4 _ϵ_ [2]


To establish the above inequality, Gong et al. (2024) partition the whole domain R _[d]_ into multiple
disjoint parts: (1) _U_, (2) a neighborhood of the global minimum but outside of _U_, (3) neighborhoods
of local maximum, (4) beyond a compact set that contains all critical points, and (5) the rest. We
discuss our treatment of each subdomain.


- On (1), we follow the choice of _U_ in (Gong et al., 2024) so the local Poincar´e inequality there
directly holds.


- On (2), i.e. in the neighborhood of the global minimum (note that under the assumptions of (Gong
et al., 2024), all local minima are global minima), but outside of the neighborhood _U_, we follow
the argument as (Gong et al., 2024).


- On (4), Beyond a compact set that contains all the local minima and maximum, we can verify that
_Vc_ above fulfilles the requirements of _V_ in (Gong et al., 2024) and hence the argument directly
carries over.


- On (3), i.e. in a neighborhood of the local maximum, since the Laplacian is already negative, one
can directly obtain eq. (24). Note that we will pick this neighborhood to be the ball centered at
_x_ = 0 with radius 0 _._ 1 for _Vc_, denoted by B(0 _,_ 0 _._ 1).


- On (5), i.e. within the said compact set, but outside of the neighborhoods of the global minimum
and local maximum, (Gong et al., 2024) requires the Laplacian to be bounded. We note that the
analysis in (Gong et al., 2024) is a bit loose and they require the boundedness to hold on the whole
compact set. However, there is no need to assume the boundedness of the Laplacian on B(0 _,_ 0 _._ 1)
as eq. (24) is already established in (3).


Based on the above discussion, we notice that the global bound on the Laplacian of _Vc_ is only
required within a compact set, but outside of B(0 _,_ 0 _._ 1), which is hence a constant independent of _ϵ_ .
We hence obtain the Ω(1) bound on the Poincar´e constant.


34