# SYMMETRY-AWARE BAYESIAN OPTIMIZATION VIA MAX KERNELS


**Anthony Bardou**
School of Computer and Communication Sciences
EPFL
Lausanne, Switzerland
anthony.bardou@epfl.ch


**Aryan Ahadinia & Patrick Thiran**
School of Computer and Communication Sciences
EPFL
Lausanne, Switzerland
_{_ aryan.ahadinia,patrick.thiran _}_ @epfl.ch


ABSTRACT


**Antoine Gonon**
Institute of Mathematics
EPFL
Lausanne, Switzerland
antoine.gonon@epfl.ch


Bayesian Optimization (BO) is a powerful framework for optimizing noisy,
expensive-to-evaluate black-box functions. When the objective exhibits invariances under a group action, exploiting these symmetries can substantially improve
BO efficiency. While using maximum similarity across group orbits has long
been considered in other domains, the fact that the max kernel is not positive
semidefinite (PSD) has prevented its use in BO. In this work, we revisit this idea
by considering a PSD projection of the max kernel. Compared to existing invariant (and non-invariant) kernels, we show it achieves significantly lower regret on
both synthetic and real-world BO benchmarks, without increasing computational
complexity. **Code.** [github.com/abardou/max-kernel.](https://github.com/abardou/max-kernel)


1 INTRODUCTION


Many real-world problems can be framed as the optimization of a noisy, expensive-to-evaluate
black-box function _f_ _[⋆]_ : _S_ _⊂_ R _[d]_ _→_ R. Bayesian Optimization (BO) provides a principled and
sample-efficient framework for tackling this problem, with asymptotic guarantees of global optimality
complementing its empirical success. As a result, BO has been widely adopted across diverse domains
such as robotics (Lizotte et al., 2007), computational biology (Gonzalez et al., 2015) and computer
networks (Bardou et al., 2025).


For a black-box function _f_ _[⋆]_ belonging to the Reproducing Kernel Hilbert Space (RKHS) _Hk_
associated with a kernel _k_ : _S_ _× S_ _→_ R, BO proceeds by placing a Gaussian Process (GP) prior
_f_ _∼GP_ (0 _, k_ ) over functions in _Hk_ . The kernel _k_ determines the covariance structure of the GP and
thus encodes prior assumptions about _f_ _[⋆]_ . Incorporating suitable prior knowledge can substantially
improve convergence and sample efficiency. In many applications, the objective is known to be
invariant under the action of a group _G_, that is,

_f_ _[⋆]_ ( _**x**_ ) = _f_ _[⋆]_ ( _g_ _**x**_ ) for all _g_ _∈G._

For instance, in molecular property prediction, _f_ _[⋆]_ may be invariant to rotations of the underlying
molecular structure (Glielmo et al., 2017). In such cases, designing kernels that explicitly incorporate
_G_ -invariance becomes essential. Ginsbourger et al. (2012) showed that for a centered GP to be
_G_ -invariant, its covariance function must also be invariant under _G_ . Motivated by this, we revisit a
simple idea— _keep the best alignment over each orbit_ —and apply it to BO.


Given a base kernel _k_ b and a symmetry group _G_, define

_k_ max( _**x**_ _,_ _**x**_ _[′]_ ) = max             - _g_ _**x**_ _,_ _g_ _[′]_ _**x**_ _[′]_ [�] _,_ (1)
_g,g_ _[′]_ _∈G_ _[k]_ [b]

so that the similarity between _**x**_ and _**x**_ _[′]_ is the best alignment over their orbits.


1


kmax(x, x [′] )


~~||~~ x ~~||~~ 2


Figure 1: (Left) Colored level sets of a two-dimensional function _f_ _[⋆]_ ( _**x**_ ) invariant under planar
rotations (see (16)): if _∥_ _**x**_ _∥_ 2 = _∥_ _**x**_ _[′]_ _∥_ 2, then _f_ _[⋆]_ ( _**x**_ ) = _f_ _[⋆]_ ( _**x**_ _[′]_ ). (Center/Right) Rotation-invariant
kernels derived from an RBF base kernel (lengthscale 1 _/_ 2), visualized as a function of ( _∥_ _**x**_ _∥_ 2 _, ∥_ _**x**_ _[′]_ _∥_ 2).
_k_ max (right) captures the correct invariance, while _k_ avg (center) only approximates it.


The intuition for using the max-alignment is that when the objective is invariant under a group of
transformations, two inputs can become very similar _after_ applying the right group element, even if
they differ a lot in their original positions. For instance, in an image-based problem with rotation
invariance, two rotated images of the same object (e.g., cats) should in principle be treated similarly
by the optimizer since they correspond to the same objective value. However, most rotations will not
align the images well; and if the optimizer compares images with _ℓ_ [2] distances, only a small number
of them can give a good match. In such settings, taking the _maximum_ similarity over all group actions
is natural: among all transformations, typically only one (or a few) reveal a true alignment. Averaging
over all rotations would dilute this information—most transformed pairs look different—whereas
the max retains the one transformation that matters. This “best-alignment” principle is the core
motivation behind _k_ max and is expected to provide a clearer signal to the optimizer about which
inputs should be treated similarly, compared, e.g., to an averaging approach.


While _k_ max is symmetric and _G_ -invariant, it is however not guaranteed to be positive semi-definite
(PSD), a property required for the standard Gaussian-process machinery underlying BO (see Appendix C.1). To address this, we introduce a PSD version of _k_ max.


**A PSD, invariant surrogate via projection + Nystrom.¨** On a finite design set _D_, we form the Gram
matrix of _k_ max and project it onto the PSD cone (eigenvalue clipping), obtaining _**K**_ +. Denoting by
_**K**_ + _[†]_ [the Moore-Penrose pseudo-inverse of] _**[ K]**_ [+][, we then define the] _[ G]_ [-invariant, PSD kernel]

_k_ + [(] _[D]_ [)][(] _**[x]**_ _[,]_ _**[ x]**_ _[′]_ [)] [=] _[k]_ [max][(] _**[x]**_ _[,][ D]_ [)] _**[ K]**_ + _[†]_ _[k]_ [max][(] _[D][,]_ _**[ x]**_ _[′]_ [)] _[.]_ (2)

Equivalently, _k_ + [(] _[D]_ [)][(] _**[x]**_ _[,]_ _**[ x]**_ _[′]_ [)] [=] _[ϕ]_ [(] _**[x]**_ [)] _[⊤][ϕ]_ [(] _**[x]**_ _[′]_ [)] [with] [features] _[ϕ]_ [(] _**[x]**_ [)] [=] _**[K]**_ + _[†][/]_ [2] _k_ max( _D,_ _**x**_ ), which makes
positive semidefiniteness immediate. By construction, _k_ + [(] _[D]_ [)] (i) coincides with _k_ max on _D_ whenever
_k_ max is already PSD, and (ii) has per-iteration asymptotic cost comparable to orbit-averaged kernels;
details in Section 3.2.


**Results.** The max-alignment heuristic does translate into concrete benefits for BO, which we observe
throughout the paper. The resulting kernel is geometrically better aligned with the true structure of
the problem (Figures 1 and 2). In practice, this makes (i) the acquisition function more faithful as it
avoids redundant exploration of points that are already explored up to symmetry, and (ii) uncertainty
modeling also more faithful: it gains confidence in unexplored regions that correspond to symmetryequivalent points. Across synthetic benchmarks with finite and continuous groups, a wireless-network
design task and a particle packing problem, we show that _k_ + [(] _[D]_ [)] consistently attains lower cumulative
and simple regret than both the base kernel and the orbit-averaged alternative, with gains increasing
with _|G|_ .


**Relation with spectral-based theory.** Mainstream BO theory links fast eigendecay of the kernel to
small regret upper bounds (Srinivas et al., 2012; Valko et al., 2013; Scarlett et al., 2017; Whitehouse
et al., 2023). Surprisingly, we find the opposite trend in our setting: _k_ + [(] _[D]_ [)] typically has a _slower_
empirical eigendecay than _k_ avg, yet consistently achieves _better_ _(lower)_ regret in practice. This
directly challenges the usual spectral intuition: our results reveal a clear mismatch between spectral
predictions and empirical performance, suggesting that eigendecay alone does not capture the
advantages of _k_ + [(] _[D]_ [)][.] [As] [we] [discuss] [later,] [geometric] [considerations] [(the] [alignment] [of] [the] [kernel]


2


eigenvectors with the directions that matter for optimization) and approximation hardness of the
blackbox _f_ _[⋆]_ in the RKHS likely play an essential role beyond pure spectral rates.


**Summary of the contributions.** We propose _k_ max as a _max-alignment_ route to _G_ -invariance, turn it
into a valid GP kernel for BO via PSD projection and Nystrom, and show¨ _k_ + [(] _[D]_ [)] is _G_ -invariant, equals
_k_ max on _D_ when _k_ max is PSD, and matches the asymptotic cost of orbit-averaged kernels (Section 3).
We demonstrate consistent BO gains over orbit averaging across BO benchmarks (Section 4), and we
analyze why eigendecay alone does not explain these gains (Section 5).


2 BACKGROUND


2.1 BAYESIAN OPTIMIZATION IN A NUTSHELL


**Problem.** We seek to maximize an expensive-to-evaluate, black-box objective _f_ _[⋆]_ : _S →_ R under the
assumption that _f_ _[⋆]_ is in the RKHS _Hk_ of a PSD kernel _k_ : _S × S_ _→_ R. Each query _**x**_ _∈S_ returns a
noisy observation _y_ = _f_ _[⋆]_ ( _**x**_ ) + _ε_, where _ε ∼N_ (0 _, σ_ 0 [2][)][.] [Let] _[ Z][t]_ [=] _[ {]_ [(] _**[x]**_ _[i][, y][i]_ [)] _[}][t]_ _i_ =1 [denote the dataset]
after _t_ evaluations, and write _Dt_ = ( _**x**_ 1 _, . . .,_ _**x**_ _t_ ) and _**y**_ _t_ = ( _y_ 1 _, . . ., yt_ ) _[⊤]_ .


**Surrogate model:** **the GP prior.** BO maintains a probabilistic surrogate _f_ over functions in _Hk_ to
guide sampling of new queries _**x**_ _∈S_ with the goal of converging to arg max _x∈S f_ _[⋆]_ ( _**x**_ ). A common
choice is a zero-mean Gaussian process (GP) (Rasmussen & Williams, 2006),


_f_ _∼GP_ (0 _, k_ ) _,_


Conditionally on the dataset of queried points _Zt_ after _t_ evaluations, the posterior _f_ _| Zt_ is still a GP
with posterior mean and covariance


_µt_ ( _**x**_ ) = _k_ ( _**x**_ _, Dt_ )            - _**K**_ _t_ + _σ_ 0 [2] _**[I]**_ _[t]_            - _−_ 1 _**y**_ _t,_ (3)

Cov _t_ ( _**x**_ _,_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _−_ _k_ ( _**x**_ _, Dt_ )       - _**K**_ _t_ + _σ_ 0 [2] _**[I]**_ _[t]_       - _−_ 1 _k_ ( _Dt,_ _**x**_ _′_ ) _,_ (4)


where _**K**_ _t_ = _k_ ( _Dt, Dt_ ) _∈_ R _[t][×][t]_, _**I**_ _t_ is the _t × t_ identity, and _k_ ( _**x**_ _, Dt_ ) = [ _k_ ( _**x**_ _,_ _**x**_ 1) _, . . ., k_ ( _**x**_ _,_ _**x**_ _t_ )].

The GP posterior plays the role of a refined surrogate for _f_ _[⋆]_ throughout the optimization process. At
iteration _t_, a BO algorithm completes the next steps:


**Step 1.** Form the Gram matrix _**K**_ _t_ = _k_ ( _Dt, Dt_ ) using all past queries.

**Step 2.** Compute the inverse of _**K**_ _t_ + _σ_ 0 [2] _**[I]**_ _[t]_ [(with fixed hyperparameter] _[ σ]_ [0][) and plug it into][ (3)][-][(4)][ to]
obtain the posterior mean and covariance functions ( _µt,_ Cov _t_ ).


**Step** **3.** Select the next query by maximizing an acquisition function _αt_ : _S_ _→_ R built from
( _µt,_ Cov _t_ ) (e.g., GP-UCB (Srinivas et al., 2012) or Expected Improvement (Jones et al., 1998)). This
is where BO balances _exploration_ (learning _f_ _[⋆]_ ) and _exploitation_ (sampling near current optima). The
pair ( _µt, σt_ [2][)][ can be viewed as the algorithm’s current best estimate of the unknown function and its]
uncertainty.


The dataset is then updated with the new query:


_**x**_ _t_ +1 _∈_ arg max _αt_ ( _**x**_ ) _,_ _yt_ +1 = _f_ _[⋆]_ ( _**x**_ _t_ +1) + _εt_ +1 _,_
_**x**_ _∈S_


and the loop repeats until a stopping criterion is met.


**Measuring performance with regret.** We follow the common practice in BO: for experiments where
_f_ _[⋆]_ is known, we measure the regret on the deterministic _f_ _[⋆]_ _∈Hk_, and when discussing theoretical
regret bounds we refer to the regret on _f_ _∼GP_ (0 _, k_ ) (Garnett, 2023). In both cases, for _h_ = _f_ or
_h_ = _f_ _[⋆]_, the _instantaneous regret_ at timestep _t_ is _rt_ = max _**x**_ _∈S h_ ( _**x**_ ) _−_ _h_ ( _**x**_ _t_ ), the _cumulative regret_
at horizon _T_ is _RT_ = [�] _t_ _[T]_ =1 _[r][t]_ [, and the] _[ simple regret]_ [is] _[ s][T]_ [=] [max] _**[x]**_ _[∈S][ h]_ [(] _**[x]**_ [)] _[ −]_ [max][1] _[≤][t][≤][T]_ _[h]_ [(] _**[x]**_ _[t]_ [)][.]
A BO algorithm with a sublinear regret (i.e., _RT_ _∈_ _o_ ( _T_ )) is called _no-regret_ and offers asymptotic
global optimization guarantees on _f_ _[⋆]_ . Most standard cumulative regret upper bounds are established
in terms of the eigendecay of the operator spectrum of the kernel _k_ (Srinivas et al., 2012; Valko et al.,
2013; Scarlett et al., 2017; Whitehouse et al., 2023).


3


2.2 INVARIANCE IN BAYESIAN OPTIMIZATION


In many applications, the objective function _f_ _[⋆]_ is invariant under the action of a known symmetry
group _G_ on _S_, i.e., _f_ _[⋆]_ ( _**x**_ ) = _f_ _[⋆]_ ( _g_ _**x**_ ) for all _g_ _∈G_ . When such invariances are ignored, BO
algorithms may waste evaluations by treating all points within the same _|G|_ -orbit as distinct. Given
a non-invariant base kernel _k_ b and an arbitrary symmetry group _G_, both provided by the user, this
section reviews existing strategies for incorporating group invariance into BO and positions our
contribution within this literature.


**Data augmentation.** A popular way to enforce symmetry is to expand the dataset _Z_ itself, as it is
often done in computer vision (Krizhevsky et al., 2012). For each acquired observation ( _**x**_ _t, yt_ ), one
augments _Z_ with all transformed copies _{_ ( _g_ _**x**_ _t, yt_ ) _}g∈G_, while leaving the base kernel _k_ b unchanged.
However, since BO scales as _O_ ( _|Z|_ [3] ), this approach quickly becomes computationally prohibitive
and is inapplicable to continuous symmetry groups. For completeness, we include in Appendix F
a numerical comparison of our approach with data augmentation, showing that data augmentation
scales poorly with the size of the group, and does not meet the performance of the average or max
kernel even when using all symmetry augmentations.


**Search space restriction.** Another approach is to restrict the search domain to a fundamental region
_SG_ _⊆S_ whose _G_ -orbit covers _S_ : [�] _g∈G_ _[g][S][G]_ [=] _[S]_ [(e.g.,] [Baird] [et] [al.] [(2023b)).] [For] [example,] [if]

_S_ = [ _−_ 1 _,_ 1] [2] and _G_ is the group of _π/_ 2-rotations, one may work on _SG_ = [0 _,_ 1] [2] while keeping the
kernel unchanged. This viewpoint corresponds to working directly with the quotient _S/G_ embedded
in _S_ .


This line of work is complementary to ours. In BO, one must choose both a search domain and a
kernel: fundamental domains address the former, while our construction helps with the latter. Even if
we decide to run BO on _SG_, one still needs a good invariant kernel on _SG_, and our invariant kernels
can be used in that setting as well. We refer to Appendix G for a short example illustrating the
practical difficulties of explicitly optimizing over a fundamental domain, and how the design of the
kernel is complementary to that decision.


**Invariant** **kernels.** A principled way to incorporate prior _G_ -invariance of _f_ _[⋆]_ is to consider a _G_ invariant GP prior _f_, i.e., a GP whose sample paths _**x**_ _∈S_ _�→_ _f_ ( _**x**_ _, ω_ ) obtained by fixing one outcome
_ω_ in the probability space are themselves invariant under _G_ . Ginsbourger et al. (2012) established that
such GPs necessarily admit a _G_ -invariant covariance function [1], meaning _k_ ( _g_ _**x**_ _, g_ _[′]_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) for
all _**x**_ _,_ _**x**_ _[′]_ _∈S_ and _g, g_ _[′]_ _∈G_ . The central question then becomes: how can one construct an invariant
kernel _k_ from an arbitrary base kernel _k_ b and symmetry group _G_ ? An elegant solution, dating back to
Kondor (2008) and recently advocated for BO by Brown et al. (2024), is to average _k_ b over _G_ -orbits:

_k_ avg( _**x**_ _,_ _**x**_ _[′]_ ) = _|G|_ 1 [2]            - _k_ b( _g_ _**x**_ _, g_ _[′]_ _**x**_ _[′]_ ) _._ (5)

_g,g_ _[′]_ _∈G_


This construction is not only guaranteed to be _G_ -invariant, but also admits a clean functional interpretation: if _Hk_ b and _Hk_ avg denote the RKHS induced by _k_ b and _k_ avg respectively, then _Hk_ avg
coincides exactly with the subspace of _G_ -invariant functions in _Hk_ b (Theorem 4.4.3 in Kondor
(2008)). Consequently, _k_ avg (up to normalization) has gained popularity as the standard off-the-shelf
kernel for BO in symmetric settings (Glielmo et al., 2017; Kim et al., 2021; Brown et al., 2024).


A complementary idea in kernel methods is to retain the _best_ latent alignment between two orbits via
a maximum, as in convolution/best-match kernels for structured data (Gartner, 2003; Vishwanathan¨
et al., 2003) and follow-up work across domains (Frohlich et al., 2005; Zhang, 2010; Curtin et al.,¨
2013). Max-alignment kernels, however, are not PSD in general, leading to indefinite Gram matrices.
This has motivated two families of remedies: (i) explicit Kre˘ın-space formulations (Ong et al., 2004;
Oglic & Gartner,¨ 2018), and (ii) simple PSD corrections such as eigenvalue clipping/flipping in
SVMs (Luss & D' aspremont, 2007; Chen et al., 2009), which are empirically effective.


**Our adaptation to BO.** Guided by the above, we adopt the max-alignment view for BO. To ensure
positive definiteness, we project _k_ max (see (1)) onto a PSD kernel _k_ + [(] _[D]_ [)][, which coincides with] _[ k]_ [max]
whenever the latter is already PSD. This preserves the sharp, high-contrast orbit alignments of _k_ max


1Up to modification, i.e., there is another GP _f ′_ such that for every _x_ _∈S_, P( _f_ ( _**x**_ ) = _f ′_ ( _**x**_ )) = 1 and _f ′_

has invariant paths and invariant covariance, see Property 3.3 in Ginsbourger et al. (2012).


4


while ensuring compatibility with the BO framework. Moreover, it maintains a per-iteration BO
complexity comparable to that of orbit-averaged kernels (see Section 2.2). In our experiments, _k_ + [(] _[D]_ [)]
better reflects the intended symmetries of standard synthetic objectives and achieves substantially
lower cumulative regret. Interestingly, these empirical gains are not mirrored by existing eigendecaybased upper bounds, a point we return to in Section 5.


3 THE MAX KERNEL


We have introduced the max-alignment kernel _k_ max and its PSD surrogate _k_ + [(] _[D]_ [)] in (2). This section
explains _why k_ max is a natural _G_ -invariant covariance, clarifies how it differs from orbit averaging
through examples, and records the practical PSD construction we use in BO.


3.1 MOTIVATION: _k_ max AS A VALID COVARIANCE


A natural way to motivate _k_ max is to exhibit _G_ -invariant GPs whose covariance equals _k_ max.

**Construction.** Let _h ∼GP_ (0 _, k_ b) with an isotropic base kernel _k_ b( _**x**_ _,_ _**x**_ _[′]_ ) = _κ_ ( _∥_ _**x**_ _−_ _**x**_ _[′]_ _∥_ 2) with _κ_
nonincreasing (e.g., popular ones such as RBF, Matern).´ Consider a map _ϕG_ such that (i) _ϕG_ ( _**x**_ ) =
_ϕG_ ( _g_ _**x**_ ) for all _g_ _∈G_ and (ii) _∥ϕG_ ( _**x**_ ) _−ϕG_ ( _**x**_ _[′]_ ) _∥_ 2 = min _g,g′ ∥g_ _**x**_ _−g_ _[′]_ _**x**_ _[′]_ _∥_ 2. Define _f_ ( _**x**_ ) = _h_ ( _ϕG_ ( _**x**_ )).
Then _f_ is _G_ -invariant and:


**Proposition 1.** _Under the construction above, f_ _∼GP_ (0 _, k_ max) _with k_ max _given by_ (1) _._


def _f_ (ii)
_Proof sketch, details in Appendix A._ Cov( _f_ ( _**x**_ ) _, f_ ( _**x**_ _[′]_ )) = _k_ b( _ϕG_ ( _**x**_ ) _, ϕG_ ( _**x**_ _[′]_ )) = _κ_ (min _g,g′ ∥g_ _**x**_ _−_
_g_ _[′]_ _**x**_ _[′]_ _∥_ 2), and monotonicity of _κ_ converts the min-distance into max _g,g′ k_ b( _g_ _**x**_ _, g_ _[′]_ _**x**_ _[′]_ ).


This shows that _k_ max naturally arises as the covariance of valid _G_ -invariant GPs. In contrast, the
common approach to invariance in BO is to build _k_ avg by averaging a base kernel as in (5). But
averaging and maximization induce fundamentally different geometries:

**Lemma** **2.** _For_ _any_ _base_ _kernel_ _k_ b _and_ _any_ _(double)_ _orbit_ _O_ ( _**x**_ _,_ _**x**_ _[′]_ ) := _{_ ( _g_ _**x**_ _, g_ _[′]_ _**x**_ _[′]_ ) _, g, g_ _[′]_ _∈G},_
_k_ avg = _k_ max _on O_ ( _**x**_ _,_ _**x**_ _[′]_ ) _if and only if k_ b = _k_ max _on that orbit._


Indeed, an average reaches the maximum only when every term is maximal. Thus _k_ avg can never
reproduce the geometry of _k_ max, except in the degenerate case where the base kernel is already
_k_ max, making averaging redundant. One might wonder whether this limitation of _k_ avg could be
circumvented by building it from a _different_ base kernel than the one used for _k_ max. In Appendix A.2
we show that, under mild assumptions satisfied by standard kernels (upper-bounded by 1, with
equality _k_ ( _**x**_ _,_ _**x**_ ) = 1 along the diagonal), _k_ avg and _k_ max can coincide only in the trivial case where
the base kernel of _k_ avg is already invariant for pairs of points belonging to the same orbit. Thus, even
in this more general setting, averaging does not reproduce the geometry of maximization (except if
the base kernel already had invariances).


To make this contrast concrete, we now examine a simple example (radial invariance with an RBF
base kernel) where _k_ max and _k_ avg can be computed in closed form.

**Example 3** (Radial invariance with _k_ max) **.** _Let G_ _be the group of planar rotations and k_ b( _**x**_ _,_ _**x**_ _[′]_ ) =
exp� _−∥_ _**x**_ _−_ _**x**_ _[′]_ _∥_ [2] 2 _[/]_ [2] _[l]_ [2][�] _be an RBF kernel._ _With ϕG_ ( _**x**_ ) = _∥_ _**x**_ _∥_ 2 _,_

_k_ max( _**x**_ _,_ _**x**_ _[′]_ ) = exp� _−_ ( _∥_ _**x**_ _∥_ 2 _−∥_ _**x**_ _[′]_ _∥_ 2) [2] _/_ 2 _l_ [2][�] _,_ _k_ avg( _**x**_ _,_ _**x**_ _[′]_ ) = exp� _−_ _[∥]_ _**[x]**_ _[∥]_ 2 [2] 2 [+] _l_ [2] _[∥]_ _**[x]**_ _[′][∥]_ [2] 2 - _I_ 0� _∥_ _**x**_ _∥_ 2 _l∥_ [2] _**x**_ _[′]_ _∥_ 2 - _,_


_with_ _I_ 0 _the_ _modified_ _Bessel_ _function_ _(derivation_ _in_ _Appendix_ _B)._ _As_ _illustrated_ _in_ _Figure_ _1,_ _the_
_two_ _kernels_ _k_ max _and_ _k_ avg _induce_ _qualitatively_ _different_ _similarity_ _structures._ _By_ _construction,_
_k_ max _assigns large similarity whenever ∥_ _**x**_ _∥_ 2 _≈∥_ _**x**_ _[′]_ _∥_ 2 _._ _If ∥_ _**x**_ _∥_ 2 = _∥_ _**x**_ _[′]_ _∥_ 2 _, the function f_ _[⋆]_ _satisfies_
_f_ _[⋆]_ ( _**x**_ ) = _f_ _[⋆]_ ( _**x**_ _[′]_ ) _since it is invariant under rotations, and k_ max _exactly recovers this invariance by_
_assigning maximal similarity k_ max( _**x**_ _,_ _**x**_ _[′]_ ) = 1 _._ _In contrast, k_ avg _only approximates this behavior:_
_its iso-similarity curves as a function of_ ( _∥_ _**x**_ _∥_ 2 _, ∥_ _**x**_ _[′]_ _∥_ 2) _correspond to distorted balls, and two points_
_with identical norms may be ranked as highly dissimilar (see the diagonal ∥_ _**x**_ _∥_ 2 = _∥_ _**x**_ _[′]_ _∥_ 2 _of the right_
_plot in Figure 1)._ _This mismatch highlights that while both constructions enforce rotation invariance,_
_only k_ max _preserves the correct notion of similarity._


5


Table 1: Complexity per BO iteration. Here _|G|_ _[∗]_ denotes either _|G|_ or _|G|_ [2] depending on whether the
orbit terms reduce to a single sum (when _k_ b( _g_ _**x**_ _,_ _**x**_ _[′]_ ) suffices) or require a double sum over ( _g, g_ _[′]_ ); _m_
is the number of candidate points used in acquisition optimization. The row _Per-candidate acquisition_
_evaluation_ gives the cost of a single acquisition evaluation; for one BO iteration this row is multiplied
by _m_ and added to the other rows to obtain the total.


Base kernel _k_ b Averaged _k_ avg Projected _k_ + [(] _[D]_ [)]
Gram matrix ( _n × n_ ) _O_ ( _n_ [2] ) _O_ ( _n_ [2] _|G|_ _[∗]_ ) _O_ ( _n_ [2] _|G|_ _[∗]_ )
SVD / inversion _O_ ( _n_ [3] ) _O_ ( _n_ [3] ) _O_ ( _n_ [3] )
PSD projection  -  - _O_ ( _n_ [3] ) [4]
Per-candidate acq. eval. _O_ (1) _O_ ( _|G|_ _[∗]_ ) _O_ ( _n|G|_ _[∗]_ )

**Total for 1 BO iteration** _O_ ( _m_ + _n_ [2] + _n_ [3] ) _O_ (( _m_ + _n_ [2] ) _|G|_ _[∗]_ + _n_ [3] ) _O_ (( _mn_ + _n_ [2] ) _|G|_ _[∗]_ + _n_ [3] )


3.2 A PSD EXTENSION OF _k_ max: WHAT WE USE IN PRACTICE


Because _k_ max is not PSD in general, we apply a standard projection step on the finite design set
_D_ = _{_ _**x**_ 1 _, . . .,_ _**x**_ _n}_ . Let _**K**_ = _k_ max( _D, D_ ) with eigendecomposition _**K**_ = _**Q**_ **Λ** _**Q**_ _[⊤]_ and define [2]
(with the max applied elementwise)


_**K**_ + = _**Q**_ max(0 _,_ **Λ** ) _**Q**_ _[⊤]_ _._ (6)


We then use the Nystrom extension¨ [3] (Williams & Seeger, 2000) to evaluate cross-covariances with
new points, yielding the PSD, _G_ -invariant surrogate _k_ + [(] _[D]_ [)] given in (2) and that we reproduce here:

_k_ + [(] _[D]_ [)][(] _**[x]**_ _[,]_ _**[ x]**_ _[′]_ [)] [:=] _[k]_ [max][(] _**[x]**_ _[,][ D]_ [)] _**[ K]**_ + _[†]_ _[k]_ [max][(] _[D][,]_ _**[ x]**_ _[′]_ [)] _[.]_ (7)


**Key properties of** _k_ + [(] _[D]_ [)] **[:]**

- _PSD & invariance._ _k_ + [(] _[D]_ [)] is PSD and inherits argumentwise _G_ -invariance [5] of _k_ max.

- _Consistency with k_ max _._ If _**K**_ _⪰_ 0, then _**K**_ + = _**K**_ and _k_ + [(] _[D]_ [)] agrees with _k_ max on _D × D_ .

- _Cost._ Each BO iteration involves (i) building the Gram matrix on _D_, (ii) inverting the Gram matrix
to build the acquisition function, and (iii) _m_ kernel evaluations when optimizing the acquisition
function. Step (ii) has the same cost as the SVD of _**K**_ needed to compute both _**K**_ + and _**K**_ + _[†]_ [, which]
makes _k_ + [(] _[D]_ [)] having the same asymptotic per-iteration cost as _k_ avg; its per-query evaluations are
more expensive, but this difference is negligible as long as we keep _m_ ≲ _n_ . A concise complexity
summary is provided in Table 1, and example of runtimes in Table 3.

- _Regularity._ For finite groups, _k_ max is a max of finitely many smooth maps and is almost everywhere (a.e.) differentiable; the Nystrom extension preserves a.e. differentiability in each argument.¨
For continuous groups, smoothness can sometimes be obtained via closed-form formulas (e.g., as
in Example 3).

We now illustrate the behavior of _k_ + [(] _[D]_ [)] versus _k_ avg (in this situation, _k_ max is not PSD and the
projection step is indeed needed to restore positive semidefiniteness).

**Example 4** (Ackley function with _k_ +) **.** _Figure 2 compares k_ + [(] _[D]_ [)] _and k_ avg _on the one-dimensional_
_Ackley function (see_ (15) _)._ _The projected kernel k_ + [(] _[D]_ [)] _preserves the expected pairwise symmetries_
_(invariance along x_ = _y and x_ = _−y) and spreads mass more evenly across the symmetric regions,_
_whereas k_ avg _concentrates covariance mostly near the origin._ _Thus, k_ + [(] _[D]_ [)] _better reflects the symmetry_
_geometry of the problem, echoing the qualitative difference observed in Example 3._


**Beyond the finite view (details in Appendix C).** The PSD projection with Nystrom in Equation (7) is¨
a practical, data-dependent construction. It can be seen as the finite-sample face of a broader, intrinsic


2 _**K**_ + does not depend on the choice of the eigendecomposition, see Lemma 7 in the appendix.
3It indeed extends _**K**_ + since _k_ + [(] _[D]_ [)][(] _**[x]**_ _[i][,]_ _**[ x]**_ _[j]_ [)] [=] _**[K]**_ _[i,]_ [:] _**[ K]**_ + _[†]_ _**[K]**_ [:] _[,j]_ [=] [(] _**[KK]**_ + _[†]_ _**[K]**_ [)] _[ij]_ [=] [(] _**[K]**_ [+][)] _[ij]_ [.]
4One SVD of _**K**_ suffices to obtain both _**K**_ + and _**K**_ + _†_ [,] [so] [the] [extra] [PSD] [projection] [does] [not] [increase]
asymptotic cost.
5 _k_ max( _g_ _**x**_ _,_ _**x**_ _′_ ) = _k_ max( _**x**_ _,_ _**x**_ _′_ ) implies _k_ max( _g_ _**x**_ _, D_ ) = _k_ max( _**x**_ _, D_ ), hence invariance of _k_ + [(] _[D]_ [)][.]


6


1.0


0.5


0.0


0.5


1.0


0.5


0.0


0.5


k+ [(] ) [(][x][,][ x][′][)]


0


1


2


3


4


1.0

x


x


1.0

1.0 0.5 0.0 0.5 1.0
x


Figure 2: (Left) One-dimensional Ackley function _f_ _[⋆]_ (see (15)), invariant up to coordinate-wise
sign-flips, and GP posterior means _µt_ ( _**x**_ ) as in (3) for _k_ + [(] _[D]_ [)] (orange diamond) and _k_ avg (green circles)
built from _D_ (black crosses). (Center) Covariance structure induced by _k_ + [(] _[D]_ [)][.] [(Right) Covariance]
structure induced by _k_ avg. Both kernels are invariant to reflections across _x_ = _y_ and _x_ = _−y_, but
_k_ avg concentrates covariance near 0, while _k_ + [(] _[D]_ [)] better reflects the underlying symmetry geometry.
Consequently, the GP posterior mean induced by _k_ + [(] _[D]_ [)] is the best at fitting the objective (left).

Table 2: Performance of _k_ b, _k_ avg, and _k_ + [(] _[D]_ [)] across benchmarks. For each kernel _k_ _∈{k_ b _, k_ avg _, k_ + [(] _[D]_ [)] _[}]_
we report _m ± s_ err, where _m_ is the empirical mean over 10 seeds (lower is better) and _s_ err is the
empirical standard error. Best mean is **bold** ; means _m_ whose 95% confidence interval ( _m ±_ 1 _._ 96 _s_ err)
confidence interval overlap with the best are underlined. Performance is measured by cumulative
regret on synthetic benchmarks and by negated simple reward on real-world experiments.


**Benchmark** _|G|_ _k_ b _k_ avg _k_ + [(] _[D]_ [)]

_Synthetic (Cumulative Reg.)_
Ackley2d 8 382 _._ 7 _±_ 5 _._ 7 128 _._ 2 _±_ 10 _._ 4 _._ **4** _±_ **3** _._ **6**
Griewank6d 64 3840 _._ 3 _±_ 177 _._ 7 3067 _._ 4 _±_ 841 _._ 9 **1832** _._ **6** _±_ _._ **3**
Rastrigin5d 3 _,_ 840 3568 _._ 5 _±_ 91 _._ 3 1583 _._ 5 _±_ 341 _._ 9 _._ **4** _±_ **70** _._ **6**
Radial2d _∞_ 388 _._ 6 _±_ 20 _._ 3 480 _._ 9 _±_ 76 _._ 4 _._ **7** _±_ **11** _._ **6**
Scaling2d _∞_ 1820 _._ 6 _±_ 1135 _._ 4 3361 _._ 8 _±_ 742 _._ 9 **25** _._ **4** _±_ **6** _._ **4**


_Real-World (Neg._ _Simple Rew.)_
WLAN8d 24 _−_ 65 _._ 0 _±_ 3 _._ 2 _−_ 51 _._ 8 _±_ 1 _._ 7 _−_ **74** _._ **4** _±_ **0** _._ **7**
PartPack6d _∞_ _−_ 0 _._ 79 _±_ 0 _._ 10 _−_ 0 _._ 69 _±_ 0 _._ 01 _−_ **0** _._ **92** _±_ **0** _._ **10**


definition that does not depend on _D_ . Since _k_ max is symmetric, it admits a spectral decomposition
_k_ max( _**x**_ _,_ _**x**_ _[′]_ ) = [�] _i_ _[λ][i][ϕ][i]_ [(] _**[x]**_ [)] _[ϕ][i]_ [(] _**[x]**_ _[′]_ [)][ in] _[ L]_ [2][, and we can always define (a.e.)]

_k_ +( _**x**_ _,_ _**x**_ _[′]_ ) :=            - max(0 _, λi_ ) _ϕi_ ( _**x**_ ) _ϕi_ ( _**x**_ _[′]_ ) _,_


_i_


with _k_ + = _k_ max whenever _k_ max is already PSD. On finite domains, this precisely reduces to the
matrix PSD projection in (6). In Appendix C we formalize the infinite-domain construction via
integral operators, prove that _k_ + is _G_ -invariant, and show that the finite projection + Nystrom in¨ (7)
converges to _k_ + at the spectral (Hilbert-Schmidt) level under iid sampling (Appendix C.4).


**Takeaway.** _k_ max is the exact covariance of a natural class of _G_ -invariant GPs and induces a search
geometry that preserves high-contrast orbit alignments (Examples 3 and 4). The PSD projection +
Nystrom step yields a valid GP kernel¨ _k_ + [(] _[D]_ [)] without introducing extra asymptotic complexity. We
now measure its practical impact in Section 4.


4 EXPERIMENTS


We evaluate _k_ + [(] _[D]_ [)] against two baselines: (i) the off-the-shelf kernel _k_ b (no symmetry handling), and
(ii) the orbit-averaged kernel _k_ avg (Brown et al., 2024). Benchmarks include standard synthetic
objectives and two real-world tasks with known invariances (a wireless network design task and a
particle packing problem). We ask: _(Q1) Does k_ + [(] _[D]_ [)] _reduce simple/cumulative regret vs. k_ avg _?_ and
_(Q2) How does performance scale with the size of the symmetry group and dimension?_ The full
experimental setup is described in Appendix E.


7


4000


3000


2000


1000


0


Iteration T


30

40

50

60

70


Iteration T


500

400

300

200

100

0


Iteration T


Figure 3: Cumulative regret and negated simple reward under GP-UCB with _k_ b (blue crosses),
_k_ avg (orange diamonds), and _k_ + [(] _[D]_ [)] (green circles) on a selection of benchmarks (all benchmarks in
Appendix E). Shaded regions show the standard error ( _±serr_ ) over 10 seeds.


**Headline:** _k_ + [(] _[D]_ [)] **wins** **on** **every** **task.** Across all benchmarks (Table 2), _k_ + [(] _[D]_ [)] achieves the best
performance with up to 50% of improvement. This answers **Q1** positively. Regarding **Q2**, we will see
that as the group size increases, _k_ + [(] _[D]_ [)] stays strong, while _k_ avg degrades and can even underperform
the non-invariant base kernel _k_ b.

**Setup in one glance.** We run GP-UCB with each kernel _k_ _∈{k_ b _, k_ avg _, k_ + [(] _[D]_ [)] _[}]_ [, using the same acqui-]
sition and optimization budgets. We report results averaged over 10 seeds. All the hyperparameters
and group actions are detailed in Appendix E.


4.1 SYNTHETIC BENCHMARKS


We consider synthetic functions _f_ _[⋆]_ (Ackley, Griewank, Rastrigin, etc.) that exhibit symmetries
(such as permutations, coordinate-wise sign-flips, rotations, rescaling) and are classically considered
as challenging to optimize in the BO literature (Qian et al., 2021; Bardou et al., 2024). We cover
dimensions _d_ = 2 to _d_ = 6 and group sizes _|G|_ = 8 to _|G|_ = _∞_ . We evaluate performance
using the cumulative regret _RT_ = [�] _i_ _[T]_ =1 - _f_ _[⋆]_ ( _**x**_ _[∗]_ ) _−_ _f_ _[⋆]_ ( _**x**_ _t_ )� since the global maximizer _**x**_ _[∗]_ =
arg max _**x**_ _∈S f_ _[⋆]_ ( _**x**_ ) is known.

**Finite groups:** **the gap widens as** _|G|_ **grows.** With Matern-5/2 base´ _k_ b on Ackley2d ( _|G|_ =8), _k_ avg
and _k_ + [(] _[D]_ [)] are tied; both dominate _k_ b. As _|G|_ increases (Griewank6d, _|G|_ =64; Rastrigin5d, _|G|_ =3 _,_ 840),
_k_ + [(] _[D]_ [)] increasingly outperforms _k_ avg achieving cumulative regrets that are, on average, 40% and 49%
lower respectively (Table 2, Figure 3 left panel, and Appendix E for the whole set of figures).


**Continuous groups:** _k_ avg **can underperform even** _k_ b **.** For radial and scaling invariances (continuous
groups; RBF base), _k_ avg degrades relative to _k_ b, while _k_ + [(] _[D]_ [)] remains strong (Figure 3 center panel,
and Appendix E for the whole set of figures).


4.2 REAL-WORLD EXPERIMENTS


We consider two real-world experiments that are described in detail in Appendix E: the design of
a wireless network (8-dimensional, invariant to permutations of pairs of parameters) and a particle
packing problem (6-dimensional, invariant to the rescaling of some parameters and to permutations
of pairs of parameters). For both benchmarks, performance is evaluated using the negated best reward
min _t∈_ [ _T_ ] _−f_ _[⋆]_ ( _**x**_ _t_ ) attained during optimization (the regret cannot be computed because the max of
_f_ _[⋆]_ is unknown). Note that we consider min _t∈_ [ _T_ ] _−f_ _[⋆]_ ( _**x**_ _t_ ) instead of the cumulated _−_ [�] _t_ _[f][ ⋆]_ [(] _**[x]**_ _[t]_ [)]

because the goal is to assess the quality of the best combination of parameters discovered by the
optimizer, rather than the cumulative negative reward across all explored combinations.

_k_ + [(] _[D]_ [)] **finds better combinations of parameters.** For the design of a wireless network or for the
particle packing problem, _k_ + [(] _[D]_ [)] consistently discovers combinations of parameters with larger utility
than both _k_ avg and _k_ b (Figure 3 right; Appendix E for more figures).


8


Eigenvalue index i


Eigenvalue index i


15


10


5


Number of Symmetries |G|


10 1

10 3


10 5

10 7

10 9


10 11


10 1

10 3

10 5

10 7

10 9

10 11


Eigenvalue index i


10 1

10 3


10 5

10 7

10 9


10 11


10 [0]

10 2

10 4

10 6

10 8

10 10

10 12


60


40


20


0


Number of Symmetries |G|


Eigenvalue index i


Figure 4: **Left column:** Final average regret _RT /T_ for _k_ b (blue crosses), _k_ avg (orange diamonds),
and _k_ + [(] _[D]_ [)] (green circles) on Ackley (top) and Rastrigin (bottom), averaged over 10 seeds with standard
error bars. **Middle and right columns:** Empirical eigendecays under different bases and groups
(ordered eigenvalues of the Gram-matrix divided by _n_ ), typical behavior on a single seed.


4.3 ROBUSTNESS TO GROUP SIZE


Both synthetic and real-world benchmarks suggest that _k_ avg performs comparably to _k_ + [(] _[D]_ [)] when
the group size _|G|_ is small, but its performance deteriorates as _|G|_ grows, whereas _k_ + [(] _[D]_ [)] remains
stable. To investigate this effect more systematically, we conduct additional experiments on the
_d_ -dimensional Ackley and Rastrigin benchmarks, each invariant under the hyperoctahedral group
_G_ of size _|G|_ = 2 _[d]_ _d_ ! (permutations _×_ coordinate-wise sign-flips). We compare the average regret
of _k_ avg and _k_ + [(] _[D]_ [)] after 50 iterations of GP-UCB for dimensions _d_ = 1 _, . . .,_ 5, and include _k_ b as a
baseline to control for the effect of increasing _d_ .


The results are shown in Figure 4 (left column) . Both experiments reveal the same trend: while
_k_ avg consistently outperforms _k_ b, its performance also deteriorates as _|G|_ increases. In contrast, _k_ + [(] _[D]_ [)]
remains largely unaffected by the growing number of symmetries, demonstrating a clear robustness
to group size. In the next section, we discuss several explanations for these empirical observations.

**Takeaway.** _k_ + [(] _[D]_ [)] consistently matches or outperforms _k_ avg and _k_ b, with the largest gains at large _|G|_ .
The evidence suggests that (i) _how_ a kernel encodes orbit alignments matters as much as _whether_
it is invariant, and (ii) averaging across many alignments can dilute informative similarities. These
themes reconnect with our discussion in Section 5 and motivate analyses beyond eigendecay rates.


5 SPECTRAL ANALYSIS AND REGRET BOUNDS


So far, _k_ + [(] _[D]_ [)] has shown consistently lower regret than _k_ avg, despite comparable computational cost.
A natural question is: _can existing BO theory account for such a gap?_ Current regret bounds for GP
surrogates proceed via the information gain, which is shaped by the decay of the operator spectrum
of the kernel. In particular, faster spectral decay leads to tighter regret upper bounds in standard
analyses (Srinivas et al., 2012; Valko et al., 2013; Scarlett et al., 2017; Whitehouse et al., 2023). We
now compare the eigendecay of _k_ + [(] _[D]_ [)] and _k_ avg, and ask whether it can explain the empirical gap.


**Empirical eigendecays:** **similar or** _**faster**_ **decay for** _k_ avg **.** Across our benchmarks, the empirical
spectra of _k_ + [(] _[D]_ [)] and _k_ avg exhibit very similar log–log slopes (decay rates). In several settings, _k_ avg’s
eigenvalues decay even _faster_ than those of _k_ +; see Figure 4 (middle and right columns). Under
the usual theory, this would translate into similar, or potentially _tighter_, upper bounds for methods
run with _k_ avg compared to those with _k_ + [(] _[D]_ [)][.] [A more detailed discussion of the empirical spectra in]
Figure 4 and further insights are in Appendix D.


9


**Limitations of eigendecay as an explanation.** Since _k_ avg matches or exceeds _k_ + [(] _[D]_ [)] in empirical
decay rate, standard theory would predict similar or better regret upper bounds. Yet in practice we
consistently observe lower regret for _k_ + [(] _[D]_ [)] (Section 4). This suggests that eigendecay alone does not
capture the structural advantages of _k_ + [(] _[D]_ [)][.] [We outline possible explanations in the conclusion.]


6 CONCLUSION


Our spectral analysis highlights a gap between theory and practice: although _k_ avg often exhibits _faster_
empirical eigendecay than _k_ + [(] _[D]_ [)][, the latter consistently achieves lower regret.] [Standard eigendecay]
arguments thus fail to explain the observed advantage of _k_ + [(] _[D]_ [)][.] [We hypothesize two complementary]
explanations.


First, **geometry vs. rates:** eigendecay quantifies how fast spectra shrink but ignores _which_ eigenfunctions are emphasized. In practice, _k_ avg often introduces _similarity reversals_, distorting the search
geometry (Figure 1), whereas _k_ + [(] _[D]_ [)] preserves high-contrast alignments between orbits, just as _k_ max.

Second, **approximation hardness:** standard BO guarantees assume that the black-box _f_ _[⋆]_ belongs
to the RKHS _Hk_ of the chosen kernel _k_ . In practice, this assumption is unverifiable. One might
therefore wonder whether the empirical gap between _k_ avg and _k_ + [(] _[D]_ [)] could stem from _misspecification_,
i.e., different distances _d_ ( _f_ _[⋆]_ _, Hk_ avg ) and _d_ ( _f_ _[⋆]_ _, Hk_ +( _D_ ) [)][.] [Indeed, existing work (Bogunovic & Krause,]

2021) shows that when _d_ ( _f_ _[⋆]_ _, Hk_ ) _>_ 0, regret necessarily degrades linearly with this distance.


However, this explanation does not apply in our setting. With an RBF base kernel _k_ b, _Hk_ b is universal
(Micchelli et al., 2006), so _Hk_ avg contains all invariant functions [6] including _f_ _[⋆]_, so _d_ ( _f_ _[⋆]_ _, Hk_ avg ) = 0
and cannot exceed _d_ ( _f_ _[⋆]_ _, Hk_ +( _D_ ) [)][.] [Yet] _[ k]_ [avg] [still performs worse than] _[ k]_ + [(] _[D]_ [)][.]

This suggests that the relevant issue is not membership in _Hk_ avg or _Hk_ +( _D_ ) [,] [but] _[how]_ _[difficult]_ _[f][ ⋆]_

_is to approximate within each RKHS_ . BO ultimately constructs approximations from finite linear
combinations of atoms _k_ ( _**x**_ _t, ·_ ). Different kernels can thus induce different approximation rates.


This perspective might explain both our results and those of Brown et al. (2024). In their setup,
the synthethic benchmarks _f_ _[⋆]_ are explicit linear combinations of relatively few _k_ avg( _**x**_ _t, ·_ ) atoms
(between 64 and 512, depending on dimension; see their Appendix B.1). In such regimes, _k_ avg
performs very well, likely because the GP posterior mean can in principle exactly recover _f_ _[⋆]_ once
those _**x**_ _t_ are sampled. Typical BO objectives do not exhibit this sparse expansion structure, which
may explain why in our experiments _k_ avg consistently underperforms _k_ + [(] _[D]_ [)][, and sometimes even the]
base kernel.


Developing regret bounds that capture this notion of approximation hardness, beyond mere membership in _Hk_, appears essential to bridge the gap between theory and empirical performance.


Finally, while our focus has been empirical, we note that the intrinsic data-independent version of
_k_ + [(] _[D]_ [)][, which we called] _[ k]_ [+] [and which we mentioned at the end of Section 3.2 (introduced formally]
in Appendix C), provides a natural, data-independent analogue of the practical kernel _k_ + [(] _[D]_ [)][.] [We see]
_k_ + as a convenient object for future theoretical work, as it cleanly isolates the PSD projection of
_k_ max from the additional data dependence introduced by Nystrom.¨ We believe that it makes _k_ + a
convenient starting point for any future theoretical work, in the same spirit as gradient flow serving as
an idealized analogue of gradient descent.


ACKNOWLEDGMENTS


We thank the anonymous reviewers for their constructive suggestions, which led to several improvements and clarifications in the final version. This work was supported in part by the Swiss State
Secretariat for Education, Research and Innovation (SERI) under contract number MB22.00027.


6Consider ( _Pf_ )( _**x**_ ) = [�] _g∈G_ _[f]_ [(] _[g]_ _**[x]**_ [)] _[/][|G|]_ [, the projection onto] _[ H][k]_ [avg] [(Brown et al., 2024, Appendix A), and]

observe that if _fn_ _→_ _f_ _[⋆]_ with _fn_ _∈Hk_ b then _Pfn_ _→_ _f_ _[⋆]_ with _Pfn_ _∈Hk_ avg . So universality of _Hk_ b implies
universality of _Hk_ avg .


10


REFERENCES


Sterling Baird, Jason R. Hall, and Taylor D. Sparks. Compactness matters: Improving bayesian
optimization efficiency of materials formulations through invariant search spaces. _chemrxiv_, 2023a.
doi: 10.26434/chemrxiv-2022-nz2w8-v3.


Sterling G. Baird, Jason R. Hall, and Taylor D. Sparks. Compactness matters: Improving bayesian
optimization efficiency of materials formulations through invariant search spaces. _Computational_
_Materials Science_, 224:112134, 2023b. ISSN 0927-0256. doi: https://doi.org/10.1016/j.commatsci.
2023.112134.


Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization.
_Advances in neural information processing systems_, 33:21524–21538, 2020.


Anthony Bardou, Patrick Thiran, and Giovanni Ranieri. This too shall pass: Removing stale
observations in dynamic bayesian optimization. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 37:42696–42737, 2024.


Anthony Bardou, Jean-Marie Gorce, and Thomas Begin. Assessing the performance of noma in a
multi-cell context: A general evaluation framework. _IEEE Transactions on Wireless Communica-_
_tions_, 2025.


Abibasheer Basheerudeen and Sivakumar Anandan. Particle packing approach for designing the
mortar phase of self compacting concrete. _Engineering Journal_, 18(2):127–140, 4 2014.


Rajendra Bhatia and Ludwig Elsner. The hoffman-wielandt inequality in infinite dimensions. _Pro-_
_ceedings of the Indian Academy of Sciences – Mathematical Sciences_, 104(4):483–494, Aug 1994.
doi: 10.1007/BF02867116. [URL https://link.springer.com/article/10.1007/](https://link.springer.com/article/10.1007/BF02867116)
[BF02867116.](https://link.springer.com/article/10.1007/BF02867116)


Ilija Bogunovic and Andreas Krause. Misspecified gaussian process bandit optimization. _Advances_
_in neural information processing systems_, 34:3004–3015, 2021.


Theodore Brown, Alexandru Cioba, and Ilija Bogunovic. Sample-efficient bayesian optimisation
using known invariances. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _38:_ _Annual_
_Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_ _2024,_ _NeurIPS_ _2024,_ _Vancouver,_ _BC,_
_Canada, December 10 - 15, 2024_, 2024.


Yihua Chen, Maya R Gupta, and Benjamin Recht. Learning kernels from indefinite similarities. In
_Proceedings of the 26th Annual International Conference on Machine Learning_, pp. 145–152,
2009.


John B. Conway. _A Course in Functional Analysis_, volume 96 of _Graduate Texts in Mathematics_ .
Springer, 2007. ISBN 978-1-4757-4383-8. doi: 10.1007/978-1-4757-4383-8.


Ryan R Curtin, Parikshit Ram, and Alexander G Gray. Fast exact max-kernel search. In _Proceedings_
_of the 2013 SIAM International Conference on Data Mining_, pp. 1–9. SIAM, 2013.


Holger Frohlich, J¨ org K Wegner, Florian Sieker, and Andreas Zell.¨ Optimal assignment kernels for
attributed molecular graphs. In _Proceedings of the 22nd international conference on Machine_
_learning_, pp. 225–232, 2005.


Roman Garnett. _Bayesian optimization_ . Cambridge University Press, 2023.


Thomas Gartner.¨ A survey of kernels for structured data. _ACM SIGKDD explorations newsletter_, 5
(1):49–58, 2003.


David Ginsbourger, Xavier Bay, Olivier Roustant, and Laurent Carraro. Argumentwise invariant
kernels for the approximation of invariant functions. _Ann. Fac. Sci. Toulouse Math. (6)_, 21(3):
501–527, 2012. ISSN 0240-2963,2258-7519. doi: 10.5802/afst.1343. URL [https://doi.](https://doi.org/10.5802/afst.1343)
[org/10.5802/afst.1343.](https://doi.org/10.5802/afst.1343)


11


Aldo Glielmo, Peter Sollich, and Alessandro De Vita. Accurate interatomic force fields via machine
learning with covariant kernels. _Physical Review B_, 95(21):214302, 2017.


Javier Gonzalez, Joseph Longworth, David C James, and Neil D Lawrence. Bayesian optimization
for synthetic gene design. _arXiv preprint arXiv:1505.01627_, 2015.


Nicholas J. Higham. Computing a nearest symmetric positive semidefinite matrix. _Linear_
_Algebra_ _and_ _its_ _Applications_, 103:103–118, 1988. ISSN 0024-3795. doi: https://doi.org/
10.1016/0024-3795(88)90223-6. [URL https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/0024379588902236)
[article/pii/0024379588902236.](https://www.sciencedirect.com/science/article/pii/0024379588902236)


Donald R Jones, Matthias Schonlau, and William J Welch. Efficient global optimization of expensive
black-box functions. _Journal of Global optimization_, 13(4):455–492, 1998.


Jungtaek Kim, Michael McCourt, Tackgeun You, Saehoon Kim, and Seungjin Choi. Bayesian
optimization with approximate set kernels. _Machine Learning_, 110(5):857–879, 2021.


Vladimir Koltchinskii and Evarist Gine.´ Random matrix approximation of spectra of integral
operators. _Bernoulli_, 6(1):113–167, 2000. ISSN 1350-7265,1573-9759. doi: 10.2307/3318636.
[URL https://doi.org/10.2307/3318636.](https://doi.org/10.2307/3318636)


Risi Kondor. _Group theoretical methods in machine learning_ . PhD thesis, Columbia University,
2008. [URL https://people.cs.uchicago.edu/˜risi/papers/KondorThesis.](https://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf)
[pdf.](https://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf) Ph.D. thesis.


Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep
convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 25. Curran Associates, Inc., 2012. [URL https://proceedings.neurips.cc/paper_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
[2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)


John M. Lee. _Introduction_ _to_ _smooth_ _manifolds_, volume 218 of _Graduate_ _Texts_ _in_ _Mathematics_ .
Springer, New York, second edition, 2013. ISBN 978-1-4419-9981-8.


Pengfei Li, Xiaoyan Wang, and Hanbo Cao. Empirical compression model of ultra-high-performance
concrete considering the effect of cement hydration on particle packing characteristics. _Materials_,
16(13), 2023. ISSN 1996-1944. doi: 10.3390/ma16134585. [URL https://www.mdpi.com/](https://www.mdpi.com/1996-1944/16/13/4585)
[1996-1944/16/13/4585.](https://www.mdpi.com/1996-1944/16/13/4585)


Daniel J Lizotte, Tao Wang, Michael H Bowling, Dale Schuurmans, et al. Automatic gait optimization
with gaussian process regression. In _IJCAI_, volume 7, pp. 944–949, 2007.


Ronny Luss and Alexandre D' aspremont. Support vector machine classification with
indefinite kernels. In J. Platt, D. Koller, Y. Singer, and S. Roweis (eds.), _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 20. Curran Associates, Inc.,
2007. URL [https://proceedings.neurips.cc/paper_files/paper/2007/](https://proceedings.neurips.cc/paper_files/paper/2007/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper.pdf)
[file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2007/file/c0c7c76d30bd3dcaefc96f40275bdc0a-Paper.pdf)


Charles A Micchelli, Yuesheng Xu, and Haizhang Zhang. Universal kernels. _Journal of Machine_
_Learning Research_, 7(12), 2006.


Dino Oglic and Thomas Gartner.¨ Learning in reproducing kernel kreın spaces. In _International_
_conference on machine learning_, pp. 3859–3867. PMLR, 2018.


Cheng Soon Ong, Xavier Mary, Stephane Canu, and Alexander J Smola.´ Learning with non-positive
kernels. In _Proceedings of the twenty-first international conference on Machine learning_, pp. 81,
2004.


Peter Petersen. _Riemannian geometry_, volume 171. Springer, 2006.


Chao Qian, Hang Xiong, and Ke Xue. Bayesian optimization using pseudo-points. In _Proceedings_
_of_ _the_ _Twenty-Ninth_ _International_ _Conference_ _on_ _International_ _Joint_ _Conferences_ _on_ _Artificial_
_Intelligence_, pp. 3044–3050, 2021.


12


Carl Edward Rasmussen and Christopher K. I. Williams. _Gaussian processes for machine learning_ .
Adaptive Computation and Machine Learning. MIT Press, Cambridge, MA, 2006. ISBN 978-0262-18253-9.


Michael Reed and Barry Simon. Vi - bounded operators. In _Methods of Modern Mathematical Physics_,
pp. 182–220. Academic Press, 1972. ISBN 978-0-12-585001-8. doi: https://doi.org/10.1016/
B978-0-12-585001-8.50012-X. URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/B978012585001850012X)
[article/pii/B978012585001850012X.](https://www.sciencedirect.com/science/article/pii/B978012585001850012X)


Jonathan Scarlett, Ilija Bogunovic, and Volkan Cevher. Lower bounds on regret for noisy gaussian
process bandit optimization. In _Conference on Learning Theory_, pp. 1723–1742. PMLR, 2017.


Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias W. Seeger. Information-theoretic
regret bounds for gaussian process optimization in the bandit setting. _IEEE_ _Transactions_ _on_
_Information Theory_, 58(5):3250–3265, 2012. doi: doi:10.1109/tit.2011.2182033.


Sylia Mekhmoukh Taleb, Yassine Meraihi, Asma Benmessaoud Gabis, Seyedali Mirjalili, and Amar
Ramdane-Cherif. Nodes placement in wireless mesh networks using optimization approaches: a
survey. _Neural Computing and Applications_, 34(7):5283–5319, 2022.


Aidan P. Thompson, H. Metin Aktulga, Richard Berger, Dan S. Bolintineanu, W. Michael Brown,
Paul S. Crozier, Pieter J. in ’t Veld, Axel Kohlmeyer, Stan G. Moore, Trung Dac Nguyen, Ray
Shan, Mark J. Stevens, Julien Tranchida, Christian Trott, and Steven J. Plimpton. Lammps - a
flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum
scales. _Computer Physics Communications_, 271:108171, 2022. ISSN 0010-4655. doi: https://doi.
org/10.1016/j.cpc.2021.108171. URL [https://www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/S0010465521002836)
[article/pii/S0010465521002836.](https://www.sciencedirect.com/science/article/pii/S0010465521002836)


Michal Valko, Nathaniel Korda, Remi´ Munos, Ilias Flaounas, and Nelo Cristianini. Finite-time
analysis of kernelised contextual bandits. _arXiv preprint arXiv:1309.6869_, 2013.


SVN Vishwanathan, Alexander J Smola, et al. Fast kernels for string and tree matching. _Advances in_
_neural information processing systems_, 15:569–576, 2003.


Justin Whitehouse, Aaditya Ramdas, and Steven Z Wu. On the sublinear regret of gp-ucb. _Advances_
_in Neural Information Processing Systems_, 36:35266–35276, 2023.


Christopher Williams and Matthias Seeger. Using the nystrom method to speed up kernel machines.¨
In T. Leen, T. Dietterich, and V. Tresp (eds.), _Advances in Neural Information Processing Sys-_
_tems_, volume 13. MIT Press, 2000. [URL https://proceedings.neurips.cc/paper_](https://proceedings.neurips.cc/paper_files/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)
[files/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)


Mohamed Younis and Kemal Akkaya. Strategies and techniques for node placement in wireless
sensor networks: A survey. _Ad Hoc Networks_, 6(4):621–655, 2008.


Ziming Zhang. _Maximum Similarity Based Feature Matching and Adaptive Multiple Kernel Learning_
_for Object Recognition_ . PhD thesis, Simon Fraser University, 2010. PhD thesis.


13


A PROOFS FOR SECTION 3


A.1 FULL STATEMENT AND PROOF OF PROPOSITION 1


We state Proposition 1 formally and give a slightly more detailed proof.
**Proposition 5** (Max-kernel covariance for invariant GPs) **.** _Let S, Sh_ _⊂_ R _[d]_ _be measurable spaces and_
_let a (finite or compact) group G_ _act measurably on S._ _Let h ∼GP_ (0 _, k_ b) _be a GP on Sh_ _with an_
_isotropic base kernel k_ b : ( _**x**_ _,_ _**x**_ _[′]_ ) _∈S ×S_ _�→_ _κ_ ( _∥_ _**x**_ _−_ _**x**_ _[′]_ _∥_ 2) _where κ_ : R _≥_ 0 _→_ R _≥_ 0 _is nonincreasing._
_Assume there exists ϕG_ : _S_ _→Sh_ _satisfying (i)_ invariance: _ϕG_ ( _**x**_ ) = _ϕG_ ( _g_ _**x**_ ) _for all g_ _∈G,_ _**x**_ _∈S;_
_and (ii)_ minimal-distance representativity: _∥ϕG_ ( _**x**_ ) _−_ _ϕG_ ( _**x**_ _[′]_ ) _∥_ 2 = min _g,g′∈G ∥g_ _**x**_ _−_ _g_ _[′]_ _**x**_ _[′]_ _∥_ 2 _._ _Define_
_f_ ( _**x**_ ) = _h_ ( _ϕG_ ( _**x**_ )) _._ _Then f_ _∼GP_ (0 _, k_ max) _and it is G-invariant._


_Proof._ Since _g_ is a GP, _f_ is also a GP, and invariance follows from (i). Its covariance kernel is _k_ max
since:

Cov [ _f_ ( _**x**_ ) _, f_ ( _**x**_ _[′]_ )] = Cov [ _h_ ( _ϕG_ ( _**x**_ )) _, h_ ( _ϕG_ ( _**x**_ _[′]_ ))]

= _k_ b( _ϕG_ ( _**x**_ ) _, ϕG_ ( _**x**_ _[′]_ ))


= _κ_ ( min (8)
_g,g_ _[′]_ _∈G_ _[||][g]_ _**[x]**_ _[ −]_ _[g][′]_ _**[x]**_ _[′][||]_ [2][)]


= max (9)
_g,g_ _[′]_ _∈G_ _[κ]_ [(] _[||][g]_ _**[x]**_ _[ −]_ _[g][′]_ _**[x]**_ _[′][||]_ [2][)]

= _k_ max( _**x**_ _,_ _**x**_ _[′]_ ) (10)


where we used (ii) in Equation (8), and monotonicity of _κ_ in Equation (9). Note that compactness of
_G_ guarantees that the minimum in (ii) is indeed achieved, which makes Equation (9) true even when
_κ_ is not necessarily continuous.


A.2 AVERAGING VS MAXIMIZATION WITH DIFFERENT BASE KERNELS


We extend Lemma 2 to the case where _k_ avg and _k_ max are built from _different_ base kernels. The result
shows that even in this more flexible setting, the coincidence of _k_ avg and _k_ max can only occur in
degenerate situations.
**Lemma 6.** _Let k_ b _and k_ b _[′]_ _[be two base kernels such that][ ∥][k]_ [b] _[∥][∞]_ [=] _[ ∥][k]_ b _[′]_ _[∥][∞]_ [= 1] _[ and][ k]_ b _[′]_ [(] _**[x]**_ _[,]_ _**[ x]**_ [) = 1] _[ for]_
_all_ _**x**_ _._ _Let k_ avg _be the group-averaged kernel built from k_ b _and k_ max _be the maximization kernel built_
_from k_ b _[′]_ _[.]_ _[It holds]_

_k_ avg = _k_ max _on the orbit O_ ( _**x**_ _, g_ _**x**_ ) := _{_ ( _h_ _**x**_ _, h_ _[′]_ _g_ _**x**_ ) _,_ _h, h_ _[′]_ _∈G}_


_for every_ _**x**_ _∈X_ _and g_ _∈G, if and only if_


_k_ b( _**x**_ _, g_ _**x**_ ) = _k_ max( _**x**_ _, g_ _**x**_ ) = 1 _for every_ _**x**_ _and g_ _∈G._

_In particular, this forces k_ b _to already exhibit a form of G-invariance on pairs_ ( _**x**_ _, g_ _**x**_ ) _._


_Proof._ ( _⇒_ ) Fix _**x**_ and _g_ _∈G_ . Since by assumption _k_ b _[′]_ [is bounded by][ 1][ and] _[ k]_ b _[′]_ [(] _**[x]**_ _[,]_ _**[ x]**_ [) = 1][:]

1 _≥_ _k_ max( _**x**_ _, g_ _**x**_ ) = max b [(] _[h]_ _**[x]**_ _[, h][′][g]_ _**[x]**_ [)] _[ ≥]_ _[k]_ b _[′]_ [(] _**[x]**_ _[,]_ _**[ x]**_ [) = 1]
_h,h_ _[′]_ _∈G_ _[k][′]_


so _k_ max( _**x**_ _, g_ _**x**_ ) = 1.


Now consider _k_ avg. By definition,

_k_ avg( _**x**_ _, g_ _**x**_ ) = _|G|_ 1 [2]            - _k_ b( _h_ _**x**_ _, h_ _[′]_ _g_ _**x**_ ) _._

_h,h_ _[′]_ _∈G_


Each summand is bounded by 1 and the average is equal to 1 as _k_ avg( _**x**_ _, g_ _**x**_ ) = _k_ max( _**x**_ _, g_ _**x**_ ) = 1.
Therefore each term is equal to 1, which proves _k_ b = _k_ max = 1 on _O_ ( _**x**_ _, g_ _**x**_ ). As this is true for
every _**x**_ _, g_ _∈G_, this shows the result. The converse is immediate.


This shows that even when allowing different base kernels for _k_ avg and _k_ max, equality between
the two kernels requires _k_ b to already be argumentwise _G_ -invariant on pairs ( _**x**_ _, g_ _**x**_ ). This fails for
standard choices (e.g. RBF kernels with translation or rotation groups), so averaging cannot replicate
maximization in practice.


14


B RADIAL INVARIANCE: CLOSED FORM FOR _k_ avg


We prove the formulas provided in Example 3. Let _G_ = SO(2) act on R [2] by in-plane rotations,
and let _k_ b be the RBF kernel with lengthscale _l_ : _k_ b( _**x**_ _,_ _**x**_ _[′]_ ) = exp� _−∥_ _**x**_ _−_ _**x**_ _[′]_ _∥_ [2] 2 _[/]_ [(2] _[l]_ [2][)] �. Writing
_**x**_ = ( _r, θ_ ) and _**x**_ _[′]_ = ( _s, φ_ ) in polar coordinates, we have


   exp _−_ _[r]_ [2][+] _[s]_ [2] _[−]_ [2] _[rs]_ [ cos(] 2 _l_ [2] _[θ][−][φ]_ [+] _[α][−][β]_ [)]
0


2 _l_ [2] _[θ][−][φ]_ [+] _[α][−][β]_ [)] - _dα dβ._


1
_k_ avg( _**x**_ _,_ _**x**_ _[′]_ ) =
(2 _π_ ) [2]


- 2 _π_


0


- 2 _π_


Integrating out the absolute angle and keeping only the relative angle _ψ_ = _θ −_ _φ_ + _α −_ _β_ yields


2 [+] _l_ [2] _[s]_ [2] - _I_ 0� _rsl_ [2] - _,_


      -       _k_ avg( _**x**_ _,_ _**x**_ _[′]_ ) = exp _−_ _[r]_ [2] 2 [+] _l_ [2] _[s]_ [2] _·_ 2 [1] _π_


- 2 _π_


exp� _rsl_ [2] [cos] _[ ψ]_  - _dψ_ = exp� _−_ _[r]_ [2] 2 [+] _l_ [2] _[s]_ [2]
0


where _I_ 0( _z_ ) = 21 _π_ �02 _π_ _e_ _[z]_ [ cos] _[ ψ]_ _dψ_ is the modified Bessel function of order 0.


C AN INTRINSIC PSD PROJECTION _k_ + AND ITS PROPERTIES


We first recall in Appendix C.1 why positive semidefiniteness is essential in the BO framework. Then,
we turn to the intrinsic structure underlying our practical construction.

In the main text we defined a _data-dependent_ kernel _k_ + [(] _[D]_ [)][, obtained by projecting the Gram matrix of]
_k_ max on a finite set of samples _D_ onto the PSD cone and extending it via Nystrom.¨ This finite-sample
construction _k_ + [(] _[D]_ [)] is the star of the show in practice, as it is convenient to compute and exhibits strong
empirical performance. However, its data-dependence can complicate theoretical analysis.

In this appendix, we show that _k_ + [(] _[D]_ [)] is the finite-sample facet of a broader, intrinsic _data-independent_
PSD projection _k_ + of _k_ max which (i) preserves the _G_ -invariance of _k_ max, (ii) coincides with _k_ max
whenever _k_ max is already PSD. Since the PSD projection of _k_ max discussed here can also be applied
to any other indefinite kernel _k_, we directly introduce it for an arbitrary kernel _k_ .


We present the finite-domain “matrix” construction to build intuition in Appendix C.2, and then lift it
to general domains via integral operators.


C.1 WHY PSDNESS OF _k_ MATTERS


In this paper, we consider _k_ = _k_ max and then project it onto a PSD kernel. Although there is _no_
_technical impossibility_ in running a BO loop as in Section 2.1 with a kernel _k_ that is not PSD, [7] doing
so is poorly motivated: the fundamental assumptions underlying BO no longer apply, and the key
quantities lose their meaning. In particular:


- the assumption _f_ _[⋆]_ _∈Hk_ no longer makes sense because _Hk_ is not defined for non-PSD kernels;

- the usual interpretation of BO as maintaining a GP prior whose posteriors provide increasingly
refined approximations of _f_ _[⋆]_ no longer holds (in particular _µt_ and Cov _t_ are no longer GP posterior
mean or covariance), since _k_ is not a valid covariance structure for the prior;

- acquisition functions (UCB, EI, etc.) lose their principled exploration-exploitation meaning and
may now behave unpredictably.


C.2 WARMUP: FINITE DOMAINS


We start on a finite domain _S_ to build intuition. In that case, _k_ + is simply Frobenius-nearest PSD
truncation of the Gram matrix on the _full domain S_, which is unique, basis-independent, preserves
_G_ -invariance, and coincides with _k_ when _k_ is already PSD.


Let _S_ = _{_ _**x**_ 1 _, . . .,_ _**x**_ _N_ _}_ be finite, and let _G_ act on _S_ . Consider any symmetric kernel _k_ on _S_ with
Gram matrix _**K**_ _∈_ R _[N]_ _[×][N]_ (possibly indefinite) given by _**K**_ _ij_ = _k_ ( _**x**_ _i,_ _**x**_ _j_ ). We define _k_ + as the
kernel corresponding to the Frobenius-nearest PSD projection of _**K**_ (Higham, 1988).


7Only step 2 in Section 2.1 may fail if _**K**_ _t_ + _σ_ 02 _**[I]**_ _t_ [is non-invertible.] [One can use a pseudo-inverse or a]
very large _σ_ 0, but the latter makes the posterior variance nearly flat, degenerating the procedure into blind
exploitation.


15


**Lemma 7** (Frobenius PSD projection and explicit form (Higham, 1988)) **.** _The optimization problem_
_**K**_ + := arg min _**P**_ _⪰_ 0 _∥_ _**P**_ _−_ _**K**_ _∥F_ _has_ _a_ _unique_ _solution_ _and,_ _for_ _any_ _eigendecomposition_ _**K**_ =
_Q_ Λ _Q_ _[⊤]_ _, it is given by_

_**K**_ + = _**Q**_ max(0 _,_ **Λ** ) _**Q**_ _[⊤]_ _,_

_where_ max(0 _, ·_ ) _acts entrywise on_ **Λ** _._ _In particular, the matrix_ _**K**_ + _depends only on_ _**K**_ _(not on the_
_chosen eigenbasis), satisfies_ _**K**_ + _⪰_ 0 _, and_ _**K**_ + = _**K**_ _iff_ _**K**_ _⪰_ 0 _._


We _define k_ +, the (Frobenius) PSD projection of _k_, as:


_k_ +( _xi, xj_ ) := ( _**K**_ +) _ij,_ _i, j_ _∈_ [ _N_ ] _._ (11)


**Inheritance of** _G_ **-invariance.** Each element _g_ _∈G_ induces a permutation of the elements of _S_ : let
_πg_ be the permutations of the integers _j_ _∈{_ 1 _, . . ., N_ _}_ defined by _g_ _**x**_ _j_ = _**x**_ _πg_ ( _j_ ). Denote by _**P**_ _g_ the
permutation matrix associated with _πg_ . For every vector _**v**_, the matrix _**P**_ _g_ acts as ( _**P**_ _g_ _**v**_ ) _i_ = _**v**_ _πg−_ 1( _i_ )
which is equivalent to the action on canonical vectors _**P**_ _g_ _**e**_ _j_ = _**e**_ _πg_ ( _j_ ) or ( _**P**_ _g_ ) _ij_ = 1 _i_ = _πg_ ( _j_ ).


Invariance in the first component guarantees _k_ max( _**x**_ _πg_ ( _i_ ) _,_ _**x**_ _j_ ) = _k_ max( _g_ _**x**_ _i,_ _**x**_ _j_ ) = _k_ max( _**x**_ _i,_ _**x**_ _j_ ) for
every _i, j_ _∈{_ 1 _, . . ., N_ _}_, i.e., the rows of _**K**_ = ( _k_ ( _**x**_ _i,_ _**x**_ _j_ )) _i,j_ are invariant under the permutation _πg_,
hence _**P**_ _g_ _**K**_ = _**K**_ . Thus, for any positive integer _m_, _**P**_ _g_ _**K**_ _[m]_ = ( _**P**_ _g_ _**K**_ ) _**K**_ _[m][−]_ [1] = _**K**_ _[m]_ so for any
polynomial _p_ such that _p_ (0) = 0, _**P**_ _gp_ ( _**K**_ ) = _p_ ( _**K**_ ). Now consider a sequence ( _pn_ ) _n_ of polynomials
such that [8] _pn_ (0) = 0 and _|pn_ ( _λ_ ) _−_ max(0 _, λ_ ) _|_ _→_ [for] [any] _[λ]_ [in] [the] [spectrum] [of] _**[K]**_ [.] [In] [the]
_n→∞_ [0]

limit _**P**_ _g_ _**K**_ + = _**K**_ +, hence _k_ + is invariant under the action of _G_ on the first variable ( _k_ +( _g_ _**x**_ _,_ _**x**_ _[′]_ ) =
_k_ +( _**x**_ _,_ _**x**_ _[′]_ )), and invariance along the second one follows by symmetry ( _**K**_ + _**P**_ _g_ _[⊤]_ [=] _**[ K]**_ [+][).] [This shows]
that _k_ + inherits from the _G_ -invariance of _k_ (equivalently, _**P**_ _g_ _**K**_ = _**K**_ = _**KP**_ _g_ _[⊤]_ [for all] _[ g]_ [).] [We collect]
this result in the next lemma.


**Lemma** **8** (Invariance is preserved by the projection) **.** _Consider_ _g_ _∈G._ _If_ _Pg_ _**K**_ = _**K**_ _,_ _then_
_Pg_ _**K**_ + = _**K**_ + = _**K**_ + _Pg_ _[⊤][.]_ _[Hence the projected kernel][ k]_ [+] _[is][ G][-invariant on][ S × S][.]_


**Relation to the practical Nystrom kernel.¨** If the set _D_ = _{_ _**x**_ 1 _, . . .,_ _**x**_ _n}_ used to build _k_ + [(] _[D]_ [)] (Equation (7)) equals the whole domain _D_ = _S_, then _k_ + [(] _[D]_ [)] = _k_ +. Indeed, _k_ + [(] _[D]_ [)][(] _**[x]**_ _[i][,]_ _**[ x]**_ _[j]_ [) =] _**[ K]**_ _[i]_ [:] _**[K]**_ + _[†]_ _**[K]**_ [:] _[j]_ [=]
( _**KK**_ + _[†]_ _**[K]**_ [)] _[ij]_ [= (] _**[K]**_ [+][)] _[ij]_ [on] _[ D × D]_ [, and the latter is the definition of] _[ k]_ [+] [on finite domains.]


We now generalize the matrix considerations above using integral operators. The finite-domain
construction is recovered as a special case.


C.3 GENERAL DEFINITION (VIA INTEGRAL OPERATORS THEORY)


We lift the finite-domain construction of the previous subsection to general domains by viewing
_k_ as a Hilbert–Schmidt operator and defining _k_ + as the positive part of _Tk_ ; this yields a PSD,
data-independent kernel that inherits any _G_ -invariance and equals _k_ whenever _k_ is PSD.


Let ( _S, T, µ_ ) be a probability space. For a measurable, symmetric kernel _k_ : _S_ _× S_ _→_ R with
_k_ _∈_ _L_ [2] ( _µ ⊗_ _µ_ ), let the (compact, self-adjoint) Hilbert-Schmidt operator _Tk_ : _L_ [2] ( _µ_ ) _→_ _L_ [2] ( _µ_ ) be


                ( _Tkf_ )( _**x**_ ) = _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _f_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ _[′]_ ) _._

_S_


(Note that in the finite-domain case, _f_ is a vector indexed by the domain and if _µ_ is the uniform
measure then _Tk_ is simply multiplication by the Gram matrix _**K**_ normalized by the domain size.) By
the spectral theorem, there exist ( _λi, ϕi_ ) _i≥_ 1 with _{ϕi}_ orthonormal in _L_ [2] ( _µ_ ) and ( _λi_ ) _∈_ _ℓ_ [2] (possibly
of mixed signs) such that _Tk_ = [�] _i≥_ 1 _[λ][i][ ϕ][i][ ⊗]_ _[ϕ][i]_ [in] _[ L]_ [2][(] _[µ]_ [)][ where for every] _[ u, v]_ _[∈]_ _[L]_ [2][(] _[µ]_ [)][,] _[ u][ ⊗]_ _[v]_ [ is]

the rank-one operator _L_ [2] ( _µ_ ) _→_ _L_ [2] ( _µ_ ) such that ( _u ⊗_ _v_ ) _f_ := _⟨f, v⟩_ _u_ for every _f_ _∈_ _L_ [2] ( _µ_ ).


8We can impose _pn_ (0) = 0 since _f_ (0) = 0. Indeed, take _pn_ ( _λ_ ) = _qn_ ( _λ_ ) _−_ _qn_ (0) where _qn_ is a sequence
given by Weierstrass’ theorem, which converges to _f_ ( _λ_ ) = max(0 _, λ_ ) on the spectrum of _**K**_ . We have
_|pn_ ( _λ_ ) _−_ _f_ ( _λ_ ) _| ≤|qn_ ( _λ_ ) _−_ _f_ ( _λ_ ) _|_ + _|qn_ (0) _|_ and because _f_ (0) = 0 we get _|qn_ (0) _|_ = _|qn_ (0) _−_ _f_ (0) _| →_ 0.


16


**Generic definition of** _k_ + **via operator theory.** Define the positive part of _Tk_ =

[�]


**Generic definition of** _k_ + **via operator theory.** Define the positive part of _Tk_ = [�] _i_ _[λ][i][ ϕ][i][ ⊗]_ _[ϕ][i]_ [by]

_Tk_ [+] [:=][ �] _i_ [(] _[λ][i]_ [)][+] _[ ϕ][i][ ⊗]_ _[ϕ][i]_ [, where][ (] _[t]_ [)][+] [= max] _[{][t,]_ [ 0] _[}]_ [.] [Since][ �] _i_ [((] _[λ][i]_ [)][+][)][2] _[≤]_ [�] _i_ _[λ]_ _i_ [2] _[<][ ∞]_ [, the series]


_i_ [((] _[λ][i]_ [)][+][)][2] _[≤]_ [�]


_i_ [(] _[λ][i]_ [)][+] _[ ϕ][i][ ⊗]_ _[ϕ][i]_ [, where][ (] _[t]_ [)][+] [= max] _[{][t,]_ [ 0] _[}]_ [.] [Since][ �]


_i_ _[λ]_ _i_ [2] _[<][ ∞]_ [, the series]


_k_ +( _**x**_ _,_ _**x**_ _[′]_ ) := �( _λi_ )+ _ϕi_ ( _**x**_ ) _ϕi_ ( _**x**_ _[′]_ ) ( _µ ⊗_ _µ_ -a.e.) _._ (12)

_i≥_ 1


converges in _L_ [2] ( _µ ⊗_ _µ_ ) and defines a kernel _µ ⊗_ _µ_ -almost everywhere. By construction [9] _Tk_ + = _Tk_ [+][,]
hence _k_ + is PSD as a kernel a.e., and PSD in the operator sense: - _f, Tk_ + _f_ - _≥_ 0 for all _f_ _∈_ _L_ [2] ( _µ_ ).
In particular, if _k_ was already PSD (all _λi_ _≥_ 0), then _k_ + = _k_ (up to null sets). It also inherits
_G_ -invariance of _k_ if _k_ is indeed invariant (the proof mimics the finite-domain case, we give the full
details for completeness in Appendix C.7).


C.4 FROM THE FINITE-SAMPLE PROJECTION TO THE INTRINSIC LIMIT: WHAT CONVERGES TO
WHAT?


We relate the practical, data-dependent Nystrom kernel¨ _k_ + [(] _[D]_ [)] (Equation (7)) to the intrinsic _k_ +: under
iid sampling, the empirical spectra of _k_ + [(] _[D]_ [)] _[/][|D|]_ [ converge to that of] _[ T][k]_ + [, with rates under mild moment]
assumptions. This shows that eigendecay-based regret analysis


**Notations.** Let _X_ 1 _, X_ 2 _, · · ·_ _∼_ _µ_ i.i.d. and _Dn_ = _{X_ 1 _, . . ., Xn}_ . We write _**K**_ _n_ := _k_ ( _Dn, Dn_ ),
_**K**_ _n_ [+] [:=] [arg min] _**[P]**_ _[ ⪰]_ [0] _[∥]_ _**[P]**_ _[−]_ _**[K]**_ _[n][∥][F]_ [,] _**K**_ [˜] _n_ := _**K**_ _n/n_, and recall that the practical (data-dependent)
kernel defined in Equation (7) is

_k_ + [(] _[D][n]_ [)] ( _**x**_ _,_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _, Dn_ ) ( _**K**_ _n_ [+][)] _[†][ k]_ [(] _[D][n][,]_ _**[ x]**_ _[′]_ [)] _[.]_


We denote by _λ_ ( _T_ ) the (ordered, nonincreasing, each counted with its multiplicity) sequence of eigenvalues of a compact self-adjoint operator _T_, and by _δ_ 2� _λ_ ( _T_ ) _, λ_ ( _S_ )� := �� _i_ _[|][λ][i]_ [(] _[T]_ [)] _[ −]_ _[λ][i]_ [(] _[S]_ [)] _[|]_ [2][�][1] _[/]_ [2]

the spectral _ℓ_ 2 distance. For symmetric matrices _**M**_, _λ_ ( _**M**_ ) denotes the nonincreasing sequence
of eigenvalues of _**M**_ (with multiplicity) padded with an infinite number of zeros. For a bounded
operator _A_, _∥A∥_ HS and _∥A∥_ op denote the Hilbert-Schmidt and operator norms, respectively. We
include in Appendix C.5 a reminder on the different notions of norms and convergence, and we now
recall the essentials.


**Relations** **between** **convergence** **notions.** For compact self-adjoint operators: (i)
max - _δ_ 2� _λ_ ( _Tn_ ) _, λ_ ( _T_ )� _, ∥Tn −_ _T_ _∥_ op� _≤_ _∥Tn_ _−_ _T_ _∥_ HS (Reed & Simon, 1972; Bhatia & Elsner, 1994); (ii) converse inequalities do not hold in infinite dimension (see Appendix C.5 for
examples). Thus, HS convergence is the strongest notion of convergence we manipulate here.


We now present convergence guarantees of the data-dependent construction _k_ + [(] _[D][n]_ [)] _/n_ to the intrinsic
_k_ + under progressively stronger assumptions. With minimal assumptions we obtain almost-sure
spectral consistency in the _δ_ 2 metric; with stronger assumptions we obtain quantitative rates in HS
norm (hence also spectral _ℓ_ 2 in probability).


**(a) Weak a.s. spectral consistency of positive parts (minimal assumptions).**

**Proposition 9.** _Assume the symmetric (not necessarily PSD) kernel k is in L_ [2] ( _µ ⊗_ _µ_ ) _so that Tk_ _is_
_Hilbert-Schmidt._ _Let_ _S_ [�] _n_ : _L_ [2] ( _µn_ ) _→_ _L_ [2] ( _µn_ ) _be the integral operator with kernel k_ + [(] _[D][n]_ [)] ( _**x**_ _,_ _**x**_ _[′]_ ) _/n_
_defined by:_


( _S_ [�] _nf_ )( _**x**_ ) = [1]

_n_


_n_

- _k_ + [(] _[D][n]_ [)] ( _**x**_ _, Xj_ ) _f_ ( _Xj_ ) _._ (13)

_j_ =1


_Assume the Xi_ _are pairwise distinct almost surely._ _Then, almost surely,_


             -             -             -             - [�]
_δ_ 2 _λ_ �� _Sn_ _,_ _λ_ _Tk_ + _−→_
_n→∞_ [0] _[.]_


9   - ��   Indeed, by definition ( _Tk_ + _f_ )( _**x**_ ) = _S_ _i≥_ 1 [(] _[λ][i]_ [)][+] _[ϕ][i]_ [(] _**[x]**_ [)] _[ϕ][i]_ [(] _**[x]**_ _[′]_ [)] _f_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ _[′]_ ) =

- ��� - - - [+] 


9Indeed, by definition ( _Tk_ + _f_ )( _**x**_ ) = 


��
_S_


���
_i≥_ 1 [(] _[λ][i]_ [)][+] _[ ⟨][f, ϕ][i][⟩]_ _[ϕ][i]_ [(] _**[x]**_ [) =]


_i≥_ 1 [(] _[λ][i]_ [)][+] _[ ϕ][i][ ⊗]_ _[ϕ][i]_ - _f_ �( _**x**_ ) = - _Tk_ [+] _[f]_ �( _**x**_ ).


17


_Proof._ Let _**K**_ _n_ be the empirical operator on R _[n]_ with matrix _n_ [1] [(] _[k]_ [(] _[X][i][, X][j]_ [))] _[i,j]_ [and] [let] _[λ]_ [(] _**[K]**_ _[n]_ [)] [be]

its ordered spectrum (nonincreasing, with multiplicity) padded with an infinite number of zeros.
Theorem 3.1 of Koltchinskii & Gin´e (2000) shows that _δ_ 2( _λ_ ( _**K**_ _n_ ) _, λ_ ( _Tk_ )) _→_ 0 as _n →∞_ .

Let _**K**_ _n_ [+] [be the positive part of] _**[ K]**_ _[n]_ [(i.e., its Frobenius PSD projection).] [Since] _[ λ]_ _[�→]_ [max(0] _[, λ]_ [)][ is]
1-Lipschitz, we have for any operators _T, S_ :


_|_ max(0 _, λi_ ( _T_ )) _−_ max(0 _, λi_ ( _S_ )) _| ≤_   
_i_ _i_


_δ_ 2( _λ_ ( _T_ +) _, λ_ ( _S_ +)) = 


_|λi_ ( _T_ ) _−λi_ ( _S_ ) _|_ = _δ_ 2( _λ_ ( _T_ ) _, λ_ ( _S_ )) _._

_i_


We deduce that _δ_ 2( _λ_ ( _**K**_ _n_ [+][)] _[, λ]_ [(] _[T][k]_ + [))] _[ →]_ [0][ as] _[ n][ →∞]_ [.]

It remains to observe that the spectrum of _**K**_ _n_ [+] [as an operator on][ R] _[n]_ [ is the same as] _[S]_ [�] _[n]_ [:] _[ L]_ [2][(] _[µ][n]_ [)] _[ →]_
_L_ [2] ( _µn_ ). This identification is standard (e.g., see above Equation 1.2 in Koltchinskii & Gine (2000)).´
For completeness, we include the formal arguments of Koltchinskii & Gine (2000) in Lemma 12,´
which shows that we can identify the spectrum of _k_ + [(] _[D][n]_ [)] ( _Dn, Dn_ ) _/n_ with the one of _**K**_ _n_ [+] [a.s.] [if the]
iid _Xi_ _∼_ _µ_ are pairwise distinct a.s, which is true as soon as _µ_ is non-atomic; otherwise one can
index the _distinct_ atoms and work in R _[m]_ with _m_ = #supp( _µn_ ), obtaining the same spectral identity
on that subspace.


**(b) Expected HS convergence with** _O_ ( _n_ _[−]_ [1] _[/]_ [2] ) **rate (stronger assumption).** Define the empirical
integral operator ( _Tnf_ )( _**x**_ ) := _n_ [1] - _ni_ =1 _[k]_ [(] _**[x]**_ _[, X][i]_ [)] _[f]_ [(] _[X][i]_ [)][ and] _[ D][n]_ [:=] _[ T][n][ −]_ _[T][k]_ [.] [Let][ (] _[λ][i][, ϕ][i]_ [)] _[i][≥]_ [1] [be an]

eigensystem of _Tk_ in _L_ [2] ( _µ_ ). Assume the following fourth-order summability condition holds:


_C_ := - _λ_ [2] _i_

_i,j≥_ 1


_ϕi_ ( _**x**_ ) [2] _ϕj_ ( _**x**_ ) [2] _dµ_ ( _**x**_ ) _<_ _∞._ (14)
_S_


**Proposition 10** (Expected HS rate) **.** _Under k_ _∈_ _L_ [2] ( _µ ⊗_ _µ_ ) _and_ (14) _,_


E� _∥Dn∥_ [2] HS� _≤_ _[C]_


_n_ _[.]_


~~�~~
_n_ _[,]_ E� _∥Dn∥_ HS� _≤_ _Cn_


_Consequently, ∥Dn∥_ HS = _O_ P( _n_ _[−]_ [1] _[/]_ [2] ) _and therefore using the same notations as in Proposition 9_


      -       -       -       -       _δ_ 2 _λ_ ( _**K**_ _n_ [+][)] _[, λ]_ [(] _[T]_ [ +] _k_ [)] = _O_ P( _n_ _[−]_ [1] _[/]_ [2] ) _,_ _δ_ 2 _λ_ �� _Sn_ _, λ_ ( _Tk_ +) = _O_ P( _n_ _[−]_ [1] _[/]_ [2] ) _._


_Proof._ Fix any _f_ _∈_ _L_ [2] ( _µ_ ). By Fubini-Tonelli for non-negative functions, we have:

E� _∥Dnf_ _∥_ [2] _L_ [2] ( _µ_ )� =          - E��( _Dnf_ )( _**x**_ )�2 [�] _dµ_ ( _**x**_ ) _._

_S_


By definition


_k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _f_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ _[′]_ )
_S_


( _Dnf_ )( _**x**_ ) = [1]

_n_


_n_


        
- _k_ ( _**x**_ _, Xi_ ) _f_ ( _Xi_ ) _−_


_i_ =1


where the randomness comes from the i.i.d. _Xi_ _∼_ _µ_ . Hence E�( _Dnf_ )( _**x**_ )� = 0 and for any fixed _**x**_


E��( _Dnf_ )( _**x**_ )�2 [�] = Var�( _Dnf_ )( _**x**_ )� = [1]


   -   
[1] _k_ ( _**x**_ _, X_ ) _f_ ( _X_ ) _≤_ [1]

_n_ [Var] _n_


_n_


_k_ ( _**x**_ _,_ _**x**_ _[′]_ ) [2] _f_ ( _**x**_ _[′]_ ) [2] _dµ_ ( _**x**_ _[′]_ ) _._
_S_


The Hilbert-Schmidt spectral theorem gives the expansion _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) = [�] _i_ _[λ][i][ϕ][i]_ [(] _**[x]**_ [)] _[ϕ][i]_ [(] _**[x]**_ _[′]_ [)][ in] _[ L]_ [2][(] _[µ][ ⊗]_

_µ_ ), with ( _λi_ ) _i_ _∈_ _ℓ_ [2] and ( _ϕi_ ) _i_ an orthonormal set of _L_ [2] ( _µ_ ) (see Equation 3.2 in Koltchinskii & Gine´
(2000), Corollary 5.4 in Conway (2007)). Thus

    -    


The Hilbert-Schmidt spectral theorem gives the expansion _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) = [�]


E��( _Dnf_ )( _**x**_ )�2 [�] _dµ_ ( _**x**_ ) _≤_ [1]
_S_ _n_


_n_


_k_ ( _**x**_ _,_ _**x**_ _[′]_ ) [2] _f_ ( _**x**_ _[′]_ ) [2] _dµ_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ )
_S_


= - _λiλj_


_i,j_


_ϕi_ ( _**x**_ _[′]_ ) _ϕj_ ( _**x**_ _[′]_ ) _f_ ( _**x**_ _[′]_ ) [2] _⟨ϕi, ϕj⟩_
_S_ ��� =1 _i_ = _j_


_dµ_ ( _**x**_ _[′]_ )


= - _λ_ [2] _i_


_i_


_ϕi_ ( _**x**_ _[′]_ ) [2] _f_ ( _**x**_ _[′]_ ) [2] _dµ_ ( _**x**_ _[′]_ ) _._
_S_


18


Taking _f_ = _ϕj_ for a fixed _j_ yields


E� _∥Dnϕj∥_ [2] _L_ [2] ( _µ_ )� _≤_ _n_ [1]


- _λ_ [2] _i_


_i_


_ϕi_ ( _**x**_ _[′]_ ) [2] _ϕj_ ( _**x**_ _[′]_ ) [2] _dµ_ ( _**x**_ _[′]_ ) _._
_S_


Since _∥Dnf_ _∥_ [2] HS [=][ �] _j_ _[∥][D][n][ϕ][j][∥]_ _L_ [2] [2] ( _µ_ ) [, we get the main claim:]

E� _∥Dn∥_ [2] HS� _≤_ _[C]_ _n_ _[.]_


Jensen gives the bound for E _∥Dn∥_ HS. Finally, _δ_ 2( _λ_ ( _**K**_ _n_ ) _, λ_ ( _Tk_ )) _≤∥Dn∥_ HS (Hoffman-Wielandt
inequality in infinite dimension (Bhatia & Elsner, 1994)), and _λ �→_ max(0 _, λ_ ) is 1-Lipschitz on R,
hence the spectral bound probability claim using Markov’s inequality, and Lemma 12 transfers this
claims to _S_ [�] _n_ .
**Remark 11** (On assumption (14)) **.** _Condition_ (14) _is a fourth-order integrability requirement that_
_controls eigenfunction overlaps._ _It is standard in random Nystrom analyses (see, e.g., Equations (4.3)¨_
_and (4.11) of Koltchinskii & Gine (2000)) and stronger than´_ _k_ _∈_ _L_ [2] _, but it yields a dimension-free_
_O_ ( _n_ _[−]_ [1] _[/]_ [2] ) _rate in HS norm._


**(c)** **High-probability** **HS** **rates** **(heavier** **but** **more** **precise).** Under slightly stronger _L_ [4] -type
conditions on eigenfunctions, the section 4 in Koltchinskii & Gine (2000) gives more more precise´
statements on the rates in Proposition 10, and we directly refer the reader to it.


**Application** **to** _k_ max **and** **to** **the** **BO** **kernels** **in** **the** **paper.** When _k_ = _k_ max is bounded on a
compact domain _S_ (as in all our experiments), _k_ _∈_ _L_ [2] ( _µ ⊗_ _µ_ ) for any probability measure _µ_ on _S_,
so _Tk_ max is Hilbert-Schmidt and Proposition 9 applies. In particular, the integral operator associated
with _k_ + [(] _[D][n]_ [)] _/n_, called _S_ [�] _n_ (Equation (13)) satisfies

            -             -             -             - [�] a.s.
_δ_ 2 _λ_ �� _Sn_ _,_ _λ_ _Tk_ + _−−−−→_ 0 _._
_n→∞_


This clarifies the two objects introduced in the main text: the _intrinsic_ _k_ + is the unique dataindependent target, while the _practical_ kernel _k_ + [(] _[D][n]_ [)] (finite PSD projection + Nystrom) is an on-path¨
approximation whose spectrum converges (once normalized by _n_ ) to that of _k_ + under i.i.d. sampling.


The following subsections are only optional complementary materials added to help building intuitions
on the convergence results stated above.


C.5 REMINDERS ON THE DIFFERENT TYPE OF CONVERGENCES FOR BOUNDED LINEAR
OPERATORS


This subsection recalls standard notions of operator convergence, included only as background to
help build intuition for the convergence results above.


**Definitions (operator norm, HS norm, spectral distance).** Let _H_ be a separable Hilbert space
with orthonormal basis _{ei}i≥_ 1. For a bounded linear operator _T_ : _H→H_,


�� �1 _/_ 2
_∥T_ _∥_ op := sup _∥Tf_ _∥H,_ _∥T_ _∥_ HS := _∥Tei∥_ [2] _H_ _._
_∥f_ _∥H_ =1 _i≥_ 1


The HS norm is basis-independent. When _T_ is an _integral_ operator with kernel _k_ _∈_ _L_ [2] ( _µ ⊗_ _µ_ ) on
_L_ [2] ( _µ_ ) (Reed & Simon, 1972)


��
_∥T_ _∥_ [2] HS [=] _|k_ ( _x, y_ ) _|_ [2] _dµ_ ( _x_ ) _dµ_ ( _y_ ) _._

_S×S_


For finite matrices, _∥A∥_ HS = _∥A∥F_ (Frobenius). We say _Tn →_ _T_ in HS norm if _∥Tn −_ _T_ _∥_ HS _→_
0, and we say _Tn_ _→_ _T_ spectrally if _δ_ 2� _λ_ ( _Tn_ ) _, λ_ ( _T_ )� _→_ 0, where we recall that _λ_ ( _T_ ) is the
_ordered_ eigenvalues of a compact self-adjoint operator _T_, and where the spectral _ℓ_ 2-distance is
_δ_ 2( _λ_ ( _T_ ) _, λ_ ( _S_ )) := �� _i_ _[|][λ][i]_ [(] _[T]_ [)] _[ −]_ _[λ][i]_ [(] _[S]_ [)] _[|]_ [2][�][1] _[/]_ [2][.]


19


**Which convergences matter, and how they relate (reminders on well-known facts).** We compare
three notions: (i) _operator norm_ convergence _∥Tn_ _−T_ _∥_ op _→_ 0; (ii) _Hilbert-Schmidt (HS)_ convergence
_∥Tn_ _−_ _T_ _∥_ HS _→_ 0; (iii) _spectral_ convergence in _δ_ 2, i.e., _δ_ 2� _λ_ ( _Tn_ ) _, λ_ ( _T_ )� := �� _i_ _[|][λ][i]_ [(] _[T][n]_ [)] _[−]_

_λi_ ( _T_ ) _|_ [2][�][1] _[/]_ [2] _→_ 0, where _λ_ ( _·_ ) denotes the ordered eigenvalues of a compact self-adjoint operator. We
recall the following well-known facts, useful to grasp the convergence results we state next.


**(1)** **HS** = _⇒_ **spectral** _δ_ 2 **.** For compact self-adjoint operators the (infinite-dimensional) HoffmanWielandt inequality yields (Bhatia & Elsner, 1994)


_δ_ 2� _λ_ ( _Tn_ ) _, λ_ ( _T_ )� _≤∥Tn −_ _T_ _∥_ HS _._


**(2)** **HS** = _⇒_ **operator** **norm.** For every Hilbert-Schmidt operator _S_, _∥S∥_ op _≤∥S∥_ HS. Indeed
for unit vectors _x, y_ _∈_ _H_, using _x_ = [�] _i_ _[⟨][x, e][i][⟩][e][i]_ [,] [we] [have] _[⟨][Sx, y][⟩]_ [=] [�] _i∈I_ _[⟨][x, e][i][⟩⟨][Se][i][, y][⟩][.]_ [By]


_i_ _[⟨][x, e][i][⟩][e][i]_ [,] [we] [have] _[⟨][Sx, y][⟩]_ [=] [�]


for unit vectors _x, y_ _∈_ _H_, using _x_ = [�] _i_ _[⟨][x, e][i][⟩][e][i]_ [,] [we] [have] _[⟨][Sx, y][⟩]_ [=] [�] _i∈I_ _[⟨][x, e][i][⟩⟨][Se][i][, y][⟩][.]_ [By]

Cauchy-Schwarz:


_|⟨x, ei⟩|_ [2][�][1] _[/]_ [2][��]

_i∈I_ _i∈I_


��
_|⟨Sx, y⟩| ≤_


_|⟨Sei, y⟩|_ [2][�][1] _[/]_ [2] _._

_i∈I_


The first factor equals _∥x∥_ = 1, and for the second we use _|⟨Sei, y⟩| ≤∥Sei∥∥y∥_ = _∥Sei∥_ to get

    - [2]     - [2] [2]


- _|⟨Sei, y⟩|_ [2] _≤_ 

_i∈I_ _i∈I_


- _∥Sei∥_ [2] = _∥S∥_ [2] HS _[.]_

_i∈I_


Hence _|⟨Sx, y⟩| ≤∥S∥_ HS. Taking the supremum over all unit _y_ gives


_∥Sx∥_ = sup _|⟨Sx, y⟩| ≤∥S∥_ HS _,_
_∥y∥_ =1


and then taking the supremum over all unit _x_ yields


_∥S∥_ op = sup _∥Sx∥≤∥S∥_ HS _._
_∥x∥_ =1


**(3)** **Spectral** _δ_ 2 **does** _**not**_ **imply** **HS** **nor** **operator** **norm.** Even if eigenvalues match in _ℓ_ 2, the
operators may be far in norm because eigenvectors can rotate. Let _T_ = diag(1 _,_ 1 _/_ 2 _,_ 1 _/_ 3 _, . . ._ ) in the
canonical basis ( _ei_ ) _i≥_ 1, and let _Un_ swap _e_ 1 and _en_ . Set _Tn_ := _UnTUn_ _[∗]_ [.] [Then] _[ λ]_ [(] _[T][n]_ [) =] _[ λ]_ [(] _[T]_ [)][ for all]
_n_ (same ordered spectrum), so _δ_ 2( _λ_ ( _Tn_ ) _, λ_ ( _T_ )) = 0. Yet _∥_ ( _Tn −_ _T_ ) _e_ 1 _∥_ = _∥_ ( _UnTUn_ _[∗]_ _[−]_ _[T]_ [)] _[e]_ [1] _[∥]_ [=]
_∥_ (1 _/n −_ 1) _e_ 1 _∥_ = 1 _−_ 1 _/n_, hence _∥Tn −_ _T_ _∥_ op _≥_ 1 _−_ 1 _/n →_ 1 and, a fortiori, _∥Tn −_ _T_ _∥_ HS _̸→_ 0.


**(4) Operator norm does** _**not**_ **imply spectral** _δ_ 2 **.** Let _T_ = 0 and _Tn_ be diagonal with the first _mn_
entries equal to _εn_ and the rest 0. Choose _εn_ := _n_ _[−]_ [1] _[/]_ [2] and _mn_ := _n_ . Then _∥Tn∥_ op = _εn_ _→_ 0 but

_δ_ 2� _λ_ ( _Tn_ ) _, λ_ ( _T_ )� = �� _mi_ =1 _n_ _[ε]_ _n_ [2] �1 _/_ 2 = - _n ·_ (1 _/n_ ) = 1.


**(5) Two useful corollaries.** (a) Spectral _δ_ 2-convergence implies convergence of the _largest_ eigenvalue,
since sup _i |λi_ ( _Tn_ ) _−_ _λi_ ( _T_ ) _|_ _≤_ _δ_ 2( _λ_ ( _Tn_ ) _, λ_ ( _T_ )). (b) Operator-norm convergence forces uniform
eigenvalue deviations to vanish by Weyl’s inequality: sup _i |λi_ ( _Tn_ ) _−_ _λi_ ( _T_ ) _| ≤∥Tn −_ _T_ _∥_ op, but it
does _not_ control the _ℓ_ 2-sum of all deviations.


_Takeaway._ HS is the strongest notion here: it simultaneously implies spectral _δ_ 2-convergence (and
thus convergence of eigenvalue-based quantities) and operator-norm convergence. The converses fail
in infinite dimension because eigenvectors can drift and an infinite number of tiny eigenvalue errors
can accumulate.


C.6 IDENTIFICATION OF THE SPECTRUM OF AN EMPIRICAL OPERATOR IN _L_ [2] ( _µn_ ) AND ITS
MATRIX COUNTERPART


Here we show how the spectrum of the empirical operator can be identified with that of its matrix
form. This is complementary material meant to clarify how operator-level and matrix-level viewpoints
connect (which is useful, e.g., in the proof of Proposition 9).


20


**Lemma 12** (Empirical Nystrom spectral identity)¨ **.** _Let_ _**K**_ _n_ := _n_ [1] - _k_ ( _**x**_ _i,_ _**x**_ _j_ )� _ni,j_ =1 _[and let]_ _**[ K]**_ _n_ [+] _[be]_

_its_ _spectral_ _positive_ _part_ _(the_ _Frobenius-nearest_ _PSD_ _projection)._ _Define_ _the_ _empirical_ _measure_
_µn_ := [1] - _ni_ =1 _[δ]_ _**[x]**_ _i_ _[and the Nystr¨om kernel]_


**Lemma 12** (Empirical Nystrom spectral identity)¨ **.** _Let_ _**K**_ _n_ := [1]


_n_ [1] - _ni_ =1 _[δ]_ _**[x]**_ _i_ _[and the Nystr¨om kernel]_


_k_ + [(] _[D][n]_ [)] ( _**x**_ _,_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _, Dn_ ) ( _**K**_ _n_ [+][)] _[†][ k]_ [(] _[D][n][,]_ _**[ x]**_ _[′]_ [)] _[.]_

_Let_ _S_ [�] _n_ : _L_ [2] ( _µn_ ) _→_ _L_ [2] ( _µn_ ) _be the integral operator with kernel k_ + [(] _[D][n]_ [)] ( _**x**_ _,_ _**x**_ _[′]_ ) _/n, i.e._


( _S_ [�] _nf_ )( _**x**_ ) = [1]

_n_


_n_

- _k_ + [(] _[D][n]_ [)] ( _**x**_ _,_ _**x**_ _j_ ) _f_ ( _**x**_ _j_ ) _._

_j_ =1


_The_ _map_ _E_ : _L_ [2] ( _µn_ ) _→_ R _[n]_ _,_ _Ef_ := ~~_√_~~ 1 _n_ - _f_ ( _**x**_ 1) _, . . ., f_ ( _**x**_ _n_ )� _⊤,_ _is_ _an_ _isometry:_ _∥Ef_ _∥_ R _n_ =
_∥f_ _∥L_ 2( _µn_ ) _, and we have the intertwining identity_

_E_ _S_ [�] _n_ = _**K**_ _n_ [+] _[E.]_

_If, in addition, the sample points_ _**x**_ 1 _, . . .,_ _**x**_ _n are pairwise distinct, then E is an isometric isomorphism_
_(hence invertible) and_

_λ_ �� _Sn_          - = _λ_          - _**K**_ _n_ [+]          - = _λ_          - _k_ + [(] _[D][n]_ [)] ( _Dn, Dn_ ) _/n_          - _._


_Proof._ First note the on-sample identity _k_ + [(] _[D][n]_ [)] ( _**x**_ _i,_ _**x**_ _j_ ) = ( _**K**_ [+] ) _ij_ for the unscaled _**K**_ =
( _k_ ( _**x**_ _i,_ _**x**_ _j_ )) _i,j_, which follows from _**K**_ ( _**K**_ [+] ) _[†]_ _**K**_ = _**K**_ [+] . Hence _k_ + [(] _[D][n]_ [)] ( _Dn, Dn_ ) = _**K**_ [+] and therefore
_k_ + [(] _[D][n]_ [)] ( _Dn, Dn_ ) _/n_ = _**K**_ _n_ [+][.]

For _f_ _∈_ _L_ [2] ( _µn_ ) and each _i ∈{_ 1 _, . . ., n}_,


_n_
�( _**K**_ _n_ [+][)] _[ij]_ _[f]_ [(] _[X][j]_ [) =] _[ √][n]_ - _**K**_ _n_ [+] _[Ef]_ 
_j_ =1


_√_
_n_  - _ES_ [�] _nf_  


[1]
_i_ [= (] _[S]_ [�] _[n][f]_ [)(] _**[x]**_ _[i]_ [) =] _n_


_n_

- _k_ + [(] _[D][n]_ [)] ( _**x**_ _i,_ _**x**_ _j_ ) _f_ ( _**x**_ _j_ ) =

_j_ =1


_i_ _[,]_


which proves _E_ _S_ [�] _n_ = _**K**_ _n_ [+] _[E]_ [.] [Since] _[ E]_ [is an isometry by definition of the] _[ L]_ [2][(] _[µ][n]_ [)][ inner product, if]
the _Xi_ are pairwise distinct then _E_ is bijective and conjugates _S_ [�] _n_ with _**K**_ _n_ [+][, so the spectra (wi][th]
multiplicities) coincide.


C.7 PROOF OF _G_ -INVARIANCE OF _k_ + FOR GENERAL DOMAINS


We conclude this appendix with the formal proof that _k_ + defined in (12) inherits from any groupinvariance of _k_ . This proof is not needed for the main results but is included for completeness. It
makes explicit why _k_ + preserves any _G_ -invariance of _k_ . The proof follows the one for finite domains
but is heavier in notations because it is now stated using integral operators to generalize the matrix
manipulations of finite domains. For finite domains, denoting by _**K**_ the Gram matrix of _k_ over
the whole domain and _**P**_ _g_ the permutation matrix induced by the action of _g_ _∈G_ on the domain,
invariance of _k_ is equivalent to _**P**_ _g_ _**K**_ = _**KP**_ _g_ _[⊤]_ [=] _**[K]**_ [.] [Thus any polynomial] _[ p]_ [(] _**[K]**_ [)][ of] _**[ K]**_ [such that]
_p_ (0) = 0 inherits from this invariance since we still have _**P**_ _gp_ ( _**K**_ ) = _p_ ( _**K**_ ) _**P**_ _g_ _[⊤]_ [=] _[ p]_ [(] _**[K]**_ [)][.] [And at the]
limit, we get invariance of _**K**_ +. Here, we mimic this proof, and we start by introducing the equivalent
integral operator form of the characterization _**P**_ _g_ _**K**_ = _**KP**_ _g_ _[⊤]_ [=] _**[ K]**_ [for general domains.]

**Lemma 13** (Kernel invariance _⇐⇒_ operator commutation) **.** _Let_ ( _S, T, µ_ ) _be a probability space_
_and let G_ _act measurably on S._ _Assume µ is G-invariant._ _Let Ug_ : _L_ [2] ( _µ_ ) _→_ _L_ [2] ( _µ_ ) _be the unitary_
_representation_ ( _Ugf_ )( _**x**_ ) := _f_ ( _g_ _[−]_ [1] _**x**_ ) _._ _Let_ _k_ _∈_ _L_ [2] ( _µ ⊗_ _µ_ ) _be_ _a_ _symmetric_ _kernel_ _with_ _integral_
_operator_ ( _Tkf_ )( _**x**_ ) = - _S_ _[k]_ [(] _**[x]**_ _[,]_ _**[ x]**_ _[′]_ [)] _[f]_ [(] _**[x]**_ _[′]_ [)] _[ dµ]_ [(] _**[x]**_ _[′]_ [)] _[.][ Then the following are equivalent:]_

(i) _k is argumentwise G-invariant:_ _k_ ( _g_ _**x**_ _,_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _, g_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _for µ ⊗_ _µ-a.e._ ( _**x**_ _,_ _**x**_ _[′]_ ) _and all_
_g_ _∈G._
(ii) _Tk_ _satisfies UgTk_ = _TkUg_ = _Tk_ _on L_ [2] ( _µ_ ) _for all g_ _∈G._


_Proof._ _(i)⇒(ii)._ For any _f_ _∈_ _L_ [2] ( _µ_ ),


                   ( _UgTkf_ )( _**x**_ ) = ( _Tkf_ )( _g_ _[−]_ [1] _**x**_ ) = _k_ ( _g_ _[−]_ [1] _**x**_ _,_ _**x**_ _[′]_ ) _f_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ _[′]_ ) _._


21


By invariance of _k_ in the first argument _UgTk_ = _Tk_ . Hence _Tk_ _[∗][U]_ _g_ _[ ∗]_ [=] _[ T][ ∗]_ _k_ [and] _[ T][ ∗]_ _k_ [=] _[ T][k]_ [(self-adjoint)]
and _Ug_ _[∗]_ [=] _[ U]_ _g_ _[−]_ [1] [so] _[ T][k][U]_ _g_ _[−]_ [1] [=] _[ T][k]_ [.] [This is true for all] _[ g]_ _[∈G]_ [hence] _[ U][g][T][k]_ [=] _[ T][k][U][g]_ [=] _[ T][k]_ [.]


_(ii)⇒(i)._ For _φ, ψ_ _∈_ _L_ [2] ( _µ_ ),

��
_k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _φ_ ( _**x**_ ) _ψ_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ ) _dµ_ ( _**x**_ _[′]_ ) = _⟨φ, Tkψ⟩_ = _⟨φ, TkUgψ⟩._


Expanding the last inner product, we get by change of variable and invariance of _µ_

�� ��
_k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _φ_ ( _**x**_ ) _ψ_ ( _g_ _[−]_ [1] _**x**_ _[′]_ ) _dµ_ ( _**x**_ ) _dµ_ ( _**x**_ _[′]_ ) = _k_ ( _**x**_ _, g_ _**x**_ _[′]_ ) _φ_ ( _**x**_ ) _ψ_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ ) _dµ_ ( _**x**_ _[′]_ ) _._


Hence for all _φ, ψ_, �� [ _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _−_ _k_ ( _**x**_ _, g_ _**x**_ _[′]_ )] _φ_ ( _**x**_ ) _ψ_ ( _**x**_ _[′]_ ) _dµ_ ( _**x**_ ) _dµ_ ( _**x**_ _[′]_ ) = 0 _,_ which implies
_k_ ( _**x**_ _, g_ _**x**_ _[′]_ ) = _k_ ( _**x**_ _,_ _**x**_ _[′]_ ) _µ ⊗_ _µ_ -a.e. Symmetry implies argumentwise _G_ -invariance.


We now show that _UgT_ = _T_ is preserved if we apply a function _f_ such that _f_ (0) = 0 to the spectrum
of _T_ .
**Lemma 14** (Borel functional calculus preserves invariance) **.** _Let T_ _be a self-adjoint compact operator_
_on a Hilbert space H with eigendecomposition T_ = [�] _i_ _[λ][i][ϕ][i][ ⊗]_ _[ϕ][i][, and let][ {][U][g][}][g][∈G]_ _[be a unitary]_


_on a Hilbert space H with eigendecomposition T_ = [�] _i_ _[λ][i][ϕ][i][ ⊗]_ _[ϕ][i][, and let][ {][U][g][}][g][∈G]_ _[be a unitary]_

_representation such that UgT_ = _TUg_ = _T_ _for all g_ _∈G._ _For a bounded Borel function f_ : R _→_ R _,_
_define f_ ( _T_ ) = [�] _i_ _[f]_ [(] _[λ][i]_ [)] _[ϕ][i][ ⊗]_ _[ϕ][i][.]_ _[Then for such][ f]_ _[with][ f]_ [(0) = 0] _[, we have]_

_Ugf_ ( _T_ ) = _f_ ( _T_ ) _Ug_ = _f_ ( _T_ ) _for all g_ _∈G._


_Proof._ **Proof sketch:** The assumption _UgT_ = _T_ forces _Ug_ to act as the identity on each nonzero
eigenspace of _T_, which directly yields _Ugf_ ( _T_ ) = _f_ ( _T_ ) for any bounded Borel _f_ with _f_ (0) = 0.


**Step 1 (spectral decomposition for compact self-adjoint** _T_ **without measures).** Since _T_ is compact
and self-adjoint, its spectrum is _σ_ ( _T_ ) = _{_ 0 _} ∪{λn_ : _n_ _∈_ _I}_ where _I_ is finite or countable, each
_λn_ = 0 is an eigenvalue of finite multiplicity, and _λn_ _→_ 0 if infinite. Let _Eλ_ denote the eigenspace
for _λ ̸_ = 0, and let _E_ 0 = ker _T_ . We have the orthogonal decomposition


_i_ _[f]_ [(] _[λ][i]_ [)] _[ϕ][i][ ⊗]_ _[ϕ][i][.]_ _[Then for such][ f]_ _[with][ f]_ [(0) = 0] _[, we have]_


_H_ = _E_ 0 _⊕_         - _Eλ,_


_λ∈σ_ ( _T_ ) _\{_ 0 _}_


and _T_ acts as scalar multiplication on each _Eλ_ : _T_ _|Eλ_ = _λ_ Id _Eλ_, _T_ _|E_ 0 = 0. Let _Pλ_ be the
orthogonal projector onto _Eλ_ (for _λ_ = 0) and _P_ 0 onto _E_ 0. Then for every _v_ _∈H_ with expansion
_v_ = _v_ 0 + [�] _λ_ =0 _[v][λ]_ [ (] _[v][λ]_ [:=] _[ P][λ][v]_ [), we have]

_Tv_ =              - _λ vλ._


_λ_ =0


**Step 2 (** _Ug_ **fixes each nonzero eigenspace pointwise).** From _UgT_ = _T_ we get, for any _v_ _∈_ _Eλ_ with
_λ ̸_ = 0,
_λ Ugv_ = _Ug_ ( _Tv_ ) = _Tv_ = _λ v,_
hence _Ugv_ = _v_ . Thus _Ug_ acts as the identity on each _Eλ_ ( _λ ̸_ = 0). Equivalently, _UgPλ_ = _PλUg_ = _Pλ_
for all _λ ̸_ = 0. (There is no restriction on _Ug_ inside _E_ 0 = ker _T_ .)


**Step** **3** **(defining** _f_ ( _T_ ) **for** **bounded** **Borel** _f_ **with** _f_ (0) = 0 **).** Because _σ_ ( _T_ ) _\ {_ 0 _}_ is at most
countable and _T_ is diagonal on _{Eλ}_, we can define _f_ ( _T_ ) by applying _f_ on the spectrum of _T_ as


_f_ ( _T_ ) _v_ := 


 - _f_ ( _λ_ ) _vλ,_ _v_ = _v_ 0 + 

_λ∈σ_ ( _T_ ) _\{_ 0 _}_ _λ_ =0


_vλ,_ _vλ_ _∈_ _Eλ._

_λ_ =0


The series converges in norm since the _Eλ_ are mutually orthogonal and _∥f_ ( _T_ ) _v∥_ [2] =

- _λ_ =0 _[|][f]_ [(] _[λ]_ [)] _[|]_ [2] _[∥][v][λ][∥]_ [2] _[≤]_ - sup _λ_ =0 _|f_ ( _λ_ ) _|_ [2][��] _λ_ =0 _[∥][v][λ][∥]_ [2] _[≤∥][f]_ _[∥]_ [2] _∞_ _[∥][v][∥]_ [2][.] [Thus] _[f]_ [(] _[T]_ [)] [is] [a] [bounded]

operator with _∥f_ ( _T_ ) _∥≤∥f_ _∥∞_ . (When _f_ (0) = 0, there is no contribution on _E_ 0.)


**Step 4 (invariance and commutation).** For _v_ = _v_ 0 + [�]


**Step 4 (invariance and commutation).** For _v_ = _v_ 0 + [�] _λ_ =0 _[v][λ]_ [as above and any] _[ g]_ _[∈G]_ [, Step 2]

gives _Ugv_ = _Ugv_ 0 + [�] _λ_ =0 _[v][λ]_ [ and] _[ P][λ][U][g]_ [=] _[ P][λ]_ [ for] _[ λ][ ̸]_ [= 0][.] [Hence]


_λ_ =0 _[v][λ]_ [ and] _[ P][λ][U][g]_ [=] _[ P][λ]_ [ for] _[ λ][ ̸]_ [= 0][.] [Hence]


��
_Ugf_ ( _T_ ) _v_ = _Ug_


_f_ ( _λ_ ) _vλ_   - =   
_λ_ =0 _λ_ =0


- _f_ ( _λ_ ) _Ugvλ_ = 

_λ_ =0 _λ_ =0


_f_ ( _λ_ ) _vλ_ = _f_ ( _T_ ) _v,_

_λ_ =0


22


i.e., _Ugf_ ( _T_ ) = _f_ ( _T_ ). In particular _Ugf_ ( _T_ ) = _f_ ( _T_ ) _Ug_ = _f_ ( _T_ ) for all _g_ _∈G_ .


**Consequence.** If _k_ is _G_ -invariant, then so is _k_ + (Equation (12)).


D EIGENDECAY COMPARISON


In this appendix, we discuss in more details the empirical observations made in Section 5 and formally
derive some inequalities between Schatten norms of integral operators associated with _k_ avg and _k_ +.


D.1 EMPIRICAL OBSERVATIONS


Here, we further discuss the empirical spectra reported in Figure 4 (middle and right columns).


**Computation of spectra.** The normalized Gram matrices _**K**_ _/n_ (where _**K**_ = ( _k_ ( _**x**_ _i,_ _**x**_ _j_ ))1 _≤i,j≤n_ )
reported in Figure 4 are computed from _n_ = 3000 i.i.d. samples _**x**_ _i_ _∈S_ . We compare the spectra
obtained with _k_ _∈{k_ b _, k_ avg _, k_ + [(] _[D]_ [)] _[}]_ [ with] _[ D]_ [=] _[ {]_ _**[x]**_ [1] _[, . . .,]_ _**[ x]**_ _[n][}]_ [ and each] _**[ x]**_ _[i]_ [being chosen uniformly in]
_S_ = [ _−_ 1 _,_ 1]. We also report the spectrum of _k_ b when observations _**x**_ _i_ are instead sampled from an
alternative domain _S_ _[′]_ of reduced volume, chosen such that vol( _S_ _[′]_ ) = vol( _S_ ) _/|G|_ . Finally, note that
because _D_ is a set of i.i.d. observations, the spectrum of _k_ + [(] _[D]_ [)] approximates the one of _k_ + on _S_ (see
Appendix C.4) so our observations transfer to _k_ +.


_k_ + [(] _[D]_ [)] **on** _S_ **vs.** _k_ b **on** _S_ _[′]_ **.** For the base kernels _k_ b and groups _G_ considered, the spectrum of _k_ + [(] _[D]_ [)] on
_S_ = [ _−_ 1 _,_ 1] exactly matches that of _k_ b on the reduced domain _S_ _[′]_ . This indicates that _k_ + [(] _[D]_ [)] faithfully
incorporates the extra similarities induced by _G_ -invariance: it retains the eigendecay of _k_ b, but as if it
were defined on the quotient space _S/G_ of effective volume vol( _S_ ) _/|G|_ . [10]


_k_ + [(] _[D]_ [)] **on** _S_ **vs.** _k_ avg **on** _S_ **.** From Figure 4 (middle and right columns), it is clear that the spectrum
of _k_ avg decays at least as fast as that of _k_ + [(] _[D]_ [)][.] [They coincide for the RBF kernel and] _[ k]_ [avg] [decays even]
faster for the Matern kernel.´ In principle, this suggests that _k_ avg should admit tighter information-gain
bounds and thus better regret guarantees. However, our empirical results contradict this prediction, as
_k_ + [(] _[D]_ [)] consistently outperforms _k_ avg. This discrepancy highlights the fact that eigendecay alone does
not fully explain BO performance, as pointed out in Sections 5 and 6.


D.2 SCHATTEN NORM INEQUALITIES


While the empirical spectra in Appendix D.1 already highlight a mismatch between eigendecay and
observed BO performance, one may ask whether formal inequalities between the operators induced
by _k_ avg and _k_ + can be established. We record here for completeness that it is possible to control the
Schatten class of _k_ + in terms of the one of _k_ avg.


Assume: ( _S, µ_ ) is a probability space on which a finite group _G_ acts measurably, and the base kernel
_k_ b is bounded, symmetric, PSD, and nonnegative. Define

_k_ avg( _**x**_ _,_ _**x**_ _[′]_ ) := 1    - _k_ b( _g_ _**x**_ _, g_ _[′]_ _**x**_ _[′]_ ) _,_ _k_ max( _**x**_ _,_ _**x**_ _[′]_ ) := max
_|G|_ [2] _g,g_ _[′]_ _∈G_ _[k]_ [b][(] _[g]_ _**[x]**_ _[, g][′]_ _**[x]**_ _[′]_ [)]

_g,g_ _[′]_ _∈G_


and _k_ + as the kernel corresponding to the positive part of _Tk_ max : _Tk_ + = ( _Tk_ max )+.


**Schatten norm interpolation.** Let _H_ = _L_ [2] ( _µ_ ) be the separable Hilbert space of squared integrable
functions on ( _S, µ_ ), _T_ : _H_ _→_ _H_ a compact operator, and write _si_ ( _T_ ) for the singular values of _T_, i.e.
_si_ ( _T_ ) = - _λi_ ( _T_ _[∗]_ _T_ ), arranged in nonincreasing order and counted with multiplicity. The Schatten- _p_
norm is defined as


��
_∥T_ _∥Sp_ := _si_ ( _T_ ) _[p]_ [�][1] _[/p]_ _,_ 1 _≤_ _p < ∞,_ _∥T_ _∥S∞_ := sup _si_ ( _T_ ) _._

_i_
_i_

10For a finite group _G_ of isometries, one indeed has vol( _S/G_ ) = vol( _S_ ) _/|G|_ (Petersen, 2006).


23


**Lemma** **15** (Monotonicity for pointwise kernels) **.** _If_ _two_ _kernels_ _k, k_ _[′]_ _are_ _bounded_ _and_ _satisfy_
0 _≤_ _k_ _≤_ _k_ _[′]_ _pointwise,_ _then_ _∥Tk∥Sp_ _≤∥Tk′∥Sp_ _for_ _p_ = 2 _, ∞._ _If_ _k_ _and_ _k_ _[′]_ _are_ _also_ _PSD,_ _then_
_∥Tk∥Sp_ _≤∥Tk′∥Sp_ _for p_ = 1 _too._


_Proof._ For _p_ = _∞_, the Schatten _p_ -norm is the operator norm _∥T_ _∥_ op = sup _∥f_ _∥H_ =1 _∥Tf_ _∥H_ . Pointwise 0 _≤_ _k_ _≤_ _k_ _[′]_ implies _∥Tkf_ _∥H_ _≤∥Tk′|f_ _|∥H_ _≤∥Tk′∥S∞_ _∥f_ _∥H_, so taking the supremum over
_∥f_ _∥H_ = 1 yields _∥Tk∥S∞_ _≤∥Tk′∥S∞_ . If _T_ = _Tk_ is the integral operator associated with a nonnegative kernel _k_, then _∥Tk∥S_ 2 = _∥k∥L_ 2( _µ⊗µ_ ). Hence pointwise 0 _≤_ _k_ _≤_ _k_ _[′]_ gives _∥Tk∥S_ 2 _≤∥Tk′∥S_ 2
for _p_ = 2 as well. Finally when _k_ is PSD, we have _∥Tk∥S_ 2 = - _x_ _[k]_ [(] _[x, x]_ [)] _[dµ]_ [(] _[x]_ [)][ (and similarly for] _[ k][′]_ [)]

and again a pointwise comparison yields the result.


From this we immediately obtain, for our specific kernels that for _p_ = 2 _, ∞_, and also _p_ = 1 if _k_ max
is PSD:


_k_ avg _≤_ _k_ max _≤|G|_ [2] _k_ avg _⇒_ _∥Tk_ avg _∥Sp_ _≤∥Tk_ max _∥Sp_ _≤|G|_ [2] _∥Tk_ avg _∥Sp_


**Lemma** **16** (Interpolation inequalities for Schatten norms) **.** _For_ _any_ _nonnegative_ _sequence_ _a_ =
( _ai_ ) _i≥_ 1 _one has_

_∥a∥ℓp_ _≤∥a∥ℓ_ [2][2] _[/p]_ _∥a∥ℓ_ [1] _[∞][−]_ [2] _[/p]_ ( _p ≥_ 2) _,_


_∥a∥_ _[p]_ _ℓ_ _[p]_ _[≤∥][a][∥]_ _ℓ_ [2][1] _[−][p]_ _∥a∥ℓ_ [2(][2] _[p][−]_ [1)] (1 _≤_ _p ≤_ 2) _._


_i_ _[a]_ _i_ _[p]_ [=][ �]


_Proof._ For _p ≥_ 2, [�] _i_ _[a]_ _i_ _[p]_ [=][ �] _i_ _[a]_ _i_ _[p]_ [2] _a_ [2] _i_ _[≤∥][a][∥]_ _ℓ_ _[p][∞]_ [2] - _i_ _[a]_ _i_ [2][, giving the stated inequality.] [For][ 1] _[ ≤]_ _[p][ ≤]_

2, write

     - _[p]_      - [2] _[−][p]_ [2(] _[p][−]_ [1)]


_Proof._ For _p ≥_ 2, [�]


_i_ _[a]_ _i_ _[p][−]_ [2] _a_ [2] _i_ _[≤∥][a][∥]_ _ℓ_ _[p][∞][−]_ [2] 


_ai_ [2] _[−][p]_ _ai_ [2(] _[p][−]_ [1)] _._
_i_


_a_ _[p]_ _i_ [=]  _i_ _i_


Let _r_ = 2 _−_ 1 _p_ [and] _[s]_ [=] _p−_ 1 1 [(with] [the] [usual] [convention] [1] _[/]_ [0] [=] _[∞]_ [).] [For] [1] _[<]_ _[p]_ _[<]_ [2] [we] [have]
1 _< r, s < ∞_ and by H¨older,

  - _[p]_ �� [2] _[−][p]_ _[r]_ [�][1] _[/r]_ [��] [2(] _[p][−]_ [1)] _[s]_ [�][1] _[/s]_ [�] 1 _/r_ [�] [2] 1 _/s_


( _a_ [2] _i_ _[−][p]_ ) _[r]_ [�][1] _[/r]_ [��]
_i_ _i_


��
_a_ _[p]_ _i_ _[≤]_
_i_ _i_


( _a_ [2(] _i_ _[p][−]_ [1)] ) _[s]_ [�][1] _[/s]_ =  - [�]
_i_ _i_


_a_ [2] _i_ �1 _/s._

_i_


_ai_ �1 _/r_  - [�]

_i_ _i_


Since 1 _/r_ = 2 _−_ _p_ and 1 _/s_ = _p −_ 1, this gives


_∥a∥_ _[p]_ _ℓ_ _[p]_ _[≤∥][a][∥]_ _ℓ_ [2][1] _[−][p]_ _∥a∥ℓ_ [2(][2] _[p][−]_ [1)] _._


The endpoint cases _p_ = 1 _,_ 2 follow by continuity (and are trivial directly).


Applied to _ai_ = _si_ ( _T_ ), Lemma 16 yields the standard Schatten interpolation inequalities:

_∥T_ _∥Sp_ _≤∥T_ _∥S_ [2] 2 _[/p]_ _[∥][T]_ _[∥]_ _S_ [1] _∞_ _[−]_ [2] _[/p]_ _,_ ( _p ≥_ 2) _,_


               -                - _p_ [2] _[−]_ [1][ �] �1 _−_ _p_ [1]
_∥T_ _∥Sp_ _≤_ _∥T_ _∥S_ 1 _∥T_ _∥_ [2] _S_ 2 _,_ (1 _≤_ _p ≤_ 2) _._


Since the spectrum of _Tk_ + is the positive part of the one of _Tk_ max, we have _∥Tk_ + _∥Sp_ _≤∥Tk_ max _∥Sp_ .
We deduce the next lemma.


**Lemma 17.** _For p ≥_ 2 _:_

_∥Tk_ + _∥Sp_ _≤∥Tk_ max _∥Sp_ _≤|G|∥Tk_ avg _∥S_ [2] 2 _[/p]_ _[∥][T][k]_ avg _[∥]_ _S_ [1] _∞_ _[−]_ [2] _[/p]_
_and if k_ max _is already PSD then for_ 1 _≤_ _p ≤_ 2 _:_

_∥Tk_ + _∥Sp_ = _∥Tk_ max _∥Sp_ _≤|G|_       - _∥Tk_ avg _∥S_ 1�2 _/p−_ 1 � _∥Tk_ avg _∥_ [2] _S_ 2�1 _−_ 1 _/p_


_and_

_∥Tk_ avg _∥Sp_ _≤_          - _∥Tk_ max _∥S_ 1�2 _/p−_ 1 � _∥Tk_ max _∥_ [2] _S_ 2�1 _−_ 1 _/p._


24


400


300


200


100


0


Iteration T


Iteration T


10 [3]


10 [2]


10 [1]


4000


3000


2000


1000


0


3000


2000


1000


0


Iteration T


500

400

300

200

100

0


Iteration T


Iteration T


Figure 5: Cumulative regret under GP-UCB with _k_ b (blue crosses), _k_ avg (orange diamonds), and
_k_ + [(] _[D]_ [)] (green circles) on synthetic benchmarks. Shaded areas: standard error over 10 seeds.


0.7


0.8


0.9


30

40

50

60

70


Iteration T


Iteration T


Figure 6: Negated simple reward under GP-UCB with _k_ b (blue crosses), _k_ avg (orange diamonds),
and _k_ + [(] _[D]_ [)] (green circles) on real-world experiments. Shaded areas: standard error over 10 seeds.


E BENCHMARKS


In this appendix, we present additional results and describe the experimental setup of Section 4 in
detail.


E.1 EXPERIMENTAL FIGURES


We provide the whole set of figures generated from our experiments on synthetic benchmarks
(Figure 5) and on real-world problems (Figure 6).


E.2 EXPERIMENTAL DETAILS


In our experiments, every BO algorithm is implemented with the same BO library, namely
BOTorch (Balandat et al., 2020). All of them are initialized with five observations sampled uniformly
in _S_ . After that, at each iteration _t_, every BO algorithm must:


- **Fit its kernel hyperparameters.** This is done by gradient ascent of the Gaussian likelihood, as
recommended by BOTorch. The hyperparameters are the signal variance _λ_, the lengthscale _l_ and
the observational noise level _σ_ 0 [2][.]

- **Optimize** **GP-UCB** **to** **find** _**x**_ _t_ **.** This is done by multi-start gradient ascent, using the
optimize ~~a~~ cqf function from BOTorch. As values of _βt_ recommended by Srinivas et al.
(2012) turn out to be too exploratory in practice, we set _βt_ = 0 _._ 5 _d_ log( _t_ ).

- **Observe** _y_ ( _**x**_ _t_ ) = _f_ ( _**x**_ _t_ ) + _ϵt_ **.** Function values are corrupted by noise whose variance is 2% of the
signal variance.


25


We optimize over 50 iterations and typically measure the cumulated regret along the optimizer’s
trajectory.


All experiments are replicated across ten independent seeds and are run on a laptop equipped with
an Intel Core i9-9980HK @ 2.40 GHz with 8 cores (16 threads). No graphics card was used to
speed up GP inference. The typical time for each maximization problem ranged from _∼_ 1 minute
(two-dimensional Ackley, _|G|_ = 8) to _∼_ 15 minutes (five-dimensional Rastrigin, _|G|_ = 3840). The
particle packing problem was by far the most time-consuming experiment due to the expensive
physics simulator used for computing the objective value of each new query ( _∼_ 4 hours for 30 BO
iterations, which we repeated on 10 seeds for each kernel).


E.3 BENCHMARKS


We maximize the following functions.


**Ackley.** The _d_ -dimensional Ackley function _f_ Ackley on _S_ = [ _−_ 16 _,_ 16] _[d]_ with global maximum
_f_ Ackley( **0** ) = 0, with _−f_ Ackley defined by:


_d_

- _x_ [2] _i_

_i_ =1


~~�~~


- [1]
_d_


1

_d_


cos( _cxi_ )

_i_ =1





 _−_ exp


_d_


_−f_ Ackley( _**x**_ ) = _−a_ exp





 _−b_


+ _a_ + exp(1) _,_ (15)


where we set _a_ = 20, _b_ = 0 _._ 2 and _c_ = 2 _π_ as recommended.


The _d_ -dimensional Ackley is invariant to the hyperoctahedral group in _d_ dimensions, which includes
permutations composed with coordinate-wise sign-flips. Consequently, in _d_ dimensions, _|G|_ =
2 _[d]_ _d_ ! .
���� ����
sign-flips permutations


**Griewank.** The _d_ -dimensional Griewank function _f_ Griewank on _S_ = [ _−_ 600 _,_ 600] _[d]_ with global
maximum _f_ Griewank( **0** ) = 0, with _−f_ Griewank defined by:


_d_


- cos - ~~_√_~~ _xi_

_i_

_i_ =1


_i_


_−f_ Griewank( _**x**_ ) =


_d_


_i_ =1


_x_ [2] _i_
4000 _[−]_


+ 1 _._


The _d_ -dimensional Griewank is invariant to coordinate-wise sign-flips of all _d_ coordinates. Therefore,
in _d_ dimensions, _|G|_ = 2 _[d]_ .


**Rastrigin.** The _d_ -dimensional Rastrigin _f_ Rastrigin on _S_ = [ _−_ 5 _._ 12 _,_ 5 _._ 12] _[d]_ with global maximum
_f_ Rastrigin( **0** ) = 0, with _−f_ Rastrigin defined by:


_−f_ Rastrigin( _**x**_ ) = 10 _d_ +


_d_


_i_ =1


- _x_ [2] _i_ _[−]_ [10 cos (2] _[πx][i]_ [)] - _._


The _d_ -dimensional Rastrigin is invariant to the hyperoctahedral group in _d_ dimensions, which
includes permutations composed with coordinate-wise sign-flips. Consequently, in _d_ dimensions,
_|G|_ = 2 _[d]_ _d_ ! .
���� ����
sign-flips permutations


**Radial.** Our radial benchmark is defined on _S_ = [ _−_ 10 _,_ 10] [2] with global maxima _f_ Radial( _**x**_ _[∗]_ ) = 0,
where _**x**_ _[∗]_ is any _**x**_ _∈S_ such that _||_ _**x**_ _||_ 2 = _ab_ . It has the following expression:


_f_ Radial( _**x**_ ) = _f_ Rastrigin


- _||_ _**x**_ _||_ 2 _−_ _b_ - (16)
_a_


_√_
where we set _a_ = 10


2, _b_ = 0 _._ 8 and where _f_ Rastrigin is the one-dimensional Rastrigin benchmark.


Our radial benchmark is invariant to planar rotations. Consequently, _G_ comprises an uncountably
infinite number of symmetries.


26


**Scaling.** Our scaling benchmark is defined on _S_ = [0 _._ 1 _,_ 10] [2] with global maxima _f_ Scaling( _**x**_ _[∗]_ ) = 0,
where _**x**_ _[∗]_ is any _**x**_ = ( _x_ 1 _, x_ 2) _∈S_ such that _x_ 1 = _x_ 2. The function _−f_ Scaling has the following
expression:

_−f_ Scaling( _**x**_ ) =            - _x_ 1 _−_ 1�2 _._
_x_ 2


Our scaling benchmark is invariant to rescaling of both coordinates. Consequently, _G_ comprises an
uncountably infinite number of symmetries.


**WLAN.** The goal of the WLAN benchmark is to place _m_ access points (APs) inside a square
region _A_ = [ _−_ 50 _,_ 50] [2] so as to maximize the total communication quality over _p_ users located in _A_,
a recurring problem in wireless network design (Younis & Akkaya, 2008; Taleb et al., 2022). Given
a set of AP positions, each user connects to its closest AP, and the resulting network throughput—
computed from the Signal to Interference plus Noise Ratio (SINR) and Shannon capacities—defines
the value of the objective function.


The user positions _{_ ( _uj, vj_ ) _}j∈_ [ _p_ ] _⊂A_ and all physical parameters ( _W_, _L_, _λ_, _N_ ) are given. The
region _A_ itself is fixed.


The variables of the problem are the AP locations

( _**x**_ _,_ _**y**_ ) = (( _x_ 1 _, . . ., xm_ ) _,_ ( _y_ 1 _, . . ., ym_ )) _∈S_ = _A_ _[m]_ _,_


so the search space is 2 _m_ -dimensional. Every quantity below—AP–user associations, distances,
received powers, SINRs, and capacities—depends on ( _**x**_ _,_ _**y**_ ).


For a candidate placement _{_ ( _xi, yi_ ) _}_, each user attaches to its nearest AP. Thus AP _i_ serves the users
in
_U_ ( _xi, yi_ ) = _{ j_ _∈_ [ _p_ ] : _dij_ _≤_ _dkj_ for all _k_ = _i },_

(ties are resolved arbitrarily) where the distance to user _j_ is


~~�~~
_dij_ = ( _xi −_ _uj_ ) [2] + ( _yi −_ _vj_ ) [2] _._


For any associated pair ( _i, j_ ), the power received by user _j_ from AP _i_ is

_Pij_ = 10 _[−][L/]_ [10] min( _d_ _[−]_ _ij_ _[λ][,]_ [ 1)] _[,]_


and the SINR is

_Pij_
_γij_ = _._
_N_ + [�] _k_ = _i_ _[P][kj]_

The corresponding Shannon capacity is


_Cij_ = _W_ log2(1 + _γij_ ) _._


Maximizing the WLAN performance amounts to maximizing the total throughput (the cumulated
sum of Shannon capacities for every AP-user association):


 - _Cij_


_j∈U_ ( _xi,yi_ )


_f_ WLAN( _**x**_ _,_ _**y**_ ) =


viewed as a function of the AP locations ( _**x**_ _,_ _**y**_ ).


_m_


_i_ =1


In our experiment, we set _W_ = 1 MHz, _L_ = 46 _._ 67 dBm, _λ_ = 3, _N_ = _−_ 85 dBm, _m_ = 4 APs and
_p_ = 16 users.


Our objective _f_ WLAN is invariant to any permutation of the APs: permuting both _**x**_ and _**y**_ with the
same permutation leaves the objective value unchanged. Therefore, _|G|_ = _m_ !.


Figure 7 shows the best AP-placement found by GP-UCB using _k_ + [(] _[D]_ [)] on one training run.


**Particle packing problem.** The particle packing fraction (PPF) problem models how a mixture of
spherical particles settles under gravity inside a fixed rectangular box. This setting originates from


27


Figure 7: WN with the best positions of APs found by GP-UCB with _k_ + [(] _[D]_ [)][.] [APs are depicted by red]
triangles and users with blue circles. The throughput for each user is shown in Mbps.


granular-material physics and is routinely used in materials science and civil engineering (e.g., in
the design of concrete mixes (Li et al., 2023; Basheerudeen & Anandan, 2014) by tuning the size
distribution and proportions of aggregates to maximize packing density; for instance to need less
cement and water, and get better mechanical properties).


People literally design concrete mixes by tuning the size distribution and proportions of aggregates to
maximize packing density (so you need less cement and water, and get better mechanical properties).


In this problem, a mixture of particles is first instantiated inside the box according to prescribed
mixture parameters, and the particles are then allowed to fall under gravity. Collisions, frictions and
rearrangements determine the final configuration, and the packing fraction is defined as the ratio
between the total particle volume and the volume of the smallest axis-aligned box that contains all
particles after settling.


We fix the number of particle types to _n_ . Each type _i_ is described by:


- a diameter _di_ in a prescribed interval [ _d_ min _, d_ max],

- a share _si_ in [ _s_ min _, s_ max], representing the relative proportion of particles of that type in the mixture.


Thus the optimization variable is


_**x**_ = ( _d_ 1 _, . . ., dn,_ _s_ 1 _, . . ., sn_ ) _._


The box size and the total initial particle volume _Vp_ (which then remains constant during the
simulation) are fixed in all experiments.


Given a mixture specification _**x**_ = ( _d_ 1 _, . . ., dn,_ _s_ 1 _, . . ., sn_ ), the initial particle configuration is
generated by repeatedly sampling particles until a fixed total particle volume _Vp_ is reached. Particles
are sampled independently as follows: (i) sample a type _i ∈{_ 1 _, . . ., n}_ with probability proportional
to its share _si_, (ii) sample a location uniformly at random in the container and put a particle of
diameter _di_ there. If any overlap of particles occurs during initialization, positions are adjusted locally
so that the configuration becomes valid. From this randomized initial state, the system evolves under
gravity, in practice we use a physics-based simulator (LAMMPS (Thompson et al., 2022)) for that.
The simulation proceeds until the particles reach a mechanically stable configuration, as illustrated in
Figure 8. If _Vo_ ( _**x**_ ) denotes the volume of the smallest axis-aligned box enclosing all particles at the
end of the dynamics (i.e., the container volume after settling), the particle packing fraction is


_Vp_
PPF( _**x**_ ) =
_Vo_ ( _**x**_ ) _[,]_


and we aim at maximizing this as a function of the mixture parameters _**x**_ . To our knowledge, there is
no accurate closed-form expression for this dynamical packing fraction in our setup, so evaluating
PPF( _**x**_ ) requires running the full physical simulation. Indeed: PFF( _**x**_ ) is actually a random variable:
given any mixture parameters _**x**_, _Vo_ ( _**x**_ ) depends on the random initialization of the particles in the
container, so there is observational noise induced by this random initialization. Moreover, even if the
random seed was fixed, because _Vo_ ( _**x**_ ) depends on complex interactions during the fall—collisions,
friction, and rearrangements, there is still no closed form available: evaluating PPF( _**x**_ ) always


28


requires running this full physical simulation. This makes the objective function costly and genuinely
black-box, a typical regime where BO is well motivated.


Figure 8: Particles settling under gravity in a fixed-size box. A single evaluation of PPF( _**x**_ ) requires
simulating the fall from a randomized initial configuration (left) to a mechanically stable state (right),
making the objective expensive and simulation-based.


Two symmetries are inherent to this formulation:


1. _Share scaling:_ multiplying all _si_ by the same positive factor leaves the resulting mixture unchanged
(the mixture only involves normalized shares).
2. _Permutation symmetry:_ permuting the ( _di, si_ ) pairs does not change the mixture either.


In practice, we take _n_ = 3, which is the smallest setting where the problem starts to be interesting (no
easy solution) while keeping simulation costs manageable. We constrain the diameters and shares to


_di_ _∈_ [0 _._ 35 _,_ 0 _._ 80] _,_ _si_ _∈_ [0 _._ 1 _,_ 1 _._ 0] _,_


chosen so that (i) all particles remain sufficiently small relative to the fixed box size, and (ii) each
type is represented in non-negligible quantity.


Baird et al. (2023a) previously applied BO to this problem (for solid rocket fuel design) and handled
these symmetries by restricting the search to a fundamental domain and applying standard kernels
there. In contrast, we keep the domain unchanged and instead use kernels that are _invariant_ under
the symmetries of the problem. A conceptual comparison between these two symmetry-handling
strategies is provided in Appendix G.


F COMPARISON OF SYMMETRY-INVARIANT KERNELS WITH THE
DATA-AUGMENTATION APPROACH


Given the widespread use of data augmentation (DA), we compare symmetry-invariant kernels
with the simple baseline corresponding to using the base kernel combined with DA. We find that
symmetry-invariant kernels perform better overall.

DA consists of replacing each input _x_ in the dataset by ( _gx_ ) _g∈Gx′_ with _Gx_ _[′]_ _[⊂G]_ [, and BO is run on this]
augmented dataset. We consider two scenarios: (i) using all augmentations for small groups ( _Gx_ _[′]_ [=] _[ G]_
for all _x_ ) so that ( _gx_ ) _g∈G′_ is simply the orbit of _x_, and (ii) using a random subset _Gx_ _[′]_ _[⊂G]_ [for larger]
groups (chosen independently for every _x_, drawn uniformly without replacement).


On the two-dimensional Ackley function (left panel of Figure 9), _kb_ is applied to a dataset augmented
with all symmetries ( _|G|_ = 8). In this case, _kb_ with DA achieves slightly better (lower) cumulative
regret than _kb_ alone. Its performance, however, remains worse that of the average kernel _k_ avg and the
PSD projection of the max kernel _k_ + [(] _[D]_ [)][.] [A similar pattern appears on the three-dimensional Ackley]
function (right panel of Figure 9), where DA uses 20 augmentations sampled without replacement
from _G_ ( _|G|_ = 48).


We also report the runtime of each method. These results show that _kb_ +DA scales less favorably than
_k_ avg and _k_ + [(] _[D]_ [)][, even when using only a moderate random subset of augmentations.] [Overall, these]
experiments suggest that using symmetry-invariant kernels directly is more practical for Bayesian
optimization than relying on data augmentation.


29


400


300


200


100


0


Iteration T


400


200


0


Iteration T


Figure 9: Cumulative regret on the two-dimensional (resp., three-dimensional) Ackley function, with
_|G|_ = 8 (resp., _|G|_ = 48).


Table 3: Average wall-clock time in seconds per iteration for each method on the two-dimensional
(resp., three-dimensional) Ackley function.


**Benchmark** _|G|_ _kb_ _kb_ with DA _k_ avg _k_ + [(] _[D]_ [)]
Ackley2d 8 0.416 _±_ 0.253 0.599 _±_ 0.279 0.451 _±_ 0.273 0.924 _±_ 0.444
Ackley3d 48 0.506 _±_ 0.336 2.665 _±_ 2.950 0.590 _±_ 0.384 1.307 _±_ 0.724


G WORKING WITH FUNDAMENTAL DOMAINS AND QUOTIENTS


This appendix expands on the brief discussion in Section 2.2 about search-space restriction and
explains why our approach targets kernel design rather than the choice of domain. The goal is to
clarify that both ingredients, a good domain and a good kernel, are needed and complementary.


G.1 FUNDAMENTAL DOMAINS AS QUOTIENT REPRESENTATIONS


Given a domain _S_ and a group action _G_, restricting the search to a fundamental domain amounts
to choosing a concrete embedded representation of the quotient space _S/G_ in _S_ . While this is
conceptually elegant, the practical implementation depends heavily on the pair ( _S, G_ ) and must be
re-derived for each new problem.


G.2 EXAMPLE: PERMUTATIONS OF R _[d]_


In several of our experiments, _S_ = [ _a, b_ ] _[d]_ and _G_ = _Sd_ acts by permuting coordinates. Two vectors
are equivalent if one is a permutation of the other. A natural choice of fundamental domain is the
_sorted cone_
_C_ = _{x ∈_ [ _a, b_ ] _[d]_ : _x_ 1 _≤_ _x_ 2 _≤· · · ≤_ _xd},_
which is one possible representation of the quotient _S/G_ (other equivalent views include multisets or
_d_ -atomic probability measures, but these views does not lead to subsets of the original domain _S_ so
they do not qualify as ”fundamental domains”).


Even in this simple case, two practical issues appear.


_(1) One must characterize and project onto the quotient, and check that it is “smooth enough”._ Most
BO implementations assume that the search domain is a box [ _a, b_ ] _[d]_ for which enforcing feasibility of
the iterates is straightforward (via coordinatewise clipping _x �→_ max( _a,_ min( _b, x_ ))). If we optimize
an acquisition function over the fundamental domain _C_ instead, any gradient-based or heuristic
optimizer will typically propose points _x_ that lie outside _C_, and these must be projected back. This
requires (i) describing the quotient _S/G_ via an explicit embedded representation (here, _C_ _⊂S_ ) and
(ii) figuring out how to implement the projection. For _C_, projecting _x_ onto it amounts to solving


proj _C_ ( _x_ ) _∈_ arg min
_y_ 1 _≤···≤yd_ _[∥][y][ −]_ _[x][∥]_ [2] _[,]_


which can be solved efficiently using known algorithms (e.g. the pool adjacent violators algorithm).
Our point is not that this particular projection is hard, but that for each new pair ( _S, G_ ) the user must


30


again derive an explicit model of the quotient and a practical projection operator, which can be a
burden depending on their goals and familiarity with quotients and the problem at hand.


_Smoothness assumptions also need to be checked._ The cone _C_ is not a smooth manifold, implying that
the projection is not smooth everywhere and gradients are not smooth (or even properly defined) at
certain points. Here, the singularities form a zero-measure set: they occur at points with some equal
coordinates (this is because the action of _G_ is not free; in contrast, if the action were free, proper,
and smooth, Theorem 21.10 in Lee (2013) would guarantee that the quotient is a smooth manifold).
For many constrained sets, singularities similarly form a negligible set and may be harmless for
optimization (initialization and gradient descent are likely to avoid them), but this depends on the
specific quotient and must be verified on a case-by-case basis.


Overall, working in the quotient means that the user must (i) characterize and project onto a potentially
non-smooth quotient, and (ii) check that its singularities do not cause difficulties for the optimization
method they use. Doing this for each new ( _S, G_ ) may be burdensome. This is why, in this paper, we
choose to avoid optimizing in a fundamental domain and instead provide kernels that can be used in a
plug-and-play manner directly on _S_ . These same kernels could also be used on the quotient space (by
interpreting them as kernels on equivalence classes), so our approach is complementary to, rather
than in competition with, the choice of the search domain.


_(2) One must still choose a kernel on equivalence classes._ Working on _S/G_ does not remove the
modelling choice: one still needs to pick a kernel _k_ ([ _x_ ] _,_ [ _y_ ]), and there is no canonical option even in
the permutation example. The quotient can be described in several equivalent ways (sorted vectors in
_C_, multisets, or atomic measures), and each viewpoint naturally suggests different classes of kernels
or distances. This is precisely the type of question our paper addresses: how to construct a good
kernel that is invariant to the symmetries? We study a natural construction: start with a “good” kernel
on _S_ (e.g. one that makes sense locally on _S_ to measure similarity before accounting for symmetries),
and then make it invariant by aggregating via mean or max. The resulting kernels are _G_ -invariant and
thus well-defined on the quotient, and our results show that the max-based construction shows good
properties, both empirically and geometrically.


H USE OF LLMS


We made limited use of large language models (GPT-5) during the preparation of this manuscript.
Their role was strictly restricted to grammar correction, improving clarity and conciseness, emphasizing text (e.g., bolding), and formatting tables. They were not used for generating technical content,
suggesting new concepts, or contributing to proofs or results. All ideas, proofs, experiments, and
findings are entirely our own. Every rephrased passage was carefully reviewed and validated by the
authors to ensure correctness and faithfulness to our original intent. No unverified or plagiarized
content was introduced.


31