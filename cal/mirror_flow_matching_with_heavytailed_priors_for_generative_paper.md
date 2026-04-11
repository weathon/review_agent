# MIRROR FLOW MATCHING WITH HEAVY-TAILED PRI- ORS FOR GENERATIVE MODELING ON CONVEX DO
## MAINS


**Shiqian Ma**
Department of CMOR
Rice University
Houston, TX 77005
sqma@rice.edu


**Yunrui Guan**
Department of CMOR
Rice University
Houston, TX 77005
yg83@rice.edu


**Krishnakumar Balasubramanian**
Department of Statistics
University of California, Davis
Davis, CA 95616
kbala@ucdavis.edu


ABSTRACT


We study generative modeling on convex domains using flow matching and mirror
maps, and identify two fundamental challenges. First, standard log-barrier mirror
maps induce heavy-tailed dual distributions, leading to ill-posed dynamics. Second,
coupling with Gaussian priors performs poorly when matching heavy-tailed targets.
To address these issues, we propose Mirror Flow Matching based on a _regularized_
_mirror map_ that controls dual tail behavior and guarantees finite moments, together
with coupling to a Student- _t_ prior that aligns with heavy-tailed targets and stabilizes
training. We provide theoretical guarantees, including spatial Lipschitzness and
temporal regularity of the velocity field, Wasserstein convergence rates for flow
matching with Student- _t_ priors and primal-space guarantees for constrained generation, under _ε_ -accurate learned velocity fields. Empirically, our method outperforms
baselines in synthetic convex-domain simulations and achieves competitive sample
quality on real-world constrained generative tasks.


1 INTRODUCTION


Flow matching (Lipman et al., 2023; Liu et al., 2023c; Albergo et al., 2023; Albergo & VandenEijnden, 2023; Tong et al., 2024; Chen & Lipman, 2024) has emerged as a powerful framework
for generative modeling, unifying score-based diffusion and optimal transport approaches under a
single perspective. The central idea in flow matching is to construct a continuous-time deterministic
flow that transports a simple prior distribution (e.g., Gaussian) to a complex target distribution, by
learning its velocity field. Formally, given random variables _X_ 0 _∼_ _π_ 0 and _X_ 1 _∼_ _π_ 1, both supported
on R _[d]_, we seek a time-dependent vector field _v_ : R _[d]_ _×_ [0 _,_ 1] _→_ R _[d]_ such that the solution of the ODE
_dXt_ = _v_ ( _Xt, t_ ) _dt,_ with _X_ 0 _∼_ _π_ 0, satisfies _X_ 1 _∼_ _π_ 1. A simple construction is based on straight-line
interpolation _Xt_ = (1 _−_ _t_ ) _X_ 0 + _tX_ 1, which yields the conditional velocity field _v_ _[∗]_ ( _x, t_ ) = E[ _X_ 1 _−_
_X_ 0 _|_ _Xt_ = _x_ ]. This vector field _v_ _[∗]_ minimizes the regression loss min _v_ E[ _∥v_ ( _Xt, t_ ) _−_ _dt_ _[d]_ _[X][t][∥]_ [2][]] _[,]_

making it the optimal velocity field for the interpolation path. Since computing _v_ _[∗]_ exactly is
intractable, modern flow-matching methods approximate _v_ with a neural network and simulate
the ODE numerically. This pathwise formulation leads to scalable training objectives, principled
continuous-time generative processes, and improved sample quality.


**Constrained** **Flow** **Matching.** In many applications, the target is supported on constrained domains like polytope, simplex, or positive semidefinite matrices, rather than the full Euclidean space.
Examples include molecular generation, where atoms and bonds must satisfy physical stability
constraints (Fishman et al., 2023b), preference alignment (Kim et al., 2024), policy optimization
and physical constraints for robotics (Zhang et al., 2025; Utkarsh et al., 2025) and watermarked
content generation (Liu et al., 2023a). Standard flow-based methods fail in this setting: projecting
unconstrained samples back onto the domain distorts the distribution.


**Related works.** Several strategies address the challenge in constrained flow matching, including
reflection-based methods (Lou & Ermon, 2023; Fishman et al., 2023a; Xie et al., 2024; Christopher
et al., 2024) that keep trajectories inside the domain using boundary normals; mirror-map diffusion


1


models (Liu et al., 2023a; Feng et al., 2025) that transform constrained problems into unconstrained
ones using mirror-maps; gauge-map approaches (Li et al., 2025) that enforce feasibility via reflections;
and distance-penalty methods (Huan et al., 2025; Khalafi et al., 2024) that penalize distance to the
constraint set, at notable computational cost. Despite this progress, no framework yet ensures
constraint satisfaction while providing convergence rates for flow matching.


In this work, we focus on the development of _mirror flow matching_, where the velocity field is adapted
to the geometry of the constraint set. Formally, let _K_ = _{ϕi_ ( _x_ ) _<_ 0 _,_ _ϕ_ : R _[d]_ _→_ R _,_ _i_ = 1 _, . . . m}_,
where _ϕi_ are smooth convex functions, be a closed convex set, and suppose the target distribution
_π_ 1 is supported on _K_ . Our approach is based on constructing a mirror map _∇_ Ψ : _K_ _→_ R _[d]_,
where Ψ : _K_ _→_ R is a strictly convex, differentiable potential. The mirror map transports points
from the constrained _primal_ space _K_ to an unconstrained _dual_ space. In this dual space, one can
perform standard (unconstrained) flow matching, i.e., define _Zt_ = _∇_ Ψ( _Xt_ ), and evolve it via
_dZt_ = _v_ _[D]_ ( _Zt, t_ ) _dt_ with _Z_ 0 = _∇_ Ψ( _X_ 0) _,_ where _v_ _[D]_ is a velocity field learned by minimizing the
unconstrained flow matching objective. The primal trajectory is then recovered by mapping back
using the inverse mirror map _Xt_ = ( _∇_ Ψ) _[−]_ [1] ( _Zt_ ). This mirror-descent-based formulation ensures that
the entire trajectory _{Xt}t∈_ [0 _,_ 1] remains in _K_ while leveraging the flexibility of unconstrained flow
matching in the dual space. Thus, mirror flow matching combines geometry-aware sampling with
scalable learning, broadening the applicability of flow models to structured domains that naturally
arise in the aforementioned application areas.


1.1 CHALLENGES AND SOLUTIONS


**Methodological** **Challenges.** Extending flow matching to constrained domains via mirror maps
introduces key challenges. First, the transformed target distribution in the dual space may have
heavy tails, causing standard mirror maps (e.g., log-barrier) to violate moment conditions required
for well-posed flow ODEs (Figure 1, red dots). We address this with a _regularized mirror map_ that
controls heavy tails and ensures finite _p_ -th moments for all _p ≥_ 1 (Figure 1, blue dots), stabilizing
training. Second, Gaussian priors often mismatch the heavy-tailed dual distributions; we instead adopt
a _Student-t prior_, improving alignment, sample quality, and stability. Together, these modifications
overcome limitations of standard log-barrier and Gaussian priors, yielding high-fidelity constrained
generative modeling. A visual illustration is provided in Appendix Section A.


**Theoretical** **Challenges.** In addition to the methodological issues above, theoretical analysis of
mirror flow matching poses challenges. Rigorous error bounds for the sampling stage require the
velocity field _v_ ( _x, t_ ) to be Lipschitz in _x_ (Benton et al., 2024; Bansal et al., 2024; Zhou & Liu, 2025;
Gao et al., 2024), while ODE discretization error further requires Lipschitz continuity in both _x_ and _t_
(Bansal et al., 2024; Zhou & Liu, 2025). However, the dual velocity field _v_ _[D]_ ( _z, t_ ) is generally not
Lipschitz over _t ∈_ [0 _,_ 1]. Partial progress includes spatial Lipschitzness on _t ∈_ [0 _, T_ ] ⊊ [0 _,_ 1] under
bounded _π_ 1 (Benton et al., 2024; Zhou & Liu, 2025) or Gaussian-like _π_ 1 (Gao et al., 2024). In general,
unbounded _π_ 1 can induce polynomial growth in _∥∇xv_ ( _x, t_ ) _∥_ as _∥x∥_ grows and singularities near
_t_ = 1, motivating _early stopping_ . Recent work (Cordero-Encinar et al., 2025) leverages Log-Sobolev
inequalities to establish spatial Lipschitzness, though _t_ -Lipschitzness is not addressed. We overcome
this challenge by using t-distribution as priors. While such priors have been explored empirically
(for example, (Pandey et al., 2025, Appendix B)), our motivation comes from addressing the above
theoretical challenge.


**Contributions.** In this work, we introduce flow matching with a Student- _t_ prior (see Section 3) and
provide new theoretical guarantees establishing both spatial Lipschitzness and temporal regularity
(see Proposition 4.1). This result enables us to obtain explicit error bounds under substantially more
general target distributions (see Theorem 3) in the dual Euclidean space under the assumption that
the learned velocity fields approximates the true dynamics up to _ε_ -accuracy. Finally in Theorem 4 we
further prove _primal-space guarantees_ for constrained dynamics.


2 INGREDIENTS FOR DESIGNING MIRROR FLOW MATCHING


2.1 INGREDIENT 1: THE MIRROR MAP


Before introducing our proposed mirror map, we first explain why the classical log-barrier is not
suitable in our setting. The main issue arises from our first identified challenge: ensuring the existence


2


of moments. As the following general result shows, if the log-barrier transformation induces heavy
tails, then even low-order moments (e.g., the second moment) may fail to exist.

**Lemma 2.1.** _Let Y_ _be a random variable in_ R _[d]_ _with law P_ _._ _Then, (i) if P_ ( _∥Y ∥≥_ _R_ ) _≥_ _C/R_ _[p]_ _for_
_some constant C_ _>_ 0 _, then_ E[ _∥Y ∥_ _[p]_ ] _does not exist, and (ii) if P_ ( _∥Y ∥≥_ _R_ ) _≤_ _C/R_ _[β]_ _with β_ _>_ _p,_
_then_ E[ _∥Y ∥_ _[p]_ ] _is finite._


In addition to controlling tails, we would also like the geometry
induced by the mirror map to have a desirable metric property: the
metric in the dual space should be _stronger_ than that in the primal
space. Formally, we require


_∥x −_ _y∥≤_ _L_ Ψ _∥∇_ Ψ( _x_ ) _−∇_ Ψ( _y_ ) _∥,_ _∀x, y_ _∈K,_ (1)


for some constant _L_ Ψ _>_ 0. To see why this is important, we first
recall some definitions of _p_ -Wasserstein distance in primal space
and dual space. Let _ν, µ_ be two probability measures on _K_ . Then
we have:


_Wp_ ( _ν, µ_ ) _[p]_ = _γ∈_ Γ(inf _ν,µ_ ) [E] _[γ]_ [[] _[∥][x][ −]_ _[y][∥][p]_ []] _[,]_

_Wp,_ Ψ( _ν, µ_ ) _[p]_ = _γ∈_ Γ(inf _ν,µ_ ) [E] _[γ]_ [[] _[∥∇]_ [Ψ(] _[x]_ [)] _[ −∇]_ [Ψ(] _[y]_ [)] _[∥][p]_ []] _[,]_


where _γ_ _∈_ Γ( _ν, µ_ ) means _γ_ is a coupling of _ν, µ_ . The first one
is just the Wasserstein distance for _K_ under Euclidean distance,
and the second one is actually the Wasserstein distance in the dual
space. To see this, let _ν_ _[′]_ _, µ_ _[′]_ denote the distribution of _ν, µ_ in dual
space, i.e., _ν_ _[′]_ = ( _∇_ Ψ)# _ν_ and _µ_ _[′]_ = ( _∇_ Ψ)# _µ_ . Then we have
_W_ 2 _,_ Ψ( _µ, ν_ ) = _W_ 2( _µ_ _[′]_ _, ν_ _[′]_ ). We remark that _W_ 2 _,_ Ψ was used to analyze the convergence of mirror Langevin algorithm (e.g., see Li et al.
(2022)).


Figure 1: Dual space distribution comparison between the
log barrier and our mirror map
( _κ_ = 0 _._ 5). The primal distribution is a truncated Gaussian
mixture within a polytope (see
Appendix A). The log barrier
yields a heavy-tailed distribution, while our mirror map produces a much lighter tail.


In general, an upper bound for _W_ 2 _,_ Ψ( _ν, µ_ ) doesn’t directly imply an
error bound for _W_ 2( _ν, µ_ ) in the primal space. But under inequality
(1), Wasserstein distances in the primal space can be controlled by those in the dual space:


_W_ 2( _ν, µ_ ) [2] = inf inf Ψ _[∥∇]_ [Ψ(] _[x]_ [)] _[ −∇]_ [Ψ(] _[y]_ [)] _[∥]_ [2][] =] _[ L]_ Ψ [2] _[W]_ [2] _[,]_ [Ψ][(] _[ν, µ]_ [)][2] _[.]_
_γ∈_ Γ( _ν,µ_ ) [E] _[γ]_ [[] _[∥][x][ −]_ _[y][∥]_ [2][]] _[ ≤]_ _γ∈_ Γ( _ν,µ_ ) [E] _[γ]_ [[] _[L]_ [2]


Inequality (1) is equivalent to _∇_ Ψ _[∗]_ being _L_ Ψ-Lipschitz. Since _∇_ [2] Ψ and _∇_ [2] Ψ _[∗]_ are inverses of each
other, this condition is in turn equivalent to Ψ being strongly convex. However, classical mirror maps
are generally only _strictly_ convex, not strongly convex. As a result, _L_ Ψ can be arbitrarily large in
certain domains; for instance, even for simple 2D polytopes with three facets ( _d_ = 2 _, m_ = 3), the
constant _L_ Ψ may blow up (see Example 5 in the Appendix).


These observations suggest that we need to design a new mirror map that balances tail behavior and
convexity. In particular, the desired mirror map should satisfy the following goals:


1. Transform the constrained distribution into an unconstrained distribution on R _[d]_ .
2. Ensure that key moments (e.g., the second moment) of the transformed distribution exist.
3. Be strongly convex, so that convergence guarantees in the dual Euclidean metric can be
transferred to guarantees in the primal Euclidean metric.


Motivated by the mirror-map framework of Vural et al. (2022), we propose in Proposition 2.2 a
_modified log-barrier_ that achieves these properties.


**Proposition** **2.2.** _Let_ _K_ = _{ϕi_ ( _x_ ) _<_ 0 _, ∀i_ _∈_ [ _m_ ] _},_ _where_ _ϕi_ _are_ _smooth_ _convex_ _functions_ _with_
_bounded_ _gradient._ _Let_ Ψ( _x_ ) = _−_ 1 _−_ 1 _κ_ - _mi_ =1 [(] _[−][ϕ][i]_ [(] _[x]_ [))][1] _[−][κ]_ [+] [1] 2 _[∥][x][∥]_ [2] _[.]_ _[Then]_ _[we]_ _[have]_ _[W]_ [2][(] _[ν, µ]_ [)] _[≤]_

_W_ 2 _,_ Ψ( _ν, µ_ ) _._ _Denote_ _Kδ_ = _{x_ _∈K_ : _−ϕi_ ( _x_ ) _≥_ _δ}._ _Let_ _X_ _be_ _a_ _random_ _variable_ _on_ _K_ _whose_
_law_ _is_ _denoted_ _as_ _P_ _._ _Assume_ _there_ _exists_ _positive_ _constants_ _CK, β, δ_ 0 _s.t._ _for_ _all_ 0 _<_ _δ_ _<_ _δ_ 0 _it_
_holds that P_ ( _K\Kδ_ ) _≤_ _CKδ_ _[β]_ _._ _Then there exists some constant C_ _s.t._ _in the dual space_ R _[d]_ _, for all_
_R_ _≥_ _C_ _[′]_ _/δ_ 0 _[κ]_ _[(here]_ _[C]_ _[′]_ _[is]_ _[some]_ _[constant]_ _[that]_ _[depends]_ _[on]_ _[K][),]_ _[P]_ [(] _[∥∇]_ [Ψ(] _[X]_ [)] _[∥≥]_ _[R]_ [)] _[≤]_ _[C/R][β/κ][.]_ _[By]_
_choosing κ < β/p, we can guarantee_ E[ _∥∇_ Ψ( _X_ ) _∥_ _[p]_ ] _exists._


3


Specific examples (including _L_ 2 ball and polytopes) are discussed in Appendix Section B. We verify
that the boundary-measure condition _P_ ( _K \ Kδ_ ) _≤_ _CKδ_ _[β]_ is natural in typical cases.

**Example 1** (Uniform distribution on the cube) **.** Let _K_ = [ _−_ 1 _,_ 1] _[d]_ and let _P_ be the uniform distribution
on _K_ . Define the _δ_ -interior as _Kδ_ = _{x_ _∈K_ : _d_ ( _x, ∂K_ ) _≥_ _δ}_ . Then the boundary layer has
probability mass _P_ ( _K\Kδ_ ) = [2] _[d][−]_ [(2] 2 _[d][−]_ [2] _[δ]_ [)] _[d]_ = 1 _−_ (1 _−δ_ ) _[d]_ _._ Using the first-order expansion (1 _−δ_ ) _[d]_ _≈_

1 _−_ _dδ_, we obtain _P_ ( _K \ Kδ_ ) _≈_ _dδ_ . Hence the condition _P_ ( _K \ Kδ_ ) _≤_ _CKδ_ _[β]_ holds with _β_ = 1 and
_CK_ = _d_ . This shows the assumption is mild and satisfied by standard convex bodies such as the cube
under uniform measure.


2.2 INGREDIENT 2: THE PRIOR DISTRIBUTION


For flow matching, let the target distribution be denoted by _X_ 1 _∼_ _π_ 1 with density _p_, and let
the initial distribution (prior) be _X_ 0 _∼_ _π_ 0. The evolution between _π_ 0 and _π_ 1 is described by a
time-dependent vector field, where _v_ ( _x, t_ ) denotes the true vector field. Considering straight-line
interpolation, by definition, the velocity field at a point ( _x, t_ ) is the conditional expectation of the
instantaneous displacement along this interpolation: _v_ ( _x, t_ ) = E[ _X_ 1 _−_ _X_ 0 _|_ _Xt_ = _x_ ] _._ To make
this expression explicit (Karras et al., 2022; Wan et al., 2025), note that the interpolation relation
_Xt_ = (1 _−_ _t_ ) _X_ 0 + _tX_ 1 can be inverted to obtain _X_ 0 = 1 _−_ 1 _t_ - _Xt −_ _tX_ 1� _._ Substituting this into the
displacement _X_ 1 _−_ _X_ 0 yields _X_ 1 _−_ _X_ 0 = _−_ 1 _−_ 1 _t_ _[X][t]_ [ +] 1 _−_ 1 _t_ _[X]_ [1] _[.]_ [ Taking conditional expectation given]
_Xt_ = _x_, we obtain the closed-form expression for the true velocity field:


     - 1 1
_v_ ( _x, t_ ) = E _−_
1 _−_ _t_ _[X][t]_ [ +] 1 _−_ _t_ _[X]_ [1]


    - 1 1
���� _Xt_ = _x_ = _−_ 1 _−_ _t_ _[x]_ [ +] 1 _−_ _t_ [E][[] _[X]_ [1] _[|][ X][t]_ [=] _[ x]_ []] _[.]_


Thus, the vector field _v_ ( _x, t_ ) consists of two interpretable terms: a deterministic contraction term

_−_ 1 _−_ 1 _t_ _[x]_ [ that pulls] _[ x]_ [ toward the origin, and a prediction term] 1 _−_ 1 _t_ [E][[] _[X]_ [1] _[|][ X][t]_ [=] _[ x]_ []][ that directs the flow]
toward the target distribution _π_ 1.


A crucial modeling choice in flow matching is the prior distribution. The choice of the prior
distribution affects this conditonal expectation E[ _X_ 1 _| Xt_ = _x_ ] significantly. While Gaussian priors
are the standard choice in unconstrained generative modeling, they are poorly suited when the target
distribution exhibits heavy tails. The following example illustrates this pathology. Denote standard
Student t distribution as _td,ν_ ( _x_ ) = _Cν,d_ (1 + [1] _[∥][x][∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_ .


[1] 2

_ν_ _[∥][x][∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 .


[1] 2

2 _[x]_ [2][)] _[−]_ [3]


**Example 2.** Consider the one-dimensional target density _X_ 1 _∼_ _p_ ( _x_ ) _∝_ (1 + [1]


**Example 2.** Consider the one-dimensional target density _X_ 1 _∼_ _p_ ( _x_ ) _∝_ (1 + [1] 2 _[x]_ [2][)] _[−]_ 2 . Suppose we

use a Gaussian prior _X_ 0 _∼N_ (0 _,_ 1). Then the conditional distribution of _X_ 1 given an interpolated
point _Xt_ = _x_, is given by


         _p_ ( _X_ 1 _|Xt_ = _x_ ) _∝_ _g_ ( _x_ 1) := exp _−_ [(] _[tx]_ [1] _[ −]_ _[x]_ [)][2]

2(1 _−_ _t_ ) [2]


�� 1
1 + [1] 2 _[x]_ 1 [2]


- 2 [3]
_._


This conditional distribution develops two modes: one near _x_ 1 = 0 and another near _x_ 1 _≈_ _x/t_ .
Although the _t →_ 0 limit will not cause a singularity (Wan et al., 2025), we emphasize that for large
values of _∥x∥_, the vector field would scales as exp( _x_ [2] ) for some small values of _t_, implying that the
true velocity field _v_ ( _x, t_ ) can blow up super-exponentially in _x_ . Furthermore, as discussed in Wan
et al. (2025); Zhou & Liu (2025), singularities exist as _t →_ 1. By contrast, if we replace the Gaussian
prior with a heavy-tailed Student- _t_ prior (e.g., with _ν_ = 1), the conditional density becomes


_p_ ( _X_ 1 _|Xt_ = _x_ ) _∝_ _g_ ( _x_ 1) = �1 + ��� _x_ 1 _−−txt_ 1 ���2 [�] _−_ 1 [�] 1 +1 [1] 2 _[x]_ 1 [2]


- 2 [3]
_,_


for which the dominant mode remains near _x_ 1 = 0 even as _x_ being large, over _t ∈_ [0 _, T_ ] ⊊ [0 _,_ 1]. In
this case, the conditional expectation does not explode with _x_, and the resulting velocity field remains
controlled. See Appendix Section C for a visualization.


This example highlights a key principle: when the target distribution is heavier-tailed than the prior,
the conditional distribution is likely to have a mode that is dominant near _[x]_ _t_ [for some values of] _[ t]_ [.] [Then]

the induced velocity field can diverge at large _∥x∥_, producing ill-posed dynamics and complicating
error analysis. In particular, such blow-ups directly cause the Lipschitz constant of _v_ ( _x, t_ ) to diverge


4


(a) Primal space trajectory (b) Dual space trajectory


Figure 2: Visualization of interpolations in primal and dual spaces – Straight line interpolation in the
dual space (Figure (b)) corresponds to curved “geodesic” interpolation in primal space Figure (a)).


as _∥x∥→∞_, necessitating additional assumptions on the tail of data distribution (e.g., bounded
support) (Benton et al., 2024; Bansal et al., 2024; Gao et al., 2024; Zhou & Liu, 2025). Choosing a
Student- _t_ prior prevents these blow-ups by making the data distribution to dominate the tail behavior
of the conditional distribution, suppressing the mode near _x/t_ . In this way, the mode near zero
will be dominant, ensuring controlled velocity fields, finite-moment guarantees of the interpolation
conditional distribution, and stability in both theoretical analysis and practical training.


3 MIRROR FLOW MATCHING


Recall from Section 2 that we discussed choices of mirror maps for closed convex sets of the form
_K_ = _{x ∈_ R _[d]_ : _ϕi_ ( _x_ ) _<_ 0 _,_ _∀i ∈_ [ _m_ ] _}_ . In mirror flow matching, both the prior _π_ 0 and the target _π_ 1
are required to be supported on _K_ . The objective is to learn a continuous-time flow _Xt_ defined by the
ODE _dt_ _[d]_ _[X][t]_ [=] _[ v][P]_ [ (] _[X][t][, t]_ [)][ with] _[ X]_ [0] _[∼]_ _[π]_ [0][(] _[x]_ [)][ that transports] _[ π]_ [0][ to] _[ π]_ [1][ over the interval] _[ t][ ∈]_ [[0] _[,]_ [ 1]][.]


Mirror flow matching achieves this transport by interpolating in a transformed (mirror) space. Given
a mirror map _∇_ Ψ, we map _x ∈K_ into the dual space via _z_ = _∇_ Ψ( _x_ ). As shown in Li et al. (2022),
the dual Euclidean space (R _[d]_ _, Id_ ) is isometric to the primal space equipped with the squared Hessian
metric ( _K,_ ( _∇_ [2] Ψ) [2] ). We denote these metrics as _g_ _[P]_ = ( _∇_ [2] Ψ) [2] and _g_ _[D]_ = _Id._ The procedure is then
as follows: (1.) Map primal data to dual space: _z_ = _∇_ Ψ( _x_ ). (2.) Perform flow matching in the dual
space using straight-line interpolation _Zt_ = (1 _−_ _t_ ) _Z_ 0 + _tZ_ 1. (3.) After generating samples _z_ ˆ in the
dual space, map them back to primal space using the inverse mirror map _x_ ˆ = _∇_ Ψ _[∗]_ (ˆ _z_ ). In particular,
interpolation in primal space is defined as _Xt_ = _∇_ Ψ _[∗]_ ( _Zt_ ) _,_ which can be interpreted as the _geodesic_
_interpolation_ between _X_ 0 and _X_ 1 under the squared Hessian metric. See Figure 2 for an illustrative
trajectory visualization in both the primal and dual spaces.


**Relation between dual and primal velocity fields.** Consider a dual-space flow _Zt_ defined by vector
field _v_ _[D]_ . By direct differentiation, the corresponding primal velocity field is


                 - _d_                  
_v_ _[P]_ ( _Xt, t_ ) := _[d]_ [=] _[ ∇]_ [2][Ψ] _[∗]_ [(] _[Z][t]_ [)] = _∇_ [2] Ψ _[∗]_ ( _Zt_ ) _v_ _[D]_ ( _Zt, t_ ) _._ (2)

_dt_ _[X][t]_ _dt_ _[Z][t]_


The flow matching objective in the dual space is

min _v_ E _t,Z_ 0 _,Z_ 1� _∥v_ _[D]_ ( _Zt, t_ ) _−_ _dt_ _[d]_ _[Z][t][∥]_ _g_ [2] _[D]_      - _,_ _Zt_ = (1 _−_ _t_ ) _Z_ 0 + _tZ_ 1 _,_ (3)


whose solution is known to be the conditional expectation _v_ _[D]_ ( _z, t_ ) = E[ _dt_ _[d]_ _[Z][t]_ _[|][ Z][t]_ [=] _[ z]_ []][ (Liu et al.,]

2023b). The following proposition establishes the equivalence between primal and dual formulations.
**Proposition** **3.1.** _Learning_ _a_ _vector_ _field_ _in_ _the_ _dual_ _Euclidean_ _space_ (R _[d]_ _, Id_ ) _is_ _equivalent_ _to_
_learning a vector field in the primal space_ ( _K,_ ( _∇_ [2] Ψ) [2] ) _._ _Specifically,_


   min E _∥v_ _[P]_ ( _Xt, t_ ) _−_ _[d]_
_v_


    -     _dt_ _[d]_ _[X][t][∥]_ _g_ [2] _[P]_ _and_ min _v_ E _∥v_ _[D]_ ( _Zt, t_ ) _−_ _[d]_


_dt_ _[d]_ _[Z][t][∥]_ _g_ [2] _[D]_ 


_are equivalent,_ _with_ _the correspondence_ _v_ _[D]_ ( _z, t_ ) = _∇_ [2] Ψ( _x_ ) _v_ _[P]_ ( _x, t_ ) _._ _Moreover,_ _the_ _primal flow_
_matching objective is solved by v_ _[P]_ ( _x, t_ ) = E� _dtd_ _[X][t]_ �� _Xt_ = _x_ - _._


5


**Algorithm 1** Mirror Flow matching with Student t distribution


1: Map data distribution from _K_ to R _[d]_ using _∇_ Ψ, obtain samples for _Z_ 1.
2: Learn a vector field _v_ ˆ _[D]_ ( _z, t_ ) with prior _π_ 0( _x_ ) _∼_ _td,ν_ via
min _v_ ˆ _D_ E _t,Z_ 0 _∼π_ 0 _D_ _[,Z]_ [1] _[∼][π]_ 1 _[D]_  - _∥v_ ˆ _[D]_ ( _Zt, t_ ) _−_ ( _Z_ 1 _−_ _Z_ 0) _∥_ [2][�] where _Zt_ = _tZ_ 1 + (1 _−_ _t_ ) _Z_ 0.


3: Choose step size _h_ for Euler discretization s.t. [1]


Choose step size _h_ for Euler discretization s.t. _h_ [is integer.] [Choose] _[ T]_ _[∈]_ [(0] _[,]_ [ 1)][ as early stopping]

time, satisfying _[T]_ _[∈]_ [Z][.]


time, satisfying _h_ _[∈]_ [Z][.]

4: Perform Euler discretization to sample from _π_ 1 _[D]_ [with constant step size] _[ h]_ [, up to time] _[ T]_ [:]
5: Generate _z_ 0 _∼_ _π_ 0 _[D]_ [.]
6: **for** _k_ = 0 to _[T]_ _[−]_ [1] **[ do]**


6: **for** _k_ = 0 to _h_ _[−]_ [1] **[ do]**

7: _zh_ ( _k_ +1) = _zhk_ + _hv_ ˆ( _zk, hk_ )
8: **end for**
9: Denote the obtained sample by _zT_ _∼_ _π_ ˆ _T_ _[D]_ [.]
10: Map samples _zT_ back to _K_ using _∇_ Ψ _[∗]_ to obtain _xT_ .


This result shows that training in the dual space with straight-line interpolation is equivalent to
training in the primal space with geodesic interpolation under the squared Hessian metric. From an
algorithmic standpoint, this equivalence is highly convenient: we can train the dual-space vector field
_v_ _[D]_, which is simpler due to its Euclidean geometry, and recover the primal vector field _v_ _[P]_ by the
transformation in equation 2. Thus, the difficult geometry of _K_ is automatically handled by the mirror
map, while optimization is carried out in an unconstrained Euclidean space.


The algorithmic procedure for mirror flow matching is summarized in Algorithm 1. This pipeline
leverages the simplicity of Euclidean training in dual space, while ensuring that the generated samples
respect the original convex constraints in primal space. Here, _h_ denotes the step size (in sampling
stage) and _T_ _<_ 1 denotes the terminal time if early stopping is adopted.


4 THEORETICAL RESULTS


In this section, we provide a theoretical analysis of error bounds for flow matching. A key component
of our analysis is the accuracy of the neural network used to approximate the target velocity field.
We adopt the following assumption, which is standard in the literature on flow-based generative
modeling (see, e.g., Benton et al. (2024); Bansal et al. (2024); Li et al. (2025)) as well as in the study
of diffusion models (see, e.g., Chen et al. (2023); Li et al. (2024)). Theoretical justification for this
assumption can be found in Wang et al. (2024); Zhou & Liu (2025), where the authors establish
that such an _ε_ -level approximation error can be achieved by a neural network under suitable training
conditions.


**Assumption 1.** _(Neural Network Estimation Error) Let v_ ( _x, t_ ) _denote the true velocity field and_
_v_ ˆ( _x, t_ ) _its neural network approximation._ _We assume that the approximation error is bounded in_
_mean square, i.e.,_ E� _∥v_ ( _x, t_ ) _−_ _v_ ˆ( _x, t_ ) _∥_ [2][�] _≤_ _ε_ [2] _._


Intuitively, Assumption 1 states that the learned velocity field ˆ _v_ is close to the true velocity field _v_
in an average sense across both space and time. The parameter _ε_ therefore quantifies the quality of
the neural network approximation: smaller _ε_ implies a more accurate approximation, which directly
translates into higher fidelity of the generated samples.


4.1 GUARANTEES FOR EUCLIDEAN FLOW MATCHING WITH T-DISTRIBUTION PRIORS


In this subsection, we provide an error analysis for flow matching in Euclidean space when the
prior distribution is chosen to be a Student- _t_ distribution (henceforth referred to as _t-Flow_ ). Our
analysis applies to the general framework of flow matching with straight-line interpolation, and is not
restricted to the mirror flow matching setup. To maintain notation consistency, we denote random
variables as _Z_ _∈_ R _[d]_ with density _π_ 1 _[D]_ [.] [We begin by introducing the assumptions required.]

**Assumption 2** (Finite Moments) **.** _Let Z_ 0 _denote the prior (chosen as Student-t) random variable_
_and Z_ 1 _denote the target random variable, both supported on_ R _[d]_ _._ _We assume that they have finite_
_second moments, i.e.,_ E[ _∥Z_ 0 _∥_ [2] ] _< ∞,_ E[ _∥Z_ 1 _∥_ [2] ] _< ∞, which is necessary for well-definedness._


6


**Assumption 3** (Polynomial Tail Bound) **.** _Let π_ 1 _[D]_ [(] _[x]_ [)] _[ denote the probability density function of the]_
_data distribution supported on_ R _[d]_ _._ _It is assumed to satisfy:_ _(1) For ∥x∥≥_ 1 _, we have π_ 1 _[D]_ [(] _[x]_ [)] _[ ≤]_ _∥xC∥_ _[α]_ _[,]_
_and (2) For ∥x∥_ _<_ 1 _, we have π_ 1 _[D]_ [(] _[x]_ [)] _[ ≤]_ _[C][u][.]_


The above assumption allows the target distribution to be heavy-tailed, covering a wide range of
realistic distributions. We next establish Lipschitz guarantees for the true vector field, showing that
under Assumption 3, the velocity field induced by t-Flow is both spatially Lipschitz and admits a
controlled temporal derivative, which is crucial for bounding the discretization error in Theorem 3.
**Proposition 4.1.** _Let v_ _[D]_ _be the minimizer of the t-Flow objective (Equation 3)._ _Under Assumption 3_
_with α ≥_ 2 _d_ + _ν_ + 2 _, there exist constants B_ 1 _, B_ 2 _such that, for all t ∈_ [0 _, T_ ] _:_


_1._ _(Spatial Lipschitzness) The vector field v_ _[D]_ ( _z, t_ ) _is L_ 1 _-Lipschitz in z, with L_ 1 := (1 _d−_ + _Tν_ ) [2] _[B]_ [1] _[.]_
_2._ _(Temporal Regularity) The time derivative of the velocity field is bounded as_
��� _∂t∂_ _[v]_ [(] _[z, t]_ [)] ��� _≤_ (1 _−_ 1 _T_ ) [2] _[∥][z][∥]_ [+] (1 _−_ 1 _T_ ) [2] _[B]_ [1][ +] 1 _−_ 1 _T_ _ν_ + _ν_ _d_ 2(13 _−_ _[√]_ _νT_ ) [2]      - _B_ 2 + 3 _B_ 1 [2]      - _._


_ν_ _d_ 2(13 _−_ _[√]_ _νT_ ) [2] - _B_ 2 + 3 _B_ 1 [2] - _._


The proof is deferred to Appendix G.1. To the best of our knowledge, the only prior work that
controlled the temporal Lipschitzness of the vector field in order to bound discretization error is
Zhou & Liu (2025). However, their analysis required the data distribution to have bounded support,
whereas our result only assumes a polynomial tail bound. For spatial Lipschitzness, existing results
either imposed stronger conditions on the data distribution (Zhou & Liu, 2025; Benton et al., 2024;
Gao et al., 2024) or studied different problem settings (Cordero-Encinar et al., 2025). We can now
quantify the discretization error of t-Flow.
**Theorem 3** (Discretization Error of t-Flow) **.** _Consider t-Flow in Euclidean space._ _Let π_ 1 _[D]_ _[denote]_
_the data distribution supported on_ R _[d]_ _, and_ _π_ ˆ _T_ _[D]_ _[be the law of generated sample]_ _[z][T]_ _[obtained by Euler]_
_discretization with constant step size h, up to time T_ _(see line 9 of Algorithm 1)._ _Under Assumption_
_3 with α_ _≥_ 2 _d_ + _ν_ + 2 _,_ _Assumption 2,_ _and Assumption 1,_ _there exists a constant D_ 3 _,_ _depending_
_polynomially on_ 1 _−_ 1 _T_ _[,][ d][,][ ν][, and on][ B]_ [1] _[, B]_ [2] _[,]_ [ E][[] _[∥][Z]_ [1] _[∥]_ [2][]] _[,]_ [ E][[] _[∥][Z]_ [0] _[∥]_ [2][]] _[, such that]_


_W_ 2( _π_ 1 _[D][,]_ [ ˆ] _[π]_ _T_ _[D]_ [)] _[ ≤]_ _[e]_ [6] _[L]_ [1]

_L_ 1


~~�~~ _h_ [2] _D_ 3 + _ε_ [2] + (1 _−_ _T_ ) ~~�~~ 2�E[ _∥Z_ 1 _∥_ [2] ] + E[ _∥Z_ 0 _∥_ [2] ]� _._


The proof is provided in Appendix G.2. The error bound consists of two terms. The first term captures
the discretization error (from Euler steps of size _h_ ) and the neural network approximation error
(measured by _ε_ ). Both vanish as _h →_ 0 and _ε →_ 0.The second term corresponds to early stopping
error, which decreases to zero as _T_ _→_ 1. Thus, by taking _T_ close to 1 and ensuring accurate vector
field approximation with sufficiently small step size, we can guarantee high-quality samples.


We now compare our result with recent related works. Bansal et al. (2024) did not analyze the
Lipschitz properties of the velocity field, but instead imposed them as assumptions. Zhou & Liu
(2025) established both spatial and temporal Lipschitzness and further analyzed neural network
approximation, but required the data distribution to have bounded support. We note that the exponential dependence on the spatial Lipschitz constant _L_ 1 arises due to non-convexity, and also appears
in existing analyses (Bansal et al., 2024; Zhou & Liu, 2025). It is plausible that this exponential
dependency could be improved to polynomial dependence by following the probabilistic coupling
strategy in Chen et al. (2023), though the resulting algorithm is not purely deterministic.


4.2 PRIMAL SPACE GUARANTEE FOR MIRROR FLOW MATCHING


We next obtain the following primal space guarantee. First note that that the primal space ( _K, g_ _[P]_ )
and the dual space (R _[d]_ _, g_ _[D]_ ) are isometric. Hence, we have the following result.
**Lemma 4.2.** _If the vector field v_ _[D]_ _is L_ 1 _Lipschitz in the dual space_ (R _[d]_ _, g_ _[D]_ ) _, it is L_ 1 _Lipschitz in_
_the primal space_ ( _K, g_ _[P]_ ) _(under the squared Hessian metric)._


To relate Assumption 3 with the distribution in primal space, we impose the following condition.
**Assumption 4.** _(Primal Space Probability Density Function)._ _Denote πEuc_ _[P]_ [(] _[x]_ [)] _[ as the probability]_
_density function for π_ 1 _[P]_ _[in the primal space, under Euclidean metric.]_ _[Assume that][ π]_ _Euc_ _[P]_ [(] _[x]_ [)] _[ is smooth]_
_and that there exists a small constant δ_ 0 _such that_ sup _x∈K\Kδ πEuc_ _[P]_ [(] _[x]_ [)] _[ ≤]_ _[C][pdf]_ _[δ][γ][,][ ∀][δ]_ _[≤]_ _[δ]_ [0] _[.]_


7


**Theorem 4.** _Let_ _π_ ˆ _T_ _[P]_ _[be the law of output samples generated by Algorithm 1 (i.e., the law of]_ _[x][T]_ _[in]_
_Line 10)._ _Under Assumption 1 and 4, with κ ≤_ 2 _d_ + _γν_ +2 _[, and we further require][ κ <]_ _[β]_ 2 _[, there exists]_

      _constant L_ 1 _, D_ 3 _and M_ := 2�E[ _∥Z_ 1 _∥_ [2] ] + E[ _∥Z_ 0 _∥_ [2] ]� _such that_


_W_ 2( _π_ 1 _[P]_ _[,]_ [ ˆ] _[π]_ _T_ _[P]_ [)] _[ ≤]_ _[e]_ [6] _[L]_ [1] - _h_ [2] _D_ 3 + _ε_ [2] + (1 _−_ _T_ ) _M._

_L_ 1


The proof is provided in Appendix G.3 and essentially follows by Proposition 2.2 and Theorem 3.


5 EXPERIMENTS


We demonstrate the effectiveness of our approach by performing numerical simulation (see section
5.1) and real world data experiments on AFHQv2 dataset (see section 5.2). The numerical simulation
is performed on a personal laptop using a CPU. The real world data experiments were performed on
a single A100 GPU.


5.1 NUMERICAL SIMULATION


We build on the experimental setup of Li et al. (2025) and conduct numerical simulations on two
representative constrained generative modeling tasks. The first task is a 10-dimensional polytope
problem, defined as _{x_ _∈_ R [10] : _a_ _[⊤]_ _i_ _[x]_ _[<]_ _[b][i][,]_ _[i]_ [=] [1] _[,]_ [ 2] _[, . . .,]_ [ 30] _[}]_ [,] [with] [constraints] [loaded] [from] [a]
pre-specified data file from Li et al. (2025). The target distribution is a uniform mixture of Gaussians,
where the means are partly sampled at random and partly human-designed to stress-test the model
(e.g., ( _−_ 3 _, −_ 3 _,_ 3 _,_ 3 _, . . ., −_ 3) _∈_ R [10] ), and covariances are fixed to 0 _._ 4 _I_ 10. The second task is a
6-dimensional _L_ 2 ball problem, defined as _{x ∈_ R [6] : _∥x∥_ [2] _<_ 25 _}_, with target distributions generated
in the same manner as in the polytope case. For both tasks, we used a simple MLP network with
4 layers, and hidden layer size being 128. We used ELU activation function. We perform 10 _,_ 000
training iterations.


We implemented our method with _κ_ = 0 _._ 3 and used a _t_ -Flow prior with _ν_ = 10. As shown in Table 1
and Table 2, our approach consistently outperforms both Gauge Flow Matching (Li et al., 2025)
and Reflected Flow Matching (RFM) (Xie et al., 2024). Across both tasks, our method achieves
lower KL divergence and smaller Maximum Mean Discrepancy (MMD) values, while simultaneously
guaranteeing sample feasibility. For the _L_ 2 ball case, Gauge Flow Matching is omitted since it
coincides with Reflected Flow Matching.


We also evaluated the performance of our approach under different choices of _µ, κ_ for a 10 dimensional polytope task. The results are presented in Figure 3. In addition, we visualized the dual space
distribution to justify that our mirror map doesn’t induce heavy tail, whereas the log-barrier does;
See Appendix D. Empirically, we observed that t-Flow outperforms G-Flow. Also, larger values of _κ_
would induce a tail that is heavier than smaller values of _κ_ . The result indicates that a large _ν_ would
require a smaller _κ_, which is consistent with our theoretical findings.


(a) MMD vs Kappa


Figure 3: We test the performance of t-Flow and G-Flow for different values of _κ_ . For t-Flow, we
compare the performance among different choices of _ν_ (degree of freedom). We use RFM as baseline
comparison. Networks are trained with 40 _,_ 000 iterations.


8


Table 1: Performance comparison with 10-dimensional polytope constraints. Results are based on an
average of 10 runs, each run with an average of 10 _,_ 000 samples. MMD values are scaled by 10 _[−]_ [2] .


**Method** **MMD** _↓_ **KL Divergence** _↓_ **Feasibility**


Mirror t-Flow **0** _._ _±_ **0** _._ **1** _._ _±_ **0** _._ 100%
Mirror G-Flow 1 _._ 006 _±_ 0 _._ 016 1 _._ 447 _±_ 0 _._ 046 100%
Gauge Vanilla (Li et al., 2025) 1 _._ 828 _±_ 0 _._ 011 5 _._ 023 _±_ 0 _._ 073 95 _._ 257 _±_ 0 _._ 150%
Gauge Reflect (Li et al., 2025) 1 _._ 830 _±_ 0 _._ 011 5 _._ 057 _±_ 0 _._ 075 100%
RFM (Xie et al., 2024) 1 _._ 217 _±_ 0 _._ 007 2 _._ 034 _±_ 0 _._ 052 100%
MDM (Liu et al., 2023a) 1 _._ 258 _±_ 0 _._ 013 1 _._ 708 _±_ 0 _._ 054 100%


Table 2: Performance comparison on 6-dimensional _L_ 2 ball constraints. Results are based on an
average of 10 runs, each run with an average of 10 _,_ 000 samples. MMD values are scaled by 10 _[−]_ [2] .


**Method** **MMD** _↓_ **KL Divergence** _↓_ **Feasibility**


Mirror t-Flow **5** _._ _±_ **0** _._ **0** _._ _±_ **0** _._ 100%
Mirror G-Flow 6 _._ 244 _±_ 0 _._ 286 0 _._ 176 _±_ 0 _._ 015 100%
RFM (Xie et al., 2024) 5 _._ 935 _±_ 0 _._ 222 0 _._ 285 _±_ 0 _._ 012 100%
MDM (Liu et al., 2023a) 36 _._ 156 _±_ 0 _._ 102 8 _._ 017 _±_ 0 _._ 046 100%


We also implemented MDM (Liu et al., 2023a) for both tasks. We remark that in the original
MDM paper, the author only provided a closed-form formula of their mirror map (and its inverse)
under specific assumptions of the polytope, which can’t be applied for an arbitrary polytope. For
an implementation of log-barrier, the inverse mirror map is difficult to solve, and therefore we
implemented MDM with regularized log-barrier. For _L_ 2 ball case, we implemented MDM with the
closed form mirror map provided in Liu et al. (2023a). We observed that the neural network failed to
learn useful information, which is likely due to the heavy tailed nature of the inducede dual space
distribution.


These experiments highlight the advantages of our method. By jointly choosing mirror maps
and priors based on careful analysis, our approach achieves superior performance on numerical
benchmarks while preserving feasibility by construction. The ability to obtain tighter divergence
metrics under strict feasibility underscores its promise for high-dimensional constrained generative
modeling, demonstrating robustness across geometries (polytope vs. _L_ 2 ball) and scalability to
practical domains where constraints are central.


5.2 REAL-DATA APPLICATION: WATERMARKED IMAGE GENERATION


Following Liu et al. (2023a), we evaluate our method on the task of 64 _×_ 64 watermarked image
generation using the AFHQv2 dataset. We begin by generating parameters ( _ai, bi, ci_ ), which serve as
user-specific private keys. These parameters define a polytope _K_ = _{x_ : _ci_ _< a_ _[⊤]_ _i_ _[x < b][i][}]_ [, where an]
image can be vectorized and checked for feasibility: an image lying inside _K_ is verifiably generated by
the model. During training, we first watermark the AFHQv2 images by projecting them (with added
noise) onto the polytope, thereby producing a watermarked dataset. We then use these watermarked
images as training data and compare the performance of Mirror Diffusion Models (MDM) (Liu et al.,
2023a) with our proposed Mirror _t_ -Flow approach.


A crucial component is the initialization used for the models. We first train both methods with random
neural network initialization under a limited training budget (24 hours). We set the mirror map
parameter as _κ_ = 0 _._ 1 for our method, with random initialization. We first report the CMMD metric
(Jayasumana et al., 2024). CMMD combines CLIP embedding with Maximum Mean Discrepancy
metric and is considered more reliable than FID for evaluating generative models. With 10 _,_ 000 generated images, our approach achieves a CMMD score of 0 _._ 177, which is competitive with the MDM
baseline (Liu et al., 2023a), calculated to be 0 _._ 152. Nevertheless, as shown in Figure 4(a), our method


9


Table 3: Performance comparison on watermarked image generation on the AFHQ2 dataset. Both
implementations are initialized at EDM (Karras et al., 2022) checkpoint. For MDM, we use the code
from Liu et al. (2023a). For flow matching, we apply the training framework from Lee et al. (2024).


**Method** **FID (50k)** _↓_ **CMMD** **training time**


Mirror Flow ( _κ_ = 0 _._ 05) 4 _._ 27 0 _._ 023 3 hours
Mirrod Diffusion Model (Liu et al., 2023a) 7 _._ 29 0 _._ 170 13 hours


(a) With random initialization (b) With EDM checkpoint initialization


Figure 4: Samples of generated watermarked images from the AFHQv2 64 _×_ 64 dataset. Constraint
satisfaction were checked with built-in functions of Liu et al. (2023a).


already produces visually high-quality samples within a limited training budget, demonstrating strong
potential for further improvements with better initializations.


Towards that, in Table 3 we next report results when the models are initialized at EDM (Karras
et al., 2022) checkpoint for AFHQv2 dataset; the corresponding sample images are displayed in
in Figure 4(b). We note that in this case, our method achieves superior CMMD and FID scores,
requiring a smaller amount of training time. Finally, we remark that if we initialize at the checkpoint
for a flow matching model from Lee et al. (2024), the FID (50k) can achieve 3 _._ 14 after 1 _._ 5 hours
of training. This value is similar to 3 _._ 05 reported in Liu et al. (2023a), while fully executing their
scheduled number of iterations could result in an estimated training time up to several hundred hours
in our experimental setup.


6 CONCLUSION


We introduced _t-Flow_, a flow-matching framework with Student- _t_ priors, and established rigorous
guarantees on both spatial Lipschitzness and temporal regularity of the underlying velocity field . Our
analysis yielded the first error bounds for flow matching under polynomial tail assumptions, thereby
extending prior results beyond bounded-support assumptions. We further demonstrated that _t_ -Flow
provides robust sample quality in practice, particularly in scenarios where Gaussian priors fail to
capture heavy-tailed structures. Beyond technical guarantees, our results emphasize that successful
generative modeling on complex domains requires a careful co-design of mirror maps and priors,
rather than defaulting to standard choices. This perspective opens up several promising avenues.
One direction is exploring adaptive choices of degrees of freedom in the _t_ -prior could yield even
more flexibility, enabling flows that automatically adapt to local tail behavior of the data. Another is
extending _t_ -Flow to constrained domains with non-convex geometry, potentially leveraging landing
techniques. On the theory front, improving the exponential dependence on Lipschitz constants, for
example via probabilistic couplings or randomized flow strategies is interesting. Finally integrating
_t_ -Flow with hybrid diffusion–flow architectures and energy-based models offers yet another exciting
path, combining the complementary strengths of these paradigms.


ACKNOWLEDGMENTS


Research of Krishnakumar Balasubramanian was supported in part by National Science Foundation
(NSF) grant DMS-2413426. Research of Shiqian Ma was supported in part by NSF grants CCF2311275 and ECCS-2326591 and ONR grant N00014-24-1-2705.


10


REPRODUCIBILITY STATEMENT


Proofs for the theoretical results are presented in the Appendix. Codes to reproduce the experimental
results are provided in the supplementary material. **LLM usage:** LLM was used only to polish the
writing.


REFERENCES


Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying
framework for flows and diffusions. _arXiv preprint arXiv:2303.08797_, 2023.


Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic
interpolants. In _The Eleventh International Conference on Learning Representations_, 2023. URL
[https://openreview.net/forum?id=li7qeBbCR1t.](https://openreview.net/forum?id=li7qeBbCR1t)


Vansh Bansal, Saptarshi Roy, Purnamrita Sarkar, and Alessandro Rinaldo. On the wasserstein
convergence and straightness of rectified flow. _arXiv preprint arXiv:2410.14949_, 2024.


Joe Benton, George Deligiannidis, and Arnaud Doucet. Error bounds for flow matching methods. _Transactions_ _on_ _Machine_ _Learning_ _Research_, 2024. ISSN 2835-8856. URL [https:](https://openreview.net/forum?id=uqQPyWFDhY)
[//openreview.net/forum?id=uqQPyWFDhY.](https://openreview.net/forum?id=uqQPyWFDhY)


Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In _The_ _Twelfth_
_International Conference on Learning Representations_, 2024. [URL https://openreview.](https://openreview.net/forum?id=g7ohDlTITL)
[net/forum?id=g7ohDlTITL.](https://openreview.net/forum?id=g7ohDlTITL)


Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, and Adil Salim. The probability flow
ode is provably fast. _Advances in Neural Information Processing Systems_, 36:68552–68575, 2023.


Jacob K Christopher, Stephen Baek, and Nando Fioretto. Constrained synthesis with projected
diffusion models. _Advances in Neural Information Processing Systems_, 37:89307–89333, 2024.


Paula Cordero-Encinar, O Deniz Akyildiz, and Andrew B Duncan. Non-asymptotic analysis of
diffusion annealed langevin monte carlo for generative modelling. _arXiv preprint arXiv:2502.09306_,
2025.


Berthy Feng, Ricardo Baptista, and Katherine Bouman. Neural approximate mirror maps for constrained diffusion models. In _The Thirteenth International Conference on Learning Representations_,
2025. [URL https://openreview.net/forum?id=vgZDcUetWS.](https://openreview.net/forum?id=vgZDcUetWS)


Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, and Michael John Hutchinson.
Diffusion models for constrained domains. _Transactions on Machine Learning Research_, 2023a.
ISSN 2835-8856. URL [https://openreview.net/forum?id=xuWTFQ4VGO.](https://openreview.net/forum?id=xuWTFQ4VGO) Expert
Certification.


Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, and Valentin De Bortoli. Metropolis
sampling for constrained diffusion models. _Advances in Neural Information Processing Systems_,
36:62296–62331, 2023b.


Yuan Gao, Jian Huang, and Yuling Jiao. Gaussian interpolation flows. _Journal of Machine Learning_
_Research_, 25(253):1–52, 2024.


Zhengyan Huan, Jacob Boerma, Li-Ping Liu, and Shuchin Aeron. Efficient constraint-aware flow
matching via randomized exploration. _arXiv preprint arXiv:2508.13316_, 2025.


Tuomas Hytonen,¨ Jan Van Neerven, Mark Veraar, and Lutz Weis. _Analysis_ _in_ _Banach_ _spaces_,
volume 1. Springer, 2016.


Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and
Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In
_Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_, pp.
9307–9315, 2024.


11


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _Advances in neural information processing systems_, 35:26565–26577,
2022.


Shervin Khalafi, Dongsheng Ding, and Alejandro Ribeiro. Constrained diffusion models via dual
training. _Advances in Neural Information Processing Systems_, 37:26543–26576, 2024.


Minu Kim, Yongsik Lee, Sehyeok Kang, Jihwan Oh, Song Chong, and Se-Young Yun. Preference
alignment with flow matching. _Advances in Neural Information Processing Systems_, 37:35140–
35164, 2024.


Sangyun Lee, Zinan Lin, and Giulia Fanti. Improving the training of rectified flows. _Advances in_
_neural information processing systems_, 37:63082–63109, 2024.


Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Towards non-asymptotic convergence for diffusionbased generative models. In _The Twelfth International Conference on Learning Representations_,
2024. [URL https://openreview.net/forum?id=4VGEeER6W9.](https://openreview.net/forum?id=4VGEeER6W9)


Ruilin Li, Molei Tao, Santosh S Vempala, and Andre Wibisono. The mirror langevin algorithm
converges with vanishing bias. In _International Conference on Algorithmic Learning Theory_, pp.
718–742. PMLR, 2022.


Xinpeng Li, Enming Liang, and Minghua Chen. Gauge flow matching for efficient constrained
generative modeling over general convex set. In _ICLR 2025 Workshop on Deep Generative Model_
_in Machine Learning:_ _Theory, Principle and Efficacy_, 2025.


Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow
matching for generative modeling. In _The Eleventh International Conference on Learning Repre-_
_sentations_, 2023. [URL https://openreview.net/forum?id=PqvMRDCJT9t.](https://openreview.net/forum?id=PqvMRDCJT9t)


Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao. Mirror diffusion models for
constrained and watermarked generation. _Advances in Neural Information Processing Systems_, 36:
42898–42917, 2023a.


Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer
data with rectified flow. In _The Eleventh International Conference on Learning Representations_,
2023b. [URL https://openreview.net/forum?id=XVjTT1nw5z.](https://openreview.net/forum?id=XVjTT1nw5z)


Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer
data with rectified flow. In _The Eleventh International Conference on Learning Representations_,
2023c. [URL https://openreview.net/forum?id=XVjTT1nw5z.](https://openreview.net/forum?id=XVjTT1nw5z)


Aaron Lou and Stefano Ermon. Reflected diffusion models. In _International Conference on Machine_
_Learning_, pp. 22675–22701. PMLR, 2023.


Kushagra Pandey, Jaideep Pathak, Yilun Xu, Stephan Mandt, Michael Pritchard, Arash Vahdat,
and Morteza Mardani. Heavy-tailed diffusion models. In _The Thirteenth International Confer-_
_ence on Learning Representations_, 2025. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=tozlOEN4qp)
[tozlOEN4qp.](https://openreview.net/forum?id=tozlOEN4qp)


Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models
with minibatch optimal transport. _Transactions on Machine Learning Research_, 2024. ISSN 28358856. [URL https://openreview.net/forum?id=CD9Snc73AW.](https://openreview.net/forum?id=CD9Snc73AW) Expert Certification.


Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, and Christopher Vincent
Rackauckas. Physics-constrained flow matching: Sampling generative models with hard constraints.
_arXiv preprint arXiv:2506.04171_, 2025.


Nuri Mert Vural, Lu Yu, Krishna Balasubramanian, Stanislav Volgushev, and Murat A Erdogdu.
Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise variance.
In _Conference on Learning Theory_, pp. 65–102. PMLR, 2022.


12


Zhengchao Wan, Qingsong Wang, Gal Mishne, and Yusu Wang. Elucidating flow matching ODE
dynamics via data geometry and denoisers. In _Forty-second International Conference on Machine_
_Learning_, 2025. [URL https://openreview.net/forum?id=f5czhqYK3H.](https://openreview.net/forum?id=f5czhqYK3H)


Yuqing Wang, Ye He, and Molei Tao. Evaluating the design space of diffusion-based generative
models. _Advances in Neural Information Processing Systems_, 37:19307–19352, 2024.


Tianyu Xie, Yu Zhu, Longlin Yu, Tong Yang, Ziheng Cheng, Shiyue Zhang, Xiangyu Zhang, and
Cheng Zhang. Reflected flow matching. In _Forty-first_ _International_ _Conference_ _on_ _Machine_
_Learning_, 2024. [URL https://openreview.net/forum?id=Sf5KYznS2G.](https://openreview.net/forum?id=Sf5KYznS2G)


Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, and Shuaicheng Liu. Flowpolicy:
Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation.
In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 39, pp. 14754–14762,
2025.


Zhengyu Zhou and Weiwei Liu. An error analysis of flow matching for deep generative modeling. In _Forty-second_ _International_ _Conference_ _on_ _Machine_ _Learning_, 2025. URL [https:](https://openreview.net/forum?id=vES22INUKm)
[//openreview.net/forum?id=vES22INUKm.](https://openreview.net/forum?id=vES22INUKm)


A VISUAL ILLUSTRATION OF METHODOLOGICAL CHALLENGES


We illustrate the benefits of our approach in Figure 5. The constraint set is a polytope _K_ = _{x ∈_ R [2] :
_Ax < b}_ with


�1 _−_ 1 1 _−_ 5 _−_ 1 _/_ 3�
_A_ _[⊤]_ = _,_ _b_ _[⊤]_ = (10 30 1 90 5) _._
1 _−_ 1 _−_ 1 1 1


The target _π_ 1 is a mixture of three Gaussians, truncated to _K_ : _N_ ([ _−_ 10 _,_ 0] _[T]_ _,_ diag (8 _,_ 2)) with weight
0 _._ 6, _N_ ([ _−_ 15 _, −_ 10] _[T]_ _,_ diag (1 _,_ 1)) with weight 0 _._ 2, and _N_ ([3 _,_ 3] _[T]_ _,_ diag (0 _._ 5 _,_ 0 _._ 25)) with weight 0 _._ 2.
We compare G-flow (Gaussian prior) and t-flow (Student- _t_ prior) under both the log-barrier mirror map and our proposed regularized map (Figures 5(b)–5(e)), alongside samples from the true
target (Figure 5(a)). Vector fields were parameterized by neural networks and simulated via Euler
discretization ( _h_ = 0 _._ 1) with early stopping. As shown in Figure 5, our approach achieves robust
mode recovery and faithful constrained sampling, consistently outperforming Gaussian-based flow
methods.


B EXAMPLES VERIFYING PROPOSITION 2.2


Proposition 2.2 can be specialized to several classical examples of convex sets.


1. _L_ 2 **ball.** Consider the closed Euclidean ball _K_ = _{x ∈_ R _[d]_ : _∥x∥_ _< R}_ . Define the mirror potential
Ψ( _x_ ) = _−_ 1 _−_ 1 _κ_  - _R_ [2] _−∥x∥_ [2][�][1] _[−][κ]_ + [1] 2 _[∥][x][∥]_ [2] _[.]_ [ In this case the barrier function is] _[ ϕ]_ [(] _[x]_ [)] [=] _[∥][x][∥]_ [2] _[−]_ _[R]_ [2][,]

which is clearly smooth and convex. Moreover, its gradient is bounded on _K_, satisfying the required
assumptions.

2. **Polytope.** Let _K_ = _{x ∈_ R _[d]_ : _a_ _[T]_ _i_ _[x < b][i][,]_ _[∀][i][ ∈]_ [[] _[m]_ []] _[}]_ [ be a polytope defined by] _[ m]_ [ linear inequalities.]
Define the potential Ψ( _x_ ) = _−_ [�] _[m]_ _i_ =1 1 _−_ 1 _κ_  - _bi −_ _a_ _[T]_ _i_ _[x]_ �1 _−κ_ + 12  - _dj_ =1 _[x]_ _j_ [2][.] [Here the barrier functions]
are _ϕi_ ( _x_ ) = _a_ _[T]_ _i_ _[x][ −]_ _[b][i]_ [.] [Each] _[ ϕ][i]_ [ is affine (hence smooth and convex), with Hessian] _[ ∇]_ [2] _[ϕ][i]_ [(] _[x]_ [) = 0][, and]
its gradient is bounded uniformly. Thus the conditions are again satisfied.


C VISUAL ILLUSTRATION CORRESPONDING SECTION 2.2


We illustrate the blow-up phenomenon discussed in Section 2.2. In Figure 7(a), 7(b), 7(c) we illustrate
the _t →_ 0 limit, blows-up for small _t_, and _t →_ 1 limit respectively, for the G-flow. The corresponding
Figure 7(d), 7(e), 7(f) for the t-flow is more benign.


13


(a) Ground truth (b) G-flow Log Barrier (c) t-flow Log Barrier


(d) G-flow proposed mirror map (e) T-flow proposed mirror map


Figure 5: Figure 5(a) shows the ground-truth reference distribution. Figures 5(b) and 5(c) illustrate
that the log-barrier method performs poorly (both with G or t-flow), while Figure 5(d) demonstrates
that G-flow (with our mirror map) fails to capture the mode centered near ( _−_ 10 _,_ 0). In contrast,
Figure 5(e) shows that t-flow with our mirror map covers the target distribution better. All results are
obtained with discretization step size _h_ = 0 _._ 1. See also Figure 6 for a zoomed-in illustration near the
boundary.


(a) Ground truth (b) G-flow proposed mirror map (c) t-flow proposed mirror map


Figure 6: We generate a total of 10 _,_ 000 samples, but for visualization we only display those lying
in the boundary region _x_ _∈_ [ _−_ 14 _, −_ 12] _, y_ _∈_ [0 _,_ 2]. Figure 6(a) shows the ground-truth reference
distribution. Figures 6(b) and 6(c) demonstrate that, near the boundary, t-flow provides a closer
approximation to the ground truth than G-flow.


D ADDITIONAL EXPERIMENTAL RESULTS


We visualize the dual space distribution induced by different mirror maps. To plot the figure, we
generate 10 _,_ 000 true samples inside a 10 dimensional polytope, and map them to a dual space using
different mirror maps. Then we select a two-dimensional subspace _y_ 2 _× y_ 5 of the dual space R [10], and
visualize the samples in this subspace. See Figure 8. We observe that the log-barrier indeed produces
heavy tails, whereas our mirror map with _κ_ = 0 _._ 2 doesn’t.


14


(a) G Prior: _t →_ 0 (b) G Prior: small _t_ (c) G Prior: large _t_, large _x_


(d) t Prior: _t →_ 0 (e) t Prior: small _t_ (f) t Prior: large _t_, large _x_


Figure 7: Illustration for Example 2. (i) Figures 7(a) and 7(d) demonstrate that in the limit _t →_ 0, the
distribution remains well-behaved and does not blow up. (ii) Figure 7(b) shows that for sufficiently
large values of _x_ (here we choose a moderately large _x_ for readability), there exists a small value of _t_
such that the flow with a Gaussian prior diverges. (iii) Figure 7(e) illustrates that such divergence does
not occur when using a Student- _t_ prior. (iv) Finally, Figures 7(c) and 7(f) show that as _t_ approaches 1,
the Gaussian-prior flow becomes unstable, whereas the Student- _t_ prior remains stable.


(a) Dual Space Distribution (b) Dual Space Distribution (Zoomed)


Figure 8: We compare the difference between our mirror map (with different _κ_ ), and the log-barrier.


E PROOFS FOR SECTION 2


**Proof.** [Proof of Lemma 2.1] Recall that for any one dimensional random variable _X_, we have

    - _∞_    - _∞_    - _∞_    - _X_


_∞_ - _∞_

_P_ ( _X_ _≥_ _t_ ) _dt_ =
0 0


_∞_ - _∞_

E[ 1 _X≥t_ ] _dt_ = E[
0 0


_∞_ - _X_

1 _X≥tdt_ ] = E[
0 0


_dt_ ] = E[ _X_ ] _._
0


15


1. **First claim.**
Assume _P_ ( _∥Y ∥≥_ _R_ ) _≥_ _RC_ _[p]_ [.] [Hence we know (where] _[ s]_ [ :=] _[ t]_ [1] _[/p]_ [, so that] _[ dt]_ [ =] _[ ps][p][−]_ [1] _[ds]_ [)]


     - _∞_
E[ _∥Y ∥_ _[p]_ ] =


_∞_ - _∞_

_P_ ( _∥Y ∥_ _[p]_ _≥_ _t_ ) _dt_ =
0 0


_∞_ - _∞_

_P_ ( _∥Y ∥≥_ _t_ [1] _[/p]_ ) _dt_ =
0 0


_P_ ( _∥Y ∥≥_ _s_ ) _ps_ _[p][−]_ [1] _ds_
0


 - _∞_
_≥_

0


_C_ - _∞_
_s_ _[p]_ _[ps][p][−]_ [1] _[ds]_ [ =] 0 _Cps_ _[−]_ [1] _ds._


The integral does not converge.

2. **Second claim.**
Assume _P_ ( _∥Y ∥≥_ _R_ ) _≤_ _RC_ _[β]_ [.]


     - _∞_
E[ _∥Y ∥_ _[p]_ ] =


_∞_ - _∞_

_P_ ( _∥Y ∥_ _[p]_ _≥_ _t_ ) _dt_ =
0 0


_∞_ - _∞_

_P_ ( _∥Y ∥≥_ _t_ [1] _[/p]_ ) _dt_ =
0 0


_P_ ( _∥Y ∥≥_ _s_ ) _ps_ _[p][−]_ [1] _ds_
0


 - _∞_
_≤_

0


_C_ - _∞_
_s_ _[β]_ _[ps][p][−]_ [1] _[ds]_ [ =] 0 _Cps_ _[p][−]_ [1] _[−][β]_ _ds._


The integral converges iff _p −_ 1 _−_ _β_ _< −_ 1, i.e., _β_ _> p_ .


**Example 5.** Let _K ⊆_ R [2] be a triangle defined by the following inequalities:


100 _x_ 1 + 0 _._ 01 _x_ 2 _≤_ 1 _,_
100 _x_ 1 _−_ 0 _._ 01 _x_ 2 _≤_ 1 _,_
_−x_ 1 _≤_ 0 _._

Recall that for each constraint _a_ _[T]_ _i_ _[x][ ≤]_ _[b][i]_ [ we can define] _[ ψ][i]_ [(] _[x]_ [) =] _[ −]_ [log(] _[b][i]_ _[−][a]_ _i_ _[T]_ _[x]_ [)][. Then the log-barrier]
is _ψ_ ( _x_ ) = [�] _i_ _[ψ][i]_ [(] _[x]_ [)][.] [Take derivative, we obtain] _[ ∇][ψ]_ [(] _[x]_ [) =][ �] _i_ _bi−_ 1 _a_ _[T]_ _i_ _[x]_ _[a][i]_ [.]


- 100

_−_ 0 _._ 01


+ [1]

_k_ 1


- _−_ 1�
0


- 100 - 1

+

0 _._ 01 1 _−_ 100 _k_ 1 + 0 _._ 01 _k_ 2


_∇ψ_ ( _k_ 1 _, k_ 2) = 

_i_


1 1

[=]
_bi −_ _a_ _[T]_ _i_ _[xa][i]_ 1 _−_ 100 _k_ 1 _−_ 0 _._ 01 _k_ 2


=


�100( (1 _−_ 100 _k_ 1 _−_ 0 _._ 012 _k−_ 2200)(1 _k−_ 1100 _k_ 1+0 _._ 01 _k_ 2) [)] _[ −]_ _k_ [1] 1

0 _._ 01( (1 _−_ 100 _k_ 1 _−_ 0 _._ 010 _k._ 202)(1 _k−_ 2 100 _k_ 1+0 _._ 01 _k_ 2) [)]


_._


Consider two points ( _k_ 1 _, k_ 2) _,_ ( _k_ 1 _, −k_ 2) _∈_ R [2] in the dual space:


_∥∇ψ_ ( _k_ 1 _, k_ 2) _−∇ψ_ ( _k_ 1 _, −k_ 2) _∥_ =
����


_._
�����


- 0
0 _._ 01( (1 _−_ 100 _k_ 1 _−_ 0 _._ 010 _k._ 204)(1 _k−_ 2 100 _k_ 1+0 _._ 01 _k_ 2) [)]


Hence
_∥∇ψ_ ( _k_ 1 _, k_ 2) _−∇ψ_ ( _k_ 1 _, −k_ 2) _∥_

_∥_ ( _k_ 1 _, k_ 2) _[T]_ _−_ ( _k_ 1 _, −k_ 2) _[T]_ _∥_

= 0 _._ 01( (1 _−_ 100 _k_ 1 _−_ 0 _._ 010 _k._ 204)(1 _k−_ 2 100 _k_ 1+0 _._ 01 _k_ 2) [)] = 2 _×_ 10 _[−]_ [4] 1

2 _k_ 2 (1 _−_ 100 _k_ 1 _−_ 0 _._ 01 _k_ 2)(1 _−_ 100 _k_ 1 + 0 _._ 01 _k_ 2) _[.]_

When ( _k_ 1 _, k_ 2) _→_ 0, we have _[∥∇]_ _∥_ _[ψ]_ ( _k_ [(] 1 _[k]_ _,k_ [1] _[,k]_ 2) [2] _[T]_ [)] _−_ _[−∇]_ ( _k_ _[ψ]_ 1 _,_ [(] _−_ _[k]_ [1] _k_ _[,]_ 2 _[−]_ ) _[T][k]_ [2] _∥_ [)] _[∥]_ _→_ 2 _×_ 10 _[−]_ [4] .


The above example shows that, there are cases when the polytope is “ill-shaped”, and leading to a
very large _Lψ_ .


**Proof.** [Proof of Proposition 2.2] We have


_∇_ Ψ( _x_ ) =


_m_
�( _−ϕi_ ( _x_ )) _[−][κ]_ _∇ϕi_ ( _x_ ) + _x,_


_i_ =1


_m_
�( _−ϕi_ ( _x_ )) _[−][κ]_ _∇_ [2] _ϕi_ ( _x_ ) + _I._


_i_ =1


_∇_ [2] Ψ( _x_ ) = _κ_


_m_
�( _−ϕi_ ( _x_ )) _[−][κ][−]_ [1] _∇ϕi_ ( _x_ ) _∇ϕi_ ( _x_ ) _[T]_ +


_i_ =1


16


Note that _∇_ [2] _ϕi_ ( _x_ ) _⪰_ 0 due to convexity of _ϕi_ . So we have _∇_ [2] Ψ( _x_ ) _⪰_ _I_ . It follows that
_W_ 2( _ν, µ_ ) _≤_ _W_ 2 _,_ Ψ( _ν, µ_ ) _._


Furthermore, _∇_ Ψ( _x_ ) = [�] _i_ _[m]_ =1 [(] _[−][ϕ][i]_ [(] _[x]_ [))] _[−][κ][∇][ϕ][i]_ [(] _[x]_ [) +] _[ x]_ [ so we know]


_m_


_i_ =1


1
_δ_ _[κ]_ _[∥∇][ϕ][i]_ [(] _[x]_ [)] _[∥][.]_


_∥∇_ Ψ( _x_ ) _∥≤∥x∥_ +


_m_


- _∥_ ( _−ϕi_ ( _x_ )) _[−][κ]_ _∇ϕi_ ( _x_ ) _∥≤∥x∥_ +


_i_ =1


Since we assumed _ϕi_ ( _x_ ) are of bounded gradient, we know _∥∇_ Ψ( _x_ ) _∥_ = _[C]_ _δ_ _[κ][′]_ [for some] _[ C]_ _[′]_ [.]


Denote


_Rδ,κ_ = _[C]_ _[′]_ _∥∇_ Ψ( _x_ ) _∥._

_δ_ _[κ]_ _[≥]_ _x_ [sup] _∈Kδ_


Hence we know, _Rδ,κ_ is such that _{x ∈_ R _[d]_ : _∥x∥≤_ _Rδ,κ} ⊇∇_ Ψ( _Kδ_ ). It follows that

_C_
_P_ ( _∥∇_ Ψ( _X_ ) _∥≥_ _Rδ,κ_ ) _≤_ _P_ ( _K\Kδ_ ) _≤_ _CKδ_ _[β]_ = _._

_Rδ,κ_ _[β/κ]_

where note that _Rδ,κ_ _[β/κ]_ _C_ [=] ( _δ_ _[C]_ ~~_[κ]_~~ _[′]_ _C_ [)] _[β/κ]_ [=] _[ C][K][δ][β]_ [.]


F PROOFS FOR SECTION 3


We first provide some definitions related to conditional expectation in an abstract vector space. We
follow the notation in Hytonen et al. (2016). Let¨ ( _S, A_ ) be a measurable space, and _X_ a Banach space.
_L_ _[p]_ ( _S_ ; _X_ ) denote the linear space of all _µ_ -measurable functions from _S_ to _X_, with - _S_ _[∥][f]_ _[∥][p][dµ <][ ∞]_ [.]

When _F_ is a sub- _σ_ -algebra of _A_, _L_ _[p]_ ( _S_ ; _F_ ; _X_ ) represent the _Lp_ space w.r.t. ( _S, F_ _, µ|F_ ).
**Definition F.1.** _(Hyt¨onen et al., 2016, Theorem 2.6.23 and Proposition 2.6.31)_


_If_ _µ_ _is_ _σ-finite_ _on_ _the_ _sub-algebra_ _F_ _,_ _then_ _every_ _f_ _∈_ _L_ [1] ( _S_ ; _X_ ) _admits_ _a_ _unique_ _conditional_
_expectation with respect to F_ _._ _It satisfies_

            -            


      E[ _f_ _|F_ ] _dµ_ =
_F_


_fdµ, ∀F_ _∈_ _F_ _._
_F_


_Furthermore,_ _let_ _g_ _∈_ _L_ [0] ( _S_ ; _F_ ; _X_ 1) _,_ _and_ _that_ _f_ _∈_ _L_ [1] ( _S_ ; _X_ 2) _be_ _σ-integrable_ _over_ _F_ _._ _Let_ _β_ :
_X_ 1 _× X_ 2 _→_ _Y_ _be a bounded bi-linear map._ _Then β_ ( _g, f_ ) _∈_ _L_ [0] ( _S_ ; _Y_ ) _is σ-integrable over F_ _, and_
_we have_
E[ _β_ ( _g, f_ ) _|F_ ] = _β_ ( _g,_ E[ _f_ _|F_ ]) _a.s._


**Proof.** [Proof of Proposition 3.1] In primal space, the corresponding interpolation would be
_d_ [=] _[d]_ _[d]_
_dt_ _[X][t]_ _dt_ _[∇][ψ][∗]_ [(] _[Z][t]_ [) =] _[ ∇]_ [2] _[ψ][∗]_ [(] _[Z][t]_ [)] _dt_ _[Z][t][.]_


_[d]_ _[d]_

_dt_ _[∇][ψ][∗]_ [(] _[Z][t]_ [) =] _[ ∇]_ [2] _[ψ][∗]_ [(] _[Z][t]_ [)]


_dt_ _[Z][t][.]_


Recall that the two minimization problems are:


   min E _∥v_ _[P]_ ( _Xt, t_ ) _−_ _[d]_
_v_


    -    
_[d]_ _g_ _[P]_ and min E _∥v_ _[D]_ ( _Zt, t_ ) _−_ _[d]_

_dt_ _[X][t][∥]_ [2] _v_


_dt_ _[d]_ _[Z][t][∥]_ _g_ [2] _[D]_ 


respectively.


Recall that _∇_ [2] _ψ_ evaluated at _x_ is the inverse of _∇_ [2] _ψ_ _[∗]_ evaluated at _z_ = _∇ψ_ ( _x_ ), i.e., _∇_ [2] _ψ_ ( _x_ ) =
( _∇_ [2] _ψ_ _[∗]_ ( _∇ψ_ ( _x_ ))) _[−]_ [1] . Hence we obtain _∇_ [2] _ψ_ ( _x_ ) _◦∇_ [2] _ψ_ _[∗]_ ( _z_ ) _dt_ _[d]_ _[Z][t]_ [=] _dt_ _[d]_ _[Z][t]_ [.] [Condition on] _[ X][t]_ [=] _[ x]_ [, we]

have


_dt_ _[d]_ _[X][t][∥]_ _g_ [2] _[P]_ [=] _[ g][P]_ - _v_ _[P]_ ( _Xt, t_ ) _−_ _[d]_


_dt_ _[X][t]_


_∥v_ _[P]_ ( _Xt, t_ ) _−_ _[d]_


_[d]_

_dt_ _[X][t][, v][P]_ [ (] _[X][t][, t]_ [)] _[ −]_ _[d]_


    =( _∇_ [2] _ψ_ ( _x_ )) [2] _v_ _[P]_ ( _Xt, t_ ) _−∇_ [2] _ψ_ _[∗]_ ( _Zt_ ) _[d]_


_dt_ _[Z][t]_


_[d]_ _[d]_

_dt_ _[Z][t][, v][P]_ [ (] _[X][t][, t]_ [)] _[ −∇]_ [2] _[ψ][∗]_ [(] _[Z][t]_ [)]


 


  = _g_ _[D]_ _∇_ [2] _ψ_ ( _x_ ) _v_ _[P]_ ( _x, t_ ) _−_ _[d]_


_dt_ _[Z][t]_


_[d]_

_dt_ _[Z][t][,][ ∇]_ [2] _[ψ]_ [(] _[x]_ [)] _[v][P]_ [ (] _[x, t]_ [)] _[ −]_ _[d]_


= _∥∇_ [2] _ψ_ ( _x_ ) _v_ _[P]_ ( _x, t_ ) _−_ _[d]_ _g_ _[D]_ _[.]_

_dt_ _[Z][t][∥]_ [2]


17


Hence we get _∥v_ _[P]_ ( _x, t_ ) _−_ _[d]_


_dt_ _[d]_ _[Z][t][∥]_ _g_ [2] _[D]_ [or equivalently] _[ ∥][v][D]_ [(] _[z, t]_ [)] _[ −]_


_dt_ _[d]_ _[X][t][∥]_ _g_ [2] _[P]_ [=] _[∥∇]_ [2] _[ψ]_ [(] _[x]_ [)] _[v][P]_ [ (] _[x, t]_ [)] _[ −]_ _dt_ _[d]_


_dtd_ _[Z][t][∥]_ _g_ [2] _[D]_ [=] _[ ∥∇]_ [2] _[ψ][∗]_ [(] _[z]_ [)] _[v][D]_ [(] _[z, t]_ [)] _[ −]_ _dt_ _[d]_ _[X][t][∥]_ _g_ [2] _[P]_ [ .] [So we get]


_v_ _[D]_ ( _z, t_ ) = _∇_ [2] _ψ_ ( _x_ ) _v_ _[P]_ ( _x, t_ ) _,_ _v_ _[P]_ ( _x, t_ ) = _∇_ [2] _ψ_ _[∗]_ ( _z_ ) _v_ _[D]_ ( _z, t_ ) _._


The equivalence follows from the change of variable formula.


Now we show the last claim. Now consider _G_ to be the sigma algebra corresponding to _Xt_ = _x_ . Note
that each tangent space _TxM_ is a Hilbert space, with Riemannian metric _g_ . Then for any _Y_ (that is
measurable in _G_ ), we have


E[ _∥_ _[d]_


_[d]_ _g_ ( _x_ ) [] =][ E][[] _[∥]_ _[d]_

_dt_ _[X][t][ −]_ _[Y][ ∥]_ [2] _dt_


_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[] _dt_

_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[] _dt_

_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[]


_[d]_ [=] _[ x]_ [] +][ E][[] _[d]_

_dt_ _[X][t][|][X][t]_


[=] _[ x]_ []] _[ −]_ _[Y][ ⟩]_ []] _[.]_
_dt_ _[X][t][|][X][t]_


[=] _[ x]_ []] _[ −]_ _[Y][ ∥]_ [2] _g_ ( _x_ ) []]
_dt_ _[X][t][|][X][t]_


= E[ _∥_ _[d]_

_dt_

+ 2E[ _⟨_ _[d]_


_[d]_ [=] _[ x]_ []] _[∥]_ [2] _g_ ( _x_ ) [] +][ E][[] _[∥]_ [E][[] _[d]_

_dt_ _[X][t][|][X][t]_


_[d]_ [=] _[ x]_ []] _[,]_ [ E][[] _[d]_

_dt_ _[X][t][|][X][t]_


[=] _[ x]_ []] _[ −]_ _[Y][ ∥]_ [2] _g_ ( _x_ ) []]
_dt_ _[X][t][|][X][t]_


Since _f_ := E[ _dt_ _[d]_ _[X][t][|][X][t]_ [=] _[ x]_ []] _[ −]_ _[Y]_ [is measurable in] _[ G]_ [, we have]


E[ _⟨_ _[d]_


_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[]


_[d]_ [=] _[ x]_ []] _[, f]_ _[⟩]_ [] =][ E][[] _[⟨]_ _[d]_

_dt_ _[X][t][|][X][t]_


_[d]_ _[d]_

_dt_ _[X][t][, f]_ _[⟩]_ []] _[ −]_ [E][[] _[⟨]_ [E][[]

_[d]_ _[d]_

_dt_ _[X][t][, f]_ _[⟩]_ []] _[ −⟨]_ [E][[][E][[]


[=] _[ x]_ []] _[, f]_ _[⟩]_ []]
_dt_ _[X][t][|][X][t]_


= E[ _⟨_ _[d]_


[=] _[ x]_ []]] _[, f]_ _[⟩]_ [= 0] _[.]_
_dt_ _[X][t][|][X][t]_


where the last equality is by tower property (Hyt¨onen et al., 2016, Proposition 2.6.33).


Hence we get


E[ _∥_ _[d]_


_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[] _dt_

_[d]_ _[d]_

_dt_ _[X][t][ −]_ [E][[] _dt_


_[d]_ _g_ ( _x_ ) [] =][ E][[] _[∥]_ _[d]_

_dt_ _[X][t][ −]_ _[Y][ ∥]_ [2] _dt_


_[d]_ [=] _[ x]_ []] _[∥]_ [2] _g_ ( _x_ ) [] +][ E][[] _[∥]_ [E][[] _[d]_

_dt_ _[X][t][|][X][t]_


[=] _[ x]_ []] _[ −]_ _[Y][ ∥]_ [2] _g_ ( _x_ ) []]
_dt_ _[X][t][|][X][t]_


_≥_ E[ _∥_ _[d]_


[=] _[ x]_ []] _[∥]_ [2] _g_ ( _x_ ) []] _[,][ ∀][Y]_ _[∈G][.]_
_dt_ _[X][t][|][X][t]_


It follows that among all _Y_ being measurable in _G_, the choice _Y_ = E[ _[d]_


It follows that among all _Y_ being measurable in _G_, the choice _Y_ = E[ _dt_ _[X][t][|][X][t]_ [=] _[ x]_ []][ minimize the]

problem. Hence _v_ _[P]_ ( _x, t_ ) = E[ _[d]_ _[X][t][|][X][t]_ [=] _[ x]_ []][.]


_dt_ _[X][t][|][X][t]_ [=] _[ x]_ []][.]


G PROOFS FOR SECTION 4


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


We start with several intermediate results. Define _pt,x_ ( _z_ 1) = (1+ _ν_ [1]


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1+ _ν_ _[∥]_ 1 _−t_ [1] _[∥]_ [)] 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_ [Through-]

2 _p_ ( _z_ ) _dz_ [.]


R _[d]_ [(1+] _ν_ [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


out this section, to make the notation compatible with _pt,x_ ( _z_ 1), we use _p_ to denote the probability
density function of the data distribution, supported on Euclidean space.


**Proposition G.1.** _Under Assumption 3 with α ≥_ 2 _d_ + _ν_ + 2 _, there exists a constant B that doesn’_
_depend on t, x s.t._ _for all t ∈_ [0 _, T_ ] _,_


_B_
E _pt,x_ ( _z_ 1)[ _∥z_ 1 _∥_ [2] ] _≤_ (1 _−_ _T_ ) _[ν]_ [+] _[d]_ _[.]_


_In other words, we have that, for all T_ _∈_ (0 _,_ 1) _, there exists B_ 1 _, B_ 2 _independent of x, so that_


sup E _pt,x_ [ _∥z_ 1 _∥_ ] _≤_ _B_ 1 _, ∀x,_
_t∈_ [0 _,T_ ]

sup E _pt,x_ [ _∥z_ 1 _∥_ [2] ] _≤_ _B_ 2 _, ∀x._
_t∈_ [0 _,T_ ]


**Proof.** [Proof of Proposition G.1]


18


To derive the desired upper bound, we aim to upper bound _pt,x_ ( _z_ 1). We first derive a lower bound on
the normalizing constant:


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 _p_ ( _z_ 1) _dz_ 1


[1]
R _[d]_ [(1 +] _ν_


_ν_ + _d_

_[x]_ _t_ _[−]_ _[z]_ [1] _[∥]_ [2] [)] 2


( [1] _[−]_ _t_ _[t]_
R _[d]_ [(] ( [1] _[−][t]_ [)][2] [+] [1]


_ν_ + _d_

_[x]_ _t_ _[−]_ _[z]_ [1] _[∥]_ [2] [)] 2


( [1] _[−][t]_


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


 =


( [1] _[−][t]_


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


( ( [1] _[−]_ _t_ _[t]_
_∥z_ 1 _∥≤R_ 0 ( [1] _[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


+ _d_ 
2 _p_ ( _z_ 1) _dz_ 1 _≥_


2 _p_ ( _z_ 1) _dz_ 1


_[−][t]_

_t_ [)][2]

[1]

_ν_ _[∥]_ _[x]_ _t_


_[t]_

_t_ [)][2]

[1]

_ν_ _[∥]_ _[x]_ _t_


_ν_ + _d_

2

_[x]_ _t_ _[∥]_ + _R_ 0) [2] [)]


_≥_ ( ( [1] _[−]_ _t_ _[t]_


_ν_ + _d_

2

_[x]_ _t_ _[∥]_ + _R_ 0) [2] [)]


_[t]_

_t_ [)][2]


( [1] _[−][t]_


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


_[−][t]_

_t_ [)][2]


2


( [1] _[−]_ _t_ _[t]_

_[C]_ _[′]_ ) _≥_ (

_R_ 0 _[β]_ ( [1] _[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1] [(]


+ _d_

2 (1 _−_ _[C]_ _[′]_


+ _d_

2 [1]


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


( [1] _[−][t]_


[1] _[∥][x][∥]_

_ν_ [(] _t_


[1] _[∥][x][∥]_

_ν_ [(] _t_


[1] (1 _−_ _t_ ) [2]

2 [(] (1 _−_ _t_ ) [2] + [1] [(] _[∥][x]_


_≥_ [1]


(1 _−_ _t_ ) [2] + [1]


_ν_ + _d_

[1] [)] 2

_ν_ [(] _[∥][x][∥]_ [+] _[ tR]_ [0][)][2]


2 _._


We will split R _[d]_ into different regions, and derive upper bounds of _pt,x_ ( _z_ 1) for each of them.


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ _[∥]_ 2 _[x]_ _t_ _[∥]_


_[x][∥]_

2 _t_ [+] _[ R]_ [0][, which implies] _[ ∥][z]_ [1] _[∥≥]_ _[∥]_ 2 _[x]_ _t_ _[∥]_


1. Region 1 _∥_ _[x]_


_[∥]_ 2 _[x]_ _t_ _[∥]_ [+] _[ R]_ [0][.] [We remark that it suffices to consider] _[∥]_ 2 _[x]_ _t_ _[∥]_


Region 1 _∥_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ 2 _t_ [+] _[ R]_ [0][.] [We remark that it suffices to consider] 2 _t_ _[≥]_ [2] _[R]_ [0][ so that]

_∥_ _[x]_ _[∥−∥][z]_ [1] _[∥≤∥]_ _[x]_ _[−]_ _[z]_ [1] _[∥≤]_ _[∥][x][∥]_ [+] _[ R]_ [0][, which implies] _[ ∥][z]_ [1] _[∥≥]_ _[∥][x][∥]_ _[−]_ _[R]_ [0] _[≥]_ _[R]_ [0][.]


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ _[∥]_ 2 _[x]_ _t_ _[∥]_


2 _t_ _[−]_ _[R]_ [0] _[≥]_ _[R]_ [0][.]


_[x]_ _t_ _[∥−∥][z]_ [1] _[∥≤∥]_ _[x]_ _t_


Otherwise, if _[∥][x][∥]_


_[x]_ _t_ _[∥≤∥]_ _[x]_ _t_


_[x][∥]_

2 _t_ _<_ 2 _R_ 0, _∥z_ 1 _∥−∥_ _[x]_ _t_


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ _∥_ 2 _xt∥_


Otherwise, if 2 _t_ _<_ 2 _R_ 0, _∥z_ 1 _∥−∥_ _t_ _[∥≤∥]_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ 2 _t_ [+] _[R]_ [0][,] [i.e.,] [we] [have] [that]

_∥z_ 1 _∥≤_ 7 _R_ 0, so that - _[∥][x][∥]_ _[z]_ [1] _[∥]_ [2] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1] _[≤]_ [49] _[R]_ 0 [2] [which is of constant order.]


_xt_ [(] _[R]_ [=] _[∥]_ 2 _[x]_ _t_ _[∥]_


_B_ _x_


_[∥]_ 2 _[x]_ _t_ _[∥]_ [+] _[R]_ [0][)] _[ ∥][z]_ [1] _[∥]_ [2] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1] _[≤]_ [49] _[R]_ 0 [2] [which is of constant order.]


Now since _[∥]_ 2 _[x]_ _t_ _[∥]_ _[≥]_ [2] _[R]_ [0][, we alternatively have]


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 _p_ ( _z_ 1) _dz_ 1


[1]
R _[d]_ [(1 +] _ν_


[1] (1 _−_ _t_ ) [2]

2 [(] (1 _−_ _t_ ) [2] + [1] [(] _[∥][x]_


_≥_ [1]


(1 _−_ _t_ ) [2] + [1]


_ν_ + _d_

[1] [)] 2

_ν_ [(] _[∥][x][∥]_ [+] _[ tR]_ [0][)][2]


(1 _−_ _t_ ) [2] + [2]


_ν_ + _d_

[2] [)] 2

_ν_ _[∥][x][∥]_ [2]


[1] (1 _−_ _t_ ) [2]

2 [(] (1 _−_ _t_ ) [2] + [2]


+ _d_

2 _≥_ [1]


2 _._


Using


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


(1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 _p_ ( _z_ 1)


1
=


(1 + [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1


+ _d_

2 _[p]_ [(] _[z]_ [1][)] _[ ≤]_ (1 + [1] _[∥]_ _[x][−]_


2

+ _d_ _[C]_ _ν_ +2 _d_

2 [(] _∥z_ _∥_ _ν_


(1 + [1]


_[−][tz]_ [1] _ν_ +2 _d_

1 _−t_ _[∥]_ [2][)]


_[−][tz]_ [1] _ν_ +2 _d_

1 _−t_ _[∥]_ [2][)]


2

[2] _C_ _ν_ + _d_


2

_[x]_ _t_ _[−]_ _[z]_ [1] _[∥]_ [2][)] _∥z_ 1 _∥_ _ν_


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


1
=(
(1 + [1] _[∥]_


2
_C_ _ν_ + _d_
1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _∥z_ 1 _∥_ _ν_ 2+ _α_


(( [1] _[−][t]_


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


+2 _d_ = ( ( [1] _[−]_ _t_ _[t]_


_[t]_

_t_ [)][2]

[1]

_ν_ _[∥]_ _[x]_ _t_


2
_ν_ + _d_
_≤_ ( _[C]_


2 _,_


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


we obtain


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_pt,x_ ( _z_ 1) = (1 + _ν_ [1]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


(1 _−_ _t_ ) [2] + [2]


[2] [)] _[−]_ _[ν]_ [+] 2 _[d]_

_ν_ _[∥][x][∥]_ [2]


2


+ _d_ (1 _−_ _t_ ) [2]

2 (


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


2

_p_ ( _z_ 1) _ν_ + _d_

[+] _[d]_ _≤_ 2( _[C]_ 2

2 _p_ ( _z_ ) _dz_ _∥z_ 1 _∥_ _ν_ +


R _[d]_ [(1 +] [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


= 2( [(1] _[ −]_ _[t]_ [)][2] [+] _ν_ [2]


2

_[t]_ [)][2] [+] _ν_ [2] _[∥][x][∥]_ [2] _C_ _ν_ + _d_

2

(1 _−_ _t_ ) [2] _ν_ +


[2]

_ν_ _[∥][x][∥]_ [2]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2 _._


19


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ _[∥]_ 2 _[x]_ _t_ _[∥]_


When _∥_ _[x]_


_∥_ _t_ _[−]_ _[z]_ [1] _[∥≤]_ 2 _t_ [+] _[ R]_ [0][,]


_xt_ [(] _[R]_ [=] _[∥]_ 2 _[x]_ _t_ _[∥]_


_B_ _x_


_∥z_ 1 _∥_ [2] _pt,x_ ( _z_ 1) _dz_ 1

_[x][∥]_

2 _t_ [+] _[R]_ [0][)]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2 _dz_ 1


  _≤_ 2


_xt_ [(] _[R]_ [=] _[∥]_ 2 _[x]_ _t_ _[∥]_


[2]

_ν_ _[∥][x][∥]_ [2]


2

_[t]_ [)][2] [+] _ν_ [2] _[∥][x][∥]_ [2] _C_ _ν_ + _d_

2

(1 _−_ _t_ ) [2] _ν_


_B_ _x_


_∥z_ 1 _∥_ [2] ( [(1] _[ −]_ _[t]_ [)][2] [+] _ν_ [2]

_[∥]_ 2 _[x]_ _t_ _[∥]_ [+] _[R]_ [0][)] (1 _−_ _t_ )


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2


_[x]_ _t_ [(] _[R]_ [ =] _[∥][x][∥]_


_xt_ [(] _[R]_ [=] _[∥]_ 2 _[x]_ _t_ _[∥]_


_≤_ 2Vol( _B_ _[x]_


sup
2 _t_ [+] _[ R]_ [0][))]

_[∥][x][∥]_


[2]

_ν_ _[∥][x][∥]_ [2]


2

_[t]_ [)][2] [+] _ν_ [2] _[∥][x][∥]_ [2] _C_ _ν_ + _d_

2

(1 _−_ _t_ ) [2] _ν_


_B_ _x_


_∥z_ 1 _∥_ [2] ( [(1] _[ −]_ _[t]_ [)][2] [+] _ν_ [2]

(1 _−_ _t_ ) [2]

_[∥]_ 2 _[x]_ _t_ _[∥]_ [+] _[R]_ [0][)]


_ν_ + _d_

2

2 _α−_ 4

_ν_ + _d_ [)]


_≤_ 2 _CB_ ( _[∥][x][∥]_


sup
_t_ [)] _[d]_


(

_[∥][x][∥]_

4 _t_ [)]


_ν_ 3 _[∥][x][∥]_ [2] _C_ _ν_ +2 _d_

2 _α_

(1 _−_ _T_ ) [2] _ν_


3
_ν_ _[∥][x][∥]_ [2]


2


_xt_ [(] _[R]_ [=] [3] _[∥]_ 4 _[x]_ _t_ _[∥]_


2 _α−_ 4
_∥z_ 1 _∥_ _ν_ + _d_


_B_ _x_


[1]
_≤_ 2 _CB∥x∥_ _[d]_

_t_ _[d]_ [(]


[1]
_≤_ 2 _CB∥x∥_ _[d]_


_ν_ 3 _C_ _ν_ +2 _d_

2 _α_

(1 _−_ _T_ ) [2] _∥x∥_ _[−]_ [2] _[x]_ _ν_ +


_∥_ _[x]_


2 _α−_ 4
4 _[x]_ _t_ _[∥]_ _ν_ + _d_


2


_ν_ + _d_

2

2 _α−_ 4

_ν_ + _d_ [)]


[1]
_≤_ 2 _CB∥x∥_ _[d]_

_t_ _[d]_ [(]


[1]
_≤_ 2 _CB∥x∥_ _[d]_


2 _α−_ 4 _−_ 2 _ν−_ 2 _d_
(1 _−_ _T_ ) [2] _∥x∥_ _ν_ + _d_


_ν_ 3 _[C]_ _ν_ +2 _d_ (4 _t_ ) 2 _να_ + _−d_ 4


_ν_ + _d_

4 _−_ 2 _ν−_ 2 _d_ ) 2

_ν_ + _d_


_ν_ + _d_


2


+ _d_ 
2 _,_


1
= _∥x∥_ [2] _[d]_ [+] _[ν]_ [+2] _[−][α]_ _t_ _[α][−]_ [2] _[−][d]_
(1 _−_ _T_ ) _[ν]_ [+] _[d]_


2 _CB_ 4 _[α][−]_ [2] _C_ ( [3]


_ν_ + _d_

2

_ν_ [)]


where observe that [2] _[α][−]_ [4]


_ν_ + _d_ and


_[α][−]_ [4] [2] _[α][−]_ [4] _[−]_ [2] _[ν][−]_ [2] _[d]_

_ν_ + _d_ _[−]_ [2 =] _ν_ + _d_


1
_∥x∥_ _[d]_ (


+ _d_

2 = _∥x∥_ _[d][−]_ [(] _[α][−]_ [2] _[−][ν][−][d]_ [)] = _∥x∥_ [2] _[d]_ [+] _[ν]_ [+2] _[−][α]_ _._


_ν_ + _d_

4 _−_ 2 _ν−_ 2 _d_ ) 2

_ν_ + _d_


[4] _[−]_ [2] _[ν][−]_ [2] _[d]_ _ν_ + _d_

_ν_ + _d_ 2


2 _α−_ 4 _−_ 2 _ν−_ 2 _d_
_∥x∥_ _ν_ + _d_


+ _d_

2 = _∥x∥_ _[d]_ _∥x∥_ _[−]_ [2] _[α][−]_ [4] _ν_ _[−]_ + [2] _d_ _[ν][−]_ [2] _[d]_


To control the second moment so that it doesn’t explode with _∥x∥_, we need _α ≥_ 2 _d_ + _ν_ + 2.
2. Region 2 _∥_ _[x]_ _[−]_ _[z]_ [1] _[∥≥]_ [1] _[∥][x][∥]_ [+] _[ R]_ [0][ and] _[ ∥][z]_ [1] _[∥≥]_ [1]


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≥]_ 2 [1] _t_


Region 2 _∥_ _t_ _[−]_ _[z]_ [1] _[∥≥]_ 2 _t_ _[∥][x][∥]_ [+] _[ R]_ [0][ and] _[ ∥][z]_ [1] _[∥≥]_ [1]

For this case, we can have a sharper upper bound on _pt,x_ ( _z_ 1).


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


(1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 _p_ ( _z_ 1)


1
=


(1 + [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1


+ _d_

2 _[p]_ [(] _[z]_ [1][)] _[ ≤]_ (1 + [1] _[∥]_ _[x][−]_


2

+ _d_ _[C]_ _ν_ +2 _d_

2 [(] _∥z_ _∥_ _ν_ +


(1 + [1]


_[−][tz]_ [1] _ν_ +2 _d_

1 _−t_ _[∥]_ [2][)]


_[−][tz]_ [1] _ν_ +2 _d_

1 _−t_ _[∥]_ [2][)]


2

[2] _C_ _ν_ + _d_


2

_[x]_ _t_ _[−]_ _[z]_ [1] _[∥]_ [2][)] _∥z_ 1 _∥_ _ν_


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


1
=(
(1 + [1] _[∥]_


2
_C_ _ν_ + _d_
1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _∥z_ 1 _∥_ _ν_ 2


(( [1] _[−][t]_


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


+2 _d_ = ( ( [1] _[−]_ _t_ _[t]_


_[t]_

_t_ [)][2]

[1]

_ν_ _[∥]_ _[x]_ _t_


_≤_ ( ( [1] _[−]_ _t_ _[t]_


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


+ _d_ (1 _−_ _t_ ) [2]

2 = (


((1 _−_ _t_ ) [2] + [1]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2 _._


[1] [1]

_ν_ [(] 2


_[−]_ _t_ _[t]_ [)][2] [+] _ν_ [1]


_[−]_ _t_ _[t]_ [)][2] _C_ _ν_ +2 _d_

2

2 [1] _t_ _[∥][x][∥]_ [+] _[ R]_ [0][)][2][)] _∥z_ 1 _∥_ _ν_


_[t]_

_t_ [)][2]


2

_t_ ) [2] _C_ _ν_ + _d_


2

[1] 2 _[∥][x][∥]_ [+] _[ tR]_ [0][)][2][)] _∥z_ 1 _∥_ _ν_


(( [1] _[−][t]_


[1] [1]

_ν_ [(] 2 _t_


Hence


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_pt,x_ ( _z_ 1) = (1 + _ν_ [1]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2 _p_ ( _z_ ) _dz_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


R _[d]_ [(1 +] [1]


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


(1 _−_ _t_ ) [2]
_≤_ 2(


+ _d_ (1 _−_ _t_ ) [2]

2 (


(1 _−_ _t_ ) [2] + [1]


[1] [)] _[−]_ _[ν]_ [+] 2 _[d]_

_ν_ [(] _[∥][x][∥]_ [+] _[ tR]_ [0][)][2]


(1 _−_ _t_ ) [2] + [1]


2


[1] [1]

_ν_ [(] 2


= 2( [(1] _[ −]_ _[t]_ [)][2] [+] _ν_ [1]


2

_t_ ) [2] _C_ _ν_ + _d_


2

[1] 2 _[∥][x][∥]_ [+] _[ tR]_ [0][)][2] _∥z_ 1 _∥_ _ν_


2

_[∥][x][∥]_ [+] _[ tR]_ [0][)][2] _C_ _ν_ + _d_


2

[1] 2 _[∥][x][∥]_ [+] _[ tR]_ [0][)][2] _∥z_ 1 _∥_ _ν_


[1]

_ν_ [(] _[∥][x][∥]_ [+] _[ tR]_ [0][)][2]


_ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2
_C_ _ν_ + _d_ _ν_ + _d_

2

2 _α_
_∥z_ 1 _∥_ _ν_ + _d_ [)]


2 _._


(1 _−_ _t_ ) [2] + [1]


[1] [1]

_ν_ [(] 2


We see that for _∥_ _[x]_


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≥]_ 2 [1]


We see that for _∥_ _t_ _[−]_ _[z]_ [1] _[∥≥]_ 2 _t_ _[∥][x][∥]_ [+] _[ R]_ [0][,] _[ p][t,x]_ [(] _[z]_ [1][)][ has a polynomial tail bound that doesn’t]

depend on _x, t_ . Thus - _[x]_ [1] _[z]_ [1] _[∥]_ [2] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1] [can] [be] [bounded] [by]


depend on _x, t_ . Thus - _∥_ _[x]_ _t_ _[−][z]_ [1] _[∥≥]_ 2 [1] _t_ _[∥][x][∥]_ [+] _[R]_ [0] [and] _[ ∥][z]_ [1] _[∥≥]_ [1] _[ ∥][z]_ [1] _[∥]_ [2] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1] [can] [be] [bounded] [by]

some constant that doesn’t depend on _x, t_ :

   -   


_∥_ _[x]_


_[x]_ _t_ _[−][z]_ [1] _[∥≥]_ 2 [1]


_∥_ _[x]_


_[x]_ _t_ _[−][z]_ [1] _[∥≥]_ 2 [1]


                _∥z_ 1 _∥_ [2] _pt,x_ ( _z_ 1) _dz_ 1 _≤_ _C_ _[′]_
2 [1] _t_ _[∥][x][∥]_ [+] _[R]_ [0] [and] _[ ∥][z]_ [1] _[∥≥]_ [1]


_∥z_ 1 _∥≥_ 1


1
_∥z_ 1 _∥_ _[α][−]_ [2] _[dz]_ [1] _[.]_


20


The convergence of the integral is equivalent to the convergence of �1 _∞_ _r_ _[d][−][α]_ [+1] . When
_α ≥_ 2 _d_ + _ν_ + 2, it converges.


3. Region 3 _∥_ _[x]_


_[x]_ _t_ _[−]_ _[z]_ [1] _[∥≥]_ 2 [1] _t_


2 _t_ _[∥][x][∥]_ [+] _[ R]_ [0][ and] _[ ∥][z]_ [1] _[∥≤]_ [1][.]


We simply have 


_[x]_ _t_ _[−][z]_ [1] _[∥≥]_ 2 [1] _t_


_∥_ _[x]_


2 [1] _t_ _[∥][x][∥]_ [+] _[R]_ [0] [and] _[ ∥][z]_ [1] _[∥≤]_ [1] _[ ∥][z]_ [1] _[∥]_ [2] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1] _[≤]_ [1][.]


With the above Proposition, we can prove Lemma G.2 and G.3, which are the key ingredients in
proving the Lipschitzness of _v_ .


**Lemma G.2.** _Under Assumption 3, we have_


_[ d]_ 2 _[√]_ _ν_

_ν_ 1 _−_ _t_ [E] _[p][t]_ [(] _[z]_ [1] _[|][x]_ [)][[] _[∥][z]_ [1] _[∥]_ []] _[ ≤]_ _[ν]_ [ +] _ν_ _[ d]_


_∥∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] _∥≤_ _[ν]_ [ +] _[ d]_


_[ d]_ 2 _[√]_ _ν_

_ν_ 1 _−_ _T_ _[B]_ [1] _[,][ ∀][t][ ∈]_ [[0] _[, T]_ []] _[.]_


**Proof.** [Proof of Lemma G.2]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


_[d]_ _[z]_ [1][(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1) _dz_ 1

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]

[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] = _∇x_


R _[d]_ _[z]_ [1][(1 +] [1]


R _[d]_ [(1 +] [1]


2 _p_ ( _z_ ) _dz_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1]
R _[d][ z]_ [1] _[∇][x]_ 


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


 =


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


[+] _[d]_ _dz_ 1

2 _p_ ( _z_ ) _dz_


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


[+] _[d]_ �2 _[dz]_ 1

2 _p_ ( _z_ ) _dz_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_ - _T_ _p_ ( _z_ 1) 
2


 =

R _[d][ z]_ [1]


_∇x_ (1 + [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


( _z_ 1) �R _[d]_ [(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ ) _dz_

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_ �2


R _[d]_ [(1 +] [1]


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]
R _[d]_ [(1 +] _ν_


[+] 2 _[d]_ _p_ ( _z_ ) _dz_ - _T_ (1 + _ν_ [1]


 
_−_

R _[d][ z]_ [1]


_∇x_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_


2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_ �2 _[dz]_ 1 _[.]_

2 _p_ ( _z_ ) _dz_


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_

_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


Observe that


Hence


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


_∇x_ (1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


2


[+] 2 _[d]_ _−_ 1( _∇x_ 1

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[ +] _[ d]_

(1 + [1]
2 _ν_

[ +] _[ d]_

(1 + [1]
2 _ν_


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_

_[ −]_ _[tz]_ [1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


= _−_ _[ν]_ [ +] _[ d]_

2

= _−_ _[ν]_ [ +] _[ d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]

[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

_ν_ [(2] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

1 _−_ _t_ _[∥]_ [2][)]


[+] 2 _[d]_ _−_ 1 [1]


_[ −]_ _[tz]_ [1] 1

1 _−_ _t_ [)] 1 _−_ _t_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


= _−_ (1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_[−][tz]_ [1] [(] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]

1 _−t_ _[∥]_ [2]


_[ d]_ 1

_ν_ 1 + [1] _[∥]_ _[x]_


_[ −]_ _[tz]_ [1] 1

1 _−_ _t_ [)] 1 _−_ _t_ _[.]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


         _∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] = _−_

R _[d][ z]_ [1]


         _∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] = _−_


(1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_[−][tz]_ [1] [(] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]

1 _−t_ _[∥]_ [2]


- _T_


_[ d]_ 1

_ν_ 1 + [1] _[∥]_


_[ −]_ _[tz]_ [1] 1

1 _−_ _t_ [)] 1 _−_ _t_


[+] _[d]_ �2 _[dz]_ 1

2 _p_ ( _z_ ) _dz_


R _[d]_ [(1 +] [1]


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


_p_ ( _z_ 1) 


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


( _z_ 1) �R _[d]_ [(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ ) _dz_

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_ �2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


R _[d]_ [(1 +] [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_


2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


 +

R _[d][ z]_ [1]


 +


��


[1]
R _[d]_ [(1 +] _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ [(] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_

1 _−t_ _[∥]_ [2]


- _T_


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


[ +] _[ d]_ 1

_ν_ 1 + [1] _[∥]_


_[ −]_ _[tz]_ 1

1 _−_ _t_ [)] 1 _−_ _t_ _[p]_ [(] _[z]_ [)] _[dz]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


(1 + [1]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_ �2 _[dz]_ 1 _[.]_

2 _p_ ( _z_ ) _dz_


R _[d]_ [(1 +] [1]


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


21


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


Recall that we use the notation _pt,x_ ( _z_ 1) = (1+ _ν_ [1]


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1+ _ν_ _[∥]_ 1 _−t_ [1] _[∥]_ [)] 2 _p_ ( _z_ 1)

 - [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_

2 _p_ ( _z_ ) _dz_ [.]


R _[d]_ [(1+] _ν_ [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


_[−][tz]_ [1] [(] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]

1 _−t_ _[∥]_ [2]


- _T_


         _∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] = _−_

R _[d][ z]_ [1]


_ν_ + _d_


+ _d_ 1

_ν_ 1 + [1] _[∥]_ _[x]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_[ −]_ _[tz]_ [1] 1

1 _−_ _t_ [)] 1 _−_ _t_


_pt,x_ ( _z_ 1) _dz_ 1


_d_ 1

_ν_ 1 + [1] _[∥]_


 +

R _[d][ z]_ [1] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1]


��


_ν_ + _d_

R _[d]_ _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ [(] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_

1 _−t_ _[∥]_ [2]


- _T_


_[x][ −]_ _[tz]_ 1

1 _−_ _t_ [)] 1 _−_ _t_ _[p]_ [(] _[z][|][x]_ [)] _[dz]_


_._


In general, we have E[ _XY_ _[T]_ ] _−_ E[ _X_ ]E[ _Y_ ] _[T]_ = E[( _X_ _−_ E[ _X_ ])( _Y_ _−_ E[ _Y_ ]) _[T]_ ]. Let _X_ = _z_ 1 and
_Y_ = _[ν]_ [+] _ν_ _[d]_ 1+ _ν_ [1] _[∥]_ _[x]_ 11 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2] [(] _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] [)] 1 _−_ 1 _t_ [.] [To bound] _[ ∇][x]_ [E][[] _[Z]_ [1] _[|][Z][t]_ [=] _[ x]_ []][, we consider any unit vector] _[ v]_ [:]

_v_ _[T]_ _∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] _v_ = E[ _v_ _[T]_ ( _X_ _−_ E[ _X_ ]) _· v_ _[T]_ ( _Y_ _−_ E[ _Y_ ])] _≤_ E[ _∥X_ _−_ E[ _X_ ] _∥· ∥Y_ _−_ E[ _Y_ ] _∥_ ]
_≤_ E[( _∥X∥_ + _∥_ E[ _X_ ] _∥_ )( _∥Y ∥_ + _∥_ E[ _Y_ ] _∥_ )] _≤_ E[ _∥X∥∥Y ∥_ ] + 3E[ _∥X∥_ ]E[ _∥Y ∥_ ] _._

We have


_[−][tz]_ [1] [(] _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]

1 _−t_ _[∥]_ [2]


[+] _[d]_ 1

_ν_ 1+ [1] _[∥]_ _[x]_


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1 _[−]_ _−_ _[tz]_ _t_ [1] [)] 1 _−_ 1 _t_ [.] [To bound] _[ ∇][x]_ [E][[] _[Z]_ [1] _[|][Z][t]_ [=] _[ x]_ []][, we consider any unit vector] _[ v]_ [:]


_∥Y ∥_ = _ν_ + _d_ _∥x −_ _tz_ 1 _∥_
_ν_ (1 _−_ _t_ ) [2] 1 + [1] _[∥]_ _[x][−][tz]_ [1]


_tz_ 1 _∥_

_[−][tz]_ [1] [=] _[ν]_ [ +] _ν_ _[ d]_

1 _−t_ _[∥]_ [2]


[1] _[.]_

_ν_ _[∥][x][ −]_ _[tz]_ [1] _[∥]_ [2]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


[ +] _[ d]_ _∥x −_ _tz_ 1 _∥_

_ν_ (1 _−_ _t_ ) [2] + [1] _[∥][x]_


At (1 _−_ _t_ ) [2] = _ν_ [1] _[∥][x][ −]_ _[tz]_ [1] _[∥]_ [2][,] _[ ∥][Y][ ∥]_ [reach maximum,]


_√_
_ν_
2(1 _−_ _t_ ) _[.]_


sup _∥Y ∥_ = _[ν]_ [ +] _[ d]_
_z_ 1 _ν_


Therefore


2 _[√]_ _ν_

_∥∇x_ E[ _Z_ 1 _|Zt_ = _x_ ] _∥≤_ _[ν]_ [ +] _[ d]_

_ν_ 1 _−_ _t_ [E] _[p][t]_ [(] _[z]_ [1] _[|][x]_ [)][[] _[∥][z]_ [1] _[∥]_ []] _[.]_


**Lemma G.3.** _Under Assumption 3 with α ≥_ 2 _d_ + _ν_ + 2 _, we have_


_ν_ _[ d]_ 2(13 _−_ _[√]_ _νT_ ) [2] - _B_ 2 + 3 _B_ 1 [2] - _, ∀t ∈_ [0 _, T_ ] _._


_∥_ _[∂]_


_[∂]_ [=] _[ x]_ []] _[∥≤]_ _[ν]_ [ +] _[ d]_

_∂t_ [E][[] _[Z]_ [1] _[|][Z][t]_ _ν_


[ +] _[ d]_ 3 _[√]_ _ν_ �E[ _∥z_ 1 _∥_ [2] ] + 3E[ _∥z_ 1 _∥_ ] [2][�] _≤_ _[ν]_ [ +] _[ d]_

_ν_ 2(1 _−_ _t_ ) [2] _ν_


**Proof.** [Proof of Lemma G.3]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


_[z]_ [1][(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1) _dz_ 1

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_∂_ [=] _[ x]_ [] =] _[∂]_
_∂t_ [E][[] _[Z]_ [1] _[|][Z][t]_ _∂t_


R _[d]_ _[z]_ [1][(1 +] [1]


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


2 _p_ ( _z_ ) _dz_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1]
R _[d][ z]_ [1] _[∇][x]_ 


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


 =


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_ _dz_ 1

2 _p_ ( _z_ ) _dz_


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


[+] _[d]_ �2 _[dz]_ 1

2 _p_ ( _z_ ) _dz_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_ - _p_ ( _z_ 1) 
2


 =

R _[d][ z]_ [1]


- _∂_ [1]
_∂t_ [(1 +] _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


( _z_ 1) �R _[d]_ [(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ ) _dz_

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_ �2


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]
R _[d]_ [(1 +] _ν_


[+] 2 _[d]_ _p_ ( _z_ ) _dz_ - (1 + _ν_ [1]


 
_−_

R _[d][ z]_ [1]


 
_−_


- _∂_

_∂t_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_


2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


R _[d]_ [(1 +] [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_ �2 _[dz]_ 1 _[.]_

2 _p_ ( _z_ ) _dz_


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_

_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


Observe that
_∂_ [1]
_∂t_ [(1 +] _ν_


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


2


= _−_ _[ν]_ [ +] _[ d]_

2

= _−_ _[ν]_ [ +] _[ d]_


_[ d]_

(1 + [1]
2 _ν_

_[ d]_

(1 + [1]
2 _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]

[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_

_[ −]_ _[tz]_ [1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] 2 _[d]_ _−_ 1( [1]


[1]

1 _−_ _t_ _[∥]_ [2][)]


[1]

_ν_ [(2] _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1] _∂_

_ν_ _∂t_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


(1 _−_ _t_ ) [2]


[+] 2 _[d]_ _−_ 1 [1]


_[ −]_ _[tz]_ [1] _[x][ −]_ _[z]_ [1]

1 _−_ _t_ [)] _[T]_ (1 _−_ _t_ )


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1

_[−][tz]_ [1] (1 _−_ _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z]_ 1 _[T]_ _[x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z]_ [1] _[∥]_ [2][)] _[.]_

1 _−t_ _[∥]_ [2]


= _−_ (1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


_[ d]_ 1

_ν_ 1 + [1] _[∥]_


22


Hence


_∇x_ E[ _Z_ 1 _|Zt_ = _x_ ]


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


 =

R _[d][ z]_ [1]


 =


(1 + [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1

_[−][tz]_ [1] (1 _−_ _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z]_ 1 _[T]_ _[x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z]_ [1] _[∥]_ [2][)]

1 _−t_ _[∥]_ [2]


_[ d]_ 1

_ν_ 1 + [1] _[∥]_


[+] _[d]_ �2 _[dz]_ 1

2 _p_ ( _z_ ) _dz_


R _[d]_ [(1 +] [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


_p_ ( _z_ 1) 


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


( _z_ 1) �R _[d]_ [(1 +] _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ ) _dz_

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_ �2


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


R _[d]_ [(1 +] [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[ −]_ _−_ _[tz]_ _t_ [1]


[1]

2

1 _−_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


 
_−_

R _[d][ z]_ [1]


 
_−_


��


[1]
R _[d]_ [(1 +] _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1

_[−][tz]_ [1] (1 _−_ _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z]_ 1 _[T]_ _[x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z]_ [1] _[∥]_ [2][)] _[p]_ [(] _[z]_ [)] _[dz]_

1 _−t_ _[∥]_ [2]


[+] _[d]_

2 _[ν]_ [ +] _[ d]_


[ +] _[ d]_ 1

_ν_ 1 + [1] _[∥]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


(1 + [1]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1 + _ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1] _[∥]_ [2][)] _[−]_ 2 _p_ ( _z_ 1)

�� [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_ �2 _[dz]_ 1 _[.]_

2 _p_ ( _z_ ) _dz_


R _[d]_ [(1 +] [1]


_[x]_ 1 _[−]_ _−_ _[tz]_ _t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] 2 _[d]_


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


Define _pt,x_ ( _z_ 1) = (1+ _ν_ [1]


_[−][tz]_ [1] 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


(1+ _ν_ _[∥]_ 1 _−t_ [1] _[∥]_ [)] 2 _p_ ( _z_ 1)

- [1] _[x][−][tz]_ [2] _[−]_ _[ν]_ [+] _[d]_


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


[+] _[d]_

2 _p_ ( _z_ ) _dz_ [.]


R _[d]_ [(1+] _ν_ [1]


_[−][tz]_ 2

1 _−t_ _[∥]_ [2][)] _[−]_ _[ν]_ [+] _[d]_


_∂_

[=] _[ x]_ []]
_∂t_ [E][[] _[Z]_ [1] _[|][Z][t]_


1

_[−][tz]_ [1] (1 _−_ _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z]_ 1 _[T]_ _[x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z]_ [1] _[∥]_ [2][)]

1 _−t_ _[∥]_ [2]


 =

R _[d][ z]_ [1]


 =


_ν_ + _d_


_d_ 1

_ν_ 1 + [1] _[∥]_ _[x]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


_pt,x_ ( _z_ 1) _dz_ 1


_d_ 1

_ν_ 1 + [1] _[∥]_


 
_−_

R _[d][ z]_ [1] _[p][t,x]_ [(] _[z]_ [1][)] _[dz]_ [1]


��


_ν_ + _d_

R _[d]_ _ν_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_


1

_[−][tz]_ (1 _−_ _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z][T][ x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z][∥]_ [2][)] _[p]_ [(] _[z][|][x]_ [)] _[dz]_

1 _−t_ _[∥]_ [2]


_._


Define _X_ = _z_ 1, _Y_ = _[ν]_ [+] _[d]_


[+] _[d]_ 1

_ν_ 1+ [1] _[∥]_ _[x]_


_ν_ [1] _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1 _[−][tz]_ [1] (1 _−_ 1 _t_ ) [3] [(] _[∥][x][∥]_ [2] _[ −]_ _[z]_ 1 _[T]_ _[x]_ [(1 +] _[ t]_ [) +] _[ t][∥][z]_ [1] _[∥]_ [2][)][.] [Note that]

1 _−t_ _[∥]_ [2]


[ +] _[ d]_ 1

_ν_ 1 + [1] _[∥]_ _[x]_


_∥Y ∥_ = _∥_ _[ν]_ [ +] _[ d]_


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1

_[−][tz]_ [1] (1 _−_ _t_ ) [3] [(] _[x][ −]_ _[tz]_ [1][)] _[T]_ [ (] _[x][ −]_ _[z]_ [1][)] _[∥]_

1 _−t_ _[∥]_ [2]


_≤_ _ν_ + _d_ _∥x −_ _tz_ 1 _∥∥x −_ _z_ 1 _∥_
_ν_ (1 _−_ _t_ ) [3] 1 + [1] _[∥]_ _[x][−][tz]_ [1] _[∥]_ [2]


[ +] _[ d]_ 1 _∥x −_ _tz_ 1 _∥∥x −_ _z_ 1 _∥_

_ν_ 1 _−_ _t_ (1 _−_ _t_ ) [2] + [1] _[∥][x][ −]_ _[tz]_ [1]


_∥z_ 1 _∥._
_ν_


_∥∥_ [1] _x −_ _z_ 1 _∥_ _[≤]_ _[ν]_ [ +] _ν_ _[ d]_

_ν_ _[∥][x][ −]_ _[tz]_ [1] _[∥]_ [2]


[1]

_ν_ _[∥]_ _[x]_ 1 _[−]_ _−_ _[tz]_ _t_ [1]


1 + [1]


_x −_ _z_ 1 _∥_

_[−][tz]_ [1] = _[ν]_ [ +] _ν_ _[ d]_

1 _−t_ _[∥]_ [2]


where observe that if _∥z_ 1 _∥≤_ [1]


where observe that if _∥z_ 1 _∥≤_ [1] 2 _[∥][x][∥]_ [,] [we have] _[ ∥][x][ −]_ _[z]_ [1] _[∥≤]_ [2] _[∥][x][ −]_ _[tz]_ [1] _[∥]_ [.] [Then] _[ ∥][Y][ ∥≤]_ _[ν]_ _ν_ [+][2] _[d]_ 1 _−_ 1 _t_ [.] [If]

_∥z_ 1 _∥≥_ [1] _[∥][x][∥]_ [,]


[1] 2 _[∥][x][∥]_ [,] [we have] _[ ∥][x][ −]_ _[z]_ [1] _[∥≤]_ [2] _[∥][x][ −]_ _[tz]_ [1] _[∥]_ [.] [Then] _[ ∥][Y][ ∥≤]_ _[ν]_ _ν_ [+][2] _[d]_


2 _[∥][x][∥]_ [,]


_√_
_ν_ 1
2(1 _−_ _t_ ) 1 _−_ _t_ _[∥][x][ −]_ _[z]_ [1] _[∥≤]_ _[ν]_ [ +] _ν_ _[ d]_


_√_
_ν_


_ν_ 1

2 (1 _−_ _t_ ) [2] [(] _[∥][x][∥]_ [+] _[ ∥][z]_ [1] _[∥]_ [)] _[ ≤∥][z]_ [1] _[∥]_ _[ν]_ [ +] _ν_ _[ d]_


_[ d]_ 3 _[√]_ _ν_


_ν_ 2


_∥Y ∥≤_ _[ν]_ [ +] _[ d]_

_ν_


_∥Y ∥≤_ _[ν]_ [ +] _[ d]_


_ν_ 1

2 (1 _−_ _t_ ) [2] _[.]_


Recall: E[ _XY_ ] _−_ E[ _X_ ]E[ _Y_ ] = E[( _X_ _−_ E[ _X_ ])( _Y_ _−_ E[ _Y_ ])]. Therefore

_∥_ _[∂]_ [=] _[ x]_ []] _[∥≤]_ [E][[] _[v][T]_ [ (] _[X]_ _[−]_ [E][[] _[X]_ [])(] _[Y]_ _[−]_ [E][[] _[Y]_ [ ])]] _[ ≤]_ [E][[] _[∥][X]_ _[−]_ [E][[] _[X]_ []] _[∥· ∥][Y]_ _[−]_ [E][[] _[Y]_ [ ]] _[∥]_ []]

_∂t_ [E][[] _[Z]_ [1] _[|][Z][t]_


_≤_ E[( _∥X∥_ + _∥_ E[ _X_ ] _∥_ )( _∥Y ∥_ + _∥_ E[ _Y_ ] _∥_ )] _≤_ E[ _∥X∥∥Y ∥_ ] + 3E[ _∥X∥_ ]E[ _∥Y ∥_ ]


[ +] _[ d]_ 3 _[√]_ _ν_


_ν_ 2


_≤_ _[ν]_ [ +] _[ d]_


_ν_ 1 �E[ _∥z_ 1 _∥_ [2] ] + 3E[ _∥z_ 1 _∥_ ] [2][�] _._

2 (1 _−_ _t_ ) [2]


The following Lemma will be used when analyzing the discretization error.

**Lemma G.4.** _Under Assumption 3 with α_ _≥_ 2 _d_ + _ν_ + 2 _and Assumption 2, there exists D_ 3 _that_
_depends polynomially in_ 1 _−_ 1 _T_ _[, d, ν]_ _[and][ B]_ [1] _[, B]_ [2] _[,]_ [ E][[] _[∥][Z]_ [1] _[∥∥]_ [2][]] _[,]_ [ E][[] _[∥][Z]_ [0] _[∥∥]_ [2][]] _[ s.t.]_


E[ _∥v_ ( _Zt, t_ ) _−_ _v_ ( _Zti, ti_ ) _∥_ [2] ] _≤_ _h_ [2] _D_ 3 _._


23


**Proof.** [Proof of Lemma G.4]


By chain rule,


_d_ _[∂]_
_dt_ _[v]_ [(] _[Z][t][, t]_ [) =] _∂t_


_[∂]_ _[∂]_

_∂t_ _[v]_ [(] _[Z][t][, t]_ [) +] _∂x_


_[∂]_

_∂x_ _[v]_ [(] _[Z][t][, t]_ [)] _[ ◦]_ _[∂]_


_∂t_ _[Z][t][,]_


and therefore (note that _∂t_ _[∂]_ _[Z][t]_ [=] _[ v]_ [(] _[Z][t][, t]_ [)][)]


_[d]_

_dt_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥≤∥]_ _[∂]_


_∂x_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥· ∥][v]_ [(] _[Z][t][, t]_ [)] _[∥][.]_


_∥_ _[d]_


_[∂]_

_∂t_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥]_ [+] _[ ∥]_ _[∂]_


Recall


_∥_ _[∂]_


_[∂]_ 1 1 1 _ν_ + _d_

_∂t_ _[v]_ [(] _[x, t]_ [)] _[∥≤]_ (1 _−_ _T_ ) [2] _[∥][x][∥]_ [+] (1 _−_ _T_ ) [2] _[B]_ [1][ +] 1 _−_ _T_ _ν_


_[ d]_ 2 _[√]_ _ν_

_ν_ (1 _−_ _T_ ) [2] _[B]_ [1] _[,][ ∀][t][ ∈]_ [[0] _[, T]_ []] _[.]_


_ν_ _d_ 2(13 _−_ _[√]_ _νT_ ) [2] - _B_ 2 + 3 _B_ 1 [2] - _, ∀t ∈_ [0 _, T_ ]


1
_∥∇xv_ ( _x, t_ ) _∥≤_ _[ν]_ [ +] _[ d]_
1 _−_ _T_ [+] _ν_


and


1 1 1
_∥v_ ( _x, t_ ) _∥_ = _∥−_ [=] _[ x]_ []] _[∥≤]_ [=] _[ x]_ []] _[∥]_ [)]
1 _−_ _t_ _[x]_ [ +] 1 _−_ _t_ [E][[] _[Z]_ [1] _[|][Z][t]_ 1 _−_ _T_ [(] _[∥][x][∥]_ [+] _[ ∥]_ [E][[] _[Z]_ [1] _[|][Z][t]_

1 1
_≤_ [=] _[ x]_ []) =]
1 _−_ _T_ [(] _[∥][x][∥]_ [+][ E][[] _[∥][Z]_ [1] _[∥|][Z][t]_ 1 _−_ _T_ [(] _[∥][x][∥]_ [+] _[ B]_ [1][)] _[ .]_


Hence we have


_∥_ _[d]_


_[d]_ 1 1 1 _ν_ + _d_

_dt_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥≤]_ (1 _−_ _T_ ) [2] _[∥][Z][t][∥]_ [+] (1 _−_ _T_ ) [2] _[B]_ [1][ +] 1 _−_ _T_ _ν_


_[ d]_ 2 _[√]_ _ν_

_ν_ (1 _−_ _T_ ) [2] _[B]_ [1]


- 1

_·_
1 _−_ _T_ [(] _[∥][Z][t][∥]_ [+] _[ B]_ [1][)] _[ ∀][t][ ∈]_ [[0] _[, T]_ []] _[.]_


_ν_ _d_ 2(13 _−_ _[√]_ _νT_ ) [2] - _B_ 2 + 3 _B_ 1 [2] 


+ - 1 _[ν]_ [ +] _[ d]_
1 _−_ _T_ [+] _ν_


It follows that there exists _D_ 1 _, D_ 2 (that depends polynomially in 1 _−_ 1 _T_ _[, d, ν, B]_ [1] _[, B]_ [2][) s.t.]


_∥_ _[d]_ _[≤]_ _[D]_ [1] _[∥][Z][t][∥]_ [2][ +] _[ D]_ [2] _[,][ ∀][t][ ∈]_ [[0] _[, T]_ []] _[.]_

_dt_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥]_ [2]


Recall that Law( _Zt_ ) = Law( _tZ_ 1 + (1 _−_ _t_ ) _Z_ 0). Hence


E[ _∥Zt∥_ [2] ] = E[ _∥tZ_ 1 + (1 _−_ _t_ ) _Z_ 0 _∥_ [2] ] = _t_ [2] E[ _∥Z_ 1 _∥∥_ [2] ] + (1 _−_ _t_ ) [2] E[ _∥Z_ 0 _∥_ [2] ] + 2 _t_ (1 _−_ _t_ )E[ _Z_ 0 _[T]_ _[Z]_ [1][]]

_≤_ 2E[ _∥Z_ 1 _∥∥_ [2] ] + 2E[ _∥Z_ 0 _∥_ [2] ] _._


which implies there exists _D_ 3 (that depends polynomially in
1
1 _−T_ _[, d, ν, B]_ [1] _[, B]_ [2] _[,]_ [ E][[] _[∥][Z]_ [1] _[∥∥]_ [2][]] _[,]_ [ E][[] _[∥][Z]_ [0] _[∥∥]_ [2][]][) s.t.]


E[ _∥_ _[d]_

_dt_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥]_ [2][]] _[ ≤]_ _[D]_ [3] _[.]_


By Jensen’s inequality,


             - _t_
E[ _∥v_ ( _Zt, t_ ) _−_ _v_ ( _Zti, ti_ ) _∥_ [2] ] = E[ _∥_

_ti_


- _d_ - - _t_
_ds∥_ [2] ] _≤_ ( _t −_ _ti_ )E[
_ds_ _[v]_ [(] _[Z][s][, s]_ [)] _ti_


2

_d_

_ds_ ]

���� _ds_ _[v]_ [(] _[Z][s][, s]_ [)] ����


_≤_ _h_ [2] E[ _∥_ _[d]_

_dt_ _[v]_ [(] _[Z][t][, t]_ [)] _[∥]_ [2][]] _[ ≤]_ _[h]_ [2] _[D]_ [3] _[.]_


24


G.1 PROOF OF PROPOSITION 4.1


**Proof.** [Proof of Proposition 4.1] Using Lemma G.2,


1 1
_∥∇xv_ ( _z, t_ ) _∥_ = _∥−_ [+] [=] _[ x]_ []] _[∥]_
1 _−_ _t_ _[I]_ 1 _−_ _t_ _[∇][x]_ [E][[] _[Z]_ [1] _[|][Z][t]_


_[ d]_ 2 _[√]_ _ν_

_ν_ (1 _−_ _T_ ) [2] _[B]_ [1] _[,][ ∀][t][ ∈]_ [[0] _[, T]_ []] _[.]_


1
_≤_ _[ν]_ [ +] _[ d]_
1 _−_ _T_ [+] _ν_


Notice that


_∂_ _[∂]_ 1 1 [=] _[ z]_ [])]
_∂t_ _[v]_ [(] _[z, t]_ [) =] _∂t_ [(] _[−]_ 1 _−_ _t_ _[z]_ [ +] 1 _−_ _t_ [E][[] _[Z]_ [1] _[|][Z][t]_


1 1 1 _∂_
= _−_ (1 _−_ _t_ ) [2] _[z]_ [ +] (1 _−_ _t_ ) [2] [E][[] _[Z]_ [1] _[|][Z][t]_ [=] _[ z]_ [] +] 1 _−_ _t_ _∂t_ [E][[] _[Z]_ [1] _[|][Z][t]_ [=] _[ z]_ []] _[.]_


Using Lemma G.3, we have


_[∂]_ 1 1 1 _ν_ + _d_

_∂t_ _[v]_ [(] _[z, t]_ [)] _[∥≤]_ (1 _−_ _T_ ) [2] _[∥][z][∥]_ [+] (1 _−_ _T_ ) [2] _[B]_ [1][ +] 1 _−_ _T_ _ν_


_∥_ _[∂]_


_ν_ _d_ 2(13 _−_ _[√]_ _νT_ ) [2] - _B_ 2 + 3 _B_ 1 [2] - _, ∀t ∈_ [0 _, T_ ] _._


G.2 PROOF OF THEOREM 3


**Proof.** [Proof of Theorem 3]


Define


By direct computation,


_dZt_ = _v_ ( _Zt, t_ ) _dt, Z_ 0 _∼_ _π_ 0 _,_


_dY_ _t_ = _v_ ˆ( _Y_ _ti, ti_ ) _dt, Y_ 0 = _Z_ 0 _._


_d_

_[t][∥]_ [2] [= 2] _[⟨][Z][t][ −]_ _[Y]_ _[t][,]_ _[d]_
_dt_ _[∥][Z][t][ −]_ _[Y]_


_[d]_

_dt_ _[Z][t][ −]_ _[d]_


_[t][⟩]_ [= 2] _[⟨][Z][t][ −]_ _[Y]_ _[t][, v]_ [(] _[Z][t][, t]_ [)] _[ −]_ _[G]_ [ˆ][(] _[Y]_ _[t][i][, t][i]_ [)] _[⟩]_
_dt_ _[Y]_


= 2 _⟨Zt −_ _Y_ _t, v_ ( _Zt, t_ ) _−_ _v_ ( _Zti, ti_ ) _⟩_ + 2 _⟨Zt −_ _Y_ _t, v_ ( _Zti, ti_ ) _−_ _v_ ( _Y_ _ti, ti_ ) _⟩_

+ 2 _⟨Zt −_ _Y_ _t, v_ ( _Y_ _ti, ti_ ) _−_ _v_ ˆ( _Y_ _ti, ti_ ) _⟩._


Using Young’s inequality, we can bound the rest of the terms as follows.


1. We bound the first term. By Lemma G.4,


2E[ _⟨Zt −_ _Y_ _t, v_ ( _Zt, t_ ) _−_ _v_ ( _Zti, t_ ) _⟩_ ]

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + _L_ [1] 1 E[ _∥v_ ( _Zt, t_ ) _−_ _v_ ( _Zti, t_ ) _∥_ [2] ]

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + [1] _h_ [2] _D_ 3 _._

_L_ 1


2. We bound the second term.


2E[ _⟨Zt −_ _Y_ _t, v_ ( _Zti, ti_ ) _−_ _v_ ( _Y_ _ti, ti_ ) _⟩_ ]

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + _L_ [1] 1 E[ _∥v_ ( _Zti, ti_ ) _−_ _v_ ( _Y_ _ti, ti_ ) _∥_ [2] ]

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + [1] _L_ [2] 1 [E][[] _[∥][Z][t]_ _i_ _[−]_ _[Y]_ _[t]_ _i_ _[∥]_ [2][]] _[.]_

_L_ 1


Here we used Proposition 4.1.


25


3. We bound the third term. Recall that we assumed E[ _∥v_ ( _x, t_ ) _−_ _v_ ˆ( _x, t_ ) _∥_ [2] ] _≤_ _ε_ [2] . Then


2 _⟨Zt −_ _Y_ _t, v_ ( _Y_ _ti, ti_ ) _−_ _v_ ˆ( _Y_ _ti, ti_ ) _⟩_

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + _L_ [1] 1 E[ _∥v_ ( _Y_ _ti, ti_ ) _−_ _v_ ˆ( _Y_ _ti, ti_ ) _∥_ [2] ]

_≤L_ 1E[ _∥Zt −_ _Y_ _t∥_ [2] ] + [1] _ε_ [2] _._

_L_ 1


Together,
_d_ _[t][∥]_ [2][]] _[ ≤]_ [3] _[L]_ [1][E][[] _[∥][Z][t][ −]_ _[Y]_ _[t][∥]_ [2][] +] [1]
_dt_ [E][[] _[∥][Z][t][ −]_ _[Y]_ _L_ 1


- _h_ [2] _D_ 3 + _L_ [2] 1 [E][[] _[∥][Z][t]_ _i_ _[−]_ _[Y]_ _[t]_ _i_ _[∥]_ [2][] +] _[ ε]_ [2][�] _._


Define
_K_ = _h_ [2] _D_ 3 + _L_ [2] 1 [E][[] _[∥][Z][t]_ _i_ _[−]_ _[Y]_ _[t]_ _i_ _[∥]_ [2][] +] _[ ε]_ [2] _[.]_
Then
E[ _∥Zti_ +1 _−_ _Y_ _ti_ +1 _∥_ [2] ]


_≤e_ [3] _[L]_ [1] _[h]_ E[ _∥Zti_ _−_ _Y_ _ti∥_ [2] ] + _L_ [3] 1


- _ti_ +1

_e_ [3] _[L]_ [1][(] _[t][i]_ [+1] _[−][t]_ [)] ( _K_ ) _dt_
_ti_


_≤e_ [3] _[L]_ [1] _[h]_ E[ _∥Zti_ _−_ _Y_ _ti∥_ [2] ] + _[e]_ [3] _[L]_ [1] _L_ _[h]_ [2] 1 _[ −]_ [1] _K_


= _e_ [3] _[L]_ [1] _[h]_ E[ _∥Zti_ _−_ _Y_ _ti∥_ [2] ] + _[e]_ [3] _[L]_ [1] _L_ _[h]_ [2] 1 _[ −]_ [1] ( _h_ [2] _D_ 3 + _ε_ [2] ) + ( _e_ [3] _[L]_ [1] _[h]_ _−_ 1)E[ _∥Zti_ _−_ _Y_ _ti∥_ [2] ]


_≤_ (2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1)E[ _∥Zti_ _−_ _Y_ _ti∥_ [2] ] + _[e]_ [3] _[L]_ [1] _L_ _[h]_ [2] 1 _[ −]_ [1] ( _h_ [2] _D_ 3 + _ε_ [2] ) _._


For _Ai_ +1 _≤_ (2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1) _Ai_ + _[e]_ [3] _[L]_ _L_ [1] _[h]_ [2] 1 _[−]_ [1] _B_ with _A_ 0 = 0, we have


_[ −]_ [(2] _[e]_ [3] _[L]_ [1] _[h][ −]_ [1)] _[n]_ _e_ [3] _[L]_ [1] _[h]_ _−_ 1

1 _−_ (2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1) _L_ [2]


[1] _[h]_ _−_ 1

_B_ _≤_ [(2] _[e]_ [3] _[L]_ [1] _[h][ −]_ [1)] _[n][ −]_ [1]
_L_ [2] 1 2 _L_ [2] 1


_B._
2 _L_ [2] 1


[1] _[h][ −]_ [1]

_B_ = [1] _[ −]_ [(2] _[e]_ [3] _[L]_ [1] _[h][ −]_ [1)] _[n]_
_L_ [2] 1 1 _−_ (2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1)


_An_ =


_n−_ 1
�(2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1) _[i]_ [ 2] _[e]_ [3] _[L]_ [1] _[h][ −]_ [1]

_i_ =0 _L_ [2] 1


_n−_ 1


In general, for _x ∈_ [0 _,_ 1] we have _e_ _[x]_ _≤_ 1 + 2 _x_ . Hence 2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1 _≤_ _e_ [3] _[L]_ [1] _[h]_ + 6 _L_ 1 _h ≤_ 1 + 12 _L_ 1 _h_ .
And we get (2 _e_ [3] _[L]_ [1] _[h]_ _−_ 1) _[n]_ _≤_ (1 + 12 _L_ 1 _h_ ) _[n]_ _≤_ (1 + [12] _n_ _[L]_ [1] [)] _[n]_ _[≤]_ _[e]_ [12] _[L]_ [1][.]


Hence


This implies


Consequently,


E[ _∥ZT_ _−_ _Y_ _T ∥_ [2] ] _≤_ _[e]_ [12] _[L]_ [1] ( _h_ [2] _D_ 3 + _ε_ [2] ) _._

_L_ [2] 1


_W_ 2( _πT_ _[D][,]_ [ ˆ] _[π]_ _T_ _[D]_ [)] _[ ≤]_ _[e]_ [6] _[L]_ [1]

_L_ 1


~~�~~ _h_ [2] _D_ 3 + _ε_ [2] _._


_W_ 2( _π_ 1 _[D][,]_ [ ˆ] _[π]_ _T_ _[D]_ [)] _[ ≤]_ _[e]_ [6] _[L]_ [1]

_L_ 1


~~�~~ ~~�~~
_h_ [2] _D_ 3 + _ε_ [2] + (1 _−_ _T_ ) 2(E[ _∥Z_ 1 _∥_ [2] ] + E[ _∥Z_ 0 _∥_ [2] ]) _._


G.3 PROOF OF THEOREM 4


**Proof.** [Proof of Lemma 4.2] Note that we have

                      -                      _∥v_ _[P]_ ( _x_ 1 _, t_ ) _−_ _Px_ _[x]_ 2 [1] _[v][P]_ [ (] _[x]_ [2] _[, t]_ [)] _[∥]_ _g_ _[P]_ ( _x_ 1) [=] _[ ∥∇]_ [2][Ψ(] _[x]_ [1][)] _∇_ [2] Ψ _[∗]_ ( _z_ 1) _v_ _[D]_ ( _z_ 1 _, t_ ) _−_ _Px_ _[x]_ 2 [1] _[∇]_ [2][Ψ] _[∗]_ [(] _[z]_ [1][)] _[v][D]_ [(] _[z]_ [2] _[, t]_ [)] _∥gD_ ( _z_ 1)
= _∥v_ _[D]_ ( _z_ 1 _, t_ ) _−∇_ [2] Ψ( _x_ 1) _Px_ _[x]_ 2 [1] _[∇]_ [2][Ψ] _[∗]_ [(] _[z]_ [1][)] _[v][D]_ [(] _[z]_ [2] _[, t]_ [)] _[∥]_ _g_ _[D]_ ( _z_ 1)
= _∥v_ _[D]_ ( _z_ 1 _, t_ ) _−∇_ [2] Ψ( _x_ 1) _∇_ [2] Ψ _[∗]_ ( _z_ 1) _Pz_ _[z]_ 2 [1] _[v][D]_ [(] _[z]_ [2] _[, t]_ [)] _[∥]_ _g_ _[D]_ ( _z_ 1)
= _∥v_ _[D]_ ( _z_ 1 _, t_ ) _−_ _v_ _[D]_ ( _z_ 2 _, t_ ) _∥gD_ ( _z_ 1) _,_


26


where _Px_ _[y]_ [denotes parallel transport from] _[ x]_ [ to] _[ y]_ [.] [This proves the result.]


**Lemma G.5.** _Under Assumption 4, For κ_ _≤_ 2 _d_ + _γν_ +2 _[, we can guarantee Assumption 3 holds with]_
_α ≥_ 2 _d_ + _ν_ + 2 _._


**Proof.** Using the change of variable formula, together with the fact that the determinant of a matrix
equals to the product of all its eigenvalues, we know


_dπEuc_ _[P]_ [(] _[x]_ [) =] �det _∇_ [2] Ψ( _x_ ) _dπHess_ _[P]_ [(] _[x]_ [)] _[ ≥]_ _[dπ]_ _Hess_ _[P]_ [(] _[x]_ [)] _[,]_


where _πEuc_ _[P]_ _[, π]_ _Hess_ _[P]_ [denotes the probability density function of the target distribution in primal space,]
under the Euclidean metric and squared Hessian metric, respectively. Furthermore, the isometric
mapping from primal space to dual space guarantees that


_πEuc_ _[P]_ [(] _[x]_ [)] _[ ≥]_ _[π][D]_ [(] _[z]_ [)] _[.]_


Notice that

sup _∥∇_ Ψ( _x_ ) _∥≤_ _[C]_ _[′]_
_x∈Kδ_ _δ_ _[κ]_ _[.]_


Since we assumed sup _x∈K\Kδ πEuc_ _[P]_ [(] _[x]_ [)] _[ ≤]_ _[C][pdf]_ _[δ][γ]_ [, we have]

_π_ _[D]_ ( _z_ ) _≤_ _πEuc_ _[P]_ [(] _[x]_ [)] _[ ≤]_ _[C][pdf]_ _[δ][γ][,][ ∀][z]_ _[≥]_ _[C]_ _δ_ _[κ][′]_ _[.]_


Using _δ_ _[γ]_ = ( ( _δ_ 1 [1] ~~_[κ]_~~ [)] [)] _[γ/κ]_ [, we conclude that there exists some] _[ C]_ _[>]_ [ 0][ s.t.]


_C_
_π_ _[D]_ ( _z_ ) _≤_ _∥z∥_ _[γ/κ]_ _[,][ ∀][z]_ _[≥]_ [1] _[.]_

To guarantee _γ/κ ≥_ 2 _d_ + _ν_ + 2, we need _κ ≤_ 2 _d_ + _γν_ +2 [.]


**Proof.** [Proof of Theorem 4] Using Lemma G.5, we know Assumption 3 holds with _α ≥_ 2 _d_ + _ν_ + 2.
The result follows from applying Proposition 2.2 and Theorem 3.


27