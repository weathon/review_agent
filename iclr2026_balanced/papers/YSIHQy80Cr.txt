# EFFICIENT DIFFERENTIABLE CONTACT MODEL
## WITH LONG-RANGE INFLUENCE


**Xiaohan Ye** [1] **, Kui Wu** [2] **, Taku Komura** [1] **, Zherong Pan** [3]


1The University of Hong Kong 2LIGHTSPEED 3Meta


ABSTRACT


With the maturation of differentiable physics, its role in various downstream applications—such as model-predictive control, robotic design optimization, and
neural PDE solvers—has become increasingly important. However, the derivative information provided by differentiable simulators can exhibit abrupt changes
or vanish altogether, impeding the convergence of gradient-based optimizers. In
this work, we demonstrate that such erratic gradient behavior is closely tied to
the design of contact models. We further introduce a set of properties that a
contact model must satisfy to ensure well-behaved gradient information. Lastly,
we present a practical contact model for differentiable rigid-body simulators that
satisfies all of these properties while maintaining computational efficiency. Our
experiments show that, even from simple initializations, our contact model can
discover complex, contact-rich control signals, enabling the successful execution
of a range of downstream locomotion and manipulation tasks.


1 INTRODUCTION


Recent advancements in differentiable physical models (Werling et al., 2021; Huang et al., 2024)
have unlocked a range of downstream applications, including model-based reinforcement learning Xu et al. (2022), shooting-based controller optimization (Amos et al., 2018), and robot codesign (Xu et al., 2021). State-of-the-art models now extend to various material types, encompassing
both rigid and deformable bodies, while offering first-order gradient information. A notable advantage of differentiable physics is its ability to automatically discover contact-rich motions from trivial
initializations (Mordatch et al., 2012; Pang et al., 2023). Achieving this, however, requires an ideal
contact model that strikes a balance between accurately approximating physical contact mechanisms
and providing meaningful gradient information. Over the years, significant efforts have been made
to optimize this balance (Werling et al., 2021; Huang et al., 2024; Le Cleac’h et al., 2023). Despite
their significant progress, recent systematic analyses (Antonova et al., 2023; Suh et al., 2022a;b)
have highlighted several pitfalls in the gradient information provided by differentiable physics systems. While analytic gradients are beneficial across much of the objective landscape, they can exhibit rugged behavior when optimizing over non-smooth interactions and may vanish in nearly flat
regions. Consequently, optimizers are often prone to becoming trapped in undesirable local optima.


To address these challenges, existing techniques (Antonova et al., 2023; Li et al., 2022a) employ
global optimization algorithms, such as Bayesian optimization and optimal transportation, to escape local optima. While we agree that global search methods are crucial for complementing local
gradient-based optimization, we argue that many of the issues with poor gradient information can
be mitigated by improving the contact models within existing differentiable physics frameworks.
Specifically, both rugged and vanishing gradients stem from the contact model itself. When two
rigid objects come into contact, the abrupt introduction of contact forces results in rugged gradients.
Conversely, in the absence of contact, the lack of direct interaction leads to vanishing gradients.


In this paper, we make both theoretical and practical contributions, both aimed at enhancing the
gradient landscape of differentiable rigid body simulators. Theoretically, we introduce in Section 3
a set of properties that define a well-behaved contact model. These properties ensure that the contact model supports differentiation and can provably prevent inter-penetration, generate physically
plausible contact forces, and provide non-vanishing gradients even when objects are arbitrarily far


1


apart. The last property leads to long-range influence, allowing us to discover novel contact-rich
motions, even from a trivial initialization where objects are distant from each other. Practically, we
present a computationally tractable contact model in Section 4, which satisfies all these properties
as proved in our Appendix A.1 and is applicable to arbitrary articulated bodies represented using
triangle meshes. Further in Section 5, we significantly improve the computational efficiency for
evaluating the contact model using a Bounding Sphere Hierarchy (BSH). We have incorporated our
contact model into a full-featured rigid body simulator and experimented on a row of robotic manipulation and locomotion tasks. Our results show that our method can discover complex, contact-rich
control signals from trivial initialization, while previous models can get optimizers stuck at trivial
local minima or suffer from slow convergence.


2 RELATED WORK


The idea of differentiable physics originates from the pioneering work Todorov (2011), which is
then built into the MuJoCo simulator (Tassa et al., 2012), where gradients are computed via costly
finite difference schemes. Differentiable simulators are then proposed to use more efficient analytic
gradient information. Early works apply this idea to rigid body simulators (de Avila Belbute-Peres
et al., 2018) and reduced-order deformable body simulators (Pan & Manocha, 2018). The idea is
then adopted in other simulator models (Newbury et al., 2024) for elastic and plastic deformations,
articulated bodies, and fluid bodies, to name just a few. Since their invention, differentiable simulators have found many applications in computer graphics, robotics, and machine learning. Early
works along this line use differentiable simulators to perform model-predictive control (Tassa et al.,
2012) and guide deep policy search (Levine & Koltun, 2013). Differentiable simulators can be
combined with appearance models to perform state estimation (Ma et al., 2022) and system identification (Le Lidec et al., 2021). In computer graphics, animators use differentiable simulators to
inversely design initial and boundary conditions (Li et al., 2023; Stuyck & Chen, 2023; Du et al.,
2021). They can also provide gradient information for physics-informed machine learning such as
neural PDE solvers (Heiden et al., 2021) and neural motion planning (Toussaint et al., 2019).


Certain substeps in a simulation procedure are inherently non-differentiable, of which the most
important substep is contact handling. It is known that gradient information is lost for collision
detection with thin-shell-like objects (Harmon et al., 2009; Li et al., 2022a;b) and the sudden change
of contact forces in collision responses incur non-smoothness, which requires manually choosing a
specific direction in the Clark subdifferential (Werling et al., 2021). These factors can compromise
the gradient information, hindering downstream optimizers’ performance. In Suh et al. (2022a),
authors propose a mixed gradient-free and gradient-based optimizer to boost the performance of
policy search. On a parallel front, Li et al. (2020) showed that the contact model can be reformulated
to prevent gradient vanishing for thin-shell-like objects. Differentiable simulator (Huang et al.,
2024) built on top of this technique exhibits better robustness. However, we show that even with this
technique, optimizers can still suffer from vanishing gradients.


3 DIFFERENTIABLE PHYSICS WITH WELL-BEHAVED CONTACT MODEL


In this section, we first formulate the problem of a differentiable physical model, and then formalize
the properties of a well-behaved contact model that provides useful gradient information.


3.1 DIFFERENTIABLE PHYSICS MODEL


Throughout the paper, we consider articulated bodies geometrically represented using triangle
meshes. This is the representation adopted by a majority of differentiable contact models (Werling et al., 2021; Huang et al., 2024; Xu et al., 2021). Formally, we assume the configuration
of an articulated body is represented using a set of _V_ vertices located at _x_ 1 _,_ ⋯ _,V_ ∈ R [3], and we
use _x_ without subscript to denote the concatenated vertex vector _x_ ∈ R [3] _[V]_ . The vertices are
connected to form a set of _T_ triangles _t_ 1 _,_ ⋯ _,T_, with each cornering three vertices and defined as
_ti_ = { _i_ (1) _,i_ (2) _,i_ (3)} ⊂{1 _,_ ⋯ _,V_ }. The configuration space _x_ ∈C ⊂ R [3] _[V]_ can be divided into a
penetrating set Cobs = { _x_ ∈C∣∃ _ti_ ≠ _tj_ ∧ CH( _xi_ ( _k_ )∈ _ti_ ) ∩ CH( _xj_ ( _k_ )∈ _tj_ ) ≠∅}, where CH is the closed


2


convex hull of a set of vertices, and a penetration-free set Cfree = C/Cobs. The goal of a contact model
is to impose contact forces on the rigid object to ensure that _x_ ∈Cfree.

A discrete-time physical model can be cast as a time transition function _x_ _[t]_ [+][1] = _f_ ( _x_ _[t]_ _,x_ _[t]_ [−][1] _,δt_ ),
where we use superscript to denote the time index, _x_ _[t]_ is the kinematic state of a body at the _t_ th time
instance, and _δt_ is the timestep size. The concatenation of two kinematic states ⟨ _x_ _[t]_ _,x_ _[t]_ [−][1] ⟩ composes
a dynamic state of the body, with velocity approximated as ( _x_ _[t]_ - _x_ _[t]_ [−][1] )/ _δt_ . It is well-known (Marsden
& West, 2001; Gast et al., 2015) that the transition function can be cast as a numerical optimization:

_x_ _[t]_ [+][1] ∈ argmin _xt_ ⋆+1 [L(] _[x][t]_ ⋆ [+][1] _[,x][t][,x][t]_ [−][1] _[,δt]_ [)] _[,]_ (1)


which leads to stable performance of modern differentiable position-based dynamics, such as Huang
et al. (2024). In these position-based models, the Lagrangian function L contains various terms that
model different behaviors. Specifically, we define:

L( _x_ _[t]_ ⋆ [+][1] _[,x][t][,x][t]_ [−][1] _[,δt]_ [) = I(] _[x][t]_ ⋆ [+][1] _[,x][t][,x][t]_ [−][1] _[,δt]_ [) +] _[ µ]_ [P(] _[x][t]_ ⋆ [+][1][) + D(] _[x][t]_ ⋆ [+][1] _[,x][t][,δt]_ [)] _[,]_


where the term I models inertial acceleration, P models the contact potential weighted by a parameter _µ_, and D models frictional damping. We refer readers to Huang et al. (2024) for more details of
other terms and we focus on the term P in this paper. In optimization-based simulators, the contact
mechanics are mainly determined by the potential P, whose proper definition has been discussed,
e.g., in Fisher & Lin (2001); Harmon et al. (2009); Li et al. (2020); Ye et al. (2025).


3.2 PROPERTIES OF CONTACT POTENTIAL


We propose four aspects of indispensable properties of a well-behaved contact potential. **Barrier**
**potential:** It is well-known in the theory of continuous collision handling (Brochu et al., 2012) that a
penetrating state _x_ _[t]_ ∈Cobs corresponds to non-smooth landscape of the contact potential. Therefore,
a well-behaved contact potential should provably prevent any penetrations by ensuring that _x_ _[t]_ ∈Cfree
for any _t_ . This property is first proposed and achieved in Harmon et al. (2009), where authors
designed P to be a positive layered potential function that is infinite when _x_ _[t]_ ∈Cobs. As a result, a
numerical optimizer with globalization techniques, such as line search and trust-region, can ensure
the monotonic decrease of the Lagrangian L, leading to the finiteness of P and thus _x_ _[t]_ [+][1] ∈Cfree. Li
et al. (2020) further elucidates that a potential P should act as a log-barrier function in interior point
optimization, which is formalized as our first property below:

**Property 3.1** (Barrier-Form) **.** P( _x_ ) ≥ 0 _is continuous for any x_ ∈C _and_ P( _x_ ) = ∞ _iff x_ ∈C _obs._


Note that Barrier-Form does not describe the exact contact mechanics, because an exact contact
model can only impose contact forces on bodies when they are exactly touching, i.e., _x_ ∈ _∂_ Cobs, but
our contact potential induces the generalized contact force − _∂_ P/ _∂x_ even when _x_ ∉Cobs. To mitigate
this issue, Li et al. (2020) proposes to iteratively approximate the true contact mechanics by tuning
the coefficient _µ_ . Indeed, we can easily verify that lim _µ_ →0+ _µ_ P converges to the indicator function
that equals to ∞ if _x_ ∈ _∂_ Cobs and 0 otherwise. **Smoothness:** In order to utilize the primal log-barrier
method for computing _x_ _[t]_ [+][1] by solving optimization would require P to be at least differentiable.
To this end, Li et al. (2020) proposed a differentiable surrogate of the triangle-triangle distance
function. Unfortunately, although differentiability is enough for solving Equation 1, it is not enough
for providing reliable gradient information. Indeed, the gradient of a numerical optimization takes
the following form by the implicit function theorem:


_∂x_ _[t]_ [+][1]


_∂x_ _[t]_ [+][1] _[∂]_ [2][L]

_∂_ ( _x_ _[t]_ _, x_ _[t]_ [−][1] ) [= −[] _∂x_ _[t]_ [+]


−1
_∂_ [2] L
_∂x_ _[t]_ [+][1] _∂_ ( _x_ _[t]_ _, x_ _[t]_ [−][1] ) _[,]_


[]]
_∂x_ _[t]_ [+][12]


whose proper evaluation requires P to be twice-differentiable, which is not satisfied in Li et al.
(2020); Huang et al. (2024) as shown in our Appendix A.4, leading to our second property:

**Property 3.2** (Smoothness) **.** P _is twice differentiable at x_ ∈C _free._


Unfortunately, being numerically well-defined does not guarantee that the gradient information can
effectively guide the optimizer to find meaningful solutions for downstream applications. To this
end, we introduce two other properties that ensure the contact model is non-prehensile and nonvanishing. **Non-prehensile:** We know that a passive contact can only impose unilateral pushing


3


forces between a pair of contacting objects, instead of pulling objects together. Formally, we introduce a sufficient condition to ensure non-prehensile forces. Let us define two index subsets of
well-separated vertices I ∩J = ∅ and I ∪J ⊆{1 _,_ ⋯ _,V_ }, such that the convex hull of these sets
of vertices are non-overlapping, i.e. CH( _xi_ ∈I) ∩ CH( _xj_ ∈J ) = ∅. A contact potential between these
two subsets can be defined as a pair-wise contact term P [I∪J] ( _xi_ ∈I _,xj_ ∈J ) or P [I∪J] for short. Note
that we can also establish Barrier-Form and Smoothness for a pair-wise contact term P [I∪J] . To
this end, we define Cobs [I∪J] = { _x_ ∈C∣∃ _ti_ ≠ _tj_ ∧ _ti_ ∪ _tj_ ⊆I ∪J _,_ CH( _xi_ ′∈ _ti_ ) ∩ CH( _xj_ ′∈ _tj_ ) ≠∅} and
Cfree [I∪J] = C/Cobs [I∪J] and we have the following Barrier-Form and Smoothness for pairwise contact
terms:
**Definition 3.1** ( Barrier-Form and Smoothness for pairwise contact terms) **.** P [I∪J] _pertains_ _Barrier-_
_Form if_ P [I∪J] ≥ 0 _for any x_ ∈C _and_ P [I∪J] = ∞ _iff x_ ∈C _obs_ [I∪J] _[.]_ [P] [I∪J] _[pertains]_ _[Smoothness if it is]_
_twice differentiable at x_ ∈C _free_ [I∪J] _[.]_


P [I∪J] induces the following contact force on any _xi_ ∈I or _xj_ ∈J :


_∂_ _∂_
_fi_ [I∪J] ∈I = − P [I∪J] ( _xi_ ∈I _,xj_ ∈J ) _fj_ [I∪J] ∈J [= −] P [I∪J] ( _xi_ ∈I _,xj_ ∈J ) _._
_∂xi_ ∈I _∂xj_ ∈J

To allow only non-prehensile forces, we require that each _fi_ [I∪J] is pointing from CH( _xj_ ∈J ) to
CH( _xi_ ∈I) and vice versa. Formally, we define the set of non-zero vectors pointing from CH( _xj_ ∈J )
to CH( _xi_ ∈I) as FJ →I = { _α_ ( _a_ - _b_ )∣ _α_ - 0 ∧ _a_ ∈ CH( _xi_ ∈I) ∧ _b_ ∈ CH( _xj_ ∈J )}, and require that:

∀ _i_ ∈I ∶ _fi_ [I∪J] ∈FJ →I and ∀ _j_ ∈J ∶ _fj_ [I∪J] ∈FI→J _._ (2)


**Non-vanishing:** Our final property is of paramount importance and ensures that a differentiable
simulator provides non-zero gradient information at arbitrary configuration. This is ensured by
our definition of the non-prehensile force set FJ →I. Indeed, since CH( _xj_ ∈J ) and CH( _xi_ ∈I) are
disjoint, closed convex sets, for any _α_ ( _a_ - _b_ ) ∈FJ →I, we have _a_ ≠ _b_ and _α_ - 0, leading to _fi_ [I∪J] ≠ 0
for all _i_ ∈I. We further ensure that the contact forces between every pair of geometric primitives
(triangles) are taken into consideration. Put together, we can ensure both properties by requiring that
P is a summation of pairwise contact terms P [I∪J] between well-separated vertex clusters:

**Property** **3.3** (Non-prehensile & Non-vanishing) **.** _At_ _every_ _x_ ∈C _free,_ _we_ _can_ _define_ _a_ _finite_ _family_
_of_ _set_ _pairs_ A( _x_ ) = {⟨I _,_ J ⟩∣I ∩J = ∅∧I ∪J ⊆{1 _,_ ⋯ _,V_ } ∧ _CH_ ( _xi_ ∈I) ∩ _CH_ ( _xj_ ∈J ) = ∅} _._ _We_
_have_ P = ∑⟨I _,_ J ⟩∈A( _x_ ) P [I∪J] _such_ _that_ _every term_ P [I∪J] _satisfy Equation_ _2,_ _and for_ _every pair_ _of_
_triangles_ ⟨ _ti,tj_ ⟩ _on different rigid bodies, we have ti_ ∪ _tj_ ⊆I ∪J _for at least one_ ⟨I _,_ J ⟩∈A( _x_ ) _._


This is an important property that allows the gradient information to be provided for arbitrarily distant objects. In many applications, such gradient information can help a local optimizer discover
contact-rich motions from trivial initial guesses. Regretfully, we are not aware of any contact model
that pertains Barrier-Form, Smoothness, and Non-prehensile & Non-vanishing at the same time.
We summarize the failure cases for various properties in Figure 2 and compare the property completeness in Table 1. In Figure 1, we illustrate the main idea behind our contact model in Section 5
that satisfies Non-prehensile & Non-vanishing.


Figure 1: A 2D illustration of our contact force between a pair of hexagons satisfying Nonprehensile & Non-vanishing. Each hexagons have 6 line segments in 2D (resp. triangles in 3D).
Left: We compute the exact contact force (arrow) between each pair of nearby line segments. Middle: For faraway pairs of line segments, computing exact contact forces would involve too many
segment pairs, e.g. 4 pairs of forces between 2 edges on each hexagon. Right: Instead, we group
faraway segments and approximate the contact forces between centers of bounding circles in 2D
(resp. bounding spheres in 3D).


4


Barrier-Form Smoothness Non-prehensile Non-vanishing


Figure 2: We illustrate failure cases for various properties, where we assume two brown boxes are
separated by distance _x_ and plot the contact potential P( _x_ ) that pertains the property in green and
fails the property in red. Our Barrier-Form requires P to tend to infinity as _x_ → 0 [+] . Smoothness
requires P to have well-defined second-order derivatives. Non-prehensile requires the contact force
to always push the two boxes apart. Non-vanishing requires P and thus the contact force to be
non-vanishing for arbitrarily large _x_ .


Formulation Barrier-Form Smoothness Non-prehensile Non-vanishing


Turpin et al. (2022); Schwarke et al. (2025) ✗ ✗ ✓ ✓


Werling et al. (2021); Xu et al. (2022) ✗ ✗ ✓ ✗


Fisher & Lin (2001); Guendelman et al. (2003) ✗ ✗ ✓ ✗


Harmon et al. (2009); Li et al. (2020) ✓ ✗ ✓ ✗


Ye et al. (2025) ✓ ✓ ✓ ✗


Ours ✓ ✓ ✓ ✓


Table 1: Comparison of property completeness. Contact models based on complementary conditions Werling et al. (2021); Xu et al. (2022) or soft penalty functions Fisher & Lin (2001); Guendelman et al. (2003) cannot guarantee intersection-free or sufficient smoothness. Contact models based
on the log-barrier functions Harmon et al. (2009); Li et al. (2020) only have first-order derivatives,
which does not support differentiation using the inverse function theorem. Finally, prior contact
models have vanishing gradient when the distance is larger than a small margin. Although, Turpin
et al. (2022); Schwarke et al. (2025) provides non-vanishing gradients, they still cannot guarantee
intersection-free and smoothness, limited by penalty models.


4 WELL-BEHAVED CONTACT POTENTIAL


In this section, we propose a practical and well-behaved contact potential. We start by showing
that slightly modifying an existing contact potential (Liang et al., 2024; Ye et al., 2025) makes it
well-behaved, but such a potential is slow to compute. We then improve its computational efficacy
in Section 5 by borrowing ideas from the well-known hierarchical algorithm (Barnes & Hut, 1986)
for N-body simulation.


Barrier-Form requires that our contact potential acts as a primal barrier function. However, existing barrier potential functions (Harmon et al., 2009; Li et al., 2020) is derived from a modified
triangle-triangle distance function, denoted as _d_ ( _xi_ ( _k_ )∈ _ti,xj_ ( _k_ )∈ _tj_ ), which is then assembled to form
the following contact potential: P = ∑ _ti_ ≠ _tj P_ ( _d_ ( _xi_ ( _k_ )∈ _ti,xj_ ( _k_ )∈ _tj_ )), with _P_ being some locally
supported barrier function. Unfortunately, this potential is at most first-order differentiable and violates Smoothness and the local support of _P_ violates Non-vanishing. Instead, we propose to adopt
the more general contact potential (Liang et al., 2024; Ye et al., 2025) between a pair of convex hulls.
These methods use the separating hyperplane theorem to prevent two convex hulls from colliding
by inserting a separating hyperplane between them. This hyperplane is then modeled as a physical
object with zero-mass, which serves as auxiliary variables to formulate the contact potential. It has
been shown that such general contact potential is globally twice differentiable, which serves as a
good starting point for our derivation. Since a triangle is convex by nature, the more general contact
potential can be adopted to serve as P _[t][i]_ [∪] _[t][j]_ . Specifically, given a pair of triangles _ti_ and _tj_, since

_T_ 4
the two triangles are both convex sets, we can define a separating plane _pij_ = ( _n_ _[T]_ _ij_ _[, d][ij]_ [ )] ∈ R

between them if the two sets are disjoint, with _nij_ and _dij_ being the normal and offset, such that:
⟨ _xi_ ( _k_ )∈ _ti,nij_ ⟩+ _dij_ - 0 and ⟨ _xj_ ( _k_ )∈ _tj_ _,nij_ ⟩+ _dij_ < 0. As a result, we introduce the following potential


5


via a nested optimization:

P _[t][i]_ [∪] _[t][j]_ = min _pij_

[L] _[ij]_ [(] _[p][ij][, x][i]_ [(] _[k]_ [)∈] _[t][i]_ _[, x][j]_ [(] _[k]_ [)∈] _[t][j]_ [)]


1
(⟨ _xi_ ( _k_ ) _, nij_ ⟩+ _dij_ ) [+] [+]


1
(⟨ _xj_ ( _k_ ) _,_ - _nij_ ⟩− _dij_ ) [+] []] _[,]_


3
∑
_k_ =1


1
L _ij_ ( _pij, xi_ ( _k_ )∈ _ti_ _, xj_ ( _k_ )∈ _tj_ ) = [12 (1 −∥ _nij_ ∥) [+] [+]


3
∑
_k_ =1


where we purposefully introduce a constant coefficient 12 in the first term so that our follow-up
derivations take a simpler form, and other positive coefficients can be used as well. As a main
point of departure from the original formulation in Liang et al. (2024); Ye et al. (2025), we do not
use locally supported log-barrier function, but define the potential function 1/(●) [+] = 1/ max(● _,_ 0)
that has a global support on R [+] to prevent vanishing gradient. Note that _nij,dij_ are computed by
minimizing L _ij_ and thus _nij_ is not normalized. It is easy to verify that the objective function defined
in P _[t][i]_ [∪] _[t][j]_ is a strictly convex function with a unique minimizer, so that P _[t][i]_ [∪] _[t][j]_ is a well-defined
function. With the pair-wise potential defined, we can assemble them and define:

P = ∑ P _[t][i]_ [∪] _[t][j]_ ( _xi_ ( _k_ )∈ _ti,xj_ ( _k_ )∈ _tj_ ) _,_ (3)
_ti_ ≠ _tj_


where the summation is taken over triangle pairs on different rigid bodies. We now show that the
so-defined contact potential pertains all our desired properties.

**Lemma** **4.1.** _Each_ _pair-wise_ _potential_ P _[t][i]_ [∪] _[t][j]_ _in_ _Equation_ _3_ _pertains_ _Barrier-Form,_ _Smoothness,_
_and_ _satisfies_ _Equation_ _2,_ _so_ _that_ _the_ _potential_ P _pertains_ _Barrier-Form,_ _Smoothness,_ _and_ _Non-_
_prehensile & Non-vanishing._


At this point, we have shown that Equation 3 is a well-behaved contact potential function. Remarkably, this function is computationally practical. Indeed, each term P _[t][i]_ [∪] _[t][j]_ involves a small 4Doptimization problem with a strictly convex objective function, which can be solved efficiently using
Newton’s method to evaluate _pij_ . The first and second derivatives of P _[t][i]_ [∪] _[t][j]_ can then be computed
using the inverse function theorem. However, a brute force computation of the potential function
P is not efficient, since it involves terms that account for the contact potential between each pair of
disjoint triangles which increases in the square order of triangles.


5 EFFICIENT CONTACT POTENTIAL EVALUATION


The computational challenge of evaluating Equation 3 lies in accounting for the contact potentials
between all pairs of disjoint triangles. This scenario closely parallels the N-body simulation problem, where the forces between all pairs of particles must be computed. Instead of performing a
brute-force summation with a computational cost of _O_ ( _N_ [2] ), efficient algorithms such as the tree
code (Barnes & Hut, 1986) and the fast multipole expansion (Greengard & Rokhlin, 1987) achieve
costs of _O_ ( _N_ log( _N_ )) and _O_ ( _N_ ), respectively. These methods rely on the multipole expansion to
separate the influences of source particles from those of target particles. However, several factors
make these algorithms unsuitable for our case. First, the multipole expansions for our contact potential P _[t][i]_ [∪] _[t][j]_ remain undefined. Second, even if such expansions could be derived, the abrupt transition
between the exact potential and its multipole approximation could introduce discontinuities, thereby
violating Smoothness. Inspired by the fast multipole method (Greengard & Rokhlin, 1987), we
propose instead a modified potential that is also well-behaved and can be evaluated hierarchically.
Our main idea is to smoothly transit from the exact potential function P _[t][i]_ [∪] _[t][j]_ to simplified functions
that can be hierarchically evaluated.


5.1 SMOOTH TRANSITION BETWEEN POTENTIALS


Let us consider the pairwise potential between index set I and J . For an index set I, we define its
bounding sphere to be centered at _x_ I = ∑ _i_ ∈I _xi_ /∣I∣ with radius _R_ I ≥ max _i_ ∈I ∣ _x_ I - _xi_ ∣. Suppose
there are two versions of the potential denoted as P _d_ [I∪J] 1 and P _d_ [I∪J] 2, we can smoothly blend the two
functions when the distance between _x_ I and _x_ J grows from _d_ 1 to _d_ 2 with _d_ 1 < _d_ 2 as illustrated
in Figure 3, yielding the following blending potential:

P _d_ [I∪J] 1→ _d_ 2 [= (][1][ −] _[ϕ][d]_ [1][→] _[d]_ [2][(] _[x]_ [))P] _d_ [I∪J] 1 + _ϕd_ 1→ _d_ 2( _x_ )P _d_ [I∪J] 2 _,_ (4)


6


where we define the interpolation function as:


_ϕd_ 1→ _d_ 2( _x_ ) = Φ((∥ _x_ I − _x_ J ∥− _d_ 1)/( _d_ 2 − _d_ 1)) and Φ( _d_ ) = max(min(6 _d_ [5] - 15 _d_ [4] + 10 _d_ [3] _,_ 1) _,_ 0) _._


Similar to the tree code algorithm (Barnes & Hut, 1986), our goal of blending is to gradually replace
exact potential functions with faster-to-compute approximations. Such blending should not happen
when two sets of vertices are too close to each other. In practice, we only allow blending when
_R_ I + _R_ J ≤ _d_ 1 where _R_ I _,R_ J are the radii of the bounding spheres of I _,_ J . We first show the
well-behaved nature of potential functions is invariant to blending:

**Lemma 5.1.** _Taking the following assumptions:_ _i) R_ I + _R_ J ≤ _d_ 1 < _d_ 2 _; ii)_ P _d_ [I∪J] 1 _pertains_ _Barrier-_
_Form,_ _Smoothness,_ _and_ _satisfies_ _Equation_ _2;_ _iii)_ 0 ≤P _d_ [I∪J] 2 ≤P _d_ [I∪J] 1 _when_ ∥ _x_ I - _x_ J ∥≥ _d_ 1 _;_ _iv)_
P _d_ [I∪J] 2 _has_ _Smoothness, and satisfies Equation 2, then_ P _d_ [I∪J] 1→ _d_ 2 _[has the same properties as]_ [ P] _d_ [I∪J] 1 _._


Lemma 5.1 can be immediately used to blend our potential function P _[t][i]_ [∪] _[t][j]_ with a much simpler,
closed-form function. Consider moving all three vertices of _ti_ to the center point _xti_ = ( _xi_ (1)+ _xi_ (2)+
_xi_ (3))/3 and similarly moving _tj_ to _xtj_, then the potential P _[t][i]_ [∪] _[t][j]_ takes the following (centered) form
after some basic algebraic manipulation:


1 3
P _c_ _[t][i]_ [∪] _[t][j]_ = min _pij_ [[][12] (1 −∥ _nij_ ∥) [+] [+] _k_ ∑=1


1 3
(⟨ _xti,nij_ ⟩+ _dij_ ) [+] [+] _k_ ∑=1


2

1 1
(⟨ _xtj_ _,_ - _nij_ ⟩− _dij_ ) [+] [] =][ 12] [[][1][ +] ∥ _xti_ - _xtj_ ∥ [1][/][2] []] _._ (5)


We are now ready to apply Lemma 5.1 to blend P _[t][i]_ [∪] _[t][j]_ and P _c_ _[t][i]_ [∪] _[t][j]_ in a well-behaved manner:

**Corollary** **5.2.** _If_ _we_ _define_ P _d_ _[t][i]_ 1 [∪] _[t][j]_ = P _[t][i]_ [∪] _[t][j]_ _and_ P _d_ _[t][i]_ 2 [∪] _[t][j]_ = P _c_ _[t][i]_ [∪] _[t][j]_ _,_ _then_ P _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2 _[pertains]_ _[Barrier-]_
_Form,_ _Smoothness, and satisfies Equation 2._


Intuitively, Corollary 5.2 allows us to use the exact potential P _[t][i]_ [∪] _[t][j]_ when two triangles are very
close to each other, while switching to a simpler, closed-form potential P _c_ _[t][i]_ [∪] _[t][j]_ when the centers of
two triangles are well-separated by some distance _d_ 2.


5.2 HIERARCHICAL POTENTIAL BLENDING


Rti


ing techniques to smoothly replace the costly
contact potential P _[t][i]_ [∪] _[t][j]_ with a more computa
However, since this blending is applied only to

Figure 3: Illustration of our BSH-based contact

individual pairs of triangles, the approach still
requires summing over _O_ ( _T_ [2] ) terms. In this potential. When two triangles are nearby, we use
section, we fully unlock the potential of blend- the exact potential based on separating plane _pij_

(left). When the center of bounding sphere is sep
ing by hierarchically merging triangles to construct a BSH (Agarwal et al., 2004; Bradshaw arated by at least ( _Rti_ + _Rtj_ )(1 + _ϵ_ ), we use the

centered potential in Equation 7 (middle). These

& O’Sullivan, 2004) for each rigid body, and

two cases are combined by smooth blending. The

then smoothly replace the contact potential of

centered potential can be calculated hierarchically

each sphere with a single term. Specifically, we

for clusters of triangles (right).

adopt a layered hierarchy, where each sphere
tightly encapsulates all the bounding spheres of its two children. Another widely used option is
to use a wrapped hierarchy, where each sphere tightly encapsulates the actual geometry. Although
wrapped hierarchy achieves tighter bound, the layered hierarchy is required in our method to ensure
well-behaved properties. Our BSH is defined below:

**Definition** **5.3.** _A_ _BSH_ _is_ _a_ _binary_ _tree,_ _where_ _each_ _node_ _contains_ _an_ _index_ _subset_ _of_ _vertices_ I ⊆
{1 _,_ ⋯ _,N_ } _that is the union of the two subsets of its left and right children, denoted as_ I = I _l_ ∪I _r._
_The radius R_ I _is the smallest radius encapsulating the bounding spheres of two children._ _Each leaf_
_node stores a single triangle ti._ _Further, each node’s sphere is centered at x_ I _with radius R_ I _._


ε


Figure 3: Illustration of our BSH-based contact
potential. When two triangles are nearby, we use
the exact potential based on separating plane _pij_
(left). When the center of bounding sphere is separated by at least ( _Rti_ + _Rtj_ )(1 + _ϵ_ ), we use the
centered potential in Equation 7 (middle). These
two cases are combined by smooth blending. The
centered potential can be calculated hierarchically
for clusters of triangles (right).


Throughout the paper, we use the associated index subset I to refer to a BSH node. There are
many ways to practically construct our BSH, mostly using a greedy algorithm to iteratively merge
nodes, and we adopt the technique of Bradshaw & O’Sullivan (2004). Given the BSH, we propose
a recursive definition of our contact potential PBSH [I∪J] [,] [one for each node pair][ I] [and][ J] [on two rigid]


7


bodies, and use the potential of the root nodes as our final contact potential. Finally, we show that
our definition pertains Barrier-Form, Smoothness, and Non-prehensile & Non-vanishing. We start
from the base case. For a pair of leaf nodes _ti_ ∪ _tj_, we use Corollary 5.2 to define the following
pairwise potential:


_d_ 1 = _Rti_ + _Rtj_ and _d_ 2 = (1 + _ϵ_ ) _d_ 1
P _d_ _[t][i]_ 1 [∪] _[t][j]_ = P _[t][i]_ [∪] _[t][j]_ and P _d_ _[t][i]_ 2 [∪] _[t][j]_ = P _c_ _[t][i]_ [∪] _[t][j]_ _._ (6)
P _[t][i]_ [∪] _[t][j]_

⎧⎪⎪⎪⎪⎨⎪⎪⎪⎪⎩ BSH [= P] _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2


Specifically, we blend the exact potential between the pair of triangles and the closed-form centered
potential, when the distance between triangle centers grows by a factor of _ϵ_ . We leave _ϵ_ as a userdefined margin that controls the exactness of potential evaluation. Next, given an arbitrary internal
node with two child nodes being I and J, we recursively replace the more accurate potential between child nodes with a single potential between parent nodes. Let us suppose I ∩J = ∅, we
define a potential of similar form as Equation 5:


_d_ 1 = _Rti_ + _Rtj_ and _d_ 2 = (1 + _ϵ_ ) _d_ 1
P _d_ _[t][i]_ 1 [∪] _[t][j]_ = P _[t][i]_ [∪] _[t][j]_ and P _d_ _[t][i]_ 2 [∪] _[t][j]_ = P _c_ _[t][i]_ [∪] _[t][j]_
P _[t][i]_ [∪] _[t][j]_
BSH [= P] _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2


_._ (6)


√
P _c_ [I∩J] = 12 [1 + 1/


2
∥ _x_ I − _x_ J ∥] _,_ (7)


which is the potential penalizing distance between two sphere centers. We can only use the centered
potential when the two spheres are well-separated, i.e. _R_ I + _R_ J ≤∥ _x_ I − _x_ J ∥. Otherwise, we have
to use the more accurate potential by descending the tree and sum up the pair-wise terms between
each pair of child nodes. Specifically, we define the set of child nodes as _C_ (I) = {I _l,_ I _r_ } if I is an
internal node and _C_ (I) = {I} if I is a leaf node. Finally, we define the following potential between
internal node:


_d_ 1 = _R_ I + _R_ J and _d_ 2 = (1 + _ϵ_ ) _d_ 1
P _d_ [I∪J] 1 = ∑I _c_ ∈ _C_ (I) ∑J _c_ ∈ _C_ (J ) PBSH [I] _[c]_ [∪J] _[c]_ and P _d_ [I∪J] 2 = P _c_ [I∪J] _._ (8)
P [I∪J]

⎧⎪⎪⎪⎪⎨⎪⎪⎪⎪⎩ BSH [= P] _d_ [I∪J] 1→ _d_ 2


The main idea behind our formulation is illustrated in Figure 3 and we are ready to present our main
result, which shows that the so-defined contact potential is well-behaved:

**Theorem** **5.4.** _If_ _ϵ_ - 0 _then_ P = ∑I≠J P _BSH_ [I∪J] _pertains_ _Barrier-Form,_ _Smoothness,_ _and_ _Non-_
_prehensile_ _&_ _Non-vanishing,_ _where_ _the_ _summation_ _is_ _taken_ _over_ _the_ _root_ _nodes_ _of_ _different_ _rigid_
_bodies._


This result lays the foundation for our efficient-to-evaluate and well-behaved contact model. Although analyzing the cost of evaluating P could be rather difficult for general cases, we follow the
idea of fast multiple expansion (Greengard & Rokhlin, 1987) and analyze the cost of evaluating the
contact potential for a uniform grid, where we show in Appendix A.2 that the cost is _O_ ( _T_ ). Finally,
we notice that a contact model should account for frictional contact forces. In Appendix A.3, we
show that the frictional damping potential proposed in Ye et al. (2025) can be slightly extended to
ours.


6 EVALUATION


_d_ 1 = _R_ I + _R_ J and _d_ 2 = (1 + _ϵ_ ) _d_ 1
P _d_ [I∪J] 1 = ∑I _c_ ∈ _C_ (I) ∑J _c_ ∈ _C_ (J ) PBSH [I] _[c]_ [∪J] _[c]_ and P _d_ [I∪J] 2 = P _c_ [I∪J]
P [I∪J]
BSH [= P] _d_ [I∪J] 1→ _d_ 2


_._ (8)


We evaluated our method in a row of five contact-rich
manipulation and locomotion tasks: Billiards, Push, Sort,
Ant-Push and Gather. We optimize the sequence of control signals using gradient descent at a fixed learning rate
to minimize user-defined loss functions. More experiments and details are deferred to Appendix A.6. For fairness, we compare our contact model with the standard
IPC model used in Li et al. (2020); Huang et al. (2024),
and SDRS contact model proposed by Ye et al. (2025),
which only violates Non-vanishing. We also compare
with MuJoCo simulator (Tassa et al., 2012), which uses
soft contact and provides gradient by finite-difference
schemes.Finally, we compare with Suh et al. (2022b) that
uses first-order bundled gradient.


8


**Physics** **Accuracy:** Due to our method generating contact forces between objects that are not in
direct contact, we need to validate the physical accuracy of our contact model under different contact coefficients _µ_ . We validated the contact model’s physical accuracy through the book stacking problem (Hall, 2005) in Figure 4. In this scenario, we stacked 20 planks with dimensions of
2 _._ 0 _m_ × 0 _._ 4 _m_ × 0 _._ 2 _m_ and mass of 0.16kg sequentially, extending each plank outward to the maximum theoretical distance from the bottom plank without collapsing. To account for simulation
errors, they were shifted inward by 0.1% of the plank’s length. We verified whether the system
could remain stable under different _µ_ . Subsequently, we measured the margin between each plank
under different _µ_ and calculated the error between the contact force the top plank received from
adjacent planks and the theoretical value. The results, shown in Table 2, indicate that the system
remains stable when _µ_ < 1 _e_ [−][6], with margin errors in the millimeter range, and the errors between
contact forces and theoretical values are negligible.


Contact Coefficient _µ_ 1 _e_ [−][5] 1 _e_ [−][6] 1 _e_ [−][7] 1 _e_ [−][8] 1 _e_ [−][10]

Margin (m) 1 _._ 82 _e_ [−][2] 5 _._ 67 _e_ [−][3] 1 _._ 47 _e_ [−][3] 4 _._ 12 _e_ [−][4] 3 _._ 13 _e_ [−][5]

Contact force (N) 6 _._ 38 _e_ [−][3] 6 _._ 40 _e_ [−][4] 6 _._ 42 _e_ [−][5] 6 _._ 42 _e_ [−][6] 6 _._ 42 _e_ [−][8]
Success ✗ ✓ ✓ ✓ ✓


Table 2: The relationship between the margin between the planks, the error in the contact force


#Epoches


1000

800

600

400

200

0


#Epoches


**Billiards:** In this benchmark, we have 16 balls on the ground with one target red ball whose initial
horizontal position and velocity can be controlled. The goal of control is for the two green balls to
reach the target positions (green circle), where the loss function is the squared distance between
the green balls and the center of green circles. We experiment with two different methods for
setting initial solutions. Our first method uses trivial initialization where certain rigid objects in
a scenario are far apart, for which the gradient might vanish except our method. Our second method
uses random sampling of control signals to find an initial solution for which gradient information
does not vanish. The convergence history of various contact models and initialization strategies is
summarized in Figure 5. We optimize a trajectory with 100 timesteps at a timestep size of 0 _._ 04.
Except our method, other methods cannot make any progress without sampling due to gradient
vanishing, which can be fixed via sampling, while our method achieves faster convergence, with


9


7.5

5.0

2.5

0.0

2.5

5.0


sition and orientation of a rod to push the red 2.5

0.0

box to reach a target red circle, where the loss 2.5
is the squared distance between the box and 5.0


#Steps


6


4


2


0


#Steps


15


10


5


0


#Epoches


robot to push a box to the target position. Once again, our method significantly outperforms other
methods, as illustrated int Figure 9.


7 CONCLUSION


We present a detailed analysis of the qualifications for a contact model to be well-behaved, which
strictly prevents collisions, supports differentiable simulations, induce non-prehensile forces, and
avoids vanishing gradients. By hierarchically evaluating the contact potentials assisted by a BSH,
we further present a well-behaved contact model that is also efficient to evaluate. By analysis on
the special case of a uniform grid, we show that the complexity of evaluating our contact potential
is linear. Through evaluations on various motion planning and control tasks, we highlight that our
model can guide a gradient-based optimizer to search for complex motion plans and locomotion
gaits that are impossible for previous contact models. Our method is not without its problems. First,
we can only handle rigid bodies, and we cannot deal with more general deformable objects for
soft robot locomotion or soft object manipulation. This is because our bounding spheres might not
bound the actual triangles if deformation happens, which could potentially violate Non-prehensile
& Non-vanishing. Second, our contact potential involves a recursive definition and requires a nested
optimization between pairs of triangles, which incurs considerable overhead to a conventional rigid
body simulator.


10


REFERENCES


Pankaj Agarwal, Leonidas Guibas, An Nguyen, Daniel Russel, and Li Zhang. Collision detection
for deforming necklaces. _Computational Geometry_, 28(2-3):137–163, 2004.


Brandon Amos, Ivan Jimenez, Jacob Sacks, Byron Boots, and J Zico Kolter. Differentiable mpc for
end-to-end planning and control. _Advances in neural information processing systems_, 31, 2018.


Rika Antonova, Jingyun Yang, Krishna Murthy Jatavallabhula, and Jeannette Bohg. Rethinking
optimization with differentiable simulation from a global perspective. In _Conference_ _on_ _Robot_
_Learning_, pp. 276–286. PMLR, 2023.


Josh Barnes and Piet Hut. A hierarchical o (n log n) force-calculation algorithm. _nature_, 324(6096):
446–449, 1986.


Gareth Bradshaw and Carol O’Sullivan. Adaptive medial-axis approximation for sphere-tree construction. _ACM Transactions on Graphics (TOG)_, 23(1):1–26, 2004.


Tyson Brochu, Essex Edwards, and Robert Bridson. Efficient geometrically exact continuous collision detection. _ACM Transactions on Graphics (TOG)_, 31(4):1–7, 2012.


Filipe de Avila Belbute-Peres, Kevin Smith, Kelsey Allen, Josh Tenenbaum, and J Zico Kolter. Endto-end differentiable physics for learning and control. _Advances in neural information processing_
_systems_, 31, 2018.


Asen L Dontchev and R Tyrrell Rockafellar. _Implicit functions and solution mappings_, volume 543.
Springer, 2009.


Tao Du, Kui Wu, Pingchuan Ma, Sebastien Wah, Andrew Spielberg, Daniela Rus, and Wojciech
Matusik. Diffpd: Differentiable projective dynamics. _ACM_ _Trans._ _Graph._, 41(2), November
2021. ISSN 0730-0301.


Susan Fisher and Ming C Lin. Deformed distance fields for simulation of non-penetrating flexible
bodies. In _Computer Animation and Simulation 2001: Proceedings of the Eurographics Workshop_
_in Manchester, UK, September 2–3, 2001_, pp. 99–111. Springer, 2001.


Theodore F Gast, Craig Schroeder, Alexey Stomakhin, Chenfanfu Jiang, and Joseph M Teran. Optimization integrator for large time steps. _IEEE transactions on visualization and computer graph-_
_ics_, 21(10):1103–1115, 2015.


Leslie Greengard and Vladimir Rokhlin. A fast algorithm for particle simulations. _Journal_ _of_
_computational physics_, 73(2):325–348, 1987.


Eran Guendelman, Robert Bridson, and Ronald Fedkiw. Nonconvex rigid bodies with stacking.
_ACM transactions on graphics (TOG)_, 22(3):871–878, 2003.


John F Hall. Fun with stacking blocks. _American journal of physics_, 73(12):1107–1116, 2005.


David Harmon, Etienne Vouga, Breannan Smith, Rasmus Tamstorf, and Eitan Grinspun. Asynchronous contact mechanics. _ACM_ _Trans._ _Graph._, 28(3), July 2009. ISSN 0730-0301. doi:
10.1145/1531326.1531393. [URL https://doi.org/10.1145/1531326.1531393.](https://doi.org/10.1145/1531326.1531393)


Eric Heiden, David Millard, Erwin Coumans, Yizhou Sheng, and Gaurav S Sukhatme. Neuralsim:
Augmenting differentiable simulators with neural networks. In _2021 IEEE International Confer-_
_ence on Robotics and Automation (ICRA)_, pp. 9474–9481. IEEE, 2021.


Yuanming Hu, Jiancheng Liu, Andrew Spielberg, Joshua B Tenenbaum, William T Freeman, Jiajun
Wu, Daniela Rus, and Wojciech Matusik. Chainqueen: A real-time differentiable physical simulator for soft robotics. In _2019 International conference on robotics and automation (ICRA)_, pp.
6265–6271. IEEE, 2019.


Zizhou Huang, Davi Colli Tozoni, Arvi Gjoka, Zachary Ferguson, Teseo Schneider, Daniele
Panozzo, and Denis Zorin. Differentiable solver for time-dependent deformation problems with
contact. _ACM Transactions on Graphics_, 43(3):1–30, 2024.


11


Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv_ _preprint_
_arXiv:1412.6980_, 2014.


Simon Le Cleac’h, Mac Schwager, Zachary Manchester, Vikas Sindhwani, Pete Florence, and
Sumeet Singh. Single-level differentiable contact simulation. _IEEE_ _Robotics_ _and_ _Automation_
_Letters_, 8(7):4012–4019, 2023. doi: 10.1109/LRA.2023.3268824.


Quentin Le Lidec, Igor Kalevatykh, Ivan Laptev, Cordelia Schmid, and Justin Carpentier. Differentiable simulation for physical system identification. _IEEE Robotics and Automation Letters_, 6(2):
3413–3420, 2021.


Sergey Levine and Vladlen Koltun. Guided policy search. In _International conference on machine_
_learning_, pp. 1–9. PMLR, 2013.


Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy R Langlois, Denis Zorin, Daniele
Panozzo, Chenfanfu Jiang, and Danny M Kaufman. Incremental potential contact: intersectionand inversion-free, large-deformation dynamics. _ACM Trans. Graph._, 39(4):49, 2020.


Sizhe Li, Zhiao Huang, Tao Du, Hao Su, Joshua Tenenbaum, and Chuang Gan. Contact Points
Discovery for Soft-Body Manipulations with Differentiable Physics. In _International Conference_
_on Learning Representations (ICLR)_, 2022a.


Yifei Li, Tao Du, Kui Wu, Jie Xu, and Wojciech Matusik. Diffcloth: Differentiable cloth simulation
with dry frictional contact. _ACM Trans. Graph._, 42(1), October 2022b. ISSN 0730-0301.


Zhehao Li, Qingyu Xu, Xiaohan Ye, Bo Ren, and Ligang Liu. Difffr: Differentiable sph-based
fluid-rigid coupling for rigid body control. _ACM Transactions on Graphics (TOG)_, 42(6):1–17,
2023.


Chen Liang, Xifeng Gao, Kui Wu, and Zherong Pan. Second-order convergent collision-constrained
optimization-based planner. _IEEE Robotics and Automation Letters_, 2024.


Pingchuan Ma, Tao Du, Joshua B. Tenenbaum, Wojciech Matusik, and Chuang Gan. RISP:
Rendering-invariant state predictor with differentiable simulation and rendering for cross-domain
parameter estimation. In _International_ _Conference_ _on_ _Learning_ _Representations_, 2022. URL
[https://openreview.net/forum?id=uSE03demja.](https://openreview.net/forum?id=uSE03demja)


Jerrold E Marsden and Matthew West. Discrete mechanics and variational integrators. _Acta numer-_
_ica_, 10:357–514, 2001.


Igor Mordatch, Emanuel Todorov, and Zoran Popovi´c. Discovery of complex behaviors through
contact-invariant optimization. _ACM Transactions on Graphics (ToG)_, 31(4):1–8, 2012.


Rhys Newbury, Jack Collins, Kerry He, Jiahe Pan, Ingmar Posner, David Howard, and Akansel
Cosgun. A review of differentiable simulators. _IEEE Access_, 2024.


Zherong Pan and Dinesh Manocha. Active animations of reduced deformable models with environment interactions. _ACM Transactions on Graphics (TOG)_, 37(3):1–17, 2018.


Tao Pang, HJ Terry Suh, Lujie Yang, and Russ Tedrake. Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models. _IEEE_ _Transactions_ _on_ _robotics_,
2023.


Clemens Schwarke, Victor Klemm, Joshua Bagajo, Jean Pierre Sleiman, Ignat Georgiev, Jesus Tordesillas Torres, and Marco Hutter. Learning deployable locomotion control via differentiable simulation. In _9th Annual Conference on Robot Learning_, 2025.


Tuur Stuyck and Hsiao-yu Chen. Diffxpbd: Differentiable position-based simulation of compliant
constraint dynamics. _Proceedings of the ACM on Computer Graphics and Interactive Techniques_,
6(3):1–14, 2023.


Hyung Ju Suh, Max Simchowitz, Kaiqing Zhang, and Russ Tedrake. Do differentiable simulators
give better policy gradients? In _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 20668–
20696. PMLR, 2022a.


12


Hyung Ju Terry Suh, Tao Pang, and Russ Tedrake. Bundled gradients through contact via randomized smoothing. _IEEE Robotics and Automation Letters_, 7(2):4000–4007, 2022b.


Yuval Tassa, Tom Erez, and Emanuel Todorov. Synthesis and stabilization of complex behaviors
through online trajectory optimization. In _2012 IEEE/RSJ International Conference on Intelligent_
_Robots and Systems_, pp. 4906–4913, 2012. doi: 10.1109/IROS.2012.6386025.


Emanuel Todorov. A convex, smooth and invertible contact model for trajectory optimization. In
_2011_ _IEEE_ _International_ _Conference_ _on_ _Robotics_ _and_ _Automation_, pp. 1071–1076, 2011. doi:
10.1109/ICRA.2011.5979814.


Marc Toussaint, Kelsey R. Allen, Kevin A. Smith, and Joshua B. Tenenbaum. Differentiable physics
and stable modes for tool-use and manipulation planning  - extended abtract. In _Proceedings_ _of_
_the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19_, pp. 6231–
6235. International Joint Conferences on Artificial Intelligence Organization, 7 2019. doi: 10.
24963/ijcai.2019/869. [URL https://doi.org/10.24963/ijcai.2019/869.](https://doi.org/10.24963/ijcai.2019/869)


Dylan Turpin, Liquan Wang, Eric Heiden, Yun-Chun Chen, Miles Macklin, Stavros Tsogkas, Sven
Dickinson, and Animesh Garg. Grasp’d: Differentiable contact-rich grasp synthesis for multifingered hands. In _European Conference on Computer Vision_, pp. 201–221. Springer, 2022.


Keenon Werling, Dalton Omens, Jeongseok Lee, Ioannis Exarchos, and C. Karen Liu. Fast
and Feature-Complete Differentiable Physics Engine for Articulated Rigid Bodies with Contact Constraints. In _Proceedings_ _of_ _Robotics:_ _Science_ _and_ _Systems_, Virtual, July 2021. doi:
10.15607/RSS.2021.XVII.034.


Jie Xu, Tao Chen, Lara Zlokapa, Michael Foshey, Wojciech Matusik, Shinjiro Sueda, and Pulkit
Agrawal. An End-to-End Differentiable Framework for Contact-Aware Robot Design. In _Pro-_
_ceedings_ _of_ _Robotics:_ _Science_ _and_ _Systems_, Virtual, July 2021. doi: 10.15607/RSS.2021.XVII.
008.


Jie Xu, Miles Macklin, Viktor Makoviychuk, Yashraj Narang, Animesh Garg, Fabio Ramos, and
Wojciech Matusik. Accelerated policy learning with parallel differentiable simulation. In _In-_
_ternational_ _Conference_ _on_ _Learning_ _Representations_, 2022. URL [https://openreview.](https://openreview.net/forum?id=ZSKRQMvttc)
[net/forum?id=ZSKRQMvttc.](https://openreview.net/forum?id=ZSKRQMvttc)


Xiaohan Ye, Xifeng Gao, Kui Wu, Zherong Pan, and Taku Komura. Sdrs: Shape-differentiable robot
simulator. _IEEE Transactions on Robotics_, pp. 1–20, 2025. doi: 10.1109/TRO.2025.3636344.


13


A APPENDIX


A.1 ADDITIONAL PROOFS


_proof of Lemma 4.1._ **Barrier-Form** If _x_ ∈Cfree then each pair of _ti,tj_ on different rigid bodies is
disjoint as the convex hulls are closed sets. Therefore, by the separating plane theorem, there exists
a separating plane _pij_ and some positive _ϵij_ - 0 such that:


∥ _nij_ ∥= 1 ∧⟨ _xi_ ( _k_ )∈ _ti,nij_ ⟩+ _dij_ ≥ _ϵij_ /2 ∧⟨ _xj_ ( _k_ )∈ _tj_ _,nij_ ⟩+ _dij_ ≤− _ϵij_ /2 _._

Clearly, P _[t][i]_ [∪] _[t][j]_ ≤L _ij_ ( _pij_ /2 _,xi_ ( _k_ )∈ _ti,xj_ ( _k_ )∈ _tj_ ) < ∞. On the other hand, if _x_ ∈Cobs, then there
exists a non-disjoint pair _ti,tj_, so for any separating plane _pij_ there exists ⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ≤ 0 or
_xj_ ( _k_ ) _,nij_ ⟩+ _dij_ ≥ 0, leading to P _[t][i]_ [∪] _[t][j]_ = ∞. Finally, at any feasible solution _pij_, we must have
∥ _nij_ ∥> 0 because otherwise, we have:


3

L _ij_ = ∑
_k_ =1


1 3
( _dij_ ) [+] [+] _k_ ∑=1


1
(− _dij_ ) [+] [= ∞] _[.]_


We have thus established Barrier-Form for each P _[t][i]_ [∪] _[t][j]_ and thus P.


**Smoothness** This follows from the inverse function theorem (Dontchev & Rockafellar, 2009), the
smoothness of problem data L _ij_, and strictly convexity of L _ij_ .


**Non-prehensile & Non-vanishing** We can simply define A( _x_ ) = {⟨ _ti,tj_ ⟩∣ _ti_ ≠ _tj_ } such that every
pair of disjoint triangles ⟨ _ti,tj_ ⟩ appears in exactly one term of P _[t][i]_ [∪] _[t][j]_ . Thus, we only need to verify
that _fi_ _[t]_ ( _[i]_ _k_ [∪] )∈ _[t][j]_ _ti_ [∈F] _[t][j]_ [→] _[t][i]_ [and the case with] _[ f]_ _j_ _[ t]_ ( _[i]_ _k_ [∪] )∈ _[t][j]_ _tj_ [is symmetric.] [By the implicit function theorem, we]
can derive the analytic formula:

_f_ _[t][i]_ [∪] _[t][j]_ _nij_
_i_ ( _k_ )∈ _ti_ [=] (⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] _[.]_


Suppose _nij_ = _α_ ( _a_ - _b_ ) for some _a_ ∈ CH( _xi_ ( _k_ )∈ _ti_ ), _b_ ∈ CH( _xj_ ( _k_ )∈ _tj_ ), and _α_ - 0, then we have
_fi_ _[t]_ ( _[i]_ _k_ [∪] )∈ _[t][j]_ _ti_ [=] _[α]_ [(] _[a]_ [ −] _[b]_ [)/(⟨] _[x][i]_ [(] _[k]_ [)] _[,n][ij]_ [⟩+] _[ d][ij]_ [)][2] [∈F] _[t][j]_ [→] _[t][i]_ [.] [Therefore,] [we] [can] [in] [turn] [prove] [the] [sufficient]
condition that _nij_ = _α_ ( _a_ - _b_ ). Due to the optimality of L _ij_ with respect to _pij_, we have:


_nij_ 3

0 = _[∂]_ _∂n_ [L] _ij_ _[ij]_ = 12 ∥ _nij_ ∥(1 −∥ _nij_ ∥) [2] [−] _k_ ∑=1


_xi_ ( _k_ ) 3
(⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] [+] _k_ ∑=1


_xj_ ( _k_ )
(⟨ _xj_ ( _k_ ) _,_ - _nij_ ⟩− _dij_ ) [2]


3

0 = _[∂]_ [L] _[ij]_ = − ∑

_∂dij_ _k_ =1


1 3
(⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] [+] _k_ ∑=1


1
(⟨ _xj_ ( _k_ ) _,_ - _nij_ ⟩− _dij_ ) [2] _[.]_


From the above two equations, we can conclude that _nij_ = _α_ ( _a_ - _b_ ) by defining:


_α_ = [∥] _[n][ij]_ [∥(][1][ −∥] _[n][ij]_ [∥)]

12


3

∑
_k_ =1


1
(⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] [>][ 0]


3
_a_ =[ ∑
_k_ =1


3
_b_ =[ ∑
_k_ =1


thus all is proved.


_xi_ ( _k_ ) 3
(⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] []/[] _k_ ∑=1


1
(⟨ _xi_ ( _k_ ) _,nij_ ⟩+ _dij_ ) [2] [] ∈] [CH][(] _[x][i]_ [(] _[k]_ [)∈] _[t][i]_ [)]


_xj_ ( _k_ ) 3
(⟨ _xj_ ( _k_ ) _,_ - _nij_ ⟩− _dij_ ) [2] []/[] _k_ ∑=1


1
(⟨ _xj_ ( _k_ ) _,_ - _nij_ ⟩− _dij_ ) [2] [] ∈] [CH][(] _[x][j]_ [(] _[k]_ [)∈] _[t][j]_ [)] _[,]_


_Proof of Lemma 5.1._ **Barrier-Form** Case I: When ∥ _x_ I - _x_ J ∥< _d_ 1, P _d_ [I∪J] 1→ _d_ 2 [=] [P] _d_ [I∪J] 1 which pertains Barrier-Form since P _d_ [I∪J] 1 pertains Barrier-Form by assumption. Case II: When ∥ _x_ I − _x_ J ∥=
_d_ 1 ≥ _R_ I + _R_ J, there are two sub-cases. Case II.a: If _x_ ∈Cobs [I∪J] [, e.g.] [where two bounding spheres]
are just touching and the touching point lies on a common triangle, then 0 ≤P _d_ [I∪J] 2 ≤P _d_ [I∪J] 1 = ∞
where the first two inequalities are due to our assumption and the last equality is due to P _d_ [I∪J] 1
pertaining Barrier-Form, so P _d_ [I∪J] 1→ _d_ 2 [=] [P] _d_ [I∪J] 1 = ∞. Case II.b: If _x_ ∈Cfree [I∪J] [,] [then] [we] [have]


14


0 ≤P _d_ [I∪J] 2 ≤P _d_ [I∪J] 1 < ∞ following the same reasoning as case II.a, so 0 ≤P _d_ [I∪J] 1→ _d_ 2 [<] [∞][.] [Case]
III: When ∥ _x_ I - _x_ J ∥> _d_ 1 ≥ _R_ I + _R_ J, then we must have _x_ ∈Cfree [I∪J] and the same analysis as case
II.b leads to 0 ≤P _d_ [I∪J] 1→ _d_ 2 [< ∞][. Thus, we have verified that][ P] _d_ [I∪J] 1→ _d_ 2 [pertains] [Barrier-Form in all cases.]

**Smoothness** This is due to Smoothness in P _d_ [I∪J] 1, P _d_ [I∪J] 2, and the second differentiability of _ϕd_ 1→ _d_ 2.

**Non-prehensile & Non-vanishing** The force _fi_ [I∪J] ∈I induced by P _d_ [I∪J] 1→ _d_ 2 [takes the following form:]


_fi_ [I∪J] ∈I =−(1 − _ϕd_ 1→ _d_ 2( _x_ )) _∂∂x_ P _d_ [I∪J] _i_ 1∈I

    - **���������������������������������������** **���������������������������������������**
term I


_∂_ P _d_ [I∪J] 2

- _ϕd_ 1→ _d_ 2( _x_ ) _∂xi_ ∈I

- **����������������������������** - **����������������������������**
term II


[−] _[x]_ [J] [∥−] _[d]_ [1][)/(] _[d]_ [2] [−] _[d]_ [1][))]
−(P _d_ [I∪J] 2 −P _d_ [I∪J] 1 ) _[ϕ]_ [′][((∥] ( _[x]_ _d_ [I] 2 − _d_ 1)∥ _x_ I − _x_ J ∥∣I∣ ( _x_ I − _x_ J )


(9)

_._


         - **��������������������������������������������������������������������������������������������������������**         - **��������������������������������������������������������������������������������������������������������**
term III


There are three terms and we show that each term belongs to FJ →I. For term I, we know that

- _∂_ P _d_ [I∪J] 1 / _∂xi_ ∈I ∈FJ →I since P _d_ [I∪J] 1 satisfies Non-prehensile & Non-vanishing. Since the coefficient (1 − _ϕd_ 1→ _d_ 2( _x_ )) ≥ 0 and FJ →I is a cone, we conclude that term I is zero or belongs to
FJ →I. The same reasoning applies to term II. For term III, _x_ I - _x_ J belongs to FJ →I. Since
P _d_ [I∪J] 2 ≤P _d_ [I∪J] 1 by our assumption, the remaining coefficient is non-negative, thus term III is zero
or belongs to FJ →I. Finally, at least one of term I or term II is non-zero, so we conclude that
_fi_ [I∪J] ∈I ∈FJ →I and all is proved.


_Proof of Corollary 5.2._ We first show that Equation 5 is correct. The objective function in Equation 5 is derived by replacing all _xi_ ( _k_ ) and _xj_ ( _k_ ) in L _ij_ with the center points _xti_ and _xtj_, respectively. By symmetry, the optimal separating plane must be the middle surface between _xti_ and _xtj_,
taking the following form:


_nij_ = _α_ ( _xti_            - _xtj_ ) and _dij_ = − _α_ /2⟨ _xti_            - _xtj_ _,xti_ + _xtj_ ⟩ _._


Plugging the optimal separating plane and solving for _α_ leads to Equation 5. Next, we show that all
three assumptions in Lemma 5.1 hold. In fact, there are only two non-trivial assumptions. We first
show that P _c_ _[t][i]_ [∪] _[t][j]_ ≤P _[t][i]_ [∪] _[t][j]_ when ∥ _x_ I − _x_ J ∥≥ _d_ 1. Let use denote _p_ [⋆] _ij_ [as the optimal separating plane]
for P _[t][i]_ [∪] _[t][j]_, then we have the following inequality:


1 3
P _c_ _[t][i]_ [∪] _[t][j]_ =argmin _pij_ [12 (1 −∥ _nij_ ∥) [+] [+] _k_ ∑=1


1 3
(⟨ _xti,nij_ ⟩+ _dij_ ) [+] [+] _k_ ∑=1


1
(⟨ _xtj_ _,_ - _nij_ ⟩− _dij_ ) [+] []]


1 3
≤12 (1 −∥ _n_ [⋆] _ij_ [∥)][+] [+] _k_ ∑=1

1 3
≤12 (1 −∥ _n_ [⋆] _ij_ [∥)][+] [+] _k_ ∑=1

=P _[t][i]_ [∪] _[t][j]_ _,_


1 3
(⟨ _xti,n_ [⋆] _ij_ [⟩+] _[ d]_ [⋆] _ij_ [)][+] [+] _k_ ∑=1


1
(⟨ _xtj_ _,_ - _n_ [⋆] _ij_ [⟩−] _[d]_ [⋆] _ij_ [)][+]


1 3
(⟨ _xi_ ( _k_ ) _,n_ [⋆] _ij_ [⟩+] _[ d]_ [⋆] _ij_ [)][+] [+] _k_ ∑=1


1
(⟨ _xj_ ( _k_ ) _,_ - _n_ [⋆] _ij_ [⟩−] _[d]_ [⋆] _ij_ [)][+]


where the first inequality is due to optimality of P _c_ _[t][i]_ [∪] _[t][j]_, the second inequality is due to the convexity
of function 1/(⟨● _,n_ [⋆] _ij_ [⟩+] _[ d]_ [⋆] _ij_ [)][+] [and][ 1][/(⟨●] _[,]_ [−] _[n]_ [⋆] _ij_ [⟩−] _[d]_ [⋆] _ij_ [)][+][.] [We then show that][ P] _c_ _[t][i]_ [∪] _[t][j]_ satisfies Equation 2. The force on any _xi_ ( _k_ ) takes the following form:

_fi_ [I∪J] ∈I = ∥ [4] _x_ [(] _t_ _[x]_ _i_               - _[t][i]_ _x_ [−] _t_ _[x]_ _j_ ∥ _[t][j]_ [5][)][/][2] [[][1][ +] ∥ _xti_ −1 _xtj_ ∥ [1][/][2] []] _[,]_


which clearly belongs to FJ →I, thus all is proved.

**Lemma A.1.** _If ϵ_ - 0 _then_ P _BSH_ [I∪J] _[pertains]_ _[Barrier-Form for any node pair]_ [ I] _[,]_ [J] _[of two rigid bodies.]_


_Proof._ First, by induction from leaf to the root node, we can verify that 0 ≤PBSH [I∪J] [.] [Second, suppose]
_x_ ∈Cfree, then all the pair-wise terms between leaf nodes PBSH _[t][i]_ [∪] _[t][j]_ < ∞. Further, for all the centered


15


potential in Equation 7, we have P _c_ [I∪J] < ∞ because they are evaluated only when _R_ I + _R_ J ≤
∥ _x_ I − _x_ J ∥. The root potential PBSH [I∪J] [is then derived by a finite number of blending and summation]
so we have PBSH [I∪J] [< ∞][.] [Third, at any] _[ x]_ [ ∈C][obs][, there exists a non-disjoint pair] _[ t][i][,t][j]_ [belonging to the]
two rigid bodies. We will show the following two claims hold by induction from leaf to root:


    - PBSH [I∪J] [= ∞] [at any node such that] _[ t][i]_ [ ⊆I] [and] _[ t][j]_ [⊆J][ .]


**Base Step:** PBSH _[t][i]_ [∪] _[t][j]_ [= P] _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2 [= ∞] [by Corollary 5.2.]


**Inductive** **Step:** We assume our first claim holds for any _ti_ ∈I _c_ ∈ _C_ (I) and _tj_ ∈J _c_ ∈ _C_ (J ). If
_ti_ ⊆I and _tj_ ⊆J, we must have ∥ _x_ I - _x_ J ∥≤ _R_ I + _R_ J due to the pair _ti,tj_ being non-disjoint.
The children set satisfying _ti_ ∈I _c_ ∈ _C_ (I) and _tj_ ∈J _c_ ∈ _C_ (J ) can always be found, so we have
PBSH [I∪J] [= P] _d_ [I∪J] 1 ≥PBSH [I] _[c]_ [∪J] _[c]_ = ∞.


To prove that PBSH [I∪J] [pertains] [Non-prehensile] [&] [Non-vanishing,] [i.e.,] [the] [non-prehensile] [and] [non-]
vanishing property, we also need to use induction. To this end, we establish the non-prehensile
property for an index subset as follows:

**Definition A.2.** _The pairwise potential_ P _BSH_ [I∪J] _[pertains]_ _[Non-prehensile & Non-vanishing restricted]_
_to_ ⟨I _,_ J ⟩ _if, at every x_ ∈C _free, we can define a finite family of set pairs_ AI∪J ( _x_ ) _such that_ P _BSH_ [I∪J] [=]
∑⟨I [′] _,_ J [′] ⟩∈AI∪J ( _x_ ) P _BSH_ [I][′][∪J][ ′] _,_ _where_ _every_ _term_ P _BSH_ [I][′][∪J][ ′] _satisfy_ _Equation_ _2._ _Further,_ _for_ _every_ _pair_ _of_
_disjoint triangles_ ⟨ _ti,tj_ ⟩ _such that ti_ ∈I _and tj_ ∈J _or vice versa, we have ti_ ∪ _tj_ ⊆I [′] ∪J [′] _for at_
_least one_ ⟨I [′] _,_ J [′] ⟩∈AI∪J ( _x_ ) _._

**Lemma A.3.** _If ϵ_ - 0 _then_ P _BSH_ [I∪J] _[pertains]_ _[Non-prehensile & Non-vanishing restricted to]_ [ ⟨I] _[,]_ [J ⟩] _[for]_
_any node pair_ I _,_ J _of two rigid bodies._


_Proof._ At _x_ ∈Cfree, we show the following two claims by induction from leaf to root:


    - The pairwise potential P _d_ [I∪J] 1 ≥PBSH [I∪J] [≥P] _d_ [I∪J] 2 at any node when ∥ _x_ I − _x_ J ∥≥ _R_ I + _R_ J .


    - The pairwise potential PBSH [I∪J] pertains Non-prehensile & Non-vanishing restricted to
⟨I _,_ J ⟩ for any node pair.


**Base Step:** For the pairwise potential PBSH _[t][i]_ [∪] _[t][j]_ [= P] _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2 [, we define][ A] _[t][i]_ [∪] _[t][j]_ [(] _[x]_ [) = {⟨] _[t][i][,t][j]_ [⟩}][, then] [Non-]
prehensile & Non-vanishing and the fact that PBSH _[t][i]_ [∪] _[t][j]_ [≥P] _c_ _[t][i]_ [∪] _[t][j]_ follows from Corollary 5.2.

**Inductive Step I:** We assume our first claim holds for all I _c_ ∈ _C_ (I) and J _c_ ∈ _C_ (J ), i.e. PBSH [I] _[c]_ [∪J] _[c]_ ≥
P _c_ [I] _[c]_ [∪J] _[c]_ when ∥ _x_ I _c_ - _x_ J _c_ ∥≥ _R_ I _c_ + _R_ J _c_ . We first show that P _d_ [I∪J] 2 ≤P _d_ [I∪J] 1 when ∥ _x_ I - _x_ J ∥≥
_R_ I + _R_ J . We note that I ∪J contains at least 3 triangles, otherwise we reduce to the base case, so
there are at least 2 terms of form PBSH [I] _[c]_ [∪J] _[c]_ . Each such term has the following lower bound:


1
PBSH [I] _[c]_ [∪J] _[c]_ ≥P _c_ [I] _[c]_ [∩J] _[c]_ = 12 1 + ~~√~~
⎡⎢⎢⎢⎣ ∥ _x_ I _c_                                               - _x_ J _c_ ∥


2
1
≥ 12 1 + ~~√~~
⎡⎢⎢⎢⎣ ∥ _x_ I − _x_ J


⎤⎥⎥⎥⎦


∥ _x_ I − _x_ J ∥+ _R_ I + _R_ J


2

_._

⎤⎥⎥⎥⎦


Here, the first inequality is due to our inductive condition and the fact that our BSH is a layered
hierarchy by Definition A.2, so that ∥ _x_ I - _x_ J ∥≥ _R_ I + _R_ J implies ∥ _x_ I _c_ - _x_ J _c_ ∥≥ _R_ J _c_ + _R_ J _c_ . The
second inequality is because _x_ I _c_ (resp. _x_ J _c_ ) is at most _R_ I (resp. _R_ J ) from _x_ I (resp. _x_ J ). Using


16


the above lower bound, we derive the following estimate:


1
P _d_ [I∪J] 1 ≥24 1 + ~~√~~
⎡⎢⎢⎢⎣ ∥ _x_ I − _x_ J ∥+ _R_ I + _R_ J


2

⎤⎥⎥⎥⎦


48 24
≥24 + ~~√~~ +

∥ _x_ I − _x_ J ∥+ _R_ I + _R_ J ∥ _x_ I − _x_ J ∥+ _R_ I + _R_ J


48 24
≥24 + ~~√~~ +

2∥ _x_ I − _x_ J ∥ 2∥ _x_ I − _x_ J ∥


24 12 1

+ 1 + ~~√~~
∥ _x_ I − _x_ J ∥ ∥ _x_ I − _x_ J ∥ [=][ 12] ⎡⎢⎢⎢⎣ ∥ _x_ I


24
≥12 + ~~√~~


∥ _x_ I − _x_ J ∥


2
= P _c_ [I∪J] = P _d_ [I∪J] 2 _,_

⎤⎥⎥⎥⎦


where we use the fact that ∥ _x_ I - _x_ J ∥≥ _d_ 1 = _R_ I + _R_ J in the third inequality. As a result, we have
our first claim holds for I ∪J by the definition of the blending Equation 4.

**Inductive Step II:** We assume our second claim holds for all I _c_ ∈ _C_ (I) and J _c_ ∈ _C_ (J ), i.e. PBSH [I] _[c]_ [∪J] _[c]_
pertains Non-prehensile & Non-vanishing restricted to ⟨I _c,_ J _c_ ⟩. We show that PBSH [I∪J] [pertains] [Non-]
prehensile & Non-vanishing restricted to ⟨I _,_ J ⟩ by considering three cases. Case II.a: If ∥ _x_ I _x_ J ∥≤ _d_ 1, then PBSH [I∪J] [=] [P] _d_ [I∪J] 1 consists of terms of form PBSH [I] _[c]_ [∪J] _[c]_ each satisfying Non-prehensile
& Non-vanishing by our inductive condition. Let us now define the finite family of set pairs by
the union AI∪J ( _x_ ) = ∪I _c_ ∈ _C_ (I) _,_ J _c_ ∈ _C_ (J )AI _c_ ∪J _c_ ( _x_ ). It can be verified that this union is a disjoint
union, and for every pair of disjoint triangles ⟨ _ti_ ∈I _,tj_ ∈J ⟩, we have _ti_ ∪ _tj_ ⊆I [′] ∪J [′] belongs
to exactly one such AI _c_ ∪J _c_ ( _x_ ). Further, we have PBSH [I∪J] [=] [∑][⟨I][′] _[,]_ [J][ ′][⟩∈A] I∪J [(] _[x]_ [)] [P] BSH [I][′][∪J][ ′], where each
PBSH [I][′][∪J][ ′] satisfies Equation 2 due to Non-prehensile & Non-vanishing of the corresponding PBSH [I] _[c]_ [∪J] _[c]_
by our inductive condition. We have thus verified Non-prehensile & Non-vanishing of PBSH [I∪J] [.] [Case]
II.b: If ∥ _x_ I - _x_ J ∥≥ _d_ 2, then PBSH [I∪J] [=] [P] _d_ [I∪J] 2 = P _c_ [I∪J] is a singled, centered potential. We trivially
define AI∪J ( _x_ ) = {⟨I _,_ J ⟩} then clearly every pair of disjoint triangles _ti_ ∪ _tj_ ⊆I ∪J ∈AI∪J ( _x_ ).
Further, the induced force takes the following form:

_fi_ [I∪J] ∈I = ∣I∣∥12 _x_ ( _x_ I −I              - _xx_ J ∥J [5] ) [/][2] [[][1][ +] ∥ _x_ I − 1 _x_ J ∥ [1][/][2] []] _[,]_


which clearly belongs to FJ →I, thus we have verified all conditions in Non-prehensile & Nonvanishing of PBSH [I∪J] [. Case II.c: If] _[ d]_ [1][ < ∥] _[x]_ [I][ −] _[x]_ [J][ ∥<] _[ d]_ [2][, then][ P] BSH [I∪J] [is a blending of][ P] _d_ [I∪J] 1 in Case II.a
and P _d_ [I∪J] 2 in Case II.b, with strictly positive weights. We again trivially define AI∪J ( _x_ ) = {⟨I _,_ J ⟩}
to have every pair of disjoint triangles _ti_ ∪ _tj_ ⊆I ∪J ∈AI∪J ( _x_ ). The only condition we need to
verify is that PBSH [I∪J] [satisfies Equation 2 as a single term.] [We use a similar technique as in the proof]
of Lemma 5.1 by expanding the force term to get Equation 9 where there are three terms. For term
II, We have by the analysis in case II.b that − _∂_ P _d_ [I∪J] 2 / _∂xi_ ∈I ∈FJ →I and _ϕd_ 1→ _d_ 2( _x_ ) > 0 is strictly
positive, so term II belongs to FJ →I. Term III is zero or belongs to FJ →I because P _d_ [I∪J] 1 ≥P _d_ [I∪J] 2
by our first claim. For term I, we know from the analysis in Case II.a that:

_∂_ P _d_ [I∪J] 1 BSH

          - = ∑           - _[∂]_ [P] [I][′][∪J][ ′] _,_
_∂xi_ ∈I ⟨I [′] _,_ J [′] ⟩∈AI∪J ( _x_ ) _∂xi_ ∈I

where each term − _∂_ PBSH [I][′][∪J][ ′] / _∂xi_ ∈I is zero or belongs to FJ ′→I′ ⊆FJ →I. Combined with the fact
that the coefficient (1− _ϕd_ 1→ _d_ 2( _x_ )) > 0 is strictly positive, we conclude that term I is zero or belongs
to FJ →I. As a result, we see that PBSH [I∪J] [satisfies Equation 2 as a single term, so] [Non-prehensile ][&]
Non-vanishing holds.


_Proof of Theorem 5.4._ Barrier-Form and Non-prehensile & Non-vanishing follows
from Lemma A.1 and Lemma A.3, respectively. Smoothness follows from the fact that
P [I∪J]
BSH [is derived by a finite number of blending between pairwise potentials, and all potentials a][nd]
blending operators are twice differentiable.


17


Level 2


Figure 10: A special BSH constructed for the uniform grid. The levels of the BSH are indexed from
bottom up. The −1th level contains only leaves. Every (2 _i_ + 1)th level ( _i_ ≥ 0) merges two diagonal,
rectangular blocks. Therefore, every (2 _i_ )th level consists of a cubic mesh block of side length 2 _[i]_ .


A.2 COMPLEXITY ANALYSIS FOR A UNIFORM GRID


We show that, in the special case of two rigid bodies in the shape of a square uniform grid with
infinitesimal distace to each other, as illustrated in Figure 11, the cost of evaluating PBSH is _O_ ( _T_ ).
Here we assume a 2D uniform grid with _N_ [2] grid cells so that _T_ = _O_ ( _N_ [2] ). Without a loss of
generality, we assume the grid size is 1. Further, for ease of analysis, we adopt a special construction
of BSH as illustrated in Figure 10. It is easy to see that the bounding sphere radius of nodes in each
level is all the same, which is denoted as _ri_ . We have the following results for estimating _ri_, which
can be derived directly from the construction of PBSH [I∪J] [:]


~~√~~ √

- _r_ −1 = 5/3 _r_ 0 =


√
2+2


2 5

6


√

- _r_ 2 _i_ −1 = _r_ 2 _i_ ≤ 2 _[i]_ [−][1][/][2] _Cr_ ∀ _i_ - 0 with _Cr_ = [1][+]


10
3


To derive the second property, note that the tightest bounding sphere for the geometry of a (2 _i_ )-level
node is 2 _[i]_ [−][1][/][2] . This implies that the tightest bounding sphere for the geometry of a 0-level node is
~~√~~ ~~√~~ ~~√~~
1/ 2. However, the actual _r_ 0 = ( 1 + 10)/3, so the bounding sphere is unnecessarily scaled by

_Cr_ . By induction, we can prove that we can scale all tightest bounding spheres by _Cr_ accordingly
to satisfy Definition 5.3.


~~√~~
2. However, the actual _r_ 0 = (


~~√~~
1 +


To analyze the cost of evaluating PBSH [I∪J] [,] [we]
note that, the cost reduces to evaluating a series
of interaction terms either using centered potential (P _c_ [I∪J] ) or between leaf nodes (P _[t][i]_ [∪] _[t][j]_ ).
Further by the recursive definition and the special structure of our BSH, the interaction terms
are always computed between two nodes at the
same level. Therefore, we can upper bound the
number of interaction terms level by level.


**Case** **I:** (2 _i_ ) **-level** **Node** We focus on the
nodes at (2 _i_ )th level, which bounds a cubic
mesh block of side length 2 _[i]_ . We can denote these mesh blocks as B [2] _mn_ _[i]_ [indexed] [us-]
ing subscript _mn_ . In other words, B [2] _mn_ _[i]_ [con-]
sists of all triangles with coordinates within

[ _m_ 2 _[i]_ _,_ ( _m_ + 1)2 _[i]_ ] × [ _n_ 2 _[i]_ _,_ ( _n_ + 1)2 _[i]_ ]. For two


I


Figure 11: Two rigid bodies in the shape of a
square uniform grid with infinitesimal distance
_d_ → 0.


18


such blocks indexed by B [2] _mn_ _[i]_ [⊂I][ and][ B][2] _m_ _[i]_ [′] _n_ [′] [⊂I][ on the two rigid bodies, we can define their distance]
as:

_d_ (B [2] _mn_ _[i]_ _[,]_ [B][2] _m_ _[i]_ [′] _n_ [′][) =][ max][(∣] _[m]_ [ −] _[m]_ [′][∣] _[,]_ [∣] _[n]_ [ −] _[n]_ [′][∣)] _[.]_


Now suppose _d_ (B [2] _mn_ _[i]_ _[,]_ [B][2] _m_ _[i]_ [′] _n_ [′][) ≥] [2][, we can be sure that][ B] _mn_ [2] _[i]_ [and][ B] _m_ [2] _[i]_ [′] _n_ [′] [belong to different][ (][2] _[i]_ [ +][ 2][)][-]
level blocks. Specifically, we define B [2] _mn_ _[i]_ [⊆] [B] _m_ [2] ¯ _[i]_ [+] _n_ ¯ [2] [and] [B][2] _m_ _[i]_ [′] _n_ [′] [⊆] [B][2] _m_ ¯ _[i]_ [+][′] _n_ ¯ [2][′] [and] [we] [have] [the] [following]
relationship:


_mn_ _[,]_ [B][2] _m_ _[i]_ [′] _n_ [′][)]
_d_ (B [2] _m_ ¯ _[i]_ [+] _n_ ¯ [2] _[,]_ [B] _m_ [2] ¯ _[i]_ [+][′] _n_ ¯ [2][′][) ≥⌊] _[d]_ [(][B][2] _[i]_ ⌋ _._ (10)
2


Now let us define I = B [2] _m_ ¯ _[i]_ [+] _n_ ¯ [2] [and] [J] [=] [B][2] _m_ ¯ _[i]_ [+][′] _n_ ¯ [2][′][,] [we] [know] [that] [if] [the] [distance] [between] [the] [two] [cubic]
mesh blocks is sufficiently faraway, the potential term P [I∪J] reduces to the centered potential P _c_ [I∪J]
can be computed via Equation 7 without utilizing any information from lower levels. A sufficient
condition for this to happen is as follows:


_d_ (B [2] _m_ ¯ _[i]_ [+] _n_ ¯ [2] _[,]_ [B] _m_ [2] ¯ _[i]_ [+][′] _n_ ¯ [2][′][)][2] _[i]_ [ ≥] [2] _[r]_ [2] _[i]_ [+][2][(][1][ +] _[ ϵ]_ [)] _[.]_ (11)


~~√~~
Combining Equation 10 and Equation 11, we know that if _d_ (B [2] _mn_ _[i]_ _[,]_ [B][2] _m_ _[i]_ [′] _n_ [′][)] [≥] [2][⌈][2]


Combining Equation 10 and Equation 11, we know that if _d_ (B _mn_ _[,]_ [B] _m_ [′] _n_ [′][)] [≥] [2][⌈][2] 2 _Cr_ (1 + _ϵ_ )⌉,

then the interaction between the two (2 _i_ )-level blocks would be handled by the two (2 _i_ + 2)-level
super-blocks. Therefore, in order to compute the contact potential, we only need to evaluate the
~~√~~
interaction between _Bmn_ [2] _[i]_ [and] [at] [most] [(][1][ +][ 4][⌈][2] 2 _Cr_ (1 + _ϵ_ )⌉) [2] blocks around it. The number of


interaction between _Bmn_ [2] _[i]_ [and] [at] [most] [(][1][ +][ 4][⌈][2] 2 _Cr_ (1 + _ϵ_ )⌉) [2] blocks around it. The number of

(2 _i_ )-level interaction terms of form P _c_ [I∪J] is:


_O_ (⌈ _[N]_


2 ~~√~~
(1 + 4⌈2


[⌉]
2 _[i]_


2 _Cr_ (1 + _ϵ_ )⌉) [2] ) _,_


where the first part ⌈ _N_ /2 _[i]_ ⌉ [2] is the number of (2 _i_ )-level blocks and the second part is the number of
other (2 _i_ )-level blocks, with which an interaction term P _c_ [I∪J] needs to be calculated.


**Case II:** (2 _i_ - 1) **-level Node** We have finished analyzing the cost of (2 _i_ )-level node interactions.
The case with (2 _i_ - 1)-level node iterations ( _i_ ≥ 1) is almost identical. Each (2 _i_ - 1)-level node
consists of half the triangles of a (2 _i_ )-level node, so we can assume the (2 _i_ )-level node _Bmn_ [2] _[i]_ [has]
two children denoted as B _mn_ [2] _[i]_ [−][1] _[,l]_ and B [2] _mn_ _[i]_ [−][1] _[,r]_ . Without a loss of generality, we only consider B [2] _mn_ _[i]_ [−][1] _[,l]_,
which has the same bounding sphere center as that of B [2] _mn_ _[i]_ [.] ~~√~~ [By] [the] [analysis] [of] [Case] [I,] [we] [know]
that, for all the children of B [2] _m_ _[i]_ [′] _n_ [′] [with] _[d]_ [(][B][2] _mn_ _[i]_ _[,]_ [B][2] _m_ _[i]_ [′] _n_ [′][)] [≥] [2][⌈][2] 2 _Cr_ (1 + _ϵ_ )⌉, their interaction with


that, for all the children of B _m_ [′] _n_ [′] [with] _[d]_ [(][B] _mn_ _[,]_ [B] _m_ [′] _n_ [′][)] [≥] [2][⌈][2] 2 _Cr_ (1 + _ϵ_ )⌉, their interaction with

B _mn_ [2] _[i]_ [−][1] _[,l]_ would be taken care of at level (2 _i_ + 2). Again, we conclud ~~√~~ e that we only need to evaluate
the interaction between B [2] _mn_ _[i]_ [−][1] _[,l]_ and the children of at most (1 + 4⌈2 2 _Cr_ (1 + _ϵ_ )⌉) [2] (2 _i_ )-level nodes


the interaction between B [2] _mn_ _[i]_ [−][1] _[,l]_ and the children of at most (1 + 4⌈2 2 _Cr_ (1 + _ϵ_ )⌉) [2] (2 _i_ )-level nodes

around it. The number of (2 _i_ - 1)-level interaction terms of form P _c_ [I∪J] is again:


_O_ (⌈ _[N]_


2 ~~√~~
(1 + 4⌈2


[⌉]
2 _[i]_


2 _Cr_ (1 + _ϵ_ )⌉) [2] ) _._


**Case III:** −1 **-level Leaf Node** The case with leaf nodes is exactly the same as that of (2 _i_ - 1)-level
nodes. Each 0-level node has two children at −1-level denoted as B [−] _mn_ [1] _[,l]_ [and][ B][−] _mn_ [1] _[,r]_ [.] [Focusing on][ B][−] _mn_ [1] _[,l]_
and ~~√~~ by the analysis of Case II, we know that, for all the children of B [0] _m_ [′] _n_ [′] [with] _[d]_ [(][B][0] _mn_ _[,]_ [B][0] _m_ [′] _n_ [′][)] [≥]
2⌈2 2 _Cr_ (1 + _ϵ_ )⌉, their interaction with B [−][1] _[,l]_ [would] [be] [taken] [care] [of] [at] [level] [2][.] [Therefore,] [we]


2⌈2 2 _Cr_ (1 + _ϵ_ )⌉, their interaction with B [−] _mn_ [1] _[,l]_ [would] [be] [taken] [care] [of] [at] [level] [2][.] [Therefore,] [we]

conclude ~~√~~ that we only need to evaluate the interaction between B [−] _mn_ [1] _[,l]_ [and] [the] [children] [of] [at] [most]
(1 + 4⌈2 2 _Cr_ (1 + _ϵ_ )⌉) [2] 0-level nodes around it. The number of (2 _i_ - 1)-level interaction terms of


(1 + 4⌈2 2 _Cr_ (1 + _ϵ_ )⌉) [2] 0-level nodes around it. The number of (2 _i_ - 1)-level interaction terms of

form P _[t][i]_ [∪] _[t][j]_ is again:


~~√~~
_O_ ( _N_ [2] (1 + 4⌈2 2 _Cr_ (1 + _ϵ_ )⌉) [2] ) _._


Put everything together, the cost of evaluating PBSH is:


⌈log2 _N_ ⌉
∑ _O_ (⌈ _[N]_
_i_ =0 2 _[i]_


2
) = _O_ ( _N_ [2] ) = _O_ ( _T_ ) _._


[⌉]
2 _[i]_


19


A.3 FRICTIONAL CONTACT MODELING


A feature-complete potential function should be able to handle frictional contacts. To this end, we
adopt the technique proposed by Li et al. (2020); Ye et al. (2025) and consider the contact potential
between the pair of triangles P _[t][i]_ [∪] _[t][j]_ ( _x_ _[t]_ _i_ (1) _[,x]_ _i_ _[t]_ (2) _[,x]_ _i_ _[t]_ (3) _[,x]_ _j_ _[t]_ (1) _[,x]_ _j_ _[t]_ (2) _[,x]_ _j_ _[t]_ (3) [)][, where we write explicitly]
the six vertices related to this potential function. The negative gradient norm of this potential is the
normal force applied on the corresponding vertices. Li et al. (2020) proposes to model the contact
potential as a tangential velocity damping term weight by the normal force magnitude. Put together,
the frictional damping term takes the following form:


3
_D_ ∥( _x_ _[t]_ _i_ ( [+] _k_ [1] ) _[,x]_ _i_ _[t]_ ( _k_ ) _[,δt]_ [) +] ∑ _λ_
����������� _k_ =1 �����������


P _[t][i]_ [∪] _[t][j]_

_x_ _[t]_
_j_ ( _k_ )


_D_ ∥( _x_ _[t]_ _j_ [+] ( _k_ [1] ) _[,x]_ _j_ _[t]_ ( _k_ ) _[,δt]_ [)] _,_
����������� ⎤⎥⎥⎥⎥⎦


D( _x_ _[t]_ [+][1] _,x_ _[t]_ _,δt_ ) = ∑
_ti_ ≠ _tj_


3
∑ _λ_

⎡⎢⎢⎢⎢⎣ _k_ =1 �����������


P _[t][i]_ [∪] _[t][j]_

_x_ _[t]_
_i_ ( _k_ )


where the summation is taken over triangle pairs on different rigid bodies, _D_ ∥ is the tangential
velocity damping term, which penalizes the relative velocity between _ti_ and _tj_, and _λ_ is the frictional
coefficient. However, the above formulation is not strictly second-order differentiable. To fix this
problem, Ye et al. (2025) proposed a novel definition of _D_ ∥, which does not penalize the relative
velocity between _ti_ and _tj_ . Instead, they penalize the relative velocity between the two triangles and
the separating plane _pij_, assuming the plane is a physical object with zero-mass. We refer readers to
their work for more details.


The original frictional damping term can only deals with frictions between triangles. However,
our contact potential PBSH [I∪J] [is] [a] [hierarchical blending] [of potentials] [between triangles and] [centered]
potentials. To extend the frictional damping term to use our PBSH [I∪J] [,] [we] [propose] [to] [disregard] [the]
centered potentials and only consider potentials between triangles. Specifically, we propose the
following potential between a pair of triangles:


_d_ 1 = _Rti_ + _Rtj_ and _d_ 2 = (1 + _ϵ_ ) _d_ 1
P _d_ _[t][i]_ 1 [∪] _[t][j]_ = P _[t][i]_ [∪] _[t][j]_ and P _d_ _[t][i]_ 2 [∪] _[t][j]_ = 0
P _[t][i]_ [∪] _[t][j]_

⎧⎪⎪⎪⎪⎨⎪⎪⎪⎪⎩ local [= P] _d_ _[t][i]_ 1 [∪] → _[t]_ _d_ _[j]_ 2


_._ (12)


Compared with the potential between leaf nodes in Equation 6, Equation 12 is not globally supported
and vanishes when the distance between triangles is larger than _d_ 2, denoted using subscript ●local.
This design choice would not cause gradient vanish because our normal contact potential P always
provides non-vanishing gradient. In parallel, the benefit of using a locally supported function is
that we can use a bounding volume hierarchy to quickly reject faraway triangles as done in Li et al.
(2020); Ye et al. (2025). Equation 12 is plugged into the frictional damping term to yield our final
formulation:


3
_D_ ∥( _x_ _[t]_ _i_ ( [+] _k_ [1] ) _[,x]_ _i_ _[t]_ ( _k_ ) _[,δt]_ [) +] ∑ _λ_
����������� _k_ =1 �����������


_D_ ∥( _x_ _[t]_ _j_ [+] ( _k_ [1] ) _[,x]_ _j_ _[t]_ ( _k_ ) _[,δt]_ [)] _._
����������� ⎤⎥⎥⎥⎥⎦


P _[t][i]_ [∪] _[t][j]_
local
_x_ _[t]_
_i_ ( _k_ )


P _[t][i]_ [∪] _[t][j]_
local
_x_ _[t]_
_j_ ( _k_ )


D( _x_ _[t]_ [+][1] _,x_ _[t]_ _,δt_ ) = ∑
_ti_ ≠ _tj_


3
∑ _λ_

⎡⎢⎢⎢⎢⎣ _k_ =1 �����������


Intuitively, our frictional damping model assumes that two triangles can only impose frictional
damping forces on each other if their distance is less than _d_ 2. Otherwise, the two triangles can
only impose normal forces, but not frictional damping forces. This design choice preserves the
property of non-vanishing gradient, but also allows efficient evaluation. Finally, we emphasize that
as _µ_ → 0, both our contact potential and frictional damping term converges to the exact frictional
contact model with Coulomb friction.


20


A.4 TWICE-DIFFERENTIABILITY OF IPC


We show that the IPC contact model (Li et al.,
2020) is differentiable but not twice differen
0.5

tiable. Let us consider a simple 2D case, where
the only geometric primitive pairs that incur
collision potential is between a point and a line- 0.0
segment. Let us now assume the toy example
with a single geometric primitive: a line segment with two end points located at ( 1 _,_ 0) and −0.5
( 2 _,_ 0 ) and a point moving on the line ( _x,_ 1 )
with _x_ ∈[0 _,_ 3]. The IPC potential for this toy 0 1 2 3

x

example is formulated as: P( _x_ ) = − log( _d_ ( _x_ )),
where _d_ ( _x_ ) is the distance between the point Figure 12: We plot the value of _∂d_ ( _x_ )/ _∂x_ when
and the line-segment, parameterized by a sin- _x_ ∈[0 _,_ 3] in a toy example under the IPC contact
gle scalar _x_ . Clearly, the differentiability of P model.
relies on the differentiability of _d_ ( _x_ ). As analyzed in Li et al. (2020), _d_ ( _x_ ) is differentiable,
so _∂d_ ( _x_ )/ _∂x_ is well-defined. In Figure 12, we plot the value of _∂d_ ( _x_ )/ _∂x_ when _x_ ∈[0 _,_ 3]. Clearly,
_∂d_ ( _x_ )/ _∂x_ is not a differentiable function, so we conclude that P( _x_ ) cannot be twice-differentiable.
The non-smoothness is due to switching between different Voronoi regions. When _x_ ≤ 1 or _x_ ≥ 2,
the closest point on the line-segment is a vertex. Instead, when _x_ ∈(1 _,_ 2), the closest point lies
interior to the line segment.


0.5


0.0


−0.5


0 1 2 3
x


Figure 12: We plot the value of _∂d_ ( _x_ )/ _∂x_ when
_x_ ∈[0 _,_ 3] in a toy example under the IPC contact
model.


A.5 HIERARCHICAL POTENTIAL BLENDING


As the same in Li et al. (2020); Ye et al. (2025), we use the Newton method equipped with line-search
to solve the optimization problem of Equation 1, where we recursively accumulate the contact
potential, its gradient, and Hessian based on BSH mentioned in Section 5 to accelerate the contact
potential evaluation. Specifically, we establish a BSH for each joint in the scene and then recursively
perform collision detection based on BSH while accumulating potential. When detecting a potential
collision pair where nodes I and J are both leaf nodes, or the distance between two nodes is greater
than _d_ 2, it reaches the end of the recursion and return P _d_ [I∪J] 1→ _d_ 2 [and] [P] _d_ [I∪J] 2 respectively. Otherwise,
we continue to search for their child nodes and blend it’s potential using Equation 8. We summarize
the algorithm of hierarchical potential blending in Algorithm 1.


**Algorithm 1** Hierarchical Potential Blending


**function** PROCESS PAIR(I _,_ J )


P _d_ [I∪J] 1 ← 0
P _d_ [I∪J] 2 ←P _c_ [I∪J] - _Equation_ 7
**if** ∥ _x_ I − _x_ J ∥> _d_ 2 **then**


P [I∪J]
BSH [←P] _d_ [I∪J] 2
**return**
**if** I = _ti_ is leaf and J = _tj_ is leaf **then**


P _d_ [I∪J] 1 ←P _[t][i]_ [∪] _[t][j]_  - _Section_ 4
Plug P _d_ 1 _,_ P _d_ 2 in Equation 7 for P _d_ 1→ _d_ 2
P [I∪J]
BSH [←P] _d_ [I∪J] 1→ _d_ 2
**return**
**for** each pair < I _c,_ J _c_ - in < I _,_ J >’s children **do**


PROCESS PAIR(I _c,_ J _c_ )
P [I∪J] + = P [I] _[c]_ [∪J] _[c]_
_d_ 1 BSH
Plug P _d_ 1 _,_ P _d_ 2 in Equation 4 for P _d_ 1→ _d_ 2
P [I∪J]
BSH [←P] _d_ [I∪J] 1→ _d_ 2
**return**


21


A.6 EXPERIMENTAL DETAILS


In this section, we provide experimental details and extended evaluations.


Parameter Billiards Push Ant-Push Sort Gather Gather-Bunny


trajectory horizon _H_ 100 200 240 700 300 550
receding horizon _h_ / 48 / 16 32 32
potential coefficient _µ_ 1 _e_ [−][7] 1 _e_ [−][6] 1 _e_ [−][8] 3 _e_ [−][8] 3 _e_ [−][8] 5 _e_ [−][9]

_α_ learning rate 3 _e_ [−][2] 1 _e_ [−][2] 3 _e_ [−][2] 1 _e_ [−][2] 1 _e_ [−][2] 1 _e_ [−][2]

Adam ( _β_ 1 _, β_ 2) (0.3,0.5) (0.3,0.5) (0.3,0.5) (0.3,0.5) (0.3,0.5) (0.3,0.5)
number of iterations 400 50 100 100 60 60
degrees of freedom 96 12 22 66 66 66
timesteps ∆ _t_ 0.04 0.04 0.02 0.04 0.04 0.04


Table 3: Parameter settings for different benchmarks. Here, the trajectory horizon _H_ represents the total
number of frames in the entire trajectory. For benchmarks requiring receding horizon execution, the receding
horizon _h_ represents the number of frames in the sub-trajectory. The potential coefficient _µ_ represents the
contact energy coefficient, where a smaller _µ_ indicates a more physically accurate contact mechanism. _α_, the
learning rate, denotes the step size for optimization, and the number of iterations specifies the total optimization
steps. For all the benchmarks, we set the hyper-parameter of the Adam optimizer (Kingma & Ba, 2014), _β_ 1
and _β_ 2, to small values, which helps escaping from local minima. ∆ _t_ represents the timestep for each frame.
Our simulator allows stable, penetration-free simulation even under relatively large timesteps.


A.6.1 BASELINES


We compare our method against three state-of-the-art baselines. The first is the IPC contact
model (Li et al., 2020), employed in the differentiable simulator of (Huang et al., 2024). However, due to the model’s lack of twice-differentiability, the implicit function theorem does not apply.
As a result, Huang et al. (2024) resort to a few iterations of gradient descent to approximately
solve Equation 1. While this yields usable gradient information, it is well known that the gradients
can vanish when the interacting geometric primitives are not in close proximity. The second baseline
is SDRS contact model (Ye et al., 2025), it’s a differentiable model with twice-differentiability. But
similar to IPC, the gradient also vanish when the interacting convex hulls are far apart. The third
baseline is the Gradient Bundle (GB) method (Suh et al., 2022b), which addresses gradient vanishing through sampling, evaluated in practice via Monte Carlo methods. However, when primitives
are far apart, the likelihood of sampling a non-zero gradient decreases significantly. Consequently,
gradients obtained from GB can be both noisy and prone to vanishing with high probability.


A.6.2 BENCHMARK DETAILS


We implement a full-featured rigid body simulator based on our novel contact model. Each benchmark scenario includes controlled objects and target objects. The controlled objects are actuated
using a built-in PD controller, and the objective across all benchmarks is to manipulate collisions
and contacts to move the target objects to their designated spatial positions. In all experiments, we
use the following loss to measure the progress of optimizers:

ReLU(∥ _x_ COM − _x_ [⋆] COM [∥][2][ −] _[ϵ]_ target [2] [)] _[,]_

where _x_ COM and _x_ [⋆] COM [are] [the] [position] [and] [desired] [position] [of] [the] [center] [of] [mass] [of] [some] [target]
object. _ϵ_ target is the error coefficient, indicating that tasks are considered successful for certain objects
when they are within _ϵ_ target of the goal. The statistics of benchmarks are summarized in Table 3.


**Billiards** In this task, the indices of the two target balls and their target locations are randomly selected. The objective is to control a distinct red ball so that it strikes the target balls through contact,
moving them to their respective target positions. Each ball has 6 degrees of freedom, resulting in a
total system of 96 degrees of freedom. We only control the initial horizontal positions and velocities
of the red ball, corresponding to 4 control dimensions.


**Push** In this benchmark, the task goal is to control a rod to push a box to the target region. The
system consists of 2 objects with a total of 12 degrees of freedom. At each timestep, a continuous


22


6-dimensional control signal is generated to control the rod. The control signal at each timestep is
obtained by solving a receding-horizon optimization problem.


**Ant-Push** In this benchmark, our goal is to drive the ant robot to move and push the box to the
target position. The ant consists of a base, four large legs, and four small legs. The base includes 3
translational degrees of freedom and 1 rotational degree of freedom, while the upper legs and lower
legs are connected using ball joints and hinge joints, respectively. As a result, the ant has a total of
16 degrees of freedom in the kinematic state. Combined with the 6 degrees of freedom of the box,
the system has a total of 22 degrees of freedom in the kinematic state, of which we can control 12
degrees of freedom in the ant’s legs. We use 4 accumulated sine wave signals to parameterize our
controller for each degree of freedom of the ant’s legs as done in Hu et al. (2019). In this case, the
decision variables are the amplitude, frequency, and phase of the sine waves.


**Sort** In this benchmark, two types of cubes are mixed together on the ground. Use a rod to push
each type of cube to its target location without mixing with each other. We set up 10 target cubes
and one rod, so the system kinematic state has a total of 66 degrees of freedom and we can control
the 3 translational degrees of freedom of the rod via PD controller.


**Gather** In this task, 10 cubes are randomly put on the ground. Use a rod to collect all the objects
into one area. Again the task has a total of 66 degrees of freedom and we control 3 translational
and 1 horizontal rotational degrees of freedom of the rod. To further validate the efficiency of our
method, we can handle objects with more complex geometries in this scenario. We replace the cubes
with bunnies and successfully complete the gather task, we call this benchmark Gather-Bunny.


A.6.3 ABLATION STUDY OF CONTACT PROPERTY


In Figure 3, we discuss the various properties that a contact model needs to possess. Here, we will
analyze the impact of these different properties on the contact model’s performance in simulation
and policy learning. Collision models without Barrier-Form would allow intersection between objects. It is well known that the distance function between objects is non-smooth when the objects
are in collision. Therefore, if we take away Barrier-Form, then Smoothness will be violated automatically. Non-prehensile requires that normal collision forces between objects are pushing them
apart, instead of pulling them together. Almost all existing collision models satisfy this property.
We conduct an ablation study on Non-vanishing. Since the only difference between our method
and SDRS model (Ye et al., 2025) is the additional satisfaction of Non-prehensile, the results of
this ablation study can be observed in the comparison between our method and SDRS across various benchmarks in Section 6. This demonstrates the necessity of satisfying this property for policy
convergence when objects are far apart.


We note that it is possible to take away Smoothness alone and conduct an ablation study. To this
end, we modify our Equation 3 to use Li et al. (2020) instead of Ye et al. (2025). In other words,
we combine Li et al. (2020) with a globally supported barrier function instead of the original locally
supported version. By doing so, we still have Non-vanishing but fails Smoothness. We tested the
results on the Gather and Push benchmarks in Figure 13, showing that without Smoothness, the
policy convergence speed is significantly slower or may even fail to converge completely.


A.6.4 INFLUENCE OF STIFFNESS AND TIMESTEP


It is well known that the stiffness of a system and the timestep _δt_ value can affect the evaluation
of gradients. Here, we validate the sensitivity of our contact model to stiffness and timestep. We
demonstrate the impact of stiffness and timestep on policy optimization by verifying the convergence of strategies on the Push task. In our contact model, stiffness is determined by the contact
coefficient _µ_, where a smaller _µ_ corresponds to a stronger system stiffness. Firstly, we fix _µ_ = 1 _e_ [−][7]
and record the simulation time required for strategy convergence under different timesteps. We observe that strategies converge faster with larger timesteps. This is because, when using MPC for
decision-making, a larger timestep results in fewer frames for the same horizon length, reducing
the number of gradient multiplications and avoiding the issues of gradient explosion and vanishing
gradients that are common in optimization problems. Since our method is unconditionally robust to
timestep size, for a specific task, we should choose the largest timestep within a reasonable range to


23


#Steps


4


3


2


1


0


#Steps


6


4


2


0


Figure 13: The convergency history with or without Smoothness on Push (left) and Gather (right)
benchmark.


accelerate strategy convergence. Secondly, we fix _δt_ = 0 _._ 04 _s_ and record the simulation time required
for strategy convergence under different _µ_ values. We find that only when the system stiffness is extremely high ( _µ_ = 1 _e_ [−][10] ), leading to drastic gradient changes caused by stiffness, does the strategy
convergence noticeably slow down. These results are shown in Table 4 and Table 5, respectively.


Timestep _δt_ 0 _._ 01 _s_ 0 _._ 02 _s_ 0 _._ 04 _s_ 0 _._ 08 _s_


Simulation Time to Converge 5 _._ 76 _s_ 3 _._ 28 _s_ 2 _._ 96 _s_ 2 _._ 80 _s_


Table 4: The simulation time required for policy convergence on the Push task under different _δt_ .


Contact Coefficient _µ_ 1 _e_ [−][6] 1 _e_ [−][7] 1 _e_ [−][8] 1 _e_ [−][10]


Simulation Time to Converge 3 _._ 20 _s_ 3 _._ 28 _s_ 3 _._ 24 _s_ 4 _._ 80 _s_


Table 5: The simulation time required for policy convergence on the Push task under different _µ_ .


24


A.6.5 COMPUTATIONAL EFFICIENCY


growth of brute-force computation and the near-linear scaling of our method. Interestingly, the IPC
model (Li et al., 2020) maintained a consistent per-frame cost of approximately 0.5s across all resolutions. This is attributed to the optimizer requiring fewer Newton iterations as mesh resolution
increases, offsetting the higher evaluation cost of the contact model. We have also compared our
performance using more complex, non-convex geometric shapes. As illustrated in Figure 14, we
replace the cubes in our Gather benchmark with bunny meshes, with 2784 triangles in total. In
this benchmark, named Gather-Bunny, we compare the average per-frame cost using brute-force
computation, our method, and the IPC model, where the cost is 309.62s, 3.89s and 3.14s respectively. And in Figure 15, we have designed a more challenging articulated manipulation task, named
Franka Push, where we use a 7-degree-of-freedom Franka robotic arm to push a bunny to a speci


#Steps


25