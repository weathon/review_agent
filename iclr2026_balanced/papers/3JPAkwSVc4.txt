# QUOTIENT-SPACE DIFFUSION MODELS


**Yixian Xu** [1] _[∗]_ **,** **Yusong Wang** [2] _[,]_ [5] _[∗]_ **,** **Shengjie Luo** [1] **,** **Kaiyuan Gao** [3] **,** **Tianyu He** [4] **,**
**Di He** [1] _[†]_ **,** **Chang Liu** [5] _[†]_

1 State Key Laboratory of General Artificial Intelligence, Peking University, Beijing, China
2 State Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Institute of Artificial
Intelligence and Robotics, Xi’an Jiaotong University
3 Huazhong University of Science and Technology
4 Microsoft Research Asia
5 Zhongguancun Academy


ABSTRACT


Diffusion-based generative models have reformed generative AI, and have enabled
new capabilities in the science domain, for example, generating 3D structures of
molecules. Due to the intrinsic problem structure of certain tasks, there is often a
_symmetry_ in the system, which identifies objects that can be converted by a group
action as equivalent, hence the target distribution is essentially defined on the _quo-_
_tient space_ with respect to the group. In this work, we establish a formal framework for diffusion modeling on a general quotient space, and apply it to molecular
structure generation which follows the special Euclidean group SE(3) symmetry.
The framework reduces the necessity of learning the component corresponding
to the group action, hence simplifies learning difficulty over conventional groupequivariant diffusion models, and the sampler guarantees recovering the target
distribution, while heuristic alignment strategies lack proper samplers. The arguments are empirically validated on structure generation for small molecules and
proteins, indicating that the principled quotient-space diffusion model provides a
new framework that outperforms previous symmetry treatments.


1 INTRODUCTION


Diffusion models have emerged as the dominant approach for modeling distributions in highdimensional spaces. Building on their success in real-world domains such as images (Ho et al.,
2020; Song et al., 2021), audios (Kong et al., 2021; Evans et al., 2024), and videos (Ho et al., 2022;
Li et al., 2023), diffusion models are now increasingly adopted in scientific applications, ranging
from fluid field solving (Bastek et al., 2025), electronic structure prediction (Kim et al., 2025),
molecular structure generation (Xu et al., 2022; Abramson et al., 2024; Hassan et al., 2024; Geffner
et al., 2025), and thermodynamic ensemble modeling (Zheng et al., 2024; Lewis et al., 2025).


Compared with general tasks, scientific applications often exhibit inherent _symmetry_ structures,
wherein objects that can be related through specific transformations are regarded as equivalent.
Consider molecular structure generation as a representative example. A molecular structure can
be represented as a vector in R [3] _[N]_ by concatenating the 3D coordinates of its _N_ atoms. However,
because the choice of coordinate system is arbitrary, vectors in R [3] _[N]_ that differ only by a global 3D
translation or rotation of all atoms correspond to the same underlying structure. Mathematically,
such transformations typically form a Lie group — for example, the special Euclidean group SE(3)
in the case of molecular structures, which formally characterizes the symmetry.


The common treatment is putting the target distribution in the original space but assigning the same
probability to equivalent objects, resulting in a distribution that is invariant under group action.
This can be implemented by augmenting training data by applying randomly chosen group actions
(Abramson et al., 2024), or using a group equivariant model (Xu et al., 2022; Hoogeboom et al.,
2022b), which guarantees invariance if the starting prior distribution is invariant (K¨ohler et al., 2020).
Nevertheless, we shall show that this treatment still has room to improve, as the neural network
model, which is intended for updating the sample in each diffusion simulation step, still needs to


_∗_ Equal contribution.

_†_ Correspondence to: Di He _<_ dihe@pku.edu.cn _>_, Chang Liu _<_ liuchang@bza.edu.cn _>_ .


1


Table 1: Comparison among different training strategies in presence of a symmetry group. Learning
difficulty is measured by whether the need to predict in the equivalent degrees of freedom (DoFs),
induced by the group actions, is removed, and (if not) whether the variance on the equivalent DoFs
is removed. Sampling compatibility means whether there is a sampler that exactly reproduces the
target distribution. The denoising form of diffusion model **D** _θ_ is used to express the loss functions,
where _A_ **y** ( **x** ) (Eq. (11)) represents aligning **x** towards **y**, and _θ_ [¯] denotes treating _θ_ as constant
( _i.e._, stop-gradient). The conclusions hold using either an equivariant architecture or a general
architecture with data augmentation. See Sec. 3.4 for details.


Reduction of learning difficulty
Sampling
Removal of Removal of variance compatibility
equivalent DoFs on equivalent DoFs


Training strategy
for **D** _θ_


Optimal
solution
of **D** _θ_


EConventional loss _∥_ **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1 _∥_ [2] E[ **x** 1 _|_ **x** _t_ ] ✗ ✗ ✓

E _∥_ GeoDiff alignment loss **D** _θ_ ( **x** _t, t_ ) _−A_ **x** _t_ ( **x** 1) _∥_ [2] E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ] ✗ ✓ ✗

E�� **D** _θ_ AF3 alignment loss( **x** _t, t_ ) _−A_ **D** _θ_ ¯ [(] **[x]** _[t][,t]_ [)][(] **[x]** [1][)] ��2 for arbitrary _g ·_ E[ _A_ **x** _t_ ( **x** _g_ 1) _|∈G_ **x** _t_ ] ✓ ✓ ✗

quotient-space diffusion loss E[ _P_ **x** _t_ ( **x** 1) _|_ **x** _t_ ] + **v** _[V]_ ✓ ✓ ✓
E _∥P_ **x** _t_ ( **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1) _∥_ [2] for arbitrary **v** _[V]_ _∈_ Ker( _P_ **x** _t_ )


learn a _specific_ movement within the equivalent class ( _e.g._, rotating a molecular structure), which
is unnecessary as _any_ such a movement does not update the intrinsic system state ( _e.g._, the shape
of a molecular structure) hence is acceptable. In hope to remove this redundancy, there are a few
heuristic treatments using alignment, _i.e._, adjusting the prediction target within its equivalent class
according to a reference to remove these equivalent degrees of freedom (Xu et al., 2022; Abramson
et al., 2024). But we find that the corresponding sampling process becomes incompatible with such
training strategies, even with heuristic fix attempts (Wohlwend et al., 2025).


In this work, we develop a principled approach to building a diffusion model considering the intrinsic
symmetry of the system. In particular, we leverage the concept of _quotient_ _space_, in which a set
of equivalent objects (equivalent class) are treated as one element. It is the formal mathematical
construction that reflects the intrinsic variability of the system. We first derive the diffusion process
on a general quotient space based on the correspondence between the Wiener processes on the two
spaces. Considering that the quotient space is generally not Euclidean, hence it is hard to directly
carry out a simulation on it, we further leverage the mathematical construction of horizontal lift to
induce a diffusion process back in the original space that can easily implement the quotient-space
diffusion process. The resulting process effectively amounts to projecting the update vector in the
original diffusion process onto the subspace that does not induce a movement within the equivalent
class ( _e.g._, rotation). We show that this process _guarantees producing the correct target distribution_,
meanwhile _reduces learning difficulty_ by removing the necessity to learn a specific movement within
an equivalent class. A visualization example in the 2-dimensional plane with SO(2) symmetry is
shown in Fig. 1. In this example, the lifted process only has radial movements (Fig. 1(Left)) as
the quotient space R [2] _/_ SO(2) is isomorphic to the half real line and recovers the correct target
distribution as conventional equivariant diffusion models (Fig. 1(Middle, Right)). A conceptual
comparison with existing methods is shown in Table 1. The quotient-space diffusion admits either
an equivariant model or a general model with data augmentation.


As a representative application, we deduce the specific training and sampling algorithms in the
R [3] _[N]_ _/_ SE(3) scenario for molecular structure generation, which relaxes the model from learning a
translation and rotation movement, while the sampling process keeps the structure with constant
position and orientation. We study the empirical performance of quotient-space diffusion models
on small molecule structure generation and protein backbone design tasks. The results show that
our methods can consistently improve the generation performance in these applications over conventional equivariant diffusion models and using alignment strategies. Our method achieves 9%23% relative improvements of ET-Flow (Hassan et al., 2024) on GEOM-QM9 and GEOM-DRUGS
datasets, surpassing previous heuristic alignment methods. For the protein structure generation task,
our method surpasses the state-of-the-art Prote´ına model (Geffner et al., 2025) with the same parameter scale (60M) in a large margin and also outperforms the much larger model (200M) on most key
distributional metrics.


2


Figure 1: A motivative illustration highlighting the behavior of the quotient-space diffusion model
against the conventional equivariant diffusion model for modeling a distribution on R [2] (as _M_ )
with SO(2) (as _G_ ) symmetry, whose density is represented by the gray scale. **(Left)** SDE sampling
trajectories by the two diffusion models. The same color indicates the same starting point (the
round dot). The quotient-space diffusion model moves each sample only along the ray from the
origin, which can be understood as only traversing the quotient space R [2] _/_ SO(2), _i.e._, traversing
over origin-centered concentric circles, without moving within an equivalent class, _i.e._, an origincentered circle. The conventional equivariant diffusion model moves each sample over the whole
R [2] space, requiring subtler simulation treatment. **(Middle)** Samples generated by the conventional
equivariant diffusion model. **(Right)** Samples generated by the quotient-space diffusion model,
which also recovers the data distribution as guaranteed. Moreover, the quotient-space diffusion
simplifies learning difficulty: the neural network does not need to learn anything in the output
subspace that is responsible for intra-equivalent-class movement (Eq. (10)).


2 BACKGROUND


2.1 DIFFUSION-BASED GENERATIVE MODELS ON EUCLIDEAN SPACE


The main idea of diffusion models is to construct a step-by-step transformation from a simple prior
distribution to a complex target distribution. In this paper, we follow the Stochastic Interpolant
framework (Albergo et al., 2023), which unifies diffusion models and flow matching models (Lipman et al., 2023; Liu et al., 2023). Let _p_ target( **x** ) be the target distribution. The following linear
interpolation is constructed:

**x** _t_ = _αt_ **x** 0 + _βt_ **x** 1 + _γt_ **ϵ** _,_ ( **x** 0 _,_ **x** 1) _∼_ _p_ joint _,_ **ϵ** _∼N_ (0 _,_ **I** ) _,_ _t ∈_ [0 _,_ 1] (1)

where _p_ joint is a pre-defined joint distribution of ( **x** 0 _,_ **x** 1) with marginals **x** 0 _∼_ _p_ prior and **x** 1 _∼_
_p_ target. The coefficients _αt, βt, γt_ satisfy the boundary conditions _α_ 0 = 1, _β_ 0 = 0, _γ_ 0 = 0, and
_α_ 1 = 0, _β_ 1 = 1, _γ_ 1 = 0. Under these conditions, the following ordinary differential equation (ODE)
can transform _p_ prior to _p_ target (Albergo et al., 2023, Cor. 2.18):

d **x** _t_ = **v** ( **x** _t, t_ ) d _t,_ where **v** ( **x** _t, t_ ) := E[ _αt_ _[′]_ **[x]** [0] [+] _[ β]_ _t_ _[′]_ **[x]** [1] [+] _[ γ]_ _t_ _[′]_ **[ϵ]** _[ |]_ **[ x]** _[t]_ []] _[.]_ (2)

The velocity vector field **v** ( **x** _t, t_ ) is typically trained with the objective: _L_ ( _θ_ ) :=
E _p_ ( _t_ ) _w_ ( _t_ )E _p_ joint( **x** 0 _,_ **x** 1) _p_ ( **ϵ** ) _∥_ **v** _θ_ ( **x** _t, t_ ) _−_ ( _αt_ _[′]_ **[x]** [0] [+] _[ β]_ _t_ _[′]_ **[x]** [1] [+] _[ γ]_ _t_ _[′]_ **[ϵ]** [)] _[∥]_ [2][, where the prime denotes the time]
derivative, and _p_ ( _t_ ) and _w_ ( _t_ ) control the sampling distribution and weighting over time. There is
also a stochastic process for sample generation, given by :

d **x** _t_ = ( **v** ( **x** _t, t_ ) + _ηt_ **s** ( **x** _t, t_ )) d _t_ + ~~�~~ 2 _ηt_ d **w** _t,_ where **s** ( **x** _t, t_ ) := _∇_ **x** _t_ log _pt_ ( **x** _t_ ) (3)

is called the score function, and _ηt_ _≥_ 0 is a non-negative smooth function (Albergo et al., 2023,
Cor. 2.10). In the special case where _p_ prior = _N_ ( **0** _,_ **I** ) (the _one-sided stochastic interpolant_ (Albergo
et al., 2023, Def. 3.4)), contributions of **x** 0 and **ϵ** can be combined as **x** _t_ = _α_ ˆ _t_ **ϵ** + _βt_ **x** 1, where _α_ ˆ _t_ =

- _αt_ [2] [+] _[ γ]_ _t_ [2][, and the score function can be expressed by the velocity field:] **[s]** [(] **[x]** _[t][, t]_ [) =] _[β]_ _α_ ˆ _t_ _[′]_ _t_ **[x]** (ˆ _α_ _[t][−][′]_ _t_ _[β][β][t][t][−]_ **[v]** _[α]_ [(][ˆ] **[x]** _[t][t][β][,t]_ _t_ _[′]_ [)][)] [.]

A convenient variant to formulate the learning task is to define the **v** _θ_ ( **x** _t, t_ ) model with a neural
network **D** _θ_ ( **x** _t, t_ ) which reformulates the objective:


_t_ **[x]** _[t]_ _[−]_ [(ˆ] _[α]_ _t_ _[′]_ _[β][t]_ _[−]_ _[α]_ [ˆ] _[t][β]_ _t_ _[′]_ [)] **[D]** _[θ]_ [(] **[x]** _[t][, t]_ [)]
**v** _θ_ ( **x** _t, t_ ) := _[α]_ [ˆ] _[′]_ _,_ (4)
_α_ ˆ _t_


_L_ ( _θ_ ) := E _p_ ( _t_ ) _w_ ( _t_ ) [(ˆ] _[α]_ _t_ _[′]_ _[β][t]_ _[−]_ _[α]_ [ˆ] _[t][β]_ _t_ _[′]_ [)][2]


_α_ ˆ _t_ [2] _[t]_ _t_ E _p_ ( **x** 1 _,_ **x** _t_ ) _∥_ **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1 _∥_ [2] _,_ (5)


3


where _p_ ( **x** 1 _,_ **x** _t_ ) is derived from Eq. (1) by integrating out **x** 0 and **ϵ** . This objective conveys the
intuition of recovering the clean-data sample **x** 1 from a noisy sample **x** _t_, hence **D** _θ_ ( **x** _t, t_ ) is called
a denoising model and suits prevalent architectures. We adopt this form of a diffusion model below.


2.2 FROM EUCLIDEAN SPACE TO QUOTIENT MANIFOLD


Tasks in scientific domains often involve inherent symmetry, where objects related by certain transformations are considered equivalent. A formal and inclusive description of symmetry in a system
requires both the geometry of the configuration space and the algebraic structure of the transformations, which leads to the concepts of manifolds and Lie groups.


**Manifold and Lie groups.** A (smooth) manifold is a geometric object that generalizes the Euclidean
space to allow spatial heterogeneity. Typically, a manifold is endowed with a Riemannian metric,
_i.e._, an inner product in each tangent space, which leads to common concepts like curve length, distance, measure, gradient, Laplacian, and Wiener process on the manifold (Appx. B.1). Symmetries
are formally represented by transformations that connect equivalent ( _i.e._, symmetric) objects, which
constitute a group. A continuously-parameterized group that is also a manifold is called a Lie group.


We consider the general case where the configuration space of the system is an _M_ -dimensional
Riemannian manifold _M_ . The symmetry of the system is represented by a _G_ -dimensional Lie
group _G_ acting on _M_ . A distribution _p_ on _M_ is said _G_ -invariant if _p_ ( _g ·_ **x** ) = _p_ ( **x** ), _∀g_ _∈G,_ **x** _∈M_ .
This invariance implies that all equivalent points _{g ·_ **x** _|_ _g_ _∈G}_, collectively called an equivalent
class, are assigned with the same probability.


**Quotient** **space.** The symmetry group defines an equivalent relation in _M_, _i.e._, **x** 1 and **x** 2 are
equivalent, if there exists a group action _g_ _∈G_ such that _g ·_ **x** 1 = **x** 2, which is indeed an equivalent
relation due to properties of a group. The quotient space _Q_ := _M/G_ treats equivalent objects under
the action of _G_ as one element, hence reflects the intrinsic variability of the system. There is a
natural mapping called the projection connecting the two spaces: _π_ ( **x** ) := _{g ·_ **x** _|_ _g_ _∈G}_ . Under
appropriate conditions, the quotient space is a smooth manifold with dimension _M_ _−_ _G_ (Appx. C).
However, defining a diffusion process on this space is non-trivial, necessitating the extension of
“velocity” and Wiener process from Euclidean space to the manifold.


**Tangent vector.** On a manifold _M_, the velocity of a process at a certain point **x** is represented as
a tangent vector at **x**, intuitively representing an infinitesimal movement. All tangent vectors at **x**
constitute a linear space _T_ **x** _M_ called the tangent space at **x** . Since a manifold is typically curved,
tangent spaces at different points are regarded as different linear spaces, but with a transformation on
the manifold, _e.g._, a group action _Lg_ : **x** _�→_ _g ·_ **x**, an associated mapping ( _Lg_ ) _∗_ **x** : _T_ **x** _M →_ _Tg·_ **x** _M_
between the tangent spaces can be defined by linking infinitesimal movements around **x** and around
_g ·_ **x** by _Lg_ (Appx. B). With this construction, we can define a _G_ -equivariant vector field on _M_ if
it is unchanged under the group action: ( _Lg_ ) _∗_ **x** ( **vx** ) = **v** _g·_ **x** . The projection mapping _π_ naturally
induces a projection of tangent vectors onto the quotient space by _π∗_ **x** : _T_ **x** _M →_ _Tπ_ ( **x** ) _Q_ .


**Wiener process on a manifold.** In Euclidean space, the Wiener process is generated by the Laplacian operator [1] 2 [∆][.] [The] [Laplace-Beltrami] [operator,] [defined] [from] [a] [Riemannian] [metric,] [serves] [as] [a]

counterpart on a manifold, and defines the Wiener process to the manifold. Under a symmetry group
_G_, we require a meaningful stochastic process on the manifold _M_ as _G_ -invariant, meaning that its
marginal distribution is _G_ -invariant at any time step. See Appx. B for details.


3 METHODS


As the quotient space represents the “essential states” of a system with symmetry, a principled diffusion model for the system is expected to be built on it. In this section, we unroll the development
of the quotient-space diffusion model by deriving the projected diffusion process onto the quotient
space, then lift it back into the total space ( _i.e._, the original space) for convenient implementation.
We then derive the specialization in the R [3] _[N]_ _/_ SE(3) case for molecular structure generation, followed by training and sampling algorithms. We highlight the merit of the quotient-space diffusion
in reducing training difficulty and sampler soundness with a comparative analysis with existing
treatments considering symmetry.


4


3.1 DIFFUSION PROCESS ON A GENERAL QUOTIENT SPACE


If the diffusion process in _M_ is _G_ -invariant, the distribution at any time step can be viewed as a
distribution in the quotient space _Q_, then we can view the process as a stochastic process in _Q_ . By
leveraging the projection mapping _π_ : _M_ _→Q_, we can map a diffusion process _{_ **x** _t}t∈_ [0 _,T_ ] in _M_
(Eq. (3)) onto the quotient space as _{_ **y** _t_ := _π_ ( **x** _t_ ) _}t∈_ [0 _,T_ ]. This is a stochastic process on _Q_, but
its expression as a diffusion process on _Q_ using specifiers defining the diffusion process of **x** _t_ is
desired. The following theorem gives an explicit answer.
**Theorem 1.** _Assume {_ **x** _t}t∈_ [0 _,T_ ] _is a diffusion process on M, specified by the following SDE:_


d **x** _t_ = **b** _t_ ( **x** _t_ ) d _t_ + _σt_ d **w** _t,_ **x** 0 _∼_ _p_ prior _,_ (6)

_where_ **b** _t_ _is_ _a_ _G-equivariant_ _time-dependent_ _vector_ _field_ _on_ _M,_ **w** _t_ _is_ _the_ _Wiener_ _process_ _on_ _M_
_that is also G-invariant, and p_ prior _is a G-invariant distribution._ _Then the projected process {_ **y** _t_ :=
_π_ ( **x** _t_ ) _}t∈_ [0 _,T_ ] _onto the quotient space Q_ := _M/G_ _is the solution to the following SDE:_


       -       d **y** _t_ = ( _π∗_ **b** _t_ )( **y** _t_ ) _−_ _[σ]_ _t_ [2] d _t_ + _σt_ d **ω** _t,_ **y** 0 _∼_ _π_ # _p_ prior _,_ (7)
2 **[h]** [(] **[y]** _[t]_ [)]


_where_ _π∗_ **b** _t_ _is_ _the_ _projected_ _vector_ _field_ _of_ **b** _t_ _onto_ _Q_ _induced_ _by_ _π,_ **h** ( **y** ) _is_ _the_ _mean_ _curvature_
_vector field of Q reflecting the geometry of Q,_ **ω** _t_ _is the Wiener process on Q, and π_ # _p_ prior _is the_
_pushed-forward distribution of p_ prior _(_ i.e. _,_ **y** 0 = _π_ ( **x** 0) _where_ **x** 0 _∼_ _p_ prior _)._


See Appx. D.1 for formal definitions of the concepts and
the proof. Thm. 1 shows that the projected process is
indeed a diffusion process on _Q_, which consists of the
projected vector field and corresponding Wiener diffusion
process, and perhaps unexpectedly, an additional vector field reflecting the curvature of _Q_ . As the quotient
space squeezes an equivalent class as one point, a process viewed on the quotient space should accommodate
for the change of the volume of the equivalent class along
the movement. This additional vector is the gradient ( _i.e._,
the change rates in all movement directions) of the volume of the equivalent class.


Although the diffusion process on the quotient space is
defined, it is not convenient to simulate it in the quotient Figure 2: Illustration of the relation bespace directly due to the non-trivial geometric structure tween the total space _M_ and the quoof _Q_ . Nevertheless, the quotient-space diffusion enables tient space _Q_ and the correspondence of
us a principled view to reduce the unnecessary movement tangent vectors among them.
within equivalent classes. A key observation from Thm. 1
is that if **b** 1 = **v** + **b** 2 where **vx** _∈_ Ker _π∗_ **x** := _{_ **v** _∈_ _T_ **x** _M_ _|_ _π∗_ **x** ( **v** ) = **0** _}, ∀_ **x** _∈M_, then the
corresponding SDE in Eq. (6) has the same projection in the quotient space. This implies that the
components in Ker _π∗_ **x** are not really necessary.


For better characterization of the necessary component, we focus on the tangent space of _M_ at
**x** . The tangent space _T_ **x** _M_ is a linear space with the same dimensionality as _M_ . Define the
vertical space _V_ **x** := Ker _π∗_ **x** ( _G_ -dimensional) corresponding to the infinitesimal action of the group
_G_ . Since _T_ **x** _M_ has an inner product (because _M_ is a Riemannian manifold), we can define the
horizontal space _H_ **x** := (Ker _π∗_ **x** ) _[⊥]_ as the orthogonal complement of _V_ **x** . Then any tangent vector
in _T_ **x** _M_ has an orthonormal decomposition **v** = **v** _[V]_ + **v** _[H]_, where **v** _[V]_, **v** _[H]_ is the vertical and
horizontal component respectively; see Fig. 2 for visualization. Thus **v** _[H]_ is the necessary part of the
vector field **v** .


Thanks to the quotient structure, we can leverage a correspondence between the diffusion process on
_M_ and _Q_ . For a diffusion process **y** _t_, there exists a diffusion process ˜ **x** _t_ in _M_ such that _π_ (˜ **x** _t_ ) = **y** _t_
and **x** ˜ _t_ only has horizontal movement, which is called the horizontal lift of **y** _t_ (see Appx. D.2 for
formal definitions). The horizontal lift of **y** _t_ is given explicitly in the following theorem.
**Theorem 2.** _The horizontal lift of Eq._ (7) _has the following explicit expression:_


       -        d˜ **x** _t_ = _P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _−_ _[σ]_ 2 _t_ [2] **h** ˜(˜ **x** _t_ ) d _t_ + _σt_ d ˜ **w** _t,_ **x** ˜0 _∼_ _p_ prior _,_ (8)


5


_where P_ **x** ( **v** ) := **v** _[H]_ _is the horizontal projection in the tangent space of M,_ **h** [˜] _is the horizontal lift_
_of the mean curvature vector_ **h** _in Eq._ (7) _,_ **w** ˜ _t is the horizontal lift of the Wiener process on Q._


See Appx. D.2 for the proof. Comparing the expression between Eq. (6) and Eq. (8), we can observe
that the lifted process is not simply given by adding a horizontal projection _P_ **x** on each term of the
SDE, and an additional term depending on the curvature of the quotient space arises. This term
arises in Eq. (7) and remains after the horizontal lift. Intuitively, this term corrects the possible side
effects by projection so that the resulting diffusion process still yields the desired target distribution.
The horizontal projection _P_ **x** and the mean curvature vector field can be calculated in specific cases,
so Eq. (8) has explicit form when _Q_ is specified.


As mentioned, Eq. (8) only has horizontal movements, in other words, it does not have any movement in the equivalent class. This process reduces unnecessary movement and helps to reduce
sampling trajectory length. From this viewpoint, previous methods do not reduce these unnecessary
movements, although they have the equivalent diffusion process in the quotient space. The formal
results are summarized in the following corollary. See Appx. D.2 for proof.


**Corollary 3.** **x** ˜1 _(defined by Eq._ (8) _) has the same distribution on Q with_ **x** 1 _(defined by Eq._ (6) _)._
_When σt_ = 0 _, ∀_ **x** 0 _∈M, Eq._ (8) _has shorter trajectory length than Eq._ (6) _._


3.2 SPECIAL CASE: THE SHAPE SPACE


The abstract results in the previous section give the direction for practical implementation. In this
subsection, we focus on the space of 3-dimensional coordinates of _N_ points under the symmetry
defined by the special Euclidean group SE(3), composed of the 3-dimensional translation group
and the 3-dimensional rotation SO(3). By definition, an element of this space is structured as **x** :=
( **x** [(1)] _, · · ·_ _,_ **x** [(] _[N]_ [)] ), where **x** [(] _[i]_ [)] _∈_ R [3], and the SE(3) group acts on **x** by translating and rotating
each **x** [(] _[i]_ [)] . Since the translation group is not compact, there does not exist a translational invariant
distribution. We (as well as many others (Yim et al., 2023; Lin et al., 2024)) hence represent the
quotient space w.r.t this group by considering the center-of-mass(CoM)-free subspace _M_ := _{_ **x** _∈_
R [3] _[N]_ _|_ _N_ 1 - _Ni_ =1 **[x]** [(] _[i]_ [)] [=] **[0]** _[}]_ [,] [and] [consider] [the] [SO(3)] [action] [on] [it.][1] [The] [resulting] [quotient] [space]
_Q_ := _M/_ SO(3), as the concrete construction for R [3] _[N]_ _/_ SE(3), is a smooth manifold under certain
conditions (Appx. C.1). Each element in _Q_ represents _N_ -point configurations that are equivalent
under altogether translation and rotation, therefore, _Q_ is regarded as the “shape space” reflecting
the intrinsically different states of the _N_ points. Now we can develop the correspondence between
the diffusion process in _M_ (Eq. (6)) and the its horizontal lift from the quotient space projection
(Eq. (8)). The results are summarized in the following theorem.
**Theorem** **4.** _Assume_ **x** _t_ _is_ _a_ _diffusion_ _process_ _in_ _the_ _CoM_ _subspace_ _M_ _⊂_ R [3] _[N]_ _,_ _given_ _by_ _the_
_following SDE:_ d **x** _t_ = **b** _t_ ( **x** _t_ ) d _t_ + _σt_ d **w** _t,_ **x** 0 _∼_ _p_ prior _where_ **b** _t_ ( **x** _t_ ) _is a_ SO(3) _-equivariant vector_
_field,∀t_ _∈_ [0 _, T_ ] _,_ _p_ prior _is the G-invariant prior distribution,_ **w** _t_ _is the standard Wiener process on_
_CoM. The horizontal lift of the process π_ ( **x** _t_ ) _is :_


       -       d˜ **x** _t_ = _P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _−_ _[σ]_ _t_ [2] **h** ˜(˜ **x** _t_ ) d _t_ + _σtP_ **x** ˜ _t_ d **w** _t,_ **x** ˜0 _∼_ _p_ prior _,_ (9)
2

_where the P_ **x** _is the horizontal projection operator at_ **x** _and_ **h** [˜] ( **x** ) _is the horizontal lift of the mean_
_curvature vector._ _The explicit expressions of P_ _and_ **h** [˜] _are shown as follows:_


     - _N_      _P_ **x** ( **v** ) = **v** _−_ **J** _[−]_ [1] - **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)]


_i_ =1


_×_ **x** _,_ _∀_ **v** _∈_ _T_ **x** _M,_ _and_


_N_


**x** [(] _[i]_ [)] **x** [(] _[i]_ [)] _[⊤]_ _∈_ R [3] _[×]_ [3] _._

_i_ =1


**h** ˜ [(] _[i]_ [)] ( **x** ) = _−_ - tr( **J** _[−]_ [1] ) **I** _−_ **J** _[−]_ [1][�] **x** [(] _[i]_ [)] _,_ _where_ **J** :=


_N_

- _∥_ **x** [(] _[i]_ [)] _∥_ [2] **I** _−_


_i_ =1


See Appx. D.3 for proof. From the results of Thm. 4, we can deduce that _π_ ( **x** _t_ ) has the same
marginal distribution with _π_ (˜ **x** _t_ ) in Eq. (9) (Cor. 3). If we consider the generation process in Eq. (2)
or Eq. (3) as **x** _t_, we can construct the corresponding horizontal process **x** ˜ _t_ that can generated the
same target distribution on the quotient space. Motivated by this fact, we can improve the training
and inference method of diffusion based generative models by leveraging the quotient structure.


1Technically, to guarantee proper structures, _M_ needs to exclude an negligible subset; see Appx. C.1.


6


3.3 PRACTICAL IMPLEMENTATIONS


Previous results describe how we can construct a diffusion process in the quotient space using the
coordinates in the total space. If we have a diffusion process on the total space, we can construct
the horizontal lift of its projection process, which has no vertical velocity along its trajectory and
the two processes are the same on quotient space. This fact implies that the vertical components of
the original diffusion process are not dispensable and enables us to design a more efficient training
and sampling algorithm of the diffusion model based on the quotient structure. In practice, we often
set the total space as the Euclidean space. Next, we show the training and sampling methods for the
special case _p_ prior = _N_ ( **0** _,_ **I** ), and the general case is shown in Appx. E.


**Training objective.** The diffusion model on the total space _M_ is trained by the objective Eq. (5).
Since the vertical components of the velocity are not strictly needed, we propose to supervise the
model only on the horizontal components and allow arbitrary vertical output of the model. We leverage the horizontal projection operator _P_ **x** (Thm. 4) and construct the horizontal training objective:

_L_ ( _θ_ ) := E _p_ ( _t_ ) _w_ ( _t_ )E _p_ ( **x** 1 _,_ **x** _t_ ) _∥P_ **x** _t_ ( **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1) _∥_ [2] _._ (10)

We can see that **D** _θ_ + **v** _[V]_ has the same loss value with **D** _θ_, where **v** _[V]_ is an arbitrary vertical vector.


**ODE** **sampler.** After the training stage, _P_ **x** _t_ ( **D** _θ_ ( **x** _t, t_ )) is an approximation of the ground truth
denoiser in the horizontal subspace. For the ODE sampler, we simulate the horizontal lift of the
projected ODE, which is given by [d] d **[x]** _t_ _[t]_ [=] _[P]_ **[x]** _[t]_ **[v]** _[θ]_ [(] **[x]** _[t][, t]_ [)d] _[t,]_ [where] **[v]** _[θ]_ [(] **[x]** _[t][, t]_ [)] [is] [given] [by] [Eq.] [(][4][).] [In]

practice, the ODE process is approximated by numerical solvers.


**SDE sampler.** For the stochastic sampler, the we need to simulate the horizontal lift of the projected
original SDE in Eq. (3). According to Thm. 1 and Thm. 4, the lifted process is given by

d **x** _t_ = _P_ **x** _t_ ( **v** _θ_ ( **x** _t, t_ ) + _gt_ **s** _θ_ ( **x** _t, t_ )) d _t_ + _γηt_ **h** ( **x** _t_ )d _t_ + ~~�~~ 2 _γηtP_ **x** _t_ d **w** _t,_

where **s** _θ_ ( **x** _t, t_ ) = _−_ **[x]** _[t][−][β][t]_ _α_ **[D]** ˆ [2] _t_ _[θ]_ [(] **[x]** _[t][,t]_ [)] and we introduce the hyperparameter _γ_ for protein generation

following Geffner et al. (2025). The details are summarized in Algorithm 1 and 3.


3.4 ANALYSIS ON EXISTING TREATMENTS FOR SYMMETRY


In this section, we make a detailed analysis on existing methods that handle symmetry, and verify the
conclusions in Table 1. In contrast to our quotient-space diffusion, we find that they either have not
fully leveraged the symmetry to reduce model-learning difficulty, or do not have a proper sampler.


**Conventional equivariant diffusion models and data augmentation.** A common treatment is by
assigning equal probability to equivalent objects, resulting in an invariant target distribution _p_ ( **x** 1).
This can be implemented by augmenting data samples by applying randomly chosen group actions,
mimicking sampling from the invariant distribution, or using an invariant prior distribution and an
equivariant architecture securing **D** _θ_ ( _g ·_ **x** _, t_ ) = _g ·_ **D** _θ_ ( **x** _, t_ ). The training strategy is the same as
modeling a general distribution in the original space following Eq. (5), and the standard samplers
by Eqs. (2, 3) remain valid. For each value of **x** _t_, this objective asks the model to minimize the
average of _∥_ **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1 _∥_ [2] terms where **x** 1 come from _p_ ( **x** 1 _|_ **x** _t_ ), so the optimal solution is the
conditional expectation E[ **x** 1 _|_ **x** _t_ ].


Fig. 3 shows an example and reveals characteristics of the training strategy. The example considers
generating the structure of a diatomic molecule, where the target distribution _p_ ( **x** 1) concentrates on
a single structure **x** _[⋆]_ up to a uniform random orientation (Left). For a given **x** _t_, samples of _p_ ( **x** 1 _|_ **x** _t_ )
are **x** _[⋆]_ structures posed in orientations distributed around the orientation of **x** _t_ (Middle). Indeed, an
**x** 1 sample more closely oriented with **x** _t_ would have a higher probability to produce the given **x** _t_ in
the diffusion process, so there is a specific orientation correspondence between the learning target
E[ **x** 1 _|_ **x** _t_ ] and **x** _t_ . So the model is still asked to learn a correspondence in the equivalent degrees
of freedom (DoFs) ( _i.e._, rotation of the output), in contrast to the quotient-space case in Eq. (10)
where the model is unconstrained in the vertical space ( _i.e._, tangent space of the rotation group).
Moreover, the **x** 1 samples are not all posed in the orientation of **x** _t_ because **x** _[⋆]_ in other orientations
can also generate this **x** _t_ through the diffusion process. So the model learns the correspondence in
the equivalent DoFs from samples with a variance, leading to another aspect of learning difficulty.


**GeoDiff alignment.** To reduce the learning difficulty, some heuristic treatments are proposed based
on alignment. The first representative alignment used in GeoDiff (Xu et al., 2022) uses the following


7


Figure 3: Illustration of denoising-model learning target using conventional training and using
GeoDiff alignment. **(Left)** The example considers the structure distribution _p_ ( **x** 1) of a diatomic
molecule, which concentrates on a single structure **x** _[⋆]_ up to a uniform random orientation. **(Middle)**
Given an **x** _t_ sample, the corresponding **x** 1 samples distribute with a variance, and their expectation
E[ **x** 1 _|_ **x** _t_ ] is the conventional learning target, which is _not_ equivalent to **x** _[⋆]_ (the bond is shorter).
**(Right)** Given an **x** _t_ sample, all the **x** 1 samples after alignment coincide with **x** _[⋆]_ posed in the
orientation of **x** _t_, which is also the learning target of GeoDiff E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ].
training loss: E _p_ ( **x** 1 _,_ **x** _t_ ) _∥_ **D** _θ_ ( **x** _t, t_ ) _−A_ **x** _t_ ( **x** 1) _∥_ [2], where the alignment operation is defined as:

_A_ **y** ( **x** ) := argmin **x** _′∈{g·_ **x** _|g∈G} d_ ( **x** _[′]_ _,_ **y** ) _,_ (11)

where _d_ ( _·, ·_ ) is the distance metric on _M_ . With an illustration in Fig. 3(Right), the learning task can
be understood as that for a given value of **x** _t_, the model output needs to fit _A_ **x** _t_ ( **x** 1) samples, which
are all posed in the orientation of **x** _t_, and they all coincide with the **x** _[⋆]_ structure in the orientation
of **x** _t_ . This supervises the model to the target E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ] from samples with no variance in the
equivalent DoFs ( _i.e._, rotation of the output), hence reduces certain learning difficulty. Nevertheless,
this target still requires the model to learn a specific mapping in the equivalent DoFs, hence does not
enjoy the learning advantage in the quotient-space case that relaxes the learning in the DoFs.


A caveat of this alignment approach is that a proper sampler needs to be developed, as the conventional samplers still require a model targeting E[ **x** 1 _|_ **x** _t_ ], which is different from E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ].
Fig. 3 illustrates this difference: E[ **x** 1 _|_ **x** _t_ ] averages diversely oriented **x** _[⋆]_ structures, resulting in a
different shape than **x** _[⋆]_ (the bond is shorter), while E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ] is just **x** _[⋆]_ in the orientation of **x** _t_ .
**AF3 alignment.** Another alignment approach, which is used in Alphafold 3 (AF3) (Abramson et al.,
2
2024), aligns the **x** 1 samples towards the model output: E _p_ ( **x** 1 _,_ **x** _t_ )�� **D** _θ_ ( **x** _t, t_ ) _−A_ **D** _θ_ ¯ [(] **[x]** _[t][,t]_ [)][(] **[x]** [1][)] ��,
where _θ_ [¯] is treated constant in optimization. This loss function allows the model output to differ by
an arbitrary group action ( _e.g._, rotation), hence removes the need to learn a specific target in the
equivalent DoFs. Indeed, for an arbitrary group action _g_ **x** _t,t_, a new denoising model _g_ **x** _t,t ·_ **D** _θ_ ( **x** _t, t_ )
achieves the same loss since _∥g_ **x** _t,t_ _·_ **D** _θ_ ( **x** _t, t_ ) _−Ag_ **x** _t,t·_ **D** _θ_ ¯ [(] **[x]** _[t][,t]_ [)][(] **[x]** [1][)] _[∥]_ [2] [=] _[∥][g]_ **[x]** _[t][,t]_ _[·]_ **[D]** _[θ]_ [(] **[x]** _[t][, t]_ [)] _[−]_
_g_ **x** _t,t_ _· A_ **D** _θ_ ¯ [(] **[x]** _[t][,t]_ [)][(] **[x]** [1][)] _[∥]_ [2] [=] _[∥]_ **[D]** _[θ]_ [(] **[x]** _[t][, t]_ [)] _[ −A]_ **[D]** _θ_ [¯][(] **[x]** _[t][,t]_ [)][(] **[x]** [1][)] _[∥]_ [2][,] [where] [the] [last] [equality] [holds] [since] [the]
group preserves metric (Appx. C). Up to this DoF, the learning target is the same as GeoDiff’s
E[ _A_ **x** _t_ ( **x** 1) _|_ **x** _t_ ], since all the **x** 1 samples are averaged after aligned with the same reference.


In the sampling process, the arbitrariness in the equivalent DoFs ( _e.g._, orientation) of the learned
model **D** _θ_ ( **x** _t, t_ ) leads to an arbitrariness [2] in the vector field **v** _θ_ ( **x** _t, t_ ) through Eq. (4). Hence there is
no guarantee of recovering the target distribution using conventional samplers. This problem is also
noted by Boltz-1 (Wohlwend et al., 2025), which proposes to align the prediction **D** _θ_ ( **x** _t, t_ ) towards
**x** _t_ in the sampling process. As the AF3 target is the same as GeoDiff’s up to an arbitrary rotation,
this amounts to using the GeoDiff model for sampling, which still cannot guarantee producing the
target distribution as concluded above. These discussions are summarized in Table 1.


4 EXPERIMENTS


In this section, we study the empirical performance of our quotient-space diffusion model. We
carefully conduct several experiments covering different types of data, scales and scenarios. To
evaluate our quotient space diffusion model framework for real-world applications, we focus on the
molecule structure generation protein backbone design tasks, in which we consider the diffusion
models on R [3] _[N]_ _/_ SE(3) (Sec. 3.2). The details of all experiments are shown in Appx. G.


4.1 STRUCTURE GENERATION FOR SMALL MOLECULES


**Datasets.** First, we evaluate our framework on the molecule structure generation task. In this scenario, our goal is to generate the 3D coordinates of a molecule given the graph structure of the


2This is not even an arbitrary group action ( _e.g._, rotation) since **x** _t_ does not vary together with the arbitrariness of **D** _θ_ ( **x** _t, t_ ).


8


Table 2: The effect of the quotient-space diffusion scheme for molecular structure generation on
the GEOM-QM9 and the GEOM-DRUGS datasets using the ET-Flow(SO(3)) and ET-Flow(O(3))
architectures. We use the same sampling steps of 50 NFEs for fair comparison. Best results are
marked in **bold** . Best results for the same architecture are underlined.


Recall Precision
Datasets Methods Coverage _↑_ AMR _↓_ Coverage _↑_ AMR _↓_

mean median mean median mean median mean median


GEOM-QM9
(Positive samples
are within
0 _._ 5 A RMSD.) [˚]


GEOM-DRUGS
(Positive samples
are within
0 _._ 75 A RMSD.) [˚]


CGCF 69.47 96.15 0.425 0.374 38.20 33.33 0.711 0.695
GeoDiff 76.50 **100.00** 0.297 0.229 50.00 33.50 1.524 0.510
GeoMol 91.50 **100.00** 0.225 0.193 87.60 **100.00** 0.270 0.241
Torsional Diff. 92.80 **100.00** 0.178 0.147 92.70 **100.00** 0.221 0.195
MCF 95.0 **100.00** 0.103 0.044 93.7 **100.00** 0.119 0.055


ET-Flow(SO(3)) 95.98 **100.00** 0.076 0.030 92.10 **100.00** 0.110 0.047
+ Geodiff alignment 95.71 **100.00** 0.085 0.040 **95.20** **100.00** 0.098 0.050
+ AF3 alignment 92.67 **100.00** 0.131 0.070 84.38 **100.00** 0.205 0.146
**+ Quotient-space diffusion** **96.40** **100.00** **0.069** **0.024** 93.30 **100.00** **0.096** **0.036**


GeoDiff 42.10 37.80 0.835 0.809 24.90 14.50 1.136 1.090
GeoMol 44.60 41.40 0.875 0.834 43.00 36.40 0.928 0.841
Torsional Diff. 72.70 80.00 0.582 0.565 55.20 56.90 0.778 0.729
MCF - S (13M) 79.4 87.5 0.512 0.492 57.4 57.6 0.761 0.715
MCF - B (62M) 84.0 91.5 0.427 0.402 64.0 66.2 0.667 0.605
MCF - L (242M) **84.7** **92.2** **0.390** **0.247** 66.8 71.3 0.618 0.530


ET-Flow (8.3M) 79.53 84.57 0.452 0.419 **74.38** **81.04** **0.541** **0.470**
+ reproduction 78.94 84.24 0.489 0.472 66.24 70.42 0.651 0.595
**+ Quotient-space diffusion** 79.86 85.71 0.459 0.433 72.70 79.63 0.565 0.501


ET-Flow(SO(3)) (9.1M) 78.18 83.33 0.480 0.459 67.27 71.15 0.637 0.567
+ reproduction 74.91 80.90 0.541 0.515 60.33 62.71 0.724 0.665
+ Geodiff alignment 75.11 80.74 0.545 0.526 59.58 60.48 0.734 0.678
+ AF3 alignment 71.66 76.09 0.572 0.570 52.21 50.00 0.828 0.793
**+ Quotient-space diffusion** 78.50 84.20 0.477 0.455 67.35 71.42 0.635 0.563


molecule. We conduct the experiments on the GEOM datasets (Axelrod & Gomez-Bombarelli,
2022), which provides structure ensembles generated by metadynamics in CREST (Pracht et al.,
2024) and we focus on the GEOM-QM9 and GEOM-DRUGS datasets. Following the data processing and splits from Hassan et al. (2024), we use the random splits with train/validation/test of
243473/30433/1000 for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9. In addition,
data with disconnect molecule graph are removed for GEOM-DRUGS.


**Setting.** We primarily follow the setting in Hassan et al. (2024). We use an equivariant graph
transformer architecture from ET-Flow (Hassan et al., 2024) and set the Gaussian distribution as
prior distribution on GEOM-QM9 and use the harmonic prior for GEOM-DRUGS (Volk et al.,
2023). We fix the architecture as ET-Flow(SO(3)) for experiments on GEOM-QM9, and use the
ET-Flow(O(3)), ET-Flow(SO(3)) architecture on the GEOM-DRUGS dataset. Following Jing et al.
(2022); Xu et al. (2022), we report the RMSD-based metrics, _e.g._ Coverage and Average Minimum
RMSD (AMR) between the generated and ground truth structure ensembles.


**Results.** The results are presented in Table 2 for the GEOM-QM9 and GEOM-DRUGS datasets. As
shown, our proposed quotient-space diffusion framework consistently outperforms prior methods
and alignment techniques in terms of generation quality on both datasets. Our framework reduces
learning difficulty by removing redundant components, enabling us to further improve the performance of the ET-Flow framework [3] on both datasets. On the GEOM-QM9 dataset, our quotient-space
diffusion model framework surpasses strong baselines such as MCF (Wang et al., 2023) and the ETFlow framework with other heuristic alignment methods among most of the RMSD-based metrics.
On the GEOM-DRUGS dataset, our framework not only significantly surpasses the ET-Flow baseline with heuristic alignment methods, since these methods are incompatible with training, but also
achieves competitive performance against the larger MCF-L (242M) model (Wang et al., 2023) on
the Precision metrics.


4.2 PROTEIN BACKBONE DESIGN


**Setting.** To demonstrate the advantage of our quotient-space diffusion model for larger and more
relevant molecules, we perform a comparative analysis on the task of protein structure generation
against the state-of-the-art Prote´ına model (Geffner et al., 2025). We select their most efficient


3We reproduce the results using the released configurations: [https://github.com/](https://github.com/shenoynikhil/ETFlow)
[shenoynikhil/ETFlow.](https://github.com/shenoynikhil/ETFlow) Due to changes in the data processing pipeline, our reproduced results do
not exactly match those reported in the original paper.


9


Table 3: The effect of the quotient-space diffusion scheme for protein structure generation using the
Prote´ına model. Best results are marked in **bold** .


FPSD vs. fS fJSD vs.
Settings Methods Designability (%) _↑_

PDB _↓_ AFDB _↓_ (C/A/T) _↑_ PDB _↓_ AFDB _↓_


Representative
References


SDE
Sampling


ODE
Sampling


FrameDiff 65.4 194.2 258.1 2.46/5.78/23.35 1.04 1.42
FoldFlow (base) 96.6 601.5 566.2 1.06/1.79/9.72 3.18 3.10
FoldFlow (stoc.) 97.0 543.6 520.4 1.21/2.09/11.59 3.69 2.71
FoldFlow (OT) 97.2 431.4 414.1 1.35/3.10/13.62 2.90 2.32
FrameFlow 88.6 129.9 159.9 2.52/5.88/27.00 0.68 0.91
ESM3 22.0 933.9 855.4 3.19/6.71/17.73 1.53 0.98
Chroma 74.8 189.0 184.1 2.34/4.95/18.15 1.00 1.08
RFDiffusion 94.4 253.7 252.4 2.25/5.06/19.83 1.21 1.13
Proteus 94.2 225.7 226.2 2.26/5.46/16.22 1.41 1.37
Genie2 95.2 350.0 313.8 1.55/3.66/11.65 2.21 1.70


Prote´ına _M_ [small] FS _[, γ]_ [= 0] _[.]_ [35] 96.0 386.5 378.2 1.77/4.97/17.78 2.17 1.73
**+ Quotient-space diffusion** **97.6** 274.7 277.1 2.24/6.69/20.99 1.68 1.55
Prote´ına _M_ [small] FS _[, γ]_ [= 0] _[.]_ [45] 92.2 332.9 320.4 1.83/5.01/20.22 1.93 1.49
**+ Quotient-space diffusion** 92.6 244.5 246.3 2.24/6.68/23.47 1.43 1.28
Prote´ına _M_ [small] FS _[, γ]_ [= 0] _[.]_ [50] 89.2 306.2 290.8 1.86/4.92/21.15 1.81 1.36
**+ Quotient-space diffusion** 90.2 228.0 228.7 2.25/6.59/25.24 1.32 1.17


Prote´ına _M_ FS 19.6 85.4 21.4 2.51/5.65/27.35 0.59 **0.09**
Prote´ına _M_ [small] FS 13.8 83.2 21.9 2.45/5.63/31.76 0.58 0.12
+ AF3 alignment 3.8 229.0 82.4 2.18/4.30/14.28 1.35 0.36
**+ Quotient-space diffusion** 15.6 **69.9** **17.6** **2.57/6.40/32.14** **0.41** 0.11


variant _M_ [small] FS [,] [a] [60M] [parameter] [transformer] [trained] [on] [the] [Foldseek] [AFDB] [clusters] [(] _[D]_ [FS][)] [that]
forgoes triangle layers and pair representation updates, as a strong and relevant baseline. We train
the quotient-space diffusion model from scratch using the identical architecture on the identical
dataset. For evaluation, both our model and the officially released Prote´ına checkpoint are sampled
using 400 steps with self-conditioning. We explore the designability-diversity trade-off by testing
a range of noise scales, _γ_ _∈{_ 0 _._ 35 _,_ 0 _._ 45 _,_ 0 _._ 5 _}_ [4] . To faithfully evaluate the distributional metrics
proposed by Geffner et al. (2025), we utilize ODE sampling.
**Results.** The results in Table 3 highlight the superiority of our quotient space framework, which,
unlike alignment-based strategies (adapted from AF3 and Boltz-1), provides a theoretical guarantee
for sampling the correct target distribution. The alignment-based methods fail to recover this distribution, with performance metrics falling short of even data-augmented, semi-equivariant baselines.
We attribute this failure to a fundamental incompatibility between their samplers and the learned
models. Furthermore, our formulation effectively reduces learning difficulty by removing the need
to learn a specific target in redundant spatial transformations, enabling the model to capture key
structural features more efficiently than standard semi-equivariant baselines. This advantage of efficiency leads to significant results: our 60M parameter model not only surpasses its direct baseline
across both SDE at all noise scales and ODE sampling setting, but also outperforms the much larger
200M _M_ FS model on most key distributional metrics. This provides compelling evidence that a quotient space framework ensuring both sampling fidelity and learning efficiency is key to advancing
generative protein models.


5 CONCLUSION


In this work, we formally construct a framework for building diffusion models on the quotient space
over a group, in hope for a principled approach to handle symmetry in a generative task. We explicitly give the expression of the diffusion process on the quotient space, then also construct a
corresponding diffusion process in the original space for easier implementation. The resulting training algorithm reduces learning difficulty by removing the need to predict the tangent vector in the
direction along group action, and the resulting sampling process guarantees producing the target
distribution while removes the unnecessary movement in the group-action direction. We instantiate
the method in the case of R [3] _[N]_ _/_ SE(3) for molecular structure generation. Empirical results on structure sampling for small molecules from the GEOM-QM9 and GEOM-DRUGS datasets and protein
backbone generation demonstrate the better generation quality and design success rate over existing conventional equivariant diffusion models and alignment-based approaches given equal or fewer
training epochs, demonstrating the practical advantages from this principled framework to handling
symmetry in diffusion models.


4Due to a known bug in a previous version of Foldseek (Daras et al., 2025, Appendix B), our comparative
analysis in the main text is focused solely on the designability. More comprehensive metrics evaluating our
self-sampled structures are provided in Table 5.


10


ACKNOWLEDGMENTS


This work is supported by Zhongguancun Academy (Grant No. C20250506). DH is supported by
National Science Foundation of China (NSFC62376007), National Science Foundation of China
(under Key Project No. 92570203), Beijing Natural Science Foundation (Z250001) and Beijing
Major Science and Technology Project under Contract no. Z251100008425004.


6 ETHICS STATEMENT


This work adheres to the ICLR Code of Ethics. Our study does not involve human subjects, personal data, or sensitive demographic information. All experiments are conducted on publicly available benchmark datasets, which are widely used in the machine learning community. No new data
collection or human/animal experimentation was performed.


7 REPRODUCIBILITY STATEMENT


To facilitate the reproducibility of our research, we provide comprehensive details throughout the
paper and its supplementary materials. We begin by establishing the necessary foundational knowledge in Sec. 2.1 and Appx. B. For all theoretical claims and proofs presented in the main text, we
offer detailed step-by-step derivations in Appx. D. Our experiments are thoroughly documented; the
datasets, training procedures, and evaluation protocols are carefully described in Sec. 4 and Appx. G.
Upon acceptance of this paper, we commit to making our full codebase and all model checkpoints
publicly available to ensure that the community can fully reproduce our results.


8 THE USE OF LARGE LANGUAGE MODELS (LLMS)


In the preparation of this manuscript, LLMs were employed as a writing assistant to refine the
language and improve the grammar. Furthermore, we utilized LLMs to assist in verifying our mathematical formulas for notational consistency. Following this process, all textual and mathematical
content was meticulously reviewed, revised, and validated by the authors, who assume full responsibility for the final work presented.


REFERENCES


Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf
Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure
prediction of biomolecular interactions with AlphaFold 3. _Nature_, pp. 1–3, 2024.


Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying
framework for flows and diffusions. _arXiv preprint arXiv:2303.08797_, 2023.


Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured
denoising diffusion models in discrete state-spaces. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 34:17981–17993, 2021.


Simon Axelrod and Rafael Gomez-Bombarelli. GEOM, energy-annotated molecular conformations
for property prediction and molecular generation. _Scientific Data_, 9(1):185, 2022.


Jan-Hendrik Bastek, WaiChing Sun, and Dennis Kochmann. Physics-informed diffusion models.
In _The Thirteenth_ _International Conference on Learning Representations_, 2025. [URL https:](https://openreview.net/forum?id=tpYeermigp)
[//openreview.net/forum?id=tpYeermigp.](https://openreview.net/forum?id=tpYeermigp)


Fabrice Baudoin, Nizar Demni, and Jing Wang. _Stochastic areas, horizontal Brownian motions, and_
_hypoelliptic heat kernels_ . EMS Press, 2024.


Isaac Chavel. _Riemannian_ _geometry:_ _a_ _modern_ _introduction_ . Number 108. Cambridge university
press, 1995.


Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. _arXiv_ _preprint_
_arXiv:2302.03660_, 2023.


11


Franc¸ois Cornet, Federico Bergamin, Arghya Bhowmik, Juan Maria Garcia Lastra, Jes Frellsen, and
Mikkel N Schmidt. Kinetic langevin diffusion for crystalline materials generation. _arXiv preprint_
_arXiv:2507.03602_, 2025.


Giannis Daras, Jeffrey Ouyang-Zhang, Krithika Ravishankar, William Daspit, Costis Daskalakis,
Qiang Liu, Adam Klivans, and Daniel J Diaz. Ambient proteins: Training diffusion models on
low quality structures. _bioRxiv_, pp. 2025–07, 2025.


Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and
Arnaud Doucet. Riemannian score-based generative modelling. _Advances in neural information_
_processing systems_, 35:2406–2422, 2022.


Zach Evans, Cj Carr, Josiah Taylor, Scott H. Hawley, and Jordi Pons. Fast timing-conditioned
latent audio diffusion. In _Proceedings_ _of_ _the_ _41st_ _International_ _Conference_ _on_ _Machine_ _Learn-_
_ing_, volume 235 of _Proceedings_ _of Machine Learning Research_, pp. 12652–12665, 2024. URL
[https://proceedings.mlr.press/v235/evans24a.html.](https://proceedings.mlr.press/v235/evans24a.html)


Octavian Ganea, Lagnajit Pattanaik, Connor Coley, Regina Barzilay, Klavs Jensen, William Green,
and Tommi Jaakkola. Geomol: Torsional geometric generation of molecular 3d conformer ensembles. _Advances in Neural Information Processing Systems_, 34:13757–13769, 2021.


Tomas Geffner, Kieran Didi, Zuobai Zhang, Danny Reidenbach, Zhonglin Cao, Jason Yim, Mario
Geiger, Christian Dallago, Emine Kucukbenli, Arash Vahdat, and Karsten Kreis. Proteina: Scaling flow-based protein structure generative models. In _The_ _Thirteenth_ _International_ _Confer-_
_ence_ _on_ _Learning_ _Representations_, 2025. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=TVQLu34bdw)
[TVQLu34bdw.](https://openreview.net/forum?id=TVQLu34bdw)


Majdi Hassan, Nikhil Shenoy, Jungyoon Lee, Hannes St¨ark, Stephan Thaler, and Dominique Beaini.
ET-Flow: Equivariant flow-matching for molecular conformer generation. _Advances_ _in_ _Neural_
_Information Processing Systems_, 37:128798–128824, 2024.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In _Advances_
_in Neural Information Processing Systems_, volume 33, pp. 6840–6851, 2020.


Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P
Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition
video generation with diffusion models. _arXiv preprint arXiv:2210.02303_, 2022.


Emiel Hoogeboom, Vıctor Garcia Satorras, Cl´ement Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In _International conference on machine learning_, pp. 8867–
8887. PMLR, 2022a.


Emiel Hoogeboom, V´ıctor Garcia Satorras, Cl´ement Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3D. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba
Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference_
_on_ _Machine_ _Learning_, volume 162 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 8867–
8887. PMLR, 17–23 Jul 2022b.


Elton P Hsu. _Stochastic analysis on manifolds_ . Number 38. American Mathematical Soc., 2002.


Chenqing Hua, Sitao Luan, Minkai Xu, Zhitao Ying, Jie Fu, Stefano Ermon, and Doina Precup.
Mudiff: Unified diffusion for complete molecule generation. In _Learning on Graphs Conference_,
pp. 33–1. PMLR, 2024.


Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion models. _Advances in Neural Information Processing Systems_, 35:2750–2761,
2022.


Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi Jaakkola. Torsional diffusion for molecular conformer generation. _Advances in Neural Information Processing Systems_,
35:24240–24253, 2022.


12


Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. _Advances in neural information processing systems_, 35:26565–26577,
2022.


Seongsu Kim, Nayoung Kim, Dongwoo Kim, and Sungsoo Ahn. High-order equivariant flow matching for density functional theory Hamiltonian prediction. _arXiv preprint arXiv:2505.18817_, 2025.


Jonas K¨ohler, Leon Klein, and Frank No´e. Equivariant flows: exact likelihood generative learning
for symmetric densities. In _International conference on machine learning_, pp. 5361–5370. PMLR,
2020.


Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan C. Catanzaro. DiffWave: A versatile
diffusion model for audio synthesis. In _International_ _Conference_ _on_ _Learning_ _Representations_
_(ICLR)_, 2021. [URL https://openreview.net/forum?id=a-xFK8Ymz5J.](https://openreview.net/forum?id=a-xFK8Ymz5J)


John M Lee. Smooth manifolds. In _Introduction to smooth manifolds_, pp. 1–29. Springer, 2003.


John M Lee. _Introduction to Riemannian manifolds_, volume 2. Springer, 2018.


Sarah Lewis, Tim Hempel, Jos´e Jim´enez-Luna, Michael Gastegger, Yu Xie, Andrew Y. K. Foong,
Victor Garc´ıa Satorras, Osama Abdin, Bastiaan S. Veeling, Iryna Zaporozhets, Yaoyi Chen, Soojung Yang, Adam E. Foster, Arne Schneuing, Jigyasa Nigam, Federico Barbero, Vincent Stimper,
Andrew Campbell, Jason Yim, Marten Lienen, Yu Shi, Shuxin Zheng, Hannes Schulz, Usman
Munir, Roberto Sordillo, Ryota Tomioka, Cecilia Clementi, and Frank No´e. Scalable emulation
of protein equilibrium ensembles with generative deep learning. _Science_, 389(6761):eadv9817,
2025. doi: 10.1126/science.adv9817. [URL https://www.science.org/doi/abs/10.](https://www.science.org/doi/abs/10.1126/science.adv9817)
[1126/science.adv9817.](https://www.science.org/doi/abs/10.1126/science.adv9817)


Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng,
Errui Ding, and Jingdong Wang. VideoGen: A reference–guided latent diffusion approach for
high definition text-to-video generation. _arXiv preprint arXiv:2309.00398_, 2023. [URL https:](https://arxiv.org/abs/2309.00398)
[//arxiv.org/abs/2309.00398.](https://arxiv.org/abs/2309.00398)


Peijia Lin, Pin Chen, Rui Jiao, Qing Mo, Cen Jianhuan, Wenbing Huang, Yang Liu, Dan Huang,
and Yutong Lu. Equivariant diffusion for crystal structure prediction. In _Forty-first International_
_Conference on Machine Learning_, 2024.


Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow
matching for generative modeling. In _The Eleventh International Conference on Learning Repre-_
_sentations_, 2023. [URL https://openreview.net/forum?id=PqvMRDCJT9t.](https://openreview.net/forum?id=PqvMRDCJT9t)


Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow. In _The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_ _Repre-_
_sentations_, 2023. [URL https://openreview.net/forum?id=XVjTT1nw5z.](https://openreview.net/forum?id=XVjTT1nw5z)


Philipp Pracht, Stefan Grimme, Christoph Bannwarth, Fabian Bohle, Sebastian Ehlert, Gereon Feldmann, Johannes Gorges, Marcel M¨uller, Tim Neudecker, Christoph Plett, et al. Crest—a program
for the exploration of low-energy molecular chemical space. _The_ _Journal_ _of_ _Chemical_ _Physics_,
160(11), 2024.


Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du,
Carla Gomes, Tom L Blundell, Pietro Lio, et al. Structure-based drug design with equivariant
diffusion models. _Nature Computational Science_, 4(12):899–909, 2024.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In _Interna-_
_tional Conference on Learning Representations_, 2021.


Anton Thalmaier. Stochastic riemannian geometry. 2023.


Jos Torge, Charles Harris, Simon V Mathis, and Pietro Lio. Diffhopp: A graph diffusion model for
novel drug design via scaffold hopping. _arXiv preprint arXiv:2308.07416_, 2023.


13


Amanda A Volk, Robert W Epps, Daniel T Yonemoto, Benjamin S Masters, Felix N Castellano,
Kristofer G Reyes, and Milad Abolhasani. AlphaFlow: autonomous discovery and optimization
of multi-step chemistry using a self-driven fluidic lab guided by reinforcement learning. _Nature_
_Communications_, 14(1):1403, 2023.


Yuyang Wang, Ahmed A Elhag, Navdeep Jaitly, Joshua M Susskind, and Miguel Angel
Bautista. Swallowing the bitter pill: Simplified scalable conformer generation. _arXiv_ _preprint_
_arXiv:2311.17932_, 2023.


Jeremy Wohlwend, Gabriele Corso, Saro Passaro, Noah Getz, Mateo Reveiz, Ken Leidal, Wojtek
Swiderski, Liam Atkinson, Tally Portnoi, Itamar Chinn, et al. Boltz-1 democratizing biomolecular
interaction modeling. _BioRxiv_, pp. 2024–11, 2025.


Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and Qiang Liu. Diffusion-based molecule
generation with informative prior bridges. _Advances_ _in_ _neural_ _information_ _processing_ _systems_,
35:36533–36545, 2022.


Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. GeoDiff: A geometric diffusion model for molecular conformation generation. In _International_ _Conference_ _on_
_Learning Representations_, 2022.


Minkai Xu, Alexander S Powers, Ron O Dror, Stefano Ermon, and Jure Leskovec. Geometric latent
diffusion models for 3d molecule generation. In _International Conference on Machine Learning_,
pp. 38592–38610. PMLR, 2023.


Jason Yim, Brian L Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay,
and Tommi Jaakkola. SE(3) diffusion model with application to protein backbone generation. In
_International Conference on Machine Learning_, pp. 40001–40039, 2023.


Shuxin Zheng, Jiyan He, Chang Liu, Yu Shi, Ziheng Lu, Weitao Feng, Fusong Ju, Jiaxi Wang,
Jianwei Zhu, Yaosen Min, He Zhang, Shidi Tang, Hongxia Hao, Peiran Jin, Chi Chen, Frank
No´e, Haiguang Liu, and Tie-Yan Liu. Predicting equilibrium distributions for molecular systems with deep learning. _Nature_ _Machine_ _Intelligence_, 2024. ISSN 2522-5839. doi: 10.1038/
s42256-024-00837-3.


Yuchen Zhu, Tianrong Chen, Lingkai Kong, Evangelos A Theodorou, and Molei Tao. Trivialized momentum facilitates diffusion generative modeling on lie groups. _arXiv_ _preprint_
_arXiv:2405.16381_, 2024.


14


APPENDIX


The organization of the appendix are as follows. In Appx. A, we briefly discuss the related work
relevant to our research. In Appx. B, we review some background knowledge of Riemannian geometry and stochastic calculus on the manifold. In Appx. C, we give the details of the Riemannian
structures of the quotient space. In Appx. D, we give all the proofs of the theorems in the main text.
In Appx. E, we show our methods for the general case. In Appx. F, we give some additional results
and discussions. Finally, the details of the experiments are given in Appx. G.


A RELATED WORK


**Diffusion models on Riemannian manifolds.** As the quotient has the Riemannian manifold structure, several previous works construct the diffusion model on the Riemannian manifolds. De Bortoli
et al. (2022) constructs diffusion models using different overlapping local coordinate systems of the
manifold and requires geodesic random walk to simulate the forward process. Huang et al. (2022);
Chen & Lipman (2023) construct diffusion models in an embedding space which allows a global
representation but requires explicit geodesic formula of the manifold. Zhu et al. (2024) constructs
the reverse of kinetic Langevin dynamics on a Lie group to perform generative modeling. Such an
approach is not designed for and not readily applicable to the quotient space, which has a different
geometric structure from the Lie group. In our quotient space case, the specialty with a quotient
structure enables us to construct diffusion models using the coordinate systems of the total space
without relying on an embedding of the quotient in the total space (unnecessarily an embedding
space), which is more practical to implement yet still general.


**Geometric** **diffusion** **models.** To ensure physical symmetry in the generation process, a mainstream strategy integrates fundamental physical constraints, such as SE(3) equivariance, directly
into the diffusion model’s architecture. This approach, pioneered by models like EDM (Hoogeboom
et al., 2022a), typically employs an EGNN to operate directly on atomic coordinates, using techniques like zero center of mass adjustments to guarantee translational invariance. This foundational
concept was subsequently extended in several directions. For instance, the approach was adapted for
Diffusion Bridges in models like EDM-Bridge (Wu et al., 2022) and for diffusion in a latent space in
models like GeoLDM (Xu et al., 2023). These equivariant diffusion techniques have been successfully applied across a range of molecular tasks. For structure generation, models like GeoDiff (Xu
et al., 2022) predict 3D structures from molecular graphs. In molecular optimization, methods such
as DiffHopp (Torge et al., 2023) refine existing molecules to enhance desired properties. For de
novo design, a key advancement has been to combine discrete diffusion models (D3PM) (Austin
et al., 2021) for 2D topology with continuous equivariant diffusion for 3D geometry, enabling joint
generation as seen in models like DiffSBDD (Schneuing et al., 2024) and MUDiff (Hua et al., 2024).
A similar problem has also been considered in crystalline structure generation, where the intrinsic
periodic translation invariance is an intrinsic symmetry. Lin et al. (2024) highlighted the intrinsic
periodic translation symmetry that has been omitted for a long time in the field of periodic crystalline
structure generation. The work designed a modified diffusion process that induces a transition kernel that is invariant under periodic translation, leading to a learning target for the score model that is
invariant under periodic translation. Cornet et al. (2025) proposes a novel method that generalizes
the Trivialized Diffusion Model framework for fractional coordinates to model the intrinsic periodic
translation symmetry using flat coordinates. The proposed method considers the process with the
velocity restricted to the CoM-free linear subspace. They have achieved the removal of variance on
equivalent DoFs, but still asks the neural network model to learn to predict a specific target in the
equivalent DoFs.


**Learning with alignment** To reduce learning difficulty, some heuristic treatments (learning with
alignment) have been proposed in hope to reduce the DoFs corresponding to the symmetry group
action. The alignment strategy used in GeoDiff (Xu et al., 2022) aligns the target structure to the
noisy input by finding an optimal rigid transformation that minimizes the distance between them.
Another approach, proposed in AlphaFold 3 (AF3) (Abramson et al., 2024), aligns the target samples to the model output structure. As discussed in the main text, these two alignment-based training
frameworks lack a definite guarantee for recovering the correct target distribution, and is incompatible with the sampling process. Boltz-1 (Wohlwend et al., 2025), an open-source replication of AF3,
noticed this issue and proposed a modification in sampling to align the denoised structure to the


15


structure in the current generation step before updating. Nevertheless, as discussed in Sec. 3.4, this,
together with the training protocol of AF3, amounts to the operation of GeoDiff, still questioning
the sampling process.


B BACKGROUND IN RIEMANNIAN GEOMETRY AND STOCHASTIC CALCULUS


B.1 RIEMANNIAN GEOMETRY


In this section, we review some background on differential geometry and Riemannian geometry. For
a systematic treatment of the subject, please refer to standard textbooks Lee (2003; 2018).


First, we give the formal definition of the smooth manifold. A manifold is a general topological
space that locally has a Euclidean structure.


**Definition 5.** An _M_ **-dimensional topological manifold** is a topological space _M_ such that:


    - _M_ is locally Euclidean, _i.e._ locally homeomorphic to R _[M]_ . Formally, _∀x_ _∈M_, [5] there
exists an open neighborhood _x ∈U_ _⊂M_ that is homeomorphic to some open set _V_ _⊂M_ .
We call the homeomorphism _ϕ_ : _U_ _→V_ _⊂_ R _[M]_ a **coordinate system** or a chart.


    - _M_ is a Hausdorff topological space.


    - _M_ has a countable basis for its topology.


A smooth manifold is a topological manifold with an additional smooth structure, which is defined
as follows.


**Definition 6.** A smooth structure on a _M_ -dimensional topological space _M_ is a collection of coordinate systems _C_ = _{_ ( _U_ [(] _[α]_ [)] _, ϕ_ [(] _[α]_ [)] ) : _α ∈A}_ which satisfies the following properties:


    - The collection _C_ covers _M_ : [�] _α∈A_ _[U]_ [(] _[α]_ [)] [=] _[ M]_ [;]

    - For any _α, β_ _∈A_, the transition function _ϕ_ [(] _[α]_ [)] _◦_ _ϕ_ [(] _[β]_ [)] _[−]_ [1] is a smooth map;


    - _C_ is a maximal collection, _i.e._ if ( _U, ϕ_ ) is a coordinate system such that for all _α ∈A_ that
the maps _ϕ ◦_ _ϕ_ [(] _[α]_ [)] _[−]_ [1] and _ϕ_ [(] _[α]_ [)] _◦_ _ϕ_ _[−]_ [1] are smooth, then ( _U, ϕ_ ) _∈C_ .


The pair ( _M, C_ ) is called a **smooth manifold** of dimension _M_ .


If there is a coordinate system ( _U, ϕ_ ) around a point _x_ _∈M_, then in this neighborhood of _x_, the
manifold admits a coordinate chart _x_ _[i]_ ( _x_ ) := _ϕ_ _[i]_ ( _x_ ) and a manifold point in the neighborhood can be
expressed as a vector **x** ( _x_ ) = ( _x_ [1] ( _x_ ) _, · · ·_ _, x_ _[M]_ ( _x_ )) _[⊤]_ .


With the smooth structure, we can define a smooth function on the manifold and a smooth mapping
between smooth manifolds.


**Definition 7.** Let _M, N_ be smooth manifolds with dimensions _M, N_ respectively.


    - A function _f_ : _M_ _→_ R is called a **smooth** **function** if its vectorized form _f_ _◦_ _ϕ_ _[−]_ [1] :
_ϕ_ _[−]_ [1] ( _U_ ) _→_ R is smooth on _ϕ_ _[−]_ [1] ( _U_ ) _⊂_ R _[M]_ for all smooth coordinate systems ( _U, ϕ_ ) of
_M_ . Denote all the smooth functions on _M_ as _C_ _[∞]_ ( _M_ ).


    - A map _F_ : _M →N_ is called a **smooth map** if its vectorized form _ψ_ _◦F ◦ϕ_ _[−]_ [1] : _ϕ_ _[−]_ [1] ( _U_ ) _→_
_ψ_ ( _V_ ) is smooth for all smooth coordinate systems ( _U, ϕ_ ) of _M_ and ( _V, ψ_ ).


A smooth map _F_ : _M →N_ which is invertible and whose inverse is smooth is called a diffeomorphism. In this case we say that _M_ and _N_ are diffeomorphic manifolds.


To define movement on a smooth manifold _M_, we need to define tangent vectors on the manifold.


5On an abstract manifold, a point is an abstract object and may not be a vector by itself, so we do not use
a boldface notation. A vector representation as the coordinates is available after choosing a (local) coordinate
system.


16


**Definition 8.** Let _M_ be a smooth manifold, and _x_ _∈M_ is a point, and _U_ is a neighborhood of it.
A linear map **v** : _C_ _[∞]_ ( _U_ ) _→_ R is called a derivative at _x_ if it satisfies

**v** ( _fg_ ) = _f_ ( _x_ ) **v** ( _g_ ) + _g_ ( _x_ ) **v** ( _f_ ) _,_ _∀f, g_ _∈_ _C_ _[∞]_ ( _U_ ) _._

The set of all the derivatives of _C_ _[∞]_ ( _U_ ) in _x_, denoted by _TxM_, is a vector space called the **tangent**
**space** to _M_ at _x_ . An element of _TxM_ is called a **tangent vector** at _x_ .


**Definition 9.** Let _M, N_ be smooth manifolds and _F_ : _M →N_ be a smooth map. Let _x ∈M_ and
_V_ _⊆N_ be a neighborhood of _F_ ( _x_ ). Then _F_ induces a **push-forward map** over the tangent spaces,
_F∗x_ : _TxM →_ _TF_ ( _x_ ) _N_, is defined as:

_F∗x_ ( **v** )( _f_ ) := **v** ( _f_ _◦_ _F_ ) _,_ _∀f_ _∈_ _C_ _[∞]_ ( _V_ ) _,_ **v** _∈_ _TxM._


When a coordinate system ( _U, ϕ_ ) around _x_ and ( _V, ψ_ ) around _F_ ( _x_ ) are chosen, the coordinate
expression for _F∗x_ is just the Jacobian matrix of its vectorized form _ψ ◦_ _F_ _◦_ _ϕ_ _[−]_ [1], _i.e._, _∇_ ( _ψ ◦_ _F_ _◦_
_ϕ_ _[−]_ [1] )( _x_ ). So _F∗_ is also called the differential of _F_ and also admits the notation d _F_ .


The **tangent bundle** _T_ _M_ of a smooth manifold _M_ is the union of the tangent spaces of each points,
_i.e._ _T_ _M_ := [�] _x∈M_ _[T][x][M]_ [.] [Similar to the total derivative of the smooth map in Euclidean space, the]
differential of a smooth map between smooth manifolds is a linear map between tangent spaces.


A **vector field v** on a smooth manifold _M_ is a correspondence that associates to each point _x ∈M_
a vector **v** _x_ _∈_ _TxM_ . The vector field is smooth if the mapping **v** : _M_ _→_ _T_ _M_ is smooth. Denote
all the smooth vector fields on _M_ by _X_ ( _M_ ). With the definition of a vector field, we can define
the solution of ordinary differential equation (ODE) on the manifold. The idea is similar to the
definition in Euclidean space, the solution of the ODE is a curve whose velocity at each point is the
same as the vector field.


**Definition 10.** Let **v** be a smooth vector field on the smooth manifold _M_ . An **integral curve of v**
is a differentiable curve _γ_ : [0 _, T_ ] _→M_ whose velocity at each point is equal to the value of **v** at
that point:

_γ_ _[′]_ ( _t_ ) = **v** _γ_ ( _t_ ) _,_ _∀t ∈_ [0 _, T_ ] _._


Let _Tx_ _[∗][M]_ [ be the dual space of] _[ T][x][M]_ [, which is called the cotangent space of] _[ M]_ [ at] _[ x]_ [.] [The] **[ cotangent]**
**bundle** _T_ _[∗]_ _M_ is the union of the cotangent space of each points, _i.e._ _T_ _[∗]_ _M_ := [�] _p∈M_ _[T][ ∗]_ _x_ _[M]_ [.]

**Definition 11.** A **1-form** Θ on smooth manifold _M_ is a correspondence that associates to each point
_x ∈M_ a covector Θ _x_ _∈_ _Tx_ _[∗][M]_ [.] [The 1-form is smooth if the mapping][ Θ :] _[ M →]_ _[T][ ∗][M]_ [ is smooth.]


With the definition of a smooth manifold, we can define a continuous group with good properties.


**Definition 12.** A **Lie group** is a smooth manifold _G_ that is also a group with the property that the
multiplication map _G_ _× G_ _→G,_ ( _g, h_ ) _�→_ _g · h_ and the inversion map _G_ _→G, g_ _�→_ _g_ _[−]_ [1] are both
smooth.


Define the left multiplication mapping _Lg_ ( _h_ ) = _gh_, which is introduced to differentiate _g_ as a Liegroup element and as an action on a group element. A vector field **v** on _G_ is said to be left-invariant
if it’s invariant under all left multiplications, _i.e._ ( _Lg_ ) _∗g_ _[′]_ ( **v** _g_ _[′]_ ) = **v** _gg_ _[′]_ .


**Definition 13.** A Lie algebra is a real vector space g endowed with a map called the bracket [ _·, ·_ ] :
g _×_ g _→_ g that satisfies the following properties for all _X, Y, Z_ _∈_ g:


    - Bilinearity: _∀a, b ∈_ R,


[ _aX_ + _bY, Z_ ] = _a_ [ _X, Z_ ] + _b_ [ _Y, Z_ ] _,_ [ _Z, aX_ + _bY_ ] = _a_ [ _Z, X_ ] + _b_ [ _Z, Y_ ];


    - Antisymmetry: [ _X, Y_ ] = _−_ [ _Y, X_ ];


    - Jacobi Identity: [ _X,_ [ _Y, Z_ ]] + [ _Y,_ [ _Z, X_ ]] + [ _Z,_ [ _X, Y_ ]] = 0.


The Lie algebra of all smooth left-invariant vector fields on a Lie group _G_ is called the **Lie algebra**
**of** _G_, which has the same dimension with _G_ .


**Example** **14.** The Lie algebra of the group SO(3), denoted by so(3), is given by all the 3dimensional antisymmetric matrices so(3) = _{_ **A** _∈_ R [3] _[×]_ [3] _|_ **A** + **A** _[⊤]_ = 0 _}_ .


17


Smooth manifold is a topological structure. If we want to define the ”length of the velocity” and
distance between two points on the manifold, a metric on the tangent space is required. Such a metric
endows the metric with an additional geometry structure. The formal definitions are as follows.
**Definition 15.** A **Riemannian metric** on a smooth manifold is a correspondence which associates
to each point _p_ of _M_ an inner product _⟨·, ·⟩x_ that varies smoothly on _M_ . In other words, for any
two smooth vector fields **u** _,_ **v**, _⟨_ **u** _,_ **v** _⟩_ is a smooth function on _M_ . A smooth manifold with a given
Riemannian metric is called a **Riemannian manifold** .


To define the ”difference” between tangent space at different points, we need to introduce a concept
called affine connection.
**Definition 16.** An **affine connection** _∇_ on a Riemannian manifold is a mapping

_∇_ : _X_ ( _M_ ) _× X_ ( _M_ ) _→_ _X_ ( _M_ )


which is denoted by ( **u** _,_ **v** ) _→∇_ **uv** which satisfies the following properties:


    - _∇_ **uv** is linear over _C_ _[∞]_ ( _M_ ) in **u** : _∀f_ [(1)] _, f_ [(2)] _∈_ _C_ _[∞]_ ( _M_ ) and **u** [(1)] _,_ **u** [(2)] _∈_ _X_ ( _M_ ),

_∇f_ (1) **u** (1)+ _f_ (2) **u** (2) **v** = _f_ [(1)] _∇_ **u** (1) **v** + _f_ [(2)] _∇_ **u** (2) **v** ;


    - _∇_ **uv** is linear over R in **v** : _∀a_ [(1)] _, a_ [(2)] _∈_ R and **v** [(1)] _,_ **v** [(2)] _∈_ _X_ ( _M_ ),

_∇_ **u** (1)( _a_ [(1)] **v** [(1)] + _a_ [(2)] **v** [(2)] ) = _a_ [(1)] _∇_ **uv** [(1)] + _a_ [(2)] _∇_ **uv** [(2)] ;


    - _∇_ satisfies the following product rule: _∀f_ _∈_ _C_ _[∞]_ ( _M_ ),


_∇_ **u** ( _f_ **v** ) = _f_ _∇_ **uv** + ( **u** _f_ ) **v** _._


A connection is called the **Levi-Civita connection** if satisfies the following additional properties:


    - _∇_ is compatible with metric: _∇_ **u**     - **v** [(1)] _,_ **v** [(2)][�] =     - _∇_ **uv** [(1)] _,_ **v** [(2)][�] +     - **v** [(1)] _, ∇_ **uv** [(2)][�] ;


    - _∇_ is torsion-free: _∇_ **uv** _−∇_ **vu** = **u** ( **v** ( _·_ )) _−_ **v** ( **u** ( _·_ )).


The Levi-Civita connection is the connection with nice properties. Its existence and uniqueness is a
fundamental result of Riemannian geometry.
**Theorem** **17.** _(Fundamental_ _Theorem_ _of_ _Riemannian_ _Geometry_ _(Lee,_ _2018,_ _Thm._ _5.10))_ _Assume_
( _M, ⟨·, ·⟩_ ) _is a Riemannian manifold._ _Then there exists a unique Levi-Civita connection._


As the end of this subsection, we introduce the Laplace-Beltrami operator on the manifold, which is
used to define the Wiener process on the manifold.
**Definition 18.** Let _∇_ be the Levi-Civita connection on _M_ . The Hessian of _f_ _∈_ _C_ _[∞]_ ( _M_ ) is defined
by

Hess( _f_ )( **u** _,_ **v** ) := **v** ( **u** ( _f_ )) _−_ ( _∇_ **vu** )( _f_ ) _,_ _∀_ **u** _,_ **v** _∈_ _X_ ( _M_ ) _._


The Laplace-Beltrami operator ∆ is defined as the trace of Hessian. In other words, ∆ _f_ :=

- _M_
_i_ =1 [Hess(] **[e]** _[i][,]_ **[ e]** _[i]_ [)][ where] _[ {]_ **[e]** [1] _[, ...,]_ **[ e]** _[M]_ _[}]_ [ is an orthonormal basis for] _[ T][x][M]_ [.]


B.2 STOCHASTIC CALCULUS ON A MANIFOLD


With the Riemannian structure defined in the previous section, we can consider the definition of
stochastic differential equations (SDE) and diffusion processes on the manifold. For a systematic
treatment of the subject, please refer to standard textbooks Hsu (2002); Thalmaier (2023). First, we
recall the definition of SDE and diffusion process in Euclidean space.
**Definition 19.** The _infinitesimal generator_ of a stochastic process ( **x** _t_ ) _t_ for a function _ϕ_ ( **x** ) is

_Ltϕ_ ( **x** ) = lim E[ _ϕ_ ( **x** _t_ + _s_ ) _|_ **x** _t_ = **x** ] _−_ _ϕ_ ( **x** ) _,_
_s→_ 0 [+] _s_

where _ϕ_ is a suitably regular function. For an Itˆo process defined as the solution to the SDE d **x** _t_ =
**f** ( **x** _t, t_ ) d _t_ + **Σ** ( **x** _t, t_ ) d **w** _t_, the generator is


_Lt_ =


_D_


- **f** _[i]_ ( **x** _, t_ ) _∂i_ + [1]

2

_i_ =1


2


_D_


_i,j_ =1


18


- **Σ** ( **x** _, t_ ) **Σ** ( **x** _, t_ ) _[⊤]_ [�] _[ij]_ _∂i∂j._


On the other hand, the diffusion process can also be defined by its generator.


**Definition** **20.** A _D_ -dimensional stochastic process **x** _t_ with continuous sample path defined on a
probability space (Ω _, F_ _,_ P) is called a diffusion process generated by a smooth second-order elliptic
operator _Lt_ if the following hold: _∀f_ _∈_ _C_ _[∞]_ (R _[D]_ ), the process

                    - _t_
_Mt_ _[f]_ [=] _[ f]_ [(] **[x]** _[t]_ [)] _[ −]_ _[f]_ [(] **[x]** [0][)] _[ −]_ _Lsf_ ( **x** _s_ ) d _s_

0

is a _Ft_ -martingale.


To generalize the definition of SDE to a Riemannian manifold _M_, we need to define the secondorder differential operator on the manifold. Let _M_ be an _M_ -dimensional Riemannian manifold. A
second-order partial differential operator on _M_ is of the form


_L_ = **v** [(0)] +


_R_

- **v** [(] _[k]_ [)2] _,_ where **v** [(] _[k]_ [)] _∈_ _X_ ( _M_ ) _,_


_k_ =1


for some _R ∈_ N [+] . The square of a vector field is understood as the decomposition of derivatives:

**v** [(] _[k]_ [)2] ( _f_ ) := **v** [(] _[k]_ [)] ( **v** [(] _[k]_ [)] ( _f_ )) _,_ _∀f_ _∈_ _C_ _[∞]_ ( _M_ ) _._


The vector fields can be generalized to the time-dependent case. Now we can extend the definition
of a diffusion process on a Riemannian manifold.


**Definition 21.** (Thalmaier, 2023, Def. 1.1.3) Let (Ω _, F_ _,_ P; ( _F_ ) _t_ ⩾0) be a probability space equipped
with increasing sequence of sub- _σ_ -algebra _Ft_ _⊆_ _F_ . An adapted continuous process **x** _t_ taking
values in _M_, is called _Lt_ -diffusion if for all test functions _f_ _∈_ _Cc_ _[∞]_ [(] _[M]_ [)][, the process]

                  - _t_
_Nt_ _[f]_ [:=] _[ f]_ [(] **[x]** _[t]_ [)] _[ −]_ _[f]_ [(] **[x]** [0][)] _[ −]_ ( _Lsf_ )( **x** _s_ ) d _s,_ _t_ ⩾ 0 _,_

0

is a martingale, _i.e._ E[ _Nt_ _[f]_ _[−]_ _[N]_ _s_ _[ f]_ _[|][ F][s]_ [] = 0] _[,]_ _∀s_ ⩽ _t_ .


For a special case, we can define the Wiener process on the Riemannian manifold _M_ .

**Definition** **22.** A Wiener process **w** _t_ on _M_ is a diffusion process with generator [1] 2 [∆][,] [where] [∆] [is]

the Laplace-Beltrami operator of _M_, _i.e._ **w** _t_ is a continuous stochastic process on _M_ such that for
any _f_ _∈_ _C_ _[∞]_ ( _M_ ),


_f_ ( **x** _t_ ) _−_ [1]

2


- _t_

∆ _f_ ( **w** _s_ ) d _s,_
0


is a local martingale up to a valid time period.


For stochastic differential geometry, the Stratonovitch integral is more convenient than the Itˆo Integral. The Stratonovitch differential effectively subsumes the deterministic second-order effect of
the Wiener process from the quadratic variation into the drift term, so that it satisfies the ordinary
chain rule of calculus. This property enables a clear correspondence between the diffusion process
under a diffeomorphism between two Riemannian manifolds. Next, we give the definition of the
Stratonovitch integral on the Euclidean space and its generalization to Riemannian manifolds.


**Definition 23.** For continuous real-valued semimartingales **x** and **y**, let **x** _◦_ d **y** := **x** d **y** + [1]


**Definition 23.** For continuous real-valued semimartingales **x** and **y**, let **x** _◦_ d **y** := **x** d **y** + 2 [d[] **[x]** _[,]_ **[ y]** []]

be the Stratonovitch differential. Here **x** d **y** is the usual Itˆo differential, and d[ **x** _,_ **y** ] := d **x** d **y** is the
quadratic co-variation of **x** and **y** . The integral

           - _t_            - _t_            -            


_t_ - _t_

**x** _◦_ d **y** =
0 0


**x** d **y** + [1]


0


2 [d[] **[x]** _[,]_ **[ y]** []] _[t]_


is called the Stratonovitch integral of **x** with respect to **y** .


**Proposition 24.** _(Itˆo-Stratonovitch formula (Thalmaier, 2023, Prop. 1.2.10)). Let_ **x** _be a continuous_
R _[D]_ _-valued semimartingale and f_ _∈_ _C_ _[∞]_ (R _[D]_ ) _._ _Then_ d _f_ ( **x** ) = _⟨∇f_ ( **x** ) _, ◦_ d **x** _⟩._


The Itˆo-Stratonovitch formula shows the advantage of the Stratonovich differential: it satisfies the
usual chain rule of classical calculus. So at least formally, classical differential calculus can be
applied in calculations involving Stratonovich differentials.


19


**Proposition 25.** _(Thalmaier, 2023, Prop. 1.2.11) Solutions to the Stratonovitch SDE_


d **x** _t_ = **b** ( **x** _t, t_ ) d _t_ + **Σ** ( **x** _t, t_ ) _◦_ d **w** _t_ (12)

_define L-diffusions for the operator_


_D_

- **Σ** _[i]_ _k_ _[∂][i][.]_


_i_ =1


_L_ = **v** [(0)] + [1]

2


_D_


**v** [(] _[k]_ [)2] _,_ _where_ **v** [(0)] = **b** _,_ **v** [(] _[k]_ [)] =

_k_ =1


From this result, we can see that Eq. (12) describes the same diffusion process as the following Itˆo
SDE:


   d **x** _t_ = **b** ( **x** _t, t_ ) + [1]

2


_D_

- **v** _∗_ [(] _[k]_ [)][(] **[v]** [(] _[k]_ [)][)] - d _t_ + **Σ** ( **x** _t, t_ ) d **w** _t,_


_k_ =1


where **v** _∗_ [(] _[k]_ [)][(] **[v]** [(] _[k]_ [)][) :=][ �] _[D]_ _i,j_ =1 [(] _[∂][j]_ **[v]** [(] _[k]_ [)] _[i]_ [)] **[v]** [(] _[k]_ [)] _[j][∂][i]_ [.]


Now we can generalize the definition of SDE to the Riemannian manifold case. An SDE on manifold
_M_ can be defined by vector fields **v** [(0)] _,_ **v** [(1)] _, ...,_ **v** [(] _[M]_ [)] on _M_ . Let **w** be the R _[M]_ -valued Wiener
process and **x** 0 be an _M_ -valued random variable serving as the initial value of the solution. The
equation is symbolically written as


d **x** _t_ = **v** [(0)] ( **x** _t, t_ ) d _t_ +


_D_

- **v** [(] _[k]_ [)] ( **x** _t, t_ ) _◦_ d **w** _t_ _[k][.]_ (13)

_k_ =1


**Definition 26.** An _M_ -valued semimartingale **x** _t_ defined up to a proper stopping time _τ_ is a solution
to the SDE Eq. (13) up to _τ_ if for all _f_ _∈_ _C_ _[∞]_ ( _M_ ),


_D_

- **v** [(] _[k]_ [)] ( _f_ )( **x** _s, s_ ) _◦_ d **w** _t_ _[k]_


_k_ =1


       - _t_
_f_ ( **x** _t_ ) = _f_ ( **x** 0) +

0


**v** [(0)] ( _f_ )( **x** _s, s_ ) d _s_ +


_,_ 0 ⩽ _t < τ._


**Proposition 27.** _(Thalmaier, 2023, Cor. 1.2.19) Let L_ = **v** [(0)] + [1] 2 - _Dk_ =1 **[v]** [(] _[k]_ [)2] _[ and]_ **[ x]** _[t][ be the solution]_

_to the SDE Eq._ (13) _._ _Then for all f_ _∈_ _C_ _[∞]_ ( _M_ ) _,_


         - _t_
_Nt_ _[f]_ [:=] _[ f]_ [(] **[x]** _[t]_ [)] _[ −]_ _[f]_ [(] **[x]** [0][)] _[ −]_ ( _Lsf_ )( **x** _s_ )d _s,_ _t_ ⩾ 0 _,_

0


_is_ _a_ _martingale._ _In_ _other_ _words,_ _the_ _solution_ _of_ _SDE_ _Eq._ (13) _is_ _a_ _L_ _diffusion_ _to_ _the_ _operator_
_L_ = **v** [(0)] + [1] 2 - _Dk_ =1 **[v]** [(] _[k]_ [)2] _[.]_


C CONSTRUCTION OF QUOTIENT SPACE


In this section, we describe a rigorous construction of the quotient space and endow it with a manifold structure. Please refer to the standard textbooks Lee (2018) for the systematic treatments.
Assume that the total space _M_ is a Riemannian manifold and _G_ is a compact Lie group. First we
give the formal definition of the group action.


**Definition 28.** Let _G_ be a group and _M_ is a Riemannian manifold. A left action of _G_ on _M_ is a map
_G ×M →M_, ( _g,_ **x** ) _�→_ _g_ _·_ **x**, satisfying _g_ 1 _·_ ( _g_ 2 _·_ **x** ) = ( _g_ 1 _g_ 2) _·_ **x** and _e_ _·_ **x** = **x** _, ∀g_ 1 _, g_ 2 _∈G,_ **x** _∈M_ .
An action is smooth if its defining map _G_ _×_ _M_ _→M_ is smooth. We also reload the notation
_Lg_ ( **x** ) := _g ·_ **x** for distinguishing _g_ from its action on the manifold.


In the case where the Lie group acts on a Riemannian manifold, to draw meaningful conclusions,
we would expect some compatibility between group action and the Riemannian metric, which is the
concept of an isometric action. Moreover, to ensure the topological structure of the quotient space
so as to define useful constructions on the quotient space, concepts of a free action and proper action
are introduced.


**Definition 29.** **(1)** A smooth action is said to be an _isometric_ action if the map _Lg_ : _M →M,_ **x** _�→_
_g ·_ **x** is an isometry for any _g_ _∈G_, _i.e._,

_⟨_ **u** _,_ **v** _⟩_ **x** = _⟨_ ( _Lg_ ) _∗_ **x** ( **u** ) _,_ ( _Lg_ ) _∗_ **x** ( **v** ) _⟩g·_ **x** _._ (14)


20


**(2)** A smooth action is said to be _free_ if for any **x** _∈M_, _g ·_ **x** = **x** indicates _g_ = _e_ . **(3)** A smooth
action is said to be _proper_ if the map _G_ _× M_ _→M × M,_ ( _g,_ **x** ) _�→_ ( _g_ _·_ **x** _,_ **x** ) is a proper map,
meaning that the preimage of every compact set is compact.


For the properness, there is a convenient characterization.
**Proposition** **30.** _(Lee,_ _2018,_ _Prop._ _C.15)_ _Assume_ _G_ _is_ _a_ _Lie_ _group_ _acting_ _smoothly_ _on_ _the_ _smooth_
_manifold_ _M._ _The_ _action_ _is_ _proper_ _if_ _and_ _only_ _if_ _the_ _following_ _condition_ _is_ _satisfied:_ _if_ ( _pi_ ) _is_
_a_ _sequence_ _in_ _M_ _and_ ( _gi_ ) _is_ _a_ _sequence_ _in_ _G_ _such_ _that_ _both_ ( _pi_ ) _and_ ( _gi_ _· pi_ ) _converge,_ _then_ _a_
_subsequence_ _of_ ( _gi_ ) _converges._ _Particularly,_ _every_ _smooth_ _action_ _by_ _a_ _compact_ _Lie_ _group_ _on_ _a_
_smooth manifold is proper._


The group action typically represents a symmetry in the sense that points that can be transformed to
each other by a group action are regarded as symmetric, _i.e._, they are equivalent. Therefore, we can
define an equivalence relation _∼_ on _M_ as **x** [(1)] _∼_ **x** [(2)] if _∃g_ _∈G,_ **x** [(1)] = _g ·_ **x** [(2)] . The equivalence
class with representative **x** is defined as the set of all points that are equivalent to **x** . The quotient
space _Q_ := _M/G_ (as a set) is defined under this equivalence relation, which consists of equivalence
classes under the relation _∼_ . The original space _M_ is referred to as the total space. There is a
natural mapping called projection that connects the total space and the quotient space, which maps
any **x** _∈M_ to the equivalent class it represents. In this case where the equivalence is defined by a
Lie group, the projection mapping can be written as:


_π_ : _M →Q,_ _π_ ( **x** ) := _{g ·_ **x** _| g_ _∈G}._


Due to this expression, the equivalent class in such a case is the orbit of the Lie group _G_ at **x** .
Therefore, it can be understood that an equivalent class is a “representation” (literal meaning; not
the mathematical concept) of the Lie group, hence can also adopt manifold structures of _G_ under the
mentioned “good” conditions. Also, the ( _M, Q, π_ ) structure forms a fiber bundle, in which context
the equivalent class is also called a fiber at _π_ ( **x** ), and this special fiber bundle induced from a Lie
group action is called a principal _G_ -bundle.


Moreover, under certain conditions, the quotient space inherits the Riemannian structure of the total
space _M_ through the projection mapping.
**Theorem 31.** _(Lee, 2018, Cor. 2.29) Let M be a Riemannian manifold, and G be a Lie group acting_
_smoothly,_ _freely,_ _properly,_ _and_ _isometrically_ _on_ _M._ _Then_ _the_ _quotient_ _space_ _M/G_ _has_ _a_ _unique_
_smooth manifold structure and Riemannian metric such that π is a Riemannian submersion._


We will assume the conditions, _i.e._, _G_ be a Lie group acting smoothly, freely, properly, and isometrically on _M_, in the following development. Given that _Q_ is a smooth manifold, the projection
mapping induces a linear mapping _π∗_ between the tangent spaces of the two manifolds. It introduces
more structures in the total space _M_ . In each tangent space _TxM_, we can define a subspace of it,
called the _vertical space_, by the kernel of _π∗_ :

_V_ **x** := Ker _π∗_ **x** _._


By this definition, tangent vectors in the vertical space can be understood that it does not move **x** in
a way that alters the projection onto _Q_ by _π_, so the movement stays within the equivalent class. The
vertical space can then be understood as the tangent space of the equivalent class. As mentioned
above, in this case where the quotient space is induced from the Lie group _G_, the equivalent class is
a “representation” of the Lie group, hence the vertical space is a “mirror” of the tangent space of the
Lie group, which is in turn isomorphic to the Lie algebra g of the Lie group _G_ .


To complete the whole tangent space, a concept of horizontal space _H_ **x** is expected. In general, the
horizontal space _H_ **x** is a linear subspace of _T_ **x** _M_ that makes up _T_ **x** _M_ by direct sum with _V_ **x** :

_T_ **x** _M_ = _V_ **x** _⊕H_ **x** _._

Under this direct-sum construction, any tangent vector **v** _∈_ _T_ **x** _M_ can then be _uniquely_ decomposed
into the vertical and horizontal components, **v** = **v** _[V]_ + **v** _[H]_ . Correspondingly, a vector field on _M_ is
called a vertical/horizontal vector field if it takes a vertical/horizontal tangent vector at every point.
Every smooth vector field **v** on _M_ can be expressed uniquely in the form **v** = **v** _[V]_ + **v** _[H]_, where both
the vertical and horizontal vector fields are smooth (Lee, 2018, Prop. 2.25). For future reference, we
assign a convenient notation to the horizontal projection within _T_ **x** _M_ itself:

_P_ **x** ( **v** ) := **v** _[H]_ _,_ _∀_ **v** _∈_ _T_ **x** _M._


21


Nevertheless, the horizontal space _H_ **x** as a subspace that makes up the tangent space _T_ **x** _M_ by the
direct sum with _V_ **x** is not unique. Therefore, a smooth correspondence from **x** to such an _H_ **x** is
an independent structure, referred to as the “connection” in the fiber-bundle context. In the current
specific case where _M_ endows a Riemannian structure, we can uniquely define the horizontal space
as the orthogonal complement under the inner product in the tangent space:

_H_ **x** := _V_ **x** _⊥_ ( _T_ **x** _M,⟨·,·⟩_ **x** ) _,_ (15)


which gives a canonical “connection”.


As would be expected, in contrast to vertical tangent vectors, a horizontal tangent vector represents a
movement through different equivalent classes, corresponding to a movement on the quotient space
_Q_ . Therefore, we can construct the concept of horizontal lift which establishes a correspondence
from a vector field on _Q_ to a horizontal vector field on _M_ .


**Definition** **32.** Given a vector field **u** on _Q_, a vector field **u** ˜ on _M_ is called a _horizontal_ _lift_ of
**u**, if **u** ˜ is a horizontal vector field, _i.e._, **u** ˜ **x** _∈H_ **x** for all **x** _∈M_, and **u** ˜ is _π_ -related to **u** by
_π∗_ **x** (˜ **ux** ) = **u** _π_ ( **x** ).

**Proposition** **33.** _(Lee,_ _2018,_ _Prop._ _2.25)_ _Given_ _a_ _smooth_ _connection_ **x** _�→H_ **x** _and_ _assuming_ _π_ :
_M_ _→Q_ _is_ _a_ _smooth_ _submersion,_ _every_ _smooth_ _vector_ _field_ _on_ _Q_ _always_ _has_ _a_ _unique_ _smooth_
_horizontal lift to M._


If the connection is induced from the Riemannian structure of _M_ by Eq. (15) and if the group
action is isometric, then a nice compatibility can be derived. For a quotient-space tangent vector
**u** _∈_ _T_ **y** _Q_ at some **y** _∈Q_, consider two ways to construct a tangent vector at some point **x** _∈_
_π_ _[−]_ [1] ( **y** ) in the equivalent class. The first way is directly by the horizontal lift, which gives **u** ˜ **x**,
which is the unique horizontal tangent vector such that _π∗_ **x** (˜ **ux** ) = **u** . The other way is to first
horizontal lift **u** to another point **x** _[′]_ _∈_ _π_ _[−]_ [1] ( **y** ) in the equivalent class, then push it forward to
the tangent space at **x** by a transformation that maps **x** _[′]_ to **x** . Since both points lie in the same
equivalent class and the Lie group acts on the manifold freely, there exists a unique group action
_g_ _∈G_ such that **x** = _g_ _·_ **x** _[′]_ = _Lg_ ( **x** _[′]_ ), so the resulting tangent vector is ( _Lg_ ) _∗_ **x** _′_ (˜ **ux** _′_ ). Noting
that ( _Lg_ ) _∗_ **x** _′_ preserves the metric between _T_ **x** _′M_ and _T_ **x** _M_ (see Eq. (14)), we know that it also
preserves the horizontal spaces, _i.e._, ( _Lg_ ) _∗_ **x** _′_ ( _H_ **x** _′_ ) = _H_ **x**, so ( _Lg_ ) _∗_ **x** _′_ (˜ **ux** _′_ ) _∈H_ **x** . Moreover,
_π∗_ **x** �( _Lg_ ) _∗_ **x** _′_ (˜ **ux** _′_ )� = ( _π◦Lg_ ) _∗_ **x** _′_ (˜ **ux** _′_ ) = _π∗_ **x** _′_ (˜ **ux** _′_ ) = **u** also projects to the quotient-space tangent
vector **u** (noting that _π ◦_ _Lg_ = _π_ for any _g_ _∈G_, and recalling the definition of horizontal lift), by
the uniqueness of the horizontal tangent vector that projects to **u**, we have **u** ˜ **x** = ( _Lg_ ) _∗_ **x** _′_ (˜ **ux** _′_ ), or
equivalently,
**u** ˜ _g·_ **x** = ( _Lg_ ) _∗_ **x** (˜ **ux** ) _,_ _∀_ **x** _∈_ _π_ _[−]_ [1] ( **y** ) _, g_ _∈G._ (16)


The unique existence of the correspondence from _Tπ_ ( **x** ) _Q_ back to _T_ **x** _M_ by horizontal lift (Prop. 33)
allows us to introduce a Riemannian structure on _Q_ from that on _M_ . For any **y** _∈Q_ and **u** [(1)] _,_ **u** [(2)] _∈_
_T_ **y** _Q_, define:

_⟨_ **u** [(1)] _,_ **u** [(2)] _⟩_ _[Q]_ **y** [:=] _[ ⟨]_ **[u]** [˜] **x** [(1)] _[,]_ [ ˜] **[u]** **x** [(2)] _[⟩]_ **x** _[M][,]_ (17)

for any **x** _∈_ _π_ _[−]_ [1] ( **y** ). This is well-defined since the right hand side is independent the choice
of **x** due to the horizontal-lift–push-forward compatibility (Eq. (16)) and isometry (Eq. (14)):
_⟨_ **u** ˜ [(1)] _g·_ **x** _[,]_ [ ˜] **[u]** _g_ [(2)] _·_ **x** _[⟩][M]_ _g·_ **x** [=] _[ ⟨]_ [(] _[L][g]_ [)] _[∗]_ **[x]** [(˜] **[u]** **x** [(1)][)] _[,]_ [ (] _[L]_ _g_ [)] _∗_ **x** [(˜] **[u]** **x** [(2)][)] _[⟩][M]_ _g·_ **x** [=] _[ ⟨]_ **[u]** [˜] **x** [(1)] _[,]_ [ ˜] **[u]** **x** [(2)] _[⟩][M]_ **x** [.]


Subsequent constructions on the quotient space _Q_ can be induced from this Riemannian structure.
Due to its compatibility with the original Riemannian manifold _M_, these constructions have direct
connections to their counterparts on _M_ . Particularly, the Levi-Civita connections on _Q_ and _M_
follow the relation below.

**Proposition** **34.** _(Lee,_ _2018,_ _Exercise._ _5.6)_ _Let_ _∇_ _[M]_ _and_ _∇_ _[Q]_ _denote_ _the_ _Levi-Civita_ _connections_
_on M, Q, respectively, where ∇_ _[Q]_ _is constructed from the Riemannian metric induced from that of_
_M by Eq._ (17) _._ _Then for any vector fields_ **u** [(1)] _,_ **u** [(2)] _on Q, denoting their horizontal lifts to M as_
**u** ˜ [(1)] _,_ ˜ **u** [(2)] _, we have:_


_∇_ _[Q]_ **u**         - [(1)] **[u]** [(2)] [= (] _[∇][M]_ **u** ˜ [(1)][ ˜] **[u]** [(2)][)] _[H][,]_


_where_ _∇_ _[Q]_ - _[denotes the horizontal lift of the vector field][ ∇][Q]_ _[on][ Q][.]_
**u** [(1)] **[u]** [(2)] **u** [(1)] **[u]** [(2)]


22


C.1 THE SHAPE SPACE R [3] _[N]_ _/_ SE(3)


For a concrete and practically highly concerned example, we consider the shape space R [3] _[N]_ _/_ SE(3).
In this example, each R [3] _[N]_ element is structured as:

**x** = ( **x** [(1)] _,_ **x** [(2)] _, · · ·_ _,_ **x** [(] _[N]_ [)] ) _∈_ R [3] _[N]_ _,_ with each **x** [(] _[i]_ [)] _∈_ R [3] _,_


which represents the 3-dimensional coordinates of _N_ points in R [3] (point cloud). The SE(3) group
is composed of the 3-dimensional translation group and the 3-dimensional rotation group SO(3).
Since the translation group is not compact, there does not exist a probability distribution that is
translation invariant. We (as well as many others (Yim et al., 2023; Lin et al., 2024)) hence represent
the quotient space w.r.t this group by suppressing these equivalent DoFs by choosing a canonical
translational position by anchoring the center of mass (CoM) of the point cloud at the origin, and
consider the resulting CoM-free subspace _M_ _[◦]_ := _{_ **x** _∈_ R [3] _[N]_ _|_ _N_ 1 - _Ni_ =1 **[x]** [(] _[i]_ [)] [=] **[0]** _[}]_ [6] [and consider]
the SO(3) action on it. Since the constraint is linear, this space _M_ _[◦]_ is a linear subspace of R [3] _[N]_,
and it is naturally a Riemannian manifold with the standard inner product of R [3] _[N]_ . An element of
the SO(3) group is given by a 3 _×_ 3 rotation matrix for which we reload the notation _g_ . The natural
action of _g_ on **x** is defined as _g ·_ **x** = - _g_ **x** [(1)] _,_ _g_ **x** [(2)] _,_ _· · ·_ _,_ _g_ **x** [(] _[N]_ [)][�], _i.e._ the rotation is applied on
each point of the system.


Unfortunately, SO(3) does not act freely (see Def. 29) on _M_ _[◦]_ in some degenerate cases, _e.g._ all
the points lie on a straight line. So we define the subset _D_ _⊂M_ _[◦]_ that SO(3) does not have free
action on it; _i.e._, for any **x** _∈D_, there exists a nontrivial action _g_ = _e ∈_ SO(3) such that _g ·_ **x** = **x**,
indicating that _D_ contains points that have a higher symmetry beyond SO(3). For a converging
sequence _{_ **x** _i}i_ in _D_ which converges to **x** _∈M_ _[◦]_ as _i_ _→∞_, there exists a sequence _{gi}i_ in _G_
such that _gi ·_ **x** _i_ = **x** _i_ . Since the group SO(3) is compact and the group action is continuous, _{gi}i_
has a convergent subsequence that converges to _g_ _∈G_, which satisfies _g ·_ **x** = **x** . Hence **x** _∈D_,
and therefore, _D_ is closed. Subsequently, _M_ := _M_ _[◦]_ _\_ _D_ is still a smooth manifold. As _D_ is
measure-zero in _M_ _[◦]_ (since any _g_ _∈_ SO(3) is non-singular, the equation _g ·_ **x** = **x** reduces degrees
of freedom of **x** ), it is unlikely for a real simulation in _M_ _[◦]_ to hit the set _D_, making negligible
difference algorithmically.


By removing the degenerate set _D_, SO(3) can now act freely and smoothly on _M_ . Moreover, since
the SO(3) action is isometric in the Euclidean space and _M_ inherits the same metric, SO(3) also
acts isometrically on _M_ . Since SO(3) is a compact group, by Prop. 30, the action is also proper.
Now that the action is smooth, free, proper, and isometric, by Thm. 31, the quotient space _Q_ :=
_M/_ SO(3) is a Riemannian manifold and the projection _π_ : _M_ _→Q_ is a Riemannian submersion.
Since the each element in this quotient space _Q_ is an equivalent class containing equivalent pointcloud configurations, we refer to this space _Q_ as the “shape space”.


By the projection mapping _π_ : _M_ _→Q_, the vertical space _V_ **x** := Ker _π∗_ **x** can already be defined,
which reflects the infinitesimal movements in _M_ by group actions, which amounts to movements
within the equivalent class _π_ ( **x** ) (Appx. B). Since _M_ is a Riemannian manifold (tangent space inner
product is inherited from the standard Euclidean inner product), we can define the horizontal space
_H_ **x** := ( _V_ **x** ) _[⊥][T]_ **[x]** _[M]_ as the orthogonal complement of _V_ **x** in _T_ **x** _M_ . Since _V_ **x** and _H_ **x** recover _T_ **x** _M_
by direct sum, any tangent vector **v** _∈_ _T_ **x** _M_ can thus be uniquely decomposed as the addition of a
vertical component and horizontal component.


On this concrete example, the vertical and horizontal spaces can be expressed explicitly. Since the
vertical space is induced from group action, which acts freely, so this space is isomorphic to the
tangent space of the Lie group, _i.e._, the Lie algebra. For _G_ = SO(3), the Lie algebra so(3) is the set
of antisymmetric 3 _×_ 3 matrices. So the vertical space is given by:

_V_ **x** = _{_ ( **Ax** [(1)] _,_ **Ax** [(2)] _,_ _· · ·_ _,_ **Ax** [(] _[N]_ [)] ) _|_ **A** _∈_ so(3) _}._

Using the 3-dimensional representation **a** = ( **a** [1] _,_ **a** [2] _,_ **a** [3] ) _[⊤]_ _∈_ R [3] for so(3), any antisymmetric 3 _×_ 3


         - 0 _−_ **a** [3] **a** [2]
matrix can be represented as **A** = **a** [3] 0 _−_ **a** [1]

_−_ **a** [2] **a** [1] 0


. Following this representation, **Ax** [(] _[i]_ [)] = **a** _×_ **x** [(] _[i]_ [)],


6Here we choose a simple form of CoM by treating atoms equally weighted to avoid unnecessary notation
complexity. In fact, any choice to determine one point in R [3] from the _N_ points suffices the reduction of the
translation DoFs (as long as proper permutational invariance is guaranteed).


23


where “ _×_ ” denotes the usual cross product on R [3], so the vertical space can also be written as:

_V_ **x** = _{_ ( **a** _×_ **x** [(1)] _,_ **a** _×_ **x** [(2)] _,_ _· · ·_ _,_ **a** _×_ **x** [(] _[N]_ [)] ) _|_ **a** _∈_ R [3] _}._


The horizontal space, which is the orthogonal complement of the vertical space, is given by


_N_

- **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] = **0** - _,_


_i_ =1


  _H_ **x** = **v** = ( **v** [(1)] _, · · ·_ _,_ **v** [(] _[N]_ [)] ) _∈_ R [3] _[N]_ [��]

                        


_N_

- **v** [(] _[i]_ [)] = 0 _,_


_i_ =1


since the **v** [(] _[i]_ [)] vectors should keep the CoM fixed, and as the orthogonal complement, they should
also satisfy [�] _i_ _[N]_ =1 **[v]** [(] _[i]_ [)] _[ ·]_ [ (] **[a]** _[ ×]_ **[ x]** [(] _[i]_ [)][) =] **[ a]** _[ ·]_ [ (][�] _i_ _[N]_ =1 **[x]** [(] _[i]_ [)] _[ ×]_ **[ v]** [(] _[i]_ [)][) =] **[ 0]** [ for any] **[ a]** _[ ∈]_ [R][3][.]

Given the construction of the horizontal space ( _i.e._, a “connection”), the horizontal lift for a quotientspace tangent vector (and vector field) can be derived. As SO(3) acts smoothly, freely, properly, and
isometrically on _M_, a Riemannian structure can be induced for _Q_ from that of _M_, which is inherited
from the standard Euclidean metric.


D PROOFS


D.1 PROOF OF THM. 1


**Theorem 1’.** Assume _{_ **x** _t}t∈_ [0 _,T_ ] is a diffusion process on _M_, specified by the following SDE:


d **x** _t_ = **b** _t_ ( **x** _t_ ) d _t_ + _σt_ d **w** _t,_ **x** 0 _∼_ _p_ prior _,_ (6’)

where **b** _t_ is a _G_ -equivariant time-dependent vector field on _M_, **w** _t_ is the Wiener process on _M_ that is
_G_ -invariant, and _p_ prior is a _G_ -invariant distribution. Then the projected process _{_ **y** _t_ := _π_ ( **x** _t_ ) _}t∈_ [0 _,T_ ]
onto the quotient space _Q_ := _M/G_ is the solution to the following SDE:


       -       d **y** _t_ = ( _π∗_ **b** _t_ )( **y** _t_ ) _−_ _[σ]_ _t_ [2] d _t_ + _σt_ d **ω** _t,_ **y** 0 _∼_ _π_ # _p_ prior _,_ (7’)
2 **[h]** [(] **[y]** _[t]_ [)]


where: **(1)** _π∗_ **b** _t_ is the pushed-forward vector field of **b** _t_ induced by _π_, _i.e._, ( _π∗_ **b** _t_ )( **y** _t_ ) :=
_π∗_ **x** _t_ **b** _t_ ( **x** _t_ ), which is the same for any **x** _t_ _∈_ _π_ _[−]_ [1] ( **y** _t_ ) due to the _G_ -equivariance of **b** _t_ ; **(2) h** ( **y** _t_ ) :=
_π∗_ **x** _t_ ( [�] _[M]_ _i_ = _M_ _−G_ +1 _[∇]_ **[e]** _i_ **[e]** _[i]_ [)][ for any] **[ x]** _[t]_ _[∈]_ _[π][−]_ [1][(] **[y]** _[t]_ [)][ is the mean curvature vector at] **[ y]** _[t]_ [, where] _[ {]_ **[e]** _[i][}][M]_ _i_ =1
is an orthonormal basis of _T_ **x** _tM_ such that _V_ **x** _t_ = span _{_ **e** _i}_ _[M]_ _i_ = _M_ _−G_ +1 [;] **[ (3)][ ω]** _[t]_ [ is the Wiener process]
on _Q_ ; and **(4)** _π_ # _p_ prior is the pushed-forward distribution of _p_ prior, _i.e._, its samples can be produced
by **y** 0 = _π_ ( **x** 0) where **x** 0 _∼_ _p_ prior.


_**Proof**_ _._ As **x** _t_ is a diffusion process on _M_ given by the the SDE d **x** _t_ = **b** _t_ ( **x** _t_ ) d _t_ + _σt_ d **w** _t_, by
Prop. 27, **x** _t_ is a _Lt_ -diffusion and the generator is

_Lt_ = **b** _t_ + _[σ]_ _t_ [2]
2 [∆] _[M][.]_

Let _{_ **e** _i}_ _[M]_ _i_ =1 [be] [an] [orthonormal] [basis] [of] _[T]_ **[x]** _t_ _[M]_ [such] [that] _[H]_ **[x]** _t_ = span _{_ **e** _i}_ _[M]_ _i_ =1 _[−][G]_, _V_ **x** _t_ =
span _{_ **e** _i}_ _[M]_ _i_ = _M_ _−G_ +1 [.] Then by the Riemannian submersion construction of _π_ : _M_ _→Q_ (see
Appx. C), _{_ **e** ˜ _i_ := _π∗_ **xe** _i}_ _[M]_ _i_ =1 _[−][G]_ is an orthonormal basis of _Tπ_ ( **x** _t_ ) _Q_ . Let _∇_ _[M]_ and _∇_ _[Q]_ be the
Levi-Civita connections on _M, Q_, respectively, where _∇_ _[Q]_ is induced from the Riemannian metric inherited from _M_ . Using the local expression of the Laplace-Beltrami operator (Def. 18), the
generator is given by


_Lt_ = **b** _t_ + _[σ]_ _t_ [2] [=] **[ b]** _[t]_ [ +] _[σ]_ _t_ [2]
2 [∆] _[M]_ 2


_M_
�( **e** _i_ ( **e** _i_ ( _·_ )) _−∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)]

_i_ =1


_M_

- _∇_ _[M]_ **e** _i_ **[e]** _[i]_

_i_ =1


+ _[σ]_ _t_ [2]
2


_M_

- **e** [2] _i_ _[.]_

_i_ =1


=


**b** _t −_ _[σ]_ _t_ [2]
2


24


Then the process is the solution to the following Stratonovitch SDE


d **x** _t_ = **v** [(0)] ( **x** _t, t_ )d _t_ +


_M_

- **v** [(] _[i]_ [)] ( **x** _t, t_ ) _◦_ d **w** _t_ _[i][,]_


_i_ =1


where **v** [(0)] := **b** _t −_ _[σ]_ _t_ [2]
2


_M_

- _∇_ _[M]_ **e** _i_ **[e]** _[i][,]_

_i_ =1


and **v** [(] _[i]_ [)] := _σt_ **e** _i_ for _i_ = 1 _, · · ·_ _, M._

By Def. 26, for all _f_ _∈_ _C_ _[∞]_ ( _M_ ),


_._


       - _t_
_f_ ( **x** _t_ ) = _f_ ( **x** 0) +

0


**v** [(0)] ( _f_ )( **x** _s, s_ )d _s_ +


_M_

- **v** [(] _[i]_ [)] ( _f_ )( **x** _s, s_ ) _◦_ d **w** _s_ _[i]_


_i_ =1


_M_


Let _f_ [˜] _∈_ _C_ _[∞]_ ( _Q_ ), then _f_ := _f_ [˜] _◦_ _π_ _∈_ _C_ _[∞]_ ( _M_ ), then


**v** [(0)] ( _f_ [˜] _◦_ _π_ )( **x** _s, s_ )d _s_ +


         - _t_
_f_ ˜( _π_ ( **x** _t_ )) = _f_ ˜( _π_ ( **x** 0)) +

0


         - _t_
= _f_ [˜] ( _π_ ( **x** 0)) +

0


_M_

- **v** [(] _[i]_ [)] ( _f_ [˜] _◦_ _π_ )( **x** _s, s_ ) _◦_ d **w** _s_ _[i]_


_i_ =1


_,_


( _π∗_ **v** [(0)] )( _f_ [˜] )( _π_ ( **x** _s_ ) _, s_ )d _s_ +


_M_
�( _π∗_ **v** [(] _[i]_ [)] )( _f_ [˜] )( _π_ ( **x** _s_ ) _, s_ ) _◦_ d **w** _s_ _[i]_


_i_ =1


_,_


by Def. 9. Since _f_ [˜] is arbitrary, by Def. 26, **y** _t_ := _π_ ( **x** _t_ ) is the solution to


d **y** _t_ = _π∗_ **v** [(0)] ( **y** _t, t_ )d _t_ +


_M_

- _π∗_ **v** [(] _[i]_ [)] ( **y** _t, t_ ) _◦_ d **w** _t_ _[i][.]_


_i_ =1


We first need to check that the projected vector field is well defined. In fact, we only need to check
that _π∗_ **b** is well defined. Since **b** is _G_ -equivariant, then for any _g_ _∈G_, ( _Lg_ ) _∗_ **b** _t_ ( **x** ) = **b** _t_ ( _g_ _·_ **x** ).
Then _π∗_ ( **b** _t_ ( _g ·_ **x** )) = _π∗_ (( _Lg_ ) _∗_ **b** _t_ ( **x** )) = ( _π ◦_ _Lg_ ) _∗_ ( **b** _t_ ( **x** )) = _π∗_ ( **b** _t_ ( **x** )), where we have used the
chain rule in the second-last step, and the last step holds since _π_ _◦_ _Lg_ and _π_ projects to the same
equivalent class ( **x** and _g_ _·_ **x** are in the same equivalent class). By a notational equivalence that
_π∗_ ( **b** _t_ ( **x** )) = _π∗_ **b** _t_ ( _π_ ( **x** )), we know that _π∗_ **b** _t_ ( **y** ) is the same on the equivalent class regardless of
the choice of **x** in _π_ _[−]_ [1] ( **y** ), which implies that the projected vector field _π∗_ **b** _t_ is well defined.


Next, we calculate the expression of the projected vector field. Since _H_ **x** = span _{_ **e** 1 _, · · ·_ _,_ **e** _M_ _−G}_,
_V_ **x** = span _{_ **e** _M_ _−G_ +1 _, · · ·_ _,_ **e** _M_ _}_, we have

_π∗_ **xe** _i_ =              - **e** ˜ _i,_ if _i_ ⩽ _M_ _−_ _G,_
0 _,_ if _i_ ⩾ _M_ _−_ _G_ + 1 _,_


so _π∗_ **x** ( **v** [(] _[i]_ [)] ) = _σt_ **e** ˜ _i_ for _i_ = 1 _, · · ·_ _, M_ _−_ _G_ and _π∗_ **x** ( **v** [(] _[i]_ [)] ) = 0 for _i_ ⩾ _M_ _−_ _G_ + 1. For the drift
term, using Prop. 34, we have


_M_

 - _π∗_ ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)]

_i_ = _M_ _−G_ +1


_π∗_ **v** [(0)] ( **y** _, t_ ) = _π∗_ **b** _t_ ( **y** ) _−_ _[σ]_ _t_ [2]
2


= _π∗_ **b** _t_ ( **y** ) _−_ _[σ]_ _t_ [2]
2


= _π∗_ **b** _t_ ( **y** ) _−_ _[σ]_ _t_ [2]
2


= _π∗_ **b** _t_ ( **y** ) _−_ _[σ]_ _t_ [2]
2


_M_

- _π∗_ ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)]

_i_ =1


_M_ _−G_

- _π∗_ ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)] _[ −]_ _[σ]_ _t_ [2]

2
_i_ =1


_M_ _−G_

- _∇_ _[Q]_ **e** ˜ _i_ **[e]** [˜] _[i][ −]_ _[σ]_ _t_ [2]

2 **[h]** [(] **[y]** [)] _[.]_
_i_ =1


25


_M_ _−G_

- _∇_ _[Q]_ **e** ˜ _i_ **[e]** [˜] _[i][ −]_ _[σ]_ _t_ [2]

2
_i_ =1


_M_

 - _π∗_ ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)]

_i_ = _M_ _−G_ +1


So the generator of the process **y** _t_ is


_M_ _−G_

- **e** ˜ [2] _i_


_i_ =1


_L_ ˜ _s_ = _π∗_ **b** _t −_ _[σ]_ _t_ [2]
2


_M_ _−G_

- _∇_ _[Q]_ **e** ˜ _i_ **[e]** [˜] _[i][ −]_ _[σ]_ _t_ [2] _[σ]_ _t_ [2]

2 **[h]** [ +] 2
_i_ =1


_M_ _−G_

- _∇_ _[Q]_ **e** ˜ _i_ **[e]** [˜] _[i]_

_i_ =1


 - = _π∗_ **b** _t −_ _[σ]_ _t_ [2] + _[σ]_ _t_ [2]
2 **[h]** 2


- _M_ _−G_

 - **e** ˜ [2] _i_ _[−]_

_i_ =1


        -        = _π∗_ **b** _t −_ _[σ]_ _t_ [2] + _[σ]_ _t_ [2]
2 **[h]** 2 [∆] _[Q][.]_


Then we can conclude that the projected process **y** _t_ := _π_ ( **x** _t_ ) is the solution to the following SDE


         -          d **y** _t_ = ( _π∗_ **b** _t_ )( **y** _t_ ) _−_ _[σ]_ _t_ [2] d _t_ + _σt_ d **ω** _t,_
2 **[h]** [(] **[y]** _[t]_ [)]


where _π∗_ **b** _t_ is the push-forward vector field, **h** ( **y** _t_ ) is the mean curvature vector at **y** _t_ and **ω** _t_ is the
standard Wiener process on the quotient space _Q_ .


D.2 PROOF OF THM. 2


In Def. 32, we define the horizontal lift of a vector field that generates a deterministic flow. In fact,
for a stochastic process on _Q_, we can define the horizontal lift for it similarly. First, we need to
define the stochastic line integral, which is the integration of a one-form along the trajectory of a
stochastic process.


**Definition 35.** (Hsu, 2002, Prop. 2.4.2) Let Θ be a 1-form (Def. 11) on _M_ and **x** _t_ the solution to
the equation


d **x** _t_ = **v** [(0)] ( **x** _t, t_ )d _t_ +


_D_

- **v** [(] _[i]_ [)] ( **x** _t, t_ ) _◦_ d **w** _t_ _[i][.]_


_i_ =1


Then

       - _t_


_t_ - _t_

Θ( **v** [(0)] )( **x** _s_ ) d _s_ +
0 0


_t_ - _t_

Θ **x** _s_ d _s_ =
0 0


0


_D_

- Θ( **v** [(] _[i]_ [)] )( **x** _s_ ) _◦_ d **w** _s_ _[i]_ _[.]_


_i_ =1


**Definition 36.** (Baudoin et al., 2024, Def. 3.1.9) A semimartingale ( **x** _t_ ) _t_ on _M_ is called horizontal
if for every 1-form Θ on _M_ whose kernel contains the horizontal space _H_, one has �0 _t_ [Θ] **[x]** _[s]_ [ d] _[s]_ [ = 0][,]
for all _t_ ⩾ 0. Let ( **y** _t_ ) _t_ be a semimartingale on _Q_ such that **y** 0 is a point of _Q_ . Then for a given
starting point **x** 0 _∈_ _π_ _[−]_ [1] ( **y** 0), there exists a unique horizontal semimartingale **x** _t_ on _M_ such that **x** _t_
starts from **x** 0 and _π_ ( **x** _t_ ) = **y** _t_ for all _t_ ⩾ 0. This process ( **x** _t_ ) _t_ is called the horizontal lift of ( **y** _t_ ) _t_
from **x** 0.


**Theorem 2’.** The horizontal lift of Eq. (7) has the following explicit expression:


       -        d˜ **x** _t_ = _P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _−_ _[σ]_ 2 _t_ [2] **h** ˜(˜ **x** _t_ ) d _t_ + _σt_ d ˜ **w** _t,_ **x** ˜0 _∼_ _p_ prior _,_ (8’)

where _P_ **x** ( **v** ) := **v** _[H]_ is the horizontal projection in the tangent space of _M_, **h** [˜] is the horizontal lift
of the mean curvature vector, and **w** ˜ _t_ is the horizontal lift of the Wiener process on _Q_ .


_**Proof**_ _._ We only need to check the definition of the horizontal lift (Def. 36). Again, assume
_{_ **e** 1 _, ...,_ **e** _M_ _}_ is a local orthonormal basis of _M_ and _H_ **x** = span _{_ **e** 1 _, · · ·_ _,_ **e** _M_ _−G}_, _V_ **x** =
span _{_ **e** _M_ _−G_ +1 _, · · ·_ _,_ **e** _M_ _}_ . Then by the Riemannian submersion construction of _π_ : _M_ _→Q_ (see
Appx. C), _{_ **e** ˜ _i_ := _π∗_ **e** _i}i_ =1 _,_ 2 _,...,M_ _−G_ is a local orthonormal basis of _Q_ . Let _∇_ _[M]_ and _∇_ _[Q]_ be the
Levi-Civita connection on _M, Q_, respectively, where _∇_ _[Q]_ is induced from the Riemannian metric
inherited from _M_ .


26


Now we calculate the generator of the SDE in Eq. (8’):


  -  _L_ ˜ _t_ = _P_ **b** _t −_ _[σ]_ _t_ [2] **h** ˜ + _[σ]_ _t_ [2]
2 2


_M_

- - _P_ ( **e** _i_ ) [2] _−_ _P_ _∇_ _[M]_ **e** _i_ **[e]** _[i]_ - (18)

_i_ =1


        -        = **b** _[H]_ _t_ _[−]_ _[σ]_ _t_ [2] **h** ˜ + _[σ]_ _t_ [2]
2 2


Its projection under _π∗_ is given by


_M_ _−G_

- - **e** [2] _i_ _[−]_ [(] _[∇]_ **e** _[M]_ _i_ **[e]** _[i]_ [)] _[H]_ [�] _._

_i_ =1


  -  _Lt_ = _π∗_ **b** _t −_ _[σ]_ _t_ [2] + _[σ]_ _t_ [2]
2 **[h]** 2


_M_ _−G_


_i_ =1


**e** ˜ [2] _i_ _[−]_ [(] _[∇]_ **e** _[M]_ ˜ _i_ **[e]** [˜] _[i]_ [)] _[H]_ [�] _,_


which is the generator of Eq. (7). So we have _π_ (˜ **x** _t_ ) = **y** _t_, where **y** _t_ is defined in Eq. (7).


Let Θ be a 1-form on _M_ whose kernel contains the horizontal space _H_ everywhere. From Eq. (18),
**x** ˜ _t_ is the following SDE


d **x** _t_ = **v** [(0)] ( **x** _t, t_ )d _t_ +


_M_

- **v** [(] _[i]_ [)] ( **x** _t, t_ ) _◦_ d **w** _t_ _[i][,]_


_i_ =1


    -     where **v** [(0)] = **b** _[H]_ _t_ _[−]_ _[σ]_ _t_ [2] **h** ˜ _−_ _[σ]_ _t_ [2]
2 2


_M_ _−G_

- ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)] _[H][,]_ **v** [(] _[i]_ [)] = _σt_ **e** _i._

_i_ =1


Then the line integral

          - _t_


_t_ - _t_

Θ **x** _s_ d _s_ =
0 0


_M_

- Θ( **v** [(] _[i]_ [)] )(˜ **x** _s_ ) _◦_ d **w** _s_ _[i]_ [= 0] _[,]_

_i_ =0


0


since **v** [(] _[i]_ [)] _∈H,_ Θ( **v** [(] _[i]_ [)] ) = 0. So we can conclude that **x** ˜ _t_ is the horizontal lift of **y** _t_ .


**Corollary 3’.** **x** ˜1 (defined by Eq. (8)) has the same distribution on _Q_ with **x** 1 (defined by Eq. (6)).
When _σt_ = 0, _∀_ **x** 0 _∈M_, Eq. (8) has shorter trajectory length than Eq. (6):

      - 1       - 1


1  - 1

_⟨P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _, P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _⟩_ _[M]_ d _t_ ⩽
0 0


_⟨_ **b** _t_ ( **x** _t_ ) _,_ **b** _t_ ( **x** _t_ )) _⟩_ _[M]_ d _t._
0


_**Proof**_ _._ By definition of horizontal lift, _π_ (˜ **x** _t_ ) = **y** _t_ = _π_ ( **x** _t_ ) _, ∀t_ _∈_ [0 _,_ 1], then **x** ˜1 (defined by
Eq. (8)) has the same distribution on _Q_ with **x** 1 (defined by Eq. (6)). Since _π_ (˜ **x** _t_ ) = _π_ ( **x** _t_ ), then
**x** _t_ = _gt ·_ ˜ **x** _t, gt_ _∈G_ . Then by the _G_ -equivariant property of **b**, we have

  - 1  - 1


1 - 1

_⟨_ **b** _t_ ( **x** _t_ ) _,_ **b** _t_ ( **x** _t_ )) _⟩_ _[M]_ d _t_ =
0 0


_⟨_ **b** _t_ ( _gt ·_ ˜ **x** _t_ ) _,_ **b** _t_ ( _gt ·_ ˜ **x** _t_ )) _⟩_ _[M]_ d _t_
0


 - 1
= _⟨_ ( _Lgt_ ) _∗_ **x** ˜ _t_ **b** _t_ (˜ **x** _t_ ) _,_ ( _Lgt_ ) _∗_ **x** ˜ _t_ **b** _t_ (˜ **x** _t_ )) _⟩_ _[M]_ d _t_

0


 - 1
= _⟨_ **b** _t_ (˜ **x** _t_ ) _,_ **b** _t_ (˜ **x** _t_ )) _⟩_ _[M]_ d _t_

0


 - 1
=

0

 - 1
⩾


0


�� **b** _t_ (˜ **x** _t_ ) _[H]_ _,_ **b** _t_ (˜ **x** _t_ ) _[H]_ )� _M_ + - **b** _t_ (˜ **x** _t_ ) _[V]_ _,_ **b** _t_ (˜ **x** _t_ ) _[V]_ )� _M_ [�] d _t_


- **b** _t_ (˜ **x** _t_ ) _[H]_ _,_ **b** _t_ (˜ **x** _t_ ) _[H]_ )� _M_ d _t_


             - 1
= _⟨P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _, P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _⟩_ _[M]_ d _t._

0


D.3 PROOF OF THM. 4


For the calculation of the mean curvature vector, we can embed the equivalent class _π_ _[−]_ [1] ( **y** )
into the total space where **y** _∈Q_ . Thus, we can define the embedding Φ **[x]** : _G_ _→M_


27


by Φ **[x]** ( _g_ ) = _g_ _·_ **x** . For **x** _∈_ _π_ _[−]_ [1] ( **y** ) the horizontal lift of mean curvature vector is defined
by **h** [˜] ( **x** ) := ( [�] _[M]_ _i_ = _M_ _−G_ +1 _[∇]_ **[e]** _i_ **[e]** _[i]_ [)] _[H]_ [,] [where] _[{]_ **[e]** _[i][}][M]_ _i_ =1 [is] [a] [local] [orthonormal] [basis] [of] _[T]_ **[x]** _[M]_ [and]
_V_ **x** = span _{_ **e** _M_ _−G_ +1 _, · · ·_ _,_ **e** _M_ _}_ . The mean curvature vector has a nice geometric relation to the
volume of the equivalent class that helps us to calculate it.


**Definition 37.** Let Φ : _G_ _→M_ be an immersion. A smooth variation of Φ is a smooth mapping
_F_ : _P_ _×_ ( _−ϵ, ϵ_ ) _→M_ satisfying:


    - For any _t ∈_ ( _−ϵ, ϵ_ ), Φ _t_ = _F_ ( _·, t_ ) is an immersion;


    - Φ0 = _F_ ( _·,_ 0) = Φ;


**Proposition** **38.** _(First_ _variation_ _of_ _volume_ _(Chavel,_ _1995,_ _Exercise._ _III.14))_ _The_ _mean_ _curvature_
_vector_ **h** [˜] ( **x** ) _satisfies the following formula:_


d

d _t_


       Vol( _G, t_ ) = _−_ _⟨_ **h** [˜] _,_ **v** _⟩_ dVol( _G,_ 0) _,_
���� _t_ =0 _G_


_where_ **v** = _F∗_ ( _∂t_ _[∂]_ [)] _[.]_


In local orthonormal frame _{_ **e** ¯ _i}_ _[G]_ _i_ =1 [of] _[ G]_ [, the volume of] _[ G]_ [is defined by]


     Vol( _G, t_ ) :=

_G_


~~�~~ det( **G** _t_ ) d _g_ [1] _∧· · · ∧_ d _g_ _[G]_ _,_


where **G** _[ij]_ _t_ [=] _[ ⟨]_ [Φ] _[∗][t]_ **[e]** [¯] _[i][,]_ [ Φ] _[∗][t][e]_ [¯] _[j][⟩][M]_ [,][ d] _[g][i]_ [ is the dual form of][ ¯] **[e]** _[i]_ [,] _[ i.e.]_ [d] _[g][i]_ [(¯] **[e]** _[j]_ [) = 1][ if] _[ i]_ [ =] _[ j]_ [, and][ d] _[g][i]_ [(¯] **[e]** _[j]_ [) =]
0 if _i ̸_ = _j_ .


**Theorem** **4’.** Assume **x** _t_ is a diffusion process in the CoM subspace _M_ _⊂_ R [3] _[N]_, given by the
following SDE:


d **x** _t_ = **b** _t_ ( **x** _t_ ) d _t_ + _σt_ d **w** _t,_

where **b** _t_ ( **x** _t_ ) is a SO(3)-equivariant vector field _∀t_ _∈_ [0 _, T_ ], **w** _t_ is the standard Wiener process on
CoM. The horizontal lift of the process _π_ ( **x** _t_ ) is given by the following SDE:


         -         d˜ **x** _t_ = _P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _−_ _[σ]_ 2 _t_ [2] **h** ˜(˜ **x** _t_ ) d _t_ + _σtP_ **x** ˜ _t_ d **w** _t,_ (9’)

where the _P_ **x** is the horizontal projection operator at **x** and **h** [˜] ( **x** ) is the horizontal lift of mean
curvature vector. The explicit expressions of _P_ and **h** [˜] are shown as follows:


     - _N_      _P_ **x** ( **v** ) = **v** _−_ **J** _[−]_ [1] - **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)]


_i_ =1


_×_ **x** _, ∀_ **v** _∈_ _T_ **x** _M_


_N_


**x** [(] _[i]_ [)] **x** [(] _[i]_ [)] _[⊤]_ _∈_ R [3] _[×]_ [3] _._

_i_ =1


**h** ˜ [(] _[i]_ [)] ( **x** ) = _−_ (tr( **J** _[−]_ [1] ) **I** _−_ **J** _[−]_ [1] ) **x** [(] _[i]_ [)] _,_ where **J** =


_N_

- _∥_ **x** [(] _[i]_ [)] _∥_ [2] **I** _−_


_i_ =1


_**Proof**_ _._ Let _{_ **e** 1 _, ...,_ **e** _M_ _}_ be an orthonormal basis for _T_ **x** _M_, which is ordered such that _H_ **x** =
span _{_ **e** 1 _, · · ·_ _,_ **e** _M_ _−G}_, and _V_ **x** = span _{_ **e** _M_ _−G_ +1 _, · · ·_ _,_ **e** _M_ _}_ . Then by the Riemannian submersion
construction of _π_ : _M →Q_ (see Appx. C), _{_ **e** ˜ _i_ := _π∗_ **xe** _i}i_ =1 _,_ 2 _,...,M_ _−G_ is a local orthonormal basis
of _Q_ . Let _∇_ _[M]_ and _∇_ _[Q]_ be the Levi-Civita connection on _M, Q_, respectively, where _∇_ _[Q]_ is induced
from the induced Riemannian structure from _M_ on _Q_ . As shown in Appx. D.2, the horizontal lift
of Eq. (8) has the generator


         -         _Lt_ = **b** _[H]_ _t_ _[−]_ _[σ]_ _t_ [2] **h** ˜ + _[σ]_ _t_ [2]
2 2


By Prop. 34, [�] _i_ _[M]_ =1 _[−][G]_ ( _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [)] _[V]_ [= 0][, then]


_M_ _−G_

- **e** [2] _i_ _[−]_ [(] _[∇]_ **e** _[M]_ _i_ **[e]** _[i]_ [)] _[H][.]_

_i_ =1


  -  _Lt_ = **b** _[H]_ _t_ _[−]_ _[σ]_ _t_ [2] **h** ˜ + _[σ]_ _t_ [2]
2 2


28


_M_ _−G_

- **e** [2] _i_ _[−]_ [(] _[∇]_ **e** _[M]_ _i_ **[e]** _[i]_ [)] _[.]_

_i_ =1


Since _M_ is a Euclidean space, _∇_ _[M]_ **e** _i_ **[e]** _[i]_ [=][ �] _[M]_ _j_ =1 **[e]** _[i]_ [(] **[e]** _i_ _[j]_ [)] _[∂][j]_ [, where] **[ e]** _[j]_ _i_ [is the] _[ j]_ [-th component of] **[ e]** _[i]_ [and]
_∂j_ = _∂/∂xj_ . Since **b** _[H]_ _t_ [(] **[x]** [) =] _[ P]_ **[x][b]** _[t]_ [(] **[x]** [)][, then the generator becomes]


  -  _Lt_ = **b** _[H]_ _t_ [(] **[x]** [)] _[ −]_ _[σ]_ _t_ [2] **h** ˜( **x** ) + _[σ]_ _t_ [2]
2 2


_M_ _−G_

- **e** [2] _i_ _[−]_ [(] _[∇]_ **e** _[M]_ _i_ **[e]** _[i]_ [)]

_i_ =1


_M_

- **e** _[j]_ _i_ **[e]** _i_ _[k][∂][j][∂][k]_

_j,k_ =1


 - = _P_ **xb** _t_ ( **x** ) _−_ _[σ]_ _t_ [2] **h** ˜( **x** ) + _[σ]_ _t_ [2]
2 2


 - = _P_ **xb** _t_ ( **x** ) _−_ _[σ]_ _t_ [2] **h** ˜( **x** ) + _[σ]_ _t_ [2]
2 2


 - = _P_ **xb** _t_ ( **x** ) _−_ _[σ]_ _t_ [2] **h** ˜( **x** ) + _[σ]_ _t_ [2]
2 2


_M_ _−G_


_i_ =1


_M_ _−G_


_i_ =1


_M_

- ( _P_ **x** ) _[jk]_ _∂j∂k,_


_j,k_ =1


_M_

- **e** _[j]_ _i_ [(] _[∂][j]_ **[e]** _i_ _[k]_ [)] _[∂][k]_ [+] **[ e]** _[j]_ _i_ **[e]** _i_ _[k][∂][j][∂][k]_ _[−]_ **[e]** _[j]_ _i_ [(] _[∂][j]_ **[e]** _i_ _[k]_ [)] _[∂][k]_

_j,k_ =1


where we use _P_ **x** = [�] _i_ _[M]_ =1 _[−][G]_ **e** _i_ **e** _[⊤]_ _i_ [is a projection operator.] [Then] _[ L][t]_ [ is the generator of]

         -         d˜ **x** _t_ = _P_ **x** ˜ _t_ ( **b** _t_ (˜ **x** _t_ )) _−_ _[σ]_ 2 _t_ [2] **h** ˜(˜ **x** _t_ ) d _t_ + _σtP_ **x** ˜ _t_ d **w** _t._

For the explicit calculation, recall that in this case, the tangent space _T_ **x** _M_ of _M_ at **x** has the
following decomposition:


    - The vertical tangent space _V_ **x** :

_V_ **x** = _{_ ( **l** _×_ **x** [(1)] _,_ **l** _×_ **x** [(2)] _,_ _· · ·_ _,_ **l** _×_ **x** [(] _[N]_ [)] ) _∈_ R [3] _[N]_ _|_ **l** _∈_ R [3] _}._


    - The horizontal space _H_ **x**, which is the orthogonal complement of the vertical space:


_N_

  -   -   _H_ **x** = **v** = ( **v** [(1)] _, · · ·_ _,_ **v** [(] _[N]_ [)] ) _∈_ R [3] _[N]_ _|_ **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] = 0 _._


_i_ =1


The horizontal projection mapping is defined by _P_ **x** ( **v** ) = **v** _[H]_ = **v** _−_ **v** _[V]_ _, ∀_ **v** _∈_ _T_ **x** _M_, and we can
find an explicit form of it. By definition, [�] _i_ _[N]_ =1 **[x]** [(] _[i]_ [)] _[ ×]_ **[ v]** [(] _[i]_ [)] _[H]_ [=] **[ 0]** [, then]


_N_

- **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] =


_i_ =1


_N_

- **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] _[V]_ _._


_i_ =1


Assume **v** _[V]_ = ( **l** _×_ **x** [(1)] _,_ **l** _×_ **x** [(2)] _,_ _· · ·_ _,_ **l** _×_ **x** [(] _[N]_ [)] ), then


_N_

- **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] =


_i_ =1


=


=


=


_N_

- **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)] _[V]_


_i_ =1


- _N_

 - _∥_ **x** [(] _[i]_ [)] _∥_ [2] **I** _−_


_i_ =1


_N_

- **x** [(] _[i]_ [)] _×_ ( **l** _×_ **x** [(] _[i]_ [)] )


_i_ =1


_N_
�� **x** [(] _[i]_ [)] _,_ **x** [(] _[i]_ [)][�] **l** _−_ - **x** [(] _[i]_ [)] _,_ **l** - **x** [(] _[i]_ [)]


_i_ =1


_N_ 
- **x** [(] _[i]_ [)] **x** [(] _[i]_ [)] _[⊤]_


_i_ =1


**l** _,_


where we use the identity **x** [(] _[i]_ [)] _×_ ( **l** _×_ **x** [(] _[i]_ [)] ) = - **x** [(] _[i]_ [)] _,_ **x** [(] _[i]_ [)][�] **l** _−_ - **x** [(] _[i]_ [)] _,_ **l** - **x** [(] _[i]_ [)] . Denote


_N_

- **x** [(] _[i]_ [)] **x** [(] _[i]_ [)] _[⊤]_ _._


_i_ =1


**J** :=


_N_

- _∥_ **x** [(] _[i]_ [)] _∥_ [2] **I** _−_


_i_ =1


29


And we have **l** = **J** _[−]_ [1] ( [�] _[N]_ _i_ =1 **[x]** [(] _[i]_ [)] _[ ×]_ **[ v]** [(] _[i]_ [)][)][, and]

**v** _[V]_ = ( **l** _×_ **x** [(1)] _,_ **l** _×_ **x** [(2)] _,_ _· · ·_ _,_ **l** _×_ **x** [(] _[N]_ [)] )


  - _N_   = **J** _[−]_ [1] - **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)]


_i_ =1


_×_ **x** _._


Then


      - _N_       _P_ **xv** = **v** _[H]_ = **v** _−_ **J** _[−]_ [1] - **x** [(] _[i]_ [)] _×_ **v** [(] _[i]_ [)]


_i_ =1


_×_ **x** _, ∀_ **v** _∈_ _T_ **x** _M._


For the calculations of the mean curvature vector **h** [˜], we can use Prop. 38. As _G_ = SO(3), its local
_√_
frame (the norm of each vector us 2) is given by the following matrices:


**e** ¯1 =


�0 0 0 0 0 _−_ 1
0 1 0


_,_ **e** ¯2 =


- 0 0 1�
0 0 0

_−_ 1 0 0


_,_ **e** ¯3 =


�0 _−_ 1 0�
1 0 0
0 0 0


_._


Then the Gram matrix **G** is defined by **G** _[ij]_ := _⟨_ **e** ¯ _i_ **x** _,_ ¯ **e** _j_ **x** _⟩_ . By direct calculations, we have **G** = **J** .
Then by Prop. 38,
_√_
**h** ˜( **x** ) = _−∇_ log det **G** _._


Using Jacobi’s formula in matrix calculus, d log det **G** = tr( **J** _[−]_ [1] d **J** ). Then by


_N_


- **x** [(] _[i]_ [)] **x** [(] _[i]_ [)] _[⊤]_ _,_ _∂_ **J**

_i_ =1 _∂_ **x** [(] _j_


_∂_ **J** - 
= 2 **x** [(] _j_ _[i]_ [)] **[I]** _[ −]_ [(] _[δ][j]_ **[x]** [(] _[i]_ [)] _[⊤]_ [+] **[ x]** [(] _[i]_ [)] _[δ]_ _j_ _[⊤]_ [)] _,_
_∂_ **x** [(] _j_ _[i]_ [)]


**J** :=


_N_

- _∥_ **x** [(] _[i]_ [)] _∥_ [2] **I** _−_


_i_ =1


where _δj_ _∈_ R [3] is a one-hot vector at _j_ . Then


tr


_[∂]_ **[J]**
**J** _[−]_ [1]


_∂_ **x** _j_ [(] _[i]_ [)]


= 2 tr( **J** _[−]_ [1] ) **x** [(] _j_ _[i]_ [)] _−_ 2 _δj_ _[⊤]_ **[J]** _[−]_ [1] **[x]** [(] _[i]_ [)] _[.]_


Then we have


**h** ˜ [(] _[i]_ [)] ( **x** ) = _−_ [1]

2 _[∇]_ [log det] **[ G]** [ =] _[ −]_ [(tr(] **[J]** _[−]_ [1][)] **[I]** _[ −]_ **[J]** _[−]_ [1][)] **[x]** [(] _[i]_ [)] _[.]_


E TRAINING AND SAMPLING METHOD IN GENERAL CASE


**Training** **Objective** The diffusion model on the total space _M_ is trained by the denoising score
matching objective. Since the vertical components of the velocity are not strictly needed, we propose to supervise the model only on the horizontal components and allow arbitrary vertical output
of the model. Recall that the horizontal projection operator _P_ **x** projects a vector to its horizontal
component, _i.e._ _P_ **x** ( **v** ) = **v** _[H]_ . Thus the improved training objective is given by

_L_ ( _θ_ ) := E _p_ ( _t_ ) _w_ ( _t_ )E( **x** 0 _,_ **x** 1) _∼p_ joint _,_ **ϵ** _∼N_ (0 _,_ **I** ) _∥P_ **x** _t_ ( **v** _θ_ ( **x** _t, t_ ) _−_ ( _αt_ _[′]_ **[x]** [0] [+] _[ β]_ _t_ _[′]_ **[x]** [1] [+] _[ γ]_ _t_ _[′]_ **[ϵ]** [))] _[ ∥]_ [2] _[.]_


**ODE** **Sampler** After the training stage, _P_ **x** _t_ ( **v** _θ_ ( **x** _t, t_ )) is an approximation of the ground truth
vector field in the horizontal subspace. For the deterministic sampler, we need to simulate the
horizontal lift of the projected ODE, which is given by

d **x** _t_

d _t_ [=] _[ P]_ **[x]** _[t]_ **[v]** [(] **[x]** _[t][, t]_ [)d] _[t.]_

In practice, the ODE process is approximated by numerical solvers, _e.g._ the Euler method and
Runge-Kutta methods.


**SDE Sampler** For the stochastic sampler, we need to simulate the horizontal lift of the projected
original SDE in Eq. (3). According to Thm. 1 and Thm. 4, the lifted process is given by

d **x** _t_ = _P_ **x** _t_ ( **v** _θ_ ( **x** _t, t_ ) + _gt_ **s** _θ_ ( **x** _t, t_ )) d _t_ + _γηt_ **h** ( **x** _t_ )d _t_ + ~~�~~ 2 _γηtP_ **x** _t_ d **w** _t,_


30


where we introduce the hyperparameter _γ_ for protein generation following Geffner et al. (2025).
The training and sampling algorithm is summarized in Algorithm 2 and 3.


**Algorithm 1** Training for _p_ prior = _N_ (0 _,_ **I** )


1: **repeat**
2: ( **x** 0 _,_ **x** 1) _∼_ _p_ joint
3: _t ∼_ _pt_
4: **x** _t_ = _α_ ˆ _t_ **x** 0 + _βt_ **x** 1
5: Take a gradient descent step on

_∇θ_ _w_ ( _t_ ) _∥P_ **x** _t_ ( **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1) _∥_ [2]

6: **until** converged


**Algorithm 3** Sampling


**Algorithm 2** Training for general _p_ prior


1: **repeat**
2: ( **x** 0 _,_ **x** 1) _∼_ _p_ joint, **ϵ** _∼N_ ( **0** _,_ **I** )
3: _t ∼_ _pt_
4: **x** _t_ = _αt_ **x** 0 + _βt_ **x** 1 + _γt_ **ϵ**
5: **v** _t_ = _αt_ _[′]_ **[x]** 0 [+] _[ β]_ _t_ _[′]_ **[x]** 1 [+] _[ γ]_ _t_ _[′]_ **[ϵ]**
6: Take a gradient descent step on

_∇θ_ _w_ ( _t_ ) _∥P_ **x** _t_ ( **v** _θ_ ( **x** _t, t_ ) _−_ **v** _t_ ) _∥_ [2]

7: **until** converged


1: **x** 0 _∼_ _p_ prior
2: **for** _i_ = 0 **to** _K −_ 1 **do**
3: ∆ _ti_ = _ti_ +1 _−_ _ti_
4: **if** ODE sampling **then**
5: **x** _ti_ +1 = **x** _ti_ + _P_ **x** _ti_ **v** _θ_ ( **x** _ti_ _, ti_ )∆ _ti_
6: **end if**
7: **if** SDE sampling **then**
8: **d** _i_ = _P_ **x** _ti_ ( **v** _θ_ ( **x** _ti_ _, ti_ ) + _ηti_ **s** _θ_ ( **x** _ti_ _, ti_ )) + _γgti_ **h** ( **x** _ti_ )
9: **ϵ** _∼N_ ( **0** _,_ **I** )
10: **x** _ti_ +1 = **x** _ti_ + **d** _i_ ∆ _ti_ + ~~�~~ 2 _γηti_ ∆ _tiP_ **x** _ti_ **ϵ**
11: **end if**
12: **end for**


F ADDITIONAL EXPERIMENTAL RESULTS


F.1 EFFICIENCY AND COMPLEXITY ANALYSIS


**Complexity analysis.** In this subsection, we give a detailed discussion on the computational cost
of our method. As mentioned in Thm. 4, we need to compute the inversion of the matrix _I_ and the
cross product for the horizontal projection operator _P_ **x** and the mean curvature vector **h** [˜] ( **x** ). For
the calculation of _I_ _[−]_ [1], notice that _I_ is always a 3 _×_ 3 matrix, so construction cost of _I_ _[−]_ [1] is only
linear _O_ ( _N_ ), where _N_ is the number of atoms (linear _O_ ( _N_ ) cost for constructing _I_, and constant
_O_ (1) cost for inversion). The cross product is conducted atom-wise, so its computational cost is also
linear _O_ ( _N_ ). So we can conclude that the overall computational complexity is _O_ ( _N_ ) for both _P_ **x**
and **h** [˜] ( **x** ).


We would like to mention that the alignment operation adopted in the heuristic alignment-based
diffusion strategies also has the same complexity. To see this, for aligning **x** _∈_ R [3] _[×][N]_ towards **y** _∈_
1
R [3] _[×][N]_, the Kabsch-Umeyama algorithm constructs the optimal rotation matrix as ( **H** _[⊤]_ **H** ) 2 **H** _[−]_ [1],
where **H** := **yx** _[⊤]_ _∈_ R [3] _[×]_ [3] requires a linear _O_ ( _N_ ) cost. In practice, the _O_ ( _N_ ) computational
cost is negligible compared to the cost of gradient back-propagation through the neural network. A
comparison of practical training times is shown in the following table.


Methods Original GeoDiff Af3 align- Quotientdiffusion alignment ment space
diffusion
training speed (iters/s) 4.19 4.07 4.08 4.10


All the results are tested on a single Nvidia A100 GPU. From the results, we can see that the additional computational cost brought by the alignment and projection is negligible.


31


Figure 4: Training loss vs. training epochs. We find that our training is stable in practice.


**Numerical** **stability.** In our quotient-space diffusion model framework, we need to calculate the
matrix inversion of **J**, which may have numerical issues for near-collinear systems of points. In
practice, we add an _ϵ_ **I** term before conducting matrix inversion, that is, we calculate ( _ϵ_ **I** + _I_ ) _[−]_ [1]
in practice, where **I** is the 3 _×_ 3 identity matrix. This treatment is widely adopted in algorithms
facing similar situations, _e.g._, the practical implementation of the Kabsch-Umeyama algorithm for
alignment. Our typical choice of _ϵ_ is 1e-8, and we found that the training process is stable under this
setting. We have shown the training curve of the model on the protein backbone generation task in
Fig. 4, which indicates no numerical issues arise during the training process.


F.2 THE IMPLEMENTATION OF _G_ -EQUIVARIANT VECTOR FIELD


In Thm. 4, we require that the vector field is SO(3)-equivariant. In practice, this can be implemented
by using a SO(3)-equivariant network architecture or applying data augmentation. In this subsection, we justify that both of these choices are valid, such that the diffusion model can generate a
SO(3)-invariant distribution.


**Diffusion model with data augmentation.** The optimal solution of the Euclidean diffusion model
is given by **D** _[∗]_ _θ_ [(] **[x]** _[t]_ [)] [=] [E][[] **[x]** [1] _[|]_ **[x]** _[t]_ []][ (][Song et al.][,][ 2021][;][ Karras et al.][,][ 2022][).] [When the data distribution]
is augmented by random rotation, the data distribution becomes SO(3)-invariant. Thus, the optimal diffusion model can recover the SO(3)-invariant data distribution. When the transition density _p_ ( **x** _t|_ **x** 1) is SO(3)-equivariant, _i.e._ _p_ ( **x** _t|_ **x** 1) = _p_ ( _g_ _·_ **x** _t|g_ _·_ **x** 1) _, ∀g_ _∈_ SO(3), the optimal
network is SO(3)-equivariant. To see this, let _g_ _∈_ SO(3) be an arbitrary rotation matrix. Since
**D** _[∗]_ _θ_ [(] _[g][ ·]_ **[ x]** _[t]_ [) =][ E][[] **[x]** [1] _[|][g][ ·]_ **[ x]** _[t]_ []][, by the Bayes formula,]

E[ **x** 1 _|g ·_ **x** _t_ ] = [E] _[p]_ [target][(] **[x]** [1][)][[] **[x]** [1] _[p]_ [(] _[g][ ·]_ **[ x]** _[t][|]_ **[x]** [1][)]]

E _p_ target( **x** 1)[ _p_ ( _g ·_ **x** _t|_ **x** 1)]

= [E] _[p]_ [target][(] **[x]** [1][)][[] **[x]** [1] _[p]_ [(] **[x]** _[t][|][g][−]_ [1] _[·]_ **[ x]** [1][)]]

E _p_ target( **x** 1)[ _p_ ( **x** _t|g_ _[−]_ [1] **x** 1)]

_[·]_ **[ x]** [1][)]]
= _[g][ ·]_ [ E] _[p]_ [target][(] _[g][−]_ [1] **[x]** [1][)][[] _[g][−]_ [1] **[x]** [1] _[p]_ [(] **[x]** _[t][|][g][−]_ [1]

E _p_ target( _g−_ 1 **x** 1)[ _p_ ( **x** _t|g_ _[−]_ [1] **x** 1)]

= _g ·_ E[ **x** 1 _|_ **x** _t_ ] _,_


where we use the equivariance property of the transition density to get the second equality and
the invariance property of _p_ target to get the third equality. Thus, we can conclude that the optimal
solution under these conditions is SO(3)-equivariant. Geffner et al. (2025) also gives an empirical
validation that a well-trained neural network becomes nearly equivariant even if its architecture is
not equivariant.


**Equivariant architecture.** When the model is required to be SO(3)-equivariant, the optimal solution of the diffusion model is not E[ **x** 1 _|_ **x** _t_ ]. To figure out the optimal solution, we consider the
training loss at time _t_ . The loss function at _t_ is given by

_Lt_ ( _θ_ ) = E _∥_ **D** _θ_ ( **x** _t, t_ ) _−_ **x** 1 _∥_ [2]


 = d [3] _[N]_ **x** 1


d [3] _[N]_ **x** _t_ _p_ ( **x** 1 _,_ **x** _t_ )  - _∥_ **D** _θ_ ( **x** _t, t_ ) _∥_ [2] + _∥_ **x** 1 _∥_ [2] _−_ 2 _⟨_ **D** _θ_ ( **x** _t, t_ ) _,_ **x** 1 _⟩_  - _._


32


The optimal solution satisfies

**D** _[∗]_ _θ_ [(] **[x]** _[t][, t]_ [) =] argmin _Lt_ ( _θ_ ) _._
**D** _θ_ is SO(3) equivariant


The training loss can be simplified using the equivariant constraint:


    _Lt_ ( _θ_ ) = d [3] _[N]_ **x** 1


d [3] _[N]_ **x** _t p_ ( **x** 1 _,_ **x** _t_ )  - _∥_ **D** _θ_ ( **x** _t_ ) _∥_ [2] + _∥_ **x** 1 _∥_ [2] _−_ 2 _⟨_ **D** _θ_ ( **x** _t_ ) _,_ **x** 1 _⟩_  


 = d **r** _t_

R [3] _[N]_ _/_ SO(3)


- 
d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g ·_ **r** _t_ )   - _∥_ **D** _θ_ ( _g ·_ **r** _t_ ) _∥_ [2] + _∥_ **x** 1 _∥_ [2] _−_ 2 _⟨_ **D** _θ_ ( _g ·_ **r** _t_ ) _,_ **x** 1 _⟩_   - _._
SO(3)


Since **D** _θ_ is SO(3)-equivariant, **D** _θ_ ( _g ·_ **r** _t_ ) = _g ·_ **D** _θ_ ( _rt_ ), then we have


    _Lt_ ( _θ_ ) = d **r** _t_

R [3] _[N]_ _/_ SO(3)


- 
d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g ·_ **r** _t_ )   - _∥_ **D** _θ_ ( **r** _t_ ) _∥_ [2] + _∥_ **x** 1 _∥_ [2] _−_ 2 _⟨g ·_ **D** 1 _θ_ ( **r** _t_ ) _,_ **x** 1 _⟩_   - _._
SO(3)


Define _p_ ( **r** _t_ ) = 


SO(3) [d] _[g]_ - d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g ·_ **r** _t_ ), and _p_ ( **x** 1 _, g_ _|_ **r** _t_ ) = _[p]_ [(] **[x]** _p_ [1] ( _[,g]_ **r** _t_ _[·]_ ) **[r]** _[t]_ [)] . Then we have


           _p_ ( **r** _t_ ) _∥_ **D** _θ_ ( **r** _t_ ) _∥_ [2] _−_ 2 _⟨_ **D** _θ_ ( **r** _t_ ) _,_


   d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g ·_ **r** _t_ ) _g_ _[−]_ [1] _·_ **x** 1 _⟩_
SO(3)


    _Lt_ ( _θ_ ) = d **r** _t_

R [3] _[N]_ _/_ SO(3)


   + d **r** _t_

R [3] _[N]_ _/_ SO(3)


So we can conclude that


- 
d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g_ **r** _t_ ) _∥_ **x** 1 _∥_ [2] _._
SO(3)


     -     **D** _[∗]_ _θ_ [(] **[r]** _[t][, t]_ [) =] d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g_ _|_ **r** _t_ ) _g_ _[−]_ [1] _·_ **x** 1 _,_

SO(3)


     -      **D** _[∗]_ _θ_ [(] _[g][′][ ·]_ **[ r]** _[t]_ [) =] d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g_ _|_ **r** _t_ ) _g_ _[′]_ _· g_ _[−]_ [1] _·_ **x** 1 _, ∀g_ _∈_ SO(3) _._

SO(3)


Notice that


    -     **D** _[∗]_ _θ_ [(] **[r]** _[t]_ [) =] d _g_ d [3] _[N]_ **x** 1 _p_ ( **x** 1 _, g_ _|_ **r** _t_ ) _g_ _[−]_ [1] _·_ **x** 1

SO(3)


SO(3) [d] _[g]_ - d [3] _[N]_ **x** 1 _p_ ( _g ·_ **x** 1) _p_ ( _g ·_ **r** _t_ _| g ·_ **x** 1) **x** 1
�SO(3) [d] _[g]_ - d [3] _[N]_ **x** 1 _p_ ( _g ·_ **x** 1) _p_ ( _g ·_ **r** _t_ _| g ·_ **x** 1)


SO(3) [d] _[g]_ - d [3] _[N]_ **x** 1 _p_ ( _g ·_ **x** 1) _p_ ( **r** _t_ _|_ **x** 1) **x** 1
�SO(3) [d] _[g]_ - d [3] _[N]_ **x** 1 _p_ ( _g ·_ **x** 1) _p_ ( **r** _t_ _|_ **x** 1) _[,]_


=


=


which is equivalent to the case _p_ data = �SO(3) [d] _[g]_ _[p]_ [(] _[g][ ·]_ **[ x]** [1][)][,] _[ i.e.]_ [using the augmentation by random]

SO(3) rotation.


F.3 TRAINING AND SAMPLING ACCELERATION


In this subsection, we study the training and sampling convergence speed of different methods. For
the training convergence speed comparison, we plot the generation performance measured by the
precision AMR median metric with respect to the training epochs for previous heuristic alignment
methods and our quotient-space diffusion model in Fig. 5(Left). We only focus on the first 100
epochs for all the methods. These models are trained with the same architecture ET-Flow (SO(3))
and training configurations on the GEOM-DRUGS dataset. The results indicate that our method
achieves a similar convergence speed to the AF3 heuristic method, because both methods reduce the
learning difficulty of the model, as shown in Table 1. This theoretical benefit leads to faster convergence than the GeoDiff alignment method. We also notice that the AF3 alignment method starts to
get worse generation performance after 80 training epochs. This happens due to the incompatibility
between the training loss and the generation performance metric, as the AF3 method is originally
designed for the protein structure prediction task, which is not evaluated by distributional metrics.


For the sampling convergence speed comparison, we plot the generation performance measured by
the precision AMR median metric with respect to the number of function evaluations (NFE) for the
sampling process in Fig. 5(Right). For all these methods trained on the GEOM-DRUGS dataset,
we use the Flow Matching ODE sampler (Lipman et al., 2023) with Euler discretization. From


33


Figure 5: Training and sampling convergence speed comparison on GEOM-DRUGS. **(Left)** The
relationship between training epochs and generation performance measured by the precision AMR
median metric. **(Right)** The relationship between the number of function evaluations (NFE) for
sampling and generation performance measured by the precision AMR median metric.


the results, we can observe that models trained with different strategies exhibit similar convergence
trends (performance gradually degrades as NFE decreases), our quotient-space diffusion framework
consistently outperforms all baselines across every NFE setting.


F.4 QUOTIENT SPACE BEYOND R [3] _[N]_ _/_ SE(3)


Our framework can generalize to quotient spaces generated by symmetry groups beyond the special
Euclidean group SE(3). Possible examples include the U(1) symmetry in quantum wavefunctions,
the SU(2) symmetry in particle physics, and the SO(3) symmetry in higher ( _>_ 3) representation
spaces for tasks including the mean-field electron Hamiltonian matrix prediction. In this work, we
focus on the SE(3) case for its significant relevance to scientific research (Abramson et al., 2024).
Applications of our framework on the mentioned more diverse systems above are left as future work.


G EXPERIMENTS


G.1 MOLECULAR STRUCTURE GENERATION


This appendix summarizes our experimental setup, which strictly follows that of ET-Flow (Hassan
et al., 2024). We detail the datasets, model architecture, training, sampling, and evaluation. For a
more comprehensive discussion of each component, we refer the reader to the appendices of their
original paper.


**Dataset.** First, we evaluate our framework on the molecule structure generation task. In this scenario, our goal is to generate the 3D coordinates of a molecule given the graph structure of the
molecule. We conduct the experiments on the GEOM datasets (Axelrod & Gomez-Bombarelli,
2022), which provide structure ensembles generated by metadynamics in CREST (Pracht et al.,
2024), and we focus on the GEOM-QM9 and GEOM-DRUGS datasets. Following the data processing and splits from (Hassan et al., 2024), we use the random splits with train/validation/test
of 243473/30433/1000 for GEOM-DRUGS and 106586/13323/1000 for GEOM-QM9. In addition,
data with disconnected molecule graphs are removed for GEOM-DRUGS (Hassan et al., 2024). Our
reproduction is based on the modified data-processing pipeline following the released configs thus
different from the results reported in the original paper.


**Settings.** We primarily follow the setting in (Hassan et al., 2024). We set the Gaussian distribution
as the prior distribution on GEOM-QM9 and use the harmonic prior for GEOM-DRUGS (Volk
et al., 2023). Following (Jing et al., 2022; Xu et al., 2022), we report the RMSD-based metrics,
_e.g._ Coverage and Average Minimum RMSD (AMR) between generated and ground truth structure
ensembles. We parameterize **v** _θ_ by using equivariant graph transformer architectures from ETFlow (Hassan et al., 2024), including the O(3) and SO(3) equivariant variants, which also serves as a
verification that our framework is compatible with different backbone models. For training, we use


34


AdamW as the optimizer, and set the hyper-parameter _ϵ_ to 1e-8 and ( _β_ 1 _, β_ 2) to (0.9,0.999). We use
the dynamic gradient clipping as (Hassan et al., 2024; Hoogeboom et al., 2022b). The peak learning
rate is set to 5e-4 for GEOM-DRUGS and 7e-4 for GEOM-QM9. The batch size is set to 48 for
GEOM-DRUGS and 128 for GEOM-QM9. The weight decay is set to 1e-8. The model is trained
for 1000 epochs for both datasets. The noise scale _σ_ is set to 0 _._ 1. We also use 50 time steps with the
Euler solver for sampling. All models are trained on 8 NVIDIA A100 GPUs.


**Baselines.** Following (Hassan et al., 2024), we choose strong baselines trained on GEOM-DRUGS
and GEOM-QM9 for a challenging comparison. We report the performance of GeoMol (Ganea
et al., 2021), GeoDiff (Xu et al., 2022), Torsional Diffusion (Jing et al., 2022), and MCF (Wang
et al., 2023).


G.2 PROTEIN


This appendix summarizes our experimental setup, which strictly follows that of Prote´ına (Geffner
et al., 2025). We detail the datasets, model architecture, training, sampling, and evaluation. For a
more comprehensive discussion of each component, we refer the reader to the appendices of their
original paper.


G.2.1 DATASET


For training, we utilize the Foldseek AFDB clusters ( _D_ FS) dataset as curated and described in
the Prote´ına. This dataset is a high-quality, non-redundant subset of the AlphaFold Database
(AFDB), containing 588,318 cluster-representative protein structures with lengths between 32 and
256 residues. The dataset is annotated with hierarchical CATH labels, which are leveraged during training. Our data processing and handling strictly follow the pipeline detailed in Appendix M
of (Geffner et al., 2025).


G.2.2 MODEL ARCHITECTURE AND TRAINING


Our model architecture is the same as the efficient, non-equivariant transformer proposed
by (Geffner et al., 2025). Specifically, we adopt the variant that forgoes the use of computationally
expensive triangle update layers. The model is trained using the conditional flow matching (CFM)
objective. Key aspects of the training protocol from Prote´ına are preserved, including their novel
Beta-Uniform mixture for the time-sampling distribution _p_ ( _t_ ), the use of self-conditioning, and data
augmentation with random rotations. All model and training hyperparameters, such as embedding
dimensions, number of layers, attention heads, and optimizer settings, are kept consistent with hyperparameters saved in their released checkpoint _M_ [small] FS [. The hyperparameters for the] _[ M]_ FS [small] model
are detailed in Table 4, in comparison with the larger models from the original Prote´ına paper.


G.2.3 SAMPLING


To facilitate a direct comparison with the publicly available Prote´ına checkpoints, we trained our
model with an identical hierarchical fold class conditioning mechanism. However, to ensure a fair
assessment of foundational generative capabilities, all experiments reported in our main text were
performed in a strictly unconditional setting. We applied the same sampling protocol across all
models, using 400 sampling steps and enabling self-conditioning, which consistently improved performance. No other guidance techniques, such as autoguidance, were utilized. We use deterministic ODE sampling to assess distributional fidelity and SDE sampling to explore the designabilitydiversity trade-off. We adapt the SDE formulation and its Euler-Maruyama numerical scheme, detailed in Appendix I of (Geffner et al., 2025), for our quotient space framework, while retaining all
other configurations, such as the sampling scheduler and _g_ ( _t_ ), from the original paper.


G.2.4 EVALUATION


We evaluate our models rigorously adheres to the metrics established and validated in the Prote´ına
paper. We assess model performance using the standard suite of metrics in protein design:


    - **Designability.** Quantified by the self-consistency RMSD (scRMSD) protocol, using ProteinMPNN for inverse folding and ESMFold for structure prediction, with a success threshold of scRMSD less than 2 A. [˚]


35


Table 4: Hyperparameters for Prote´ına model.


**Hyperparameter** _M_ FS _M_ [no-tri] FS _M_ [small] FS

**Prote´ına Architecture**
sequence repr dim 768 768 512
# registers 10 10 10
sequence cond dim 512 512 128
_t_ sinusoidal enc dim 256 256 196
idx. sinusoidal enc dim 128 128 196
fold emb dim 256 256 196
pair repr dim 512 512 196
seq separation dim 128 128 128
pair distances dim ( _xt_ ) 64 64 64
pair distances dim ( _x_ ˜( _xt_ )) 128 128 128
pair distances min ( A) [˚] 1 1 1
pair distances max ( A) [˚] 30 30 30
# attention heads 12 12 12
# tranformer layers 15 15 12
# triangle layers 5            -            # trainable parameters 200M 200M 60M


**Prote´ına Training**
# steps 200K 360K 150K
batch size per GPU 4 10 5
# GPUs 128 96 16
# grad. acc. steps 1 1 1


    - **Diversity.** Measured in two ways: by the average pairwise TM-score among designable
samples, and by the number of distinct structural clusters identified by Foldseek at a TMscore threshold of 0.5.

    - **Novelty.** Assessed by calculating the maximum TM-score of each designable sample
against reference structures in the PDB and AFDB databases.


We also adopt the novel probabilistic metrics introduced by (Geffner et al., 2025), to measure how
well our model captures the true distribution of protein structures:


    - **FPSD.** Measured the distributional similarity between generated and reference structures
in the feature space of a pre-trained fold class predictor.


    - **fS.** Evaluated both the quality and diversity of samples based on the confidence and entropy
of fold class predictions.


    - **fJSD.** Quantified the similarity between the categorical fold class distributions of generated
and reference sets.


It is noteworthy that we have omitted the Diversity and Novelty metrics from our main text to avoid
comparisons with potentially inaccurate results in the literature. This decision is based on a bug
recently identified in the alntmscore output of FoldSeek versions prior to v10 (release 10-941cd33),
which renders many previously reported TM-based metrics incorrect (also found in (Daras et al.,
2025)). To provide a controlled and accurate benchmark, we conducted our own analysis using
the FoldSeek v10 (release 10-941cd33). We limited this re-evaluation to the released small Prote´ına
model and our corresponding model trained in the quotient space. The full results of this comparison
are summarized in Table 5.


36


Table 5: Complete performance comparison of the released Prote´ına checkpoints against our version
in the quotient space. Best results are marked in **bold** .


Diversity Novelty vs. FPSD vs. fS fJSD vs.
Model Designability (%)

Cluster _↑_ TM-Sc. _↓_ PDB _↓_ AFDB _↓_ PDB _↓_ AFDB _↓_ (C/A/T) _↑_ PDB _↓_ AFDB _↓_


**SDE Sampling**
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [35] 96.0 0.44 (209) 0.50 0.86 0.91 386.5 378.2 1.77/4.97/17.78 2.17 1.73
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [35][ + ours] **97.6** 0.40 (197) 0.48 0.86 0.91 274.7 277.1 2.24/6.69/20.99 1.68 1.55
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [45] 92.2 0.55 (253) 0.49 0.84 0.90 332.9 320.4 1.83/5.01/20.22 1.93 1.49
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [45][ + ours] 92.6 0.51 (253) 0.47 0.85 0.90 244.5 246.3 2.24/6.68/23.47 1.43 1.28
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [50] 89.2 0.57 (255) 0.48 0.83 0.89 306.2 290.8 1.86/4.92/21.15 1.81 1.36
_M_ [small] FS _[, γ]_ [= 0] _[.]_ [50][ + ours] 90.2 0.51 (231) 0.47 0.84 0.90 228.0 228.7 2.25/6.59/25.24 1.32 1.17


**ODE Sampling**
_M_ [small] FS 13.8 0.90 (62) 0.43 0.80 0.87 83.18 21.93 2.45/5.63/31.76 0.58 0.12
_M_ [small] FS + ours 15.6 0.87 (68) 0.43 0.80 0.86 **69.94** **17.56** **2.57/6.40/32.14** **0.41** 0.11


37