# ADVERSARIAL ATTACK ON TENSOR RING DECOMPO
## SITION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Tensor ring (TR) decomposition, a powerful tool for handling high-dimensional
data, has been widely applied in various fields such as computer vision and recommender systems. However, the vulnerability of TR decomposition to adversarial perturbations has not been systematically studied, and it remains unclear
how adversarial perturbations affect its low-rank approximation performance. To
tackle this problem, we introduce a novel adversarial attack approach on tensor
ring decomposition (AdaTR), formulated as an asymmetric max–min objective.
Specifically, we aim to find the optimal perturbation that maximizes the reconstruction error of the low-TR-rank approximation. Furthermore, to alleviate the
memory and computational overhead caused by iterative dependency during attacks, we propose a novel faster approximate gradient attack model (FAG-AdaTR)
that avoids step-by-step perturbation tensor tracking while maintaining high attack
effectiveness. Subsequently, we develop a gradient descent algorithm with numerical convergence guarantees. Numerical experiments on tensor decomposition,
completion, and recommender systems using color images and videos validate the
attack effectiveness of the proposed methods.


1 INTRODUCTION


Tensor decompositions aim to decompose the higher-order tensor to a set of low dimensional factors, which have attracted significant attention in various fields, including machine learning (Kolda
& Bader, 2009), quantum physics (Sidiropoulos et al., 2017), signal processing (Sch¨utt et al., 2020),
brain science (Kang et al., 2013), and chemometrics (Acar et al., 2011). Different from matrix decomposition, there is no unique definition for the corresponding tensor decomposition. The CANDECOMP/PARAFAC (CP) decomposition (Hitchcock, 1927) can be regarded as a special case of
the Tucker (Tucker, 1966) decomposition, where the core factor has nonzero entries only on the
super-diagonal. However, Tucker decomposition suffers from restrictive bounds on its Tucker ranks,
limiting its ability to capture rich structural information in high-order tensors. To address this issue,
tensor train (TT) (Oseledets, 2011) and tensor ring (TR) (Zhao et al., 2016) decompositions have
been proposed and have shown strong performance on tensor decomposition tasks. Specifically, TT
decomposition represents an _N_ th-order tensor using ( _N_ _−_ 2) third-order tensors and two matrices,
while TR decomposition factorizes it into _N_ third-order tensors. TT can further be viewed as a
special case of TR, where the border tensor ranks are constrained to one.


Recently, Goodfellow et al. (2014) demonstrated that machine learning methods are vulnerable to
adversarial attacks and has been widely verified in various fields (Ebrahimi et al., 2017; Zou et al.,
2023; Wang et al., 2025). Motivated by this, adversarial training for the nonnegative matrix factorization (ANMF) model has been investigated to improve the robustness and predictive performance
of NMF (Luo et al., 2020). However, their formulation does not make it easy to choose the instancespecific target. Therefore, Cai et al. (2021) proposed the novel adversarially-trained NMF (ATNMF)
to tackle this problem, which can be written as follows:

min F
**W** _,_ **H** _≥_ 0 _[∥]_ **[X]** [ +] **[ E]** _[ −]_ **[WH]** _[∥]_ [2]

(1)
s.t. **E** = arg max _∥_ **X** + **E** _−_ **WH** _∥_ [2] F _[,]_ **[ X]** [ +] **[ E]** _[ ≥]_ [0] _[,][ ∥]_ **[E]** _[∥]_ [2] F _[< ϵ,]_
**E**

where the **X** _∈_ R _[I][×][J]_ denotes the original data matrix, and **W** _∈_ R _[I][×][R]_ _,_ **H** _∈_ R _[R][×][J]_ denote the nonnegative factors, and **E** _∈_ R _[I][×][J]_ denotes the perturbation matrix. The _ϵ_ denotes the energy budget


1


|Col1|FAG-A<br>ATTR|daTR|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||TRD|mean|mean||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


|Col1|FAG-A<br>ATTR|daTR|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||TRD|mean|mean||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


Figure 1: Average reconstruction error _∥X −_ _X∥_ [ˆ] [2] F [under different perturbation budget] _[ ϵ]_ [ for (a) Rank]
= 5 and (b) Rank = 10. Results are averaged over eight images. The proposed methods (AdaTR and
FAG-AdaTR) are compared with ATTR and the TR decomposition baseline.


of the perturbation matrix **E**, and _∥· ∥_ F denotes the Frobenius norm. This approach is particularly
beneficial across various real-world applications, include matrix completion, link prediction, recommender systems (Seyedi et al., 2023; Mahmoodi et al., 2023; 2024; Zhang et al., 2024). When
a limited budget _ϵ_ is given, the approach Eq. (1) can usually improve the predictive performance
rather than attacking matrix factorization (Zhang et al., 2024).


Compared to employing adversarial training to improve the robustness of matrix factorization, verifying whether matrix factorization itself is inherently vulnerable is more challenging. Related studies include maliciously injecting outlier samples into the matrix (i.e., **X** adv = [ **X** _,_ **z** ]) to cause the
attacked subspace to deviate from the original data subspace (Pimentel-Alarc´on et al., 2017; Li et al.,
2021), adding adversarial perturbations that lead to subspace deviation (Li et al., 2020), or breaking
the uniqueness of NMF (Vu et al., 2024). These studies have mainly focused on subspace deviation
or on compromising the uniqueness of the factorization. However, the vulnerability of the most
general matrix or tensor decomposition remains an open question. This motivates us to ask the
following question:


- **RQ1:** Are matrix/tensor factorization vulnerable to adversarial attacks?


- **RQ2:** How can we design the adversarial attack approach for matrix/tensor decompositions?


To address these questions, we first extend the concept of ATNMF to TR decomposition, yielding
an ATTR baseline approach. Here, vulnerability refers to which a perturbation added to the input tensor, within a given budget, can increase the resulting reconstruction error. As illustrated in
Fig.1, ATTR exhibits behavior consistent with ATNMF: under small perturbations, ATTR slightly
improves predictive performance. However, when evaluated under the same perturbation budget,
our proposed methods (AdaTR and FAG-AdaTR) lead to substantially larger low-rank approximation errors. This provides clear evidence that TR decomposition is truly vulnerable to adversarial
perturbations, thereby answering **RQ1** . All ALS-based tensor decompositions—whether for reconstruction, completion, or recommendation—are entirely driven by the observed input tensor, even
small perturbations injected at the input propagate through all update steps and get amplified by the
low-rank structure, ultimately causing large reconstruction errors (and consequently large prediction
errors in recommendation).


Having established the vulnerability of TR decomposition, we next turn to **RQ2** . To this end, we
propose a novel asymmetric adversarial attack approach for TR decomposition, termed AdaTR. In
particular, we define the low-rank approximation error as the attack objective and model the perturbation as a learnable adversarial perturbation tensor. Different from the traditional adversarial
training approach as in Eq. (1), the proposed AdaTR adopts an asymmetric max-min objective. This
design enables the attacker to directly maximize the low-rank approximation error in TR decomposition. Furthermore, to mitigate the high computational overhead of long iterative dependencies, we
introduce FAG-AdaTR, a faster attack algorithm with an approximate gradient. The key contributions of this work are summarized as follows:


2


Epsilon


(b) TR-Ranks = [10,10,10]


17


16


15


14


13


Epsilon


(a) TR-Ranks = [5,5,5]


10


9


8


7


6


5


4


3


- We elaborately design an asymmetric adversarial attack approach on TR decomposition
(AdaTR). This approach provides the first evidence that tensor decomposition models are
susceptible to adversarial attacks.

    - AdaTR requires backtracking TR iterative updates, demanding substantial peak memory.
To alleviate this problem, we propose a faster algorithm with approximate gradient on TR
decomposition (FAG-AdaTR).

    - Extensive experiments show that our proposed attacks substantially degrade performance in
tensor decomposition, completion, and recommendation tasks, and are capable of causing
significant errors even with tiny perturbations.


2 NOTATIONS


_where X_ _∈_ R _[I]_ [1] _[×][I]_ [2] _[×···×][I][N]_ _denotes an N_ _th-order tensor,_ _and Gn_ _∈_ R _[R][n][×][I][n][×][R][n]_ [+1] _are third-order_
_factors._ _Symbolically, we employ X_ = _TR_ ([ _G_ ]) = _TR_ ( _G_ 1 _, G_ 2 _, · · ·_ _, GN_ ) _to denote TR decomposition._
**Definition** **4** (Tensor Mode- _k_ Unfolding (Zhao et al., 2016)) **.** _Given_ _an_ _N_ _th-order_ _tensor_ _X_ _∈_
R _[I]_ [1] _[×][I]_ [2] _[×···][I][N]_ _, the tensor mode-k unfolding of X_ _is given as follows:_


**X** [ _k_ ]( _ik, ik_ +1 _ik_ +2 _· · · iN_ _i_ 1 _i_ 2 _· · · ik−_ 1) = _X_ ( _i_ 1 _, i_ 2 _, · · ·_ _, iN_ ) _,_ (3)


_and the classical mode-k unfolding of X_ _is given as follows:_


**X** ( _k_ )( _ik, i_ 1 _i_ 2 _· · · ik−_ 1 _ik_ +1 _ik_ +2 _· · · iN_ ) = _X_ ( _i_ 1 _, i_ 2 _, · · ·_ _, iN_ ) _,_ (4)


_where_ **X** [ _k_ ] _and_ **X** ( _k_ ) _are the size of Ik ×_ [�] _j_ = _k_ _[I][j]_ _[matrices.]_


4 ADVERSARIAL ATTACK ON TENSOR RING DECOMPOSITION


4.1 WHY WE NEED AN ASYMMETRIC ADVERSARIAL ATTACK FRAMEWORK ON TENSOR
DECOMPOSITION


The ATNMF algorithm can be naturally extended to tensor decomposition. As an example, we consider the tensor ring (TR) decomposition, which allows us to use this adversarial training approach


3


In this paper, the scalars are denoted by standard lowercase or uppercase letters (e.g., _x_, _X_ ), vectors by bold lowercase letters (e.g., **x** ), and matrices by bold uppercase letters (e.g., **X** ). Higherorder tensors with order _N_ _≥_ 3 are denoted by calligraphic letters, e.g., _X_ _∈_ R _[I]_ [1] _[×][I]_ [2] _[×···×][I][N]_ .
The ( _i_ 1 _, i_ 2 _, ·, iN_ )-th element of _X_ is denoted by _X_ ( _i_ 1 _, i_ 2 _, . . ., iN_ ) or equivalently _xi_ 1 _,i_ 2 _,...,iN_ . The


~~�~~
Frobenius norm of a tensor is defined as _∥X∥_ F = 


_i_ 1 _,i_ 2 _,···,iN_ _[X]_ [(] _[i]_ [1] _[, i]_ [2] _[,][ · · ·]_ _[, i][N]_ [)][. The set notation]


is denoted by [ _G_ ] := _{G_ 1 _, G_ 2 _, · · ·_ _, GN_ _}_ .


3 PRELIMINARIES


**Definition 1** (Tensor Composition) **.** _We call the process of generating the N_ _-th order tensor X_ _from_
_the factors {G_ 1 _, G_ 2 _, · · ·_ _, GN_ _} in special tensor network contraction as the tensor composition, which_
_can be written as X_ = _TN_ ([ _G_ ]) _._ _Furthermore, we can also write the tensor composition except the_
_factor Gk_ _as TN_ ( _{G_ 1 _, · · ·_ _, Gk−_ 1 _, Gk_ +1 _, · · ·_ _, GN_ _}_ ) _or TN_ ([ _G_ ] _, /Gk_ ) _._
**Definition 2** (Tensor Decomposition) **.** _We call the process of learning the N_ _factors G_ _of the N_ _-th_
_order_ _tensor_ _X_ _in_ _a_ _specific_ _tensor_ _network_ _method_ _as_ _tensor_ _decomposition._ _The_ _decomposition_
_operator of the tensor X_ _can be written as X_ _≈_ _TN_ ([ _G_ ]) _._

**Definition 3** (Tensor Ring Decomposition (Zhao et al., 2016)) **.** _The TR decomposition representa-_
_tion is given as follows,_


_X_ ( _i_ 1 _, i_ 2 _, · · ·_ _, iN_ ) =


_R_ 1 _,···,RN_

 - _G_ 1( _r_ 1 _, i_ 1 _, r_ 2) _G_ 2( _r_ 2 _, i_ 2 _, r_ 3) _· · · GN_ ( _rN_ _, iN_ _, r_ 1) _,_ (2)

_r_ 1 _,···,rN_


Figure 2: Illustration of TR-ALS algorithm. If the adversarial perturbation tensor _E_ is added to the
input _X_, it propagates through each unfolding and update step, eventually leading to a perturbed
reconstruction _X_ [ˆ] .


to it (ATTR). Specifically, we can describe ATTR as follows:


1
max [+] _[ E]_ _[−]_ [TR([] _[G]_ [])] _[∥]_ [2] F _[.]_ (5)
_∥E∥_ [2] F _[≤][ϵ]_ [ min] [ _G_ ] 2 _[∥X]_


However, this symmetric min–max objective suffers from a fundamental limitation. In practice, the
update of _E_ does not explicitly maximize the reconstruction error of TR decomposition; instead, it
degenerates into maximizing the difference between successive perturbations, i.e., _∥E_ [(] _[t]_ [)] _−E_ [(] _[t][−]_ [1)] _∥_ [2] F [.]
As a result, ATTR cannot guarantee that the perturbation _E_ effectively degrades the decomposition
performance. In fact, under tiny perturbations, ATTR may even _improve_ the predictive performance
of TR decomposition. This somewhat counterintuitive phenomenon can be explained theoretically
as follows.
**Theorem 1.** _Let δ be the reconstruction error of standard TR-ALS algorithm, and R_ 2 _be the residual_
_term of ATTR algorithm._ _If the perturbation budget ϵ satisfies:_
_√_ _√_
_ϵ <_ _δ −∥R_ 2 _∥F,_ (6)


_then ATTR achieves a smaller reconstruction error than standard TR-ALS._


The proof is deferred to Appendix B.

**Remark 1.** _This result shows that when the perturbation strength ϵ is sufficiently small relative to_
_the gap between the TR-ALS error bound δ and the ATTR residual ∥R_ 2 _∥_ [2] _F_ _[, adversarial training can]_
_improve predictive performance instead of degrading it._


This paradoxical behavior demonstrates the inherent limitation of ATTR: it does not ensure that
perturbations effectively attack the decomposition. This limitation motivates the need for an asymmetric adversarial attack approach to maximize the low-rank approximation error on the tensor
decomposition.


When we focus on minimizing the factors [ _G_ ] of Eq. (5), this model can be formulated as follows:


1
min [+] _[ E]_ _[−]_ [TR([] _[G]_ [])] _[∥]_ [2] F _[,]_ (7)

[ _G_ ] 2 _[∥X]_


where its closed-form solution can be obtained by using the ALS algorithm:

( **G** [(] _n_ _[t]_ [)][)] (2) [= (] **[X]** [ _n_ ] [+] **[ E]** [ _n_ ] [)] **[G]** [(] = _[t]_ _n_ _[−]_ [1)] _[†]_ _,_ (8)

where the **G** = _n_ denotes the unfold tensor composed by [ _G_ ] without factor _Gn_, and ( **G** _n_ )(2) denotes
the mode-2 unfolding of _Gn_ . It is clear that the update of _Gn_ inevitably involves the perturbation
tensor _E_ . Consequently, the updated _G_ = [(] _[t]_ _n_ [)] _[,]_ _[n]_ [=] [1] _[,]_ [ 2] _[, . . ., N]_ [is contaminated by] _[ E]_ [, and therefore] _[ G]_
can be regarded as a function of adversarial perturbation _E_ :


_Gn_ ( _E_ ) := _fn_ ( _E, G_ = _k_ ; _X_ ) _,_ (9)

where _fn_ ( _·_ ) denotes an intrinsic function that mapping the _E_ and _G_ = _n_ to _Gn_ . Fig. 2 illustrates the
gradient flow between the core factors and adversarial perturbation _E_ during the update process.


4


4.2 ADATR ATTACK ALGORITHM


Intuitively, the attacker injects a small, bounded perturbation _E_ into the observed tensor _X_ . In the
traditional symmetric min-max approach as in Eq. (1), the defender (low-rank approximation using
TR-ALS) usually estimates the core factors by minimizing the approximation error on the perturbed
tensor _X_ + _E_ . And then the attacker will maximize _∥X_ + _E_ _−_ TR([ _G_ [(] _[t][−]_ [1)] ]) _∥_ [2] F [with] [fixed] [[] _[G]_ []][.]
However, this deviation does not match the attack goal for tensor decomposition, i.e., to maximize
low-rank approximation error.


To this end, we propose an asymmetric max-min objective to maximize the low-rank approximation
error with respect to _E_ via the intrinsic function _fn_ in Eq. (9), while concurrently minimizing the
approximation error with respect to the core factors [ _G_ ] using a standard low-rank approximation
procedure. Formally, the attacker is modeled by the following bilevel optimization:


1
max _[−]_ [TR([] _[G]_ [(] _[T]_ [ )][(] _[E]_ [)])] _[∥]_ [2] F _[,]_
_E_ 2 _[∥X]_

(10)
1
s.t. [ _G_ [(] _[T]_ [ )] ( _E_ )] = arg min [+] _[ E]_ _[−]_ [TR([] _[G]_ [])] _[∥]_ [2] F _[,]_ _∥E∥_ [2] F _[< ϵ,]_

[ _G_ ] 2 _[∥X]_


where [ _G_ [(] _[T]_ [ )] ( _E_ )] indicates that _Gn_ [(] _[T]_ [ )] is a function of _E_, as defined in Eq. (9), and is obtained from
the minimization problem at the _T_ -th iteration of the TR decomposition.


To maximize Eq. (10) with respect to _E_, we first let _g_ := 2 [1] _[∥X]_ _[−]_ [TR([] _[G]_ [(] _[T]_ [ )][(] _[E]_ [)])] _[∥]_ [2] F [,] [and combine]

with Eq. (9), we can calculate the gradient for _E_ according to the chain rule:


Since _E_ is involved in the dependencies across different _G_ ’s (as illustrated in Fig. 2), explicitly deriving the gradient expression with respect to _E_ becomes extremely complex. Therefore, in practice,
we compute the gradient of _E_ using PyTorch’s automatic differentiation engine and update using
gradient ascent:

_E_ [(] _[t]_ [)] = _E_ [(] _[t][−]_ [1)] + _η_ _[∂g]_ (12)

_∂E_ _[,]_

until reaching the convergence conditions. We summarize the AdaTR algorithm in Algorithm 1.


Noting that adversarial training is not applicable in our setting, since tensor decompositions are nonparametric procedures that recompute factor tensors from scratch for each input and therefore do not
retain trainable parameters for robustness learning. At the same time, although we instantiate the
attack with TR decomposition, the bilevel formulation itself is general and can be directly applied
to other ALS-based tensor models (e.g., CP, Tucker, TT).


4.3 FASTER APPROXIMATE GRADIENT ATTACK MODEL


The proposed AdaTR algorithm needs extensive backpropagation on the intrinsic function _fn_, which
imposes considerable computational overhead. To alleviate this problem, we introduce a faster
approximate gradient strategy of the adversarial attack algorithm (FAG-AdaTR) in this subsection.
Specifically, the method leverages only the gradient of factors [ _G_ [(] _[t]_ [=] _[T]_ [ )] ] update, thereby reducing
resource consumption while preserving the effectiveness of the optimization process.


According to the proposed Eq. (10), we can rewrite it as follows with matrix formulation in the _T_ -th
iteration:
1
max (2) ( _E_ ) **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] ( _E_ ) _∥_ [2] F _[.]_ (13)
**E** [ _n_ ] 2 _[∥]_ **[X]** [[] _[n]_ []] _[ −]_ **[G]** _[n]_ [(] _[t]_ [=] _[T]_ [ )]

To simplify the gradient calculation, we assume **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] ( _E_ ) is independent of _E_ . Therefore, Eq. (13)
can be reformulated as
1
max (2) ( _E_ ) **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] _∥_ [2] F _[.]_ (14)
**E** [ _n_ ] 2 _[∥]_ **[X]** [[] _[n]_ []] _[ −]_ **[G]** _[n]_ [(] _[t]_ [=] _[T]_ [ )]

However, the matrix **G** _[n]_ (2) [(] _[t]_ [=] _[T]_ [ )] ( _E_ ) remains coupled with _E_ across different iterations _t_, which makes

its explicit formulation intractable. We further assume that both **G** _[n]_ (2) [(] _[t]_ [=] _[T][ −]_ [1)] ( _E_ ) and **G** [(] = _[t]_ _n_ [=] _[T][ −]_ [1)] ( _E_ )


5


_∂g_ _[∂g]_
_∂E_ [=] _∂fN_


_∂fN_ _∂g_

_∂E_ [+] _∂fN_ _−_ 1


_N_ _−_ 1

+ _· · ·_ + _[∂g]_
_∂E_ _∂f_ 1


_∂f_ 1


_∂fN_ _−_ 1


_∂f_ 1

(11)
_∂E_ _[.]_


are the variables independent of _E_ . Based on the above assumption, Eq. (14) can be rewritten as


max
**E** [ _n_ ] _[h][n]_ [(] **[E]** [[] _[n]_ []][)] _[,]_ (15)

where _hn_ ( **E** [ _n_ ]) := 2 [1] _[∥]_ **[X]** [[] _[n]_ []] _[−]_ [(] **[X]** [[] _[n]_ []] [+] **[ E]** [[] _[n]_ []][)] **[G]** [(] = _[t]_ _n_ [=] _[T][ −]_ [1)] _[†]_ **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] _∥_ [2] F [.] [Thus,] [the explicit gradient for-]

mulation with respect to the loss function Eq. (15) can be obtained directly:


         -          _∇_ **E** [ _n_ ] _hn_ ( **E** [ _n_ ]) = **X** [ _n_ ] _−_ ( **X** [ _n_ ] + **E** [ _n_ ]) **G** [(] = _[t]_ _n_ [=] _[T][ −]_ [1)] _[†]_ **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] **G** [(] = _[t]_ _n_ [=] _[T]_ [ )] _[†]_ **G** [(] = _[t]_ _n_ [=] _[T][ −]_ [1)] _,_ (16)


which allows the gradient ascent update for _E_ :


where _ωn_ = _In/_ ( [�] _j_ _[I][j]_ [)] [is] [denotes] [the] [mode-] _[n]_ [weight,] [and] [Fold] _[n]_ [(] _[·]_ [)] [:] [R] _[I][n][×][I]_ [1] _[I]_ [2] _[···][I][N]_ _→_

R _[I]_ [1] _[×][I]_ [2] _[×···×][I][N]_ denotes the tensor folding operation.


In contrast to the proposed AdaTR algorithm, FAG-AdaTR reduces the dependency of the adversarial perturbation _E_ on different iterations and different core tensors, thereby allowing the gradient to
be computed more efficiently. The overall optimization procedure for FAG-AdaTR is summarized
in Algorithm 2.


4.4 CONVERGENCE ANALYSIS


In this subsection, we first establish convergence Theorem 2 of AdaTR. Lemma 1 then clarifies
AdaTR cannot exhibit the collapse behavior observed in ATTR based on Theorem 2. Finally, Theorems 3-4 extend the convergence guarantees to the FAG-AdaTR.

**Theorem** **2** (Convergence of AdaTR) **.** _Suppose_ _that_ _assumptions:_ _(i)_ _the_ _map_ _E_ _�→_ [ _G_ [(] _[T]_ [ )] ( _E_ )] _is_
_differentiable on B_ = _{E_ : _∥E∥_ [2] _F_ _[≤]_ _[ϵ][}][ with bounded Jacobian; (ii) the TR reconstruction]_ [ TR(] _[·]_ [)] _[ is]_
_smooth on bounded sets._ _The proposed AdaTR stepsizes satisfy_ 0 _<_ _η_ _≤_ _ηt_ _≤_ _η_ ¯ _≤_ 1 _/L for all t._
_Then the sequence {E_ [(] _[t]_ [)] _} generated by Alg._ _(1) has the following conclusions:_


_1._ _The objective values are monotonically nondecreasing._


_2._ _The sequence {E_ [(] _[t]_ [)] _} is the Cauchy sequence._


_3._ _Any limit point of sequence {E_ [(] _[t]_ [)] _} statisfies the KKT conditions of problem (_ 10 _)._


Detailed lemmas and proofs can be found in Appendix C.


**Lemma 1** (Monotonicity prevents collapse) **.** _Assume (i) Theorem 1 holds so that the perturbation_ _E_

[�]
_produced by ATTR satisfies g_ ( _E_ [�] ) _< g_ ( **0** ) _, and (ii) AdaTR is initialized at E_ [(0)] _with g_ ( _E_ [(0)] ) _> g_ ( **0** ) _._
_Then, by the monotonic ascent property of AdaTR (Theorem 2), we have_


_g_ ( _E_ [(] _[t]_ [)] ) _≥_ _g_ ( _E_ [(0)] ) _>_ _g_ ( **0** ) _>_ _g_ ( _E_ [�] ) _,_ _∀t ≥_ 0 _,_ (18)


_where the_ _E_ _is the perturbation generated by ATTR, g_ ( **0** ) _is the reconstruction error of clean tensor._

[�]
_Thus, in the regime where ATTR reduces the reconstruction error of X_ _, AdaTR always increases it_
_and therefore cannot exhibit the same collapse behavior reported in ATTR._


The proof is provided in Appendix C.3.

**Theorem** **3** (Convergence of FAG-AdaTR) **.** _Suppose_ _that_ _the_ _TR_ _factors_ _{_ **G** [(] = _[T]_ _n_ _[ −]_ [1)] _,_ **G** [(] = _[T]_ _n_ [ )] _[}]_ _n_ _[N]_ =1 _[re-]_
_main_ _bounded_ _on_ _the_ _perturbation_ _ball_ _B_ = _{E_ : _∥E∥_ [2] _F_ _[≤]_ _[ϵ][}][,]_ _[and]_ _[that]_ _[the]_ _[step]_ _[sizes]_ _[satisfy]_
0 _<_ _η_ _≤_ _ηt_ _≤_ _η_ ¯ _≤_ 1 _/L_ _for_ _all_ _t,_ _where_ _L_ _>_ 0 _is_ _a_ _Lipschitz_ _constant_ _of_ _∇g_ ˜ _on_ _B._ _Then_ _the_
_sequence {E_ [(] _[t]_ [)] _} generated by Alg. 2 satisfies:_


_1._ _The objective values are monotonically nondecreasing._
_2._ _The sequence {E_ [(] _[t]_ [)] _} is Cauchy, and hence convergent in B._
_3._ _Any limit point E_ _[⋆]_ _of {E_ [(] _[t]_ [)] _} satisfies the KKT stationarity conditions for the projected maximiza-_
_tion problem (_ 15 _)_


6


_E_ [(] _[t]_ [)] = _E_ [(] _[t][−]_ [1)] + _η_


_N_

- _ωn_ Fold _n_ ( _∇_ **E** [ _n_ ] _hn_ ( **E** [ _n_ ])) _,_ (17)

_n_ =1


The proof follows standard arguments for projected gradient methods on smooth objectives and is
deferred to Appendix D.

**Theorem 4** (Approximate stationarity of FAG-AdaTR) **.** _Let E_ _[⋆]_ _be any limit point of FAG-AdaTR._
_Assume that ∥∇g_ ( _E_ ) _−∇g_ ˜( _E_ ) _∥F_ _≤_ _εg_ _holds on B_ = _{E_ : _∥E∥_ [2] _F_ _[≤]_ _[ϵ][}][.]_ _[Then]_

_⟨∇g_ ( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ _⟩≥−_ _[√]_ _ϵ εg,_ _∀Y_ _∈_ _B._ (19)

_Hence, E_ _[⋆]_ _is an O_ ( _[√]_ _ϵεg_ ) _-approximate stationary point of the true objective g_ ( _E_ ) _._


The proof is provided in Appendix E.


5 EXPERIMENTAL RESULTS


In this section, we compare the proposed methods with the ATTR method under the defense of
different TR decomposition methods. Specifically, the defense baselines include TR-ALS (Zhao
et al., 2016), TRPCA-TNN (Lu et al., 2019), TRNNM (Yu et al., 2019), HQTRC (He & Atia, 2022),
and LRTC-TV (Li et al., 2017). We evaluate the proposed methods on three types of tensor data
(color images, videos, and recommender system datasets) and test them across three representative
tasks: tensor decomposition, tensor completion, and recommendation. Implementation details can
be found in the Appendix.


5.1 COLOR IMAGES DECOMPOSITION ATTACK


In this subsection, we evaluate adversarial attacks on color image decomposition using tensor-based
defense methods. The eight widely-used color images are chosen from the DIV2K dataset [1] for
testing data, and each color image is of the size 672 _×_ 1020 _×_ 3 and normalized into [0, 1].


Table 1: RSE matrix: mean _±_ variance across runs. Lower is better; **bold** marks the worst (highest
RSE) per defense (column).


Attack _\_ Defense TR-ALS TRPCA-TNN TRNNM HQTRC-Cor HQTRC-Cau HQTRC-Hub LRTC-TV


Clean 0.202 _±_ 0.010 0.111 _±_ 0.001 0.152 _±_ 0.001 0.163 _±_ 0.006 0.154 _±_ 0.005 0.126 _±_ 0.002 0.147 _±_ 0.003
Gauss Noise 0.230 _±_ 0.008 0.376 _±_ 0.005 0.546 _±_ 0.011 0.346 _±_ 0.005 0.330 _±_ 0.004 0.377 _±_ 0.005 0.271 _±_ 0.003
ATTR-gen 0.289 _±_ 0.010 0.554 _±_ 0.010 0.556 _±_ 0.011 0.351 _±_ 0.005 0.346 _±_ 0.005 0.360 _±_ 0.005 0.300 _±_ 0.004
AdaTR-gen **0.794** _±_ **0.025** 0.681 _±_ 0.020 0.680 _±_ 0.016 0.453 _±_ 0.011 0.444 _±_ 0.011 0.465 _±_ 0.010 0.340 _±_ 0.005
FAG-AdaTR-gen 0.744 _±_ 0.020 **0.703** _±_ **0.018** **0.686** _±_ **0.017** **0.564** _±_ **0.018** **0.556** _±_ **0.018** **0.578** _±_ **0.017** **0.560** _±_ **0.017**


Tab. 1 shows all methods’ average RSE values over eight color images. The best results are highlighted in bold. It can be seen that the proposed methods achieve superior results to the ATTR in all
cases. Especially in the color Fig. 8 of Appendix, we can see that the reconstructed image of the
proposed AdaTR attack on TR-ALS makes the person indistinguishable to the human eye. Although
our attack was performed only against the TR-ALS algorithm, the adversarial images generated by
attacking TR-ALS transfer to all tested TR-based defense methods and consistently produce the
strongest attack results.


5.2 VIDEOS DECOMPOSITION ATTACK


In this subsection, we evaluate the effectiveness of the proposed method on color video data [2] for
the tensor decomposition task. For fair comparison, we randomly select the seven color videos,
and each video in the dataset consists of at least 150 frames. Moreover, Zhou et. al (Zhou et al.,
2017) find that the color video _news_ of 30 frames has much more redundant information in their
experiment. Thus, we select the ten consistent frames for each color video. Each video segment is
thus represented as a fourth-order tensor of size 144 _×_ 176 _×_ 3 _×_ 10 (spatial height _×_ spatial width
_×_ color channel _×_ frame).


Fig. 3 presents the evaluation results of the average PSNR, SSIM, RSE of all methods. The numerical comparison in Fig. 3 clearly demonstrates the superiority of the proposed methods. The


[1https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
[2http://trace.eas.asu.edu/yuv/](http://trace.eas.asu.edu/yuv/)


7


Figure 3: Average RSE, PSNR, and SSIM results over seven videos on tensor decomposition tasks.


proposed methods achieve approximately 2 times gain in average RSE compared to TR-ALS, which
demonstrates its competitive advantage in terms of the attack on tensor decomposition. To facilitate
a more comprehensive visual comparison, Fig. 20 presents the reconstruction results for the 5th
frame of the color video _news_ . It can be clearly observed that the proposed method is able to attack
local details and destroy the global structure.


5.3 TENSOR COMPLETION ATTACK


In this subsection, we conduct experiments on color images to assess the effectiveness of the proposed method for the tensor completion task. The same testing data used as subsection 5.1, and each
color image is normalized into [0, 1]. Fig. 4 presents the TC results for the randomly chosen color
image with a sampling rate of 0.2. It can be clearly observed that the reconstructed image of the
proposed AdaTR attack on TR-ALS makes the person indistinguishable to the human eye.


Figure 4: Visual example on tensor completion tasks under different attacks and defenses. Results
are shown for one sample image (other seven cases are provided in the Appendix H).


5.4 RECOMMENDER SYSTEMS DECOMPOSITION ATTACK


In this subsection, we conduct experiments on a recommender systems dataset to assess the effectiveness of the proposed method for the recommendation task. To validate the recommendation performance, we extend the proposed AdaTR attack algorithm to the NMF model, termed AdaNMF. We
use two datasets, including a synthetic dataset and the widely-used MovieLens-100K dataset [3] . The
synthetic dataset is generated by randomly sampling 500 users and 450 items, with ratings ranging
from 1 to 5,and we sample only 12% of the entries as observations. The MovieLens-100K dataset
consists of 943 users and 1682 items, with about 6% of the user–item pairs observed. All training,
testing, and perturbation operations are performed strictly on these observed entries. We preprocess


[3https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)


8


Figure 5: Adversarial attack in TR-ALS decomposition with different ranks defend. (a) PSNR, (b)
SSIM, and (c) RSE results over a color image on tensor decomposition tasks.


the datasets by normalizing the ratings to the range [1, 5] and splitting them into training (80%) and
testing (20%) sets. Regarding the NMF ranks in the proposed method, we set _R_ = 20. Moreover,
the values of _ϵ_ and _η_ are set to 10 and 0.05, respectively. And **inner** ~~**n**~~ **um** and **outer** ~~**n**~~ **um** are
fixed to 200 and 100 in all experiments.


Table 2: NMF recommender performance under different perturbations on Synthetic and
MovieLens-100K datasets. Columns indicate desired direction: RMSE ( _↓_ better), Precision@10 ( _↑_
better), Recall@10 ( _↑_ better). **Bold** highlights effective attacks (AdaNMF) where RMSE increases
and Precision/Recall decreases vs. Clean.


**Synthetic** **MovieLens-100K**
**Condition**

RMSE ↑ P@10 ↓ R@10 ↓ RMSE ↑ P@10 ↓ R@10 ↓


Clean 3.718 0.0132 0.0423 3.967 0.0200 0.0992
Gaussian noise 3.726 0.0124 0.0333 3.983 0.0153 0.0766
ATNMF **4.026** 0.0142 0.0502 **4.089** 0.0123 0.0728
**AdaNMF** 4.024 **0.0089** **0.0266** 4.078 **0.0076** **0.0461**


Tab. 2 presents the evaluation results of the RMSE, Precision@10, and Recall@10 of all methods.
The best results are highlighted in bold. It can be seen that the proposed AdaNMF method achieves
superior results to the ATNMF in most cases.


5.5 TR-RANKS ROBUSTNESS OF ATTACKS


In this subsection, we conduct experiments to evaluate the robustness of the proposed methods
against different TR-ranks. We randomly select one of the color images from the DIV2K dataset as
the testing data. Fig. 5 shows the RSE, PSNR, and SSIM values of the reconstructed image under
different TR-ranks. It can be seen that the proposed methods achieve superior robustness results
compared to the ATTR in all cases. Noting that we only attack the TR-ALS algorithm with the same
TR-ranks _R_ 1 = _R_ 2 = _R_ 3 = 5 in all the attack methods.


5.6 JPEG, PNG IMAGE DEFENDING


In this subsection, we evaluate the effectiveness of the proposed methods under two common image compression formats: PNG and JPEG, since encoding/decoding may partially remove smallmagnitude adversarial perturbations. We randomly select one of the color images from the DIV2K
dataset as the testing data. Tab. 3 shows the RSE, PSNR, and SSIM values of the reconstructed
image under different image storage formats. It can be seen that the JPEG and PNG image storage
formats have little effect on the performance of the proposed methods.


5.7 HYPERPARAMETER EXPERIMENT


In this subsection, we conduct experiments to evaluate the impact of the hyperparameter _ϵ_ on the
performance of the proposed methods. We randomly select eight of the color images from the


9


Table 3: Comparison between **PNG** and **JPEG** image compression formats.

|PNG JPEG<br>Method<br>RSE ↓ PSNR ↑ SSIM ↑ RSE ↓ PSNR ↑ SSIM ↑|Col2|
|---|---|
|Clean<br>0.087<br>25.749<br>0.839<br>Gaussian Noise<br>0.154<br>20.818<br>0.682<br>ATTR-gen<br>0.174<br>19.744<br>0.657<br>AG-AdaTR-gen<br>0.348<br>13.725<br>0.326<br>AdaTR-gen<br>**0.521**<br>**10.229**<br>**0.156**|0.088<br>25.648<br>0.835<br>0.165<br>20.234<br>0.669<br>0.186<br>19.174<br>0.642<br>0.334<br>14.083<br>0.337<br>**0.495**<br>**10.667**<br>**0.175**|


DIV2K dataset as the testing data. As shown in Fig. 1, our methods are effective even at small
perturbation budgets, while ATTR can actually improve reconstruction performance when _ϵ_ is very
small.


5.8 CONVERGENCE ANALYSIS


In this subsection, we experimentally analyze the numerical convergence behaviour to verify the
convergence of the proposed methods. Fig. 6 illustrates the average reconstructed error value curves
of the eight color images with _ϵ_ = 66. We can observe that the loss function value converges to a
specific value in the end, which implies that the proposed method is convergent numerically. The
experimental results support the efficacy of the proposed methods in achieving convergence and
validate their usefulness in practical scenarios.


|Col1|AdaTR<br>FAG-AdaT|R|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


|Col1|AdaTR<br>FAG-AdaTR|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


Figure 6: Convergence of the proposed AdaTR and FAG-AdaTR methods in terms of reconstruction
error, averaged over 8 images decomposition with perturbation budget _ϵ_ = 66. Results are shown
for (a) TR-Ranks = [5,5,5] and (b) TR-Ranks = [10,10,10].


6 CONCLUSION


This paper proposes a novel asymmetric adversarial attack approach on TR decomposition (AdaTR)
via min-max optimization, which can generate perturbation to the original tensor that significantly
degrade the performance of TR decomposition. To address the high computational cost of AdaTR,
we further propose a faster approximate gradient adversarial attack on TR decomposition (FAGAdaTR) while maintaining strong attack effectiveness. Extensive experiments on color images,
videos, and recommender systems demonstrate the effectiveness of the proposed methods in attacking TR decomposition and its applications. In future work, we will extend the proposed methods
to other tensor decomposition models (Zheng et al., 2021; Wu et al., 2022; Loeschcke et al., 2024),
and explore broader applications. In particular, when applied to large language model (LLM) compression (Hajimolahoseini et al., 2021; Ma et al., 2019), recommender systems (Chen et al., 2021),
or tensor decomposition-based purification (Entezari & Papalexakis, 2022; Bhattarai et al., 2023),
our approach highlights the importance of security issues in these domains, since their performance
may also be affected by the vulnerability of tensor decomposition.


10


13.8


13.6


13.4


13.2


13.0


12.8


12.6


Step


(a) TR-Ranks = [5,5,5]


5.50


5.25


5.00


4.75


4.50


4.25


4.00


Step


(b) TR-Ranks = [10,10,10]


ETHICS STATEMENT


This work uses only computational methods and publicly available datasets, with no human subjects
or private data. It follows the ICLR Code of Ethics, with no conflicts of interest. While acknowledging potential dual-use concerns, we stress responsible deployment and adhere to research integrity.
All methods and results are reported transparently to support reproducibility.


REPRODUCIBILITY STATEMENT


We provide implementation details in the appendix to support reproduction of the main results.


REFERENCES


Evrim Acar, Daniel M Dunlavy, Tamara G Kolda, and Morten Mørup. Scalable tensor factorizations
for incomplete data. _Chemometrics and Intelligent Laboratory Systems_, 106(1):41–56, 2011.


Manish Bhattarai, Mehmet Cagri Kaymak, Ryan Barron, Ben Nebgen, Kim Rasmussen, and Boian S
Alexandrov. Robust adversarial defense by tensor factorization. In _2023 International Conference_
_on Machine Learning and Applications (ICMLA)_, pp. 308–315. IEEE, 2023.


Ting Cai, Vincent YF Tan, and C´edric F´evotte. Adversarially-trained nonnegative matrix factorization. _IEEE Signal Processing Letters_, 28:1415–1419, 2021.


Zhengyu Chen, Ziqing Xu, and Donglin Wang. Deep transfer tensor decomposition with orthogonal constraint for recommender systems. In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_ _artificial_
_intelligence_, volume 35, pp. 4010–4018, 2021.


Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip: White-box adversarial examples
for text classification. _arXiv preprint arXiv:1712.06751_, 2017.


Negin Entezari and Evangelos E Papalexakis. Tensorshield: Tensor-based defense against adversarial attacks on images. In _MILCOM_ _2022-2022_ _IEEE_ _Military_ _Communications_ _Conference_
_(MILCOM)_, pp. 999–1004. IEEE, 2022.


Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial
examples. _arXiv preprint arXiv:1412.6572_, 2014.


Habib Hajimolahoseini, Mehdi Rezagholizadeh, Vahid Partovinia, Marzieh Tahaei, Omar Mohamed
Awad, and Yang Liu. Compressing pre-trained language models using progressive low rank decomposition. _Advances in Neural Information Processing Systems_, 35:6–14, 2021.


Yicong He and George K Atia. Coarse to fine two-stage approach to robust tensor completion of
visual data. _IEEE Transactions on Cybernetics_, 54(1):136–149, 2022.


Yicong He and George K Atia. Scalable and robust tensor ring decomposition for large-scale data.
In _Uncertainty in Artificial Intelligence_, pp. 860–869. PMLR, 2023.


Frank L Hitchcock. The expression of a tensor or a polyadic as a sum of products. _Journal_ _of_
_Mathematics and Physics_, 6(1-4):164–189, 1927.


Do-Hyung Kang, Hang Joon Jo, Wi Hoon Jung, Sun Hyung Kim, Ye-Ha Jung, Chi-Hoon Choi,
Ul Soon Lee, Seung Chan An, Joon Hwan Jang, and Jun Soo Kwon. The effect of meditation
on brain structure: cortical thickness mapping and diffusion tensor imaging. _Social cognitive and_
_affective neuroscience_, 8(1):27–33, 2013.


Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. _SIAM review_, 51(3):
455–500, 2009.


Fuwei Li, Lifeng Lai, and Shuguang Cui. On the adversarial robustness of subspace learning. _IEEE_
_Transactions on Signal Processing_, 68:1470–1483, 2020.


11


Xutao Li, Yunming Ye, and Xiaofei Xu. Low-rank tensor completion with total variation for visual
data inpainting. In _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_, volume 31,
2017.


Ying Li, Fuwei Li, Lifeng Lai, and Jun Wu. On the adversarial robustness of principal component
analysis. In _ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal_
_Processing (ICASSP)_, pp. 3695–3699. IEEE, 2021.


Sebastian Loeschcke, Dan Wang, Christian Leth-Espensen, Serge Belongie, Michael J Kastoryano,
and Sagie Benaim. Coarse-to-fine tensor trains for compact visual representations. _arXiv preprint_
_arXiv:2406.04332_, 2024.


Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin, and Shuicheng Yan. Tensor robust
principal component analysis with a new tensor nuclear norm. _IEEE_ _transactions_ _on_ _pattern_
_analysis and machine intelligence_, 42(4):925–938, 2019.


Lei Luo, Yanfu Zhang, and Heng Huang. Adversarial nonnegative matrix factorization. In _Interna-_
_tional Conference on Machine Learning_, pp. 6479–6488. PMLR, 2020.


Xindian Ma, Peng Zhang, Shuai Zhang, Nan Duan, Yuexian Hou, Ming Zhou, and Dawei Song.
A tensorized transformer for language modeling. _Advances_ _in_ _neural_ _information_ _processing_
_systems_, 32, 2019.


Reza Mahmoodi, Seyed Amjad Seyedi, Fardin Akhlaghian Tab, and Alireza Abdollahpouri. Link
prediction by adversarial nonnegative matrix factorization. _Knowledge-based_ _systems_, 280:
110998, 2023.


Reza Mahmoodi, Seyed Amjad Seyedi, Alireza Abdollahpouri, and Fardin Akhlaghian Tab. Enhancing link prediction through adversarial training in deep nonnegative matrix factorization.
_Engineering Applications of Artificial Intelligence_, 133:108641, 2024.


Ivan V Oseledets. Tensor-train decomposition. _SIAM Journal on Scientific Computing_, 33(5):2295–
2317, 2011.


Daniel L Pimentel-Alarc´on, Aritra Biswas, and Claudia R Sol´ıs-Lemus. Adversarial principal component analysis. In _2017 IEEE International Symposium on Information Theory (ISIT)_, pp. 2363–
2367. IEEE, 2017.


Kristof T Sch¨utt, Stefan Chmiela, O Anatole Von Lilienfeld, Alexandre Tkatchenko, Koji Tsuda,
and Klaus-Robert M¨uller. Machine learning meets quantum physics. _Lecture_ _Notes_ _in_ _Physics_,
2020.


Seyed Amjad Seyedi, Fardin Akhlaghian Tab, Abdulrahman Lotfi, Navid Salahian, and Jovan
Chavoshinejad. Elastic adversarial deep nonnegative matrix factorization for matrix completion.
_Information Sciences_, 621:562–579, 2023.


Nicholas D Sidiropoulos, Lieven De Lathauwer, Xiao Fu, Kejun Huang, Evangelos E Papalexakis,
and Christos Faloutsos. Tensor decomposition for signal processing and machine learning. _IEEE_
_Transactions on signal processing_, 65(13):3551–3582, 2017.


Ledyard R Tucker. Some mathematical notes on three-mode factor analysis. _Psychometrika_, 31(3):
279–311, 1966.


Minh Vu, Ben Nebgen, Erik Skau, Geigh Zollicoffer, Juan Castorena, Kim Rasmussen, Boian
Alexandrov, and Manish Bhattarai. Lafa: Latent feature attacks on non-negative matrix factorization. _arXiv preprint arXiv:2408.03909_, 2024.


Lu Wang, Tianyuan Zhang, Yang Qu, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu, and
Dacheng Tao. Black-box adversarial attack on vision language models for autonomous driving.
_arXiv preprint arXiv:2501.13563_, 2025.


Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment:
from error visibility to structural similarity. _IEEE transactions on image processing_, 13(4):600–
612, 2004.


12


Zhong-Cheng Wu, Ting-Zhu Huang, Liang-Jian Deng, Hong-Xia Dou, and Deyu Meng. Tensor
wheel decomposition and its tensor completion application. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 35:27008–27020, 2022.


Jinshi Yu, Chao Li, Qibin Zhao, and Guoxu Zhao. Tensor-ring nuclear norm minimization and
application for visual: Data completion. In _ICASSP 2019-2019 IEEE international conference on_
_acoustics, speech and signal processing (ICASSP)_, pp. 3142–3146. IEEE, 2019.


Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, and Xueqi Cheng. Understanding and
improving adversarial collaborative filtering for robust recommendation. _Advances_ _in_ _Neural_
_Information Processing Systems_, 37:120381–120417, 2024.


Qibin Zhao, Guoxu Zhou, Shengli Xie, Liqing Zhang, and Andrzej Cichocki. Tensor ring decomposition. _arXiv preprint arXiv:1606.05535_, 2016.


Yu-Bang Zheng, Ting-Zhu Huang, Xi-Le Zhao, Qibin Zhao, and Tai-Xiang Jiang. Fully-connected
tensor network decomposition and its application to higher-order tensor completion. In _Proceed-_
_ings of the AAAI conference on artificial intelligence_, volume 35, pp. 11071–11078, 2021.


Pan Zhou, Canyi Lu, Zhouchen Lin, and Chao Zhang. Tensor factorization for low-rank tensor
completion. _IEEE Transactions on Image Processing_, 27(3):1152–1163, 2017.


Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models. _arXiv_ _preprint_
_arXiv:2307.15043_, 2023.


13


A STATEMENT OF THE USE OF LARGE LANGUAGE MODELS (LLMS)


In this paper, we just used the LLM, ChatGPT, to polish the language of the paper. We did not use
LLMs to generate any content or ideas in this work. We have verified the accuracy of all content and
ideas in the paper.


B PROOF OF THEOREM 1


_Proof._ For standard TR-ALS, the tensor _X_ can be expressed as


_X_ = TR([ _G_ ]) + _R_ 1 _,_ _∥R_ 1 _∥_ [2] F [=] _[ δ,]_ (20)


where the _R_ 1 is residual term of TR-ALS. For ATTR, we have

_X_ = TR([ _G_ ]) _−E_ + _R_ 2 _,_ _∥−E_ + _R_ 2 _∥_ [2] F _[≤]_ [(] _[∥E∥]_ [F] [+] _[ ∥R]_ [2] _[∥]_ [F][)][2] _[,]_ (21)

where _∥E∥_ [2] F _[≤]_ _[ϵ]_ [.]


If
( _∥E∥_ F + _∥R_ 2 _∥_ F) [2] _< ∥R_ 1 _∥_ [2] F _[,]_
then ATTR yields a strictly smaller reconstruction error. Since _∥R_ 1 _∥_ [2] F [=] _[δ]_ [and] _[∥E∥]_ [2] F _[≤]_ _[ϵ]_ [,] [this]
condition is satisfied whenever _√_ _√_
_ϵ <_ _δ −∥R_ 2 _∥_ F _._


Thus, ATTR achieves a smaller reconstruction error than standard TR-ALS under the stated condition.


C PROOF OF THEOREM 2


We first prove the smoothness of the surrogate objective, and boundedness of the variables. Then
we prove they are the Cauchy sequence if Algorithm 1. To prove the boundedness of multipliers of
Algorithm 1, we first introduce the following lemma.


C.1 SMOOTHNESS OF THE SURROGATE OBJECTIVE


**Lemma 2.** _Chain rule for ∇E_ _g_ ( _E_ ) _Assuming that:_ _(i) the map E_ _�→_ [ _G_ [(] _[T]_ [ )] ( _E_ )] _is differentiable on B_
_with bounded Jacobian; (ii) the TR reconstruction_ TR( _·_ ) _is smooth on bounded sets._


_Then, let_
_h_ ([ _G_ ]) = [1] 2 �� _X_ _−_ TR�[ _G_ [(] _[T]_ [ )] ( _E_ )]���2 _F_ _[.]_ (22)

_Then_

                    -                     _∇E_ _g_ ( _E_ ) = _J_ [(] _[T]_ [ )] ( _E_ ) _[⊤]_ _∇_ [ _G_ ( _T_ )( _E_ )] _h_ [ _G_ [(] _[T]_ [ )] ( _E_ )] _,_ (23)


_Proof._ Let _e_ = vec( _E_ ), _θ_ ( _E_ ) = vec([ _G_ [(] _[T]_ [ )] ( _E_ )]). Then _g_ ( _E_ ) = _h_ ( _θ_ ( _E_ )). By the multivariate chain
rule,
_∇eg_ = ( _∂θ/∂e_ ) _[⊤]_ _∇θh_ ( _θ_ ) _,_

giving Eq. (23).


**Lemma 3.** _Lipschitz continuity of ∇E_ _g_ ( _E_ )


_Assume that for all E_ 1 _, E_ 2 _∈B:_


_∥_ [ _G_ [(] _[T]_ [ )] ( _E_ 1)] _−_ [ _G_ [(] _[T]_ [ )] ( _E_ 2)] _∥F_ _≤_ _LG∥E_ 1 _−E_ 2 _∥F,_


14


_where_


�[ _G_ [(] _[T]_ [ )] ( _E_ )]�
_J_ [(] _[T]_ [ )] ( _E_ ) = _[∂]_ [vec] _._ (24)

_∂_ vec( _E_ )


_∥J_ [(] _[T]_ [ )] ( _E_ ) _∥≤_ _MJ_ _,_ _∥J_ [(] _[T]_ [ )] ( _E_ 1) _−_ _J_ [(] _[T]_ [ )] ( _E_ 2) _∥≤_ _LJ_ _∥E_ 1 _−E_ 2 _∥F,_


_and ∇_ [ _G_ ] _h_ ([ _G_ ]) _is Lipschitz and bounded with constants Lh, Mh on {_ [ _G_ [(] _[T]_ [ )] ( _E_ )] : _E_ _∈B}._


_Then_
_∥∇E_ 1 _g_ ( _E_ 1) _−∇E_ 2 _g_ ( _E_ 2) _∥F_ _≤_ _L∥E_ 1 _−E_ 2 _∥F,_ _L_ := _MJ_ _LhLG_ + _LJ_ _Mh._ (25)


_Proof._ Let


                       -                        _Ji_ := _J_ [(] _[T]_ [ )] ( _Ei_ ) _,_ [ _G_ [(] _[T]_ [ )] ] _i_ := [ _G_ [(] _[T]_ [ )] ( _Ei_ )] _,_ _vi_ := _∇_ [ _G_ ( _T_ )] _ih_ [ _G_ [(] _[T]_ [ )] ] _i_ _,_ _i_ = 1 _,_ 2 _._ (26)


By Lemma 1, one has
_∇Eig_ ( _Ei_ ) = _Ji_ _[⊤][v][i][,]_ _i_ = 1 _,_ 2 _._ (27)
Thus,
_∇E_ 1 _g_ ( _E_ 1) _−∇E_ 2 _g_ ( _E_ 2) = _J_ 1 _[⊤]_ [(] _[v]_ [1] _[−]_ _[v]_ [2][) + (] _[J]_ 1 _[⊤]_ _[−]_ _[J]_ 2 _[⊤]_ [)] _[v]_ [2] _[.]_ (28)
Taking norms and applying the triangle inequality yields

�� _∇E_ 1 _g_ ( _E_ 1) _−∇E_ 2 _g_ ( _E_ 2)�� _≤∥J_ 1 _⊤_ [(] _[v]_ [1] _[−]_ _[v]_ [2][)] _[∥]_ [+] _[ ∥]_ [(] _[J]_ 1 _[⊤]_ _[−]_ _[J]_ 2 _[⊤]_ [)] _[v]_ [2] _[∥]_ [=:] _[ T]_ [1] [+] _[ T]_ [2] _[.]_ (29)


We bound the two terms separately.


BOUND ON _T_ 1. By submultiplicativity of the operator norm,

_T_ 1 = _∥J_ 1 _[⊤]_ [(] _[v]_ [1] _[−]_ _[v]_ [2][)] _[∥≤∥][J]_ 1 _[⊤][∥∥][v]_ [1] _[−]_ _[v]_ [2] _[∥]_ [=] _[ ∥][J]_ [1] _[∥∥][v]_ [1] _[−]_ _[v]_ [2] _[∥][.]_ (30)


Using the assumption _∥J_ [(] _[T]_ [ )] ( _E_ ) _∥≤_ _MJ_,


_∥J_ 1 _∥≤_ _MJ_ _._ (31)


Since _∇_ [ _G_ ] _h_ is _Lh_ -Lipschitz,


( _T_ ) ( _T_ ) ( _T_ ) ( _T_ )
_∥v_ 1 _−_ _v_ 2 _∥_ = �� _∇_ [ _G_ ( _T_ )]1 _h_ ([ _G_ ]1) _−∇_ [ _G_ ( _T_ )]2 _h_ ([ _G_ ]2)�� _≤_ _Lh_ ��[ _G_ ]1 _−_ [ _G_ ]2�� _._ (32)


Finally, by the Lipschitz property of the ALS map,

��[ _G_ ( _T_ )]1 _−_ [ _G_ ( _T_ )]2�� _≤_ _LG ∥E_ 1 _−E_ 2 _∥F ._ (33)


Combining Eq. (30)–Eq. (33) gives


_T_ 1 _≤_ _MJ_ _LhLG∥E_ 1 _−E_ 2 _∥F ._ (34)


BOUND ON _T_ 2. Similarly,

_T_ 2 = _∥_ ( _J_ 1 _[⊤]_ _[−]_ _[J]_ 2 _[⊤]_ [)] _[v]_ [2] _[∥≤∥][J]_ 1 _[⊤]_ _[−]_ _[J]_ 2 _[⊤][∥∥][v]_ [2] _[∥]_ [=] _[ ∥][J]_ [1] _[−]_ _[J]_ [2] _[∥∥][v]_ [2] _[∥][.]_ (35)


Using the Lipschitz assumption on the _jI_,


_∥J_ 1 _−_ _J_ 2 _∥≤_ _LJ_ _∥E_ 1 _−E_ 2 _∥F ._ (36)


Using the boundedness of _∇_ [ _G_ ] _h_,


( _T_ )
_∥v_ 2 _∥_ = �� _∇_ [ _G_ ( _T_ )]2 _h_ ([ _G_ ]2)�� _≤_ _Mh._ (37)


Therefore,
_T_ 2 _≤_ _LJ_ _Mh∥E_ 1 _−E_ 2 _∥F ._ (38)


FINAL BOUND. Combining Eq. (34) and Eq. (38) with Eq. (29) yields

�� _∇E_ 1 _g_ ( _E_ 1) _−∇E_ 2 _g_ ( _E_ 2)�� _≤_ ( _MJ_ _LhLG_ + _LJ_ _Mh_ ) _∥E_ 1 _−E_ 2 _∥F ._ (39)


Thus the Lipschitz constant is _L_ = _MJ_ _LhLG_ + _LJ_ _Mh_, which proves Eq. (25).


15


C.2 PROOF OF THEOREM 2


_Proof._ Since _∇g_ is _L_ -Lipschitz (Lemma 2), the function _g_ is _L_ -smooth. Thus, for all _E,_ _E_ _∈B_,

[�]

_g_ ( _E_ [�] ) _≥_ _g_ ( _E_ ) + _⟨∇g_ ( _E_ ) _,_ _E_ [�] _−E⟩−_ _[L]_ 2 _[∥][E]_ [�] _[−E∥]_ [2] _F_ _[.]_ (40)


Let _E_ = _E_ [(] _[t]_ [)], _E_ = _E_ [(] _[t]_ [+1)], and denote

[�]

_Z_ _[t]_ = _E_ [(] _[t]_ [)] + _ηt∇g_ ( _E_ [(] _[t]_ [)] ) _._


Since _E_ [(] _[t]_ [+1)] = Π _B_ ( _Z_ _[t]_ ), the optimality condition of the Euclidean projection onto the convex set _B_
implies
_⟨Z_ _[t]_ _−E_ [(] _[t]_ [+1)] _,_ _E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _⟩≥_ 0 _._

Substituting _Z_ _[t]_ = _E_ [(] _[t]_ [)] + _ηt∇g_ ( _E_ [(] _[t]_ [)] ) yields


_ηt ⟨∇g_ ( _E_ [(] _[t]_ [)] ) _,_ _E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _⟩≥∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[.]_ (41)


Plugging Eq. (41) into the smoothness inequality Eq. (40) gives


and because _g_ is bounded above on _B_, letting _T_ _→∞_ yields


_∞_

     - _∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[<][ ∞][,]_ (43)

_t_ =0


implying _∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥F_ _→_ 0. Hence, the sequence _{E_ [(] _[t]_ [)] ) _}_ is Cauchy sequence.


By compactness, _{E_ [(] _[t]_ [)] _}_ admits a convergent subsequence _E_ _[t][k]_ _→E_ _[⋆]_ . Since _ηt_ _∈_ [ _η,_ ¯ _η_ ], we may
assume _ηtk_ _→_ _η_ _[⋆]_ _∈_ [ _η,_ ¯ _η_ ]. Using Eq. (43), _E_ _[t][k]_ [+1] _−E_ _[t][k]_ _→_ 0, hence _E_ _[t][k]_ [+1] _→E_ _[⋆]_ as well.


Passing to the limit in the update rule,

_E_ _[t][k]_ [+1] = Π _B_             - _E_ _[t][k]_ + _ηtk_ _∇g_ ( _E_ _[t][k]_ )� _,_


and using continuity of _∇g_ and Π _B_, we obtain the fixed-point relation

_E_ _[⋆]_ = Π _B_ ( _E_ _[⋆]_ + _η_ _[⋆]_ _∇g_ ( _E_ _[⋆]_ )) _._ (44)


The projection fixed-point condition Eq. (44) is equivalent to the variational inequality


_⟨∇g_ ( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ _⟩≤_ 0 _,_ _∀Y_ _∈B,_


which are exactly the KKT conditions for maximizing _g_ over _B_ . Thus, every limit point of the
sequence is KKT-stationary.


16


        - 1
_g_ ( _E_ [(] _[t]_ [+1)] ) _≥_ _g_ ( _E_ [(] _[t]_ [)] ) + _−_ _[L]_
_ηt_ 2


_∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[.]_ (42)


Since _ηt_ _≤_ 1 _/L_, the coefficient is nonnegative, proving monotonic ascent:


_g_ ( _E_ [(] _[t]_ [+1)] ) _≥_ _g_ ( _E_ [(] _[t]_ [)] ) _._


Because _B_ is compact, _{g_ ( _E_ [(] _[t]_ [)] ) _}_ is monotone and bounded above, and hence convergent.


Summing Eq. (42) from _t_ = 0 to _T_,


_g_ ( _E_ [(] _[t]_ [+1)] ) _−_ _g_ ( _E_ [0] ) _≥_


_T_


_t_ =0


- 1

_−_ _[L]_
_ηt_ 2


_∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[.]_


Since _ηt_ _≤_ 1 _/L_,
1

_−_ _[L]_
_ηt_ 2


_[L]_

2 _[≥]_ _[L]_ 2


2 [=:] _[ c >]_ [ 0] _[,]_


C.3 PROOF OF LEMMA 1


_Proof._ Theorem 2 gives _g_ ( _E_ [(] _[t]_ [+1)] ) _≥_ _g_ ( _E_ [(] _[t]_ [)] ) _≥_ _g_ ( _E_ [(0)] ). Combining _g_ ( _E_ [(0)] ) _>_ _g_ ( **0** ) with _g_ ( _E_ [�] ) _<_
_g_ ( **0** ) (Theorem 1) yields the claim.


D PROOF OF THEOREM 3


In this appendix we establish the convergence of FAG-AdaTR stated in Theorem 3. The key observation is that each mode-wise loss _hn_ ( **E** [ _n_ ]) is a smooth quadratic function of **E** [ _n_ ], and thus
its gradient is Lipschitz on bounded sets. Summing over _n_ preserves smoothness and Lipschitz
continuity of the global surrogate ˜ _g_ .


D.1 SMOOTHNESS OF THE SURROGATE OBJECTIVE


We first show that every mode-wise loss _hn_ has a Lipschitz continuous gradient.


**Lemma 4** (Lipschitz continuity of _∇hn_ ) **.** _Fix n ∈{_ 1 _, . . ., N_ _} and define_


_In particular, for all E_ 1 _, E_ 2 _∈_ _B,_

�� _∇g_ ˜( _E_ 1) _−∇g_ ˜( _E_ 2)�� _F_ _[≤]_ _[L]_ �� _E_ 1 _−E_ 2�� _F_ _[.]_ (54)


17


**M** _n_ := **G** [(] = _[T]_ _n_ _[ −]_ [1)] _[ †]_ **G** [(] = _[T]_ _n_ [ )] _[∈]_ [R] 


_j_ = _n_ _[I][j]_ _[×]_ [�]


_j_ = _n_ _[I][j]_ _._ (45)


_Assume that_ **M** _n is bounded on the perturbation ball B, i.e., there exists Cn_ _>_ 0 _such that ∥_ **M** _n∥_ 2 _≤_
_Cn_ _for all iterates._ _Then hn_ _is Ln-smooth on B with_


_Ln_ = �� **M** _⊤n_ **[M]** _[n]_ ��2 _[≤]_ _[C]_ _n_ [2] _[.]_ (46)


_In particular, for any E_ 1 _, E_ 2 _∈_ _B,_

�� _∇hn_ (( **E** 1)[ _n_ ]) _−∇hn_ (( **E** 2)[ _n_ ])�� _F_ _[≤]_ _[L][n]_ ��( **E** 1)[ _n_ ] _−_ ( **E** 2)[ _n_ ]�� _F_ _[.]_ (47)


_Proof._ By definition,


_hn_ ( **E** [ _n_ ]) = [1] 2


�� **X** [ _n_ ] _−_ ( **X** [ _n_ ] + **E** [ _n_ ]) **M** _n_ ��2 _F_ [=] [1] 2


2
�� **R** _n −_ **E** [ _n_ ] **M** _n_ �� _F_ _[,]_ (48)


where **R** _n_ := **X** [ _n_ ] _−_ **X** [ _n_ ] **M** _n_ is independent of **E** [ _n_ ]. Expanding the gradient of this quadratic form
yields
_∇_ **E** [ _n_ ] _hn_ ( **E** [ _n_ ]) =        - **E** [ _n_ ] **M** _n −_ **R** _n_        - **M** _[⊤]_ _n_ _[.]_ (49)


Thus, for any ( **E** 1)[ _n_ ] _,_ ( **E** 2)[ _n_ ],


_∇hn_ (( **E** 1)[ _n_ ]) _−∇hn_ (( **E** 2)[ _n_ ]) = �(( **E** 1)[ _n_ ] _−_ ( **E** 2)[ _n_ ]) **M** _n_      - **M** _[⊤]_ _n_ _[,]_ (50)
�� _∇hn_ (( **E** 1)[ _n_ ]) _−∇hn_ (( **E** 2)[ _n_ ])�� _F_ _[≤]_ ��( **E** 1)[ _n_ ] _−_ ( **E** 2)[ _n_ ]�� _F_ �� **M** _n_ **M** _⊤n_ ��2 (51)

= �� **M** _⊤n_ **[M]** _[n]_ ��2 ��( **E** 1)[ _n_ ] _−_ ( **E** 2)[ _n_ ]�� _F_ _[.]_ (52)


Therefore _Ln_ = _∥_ **M** _[⊤]_ _n_ **[M]** _[n][∥]_ [2] [is a Lipschitz constant for] _[ ∇][h][n]_ [on] _[ B]_ [.] [The bound] _[ L][n]_ _[≤]_ _[C]_ _n_ [2] [follo][ws]
from _∥_ **M** _[⊤]_ _n_ **[M]** _[n][∥]_ [2] _[≤∥]_ **[M]** _[n][∥]_ [2] 2 [.]


We now lift this property from the mode-wise losses _hn_ to the full surrogate _g_ ˜( **E** ) =

- _N_
_n_ =1 _[ω][n][h][n]_ [(] **[E]** [[] _[n]_ []][)][.]

**Lemma 5** (Lipschitz continuity of _∇g_ ˜) **.** _Under the assumptions of Lemma 4, the surrogate objective_
_g_ ˜ _is L-smooth on B, with_


_L_ _≤_


_N_

- _ωnLn._ (53)


_n_ =1


_Proof._ By linearity of the gradient,


Hence _∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥F_ _→_ 0 and _{E_ [(] _[t]_ [)] _}_ is a Cauchy sequence in the complete metric space _B_, so
it converges.


Finally, let _E_ _[⋆]_ be any limit point of _{E_ [(] _[t]_ [)] _}_ and consider a subsequence _E_ [(] _[t][k]_ [)] _→E_ _[⋆]_ . Since _∥E_ [(] _[t][k]_ [+1)] _−_
_E_ [(] _[t][k]_ [)] _∥F_ _→_ 0, we also have _E_ [(] _[t][k]_ [+1)] _→E_ _[⋆]_ . Passing to the limit in
_E_ [(] _[t][k]_ [+1)] = Π _B_            - _E_ [(] _[t][k]_ [)] + _ηtk_ _∇g_ ˜( _E_ [(] _[t][k]_ [)] )� (65)
and using continuity of Π _B_ and _∇g_ ˜ yields the fixed-point relation
_E_ _[⋆]_ = Π _B_              - _E_ _[⋆]_ + _η_ _[⋆]_ _∇g_ ˜( _E_ _[⋆]_ )� _,_ (66)
for some accumulation point _η_ _[⋆]_ _∈_ [ _η,_ ¯ _η_ ]. This fixed-point condition is equivalent to the first-order
optimality (KKT stationarity) condition for the constrained maximization max _E∈B_ ˜ _g_ ( _E_ ), namely
_⟨∇g_ ˜( _E_ _[⋆]_ ) _, Y_ _−E_ _[⋆]_ _⟩≤_ 0 _,_ _∀Y_ _∈_ _B._ (67)
Thus every limit point of _{E_ [(] _[t]_ [)] _}_ is a KKT point of max _E∈B_ ˜ _g_ ( _E_ ), which completes the proof.


18


_∇g_ ˜( _E_ ) =


_N_

- _ωn_ Fold _n_ - _∇hn_ ( **E** [ _n_ ])� _,_ (55)


_n_ =1


where Fold _n_ ( _·_ ) denotes the inverse of the mode- _n_ unfolding. For any _E_ 1 _, E_ 2 _∈_ _B_,


�� _∇g_ ˜( _E_ 1) _−∇g_ ˜( _E_ 2)�� _F_ _[≤]_


=


_≤_


_N_

- _ωn_ ��Fold _n_ - _∇hn_ (( **E** 1)[ _n_ ]) _−∇hn_ (( **E** 2)[ _n_ ])��� _F_ (56)

_n_ =1


_N_

- _ωn_ �� _∇hn_ (( **E** 1)[ _n_ ]) _−∇hn_ (( **E** 2)[ _n_ ])�� _F_ (57)

_n_ =1


_N_

- _ωnLn_ ��( **E** 1)[ _n_ ] _−_ ( **E** 2)[ _n_ ]�� _F_ (58)

_n_ =1


_N_
= - - _ωnLn_ - _∥E_ 1 _−E_ 2 _∥F ._ (59)


_n_ =1


Thus _∇g_ ˜ is Lipschitz on _B_ with constant _L ≤_ [�] _[N]_ _n_ =1 _[ω][n][L][n]_ [.]


D.2 PROOF OF THEOREM 3


_Proof._ By Lemma 5, ˜ _g_ is _L_ -smooth on the compact set _B_ . For any _E, E_ _[′]_ _∈_ _B_, _L_ -smoothness implies
the standard inequality

_g_ ˜( _E_ _[′]_ ) _≥_ _g_ ˜( _E_ ) + _⟨∇g_ ˜( _E_ ) _, E_ _[′]_ _−E⟩−_ _[L]_ _F_ _[.]_ (60)

2 _[∥E]_ _[′][ −E∥]_ [2]


Let _E_ = _E_ [(] _[t]_ [)] and _Z_ [(] _[t]_ [)] = _E_ [(] _[t]_ [)] + _ηt∇g_ ˜( _E_ [(] _[t]_ [)] ). By the optimality condition of the Euclidean projection
onto the convex set _B_, the update _E_ [(] _[t]_ [+1)] = Π _B_ ( _Z_ [(] _[t]_ [)] ) satisfies

             - _Z_ [(] _[t]_ [)] _−E_ [(] _[t]_ [+1)] _, E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)][�] _≥_ 0 _._ (61)

Substituting _Z_ [(] _[t]_ [)] and rearranging yields
_ηt_             - _∇g_ ˜( _E_ [(] _[t]_ [)] ) _, E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)][�] _≥∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[.]_ (62)

Combining with _L_ -smoothness (with _E_ _[′]_ = _E_ [(] _[t]_ [+1)] ) gives


         - 1
_g_ ˜( _E_ [(] _[t]_ [+1)] ) _≥_ _g_ ˜( _E_ [(] _[t]_ [)] ) + _−_ _[L]_
_ηt_ 2


_∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[.]_ (63)


By the step size condition _ηt_ _≤_ 1 _/L_, the coefficient _η_ 1 _t_ _[−]_ _[L]_ 2 [is nonnegative, and thus] _[g]_ [˜][(] _[E]_ [(] _[t]_ [+1)][)] _[≥]_

_g_ ˜( _E_ [(] _[t]_ [)] ) for all _t_ . Since _B_ is compact and ˜ _g_ is continuous, ˜ _g_ is bounded above on _B_, so the monotone
sequence _{g_ ˜( _E_ [(] _[t]_ [)] ) _}_ converges.


Summing the inequality over _t_ = 0 _, . . ., T_ and using _ηt_ _≤_ 1 _/L_ yields


_T_


 -  
[2] _g_ ˜( _E_ [(] _[T]_ [ +1)] ) _−_ _g_ ˜( _E_ [(0)] ) _≤_ [2]

_L_ _L_


 - sup _g_ ˜( _E_ ) _−_ _g_ ˜( _E_ [(0)] )� _< ∞._ (64)
_L_ _E∈B_


- _∥E_ [(] _[t]_ [+1)] _−E_ [(] _[t]_ [)] _∥_ [2] _F_ _[≤]_ [2]

_L_

_t_ =0


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


E PROOF OF THEOREM 4


Let _E_ _[⋆]_ be any limit point of FAG-AdaTR. From Appendix D, every such limit point satisfies the
KKT stationarity condition for the surrogate maximization problem max _E∈B_ ˜ _g_ ( _E_ ):

             - _∇g_ ˜( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [�] _≤_ 0 _,_ _∀Y_ _∈_ _B,_ (68)

where
_B_ := _{E_ : _∥E∥_ [2] _F_ _[≤]_ _[ϵ][}][.]_


For any _Y_ _∈_ _B_, decompose

     - _∇g_ ( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [�] =      - _∇g_ ˜( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [�] +      - _∇g_ ( _E_ _[⋆]_ ) _−∇g_ ˜( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [�] _._ (69)


The first term is nonpositive due to Eq. (68). For the second term, apply the Cauchy–Schwarz
inequality together with the gradient mismatch bound _∥∇g_ ( _E_ ) _−∇g_ ˜( _E_ ) _∥F_ _≤_ _εg_ :
��� _∇g_ ( _E_ _[⋆]_ ) _−∇g_ ˜( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [���] _≤_ _εg_ �� _Y_ _−E_ _⋆_ �� _F_ _[.]_ (70)

Because both _Y_ and _E_ _[⋆]_ lie in the radius- _[√]_ _ϵ_ Frobenius ball _B_, we have
�� _Y_ _−E_ _⋆_ �� _F_ _[≤]_ �� _Y_ �� _F_ [+] �� _E_ _⋆_ �� _F_ _[≤]_ [2] _[√][ϵ.]_
Hence,
��� _∇g_ ( _E_ _[⋆]_ ) _−∇g_ ˜( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [���] _≤_ 2 _[√]_ _ϵ εg._ (71)


Combining the bounds for the two terms yields

           - _∇g_ ( _E_ _[⋆]_ ) _,_ _Y_ _−E_ _[⋆]_ [�] _≥−_ 2 _[√]_ _ϵ εg,_ _∀Y_ _∈_ _B._ (72)

This shows that _E_ _[⋆]_ is an _O_ ( _[√]_ _ϵ εg_ )-approximate stationary point of _g_ ( _E_ ), completing the proof.


F ALGORITHMIC DETAILS


Here we provide the detailed pseudocode for the proposed methods.


**Algorithm 1** Adversarial Attack on TR Decomposition (AdaTR)


1: **Input:** tensor _X_, attack budget _ϵ_, learning rate _η_, outer iterations _T_ out, inner iterations _T_ in, TR
ranks _R_
2: **Output:** adversarial tensor _X_ [ˆ]
3: Initialize perturbation _E_ _∼N_ (0 _,_ 1) and project to _∥E∥_ [2] F _[≤]_ _[ϵ]_
4: **for** _t_ = 1 to _T_ out **do**
5: _E_ old _←E_
6: _X_ ˆ _←X_ + _E_
7: **for** _k_ = 1 to _T_ in **do**
8: Update factors [ _G_ ] via Eq. (8) on _X_ [ˆ]
9: **end for**
10: Compute loss _L_ = _−g_ = _−∥X_ _−_ TR([ _G_ [(] _[T]_ [ )] ( _E_ )]) _∥_ [2] F
11: Update _E_ _←E_ + _η_ _∂_ _[∂g]_ _E_ (backpropagation)

12: Project _E_ to _∥E∥_ [2] F _[≤]_ _[ϵ]_
13: If _∥E_ _−E_ old _∥_ F _/∥E_ old _∥_ F _<_ tol, break
14: **end for**
15: Return _X_ [ˆ] = _X_ + _E_


G COMPLEXITY ANALYSIS


Assuming the TR rank _R_ 1 = _· · ·_ = _RN_ = _R_ and data size _I_ 1 = _· · ·_ = _IN_ = _I_, the time complexity
of one TR-ALS inner iteration is _O_ ( _NI_ _[N]_ _R_ [4] + _NR_ [6] ) (He & Atia, 2023). AdaTR performs _T_ in
times inner iterations, and the backward pass has the same order as the forward computation, so each
outer iteration costs _O_ (2 _T_ in( _NI_ _[N]_ _R_ [4] + _NR_ [6] )). FAG-AdaTR uses the closed-form gradient instead
of back-propagation through TR-ALS, and thus each outer iteration still costs _O_ ( _T_ in( _NI_ _[N]_ _R_ [4] +
_NR_ [6] )), but with a smaller constant factor in practice.


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


**Algorithm 2** Faster Approximate Gradient Adversarial Attack on TR Decomposition (FAG-AdaTR)


1: **Input:** tensor _X_, attack budget _ϵ_, learning rate _η_, outer iterations _T_ out, inner iterations _T_ in, TR
ranks _R_
2: **Output:** adversarial tensor _X_ [ˆ]
3: Initialize perturbation _E_ _∼N_ (0 _,_ 1) and project to _∥E∥_ F _≤_ _ϵ_
4: **for** _t_ = 1 to _T_ out **do**
5: _E_ old _←E_
6: _X_ ˆ _←X_ + _E_
7: **for** _k_ = 1 to _T_ in **do**
8: Update TR factors [ _G_ ] by Eq. (8) on _X_ [ˆ]
9: **end for**
10: Update the gradient of _∇_ **E** [ _n_ ] _hn_ ( **E** [ _n_ ]) by Eq. (16)
11: Update perturbation _E_ by Eq. (17) with [ _G_ ]
12: Project _E_ to _∥E∥_ [2] F _[≤]_ _[ϵ]_
13: If _∥E_ _−E_ old _∥_ F _/∥E_ old _∥_ F _<_ tol, break
14: **end for**
15: Return _X_ [ˆ] = _X_ + _E_


H ADDITIONAL EXPERIMENTAL


H.1 BASELINE METHODS


The attack methods used in this paper are:


    - _Gaussian Noise_ : The budget of Gaussian noise is consistent with the other methods, which
satisfy _∥E∥_ [2] F _[≤]_ _[ϵ]_ [. We add Gaussian noise to the clean tensor] _[ X]_ [to get the adversarial tensor.]

1

    - _ATTR-gen_ : We adopt the ATTR formulation max _∥E∥_ 2F _[≤][ϵ]_ [ min][[] _[G]_ []] 2 _[∥X]_ [ +] _[E −]_ [TR([] _[G]_ [])] _[∥]_ F [2] [with]
the same perturbation budget _ϵ_ as the other baselines.

    - _AdaTR-gen_ : The proposed AdaTR method generates the adversarial tensor under the given
perturbation budget.

    - _FAG-AdaTR-gen_ : The proposed FAG-daTR method generates the adversarial tensor under
the given perturbation budget.


The defense methods used in this paper are:


    - _TR-ALS_ (Zhao et al., 2016): The target model to evaluate various adversarial attack algorithms.

    - _TRPCA-TNN_ (Lu et al., 2019): This method aims to recover the low-rank and sparse tensor
from the original tensor, which might defend against adversarial attacks on the tensor ring
decomposition.

    - _TRNNM_ (Yu et al., 2019): This method completes tensors by enforcing a nuclear norm
under the tensor ring structure, which may help suppress adversarial perturbations.

    - _HQTRC_ (He & Atia, 2022): This method leverages the coarse-to-fine framework to improve the robustness of the tensor ring decomposition.

    - _LRTC-TV_ (Li et al., 2017): This method uses the local smooth and piecewise priors to
improve the recovery accuracy.


H.2 IMPLEMENTATION DETAILS


We provide the detailed parameter settings and implementation environment used in our experiments.


The peak signal-to-noise rate (PSNR), the structural similarity (SSIM) (Wang et al., 2004), and
residual standard error (RSE) are three quality metrics we used for numerical comparison. Besides,
the hyperparameters of comparison algorithms are fine-tuned to the best results according to the


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


suggested range given by the authors in all the following experiments. For all methods, the maximum numbers of both inner and outer iterations were fixed at 100 by default. The TR-rank is fixed
to _R_ 1 = _R_ 2 = _R_ 3 = 5 throughout all experiments; if a different rank is used, it will be explicitly
stated. Moreover, the values of _ϵ_ and _η_ are set to 500 and 0.01, respectively by default. It is worth
noting that all experiments use random sampling obeying a uniform distribution. We implement
these algorithms on a remote server running Ubuntu 20.04 LTS with 256 GB RAM and a single
NVIDIA RTX A5000 GPU (24 GB).


H.3 RUNNING TIME ANALYSIS


To further compare the computational efficiency of AdaTR and FAG-AdaTR, we report their average
running time and variance across color videos and color images under the same size of input tensor,
rank, and inner ALS iterations. The results are summarized in Table 4, showing that FAG-AdaTR
achieves a significant speedup over AdaTR while maintaining comparable attack effectiveness.


Table 4: Comparison of FAG-AdaTR and AdaTR in runtime and peak memory on color videos and
color images. Runtime is reported as mean _±_ variance (seconds).


**Method** **Video Time** **Image Time** **Video Memory** **Image Memory**


FAG-AdaTR 28 _._ 55 _±_ 4 _._ 93 34 _._ 57 _±_ 0 _._ 33 999.82 MB 50.04 MB
AdaTR 44 _._ 78 _±_ 12 _._ 91 53 _._ 62 _±_ 0 _._ 56 8.57 GB 531.39 MB


H.4 EXTENTION TO OTHER TENSOR TECOMPOSITIONS


In this section, we extend our experiments to Tucker-ALS, CP-ALS, and TT-ALS by directly replacing the TR decomposition operator in our asymmetric bilevel objective with the corresponding
multilinear operators. All of these experiments are tested in the 8 color images from the DIV2K
dataset. Due to the Ada-Tucker and Ada-CP requiring more computation resources, which might
cause CUDA to run out of memory, we resize the image to the same small size of 150 _×_ 150 _×_ 3 for
all experiments.


Table 5: Comparison of reconstruction error under clean and attack conditions.


**Method** **Clean Mean** _±_ **Std** **Attack Mean** _±_ **Std**


TR 11.714 _±_ 5.108 **19.209** _±_ **4.080**
TT 11.998 _±_ 4.996 **19.633** _±_ **4.037**
Tucker 12.704 _±_ 5.296 **19.650** _±_ **3.957**
CP 12.310 _±_ 5.223 **17.458** _±_ **3.868**


The results are summarized in Table 5, showing that the proposed attack framework is effective
across different tensor decomposition methods.


H.5 EFFECTIVENESS OF ATTR AS A DEFENSE


To further examine the defensive potential of ATTR, we evaluate its performance under different
perturbation budgets. Table 6 summarizes the results for _ϵ_ = 10 and _ϵ_ = 100 on the image decomposition task.


When the perturbation budget is small (e.g., _ϵ_ = 10), ATTR shows a limited defensive effect. However, as the perturbation budget increases (e.g., _ϵ_ = 100), the defensive effect becomes negligible.
Both reconstruction error (RSE) and perceptual metrics (PSNR/SSIM) deteriorate significantly, and
ATTR fails to prevent the attack from degrading performance.


In summary, while ATTR can provide marginal robustness under small perturbations, it does not
offer effective defense against stronger adversarial attacks.


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


Table 6: Average performance on 8 images under different attacks. Metrics: PSNR ( _↑_ ), RSE ( _↓_ ),
SSIM ( _↑_ ). Best attack (largest RSE, lowest PSNR/SSIM) is highlighted in **bold** .


**ATTR** **TRALS**


**Method** PSNR RSE SSIM PSNR RSE SSIM


_ϵ_ = 10


Clean 22.0015 0.182255 0.566261 21.9238 0.183665 0.563948
Gaussian Noise 22.0017 0.182268 0.566273 21.9238 0.183665 0.563933
FAG-AdaTR-gen 21.9997 0.182327 0.566151 21.9210 0.183741 0.563788
ATTR-gen **21.9929** **0.182361** **0.565766** **21.8366** **0.184679** **0.561431**


_ϵ_ = 100


Clean 22.0006 0.182191 0.565995 21.8897 0.184207 0.562359
Gaussian Noise 21.8999 0.183687 0.546379 21.7429 0.185253 0.541178
FAG-AdaTR-gen 19.0929 0.236054 0.469656 19.0619 0.236838 0.467986
ATTR-gen **18.3931** **0.263973** **0.287658** **18.2523** **0.267261** **0.281135**


Table 7: Performance on MovieLens-100 under _F_ -norm and _L∞_ -norm perturbations with 10% sample ratios. Three evaluation metrics are reported: RMSE, Precision@10 (P@10), and Recall@10
(R@10).


**RMSE** **P@10** **R@10**
**Method**

_F_ _L∞_ _F_ _L∞_ _F_ _L∞_

ATNMF **4.025488** **4.026136** 0.012267 0.011200 0.043085 0.043204
AdaNMF 4.023416 4.025945 **0.007733** **0.005867** **0.026103** **0.023211**


H.6 EVALUATION UNDER _F_ -NORM AND _L∞_ -NORM PERTURBATIONS


In this section, we propose _L∞_ -Norm perturbations to evaluate the performance of AdaTR and FAGAdaTR on the MovieLens-100 dataset, 8 color images, and 7 videos, in addition to the previously
used _F_ -Norm perturbations. The results are summarized in Table 7 and Table 8. Noting that _L∞_ Norm perturbations and _F∞_ -Norm perturbations have the same energy budget.


Overall, these results show that the proposed attack remains the most effective under both _F_ -norm
and _L∞_ -norm constraints. For visual data such as images and videos, _F_ -norm perturbations may
create sharp local artifacts that are perceptible to humans, so the _L∞_ constraint is generally preferable and achieves comparable attack strength without noticeable distortions. In contrast, recommendation models operate on latent tensors that are not directly observed, allowing _F_ -norm attacks to
exploit a larger feasible space and thus yield stronger perturbations. Accordingly, we recommend
using _L∞_ for perceptual data and _F_ -norm for recommendation tasks.


H.7 ADDITIONAL EXPERIMENT ON PRINCIPAL ANGLE MAXIMIZATION ATTACK


In this section, we provide an additional experiment that evaluates our method under a recently–considered adversarial strategy based on _principal_ _angle_ _maximization_ . This attack is inspired by classical subspace-based adversarial analysis, where the adversary seeks a rank-one perturbation ∆ **X** = **ab** _[⊤]_ that maximizes the largest principal angle between the clean feature subspace
and the perturbed feature subspace. Such an attack is related to the method proposed in Li et al.
(2020) and aims at shifting the principal components as much as possible within a constrained perturbation budget.


We follow this principal-angle attack formulation and apply it to our vision reconstruction setting.
Six images are randomly sampled from the evaluation set, and each image is perturbed by the sub

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


Table 8: Performance on Video and Image datasets under clean, _F_ -norm, and _L∞_ -norm perturbations. Best (strongest) attack results are highlighted.


**Video (Reconstruction Error)** **Image (Reconstruction Error)**
**Method**

Clean _F_ Attack _L∞_ Attack Clean _F_ Attack _L∞_ Attack


AdaTR 145.9135 **193.9599** **188.0390** 11.5323 **23.4008** **22.3264**
FAG-AdaTR 145.9135 184.8151 162.4317 11.5323 22.1166 20.4082
ATTR 145.9135 156.9865 156.9865 11.5323 17.0802 17.0802


space attack under the same perturbation budget as in our main experiments. All the TR ranks are
selected as 3.


For each method, we report the reconstruction error. The results are summarized in Table 9.


Table 9: Reconstruction errors under principal angle maximization attack on six randomly selected
images. Lower is better.


Image Clean Gaussian ATTR PCA ~~A~~ ttack FAG-AdaTR AdaTR


1 7.4594 7.5276 7.4819 7.7084 8.4119 **9.2618**
2 7.7996 7.8555 7.9134 7.9134 8.6990 **10.3844**
3 10.3183 10.3592 10.2125 10.4432 11.0245 **12.6237**
4 7.5878 7.6523 7.6144 7.8395 8.5204 **9.1112**
5 16.8553 16.8838 16.8432 16.9065 17.2952 **18.4781**
6 14.0037 14.0385 13.7092 14.0992 14.5313 **15.5617**


**Discussion.** Although the principal-angle maximization attack is effective in shifting the underlying subspace, our methods still significantly outperform it. The reason is that the subspace deviation
induced by principal-angle maximization does _not_ necessarily correspond to a large reconstruction
error. Principal angles measure the worst-case discrepancy between subspaces, but they do not directly control how the corrupted features affect pixel-level reconstruction. In contrast, both AdaTR
and FAG-AdaTR are designed to minimize the _actual reconstruction error_, and thus remain robust
even when the adversary succeeds in enlarging the principal angle.


H.8 ADDITIONAL EXPERIMENTAL RESULTS


We include the complete experimental results of PSNR and RSE in color images decomposition
(Tab. 10 and Tab. 11), visual results in color images decomposition (Fig. 7-14), visual results in
color video decomposition (Fig. 15-21), and visual results in tensor completion (Fig. 22-27) in the
appendix due to the space limitation of the paper.


Table 10: PSNR matrix: mean _±_ variance across runs. Higher is better; **bold** marks the worst (lowest
PSNR) per defense.


Attack _\_ Defense TR-ALS TRPCA-TNN TRNNM HQTRC-Cor HQTRC-Cau HQTRC-Hub LRTC-TV


Clean 20.808 _±_ 18.572 26.349 _±_ 12.237 23.813 _±_ 4.898 22.864 _±_ 24.112 23.242 _±_ 21.633 25.327 _±_ 18.472 23.165 _±_ 9.119
Gauss Noise 19.539 _±_ 8.054 14.689 _±_ 0.302 11.485 _±_ 0.031 15.459 _±_ 0.152 15.856 _±_ 0.234 14.731 _±_ 0.091 17.619 _±_ 0.384
ATTR-gen 17.734 _±_ 11.846 11.315 _±_ 0.016 11.286 _±_ 0.027 15.294 _±_ 0.134 15.420 _±_ 0.122 15.071 _±_ 0.128 16.668 _±_ 0.991
AdaTR-gen **8.547** _±_ **0.388** 9.892 _±_ 0.327 9.877 _±_ 0.061 13.477 _±_ 0.784 13.662 _±_ 1.063 13.195 _±_ 0.342 15.902 _±_ 0.550
FAG-AdaTR-gen 8.800 _±_ 0.106 **9.314** _±_ **0.026** **9.516** _±_ **0.016** **11.357** _±_ **1.576** **11.492** _±_ **1.712** **11.125** _±_ **0.914** **11.592** _±_ **1.087**


23


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


Figure 7: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 1.


Figure 8: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 2.


Figure 9: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 3.


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


Figure 10: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 4.


Figure 11: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 5.


Figure 12: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 6.


25


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


Figure 13: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 7.


Figure 14: Visual results on tensor decomposition tasks under different attacks and defenses for
Image 8.


Figure 15: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 1.


26


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


Figure 16: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 2.


Figure 17: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 3.


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


Figure 18: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 4.


Figure 19: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 5.


28


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


Figure 20: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 6.


Figure 21: Visual results on tensor decomposition tasks under different attacks and defenses for the
5th frame of Video 7.


29


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


Figure 22: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 1 with SR=20%.


Figure 23: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 2 with SR=20%.


30


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


Figure 24: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 3 with SR=20%.


Figure 25: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 4 with SR=20%.


31


**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


Figure 26: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 5 with SR=20%.


Figure 27: Additional visual results on tensor completion tasks under different attacks and defenses
for Image 6 with SR=20%.


32


**1728**

**1729**


**1730**

**1731**

**1732**

**1733**

**1734**

**1735**


**1736**

**1737**

**1738**

**1739**

**1740**

**1741**


**1742**

**1743**

**1744**

**1745**

**1746**


**1747**

**1748**

**1749**

**1750**

**1751**

**1752**


**1753**

**1754**

**1755**

**1756**

**1757**

**1758**


**1759**

**1760**

**1761**

**1762**

**1763**

**1764**


**1765**

**1766**

**1767**

**1768**

**1769**


**1770**

**1771**

**1772**

**1773**

**1774**

**1775**


**1776**

**1777**

**1778**

**1779**

**1780**

**1781**


Table 11: SSIM matrix: mean _±_ variance across runs. Higher is better; **bold** marks the worst (lowest
SSIM) per defense (column).


Attack _\_ Defense TR-ALS TRPCA-TNN TRNNM HQTRC-Cor HQTRC-Cau HQTRC-Hub LRTC-TV


Clean 0.641 _±_ 0.025 0.896 _±_ 0.000 0.839 _±_ 0.001 0.825 _±_ 0.005 0.831 _±_ 0.004 0.863 _±_ 0.001 0.826 _±_ 0.001
Gauss Noise 0.572 _±_ 0.010 0.524 _±_ 0.014 0.484 _±_ 0.028 0.552 _±_ 0.014 0.557 _±_ 0.012 0.547 _±_ 0.019 0.574 _±_ 0.023
ATTR-gen 0.507 _±_ 0.012 0.453 _±_ 0.024 0.485 _±_ 0.025 0.596 _±_ 0.025 0.598 _±_ 0.025 0.593 _±_ 0.025 0.567 _±_ 0.021
AdaTR-gen **0.122** _±_ **0.001** 0.409 _±_ 0.022 0.409 _±_ 0.023 0.492 _±_ 0.018 0.489 _±_ 0.017 0.498 _±_ 0.019 0.515 _±_ 0.021
FAG-AdaTR-gen 0.220 _±_ 0.004 **0.286** _±_ **0.008** **0.286** _±_ **0.009** **0.335** _±_ **0.005** **0.340** _±_ **0.005** **0.340** _±_ **0.006** **0.336** _±_ **0.006**


33