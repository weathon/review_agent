# THE EFFECT OF ATTENTION HEAD COUNT ON TRANS## FORMER APPROXIMATION


**Penghao Yu**
Department of Mathematics
National University of Singapore
penghaoyu@u.nus.edu


**Zeyu Bao**
Department of Mathematics
National University of Singapore
zeyu@u.nus.edu


**Qianxiao Li**
Department of Mathematics
Institute for Functional Intelligent Materials
National University of Singapore
qianxiao@nus.edu.sg


**Haotian Jiang**
Institute for Functional Intelligent Materials
National University of Singapore
haotian@nus.edu.sg


**Ruoxi Yu**
Center for Data Science
Peking University
yuruoxi@stu.pku.edu.cn


ABSTRACT


Transformer has become the dominant architecture for sequence modeling, yet a
detailed understanding of how its structural parameters influence expressive power
remains limited. In this work, we study the approximation properties of transformers, with particular emphasis on the role of the number of attention heads. Our
analysis begins with the introduction of a generalized _D_ -retrieval task, which we
prove to be dense in the space of continuous functions, thereby providing the basis
for our theoretical framework. We then establish both upper and lower bounds on
the parameter complexity required for _ϵ_ -approximation. Specifically, we show that
transformers with sufficiently many heads admit efficient approximation, whereas
with too few heads, the number of parameters must scale at least as _O_ (1 _/ϵ_ _[cT]_ ), for
some constant _c_ and sequence length _T_ . To the best of our knowledge, this constitutes the first rigorous lower bound of this type in a nonlinear and practically
relevant setting. We further examine the single-head case and demonstrate that
an embedding dimension of order _O_ ( _T_ ) allows complete memorization of the input, where approximation is entirely achieved by the feed-forward block. Finally,
we validate our theoretical findings with experiments on both synthetic data and
real-world tasks, illustrating the practical relevance of our results.


1 INTRODUCTION


The transformer architecture (Vaswani et al., 2017) has become the foundation of modern sequence
modeling, driving progress in natural language processing (Devlin et al., 2019; Brown et al., 2020),
computer vision (Dosovitskiy et al., 2021), and multi-modal learning (Radford et al., 2021). Its
ability to scale has enabled breakthroughs such as BERT, GPT, and ViT, making it the dominant
paradigm across domains. Despite this remarkable empirical success, the theoretical principles underlying transformer expressivity remain incomplete. In particular, while universal approximation
results establish that transformers can approximate arbitrary sequence-to-sequence mappings (Yun
et al., 2020a; P´erez et al., 2021), much less is known about how their structural hyperparameters
influence approximation efficiency.


Among transformer hyperparameters, the number of attention heads plays a central role. In practice,
large models often adopt head counts such as 32, 64, or 128 (e.g., Devlin et al. (2019); Dosovitskiy
et al. (2021); Touvron et al. (2023); Jiang et al. (2023); Grattafiori et al. (2024) see Table 10 for


1


more ), yet this choice is largely heuristic: there is no principled understanding of how many heads
are needed for a given task, nor of the costs incurred when the head count is insufficient. Theoretical progress on this question has so far been limited. Most existing results focus on upper bounds,
showing that transformers with sufficiently many heads or with extremely large embedding dimension in the single-head case can achieve universal approximation or good approximation rate, but
offering little insight into the limitations that arise when the head count is insufficient. Moreover,
many analyses rely on strong simplifications—such as restricting to linear embeddings, isolating the
attention block, or linearizing the architecture. While these assumptions make the problem more
tractable, they severely restrict the model’s expressive power and prevent the derivation of rigorous
lower bounds in realistic nonlinear settings.


In this work, we address this gap by analyzing single-layer transformers on sequence-to-vector tasks.
To this end, we introduce a new function class, the _generalized D-retrieval tasks_, which we design
as a structured but expressive family motivated by retrieval problems. Each coordinate is defined
by _z_ ¯ _i_ ( _XT_ ) = min _t∈Si fi_ ( _x_ ( _t_ )) _, i_ = 1 _, . . ., D_ for subsets _Si_ _⊆_ [ _T_ ], and the overall target takes
the form _H_ ( _XT_ ) = _F_ 0(¯ _z_ 1( _XT_ ) _, . . .,_ ¯ _zD_ ( _XT_ )). By construction, this class extends retrieval-style
problems while being dense in the space of continuous sequence-to-vector mappings, ensuring that
results obtained in this setting reflect general approximation behavior.


The central challenge arises when the number of heads _h_ is smaller than the intrinsic dimension
_D_ of the target. In this case, multiple coordinates _zi_ ( _XT_ ) must be represented by the same head,
creating an information bottleneck: the attention layer maps distinct sequences to nearly indistinguishable representations, forcing the feed-forward network to perform the separation. We show that
overcoming this bottleneck requires parameter growth exponential in the sequence length _T_, namely
_O_ (1 _/ϵ_ _[cT]_ ) parameters for _ϵ_ -accuracy, thus establishing the first rigorous lower bounds for transformers in nonlinear settings. In contrast, when _h_ _≥_ _D_, heads can specialize to distinct coordinates _zi_,
eliminating the bottleneck and enabling efficient approximation.


Our results advance the theoretical understanding of attention by showing, that insufficient head
count provably limits expressivity in realistic regimes. Experiments on both synthetic tasks and
real-world retrieval data confirm that the predicted scaling laws persist in practice.


**Contributions.** Our main contributions are as follows:


First, we establish the first rigorous lower bounds for transformers in nonlinear settings, showing
that when _h < D_, parameter complexity grows exponentially with sequence length.


Second, we provide constructive upper bounds, proving that _h ≥_ _D_ enables efficient approximation
with parameter growth independent of sequence length _T_ .


Third, in the memorization regime, single-head transformers with embedding dimension _n_ _≥_ _Td_
approximate by memorizing sequences, with the complexity residing in the feed-forward block.


2 RELATED WORK


Several works have studied the approximation and expressivity properties of transformers. The universal approximation property was first established in Yun et al. (2020a), and later extended to transformers with sparse attention matrices in Yun et al. (2020b). The approximation rate of single-layer
transformers with one head was analyzed in Jiang & Li (2024). Amsel et al. (2024) investigated
how the rank of the attention matrix influences expressivity for a specific nearest-neighbor target
can be constructed. They showed that when the rank is insufficient, the number of heads required
for approximation grows exponentially, independent of sequence length. In a related direction,
Bhojanapalli et al. (2020) argued that setting the rank of the attention matrix equal to the sequence
length enhances expressivity. Beyond finite-dimensional settings, Takakura & Suzuki (2023) considered sequences of infinite dimension, characterizing approximation rates in terms of target function
smoothness. Similarly, Wang et al. (2024) studied special classes of target functions and demonstrated that approximation error scales polynomially with the number of heads. In addition to these
approximation-theoretic results, several works have investigated broader notions of expressivity.
Dehghani et al. (2019); P´erez et al. (2021) established the Turing completeness of transformers,
and Giannou et al. (2023) showed that transformers can represent arbitrary computer programs in


2


a looped setting. Finally, Mahdavi et al. (2023) examined memorization capacity, proving that the
number of samples that can be stored scales linearly with the number of heads.


3 PRELIMINARIES


**Input and Output** We consider the input space


_XT_ =           - _x_ ( _s_ ) _∈_ [0 _,_ 1] _[d]_ : _s ∈_ [ _T_ ]           - _,_ (1)


where [ _T_ ] = _{_ 1 _, . . ., T_ _}_ . We call _T_ the length of the input sequence. The output is a single vector
_y_ _∈_ R _[l]_, where _l_ is independent of _T_ and specified by the task.


For example, in a text retrieval task one may take _d_ to be the max number of tokens per candidate,
_T_ the number of candidates, and _l_ the size of the output representation.


**Input Representation.** Each token is mapped to an _E_ -dimensional vector by a trainable encoder


_Pϕ_ : [0 _,_ 1] _[d]_ _×_ [ _T_ ] _→_ R _[E]_ _,_ ( _**x**_ _, s_ ) _�→_ _Pϕ_ ( _**x**_ _, s_ ) _,_


which jointly incorporates the content _**x**_ and its position _s_ . Given _XT_ = _{x_ ( _s_ ) _}_ _[T]_ _s_ =1 _[∈X][T]_ [,] [the]
embedded sequence is


_X_ ˆ _T_ = _{_ ˆ _x_ ( _s_ ) = _Pϕ_ ( _x_ ( _s_ ) _, s_ ) _∈_ R _[E]_ : _s ∈_ [ _T_ ] _}._ (2)


This formulation subsumes common designs where _Pϕ_ combines a content embedding with either
learned or deterministic positional encoding.


For example, if Emb( _x_ ) is a content embedding map and _p_ ( _t_ ) a positional code, then common
schemes correspond to


_Pϕ_ ( _x_ ( _t_ ) _, t_ ) = Emb( _x_ ( _t_ )) + _p_ ( _t_ ) (additive PE) _,_


or
_Pϕ_ ( _x_ ( _t_ ) _, t_ ) = �Emb( _x_ ( _t_ )) _,_ _p_ ( _t_ )� (concatenated PE) _._


Following common practice, we append a trainable _classification_ _token_ _c_ ˆ0 _∈_ R _[E]_ to the sequence.
The final input to the transformer is


_X_ ˆ [ _T_ ] = _{x_ ˆ(1) _, . . .,_ ˆ _x_ ( _T_ ) _,_ ˆ _c_ 0 _} ∈_ R _[E][×]_ [(] _[T]_ [ +1)] _,_ (3)


and the output _y_ ˆ is taken from the ( _T_ +1)-th position corresponding to ˆ _c_ 0.


**Transformer Hypothesis Class** With the input space and embedding defined, we then formulate
the transformer hypothesis space.


We consider a single-layer transformer based on the standard architecture (Vaswani et al., 2017),
with two modifications introduced for analytical simplicity. Firstly, we omit layer normalization
to simplify the analysis, while acknowledging its practical importance, and we conjecture that our
key lower bound (Theorem 2 (2)) still holds when layer normalization is present. Secondly, we
exclude residual connections outside the feed-forward network. In the single-layer sequence-tovector setting, where the output is read from a designated classification token, the residual branch
can be merged into the feed-forward transformation by reparameterization, thus these likewise do
not alter the expressive power of the architecture.


For an _h_ -head, single-layer transformer, let _n_ denote the embedding dimension _per head_ and _E_ =
_nh_ the total embedding dimension. The output is

_y_ ˆ = _H_ [ˆ] ( _X_ [ˆ] [ _T_ ]) = _F_ [ˆ]  - _c_ ˆ0 + _WO_ Concat _[h]_ _i_ =1�� _[T]_ _σ_ �( _WQ,ic_ ˆ0) _[⊤]_ _WK,ix_ ˆ( _t_ )� _WV,ix_ ˆ( _t_ )�� _,_ (4)

_t_ =1


where for each head _i_, _WQ,i, WK,i_ _∈_ R _[n][×][E]_ are the query/key projection matrices, _WV,i_ _∈_ R _[n][×][E]_

is the value projection, _WO_ _∈_ R _[E][×][E]_ is the output projection applied to the concatenated heads, and


3


_F_ ˆ : R _[E]_ _→_ R _[l]_ is a feed-forward network which we call it the _feed-forward block_ . The softmax with
scaling factor _β_ is defined by

exp� _β ρ_ ( _t_ )�
_σ_ [ _ρ_ ]( _t_ ) =          - _Tt_ _[′]_ =1 [exp]          - _β ρ_ ( _t_ _[′]_ )� _,_ _β_ _>_ 0 _._ (5)

and _β_ _>_ 0 may be chosen arbitrarily large in order to make the softmax attention mechanism
approximate a hardmax.


We denote this family by
_H_ ( _h, n, d, T, M_ ) _,_ (6)
the class of single-layer transformers with _h_ heads, per-head embedding dimension _n_, input dimension _d_, sequence length _T_, and parameter count _M_ . Each _H_ _∈H_ ( _h, n, d, T, M_ ) is a mapping
_H_ : R _[d][×][T]_ _→_ R _[l]_ _,_
implemented by the encoder _Pϕ_ : [0 _,_ 1] _[d]_ _×_ [ _T_ ] _→_ R _[nh]_, concatenation of the classification token
_c_ ˆ0, a multi-head attention layer with projections _{WQ,i, WK,i, WV,i}_ _[h]_ _i_ =1 _[, W][O]_ [,] [and] [a] [feed-forward]
network _F_ [ˆ] : R _[nh]_ _→_ R _[l]_ . Thus _H_ has the form equation 4, with parameter count _k_ referring to the
weights and biases in FFNs ( _Pϕ,_ _F_ [ˆ] ).


**Approximation Problem** With the hypothesis class specified, we now formalize the approximation problem, which provides the framework for analyzing the expressive power of transformers.
**Definition 1** ( _ϵ_ -approximation) **.** _Let XT_ _⊂_ R _[d][×][T]_ _be a compact domain, and let F_ : _XT_ _→_ R _[l]_ _be a_
_target function._ _We say that the hypothesis class H_ ( _h, n, d, T, M_ ) _ϵ-approximates F_ _on XT_ _if there_
_exists_ _H_ [ˆ] _∈H_ ( _h, n, d, T, M_ ) _such that_

sup _∥H_ [ˆ] ( _XT_ ) _−_ _F_ ( _XT_ ) _∥∞_ _< ϵ._
_XT ∈XT_


4 GENERALIZED _D_ -RETRIEVAL TASKS


**Target functions.** To motivate our construction, consider a simple one-dimensional example:
_XT_ = _{ XT_ = ( _x_ (1) _, . . ., x_ ( _T_ )) : _x_ ( _t_ ) _∈_ [0 _,_ 1] _},_
with target
_H_ ( _XT_ ) = max [+] min (7)
1 _≤t≤T_ _[x]_ [(] _[t]_ [)] 1 _≤t≤T_ _[x]_ [(] _[t]_ [)] _[.]_

This task requires the model to extract two distinct features from the sequence—the maximum and
the minimum—before combining them. It can thus be viewed as a retrieval problem with two
independent features being aggregated.


This example illustrates the broader idea behind our target class: retrieval-style problems where multiple salient features must be identified and combined. We now formalize this intuition by defining
the family of _generalized D-retrieval tasks_ .


**Mathematical Formulation** Formally, for each _i_ = 1 _, . . ., D_, let _fi_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] be _C_ [2] and
define
_z_ ¯ _i_ ( _XT_ ) = min _Si_ _⊆_ [ _T_ ] _,_ _|Si| ≥_ [1] (8)
_t∈Si_ _[f][i]_ [(] _[x]_ [(] _[t]_ [))] _[,]_ 4 _[T,]_

so that _z_ ¯( _XT_ ) = (¯ _z_ 1( _XT_ ) _, . . .,_ ¯ _zD_ ( _XT_ )) _∈_ [0 _,_ 1] _[D]_ . The target is then
_H_ ( _XT_ ) = _F_ 0� _z_ ¯( _XT_ )� _,_ (9)

where _F_ 0 : [0 _,_ 1] _[D]_ _→_ R is _C_ [1] . For vector-valued targets _H_ : [0 _,_ 1] _[d][×][T]_ _→_ R _[l]_ defined with the same
functions _fi_, subsets _Si_, and an outer map _F_ 0 : [0 _,_ 1] _[D]_ _→_ R _[l]_, the extension is applied coordinatewise, since each coordinate function of _F_ 0 can be approximated independently. Therefore, it suffices
to consider the scalar-valued case. We denote by _FD_ _[d,T]_ the class of all such functions _H_ .


Related sparse sequence-to-sequence retrieval tasks, such as the _q_ -sparse token regression (qSTR)
model of (Mousavi-Hosseini et al., 2025), where each output position depends on at most _q_ relevant
input tokens, can be viewed as sequence-to-sequence analogues of our formulation. Their results on
the sample complexity of single-layer Transformers with at least _q_ heads are complementary to our
approximation-theoretic guarantees in the generalized _D_ -retrieval setting.


4


**Assumptions** **on** **the** **target** **class** For the theoretical analysis to be tractable we impose the following conditions:
**Assumption 1** (Model constraints) **.** The model constraints are as follows:
(1.1) The embedding _Pϕ_ satisfies
_∥Pϕ_ ( _x_ ( _s_ ) _, s_ ) _∥_ 2 _≤_ 1 _,_ _∀_ _s ∈_ [ _T_ ] _,_ _XT_ _∈XT,_
ensuring embedded inputs remain uniformly bounded.
(1.2) The post-attention mapping _F_ [ˆ] is a two-layer feed-forward network with 1-Lipschitz activation,
hence a universal approximator on compact domains.
(1.3) All weights in _F_ [ˆ] and entries of the attention matrices _{WQ,i, WK,i, WV,i}, WO_ are bounded
in magnitude by 1, ensuring stability of the model.
**Assumption 2** (Target class constraint) **.** The target functions defined in equation 9 satisfy the following:
(2.1) Each _fi_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] attains its unique global minimum _zi_ at a point _x_ [(] _[i]_ [)] _∈_ [0 _,_ 1] _[d]_ .
(2.2) The minimizers _{x_ [(] _[i]_ [)] _}_ _[D]_ _i_ =1 [are pairwise distinct.]
(2.3) The Hessian _∇_ [2] _x_ _[f][i]_ [(] _[x]_ [(] _[i]_ [)][)][ is positive definite for all] _[ i]_ [ = 1] _[, . . ., D]_ [.]
(2.4) The gradient _∇zF_ 0( _z_ 1 _, . . ., zD_ ) has all coordinates strictly nonzero.
_Remark._ Assumption 2 excludes only degenerate cases while preserving broad generality for both
the functions _fi_ and the outer map _F_ 0. More specifically: Assumptions (2.1) and (2.4) ensure that
each _fi_ behaves regularly around its minimizer. A degenerate example excluded by these assumptions is _fi_ ( _x_ ) _≡_ _c_ 0 for constant _c_ 0, which is totally independent of the input; Assumption (2.2)
requires distinct minimizers, which allows partitioning the space into _D_ disjoint basins around each
minimizer _x_ [(] _[i]_ [)] . A degenerate example excluded by this assumption is _f_ 1 = _f_ 2 = _· · ·_ = _fD_ ; Assumption (2.3) enforces sensitivity of the target to small perturbations near the minimizers, ruling
out trivial flat cases (such as _F_ 0 _≡_ _c_ 0 for constant _c_ 0) where no meaningful separation can be made.


Having introduced the generalized _D_ -retrieval tasks, it remains to ask whether this family is sufficiently expressive. To address this, we now establish the _universality of the target class_ : the family
is dense in the space of continuous sequence-to-vector mappings.
**Theorem** **1** (Density of the target class) **.** _For_ _fixed_ _d, T_ _,_ _the_ _family_ _{FD}_ _[∞]_ _D_ =1 _[is]_ _[dense]_ _[in]_ _[C]_ [(] _[X][T]_ [ )] _[.]_
_That is, for every F_ _∈_ _C_ ( _XT_ ) _and every ϵ >_ 0 _, there exists D and f_ _∈FD_ _such that_
max
_X∈XT_ _[|][F]_ [(] _[X]_ [)] _[ −]_ _[f]_ [(] _[X]_ [)] _[| ≤]_ _[ϵ.]_


The proof is deferred to Appendix A.1. This density property highlights that our specially designed
target family is not overly restrictive; rather, it forms a sufficiently general class to capture arbitrary
continuous sequence-to-vector mappings. Beyond density, we show that _D_ is indeed the _intrinsic_
_dimension_ of this target, which means that it is the unique _D_ _≪_ _T_ for which the target _H_ can be
represented in the generalized _D_ -retrieval task form.
**Corollary 1** (Uniqueness of intrinsic dimension) **.** _If task H_ _can be represented by_ ( _{fi, Si}_ _[D]_ _i_ =1 [1] _[, F]_ [0][)]
_and_ ( _{f_ [˜] _i,_ _S_ [˜] _i}i_ _[D]_ =1 [2] _[,]_ _[F]_ [˜][0][)] _[, satisfying Assumption 1 and 2 with][ D]_ 1 [2] [+] _[ D]_ 2 [2] _[≤]_ 501 _[T]_ _[, then][ D]_ [1] [=] _[ D]_ [2] _[.]_


This corollary justifies that _D_ comes from the intrinsic property of the target and is invariant across
its form of representation. The proof is deferred to Appendix A.2


5 APPROXIMATION RATE OF GENERALIZED _D_ -RETRIEVAL TASKS


Theorem 1 establishes that the generalized _D_ -retrieval tasks form a dense family in the space of
continuous sequence-to-vector functions. The next step is to analyze the efficiency with which transformers approximate these functions. To this end, we begin by stating two standard approximation
assumptions regarding how well the fundamental building blocks of the target can be approximated.
**Assumption 3** (Approximation of components) **.** We assume the following approximation properties
hold.
(A1) There exist constants _C_ 1 _>_ 0 and _γ_ _>_ 0 such that for every _δ_ _>_ 0, the function _F_ 0 : [0 _,_ 1] _[D]_ _→_
R can be _δ_ -approximated by a two-layer feed-forward network Φ _δ_ of width at most _C_ 1 _/δ_ _[γ]_, i.e.,

sup
_x∈_ [0 _,_ 1] _[D][ |][F]_ [0][(] _[x]_ [)] _[ −]_ [Φ] _[δ]_ [(] _[x]_ [)] _[| ≤]_ _[δ.]_


5


(A2) There exist constants _C_ 2 _>_ 0 and _γ_ _>_ 0 (possibly different from (A1)) such that for each
_i_ = 1 _, . . ., D_ and every _δ_ _>_ 0, the function _fi_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] can be _δ_ -approximated by a
two-layer feed-forward network Ψ _i,δ_ of width at most _C_ 2 _/δ_ _[γ]_, i.e.,
sup
_x∈_ [0 _,_ 1] _[d][ |][f][i]_ [(] _[x]_ [)] _[ −]_ [Ψ] _[i,δ]_ [(] _[x]_ [)] _[| ≤]_ _[δ.]_


These assumptions are reasonable: by the classical result of (Cybenko, 1989), two-layer networks
can approximate continuous functions on compact domains. In particular, if the Barron norm is
finite, one may take _γ_ = 2 (Barron, 1993); even in the worst case, setting _γ_ = max( _d, D_ ) yields
approximation rates comparable to uniform grid discretizations, which still suffices for our analysis.


We now present our main theoretical result. It establishes upper and lower bounds on the approximation rates of transformers within the generalized _D_ -retrieval framework. In particular, the lower
bound in part (2) provides the first rigorous evidence that insufficient head count _h_ _<_ _D_ leads to
exponential parameter complexity, revealing a fundamental expressivity bottleneck.
**Theorem** **2** (Approximation rates of transformers) **.** _Fix_ _d, T_ _._ _Under_ _Assumption_ _3,_ _the_ _following_
_hold for any target H_ _∈FD_ _[d,T]_ _[:]_


_(1)_ _**Sufficient**_ _**expressivity**_ _**with**_ _D_ _**heads.**_ _For_ _h_ = _D_ _and_ _embedding_ _dimension_ _n_ = 2 _per_
_head,_ _there_ _exists_ _a_ _constant_ _Cd,D,T_ _>_ 0 _such_ _that_ _∀M_ _>_ _[C][d,D,T]_ _ϵ_ _[γ]_ _._ _the_ _hypothesis_ _class_

_H_ ( _h, n, d, T, M_ ) _ϵ-approximates H._


_(2)_ _**Lower bound with**_ _s < D_ _**heads.**_ _For h_ = _s < D, define_


[1]
_k_ = [(] 4


_−_ 1 _,_
( _n_ + 1) _s_ + 1


4 _[T]_ _[−]_ _[s][ −]_ _[D]_ [ + 1)]


_then_
min� _M_ : _H_ ( _h, n, d, T, M_ ) _ϵ-approximates H_         - = Ω� _ϵ_ 1 _[k]_         - _._


_(3)_ _**Single-head large embedding dimension.**_ _For h_ = 1 _and per-head embedding dimension_
_n_ _≥_ _Td,_ _if_ _the_ _feed-forward_ _block_ _is_ _a_ 5 _-layer_ _ReLU_ _neural_ _network,_ _then_ _there_ _exists_ _a_
_constant Cd,D,T_ _>_ 0 _such that for all M_ _>_ _[C]_ _ϵ_ _[d,D,T]_ [1+] _[γ]_ _[, the hypothesis class][ H]_ [(] _[h, n, d, T, M]_ [)]

_can ϵ-approximate H._

_Remark._ We clarify the precise role of the assumptions and constants in Theorem 2.
(1)Theorems 2 (1) and 2 (3) require only Assumption 3, while Theorem 2 (2) relies only on Assumptions 1 and 2.
(2) The constant in Theorem 2 can be made explicit as _Cd,D,T_ = _Cd,D_ ( _rT_ ) _[−][α][d,D][T]_,where _r_ _>_ 0
is determined by the local behavior of the functions _fi_ around _x_ [(] _[i]_ [)] and of _F_ 0, and _αd,D_ depends
only on _d_ and _D_ . This form is valid in the regime _d, D_ _≪_ _T_ _≪_ 1 _/ϵ_ .
(3) The exponent coefficients in Theorems 2 (1) and 2 (3) differ because, in Theorem 2 (3), the
network _F_ [ˆ] also needs Ω( _T/ϵ_ ) parameters to approximate the “max-like” operation. This yields a
bound of the form _M_ _≤_ _ϵ_ [max(1] 1 _[,γ]_ [)] [, and for notational simplicity we write] _[ M]_ _[≤]_ [1] _[/ϵ]_ [1+] _[γ]_ [.]
We provide the detailed proof in Appendix A.2. We also justify the tightness of Theorem 2 (2) in
Appendix A.2.2.


Theorem 2 highlights how approximation efficiency depends on head count: enough heads allow
specialization, too few force inefficient compression, and a single large head can rely on memorization. To illustrate these cases concretely, we now revisit the toy example from equation 7 and discuss
how each part of the theorem works in that setting.


**Case (1):** _h ≥_ _D_ **heads.** Theorem 2 (1) shows that when the number of heads matches the intrinsic
dimension _D_ of the target, the transformer can allocate one head per component feature, allowing
each head to specialize and leaving the feed-forward block to aggregate their outputs. This yields
efficient approximation with _O_ ( _M_ _[−]_ [1] _[/γ]_ ) error for parameter count _M_, independent of sequence
length _T_ .


In the toy example with _D_ = 2, one head naturally tracks the maximum and the other the minimum,
so the task is solved directly without incurring inefficiency. This illustrates how having “enough
heads” removes the unfavorable scaling in _T_ and explains the practical advantage of multiple heads
beyond universal approximation results (e.g., Kajitsuka & Sato (2023)).


6


**Case (2):** _h < D_ **heads.** Theorem 2 (2) establishes that when the number of heads is smaller than
the intrinsic dimension _D_, the parameter count required to achieve a given accuracy can grow exponentially in the sequence length _T_ . This lower bound highlights why insufficient heads lead to severe
inefficiency. Intuitively, each head can be viewed as specializing in one coordinate of the minima
structure in equation 9. When _h < D_, a single head must encode multiple roles simultaneously.


In the toy example with _D_ = 2, one head is forced to capture both the maximum and the minimum
across all _T_ positions. Since softmax attention only produces weighted averages, the head must
effectively encode information from multiple sequence elements simultaneously. As _T_ increases,
the number of relevant elements to distinguish grows linearly with _T_, yet they are compressed into
an _n_ -dimensional vector whose size does not scale with _T_ . The feed-forward block must then
disentangle these increasingly entangled representations, which requires parameters exponential in
_T_ . This explains why the parameter requirement scales as Ω(1 _/ϵ_ _[cT]_ ) and why the scaling improves
dramatically once _h ≥_ _D_ .


Theorem 2 (2) is proved with the following idea: (i) each head selects its most responsive locations
( _yj, tj_ ) in _D_ disjoint minima basins around _x_ [(] _[i]_ [)] ; (ii) because _s_ _<_ _D_, there is at least one segment
_Gi_ _⊂_ _B_ ( _x_ [(] _[i]_ [)] _, r_ ) that no head focuses on. We then consider the segment _Gi_ in it; (iii) along this
segment (suppose it is _G_ 1), we construct many candidate subsequences and, by a pigeonhole argument, obtain two subsequences _Z_ 1 _, Z_ 2 whose post-attention representations are almost identical
but whose contribution to _f_ 1 differs; (iv) these subsequences are then embedded into full sequences
_W_ 1 _, W_ 2, which the target function separates by at least 3 _ϵ_, while the attention block maps them
within _O_ ( _ϵ_ _[k]_ [+1] ), forcing a large feed-forward network.


The intuition by which we derive the large network is different from geometric arguments in existing
works such as (Yehudai & Shamir, 2019). We directly made use of the fact that the network must be
either large or have large parameters to approximate a function with large Lipschitz norm.


**Case** **(3):** **single** **head** **with** **large** **embedding.** Theorem 2 (3) shows that when the embedding
dimension scales with the sequence length, _E_ = _n ≥_ _Td_, the model can encode the entire sequence
into the classification token ˆ _c_ 0. Concretely, each input can be embedded as _et ⊗_ _x_ ( _t_ ) _∈_ R _[T d]_, where
_et_ is the _t_ -th standard basis vector, so that trivial attention aggregates to


1
_T_ [(] _[x]_ [(1)] _[, . . ., x]_ [(] _[T]_ [))] _[ ∈]_ [[0] _[,]_ [ 1]] _[T][,]_


which preserves the full sequence. The feed-forward block _F_ [ˆ] can then recover the target relation
efficiently. Unlike memorization of training data (Mahdavi et al., 2023), this mechanism can generalize since it captures the relation itself. Moreover, approximating extrema functions such as max
and min with a shallow ReLU network is straightforward (see Lemma 9), requiring width _O_ ( _T/ϵ_ )
for _ϵ_ accuracy. However, this regime is impractical, as it demands embedding dimensions that grow
linearly with _T_ .


As deeper transformer are more commonly used in practice, here we conjecture the extension of
Theorem 2 (2) onto the _L_ -layer case.


**Conjecture** **1** (Multilayer transformer case) **.** A necessary condition for efficient approximation is
_L · h_ _≥_ _D_ . When the head number is insufficient across layers, we conjecture the lower-bound
scaling for some constants _aL, bL_ _>_ 0 depending only on depth _L_ to be

log(ParamCount) = Ω� _|_ log _ϵ| ·_ _[a][L]_ _nh_ _[ T]_ _[bL]_            - _,_


Experiments on the synthetic dataset in Section 6.1 with 2-layer transformer with no positional
encoding and no layer norm has also verified this transition at _D_ = _L · h_ . (See Table 5 in Appendix)


6 EXPERIMENTS


Theorem 2 provides theoretical insights into how the approximation ability of transformers depends
on the number of heads. In this section, we illustrate these insights empirically. We begin with synthetic tasks that mirror the structure of the generalized _D_ -retrieval tasks, and then turn to real datasets
(MS MARCO and CIFAR-10) to examine whether similar scaling behaviors arise in practice.


7


6.1 NUMERICAL VERIFICATION OF THEOREM 2 WITH SYNTHETIC DATASET


We design a synthetic task aligned with the target class analyzed in Theorem 2. Given a sequence
_X_ = _{x_ (1) _, . . ., x_ ( _T_ ) _}_ of length _T_ with _x_ ( _t_ ) _∈_ R [4], the output is


_y_ =


4

- max _i_ _[x]_ [(] _[t]_ [)] _[,]_

1 _≤t≤T_ _[a][⊤]_
_i_ =1


where _a_ 1 _, . . ., a_ 4 _∈_ R [4] are fixed. Inputs are sampled i.i.d. from _x_ ( _t_ ) _∼N_ (0 _, I_ 4). For _T_ _∈_
_{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ we generate 8000 training and 2000 validation examples.


On this task, we evaluate single-layer transformers with head numbers _h_ _∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _}_ and fixed
per-head embedding dimension. Each _x_ ( _t_ ) is embedded via a two-layer ReLU MLP and concatenated with a trainable classification token _c_ 0, after which a single-layer multi-head attention block
(without residuals or normalization) processes the sequence. A two-layer GeLU MLP then outputs
the scalar prediction. Both MLPs have the same hidden dimension _N_ .


Then for each ( _h, T_ ), models are trained under multiple random seeds. We report the _minimal_
_normalized_ _mean_ _squared_ _error_ _(NMSE)_ across seeds to reduce optimization noise and highlight
expressivity. NMSE, equivalent to 1 _−_ _R_ [2], corrects for the variance shrinkage of maxima as _T_
grows, thus enabling fair comparison across lengths. Further training details and explanations are
given in Appendix B.1.1.


0.05

0.04

0.03

0.02


0.01


2e-4

0.00


|Col1|Col2|Col3|S|equence|Length|(T)|
|---|---|---|---|---|---|---|
|||||<br><br>|<br>T=8<br>||
||||||T=16<br>||
||||||~~=32~~<br>T=64<br>~~=128~~||
||||||||
||||||||
||||||||


Number of Heads


4.0


3.5


3.0


2.5


T
T=8
T=16
T=32
T=64
T=128


|Col1|Col2|H = 3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|3.8<br>3.6<br>3.4<br>3|3.8<br>3.6<br>3.4<br>3|3.8<br>3.6<br>3.4<br>3|3.8<br>3.6<br>3.4<br>3|


Log Min Validation NMSE


4.0


3.5


3.0


2.5


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|6.8<br>6.6<br>6.4<br>6.2<br>6.0<br>|


Log Min Validation NMSE


(a) **NMSE vs. Number of Heads** _h_ **.**


(b) **Log** _N_ **vs.** **Log Accuracy (NMSE)**


Figure 1: Results on the synthetic example. (a) NMSE vs. number of heads _h_ for sequence lengths
_T_ _∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_, hidden dimension fixed at _N_ = 32. Note that there is a transition at _h_ = 4.
(A table of mean and variance values corresponding to these curves is provided in Table 2.) (b) Log
Hidden Dimension _N_ vs. Log Accuracy for different sequence lengths _T_ . The parameter count _k_
for the MLPs change linearly with _N_ . (Plots for _h_ = 1 and _h_ = 2 are in Figure 4.)


Figure 1a shows minimal validation NMSE versus head number _h_ across sequence lengths _T_ . Performance improves monotonically with _h_ and exhibits a clear transition near the intrinsic dimension
_D_ = 4. For _h_ _<_ _D_, NMSE grows with _T_, as limited heads must encode multiple extrema and
the FFN becomes inefficient. Once _h_ _≥_ _D_, curves flatten across _T_, indicating that heads specialize to different coordinates and the FFN aggregates them very effectively. Normalization by NMSE
ensures comparability across _T_, despite the increasing concentration of the max-of-Gaussians target.


Figure 1b highlights a phase transition between _h_ = 3 and _h_ = 4, with _h_ = _D_ = 4 equal to
the intrinsic dimension of the target. When _h_ _≤_ 3, the negative log NMSE scales approximately
linearly with the log parameter count (proportional to the MLP hidden dimension _N_ ), in agreement
with Theorem 2 (2). Moreover, for a fixed parameter count, larger _T_ yields higher NMSE (worse
approximation). Equivalently, as indicated by the fitted scaling lines, achieving the same error requires larger parameter counts when _T_ increases, in line with Theorem 2 (2). In contrast, for _h_ = 4
these trends change qualitatively. Validation error reaches the order of 10 _[−]_ [6], indicating near-perfect
generalization, yet the slope with respect to parameter count reverses: larger MLPs yield slightly
higher validation NMSE, a signature of tiny overfitting. The dependence on _T_ also changes in this
regime; see Remark B.1.1 in Appendix for details.


We also conducted experiments on synthetic data with the widely used scheme of fixing _E_ = _nh_ =
32 constant(For _h_ = 3 _,_ 5, we choose per-head embedding dimension to be _⌈_ 32 _/h⌉_, and total embedding dimension becomes _E_ = 33 _,_ 35. See Table 3 in Appendix for details), as well as experiments


8


on synthetic datasets with _D_ = 3 (See Table 4 in Appendix for details). Both of the above experiments demonstrate similar trends to the _D_ = 4 experiments, implying the robustness of our
results.


6.2 EXPERIMENTS ON REAL DATASETS


We conduct two additional experiments on real datasets to assess the practical relevance of our theoretical findings. The first is a text retrieval task based on MS MARCO, and the second is an image
classification task based on CIFAR-10. As there is no natural NMSE-like metric on retrieval tasks
and accuracy is most widely used, we focus on training accuracy to isolate architectural expressivity
from issues related to optimization or data scarcity. For completeness, we also report test accuracy
for both experiments in Table 7 and 9 in Appendix. The experiments examine whether the phase
transition around the intrinsic dimension _D_, predicted by Theorem 2, also manifests in practice.


**MS MARCO (text retrieval).** We construct retrieval-style datasets from the MS MARCO passage
ranking collection (Bajaj et al., 2016), where each query is paired with one positive passage and _T −_ 1
mined hard negatives ( _T_ _∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ ). We train a two-layer transformer encoder with per-head
embedding dimension fixed at 32, varying the number of heads across _{_ 1 _,_ 2 _,_ 4 _,_ 6 _,_ 8 _,_ 10 _,_ 12 _,_ 14 _,_ 16 _}_ .
Input text is tokenized using the BERT tokenizer, and word, positional, and segment embeddings
from pretrained BERT (Devlin et al., 2019) are kept frozen. These 768-dimensional embeddings are
linearly projected to the embedding size _E_ = heads _×_ 32, after which only the projection and transformer layers are trained. Full dataset construction and training details are given in Appendix B.2.


**CIFAR-10** **(image** **classification).** We further evaluate on the CIFAR-10 dataset (Krizhevsky,
2009) using a four-layer Vision transformer (ViT) (Dosovitskiy et al., 2021). Each image of size
32 _×_ 32 is divided into non-overlapping 8 _×_ 8 patches (patch size = 8), which are linearly embedded. The transformer encoder has per-head embedding dimension fixed at 16, and we vary the
number of heads across _h_ _∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 10 _,_ 11 _,_ 12 _,_ 13 _,_ 14 _,_ 16 _,_ 20 _,_ 24 _}_ . To vary the sequence length,
we extend the border with interpolation around each image to enlarge its side length, after which the

sequence length is given by - image side lengthpatch size �2 +1, including the classification token. Figure 6 shows

some of the examples. Full dataset preprocessing and training details are provided in Appendix B.3.


**Result analysis.** Both experiments exhibit the same qualitative trend as in the synthetic setting.
Figure 2a shows that in the text retrieval experiment, when _h_ _<_ 12, accuracy declines as the sequence length _T_ increases, consistent with Theorem 2 (2). Once _h_ _>_ 12, this dependence on
_T_ disappears, and performance remains stable. Taking _Err_ ( _h, T_ ) = 1 _−_ Accuracy( _h, T_ ) as error, by using _cT_ _[β]_ exp( _αh/T_ _[δ]_ ) to approximate ( _Err_ ( _h, T_ )) in log scale under MAE and drop-outs
( _h_ = 1 _,_ 12 _,_ 14 _,_ 16 are dropped out as outliers, _δ_ = 0 _._ 25 _>_ 0 _, α_ = _−_ 1 _._ 40 _<_ 0), figure 2b illustrates
that when _h_ _<_ 12, _−_ log( _Err_ ( _h, T_ )) _∝_ _h/T_ _[δ]_, highly consistent with the order in Theorem 2 (2)
under fixed parameter count _M_ . The flattening of curves after _h >_ 12 is also consistent with theory.


Figure 2c shows similar trend in image classification, with intrinsic dimension at _h_ = 10. Figures 2d
and 2e illustrates weighted reversal score, calculated by _R_ ( _h_ ) = _w_ 1 _h_ - _T_ 1 _<T_ 2 [max((] _[err]_ [(] _[T]_ [1][)] _[−]_

_err_ ( _T_ 2)) _,_ 0) with normalization factor _wh_ = max _T_ _err_ ( _T_ ) _−_ min _T_ _err_ ( _T_ ), detects the existence of
longer _T_ yielding smaller error for this head number _h_ . Such phenomenon leads to positive _R_ ( _h_ ),
and it also indicates phase transition as explained in remark B.1.1. Figure 2e further verified this.


7 CONCLUSION


In this work we investigated the approximation properties of single-layer transformers. We first
introduced a structured target family, the generalized _D_ -retrieval task, that is broad enough to
capture general sequence-to-vector mappings (Theorem 1). Within this setting, we analyzed
how the approximation efficiency of transformers depends on architectural choices, especially
the number of head. Our results indicate that having a sufficient number of heads leads to
efficient approximation, while an insufficient number of heads forces the parameter count to
grow exponentially with sequence length _T_ . We also examined the single-head case, where large
embedding dimension allows sequence memorization but shifts the complexity to the feed-forward


9


100%


10 [0]

10 1

10 2

10 3

10 4

10 5

10 6


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
|||||||||||
||~~Se~~|~~q Len~~<br>T|~~ (T)~~<br>8|||||||
|||T=<br>|16<br>|||||||
|||||||||||
|||~~T~~<br>T=|~~32~~<br>64|||||||
|||||||||||


1 2 4 6 8 10 12 14 16
Number of Heads


80%


60%


40%


20%


0%


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|||||||||||
||||||S|S||||
||||||S|S|eq Le<br>|n (T)<br>||
||||||||~~T~~<br>T<br>~~T~~|~~=8~~<br>=16<br>~~=32~~||
||||||||T|=64||


Number of Heads


(a) **Accuracy vs. Number of Heads for different**
_T_ **(Text Retrieval).**

100


(b) **Log(1-Accuracy)** **and** **its** **prediction** **(Text**
**Retrieval)**


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||||||~~S~~|~~eq Le~~<br>T=<br>T=<br>|~~ n (T)~~<br>65<br>145<br>|
||||||||||
||||||||||
||||||||~~T~~<br>T=<br>T=|~~257~~<br>577<br>1025|
||||||||||


Number of Heads


Number of Heads


80


60


40


2 4 8 10121416 20 24
Number of Heads


4


2


0


4


3


2


1


0


(c) **Train accuracy (** % **) vs. Number**
**of Heads (Image).**


(d) **Weighted** **Reversal** **score** **vs.**
**Head Number (Image)**


(e) **Weighted** **Reversal** **score** **vs.**
**Head Number (Synthetic)**


Figure 2: **Experiments** **on** **real** **datasets.** Training performance with different numbers of heads
_h_ across different sequence lengths _T_ . (a) Accuracy vs. number of heads for different _T_ in text
retrieval; phase transition near _h_ = 12. Mean and standard deviation see Table 6. MRR shows a
similar trend, see Fig. 5 in the appendix. (b) Phase transition for text retrieval. (c) Accuracy vs. number of heads for different _T_ in image classification; phase transition near _h_ = 10. Mean and standard
deviation see Table 8. (d) Weighted Reversal Score for Image Classification, _err_ = 1 _−_ _Accuracy_ .
The plot becomes positive when _h_ _≥_ 10, indicating phase transition. (e) Weighted Reversal Score
for Synthetic Experiment, it becomes positive at _h_ = 4, exactly the intrinsic dimension of the task.


block (Theorem 2). These findings clarify the roles played by head count in transformer expressivity.


Our experiments on both synthetic and real datasets reveal a non-trivial phase transition around the
intrinsic dimension _D_, consistent with theoretical analysis. When the number of heads is below
_D_, models exhibit higher error for the same parameter count as sequence length _T_ increases. Once
the head count reaches or exceeds _D_, approximation rate becomes independent of sequence lengths
_T_ . This transition is also observed in real-world datasets with deeper architectures, indicating that
the notion of intrinsic dimension is not only theoretical but also practically relevant. In particular,
beyond fully training models, analyzing head contributions early in training to estimate how many
heads meaningfully affect the output, or training multiple models with varying depths and head
counts while tracking how error scales with _T_ are potential ways to probe the task’s intrinsic dimension. These experiments might demonstrate whether the inferred intrinsic dimension is stable across
architectures, thereby informing head-count selection and the head number to retain under pruning.


**Limitations.** We conclude by noting several limitations of this study. Firstly, although the analyzed target class is dense, the phenomena of interest are most naturally manifested in retrieval-style
tasks aligned with our setting. Secondly, our analysis is restricted to single-layer transformers;
while experiments on real datasets supports Conjecture 1 in deeper architectures, a rigorous multilayer theory remains open. Finally, the tradeoff between sequence memorization and pattern learning—observed for shorter sequences (cf. Remark B.1.1)—has not yet been established rigorously
and warrants further investigation.


10


ACKNOWLEDGMENTS


This research is supported by the National Research Foundation, Singapore, under the NRF fellowship (project No. NRF-NRFF13-2021-0005). The computational work for this article was fully
performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg).


REFERENCES


Noah Amsel, Gilad Yehudai, and Joan Bruna. Quality over Quantity in Attention Layers: When
Adding More Heads Hurts. In _The Thirteenth International Conference on Learning Representa-_
_tions_, October 2024.


Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. Ms marco: A human generated
machine reading comprehension dataset. _arXiv preprint arXiv:1611.09268_, 2016.


A.R. Barron. Universal approximation bounds for superpositions of a sigmoidal function. _IEEE_
_Transactions on Information Theory_, 39(3):930–945, May 1993. ISSN 1557-9654. doi: 10.1109/
18.256500.


Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. Lowrank bottleneck in multi-head attention models. In _International conference on machine learning_,
pp. 864–873. PMLR, 2020.


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _Advances in neural information processing systems_, 33:1877–1901, 2020.


G. Cybenko. Approximation by superpositions of a sigmoidal function. _Mathematics_ _of_ _Con-_
_trol,_ _Signals_ _and_ _Systems_, 2(4):303–314, December 1989. ISSN 1435-568X. doi: 10.1007/
BF02551274.


Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal
Transformers. In _arXiv:1807.03819 [Cs, Stat]_, March 2019.


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding. _arXiv:1810.04805 [cs]_, May 2019.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at
Scale. In _International Conference on Learning Representations_, 2021.


Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris
Papailiopoulos. Looped Transformers as Programmable Computers. In _Proceedings of the 40th_
_International Conference on Machine Learning_, pp. 11398–11442. PMLR, July 2023.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd
of models. _arXiv preprint arXiv:2407.21783_, 2024.


Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier,
L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas
Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b. _arXiv preprint arXiv:2310.06825_,
2023.


Haotian Jiang and Qianxiao Li. Approximation Rate of the Transformer Architecture for Sequence
Modeling. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_,
2024.


Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billion-scale similarity search with gpus. _IEEE_
_Transactions on Big Data_, 7(3):535–547, 2019.


11


Tokio Kajitsuka and Issei Sato. Are transformers with one layer self-attention using low-rank weight
matrices universal approximators? _arXiv preprint arXiv:2307.14023_, 2023.


A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. In _Technical_ _Report_ _0_,
2009.


Sadegh Mahdavi, Renjie Liao, and Christos Thrampoulidis. Memorization capacity of multi-head
attention in transformers. _arXiv preprint arXiv:2306.02010_, 2023.


Alireza Mousavi-Hosseini, Clayton Sanford, Denny Wu, and Murat A Erdogdu. When do transformers outperform feedforward and recurrent networks? a statistical perspective. _arXiv preprint_
_arXiv:2503.11272_, 2025.


Jorge P´erez, Pablo Barcel´o, and Javier Marinkovic. Attention is Turing-Complete. _Journal_ _of_
_Machine Learning Research_, 22(75):1–35, 2021. ISSN 1533-7928.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In _International conference on machine learning_, pp.
8748–8763. PmLR, 2021.


Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. _Found._ _Trends_ _Inf._ _Retr._, 3(4):333–389, April 2009. ISSN 1554-0669. doi: 10.1561/
1500000019.


Shokichi Takakura and Taiji Suzuki. Approximation and estimation ability of transformers for
sequence-to-sequence functions with infinite dimensional input. In _International Conference on_
_Machine Learning_, pp. 33416–33447. PMLR, 2023.


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances_ _in_ _neural_ _informa-_
_tion processing systems_, 30, 2017.


Mingze Wang et al. Understanding the expressive power and mechanisms of transformer for sequence modeling. _Advances in Neural Information Processing Systems_, 37:25781–25856, 2024.


Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding
neural networks. _Advances in neural information processing systems_, 32, 2019.


Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are
Transformers universal approximators of sequence-to-sequence functions? In _arXiv:1912.10077_

_[Cs, Stat]_, February 2020a.


Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers. In _Advances in Neural Information Processing Systems_, volume 33, pp. 13783–13794.
Curran Associates, Inc., 2020b.


12


A PROOFS OF MAIN THEOREMS


A.1 PROOF OF THEOREM 1


**Proof** **Sketch.** The proof proceeds in three steps. First, by Lemma 1, we approximate a broader
function class that relaxes the smoothness requirements and the assumptions in Assumption 2. Second, Lemmas 2 and 3 show that by constructing appropriate _Si_, we can faithfully recover all information from the original input sequence with simple _fi_ . Finally, the outer function _F_ 0 can be
applied to approximate an arbitrary sequence-to-vector target within this class. Together, these steps
establish the result.


_Proof of Theorem 1._ To prove Theorem 1, we first establish a few auxiliary lemmas.

**Lemma 1** (Relaxed target class and closure equivalence) **.** _Let_ _F_ [�] _D_ _[d,T]_ _be defined as in_ _9 but with only_
_fi_ _∈_ _C_ ([0 _,_ 1] _[d]_ _,_ [0 _,_ 1]) _, i.e., we drop ”unique minimizer”, ”pairwise distinct”and ”PD Hessian”.Set_
_F_ _[d,T]_ := [�] _D≥_ 1 _[F]_ _D_ _[d,T]_ _and_ _F_ [�] _[d,T]_ := [�] _D≥_ 1 _[F]_ [�] _D_ _[d,T]_ _[.]_ _[Then]_

_[∥·∥][∞]_ _∥·∥∞_
_F_ _[d,T]_ = _F_ [�] _[d,T]_ _on XT ._


_Proof._ Fix _H_ [�] ( _XT_ ) = _F_ [�] 0� _z_ �1( _XT_ ) _, . . .,_ - _zD_ ( _XT_ )� _∈_ _F_ [�] _D_ _[d,T]_ with _z_ - _i_ ( _XT_ ) = min _t∈Si_ _f_ [�] _i_ ( _x_ ( _t_ )) and
_f_ - _i_ _∈_ _C_ ([0 _,_ 1] _[d]_ _,_ [0 _,_ 1]). Let _ε_ _>_ 0. We will construct _H_ _∈F_ _[d,T]_ with _∥H_ _−_ _H_ - _∥∞_ _≤_ _ε_ . Firstly, by
Stone–Weierstrass, choose _pi_ _∈_ _C_ _[∞]_ ([0 _,_ 1] _[d]_ ) so that _∥pi_ _−_ _f_ [�] _i∥∞_ _≤_ _η_, where _η_ _>_ 0 will be fixed
later. Because the uniform approximation can slightly leave [0 _,_ 1], compose with a smooth strictly
increasing squashing _s_ : [ _−c,_ 1 + _c_ ] _→_ [0 _,_ 1] with _s_ ( _u_ ) = _u_ on [0 _,_ 1] and _∥s ◦_ _pi −_ _pi∥∞_ _≤_ _η_ (for
small enough _c >_ 0), and replace _pi_ by _s ◦_ _pi_ . We still write _pi_ and retain _∥pi −_ _f_ [�] _i∥∞_ _≤_ 2 _η_ .

Secondly, let _ξi_ _∈_ arg min _x∈_ [0 _,_ 1] _d pi_ ( _x_ ) (nonempty by compactness). Pick _r_ _∈_ (0 _,_ [1] 4 [)][ small and a]

_C_ _[∞]_ bump _ϕi_ supported in _B_ ( _ξi,_ 2 _r_ ) _∩_ [0 _,_ 1] _[d]_, with _ϕi_ ( _ξi_ ) = 1, _∇ϕi_ ( _ξi_ ) = 0, and with _∇_ [2] _ϕi_ ( _ξi_ )
_negative definite_ . [1] Define, for parameters _δi,_ 1 _, δi,_ 2 _>_ 0 to be fixed,


_gi_ ( _x_ ) := _pi_ ( _x_ ) _−_ _δi,_ 1 _ϕi_ ( _x_ ) + _δi,_ 2 _ϕi_ ( _x_ ) _∥x −_ _ξi∥_ [2] _._


(i) Since _gi_ ( _ξi_ ) = _pi_ ( _ξi_ ) _−_ _δi,_ 1 while _gi_ ( _x_ ) _≥_ _pi_ ( _x_ ) whenever _ϕi_ ( _x_ ) = 0 and _gi_ ( _x_ ) _>_ _pi_ ( _x_ ) for
_x ∈_ _B_ ( _ξi,_ 2 _r_ ) _\ {ξi}_, we get that _ξi_ _is the unique global minimizer_ of _gi_ .


(ii) At _ξi_, because _∇ϕi_ ( _ξi_ ) = 0,


_∇_ [2] _gi_ ( _ξi_ ) = _∇_ [2] _pi_ ( _ξi_ ) _−_ _δi,_ 1 _∇_ [2] _ϕi_ ( _ξi_ ) + 2 _δi,_ 2 _I._


Here _−∇_ [2] _ϕi_ ( _ξi_ ) _≻_ 0, so choosing ( _δi,_ 1 _, δi,_ 2) suitably makes _∇_ [2] _gi_ ( _ξi_ ) _≻_ 0 (PD). Because _∥ϕi∥∞_ _≤_
1 and _∥∥x −_ _ξi∥_ [2] _∥∞_ _≤_ _d_ on [0 _,_ 1] _[d]_,


_∥gi −_ _pi∥∞_ _≤_ _δi,_ 1 + _δi,_ 2 (2 _r_ ) [2] _._


Hence, by taking _r_ small and then _δi,_ 1 _, δi,_ 2 small (using the _r_ _[−]_ [2] scaling in _∇_ [2] _ϕi_ ( _ξi_ ) to keep the
Hessian PD), we can ensure both PD at _ξi_ and _∥gi −_ _pi∥∞_ _≤_ _η_ .


Thirdly, we use tiny translation to remove distinctiveness. It may happen that _ξi_ = _ξi′_ for some _i ̸_ =
_i_ _[′]_ . Choose pairwise distinct small vectors _vi_ _∈_ R _[d]_ and fix a smooth cutoff _χ_ _∈_ _C_ _[∞]_ ([0 _,_ 1] _[d]_ _,_ [0 _,_ 1])
that equals 1 on [ _r,_ 1 _−_ _r_ ] _[d]_ and vanishes near the boundary. Define a _C_ _[∞]_ diffeomorphism of the
cube,
Φ _i_ ( _x_ ) := _x_ _−_ _εi χ_ ( _x_ ) _vi,_ with _εi_ _>_ 0 small _._

Then Φ _i_ is arbitrarily close to the identity in _C_ [1] for small _εi_, maps [0 _,_ 1] _[d]_ to itself, and _hi_ := _gi ◦_ Φ _i_
has a (unique) minimizer at _x_ [(] _[i]_ [)] := Φ _[−]_ _i_ [1][(] _[ξ][i]_ [)][.] [For different] _[ i]_ [, these points are distinct if the] _[ v][i]_ [’s are]
distinct and _εi_ ’s are small but nonzero. Moreover, because _∇gi_ ( _ξi_ ) = 0, the Hessian at _x_ [(] _[i]_ [)] satisfies


_∇_ [2] _hi_ ( _x_ [(] _[i]_ [)] ) = _D_ Φ _i_ ( _x_ [(] _[i]_ [)] ) _[⊤]_ _∇_ [2] _gi_ ( _ξi_ ) _D_ Φ _i_ ( _x_ [(] _[i]_ [)] ) _≻_ 0 _,_


1For instance take _ϕi_ ( _x_ ) = _ψ_ ( _∥x −_ _ξi∥_ 2 _/r_ 2) with _ψ_ (0) = 1, _ψ′_ _<_ 0 near 0, _ψ_ _≡_ 0 on [1 _, ∞_ ); then
_∇_ [2] _ϕi_ ( _ξi_ ) _≺_ 0 and its norm scales like _r_ _[−]_ [2] .


13


_D≥_ 1 _[F]_ _D_ _[d,T]_ _and_ _F_ [�] _[d,T]_ := [�]


_D≥_ 1 _D_ _[.]_ _[Then]_

_[F]_ [�] _[d,T]_


so PD is preserved. Since _gi_ is Lipschitz on the compact cube, _∥hi_ _−_ _gi∥∞_ _≤_ _Li εi_ for some _Li_,
hence by taking _εi_ small we get _∥hi −_ _gi∥∞_ _≤_ _η_ .


Finally, if needed, compose with the same strictly increasing squashing _s_ as in Step 1 and set


_fi_ := _s ◦_ _hi_ _∈_ _C_ [2] ([0 _,_ 1] _[d]_ _,_ [0 _,_ 1]) _._


Because _s_ is strictly increasing, it preserves the minimizer location and, at the minimizer _x_ [(] _[i]_ [)], _∇_ [2] ( _s◦_
_hi_ )( _x_ [(] _[i]_ [)] ) = _s_ _[′]_ [�] _hi_ ( _x_ [(] _[i]_ [)] )� _∇_ [2] _hi_ ( _x_ [(] _[i]_ [)] ) _≻_ 0. Also _∥fi −_ _hi∥∞_ _≤_ _η_ by construction.


Collecting the bounds from previous deduction:


_∥fi −_ _f_ [�] _i∥∞_ _≤∥pi −_ _f_ [�] _i∥∞_

        - ��        _≤_ 2 _η_


+ _∥gi −_ _pi∥∞_

 - ��  _≤η_


+ _∥hi −_ _gi∥∞_

 - ��  _≤η_


+ _∥fi −_ _hi∥∞_

 - ��  _≤η_


_≤_ 5 _η._


For each _i_, the map _u_ _�→_ min _t∈Si ut_ is 1-Lipschitz in _∥· ∥∞_ . Hence the corresponding features
_z_ ¯ _i_ ( _XT_ ) := min _t∈Si fi_ ( _x_ ( _t_ )) and � _zi_ ( _XT_ ) := min _t∈Si_ _f_ [�] _i_ ( _x_ ( _t_ )) satisfy _∥z_ ¯ _i −_ _z_ - _i∥∞_ _≤_ 5 _η_ . Let _ωF_ �0 [be]

a modulus of continuity of _F_ [�] 0 on [0 _,_ 1] _[D]_ . Choose _η_ so small that _ωF_ �0 [(5] _[η]_ [)] _[ ≤]_ _[ε/]_ [2][.] [Then]
��� _F_ 0� _z_ ¯( _XT_ )� _−_ _F_ [�] 0� _z_ �( _XT_ )��� _∞_ _[≤]_ _[ε/]_ [2] _[.]_

Finally, approximate _F_ [�] 0 uniformly on [0 _,_ 1] _[D]_ by some _F_ 0 _∈_ _C_ [1] ([0 _,_ 1] _[D]_ ) within _ε/_ 2 (Stone–
Weierstrass). Setting


_H_ ( _XT_ ) := _F_ 0� _z_ ¯1( _XT_ ) _, . . .,_ ¯ _zD_ ( _XT_ )� _∈F_ _[d,T]_ _,_


we obtain
_∥H_ _−_ _H_ [�] _∥∞_ _≤∥F_ 0(¯ _z_ ) _−_ _F_ [�] 0(¯ _z_ ) _∥∞_

               - ��                _≤ε/_ 2


+ _∥F_ [�] 0( _z_ ) _−_ _F_ [�] 0( _z_ ) _∥∞_

      
 - ��  _≤ε/_ 2


_≤_ _ε._


This shows _F_ _[d,T]_ _⊂F_ _[d,T]_ _[∥·∥][∞]_ . The reverse inclusion is simple, hence we have the lemma.

[�]


_Remark._ Thus, we now focus on the relaxed class _F_ _[d,T]_ and Lemma 1 lifts the result to the original

[�]
class _F_ _[d,T]_ .


**Lemma 2** (Order-statistic in the relaxed class) **.** _Without loss of generation, suppose_ 4 _|T_ _._ _Let m_ =
_T_ 4 _[.]_ _[For each][ j]_ _[∈]_ [[] _[d]_ []] _[ and][ X][T]_ [=] _[ {][x]_ [(1)] _[, . . ., x]_ [(] _[T]_ [)] _[}][, define]_

_Uj_ ( _XT_ ) := max min
_B⊆_ [ _T_ ] _u∈B_ _[x]_ [(] _[u]_ [)] _[j][,]_
_|B|_ = _m_


_and for each fixed t ∈_ [ _T_ ] _,_


_Yt,j_ ( _XT_ ) := max min _Zt,j_ ( _XT_ ) := max min
_A⊆_ [ _T_ ] _u∈A_ _[x]_ [(] _[u]_ [)] _[j][,]_ _A⊆_ [ _T_ ] _u∈A_ [(1] _[ −]_ _[x]_ [(] _[u]_ [)] _[j]_ [)] _[.]_
_|A|_ = _m,_ _t∈A_ _|A|_ = _m,_ _t∈A_


_Let_ _v_ 1 _,j_ _≥· · ·_ _≥_ _vT,j_ _be_ _the_ _sorted_ _values_ _of_ _{x_ (1) _j, . . ., x_ ( _T_ ) _j}_ _and_ _set_ _Uj_ = _vm,j._ _For_ _the_
_multi-set_ _{x_ ( _u_ ) _j_ : _u_ _∈_ [ _T_ ] _},_ _let_ _v_ 1 _,j_ _≥· · ·_ _≥_ _vT,j_ _(nonincreasing)_ _and_ _w_ 1 _,j_ _≤· · ·_ _≤_ _wT,j_
_(nondecreasing)._ _Then we have_


_Yt,j_ = min _{x_ ( _t_ ) _j,_ _vm,j},_ 1 _−_ _Zt,j_ = max _{x_ ( _t_ ) _j,_ _wm,j}._


_In particular:_


_x_ ( _t_ ) _j_ _≤_ _vm,j_ _⇒_ _Yt,j_ = _x_ ( _t_ ) _j,_ _x_ ( _t_ ) _j_ _≥_ _wm,j_ _⇒_ 1 _−_ _Zt,j_ = _x_ ( _t_ ) _j._


_Proof._ Among all _m_ -subsets _B_, the maximum of min _u∈B x_ ( _u_ ) _j_ is attained by picking the _m_ largest
coordinates, so _Uj_ = _vm,j_ . Forcing _t_ _∈_ _A_, the choice of the other _m −_ 1 indices to maximize the
minimum is the _m_ _−_ 1 largest among _{x_ ( _u_ ) _j_ : _u_ = _t}_, hence _Yt,j_ = min _{x_ ( _t_ ) _j, vm,j}_ . For
_Zt,j_, note that min _u∈A_ (1 _−_ _x_ ( _u_ ) _j_ ) = 1 _−_ max _u∈A x_ ( _u_ ) _j_, so maximizing it over _A_ is the same
as minimizing max _u∈A x_ ( _u_ ) _j_, which picks _t_ plus the ( _m_ _−_ 1) smallest leading to 1 _−_ _Zt,j_ . The
particular statements follow immediately.


14


**Lemma 3** (Smooth selector) **.** _For q_ _>_ 0 _define_

[)][2][ (1] _[ −]_ _[Z][t,j]_ [) +] _[ e][q]_ [(] _[ Y][t,j]_ _[−][v][m,j]_ [)][2] _[ Y][t,j]_
_x_       - _q_ ( _t_ ) _j_ := _[e][q]_ [( 1] _[−][Z][t,j]_ _e_ _[q][−]_ [( 1] _[w][−][m,j][Z][t,j]_ _[−][w][m,j]_ [)][2] + _e_ _[q]_ [(] _[ Y][t,j]_ _[−][v][m,j]_ [)][2] _._


_Then_ - _xq_ ( _t_ ) _j_ _→_ _x_ ( _t_ ) _j_ _uniformly on XT_ _as q_ _→∞._


_Proof._ From Lemma 2, if _x_ ( _t_ ) _j_ _≥_ _Uj_, we have _Yt,j_ = _vm,j_ and ( _Yt,j −vm,j_ ) [2] = 0 while 1 _−Zt,j_ =
_x_ ( _t_ ) _j_ and (1 _−_ _Zt,j −_ _wm,j_ ) [2] _>_ 0. Thus, as _q_ _→∞_, the weight concentrates on (1 _−_ _Zt,j_ ) = _x_ ( _t_ ) _j_ .
If _x_ ( _t_ ) _j_ _≤_ _Uj_ and _wm,j_ _≥_ _x_ ( _t_ ) _j_, we have (1 _−_ _Zt,j_ _−_ _wm,j_ ) [2] = 0 while _Yt,j_ = _x_ ( _t_ ) _j_ and
( _Yt,j_ _−_ _vm,j_ ) [2] _>_ 0 concentrating on _Yt,j_ = _x_ ( _t_ ) _j_ . If _wm,j_ _≤_ _x_ ( _t_ ) _j_ _≤_ _Uj_, we have 1 _−_ _Zt,j_ =
_Yt,j_ = _x_ ( _t_ ) _j_, so either of three settings leads to _x_ ( _t_ ) _j_ . The compactness of _XT_ gives us the uniform
property.


Now, we begin our formal proof for Theorem 1.


By Lemma 1, for any _F_ _∈_ _C_ ( _XT_ ) and _ε >_ 0, it suffices to construct an _H_ [�] _∈_ _F_ [�] _[d,T]_ with _∥F_ _−H_ [�] _∥∞_ _≤_
_ε/_ 2, since the lemma 1 could lift it to _F_ _[d,T]_ with another _ε/_ 2.


Fix _m_ = _[T]_ 4 [.] [For each coordinate] _[ j]_ [and each] _[ S]_ _[⊆]_ [[] _[T]_ []][ with] _[ |][S][|]_ [ =] _[ m]_ [, include the relaxed primitives]


_z_ [(] _[j,S]_ [)] ( _XT_ ) := min _z_ ¯ [(] _[j,S]_ [)] ( _XT_ ) := min
_t∈S_ _[x]_ [(] _[t]_ [)] _[j][,]_ _t∈S_ [(1] _[ −]_ _[x]_ [(] _[t]_ [)] _[j]_ [)] _[.]_


To form the thresholds _vm,j_ and _wm,j_ needed in Lemma 2, additionally include:


min for all _S_ _⊆_ [ _T_ ] _\ {t}_ with _|S|_ = _m,_
_t∈S_ _[x]_ [(] _[t]_ [)] _[j]_


and
min for all _S_ _⊆_ [ _T_ ] _\ {t}_ with _|S|_ = _T_ _−_ _m,_
_t∈S_ [(1] _[ −]_ _[x]_ [(] _[t]_ [)] _[j]_ [)]


Using smooth log-sum-exp (softmax) in the outer function _F_ [�] 0, we can recover the subset-wise maximum required to compute _Uj_, _Yt,j_, _Zt,j_, _vm,j_ and _wm,j_ from these primitives.


By Lemma 3, for any _δ_ _>_ 0 there exists _q_ such that


max max
_XT ∈XT_ _t∈_ [ _T_ ] _, j∈_ [ _d_ ]


��� _xq_ ( _t_ ) _j_ _−_ _x_ ( _t_ ) _j_ �� _≤_ _δ._


By uniform continuity of _F_ on the compact _XT_, choose _δ_ so that this implies _|F_ ( _XT_ ) _−_
_F_ ( _X_ [�] _q_ ( _T_ )) _| ≤_ _ε/_ 4 for all _XT_, where _X_ [�] _q_ ( _T_ ) stacks the coordinates � _xq_ ( _t_ ) _j_ . We approximate the continuous map _u_ _�→_ _F_ ( _u_ ) on [0 _,_ 1] _[dT]_ uniformly by a polynomial _P_ within _ε/_ 4 (Stone–Weierstrass).
Define
_H_         - ( _XT_ ) :=         - _P_ _◦_ vec�( _X_ [�] _q_ ( _T_ )) _,_

which is a _C_ [1] function of the inner features. Hence _H_ _∈_ _F_ [�] _[d,T]_ and

[�]


_∥F_ _−_ _H_ [�] _∥∞_ _≤∥F_ _−_ _F_ _◦_ _X_ [�] _q∥∞_

       - ��        _≤ε/_ 4


+ _∥F_ _◦_ _X_ [�] _q −_ _P_ _◦_ _X_ [�] _q∥∞_

 - ��  _≤ε/_ 4


_≤_ _ε/_ 2 _._


Apply Lemma 1 to replace each relaxed primitives by admissible _C_ [2] functions with unique minimizers and to replace _F_ [�] 0 by function _F_ 0 so that the final error increases by at most _ε/_ 2. This leads
to _f_ _∈FD_ _[d,T]_ with _∥F_ _−_ _f_ _∥∞_ _≤_ _ε_ .


A.2 PROOF OF THEOREM 2


A.2.1 PROOF OF THEOREM 2 (1)


Here we prove Theorem 2 (1)


15


**Proof** **Sketch.** The idea is straightforward. Each attention head is assigned to approximate one
term min _t∈Si fi_ ( _x_ ( _t_ )). Once these components are extracted, the outer function _F_ 0 can be approximated by a suitable _F_ [ˆ], completing the construction.


_Proof of Theorem 2 (1):_ _Sufficient expressivity with D heads._ Fix _d, T_ and _α ∈_ (0 _,_ 1), and let _ε >_ 0
be given. Throughout the proof, constants depending only on ( _d, D, T_ ) are absorbed into _Cd,D,T_ _>_
0 which may change from line to line.


In the display equation 4, each head produces an _n_ -dimensional vector and Concat _[h]_ _i_ =1 [gives a vector]
in R _[nh]_ before _F_ [ˆ] . For the construction, we realize the usual _block-by-head_ parameterization, which
means that the encoder outputs a block-decomposed embedding


_x_ ˆ( _t_ ) =        - _x_ ˆ [(1)] ( _t_ ) _, . . .,_ ˆ _x_ [(] _[D]_ [)] ( _t_ )� _∈_ R [2] _[D]_ _,_ _x_ ˆ [(] _[i]_ [)] ( _t_ ) _∈_ R [2] _,_


and the _i_ -th head only reads the _i_ -th block via block-diagonal _WQ,i, WV,i_ (entries bounded by 1).
This keeps the parameter counts within the same order. We therefore set the _per-head_ _embedding_
_dimension_ to _n_ = 2.


Firstly, from Assumption (A2), for any _δ_ _>_ 0 there exist two-layer FFNs Ψ _i,δ_ : [0 _,_ 1] _[d]_ _→_ [0 _,_ 1] such
that

max width(Ψ _i,δ_ ) _≤_ _[C]_ [2] (10)
_x∈_ [0 _,_ 1] _[d]_ _[|][f][i]_ [(] _[x]_ [)] _[ −]_ [Ψ] _[i,δ]_ [(] _[x]_ [)] _[| ≤]_ _[δ,]_ _δ_ _[γ][f]_ _[,]_


where _γf_ _>_ 0 is the exponent from (A2). Define for each head _i_, the position gate


   0 _,_ _s ∈_ _Si,_

_ri_ ( _s_ ) := _s ∈_ [ _T_ ] _._

_−_ 1 _,_ _s_ _∈/_ _Si,_


(Recall _Si_ _⊂_ [ _T_ ] with _|Si| ≥_ _αT_ by equation 8.) We implement the encoder _Pϕ_ so that its _i_ -th block
is
_x_ ˆ [(] _[i]_ [)] ( _t_ ) =       - Ψ _i,δ_ ( _x_ ( _t_ )) _,_ _ri_ ( _t_ )       - _∈_ [0 _,_ 1] _× {−_ 1 _,_ 0 _} ⊂_ [ _−_ 1 _,_ 1] [2] _._ (11)

_√_
This choice follows _∥x_ ˆ( _t_ ) _∥_ 2 _≤_ 2 _D_ . After a fixed rescaling (absorbed into _β_ ), this meets the norm

constraint.


Secondly, we would like to use head-wise attention to isolate the minimum on _Si_ . For each head _i_,
we take a single attention logit ( _mh_ = 1) by choosing


_WO,i_ = _I,_ _WK,i_ = [ _−_ 1 1] _,_ _WQ,ic_ ˆ0 = 1 _,_ _WV,i_ = _I_ 2 _._


All entries are within the allowed bound 1. With the block equation 11, the (pre-softmax) score of
token _t_ in head _i_ is


_ρi_ ( _t_ ) = ( _WK,ix_ ˆ [(] _[i]_ [)] ( _t_ )) _[⊤]_ ( _WQ,ic_ ˆ0) = _−_ Ψ _i,δ_          - _x_ ( _t_ )� + _ri_ ( _t_ ) _._ (12)


Let _σ_ [ _ρi_ ] be the softmax equation 5 with _β_ _>_ 0. Define the head- _i_ value readout (first coordinate of
the head output)


_z_ ˜ _i_ ( _XT_ ) :=


_T_

- _σ_ [ _ρi_ ]( _t_ ) Ψ _i,δ_ - _x_ ( _t_ )� _._ (13)


_t_ =1


Here, the second coordinate is unused. _F_ [ˆ] could ignore it via a fixed linear projection, counted in the
constant _Cd,D,T_ .


Now, we give a uniform bound on _Si_ . Take _at_ := Ψ _i,δ_ ( _x_ ( _t_ )) _∈_ [0 _,_ 1] and split the sum into _Si_ and
_Si_ _[c]_ [.] [From] _[ r][i]_ [(] _[t]_ [) = 0][ on] _[ S][i]_ [and] _[ r][i]_ [(] _[t]_ [) =] _[ −]_ [1][ on] _[ S]_ _i_ _[c]_ [, we have]


_e_ _[β]_ [(] _[−][a][t]_ [+] _[r][i]_ [(] _[t]_ [))]
_σ_ [ _ρi_ ]( _t_ ) = - _[−][βa][u]_


_[≤]_
_u∈Si_ _[c]_ _[e][−][β]_ [(] _[a][u]_ [+1)]


 _e_ _[−][βa][t]_
 - _u∈Si_ _[e][−][βa][u]_ _[,]_ _t ∈_ _Si,_





_e_ _[−][βa][t]_
_e_ _[−][β]_ _·_

 


_u∈Si_ _[e][−][βa][u]_ [+][ �]


_[,]_ _t ∈_ _Si_ _[c][.]_
_u∈Si_ _[e][−][βa][u]_


16


Hence, we have


_i_ _[.]_ (14)

_u∈Si_ _[e][−][βa][u]_


  + _e_ _[−][β]_


_t∈Si_ _[c]_ _[a][t][e][−][βa][t]_

- _[−][βa][u]_


_z_ ˜ _i_ = 


- _σ_ [ _ρi_ ]( _t_ ) _at_ + 
_t∈Si_ _t∈S_


_σ_ [ _ρi_ ]( _t_ ) _at_ _≤_
_t∈Si_ _[c]_


_u∈Si_ _[e][−][βa][u]_

- �� Gibbs mean on _Si_


_t∈Si_ _[a][t][e][−][βa][t]_

- _[−][βa][u]_


To simplify, we denote _a∗_ := min _t∈Si at_ and _bt_ := _at −_ _a∗_ _∈_ [0 _,_ 1] for _t ∈_ _Si_ . Then, we have

  - _[−][βa][t]_  - _[−][βb][t]_


_t∈Si_ _[a][t][e][−][βa][t]_

- _[−][βa][u]_


- _bte_ _[−][βb][t]_ _≤_ _a∗_ + _[|][S][i][| −]_ [1]

_eβ_

_t∈Si_


_,_
_eβ_


_i_ [=] _[ a][∗]_ [+]

_u∈Si_ _[e][−][βa][u]_


_t∈Si_ _[b][t][e][−][βb][t]_

- _[−][βb][u]_


_∈Si_ _≤_ _a∗_ + 
_u∈Si_ _[e][−][βb][u]_ _t∈S_


The inequality comes from sup _b∈_ [0 _,_ 1] _be_ _[−][βb]_ = _e_ _[−]_ [1] _/β_ for _β_ _≥_ 1 and one of the _bt_ is 0, so the
denominator in the middle fraction is _≥_ 1. For the _Si_ _[c]_ [term in 14, we use] _[ a][t]_ _[≤]_ [1][ and][ �] _u∈Si_ _[e][−][βa][u]_ _[≥]_
1 to get

      
_t∈Si_ _[c]_ _[a][t][e][−][βa][t]_ _−β_

_e_ _[−][β]_            - _≤_ _e_ _[−][β]_ [ ��] _Si_ _[c]_ �� _≤_ _e_ _T._

_u∈Si_ _[e][−][βa][u]_

Combining the two bounds, we have the uniform estimate:

min   - _x_ ( _t_ )� _≤_ _z_ ˜ _i_ ( _XT_ ) _≤_ min   - _x_ ( _t_ )� + _[|][S][i][| −]_ [1] + _Te_ _[−][β]_ ( _β_ _≥_ 1) _._ (15)
_t∈Si_ [Ψ] _[i,δ]_ _t∈Si_ [Ψ] _[i,δ]_ _eβ_


In particular, since _|Si| ≤_ _T_, there is a constant _CT_ with

0 _≤_ _z_ ˜ _i_ ( _XT_ ) _−_ min    - _x_ ( _t_ )� _≤_ _CT_    - 1 _,_ _CT_ := max _{T/e,_ _T_ _}._ (16)
_t∈Si_ [Ψ] _[i,δ]_ _β_ [+] _[ e][−][β]_ [�]


Thirdly, we need to lift bounds from _z_ ˜ _i_ to _zi_ . From equation 10 and the definition of _zi_,

                -                 -                 -                 - [�]
min _x_ ( _t_ ) _−_ min _x_ ( _t_ ) _≤_ _δ._
��� _t∈Si_ _[f][i]_ _t∈Si_ [Ψ] _[i,δ]_ ��


Together with equation 16,

              - 1
�� _z_ ˜ _i_ ( _XT_ ) _−_ _z_ ¯ _i_ ( _XT_ )�� _≤_ _δ_ + _CT_ for all _XT_ _∈XT,_ _i_ = 1 _, . . ., D._ (17)
_β_ [+] _[ e][−][β]_ [�]

Let _L_ 0 := sup _z∈_ [0 _,_ 1] _D ∥∇F_ 0( _z_ ) _∥_ 1 _< ∞_ (compactness and _C_ [1] ). Choose


_ε_           -           - 4 _CT L_ 0 _D_
_δ_ := _β_ _≥_ _βε_ := max 1 _,_ [4] _[C][T][ L]_ [0] _[D]_ _,_ log
4 _L_ 0 _D_ _[,]_ _ε_ _ε_


Then by equation 17,


- [�]
_._


_ε_
_∥z_ ˜( _XT_ ) _−_ _z_ ( _XT_ ) _∥∞_ _≤_ for all _XT_ _∈XT ._ (18)
2 _L_ 0 _D_


Finally, we constrcut the approximation for _F_ 0 and count the number of parameter. By Assumption (A1), there exists a two-layer FFN Φ _δ_ 0 : [0 _,_ 1] _[D]_ _→_ R with width _≤_ _C_ 1 _/δ_ 0 _[γ]_ [0] [(for some] _[ γ]_ [0] _[>]_ [ 0][)]
such that
_z∈_ max[0 _,_ 1] _[D]_ _[|][F]_ [0][(] _[z]_ [)] _[ −]_ [Φ] _[δ]_ [0][(] _[z]_ [)] _[| ≤]_ _[δ]_ [0] _[.]_

Set _δ_ 0 := _ε/_ 2. Define the model’s final feed-forward _F_ [ˆ] to project R [2] _[D]_ _→_ R _[D]_ by keeping the first
coordinate of each head (a fixed linear map with entries in _{_ 0 _,_ 1 _}_ ) and apply Φ _δ_ 0.


Then for all _XT_, we have
�� _F_ ˆ(Concat _i_ ( _·_ )) _−_ _F_ 0� _z_ ( _XT_ )��� _≤_ ��Φ _δ_ 0� _z_ ˜( _XT_ )� _−_ Φ _δ_ 0� _z_ ( _XT_ )��� + ��Φ _δ_ 0� _z_ ( _XT_ )� _−_ _F_ 0� _z_ ( _XT_ )���

_≤_ _L_ 0 _∥z_ ˜( _XT_ ) _−_ _z_ ( _XT_ ) _∥_ 1 + _δ_ 0
_≤_ _L_ 0 _D ∥z_ ˜( _XT_ ) _−_ _z_ ( _XT_ ) _∥∞_ + _ε/_ 2
_≤_ _ε/_ 2 + _ε/_ 2 = _ε,_


where we used equation 18 in the last inequality.


Here, the trainable components are composed of three parts:


17


- the _D_ subnetworks Ψ _i,δ_ inside the encoder blocks equation 11;


    - the fixed-size projections _WQ,i, WK,i, WV,i_ (size _O_ ( _D_ ) and independent of _ε_ );


    - the two-layer FFN Φ _δ_ 0 used inside _F_ [ˆ] .


Thus


_M_ _≤_ _C_ _[′]_ _D ·_ [1]


[1] _[C]_ _[′′][ ·]_ [1]

_δ_ _[γ][f]_ [+] _δ_ _[γ]_


+ _C_ _[′′′]_ for constants _C_ _[′]_ _, C_ _[′′]_ _, C_ _[′′′]_ = _Cd,D,T ._
_δ_ 0 _[γ]_ [0]


With _δ_ = Θ( _ε_ ) and _δ_ 0 = Θ( _ε_ ) chosen above,


_M_ _≤_ _[C][d,D,T]_ _,_ _γ_ := max _{γf_ _, γ_ 0 _},_

_ε_ _[γ]_


and the construction uses _h_ = _D_ heads with per-head dimension _n_ = 2 and achieves _ε_ approximation on _XT_ . This proves Theorem 2 (1).


A.2.2 PROOF OF THEOREM 2 (2)


**Proof Sketch.** The argument proceeds in two parts. The core idea is to construct two sequences
whose representations after the attention layer are indistinguishably close, on the order of _O_ ( _ϵ_ _[k]_ [+1] ),
yet whose target outputs differ by at least 3 _ϵ_ . Lemma 4 then implies the lower bound on the parameter count required for approximation.


Using Lemmas 5, 6, and 7, we obtain _D_ disjoint neighborhoods around the minima _x_ [(] _[i]_ [)] . Since
_D_ _>_ _s_ = _h_, there exists at least one neighborhood not selected by the _s_ heads. Within this region, the pigeonhole principle guarantees the existence of two distinct subsequences. By carefully
designing these subsequences, we ensure that their outputs after the attention layer are nearly indistinguishable, while their target values differ by at least 3 _ϵ_ . Extending them to full sequences
completes the construction.


We now turn to the full proof. To establish Theorem 2 (2), we begin by introducing several auxiliary
lemmas that will serve as building blocks for the argument. Lemma 5, 6, and 7 are only to set up the
approximation problem into a more tractable form.
**Lemma 4.** _Let v_ 1 _, v_ 2 _∈_ R _[n]_ _._ _Suppose_


_∥v_ 1 _−_ _v_ 2 _∥_ 2 _≤_ _A_ _and_ _∥F_ [ˆ] ( _v_ 1) _−_ _F_ [ˆ] ( _v_ 2) _∥≥_ _B,_


_where_ _F_ [ˆ] : R _[n]_ _→_ R _[m]_ _is a two-layer feed-forward network satisfying the constraints stated above._
_Then_ _F_ [ˆ] _must use at least_


 - _B_
Ω
_A_ ~~_[√]_~~ _n_


_parameters._


_Proof of Lemma 4._ Let ∆ _x_ := _v_ 1 _−_ _v_ 2 and ∆ _F_ := _F_ [ˆ] ( _v_ 1) _−_ _F_ [ˆ] ( _v_ 2). Suppose the two-layer network
with width _p_ be
_F_ ˆ( _x_ ) = _V_ _σ_ ( _Ux_ + _b_ ) + _c,_
where _U_ _∈_ R _[p][×][n]_, _V_ _∈_ R _[m][×][p]_, _b_ _∈_ R _[p]_, _c_ _∈_ R _[m]_, _σ_ is 1-Lipschitz acting coordinate-wise and every
entry of _U, V, b, c_ has magnitude at most 1.


For the _j_ -th output coordinate, we have


∆ _Fj_ =


_p_

- _Vjr_ - _σ_ ( _u_ _[⊤]_ _r_ _[v]_ [1] [+] _[ b][r]_ [)] _[ −]_ _[σ]_ [(] _[u][⊤]_ _r_ _[v]_ [2] [+] _[ b][r]_ [)] - _,_

_r_ =1


where _u_ _[⊤]_ _r_ [is the] _[ r]_ [-th row of] _[ U]_ [.] [Using the][ 1][-Lipschitz property of] _[ σ]_ [ and Cauchy–Schwarz inequality,]
we have


_p_


_|Vjr| ∥ur∥_ 2 _∥_ ∆ _x∥_ 2 _._

_r_ =1


_|_ ∆ _Fj| ≤_


_p_


- _|Vjr| |u_ _[⊤]_ _r_ [∆] _[x][| ≤]_

_r_ =1


18


By the entrywise weight bound, _∥ur∥_ 2 _≤_ _[√]_ _n_ and _|Vjr| ≤_ 1. Therefore, for all _j_, we have

_|_ ∆ _Fj|_ _≤_ _p_ _[√]_ _n ∥_ ∆ _x∥_ 2 _._ (19)


Let _∥· ∥_ be the norm used in the lemma statement. By norm equivalence in finite dimensions, there
exists _cm_ _∈_ (0 _,_ 1] depending only on the chosen norm and _m_ such that

_∥y∥≤_ [1] _∥y∥∞_ for all _y_ _∈_ R _[m]_ _._

_cm_

_∥_ ∆ _F_ _∥≥_ _B_ implies _∥_ ∆ _F_ _∥∞_ _≥_ _cmB_, so there is some _j_ _[⋆]_ with


_|_ ∆ _Fj⋆_ _|_ _≥_ _cmB._


Combining this with equation 19 and _∥_ ∆ _x∥_ 2 _≤_ _A_, we have

_cmB_ _≤_ _p_ _[√]_ _n A_ = _⇒_ _p_ _≥_ _[c][m][ B]_

_A_ ~~_[√]_~~ _n_ _[.]_


Finally, let _p_ eff _≤_ _p_ be the number of hidden units that actually affect the output, i.e., those with a
nonzero row in _U_ and a nonzero entry in the _j_ _[⋆]_ -th row of _V_ . The above bound holds with _p_ eff in
place of _p_, hence _p_ eff _≥_ _cmB/_ ( _A_ _[√]_ _n_ ). Each such unit uses at least one nonzero parameter in _U_ and
one in _V_, so the parameter counts _k_ satisfy _k_ _≥_ _p_ eff . Therefore


     - _B_

_k_ _≥_ _[c][m][ B]_ [Ω]

_A_ ~~_[√]_~~ _n_ [=] _A_ ~~_[√]_~~ _n_


_,_


which proves the lemma.


**Lemma** **5.** _There_ _exists_ _R_ _>_ 0 _such_ _that_ _for_ _every_ _i_ _∈{_ 1 _, . . ., D}_ _and_ _every_ _r_ _<_ _R,_ _there_ _exist_
_constants δi_ _>_ 0 _and Li_ _>_ 0 _with the following property:_ _there exists a segment Gi_ _⊂_ _B_ ( _x_ [(] _[i]_ [)] _, r_ ) _of_
_length δi_ _such that_
_|fi_ ( _x_ ) _−_ _fi_ ( _y_ ) _|_ _≥_ _Li ∥x −_ _y∥_ 2 _,_ _∀_ _x, y_ _∈_ _Gi._
_and moreover_
_fi_ ( _x_ ) _> zi,_ _∀_ _x ∈_ _Gi,_


_Proof of Lemma 5._ Fix _i_ _∈{_ 1 _, . . ., D}_ and denote _x_ _[⋆]_ := _x_ [(] _[i]_ [)], _f_ := _fi_ and _H⋆_ := _∇_ [2] _x_ _[f]_ [(] _[x][⋆]_ [)][.] [By]
positive definiteness, let _λi_ := _λ_ min( _H⋆_ ) _>_ 0. By continuity of _∇_ [2] _f_, there exists _Ri_ [H] _[>]_ [ 0][ such that]

_∇_ [2] _f_ ( _x_ ) _⪰_ _λ_ 2 _i_ _[I]_ for all _x ∈_ _B_ ( _x_ _[⋆]_ _, Ri_ [H][)] _[.]_

Set _µi_ := _λi/_ 2 _>_ 0. Because the domain is [0 _,_ 1] _[d]_ and _x_ _[⋆]_ _∈_ [0 _,_ 1] _[d]_, we could choose a unit vector
_vi_ pointing strictly into the cube at _x_ _[⋆]_ (if _x_ _[⋆]_ is interior, take any unit vector). Define

_τi_ := sup _{ t >_ 0 : _x_ _[⋆]_ + _svi_ _∈_ [0 _,_ 1] _[d]_ for all _s ∈_ [0 _, t_ ] _}_ _>_ 0 _,_

and set _Ri_ := min _{Ri_ [H] _[,]_ _[τ][i][}]_ [.] [Take] _[ R]_ [:=] [min][1] _[≤][i][≤][D][ R][i]_ _[>]_ [0][.] [Fix any] _[ r]_ _[∈]_ [(0] _[, R]_ [)][ and consider the]
restriction
_g_ ( _t_ ) := _f_ ( _x_ _[⋆]_ + _tvi_ ) _,_ _t ∈_ [0 _, r_ ] _._

Then
_g_ _[′′]_ ( _t_ ) = _vi_ _[⊤][∇]_ [2] _[f]_ [(] _[x][⋆]_ [+] _[ tv][i]_ [)] _[ v][i]_ _[≥]_ _[µ][i]_ for all _t ∈_ [0 _, r_ ] _._


Since _x_ _[⋆]_ minimizes _f_ on [0 _,_ 1] _[d]_ and _vi_ is feasible inward, we have _g_ ( _t_ ) _≥_ _g_ (0) for small _t_ _≥_ 0
leading to the one-sided derivative _g_ _[′]_ (0+) _≥_ 0 (if _x_ _[⋆]_ is interior then _∇f_ ( _x_ _[⋆]_ ) = 0 so _g_ _[′]_ (0) = 0).
Because _g_ _[′′]_ _≥_ _µi_, the derivative _g_ _[′]_ is increasing and thus

_g_ _[′]_ ( _t_ ) _≥_ _g_ _[′]_ (0+) + _µit_ _≥_ _µit,_ _t ∈_ [0 _, r_ ] _._


Let _a_ := _r/_ 4 and _b_ := _r/_ 2 and define the segment

_Gi_ := _{ x_ _[⋆]_ + _tvi_ : _t ∈_ [ _a, b_ ] _}_ _⊂_ _B_ ( _x_ _[⋆]_ _, r_ ) _,_

whose length is _δi_ := _b −_ _a_ = _r/_ 4. For any _x_ = _x_ _[⋆]_ + _tvi_ and _y_ = _x_ _[⋆]_ + _svi_ in _Gi_ with _t_ _>_ _s_, the
mean value theorem gives some _ξ_ _∈_ ( _s, t_ ) _⊂_ [ _a, b_ ] such that


_|f_ ( _x_ ) _−_ _f_ ( _y_ ) _|_ = _|g_ ( _t_ ) _−_ _g_ ( _s_ ) _|_ = _|g_ _[′]_ ( _ξ_ ) _| |t −_ _s|_ _≥_ _µia |t −_ _s|_ = - _λir_
8


19


_∥x −_ _y∥_ 2 _._


Therefore the choice
_Li_ := _[λ][i][r]_ _>_ 0

8

works uniformly for all _x, y_ _∈_ _Gi_ . The segment _Gi_ does not contain _x_ _[⋆]_, for all its points are at
distance at least _a_ = _r/_ 4 _>_ 0 from _x_ _[⋆]_ . By uniqueness of the minimizer, _f_ ( _x_ ) _> f_ ( _x_ _[⋆]_ ) = _zi_ for all
_x ∈_ _Gi_ .

The lemma holds with _δi_ = _r/_ 4 and _Li_ = ( _λ_ min( _∇_ [2] _x_ _[f][i]_ [(] _[x]_ [(] _[i]_ [)][))] _[ r]_ [)] _[/]_ [8][.]


**Lemma** **6.** _Let_ ( _z_ 1 _, . . ., zD_ ) _denote_ _the_ _minima_ _defined_ _above._ _Then_ _there_ _exist_ _constants_ _r_ 0 _>_ 0
_and_ _L_ 0 _>_ 0 _such_ _that_ _the_ _following_ _holds:_ _for_ _any_ _i_ _∈{_ 1 _, . . ., D}_ _and_ _any_ _perturbation_ _δ_ 0 _with_
_|δ_ 0 _| < r_ 0 _,_
�� _F_ 0( _z_ 1 _, . . ., zi_ + _δ_ 0 _, . . ., zD_ ) _−_ _F_ 0( _z_ 1 _, . . ., zD_ )�� _≥_ _L_ 0 _|δ_ 0 _|._


_Proof of Lemma 6._ Denote _ei_ for the _i_ -th standard basis vector of R _[D]_ . By assumption, _mi_ :=
�� _∂iF_ 0( _z_ )�� _>_ 0 for each _i_ . Since _F_ 0 _∈_ _C_ 1, the map _u �→_ _∂iF_ 0( _u_ ) is continuous at _z_ . Hence for each
_i_, there exists _ri_ [cont] _>_ 0 such that
�� _∂iF_ 0( _u_ )�� _≥_ 12 _[m][i]_ whenever _∥u −_ _z∥∞_ _< ri_ [cont] _._

If necessary, shrink _ri_ [cont] so that the line segment _{ z_ + _tei_ : _|t|_ _<_ _ri_ [cont] _}_ lies in [0 _,_ 1] _[D]_ . Define
uniform constants

_L_ 0 := 12 1 _≤_ [min] _i≤D_ _[m][i]_ _[>]_ [0] _[,]_ _r_ 0 := 1 _≤_ min _i≤D_ _[r]_ _i_ [cont] _>_ 0 _._


Fix _i_ and _δ_ 0 with _|δ_ 0 _|_ _<_ _r_ 0. Consider the one-dimensional slice _gi_ ( _t_ ) := _F_ 0( _z_ + _tei_ ) for _|t|_ _<_ _r_ 0.
Then _gi_ is _C_ [1] and _gi_ _[′]_ [(] _[t]_ [)] [=] _[∂][i][F]_ [0][(] _[z]_ [ +] _[ te][i]_ [)][.] [By the mean value theorem, there exists] _[ θ]_ _[∈]_ [(0] _[,]_ [ 1)][ such]
that
_F_ 0( _z_ + _δ_ 0 _ei_ ) _−_ _F_ 0( _z_ ) = _gi_ _[′]_ [(] _[θδ]_ [0][)] _[ δ]_ [0] [=] _[ ∂][i][F]_ [0]       - _z_ + _θδ_ 0 _ei_       - _δ_ 0 _._

Taking absolute values and using the lower bound on �� _∂iF_ 0( _·_ )�� inside the _ℓ∞_ -ball of radius _r_ 0 around
_z_, we have
�� _F_ 0( _z_ + _δ_ 0 _ei_ ) _−_ _F_ 0( _z_ )�� _≥_ _L_ 0 _|δ_ 0 _|,_
which is the desired inequality.


**Lemma 7.** _Let zi_ = min _x∈_ [0 _,_ 1] _d fi_ ( _x_ ) _and let x_ [(] _[i]_ [)] _denote the unique minimizer of fi_ _(as assumed_
_above)._ _Then there exist constants R_ 0 _>_ 0 _and ε_ 0 _>_ 0 _such that:_


_1._ _The open balls {B_ ( _x_ [(] _[i]_ [)] _, R_ 0) _}_ _[D]_ _i_ =1 _[are pairwise disjoint.]_


_2._ _For each i ∈{_ 1 _, . . ., D} and every x ∈_ [0 _,_ 1] _[d]_ _\ B_ ( _x_ [(] _[i]_ [)] _, R_ 0) _,_


_fi_ ( _x_ ) _>_ _zi_ + _ε_ 0 _._


_Proof of Lemma 7._ Since the minimizers _{x_ [(] _[i]_ [)] _}_ _[D]_ _i_ =1 [are] [pairwise] [distinct] [and] [finite] [in] [number,] [we]
have
∆:= min _i_ = _j_ �� _x_ ( _i_ ) _−_ _x_ ( _j_ )��2 _[>]_ [0] _[.]_

Set _R_ 0 := [1] 2 [∆][.] [If] _[ i][ ̸]_ [=] _[ j]_ [and] _[ x][ ∈]_ _[B]_ [(] _[x]_ [(] _[i]_ [)] _[, R]_ [0][)][, by the triangle inequality, we have]


_∥x −_ _x_ [(] _[j]_ [)] _∥_ 2 _≥∥x_ [(] _[i]_ [)] _−_ _x_ [(] _[j]_ [)] _∥_ 2 _−∥x −_ _x_ [(] _[i]_ [)] _∥_ 2 _>_ ∆ _−_ _R_ 0 = _R_ 0 _,_


so _x_ _∈/_ _B_ ( _x_ [(] _[j]_ [)] _, R_ 0). Hence the balls are pairwise disjoint, proving the first part.


For the second part, fix _i_ and define the compact set _Ki_ := [0 _,_ 1] _[d]_ _\ B_ ( _x_ [(] _[i]_ [)] _, R_ 0). The continuity of
_fi_ implies that the minimum
_mi_ := min
_x∈Ki_ _[f][i]_ [(] _[x]_ [)]

is attained on _Ki_ . Because _x_ [(] _[i]_ [)] _∈/_ _Ki_ and _x_ [(] _[i]_ [)] is the unique global minimizer on [0 _,_ 1] _[d]_, we have
_mi_ _> zi_ . Let _εi_ := _mi −_ _zi_ _>_ 0 and set

_ε_ 0 := 21 1 _≤_ [min] _i≤D_ _[ε][i]_ _[>]_ [0] _[.]_


20


**Notation fow (dependency structure)** **Meaning**
_x_ [(] _[i]_ [)] Point where _fi_ achieves minimum
_→_ _B_ ( _x_ [(] _[i]_ [)] _, r_ ) Basin region for retrieval coordinate _i_
(Basin around _x_ [(] _[i]_ [)] )
_→_ _Gi, Ki_ Monotone local segment near _x_ [(] _[i]_ [)]

(In _B_ ( _x_ [(] _[i]_ [)] _, r_ ))


_P_ 0 The set of all candidate points.
(We only choose _xt_ _∈_ _P_ 0)
_Si_ Index partition for retrieval coordinate _i_
( _i_ = 1 _, . . ., D_ )


Attention head _j_ Defines response at position _t_
_→_ _λj_ ( _x, t_ ) Attention score
_→_ ( _yj, tj_ ) Maximum-attention point selected by
head _j_ in _P_ 0 _× Sj_
_→_ _Y_ = _{y_ 1 _, . . ., ys}_ Chosen maximizers of attention score
(one per head)
_→_ _vj_ ( _x, t_ ) Value embedding


WLOG, suppose _K_ 1 _∩_ _Y_ = _∅_ .
_→_ _T_ 0 _T_ 0 _⊂_ _S_ 1, indices not in ( _yj, tj_ ), _j_ = 1 _, . . ., s_
_→_ _η_ : [0 _,_ 1] _→_ _G_ 1 Coordinate system on _G_ 1
_→_ _q_ = _f_ 1 _◦_ _η_ Rewriting _f_ 1 _|G_ 1 into the coordinate system.
_→_ _Ut_ Discrete grid on [0 _,_ 1] at index _t_
_→_ _zℓ_ ( _t_ ) Candidate point for subsequence _ℓ_, _zℓ_ ( _t_ ) _∈_ _η_ ( _Ut_ )


Adversarial subsequences
_→_ _Zℓ_ = ( _zℓ_ (1) _, . . ., zℓ_ ( _T_ 0)) Two subsequences almost
indistinguishable by attention head.
_→_ _Wℓ_ Full sequence embedding _Zℓ_
_→_ _wℓ_ ( _t_ ) Token of _Wℓ_ of index _t_
_→_ _I_ 1 _, I_ 2 _, I_ 3 Partition of indices: differ / agree / remaining


Per-head analysis
_→_ _Qj,i_ Attention mass on _Ij_ ( _j_ _∈{_ 1 _,_ 2 _,_ 3 _}, i ∈{_ 1 _, . . ., s}_ )
_→_ _Vj,i_ Weighted value average on _Ij_


Table 1: Flow-style dependency map of notation introduced in the proof of Theorem 2.2.


Then, for every _x ∈_ _Ki_, we have


_fi_ ( _x_ ) _≥_ _mi_ = _zi_ + _εi_ _≥_ _zi_ + 2 _ε_ 0 _>_ _zi_ + _ε_ 0 _,_


Thus, we have proved this lemma.


Before the proof, We also provide a notation table to help with understanding in Table 1.


_Proof of Theorem 2 (2)._ Given the target function under the assumptions. For any given single-layer
transformer defined in the main context, our goal is to find two different sequences such that their
output in the part

Concat _[h]_ _i_ =1�� _[T]_ _σ_ �( _WK,ix_ ˆ( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,ix_ ˆ( _t_ )� (20)

_t_ =1


are very close (differs by only _O_ ( _ϵ_ _[k]_ [+1] )), but their output from the target function differs by at least
3 _ϵ_, then according to lemma 4, we have the required parameter count for the FFN _F_ [ˆ] to be at least
Ω(1 _/ϵ_ _[k]_ ).


21


**Notations** For each head _i_ = 1 _, . . ., s_, define the attention weight function


_λi_ ( _x, t_ ) = exp� _γ_ ( _WK,iPϕ_ ( _x, t_ )) _[⊤]_ _WQ,ic_ ˆ0� _,_


and the value mapping
_vi_ ( _x, t_ ) = _WV,iPϕ_ ( _x, t_ ) _∈_ R _[n]_ _,_

where _γ_ _>_ 0 is the softmax scaling factor, _WQ,i, WK,i_ _∈_ R _[n][×][E]_ are the query and key projections,
and _WV,i_ _∈_ R _[n][×][E]_ is the value projection for head _i_ .


**Notation of sets** Without loss of generality, we assume _x_ [(] _[i]_ [)] belongs to the interior of [0 _,_ 1] _[d]_, and
the other case can be treated with the same method below. From lemma 5, lemma 6 and lemma 7
we have that there exists _R_ _>_ 0 and segments _Gi_ _⊂_ _B_ ( _x_ [(] _[i]_ [)] _, R_ ) _, i_ = 1 _, . . ., D_ and _L, δ_ 0 _, r_ _>_ 0
satisfying the following:


    - _∀i_, _∀x, y_ _∈_ _Gi_, we have _|fi_ ( _x_ ) _−_ _fi_ ( _y_ ) _| > L∥x −_ _y∥_ 2.


    - _∀j_ = _i_ and _∀x ∈_ _B_ ( _x_ [(] _[j]_ [)] _, R_ ) _, y_ _∈_ _B_ ( _x_ [(] _[i]_ [)] _, R_ ), we have _fi_ ( _y_ ) _−_ _fi_ ( _x_ ) _> δ_ 0.


    - The length of _Gi_ is _r_, _∀i_ = 1 _, . . ., D_ .


    - For any _i ∈{_ 1 _, . . ., D}_ and any perturbation _δ_ 1 with _|δ_ 1 _| <_ max _x∈B_ ( _x_ ( _i_ ) _,R_ )( _fi_ ( _x_ ) _−_ _zi_ ),

�� _F_ 0( _z_ 1 _, . . ., zi_ + _δ_ 1 _, . . ., zD_ ) _−_ _F_ 0( _z_ 1 _, . . ., zD_ )�� _≥_ _L |δ_ 1 _|._


We denote by _Ki_ := _Gi_ _∪{x_ [(] _[i]_ [)] _}, i_ = 1 _, . . ., D_, and _P_ 0 = _∪_ _[D]_ _i_ =1 _[K][i]_ [. Recall that] _[ k]_ [=]


1
4 _[T][ −][s][−][D]_ [+1]


We denote by _Ki_ := _Gi_ _∪{x_ [(] _[i]_ [)] _}, i_ = 1 _, . . ., D_, and _P_ 0 = _∪_ _[D]_ _i_ =1 _[K][i]_ [. Recall that] _[ k]_ [=] 4( _n_ +1) _s_ +1 _[−]_ [1][.]

We assume without loss of generality that _k_ _>_ 0 and [1] 4 _[T]_ _[−]_ _[s][ −]_ _[D]_ [ + 1] _[>]_ [0][,] [otherwise] [the] [result]

would be trivial.


**Max weight for each head** For _j_ = 1 _, . . ., s_, define recursively the pairs ( _yj, tj_ ) as follows:


    - For the first head,
( _y_ 1 _, t_ 1) = arg max _λ_ 1( _y, t_ ) _._
_y∈P_ 0
_t∈S_ 1


- For _j_ _>_ 1,
( _yj, tj_ ) = arg max
_y∈P_ 0
_t∈Sj_
_t/∈{t_ 1 _,...,tj−_ 1 _}_


_λj_ ( _y, t_ ) _._


    - If maximum can be obtained at multiple ( _y, t_ ), then choose one of them.


Let _Y_ = _{y_ 1 _, . . ., ys}_ . Since the sets _K_ 1 _, . . ., KD_ are pairwise disjoint and _s_ _<_ _D_, there exists at
least one index _i ∈{_ 1 _, . . ., D}_ such that


_Ki_ _∩_ _Y_ = ∅ _._

Without loss of generality, we assume that _i_ = 1. As we have _|Si| ≥_ [1] 4 _[T]_ _[> s]_ [+] _[D]_ _[−]_ [1] _[, i]_ [ = 1] _[, . . ., D]_ [,]

we have that there exists a set of ( _t_ _[∗]_ 2 _[, . . ., t][∗]_ _D_ [)][ such that]


    - _t_ _[∗]_ _j_ _[∈{][/]_ _[t]_ [1] _[, . . ., t][s][}]_ [, for] _[ j]_ [= 2] _[, . . ., D]_ [.]

    - _t_ _[∗]_ _j_ [are pairwise distinct.]

    - _t_ _[∗]_ _j_ _[∈]_ _[S][j][, j]_ [= 2] _[, . . ., D]_ [.]


Let _T_ 0 = 14 _[T]_ _[−]_ _[s]_ _[−]_ _[D]_ [+] [1] _[>]_ [0] [and] [assume] [that] _[T]_ [0] [is] [a] [integer.] Then we have _|S_ 1 _−_
_{t_ 1 _, . . ., ts, t_ _[∗]_ 2 _[, . . ., t][∗]_ _D_ _[}|]_ _[≥]_ _[T]_ [0] _[>]_ [0][.] Without loss of generality, suppose _{_ 1 _,_ 2 _, . . ., T_ 0 _}_ _⊂_
_S_ 1 _−{t_ 1 _, . . ., ts, t_ _[∗]_ 2 _[, . . ., t][∗]_ _D_ _[}]_ [.]


22


**Sequences to be considered** As _G_ 1 is a segment of length _r_, then it is natural to assign coordinate
system _η_ : [0 _,_ 1] _→_ _G_ 1 on _G_ 1, with _q_ := _f_ 1 _◦_ _η_ being a monotonically increasing function on [0 _,_ 1].
The monotone property is as a result of _|f_ 1( _x_ ) _−_ _f_ 1( _y_ ) _| ≥_ _L∥x −_ _y∥_ 2.
We denote by _M_ = _T_ 0 _⌊_ 3 _[rL]_ _T_ 0 [2] _ϵ_ _[⌋]_ [.] [As] _[ T]_ [0] _[|][M]_ [, Construct the following] _[ T]_ [0][ sets:]


_Uj_ = _[j][ −]_ [1]


_[T]_ [0]

_M_

[1]

_M_ _[, . . .,]_ [ (] _M_


[1]

+ _{_ [1]
_T_ 0 _M_


[0]

_M_ [)]

[= 1] _[, . . ., T]_ [0] (21)
_M_ _[}][, j]_


We have _|Uj|_ = _⌊_ 3 _[rL]_ _T_ 0 [2] _ϵ_ _[⌋]_ [=] _[ O]_ [(1] _[/ϵ]_ [)][.]

_Claim_ 2.1 _._ Existence of two distinct sub-sequence
There exists two subsequences _z_ 1(1) _, . . ., z_ 1( _T_ 0) and _z_ 2(1) _, . . . z_ 2( _T_ 0) with _zi_ ( _t_ ) _∈_ _η_ ( _Ut_ ) satisfying
the following conditions.


����


����2 _≤_ _[ϵ]_ 3 _[k]_ _T_ [+1] 0 [, for] _[ i]_ [ = 1] _[, . . ., s]_ [.]


- _T_ 0
_t_ =1� _[λ]_ _T_ _[i]_ [(] 0 _[z]_ [1][(] _[t]_ [)] _[,t]_ [)] _[v][i]_ [(] _[z]_ [1][(] _[t]_ [)] _[,t]_ [)] _−_
_t_ =1 _[λ][i]_ [(] _[z]_ [1][(] _[t]_ [)] _[,t]_ [)]


- _T_ 0
_t_ =1 _[λ][i]_ [(] _[z]_ [2][(] _[t]_ [)] _[,t]_ [)] _[v][i]_ [(] _[z]_ [2][(] _[t]_ [)] _[,t]_ [)]

 - _T_ 0
_t_ =1 _[λ][i]_ [(] _[z]_ [2][(] _[t]_ [)] _[,t]_ [)]


For each _i_ = 1 _, . . ., s_, either of the following holds:


1.


�� _TtTt_ =1=100 _[λ][λ][i][i]_ [(][(] _[z][z]_ [2][1][(][(] _[t][t]_ [)][)] _[,t][,t]_ [)][)] _[∈]_ �1 _/_ (1 + 12 _[ϵ][k]_ [+1] _T_ 0


12 _[ϵ][k]_ [+1] _T_ 0 [2] [)] _[,]_ [1 +] 12 _[ϵ][k]_ [+1] _T_ 0 [2]


12 _T_ 0 [2]


.


2. max _j_ =1 _,_ 2 - _Tt_ =10 _[λ][i]_ [(] _[z][j]_ [(] _[t]_ [)] _[, t]_ [)] _[ ≤]_ _[ϵ][k]_ 4 [+1] - _sw_ =1 _[λ][i]_ [(] _[y][w][, t][w]_ [)][.]


_Proof._ We compare the orders of 1 _/ϵ_ appearing on both sides of the conditions.


First, since _|Ut|_ = _O_ (1 _/ϵ_ ) for each _t_, the total number of possible choices of subsequences
( _z_ (1) _, . . ., z_ ( _T_ 0)) is at most _O_ (1 _/ϵ_ _[T]_ [0] ).


Next, to satisfy condition (1), note that both vectors involved are _n_ -dimensional with norms bounded
by 1. Thus, the discretization required to achieve accuracy _ϵ_ _[k]_ [+1] _/_ (3 _T_ 0) in the _ℓ_ 2 norm leads to at
most _O_ (1 _/ϵ_ [(] _[k]_ [+1)] _[ns]_ ) distinct possibilities, since there are _s_ heads.


For condition (2), observe that


_s_

- _λi_ ( _yw, tw_ ) _≥_ _T_ 10 _j_ [max] =1 _,_ 2

_w_ =1


_T_ 0

- _λi_ ( _zj_ ( _t_ ) _, t_ ) _._


_t_ =1


Hence, for each _i_, the relevant interval can be partitioned into at most _O_ - _−ϵ_ _[k]_ log [+1] _ϵ_ - sub-intervals.

Taken across _s_ heads, this contributes at most _O_ �( _−_ log _ϵ_ ) _[s]_ _/ϵ_ [(] _[k]_ [+1)] _[s]_ [�] possibilities.


Combining the two conditions, the total number of distinct admissible cases is bounded above by


           - ( _−_ log _ϵ_ ) _[s]_            _O_ _._

_ϵ_ [(] _[k]_ [+1)] _[ns]_ [+(] _[k]_ [+1)] _[s]_

Since _T_ 0 _≥_ ( _k_ + 1) _ns_ + ( _k_ + 1) _s_ + 1, we have


- - ( _−_ log _ϵ_ ) _[s]_
_≫_ _O_

_ϵ_ [(] _[k]_ [+1)] _[ns]_ [+(] _[k]_ [+1)] _[s]_


- - ( _−_ log _ϵ_ ) _[s]_
_≫_ _O_


_._


 - 1
_O_
_ϵ_ _[T]_ [0]


Therefore, by the pigeonhole principle, there must exist two distinct subsequences
( _z_ 1(1) _, . . ., z_ 1( _T_ 0)) and ( _z_ 2(1) _, . . ., z_ 2( _T_ 0)) satisfying all the conditions of Claim 2.1.


**Construction** **of** **Distinct** **sequences** From Claim 2.1, we have constructed two sub-sequences
_Z_ 1 _, Z_ 2 satisfying the given conditions. We now consider the construction of two full input sequence
_W_ 1 _, W_ 2:


    - For _t_ = 1 _, . . ., T_ 0, if _z_ 1( _t_ ) = _z_ 2( _t_ ), then _w_ 1( _t_ ) = _w_ 2( _t_ ) = _x_ [(] _[D]_ [)] . Otherwise, _w_ 1( _t_ ) =
_z_ 1( _t_ ) _, w_ 2( _t_ ) = _z_ 2( _t_ ).


    - _wj_ ( _ti_ ) = _yi_, _i_ = 1 _, . . ., s_ ; _j_ = 1 _,_ 2.


    - _wj_ ( _t_ _[∗]_ _i_ [) =] _[ x]_ [(] _[i]_ [)][,] _[ i]_ [ = 2] _[, . . ., D]_ [;] _j_ = 1 _,_ 2.


    - For all other _t_, _wj_ ( _t_ ) = _x_ [(] _[D]_ [)] .


23


**Difference** **of** _W_ 1 _, W_ 2 **applied** **to** **target** **function** Denote by _I_ 1 the set of all indices _t_ with
_z_ 1( _t_ ) = _z_ 2( _t_ ), and _I_ 2 = [ _T_ 0] _−_ _I_ 1, _I_ 3 = [ _T_ ] _−_ _I_ 1. It is clear from the difference of _Z_ 1 _, Z_ 2 that
_I_ 1 = ∅.
We then define the following notations for the simplicity of calculation (defined for each head _i_ =
1 _, . . ., s_ ):


    - _Q_ 1 _,i_ = [�] _t∈I_ 1 _[λ][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [)][.]

    - _Q_ 2 _,i_ = [�] _t∈I_ 1 _[λ][i]_ [(] _[w]_ [2][(] _[t]_ [)] _[, t]_ [)][.]

    - _V_ 1 _,i_ = ( [�] _t∈I_ 1 _[λ][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [)] _[v][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [))] _[/Q]_ [1] _[,i]_ [.]


    - _V_ 2 _,i_ = ( [�] _t∈I_ 1 _[λ][i]_ [(] _[w]_ [2][(] _[t]_ [)] _[, t]_ [)] _[v][i]_ [(] _[w]_ [2][(] _[t]_ [)] _[, t]_ [))] _[/Q]_ [2] _[,i]_ [.]


    - _Q_ 3 _,i_ = [�] _t∈I_ 2 _[λ][i]_ [(] _[z]_ [1][(] _[t]_ [)] _[, t]_ [)][, which is also the same if defined on] _[ Z]_ [2][.]

    - _V_ 3 _,i_ = ( [�] _t∈I_ 2 _[λ][i]_ [(] _[z]_ [1][(] _[t]_ [)] _[, t]_ [)] _[v][i]_ [(] _[z]_ [1][(] _[t]_ [)] _[, t]_ [))] _[/Q]_ [3] _[,i]_ [, which is the same if defined on] _[ Z]_ [2][.]


    - _Q_ 4 _,i_ = [�] _t∈I_ 3 _[λ][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [)][, which is the same if defined on] _[ W]_ [2][.]

    - _V_ 4 _,i_ = ( [�] _t∈I_ 3 _[λ][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [)] _[v][i]_ [(] _[w]_ [1][(] _[t]_ [)] _[, t]_ [))] _[/Q]_ [4] _[,i]_ [, which is the same if defined on] _[ W]_ [2][.]


As _λi_ () maps to positive values, _Vj,i_ are convex combinations of _vi_ (), whose norm is bounded by 1
according to the constraint section 1. Therefore _∥Vj,i∥≤_ 1, _j_ = 1 _,_ 2 _,_ 3 _,_ 4.


As _f_ 1 _◦_ _η_ is monotone on [0 _,_ 1], let _t_ [˜] = max _t∈I_ 1 _t_, then we have


    - max _t∈S_ 1 _f_ 1( _w_ 1( _t_ )) = _f_ 1( _w_ 1( _t_ [˜] )).


    - max _t∈S_ 1 _f_ 1( _w_ 2( _t_ )) = _f_ 1( _w_ 2( _t_ [˜] )).


And by construction we know that

_∥_ ( _w_ 1( _t_ [˜] )) _−_ ( _w_ 2( _t_ [˜] )) _∥≥_ _[r]_ (22)

_M_


which is the minimal distance for any two points in _Ut_ ˜. Then we have

_|f_ 1( _w_ 1( _t_ [˜] )) _−_ _f_ 1( _w_ 2( _t_ [˜] )) _| ≥_ _[rL]_ (23)

_M_


As we have for _i_ = 2 _, . . ., D_


    - max _t∈Si fi_ ( _w_ 1( _t_ )) = _z_ [(] _[i]_ [)] .


    - max _t∈Si fi_ ( _w_ 2( _t_ )) = _z_ [(] _[i]_ [)] .


Then following the perturbation property of _F_ 0 defined above we have that the difference of output
between the two sequence to be at least _[rL]_ _M_ [2] [, which is greater than][ 3] _[ϵ]_ [. Then] _[ ϵ]_ [-approximation requires]

that _|Model_ ( _W_ 1) _−_ _Model_ ( _W_ 2) _| ≥_ _ϵ_ .


_W_ 1 **and** _W_ 2 **are close after attention layer** For any given head _i_, we consider the the two cases
given in 2.1.


**Case 1** Case 1 can be rewritten as follows:


_Q_ _[V]_ 2 [2] _,i_ _[,i]_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_ _∥_ 2 _≤_ _[ϵ]_ 3 _[k]_ _T_ [+1] 0


- _∥_ _[Q]_ [1] _[,i]_ _Q_ _[V]_ [1] 1 _[,i]_ _,i_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_ _−_ _[Q]_ [2] _[,i]_ _Q_ _[V]_ 2 [2] _,i_ _[,i]_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_


3 _T_ 0 [.]


- _QQ_ 12 _,i,i_ ++ _QQ_ 33 _,i,i_ _[∈]_ �1 _/_ (1 + 12 _[ϵ][k]_ [+1] _T_ 0 [2]


12 _T_ 0 [2]


12 _[ϵ][k]_ [+1] _T_ 0 [2] [)] _[,]_ [1 +] 12 _[ϵ][k]_ [+1] _T_ 0 [2]


.


24


Without loss of generality, we assume _Q_ 1 _,i_ _≥_ _Q_ 2 _,i_ . By calculation, we have


_Q_ 1 _,iV_ 1 _,i_ + _Q_ 3 _,iV_ 3 _,i_


_V_ 1 _,i_ + _Q_ 3 _,iV_ 3 _,i_

_−_ _[Q]_ [2] _[,i][V]_ [2] _[,i]_ [ +] _[ Q]_ [3] _[,i][V]_ [3] _[,i]_
_Q_ 1 _,i_ + _Q_ 3 _,i_ _Q_ 2 _,i_ + _Q_ 3 _,i_


(24)
_Q_ 2 _,i_ + _Q_ 3 _,i_


_Q_ 1 _,i_

= _[Q]_ [3] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [3] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] + ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _._ (25)

( _Q_ 1 _,i_ + _Q_ 3 _,i_ )( _Q_ 2 _,i_ + _Q_ 3 _,i_ ) _Q_ 1 _,i_ + _Q_ 3 _,i_


We have already known that _Q_ 4 _,i_ _≥_ _[Q]_ [1] _[,i]_ _T_ [+] 0 _[Q]_ [2] _[,i]_ (As _Q_ 4 _,i_ has the max weight of each head in it). Then


_∥_ _[Q]_ [4] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] (26)

( _Q_ 1 _,i_ + _Q_ 4 _,i_ )( _Q_ 2 _,i_ + _Q_ 4 _,i_ ) _[∥]_


_≤∥_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _∥_ (27)

( _Q_ 1 _,i_ + _Q_ 4 _,i_ )


_≤∥_ _[T]_ [0][(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _∥_ (28)

( _Q_ 1 _,i_ + _Q_ 3 _,i_ )


_≤∥_ _[T]_ [0] _[ϵ][k]_ [+1][(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _∥_ (29)

12 _T_ 0 [2]


_≤_ _[ϵ][k]_ [+1] (30)

6 _T_ 0


Similarly, we also have


[3] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [3] _[,i][ −]_ _[V]_ [2] _[,i]_ [)]

( _Q_ 1 _,i_ + _Q_ 3 _,i_ )( _Q_ 2 _,i_ + _Q_ 3 _,i_ ) _[∥≤]_ _[ϵ]_ 6 _[k]_ _T_ [+1] 0


_∥_ _[Q]_ [3] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [3] _[,i][ −]_ _[V]_ [2] _[,i]_ [)]


(31)
6 _T_ 0


From inequality 26 and substituting equation 24, we have


_∥_ _Q_ 1 _,i_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥≤_ _[ϵ][k]_ [+1]
_Q_ 1 _,i_ + _Q_ 3 _,i_ 6 _T_ 0


_[k]_ [+1]

+ _[ϵ][k]_ [+1]
6 _T_ 0 3 _T_ 0


_[k]_ [+1]

= _[ϵ][k]_ [+1]
3 _T_ 0 2 _T_ 0


(32)
2 _T_ 0


Therefore


Thus


_∥_ _Q_ 1 _,i_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (33)
_Q_ 1 _,i_ + _Q_ 4 _,i_

_T_ 0 _Q_ 1 _,i_
_≤∥_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (34)
_Q_ 1 _,i_ + _Q_ 3 _,i_

_≤_ _[ϵ][k]_ [+1] (35)

2


_∥_ _[Q]_ [1] _[,i][V]_ [1] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_


_[V]_ [1] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_ _−_ _[Q]_ [2] _[,i][V]_ [2] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_

_Q_ 1 _,i_ + _Q_ 4 _,i_ _Q_ 2 _,i_ + _Q_ 4 _,i_


_∥_ 2 (36)
_Q_ 2 _,i_ + _Q_ 4 _,i_


_≤∥_ _[Q]_ [4] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _Q_ 1 _,i_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (37)

( _Q_ 1 _,i_ + _Q_ 4 _,i_ )( _Q_ 2 _,i_ + _Q_ 4 _,i_ ) _[∥]_ [+] _[ ∥]_ _Q_ 1 _,i_ + _Q_ 4 _,i_


_≤_ _[ϵ][k]_ [+1]


_[k]_ [+1]

+ _[ϵ][k]_ [+1]
6 _T_ 0 2


(38)
2


_≤ϵ_ _[k]_ [+1] (39)


**Case 2** Case 2 can be rewritten as follows:


_Q_ _[V]_ 2 [2] _,i_ _[,i]_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_ _∥_ 2 _≤_ _[ϵ]_ 3 _[k]_ _T_ [+1] 0


- _∥_ _[Q]_ [1] _[,i]_ _Q_ _[V]_ [1] 1 _[,i]_ _,i_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_ _−_ _[Q]_ [2] _[,i]_ _Q_ _[V]_ 2 [2] _,i_ _[,i]_ [+] + _[Q]_ _Q_ [3] 3 _[,i]_ _,i_ _[V]_ [3] _[,i]_


3 _T_ 0 [.]


- _Q_ 1 _,i_ + _Q_ 3 _,i_ _≤_ _[ϵ][k]_ 4 [+1] - _sw_ =1 _[λ][i]_ [(] _[y][w][, t][w]_ [)] _[ ≤]_ _[ϵ][k]_ 4 [+1] _[Q]_ [4] _[,i]_ [.]


- _Q_ 2 _,i_ + _Q_ 3 _,i_ _≤_ _[ϵ][k]_ 4 [+1] - _sw_ =1 _[λ][i]_ [(] _[y][w][, t][w]_ [)] _[ ≤]_ _[ϵ][k]_ 4 [+1] _[Q]_ [4] _[,i]_ [.]


25


Thus


_∥_ _[Q]_ [1] _[,i][V]_ [1] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_


_∥_ 2 (40)
_Q_ 2 _,i_ + _Q_ 4 _,i_


_[V]_ [1] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_ _−_ _[Q]_ [2] _[,i][V]_ [2] _[,i]_ [ +] _[ Q]_ [4] _[,i][V]_ [4] _[,i]_

_Q_ 1 _,i_ + _Q_ 4 _,i_ _Q_ 2 _,i_ + _Q_ 4 _,i_


_≤∥_ _[Q]_ [4] _[,i]_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _Q_ 1 _,i_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (41)

( _Q_ 1 _,i_ + _Q_ 4 _,i_ )( _Q_ 2 _,i_ + _Q_ 4 _,i_ ) _[∥]_ [+] _[ ∥]_ _Q_ 1 _,i_ + _Q_ 4 _,i_


_≤∥_ [(] _[Q]_ [2] _[,i][ −]_ _[Q]_ [1] _[,i]_ [)(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)] _∥_ + _∥_ _Q_ 1 _,i_ ( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (42)

( _Q_ 2 _,i_ + _Q_ 4 _,i_ ) _Q_ 1 _,i_ + _Q_ 4 _,i_


_≤∥_ _[ϵ][k]_ [+1][(] _[V]_ [4] _[,i][ −]_ _[V]_ [2] _[,i]_ [)]


_[,i][ −]_ _[V]_ [2] _[,i]_ [)]

_∥_ + _∥_ _[ϵ][k]_ [+1]
4 4


( _V_ 1 _,i −_ _V_ 2 _,i_ ) _∥_ (43)
4


_≤ϵ_ _[k]_ [+1] (44)


And it can be seen from definition that


    - _Q_ 1 _,iQV_ 11 _,i,i_ ++ _QQ_ 44 _,i,iV_ 4 _,i_ is the output of the _i_ -th head of the attention layer with input sequence

_W_ 1. (which means _[Q]_ [1] _[,i]_ _Q_ _[V]_ [1] 1 _[,i]_ _,i_ [+] + _[Q]_ _Q_ [4] 4 _[,i]_ _,i_ _[V]_ [4] _[,i]_ = [�] _t_ _[T]_ =1 _[σ]_ �( _WK,iw_ ˆ1( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ1( _t_ )).


    - _Q_ 2 _,iQV_ 12 _,i,i_ ++ _QQ_ 44 _,i,iV_ 4 _,i_ is the output of the _i_ -th head of the attention layer with input sequence

_W_ 2. (which means _[Q]_ [2] _[,i]_ _Q_ _[V]_ [2] 2 _[,i]_ _,i_ [+] + _[Q]_ _Q_ [4] 4 _[,i]_ _,i_ _[V]_ [4] _[,i]_ = [�] _t_ _[T]_ =1 _[σ]_ �( _WK,iw_ ˆ2( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ2( _t_ )).


Then for each _i_ = 1 _, . . ., s_, we have that


_T_

- _σ_ �( _WK,iw_ ˆ2( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ2( _t_ ) _∥≤_ _ϵ_ _[k]_ [+1]


_t_ =1


_∥_


_T_

- _σ_ �( _WK,iw_ ˆ1( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ1( _t_ ) _−_


_t_ =1


(45)


Therefore, as _WO_ have entries bounded by 1, we have

_∥_       - _c_ ˆ0 + _WO_ Concat _[h]_ _i_ =1�� _[T]_ _σ_ �( _WK,iw_ ˆ1( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ1( _t_ )�� (46)

_t_ =1

_−_    - _c_ ˆ0 + _WO_ Concat _[h]_ _i_ =1�� _[T]_ _σ_ �( _WK,iw_ ˆ2( _t_ )) _[⊤]_ _WQ,ic_ ˆ0� _WV,iw_ ˆ2( _t_ )�� _∥_ (47)

_t_ =1

_≤sϵ_ _[k]_ [+1] (48)


However, it has been proven above that we need _|Model_ ( _W_ 1) _−_ _Model_ ( _W_ 2) _|_ _≥_ _ϵ_ to achieve _ϵ_ approximation of the target function. According to lemma 4, the required parameter count of the
FFN _F_ [ˆ] is of order Ω( _ϵ/ϵ_ _[k]_ [+1] ). Thus the parameter count required to achieve _ϵ_ -approximation is
Ω(1 _/ϵ_ _[k]_ ).


_Remark._ **Tightness of Theorem 2 (2)** The lower bound in Theorem 2 (2) remains essentially tight
under several relaxations of the feed-forward block _F_ [ˆ] . If _F_ [ˆ] uses Heaviside activations instead
of 1-Lipschitz activations, matching upper bounds can be constructed, but this case is impractical
since Heaviside activations are rarely used in practice. If parameter norms are permitted to scale
as _O_ ( _T_ [1] _[/ϵ]_ ), the parameter count can be reduced to _O_ (1 _/ϵ_ _[γ]_ [+1] ), though this scenario is likewise
unrealistic in practical settings. Finally, if _F_ [ˆ] is allowed up to five layers, the lower bound changes
to Ω(1 _/ϵ_ _[k/]_ [4] ), which does not alter the qualitative conclusion.


A.3 PROOF OF THEOREM 2 (3)


**Proof Sketch.** The argument is based on an explicit construction. We begin with trivial attention,
so that the post-attention output is simply the averaged concatenation _T_ [1] [(] _[x]_ [(1)] _[, . . ., x]_ [(] _[T]_ [))] _[∈]_ [R] _[T d]_ [.]

The feed-forward block can then be used to compute the transformations _fi_ ( _x_ ( _t_ )), perform the
necessary comparisons, and approximate _F_ 0, as ensured by Lemmas 9 and 8.


26


Having outlined the main idea, we now proceed to the detailed proof. As a first step, we introduce
several auxiliary lemmas that will be used in the argument.


**Lemma** **8.** _Fix_ _a_ _pointwise_ _activation_ _σ_ _(e.g.,_ _ReLU_ _or_ _any_ _activation_ _used_ _in_ _this_ _paper)._ _Let_
_F_ 1 : R _[m]_ [1] _→_ R _[m]_ [2] _be a_ 2 _-layer fully connected network, F_ 2 : R _[m]_ [2] _→_ R _[m]_ [3] _a_ 3 _-layer fully connected_
_network,_ _and_ _F_ 3 : R _[m]_ [3] _→_ R _a_ 2 _-layer_ _fully_ _connected_ _network._ _Let_ _W_ 1 _, W_ 2 _, W_ 3 _denote_ _their_
_respective_ _(maximum)_ _hidden_ _widths,_ _and_ _set_ _W_ := max _{W_ 1 _, W_ 2 _, W_ 3 _}._ _Then_ _there_ _exists_ _a_ 5 _-_
_layer fully connected network G_ : R _[m]_ [1] _→_ R _with activation σ_ _and hidden width at most W_ _such_
_that_
_G_ ( _x_ ) = _F_ 3� _F_ 2� _F_ 1( _x_ )�� _for all x ∈_ R _[m]_ [1] _._


_Proof._ Proof of Lemma 8 Write the three networks in affine–nonlinearity form (with a pointwise
activation _σ_ ):


_F_ 1( _x_ ) = _A_ 1 _σ_ ( _B_ 1 _x_ + _b_ 1) + _a_ 1 _,_ _x ∈_ R _[m]_ [1] _,_ _F_ 1( _x_ ) _∈_ R _[m]_ [2] _,_

_F_ 2( _u_ ) = _C_ 2 _σ_     - _D_ 2 _σ_ ( _E_ 2 _u_ + _e_ 2) + _d_ 2� + _c_ 2 _,_ _u ∈_ R _[m]_ [2] _,_ _F_ 2( _u_ ) _∈_ R _[m]_ [3] _,_

_F_ 3( _v_ ) = _p_ 3 _σ_ ( _Q_ 3 _v_ + _q_ 3) + _r_ 3 _,_ _v_ _∈_ R _[m]_ [3] _,_ _F_ 3( _v_ ) _∈_ R _._


Define a 5-layer fully connected network _G_ : R _[m]_ [1] _→_ R by stacking the hidden layers of _F_ 1 (one),
_F_ 2 (two), and _F_ 3 (one), keeping their original widths:


_h_ 1( _x_ ) := _σ_ ( _B_ 1 _x_ + _b_ 1) _,_
_u_ ( _x_ ) := _A_ 1 _h_ 1( _x_ ) + _a_ 1 _,_
_h_ 2( _x_ ) := _σ_ ( _E_ 2 _u_ ( _x_ ) + _e_ 2) _,_
_h_ 3( _x_ ) := _σ_ ( _D_ 2 _h_ 2( _x_ ) + _d_ 2) _,_
_v_ ( _x_ ) := _C_ 2 _h_ 3( _x_ ) + _c_ 2 _,_
_h_ 4( _x_ ) := _σ_ ( _Q_ 3 _v_ ( _x_ ) + _q_ 3) _,_
_G_ ( _x_ ) := _p_ 3 _h_ 4( _x_ ) + _r_ 3 _._


By construction,


_G_ ( _x_ ) = _p_ 3 _σ_ - _Q_ 3� _C_ 2 _σ_ ( _D_ 2 _σ_ ( _E_ 2( _A_ 1 _σ_ ( _B_ 1 _x_ + _b_ 1)+ _a_ 1)+ _e_ 2)+ _d_ 2)+ _c_ 2�+ _q_ 3�+ _r_ 3 = _F_ 3� _F_ 2� _F_ 1( _x_ )�� _._


Thus _G_ realizes the composition exactly, has 4 hidden layers (hence 5 layers total), and its hidden
widths are precisely those of the constituent hidden layers of _F_ 1, _F_ 2, and _F_ 3.


**Lemma** **9** (Approximating max with a shallow ReLU network) **.** _Let_ _f_ : [0 _,_ 1] _[T]_ _→_ R _be_
_f_ ( _x_ 1 _, . . ., xT_ ) = max _{x_ 1 _, . . ., xT }._ _For_ _any_ _ϵ_ _∈_ (0 _,_ 1] _,_ _there_ _exists_ _a_ _fully_ _connected_ _ReLU_ _net-_
_work_ _f_ [ˆ] _with_ three layers _(i.e., two hidden layers and one output layer), whose hidden-layer widths_
_are each at most_ 2 _T_ _⌈_ 1 _/ϵ⌉, such that_ _f_ [ˆ] _ϵ-approximates f_ _._


_Proof._ Proof of Lemma 9 Let
_n_ = _⌈_ 1 _/ϵ⌉._


For each coordinate _t_ _∈_ [ _T_ ] and each grid index _i_ = 0 _,_ 1 _, . . ., n −_ 1, define the first hidden layer
neurons by
_h_ 1( _t, i_ ) = ReLU� _xt −_ _n_ _[i]_             - _._

For each _j_ = 0 _,_ 1 _, . . ., n −_ 1, define the second hidden layer neurons by


- _T_

 


_h_ 1( _t, j_ ) _−_ _n_ [1]
_t_ =1


_n_


_h_ 2( _j_ ) = ReLU


- _T_

 


_h_ 1( _t, j_ )

_t_ =1


_−_ ReLU


_._


Finally, the output of the network is given by


_f_ ˆ( _x_ 1 _, . . ., xT_ ) =


27


_n−_ 1

- _h_ 2( _j_ ) _._


_j_ =0


_Claim._ Fix _j_ _∈{_ 0 _, . . ., n −_ 1 _}_ and set


_T_


- ReLU� _xt −_ _[j]_

_n_

_t_ =1


_Sj_ =


_T_

- _h_ 1( _t, j_ ) =


_t_ =1


_n_


_._


By definition,


           _h_ 2( _j_ ) = ReLU( _Sj_ ) _−_ ReLU _Sj_ _−_ [1]

_n_


           _h_ 2( _j_ ) = ReLU( _Sj_ ) _−_ ReLU _Sj_ _−_ [1]


_._


1) If _h_ 2( _j_ ) _>_ 0, then necessarily _Sj_ _>_ 0 (since ReLU( _z_ ) _>_ 0 iff _z_ _>_ 0), hence there exists some _t_
with
ReLU� _xt −_ _[j]_          - _>_ 0 _⇐⇒_ _xt_ _>_ _[j]_ _[.]_


_n_ _[j]_ - _>_ 0 _⇐⇒_ _xt_ _>_ _n_ _[j]_


_n_ _[.]_


Thus _h_ 2( _j_ ) _>_ 0 only if _∃_ _t_ with _xt_ _> j/n_ .


2) If there exists _t_ with _xt_ _>_ ( _j_ + 1) _/n_, then

_Sj_ _≥_ ReLU� _xt −_ _n_ _[j]_               - _>_ _n_ 1 _[.]_


Therefore _Sj_ _≥_ _n_ [1] [, and we get]

_h_ 2( _j_ ) = _Sj_ _−_            - _Sj_ _−_ _n_ [1]            - = _n_ 1 _[.]_


Fix _x ∈_ [0 _,_ 1] _[T]_ and let _j_ be such that max _t xt_ _∈_ ( _j/n,_ ( _j_ + 1) _/n_ ]. By construction,

_h_ 2( _k_ ) = 0 for _k_ _≥_ _j_ + 1 _,_ _h_ 2( _k_ ) = _n_ [1] for _k_ _≤_ _j −_ 1 _,_

and for _k_ = _j_ we have


 _n_ _[j]_ _._


0 _≤_ _h_ 2( _j_ ) = ReLU( _Sj_ ) _−_ ReLU� _Sj_ _−_ _n_ [1] - _≤_ _n_ 1 _[,]_ _Sj_ :=


_T_


- ReLU� _xt −_ _n_ _[j]_

_t_ =1


Hence


_j−_ 1


_k_ =0


_f_ ˆ( _x_ ) =


_n−_ 1

- _h_ 2( _k_ ) =


_k_ =0


1 - _j_ _[j]_
_n_ [+] _[ h]_ [2][(] _[j]_ [)] _[ ∈]_ _n_ _[,]_


_[j]_ [1]

_n_ [+] _n_


_n_ [1] - = - _nj_ _[,]_ _[j]_ [+1] _n_


[+1] _n_ - _._


Since max _t xt_ _∈_ ( _j/n,_ ( _j_ + 1) _/n_ ], it follows that

0 _≤|f_ [ˆ] ( _x_ ) _−_ max _t_ _xt|_ _≤_ _n_ 1 _[≤]_ _[ϵ.]_

Therefore _f_ [ˆ] _ϵ_ -approximates _f_ ( _x_ ) = max _t xt_ on [0 _,_ 1] _[T]_ .


_Proof._ Theorem 2 (3)
We begin by fixing the embedding with positional information. Let _Pϕ_ : [0 _,_ 1] _[d]_ _×_ [ _T_ ] _→_ R _[dT]_ be
defined by
_Pϕ_ ( _x_ ( _t_ ) _, t_ ) = (0 _, . . .,_ 0 _,_ _x_ ( _t_ ) _,_ 0 _, . . .,_ 0) _,_
where the vector _x_ ( _t_ ) occupies the _t_ -th block of dimension _d_, and all other blocks are zero. With the
classification token ˆ _c_ 0 = 0, the attention layer reduces to a trivial aggregation, and the output (prior
to the feed-forward network) is


1
_T_ [(] _[x]_ [(1)] _[, . . ., x]_ [(] _[T]_ [))] _[ ∈]_ [[0] _[,]_ [ 1]] _[dT][ .]_

Given a target accuracy _ϵ >_ 0, we construct three feed-forward networks _F_ 1 _, F_ 2 _, F_ 3 as follows.


**Step 1:** **Approximating the component functions.** Define


_T_ [1] [[0] _[,]_ [ 1]] _[d][×][T]_ _[→]_ [R] _[D][×][T][,]_ _F_ 1� _T_ 1 _[x]_ [(1)] _[, . . .,]_ _T_ [1]


_F_ 1 : [1]


_T_ [1] _[x]_ [(] _[T]_ [)] - = ( _u_ (1) _, . . ., u_ ( _T_ )) _,_


where each _u_ ( _t_ ) _∈_ R _[D]_ satisfies


_|u_ ( _t_ ) _i −_ _fi_ ( _x_ ( _t_ )) _| ≤_ _ϵ_ for all _i_ = 1 _, . . ., D._


By Assumption 3, such an approximation can be implemented by a two-layer FFN with parameter
count _O_ (1 _/ϵ_ _[γ]_ ).


28


**Step 2:** **Approximating the minimization.** Let _F_ 2 _[′]_ [:][ R] _[D][×][T]_ _[→]_ [R] _[D]_ [be defined by]

_F_ 2 _[′]_ [(] _[u]_ [(1)] _[, . . ., u]_ [(] _[T]_ [)) = (] _[u]_ [1] _[, . . ., u][D]_ [)] _[,]_ _ui_ = min
_t∈Si_ _[u]_ [(] _[t]_ [)] _[i][.]_


By Lemma 9 (which works the same for taking minimum), there exists a three-layer ReLU network
with _O_ (1 _/ϵ_ ) parameters that _ϵ_ -approximates _F_ 2 _[′]_ [.] [We denote this approximation by] _[ F]_ [2][.]


**Step 3:** **Approximating the outer function.** Finally, let _F_ 3 : R _[D]_ _→_ R be a two-layer FFN that
_ϵ_ -approximates _F_ 0, with parameter count _O_ (1 _/ϵ_ _[γ]_ ).


**Composition.** Since _F_ 0 is _C_ [1] on a compact domain, it is Lipschitz with constant _L_, and the min
operator is 1-Lipschitz. Therefore, the composed network


_F_ 3 _◦_ _F_ 2 _◦_ _F_ 1


provides an _Lϵ_ -approximation of the target function, with total parameter count


_O_ (1 _/ϵ_ _[γ]_ [+1] ) _._


According to lemma 8, _F_ 3 _◦_ _F_ 2 _◦_ _F_ 1 can be written equivalently as a five-layer FFN.


A.4 PROOF OF COROLLARY 1


**Proof Sketch.** It is a direct corollary of Theorem 2 (1) and 2 (2).


_Proof._ Suppose _D_ 1 _<_ _D_ 2. Let _M_ 0 be the minimal positive integer such that _H_ ( _D_ 1 _,_ 2 _, d, T, M_ 0)
_ϵ_ -approximates _H_ . Then with representation ( _{fi, Si}_ _[D]_ _i_ =1 [1] _[, F]_ [0][)][,] [Theorem] [2] [(1)] [suggests] [that] [there]
exists a positive _Cd,D_ 1 _,T_ such that

_M_ 0 _≤_ _[C][d,D]_ [1] _[,T]_ (49)

_ϵ_ _[γ]_


With representation ( _{f_ [˜] _i,_ _S_ [˜] _i}_ _[D]_ _i_ =1 [2] _[,]_ _[F]_ [˜][0][)][,] [Theorem] [2] [(2)] [suggests] [that] [there] [exists] [a] [positive] _[C][d,D]_ 2 _[,T]_
such that


[2] _[,T]_ for _k_ = [(] [1] 4

_ϵ_ _[k]_


_−_ 1 (50)
3 _D_ 1 + 1


_M_ 0 _≥_ _[C][d,D]_ [2] _[,T]_


4 _[T]_ _[−]_ _[D]_ [1] _[−]_ _[D]_ [2] [+ 1)]


As _fi_ and _F_ 0 are at least _C_ [1] smooth, we have _γ_ _≤_ max( _D_ 1 _, D_ 2). Thus with _D_ 1 [2] [+] _[ D]_ 2 [2] _[≤]_ 501 _[T]_ [, we]
have _k_ _> γ_ . Then there exist _ϵ >_ 0 such that


_Cd,D_ 2 _,T_


(51)
_ϵ_ _[γ]_


_d,D_ 2 _,T_ _>_ _[C][d,D]_ [1] _[,T]_

_ϵ_ _[k]_ _ϵ_ _[γ]_


This leads to a contradiction. Thus _D_ 1 = _D_ 2.


29


B EXPERIMENT DETAILS


B.1 DETAILS FOR EXPERIMENT 1


B.1.1 EXPERIMENTAL DETAILS FOR SECTION 6.1


**Data generation.** The intrinsic dimension of the synthetic task is _D_ = 4. For each sequence length
_T_ _∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ we generate 8000 training and 2000 validation examples. The inputs are
i.i.d. Gaussian samples _x_ ( _t_ ) _∼N_ (0 _, I_ 4).


**Model** **architecture.** Each input vector _x_ ( _t_ ) is first mapped to R [8] _[h]_ by a two-layer feed-forward
network with hidden dimension _N_ and ReLU activations, ensuring a per-head embedding dimension of 8. A trainable classification token _c_ 0 is appended, and no positional encoding is used since
the task is permutation invariant. The sequence is processed by a single-layer multi-head attention
block without residual connections or layer normalization, consistent with the theoretical setting.
The output is concatenated and passed through a two-layer GeLU-activated feed-forward network
with hidden dimension _N_, yielding the final scalar prediction. The fixed hidden size ensures comparability of parameter counts across different _h_ .


**Training protocol.** Each configuration ( _h, T_ ) is trained separately under multiple random seeds.
To reduce the effect of optimization variance, we report the _minimal validation error_ achieved across
seeds. This choice isolates expressivity limitations of the architecture from randomness in training
dynamics.


**Evaluation** **metric.** We adopt the normalized mean squared error (NMSE), defined as mean
squared error divided by the variance of the targets. As _T_ increases, maxima of Gaussian samples concentrate, shrinking target variance and making trivial predictors appear competitive under
raw MSE. (An intuition is that suppose the target output be _YT_ = max1 _≤t≤T_ _xt_ with input tokens
_xt_ _∼N_ (0 _,_ 1) independently, then Var( _YT_ ) _decreases_ as _T_ increases, because _YT_ concentrates more
tightly around its growing mean.)Normalization by variance corrects this effect and ensures comparability across lengths. NMSE is also equivalent to 1 _−_ _R_ [2], where _R_ [2] is the standard coefficient of
determination.


**Variance** **across** **seeds.** While mean performance across seeds is also informative, reporting the
minimal validation NMSE highlights the best achievable accuracy for a given architecture. This emphasizes limitations due to model capacity rather than training noise. Tables showing seed variance
are included for completeness (Table 2).


_Remark._ When _h_ _≥_ _D_ = 4, we also observe that the validation NMSE first decreases rapidly and
then increases slowly as _T_ grows. For shorter sequences, the model with enough heads can either
capture the pattern through attention (Theorem 2 (1)) or rely on a memorization-based strategy with
the feed-forward network (Theorem 2 (3)). Both approaches generalize reasonably well, but the
memorization-based one does so less effectively. For longer sequences, memorization becomes
infeasible and the model relies on attention, which generalizes better; however, longer sequences
may also be more sensitive to parameterization, and the observed curve likely reflects a tradeoff
between these effects. See Figure 3 in Appendix.


B.1.2 FIGURES AND TABLES FOR SYNTHETIC EXPERIMENT 6.1


|Heads|T=8 T=16 T=32 T=64 T=128|
|---|---|
|1<br>2<br>3<br>4<br>5|7_._01_ ×_ 10_−_2 _±_ 5_._99_ ×_ 10_−_2<br>1_._09_ ×_ 10_−_1 _±_ 9_._93_ ×_ 10_−_2<br>1_._10_ ×_ 10_−_1 _±_ 9_._36_ ×_ 10_−_2<br>1_._14_ ×_ 10_−_1 _±_ 8_._53_ ×_ 10_−_2<br>1_._45_ ×_ 10_−_1 _±_ 1_._05_ ×_ 10_−_1<br>7_._31_ ×_ 10_−_3 _±_ 4_._75_ ×_ 10_−_4<br>8_._41_ ×_ 10_−_3 _±_ 7_._97_ ×_ 10_−_4<br>9_._42_ ×_ 10_−_3 _±_ 6_._42_ ×_ 10_−_4<br>1_._31_ ×_ 10_−_2 _±_ 1_._16_ ×_ 10_−_2<br>1_._47_ ×_ 10_−_2 _±_ 1_._21_ ×_ 10_−_2<br>6_._94_ ×_ 10_−_4 _±_ 2_._90_ ×_ 10_−_4<br>6_._40_ ×_ 10_−_4 _±_ 3_._87_ ×_ 10_−_4<br>9_._09_ ×_ 10_−_4 _±_ 4_._31_ ×_ 10_−_4<br>1_._31_ ×_ 10_−_3 _±_ 5_._10_ ×_ 10_−_4<br>1_._58_ ×_ 10_−_3 _±_ 5_._21_ ×_ 10_−_4<br>6_._10_ ×_ 10_−_5 _±_ 1_._52_ ×_ 10_−_4<br>4_._36_ ×_ 10_−_5 _±_ 1_._93_ ×_ 10_−_4<br>4_._80_ ×_ 10_−_5 _±_ 2_._30_ ×_ 10_−_4<br>8_._75_ ×_ 10_−_6 _±_ 5_._58_ ×_ 10_−_5<br>5_._23_ ×_ 10_−_6 _±_ 5_._67_ ×_ 10_−_6<br>3_._35_ ×_ 10_−_5 _±_ 5_._84_ ×_ 10_−_5<br>1_._10_ ×_ 10_−_5 _±_ 2_._36_ ×_ 10_−_5<br>4_._91_ ×_ 10_−_6 _±_ 6_._32_ ×_ 10_−_6<br>4_._19_ ×_ 10_−_6 _±_ 8_._39_ ×_ 10_−_6<br>3_._99_ ×_ 10_−_6 _±_ 4_._29_ ×_ 10_−_6|


Table 2: Error bar for synthetic dataset. NMSE(Mean ± Standard Deviation) for different sequence
lengths _T_ and number of heads.


30


8
6

4


2


0


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
|Nu|||||||
|Nu|mber of|Heads<br>4<br>5|Heads<br>4<br>5|Heads<br>4<br>5|Heads<br>4<br>5|Heads<br>4<br>5|
|Nu|||||||


Sequence Length


Figure 3: A zoom in plot of Figure1, which shows that when the number of head is enough, the loss
first decreases and then increases, as explained in the remark B.1.1


4.0


3.5


3.0


2.5


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||


Log Min Validation NMSE


4.0


3.5


3.0


2.5


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


Log Min Validation NMSE


Figure 4: Additional plot of 1b for _H_ = 1 and _H_ = 2.

|Heads|T=8 T=16 T=32 T=64 T=128|
|---|---|
|1<br>2<br>3<br>4<br>5|1_._75_ ×_ 10_−_2<br>1_._98_ ×_ 10_−_2<br>2_._06_ ×_ 10_−_2<br>2_._54_ ×_ 10_−_2<br>3_._03_ ×_ 10_−_2<br>7_._17_ ×_ 10_−_3<br>7_._39_ ×_ 10_−_3<br>7_._82_ ×_ 10_−_3<br>8_._57_ ×_ 10_−_3<br>1_._02_ ×_ 10_−_2<br>2_._11_ ×_ 10_−_4<br>2_._17_ ×_ 10_−_4<br>2_._73_ ×_ 10_−_4<br>3_._71_ ×_ 10_−_4<br>4_._77_ ×_ 10_−_4<br>1_._32_ ×_ 10_−_6<br>5_._59_ ×_ 10_−_7<br>3_._40_ ×_ 10_−_7<br>3_._46_ ×_ 10_−_7<br>5_._70_ ×_ 10_−_7<br>2_._19_ ×_ 10_−_6<br>4_._33_ ×_ 10_−_7<br>3_._22_ ×_ 10_−_7<br>2_._73_ ×_ 10_−_7<br>2_._66_ ×_ 10_−_7|


Table 3: Validation NMSE under fixed total embedding dimension _E_ = _nh_ = 32.

|Heads|T=8 T=16 T=32 T=64 T=128|
|---|---|
|1<br>2<br>3<br>4|1_._38_ ×_ 10_−_2<br>1_._63_ ×_ 10_−_2<br>1_._84_ ×_ 10_−_2<br>2_._17_ ×_ 10_−_2<br>2_._31_ ×_ 10_−_2<br>1_._09_ ×_ 10_−_3<br>7_._08_ ×_ 10_−_4<br>7_._24_ ×_ 10_−_4<br>7_._76_ ×_ 10_−_4<br>1_._11_ ×_ 10_−_3<br>4_._18_ ×_ 10_−_7<br>1_._72_ ×_ 10_−_7<br>1_._17_ ×_ 10_−_7<br>3_._58_ ×_ 10_−_7<br>2_._11_ ×_ 10_−_7<br>5_._56_ ×_ 10_−_7<br>1_._22_ ×_ 10_−_7<br>6_._89_ ×_ 10_−_8<br>1_._85_ ×_ 10_−_7<br>3_._48_ ×_ 10_−_7|


Table 4: Approximation error for the _D_ = 3 retrieval task under fixed total embedding dimension
_E_ = _nh_ .


31


|Heads|T=8 T=16 T=32 T=64 T=128|
|---|---|
|1<br>2<br>3<br>4<br>5|2_._12_ ×_ 10_−_4<br>1_._85_ ×_ 10_−_4<br>2_._22_ ×_ 10_−_4<br>3_._13_ ×_ 10_−_4<br>4_._28_ ×_ 10_−_4<br>7_._22_ ×_ 10_−_6<br>2_._69_ ×_ 10_−_6<br>3_._50_ ×_ 10_−_6<br>3_._07_ ×_ 10_−_6<br>3_._83_ ×_ 10_−_6<br>8_._16_ ×_ 10_−_6<br>1_._83_ ×_ 10_−_6<br>1_._73_ ×_ 10_−_6<br>3_._86_ ×_ 10_−_6<br>3_._50_ ×_ 10_−_6<br>3_._68_ ×_ 10_−_6<br>1_._87_ ×_ 10_−_6<br>2_._60_ ×_ 10_−_6<br>4_._32_ ×_ 10_−_6<br>5_._98_ ×_ 10_−_6<br>6_._15_ ×_ 10_−_6<br>3_._02_ ×_ 10_−_6<br>3_._31_ ×_ 10_−_6<br>3_._78_ ×_ 10_−_6<br>5_._34_ ×_ 10_−_6|


Table 5: Two-layer transformer on the synthetic task ( _D_ = 4, NoPE, NoLN, fixed total embedding
dimension _E_ = _nh_ = 32).


B.2 MS MARCO TEXT RETRIEVAL


B.2.1 EXPERIMENT DETAILS FOR MS MARCO (TEXT RETRIEVAL) EXPERIMENT


**Dataset construction.** We construct retrieval-style datasets from the MS MARCO passage ranking collection (Bajaj et al., 2016). Since the original dataset associates each query with only a few
candidate passages, we enlarge the candidate set by mining hard negatives. Specifically, BM25
(Robertson & Zaragoza, 2009) is used to mine local negatives and FAISS (Johnson et al., 2019) similarity search to retrieve global negatives, reducing redundancy across queries. For each query, the
sequence length _T_ is defined as the total number of candidates (one positive and _T_ _−_ 1 negatives),
with _T_ _∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ . We build datasets containing 28 _,_ 000 training queries and 2 _,_ 000 validation
queries for each _T_ .


**Model and training setup.** We evaluate a two-layer Transformer encoder with per-head embedding dimension fixed at 32, while varying the number of heads across _{_ 1 _,_ 2 _,_ 4 _,_ 6 _,_ 8 _,_ 10 _,_ 12 _,_ 14 _,_ 16 _}_ .
Tokenization and input embeddings follow the BERT tokenizer and frozen BERT word, position,
and segment embeddings (Devlin et al., 2019), projected to the model hidden size _h_ = heads _×_ 32.
Only the projection and Transformer layers are trained. We report training top-1 accuracy, focusing
on training performance since MS MARCO with BM25-mined negatives is particularly challenging
for validation, and the difference can be seen in training metrics. Training MRR is also reported in
Fig 5, with similar trend as training accuracy.


B.2.2 FIGURES AND TABLES FOR EXPERIMENT

|Heads|T=8 T=16 T=32 T=64|
|---|---|
|1<br>2<br>4<br>6<br>8<br>12<br>16|0_._597_ ±_ 0_._003<br>0_._450_ ±_ 0_._005<br>0_._303_ ±_ 0_._003<br>0_._154_ ±_ 0_._002<br>0_._771_ ±_ 0_._003<br>0_._647_ ±_ 0_._003<br>0_._486_ ±_ 0_._002<br>0_._286_ ±_ 0_._002<br>0_._956_ ±_ 0_._002<br>0_._900_ ±_ 0_._002<br>0_._793_ ±_ 0_._002<br>0_._580_ ±_ 0_._004<br>0_._992_ ±_ 0_._000<br>0_._977_ ±_ 0_._001<br>0_._937_ ±_ 0_._001<br>0_._814_ ±_ 0_._002<br>0_._998_ ±_ 0_._000<br>0_._995_ ±_ 0_._000<br>0_._983_ ±_ 0_._001<br>0_._932_ ±_ 0_._002<br>1_._000_ ±_ 0_._000<br>0_._999_ ±_ 0_._000<br>0_._998_ ±_ 0_._000<br>0_._991_ ±_ 0_._001<br>1_._000_ ±_ 0_._000<br>1_._000_ ±_ 0_._000<br>0_._999_ ±_ 0_._000<br>0_._996_ ±_ 0_._000|


Table 6: Error bar for MS Marco dataset. Accuracy (Mean ± Standard Deviation) for different
sequence lengths _T_ and number of heads.


B.3 CIFAR-10 IMAGE CLASSIFICATION


B.3.1 DATASET CONSTRUCTION


We create image classification datasets from the CIFAR-10 dataset using a padded preprocessing
approach. The original CIFAR-10 images have dimensions of 32 _×_ 32 pixels. To generate datasets
with larger image sizes, we apply padding to achieve sizes in the set _{_ 32 _,_ 48 _,_ 64 _,_ 96 _,_ 128 _}_ . The
original image is randomly positioned within the enlarged frame, with padding filled using the colors
of the border pixels. An illustration is provided in Figure 6. By apply this padding method we are
creating tasks with increasing difficulty. The background is enlarged, making models need more


32


|Heads|T=8 T=16 T=32 T=64|
|---|---|
|1<br>2<br>4<br>6<br>8<br>12<br>16|0_._5107_ ±_ 0_._0069<br>0_._3917_ ±_ 0_._0071<br>0_._2542_ ±_ 0_._0049<br>0_._1257_ ±_ 0_._0057<br>0_._5221_ ±_ 0_._0102<br>0_._4205_ ±_ 0_._0056<br>0_._2712_ ±_ 0_._0067<br>0_._1369_ ±_ 0_._0038<br>0_._5076_ ±_ 0_._0112<br>0_._4048_ ±_ 0_._0070<br>0_._2547_ ±_ 0_._0093<br>0_._1139_ ±_ 0_._0061<br>0_._5153_ ±_ 0_._0112<br>0_._3865_ ±_ 0_._0103<br>0_._2397_ ±_ 0_._0098<br>0_._1018_ ±_ 0_._0057<br>0_._5058_ ±_ 0_._0082<br>0_._3801_ ±_ 0_._0084<br>0_._2308_ ±_ 0_._0068<br>0_._0983_ ±_ 0_._0050<br>0_._5184_ ±_ 0_._0073<br>0_._3721_ ±_ 0_._0107<br>0_._2219_ ±_ 0_._0091<br>0_._0902_ ±_ 0_._0054<br>0_._5021_ ±_ 0_._0123<br>0_._2816_ ±_ 0_._0418<br>0_._2170_ ±_ 0_._0075<br>0_._0878_ ±_ 0_._0058|


Table 7: MS Marco Validation Accuracy (Mean ± Standard Deviation) for different sequence lengths
_T_ and number of heads.


1.0


0.8


0.6


0.4


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
||||||<br>T=8<br>T=16<br>|||
||||||~~T=32~~<br>T=64|||
|||||||||


Number of Heads


Figure 5: Plot of training mrr for MS MARCO dataset.


effort to learn how to extract useful information. The random placement make sure the padded
outside aera cannot be simply ignored by position encodings.


Each image is divided into non-overlapping patches of size 8 _×_ 8 pixels, resulting in a sequence of
patches for each image. For each image size, the sequence length _T_ is defined as the total number
of patches plus one additional class token, with _T_ = _{_ 17 _,_ 37 _,_ 65 _,_ 145 _,_ 257 _}_ . We adopt the standard
CIFAR-10 data splits, which include 50 _,_ 000 training images and 10 _,_ 000 test images across 10
classes.


B.3.2 MODEL TRAINING SETUP


We evaluate a Vision Transformer (ViT) with four layers and a per-head embedding dimension of
16, while varying the number of attention heads in different configurations. Each image patch is
embedded through a linear projection, and positional embeddings are added along with a learnable
class token. No global convolutional embedding is used.


Input processing follows the standard ViT procedure, including patch embedding of size 8 _×_ 8, positional encoding, and aggregation of the class token for final classification. The model is trained
using the AdamW optimizer with cosine annealing learning rate scheduling. Standard architectural
techniques, such as layer normalization, residual connections, and dropout, are employed for regularization.


33


Figure 6: Examples of the padded images from the dataset.


**Heads** **Seq=65** **Seq=145** **Seq=257** **Seq=577** **Seq=1025**
1 4 _._ 78 _×_ 10 [1] _±_ 4 _._ 50 _×_ 10 _[−]_ [1] 4 _._ 37 _×_ 10 [1] _±_ 4 _._ 70 _×_ 10 _[−]_ [1] 4 _._ 20 _×_ 10 [1] _±_ 5 _._ 00 _×_ 10 _[−]_ [1] 4 _._ 08 _×_ 10 [1] _±_ 6 _._ 50 _×_ 10 _[−]_ [1] 4 _._ 00 _×_ 10 [1] _±_ 7 _._ 20 _×_ 10 _[−]_ [1]

2 5 _._ 97 _×_ 10 [1] _±_ 4 _._ 50 _×_ 10 _[−]_ [1] 5 _._ 52 _×_ 10 [1] _±_ 3 _._ 80 _×_ 10 _[−]_ [1] 5 _._ 34 _×_ 10 [1] _±_ 4 _._ 40 _×_ 10 _[−]_ [1] 5 _._ 15 _×_ 10 [1] _±_ 4 _._ 70 _×_ 10 _[−]_ [1] 5 _._ 08 _×_ 10 [1] _±_ 7 _._ 40 _×_ 10 _[−]_ [1]

4 7 _._ 55 _×_ 10 [1] _±_ 2 _._ 10 _×_ 10 _[−]_ [1] 7 _._ 03 _×_ 10 [1] _±_ 3 _._ 20 _×_ 10 _[−]_ [1] 6 _._ 85 _×_ 10 [1] _±_ 7 _._ 70 _×_ 10 _[−]_ [1] 6 _._ 62 _×_ 10 [1] _±_ 6 _._ 00 _×_ 10 _[−]_ [1] 6 _._ 58 _×_ 10 [1] _±_ 7 _._ 20 _×_ 10 _[−]_ [1]

8 9 _._ 51 _×_ 10 [1] _±_ 1 _._ 50 _×_ 10 _[−]_ [1] 9 _._ 26 _×_ 10 [1] _±_ 4 _._ 70 _×_ 10 _[−]_ [1] 9 _._ 14 _×_ 10 [1] _±_ 6 _._ 20 _×_ 10 _[−]_ [1] 9 _._ 07 _×_ 10 [1] _±_ 1 _._ 02 _×_ 10 [0] 9 _._ 02 _×_ 10 [1] _±_ 1 _._ 00 _×_ 10 [0]

10 9 _._ 81 _×_ 10 [1] _±_ 5 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 73 _×_ 10 [1] _±_ 2 _._ 40 _×_ 10 _[−]_ [1] 9 _._ 67 _×_ 10 [1] _±_ 4 _._ 60 _×_ 10 _[−]_ [1] 9 _._ 65 _×_ 10 [1] _±_ 3 _._ 20 _×_ 10 _[−]_ [1] 9 _._ 67 _×_ 10 [1] _±_ 7 _._ 30 _×_ 10 _[−]_ [1]

11 9 _._ 88 _×_ 10 [1] _±_ 1 _._ 20 _×_ 10 _[−]_ [1] 9 _._ 83 _×_ 10 [1] _±_ 1 _._ 20 _×_ 10 _[−]_ [1] 9 _._ 81 _×_ 10 [1] _±_ 2 _._ 40 _×_ 10 _[−]_ [1] 9 _._ 77 _×_ 10 [1] _±_ 2 _._ 10 _×_ 10 _[−]_ [1] 9 _._ 79 _×_ 10 [1] _±_ 2 _._ 20 _×_ 10 _[−]_ [1]

12 9 _._ 92 _×_ 10 [1] _±_ 3 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 89 _×_ 10 [1] _±_ 1 _._ 60 _×_ 10 _[−]_ [1] 9 _._ 86 _×_ 10 [1] _±_ 2 _._ 40 _×_ 10 _[−]_ [1] 9 _._ 86 _×_ 10 [1] _±_ 1 _._ 90 _×_ 10 _[−]_ [1] 9 _._ 86 _×_ 10 [1] _±_ 2 _._ 90 _×_ 10 _[−]_ [1]

13 9 _._ 94 _×_ 10 [1] _±_ 6 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 93 _×_ 10 [1] _±_ 6 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 91 _×_ 10 [1] _±_ 1 _._ 10 _×_ 10 _[−]_ [1] 9 _._ 90 _×_ 10 [1] _±_ 2 _._ 00 _×_ 10 _[−]_ [1] 9 _._ 91 _×_ 10 [1] _±_ 2 _._ 50 _×_ 10 _[−]_ [1]

14 9 _._ 96 _×_ 10 [1] _±_ 3 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 94 _×_ 10 [1] _±_ 9 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 93 _×_ 10 [1] _±_ 1 _._ 00 _×_ 10 _[−]_ [1] 9 _._ 93 _×_ 10 [1] _±_ 1 _._ 70 _×_ 10 _[−]_ [1] 9 _._ 93 _×_ 10 [1] _±_ 2 _._ 30 _×_ 10 _[−]_ [1]

16 9 _._ 97 _×_ 10 [1] _±_ 3 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 96 _×_ 10 [1] _±_ 2 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 95 _×_ 10 [1] _±_ 7 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 96 _×_ 10 [1] _±_ 1 _._ 40 _×_ 10 _[−]_ [1] 9 _._ 96 _×_ 10 [1] _±_ 1 _._ 70 _×_ 10 _[−]_ [1]

20 9 _._ 98 _×_ 10 [1] _±_ 1 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 97 _×_ 10 [1] _±_ 5 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 97 _×_ 10 [1] _±_ 8 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 98 _×_ 10 [1] _±_ 6 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 99 _×_ 10 [1] _±_ 7 _._ 00 _×_ 10 _[−]_ [2]

24 9 _._ 99 _×_ 10 [1] _±_ 2 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 98 _×_ 10 [1] _±_ 2 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 98 _×_ 10 [1] _±_ 4 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 99 _×_ 10 [1] _±_ 4 _._ 00 _×_ 10 _[−]_ [2] 9 _._ 99 _×_ 10 [1] _±_ 5 _._ 00 _×_ 10 _[−]_ [2]


Table 8: Error bar for Image task. Accuracy (Mean ± Standard Deviation) for different sequence
lengths and number of heads.


C LARGE LANGUAGE MODEL USAGE


Large language models were used only for linguistic refinement (e.g., polishing sentences and checking grammar). The core ideas, theoretical results, experimental design, and analyses presented in
this paper were entirely developed by the authors without assistance from large language models.


34


|Heads|T=65 T=145 T=257 T=577 T=1025|
|---|---|
|1<br>2<br>4<br>8<br>10<br>11<br>12<br>13<br>14<br>16<br>20<br>24|50_._50_ ±_ 0_._44<br>45_._85_ ±_ 0_._58<br>43_._53_ ±_ 0_._58<br>41_._43_ ±_ 0_._97<br>39_._95_ ±_ 0_._75<br>60_._01_ ±_ 0_._57<br>55_._01_ ±_ 0_._25<br>53_._12_ ±_ 0_._69<br>49_._95_ ±_ 0_._80<br>48_._53_ ±_ 0_._83<br>67_._98_ ±_ 0_._43<br>63_._49_ ±_ 0_._70<br>61_._42_ ±_ 0_._61<br>57_._19_ ±_ 0_._54<br>55_._31_ ±_ 0_._85<br>69_._65_ ±_ 0_._55<br>66_._06_ ±_ 0_._53<br>62_._43_ ±_ 0_._95<br>57_._58_ ±_ 0_._59<br>56_._18_ ±_ 1_._14<br>69_._70_ ±_ 0_._24<br>65_._64_ ±_ 0_._37<br>62_._45_ ±_ 0_._49<br>57_._21_ ±_ 1_._03<br>54_._44_ ±_ 1_._49<br>69_._84_ ±_ 0_._36<br>65_._26_ ±_ 0_._43<br>61_._97_ ±_ 0_._29<br>56_._95_ ±_ 0_._63<br>53_._77_ ±_ 2_._91<br>69_._66_ ±_ 0_._39<br>65_._79_ ±_ 0_._56<br>62_._63_ ±_ 0_._36<br>56_._57_ ±_ 0_._66<br>53_._17_ ±_ 1_._06<br>69_._72_ ±_ 0_._18<br>65_._30_ ±_ 0_._59<br>61_._66_ ±_ 0_._58<br>54_._81_ ±_ 2_._23<br>52_._90_ ±_ 1_._45<br>69_._49_ ±_ 0_._48<br>65_._25_ ±_ 0_._48<br>61_._32_ ±_ 0_._95<br>54_._01_ ±_ 2_._13<br>50_._42_ ±_ 2_._48<br>69_._69_ ±_ 0_._33<br>64_._24_ ±_ 0_._33<br>59_._29_ ±_ 1_._12<br>49_._68_ ±_ 1_._39<br>48_._51_ ±_ 2_._59<br>67_._99_ ±_ 0_._35<br>61_._49_ ±_ 1_._07<br>55_._25_ ±_ 2_._12<br>48_._65_ ±_ 0_._85<br>46_._89_ ±_ 1_._07<br>65_._12_ ±_ 0_._68<br>56_._80_ ±_ 1_._09<br>52_._14_ ±_ 0_._56<br>48_._74_ ±_ 0_._60<br>48_._39_ ±_ 0_._76|


Table 9: Error bar for Image task. Validation Accuracy (Mean ± Standard Deviation) for different
sequence lengths and number of heads.


Table 10: Hyperparameter settings of popular transformer models. Only d (embedding dimension),
L (number of layers), and H (number of attention heads) are shown for brevity.

|H|Model|d|L|Year|
|---|---|---|---|---|
|8<br>8<br>12<br>16<br>16<br>16<br>28<br>32<br>32<br>32<br>32<br>32<br>32<br>40<br>40<br>56<br>64<br>64<br>64<br>96<br>96<br>128<br>128<br>128<br>128|Attention is all you need<br>Gemma 2B<br>GPT<br>BERT-Large<br>ViT-Huge<br>Gemma 7B<br>Turing-NLG<br>LLaMA-7B<br>Baichuan 2-7B<br>Mistral 7B<br>Yi-6B<br>LLaMA 3-8B<br>Mixtral 8x7B<br>LLaMA-13B<br>Baichuan 2-13B<br>Yi-34B<br>LLaMA-65B<br>Llama-2-70B<br>LLaMA 3-70B<br>GPT-3<br>Jurassic-1<br>MT-NLG<br>LaMDA<br>LLaMA 3.1-405B<br>DeepSeek-V2|512<br>2,048<br>768<br>1,024<br>1,280<br>3,072<br>4,256<br>4,096<br>4,096<br>4,096<br>4,096<br>4,096<br>4,096<br>5,120<br>5,120<br>7,168<br>8,192<br>8,192<br>8,192<br>12,288<br>13,824<br>20,480<br>8,192<br>16,384<br>5,120|6<br>18<br>12<br>24<br>32<br>28<br>78<br>32<br>32<br>32<br>32<br>32<br>32<br>40<br>40<br>60<br>80<br>80<br>80<br>96<br>76<br>105<br>64<br>126<br>60|2017<br>2024<br>2018<br>2019<br>2021<br>2024<br>2020<br>2023<br>2023<br>2023<br>2023<br>2024<br>2024<br>2023<br>2023<br>2023<br>2023<br>2023<br>2024<br>2020<br>2021<br>2021<br>2022<br>2024<br>2024|


35