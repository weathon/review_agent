# Enforcing Orderedness in SAEs to Improve Feature Consistency

**Anonymous** **authors**
Paper under double-blind review


Abstract


Sparse autoencoders (SAEs) have been widely used for interpretability of
neural networks, but their learned features often vary across seeds and
hyperparameter settings. We introduce Ordered Sparse Autoencoders
(OSAE), which extend Matryoshka SAEs by (1) establishing a strict ordering of latent features and (2) deterministically using every feature
dimension, avoiding the sampling-based approximations of prior nested
SAE methods. Theoretically, we show that OSAEs resolve permutation
non-identifiability in settings of sparse dictionary learning where solutions
are unique (up to natural symmetries). Empirically on Gemma2-2B and
Pythia-70M, we show that OSAEs can help improve consistency compared
to Matryoshka baselines.


1 Introduction


Sparse autoencoders (SAEs) have become central to unsupervised representation learning.
Enforcing sparsity in the latent space yields interpretable, often disentangled features, enabling progress in clustering, visualization, and scientific discovery (Vincent et al., 2010;
Coates et al., 2011; Ng, 2011). Yet despite their success, SAEs suffer from a critical shortcoming: the set of features they learn can vary across random seeds, initialization schemes,
and hyperparameter settings, leading to poor reproducibility and undermining any mechanistic interpretation of individual latent dimensions (Song et al., 2025; Fel et al., 2025).
Several strategies have been proposed to mitigate this instability. These include regularization techniques such as orthonormality penalties (Lee et al., 2025), structured sparsity
constraints like group or tree sparsity (Jenatton et al., 2010), and post-hoc alignment or
averaging of learned dictionaries across runs (Ghorbani et al., 2020).


One way to reduce the size of each equivalence class of solutions is to enforce structural
constraints into the loss function. In particular, Matryoshka SAEs (Bussmann et al., 2025)
are introduced to resolve a notion of hierarchy in feature learning. Their work defines an
ordering on features by thir level of abstraction: ”comma” is a lower-level feature than
”punctuation mark”. Matryoshka SAEs sample a small number of dictionary sizes per
batch, thereby capturing multiscale features and partially breaking permutation symmetry.
Despite these advances, Matryoshka SAEs treat features within each sampled group as
exchangeable, a limitation from sampling only a handful of dictionary sizes (e.g., up to 10
per batch).


In this work, we introduce _Ordered Sparse Autoencoders_ (OSAE), which extends Matryoshka
SAEs by enforcing a strict ordering of latent dimensions. Drawing on the concept of nested
dropout—which imposes an explicit ordering by stochastically truncating latent codes (Rippel et al., 2014)—OSAE treats each non-zero feature as its own dictionary size.


Our key contributions are:


   - We propose Ordered Sparse Autoencoders (OSAE), which enforce deterministic
feature ordering.


   - We present theoretical results for ordered feature recovery by nested dropout loss
in a special case of overcomplete sparse dictionary learning.


1


Top _m_ ( _zi_ ) _j_ =           - _z_ 0 _i,j,_ _,_ otherwiseif _|zi,j|_ in _,_ top _m,_


extended column-wise to Top _m_ ( _Z_ ).


2.2 Nested dropout in the (under)complete setting


Consider the (under)complete linear autoencoder with representation dimension _K_ _≤_ _d_ .
The standard reconstruction loss

_L_ AE( _D, E_ ) = _∥X_ _−_ _D Z∥_ [2] _F_

recovers the top- _K_ principal subspace but leaves _D_ defined only up to an invertible transformation Baldi and Hornik (1989); Bourlard and Kamp (1988); Plaut (2018). Rippel et al.
(2014) introduce the nested dropout loss, which minimizes


2
_L_ ND( _D, E_ ) = E _ℓ∼p_ ND�� _X_ _−_ _D_ Λ _ℓ_ _Z_ �� _F_ _[,]_

where _p_ ND( _ℓ_ ) is a distribution over _{_ 1 _, . . ., k}_ with full support. With _D_ _[⊤]_ _D_ = _I_, they theoretically show that this loss uniquely recovers the PCA eigenbasis in descending-eigenvalue
order, rather than merely its subspace.


In the next section we extend this idea to the overcomplete, hard- _m_ sparse setting by inserting a Top- _m_ mask into the same expectation to obtain our Ordered Sparse Autoencoder
(O-SAE).


2.3 Sparse dictionary learning


To understand the non-identifiability challenges faced by sparse autoencoders in the overcomplete regime, we first discuss classical sparse dictionary learning. The goal is to generalize PCA’s fixed-size eigenbasis to an overcomplete dictionary of atoms that admits sparse
representations. Concretely, each data vector _xi_ is modeled as

_X_ = [ _x_ 1 _, . . ., xN_ ] _∈_ R _[d][×][N]_ _,_ _X_ = _D Y,_


where
_D_ = [ _d_ 1 _, . . ., dK_ ] _∈_ R _[d][×][K]_ _,_ _Y_ = [ _y_ 1 _, . . ., yN_ ] _∈_ R _[K][×][N]_ _,_


2


   - We demonstrate improvement in feature consistency when using OSAEs on
Gemma2-2B and Pythia-70M.


2 Problem setup


2.1 Preliminaries


Throughout, we use the following notation:


   - _X_ = [ _x_ 1 _, . . ., xN_ ] _∈_ R _[d][×][N]_ : the data matrix whose columns _xi_ _∈_ R _[d]_ are samples.

   - _E_ : R _[d]_ _→_ R _[K]_ : the encoder mapping each input _xi_ to a code _zi_ = _E_ ( _xi_ ).

   - _D_ _∈_ R _[d][×][K]_ : the decoder or dictionary matrix, whose columns _dj_ _∈_ R _[d]_ are basis
atoms.

   - _Z_ = _E_ ( _X_ ) _∈_ R _[K][×][N]_ : the code matrix, whose columns are the encoded vectors _zi_ .


We will consider two settings:


   - **(Under)complete** **(** _K_ _≤_ _d_ **).** _D_ spans a _K_ -dimensional subspace (the PCA case).

   - **Overcomplete** **(** _K_ _> d_ **).** _D_ is a dictionary of _K_ atoms for sparse coding.


Define for all _ℓ_ = 1 _, . . ., K_ :


Λ _ℓ_ = - _I_ 0 _ℓ_ 00


_∈{_ 0 _,_ 1 _}_ _[K][×][K]_ _,_


with unit-norm atoms _∥dj∥_ 2 = 1 and sparse codes _∥yi∥_ 0 _≤_ _m_ _≪_ _K_ . That is, each sample
_xi_ is assumed to be generated by a linear combination of a small subset of the dictionary
atoms.

Whereas PCA solves min _∥X_ _−_ _DZ∥_ [2] _F_ [under] [a] [rank] [constraint] _[K]_ _[≤]_ _[d]_ [,] [sparse] [dictionary]
learning (SDL) tackles
min _[−]_ _[D Y][ ∥]_ [2] _F_ subject to _∥yi∥_ 0 _≤_ _m,_
_D,Y_ _[∥][X]_

an NP-hard problem due to the combinatorial nature of the _ℓ_ 0 sparsity constraint. In practice, this objective is typically approximated using greedy methods like orthogonal matching
pursuit (OMP) Pati et al. (1993); Tropp and Gilbert (2007), alternating minimization algorithms such as K-SVD Aharon et al. (2006), or online optimization techniques Mairal et al.
(2010).


A key challenge in SDL is the issue of non-identifiability: many dictionaries _D_ and code
matrices _Y_ can produce the same reconstruction _X_, especially in the overcomplete setting.
Even in the ideal noiseless case, identifiability of the ground-truth dictionary _D_ _[∗]_ is only
possible under strong structural assumptions.


**Spark** **and** **uniqueness.** The _spark_ of a dictionary _D_,
spark( _D_ ) = min _{∥z∥_ 0 : _Dz_ = 0 _,_ _z_ = 0 _},_
measures the size of the smallest linearly dependent set of atoms. If spark( _D_ ) _>_ 2 _m_, then
any _m_ -sparse representation _y_ = _Dz_ is unique, guaranteeing identifiability in sparse coding.
However, computing spark is NP-hard, so practitioners often rely on relaxed surrogate
conditions:


   - _Mutual coherence_ _µ_ ( _D_ ) = max _i_ = _j |d_ _[⊤]_ _i_ _[d][j][|]_ [, with] _[ µ]_ [(] _[D]_ [)(] _[m][−]_ [1)] _[ <]_ [ 1 ensuring uniqueness]
via greedy methods such as OMP Tropp (2004).

   - _Restricted_ _isometry_ _property_ (RIP), which ensures _D_ approximately preserves the
norms of all _m_ -sparse vectors Candes and Tao (2005).


Recent work has begun applying these identifiability conditions to sparse autoencoders. In
particular, Song et al. (2025) show that if a Top- _k_ SAE achieves exact sparsity and zero
reconstruction error, then the encoder-decoder pair satisfies a round-trip condition that
implies spark( _D_ ) _>_ 2 _k_, guaranteeing uniqueness of the learned features up to permutation
and scaling. Our work builds on this by explicitly reducing _permutation_ _ambiguity_ _during_
_training_ itself.


2.4 _ℓ_ -prefix reconstruction objective (Top- _m_ ).


We define:
2
_Lℓ_ ( _D, E_ ) = �� _X_ _−_ _D_ Λ _ℓ_ Top _m_ ( _Z_ )�� _F_ _[.]_
This objective minimizes reconstruction loss when we use the top- _k_ codes and then truncate
to the first _ℓ_ dimensions. When _ℓ_ = _K_, this becomes the standard full-code reconstruction
2
loss �� _X_ _−_ _D_ Top _k_ ( _Z_ )�� _F_ [that] [standard] [top-] _[m]_ [SAEs] [minimize.]


2.5 Matryoshka SAE objective (Top- _m_ ).


Matryoshka SAEs (Bussmann et al., 2025) partition the _K_ atoms into a small collection of
nested “groups” of increasing size _M_ = _{ℓ_ 1 _< · · · < ℓL}_ . At each training step, one group 1: _ℓ_
is sampled with probability _p_ MSAE( _ℓ_ ), and only atoms _d_ 1 _, . . ., dℓ_ (and their corresponding
code entries) are used for reconstruction:
_L_ MSAE( _D, E_ ) = E _ℓ∼p_ MSAE� _Lℓ_ ( _D, E_ )�

=          - _p_ MSAE( _ℓ_ ) �� _X_ _−_ _D_ Λ _ℓ_ Top _m_ ( _Z_ )��2 _F_ _[.]_

_ℓ∈M_

By enforcing reconstruction over only a handful of group sizes (e.g. 5–10 per batch), Matryoshka SAE captures multiscale features while partially breaking permutation symmetry
within each group.


3


By covering all prefixes in expectation, this objective enforces a strict ordering of features.


2.7 Consistency evaluation


To quantify how reproducibly SAEs recover the same features across seeds, we adopt the
_stability_ metric from Fel et al. (2025). Let _D, D_ _[′]_ _∈_ R _[d][×][K]_ be two learned decoder matrices
with unit-norm columns. We define

Stab( _D, D_ _[′]_ ) = max 1             - _D_ _[⊤]_ _P_ _D_ _[′]_ [�] _,_
_P ∈P_ _K_ [tr]


where _P_ is the set of all _K_ _× K_ permutation matrices. This computes the average cosine
similarity between matched atoms after optimal re-indexing via the Hungarian algorithm.
When we compare a learned dictionary _D_ against the ground-truth dictionary _D_ _[⋆]_, stability
also serves as a _feature_ _recovery_ _fidelity_ metric, indicating how accurately the true atoms
are recovered.


2.8 Orderedness evaluation


To quantify how similarly in order SAEs recover features across seeds, we introduce an
orderedness metric. Let _D, D_ _[′]_ _∈_ R _[d][×][K]_ be two dictionaries, each with an inherent ordering
of their atoms (e.g. by frequency, abstraction, or another criterion). After matching each
atom _dj_ in _D_ to its best-corresponding atom _d_ _[′]_ _µ_ ( _j_ ) [in] _[D][′]_ [via] [the] [Hungarian] [algorithm,] [we]
obtain a permutation vector


_µ_ =          - _µ_ (1) _, µ_ (2) _, . . ., µ_ ( _K_ )� _∈{_ 1 _, . . ., K}_ _[K]_ _._


We then define the _orderedness_ between _D_ and _D_ _[′]_ as the Spearman rank correlation between
their index sequences:

Ord( _D, D_ _[′]_ ) = Spearman�(1 _, . . ., K_ ) _,_ _µ_      - = 1 _−_ 6 [�] _j_ _[K]_ =1� _j −_ _µ_ ( _j_ )�2 _._

_K_ ( _K_ [2] _−_ 1)


A value Ord( _D, D_ _[′]_ ) = 1 indicates perfect matching of their orderings.


3 Exact recovery of ordered features


Define the domain of optimization to be


_F_ = �( _D, E_ ) �� _D_ = [ _d_ 1 _, . . ., dK_ ] _∈_ R _d×K,_ _∥dj∥_ 2 = 1 ( _∀j_ = 1 _, . . ., K_ ) _,_ _E_ : R _[d]_ _→_ R _[K]_ [�] _._


In other words, _F_ consists of all decoder–encoder pairs ( _D, E_ ) in which each dictionary
atom _dj_ has unit _ℓ_ 2-norm, and _E_ is an arbitrary mapping from R _[d]_ to R _[K]_ .


Suppose _X_ = _D_ _[∗]_ _Y_ _[∗]_ _,_ where _D_ _[∗]_ = [ _d_ _[∗]_ 1 _[, . . ., d][∗]_ _K_ []] _[∈]_ [R] _[d][×][K]_ [has] [unit-norm] [columns] [satisfying]
spark(D _[∗]_ ) _>_ 2m, and _Y_ _[∗]_ _∈_ R _[K][×][N]_ _m_ -sparse columns.

**Lemma** **3.1.** _Any_ _minimiser_ _of_ _L_ ND _also_ _minimises_ _the_ _full-prefix_ _loss_ _Lk._ _That_ _is,_


argmin(D _,_ E) _∈F_ _L_ ND _⊆_ argmin(D _,_ E) _∈F_ _L_ K _._


4


2.6 Nested dropout objective (Top- _m_ ).


We extend nested dropout (Rippel et al., 2014) by treating each individual atom _dj_ as its
own “group,” so that sampling a prefix _ℓ_ means retaining exactly atoms 1 through _ℓ_ and
dropping the rest. Let _p_ ND( _ℓ_ ) be a distribution over _{_ 1 _, . . ., m}_ with full support. The
nested-dropout loss is


_L_ ND( _D, E_ ) = E _ℓ∼p_ ND� _Lℓ_ ( _D, E_ )�


=


_m_

- _p_ ND( _ℓ_ ) �� _X_ _−_ _D_ Λ _ℓ_ Top _m_ ( _Z_ )��2 _F_ _[.]_

_ℓ_ =1


The proof is deferred to Appendix A

**Theorem** **3.1.** _[Exact_ _ordered_ _recovery_ _under_ _spark_ _condition]_ _Assume_ _the_ _columns_ _of_ _Y_ _[∗]_
_are_ _nonnegative_ _(to_ _resolve_ _sign_ _ambiguity)_ _and_ _“true”_ _atoms_ _are_ _ordered_ _so_ _that_

_|{ i_ : _y_ 1 _[∗]_ _,i_ _[>]_ [ 0] _[}|]_ _[≥|{][ i]_ [ :] _[ y]_ 2 _[∗]_ _,i_ _[>]_ [ 0] _[}|]_ _[≥· · ·]_ _[≥|{][ i]_ [ :] _[ y]_ _K,i_ _[∗]_ _[>]_ [ 0] _[}|][.]_


_Then_ _any_ _global_ _minimiser_ ( _D,_ [�] _E_ [�] ) _∈F_ _of_ _the_ _nested-dropout_ _loss_ _L_ ND _satisfies_

_D_         - = _D_ _[∗]_ _,_ _E_ �( _X_ ) = _Y_ _[∗]_


_Proof._ By Lemma 3.1, any minimiser of _L_ ND also minimises the full-prefix loss _LK_ . Hence
( _D,_ [�] _Y_ [�] ) with _Y_ [�] = Topm��E(X)� satisfies


_D_         - _Y_         - = _X,_ _∥y_         - _i∥_ 0 _≤_ _m._


The uniqueness result under the spark condition then gives


_D_         - = _D_ _[∗]_ _P_ _S,_ _Y_         - = _S_ _[−]_ [1] _P_ _[⊤]_ _Y_ _[∗]_ _,_


for some permutation matrix _P_ and invertible diagonal _S_ .


Since all columns of _Y_ _[∗]_ and _Y_ are nonnegative, _S_ must be the identity (no sign-flips or

[�]
rescaling). Finally, because the atoms were assumed ordered by their sparsity-support frequencies, the only permutation that preserves that ordering is the identity. Hence _P_ = _I_
and
_D_          - = _D_ _[∗]_ _,_ _Y_          - = _Y_ _[∗]_ _,_


as claimed.


3.1 Toy Gaussian model


Theorem 3.1 gives theoretical guarantees that SAEs minimising nested dropout loss can
achieve perfect consistency and orderedness under certain conditions of the data and sparsity.
We evaluate under the following synthetic generative model:


1. **Parameters.** Fix dimensions _d, n, K_ and sparsity level _m ≤_ _d_ . Assume an _ordering_
distribution _π_ = ( _π_ 1 _, . . ., πK_ ) with _π_ 1 _≥_ _π_ 2 _≥· · · ≥_ _πK_ _>_ 0 and [�] _j_ _[K]_ =1 _[π][j]_ [= 1.]

2. **Dictionary** **generation.**

_D_ = [ _d_ 1 _, . . ., dK_ ] _∈_ R _[d][×][K]_ _,_ _dj_ iid _∼N_ �0 _,_ _d_ [1] _[I][d]_      - _,_ _dj_ _←_ _∥ddjj∥_ 2 _._


3. **Code** **generation.** For each sample _i_ = 1 _, . . ., n_ :


(a) Sample a support of size _m_ by drawing indices without replacement according
to _π_ :
_Si_ _∼_ MultisetSample( _π, m_ ) _._


(b) Let

              - _zij,_ _j_ _∈_ _Si,_ iid
_yi_ _∈_ R _[K]_ _,_ ( _yi_ ) _j_ = _zij_ _∼N_ (0 _,_ 1) _._
0 _,_ _j_ _∈/_ _Si,_


(c) For the purposes of this toy model, we remove sign ambiguity: _yi_ _←|yi|_ .

Collect into _Y_ = [ _y_ 1 _, . . ., yn_ ] _∈_ R _[K][×][n]_ _._


4. **Data** **matrix.**
_X_ = _D Y_ _∈_ R _[d][×][n]_ _._


Under this model, atoms with smaller index _j_ appear more frequently in the data (higher
_πj_ ), inducing a ground-truth ordering that we will attempt to recover via O-SAEs. Since the
dictionary is drawn from a standard Gaussian ensemble, the spark condition for uniqueness
is satisfied with high probability (Hillar and Sommer, 2015).


5


**Unit** **sweeping.** Nested dropout samples a truncation index _b ∼_ _pB_ ( _·_ ), so gradients onto
late units shrink exponentially with index. To avoid starving these units, we employ _unit_
_sweeping_ (Rippel et al., 2014): once a lower-index unit has effectively converged, we _freeze_
its encoder row and decoder column (stop backprop through that unit) and continue training
the remaining, unfrozen units. Practically, we use a simple “clockwork” schedule that freezes
one additional unit every _T_ epochs (from 1 _→_ _K_ ), after a short burn-in; frozen units remain
in the forward pass, and we renormalize decoder columns to unit norm after each freeze.
Results in Fig. 4 and Table 1 use unit sweeping, which we find improves stability and ordered
recovery in this setting.


**Evaluation.** We evaluate Ordered SAE, Matryoshka SAE (Fixed and Random, with
five groups), and Vanilla top- _m_ SAEs on the toy model above with ( _d, K, m, N_ ) =
(80 _,_ 100 _,_ 5 _,_ 100 _,_ 000) and a Zipf support prior _πj_ _∝_ _j_ _[−][α]_, _α_ = 1 _._ 2, which induces a strict
ground-truth ordering. Hyperparameters are selected by lowest validation reconstruction
error at the target sparsity. We use the warmup schedule for _k_ (from _K_ down to _m_ ) and
train with _unit_ _sweeping_ . Qualitative recovery patterns are shown in Fig. 1 for Fixed MSAE
and OSAE; full results can be found in Fig. 4. Quantitative results (mean _±_ std) are summarized in Table 1. Importantly, Stab( _D, D_ _[′]_ ) is averaged over �102 - = 45 seed pairs, while
Stab( _D, D_ _[∗]_ ) and Ord( _D, D_ _[∗]_ ) are averaged over 10 seed _→_ _D_ _[∗]_ comparisons; reconstruction
loss is MSE on a held-out set.


For each model, we choose the hyperparameter configuration that achieves the lowest validation reconstruction error at the target sparsity level.


To stabilize training and improve recovery, we adopt a warmup strategy for top- _k_ truncation.
Specifically, we begin training with a large truncation size _k_ init _≥_ _m_ (typically _k_ init = _K_ )
and gradually decrease _k_ to the target sparsity _m_ over a fixed number of epochs. This
schedule smooths the optimization landscape by initially allowing dense activations before
progressively enforcing sparsity. We find this warmup significantly improves convergence,
particularly for O-SAE and Matryoshka models where early features receive higher training
pressure. Additional results are shown in Appendix B.


**Model** **Stab(** _D_ **,** _D_ _[′]_ **)** **Stab(** _D_ **,** _D_ _[∗]_ **)** **Ord(** _D_ **,** _D_ _[∗]_ **)** **Reconstruction** **loss**


Vanilla SAE 0 _._ 572 (0 _._ 00964) 0 _._ 479 (0 _._ 0104) 0 _._ 0162 (0 _._ 128) 0 _._ 0257 (0 _._ 000841)
Fixed MSAE 0 _._ 538 (0 _._ 0114) 0 _._ 502 (0 _._ 0160) 0 _._ 119 (0 _._ 673) 0 _._ 0339 (0 _._ 00635)
Random MSAE 0 _._ 531 (0 _._ 0150) 0 _._ 480 (0 _._ 0106) 0 _._ 0544 (0 _._ 0760) 0 _._ 0309 (0 _._ 00366)
**OSAE** **(ours)** **0** _**.**_ **664 (0** _**.**_ **0191)** **0** _**.**_ **814 (0** _**.**_ **0195)** **0** _**.**_ **734 (0** _**.**_ **0758)** **0** _**.**_ **00725 (0** _**.**_ **000746)**


Table 1: Summary metrics on the toy Gaussian model (mean _±_ std).


We found it surprising that our O-SAEs achieved the lowest reconstruction loss here despite
possessing inductive biases that restrict the solution class. Thus for a controlled evalation
with minimal influence from hyperparameter bias, we ran further evaluations for O-SAEs in
Appendix B.3 directly extending Song et al. (2025)’s synthetic experiments. There, O-SAEs
achieve a similar consistency and higher orderedness compared to baseline architectures
despite a higher MSE loss, helping support our findings here.


4 Results on empirical data


4.1 Text-Based Evaluations on Gemma-2 2B


**Setup.** We evaluate the orderedness and stability of pairs of random seeds for Ordered
SAEs and Matryoshka SAE variants trained on Gemma-2 2B (Team et al., 2024). We
train these SAEs with dictionary size 4096 on layer 12 with K=80, collecting activations
from Gemma-2 2B on an uncopyrighted portion of the Pile (Gao et al., 2020). We use the
same piecewise distribution for the O-SAE as Matryoshka baselines for these experiments.
Unit sweeping is not employed for these experiments. We note that O-SAEs have slightly
worse reconstruction loss compared to baselines, potentially due to a restricted solution


6


Figure 1: Recovery across SAE variants on a Gaussian toy model with ( _d, K, m, N_ ) =
(80 _,_ 100 _,_ 5 _,_ 100 000). Each panel plots the Hungarian matching between learned decoder
atoms _D_ and ground truth _D_ _[∗]_ (one dot per matched pair; color encodes cosine similarity).
Ordered SAEs achieve higher stability Stab( _D, D_ _[∗]_ ) (mean matched cosine) and higher orderedness Ord( _D, D_ _[∗]_ ) (order agreement), meaning they recover features more faithfully and
in order.


space or, for a simpler explanation, our limited hyperparameter sweeps. Thus, we compare
checkpoints with similar loss values for a fair orderedness and stability comparison.


**O-SAEs** **achieve** **greater** **orderedness** **and** **stability** **in** **Gemma-2** **2B.** In Figure 2, we
demonstrate that O-SAEs achieve better orderedness (defined in 2.8) than the Fixed-prefix
MSAE and Random-prefix MSAE with 5 groups. Although the overall average stability
(defined in 2.7) is lower for O-SAE than the Matryoshka variants, we find that the most
significant features have higher stability. For both metrics, we calculate the truncated
metrics for the first _p_ prefix length features. We see that O-SAEs have a lower prefixlength stability compared to the Random MSAE at a crossover point between 1024 and
2048 features, where features are less significant in the feature ordering.


4.2 Consistency across datasets on Pythia 70m


We furthermore test the generalizability of orderedness and stability of O-SAEs trained
on a different model, Pythia-70M (Biderman et al., 2023), and also evaluate consistency
when activations are collected on different datasets. Both the Pile (Gao et al., 2020) and
Dolma (Soldaini et al., 2024) are diverse and general pretraining mixes, so we select these
datasets to represent a wide distribution of language while not drawing from the exact same
data sources. If SAEs learn a good representation of general language, we ideally hope that
the representations are consistent across datasets. We evaluate (1) same-dataset consistency
similar to the previous section and (2) cross-dataset consistency by evaluating pairs of SAEs
where one is trained on activations on the Pile and the other on activations from Dolma.


**Setup.** We train with the same hyperparameters as Gemma2-2B, although Pythia-70M
is much smaller and has fewer layers. We use layer 3 of Pythia-70M, roughly halfway, to
approximate the same position as layer 12 in Gemma2-2B. We train five seeds per (model,
dataset) pair. In these experiments, we evaluate checkpoints after training for 50M tokens,
reaching 0.02-0.03 recon loss.


**Similar** **trends** **on** **the** **Pile** **and** **Dolma** **datasets.** In Figure 3a, we evaluate samedataset orderedness and stability on the SAEs trained on the Pile and Dolma. We see
similar trends of higher orderedness for O-SAEs while stability degrades after time. All
SAE variants maintain similar performance between the Pile and Dolma in same-dataset
evaluations.


7


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
||||||||||O<br>~~Ra~~|AE (Ours<br>~~ndom MS~~|<br>~~ AE~~|
||||||||||<br>Fix|<br>ed MSAE||


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|OS<br>Ra<br>Fix|AE (Ours<br>ndom MS<br>ed MSAE|)<br>AE|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


Figure 2: SAEs trained on Gemma2-2B. (a) Orderedness evaluated at different prefix
lengths. O-SAE’s have the most consistently ordered features almost reaching an average **Ord(** _D_ **,** _D_ _[′]_ **)** of 0.8. As expected, we observe orderedness close to 0.0 for the first 128
features of the Fixed MSAE since the first group size is 128, whereafter it jumps up to
values between around 0.5. (b) O-SAEs have high stability for the first portion of features,
before a sharp decline for later features. �92� = 36 pairs of seeds are evaluated per method
and 95% confidence intervals are visualized.


We include additional visualizations of how measures of orderedness and stability evolve
with increasing amounts of training on same-dataset evaluations on the Pile and Dolma in
Figure 11 and 12 in Appendix C. Interestingly, there are some cases for both O-SAEs and
Random MSAEs where as training progresses, high levels of orderedness and stability in
earlier prefixes decrease while these measures at later prefixes increase. We suspect this
may be an artifact of the probability distribution or over-training, but we plan to run more
ablations.


**O-SAEs** **also** **improve** **orderedness** **in** **cross-dataset** **comparisons.** In Figure 3b,
we evaluate orderedness and stability of SAEs in cross-dataset settings, where one SAE is
trained on the Pile and the other on Dolma. We observe that cross-dataset orderedness and
stability matches trends from same-dataset results in Figure 3a, with a decline of around
0.1-0.2 from same-dataset results.


When we use same-seed initialization, O-SAEs achieve near-1.0 value in Orderedness and
stability of 0.8 at full prefix length, when trained on different datasets. Orderedness and
stability also increase in early prefix dictionary positions compared to the cross-seed settings
for both O-SAEs and Random MSAEs; however, increases are more substantial for O-SAEs.
We hypothesize the greater jump in full-prefix orderedness and stability metrics for OSAEs are because they do not update their latter indices as much as earlier indices; this is
further supported by Figure 13 and 14 in Appendix C showing minimal full-prefix changes
in orderedness when comparing trained models against their initialization state for O-SAEs
and significant full-prefix drops for Random MSAEs when comparing trained models against
their initialization state.


5 Limitations


Our theoretical guarantees assume idealized conditions—sparsity and uniqueness assumptions and a well-specified ordering prior—that may not hold exactly in real data. If the
imposed order is misspecified, the objective can over-regularize and suppress equally valid
alternative bases. The training cost for O-SAEs is higher because covering prefixes requires
more compute, so practical deployments may require approximations or careful schedule design. Performance is also sensitive to design choices such as the prefix distribution and unit
sweeping. Finally, our empirical study focuses on controlled settings intended to evaluate the
mechanism rather than to exhaustively benchmark task performance across architectures.


8


0.8


0.6


0.4


0.2


0.0


0.2


Orderedness across dictionary sizes


Prefix Length of Dictionary


(a) **Ord(** _D_ **,** _D_ _[′]_ **)**


0.7


0.6


0.5


0.4


0.3


Stability across dictionary sizes


Prefix Length of Dictionary


(b) **Stab(** _D_ **,** _D_ _[′]_ **)**


|Col1|Col2|Col3|Col4|Col5|OS<br>OS<br>Ra|AE (Pile)<br>AE (Dolm<br>ndom MS|a)<br>AE (Pile)|R<br>Fi<br>Fi|andom M<br>xed MSA<br>xed MSA|SAE (Dolm<br>E (Pile)<br>E (Dolma)|a)|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


|Col1|Col2|Col3|Col4|Col5|OS|AE (Pile)|Col8|R|andom M|SAE (Dolm|a)|
|---|---|---|---|---|---|---|---|---|---|---|---|
||||||~~OS~~<br>Ra|~~E (Dolm~~<br>dom MS|~~ )~~<br> E (Pile)|~~F~~<br>Fi|~~xed MSA~~<br>xed MSA|~~  (Pile)~~<br> E (Dolma)||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|


|Col1|Col2|O<br>R|SAE (Cros<br>andom M|s Seed)<br>SAE (Sam|e Seed)|Fi<br>Fi|xed MSAE<br>xed MSAE|(Same S<br>(Cross S|eed)<br>eed)|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


|Col1|Col2|O<br>R|SAE (Cros<br>andom M|s Seed)<br>SAE (Sam|e Seed)|Fi<br>Fi|xed MSAE<br>xed MSAE|(Same S<br>(Cross S|eed)<br>eed)|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||
|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|4<br>8<br>16<br>32<br>64<br>128<br>256<br>512<br>1024<br>2048<br>4096<br>Prefix Length of Dictionary|


(b) **Ord(** _D_ **,** _D_ _[′]_ **)** and **Stab(** _D_ **,** _D_ _[′]_ **)** on Cross-dataset setting


Figure 3: SAEs trained on Pythia-70M (a) O-SAEs demonstrate an improvement in orderedness over Random MSAEs and Fixed MSAEs on the Pile and Dolma after prefix length 128.
O-SAE stability is likewise stronger than Random MSAE on both datasets for the beginning
features, but crosses over at around prefix length 128. Fixed MSAE stability is higher than
O-SAE, but has lower orderedness. (n=10). (b) In the cross-dataset, cross-seed setting
we observe that O-SAE has modest improvements in orderedness against Random MSAE
and sizable stability gains against Random MSAE before prefix 128. O-SAEs and Random
MSAEs demonstrate improvements to orderedness and stability when using the same seed
despite training on different datasets; however, the improvements are larger for O-SAEs.
(Cross-Seed: n=20. Same-Seed: n=5)


6 Discussion


By optimizing an ordered, nested-prefix objective, we _shrink_ _the_ _solution_ _class_ of sparse autoencoders. The plain reconstruction loss admits large equivalence classes (permutations and
near-mixings of features) that undermine reproducibility; the ordered objective effectively
selects a canonical basis. This narrowing of admissible solutions is a training-time structural prior, which (i) constrains the hypothesis space, (ii) alters the optimization geometry
toward a feature-curriculum, and (iii) makes the learned representation more comparable
across runs and hyperparameters. We view ordering as a general mechanism for enforcing identifiability into overcomplete models, complementary to sparsity and incoherence
assumptions in classical dictionary learning. Furthermore, we provide empirical results for
O-SAEs on Gemma2-2B and Pythia 70m, trained on the Pile and Dolma, that demonstrate
greater orderedness and stability in earlier features, while sometimes at the cost of lower
stability in later, less-significant features.


9


Stability across dictionary sizes


0.8


0.7


0.6


0.5


0.4


0.3


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


Orderedness across dictionary sizes


Prefix Length of Dictionary


(a) **Ord(** _D_ **,** _D_ _[′]_ **)** and **Stab(** _D_ **,** _D_ _[′]_ **)** comparing SAEs trained on The Pile and Dolma


Orderedness across dictionary sizes


Stability across dictionary sizes


0.8


0.7


0.6


0.5


0.4


0.3


Prefix Length of Dictionary


Use of LLMs


We made limited use of LLMs during paper preparation. Specifically, we used them to help
write scripts for generating plots, and to suggest edits aimed at improving clarity of the
text. All scientific claims are validated by the authors.


References


Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network
with a local denoising criterion. _Journal of Machine Learning Research_, 11(12):3371–3408,
2010.


Adam Coates, Andrew Y Ng, and Honglak Lee. An analysis of single-layer networks in
unsupervised feature learning. In _AISTATS_, 2011.


Andrew Ng. Sparse autoencoder. _CS294A_ _Lecture_ _notes_, 72(2011):1–19, 2011.


Xiangchen Song, Sarah Zhao, Aidan Lee, Binxu Wang, and Talia Konkle. Position:
Mechanistic interpretability should prioritize feature consistency in saes. _arXiv_ _preprint_
_arXiv:2505.20254_, 2025.


Thomas Fel, Ekdeep Singh Lubana, Jacob S Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle.
Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large
vision models. _arXiv_ _preprint_ _arXiv:2502.12892_, 2025.


Hongyu Lee, Kevin Fang, Xinyue Zhao, Binxu Wang, Demba Ba, and Talia Konkle. Evaluating and designing sparse autoencoders by approximating quasi-orthogonality. _arXiv_
_preprint_ _arXiv:2503.24277_, 2025.


Rodolphe Jenatton, Jean-Yves Audibert, and Francis Bach. Structured sparse principal
component analysis. In _AISTATS_, pages 366–373, 2010.


Amirata Ghorbani, James Wexler, James Zou, and Been Kim. Neuron shapley: Discovering the responsible neurons. In _International_ _Conference_ _on_ _Artificial_ _Intelligence_ _and_
_Statistics_, pages 519–530, 2020.


Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning multi-level
features with matryoshka sparse autoencoders. _arXiv_ _preprint_ _arXiv:2503.17547_, 2025.


Oren Rippel, Michael A. Gelbart, and Ryan P. Adams. Learning ordered representations
with nested dropout. In _International Conference on Machine Learning_, pages 1746–1754,
2014.


Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning
from examples without local minima. _Neural_ _Networks_, 2(1):53–58, 1989.


Herv´e Bourlard and Yves Kamp. Auto-association by multilayer perceptrons and singular
value decomposition. In _Biological_ _Cybernetics_, volume 59, pages 291–294, 1988.


Eugene Plaut. From principal subspaces to principal components with linear autoencoders.
_arXiv_ _preprint_ _arXiv:1804.10253_, 2018.


Y. C. Pati, R. Rezaiifar, and P. S. Krishnaprasad. Orthogonal matching pursuit: Recursive
function approximation with applications to wavelet decomposition. In _Conference Record_
_of_ _The_ _Twenty-Seventh_ _Asilomar_ _Conference_ _on_ _Signals,_ _Systems_ _and_ _Computers_, pages
40–44. IEEE, 1993.


Joel A. Tropp and Anna C. Gilbert. Signal recovery from random measurements via orthogonal matching pursuit. _IEEE Transactions on Information Theory_, 53(12):4655–4666,
2007.


10


Michal Aharon, Michael Elad, and Alfred Bruckstein. K-svd: An algorithm for designing
overcomplete dictionaries for sparse representation. _IEEE_ _Transactions_ _on_ _Signal_ _Pro-_
_cessing_, 54(11):4311–4322, 2006.


Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online learning for matrix
factorization and sparse coding. _Journal_ _of_ _Machine_ _Learning_ _Research_, 11:19–60, 2010.


Joel A Tropp. Greed is good: Algorithmic results for sparse approximation. _IEEE_ _Trans-_
_actions_ _on_ _Information_ _Theory_, 50(10):2231–2242, 2004.


Emmanuel J Candes and Terence Tao. Decoding by linear programming. _IEEE Transactions_
_on_ _Information_ _Theory_, 51(12):4203–4215, 2005.


Christopher J. Hillar and Friedrich T. Sommer. When can dictionary learning uniquely
recover sparse data from subsamples? _arXiv_ _preprint_ _arXiv:1106.3616_, 2015.


Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin,
Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre
Ram´e, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos,
Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr
Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, JeanBastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn,
Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen,
Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot,
Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar,
Dominika Rogozi´nska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica
Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei,
Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Pluci´nska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway,
Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong
Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell,
Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan
Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson,
Martin G¨orner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz,
Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk
Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda,
Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton,
Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir
Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, S´ebastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti
Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh
Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han,
Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe
Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav
Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet,
Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy,
Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a
practical size, 2024. URL `[https://arxiv.org/abs/2408.00118](https://arxiv.org/abs/2408.00118)` .


Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster,
Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor
Leahy. The pile: An 800gb dataset of diverse text for language modeling, 2020. URL
`[https://arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)` .


11


Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric
Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward
Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for
analyzing large language models across training and scaling, 2023. URL `[https://arxiv.](https://arxiv.org/abs/2304.01373)`
`[org/abs/2304.01373](https://arxiv.org/abs/2304.01373)` .


Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell
Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann,
Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E.
Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an
open corpus of three trillion tokens for language model pretraining research, 2024. URL
`[https://arxiv.org/abs/2402.00159](https://arxiv.org/abs/2402.00159)` .


Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al
Moubayed, Lee Sharkey, and Neel Nanda. Sparse autoencoders do not find canonical
units of analysis, 2025. URL `[https://arxiv.org/abs/2502.04878](https://arxiv.org/abs/2502.04878)` .


12


Since all _Lℓ<K_ remain fixed,
_L_ ND( _D_ _[ϵ]_ _, E_ _[⋆]_ ) _< L_ ND( _D_ _[⋆]_ _, E_ _[⋆]_ ) _,_
contradicting the global minimality of ( _D_ _[⋆]_ _, E_ _[⋆]_ ). Therefore no residual can remain, and
every minimiser of _L_ ND must satisfy _∥R∥F_ = 0, i.e. also minimise _LK_ .


13


A Proofs


_Proof_ _of_ _Lemma_ _3.1._ Assume for contradiction that ( _D_ _[⋆]_ _, E_ _[⋆]_ ) _∈F_ globally minimizes _L_ ND
but _does_ _not_ minimize _LK_ . Write

1
_Z_ _[⋆]_ = _E_ _[⋆]_ ( _X_ ) _,_ _R_ _[⋆]_ = _X_ _−_ _D_ _[⋆]_ Top _m_ ( _Z_ _[⋆]_ ) _,_ ∆= _F_ [=] _[L][K]_ [(] _[D][⋆][, E][⋆]_ [)] _[>]_ [0] _[.]_
_N_ _[∥][R][⋆][∥]_ [2]

Denote the _K_ th row of Top _m_ ( _Z_ _[⋆]_ ) by _yK_ _[⋆]_ _·_ _[∈]_ [R] _[n]_ [.]

If _yK_ _[⋆]_ _·_ [=] [0] [then] [none] [of] [the] [prefix] [losses] [ever] [see] [atom] _[K]_ [,] [so] [we] [could] [remove] [it] [entirely]
and re-index, strictly reducing the nested-dropout loss unless _R_ _[∗]_ = 0. Hence for a strict
counterexample we may assume _yK_ _[⋆]_ _·_ [= 0.”]


Set


_v_ = _R_ _[⋆]_ _yK_ _[⋆]_ _·_ [=]


_n_

- _r·_ _[⋆]_ _i_ _[y]_ _K,i_ _[⋆]_ _[∈]_ [R] _[d][,]_

_i_ =1


which is nonzero since _R_ _[⋆]_ = 0 and _yK_ _[⋆]_ _·_ [= 0.] [Define]

( _I_ _−_ _d_ _[⋆]_ _K_ [(] _[d][⋆]_ _K_ [)] _[⊤]_ [)] _[ v]_
_u_ = _._
��( _I_ _−_ _d⋆K_ [(] _[d][⋆]_ _K_ [)] _[⊤]_ [)] _[ v]_ ��2

Then _u_ is unit-length, _u ⊥_ _d_ _[⋆]_ _K_ [,] [and]


_n_


_r·_ _[⋆]_ _i_ _[y]_ _K,i_ _[⋆]_  - = _u_ _[⊤]_ _v_ = ��( _I_ _−_ _d⋆K_ [(] _[d]_ _K_ _[⋆]_ [)] _[⊤]_ [)] _[ v]_ ��2 _[>]_ [0] _[.]_
_i_


- _yK,i_ _[⋆]_ [(] _[u][⊤][r]_ _·_ _[⋆]_ _i_ [) =] _[u][⊤]_ [��]

_i_ =1 _i_


Now for small _ϵ >_ 0 define a perturbation of only the _K_ th atom:

_d_ _[⋆]_ _K_ [+] _[ ϵ u]_
_d_ [new] _K_ = = _d_ _[⋆]_ _K_ [+] _[ ϵ u][ −]_ [1] 2 _[ϵ]_ [2] _[ d]_ _K_ _[⋆]_ [+] _[ O]_ [(] _[ϵ]_ [3][)] _[,]_
_∥d_ _[⋆]_ _K_ [+] _[ ϵ u][∥]_ [2]

and leave _E_ _[⋆]_ (hence Top _m_ ( _Z_ )) unchanged. All other atoms remain as in _D_ _[⋆]_, so ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) _∈_
_F_ .


Note that every prefix _ℓ< K_ satisfies
_D_ _[ϵ]_ Λ _ℓ_ Top _m_ ( _Z_ _[⋆]_ ) = _D_ _[⋆]_ Λ _ℓ_ Top _m_ ( _Z_ _[⋆]_ ) _,_
hence _Lℓ_ ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) = _Lℓ_ ( _D_ _[⋆]_ _, E_ _[⋆]_ ) for all _ℓ< K_ . Therefore


- _pℓ_ _Lℓ_ ( _D_ _[⋆]_ _, E_ _[⋆]_ ) _._


_ℓ<K_


_L_ ND( _D_ _[ϵ]_ _, E_ _[⋆]_ ) =


_K_


- _pℓ_ _Lℓ_ ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) = _pK LK_ ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) + 

_ℓ_ =1 _ℓ<K_


It suffices to show _LK_ ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) _< LK_ ( _D_ _[⋆]_ _, E_ _[⋆]_ ).


Since Top _m_ ( _Z_ ) is unchanged,

_R_ _[ϵ]_ = _X_ _−_ _D_ _[ϵ]_ Top _m_ ( _Z_ _[⋆]_ ) = _R_ _[⋆]_ _−_       - _d_ [new] _K_ _−_ _d_ _[⋆]_ _K_       - _yK_ _[⋆]_ _·⊤._

Using _d_ [new] _K_ _−_ _d_ _[⋆]_ _K_ [=] _[ ϵ u][ −]_ [1] 2 _[ϵ]_ [2] _[ d]_ _K_ _[⋆]_ [+] _[ O]_ [(] _[ϵ]_ [3][),] [one] [finds,] [entrywise,]

_rαi_ _[ϵ]_ [=] _[ r]_ _αi_ _[⋆]_ _[−]_ _[ϵ u][α]_ _[y]_ _K,i_ _[⋆]_ [+] _[ O]_ [(] _[ϵ]_ [2][)] _[.]_


Squaring and summing,


�( _rαi_ _[ϵ]_ [)][2] [=] 

_α,i_ _α,i_


_yK,i_ _[⋆]_ [(] _[u][⊤][r]_ _·_ _[⋆]_ _i_ [) +] _[ O]_ [(] _[ϵ]_ [2][)] _[.]_
_i_


_∥R_ _[ϵ]_ _∥_ [2] _F_ [=] 


�( _rαi_ _[⋆]_ [)][2] _[ −]_ [2] _[ϵ]_ 

_α,i_ _i_


By our choice of _u_, the coefficient [�] _i_ _[y]_ _K,i_ _[⋆]_ [(] _[u][⊤][r]_ _·_ _[⋆]_ _i_ [) is strictly positive, so for sufficiently small]

_ϵ_ the linear term makes _∥R_ _[ϵ]_ _∥_ [2] _F_ _[<][ ∥][R][⋆][∥]_ _F_ [2] [.] [Equivalently,]


_LK_ ( _D_ _[ϵ]_ _, E_ _[⋆]_ ) = [1]


_N_ [1] _[∥][R][⋆][∥]_ _F_ [2] [=] _[ L][K]_ [(] _[D][⋆][, E][⋆]_ [)] _[.]_


_N_ [1] _[∥][R][ϵ][∥]_ _F_ [2] _[<]_ _N_ [1]


Stab _Z_ ( _Z_ [(] _[s]_ [)] _, Y_ _[∗]_ ) = [1]


_K_ [1] [tr] - _R_ [(] _[ρ]_ [)] _P_ - _,_ Stab _Z_ ( _Z_ [(] _[a]_ [)] _, Z_ [(] _[b]_ [)] ) = _K_ [1]


Figure 4: Recovery across SAE variants on a Gaussian toy model with ( _d, K, m, N_ ) =
(80 _,_ 100 _,_ 5 _,_ 100 000). Each panel plots the Hungarian matching between learned decoder
atoms _D_ and ground truth _D_ _[∗]_ (one dot per matched pair; color encodes cosine similarity).
Ordered SAEs achieve higher stability Stab( _D, D_ _[∗]_ ) (mean matched cosine) and higher orderedness Ord( _D, D_ _[∗]_ ) (order agreement), meaning they recover features more faithfully and
in order.


B Toy model results


B.1 Activation–stream stability and orderedness


Let _Z_ [(] _[s]_ [)] = _E_ [(] _[s]_ [)] ( _X_ ) _∈_ R _[K][×][N]_ be the code matrix for seed _s_ (rows are unit activations over the
evaluation set), and let _Y_ _[∗]_ _∈_ R _[K][×][N]_ be the ground-truth activations. We replace decoderspace cosine with _Pearson_ _correlation_ on the activation stream and compute stability via
Hungarian assignment:

_Rij_ [(] _[ρ]_ [)] [= corr]          - _Zi_ [(] : _[s]_ [)] _[,]_ _[Y]_ _j_ _[∗]_ :� _,_ _Sij_ [(] _[ρ]_ [)] = corr� _Zi_ [(] : _[a]_ [)] _[,]_ _[Z]_ _j_ [(] : _[b]_ [)]          - _._


With _P_ the optimal permutation matrix, we report


_K_ [1] [tr] - _S_ [(] _[ρ]_ [)] _P_ - _,_


14


and define orderedness on the induced permutation as


Ord _Z_ = Spearman�(1 _, . . ., K_ ) _,_ ( _µ_ (1) _, . . ., µ_ ( _K_ ))� _,_ where _µ_ is read off from _P._


These _Z_ -based measures complement the decoder–cosine results in the main text. Whereas
Fig. 4 visualizes only matched pairs, the figures below show the _full_ _K × K_ similarity fields.
In the top row we also show a small raster (50 evaluation inputs) to compare activation
patterns _Y_ _[∗]_ vs. _Z_ [(0)] vs. _Z_ [(1)] . _Note:_ for Vanilla and Matryoshka models, activations can be
much larger than _Y_ _[∗]_, so per-panel normalization makes the rasters not directly comparable
in scale; this effect is less pronounced for the ordered model.


Figure 5: **Vanilla** **SAE** **(example** **seed** **pair).** _Top:_ activation rasters for 50 eval inputs
(left: _Y_ _[∗]_, middle: _Z_ [(0)], right: _Z_ [(1)] ). _Middle:_ all-pairs activation–Pearson matrices (left:
_Z_ [(0)] vs. _Y_ _[∗]_, middle: _Z_ [(1)] vs. _Y_ _[∗]_, right: _Z_ [(0)] vs. _Z_ [(1)] ) used for Stab _Z_ and Ord _Z_ . _Bottom:_
all-pairs decoder–cosine matrices (left: _D_ [(0)] vs. _D_ _[∗]_, middle: _D_ [(1)] vs. _D_ _[∗]_, right: _D_ [(0)] vs.
_D_ [(1)] ); this extends Fig. 4 from matched pairs to all pairs.


B.2 Additional dictionary sizes


We repeat the activation–stream analysis at smaller widths with matching sparsities,
( _K, m_ ) _∈{_ (10 _,_ 2) _,_ (30 _,_ 3) _,_ (50 _,_ 5) _}_ . Each panel uses the same three-row layout as Appendix
B.1: _top_ —activation rasters for 50 evaluation inputs (left: _Y_ _[∗]_, middle: _Z_ [(0)], right: _Z_ [(1)] );
_middle_ —all-pairs activation–Pearson matrices (left: _Z_ [(0)] vs. _Y_ _[∗]_, middle: _Z_ [(1)] vs. _Y_ _[∗]_, right:
_Z_ [(0)] vs. _Z_ [(1)] ) used for Stab _Z_ and Ord _Z_ ; _bottom_ —all-pairs decoder–cosine matrices (left:
_D_ [(0)] vs. _D_ _[∗]_, middle: _D_ [(1)] vs. _D_ _[∗]_, right: _D_ [(0)] vs. _D_ [(1)] ). Figures show a _single_ _example_ _seed_
_pair_ for brevity; Matryoshka variants are omitted. Note that in Vanilla, activations can be
much larger than _Y_ _[∗]_, so per-panel normalization of the top row can make scales not directly
comparable; this effect is less pronounced for O-SAE.


B.3 Zipfian toy model: high consistency with moderate orderedness


**Setup.** We evaluate Ordered SAEs (O-SAEs; “Ordered TopK” in the legend) on a synthetic Zipfian activation process. Following Song et al. (2025), inputs live in R [16] with an
overcomplete ground-truth dictionary of 32 atoms; _k_ = 3 features are active per sample, and


15


Figure 6: **Fixed** **MSAE** **(example** **seed** **pair).** Same layout as Fig. 5: top—activation
rasters ( _Y_ _[∗]_, _Z_ [(0)], _Z_ [(1)] ); middle—all-pairs activation–Pearson ( _Z_ [(0)] vs. _Y_ _[∗]_, _Z_ [(1)] vs. _Y_ _[∗]_, _Z_ [(0)]

vs. _Z_ [(1)] ); bottom—all-pairs decoder–cosine ( _D_ [(0)] vs. _D_ _[∗]_, _D_ [(1)] vs. _D_ _[∗]_, _D_ [(0)] vs. _D_ [(1)] ).


Figure 7: **Random** **MSAE** **(example** **seed** **pair).** Same layout as Fig. 5; top—activation
rasters; middle—activation–Pearson; bottom—decoder–cosine; columns are (0 vs. _Y_ _[∗]_ ), (1
vs. _Y_ _[∗]_ ), (0 vs. 1).


16


Figure 8: **O-SAE** **(example** **seed** **pair).** Same layout as Fig. 5. Top row rasters (50
inputs); middle row all-pairs activation–Pearson for Stab _Z_ /Ord _Z_ ; bottom row all-pairs decoder–cosine extending Fig. 4 to all pairs.


we draw _N_ = 50 _,_ 000 samples. Unless noted otherwise: Gaussian features, Zipf exponent
_α_ swept across panels, 30 _,_ 000 training steps, learning rate 10 _[−]_ [4], _ℓ_ 1 coefficient 0 _._ 01, and
results are averaged over 5 seeds. We compare to TopK, two Matryoshka variants (“fixed”
and “random”), and a vanilla SAE. We weighted the features for Matryoshka and Ordered
SAEs by the Zipfian alpha value of the data. For fixed Matryoshka, we used 8 groups. For
random Matryoshka, we used 4 truncations.


**Results.** Figure 10 shows that O-SAEs achieve _consistency_ comparable to the strongest
baselines: for small _α_ (near-uniform usage) ground-truth stability is _≈_ 0 _._ 9 for O-SAE, TopK,
and Matryoshka, and remains competitive as skew increases. Unlike TopK and vanilla, OSAEs also exhibit _orderedness_ : the learned atom ordering correlates with the ground-truth
order (Spearman _ρ_ _≈_ 0 _._ 5 at low _α_, remaining positive across the sweep), while pairwise
orderedness likewise improves relative to vanilla. This ordering bias comes with a trade-off
in global _ℓ_ 2 reconstruction error, where O-SAEs are higher than TopK/vanilla.


A final observation is that our _Frequency-Invariant_ _Feature_ _Reconstruction_ _(FIFR)_ _Error_
(Sec. B.4) tracks dictionary stability across methods and _α_ much better than the global
MSE. In ordered/Zipfian regimes, rare features contribute little to MSE and can be underfit
without a visible penalty, whereas FIFR Error exposes such failures. Empirically, when
TopK and O-SAEs attain a FIFR Error comparable to vanilla SAEs, their ground-truth
stability also converges to vanilla, despite differences in global MSE.


Going forward, we hypothesize that with better hyperparameter tuning and methods such
as unit sweeping, it is possible to achieve lower L2 and FIFR error with O-SAEs and thus
higher orderedness than baseline architectures while maintaining 0.9 consistency.


B.4 MSE underweights rare features; a frequency-invariant error


**Motivation.** In ordered/Zipfian settings, some features appear far more often than others.
The global reconstruction MSE E _∥x −_ _x_ ˆ _∥_ [2] 2 [therefore] [emphasizes] [frequent] [features] [and] [can]


17


Vanilla O - SAE

(a) ( _K, m_ ) = (10 _,_ 2): example seed pair.


Vanilla O - SAE

(b) ( _K, m_ ) = (30 _,_ 3): example seed pair.


Vanilla O - SAE

(c) ( _K, m_ ) = (50 _,_ 5): example seed pair.


Figure 9: Top: activation rasters for 50 inputs; middle: all-pairs activation–Pearson; bottom:
all-pairs decoder–cosine. Columns within each row are (0 vs. _Y_ _[∗]_ ), (1 vs. _Y_ _[∗]_ ), (0 vs. 1).


look “good” while rare features are poorly reconstructed. We seek a metric that (i) treats
each feature equally regardless of frequency and (ii) scores the fidelity of its _per-feature_
contribution to the reconstruction.


**Definition** **(Frequency-Invariant** **Feature** **Reconstruction** _**Error**_ **).** Let the groundtruth dictionary be _A_ _[⋆]_ _∈_ R _[m][×][n]_ with atoms (rows) _a_ _[⋆]_ _j_ [, true codes] _[ S][⋆]_ _[∈]_ [R] _[N]_ _[×][m]_ [, learned decoder]
_A ∈_ R _[m][×][n]_ with atoms _ak_, and inferred features _F ∈_ R _[N]_ _[×][m]_ . We align atoms by Hungarian


18


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


Figure 10: **Zipfian** **toy-model** **comparison** **of** **SAEs** **(5** **seeds).** Each panel sweeps
the Zipf exponent _α_ controlling activation skew (higher _α_ _⇒_ rarer tail features). _Top_ _row:_
O-SAE (Ordered TopK) matches TopK/Matryoshka on ground-truth and pairwise stability
( _∼_ 0 _._ 9 at low _α_, remaining competitive as skew rises). _Middle_ _row:_ O-SAE exhibits positive
orderedness (Spearman _ρ_ with ground-truth ordering _≈_ 0 _._ 5 at low _α_ ; pairwise orderedness
likewise improves), unlike vanilla/TopK. _Bottom_ _row:_ O-SAE has higher global _ℓ_ 2 error,
but the proposed _Frequency-Invariant_ _Feature_ _Reconstruction_ _(FIFR)_ _Error_ better predicts
stability across methods and _α_ : when FIFR Error aligns across methods, ground-truth
stability aligns as well. Experimental details: input dim 16, dictionary size 32, _k_ = 3,
_N_ =50k, Gaussian features, 30k steps, lr=10 _[−]_ [4], _ℓ_ 1 =0 _._ 01.


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


**Properties.** (i) _Frequency-invariant:_ macro-averaging across features prevents frequent
atoms from dominating. (ii) _Per-feature_ _scale-invariant:_ normalizing by the energy of _c_ _[⋆]_ _ij_
removes dictionary–code scaling ambiguity. (iii) _Permutation-invariant:_ alignment via _π_
factors out atom ordering. (iv) _Interpretable:_ FIFR Error equals 0 iff per-feature components are recovered exactly; larger values indicate worse reconstruction (and can exceed
1). As seen in Fig. 10, FIFR Error correlates strongly with dictionary stability in Zipfian
regimes, whereas global MSE does not.


20


assignment on absolute correlations of _ℓ_ 2-normalized atoms:


_a_ _[⋆]_ _j_ _ak_
_a_ ˜ _[⋆]_ _j_ [=] _∥a_ _[⋆]_ _j_ _[∥]_ [2] _,_ _a_ ˜ _k_ = _∥ak∥_ 2 _,_ _Cjk_ = _⟨a_ ˜ _[⋆]_ _j_ _[,]_ [ ˜] _[a][k][⟩][,]_ _π_ _∈_ arg max _σ_


_m_

- _|Cj,σ_ ( _j_ ) _|._


_j_ =1


For feature _j_, let _Ij_ = _{i_ : _s_ _[⋆]_ _ij_ [= 0] _[}]_ [.] [Define] [true] [and] [estimated] [per-sample] [components]

_c_ _[⋆]_ _ij_ [=] _[ s]_ _ij_ _[⋆]_ _[a]_ _j_ _[⋆][,]_ _c_ ˆ _ij_ = _fi,π_ ( _j_ ) _aπ_ ( _j_ ) _._


With _ε_ = 10 _[−]_ [12],


_∈Ij_ _ij_ _[ij]_ 2 FIFR( _A_ _[⋆]_ _, S_ _[⋆]_ ; _A, F_ ) = [1]

_i∈Ij_ _[∥][c]_ _ij_ _[⋆]_ _[∥]_ 2 [2] [+] _[ ε]_ _[,]_ _|J_


_rj_ =


1 _|Ij_ _|_


1 _j_ _|_ _i∈Ij_ _[∥][c]_ _ij_ _[⋆]_ _[−]_ _[c]_ [ˆ] _[ij][∥]_ 2 [2]

1 _|Ij_ _|_ _i∈Ij_ _[∥][c]_ _ij_ _[⋆]_ _[∥]_ 2 [2] [+] _[ ε]_


_|J|_


- _rj,_ _J_ = _{j_ : _|Ij| >_ 0 _}._


_j∈J_


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


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||0<br>500<br>|0<br>500<br>|0<br>500<br>|k<br><br>20M<br>25M<br>|k<br><br>20M<br>25M<br>||
||0<br>500<br>|0<br>500<br>|0<br>500<br>|k<br><br>20M<br>25M<br>|k<br><br>20M<br>25M<br>|k<br><br>20M<br>25M<br>|
|||2.5M<br>5M<br>10M<br>15M|2.5M<br>5M<br>10M<br>15M|30M<br>35M<br>40M<br>45M|||
|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|101<br>102<br>103<br>Prefix Length of Dictionary<br>OSAE on Dolma|
||||||||
||||||||
||||||||
||20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>||||
|0<br>500k<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|
|2.5M<br>5M<br>10M<br>15M|30M<br>35M<br>40M<br>45M|30M<br>35M<br>40M<br>45M|||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|0<br>500<br>2.5<br>5M|20M<br>k 25M<br>M 30M<br>35M|
|---|---|---|---|---|
|||~~10~~<br>15|~~10~~<br>15|~~M~~<br><br><br>~~40M~~<br>45M|
|||~~10~~<br>15|~~10~~<br>15||
||||||
||||||
||||||
|101<br>102<br>103<br>Prefix Length of Dictionary<br>Random MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Random MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Random MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Random MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Random MSAE on Dolma<br><br>|
||||0<br>500<br>2.5<br>5M<br>|k<br>M<br><br>20M<br>25M<br>30M<br>35M<br>|
|||~~10~~<br>15|~~10~~<br>15|~~M~~<br><br><br>~~40M~~<br>45M|
|||~~10~~<br>15|~~10~~<br>15||
||||||
||||||
||||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|Col4|0<br>500<br>2.5<br>5M|Col6|20M<br>k 25M<br>M 30M<br>35M|
|---|---|---|---|---|---|---|
||||~~10~~<br>15|~~10~~<br>15|~~10~~<br>15|~~M~~<br><br><br>~~40M~~<br>45M|
||||~~10~~<br>15|~~10~~<br>15|~~10~~<br>15||
||||||||
||||||||
||||||||
|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|101<br>102<br>103<br>Prefix Length of Dictionary<br>Fixed MSAE on Dolma<br><br>|
|||0<br>500<br>2.5M<br>5M<br>|k<br><br><br>20M<br>25M<br>30M<br>35M<br>|k<br><br><br>20M<br>25M<br>30M<br>35M<br>|||
||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|~~40M~~<br>45M|~~40M~~<br>45M||
||||||||
||||||||
||||||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


Figure 11: Prefix **Ord(** _D_ **,** _D_ _[′]_ **)** plotted with increasing training tokens. Top row is trained
on the Pile, and the bottom row is trained on Dolma. (n=1 pair of seeds)


21


C Empirical results


Figure 11 and 12 show how orderedness and stability metrics change as SAEs are trained
for O-SAE, Random MSAE, and Fixed MSAE on the Pile and Dolma. Orderedness and
stability tend to increase over the 45M tokens illustrated in the figures, with some exceptions
at lower prefix lengths going down while higher prefix length measures increase.


Prefix Orderedness with More Training Tokens


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


OSAE on Pile


Random MSAE on Pile


Fixed MSAE on Pile


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


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


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|0<br>500k<br>2.5M<br>5M|20M<br>25M<br>30M<br>35M|
|---|---|---|---|---|---|
||||~~10M~~<br>15M|~~10M~~<br>15M|~~40M~~<br>45M|
||||~~10M~~<br>15M|~~10M~~<br>15M||
|||||||
|||||||
|||||||


Figure 12: Prefix **Stab(** _D_ **,** _D_ _[′]_ **)** plotted with increasing training tokens. Top row is trained
on the Pile, and the bottom row is trained on Dolma. (n=1 pair of seeds)


D Empirical results - SAE Stitching


**O-SAEs** **decrease** **the** **number** **of** **novel** **features** **found** **by** **SAE** **Stitching**


A key limitation of standard SAEs is their incompleteness, since they often fail to recover
the full set of canonical features in a model’s representations. Prior work Leask et al. (2025)
highlights this issue using SAE stitching. In this procedure, we take a feature (latent)
discovered by a larger SAE and “stitch” it into a smaller SAE. If this stitched latent improves
reconstruction performance, it suggests that the smaller SAE was missing this information
entirely, which means the larger SAE has uncovered a novel feature. If reconstruction
worsens instead, the stitched latent is overlapping with existing ones, which means the SAE
is redundantly encoding the same information. We call this a reconstruction feature.


Our experiments show that O-SAEs substantially reduce the fraction of novel features discovered via stitching. In other words, O-SAEs capture more of the underlying structure up
front, leaving fewer important features uncovered compared to standard SAEs. This reduction in incompleteness directly addresses one of the main critiques of sparse autoencoders:
while traditional SAEs leave gaps in the feature set, O-SAEs close those gaps by providing
a more complete and less redundant decomposition.


**SAE** **Type** **Novel** **Feature** **%** **Reconstruction** **%** **No** **MSE** **Change** **%**
BatchTopK 73.8% 21.2% 5.0%
Random MSAE 52.7% 11.4% 35.9%
O-SAE **33.8%** **64.8%** 1.4%


Table 2: Novel Feature, Reconstruction, and No MSE Change Percentages of various SAE
types when stitching 65536-sized features into the corresponding 4096-sized SAE.


In Table 2, the BatchTopK baseline demonstrates 73.8% novel features, indicating strong
incompleteness. While O-SAE’s 33.8% novel features are still substantial but better than
the baseline. Random MSAE falls in between at 52.7% novel features with the caveat that


22


Prefix Stability with More Training Tokens


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


OSAE on Pile


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary

OSAE on Dolma


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


Random MSAE on Pile


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary

Random MSAE on Dolma


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


Fixed MSAE on Pile


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary

Fixed MSAE on Dolma


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


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


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||20M<br>25M<br>|20M<br>25M<br>|||
|0<br>500k<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|
|2.5M<br>5M<br>10M<br>15M|30M<br>35M<br>40M<br>45M||||


|Col1|Pile against|Col3|its Initialization|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||0<br>|0<br>|20M<br>|20M<br>||
||0<br>|0<br>|20<br>|20<br>|20<br>|
|||~~500~~<br>2.5M<br>5M<br>10M<br>15M|~~25~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||0<br>|0<br>|20M<br>|20M<br>||
||0<br>|0<br>|20M<br>|20M<br>|20M<br>|
|||~~500~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||20M<br>25M<br>|20M<br>25M<br>|||
||0<br>500k<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|
||2.5M<br>5M<br>10M<br>15M|30M<br>35M<br>40M<br>45M||||
|||||||
|||||||


|Col1|500k 25<br>2.5M 30<br>5M 35<br>10M 40<br>15M 45|M<br>M<br>M<br>M<br>M|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||20M<br>|20M<br>||
|||0<br>|20M<br>|20M<br>|20M<br>|
|||~~500k~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||


Figure 13: O-SAE Orderedness and Stability. (left) Cross dataset compares O-SAE trained
on Pile against O-SAE trained Dolma. They use the same seed, so initial checkpoints start
with 1.0 orderedness and stability. (middle) Shows the progression of checkpoints from OSAE trained on the Pile, when compared against its initialized checkpoint. This gives a
relative measure of deviation from initialization. (right) Progression of checkpoints trained
on Dolma compared against its initialized checkpoint.


23


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


Cross Dataset


O-SAE Orderedness with Training Progression


1.0


0.8


Dolma against its Initialization


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


Cross Dataset


0.6


0.4


0.2


0.0


O-SAE Stability with Training Progression


Pile against its Initialization


Dolma against its Initialization


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


1.0


0.8


0.6


0.4


0.2


0.0


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


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||20M<br>25M<br>|20M<br>25M<br>|||
|0<br>500k<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|
|2.5M<br>5M<br>10M<br>15M|30M<br>35M<br>40M<br>45M||||


|Col1|Pile against|Col3|Col4|its Initialization|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||0<br>|0<br>|0<br>|20M<br>|20M<br>||
||0<br>|0<br>|0<br>|20M<br>|20M<br>|20M<br>|
||||~~500~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
||||||||
||||||||
|101|101|102<br>103<br>Prefix Length of Dictionary|102<br>103<br>Prefix Length of Dictionary|102<br>103<br>Prefix Length of Dictionary|102<br>103<br>Prefix Length of Dictionary|102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||0<br>|0<br>|20M<br>|20M<br>||
||0<br>|0<br>|20M<br>|20M<br>|20M<br>|
|||~~500~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||
|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|101<br>102<br>103<br>Prefix Length of Dictionary|


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||20M<br>25M<br>|20M<br>25M<br>|||
||0<br>500k<br><br><br><br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|20M<br>25M<br>|
||2.5M<br>5M<br>10M<br>15M<br><br><br><br><br>|30M<br>35M<br>40M<br>45M||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||||20M<br>|20M<br>||
|||0<br>|20M<br>|20M<br>|20M<br>|
|||~~500k~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||||20M<br>|20M<br>||
|||0<br>|20M<br>|20M<br>|20M<br>|
|||~~500k~~<br>2.5M<br>5M<br>10M<br>15M|~~25M~~<br>30M<br>35M<br>40M<br>45M|||
|||||||
|||||||


Figure 14: Random MSAE Orderedness and Stability. (left) Cross dataset compares Random MSAE trained on Pile against Random MSAE trained Dolma. They use the same
seed, so initial checkpoints start with 1.0 orderedness and stability. (middle) Shows the progression of checkpoints from Random MSAE trained on the Pile, when compared against its
initialized checkpoint. This gives a relative measure of deviation from initialization. (right)
Progression of checkpoints trained on Dolma compared against its initialized checkpoint.


24


Dolma against its Initialization


Cross Dataset


Random MSAE Orderedness with Training Progression


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


Cross Dataset


Random MSAE Stability with Training Progression


Pile against its Initialization


1.0


0.8


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


Dolma against its Initialization


10 [0] 10 [1] 10 [2] 10 [3]

Prefix Length of Dictionary


0.6


0.4


0.2


0.0


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


it has a higher fraction of non-activating features for the limited number of samples tested
on. This shows how increasing degrees of hierarchy decrease the novel percentage between
65536 and 4096 sized SAEs.


25