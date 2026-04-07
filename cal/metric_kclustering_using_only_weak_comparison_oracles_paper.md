# METRIC k -CLUSTERING USING ONLY WEAK COMPAR## ISON ORACLES


**Rahul Raychaudhury** [1] **Aryan Esmailpour** [2] **Sainyam Galhotra** [3] **Stavros Sintos** [2]

1Duke University 2University of Illinois Chicago 3Cornell University


ABSTRACT


Clustering is a fundamental primitive in unsupervised learning. However, classical algorithms for _k_ -clustering (such as _k_ -median and _k_ -means) assume access
to exact pairwise distances, which is an unrealistic requirement in many modern
applications. We study clustering in the _Rank-model (R-model)_, where access to
distances is entirely replaced by a _quadruplet_ _oracle_ that provides only relative
distance comparisons. In practice, such an oracle can represent learned models or
human feedback, and is expected to be noisy and entail an access cost.
Given a metric space with _n_ input items, we design randomized algorithms that,
using only a noisy quadruplet oracle, compute a set of _O_ ( _k ·_ polylog( _n_ )) centers
along with a mapping from the input items to the centers such that the clustering
cost of the mapping is at most constant times the optimum _k_ -clustering cost. Our
method achieves a query complexity of _O_ ( _n · k_ _·_ polylog( _n_ )) for arbitrary metric spaces and improves to _O_ (( _n_ + _k_ [2] ) _·_ polylog( _n_ )) when the underlying metric
has bounded doubling dimension. When the metric has bounded doubling dimension we can further improve the approximation from constant to 1 + _ε_, for any
arbitrarily small constant _ε_ _∈_ (0 _,_ 1), while preserving the same asymptotic query
complexity. Our framework demonstrates how noisy, low-cost oracles, such as
those derived from large language models, can be systematically integrated into
scalable clustering algorithms.


1 INTRODUCTION


Clustering is a fundamental problem in unsupervised learning. Traditional methods like _k_ -center,
_k_ -median, and _k_ -means all rely on computing pairwise distances. For their output clusters to be
meaningful, these distances must reflect the user’s notion of semantic similarity. However, designing
such tailored distance measures is especially difficult for complex data like images. Even when
a distance function is well defined, evaluating distances between certain types of objects can be
prohibitively expensive.


Motivated by these challenges, there has been a long line of work that avoids direct distance computations and instead uses _oracles_ . Oracles serve as abstractions for machine learning models or
human feedback that provide partial information about the relative distance between the points.
Oracle-based models have been studied for _k_ -clustering Bateni et al. (2024); Braverman et al.
(2025a); Addanki et al. (2021); Galhotra et al. (2024); Raychaudhury et al. (2025), hierarchical
clustering Emamjomeh-Zadeh & Kempe (2018); Chatziafratis et al. (2018); Ghoshdastidar et al.
(2019), correlation clustering Ukkonen (2017); Silwal et al. (2023) among others.


In this paper, we study clustering in the _Rank-model_ (R-model), where pairwise distances are inaccessible. Instead, one has access to a noisy quadruplet oracle, a function that, given two pairs of
input items ( _A, B_ ) and ( _C, D_ ), answers the question: _“Is A closer to B, or is C_ _closer to D?”_ . Intuitively, quadruplet queries are easier than direct distance queries because they are inherently local
and require only relative comparisons, compared to distances which are global. Quadruplet queries
are also more practical than the commonly studied optimal-cluster queries, which must return the
correct clusters for the queried points. In practice, a quadruplet oracle can be realized in several
ways. A natural option is to leverage a large language model (LLM). For instance, two candidate
pairs can be presented within a fixed prompt that specifies the intuitive similarity metric of interest,
and the model is then asked to return a categorical judgment. Another option is to use an online embedding service: embeddings are computed for each object individually, and the oracle’s decision


1


is obtained by comparing similarity scores for pairs ( _A, B_ ) and ( _C, D_ ), returning whichever pair
appears more similar. A further possibility is to train a dedicated quadruplet oracle using learningto-rank methods on annotated data Liu et al. (2009), where the labels themselves may come from
crowdsourcing. Regardless of the implementation, we generally expect the oracle to be noisy and
to have some cost associated with access. For example, with embedding-based oracles, accuracy
depends on how well the embedding space aligns with the semantic notion of similarity. In terms of
access costs, LLMs and embedding services incur a direct financial cost.


The study of clustering in the R-model was initiated by Addanki et al. (2021), who considered
problems such as _k_ -center and hierarchical clustering. Subsequently, Galhotra et al. (2024) showed
that no _o_ ( _n_ )-approximation, where _n_ is the number of items in the metric space, is possible for
_k_ -median and _k_ -means clustering without distance information and introduced the Rank-Measure
(RM) model: Along with a quadruplet oracle, they allow access to a distance oracle that returns
the exact distance between two input items. They further established several results in this setting.
Recently, Raychaudhury et al. (2025), showed that _k_ -clustering is possible in the RM-model using
_O_ ( _nk_ polylog _n_ ) quadruplet queries and only _O_ (polylog _n_ ) distance queries. When the doubling
dimension of the input metric space is bounded, they further improve the quadruplet queries to
_O_ (( _n_ + _k_ [2] )polylog _n_ ) while distance queries remain _O_ (polylog _n_ ). These query complexities are
near-optimal within logarithmic factors.


Although these results provide strong guarantees, two challenges remain. First, in practice, a strong
distance oracle may just not be available to evaluate distances accurately. Second, it is critical
in prior work to interleave distance queries with quadruplet queries, which can be problematic.
For example, obtaining exact distances may itself require solving an NP-hard problem, creating a
computational bottleneck. Motivated by these considerations, in this paper, we ask, what is the best
we can do for clustering when we have access only to a noisy quadruplet oracle? In Appendix A,
we show that at least 2 _k −_ 1 centers are necessary to obtain any _o_ ( _n_ )-approximation algorithm for
_k_ -median/means clustering in the R-model. Hence, we ask the following questions:


_“In the R-model, can we compute a set of O_ ( _k_ polylog _n_ ) _centers and a mapping from each item to_
_a center, using O_ ( _n k_ polylog _n_ ) _quadruplet queries, such that the clustering cost is comparable to_
_the optimal cost with k centers?”_


In many practical settings, the data live in high dimensions but have low intrinsic complexity (e.g.,
small doubling dimension) Nakis et al. (2025); Roweis & Saul (2000); Tenenbaum et al. (2000).
For example, Euclidean space with a fixed number of dimensions is a metric space with a constant doubling dimension. A natural question is whether this additional structure may be beneficial.
Specifically, we ask:


_“When the intrinsic dimensionality is small, can we reduce the query complexity to O_ ( _n_ polylog _n_ ) _?_
_Moreover, can we improve the approximation quality?”_


In this paper, we provide affirmative answers to both questions. We emphasize that our algorithms
do not directly output exactly _k_ centers. Instead, they return a set of _O_ ( _k_ polylog _n_ ) centers together
with an assignment of every input point to one of these centers, yielding a clustering whose cost is
within a constant factor of the optimal _k_ -clustering cost. We note that having such a small set of
centers along with a mapping function is very useful in practice, as it shifts the burden of clustering
to a substantially smaller set. For instance, consider clustering MNIST digits 0 _, . . .,_ 9. If one can
extract a representative subset of only a few hundred images along with a good mapping, then human
annotators would only need to identify the correct class of these, while the mapping automatically
ensures that the remaining images are mapped to the correct cluster. Next, we present the formal
setting and summarize our contributions.


1.1 PROBLEM SETUP AND CONTRIBUTIONS


We require some preliminary definitions before formally presenting the model and our results. Let
Σ = (V _, d_ ) be a finite metric space with _d_ : V _×_ V _→_ R _≥_ 0. We consider metric spaces with _|_ V _|_ = _n_ .
Any such space can be viewed as a weighted complete graph. We use E to denote the set of all edges
between vertices in V. The _doubling_ _dimension_ dim(Σ) is the smallest _δ_ such that every ball of
radius _ρ_ can be covered by 2 _[δ]_ balls of radius _ρ/_ 2. We say Σ has bounded doubling dimension if
dim(Σ) _≤_ _δ_ 0 for some fixed constant _δ_ 0.


2


**Clustering Cost.** For v _∈_ V and U _⊆_ V, define _d_ (v _,_ U) = minu _∈_ U _d_ (v _,_ u). Let _k, p_ _∈_ Z _≥_ 1. For
U _,_ W _⊆_ V, let COST _[p]_ U [(][W][)] [=] [�] w _∈_ W [(] _[d]_ [(][w] _[,]_ [ U][))] _[p]_ [.] [For][ W] _[⊆]_ [V][, the optimal][ (] _[k, p]_ [)][-clustering cost is]

OPT _[p]_ _k_ [(][W][)] [=] [min][U] _[⊆]_ [V] _[,][ |]_ [U] _[|]_ [=] _[k]_ [ COST] _[p]_ U [(][W][)][;] [if][ W] [=] [V][, we simply write][ OPT] _[p]_ _k_ [.] [A] _[ β]_ [-approximation]
algorithm for ( _k, p_ ) clustering returns a set _A_ such that COST _[p]_ _A_ [(][V][)] _[≤]_ _[β]_ _[·]_ [ OPT] _[p]_ _k_ [,] [where] _[β]_ _[≥]_ [1][.]
For _p_ = 1 _,_ 2 the ( _k, p_ ) clustering corresponds to _k_ -median and _k_ -means clustering, respectively.
Throughout the paper we assume that _p_ = _O_ (1).


**Coresets** **and** **Coreset+.** Next, we recall the standard definition of a coreset and introduce the
stronger notion of a Coreset+.


_O_ (1)-coreset: A set T _⊆_ V along with a weight function _ω_ : T _→_ R _>_ 0 is called an _O_ (1)-coreset
for ( _k, p_ )-clustering if any subset _A_ _⊆_ T such that _|A|_ = _k_ and [�] t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,][ A]_ [))] _[p]_ _[≤]_ _[β]_ _[·]_


for ( _k, p_ )-clustering if any subset _A_ _⊆_ T such that _|A|_ = _k_ and [�] t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,][ A]_ [))] _[p]_ _[≤]_ _[β]_ _[·]_

minU _⊆_ T _, |_ U _|_ = _k_ �t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,]_ [ U][))] _[p]_ [ satisfies COST] _A_ _[p]_ [(][V][)] _[ ≤]_ _[O]_ [(] _[β]_ [)] _[ ·]_ [ OPT] _[p]_ _k_ [.]


( _k, ε_ )-coreset: For a real number _ε ∈_ (0 _,_ 1), a set T _⊆_ V along with a weight function _ω_ : V _→_ R _>_ 0
is called a ( _k, ε_ )-coreset for ( _k, p_ )-clustering if for any subset _A_ _⊆_ V with _|A|_ = _k_, it holds that
(1 _−_ _ε_ )COST _[p]_ _A_ [(][V][)] _[ ≤]_ [�] t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,][ A]_ [))] _[p]_ _[≤]_ [(1 +] _[ ε]_ [)][COST] _[p]_ _A_ [(][V][)][.]


t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,]_ [ U][))] _[p]_ [ satisfies COST] _A_ _[p]_ [(][V][)] _[ ≤]_ _[O]_ [(] _[β]_ [)] _[ ·]_ [ OPT] _[p]_ _k_ [.]


( _k, ε_ )-coreset: For a real number _ε ∈_ (0 _,_ 1), a set T _⊆_ V along with a weight function _ω_ : V _→_ R _>_ 0
is called a ( _k, ε_ )-coreset for ( _k, p_ )-clustering if for any subset _A_ _⊆_ V with _|A|_ = _k_, it holds that
(1 _−_ _ε_ )COST _[p]_ _A_ [(][V][)] _[ ≤]_ [�] t _∈_ T _[ω]_ [(][t][)(] _[d]_ [(][t] _[,][ A]_ [))] _[p]_ _[≤]_ [(1 +] _[ ε]_ [)][COST] _[p]_ _A_ [(][V][)][.]

_α_ - _Coreset+_ : For a positive real number _α_, a set C _⊆_ V together with a mapping _M_ : V _→_ C is
called an _α_ -Coreset+ if [�] _v∈_ V [(] _[d]_ [(] _[v,][ M]_ [(] _[v]_ [)))] _[p]_ _[≤]_ _[α][ ·]_ [ OPT] _[p]_ _k_ [.]

It follows from well-known results (e.g., Har-Peled & Mazumdar (2004); Chen (2009)) that any
_O_ (1)-Coreset+ directly yields an _O_ (1)-coreset: define the weight function _ω_ : C _→_ R _>_ 0 by
_ω_ ( _c_ ) = [�] _v∈_ V **[1]** _[{M]_ [(] _[v]_ [)] [=] _[c][}]_ [.] [Furthermore,] [for] _[p]_ [=] [1][,] [it] [follows] [that] [an] _[ε]_ [-][Coreset+] [can] [be]
converted to a ( _k, ε_ )-coreset in the same way.


_α_ - _Coreset+_ : For a positive real number _α_, a set C _⊆_ V together with a mapping _M_ : V _→_ C is
called an _α_ -Coreset+ if [�] _v∈_ V [(] _[d]_ [(] _[v,][ M]_ [(] _[v]_ [)))] _[p]_ _[≤]_ _[α][ ·]_ [ OPT] _[p]_ _k_ [.]


**R-model.** In the R-model, direct access to the distance function _d_ is unavailable. Instead, all
distance-related information must be obtained through a noisy quadruplet oracle. Formally, for an
error parameter _φ ∈_ �0 _,_ [1] 4 �, a _quadruplet oracle with probabilistic noise φ_ is a function _Q_ [˜] : E _×_ E _→_

_{_ YES _,_ NO _}_ that, given two edges **e** 1 _,_ **e** 2 _∈_ E, where **e** 1 = _{_ v1 _,_ v2 _}_, and **e** 2 = _{_ v3 _,_ v4 _}_, outputs


_Q_ ˜( **e** 1 _,_ **e** 2) = �YES, with probability at least 1 _−_ _φ,_ if _d_ (v1 _,_ v2) _≤_ _d_ (v3 _,_ v4) _,_
NO, with probability at least 1 _−_ _φ,_ if _d_ (v1 _,_ v2) _> d_ (v3 _,_ v4) _._


In other words, the oracle fails to identify the closer pair with probability at most _φ_ . Furthermore,
the randomness is independent across distinct queries and is fixed once per edge pair; thus, repeated
calls to _Q_ [˜] ( **e** 1 _,_ **e** 2) always return the same result, and flipping the order of the edges always flips the
answer. This property is referred to as _persistence_ .


**Our Results.** As a warm-up, in Appendix A, we show that at least 2 _k −_ 1 centers are necessary
to obtain any _o_ ( _n_ )-approximation algorithm for _k_ -median/means clustering in the R-model. On the
other hand, we obtain the following positive results for clustering in the R-model.

|Col1|Col2|General|Col4|Doubling|Col6|
|---|---|---|---|---|---|
|||centers|queries<br>|centers|queries<br>|
|RM-model|Galhotra et al. (2024)|_k_|˜_O_(_nk_), ˜_O_(_k_2)<br>|_k_|˜_O_(_nk_), ˜_O_(_k_2)<br>|
|RM-model|Raychaudhury et al. (2025)|_k_|˜_O_(_nk_), ˜_O_(1)<br>|_k_|˜_O_(_k_2 +_ n_), ˜_O_(1)<br>|
|R-model|Addanki et al. (2021)|_k_<br>|˜_O_(_nk_2)<br>|_k_<br>|˜_O_(_nk_2)<br>|
|R-model|**NEW**|˜_O_(_k_)|˜_O_(_nk_)|˜_O_(_k_)|˜_O_(_k_2 +_ n_)|


Table 1: Comparison of our algorithms for _k_ -clustering in the R-model with known clustering algorithms in the R-model and RM-model on general and doubling metric spaces. They are all _O_ (1)approximation algorithms. For every algorithm we show the number of centers it returns and the
number of oracle queries it executes. In the RM-model, the first (resp. second) quantity in the
column queries shows the number of queries to the quadruplet (resp. distance) oracle. The notation _O_ [˜] ( _·_ ) hides a polylog ( _n_ ) factor. Galhotra et al. (2024); Raychaudhury et al. (2025) studied
_k_ -median/means clustering. Addanki et al. (2021) only studied the _k_ -center clustering in the Rmodel however their algorithm only holds if the optimal clusters are of size Ω( _[√]_ _n_ ).


_•_ For general metric spaces, we give an algorithm that constructs an _O_ (1)-Coreset+ for ( _k, p_ )clustering of size _O_ ( _k_ polylog _n_ ), using _O_ ( _nk_ polylog _n_ ) queries to the quadruplet oracle.


3


_•_ For metric spaces with bounded doubling dimension (for example, the Euclidean space with constant number of dimensions), we design an algorithm that constructs an _O_ (1)-Coreset+ of size
_O_ ( _k_ polylog _n_ ), using only _O_ (( _n_ + _k_ [2] )polylog _n_ ) quadruplet queries.


_•_ For the special case of _k_ -median in doubling metrics, we further obtain an _ε_ -Coreset+ of size
_O_ ( _k_ polylog _n_ ) with the same query complexity, i.e., _O_ (( _n_ + _k_ [2] )polylog _n_ ).


Our main results and the comparison with other known algorithms in the RM and R-model are
shown in Table 1.


1.2 RELATED WORK


Due to the large body of related work, we focus mostly on clustering problems under related oraclebased models. We discuss additional related work in Appendix E.


Addanki et al. (2021) initiated the study of _k_ -center clustering with access to a quadruplet oracle. In
the R-model (with probabilistic noise), they developed an algorithm under structural assumptions on
the optimal clustering. To the best of our knowledge, this remains the only prior work on clustering
under purely the R-model.


More recently, Galhotra et al. (2024) showed that even with a perfect quadruplet oracle, no _O_ (1)approximation is possible for _k_ -means or _k_ -median without distance queries. This motivated their
weak-strong framework: a weak oracle provides inexpensive quadruplet comparisons, while a strong
oracle supplies exact distances at a higher cost. Within this framework, they designed _O_ (1)approximation algorithms for _k_ -center, _k_ -median, and _k_ -means in general metric spaces, achieving query complexities of _O_ ( _nk_ polylog _n_ ) to the quadruplet oracle and _O_ ( _k_ [2] polylog _n_ ) to the distance oracle. In a follow-up work, Raychaudhury et al. (2025) improved the number of calls to
the distance oracle from _O_ ( _k_ [2] polylog _n_ ) to _O_ (polylog _n_ ). They also study clustering problems in
metric spaces with bounded doubling dimension, designing _O_ (1)-approximation algorithms with
_O_ (( _n_ + _k_ [2] )polylog _n_ ) calls to the quadruplet oracle and _O_ (polylog _n_ ) calls to the distance oracle. Specifically, for _k_ -center clustering Raychaudhury et al. (2025) construct _O_ (1)-coreset of size
_O_ ( _k_ polylog _n_ ) executing _O_ ( _nk_ polylog _n_ ) (resp. _O_ (( _n_ + _k_ [2] )polylog _n_ ) in doubling metrics) queries
to the quadruplet oracle (R-model). However, using their techniques, no coreset construction for
_k_ -median or _k_ -means can be constructed in the R-model, i.e., an exact distance oracle is necessary.


Independently, Bateni et al. (2024) investigated _k_ -clustering and the MST problem in general metrics
under a related weak-strong framework. Their strong oracle matches that of Galhotra et al. (2024),
but their weak oracle differs: given _u, v_ _∈_ V, it outputs _d_ ( _u, v_ ) with probability at least 1 _−_ _ε_,
and an arbitrary value otherwise. They obtained _O_ (1)-approximations for _k_ -center, _k_ -median, and
_k_ -means using _O_ ( _nk_ polylog _n_ ) weak queries and _O_ ( _k_ [2] polylog _n_ ) strong queries. Under the same
strong–weak model, Braverman et al. (2025a) studied coresets for clustering problems.


There is a rich line of work on approximate sorting with a probabilistic comparison oracle. In this
model, it is well-known that the _maximum_ _dislocation_ cannot be improved beyond _O_ (log _n_ ); in
particular, no algorithm can guarantee that every element is placed within _o_ (log _n_ ) positions of its
true rank. After a series of results Braverman & Mossel (2008); Braverman et al. (2016); Geissmann
et al. (2017; 2020), it was recently shown in Geissmann et al. (2025) that _O_ (log _n_ ) dislocation can
be achieved with _O_ ( _n_ log _n_ ) queries with high probability, when the noise _φ <_ 1 _/_ 4.


2 TECHNICAL PRELIMINARIES


Let Σ = (V _, d_ ) be a metric space with _d_ : V _×_ V _→_ R _≥_ 0. We consider finite metric spaces
with _|_ V _|_ = _n_ . Any finite metric space can be viewed as a weighted complete graph, and we often
use graph terminology. For U _,_ W _⊆_ V, let E(U _,_ W) = _{{_ u _,_ w _}_ _|_ u _∈_ U _,_ w _∈_ W _,_ u = w _}_,
and E(U) := E(U _,_ U). For an edge set X, let V(X) denote the set of vertices incident to edges
in X. For **e** = _{_ u _,_ v _}_ _∈_ E(V), we define _d_ ( **e** ) := _d_ (u _,_ v). For U _,_ W _⊆_ V and u _∈_ U, define
POSW(u; U) = 1 + _|{_ x _∈_ U : _d_ (x _,_ W) _<_ _d_ (u _,_ W) _}|_, the position of u when U is ordered by
nearest-neighbor distance to W.


While in the R-model we do not have access to distances, the quadruplet oracle allows us to approximately order edges based on their weights (distance). For an edge set X _⊆_ E, let _π_ X denote an
_ordered_ _sequence_ of the edges in X. For an edge **e** _∈_ X, we use RANKX( **e** ) to denote its position


4


among the edges in X when sorted in ascending order of distance, [1] and RANK _π_ X( **e** ) for its index
in _π_ X. The _dislocation_ of **e** under _π_ X is defined as _|_ RANK _π_ X( **e** ) _−_ RANKX( **e** ) _|_ . The maximum
dislocation of _π_ X is bounded by _D_ if the dislocation of every edge in X is bounded by _D_ . It is
easy to verify that if _π_ X has maximum dislocation _D_, then any subsequence _π_ _⊑_ _π_ X also has maximum dislocation at most _D_ . The next lemma follows from a straightforward application of a result
by Geissmann et al. (2025).
**Lemma** **2.1.** _Let_ Σ = (V _, d_ ) _be_ _a_ _metric_ _with_ _|_ V _|_ = _n,_ _and_ E = E(V) _be_ _its_ _edge_ _set._ _Suppose_
_Q_ ˜ : E _×_ E _→{_ YES _,_ NO _} is a probabilistic quadruplet oracle with noise φ ∈_ [0 _,_ [1] 4 []] _[.]_ _[There exists an]_

_algorithm_ PROBSORT _such that for any_ X _⊆_ E _, with probability_ 1 _−_ _n_ _[−]_ [4] _[/]_ [3] _,_ PROBSORT(X) _outputs_
_an ordering π_ X _with maximum dislocation O_ (log _n_ ) _using O_ (max( _|_ X _|, n_ ) log _n_ ) _queries to_ _Q_ [˜] _._


A small bound on the maximum dislocation of _π_ X ensures that the ranks are approximately preserved; however, it offers no guarantees about the relative magnitudes of edges that appear in the
wrong order. For such guarantees, we require a stronger notion of approximate ordering. For an
index _i_ _∈_ [ _|_ X _|_ ], let _π_ X[ _i_ ] _∈_ X denote the edge in the _i_ -th position of _π_ X. For a constant _α_ _≥_ 1, we
say _π_ X is _α_ - _sorted_, if for all _i_ _<_ _j_ _∈_ [ _|_ X _|_ ], _d_ ( _π_ X[ _i_ ]) _≤_ _α d_ ( _π_ X[ _j_ ]). While in this paper we study
quadruplet oracles under the probabilistic noise model, prior work (see, for example Addanki et al.
(2021)) also considered a weaker _adversarial noise model_ .


_Quadruplet Oracle with Adversarial Noise._ [2] Let _µ_ _∈_ R _≥_ 0 be a constant. A quadruplet oracle with
_adversarial_ _noise_ _µ_ is a function _Q_ : E _×_ E _→{_ YES _,_ NO _}_ that, given two edges **e** 1 = _{_ v1 _,_ v2 _}_
1
and **e** 2 = _{_ v3 _,_ v4 _}_, outputs YES if _d_ (v1 _,_ v2) _≤_ 1+ _µ_ _[d]_ [(][v][3] _[,]_ [ v][4][)][,] [N][O][ if] _[ d]_ [(][v][1] _[,]_ [ v][2][)] _[≥]_ [(1 +] _[ µ]_ [)] _[ d]_ [(][v][3] _[,]_ [ v][4][)][,]
and an adversarially chosen (non-adaptive) answer whenever the ratio _d_ (v1 _,_ v2) _/d_ (v3 _,_ v4) lies in the
interval [1 _/_ (1 + _µ_ ) _,_ 1 + _µ_ ].


In other words, the oracle gives the correct response when the edge weights are not relatively close
but may be adversarially wrong otherwise. The next lemma shows that under the adversarial noise
model, it is possible to compute an _O_ (1)-sorted sequence of edges using the quadruplet oracle.
**Lemma 2.2.** _Let_ Σ = (V _, d_ ) _be a metric with |_ V _|_ = _n,_ E = E(V) _the edge set, and Q an adversarial_
_quadruplet oracle with noise µ_ _≥_ 0 _._ _There exists an algorithm_ ADVSORT _such that for any_ X _⊆_ E
_of_ _size_ _m,_ _with_ _probability_ 1 _−_ _n_ _[−]_ [4] _,_ ADVSORT(X) _outputs_ _a_ (1 + _µ_ ) [2] _-sorted_ _sequence_ _π_ X _using_
_O_ ( _m_ polylog _n_ ) _queries to Q._


The proof of Lemma 2.2 can be found in Raychaudhury et al. (2025) and is based on a similar result
by Acharya et al. (2018). The actual algorithm, ADVSORT, is a slight modification of the classic
randomized QUICKSORT algorithm.
Although in this paper we work in the probabilistic noise model, in the analysis of our algorithms, we
show that when certain very specific structural conditions hold, it is feasible to emulate an adversarial
quadruplet oracle by appropriate calls to the probabilistic quadruplet oracle. In such situations, we
will be able to plug in an _emulated adversarial quadruplet oracle_ into ADVSORT to obtain an _O_ (1)sorted ordering of edges. We discuss more in the technical overview.


3 TECHNICAL OVERVIEW


Let Σ = (V _, d_ ) be a metric space accessible in the R-model, i.e., there exists a noisy probabilistic
quadruplet oracle _Q_ [˜] that compares edge weights in Σ. We assume that the error rate of _Q_ [˜] satisfies
_φ_ _<_ 1 _/_ 4. Our goal is to design an algorithm which, given parameters _k, p_ _∈_ Z _≥_ 1 and access to
_Q_ ˜, returns a Coreset+ for ( _k, p_ )-clustering of size _O_ ( _k_ polylog _n_ ). For simplicity, we focus on the
_k_ -median objective ( _p_ = 1), but our approach extends naturally to any constant _p >_ 1.
The remainder of this section is organized as follows. We first describe a generic approach for
computing a Coreset+ under a perfect quadruplet oracle. We then present our algorithm for
general metric spaces, ALG-G, which adapts this high-level strategy to the noisy setting using
_O_ ( _n k_ polylog _n_ ) queries. Next, we introduce ALG-D, which further reduces the query complexity
to _O_ (( _n_ + _k_ [2] ) polylog _n_ ) when Σ has bounded doubling dimension. Finally, we outline a refinement method, ALG-DI, which takes a Coreset+ returned by the previous algorithm and builds an


1For simplicity, we assume all pairwise distances are unique.
2Most of our results extend to the adversarial model, but we focus on the more challenging probabilistic
case.


5


_ε_ -Coreset+ with _O_ ( _n_ polylog _n_ ) additional queries. Due to space constraints, full details and proofs
of these algorithms are deferred to Appendix B, C, and D, respectively.


Let C _[⋆]_ denote an optimal _k_ -median solution, i.e., _|_ C _[⋆]_ _|_ = _k_ and COST [1] C _[⋆]_ [(][V][) =][ OPT][1] _k_ [.] [Recall that a]
Coreset+ is defined as a set C _⊆_ V together with a mapping _M_ : V _→_ C such that [�] v _∈_ V _[d]_ [(][v] _[,]_ [ C][)] _[ ≤]_

_O_ (1) _·_ OPT [1] _k_ [for all][ v] _[∈]_ [V][.] [The generic algorithm is based on a well-known] _[ sampling property]_ [ of]
_k_ -median clustering Har-Peled & Mazumdar (2004); Mettu & Plaxton (2002). Suppose we take a
random sample _S_ _⊆_ V of size Θ( _k_ polylog _n_ ). We call a vertex v _∈_ V _good_ if _d_ (v _, S_ ) _<_ 2 _d_ (v _,_ C _[⋆]_ ),
and _bad_ otherwise. Let V _b_ _⊆_ V denote the set of bad vertices. It can be shown that, with high
probability, _|_ V _b|_ = _o_ ( _|_ V _|_ ).


**Generic Algorithm.** The above sampling property leads to a natural recursive sampling algorithm.
Sample a set _S_ _⊆_ V of _O_ ( _k_ polylog _n_ ) vertices, order the vertices in V in ascending order by their
nearest-neighbor distance to _S_, remove a constant fraction subset of the first half from V, and recurse.
The process continues until there are _O_ ( _k_ polylog _n_ ) remaining vertices. The union of all samples
across rounds (along with the remaining vertices in the last round) constructs an _O_ (1)-Coreset+ of
size _O_ ( _k_ polylog _n_ ). The mapping is defined by assigning each vertex _v_ _∈_ V to its nearest neighbor
among the sampled vertices from the round in which _v_ was removed. The main argument is that in
any round, there are sufficiently many good vertices in the second half (with larger distances from
_S_ ) that can account for the accidentally removed bad vertices. By a careful analysis, one can show
that across rounds, there exists a bijection between the bad vertices and these good vertices.


At a high level, our algorithms ALG-G and ALG-D emulate
this approach. However, there are several obstacles. In order to succeed, in each round we need to map vertices in V
to (approximate) nearest neighbors in _S_ before ordering them
by distance. Although in the R-model we can order the edges
E( _S,_ V) with _O_ (log _n_ ) dislocation using the PROBSORT primitive, such an ordering is not sufficient to find approximate
nearest neighbors. We need a more sophisticated approach.


**Overview of ALG-G (Appendix B).** The algorithm operates
in rounds. Let V _i_ _⊆_ V denote the set of active vertices in
round _i_ . In each round, the algorithm takes two random samples _Si_ [(1)] and _Si_ [(2)] of sizes Θ( _k_ log [2] _n_ ) and Θ( _k_ log [3] _n_ ) respectively. The first sample _Si_ [(1)] plays the same role as the
sample set in the generic algorithm described above. The
second sample set _Si_ [(2)] is used as follows. The algorithm
applies the PROBSORT primitive to approximately order the
edges X _i_ = E( _Si_ [(1)] _, Si_ [(2)] ), requiring _O_ ( _k_ [2] polylog _n_ ) quadruplet queries. From this ordering, the algorithm identifies for
each s _∈Si_ [(1)] two disjoint sets of Θ(log _n_ ) vertices: a _kernel_
_set_ KERNEL _i_ (s) _⊂Si_ [(2)] and a _guard_ _set_ GUARD _i_ (s) _⊂Si_ [(2)] .
These sets satisfy the following properties:

(i) For every s _∈_ _Si_ [(1)] and every v _∈_ KERNEL _i_ (s) _∪_
GUARD _i_ (s), POSs(v _,_ V _i_ ) _≤_ _k_ polylog _|_ V _i|_ _n_ [.]

(ii) For every s _∈Si_ [(1)], any w _∈_ KERNEL _i_ (s), and any g _∈_
GUARD _i_ (s), _d_ (s _,_ w) _< d_ (s _,_ g).


Figure 1: Let s _∈Si_ [(1)] be the
green vertex. The vertices in
KERNEL _i_ (s) are shown in red, and
those in GUARD _i_ (s) are shown in
blue. All remaining vertices in
V are depicted as black points.
The combined set KERNEL _i_ (s) _∪_
GUARD _i_ (s) consists of vertices
that are close to s in rank relative
to all vertices in V, with those in
KERNEL _i_ (s) being closer to s than
those in GUARD _i_ (s). All black
vertices inside the red circle will
be filtered out. No vertex outside
the blue circle will be filtered out.
Some vertices between the red and
blue circles may be filtered out.


The first property implies that for any s _∈Si_ [(1)] vertices in both
KERNEL _i_ (s) and GUARD _i_ (s) are very close to s in terms of rank. The second property ensures that
kernel vertices are strictly closer than their respective guard vertices. An example of a kernel and
guard set is shown in Figure 1. These sets play complementary roles: guard vertices are used to filter
all vertices from V _i_ that are too close to _Si_ [(1)], while kernel vertices are used to compute approximate
nearest neighbors in _Si_ [(1)] for the rest of the vertices.

_Filtering._ Next, the algorithm for each v _∈_ V _i_ _\_ ( _Si_ [(1)] _∪Si_ [(2)] ) and s _∈Si_ [(1)], compares against the
guard vertices to compute _proximity scores_ :


6


PCOUNTs(v) := **1** _{Q_ [˜] ( _{_ s _,_ v _}, {_ s _,_ g _}_ ) returns “ _d_ (s _,_ v) _≤_ _d_ (s _,_ g)” _},_ (1)


g _∈_ GUARD _i_ (s)


Based on these scores, the algorithm computes a subset V _i_ _[′]_ [=] _[{]_ [v] _[∈]_ [V] _[i]_ _[\]_ [(] _[S]_ _i_ [(1)] _∪Si_ [(2)] ) _|_
maxs _∈Si_ (1) PCOUNTs(v) _<_ _⌊m_ win _/_ 2 _⌋}_, where _m_ win is a suitable threshold of size Θ(log _n_ ). Computing V _i_ _[′]_ [requires] _[O]_ [(] _[nk]_ [ polylog] _[ n]_ [)] [quadruplet] [queries.] [Recall] [that] [the] [generic] [algorithm,] [in] [each]
round, orders vertices based on their nearest neighbor distance to the sample and removes a fraction
of the first half. In the presence of noise, we cannot find the nearest neighbors of all vertices in V _i_ . It
turns out that V _i_ _[′]_ [has sufficient structural properties for the algorithm to be able to find their nearest]
neighbors in _Si_ [(1)] . In the analysis, we show that filtering guarantees that no vertex in V _i_ _[′]_ [is closer to]
a sample vertex s _∈Si_ [(1)] than the kernel vertices KERNEL _i_ (s), while at the same time not discarding
too many additional vertices. Formally, filtering ensures that with high probability, _|_ V _i_ _[′][| ≥]_ [3] 5 _[|]_ [V] _[i][|]_ [ and]

_∀_ v _∈_ V _i_ _[′][,]_ _[∀]_ [s] _[∈S]_ _i_ [(1)] : _d_ (v _,_ s) _>_ _r_ s, where _r_ s := maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) denotes the _kernel radius_
of s. An example of filtering vertices is shown in Figure 1.


_Finding approximate nearest neighbors._ The key insight is the following. Consider any two sample
vertices s1 _,_ s2 _∈Si_ [(1)] and any vertices v1 _,_ v2 _∈_ V _\ Si_ [(2)], and suppose the following conditions hold:


(i) _d_ (s1 _,_ v1) _> r_ s1 and _d_ (s2 _,_ v2) _> r_ s2,
(ii) we know which kernel has the smaller radius, i.e., whether _r_ s1 _≤_ _r_ s2 or vice versa.


In the analysis, we show that when the above conditions hold, by making appropriate comparisons
with the smaller kernel, we can design a test procedure that answers whether _d_ (s1 _,_ v1) _<_ _d_ (s2 _,_ v2).
The test is correct with high probability when the two distances differ by more than a 2 factor, i.e.,
_d_ (s1 _,_ v1) _/d_ (s2 _,_ v2) _>_ 2 or _d_ (s2 _,_ v2) _/d_ (s1 _,_ v1) _>_ 2. Observe that this behavior is similar to a
_quadruplet oracle with adversarial noise_ with error _µ_ = 1 (see Section 2). We design a procedure
ALG-TESTER (see Section B.2), based on this test.
Our algorithm runs ADVSORT (see Section 2) with ALG-TESTER as the comparator to order the
edges in Y _i_ = E( _Si_ [(1)] _,_ V _i_ _[′]_ [)][.] [Whenever] [A][LG][-T][ESTER] [is] [asked] [to] [compare] [two] [edges] [from] [Y] _[i]_ [,] [the]
first condition is satisfied by the construction of V _i_ _[′]_ [,] [and] [A][LG][-T][ESTER] [uses] [the] [ordering] [of] [X] _[i]_ [to]
determine which kernel has a smaller radius. By the preceding discussion, on all such queries, it
behaves akin to an adversarial quadruplet oracle with noise _µ_ = 1. This immediately yields a 4approximate nearest neighbor in _Si_ [(1)] for every vertex in V _i_ _[′]_ [.] [Each] [call] [to] [A][LG][-T][ESTER] [triggers]
_O_ (log _n_ ) quadruplet queries, and ADVSORT calls ALG-TESTER a total of _O_ ( _nk_ polylog _n_ ) times.

The previous step computes an approximate nearest neighbor for each vertex in V _i_ _[′]_ [within] [the] [set]
_Si_ [(1)] . In the next step, the algorithm again applies ADVSORT, as before, to approximately order the
vertices in V _i_ _[′]_ [according] [to] [their] [estimated] [nearest-neighbor] [distances,] [and] [then] [identifies] [a] [prefix]
V _i_ _[′′]_ _[⊆]_ [V] _i_ _[′]_ [.] [Intuitively, this set][ V] _i_ _[′′]_ [corresponds to the set of vertices removed by the generic algorithm]
in a round. The algorithm also _maps_ each vertex in V _i_ _[′′]_ [to their neighbor in] _[ S]_ _i_ [(1)] found previously.
Next, the algorithm recurses on V _i_ +1 = V _i_ _\_ (V _i_ _[′′]_ _[∪S]_ _i_ [1] _[∪S]_ _i_ [2][)][.] [The] [process] [terminates] [after] _[r]_ [=]
_O_ (log _n_ ) rounds, at which point we output the set C = [�] _i_ _[r]_ =1 [(] _[S]_ _i_ [(1)] _∪Si_ [(2)] ) of _O_ ( _k_ polylog _n_ ) centers
that consists of the union of all samples. The mapping function _M_ is defined by the per-round
mappings of the sets V _i_ _[′′]_ [to the sets] _[ S]_ _i_ [(1)] . The complete algorithm along with the analysis is available
in Appendix B. We conclude with the next theorem.
**Theorem** **3.1.** _Let_ Σ = (V _, d_ ) _be_ _a_ _finite_ _metric_ _space_ _of_ _size_ _|_ V _|_ = _n,_ _which_ _is_ _accessible_ _under_
_the_ _R-model._ _There_ _exists_ _a_ _randomized_ _algorithm_ ALG-G _that,_ _given_ _parameters_ _k, p_ _∈_ Z+ _,_
_with_ _high_ _probability_ _returns_ _an_ _O_ (1) _-Coreset+_ _for_ ( _k, p_ ) _-clustering_ _of_ _size_ _O_ ( _k_ polylog _n_ ) _using_
_O_ ( _nk_ polylog _n_ ) _calls to the quadruplet oracle._


**Overview** **of** **ALG-D** **(Appendix** **C).** When Σ has bounded doubling dimension, we present an
algorithm ALG-D (Section C.1) that reduces the total number of quadruplet queries to _O_ (( _n_ +
_k_ [2] ) polylog _n_ ). The algorithm follows a similar recursive-sampling framework as ALG-G. Each
round begins by drawing two random samples _Si_ [(1)] and _Si_ [(2)], and computing the kernel and guard
sets for every s _∈Si_ [(1)] . However, in order to achieve the desired query complexity bounds, we
cannot afford to perform the full filtering and nearest-neighbor procedures used in ALG-G. Instead,
the algorithm proceeds as follows.


7


_Partitioning._ We first partition the vertices in _Si_ [(1)] into classes _Si_ [(1] _[,]_ [1)] _, . . ., Si_ [(1] _[,χ][i]_ [)] such that no two
vertices in the same class are _close_ . To do this, we construct a _conflict graph Gi_ whose vertex set
is _Si_ [(1)], and add an edge between any two vertices that are close. Closeness is determined using
proximity scores derived from the guard sets, as in Equation (1). In the analysis, we show that _Gi_
is _O_ (log _n_ )- _degenerate_, i.e., every subgraph of _Gi_ has a vertex of degree at most _O_ (log _n_ ). It is
known that a _ξ_ -degenerate graph can be properly colored with _ξ_ + 1 colors using a simple greedy
algorithm Lick & White (1970). We use such a coloring to obtain the classes _Si_ [(1] _[,]_ [1)] _, . . ., Si_ [(1] _[,χ][i]_ [)] .

_Nearest neighbors._ Our next goal is to compute approximate nearest neighbors for vertices in V _i \_
( _Si_ [(1)] _∪Si_ [(2)] ) with respect to each class _Si_ [(1] _[,j]_ [)] . In Raychaudhury et al. (2025) it was shown that
given two disjoint sets U _,_ W _⊆_ V and access to an adversarial quadruplet oracle with noise _µ_, that
can answer quadruplet queries of the form ( _{_ **e** 1 _}, {_ **e** 2 _}_ ), where **e** 1 _∈_ E(U _,_ U), **e** 2 _∈_ E(U _,_ W),
one can construct a data structure, such that given a vertex v _∈_ W, it returns a subset of size
_O_ (polylog _n_ ) containing at least one vertex u _∈_ U, such that _d_ (w _,_ u) _≤_ _O_ (1) _·_ _d_ (w _,_ U). Our
plan is to apply this result to each class _Si_ [(1] _[,j]_ [)] for each _j_ = 1 _, . . ., χi_, where we set U = _Si_ [(1] _[,j]_ [)]
and W = V _i_ _\_ ( _Si_ [(1)] _∪Si_ [(2)] ). To simulate the adversarial oracle, we again use ALG-TESTER as
in the general metric case. However, unlike in ALG-G, we have not pre-filtered close vertices in
V _i \_ ( _Si_ [(1)] _∪Si_ [(1)] ). If we apply ALG-TESTER to edges containing such vertices, it is not guaranteed
to behave like an adversarial quadruplet oracle. Hence, we adopt a _lazy filtering_ strategy: whenever
ALG-TESTER is invoked to compare a pair of edges, we first compute proximity scores to ensure
that the vertices are not too close. If a vertex is found to be too close, we discard it. This ensures
correctness while avoiding the heavy global filtering step of ALG-G. In the analysis, we show that
the overall number of quadruplet oracles required in this step is _O_ ( _n_ polylog _n_ ).


Post finding approximate nearest neighbors, we proceed similarly as in ALG-G. The full details are
in Appendix C. Overall, we get the next result.


**Theorem** **3.2.** _Let_ Σ = (V _, d_ ) _be_ _a_ _finite_ _metric_ _space_ _of_ _size_ _|_ V _|_ = _n_ _with_ _bounded_ _doubling_
_dimension,_ _which_ _is_ _accessible_ _under_ _the_ _R-model._ _There_ _exists_ _a_ _randomized_ _algorithm_ ALG-D
_that,_ _given_ _parameters_ _k, p_ _∈_ Z+ _,_ _with_ _high_ _probability_ _returns_ _an_ _O_ (1) _-Coreset+_ _for_ ( _k, p_ ) _-_
_clustering of size O_ ( _k_ polylog _n_ ) _using O_ (( _n_ + _k_ [2] ) polylog _n_ ) _calls to the quadruplet oracle._


**Overview of** **ALG-DI** **(Appendix D).** When the underlying metric has bounded doubling dimension, we show that, given a Coreset+ consisting of a set C _⊆_ V and a mapping _M_ : V _→_ C computed via ALG-G or ALG-D, we can compute an _ε_ -Coreset+, (C [+] _, M_ [+] ) using only _O_ ( _n_ polylog _n_ )
additional queries, for _k_ -median clustering.
Consider a vertex s _∈_ C, and let Us denote the set of vertices mapped to it, i.e., Us = _{_ u _∈_ V :
_M_ (u) = s _}_ . Let _α_ s = maxu _∈_ Us _d_ (u _,_ s). Suppose we order the vertices in Us by _d_ ( _·,_ s) and partition them into buckets: _B_ 0 has maximum distance _α_ s _/n_ [2], _B_ 1 covers ( _α_ s _/n_ [2] _,_ 2 _α_ s _/n_ [2] ], _B_ 2 covers
(2 _α_ s _/n_ [2] _,_ 4 _α_ s _/n_ [2] ] and so on, doubling the outer radius each time. It is easy to see that there are at
most _O_ (log _n_ ) buckets since _α_ s _≤_ _O_ (1) _·_ OPT [1] _k_ [.] [Since] [the] [doubling] [dimension] [is] [bounded,] [it] [is]
well known that if we had access to such a partition, we could improve the approximation quality by
choosing a constant number of vertices from each bucket (see, e.g., Har-Peled & Mazumdar (2004)).
Our high-level strategy is to follow a similar approach. Unfortunately, without access to distances,
we cannot compute such a partition directly. Our algorithm operates as follows.


For every s _∈_ C, it first obtains a _O_ (1)-sorted ordering of the edges E(s _,_ Us). This does not require any quadruplet queries and can simply be extracted from the post-ANN ordering computed by
ALG-D or ALG-G. We then perform a multistep sampling on this order: in the first round, we sample Θ(polylog _n_ ) vertices from Us, in the second round from the last half, then from the last quarter,
and so on, halving the suffix each time. Let Ws be all the samples obtained from Us. Repeating this
for all s _∈_ C accumulates _O_ ( _k_ polylog _n_ ) new samples. We set C [+] = C _∪_ ( [�] s _∈_ C [W][s][)][. The algorithm]

then orders the set [�] s _∈_ C [E][(][W][s] _[,]_ [ U][s][)][ using] [P][ROB][S][ORT] [and uses that information to construct a new]

mapping _M_ [+] ; we skip those details here. This step uses _O_ ( _n_ polylog _n_ ) additional queries.


The crux is the analysis. Although we cannot ensure that we hit every bucket for a s _∈_ C exactly, we
show that our sampling hits relevant distance scales with high probability. We carefully argue that
the revised mapping is indeed sufficient.


8


12.5


10.0


7.5


5.0


2.5


0.0


2.5


5.0


0 5 10 15 20


(a) Coreset Points


0 5 10 15 20


(b) Our Results


0 5 10 15 20


(c) Baseline


Method Cost


Optimal 2 _._ 9 _×_ 10 [4]

Ours 3 _._ 1 _×_ 10 [4]

Baseline 4 _×_ 10 [5]


(d) Clustering Cost


12.5


10.0


7.5


5.0


2.5


0.0


2.5


5.0


12.5


10.0


7.5


5.0


2.5


0.0


2.5


5.0


Figure 2: Comparison of the _k_ -means clustering results obtained by our method against the baseline
and the optimal clustering.


**Theorem 3.3.** _Let_ Σ = (V _, d_ ) _be a finite metric space of size |_ V _|_ = _n with bounded doubling di-_
_mension, which is accessible under the R-model. There exists a randomized algorithm_ ALG-DI _that,_
_given a parameter k_ _∈_ Z+ _, with high probability returns an ε-Coreset+ for k-median clustering of_
_size O_ ( _k_ polylog _n_ ) _using O_ (( _n_ + _k_ [2] ) polylog _n_ ) _calls to the quadruplet oracle, where ε_ _∈_ (0 _,_ 1) _is_
_an arbitrarily small constant._


4 EXPERIMENTS


In this section, we present preliminary experimental results based on a basic implementation of
our algorithm for _k_ -means clustering. We evaluate the quality of the results obtained from our
algorithm, comparing it against the clustering obtained by _k_ -means++ algorithm over the ground
truth data points, and the clustering obtained by a baseline that always trusts the answers of the
noisy quadruplet oracle. The complete descriptions of our algorithm and the baseline are provided
later. We refer to the cluster centers obtained from running _k_ -means++ algorithm over the ground
truth data as _true centers_ and the _k_ -means cost of the true centers as optimal cost [3] . We use both a
synthetic and two real datasets whose points are used as the ground truth data in the experiments. By
default, we simulate the quadruplet oracle with probabilistic noise and set the error rate to _φ_ = 0 _._ 15.
The oracle has access to the ground truth data and answers correctly with probability 1 _−_ _φ_ = 0 _._ 85.
Both the baseline algorithm and our algorithm can only access the data via this noisy oracle and
do not have direct access to the ground truth data. The main goal of our experiments is to show
the effectiveness of our Coreset+ construction and especially the mapping of the points to the
Coreset+ centers. All methods were implemented in Python, and the experiments were conducted
using the free version of Google Colab.


4.1 EXPERIMENTAL SETUP


**Synthetic** **dataset.** For our experiments, we generate a synthetic dataset consisting of twodimensional points arranged in _k_ = 5 clusters of approximately equal size. For each cluster, the
points are randomly generated following the Gaussian distribution with a standard deviation equal
to 1. After generating the data, all the points are randomly shuffled to avoid any relation between
ordering and the clusters. Using this procedure, we generate a dataset of 10 [4] synthetic points as
shown in Figure 2. We use this dataset as the ground truth data in the experiments.

**Real datasets.** We use two real-world datasets used to evaluate clustering algorithms ( Braverman
et al. (2025a); Huang et al. (2019)): the _Adult_ dataset Becker & Kohavi (1996) and the _Default_
_of_ _Credit_ _Card_ _Clients_ dataset Yeh (2009). For Adult dataset, we use eight numerical attributes,
resulting in a collection of roughly 50 _,_ 000 points in 8 dimensions. For Credit dataset, we select nine
numerical attributes, resulting in a collection of roughly 30 _,_ 000 points in 9 dimensions. The values
of all attributes are then normalized to be in the range [0 _,_ 1]. Similarly to Braverman et al. (2025a),
due to computational constraints, we sample 2 _,_ 000 data points from each dataset using Meyerson
sampling Meyerson (2001): We begin with an empty set _S_ and process the points sequentially. The
first point is added to _S_ . For each subsequent point, we add it to _S_ with probability proportional to
its distance from the current set _S_ . We employ Meyerson sampling instead of uniform sampling, as
its distance-based selection better preserves the geometric structure of the dataset and reduces the
risk of underrepresenting smaller clusters.


3 _k_ -means clustering is an NP-hard problem, and thus the _k_ -means++ algorithm does not always yield the
optimal solution. Nevertheless, we treat the clustering derived from the ground-truth data points as optimal.


9


|Ours Baseline|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|~~Baseline~~<br>Optimal|||||
||||||


0.05 0.10 0.15 0.20 0.25
Error


(b) Credit


|Col1|Col2|Col3|Col4|Ours Baseline|
|---|---|---|---|---|
|||||Baselin<br>~~Optimal~~|
||||||


4 5 6 7 8
k


(b) Credit


800


700


600


500


400


300


200


100


800


700


600


500


400


300


200


1000


900


800


700


600


500


400


300


200


800


700


600


500


400


300


200


100


|Col1|Col2|Col3|Col4|Ours Baseline|
|---|---|---|---|---|
|||||Baselin<br>~~Optimal~~|
||||||


4 5 6 7 8
k


(a) Adult


|Ours Baseline|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Baseline<br>Optimal|||||
||||||


0.05 0.10 0.15 0.20 0.25
Error


(a) Adult


Figure 3: Clustering cost varying _k_


Figure 4: Clustering cost varying _φ_


**Our** **approach.** We first run ALG-G from Theorem 3.1 to obtain an _O_ (1)-Coreset+. Then, as
described in Section 1, the obtained centers and the mapping are used to construct an _O_ (1)-coreset:
each Coreset+ point gets a weight equal to the number of points mapped to it. After constructing
the weighted coreset, we assume access to a distance oracle, as in the RM-model of Raychaudhury
et al. (2025). We then apply _k_ -means++ algorithm on the weighted coreset to obtain the final set of
_k_ centers, invoking the distance oracle when necessary.

**Baseline.** The baseline directly applies the Generic algorithm (Section 3) using the quadruplet
oracle. Although the oracle may return an incorrect answer with probability _φ_, the algorithm assumes every response is correct and proceeds accordingly. From the returned set of points and the
mapping, we construct a weighted set of centers (coreset) in the same manner as our algorithm: each
sampled center is assigned a weight equal to the number of data points mapped to it. Finally, we
run _k_ -means++ algorithm on the weighted coreset to obtain the final _k_ centers, invoking the distance
oracle whenever the exact distance between two coreset points is required.


4.2 EXPERIMENTAL RESULTS


**Results on synthetic dataset.** We observe that the coreset produced by our algorithm contains only
187 points. This corresponds to a 98% reduction in size, from the original 10 [4] input points down
to just 187 coreset points. Figure 2a shows the set of coreset points. Instead of using the original
large dataset, this much smaller coreset is used to obtain the final _k_ centers running the _k_ -means++
algorithm, assuming access to a distance oracle, as described above.


We next evaluate the clustering quality by comparing the results of our algorithm against both the
baseline and the ground-truth centers. As shown in Figure 2, the centers identified by our method
closely match the true centers (see also Figure 2b). By contrast, the baseline fails to recover some
clusters due to erroneous mapping, as illustrated in Figure 2c, leading to noticeably poorer clustering
performance. This limitation arises because the baseline unconditionally trusts the noisy oracle
during the Generic algorithm. Table 2d reports the corresponding clustering costs. The _k_ -means
cost is defined as the sum of squared distances from each point to its mapped center. Our method
achieves a cost within 7% of the optimal solution, while the baseline incurs a cost exceeding the
optimum by more than 1200%.


**Results** **on** **real** **datasets.** Next, we compare the clustering cost of our method with both the
baseline and the optimal cost on real datasets. In Figure 3, we fix the error rate at _φ_ = 0 _._ 15 and
vary the number of clusters _k_ _∈{_ 4 _,_ 5 _,_ 6 _,_ 7 _,_ 8 _}_ . In both datasets, our method achieves a clustering
cost that is very close to the optimum. In contrast, the baseline consistently yields a clustering
cost that is 2 _._ 5 to 4 times higher than that of our approach. As expected, the clustering cost of all
methods decreases as the number of clusters increases. Finally, in Figure 4, we fix the number of
clusters at _k_ = 6 and vary the oracle’s probabilistic noise level _φ_ _∈{_ 0 _._ 05 _,_ 0 _._ 1 _,_ 0 _._ 15 _,_ 0 _._ 2 _,_ 0 _._ 25 _}_ . As
before, in both datasets, the clustering cost achieved by our method remains close to the optimum.
Moreover, consistent with our theoretical guarantees, the clustering cost of our method is essentially
independent of _φ_ . In contrast, the clustering cost of the baseline increases substantially as _φ_ grows.


5 CONCLUSION
We proposed near-optimal algorithms for constructing coresets for ( _k, p_ )-clustering in the R-model.
Our results open several directions for future research. First, it is interesting to study whether the
techniques can be extended to other clustering objectives (such as hierarchical clustering or sumof-radii clustering) and to related graph problems (such as the Minimum Spanning Tree problem)
in the R-model. Second, we plan to generalize our framework to non-metric graphs, to alternative
oracle models (such as triplet oracles), and to different error models.


10


ACKNOWLEDGMENTS


This work has been partially supported by NSF grants IIS-2348919, IIS-2402823, and a grant by
Infosys.


REFERENCES


Jayadev Acharya, Moein Falahatgar, Ashkan Jafarpour, Alon Orlitsky, and Ananda Theertha Suresh.
Maximum selection and sorting with adversarial comparators. _The Journal of Machine Learning_
_Research_, 19(1):2427–2457, 2018.


Raghavendra Addanki, Sainyam Galhotra, and Barna Saha. How to design robust algorithms using
noisy comparison oracle. _Proceedings of the VLDB Endowment_, 14(10):1703–1716, 2021.


Pankaj K Agarwal, Aryan Esmailpour, Xiao Hu, Stavros Sintos, and Jun Yang. Computing a wellrepresentative summary of conjunctive query results. _Proceedings of the ACM on Management of_
_Data_, 2(5):1–27, 2024.


Nir Ailon, Anup Bhattacharya, Ragesh Jaiswal, and Amit Kumar. Approximate clustering with
same-cluster queries. In _9th_ _Innovations_ _in_ _Theoretical_ _Computer_ _Science_ _Conference_ _(ITCS_
_2018)_, volume 94, pp. 40. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2018.


Hassan Ashtiani, Shrinu Kushagra, and Shai Ben-David. Clustering with same-cluster queries. In
_Advances in neural information processing systems_, pp. 3216–3224, 2016.


MohammadHossein Bateni, Prathamesh Dharangutte, Rajesh Jayaram, and Chen Wang. Metric
clustering and mst with strong and weak distance oracles. In _The Thirty Seventh Annual Confer-_
_ence on Learning Theory_, pp. 498–550. PMLR, 2024.


Barry Becker and Ron Kohavi. Adult. UCI Machine Learning Repository, 1996.


Lorenzo Beretta, Franco Maria Nardini, Roberto Trani, and Rossano Venturini. An optimal algorithm for finding champions in tournament graphs. _IEEE Transactions on Knowledge and Data_
_Engineering_, 35(10):10197–10209, 2023.


Enrico Bianchi and Paolo Penna. Optimal clustering in stable instances using combinations of exact
and noisy ordinal queries. _Algorithms_, 14(2):55, 2021.


Mark Braverman and Elchanan Mossel. Noisy sorting without resampling. In _Proceedings_ _of_ _the_
_nineteenth_ _annual_ _ACM-SIAM_ _symposium_ _on_ _Discrete_ _algorithms_, pp. 268–276. Society for Industrial and Applied Mathematics, 2008.


Mark Braverman, Jieming Mao, and S Matthew Weinberg. Parallel algorithms for select and partition with noisy comparisons. In _Proceedings_ _of_ _the_ _forty-eighth_ _annual_ _ACM_ _symposium_ _on_
_Theory of Computing_, pp. 851–862, 2016.


Vladimir Braverman, Shaofeng H.-C. Jiang, Robert Krauthgamer, and Xuan Wu. Coresets for clustering in excluded-minor graphs and beyond. In _Proceedings of the Thirty-Second Annual ACM-_
_SIAM_ _Symposium_ _on_ _Discrete_ _Algorithms_, SODA ’21, pp. 2679–2696, USA, 2021. Society for
Industrial and Applied Mathematics. ISBN 9781611976465.


Vladimir Braverman, Vincent Cohen-Addad, H-C Shaofeng Jiang, Robert Krauthgamer, Chris
Schwiegelshohn, Mads Bech Toftrup, and Xuan Wu. The power of uniform sampling for coresets. In _2022_ _IEEE_ _63rd_ _Annual_ _Symposium_ _on_ _Foundations_ _of_ _Computer_ _Science_ _(FOCS)_, pp.
462–473. IEEE, 2022.


Vladimir Braverman, Prathamesh Dharangutte, Vihan Shah, and Chen Wang. Learning-augmented
maximum independent set. _Approximation,_ _Randomization,_ _and_ _Combinatorial_ _Optimization._
_Algorithms and Techniques_, 2024.


Vladimir Braverman, Prathamesh Dharangutte, Shaofeng H-C Jiang, Hoai-An Nguyen, Chen Wang,
Yubo Zhang, and Samson Zhou. Relative error fair clustering in the weak-strong oracle model.
In _Forty-second International Conference on Machine Learning_, 2025a.


11


Vladimir Braverman, Jon C Ergun, Chen Wang, and Samson Zhou. Learning-augmented hierarchical clustering. In _Forty-second International Conference on Machine Learning_, 2025b.


Moses Charikar, Sudipto Guha, Eva Tardos, and David B. Shmoys. A constant-factor approximation [´]
algorithm for the _k_ -median problem. In _Proceedings_ _of_ _the_ _31st_ _Annual_ _ACM_ _Symposium_ _on_
_Theory of Computing (STOC)_, pp. 1–10. ACM, 1999.


Vaggos Chatziafratis, Rad Niazadeh, and Moses Charikar. Hierarchical clustering with structural
constraints. In _International conference on machine learning_, pp. 774–783. PMLR, 2018.


Jiaxiang Chen, Qingyuan Yang, Ruomin Huang, and Hu Ding. Coresets for relational data and the
applications. _Advances in Neural Information Processing Systems_, 35:434–448, 2022.


Ke Chen. On coresets for k-median and k-means clustering in metric and euclidean spaces and their
applications. volume 39, pp. 923–947, 2009. doi: 10.1137/070699007.


I Chien, Chao Pan, and Olgica Milenkovic. Query k-means clustering and the double dixie cup
problem. In _Advances in Neural Information Processing Systems_, pp. 6649–6658, 2018.


Tuhinangshu Choudhury, Dhruti Shah, and Nikhil Karamchandani. Top-m clustering with a noisy
oracle. In _2019 National Conference on Communications (NCC)_, pp. 1–6. IEEE, 2019.


Eleonora Ciceri, Piero Fraternali, Davide Martinenghi, and Marco Tagliasacchi. Crowdsourcing
for top-k query processing over uncertain data. _IEEE_ _Transactions_ _on_ _Knowledge_ _and_ _Data_
_Engineering_, 28(1):41–53, 2015.


Ryan Curtin, Benjamin Moseley, Hung Ngo, XuanLong Nguyen, Dan Olteanu, and Maximilian
Schleich. Rk-means: Fast clustering for relational data. In _International Conference on Artificial_
_Intelligence and Statistics_, pp. 2742–2752. PMLR, 2020.


Susan Davidson, Sanjeev Khanna, Tova Milo, and Sudeepa Roy. Top-k and clustering with noisy
comparisons. _ACM Transactions on Database Systems (TODS)_, 39(4):1–39, 2014.


Yinhao Dong, Pan Peng, and Ali Vakilian. Learning-augmented streaming algorithms for approximating max-cut. In _16th Innovations in Theoretical Computer Science Conference (ITCS 2025)_,
pp. 44–1, 2025.


Eyal Dushkin and Tova Milo. Top-k sorting under partial order information. In _Proceedings of the_
_2018 International Conference on Management of Data_, pp. 1007–1019, 2018.


Ehsan Emamjomeh-Zadeh and David Kempe. Adaptive hierarchical clustering using ordinal queries.
In _Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms_, pp.
415–429. SIAM, 2018.


Jon C. Ergun, Zhili Feng, Sandeep Silwal, David P. Woodruff, and Samson Zhou. Learningaugmented $k$-means clustering. In _The_ _Tenth_ _International_ _Conference_ _on_ _Learning_ _Repre-_
_sentations, ICLR_, 2022.


Aryan Esmailpour and Stavros Sintos. Improved approximation algorithms for relational clustering.
_Proceedings of the ACM on Management of Data_, 2(5):1–27, 2024.


Chunkai Fu, Brandon G Nguyen, Jung Hoon Seo, Ryan S Zesch, and Samson Zhou. Learningaugmented search data structures. In _The Thirteenth International Conference on Learning Rep-_
_resentations_, 2025.


Sainyam Galhotra, Sandhya Saisubramanian, and Shlomo Zilberstein. Learning to generate fair
clusters from demonstrations. In _Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics,_
_and Society_, pp. 491–501, 2021.


Sainyam Galhotra, Rahul Raychaudhury, and Stavros Sintos. k-clustering with comparison and
distance oracles. _Proceedings of the ACM on Management of Data_, 2(5):1–26, 2024.


Barbara Geissmann, Stefano Leucci, Chih-Hung Liu, and Paolo Penna. Sorting with recurrent comparison errors. In _28th International Symposium on Algorithms and Computation (ISAAC 2017)_,
2017.


12


Barbara Geissmann, Stefano Leucci, Chih-Hung Liu, and Paolo Penna. Optimal dislocation with
persistent errors in subquadratic time. _Theory of Computing Systems_, 64(3):508–521, 2020.


Barbara Geissmann, Stefano Leucci, Chih-Hung Liu, and Paolo Penna. An optimal sorting algorithm
for persistent random comparison faults. _arXiv preprint arXiv:2508.19785_, 2025.


Debarghya Ghoshdastidar, Micha¨el Perrot, and Ulrike von Luxburg. Foundations of comparisonbased hierarchical clustering. In _Advances in Neural Information Processing Systems_, pp. 7454–
7464, 2019.


Kasper Green Larsen, Michael Mitzenmacher, and Charalampos Tsourakakis. Clustering with a
faulty oracle. In _Proceedings of The Web Conference 2020_, WWW ’20, pp. 2831–2834, 2020.


Elena Grigorescu, Young-San Lin, Sandeep Silwal, Maoyuan Song, and Samson Zhou. Learningaugmented algorithms for online linear and semidefinite programming. _Advances_ _in_ _Neural_ _In-_
_formation Processing Systems_, 35:38643–38654, 2022.


Stephen Guo, Aditya Parameswaran, and Hector Garcia-Molina. So who won? dynamic max discovery with the crowd. In _Proceedings of the 2012 ACM SIGMOD International Conference on_
_Management of Data_, pp. 385–396, 2012.


Sariel Har-Peled and Soham Mazumdar. On coresets for k-means and k-median clustering. In
_Proceedings_ _of_ _the_ _thirty-sixth_ _annual_ _ACM_ _symposium_ _on_ _Theory_ _of_ _computing_, pp. 291–300,
2004.


Max Hopkins, Daniel Kane, Shachar Lovett, and Gaurav Mahajan. Noise-tolerant, reliable active classification with comparison queries. In _Conference on Learning Theory_, pp. 1957–2006.
PMLR, 2020.


Chen-Yu Hsu, Piotr Indyk, Dina Katabi, and Ali Vakilian. Learning-based frequency estimation
algorithms. In _International Conference on Learning Representations_, 2019.


Lingxiao Huang, Shaofeng Jiang, and Nisheeth Vishnoi. Coresets for clustering with fairness constraints. _Advances in neural information processing systems_, 32, 2019.


Wasim Huleihel, Arya Mazumdar, Muriel M´edard, and Soumyabrata Pal. Same-cluster querying for
overlapping clusters. In _Advances in Neural Information Processing Systems_, pp. 10485–10495,
2019.


Christina Ilvento. Metric learning for individual fairness. In _1st_ _Symposium_ _on_ _Foundations_ _of_
_Responsible Computing (FORC 2020)_, 2020.


Piotr Indyk, Ali Vakilian, and Yang Yuan. Learning-based low-rank approximations. _Advances in_
_Neural Information Processing Systems_, 32, 2019.


Ehsan Kazemi, Lin Chen, Sanjoy Dasgupta, and Amin Karbasi. Comparison based learning from
weak oracles. In _International Conference on Artificial Intelligence and Statistics_, pp. 1849–1858.
PMLR, 2018.


Taewan Kim and Joydeep Ghosh. Relaxed oracles for semi-supervised clustering. _arXiv_ _preprint_
_arXiv:1711.07433_, 2017a.


Taewan Kim and Joydeep Ghosh. Semi-supervised active clustering with weak oracles. _arXiv_
_preprint arXiv:1709.03202_, 2017b.


Rolf Klein, Rainer Penninger, Christian Sohler, and David P Woodruff. Tolerant algorithms. In
_European Symposium on Algorithms_, pp. 736–747. Springer, 2011.


Ngai Meng Kou, Yan Li, Hao Wang, Leong Hou U, and Zhiguo Gong. Crowdsourced top-k queries
by confidence-aware pairwise judgments. In _Proceedings of the 2017 ACM International Confer-_
_ence on Management of Data_, pp. 1415–1430, 2017.


Don R Lick and Arthur T White. k-degenerate graphs. _Canadian_ _Journal_ _of_ _Mathematics_, 22(5):
1082–1096, 1970.


13


Tie-Yan Liu et al. Learning to rank for information retrieval. _Foundations and Trends® in Informa-_
_tion Retrieval_, 3(3):225–331, 2009.


Stuart Lloyd. Least squares quantization in pcm. _IEEE Transactions on Information Theory_, 28(2):
129–137, 1982.


Arya Mazumdar and Barna Saha. Clustering with noisy queries. In _Advances in Neural Information_
_Processing Systems_, pp. 5788–5799, 2017a.


Arya Mazumdar and Barna Saha. Query complexity of clustering with side information. In _Advances_
_in Neural Information Processing Systems_, pp. 4682–4693, 2017b.


Ramgopal R. Mettu and C. Greg Plaxton. Optimal time bounds for approximate clustering. In Adnan
Darwiche and Nir Friedman (eds.), _UAI ’02, Proceedings of the 18th Conference in Uncertainty_
_in Artificial Intelligence_, pp. 344–351. Morgan Kaufmann, 2002.


Adam Meyerson. Online facility location. In _Proceedings 42nd IEEE Symposium on Foundations_
_of Computer Science_, pp. 426–431. IEEE, 2001.


Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions. _Communications of_
_the ACM_, 65(7):33–35, 2022.


Benjamin Moseley, Kirk Pruhs, Alireza Samadian, and Yuyan Wang. Relational algorithms for kmeans clustering. In _48th International Colloquium on Automata, Languages, and Programming_
_(ICALP 2021)_, pp. 97–1, 2021.


Nikolaos Nakis, Niels Raunkjær Holm, Andreas Lyhne Fiehn, and Morten Mørup. How low can
you go? searching for the intrinsic dimensionality of complex networks using metric node embeddings. In _International Conference on Learning Representations (ICLR)_, 2025.


Vassilis Polychronopoulos, Luca De Alfaro, James Davis, Hector Garcia-Molina, and Neoklis Polyzotis. Human-powered top-k lists. In _WebDB_, pp. 25–30, 2013.


Rahul Raychaudhury, Wen-Zhi Li, Syamantak Das, Sainyam Galhotra, and Stavros Sintos. Metric clustering and graph optimization problems using weak comparison oracles. _Proceedings of_
_Machine Learning Research vol_, 291:1–54, 2025.


Sam T Roweis and Lawrence K Saul. Nonlinear dimensionality reduction by locally linear embedding. _science_, 290(5500):2323–2326, 2000.


Sandeep Silwal, Sara Ahmadian, Andrew Nystrom, Andrew McCallum, Deepak Ramachandran,
and Mehran Kazemi. Kwikbucks: Correlation clustering with cheap-weak and expensive-strong
signals. In _Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Pro-_
_cessing (SustaiNLP)_, pp. 1–31, 2023.


Vaishali Surianarayanan, Neeraj Kumar, and Stavros Sintos. Clustering with set outliers and applications in relational clustering. _Proceedings of the ACM on Management of Data_, 3(5):1–27,
2025.


Omer Tamuz, Ce Liu, Serge Belongie, Ohad Shamir, and Adam Tauman Kalai. Adaptively learning
the crowd kernel. In _Proceedings of the 28th International Conference on International Confer-_
_ence on Machine Learning_, pp. 673–680, 2011.


Joshua B Tenenbaum, Vin de Silva, and John C Langford. A global geometric framework for
nonlinear dimensionality reduction. _science_, 290(5500):2319–2323, 2000.


Antti Ukkonen. Crowdsourced correlation clustering with relative distance comparisons. In _2017_
_IEEE International Conference on Data Mining (ICDM)_, pp. 1117–1122. IEEE, 2017.


Petros Venetis, Hector Garcia-Molina, Kerui Huang, and Neoklis Polyzotis. Max algorithms in
crowdsourcing environments. In _Proceedings of the 21st international conference on World Wide_
_Web_, pp. 989–998, 2012.


14


Victor Verdugo. Skyline computation with noisy comparisons. In _Combinatorial Algorithms:_ _31st_
_International_ _Workshop,_ _IWOCA_ _2020,_ _Bordeaux,_ _France,_ _June_ _8–10,_ _2020,_ _Proceedings_, pp.
289. Springer.


Haike Xu, Sandeep Silwal, and Piotr Indyk. A bi-metric framework for fast similarity search. _arXiv_
_preprint arXiv:2406.02891_, 2024.


I-Cheng Yeh. Default of credit card clients. UCI Machine Learning Repository, 2009.


15


A LOWER BOUND FOR _k_ -CLUSTERING IN THE R-MODEL.


**Theorem A.1.** _Every o_ ( _n_ ) _-approximation algorithm for the k-median/means clustering problem in_
_the_ _R-model_ _must_ _contain_ _at_ _least_ 2 _k_ _−_ 1 _centers,_ _where_ _n_ _is_ _the_ _number_ _of_ _vertices_ _in_ _the_ _input_
_metric space and_ 3 _≤_ _k_ = _O_ (1) _._


_Proof._ We show the proof for the _k_ -median clustering, however all results can be extended to _k_ means clustering. We show a stronger result, assuming that the probabilistic noise is 0, i.e., perfect
oracle.


We construct a finite metric space Σ = (V _, d_ ) of _n_ vertices as follows. For simplicity, we assume
that _n_ _[′]_ = _nk_ [is] [an] [integer.] [A] [set] [of] _[k]_ _[−]_ [1] [vertices] _[U]_ [=] _[u]_ [1] _[, . . ., u][k][−]_ [1] [are] [contained] [in] [V] [with]
_d_ ( _ui, uj_ ) = _ζ_, for every _i_ = _j_ _∈_ [ _k −_ 1], for a parameter _ζ_ _>_ 0 that we specify later. Furthermore,
there exists a set _Y_ of _n_ _[′]_ vertices _y_ 1 [(1)] _[, . . ., y]_ _n_ [(1)] _[′]_ [such that] _[ d]_ [(] _[y]_ _i_ [(1)] _, yj_ [(1)][)] [=] [0][ for every] _[ i]_ [=] _[j]_ _[∈]_ [[] _[n][′]_ []][.]

Moreover, there are _k −_ 1 groups of vertices, _X_ 2 = _{x_ [(2)] 1 _[, . . ., x]_ _n_ [(2)] _[′]_ _−_ 1 _[}]_ [,] _[X]_ [3] [=] _[{][x]_ [(3)] 1 _[, . . ., x]_ _n_ [(3)] _[′]_ _−_ 1 _[}]_ [,]

_. . ._, _Xk_ = _{x_ [(] 1 _[k]_ [)] _[, . . ., x]_ _n_ [(] _[k][′]_ [)] _−_ 1 _[}]_ [ such that each group contains] _[ n][′][ −]_ [1][ vertices, while] _[ d]_ [(] _[x]_ _i_ [(] _[h]_ [)] _, x_ [(] _j_ _[h]_ [)] ) = 0
for every _h ∈{_ 2 _, . . ., k}_ and every _i ̸_ = _j_ _∈_ [ _n_ _[′]_ _−_ 1]. Let _X_ = [�] _i_ =2 _,...,k_ _[X][i]_ [.] [For every] _[ y][i]_ _[∈]_ _[Y]_ [(for]

_i_ _∈_ [ _n_ _[′]_ ]) and _x_ _∈_ _X_, _d_ ( _x, yi_ ) = 1. For every _x_ 1 _∈_ _Xi_ and _x_ 2 _∈_ _Xj_, where _i_ = _j_ _∈{_ 2 _, . . ., k}_,
_d_ ( _x_ 1 _, x_ 2) = 1. Finally, for every vertex _t ∈_ _X_ _∪_ _Y_ and _u ∈_ _U_, _d_ ( _t, u_ ) = _ζ_ . It is easy to verify that
the function _d_ ( _·_ ) satisfies the triangle inequality, so Σ is a metric space.


We note that under the R-model (with probabilistic noise 0) an algorithm cannot distinguish between
an instance of the metric space where _ζ_ = _n_ [3] and _ζ_ = 2. Indeed, for both choices, _ζ_ _>_ 1, so the
ordering of all pairwise distances of the vertices in V remain the same for _ζ_ = _n_ [3] and _ζ_ = 2.


Next, we show that no set of centers of size at most 2 _k −_ 2 returns an _o_ ( _n_ )-approximation solution
for both undistinguishable instances of _k_ -median clustering on V with _ζ_ = 2 and _ζ_ = _n_ [3] .

First, consider the case where _ζ_ = _n_ [3] . The optimum set for _k_ -median clustering for V is _P_ 1 _[∗]_ [=]
_U_ _∪{y}_, where _y_ _∈_ _Y_, with OPT [1] = COST [1] _P_ 1 _[∗]_ [(][V][) =] - _nk_ _[−]_ [1] - ( _k −_ 1).


If _ζ_ = 2, then the optimum set for _k_ -median clustering on V is _P_ 2 _[∗]_ [=] _[ {][y][} ∪]_ ��


    _h_ =2 _,...k_ _[x][h]_, where


_y_ _∈_ _Y_, and _xh_ _∈_ _Xh_ for every _h ∈{_ 2 _, . . ., k}_ with OPT [1] = COST [1] _P_ 2 _[∗]_ [(][V][) = 2(] _[k][ −]_ [1)][.]


For _ζ_ = 2, any _o_ ( _n_ )-approximation solution _R_ _⊆_ V of size at most 2 _k_ _−_ 2 must include one
(arbitrary) vertex from every group _Xi_ (for every _i_ = _{_ 2 _, . . ., k}_ ) and one (arbitrary) vertex from
_Y_ . If this is not the case, then the _n_ _k_ -median cost will be at least _[n]_ _k_ _[−]_ [1][ leading to an approximation]

_k_ _[−]_ [1]

ratio 2( _k−_ 1) [=] [Ω(] _[n]_ [)][.] [Without loss of generality assume that a set] _[ R]_ [contains one arbitrary vertex]

from _Y_, one arbitrary vertex for every _Xi_, for every _i_ = _{_ 2 _, . . ., k}_, and _k_ _−_ 2 arbitrary vertices
from _U_ . Notice that COST [1] _R_ [(][V][)] [=] [2][.] [Recall that in the R-model no algorithm can distinguish the]
two instances with _ζ_ = 2 and _ζ_ = _n_ [3] . Hence, if _R_ is selected as a solution of size 2 _k −_ 2 for the
_k_ -median instance when _ζ_ = 2, then _R_ will also be selected as a solution for the _k_ -median instance
when _ζ_ = _n_ [3] . However, when _ζ_ = _n_ [3], then OPT [1] = - _nk_ _[−]_ [1] - ( _k −_ 1) and COST [1] _R_ [(][V][) =] _[ n]_ [3][ so the]
approximation ratio is Ω( _n_ [2] ). The result follows.


B ALGORITHM FOR GENERAL METRICS


We are given a set of vertices V from a metric space Σ = (V _, d_ ) with _|_ V _|_ = _n_, and access to a noisy
quadruplet oracle _Q_ [˜] over Σ with known error rate _φ_ _<_ 2 [1] [.] [Given cluster size] _[ k]_ [and parameter] _[ p]_ _[∈]_

Z _≥_ 1 _∪{∞}_, the algorithm outputs a set C _⊆_ V of size _O_ ( _k_ polylog _n_ ) and a mapping _M_ : V _→_ C.
For clarity of exposition, we restrict to the _k_ -median case ( _p_ = 1) and assume _φ_ is bounded by a
fixed constant below 1 _/_ 4. The rest of this section is organized as follows: Section B.1 presents the
main algorithm, while Section B.2 describes a key subroutine.


16


B.1 ALG-G


**Overview.** The algorithm proceeds in rounds. Initially, V1 = V. In round _i_, the algorithm processes
V _i_ and removes a subset of vertices to obtain a smaller set V _i_ +1 _⊂_ V _i_ . The algorithm terminates
when _|_ V _i|_ = _O_ ( _k_ log [3] _n_ ) or if the number of rounds exceeds _r_ = _O_ (log _n_ ). Throughout, we use
_D_ = _O_ (log _n_ ) to denote the maximum dislocation bound of PROBSORT on any set of edges.


**Round** _i_ **.** The algorithm proceeds as follows.


1. _Sampling._ Sample uniformly at random (with replacement) a set of vertices _Si_ [(1)] _,_ _Si_ [(2)] _⊆_ V _i_
of sizes _m_ S1 = _c_ S1 _k_ log [2] _n_ and _m_ S2 = _c_ S2 _k_ log [3] _n_ respectively, where _c_ S1 _, c_ S2 are suitable
constants. Let _Si_ := _Si_ [(1)] _∪Si_ [(2)] .

2. _Ordering edges._ Let X _i_ = E( _Si_ [(1)] _, Si_ [(2)] ) and compute _π_ X _i_ = PROBSORT(X _i_ ).

3. _Kernel_ _and_ _guard_ _sets._ Let _m_ win = 2 max _{c_ win log _n, D}_, where _c_ win is a sufficiently large
constant. For each s _∈Si_ [(1)], let X _i,_ s = E(s _, Si_ [(2)] ) and _π_ X _i,_ s the ordering of X _i,_ s induced by _π_ X _i_ .
For every s _∈Si_ [(1)], compute

KERNEL _i_ (s) = _{_ w _∈Si_ [(2)] : RANK _π_ X _i,_ s [ _{_ s _,_ w _}_ ] _≤_ _m_ win _},_


GUARD _i_ (s) = _{_ g _∈Si_ [(2)] : _m_ win + 2 _D_ _<_ RANK _π_ X _i,_ s [ _{_ s _,_ g _}_ ] _≤_ 2 _m_ win + 2 _D}._


4. _Filtering._ For any s _∈Si_ [(1)] and v _∈_ V _i_, define the _proximity score_


     PCOUNTs(v) := **1** _{Q_ [˜] ( _{_ s _,_ v _}, {_ s _,_ g _}_ ) returns “ _d_ (s _,_ v) _≤_ _d_ (s _,_ g)” _},_


g _∈_ GUARD _i_ (s)


where **1** _{_ condition _}_ is the indicator function that returns 1 if the condition holds, and 0 otherwise. Compute V _i_ _[′]_ [=] _[ {]_ [v] _[ ∈]_ [V] _[i][ \ S][i]_ _[|]_ [ max] s _∈Si_ [(1)] PCOUNTs(v) _< ⌊m_ win _/_ 2 _⌋}._

5. _Approximate_ _nearest_ _neighbors._ Let Y _i_ = E( _Si_ [(1)] _,_ V _i_ _[′]_ [)][.] [Compute] _[π]_ [Y] _i_ [=] [A][DV][S][ORT][(][Y] _[i]_ [)] [using]
ALG-TESTER (Subsection B.2) as the comparator. For each v _∈_ V _i_ _[′]_ [,] [let] [f] [v] [be] [its] _[first]_ [incident]
edge in _π_ Y _i_ . Set N _i_ = _{_ f v : v _∈_ V _i_ _[′][}][.]_

6. _Safe-set and mapping._ Let _π_ N _i_ be the ordering of N _i_ induced by _π_ Y _i_ . Define V _i_ _[′′]_ [=] _[{]_ [v] _[∈]_ [V] _i_ _[′]_ [:]

RANK _π_ N _i_ ( f v) _≤|_ V _i|/_ 4 _}_ . For every v _∈_ V _i_ _[′′]_ [,] [define] _[M][i]_ [(][v][)] [as] [the] [endpoint] [of] [f] [v] [in] _[S]_ _i_ [(1)] . For
every v _∈Si_, define _Mi_ (v) := v.

7. _Recurse._ Set V _i_ +1 = V _i \_ (V _i_ _[′′]_ _[∪S][i]_ [)][ and proceed to next round.]


**Final output.** Let _i_ _[⋆]_ denote the last round. The final centers are C := V _i⋆_ _∪_ [�] _[i]_ _i_ _[⋆]_ =1 _[−]_ [1] _[S][i]_ [.] [Let] _[ M][i][⋆]_
denote the function that maps every vertex in V _i⋆_ to itself. Finally, define the global mapping _M_ as
the union of the per round mappings _M_ := [�] _i_ _[i]_ =1 _[⋆]_ _[M][i]_ [.] [This is a function from][ V] _[�→]_ [C][ since each]

_Mi_ is defined on a disjoint round-specific set.


**Weighted** **coreset.** We can also obtain a weighted coreset by assigning to each u _∈_ C a weight
equal to the number of original vertices mapped to it, namely _w_ (u) := �� _{_ v _∈_ V _|_ _M_ (v) = u _}_ ��.
The pair (C _, w_ ) defines the coreset.


B.2 ALG-TESTER


Whenever the tester is invoked in round _i_, it is given two edges **e** 1 = _{_ s1 _,_ v1 _}_ and **e** 2 =
_{_ s2 _,_ v2 _}_, with s1 _,_ s2 _∈Si_ [(1)] and v1 _,_ v2 _∈_ V _i_ _\_ _Si_ [(2)] . It is also given access to the kernel sets KERNEL _i_ (s1) _,_ KERNEL _i_ (s2) and the global ordering _π_ X _i_ . From these, it forms Z =
E(s1 _,_ KERNEL _i_ (s1)) _∪_ E(s2 _,_ KERNEL _i_ (s2)), and computes _π_ Z, the ordering of Z induced by _π_ X _i_ .


1. **Case** s1 = s2 **.**


17


(a) _Kernel_ _Selection_ . Let **e** _[⋆]_ be the last edge in _π_ Z. Determine whether it belongs
to E(s1 _,_ KERNEL _i_ (s1)) or E(s2 _,_ KERNEL _i_ (s2)). Without loss of generality, assume
**e** _[⋆]_ _∈_ E(s1 _,_ KERNEL _i_ (s1)). Remove every vertex w _∈_ KERNEL _i_ (s2) such that
RANK _π_ Z( _{_ s2 _,_ w _}_ ) _∈_ [2 _m_ win _−D,_ 2 _m_ win], and call the remainder KERNEL _[′]_ (s2).
(b) _Majority Test_ . Compute


     TCOUNT = **1** _{Q_ [˜] ( _{_ s1 _,_ v1 _}, {_ w _,_ v2 _}_ ) says “ _d_ (s1 _,_ v1) _> d_ (w _,_ v2)” _}._

w _∈_ KERNEL _[′]_ (s2)


Output “ _d_ (s1 _,_ v1) _>_ _d_ (s2 _,_ v2)” if TCOUNT _>_ _⌊_ [1] 2 _[|]_ [KERNEL] _[′]_ [(][s][2][)] _[|⌋]_ [,] [and] [the] [opposite] [other-]

wise.

2. **Case** s1 = s2 **.** Set KERNEL _[′]_ (s2) := KERNEL _i_ (s2) and run the same _majority test_ as in Case (a).


B.3 ANALYSIS


We first establish that no quadruplet query is ever repeated between rounds and within certain steps
of a round. We refer to this as the _isolation_ _property_ . Next, in Section B.3.1 we prove certain
properties that hold in each round. Finally, in Section B.3.2, we combine the guarantees to establish
global correctness.

**Lemma B.1** (Isolation) **.** _The algorithm satisfies the following properties:_


_1._ _No two rounds rely on the outcome of the same quadruplet query._


_2._ _Within any round, there is no overlap between the quadruplet queries used by_ PROBSORT(X _i_ ) _,_
_with_ _those_ _required_ _to_ _compute_ _any_ _proximity_ _score_ PCOUNTs( _·_ ) _,_ _or_ _those_ _required_ _by_ _any_
_invocation of_ ALG-TESTER _._


_Proof._ Since V _i_ +1 = V _i_ _[′′]_ _[\]_ [ (] _[S]_ _i_ [(1)] _∪Si_ [(2)] ), it follows that V _i_ _∩_ ( _Si_ [(1)] _−_ 1 _[∪S]_ _i_ [(2)] _−_ 1 [)] [=] _[∅]_ [.] [Therefore,] [the]

sampled sets _Si_ [(1)] _, Si_ [(2)] _⊆_ V _i_ are disjoint from previous samples. This implies that the corresponding
X _i_ = E( _Si_ [(1)] _, Si_ [(2)] ), KERNEL _i_ ( _·_ ) _⊆Si_ [(2)], and GUARD _i_ ( _·_ ) _⊆Si_ [(2)] are disjoint from previous rounds.
Hence, no quadruplet query is ever reused across rounds.


Within a round _i_, for any s _∈Si_ [(1)], the computation of PCOUNTs(v) is always performed with
v _∈_ V _i_ _\Si_ [(2)] . Thus, every query triggered by it involves one edge from E( _Si_ [(1)] _,_ V _i_ _\Si_ [(2)] ). Similarly,
every query issued by ALG-TESTER includes at least one edge from E( _Si_ [(1)] _,_ V _i_ _\_ _Si_ [(2)] ). Since
X _i ∩_ E( _Si_ [(1)] _,_ V _i \ Si_ [(2)] ) = _∅_, quadruplet queries used for computing proximity scores or those used
by ALG-TESTER cannot overlap with any query used by PROBSORT(X _i_ ), which always compares
two edges from X _i_ .


B.3.1 PER-ROUND GUARANTEES


We first show that in every round the ordering _π_ X _i_ has bounded dislocation, the kernel and guard
sets are confined to the top-ranked vertices, and that kernels are always closer than guards

**Lemma** **B.2.** _In_ _any_ _round_ _i,_ _with_ _probability_ _at_ _least_ 1 _−_ _n_ _[−]_ [Ω(1)] _the_ _following_ _properties_ _hold_
_simultaneously:_


_(i)_ _π_ X _i_ _has maximum dislocation D_ = _O_ (log _n_ ) _._

_(ii)_ _For every_ s _∈Si_ [(1)] _and every_ v _∈_ KERNEL _i_ (s) _∪_ GUARD _i_ (s) _,_ POSs(v _,_ V _i_ ) _≤_ 100 _|_ V _mi|_ S1 _[.]_


_(iii)_ _For every_ s _∈Si_ [(1)] _, any_ w _∈_ KERNEL _i_ (s) _, and any_ g _∈_ GUARD _i_ (s) _, d_ (s _,_ w) _< d_ (s _,_ g) _._


_Proof._ We first start with the straightforward proof of (i). By Lemma B.1, quadruplet queries are
not repeated between rounds. Therefore, applying Lemma 2.1 to X _i_ = E( _Si_ [(1)] _, Si_ [(2)] ) _,_ we have that
_π_ X _i_ has maximum dislocation at most _D_ with probability at least 1 _−_ _n_ _[−]_ [4] _[/]_ [3] .


18


Then we show (ii). Fix some round _i_ . For any s _∈Si_ [(1)], define _B_ (s) := v _∈_ V _i_ : POSs(v _,_ V _i_ ) _≤_


_|_ V _i|_
100 _m_ S1


. For any fixed s _∈Si_ [(1)], since _Si_ [(2)] is drawn uniformly and independently with replace


(2) _|B_ (s) _|_
ment, the random variable _N_ s := �� _Si_ _∩_ _B_ (s) �� is binomial with mean E[ _N_ s] = _m_ S2 _·_ _|_ V _i|_ =


100 _m m_ S2 S1 _[.]_ [Recall] [that] _[m]_ [S2] [=] _[c]_ [S2] _[ k]_ [ log][3] _[ n]_ [and] _[m]_ [S1] [=] _[c]_ [S1] _[k]_ [ log][2] _[ n]_ [.] [Thus] [E][[] _[N]_ [s][]] [=] _[c]_ [S2] _c_ [ log] S1 _[ n]_


100 _m_ S2 S1 _[.]_ [Recall] [that] _[m]_ [S2] [=] _[c]_ [S2] _[ k]_ [ log] [and] _[m]_ [S1] [=] _[c]_ [S1] _[k]_ [ log] [.] [Thus] [E][[] _[N]_ [s][]] [=] [S2] _c_ S1 . Observe

that _m_ win = 2 max _{c_ win log _n, D}_ = Θ(log _n_ ).


Assume _c_ S2 is large enough so that E[ _N_ s] _≥_ max _{_ 1000 _m_ win _,_ 1000 log _n }._ By a standard Chernoff
bound, we get

Pr� _N_ s _<_ 500 _m_ win� _≤_ Pr� _N_ s _<_ 2 [1] [E][[] _[N]_ [s][]]      - _≤_ exp( _−_ E[ _N_ s] _/_ 8) _≤_ _n_ _[−]_ [6] _._

Taking a union bound over all s _∈Si_ [(1)] with probability at least 1 _−_ _n_ _[−]_ [4], _N_ s _≥_ 500 _m_ win for every
s _∈Si_ [(1)] .

So far the lower bound on _N_ s and (i) hold with probability 1 _−_ _n_ _[−]_ [Ω(1)] . We now continue the
analysis conditioned on this. Fix some s _∈Si_ [(1)] . Since any restriction _π_ X _i,_ s inherits the same
maximum dislocation bound as _π_ X _i_, it also has maximum dislocation _D_ . Thus, for any edge _{_ s _,_ s _[′]_ _} ∈_
E(s _, Si_ [(2)] ), if s _[′]_ _∈/_ _B_ (s), then RANK _π_ X _i,_ s [ _{_ s _,_ s _[′]_ _}_ ] must be at least _N_ s _−D_ _≥_ 500 _m_ win _−D_ . Therefore,
for every edge _{_ s _,_ v _} ∈_ E(s _,_ KERNEL _i_ (s) _∪_ GUARD _i_ (s)), it holds that RANK _π_ X _i,_ s [ _{_ s _,_ v _}_ ] _≤_ 2 _m_ win +

2 _D_ _≤_ 500 _m_ win _−D_ . By definition, v _∈Si_ [(2)], so v _∈_ _B_ (s).


Finally, we show (iii). We note that for any w _∈_ KERNEL _i_ (s) and g _∈_ GUARD _i_ (s) we have


RANK _π_ X _i,_ s _{_ s _,_ w _}_ _≤_ _m_ win _,_ RANK _π_ X _i,_ s _{_ s _,_ g _}_ _>_ _m_ win + 2 _D._


With maximum dislocation _D_, the true rank of every kernel edge is at most _m_ win + _D_ while the true
rank of every guard edge exceeds _m_ win + _D_ . Hence, every kernel edge precedes every guard edge
in the true order, and therefore _d_ (s _,_ w) _< d_ (s _,_ g). Thus, all of (i),(ii), (iii) hold.


Next, we show that the proximity score is a reliable indicator: vertices nearer than all kernels yield
large scores, while those farther than all guards yield small scores.
**Lemma** **B.3.** _In_ _any_ _round_ _i,_ _conditioned_ _on_ _Lemma_ _B.2,_ _with_ _probability_ _at_ _least_ 1 _−_ _n_ _[−]_ [4] _,_ _the_
_following hold simultaneously for every_ s _∈Si_ [(1)] _and every_ v _∈_ V _i \ Si_ [(2)] _:_


_(i)_ _If d_ (s _,_ v) _≤_ maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) _then_ PCOUNTs(v) _> ⌊m_ win _/_ 2 _⌋._


_(ii)_ _If d_ (s _,_ v) _>_ maxg _∈_ GUARD _i_ (s) _d_ (s _,_ g) _then_ PCOUNTs(v) _≤⌊m_ win _/_ 2 _⌋._


_(iii)_ _If_ PCOUNTs(v) _> ⌊m_ win _/_ 2 _⌋_ _then_ POSs(v _,_ V _i_ ) _≤_ 100 _|_ V _mi|_ S1


_Proof._ Fix a round _i_ and a vertex s _∈Si_ [(1)] . Consider any v _∈_ V _i \ {_ s _}_ . Recall that


     PCOUNTs(v) := **1** _{Q_ [˜] ( _{_ s _,_ v _}, {_ s _,_ g _}_ ) says _d_ (s _,_ v) _≤_ _d_ (s _,_ g) _}._


g _∈_ GUARD _i_ (s)


By Lemma B.2, we have maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) _<_ ming _∈_ GUARD _i_ (s) _d_ (s _,_ g). Moreover, by
Lemma B.1, quadruplet queries involved in computing PCOUNTs( _·_ ) do not overlap with those used
by PROBSORT(X _i_ ).


If _d_ (s _,_ v) _≤_ maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w), then the correct answer to every comparison is “Yes”. For

                                        sufficiently large _c_ win in _m_ win = 2 max _{c_ win log _n, D}_, Chernoff bounds yield Pr PCOUNTs(v) _<_
_⌊m_ win _/_ 2 _⌋_ - _≤_ exp( _−_ Θ( _m_ win)) _≤_ _n_ _[−]_ [7] . Similarly, if _d_ (s _,_ v) _>_ maxg _∈_ GUARD _i_ (s) _d_ (s _,_ g), then the

                         -                          correct answer to every comparison is “No” and Pr PCOUNTs(v) _>_ _⌊m_ win _/_ 2 _⌋_ _≤_ _n_ _[−]_ [7] . Taking
a union bound over all pairs (s _,_ v) with s _∈Si_ [(1)] and v _∈_ V _i_, both events (i) and (ii) hold with
probability at least 1 _−_ _n_ _[−]_ [4] .


19


Finally, from (ii), if PCOUNTs(v) _>_ _⌊m_ win _/_ 2 _⌋_ then _d_ (s _,_ v) _≤_ maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w). From
Lemma B.2 we have that for every vertex w _∈_ KERNEL _i_ (s) it holds that POSs(w _,_ V _i_ ) _≤_ 100 _|_ V _mi|_ S1 [.]

Hence, POSs(v _,_ V _i_ ) _≤_ maxw _∈_ KERNEL _i_ (s) POSs(w _,_ V _i_ ) _≤_ 100 _|_ V _mi|_ S1 [.]


Next, we establish that filtering removes only a small fraction of vertices, while ensuring that all
survivors are well-separated from the kernel sets.

**Lemma** **B.4.** _In_ _any_ _round i,_ _conditioned on Lemma B.2,_ _with probability_ 1 _−_ _n_ _[−]_ [4] _,_ _the following_
_hold simultaneously:_


_(i)_ _Every_ v _∈_ V _i_ _[′]_ _[satisfies][ d]_ [(][s] _[,]_ [ v][)] _[ >]_ [ max][w] _[∈]_ [KERNEL] _i_ [(][s][)] _[d]_ [(][s] _[,]_ [ w][)] _[ for all]_ [ s] _[ ∈S]_ _i_ [(1)] _._


_(ii)_ _|_ V _i_ _[′][|]_ _[≥]_ 35 _[|]_ [V] _[i][|][.]_


_Proof._ Conditioned on Lemma B.2, the conditions in Lemma B.3 hold with probability at least
1 _−_ _n_ _[−]_ [4] .

Case (i) : Lemma B.3 implies that for every s _∈Si_ [(1)] and every v _∈_ V _i_ _\_ _Si_ [(2)] if _d_ (s _,_ v) _≤_
maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) then PCOUNTs(v) _> ⌊m_ win _/_ 2 _⌋_ . By the definition of V _i_ _[′]_ [, a node][ v] _[ ∈]_ [V] _[i][ \ S][i]_
is excluded from V _i_ _[′]_ [if][ max] s _∈Si_ [(1)] PCOUNTs(v) _≥⌊m_ win _/_ 2 _⌋_ . So it follows directly that every v _∈_ V _i_ _[′]_

must satisfy the claimed inequality for all s _∈Si_ [(1)] .

Case (ii) : Lemma B.2 guarantees that for each s _∈Si_ [(1)],


KERNEL _i_ (s) _∪_ GUARD _i_ (s) _⊆{_ u _∈_ V _i_ : POSs(u _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1) _}._


Lemma B.3 (iii) implies that for every s _∈Si_ [(1)] and every v _∈_ V _i_ _\_ _Si_ [(2)], if POSs(v _,_ V _i_ ) _>_
_|_ V _i|/_ (100 _m_ S1) then PCOUNTs(v) _≤⌊m_ win _/_ 2 _⌋_ . Hence for every vertex v _∈_ V _i_ _\ Si_ [(2)] that is not
included in V _i_ _[′]_ [there exists at least one][ s] _[∈S]_ _i_ [(1)] such that POSs(v _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1). Thus, any
vertex s _∈Si_ [(2)] can cause at most _|_ V _i|/_ (100 _m_ S1) vertices from V _i \ Si_ [(2)] to not be included in V _i_ _[′]_ [.]
There are _m_ S1 samples in _Si_ [(1)] so at most _|_ V _i|/_ 100 vertices will not be included in V _i_ _[′]_ [.] [Therefore,]
_|_ V _i_ _[′][|]_ _[≥|]_ [V] _[i][| −|]_ [V] _[i][|][/]_ [100] _[≥]_ [(3] _[/]_ [5)] _[|]_ [V] _[i][|]_ [, as claimed.]


**Lemma** **B.5.** _In_ _any_ _round_ _i,_ _conditioned_ _on_ _Lemma_ _B.2,_ _the_ _following_ _holds_ _with_ _probability_ _at_
_least_ 1 _−_ _n_ _[−]_ [4] _:_ _for any query q_ = ( _{_ s1 _,_ v1 _}, {_ s2 _,_ v2 _}_ ) _with_ s1 _,_ s2 _∈Si_ [(1)] _and_ v1 _,_ v2 _∈_ V _i \ Si_ [(2)] _,_


_if_
_d_ (s _j,_ v _j_ ) _>_ max _for j_ = 1 _,_ 2 _,_
w _∈_ KERNEL _i_ (s _j_ ) _[d]_ [(][s] _[j][,]_ [ w][)]


_then_ ALG-TESTER( _q_ ) _behaves like an adversarial quadruplet oracle with error µ_ = 1 _._


_Proof._ We focus on the case s1 = s2; the other case is simpler. Assume that **e** _[⋆]_ _∈_
E(s1 _,_ KERNEL _i_ (s1)) such that RANK _π_ Z( **e** _[⋆]_ ) = _|_ Z _|_ . The tester discards any vertex u from
KERNEL _i_ (s2) with RANK _π_ Z( _{_ s2 _,_ u _}_ ) _∈_ [ _|_ Z _|_ _−D, |_ Z _|_ ). Since _π_ Z has maximum dislocation _D_,
every w _∈_ KERNEL _[′]_ _i_ [(][s][2][)][ satisfies]


_d_ (s2 _,_ w) _≤_ _d_ ( **e** _[⋆]_ ) _< d_ (s1 _,_ v1) _._


Moreover, assuming the constant _c_ win is large enough in _m_ win := 2 max _{c_ win log _n, D}_, it holds
_|_ KERNEL _[′]_ _i_ [(][s][2][)] _[|]_ [ = 2] _[m]_ [win] _[ −D]_ [= Ω(log] _[ n]_ [)][, since] _[ D]_ [=] _[ O]_ [(log] _[ n]_ [)][.]


Each oracle query is correct independently with probability at least 1 _−_ _φ_ _>_ 3 _/_ 4. Let _m_ :=
_|_ KERNEL _[′]_ _i_ [(][s][2][)] _[|]_ [.] [The] [tester] [outputs] [“yes”] [if] [more] [than] _[τ]_ [:=] _[⌊][m/]_ [2] _[⌋]_ [comparisons] [say] _[{]_ [s][1] _[,]_ [ v][1] _[}]_ [is]
larger, and “no” otherwise.


20


_Case 1:_ TCOUNT _>_ _τ_ _(tester outputs “yes”)._ This can only be wrong if _d_ (v2 _,_ s2) _>_ 2 _d_ (v1 _,_ s1). In
that case, for every w _∈_ KERNEL _[′]_ _i_ [(][s][2][)][,]

_d_ (v2 _,_ w) _≥_ _d_ (v2 _,_ s2) _−_ _d_ (s2 _,_ w) _≥_ _d_ (v2 _,_ s2) _−_ _d_ ( **e** _[⋆]_ ) _>_ 2 _d_ (v1 _,_ s1) _−_ _d_ ( **e** _[⋆]_ ) _>_ _d_ (v1 _,_ s1) _._


Thus every “yes” response is incorrect, and by Chernoff bounds Pr[ _T_ _> τ_ ] _≤_ _n_ _[−]_ [8] .


_Case 2:_ TCOUNT _≤_ _τ_ _(tester outputs “no”)._ This can only be wrong if _d_ (v1 _,_ s1) _>_ 2 _d_ (v2 _,_ s2). In
that case, for every w _∈_ KERNEL _[′]_ _i_ [(][s][2][)][,]


_d_ (v2 _,_ w) _≤_ _d_ (v2 _,_ s2) + _d_ (s2 _,_ w) _<_ 2 _d_ (v2 _,_ s2) _<_ _d_ (v1 _,_ s1) _,_


so every “no” response is incorrect. Again, Pr[ _T_ _≤_ _τ_ ] _≤_ _n_ _[−]_ [8] .


The case s1 = s2 is analogous: in Case 1 we have directly _d_ (s2 _,_ w) _≤_ _d_ (s2 _,_ v1), and in Case 2 we
have _d_ (s2 _,_ w) _≤_ _d_ (v2 _,_ s2). Thus, in all cases, the tester fails with probability at most _n_ _[−]_ [8] .


Finally, union bounding over all _O_ ( _n_ [4] ) possible queries in round _i_ gives overall failure probability
at most _n_ _[−]_ [4], as claimed.


**Lemma B.6.** _In any round i of_ ALG-G _, conditioned on Lemma B.2, with probability at least_ 1 _−n_ _[−]_ [3] _,_
_the following holds:_


_(i)_ _for every_ v _∈_ V _i_ _[′′][,][ d]_ [(][v] _[,][ M][i]_ [(][v][))] _[ ≤]_ [4] _[ d]_ [(][v] _[,][ S]_ _i_ [(1)] ) _._

_(ii)_ _|_ V _i_ _[′′][|]_ _[≥|]_ [V] _[i][|][/]_ [4] _[.]_


_Proof._ Conditioned on Lemma B.2, each of Lemma B.3, Lemma B.4, and Lemma B.5 fails with
probability at most _n_ _[−]_ [4] . By a union bound, all of them hold simultaneously with probability 1 _−_
3 _n_ _[−]_ [4] . We analyze under this assumption.

By Lemma B.4, for every _v_ _∈_ V _i_ _[′]_ [and every][ s] _[∈S]_ _i_ [(1)] we have _d_ (s _, v_ ) _>_ maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) _._

Since ADVSORT only calls ALG-TESTER on pairs of edges in Y _i_ = E( _Si_ [(1)] _,_ V _i_ _[′]_ [)][,] [every] [invo-]
cation of ALG-TESTER satisfies the preconditions of Lemma B.5. Specifically, for any query
_q_ = ( _{_ s1 _,_ v1 _}, {_ s2 _,_ v2 _}_ ) with s1 _,_ s2 _∈Si_ [(1)] and v1 _,_ v2 _∈_ V _i_ _[′]_ [,] [the] [following] [condition] [holds] [for]
_j_ = 1 _,_ 2:
_d_ (v _j,_ s _j_ ) _>_ max
w _∈_ KERNEL _i_ (s _j_ ) _[d]_ [(][s] _[j][,]_ [ w][)] _[.]_


Therefore, by Lemma B.5, whenever ALG-TESTER is invoked in behaves like an adversarial quadruplet oracle with error _µ_ = 1. It follows from Lemma 2.2 that the ordering _π_ Y _i_ is 4-sorted: for any
**e** 1 _,_ **e** 2 _∈_ Y _i_,
RANK _π_ Y _i_ ( **e** 1) _≤_ RANK _π_ Y _i_ ( **e** 2) _⇒_ _d_ ( **e** 1) _≤_ 4 _d_ ( **e** 2) _._

Let f v be the first edge incident to v _∈_ V _i_ _[′]_ [in] _[ π]_ [Y] _i_ [, and recall] _[ M][i]_ [(][v][)][ is its endpoint in] _[ S]_ _i_ [(1)] . Then for
any _s ∈Si_ [(1)],
_d_ (v _, Mi_ (v)) = _d_ ( f v) _≤_ 4 _d_ ( _{s,_ v _}_ ) _,_

hence _d_ (v _, Mi_ (v)) _≤_ 4 _· d_ (v _, Si_ [(1)] ). This proves (i).

For (ii), Lemma B.4 guarantees _|_ V _i_ _[′][| ≥]_ [3] 5 _[|]_ [V] _[i][|]_ [.] [The safe set is]

V _i_ _[′′]_ [=] _[ {]_ [v] _[ ∈]_ [V] _i_ _[′]_ [:] [RANK] _[π]_ N _i_ [(] [f] [v][)] _[ ≤|]_ [V] _[i][|][/]_ [4] _[}][.]_

Since _|_ V _i_ _[′][|]_ _[≥]_ [3] _[|]_ [V] _[i][|][/]_ [5][,] [at] [least] _[|]_ [V] _[i][|][/]_ [4] [vertices] [of] [V] _i_ _[′]_ [satisfy] [the] [rank] [condition,] [and] [hence] _[|]_ [V] _i_ _[′′][|]_ _[≥]_
_|_ V _i|/_ 4. This proves (ii).


**Structural** **Property.** Let OPT [1] _k_ [(][V][)] [be] [the] [optimal] _[k]_ [-median] [cost,] [and] [let] [C] _[⋆]_ [denote] [the] [set] [of]
centers in an optimal solution. Let _L_ := OPT [1] _k_ [(][V][)] _[/n]_ [.] [For a round] _[ i]_ [, define a vertex][ v] _[ ∈]_ [V] _[i]_ [as] _[ good]_
if
_d_ (v _, Si_ [(1)] ) _≤_ max _{L,_ 2 _· d_ (v _,_ C _[⋆]_ ) _},_

and _bad_ otherwise. Let V _i_ _[g][,]_ [ V] _i_ _[b]_ [be the good/bad sets so][ V] _[i]_ [=][ V] _i_ _[g]_ _[∪]_ [V] _i_ _[b]_ [.]


21


**Lemma B.7.** _In any round i, with probability at least_ 1 _−_ _n_ _[−]_ [3] _,_


_|_ V _i_ _[b][|]_ [=] _[O]_                             - log _|_ V _i n|_                             - _._


_Proof._ Let c _∈_ C _[⋆]_ . Furthermore, let Nc _⊆_ V be the set of all vertices that get mapped to c
according to the optimal clustering. Fix a round _i_, and let s _[∗]_ _∈Si_ [(1)] be the vertex such that
s _[⋆]_ = arg min (1) _d_ (c _,_ s).
s _∈Si_


Define
B = _{_ v _∈_ Nc _∩_ V _i_ : POSc(v _,_ V _i_ ) _<_ POSc(s _[∗]_ _,_ V _i_ ) _}._

Observe that if v _∈_ (Nc _∩_ V _i_ ) _\_ B, then _d_ (v _,_ c) _≥_ _d_ (s _[∗]_ _,_ c) and by triangle inequality _d_ (v _,_ s _[∗]_ ) _≤_
2 _d_ (v _,_ c). Hence, every v _∈_ V _i \_ B is good. Therefore, V _i_ _[b]_ _[∩]_ [N][c] _[⊆]_ [B][ and] _[ |]_ [N][c] _[ ∩]_ [V] _i_ _[b][| ≤]_ [POS] _[c]_ [(][s] _[∗][,]_ [ V] _[i]_ [)][.]

For any constant _γ_ _≤_ _c_ S1,


- _|_ V _i_ _[b]_ _[∩]_ [N] _[c][| ≥]_ _kγ|_ logV _i| n_ - _≤_ Pr _Si_


Pr
_Si_


- _|_ V _i|_ - - 1 - _|Si_ [(1)] _|_ 1
POS _c_ (s _[∗]_ _,_ V _i_ ) _≥_ _kγ_ log _n_ _≤_ 1 _−_ _kγ_ log _n_ _≤_ _n_ [Ω(1)] _[.]_


Union bounding over _c ∈_ C _[⋆]_ gives


Pr _Si_ - _|_ V _i_ _[b][| ≥]_ _γ_ log _|_ V _i| n_ - _≤_ _n_ [Ω(1)] 1 _[.]_


B.3.2 PUTTING EVERYTHING TOGETHER


Define the event IDEAL as all the conditions in all preceding lemmas hold. We perform subsequent
analysis under this assumption. We prove structural lemmas: in each round, there are sufficiently
many surviving good vertices to control the cost of the removed bad ones (Lemma B.9), and across
all rounds we can construct an injection from bad to good vertices (Lemma B.10). We use these
structural guarantees to bound the total cost by _O_ (1) _·_ OPT [1] _k_ [(][V][)][ (Corollary B.11).]

**Lemma B.8.** _With probability at least_ 1 _−_ _n_ _[−]_ [Ω(1)] _, the event_ IDEAL _holds; i.e., all properties from_
_Lemmas B.2,B.3, B.4, B.5,B.6,B.7 holds simultaneously across all r_ = _O_ (log _n_ ) _rounds._


Let _Ri_ = V _i_ _[′′]_ _[∪S]_ _i_ [(1)] _∪Si_ [(2)] be the set of vertices removed in round _i_ . Define V [¯] _i_ _[b]_ [:=] [V] _i_ _[b]_ _[∩]_ [V] _i_ _[′′]_ [to be]
the subset of bad vertices that actually get included in the safe-set in round _i_, and let V [¯] _[b]_ := [�] _i_ _[r]_ =1 [V][¯] _i_ _[b]_
denote the collection of all such vertices across all rounds. We mainly need to worry about these.
Similarly, define V [¯] _i_ _[g]_ [:=] [V] _i_ _[g]_ _[\ R][i]_ [as the set of good vertices that are not included in the safe-set in]
round _i_, and let V [¯] _[g]_ := [�] _i_ _[r]_ =1 [V][¯] _i_ _[g]_ [be the union of all such good vertices.]

**Lemma B.9.** _Conditioned on the event_ IDEAL _, the following hold for every round i:_


_1._ _|_ V [¯] _i_ _[g][|]_ _[≥|]_ [V] _[i][|][/]_ [100] _[.]_

_2._ _For any_ b _∈_ V [¯] _i_ _[b]_ _[and]_ [ g] _[ ∈]_ [V][¯] _i_ _[g][,]_ _d_ (b _, Mi_ (b)) _≤_ 4 _d_ (g _,_ C _[⋆]_ ) _._


_Proof._ By Lemma B.4, _|_ V _i_ _[′][|]_ _[≥]_ 5 [3]


_Proof._ By Lemma B.4, _|_ V _i_ _[|]_ _[≥]_ 5 _[|]_ [V] _[i][|]_ [.] [Since the safe-set removes exactly] _[ |]_ [V] _[i][|][/]_ [4][ vertices, it follows]

that
_|_ V _i_ _[′]_ _[\]_ [ V] _i_ _[′′][|]_ _[≥]_ 35 _[|]_ [V] _[i][| −]_ [1] 4 _[|]_ [V] _[i][|]_ [=] 207 _[|]_ [V] _[i][|][.]_

By Lemma B.7, at most _O_ ( _|_ V _i|/_ log _n_ ) of these are bad, and since _Ri_ = V _i_ _[′′]_ _[∪S]_ _i_ [(1)] _∪Si_ [(2)] by
ensuring _ni_ _>_ Ω( _m_ S1) _,_ Ω( _m_ S2) and, we can ensure that


[1] 4 _[|]_ [V] _[i][|]_ [=] 207 _[|]_ [V] _[i][|][.]_


7
_|_ V [¯] _i_ _[g][|]_ [ =] _[ |]_ [V] _i_ _[g]_ _[\ R][i][|]_ [ =] _[ |]_ [(][V] _i_ _[′]_ _[\ R][i]_ [)] _[ \]_ [ V] _i_ _[b][|]_ _[≥]_ _[|]_ [V] _[i][|]_
20 _[|]_ [V] _[i][| −]_ _[O]_ [(] log _n_ [)] _[ ≥|]_ [V] _[i][|][/]_ [100] _[.]_


always holds for suitably large _n_ .


22


Next, recall that V _i_ _[′′]_ [consists of vertices that appear among the the] _[ ⌊|]_ [V] _[i][|][/]_ [4] _[⌋]_ [edges in] _[ π]_ [N] _i_ [.] [Since] _[ π]_ [N] _i_
is 4-sorted (Lemma 2.2), for any u _∈_ V _i_ _[′′]_ [and any][ w] _[ ∈]_ [V] _i_ _[′]_ _[\]_ [ V] _i_ _[′′]_ [we have]


_d_ ( f u) _≤_ 4 _d_ ( f w) _._


By construction of _Mi_, this yields

_d_ (u _, Mi_ (u)) _≤_ 4 _d_ (w _,_ C _[⋆]_ ) for any u _∈_ V _i_ _[′′][,]_ [w] _[ ∈]_ [V] _i_ _[′]_ _[\]_ [ V] _i_ _[′′][.]_


Since V [¯] _i_ _[g]_ _[⊆]_ [V] _i_ _[′]_ _[\]_ [ V] _i_ _[′′]_ [and] [V][¯] _i_ _[b]_ _[⊆]_ [V] _i_ _[′]_ [, this implies that for any][ g] _[ ∈]_ [V][¯] _i_ _[g]_ [and][ b] _[ ∈]_ [V][¯] _i_ _[b]_

_d_ (b _, Mi_ (b)) _≤_ 4 _d_ (g _,_ C _[⋆]_ ) _._


Note that in the case of the last round V _r_ _[b]_ [=] _[ ∅]_ [and the claim vacuously holds.]


**Lemma B.10.** _Conditioned on the event_ IDEAL _, there exists a map ψ_ : V [¯] _[b]_ _→_ V [¯] _[g]_ _such that:_


_(i)_ _for every_ b _∈_ V [¯] _i_ _[b][, the image satisfies][ ψ]_ [(][b][)] _[∈]_ [V][¯] _i_ _[g]_ _[(i.e. each removed bad vertex is mapped]_
_to a surviving good vertex from the same round),_


_(ii)_ _ψ is an injection._


_Proof._ By Lemma B.7, in every round _i_, _|_ V _i_ _[b]_ _[∩]_ [V] _i_ _[′′][|]_ _≤_ 100 log _|_ V _i|_ _n_ _[,]_ [while] [by] [Lemma] [B.9,] [V][¯] _i_ _[g]_ _[⊆]_


(V _i_ _[′]_ _[\ R][i]_ [)] _[ ∩]_ [V] _i_ _[g]_ [and of size] _[ |]_ [V][¯] _i_ _[g][|]_ _[≥]_ _|_ 100V _i|_ _[.]_ [ Moreover, since each round removes exactly a] [1] 4


(V _i_ _[\ R][i]_ [)] _[ ∩]_ [V] _i_ [and of size] _[ |]_ [V] _i_ _[|]_ _[≥]_ 100 _i_ _[.]_ [ Moreover, since each round removes exactly a] 4 [fraction]

of V _i_, we have


_|_ V _i_ +1 _|_ _≤_ [3]


4 _[|]_ [V] _[i][|][.]_


We construct _ψ_ by reverse induction over the rounds. For the final round _i_ = _r_, the claim holds by
default as there are no bad vertices, i.e. V _r_ _[b]_ [=] _[ ∅]_ [.]


Assume injections _ψj_ are defined for all _j_ _>_ _i_, such that the injection property is preserved so far.
Let


U :=


denote the set of _used_ vertices. Therefore,


 - _ψj_ (V _j_ _[b]_ _[∩]_ [V] _j_ _[′′]_ [)] _[,]_

_j_ = _i_ +1


_r_


_∞_
�(3 _/_ 4) _[t]_ = 3 _|_ V _i|_

100 log _n_ _[.]_
_t_ =1


_|_ U _|_ _≤_


_r_


_j_ = _i_ +1


_|_ V _j|_ _|_ V _i|_
100 log _n_ _[≤]_ 100 log _n_


Now define V _i_ [rem] := V [¯] _i_ _[g]_ _[\]_ [ U][.] [Then]


_|_ V _i_ [rem] _|_ _≥_ _[|]_ [V] _[i][|]_


3 _|_ V _i|_

_[|]_ [V] _[i][|]_

100 _[−]_ 100 log _n_ _[≥]_ _[|]_ 150 [V] _[i][|]_


_|_ V _i|_

_[|]_ [V] _[i][|]_ _i_ _[| ≥|]_ [V][¯] _i_ _[b][|]_

150 _[≥]_ 100 log _n_ _[≥|]_ [V] _[b]_


for sufficiently large _n_ . Thus an injection _ψi_ : V [¯] _i_ _[b]_ _[→]_ [V] _i_ [rem] exists, and its image is disjoint from U.
This completes the proof.


**Corollary B.11.** _Conditioned on the event_ IDEAL _,_ [�] v _∈_ V _[d]_ [(][v] _[,][ M]_ [(][v][))] [=] _[O]_ [(1)] _[ ·]_ [ OPT][1] _k_ [(][V][)] _[ .]_


_Proof._ By Lemma B.10 and Lemma B.9 there exists an injection _ψ_ : V [¯] _[b]_ _→_ V [¯] _[g]_ such that
_d_ (b _, µ_ (b)) _≤_ 4 _d_ (Ψ(b) _,_ C _[⋆]_ ) _._


Summing over V [¯] _[b]_ and using that _ψ_ is injective,

  -  


- _d_ (v _,_ C _[⋆]_ ) = 4 OPT [1] _k_ [(][V][)] _[.]_

v _∈_ V


- _d_ (b _, M_ (b)) _≤_ 4 

b _∈_ V [¯] _[b]_ b _∈_ V [¯]


- _d_ (Ψ(b) _,_ C _[⋆]_ ) _≤_ 4 

b _∈_ V [¯] _[b]_ v _∈_ V


23


Note that by the definition of “good”, for every v _∈_ V _\_ V _i_ _[b]_ [must have been good, i.e.,]

_d_ (v _, M_ (v)) _≤_ max _{L,_ 4 _d_ (v _,_ C _[⋆]_ ) _},_ _L_ := OPT [1] _k_ [(][V][)] _[/n.]_


Hence


- _k_ [(][V][)]

max _{L,_ _C d_ (g _,_ C _[⋆]_ ) _}_ _≤_ 4 OPT [1] _k_ [(][V][)+] _[n][·]_ [OPT][1] = 5 OPT [1] _k_ [(][V][)] _[.]_
_n_
v _∈_ V _\_ V [¯] _[b]_


- _d_ (v _, µ_ (v)) _≤_ 

g _∈_ V _\_ V [¯] _[b]_ v _∈_ V _\_


Taken together,

     - _d_ (v _, M_ (v)) = _O_ (1) _·_ OPT [1] _k_ [(][V][)] _[,]_

v _∈_ V


**Query** **Complexity.** Since PROBSORT( _·_ ) is always invoked on a set of size _m_ S1 _·_ _m_ S2 =
_O_ ( _k_ [2] polylog _n_ ), each call uses _O_ ( _k_ [2] polylog _n_ ) queries. Computing proximity scores requires
one evaluation per (s _,_ v) pair, i.e. _O_ ( _|Si_ [(1)] _|_ _·_ _|_ V _i|_ ) = _O_ ( _nk_ polylog _n_ ) queries in a round. Each
invocation of ALG-TESTER occurs within ADVSORT on Y _i_ = E( _Si_ [(1)] _,_ V _i_ _[′]_ [)][,] [contributing] [an-]
other _O_ ( _nk_ polylog _n_ ) queries. Therefore, the total per-round query complexity is _O_ (( _nk_ +
_k_ [2] )polylog _n_ ) = _O_ ( _nk ·_ polylog _n_ ). Since there are _r_ = _O_ (log _n_ ) rounds, the overall query complexity is _O_ - _nk ·_ polylog _n_ �.


Notice that all results, in this section, up to constant factors in asymptotic analysis, can be extended
to any ( _k, p_ )-clustering instance, where _p_ is a constant or _∞_ . We conclude with Theorem 3.1.


C BOUNDED DOUBLING DIMENSION


In this section, we show how to improve the query complexity of ALG-G to _O_ (( _n_ + _k_ [2] ) polylog _n_ )
when the underlying metric has a bounded doubling dimension. We use the following data structure
and query procedure from Raychaudhury et al. (2025).


**Lemma** **C.1.** _Let_ Σ = (V _, d_ ) _be_ _a_ _metric_ _of_ _bounded_ _doubling_ _dimension_ _with_ _|_ V _|_ = _n,_ _and_
E(V) = E(V _,_ V) _accessible_ _under_ _the_ _R-model._ _For_ _S_ _⊆_ V _and_ _an_ _α-sorted_ _ordering_ _π_ E( _S_ ) _,_
_there_ _exists_ _a_ _procedure_ CONSTRUCT( _S, π_ E( _S_ )) _that_ _builds_ _a_ _structure_ T _without_ _executing_ _any_
_quadruplet oracle, such that given any_ v _∈_ V _\ S, a procedure_ TRAVERSE(T _,_ v) _returns a set F_ _⊆_
E(v _, S_ ) _of size O_ (polylog _n_ ) _such that_ min **e** _∈F d_ ( **e** ) _≤_ 4 _α · d_ (v _, S_ ) _._ _The traversal requires answers_
_to_ _O_ (polylog _n_ ) _quadruplet_ _queries_ _of_ _the_ _form_ ( _{_ s1 _,_ s2 _}, {_ s3 _,_ v _}_ ) _with_ s _i_ _∈S_ _from_ _a_ _adversarial_
_quadruplet oracle with noise µ ≤_ 1 _._


C.1 ALG-D


Similar to the general metric case, the algorithm proceeds in rounds. We describe the steps of a
generic round _i_ . The first three steps are identical to ALG-G, but we restate them for convenience.


1. _Sampling._ Sample uniformly at random a set of vertices _Si_ [(1)] _,_ _Si_ [(2)] _⊆_ V _i_ of sizes _m_ S1 =
_c_ S1 _k_ log [2] _n_ and _m_ S2 = _c_ S2 _k_ log [3] _n_ respectively, where _c_ S1 _, c_ S2 are suitable constants. Let
_Si_ := _Si_ [(1)] _∪Si_ [(2)] .

2. _Order edges._ Let X _i_ = E( _Si_ [(1)] _, Si_ [(2)] ) and compute _π_ X _i_ = PROBSORT(X _i_ ).

3. _Kernel_ _and_ _guard_ _sets._ Let _m_ win = 2 max _{c_ win log _n, D}_ . For each s _∈Si_ [(1)], let X _i,_ s =
E(s _, Si_ [(2)] ) and _π_ X _i,_ s the ordering of X _i,_ s induced by _π_ X _i_ . For every s _∈Si_ [(1)], compute

KERNEL _i_ (s) = _{_ w _∈Si_ [(2)] : RANK _π_ X _i,_ s [ _{_ s _,_ w _}_ ] _≤_ _m_ win _},_


GUARD _i_ (s) = _{_ g _∈Si_ [(2)] : _m_ win + _D_ _<_ RANK _π_ X _i,_ s [ _{_ s _,_ g _}_ ] _≤_ 2 _m_ win + _D}._


24


4. _Identify close pairs_ . For each s _̸_ = s _[′]_ _∈Si_ [(1)], compute PCOUNTs(s _[′]_ ). Define _{_ s _,_ s _[′]_ _}_ as _close_ if

max _{_ PCOUNTs(s _[′]_ ) _,_ PCOUNTs _′_ (s) _} ≥⌊m_ win _/_ 2 _⌋._


5. _Partition_ _into_ _classes._ Construct a graph _Gi_ on _Si_ [(1)] whose edges are all close pairs as defined
above. As we show in the analysis, the graph _Gi_ is _O_ (log _n_ ) _-degenerate_, i.e., in any subgraph of
_Gi_ there exists at least one vertex with degree _O_ (log _n_ ). We run the greedy coloring algorithm
for degenerate graphs Lick & White (1970) on _Gi_ to get a coloring of the vertices in _Gi_ and let
_χi_ be the number of different colors. We partition _Si_ [(1)] into classes _Si_ [(1] _[,]_ [1)] _, . . ., Si_ [(1] _[,χ][i]_ [)] based on
their colors. Let E [(] _i_ _[j]_ [)] = _{_ (u _,_ v) _|_ u _,_ v _∈Si_ [(1] _[,j]_ [)] _}_ for each _j_ = 1 _, . . ., χi_ .

6. _Approximate nearest neighbors._ For each class _Si_ [(1] _[,j]_ [)] :

(a) _Build._ Compute _π_ E( _ij_ ) = ADVSORT(E [(] _i_ _[j]_ [)][)][ with A][LG][-T][ESTER][ as the comparator, then con-]

struct T _i_ [(] _[j]_ [)] = CONSTRUCT( _Si_ [(1] _[,j]_ [)] _, π_ E( _ij_ ) [)][.]

(b) _Traverse._ For each v _∈_ V _i_ _\_ _Si_, run TRAVERSE(T _i_ [(] _[j]_ [)] _,_ v). During execution, the procedure issues comparisons of the form ( _{_ s1 _,_ s2 _}, {_ s3 _,_ v _}_ ) with s1 _,_ s2 _,_ s3 _∈Si_ [(1] _[,j]_ [)] . For each
comparison, first check whether


max _{_ PCOUNTs1(v) _,_ PCOUNTs2(v) _,_ PCOUNTs3(v) _}_ _≥⌊m_ win _/_ 2 _⌋._


If the condition holds, eliminate v and proceed to the next vertex. Otherwise, call
ALG-TESTER( _{_ s1 _,_ s2 _}, {_ s3 _,_ v _}_ ) and pass its response to TRAVERSE. If v is not eliminated,
TRAVERSE outputs _O_ (polylog _n_ ) edges from E(v _, Si_ [(1] _[,j]_ [)] ).

7. _Collect results._ Let Y [(] _i_ _[j]_ [)] be the set of edges returned for class _j_ in the previous step, and define
Y� _i_ = [�] _j_ _[χ]_ =1 _[i]_ [Y] _i_ [(] _[j]_ [)][.] [For every eliminated vertex][ v][,] [remove all incident edges from] [Y][�] _[i]_ [,] [and denote]
the remaining set by Y _i_ . Let V _i_ _[′]_ [:=][ V][(][Y] _[i]_ [)] _[ \ S][i]_ [.]
8. _Final ordering._ Compute _π_ Y _i_ = ADVSORT(Y _i_ ) using ALG-TESTER as the comparator.
For each v _∈_ V _i_ _[′]_ [, let] [ f] [v] [be the first edge incident to][ v][ in] _[ π]_ [Y] _i_ [.]
9. _Safe-set and mapping._ Let _π_ N _i_ be the ordering of N _i_ induced by _π_ Y _i_ . Define V _i_ _[′′]_ [=] _[{]_ [v] _[∈]_ [V] _i_ _[′]_ [:]

RANK _π_ N _i_ ( f v) _≤|_ V _i|/_ 4 _}_ . For every v _∈_ V _i_ _[′′]_ [,] [define] _[M][i]_ [(][v][)] [as] [the] [endpoint] [of] [f] [v] [in] _[S]_ _i_ [(1)] . For
every v _∈Si_, define _Mi_ (v) := v.
10. _Recurse._ Set V _i_ +1 = V _i \_ (V _i_ _[′′]_ _[∪S][i]_ [)][ and proceed to next round.]


C.2 ANALYSIS


**Lemma C.2.** _In any round i, conditioned on Lemma B.2 with probability_ 1 _−_ _n_ _[−]_ [4] _, χi_ = _O_ (log _n_ ) _._


_Proof._ Fix a round _i_ . By Lemma B.3, we can argue that w.h.p. for every _{_ s _,_ s _[′]_ _} ∈_ E( _Si_ [(1)] ), if _{_ s _,_ s _[′]_ _}_
is close then POSs(s _[′]_ _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1) or POSs _′_ (s _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1).


_′_ (1)
For any s _∈Si_ [(1)] define deg [+] (s) := �� _{_ s _∈Si_ _\ {_ s _}_ : POSs(s _[′]_ _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1) _}_ �� _._ We note
that deg [+] (s) is not the degree of s in _Gi_ .

Recall _m_ S1 = _c_ S1 _k_ log [2] _n_ . For a uniformly random s _[′]_ _∈_ V _i \ {_ s _}_,


 Pr


_|_ V _i|_
POSs(s _[′]_ ; V _i_ ) _≤_
100 _m_ S1


- 1
_≤_ _._
100 _m_ S1


Over the _m_ 1 _−_ 1 (with-replacement) draws in _Si_ [(1)] _\ {_ s _}_, E[deg [+] ( _s_ )] _≤_ 1001 [.] [By] [the] [Chernoff]
bound, for any constant _c_ _≥_ 10 and large enough _n_, Pr�deg [+] ( _s_ ) _≥_ _C_ log _n_ - _≤_ _n_ _[−]_ [3] _._ By union
bound for every s _∈Si_ [(1)] with probability at least 1 _−_ _n_ 1 [2] [it holds that][ deg][+][(][s][) =] _[ O]_ [(log] _[ n]_ [)][.]

From the above argument, we can conclude that for every _s_ _∈Si_ [(1)] can contribute to at most
_O_ (log _n_ ) edges being close. Thus, with high probability, for any subgraph _H_ of _Gi_, the total number


25


of edges is bounded by _O_ (log _n_ ) _· |_ V( _H_ ) _|_ . Thus, for every subgraph _H_ of _Gi_ there is at least some
vertex with degree at most _O_ (log _n_ ), and the graph _Gi_ is _O_ (log _n_ )-degenerate Lick & White (1970).
It is known that the greedy coloring according to the degeneracy ordering can color graph _Gi_ with
_χi_ = _O_ (log _n_ ) colors Lick & White (1970).


**Lemma** **C.3.** _In_ _any_ _round_ _i,_ _conditioned_ _on_ _Lemma_ _B.2,_ _with_ _probability_ 1 _−_ _n_ _[−]_ [3] _,_ _whenever_
ALG-TESTER _is invoked, it behaves like an adversarial quadruplet oracle with error µ_ = 1 _._


_Proof._ Conditioned on Lemma B.2, each of Lemma B.3 and Lemma B.5 holds with probability
1 _−_ _n_ _[−]_ [4] . By a union bound, they simultaneously hold with probability 1 _−_ _n_ _[−]_ [3] . We perform the
analysis under this assumption.


By Lemma B.5, ALG-TESTER behaves like an adversarial oracle with error _µ_ = 1 if for any query
_q_ = ( _{_ s _[′]_ 1 _[,]_ [ v] 1 _[′]_ _[}][,][ {]_ [s] _[′]_ 2 _[,]_ [ v] 2 _[′]_ _[}]_ [)][, where][ s] _[′]_ 1 _[,]_ [ s] _[′]_ 2 _[∈S]_ _i_ [(1)] and v1 _[′]_ _[,]_ [ v] 2 _[′]_ _[∈]_ [V] _[i]_ _[\ S]_ _i_ [(2)], it holds that
_d_ (v _j_ _[′]_ _[,]_ [ s] _[′]_ _j_ [)] _[>]_ max _j_ _[,]_ [ w][)] ( _j_ = 1 _,_ 2) _._ (2)
w _∈_ KERNEL _i_ (s _j_ ) _[d]_ [(][s] _[′]_


By Lemma B.3, if _d_ (s _,_ v) _≤_ maxw _∈_ KERNEL _i_ (s) _d_ (s _,_ w) then PCOUNTs(v) _> ⌊m_ win _/_ 2 _⌋_ .


_Construction_ _phase._ Every call compares _{_ s1 _,_ s2 _}, {_ s3 _,_ s4 _}_ with s _ℓ_ _∈Si_ [(1] _[,j]_ [)] . Since each _Si_ [(1] _[,j]_ [)] is
an independent set of _Gi_ we have that s1 is not close to s2 and s3 is not close to s4. Consider
the pair _{_ s1 _,_ s2 _}_ . Since it is not close, we have max _{_ PCOUNTs1(s2) _,_ PCOUNTs2(s1) _}_ _<_ _⌊m_ win _/_ 2 _⌋_ .
Without loss of generality, assume that PCOUNTs1(s2) _< ⌊m_ win _/_ 2 _⌋_ . From Lemma B.3, we have that
_d_ (s1 _,_ s2) _>_ maxw _∈_ KERNEL _i_ (s1) _d_ (s1 _,_ w). Similarly, the same inequality holds for the pair _{_ s3 _,_ s4 _}_, and
hence (2) is satisfied.


_Traverse_ _phase._ Queries are of the form ( _{_ s1 _,_ s2 _}, {_ s3 _,_ v _}_ ) with s _ℓ_ _∈Si_ [(1] _[,j]_ [)] and v _∈_ V _i_ _\_ _Si_ .
Before invoking ALG-TESTER, the algorithm ensures PCOUNTs3(v) _≤⌊m_ win _/_ 2 _⌋_, which implies
_d_ (v _,_ s3) _>_ maxw _∈_ KERNEL _i_ (s3) _d_ (s3 _,_ w). Combined with the argument for _{_ s1 _,_ s2 _}_ above, the tester
precondition (2) holds.


_Final_ _sorting._ Queries are of the form ( _{_ s1 _,_ v1 _}, {_ s2 _,_ v2 _}_ ) with s1 _,_ s2 _∈Si_ [(1] _[,j]_ [)] and v1 _,_ v2 _∈_ V _i_ _[′]_ [.]
Since any such _{_ s _,_ v _}_ must have passed the traverse step, we have PCOUNTs(v) _≤⌊m_ win _/_ 2 _⌋_, and
therefore (2) holds.


**Lemma C.4.** _In any round i of_ ALG-D _, conditioned on Lemma B.2, with probability at least_ 1 _−n_ _[−]_ [3] _,_
_the following holds:_


_(i)_ _For every_ v _∈_ V _i_ _[′′][,][ d]_ [(][v] _[,][ M][i]_ [(][v][))] _[ ≤]_ [64] _[ d]_ [(][v] _[,][ S]_ _i_ [(1)] ) _._

_(ii)_ _|_ V _i_ _[′′][|]_ _[≥|]_ [V] _[i][|][/]_ [4] _[.]_


_Proof._ Conditioned on Lemma B.2, each of Lemma B.3, Lemma B.4, and Lemma C.3 holds with
probability 1 _−_ _n_ _[−]_ [4] . By a union bound, they simultaneously hold with probability 1 _−_ _n_ _[−]_ [3] . We
perform the analysis under this assumption.


By Lemma C.3, all calls to ALG-TESTER are correct. Hence, by Lemma C.1 both the construction
steps succeed and every traverse step (for non-eliminated vertices) is successful for every vertex in
_Si_ [(1)] . From Lemmas C.3 and 2.2, the ordering _π_ E( _ij_ ) is 4-sorted, for every _j_ = 1 _, . . ., χi_ . From

Lemma C.1, for every vertex v _∈_ V _i \ Si_, the set Y _i_ contains an edge **e** v _,j_ _∈_ E(v _, Si_ [(1] _[,j]_ [)] ), for each
class _Si_ [(1] _[,j]_ [)], such that _d_ ( **e** v _,j_ ) _≤_ 16 _d_ (v _, Si_ [(] _[j]_ [)] ). Similarly, from Lemmas C.3 and 2.2 the ordering _π_ Y _i_
is 4-sorted.

Let f v be the lowest rank edge incident to v _∈_ V _i_ _[′]_ [in] _[ π]_ [Y] _i_ [,] [and recall] _[ M][i]_ [(][v][)][ is its endpoint in] _[ S]_ _i_ [(1)] .
Then for every v _∈_ V _i_ _[′′]_ [,]

_d_ (v _, Mi_ (v)) = _d_ ( f v) _≤_ 4 min _i_ ) _._
**e** v _,j_ _∈_ Y _i_ _[d]_ [(] **[e]** [v] _[,j]_ [)] _[ ≤]_ [64] _[ d]_ [(][v] _[,][ S]_ [(1)]


26


For part(ii), observe that any vertex eliminated at any stage must have satisfied PCOUNTs(v) _>_
_⌊m_ win _/_ 2 _⌋_ for some s, which by Lemma B.3 implies POSs(v _,_ V _i_ ) _≤|_ V _i|/_ (100 _m_ S1). Hence every eliminated vertex lies within the lowest-ranked _|_ V _i|/_ (100 _m_ S1) for some s, so in total at most
_|_ V _i|/_ 100 vertices can be eliminated. Thus, _|_ V _i_ _[′][|]_ _[≥]_ [(3] _[/]_ [5)] _[|]_ [V] _[i][|]_ [,] [which] [immediately] [implies] [that]
_|_ V _i_ _[′′][| ≥]_ _[|]_ [V] 4 _[|]_ [.]


Using the results of Lemmas C.3, C.4 along with the analysis in Section B.3.2 for general metrics,
we conclude that [�] v _∈_ V _[d]_ [(][v] _[,][ M]_ [(][v][))] [=] _[ O]_ [(1)] _[ ·]_ [ OPT][1] _k_ [(][V][)][, with high probability.]


**Query** **Complexity.** In step 2 of the algorithm, PROBSORT(X _i_ ) uses _O_ (max _{k_ [2] _, n}_ polylog _n_ )
quadruplet queries since _|_ X _i|_ = _O_ ( _k_ [2] polylog _n_ ). For every s _∈Si_ [(1)], _|_ KERNEL _i_ (s) _|,_ GUARD(s) =
_O_ (polylog _n_ ). Hence, in step 4 the algorithm calls _O_ ( _k_ [2] polylog _n_ ) queries to the quadruplet oracle to compute PCOUNTs(s _[′]_ ) for every pair s = s _[′]_ _∈Si_ [(1)] _×_ _Si_ [(1)] . In step 6(a),

- _j_ =1 _,...,χi_ _[|][E]_ _i_ [(] _[j]_ [)] _|_ = _O_ ( _k_ [2] polylog _n_ ) so ADVSORT(E [(] _i_ _[j]_ [)][)] [with] [A][LG][-T][ESTER] [as] [the] [comparator]

use _O_ ( _k_ [2] polylog _n_ ) queries to the quadruplet oracle over all classes _Si_ [(1] _[,j]_ [)] . In step 6(b), for each
v _∈_ V _i_ _\Si_, the TRAVERSE(T _i_ [(] _[j]_ [)] _,_ v) (including the computations of PCOUNTs _h_ (v)) use _O_ (polylog _n_ )
queries, so in total step 6(b) executes _O_ ( _n_ polylog _n_ ) queries to the quadruplet oracle. In step 8,
_|_ Y _i|_ = _O_ ( _n_ polylog _n_ ), so the procedure ADVSORT(Y _i_ ) using ALG-TESTER as the comparator, runs
_O_ ( _n_ polylog _n_ ) queries to the quadruplet oracle. All other steps of the algorithm do not execute any
quadruplet oracle query. Overall, our algorithm calls the quadruplet oracle _O_ (( _n_ + _k_ [2] )polylog _n_ )
times. We conclude with Theorem 3.2.


D IMPROVING THE APPROXIMATION QUALITY


We now present a technique for improving the approximation ratio when the underlying metric Σ
has bounded doubling dimension, _ζ_ = _O_ (1), while still ensuring that the number of centers used
is Θ( _k_ polylog _n_ ). In fact, our new procedure also constructs a ( _k, ε_ )-coreset. We assume that
_ε ∈_ (0 _,_ 1) is an arbitrarily small constant.


D.1 ALG-DI


**Context.** Suppose we have run ALG-D(V) and obtained (C _, M_ ). Let _r_ ¯ be the total number of
rounds in ALG-D(V) and set _S_ [(1)] := [�] _i_ _[r]_ [¯] =1 _[−]_ [1] _[S]_ _i_ [(1)] and V _[′]_ := V _\_ C. By construction, _M_ (v) _∈S_ [(1)]
for every v _∈_ V _[′]_ and _M_ (v) = v for every v _∈_ C.


**Algorithm.** The algorithm proceeds as follows:


1. _Initialization._ Set Z _←∅_ .
2. _Per-center processing._ For each s _∈S_ [(1)] :


(a) _Assigned set._ Define
Us := _{_ v _∈_ V _[′]_ _| M_ (v) = s _}._
If s was chosen in round _i_ of ALG-D(V), then every v _∈_ Us satisfies _{_ s _,_ v _}_ _∈_ N _i_ . Let _π_ Us
be the order on Us induced by _π_ N _i_ .
(b) _Level sets._ For _t_ = 0 _,_ 1 _,_ 2 _, . . ., ⌈_ log _|_ Us _|⌉_ define

U _[t]_ s [:=] �v _∈_ Us �� RANK _π_ Us (v) _≥|_ Us _|/_ 2 _t_          - _._


Let
_t_ s := min� _t ≥_ 1 �� _|_ U _t_ s _[| ≤]_ _[c]_ [IMP] [log][3] _[ n]_                - _,_
where _c_ IMP is a sufficiently large constant that depends on _ε_ and _ζ_ .
(c) _Sampling._ For every _t_ _<_ _t_ s, sample with replacement a subset Ws _[t]_ _[⊆]_ [U] s _[t]_ [of] [size] _[|]_ [W] s _[t][|]_ [=]
_c_ IMP log [3] _n_ . Set


Ws := - _[t]_ - [s] _[−]_ [1]


- Ws _[t]_ - _∪_ U _[t]_ s [s] _[.]_

_t_ =0


27


(d) _Edge sets._ For every _t < t_ s, compute

Z _[t]_ s [:=] [E] �U _[t]_ s _[\]_ [ W][s] _[,]_ [W][s]             - _,_


and update
Z _←_ Z _∪_ Z _[t]_ s _[.]_

3. _Augment centers._ Let W = [�] s _∈S_ [(1)][ W][s][.] [Set][ C][+] [:=][ C] _[ ∪]_ [W][.] [For every][ v] _[ ∈]_ [C][+][, set] _[ M]_ [+][(][v][) :=][ v][.]

4. _Ordering._ Compute _π_ Z := PROBSORT(Z).

5. _Final mapping._ For each v _∈_ V _[′]_ :

(a) Let s := _M_ (v) (the old mapping). Define Ev := E(v _,_ Ws). Let _π_ Ev be the ordering of Ev
induced by _π_ Z.
(b) Let _{_ v _,_ w _[⋆]_ _}_ be the first edge of _π_ Ev ; note that w _[⋆]_ _∈_ Ws. Set _M_ [+] (v) := w _[⋆]_ .


D.2 ANALYSIS OF ALG-DI


First, it is straightforward that ALG-DI satisfies the isolation property, i.e., no quadruplet query is
ever repeated between ALG-D ALG-DI. We note that all quadruplet oracles executed in ALG-D
involved at least one vertex from C. However, in ALG-DI, after the execution of ALG-D, the set C
is removed from V _[′]_, so no quadruplet query will be repeated.


**Query** **Complexity.** From the analysis of ALG-D, we showed that we executed _O_ (( _n_ +
_k_ [2] )polylog _n_ ) queries to the quadruplet oracle. Then, in step (4) of ALG-DI we execute
PROBSORT(Z). The set Z contains _O_ ( _k_ polylog _n_ ) edges, so we execute _O_ ( _n_ polylog _n_ ) additional
queries to the quadruplet oracle. Overall, ALG-DI executes _O_ (( _n_ + _k_ [2] )polylog _n_ ) queries to the
quadruplet oracle.


D.2.1 FOCUSING ON A FIXED VERTEX


In the following, we focus on a fixed s _∈S_ [(1)] . Let _|_ Us _|_ = _m_ . Suppose [�] u _∈_ Us _[d]_ [(][s] _[,]_ [ u][)] [=] _[m][L]_ [ for a]
suitable _L ≥_ 0. This also implies that maxu _∈_ Us _d_ (s _,_ u) _≤_ _m L_ .


**Fixed buckets.** We know from the analysis in Section C.2 that _π_ Us is 4-sorted. Based on the order
_π_ Us, define a contiguous partition of Us into buckets _{Bt}_ _[b]_ _t_ =0 [,] [where] _[ b]_ [ is defined next,] [as follows.]
Let _δ_ = 100 _εc_ app [, where] _[ c]_ [app] [is the approximation ratio of A][LG][-D.] [For] _[ t]_ [=] [0] _[,]_ [ 1] _[,]_ [ 2] _[, . . .]_ [, let] _[ B][t]_ [be the]
next consecutive block in the order ending at the rightmost element whose distance from s is at most
2 _[t]_ _δL_ :
_B_ 0 = _{_ initial consecutive block up to the last u with _d_ (s _,_ u) _≤_ _δL},_


_B_ 1 = _{_ next block up to the last u with _d_ (s _,_ u) _≤_ 2 _δL},_ _. . ._

Since every u _∈_ Us satisfies _d_ (s _,_ u) _≤_ _mL_, the number of buckets is


  _b_ = _O_ log _[m][L]_

_δL_


= _O_ (log _[m]_


[=] _[O]_ [(log] _[n]_
_δ_ [)] _δ_


_δ_ [) =] _[ O]_ [(log] _[ n]_ [)] _[.]_


For the sake of analysis, consider the following recursive partitioning of _{Bt}_ _[b]_ _t_ =1 [(note] [that] [we]
exclude _B_ 0):


**Partitioning.** Let _i_ 0 := 0. For rounds _r_ = 1 _,_ 2 _, . . ._ do:


1. Compute _mr_ := [�] _t_ _[b]_ = _ir−_ 1+1 _[|][B][t][|]_ [.] [If] _[ m][r]_ _[≤]_ _[c]_ [IMP][ log][3] _[ n]_ [, then stop, else proceed.]

2. Define the _heavy threshold_ at round _r_ by _τr_ := 100 _m_ log _r_ _n_
3. Let _ir_ be the largest index _t_ _∈{ ir−_ 1 + 1 _, . . ., b }_ such that the bucket _Bt_ contains at least _τr_
vertices.

4. Remove the entire prefix of fixed buckets _{Bt}_ _[i]_ _t_ _[r]_ = _ir−_ 1+1


We now prove certain properties of the above partitioning. Let _r_ _[⋆]_ be the index of the last round.


28


**Lemma D.1.** _In every round r,_ _if mr_ _≥_ _c_ IMP log [3] _n,_ _there exists an index ir_ _∈{ir−_ 1 + 1 _, . . ., b}_
_such that |Bir_ _|_ _≥_ _τr_ = 100 _m_ log _r_ _n_ _[.]_ _[Furthermore,][ m][r]_ [+1] _[≤]_ 100 _mr_ _[and][ r][⋆]_ _[≤]_ [log] _[ n][.]_


_Proof._ Let _br_ := _b −_ _ir−_ 1 be the number of active buckets in round _r_ . If every active bucket was
strictly smaller than 100 _m_ log _r_ _n_ [, then]


100 _[,]_


_mr_ =


 - _|Bt|_ _<_ _br ·_ _mr_ _mr_ _[m][r]_

100 log _n_ _[≤]_ [log] _[ n][ ·]_ 100 log _n_ [=] 100
_t_ = _ir−_ 1+1


_b_


which is a contradiction. Hence, a heavy bucket exists; let _ir_ be the largest index with _|Bir_ _|_ _≥_
100 _m_ log _r_ _n_ [.] [The above argument also] [implies that all buckets to the right of] _[ i][r]_ [can have a total of at]
most _[m][r]_ [vertices.] [Thus,] _[ m][r]_ [+1] _[≤]_ _[m][r]_ [which directly implies that] _[ r][⋆]_ _[≤]_ [log] _[ n]_ [.]


100 _[m][r]_ [vertices.] [Thus,] _[ m][r]_ [+1] _[≤]_ 100 _[m][r]_


100 _[m][r]_ [which directly implies that] _[ r][⋆]_ _[≤]_ [log] _[ n]_ [.]


For any round _r_ _<_ _r_ _[⋆]_, mark an active bucket in round _r_ as _light_ if its cardinality is less than
_δmr/_ (1000 log [2] _n_ ). Let LIGHT _r_ := _{ t ∈{ir−_ 1 + 1 _, . . ., ir −_ 1 _} | |Bt| < δmr/_ (1000 log [2] _n_ ) _}_ be
the index set of light buckets in round _r_ .


**Lemma D.2.** _For any round r,_ [�]


  _t∈_ LIGHT _r_


v _∈Bt_ _[d]_ [(][s] _[,]_ [ v][)] _[≤]_ _[δ][ ·]_ [ �] v


v _∈Bir_ _[d]_ [(][s] _[,]_ [ v][)] _[.]_


_Proof._ Since _π_ Us is 4-sorted and the buckets _{Bt}_ are contiguous in the order, for any _t ∈_ LIGHT _r_,
for every v _[′]_ _∈_ _Bt_, and any v _∈_ _Bir_, _d_ (s _,_ v _[′]_ ) _≤_ 4 _d_ (s _,_ v).


Let ¯v = arg minv _∈Bir_ _d_ (v _,_ s). Therefore,


 

_t∈_ LIGHT _r_


- _d_ (s _,_ u) _≤_ - 

u _∈Bt_ _t∈_ L


 - _|Bt|_ - _·_ 4 _d_ (¯v _,_ s) _≤_ _d_ (¯v _,_ s) _·_ _bδmr_ _δmr_

250 log [2] _n_ _[≤]_ _[d]_ [(¯][v] _[,]_ [ s][)] _[ ·]_ 250 log _n_ _[.]_
_t∈_ LIGHT _r_


Furthermore, _d_ (¯v _,_ s) _·_ 100 log _mr_ _n_ _[≤]_ - _d_ (s _,_ v), so the result follows.

v _∈Bir_


**Lemma** **D.3.** _With_ _probability_ _at_ _least_ 1 _−_ _n_ _[−]_ [10] _,_ _for_ _every_ _non-light_ _bucket_ _B_ _(i.e.,_ _|B|_ _≥_
_δmr/_ (1000 log [2] _n_ ) _in_ _some_ _round_ _r,_ ALG-DI _uniformly_ _samples_ _at_ _least_ SIZE = _γ_ log _n_ _vertices_
_from B, where γ_ _is a suitable constant._


_Proof._ Consider a _non-light_ bucket _B_ in some round _r_ . Thus, it has at least 1000 log _δ mr_ [2] _n_ [vertices.]
Consider the level sets used by ALG-D with respect to s,

U _[t]_ s [=] _[{][ u][ ∈]_ [U][s] [:] [RANK] _[π]_ Us [(] _[u]_ [)] _[ ≥|]_ [U][s] _[|][/]_ [2] _[t][ }][.]_

Choose _tB_ = max _{ t_ : _B_ _⊆_ U _[t]_ s _[}]_ [.] [By maximality of] _[ t][B]_ [and the contiguous nature of the buckets,]
_|_ U _[t]_ s _[B]_ _[|]_ _[≤]_ [2] _[m][r]_ [.] [For level] _[ t][B]_ [the] [A][LG][-DI] [samples uniformly (with replacement) a set][ W] s _[t][B]_ of size
_c_ IMP log [3] _n_ uniformly from U _[t]_ s _[B]_ [.] [Let] _[ X]_ [be the number of sampled vertices that fall in] _[ B]_ [.] [Then]


2000 [log] _[ n.]_


E[ _X_ ] = _c_ IMP log [3] _n ·_ _[|][B][|]_

_|_ U _[t]_ s _[B]_ _[|]_ _[≥]_ _[c]_ [IMP][ log][3] _[ n][ ·]_


_δmr_
1000 log [2] _n_


[2] _n_

= _[δc]_ [IMP]
2 _mr_ 2000


By a Chernoff bound, for sufficiently large _c_ IMP, Pr[ _X_ _<_ SIZE = _γ_ log _n_ ] _≤_ _n_ _[−]_ [12] . Taking a
union bound over all _O_ (log [2] _n_ ) possible non-light buckets across _r_ _[⋆]_ = _O_ (log _n_ ) rounds proves the
claim.


**Lemma D.4.** _Conditioned on the correctness of_ PROBSORT(Z) _(so that all induced orders used be-_
_low have maximum dislocation D), there is a choice of absolute constants such that, with probability_
_at least_ 1 _−_ _n_ _[−]_ [8] _, for every non-light bucket B in any round r,_

     - [+]     


- _d_ �v _, M_ [+] (v)� _≤_ _δ ·_ 

v _∈B_ v _∈B_


_d_ (s _,_ v) _._

v _∈B_


29


_Proof._ Fix a non-light bucket _B_ in some round _r_ . By the bucket construction and the 4-sorted
property of _π_ Us, there exists a scale _α >_ 0 such that


_α_ _≤_ _d_ (s _,_ v) _≤_ 4 _α_ for every v _∈_ _B._


Since the doubling dimension is _ζ_, it is known that _B_ can be covered by a set of _m_ = Θ( _δ_ _[−][ζ]_ )

                               balls of diameter at most ( _δα_ ) _/_ 10 whose centers lie in _B_ . Consider a collection of such balls and
partition _B_ in _m_ _components_, breaking ties arbitrarily if two points lie in the same ball. Call a

      component _light_ if it contains fewer than _|B|/m_ [2] vertices, and _heavy_ otherwise. Then the union of
                          light components contains at most (1 _/m_ ) _|B|_ vertices.
                      
_Contribution_ _of_ _heavy_ _components._ Fix a heavy component _B_ _[′]_ _⊆_ _B_ . By Lemma D.3 (applied to
the present round _r_ and bucket _B_ ), ALG-DI draws at least SIZE = _γ_ log _n_ independent uniform
samples from _B_ (with replacement), where _γ_ _>_ 0 is a sufficiently large absolute constant. Let _X_ be
the number of samples that land in _B_ _[′]_ . Since it contains at least _|B|/m_ [2] vertices,
                                       


SIZE

= _[γ]_ [ log] _[ n]_
_m_ [2] _m_ [2]

- 


SIZE
E[ _X_ ] _≥_


_._
_m_ [2]


Assume _γ_ is large enough (it depends on _ε, ζ_ ) so that E[ _X_ ] _≥_ 100 max _{_ log _n, D}_ . Then by a
Chernoff bound, Pr� _X_ _≤D_ - _≤_ _n_ _[−]_ [12] . By a union bound over all (at most _m_ ) heavy components,

                            with probability at least 1 _−_ _n_ _[−]_ [11], each heavy component contains at least _D_ samples.


Now fix a v inside _B_ _[′]_ . Recall that Ws _⊆_ Us denotes the collection of all vertices including samples
ALG-DI computes while processing s. Note that Ws _⊆_ C [+] . Since all vertices in _B_ _[′]_ are within
distance _δα/_ 10, for every vertex w _∈_ Ws _∩_ _B_ _[′]_ we have

_d_ (v _,_ w) _≤_ _[δα]_

10 _[.]_


By the above arguments, _|_ Ws _∩_ _B| > D_ .

Since the correctness of PROBSORT(Z) implies that the induced order _π_ Ev (the restriction of _π_ Z to
E(v _,_ W)) has maximum dislocation _D_, the first edge _{_ v _,_ w _[⋆]_ _}_ in _π_ Ev, where w _[⋆]_ _∈_ W _⊇_ Ws must
satisfy

_d_ (v _,_ w _[⋆]_ ) = _d_ (v _, M_ [+] (v)) _≤_ _[δα]_

10


Summing over all vertices in all heavy components of _B_ contributes at most ( _δ/_ 10) _· α · |B|_ .


_Contribution of light components._ The union of light contains at most (1 _/m_ ) _|B|_ vertices. For any
                                          v in a light component and any sampled w _∈_ Ws _∩_ _B_, the triangle inequality and the bucket scale
imply
_d_ (v _,_ w) _≤_ _d_ (v _,_ s) + _d_ (s _,_ w) _≤_ 4 _α_ + 4 _α_ = 8 _α._

Since Lemma D.3 guarantees _|_ W _∩_ _B|_ _≥_ SIZE = _γ_ log _n_ _≥D_, the first edge in _π_ Ev has length at
most 8 _α_ . Therefore, vertices in light components within _B_ contribute at most (1 _/m_ ) _· |B| ·_ 8 _α_ .
                                              
_Putting everything together._ Adding contributions of light and heavy components of _B_, we get that

    - [+]     - _δ_ [8]     


- _d_ �v _, M_ [+] (v)� _≤_ - _δ_ [8]

10 [+] _m_
v _∈B_ 


_m_


_|B| α._


Since _δ_ _<_ 1, _ζ_ _≥_ 1 and _m_ - = Θ( _δ_ _[−][ζ]_ ), we have that _m_ - [8] _[≤]_ 10 _δ_ [, and the result follows.] [The probability]

bound 1 _−_ _n_ _[−]_ [8] follows from a union bound over all _O_ (log [2] _n_ ) non-light buckets of s.


**Lemma D.5.** _Fix_ s _∈S_ [(1)] _._ _Conditioned on the correctness of_ PROBSORT(Z) _,_ _with probability at_
_least_ 1 _−_ _n_ _[−]_ [8] _,_

    - [+] _ε_     


- _d_ �v _, M_ [+] (v)� _≤_ _ε_ _·_ 
_c_ app
v _∈_ Us v _∈_ Us


_d_ (s _,_ v) _._

v _∈_ Us


_Proof._ We argue for a _single_ round _r_ and then sum over rounds. Let _ir_ be the index of the rightmost
heavy bucket in that round. Henceforth we assume Lemma D.2 and Lemma D.4 holds.


30


By Lemma D.2,

     

_t∈_ LIGHT _r_


- _d_ (s _,_ v) _≤_ _δ ·_ 

v _∈Bt_ v _∈B_


_d_ (s _,_ v) _._

v _∈Bir_


By Lemma D.4 gives, w.h.p., for each _non-light_ bucket _B_ in every round,

     - [+]     


- _d_ �v _, M_ [+] (v)� _≤_ _δ ·_ 

v _∈B_ v _∈B_


_d_ (s _,_ v) _._

v _∈B_


Adding light and non-light contributions within round _r_,


_ir_

 

_t_ = _ir−_ 1+1


- _d_ �v _, M_ [+] (v)� _≤_ 2 _δ ·_ 

v _∈Bt_ v _∈B_


_d_ (s _,_ v) _._

v _∈Bir_


Since the buckets are disjoint across rounds, summing the above inequality over all rounds, yields

   - [+]   -   


 - _d_ (s _,_ v) _≤_ 2 _δ ·_ 

v _∈_ Us _\B_ 0 v _∈_ Us


 - _d_ �v _, M_ [+] (v)� _≤_ 2 _δ ·_ 

v _∈_ Us _\B_ 0 v _∈_ Us _\_


_d_ (s _,_ v) _._

v _∈_ Us


[�] _d_ (v _, M_ [+] (v)) = 
v _∈B_ 0 v _∈B_


[�] _d_ (v _,_ s). Combining the

v _∈_ Us


By design for the bucket _B_ 0,

[�]


- _d_ (v _,_ s) _≤_ _δ_ _·_ [�]

v _∈B_ 0 v _∈_ U


above, we get that,


- _d_ �v _, M_ [+] (v)� _≤_ 3 _δ ·_ 

v _∈_ Us v _∈_ U


_d_ (s _,_ v) _._

v _∈_ Us


Since _δ_ = 1001 _[·]_ _c_ app _ε_ _[,]_ [ the stated bound holds.]


D.2.2 PUTTING EVERYTHING TOGETHER


**Corollary D.6.** _Conditioned on the correctness of_ PROBSORT(Z) _, with probability at least_ 1 _−_ _n_ _[−]_ [7]
_the refined mapping M_ [+] _satisfies_

     - _d_ �v _, M_ [+] (v)� _≤_ _ε ·_ OPT [1] _k_ [(][V][)] _[.]_

v _∈_ V


_Proof._ Fix the event under which Lemma D.5 holds for _every_ s _∈S_ [(1)] . Since each instance holds
with probability at least 1 _−_ _n_ _[−]_ [8] and _|S_ [(1)] _| ≤_ _n_, a union bound gives overall success probability at
least 1 _−_ _n_ _[−]_ [7] . Summing Lemma D.5 over all s _∈S_ [(1)] yields

- [+] - - [+] _ε_ - - _ε_ 


- _d_ �v _, M_ [+] (v)� = 

v _∈_ V _\_ C s _∈S_


_d_ (v _, M_ (v)) _._

v _∈_ V _\_ C


s _∈S_ [(1)]


- _d_ �v _, M_ [+] (v)� _≤_ _ε_ _·_ 
_c_ app
v _∈_ Us s _∈S_


s _∈S_ [(1)]


- _d_ (s _,_ v) = _ε_ _·_ 
_c_ app
v _∈_ Us v _∈_ V _\_


We also know that


- _d_ �v _, M_ [+] (v)� = 

v _∈_ C v _∈_ C


_d_ �v _, M_ (v)� = 0

v _∈_ C


Since _M_ is a _c_ app–approximation mapping (from ALG-D), we have

     - _d_ (v _, M_ (v)) _≤_ _c_ app _·_ OPT [1] _k_ [(][V][)] _[.]_

v _∈_ V


Putting everything together gives

   - [+]


- _d_ (v _, M_ (v)) _≤_ _ε ·_ OPT [1] _k_ [(][V][)] _[,]_

v _∈_ V


- _d_ �v _, M_ [+] (v)� _≤_ _ε_ _·_ 
_c_ app
v _∈_ V v _∈_ V


as claimed.


We conclude with Theorem 3.3.


31


E ADDITIONAL RELATED WORK


Clustering is a classical problem that has been studied for decades Lloyd (1982); Charikar et al.
(1999); Har-Peled & Mazumdar (2004); Chen (2009). For _k_ -median and _k_ -means, Charikar et al.
(1999) presented the first constant-factor polynomial-time approximation, and subsequent work
Mettu & Plaxton (2002) improved the results. The notion of _coresets_ (compact summaries for scalable clustering) for _k_ -median and _k_ -means clustering was introduced in Har-Peled & Mazumdar
(2004) and later advanced by Chen (2009). More recently, results of Braverman et al. (2021; 2022)
have improved the sizes of the obtained coresets for clustering.


In another line of work, Xu et al. (2024) studied a weak-strong model in doubling metrics where
the weak oracle returns values within a multiplicative factor _C_ _>_ 1 of the true distance, while the
strong oracle provides exact distances. Their focus, however, was on building data structures for
approximate nearest neighbor search. Finally, Silwal et al. (2023) examined correlation clustering
within a weak-strong framework, but their techniques and oracle definitions do not extend to the
_k_ -center/median/means setting.


Much of the oracle-based clustering literature has focused on (faulty or exact) cluster
queries Mazumdar & Saha (2017a); Huleihel et al. (2019); Mazumdar & Saha (2017b); Choudhury
et al. (2019); Green Larsen et al. (2020); Galhotra et al. (2021), which directly identify groundtruth clusters. For _k_ -means Ashtiani et al. (2016); Chien et al. (2018); Kim & Ghosh (2017a;b);
Bianchi & Penna (2021), and _k_ -median Ailon et al. (2018), such queries, often combined with distance information, have led to stronger approximation guarantees. This is also is closely related
to learning-augmented algorithms Fu et al. (2025); Dong et al. (2025); Braverman et al. (2025b;
2024); Grigorescu et al. (2022); Ergun et al. (2022); Mitzenmacher & Vassilvitskii (2022); Indyk
et al. (2019); Hsu et al. (2019).


Beyond clustering, distance-based comparison oracles have been applied to a wide range of
problems. Examples include learning fairness metrics Ilvento (2020), hierarchical clustering Emamjomeh-Zadeh & Kempe (2018); Chatziafratis et al. (2018); Ghoshdastidar et al. (2019),
correlation clustering Ukkonen (2017), classification Tamuz et al. (2011); Hopkins et al. (2020),
and knowledge and data engineering Beretta et al. (2023). They have also been leveraged in tasks
such as finding the maximum element Guo et al. (2012); Venetis et al. (2012), top- _k_ selection Klein
et al. (2011); Polychronopoulos et al. (2013); Ciceri et al. (2015); Davidson et al. (2014); Kou et al.
(2017); Dushkin & Milo (2018), information retrieval Kazemi et al. (2018), and skyline computation Verdugo.


Finally, in relational clustering, the objective is to cluster the output of a join (or conjunctive) query.
Since the number of join results can be extremely large, materializing them explicitly often leads to
prohibitively slow algorithms. To address this, recent works Agarwal et al. (2024); Esmailpour &
Sintos (2024); Chen et al. (2022); Moseley et al. (2021); Curtin et al. (2020); Surianarayanan et al.
(2025) introduce relational oracles that provide summary statistics of the join output together with
access to selected key tuples, enabling clustering to be performed more efficiently. This framework,
however, differs from the R-model, where the main challenge lies in evaluating distances between
items through noisy comparisons. In relational clustering, by contrast, the difficulty arises primarily from the combinatorial explosion of join results rather than from the complexity of distance
evaluation.


USE OF LARGE LANGUAGE MODELS


Large Language Models (LLMs) were used as general-purpose assist tools. Specifically, we drafted
all text ourselves, and in several places (mainly in the introduction) we provided our own written paragraphs to ChatGPT-5 Plus with the instruction to “polish the text.” In addition, we used
ChatGPT-5 Plus in a limited way to help translate some of our pseudocode into Python code for basic prototyping. No part of the research process, including problem formulation, algorithm design,
theoretical development, or experimental analysis, relied on LLMs. All ideas, results, and scientific
contributions are entirely our own.


32