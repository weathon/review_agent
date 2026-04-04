## DISTRIBUTIONAL EQUIVALENCE IN LINEAR NON- GAUSSIAN LATENT-VARIABLE CYCLIC CAUSAL MOD- ELS: CHARACTERIZATION AND LEARNING


**Haoyue Dai** [1] **Immanuel Albrecht** [2] **Peter Spirtes** [1] **Kun Zhang** [1] _[,]_ [3]

1Carnegie Mellon University 2FernUniversität in Hagen 3MBZUAI


ABSTRACT


Causal discovery with latent variables is a fundamental task. Yet most existing
methods rely on strong structural assumptions, such as enforcing specific indicator
patterns for latents or restricting how they can interact with others. We argue that
a core obstacle to a general, structural-assumption-free approach is the lack of an
equivalence characterization: without knowing _what_ can be identified, one generally cannot design methods for _how_ to identify it. In this work, we aim to close this
gap for linear non-Gaussian models. We establish the graphical criterion for when
two graphs with arbitrary latent structure and cycles are _distributionally equivalent_,
that is, they induce the same observed distribution set. Key to our approach is
a new tool, _edge rank_ constraints, which fills a missing piece in the toolbox for
latent-variable causal discovery in even broader settings. We further provide a
procedure to traverse the whole equivalence class and develop an algorithm to
recover models from data up to such equivalence. To our knowledge, this is the first
equivalence characterization with latent variables in any parametric setting without
structural assumptions, and hence the first structural-assumption-free discovery
method. [Code and an interactive demo are available at https://equiv.cc.](https://equiv.cc)


1 INTRODUCTION


At the core of scientific inquiry lies causal discovery, the task of learning causal relations from
observational data (Spirtes et al., 2000; Pearl, 2009). In many real-world scenarios, the variables of
interest can be unobserved. For instance, in psychology, personality traits are hidden behind survey
responses, and in biology, crucial regulators may be unobserved due to technical inaccessibility.
Discovering the causal structure with these latent variables, referred to as latent-variable causal
discovery, is essential for understanding and reasoning, yet remains a challenging task.


Latent-variable causal discovery has seen significant development over the past three decades. A
milestone was the Fast Causal Inference (FCI) algorithm (Spirtes, 1992), which exploits conditional
independence (CI) constraints under hidden confounding. However, FCI is typically not regarded as
a method of latent-variable causal discovery, as it focuses solely on causal relations among observed
variables, with no intension or capability to identify those among latent variables. In fact, though FCI
is already maximally informative under nonparametric CI constraints (Richardson & Spirtes, 2002;
Zhang, 2008a), it is still not informative enough for recovering latent structure.


This limitation has motivated the development of many recent approaches that go beyond CI constraints, typically by introducing parametric assumptions, such as linearity (Silva et al., 2003; Dong
et al., 2024), non-Gaussianity (Hoyer et al., 2008; Jin et al., 2024), mixture models (Kivva et al., 2021),
and distribution shifts (Zhang et al., 2024). Within each setting, a rich array of techniques has emerged.
For example, in the linear non-Gaussian setting alone, methods have been developed based on overcomplete independent component analysis (OICA) (Salehkaleybar et al., 2020), regression (Tashiro
et al., 2014), Bayesian estimation (Shimizu & Bollen, 2014), independence testing (Xie et al., 2020),
cumulants (Robeva & Seby, 2021), independent subspace analysis (Dai et al., 2024), and many more.


However, despite this prosperity, most methods share a clear limitation: they rely on structural
assumptions, often about how latent variables are indicated and how they can interact with others.
Common examples include measurement models where observed variables have to be pure measure

1


ments of latents (Silva & Scheines, 2004; Zhang et al., 2018); hierarchical models that prohibit effects
from observed variables (Choi et al., 2011; Huang et al., 2020); sufficient number of pure children per
latent (Squires et al., 2022; Jin et al., 2024); and assumptions like triangle- or bow-freeness (Dong
et al., 2024; Wang & Drton, 2023). In addition, most methods also assume acyclicity, even though
feedback loops are common in real systems. These assumptions, often overly strong and untestable,
not only limit applicability but also complicate method selection for practitioners.


A pressing question naturally arises: after decades of progress, is it possible now to have a general
structural-assumption-free approach for latent-variable causal discovery that, like FCI, allows arbitrary
relations among latent and observed variables, yet goes beyond FCI’s limited informativeness?


One core obstacle, we argue, is the lack of a general equivalence characterization with latent variables.
Equivalence is a notion fundamental to causal discovery: when different causal models induce the
same observed distribution set (known as _distributional_ _equivalence_ ), no method can, or should,
distinguish among them, without extra information like interventions or sparsity constraints. The
expected discovery output is thus the entire _equivalence class_, the best one can hope to identify from
data. In practice, equivalence can also be defined more coarsely, depending on the specific constraints
used. One example is _Markov equivalence_, capturing when models entail the same CI constraints. A
well-known and nice result is that in causally sufficient, acyclic, and nonparametric models, Markov
equivalence coincides with distributional equivalence (Spirtes et al., 2000); the resulting equivalence
class is represented by a completed partially directed acyclic graph (CPDAG).


In the presence of cycles or latent variables, however, equivalence characterization becomes more
complex. For example, the nice coincidence between Markov and distributional equivalences breaks
down, even with only cycles (Spirtes, 1994; Mooij & Claassen, 2020), or only latent variables (Verma
& Pearl, 1991; Richardson et al., 2023), let alone both. The resulting equivalence classes, be it
Markov (Richardson & Spirtes, 2002; Claassen & Mooij, 2023) or distributional (Nowzohour et al.,
2017; Evans, 2018), also become far more complex. Such complications carry over to parametric
settings: for cycles alone, distributional equivalence has been characterized in linear non-Gaussian
and Gaussian models (Lacerda et al., 2008; Ghassami et al., 2020); yet for latent variables, no
characterization of any kind, whether distributional or constraint specific, is currently known to us.
The closest result (Adams et al., 2021) gives conditions for when a linear non-Gaussian acyclic model
can be uniquely identified, but leaves open describing the equivalence when such identifiability fails.


All such complications from latent variables and cycles have so far prevented a general equivalence
characterization, which is exactly what obstruct progress towards a structural-assumption-free method.
The need for such a characterization is yet clear: without knowing _what_ can be identified, one generally cannot design methods for _how_ to identify it. This is echoed in history: PC algorithm followed
CPDAGs; FCI’s guarantee followed maximal ancestral graphs (MAGs) (Richardson & Spirtes, 2002).


Our goal in this work is hence to overcome these challenges and establish a general equivalence
notion with latent variables and cycles. We focus on linear non-Gaussian models, a parametric setting
that has received much attention. Under this setting, we address three questions: **1)** When are two
graphs with arbitrary latent variables and cycles equivalent? **2)** How can one traverse the entire
equivalence class? **3)** How can one recover latent-variable models up to equivalence from data?


Centered around these three questions, our contributions are summarized as follows:


1. We present a general equivalence notion that allows arbitrary latent structure and cycles in linear
non-Gaussian models. This is the first such result known to us in any parametric setting (§2).
2. We introduce a new tool, _edge rank_ constraints. It contributes a missing piece to the broader
toolbox for latent-variable causal discovery, with potential use across many settings (§3).
3. We characterize equivalence graphically and provide procedures to traverse the entire class. Results are cleaner than expected. [We provide an interactive demo at https://equiv.cc (§4).](https://equiv.cc)
4. We develop an efficient algorithm to recover the equivalence class from data, which is, to our
knowledge, the first structural-assumption-free method for latent-variable causal discovery (§5).


2 PROBLEM SETUP


In this section, we lay the groundwork for our study. In §2.1, we define the notion of distributional
equivalence in linear non-Gaussian latent-variable causal models. Then in §2.2, we introduce the
idea of irreducibility to rule out trivial cases, clearing the way for the main results to come.


2


2.1 PRELIMINARIES FOR LINEAR NON-GAUSSIAN MODELS


**Notations** **on** **matrices.** For a matrix _M_, we let _Mi,j_ be its ( _i, j_ )-th entry. For two index sets
_A, B_, we let _MA,B_ = ( _Ma,b_ ) _a∈A,b∈B_ be the submatrix of _M_ with rows indexed by _A_ and columns
indexed by _B_ . We let _MA,_ : be the rows in _M_ indexed by _A_, and similarly _M_ : _,B_ for the columns. For
a finite set _A_, we denote its cardinality by _|A|_ . We denote by Scale( _d_ ) the set of all _d × d_ diagonal
matrices with nonzero diagonal entries, and by Perm( _d_ ) the set of all _d × d_ permutation matrices.
For a permutation _π_ : _V_ _→_ _V_ on a ground set _V_, we denote _π_ ( _F_ ) := _{π_ ( _i_ ) : _i_ _∈_ _F_ _}_ for any set
_F_ _⊆_ _V_, and extend this notation to families of sets by _π_ ( _F_ ) := _{π_ ( _F_ ) : _F_ _∈F}_ for _F_ _⊂_ 2 _[V]_ .

**Notations on graphs.** Throughout, by a _digraph_ we refer to a directed graph that may contain
cycles but no self-loops (edges from a vertex to itself). In a digraph _G_, let _V_ ( _G_ ) be its vertex set. For
vertices _a, b_, we say _a_ is a _parent_ of _b_ and _b_ is a _child_ of _a_, denoted by _a_ pa _G_ ( _b_ ) and _b_ ch _G_ ( _a_ ),
_∈_ _∈_
when _a_ _→_ _b_ is an edge in _G_, written _a_ _→_ _b_ _∈G_ ; _a_ is an _ancestor_ of _b_ and _b_ is a _descendant_ of _a_,
denoted by _a_ an _G_ ( _b_ ) and _b_ de _G_ ( _a_ ), when _a_ = _b_ or there is a directed path _a_ _b_ in .
_∈_ _∈_ _→· · ·_ _→_ _G_
These notations extend to sets: e.g., for a vertex set _A_, an _G_ ( _A_ ) := [�] _a∈A_ [an] _[G]_ [(] _[a]_ [)][.]

**Linear non-Gaussian (LiNG) causal models.** We consider a _linear non-Gaussian model_ associated
with a digraph, in which random variables _V_ = ( _V_ 1 _,_ _, V|V |_ ) _[⊤]_, corresponding to the vertices of
_G_ _· · ·_
_G_, are generated according to the structural equation:
_V_ = _BV_ + _E,_ (1)
where _E_ = ( _E_ 1 _,_ _, E|V |_ ) _[⊤]_ consists of mutually independent, non-constant, non-Gaussian exoge_· · ·_
nous noise terms. The matrix _B_ _∈B_ ( _G_ ) is a weighted _adjacency matrix_ (whose entries represent
_direct causal effects_ ) that follows _G_, where _B_ ( _G_ ), all adjacency matrices that _follow G_, is defined as:

( ) := _B_ R _[|][V][ |×|][V][ |]_ : _BVj_ _,Vi_ = 0 = _Vi_ _Vj_ _._ (2)
_B_ _G_ _{_ _∈_ _⇒_ _→_ _∈G}_
Assuming _I_ _−_ _B_ is invertible, solving for Equation (1) gives an equivalent mixing form:
_V_ = ( _I_ _−_ _B_ ) _[−]_ [1] _E_ =: _AE,_ (3)
where _A_ is called the weighted _mixing matrix_ . The entry _AVj_ _,Vi_ represents the _total causal effect_
from _Vi_ to _Vj_ . All mixing matrices that follow, denoted by ( ), is defined as:
_G_ _A_ _G_

_A_ ( _G_ ) := _{_ ( _I_ _−_ _B_ ) _[−]_ [1] : _B_ _∈B_ ( _G_ ) _,_ _I −_ _B_ invertible _}._ (4)


**Latent-variable LiNG models.** Let the vertices _V_ of a digraph _G_ be partitioned as _V_ = _L ∪_ _X_,
where _L_ denotes _latent_ (unobserved) variables and _X_ denotes _observed_ variables. A _latent-variable_
_model_ is specified by the tuple ( _G, X_ ), with latent variables _L_ omitted when clear from context.

Given a full mixing matrix _A_ ( ), the submatrix _AX,_ : R _[|][X][|×|][V][ |]_ maps exogenous noise terms
_∈A_ _G_ _∈_
to the observed variables. The collection of such wide rectangular mixing matrices is defined as:
( _, X_ ) := _AX,_ : : _A_ ( ) _._ (5)
_A_ _G_ _{_ _∈A_ _G_ _}_
Accordingly, the induced _observed distribution set_ of _G_ on _X_, that is, the set of all distributions over
_X_ that can arise from a LiNG model over ( _G, X_ ), denoted _P_ ( _G, X_ ), is given by:
_P_ ( _G, X_ ) := _{p_ ( _X_ ) : _X_ = _AE,_ _A ∈A_ ( _G, X_ ) _,_ _E_ _∈_ NG( _|V |_ ) _},_ (6)
where _p_ ( _X_ ) denotes the probability distribution of the random vector _X_, and NG( _d_ ) denotes the set
of all _d_ -dim random vectors with mutually independent, non-constant, and non-Gaussian components.


We are now ready to formalize the central notion of this work: distributional equivalence.
**Definition 1** ( **Distributional equivalence** ) **.** Let _G_ and _H_ be two digraphs with possibly different
vertices, and _X_ _⊆_ _V_ ( _G_ ) _∩V_ ( _H_ ) be the shared observed variables. We say _X_ _G_ and _H_ are _distributionally_
_equivalent_ (or for short, _equivalent_ ) on _X_, denoted by _G_ _∼H_, when _P_ ( _G, X_ ) = _P_ ( _H, X_ ).

The equivalence (Definition 1) captures when two models yield identical observed distribution set,
i.e., observationally indistinguishable. With this notion in place, next we clean up some trivialities.


2.2 IRREDUCIBILITY: TO FIRST RULE OUT TRIVIAL CASES OF EQUIVALENCE


To study identifiability, let us first see what is inherently non-identifiable. For instance, one can
freely add latent vertices that are not ancestors of any observed variables _X_ to a digraph _G_ without
affecting _P_ ( _G, X_ ), yielding trivially equivalent models. Identifying those latents is both impossible
and meaningless. To rule out such trivialities, we introduce the notion of _irreducibility_ .


3


|L<br>2|Col2|
|---|---|
|||


|L L L<br>1 2 3|Col2|Col3|
|---|---|---|
|_L_1|_L_3|_L_3|
||||


Figure 1: Examples of reducing models to their irreducible forms via the procedure in Proposition 2.
Throughout, white circles denote observed variables and grey squares denote latent variables.


**Definition 2** ( **Irreducibility** ) **.** We say a latent-variable model _X_ ( _G, X_ ) is _irreducible_, when there exists
no digraph _H_ with _|V_ ( _H_ ) _| < |V_ ( _G_ ) _|_ such that _G_ _∼H_ .

Irreducibility captures when an observed distribution set cannot arise from any other model with
fewer latent variables. We now present a simple graphical condition for this property.
**Proposition 1** ( **Graphical condition for irreducibility** ) **.** _A model_ ( _G, X_ ) _is irreducible, if and only_
_if for each non-empty set_ _**l**_ _L,_ ch _G_ ( _**l**_ ) _**l**_ 2 _, i.e., it has more than one child outside._
_⊆_ _|_ _\_ _| ≥_

Note that when is acyclic, it suffices to check each single _Li_ _L_, consistent with the condition previ_G_ _∈_
ously derived by Salehkaleybar et al. (2020). The proof of Proposition 1, along with others, is provided
in Appendix B. The key idea here is that any violation of the condition leads to proportional columns
in mixing matrices _A_ ( _G, X_ ), so that the observed distributions can be equivalently generated by a
smaller graph with these columns merged to one. Conversely, identifiability results of OICA (Eriksson
& Koivunen, 2004) suggest that as long as in the absence of such proportional columns, the mixing
matrix is identifiable up to column scaling and permutation, so the number of latents is identifiable.


We next provide an explicit procedure for reducing an arbitrary model to its irreducible form.

**Proposition 2** ( **Procedure of reduction to the irreducible form** ) **.** _Given any latent-variable model_
_X_
( _G, X_ ) _, the following procedure outputs a digraph H such that H_ _∼G_ _and_ ( _H, X_ ) _is irreducible._

_Step 1._ _Initialize H as G._
_Step 2._ _Remove vertices V_ ( ) an _H_ ( _X_ ) _from_ _, i.e., remove latents who have no effects on X._
_H_ _\_ _H_
_Step 3._ _Identify the maximal redundant latents in the remaining latent vertices:_

mrl := _**l**_ _V_ ( ) _X_ : _**l**_ _>_ 0 _,_ ch _H_ ( _**l**_ ) _**l**_ _<_ 2 _,_ _and_ _**l**_ _[′]_ ⊋ _**l**_ _,_ ch _H_ ( _**l**_ _**[′]**_ ) _**l**_ _**[′]**_ 2 _._ (7)
_{_ _⊆_ _H_ _\_ _|_ _|_ _|_ _\_ _|_ _∀_ _|_ _\_ _| ≥_ _}_

_Step 4._ _For each_ _**l**_ mrl _, let c be the exact child in_ ch _H_ ( _**l**_ ) _**l**_ _; for each parent p_ pa _H_ ( _**l**_ ) _**l**_ _c_ _,_
_∈_ _\_ _∈_ _\_ _\{_ _}_
_add an edge p →_ _c into H if not already present; finally, remove_ _**l**_ _vertices from H._


Illustrative examples of this reduction are shown in Figure 1. This reduction lets us, without loss of
generality, restrict attention to irreducible models for the remainder, as arbitrary models are equivalent
if and only if their irreducible forms are equivalent. Note that irreducibility is not a structural
assumption as discussed in §1, but rather a canonicalization to eliminate trivialities. As a side note,
applying the reduction in Proposition 2 does not increase the number of edges or cycles.


3 DEVELOPING GRAPHICAL TOOLS FOR CHARACTERIZING EQUIVALENCE


In the previous section, we defined distributional equivalence and irreducibility to rule out trivial
unidentifiable cases, so we can focus solely on irreducible models in what follows. Then, when are
two irreducible models equivalent? In this section, we tackle this question step by step.


Specifically, in §3.1 we first show that distributional equivalence reduces to an algebraic condition on
mixing matrices, and further to a graphical condition involving a concept familiar to the community:
_path_ _ranks_, given by max-flow-min-cuts in digraphs. Although familiar, path ranks are difficult
to work with due to their global, non-local nature, as we illustrate in §3.2. To overcome this, we
introduce a new tool: _edge ranks_, a local, edge-level constraint that complements path ranks and is
easier to manipulate. This new tool, developed in §3.3, not only enables our final result to come in
the next section, but also enriches the broader rank-based picture beyond our specific setting.


3.1 EQUIVALENCE VIA PATH RANKS


We start by examining the algebra behind distributional equivalence. By Definition 2, all equivalent
irreducible models must have the same number of latents. This follows from OICA, which guarantees


4


exact recovery of the number of (nontrivial) latent variables. Hence, in what follows, when considering
the equivalence of two irreducible models ( _G, X_ ) and ( _H, X_ ), we can, without loss of generality,
denote their latent variables by a same set of labels, so that _V_ ( _G_ ) = _V_ ( _H_ ) = _X_ _∪_ _L_ .

We then observe that distributional equivalence can be rephrased in terms of the mixing matrices:
two models are equivalent if and only if for every mixing matrix one model can generate, the other
can also generate a version of it up to column scaling and permutation, and vice versa, due to the
scaling and permutation closedness of exogenous noise terms. Formally,

**Lemma** **1** ( **Equivalence** **via** **mixing** **matrices** **closure** ) **.** _Two_ _irreducible_ _models_ _are_ _equivalent,_
_written G_ _∼HX_ _, if and only if A_ ( _G, X_ ) = _A_ ( _H, X_ ) _, where for a set of matrices A ⊆_ R _m×d, we let:_

_A_ := _{APD_ : _A ∈A,_ _P_ _∈_ Perm( _d_ ) _,_ _D_ _∈_ Scale( _d_ ) _},_ (8)
_that is, the closure of A up to column scaling and permutation._

Then, what are exactly these mixing matrices, namely, _A_ ( _G, X_ )? As defined in Equations (2) to (5),
it arises from a mapping over the free parameters in adjacency matrices. Concretely, each entry of the
mixing matrix is a rational function: the numerator polynomial reflects “total causal effects” between
variables, and the denominator polynomial accounts for “global cycle discounts”, which is simply 1
when the digraph is acyclic. In cyclic cases, there is a small pathological locus where denominators
vanish, that is, where _I_ _−_ _B_ becomes singular and cycles “cancel themselves.” But as we will show
in the proof, this does not affect our results. So for now, let us progress with the Zariski closure of
_A_ ( _G, X_ ), an algebraic variety that can be defined by finitely many _equality constraints_ .

We now study these constraints. One fundamental class of them is the so-called _rank constraints_,
which admits a nice graphical interpretation in terms of _max-flow-min-cut_ in digraphs, defined below:
**Definition 3** ( **Path ranks** ) **.** In a digraph _G_, for two sets of vertices _Z, Y_ _⊆_ _V_ ( _G_ ), the _path rank_
_ρG_ ( _Z, Y_ ) is defined as the maximum number of vertex-disjoint directed paths from _Y_ to _Z_ in .
_G_
By (Menger, 1927), this max-flow quantity can also be defined by its min-cut version:
_ρG_ ( _Z, Y_ ) := min [ensures no directed path from] _[ Y][ \]_ _**[c]**_ [ to] _[ Z][\]_ _**[c]**_ _[}][.]_ (9)
_**c**_ _⊆V_ ( _G_ )

_[{|]_ _**[c]**_ _[|]_ [ :] _**[ c]**_ [’s removal from] _[ G]_

These purely graphical quantities can be read off from the mixing matrices by examining the matrix
ranks of corresponding submatrices, which is the well-known (path) rank constraint:
**Lemma 2** ( **Path rank constraints in mixing matrices** ) **.** _In a digraph G, for any two sets of vertices_
_Z, Y_ _⊆_ _V_ ( _G_ ) _that need not be disjoint, the following equality holds for generic choice of A ∈A_ ( _G_ ) _:_
rank( _AZ,Y_ ) = _ρG_ ( _Z, Y_ ) _._ (10)

_Here,_ rank _denotes the usual matrix rank, and “generic” means the equality holds almost everywhere_
_except for a Lebesgue measure zero set where coincidental lower matrix ranks occur._


Rank constraints bridge algebra in matrices with geometry in digraphs. They were initially proved for
acyclic graphs only (Lindström, 1973; Gessel & Viennot, 1985), and later generalized by (Talaska,
2012). They are powerful: as we will show in the proof, rank constraints alone, together with a
column permutation, suffice to determine equivalence. We directly state the result below:

**Lemma 3** ( **Equivalence via path ranks** ) **.** _Two irreducible models are distributionally equivalent,_
_X_
_written G_ _∼H, if and only if there exists a permutation π over the vertices V_ ( _G_ ) _, such that_
_ρG_ ( _Z, Y_ ) = _ρH_ ( _Z, π_ ( _Y_ )) _for all Z_ _X_ _and_ _Y_ _V_ ( ) _._ (11)
_⊆_ _⊆_ _G_


From Lemma 1 to Lemma 3, so far we have arrived at a first purely graphical view of equivalence.


3.2 THE COMPLEXITY OF MANIPULATING PATH RANKS


In §3.1 we have arrived at Lemma 3, a purely graphical characterization of equivalence, which,
perhaps surprisingly, is expressed in terms of a familiar concept: path ranks. However, this is only a
start and far from operational: verifying it requires searching over all vertex permutations and all
( _Z, Y_ ) pairs, which quickly becomes intractable due to their factorial and exponential growth, let
alone the costly graph traversal required for each single path rank computation. As an analogy to the
acyclic, causally sufficient case, Lemma 3 is like saying “having all the same d-separations,” whereas
what we seek is something simpler and more local, like “same adjacencies and v-structures.”


5


_Z_ 1
_Z_ 2

...
_Zn_


0 0 _×_ _×_ 0 _· · ·_ 0







Figure 2: An illustration of path ranks, edge ranks, and their duality. Left: a digraph _G_ with vertices
_V_ partitioned to _Y_, _C_, and _Z_, shown by different colors. The path rank _ρG_ ( _Z, Y_ ) = 2, with _C_ being
a min-cut. Right: the dual edge rank _rG_ ( _V_ _Y,_ _V_ _Z_ ) = 4, given by the maximum bipartite matching
_\_ _\_
from _V \Z_ to _V \Y_, i.e., from _Y_ _∪_ _C_ to _Z_ _∪_ _C_, with four matched edges highlighted in red. Four
corresponding nonzero entries placed on diagonal, also in red, confirm mrank( _Q_ [(] _V_ _[G]_ _\_ [)] _Y,_ _V \Z_ [)] [=] [4][.]
One may examine the duality in Theorem 1: w.l.o.g. let _m ≤_ _n_, there is _m −_ 2 = _m_ + _n_ + 2 _−_ _n −_ 4.


Then, does a simpler local condition naturally follow from Lemma 3? Unfortunately, not quite. Path
ranks are hard to work with due to their global nature: they summarize the size of “bottlenecks”,
but say nothing about which paths are involved or how they interact. Each single edge may lie on
multiple bottlenecks, so even a small local alteration to a digraph may trigger unpredictable global
changes in path ranks. Conversely, with latent variables, seemingly very different digraphs can still
share the same path ranks. We illustrate such complexity with the following example.
**Example 1** ( **Complexity of viewing equivalence via path ranks** ) **.** Consider the digraph _G_ on the
left of Figure 2, with vertices partitioned into _Y_, _C_, and _Z_ . Obviously, the path rank _ρG_ ( _Z, Y_ ) = 2.
Now, suppose vertices _C_ 1 _, C_ 2 become latent and all others remain observed. What models are
_{_ _}_
equivalent? This is not obvious anymore. It usually takes some thought to realize that adding edges or
cycles within _C_, or removing one or two edges from _C_ to _Z_, still preserves path ranks as in Lemma 3.
What about the _Y_ to _C_ structure then? This is more subtle: when _n >_ 2, it must remain fixed; but
when _n_ = 2, _C_ is no longer a unique bottleneck, and suddenly, _Y_ can point freely to both _C_ and _Z_ .


Things become even less intuitive when other variables are latent. For example, with _m_ = _n_ = 4, if
_C_ 1 _, C_ 2 [are latent, there are 17 digraphs in the equivalence class (view them online).](https://equiv.cc/example1_C_latent) When _Y_ 1 _, Y_ 2
_{_ _}_ _{_ _}_
or _Y_ 1 _, C_ 1 [are latent, this number comes to 872 (view) and 1](https://equiv.cc/example1_Y_latent) _,_ [024 (view), respectively.](https://equiv.cc/example1_YC_latent) Note that all
_{_ _}_
this comes from a well structured digraph; arbitrary structures only lead to greater complexity. _△_

Example 1 illustrates the complexity of path ranks in inferring graph structures. In fact, this complexity is well recognized in literature: despite various techniques developed to estimate path ranks from
data (Dai et al., 2022; Sturma et al., 2024), and well-studied counterparts in the linear Gaussian (Sullivant et al., 2010) and discrete settings (Chen et al., 2024b) or even with selection bias (Dai et al.,
2025b), when it comes to structure learning from ranks, usually restrictive structural assumptions are
required to ensure clean interpretation to where and how these paths can be.


All observations above motivate a question: is there a more local, graph-manipulable alternative to
path ranks, not just for building equivalence in this work but also as a piece in the broader toolbox?
Interestingly, the answer is yes, and we develop such a tool next: edge ranks.


3.3 EDGE RANKS: A NEW TOOL IN THE RANK-BASED PICTURE


We now introduce a new tool: _edge ranks_ . As the name suggests, edge ranks directly operate on
edges in digraphs, which is more local and accessible in contrast to the paths used in path ranks. For
intuition, one may refer to Figure 2, which illustrates all the concepts and results below.


Let us first define edge ranks, similar to how we define path ranks previously in Definition 3:
**Definition 4** ( **Edge ranks** ) **.** In a digraph _G_, for two sets of vertices _Z, Y_ _⊆_ _V_ ( _G_ ), the _edge rank_
_rG_ ( _Z, Y_ ) is defined as the size of the maximum bipartite matching from _Y_ to _Z_ via edges in, where
_G_
self-matches ( _a_ to _a_ for _a ∈_ _Y_ _∩_ _Z_ ) are allowed. Edge ranks also admit a min-cut version:
_rG_ ( _Z, Y_ ) := min [there is no edge from] _[ Y][ \]_ _**[y]**_ [ to] _[ Z][\]_ _**[z]**_ [ in] _[ G}][.]_ (12)
_**z**_ _⊆Z,_ _**y**_ _⊆Y,_ _**z**_ _∪_ _**y**_ _⊇Z∩Y_

_[{|]_ _**[z]**_ _[|]_ [ +] _[ |]_ _**[y]**_ _[|]_ [ :]

In parallel to how path ranks correspond to matrix ranks of mixing submatrices (cf. Lemma 2), the
pure graphical quantities of edge ranks also have their algebraic counterpart. This time, it is not the
mixing matrices at play, but directly the adjacencies.


6


For clarity, let us introduce a new matrix notation, _Q_, in addition to the already familiar notations of
_B_ and _A_, and a new notion of _matching ranks_, in addition to the already familiar matrix ranks.
**Definition 5** ( **Support matrix** ) **.** For a digraph _G_, its binary _support matrix_ in shape _|V_ ( _G_ ) _| × |V_ ( _G_ ) _|_,
denoted _Q_ [(] _[G]_ [)], is given by:


_Q_ [(] _V_ _[G]_ _j_ [)] _,Vi_ [=] [‘] _[ ×]_ [ ’ if] _[ V][i]_ [=] _[ V][j]_ [or] _[ V][i]_ _[→]_ _[V][j]_ _[∈G][,]_ [and][ 0][ otherwise] _[.]_ (13)


**Definition 6** ( **Matching rank of a matrix** ) **.** The _matching rank_ of a matrix _M_ _∈_ K _[m][×][n]_ is given by:


mrank( _M_ ) := max _P ∈_ Perm( _n_ )


_i_ =1 _,···,_ min( _m,n_ ) [1] [((] _[MP]_ [)] _[i,i]_ (14)

[= 0)] _[.]_


In simple terms, the matching rank of a matrix, denoted mrank, is the maximum number of nonzero
entries that can be positioned on the diagonal by permuting its columns (or rows).


We can now give the edge rank constraints, as a counterpart to path rank constraints (cf. Lemma 2).
Unlike the algebraic efforts required there, this result follows immediately from definition:
**Lemma 4** ( **Edge rank constraints in support matrices** ) **.** _In a digraph G, for any two sets of vertices_
_Z, Y_ _⊆_ _V_ ( _G_ ) _that need not be disjoint, the following equality holds:_

mrank( _Q_ [(] _Z,Y_ _[G]_ [)] [)] [=] _[r][G]_ [(] _[Z, Y]_ [ )] _[.]_ (15)


So far, we have defined both path ranks and edge ranks, which at first glance appear so different:
graphically, one is global, focusing on paths, while the other is local, operating on edges; algebraically,
one is tied to weighted mixing matrices, the other to binary support matrices. However, despite these
apparent differences, a surprising and elegant duality exists between them:
**Theorem 1** ( **Duality between path ranks and edge ranks** ) **.** _In a digraph G_ _with vertices V, for any_
_two sets of vertices Z, Y_ _⊆_ _V_ _that need not be disjoint, the following equality holds:_

min( _Z_ _,_ _Y_ ) _ρG_ ( _Z, Y_ ) = _V_ max( _Z_ _,_ _Y_ ) _rG_ ( _V_ _Y,_ _V_ _Z_ ) _._ (16)
_|_ _|_ _|_ _|_ _−_ _|_ _| −_ _|_ _|_ _|_ _|_ _−_ _\_ _\_


This duality is powerful: it suggests that every statement phrased in terms of path ranks and its
variants, including the familiar _d-separation_ and _t-separation_, can be equivalently rephrased in terms
of edge ranks. It reveals that, despite the very different graphical objects involved in the two ranks,
they offer complementary perspectives on a same notion in the digraph, namely, _bottleneck_, which
captures how dependencies arise in observed data, and is thus central to causal discovery.


In fact, this duality has long been studied in the matroid community (K˝onig, 1931; Perfect, 1968;
Ingleton & Piff, 1973), while only the path rank side has been well known in causal discovery. We
thus introduce edge ranks here, filling the other side to the rank-based toolbox. It is not that edge
ranks are always better, but having both perspectives is beneficial. Within this work, edge ranks
indeed lead to simpler derivations. For instance, let us rephrase Lemma 3 using edge ranks below:

**Lemma 5** ( **Equivalence via edge ranks** ) **.** _Two irreducible models are distributionally equivalent,_
_X_
_written G_ _∼H, if and only if there exists a permutation π over the vertices V_ ( _G_ ) _, such that_

_rG_ ( _Z, Y_ ) = _rH_ ( _π_ ( _Z_ ) _, Y_ ) _for all Z, Y_ _V_ ( ) _with_ _L_ _Y._ (17)
_⊆_ _G_ _⊆_


As we will see in the next section, this formulation paves the way to our final criterion for equivalence.
To conclude this section, we provide a side-by-side comparison of two ranks (Table 1; Appendix C.1).


4 THE GRAPHICAL CHARACTERIZATION OF DISTRIBUTIONAL EQUIVALENCE


In previous sections, through a step-by-step breakdown of equivalence, we have arrived at a key result,
Lemma 5, which, notably, is framed by a new tool we introduced: edge ranks. Building on this foundation, in this section, we provide our final graphical criterion for distributional equivalence, and present
a transformational characterization that enables traversal of all digraphs in the equivalence class.


We first study the task of deciding whether two given models are equivalent. For this purpose,
although Lemma 5 offers a more local condition for each rank check, it still requires a large number
of total checks: one must go through all sets _Y_ _⊇_ _L_, which amounts to all subsets _**x**_ _⊆_ _X_ . As noted
in our earlier analogy (§3.2), this remains akin to “same d-separations,” instead of a practical criterion
like “same adjacencies and v-structures.” Then, does Lemma 5 yield such a practical criterion?


7


Fortunately, this time, the answer is yes. Unlike the complexities encountered with path ranks in §3.2,
edge ranks allow Lemma 5 to admit a nice local decomposition: instead of checking all subsets _**x**_ _⊆_ _X_,
it suffices to check each singleton _Xi_ _X_ independently. This yields our final graphical criterion:
_∈_


**Theorem 2** ( **Graphical criterion for distributional equivalence** ) **.** _In a digraph G, we define the_
_“children bases” of a vertex set Y_ _⊆_ _V_ ( _G_ ) _as vertex sets that admit perfect edge matchings from Y :_


bases _G_ ( _Y_ ) := _Z_ ch _G_ ( _Y_ ) _Y_ : _rG_ ( _Z, Y_ ) = _Z_ = _Y_ _._ (18)
_{_ _⊆_ _∪_ _|_ _|_ _|_ _|}_


_Then, two irreducible models_ ( _G, X_ ) _and_ ( _H, X_ ) _are distributionally equivalent, if and only if there_
_exists a permutation π over the vertices V_ ( _G_ ) _, such that the following conditions hold:_
�bases _G_ ( _L_ ) = _π_ (bases _H_ ( _L_ )) _,_ _and_
(19)
bases _G_ ( _L_ _Xi_ ) = _π_ (bases _H_ ( _L_ _Xi_ )) _for each Xi_ _X._
_∪{_ _}_ _∪{_ _}_ _∈_


To interpret this criterion, let us consider the causally sufficient case where _L_ = ∅. In this case, each
bases _G_ ( _Xi_ ) is just _Xi_ with its children. Then, Theorem 2 immediately reduces to the classical
_{_ _}_
result of exact digraph identification up to permutation (Lacerda et al., 2008). Interestingly, that result
has recently been revisited also from a bipartite matching view used here (Sharifian et al., 2025).


Having established Theorem 2 as an efficient criterion for determining equivalence, we now turn to
another task of traversing all digraphs in an equivalence class. For this purpose, however, a determining criterion alone offers little guidance. Again, we recall the analogy with Markov equivalence. Note
that except for the criterion of “same adjacencies and v-structures,” there is an alternative characterization: “two acyclic digraphs are equivalent if and only if one can reach the other via a sequence
of _covered edge reversals_,” known as “Meek conjecture” (Meek, 1997). Such a transformational
characterization offers a natural way for equivalence class traversal. In light of it, we next develop
such a transformational characterization, analogous to “Meek conjecture” for our setting.


We start with the permutation part in Theorem 2, which corresponds to row permutations to the
support matrix _Q_ [(] _[G]_ [)] . Such permutations must result in valid support matrices, i.e., ones with nonzero
diagonals. By cycle decomposition of permutations, this leads to an observation: disjoint cycles in
the digraph can be freely reversed without affecting equivalence. Formally:

**Lemma 6** ( **Admissible cycle reversals** ) **.** _For a digraph G, let C_ _be any collection of vertex-disjoint_
_simple cycles in_ _._ _Define a new digraph_ _where for each edge Vi_ _Vj_ _:_
_G_ _H_ _→_ _∈G_


_1._ _If Vi_ _Vj_ _is on a cycle in_ _, then include Vj_ _Vi_ _in_ _;_
_→_ _C_ _→_ _H_
_2._ _Otherwise, if Vj_ _is on a cycle in_ _with the predecessor Vk_ _Vj, then include Vi_ _Vk_ _in_ _;_
_C_ _→_ _→_ _H_
_3._ _Otherwise, simply include Vi_ _Vj_ _in_ _._
_→_ _H_


_X_
_Then, with this new H, the equivalence G_ _∼H still holds, for every X_ _⊆_ _V_ ( _G_ ) _._


This result was also shown by (Lacerda et al., 2008). It highlights that in the linear non-Gaussian setting, cycles do not introduce substantial complexity. One may illustrate it using examples in Figure 3.


We then examine a more subtle part in Theorem 2, concerning edge rank equivalence, that is, when
all the involved perfect bipartite matchings via edges are unchanged. Intuitively, it is about how edges
are structurally “crucial” for maintaining matchings. This leads to the following criterion about edge
additions or deletions, corresponding to flipping entries in the support matrix:

**Lemma 7** ( **Admissible edge additions/deletions** ) **.** _Let_ ( _G, X_ ) _be an irreducible model._ _For any_
_edge Vi_ _Vj_ _not currently in_ _, adding it to_ _preserves equivalence on X_ _if and only if:_
_→_ _G_ _G_


_rG_ ( _Vi’s nonchildren_ _Vj_ _,_ _L_ _Vi_ ) _<_ _rG_ ( _Vi’s nonchildren,_ _L_ _Vi_ ) _,_ (20)
_\{_ _}_ _\{_ _}_ _\{_ _}_


_where_ _Vi’s nonchildren_ _denotes_ _V_ ( _G_ ) _\_ ch _G_ ( _Vi_ ) _\{Vi},_ _i.e.,_ _zero_ _entries_ _in_ _support_ _column_ _Q_ [(] : _,V_ _[G]_ [)] _i_ _[.]_
_Conversely, an edge can be deleted if and only if it can be re-added by this criterion._


In layman’s term, Lemma 7 says that an edge _Vi_ _Vj_ can be added, only when in the bipartite graph
_→_
from latents to all vertices currently not _Vi_ ’s children (including _Vj_ ), _Vj_ stands as a “pillar” across
the maximum matchings; in matroid terms, it is a _coloop_ . Then, since _Vj_ is already a “pillar”, adding
this edge will not be noticed by any _Y_ containing latent variables. Note that both _Vi_ and _Vj_ may be
in _X_ or _L_ : edges can be added within each or in either direction. Let us examine an example.


8


|G<br>1 L L<br>1 2<br>X X X<br>1 2 3|Edge +/−|G<br>2 L L<br>1 2<br>X X X<br>1 2 3|Edge +/−|G<br>3 L L<br>1 2<br>X X X<br>1 2 3|Col6|
|---|---|---|---|---|---|
|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_1|_Edge_ +_/−_|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_2|_Edge_ +_/−_|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_3||
|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_4|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_4|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_5|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_5|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_6|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_6|
|_L_1<br>_L_2<br>_X_1<br>_X_2<br>_X_3<br>_G_4||||||


Figure 3: An example distributional equivalence class consisting of 6 digraphs up to _L_ -relabeling.


**Example 2** ( **Illustrating edge additions via Lemma 7** ) **.** We consider the digraph 1 in Figure 3,
_G_
and check why the edge _X_ 2 _L_ 2 can be added. From _L_ _X_ 2 = _L_ 1 _, L_ 2 to _X_ 2’s nonchildren
_→_ _\{_ _}_ _{_ _}_
_L_ 1 _, L_ 2 _, X_ 1, there is a full matching of size 2, with ( _L_ 1 _, L_ 2) matched to either ( _L_ 1 _, L_ 2) or ( _X_ 1 _, L_ 2).
_{_ _}_
Since _L_ 2 appears in both as a “pillar”, adding _X_ 2 _L_ 2 preserves edge ranks. In contrast, _X_ 2 _L_ 1
_→_ _→_
cannot be added, which, for instance, will change _rG_ 1( _L_ 1 _, L_ 2 _, X_ 1 _,_ _L_ 1 _, L_ 2 _, X_ 2 ) from 2 to 3.
_{_ _}_ _{_ _}_ _△_


We have introduced two graphical operations that preserve equivalence, namely, cycle reversals and
edge additions/deletions. Remarkably, these two operations are not only sufficient but also necessary:
together, they fully characterize equivalence. This brings us to our transformational characterization:


**Theorem 3** ( **Transformational characterization of the equivalence class** ) **.** _Two irreducible models_
( _G, X_ ) _and_ ( _H, X_ ) _are equivalent if and only if G_ _can be transformed into H, up to L-relabeling, via_
_a sequence of admissible cycle reversals and edge additions/deletions, as defined in Lemmas 6 and 7._

_Here, “up to L-relabeling” means there exists a relabeling of L in H yielding a digraph H_ _[′]_ _such_
_that G_ _reaches H_ _[′]_ _via the sequence._ _Moreover, at most one cycle reversal is needed in this sequence._


Thanks to this transformational characterization, Theorem 3 offers a natural way to traverse an
equivalence class by e.g., running BFS or DFS over the space of digraphs connected via admissible
operations. Such equivalence class structures are illustrated by Figure 3, Figure 5 (Appendix C.2),
[and more in our online demo.](https://equiv.cc) Note that this traversal can be further accelerated in implementation,
by traversing each vertex’s children independently in parallel (Lemmas 9 and 12; Appendix B).


Finally, let us return once more to the analogy with Markov equivalence. We have now established
counterparts of both “same adjacencies and v-structures” and “Meek conjecture”. A natural question
is then whether a counterpart of the CPDAG, an informative presentation of the equivalence class, can
also be developed. The answer is again yes. We show that within each cycle-reversal configuration,
there exists a unique maximal equivalent digraph of which all others are subgraphs. We further
provide efficient criteria to construct this maximal digraph, and to determine edges invariant across
the equivalence class (similar to arrows in a CPDAG). Due to space limit, this result is presented
in Theorem 4 (Appendix C.3). To conclude this section, we provide a side-by-side overview that
places our results with their analogues across various classical settings (Table 2; Appendix C.5).


5 ALGORITHM AND EVALUATION


In this section, we develop a structural-assumption-free algorithm to recover the underlying causal
models from observed data up to distributional equivalence. We name this algorithm as general latentvariable Linear Non-Gaussian causal discovery (glvLiNG). Evaluation results are also provided.


**Algorithm.** The glvLiNG pipeline consists of three main steps: it first runs OICA on data to estimate
a mixing matrix _A_ [˜], then constructs a digraph _G_ [˜] to realize rank patterns in _A_ [˜], and finally, starting
from _G_ [˜], traverses the equivalence class using the procedure introduced in Theorem 3. Under the
assumptions of access to an oracle OICA and faithfulness (no coincidental low ranks in the mixing
matrix beyond those structurally entailed; formally stated in Assumption 1 at Appendix A), glvLiNG
is guaranteed to recover the entire class of irreducible models equivalent to the ground-truth model.


9


Proofs and detailed formulations of the glvLiNG algorithm are deferred to Appendix A for page limit.
Here, we briefly highlight the core second step: constructing a digraph to realize the observed ranks.


The main challenge lies in this second step, a rank realization task. While the satisfiability nature of
this task may suggest brute-force solutions like integer programming, glvLiNG instead offers a more
efficient constraint-based approach. Specifically, it proceeds in two phases. Phase 1 recovers edges
from latent variables _L_ to all variables _V_, which reduces to a bipartite realization problem known in
matroid theory. Phase 2 is more delicate: recover edges from observed variables _X_ to _V_ . This may
seem combinatorially complex at first glance, since _all_ ranks induced by _all_ subsets of _X_ must be
jointly satisfied (Lemma 3). Fortunately, as we have shown in Theorem 2, these global constraints
admit a local decomposition, allowing each single _Xi_ ’s outgoing edges to be recovered independently.
To recover these edges, we give an explicit construction (Lemma 10 in Appendix A) based directly
on querying ranks in the OICA mixing matrix, with no need for solving complex constraint systems.


**Evaluation.** We evaluate our approach from five aspects: 1) quantifying the sizes of equivalence
classes, 2) assessing glvLiNG’s runtime, 3) benchmarking existing methods under oracle inputs, 4)
evaluating glvLiNG’s performance in simulations, and 5) applying glvLiNG to a real-world dataset.


For 1), we quantify the sizes of equivalence classes, in order to provide an illustrative sense of the
uncertainty in latent-variable models. We exhaustively partition digraphs with up to 6 vertices under
various latent configurations. For example, there are 1 _,_ 027 _,_ 080 weakly connected digraphs with 5 vertices, of which 26 _,_ 430 are acyclic. When the first 2 vertices are latent, 480 _,_ 640 of these digraphs yield
irreducible models, which finally form 783 equivalence classes. Full statistics are shown in Table 3.


For 2), we assess the efficiency gain enabled by glvLiNG’s constraint-based design. We compare
the execution time against a linear programming baseline for constructing digraphs to satisfy ranks
of oracle OICA mixing matrices. Results confirm substantial speedup: glvLiNG solves cases with
_n_ = 10 vertices in under 5s, while the baseline takes hours beyond _n_ = 5. Full results in Table 4.


For 3), we examine how existing methods behave under structural misspecification by applying them
to arbitrary latent-variable models possibly beyond their assumptions. We evaluate LaHiCaSl (Xie
et al., 2024) and PO-LiNGAM (Jin et al., 2024), given oracle access to their required tests. Both methods tend to produce overly sparse graphs and misidentify over half of the edges. Full results in Table 5.


For 4), we evaluate glvLiNG with existing methods under finite samples. We simulate data from
random irreducible models, varying numbers of observed and latent variables, graph density, and
sample size. We observe that glvLiNG performs particularly better than baselines on denser graphs
and stays more robust to latent dimensionality, likely due to avoiding model misspecification, while
baselines perform better on sparser graphs. Full setup and results are provided in Appendix D.4.


For 5), we apply glvLiNG to a real-world dataset of daily stock returns (Jan 2000-Jun 2005) from
14 major Hong Kong companies spanning banking, real estate, utilities, and commerce. glvLiNG
recovers meaningful patterns, such as major banks acting as central causal sources. The two latent
variables recovered seem also to admit plausible interpretations. Full results are in Appendix D.5.


**Final remarks.** We conclude with a reflection on the use of OICA in glvLiNG. While one may be
concerned about OICA’s known inefficiency in practice, we would like to note that the main focus of
this work is to characterize distributional equivalence. The glvLiNG algorithm serves more as a proof
of concept, showing that such equivalence is indeed recoverable without any structural assumption.


That said, we do see two promising directions for future improvement. 1) For estimation, several
existing methods allow partial access to rank information in the mixing matrix without explicitly
running OICA. They could be integrated into glvLiNG. 2) For algorithmic efficiency, while glvLiNG
already scales well, further pruning is possible. For instance, Theorem 3 implies that ancestral
relations among observed variables are identifiable, which may help reduce the search space.


6 CONCLUSION AND LIMITATIONS

In this work, we provide a graphical characterization of distributional equivalence for linear nonGaussian latent-variable models. Based on it, we develop a constraint-based algorithm, glvLiNG, that
recovers the underlying model up to equivalence from data without any structural assumptions. Central
to our approach is the introduction of edge rank constraints, a new tool in the rank-based picture. One
limitation is the use of OICA in glvLiNG, as discussed above. Future directions include developing
OICA-free algorithms, and extending new tools to broader settings like linear Gaussian systems.


10


ACKNOWLEDGMENT


We would like to acknowledge the support from NSF Award No. 2229881, AI Institute for Societal
Decision Making (AI-SDM), the National Institutes of Health (NIH) under Contract R01HL159805,
and grants from Quris AI, Florin Court Capital, MBZUAI-WIS Joint Program, and the Al Deira
Causal Education project. We also thank the anonymous reviewers for their helpful suggestions.


**Large Language Models Usage:** We used large language models only to aid or polish writing, at
the sentence level.


**Ethics Statement:** This paper presents work whose goal is to advance the field of causal discovery.
We do not see any ethical or societal concerns that need to be disclosed.


**Reproducibility Statement:** We provide code for our algorithm, glvLiNG, along with an interac[tive demo for traversing equivalence classes, available at https://equiv.cc.](https://equiv.cc)


REFERENCES


Jeffrey Adams, Niels Hansen, and Kun Zhang. Identification of partially observed linear causal models:
Graphical conditions for the Non-Gaussian and heterogeneous cases. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 34:22822–22833, 2021.


Ayesha R Ali, Thomas S Richardson, Peter L Spirtes, and Jiji Zhang. Towards characterizing Markov equivalence
classes for directed acyclic graphs with latent variables. _arXiv preprint arXiv:1207.1365_, 2005.


Carlos Améndola, Mathias Drton, Alexandros Grosdos, Roser Homs, and Elina Robeva. Third-order moment
varieties of linear Non-Gaussian graphical models. _Information and Inference:_ _A Journal of the IMA_, 12(3):
iaad007, 2023.


Carlos Améndola, Tobias Boege, Benjamin Hollering, and Pratik Misra. Structural identifiability of graphical
continuous lyapunov models. _arXiv preprint arXiv:2510.04985_, 2025.


Animashree Anandkumar, Daniel Hsu, Adel Javanmard, and Sham Kakade. Learning linear Bayesian networks
with latent variables. In _International Conference on Machine Learning_, pp. 249–257. PMLR, 2013.


Steen A Andersson, David Madigan, and Michael D Perlman. A characterization of Markov equivalence classes
for acyclic digraphs. _The Annals of Statistics_, 25(2):505–541, 1997.


Bryan Andrews, Peter Spirtes, and Gregory F Cooper. On the completeness of causal discovery in the presence of
latent confounding with tiered background knowledge. In _International Conference on Artificial Intelligence_
_and Statistics_, pp. 4002–4011. PMLR, 2020.


Thomas H. Brylawski. An Affine Representation for Transversal Geometries. _Studies_ _in_ _Applied_ _Mathe-_
_matics_, 54(2):143–160, 1975. doi: 10.1002/sapm1975542143. URL [https://doi.org/10.1002/](https://doi.org/10.1002/sapm1975542143)
[sapm1975542143.](https://doi.org/10.1002/sapm1975542143)


Eunice Yuh-Jie Chen, Arthur Choi Choi, and Adnan Darwiche. Enumerating equivalence classes of Bayesian
networks using ec graphs. In _Artificial Intelligence and Statistics_, pp. 591–599. PMLR, 2016.


Wei Chen, Zhiyi Huang, Ruichu Cai, Zhifeng Hao, and Kun Zhang. Identification of causal structure with latent
variables based on higher order cumulants. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
volume 38, pp. 20353–20361, 2024a.


Zhengming Chen, Ruichu Cai, Feng Xie, Jie Qiao, Anpeng Wu, Zijian Li, Zhifeng Hao, and Kun Zhang.
Learning discrete latent variable structures with tensor rank conditions. _Advances in Neural Information_
_Processing Systems_, 37:17398–17427, 2024b.


Zhengming Chen, Yewei Xia, Feng Xie, Jie Qiao, Zhifeng Hao, Ruichu Cai, and Kun Zhang. Identification
of latent confounders via investigating the tensor ranks of the nonlinear observations. In _Forty-second_
_International Conference on Machine Learning_, 2025. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=WH3ZRH2jno)
[WH3ZRH2jno.](https://openreview.net/forum?id=WH3ZRH2jno)


David Maxwell Chickering. A transformational characterization of equivalent Bayesian network structures. In
_Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence_, UAI’95, pp. 87–98, 1995.


11


David Maxwell Chickering. Optimal structure identification with greedy search. _Journal of machine learning_
_research_, 3(Nov):507–554, 2002.


Myung Jin Choi, Vincent YF Tan, Animashree Anandkumar, and Alan S Willsky. Learning latent tree graphical
models. _The Journal of Machine Learning Research_, 12:1771–1812, 2011.


Tom Claassen and Ioan G Bucur. Greedy equivalence search in the presence of latent confounders. In _Conference_
_on Uncertainty in Artificial Intelligence_, 2022.


Tom Claassen and Joris M Mooij. Establishing Markov equivalence in cyclic directed graphs. In _Uncertainty in_
_Artificial Intelligence_, pp. 433–442. PMLR, 2023.


Ruifei Cui, Perry Groot, Moritz Schauer, and Tom Heskes. Learning the causal structure of copula models with
latent variables. _UAI_, 2018.


Haoyue Dai, Peter Spirtes, and Kun Zhang. Independence testing-based approach to causal discovery under
measurement error and linear Non-Gaussian models. _Advances in Neural Information Processing Systems_,
35:27524–27536, 2022.


Haoyue Dai, Ignavier Ng, Yujia Zheng, Zhengqing Gao, and Kun Zhang. Local causal discovery with linear NonGaussian cyclic models. In _International Conference on Artificial Intelligence and Statistics_, pp. 154–162.
PMLR, 2024.


Haoyue Dai, Ignavier Ng, Jianle Sun, Zeyu Tang, Gongxu Luo, Xinshuai Dong, Peter Spirtes, and Kun
Zhang. When selection meets intervention: Additional complexities in causal discovery. In _The Thirteenth_
_International Conference on Learning Representations_, 2025a.


Haoyue Dai, Yiwen Qiu, Ignavier Ng, Xinshuai Dong, Peter Spirtes, and Kun Zhang. Latent variable causal
discovery under selection bias. In _Forty-second International Conference on Machine Learning_, 2025b. URL
[https://openreview.net/forum?id=W9YdVrSJIh.](https://openreview.net/forum?id=W9YdVrSJIh)


Xinshuai Dong, Biwei Huang, Ignavier Ng, Xiangchen Song, Yujia Zheng, Songyao Jin, Roberto Legaspi, Peter
Spirtes, and Kun Zhang. A versatile causal discovery framework to allow causally-related hidden variables.
In _The Twelfth International Conference on Learning Representations_, 2024.


Xinshuai Dong, Ignavier Ng, Boyang Sun, Haoyue Dai, Guang-Yuan Hao, Shunxing Fan, Peter Spirtes, Yumou
Qiu, and Kun Zhang. Permutation-based rank test in the presence of discretization and application in causal
discovery with mixed data. In _Forty-second International Conference on Machine Learning_, 2025. URL
[https://openreview.net/forum?id=VBTHduhm4K.](https://openreview.net/forum?id=VBTHduhm4K)


Xinshuai Dong, Ignavier Ng, Haoyue Dai, Jiaqi Sun, Xiangchen Song, Peter Spirtes, and Kun Zhang. Score-based
greedy search for structure identification of partially observed linear causal models. In _The Fourteenth Inter-_
_national Conference on Learning Representations_, 2026. [URL https://openreview.net/forum?](https://openreview.net/forum?id=BNHplerBYE)
[id=BNHplerBYE.](https://openreview.net/forum?id=BNHplerBYE)


Mathias Drton. Algebraic problems in structural equation modeling. In _The 50th anniversary of Gröbner bases_,
volume 77, pp. 35–87. Mathematical Society of Japan, 2018.


Mathias Drton, Bernd Sturmfels, and Seth Sullivant. Algebraic factor analysis: tetrads, pentads and beyond.
_Probability Theory and Related Fields_, 138(3):463–493, 2007.


Mathias Drton, Marina Garrote-López, Niko Nikov, Elina Robeva, and Y Samuel Wang. Causal discovery
for linear Non-Gaussian models with disjoint cycles. In _The 41st Conference on Uncertainty in Artificial_
_Intelligence_, 2025a.


Mathias Drton, Benjamin Hollering, and Jun Wu. Identifiability of homoscedastic linear structural equation
models using algebraic matroids. _Advances_ _in_ _Applied_ _Mathematics_, 163:102794, 2025b. ISSN 01968858. doi: https://doi.org/10.1016/j.aam.2024.102794. [URL https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S019688582400126X)
[science/article/pii/S019688582400126X.](https://www.sciencedirect.com/science/article/pii/S019688582400126X)


Bao Duong and Thi Kim Hue Nguyen. Normalizing flows for conditional independence testing. _Knowledge and_
_Information Systems_, 2024.


Frederick Eberhardt. Almost optimal intervention sets for causal discovery. In _Proceedings of the Twenty-Fourth_
_Conference on Uncertainty in Artificial Intelligence_, pp. 161–168, 2008.


Frederick Eberhardt and Richard Scheines. Interventions and causal inference. _Philosophy of science_, 74(5):
981–995, 2007.


12


Frederick Eberhardt, Clark Glymour, and Richard Scheines. On the number of experiments sufficient and in the
worst case necessary to identify all causal relations among n variables. In _Proceedings of the Twenty-First_
_Conference on Uncertainty in Artificial Intelligence_, pp. 178–184, 2005.


Jan Eriksson and Visa Koivunen. Identifiability, separability, and uniqueness of linear ica models. _IEEE signal_
_processing letters_, 11(7):601–604, 2004.


Robin J Evans. Graphs for margins of Bayesian networks. _Scandinavian Journal of Statistics_, 43(3):625–648,
2016.


Robin J Evans. Margins of discrete Bayesian networks. _The Annals of Statistics_, 46(6A):2623–2656, 2018.


Patrick Forré and Joris M Mooij. Markov properties for graphical models with cycles and latent variables. _arXiv_
_preprint arXiv:1710.08775_, 2017.


Morten Frydenberg. The chain graph Markov property. _Scandinavian journal of statistics_, pp. 333–353, 1990.


Dan Geiger and Christopher Meek. Graphical models and exponential families. _arXiv preprint arXiv:1301.7376_,
1996.


Ira Gessel and Gérard Viennot. Binomial determinants, paths, and hook length formulae. _Advances_ _in_
_mathematics_, 58(3):300–321, 1985.


AmirEmad Ghassami, Saber Salehkaleybar, Negar Kiyavash, and Kun Zhang. Learning causal structures using
regression invariance. _Advances in Neural Information Processing Systems_, 30, 2017.


AmirEmad Ghassami, Alan Yang, Negar Kiyavash, and Kun Zhang. Characterizing distribution equivalence and
structure learning for cyclic and acyclic directed graphs. In _International Conference on Machine Learning_,
pp. 3494–3504. PMLR, 2020.


Yuqi Gu and Gongjun Xu. Identifiability of hierarchical latent attribute models. _arXiv preprint arXiv:1906.07869_,
2023.


Alain Hauser and Peter Bühlmann. Characterization and greedy learning of interventional Markov equivalence
classes of directed acyclic graphs. _The Journal of Machine Learning Research_, 13(1):2409–2464, 2012.


Alain Hauser and Peter Bühlmann. Jointly interventional and observational data: estimation of interventional
Markov equivalence classes of directed acyclic graphs. _Journal of the Royal Statistical Society Series B:_
_Statistical Methodology_, 77(1):291–318, 2015.


Yang-Bo He and Zhi Geng. Active learning of causal networks with intervention experiments and optimal
designs. _Journal of Machine Learning Research_, 9(Nov):2523–2547, 2008.


Patrik O Hoyer, Shohei Shimizu, Antti J Kerminen, and Markus Palviainen. Estimation of causal effects using
linear Non-Gaussian causal models with hidden variables. _International Journal of Approximate Reasoning_,
49(2):362–378, 2008.


Yingyao Hu. Identification and estimation of nonlinear models with misclassification error using instrumental
variables: A general solution. _Journal of Econometrics_, 144(1):27–61, 2008.


Yingyao Hu. The econometrics of unobservables: Applications of measurement error models in empirical
industrial organization and labor economics. _Journal of econometrics_, 200(2):154–168, 2017.


Zhongyi Hu and Robin Evans. Faster algorithms for Markov equivalence. In _Conference on Uncertainty in_
_Artificial Intelligence_, pp. 739–748. PMLR, 2020.


Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard
Schölkopf. Causal discovery from heterogeneous/nonstationary data. _Journal of Machine Learning Research_,
21(89):1–53, 2020.


Biwei Huang, Charles Jia Han Low, Feng Xie, Clark Glymour, and Kun Zhang. Latent hierarchical causal
structure discovery with rank constraints. _Advances in neural information processing systems_, 35:5549–5561,
2022.


Aubrey W Ingleton and Mike J Piff. Gammoids and transversal matroids. _Journal of Combinatorial Theory,_
_Series B_, 15(1):51–68, 1973.


13


Fattaneh Jabbari, Joseph Ramsey, Peter Spirtes, and Gregory Cooper. Discovery of causal models that contain
latent variables through Bayesian scoring of independence constraints. In _Machine Learning and Knowledge_
_Discovery in Databases:_ _European Conference, ECML PKDD 2017, Skopje, Macedonia, September 18–22,_
_2017, Proceedings, Part II 10_, pp. 142–157. Springer, 2017.


Yibo Jiang and Bryon Aragam. Learning nonparametric latent causal graphs with unknown interventions.
_Advances in Neural Information Processing Systems_, 36:60468–60513, 2023.


Songyao Jin, Feng Xie, Guangyi Chen, Biwei Huang, Zhengming Chen, Xinshuai Dong, and Kun Zhang.
Structural estimation of partially observed linear Non-Gaussian acyclic model: A practical approach with
identifiability. In _The Twelfth International Conference on Learning Representations_, 2024.


Joseph Johnson and Pardis Semnani. Characteristic imsets for cyclic linear causal models and the chickering
ideal. _arXiv preprint arXiv:2506.13407_, 2025.


Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, and Bryon Aragam. Learning latent causal graphs via
mixture oracles. _Advances in Neural Information Processing Systems_, 34:18087–18101, 2021.


Yaroslav Kivva, Saber Salehkaleybar, and Negar Kiyavash. A cross-moment approach for causal effect estimation.
_Advances in Neural Information Processing Systems_, 36:9944–9955, 2023.


Murat Kocaoglu, Amin Jaber, Karthikeyan Shanmugam, and Elias Bareinboim. Characterization and learning
of causal graphs with latent variables from soft interventions. _Advances in Neural Information Processing_
_Systems_, 32, 2019.


Dénes K˝onig. Graphs and matrices. _Matematikai és Fizikai Lapok_, 38:116–119, 1931.


Erich Kummerfeld and Joseph Ramsey. Causal clustering for 1-factor measurement models. In _Proceedings of_
_the 22nd ACM SIGKDD international conference on knowledge discovery and data mining_, pp. 1655–1664,
2016.


Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. Discovering cyclic causal models by
independent components analysis. In _Conference on Uncertainty in Artificial Intelligence_, 2008.


Bernt Lindström. On the vector representations of induced matroids. _Bulletin of the London Mathematical_
_Society_, 5(1):85–90, 1973.


Yiheng Liu, Elina Robeva, and Huanqing Wang. Learning linear Non-Gaussian graphical models with multidirected edges. _Journal of Causal Inference_, 9(1):250–263, 2021.


Gongxu Luo, Haoyue Dai, Loka Li, Chengqian Gao, Boyang Sun, and Kun Zhang. Gene regulatory network
inference in the presence of selection bias and latent confounders. In _The Thirty-ninth Annual Conference_
_on_ _Neural_ _Information_ _Processing_ _Systems_, 2025. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=14irEkV01l)
[14irEkV01l.](https://openreview.net/forum?id=14irEkV01l)


Gongxu Luo, Loka Li, Guangyi Chen, Haoyue Dai, and Kun Zhang. Characterization and learning of causal
graphs with latent confounders and post-treatment selection from interventional data. In _The Fourteenth Inter-_
_national Conference on Learning Representations_, 2026. [URL https://openreview.net/forum?](https://openreview.net/forum?id=qclNnbjxNJ)
[id=qclNnbjxNJ.](https://openreview.net/forum?id=qclNnbjxNJ)


Yichen Lyu and Pengkun Yang. Identifiability and estimation in high-dimensional nonparametric latent structure
models. In Nika Haghtalab and Ankur Moitra (eds.), _Proceedings of Thirty Eighth Conference on Learning_
_Theory_, volume 291 of _Proceedings of Machine Learning Research_, pp. 3879–3880. PMLR, 30 Jun–04 Jul
2025.


Takashi Nicholas Maeda and Shohei Shimizu. RCD: Repetitive causal discovery of linear non-Gaussian acyclic
models with latent confounders. In _International Conference on Artificial Intelligence and Statistics_, 2020.


Sara Magliacane, Tom Claassen, and Joris M Mooij. Ancestral causal inference. _Advances in Neural Information_
_Processing Systems_, 29, 2016.


Alex Markham and Moritz Grosse-Wentrup. Measurement dependence inducing latent causal models. In
_Conference on Uncertainty in Artificial Intelligence_, pp. 590–599. PMLR, 2020.


Alex Markham, Danai Deligeorgaki, Pratik Misra, and Liam Solus. A transformational characterization of
unconditionally equivalent Bayesian networks. In _International_ _Conference_ _on_ _Probabilistic_ _Graphical_
_Models_, pp. 109–120. PMLR, 2022.


14


J.H. Mason. On a class of matroids arising from paths in graphs. _Proceedings of the London Mathematical_
_Society_, 3(1):55–74, 1972.


Christopher Meek. Causal inference and causal explanation with background knowledge. In _Proceedings of the_
_Eleventh conference on Uncertainty in artificial intelligence_, pp. 403–410, 1995.


Christopher Meek. _Graphical Models:_ _Selecting causal and statistical models_ . PhD thesis, Carnegie Mellon
University, 1997.


Nicolai Meinshausen, Alain Hauser, Joris M Mooij, Jonas Peters, Philip Versteeg, and Peter Bühlmann. Methods
for causal inference from gene perturbation experiments and validation. _Proceedings of the National Academy_
_of Sciences_, 113(27):7361–7368, 2016.


Karl Menger. Zur allgemeinen kurventheorie. _Fund. Math._, 10:96–1159, 1927.


Joris M Mooij and Tom Claassen. Constraint-based causal discovery using partial ancestral graphs in the
presence of cycles. In _Conference on Uncertainty in Artificial Intelligence_, pp. 1159–1168. Pmlr, 2020.


Joris M Mooij, Sara Magliacane, and Tom Claassen. Joint causal inference from multiple contexts. _Journal of_
_Machine Learning Research_, 21(99):1–108, 2020.


Ignavier Ng, Xinshuai Dong, Haoyue Dai, Biwei Huang, Peter Spirtes, and Kun Zhang. Score-based causal
discovery of latent variable causal models. In _Forty-first International Conference on Machine Learning_,
2024.


Christopher Nowzohour, Marloes H Maathuis, Robin J Evans, and Peter Bühlmann. Distributional equivalence
and structure learning for bow-free acyclic path diagrams. _Electronic Journal of Statistics_, 2017.


Juan Miguel Ogarrio, Peter Spirtes, and Joe Ramsey. A hybrid causal search algorithm for latent variable models.
In _Proceedings of the Eighth International Conference on Probabilistic Graphical Models_, pp. 368–379,
2016.


James G Oxley. _Matroid theory_, volume 3. Oxford University Press, USA, 2006.


Judea Pearl. _Causality:_ _models, reasoning and inference_ . Cambridge University Press, 2009.


Judea Pearl and Rina Dechter. Identifying independencies in causal graphs with feedback. _ArXiv_, 1996.


Hazel Perfect. Applications of menger’s graph theorem. _Journal of Mathematical Analysis and Applications_, 22
(1):96–111, 1968.


Emilija Perkovi´c, Markus Kalisch, and Maloes H Maathuis. Interpreting and using CPDAGs with background
knowledge. _arXiv preprint arXiv:1707.02171_, 2017.


Anastasia Podosinnikova, Amelia Perry, Alexander S Wein, Francis Bach, Alexandre d’Aspremont, and David
Sontag. Overcomplete independent component analysis via sdp. In _The 22nd international conference on_
_artificial intelligence and statistics_, pp. 2583–2592. PMLR, 2019.


Thomas Richardson and Peter Spirtes. Ancestral graph Markov models. _The Annals of Statistics_, 30(4):962–1030,
2002.


Thomas S Richardson. _Discovering_ _cyclic_ _causal_ _structure_ . Department of Philosophy, Carnegie Mellon
University, 1996.


Thomas S Richardson, Robin J Evans, James M Robins, and Ilya Shpitser. Nested Markov properties for acyclic
directed mixed graphs. _The Annals of Statistics_, 51(1):334–361, 2023.


Elina Robeva and Jean-Baptiste Seby. Multi-trek separation in linear structural equation models. _SIAM Journal_
_on Applied Algebra and Geometry_, 5(2):278–303, 2021.


Dominik Rothenhäusler, Christina Heinze, Jonas Peters, and Nicolai Meinshausen. Backshift: Learning causal
cyclic graphs from unknown shift interventions. _Advances in Neural Information Processing Systems_, 28,
2015.


Alberto Roverato, Milan Studeny, and David Madigan.` A graphical representation of equivalence classes of amp
chain graphs. _Journal of Machine Learning Research_, 7(6), 2006.


Saber Salehkaleybar, AmirEmad Ghassami, Negar Kiyavash, and Kun Zhang. Learning linear Non-Gaussian
causal models in the presence of latent variables. _The_ _Journal_ _of_ _Machine_ _Learning_ _Research_, 21(1):
1436–1459, 2020.


15


Daniela Schkoda and Mathias Drton. Goodness-of-fit tests for linear Non-Gaussian structural equation models.
_Biometrika_, pp. asaf046, 2025.


Daniela Schkoda, Elina Robeva, and Mathias Drton. Causal discovery of linear Non-Gaussian causal models
with unobserved confounding. _arXiv preprint arXiv:2408.04907_, 2024.


Ehsan Sharifian, Saber Salehkaleybar, and Negar Kiyavash. Near-optimal experiment design in linear NonGaussian cyclic models. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_,
2025. [URL https://openreview.net/forum?id=opAU0pYlcP.](https://openreview.net/forum?id=opAU0pYlcP)


Shohei Shimizu. _Statistical causal discovery:_ _LiNGAM approach_ . Springer, 2022.


Shohei Shimizu and Kenneth Bollen. Bayesian estimation of causal direction in acyclic structural equation
models with individual-specific confounder variables and Non-Gaussian distributions. _Journal of Machine_
_Learning Research-JMLR_, 15(1):2629–2652, 2014.


Shohei Shimizu, Patrik O Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. A linear Non-Gaussian
acyclic model for causal discovery. _Journal of Machine Learning Research_, 7(10), 2006.


Ricardo Silva and Richard Scheines. _Generalized measurement models_ . Carnegie Mellon University. Center for
Automated Learning and Discovery, 2004.


Ricardo Silva and Shohei Shimizu. Learning instrumental variables with structural and Non-Gaussianity
assumptions. _Journal of Machine Learning Research_, 18(120):1–49, 2017.


Ricardo Silva, Richard Scheines, Clark Glymour, and Peter Spirtes. Learning measurement models for unobserved variables. In _Proceedings of the Nineteenth Conference on Uncertainty in Artificial Intelligence_,
UAI’03, 2003.


Ricardo Silva, Richard Scheines, Clark Glymour, and Peter Spirtes. Learning the structure of linear latent
variable models. _Journal of Machine Learning Research_, 7(2), 2006.


Peter Spirtes. Building causal graphs from statistical data in the presence of latent variables. _Department of_
_Philosophy technical report_, 1992.


Peter Spirtes. Conditional independence in directed cyclic graphical models representing feedback or mixtures.
Technical report, Philosophy, Methodology and Logic Technical Report 59, CMU, 1994.


Peter Spirtes and Clark Glymour. An algorithm for fast recovery of sparse causal graphs. _Social_ _Science_
_Computer Review_, 9:62–72, 1991.


Peter Spirtes and Thomas Richardson. A polynomial time algorithm for determining DAG equivalence in the
presence of latent variables and selection bias. In _Proceedings of the 6th International Workshop on Artificial_
_Intelligence and Statistics_, volume 12, 1996.


Peter Spirtes and Thomas Verma. Equivalence of causal models with latent variables. _Carnegie Mellon University_
_Tech Report_, 1992.


Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. _Causation, prediction, and search_ .
MIT press, 2000.


Peter L Spirtes. Calculation of entailed rank constraints in partially non-linear and cyclic models. _arXiv preprint_
_arXiv:1309.7004_, 2013.


Chandler Squires, Yuhao Wang, and Caroline Uhler. Permutation-based causal structure learning with unknown
intervention targets. In _Conference on Uncertainty in Artificial Intelligence_, pp. 1039–1048. PMLR, 2020.


Chandler Squires, Annie Yun, Eshaan Nichani, Raj Agrawal, and Caroline Uhler. Causal structure discovery
between clusters of nodes induced by latent factors. In _Conference on Causal Learning and Reasoning_, pp.
669–687. PMLR, 2022.


Bertran Steinsky. Enumeration of labelled chain graphs and labelled essential directed acyclic graphs. _Discrete_
_mathematics_, 270(1-3):267–278, 2003.


Nils Sturma, Chandler Squires, Mathias Drton, and Caroline Uhler. Unpaired multi-domain causal representation
learning. _Advances in Neural Information Processing Systems_, 36, 2024.


Seth Sullivant, Kelli Talaska, and Jan Draisma. Trek separation for Gaussian graphical models. _The Annals of_
_Statistics_, 38(3):1665–1685, 2010.


16


Kelli Talaska. Determinants of weighted path matrices. _arXiv:_ _Combinatorics_, 2012. [URL https://api.](https://api.semanticscholar.org/CorpusID:119671799)
[semanticscholar.org/CorpusID:119671799.](https://api.semanticscholar.org/CorpusID:119671799)


Tatsuya Tashiro, Shohei Shimizu, Aapo Hyvärinen, and Takashi Washio. Parcelingam: A causal ordering method
robust against latent confounders. _Neural computation_, 26(1):57–83, 2014.


Eric J Tchetgen Tchetgen, Andrew Ying, Yifan Cui, Xu Shi, and Wang Miao. An introduction to proximal
causal inference. _Statistical Science_, 39(3):375–390, 2024.


Henry Teicher. Identifiability of mixtures of product measures. _The Annals of Mathematical Statistics_, 38(4):
1300–1302, 1967.


Jin Tian. Generating Markov equivalent maximal ancestral graphs by single edge replacement. In _Conference_
_on Uncertainty in Artificial Intelligence_, 2005.


Jin Tian and Judea Pearl. Causal discovery from changes. _arXiv preprint arXiv:1301.2312_, 2001.


Daniele Tramontano, Anthea Monod, and Mathias Drton. Learning linear Non-Gaussian polytree models. In
_Uncertainty in Artificial Intelligence_, pp. 1960–1969. PMLR, 2022.


Daniele Tramontano, Mathias Drton, and Jalal Etesami. Parameter identification in linear Non-Gaussian causal
models under general confounding. _arXiv preprint arXiv:2405.20856_, 2024.


Daniele Tramontano, Yaroslav Kivva, Saber Salehkaleybar, Negar Kiyavash, and Mathias Drton. Causal effect
identification in lvLiNGAM from higher-order cumulants. In _Forty-second_ _International_ _Conference_ _on_
_Machine Learning_, 2025. [URL https://openreview.net/forum?id=39JKH8k3FS.](https://openreview.net/forum?id=39JKH8k3FS)


Aparajithan Venkateswaran and Emilija Perkovi´c. Towards complete causal explanation with expert knowledge.
_arXiv preprint arXiv:2407.07338_, 2024.


Thomas S Verma and Judea Pearl. Equivalence and synthesis of causal models. In _Uncertainty in Artificial_
_Intelligence_, 1991.


Tian-Zuo Wang, Wen-Bo Du, and Zhi-Hua Zhou. An efficient maximal ancestral graph listing algorithm. In
_Forty-first International Conference on Machine Learning_, 2024.


Tian-Zuo Wang, Wen-Bo Du, and Zhi-Hua Zhou. Polynomial-delay mag listing with novel locally complete
orientation rules. In _Forty-second International Conference on Machine Learning_, 2025.


Y Samuel Wang and Mathias Drton. Causal discovery with unobserved confounding and Non-Gaussian data.
_Journal of Machine Learning Research_, 24(271):1–61, 2023.


Yuhao Wang, Liam Solus, Karren Yang, and Caroline Uhler. Permutation-based causal inference algorithms
with interventions. _Advances in Neural Information Processing Systems_, 30, 2017.


Marcel Wienöbst, Malte Luttermann, Max Bannach, and Maciej Liskiewicz. Efficient enumeration of Markov
equivalent DAGs. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 37, pp. 12313–
12320, 2023.


Yewei Xia, Zhengming Chen, Haoyue Dai, Fuhong Wang, Yixin Ren, Yiqing Li, Kun Zhang, and Shuigeng Zhou.
Conditional independent component analysis for estimating causal structure with latent variables. In _The_
_Fourteenth International Conference on Learning Representations_, 2026. [URL https://openreview.](https://openreview.net/forum?id=TAOpnCPnjg)
[net/forum?id=TAOpnCPnjg.](https://openreview.net/forum?id=TAOpnCPnjg)


Feng Xie, Ruichu Cai, Biwei Huang, Clark Glymour, Zhifeng Hao, and Kun Zhang. Generalized independent
noise condition for estimating latent variable causal graphs. _Advances in Neural Information Processing_
_Systems_, 33:14891–14902, 2020.


Feng Xie, Yangbo He, Zhi Geng, Zhengming Chen, Ru Hou, and Kun Zhang. Testability of instrumental
variables in linear Non-Gaussian acyclic causal models. _Entropy_, 24(4):512, 2022.


Feng Xie, Biwei Huang, Zhengming Chen, Ruichu Cai, Clark Glymour, Zhi Geng, and Kun Zhang. Generalized
independent noise condition for estimating causal structure with latent variables. _Journal of Machine Learning_
_Research_, 25(191):1–61, 2024. [URL http://jmlr.org/papers/v25/23-1052.html.](http://jmlr.org/papers/v25/23-1052.html)


Baoying Yang, Jing Qin, Jing Ning, and Yukun Liu. Double robust conditional independence test for novel
biomarkers given established risk factors with survival data. _Biometrics_, 81(4):ujaf133, 2025.


Karren Yang, Abigail Katcoff, and Caroline Uhler. Characterizing and learning equivalence classes of causal
DAGs under interventions. In _International Conference on Machine Learning_, pp. 5541–5550. PMLR, 2018.


17


Yuqin Yang, AmirEmad Ghassami, Mohamed Nafea, Negar Kiyavash, Kun Zhang, and Ilya Shpitser. Causal
discovery in linear latent variable models subject to measurement error. _Advances in Neural Information_
_Processing Systems_, 35:874–886, 2022.


Yuqin Yang, Mohamed S Nafea, Negar Kiyavash, Kun Zhang, and AmirEmad Ghassami. Causal discovery in
linear models with unobserved variables and measurement error. In _NeurIPS 2024 Causal Representation_
_Learning Workshop_, 2024. [URL https://openreview.net/forum?id=L1Zfs3wgCg.](https://openreview.net/forum?id=L1Zfs3wgCg)


Binghua Yao and Joris Mooij. Sigma-maximal ancestral graphs. In _The 41st Conference on Uncertainty in_
_Artificial Intelligence_, 2025. [URL https://openreview.net/forum?id=8dpnJlEdrH.](https://openreview.net/forum?id=8dpnJlEdrH)


Dingling Yao, Dario Rancati, Riccardo Cadei, Marco Fumero, and Francesco Locatello. Unifying causal
representation learning with the invariance principle. In _The Thirteenth International Conference on Learning_
_Representations_, 2025. [URL https://openreview.net/forum?id=lk2Qk5xjeu.](https://openreview.net/forum?id=lk2Qk5xjeu)


Bixi Zhang and Wolfgang Wiedermann. Covariate selection in causal learning under Non-Gaussianity. _Behavior_
_Research Methods_, 56(4):4019–4037, 2024.


Jiaqi Zhang, Kristjan Greenewald, Chandler Squires, Akash Srivastava, Karthikeyan Shanmugam, and Caroline
Uhler. Identifiability guarantees for causal disentanglement from soft interventions. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_, 2023.


Jiji Zhang. On the completeness of orientation rules for causal discovery in the presence of latent confounders
and selection bias. _Artificial Intelligence_, 172(16-17):1873–1896, 2008a.


Jiji Zhang. Causal reasoning with ancestral graphs. _Journal of Machine Learning Research_, 9(7), 2008b.


Jiji Zhang and Peter Spirtes. A transformational characterization of Markov equivalence for directed acyclic
graphs with latent variables. In _Conference on Uncertainty in Artificial Intelligence_, 2005.


Kun Zhang, Biwei Huang, Jiji Zhang, Bernhard Schölkopf, and Clark Glymour. Discovery and visualization of
nonstationary causal models. _arXiv preprint arXiv:1509.08056_, 2015.


Kun Zhang, Mingming Gong, Joseph D Ramsey, Kayhan Batmanghelich, Peter Spirtes, and Clark Glymour.
Causal discovery with linear Non-Gaussian models under measurement error: Structural identifiability results.
In _UAI_, pp. 1063–1072, 2018.


Kun Zhang, Shaoan Xie, Ignavier Ng, and Yujia Zheng. Causal representation learning from multiple distributions: A general setting. In _Forty-first_ _International_ _Conference_ _on_ _Machine_ _Learning_, 2024. URL
[https://openreview.net/forum?id=Pte6iiXvpf.](https://openreview.net/forum?id=Pte6iiXvpf)


18


# **Appendix**

### **Table of Contents**


**A** **Details of the glvLiNG algorithm** **19**

A.1 Basics of Transversal Matroids . . . . . . . . . . . . . . . . . . . . . . . . . . 19

A.2 Algorithm Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

A.3 Phase 1: Bipartite Graph Realization . . . . . . . . . . . . . . . . . . . . . . . 22

A.4 Phase 2: Augmenting a Bipartite Graph for Matroid Extensions . . . . . . . . . 22


**B** **Proofs of Main Results** **24**

B.1 Proofs of Irreducibility Results . . . . . . . . . . . . . . . . . . . . . . . . . . 24

B.2 Proof from Rank Equivalence to Distributional Equivalence . . . . . . . . . . . 25

B.3 Proofs of Matroid-Preserving Column Augmentation: a Core Component . . . . 27

B.3.1 Constructing Particular Solutions to a Column Augmentation . . . . . . 27

B.3.2 Traversing All Solutions to a Column Augmentation . . . . . . . . . . . 28

B.3.3 From One Column Augmentation to Multiple Columns’ Joint Augmentation 31

B.4 Proofs of the Graphical Criterion and Transformational Characterization . . . . . 32

B.5 Other Immediate or Known Results . . . . . . . . . . . . . . . . . . . . . . . . 35


**C** **Discussion** **36**

C.1 Summary: A Side-by-side Comparison Between Path Ranks and Edge Ranks . . 36

C.2 Another Example Distributional Equivalence Class . . . . . . . . . . . . . . . . 36

C.3 A Presentation of the Equivalence Class . . . . . . . . . . . . . . . . . . . . . . 37

C.4 Examples of Non-Rank Constraints in Mixing Matrices . . . . . . . . . . . . . 38

C.5 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39


**D** **Evaluation Results** **43**

D.1 Quantifying the Sizes of Equivalence Classes . . . . . . . . . . . . . . . . . . . 43

D.2 Assessing glvLiNG Algorithm’s Runtime . . . . . . . . . . . . . . . . . . . . . 44

D.3 Benchmarking Existing Methods under Oracle Inputs . . . . . . . . . . . . . . 45

D.4 Evaluating glvLiNG’s Performance with Existing Methods in Simulations . . . . 46

D.5 Analyzing a Real-World Dataset with glvLiNG Algorithm . . . . . . . . . . . . 47


A DETAILS OF THE GLVLING ALGORITHM


A.1 BASICS OF TRANSVERSAL MATROIDS


Before proceeding, let us introduce some basic concepts from matroid theory that will be used later.
Throughout, we define matroids in terms of binary matrices, interpreted as adjacency matrices of
bipartite graphs where columns point to rows. The matroid is defined over row indices, corresponding
to what is known as a _transversal matroid_ . For more, one may refer to (Oxley, 2006).
**Definition 7** ( **Basics of transversal matroid** ) **.** Let _Q ∈{_ 0 _,_ 1 _}_ _[m][×][n]_ be a binary matrix, interpreted
as the adjacency matrix of a bipartite graph where columns [ _n_ ] point to rows _E_ := [ _m_ ], where _E_ is
called the _ground set_ . For simplicity, for each row set _Z_ _⊆_ _E_ we denote its rank as:

_r_ ( _Z_ ) := mrank( _QZ,_ :) _,_ (A.1)


though with a slight notation abuse to the letter _r_ we used previously for edge ranks (Definition 4).
Here, mrank is the matching rank we defined in Definition 6. This rank function _r_ turns _E_ into a
transversal matroid presented by _Q_ . We record the following basic concepts of this matroid, together
with some useful properties, written directly in terms of _Q_ and _r_ :


19


**Independent/dependent sets.**
Ind( _Q_ ) := _{Z_ _⊆_ _E_ : _r_ ( _Z_ ) = _|Z|},_


(A.2)
Dep( _Q_ ) := 2 _[E]_ _\_ Ind( _Q_ ) _._


_Note._ _Z_ is independent if and only if the rows _Z_ admit a matching into the columns; dependent sets
are those with a matching deficiency.


**Bases.**
bases( _Q_ ) := _{B_ _⊆_ _E_ : _B_ _∈_ Ind( _Q_ ) _,_ _|B|_ = _r_ ( _E_ ) _}._ (A.3)
_Note._ Bases are the maximal independent sets: all its subsets are independent, and all its proper
supersets are dependent. Bases are the maximum-cardinality independent sets (all have size _r_ ( _E_ )).
Bases uniquely determine the matroid.


**Circuits.**
circuits( _Q_ ) := _{C_ _⊆_ _E_ : _C_ _∈_ Dep( _Q_ ) and _∀_ _C_ _[′]_ ⊊ _C,_ _C_ _[′]_ _∈_ Ind( _Q_ ) _}._ (A.4)
_Note._ Circuits are the minimal dependent sets: _r_ ( _C_ ) = _|C|−_ 1 and every proper subset is independent.
Every dependent set contains a circuit as subset. Circuits do not necessarily have the same cardinalities.
Circuits uniquely determine the matroid.


**Cocircuits.**
cocircuits( _Q_ ) := _{D_ _⊆_ _E_ : _r_ ( _E \ D_ ) = _r_ ( _E_ ) _−_ 1 and _∀_ _D_ _[′]_ ⊊ _D,_ _r_ ( _E \ D_ _[′]_ ) = _r_ ( _E_ ) _}._ (A.5)
_Note._ Cocircuits are the minimal rank-dropping blockers: removing _D_ lowers the full rank (by exactly
one), while removing any proper subset does not. Cocircuits meet every basis:
_|D ∩_ _B| ≥_ 1 _,_ _∀_ _D_ _∈_ cocircuits( _Q_ ) _,_ _B_ _∈_ bases( _Q_ ) _._ (A.6)
Equivalently, _D_ is a minimal set intersecting all bases. By minimal we mean, for any cocircuit _D_, no
proper subset of _D_ can be a cocircuit.


When cocircuits meet circuits, the intersection size is never 1. In particular,
_|D ∩_ _C| ∈{_ 0 _,_ 2 _,_ 3 _, · · · },_ _∀_ _D_ _∈_ cocircuits( _Q_ ) _,_ _C_ _∈_ circuits( _Q_ ) _._ (A.7)
Equivalently, _D_ is a minimal set not intersecting any circuit with size 1.


Cocircuits uniquely determine the matroid.


**Coloops.**
coloops( _Q_ ) := _{e ∈_ _E_ : _r_ ( _E \ {e}_ ) = _r_ ( _E_ ) _−_ 1 _}._ (A.8)
_Note._ A coloop is an element whose presence always increases rank by 1. A coloop is an element
that is in every basis. For each element _e ∈_ _E_, the following are equivalent:
_e ∈_ coloops( _Q_ ) _⇐⇒{e} ∈_ cocircuits( _Q_ ) _⇐⇒_ _e_ is in every basis _⇐⇒_ _e_ is in no circuit _._ (A.9)
Coloops _do not_ determine the matroid.


**Flats.**
flats( _Q_ ) := _{F_ _⊆_ _E_ : _∀_ _x ∈_ _E\F,_ _r_ ( _F_ _∪{x}_ ) = _r_ ( _F_ ) + 1 _}._ (A.10)
_Note._ Flats are the _⊆_ -maximal sets that have a given rank _r_ ( _F_ ). The family of flats uniquely
determines a matroid.


**Fundamental circuit with respect to a basis.** For any basis _B_ and any _e_ _∈_ _E_ _\ B_, there is a
unique circuit _CB_ ( _e_ ) such that
_e_ _CB_ ( _e_ ) _B_ _e_ _,_ (A.11)
_∈_ _⊆_ _∪{_ _}_
called the fundamental circuit of _e_ w.r.t. _B_ . Moreover, for every _f_ _CB_ ( _e_ ) _e_, the set _B_ _f_ _e_
_∈_ _\{_ _}_ _\{_ _}∪{_ _}_
is a basis. Every circuit is a fundamental circuit to some _B_ and _e_ .


**Fundamental cocircuit with respect to a basis.** For any basis _B_ and any _e ∈_ _B_, there is a unique
cocircuit _DB_ ( _e_ ) such that
_e_ _DB_ ( _e_ ) ( _E_ _B_ ) _e_ _,_ (A.12)
_∈_ _⊆_ _\_ _∪{_ _}_
called the fundamental cocircuit of _e_ w.r.t. _B_ . Moreover, for every _f_ _DB_ ( _e_ ) _e_, the set
_∈_ _\_ _{_ _}_
_B\{e} ∪{f_ _}_ is a basis. Every cocircuit is a fundamental cocircuit to some _B_ and _e_ .


Having introduced these basics of transvesal matroids, below we explain our algorithm in detail.


20


A.2 ALGORITHM OVERVIEW


Let us first formally define the faithfulness assumption required by the glvLiNG algorithm:
**Assumption 1** ( **Faithfulness** ) **.** Let _dX_ := _X_ be the number of observed variables. We assume
_|_ _|_
that in the true mixing matrix _AX,_ : that generates data (as defined in Equation (5)), all _dX_ _dX_
_×_
and ( _dX_ 1) ( _dX_ 1) minors exhibit matrix ranks consistent with the corresponding path ranks

_−_ _×_ _−_
entailed by the true causal graph (as characterized in Lemma 2).


In other words, there is no coincidental parameter cancellation in the data generating process that
would lead to matrix ranks lower than those structurally entailed by the graph. Note that such
faithfulness assumption, often also referred to as the genericity assumption, is standard in the
literature (Adams et al., 2021). It holds almost everywhere in the parameter space except for a
Lebesgue measure zero set where coincidental lower ranks occur.


We next elaborate on our glvLiNG algorithm. The core to the glvLiNG algorithm is to query
matrix ranks from the mixing matrix estimated from data using overcomplete ICA (OICA), and then
construct a binary support matrix (corresponding to a digraph) that satisfies these matrix ranks.

Let _p_ ( _X_ ) be a data distribution generically generated by an unknown latent-variable model ( _G, X_ ),
that is, _p_ ( _X_ ) _∈P_ ( _G, X_ ). Without loss of generality we assume that ( _G, X_ ) is irreducible. Let
_A_ ˜ _∈_ R _[|][X][|×|][V][ |]_ be a mixing matrix estimated on _p_ ( _X_ ) by OICA, and for now, we index the rows and
columns of _A_ [˜] by _X_ and _V_, respectively. By the identifiability of OICA, _A_ [˜] is the true mixing matrix
up to column permutation and scaling. Further, with the duality between path ranks and edge ranks
(Theorem 1), we have that, there exists a permutation _π_ of _V_, such that for all _Z_ _⊆_ _X_ and _Y_ _⊆_ _V_,
the following equality holds:

rank( _A_ [˜] _Z,Y_ ) = _ρG_ ( _Z, π_ ( _Y_ )) = _|Z|_ + _|Y | −|V |_ + _rG_ ( _V \π_ ( _Y_ ) _, V \Z_ ) _._ (A.13)

In other words, there exists an unknown binary matrix _Q_ _∈{_ 0 _,_ 1 _}_ _[|][V][ |×|][V][ |]_, whose matching ranks
mrank( _QZ,Y_ ) can be queried for any _Z, Y_ _⊆_ _V_ with _L_ _⊆_ _Y_, despite its exact entry values being
unknown. Such a matrix must exist, with one specific matrix, _Q_ [(] _[G]_ [)] with rows permuted by _π_, being
an example. As long as one can recover this matrix _Q_, one can then permute its rows to place nonzero
entries on the diagonal, and the resulting matrix must exactly be some support matrix for a digraph _X_ _H_
withmrank( _G_ _∼HQ_ ) =, by Lemma 5. _|V |_, by settingSuch a row permutation to have nonzero diagonals must also exists, as _Z, Y_ both to ∅ in Equation (A.13).


Our problem then reduces to: how can one recover such a _Q_ matrix from rank queries? We express
this problem in a more general formulation, as follows:


Clearly, the key problem posed above is essentially a satisfiability problem, and can be solved
via brute-force methods such as linear programming. However, in what follows, we present a
significantly more efficient and structured procedure. Note that the matching rank queries in the
problem are equivalent to providing the transversal matroids on each submatrix _Q_ : _,Y_ . Thus, for
convenience, throughout the rest of this appendix, we may freely use the matroid language introduced
in Definition 7.


21


**Overview of our procedure:** Our reconstruction procedure consists of two phases:


Phase 1: Impute the columns in _H_ indexed by _L_ to satisfy the matroid bases( _Q_ : _,L_ ). Equivalently,
this is to construct a bipartite graph that realizes a given transversal matroid.


Phase 2: Impute the remaining columns indexed by _X_ such that all matroids induced by _Q_ : _,L∪{_ _**x**_ _}_
for _**x**_ _⊆_ _X_ are satisfied. Although this may appear combinatorially complex at first
glance, we show that each singleton column in _X_ can in fact be imputed independently.


A.3 PHASE 1: BIPARTITE GRAPH REALIZATION


Let us first formulate the problem of Phase 1:


By duality, this problem is equivalent to reconstructing the digraph representation of the strict
gammoid that is the dual of the transversal matroid based on the seminal paper by (Mason, 1972) and
dualizing the result using the Fundamental Lemma by (Ingleton & Piff, 1973).

The bases of the dual matroid _Q_ _[∗]_ are given by bases( _Q_ _[∗]_ ) = _{E\B_ _| B_ _∈_ bases( _Q_ ) _}_ . The _α_ -system
for _Q_ _[∗]_ is defined as the bipartite graph with the following incidence relation


_IQ∗_ = ( _e,_ ( _F, i_ )) _E_ (flats( _Q_ _[∗]_ ) _,_ N) _F_ flats( _Q_ _[∗]_ ) _,_ _e_ _F,_ _i_ N _,_ 1 _i_ _αQ∗_ ( _F_ ) _,_
_{_ _∈_ _×_ _|_ _∈_ _∈_ _∈_ _≤_ _≤_ _}_
(A.17)


where


           _αQ∗_ ( _F_ ) = _F_ _rQ∗_ ( _F_ ) _αQ∗_ ( _G_ ) _._ (A.18)
_|_ _| −_ _−_ _G∈_ flats( _Q_ _[∗]_ ) _, G_ ⊊ _F_


Since _Q_ is a transversal matroid, _Q_ _[∗]_ is a strict gammoid, and therefore all _αQ∗_ ( _F_ ) 0,

- _≥_

_F ∈_ flats( _Q_ _[∗]_ ) _[α][Q][∗]_ [(] _[F]_ [) =] _[ |][E][| −]_ _[r][Q][∗]_ [(] _[E]_ [) =] _[ r][Q]_ [(] _[E]_ [)][, and the] _[ α]_ [-system for] _[ Q][∗]_ [has a maximal matching]
that covers all ( _F, i_ ), and each such matching has the property that the set of unmatched elements
from _E_ forms a basis _T_ of _Q_ _[∗]_ .


Now fix such a maximal matching, let _T_ be the unmatched basis, and let ( _Fe, ie_ ) be the vertex that
_⊆_
is matched to _e_ for all _e_ _∈_ _E\T_ . Define the digraph _D_ = ( _V, A_ ) with _V_ = _E_ and _A_ = _{_ ( _u, v_ ) _∈_
_E_ _E_ _u_ _/_ _T,_ _v_ _Fu,_ _v_ = _u_ . The digraph _D_ represents the strict gammoid _Q_ _[∗]_ (Mason, 1972).
_×_ _|_ _∈_ _∈_ _}_
Using the fundamental lemma (Ingleton & Piff, 1973), we obtain that


_H_ = ( _e, t_ ) _E_ ( _E_ _T_ ) _e_ _Ft_ (A.19)
_{_ _∈_ _×_ _\_ _|_ _∈_ _}_


represents the transversal matroid _Q_ .


A.4 PHASE 2: AUGMENTING A BIPARTITE GRAPH FOR MATROID EXTENSIONS


Having imputed the values in _H_ : _,L_, we then impute the remaining _X_ columns:


22


One may first question whether such an imputation is possible, since overall the already assigned _H_ : _,L_
only realizes the matroid equality bases( _H_ : _,L_ ) = bases( _Q_ : _,L_ ), but the exact entry values recovery
_H_ : _,L_ = _Q_ : _,L_ is not guaranteed (and also impossible).


We show that such an imputation is indeed possible, via the following result:


**Lemma** **8** ( **How** **the** **transversal** **matroid** **changes** **when** **augmenting** **more** **sources** ) **.** _For_ _two_
_binary matrices Q_ 1 0 _,_ 1 _and Q_ 2 0 _,_ 1 _, we denote by_ [ _Q_ 1 _Q_ 2] 0 _,_ 1
_∈{_ _}_ _[m][×][n]_ [1] _∈{_ _}_ _[m][×][n]_ [2] _|_ _∈{_ _}_ _[m][×]_ [(] _[n]_ [1][+] _[n]_ [2][)]
_the matrix obtained by horizontally concatenating Q_ 1 _with Q_ 2 _._ _Then, we have:_


Ind([ _Q_ 1 _Q_ 2]) = _Z_ 1 _Z_ 2 : _Z_ 1 Ind( _Q_ 1) _,_ _Z_ 2 Ind( _Q_ 2) _._ (A.21)
_|_ _{_ _∪_ _∈_ _∈_ _}_

_In other words, two matrices’ matroids sufficiently determine the matroid of their augmentation._


With Lemma 8, every bases( _H_ : _,L∪_ _**x**_ ) equals bases([ _Q_ : _,L_ _H_ : _,_ _**x**_ ]), and thus the imputation is possible.
_|_


Then, how to solve for this imputation? At the first glance, one may have concern on the complexity:
in contrast to solving Phase 1’s realization problem for only one matroid induced by _L_, now we need
to realize combinatorially many matroids induced by _L ∪_ _**x**_ for all _**x**_ _⊆_ _X_ . When trying to impute a
single column _H_ : _,Xi_, this column can appear in many subsets _**x**_ _Xi_ .
_∋_

Interestingly, all these subsets can be disentangled: one do not need to explicitly realize each _**x**_ _⊆_ _X_ .
Instead, it suffices to just realize each singleton augmentation for _Xi_ _X_ independently.
_∈_

We show this by the following result:

**Lemma 9** ( **Reducing all union equivalence checks to singleton checks** ) **.** _Let Q, H_ _∈{_ 0 _,_ 1 _}_ _[m][×][n]_
_be two binary matrices with columns partitioned as_ [ _n_ ] = _L ∪_ _X_ _for a fixed L._ _Then, the condition_


bases( _Q_ : _,L∪_ _**x**_ ) = bases( _H_ : _,L∪_ _**x**_ ) _,_ _**x**_ _X,_ (A.22)
_∀_ _⊆_


_holds, if and only if the condition_


�bases( _Q_ : _,L_ ) = bases( _H_ : _,L_ ) _,_ _and_
(A.23)
bases( _Q_ : _,L∪{Xi}_ ) = bases( _H_ : _,L∪{Xi}_ ) _,_ _Xi_ _X,_
_∀_ _∈_


_holds._


With Lemma 9, the remaining problem of Phase 2 can be reduced to solving for each singleton
augmentation, formulated as follows:


23


To solve this singleton augmentation problem, one could at worst exhaustively try all 2 _[m]_ possible
fillings. In what follows, however, we present a more efficient, deterministic construction:
**Lemma** **10** ( **Constructing** **a** **particular** **solution** **for** **singleton** **augmentation** ) **.** _Let_ _H_ _∈_
_{_ 0 _,_ 1 _}_ _[m][×]_ [(] _[l]_ [+1)] _be a binary matrix with columns partitioned as_ [ _l_ + 1] = _L ∪{x}._ _Define_

_D_ := _i_ 1 _, . . ., m_ _C_ circuits( _H_ ) : _i_ _C_ _C_ _i_ _/_ Ind( _H_ : _,L_ ) _._ (A.24)
_{_ _∈{_ _} | ∀_ _∈_ _∈_ _⇒_ _\{_ _}_ _∈_ _}_

_Define a new matrix H_ _[′]_ _where H_ : _[′]_ _,L_ [=] _[ H]_ [:] _[,L][ and the column][ x][ is replaced by][ H]_ _i,x_ _[′]_ [= 1] _[ if][ i][ ∈]_ _[D][ and]_
0 _otherwise._ _Then, the whole matroid remains unchanged after this column x replacement:_

bases( _H_ _[′]_ ) = bases( _H_ ) _._ (A.25)


With this result, we complete the final step of the Key Problem defined above and can obtain a binary
matrix _H_ that satisfies all rank constraints.


Proofs of Lemmas 8 to 10 are all given in Appendix B.


To interpret _H_ as a digraph representation, we perform a final row permutation to place nonzeros
along the diagonal. Standard algorithms such as the _n_ -rooks method may be used for the row
permutation. We keep the column indices of _L ∪_ _X_ fixed and reindex the rows to match the same
ordered list _L_ _∪_ _X_ . The resulting matrix encodes a model distributionally equivalent to the underlying
model that general the data. One may then run BFS/DFS using Theorem 3 to obtain the whole
equivalence class.


With all above, we conclude the algorithm part.


B PROOFS OF MAIN RESULTS


Note that we present the proofs in an order that differs slightly from their appearance, arranged
instead according to their logical dependencies.


B.1 PROOFS OF IRREDUCIBILITY RESULTS


The irreducibility results rely on the identifiability of (overcomplete) independent component analysis
(ICA). So we first restate them here.


A linear irreducible ICA model can be described by the equation


_X_ = _AE,_ (B.1)


where _E_ = ( _E_ 1 _,_ _, Em_ ) _[⊤]_ are unknown mutually independent random variables, namely _sources_,
and _X_ = ( _X_ 1 _,_ _· · ·, Xp_ ) _[⊤]_ are observed random variables, namely _mixtures_ . _A_ R _[p][×][m]_, namely the
_· · ·_ _∈_
_mixing matrix_, is constrained to have no pairwise proportional columns (including zero columns).
The tuple ( _A, E_ ) is called an _irreducible ICA representation_ of _X_ .
**Lemma 11** (Identifiability of ICA; (Eriksson & Koivunen, 2004)) **.** _Let_ ( _A, E_ ) _and_ ( _B, S_ ) _be two_
_irreducible ICA representations of a p-dim random vector X, where A ∈_ R _[p][×][m]_ _and B_ _∈_ R _[p][×][n]_ _._ _If_
_every component of E_ _follows a non-Gaussian distribution, then the following properties hold:_


_1._ _m_ = _n._


24


_2._ _Every column of A is proportional to some column of B, and vice versa._


_3._ _Every component of S_ _follows a non-Gaussian distribution_ [1] _._


**Proposition 1** ( **Graphical condition for irreducibility** ) **.** _A model_ ( _G, X_ ) _is irreducible, if and only_
_if for each non-empty set_ _**l**_ _L,_ ch _G_ ( _**l**_ ) _**l**_ 2 _, i.e., it has more than one child outside._
_⊆_ _|_ _\_ _| ≥_


_Proof of Proposition 1._ Due to the identifiability of OICA, irreducibility is equivalent to that there
are no proportional columns in the mixing matrix, which, with Lemma 2, is that


_ρG_ ( _X,_ _**v**_ ) 2 _,_ _**v**_ _V_ with _**v**_ 2 _._ (B.2)
_≥_ _∀_ _⊆_ _|_ _| ≥_


When _**v**_ contains 2 or more vertices from observed _X_, this condition is naturally satisfied. So we
only need to consider _**v**_ that contains at most one vertex from _X_ and at least one vertex from _L_ .


When _**v**_ contains only _L_ vertices, the violation of Equation (B.2) leads to the graphical criterion.
When _**v**_ contains one _X_ vertex, say, _Xi_, and Equation (B.2) is violated, it means the min-cut from _**v**_
to _X_ is simply _Xi_, which implies that the min-cut from the remaining latent vertices _**v**_ _Xi_ to
_X_ is either ∅ or _{_ _X}i_ . This also leads to the graphical criterion. _\ {_ _}_
_{_ _}_


**Proposition 2** ( **Procedure of reduction to the irreducible form** ) **.** _Given any latent-variable model_
_X_
( _G, X_ ) _, the following procedure outputs a digraph H such that H_ _∼G_ _and_ ( _H, X_ ) _is irreducible._

_Step 1._ _Initialize H as G._
_Step 2._ _Remove vertices V_ ( ) an _H_ ( _X_ ) _from_ _, i.e., remove latents who have no effects on X._
_H_ _\_ _H_
_Step 3._ _Identify the maximal redundant latents in the remaining latent vertices:_

mrl := _**l**_ _V_ ( ) _X_ : _**l**_ _>_ 0 _,_ ch _H_ ( _**l**_ ) _**l**_ _<_ 2 _,_ _and_ _**l**_ _[′]_ ⊋ _**l**_ _,_ ch _H_ ( _**l**_ _**[′]**_ ) _**l**_ _**[′]**_ 2 _._ (7)
_{_ _⊆_ _H_ _\_ _|_ _|_ _|_ _\_ _|_ _∀_ _|_ _\_ _| ≥_ _}_

_Step 4._ _For each_ _**l**_ mrl _, let c be the exact child in_ ch _H_ ( _**l**_ ) _**l**_ _; for each parent p_ pa _H_ ( _**l**_ ) _**l**_ _c_ _,_
_∈_ _\_ _∈_ _\_ _\{_ _}_
_add an edge p →_ _c into H if not already present; finally, remove_ _**l**_ _vertices from H._


_Proof of Proposition 2._ This graphical operation directly translates the operation to merge all maximally proportional columns in the mixing matrix into single columns. This ensures the irreducible
_H_ .

Note that by removing maximally redundant latents, the added edges in step 4 will not be removed
later, i.e., for each _**l**_ operated in step 4,


ch _H_ ( _**l**_ ) _**l**_ = 1, and (ch _H_ ( _**l**_ ) pa _H_ ( _**l**_ ) _**l**_ ) ( mrl) = ∅ _._ (B.3)
_|_ _\_ _|_ _∪_ _\_ _∩_ _∪_


This ensures the well-defined graphical operation.


B.2 PROOF FROM RANK EQUIVALENCE TO DISTRIBUTIONAL EQUIVALENCE


**Lemma 3** ( **Equivalence via path ranks** ) **.** _Two irreducible models are distributionally equivalent,_
_X_
_written G_ _∼H, if and only if there exists a permutation π over the vertices V_ ( _G_ ) _, such that_

_ρG_ ( _Z, Y_ ) = _ρH_ ( _Z, π_ ( _Y_ )) _for all Z_ _X_ _and_ _Y_ _V_ ( ) _._ (11)
_⊆_ _⊆_ _G_

1But note that unlike 2., _S_ may still be unreachable from _E_ via only permutation and scaling. In other words,
the model is _identifiable_, but not _unique_ . See Example 2 of Eriksson & Koivunen (2004).


25


_Proof of Lemma 3._ We first show the “ _⇒_ ” direction.

By Lemma 2 there is a generic choice _A ∈A_ ( _G_ ) that realizes the rank structure for all _X_ _Z, Y_ _⊆_ _V_ ( _G_ ):
rank( _AZ,Y_ ) = _ρG_ ( _Z, Y_ ). With Lemma 1 we obtain from _G_ _∼H_ with _A_ _∈A_ ( _G, X_ ) = _A_ ( _H, X_ )
that there is a matrix _B_ _∈A_ ( _H, X_ ), a permutation matrix _P_, and a scaling matrix _D_ such that
_AX,_ : = _BPD_ . Let _π_ be the column-permutation that corresponds to the matrix _P_ .

Now, let _Z_ _⊆_ _X_ and _Y_ _⊆_ _V_ ( _G_ ) = _V_ ( _H_ ), then we have the desired equation
_ρG_ ( _Z, Y_ ) = rank( _AZ,Y_ ) = rank(( _BPD_ ) _Z,Y_ ) = rank(( _BP_ ) _Z,Y_ ) (B.4)
= rank( _BZ,π_ ( _Y_ )) = _ρH_ ( _Z, π_ ( _Y_ )) _._ (B.5)


We then show the “ _⇐_ ” direction.

For any _Z_ _X_, the partial application _ρG_ ( _Z,_ _) is the rank function of a strict gammoid
_⊆_
_MG,Z_ = Γ( _, Z, V_ ( )) defined by the digraph on the ground set _V_ ( ) with the set of termi_G_ _G_ _G_ _G_
nals _Z_ . Analogously, _ρH_ ( _Z,_ _) is the rank function a strict gammoid _MH,Z_ defined by the digraph
_H_
on the same ground set and the same set of terminals.


Equation (11) implies that _π_ is an isomorphism between _MG,Z_ and _MH,Z_, for all _Z_ _X_ simultaneously. Clearly, _π_ is also an isomorphism between the dual matroids _MG_ _[∗]_ _,Z_ [and] _[ M][ ∗]_ _H,Z⊆_ [.]


It follows from the Fundamental Lemma in (Ingleton & Piff, 1973) that the support matrices _Q_ [(] _[G]_ [)] and
_Q_ [(] _[H]_ [)] define isomorphic families of transversal matroids that are represented by the corresponding
row-sub-matrices of all ( _I_ _BG_ ) with sufficiently general weights and _BG_ ( ); and ( _I_ _BH_ )
_−_ _∈B_ _G_ _−_
with sufficiently general weights and _BH_ ( ), respectively.
_∈B_ _H_

In (Brylawski, 1975) it is shown that every transversal matroid may be represented by vectors that
lie in the faces of a simplex such that for every minimal non-trivial combination of the zero vector
by a set of representing vectors, this set lies on a common simplex face with rank one less than
the cardinality of the set of vectors. Over R such vectors can be found almost surely by taking the
incidence matrix of the transversal system and choosing a random value for each nonzero entry. All
matrices over R that represent the transversal matroid can be produced by this procedure.


Choosing random values for the nonzero entries of _Q_ [(] _[G]_ [)] (and _Q_ [(] _[H]_ [)] ) gives almost surely a matrix
that represents the respective family of transversal matroids in such a general simplex position, with
nonzero entries on the diagonal. By row scaling, this matrix can be brought into the desired form
( _I_ _−_ _B_ ) where all diagonal entries are equal to 1. Scaling the columns and rows of a matrix by
nonzero factors does not alter the family of matroids represented by a matrix, and does not change
whether a matrix is in general simplex position.

The matrices in _A ∈A_ ( _G, X_ ) (and _A_ ( _H, X_ )) are row-restrictions of inverses of diagonal-1-scaled
versions of matrices in general simplex position;


_A_ = ( _SR_ ) _[−]_ _X,_ [1] : [=] _[ R]_ _X,_ _[−]_ [1] : _[S][−]_ [1] _[,]_ (B.6)


where _R_ is a randomized valuation of _Q_ [(] _[G]_ [)], and _S_ is a diagonal matrix consisting of the multiplicative
inverses of the diagonal entries of _R_ .


Let _P_ be the permutation matrix for _π_, and let _T_ be the diagonal matrix consisting of the multiplicative
inverses of the diagonal entries of _PR_ . Then there is


_A_ _[′]_ = ( _TPR_ ) _[−]_ _X,_ [1] : (B.7)

_[∈A]_ [(] _[H][, X]_ [)] _[.]_

Since _S_ and _T_ are invertible diagonal matrices, we have


_A_ _[′]_ = _RX,_ _[−]_ [1] : _[S][−]_ [1] _[SP][ −]_ [1] _[T][ −]_ [1] [=] _[ASP][ −]_ [1] _[T][ −]_ [1] [=] _[AP][ −]_ [1][(] _[S][P][ T][ −]_ [1][)] _[,]_ (B.8)

where _SP_ is a diagonal matrix with ( _SP_ ) _i,i_ = _Sπ_ ( _i_ ) _,π_ ( _i_ ).

So _A_ _[′]_ arises from _A_ by permuting and scaling columns, thus _X_ _A_ ( _G, X_ ) = _A_ ( _H, X_ ). Finally, with
Lemma 1 we have _G_ _∼H_ .


26


B.3 PROOFS OF MATROID-PRESERVING COLUMN AUGMENTATION: A CORE COMPONENT


B.3.1 CONSTRUCTING PARTICULAR SOLUTIONS TO A COLUMN AUGMENTATION


We begin with the two auxiliary lemmas introduced in Appendix A, namely, Lemmas 8 and 10.

**Lemma** **8** ( **How** **the** **transversal** **matroid** **changes** **when** **augmenting** **more** **sources** ) **.** _For_ _two_
_binary matrices Q_ 1 0 _,_ 1 _and Q_ 2 0 _,_ 1 _, we denote by_ [ _Q_ 1 _Q_ 2] 0 _,_ 1
_∈{_ _}_ _[m][×][n]_ [1] _∈{_ _}_ _[m][×][n]_ [2] _|_ _∈{_ _}_ _[m][×]_ [(] _[n]_ [1][+] _[n]_ [2][)]
_the matrix obtained by horizontally concatenating Q_ 1 _with Q_ 2 _._ _Then, we have:_
Ind([ _Q_ 1 _Q_ 2]) = _Z_ 1 _Z_ 2 : _Z_ 1 Ind( _Q_ 1) _,_ _Z_ 2 Ind( _Q_ 2) _._ (A.21)
_|_ _{_ _∪_ _∈_ _∈_ _}_
_In other words, two matrices’ matroids sufficiently determine the matroid of their augmentation._


_Proof of Lemma 8._ Straightforward. Let _N_ 1 = [ _n_ 1] and _N_ 2 = [ _n_ 2] be the column indices. For the
“split back into _⊆_ ” direction, we just consider for each _N_ 1 and _N_ 2. For the “ ” direction, _Z_ _∈_ Ind([ _Q Z_ 11 _|Q_ can be matched into2]), its matched sources in _N_ 1 and _N Z_ 1 _∪_ 2 _NZ_ 12 can becan be
_⊇_ _\_
matched into _N_ 2, which, when put together, is still independent, since _N_ 1 and _N_ 2 are disjoint.


**Lemma** **10** ( **Constructing** **a** **particular** **solution** **for** **singleton** **augmentation** ) **.** _Let_ _H_ _∈_
_{_ 0 _,_ 1 _}_ _[m][×]_ [(] _[l]_ [+1)] _be a binary matrix with columns partitioned as_ [ _l_ + 1] = _L ∪{x}._ _Define_

_D_ := _i_ 1 _, . . ., m_ _C_ circuits( _H_ ) : _i_ _C_ _C_ _i_ _/_ Ind( _H_ : _,L_ ) _._ (A.24)
_{_ _∈{_ _} | ∀_ _∈_ _∈_ _⇒_ _\{_ _}_ _∈_ _}_

_Define a new matrix H_ _[′]_ _where H_ : _[′]_ _,L_ [=] _[ H]_ [:] _[,L][ and the column][ x][ is replaced by][ H]_ _i,x_ _[′]_ [= 1] _[ if][ i][ ∈]_ _[D][ and]_
0 _otherwise._ _Then, the whole matroid remains unchanged after this column x replacement:_
bases( _H_ _[′]_ ) = bases( _H_ ) _._ (A.25)


_Proof of Lemma 10._ We show that _H_ and _H_ _[′]_ represent the same transversal matroid by comparing
their independent families.


**In case that** Ind( _H_ : _,L_ ) = Ind( _H_ ) **,** then _D_ is the set of coloops of _H_ : _,L_ . Since every maximal
partial transversal of _H_ : _,L_ already contains each _d_ _D_, the bases of _H_ _[′]_ are precisely the bases of
_∈_
_H_ : _,L_, so the matroids for _H_, _H_ : _,L_, and _H_ _[′]_ are the same.


**Otherwise,** **if** Ind( _H_ : _,L_ ) ⊊ Ind( _H_ ) **,** then _H_ : _,L_ has a maximal partial transversal that can be
extended by some element _e_ with _He,x_ = 1. Because the cardinality of the bases of _H_ is one
more than the cardinality of the bases of _H_ : _,L_, we have that every basis _B_ with respect to _H_ can
be partitioned into a basis _B_ 0 of _H_ : _,L_ and an extra element _b_ _B_ _B_ 0 where _Hb,x_ = 1. Clearly
_∈_ _\_
_B_ 0 Ind( _H_ _[′]_ ) and if _b_ _D_, then _B_ is also a basis for _H_ _[′]_ .
_∈_ _∈_

Assume that _b_ _/_ _D_, then there is a circuit _C_ of _H_ with _b_ _C_ such that _C_ _b_ Ind( _H_ : _,L_ ). The
_∈_ _∈_ _\{_ _}_ _∈_
corresponding partial transversal of _H_ : _,L_ can be extended by sending _ϕ_ ( _b_ ) = _x_, but then this extended
partial transversal proves that _C_ is independent, contradicting that _C_ is a circuit of _H_, so _b ∈_ _D_ must
be the case. Thus Ind( _H_ ) _⊆_ Ind( _H_ _[′]_ ).

Now let _B_ be a basis for _H_ _[′]_ and assume that _B_ _/_ Ind( _H_ ). By set inclusion, _B_ _/_ Ind( _H_ : _,L_ ). So
_∈_ _∈_
there is maximal partial transversal _ϕ_ of _H_ _[′]_ and some _b ∈_ _B_ such that


Hence,


_ϕ_ ( _b_ ) = _x_ and _ϕ_ [ _B\{b}_ ] _⊆_ _L._ (B.9)


_B_ _b_ Ind( _H_ : _,L_ ) Ind( _H_ ) _._ (B.10)
_\{_ _}_ _∈_ _⊆_


Since _B_ _∈/_ Ind( _H_ ), there is a circuit _C_ _⊆_ _B_ with _b ∈_ _C_ . But then _C\{b} ⊆_ _B\{b}_ is independent
in _H_ : _,L_, so _b_ _∈/_ _D_ . This is a contradiction to _Hb,ϕ_ _[′]_ ( _b_ ) [=] _[ H]_ _b,x_ _[′]_ [= 1][, because] _[ ϕ]_ [ is a partial transversal]
of _H_ _[′]_ . Therefore Ind( _H_ _[′]_ ) _⊆_ Ind( _H_ ) establishing the equality Ind( _H_ _[′]_ ) = Ind( _H_ ) which implies
bases( _H_ _[′]_ ) = bases( _H_ ) since bases are precisely the maximal independent sets.


27


B.3.2 TRAVERSING ALL SOLUTIONS TO A COLUMN AUGMENTATION


Now, we have proved the two auxiliary lemmas introduced in Appendix A.


We observe that these two lemmas are centering around one problem, as formulated in the text box
titled **“Problem of Phase 2 (Reduced)”** : suppose we already know both a transversal matroid itself
and the transversal matroid of it after augmented with an unknown singleton column, how can we
recover this unknown singleton column to satisfy these two matroids?


In Lemma 10, we have shown a particular solution. Then, what are all the possible solutions, and
how can we find them? Among all solutions, is there anything special about the particular solution
given in Lemma 10? Are there any other particular solutions that might enjoy other properties?


Let us first define all these solutions:
**Definition 8** ( **Solution set of matroid-preserving column augmentations** ) **.** Let _Q ∈{_ 0 _,_ 1 _}_ _[m][×][n]_
be a binary matrix. For each _x ∈_ [ _n_ ], we denote all column vectors that can be used to replace _Q_ ’s
column _x_ while preserving _Q_ ’s matroid by:


colaug( _Q, x_ ) := _D_ 2 [[] _[m]_ []] : bases([ _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D_ ]) = bases( _Q_ ) _,_ (B.11)
_{_ _⊆_ _|_ _}_

where the name colaug stands for “column augmentation”, 1 _D_ denotes a column vector with ones at
entries in _D_ and zeros elsewhere, and the notation [ _· | ·_ ] denotes matrices’ horizontal concatenation.
Apparently, colaug( _Q, x_ ) is non-empty, since at least the current column _Q_ : _,x_ satisfies the condition,
as well as the particular column 1 _D_ defined in Lemma 10, which may be different from _Q_ : _,x_ .


Before we study how we may traverse all solutions in colaug( _Q, x_ ), let us pay more attention to the
particular solutions, since 1) they offer an efficient way to get a solution directly from the matroids,
without the need to solving alpha systems, which leads to algorithm speedups, and 2) as we will show
below, they have meaningful implications to characterize the whole solution class.


We already have a particular solution from Lemma 10, which checks when a single-element deletion
from each new circuit still leads to dependent sets before augmentation. In other words, these items
are those who contribute to the newly introduced circuits. Following the proof to Lemma 10, we have
that these items forms not only a solution, but also a unique maximal inclusive solution:

**Corollary 1** ( **Determining items that** _**must not**_ **appear in any solution** ) **.** _Let Q ∈{_ 0 _,_ 1 _}_ _[m][×]_ [(] _[l]_ [+1)]
_be a binary matrix with columns partitioned as_ [ _l_ + 1] = _L ∪{x}._ _Let D_ _⊆_ [ _m_ ] _be the particular_
_solution constructed in Lemma 10,_ _and let_ colaug( _Q, x_ ) _⊆_ 2 [[] _[m]_ []] _be all the solutions of x-column_
_augmentation defined in Definition 8._ _Then, the following condition holds:_


_D_ =          - colaug( _Q, x_ ) _._ (B.12)


In other words, by giving a unique maximal solution, Lemma 10 characterizes which items in [ _m_ ]
_can_ be included in some solution(s), or equivalently, which items _must not_ appear in any solution.


This then naturally leads to another question: which are the items that _must_ be included in all solutions,
and more importantly, is there an efficient way to determine them, just like Lemma 10, without having
to traverse the whole solution set? The answer is yes, by taking complements to Lemma 10:


**Corollary 2** ( **Determining items that** _**must**_ **appear in all solutions** ) **.** _Let Q ∈{_ 0 _,_ 1 _}_ _[m][×]_ [(] _[l]_ [+1)] _be a_
_binary matrix with columns partitioned as L ∪{x}._ _We define the “difference in cocircuits” as_

diffcc( _Q, x_ ) := cocircuits( _Q_ ) cocircuits( _Q_ : _,L_ ) _._ (B.13)
_\_

_Further, for anyof A (with a slightly special treat on A ⊆_ 2 _[V]_ _, a set of subsets of some universe_ ∅ _) as:_ _V, we define the minimum-sized elements_

minimum( _A_ ) :=       - _{{_ ∅ _a ∈},_ _A_ : _|a|_ = min _a′∈a |a′|},_ _otherwise.if A ̸_ = ∅ _,_ (B.14)


_The minimal-inclusion elements of A (with a slightly special treat on_ ∅ _) is then:_


28


- _a_ _A_ : ∄ _a′_ _A with a′_ ⊊ _a_ _,_ _if A_ = ∅ _,_
minimal( _A_ ) := _{_ _∈_ _∈_ _}_ _̸_ (B.15)
_{_ ∅ _},_ _otherwise._


_Then, the following conditions always hold:_


_1._ _The minimum-sized new cocircuits are all particular solutions._ _Moreover, they are exactly_
_those minimal-inclusion ones among all solutions, which have a same (minimum) size:_


minimal(colaug( _Q, x_ )) = minimum(colaug( _Q, x_ )) = minimum(diffcc( _Q, x_ )) _._
(B.16)


_2._ _Non-minimum-sized new cocircuits are not solutions, that is,_

(diffcc( _Q, x_ ) _\_ minimum(diffcc( _Q, x_ ))) _∩_ colaug( _Q, x_ ) = ∅ _._ (B.17)


_3._ _The intersection of minimum-sized new cocircuits may not be a solution itself, but it charac-_
_terizes exactly items that must appear in all solutions:_

      - minimum(diffcc( _Q, x_ )) =      - colaug( _Q, x_ ) _._ (B.18)


Roughly speaking, Corollary 2 takes a complement to Corollary 1: for any valid solution, it must
complete new bases (witnessed by minimal new cocircuits), so any item that appears in every such
minimum new cocircuit must appear in all valid solutions.


We use an example to illustrate these definitions above.
**Example 3** ( **Illustration of Definition 8** ) **.** Suppose _Q_ is a matrix with row indices _{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ and
column indices _{α, β, γ}_, as follows.








_Q_ =


_α_ _β_ _γ_
1 0 _×_ 0
2 _×_ 0 0
3 _×_ _×_ _×_
4 _×_ 0 0












bases( _Q_ ) = _{{_ 1 _,_ 2 _,_ 3 _}, {_ 1 _,_ 3 _,_ 4 _}},_
circuits( _Q_ ) = _{{_ 2 _,_ 4 _}},_ (B.19)
cocircuits( _Q_ ) = _{{_ 1 _}, {_ 3 _}, {_ 2 _,_ 4 _}}._


 _,_ then


**Consider the case with** _x_ = _α_ **.** The remaining columns are:








_Q_ : _,{β,γ}_ =


_β_ _γ_
1 _×_ 0
2 0 0
3 _×_ _×_
4 0 0












bases( _Q_ : _,{β,γ}_ ) = 1 _,_ 3 _,_
_{{_ _}}_
circuits( _Q_ : _,{β,γ}_ ) = 2 _,_ 4 _,_ (B.20)
_{{_ _}_ _{_ _}}_
cocircuits( _Q_ : _,{β,γ}_ ) = 1 _,_ 3 _._
_{{_ _}_ _{_ _}}_


 _,_ with


The particular solution (also the maximal unique solution) given by Lemma 10 and Corollary 1 is:


_D_ = _{_ 1 _,_ 2 _,_ 3 _,_ 4 _},_ (B.21)

where 1 and 3 are coloops, and 2 and 4 lead to the new circuits in _Q_ .


The particular solutions given by Corollary 2 are:


minimum(diffcc( _Q, α_ )) := minimum(cocircuits( _Q_ ) cocircuits( _Q_ : _,{β,γ}_ ))
_\_

= minimum( _{{_ 1 _}, {_ 3 _}, {_ 2 _,_ 4 _}} \ {{_ 1 _}, {_ 3 _}}_ )
= minimum( _{{_ 2 _,_ 4 _}}_ ) _,_
= _{{_ 2 _,_ 4 _}}._

In total, there are four possible columns that can replace _Q_ : _,α_ without changing the matroid:


(B.22)


colaug( _Q, α_ ) = _{{_ 2 _,_ 4 _}, {_ 1 _,_ 2 _,_ 4 _}, {_ 2 _,_ 3 _,_ 4 _}, {_ 1 _,_ 2 _,_ 3 _,_ 4 _}} ._ (B.23)

For example, choose _D_ = _{_ 1 _,_ 2 _,_ 4 _} ∈_ colaug( _Q, α_ ), one may verify that


29


(B.24)








Let _Q_ _[′]_ =


_α_ _β_ _γ_
1 _×_ _×_ 0
2 _×_ 0 0
3 0 _×_ _×_
4 _×_ 0 0





 _,_ still we have bases( _Q_ _[′]_ ) = bases( _Q_ ) = _{{_ 1 _,_ 2 _,_ 3 _}, {_ 1 _,_ 3 _,_ 4 _}}._


For now, let us not consider how these whole solutions colaug( _Q, α_ ) are obtained; one may just think
of them as obtained by exhaustively searching over all 2 [4] possible columns.


**Consider the case with** _x_ = _β_ **.** The remaining columns are:








_Q_ : _,{α,γ}_ =


_α_ _γ_
1 0 0
2 _×_ 0
3 _×_ _×_
4 _×_ 0












bases( _Q_ : _,{α,γ}_ ) = 2 _,_ 3 _,_ 3 _,_ 4 _,_
_{{_ _}_ _{_ _}}_
circuits( _Q_ : _,{α,γ}_ ) = 1 _,_ 2 _,_ 4 _,_ (B.25)
_{{_ _}_ _{_ _}}_
cocircuits( _Q_ : _,{α,γ}_ ) = 3 _,_ 2 _,_ 4 _._
_{{_ _}_ _{_ _}}_


 _,_ with


So, the maximal solution (Lemma 10 and Corollary 1) is: _D_ = _{_ 1 _,_ 3 _}_ .


The minimal solutions (Corollary 2) are: minimum(diffcc( _Q, β_ )) = _{{_ 1 _}}_ .


And all the possible solutions are: colaug( _Q, β_ ) = _{{_ 1 _}, {_ 1 _,_ 3 _}}_ .


**Consider the case with** _x_ = _γ_ **.** The remaining columns are:








_Q_ : _,{α,β}_ =


_α_ _β_
1 0 _×_
2 _×_ 0
3 _×_ _×_
4 _×_ 0












bases( _Q_ : _,{α,β}_ ) = 1 _,_ 2 _,_ 1 _,_ 3 _,_ 1 _,_ 4 _,_ 2 _,_ 3 _,_ 3 _,_ 4 _,_
_{{_ _}_ _{_ _}_ _{_ _}_ _{_ _}_ _{_ _}}_
circuits( _Q_ : _,{α,β}_ ) = 2 _,_ 4 _,_ 1 _,_ 2 _,_ 3 _,_ 1 _,_ 3 _,_ 4 _,_
_{{_ _}_ _{_ _}_ _{_ _}}_
cocircuits( _Q_ : _,{α,β}_ ) = 1 _,_ 3 _,_ 1 _,_ 2 _,_ 4 _,_ 2 _,_ 3 _,_ 4 _._
_{{_ _}_ _{_ _}_ _{_ _}}_
(B.26)


 _,_ with


So, the maximal solution (Lemma 10 and Corollary 1) is: _D_ = _{_ 1 _,_ 3 _}_ .


The minimal solutions (Corollary 2) are: minimum(diffcc( _Q, γ_ )) = _{{_ 1 _}, {_ 3 _}}_ .


And all the possible solutions are: colaug( _Q, β_ ) = _{{_ 1 _}, {_ 3 _}, {_ 1 _,_ 3 _}}_ .


_△_


So far, we have introduced the definitions about “matroid-preserving column augmentations”, and
have provided particular ways to construct both the unique maximal solution and the set of minimal
solutions. Then, starting from these particular solutions, or starting from any solution, how can we
span to the whole solution set? This is answered by the following result.


We now present the structure among all column augmentations, which describes how how the whole
solutions can be traversed, and is thus important to our result about equivalence class traversal. In
particular, any two solutions can reach each other by a sequence of “edge additions/deletions”.


**Lemma 12** ( **The whole column augmentations can be traversed by edge additions/deletions** ) **.**
_For any matrix Q ∈{_ 0 _,_ 1 _}_ _[m][×][n]_ _and a column index x ∈_ [ _n_ ] _, we define a digraph termed GQ,x_ [aug] _[, which]_
_is a Hasse diagram with vertices being elements of_ colaug( _Q, x_ ) _, and edges being:_


_Di_ _Dj_ _Q,x_ [⊊] _[D][j]_ _[with][ |][D][j]_ _Di, Dj_ colaug( _Q, x_ ) _._ (B.27)
_→_ _∈G_ [aug] _[⇐⇒]_ _[D][i]_ _[\][ D][i][|]_ [ = 1] _[,]_ _∀_ _∈_


_Then, this digraph is weakly connected._


We illustrate Lemma 12 by recalling Example 3:


30


(a) _Q,α_
_G_ [aug]


_{_ 1 _}_


_{_ 1 _,_ 3 _}_


(b) _Q,β_
_G_ [aug]


(c) _Q,γ_
_G_ [aug]


Figure 4: Example Hasse diagrams (defined in Lemma 12) over the solution sets of matroid-preserving
column augmentations. Instances from the earlier Example 3 are used. In each diagram, the root
vertices, corresponding to the minimal solutions (see Corollary 2) are highlighted in red, while the
leaf vertex, corresponding to the unique maximal solution (see Corollary 1) are highlighted in blue.


_Proof of Lemma 12._ Let


_D_ = _i_ 1 _, . . ., m_ _C_ circuits( _Q_ ) : _i_ _C_ _C_ _i_ _/_ Ind( _Q_ : _,_ [ _n_ ] _\{x}_ ) _._ (B.28)
_{_ _∈{_ _} | ∀_ _∈_ _∈_ _⇒_ _\{_ _}_ _∈_ _}_

From Lemma 10, we know _D_ _∈_ colaug( _Q, x_ ).

Let _D_ _[′]_ _∈_ colaug( _Q, x_ ), then _D_ _[′]_ _⊆_ _D_, because if there is _d_ _[′]_ _∈_ _D_ _[′]_ with _d_ _[′]_ _∈/_ _D_, then there exists a
circuit _C_ in _Q_ with _d_ _[′]_ _∈_ _C_ such that _C\{d_ _[′]_ _}_ has a partial transversal omitting the column _x_ . This
partial transversal may be extended to _C_ by sending _d_ _[′]_ to the column _x_, which then can be extended
to a basis _BC_ _C_ in the transversal matroid represented by ( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′_ ). But _Q_ cannot have a
_⊇_
basis that contains one of its circuits, which implies that _D_ _[′]_ _∈/_ colaug( _Q, x_ ). So _D_ is the maximal
element of colaug( _Q, x_ ).

Letbases(( _D_ _[′]_ _Q∈_ : _,_ [ _n_ colaug(] _\{x}_ 1 _DQ, x′_ )). The maximal partial transversal) with _D_ _[′]_ = _D_ and let _d_ _∈ ϕD_ that witnesses the independence of _\D_ _[′]_ . Let _B_ _∈_ bases( _Q_ ), then _BB_ with _∈_
respect to ( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′_ ) also witnesses the independence of _B_ with respect to ( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′∪{d}_ ).
So we have


bases( _Q_ ) = bases(( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′_ )) bases(( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′∪{d}_ )) _._ (B.29)
_⊆_


Now, let _B_ bases(( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′∪{d}_ )). The maximal partial transversal witnessing _B_ with
_∈_
respect to ( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′∪{d}_ ) is also a maximal partial transversal with respect to ( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D_ ),
so _B_ bases(( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D_ )) = bases( _Q_ ). Therefore,
_∈_

bases(( _Q_ : _,_ [ _n_ ] _\{x}_ 1 _D′∪{d}_ )) = bases( _Q_ ) _,_ and _D_ _[′]_ _d_ colaug( _Q, x_ ) _._ (B.30)
_∪{_ _} ∈_


Hence, for every _D_ 0 _, D_ 1 colaug( _Q, x_ ) there is a directed path to _D_ which gives
_∈_


_D_ 0 _D_ _D_ 1 _,_ (B.31)
_→· · · →_ _←· · · ←_

Therefore, _GQ,x_ [aug] [is weakly connected.]


B.3.3 FROM ONE COLUMN AUGMENTATION TO MULTIPLE COLUMNS’ JOINT
AUGMENTATION


We have now shown how one can obtain particular solutions or traverse all solutions to a single
column augmentation. In what follows, we shift from one column augmentation to multiple column
augmentation, which directly relates to our final graphical criterion to be shown in next section: we
need to augment the whole _X_ columns, identifying their outgoing edges.


Interestingly, this seemingly combinatorially complex satisfiability problem can be decomposed
locally, that is, it suffices to satisfy each singleton column augmentation independently, shown below.


31


**Lemma 9** ( **Reducing all union equivalence checks to singleton checks** ) **.** _Let Q, H_ _∈{_ 0 _,_ 1 _}_ _[m][×][n]_
_be two binary matrices with columns partitioned as_ [ _n_ ] = _L ∪_ _X_ _for a fixed L._ _Then, the condition_

bases( _Q_ : _,L∪_ _**x**_ ) = bases( _H_ : _,L∪_ _**x**_ ) _,_ _**x**_ _X,_ (A.22)
_∀_ _⊆_


_holds, if and only if the condition_


�bases( _Q_ : _,L_ ) = bases( _H_ : _,L_ ) _,_ _and_
(A.23)
bases( _Q_ : _,L∪{Xi}_ ) = bases( _H_ : _,L∪{Xi}_ ) _,_ _Xi_ _X,_
_∀_ _∈_


_holds._


_Proof of Lemma 9._ Clearly, condition A.23 is a special case of condition A.22 with the choices
_**x**_ ∅ _,_ _X_ 1 _, . . .,_ _Xk_, thus if A.22 holds, then so does A.23.
_∈{_ _{_ _}_ _{_ _}}_

If A.22 does not hold, then there is some _**x**_ _X_ such that bases( _Q_ : _,L∪_ _**x**_ ) = bases( _H_ : _,L∪_ _**x**_ ). W.l.o.g.
_⊆_ _̸_
we may assume that there is some _B_ bases( _Q_ : _,L∪_ _**x**_ ) with _B_ _/_ bases( _H_ : _,L∪_ _**x**_ ).
_∈_ _∈_


**If** _|_ _**x**_ _| ≤_ 1 **,** then A.23 clearly does not hold.


**Now assume that** _**x**_ _>_ 1 **.** Choose _B_ bases( _Q_ : _,L∪_ _**x**_ ) bases( _H_ : _,L∪_ _**x**_ ) and a partial transversal
_|_ _|_ _∈_ _\_
_ϕ_ : _B_ _→_ _L ∪_ _**x**_ for _Q_ : _,L∪_ _**x**_ such that _δ_ := _|{b ∈_ _B_ _| Hb,ϕ_ ( _b_ ) = 0 _}|_ is minimal. Due to the minimality
of _δ_ (may be obtained via basis exchange), there is exactly one _b_ _B_ such that _Hb,ϕ_ ( _b_ ) = 0. Let
_∈_

_B_ _[′]_ = _{b_ _[′]_ _∈_ _B_ _| ϕ_ ( _b_ _[′]_ ) _∈_ _L ∪{ϕ_ ( _b_ ) _},_ (B.32)

then _B_ _[′]_ is a basis of _Q_ : _,L∪{ϕ_ ( _b_ ) _}_, but _B_ _[′]_ is not a basis of _H_ : _,L∪{ϕ_ ( _b_ ) _}_, shown below:

Assume that _B_ _[′]_ is a basis of _H_ : _,L∪{ϕ_ ( _b_ ) _}_, then there is a partial transversal _ψ_ : _B_ _[′]_ _L_ _ϕ_ ( _b_ ) .
_→_ _∪{_ _}_
Construct


_ϕ_ _[′]_ : _B_ _H_ : _,L∪_ _**x**_ (B.33)
_→_


such that


            - _ψ_ ( _b′_ ) for _b_ _[′]_ _B_ _[′]_ _,_
_ϕ_ _[′]_ ( _b_ _[′]_ ) = _∈_ (B.34)
_ϕ_ ( _b_ _[′]_ ) for _b_ _[′]_ _∈_ _B\B_ _[′]_ _._

_ϕ_ _[′]_ is a partial transversal, because


_ϕ_ [ _B\B_ _[′]_ ] _∩_ ( _L ∪{ϕ_ ( _b_ ) _}_ ) = ∅ _,_ (B.35)

due to the definition of _B_ _[′]_ . Thus if _ϕ_ _[′]_ ( _b_ 0) = _ϕ_ _[′]_ ( _b_ 1). Then:

1 _[◦]_ Either _b_ 0 _, b_ 1 _B_ _[′]_ : in this case, _ϕ_ _[′]_ ( _b_ 0) = _ψ_ ( _b_ 0) = _ψ_ ( _b_ 1) = _ϕ_ _[′]_ ( _b_ 1) implies _b_ 0 = _b_ 1;
_{_ _} ⊆_

2 _[◦]_ Otherwise we have _b_ 0 _, b_ 1 _B_ _B_ _[′]_, and then _ϕ_ _[′]_ ( _b_ 0) = _ϕ_ ( _b_ 0) = _ϕ_ ( _b_ 1) = _ϕ_ _[′]_ ( _b_ 1) implies
_{_ _}_ _⊆_ _\_
_b_ 0 = _b_ 1.

But then _ϕ_ _[′]_ witnesses that _B_ is a basis of _H_ : _,L∪_ _**x**_, contradicting the original assumption. Thus
bases( _Q_ : _,L∪{ϕ_ ( _b_ ) _}_ = bases( _H_ : _,L∪{ϕ_ ( _b_ ) _}_ and A.23 does not hold, too.


We have then finished proof to Lemma 9.


B.4 PROOFS OF THE GRAPHICAL CRITERION AND TRANSFORMATIONAL CHARACTERIZATION


We first note that the graphical criterion (Theorem 2) is a direct consequence of Lemma 9, that is,
instead of checking for bases of all subsets _**x**_ _⊆_ _X_, we only need to check for bases for each singleton
_Xi_ _X_ . Since Lemma 9 is already proved above, in this section, we focus on the proof of the
_∈_
transformational characterization (Theorem 3).


32


**Lemma 7** ( **Admissible edge additions/deletions** ) **.** _Let_ ( _G, X_ ) _be an irreducible model._ _For any_
_edge Vi_ _Vj_ _not currently in_ _, adding it to_ _preserves equivalence on X_ _if and only if:_
_→_ _G_ _G_


_rG_ ( _Vi’s nonchildren_ _Vj_ _,_ _L_ _Vi_ ) _<_ _rG_ ( _Vi’s nonchildren,_ _L_ _Vi_ ) _,_ (20)
_\{_ _}_ _\{_ _}_ _\{_ _}_


_where_ _Vi’s nonchildren_ _denotes_ _V_ ( _G_ ) _\_ ch _G_ ( _Vi_ ) _\{Vi},_ _i.e.,_ _zero_ _entries_ _in_ _support_ _column_ _Q_ [(] : _,V_ _[G]_ [)] _i_ _[.]_
_Conversely, an edge can be deleted if and only if it can be re-added by this criterion._


_Proof of Lemma 7._ Let us first prove a weaker version of this result, that is, without the permutation
part involved in checking equivalence (Lemma 5). Put formally, let _H_ be the digraph after altering an
edge _Vi_ _Vj_ in . We study the if and only if condition (in terms of this edge) for the following
_→_ _G_


_rG_ ( _Z, Y_ ) = _rH_ ( _Z, Y_ ) for all _Z_ _V_ ( ) and _L_ _Y_ _V_ ( ) (B.36)
_⊆_ _G_ _⊆_ _⊆_ _G_


to hold. According to Lemma 9, this condition holds if and only if a reduced version hold:


bases( _Q_ [(] : _,L_ _[G]_ [)] _∪{Vi}_ [)] [=] [bases(] _[Q]_ [(] : _,L_ _[H]_ [)] _∪{Vi}_ [)] _[.]_ (B.37)


That is, one only need to check whether a single transversal matroid is changed. Then, when can
an edge in a bipartite graph be altered while the transversal matroid induced by this bipartite graph
keeps unchanged? We show the condition by the following lemma.


**Lemma 13** ( **When an edge in a bipartite graph can be altered without changing the transversal**
**matroid** ) **.** _Let Q_ 0 _,_ 1 _be a binary support matrix._ _For any_ ( _Vj, Vi_ ) [ _m_ ] [ _n_ ] _such that_
_∈{_ _}_ _[m][×][n]_ _∈_ _×_
_QVj_ _,Vi_ = 0 _,_ _define_ _H_ 0 _,_ 1 _by_ _HVj_ _,Vi_ = 1 _and_ _Hz,y_ = _Qz,y_ _for_ _all_ _other_ _entries._ _For_
_∈{_ _}_ _[m][×][n]_
_convenience, denote Vi’s non-children in Q and column indices except for Vi_ _by:_

_R_ := _{z_ _∈_ [ _m_ ] : _Qz,Vi_ = 0 _}_ ; (B.38)
_Y_ := [ _n_ ] _Vi_ _._
_\{_ _}_

_Then, the following conditions are equivalent to each other:_


_1._ bases( _Q_ ) = bases( _H_ ) _;_


_2._ bases( _QR,_ :) = bases( _HR,_ :) _;_


_3._ mrank( _QR,_ :) = mrank( _HR,_ :) _;_


_4._ mrank( _QR\{Vj_ _},Y_ ) _<_ mrank( _QR,Y_ ) _, that is, Vj_ _is a coloop among R in the transversal_
_matroid induced by QR,Y, and so removing it from ground set lowers the rank (by_ 1 _)._


_Proof of Lemma 13._ We first have two immediate observations. (i) By construction, _QR,{Vi}_ is the
zero column, whereas _HR,{Vi}_ has a single 1 in row _Vj_ . (ii) For any _Z_ [ _m_ ] with _Vj_ _/_ _Z_, the
_⊆_ _∈_
submatrices _QZ,_ : and _HZ,_ : coincide, hence their matching ranks (and base behavior) coincide.


We now prove the implications among the four conditions by

(1) _⇒_ (2) _⇒_ (3) _⇐⇒_ (4) _⇒_ (2) _⇒_ (1) _._

(1) _⇒_ (2) **.** Trivial. Taking restrictions on ground sets preserves equality of matroids.

(2) _⇒_ (3) **.** Trivial. Same matroids have the same ranks.

(3) _⇐⇒_ (4) **.** Let _ν_ := mrank( _QR,Y_ ) and _ν_ _[′]_ := mrank( _QR\{Vj_ _},Y_ ). Note that

mrank( _QR,_ :) = mrank( _QR,Y_ ) = mrank( _HR,Y_ ) = _ν,_ (B.39)


because the column _Vi_ is useless (full zero) for _R_ in _Q_ . In _H_, the only new edge incident to _R_ is
( _Vi, Vj_ ); therefore any matching on _R_ in _H_ is either:


    - a matching that _does not use_ column _Vi_, hence has size at most _ν_, or


33


- a matching that _does use_ the edge ( _Vi, Vj_ ) and then matches the remaining _R_ _Vj_ into _Y_,
_\ {_ _}_
hence has size at most 1 + _ν_ _[′]_ .


Consequently,
mrank( _HR,_ :) = max _ν,_ 1 + _ν_ _[′]_ _._ (B.40)
_{_ _}_
Thus mrank( _HR,_ :) = mrank( _QR,_ :) holds if and only if 1 + _ν_ _[′]_ _ν_, i.e., _ν_ _[′]_ _< ν_, which is precisely
_≤_
(4). This proves (3) _⇐⇒_ (4).

(4) _⇒_ (2) **.** Assume (4). In the transversal matroid _M_ induced by _QR,Y_, the inequality
mrank( _QR\{Vj_ _},Y_ ) _<_ mrank( _QR,Y_ ) means that _Vj_ is a _coloop_ of _M_ (see Definition 7). A standard
matroid identity for coloops states that for all _Z_ _R_ _Vj_,
_⊆_ _\ {_ _}_

mrank( _QZ∪{Vj_ _},Y_ ) = mrank( _QZ,Y_ ) + 1 _._ (B.41)

Combining this with Equation (B.40) (applied now to _each_ _Z_ _⊆_ _R_ ) shows that adding the edge
( _Vi, Vj_ ) cannot change the matching rank of _any Z_ _R_ ; so in particular, the bases on _R_ is unchanged:
_⊆_
bases( _QR,_ :) = bases( _HR,_ :).


(2) (1) **.** For any _Z_ [ _m_ ], we write _ZR_ := _Z_ _R_ and _Z_ out := _Z_ _R_ .
_⇒_ _⊆_ _∩_ _\_


    - If a maximum matching of _HZ,_ : _does not use_ ( _Vi, Vj_ ), then it is also a matching in _QZ,_ :,
and the ranks agree.


    - If a maximum matching of _HZ,_ : _does use_ ( _Vi, Vj_ ), then its restriction to _ZR_ is a maximum
matching of _HZR,_ : that uses the column _Vi_ . By (2) (which holds for all subsets of _R_ as
shown above), there exists a maximum matching of _QZR,_ : of the _same_ size that avoids _Vi_ .
Replacing the _H_ -matching on _ZR_ by this _Q_ -matching on _ZR_ (and keeping the _Z_ out-part
unchanged) yields a matching of _QZ,_ : of the same cardinality as the original one in _HZ,_ :.


Hence mrank( _QZ,_ :) = mrank( _HZ,_ :) for all _Z_ [ _m_ ], which is equivalent to bases( _Q_ ) = bases( _H_ ).
_⊆_


All implications are proved, so the four conditions are equivalent. The result on deleting an edge is
just the same as adding back this edge from the resulted graph.


The condition shown in Lemma 13 is exactly the condition we have in Lemma 7, and hence the
weaker version without permutation (Equation (B.36)) is already proved.


The prove the full version, we only need to show that when the condition in Equation (B.37) fails,
then with any permutation they still cannot be rendered equivalent. This is straightforward, since
with one edge difference, the independent sets Ind( _Q_ [(] : _,L_ _[G]_ [)] _∪{Vi}_ [)] [and] [Ind(] _[Q]_ [(] : _,L_ _[H]_ [)] _∪{Vi}_ [)][,] [if] [not] [equal,]
must admit a strict inclusion relation between them, so there is no way for these two matroids to be
isomorphic.


We have now finished the proof of Lemma 7.


**Theorem 3** ( **Transformational characterization of the equivalence class** ) **.** _Two irreducible models_
( _G, X_ ) _and_ ( _H, X_ ) _are equivalent if and only if G_ _can be transformed into H, up to L-relabeling, via_
_a sequence of admissible cycle reversals and edge additions/deletions, as defined in Lemmas 6 and 7._

_Here, “up to L-relabeling” means there exists a relabeling of L in H yielding a digraph H_ _[′]_ _such_
_that G_ _reaches H_ _[′]_ _via the sequence._ _Moreover, at most one cycle reversal is needed in this sequence._


_Proof of Theorem 3._ The proof to this result for traversing the equivalence class for the whole class
directly relates to the helpful lemmas we have shown in Appendix B.3, i.e., how the whole solution
set of column(s) augmentation is structured.


34


Lemma 12 is core to our result: it shows that the whole space of satisfiable column augmentations
can be traversed by applying sequences of “one-edge different” operations, and these operations
are exactly the “admissible edge additions/deletions” we show in Lemma 7 (for edges from _X_ ),
and Lemma 13 (for edges from _L_ ). From Lemma 12, we also have a way to traverse all bipartite graphs
that realizes a given transversal matroid. This can be viewed as a generalization from single-column
to multi-column augmentation:


**Corollary** **3** ( **Traverse** **all** **bipartite** **graphs** **that** **realize** **a** **transversal** **matroid** ) **.** _Let_ _Q, H_ _∈_
_{_ 0 _,_ 1 _}_ _[m][×][n]_ _be two binary matrices._ _Q and H_ _induce a same transversal matroid, i.e.,_ bases( _Q_ ) =
bases( _H_ ) _, if and only if Q can reach H_ _via a sequence of admissible edge additions/deletions defined_
_in_ _Lemma_ _13,_ _followed_ _by_ _a_ _column_ _permutation._ _Moreover,_ _similar_ _to_ _Corollary_ _1,_ _among_ _all_
_matrices that can be reached from Q via sequences of edge additions/deletions, there exists a unique_
_maximal matrix whose support is the union of supports of all these reachable matrices._


Corollary 3 directly relates to our traversal on the _Q_ [(] : _,L_ _[G]_ [)] [part.] [But it is worth noting that unlike the]
independent decomposition of column augmentations for each _Xi_, here the edge additions/deletions
have to be operated within the whole matrix space. We cannot simply run column augmentation for
each _Li_ and take the Cartesian product. To see this, let _Q_ = [[1 _,_ 1]] with columns _α, β_ . Obviously,
colaug( _Q, α_ ) = _{_ ∅ _, {_ 1 _}}_, and also colaug( _Q, β_ ) = _{_ ∅ _, {_ 1 _}}_ . However, we cannot take a product
and let _Q_ _[′]_ = [[0 _,_ 0]], which induces a matroid different from _Q_ .


Now, putting Lemma 12, Corollary 3, and Lemma 9 together, we have a way to traverse all digraphs
that achieve the same matroids over the tower of all sources that include some latent vertices:


**Corollary 4** ( **Traverse all digraphs that realize same matroid tower under latents** ) **.** _Let Q, H_ _∈_
_{_ 0 _,_ 1 _}_ _[m][×][n]_ _be two binary matrices with columns partitioned as_ [ _n_ ] = _L ∪_ _X._ _Then, the condition_

bases( _Q_ : _,L∪_ _**x**_ ) = bases( _H_ : _,L∪_ _**x**_ ) _,_ _**x**_ _X,_ (B.42)
_∀_ _⊆_

_holds, if and only if the_ bases( _Q_ : _,L_ ) = bases( _H_ : _,L_ ) _, (graphical criterion in Corollary 3), and for all_
_Xi_ _X,_ _i_ [ _m_ ] : _Qi,Xi_ = 1 colaug( _H_ : _,L∪{Xi}, Xi_ ) _(graphical criterion in Lemma 12)._
_∈_ _{_ _∈_ _}_ _∈_


In other words, _Q_ can reach _H_ via a sequence of admissible edge additions/deletions, followed by a
permutation only among the columns in _L_ . Corollary 4 directly relates to our traversal on the _Q_ [(] _[G]_ [)] .


If we are to allow the equivalence up to row permutation, i.e., permuting the ground set as in Lemma 5,
only a row permutation appended to the end of the operations in Corollary 4 is needed.


Finally, a treatment to ensure nonzero diagonals for digraphs.


    - Since in our case we need to exclude matrices with zero diagonals, this row permutation
becomes the “at most one step” within the sequence (Theorem 3), instead of at the end.


    - The _L_ -relabeling part, however, can still be put at the end, since to relabel the _L_ vertices in
a digraph _G_, it is to apply a permutation on the columns _Q_ [(] : _,L_ _[G]_ [)] [first, and then to apply the]

same permutation back on the rows _Q_ [(] _L,_ _[G]_ : [)][.] [This operation still ensures the nonzero diagonals.]
This becomes the “up to _L_ -relabeling” term in Theorem 3.


We have now finished the proof of Theorem 3.


B.5 OTHER IMMEDIATE OR KNOWN RESULTS


We omit the proofs of the remaining results occurred in this manuscript: some of them follow
immediately from the already proved results, including Lemmas 1, 4 and 5, and the others are results
shown by existing work, including Lemmas 2 and 6 and Theorem 1.


35


C DISCUSSION


C.1 SUMMARY: A SIDE-BY-SIDE COMPARISON BETWEEN PATH RANKS AND EDGE RANKS


Table 1: A side-by-side comparison between path ranks and edge ranks.


**Aspect** **Path rank (Matrix rank)** **Edge rank (Matching rank)**


Intuition Algebraic independence Combinatorial independence

Full rank of a _d_ _×_ _d_ rank( _M_ ) = _d_ _⇐⇒_ the _determi-_ mrank( _M_ ) = _d_ _⇐⇒_ the _perma-_
square matrix _M_ _nant_ of _M_ is nonzero _nent_ of _M_ is nonzero


Graphical constraints in _ρG_ ( _Z, Y_ ), the maximum number of
digraphs vertex-disjoint directed paths from
_Y_ to _Z_ (Definition 3), equals the
matrix rank of generic mixing submatrices _AZ,Y_ (Lemma 2)


_rG_ ( _Z, Y_ ), the size of the maximum
bipartite matching from _Y_ to _Z_ via
direct edges (Definition 4), equals
the matching rank of the support
submatrix _Q_ [(] _Z,Y_ _[G]_ [)] [(Lemma 4)]


Matroid representations Strict gammoids in digraphs (Per- Transversal matroids in bipartite
fect, 1968) graphs (Ingleton & Piff, 1973)


Duality (Theorem 1) min( _Z_ _,_ _Y_ ) _ρG_ ( _Z, Y_ ) = _V_ max( _Z_ _,_ _Y_ ) _rG_ ( _V_ _Y,_ _V_ _Z_ )
_|_ _|_ _|_ _|_ _−_ _|_ _| −_ _|_ _|_ _|_ _|_ _−_ _\_ _\_


C.2 ANOTHER EXAMPLE DISTRIBUTIONAL EQUIVALENCE CLASS


3 4


2


7
_G_


6


10


9
_G_


1
_G_


5
_G_


8
_G_


|G<br>1 L<br>1<br>X X<br>1 2<br>X<br>3|G<br>2 L<br>1<br>X X<br>1 2<br>X<br>3|G<br>3 L<br>1<br>X X<br>1 2<br>X<br>3|G<br>4 L<br>1<br>X X<br>1 2<br>X<br>3|
|---|---|---|---|
|_G_5<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_6<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_7<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_7<br>_L_1<br>_X_1<br>_X_3<br>_X_2|
|_G_8<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_9<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_10<br>_L_1<br>_X_1<br>_X_3<br>_X_2|_G_10<br>_L_1<br>_X_1<br>_X_3<br>_X_2|


Figure 5: Left: An example distributional equivalence class consisting of 10 digraphs. Right:
Transitions among these digraphs, where solid edges indicate edge additions or deletions, and dashed
edges indicate cycle reversals.


We show another example distributional equivalence class in Figure 5, in addition to the Figure 3
already shown in main text. The points of this example, different from those of Figure 3, are that:


1. Partitioned by cycle reversals (removing the dashed edges in the right of Figure 5), the classes
connected by only edge additions/deletions (solid edges) are not necessarily isomorphic to
each other. Here, there are 3 _,_ 3 _,_ 4 digraphs within each such class, respectively.


2. To illustrate cases where cycles intersect.


36


C.3 A PRESENTATION OF THE EQUIVALENCE CLASS


In the main text, we have presented both a graphical criterion to check for equivalence (Theorem 2)
and a transformational characterization to traverse the entire equivalence class (Theorem 3). These
results are analogous to the “same adjacencies and v-structures” and the “covered edge reversal
(Meek conjecture)” in the fully observed, acyclic, Markov equivalence setting.


However, note that in that classical setting, there is another familiar result, CPDAG, which serves
as an informative presentation of the equivalence class. This naturally raises the question: can we
construct an analogous presentation in the context of this work? We answer this affirmatively. In
what follows, we outline how this presentation can be constructed step by step.


**Step** **1.** **Identifiability** **of** **ancestral** **relations** **among** **observed** **variables.** As a preliminary
observation, we first note that the ancestral relations among observed variables _X_ are invariant across
all equivalent digraphs (this follows from how admissible edge additions/deletions are defined, and
the fact that cycle reversal does not alter the ancestral relations). Thus, presenting an arbitrary digraph
in the equivalence class suffices to inform users the true ancestral relations among _X_ . For applications
such as experimental design involving observed variables, this alone is informative enough.


**Step 2.** **Unique maximal digraph within the class.** We show that under each cycle-reversal configuration, there exists a unique maximal digraph in the equivalence class such that every equivalent
digraph is a subgraph of it. We further provide an explicit construction of this maximal digraph
(Corollary 1), without needing to enumerate all equivalent digraphs and then take the maximal one.
Analogous to the “largest chain graph” in Frydenberg (1990), this maximal digraph can serve as a
basis of the presentation, informing users which causal relations are _guaranteed to be absent_ .


**Step 3.** **Characterizing edges that must appear.** Building on the previous step, we also characterize edges that must be present in _all_ equivalent digraphs. Again, these edges can be determined
efficiently via a graphical condition (Corollary 2), without needing to enumerate all equivalent
digraphs and then take the intersection. Note that it is an explicit construction, so iterative procedures
(such as arrow propagation in Meek rules (Meek, 1995)) are not needed either. These edges can be
visually highlighted on the basis maximal digraph, in analogous to the arrows in CPDAGs, or “visible
edges” in PAGs (Zhang, 2008b). They inform users which causal relations _they can fully trust_ .


Such presentation is formally defined in Theorem 4, and examples of it are shown in Figure 6.


Figure 6: Illustrative presentations of equivalence classes. **Left:** Presentation of equivalent digraphs
_G_ 1 _, G_ 2 _, G_ 3 under a same cycle-reversal configuration from Figure 3. The basis digraph shown is the
unique maximal equivalent digraph (Step 2 above). In it, solid edges denote those that must appear
in all equivalent digraphs, while dashed edges are those that can be removed (Step 3 above). One
may use Corollaries 1 and 2 to check how they are determined. **Right:** A similar presentation for
digraphs 3 _,_ 4 _,_ 7 _,_ 10 from Figure 5. **Remark:** One might ask why we present the equivalence
_G_ _G_ _G_ _G_
class separately for each cycle-reversal configuration rather than for the entire class. The reason is
that taking the union over all digraphs in the entire class can, unlike within one configuration, yield a
supergraph that is itself out of the equivalence class, potentially producing misleading interpretations.
In fact, this separation only leads to more informative presentations, shown by Theorem 4 below.


**Theorem 4** ( **Presentation of an equivalence class** ) **.** _For an irreducible model_ ( _G, X_ ) _, we construct a_
_digraph whose vertices are V_ ( _G_ ) _and whose directed edges come in two types:_ _solid and dashed._ _All_
_edges are determined by Corollary 1, and among them the solid ones are determined by Corollary 2._
_We denote this digraph by_ CP( _G_ ) _, echoing the sense of “complete partial” as in CPDAGs._


37


_For convenience, letX E_ ( _G_ ) _be the whole equivalence class, that is, the set of all digraphs H on vertices_
_V_ ( _G_ ) _such that H_ _∼G._ _Let F_ ( _G_ ) _denote the set of all digraphs reachable from G_ _via sequences of_
_admissible edge addition/deletions as defined in Lemma 7._ _Clearly, F_ ( _G_ ) _⊆E_ ( _G_ ) _._

_Then, the presentation_ CP( _G_ ) _enjoys the following properties:_

_1._ CP( _G_ ) _∈F_ ( _G_ ) _;_

_2._ _For every H ∈F_ ( _G_ ) _, the edge set of H is a subset of the edge set of_ CP( _G_ ) _;_

_3._ _The intersection of the edge sets of all H ∈F_ ( _G_ ) _equals the solid edges of_ CP( _G_ ) _;_

_4._ _For every H ∈E_ ( _G_ ) _, let_ CP( _H_ ) _be its own presentation._ _Then,_ CP( _H_ ) _can be transformed_
_into_ CP( _G_ ) _via an L-relabel and a cycle reversal (alongside the solid/dashed edge types)._


It is worth noting that a dashed edge in a presentation means that there exists at least one equivalent
digraph without this edge. However, it does not imply that dashed edges can be arbitrarily removed
without affecting equivalence: they have to obey the rank constraints. This is in a similar spirit of
undirected edges in a CPDAG: an undirected edge means that there exist at least two equivalent
DAGs who have different orientations on this edge. However, it does not imply that undirected edges
can be arbitrarily oriented: there are additional constraints like no new v-structures, no cycles, etc.


We are not sure whether such additional constraints can, or should, be also incorporated into the
presentation, or at least summarized as a set of rules like Meek rules (especially given the availability
of an interactive traversal tool). But in any case, we put it here as a possible future step:


**Step 4.** **Quantifying bounds on edges between vertex groups.** Extending step 3, one may describe
bounds on the number of edges between vertex groups (e.g., “at least 2 and at most 4 edges from
vertices _Y_ to vertices _Z_ ”). Such constraints may be presented like “underlined bows” in cyclic
digraphs (Richardson, 1996) or “hyperedges” in mDAGs (Evans, 2016). We have not developed this
result (though we hypothesize that they likely also follow from Theorem 2), because we are not sure
how much practical informativeness it can offer to users.


Lastly, we also list the presentation of prior knowledge as a future step.


**Step** **5.** **Incorporating** **additional** **prior** **knowledge.** As with other equivalence presentations,
prior knowledge such as acyclicity, stable cycles, or certain causal orderings can further refine the
equivalence class and its presentation (Perkovi´c et al., 2017). While we have not explored this part
either, it motivates future theoretical developments, such as interventional equivalence classes, and
parameter identifiability results based on the equivalence class established in this work.


C.4 EXAMPLES OF NON-RANK CONSTRAINTS IN MIXING MATRICES


In Lemma 3 we have shown that path rank equivalence in mixing matrices sufficiently lead to
distributional equivalence. However, this does not imply that there are no other constraints in mixing
matrices. As an analogy, in the causally sufficient linear Gaussian system, CI equivalence (zero
partial correlations in covariance matrices) sufficiently lead to distributional equivalence, but there
are still other constraints, like the Tetrad constraints in the covariance matrices.


Below we give an example of non-rank constraints in mixing matrices.

Consider a digraph _G_ with 4 vertices:


38


Its mixing matrix is:








_A_ =


1 2 3 4
1 1 _bc_ _c_ _ce_
2 _a_ 1 _−_ _cde_ _ac_ _ace_
3 _ab_ + _de_ _b_ 1 _e_
4 _d_ _bcd_ _cd_ 1 _−_ _abc_





1
 _×_ 1 _abc_ _cde_ _[.]_ (C.1)
_−_ _−_


We can verify the following constraint holds:


_A_ 2 _,_ 4 _A_ 3 _,_ 2 _A_ 4 _,_ 1 _−_ _A_ 2 _,_ 1 _A_ 3 _,_ 4 _A_ 4 _,_ 2
= _ace × b × d_ _−_ _a × e × bcd_ (C.2)
= 0 _._


Just like rank constraints, this constraint is also immune to arbitrary column scaling, that is, it also
survives in the OICA estimated mixing matrix. However, this is not a rank constraint.


One may also verify some other non-rank constraints in this _A_, for example,


_A_ 2 _,_ 2 _A_ 3 _,_ 4 _A_ 4 _,_ 1 + _A_ 2 _,_ 4 _A_ 3 _,_ 1 _A_ 4 _,_ 2 + _A_ 2 _,_ 1 _A_ 3 _,_ 2 _A_ 4 _,_ 4
(C.3)
= 2 _A_ 2 _,_ 1 _A_ 3 _,_ 4 _A_ 4 _,_ 2 _−_ _A_ 2 _,_ 2 _A_ 3 _,_ 1 _A_ 4 _,_ 4 _,_

and

_A_ [2] 2 _,_ 4 _[A]_ [3] _[,]_ [1] _[A]_ [3] _[,]_ [2] _[A]_ [4] _[,]_ [2] [+] _[ A]_ [2] _[,]_ [1] _[A]_ [2] _[,]_ [2] _[A]_ [2] 3 _,_ 4 _[A]_ [4] _[,]_ [2] [+] _[ A]_ [2] _[,]_ [1] _[A]_ [2] _[,]_ [4] _[A]_ [2] 3 _,_ 2 _[A]_ [4] _[,]_ [4]
(C.4)
= 2 _A_ 2 _,_ 1 _A_ 2 _,_ 4 _A_ 3 _,_ 2 _A_ 3 _,_ 4 _A_ 4 _,_ 2 + _A_ 2 _,_ 2 _A_ 2 _,_ 4 _A_ 3 _,_ 1 _A_ 3 _,_ 2 _A_ 4 _,_ 4 _,_


both of which are also immune to column scaling.


We are not sure whether there are any specific geometry interpretations underlying these equality
constraints. These examples are brutal-force searched from ideal elimination.

We notice that these equality constraints occur among the _{_ 2 _,_ 3 _,_ 4 _}_ rows, meaning that when vertex
_{_ 1 _}_ is latent and _{_ 2 _,_ 3 _,_ 4 _}_ observed, these constraints will also appear in the OICA mixing matrix. Fortunately, with Lemma 3, we know rank constraints alone can determine the distributional
equivalence, so the equivalence among these constraints as well.


For example, one may verify that these constraints also occur in all 3 digraphs in the equivalence
class, shown below, while this equivalence class is obtained only by the rank-based criterion (which
is trivial in this case since only cycle reversals are applied).


Note that the nice result of Lemma 3 only occurs at the linear non-Gaussian case, where path ranks
are one-sided, so that it can be directly dualized to a transversal matroid that can be represented by
vectors that lie in the faces of some simplex.


In the linear Gaussian case, with the two-sided path ranks in covariance matrices, there can be
more constraints. In that setting, however, rank constraints equivalence do not necessarily imply
distributional equivalence: there can be other unmatched equality constraints, e.g., the Pentad, Hexad
constraints and beyond (Drton et al., 2007), let alone other inequality constraints.


C.5 RELATED WORK


**Equivalence characterizations** We first review various approaches to characterize equivalence
of causal models. At the same time, we summarize the multiple results developed in this work and
situate them within this broader landscape of related literature.


39


Table 2: A side-by-side overview of representative works on equivalence characterizations across
different settings using different approaches. The final column summarizes this work’s contributions.


**Markov equivalence in**
**Settings** **fully observed acyclic**
**graphs**


**Markov equivalence in**
**acyclic graphs with**
**latents**


**Distributional**
**equivalence in LiNG**
**models with latents and**
**cycles (this work)**


“Same path/edge ranks up
**Level 1** “Same d-separations” “Same d-separations” to permutation”
(Lemmas 3 and 5)


“Same bases in children
for _L_ itself and with each
singleton _Xi_, up to
permutation” (Theorem 2)


“Unique maximal
equivalent graph with
edges that must always
appear, up to cycle
reversal” (Theorem 4)


“Admissible edge
additions/deletions and
cycle reversals”
(Theorem 3)


BFS/DFS by admissible
transformations
(Theorem 3), with
additional parallel speedup
by column decomposition
(Lemmas 9 and 12)


“Same FCI outputs”
(Spirtes & Verma, 1992);
“Same MAG adjacencies,
v-structures, and colliders
on discriminating paths”
(Spirtes & Richardson,
1996; Richardson &
Spirtes, 2002); “Same head
and tails” (Hu & Evans,
2020)


“Arrowhead completeness”
(Ali et al., 2005); “Full
completeness of PAG
orientations” (Zhang,
2008a); With background
knowledge (still
incomplete; (Andrews
et al., 2020;
Venkateswaran & Perkovi´c,
2024))


“Covered edge reversal”
(Zhang & Spirtes, 2005;
Tian, 2005; Ogarrio et al.,
2016; Claassen & Bucur,
2022)


MAGs traversal within one
PAG (Wang et al., 2024;
2025)


**Level 2**


**Level 3**


**Transfor-**
**mational**


**Traversal**
**algorithms**


“Same adjacencies and
minimal complexes/
v-structures” (Frydenberg,
1990; Verma & Pearl,
1991)


“Maximal (deflagged)
chain graphs; essential
graphs” (Frydenberg,
1990; Andersson et al.,
1997; Roverato et al.,
2006); “CPDAGs” (Spirtes
& Glymour, 1991; Meek,
1995)


“Covered edge reversal
(Meek conjecture)”
(Chickering, 1995; Meek,
1997; Chickering, 2002);
“Weakly covered edge
reversal” (Markham et al.,
2022) for unconditional
equivalence


DAGs traversal within one
CPDAG (Meek, 1995;
Chickering, 1995;
Wienöbst et al., 2023);
CPDAGs traversal
(Steinsky, 2003; Chen
et al., 2016)


In general, approaches to characterize equivalence can be categorized into three types:


1. **Structural characterizations,** which provide conditions for determining equivalence between given graphs, and give rise to summary presentations. However, they do not directly
lead to equivalence class traversal methods. They can be further stratified by their complexity,
informativeness, or purpose, as follows:


40


- **Level 1.** Graphical conditions necessary and sufficient for determining equivalence, but
more as definitions than practical criteria; usually require combinatorial complexities.

       - **Level 2.** Practical graphical criteria for determining equivalence; still necessary and
sufficient, but more efficient than Level 1.

       - **Level 3.** Sound and complete conditions or presentations that summarize the equivalence class. While Level 2 criteria efficiently determine equivalence, they do not fully
capture what can be identified; Level 3 addresses this gap. Level 3 can also be used for
determining equivalence, but it will be less efficient than Level 2.


2. **Transformational characterizations,** which provide natural ways for traversing the equivalence class, and are useful for developing score-based algorithms. However, as a complement, they are not suited for directly determining equivalence between given graphs, or for
developing summary presentations to the equivalence class.

3. **Traversal algorithms,** for enumerating, sampling, or counting elements of the equivalence
class, where transformational characterizations are usually helpful.


We present in Table 2 a side-by-side overview of representative prior works and this work across these
approaches in different settings. This unified view may help to better understand the contributions of
this work, and as well to clarify the methodological implications among these approaches.


There is also a wide range of additional work characterizing equivalences under many other settings,
which are not put in Table 2 due to space limit. These include efforts to develop Markov properties
and establish Markov equivalence in fully observed models with cycles, such as in linear Gaussian
settings (Richardson, 1996; Claassen & Mooij, 2023), discrete settings (Pearl & Dechter, 1996), and
general nonlinear settings (Spirtes, 1994; Forré & Mooij, 2017; Mooij & Claassen, 2020), as well as
nonlinear settings with latent variables and selection bias (Yao & Mooij, 2025). The distributional
equivalence of fully observed linear Gaussian cyclic models has been studied in Ghassami et al.
(2020); Drton et al. (2025b). Nonparametric equivalence with latent variables (usually referred to as
semi-Markov models) has also been characterized in (Evans, 2018; Markham & Grosse-Wentrup,
2020; Jiang & Aragam, 2023; Richardson et al., 2023), beyond the original CI constraints (Richardson
& Spirtes, 2002; Zhang, 2008a). When interventions or multi-domain data are involved, Markov
equivalence (typically as CIs in each domain and invariant changes across domains) has been studied
extensively in (Tian & Pearl, 2001; Eberhardt et al., 2005; Eberhardt & Scheines, 2007; Eberhardt,
2008; He & Geng, 2008; Hauser & Bühlmann, 2012; 2015; Zhang et al., 2015; Rothenhäusler et al.,
2015; Meinshausen et al., 2016; Magliacane et al., 2016; Ghassami et al., 2017; Wang et al., 2017;
Yang et al., 2018; Kocaoglu et al., 2019; Huang et al., 2020; Mooij et al., 2020; Squires et al., 2020;
Zhang et al., 2023; Dai et al., 2025a; Luo et al., 2025; 2026; Yao et al., 2025), with various focuses
spanning across latent variables, selection bias, unknown intervention targets, active experimental
design, and so on. From a method view, transformational characterizations have gained increasing
attention recently, including (Ghassami et al., 2020; Markham et al., 2022; Wang et al., 2024; 2025;
Johnson & Semnani, 2025; Améndola et al., 2025).


Below, we then provide a more comprehensive review of the relevant literature on latent-variable
causal discovery, in particular those under the linear non-Gaussian models (Shimizu et al., 2006).


**Parametric settings for latent-variable causal discovery** A prosperous line of statistical tools
beyond conditional independencies have been developed. These include rank constraints (Sullivant
et al., 2010; Spirtes et al., 2000) and more general equality constraints (Drton, 2018) in the linear
Gaussian setting; and high-order moment constraints (Xie et al., 2020; Adams et al., 2021; Robeva
& Seby, 2021; Dai et al., 2022; 2024; Chen et al., 2024a; Xia et al., 2026), which exploit nonGaussianity for identifiability. In addition to these, matrix decomposition methods (Anandkumar
et al., 2013), copula-based constraints (Cui et al., 2018), and mixture oracles (Kivva et al., 2021)
were also developed.


**Algorithms for latent-variable causal discovery** Building on these statistical tools, many latent
variable causal discovery algorithms have been proposed. Many of them fall within the constraintbased framework, by using CI tests and algebraic constraints to infer causal relations. Examples
include those based on rank or tetrad constraints (Silva et al., 2003; 2006; Silva & Scheines, 2004;


41


Choi et al., 2011; Kummerfeld & Ramsey, 2016; Huang et al., 2022; Dong et al., 2024; 2025). Recent
efforts have also attempted to formalize score-based methods for latent-variable causal discovery
(Jabbari et al., 2017; Ng et al., 2024; Dong et al., 2026).


**Linear non-Gaussian models** Thanks to the strong identifiability results given by OICA, the linear
non-Gaussian models have received much attention for causal discovery with latent variables or cycles:
(Améndola et al., 2023; Salehkaleybar et al., 2020; Wang & Drton, 2023; Maeda & Shimizu, 2020;
Silva & Shimizu, 2017; Dai et al., 2024; Yang et al., 2022; 2024; Shimizu, 2022; Drton et al., 2025a;
Liu et al., 2021; Schkoda et al., 2024; Tramontano et al., 2022; Rothenhäusler et al., 2015), together
with those discussed in §1, and many more. Beyond structure learning, LiNG models also provide
benign conditions for many other tasks, including causal effect identification (Tchetgen Tchetgen et al.,
2024; Kivva et al., 2023; Xie et al., 2022; Tramontano et al., 2024; 2025), model selection (Schkoda
& Drton, 2025), covariate selection (Zhang & Wiedermann, 2024), experimental design (Sharifian
et al., 2025), etc.


Below, we also discuss how the results in this work, especially the edge rank tools and the motivation
of a bipartite matching view, may be generalized to other parameter settings.


For the linear Gaussian setting, existing results in the literature can be directly translated into our
edge rank language. Unlike the non-Gaussian setting where the mixing matrix is identifiable, in the
Gaussian setting, only the covariance matrix is available. The graphical characterization of covariance
matrix ranks, known as “trek-separation,” has been established by Sullivant et al. (2010). Specifically,
the concept of a bottleneck, which we term the “path rank” on the one-sided directed paths, is
extended to the bottleneck along the two-sided directed paths, known as “treks". Since the duality
between path ranks and edge ranks hold universally in graphs regardless of the parametric setting,
the existing characterization on trek-based path ranks can be directly translated into trek-based edge
rank language. As for the technical roadmap, one may first note that the non-Gaussian equivalence
condition we build in this work is necessary but not sufficient for the Gaussian setting. That is, two
graphs that are equivalent in non-Gaussian models are guaranteed to remain equivalent in Gaussian
models; however, graphs that are distinguishable under non-Gaussianity may collapse into the same
equivalence class under Gaussianity. We see the closing of this gap as the most immediate future
direction for extending our current work.


For the discrete setting, our results are likely generalizable as well. Several recent works have explored
path ranks in graphs from discrete data (Gu & Xu, 2023; Chen et al., 2024b; 2025; Lyu & Yang,
2025), where the algebraic counterpart becomes the tensor ranks in the contingency table (Teicher,
1967). However, a precise graphical characterization, analogous to “trek-separation” in the Gaussian
case above, has yet to be developed. That said, such a characterization is promising, since the
linear Gaussian and discrete models behave similarly in many aspects. For example, both are closed
under marginalization and conditionalization; both admit a correspondence between Markov and
distributional equivalence, in both cyclic and acyclic cases (Geiger & Meek, 1996; Pearl & Dechter,
1996). Motivated by these parallels already noted in literature, we believe our results can also
extend to the discrete setting, and will be directly applicable once the corresponding graphical
characterization is developed.


For nonlinear or even nonparametric settings, theoretical generalization remains possible. When
the model is partially linear and partially nonlinear, low-dimensional bottlenecks in the linear
component remain directly observable through covariance ranks (Spirtes, 2013). When the model
is fully nonlinear or even nonparametric, there also exists prior results on the identifiability of
latent-variable models (Hu, 2008; 2017). Although the techniques differ, the underlying motivation
remains closely related to ranks, particularly those in the Jacobian matrix. However, despite the
theoretical meaningfulness, the practical estimation and reliable testing of these ranks remain an open
challenge. This challenge can be echoed by viewing rank constraints as generalizations of conditional
independence constraints. In linear models, conditional independencies correspond to low ranks in
the covariance matrix and can be directed tested via Fisher’s Z test. In contrast, robust conditional
independence tests in nonlinear settings are still under active development (Duong & Nguyen, 2024;
Yang et al., 2025).


42


D EVALUATION RESULTS


D.1 QUANTIFYING THE SIZES OF EQUIVALENCE CLASSES


43


D.2 ASSESSING GLVLING ALGORITHM’S RUNTIME


Table 4: Running time comparison between our glvLiNG algorithm and a mixed integer linear
programming (MILP) baseline for constructing digraphs that satisfy the rank constraints of oracle
OICA mixing matrices. Ground-truth graphs are generated from the Erd˝os–Rényi model with total
number of vertices _n_ and average in-degree avgdeg, with _ℓ_ vertices randomly designated as latent.
Each entry reports the mean and standard deviation over 50 models (when completed); empty entries
indicate runs that did not finish within 10 minutes. All times are reported in seconds. Experiments
were run on an Apple M4 chip.


_n_ _ℓ_ avgdeg MILP glvLiNG glvLiNG Phase 1 glvLiNG Phase 2


1 0.045 0.013 0.015 0.005 0.014 0.005 0.001 0.000
5 1 _±_ _±_ _±_ _±_
3 0.112 _±_ 0.008 0.020 _±_ 0.002 0.019 _±_ 0.002 0.001 _±_ 0.000

1 0.101 0.045 0.098 0.044 0.002 0.000
1 _±_ _±_ _±_
3 0.169 0.024 0.165 0.024 0.004 0.000
7 _±_ _±_ _±_


9


11


13


1 37.402 0.000 0.048 0.013 0.045 0.012 0.003 0.001
3 _±_ _±_ _±_ _±_
3 0.083 _±_ 0.014 0.075 _±_ 0.013 0.007 _±_ 0.001

1 0.691 0.304 0.687 0.303 0.004 0.001
1 _±_ _±_ _±_
3 1.129 _±_ 0.191 1.122 _±_ 0.190 0.007 _±_ 0.001

1 0.319 0.091 0.308 0.090 0.009 0.001
3 _±_ _±_ _±_
3 0.667 _±_ 0.132 0.634 _±_ 0.128 0.031 _±_ 0.007

1 0.082 0.023 0.075 0.022 0.005 0.001
5 _±_ _±_ _±_
3 0.260 _±_ 0.057 0.230 _±_ 0.051 0.023 _±_ 0.005

1 3.637 1.533 3.630 1.532 0.007 0.000
1 _±_ _±_ _±_
3 7.706 _±_ 0.883 7.693 _±_ 0.881 0.013 _±_ 0.002

1 2.174 0.499 2.142 0.500 0.030 0.002
3 _±_ _±_ _±_
3 4.979 _±_ 0.763 4.873 _±_ 0.748 0.102 _±_ 0.021

1 0.530 0.111 0.492 0.111 0.032 0.002
5 _±_ _±_ _±_
3 2.348 _±_ 0.426 2.170 _±_ 0.396 0.159 _±_ 0.045

1 22.838 9.667 22.827 9.667 0.011 0.001
1 _±_ _±_ _±_
3 38.173 _±_ 7.725 38.155 _±_ 7.721 0.017 _±_ 0.005

1 12.501 4.602 12.404 4.593 0.094 0.016
3 _±_ _±_ _±_
3 23.517 _±_ 5.807 23.362 _±_ 5.751 0.150 _±_ 0.062

1 4.277 1.430 4.069 1.433 0.190 0.009
5 _±_ _±_ _±_
3 13.150 _±_ 4.009 12.376 _±_ 3.712 0.723 _±_ 0.310


44


D.3 BENCHMARKING EXISTING METHODS UNDER ORACLE INPUTS


Table 5: Evaluation of existing methods under possible model misspecification on arbitrary latentvariable models. Ground-truth graphs are generated from the Erd˝os-Rényi model with total number
of vertices _n_ and average in-degree avgdeg, with _ℓ_ vertices randomly designated as latent. Only
irreducible models are chosen. Each entry reports the mean and standard deviation of the structural
Hamming distances (SHDs) between the result and truth over 50 random models.
Algorithms are provided with their oracle tests, that is, for them to directly query oracle generalized
independent noise (GIN) conditions from the digraph. When the number of their identified latent
variables is fewer than truth, we simply add isolated latent variables into the result. When the
identified number of latents is larger (which seems not happened), we planned to choose the removal
that leads to best result. Finally, the best possible result is reported, i.e., we choose the digraph in
the ground-truth equivalence class that is closer to their output as the truth. The latent variables are
viewed as unlabeled.


_n_ _ℓ_ avgdeg PO-LiNGAM LaHiCaSl


1 15.48 _±_ 2.75 31.64 _±_ 4.62
2 24.30 _±_ 4.41 36.80 _±_ 4.22
3 35.40 _±_ 3.61 39.96 _±_ 4.88
4 45.22 _±_ 3.53 41.04 _±_ 3.81

1 18.04 _±_ 3.99 32.36 _±_ 4.38
2 28.44 _±_ 3.48 36.68 _±_ 4.31
3 39.18 _±_ 3.60 40.42 _±_ 4.40
4 50.00 _±_ 3.45 41.00 _±_ 4.57

1 31.10 _±_ 4.87 74.22 _±_ 8.04
2 48.26 _±_ 6.67 76.80 _±_ 6.66
3 64.12 _±_ 5.59 81.64 _±_ 8.28
4 84.60 _±_ 5.73 85.56 _±_ 8.61

1 35.02 _±_ 5.74 71.70 _±_ 6.94
2 54.84 _±_ 6.04 78.44 _±_ 7.01
3 72.96 _±_ 6.62 79.98 _±_ 8.15
4 92.28 _±_ 7.71 82.50 _±_ 7.35

1 36.44 _±_ 5.63 71.34 _±_ 8.37
2 58.00 _±_ 6.24 77.90 _±_ 8.25
3 79.90 _±_ 7.34 79.16 _±_ 8.79
4 101.04 _±_ 5.90 84.56 _±_ 8.99

1 48.60 _±_ 6.12 129.88 _±_ 14.18
2 76.92 _±_ 8.29 136.36 _±_ 13.05
3 103.04 _±_ 8.30 138.24 _±_ 11.98
4 129.14 _±_ 10.46 146.86 _±_ 13.02

1 54.04 _±_ 5.23 122.78 _±_ 12.67
2 84.10 _±_ 8.44 136.04 _±_ 12.85
3 115.72 _±_ 8.72 139.76 _±_ 13.64
4 146.44 _±_ 9.10 141.64 _±_ 10.75

1 58.70 _±_ 6.36 128.46 _±_ 11.96
2 92.86 _±_ 8.41 132.48 _±_ 12.36
3 124.00 _±_ 9.60 140.76 _±_ 12.31
4 155.48 _±_ 7.83 143.76 _±_ 11.55

1 64.40 _±_ 7.38 120.08 _±_ 12.43
2 98.04 _±_ 10.43 135.24 _±_ 12.75
3 134.16 _±_ 10.34 137.52 _±_ 12.30
4 167.86 _±_ 11.28 142.76 _±_ 11.39


45


10


15


20


3


5


3


5


7


3


5


7


9


D.4 EVALUATING GLVLING’S PERFORMANCE WITH EXISTING METHODS IN SIMULATIONS


**OICA estimation part** We first describe how we handle the OICA estimation part.


For our choice of OICA implementation, we have tried multiple options and find that overall, the
MATLAB implementation [2] of SDP-ICA (Podosinnikova et al., 2019) tends to provide best estimated
mixing matrices across multiple settings. We thus adopt it in our experiments.


For the number of latent variables, although theoretically identifiable, existing OICA implementations still require specifying this number as an input. Hence, following the common practice as
in (Salehkaleybar et al., 2020), we test multiple candidate values and select the one minimizing the
loss on a held out set.


**Handling empirical ranks in an OICA matrix** We then explain how we process the mixing matrix
estimated from OICA. Having obtained an OICA-estimated mixing matrix, the core task of glvLiNG
becomes constructing a bipartite graph to realize the rank patterns in this mixing matrix, which define
a transversal matroid. When OICA is not an oracle, these empirical ranks may violate matroid axioms,
just like how conditional independencies in data may violate a graphoid in nonparametric settings.


To address this, in our implementation, we assign a “full-rank confidence score” to each relevant block
of _A_ . Specifically, let _σ_ min be _A_ ’s minimum singular value, we use the score 1+exp( _−α_ 1( _σ_ min _−ϵ_ )) [, and]
in experiments, we set _α_ = 25 and _ϵ_ = 0 _._ 02. Then, in phase 1 (recovering latent outgoing edges),
we approximate the closest valid transversal matroid that maximizes agreement with these scores.
In phase 2 (recovering observed outgoing edges), for efficiency we simply threshold these scores to
determine each variable’s outgoing edges independently. We have simulated and verified that this
procedure is robust to moderately noisy ranks, by e.g., assigning true full-rank blocks scores from
_N_ (0 _._ 75 _,_ 0 _._ 2) and others from _N_ (0 _._ 25 _,_ 0 _._ 2), both 0 _,_ 1 truncated.


**Simulation** **setup** In simulation, we compare glvLiNG with existing methods including LaHiCaSl [3] (Xie et al., 2024) and PO-LiNGAM [4] (Jin et al., 2024). We generate random Erd˝os-Rényi
model with total number of vertices _n_ from 5 to 13, number of latent variables _ℓ_ from 1 to 5, average
in-degree _d_ of 1 and 3, and sample size _N_ from 1 _,_ 000 to 200 _,_ 000. We sample data with linear causal
weights uniformly from [ _−_ 2 _._ 5 _, −_ 0 _._ 5] _∪_ [0 _._ 5 _,_ 2 _._ 5], and exogenous noise are sampled from a uniform
distribution [ _−_ 0 _._ 5 _,_ 0 _._ 5], following (Podosinnikova et al., 2019). We calculate the minimum SHD
between all graphs in the true equivalence class to the discovery output graph as the SHD result.


**Simulation results** The results are presented in Figure 7. From it we have the following observations:


First of all, it is not surprising to see that LaHiCaSl and PO-LiNGAM perform better when the
graph is sparser. For example, when _d_ = 1, these two methods perform better than glvLiNG, though
the difference remains modest. This is perhaps because, when the graph is sparser, maintaining
irreducibility typically means more edges outgoing from latent variables, while edges from observed
ones to others are fewer. This aligns well with the model assumptions of these two methods. For
example, LaHiCaSl assumes a hierarchical latent-variable model in which all observed variables
are leaf nodes. Given this additional benefit from their sparsity constraints, and the fact that both
LaHiCaSl and PO-LiNGAM estimates ranks using the GIN condition which is more efficient than
OICA, it is not surprising to see that they perform better in this setting.


However, when the graph is denser, glvLiNG performs particularly better. For example, when _d_ = 3,
glvLiNG consistently outperforms the other two methods, and the difference is considerable. This is
perhaps because, when the graph is denser, more complex structures become common, including arbitrary edges between latent and observed variables, as well as cycles. Model assumptions of existing
methods are more likely to be violated, making them less effective at recovering these structures. In
contrast, with a structural-assumption-free design, glvLiNG avoids such model misspecification, and
still allows the recovery of these structures.


2https://github.com/gilgarmish/oica
3https://github.com/jinshi201/LaHiCaSl
4https://github.com/Songyao-Jin/PO-LiNGAM


46


glvLiNG LaHiCaSl PO-LiNGAM


15


10


5


20


10


30


20


10


40


20


60


40


20


_ℓ_ = 1 _,_ _푑_ = 1 _ℓ_ = 3 _,_ _푑_ = 1 _ℓ_ = 5 _,_ _푑_ = 1 _ℓ_ = 1 _,_ _푑_ = 3 _ℓ_ = 3 _,_ _푑_ = 3 _ℓ_ = 5 _,_ _푑_ = 3


10 [3] 10 [4] 10 [5]


10 [3] 10 [4] 10 [5] 10 [3] 10 [4] 10 [5] 10 [3] 10 [4] 10 [5] 10 [3] 10 [4] 10 [5] 10 [3] 10 [4] 10 [5]

Sample size _푁_ (log scale)


Figure 7: Simulation results comparing glvLiNG with existing methods with varying sample size
_N_ (the global x-axis), and each subplot shows a setting under a specific number of total variables _n_,
number of latent variables _ℓ_, and the average in-degree _d_ . Mean and standard deviation of SHD are
calculated from 25 random irreducible models.


We also observe that glvLiNG tends to be more robust to latent dimensionality. For example, when
_n_ = 13 and _d_ = 3, increasing the number of latents _ℓ_ from 1 to 5, the average SHD of glvLiNG
increases from 33.1 to 35.7, while other method, such as PO-LiNGAM, increases from 35.3 to 50.7.
This is perhaps because glvLiNG jointly recovers all latent-outgoing edges at once, using an OICA
mixing matrix whose dimensionality is already fixed. In contrast, the other two methods requires
incrementally clustering and adding latent variables.


D.5 ANALYZING A REAL-WORLD DATASET WITH GLVLING ALGORITHM


For the real-world experiment, we use a Hong Kong stock market dataset that involves the daily
dividend/split-adjusted closing prices for 14 major stocks from January 4, 2000 to June 17, 2005
(1331 samples). These 14 stocks represent the dominant sectors of the market: 3 of them are on
banking (HSBC Holdings, Hang Seng Bank, Bank of East Asia), 5 on real estate (Cheung Kong,
Henderson Land, Hang Lung Properties, Sun Hung Kai Properties, Wharf Holdings), 3 on utilities
(CLP Holdings, HK & China Gas, HK Electric), and 3 on commerce (Hutchison, Swire Pacific ’A’,
Cathay Pacific Airways). All of them were constituents of Hang Seng Index (HSI), and they were
almost the largest companies of the Hong Kong stock market at the time.


By applying glvLiNG on this dataset, we recovered an equivalence class of causal graphs containing
2 latent variables. The presentation (see Appendix C.3) of this equivalence class is shown in Figure 8.
Here is a summary: the class consists of 19,008 causal graphs with 16=14+2 vertices, and among
them the numbers of edges range between 29 to 34. In the presentation, there are 20 “solid” (must
appear) and 14 “dashed” (may appear) edges.


This result suggests several interesting observations, as follows:


47


Figure 8: Presentation of the equivalence class that glvLiNG estimates from the stock market data.
Different colors of nodes indicate different sectors. Solid and dashed edges indicate edges that must
appear in all or at least one equivalent graph(s).


1. Large banks seem to be major upstream causes. For example, the two largest banks, HSBC
Holdings and Hang Seng Bank, together form a 2-cycle that has 9 children across sectors,
but there are no edges into them.


2. Real estates, in contrast, seem to be downstream effect receivers. For example, Cheung
Kong has 10 parents, but only 1 edge pointing out from it.


3. Utilities are heavily involved in cycles. For example, among 17 simple cycles in the graph,
CLP Holdings belongs to 11 of them. These cycles are often across sectors as utilities - real
estate - commerce - utilities.


4. One latent variable seems interpretable. It has one parent HSBC Holdings, and three
children (all with solid edges): Cheung Kong, Hutchison, and Swire Pacific ’A’. Among
them, Cheung Kong and Hutchison were two core holdings of a same group.


5. Stocks under the same sector tend to be connected more closely.


48