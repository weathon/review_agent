# REDUCING SYMMETRY INCREASE IN EQUIVARIANT NEURAL NETWORKS


**Ning Lin, Jiacheng Cen, Anyi Li, Wenbing Huang** _[∗]_ **, Hao Sun** _[∗]_
Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China
{ninglin00, jiacc.cn, li_anyi}@outlook.com,
{hwenbing, haosun}@ruc.edu.cn


ABSTRACT


Equivariant Neural Networks (ENNs) have empowered numerous applications
in scientific fields. Despite their remarkable capacity for representing geometric
structures, ENNs suffer from degraded expressivity when processing symmetric
inputs: the output representations are invariant to transformations that extend beyond the input’s symmetries. The mathematical essence of this phenomenon is that
a symmetric input, after being processed by an equivariant map, experiences an
increase in symmetry. While prior research has documented symmetry increase in
specific cases, a rigorous understanding of its underlying causes and general reduction strategies remains lacking. In this paper, we provide a detailed and in-depth
characterization of symmetry increase together with a principled framework for
its reduction: (i) For any given feature space and input symmetry group, we prove
that the increased symmetry admits an infimum determined by the structure of the
feature space; (ii) Building on this foundation, we develop a computable algorithm
to derive this infimum, and propose practical guidelines for feature design to prevent harmful symmetry increases. (iii) Under standard regularity assumptions, we
demonstrate that for _most_ equivariant maps, our guidelines effectively reduce symmetry increase. To complement our theoretical findings, we provide visualizations
and experiments on both synthetic datasets and the real-world QM9 dataset. The
results validate our theoretical predictions.


1 INTRODUCTION


Equivariant Neural Networks (ENNs) have
become a cornerstone of modern machine learning, empowering numerous
applications in scientific fields ranging
from molecular dynamics to materials design (Bronstein et al., 2021; Huang & Cen,
2026). By building in physical symmetries,
these models achieve remarkable data efficiency and generalization capabilities when
representing complex geometric structures.


(a) 3-fold (b) 4-fold

Figure 1: _k_ -fold structures.


Despite their success, ENNs exhibit a critical vulnerability when processing symmetric inputs:
their expressivity can degrade, leading to a loss of information. This phenomenon, which we term
**symmetry increase**, occurs when the output representation becomes invariant to transformations that
are not symmetries of the original input itself. A canonical example arises when processing _k_ -fold
symmetric structures. These objects, visualized in Fig. 1, possess a specific dihedral symmetry, yet
an ENN will map their distinct rotated versions to an identical feature, erasing their orientation.


This type of degradation has been documented in previous work. Empirically, it has been observed
that the degradation depends on the feature space, particularly for symmetric inputs (Joshi et al., 2023).
Theoretically, research on ENNs has shown that for _k_ -fold symmetries, selecting only low-degree


_∗_ Corresponding authors.


1


features can cause all rotated inputs to collapse into a single representation (Cen et al., 2024). This
ENN-specific issue is a modern manifestation of a general phenomenon observed in other fields. It
has been linked to physical principles such as Curie’s Principle (Smidt et al., 2021) and described
using the concept of orbit types (Kaba & Ravanbakhsh, 2023).


However, existing analyses provide an incomplete picture and lack a predictive framework. In our
analysis, this degradation of _k_ -fold caused by symmetry increase can be categorized into three distinct
types: _full degeneration_, _axial degeneration_, and _half degeneration_ (see Fig. 2) [1] . The work of Joshi
et al. (2023), while empirically important, does not theoretically explore the cause of the degradation.
The _collapse-to-zero_ theory proposed by Cen et al. (2024) addresses only full degeneration, which is
the most extreme case identified by our analysis. The broader principles discussed by Smidt et al.
(2021) and Kaba & Ravanbakhsh (2023) lack a rigorous mathematical description, and the solutions
proposed, such as in Kaba & Ravanbakhsh (2023), often involve relaxing the equivariance constraint
itself, rather than providing a solution within the equivariant framework.


In this paper, we fill this gap by providing a comprehensive mathematical characterization of symmetry
increase. Our main contributions are briefly listed as follows:


   - In § 3, we prove for any given feature space and input symmetry group, that the increased
symmetry is bounded from below by a unique symmetry infimum, which is determined entirely
by the algebraic structure of the feature space.

   - In § 4, we develop a computable algorithm to derive this infimum by analyzing the orbit types.
This provides practical guidelines for predicting and controlling potential symmetry increases,
thereby preventing harmful symmetry increases in feature design.

   - In § 5, we demonstrate that under regularity conditions, such as the manifold hypothesis for
data, our method can fully reduce symmetry increase. Specifically, for _most_ equivariant maps
or for ENNs with sufficient approximation capabilities, the output symmetry will be precisely
this predictable infimum, preventing orientational information loss.

   - In § 6, we complement our theoretical findings with empirical evidence. We provide visualizations to illustrate the proposed concepts and present experimental results on both synthetic
datasets and the real-world QM9 dataset, which validate our theoretical predictions and demonstrate the practical effectiveness of our framework.


2 PRELIMINARIES


**Group action and representation.** Consider the action of a group _G_ on a set _X_, denoted by _ρX_ .
This action is a map that assigns to each element _g_ _G_ a transformation _ρX_ ( _g_ ) : _X_ _X_, such that
_∈_ _→_
_ρX_ ( _g_ 1 _g_ 2) = _ρX_ ( _g_ 1) _ρX_ ( _g_ 2). We call such a set _X_ a _G_ -set. In particular, if _X_ is a vector space and
_ρX_ ( _g_ ) is a linear transformation for all _g_ _G_, we call _X_ a _G_ -representation [2] .
_∈_

**Equivariant map.** For maps between two _G_ -sets _X_ and _Y_, an equivariant map is one that respects
the group action, meaning that the output transforms accordingly when the input is transformed.
Formally, a map _f_ : _X_ _→_ _Y_ is equivariant if for all _g_ _∈_ _G_ and _x ∈_ _X_ :

_f_ ( _ρX_ ( _g_ )( _x_ )) = _ρY_ ( _g_ )( _f_ ( _x_ )) _._ (1)

**Example 2.1** (Equivariant Encoding of Point Clouds) **.** _The symmetry group is G_ = _H_ _Sn, where_
_×_
_H_ _is typically the special orthogonal group_ _SO_ (3) _or the orthogonal group_ _O_ (3) _,_ _and_ _Sn_ _is the_
_permutation_ _group._ _We_ _consider_ _features_ _that_ _are_ _invariant_ _to_ _both_ _permutation_ _and_ _translation._
_To achieve translation invariance, the input representation X_ _is the space of centered point clouds,_
_where H_ _acts on the coordinate of each point and Sn permutes the points._ _To achieve permutation_
_invariance, the final output are designed to be invariant with respect to Sn._ _This means the feature_
_representation of interest is a direct sum of specified irreducible representations of H._ _The task of_
_equivariant encoding is then to learn an equivariant map f_ : _X_ _→_ _Y ._

**Data symmetry.** For a _G_ -set _X_, the action partitions the space into orbits, _G_ ( _x_ ) := _{g_ ( _x_ ) _| g_ _∈_ _G}_ .
The subgroup that fixes a point is its **isotropy subgroup**, _Gx_ := _g_ _G_ _g_ ( _x_ ) = _x_ . The conjugacy
_{_ _∈_ _|_ _}_

1Although we choose _k_ -fold as an illustrative example, our theories are applicable to general cases.
2Unless otherwise specified, all groups are assumed to be compact Lie groups, and all vector spaces are
finite-dimensional real vector spaces.


2


class ( _Gx_ ) of this subgroup is the **orbit type** of _x_ . The set of all points with orbit type ( _H_ ) is _X_ ( _H_ )
and the set of points fixed by all elements of a subgroup _H_ is the fixed-point set _X_ _[H]_ .


These concepts distinguish between the global symmetry of the space and the intrinsic symmetry of
an object. The group _G_ represents the global symmetry, transforming the object between different
reference frames. An orbit _G_ ( _x_ ) thus represents a single physical object in all of its possible
orientations. This object possesses its own intrinsic symmetry, mathematically defined by the isotropy
subgroup _Gx_ for any point _x_ on its orbit. While the specific subgroup is frame-dependent, its structure
up to conjugation is constant. The orbit type ( _Gx_ ) therefore serves as a reference-frame independent
identifier that precisely describes the physical object’s intrinsic symmetry. The set of all possible
orbit types, _G_ ( _X_ ), thus catalogs all distinct symmetries that objects in the space can possess.
_O_
**Example** **2.2.** _Now_ _apply_ _the_ _setup_ _from_ _Ex._ _2.1._ _Consider_ _the_ _action_ _of_ _G_ = _O_ (3) _Sk_ +1 _on_
_×_
_centered point cloud space X for k_ _>_ 2 _._ _Let x ∈_ _X be the set of vertices of a k-fold in the xOy-plane:_

_x_ = ( _x_ 0 _, x_ 1 _, . . ., xk_ ) _,_ _where xi_ = (cos(2 _iπ/k_ ) _,_ sin(2 _iπ/k_ ) _,_ 0) _for i >_ 0 _._ (2)

_with x_ 0 _at the origin._ _The generators of Gx_ _include:(1) A rotation about the z-axis combined with a_
_cyclic permutation of x_ 1 _, . . ., xk._ _(2) A reflection across the xOz-plane combined with a product of_
_transpositions._ _(3) A reflection across the xOy-plane combined with the identity._


_Considering the projection map πX_ (( _g, σ_ )) = _g, where_ ( _g, σ_ ) _is a pair consisting of a geometric_
_transformation g_ _O_ (3) _and a permutation σ_ _Sk, we find that geometric symmetry πX_ ( _Gx_ ) _is the_
_∈_ _∈_
_dihedral group with horizontal reflection of order_ 4 _k, denoted by the Schoenflies symbol Dkh._


The symmetry of data can be altered by an equivariant map. The following theorem shows that an
equivariant map does not decrease symmetry.
**Theorem** **2.3** (Curie’s principle, Kaba & Ravanbakhsh (2023), Thm. 1) **.** _Let_ _f_ : _X_ _→_ _Y_ _be_ _a_
_G-equivariant map._ _For x ∈_ _X, the isotropy subgroup of x is contained in that of its image f_ ( _x_ ) _, i.e.,_

_Gx_ _Gf_ ( _x_ ) _._ (3)
_⊆_

Such an increase in symmetry becomes unavoidable if the feature space _Y_ cannot support the
input’s symmetry. If the orbit type ( _Gx_ ) is not present in _Y_ (i.e., ( _Gx_ ) _/_ _G_ ( _Y_ )), then the
_∈O_
equality _Gx_ = _Gf_ ( _x_ ) is impossible (otherwise, _Gf_ ( _x_ ) becomes an isotropy subgroup, implying
( _Gx_ ) _G_ ( _Y_ ), which leads to a contradiction), and the symmetry must therefore strictly increase.
_∈O_
This increase leads to a degeneration, as any transformation _g_ in the larger group _Gf_ ( _x_ ) that is not
in _Gx_ will map the distinct inputs _x_ and _g_ ( _x_ ) to the same output, since _f_ ( _g_ ( _x_ )) = _g_ ( _f_ ( _x_ )) = _f_ ( _x_ ).
Fig. 2 illustrates three possible types of such degenerations for the _k_ -fold structure from Ex. 2.2.


This section establishes that symmetry increase
is governed by an infimum determined by the
feature space. We prove the existence of this
infimum and show that its coincidence with the
input symmetry is a necessary condition for an
equivariant map to preserve symmetry.


3.1 THE INFIMUM OF SYMMETRY


Input Space Output Space
Figure 2: Three types of degeneration of _k_ -fold.


The symmetry of data can increase after being transformed by an equivariant map. This increase
can be an intentional design choice, or it can arise from subtle properties of the feature space. For
instance, in the task from Ex. 2.1, the requirement for permutation-invariant features means that the
permutation group _Sn_ is naturally introduced into the isotropy subgroup of any output. This type
of designed, unavoidable symmetry increase is formalized by the kernel of the group action on the
feature space. We define the **kernel** of the action _ρX_ as the set of group elements that fix every point
in _X_ : ker _ρX_ := _g_ _G_ _g_ ( _x_ ) = _x,_ _x_ _X_
_{_ _∈_ _|_ _∀_ _∈_ _}_

Distinguishing between intentional and unintended symmetry increase is crucial. We begin by
assuming that the group action is faithful, i.e., it has a trivial kernel with ker _ρY_ = _e_ . The case of a
_{_ _}_
nontrivial kernel will be discussed in the next subsection.


3


To characterize the behavior of symmetry increase within a representation _X_, we first establish a
partial order to compare different symmetries. An orbit type ( _H_ 1) is considered greater than or equal
to another, ( _H_ 2), written as ( _H_ 1) _≥_ ( _H_ 2), if _H_ 1 contains a conjugate of _H_ 2. This ordering reflects
that a larger orbit type corresponds to a higher symmetry.


With this framework, we are interested in the lower bound of orbit types that can be reached from
a point _x_ with a specific isotropy group _H_ = _Gx_ . The analysis can be framed around any closed
subgroup _H_, as the set of all possible isotropy subgroups is precisely the set of all closed subgroups
of _G_ (see, e.g., Field (2007); Mostow (1957)). The fixed-point space _X_ _[H]_ contains points of all higher
orbit types. This leads to the following powerful theorem, which guarantees that the lower bound on
symmetry increase is unique, corresponding to an isotropy subgroup that is unique up to conjugation.

**Theorem 3.1** (Uniqueness of Minimal Type) **.** _Let X_ _be a representation of a compact Lie group_
_G._ _For any closed subgroup H_ _⊆_ _G, a unique minimal orbit type exists among the points in the_
_fixed-point_ _subspace_ _X_ _[H]_ _._ _In_ _particular,_ _if_ ( _H_ ) _G_ ( _X_ _[H]_ ) _,_ _then_ ( _H_ ) _is_ _the_ _minimal_ _orbit_ _type_
_∈O_
_within that subspace._


The uniqueness guaranteed by this theorem allows us to define the **symmetry infimum**, denoted by
_IG_ ( _X, H_ ), as this unique minimal orbit type. In the context of symmetry increase, we are concerned
with the relationship between _I_ ( _Y, Gx_ ) and ( _Gf_ ( _x_ )). An unexpected symmetry increase occurs if
( _Gf_ ( _x_ )) _>_ _IG_ ( _Y, Gx_ ). The desired behavior for an equivariant map is captured by the following
definition. For a map between _G_ -sets _X_ and _Y_, we define an **isovariant map** as one that strictly
preserves symmetry for all _x ∈_ _X_ :
_Gx_ = _Gf_ ( _x_ ) _._ (4)

Using the concept of the symmetry infimum, we can provide a necessary condition for the existence
of isovariant maps. In § 5.1, we will see that this condition is in fact not sufficient for equivariant
maps between representations, even when we assume a trivial kernel ker _ρY_ = _e_ .
_{_ _}_
**Theorem 3.2.** _A necessary condition for the existence of an isovariant map between G-sets X_ _and_
_Y_ _is that_ _G_ ( _X_ ) _G_ ( _Y_ ) _._ _When X_ _and Y_ _are representations of a compact Lie group G, this is_
_O_ _⊆O_
_equivalent to the condition that IG_ ( _Y, H_ ) = ( _H_ ) _for all_ ( _H_ ) _G_ ( _X_ ) _._
_∈O_


3.2 EQUIVARIANCE WITH NON-TRIVIAL KERNELS


When the feature space _Y_ is restricted to have a non-trivial kernel, i.e., ker _ρY_ = _e_, the definition
_{_ _}_
of an isovariant map becomes too restrictive. Since every isotropy group _Gy_ in _Y_ contains the kernel,
any subgroup _H_ not containing ker _ρY_ cannot occur as an isotropy subgroup. Consequently, for any
map _f_ : _X_ _→_ _Y_, an input isotropy subgroup _Gx_ that does not contain the kernel must increase.

To formalize this unavoidable symmetry increase, we introduce an operator _pY_ . In the discussion
following Ex. 2.2, the presence of the _Sk_ +1 kernel forces the input isotropy subgroup _Gx_ to become
at leastdefined as the smallest subgroup containing _Dkh × Sk_ +1 in the feature space. We generalize this observation. _H_ that is compatible with the action onThe operator _Y_, given by the _pY_ ( _H_ ) is
projection _pY_ = _πY_ _[−]_ [1] _πY_, where _πY_ : _G_ _G/_ ker _ρY_ is the natural projection. This operator is

_◦_ _→_
idempotent ( _p_ [2] _Y_ [=] _[ p][Y]_ [ ) and maps any isotropy subgroup in] _[ Y]_ [to itself.] [These properties reveal why]
increasing is unavoidable. For any equivariant map _f_, the relation _Gx_ _Gf_ ( _x_ ) must hold. Applying
_⊆_
the _pY_ operator to this inclusion gives:

_Gx_ _⊆_ _pY_ ( _Gx_ ) _⊆_ _pY_ ( _Gf_ ( _x_ )) = _Gf_ ( _x_ ) _._ (5)


This unavoidable increasing from _Gx_ to _pY_ ( _Gx_ ) means our goal is not to preserve _Gx_ itself, but
to ensure no _additional_ symmetry is introduced beyond _pY_ ( _Gx_ ). This leads to a more practical
definition. We say a map _f_ : _X_ _→_ _Y_ is an **isovariant map relative to** _Y_ if for all _x ∈_ _X_ it satisfies

_pY_ ( _Gx_ ) = _Gf_ ( _x_ ) _⇐⇒_ _ρY_ ( _Gx_ ) = _ρY_ ( _Gf_ ( _x_ )) _._ (6)

When the kernel is trivial, this definition reduces to that of a standard isovariant map. With this
refined goal, we can state a necessary condition for the existence of such maps.

**Theorem 3.3.** _A necessary condition for the existence of an isovariant map relative to Y_ _from a_
_G-set_ _X_ _to_ _a_ _G-set_ _Y_ _is_ _that_ ( _pY_ ( _H_ )) _∈OG_ ( _Y_ ) _for_ _every_ ( _H_ ) _∈OG_ ( _X_ ) _._ _When_ _X_ _and_ _Y_ _are_
_representations, this is equivalent to the condition that IG_ ( _Y, H_ ) = ( _pY_ ( _H_ )) _for all_ ( _H_ ) _∈OG_ ( _X_ ) _._


4


|Col1|Col2|
|---|---|
|_R_||


**Change the Rotation Axis**


**Rotate around the Axis**

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||


Figure 3: Visualization of representation spaces. (a) A _k_ -fold structure is reoriented onto multiple
planes. (b) Each is further rotated about the perpendicular axis. (c) All structures are embedded
and projected into 2D. Marker shapes denote rotation axes, and colors denote rotation rates. Full
degeneration appears at _l_ = 0 _,_ 1, and axial degeneration at _l_ = 2 _,_ 4.


4 COMPUTATION OF ORBIT TYPES


The computation of orbit types is a classical problem, with most established results focusing on
irreducible representations, often in the context of bifurcation theory. In representation learning tasks,
however, we utilize feature spaces containing high multiplicities of these representations, which
requires us to supplement the existing computational frameworks.


4.1 ORBIT TYPES OF HIGH-MULTIPLICITY REPRESENTATIONS


For a compact Lie group _G_, any representation space _X_ can be uniquely decomposed into a direct
sum of its irreducible components. For _G_ = _SO_ (3), this decomposition is written as

_X_ = _[∼]_ [�] _l_ _[∞]_ 0=0 _[V]_ _l⊕_ = _ml_ 0 ( _X,Vl_ = _l_ 0 ) _,_ (7)

where _Vl_ = _l_ 0 is the irreducible representation corresponding to the space of spherical harmonics of
degree _l_ 0, and _m_ ( _X, Vl_ = _l_ 0) is its multiplicity. For _G_ = _O_ (3), the decomposition is similar, but the
irreducible representations must also be distinguished by parity, denoted _Vl_ = _l_ 0+ [and] _[ V][l]_ [=] _[l]_ 0 _[−]_ [.]


We begin with the foundational criterion for identifying isotropy subgroups, first established as a
necessary condition by Michel (1980).

**Theorem 4.1** (Michel’s Criterion, Michel (1980), App. A) **.** _Let V_ _be a representation of a group_
_G._ _A necessary condition for a closed subgroup H_ _to be an isotropy subgroup in V_ _is that for any_
_adjacent closed subgroup H_ _[′]_ ⊋ _H, the dimension of the fixed-point subspace strictly decreases:_

dim _V_ _[H]_ _[′]_ _<_ dim _V_ _[H]_ _._ (8)


While this condition is not sufficient for all representations, its sufficiency can be guaranteed under
certain common conditions. We define a representation _V_ of a group _G_ as a high-multiplicity
representation if for every non-zero isotypic component corresponding to an irreducible representation
_Vi_, its multiplicity _m_ ( _V, Vi_ ) is greater than dim _G_ .
**Proposition 4.2.** _For a high-multiplicity representation V, the necessary condition stated in Thm. 4.1_
_is also sufficient._


This criterion is particularly powerful for two reasons. First, it offers a computationally convenient
method in the form of a chain recursion, which only requires checking adjacent subgroups. The
dimensions of the necessary fixed-point spaces can be calculated via the trace formula (Golubitsky


5


**Algorithm 1:** Orbit Type Test for HighMultiplicity Representations

**Data:** Symmetry group _G_ ;
Closed subgroup _H_ _⊂_ _G_ ;
High-Multiplicity Rep. _V_ of _G_
**Result:** is_in(( _H_ ) _,_ _G_ ( _V_ )).

**1** Let set _S_ = _Hi_ _G O_ to be all adjacent
_{_ _} ⊂_
closed supergroups of _H_ in _G_ ;


**Algorithm 2:** Symmetry Infimum Calculation

**Data:** Symmetry group _G_ ;
Closed subgroup _H_ _⊂_ _G_ ;
Rep. _V_ of _G_ .
**Result:** Symmetry inf. _IG_ ( _V, H_ ).

**1** **if** is_in(( _H_ ) _,_ _G_ ( _V_ )) **then**

**2** **return** ( _H_ ) _O_ ;

**3** **end**

**4** Let set _S_ = _Hi_ to be all closed
_{_ _}_
supergroups of _H_ in _G_ ;


**2** _O_ _←_ ∅;


**3** _dH_ dim _V_ _[H]_ ;
_←_


**4** **for** _Hi_ _in S_ **do**


**5** _dHi_ dim _V_ _[H][i]_ ;

**6** **if** _d_ _←_ _d_ = 0


**5** _O_ _←_ ∅;


**6** **for** _Hi_ _in S_ **do**


**6** **if** _dH_ _dHi_ = 0 **then**

_−_ False


**7** **if** is_in(( _Hi_ ) _,_ _G_ ( _V_ )) **then**
_O_


**8** Add ( _Hi_ ) to ;
_O_


**7** **return** False;


**8** **end**

**9** **end**

**10** **return** True;


**10** **end**

**11** **return** min( _O_ );


**9** **end**


et al., 1988). Second, the sufficiency condition is frequently met in our applications, as it holds for all
finite groups and for feature spaces with a high number of channels. Based on the result of Prop. 4.2,
we design an orbit type test Algo. 1 and a symmetry infimum calculation Algo. 2. Using the two
algorithms described above, we have characterized all instances of symmetry increase for the closed
subgroups of _SO_ (3) or _O_ (3) in the representations _Vl_ _[⊕]_ = _[r]_ _l_ 0 [and] _[ V]_ _l_ _[⊕]_ = _[r]_ _l_ 0 _[±]_ [for] _[ r]_ _[>]_ [ 3][, respectively, see §][ C.4][.]
We now illustrate our algorithms with a simple example.

**Example 4.3.** _We illustrate our algorithms by calculating the orbit type and symmetry infimum for_
_the geometric symmetry Dkh (k_ _>_ 2 _) of the k-fold from Ex. 2.2, considered as a subgroup of O_ (3) _._
_The calculation is performed in the high-multiplicity representation space Y_ = _Vl_ _[⊕]_ = _[r]_ _l_ 0 _[(][r]_ _[>]_ [ 3] _[, l]_ [0] _[>]_ [ 0] _[).]_
_Here we provide only a sketch of the derivation, the full procedure is provided in § C.3._


_First,_ _we_ _apply_ _the_ _orbit_ _type_ _test_ _from_ _Algo._ _1._ _This_ _involves_ _comparing_ _the_ _dimension_ _of_ _the_
_fixed-point space of Dkh_ _with that of its adjacent supergroups (e.g., Dpk,h_ _and, for k_ = 4 _, Oh)._ _The_
_analysis shows that_ ( _Dkh_ ) _is an orbit type if and only if l_ 0 _k and l_ 0 _, k have the same parity._ _Next,_
_≥_
_we apply the symmetry infimum calculation from Algo. 2._ _This requires identifying the minimal orbit_
_type_ _among all_ _supergroups of_ _Dkh,_ _including_ _non-adjacent ones_ _like_ _D∞h_ _and_ _O_ (3) _._ _The final_
_results are summarized in Table 1._

Table 1: The symmetry infimum _IO_ (3)( _Vl_ _[⊕]_ = _[r]_ _l_ 0 _[, D][kh]_ [)][ for] _[ k]_ _[>]_ [ 2] _[, r]_ _[>]_ [ 3] _[, l]_ [0] _[>]_ [ 0][.]

_l_ 0 _< k_ _k_ _l_ 0 _<_ 2 _k_ _l_ 0 2 _k_
_≤_ _≥_
_l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd


_k_ is even ( _D∞h_ ) ( _O_ (3)) ( _Dkh_ ) ( _O_ (3)) ( _Dkh_ ) ( _O_ (3))
_k_ is odd ( _D∞h_ ) ( _O_ (3)) ( _D∞h_ ) ( _Dkh_ ) ( _D_ 2 _kh_ ) ( _Dkh_ )


The analysis in Ex. 4.3 predicts three types of degeneration for the _k_ -fold inputs from Ex. 2.1:

    - **Half Degeneration:** distinguish the _k_ -fold from itself rotated byThe symmetry infimum of _π/k G_ around the _x_ is ( _D_ 2 _kh z ×_ -axis. _Sk_ +1). The feature cannot

    - **Axial Degeneration:** The symmetry infimum of _Gx_ is ( _D∞h_ _Sk_ +1). The feature cannot
distinguish the _k_ -fold from itself rotated by _any_ angle around the _×_ _z_ -axis.

    - **Full Degeneration:** The symmetry infimum of _Gx_ is ( _O_ (3) _× Sk_ +1). The feature cannot
distinguish the _k_ -fold from itself rotated by _any_ angle around _any_ axis.


We consider encoding _k_ -fold point clouds using equivariant neural networks and visualize the
resulting embeddings. The three degenerations are experimentally verified in our visualizations,
with Fig. 3 showing full and axial degeneration, and Fig. 4 showing axial and half degeneration.
Although derived assuming high multiplicity ( _r_ _>_ 3) in the feature representation, these predictions
are identical for the single representation case ( _r_ = 1), see § C.4.


6


**a** **c**


**b**


**Axial Deg.**


**Symmetry Increase**


Figure 4: Visualization of representation spaces. (a) A _k_ -fold ( _k_ is odd) structure is rotated _π/k_ about
the perpendicular axis. (b) The path of symmetry increase. (c) Half degeneration appears at _l_ = 10.
At res = 98 and res = 49, the overall shape is identical, but the yellow data points ( _i.e._ the second
half of the rotation) completely cover the blue data points.


4.2 GUIDELINES FOR MANAGING SYMMETRY INCREASE


It is a general property of representations that the orbit types of a direct sum are related to those
of its components by _G_ ( _V_ 1) _G_ ( _V_ 2) _G_ ( _V_ 1 _V_ 2), and _IG_ ( _V_ 1 _V_ 2 _, H_ ) _IG_ ( _Vi, H_ ) for
_O_ _∪O_ _⊆O_ _⊕_ _⊕_ _≤_
_i_ = 1 _,_ 2, with equality conditions discussed in § C.5. These properties provide a direct mechanism
for controlling the symmetry increase of an equivariant feature, that is to choose components whose
symmetry infimum (computed as described in § C.4 for _G_ = _SO_ (3) or _O_ (3)) align with the desired
behavior for task-relevant symmetries. Regarding the selection of the feature space _Y_ in equivariant
representation learning, this principle translates into two guidelines.


For **orientation-dependent tasks** (e.g., § 6.2), when considering the kernel of feature space, it is
crucial to avoid non-trivial symmetry increase (i.e. ensuring the map is relative isovariant) since such
increases can lead to the accidental loss of orientational information. Therefore, for a given input
symmetry ( _H_ ), one should select feature components that contain the orbit type ( _pY_ ( _H_ )).


For **general tasks** (e.g., § 6.3), certain forms of symmetry increase must be avoided, as the output
symmetry reflects the dimensionality of the fixed-point subspace where the equivariant features
lie (see _Remark_ of Prop. C.2). In this context, one should generally avoid components where the
symmetry infimum indicates a severe compression of the fixed-point subspace. Specifically, one must
be cautious of components corresponding to non-trivial representations where the symmetry increases
to the full group, as this causes the component to be annihilated and lose all discriminative power.


5 DENSITY OF (ALMOST) ISOVARIANT MAPS


We now connect the preceding theory to a practical machine learning context by introducing models
for the data distribution and for the parameterized map. We show that the necessary conditions for
isovariance established previously become sufficient under a relaxed definition of isovariance.


5.1 THE MANIFOLD HYPOTHESIS


Motivated by the manifold hypothesis and the broader considerations summarized in § A.3, we
model the data distribution as being supported on a finite union of smooth, compact submanifolds
_M_ = [�] _j_ _[M][j]_ [embedded in] _[ X]_ [.] [When a group] _[ G]_ [ acts on] _[ X]_ [, this action equips each] _[ M][j]_ [with a natural]

_G_ -manifold structure.

The central question is: when does an isovariant map _f_ : _M_ _→_ _Y_ exist? The existence is a non-trivial
issue. The counterexample in Cex. D.3 demonstrates that the necessary condition of orbit type


7


inclusion is not sufficient. Specifically, it shows that an isovariant map can fail to exist precisely
because the multiplicities of the irreducible representations in the feature space are insufficient.


The non-existence of perfectly isovariant maps motivates a more practical, relaxed definition: a
map that is isovariant almost everywhere. To formalize this, we equip _M_ with the _d_ -dimensional
Hausdorff measure _µM_, where _d_ = max _j_ dim _Mj_ . This allows us to identify subsets of _measure_
_{_ _}_
_zero_ as negligible. Note that _M_ ( _H_ ) is a finite union of submanifolds, then _f_ is **almost isovariant**
**relative to** _Y_ if for every orbit type ( _H_ ) in the data support, the isovariance condition


_ρY_ ( _Gx_ ) = _ρY_ ( _Gf_ ( _x_ )) _,_ (9)

holds for all points _x_ _M_ ( _H_ ) except for a subset of _µM_ ( _H_ )-measure zero. This ensures that any
_∈_
undesired increase in symmetry occurs only on a negligible portion of the data.


5.2 GENERICITY OF (ALMOST) ISOVARIANT MAPS


For Ex. 2.1, we select TFN (Thomas et al., 2018), a classic ENN based on tensor products, as our
parameterized model. We provide the complete formulation to § D.2. An important property of
TFN parameterizations TFN is that they satisfy a universal approximation theorem (Dym & Maron,
_F_
2021). In topology, this is equivalent to _F_ TFN being dense in equivariant function space _CG_ ( _X, Y_ )
with respect to the _C_ [0] topology. In fact, we can establish a stronger approximation theorem.
**Theorem 5.1.** _In Ex. 2.1, the function families_ TFN _with smooth activation function are C_ _[∞]_ _-dense_
_F_
_in_ _the_ _space_ _of_ _smooth_ _equivariant_ _maps_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[.]_ _[That]_ _[is,]_ _[for]_ _[any]_ _[integer]_ _[r]_ _[≥]_ [0] _[,]_ _[any]_ _[map]_
_f_ _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[, any compact set][ K]_ _[⊂]_ _[X][, and any][ ϵ >]_ [ 0] _[, there exists a function][ g]_ _[∈F]_ [TFN] _[ such that]_

max _x∈K_ �� _Dkf_ ( _x_ ) _−_ _Dkg_ ( _x_ )�� _< ϵ, k_ _≤_ _r._ (10)


Here, _D_ _[k]_ denotes the _k_ -th order total derivative operator. A significant portion of maps within a
dense parameterization reflects the _generic_ properties of the mapping space. For equivariant maps, a
key generic property, closely related to almost isovariance, is that the dimension of the set of points
where the orbit type is increase from ( _H_ ) to ( _H_ _[′]_ ) by a map _f_ is constrained for a generic map. The
following theorem shows that for expressive models with _C_ _[∞]_ approximation capabilities, such as the
TFN discussed, almost isovariance is a generic property, and full relative isovariance can be achieved
by increasing representation multiplicity. As shown in Cex. D.3, this requirement is tight.
**Theorem 5.2.** _Let F_ _be a equivariant parametrization with C_ _[∞]_ _approximation capability._ _If for_
_every_ ( _H_ ) _∈OG_ ( _M_ ) _we have_ ( _pY_ ( _H_ )) _∈OG_ ( _Y_ ) _, then for any finite union of compact, smooth_
_G-submanifolds M_ _⊂_ _X, any f_ _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[, any integer][ r]_ _[≥]_ [0] _[, and any][ ϵ >]_ [ 0] _[, there exists a map]_
_g_ _∈F_ _such that_
max _x∈M_ _D_ _[k]_ _f_ ( _x_ ) _D_ _[k]_ _g_ ( _x_ ) _< ϵ, k_ _r,_ (11)
_∥_ _−_ _∥_ _≤_
_and_ _g_ _M_ _is_ _almost_ _isovariant_ _relative_ _to_ _Y ._ _Furthermore,_ _if_ _the_ _feature_ _space_ _Y_ _contains_ _a_
_|_
_representation_ _Y_ [˜] _[⊕][r]_ _for_ _an_ _integer_ _r_ _>_ max _j_ dim _Mj_ _,_ _where_ _Y_ [˜] _itself_ _satisfies_ _the_ _condition_
_{_ _}_
( _p_ ˜ _Y_ ( _H_ )) _∈OG_ ( _Y_ [˜] ) _, then the approximating map g|M_ _can be chosen to be isovariant relative to Y ._


6 EXPERIMENT


We validate our theoretical analysis through three experiments: representation-space visualizations
in § 6.1, a geometric graph discrimination task in § 6.2, and a molecular isotropic polarizability
prediction task on QM9 (Ramakrishnan et al., 2014) in § 6.3. Across all experiments, we consider
two equivariant architectures, TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024): TFN is used
in the first two experiments, whereas HEGNN is employed in the last two. All detailed experimental
settings are provided in § F.


6.1 VISUALIZATION OF REPRESENTATION SPACE


To provide a clearer illustration of our theory, we present visualizations of the representation spaces
of different degrees, obtained from the 3-fold structure.


**Dataset.** We first construct a _k_ -fold structure lying in the _xOy_ plane (here _k_ = 3), and then
apply random rotations to place it on _m_ distinct planes (here _m_ = 6), as illustrated in Fig. 4(a).


8


Subsequently, as shown in Fig. 4(b), each structure is further rotated about the axis perpendicular to
its plane. The rotation angle _θ_ _∈_ [0 _,_ 2 _π/k_ ) is uniformly discretized into res candidate values, defined
as _{_ 2 _πi/_ ( _k ·_ res) _}i_ [res] =0 _[−]_ [1][.] [Unless otherwise stated, we use][ res] [=] [49][ to verify the half-degeneration.]
We also consider the doubled resolution res = 98.

**Embeddings.** For the resulting _m ·_ res candidate structures, we compute the graph-level features
via randomly initialized single-layer TFN (Thomas et al., 2018) with detailed setting in § F.1. For
_l_ 0 = 0 the feature dimension is 1; for visualization we set the second coordinate to zero. For _l_ 0 1
_≥_
the feature dimension is 2 _l_ 0 + 1 _>_ 2; we reduce dimensionality via random projection and then
rescale all features to a common range so that visualizations are comparable. Data points are plotted
in ascending order of plane index; for structures on the same plane they are ordered by increasing
rotation angle about the axis.


**Results.** Detailed experimental results are presented in Figs. 3 and 4. The former shows the input
symmetry with _l_ 0 11 increase to ( _O_ (3) _Sk_ +1) (Full degeneration, _l_ 0 = 0 _,_ 1), ( _D∞h_ _Sk_ +1)
(Axial degeneration, _≤ l_ 0 = 2 _,_ 4), or remains non-degenerate. _×_ The latter shows the symmetry increase _×_
to ( _D_ 2 _kh_ _Sk_ +1) at _l_ 0 = 10. These experimental results are consistent with Ex. 4.3.
_×_


6.2 EXPRESSIVITY ON SYMMETRIC GRAPHS


To experimentally validate our theoretical conclusion established, we design a more comprehensive experiment following Joshi et al. (2023).


**Dataset.** We construct four symmetric _k_ -fold
structures ( _k_ _∈_ 2 _,_ 3 _,_ 4 _,_ 6), each centered at the
origin. For each structure _G_ 0 we apply a random
rotation to obtain _G_ 1, ensuring that _G_ 1 does not
coincide with the original 0. The goal is to
_G_
evaluate whether different ENNs can distinguish
0 from 1. To probe different aspects of our
_G_ _G_
theory, we treat 2D and 3D rotations separately;
in the 3D setting we additionally require that 1
_G_
is not coplanar with 0.
_G_


0
1
2
3
4
5
6
7
8
9

10
11


2D Rotation 3D Rotation


2-fold 3-fold 4-fold 6-fold 2-fold 3-fold 4-fold 6-fold


1


10 3


10 6


0


Figure 5: Heatmap of Emb. Diff. Norm.


**Embeddings.** We employed both TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024)
to compute the norm of the embedding difference across 12 configurations for each, varying by
the number of irrep channels (1, 4, 16) and layers (1-4). The extracted _l_ 0-degree embeddings are
evaluated via the norm of the difference between the embeddings of 0 and 1 as in Cen et al. (2024)
_G_ _G_
with detailed setting in § F.2.1. When this norm approaches zero, the two embeddings are numerically
indistinguishable, and hence the corresponding geometric figures cannot be told apart by the model [3] .


**Results.** The maximum value was selected for each configuration and visualized in a heatmap,
as shown in Fig. 5. The results exhibit a clear binary pattern: the values are either greater than
10 _[−]_ [3] or less than 10 _[−]_ [6] (due to numerical error), with a difference of more than 10 [3] times. This
suggests that values exceeding 10 _[−]_ [3] indicate distinguishable structures, while those below 10 _[−]_ [6]
correspond to indistinguishable structures. These findings align precisely with our theoretical
predictions. Furthermore, as the maximum value was chosen, with all norms being less than 10 _[−]_ [6],
this phenomenon is shown to be independent of the model choice, the number of channels or layers.


6.3 MOLECULE PROPERTY PREDICTION WITH PRETRAINED EQUIVARIANT FEATURES


To illustrate the guiding significance of the theory presented in this paper for practical applications,
we designed experiments on the QM9 dataset (Ramakrishnan et al., 2014) for verification.


**Dataset.** We choose to predict the molecular isotropic polarizability _α_ . It is worth noting that the
QM9 dataset (Ramakrishnan et al., 2014) contains many highly symmetric structures spanning 22


3In Joshi et al. (2023), the embeddings are directly fed to a vanilla classifier. To address issues such as
imperfect classifier training and numerical error, we slightly modify this experimental setup. We also reproduce
their original experiment in § F.2.2, and the results remain consistent with our theory.


9


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


102


101


100


10 1


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree l


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree l


10 2


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree l


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree l


Figure 6: MAE loss (in units of _a_ [3] 0 [) for isotropic polarizability prediction with degree] _[ l]_ [ =] _[ l]_ [0] [across]
molecules from the top-16 point groups by molecular count. Each boxplot shows the distribution of
errors at a given degree, while diamond markers denote the corresponding mean MAE.


molecular symmetry groups [4], with Fig. 6 reporting sample counts for the 16 most frequent groups;
the remaining seven are _D_ 3 with 2 samples and _C_ 4, _D_ 2, _D_ 6 _h_, _Oh_, _S_ 4, each appearing only once.


**Embeddings.** We adopt HEGNN (Cen et al., 2024) as our backbone. Specifically, we first pretrain
on features of _l_ _≤_ 11 to obtain a shared equivariant feature encoder, ensuring that all subsequent
configurations operate on the same embedding space. We then consider two fine-tuning strategies:
(a) using only features of _l_ = _l_ 0, and (b) using all features of _l_ _l_ 0, yielding 12 distinct prediction
_≤_
heads for each. The detailed experimental setup is given in § F.3.1.


**Results.** The results are shown in Figs. 6 and 7, with a summary of all symmetry increases in
Table 22. We note that for the majority of samples, different feature components contribute similarly
to the prediction. However, as illustrated in Fig. 6, for non-trivial feature components where molecular
symmetry increase to _O_ (3), the prediction loss is substantially higher. Furthermore, the results in
Fig. 7 show that for symmetries causing full degeneration in 1-degree features, including additional
1-degree features may not provide significant improvement to the model’s prediction performance.
This validates the design guidelines in § 4.2. Detailed case studies are provided in § F.3.2.


7 CONCLUSION


In this work, we presented a rigorous mathematical framework to address the critical issue of
symmetry increase in ENNs. We introduced the concept of the symmetry infimum, a computable
lower bound for any increase in symmetry determined by the feature space. Our central contribution
is to show that this infimum can be used to precisely predict and control the expressive degradation of
ENNs. The framework successfully explains phenomena in settings like those of Joshi et al. (2023),
which could not be fully accounted for by prior theories such as the collapse-to-zero model from Cen
et al. (2024). Our findings provide both a robust theoretical understanding and practical guidelines
for designing more reliable ENNs.


4We use the QM9 dataset as provided in PyG (Fey & Lenssen, 2019) and apply the PointGroup library (Carreras, 2025) to pre-compute and manually post-process the point groups of all molecules. As a result, our
statistics may differ slightly from those reported in previous works like Zeng et al. (2025).


10


ACKNOWLEDGMENTS


This work was supported by the National Natural Science Foundation of China (Nos. 62276269,
92270118, and 62376276), the Beijing Natural Science Foundation (No. 1232009), and the Beijing
Nova Program (No. 20230484278).


AUTHOR CONTRIBUTIONS


Ning Lin organized this project. Ning Lin led the theoretical development in § 2–§ 5 and was
responsible for the theoretical proofs of the corresponding part. Ning Lin and Jiacheng Cen jointly
led the experimental studies in § 6. Jiacheng Cen was responsible for model implementation and
code development. Anyi Li was responsible for data processing. Jiacheng Cen and Anyi Li jointly
contributed to figure preparation and visualization. Wenbing Huang and Hao Sun jointly supervised
and guided the project. All authors participated in writing and revising the manuscript.


USAGE OF LARGE LANGUAGE MODELS


We only use Large Language Models to polish our writing.


ETHICS STATEMENT


This work is a theoretical contribution in the domain of equivariant neural networks, focusing on
mathematical properties of symmetry preservation and transformation under equivariant mappings.
The research does not involve human subjects, personal data, or real-world deployments, and therefore
does not raise concerns related to privacy, fairness, bias, or potential misuse.


We affirm that this work adheres to the ICLR Code of Ethics. In particular, we have ensured honesty
in representing our contributions, accuracy in reporting our findings, and proper attribution of prior
work. As this is a theoretical study, there are no conflicts of interest, sponsorship influences, or
applications with foreseeable societal harms to disclose. We support responsible stewardship of
machine learning research and believe that foundational advances such as ours contribute positively
to the scientific community by enhancing understanding and enabling future trustworthy systems.


REPRODUCIBILITY STATEMENT


We are committed to ensuring the reproducibility of both the theoretical and experimental components
of our work. To support the reproducibility of our theoretical results, we provide complete and
self-contained proofs for all main theorems and propositions in the appendix, including detailed
derivations and necessary mathematical background. These proofs clarify all assumptions and
logical steps required to verify our claims. For the experimental component, we have made our
[implementation code publicly available at https://github.com/GLAD-RUC/SymInc.](https://github.com/GLAD-RUC/SymInc) The
codebase contains clear documentation and scripts that fully reproduce the reported results. All
experimental settings, hyperparameters, and data generation procedures are described in detail within
the supplementary materials. Together, the comprehensive theoretical appendices and open-sourced
code ensure that our findings can be rigorously verified and built upon by the research community.


REFERENCES


V.I. Arnold, S.M. Gusein-Zade, and A.N. Varchenko. _Singularities_ _of_ _Differentiable_ _Maps:_
_Classification_ _of_ _Critical_ _Points,_ _Caustics_ _and_ _Wave_ _Fronts_, volume 1. Birkhäuser Boston,
2012. doi: 10.1007/978-0-8176-8340-5. [URL https://link.springer.com/10.1007/](https://link.springer.com/10.1007/978-0-8176-8340-5)
[978-0-8176-8340-5.](https://link.springer.com/10.1007/978-0-8176-8340-5)


M. I. Aroyo. _International_ _Tables_ _for_ _Crystallography:_ _Space-group_ _symmetry_, volume A.
International Union of Crystallography, 2 edition, 2016. ISBN 978-0-470-97423-0. doi:
10.1107/97809553602060000114. [URL https://it.iucr.org/Ac/.](https://it.iucr.org/Ac/)


11


Sarp Aykent and Tian Xia. GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks.
In _The Thirteenth International Conference on LearningRepresentations_, 2025.


Perla Azzi, Rodrigue Desmorat, Julien Grivaux, and Boris Kolev. Rationality of normal forms of
isotropy strata of a representation of a compact lie group. _arXiv preprint arXiv:2301.08599_, 2023.


Edward Bierstone. General position of equivariant maps. _Transactions of the American Mathematical_
_Society_, 234(2):447–466, 1977.


Glen E. Bredon. _Introduction to compact transformation groups_, volume 46. Academic Press, 1972.


Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veliˇckovi´c. Geometric deep learning:
Grids, groups, graphs, geodesics, and gauges, 2021.


Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell, and Gabriel
Loaiza-Ganem. Verifying the union of manifolds hypothesis for image data. _arXiv_ _preprint_
_arXiv:2207.02862_, 2022.


Bin Cao, Yang Liu, Longhan Zhang, Yifan Wu, Zhixun Li, Yuyu Luo, Hong Cheng, Yang Ren, and
Tongyi ZHANG. Beyond structure: Invariant crystal property prediction with pseudo-particle ray
diffraction. In _The Fourteenth International Conference on Learning Representations_, 2026.


Abel Carreras. pointgroup: Python library to determine the point symmetry group of molecular
geometries, 2025. [URL https://github.com/abelcarreras/pointgroup.](https://github.com/abelcarreras/pointgroup)


Jiacheng Cen, Anyi Li, Ning Lin, Yuxiang Ren, Zihe Wang, and Wenbing Huang. Are high-degree
representations really unnecessary in equivariant graph neural networks? In _Advances in Neural_
_Information Processing Systems_, volume 37, pp. 26238–26266, 2024.


Jiacheng Cen, Anyi Li, Ning Lin, Tingyang Xu, Yu Rong, Deli Zhao, Zihe Wang, and Wenbing
Huang. Universally invariant learning in equivariant GNNs. In _The Thirty-ninth Annual Conference_
_on Neural Information Processing Systems_, 2025.


Alexandre Agm Duval, Victor Schmidt, Alex Hernández-Garcıa, Santiago Miret, Fragkiskos D
Malliaros, Yoshua Bengio, and David Rolnick. Faenet: Frame averaging equivariant gnn for
materials modeling. In _International Conference on Machine Learning_, pp. 9013–9033. PMLR,
2023.


Nadav Dym and Haggai Maron. On the universality of rotation equivariant point cloud networks. In
_International Conference on Learning Representations_, 2021.


Jinjia Feng, Zhewei Wei, Taifeng Wang, and Zongyang Qiu. TetraGT: Tetrahedral geometry-driven
explicit token interactions with graph transformer for molecular representation learning. In _The_
_Fourteenth International Conference on Learning Representations_, 2026.


Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. _arXiv_
_preprint arXiv:1903.02428_, 2019.


Michael Field. _Dynamics and symmetry_, volume 3. World Scientific, 2007.


Mario Geiger and Tess Smidt. e3nn: Euclidean neural networks. _arXiv preprint arXiv:2207.09453_,
2022.


Martin Golubitsky and Victor Guillemin. _Stable_ _mappings_ _and_ _their_ _singularities_ . Number 14
in Graduate texts in mathematics. Springer, 1973. ISBN 978-0-387-90073-5. doi: 10.1007/
978-1-4615-7904-5.


Martin Golubitsky, Ian Stewart, and David G. Schaeffer. _Singularities and Groups in Bifurcation_
_Theory_, volume 2, number 69 of _Applied Mathematical Sciences_ . Springer New York, 1988. ISBN
978-1-4612-8929-6 978-1-4612-4574-2. doi: 10.1007/978-1-4612-4574-2.


Ian Goodfellow, Yoshua Bengio, and Aaron Courville. _Deep Learning_, volume 1. MIT Press, 2016.


Victor Guillemin and Alan Pollack. _Differential topology_ . Prentice Hall, 1974. ISBN 978-0-13212605-2.


12


Jiaqi Han, Jiacheng Cen, Liming Wu, Zongzhao Li, Xiangzhe Kong, Rui Jiao, Ziyang Yu, Tingyang
Xu, Fandi Wu, Zihe Wang, et al. A survey of geometric graph neural networks: Data structures,
models and applications. _Frontiers of Computer Science_, 19(11):1911375, 2025.


Morris W. Hirsch. _Differential topology_ . Number 33 in Graduate texts in mathematics. SpringerVerlag, 1976. ISBN 978-1-4684-9451-8. doi: 10.1007/978-1-4684-9449-5.


Wenbing Huang and Jiacheng Cen. Geometric graph learning for drug design. _Deep Learning in_
_Drug Design_, pp. 133–151, 2026.


E Ihrig and Martin Golubitsky. Pattern selection with O(3) symmetry. _Physica_ _D:_ _Nonlinear_
_Phenomena_, 13(1-2):1–33, 1984.


Chaitanya K Joshi, Cristian Bodnar, Simon V Mathis, Taco Cohen, and Pietro Lio. On the expressive
power of geometric graph neural networks. In _International conference on machine learning_, pp.
15330–15355. PMLR, 2023.


Sékou-Oumar Kaba and Siamak Ravanbakhsh. Symmetry breaking and equivariant neural networks.
In _NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations_, 2023.


John M. Lee. _Introduction to Smooth Manifolds_, volume 218 of _Graduate Texts in Mathematics_ .
Springer, 2012. doi: 10.1007/978-1-4419-9982-5.


Anyi Li, Jiacheng Cen, Songyou Li, Mingze Li, Yang Yu, and Wenbing Huang. Geometric mixture
models for electrolyte conductivity prediction. In _The Thirty-ninth Annual Conference on Neural_
_Information Processing Systems_, 2025a.


Qi Li, Rui Jiao, Liming Wu, Tiannian Zhu, Wenbing Huang, Shifeng Jin, Yang Liu, Hongming Weng,
and Xiaolong Chen. Powder diffraction crystal structure determination using generative models.
_Nature Communications_, 16(1):7428, 2025b.


Zongzhao Li, Jiacheng Cen, Wenbing Huang, Taifeng Wang, and Le Song. Size-generalizable
RNA structure evaluation by exploring hierarchical geometries. In _The Thirteenth International_
_Conference on Learning Representations_, 2025c.


M. J. Linehan and G. E. Stedman. Little groups of irreps of O(3), SO(3), and the infinite axial
subgroups. _Journal of Physics A: Mathematical and General_, 34(34):6663–6688, 2001.


Eckhard Meinrenken. Group actions on manifolds, 2003. [URL https://www.math.toronto.](https://www.math.toronto.edu/mein/teaching/LectureNotes/action.pdf)
[edu/mein/teaching/LectureNotes/action.pdf.](https://www.math.toronto.edu/mein/teaching/LectureNotes/action.pdf)


Louis Michel. Symmetry defects and broken symmetry. configurations hidden symmetry. _Reviews of_
_Modern Physics_, 52(3):617–651, 1980. doi: 10.1103/RevModPhys.52.617.


John Willard Milnor. _Topology from the differentiable viewpoint_ . Univ. Pr. of Virginia, 8. print edition,
1990. ISBN 978-0-8139-0181-7.


G. D. Mostow. Equivariant embeddings in euclidean space. _The Annals of Mathematics_, 65(3):432,
1957. doi: 10.2307/1970055.


Ikumitsu Nagasaki. The weak isovariant borsuk-ulam theorem for compact lie groups. _Archiv der_
_Mathematik_, 81(3):348–359, 2003. doi: 10.1007/s00013-003-4693-1.


Allan Pinkus. Approximation theory of the mlp model in neural networks. _Acta numerica_, 8:143–195,
1999.


Omri Puny, Matan Atzmon, Edward J. Smith, Ishan Misra, Aditya Grover, Heli Ben-Hamu, and
Yaron Lipman. Frame averaging for invariant and equivariant network design. In _International_
_Conference on Learning Representations_, 2022.


Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum
chemistry structures and properties of 134 kilo molecules. _Scientific data_, 1(1):1–7, 2014.


Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks.
In _International Conference on Machine Learning_ . PMLR, 2021.


13


Tess E Smidt, Mario Geiger, and Benjamin Kurt Miller. Finding symmetry breaking order parameters
with euclidean neural networks. _Physical Review Research_, 3(1):L012002, 2021.


Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley.
Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds.
_arXiv preprint arXiv:1802.08219_, 2018.


D. J. A. Trotman. Counterexamples in stratification theory: two discordant horns. _Real and complex_
_singularities, Oslo_, pp. 679–686, 1976.


Stephen Willard. _General topology_ . Addison-Wesley series in mathematics. Addison-Wesley, 1970.
ISBN 978-0-201-08707-9.


Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, and Wenbing Huang. Equivariant spatio-temporal attentive graph networks to simulate physical dynamics. _Advances in Neural Information Processing_
_Systems_, 36, 2023.


Liming Wu, Wenbing Huang, Rui Jiao, Jianxing Huang, Liwei Liu, Yipeng Zhou, Hao Sun, Yang
Liu, Fuchun Sun, Yuxiang Ren, et al. Siamese foundation models for crystal structure prediction.
_arXiv preprint arXiv:2503.10471_, 2025.


Liming Wu, Rui Jiao, Qi Li, Mingze Li, Songyou Li, Shifeng Jin, and Wenbing Huang. Dmflow:
Disordered materials generation by flow matching. _arXiv preprint arXiv:2602.04734_, 2026.


Siyuan Zeng, Kuanping Gong, Yongquan Jiang, and Yan Yang. A method for predicting molecular
point group based on graph neural networks. _Artificial Intelligence Chemistry_, pp. 100097, 2025.


Yuelin Zhang, Jiacheng Cen, Jiaqi Han, Zhiqiang Zhang, JUN ZHOU, and Wenbing Huang. Improving equivariant graph neural networks on large geometric graphs via virtual nodes learning. In
_Forty-first International Conference on Machine Learning_, 2024.


Yuelin Zhang, Jiacheng Cen, Jiaqi Han, and Wenbing Huang. Fast and distributed equivariant graph
neural networks by virtual node learning. _arXiv preprint arXiv:2506.19482_, 2025.


14


CONTENTS OF APPENDIX


**A** **Backgrounds** **16**


A.1 Equivariant Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16


A.2 Closed Subgroups of _SO_ (3) or _O_ (3) . . . . . . . . . . . . . . . . . . . . . . . . . 16


A.3 Manifolds and Basic Data Assumption . . . . . . . . . . . . . . . . . . . . . . . . 17


**B** **Proofs of The Infimum of Symmetry (§ 3)** **19**


B.1 Proof of Thm. 3.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


B.2 Proof of Thm. 3.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


**C** **Symmetry Increase for** _G_ = _SO_ (3) **or** _O_ (3) **in § 4** **21**


C.1 General Orbit Type Criterion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


C.2 Proof of Prop. 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21


C.3 Detailed Calculation in Ex. 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22


C.4 Calculation of Symmetry Infimum . . . . . . . . . . . . . . . . . . . . . . . . . . 23


C.5 Composition Property of High-Multiplicity Representation . . . . . . . . . . . . . 24


**D** **Proofs of Density of (Almost) Isovariant Maps (§ 5)** **27**


D.1 Counterexample in § 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


D.2 Proof of Thm. 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


D.3 Some Results on Topology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31


D.4 Generic Equivariant Mappings . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33


D.5 Proof of Thm. 5.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38


**E** **Tables** **39**


E.1 Minimal Proper Supergroups in _SO_ (3) or _O_ (3) . . . . . . . . . . . . . . . . . . . 39


E.2 Dimensions of Fixed-point Subspaces for Subgroups of _SO_ (3) or _O_ (3) . . . . . . 40


E.3 Symmetry Infimum for Subgroups of _SO_ (3) . . . . . . . . . . . . . . . . . . . . . 41


E.4 Symmetry Infimum for Subgroups of _O_ (3) . . . . . . . . . . . . . . . . . . . . . 42


**F** **Detailed Experiment** **45**


F.1 Visualization of Representation Space . . . . . . . . . . . . . . . . . . . . . . . . 45


F.2 Expressivity on Symmetric Graphs . . . . . . . . . . . . . . . . . . . . . . . . . . 45


F.2.1 Embedding Difference Norm Experiment . . . . . . . . . . . . . . . . . . 45


F.2.2 Original GWL-Test on Symmetric Graphs . . . . . . . . . . . . . . . . . . 45


F.3 Molecule Property Prediction with Pretrained Equivariant Features . . . . . . . . . 46


F.3.1 Detailed Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . 46


F.3.2 Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46


15


A BACKGROUNDS


A.1 EQUIVARIANT NEURAL NETWORKS


Equivariant Neural Networks (ENNs) have emerged as a cornerstone of modern machine learning,
enabling a wide range of applications across the sciences (Han et al., 2025; Wu et al., 2023; Li et al.,
2025b; Wu et al., 2025; Li et al., 2025c; Feng et al., 2026; Cao et al., 2026; Wu et al., 2026; Li
et al., 2025a). Most mainstream ENNs adopt tensor product operators to design the message passing
architecture (Thomas et al., 2018). In particular, since scalarization operations ( _e.g._, norm and inner
product) can substantially reduce computational cost, Cartesian vector-based networks (Satorras et al.,
2021) and spherical scalarization networks (Cen et al., 2024; Aykent & Xia, 2025) have become
particularly popular. More specifically for asymmetric graph structures, Cen et al. (2025) point out
that networks employing scalarization operations already possess sufficient expressive power, i.e.,
universal approximation.


However, additional subtleties arise in the presence of input symmetries. The interaction between
architectural components and symmetric structures can lead to nontrivial representational effects. For
example, techniques such as global virtual nodes (Zhang et al., 2024; 2025) and reference frames
(Puny et al., 2022; Duval et al., 2023), though effective in improving model capacity, may exhibit
unintended behaviors when applied to symmetric inputs. In particular, their use can potentially induce
symmetry increase. In this work, we systematically analyze these phenomena under symmetric
structures for general equivariant neural network architectures.


A.2 CLOSED SUBGROUPS OF _SO_ (3) OR _O_ (3)


Before proceeding, we briefly review the classification of the closed subgroups of _SO_ (3) or _O_ (3).
The closed subgroups of _O_ (3), also known as point groups, are classified up to conjugacy as follows.
Throughout this paper, we use the Schoenflies notation to denote these groups.


    - **Finite** **Subgroups** : These are divided into axial and polyhedral groups. The axial
groups include the Abelian subgroups ( _Ck, S_ 2 _k, Ckh_ ) and the non-Abelian subgroups
( _Ckv, Dk, Dkh, Dkd_ ). The polyhedral groups are all non-Abelian and comprise seven
families ( _T, Td, Th, O, Oh, I, Ih_ ). Among these, the groups _Ck, Dk, T, O, I_ consist solely
of pure rotations and are subgroups of _SO_ (3).


    - **Infinite Subgroups** : These include the cylindrical groups ( _C∞, C∞h, C∞v, D∞, D∞h_ ),
which arise as limits of the axial groups, and the spherical groups, which are _K_ = _SO_ (3)
and _Kh_ = _O_ (3).


To facilitate the identification of input symmetries, we now provide a brief overview of the elemental
structure of the key finite subgroups. The identification of cylindrical symmetries, which are infinite,
can be achieved by taking the limits of these finite axial groups.


ABELIAN AXIAL GROUPS


    - _Ck_ **(Cyclic Group):** Generated by a rotation _ck_ of angle 2 _π/k_, with _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

    - _S_ 2 _k_ **(Rotation-Reflection Group):** Generated by adding a rotation-reflection element _c_ 2 _kσh_
to _Ck_, with 2 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) Rotation-reflections _c_ [2] 2 _[j]_ _k_ [+1] _σh_ about the principal axis.


    - _Ckh_ **(Cyclic** **Groups** **with** **Horizontal** **Reflection):** Generated by adding a horizontal
reflection plane _σh_ to _Ck_, with 2 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) Rotation-reflections _c_ _[j]_ _k_ _[σ][h]_ [ about the principal axis.]


16


NON-ABELIAN AXIAL GROUPS


    - _Ckv_ **(Cyclic Groups with Vertical Reflections):** Generated by adding a vertical reflection
plane _σv_ to _Ck_, with 2 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) Reflections _c_ _[j]_ _k_ _[σ][v]_ [about the vertical plane.]

    - _Dk_ **(Dihedral Groups):** Generated by adding a 2-fold rotation axis _u_ 2 perpendicular to the
principal axis of _Ck_, with 2 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) 2-fold rotations _c_ _[j]_ _k_ _[u]_ [2][ about horizontal axes.]

    - _Dkh_ **(Dihedral Groups with Horizontal Reflection):** Generated by adding a horizontal
reflection plane _σh_ to _Dk_, with 4 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) 2-fold rotations _c_ _[j]_ _k_ _[u]_ [2][ about horizontal axes.]

(3) Rotation-reflections _c_ _[j]_ _k_ _[σ][h]_ [ about the principal axis.]

(4) Reflections _c_ _[j]_ _k_ _[σ][v]_ [about the vertical plane.]

    - _Dkd_ **(Dihedral** **Groups** **with** **Dihedral** **Reflections):** Generated by adding a diagonal
reflection plane _σd_ = _c_ 2 _kσv_ to _Dk_, with 4 _k_ elements:

(1) Rotations _c_ _[j]_ _k_ [about the principal axis.]

(2) 2-fold rotations _c_ _[j]_ _k_ _[u]_ [2][ about horizontal axes.]

(3) Rotation-reflections _c_ [2] 2 _[j]_ _k_ [+1] _σh_ about the principal axis.

(4) Reflections _c_ [2] 2 _[j]_ _k_ [+1] _σv_ about the diagonal plane.


POLYHEDRAL GROUPS


    - _T_ **(Tetrahedral Group):** The rotational symmetry group of a tetrahedron, with 12 elements.

    - _Td_ **(Full Tetrahedral Group):** The full symmetry group of a tetrahedron, with 24 elements.

    - _Th_ **(Pyritohedral Group):** Generated by adding an inversion center to _T_, with 24 elements.

    - _O_ **(Octahedral Group):** The rotational symmetry group of a cube or octahedron, with 24
elements.

    - _Oh_ **(Full Octahedral Group):** The full symmetry group of a cube or octahedron, generated
by adding an inversion center to _O_, with 48 elements.

    - _I_ **(Icosahedral Group):** The rotational symmetry group of an icosahedron, with 60 elements.

    - _Ih_ **(Full Icosahedral Group):** The full symmetry group of an icosahedron, generated by
adding an inversion center to _I_, with 120 elements.


The subgroup relations among point groups, as well as the dimensions of fixed-point spaces of
_O_ (3) representations, can be readily determined. In particular, they can be derived from the minimal
subgroup relations provided in § E.1 and from the dimension table for fixed-point spaces of irreducible
_O_ (3) representations established in § E.2.


A.3 MANIFOLDS AND BASIC DATA ASSUMPTION


Following the definition in Milnor (1990), a _C_ _[r]_ manifold _M_ is mathematically defined as a subset of
some linear space R _[d]_ . For every point _x_ in this subset, there must exist a neighborhood _W_ in R _[d]_ such
that _W_ _∩_ _M_ is _C_ _[r]_ -diffeomorphic to an open set in another linear space R _[l]_ . The integer _l_ is known as
the dimension of _M_ . An alternative, intrinsic definition that does not depend on an embedding space
can also be found in Hirsch (1976).


Machine learning often assumes that the data distribution is supported on a submanifold of the input
space. The dimension of this manifold characterizes the directions in which the data can vary within


17


the input space. However, the term manifold is often used loosely in this context and does not
perfectly align with its mathematical counterpart (Goodfellow et al., 2016). First, the data manifold
may have self-intersections, causing the local dimension of data variation to differ at various points.
Second, data belonging to different classes or clusters may possess different structures and potentially
different dimensions. Lastly, stochastic factors such as observational noise can prevent the data from
forming a strict surface in the input space. We will not address this last factor, instead, we assume
that the interference from noise is minimal and ignorable.


To address the first two issues, we can relax the manifold hypothesis by assuming that the data is
supported on a finite union of submanifolds within the input space. Since a manifold with multiple
connected components can itself be viewed as a disjoint union of connected manifolds, we can
assume, without loss of generality, that each of these submanifolds is connected. For theoretical
convenience, we further assume that these manifolds are bounded and closed. By the Weierstrass
theorem, this is equivalent to assuming they are compact. We briefly outline the justification for these
assumptions below:


    - **Finite Union of Manifolds:** This assumption has been partially verified experimentally on
computer vision datasets (Brown et al., 2022). Theoretically, it encompasses classic cases of
self-intersection. A data manifold may arise as the image of a map. However, the image of
an immersion or submersion of a manifold can have self-intersections. The image of such a
map from a compact manifold is, however, a finite union of compact manifolds.


    - **Boundedness of Manifolds:** This assumption stems from the natural assumption that the
data distribution itself is bounded.


    - **Closedness** **of** **Manifolds:** This assumption covers scenarios where the data is defined
by a set of well-behaved constraints. For example, the set of points satisfying _n_ independent, differentiable constraint equations in the input space forms a closed submanifold of
codimension _n_ .


In equivariant representation learning, the data possesses symmetries, meaning that data corresponding
to the same physical object can be transformed by symmetry operations to represent different reference
frames. These symmetry transformations typically form a group, and since the transformed data is
still valid data, the data manifold must be closed under these group transformations, that is, the data
manifold must be invariant under the group action. Considering the group action for any given input,
we summarize the preceding discussions into the following assumptions about the data:


    - **Lie Group Assumption:** The transformation group _G_ is a compact Lie group.


    - **Manifold** **Assumption:** The input space _X_ is a linear space equipped with a linear
action _ρX_ of the group _G_ . That is, there exists a map _ρX_ : _G_ _GL_ ( _X_ ) such that
_→_
_ρX_ ( _g_ 1 _g_ 2) = _ρX_ ( _g_ 1) _ρX_ ( _g_ 2), and the identity element of the group is mapped to the identity
transformation. The data manifold _M_ is a finite union of compact, connected, smooth, and
_G_ -invariant submanifolds of _X_ .


The smoothness assumption is added for theoretical convenience and the connectedness assumption
does not impose additional restrictions, since each compact smooth submanifold admits only finitely
many connected components.


18


B PROOFS OF THE INFIMUM OF SYMMETRY (§ 3)


B.1 PROOF OF THM. 3.1


**Lemma B.1** (Azzi et al. (2023), Cor. 2.3) **.** _For a compact Lie group G and a representation V, let_
_v_ _∈_ _V_ _be any point._ _There exists a neighborhood U_ _of v_ _such that for any point u_ _∈_ _U_ _, we have_
( _Gu_ ) ( _Gv_ ) _._
_≤_

**Lemma B.2** (Azzi et al. (2023), Cor. 2.20) **.** _For a reductive group G and an affine algebraic variety_
_X, let x ∈_ _X_ _be a point with a closed orbit G_ ( _x_ ) _._ _There exists a Zariski neighborhood U_ _of x such_
_that for any point u_ _U_ _, we have_ ( _Gu_ ) ( _Gx_ ) _._
_∈_ _≤_

We use the definition of the complexification from Prop. 3.3 of Azzi et al. (2023). For a real vector
space _V_, its complexification is _V_ [C] := C _⊗_ _V_ . Considering an orthogonal action of _G_ on _V_, with g
being the Lie algebra of _G_, the complexification of the group is _G_ [C] := _{g_ exp( _iX_ ) _| g_ _∈_ _G, X_ _∈_ g _}_ .
By the above complexification, we obtain the following lemma.
**Lemma** **B.3.** _For_ _a_ _compact_ _Lie_ _group_ _G,_ _consider_ _the_ _inclusion_ _map_ _ι_ : _G_ _�→_ _G_ [C] _into_ _its_
_complexification._ _Let_ _H_ _and_ _K_ _be_ _subgroups_ _of_ _G_ _and_ _let_ _g_ 0 _G_ _be_ _an_ _element_ _such_ _that_
_∈_
_g_ 0 _Hg_ 0 _[−]_ [1] _K._ _Then,_
_⊆_
_ι_ ( _g_ 0) _H_ [C] _ι_ ( _g_ 0) _[−]_ [1] _K_ [C] _._ (12)
_⊆_


_Proof._ Let h and k be the Lie algebras of _H_ and _K_, respectively. Any element of _H_ [C] can be expressed
as _h_ 0 exp( _iX_ 0) for _h_ 0 _∈_ _H_ and _X_ 0 _∈_ h. Its conjugation by _g_ 0 is

_g_ 0� _h_ 0 exp( _iX_ 0)� _g_ 0 _[−]_ [1] = ( _g_ 0 _h_ 0 _g_ 0 _[−]_ [1][) exp(] _[i]_ [Ad] _[g]_ 0 _[X]_ [0][)] _[.]_ (13)

Since _g_ 0 _Hg_ 0 _[−]_ [1] _K_, the term _g_ 0 _h_ 0 _g_ 0 _[−]_ [1] belongs to _K_ . The expression thus belongs to _K_ [C] provided
_⊆_
that Ad _g_ 0 _X_ 0 k.
_∈_

This condition on the Lie algebra is obtained by differentiating the subgroup inclusion _g_ 0 _Hg_ 0 _[−]_ [1] _K_
_⊆_
at the identity element, which gives Ad _g_ 0(h) k. As _X_ 0 h, it follows immediately that Ad _g_ 0 _X_ 0
_⊆_ _∈_ _∈_
k, which completes the proof.


**Proposition B.4.** _Let V_ _be a representation of a compact Lie group G._ _For any closed subgroup H_
_of G, the set of points in the fixed-point subspace V_ _[H]_ _that have the minimal orbit type is a dense and_
_open subset of V_ _[H]_ _._


_Proof._ The proof strategy is similar to that of Prop. 2.10 in Azzi et al. (2023). We first show that the
set of points with a minimal orbit type is open, then use complexification to show it is Zariski-open,
which implies density.

First, to prove openness, let ( _K_ ) be a minimal orbit type in _V_ _[H]_ and let _x_ _∈_ _V_ _[H]_ be a point with
( _Gx_ ) = ( _K_ ). By Lem. B.1, there exists a neighborhood _U_ of _x_ such that for any _u_ _U_, ( _Gu_ ) ( _K_ ).
Since ( _K_ ) is minimal in _V_ _[H]_, any point in the open set _U_ _∩_ _V_ _[H]_ must have orbit type _∈_ ( _K_ ) _≤_ . Thus,
openness is established.

Next, we consider the complexifications _G_ [C] and _V_ [C] . For a point _x_ _∈_ _V_ with orbit type ( _K_ ), by
Lem. 3.13 of Azzi et al. (2023), we have ( _G_ [C] ) _x_ = ( _Gx_ ) [C] = _K_ [C] . By Prop. 3.14 of Azzi et al. (2023),
the orbit _G_ [C] ( _x_ ) is closed. Therefore, by Lem. B.2, there exists a Zariski neighborhood _U_ of _x_ in
_V_ [C] such that any point in this neighborhood has an isotropy type less than or equal to ( _K_ [C] ). The
set _U_ _[′′]_ _U_ = _[′]_ _U_ = _[′]_ _∩U_ _V∩_ contains a non-empty real Zariski-open subset of( _V_ [C] ) _[H]_ [C] is Zariski-open in ( _V_ [C] ) _[H]_ [C] . By Cor. A.7 of _V_ _[H]_ Azzi et al.. (2023), its real part

We now show that for any _y_ _U_ _[′′]_, its orbit type is ( _Gy_ ) = ( _K_ ). Since _y_ _V_ _[H]_, we have
_∈_ _∈_
( _K_ ) ( _Gy_ ), which implies there exists _g_ 1 _G_ such that _g_ 1 _Kg_ 1 _[−]_ [1] _Gy_ . By Lem. B.3, this gives
_≤_ _∈_ _⊆_
_g_ 1 _K_ [C] _g_ 1 _[−]_ [1] ( _Gy_ ) [C] . Since _y_ _U_ _[′′]_ _U_, we have (( _G_ [C] ) _y_ ) ( _K_ [C] ), which means there exists
_⊆_ _∈_ _⊂_ _≤_
_g_ 2 _∈_ _G_ [C] such that ( _Gy_ ) [C] _⊆_ _g_ 2 _K_ [C] _g_ 2 _[−]_ [1][.] [Combining these inclusions yields]


_g_ 1 _K_ [C] _g_ 1 _[−]_ [1] _⊆_ ( _Gy_ ) [C] _⊆_ _g_ 2 _K_ [C] _g_ 2 _[−]_ [1] _[.]_ (14)


By Lem. A.3 of Azzi et al. (2023), this implies that the real subgroups are conjugate, i.e., ( _Gy_ ) =
( _g_ 1 _Kg_ 1 _[−]_ [1][) = (] _[K]_ [)][.] [This completes the proof of density.]


19


**Theorem 3.1** (Uniqueness of Minimal Type) **.** _Let X_ _be a representation of a compact Lie group_
_G._ _For any closed subgroup H_ _⊆_ _G, a unique minimal orbit type exists among the points in the_
_fixed-point_ _subspace_ _X_ _[H]_ _._ _In_ _particular,_ _if_ ( _H_ ) _G_ ( _X_ _[H]_ ) _,_ _then_ ( _H_ ) _is_ _the_ _minimal_ _orbit_ _type_
_∈O_
_within that subspace._


_Proof of Thm. 3.1._ The density and openness of the set of points corresponding to any minimal orbit
type is established by Prop. B.4. The uniqueness then follows from the fact that two distinct open
dense sets must have a non-empty intersection.


B.2 PROOF OF THM. 3.3


**Theorem 3.3.** _A necessary condition for the existence of an isovariant map relative to Y_ _from a_
_G-set_ _X_ _to_ _a_ _G-set_ _Y_ _is_ _that_ ( _pY_ ( _H_ )) _∈OG_ ( _Y_ ) _for_ _every_ ( _H_ ) _∈OG_ ( _X_ ) _._ _When_ _X_ _and_ _Y_ _are_
_representations, this is equivalent to the condition that IG_ ( _Y, H_ ) = ( _pY_ ( _H_ )) _for all_ ( _H_ ) _∈OG_ ( _X_ ) _._


_Proof of Thm. 3.3._ The equivalence for representations follows directly from applying the operator
_pY_ to the known partial order ( _H_ ) _≤_ _IG_ ( _Y, H_ ) _≤_ ( _pY_ ( _H_ )). Since _pY_ is order-preserving and fixes
isotropy subgroups in _Y_, the inequality collapses to an equality.


20


C SYMMETRY INCREASE FOR _G_ = _SO_ (3) OR _O_ (3) IN § 4


C.1 GENERAL ORBIT TYPE CRITERION


To derive Prop. 4.2, we need more general results require criteria for identifying orbit types of general
representation. Given the sufficiency of the Ihrig-Golubitsky Criterion (Prop. C.1), we demonstrate
that for high-multiplicity representations, the conditions of the Michel Criterion imply the conditions
of the Ihrig-Golubitsky Criterion. This, in turn, establishes the sufficiency of the Michel Criterion.

Let _G_ be a compact Lie group and _X_ be the input space. The normalizer of a subgroup _H_ _⊆_ _G_
is _NG_ ( _H_ ) := _g_ _G_ _gHg_ _[−]_ [1] = _H_ . The normalizer of _H_ relative to a supergroup _H_ _[′]_ _H_ is
_{_ _∈_ _|_ _}_ _⊃_
_NG_ ( _H, H_ _[′]_ ) := _g_ _G_ _H_ _gH_ _[′]_ _g_ _[−]_ [1] . Based on the above definitions, we recall the following
_{_ _∈_ _|_ _⊂_ _}_
general criterion.


**Proposition C.1** (Ihrig-Golubitsky Criterion, Ihrig & Golubitsky (1984), Prop. 5.3) **.** _Let_ _V_ _be a_
_faithful representation of a group G._ _A sufficient condition for a closed subgroup H_ _to be an isotropy_
_subgroup in V_ _is that for every orbit type_ ( _H_ _[′]_ ) _in V_ _with_ ( _H_ _[′]_ ) _>_ ( _H_ ) _, the following inequality holds:_


dim _V_ _[H]_ _[′]_ + _αG_ ( _H, H_ _[′]_ ) := dim _V_ _[H]_ _[′]_ + dim _NG_ ( _H, H_ _[′]_ ) dim _NG_ ( _H_ _[′]_ ) _<_ dim _V_ _[H]_ _,_ (15)
_−_

_where αG_ ( _H, H_ _[′]_ ) _is the Ihrig-Golubitsky correction term._


_Remark_ . Ihrig & Golubitsky (1984) states that the condition is also necessary for _G_ = _SO_ (3) _, O_ (3).


Compared to Prop. C.1, the Linehan-Stedman Criterion (Linehan & Stedman, 2001) is more convenient due to its structure, which only requires checking adjacent subgroups. Since the associated
theoretical results are not needed in our proofs, we do not elaborate on them here.


C.2 PROOF OF PROP. 4.2


**Proposition 4.2.** _For a high-multiplicity representation V, the necessary condition stated in Thm. 4.1_
_is also sufficient._


_Proof of Prop. 4.2._ We show that for representations where each irreducible component has multiplicity _r_ _>_ dim _G_, the necessary condition from Michel’s Criterion (Thm. 4.1) becomes sufficient
by proving it implies the condition in Prop. C.1. We may assume without loss of generality that the
representation is faithful. For unfaithful representations, we proceed by factoring out the kernel from
_G_ and invoking Prop. C.1, as the procedure remains unchanged.

It follows from the condition that for any closed subgroup _H_ _[′]_ of _G_ containing _H_, we have dim _V_ _[H]_ _[′]_ _<_
dim _V_ _[H]_ . The condition implies that for at least one irreducible component _Vi_, we must have
dim _Vi_ _[H]_ _[′]_ _<_ dim _Vi_ _[H]_ [.] [Since dimensions are integers, this is equivalent to][ dim] _[ V]_ _i_ _[H]_ _[′]_ + 1 _≤_ dim _Vi_ _[H]_ [.]
Given that the multiplicity _m_ ( _Vi, V_ ) _>_ dim _G_, it follows that

_m_ ( _Vi, V_ ) dim _Vi_ _[H]_ _[′]_ + dim _G < m_ ( _Vi, V_ ) dim _Vi_ _[H]_ _[.]_ (16)


Summing this strict inequality for one such component with the non-strict inequalities for all other
components yields
dim _V_ _[H]_ _[′]_ + dim _G <_ dim _V_ _[H]_ _._ (17)


The term from Prop. C.1 is bounded by


0 _αG_ ( _H, H_ _[′]_ ) = dim _NG_ ( _H, H_ _[′]_ ) dim _NG_ ( _H_ _[′]_ ) dim _G._ (18)
_≤_ _−_ _≤_

The sufficiency of Michel’s condition follows from combining these two results:

dim _V_ _[H]_ _[′]_ _<_ dim _V_ _[H]_ = _⇒_ dim _V_ _[H]_ _[′]_ + dim _G <_ dim _V_ _[H]_ (19)

= dim _V_ _[H]_ _[′]_ + dim _NG_ ( _H, H_ _[′]_ ) dim _NG_ ( _H_ _[′]_ ) _<_ dim _V_ _[H]_ _,_ (20)
_⇒_ _−_

where the final implication uses the upper bound from Eq. (18). This shows that Michel’s condition
implies the sufficient condition from Prop. C.1.


21


C.3 DETAILED CALCULATION IN EX. 4.3


We now conduct the orbit-type test and symmetry infimum calculation for the geometric symmetry
_Dkh_ ( _k_ _>_ 2) of the _k_ -fold from Ex. 2.2. Consider _Dkh_ as a closed subgroups of _O_ (3), the calculation
is performed in the representation space _Y_ = _Vl_ _[⊕]_ = _[r]_ _l_ 0 [for] _[r]_ _[>]_ [3] [and] _[l]_ [0] _[>]_ [0][.] [We] [will] [use] [the] [pre-]
computed subgroup relations from Table 6, the dimensions of the fixed-point spaces for _Dkh_ and _Oh_
in _Vl_ = _l_ 0 from Table 8, and the orbit type test results for ( _D∞h_ ) and ( _Oh_ ) in _Vl_ _[⊕]_ = _[r]_ _l_ 0 [.]

First, we perform the orbit type test. Consider the case where _k_ = 4. In this situation, the adjacent
supergroups of _Dkh_ are of the form _Dpk,h_, where _p_ is a prime number. We classify the discussion
into 12 cases based on the parity of _k_ and _l_ 0, and the ranges _l_ 0 _<_ _k_, _k_ _l_ 0 _<_ 2 _k_, and _l_ 0 2 _k_,
_≤_ _≥_
and calculate the dimensions of the fixed-point spaces of _Vl_ = _l_ 0 for _Dkh_ and _Dpk,h_ . The calculated
dimensions are shown in Table 2. For the case _k_ = 4, we must additionally consider the supergroup
_Oh_ . Here, dim _Vl_ _[D]_ = _l_ [4] 0 _[h]_ [= dim] _[ V]_ _l_ _[O]_ = _[h]_ _l_ 0 [only when] _[ l]_ [0][ is odd, which aligns with the results from Table][ 2][.]
In all other cases, _Oh_ does not affect the orbit type test of ( _Dkh_ ). Thus, we find that ( _Dkh_ ) is an orbit
type when _l_ 0 _≥_ _k_ and when _l_ 0 and _k_ have the same parity.


Table 2: Dimension of fixed-point space for supergroups of _Dkh_ ( _k_ _>_ 2) in _Vl_ _[⊕]_ = _[r]_ _l_ 0 [(] _[r]_ _[>]_ [ 3][), organized]
by the parity of _k_ and _l_ 0.

_l_ 0 _< k_ _k_ _l_ 0 _<_ 2 _k_ _l_ 0 2 _k_
_≤_ _≥_
_l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd
_k_ is even
_Dkh_ 1 0 2 0 _d_ 0
_Dpkh_ 1 0 1 0 _< d_ 0
_k_ is odd
_Dkh_ 1 0 1 1 _d_ _d_
_D_ 2 _kh_ 1 0 1 0 _d_ 0
_Dp∗kh_ 1 0 1 0 _< d_ _< d_


Next, we proceed with the symmetry infimum calculation. Again, we first consider the case where
_k_ = 4. In this situation, the supergroups of _Dkh_ are _Dpk,h_, _D∞h_, and _O_ (3), where _O_ (3) is always
an orbit type. Using the orbit type test results just calculated and those pre-computed, we analyze the
cases based on the parity of _k_ and _l_ 0 as before, with the results summarized in Table 3. For _k_ = 4,
the additional supergroup _Oh_ does not affect the final result. This is because the supergroup _Oh_ is
never the minimal isotropy supergroup, as it only becomes an orbit type when ( _D_ 4 _h_ ) already is. This
leads us to the symmetry infimums shown in Table 1.


Table 3: Isotropy conditions for supergroups of _Dkh_ ( _k_ _>_ 2) in _Vl_ _[⊕]_ = _[r]_ _l_ 0 [(] _[r]_ _[>]_ [ 3][), organized by the parity]
of _k_ and _l_ 0. A ✓ indicates the condition is satisfied, and ✗ that it is not. In the subgroup notation, _p_
denotes a prime number and _p_ _[∗]_ denotes an odd prime number.

_l_ 0 _< k_ _k_ _l_ 0 _<_ 2 _k_ _l_ 0 2 _k_
_≤_ _≥_
_l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd _l_ 0 is even _l_ 0 is odd
_k_ is even
_Dkh_ ✗ ✗ ✓ ✗ ✓ ✗
_Dpkh_ ✗ ✗     - ✗     - ✗
_D∞h_ ✓ ✗    - ✗    - ✗
_k_ is odd
_Dkh_ ✗ ✗ ✗ ✓ ✗ ✓
_D_ 2 _kh_ ✗ ✗ ✗   - ✓   _Dp∗kh_ ✗ ✗ ✗     -     -     _D∞h_ ✓ ✗ ✓    -    -    

22


C.4 CALCULATION OF SYMMETRY INFIMUM

We calculate the orbit types for the _SO_ (3) representation _Vl_ _[⊕]_ = _[r]_ _l_ 0 [and the] _[ O]_ [(3)][ representation] _[ V]_ _l_ _[⊕]_ = _[r]_ _l_ 0 _[±]_ [.]
As the calculation procedure is highly similar to that in Ex. 4.3, we omit the detailed steps here. For
the case of _r_ = 1, the results can be found in Table B.1 and Table B.2 of Linehan & Stedman (2001).


The orbit types are calculated using the procedure in Algo. 1. This calculation requires the dimensions
of the fixed-point spaces for the closed subgroups of _SO_ (3) or _O_ (3) in the representations _Vl_ = _l_ 0 and
_Vl_ = _l_ 0 _±_ [, respectively, which are provided in §][ E.2][.]


According to Prop. 6.2 from Ihrig & Golubitsky (1984), the Ihrig-Golubitsky correction term
_α_ ( _H, H_ _[′]_ ) = 0 for all subgroups except _H_ = _Ck_ in _SO_ (3), and for all subgroups except
_H_ = _Ck, S_ 2 _k, Ckh_ in _O_ (3). Therefore, for all subgroups other than _H_ = _Ck, S_ 2 _k, Ckh_, our results
are identical to those calculated by Linehan & Stedman (2001) for _r_ = 1. We present only the results
of whether ( _Ck_ ) _∈OSO_ (3)( _Vl_ _[⊕]_ = _[r]_ _l_ 0 [)][ on Table][ 4][ and whether][ (] _[C][k]_ [)] _[,]_ [ (] _[S]_ [2] _[k]_ [)] _[,]_ [ (] _[C][kh]_ [)] _[ ∈O][O]_ [(3)][(] _[V]_ _l_ _[⊕]_ = _[r]_ _l_ 0 _[−]_ [)][ on]
Table 5.

The reason for omitting the discussion of _V_ _[⊕][r]_
_l_ = _l_ 0 [+] [is consistent with the explanation in the header of]
Table B.2 in Linehan & Stedman (2001). This is because the corresponding conclusions can be
found in the orbit type table for the _SO_ (3) representation _Vl_ _[⊕]_ = _[r]_ _l_ 0 [via] [the] [mappings] _[D][∞]_ _[→]_ _[D][∞][h]_ [,]
_C∞_ _C∞h_, _I_ _Ih_, _O_ _Oh_, _T_ _Th_, as well as _Dk_ _Dkh_ and _Ck_ _Ckh_ for even _k_, and
_→_ _→_ _→_ _→_ _→_ _→_
_Dk_ _Dkd_ and _Ck_ _S_ 2 _k_ for odd _k_ .
_→_ _→_


Table 4: Modified isotropy subgroups for the multiple irreducible representation _Vl_ _[⊕]_ = _[r]_ _l_ 0 _[, r]_ _[>]_ [3] [of]
_SO_ (3) obtained via the Michel criterion.

_H_ Condition of _l_ 0

_Ck_ _l_ 0 _k_
_≥_


Table 5: Modified isotropy subgroups for the multiple irreducible representation _V_ _[⊕][r]_ _[>]_ [3][ of]
_l_ = _l_ 0 _[−]_ _[, r]_

_O_ (3) with nontrivial reflection, obtained via the Michel criterion.

_H_ Condition of _l_ 0

_Ck_ _l_ 0 _k_
_≥_
_S_ 2 _k_ ( _k_ is even) _l_ 0 _k_
_≥_
_Ckh_ ( _k_ is odd) _l_ 0 _k_
_≥_


The complete set of orbit type calculation results can also be found in the tables for the symmetry
infimum. This is because ( _H_ ) is an orbit type of a representation _V_ if and only if the symmetry
infimum of _H_ in _V_ is ( _H_ ) itself. These non-degenerate cases are highlighted in green in the tables.


Using the calculated orbit types for the _SO_ (3) representation _Vl_ _[⊕]_ = _[r]_ _l_ 0 [and] [the] _[O]_ [(3)] [representation]
_V_ _[⊕][r]_ [we can now compute the symmetry infimum for all subgroups of] _[ SO]_ [(3)][ or] _[ O]_ [(3)][ in these]
_l_ = _l_ 0 _[±]_ [,]
representations. When calculating the symmetry infimum for all subgroups, the algorithm differs
slightly from that in Ex. 4.3. We reduce the need to enumerate all supergroups by leveraging the
symmetry infimums of adjacent supergroups. An improved version of Algo. 2 is detailed in Algo. 3.


To reduce the computational load, this algorithm employs a top-down calculation strategy. First, we
compute the results for the infinite groups, followed by the polyhedral groups. Finally, we calculate
the results for the axial groups in the order _Dkh, Dkd, Dk, Ckv, Ckh, S_ 2 _k, Ck_ . Due to the exceptional
subgroup relations shown in Table 6, special consideration for additional supergroups is required for
certain cases where _k_ _∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _,_ 5 _}_ . These cases are handled last.

The results for _SO_ (3) are presented in § E.3, and the results for _O_ (3) are in § E.4. Here, we provide
a general classification for the observed symmetry increases. For a closed subgroup _H_, an increase


23


**Algorithm 3:** Symmetry Infmum Calculation With Precomputed Results

**Data:** A symmetry group _G_ ;
a closed subgroup _H_ _⊂_ _G_ ;
a Rep. _V_ of _G_ ;
a map _M_ of previously computed symmetry infs.
**Result:** Symmetry inf. _IG_ ( _V, H_ ).

**1** **if** is_in(( _H_ ) _,_ _G_ ( _V_ )) **then**

**2** **return** ( _H_ ) _O_ ;

**3** **end**


**4** Let _S_ 0 = _Hi_ _G_ to be all adjacent closed supergroups of _H_ in _G_ ;
_{_ _} ⊂_


**5** _O_ _←_ ∅;


**6** **for** _Hi_ _in S_ 0 **do**


**7** **if** is_in(( _Hi_ ) _,_ _G_ ( _V_ )) **then**
_O_


**8** Add ( _Hi_ ) to the set ;
_O_


**9** **end**


**10** **else if** _Hi_ _is a key in the map M_ **then**


**11** Add _M_ [ _Hi_ ] = _IG_ ( _V, Hi_ ) to the set ;
_O_


**12** **end**

**13** **else**


**14** Let _Si_ = _Kj_ to be all adjacent closed supergroups of _Hi_ in _G_ ;
_{_ _}_


**15** **for** _Kj_ _in Si_ **do**


**16** **if** ( _Kj_ ) _G_ ( _V_ ) **then**
_∈O_


**17** Add ( _Kj_ ) to the set ;
_O_


**18** **end**


**19** **end**

**20** **end**

**21** **end**

**22** **return** min( _O_ );


to the full group _O_ (3) or _SO_ (3) is termed **full degeneration** and marked in red in the tables. An
increase to a supergroup of higher dimension than _H_ is termed **continuous degeneration** and marked
in blue . An increase to a supergroup of the same dimension as _H_ is termed **discrete degeneration**
and marked in yellow or light green . No symmetry increase is **no degeneration** and marked in


green .


For the subgroup _Dkh_ _O_ (3), the classification of degeneration behaviors given after Ex. 4.3 is a
_⊂_
special case of this general framework. Specifically, full degeneration is identical to the definition
here, axial degeneration corresponds to continuous degeneration, and half degeneration corresponds
to discrete degeneration.

We note that for the representation _Y_ = _V_ _[⊕][r]_
_l_ = _l_ 0 [+][, the action of] _[ G]_ [ has a non-trivial kernel, and thus]

symmetry increase is inevitable. Let _πY_ be the natural projection to the quotient _G/_ ker _ρY_ In this
context, the light green marks an increase to _πY_ _[−]_ [1][(] _[π][Y]_ [ (] _[H]_ [))][,] [which,] [as] [explained] [in] [§] [3.2][,] [is] [the]
lowest possible symmetry that _H_ can be reduced to when a non-trivial kernel is present. This
is therefore a predictable behavior. The yellow marks other exceptional cases within discrete
degeneration.


C.5 COMPOSITION PROPERTY OF HIGH-MULTIPLICITY REPRESENTATION


In the tables of § E.3 and § E.4, there exists a special class of subgroups _H_ for which any non-trivial
symmetry increase always results in an infimum ( _H_ 0) where _H_ 0 is an adjacent supergroup. We
call ( _H_ 0) the **bottleneck** of _H_, denoted by _BG_ ( _H_ ), and we say that a subgroup _H_ that possesses
a bottleneck satisfies the **bottleneck condition** . These groups that act as bottlenecks satisfy some
elegant properties. To demonstrate this, we first prove a structure theorem for the symmetry infimum
of high-multiplicity representations.


24


**Proposition C.2.** _For a high-multiplicity representation V, the lowest orbit type in the fixed-point_
_subspace V_ _[H]_ _is_ ( _GV_ _H_ ) _, where GV_ _H_ = [�] _x∈V_ _[H]_ _[G][x][ is the largest subgroup that leaves][ V]_ _[H]_ _[invariant.]_

_This shows that IG_ ( _V, H_ ) = ( _GV_ _H_ ) _._


_Proof._ Since any element in _V_ _[H]_ has at least the symmetry _GV_ _H_, we only need to prove that _GV_ _H_ is
an isotropy subgroup. Assume, for the sake of contradiction, that _GV_ _H_ is not an isotropy subgroup.
By the sufficiency of the Michel Criterion, there exists a supergroup _K_ such that _GV_ _H_ ⊊ _K_ and

dim _V_ _[H]_ = dim _V_ _[G][V H]_ = dim _V_ _[K]_ = _⇒_ _V_ _[H]_ = _V_ _[G][V H]_ = _V_ _[K]_ _._ (21)

This means that _K_ leaves all elements of _V_ _[H]_ fixed. From this, we derive a contradiction:


        _K_ _Gx_ = _GV_ _H_ _._ (22)
_⊂_

_x∈V_ _[H]_


_Remark_ . This result implies that for a high-multiplicity representation, the symmetry increase from
( _H_ ) to _IG_ ( _V, H_ ) does not alter the dimension of the fixed-point space. Therefore, for high-multiplicity
representations, the dimension of the fixed-point subspace corresponding to the input symmetry group
equals that corresponding to the symmetry infimum. Since the fixed-point subspace dimensions for
certain closed subgroups exhibit distinct regularities, the behavior of symmetry increase toward these
subgroups serves as an indicator of the underlying subspace dimension. For instance, in the cases
of _SO_ (3) or _O_ (3), for non-trivial representations, full degeneration corresponds to a 0-dimensional
fixed-point subspace, whereas for finite input subgroups, continuous degeneration corresponds to
a 1-dimensional subspace. However, we must emphasize that when a quantitative assessment of
the equivariant feature’s expressive capacity is required, relying solely on the symmetry infimum is
insufficient, as it yields only coarse, qualitative insights. In such cases, one should directly compute
the fixed-point subspace dimension. For calculations related to _SO_ (3) or _O_ (3), see § E.2.


We can prove that these groups acting as bottlenecks satisfy the property that for any irreducible representation _V_ 0, if dim _V_ 0 _[H]_ _<_ dim _V_ 0 _[H]_ [0], then dim _V_ 0 _[H]_ _<_ dim _V_ 0 _[H]_ _[′]_ holds for all adjacent supergroups
_H_ _[′]_ of _H_ . This is because if we assume there exists an _H_ _[′]_ such that equality holds, then by the Michel
Criterion, ( _H_ ) must undergo a non-trivial symmetry increase to ( _GV_ _H_ ) in the high-multiplicity
representation _V_ 0 _[⊕][r]_ . Since these increases always have the bottleneck ( _H_ 0) as their infimum, the
inequalities
dim _V_ 0 _[H]_ dim _V_ 0 _[H]_ [0] dim _V_ 0 _GV H_ (23)
_≤_ _≤_

must in fact collapse to dim _V_ 0 _[H]_ = dim _V_ 0 _[H]_ [0], which leads to a contradiction.


For any high-multiplicity representation _V_ that satisfies dim _V_ _[H]_ _<_ dim _V_ _[H]_ [0], there must exist a
component corresponding to an irreducible representation _V_ 0 in _V_ such that dim _V_ 0 _[H]_ _<_ dim _V_ 0 _[H]_ [0] .
It follows that dim _V_ 0 _[H]_ _<_ dim _V_ 0 _[H]_ _[′]_ for all adjacent supergroups _H_ _[′]_ of _H_, which in turn shows that
dim _V_ _[H]_ _<_ dim _V_ _[H]_ _[′]_ holds for all such _H_ _[′]_ . Therefore, for high-multiplicity representations, _H_ 0
controls the dimension gap of the fixed-point spaces between _H_ and its other adjacent supergroups.


This property allows us to prove a theorem regarding the direct sum of high-multiplicity representations, which in turn establishes a property for the orbit types of such direct sums.
**Theorem C.3.** _Let a subgroup H_ _satisfy the bottleneck condition._ _For two high-multiplicity represen-_
_tations V_ 1 _and V_ 2 _, we have_ ( _H_ ) _∈OG_ ( _V_ 1 _⊕_ _V_ 2) _if and only if_ ( _H_ ) _∈OG_ ( _V_ 1) _or_ ( _H_ ) _∈OG_ ( _V_ 2) _._

_Proof._ We have already shown that for any high-multiplicity representation _V_, if dim _V_ _[H]_ _<_
dim _V_ _[H]_ [0], then dim _V_ _[H]_ _<_ dim _V_ _[H]_ _[′]_ holds for all adjacent supergroups of _H_ . Therefore, assuming
( _H_ ) _G_ ( _V_ 1 _V_ 2), the Michel Criterion gives us
_∈O_ _⊕_

dim _V_ 1 _[H]_ + dim _V_ 2 _[H]_ _<_ dim _V_ 1 _[H]_ _[′]_ + dim _V_ 2 _[H]_ _[′]_ _._ (24)


By taking _H_ _[′]_ = _H_ 0 = _BG_ ( _H_ ), we see that at least one of the following two conditions must be true:

dim _Vi_ _[H]_ _<_ dim _Vi_ _[H]_ [0] _,_ _i_ = 1 _,_ 2 _._ (25)


25


This implies that for all adjacent supergroups of _H_, at least one of the following two conditions must
hold:
dim _Vi_ _[H]_ _<_ dim _Vi_ _[H]_ _[′]_ _,_ _i_ = 1 _,_ 2 _._ (26)


Therefore, ( _H_ ) is an orbit type of either _V_ 1 or _V_ 2.


The above theorem shows that when we construct a high-multiplicity representation for which ( _H_ )
is an orbit type using high-multiplicity components, the only way is to find a high-multiplicity
component that already contains ( _H_ ) as an orbit type.


For _SO_ (3), all closed subgroups satisfy the bottleneck condition. Consequently, for high-multiplicity
representations, we have _G_ ( _V_ 1 _V_ 2) = _G_ ( _V_ 1 _V_ 2). This means that _IG_ ( _V_ 1 _V_ 2 _, H_ ) will be the
minimum of _IG_ ( _V_ 1 _, H_ ) and _O_ _IG_ ( _∪V_ 2 _, H_ ). Furthermore, the set of representations _O_ _⊕_ _⊕ {Vl_ _[⊕]_ = _[r]_ _l_ 0 _[}]_ [ is sufficient]
to generate all closed subgroups as orbit types, because for any closed subgroup, there always exists
a high-multiplicity representation for which it is an orbit type.


For _O_ (3), the bottleneck condition does not necessarily hold. For example, for _H_ = _C∞_, a symmetry
increase can result in either _D∞_ or _C∞v_, which shows that no bottleneck group exists. The fact that
_C∞_ never appears as an orbit type in any representation _V_ _[⊕][r]_
_l_ = _l_ 0 _[±]_ [demonstrates this point precisely.]


This introduces a subtle issue when we apply the guideline on § 4.2: for certain orbit types, a representation exhibiting the target orbit type cannot be constructed simply by including the component
_V_ _[⊕][r]_ [Fortunately,] _[ C][∞]_ [is the sole instance of this phenomenon encountered when]
_l_ = _l_ 0 _[±]_ [associated with it.]

_G_ = _O_ (3). Regarding the construction of a _O_ (3) representation containing _C∞_, following Prop. 4.2,
it suffices to simultaneously select components _Vl_ _[⊕]_ = _[r]_ _l_ 0 _[−]_ [with both odd and even degrees] _[ l]_ [0] _[>]_ [ 0][.]


26


D PROOFS OF DENSITY OF (ALMOST) ISOVARIANT MAPS (§ 5)


D.1 COUNTEREXAMPLE IN § 5.1


**Lemma** **D.1** (Borsuk-Ulam Theorem, Guillemin & Pollack (1974), Chap. 2, Sec. 6) **.** _For_ _any_
_continuous odd function g_ : _S_ _[n]_ _→_ R _[n]_ _, there exists a point x ∈_ _S_ _[n]_ _such that g_ ( _x_ ) = 0 _._
**Lemma D.2** (Weak Borsuk-Ulam Theorem, Nagasaki (2003), Thm. A) **.** _Let M_ _and N_ _be G-spheres_
_in a representation space for a compact group G._ _If there exists an equivariant map from M_ _to N_ _,_
_then the following dimensional inequality holds:_


_φG_ (dim _M_ dim _M_ _[G]_ ) dim _N_ dim _N_ _[G]_ _,_ (27)

_−_ _≤_ _−_


_where φG_ : N N _is a non-decreasing function that diverges to infinity._
_→_


**Counterexample** **D.3.** _For_ _a_ _compact_ _Lie_ _group_ _G,_ _consider_ _a_ _representation_ _Y_ [˜] _with_ _no_ _trivial_
_component,_ _i.e._ _Y_ [˜] _[G]_ = 0 _._ _Let_ _Y_ = _Y_ [˜] _[⊕][r]_ _and_ _X_ = _Y_ _[⊕]_ [(] _[n]_ [0][+1)] _for_ _some_ _r, n_ 0 _>_ dim _G._ _By_
_{_ _}_
_Prop. 4.2, we have_ _G_ ( _X_ ) = _G_ ( _Y_ ) _, yet no isovariant map exists from the unit sphere in X_ _to Y_
_O_ _O_
_for a sufficiently large integer n_ 0 _._

_In particular, for G_ = Z2 _, if_ _Y_ [˜] _is the non-trivial irreducible representation, then for any X_ = _Y_ [˜] _[⊕][r]_ [1]
_and Y_ = _Y_ [˜] _[⊕][r]_ [2] _with r_ 1 _> r_ 2 _, no isovariant map exists from the unit sphere of X_ _to Y ._


_Proof._ For any compact Lie group _G_, consider an equivariant map _f_ from a _G_ -sphere _M_ in a vector
space _X_ to a _G_ -representation _Y_ . We assume that the multiplicities of the trivial representation
components in both _X_ and _Y_ are zero, i.e., _X_ _[G]_ = _{_ 0 _}_ and _Y_ _[G]_ = _{_ 0 _}_ . Therefore, only the origin
in _X_ and _Y_ has the orbit type ( _G_ ), and the sphere _M_ does not contain any point of orbit type ( _G_ ).
Consequently, if _f_ is an isovariant map, it must have no zeros.

From such a zero-free map _f_, we can define a map _f_ [˜] to the _G_ -sphere _N_ in _Y_ :

_f_ ˜ : _M_ _N,_ _f_ ˜( _x_ ) = _f_ ( _x_ ) _/_ _f_ ( _x_ ) 2 _._ (28)
_→_ _∥_ _∥_

Since scaling in a vector space does not change the orbit type, _f_ [˜] is also an isovariant map. Therefore,
by Lem. D.2, we obtain the dimensional relation:


_φG_ (dim _M_ dim _M_ _[G]_ ) = _φG_ (dim _X_ 1) dim _N_ dim _N_ _[G]_ = dim _Y_ 1 _._ (29)

_−_ _−_ _≤_ _−_ _−_


Since _φG_ is a non-decreasing function that diverges to infinity, there must exist an integer _n_ 0 _>_ dim _G_
such that _φG_ ( _n_ 0) _>_ dim _Y_ 1. This implies that

_−_


_φG_ (dim( _Y_ _[⊕]_ [(] _[n]_ [0][+1)] ) 1) _>_ dim _Y_ 1 _._ (30)
_−_ _−_


This shows that no isovariant map can exist from the unit _G_ -sphere _M_ in the space _X_ = _Y_ _[⊕]_ [(] _[n]_ [0][+1)]

to the space _Y_ . To complete the counterexample, let _Y_ = _Y_ [˜] _[⊕][r]_, where _r_ _>_ dim _G_ . By Prop. 4.2, we
have _G_ ( _Y_ ) = _G_ ( _X_ ), yet no isovariant map exists between them.
_O_ _O_

For the special case of _G_ = Z2, let _Y_ [˜] be the non-trivial irreducible representation. Let _X_ = _Y_ [˜] _[⊕][r]_ [1]
and _Y_ = _Y_ [˜] _[⊕][r]_ [2] with _r_ 1 _>_ _r_ 2. A map _f_ from the unit sphere in _X_ to _Y_ is equivariant if and only
if it is an odd function. Since the representations have no trivial components, an isovariant map is
equivalent to an odd function that has no zeros. However, by Lem. D.1, any odd map from a sphere in
a higher-dimensional space to a lower-dimensional space must have a zero. Thus, no such isovariant
map exists.


D.2 PROOF OF THM. 5.1


TFN is an ENN based on tensor products of hidden features. The TFN architecture is composed
of a feature lifting map followed by an equivariant pooling stage. The feature lifting map contains
equivariant convolutional layers that update node features by aggregating information from neighbors.
These convolutions employ filters built from learnable radial functions parameterized by a Multi-Layer
Perceptron (MLP) and real spherical harmonics _Ylm_ .


Mathematically, the parameterized map _f_ TFN TFN is defined as a sum over feature channels.
_∈F_
Each term in the sum is a composition _f_ pool _f_ feat, consisting of a feature lifting map followed by
_◦_


27


equivariant linear pooling. Let _Z_ represent the hidden features associated with each node. The feature
lifting map _f_ feat takes the input point cloud _X_ to node features _Z ⊗_ R _[n]_ . This map is constructed as a
composition of layers:


_f_ feat = _πZ⊗_ R _n_ ( _f_ [(] _[L]_ [)] _,_ id) ( _f_ [(1)] _,_ id) ext _C,_ (31)

_◦_ _◦· · · ◦_ _◦_ _◦_

where _C_ is a centering operation, ext is a constant extension map, and each layer _f_ [(] _[k]_ [)] updates the
node features _v_ [(] _[k][−]_ [1)] to _v_ [(] _[k]_ [)] according to the rule:


_vil_ [(] _[k]_ 3 [)] _m_ 3 [=] _[ θv]_ _il_ [(] _[k]_ 3 _[−]_ _m_ [1)] 3 [+][ �]


_l_ 1 _,m_ 1 _,l_ 2 _,m_ 2 _[C]_ ( [(] _l_ _[l]_ 2 [3] _,m_ _[,m]_ 2 [3] ) [)] _,_ ( _l_ 1 _,m_ 1) _[F]_ [ (] _m_ _[l]_ [2] 2 [)][(] _[x]_ _i_ _[−]_ _[x]_ _j_ [)] _[v]_ _jl_ [(] _[k]_ 1 _[−]_ _m_ [1)] 1 _[.]_ (32)


 _j_ = _i_


Here, _vil_ 3 _m_ 3 corresponds to the feature of type _l_ 3 at node _i_, and _C_ ( [(] _l_ _[l]_ 2 [3] _,m_ _[,m]_ 2 [3] ) [)] _,_ ( _l_ 1 _,m_ 1) [are the Clebsch-]

Gordan coefficients. The filter function is defined as _Fm_ [(] _[l]_ [)][(] _[x]_ [)] [=] _[h]_ _l_ [(] _[∥][x][∥]_ 2 [)] _[Y]_ _lm_ [(] _[x/][∥][x][∥]_ 2 [)][,] [where]
the radial function _hl_ : R _≥_ 0 R is parameterized by a Multi-Layer Perceptron (MLP). For
_→_
_G_ = _O_ (3) _Sn_, the parities of _l_ 1 and _l_ 3 must also be considered. We denote by TFN the resulting
_×_ _F_
family of TFN filters under MLP-parameterized radial functions. In contrast, when _hl_ is chosen as
polynomial parameterization, we denote the corresponding parameterization by _F_ TFN [poly] [.]

We use _L_ ( _X, Y_ ) to denote the vector space of linear maps _X_ _→_ _Y_, and _P_ ( _X, Y_ ) the vector space
of polynomial maps _X_ _Y_ . Given the _G_ -actions on _X_ and _Y_, we write _LG_ ( _X, Y_ ) _L_ ( _X, Y_ )
_→_ _⊆_
and _PG_ ( _X, Y_ ) _P_ ( _X, Y_ ) for the subspaces of _G_ -equivariant maps, i.e., _f_ ( _g_ ( _x_ )) = _g_ ( _f_ ( _x_ )) for all
_⊆_
_g_ _∈_ _G_ and _x ∈_ _X_ . If _Y_ carries the trivial action, equivariance reduces to invariance.
**Lemma** **D.4.** _Let_ _X, Y, Z_ _be_ _representations_ _of_ _a_ _group_ _G._ _Consider_ _a_ _subset_ _of_ _G-equivariant_
_polynomial maps S_ _PG_ ( _X, Z_ ) _that satisfies the spanning condition:_
_⊆_


_P_ ( _X, Y_ ) _⊆_ span( _L_ ( _Z, Y_ ) _◦_ _S_ ) := span( _{A ◦_ _p | A ∈_ _L_ ( _Z, Y_ ) _, p ∈_ _S}_ ) _._ (33)


_Then, it follows that PG_ ( _X, Y_ ) = span( _LG_ ( _Z, Y_ ) _S_ ) _._
_◦_


_In the special case where G_ = _H_ 1 _H_ 2 _, the spanning condition can be relaxed to_
_×_


_PH_ 2( _X, Y_ ) span( _L_ ( _Z, Y_ ) _S_ ) _._ (34)
_⊆_ _◦_


_Remark._ The case for _G_ = _SO_ (3) _× Sn_ is from Thm. 1 of Dym & Maron (2021). In this context, the
action of _Sn_ on the representations is faithful, while we are concerned with features that are invariant
under _Sn_ .


_Proof._ Let _f_ _PG_ ( _X, Y_ ). Since _PG_ ( _X, Y_ ) _P_ ( _X, Y_ ), by the spanning condition, _f_ can be
_∈_ _⊆_
written as a linear combination of compositions:


_f_ =


_N_


_Ai_ _pi,_ (35)
_i_ =1 _◦_


where _pi_ _S_ and _Ai_ _L_ ( _Z, Y_ ). Here, we have absorbed the expansion coefficients into the linear
_∈_ _∈_
maps _Ai_ .


Since _f_ is _G_ -equivariant, it is a fixed point of the group averaging operator. Applying this operator to
both sides of the equation gives:


             _f_ ( _x_ ) = _ρY_ ( _g_ _[−]_ [1] ) _f_ ( _ρX_ ( _g_ )( _x_ )) d _g_ (36)

_G_


 = _ρY_ ( _g_ _[−]_ [1] )

_G_


- _N_

 


_Ai_ ( _ρX_ ( _g_ )( _x_ ))

_i_ =1


d _g_ (37)


_N_

= 

_i_ =1


�� 
_ρY_ ( _g_ _[−]_ [1] ) _AiρZ_ ( _g_ ) d _g_ _pi_ ( _x_ ) _._ (38)
_G_


In the last step, we used the fact that the maps _pi_ _S_ are themselves _G_ -equivariant and moved the
_∈_
integral inward. Let us define the averaged linear maps as


                ( _AG_ ) _i_ := _ρY_ ( _g_ _[−]_ [1] ) _AiρZ_ ( _g_ ) d _g._ (39)

_G_


28


By construction, each ( _AG_ ) _i_ is a _G_ -equivariant linear map, i.e., ( _AG_ ) _i_ _LG_ ( _Z, Y_ ). The expression
_∈_
for _f_ ( _x_ ) can now be written as _f_ ( _x_ ) = [�] _i_ _[N]_ =1 [(] _[A][G]_ [)] _[i][ ◦]_ _[p][i]_ [(] _[x]_ [)][.] [This shows that any map in] _[ P][G]_ [(] _[X, Y]_ [ )]
can be expressed as a linear combination of compositions of maps from _S_ and equivariant linear
maps from _LG_ ( _Z, Y_ ). Therefore,


_PG_ ( _X, Y_ ) span( _LG_ ( _Z, Y_ ) _S_ ) _._ (40)
_⊆_ _◦_

The reverse inclusion, span( _LG_ ( _Z, Y_ ) _S_ ) _PG_ ( _X, Y_ ), is true by definition, since the composition

_◦_ _⊆_
of two equivariant maps is equivariant. Thus, the equality _PG_ ( _X, Y_ ) = span( _LG_ ( _Z, Y_ ) _S_ ) is
_◦_
established.

The proof for the relaxed condition when _f_ _PG_ ( _X, Y_ ) _PH_ 2( _X, Y_ ) and averaging over the group _G_ = _H_ 1 _× H_ 2 follows the exact same logic, by taking _H_ 1. an
_∈_ _⊆_

**Lemma D.5.** _Consider the TFN model with polynomial parametric radial function, a input space_
_X,_ _a_ _lifted_ _representation_ _space_ _Z,_ _and_ _a_ _final_ _output_ _space_ _Y ._ _Suppose_ _the_ _final_ _output_ _space_
_Y_ = _Y_ [˜] R _[n]_ _,_ _where_ _Y_ [˜] _is a representation of_ _SO_ (3) _or_ _O_ (3) _,_ _and the symmetric group_ _Sn_ _acts_
_on Y_ _by permuting the coordinates in the⊗_ R _[n]_ _factor._ _Then the family of polynomial feature maps,_

feat
_F_ [poly] _[⊆]_ _[P][G]_ [(] _[X, Z]_ [)] _[, used in TFN satisfies the relaxed spanning condition:]_

_PSn_ ( _X, Y_ ) _⊆_ span( _L_ ( _Z, Y_ ) _◦F_ feat [poly][)] _[.]_ (41)


_Remark._ Lem. 4 in Dym & Maron (2021) proves the case for _G_ = _SO_ (3) _Sn_ . The proof for
_×_
_O_ (3) is similar to that for _SO_ (3) and is therefore not repeated here. It is worth noting that in the
_O_ (3) case, the spherical harmonics map can still be used to construct the component-wise map from
representation _Vl_ =1 _−_ _[∼]_ = R [3] to symmetric algebra Sym _k_ ( _Vl_ =1 _−_ ), which in turn is used to build the
lifted representation.

**Lemma D.6.** _In Ex. 2.1, the function families F_ TFN [poly] _[contain all equivariant polynomial maps.]_


_Proof._ According to Lem. D.4, to prove that a family of equivariant maps contains all equivariant
polynomials, it is sufficient to show that it satisfies the (potentially relaxed) spanning condition. We
verify this for _F_ TFN [poly] [.]

By Lem. D.5, for _Y_ = _Y_ [˜] _⊗_ R _[n]_, the family _F_ feat [poly] [satisfies the spanning condition]

_PSn_ ( _X, Y_ ) _⊆_ span( _L_ ( _Z, Y_ ) _◦F_ feat [poly][)] _[.]_ (42)

We now only need to verify the following spanning condition:


_PSn_ ( _X,_ _Y_ [˜] ) _⊆_ span( _L_ ( _Z,_ _Y_ [˜] ) _◦F_ feat [poly][)] _[.]_ (43)


For any _g_ _PSn_ ( _X,_ _Y_ [˜] ), we construct an _Sn_ -equivariant polynomial map to _Y_ as
_∈_


_g_ ˜ : _X_ _Y,_ _g_ ˜( _x_ 1 _, . . ., xn_ ) _j_ = _g_ ( _x_ 1 _, . . ., xn_ ) for _j_ = 1 _, . . ., n._ (44)
_→_


This function is clearly an _Sn_ -equivariant polynomial because all _n_ components of its output are
identical. Thus, ˜ _g_ _PSn_ ( _X, Y_ ). By the spanning condition, we can write
_∈_


 _g_ ˜ = _i_ _Ai ◦_ _pi,_ where _Ai_ _∈_ _L_ ( _Z, Y_ ) and _pi_ _∈F_ feat [poly] _[.]_ (45)


Applying the averaging operator to both sides yields


1
_Sn_
_|_ _|_


- (˜ _g_ ( _σ_ ( _x_ ))) _σ_ ( _j_ ) = _g_ ( _x_ ) = 
_σ∈Sn_ _i_


( _ASn_ ) _i_ _pi_ ( _x_ ) _,_ (46)
_i_ _◦_


where
1
( _ASn_ ) _i_ = _Sn_
_|_ _|_


_π_ ˜ _Y_ _ρY_ ( _σ_ _[−]_ [1] ) _Ai_ _ρZ_ ( _σ_ ) _LG_ ( _Z,_ _Y_ [˜] ) _._ (47)
_σ∈Sn_ _◦_ _◦_ _◦_ _∈_


This establishes the spanning condition

_PSn_ ( _X,_ _Y_ [˜] ) _⊆_ span( _LG_ ( _Z,_ _Y_ [˜] ) _◦F_ feat [poly][)] _[.]_ (48)


29


We consider the higher-order approximation theorem of MLPs, and obtain _C_ _[∞]_ -density via higherorder approximations based on polynomial parameterization.

**Lemma D.7** (Higher-Order Approximation Theorem for MLPs, Pinkus (1999), Thm. 4.1) **.** _For any_
_compactσ_ _C_ _[m]_ (R _set_ ) _, letK_ _ϵ >⊂_ R 0 _[n]_ _.,Then there exists an MLP parameterized mapany_ _function_ _f_ _∈_ _C_ _[m]_ (R _[n]_ ) _,_ _and_ _any_ _non-polynomial gθ_ _with activation functionactivation_ _function σ_
_∈_
_such that_
max _x∈K_ _x_ 1 _[. . . ∂]_ _x_ _[k]_ _n_ _[n][f]_ [(] _[x]_ [)] _[ −]_ _[∂]_ _x_ _[k]_ 1 [1] _[. . . ∂]_ _x_ _[k]_ _n_ _[n][g][θ]_ [(] _[x]_ [)] _[|][ < ϵ]_ (49)

_[|][∂][k]_ [1]

_for all non-negative integers k_ 1 _, . . ., kn_ _with k_ 1 + _· · ·_ + _kn_ _< m._
**Theorem 5.1.** _In Ex. 2.1, the function families_ TFN _with smooth activation function are C_ _[∞]_ _-dense_
_F_
_in_ _the_ _space_ _of_ _smooth_ _equivariant_ _maps_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[.]_ _[That]_ _[is,]_ _[for]_ _[any]_ _[integer]_ _[r]_ _[≥]_ [0] _[,]_ _[any]_ _[map]_
_f_ _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[, any compact set][ K]_ _[⊂]_ _[X][, and any][ ϵ >]_ [ 0] _[, there exists a function][ g]_ _[∈F]_ [TFN] _[ such that]_

max _x∈K_ �� _Dkf_ ( _x_ ) _−_ _Dkg_ ( _x_ )�� _< ϵ, k_ _≤_ _r._ (10)


_Proof of Thm. 5.1._ On the domain of analyticity of a function, its Taylor polynomials and their
derivatives converge locally uniformly to the function. Therefore, an analytic function and its
derivatives can be uniformly approximated by a polynomial and its derivatives on any compact set.
This establishes the _C_ _[∞]_ -density of polynomial functions in the set of analytic functions. Furthermore,
by Chap. 2, Thm. 5.1 in Hirsch (1976), the set of analytic functions is _C_ _[∞]_ -dense in the space of
smooth functions. Consequently, polynomials are _C_ _[∞]_ -dense in the space of smooth functions. By
applying the group averaging operator, equivariant polynomials are also _C_ _[∞]_ -dense in the space of
equivariant functions.


We use mathematical induction to prove the _C_ _[∞]_ -density of TFN. Let
_F_
_f_ ˜ [(] _[k]_ [)] = ( _f_ [(] _[n]_ [)] _,_ id) _◦· · · ◦_ ( _f_ [(1)] _,_ id) _◦_ ext _◦_ _C._ (50)

We prove that for the TFN, the MLP-based parameterized map _f_ [˜] MLP [(] _[n]_ [)] [can approximate the polynomial-]
based parameterized map _f_ [˜] poly [(] _[n]_ [)] [.] After establishing the approximation property of the feature
maps, and noting that the equivariant pooling maps coincide in both settings, we obtain the _C_ _[∞]_ approximation of _F_ TFN [poly] [by] _[F]_ [TFN][.] [The] [desired] [approximation] [result] [then] [follows] [directly] [from]
Lem. D.6.


For the base case _n_ = 1, there is no difference in the derivatives with respect to the _X_ component
corresponding to the identity map. Therefore, we only need to discuss the output of _f_ [(1)] . Since

_f_ ˜ _il_ [(1)] 3 _m_ 3 [(] _[x]_ [1] _[, . . ., x][n]_ [) =] _[ θ]_ [ + ∆] _il_ [(1)] 3 _m_ 3 _[,]_ _i_ = 1 _,_ 2 _, . . ., n,_ (51)

we only need to consider the approximation of the derivatives of ∆ [(1)] _il_ 3 _m_ 3 [.] [Expanding][ ∆][(1)] _il_ 3 _m_ 3 [yields]


∆ [(1)] _il_ 3 _m_ 3 [=] 


 - _C_ ( [(] _l_ _[l]_ 2 [3] _,m_ _[,m]_ 2 [3] ) [)] _,_ ( _l_ 1 _,m_ 1) 
_l_ 1 _,m_ 1 _,l_ 2 _,m_ 2 _j_ = _i_


_h_ [(1)] _l_ 2 [(] _[∥][x][i][ −]_ _[x][j][∥]_ [2][)] _[Y][l]_ [2] _[m]_ [2][((] _[x][i][ −]_ _[x][j]_ [)] _[/][∥][x][i][ −]_ _[x][j][∥]_ [2][)] _[.]_ (52)
_j_ = _i_


The difference between _f_ [˜] MLP [(] _[k]_ [)] [and] _[f]_ [˜] poly [ (] _[k]_ [)] [lies in the parameterization of] _[ h][l]_ [2] [.] [When taking derivatives]
of any order of ∆ _il_ 3 _m_ 3, each term in the result is a product of derivatives of _h_ of various orders and a
fixed function, and the number of terms is finite. By Lem. D.7, these terms can be approximated to
any precision on a compact set.

Now, assume the conclusion holds for _n_ = _k −_ 1. We will prove that it also holds for _n_ = _k_ . Let

_vil_ [(] _[k]_ 3 _[−]_ _m_ [1)] 3 [=] _[f]_ [˜] _il_ [ (] _[k]_ 3 _[−]_ _m_ [1)] 3 [(] _[x]_ [1] _[, . . ., x][n]_ [)] _[.]_ (53)

The update rule is given by

_f_ ˜ _il_ [(] _[k]_ 3 [)] _m_ 3 [(] _[x]_ [1] _[, . . ., x][n]_ [) =] _[ θv]_ _il_ [(] _[k]_ 3 _[−]_ _m_ [1)] 3 [+ ∆] _il_ [(] _[k]_ 3 [)] _m_ 3 _[,]_ _i_ = 1 _,_ 2 _, . . ., n._ (54)

By the inductive hypothesis, the first term can be approximated by an MLP model to any precision.
We observe the second term. Similarly, expanding ∆ [(] _il_ _[k]_ 3 [)] _m_ 3 [yields]


∆ [(] _il_ _[k]_ 3 [)] _m_ 3 [=] 


 - _C_ ( [(] _l_ _[l]_ 2 [3] _,m_ _[,m]_ 2 [3] ) [)] _,_ ( _l_ 1 _,m_ 1) 
_l_ 1 _,m_ 1 _,l_ 2 _,m_ 2 _j_ = _i_


- _h_ [(] _l_ 2 _[k]_ [)][(] _[∥][x][i][ −]_ _[x][j][∥]_ [2][)] _[Y][l]_ [2] _[m]_ [2][((] _[x][i][ −]_ _[x][j]_ [)] _[/][∥][x][i][ −]_ _[x][j][∥]_ [2][)] _[v]_ _il_ [(] _[k]_ 1 _[−]_ _m_ [1)] 1 _[.]_

_j_ = _i_


(55)


30


The derivative of this term is a finite sum of products, where each product involves derivatives of
various orders of both _h_ [(] _l_ 2 _[k]_ [)] [and] _[ v]_ _il_ [(] _[k]_ 1 _[−]_ _m_ [1)] 1 [.]

To show that the error of this term can be controlled, we only need to prove that for a vector space
_X_, if for any compact set _K_ _X_, we consider maps _f_ _C_ ( _X_ ) and _g_ _C_ ( _X_ ), and for any _ϵ_ 1 _>_ 0
_⊂_ _∈_ _∈_
there exists _f_ 1 _∈_ _R_ 1, and for any _ϵ_ 2 _>_ 0 there exists _g_ 1 _∈_ _R_ 2 such that

max max (56)
_x∈K_ _x∈K_

_[∥][f]_ [(] _[x]_ [)] _[ −]_ _[f]_ [1][(] _[x]_ [)] _[∥]_ _[< ϵ]_ [1] _[,]_ _[∥][g]_ [(] _[x]_ [)] _[ −]_ _[g]_ [1][(] _[x]_ [)] _[∥]_ _[< ϵ]_ [2] _[,]_

then for any _ϵ >_ 0, there exist _f_ 1 _∈_ _R_ 1 _, g_ 1 _∈_ _R_ 2 such that

max (57)
_x∈K_

_[∥][f]_ [(] _[x]_ [)] _[g]_ [(] _[x]_ [)] _[ −]_ _[f]_ [1][(] _[x]_ [)] _[g]_ [1][(] _[x]_ [)] _[∥]_ _[< ϵ.]_

We bound the term as follows:


_f_ ( _x_ ) _g_ ( _x_ ) _f_ 1( _x_ ) _g_ 1( _x_ ) _f_ ( _x_ ) _g_ ( _x_ ) _g_ 1( _x_ ) + _g_ 1( _x_ ) _f_ ( _x_ ) _f_ 1( _x_ )
_∥_ _−_ _∥≤∥_ _∥∥_ _−_ _∥_ _∥_ _∥∥_ _−_ _∥_
_≤∥f_ ( _x_ ) _∥ϵ_ 2 + _∥g_ 1( _x_ ) _∥ϵ_ 1
_≤_ ( _∥f_ 1( _x_ ) _∥_ + _ϵ_ 1) _ϵ_ 2 + _∥g_ 1( _x_ ) _∥ϵ_ 1
= _∥f_ 1( _x_ ) _∥ϵ_ 2 + _∥g_ 1( _x_ ) _∥ϵ_ 1 + _ϵ_ 1 _ϵ_ 2 _._ (58)

Since a continuous function on a compact set attains its maximum, let _L_ 1 be the maximum of _∥f_ 1( _x_ ) _∥_
on _K_ and _L_ 2 be the maximum of _∥g_ 1( _x_ ) _∥_ on _K_ . We get

max (59)
_x∈K_

_[∥][f]_ [(] _[x]_ [)] _[g]_ [(] _[x]_ [)] _[ −]_ _[f]_ [1][(] _[x]_ [)] _[g]_ [1][(] _[x]_ [)] _[∥≤]_ _[L]_ [2] _[ϵ]_ [1][ +] _[ L]_ [1] _[ϵ]_ [2][ +] _[ ϵ]_ [1] _[ϵ]_ [2] _[.]_

This shows that the error can be made arbitrarily small, completing the proof.


D.3 SOME RESULTS ON TOPOLOGY


Here we adopt the definition of _NG_ ( _H_ ) and _NG_ ( _H, H_ _[′]_ ) from § C.1, and consider the twisted product
between _G_ -sets as defined in Chap. 2, Sec. 2 of Bredon (1972). We denote _XH_ := _x_ _X_ _Gx_ =
_{_ _∈_ _|_
_H}_, from which we obtain the following lemma on topology.

**Lemma D.8.** _When the action of a group G on a G-manifold M_ _is faithful, the following decomposi-_
_tions hold:_
_M_ ( _H_ _′_ ) = _MH_ _′_ _NG_ ( _H_ _′_ ) _/H_ _′_ _G/H_ _[′]_ (60)
_×_

_and_
_M_ ( _[H]_ _H_ _[′]_ ) [=] _[ M][H]_ _[′]_ _G_ [(] _[H]_ _[′]_ [)] _[/H]_ _[′]_ _[N][G]_ [(] _[H, H]_ _[′]_ [)] _[/H]_ _[′][.]_ (61)

_[×][N]_


_Proof._ The first identity is derived from Thm. 1.31 in Meinrenken (2003), which states the existence
of a homeomorphism


_σ_ : _MH_ _′_ _NG_ ( _H_ _′_ ) _/H_ _′_ _G/H_ _[′]_ _M_ ( _H_ _′_ ) _,_ given by _σ_ ([ _x, gH_ _[′]_ ]) = _g_ ( _x_ ) _._ (62)
_×_ _→_

We now prove that _σ_ ( _MH_ _′_ _×NG_ ( _H_ _′_ ) _/H_ _′_ ( _NG_ ( _H, H_ _[′]_ ) _/H_ _[′]_ )) = _M_ ( _[H]_ _H_ _[′]_ ) [.] [The] [second] [identity] [then]
follows by restricting this homeomorphism.


( ) Take an element [ _x, gH_ _[′]_ ] where _x_ _MH_ _′_ and _g_ _NG_ ( _H, H_ _[′]_ ). Its image under _σ_ is _g_ ( _x_ ). The
_⊆_ _∈_ _∈_
isotropy subgroup is _Gg_ ( _x_ ) = _gGxg_ _[−]_ [1] = _gH_ _[′]_ _g_ _[−]_ [1] . Since _g_ _NG_ ( _H, H_ _[′]_ ), we have _H_ _gH_ _[′]_ _g_ _[−]_ [1] .
_∈_ _⊆_
This implies _g_ ( _x_ ) _∈_ _M_ _[H]_, and since its orbit type is ( _H_ _[′]_ ), we have _g_ ( _x_ ) _∈_ _M_ ( _[H]_ _H_ _[′]_ ) [.]

( _⊇_ ) Take any _y_ _∈_ _M_ ( _[H]_ _H_ _[′]_ ) [.] [This means] _[ H]_ _[⊆]_ _[G][y]_ [and][ (] _[G][y]_ [) = (] _[H]_ _[′]_ [)][.] [The latter implies there exists a]
_g_ 0 _∈_ _G_ such that _Gy_ = _g_ 0 _H_ _[′]_ _g_ 0 _[−]_ [1][.] [The condition] _[ H]_ _[⊆]_ _[g]_ [0] _[H]_ _[′][g]_ 0 _[−]_ [1] implies that _g_ 0 _∈_ _NG_ ( _H, H_ _[′]_ ). Let
_x_ = _g_ 0 _[−]_ [1][(] _[y]_ [)][.] [Then] _[ G][x]_ [=] _[ H]_ _[′]_ [, so] _[ x][ ∈]_ _[M][H]_ _[′]_ [.] [We can then write] _[ y]_ [ as]

_y_ = _g_ 0( _g_ 0 _[−]_ [1][(] _[y]_ [)) =] _[ g]_ [0][(] _[x]_ [) =] _[ σ]_ [([] _[x, g]_ [0] _[H]_ _[′]_ [])] _[.]_ (63)

Since _x_ _MH_ _′_ and _g_ 0 _NG_ ( _H, H_ _[′]_ ), this shows that _y_ is in the image of the restricted domain. The
_∈_ _∈_
inclusion is thus proven.


31


From the following proposition onward, we need to invoke stratification theory. For the definitions of
Whitney conditions and stratifications, we refer to Chap. 3, Sec. 9 of Field (2007). Going forward,
unless otherwise specified, all function spaces in this paper are equipped with the (weak) _C_ _[∞]_
topology. For the topology on function spaces, we refer to Chap. 2, Sec. 1 of Hirsch (1976).

**Proposition D.9.** _Consider smooth manifolds M_ _and N_ _._ _Let_ ( _Sα, Sβ_ ) _be a pair of submanifolds_
_in_ _N_ _that_ _satisfies_ _Whitney’s_ _condition_ _(a)._ _If_ _a_ _map_ _f_ : _M_ _N_ _is_ _transverse_ _to_ _Sα_ _at_ _a_ _point_
_x_ _f_ _[−]_ [1] ( _Sα_ ) _, then there exists a neighborhood U_ _of f_ _in C_ _[∞]_ ( _M, N→_ ) _and a neighborhood V_ _of x in_
_∈_
_M_ _, such that for any map g_ _∈_ _U_ _and any point y_ _∈_ _V, g is transverse to both Sα and Sβ_ _at y._


_Proof._ The proof is adapted from Prop. 1.3 of Trotman (1976). We proceed by contradiction. Assume
the conclusion is false. Then for any neighborhood _V_ of _x_ and any neighborhood _U_ of _f_, there exists
a map _g_ _U_ for which the condition _g_ ⋔ _Sα, Sβ_ on _V_ does not hold.
_∈_

For a neighborhood _V_ 1 of _x_ with radius less than _ϵ_ 1, we can construct a sequence of maps _gn_ [(1)]
converging to _f_ such that transversality to _Sα_ or _Sβ_ fails on _V_ 1. Let the sequence of non-transverse _{_ _[}]_
points be _{yn_ [(1)] _[}]_ [.] [We can similarly construct,] [for neighborhoods] _[V]_ _t_ [of] _[x]_ [ with radius less than] _[ ϵ]_ _t_ [,]
sequences of maps _gn_ [(] _[t]_ [)] [and corresponding sequences of non-transverse points]
_{_ _[}]_ [ converging to] _[ f]_
_yn_ [(] _[t]_ [)]
_{_ _[}]_ [.]

Now, consider the diagonal sequence of maps _gn_ [(] _[n]_ [)] [and] [the] [corresponding] [sequence] [of] [non-]
_{_ _[}]_
transverse points _yn_ [(] _[n]_ [)] [We have] _[ g]_ _n_ [(] _[n]_ [)] _f_ and _yn_ [(] _[n]_ [)] _x_ . We can partition the sequence _yn_ [(] _[n]_ [)]
into a subsequence lying in _{_ _[}]_ [.] _Sα_ and another lying in _→_ _Sβ_ . _→_ At least one of these two subsequences must _{_ _[}]_
be infinite. It is therefore sufficient to negate the following proposition: There exists a sequence of
maps _{gn} →_ _f_ and a sequence of points _{yn} →_ _x_ with either _{yn} ⊂_ _Sα_ or _{yn} ⊂_ _Sβ_, such that
for each _n_, _gn_ is not transverse to _Sα_ or _Sβ_ at _yn_ .


This is impossible. Note that for a sufficiently small _ϵ_ 1, the closure of the neighborhood _V_ 1 is
compact by local compactness of the manifold, and we can consider the control of the neighborhood
_U_ over functions on this compact closure in the _C_ _[∞]_ topology of functioin space. Also note that
d( _gn_ ) _yn_ ( _TynM_ ) converges to d _fx_ ( _TxM_ ). For the case of _Sα_, by the smoothness of the manifold,
_Tgn_ ( _yn_ ) _Sα_ converges to _Tf_ ( _x_ ) _Sα_ . For the case of _Sβ_, by Whitney’s condition (a), the limit of
_Tgn_ ( _yn_ ) _Sβ_ contains _Tf_ ( _x_ ) _Sα_ . However, _f_ is transverse to _Sα_ at _x_, meaning


d _fx_ ( _TxM_ ) + _Tf_ ( _x_ ) _Sα_ = _Tf_ ( _x_ ) _N._ (64)


Therefore, due to convergence, the transversality condition for _Sα_ or _Sβ_ must hold for _gn_ at _yn_ for
sufficiently large _n_, which is a contradiction.


**Corollary D.10.** _Under the conditions of the previous proposition, assume that the stratification_
_S_ = _Sα_ _satisfies_ _Whitney’s_ _condition_ _(a)._ _If_ _a_ _map_ _f_ _is_ _transverse_ _to_ _a_ _stratum_ _Sα_ _at_ _a_ _point_
_x_ _f{_ _[−]_ [1] ( _S}α_ ) _, then there exists a neighborhood Vx_ _of x and a neighborhood Ux_ _of f_ _in C_ _[∞]_ ( _M, N_ )
_∈_
_such_ _that_ _for_ _any_ _g_ _Ux_ _and_ _any_ _y_ _Vx,_ _g_ _is_ _transverse_ _to_ _the_ _entire_ _stratification_ _S_ _in_ _the_
_∈_ _∈_
_neighborhood Vx._


_Proof._ Let _x_ be a point where _f_ is transverse to _Sα_ . We consider all strata _Sβ_ that satisfies _Sβ_ _Sα_ =
∅. Since a stratification is locally finite, there are only a finite number of such adjacent strata. _∩_ For
each such adjacent stratum _Sβ_, the pair ( _Sα, Sβ_ ) satisfies Whitney’s condition (a). By Prop. D.9,
there exist neighborhoods _Ux_ [(] _[α,β]_ [)] of _f_ and _Vx_ [(] _[α,β]_ [)] of _x_ where transversality to both _Sα_ and _Sβ_ holds.
We can then construct the desired neighborhoods by taking the finite intersection of these individual
neighborhoods labelled by _β_ :


_Ux_ := 


 - _Ux_ [(] _[α,β]_ [)] and _Vx_ := 

_Sβ_ _∩Sα_ =∅ _Sβ_ _∩Sα_


 - _Vx_ [(] _[α,β]_ [)] _._ (65)


_Sβ_ _∩Sα_ =∅


Since the intersection is finite, _Ux_ and _Vx_ are still open neighborhoods of _f_ and _x_, respectively. For
any _g_ _Ux_ and _y_ _Vx_, _g_ is transverse to _Sα_ and all its adjacent strata _Sβ_ at _y_ . Therefore, it is
_∈_ _∈_
transverse to the entire stratification _S_ within the local neighborhood _Vx_ .


32


**Lemma D.11.** _For a G-manifold M_ _and a G-representation Y, the collection of submanifolds_


( _M_ ( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) ( _H_ _′_ ) _≥_ ( _H_ ) (66)
_{_ _×_ _}_

_forms a Whitney stratification of_ ( _M_ _Y_ )( _H_ ) _._
_×_


_Proof._ By Prop. 3.7.1 of Field (2007), we obtain that these point sets in the collection are smooth
_G_ -submanifolds. By Lem. D.8, we have


( _M_ ( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) = ( _M_ ( _H_ ) _Y_ ( _H_ _′_ )) _H_ _NG_ ( _H_ ) _/H_ _G/H_ (67)
_×_ _×_ _×_


= ( _M_ ( _H_ ) _× Y_ ( _H_ _′_ )) _[H]_ ( _H_ ) _[×][N]_ _G_ [(] _[H]_ [)] _[/H]_ _[G/H]_ (68)

= ( _M_ ( _[H]_ _H_ ) _[×][ Y]_ ( _[H]_ _H_ _[′]_ ) [)] _[ ×][N]_ _G_ [(] _[H]_ [)] _[/H]_ _[G/H.]_ (69)


By local trivialization, similar to the proof of Prop. 3.9.2 of Field (2007), it is sufficient to prove that
the pair ( _Y_ ( _[H]_ _H_ _[′]_ ) _[, Y]_ ( _[H]_ _K_ ) [)][ satisfies Whitney’s condition (b).] [By Prop. 3.9.2 of][ Field][ (][2007][), the orbit type]
stratification for a representation space is a normal orbit type stratification, and therefore the pair of
strata ( _Y_ ( _H_ _′_ ) _, Y_ ( _K_ )) for ( _H_ _[′]_ ) _>_ ( _K_ ) satisfies Whitney’s conditions. We now show that ( _Y_ ( _[H]_ _H_ _[′]_ ) _[, Y]_ ( _[H]_ _K_ ) [)]
also satisfies Whitney’s conditions.


The condition holds for the pair ( _Y_ ( _H_ _′_ ) _, Y_ ( _K_ )). Therefore, for a point _x_ _Y_ ( _K_ ) _Y_ ( _H_ _′_ ), and for
_∈_ _∩_
sequences _yn_ _Y_ ( _K_ ) _x_ and _xn_ _Y_ ( _H_ _′_ ) _x_ such that the secant lines ~~_x_~~ _n_ ~~_y_~~ _n_ converge to
_{_ _}_ _⊂_ _→_ _{_ _}_ _⊂_ _→_
a line _l_ _TxY_ and the tangent spaces _TynY_ ( _K_ ) converge to a linear subspace _E_ _TxY_, we have
_⊂_ _⊂_
_l_ _⊆_ _E_ . By restricting the point _x_ to be in _Y_ ( _[H]_ _K_ ) _[∩]_ _[Y]_ ( _[H]_ _H_ _[′]_ ) [, and the sequences] _[ {][x][n][}]_ [ and] _[ {][y][n][}]_ [ to be in]
_Y_ ( _[H]_ _H_ _[′]_ ) [and] _[ Y]_ ( _[H]_ _K_ ) [respectively, the conclusion] _[ l][ ⊆]_ _[E]_ [still holds.] [Therefore, the pair][ (] _[Y]_ ( _[H]_ _H_ _[′]_ ) _[, Y]_ ( _[H]_ _K_ ) [)][ also]
satisfies Whitney’s condition (b).


**Lemma D.12.** _Let M_ _and N_ _be topological spaces, with M_ _being compact._ _Consider a continuous_
_mappreimage f_ : _M f_ _[−]_ _→_ [1] ( _CN_ ) _.._ _Let C_ _be a closed set in N_ _, and suppose there exists a neighborhood V_ _of the_


_Then there exists a neighborhood U_ _of f_ _in C_ ( _M, N_ ) _with the C_ [0] _topology (compact-open topology)_
_such that for any map g_ _∈_ _U_ _, we have g_ ( _M_ _\ V_ ) _∩_ _C_ = ∅ _._

_Proof._ Note that _M_ _\_ _V_ is a compact set, and _N_ _\_ _C_ is an open set. By the definition of the
compact-open topology, e.g. Definition 43.1 of Willard (1970), the set

_U_ := _{h ∈_ _C_ ( _M, N_ ) _| h_ ( _M_ _\ V_ ) _⊆_ _N_ _\ C}_ (70)


is an open set.

Since _V_ is a neighborhood of _f_ _[−]_ [1] ( _C_ ), we have _f_ _[−]_ [1] ( _C_ ) _⊆_ _V_, which implies _f_ ( _M_ _\ V_ ) _⊆_ _N_ _\ C_ .
Therefore, the map _f_ itself is an element of _U_, meaning _U_ is an open neighborhood of _f_ . For
any _g_ ( _M_ map _\ V_ ) _g ∩∈CU_ =, ∅its. definitionThus, this setdirectly _U_ is the desired neighborhood.implies that _g_ ( _M_ _\ V_ ) _⊆_ _N_ _\ C_, which is equivalent to


D.4 GENERIC EQUIVARIANT MAPPINGS


**Proposition D.13** (Equivariant Smooth Extension) **.** _Let M_ _be a smooth G-manifold and let S_ _⊂_ _M_ _be_
_a smooth and compact G-submanifold._ _For any smooth equivariant map f_ _∈_ _CG_ _[∞]_ [(] _[S, Y]_ [ )] _[ defined on][ S]_
_that maps into a representation space Y, there exists a smooth equivariant extension_ _f_ [˜] _∈_ _CG_ _[∞]_ [(] _[M, Y]_ [ )]
_such that its restriction to S_ _is f_ _, i.e.,_ _f_ [˜] _S_ = _f_ _._
_|_


_Proof._ The proof strategy is the same as that for the Tietze-Gleason Theorem (see Chap. 1, Thm. 2.3
of Bredon (1972)).


First, we can view the map _f_ as a collection of dim _Y_ smooth real-valued functions defined on _S_ . By
the standard smooth extension theorem, e.g., Lem. 5.34 in Lee (2012), there exists a smooth (but not
necessarily equivariant) extension _φ_ _C_ _[∞]_ ( _M, Y_ ) such that _φ_ _S_ = _f_ .
_∈_ _|_


33


We then construct an equivariant map from _φ_ using the group averaging operator. Define the map
_ψ_ : _M_ _Y_ by
_→_  


   _ψ_ ( _x_ ) =


_ρY_ ( _g_ _[−]_ [1] ) _φ_ ( _ρX_ ( _g_ )( _x_ )) d _g._ (71)
_G_


Due to the linearity of the integral and the properties of the Haar measure, this map is equivariant.
For any _h ∈_ _G_ :

              _ψ_ ( _h_ ( _x_ )) = _ρY_ ( _g_ _[−]_ [1] ) _φ_ ( _g_ ( _h_ ( _x_ ))) d _g_ (72)

_G_

              = _ρY_ (( _g_ _[′]_ _h_ _[−]_ [1] ) _[−]_ [1] ) _φ_ ( _g_ _[′]_ ( _x_ )) d _g_ _[′]_ (73)

_G_

                 = _ρY_ ( _h_ ) _ρY_ (( _g_ _[′]_ ) _[−]_ [1] ) _φ_ ( _g_ _[′]_ ( _x_ )) d _g_ _[′]_ (74)

_G_

= _ρY_ ( _h_ ) _ψ_ ( _x_ ) _._ (75)


Next, we verify that the restriction of _ψ_ to the submanifold _S_ is equal to the original map _f_ . For any
_s ∈_ _S_ :


   _ψ_ ( _s_ ) =


_ρY_ ( _g_ _[−]_ [1] ) _φ_ ( _g_ ( _s_ )) d _g_ (76)
_G_


 = _ρY_ ( _g_ _[−]_ [1] ) _f_ ( _g_ ( _s_ )) d _g_ (77)

_G_


 = _ρY_ ( _g_ _[−]_ [1] ) _ρY_ ( _g_ ) _f_ ( _s_ ) d _g_ (78)

_G_


 =


       _f_ ( _s_ ) d _g_ = _f_ ( _s_ )
_G_


1 d _g_ = _f_ ( _s_ ) _._ (79)
_G_


Finally, the smoothness of _ψ_ follows from the smoothness of _φ_ and the properties of integration over
a compact group. This can be verified by a local coordinate analysis. Thus, _ψ_ is the desired smooth
equivariant extension _f_ [˜] .


**Corollary D.14.** _Consider maps from a G-manifold X to a representation space Y, where X contains_
_a compact, smooth G-submanifold M_ _._ _Let S_ _be a subset of CG_ _[∞]_ [(] _[M, Y]_ [ )] _[ that contains an open dense]_
_set.contains an open dense subset ofThen_ _the_ _set_ _of_ _maps_ _f_ _∈_ _C CG_ _[∞]_ _G_ _[∞]_ [(] _[X, Y]_ [(] _[X, Y]_ [ )][ )] _[whose][.]_ _[restriction]_ _[to]_ _[M]_ _[lies]_ _[in]_ _[S]_ _[(i.e.,]_ _[f]_ _[|][M]_ _[∈]_ _[S][)]_ _[also]_


_Proof._ We recall that a set contains an open dense subset if and only if for any non-empty open set, its
intersection with the set contains a non-empty open subset. Let _A_ := _f_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[ |][ f]_ _[|][M]_
Our goal is to show that for any _f_ _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [ )][ and any of its open neighborhoods] _{_ _∈_ _[ U]_ [, there exists a] _[∈]_ _[S][}]_ [.]
non-empty open set _V_ _⊆_ _U_ _∩_ _A_ .

Consider the restriction map Res _M_ : _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[ →]_ _[C]_ _G_ _[∞]_ [(] _[M, Y]_ [ )][, by Prop.][ D.13][, this map is surjective.]
Furthermore, this map is continuous and open.


Let _U_ be an open neighborhood of an arbitrary map _f_ . Since Res _M_ is an open map, its image,
Res _M_ ( _U_ ), is an open neighborhood of _f_ _M_ . By our hypothesis on _S_, its intersection with the open
set Res _M_ ( _U_ ) must contain a non-empty open subset. _|_ Let us call this non-empty open set _V_ _[′]_ . We
have
_V_ _[′]_ Res _M_ ( _U_ ) _S._ (80)
_⊆_ _∩_

Now, let’s consider the preimage _V_ := Res _[−]_ _M_ [1][(] _[V]_ _[′]_ [)][.] [Since][ Res] _[M]_ [is continuous and] _[ V]_ _[′]_ [is open,] _[ V]_ [is]
an open set. Since Res _M_ is surjective, the preimage _V_ must also be non-empty. Therefore _U_ _V_ is
_∩_
a non-empty open subset of _U_ _∩_ _A_, which proves the corollary.


For _C_ _[r]_ manifolds _X_ and _Y_, _C_ _[r]_ ( _X_ ) is the set of _C_ _[r]_ real-valued functions on _X_, and _C_ _[r]_ ( _X, Y_ ) is
the set of _C_ _[r]_ maps from _X_ to _Y_ . For manifolds with a _C_ _[r]_ _G_ -action, _CG_ _[r]_ [(] _[X, Y]_ [ )][ is the set of] _[ C]_ _[r]_
equivariant maps from _X_ to _Y_ . We assume these function spaces are endowed with the _C_ _[r]_ topology;
in our proofs, we always consider the _A_ _Y_, _f_ ⋔ _A_ denotes that _f_ is transverse to _C_ _[∞]_ topology. _A_ . ForFor a map _G_ -manifolds, _f_ _∈_ _Cf_ [1] (⋔ _X, YG_ _A_ ) denotes that and a submanifold _f_ is in
_⊆_


34


equivariant general position with respect to the _G_ -submanifold _A_ defined in Bierstone (1977). For
_f_ _∈_ _C_ _[r]_ ( _X, Y_ ), the jet map _j_ _[r]_ _f_ : _X_ _→_ _J_ _[r]_ ( _X, Y_ ) maps a point _x_ to the equivalence class of the first
_r_ derivatives of _f_ at _x_ . We obtain the following results.


**Proposition D.15** (Bierstone (1977), Thm. 1.3) **.** _Let M_ _and N_ _be smooth G-manifolds._ _If P_ _is a_
_satisfyingclosed G-submanifold of f_ ⋔ _G_ _P_ _on K_ _Nforms an open subset ofand K_ _is a compact subset of CG_ _[∞]_ [(] _[M, N]_ _M_ [)] _, then the set of maps_ _[ (in the Whitney][ C]_ _[∞]_ _f_ _[topology).]_ _∈_ _CG_ _[∞]_ [(] _[M, N]_ [)]

**Proposition** **D.16** (Bierstone (1977), Thm. 1.4) **.** _Let_ _M, N_ _be_ _smooth_ _G-manifolds_ _and_ _P_ _be_ _a_
_G-submanifold of N_ _._ _The set of maps f_ _∈_ _CG_ _[∞]_ [(] _[M, N]_ [)] _[ satisfying][ f]_ [⋔] _[G]_ _[P]_ _[forms a residual subset of]_
_CG_ _[∞]_ [(] _[M, N]_ [)] _[, i.e., a countable intersection of open dense sets (in the Whitney][ C]_ _[∞]_ _[topology).]_


_Remark_ . For compact _M_, note that the Whitney _C_ _[∞]_ topology coincides with the _C_ _[∞]_ topology.


**Proposition D.17** (Stratumwise Transversality Theorem) **.** _For any orbit type_ ( _H_ ) _G_ ( _M_ ) _, a map_
_f_ _∈_ _CG_ _[∞]_ [(] _[M, N]_ [)] _[ with][ f]_ [⋔] _[G]_ _[P]_ _[satisfies the stratumwise transversality property:]_ _∈O_

_f_ _MH_ : _MH_ _N_ _[H]_ _, f_ _MH_ ⋔ _P_ _[H]_ _._ (81)
_|_ _�→_ _|_

_Alternatively, this can be expressed in the language of jets as_


_j_ [0] _f_ _M_ ( _H_ ) : _M_ ( _H_ ) ( _M_ ( _H_ ) _N_ )( _H_ ) _, j_ [0] _f_ _M_ ( _H_ ) ⋔ ( _M_ ( _H_ ) _P_ )( _H_ ) (82)
_|_ _�→_ _×_ _|_ _×_


_Remark._ The fact that _f_ ⋔ _G_ _P_ implies stratumwise transversality on the fixed-point sets is a fundamental conclusion derived from Prop. 6.4 of Bierstone (1977). Our proof below is a straightforward
corollary of this.


_Proof._ From the discussion in Prop. 6.4 of Bierstone (1977), we have


_j_ [0] ( _f_ _M_ ( _H_ ) ) : _M_ ( _H_ ) ( _M_ _N_ )( _H_ ) _,_ _j_ [0] ( _f_ _M_ ( _H_ ) ) ⋔ ( _M_ _P_ )( _H_ ) _._ (83)
_|_ _�→_ _×_ _|_ _×_

Note that the image of the map _j_ [0] ( _f_ _M_ ( _H_ ) ) is contained within _M_ ( _H_ ) _N_ . The transversality
_|_ _×_
condition is an equality of tangent spaces:


d( _j_ [0] ( _f_ _M_ ( _H_ ) )) _x_ ( _TxM_ ( _H_ )) + _Ty_ (( _M_ _P_ )( _H_ )) = _Ty_ (( _M_ _N_ )( _H_ )) _._ (84)
_|_ _×_ _×_

Intersecting both sides of this equation with _Ty_ (( _M_ ( _H_ ) _N_ )( _H_ )) yields
_×_

d( _j_ [0] ( _f_ _M_ ( _H_ ) )) _x_ ( _TxM_ ( _H_ )) + _Ty_ (( _M_ ( _H_ ) _P_ )( _H_ )) = _Ty_ (( _M_ ( _H_ ) _N_ )( _H_ )) _._ (85)
_|_ _×_ _×_

Therefore, we have


_j_ [0] ( _f_ _M_ ( _H_ ) ) : _M_ ( _H_ ) ( _M_ ( _H_ ) _N_ )( _H_ ) _,_ _j_ [0] ( _f_ _M_ ( _H_ ) ) ⋔ ( _M_ ( _H_ ) _P_ )( _H_ ) _._ (86)
_|_ _→_ _×_ _|_ _×_


**Proposition D.18** (Bierstone (1977), Sec. 7) **.** _Consider smooth G-manifolds M_ _and N_ _._ _Let P_ _be a_
_smooth G-submanifold of N_ _._ _If an equivariant map f_ _satisfies f_ ⋔ _G_ _P_ _at a point x_ _f_ _[−]_ [1] ( _P_ ) _, then_
_∈_
_there exists a neighborhood U_ _of f_ _in CG_ _[∞]_ [(] _[M, N]_ [)] _[ and a][ G][-invariant neighborhood][ V]_ _[of the orbit]_
_G_ ( _x_ ) _such that for any map g_ _U_ _and any point y_ _V, it holds that g_ ⋔ _G_ _P_ _at y._
_∈_ _∈_


_Remark._ The proposition above can also be derived from the properties of stratifications given in
Prop. D.9 by the definition of equivariant general position in Bierstone (1977).


**Proposition D.19.** _Let G be a compact Lie group,_ _M_ _be a compact,_ _smooth G-manifold,_ _and Y_
_be a G-representation space._ _The set of smooth equivariant maps f_ _∈_ _CG_ _[∞]_ [(] _[M, Y]_ [ )] _[ that satisfy the]_
_transversality condition_
_j_ [0] ( _f_ _M_ ( _H_ ) ) ⋔ ( _M_ ( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) (87)
_|_ _×_

_for all pairs of orbit types_ ( _H_ ) _and_ ( _H_ _[′]_ ) _such that_ ( _H_ ) _is present in M_ _and_ ( _H_ _[′]_ ) _≥_ ( _H_ ) _, contains_
_an open dense subset of CG_ _[∞]_ [(] _[M, Y]_ [ )] _[.]_


_Remark_ . The proof of density is straightforward. However, the openness of this property does not
generally hold. A counterexample can be constructed following Ex. 2.1 of Bierstone (1977).


35


_Proof._ Since _M_ is compact, by Proposition 3.7.2 of Field (2007), the orbit types are finite. For any
given symmetry type ( _H_ ), by Prop. D.16 the set of maps satisfying _f_ ⋔ _G_ _Y_ ( _H_ ) is an intersection of a
finite number of residual sets, and is therefore itself a residual set. By Prop. D.17 the set of maps
satisfying the transversality condition stated in the theorem is dense.


Instead of directly proving openness property, we prove a related proposition: for a map _f_ that
satisfies _f_ ⋔ _G_ _Y_ ( _H_ ) for all orbit types ( _H_ ), there exists a neighborhood _U_ of _f_ such that for any
_g_ _∈_ _U_, _j_ [0] ( _g_ _M_ ( _H_ ) ) ⋔ ( _M_ ( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) for all ( _H_ _[′]_ ) ( _H_ ) _._ (88)
_|_ _×_ _≥_
The proof proceeds by induction on the dimension of the strata in the orbit type stratification of _Y_,
ordered from lowest to highest (or equivalently, from highest to lowest symmetry). We discuss the
fixed orbit type ( _H_ ) in _M_ . Then we can construct neighborhoods that hold for all ( _H_ ) by taking the
intersection of the neighborhoods corresponding to each orbit type ( _H_ ).


We start with the lowest-dimensional strata. Let _Y_ ( _H_ 1) be a stratum of minimal dimension. Such a
stratum is unique, which corresponds to a maximal symmetry type ( _G_ ) and is a closed _G_ -submanifold
_Y_ _[G]_ . By Prop. D.15, the set of maps in general position to _Y_ ( _H_ 1) is open in _CG_ _[∞]_ [(] _[M, Y]_ [ )][.] [Thus, there]
exists a neighborhood _U_ 1 of _f_ such that for any _g_ _U_ 1, _g_ ⋔ _G_ _Y_ ( _H_ 1). By Prop. D.17, this implies
_∈_
that for all ( _H_ ),
_j_ [0] ( _g_ _M_ ( _H_ ) ) ⋔ ( _M_ ( _H_ ) _Y_ ( _H_ 1))( _H_ ) _._ (89)
_|_ _×_

Assume the proposition holds for all strata of dimension up to _k_ 1. Let _Uk−_ 1 be the neighborhood
_−_
of _f_ found from the inductive hypothesis. For any map _g_ _Uk−_ 1 and for any stratum _Y_ ( _Hi_ ) with
_∈_
dim _Y_ ( _Hi_ ) _k_ 1, we have
_≤_ _−_


_j_ [0] ( _g_ _M_ ( _H_ ) ) ⋔ ( _M_ ( _H_ ) _Y_ ( _Hi_ ))( _H_ ) for all ( _Hi_ ) ( _H_ ) _._ (90)
_|_ _×_ _≥_


Note that by Lem. D.11 the collection _S_ ( _H_ ) = ( _M_ ( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) ( _H_ _′_ ) _≥_ ( _H_ ) is a Whitney stratification and satisfies the frontier condition. Therefore, _{_ _×_ by Cor. D.10 _}_, for any _x_ _M_ ( _H_ ) with
_∈_
_f_ ( _x_ ) _Y_ ( _Hi_ ), there exists a neighborhood _Vx_ of _x_ and a neighborhood _Ux_ of _j_ [0] ( _f_ _M_ ( _H_ ) ) such that
_∈_ _|_
for any map in _Ux_, it is transverse to the stratification _S_ ( _H_ ).

Next, we need to obtain an open set in _CG_ _[∞]_ [(] _[M, Y]_ [ )][.] [We use the following sequence of maps between]
function spaces:


_CG_ _[∞]_ [(] _[M, Y]_ [ )] Res _�→M_ ( _H_ ) _CG_ _[∞]_ [(] _[M]_ ( _H_ ) _[, Y]_ [ )] (91)


_j_ [0]
_�→_ _CG_ _[∞]_ [(] _[M]_ ( _H_ ) _[,]_ [ (] _[M]_ ( _H_ ) _[×][ Y]_ [ )] ( _H_ ) [)] (92)

_�_ _C_ _[∞]_ ( _M_ ( _H_ ) _,_ ( _M_ ( _H_ ) _Y_ )( _H_ )) _._ (93)
_→_ _×_


The maps between these function spaces are continuous in _C_ _[∞]_ topology. For example, the 0-jet map
is a restriction of the continuous 0-jet map on the general function space, which can be obtained
from the continuity of jet map in the Whitney _C_ _[∞]_ topology established in Chap. 2, Prop. 3.4 of
Golubitsky & Guillemin (1973). Since the topology on the equivariant function space is induced
from the general function space, the restricted map is also continuous. Similarly, Res _M_ ( _H_ ) and the
inclusion are continuous. From this analysis, we can pull back the neighborhood _Ux_ to obtain a
neighborhood _Ux_ _[′]_ [of] _[ f]_ [in] _[ C]_ _G_ _[∞]_ [(] _[M, Y]_ [ )][ where the transversality holds.]

We deal with the global result. For any _i_ with dim _Y_ ( _Hi_ ) _k_ 1, the collection of neighborhoods
_≤_ _−_
_Vx_ _x∈f −_ 1( _Y_ ( _Hi_ )) for all _i_ with dim _Y_ ( _Hi_ ) _k_ 1 forms an open cover of _f_ _[−]_ [1] ( _Y_ ( _Hi_ )). By the frontier
_{_ _}_ _≤_ _−_
condition, any stratum that intersects the closure _Y_ ( _Hi_ ) has strictly smaller dimension than _Y_ ( _Hi_ ).
Hence [�] _i_ _[Y]_ [(] _[H]_ _i_ [)] [is closed.] [Since] _[ M]_ [is compact and] _[ f]_ [is continuous, the preimage] _[ f][ −]_ [1][(][�] _i_ _[Y]_ [(] _[H]_ _i_ [)][)][ is]


_i_ _[Y]_ [(] _[H]_ _i_ [)] [is closed.] [Since] _[ M]_ [is compact and] _[ f]_ [is continuous, the preimage] _[ f][ −]_ [1][(][�]


Hence [�] _i_ _[Y]_ [(] _[H]_ _i_ [)] [is closed.] [Since] _[ M]_ [is compact and] _[ f]_ [is continuous, the preimage] _[ f][ −]_ [1][(][�] _i_ _[Y]_ [(] _[H]_ _i_ [)][)][ is]

compact, so there exists a finite subcover _{Vxn}n_ _[N]_ =1 [of the preimage.] [We construct the neighborhood]
_Vk_ := [�] _n_ _[N]_ =1 _[V][x]_ _n_ [of] [the] [preimage] [in] _[M]_ [,] [and] [the] [neighborhood] _[U]_ _k_ [ (1)] := [�] _n_ _[N]_ =1 _[U]_ _x_ _[ ′]_ _n_ [of] _[f]_ [in] [the]
function space.


In the _C_ [0] topology, by Lem. D.12, there exists a neighborhood _Uk_ [(2)] of _f_ in _C_ ( _M, Y_ ) such that for
any _g_ _∈_ _Uk_ [(2)][,] _g_ ( _M_ _Vk_ ) _Y_ ( _Hi_ ) = ∅ _._ (94)
_\_ _∩_

_i_


In the _C_ [0] topology, by Lem. D.12, there exists a neighborhood _Uk_ [(2)] of _f_ in _C_ ( _M, Y_ ) such that for
any _g_ _∈_ _Uk_ [(2)][,] _g_ ( _M_ _Vk_ ) _Y_ ( _Hi_ ) = ∅ _._ (94)
_\_ _∩_


36


By pulling this back through the inclusions _CG_ _[∞]_ [(] _[M, Y]_ [ )] _[�][→]_ _[C]_ _[∞]_ [(] _[M, Y]_ [ )] _[�][→]_ _[C]_ [(] _[M, Y]_ [ )][, we obtain a]
neighborhood ( _Uk_ [(2)][)] _[′]_ [ in] _[ C]_ _G_ _[∞]_ [(] _[M, Y]_ [ )][.] [The neighborhood is open in the] _[ C]_ [0][ topology, so it also open]
in the _C_ _[∞]_ topology. Let _Uk_ _[′]_ [=] _[ U][k][−]_ [1] _[ ∩]_ _[U]_ _k_ [ (1)] _∩_ ( _Uk_ [(2)][)] _[′]_ [.] [For any map] _[ g]_ _[∈]_ _[U]_ _k_ _[ ′]_ [, we have]

        - _j_ 0( _g|M_ ( _H_ ) ) ⋔ _S_ ( _H_ ) for _x ∈_ _Vk_ (95)
_g_ ( _x_ ) _/_ _Y_ ( _Hi_ ) for _x_ _M_ _Vk_
_∈_ _∈_ _\_

for all _i_ with dim _Y_ ( _Hi_ ) _k_ 1.
_≤_ _−_

Now we show by contradiction that there exists a neighborhood _Uk_ _⊂_ _Uk_ _[′]_ [of] _[ f]_ [where the transversality]
condition holds for strata of dimension _k_ . By the induction hypothesis, the transversality condition
holds for all strata of dimension at most _k −_ 1. Therefore, it also holds for all strata of dimension
at most _k_ . The proof idea is from the proof of openness of equivariant general position in Sec. 7 of
Bierstone (1977), where the closedness of _P_ is replaced by the condition that _f_ does not intersect
low-dimensional _Y_ ( _Hi_ ) outside of _Vk_ .

Assume openness does not hold. Then there exists a sequence _gn_ _f_ in _Uk_ _[′]_ [such that each] _[ g][n]_
_{_ _}_ _→_
fails the stratumwise transversality condition for some stratum of dimension _k_ . Since the stratumwise
transversality condition fails, it follows from Prop. D.17 that _gn_ ⋔ _G_ _Y_ ( _Hj_ ) for all _j_ with dim _Y_ ( _Hj_ ) =
_k_ also does not hold. The points of non-transversality for these maps, _yn_, can only occur in _M_ _Vk_
_{_ _}_ _\_
and _gn_ ( _yY_ ) _j_ _[Y]_ [(] _[H]_ _j_ [)][.] [Since] _[ M]_ [is compact, there is a convergent subsequence] _[ {][y][n][i][}]_ [ with]
_∈_ [�] _[\][ V][k]_

_[′]_


and _gn_ ( _yY_ ) _j_ _[Y]_ [(] _[H]_ _j_ [)][.] [Since] _[ M]_ [is compact, there is a convergent subsequence] _[ {][y][n][i][}]_ [ with]
_∈_ [�] _[\][ V][k]_

limit _x_ . Then _gn_ ( _yY_ ) _→_ _f_ ( _x_ ). By construction of _Uk_ _[′]_ [,] _[ f]_ [(] _[M]_ _[\][ V][k]_ [)][ does not intersect] _[ Y]_ [(] _[H]_ _i_ [)] [for any]
dim _Y_ ( _Hi_ ) _k_ 1. Thus, we must have _f_ ( _x_ ) _Y_ ( _Hj_ ) for some _j_ .
_≤_ _−_ _∈_

We claim that _gni_ ( _yni_ ) _Y_ ( _Hj_ ) for all sufficiently large _i_ . Otherwise, there exists a further sub_∈_
sequence _{gniα_ ( _yniα_ ) _}_ such that _gniα_ ( _yniα_ ) _∈_ _Y_ ( _Hj′_ ) for some _j_ _[′]_ = _j_ and all _α_, while still
_gniα_ ( _yniα_ ) _→_ _f_ ( _x_ ) _∈_ _Y_ ( _Hj_ ). Hence _Y_ ( _Hj_ ) _∩_ _Y_ ( _Hj′_ ) = ∅. By the frontier condition of stratification
this implies dim _Y_ ( _Hj_ ) _<_ dim _Y_ ( _Hj′_ ). However, _Y_ ( _Hj_ ) and _Y_ ( _Hj′_ ) have the same dimension, yielding
a contradiction. Therefore, for _i_ sufficiently large, _gni_ ( _yni_ ) _Y_ ( _Hj_ ).
_∈_

We now show that this contradicts the assumption that _f_ ⋔ _G_ _Y_ ( _Hj_ ) at _x_ . If _f_ were to satisfy
_f_ ⋔ _G_ _Y_ ( _Hj_ ) at _x_, then by Prop. D.18, there would exist a neighborhood _U_ of _f_ and a _G_ -invariant
neighborhood _V_ of _G_ ( _x_ ) such that any _g_ _∈_ _U_ satisfies _g_ ⋔ _G_ _Y_ ( _Hj_ ) at any _y_ _∈_ _V_ . This contradicts the
existence of sequence _gn_ and points _yn_ where transversality fails. This completes the proof.
_{_ _}_ _{_ _}_

**Proposition D.20.** _Let F_ _be a C_ _[∞]_ _-dense family of smooth parameterized maps in CG_ _[∞]_ [(] _[X, Y]_ [ )] _[ and]_
_Mj_ _be a finite collection of compact, connected and smooth G-submanifolds of X._ _Let_
_{_ _}_


_S_ ( _H_ ) _→_ ( _H_ _′_ )( _f_ ) = _x_ _X_ ( _Gx_ ) = ( _H_ ) _,_ ( _Gf_ ( _x_ )) = ( _H_ _[′]_ ) _._ (96)
_{_ _∈_ _|_ _}_


_There is a C_ _[∞]_ _-dense subset_ _, for g_ _, the set S_ ( _H_ ) _→_ ( _H_ _′_ )( _g_ _Mj_ ) _is a disjoint union of smooth_
_G_ _⊂F_ _∈G_ _|_
_G-submanifolds of X, and its dimension satisfies_

dim _S_ ( _H_ ) _→_ ( _H_ _′_ )( _g|Mj_ ) = dim( _Mj_ )( _H_ ) _−_ (dim _Y_ _[H]_ _−_ dim _Y_ ( _[H]_ _H_ _[′]_ ) [)] (97)

_ifS_ ( _Hthe_ ) _→right-hand_ ( _H_ _′_ )( _g_ _Mj_ ) _is empty.side_ _of_ _the_ _equation_ _is_ _not_ _smaller_ _than_ dim _G_ _−_ dim _H._ _Otherwise,_ _the_ _set_
_|_

_Proof._ By Prop. D.19, for each _Mj_, there exists an open dense set in _CG_ _[∞]_ [(] _[M][j][, Y]_ [ )][ such that for any]
map _g_ in this set,

_j_ [0] ( _g_ ( _Mj_ )( _H_ ) ) ⋔ (( _Mj_ )( _H_ ) _Y_ ( _H_ _′_ ))( _H_ ) for all ( _H_ _[′]_ ) ( _H_ ) _._ (98)
_|_ _×_ _≥_

Therefore, writing codim _X_ _Y_ := dim _X_ dim _Y_ for the codimension of _Y_ in _X_, the dimension

_−_
theorem for transverse maps (see, e.g., Sec. 2.3 of Arnold et al. (2012)) yields


codim( _Mj_ )( _H_ ) _S_ ( _H_ ) _→_ ( _H_ _′_ ) = codim(( _Mj_ )( _H_ ) _×Y_ )( _H_ ) �( _Mj_ )( _H_ ) _Y_ ( _H_ _′_ )�
_×_

Furthermore, from the proof of Lem. D.11, we have


(99)
( _H_ ) _[.]_


(( _Mj_ )( _H_ ) _× Y_ ( _H_ _′_ ))( _H_ ) = (( _Mj_ ) _[H]_ ( _H_ ) _[×][ Y]_ ( _[H]_ _H_ _[′]_ ) [)] _[ ×][N]_ _G_ [(] _[H]_ [)] _[/H]_ _[G/H.]_ (100)


37


Thus, we obtain
codim( _Mj_ )( _H_ ) _S_ ( _H_ ) _→_ ( _H_ _′_ ) = codim _Y_ _H_ _Y_ ( _[H]_ _H_ _[′]_ ) _[.]_ (101)

Moreover, since the dimension of an orbit with orbit typeif the dimension of _S_ ( _H_ ) _→_ ( _H_ _′_ ) calculated above is less than ( _H_ dim) in a _G − G_ -manifold isdim _H_, then dim _S_ ( _H G_ ) _→ −_ ( _H_ dim _′_ ) is an _H_,
empty set.

For each _Mj_, we take the corresponding open dense set _Uj_ _⊂_ _CG_ _[∞]_ [(] _[M][j][, Y]_ [ )] [on] [which] [the] [maps]
satisfy the dimension theorem. _f_ _|Mj_ _∈_ _Uj_ contains an open Then, by Cor.dense set _Uj_ _[′]_ [.] D.14 [Since], the set of functions _[{][M][j][}]_ [is] [a] [finite] _f_ [collection,] _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [the][ )][intersection][ that satisfy]
_U_ = [�] _j_ _[U]_ _j_ _[ ′]_ [is also an open dense set.] [Therefore, the intersection of] _[ U]_ [with the] _[ C]_ _[∞]_ [-dense set] _[ F]_ [is]

a dense set _G_ . Thus, this intersection is the set of maps in _F_ that we sought to construct, which is
dense and whose elements satisfy the dimension theorem.


D.5 PROOF OF THM. 5.2


**Theorem 5.2.** _Let F_ _be a equivariant parametrization with C_ _[∞]_ _approximation capability._ _If for_
_every_ ( _H_ ) _∈OG_ ( _M_ ) _we have_ ( _pY_ ( _H_ )) _∈OG_ ( _Y_ ) _, then for any finite union of compact, smooth_
_G-submanifolds M_ _⊂_ _X, any f_ _∈_ _CG_ _[∞]_ [(] _[X, Y]_ [ )] _[, any integer][ r]_ _[≥]_ [0] _[, and any][ ϵ >]_ [ 0] _[, there exists a map]_
_g_ _∈F_ _such that_
max _x∈M_ _D_ _[k]_ _f_ ( _x_ ) _D_ _[k]_ _g_ ( _x_ ) _< ϵ, k_ _r,_ (11)
_∥_ _−_ _∥_ _≤_
_and_ _g_ _M_ _is_ _almost_ _isovariant_ _relative_ _to_ _Y ._ _Furthermore,_ _if_ _the_ _feature_ _space_ _Y_ _contains_ _a_
_|_
_representation_ _Y_ [˜] _[⊕][r]_ _for_ _an_ _integer_ _r_ _>_ max _j_ dim _Mj_ _,_ _where_ _Y_ [˜] _itself_ _satisfies_ _the_ _condition_
_{_ _}_
( _p_ ˜ _Y_ ( _H_ )) _∈OG_ ( _Y_ [˜] ) _, then the approximating map g|M_ _can be chosen to be isovariant relative to Y ._


_Proof of Thm. 5.2._ By Lem. D.8, we consider the relation

dim _NG_ ( _H, H_ _[′]_ ) dim _NG_ ( _H_ ) = _α_ ( _H, H_ _[′]_ ) = dim _Y_ ( _[H]_ _H_ _[′]_ ) _[H]_ _[′][.]_ (102)
_−_ _[−]_ [dim] _[ Y]_

Regarding the _G_ -representation as a faithful representation of _G/_ ker _ρY_, for orbit types satisfying
( _pY_ ( _H_ _[′]_ )) _>_ _I_ ( _H, Y_ ), the dense and open property of the minimal orbit type by Prop. B.4 in the
fixed-point space implies that for each _Mj_,

dim _Y_ _[H]_ _−_ dim _Y_ ( _[H]_ _H_ _[′]_ ) [= dim] _[ Y]_ _[H]_ _[−]_ [dim] _[ Y]_ _[H]_ _[′]_ _[−]_ _[α][G]_ [(] _[H, H]_ _[′]_ [)] _[ >]_ [ 0] _[.]_ (103)


By Prop. D.20, it implies


dim( _Mj_ )( _H_ ) _>_ dim _S_ ( _H_ ) _→_ ( _H_ _′_ )( _g_ _Mj_ ) _._ (104)
_|_


With respect to the Hausdorff measure, since the dimension of _S_ ( _H_ ) _→_ ( _H_ _′_ )( _g_ _Mj_ ) is strictly less
_H_ _[d]_ _|_
than the dimension of the manifold it lies in, its measure is zero. Then, by the finiteness of the set of
orbit types for a compact manifold and the finiteness of the collection _Mj_, it follows that _g_ _M_ is an
_{_ _}_ _|_
almost isovariant map relative to _Y_ .


Furthermore, if we require _g_ to be isovariant relative to _Y_, we need


_S_ ( _H_ ) _→_ ( _H_ _′_ )( _g_ _Mj_ ) = ∅ _._ (105)
_|_

The condition for this set to be empty is related to its codimension. Since the inequality


dim _Y_ _[H]_ dim _Y_ _[H]_ _[′]_ _αG_ ( _H, H_ _[′]_ ) 1 (106)

_−_ _−_ _≥_

scales with multiplicity _r_ to become

_r_ (dim _Y_ _[H]_ dim _Y_ _[H]_ _[′]_ ) _αG_ ( _H, H_ _[′]_ ) _r,_ (107)

_−_ _−_ _≥_

it is sufficient to choose the multiplicity for the representation _Y_ _[⊕][r]_ such that _r_ _>_ max _j_ dim _Mj_ .
_{_ _}_


38


E TABLES


All closed subgroups of _O_ (3) are denoted using Schoenflies notation, where it should be noted that
_K_ = _SO_ (3) and _Kh_ = _O_ (3). When interpreting the tables, care should be taken to recognize the
low-dimensional equivalences: _Cs_ = _C_ 1 _h_ = _C_ 1 _v_, _D_ 1 = _C_ 2, _D_ 1 _h_ = _C_ 2 _v_, and _D_ 1 _d_ = _C_ 2 _h_ . In the
tables of symmetry infimum, we list a representative subgroup from each conjugacy class and omit
the class notation (e.g., _C_ 2 instead of ( _C_ 2)) for clarity.


E.1 MINIMAL PROPER SUPERGROUPS IN _SO_ (3) OR _O_ (3)


Minimal proper supergroups table of _SO_ (3) or _O_ (3). Some of the results can be found from
Fig. 3.2.1.6 of Aroyo (2016). We only present the results for _O_ (3). The discussion for _SO_ (3) can be
obtained by removing the subgroups that are not subgroups of _SO_ (3) (i.e., the subgroups of the first
kind), namely _Ck, Dk, T, O, I, C∞, D∞_, and _K_ .


Table 6: Table of minimal proper supergroups _H_ of axial closed subgroups of _O_ (3). In the supergroup
notation, _p_ denotes a prime number and _p_ _[∗]_ denotes an odd prime number.

_G_ _H_ for General _k_ _H_ for Special _k_


_Ck_ _Cpk, S_ 2 _k, Ckh, Ckv, Dk_ _Dp∗_ ( _k_ = 2); _T_ ( _k_ = 3)
_S_ 2 _k_ _S_ 2 _p_ _[∗]_ _k, C_ 2 _k,h, Dkd_ _Th_ ( _k_ = 3)
_Ckh_ _Cpk,h, Dkh_ _Cpv_ ( _k_ = 1); _Dp_ _[∗]_ _d_ ( _k_ = 2)
_Ckv_ ( _k_ _>_ 1) _Cpk,v, Dkh, Dkd_ _Td_ ( _k_ = 3); _Dph_ ( _k_ = 2)
_Dk_ ( _k_ _>_ 1) _Dpk, Dkh, Dkd_ _T_ ( _k_ = 2); _O_ ( _k_ = 3 _,_ 4); _I_ ( _k_ = 3 _,_ 5)
_Dkh_ ( _k_ _>_ 1) _Dpk,h_ _Th_ ( _k_ = 2); _Oh_ ( _k_ = 4)
_Dkd_ ( _k_ _>_ 1) _Dp∗k,d, D_ 2 _k,h_ _Td_ ( _k_ = 2); _Oh_ ( _k_ = 3); _Ih_ ( _k_ = 3 _,_ 5)


Table 7: Table of minimal proper supergroups _H_ of other closed subgroups _O_ (3).

_G_ _H_


_T_ _Td, Th, O, I_
_Td_ _Oh_
_Th_ _Oh, Ih_
_O_ _Oh, K_
_Oh_ _Kh_
_I_ _Ih, K_
_Ih_ _Kh_
_C∞_ _C∞h, C∞,v, D∞_
_C∞h_ _D∞,h_
_C∞v_ _D∞,h_
_D∞_ _D∞,h, K_
_D∞h_ _Kh_
_K_ _Kh_


39


E.2 DIMENSIONS OF FIXED-POINT SUBSPACES FOR SUBGROUPS OF _SO_ (3) OR _O_ (3)


Dimensions table of fixed-point subspaces for subgroups of _SO_ (3) or _O_ (3). Some of the results can
be found from Table B.1 and Table B.2 of Linehan & Stedman (2001). We only present the results
for _O_ (3). The discussion for _SO_ (3) can be obtained directly from the table.


Table 8: Dimensions of fixed-point subspaces for closed subgroups of _O_ (3) acting on the irreducible

|sentations Vl=l 0, w|Col2|where ak(l) = ⌊l/k⌋and bk(l) =|Col4|⌊(l + k)/(2k)⌋.|Col6|
|---|---|---|---|---|---|
|<br>Subgroup|<br>Subgroup|<br>_l_ =_ l−_<br>0|<br>_l_ =_ l−_<br>0|<br>_l_ =_ l_+<br>0|<br>_l_ =_ l_+<br>0|
|<br>Subgroup|<br>Subgroup|_l_0 even|_l_0 odd|_l_0 even|_l_0 odd|
|_Ck_||2_ak_(_l_0) + 1|2_ak_(_l_0) + 1|2_ak_(_l_0) + 1|2_ak_(_l_0) + 1|
|_S_2_k_|_k_ even|2_bk_(_l_0)|2_bk_(_l_0)|2_a_2_k_(_l_0) + 1|2_a_2_k_(_l_0) + 1|
|_S_2_k_|_k_ odd|0|0|2_ak_(_l_0) + 1|2_ak_(_l_0) + 1|
|_Ckh_|_k_ even|0|0|2_ak_(_l_0) + 1|2_ak_(_l_0) + 1|
|_Ckh_|_k_ odd|2_bk_(_l_0)|2_bk_(_l_0)|2_a_2_k_(_l_0) + 1|2_a_2_k_(_l_0) + 1|
|_Ckv_||_ak_(_l_0)|_ak_(_l_0) + 1|_ak_(_l_0) + 1|_ak_(_l_0)|
|_Dk_||_ak_(_l_0) + 1|_ak_(_l_0)|_ak_(_l_0) + 1|_ak_(_l_0)|
|_Dkh_|_k_ even|0|0|_ak_(_l_0) + 1|_ak_(_l_0)|
|_Dkh_|_k_ odd|_bk_(_l_0)|_bk_(_l_0)|_a_2_k_(_l_0) + 1|_a_2_k_(_l_0)|
|_Dkd_|_k_ even|_bk_(_l_0)|_bk_(_l_0)|_a_2_k_(_l_0) + 1|_a_2_k_(_l_0)|
|_Dkd_|_k_ odd|0|0|_ak_(_l_0) + 1|_ak_(_l_0)|
|_T_||2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_Th_||0|0|<br>2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>2_a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_Td_||_a_3(_l_0) +_ b_2(_l_0) +_ b_1(_l_0)_ −l_0|_a_3(_l_0) +_ b_2(_l_0) +_ b_1(_l_0)_ −l_0|<br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_O_||<br> <br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br> <br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br> <br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br> <br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_Oh_||0|0|<br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_4(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_I_||<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_Ih_||0|0|<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|<br>_a_5(_l_0) +_ a_3(_l_0) +_ a_2(_l_0)_ −l_0 + 1|
|_C∞_||<br>1|<br>1|<br>1|<br>1|
|_C∞h_||0|0|1|1|
|_C∞v_||0|1|1|0|
|_D∞_||1|0|1|0|
|_D∞h_||0|0|1|0|


40


E.3 SYMMETRY INFIMUM FOR SUBGROUPS OF _SO_ (3)


Symmetry infimum table for subgroups of _SO_ (3). In the following tables, we use a color-coding
scheme to classify the types of symmetry increase. An increase to the full group is termed **full**
**degeneration** and is marked in red . An increase to a supergroup of a strictly higher dimension is

termed **continuous degeneration** and is marked in blue . An increase to a supergroup of the same

dimension is termed **discrete degeneration**, and is marked in yellow . No increase is termed **no**
**degeneration**, and is marked in green . All subgroups increase to to _K_ when _l_ = 0.


Table 9: Symmetry infimum of general axis subgroup of _SO_ (3) on _Vl_ _[⊕]_ = _[r]_ _l_ 0 [,] _[ r]_ _[>]_ [ 3][,] _[ l]_ [0] _[>]_ [ 0][.]

|Col1|l < k<br>0|Col3|l ≥k<br>0|
|---|---|---|---|
||_l_0 even|_l_0 odd|_l_0 odd|
|_Ck_|_D∞_|_C∞_|_Ck_|
|_Dk_(_k >_ 2)|_D∞_|_K_|_Dk_|


Table 10: Symmetry infimum of special axis subgroup of _SO_ (3) on _Vl_ _[⊕]_ = _[r]_ _l_ 0 [,] _[ r]_ _[>]_ [ 3][,] _[ l]_ [0] _[>]_ [ 0][.]

|Col1|l = 1<br>0|l = 2<br>0|l = 3<br>0|l ≥4<br>0|
|---|---|---|---|---|
|_D_2|_K_|_D_2|_T_|<br>_D_2|


Table 11: Symmetry infimum of polyhedral subgroup of _SO_ (3) on _Vl_ _[⊕]_ = _[r]_ _l_ 0 [,] _[r]_ _[>]_ [3][,] _[l]_ [0] _[>]_ [0][.] [The]
"Caption" in the table takes the value _l_ 0 = 6 _,_ 10 _,_ 12 _,_ 15 _,_ 16 _,_ 18 _,_ 20 22 _,_ 24 28.

|T|l = 6, 9, 10 l = 3, 7 l = 4, 8 l ≥12 other<br>0 0 0 0|
|---|---|
|_T_|<br>_T_<br>_T_<br>_O_<br>_T_<br>_K_|
|_O_|_l_0 = 4_,_ 6_,_ 8_,_ 9_,_ 10<br>_l_0 _≥_12<br>other|
|_O_|<br>_O_<br>_O_<br>_K_|
|_I_|Caption<br>_l_0 _≥_30<br>other|
|_I_|<br>_I_<br>_I_<br>_K_|


Table 12: Symmetry infimum of infinite subgroup of _SO_ (3) on _Vl_ _[⊕]_ = _[r]_ _l_ 0 [,] _[ r]_ _[>]_ [ 3][,] _[ l]_ [0] _[>]_ [ 0][.]

|Col1|l even<br>0|l odd<br>0|
|---|---|---|
|_C∞_|_D∞h_|_C∞h_|
|_D∞_|_D∞h_|_Kh_|
|_K_|_K_|_K_|


41


E.4 SYMMETRY INFIMUM FOR SUBGROUPS OF _O_ (3)


Symmetry Infimum table for Subgroups of _O_ (3). The color scheme for full, continuous, and no
degeneration is the same as for _SO_ (3). For cases of **discrete degeneration** of _l_ 0 = _l_ [+], we distinguish
between (a) light Green, which indicates the predictable increase to _π_ _[−]_ [1] ( _π_ ( _H_ )) for the projection

map _π_ : _O_ (3) _→_ _SO_ (3), and (b) yellow, which indicates all other exceptional cases of discrete
degeneration. In the following table, the first type of subgroups degenerate to _K_ and the others to _Kh_
when _l_ 0 = 0 _[−]_ . All subgroups increase to to _Kh_ when _l_ 0 = 0 [+] .

|Table 13: Symmetry infimu|Col2|um of general axis su|Col4|ubgroup of O(3) on|Col6|V l=l− 0, r > 3, l0 > 0|Col8|
|---|---|---|---|---|---|---|---|
|Subgroup|Subgroup|0_ < l_0 _< k_|0_ < l_0 _< k_|_k ≤l_0 _<_ 2_k_|_k ≤l_0 _<_ 2_k_|<br>_l_0 _≥_2_k_|<br>_l_0 _≥_2_k_|
|Subgroup|Subgroup|_l_0 even|_l_0 odd|<br>_l_0 even|_l_0 odd|<br>_l_0 even|_l_0 odd|
|_Ck_||_D∞_|_C∞v_|_Ck_|_Ck_|_Ck_|_Ck_|
|_S_2_k_|_k_ even|_Kh_|_Kh_|_S_2_k_|_S_2_k_|_S_2_k_|_S_2_k_|
|_S_2_k_|_k_ odd|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|
|_Ckh_|_k_ even|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|
|_Ckh_|_k_ odd|_Kh_|_Kh_|_Ckh_|_Ckh_|_Ckh_|_Ckh_|
|_Ckv_(_k >_ 2)|_k_ even|_Kh_|_C∞v_|_Dkd_|_Ckv_|_Ckv_|_Ckv_|
|_Ckv_(_k >_ 2)|_k_ odd|_Kh_|_C∞v_|_Dkh_|_Ckv_|_Ckv_|_Ckv_|
|_Dk_(_k >_ 2)|_k_ even|_D∞_|_Kh_|_Dk_|_Dkd_|_Dk_|_Dk_|
|_Dk_(_k >_ 2)|_k_ odd|_D∞_|_Kh_|_Dk_|_Dkh_|_Dk_|_Dk_|
|_Dkh_(_k >_ 2)|_k_ even|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|
|_Dkh_(_k >_ 2)|_k_ odd|_Kh_|_Kh_|_Dkh_|_Dkh_|_Dkh_|_Dkh_|
|_Dkd_(_k >_ 2)|_k_ even|_Kh_|_Kh_|_Dkd_|_Dkd_|_Dkd_|_Dkd_|
|_Dkd_(_k >_ 2)|_k_ odd|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|


|Table 14: Symmetry infimu|Col2|um of general axis su|Col4|ubgroup of O(3) on|Col6|V l=l+ 0, r > 3, l0 > 0|Col8|
|---|---|---|---|---|---|---|---|
|Subgroup|Subgroup|0_ < l_0 _< k_|0_ < l_0 _< k_|_k ≤l_0 _<_ 2_k_|_k ≤l_0 _<_ 2_k_|<br>_l_0 _≥_2_k_|<br>_l_0 _≥_2_k_|
|Subgroup|Subgroup|_l_0 even|_l_0 odd|<br>_l_0 even|_l_0 odd|<br>_l_0 even|_l_0 odd|
|_Ck_|_k_ even|_D∞h_|_C∞h_|_Ckh_|_Ckh_|_Ckh_|_Ckh_|
|_Ck_|_k_ odd|_D∞h_|_C∞h_|_S_2_k_|_S_2_k_|_S_2_k_|_S_2_k_|
|_S_2_k_|_k_ even|_D∞h_|_C∞h_|_D∞h_|_C∞h_|_C_2_kh_|_C_2_kh_|
|_S_2_k_|_k_ odd|_D∞h_|_C∞h_|_S_2_k_|_S_2_k_|_S_2_k_|_S_2_k_|
|_Ckh_|_k_ even|_D∞h_|_C∞h_|_Ckh_|_Ckh_|_Ckh_|_Ckh_|
|_Ckh_|_k_ odd|_D∞h_|_C∞h_|_D∞h_|_C∞h_|_C_2_kh_|_C_2_kh_|
|_Ckv_(_k >_ 2)|_k_ even|_D∞h_|_Kh_|_Dkh_|_Dkh_|_Dkh_|_Dkh_|
|_Ckv_(_k >_ 2)|_k_ odd|_D∞h_|_Kh_|_Dkd_|_Dkd_|_Dkd_|_Dkd_|
|_Dk_(_k >_ 2)|_k_ even|_D∞h_|_Kh_|_Dkh_|_Dkh_|_Dkh_|_Dkh_|
|_Dk_(_k >_ 2)|_k_ odd|_D∞h_|_Kh_|_Dkd_|_Dkd_|_Dkd_|_Dkd_|
|_Dkh_(_k >_ 2)|_k_ even|_D∞h_|_Kh_|_Dkh_|_Dkh_|_Dkh_|_Dkh_|
|_Dkh_(_k >_ 2)|_k_ odd|_D∞h_|_Kh_|_D∞h_|_Kh_|_D_2_kh_|_D_2_kh_|
|_Dkd_(_k >_ 2)|_k_ even|_D∞h_|_Kh_|_D∞h_|_Kh_|_D_2_kh_|_D_2_kh_|
|_Dkd_(_k >_ 2)|_k_ odd|_D∞h_|_Kh_|_Dkd_|_Dkd_|_Dkd_|_Dkd_|


42


|Col1|Col2|Col3|Col4|l=l 0|Col6|
|---|---|---|---|---|---|
||_l_0 = 1|_l_0 = 2|_l_0 = 3|<br>_l_0 _≥_4|<br>_l_0 _≥_4|
||_l_0 = 1|_l_0 = 2|_l_0 = 3|<br>_l_0 even|_l_0 odd|
|_C_2_v_|_C∞v_|_D_2_d_|_C_2_v_|_C_2_v_|_C_2_v_|
|_D_2|_Kh_|_D_2|_Td_|_D_2|_D_2|
|_D_2_h_|_Kh_|_Kh_|_Kh_|_Kh_|_Kh_|
|_D_2_d_|_Kh_|_D_2_d_|_Td_|_D_2_d_|_D_2_d_|


|Col1|Col2|Col3|Col4|l=l 0|Col6|
|---|---|---|---|---|---|
||_l_0 = 1|_l_0 = 2|_l_0 = 3|<br>_l_0 _≥_4|<br>_l_0 _≥_4|
||_l_0 = 1|_l_0 = 2|_l_0 = 3|<br>_l_0 even|_l_0 odd|
|_C_2_v_|_Kh_|_D_2_h_|_Th_|_D_2_h_|_D_2_h_|
|_D_2|_Kh_|_D_2_h_|_Th_|_D_2_h_|_D_2_h_|
|_D_2_h_|_Kh_|_D_2_h_|_Th_|_D_2_h_|_D_2_h_|
|_D_2_d_|_Kh_|_D∞h_|_Kh_|_D_4_h_|_D_4_h_|


Table 17: Symmetry infimum of polyhedral subgroup of _O_ (3) on _Vl_ _[⊕]_ = _[r]_ _l_ 0 _[−]_ [,] _[r]_ _[>]_ [3][,] _[l]_ [0] _[>]_ [0][.] [The]
"Caption" in the table takes the value _l_ 0 = 6 _,_ 10 _,_ 12 _,_ 15 _,_ 16 _,_ 18 _,_ 20 22 _,_ 24 28.

|Col1|− − l = 6, 9, 10 l = 3, 7 l = 4, 8 l ≥12|
|---|---|
|_T_|<br>_l_0 = 6_,_ 9_,_ 10<br>_l_0 = 3_,_ 7<br>_l_0 = 4_,_ 8<br>_l_0 _≥_12<br>other|
|_T_|<br>_T_<br>_Td_<br>_O_<br>_T_<br>_Kh_|
|_Td_|_l_0 = 3_,_ 6_,_ 7<br>_l_0 _≥_9<br>other|
|_Td_|<br>_Td_<br>_Td_<br>_Kh_|
|_Th_|all_ l_0|
|_Th_|_Kh_|
|_O_|_l_0 = 4_,_ 6_,_ 8_,_ 9_,_ 10<br>_l_0 _≥_12<br>other|
|_O_|<br>_O_<br>_O_<br>_Kh_|
|_Oh_|all_ l_0|
|_Oh_|_Kh_|
|_I_|Caption<br>_l_0 _≥_30<br>other|
|_I_|<br>_Ih_<br>_Ih_<br>_Kh_|
|_Ih_|all_ l_0|
|_Ih_|_Kh_|


43


|value l|l0 = 6, 10, 12, 15, 16, 18, 20 −22, 24 −28. l = 3, 6, 7, 9, 10 l = 4, 8 l ≥12|
|---|---|
|_T_|<br>_l_0 = 3_,_ 6_,_ 7_,_ 9_,_ 10<br>_l_0 = 4_,_ 8<br>_l_0 _≥_12<br>other|
|_T_|<br>_Th_<br>_Oh_<br>_Th_<br>_Kh_|
|_Td_|_l_0 = 4_,_ 6_,_ 8_,_ 9_,_ 10<br>_l_0 _≥_12<br>other|
|_Td_|<br>_Oh_<br>_Oh_<br>_Kh_|
|_Th_|_l_0 = 3_,_ 6_,_ 7_,_ 9_,_ 10<br>_l_0 = 4_,_ 8<br>_l_0 _≥_12<br>other|
|_Th_|<br>_Th_<br>_Oh_<br>_Th_<br>_Kh_|
|_O_|_l_0 = 4_,_ 6_,_ 8_,_ 9_,_ 10<br>_l_0 _≥_12<br>other|
|_O_|<br>_Oh_<br>_Oh_<br>_Kh_|
|_Oh_|_l_0 = 4_,_ 6_,_ 8_,_ 9_,_ 10<br>_l_0 _≥_12<br>other|
|_Oh_|<br>_Oh_<br>_Oh_<br>_Kh_|
|_I_|Caption<br>_l_0 _≥_12<br>other|
|_I_|<br>_Ih_<br>_Ih_<br>_Kh_|
|_Ih_|Caption<br>_l_0 _≥_12<br>other|
|_Ih_|<br>_Ih_<br>_Ih_<br>_Kh_|


|Col1|l even<br>0|l odd<br>0|
|---|---|---|
|_C∞_|_D∞_|_C∞v_|
|_C∞h_ =_ S_2_∞_|_Kh_|_Kh_|
|_C∞v_|_Kh_|_C∞v_|
|_D∞_|_D∞_|_Kh_|
|_D∞h_ =_ D∞d_|_Kh_|_Kh_|
|_K_|_Kh_|_Kh_|
|_Kh_|_Kh_|_Kh_|


|Col1|l even<br>0|l odd<br>0|
|---|---|---|
|_C∞_|_D∞h_|_C∞h_|
|_C∞h_ =_ S_2_∞_|_D∞h_|_C∞h_|
|_C∞v_|_D∞h_|_Kh_|
|_D∞_|_D∞h_|_Kh_|
|_D∞h_=_D∞d_|_D∞h_|_Kh_|
|_K_|_Kh_|_Kh_|
|_Kh_|_Kh_|_Kh_|


44


F DETAILED EXPERIMENT


Experiments in § 6.1 and 6.2 are conducted with randomly initialized TFN (Thomas et al., 2018)
and HEGNN (Cen et al., 2024) architectures as the underlying equivariant models, whereas the
experiments in § 6.3 first pretrain a HEGNN encoder to obtain molecular embeddings and then
fine-tune separate MLP heads for the final prediction task under different settings.


For TFN (Thomas et al., 2018), we use the implementation from GWL-test (Joshi et al., 2023), which
relies on irreducible-representation features with alternating parity. For HEGNN, we extend the
original design to a multi-channel variant. In particular, the spherical scalarized features computed
for nodes _j_ and _i_ are defined as


_**z**_ _ij,c_ [(] _[l]_ [)] [= 1] _[/]_ _√_


_C · ⟨_ _**v**_ ˜ _i,c_ [(] _[l]_ [)] _[,]_ [ ˜] _**[v]**_ _j,c_ [(] _[l]_ [)] _[⟩][,]_ (108)


where _**v**_ ˜ _i,c_ [(] _[l]_ [)] [and] _**[v]**_ [˜] _j,c_ [(] _[l]_ [)] [denote the] _[ l]_ [-th degree steerable features of channel] _[ c]_ [ among a total of] _[ C]_ [channels.]
To obtain the degree- _l_ 0 graph-level representation, we apply global mean pooling for each graph
_G_ ( _V, E_ ):
_**v**_ ˜ _G_ [(] _[l]_ _,c_ [)] [= 1] _[/][|V| ·]_ [ �] _i_ _**[v]**_ [˜] _i,c_ [(] _[l]_ [)] _[.]_ (109)


We will detail, in this section, how each experimental task processes these graph-level features.


F.1 VISUALIZATION OF REPRESENTATION SPACE


We use a TFN with single-layer to calculate the embedding. After that, to extract features of degree
_l_ 0, we append an o3.Linear layer from e3nn (Geiger & Smidt, 2022), denoting this setup as
TFN _l_ = _l_ 0 in our experiments.


F.2 EXPRESSIVITY ON SYMMETRIC GRAPHS


This section first introduces the detailed settings of § 6.2 in § F.2.1, and then introduces the reproduction of the original experiment of GWL-test (Joshi et al., 2023) in § F.2.2.


F.2.1 EMBEDDING DIFFERENCE NORM EXPERIMENT


We employed both TFN (Thomas et al., 2018) and HEGNN (Cen et al., 2024) to compute the norm
of the embedding difference across 12 configurations for each model, varying the number of channels
(1, 4, 16) and layers (1, 2, 3, 4). The degree- _l_ discrepancy between two graphs _G_ 0 and _G_ 1 is defined as

∆ [(] _[l]_ [)] = 1 _/|C| ·_ [�] _c_ _[C]_ =1 _[∥]_ _**[v]**_ [˜] _G_ [(] _[l]_ 0 [)] _,c_ _[−]_ _**[v]**_ [˜] _G_ [(] _[l]_ 1 [)] _,c_ _[∥][.]_ (110)

These choices give rise to 2 _×_ 3 _×_ 4 = 24 distinct ∆ [(] _[l]_ [)] . In Fig. 5, we report the maximum norm
across all ∆ [(] _[l]_ [)] . Since every norm is strictly positive, a maximal value below 10 _[−]_ [6] indicates that all
corresponding norms fall below 10 _[−]_ [6], meaning none of them can distinguish _G_ 0 from _G_ 1.


F.2.2 ORIGINAL GWL-TEST ON SYMMETRIC GRAPHS


**Dataset.** Same as the setting in (Joshi et al., 2023), we construct four symmetric _k_ -fold structures
( _k_ _∈{_ 2 _,_ 3 _,_ 4 _,_ 6 _}_ ), each centered at the origin. For each structure _G_ 0 we apply a random rotation to
produce 1 which ensures 1 does not coincide with the original 0. The goal is to evaluate whether
_G_ _G_ _G_
different equivariant neural network architectures can distinguish 0 from 1. To validate distinct
_G_ _G_
aspects of our theory, we consider rotations separately in 2D and 3D; in the 3D experiments we
additionally ensure that _G_ 1 is not coplanar with _G_ 0.

**Embeddings.** The extracted _l_ 0-degree embeddings from TFN are fed into a vanilla classifier for
the classification task. The model was trained for 300 epochs to ensure the classifier had sufficient
capacity to discriminate the classes.


**Results.** The detailed experimental results are presented in Table 21, which can be observed that the
color blocks in this table are completely consistent with Fig. 5. It demonstrates that our theoretical
predictions in Table 1 are in complete agreement with the empirical findings obtained from the model.


45


Such remarkable consistency not only confirms the correctness of our theoretical analysis, but also
highlights the importance of constructing mappings with appropriately chosen features.


Table 21: Results of distinguishing _k_ -fold structures rotated in 2D/3D space.


**2D Rotational Symmetry** **3D Rotational Symmetry**
**GNN Layer** 2 fold 3 fold 4 fold 6 fold 2 fold 3 fold 4 fold 6 fold


TFN _l_ =0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0
TFN _l_ =1 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0 50.0 ± 0.0


F.3 MOLECULE PROPERTY PREDICTION WITH PRETRAINED EQUIVARIANT FEATURES


F.3.1 DETAILED EXPERIMENTAL SETUP


We employ HEGNN as the encoder in our experiments. TFN is computationally prohibitive in this
setting:is set to its _L_ =tensor-product11 (the upperoperatorlimit supportedhas a complexityby e3nn),oftraining _O_ ( _L_ [6] ), aandfour-layerwhen theTFNmaximumwith onlydegreefour
irreducible-representation channels on QM9 requires roughly 10 A100 GPU-hours per epoch. This far
exceeds any practical budget. In contrast, HEGNN adopts spherical scalarization, where interactions
across different degrees are mediated solely through scalars, reducing the complexity to _O_ ( _L_ [2] ). For
this reason, we choose HEGNN as our encoder.


Concretely, we use a four-layer HEGNN with 16 irreducible-representation channels and a hidden
dimension of 64. The resulting features are passed through a scalarization layer and then into a
two-layer MLP to predict the molecular isotropic polarizability _α_ on QM9. During fine-tuning, we
freeze the encoder, apply a mask to selectively remove information, and train a separate two-layer
MLP for each setting. Specifically, our designed mask multiplies the features of the unselected
degrees by 0, followed by scalarization calculation, that is:


_**v**_ ˜ _c_ [(] _[l]_ [)] = HEGNN( ) _,_ (111)
_G_

_**s**_ [(] _[l]_ [)] = vec([ _⟨_ _**v**_ ˜ _c_ [(] _[l]_ 1 [)] _[,]_ [ ˜] _**[v]**_ _c_ [(] _[l]_ 2 [)] _[⟩]_ [])] _[C][×][C][, l]_ [ = 1] _[, . . .,]_ [ 11] _[,]_ (112)

_α_ ˆ = MLPfinetune(˜ _**v**_ [(0)] _,_ _**s**_ [(1)] _, . . .,_ _**s**_ [(11)] ) _,_ (113)


Following the standard protocol, we train on the first 110k molecules for 300 epochs and fine-tune
for an additional 30 epochs. Notably, although the final visualizations are produced by running the
trained models on the entire QM9 dataset [5], this does not affect our theoretical conclusions, as our
analysis relies solely on horizontal comparisons rather than absolute predictive performance.


F.3.2 CASE STUDIES


To further validate our theory, we analyze three prominent point groups _C_ 2 _h_, _C_ 3 _h_, and _Td_ (see Fig. 7).
Each exhibiting a representative pattern of symmetry increase. These groups display a symmetry
increase to the full group _O_ (3) under specific conditions: for _C_ 2 _h_ when _l_ 0 is odd, for _C_ 3 _h_ when
_l_ 0 = 1, and for _Td_ when _l_ 0 = 1 _,_ 2 _,_ 5. This increase corresponds to full degeneration, making the
features non-discriminative. The impact of this degeneration is evident in our experiments. Due
to the full degeneration of 1-degree features, introducing them paradoxically decreases predictive
performance for molecules with _C_ 2 _h_, _C_ 3 _h_, and _Td_ symmetry. Following the introduction of 2-degree
features, a marked improvement in performance is observed for _C_ 2 _h_ and _C_ 3 _h_ . Conversely, for _Td_,
performance again decreases, a result directly attributable to the fact that its 2-degree features also
undergo full degeneration.


5Minor discrepancies between _l_ = 0 in Fig. 6 and _l_ _≤_ 0 in Fig. 7 result from not fixing the random seed and
do not affect the conclusions.


46


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree 0 ~ l


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree 0 ~ l


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


102


101


100


10 1


10 2
0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree 0 ~ l


102


101


100


10 1


10 2


0 1 2 3 4 5 6 7 8 9 10 11
Model Degree 0 ~ l


Figure 7: MAE loss (in units of _a_ [3] 0 [) for isotropic polarizability prediction with degree] _[ l][ ≤]_ _[l]_ [0] [across]
molecules from the top-16 point groups by molecular count. Each boxplot shows the distribution of
errors at a given degree, while diamond markers denote the corresponding mean MAE.


The experiment shows strong dependence between model performance and feature choice on a
real-world dataset validates our discussion on symmetry increase and feature space dimension. It
demonstrates a critical model design principle: not only should one avoid building a model entirely
from fully degenerate features, but one should also avoid including individual feature components
that undergo full degeneration, as they can be actively detrimental to predictive performance.


Table 22: Symmetry infimum of point group symmetry on the QM9 dataset.

|0|1 2 3|4 5 6|7 8 9|10 11|
|---|---|---|---|---|
|_C_1<br>_K_|_C_1<br>_Cs_<br>_C_1|_Cs_<br>_C_1<br>_Cs_|_C_1<br>_Cs_<br>_C_1|_Cs_<br>_C_1|
|_Cs_<br>_Kh_|_Cs_<br>_C_2_h_<br>_Cs_|_C_2_h_<br>_Cs_<br>_C_2_h_|_Cs_<br>_C_2_h_<br>_Cs_|_C_2_h_<br>_Cs_|
|_C_2<br>_K_|_C∞v_<br>_C_2_h_<br>_C_2|_C_2_h_<br>_C_2<br>_C_2_h_|_C_2<br>_C_2_h_<br>_C_2|_C_2_h_<br>_C_2|
|_C_2_v_<br>_Kh_|_C∞v_<br>_D_2_h_<br>_C_2_v_|_D_2_h_<br>_C_2_v_<br>_D_2_h_|_C_2_v_<br>_D_2_h_<br>_C_2_v_|_D_2_h_<br>_C_2_v_|
|_C_3_v_<br>_Kh_|_D∞h_<br>_C∞v_<br>_C_3_v_|_D_3_d_<br>_C_3_v_<br>_D_3_d_|_C_3_v_<br>_D_3_d_<br>_C_3_v_|_D_3_d_<br>_C_3_v_|
|_C_2_h_<br>_Kh_|_Kh_<br>_C_2_h_<br>_Kh_|_C_2_h_<br>_Kh_<br>_C_2_h_|_Kh_<br>_C_2_h_<br>_Kh_|_C_2_h_<br>_Kh_|
|_Ci_ =_ S_2<br>_Kh_|_Kh_<br>_Ci_<br>_Kh_|_Ci_<br>_Kh_<br>_Ci_|_Kh_<br>_Ci_<br>_Kh_|_Ci_<br>_Kh_|
|_D_3_h_<br>_Kh_|_Kh_<br>_D∞h_<br>_D_3_h_|_D∞h_<br>_D_3_h_<br>_D_6_h_|_D_3_h_<br>_D_6_h_<br>_D_3_h_|_D_6_h_<br>_D_3_h_|
|_D_2_d_<br>_Kh_|_Kh_<br>_D∞h_<br>_Td_|_D_4_h_<br>_D_2_d_<br>_D_4_h_|_D_2_d_<br>_D_4_h_<br>_D_2_d_|_D_4_h_<br>_D_2_d_|
|_C_3<br>_K_|_C∞v_<br>_D∞h_<br>_C_3|_C_3_h_<br>_C_3<br>_C_3_h_|_C_3<br>_C_3_h_<br>_C_3|_C_3_h_<br>_C_3|
|_D∞h_<br>_Kh_|_Kh_<br>_D∞h_<br>_Kh_|_D∞h_<br>_Kh_<br>_D∞h_|_Kh_<br>_D∞h_<br>_Kh_|_D∞h_<br>_Kh_|
|_C∞v_<br>_Kh_|_C∞v_<br>_D∞h_<br>_C∞v_|_D∞h_<br>_C∞v_<br>_D∞h_|_C∞v_<br>_D∞h_<br>_C∞v_|_D∞h_<br>_C∞v_|
|_D_3_d_<br>_Kh_|_Kh_<br>_D∞h_<br>_Kh_|_D_3_d_<br>_Kh_<br>_D_3_d_|_Kh_<br>_D_3_d_<br>_Kh_|_D_3_d_<br>_Kh_|
|_D_2_h_<br>_Kh_|_Kh_<br>_D_2_h_<br>_Kh_|_D_2_h_<br>_Kh_<br>_D_2_h_|_Kh_<br>_D_2_h_<br>_Kh_|_D_2_h_<br>_Kh_|
|_Td_<br>_Kh_|_Kh_<br>_Kh_<br>_Td_|_Oh_<br>_Kh_<br>_Oh_|_Td_<br>_Oh_<br>_Td_|_Oh_<br>_Td_|
|_C_3_h_<br>_Kh_|_Kh_<br>_D∞h_<br>_C_3_h_|_D∞h_<br>_C_3_h_<br>_C_6_h_|_C_3_h_<br>_C_6_h_<br>_C_3_h_|_C_6_h_<br>_C_3_h_|


47