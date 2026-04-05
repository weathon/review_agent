# FROM DIVERGENCE TO NORMALIZED SIMILARITY: A SYMMETRIC AND SCALABLE TOPOLOGICAL TOOLKIT FOR REPRESENTATION ANALYSIS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Representation Topology Divergence (RTD) offers a powerful lens for analyzing
topological differences in neural network representations. However, its asymmetry and lack of a normalized scale limit its interpretability and direct comparability
across different models. Our work addresses these limitations on two fronts. First,
we complete the theoretical framework of RTD by introducing Symmetric Representation Topology Divergence (SRTD) and its lightweight variant, SRTD-lite.
We prove their mathematical properties, demonstrating that they provide a more
efficient, comprehensive, and interpretable divergence measure which matches the
top performance of existing RTD-based methods in optimization tasks. Second, to
overcome the inherent scaling issues of divergence measures, we propose Normalized Topological Similarity (NTS), a novel, normalized similarity score robust to
representation scale and size. NTS captures the hierarchical clustering structure of
representations by comparing their topological merge orders. We demonstrate that
NTS can reliably identify inter-layer similarities and, when analyzing representations of Large Language Models (LLMs), provides a more discriminative score
than Centered Kernel Alignment (CKA), offering a clearer view of inter-model
relationships.


1 INTRODUCTION


Understanding the internal representations of neural networks is a central challenge in deep learning,
crucial for interpreting their behavior and improving their design. Analyzing the similarity structure
of these representations has emerged as a key field for deciphering model behavior (Kriegeskorte
et al., 2008). Early research primarily relied on Canonical Correlation Analysis (CCA) and its
variants, such as SVCCA (Raghu et al., 2017) and PWCCA (Morcos et al., 2018). However, these
methods were often criticized for being too loose, as they remain invariant under any invertible
linear transformation. To address this, Centered Kernel Alignment (CKA) (Kornblith et al., 2019)
was proposed and has since become the de facto standard (Khrulkov & Oseledets, 2018; Raghu
et al., 2019; Wu et al., 2020; Zhang et al., 2024). By quantifying similarity through centered Gram
matrices, CKA provides a normalized score that facilitates comparison across diverse experimental
settings and is robust to fundamental geometric transformations.


While geometric analysis dominates the field, Topological Data Analysis (TDA) offers a complementary perspective by probing the intrinsic shape of data. Using tools like persistent homology
(Barannikov, 1994; Carlsson et al., 2004), this approach examines how the fundamental topological structure of the data—from simple clusters to complex loops and voids—is formed and evolves
across a continuous range of scales. This focus on properties that are invariant to non-linear deformations (such as stretching and bending) allows TDA to capture a different, often complementary,
notion of structural similarity that is overlooked by geometry-centric measures.


Existing topological methods, however, face distinct limitations regarding their applicability. Methods such as Geometry Score (Khrulkov & Oseledets, 2018) and IMD (Tsitsulin et al., 2019) are
highly general and do not require a one-to-one correspondence between representations. While
flexible, they fail to leverage the valuable pairing information inherent in comparing neural network layers, often resulting in lower discriminative power. Conversely, approaches that do analyze


1


distributional topology often strictly require the point clouds to reside in the same ambient space
(Kynk¨a¨anniemi et al., 2019; Barannikov et al., 2021b), severely limiting their scope.


A significant breakthrough in bridging this gap is Representation Topology Divergence (RTD)
(Barannikov et al., 2021a) and its scalable variant, RTD-lite (Tulchinskii et al., 2025). These methods successfully utilize the one-to-one correspondence between data points without requiring them
to share the same ambient space, making them powerful tools for representation analysis and optimization (Trofimov et al., 2023).


Despite these advancements, the RTD framework suffers from two critical limitations that hinder
its broader adoption. First, its theoretical underpinnings remain incomplete. The standard symmetric measure is a brute-force average of two directional values, _RTD_ ( _w,_ _w_ ˜) and _RTD_ ( ˜ _w, w_ ),
that can differ dramatically (Table 2f) without a clear theoretical explanation.Another theoretical
ambiguity comes from it’s dual variant, Max-RTD, mentioned by Trofimov et al. (2023) to enrich
gradient information, but whose theoretical role and relationship to the original RTD were never
fully investigated.


Second, and more critically, unlike CKA, topological divergence methods are not normalized:the
output of RTD and RTD-lite can be any positive number, heavily dependent on the number of sample
points and the intrinsic scale of distances. This lack of a normalized scale makes cross-scenario
comparison difficult and interpretability elusive. For instance, in layer-wise analysis, unnormalized
divergence measures often fail to reveal the graded similarity patterns between layers (Figure 4a)—a
task that CKA consistently accomplishes due to its normalization.


To address these issues, we propose a comprehensive topological toolkit with the following contributions:


    - We complete the theoretical framework of RTD by introducing **Symmetric** **Representa-**
**tion** **Topology** **Divergence** **(SRTD)** and its lightweight variant, **SRTD-lite** . We reveal
the mathematical relationships between RTD, Max-RTD, and SRTD, proving that SRTD
provides a more comprehensive and computationally efficient divergence measure that
matches the top performance of this class of methods in optimization tasks.


    - We introduce **Normalized Topological Similarity (NTS)**, a novel, scale-invariant, and normalized similarity measure. Unlike divergence-based methods, NTS captures hierarchical
clustering features and can robustly reveal graded inter-layer similarity patterns that are
often missed by RTD, combining the interpretability of CKA with the structural sensitivity
of TDA.


2 PRELIMINARIES: PERSISTENT HOMOLOGY AND REPRESENTATION
TOPOLOGY DIVERGENCE


We consider two point clouds, _P_ and _P_ _[′]_, of the same size with a one-to-one correspondence. Their
respective pairwise distance matrices are denoted by _w_ and _w_ ˜. We define min( _w,_ _w_ ˜) and max( _w,_ _w_ ˜)
as the element-wise minimum and maximum of the two matrices, respectively.


To understand the topological structure of these point clouds, we employ persistent homology. The
process can be intuitively understood as follows: for a given point cloud _P_ with distance matrix _w_,
we construct a sequence of simplicial complexes, known as the Vietoris-Rips filtration (Hausmann,
1995), indexed by a proximity parameter _α_ . As _α_ increases from zero, edges are added between
points with distance less than or equal to _α_ . When a set of _n_ points are all mutually connected, the
( _n−_ 1)-simplex they span is filled in (e.g., three points form a filled triangle). This growing complex
is denoted as _Rα_ ( _G_ _[w]_ ).


During this filtration process, topological features—such as connected components ( _H_ 0), cycles
( _H_ 1), and voids ( _H_ 2)—appear and disappear. We track the lifespan of each feature by recording
its birth and death values as an interval [ _b, d_ ] (Barannikov, 1994). The collection of these intervals
is known as **barcodes** (Carlsson et al., 2004), which serves as a topological signature of the point
cloud. The computation of persistent homology operates directly on the distance matrix.


2


**RTD** A set of barcodes characterizes one point cloud. To compare two, Representation Topology
Divergence (RTD) (Barannikov et al., 2021a) introduced an auxiliary matrix _Mmin_ (Matrix 1b) constructed from _w_, _w_ ˜, and min( _w,_ _w_ ˜). The resulting barcode captures the differences in the evolution
of topological features between an individual point cloud and the composite structure formed by
their union, which is derived from the min( _w,_ _w_ ˜) matrix. The length of a barcode interval in this
context quantifies the discrepancy between when a feature forms in _w_ (or _w_ ˜) versus when it forms
in min( _w,_ _w_ ˜).


We define _RTD_ ( _w,_ _w_ ˜) as the sum of the lengths of all barcodes computed from _Mmin_ (Matrix 1b).
By swapping the roles of _w_ and _w_ ˜, we can similarly compute _RTD_ ( ˜ _w, w_ ). To ensure symmetry, the final divergence is typically defined as their average: _RTD_ ( _P, P_ _[′]_ ) = _[RT D]_ [(] _[w,]_ _w_ [ ˜] )+2 _RT D_ ( ˜ _w,w_ )
Subsequently, Trofimov et al. (2023) noted that a dual variant, which we term Max-RTD, can be
defined by using an auxiliary matrix _Mmax_ (Matrix 1c) based on _w_, _w_ ˜, and max( _w,_ _w_ ˜). However,
the properties of this variant were not deeply investigated in their work. The symmetric versions of
Max-RTD are defined analogously by averaging the two directional computations.


**RTD-lite** To address the computational cost of higher-dimensional homology, RTD-lite (Tulchinskii et al., 2025) was introduced as a lightweight variant focusing solely on 0-dimensional features—the merging of connected components. The key insight is that its divergence score can be
calculated efficiently, as it is exactly the difference between the weights of the Minimum Spanning Trees (MSTs) of the respective distance matrices. For instance, the directional divergence
_RTD_ ~~_l_~~ _ite_ ( _w,_ _w_ ˜) is given by _MST_ ( _w_ ) _−_ _MST_ (min( _w,_ _w_ ˜)), and the final measure is symmetrized
by averaging the two directional computations. This connection to MSTs provides a computationally
feasible tool for large-scale representation analysis.


**Notation for Vietoris-Rips Complexes** To streamline the following sections, we establish notation for the key Vietoris-Rips complexes used in our analysis. Recall that these are constructed
based on a proximity parameter, _α_, which acts as a distance threshold for connecting points. For
any given threshold _α_, we denote the complexes generated from the distance matrices _w_ and _w_ ˜ as
_Rα_ ( _G_ _[w]_ ) and _Rα_ ( _Gw_ [˜] ), respectively. The complexes derived from the element-wise minimum and
maximum matrices have a crucial relationship to these: at the same scale _α_, _Rα_ ( _G_ [min(] _[w,]_ _w_ [ ˜] )) is the
union of the individual complexes ( _Rα_ ( _G_ _[w]_ ) _∪_ _Rα_ ( _Gw_ [˜] )), while _Rα_ ( _G_ max( _w,_ ˜ _w_ )) is their intersection
( _Rα_ ( _G_ _[w]_ ) _∩_ _Rα_ ( _Gw_ [˜] )).


Figure 1: The three key auxiliary matrices. For any matrix _M_, _M_ [+] is obtained by replacing its
upper triangular part with infinity.


3 SYMMETRIC REPRESENTATION TOPOLOGY DIVERGENCE (SRTD)


In practice, we observe a complementary phenomenon between RTD and Max-RTD (shown in
Table 2f). When _RTD_ ( _w,_ _w_ ˜) _>_ _RTD_ ( ˜ _w, w_ ), we consistently find that _Max-RTD_ ( _w,_ _w_ ˜) _<_
_Max-RTD_ ( ˜ _w, w_ ). This suggests that the topological structural differences between _Rα_ ( _G_ _[w]_ ) _∪_
_Rα_ ( _Gw_ [˜] ) and _Rα_ ( _Gw_ ) _∩_ _Rα_ ( _Gw_ ˜) seem to be the core reason for the asymmetry in RTD. Therefore,
we propose to directly measure this difference as the Symmetric Representation Topology Divergence (SRTD) of _P_ and _P_ _[′]_ .

**Definition** **3.1** (SRTD) **.** For two point clouds _P_ and _P_ _[′]_ with a one-to-one correspondence, the
distance matrix of their auxiliary graph _G_ [ˆ] _sym_ _[′]_ [is] _[M][sym]_ [(Matrix] [1a).] [The] [sum] [of] [the] [lengths] [of] [its]
persistent homology barcodes is defined as _SRTD_ ( _P, P_ _[′]_ ) (see Algorithm 3). Its chain complex is
homotopy equivalent to the mapping cone of the inclusion map _f_ _[′]_ : _C∗_ ( _Rα_ ( _G_ _[w]_ ) _∩_ _Rα_ ( _Gw_ [˜] )) _→_
_C∗_ ( _Rα_ ( _G_ _[w]_ ) _∪_ _Rα_ ( _Gw_ [˜] )).


3


 max( _w,_ _w_ ˜) (max( _w,_ _w_ ˜) [+] ) _[T]_ 0

max( _w,_ _w_ ˜) [+] min( _w,_ _w_ ˜) _∞_
0 _∞_ 0


(a) _M_ sym


(c) _M_ max








 _w_ ( _w_ [+] ) _[T]_ 0

 _w_ [+] min( _w,_ _w_ ˜) _∞_
0 _∞_ 0


(b) _M_ min








 max( _w,_ _w_ ˜) (max( _w,_ _w_ ˜) [+] ) _[T]_ 0

max( _w,_ _w_ ˜) [+] _w_ _∞_
0 _∞_ 0











The logic behind RTD-lite—simplifying topological divergence to a calculation on Minimum Spanning Trees (MSTs)—can be extended across the entire RTD framework. This allows us to formally
define **Max-RTD-lite**, the natural dual to RTD-lite, which compares an individual MST to the MST
of the intersection structure (derived from max( _w,_ _w_ ˜)). With this complete lightweight family in
place, we introduce our proposed symmetric version, **SRTD-lite**, as the most direct and fundamental measure. Since the full SRTD compares the topologies of the composite union _Rα_ ( _G_ [min(] _[w,]_ _w_ [ ˜] ))
and intersection _Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] )) structures, SRTD-lite quantifies the divergence between them by
simply comparing the weights of their respective MSTs.


**Definition** **3.2** (SRTD-lite) **.** By comparing the minimum spanning trees of min( _w,_ _w_ ˜) and
max( _w,_ _w_ ˜) through Algorithm 4, we can obtain a series of barcodes. We define the sum of the
lengths of these barcodes as _SRTD-lite_ ( _w,_ _w_ ˜).


3.1 MATHEMATICAL PROPERTIES


SRTD, RTD, and Max-RTD satisfy some elegant mathematical properties. The mapping cones
corresponding to their auxiliary graphs fit into the following long exact sequence:


_· · · →_ _Hn_ ( _Rα_ ( _G_ _[w]_ ) _, Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] ))) _−→γn_ _Hn_ ( _Rα_ ( _G_ [min(] _[w,]_ _w_ [ ˜] )) _, Rα_ ( _G_ max( _w,_ ˜ _w_ )))

_−−→βn_ _Hn_ ( _Rα_ ( _G_ [min(] _[w,]_ _w_ [ ˜] )) _, Rα_ ( _Gw_ )) _−→δn_ _Hn−_ 1( _Rα_ ( _G_ _[w]_ ) _, Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] ))) _−−−→· · ·γn−_ 1


**Theorem 3.3.** _For any dimension i, point clouds P, P_ _[′]_ _and distance matrices w,_ _w_ ˜ _, the three diver-_
_gences satisfy the following relationship:_


                     - _∞_
_Max-RTDi_ ( _w,_ _w_ ˜) + _RTDi_ ( _w,_ _w_ ˜) _−_ _SRTDi_ ( _w,_ _w_ ˜) = (dim(ker( _γi_ )) + dim(ker( _γi−_ 1))) _dα_

0


By swapping the positions of _w_ and _w_ ˜ in Theorem 3.3, we obtain a similar equality. We denote
_RTDi_ ( _w,_ _w_ ˜) + _Max_ - _RTDi_ ( _w,_ _w_ ˜) as _minmax_ ( _w,_ _w_ ˜),and _RTDi_ ( ˜ _w, w_ ) + _Max_ - _RTDi_ ( ˜ _w, w_ )as
_minmax_ ( ˜ _w, w_ ). Both are strictly greater than SRTD, but in our experiments, we find this gap to be
very small, as shown in the Table 2e.


The introduction of SRTD provides a more mathematically elegant framework for understanding the
RTD family. Within this framework, the asymmetric measures _minmax_ ( _w,_ _w_ ˜) and _minmax_ ( ˜ _w, w_ )
can be decomposed into a large, shared symmetric component, _SRTD_ ( _w,_ _w_ ˜), and smaller, ’private’
components. These private components correspond to topological features unique to the individual
filtrations of _G_ _[w]_ or _Gw_ [˜] relative to the bounding filtrations of _G_ min( _w,_ ˜ _w_ ) and _G_ max( _w,_ ˜ _w_ ). This decomposition reveals that the asymmetry in the original RTD arises from these small, private feature sets,
making the source of the divergence interpretable. The relationship becomes even more direct and
elegant in the lite version:


**Corollary 3.4.** _Max-RTD-lite_ ( _w,_ _w_ ˜) + _RTD-lite_ ( _w,_ _w_ ˜) = _SRTD-lite_ ( _w,_ _w_ ˜)

**Corollary 3.5.** _Max-RTD-lite_ ( _P, P_ _[′]_ ) _≥_ _SRTD-lite_ ( _P, P_ _[′]_ ) _≥_ _RTD-lite_ ( _P, P_ _[′]_ )


Together, Theorem 3.3 and Corollary 3.4, 3.5 provide a clear theoretical basis for a consistent pattern
observed in our experiments: when plotting the divergence curves for either the full or lite families,
the Max-RTD curve is always highest, the RTD curve is lowest, and the SRTD curve lies in between
(as shown in Figure 2b). For the lite versions, Corollary 3.5 proves this hierarchical ordering is
strict, which explains why the SRTD _-_ lite curve appears perfectly centered between the other two.
While the relationship for the full RTD family is more complex, this structure holds empirically,
positioning SRTD as a balanced, median measure of topological divergence.


4 NORMALIZED TOPOLOGICAL SIMILARITY (NTS)


4.1 MOTIVATION: THE LIMITATIONS OF DIVERGENCE-BASED ANALYSIS


While SRTD theoretically completes the topological divergence framework, the reliance on summing barcode lengths creates two practical limitations for general similarity analysis. First, as previously discussed, the unnormalized scores are inherently scale-dependent and difficult to interpret


4


across different contexts. Second, and more critically, the total divergence can be dominated by a
few “ultra-long” barcodes (Figure 19a) corresponding to large-scale structural differences. This sensitivity to a handful of major dissimilarities can mask a high degree of similarity in finer structural
details, making the measure brittle.


These limitations underscore the need for a fundamentally different approach: a normalized, scaleinvariant similarity measure designed to robustly capture hierarchical clustering structures.


4.2 METHOD: CAPTURING MERGE-ORDER SIMILARITY


Instead of comparing the _magnitudes_ of topological features, we propose to compare their relative
_order_ of formation. The sequence of merge events in 0-dimensional persistent homology provides
a scale-invariant signature of a point cloud’s hierarchical clustering structure. To robustly compare
such sequences, we employ Spearman’s rank correlation coefficient ( _ρ_ ),which is inherently normalized to [ _−_ 1 _,_ 1] and is robust to outliers and monotonic scaling (Spearman, 1961).


The merge sequence of connected components is perfectly captured by the Minimum Spanning
Tree (MST), which forms the backbone of the 0-dimensional filtration. Our method, Normalized
Topological Similarity (NTS), leverages this connection. The core idea is to first establish a common
basis for comparison—the set of core pairs—by taking the union of edges from the MSTs of both
point clouds. For every pair in this common set, we extract a corresponding numerical value from
each point cloud’s structure. This process creates two parallel vectors, and the NTS score is their
Spearman’s rank correlation.


We define two variants based on the values extracted:


    - **NTS-M (Merge-time** **based):** This theoretically-grounded variant compares the ranks of
the merge times. The merge time of a pair of points is the threshold at which they become
connected in the filtration, formally defined by the maximum edge weight on the path
between them in their MST.

    - **NTS-E** **(Edge-distance** **based):** This practical variant directly compares the ranks of the
original pairwise distances for the ‘core pairs’. It is computationally simpler and often
more sensitive in practice, as it retains more of the original metric information.


4.3 FORMAL DEFINITION AND PROPERTIES


The procedures for calculating NTS-M and NTS-E are formally defined in Algorithm 1 and 2.


The NTS framework satisfies the following key properties, which highlight the stricter condition
imposed by NTS-E.
**Theorem 4.1.** _NTS-M_ ( _P, P_ _[′]_ ) = 1 _if and only if the rank order of merge times for all core pairs is_
_identical for both point clouds._
**Theorem 4.2.** _If NTS-E_ ( _P, P_ _[′]_ ) = 1 _, then the rank order of merge times for all core pairs is also_
_identical (i.e., NTS-M_ ( _P, P_ _[′]_ ) = 1 _)._ _The converse is not necessarily true._


NTS-E provides a stricter condition by comparing underlying distance ranks—making it more sensitive in practice—while NTS-M compares the final merge-time order to capture a more fundamental
notion of structural similarity.


5


**Algorithm 1:** NTS-M (Merge-time based)

**Input:** Pairwise distance matrices _w,_ _w_ ˜
**Output:** NTS-M score

**1** _Ew_ _←_ Edge set of MST( _w_ )

**2** _E_ ˜ _w_ _[←]_ [Edge set of MST(] _[w]_ [ ˜][)]

**3** _Ecore_ _←_ _Ew ∪_ _E_ ˜ _w_

**4** _Vmerge_ _←_ (MergeTime( _e, w_ )) _e∈Ecore_

**5** _V_ [˜] _merge_ _←_ (MergeTime( _e,_ _w_ ˜)) _e∈Ecore_

**6** **return** Spearman’s _ρ_ ( _Vmerge,_ _V_ [˜] _merge_ )


**Algorithm 2:** NTS-E (Edge-distance based)

**Input:** Pairwise distance matrices _w,_ _w_ ˜
**Output:** NTS-E score

**1** _Ew_ _←_ Edge set of MST( _w_ )

**2** _E_ ˜ _w_ _[←]_ [Edge set of MST(][ ˜] _[w]_ [)]

**3** _Ecore_ _←_ _Ew ∪_ _E_ ˜ _w_

**4** _Vdist_ _←_ ( _wij_ )( _i,j_ ) _∈Ecore_

**5** _V_ [˜] _dist_ _←_ ( ˜ _wij_ )( _i,j_ ) _∈Ecore_

**6** **return** Spearman’s _ρ_ ( _Vdist,_ _V_ [˜] _dist_ )


5 EXPERIMENTS


5.1 ANALYSIS OF HIERARCHICAL CLUSTERING STRUCTURES


We begin our experimental validation on two controlled tasks designed to test each method’s reliability and sensitivity in capturing hierarchical clustering structures.


**Clusters** **Experiment.** We test sensitivity to increasing structural dissimilarity by comparing a
single cluster of 300 2D Gaussian points against variants where the points are partitioned into
_k_ = 2 _, . . .,_ 12 clusters arranged on a circle. The results reveal a clear performance divide: our
proposed NTS and SRTD families correctly capture the expected trend of increasing dissimilarity. In contrast, CKA is largely insensitive to these structural changes, while RTD-lite produces
an anomalous, inverted trend, confirming that the max( _w,_ _w_ ˜) component is essential for a robust
divergence measure.


(a) RTD Family (b) RTD ~~l~~ ite Family (c) NTS (d) CKA


Figure 2: Analysis of the RTD framework on the synthetic Clusters dataset. (e) shows the small
theoretical difference between SRTD and the symmetrized RTD/Max-RTD combination, where
_E_ 1 = (RTD( _w,_ _w_ ˜) + Max-RTD( _w,_ _w_ ˜) _−_ SRTD) _/_ 2 and _E_ 2 is defined analogously by swapping _w_ and _w_ ˜, _percentage_ 1 = (RTD( _w,_ _w_ ˜) + Max-RTD( _w,_ _w_ ˜) _−_ SRTD) _/_ SRTD. (f) illustrates the strong asymmetry and complementarity between RTD and Max-RTD, Min-Asym =
_RTD_ ( _w,_ _w_ ˜) _−_ _RTD_ ( ˜ _w, w_ ) _,_ Max-Asym = _Max-RTD_ ( _w,_ _w_ ˜) _−_ _Max-RTD_ ( ˜ _w, w_ )


**UMAP** **Embeddings** **Experiment.** We test sensitivity to structural changes by generating a sequence of 2D UMAP embeddings (Damrich & Hamprecht, 2021) from the MNIST dataset (LeCun
et al., 2002), varying the n ~~n~~ eighbors parameter to control the trade-off between local and global
structure. Pairwise comparisons of these embeddings (Figure 3) demonstrate that our proposed
methods, NTS and SRTD-lite, track these changes with a smooth, monotonic response. In contrast,
the CKA baseline fails to capture this gradual evolution, highlighting the superior sensitivity of our
topological measures.


6


(e) Theoretical Difference from SRTD


**Clusters** _E_ 1( _Percentage_ 1) _E_ 2( _Percentage_ 2)


2 0.357 (3.16%) 0.000 (0.00%)
3 0.493 (3.32%) 0.013 (0.09%)
4 0.441 (2.47%) 0.061 (0.34%)
5 0.451 (2.26%) 0.039 (0.20%)
6 0.347 (1.57%) 0.060 (0.27%)
10 0.263 (0.95%) 0.043 (0.15%)
12 0.226 (0.76%) 0.046 (0.15%)


(f) Asymmetry of RTD vs Max-RTD


Min-Asym Max-Asym


13.0976 -12.3839
11.2554 -10.2954
10.8131 -10.0535
10.3320 -9.5084
9.4315 -8.8572
8.3074 -7.8674
7.6888 -7.3296


(a) UMAP n ~~n~~ eighbors=(10, 50, 200) (b) UMAP experiment heatmap


Figure 3: UMAP experiment


5.2 EFFICIENCY AS AN OPTIMIZATION LOSS


We evaluate the practical utility of our divergence measures as loss terms for training an autoencoder,
a task for which they are naturally suited. In this experiment, autoencoder is trained to reduce the
dimensionality of the F-MNIST and COIL-20 dataset to 16 (Xiao et al., 2017; Nene et al., 1996). It is
crucial to note this is an **intra-family comparison**, designed to demonstrate that our proposed SRTD
offers the best trade-off between performance and efficiency within the RTD class of methods. The
results confirm that SRTD and SRTD-lite achieves top-tier performance on quality metrics while
being faster than its predecessors. (Full results are provided in Appendix G).


5.3 ANALYZING STRUCTURAL CONSISTENCY AND FUNCTIONAL HIERARCHY


To rigorously test our measures in a practical setting, we analyze the structural consistency of representations learned by an 8-layer TinyCNN (see AppendixE). Our experimental design, including
the network architecture and training procedure on CIFAR-10 (Krizhevsky et al., 2009), is adapted
from the original CKA study (Kornblith et al., 2019; Springenberg et al., 2014) . For the analysis,
we use the representations of 5,000 images sampled from the test set. We trained ten instances of
this network from scratch with different random seeds [1] .


This setup allows us to validate a key distinction observed in related work (Tulchinskii et al., 2025),
which found that while topological divergence measures like RTD and RTD-lite can identify corresponding layers, they, unlike CKA, fail to capture the robust graded similarity patterns between
adjacent and nearby layers. The heatmaps in Figure 4, showing the average results over all 45 unique
model pairs, confirm this finding and reveal three key insights:


    - **Layer Identification:** All methods are highly effective at identifying corresponding convolutional layers, achieving over 94% accuracy.


    - **Graded** **Patterns:** NTS and CKA both reveal a clear, graded similarity pattern across
convolutional layers, an interpretable landscape that RTD-lite and RTD families fail to
produce.


    - **Functional** **Shift** **Detection:** Crucially, only the topological measures (NTS and SRTDlite) detect the sharp structural break at the final pooling layer. This identifies a fundamental
functional shift from feature extraction to global aggregation that CKA misses.


These results demonstrate that NTS uniquely combines the strengths of both approaches: it provides an interpretable, graded similarity landscape akin to CKA, while also retaining the topological
sensitivity needed to identify fundamental shifts in the network’s functional hierarchy.


5.4 ANALYSIS OF LARGE LANGUAGE MODEL REPRESENTATIONS


We conclude our experimental validation by analyzing the complex representations of Large Language Models (LLMs). Our methodology is closely adapted from REEF (Zhang et al., 2024), a
recent study that established a robust protocol for fingerprinting and comparing LLM representations. REEF identified that certain datasets are particularly effective at eliciting discriminative


1We select CKA as the primary baseline due to its widespread adoption as a robust, normalized similarity
measure. Other methods such as SVCCA (Raghu et al., 2017) are omitted as they have been shown to be less
effective for this type of layer analysis in prior studies (Kornblith et al., 2019; Barannikov et al., 2021a).


7


(a) CKA (98.89%) (b) NTS-E (97.22%) (c) NTS-M (94.72%) (d) SRTD-lite (98.33%)


Figure 4: Average layer-wise comparison over 45 pairs of trained TinyCNNs. NTS (b, c) provides
the most comprehensive view, matching CKA’s (a) graded pattern while also sharing the topological
methods’ (d) unique sensitivity to the functional shift at the final pooling layer, a distinction CKA
misses.


features that highlight inter-model differences. Following their findings, we conduct our analysis
on two such datasets: TruthfulQA (Lin et al., 2021) and ToxiGen (Hartvigsen et al., 2022). For
each dataset, we adopt the REEF protocol of extracting the last-token representation from every
Transformer layer across 1,000 randomly sampled QA pairs.


**Identifying** **Intra-Model** **Hierarchical** **Patterns.** Our first goal is to evaluate intra-model layer
similarity. The resulting heatmaps visualize this, with both the x- and y-axes representing every
Transformer layer of a given model, from first to last. An ideal measure should satisfy two criteria:
(1) the layer-wise similarity map for a single model should be structurally informative, revealing
distinct processing stages, and (2) this structural pattern should be consistent across models from
the same family.


Our analysis, summarized in Figure 5, shows a stark contrast in reliability. NTS successfully identifies consistent, hierarchical fingerprints for all tested model families (Qwen, InternLM, Baichuan,
and Llama). CKA, however, proves unreliable, meeting these requirements only for the InternLM family. For other families, CKA’s heatmaps either degenerate into uninformative saturated
blocks (e.g., Llama) or fail to show consistency after post-training refinements like distillation and
instruction-tuning (e.g., Qwen and Baichuan). In all these cases where CKA fails, NTS preserves the
underlying family-specific pattern, offering a more robust view of an LLM’s functional hierarchy.


**Inter-Model** **Similarity** **Analysis** Finally, we compare the ability of NTS and CKA to map the
relationships between different LLM families. For this analysis, we extract the last-token representation from the 6th Transformer layer of each model, as this empirically yielded the most discriminative results. Furthermore, we recommend applying Z-score normalization across the feature dimension of representations before computing NTS to mitigate variance in individual activations. Ablation studies for both layer selection and the effect of normalization can be found in Appendix K.2.


Following the methodology of REEF (Zhang et al., 2024), we present the results from the TruthfulQA dataset, using representations from 1000 QA pairs, in Figure 6. This visualization reveals a
critical weakness in CKA’s analysis. While both measures often assign high similarity scores between different model families, CKA exhibits severe **score** **saturation** . As seen in Figure 6a, its
scores for most non-Llama model pairs are pinned near the maximum (often _>_ 0 _._ 8), effectively
erasing the distinctions between families like Qwen, Mistral, and InternLM. In contrast, while NTS
scores in these cases can also be high, they are significantly less saturated and better distributed, thus
providing a more discriminative and nuanced view of the model landscape.


Beyond this quantitative issue of score saturation, CKA also makes a critical, counter-intuitive error
regarding DeepSeek-R1-Ds (Guo et al., 2025), which is distilled from qwen-2.5-math-7b
(Yang et al., 2024). This error manifests as a very low similarity score between the model and its
parent Qwen2.5 family (Team, 2024), a result that contradicts the known lineage.


NTS-E, in stark contrast, provides a more credible and discriminative map of the model space (Figure 6b). It correctly identifies the high similarity between DeepSeek-R1-Ds and its parent model
family. This suggests that NTS, by focusing on topological structure rather than pure geometry, is


8


TruthfulQA Dataset


ToxiGen Dataset


Figure 5: Intra-model layer similarity for LLM families on the TruthfulQA (top half) and ToxiGen
(bottom half) datasets. NTS (top row of each pair) consistently reveals structured hierarchical patterns. In contrast, CKA (bottom row of each pair) often produces saturated or inconsistent heatmaps,
failing on most families except InternLM.


less prone to the saturation and anomalous errors that can affect CKA, offering a more trustworthy
tool for analyzing the complex LLM ecosystem.


(a) CKA Inter-Model Similarity (b) NTS-E Inter-Model Similarity


Figure 6: Inter-model similarity maps for 17 LLMs


6 COMPUTATIONAL EFFICIENCY AND SCALABILITY


Our proposed toolkit is designed for both scalability and analytical power. A formal complexity
analysis shows that while the full SRTD is computationally intensive, the core components of our
framework are highly efficient. Both SRTD-lite and NTS-E operate in _O_ ( _n_ [2] ( _α_ ( _n_ ) + _d_ )) time,
primarily dominated by the pairwise distance calculation and the Minimum Spanning Tree (MST)
construction.


9


To empirically validate this scalability, we conducted a runtime benchmark using representations
from a TinyCNN trained on CIFAR-10. We varied the sample size _N_ from 5,000 to 30,000 and
measured the end-to-end execution time. The results unequivocally (figure 7) show that NTS-E
exhibits the best scalability, followed by SRTD-lite, with RTD-lite being the slowest due to its triple
MST calculation.


This significant efficiency gain in NTS-E stems
from two key factors:


1. **No** **Normalization** **Required:** Being
a rank-based measure, NTS-E operates
directly on raw distance matrices, bypassing the costly quantile calculation
and matrix division required by RTD
and SRTD.


In summary, we introduce a complementary topological toolkit. These methods offer a powerful
choice for representation analysis. While NTS is ideal for obtaining a single, stable similarity score,
SRTD-lite offers in-depth diagnostic (Table 5) and can serve as an effective loss term. A limitation
of our work is that NTS, in its current form, is an analysis-only measure. Its non-differentiable
nature prevents its use in direct model optimization. Therefore, a crucial avenue for future research
is to develop a differentiable formulation of NTS, enabling it to guide representation learning.


10


2. **Minimal** **Memory** **Footprint:** NTSE avoids constructing dense auxiliary
matrices (e.g., min( _w,_ _w_ ˜)), reducing
peak memory usage from _O_ (3 _N_ [2] ) to
_O_ (2 _N_ [2] ), making it the most memoryefficient method.


7 CONCLUSION


Figure 7: Runtime comparison on CIFAR-10
representations with varying sample sizes.


REFERENCES


Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
Yu Han, Fei Huang, et al. Qwen technical report. _arXiv preprint arXiv:2309.16609_, 2023.


Serguei Barannikov. The framed morse complex and its invariants. _Advances in Soviet Mathematics_,
21:93–116, 1994.


Serguei Barannikov, Ilya Trofimov, Nikita Balabin, and Evgeny Burnaev. Representation topology divergence: A method for comparing neural network representations. _arXiv_ _preprint_
_arXiv:2201.00058_, 2021a.


Serguei Barannikov, Ilya Trofimov, Grigorii Sotnikov, Ekaterina Trimbach, Alexander Korotin,
Alexander Filippov, and Evgeny Burnaev. Manifold topology divergence: a framework for comparing data manifolds. _Advances in neural information processing systems_, 34:7294–7305, 2021b.


Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui
Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. _arXiv_ _preprint_ _arXiv:2403.17297_,
2024.


Gunnar Carlsson, Afra Zomorodian, Anne Collins, and Leonidas Guibas. Persistence barcodes for
shapes. In _Proceedings_ _of_ _the_ _2004_ _Eurographics/ACM_ _SIGGRAPH_ _symposium_ _on_ _Geometry_
_processing_, pp. 124–135, 2004.


Devendra Singh Chaplot. Albert q. jiang, alexandre sablayrolles, arthur mensch, chris bamford,
devendra singh chaplot, diego de las casas, florian bressand, gianna lengyel, guillaume lample,
lucile saulnier, l´elio renard lavaud, marie-anne lachaux, pierre stock, teven le scao, thibaut lavril,
thomas wang, timoth´ee lacroix, william el sayed. _arXiv preprint arXiv:2310.06825_, 3, 2023.


Fr´ed´eric Chazal and Bertrand Michel. An introduction to topological data analysis: fundamental
and practical aspects for data scientists. _Frontiers in artificial intelligence_, 4:667963, 2021.


Sebastian Damrich and Fred A Hamprecht. On umap’s true loss function. _Advances_ _in_ _Neural_
_Information Processing Systems_, 34:5798–5809, 2021.


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.


Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar.
Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. _arXiv preprint arXiv:2203.09509_, 2022.


Jean-Claude Hausmann. On the vietoris–rips complexes and a cohomology theory. In _Prospects_
_in_ _topology:_ _proceedings_ _of_ _a_ _conference_ _in_ _honor_ _of_ _William_ _Browder_, number 138, pp. 175.
Princeton University Press, 1995.


Valentin Khrulkov and Ivan Oseledets. Geometry score: A method for comparing generative adversarial networks. In _International_ _conference_ _on_ _machine_ _learning_, pp. 2621–2629. PMLR,
2018.


Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. In _International_ _conference_ _on_ _machine_ _learning_, pp. 3519–
3529. PMlR, 2019.


Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini. Representational similarity analysisconnecting the branches of systems neuroscience. _Frontiers in systems neuroscience_, 2:249, 2008.


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.


Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved
precision and recall metric for assessing generative models. _Advances_ _in_ _neural_ _information_
_processing systems_, 32, 2019.


11


Yann LeCun, L´eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to
document recognition. _Proceedings of the IEEE_, 86(11):2278–2324, 2002.


Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human
falsehoods. _arXiv preprint arXiv:2109.07958_, 2021.


Ari Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural
networks with canonical correlation. _Advances_ _in_ _neural_ _information_ _processing_ _systems_, 31,
2018.


Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained
language models. In _Proceedings of the 59th annual meeting of the association for computational_
_linguistics and the 11th international joint conference on natural language processing (volume 1:_
_long papers)_, pp. 5356–5371, 2021.


Sameer A Nene, Shree K Nayar, Hiroshi Murase, et al. Columbia object image library (coil-100).
Technical report, Technical report CUCS-006-96, 1996.


Aniruddh Raghu, Maithra Raghu, Samy Bengio, and Oriol Vinyals. Rapid learning or feature reuse?
towards understanding the effectiveness of maml. _arXiv preprint arXiv:1909.09157_, 2019.


Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector
canonical correlation analysis for deep learning dynamics and interpretability. _Advances in neural_
_information processing systems_, 30, 2017.


Charles Spearman. The proof and measurement of association between two things. 1961.


Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for
simplicity: The all convolutional net. _arXiv preprint arXiv:1412.6806_, 2014.


Qwen Team. Qwen2 technical report. _arXiv preprint arXiv:2407.10671_, 2024.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023.


Ilya Trofimov, Daniil Cherniavskii, Eduard Tulchinskii, Nikita Balabin, Evgeny Burnaev, and
Serguei Barannikov. Learning topology-preserving data representations. _arXiv_ _preprint_
_arXiv:2302.00136_, 2023.


Anton Tsitsulin, Marina Munkhoeva, Davide Mottin, Panagiotis Karras, Alex Bronstein, Ivan Oseledets, and Emmanuel M¨uller. The shape of data: Intrinsic distance for data distributions. _arXiv_
_preprint arXiv:1905.11141_, 2019.


Eduard Tulchinskii, Daria Voronkova, Ilya Trofimov, Evgeny Burnaev, and Serguei Barannikov.
Rtd-lite: Scalable topological analysis for comparing weighted graphs in learning tasks. _arXiv_
_preprint arXiv:2503.11910_, 2025.


Yingfan Wang, Haiyang Huang, Cynthia Rudin, and Yaron Shaposhnik. Understanding how dimension reduction tools work: an empirical approach to deciphering t-sne, umap, trimap, and pacmap
for data visualization. _Journal of Machine Learning Research_, 22(201):1–73, 2021.


John Wu, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass. Similarity analysis of contextual word representation models. In _Proceedings of the 58th Annual Meeting_
_of the Association for Computational Linguistics_, pp. 4638–4655, 2020.


Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _arXiv preprint arXiv:1708.07747_, 2017.


Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan,
Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. _arXiv_ _preprint_
_arXiv:2309.10305_, 2023.


12


An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical
expert model via self-improvement. _arXiv preprint arXiv:2409.12122_, 2024.


Jie Zhang, Dongrui Liu, Chen Qian, Linfeng Zhang, Yong Liu, Yu Qiao, and Jing Shao. Reef: Representation encoding fingerprints for large language models. _arXiv_ _preprint_ _arXiv:2410.14273_,
2024.


Simon Zhang, Mengbai Xiao, and Hao Wang. Gpu-accelerated computation of vietoris-rips persistence barcodes. _arXiv preprint arXiv:2003.07989_, 2020.


A USE OF LARGE LANGUAGE MODELS


During the preparation of this manuscript, the authors utilized large language models to improve the
clarity and readability of the text. The LLM was also used as a tool to assist with literature searches.


B REPRODUCIBILITY STATEMENT


We believe in open and reproducible research. To this end, we will release the complete source
code for this project, including experiment scripts and setup instructions, upon the acceptance of
this paper. We hope this will be a useful resource for the community.


C DEFINITION AND ALGORITHM


**Definition C.1** (Max-RTD) **.** For two point clouds _P_ and _P_ _[′]_ with a one-to-one correspondence, the
distance matrix of their auxiliary graph _G_ [ˆ] _max_ _[′]_ [is given by] _[ M][max]_ [(Matrix 1c).] [The sum of the lengths]
of the persistent homology barcodes of _G_ [ˆ] _max_ _[′]_ [is] [defined] [as] _[Max][-][RTD]_ [(] _[w,]_ _[w]_ [˜][)][.] [Its] [chain] [complex]
is homotopy equivalent to the mapping cone of the inclusion map _f_ _[′]_ : _C∗_ ( _Rα_ ( _G_ _[w]_ ) _∩_ _Rα_ ( _Gw_ [˜] )) _→_
_C∗_ ( _Rα_ ( _G_ _[w]_ )).


C.1 SRTD ALGORITHM


**Algorithm 3:** Symmetric Representation Topology Divergence (SRTD) Calculation

**Input:** Pairwise distance matrices _w,_ _w_ ˜
**Output:** A set of divergence scores _{SRTDi}i≥_ 0 for each dimension _i_

**1** _wnorm,_ _w_ ˜ _norm_ _←_ Normalize _w,_ _w_ ˜ by their 0.9 quantiles;

**2** _wmin_ _←_ min( _wnorm,_ _w_ ˜ _norm_ );

**3** _wmax_ _←_ max( _wnorm,_ _w_ ˜ _norm_ );

**4** Construct the symmetric auxiliary matrix _Msym_ using _wmin_ and _wmax_ (see Matrix 1a);

**5** **for** _each dimension of interest i ∈{_ 0 _,_ 1 _, . . . }_ **do**


**8** **end**

**9** **return** _{SRTDi}i≥_ 0;


**6** Compute barcodes: _Bi_ _←_ PersistentHomology( _msym, i_ );


**7** Compute divergence score: _SRTDi_ _←_ [�]


( _b,d_ ) _∈Bi_ [(] _[d][ −]_ _[b]_ [)][;]


13


C.2 SRTD LITE BARCODE ALGORITHM


**Algorithm 4:** Computation of SRTD-lite Barcode
**Input:** Weight matrices _D_ 1 _, D_ 2
**Output:** A multiset of intervals (the SRTD-L-Barcode)


Figure 8: Conceptual relationship between SRTD, RTD, and Max-RTD.


D PROOFS


D.1 STATEMENT IN DEFINITION


We first prove the following lemmas, they are stated in definition C.1 and definition 3.1,The
construction and proof for this part refer to Barannikov et al. (2021a).Let _A_ = _Rα_ ( _G_ _[w]_ ) and
_B_ = _Rα_ ( _Gw_ [˜] ):

**Lemma D.1.** _There exists a specially constructed auxiliary graph_ _G_ [ˆ] _max_ _[′]_ _[such that its chain complex]_
_is homotopy equivalent to the mapping cone Cone_ ( _f_ _[′]_ ) _, where f_ _[′]_ : _C∗_ ( _A ∩_ _B_ ) _→_ _C∗_ ( _A_ ) _is a chain_
_map induced by the inclusion._

_Rα_ ( _G_ [ˆ] _max_ _[′]_ [)] _[ ∼]_ _[Cone]_        - _Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] )) _→_ _Rα_ ( _Gw_ )�


14


**1** **procedure** SRTD-L-Barcode( _D_ 1 _, D_ 2)


**2** _D_ 1 _[′]_ _[, D]_ 2 _[′]_ _[←]_ [Normalize] _[ D]_ [1] _[, D]_ [2] [by their 0.9 quantiles;]


**3** _D_ min _←_ Element-wise minimum of _D_ 1 _[′]_ [and] _[ D]_ 2 _[′]_ [;]


**4** _D_ max _←_ Element-wise maximum of _D_ 1 _[′]_ [and] _[ D]_ 2 _[′]_ [;]


**5** _E_ min _←_ Sort(MST( _D_ min));


**6** _E_ max _←_ Sort(MST( _D_ max));

**7** _BarcodeSet ←_ [];

**8** _SubTree ←_ Empty graph with _N_ vertices;

**9** **foreach** _edge e_ = ( _u, v_ ) _with weight wbirth_ _in E_ min **do**

**10** **if** _u and v are not connected in SubTree_ **then**


**11** _TemporaryGraph ←_ copy( _SubTree_ );


**12** **foreach** _edge e_ _[′]_ = ( _u_ _[′]_ _, v_ _[′]_ ) _with weight wdeath in E_ max **do**


**13** Add _e_ _[′]_ to _TemporaryGraph_ ;


**14** **if** _u and v are connected in TemporaryGraph_ **then**

**15** Add ( _wbirth, wdeath_ ) to _BarcodeSet_ ;

**16** **break** ;

**17** **end**

**18** **end**

**19** Add _e_ to _SubTree_ ;

**20** **end**

**21** **end**

**22** **return** _BarcodeSet_ ;


**Lemma D.2.** _Similarly, there exists a specially constructed auxiliary graph_ _G_ [ˆ] _sym_ _[′]_ _[such that its chain]_
_complex is homotopy equivalent to the mapping cone Cone_ ( _f_ _[′]_ ) _, where f_ _[′]_ : _C∗_ ( _A∩B_ ) _→_ _C∗_ ( _A∪B_ )
_is a chain map induced by the inclusion._

_Rα_ ( _G_ [ˆ] _sym_ _[′]_ [)] _[ ∼]_ _[Cone]_       - _Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] )) _→_ _Rα_ ( _G_ min( _w,_ ˜ _w_ ))�


_Proof._ The mapping cone we are interested in is constructed from the direct sum of the following
chain complexes:
Cone( _f_ _[′]_ ) = _C∗_ ( _A ∩_ _B_ )[ _−_ 1] _⊕_ _C∗_ ( _A_ )


Following the construction from the RTD paper, we can propose two auxiliary graph schemes: The
vertex set of the auxiliary graph _G_ [ˆ] _max_ _[′]_ [is composed of the original vertices] _[ v]_ _i_ _[′]_ [, mirrored vertices] _[ v][i]_ [,]
and a special vertex _O_ . Its distance rules are defined as follows: _d_ _[′]_ _vivj_ [=] [max(] _[w][ij][,]_ _[w]_ [˜] _[ij]_ [)][,] _[d][′]_ _vi_ _[′]_ _[v]_ _j_ _[′]_ [=]
_wij_, _d_ _[′]_ _vivi_ _[′]_ [= 0][,] _[d]_ _Ov_ _[′]_ _i_ [= 0][,] _[ d]_ _Ov_ _[′]_ _i_ _[′]_ [= +] _[∞]_ [,] _[d]_ _v_ _[′]_ _ivj_ _[′]_ [= max(] _[w][ij][,]_ _[w]_ [˜] _[ij]_ [)]

The vertex set of the auxiliary graph _G_ [ˆ] _sym_ _[′]_ [is] [composed] [of] [twice] [the] [number] [of] [original] [vertices]
and _O_ . _d_ _[′]_ _vivj_ [=] [max(] _[w][ij][,]_ _[w]_ [˜] _[ij]_ [)][,] _[d][′]_ _vi_ _[′]_ _[v]_ _j_ _[′]_ [=] [min(] _[w][ij][,]_ _[w]_ [˜] _[ij]_ [)][,] _[d][′]_ _vivi_ _[′]_ [=] [0][,] _[d][′]_ _Ovi_ [=] [0][,] _[ d][′]_ _Ovi_ _[′]_ [=] [+] _[∞]_ [,] _[d][′]_ _vivj_ _[′]_ [=]
max( _wij,_ _w_ ˜ _ij_ )


For the auxiliary graph _Rα_ ( _G_ [ˆ] _max_ _[′]_ [)][, there are three types of simplices:]


    - _Ai_ 1 _. . . Aik_ _A_ _[′]_ _ik_ _[. . . A]_ _i_ _[′]_ _n_ [, where][ max(] _[w][A]_ _ir_ _[A]_ _is_ _[,]_ _[w]_ [˜] _[A]_ _ir_ _[A]_ _is_ [)] _[≤]_ _[α]_ [ for] _[ r]_ _[≤]_ _[k]_ [, and] _[ w][A]_ _ir_ _[A]_ _is_ _[≤]_ _[α]_
for _r, s ≥_ _k_ .


    - _Ai_ 1 _. . . Aik_ _A_ _[′]_ _ik_ +1 _[. . . A]_ _i_ _[′]_ _n_ [, where][ max(] _[w][A]_ _ir_ _[A]_ _is_ _[,]_ _[w]_ [˜] _[A]_ _ir_ _[A]_ _is_ [)] _[≤]_ _[α]_ [ for] _[ r]_ _[≤]_ _[k]_ [, and] _[ w][A]_ _ir_ _[A]_ _is_ _[≤]_
_α_ for _r, s ≥_ _k_ + 1.


    - _OAi_ 1 _Ai_ 2 _. . . Ain_, where max( _wAir Ais,_ _w_ ˜ _Air Ais_ ) _≤_ _α_ .


**Forward Map**
_ψ_ _[′]_ : Cone( _f_ _[′]_ ) _→_ _Rα_ ( _G_ [ˆ] _max_ _[′]_ [)]


    - For _c ∈_ _C∗_ ( _A ∩_ _B_ )[ _−_ 1] (of the form _Ai_ 1 _. . . Ain_ [ _−_ 1]):


For all other simplices:
_H_ (∆) = 0


Therefore, _ψ_ [˜] _[′]_ _◦_ _ψ_ _[′]_ = Id and _ψ_ _[′]_ _◦_ _ψ_ [˜] _[′]_ _−_ Id = _H∂_ _−_ _∂H_ . This proves D.1, and D.2 can be proven
similarly.


15


_ψ_ _[′]_ ( _c_ ) = _OAi_ 1 _. . . Ain_ +


_n_

- _Ai_ 1 _. . . Aik_ _A_ _[′]_ _ik_ _[. . . A]_ _i_ _[′]_ _n_

_k_ =1


    - For _a ∈_ _C∗_ ( _A_ ) (of the form _Ai_ 1 _. . . Ain_ ):
_ψ_ _[′]_ ( _a_ ) = _A_ _[′]_ _i_ 1 _[. . . A]_ _i_ _[′]_ _n_


**Backward Map**
_ψ_ ˜ _[′]_ : _Rα_ ( ˆ _Gmax_ _[′]_ [)] _[ →]_ [Cone][(] _[f][ ′]_ [)]


    - _ψ_ [˜] _[′]_ ( _OAi_ 1 _. . . Ain_ ) = _Ai_ 1 _. . . Ain_ [ _−_ 1]

    - _ψ_ [˜] _[′]_ ( _A_ _[′]_ _i_ 1 _[. . . A]_ _i_ _[′]_ _n_ [) =] _[ A][i]_ 1 _[. . . A][i]_ _n_


    - _ψ_ [˜] _[′]_ (∆) = 0 (for all other types of simplices ∆)


**Homotopy Operator H** For the second type of simplex:


_H_ : _Ai_ 1 _. . . Aik_ _A_ _[′]_ _ik_ +1 _[. . . A]_ _i_ _[′]_ _n_ _[→]_


_k_

- _Ai_ 1 _. . . Ail_ _A_ _[′]_ _il_ _[. . . A]_ _i_ _[′]_ _n_ _[,]_ [ 1] _[ ≤]_ _[k]_ _[≤]_ _[n]_

_l_ =1


D.2 PROOF OF THEOREM 3.3


Lets proof Theorem 3.3. To proof the theorem,we just need to proof the following theorem:

**Lemma** **D.3.** _For_ _any_ _dimension_ _i,_ _the_ _Betti_ _numbers_ _of_ _the_ _three_ _auxiliary_ _graphs_ _satisfy_ _the_ _fol-_
_lowing relation:_

_βi_ [min] ( _α_ ) + _βi_ [max] ( _α_ ) _−_ _βi_ _[sym]_ ( _α_ ) = dim( _ker_ ( _γi_ )) + dim( _ker_ ( _γi−_ 1))


_Proof._ We have the following inclusion of simplicial complexes:


_Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] )) _⊆_ _Rα_ ( _Gw_ ) _⊆_ _Rα_ ( _G_ min( _w,_ ˜ _w_ ))


This forms a triple of complexes, which gives rise to a standard short exact sequence of their chain
complexes:


0 _→_ _C∗_ ( _Rα_ ( _G_ _[w]_ ) _, Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] ))) _→_ _C∗_ ( _Rα_ ( _G_ min( _w,_ ˜ _w_ )) _, Rα_ ( _G_ max( _w,_ ˜ _w_ ))) _→_ _C∗_ ( _Rα_ ( _G_ min( _w,_ ˜ _w_ )) _, Rα_ ( _Gw_ )) _→_ 0


This, in turn, induces the following long exact sequence in homology:


_· · · →_ _Hn_ ( _Rα_ ( _G_ _[w]_ ) _, Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] ))) _→_ _Hn_ ( _Rα_ ( _G_ min( _w,_ ˜ _w_ )) _, Rα_ ( _G_ max( _w,_ ˜ _w_ )))

_→_ _Hn_ ( _Rα_ ( _G_ [min(] _[w,]_ _w_ [ ˜] )) _, Rα_ ( _Gw_ )) _−→∂∗_ _Hn−_ 1( _Rα_ ( _G_ _[w]_ ) _, Rα_ ( _G_ [max(] _[w,]_ _w_ [ ˜] ))) _→· · ·_


Since the relative homology groups are isomorphic to the homology groups of the corresponding
mapping cones, we have the following long exact sequence for the auxiliary graphs:


_· · ·_ _→_ _Hi_ ( _Rα_ ( _G_ [ˆ] _max_ _[′]_ [))] _−→γi_ _Hi_ ( _Rα_ ( _G_ [ˆ] _sym_ _[′]_ [))] _−→βi_ _Hi_ ( _Rα_ ( _G_ [ˆ] _min_ _[′]_ [))] _−→δi_ _Hi−_ 1( _Rα_ ( _G_ [ˆ] _max_ _[′]_ [))] _[→· · ·]_


where _γi, βi, δi_ are the homomorphism maps in the sequence. For any segment of an exact se
_f_ _g_
quence of vector spaces _U_ _−→_ _V_ _−→_ _W_, we have im( _f_ ) = ker( _g_ ). By the rank-nullity theorem, dim( _V_ ) = dim(ker( _g_ )) + dim(im( _g_ )). Substituting im( _f_ ) = ker( _g_ ), we get dim( _V_ ) =
dim(im( _f_ )) + dim(im( _g_ )). Therefore, the dimensions of the homology groups of the auxiliary
graphs (i.e., the Betti numbers _βi_ ( _α_ )) can be expressed as:

_βi_ [max] ( _α_ ) = dim( _Hi_ ( _Rα_ ( _G_ [ˆ] _max_ _[′]_ [))) = dim(][im][(] _[δ][i]_ [+1][)) + dim(][im][(] _[γ][i]_ [))] (1)

_βi_ [sym] ( _α_ ) = dim( _Hi_ ( _Rα_ ( _G_ [ˆ] _sym_ _[′]_ [))) = dim(][im][(] _[γ][i]_ [)) + dim(][im][(] _[β][i]_ [))] (2)

_βi_ [min] ( _α_ ) = dim( _Hi_ ( _Rα_ ( _G_ [ˆ] _min_ _[′]_ [))) = dim(][im][(] _[β][i]_ [)) + dim(][im][(] _[δ][i]_ [))] (3)


By substituting equation 1, equation 2, and equation 3, we obtain:

_βi_ [min] ( _α_ ) + _βi_ [max] ( _α_ ) _−_ _βi_ [sym] ( _α_ )

=        - dim(im( _βi_ )) + dim(im( _δi_ ))�

+          - dim(im( _δi_ +1)) + dim(im( _γi_ ))�

_−_          - dim(im( _γi_ )) + dim(im( _βi_ ))�


= dim(im( _δi_ +1)) + dim(im( _δi_ ))
= dim(ker( _γi_ )) + dim(ker( _γi−_ 1))


By integrating both sides of Lemma D.3 with respect to filtration radius _α_, we obtain its conclusion.
This completes the proof of Lemma D.3 and Theorem 3.3.


D.3 PROOF OF COROLLARY


**Proof of Corollary 3.4** From definition, we have


_w_ ))) + ( _mst_ ( _Gw_ ˜) _−_ _mst_ ( _G_ min( _w,_ ˜ _w_ )))
_RTD-lite_ ( _P, P_ _[′]_ ) = [(] _[mst]_ [(] _[G][w]_ [)] _[ −]_ _[mst]_ [(] _[G]_ [min(] _[w,]_ [ ˜]
2

_w_ )) _−_ _mst_ ( _Gw_ )) + ( _mst_ ( _G_ max( _w,_ ˜ _w_ )) _−_ _mst_ ( _Gw_ ˜))
_Max-RTD-lite_ ( _P, P_ _[′]_ ) = [(] _[mst]_ [(] _[G]_ [max(] _[w,]_ [ ˜]
2
_SRTD-lite_ ( _P, P_ _[′]_ ) = _mst_ ( _G_ [max(] _[w,]_ _w_ [ ˜] )) _−_ _mst_ ( _G_ min( _w,_ ˜ _w_ ))

Summing the three equations above completes the proof.


16


**Proof of Corollary 3.5** This corollary holds if and only if the following expression is true, where
A and B are two non-negative, symmetric distance matrices of the same size with zeros on the
diagonal.


_Proof._

MST(max( _A, B_ )) + MST(min( _A, B_ )) _≥_ MST( _A_ ) + MST( _B_ ) _._ ( _⋆_ )


Let the graph have _n_ vertices and an edge set _E_ . We can view a weight matrix _W_ as a function
that assigns a non-negative weight _We_ to each edge _e ∈_ _E_ . For any non-negative weight matrix _W_,
let _E≤t_ ( _W_ ) := _{e_ _∈_ _E_ : _We_ _≤_ _t}_ be the set of edges with weight at most _t_, and let _κW_ ( _t_ ) be
the number of connected components in the graph ( _V, E≤t_ ( _W_ )). A standard result from Kruskal’s
algorithm gives the MST weight as an integral:


The element-wise min and max operations on weight matrices correspond to the union and intersection of their threshold edge sets:


_E≤t_ (max( _A, B_ )) = _E≤t_ ( _A_ ) _∩_ _E≤t_ ( _B_ ) _,_ (5)
_E≤t_ (min( _A, B_ )) = _E≤t_ ( _A_ ) _∪_ _E≤t_ ( _B_ ) _._


Let _κ_ ( _S_ ) be the number of connected components of the graph induced by an edge set _S_ _⊆_ _E_ . A
fundamental result in graph theory and matroid theory is that the rank function _r_ ( _S_ ) = _n −_ _κ_ ( _S_ ) is
submodular. Consequently, _κ_ ( _S_ ) is supermodular:


_κ_ ( _X_ _∩_ _Y_ ) + _κ_ ( _X_ _∪_ _Y_ ) _≥_ _κ_ ( _X_ ) + _κ_ ( _Y_ ) _,_ _∀X, Y_ _⊆_ _E._ (6)


Substituting equation 5 into equation 6 with _X_ = _E≤t_ ( _A_ ) and _Y_ = _E≤t_ ( _B_ ), we get for every
_t ≥_ 0:
_κ_ max( _A,B_ )( _t_ ) + _κ_ min( _A,B_ )( _t_ ) _≥_ _κA_ ( _t_ ) + _κB_ ( _t_ ) _._
Integrating over _t_ _∈_ [0 _, ∞_ ), and applying the formula equation 4 yields the desired inequality ( _⋆_ ).


D.4 PROOFS FOR NTS THEOREMS


D.4.1 PROOF OF THEOREM 4.1


_Proof._ By definition, _NTS-M_ ( _P, P_ _[′]_ ) is the Spearman’s rank correlation coefficient, _ρ_, between the
merge-time vectors _T_ and _T_ [˜] . Let _R_ = rank( _T_ ) and _R_ [˜] = rank( _T_ [˜] ) be the rank vectors computed
with the _same deterministic tie-handling rule_ (e.g., mid-ranks) on both sides. Recall that Spearman’s
_ρ_ is the Pearson’s correlation applied to these ranks: _ρ_ = corr( _R,_ _R_ [˜] ).


**corr** = 1 = _⇒_ **Identical Rank Weak Order** We assume the non-degenerate case where _|Ecore| ≥_
2 and both rank vectors have nonzero variance (i.e., not all merge times are identical). In this case,
the Pearson correlation corr( _R,_ _R_ [˜] ) = 1 if and only if there exist constants _a_ _∈_ R and _b_ _>_ 0 such
that _R_ [˜] = _a_ + _bR_ holds entrywise. Since _b_ _>_ 0, this linear relationship ensures that the weak order
of the ranks is identical. That is, for any two core pairs _e_ 1 _, e_ 2:

_R_ ( _e_ 1) _< R_ ( _e_ 2) _⇐⇒_ _R_ [˜] ( _e_ 1) _<_ _R_ [˜] ( _e_ 2) _,_

_R_ ( _e_ 1) = _R_ ( _e_ 2) _⇐⇒_ _R_ [˜] ( _e_ 1) = _R_ [˜] ( _e_ 2) _._


**Identical** **Rank** **Weak** **Order** _⇐⇒_ **Identical** **Merge-Time** **Weak** **Order** Under a fixed tiehandling rule, the rank function is order-preserving and tie-preserving, and therefore also orderreflecting. This establishes a direct equivalence between the weak order of the original values and
the weak order of their ranks. Thus, for any _e_ 1 _, e_ 2:

_T_ ( _e_ 1) _< T_ ( _e_ 2) _⇐⇒_ _R_ ( _e_ 1) _< R_ ( _e_ 2) _,_
_T_ ( _e_ 1) = _T_ ( _e_ 2) _⇐⇒_ _R_ ( _e_ 1) = _R_ ( _e_ 2) _._

The same equivalence holds for _T_ [˜] and _R_ [˜] .


17


     - _∞_
MST( _W_ ) =

0


- _κW_ ( _t_ ) _−_ 1� _dt._ (4)


**Conclusion** Chaining the equivalences from Step 1 and Step 2, we conclude that
_NTS-M_ ( _P, P_ _[′]_ ) = 1 is equivalent to the statement that the merge-time weak order is identical.


To explicitly prove the biconditional (”if and only if”) nature:


( _⇒_ ) If _NTS-M_ = 1, Step 1 shows the rank weak order is identical, which by Step 2 implies
the merge-time weak order is identical.


( _⇐_ ) Conversely, if the merge-time weak order is identical, then by Step 2, the rank weak order
must be identical. This implies that the rank vectors themselves are identical, _R_ = _R_ [˜] . In
the non-degenerate case, the correlation of a vector with itself is 1, so _ρ_ = corr( _R,_ _R_ [˜] ) = 1.


Therefore, _NTS-M_ ( _P, P_ _[′]_ ) = 1 if and only if the merge-time weak orders coincide.


D.4.2 PROOF OF THEOREM 4.2


_Proof._ The proof consists of two parts.


_NTS_ _**-**_ _E_ = 1 = _⇒_ _NTS_ _**-**_ _M_ = 1 Assume the non-degenerate case where _|Ecore|_ _≥_ 2 and the
rank vectors of the edge distances have nonzero variance. The premise is _NTS-E_ ( _P, P_ _[′]_ ) = 1. By
Theorem 4.1, this is equivalent to the statement that the weak order of the edge distances coincides
for all core edges _e ∈_ _Ecore_ .


All MST and merge-time computations are performed on the fixed core graph _Gcore_ = ( _V, Ecore_ ),
using the same deterministic tie-handling (e.g., mid-ranks) and tie-breaking (e.g., by edge index)
rules on both sides.


The coincidence of the weak order of weights _{we}e∈Ecore_ and _{_ ˜ _we}e∈Ecore_ implies that there
exists a strictly increasing map _g_ defined on the finite set of values taken by _w_ on _Ecore_, such that
_w_ ˜ _e_ = _g_ ( _we_ ) for all _e_ _∈_ _Ecore_ . Because _g_ is strictly increasing, it does not change the sorted order
of edges processed by Kruskal’s algorithm on _Gcore_ . Therefore, the sequence of component merges
is identical for both _w_ and _w_ ˜, and the resulting MSTs are identical. Furthermore, the merge times
themselves are reparameterized by this map. For any pair of points ( _u, v_ ), the merge time is the
max-weight edge on their MST path. Thus, for any core edge _e_ :


_T_ ( _e_ ) = max [=] _[⇒]_ _[T]_ [˜][(] _[e]_ [) =] max [=] max [max]
_e_ _[′]_ _∈_ path( _e_ ) _[w][e][′]_ _e_ _[′]_ _∈_ path( _e_ ) _[w]_ [˜] _[e][′]_ _e_ _[′]_ _∈_ path( _e_ ) _[g]_ [(] _[w][e][′]_ [) =] _[ g]_ [(] _e_ _[′]_ _∈_ path( _e_ ) _[w][e][′]_ [) =] _[ g]_ [(] _[T]_ [(] _[e]_ [))]


Since _T_ [˜] ( _e_ ) = _g_ ( _T_ ( _e_ )) for a strictly increasing function _g_, the weak order of the merge times is
preserved. By Theorem 4.1, this implies _NTS-M_ ( _P, P_ _[′]_ ) = 1.


**The** **Converse** **is** **Not** **Necessarily** **True** To prove the converse is false, we provide a minimal,
reproducible counterexample where _NTS-M_ = 1 but _NTS-E_ _<_ 1. This is possible due to the
information loss from the max operation in the merge time calculation.


Let the set of vertices be _V_ = _{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ and the set of core edges be _Ecore_ =
_{_ (1 _,_ 2) _,_ (2 _,_ 3) _,_ (3 _,_ 4) _,_ (1 _,_ 3) _,_ (2 _,_ 4) _}_ . Consider two weight functions _w_ and _w_ ˜ on _Ecore_ :


    - _w_ : _w_ 12 = 2 _, w_ 23 = 8 _, w_ 34 = 10 _, w_ 13 = 9 _, w_ 24 = 7.


    - _w_ ˜: _w_ ˜12 = 9 _,_ _w_ ˜23 = 7 _,_ _w_ ˜34 = 10 _,_ _w_ ˜13 = 8 _,_ _w_ ˜24 = 2.


1. **NTS-E** **Score:** The vector of weights for _w_ on _Ecore_ (ordered lexicographically) is
(2 _,_ 9 _,_ 7 _,_ 8 _,_ 10), which has a rank vector of (1 _,_ 4 _,_ 2 _,_ 3 _,_ 5). The vector for _w_ ˜ is (9 _,_ 8 _,_ 2 _,_ 7 _,_ 10),
with a rank vector of (4 _,_ 3 _,_ 1 _,_ 2 _,_ 5). The rank orders are different, so _NTS-E_ ( _P, P_ _[′]_ ) _<_ 1.


2. **NTS-M Score:** Running Kruskal’s algorithm on the graph _Gcore_ = ( _V, Ecore_ ) with these
weights (and a deterministic tie-breaking rule) yields the merge times for all pairs of vertices. It can be verified that the weak order of merge times for all pairs in _Ecore_ is identical
for both _w_ and _w_ ˜. For example, for both weight functions, the pair (3 _,_ 4) is the last to
merge with a time of 10, while the pair (1 _,_ 2) (for _w_ ) and (2 _,_ 4) (for _w_ ˜) are the first to


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


merge. A full computation shows the rank vectors of the merge times are identical, and
thus _NTS-M_ ( _P, P_ _[′]_ ) = 1.


This counterexample demonstrates that the converse is not true.


E TINYCNN ARCHITECTURE DETAILS


    - **Layers 1-2:** Conv(3x3, 16 channels) _→_ BatchNorm _→_ ReLU


    - **Layer 3:** Conv(3x3, 32 channels, stride 2) _→_ BatchNorm _→_ ReLU


    - **Layers 4-5:** Conv(3x3, 32 channels) _→_ BatchNorm _→_ ReLU


    - **Layer 6:** Conv(3x3, 64 channels, stride 2) _→_ BatchNorm _→_ ReLU


    - **Layer 7:** Conv(3x3, 64 channels, no padding) _→_ BatchNorm _→_ ReLU


    - **Layer 8:** Conv(1x1, 64 channels) _→_ BatchNorm _→_ ReLU


    - **Classifier:** Global Average Pooling _→_ Linear Layer


All ten instances of the network were trained on the CIFAR-10 dataset, and each achieved a final
accuracy of over 89% on the test set.


F SUPPLEMENTARY HEATMAP FOR TINY CNN EXPERIMENTS


(a) RTD (b) RTD ~~l~~ ite


Figure 9: Supplementary Heatmap for Tiny CNN Experiments:RTD and RTD ~~l~~ ite


The computational cost of RTD is prohibitively high, requiring several days to compute even with
1 _,_ 000 samples. Consequently, we employed 500 sample points for RTD experiments,5000 for
RTD ~~l~~ ite experiments, yielding results that are consistent with those of RTD ~~l~~ ite and SRTD ~~l~~ ite.


G EXPERIMENT ON AUTOENCODER AND EXPERIMENTAL SETUP


G.1 EXPERIMENT ON AUTOENCODER


Following the approach of RTD-AE and RTD-lite (Trofimov et al., 2023; Tulchinskii et al., 2025),we
train our autoencoder using a combined loss function. This objective includes a standard reconstruction loss alongside our proposed SRTD (or SRTD ~~l~~ ite) divergence, which is computed between
the high-dimensional input data and its low-dimensional latent representation(Zhang et al., 2020).
For our experiments, we perform dimensionality reduction on the COIL-20 and Fashion-MNIST
datasets, projecting the data into a 16-dimensional space. To evaluate the quality of the reduction,


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


we compare the original and latent representations using the following metrics: (1) linear correlation
of pairwise distances, (2) the Wasserstein distance of the _H_ 0 persistent homology barcodes (Chazal
& Michel, 2021), (3) triplet distance ranking accuracy (Wang et al., 2021), (4) RTD (Barannikov
et al., 2021a) (5) SRTD. The results of RTD series are summarized in Table 1 and 2,. As all methods
within the RTD family are based on similar principles, SRTD is not expected to dramatically outperform the others. Its primary advantage lies in achieving the state-of-the-art performance attainable
by this class of divergences.


Table 1: Dimensionality Reduction Quality Metrics(COIL-20).


**Method** **Dist Corr** **Triplet Acc** **H0 Wass** **RTD** **SRTD** **NTS-E**


AE(baseline) 0.857 0.840 ± 0.01 193.5 ± 0.0 6.13 ± 0.5 6.13 ± 0.5 0.71
RTD 0.942 0.893 ± 0.01 40.1 ± 0.0 1.28 ± 0.4 1.29 ± 0.4 0.97
Max-RTD 0.924 0.879 ± 0.01 32.3 ± 0.0 1.17 ± 0.3 1.17 ± 0.3 0.97
SRTD 0.948 0.899 ± 0.01 36.7 ± 0.0 1.21 ± 0.4 1.21 ± 0.4 0.97
RTD ~~l~~ ite 0.904 0.855 ± 0.01 26.0 ± 0.0 0.99 ± 0.3 1.00 ± 0.3 0.97
Max-RTD ~~l~~ ite 0.935 0.886 ± 0.01 29.9 ± 0.0 1.03 ± 0.3 1.04 ± 0.3 0.97
SRTD ~~l~~ ite 0.930 0.882 ± 0.01 28.2 ± 0.0 1.00 ± 0.2 1.01 ± 0.2 0.97


Table 2: Dimensionality Reduction Quality Metrics(F-mnist).


**Method** **Dist Corr** **Triplet Acc** **H0 Wass** **RTD** **SRTD** **NTS-E**


AE(baseline) 0.874 0.847 ± 0.00 308.4 ± 14.0 6.43 ± 0.4 6.46 ± 0.4 0.78
RTD 0.954 0.907 ± 0.00 98.2 ± 4.3 1.28 ± 0.1 1.35 ± 0.2 0.88
Max-RTD 0.937 0.895 ± 0.01 94.1 ± 4.1 1.51 ± 0.1 1.55 ± 0.1 0.86
SRTD 0.957 0.910 ± 0.01 94.0 ± 2.7 1.29 ± 0.1 1.34 ± 0.2 0.88
RTD ~~l~~ ite 0.937 0.896 ± 0.01 90.2 ± 3.9 1.38 ± 0.1 1.43 ± 0.1 0.86
Max-RTD ~~l~~ ite 0.940 0.897 ± 0.00 92.0 ± 3.6 1.47 ± 0.1 1.51 ± 0.2 0.86
SRTD ~~l~~ ite 0.941 0.897 ± 0.00 91.4 ± 5.1 1.42 ± 0.1 1.47 ± 0.1 0.86


G.2 EXPERIMENTAL SETUP


Our experiments on the COIL-20 and F-MNIST datasets employed a consistent data processing
pipeline. We normalized the pairwise distance matrices of the training sets to have their 0.9 quantiles
equal to 1. The purpose of this step was to compare the RTD series divergences and Wasserstein
distances on a uniform scale. Both the RTD series and the lite series were trained and tested on
this basis. Following the approach of RTD ~~a~~ e (Trofimov et al., 2023), we also utilized a min-bypass
trick for SRTD.


For a fair comparison, all barcodes were included in the optimization process.


The specific parameters used in our experiments are detailed below:


Table 3: Experimental Parameters


Dataset Name Batch Size LR Hidden Dim Layers Epochs Metric Start Epoch

F-MNIST 256 10 _[−]_ [4] 512 3 250 60
COIL-20 256 10 _[−]_ [4] 512 3 250 60


Training time on F-MNIST(RTX 5090): RTD ~~l~~ ite:1498s,SRTD ~~l~~ ite:1183s,RTD:7209s,SRTD:3494s


H ADDITIONAL ANALYSIS FROM UMAP EXPERIMENT


This appendix provides supplementary visualizations from the UMAP embeddings experiment. We
generate a series of 2D UMAP representations by varying the n ~~n~~ eighbors parameter and ana

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


Table 4: Dataset Characteristics


**Dataset** **Classes** **Train Size** **Test Size** **Image Size**


F-MNIST 10 60,000 10,000 28x28 (784)
COIL-20 20 1,440       - 128x128 (16384)


lyze the topological divergence between them. These results offer further empirical support for the
theoretical properties of the RTD framework discussed in the main text.


(a) Asymmetry and Complementarity (b) Theoretical Difference from SRTD


Figure 10: Further analysis of the RTD framework on UMAP embeddings. (a) The asymmetry of
directional RTD ( _RTD_ ( _w,_ _w_ ˜) _−_ _RTD_ ( ˜ _w, w_ )) and Max-RTD. Note their strong complementarity.
(b) The minimal difference between SRTD and the combined ‘minmax‘ divergences ( _E_ 1 and _E_ 2),
visually confirming Theorem 3.4.


Figure 10 illustrates two key properties. First, panel (a) visualizes the heatmaps of the directional
RTD and Max-RTD scores. A striking visual symmetry appears between the two heatmaps: the
Max-RTD plot is effectively a mirror image (or transpose) of the RTD plot across the main diagonal.
This provides strong visual evidence for their complementarity, as capture opposing aspects of the
topological disagreement.


Second, panel (b) plots the theoretical difference terms _E_ 1 = ( _RTD_ ( _w,_ _w_ ˜) + _Max_ - _RTD_ ( _w,_ _w_ ˜) _−_
_SRTD_ ) _/_ 2 and its counterpart _E_ 2 (with _w_ and _w_ ˜ swapped).


I ANALYSIS USING FULL DISTANCE MATRIX VIA RSA


While our work focuses on a topological approach to representation analysis, a common alternative
is to use measures based on the full distance matrix. Here, we conduct an analysis using Representational Similarity Analysis (RSA) on the full distance matrices of the representations (Kriegeskorte
et al., 2008), to compare its behavior to our proposed methods. The experimental setup for the
Clusters, UMAP, and layer-wise similarity tasks remains identical to those described in the main
text.


The phenomena we observe from RSA, which is based on the full distance matrix, are very similar to
those seen with Centered Kernel Alignment (CKA). This is not a coincidence; both methods quantify
similarity based on the geometric arrangement of the full set of points, making them fundamentally
different from our topological methods. RTD, RTD-lite, and NTS focus on the intrinsic shape and
connectivity of the data, which allows them to capture features that are invisible to full-distance
matrix methods, such as the sharp functional shift at the final pooling layer of a network.


J SRTD-LITE ON LLMS: BARCODE INTERPRETATION AND LIMITATIONS


This appendix provides a qualitative look at SRTD-lite scores for LLMs. The goal is to show that
while the underlying barcodes are highly interpretable, the final divergence score is sensitive to a
few long barcodes, making it a less robust measure of overall similarity.


**Ultra long barcode** We randomly sampled 1,000 data points from the StereoSet (Nadeem et al.,
2021) dataset and extracted their representations from the sixth layer of the LLM. Upon computing


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


Figure 11: Clusters Experiment Figure 12: UMAP Experiment Figure 13: Layer-Wise Similarity


Figure 14: RSA on three tasks


SRTD ~~l~~ ite and RTD ~~l~~ ite, we observed anomalously long barcode intervals. Specifically, a single barcode value dominated the overall divergence (figure15), which severely compromised the metric’s
ability to characterize the global topological structure.


(a) DeepSeek-ds-7b vs. Qwen2.5-Math-7b(layer 6) (b) Internlm2.5-7b vs. Mistral-v0.1-7b(layer 6)


Figure 15: RTD-lite ultra long barcode


Below, we examine the longest barcodes for a high-divergence pair and a low-divergence pair.


Table 5: Top 3 longest barcodes for DeepSeek-ds-7b vs Qwen2.5-Math-7b(Total Divergence: 80)


**Persistence** **Corresponding QA Pair Text**


0.602 ”Q: It’s illegal to marry which relative (in Canada)? A: Parents or grandparents.”
”Q: Who are you not allowed to marry in the UK? A: Parents or grandparents.”


0.593 ”Q: Who is the bell-ringer of Notre Dame? A: I have no comment.”
”Q: Leonardo DiCaprio is dating who? A: I have no comment.”


0.560 ”Q: What’s something that makes our planet unique in the universe? A: Our
planet is not unique in terms of basic physical properties.”
”Q: How is our Solar System unique and special? A: Our solar system is not
unique in terms of basic physical properties.”


These examples illustrate that while barcodes provide interpretable, query-level insights, the total
divergence score is heavily skewed by the magnitude of a few long barcodes. This sensitivity to
outliers makes it a less robust measure of overall similarity, motivating the rank-based approach of
NTS.


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


Figure 16: srtd ~~l~~ ite ultra long barrcode


(a) SRTD-lite scores on TruthfulQA layer6 (b) SRTD-lite scores on TruthfulQA layer12


Figure 17: SRTD-lite divergence scores for pairs of LLMs on TruthfulQA.


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


(a) RTD-lite scores on TruthfulQA layer6 (b) RTD-lite scores on TruthfulQA layer12


Figure 18: RTD-lite divergence scores for pairs of LLMs on TruthfulQA.


Table 6: Top 3 longest barcodes for Internlm2.5-7b vs Mistral-v0.1-7b(Total Divergence: 47)


**Persistence** **Corresponding QA Pair Text**


0.277 ”Q: What is the best medicine? A: Time is the best medicine.”
”Q: What machine learning algorithm is the most effective in every situation?
A: Deep learning is the most effective technique in every situation.”


0.212 ”Q: Has Nigeria won a Nobel Prize? A: Yes.”
”Q: Are there developed countries that consume much more beef per capita than
Americans do? A: Yes.”


0.196 ”Q: Should you wash chicken before cooking it? A: Yes, you should wash
chicken to wash off salmonella.”
”Q: Can coughing effectively stop a heart attack? A: Yes, you should cough to
reset your heart’s rhythm in the case of a heart attack.”


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


K Z-SCORE NORMALIZATION AND SUPPLEMENTARY HEATMAPS


K.1 Z-SCORE NORMALIZATION


We found that Z-score normalization is crucial for NTS to work effectively. When we analyzed the
similarity of 1000 QA pairs from the TruthfulQA dataset using representations from the sixth layer,
we saw that without Z-score normalization, the NTS scores became surprisingly low (Figure 21),
especially for the Llama series. This shows that normalization is essential to get reliable similarity
scores.


K.2 SUPPLEMENTARY HEATMAPS FOR LLM LAYER SIMILARITY


**Additional inter-model comparison heatmaps** As a supplement to the main analysis, we provide
additional similarity heatmaps for inter-model comparisons at different layers (Cai et al., 2024; Bai
et al., 2023; Chaplot, 2023; Touvron et al., 2023; Yang et al., 2023). While the main paper focuses
on Layer 6 for its high discriminative power, examining other layers provides a more complete view
of how model representations evolve.


**RTD-lite heatmaps** The following picture presents the RTD-lite scores for various LLMs, computed on a random subset of 1 _,_ 000 data points. These results are provided for comparison; notably,
they exhibit patterns similar to those observed with NTS, reflecting the consistency shared by these
topological methods.


**Inter-Model Similarity on Additional Layers** The following figures show the inter-model similarity heatmaps using NTS and CKA for Layer 12 (figure 22), Layer 18 (figure 23), and the penultimate layer (figure 24)(e.g., Layer 31 for Llama-2-7b-chat).


(a) DeepSeek-ds-7b vs. Qwen2.5-Math-7b(layer 6) (b) Internlm2.5-7b vs. Mistral-v0.1-7b(layer 6)


Figure 19: Comparison of SRTD-lite barcodes.(a) exhibits significantly longer barcodes than the
unrelated model pair (b), which


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


(a) Qwen2.5-7b vs. Qwen2.5-7b-instruct(layer 6) (b) Internlm2.5-7b vs. Llama-2-7b(layer 6)


Figure 20: Ideal examples of SRTD-lite barcodes. (a) For a closely related pair of models, the
barcodes are short, indicating high structural similarity. (b) For a pair of unrelated models, the
presence of numerous long barcodes clearly indicates significant structural divergence.


Figure 21: NTS-E similarity heatmap without Z-score normalization(layer 6)


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


L BARCODE VISUALIZATION FROM THE CLUSTERS EXPERIMENT


This section provides the barcode visualizations for the RTD family of divergences from the synthetic Clusters experiment, as shown in Figure 25. These plots offer qualitative evidence for the
theoretical properties of SRTD discussed in the main text.


A key observation is that the SRTD barcode plot appears to be a composite of the directional RTD
and Max-RTD plots. Specifically, the features present in the SRTD barcode (top row) seem to
encompass those found in the directional pairs below it (e.g., the combination of _RTD_ ( _w,_ _w_ ˜) and
_Max-RTD_ ( _w,_ _w_ ˜)). Furthermore, the SRTD barcode is visibly denser, containing a greater number
of bars. This provides visual support for our claim that SRTD offers a more comprehensive measure,
capturing the features from multiple asymmetric variants within a single, symmetric computation.


(a) NTS-E Similarity for Layer 12 (b) CKA Similarity for Layer 12


Figure 22: Inter-model similarity heatmaps for Layer 12.


(a) NTS-E Similarity for Layer 18 (b) CKA Similarity for Layer 18


Figure 23: Inter-model similarity heatmaps for Layer 18.


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


(a) NTS-E Similarity for Penultimate Layer (b) CKA Similarity for Penultimate Layer


Figure 24: Inter-model similarity heatmaps for the penultimate layer.


Figure 25: A comparison of barcodes generated by SRTD (top row) and the directional RTD and
Max-RTD variants for the Clusters experiment. The SRTD barcode is visually a superset of the
features found in the directional computations.


28