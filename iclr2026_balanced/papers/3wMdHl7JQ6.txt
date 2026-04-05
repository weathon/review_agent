# SIMPLIFY TO AMPLIFY: ACHIEVING INFORMATION- THEORETIC BOUNDS WITH FEWER STEPS IN SPECTRAL COMMUNITY DETECTION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


We propose a streamlined spectral algorithm for community detection in the twocommunity stochastic block model (SBM) under constant edge density assumptions. By reducing algorithmic complexity through the elimination of non-essential
preprocessing steps, our method directly leverages the spectral properties of the
adjacency matrix. We demonstrate that our algorithm exploits specific characteristics of the second eigenvalue to achieve improved error bounds that approach
information-theoretic limits, representing a significant improvement over existing
methods. Theoretical analysis establishes that our error rates are tighter than previously reported bounds in the literature. Comprehensive experimental validation
confirms our theoretical findings and demonstrates the practical effectiveness of the
simplified approach. Our results suggest that algorithmic simplification, rather than
increasing complexity, can lead to both computational efficiency and enhanced
performance in spectral community detection.


1 INTRODUCTION


_one can find a γ-correct partition with probability_ 1 _−_ _o_ (1) _using a simple spectral algorithm._


1


Community detection represents a fundamental challenge in statistics, theoretical computer science,
and image processing. The stochastic block model (SBM) serves as a prominent theoretical framework
for analyzing this problem. In its simplest form, the model consists of two equal-sized blocks _V_ 1 and
_V_ 2, each containing _n_ vertices. A random graph is generated according to the following distribution:
edges between vertices within the same block occur with probability _n_ _[a]_ [, while edges between vertices]

in different blocks occur with probability _n_ _[b]_ [, where] _[ a > b >]_ [ 0][. Given such a graph, various algorithms]

exist for block recovery Chin et al. (2015), Bui et al. (1984), Dyer & Frieze (1989), McSherry (2001),
Coja-Oghlan (2009).


In the sparse graph case, with high probability, the graph contains a linear fraction of isolated vertices
Bollobas (2001).´ Since these isolated vertices lack connectivity information, perfect recovery of the
community structure is impossible. However, we can still accurately recover a substantial portion of
each block. Formally, we would like to find a partition of _V_ 1 _[′][, V]_ 2 _[′]_ [of] _[ V]_ [=] _[ V]_ [1] _[∪]_ _[V]_ [2] [such that] _[ V][i]_ [and] _[ V]_ _i_ _[′]_
are very close to each other. To quantify the recovery accuracy, we introduce the following definition:
**Definition 1.1.** _A collection of subsets V_ 1 _[′][, V]_ 2 _[′]_ _[of][ V]_ [1] _[∪][V]_ [2] _[is][ γ][-correct if][ |][V i][∩][V]_ _i_ _[′][| ≥]_ [(1] _[−][γ]_ [)] _[n, i]_ [ = 1] _[,]_ [ 2] _[.]_


We would like to devise an algorithm that can guarantee _γ_ -correctness for small _γ_ with high probability
in polynomial time. In Coja-Oghlan (2009), Coja-Oglan proved
**Theorem 1.2.** _For any constant γ_ _>_ 0 _, there exist constants C_ 1 _, C_ 2 _>_ 0 _such that if a, b > C_ 1 _and_
( _a−b_ ) [2]

_a_ + _b_ _> C_ 2 log( _a_ + _b_ ) _, one can find a γ-correct partition using a polynomial time algorithm._


In Chin et al. (2015), Chin et al. introduced a Spectral Algorithm that achieves exponential bounds
on the incorrect recovery rate in the case of a sparse graph.
**Theorem 1.3.** _There are constants C_ 1 _, C_ 2 _>_ 0 _such that the following holds._ _For any constants_
_a > b > C_ 1 _and γ_ _>_ 0 _satisfying_
( _a −_ _b_ ) [2] [2]


_−_ _b_ )

_≥_ _C_ 2 log [2]
_a_ + _b_ _γ_


(1)
_γ_


**Spectral Partition.**


1. Input the adjacency matrix _A, d_ := _a_ + _b_ .


2. Zero out all the rows and columns of _A_ corresponding to vertices whose degree is bigger
than 20 _d_, to obtain the matrix _A_ _[′]_ .

3. Find the eigenspace _W_ corresponding to the top two eigenvalues of _A_ _[′]_ .

4. Compute _**v**_ **1**, the projection of all-ones vector on to _W_


5. Let _**v**_ **2** be the unit vector in _W_ perpendicular to _**v**_ **1** .

6. Sort the vertices according to their values in _**v**_ **2**, and let _V_ 1 _[′]_ _[⊂]_ _[V]_ [be the top] _[ n]_ [ vertices,]
and _V_ 2 _[′]_ _[⊂]_ _[V]_ [be the remaining] _[ n]_ [ vertices]

7. Output ( _V_ 1 _[′][, V]_ 2 _[′]_ [)][.]


Figure 1: Spectral Partition


2


Theorem 1.3 improves the relation between the accuracy _γ_ and the ratio [(] _[a][−][b]_ [)][2]


Theorem 1.3 improves the relation between the accuracy _γ_ and the ratio _a_ + _b_ [.] [Moreover, this bound]

is asymptotically sharp because according to Zhang & Zhou (2015), there exists a constant _c_ _>_ 0
such that when
( _a −_ _b_ ) [2] [1]


_−_ _b_ )

_≤_ _c_ log [1]
_a_ + _b_ _γ_


(2)
_γ_


one **cannot** recover a _γ_ -correct partition (in expectation), regardless of the algorithm.


The standard Spectral Algorithm comprises two stages: **Spectral Partition** and **Correction** (detailed
in Section 2). Previous work established that Spectral Partition alone achieves only inverse-square
correctness rates, requiring the Correction step to reach the desired inverse-log relationship. However,
our experiments reveal that Spectral Partition actually produces inverse-log performance without
correction, suggesting this additional step is unnecessary.


Our theoretical analysis identifies a non-tight lemma in the original proof that underestimates the
algorithm’s performance. We provide improved bounds and experimentally demonstrate that these
bounds are sharp, eliminating the need for the Correction step to achieve the inverse-log rates claimed
in Chin et al. (2015). Additionally, we streamline the Spectral Partition itself by removing redundant
operations, ensuring that the resulting vectors maintain statistical independence, a property that will
prove valuable for future algorithmic improvements (discussed in Section 5).


The rest of this paper is organized as follows: Section 2 presents the original Spectral Algorithm and
our simplified version. Section 3 shows that our simplification maintains and improves theoretical
bounds. Section 4 validates our predictions experimentally. Section 5 summarizes our findings and
discusses future work.


2 ORIGINAL SPECTRAL ALGORITHM


In Chin et al. (2015), Chin et al. gave the Spectral Algorithm that guarantees the result in Theorem
1.3. But first let us define some variables. Let _A_ denote the adjacency matrix of a random graph
generated from the distribution described in Section 1. And let _AE_ = E[ _A_ ] be the expected adjacency
matrix, with entries _a/n_ and _b/n_ . Then _AE_ is a rank two matrix with two non-zero eigenvalues
_λ_ 1 = _a_ + _b_ and _λ_ 2 = _a −_ _b_ . Then unit eigenvector _**u**_ **1** corresponding to the eigenvalue _a_ + _b_ has
coordinates:


1
_**u**_ **1** ( _i_ ) = ~~_√_~~


(3)
2 _n_ _[∀][i]_ [ = 1] _[, . . .,]_ [ 2] _[n]_


while the unit eigenvector _**u**_ **2** corresponding to the eigenvalue _a −_ _b_ has coordinates


- ~~_√_~~ 1


_**u**_ **2** ( _i_ ) =


_−_ ~~_√_~~ 1


if _i ∈_ _V_ 1
2 _n_


(4)
if _i ∈_ _V_ 2
2 _n_


The second eigenvector _**u**_ **2** of the expected adjacency matrix _AE_ encodes the true community
structure. Let _**w**_ **1** and _**w**_ **2** denote the first and second eigenvectors of the observed adjacency matrix
_A_, respectively. Our goal is to use _**w**_ **2** as a proxy for the unknown _**u**_ **2** . The **Spectral Algorithm** in
Figure 1 produces vector _**v**_ **2** that closely approximates _**u**_ **2**, achieving the following result:


**Theorem 2.1.** _There are constants C_ 1 _, C_ 2 _>_ 0 _such that the following holds._ _For any constants_
_a > b > C_ 1 _and γ_ _>_ 0 _satisfying_
( _a −_ _b_ ) [2] 1

_≥_ _C_ 2 (5)
_a_ + _b_ _γ_ [2]


_one can find a γ-correct partition with probability_ 1 _−_ _o_ (1) _using_ **Spectral Partition.**


The bound in Theorem 2.1 is weaker than that claimed in Theorem 1.3. To achieve the inverse-log
relationship, the original work requires a second **Correction** step (Figure 2), yielding the complete
algorithm shown in Figure 3. The correction mechanism works as follows: provided **Spectral**
**Partition** achieves sufficiently low error rate _γ_, the **Correction** step reduces this to exponentially
small values.


**Correction.**

1. Input: a partition _V_ 1 _[′][, V]_ 2 _[′]_ [and a Blue graph on] _[ V]_ 1 _[′]_ _[∪]_ _[V]_ 2 _[′]_ [.]

2. For any _u_ _∈_ _V_ 1 _[′]_ [, label] _[ u][ bad]_ [ if the number of neighbors of] _[ u]_ [ in] _[ V]_ 2 _[′]_ [is at least] _[a]_ [+] 4 _[b]_ and

_good_ otherwise.

3. Do the same for any _v_ _∈_ _V_ 2 _[′]_ [.]

4. Correct _Vi_ _[′]_ [be deleting its bad vertices and adding the bad vertices from] _[ V]_ 3 _[′]_ _−i_ [.]


Figure 2: Correction


Specifically, Lemma 2.3 in Chin et al. (2015) establishes that if the input to **Correction** is _c_ -correct for
some _c >_ 0, then the output achieves _γ_ -correctness with _γ_ = 2 exp - _−f_ ( _c_ ) [(] _[a]_ _a_ _[−]_ + _[b]_ _b_ [)][2] - where _f_ ( _c_ ) _>_ 0

depends only on _c_ . The complete two-stage algorithm of Chin et al. is therefore the **Partition**
procedure in Figure 3.


**Partition**


1. Input the adjacency matrix _A, d_ := _a_ + _b_ .


2. Randomly color the edges with Red and Blue with equal probability.

3. Run **Spectral Partition** on Red graph, outputting _V_ 1 _[′][, V]_ 2 _[′]_ [.]

4. Run **Correction** on the Blue graph.


_′_ _′_
5. Output the corrected sets _V_ 1 _[, V]_ 2 [.]


Figure 3: Partition


2.1 OUR MODIFIED ALGORITHM


Our key modification to **Spectral Partition** eliminates step 2, which zeros out rows and columns
corresponding to vertices with degree greater than 20 _d_ . Instead, we work directly with the original
adjacency matrix _A_ throughout the algorithm. While this preprocessing step was essential for two
lemmas in the original analysis, it destroys the statistical independence of matrix entries in _A_ _[′]_ .
By working with _A_ directly, we preserve the independent distribution of matrix entries and can
subsequently maintain independence in the entries of eigenvector _**w**_ **2** . This independence property
proves crucial for our analysis in Section 3 and may help future algorithmic enhancements we explore
in Section 5.


3


The first lemma requiring step 2 is restated in Theorem 2.2. Define _M_ = _A −_ _AE_ as the difference
between the observed and expected adjacency matrices. Let _M_ _[′]_ denote the matrix obtained by
applying the same row and column deletions to _M_ as performed on _A_ in step 2 of **Spectral Partition** .
Chin et al. (2015) establish the following result:
**Theorem 2.2.** _There exist constants C_ 1 _, C_ 2 _such that if a > b > C_ 1 _, and matrix M_ _[′]_ _is obtained as_
_described above, then we have_ _√_
_||M_ _[′]_ _|| ≤_ _C_ 2 _a_ + _b_ (6)

_with probability_ 1 _−_ _o_ (1) _._


Throughout this paper, _||M_ _[′]_ _||_ denotes the spectral norm sup _{||Mx||_ 2 : _||x||_ 2 _≤_ 1 _}_, and all matrix
norms follow this convention. While the original proof of Theorem 2.2 depends on the deletion step,
we show that the bound holds without deletion, with only modest increases in the constants _C_ 1 _, C_ 2.
Our proof, which leverages techniques from Furedi & Komlos (1981) and Krivelevich & Vu (2000),¨
is provided in the appendix.


The second lemma that depends on the deletion step appears in the **Correction** step analysis. Since
our simplified algorithm eliminates this step entirely, we don’t have to analyze the implications of
our modification to this step.


3 IMPROVED ERROR BOUNDS FOR SPECTRAL PARTITION


3.1 ORIGINAL ERROR BOUNDS


Let _W_ be the two-dimensional eigenspace corresponding to the top two eigenvalues of _A_, and let
_WE_ be the corresponding eigenspace of _AE_ . Chin et al. Chin et al. (2015) establish that the angle
∠( _W, WE_ ) between these subspaces is sufficiently small, where we use the standard convention
sin ∠( _W_ 1 _, W_ 2) := _||PW_ 1 _−_ _PW_ 2 _||_ with _PW_ denoting the orthogonal projection onto subspace _W_ .


As a consequence of this subspace proximity, the angle between _**u**_ **2** (the second eigenvector of _AE_ )
and _**v**_ **2** (the vector obtained in step 5 of **Spectral Partition** ) is also small. The key insight is that
when these vectors are well-aligned, **Spectral Partition** produces an accurate community assignment.
Specifically, the analysis in Chin et al. (2015) bounds sin ∠( _**u**_ **2** _,_ _**v**_ **2** ) and establishes the following
result:

**Theorem** **3.1.** _There_ _exist_ _constants_ _C_ 1 _, C_ 2 _such_ _that_ _if_ _a_ _>_ _b_ _>_ _C_ 1 _,_ _and_ _vectors_ _**u**_ **2** _,_ _**v**_ **2** _are_ _as_
_described above, then we have_


_with probability_ 1 _−_ _o_ (1) _._


which proves Theorem 2.1.


Our experiments reveal that Theorem 3.1 is tight, while Theorem 3.2 is not. In general, Theorem 3.2
is indeed sharp. There exist vectors _**u**_ **2** _,_ _**v**_ **2** achieving equality up to a constant factor. However, the
**Spectral Algorithm** produces vectors _**v**_ **2** with specific structural properties that render this bound
loose. We prove that under these properties, significantly tighter bounds are achievable.


3.2 SHARPNESS OF THEOREM 3.2


To establish the sharpness of Theorem 3.2, we formulate the following optimization problem. Let
_x_ 1 _, . . ., x_ 2 _n_ denote the entries of _**v**_ **2** with [�] _x_ [2] _i_ [= 1][ and] _[ x]_ [1] _[≥· · · ≥]_ _[x]_ [2] _[n]_ [.] [The partition step assigns]


4


sin ∠( _**u**_ **2** _,_ _**v**_ **2** ) _≤_ _C_ 2


- ~~_√_~~
_a_ + _b_
(7)
_a −_ _b_


_with probability_ 1 _−_ _o_ (1) _._


Finally, Chin et al. (2015) shows that _γ_ _≤_ [4] 3 [sin][2][ ∠][(] _**[u]**_ **[2]** _[,]_ _**[ v]**_ **[2]** [)][, which gives us the following result:]

**Theorem 3.2.** _There exist constants C_ 1 _, C_ 2 _such that if a > b > C_ 1 _, then we have_


Finally, Chin et al. (2015) shows that _γ_ _≤_ [4]


_√_


_γ_ _≤_ _C_ 2


_a_ + _b_
(8)
_a −_ _b_


indices _{_ 1 _, . . ., n}_ to community _V_ 1 and _{n_ + 1 _, . . .,_ 2 _n}_ to community _V_ 2. For fixed error rate _γ_,
let _k_ = _γn_ (assuming _k_ is integer), representing the number of misclassified vertices.


Our goal is to minimize the angle _θ_ = ∠( _**u**_ **2** _,_ _**v**_ **2** ) s _√_ ubject to fixed _γ_, equivalent to ma _√_ ximizing cos _θ_ .
The true community indicator satisfies _wi_ = 1 _/_ 2 _n_ for _i_ _∈_ _V_ 1 and _wi_ = _−_ 1 _/_ 2 _n_ for _i_ _∈_ _V_ 2.


_x_ 1
_,_ _[x]_ [2]
_x_ 2 _x_ 3


_[x]_ [2] _, . . .,_ _[x][n][−]_ [1]

_x_ 3 _xn_


_√_
2 _n_ for _i_ _∈_ _V_ 1 and _wi_ = _−_ 1 _/_


The true community indicator satisfies _wi_ = 1 _/_ 2 _n_ for _i_ _∈_ _V_ 1 and _wi_ = _−_ 1 _/_ 2 _n_ for _i_ _∈_ _V_ 2.

Without misclassification:


2 _n_

- _xi_


_i_ = _n_ +1


2 _n_


cos _θ_ =


2 _n_


- _xiwi_ = ~~_√_~~ 1


_i_ =1


2 _n_


- _n_

 - _xi −_


_i_ =1


To maximize cos _θ_ under exactly _k_ misclassifications, the optimal strategy places errors among entries
with smallest magnitudes. Specifically, vertices _{n −_ _k_ + 1 _, . . ., n}_ from _V_ 1 are misassigned to _V_ 2,
while vertices _{n_ + 1 _, . . ., n_ + _k}_ from _V_ 2 are misassigned to _V_ 1, yielding:


_n_

 - _xi_ +


_i_ = _n−k_ +1


_n_ + _k_

- _xi −_


_i_ = _n_ +1


_n_ + _k_


2 _n_

 - _xi_


_i_ = _n_ + _k_ +1


(9)


1
cos _θ_ _≤_ ~~_√_~~

2 _n_


- _n−k_

 - _xi −_


_i_ =1


_n_


This bound is achieved by the assignment _x_ 1 = _· · ·_ = _xn−k_ = 1 _/_ ~~�~~ 2( _n −_ _k_ ), _xn−k_ +1 = _· · ·_ =
_xn_ + _k_ = 0, and _xn_ + _k_ +1 = _· · ·_ = _x_ 2 _n_ = _−_ 1 _/_ �2( _n −_ _k_ ), which satisfies the normalization constraint
and yields cos _θ_ = _[√]_ 1 _−_ _γ_ . Therefore _γ_ = sin [2] _θ_, confirming that Theorem 3.2 is sharp up to
constants.


3.3 STATISTICAL PROPERTIES OF THE SECOND EIGENVECTOR


Abbe et al. (2019) demonstrate that the second eigenvector can be approximated as _**w**_ **2** _≈_ _[A]_ _**[u]**_ **[2]**


Abbe et al. (2019) demonstrate that the second eigenvector can be approximated as _**w**_ **2** _≈_ _a−b_ **[2]** [with]

error bound _||_ _**w**_ **2** _−_ _[A]_ _a−_ _**[u]**_ _b_ **[2]** _[||][∞]_ [=] _[ o]_ [(1] _[/][√][n]_ [)]

The denominator _a −_ _b_ is irrelevant as _**w**_ **2** will be scaled to be a unit vector. Thus we now focus on
characterizing the distribution of _A_ _**u**_ **2** . For vertex _i ∈_ _V_ 1, the _i_ -th entry of _A_ _**u**_ **2** equals the difference
between the number of edges between _i_ and vertices in _V_ 1, and the number of edges between _i_ and
vertices in _V_ 2. Since each edge appears independently with probability _a/n_ (within-community) or
_b/n_ (between-community), this entry follows the distribution of a difference of two binomial random
variables. Specifically, let


_Y_ _∼_ Binomial( _n, a/n_ ) _−_ Binomial( _n, b/n_ ) (10)


Then each entry of _A_ _**u**_ **2** is distributed as _Y_ or _−Y_ with equal probability, depending on whether
_i ∈_ _V_ 1 or _i ∈_ _V_ 2.


3.4 APPLYING CHERNOFF BOUNDS TO RELATE _γ_ AND sin _θ_


Building on the optimization framework above, while we know the approximate distribution of the _xi_
entries, direct analysis remains computationally intractable. Instead, we leverage constraints derived
from Chernoff concentration inequalities applied to this distribution. The Chernoff bound states that
for a random variable _X_ with moment generating function _M_ ( _t_ ):


_P_ ( _X_ _≥_ _a_ ) _≤_ _M_ ( _t_ ) _e_ _[−][ta]_ _∀a, ∀t >_ 0


This bound becomes increasingly sharp in the tail regions for large values of _a_ . For approximately
bell-shaped distributions, Chernoff bounds at multiple points constrain the distribution’s tail behavior,
effectively providing lower bounds on how ”concentrated” the distribution must be around its center.


Applied to our ordered sequence _x_ 1 _≥_ _x_ 2 _≥· · · ≥_ _xn_, these concentration properties impose lower
bounds on the decay rates between consecutive entries:


_[A]_ _a−_ _**[u]**_ _b_ **[2]** _[||][∞]_ [=] _[ o]_ [(1] _[/][√][n]_ [)]


_xn_


5


Define _pa_ = _a/n_, _qa_ = 1 _−_ _pa_, _pb_ = _b/n_, _qb_ = 1 _−_ _pb_, and the optimal Chernoff parameter


Figure 4: _γ_ as a function of sin _θ_ for various approaches


Figure 4a presents our experimental validation results for _n_ = 500 _, a_ = 0 _._ 06 _n, b_ = 0 _._ 04 _n_ . The red
points represent the relationship from Theorem 3.2, while the blue points show the actual optimization
results under our Chernoff-derived constraints. The blue line displays our theoretical prediction from
Equation 11, fitted to the optimization data using ordinary least squares (OLS) regression to account
for the unit normalization of the _xi_ vector.


The results demonstrate that our Chernoff-based analysis yields significantly tighter bounds than the
original theorem. For any given value of sin _θ_, our approach provides a substantially lower upper
bound on the achievable error rate _γ_ . Furthermore, the close agreement between the blue line and
blue points confirms the accuracy of our theoretical prediction in Equation 11.


3.5 MONTE-CARLO SIMULATION AND NORMAL APPROXIMATION TO RELATE _γ_ AND sin _θ_


Given the distribution in Equation 10, we can directly generate samples of the _xi_ entries using Monte
Carlo methods, removing the need for numerical optimization. With the _xi_ values generated from
their distribution, Equation 9 provides the maximum cos _θ_ for any given error level _k_ .


6


_t_ _[∗]_ = [1] 2 [ln] - _pqaapqbb_


. Let the concentration constant be:


_n_


_p_ [3] _aqb_ [3] + _qaqb_ + _papb_
_qapb_








[1] [+] _[ √][q][a][q][b]_ [)][2] _[n]_ [ +] [1]

2 [(] _[√][p][a][p][b]_ 2





_C_ = [1]


2





_qa_ [3] _p_ [3] _b_ +
_paqb_


The Chernoff concentration inequalities translate into the following optimization constraints:


_x_ [2] 1 [+] _[ · · ·]_ [ +] _[ x]_ 2 [2] _n_ _[≤]_ [1]

_xi_ +1 _≤_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln(] _[i]_ [ + 1)] _xi_ _∀i_ = 1 _, . . ., n −_ 1

ln _C_ + ln(2 _n_ + 1) _−_ ln _i_

_xi_ _≥_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln(2] _[n]_ [ + 1] _[ −]_ _[i]_ [)] _xi_ +1 _∀i_ = _n_ + 1 _, . . .,_ 2 _n −_ 1

ln _C_ + ln(2 _n_ + 1) _−_ ln(2 _n −_ _i_ )


The complete derivation appears in the appendix. Since _C_ is known before any optimization, these
constraints together with Equation 9 as the objective function define a convex optimization problem.
We solve this optimization problem umerically to find the maximum value of cos _θ_ subject to the
above constraints. Our theoretical analysis predicts this maximum should satisfy (proof in the
appendix):


_√_


(11)


cos _θ_ _≤_


2 _n_ �ln _C_ + 1 + ln [2 +] _n_ [1]
_t_ _[∗]_ [(1] _[ −]_ _[γ]_ [)] 1 _−_ _γ_


_n_
1 _−_ _γ_


(a) _γ_ as a function of sin _θ_ : Theorem 3.2 and
Chernoff-derived bounds


(b) _γ_ as a function of sin _θ_ : Chernoff-derived
bounds and Monte Carlo / Normal approximations


While we could compute Equation 9 directly using the exact probability density function, this
approach is algebraically intractable. Instead, we use a normal approximation to simplify the analysis.
The binomial distributions in our model satisfy the standard approximation conditions: both _np ≥_ 20
and _n_ (1 _−_ _p_ ) _≥_ 20 hold for our parameter ranges, so that the approximation is reasonable.


Under this normal approximation, the difference of binomials _Y_ also approaches normality, and
consequently each entry _Xi_ becomes approximately normal. This normality assumption enables
us to derive a closed-form theoretical prediction for the performance bound. Using the normal
approximation and the structure of our optimization problem, we obtain the following theoretical
prediction (with derivation provided in the appendix):


where _ϕ_ and Φ denote the standard normal probability density function and cumulative distribution
function, respectively. In the derivation above, we assumed that the entries _xi_ follow a standard
normal distribution with mean 0 and unit variance. While the zero-mean assumption is valid, the
unit variance assumption is not. The actual entries will have a different variance determined by the
underlying binomial distributions and the problem parameters. However, since the final vector must
satisfy the normalization constraint [�] _x_ [2] _i_ [=] [1][, the entries will be appropriately scaled regardless]
of their original variance. The theoretical prediction in Equation 12 captures the correct functional
relationship between _γ_ and cos _θ_, but with a scaling factor that depends on the actual variance of the
entries.


Figure 4b presents our experimental validation using the same parameters as before: _n_ = 500,
_a_ = 0 _._ 06 _n_, _b_ = 0 _._ 04 _n_ . We conducted Monte Carlo simulations with 50 repetitions to minimize
random variation in our results. The green points represent the (sin _θ, γ_ ) pairs computed from each
simulation run, forming a ”band” due to the natural clustering of results across repetitions. The green
dashed line shows our theoretical prediction from Equation 12, fitted to the simulation data using
OLS regression to account for the normalization constraint. For comparison, we include the blue
points from our earlier Chernoff-based analysis (Section 3.4, Figure 4a). The results validate several
important aspects of our theoretical framework:


First, the close agreement between the green dashed line and the simulation points confirms that our
normal approximation in Equation 12 accurately captures the underlying relationship between error
rate and spectral alignment.


Next, the green band lies well below the blue points, demonstrating that while our Chernoff-derived
bounds are mathematically sound, they remain conservative estimates. The gap between these
approaches becomes particularly pronounced for small error rates, precisely the region most relevant
for practical applications. This suggests that the Chernoff bounds, though tight in a worst-case sense,
do not fully capture the distributional properties that emerge in typical use cases.


Perhaps most significantly, both our simulation and Chernoff analysis reveal that perfect community
recovery ( _γ_ = 0) is achievable even when the eigenvectors _**u**_ **2** and _**v**_ **2** are not perfectly aligned
(sin _θ_ _>_ 0). This indicates that the spectral method’s success depends not merely on eigenvector
alignment, but more fundamentally on whether the entry distribution of _**v**_ **2** preserves sufficient
structure to enable correct partitioning. In other words, the distributional shape of the eigenvector
entries often contains enough information to guarantee perfect classification, even in the presence of
some spectral distortion.


4 COMPARING THEORETICAL PREDICTIONS WITH SPECTRAL ALGORITHM
RESULTS


While the results in Section 3 significantly improve upon the original bounds in Theorem 3.2, all
our theoretical analyses rely on the distributional approximation given in Equation 10. As noted
previously, this approximation contains errors that, while decreasing as _O_ (1 _/_ _[√]_ _n_ ), may still affect
the accuracy of our predictions for finite sample sizes.


7


2     -     -     - 1 _−_ _γ_
cos _θ_ _≤_ ~~_√_~~ 2 _ϕ_ _−_ Φ _[−]_ [1]

2 _n_ [(2] _[n]_ [ + 1)] 2 + 1 _/n_


�� - - 1

_−_ _ϕ_ _−_ Φ _[−]_ [1]
2 + 1 _/n_


���
(12)


To validate our theoretical framework against the actual spectral algorithm performance, we conduct direct experiments on randomly generated graphs. We generate stochastic block model
instances with edge probabilities _a_ = 0 _._ 06 _n_ and _b_ = 0 _._ 04 _n_ across a range of graph sizes
_n_ _∈{_ 500 _,_ 525 _,_ 550 _, . . .,_ 1000 _}_ . For each instance, we apply our modified **Spectral** **Partition**
algorithm (omitting the degree-based deletion step) and evaluate both the error rate _γ_ (comparing the
algorithm’s partition against the true community structure) and _θ_ (the angle between the true second
eigenvector _**u**_ **2** and the computed approximation to second eigenvector _**v**_ **2** ).


Furthermore, to provide comprehensive validation across different problem scales, we repeated all
the analyses from Section 3 for the complete range of graph sizes _n ∈{_ 500 _, . . .,_ 1000 _}_, rather than
limiting our evaluation to _n_ = 500. These results, including both the Chernoff-based optimization
bounds and the Monte Carlo simulation predictions, are consolidated alongside the direct spectral
algorithm experiments in Figure 5.


The figure uses opacity to represent graph size, with _n_ = 500 shown as nearly transparent points and
_n_ = 1000 as fully opaque points, creating a visual gradient across problem scales. Different colors
distinguish the various analytical approaches:


**Red Points (Theoretical Baseline):** These represent the quadratic bound from Theorem 3.2. Since
this bound follows the relationship _γ_ = sin [2] _θ_, which is independent of _n_, the red points of different
opacities overlap completely, forming a single curve.


**Blue Points (Chernoff Analysis):** These show our Chernoff-derived bounds from Section 3.4. As
_n_ increases, the achievable frontier moves upward, indicating that the bounds become less tight for
larger graphs. This behavior reflects the conservative nature of concentration inequalities for finite
sample sizes.


**Green Points (Monte Carlo Simulation):** These represent our normal approximation approach
validated through simulation, with 10 repetitions per value of _n_ . Similar to the Chernoff bounds, the
frontier shifts upward with increasing _n_, particularly in the low- _γ_ regime.


**Orange Points and Purple Fit (Direct Algorithm Results):** The orange points show the actual
performance of our modified **Spectral Partition** algorithm on randomly generated graphs. To these
experimental results, we fit the empirical relationship:


using OLS regression, with the resulting fitted curve displayed as the purple line.


**Theoretical Significance:** The functional form in Equation 13, combined with the claims of Theorems
2.2 and 3.1, directly yields the final result stated in Theorem 1.3, thus bridging our empirical
observations with the theoretical framework.


4.1 SCALING BEHAVIOR AND CONVERGENCE ANALYSIS


Several important trends emerge as _n_ increases while maintaining constant ratios _a/n_ and _b/n_ . The
community detection problem becomes inherently easier for larger graphs, as predicted by both
Theorem 1.3 and Theorem 3.2, which allow for smaller error rates _γ_ as their left-hand sides increase.
This theoretical prediction is confirmed in our results, where larger _n_ values (higher opacity points)
consistently achieve lower _γ_ values.


More significantly, the gap between the orange points (direct algorithm results) and green points
(simulation predictions) of matching opacity decreases with increasing _n_ . This convergence validates
the error bound which asserts that approximation errors decrease as _O_ (1 _/_ _[√]_ _n_ ). The observed
convergence demonstrates that for large _n_ in the low- _γ_ regime, the relationship in Equation 13 and
our theoretical prediction in Equation 12 align closely.


This convergence provides strong empirical support for our central claim: **Spectral Partition** alone
achieves near information-theoretic performance without requiring the additional **Correction** step,
particularly as problem size increases and error rates decrease, precisely the regime most relevant for
practical applications.


8


_C_
sin _θ_ =


[�] 4


(13)
log 2 _/γ_


Figure 5: _γ_ as a function of sin _θ_ for various approaches including experimental results


5 CONCLUSION AND FUTURE WORK


We demonstrate that the spectral algorithm achieves near information-theoretic performance, through
elimination of degree-based preprocessing and the correction step. Our theoretical analysis through
Chernoff bounds, normal approximations, and Monte Carlo validation shows that spectral partition alone can achieve the inverse-logarithmic error rates previously thought to require additional
correction steps.


Experimental validation across varying graph sizes confirms that our theoretical predictions become
increasingly accurate as the error goes down with _O_ (1 _/_ _[√]_ _n_ ), with the empirical relationship sin _θ_ =
_C/_ [�][4] log 2 _/γ_ bridging our results to established theoretical frameworks. The convergence between

multiple analytical approaches in the large- _n_, low- _γ_ regime validates our central finding: spectral
partition alone suffices for near-optimal community recovery.


These results challenge the assumption that algorithmic complexity improves performance, suggesting
instead that careful theoretical analysis can reveal hidden strengths in existing methods. This ”less is
more” principle may have broader implications for spectral algorithm design.


Several directions emerge from this research: extending our analysis to unbalanced and multicommunity cases, analyzing multiple samples derived from the same distributions, developing
enhanced inference procedures, investigating computational scaling for massive graphs, analyzing
robustness under model misspecification, establishing precise connections to information-theoretic
limits, and exploring whether similar simplifications yield improvements in related spectral problems
such as graph clustering and matrix completion. The statistical independence between matrix
and vector entries preserved by our approach should facilitate these future investigations, as this
independence structure can be leveraged for more sophisticated statistical inference and analysis
techniques that would be complicated or impossible under the dependencies introduced by traditional
preprocessing steps.


6 REPRODUCIBILITY STATEMENT


To ensure reproducibility, we provide complete implementation details with specified parameters:
graph sizes _n_ _∈{_ 500 _, . . .,_ 1000 _}_, edge probabilities _a_ = 0 _._ 06 _n_ and _b_ = 0 _._ 04 _n_, and our modified
Spectral Partition algorithm that eliminates the degree-based deletion step. Monte Carlo simulations
use 50 repetitions for distributional analysis and 10 repetitions for scaling experiments. All random
seed numbers are initialized to ensure total reproducibility. Our submitted code includes scripts to
regenerate all figures and numerical results, with complete theoretical derivations provided in the
appendix.


9


REFERENCES


Emmanuel Abbe, Jianqing Fan, Kaizheng Wang, and Yiqiao Zhong. Entrywise eigenvector analysis
of random matrices with low expected rank, 2019. [URL https://arxiv.org/abs/1709.](https://arxiv.org/abs/1709.09565)
[09565.](https://arxiv.org/abs/1709.09565)


Bela Bollob´ as.´ _Random Graphs_ . Cambridge Studies in Advanced Mathematics. Cambridge University
Press, 2 edition, 2001.


Thang Nguyen Bui, Soma Chaudhuri, Frank Thomson Leighton, and Michael Sipser. Graph bisection
algorithms with good average case behavior. _Combinatorica_, 7:171–191, 1984. [URL https:](https://api.semanticscholar.org/CorpusID:32346819)
[//api.semanticscholar.org/CorpusID:32346819.](https://api.semanticscholar.org/CorpusID:32346819)


Peter Chin, Anup Rao, and Van Vu. Stochastic block model and community detection in the sparse
graphs: A spectral algorithm with optimal rate of recovery, 2015. [URL https://arxiv.org/](https://arxiv.org/abs/1501.05021)
[abs/1501.05021.](https://arxiv.org/abs/1501.05021)


Amin Coja-Oghlan. Graph partitioning via adaptive spectral techniques. _Combinatorics,_ _Proba-_
_bility and Computing_, 19:227 – 284, 2009. [URL https://api.semanticscholar.org/](https://api.semanticscholar.org/CorpusID:355743)
[CorpusID:355743.](https://api.semanticscholar.org/CorpusID:355743)


Martin E. Dyer and Alan M. Frieze. The solution of some random np-hard problems in polynomial
expected time. _J. Algorithms_, 10:451–489, 1989. [URL https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:13419364)
[org/CorpusID:13419364.](https://api.semanticscholar.org/CorpusID:13419364)


Zoltan F´ uredi and John Komlos.¨ The eigenvalues of random symmetric matrices. _Combinatorica_, 1:
233–241, 1981. [URL https://api.semanticscholar.org/CorpusID:7847476.](https://api.semanticscholar.org/CorpusID:7847476)


Michael Krivelevich and Van H. Vu. On the concentration of eigenvalues of random symmetric
matrices, 2000. [URL https://arxiv.org/abs/math-ph/0009032.](https://arxiv.org/abs/math-ph/0009032)


Frank McSherry. Spectral partitioning of random graphs. _Proceedings 2001 IEEE International Con-_
_ference on Cluster Computing_, pp. 529–537, 2001. [URL https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:10389217)
[org/CorpusID:10389217.](https://api.semanticscholar.org/CorpusID:10389217)


Anderson Y. Zhang and Harrison H. Zhou. Minimax rates of community detection in stochastic block
models, 2015. [URL https://arxiv.org/abs/1507.05313.](https://arxiv.org/abs/1507.05313)


A APPENDIX


A.1 PROOF OF THEOREM 2.2


_Proof._ The matrix _A_ has entries _Aij_ that are sampled from a Bernoulli distribution with success
probability _pij_ where _pij_ = _a/n_ if _i, j_ belong to the same community, and _pij_ = _b/n_ otherwise.
Therefore, the entries of matrix _M_ have mean zero and variance _σij_ [2] [=] _[ p][ij]_ [(1] _[ −]_ _[p][ij]_ [)] _[ ≤]_ _[σ]_ [2][ where] _[ σ]_ [2]
is the maximum variance of a single element.


Because 0 _< b < a < n/_ 2 we have:


Let _λ_ 1( _M_ ) be the largest eigenvalue of _M_ . Because _M_ is real-valued and symmetric, _λ_ 1( _M_ ) = _||M_ _||_ .
Now we use the result from Furedi & Komlos (1981) to determine¨ E[ _λ_ 1( _M_ )]. Since all entries have
mean zero and variance at most _σ_ [2], we have:


E[ _λ_ 1( _M_ )] = 2 _σ_ _[√]_ _n_ + _O_ ( _n_ [1] _[/]_ [3] log _n_ ) (15)

For large enough _n_, the first term dominates. So E[ _λ_ 1( _M_ )] = _O_ ( _σ_ _[√]_ _n_ ). Note: Furedi & Komlos¨
(1981) uses the premise that all entries have mean zero and common variance, but Krivelevich & Vu
(2000) showed that the assumption of common variance can be relaxed to _V ar_ [ _Mij_ ] _≤_ _σ_ [2] .


10


     - _a_
_σij_ [2] _[≤]_ _[σ]_ [2] [= max]
_n_ [(1] _[ −]_ _n_ _[a]_


_[b]_

_n_ [(1] _[ −]_ _n_ _[b]_


_[a]_ _[b]_

_n_ [)] _[,]_ _n_


_[b]_ - = _[a]_

_n_ [)] _n_


_n_


_≤_ _[a]_ [ +] _[ b]_ (14)

_n_


1 _−_ _[a]_

_n_


Next, also according to Krivelevich & Vu (2000), there are positive constants _c_ and _K_ such that for
any _t > K_,


_P_ [ _|λ_ 1( _M_ ) _−_ E[ _λ_ 1( _M_ )] _| ≥_ _t_ ] _≤_ _e_ _[−][ct]_ [2] (16)


Combining equations 15 and 16, there is a constant _C_ 2 such that for large enough _b_ (and consequently
_a, n_ ), we have with probability 1 _−_ _o_ (1):


A.2.3 CONVERTING BOUNDS TO OPTIMIZATION CONSTRAINTS


Now we connect this probabilistic bound to our optimization problem. If _xi_ is the _i_ -th largest element
in our sorted vector, and assuming the entries follow the theoretical distribution reasonably well, then


11


_||M_ _|| ≤_ _C_ 2 _σ_ _[√]_ _n ≤_ _C_ 2


which completes the proof for Theorem 2.2.


_√_

_a_ + _b_ _√_
~~_√_~~ _n_ (17)
_n_


A.2 PROOF OF FORMULATION AND PREDICTION FROM SECTION 3.4


A.2.1 DERIVING THE MOMENT GENERATING FUNCTION


We start by computing the moment generating function (MGF) for our random variables. Recall that
_Y_ represents the difference between two binomial distributions. The MGF of _Y_ is:


_MY_ ( _t_ ) = ( _qa_ + _pae_ _[t]_ ) _[n/]_ [2] ( _qb_ + _pbe_ _[−][t]_ ) _[n/]_ [2]


Since _−Y_ has MGF _M−Y_ ( _t_ ) = _MY_ ( _−t_ ), and each entry _Xi_ of our vector is equally likely to be _Y_
or _−Y_, the MGF of _Xi_ becomes:

_MXi_ ( _t_ ) = _[M][Y]_ [ (] _[t]_ [) +] 2 _[ M][Y]_ [ (] _[−][t]_ [)]

= [(] _[q][a]_ [ +] _[ p][a][e][t]_ [)] _[n/]_ [2][(] _[q][b]_ [ +] _[ p][b][e][−][t]_ [)] _[n/]_ [2][ + (] _[q][a]_ [ +] _[ p][a][e][−][t]_ [)] _[n/]_ [2][(] _[q][b]_ [ +] _[ p][b][e][t]_ [)] _[n/]_ [2]

2


A.2.2 APPLYING CHERNOFF BOUNDS


The Chernoff bound gives us:


_P_ ( _Xi_ _≥_ _a_ ) _≤_ _MXi_ ( _t_ ) _e_ _[−][at]_ _∀t >_ 0


This inequality holds for any positive _t_, but we want to choose the value that gives us the tightest
bound. For positive values of _a_, the distribution is dominated by the _Y_ component rather than the
_−Y_ component. The optimal choice turns out to be:


   - _paqb_

_t_ _[∗]_ = [1]

2 [ln] _qapb_


Note that _t_ _[∗]_ _>_ 0 because we assume _pa_ _>_ _pb_ (within-community edges are more likely than
between-community edges). Substituting this optimal value, we get:


_P_ ( _Xi_ _≥_ _a_ ) _≤_ _Ce_ _[−][at][∗]_


where the constant _C_ depends only on the model parameters _n_, _a_, and _b_ :


_n_


_p_ [3] _aqb_ [3] + _qaqb_ + _papb_
_qapb_








[1] [+] _[ √][q][a][q][b]_ [)][2] _[n]_ [ +] [1]

2 [(] _[√][p][a][p][b]_ 2





_C_ = [1]


2





_qa_ [3] _p_ [3] _b_ +
_paqb_


the probability that a random entry exceeds _xi_ should be approximately 2 _ni_ +1 [(since] _[ i]_ [ entries are]
larger than _xi_ out of 2 _n_ + 1 total positions). Therefore:


_i_
2 _n_ + 1 _[≤]_ _[C][ ·][ e][−][t][∗][x][i]_


Solving for _xi_ :

_xi_ _≤_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln] _[ i]_

_t_ _[∗]_


For the negative tail (when _i > n_ ), we use the symmetry of the bounds with _t_ replaced by _−t_, giving
us:

_xi_ _≥−_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln(2] _[n]_ [ + 1] _[ −]_ _[i]_ [)]

_t_ _[∗]_


A.2.4 FORMULATING THE COMPLETE OPTIMIZATION PROBLEM


Since all these quantities are known given the model parameters _n_, _a_, and _b_, we can incorporate them
into our optimization framework. However, we also need to ensure the resulting vector has unit norm.
We introduce the following constraints:


_x_ [2] 1 [+] _[ · · ·]_ [ +] _[ x]_ 2 [2] _n_ _[≤]_ [1]

_xi_ +1 _≤_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln(] _[i]_ [ + 1)] _xi_ _∀i_ = 1 _, . . ., n −_ 1

ln _C_ + ln(2 _n_ + 1) _−_ ln _i_


_xi_ _≥_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln(2] _[n]_ [ + 1] _[ −]_ _[i]_ [)] _xi_ +1 _∀i_ = _n_ + 1 _, . . .,_ 2 _n −_ 1

ln _C_ + ln(2 _n_ + 1) _−_ ln(2 _n −_ _i_ )


A.2.5 WHY THIS FORMULATION WORKS


Let us elaborate why this setup correctly captures our intentions:

First, regarding the normalization constraint [�] _x_ [2] _i_ _[≤]_ [1][:] [We use an inequality rather than equality]
to make this a convex optimization problem, which can be solved efficiently. However, the optimal
solution� _x_ 2 _i_ _[<]_ [ 1] will [, we can scale it up by some factor] automatically satisfy [�] _x_ [2] _i_ [=] _[ λ >]_ [1][.] [ 1][Here’s][ to get] _[ λ]_ [why:] **[x]** [ with][if] [ �][we][(] _[λx]_ [have] _[i]_ [)][2][a][= 1][feasible][.] [Since our objective][vector] **[x]** [with]
function cos _θ_ is positive (by construction) and linear in the entries, scaling up only improves the
objective value. Therefore, the optimizer will naturally choose the boundary case where the constraint
becomes tight.


Second, regarding the ratio constraints: The Chernoff bounds fundamentally limit how quickly the
entries can decay as we move from the largest to the smallest values. The ratio constraints enforce that
consecutive entries cannot decay faster than what the Chernoff bounds would allow. Specifically, all
entries _x_ 2 _, . . ., xn_ are constrained relative to _x_ 1 through these ratios, and all entries _xn_ +1 _, . . ., x_ 2 _n−_ 1
are constrained relative to _x_ 2 _n_ .


If some of these ratio constraints become strict (meaning the actual ratios are smaller than the bounds
allow), this doesn’t violate our theoretical framework—it simply means the actual distribution has
even better concentration than our worst-case analysis predicts. Combined with the normalization
argument above, the optimizer will find the largest possible _x_ 1 and smallest possible _x_ 2 _n_ (in absolute
value) such that the vector has unit norm, while respecting the decay rates imposed by the Chernoff
bounds.


A.2.6 DERIVING CUMULATIVE SUM APPROXIMATIONS


Starting from our Chernoff-derived bound:


_xi_ _≤_ [ln] _[ C]_ [ + ln(2] _[n]_ [ + 1)] _[ −]_ [ln] _[ i]_

_t_ _[∗]_


12


We want to approximate the partial sums _sj_ = [�] _i_ _[j]_ =1 _[x][i]_ [.] [Applying our bound:]


Which proves Equation 11. This bound represents our theoretical prediction for the maximum
achievable cos _θ_ under the Chernoff-derived constraints.


13


[ ln] _[ C]_

+ [1]
_t_ _[∗]_ _t_ _[∗]_


_t_ _[∗]_


_sj_ _≤_


_j_


_i_ =1


ln _C_ + ln(2 _n_ + 1) _−_ ln _i_


_n_ + 1) _−_ ln _i_

= _[j]_ [ ln] _[ C]_
_t_ _[∗]_ _t_ _[∗]_


_j_

- ln - 2 + 1 _/n_

_i/n_
_i_ =1


For large _n_, we can approximate the discrete sum with a continuous integral:


- - 2 _n_ + 1 - ln _C_ + ln + 1
_j_


_sj_ _≃_ _[j]_ [ ln] _[ C]_


_t_ _[∗]_


_[ C]_

+ _[n]_
_t_ _[∗]_ _t_ _[∗]_


- _j/n_ - 2 + 1 _/n_

ln
0 _x_


_dx_ = _[j]_

_t_ _[∗]_


Note that this approximation is only accurate when _j_ is large, which will be the case for our intended
application where we consider _j_ = _n −_ _k_ with small _k_ .


A.2.7 APPLYING THE APPROXIMATION TO OUR OBJECTIVE FUNCTION


Recall that our objective function is (from Equation 9):


_n_

 - _xi_ +


_i_ = _n−k_ +1


_n_ + _k_

- _xi −_


_i_ = _n_ +1


_n_ + _k_


2 _n_

 - _xi_


_i_ = _n_ + _k_ +1


2 _n_


1
cos _θ_ _≤_ ~~_√_~~

2 _n_


- _n−k_

 - _xi −_


_i_ =1


We make two key observations about the optimal solution structure:


First, due to the symmetry of our distribution and constraints, the entries exhibit approximate
symmetry around the center: _xn_ + _i_ _≃_ _xn_ +1 _−i_ for _i_ = 1 _, . . ., n_ . This allows us to simplify our
expression:


_n_

 - _xi_


_i_ = _n−k_ +1


2
cos _θ_ _≤_ ~~_√_~~

2 _n_


- _n−k_

 - _xi −_


_i_ =1


_n_


Second, and more importantly, our Chernoff constraints only establish lower bounds on the ratios
_xxi_ +1 _i_ [.] [They don’t prevent the optimizer from making some entries arbitrarily small.] [In particular,]
nothing stops the optimizer from setting _xn−k_ +1 = _· · ·_ = _xn_ = 0 and concentrating all the ”budget”
(from the normalization constraint [�] _x_ [2] _i_ [= 1][) into] _[ x]_ [1] _[, . . ., x][n][−][k]_ [.]

Turns out this is exactly what happens in our experiments. The optimizer pushes the middle entries to
zero while maximizing the contribution from the largest entries, which doesn’t violate our requirement
that the entries must be ”at least this concentrated” according to the Chernoff bounds.


Therefore, our objective simplifies to:


2 _n_ _[s][n][−][k]_


2
cos _θ_ _≤_ ~~_√_~~

2 _n_


_n−k_


- _xi_ = ~~_√_~~ 2


_i_ =1


A.2.8 FINAL THEORETICAL PREDICTION


Since _k_ = _γn_ is small, _n −_ _k_ is large, making our integral approximation valid. Substituting our
approximation for _sn−k_ :


��


- - 2 _n_ + 1
ln _C_ + 1 + ln
_n −_ _k_


2
cos _θ_ _≤_ ~~_√_~~


2

_[≃]_ ~~_√_~~
2 _n_ _[s][n][−][k]_ 2


[(] _[n][ −]_ _[k]_ [)]
2 _n_ _[·]_ _t_ _[∗]_


_t_ _[∗]_


_√_


=


2 _n_ �ln _C_ + 1 + ln [2 +] _n_ [1]
_t_ _[∗]_ [(1] _[ −]_ _[γ]_ [)] 1 _−_ _γ_


_n_
1 _−_ _γ_


A.3 PROOF OF FORMULATION AND PREDICTION FROM SECTION 3.5


Under the assumption that the entries _xi_ follow a standard normal distribution, they effectively
partition the probability density function _ϕ_ into 2 _n_ + 1 equal quantile intervals. This means:


   - _i_    -    - _i_
_xi_ = Φ _[−]_ [1] 1 _−_ = _−_ Φ _[−]_ [1]
2 _n_ + 1 2 _n_ + 1


The partial sum becomes:


- _i_ Φ _[−]_ [1] - _j_

2 _n_ + 1
_j_ =1


_i_


_j_ =1


1 - _j/n_
_n_ [Φ] _[−]_ [1] 2 + 1 _/n_


_si_ =


_i_

- _xj_ = _−_


_j_ =1


= _−n_


For large _n_ and _i_, we can approximate this discrete sum with a continuous integral:


               - _i/n_                - _x_
_si_ _≈−n_ Φ _[−]_ [1]

0 2 + 1 _/n_


To evaluate this integral, we use substitution. Let:


_dx_


   - _x_   _u_ = Φ _[−]_ [1]
2 + 1 _/n_


Which means Φ( _u_ ) = 2+1 _x/n_ [, so] _[ x]_ [=] [(2 + 1] _[/n]_ [)Φ(] _[u]_ [)][ and] _[ dx]_ [=] [(2 + 1] _[/n]_ [)] _[ϕ]_ [(] _[u]_ [)] _[du]_ [.] [When] _[ x]_ [=] _n_ _[i]_ [,]

we have _u_ = Φ _[−]_ [1][ �] 2 _ni_ +1 - = _−xi_ . Substituting into our integral:


    - _−xi_
_si_ _≈−n_


_n_


_i_    
_u ·_ 2 + [1]
_−∞_ _n_


- - _−xi_
_ϕ_ ( _u_ ) _du_ = _−_ (2 _n_ + 1) _uϕ_ ( _u_ ) _du_

_−∞_


Making another substitution _v_ = _−u_ (so _dv_ = _−du_ ):


                  - _∞_
_si_ = (2 _n_ + 1) _vϕ_ ( _v_ ) _dv_

_xi_


Now we can evaluate this integral directly using the substitution _w_ = _v_ [2] _/_ 2 (so _dw_ = _vdv_ ):

    - _∞_    - _∞_    - _∞_ _[−][x]_ _i_ [2] _[/]_ [2]


- _∞_


1
_vϕ_ ( _v_ ) _dv_ = ~~_√_~~
_xi_


- _∞_


2 _π_


1
_ve_ _[−][v]_ [2] _[/]_ [2] _dv_ = ~~_√_~~
_xi_


2 _π_


_∞_ _i_ _[/]_ [2]

_e_ _[−][w]_ _dw_ = _[e]_ ~~_√_~~ _[−][x]_ [2]
_x_ [2] _i_ _[/]_ [2] 2 _π_


= _ϕ_ ( _xi_ )
2 _π_


Therefore _si_ _≈_ (2 _n_ + 1) _ϕ_ ( _xi_ ). Since both _n_ and _n_ _−_ _k_ are large (with _k_ small), our integral
approximation is valid for both _sn_ and _sn−k_ . Accounting for the symmetry in our problem, we have:


2
cos _θ_ _≤_ ~~_√_~~


2
~~_√_~~
2 _n_ [(] _[s][n][−][k][ −]_ _[s][n]_ [) =] 2


2 _n_ [(2] _[n]_ [ + 1)(] _[ϕ]_ [(] _[x][n][−][k]_ [)] _[ −]_ _[ϕ]_ [(] _[x][n]_ [))]


Substituting the explicit expressions:


           - _n −_ _k_            -            - 1 _−_ _γ_
_xn−k_ = _−_ Φ _[−]_ [1] = _−_ Φ _[−]_ [1]
2 _n_ + 1 2 + 1 _/n_


           - _n_
_xn_ = _−_ Φ _[−]_ [1]
2 _n_ + 1


This completes the proof of Equation 12.


- - 1
= _−_ Φ _[−]_ [1]
2 + 1 _/n_


14