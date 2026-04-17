000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 In this paper we study the Kernel Density Estimation (KDE) problem: Given a dataset P of n points in Euclidean space and a kernel K(*p, q*), prepare a low space data-structure that given a query q can quickly output a 1±ϵ approximation to µ = (Pp∈P K(p, q))/n. Recent advances have used tools from Locality Sensitive Hashing (LSH) and Approximate Nearest Neighbor (ANN) search to build KDE data-structures with query time *sublinear* in 1/µ and space linear in 1/µ, with Charikar et al. (2020) achieving the current best query time of ≈ 1/µ0.173 for the popular Gaussian kernel. Our main result is a data-structure with significantly improved query time ≈ 1/µ0.05 , at the expense of somewhat higher space complexity of ≈ 1/µ4.15. More generally, our techniques give the first known query time vs space tradeoffs for KDE: for any δ ≥ 0 we can design a KDE
data-structure with space with 1/µ1+δ dependence and query time with 1/µξ(δ)
dependence, where ξ(δ) is a non-increasing function of δ. Importantly for the linear space regime, i.e δ = 0, we obtain a query time of 1/µ0.1865, improving the non-adaptive KDE bound from Charikar et al. (2020) and nearly matching the bound of Charikar et al. (2020) with a significantly simpler analysis.

## 1 Introduction

Kernel Density Estimation (KDE) is a fundamental and widely studied problem in statistical learning theory and artificial intelligence (Fan, 2018; Scholkopf & Smola, 2002; Joshi et al., 2011; Arias- ¨
Castro et al., 2016). Formally KDE is defined as follows - Given ϵ > 0 and a dataset P of n points p1*, . . . ,* pn ∈ R
d, preprocess it into a small space data-structure that allows one to quickly approximate, given a query q ∈ R
d, the quantity

$$\mu^{*}=K({\mathcal{P}},{\mathbf{q}})={\frac{1}{|{\mathcal{P}}|}}\sum_{{\mathbf{p}}\in{\mathcal{P}}}K({\mathbf{p}},{\mathbf{q}}),$$

up to multiplicative 1 ± ϵ factor with probability 0.9, where the kernel function K(p, q) is a monotone decreasing function of ∥p − q∥. The Gaussian kernel,

$$(1)$$
$$K(\mathbf{p},\mathbf{q})=e^{-\|p-q\|_{2}^{2}/(2\sigma^{2})}$$
2), (2)
is a important example of a kernel widely used and will be the main focus of our paper, although many others (eg. Laplace, exponential, polynomial) are also used sometimes used (Shawe-Taylor & Cristianini, 2004; Williams & Rasmussen, 2006). Moreover recent works have used fast Gaussian KDE primitives for speeding up attention computation in modern transformer based LLMs (Zandieh et al., 2023; Indyk et al., 2025).

Unfortunately the exact algorithm for this problem does a linear scan over P at query time and thus runs in time linear in n, making it not scalable for large datasets. Thus most practical algorithms resort to reporting approximate kernel density evaluation at query time. In the low dimensional regime tree based algorithms Greengard & Strain (1991); Gray & Moore (2001); Gan & Bailis (2017) give efficient approximations, however their running times are exponential in d making them not scalable for high-dimensional datasets. In the rest of the paper, we use the notation µ
∗ defined as µ
∗:= K(P, q) to denote the true kernel density for a query q, and µ denotes a quantity that satisfies µ
∗ ≤ µ ≤ 4µ
∗, using standard techniques we can assume such a µ is known to us (see 1 Anonymous authors Paper under double-blind review

## Abstract

# Faster Kernel Density Estimation Via Hashing Based Time-Space Tradeoffs

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Remark 3 Charikar et al. (2020)). In the high dimensional regime d = Ω(log n) uniformly sampling O(1/(ϵ 2) · 1/µ) dataset points Pe from P and reporting K(Pe, q) at query time suffices to obtain a 1 ± ϵ estimate of the true kernel density K(P, q). The line of work initiated in the work of Charikar & Siminelakis (2017) improved upon random sampling by creating Gaussian KDE data-structures with *sublinear* in 1/µ query time and linear in 1/µ space. In the subsequent discussion all methods have polynomial in d and 1/ϵ dependence in the query time and space, so we suppress them for readability. Furthermore we use Oe(·) to hide polynomial factors in d and log(n, 1/µ). Charikar
& Siminelakis (2017) used *Locality sensitive hashing* (LSH) (Indyk & Motwani, 1998; Andoni & Indyk, 2008), a fundamental technique in the approximate nearest neighbor (ANN) literature, to design Gaussian KDE data-structures with query time with a 1/
√µ dependence. Following this a line of work had subsequent improvements using LSH culminating in the work of Charikar et al. (2020)
which achieved a 1/µ0.25 dependence in the query time using a data-independent LSH and 1/µ0.173 using a much more involved data-dependent LSH. These approaches used *symmetric* LSH constructions, and our main contribution is to use advances in *asymmetric* LSH constructions (Andoni et al., 2017; Razenshteyn, 2017) to improve upon these works. We first present an overview of our contributions followed by presenting our main ideas and techniques. Finally we end the section by discussing related work.

## 1.1 Our Contributions

Our first result, that obtains data-structures for Gaussian KDE problem (see problem setup in Equation 5) with significantly improved query time using asymmetric LSH, is as follows, Theorem 1. *(Informal) There exists a data-structure for the Gaussian KDE problem with expected* query time Oe((1/ϵ2) · 1/µ0.051) *and space* Oe((1/ϵ2) · 1/µ4.15). There also exists a data-structure for the Gaussian KDE with expected query time Oe((1/ϵ2) · 1/µ0.1865) *and space* Oe((1/ϵ2) · 1/µ).

The formal version of the above theorem is presented in Theorem 17. Of course we obtain the improved query time of 1/µ0.051 at the expense of polynomial in 1/µ space, however the use of asymmetric LSH allows us to tradeoff the space and query time of our data-structure. Thus even for the linear space, i.e. with 1/µ dependence in the space, we obtain a query time with 1/µ0.1865 dependence on 1/µ that beats the previous best bound of 1/µ0.25 using non-adaptive schemes. It is slightly worse than the data-dependent scheme of Charikar et al. (2020), which achieved a 1/µ0.173 dependence, however our scheme has the advantage of being much simpler. We also show a more general result that presents time-space tradeoffs for Gaussian KDE data structures, in the following Theorem 2. (Informal) For any δ ≥ 0 there exists a data-structure for the Gaussian KDE problem with query time Oe((1/ϵ2) · 1/µξ(δ)) and space Oe((1/ϵ2) · 1/µ1+δ) where ξ(δ) as a function of δ is presented in right figure in Figure 1. To the best of our knowledge, ours is the first such tradeoff for KDE, the formal version of the above theorem is presented in Theorem 16. We now describe the main techniques used to prove our results.

## 1.2 Technical Overview

Our query time vs space complexity tradeoffs for KDE are obtained by a novel instantiation of the framework of Charikar et al. (2020) that essentially reduces the KDE problem to a version of the Approximate Near Neighbor (ANN) problem. We thus start with an overview of that framework. KDE via (density constrained) approximate nearest neighbor search (ANN). Charikar et al. (2020) reduce the problem of computing kernel density problem at a query q to logarithmic many approximate nearest neighbor (ANN) problems with the additional twist provided by density constraints. The main idea is to partition points p ∈ P into a logarithmic number of distance scales according to the value of K(p, q), then estimate the number of points in each distance scale (i.e., at a certain Euclidean distance from q), using approximate nearest neighbor search techniques such as Locality-Sensitive Hashing (LSH). Using standard scaling techniques, as in Charikar et al. (2020, Assumption 1 in Section 5), we conveniently re-write the Gaussian kernel for any point p ∈ P as follows, K(p, q) = µ
∥p−q∥
2 2 ,
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 and we denote L
q j ∈ P denote the points in P with kernel value K(*p, q*) ≈ 2
−jfor all values1 of j ∈ [0, J] for J = log(1/µ). We denote the distance scale xj = j/J, which thus conveniently allows us to write L
q j ∈ P as all points with, K(*p, q*) ≈ µ xjfor xj ∈ [0, 1].

See Section 3 for precise definitions. The framework of Charikar et al. (2020) randomly samples points in P at rate pj = (1/µ)
1−xj· 1/n, (3)
to create a subsampled dataset, then retrieves all point in L
q jsurviving in this subsampled dataset using the symmetric LSH of Andoni & Indyk (2008). Our work proposes to go beyond symmetric LSH to achieve the improvement, so it is more convenient to reformulate the Charikar et al. (2020) framework as applying a more general Approximate Near neighbor (ANN) data-structure. Recall that a (*c, r*)-ANN data-structure is an efficient datastructure that, assuming the existence of a point at distance at most r from the query, returns a point at distance at most cr. When recovering points in L
q j
, i.e. at distance scale xj , from the sampled dataset we invoke a (c, r)-ANN data-structure with the near radius r corresponding to KDE contribution ≈ µ xj and the far radius cr corresponding to KDE contribution ≈ µ. We drop the subscript j from scale xj , since we will only work with scales.

Remark 3. Note that this classical guarantee that an (*c, r*)-ANN data-structure provides does not suit us, as we need to exactly retrieve all points at distance scale ≈ x from the sampled dataset, we will provide a new analysis of a powerful (*c, r*)-ANN data-structure that takes density constraints into account and achieves *exact* recovery efficiently. Exact recovery with approximate near neighbor search. Charikar et al. (2020) use the symmetric LSH of Andoni & Indyk (2008) for this ANN problem, to provably recover points at distance scale x ∈ [0, 1] in sublinear time. The query time of this procedure is higher than that of the ANN problem because we need to retrieve point at exactly distance the near distance scale x and during hashing, points at scale y for *x < y* ≤ 1 can collide with points at x, adding time needed in scanning and discarding these intermediate points. However this query time overhead can be controlled using density constraints - a simple Markov bound allows us to bound number of points at scale y ∈ [0, 1],
n(µ)
1−y ≪ n, (4)
and furthermore it is unlikely that all such points collide with points at x. Charikar et al. (2020) bound the additional query time overhead by upper bounding the expected number of intermediate colliding points by multiplying density constraint upper bounds and with LSH collision probability of Andoni & Indyk (2008). This gives them the query time for recovering points at scale x in the subsampled dataset for any fixed x. Summing this over log many possible values of x they obtain a query time of 1/µ0.25 up to log factors. Section 3 contains the precise details of this framework.

Our contribution: query time reduction via asymmetric ANN. Our main idea is to use the asymmetric LSH construction of Andoni et al. (2017) (see Section A) instead to recover points at scale x from the subsampled dataset. For the (*c, r*)-ANN problem, this LSH allows us to design datastructures with space n 1+ρs+o(1) and query time n ρq+o(1) for any ρs, ρq ≥ 0 under the constraint,
(c 2 + 1)√ρq + (c 2 − 1)√ρs ≥ 2c. (5)
Choosing ρs = ρq recovers the symmetric LSH of Andoni & Indyk (2008), but choosing it differently allows one to *tradeoff* lower query time for higher space for recovering points at scale x. This leads to an improvement over Charikar et al. (2020) because the maximum of query time in their reduction is achieved at a different distance scale x ∈ [0, 1] than the one that yields the space bound!

Finding the best ρs, ρq under constraint 5 for every x ∈ [0, 1] can be expressed as an optimization problem (see Section 4) and solved numerically (see Section 5). The exact optimum does not seem simple to obtain analytically, and we therefore resort to numerics. One interesting phenomenon emerges: unlike the (c, r)-ANN problem, which admits a solution with constant query time, the KDE tradeoffs that we achieve (see Fig. 1) do not yield a constant query solution. We next analytically show that this is not possible with present near neighbor search technology - an exciting open problem is to either prove a formal lower bound ruling out constant query KDE in polynomial space or bypass the inherent barrier in our scheme to get a KDE data structure with constant query time.

1More precisely, the set L
q Jfor J = log(1/µ) is defined to capture all points with kernel value K(p, q) =
O(µ) - the contribution of these points can be very easily estimated from a small sample.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Why constant query KDE is not possible with known ANN results. For a fixed scale x ∈ [0, 1]
the natural choice of the query exponent ρq is to set it to 0 to ensure that at least the expected number of points colliding from the last scale y = 1, i.e. points at far distance cr with kernel value ≈ µ, is at most n o(1). As otherwise any higher ρq will lead to non-negligible contribution of points at far distance cr, as the (*c, r*)-ANN problem will have a non-negligible query time. Thus ρq = 0 is the natural choice, however again the overall query time will be higher than that for the (*c, r*)-ANN problem because of collisions from points at intermediate scales y for *x < y* ≤ 1. We now give a high level overview of this additional overhead. Fix an x ∈ [0, 1] and recall from Equation 3 that first the dataset P is subsampled at rate (1/µ)
1−x· 1/n, leading to expected dataset size (1/µ)
1−x.

If we construct an asymmetric LSH for dataset size (1/µ)
1−xand ρq = 0, the probability for a point p at scale y for *x < y* ≤ 1 to be scanned during query time turns out to be,

$$\left(\frac{1}{\mu}\right)^{-\left(\frac{(y-x)^{2}}{y(1-x)}\right)+o(1)}.\tag{1}$$
$$(6)$$
$$\left(7\right)$$

From density constraints 4, number points at scale y is at most n · (µ)
1−y, which after subsampling gets reduced to (1/µ)
y−xin expectation. Thus overall the additional overhead due to points at scale y is (1/µ)
y−xtimes the bound in Equation 6, and since there only log many values of y to consider between [x, 1] the overall overhead in query time is the following up to log factors,

$$\max_{y\in[x,1]}\left(\frac{1}{\mu}\right)^{(y-x)-\left(\frac{(y-x)^{2}}{y(1-x)}\right)+o(1)},\tag{1}$$
y∈[x,1] 
µ
In the expression above for y = x and y = 1 the exponent is o(1), however near y = x the first linear term y−x grows faster than the second term behaving roughly quadratically as (y−x)
2. Thus for any fixed x ∈ [0, 1] the maximum happens for some point inside the interval [x, 1]. Furthermore since we need to recover points at logarithmic many scales x ∈ [0, 1], the overall query time of this KDE data-structure is max of the above over all x ∈ [0, 1], which using numerical methods is approximately (1/µ)
0.09. This in general conveys the fact that even using this asymmetric LSH for query exponent ρq = 0 for all x ∈ [0, 1], one cannot obtain arbitrarily small constant query time exponent at the expense of arbitrarily large polynomial space. However we can obtain a slightly better constant query time exponent than 0.09 by optimizing setting ρq for all x ∈ [0, 1]. For any x ∈ [0, 1] and a general ρq ≥ 0, Equation 6 is as follows,

$$\left(\frac{1}{y}\right)^{(1-x)\left(\rho_{q}-\frac{x}{y(1-x)^{2}}\left(\frac{y-x}{\sqrt{x}}-(y-1)\sqrt{\rho_{q}}\right)^{2}\right)+o(1)$$
µ
,
thus for a fixed x ∈ [0, 1] the overall query time by optimizing over valid ranges of ρq is as follows,

$x\in[0,1]$ the overall query time by optimizing over valid ranges of $\mu$: $$\min_{\text{valid}\rho_q}\max_{y\in[x,1]}\left(\frac{1}{\mu}\right)^{(y-x)+(1-x)\left(\rho_q-\frac{x}{y(1-x)^2}\left(\frac{y-x}{\sqrt{x}}-(y-1)\sqrt{\rho_q}\right)^2\right)+o(1)}$$
Finally the overall query time of our KDE data-structure is then the max of the above over all x ∈ [0, 1]. Solving this optimization problem leads to a query time roughly (1/µ)
0.05. The precise details of this parameter setting and the optimization formulation are in Section 4. Query time for space 1/µ. Obviously the space of the data-structure described previously is polynomial in 1/µ, roughly 1/µ4, thus making it incomparable with previous works that had space at most 1/µ. However since the asymmetric LSH allows us to flexibly set either the space or query exponents for each recovery problems, we can carefully choose the space exponent so that the overall space of our data-structure to be at most 1/µ. This restricts the choice of the query exponent for each recovery problem as per Equation 5 leading to a higher query time. Overall this results in a data independent KDE data-structure with space 1/µ and query time 1/µ0.1865, which improves over the data independent bound of 1/µ0.25 of Charikar et al. (2020). Moreover the query exponent is within 0.02 of the exponent of the data dependent data-structure of the work of Charikar et al. (2020),
which achieves a query time 1/µ0.173, however our analysis is arguably much simpler. In general, our construction allows one to smoothly tradeoff space and query time for KDE data-structures, and the details of this are presented in Section 5.

## 1.3 Related Work

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 There is a large body of work on sublinear time KDE for low dimensional spaces, which includes the classical work on Fast Gauss Transform (Greengard & Strain, 1991) and other tree based hierarchical partitioning methods (Gray & Moore, 2001; 2003; Yang et al., 2003; Lee et al., 2005; Ram et al., 2009; Gan & Bailis, 2017). For high dimensional spaces (d = Ω(log n)), sublinear time algorithms beating random sampling for various kernels such as Gaussian and polynomial were obtained by a recent sequence of works based on implementing importance sampling via LSH (Charikar & Siminelakis, 2017; Backurs et al., 2018; Charikar et al., 2020). These importance sampling based procedures had 1/ϵ2 dependence on ϵ in query complexity, and works based on discrepancy theory and randomized space partitioning (Phillips & Tai, 2020; Charikar et al., 2024) achieve a 1/ϵ dependence. Recent works (Siminelakis et al., 2019; Backurs et al., 2019) address scalability issues of the original approach of Charikar & Siminelakis (2017) and obtain practical improvements on real world datasets.

## 2 Preliminaries

The goal of this section is to present basic notation and assumptions used throughout the paper, as well as preliminary concepts and tools regarding KDE and (c, r)-ANN data-structures.

Notation. We denote expa(b) = a band let [n] = {1*, . . . , n*} for any natural number n.

## 2.1 Basic Setup

We now present standard assumptions on parameters as part of problem setup. We first define the Gaussian Kernel. Definition 4 (Gaussian Kernel). K(p, q) = e
−
log(1/µ)
2∥p−q∥
2. We use this version of the Gaussian Kernel because an instance with general Gaussian kernel with arbitrary bandwidth parameter as in Equation 2 can be reduced to this version using standard scaling techniques (Refer to Charikar et al. (2020, Assumption 1 in Section 5)).

Definition 5 (Setup). The approximation factor is ϵ = Ω(1/ polylog n) and µ
∗ = n
−Θ(1) and dimension d = O˜(1) (see Charikar et al. (2020, Remark 1)). We assume we know a baseline approximation µ satisfying µ
∗ ≤ µ ≤ 4µ
∗(see Charikar et al. (2020, Remark 3)).

Note that µ
∗ = n
−Θ(1) is the interesting regime for this problem because for µ
∗ = n
−ω(1) under the Orthogonal Vectors Conjecture (Rubinstein, 2018), the problem cannot be solved faster than n 1−o(1) using space n 2−o(1) (Charikar & Siminelakis, 2019), and for larger values µ
∗ = n
−o(1)
random sampling solves the problem in n o(1)/ϵ2time and space.

## 2.2 (C, R)-Ann On The Sphere

We now present the definition of the (*c, r*)-ANN problem.

Definition 6 (The (c, r)-ANN problem). Given an n-point dataset P ∈ R
d, the goal is to preprocess P to answer the following queries. Given a query point q ∈ X such that there exists a data point within distance r from q, return a data point within distance cr from q. The (*c, r*)-ANN problem on the sphere is defined similarly, with the assumption that the dataset P contains points that lie on the unit sphere. We now state the asymmetric LSH of Andoni et al. (2017) as described in Razenshteyn (2017) for the (*c, r*)-ANN problem on the sphere.

Theorem 7 ((*c, r*)-ANN parameters). Razenshteyn (2017, Theorem 2.8.1) Let ϵ0 > 0 be a fixed constant. For every c > 1,1 log log n ≤ r = o(1), and for every ρq, ρs ≥ 0, such that cr ≤ 2 − ϵ0 and

$$(c^{2}+1)\cdot{\sqrt{\rho_{q}}}+(c^{2}-1)\cdot{\sqrt{\rho_{s}}}\geq2c$$
√ρs ≥ 2c (8)
there exists a data-structure for (c, r)*-ANN on a unit sphere* S
d−1 ⊂ R
d *where* d = n o(1) for a set of size n*, with space* n 1+ρs+o(1)*, query time* n ρq+o(1) *and success probability* 1 −1 n10 .

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 We make two important remarks about this data-structure. The first, this data-structure is dataindependent (see Razenshteyn (2017)). Roughly, this feature makes the data-structure more straightforward compared to data-independent ones, as they do not make any use of (or assumptions on) the dataset for preprocessing. This simpler setting allows usually for a cleaner analysis (see for example the data-dependent/independent settings in Andoni et al. (2017); Charikar et al. (2020)). Secondly, we elaborate briefly on the query procedure Algorithm 4 of this data-structure. The basic object underlying this ANN data-structure is a tree, where each inner node contains random Gaussian vectors, and the leaves contain subsets of the processed input dataset. Importantly, querying the datastructure follows multiple paths in the tree, which are determined by the correlation of the query with the Gaussian vectors stored in the inner tree nodes. Every traversed path leaves to a leaf that contains multiple points from the original dataset. We often say that the union of all points in the reached leaves *collide* with the query. We elaborate on the data-structure's query/preprocessing algorithms as well as the parameter setting for the theorem above in Appendix A. We now state properties of a key reduction to reduce general instances to the unit sphere.

Lemma 8. There exists a reduction from (c, r)-ANN problem over the ℓ2 for n*-point dataset in* R
d, to (c, r′)-ANN on the sphere problem over the ℓ2 distance for n*-points on the unit sphere in* R
d+1 where r
′ =
r R
in which all *the points are mapped to a sphere of radius* R = r · log log n and then scaled by R into the unit sphere. The pairwise distances between points are preserved up to scaling by R *and an additive factor* O(1/(r
√log log n))*. This incurs an* n o(1) *query time overhead.*
Note that the reduction from the lemma above (Lemma 8) allows for recovering the *original* (c, r)-
ANN problem, hence the points recovered by the (*c, r*′)-ANN on the sphere are converted to points in the original dataset. This standard reduction was previously used in Razenshteyn (2017); Andoni et al. (2017), and we provide more details about it in Appendix A.1.

## 3 Framework For Non-Adaptive Kde

This induces corresponding distance levels: rj := max r : f(r) ∈ (2−j, 2
−j+1]	. Here f(r) :=
K(p, p
′) for r = ∥p − p
′∥. Also define L
q J+1 := P \ Sj∈[J] L
q j
.

In this section, we introduce and generalize the framework of Charikar et al. (2020) which "reduces" KDE to an ANN problem we refer to as the Level-j Recovery. In the following, we present the KDE data-structure in terms of a data-structure for the Level-j Recovery problem.

Throughout the rest of the section, we assume that we are given an approximation parameter ϵ and some baseline approximation µ as in the setup (Definition 5) and Gaussian kernel (Definition 4).

The first concept is that of geometric level sets.

Definition 9 (Geometric level sets). Let J = ⌈log2 1 µ
⌉. For any j ∈ [J] and a query q, define the level set:
L
q j
:= pi ∈ P : K(pi, q) ∈ (2−j, 2
−J+1]	.

Similarly to Charikar et al. (2020) we will sub-sample the dataset P at different geometric rates for each j ∈ [J], with the goal of recovering points from L
q j given the query q, and thus we need the following definition of a subsampled dataset and the Level-j Recovery problem. Definition 10. For j ∈ [J + 1], let Pj be the dataset achieved by sampling P at rate pj :=
min( 1 2 jnµ
, 1) for j ≤ J and pJ+1 =
1 n
. Let mj := 1 2 jµ be the expected size of Pj .

Definition 11 (Level-j Recovery data-structure). Given the sample Pj and a point q, recover all points in L
q jfrom Pj with probability at least 1 −1 n10 . A data-structure for the Level-j Recovery problem is parameterized by its space denoted space(j) and its query time denoted query(j). Remark 12. In the paper, we will construct data-structures for the sample Pj for j ∈ [J]. We ignore the last sampled set, PJ+1, which contains, in expectation, only a constant number of points in expectation, and hence requires constant query time and space. As in Charikar et al. (2020), the main technical work is dedicated to constructing efficient datastructures for the Level-j Recovery Dj , which we use in the algorithms below. We use our datastructure for j's that are within a *range* j ∈ [c0J,(1−c1)J] where c0, c1 can be set to any arbitrarily 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
- *KDE query (Algorithm 2) runs in expected time* Oeϵ
−2· maxj∈[J] query(j).

## 4 Data-Structure For The Level-J Recovery Problem

We now present our data-structure Dj for Level-j Recovery. Notice that for r ∈ [0,
√2],
(1/µ)
−r 2/2∈ [µ, 1], and so we can focus our attention on r's within that range (as for other values small constant, our data structure and details of it are in Section 4). Assuming the *nice* range c0, c1 is fixed, for x < c0*, x >* 1 − c1 we use the data-structure from Charikar et al. (2020) for the Level-j Recovery problem for these small j's. We provide the formal statement about the guarantee of this data-structure in Appendix B.2. Data-structure Description. We now describe the preprocessing and query procedures for the KDE data-structure based on those described in Charikar et al. (2020, Algorithms 1,2). Algorithm 1: KDE PREPROCESS
Input: dataset P, precision parameter ϵ, baseline approximation µ as in Definition 5, small constants c0, c1 ∈ (0, 1/2)
1 K ←
C log n ϵ 2· µ
−o(1).

2 for K *times* do 3 for j ← 1 to J do 4 Pj ← subsample of P at rate pj from Definition 10. 5 if j < c0 · J or j > (1 − c1)J **then** 6 Preprocess Pj using the data-structure from Lemma 27.

7 **else**
8 Preprocess Pj using our new data-structure Dj from Lemma 15.

9 Store a sampling of P with probability 1/n.

## Algorithm 2: Kde Query

Input: Query q (the repetition parameter K is as in Algorithm 1). Output: A 1 ± ϵ estimate for µ
∗.

1 for K *times* do 2 for j ← 1 to J + 1 do 3 Query the Level-j Recovery data-structure on q to recover points from L
q j
, for the relevant repetition.

4 S ← the set of all recovered points for the relevant repetition.

5 Calculate the estimate Z ←Pj∈[J]
Pp∈S∩Lq j K(p,q)
pj(where pj is defined in Definition 10) for the relevant repetition.

6 **return** the average of the estimations Z across all repetitions.

Query Time and Space Requirement. We now state the theorem from Charikar et al. (2020) which parametrizes the space used by Algorithm 1 and time of Algorithm 2. Theorem 13. Charikar et al. (2020, Theorems 15, 22) For Gaussian kernel K(p, q), precision parameter ϵ and baseline approximation µ *as in the setup (Definition 5), and assuming that for any* j ∈ [J] there exists a data-structure Dj for the Level-j *Recovery problem with expected query time* query(j) *and expected space requirement* space(j), then there exists a KDE data-structure that supports (1 ± ϵ)*-multiplicative factor approximation to the KDE value with the following parameters:* We cite the relevant claims justifying the above in Appendix B.3. Next we derive expressions for query(j) and space(j) for our data-strucutre Dj we use in Algorithms 1 and 2 for Gaussian Kernel.

- *KDE preprocessing (Algorithm 1) uses expected space* Oeϵ
−2· maxj∈[J] space(j).

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 of r, the contribution of points from these distances to the kernel value of any queried point amounts to o(1/µ)). Using the Gaussian Kernel in Definition 9 gives the distance level rj =p2j/J for each j ∈ [J]. We also use the distance scale xj = j/J, hence rj =p2xj . Setting up the (c, r)**-ANN problem on the sphere.** We will use a data-structure for (*c, r*)-ANN to solve Level-j Recovery. Our dataset will be the sample Pj (see Definition 10) with expected size mj = exp1/µ(1 − xj ). The near distance will be r =p2xj and far distance cr =
√2, thus c =p1/xj . We use the data-structure from Theorem 7 for (c, r)-ANN problem on the sphere, thus to use this first we transform our points to lie on the unit sphere Lemma 8 (see Appendix A.1 for full details). This reduction incurs certain considerations, the most important of which is that in the following we make the assumption that j lies within the *nice* range [c0J,(1 − c1)J] for some small constants c0, c1 ∈ (0, 1/2). In this range, j = O(J) and the size of the dataset is mj = (1/µ)
O(1).

These simplify our calculations, and have little influence since c0, c1 are chosen arbitrarily small. The query/space requirements of our data-structure. The data-structure for the (c, r)-ANN we use is as per Theorem 7. Our data-structure Dj will build on top of this data-structure as follows. The preprocessing will remain the same, and so is the space requirement. For the query procedure we apply the query procedure of the data-structure for (*c, r*)-ANN problem on the sphere (Algorithm 4) but go over all points in the leaves reached by the ANN-query procedure. We analyze the expected number of points from level sets L
q ifor i ̸= j that appear in the leaves of the data-structure for a given query q. We formally analyze it in the our main technical lemma in the appendix, Lemma 31, which gives a data-structure for the Level-j Recovery based on the data-structure for (*c, r*)-ANN problem on the sphere from Theorem 7 for any choice of ρq, ρs that satisfies Equation (8).

Restricting the space requirement. Since the data-structure for the (*c, r*)-ANN problem on the sphere from Theorem 7 is parameterized by ρq, ρs, we need to explain the specific choice of these parameters for our setting of the Level-j Recovery data-structure. For any δ ≥ 0, we choose to set the parameters so that the space requirement of the Level-j Recovery data-structure is bounded by exp1/µ(1 + δ + o(1)). This choice enforces a constraint on the space exponent ρs:
expmj
(1 + ρs + o(1)) ≤ exp1/µ(1 + δ + o(1)) (9)
and as a result, it also enforces a constraint on the query exponent ρq by the ANN-tradeoff in Equation (8). These constrains splits the range of xj ∈ [0, 1] (correspondingly, j ∈ [J]) into two regimes, where the threshold between them is θ(δ) which is the upper bound on the regimes of xj at which Equation (9) holds. In the first regime, we call the constant query distance scales, one can set ρq ≥ 0
(which implies that the query time for the ANN problem becomes constant), since the smallest space that supports this does not exceed the query time. For the second regime we call the polynomial query distance scales, the space is upper bounded to not exceed our restriction, which enforces constrains on the allowed values ρq (which implies that the query time for the ANN problem becomes polynomial). For further discussion refer to Appendix C, this is summarized as follows.

Definition 14 (Thresholds for Query/Space Exponents). For δ ≥ 0 and x ∈ [0, 1] we let:
Threshold function: θ(δ) = 12

$\frac{1}{2}\left(\sqrt{(\delta+1)(\delta+9)}-(\delta+3)\right)$  . 
$$\rho_{s}(\delta,x)=\begin{cases}\frac{4x}{(1-x)^{2}}&\text{if}x\leq\theta(\delta)\\ \frac{\delta+x}{1-x}&\text{if}x>\theta(\delta)\end{cases}\quad,\quad\rho_{q}(\delta,x)=\begin{cases}0&\text{if}x\leq\theta(\delta)\\ \left(\frac{2\sqrt{x}-\sqrt{(1-x)(\delta+x)}}{1+x}\right)^{2}&\text{if}x>\theta(\delta)\end{cases}$$
$$\xi(\delta,x)=\min_{\rho\geq\rho_{q}(\delta,x)}\max_{y\in[x,1]}(y-x)+(1-x)\left(\rho-\frac{x}{y(1-x)^{2}}\left(\frac{y-x}{\sqrt{x}}-(y-1)\sqrt{\rho}\right)^{2}\right).$$
2!(10)
$$(10)$$
8 Putting everything together. Our data-structure for Level-j Recoveryis obtained by instantiating Lemma 31 with the parameters chosen above. Its properties are in the following lemma, and its proof is in Appendix C.

Lemma 15. For δ ≥ 0, small constants c0, c1 ∈ (0, 1/2), j ∈ [c0J,(1 − c1)J] (where xj = j/J), ρq(δ, x) from Definition 14, the data-structure Dj for the Level-j Recovery problem with preprocess and query procedures from Algorithms 5 and 6 (found in Appendix C) has (expected) query time at most: exp1/µ (ξ(*δ, x*j ) + o(1)) *and (expected) space at most:* exp1/µ (1 + δ + o(1)) *where* Space and Query Exponents Bounds (to be used in Lemma 15):

## 5 Kde Data-Structure Tradeoffs

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 The above theorem follows by plugging the parameters of the relevant data-structures into Theorem 13 (see proof in Appendix D). We also show two consequences of Theorem 16 which follow by numerical evaluations. These highlight the best query time achievable in polynomial space, and the query time achievable with linear space (see proof in Appendix D). Theorem 17. For any precision parameter ϵ and baseline approximation µ as in the setup (Definition 5), there exists a KDE data-structure for the Gaussian Kernel that allows for approximating µ
∗:= K(P, q) up to (1 ± ϵ) multiplicative factor, in the following two regimes of expected query time and space:
- *Query time at most:* exp1/µ (0.*05 +* o(1)) *and space at most:* exp1/µ (4.1 + o(1)) - *Query time at most:* exp1/µ (0.1865 + o(1)) *and space at most:* exp1/µ (1 + o(1))
The query exponent Charikar et al. (2020) get for the data-independent LSH setting is 0.25, 2, and in general they get 0.173, both cases with essentially linear space. Our main result could be interpreted as significantly improving the query time exponent over their main result, with the caveat that their space requirement is only 1/µ (compared to 1/µ4.15 for us), or from the perspective that even within the same space constraints, when δ = 0, our query exponent gets quite close to their main result with a much simpler analysis. Finally, we computed numerically the values of the query exponent ξ(*δ, x*) and the KDE query exponent ξ(δ), and plot these in Figure 1. This plot demonstrates the plateau of the KDE query time ξ(δ) at around 0.05, and that for δ ≈ 3.15 increasing the allowed space does not yield improved query time. This limitation had been discussed in Section 1.2. We discuss these plots further in Appendix D.

![8_image_0.png](8_image_0.png)

$\uparrow$ 4. 
ξ(

![8_image_1.png](8_image_1.png)

δ, x
)

In this section, we use the data-structure Dj from Lemma 15 to construct a KDE data-structure.

Since our data-structure is parameterized by δ such that its space requirement is (1/µ)
1+δ+o(1), we can also plug different value of δ and get a space-query tradeoff for our KDE data-structure as we do in Figure 1. Theorem 16. For any δ ≥ 0, precision parameter ϵ and baseline approximation µ as in the setup (Definition 5), there exists a KDE data-structure for the Gaussian Kernel (see Definition 4) that supports (1 ± ϵ)-multiplicative factor approximation to the Kernel value, in expected query time at most Oeϵ
−2· exp1/µ (ξ(δ) + o(1))*time, and expected space at most at most* Oeϵ
−2· exp1/µ (1 + δ + o(1))*where* ξ(δ) = maxx∈[0,1] ξ(δ, x) for ξ(δ, x) *from Equation* (10).

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Alexandr Andoni and Piotr Indyk. Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. *Commun. ACM*, 51(1):117122, January 2008. ISSN 0001-0782. doi:
10.1145/1327452.1327494. URL https://doi.org/10.1145/1327452.1327494.

Alexandr Andoni, Thijs Laarhoven, Ilya Razenshteyn, and Erik Waingarten. Optimal hashing-based time-space trade-offs for approximate near neighbors. In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA '17, pp. 4766, USA, 2017. Society for Industrial and Applied Mathematics.

Ery Arias-Castro, David Mason, and Bruno Pelletier. On the estimation of the gradient lines of a density and the consistency of the mean-shift algorithm. The Journal of Machine Learning Research, 17(1):1487–1514, 2016.

Arturs Backurs, Moses Charikar, Piotr Indyk, and Paris Siminelakis. Efficient density evaluation for smooth kernels. In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS), pp. 615–626. IEEE, 2018.

Arturs Backurs, Piotr Indyk, and Tal Wagner. Space and time efficient kernel density estimation in high dimensions. *Advances in neural information processing systems*, 32, 2019.

Moses Charikar and Paris Siminelakis. Hashing-based-estimators for kernel density in high dimensions. In *2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS)*, pp. 1032–1043. IEEE, 2017.

Moses Charikar and Paris Siminelakis. Multi-resolution hashing for fast pairwise summations. In 2019 IEEE 60th Annual Symposium on Foundations of Computer Science (FOCS), pp. 769–792.

IEEE, 2019.

Moses Charikar, Michael Kapralov, Navid Nouri, and Paris Siminelakis. Kernel density estimation through density constrained near neighbor search. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS), pp. 172–183. IEEE, 2020.

Moses Charikar, Michael Kapralov, and Erik Waingarten. A quasi-monte carlo data structure for smooth kernel evaluations. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pp. 5118–5144. SIAM, 2024.

Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S. Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. In Proceedings of the Twentieth Annual Symposium on Computational Geometry, SCG '04, pp. 253262, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581138857. doi: 10.1145/997817.997857. URL https:// doi.org/10.1145/997817.997857.

Jianqing Fan. Local polynomial modelling and its applications: monographs on statistics and applied probability 66. Routledge, 2018.

Edward Gan and Peter Bailis. Scalable kernel density classification via threshold-based pruning. In Proceedings of the 2017 ACM International Conference on Management of Data, pp. 945–959, 2017.

Alexander Gray and Andrew Moore. N-body'problems in statistical learning. *Advances in neural* information processing systems, 2001.

Alexander G Gray and Andrew W Moore. Nonparametric density estimation: Toward computational tractability. In *Proceedings of the 2003 SIAM International Conference on Data Mining*, pp. 203– 211. SIAM, 2003.

Leslie Greengard and John Strain. The fast gauss transform. SIAM Journal on Scientific and Statistical Computing, 12(1):79–94, 1991.

Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In *Proceedings of the thirtieth annual ACM symposium on Theory of computing*, pp. 604–613, 1998.

## References

Piotr Indyk, Michael Kapralov, Kshiteej Sheth, and Tal Wagner. Improved algorithms for kernel matrix-vector multiplication under sparsity assumptions. In the Thirteenth International Conference on Learning Representations, ICLR, 2025.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593

## A Spherical (C, R)-Ann Data-Structure From Razenshteyn (2017)

Sarang Joshi, Raj Varma Kommaraji, Jeff M Phillips, and Suresh Venkatasubramanian. Comparing distributions and shapes using the kernel distance. In Proceedings of the twenty-seventh annual symposium on Computational geometry, pp. 47–56, 2011.

Dongryeol Lee, Andrew Moore, and Alexander Gray. Dual-tree fast gauss transforms. Advances in Neural Information Processing Systems, 18, 2005.

Jeff M Phillips and Wai Ming Tai. Near-optimal coresets of kernel density estimates. *Discrete &*
Computational Geometry, 63(4):867–887, 2020.

Parikshit Ram, Dongryeol Lee, William March, and Alexander Gray. Linear-time algorithms for pairwise statistical problems. *Advances in Neural Information Processing Systems*, 22, 2009.

Ilya P. Razenshteyn. *High-dimensional similarity search and sketching: Algorithms and hardness*.

PhD thesis, Massachusetts Institute of Technology, 2017. URL https://dspace.mit.edu/ bitstream/handle/1721.1/113934/1023861862-MIT.pdf?sequence=1.

Aviad Rubinstein. Hardness of approximate nearest neighbor search. In Proceedings of the 50th annual ACM SIGACT symposium on theory of computing, pp. 1260–1268, 2018.

Bernhard Scholkopf and Alexander J Smola. ¨ Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT press, 2002.

J Shawe-Taylor and N Cristianini. Kernel methods for pattern analysis, cambridge university press, 2004, 2004.

Paris Siminelakis, Kexin Rong, Peter Bailis, Moses Charikar, and Philip Levis. Rehashing kernel evaluation in high dimensions. In *International Conference on Machine Learning*, pp. 5789–5798. PMLR, 2019.

Gregory Valiant. Finding correlations in subquadratic time, with applications to learning parities and the closest pair problem. *J. ACM*, 62(2), May 2015. ISSN 0004-5411. doi: 10.1145/2728167.

URL https://doi.org/10.1145/2728167.

Christopher KI Williams and Carl Edward Rasmussen. *Gaussian processes for machine learning*,
volume 2. MIT press Cambridge, MA, 2006.

Yang, Duraiswami, and Gumerov. Improved fast gauss transform and efficient kernel density estimation. In *Proceedings ninth IEEE international conference on computer vision*, pp. 664–671.

IEEE, 2003.

Amir Zandieh, Insu Han, Majid Daliri, and Amin Karbasi. Kdeformer: Accelerating transformers via kernel density estimation. In *International Conference on Machine Learning, ICML*, pp. 40605–40623. PMLR, 2023.

The data-structure for solving the (*c, r*)-ANN problem on the sphere from Razenshteyn (2017, Section 2.4) is central to our work, and we begin by defining the data-structure and stating its guarantees.

The data-structure is parameterized by two parameters ηs, ηq governing the space-query time tradeoff (which are related tho ρq, ρs in Theorem 7, as in Remark 24). Given a dataset P ⊂ S
d−1 of n 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 points on unit sphere in d-dimensions, preprocessing procedure is as follows: Algorithm 3: ANN (ON THE SPHERE) PREPROCESS
Input: dataset P, parameters T, K, ηs, ηq 1 Initialize a tree with K + 1 levels (from 0'th level to K'th) and an upper bound of T of the out degree. There are T
K nodes in the K'th level.

2 Let v0 denote the root, and Lv the path (excluding v0) to any node v. 3 Store a random Gaussian vector zv for each node v except the root.

4 Define:
Pv = {p ∈ P : ∀v
′ ∈ Lv, ⟨zv
′ , p⟩ ≥ η}
Every leaf v at level K stores the subset Pv explicitly.

5 Recursively build the tree as follows - For a given node v, sample T Gaussian vectors g1, ..., gT ∼ N (0, 1)d. Then for every i such that {p ∈ Pv : ⟨gi, p⟩ ≥ ηs} is non-empty, we create a new child v
′ with zv
′ = gi, and recursively process v
′.

After preprocessing the dataset, when we are given a query we use the following procedure to return an approximate near neighbor.

## Algorithm 4: Ann (On The Sphere) Query

Input: Tree from Algorithm 3, parameters T, K, ηs, ηq, query q ∈ S
d−1 1 To answer a query q ∈ S
d−1, we start from the root v0 and traverse the tree.

2 Upon traversing node v, consider every child of v for which ⟨zv, q⟩ ≥ ηq where ηq > 0, and proceed recursively.

3 If leaf node reached, return the first point with distance ≤ cr to q. ▷ See Remark 18 Remark 18. For the ANN problem, it suffices to return the first point encountered at distance *< cr* from the queried point. In our use of this algorithm we assume that all points in the leaves reached by the query algorithm are returned. To state the space and query time of the above data-structure, we will need the following notation, which will be useful for describing the properties of the LSH function.

Definition 19. For any ρ ≥ 0 and z ∈ S
d−1let F(ρ) be defined as, F(ρ) = Prz∼N(0,1)d [⟨z, u⟩ ≥ ρ]
and for any σ ≥ 0 and u ∈ S
d−1such that ∥u − z∥2 = s let G(*s, ρ, σ*) be defined as, G(*s, ρ, σ*) =
Prz∼N(0,1)d [⟨z, u⟩ ≥ ρ and ⟨z, v⟩ ≥ σ].

We now state the success probability, space and query time of the preprocess and query procedures of Algorithms 3 and 4. For the stating these claims we assume that there exists p ∈ P for query q with ∥p − q∥ ≤ r. Claim 20 (Success probability). Razenshteyn (2017, Lemma 2.8.4) For any N ≥ 0, if T ≥
10 log n G(r,ηs,ηq)
then the probability that there is at least one leaf in the data structure created by Algorithm 3 where p, q collide during Algorithm 4 is at least 1 −1 n10 3.

Claim 21 (Space). Razenshteyn (2017, Lemma 2.8.5) The expected space required for the datastructure created by Algorithm 3 is at most: n 1+o(1)· K · (T · F(ηs))K.

Claim 22 (Query time). Razenshteyn (2017, Lemma 2.8.6) If T F(ηq) ≥ 3 then the expected runtime of Algorithm 4 is at most: n o(1)·T · (T · F(ηq))K + n · (T · G(cr, ηs, ηq))K.

For the above claim, the proof actually shows the following: the expected query time spent going down the tree in Algorithm 4, without scanning the leaves is n o(1)·T · (T · F(ηq))K. Moreover, the expected number of points scanned at the leaves reached is n 1+o(1)· (T · G(cr, ηs, ηq))K. The number of points scanned is always at most one more than the number of far points, i.e., lying a distance greater than cr from q, that reached the same leaf. Additionally, we present the following corollary, implicit in Razenshteyn (2017, Lemma 2.8.6) Claim 23. For any query q and p ∈ P such that ∥p − q∥ ≥ t and each leaf ℓ in the tree constructed in Algorithm 3, the probability that both p and a query q end up in ℓ is at most: (G(t, ηs, ηq))K.

3This is a slight variation of the original claim from Razenshteyn (2017) which trivially follows from its original proof.