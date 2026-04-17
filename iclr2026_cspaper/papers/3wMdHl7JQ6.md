000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Anonymous authors Paper under double-blind review

## Abstract

We propose a streamlined spectral algorithm for community detection in the twocommunity stochastic block model (SBM) under constant edge density assumptions. By reducing algorithmic complexity through the elimination of non-essential preprocessing steps, our method directly leverages the spectral properties of the adjacency matrix. We demonstrate that our algorithm exploits specific characteristics of the second eigenvalue to achieve improved error bounds that approach information-theoretic limits, representing a significant improvement over existing methods. Theoretical analysis establishes that our error rates are tighter than previously reported bounds in the literature. Comprehensive experimental validation confirms our theoretical findings and demonstrates the practical effectiveness of the simplified approach. Our results suggest that algorithmic simplification, rather than increasing complexity, can lead to both computational efficiency and enhanced performance in spectral community detection.

## 1 Introduction

Community detection represents a fundamental challenge in statistics, theoretical computer science, and image processing. The stochastic block model (SBM) serves as a prominent theoretical framework for analyzing this problem. In its simplest form, the model consists of two equal-sized blocks V1 and V2, each containing n vertices. A random graph is generated according to the following distribution:
edges between vertices within the same block occur with probability an
, while edges between vertices in different blocks occur with probability bn
, where *a > b >* 0. Given such a graph, various algorithms exist for block recovery Chin et al. (2015), Bui et al. (1984), Dyer & Frieze (1989), McSherry (2001), Coja-Oghlan (2009). In the sparse graph case, with high probability, the graph contains a linear fraction of isolated vertices Bollobas (2001). Since these isolated vertices lack connectivity information, perfect recovery of the ´ community structure is impossible. However, we can still accurately recover a substantial portion of each block. Formally, we would like to find a partition of V
′
1, V ′2 of V = V1 ∪ V2 such that Vi and V
′
i are very close to each other. To quantify the recovery accuracy, we introduce the following definition: Definition 1.1. *A collection of subsets* V
′
1, V ′
2 of V1∪V2 is γ-correct if |V i∩V
′
i| ≥ (1−γ)n, i = 1, 2.

We would like to devise an algorithm that can guarantee γ-correctness for small γ with high probability in polynomial time. In Coja-Oghlan (2009), Coja-Oglan proved Theorem 1.2. For any constant γ > 0, there exist constants C1, C2 > 0 such that if a, b > C1 and
(a−b)
2 a+b > C2 log(a + b), one can find a γ*-correct partition using a polynomial time algorithm.*
In Chin et al. (2015), Chin et al. introduced a Spectral Algorithm that achieves exponential bounds on the incorrect recovery rate in the case of a sparse graph.

Theorem 1.3. There are constants C1, C2 > 0 *such that the following holds. For any constants* a > b > C1 and γ > 0 *satisfying*
(a − b)
2 a + b≥ C2 log 2 γ(1)
one can find a γ*-correct partition with probability* 1 − o(1) *using a simple spectral algorithm.*
1

# Simplify To Amplify: Achieving Information- Theoretic Bounds With Fewer Steps In Spectral Community Detection

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Original Spectral Algorithm

In Chin et al. (2015), Chin et al. gave the Spectral Algorithm that guarantees the result in Theorem 1.3. But first let us define some variables. Let A denote the adjacency matrix of a random graph generated from the distribution described in Section 1. And let AE = E[A] be the expected adjacency matrix, with entries a/n and b/n. Then AE is a rank two matrix with two non-zero eigenvalues λ1 = a + b and λ2 = a − b. Then unit eigenvector u1 corresponding to the eigenvalue a + b has coordinates:

$$u_{1}(i)=\frac{1}{\sqrt{2n}}\forall i=1,\ldots,2n$$

while the unit eigenvector u2 corresponding to the eigenvalue a − b has coordinates

$$({\mathfrak{I}})$$
$$u_{\mathbf{2}}(i)={\begin{cases}{\frac{1}{\sqrt{2n}}}&{{\mathrm{~if~}}i\in V_{1}}\\ {-{\frac{1}{\sqrt{2n}}}}&{{\mathrm{~if~}}i\in V_{2}}\end{cases}}$$
$$(4)^{\frac{1}{2}}$$

Spectral Partition.

1. Input the adjacency matrix *A, d* := a + b. 2. Zero out all the rows and columns of A corresponding to vertices whose degree is bigger than 20d, to obtain the matrix A′.

3. Find the eigenspace W corresponding to the top two eigenvalues of A′.

4. Compute v1, the projection of all-ones vector on to W 5. Let v2 be the unit vector in W perpendicular to v1. 6. Sort the vertices according to their values in v2, and let V
′1 ⊂ V be the top n vertices, and V
′
2 ⊂ V be the remaining n vertices 7. Output (V
′1
, V ′2
).

Figure 1: Spectral Partition Theorem 1.3 improves the relation between the accuracy γ and the ratio (a−b)
2 a+b
. Moreover, this bound

is asymptotically sharp because according to Zhang & Zhou (2015), there exists a constant c > 0 such that when
$${\frac{(a-b)^{2}}{a+b}}\leq c\log{\frac{1}{\gamma}}$$
$$(2)$$

## Γ(2)
One **Cannot** Recover A Γ-Correct Partition (In Expectation), Regardless Of The Algorithm. The Standard Spectral Algorithm Comprises Two Stages: **Spectral Partition** And **Correction** (Detailed In Section 2). Previous Work Established That Spectral Partition Alone Achieves Only Inverse-Square Correctness Rates, Requiring The Correction Step To Reach The Desired Inverse-Log Relationship. However, Our Experiments Reveal That Spectral Partition Actually Produces Inverse-Log Performance Without Correction, Suggesting This Additional Step Is Unnecessary. Our Theoretical Analysis Identifies A Non-Tight Lemma In The Original Proof That Underestimates The Algorithm'S Performance. We Provide Improved Bounds And Experimentally Demonstrate That These Bounds Are Sharp, Eliminating The Need For The Correction Step To Achieve The Inverse-Log Rates Claimed In Chin Et Al. (2015). Additionally, We Streamline The Spectral Partition Itself By Removing Redundant Operations, Ensuring That The Resulting Vectors Maintain Statistical Independence, A Property That Will
Prove Valuable For Future Algorithmic Improvements (Discussed In Section 5).
The Rest Of This Paper Is Organized As Follows: Section 2 Presents The Original Spectral Algorithm And Our Simplified Version. Section 3 Shows That Our Simplification Maintains And Improves Theoretical Bounds. Section 4 Validates Our Predictions Experimentally. Section 5 Summarizes Our Findings And Discusses Future Work. 2.1 Our Modified Algorithm

The bound in Theorem 2.1 is weaker than that claimed in Theorem 1.3. To achieve the inverse-log relationship, the original work requires a second **Correction** step (Figure 2), yielding the complete algorithm shown in Figure 3. The correction mechanism works as follows: provided **Spectral** Partition achieves sufficiently low error rate γ, the **Correction** step reduces this to exponentially small values.

Correction.

1. Input: a partition V

![2_image_0.png](2_image_0.png)

′
1, V ′2and a Blue graph on V

![2_image_1.png](2_image_1.png)

1 ∪ V
2.

2. For any u ∈ V
′1, label u bad if the number of neighbors of u in V
′2is at least a+b

![2_image_2.png](2_image_2.png)

good otherwise.

3. Do the same for any v ∈ V
′
2.

4. Correct V
′
ibe deleting its bad vertices and adding the bad vertices from V
′
3−i.

$$({\boldsymbol{5}})$$

## Figure 2: Correction

Specifically, Lemma 2.3 in Chin et al. (2015) establishes that if the input to **Correction** is c-correct for some c > 0, then the output achieves γ-correctness with γ = 2 exp −f(c)
(a−b)
2 a+b where f(c) > 0 depends only on c. The complete two-stage algorithm of Chin et al. is therefore the **Partition** procedure in Figure 3.

Partition 1. Input the adjacency matrix *A, d* := a + b.

![2_image_3.png](2_image_3.png)

![2_image_4.png](2_image_4.png) 2. Randomly color the edges with Red and Blue with equal probability. 3. Run **Spectral Partition** on Red graph, outputting V
′
1
, V ′

![2_image_5.png](2_image_5.png)

4. Run **Correction** on the Blue graph. 5. Output the corrected sets V
′
1, V 
′
2.

Figure 3: Partition Our key modification to **Spectral Partition** eliminates step 2, which zeros out rows and columns corresponding to vertices with degree greater than 20d. Instead, we work directly with the original adjacency matrix A throughout the algorithm. While this preprocessing step was essential for two lemmas in the original analysis, it destroys the statistical independence of matrix entries in A′.

By working with A directly, we preserve the independent distribution of matrix entries and can subsequently maintain independence in the entries of eigenvector w2. This independence property proves crucial for our analysis in Section 3 and may help future algorithmic enhancements we explore in Section 5.

The second eigenvector u2 of the expected adjacency matrix AE encodes the true community structure. Let w1 and w2 denote the first and second eigenvectors of the observed adjacency matrix A, respectively. Our goal is to use w2 as a proxy for the unknown u2. The **Spectral Algorithm** in Figure 1 produces vector v2 that closely approximates u2, achieving the following result: Theorem 2.1. There are constants C1, C2 > 0 *such that the following holds. For any constants* a > b > C1 and γ > 0 *satisfying*
$$\frac{(a-b)^{2}}{a+b}\geq C_{2}\frac{1}{\gamma^{2}}$$
2(5)
one can find a γ*-correct partition with probability* 1 − o(1) *using* **Spectral Partition.**
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

The first lemma requiring step 2 is restated in Theorem 2.2. Define M = A − AE as the difference between the observed and expected adjacency matrices. Let M′ denote the matrix obtained by
applying the same row and column deletions to M as performed on A in step 2 of **Spectral Partition**. Chin et al. (2015) establish the following result:
Theorem 2.2. There exist constants C1, C2 such that if a > b > C1, and matrix M′is obtained as
described above, then we have
$$||M^{\prime}||\leq C_{2}{\sqrt{a+b}}$$
√a + b (6)
with probability 1 − o(1).

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 3 Improved Error Bounds For Spectral Partition

$$\sin\angle(\mathbf{u_{2}},\mathbf{v_{2}})\leq C_{2}{\sqrt{\frac{{\sqrt{a+b}}}{a-b}}}$$
$$(7)$$

a − b(7)
with probability 1 − o(1). which proves Theorem 2.1. Our experiments reveal that Theorem 3.1 is tight, while Theorem 3.2 is not. In general, Theorem 3.2 is indeed sharp. There exist vectors u2, v2 achieving equality up to a constant factor. However, the Spectral Algorithm produces vectors v2 with specific structural properties that render this bound loose. We prove that under these properties, significantly tighter bounds are achievable. 3.2 SHARPNESS OF THEOREM 3.2 To establish the sharpness of Theorem 3.2, we formulate the following optimization problem. Let x1*, . . . , x*2n denote the entries of v2 with Px 2 i = 1 and x1 *≥ · · · ≥* x2n. The partition step assigns

$$(6)$$

Throughout this paper, ||M′|| denotes the spectral norm sup{||Mx||2 : ||x||2 ≤ 1}, and all matrix norms follow this convention. While the original proof of Theorem 2.2 depends on the deletion step, we show that the bound holds without deletion, with only modest increases in the constants C1, C2. Our proof, which leverages techniques from Furedi & Komlos (1981) and Krivelevich & Vu (2000), ¨ is provided in the appendix. The second lemma that depends on the deletion step appears in the **Correction** step analysis. Since our simplified algorithm eliminates this step entirely, we don't have to analyze the implications of our modification to this step.

## 3.1 Original Error Bounds

Let W be the two-dimensional eigenspace corresponding to the top two eigenvalues of A, and let WE be the corresponding eigenspace of AE. Chin et al. Chin et al. (2015) establish that the angle
∠(*W, W*E) between these subspaces is sufficiently small, where we use the standard convention sin ∠(W1, W2) := ||PW1 − PW2|| with PW denoting the orthogonal projection onto subspace W.

As a consequence of this subspace proximity, the angle between u2 (the second eigenvector of AE) and v2 (the vector obtained in step 5 of **Spectral Partition**) is also small. The key insight is that when these vectors are well-aligned, **Spectral Partition** produces an accurate community assignment.

Specifically, the analysis in Chin et al. (2015) bounds sin ∠(u2, v2) and establishes the following result:
Theorem 3.1. There exist constants C1, C2 such that if a > b > C1, and vectors u2, v2 are as described above, then we have

$$\gamma\leq C_{2}{\frac{\sqrt{a+b}}{a-b}}$$
a − b(8)
with probability 1 − o(1).
$$(8)$$

Finally, Chin et al. (2015) shows that γ ≤
4 3 sin2 ∠(u2, v2), which gives us the following result:
Theorem 3.2. There exist constants C1, C2 such that if a > b > C1*, then we have*

## 3.3 Statistical Properties Of The Second Eigenvector

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 3.4 Applying Chernoff Bounds To Relate Γ And Sin Θ

indices {1*, . . . , n*} to community V1 and {n + 1*, . . . ,* 2n} to community V2. For fixed error rate γ, let k = γn (assuming k is integer), representing the number of misclassified vertices.

Our goal is to minimize the angle θ = ∠(u2, v2) subject to fixed γ, equivalent to maximizing cos θ.

The true community indicator satisfies wi = 1/
√2n for i ∈ V1 and wi = −1/
√2n for i ∈ V2.

Without misclassification:

$$\cos\theta=\sum_{i=1}^{2n}x_{i}w_{i}={\frac{1}{\sqrt{2n}}}\left(\sum_{i=1}^{n}x_{i}-\sum_{i=n+1}^{2n}x_{i}\right)$$

To maximize cos θ under exactly k misclassifications, the optimal strategy places errors among entries with smallest magnitudes. Specifically, vertices {n − k + 1*, . . . , n*} from V1 are misassigned to V2, while vertices {n + 1*, . . . , n* + k} from V2 are misassigned to V1, yielding:

$$\cos\theta\leq\frac{1}{\sqrt{2n}}\left(\sum_{i=1}^{n-k}x_{i}-\sum_{i=n-k+1}^{n}x_{i}+\sum_{i=n+1}^{n+k}x_{i}-\sum_{i=n+k+1}^{2n}x_{i}\right)\tag{9}$$
$$(10)$$

This bound is achieved by the assignment x1 = *· · ·* = xn−k = 1/p2(n − k), xn−k+1 = *· · ·* =
xn+k = 0, and xn+k+1 = *· · ·* = x2n = −1/p2(n − k), which satisfies the normalization constraint and yields cos θ =
√1 − γ. Therefore γ = sin2θ, confirming that Theorem 3.2 is sharp up to constants.

Abbe et al. (2019) demonstrate that the second eigenvector can be approximated as w2 ≈
Au2 a−b with error bound ||w2 −
Au2 a−b ||∞ = o(1/
√n)
The denominator a − b is irrelevant as w2 will be scaled to be a unit vector. Thus we now focus on characterizing the distribution of Au2. For vertex i ∈ V1, the i-th entry of Au2 equals the difference between the number of edges between i and vertices in V1, and the number of edges between i and vertices in V2. Since each edge appears independently with probability a/n (within-community) or b/n (between-community), this entry follows the distribution of a difference of two binomial random variables. Specifically, let

$$Y\sim{\mathrm{Binomial}}(n,a/n)-{\mathrm{Binomial}}(n,b/n)$$

Y ∼ Binomial(*n, a/n*) − Binomial(*n, b/n*) (10)
Then each entry of Au2 is distributed as Y or −Y with equal probability, depending on whether i ∈ V1 or i ∈ V2. Building on the optimization framework above, while we know the approximate distribution of the xi entries, direct analysis remains computationally intractable. Instead, we leverage constraints derived from Chernoff concentration inequalities applied to this distribution. The Chernoff bound states that for a random variable X with moment generating function M(t):

$$P(X\geq a)\leq M(t)e^{-t a}\quad\forall a,\forall t>0$$

This bound becomes increasingly sharp in the tail regions for large values of a. For approximately bell-shaped distributions, Chernoff bounds at multiple points constrain the distribution's tail behavior, effectively providing lower bounds on how "concentrated" the distribution must be around its center.

Applied to our ordered sequence x1 ≥ x2 *≥ · · · ≥* xn, these concentration properties impose lower bounds on the decay rates between consecutive entries:

$${\frac{x_{1}}{x_{2}}},{\frac{x_{2}}{x_{3}}},\ldots,{\frac{x_{n-1}}{x_{n}}}$$

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 3.5 Monte-Carlo Simulation And Normal Approximation To Relate Γ And Sin Θ

Given the distribution in Equation 10, we can directly generate samples of the xi entries using Monte Carlo methods, removing the need for numerical optimization. With the xi values generated from their distribution, Equation 9 provides the maximum cos θ for any given error level k.

Define pa = a/n, qa = 1 − pa, pb = b/n, qb = 1 − pb, and the optimal Chernoff parameter t
∗ =
1 2 ln paqb qapb
. Let the concentration constant be:

$$C=\frac{1}{2}(\sqrt{p_{a}p_{b}}+\sqrt{q_{a}q_{b}})^{2n}+\frac{1}{2}\left(\sqrt{\frac{q_{a}^{3}p_{b}^{3}}{p_{a}q_{b}}}+\sqrt{\frac{p_{a}^{3}q_{b}^{3}}{q_{a}p_{b}}}+q_{a}q_{b}+p_{a}p_{b}\right)$$
.$n\,-\,1$  . 
$\left(11\right)$. 

n
The Chernoff concentration inequalities translate into the following optimization constraints:

* [16] A. A. K.  
2
2
$x_{i+1}\leq\frac{\ln C+\ln(2n+1)-\ln(i+1)}{\ln C+\ln(2n+1)-\ln i}x_{i}\quad\forall i=1,\ldots$
$$x_{i}\geq\frac{\ln C+\ln(2n+1)-\ln(2n+1-i)}{\ln C+\ln(2n+1)-\ln(2n-i)}x_{i+1}\quad\forall i=n+1,\ldots,2n-1.$$
The complete derivation appears in the appendix. Since C is known before any optimization, these

![5_image_0.png](5_image_0.png) constraints together with Equation 9 as the objective function define a convex optimization problem. We solve this optimization problem umerically to find the maximum value of cos θ subject to the above constraints. Our theoretical analysis predicts this maximum should satisfy (proof in the appendix):

$$\cos\theta\leq{\frac{\sqrt{2n}}{t^{*}}}(1-\gamma)\left(\ln C+1+\ln{\frac{2+{\frac{1}{n}}}{1-\gamma}}\right)$$
(11)

![5_image_1.png](5_image_1.png)

Figure 4a presents our experimental validation results for n = 500, a = 0.06*n, b* = 0.04n. The red points represent the relationship from Theorem 3.2, while the blue points show the actual optimization results under our Chernoff-derived constraints. The blue line displays our theoretical prediction from Equation 11, fitted to the optimization data using ordinary least squares (OLS) regression to account for the unit normalization of the xi vector. The results demonstrate that our Chernoff-based analysis yields significantly tighter bounds than the original theorem. For any given value of sin θ, our approach provides a substantially lower upper bound on the achievable error rate γ. Furthermore, the close agreement between the blue line and blue points confirms the accuracy of our theoretical prediction in Equation 11.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 While we could compute Equation 9 directly using the exact probability density function, this approach is algebraically intractable. Instead, we use a normal approximation to simplify the analysis. The binomial distributions in our model satisfy the standard approximation conditions: both np ≥ 20 and n(1 − p) ≥ 20 hold for our parameter ranges, so that the approximation is reasonable. Under this normal approximation, the difference of binomials Y also approaches normality, and consequently each entry Xi becomes approximately normal. This normality assumption enables us to derive a closed-form theoretical prediction for the performance bound. Using the normal approximation and the structure of our optimization problem, we obtain the following theoretical prediction (with derivation provided in the appendix):

$$\cos\theta\leq{\frac{2}{\sqrt{2n}}}(2n+1)\left(2\phi\left(-\Phi^{-1}\left({\frac{1-\gamma}{2+1/n}}\right)\right)-\phi\left(-\Phi^{-1}\left({\frac{1}{2+1/n}}\right)\right)\right)$$
$$(12)$$
 (12)
where ϕ and Φ denote the standard normal probability density function and cumulative distribution function, respectively. In the derivation above, we assumed that the entries xi follow a standard normal distribution with mean 0 and unit variance. While the zero-mean assumption is valid, the unit variance assumption is not. The actual entries will have a different variance determined by the underlying binomial distributions and the problem parameters. However, since the final vector must satisfy the normalization constraint Px 2 i = 1, the entries will be appropriately scaled regardless of their original variance. The theoretical prediction in Equation 12 captures the correct functional relationship between γ and cos θ, but with a scaling factor that depends on the actual variance of the entries. Figure 4b presents our experimental validation using the same parameters as before: n = 500, a = 0.06n, b = 0.04n. We conducted Monte Carlo simulations with 50 repetitions to minimize random variation in our results. The green points represent the (sin *θ, γ*) pairs computed from each simulation run, forming a "band" due to the natural clustering of results across repetitions. The green dashed line shows our theoretical prediction from Equation 12, fitted to the simulation data using OLS regression to account for the normalization constraint. For comparison, we include the blue points from our earlier Chernoff-based analysis (Section 3.4, Figure 4a). The results validate several important aspects of our theoretical framework: First, the close agreement between the green dashed line and the simulation points confirms that our normal approximation in Equation 12 accurately captures the underlying relationship between error rate and spectral alignment. Next, the green band lies well below the blue points, demonstrating that while our Chernoff-derived bounds are mathematically sound, they remain conservative estimates. The gap between these approaches becomes particularly pronounced for small error rates, precisely the region most relevant for practical applications. This suggests that the Chernoff bounds, though tight in a worst-case sense, do not fully capture the distributional properties that emerge in typical use cases. Perhaps most significantly, both our simulation and Chernoff analysis reveal that perfect community recovery (γ = 0) is achievable even when the eigenvectors u2 and v2 are not perfectly aligned
(sin θ > 0). This indicates that the spectral method's success depends not merely on eigenvector alignment, but more fundamentally on whether the entry distribution of v2 preserves sufficient structure to enable correct partitioning. In other words, the distributional shape of the eigenvector entries often contains enough information to guarantee perfect classification, even in the presence of some spectral distortion.

## 4 Comparing Theoretical Predictions With Spectral Algorithm Results

While the results in Section 3 significantly improve upon the original bounds in Theorem 3.2, all our theoretical analyses rely on the distributional approximation given in Equation 10. As noted previously, this approximation contains errors that, while decreasing as O(1/
√n), may still affect the accuracy of our predictions for finite sample sizes. To validate our theoretical framework against the actual spectral algorithm performance, we conduct direct experiments on randomly generated graphs. We generate stochastic block model instances with edge probabilities a = 0.06n and b = 0.04n across a range of graph sizes n ∈ {500, 525, 550*, . . . ,* 1000}. For each instance, we apply our modified **Spectral Partition** algorithm (omitting the degree-based deletion step) and evaluate both the error rate γ (comparing the algorithm's partition against the true community structure) and θ (the angle between the true second eigenvector u2 and the computed approximation to second eigenvector v2).

Furthermore, to provide comprehensive validation across different problem scales, we repeated all the analyses from Section 3 for the complete range of graph sizes n ∈ {500*, . . . ,* 1000}, rather than limiting our evaluation to n = 500. These results, including both the Chernoff-based optimization bounds and the Monte Carlo simulation predictions, are consolidated alongside the direct spectral algorithm experiments in Figure 5.

The figure uses opacity to represent graph size, with n = 500 shown as nearly transparent points and n = 1000 as fully opaque points, creating a visual gradient across problem scales. Different colors distinguish the various analytical approaches:
Red Points (Theoretical Baseline): These represent the quadratic bound from Theorem 3.2. Since this bound follows the relationship γ = sin2θ, which is independent of n, the red points of different opacities overlap completely, forming a single curve.

Blue Points (Chernoff Analysis): These show our Chernoff-derived bounds from Section 3.4. As n increases, the achievable frontier moves upward, indicating that the bounds become less tight for larger graphs. This behavior reflects the conservative nature of concentration inequalities for finite sample sizes. Green Points (Monte Carlo Simulation): These represent our normal approximation approach validated through simulation, with 10 repetitions per value of n. Similar to the Chernoff bounds, the frontier shifts upward with increasing n, particularly in the low-γ regime. Orange Points and Purple Fit (Direct Algorithm Results): The orange points show the actual performance of our modified **Spectral Partition** algorithm on randomly generated graphs. To these experimental results, we fit the empirical relationship:

$$\sin\theta={\frac{C}{\sqrt[4]{\log2/\gamma}}}$$
$$(13)^{\frac{1}{2}}$$
p4log 2/γ(13)
using OLS regression, with the resulting fitted curve displayed as the purple line. Theoretical Significance: The functional form in Equation 13, combined with the claims of Theorems 2.2 and 3.1, directly yields the final result stated in Theorem 1.3, thus bridging our empirical observations with the theoretical framework.

## 4.1 Scaling Behavior And Convergence Analysis

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Several important trends emerge as n increases while maintaining constant ratios a/n and b/n. The community detection problem becomes inherently easier for larger graphs, as predicted by both Theorem 1.3 and Theorem 3.2, which allow for smaller error rates γ as their left-hand sides increase. This theoretical prediction is confirmed in our results, where larger n values (higher opacity points)
consistently achieve lower γ values.

More significantly, the gap between the orange points (direct algorithm results) and green points
(simulation predictions) of matching opacity decreases with increasing n. This convergence validates the error bound which asserts that approximation errors decrease as O(1/
√n). The observed convergence demonstrates that for large n in the low-γ regime, the relationship in Equation 13 and our theoretical prediction in Equation 12 align closely. This convergence provides strong empirical support for our central claim: **Spectral Partition** alone achieves near information-theoretic performance without requiring the additional **Correction** step, particularly as problem size increases and error rates decrease, precisely the regime most relevant for practical applications.

![8_image_0.png](8_image_0.png)

## 5 Conclusion And Future Work

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 We demonstrate that the spectral algorithm achieves near information-theoretic performance, through elimination of degree-based preprocessing and the correction step. Our theoretical analysis through Chernoff bounds, normal approximations, and Monte Carlo validation shows that spectral partition alone can achieve the inverse-logarithmic error rates previously thought to require additional correction steps. Experimental validation across varying graph sizes confirms that our theoretical predictions become increasingly accurate as the error goes down with O(1/
√n), with the empirical relationship sin θ =
C/p4log 2/γ bridging our results to established theoretical frameworks. The convergence between multiple analytical approaches in the large-n, low-γ regime validates our central finding: spectral partition alone suffices for near-optimal community recovery. These results challenge the assumption that algorithmic complexity improves performance, suggesting instead that careful theoretical analysis can reveal hidden strengths in existing methods. This "less is more" principle may have broader implications for spectral algorithm design.

Several directions emerge from this research: extending our analysis to unbalanced and multicommunity cases, analyzing multiple samples derived from the same distributions, developing enhanced inference procedures, investigating computational scaling for massive graphs, analyzing robustness under model misspecification, establishing precise connections to information-theoretic limits, and exploring whether similar simplifications yield improvements in related spectral problems such as graph clustering and matrix completion. The statistical independence between matrix and vector entries preserved by our approach should facilitate these future investigations, as this independence structure can be leveraged for more sophisticated statistical inference and analysis techniques that would be complicated or impossible under the dependencies introduced by traditional preprocessing steps.

## 6 Reproducibility Statement

To ensure reproducibility, we provide complete implementation details with specified parameters: graph sizes n ∈ {500*, . . . ,* 1000}, edge probabilities a = 0.06n and b = 0.04n, and our modified Spectral Partition algorithm that eliminates the degree-based deletion step. Monte Carlo simulations use 50 repetitions for distributional analysis and 10 repetitions for scaling experiments. All random seed numbers are initialized to ensure total reproducibility. Our submitted code includes scripts to regenerate all figures and numerical results, with complete theoretical derivations provided in the appendix.

## References

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Thang Nguyen Bui, Soma Chaudhuri, Frank Thomson Leighton, and Michael Sipser. Graph bisection algorithms with good average case behavior. *Combinatorica*, 7:171–191, 1984. URL https:
//api.semanticscholar.org/CorpusID:32346819.

Peter Chin, Anup Rao, and Van Vu. Stochastic block model and community detection in the sparse graphs: A spectral algorithm with optimal rate of recovery, 2015. URL https://arxiv.org/ abs/1501.05021.

Amin Coja-Oghlan. Graph partitioning via adaptive spectral techniques. Combinatorics, Probability and Computing, 19:227 - 284, 2009. URL https://api.semanticscholar.org/ CorpusID:355743.

Martin E. Dyer and Alan M. Frieze. The solution of some random np-hard problems in polynomial expected time. *J. Algorithms*, 10:451–489, 1989. URL https://api.semanticscholar. org/CorpusID:13419364.

Zoltan F ´ uredi and John Komlos. The eigenvalues of random symmetric matrices. ¨ *Combinatorica*, 1:
233–241, 1981. URL https://api.semanticscholar.org/CorpusID:7847476.

Michael Krivelevich and Van H. Vu. On the concentration of eigenvalues of random symmetric matrices, 2000. URL https://arxiv.org/abs/math-ph/0009032.

Frank McSherry. Spectral partitioning of random graphs. Proceedings 2001 IEEE International Conference on Cluster Computing, pp. 529–537, 2001. URL https://api.semanticscholar.

org/CorpusID:10389217.

Anderson Y. Zhang and Harrison H. Zhou. Minimax rates of community detection in stochastic block models, 2015. URL https://arxiv.org/abs/1507.05313.

## A Appendix

A.1 PROOF OF THEOREM 2.2 Proof. The matrix A has entries Aij that are sampled from a Bernoulli distribution with success probability pij where pij = a/n if *i, j* belong to the same community, and pij = b/n otherwise.

Therefore, the entries of matrix M have mean zero and variance σ 2 ij = pij (1 − pij ) ≤ σ 2 where σ 2 is the maximum variance of a single element. Because 0 *< b < a < n/*2 we have:

$$\sigma_{ij}^{2}\leq\sigma^{2}=\max\left(\frac{a}{n}(1-\frac{a}{n}),\frac{b}{n}(1-\frac{b}{n})\right)=\frac{a}{n}\left(1-\frac{a}{n}\right)\leq\frac{a+b}{n}\tag{14}$$

Let λ1(M) be the largest eigenvalue of M. Because M is real-valued and symmetric, λ1(M) = ||M||.

Now we use the result from Furedi & Komlos (1981) to determine ¨ E[λ1(M)]. Since all entries have mean zero and variance at most σ 2, we have:

$$\mathbb{E}[\lambda_{1}(M)]=2\sigma{\sqrt{n}}+O(n^{1/3}\log n)$$
1/3log n) (15)
For large enough n, the first term dominates. So E[λ1(M)] = O(σ
√n). Note: Furedi & Komlos ¨
(1981) uses the premise that all entries have mean zero and common variance, but Krivelevich & Vu
(2000) showed that the assumption of common variance can be relaxed to *V ar*[Mij ] ≤ σ 2.

$$(15)$$

Emmanuel Abbe, Jianqing Fan, Kaizheng Wang, and Yiqiao Zhong. Entrywise eigenvector analysis of random matrices with low expected rank, 2019. URL https://arxiv.org/abs/1709. 09565.

Bela Bollob ´ as. ´ *Random Graphs*. Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2 edition, 2001.

$$P\left[|\lambda_{1}(M)-\mathbb{E}[\lambda_{1}(M)]|\geq t\right]\leq e^{-c t^{2}}$$

Combining equations 15 and 16, there is a constant C2 such that for large enough b (and consequently a, n), we have with probability 1 − o(1): which completes the proof for Theorem 2.2.

$$(16)^{\frac{1}{2}}$$
$$||M||\leq C_{2}\sigma{\sqrt{n}}\leq C_{2}{\frac{{\sqrt{a+b}}}{{\sqrt{n}}}}{\sqrt{n}}$$
$$(17)$$
$\square$
A.2 PROOF OF FORMULATION AND PREDICTION FROM SECTION 3.4 A.2.1 DERIVING THE MOMENT GENERATING FUNCTION We start by computing the moment generating function (MGF) for our random variables. Recall that Y represents the difference between two binomial distributions. The MGF of Y is: The Chernoff bound gives us:

$$P(X_{i}\geq a)\leq M_{X_{i}}(t)e^{-a t}\quad\forall t>0$$

This inequality holds for any positive t, but we want to choose the value that gives us the tightest bound. For positive values of a, the distribution is dominated by the Y component rather than the −Y component. The optimal choice turns out to be:
Now we connect this probabilistic bound to our optimization problem. If xiis the i-th largest element in our sorted vector, and assuming the entries follow the theoretical distribution reasonably well, then Next, also according to Krivelevich & Vu (2000), there are positive constants c and K such that for any t > K,

$$t^{*}={\frac{1}{2}}\ln\left({\frac{p_{a}q_{b}}{q_{a}p_{b}}}\right)$$

Note that t
∗ > 0 because we assume pa > pb (within-community edges are more likely than between-community edges). Substituting this optimal value, we get:

$$P(X_{i}\geq a)\leq C e^{-a t}$$
$$C=\frac{1}{2}(\sqrt{p_{a}p_{b}}+\sqrt{q_{a}q_{b}})^{2n}+\frac{1}{2}\left(\sqrt{\frac{q_{a}^{3}p_{b}^{3}}{p_{a}q_{b}}}+\sqrt{\frac{p_{a}^{3}q_{b}^{3}}{q_{a}p_{b}}}+q_{a}q_{b}+p_{a}p_{b}\right)$$
n

## A.2.3 Converting Bounds To Optimization Constraints

$$M_{Y}(t)=(q_{a}+p_{a}e^{t})^{n/2}(q_{b}+p_{b}e^{-t})^{n/2}$$

Since −Y has MGF M−Y (t) = MY (−t), and each entry Xi of our vector is equally likely to be Y or −Y , the MGF of Xi becomes:

$$M_{X_{i}}(t)={\frac{M_{Y}(t)+M_{Y}(-t)}{2}}$$
$$={\frac{(q_{a}+p_{a}e^{t})^{n/2}(q_{b}+p_{b}e^{-t})^{n/2}+(q_{a}+p_{a}e^{-t})^{n/2}(q_{b}+p_{b}e^{t})^{n/2}}{2}}$$
2
540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 where the constant C depends only on the model parameters n, a, and b:
A.2.2 APPLYING CHERNOFF BOUNDS

## A.2.4 Formulating The Complete Optimization Problem

Solving for xi:

$$x_{i}\leq{\frac{\ln C+\ln(2n+1)-\ln i}{t^{*}}}$$

For the negative tail (when *i > n*), we use the symmetry of the bounds with t replaced by −t, giving us:

$$x_{i}\geq-{\frac{\ln C+\ln(2n+1)-\ln(2n+1-i)}{t^{*}}}$$

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647

## A.2.6 Deriving Cumulative Sum Approximations

Starting from our Chernoff-derived bound:

$$x_{i}\leq{\frac{\ln C+\ln(2n+1)-\ln i}{t^{*}}}$$

the probability that a random entry exceeds xi should be approximately i 2n+1 (since i entries are larger than xi out of 2n + 1 total positions). Therefore:

$${\frac{i}{2n+1}}\leq C\cdot e^{-t^{*}x_{i}}$$
$$x_{1}^{2}+\cdots+x_{2n}^{2}\leq1$$
$$x_{i+1}\leq\frac{\ln C+\ln(2n+1)-\ln(i+1)}{\ln C+\ln(2n+1)-\ln i}x_{i}\quad\forall i=1,\ldots,n-1\,,$$
$$x_{i}\geq\frac{\ln C+\ln(2n+1)-\ln(2n+1-i)}{\ln C+\ln(2n+1)-\ln(2n-i)}x_{i+1}\quad\forall i=n+1,\ldots,2n-1.$$

## A.2.5 Why This Formulation Works

Let us elaborate why this setup correctly captures our intentions:
First, regarding the normalization constraint Px 2 i ≤ 1: We use an inequality rather than equality to make this a convex optimization problem, which can be solved efficiently. However, the optimal solution will automatically satisfy Px 2 P
i = 1. Here's why: if we have a feasible vector x with x 2 i < 1, we can scale it up by some factor λ > 1 to get λx with P(λxi)
2 = 1. Since our objective function cos θ is positive (by construction) and linear in the entries, scaling up only improves the objective value. Therefore, the optimizer will naturally choose the boundary case where the constraint becomes tight. Second, regarding the ratio constraints: The Chernoff bounds fundamentally limit how quickly the entries can decay as we move from the largest to the smallest values. The ratio constraints enforce that consecutive entries cannot decay faster than what the Chernoff bounds would allow. Specifically, all entries x2*, . . . , x*n are constrained relative to x1 through these ratios, and all entries xn+1*, . . . , x*2n−1 are constrained relative to x2n.

If some of these ratio constraints become strict (meaning the actual ratios are smaller than the bounds allow), this doesn't violate our theoretical framework—it simply means the actual distribution has even better concentration than our worst-case analysis predicts. Combined with the normalization argument above, the optimizer will find the largest possible x1 and smallest possible x2n (in absolute value) such that the vector has unit norm, while respecting the decay rates imposed by the Chernoff bounds.

Since all these quantities are known given the model parameters n, a, and b, we can incorporate them into our optimization framework. However, we also need to ensure the resulting vector has unit norm.

We introduce the following constraints: