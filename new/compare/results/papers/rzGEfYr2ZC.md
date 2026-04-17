000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Pruning is a common technique to reduce the compute and storage requirements of Neural Networks. While conventional approaches typically retrain the model to recover pruning-induced performance degradation, state-of-the-art Large Language Model (LLM) pruning methods operate layer-wise, minimizing the per-layer pruning error on a small calibration dataset to avoid full retraining, which is considered computationally prohibitive for LLMs. However, finding the optimal pruning mask is a hard combinatorial problem and solving it to optimality is intractable. Existing methods hence rely on greedy heuristics that ignore the weight interactions in the pruning objective. In this work, we instead consider the convex relaxation of these combinatorial constraints and solve the resulting problem using the Frank-
Wolfe (FW) algorithm. Our method drastically reduces the per-layer pruning error, outperforms strong baselines on state-of-the-art GPT architectures, and remains memory-efficient. We provide theoretical justification by showing that, combined with the convergence guarantees of the FW algorithm, we obtain an approximate solution to the original combinatorial problem upon rounding the relaxed solution to integrality.

## 1 Introduction

Pruning after training (Han et al., 2015; Gale et al., 2019; Hoefler et al., 2021; Zimmer et al., 2023; 2025) reduces the inference-time compute and memory footprint of Neural Networks with minimal impact on predictive performance. Conventional approaches obtain such *sparse* models by removing parameters using simple criteria such as their magnitude and then typically require full retraining to recover pruning-induced performance degradation. The drastic increase in model size accompanying the rise of LLMs has, however, reshaped the pruning landscape. At LLM scale, full retraining is often considered prohibitively expensive or even infeasible, resulting in a surge of interest in pruning criteria that do not require retraining. In addition, classical magnitude pruning performs no better than random pruning for LLMs (Sun et al., 2023; Yin et al., 2023), an observation attributed to activation outliers (Dettmers et al., 2022) and highly important *super-weights*
(Yu et al., 2025) in sufficiently large *Transformer* models (Vaswani et al., 2017). Consequently, state-of-the-art methods (Frantar & Alistarh, 2023; Sun et al., 2023; Zhang et al., 2024) prune layerwise: they decompose pruning into per-layer subproblems and treat layers sequentially and independently, estimating parameter importance on a small calibration set by minimizing a per-layer local pruning loss. Specifically, for a single layer with calibration input matrix X ∈ R
din×B and weights W ∈ R
dout×din , the objective is min M
∥W X − (M ⊙ W)X∥
2 F
, s.t. M ∈ {0, 1}
dout×din, ∥M∥0 ≤ k (MASK SELECTION)
where M ∈ {0, 1}
dout×din is a binary mask that enforces the target sparsity, e.g., ∥M∥0 ≤ k for unstructured pruning, and ⊙ denotes the Hadamard product. Here, B = N · L, where N is the number of samples in the calibration batch and L the sequence length. However, even for a single layer, selecting the optimal pruning mask is a hard quadratic binary optimization problem. Solving (MASK SELECTION) to optimality is computationally intractable at LLM scale because the combinatorial constraint—choosing k out of dout × din elements—results in

## Abstract

Anonymous authors Paper under double-blind review

# Don'T Be Greedy, Just Relax! Pruning Llms Via Frank-Wolfe

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

![1_image_0.png](1_image_0.png)

![1_image_1.png](1_image_1.png)

a search space that grows exponentially with the parameter count. Prior methods such as SparseGPT
and Wanda therefore resort to greedy heuristics that ignore weight interactions to remain tractable1.

In this work, we instead consider the convex relaxation of these combinatorial constraints: we approximate (MASK SELECTION) by optimizing over the convex hull of all masks, transforming the combinatorially hard problem into a tractable convex program min M
∥W X − (M ⊙ W)X∥
2 F, s.t. M ∈ [0, 1]dout×din , ∥M∥1 ≤ k (RELAXED MASK SEL.)
where M is now continuous with entries in [0, 1], and the cardinality constraint is replaced by an L1-norm budget, see Figure 1 for a visualization. The resulting convex program can be solved efficiently using the first-order Frank-Wolfe (FW) algorithm (Lacoste-Julien et al., 2013; Zeng & Figueiredo, 2014; Carderera et al., 2021; Braun et al., 2022). Notably, FW is projection-free and moves toward extreme points of the feasible set (i.e., binary masks) via a Linear Minimization Oracle
(LMO), which is efficient to compute and naturally yields sparse updates. Our method, which we term SparseFW, reduces the per-layer pruning error by up to 80% compared to state-of-the-art methods such as Wanda (Sun et al., 2023), and outperforms them on benchmark GPT architectures such as Qwen 2.5, LLaMA 3, Yi 1.5, and Gemma 2, with consistent gains in final WikiText perplexity and zero-shot accuracy. SparseFW is efficient, requires little memory overhead, easily adapts to unstructured and semi-structured sparsity patterns, is simple to implement, and scales to large models. Furthermore, unlike competing methods, SparseFW comes with strong theoretical justification: we show that, combined with the convergence guarantees of FW, rounding the relaxed solution to integrality yields an approximate solution to the original combinatorial problem. Contributions. We summarize our contributions as follows.

1. **SparseFW: A projection-free method for layerwise pruning.** We formulate the layerwise mask selection problem as a convex program over the convex hull of binary masks and propose to solve it with the Frank-Wolfe (FW) algorithm, which is projection-free and leverages an efficient LMO that naturally yields sparse updates. SparseFW is memoryefficient, simple to implement, scales to large models, and can be used to induce both unstructured and semi-structured sparsity patterns.

2. **Strong empirical performance at LLM scale.** SparseFW reduces the per-layer pruning error by up to 70% compared to state-of-the-art methods such as Wanda, and delivers consistent gains in final WikiText perplexity and zero-shot accuracy across modern GPT
architectures (e.g., Qwen 2.5, LLaMA 3, Yi 1.5, Gemma 2).

3. **Theoretical guarantees.** We provide approximation guarantees that connect the relaxed solution returned by FW after rounding to integrality to an approximate solution of the original combinatorial mask selection problem.

Our work demonstrates that classical constrained optimization techniques are not only feasible for pruning LLMs but can drastically improve upon state-of-the-art performance.

1We discuss these methods in detail in Section 2.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Related work. *Pruning after training* (Hoefler et al., 2021) is among the most popular approaches to reduce the resource demands of neural networks during inference. *Magnitude pruning* (Janowsky, 1989; Han et al., 2015) is the de facto default pruning criterion for convolutional architectures, and has been shown to yield pruned models that perform competitively, despite its simplicity (Gale et al., 2019; Zimmer et al., 2023). Various other criteria exist to decide which weights to consider unimportant (cf. LeCun et al., 1989; Hassibi & Stork, 1993; Molchanov et al., 2016; Yeom et al., 2019). With the rise of LLMs, magnitude pruning is being replaced by criteria that account for the peculiarities of LLMs (in particular, large activation outliers, cf. e.g. Dettmers et al., 2022; Yin et al., 2023) and that aim to avoid requiring retraining (Kwon et al., 2022; Frantar & Alistarh, 2023; Sun et al., 2023), which is generally considered computationally prohibitive for large models. Most importantly for our work, SparseGPT (Frantar & Alistarh, 2023), Wanda (Sun et al., 2023), and RIA (Zhang et al., 2024) address the mask selection problem (MASK SELECTION) using a greedy pruning approach, where the selection of weights to prune is performed iteratively. Our approach, on the other hand, relaxes the combinatorial constraint and takes weight interactions into account.

Frank-Wolfe (FW) or *conditional gradient* algorithms (Frank et al., 1956; Levitin & Polyak, 1966) are widely used in Machine Learning for handling complex structural requirements efficiently (Lacoste-
Julien et al., 2013; Zeng & Figueiredo, 2014; Frandi et al., 2015; Jaggi, 2013; Negiar et al. ´ , 2020), with numerous theoretical works (Lacoste-Julien, 2016; Hazan & Luo, 2016; Reddi et al., 2016) and accelerated variants (Hazan & Luo, 2016; Yurtsever et al., 2019; Shen et al., 2019; Combettes et al., 2020; Mokhtari et al., 2018; Chen et al., 2018) appearing in the literature. For a comprehensive review, see Braun et al. (2022). Recently, FW has been applied in the context of neural networks (Ravi et al., 2018; Xie et al., 2019; Berrada et al., 2018; Tsiligkaridis & Roberts, 2020), for training neural networks at scale (Pokutta et al., 2020; Pethick et al., 2025), and Miao et al. (2022) as well as Zimmer et al. (2025) use FW-variants for inducing sparsity throughout pretraining.

## 2 Methodology

We begin by discussing the preliminaries and demonstrating that three state-of-the-art LLM pruning methods, namely SparseGPT, Wanda, and RIA, address the mask selection problem (MASK SELEC- TION) using a greedy pruning approach. We then introduce the FW algorithm and our proposed method, SparseFW. Throughout this section, we use lowercase letters for scalars and vectors and uppercase letters for matrices (W, X, M). Matrix entries are denoted Wij for the element in row i, column j. Rows of matrices are denoted with lowercase subscripts: wi represents the i-th row of matrix W. We use slicing notation, e.g., Xj,: denotes the j-th row of matrix X.

## 2.1 Preliminaries And Greedy Methods

Before discussing SparseGPT, Wanda, and RIA in detail, we first note that the objective in Equation (MASK SELECTION) decomposes into a sum of dout row-wise quadratic functions

$$\|W X-(M\odot W)X\|_{F}^{2}=\sum_{i=1}^{d_{w i}}\|\left(w_{i}-m_{i}\odot w_{i}\right)X\|_{2}^{2},$$
$$(1)$$
2, (1)
with wi ∈ R
din and mi ∈ {0, 1}
din denoting the i-th row of W and M, respectively. Under unstructured sparsity, the constraint in (MASK SELECTION) couples the rows, making the problem non-separable. In contrast, semi-structured patterns such as n:m (prune M −N per block of M
weights) enforce equal per-row sparsity levels and hence fully decouple the rows. For simplicity, we will mainly discuss the row-wise formulation of Equation (1) and drop the index i. We now analyze how SparseGPT, Wanda, and RIA tackle the mask selection problem (MASK SELECTION) through greedy pruning—removing one weight at a time. These methods are optimal for their single-weight pruning objective, effectively bypassing weight interactions to simplify the problem.

SparseGPT (Frantar & Alistarh, 2023) is arguably the most popular approach and is largely based on preceding work (Frantar et al., 2022) of the authors. In practice, it prunes small blocks of weights at a time to ensure scalability to large models, instead of single weights in isolation as suggested by the theory; we briefly describe the underlying approach based on single-weight pruning. Instead of focusing solely on mask selection, SparseGPT approximates the problem of finding a sparse replacement wˆ for the weight vector w, thus combining the problems of mask selection and The greedy-best weight index q and the optimal weight reconstruction are then given by 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

$$S_{ij}^{\rm RIA}:=|W_{ij}|\left(\frac{1}{\sum_{k=1}^{d_{m}}|W_{ik}|}+\frac{1}{\sum_{k=1}^{d_{m}}|W_{kj}|}\right)\ \|X_{j}.\|_{2}.\tag{6}$$
$$\hat{w}=(1-\epsilon_{q})\odot w,q\in[d_{w}]\quad\left\{\left\|(\hat{w}-w)^{\top}X\right\|_{2}^{2}\right\}\tag{4}$$

Plugging the constraints into the objective function directly yields

$$\min_{q\in[d_{m}]}\left\{\left\|\left(\left(1-e_{q}\right)\odot w\right)-w\right)^{\top}X\right\|_{2}^{2}\right\}=\min_{q\in[d_{m}]}\left\{w_{q}^{2}(XX^{\top})_{qq}\right\}\tag{5}$$

Now note that w 2 q(XX⊤)qq = w 2 q ∥Xq,:∥
2 2
. Minimizing the latter over q is equivalent to minimizing |wq| ∥Xq,:∥2
, which is exactly the saliency score of Wanda.

While it might seem that this procedure differs from Wanda, as Wanda computes saliency scores once for all weights and not iteratively, the approaches are identical since the saliency scores do not change after pruning a weight. Wanda further enforces row-wise sparsity rather than unstructured sparsity, pruning a fixed number of weights per row. This has been found beneficial for LLMs (Sun et al., 2023); the same does not hold for other transformer-like models. RIA (Zhang et al., 2024) builds upon Wanda and uses the following saliency score: We employ full-matrix notation since RIA fundamentally depends on the matrix structure for its rowand column-wise renormalization. Letting W′ denote the rescaled weight matrix with entries

$$W_{i j}^{\prime}:=W_{i j}\left({\frac{1}{\sum_{k=1}^{d_{\mathrm{in}}}|W_{i k}|}}+{\frac{1}{\sum_{k=1}^{d_{\mathrm{out}}}|W_{k j}|}}\right)$$
.
Applying Wanda on W′to prune the weights with the smallest saliency scores yields

$$|W_{i j}^{\prime}|\,\|X_{j,:}\|_{2}=:S_{i j}^{\mathrm{RIA}},$$
ij , (7)
which is exactly the saliency score of RIA. The RIA criterion can be interpreted as using the same greedy pruning algorithm as Wanda, but applied to a rescaled weight matrix. Since solving this problem exactly is intractable, SparseGPT follows a greedy procedure to approximately solve it: at each step it finds the optimal *single* weight to prune and the corresponding optimal remaining weights, i.e., it solves

$$(2)$$
$$\operatorname*{min}_{\hat{w},q\in[d_{m}]\mathrm{{st.}}e_{q}^{\top}\hat{w}=0}\quad\|(\hat{w}-w)^{\top}X\|_{2}^{2}.$$
$$({\mathfrak{I}})$$
$$w^{*}=w-{\frac{w_{q}}{[(X X^{\top})^{-1}]_{q q}}}\,(X X^{\top})^{-1}e_{q},{\mathrm{~where~}}q\in\arg\operatorname*{min}_{q\in[d_{\mathrm{nl}}]}{\frac{w_{q}^{2}}{((X X^{\top})^{-1})_{q q}}}.$$
.
Wanda (Sun et al., 2023) computes a saliency score Si,j := |Wi,j | ∥Xj,:∥2 for each weight and then prunes the weights with the smallest saliencies. The authors motivate their approach by the observation that in LLMs, some weights with small magnitudes correspond to large-magnitude features (cf. e.g. Dettmers et al., 2022) and that their removal can lead to significant performance drops, despite their small magnitude. Wanda hence multiplies magnitude saliencies by the corresponding input activation norm to avoid pruning such small-but-important weights. We argue that Wanda can be seen as a greedy approximation to (MASK SELECTION) and focus on a single row w for simplicity. Again, we write the optimization problem for pruning one variable, but now without modifying the remaining weights: reconstruction of remaining weights by solving

$$\min_{\hat{w}}\|w^{\top}X-\hat{w}^{\top}X\|_{F}^{2},\quad\mbox{s.t.}\|\hat{w}\|_{0}\leq k.\tag{1}$$

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

We present an alternative approach to the greedy approximations discussed in the previous section, which is based on relaxing the combinatorial constraints to obtain a convex optimization problem, instead of trying to make the problem tractable by making the pruning decision on a per-weight basis. We solve the convex problem using the FW algorithm, which we introduce in the following.
The Frank-Wolfe Algorithm. When minimizing some objective function L over a set of constraints C, a classical approach is Projected Gradient Descent (PGD) which iteratively performs a gradient
step and then projects the result back to the constraint set to ensure feasibility of the iterates. However,
depending on C, this projection step may not admit an analytic solution and can be computationally expensive (Jaggi, 2013; Combettes & Pokutta, 2021). The FW algorithm is an alternative which is projection-free and often yields solutions with desirable structure. Instead of moving along the gradient direction and then requiring a projection step, FW moves towards the boundary point of the feasible region that is best aligned with the descent direction. Specifically, in each iteration t and at
iterate Mt, FW calls a Linear Minimization Oracle (LMO) on the gradient ∇L(Mt) of L at Mt to
solve
$$V_{t}=\operatorname*{arg\,min}_{V\in{\mathcal{C}}}\langle V,\nabla{\mathcal{L}}(M_{t})\rangle,$$
⟨V, ∇L(Mt)⟩, (8)
which is then used to update the parameters using the convex combination

$\left(8\right)$. 
$$M_{t+1}\leftarrow(1-\eta_{t})M_{t}+\eta_{t}V_{t},$$
$\eqref{eq:walpha}$. 
$${\mathcal{C}}_{k}=\left\{M\in[0,1]^{d_{\mathrm{out}}\times d_{\mathrm{in}}}:\|M\|_{1}\leq k\right\}.$$
$$(10)^{\frac{1}{2}}$$

$$(11)^{\frac{1}{2}}$$

Mt+1 ← (1 − ηt)Mt + ηtVt, (9)
where ηt ∈ [0, 1] is the step size. Throughout this work, we stick to the learning rate schedule
given by ηt =2
t+2 . If now M0 ∈ C, then the convex update rule ensures feasibility of all iterates.
In practice, solving Equation (8) is often much cheaper than performing a projection step. If C is
further given by the convex hull of a set of points, e.g., the vertices of a polytope, then the solution to Equation (8) is attained at one of these points. In each iteration, FW moves towards the vertices.
Relaxing the combinatorial constraints. The FW algorithm can only be applied to convex constraint
sets, which is not the case for (MASK SELECTION). We make the problem tractable by relaxing the
combinatorial constraints to their convex hull, i.e.,
Ck =M ∈ [0, 1]dout×din : ∥M∥1 ≤ k	. (10)
Given that the objective function of (MASK SELECTION) is a convex quadratic, this relaxation transforms the combinatorial mask selection problem into a convex optimization problem, which can be solved efficiently using the FW algorithm. We restate the reformulation of (RELAXED MASK SEL.) for completeness:
$$\operatorname*{min}_{M\in{\mathcal{C}}_{k}}\|W X-(M\odot W)X\|_{F}^{2}.$$
. (11)
$$[\mathsf{LMO}\left(\nabla{\mathcal{L}}(M_{t})\right)]_{i j}=\begin{cases}1&\mathrm{if}\left(i,j\right)\in\mathsf{Top-k}\left(-\nabla{\mathcal{L}}(M_{t})\right),[\nabla{\mathcal{L}}(M_{t})]_{i j}<0\\ 0&\mathrm{otherwise}\end{cases}.$$
where Top-k(∇L(Mt)) denotes the set of indices corresponding to the k entries of ∇L(Mt) with the smallest values. The LMO for Ck can be computed efficiently and naturally produces sparse updates: at most k out of dout · din entries are nonzero. While the above corresponds to unstructured sparsity, the LMO can be adapted to per-row sparsity and n:m sparsity; see Appendix D. This relaxation has the advantage that, unlike the previously discussed greedy approaches, it fully accounts for interactions between weights. However, the solution to the relaxed problem (RELAXED MASK SEL.) is not guaranteed to be feasible for the original problem (MASK SELECTION); in Section 4, we show that rounding the relaxed solution to integrality yields an approximate solution to the original problem. The sparse Linear Minimization Oracle. We next discuss how to compute the LMO for the feasible set Ck. Note that Ck is a polytope and can be described as the convex hull of its vertices, which are exactly the binary masks with at most k ones. At any vertex, all coordinates lie on box bounds 0 or 1, and the coupling constraint Pi,j Mij ≤ k is either inactive (fewer than k ones) or tight (exactly k ones); see Figure 1. Minimizing a linear function over Ck therefore consists of selecting up to k entries with the most negative coefficients and setting them to one, leaving the rest at zero. Letting
∇L(Mt) ∈ R
dout×din denote the gradient of the objective at iterate Mt, the LMO solution at step t is hence given by

$$(12)$$

## 2.2 Solving The Convex Relaxation With Frank-Wolfe

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 We present the full SparseFW algorithm in Algorithm 1. At a high level, for each layer we solve the relaxed optimization problem using the FW algorithm, starting from any binary mask that satisfies the sparsity constraints. After running for T iterations, we threshold the learned mask—whose entries lie in [0, 1]—to obtain a binary mask that meets the original sparsity constraints. The objective function and the gradient with respect to Mt are given by

$\mathcal{L}(M_{t})=\mathrm{Tr}(W(1-M_{t})XX^{\top}(1-M_{t})^{\top}W^{\top})$  $\mathcal{V}\mathcal{L}(M_{t})=-2\cdot W\odot(WXX^{\top}-(W\odot M_{t})XX^{\top})$.  
Even for small calibration datasets, the activation matrix X can be very large. For example, the largest matrix in a LLaMA-2-7B transformer block (up proj) has din = 4096. With N = 128 samples and sequence length L = 4096, X has dimensions 4096 × 524,288. Because both the objective and the gradient depend only on G := XX⊤ (which can be computed in batches), we precompute G := XX⊤ and H := W G once to drastically reduce resource demands. Note that G has dimensions 4096 × 4096, in contrast to the 4096 × 524,288 dimensions of X; this independence of the sequence length L and number of samples N is crucial for efficiency. With G and H precomputed, the gradient requires only two elementwise multiplications, a matrix–matrix multiplication, and a matrix addition:

## ∇L(Mt) = −2 · W ⊙ (H − (W ⊙ Mt)G).

In practice, we have to navigate a caveat that we did not detail in Algorithm 1 for the sake of simplicity, exact details are in the appendix. Throughout the experiments, we noticed that while FW often substantially reduces pruning error relative to baselines like Wanda, it can still produce worse final perplexity, likely due to a mismatch between local and global objectives. Constraining Sparse Frank-Wolfe (SparseFW) by fixing a fraction of very high-saliency weights (e.g., those with highest Wanda scores) as unprunable consistently improves performance. This suggests that Wanda reliably identifies weights that should be preserved, even if a more thorough local optimization would prune them. We therefore fix these weights and apply FW to the remaining ones, optimizing over a smaller search space. We ablate the impact of this ratio in Table 2 in the appendix: Surprisingly, we observe the best consistent improvements when setting α = 0.9, i.e., fixing 90% of the highest saliency weights and optimizing only over the remaining 10%. Even small α values (e.g., α = 0.1) can yield significant perplexity improvements. On the other hand, setting α = 0.0 (full FW without any fixed weights) consistently yields worse results than the baselines.

| Algorithm 1 SparseFW Require: Weight matrix W, input X, no. of nonzero entries k, iterations T, warm-start mask M0 1: G = XX⊤, H = W G ▷ Precompute buffers 2: for t = 0 to T − 1 do 3: ∇L(Mt) = −2 · W ⊙ (H − (W ⊙ Mt)G) ▷ Compute gradient 4: Vt = LMO ∇L(Mt), Ck  ▷ Compute LMO 5: ηt ← 2 t+2 6: Mt+1 ← (1 − ηt)Mt + ηtVt ▷ FW Update  1 if (i, j) ∈ Top-k(MT ) 7: [M]ij ← 0 otherwise ▷ Threshold 8: return M   |
|---|

We present our experimental methodology; our code will be made publicly available to ensure reproducibility. Our focus is on language modeling and we utilize pretrained models from Hugging-
Face (Wolf et al., 2020), including *LLaMA-3.1-8B* (Grattafiori et al., 2024), *Gemma-2-9B* (Riviere et al., 2024), *Yi-1.5-9B* (Young et al., 2025), *DeepSeek-7B-base* (Bi et al., 2024), and *Qwen2.5-7B* (Yang et al., 2025). For the calibration set, we randomly sample 2048-token sequences from the C4 dataset (Raffel et al., 2020). For validation, we select 100 sequences from the validation split. We evaluate performance using perplexity on *WikiText* (Merity et al., 2016) and zero-shot accuracy on

## 2.3 The Sparsefw Algorithm 3 Experimental Results

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png)

the EleutherAI evaluation set (Gao et al., 2023). Following Sun et al. (2023), we prune all linear layers with a uniform sparsity allocation across layers, while keeping the embedding and final linear head dense. SparseFW is compared with Wanda and RIA, as these methods also aim to find a better pruning mask by solving (MASK SELECTION); we hence do not compare directly to methods that involve a reconstruction step, such as SparseGPT (Frantar & Alistarh, 2023). We report results for both unstructured and semi-structured sparsity (Mishra et al., 2021).

SparseFW outperforms state-of-the-art mask selection methods. In Table 1, we compare SparseFW (warm-started from Wanda or RIA) to the respective baselines across five state-of-the-art GPTs and multiple sparsity regimes (50%, 60%, and 2:4). SparseFW generally performs on par with or better than the baselines in terms of perplexity; for zero-shot accuracy, SparseFW consistently outperforms competing methods. We generally observe much more consistent and bigger improvements in the higher sparsity regimes than for 50% sparsity. SparseFW successfully optimizes the matrix-wise pruning objective. We observe consistent improvement in terms of the local pruning objective over both Wanda and RIA warmstarts. Figure 2 shows the per-layer reductions relative to a Wanda Warmstart, where we observe reductions of up to 80%. In general, we found the average relative reduction over the layers to range between 20% and 40% across the different models, sparsity regimes and warmstarts.

![6_image_1.png](6_image_1.png)

Sample and iteration efficiency. Figure 3 ablates the impact of the number of SparseFW iterations (left) and the number of calibration samples (right). Fixing the amount of samples at 256, perplexity decreases up to around 2000 iterations and then flattens. We therefore use 2000 iterations throughout. In contrast, at a fixed 2000 iterations, increasing the number of calibration samples from 64 to 512 brings substantial additional perplexity gains. This trend contrasts with Wanda, whose performance 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Table 1: Perplexity (↓, lower is better) and zero-shot accuracy (↑, higher is better) comparison. We report SparseFW performance with Wanda and RIA warmstart for unstructured 50% and 60% sparsity and semi-structured 2:4 sparsity after 2000 iterations using 256 samples compared to the baseline warmstarts. We indicate the SparseFW warmstart method in parentheses. Best values are highlighted in bold. We omit standard deviations for legibility.

| Perplexity (↓)    | GEMMA-2   | YI-1.5   | DEEPSEEK-7   | QWEN2.5   | LLAMA-3   |       |    |
|-------------------|-----------|----------|--------------|-----------|-----------|-------|----|
| Method            | Sparsity  | 9B       | 9B           | 7B        | 7B        | 14B   | 8B |
| Wanda             | 11.19     | 6.58     | 7.79         | 8.45      | 7.11      | 10.09 |    |
| RIA               | 11.19     | 6.71     | 7.90         | 8.54      | 7.01      | 9.88  |    |
| 50%               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 10.67     | 6.58     | 7.89         | 8.35      | 7.10      | 10.21 |    |
| SparseFW (RIA)    | 10.77     | 6.53     | 7.93         | 8.22      | 6.98      | 9.95  |    |
| Wanda             | 16.46     | 11.38    | 11.44        | 13.47     | 10.87     | 21.53 |    |
| RIA               | 17.17     | 14.37    | 11.87        | 12.86     | 9.78      | 19.14 |    |
| 60%               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 14.83     | 10.56    | 11.99        | 12.44     | 10.28     | 17.97 |    |
| SparseFW (RIA)    | 15.07     | 10.67    | 12.41        | 11.66     | 9.65      | 18.16 |    |
| Wanda             | 17.41     | 11.58    | 11.76        | 14.40     | 11.37     | 24.82 |    |
| RIA               | 16.78     | 11.27    | 12.04        | 13.46     | 10.98     | 23.7  |    |
| 2:4               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 15.81     | 10.61    | 11.73        | 14.16     | 11.82     | 20.45 |    |
| SparseFW (RIA)    | 15.83     | 10.35    | 11.91        | 13.42     | 11.20     | 21.31 |    |
| Accuracy in % (↑) | GEMMA-2   | YI-1.5   | DEEPSEEK-7   | QWEN2.5   | LLAMA-3   |       |    |
| Method            | Sparsity  | 9B       | 9B           | 7B        | 7B        | 14B   | 8B |
| Wanda             | 68.44     | 61.04    | 56.67        | 63.72     | 67.94     | 58.78 |    |
| RIA               | 68.71     | 61.22    | 55.76        | 64.03     | 67.83     | 58.94 |    |
| 50%               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 68.42     | 62.49    | 56.8         | 64.97     | 69.44     | 60.17 |    |
| SparseFW (RIA)    | 68.67     | 62.53    | 56.24        | 65.34     | 69.19     | 59.63 |    |
| Wanda             | 63.19     | 53.7     | 50.51        | 59.44     | 63.58     | 48.08 |    |
| RIA               | 63.19     | 53.7     | 50.51        | 59.44     | 63.58     | 48.08 |    |
| 60%               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 64.46     | 54.90    | 50.56        | 61.13     | 65.59     | 51.92 |    |
| SparseFW (RIA)    | 65.35     | 55.41    | 50.65        | 61.52     | 65.80     | 52.15 |    |
| Wanda             | 63.75     | 52.92    | 50.65        | 59.11     | 63.39     | 47.13 |    |
| RIA               | 63.83     | 52.41    | 51.08        | 58.48     | 63.85     | 47.77 |    |
| 2:4               |           |          |              |           |           |       |    |
| SparseFW (Wanda)  | 63.81     | 53.78    | 51.12        | 60.15     | 64.12     | 48.43 |    |
| SparseFW (RIA)    | 63.90     | 52.54    | 50.69        | 60.15     | 64.35     | 48.54 |    |

does not seem to increase significantly with additional calibration data: increasing the sample count from 64 to 512 leads to a perplexity decrease from 25.1 to only 24.6 for Wanda. Overall, SparseFW is clearly more compute-intensive than Wanda and RIA, but we argue that spending more resources once to improve the performance of pruned models is, given that deployed LLMs now serve millions of users and inference costs scale with the number of requests, worthwhile. That being said, the results of Figure 3 indicate clear benefits of increasing the number of samples while keeping the number of iterations fixed and relatively low. While more samples require slightly more compute to build the matrix G = XX⊤, the cost of a single FW iteration is independent of the sample count.

## 4 Theoretical Results

In this section, we state a data-dependent error guarantee for the mask produced by SparseFW with respect to the original pruning objective (MASK SELECTION). This is a key benefit of SparseFW over greedy heuristics, which can yield suboptimal solutions even though the objective function is convex. We state our main result informally here, deferring full statements and proofs to the appendix. Lemma 1 (Informal). After T iterations of SparseFW, the resulting mask M *satisfies*

$${\mathcal{L}}(M)-{\mathcal{L}}(M^{*})\leq\lambda_{\operatorname*{max}}\left(Q\right)\left({\frac{k}{T}}+2\left(k+{\sqrt{2d_{i n}d_{o u t}k}}\right)\right)$$

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png)

where M∗*is an optimal mask for* (MASK SELECTION), k is the maximum number of nonzeros in the mask, Q *represents the Hessian of the objective function and* λmax(Q) *its largest eigenvalue.*
Note that Q is not equal to G = XX⊤, the latter being the Hessian of the objective w.r.t. reconstruction of the weights, not w.r.t. the mask. The bound captures two sources of error: (i) the optimization error from solving the relaxed problem (RELAXED MASK SEL.), and (ii) the *thresholding error* from converting a relaxed solution to a binary mask (Line 7 in Algorithm 1).

Optimization error. After T iterations of the FW algorithm, the resulting (continuous, not-yetthresholded) mask MT satisfies

## L(Mt ) − L(Mˆ ) ≤ Kλmax(Q)/T,

where Mˆ is an optimal solution to the relaxed problem (RELAXED MASK SEL.). In other words, by increasing the number of iterations T, FW can guarantee an arbitrarily small optimization error.

Thresholding error. The error due to thresholding can be controlled by the curvature of the objective
(captured by λmax(Q)) and the distance between the fractional iterate and its thresholded version, which in turn can be upper bounded in terms of k and the dimension of the input space dindout.

These insights explain the empirical behavior in Figure 4. The left panel reports the relative pruning error reduction (higher is better) versus FW iterations for the continuous and thresholded masks. After a short initial drop, due to the large stepsize, the continuous iterate improves consistently, as predicted by the FW convergence guarantee. In contrast, the thresholded mask first degrades as the thresholding error grows while the iterate moves through the interior of Ck. This is reflected in the right panel, which shows the average threshold residual (the *∥ · ∥*1 distance between the continuous and thresholded masks): It first rises steeply, then decreases and eventually plateaus above zero. As long as the relaxed solution is not at a vertex, the thresholding error remains nonzero, so the thresholded curve does not fully catch up to the continuous one.

## 5 Conclusion

Solving the pruning mask selection problem for LLMs is a hard combinatorial problem. In this work, we relax the binary constraints to their convex hull and solve the resulting convex problem with the FW algorithm; we call this approach SparseFW, a simple and memory-efficient layerwise method that explicitly accounts for weight interactions and supports both unstructured and semi-structured sparsity. Across modern GPT architectures, SparseFW drastically reduces the per-layer reconstruction error and improves perplexity and zero-shot accuracy over state-of-the-art LLM pruning approaches. Our work demonstrates that classical constrained optimization is a scalable and effective alternative to greedy heuristics for LLM pruning. However, our work is not without limitations. Although vanilla FW substantially reduces per-layer pruning error, this does not reliably yield lower perplexity. Without fixing part of the mask, it tends to prune weights crucial for overall performance. SparseFW successfully mitigates this by preserving a fraction of high-saliency weights from the warmstart, but the local–global objective mismatch persists; inductive biases still appear necessary for improved perplexity.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Leonard Berrada, Andrew Zisserman, and M. Pawan Kumar. Deep frank-wolfe for neural network optimization. *International Conference on Learning Representations 2019*, November 2018.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo, Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism, January 2024. URL
http://arxiv.org/abs/2401.02954.

Gabor Braun, Alejandro Carderera, Cyrille W Combettes, Hamed Hassani, Amin Karbasi, Aryan ´
Mokhtari, and Sebastian Pokutta. Conditional gradient methods. November 2022. URL https:
//conditional-gradients.org/.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/10256836.

Alejandro Carderera, Sebastian Pokutta, Christof Schutte, and Martin Weiser. Cindy: Conditional ¨
gradient-based identification of non-linear dynamics - noise-robust recovery. January 2021.

Lin Chen, Christopher Harshaw, Hamed Hassani, and Amin Karbasi. Projection-free online optimization with stochastic gradient: From convexity to submodularity. In International Conference on Machine Learning, pp. 814–823. PMLR, 2018.

Cyrille W. Combettes and Sebastian Pokutta. Complexity of linear minimization and projection on some sets. January 2021.

Cyrille W. Combettes, Christoph Spiegel, and Sebastian Pokutta. Projection-free adaptive gradients for large-scale optimization. September 2020.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale. August 2022.

Emanuele Frandi, Ricardo Nanculef, Stefano Lodi, Claudio Sartori, and Johan A. K. Suykens. Fast and scalable lasso via stochastic frank-wolfe methods with a convergence guarantee. October 2015.

Marguerite Frank, Philip Wolfe, et al. An algorithm for quadratic programming. Naval research logistics quarterly, 3(1-2):95–110, 1956.

Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned in one-shot. In *International Conference on Machine Learning*, pp. 10323–10337. PMLR, 2023.

Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. August 2022.

Trevor Gale, Erich Elsen, and Sara Hooker. The state of sparsity in deep neural networks. *arXiv* preprint arXiv:1902.09574, 2019.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzman, Frank Zhang, ´ Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur C¸ elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai