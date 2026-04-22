# Mini-batch kernel $k$-means

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 4, 4, 6

## Abstract
We present the first mini-batch kernel $k$-means algorithm, offering an order of magnitude improvement in running time compared to the full batch algorithm. A single iteration of our algorithm takes $\widetilde{O}(kb^2)$ time, significantly faster than the $O(n^2)$ time required by the full batch kernel $k$-means, where $n$ is the dataset size and $b$ is the batch size. Extensive experiments demonstrate that our algorithm consistently achieves a 10-100x speedup with minimal loss in quality, addressing the slow runtime that has limited kernel $k$-means adoption in practice. We further complement these results with a theoretical analysis under an early stopping condition, proving that with a batch size of $\widetilde{\Omega}(\max \set{\gamma^{4}, \gamma^{2}}\cdot k\epsilon^{-2} )$, the algorithm terminates in $O(\gamma^2/\epsilon)$ iterations with high probability, where $\gamma$ bounds the norm of points in feature space and $\epsilon$ is a termination threshold. Our analysis holds for any reasonable center initialization, and when using $k$-means++ initialization, the algorithm achieves an approximation ratio of $O(\log k)$ in expectation. 
For normalized kernels, such as Gaussian or Laplacian it holds that $\gamma=1$. Taking $\epsilon = O(1)$ and $b=\Theta(k\log n)$, the algorithm terminates in $O(1)$ iterations, with each iteration running in $\widetilde{O}(k^3)$ time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the first mini-batch variant of kernel k-means that enables significantly faster iterations while retaining the accuracy benefits of kernelization. The authors show that kernel Lloyd-style updates can be computed using a recursive formulation without explicitly storing centers in feature space, and further accelerate the method by truncating historical center contributions to obtain a per-iteration cost of $\tilde O(kb^2)$. A dimension-independent termination bound is established under an early-stopping rule, and the use of k-means++ initialization preserves the $O(\log k)$ approximation guarantee. Experiments on multiple datasets demonstrate that the approach achieves clustering quality close to full kernel k-means while improving runtime by one to two orders of magnitude.

### Strengths
The main contributions of the work:
1. A dimension-independent convergence analysis for mini-batch kernel k-means under early stopping.
2. A recursive DP-based update rule reducing per-iteration kernel Lloyd cost.
3. A truncation strategy enabling efficient sparse center representation with minimal quality degradation.
4. Empirical evaluation showing significant speedups and competitive clustering quality.

### Weaknesses
1. While the authors provide a bound on the per-iteration deviation between the mini-batch update and the full-data update (Lemma B.11), it remains unclear whether such errors accumulate across iterations. Therefore, the theoretical contribution falls short of providing a approximation guarantee to the original kernel $k$-means objective, especially under truncation where historical contributions are dropped.
2. The paper applies kernel k-means in a general RKHS but does not discuss the necessary expressiveness of the feature space to support the underlying cluster structure, treating the feature mapping as a generic embedding rather than a hypothesis space with theoretical constraints. Furthermore, reusing notation from standard k-means (Kanungo et al., 2004) leads to ambiguity in the kernel setting: the same distance symbol $\Delta(\cdot,\cdot)$ is used across different spaces, and kernel evaluations are not made explicit in key definitions, which obscures the algorithmic interpretation in RKHS.
3. The core idea of combining mini-batch updates with kernel k-means naturally yields runtime improvements by avoiding full data sweeps each iteration. Moreover, several theoretical components, including the termination analysis under early stopping and the learning-rate schedule, seem to be direct extensions of the framework introduced by [Schwartzman (2023)], rather than introducing fundamentally new insights beyond that prior work. Overall, while the paper is technically sound, the incremental nature of the contribution weakens its impact.
4. Another significant concern lies in the numerous presentation and formatting issues. For example, several references are incorrectly formatted (e.g., Quanrud in line 190 lacks a publication year), and line 211 contains an extraneous "and." Additionally, the notation for $X$ appears inconsistent between line 44 and line 219. Since the function $f_x$ is formally defined at line 222, it would improve logical flow to incorporate it earlier into the formulation of the objective at line 221.
5. (Minor) The manuscript lacks a proper conclusion section that summarizes the key findings and clarifies their broader significance, making the paper incomplete.

Considering the concerns regarding methodological novelty, theoretical soundness, and several presentation issues discussed above, I recommend rejection.

### Questions
1. Can you provide a global deviation bound showing that the mini-batch update trajectory remains close to the full-data kernel k-means trajectory, and consequently that the final centers provably approximate the solution of the kernel k-means objective?
2. While the experiments demonstrate solid performance on mid-sized benchmarks (e.g., MNIST, HAR), can the authors add evaluations on significantly larger datasets (n ≫ 100k) or high-dimensional embedding domains such as deep image or text representations?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the first mini-batch variant of kernel $k$-means and a practical truncation scheme that makes each iteration fast. A single iteration runs in $\tilde{O}(k b^2)$ time (where $k$ is the number of clusters, and $b$ is the batch size), instead of the $O(n^2)$ time complexity for full-batch kernel $k$-means. By incorporating a dynamic-programming view, this paper also sketches an $O(n(b+k))$ bound for per-iteration path before truncation. Empirically, they report 10–100× speedups with small quality loss. Theoretically, under an early-stopping rule, if the batch size is $b=\Omega(\max${$γ^4,γ^2kε^−2\log^2(γn/ε)\$}$)$, the algorithm stops in $O(\gamma^2/\epsilon)$ iterations with high probability. With k-means++ seeding as initialization, it can preserve the $O(\log k)$-approximation in expectation. For normalized kernels $(γ=1)$ and batch size $b=Θ(k\log n)$, the per-iteration cost can be reduced to $O(k^3\log^2n)$. Experiments on datasets such as MNIST, PenDigits, and Letters confirm strong clustering performance and scalability across Gaussian, Laplacian, and k-NN kernels.

### Strengths
The truncation of center updates yields $\tilde{O}(k b²)$ per iteration while tracking the untruncated dynamics closely, matching the empirical 10–100× speedups. The motivation for studying the mini-batch kernel $k$-means setting is clear. The proposed early-stopping analysis removes explicit dependence on (possibly infinite) feature dimension by working with parameter $\gamma$. With $k$-means++ initialization, the $O(\log k)$ approximation ratio remains in expectation. For multiple datasets (MNIST, PenDigits, Letters, HAR) and kernels (Gaussian, heat, k-NN), the proposed truncated mini-batch method often matches full kernel k-means quality with much better running times.

### Weaknesses
The main weakness can be summarized as follows.

1. The theory guarantees termination only when batch improvement falls below $\varepsilon$. It does not prove convergence to a local optimum or monotone decrease of the full objective beyond expectation bounds. This may leave practical users unsure how to set $\varepsilon$ across datasets. 

2. The quality of the proposed methods are sensitive to the kernel and hyper-parameter κ\kappaκ (e.g., Gaussian width), where the paper tunes with a heuristic + manual adjustments. A systematic sensitivity or auto-tuning study is missing. 

3. Kernel-sketch baselines are run with a fixed embedding size (150). Allowing stronger sketch sizes could alter the ranking, which deserves a further study on scaling curves. 

4. The $\tilde{O}$ notation hides polylog factors. For each iteration, it also requires $O(k\tau)$ extra memory for truncated center representations, and practical guidance for choosing $\tau$ vs. $b$ is missing. 

5. Claim of “first” mini-batch kernel $k$-means is plausible but would benefit from a sharper positioning relative to coresets/sketching/approx or other kernel methods.

### Questions
1.How should the users pick the early-stopping threshold across datasets/kernels to balance speed vs. quality? Any adaptive rule tied to variance of batch improvements?

2.If sketch dimension is increased (e.g., to match runtime), do the sketching baselines catch up in quality? A runtime–quality Pareto curve would clarify.

3.Can the authors bound the final full-data objective gap between truncated and untruncated algorithms as a function of τ(beyond Lemma 4.1’s center difference), or give a post-hoc certificate?

4.The authors note γis often ≪1 for heat/k-NN kernels. Can you translate that into concrete batch-size savings and guidance for kernel choice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents the first mini-batch kernel k-means with a truncated center update that reduces per-iteration cost to O(kb²) and delivers 10–100× speedups with small quality loss; it also provides early-stopping guarantees parameterized by γ and an O(log k) expected approximation when initialized with k-means++.

### Strengths
1. Clear algorithmic idea: truncating center histories to maintain sparse convex combinations, enabling fast assignments and updates while bounding the truncation error. 
2. Concrete theory for an inherently stochastic procedure:$\gamma$ -parameterized early-stopping bounds and k-means++ approximation inheritance without explicit feature dimensionality. 
3. Practical speedups with small quality loss across multiple datasets; plots distinguish kernel-evaluation overhead from the rest, aiding interpretation. 
4. Complementarity to existing accelerations (coresets, random features, Nyström/sketching) rather than exclusivity.

### Weaknesses
1. The theoretical results guarantee termination under an $\varepsilon$-threshold but do not ensure convergence to a stationary or local optimum. It would be helpful to relate $\varepsilon$ to objective suboptimality and explore whether a monotone (expected) decrease in the objective can be established.  
2. The performance depends heavily on kernel choice and the bandwidth parameter $\kappa$. The current approach relies on heuristics and manual tuning; incorporating sensitivity studies or automated selection (e.g., cross-validation or median heuristic) would enhance reproducibility.  
3. Although the theory suggests $\tau$, experiments indicate that much smaller $\tau$ suffices. Ablations showing how $\tau$ affects accuracy, runtime, and memory footprint would clarify this discrepancy.  
4. While per-iteration complexity is $\tilde{O}(k b^2)$, kernel computation may dominate the total runtime, especially in high-dimensional settings. A detailed runtime breakdown (kernel evaluation, assignment, update) would make scalability claims more transparent.  
5. The experiments cover only four datasets and fix the iteration number at 200 instead of using a convergence criterion. Expanding to additional datasets and comparing runs at matched objective tolerances or time budgets would better support robustness.  
6. Comparisons with coreset-based kernel $k$-means, random-feature, or Nyström approximations under equivalent computational budgets would strengthen empirical evidence. Demonstrating that the proposed method composes well with these accelerations would highlight its versatility.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents mini-batch kernel k-means, a scalable variant of kernel k-means that introduces recursive and truncated updates to handle large datasets efficiently. The method achieves a major reduction in complexity (from 𝑂(𝑛^2) to 𝑂~(𝑘𝑏^2)) while preserving clustering quality. Theoretical convergence and empirical results on several benchmarks support the approach.

### Strengths
1. Addresses an important scalability bottleneck in kernel clustering; 2. The recursive center representation in RKHS is clean and well-motivated; 3. Theoretical analysis is solid and well connected to the algorithm; 4. Experiments are carefully executed and show consistent improvements in runtime; 5. Writing is clear and easy to follow.

### Weaknesses
1. Experiments are limited to small and mid-scale datasets; large-scale tests would strengthen the contribution;  2. The influence of the truncation window 𝜏 and batch size 𝑏 is not analyzed; 3. Runtime breakdowns could be clearer—kernel computation cost may dominate in practice; 4. Novelty is moderate: the work is an effective adaptation rather than a conceptual leap.

### Questions
1. Does the early-stopping guarantee hold empirically when τ is much smaller than the theoretical lower bound (as mentioned, τ ≪ b)? 2. How sensitive is the algorithm to kernel choice (e.g., Gaussian vs Laplacian vs polynomial) in terms of both runtime and accuracy? 3. Could this approach be extended to other kernelized clustering objectives (e.g., kernel spectral clustering, kernel k-medoids)? 4. Are there any plans to release the code or scikit-learn integration (since it was mentioned that a pull request was accepted)?

I will be happy to improve my score if the authors could resolve my doubts.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the first mini-batch kernel k-means algorithm with a truncated center update strategy that maintains centers as sparse convex combinations in feature space and enables sublinear time complexity for per-iteration updates. A dynamic-programming view yields an $O(n(b+k))$ update, then truncation delivers an iteration time of $\tilde{O}(kb^2)$ with a provably small additive error. This paper gives a termination analysis under early stopping, which shows fixing a certain batch size (parameter $b$) suffices to yield a time complexity of $O(\gamma^2/\varepsilon)$ for each iteration.  With k-means++ seeding, the proposed method can achieve expected approximation of $O(\log k)$. Experiments on shows speedups over full-batch kernel k-means with comparable clustering quality.

### Strengths
1. The paper introduces a clear truncation scheme for mini-batch kernel k-means that preserves the center as a sparse convex combination and achieves $\tilde{O}kb^2)$ time complexity per iteration, which is a meaningful practical advance over quadratic full-batch updates. 

2. The theoretical analysis is clear, including a termination bound depending on batch-size requirements, and an $O(\log k)$ expected approximation under k-means++ initialization. 

3. The presentation connects an untruncated dynamic program with the truncated version, giving a coherent path from correctness to efficiency. 

4. The empirical study is straightforward and reports consistent speedups with small quality loss.

### Weaknesses
1. This paper lacks comprehensive comparisons between coresets and Nyström methods.

2. The empirical evaluation is relatively narrow in domain and scale, with four classic datasets and manual kernel parameter tuning, leaving open how the approach behaves on million-point problems or under automated bandwidth selection. 

3. Memory costs for storing truncated centers and required kernel evaluations are not deeply quantified, and kernel computation time may dominate in practice despite the black bars in plots. 

4. The empirical comparison to state-of-the-art acceleration via coresets and Nyström with relative-error guarantees appears limited.

### Questions
1. How does the method scale on much larger datasets in the one to ten million point range, and what is the proportion of total time spent on kernel computation versus center updates in that regime. 

2. Would stronger baselines using recent coreset or Nyström methods with relative-error bounds change the empirical picture, and can your truncation be composed with those techniques as suggested in the related-work discussion. 

3. What's the main advantages of the proposed method against coreset and Nyström methods?

### Soundness
3

### Presentation
2

### Contribution
2
