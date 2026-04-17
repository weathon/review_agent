# VAMO: Efficient Zeroth-Order Variance Reduction for SGD with Faster Convergence

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4

## Abstract
Optimizing large-scale nonconvex problems, common in deep learning, demands balancing rapid convergence with computational efficiency. First-order (FO) optimizers, which serve as today’s baselines, provide fast convergence and good generalization but often incur high computation and memory costs due to the large size of modern models. Conversely, zeroth-order (ZO) algorithms reduce this burden using estimated gradients, yet their slow convergence in high-dimensional settings limits practicality. We introduce VAMO (VAriance-reduced Mixed-gradient Optimizer), a stochastic variance-reduced method that extends mini-batch SGD with full-batch ZO gradients under an SVRG-style framework. VAMO's hybrid design utilizes a two-point ZO estimator to achieve a dimension-agnostic convergence rate of $\mathcal{O}(1/T + 1/b)$, where $T$ is the number of iterations and $b$ is the batch-size, surpassing the dimension-dependent slowdown of purely ZO methods and significantly improving over SGD's $\mathcal{O}(1/\sqrt{T})$ rate. Additionally, we propose a multi-point variant that mitigates the $O(1/b)$ error by adjusting the number of estimation points to balance convergence and cost. Importantly, VAMO achieves these gains with smaller dynamic memory requirements than many FO baselines, making it particularly attractive for edge deployment. Experiments including traditional neural network training and LLM finetuning confirm that VAMO not only outperforms established FO and ZO methods, but also does so with a light memory footprint.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes VAMO - a variance reduction optimization method that combines both zero-order gradient estimates and gradient to produce the update direction. Concretely, the algorithm uses the full batch and batch gradient estimates at the checkpoint (obtained from the standard zero-order method) and combines with the batch gradient at the current iterate ($g = \nabla f(x, B) - \alpha (\hat{\nabla} f (x_{\text{cpt}}, B) - \hat{\nabla}f(x_{\text{cpt}}))$). This gives a trade off between convergence and computational complexity compared with ZO-SVRG. In terms of convergence, the rate of VAMO is $1/T + 1/b$ where $b$ is the batch size. This has the extra  $1/b$ term is inherited from ZO-SVRG due to the bias in the estimator, but the first term is improved by a factor $d$. The computational complexity on the other hand is increased from $nS+bT$ to $nS+dbT$ (due to the gradient computation). An extension to this method is to use multi-point query, which can trade off between the improvement in the additional error term and the computational complexity. Experiments show that VAMO has faster convergence than other ZO methods and better memory footprint than FO methods.

### Strengths
- Combining FO and ZO in SVRG is an interesting idea and the FO order component improves the performance of the algorithm compared with other ZO methods.
- The presentation of the paper is easy to follow with detailed comparison with prior works.
- Proofs seem all good.

### Weaknesses
I have several concerns below.
- I'm not sure I understand the memory analysis in Section 4.3, B.1 and Table 3. 
  - For VAMO, why is the memory for the optimizer states only $|x|$? The algorithm needs to store $\hat{\nabla}(x_{\text{cpt}})$ but also $x_{\text{cpt}}$ to compute the estimate $\hat{\nabla}(x_{\text{cpt}}, B)$ (see also line 3-4 in Alg. 1). I don't understand how to reduce the memory to only $|x|$? If my understanding is true, I don't see an improvement in the memory for the proposed method.
  - For Adagrad, why is the optimizer states $2|x|$? Doesn't the algorithm only store the accumulation of the gradients per coordinate so the memory needed is only $|x|$? Same for Adam, shouldn't it be $2|x|$?
- The experiment appears quite weak. First of all, the paper only reports the training loss. While optimization algorithms only optimize the training loss, we also care about the test loss and accuracy. Second, one main motivation the paper mentioned is to overcome limitations of FO and ZO methods for efficient training of large models. However, the set of experiments is quite limited. MNIST is an outdated dataset as an optimization benchmark. Usually, for optimization papers, starting with CIFAR-10/100 will show clearer impacts. I also suggest that the authors use a similar experiment setup in the paper MeZO and add more experiments.

### Questions
- See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes VAMO, a new stochastic optimization algorithm for large-scale non-convex optimization.
The algorithm aims to bridge the gap between first-order (FO) methods, which have fast
convergence but high computational and memory costs, and zeroth-order (ZO) methods, which are
memory-light but suffer from dimension-dependent and slow convergence.
VAMO is a two-loop "SVRG-style" algorithm. Its central idea is to replace the computationally
expensive full-batch FO gradient (∇f(ˆx)) used in FO-SVRG with a full-batch ZO gradient estimate
(ˆ∇f(ˆx)). The inner-loop update is a novel hybrid estimator: vsk
= ∇fIk (xsk) − α(ˆ∇fIk (ˆx) − ˆ∇f(ˆx)).
This construction cleverly uses a ZO-based variance correction term that has zero expectation, making
vsk an unbiased estimator of the true gradient ∇f(xsk).
The authors provide a theoretical analysis showing the two-point (q = 1) version of VAMO achieves a
convergence rate of O(1/T +1/b), which is dimension-independent, a significant improvement over typical
ZO methods. They also propose a multi-point (q > 1) variant with a rate of O(1/T +(1−q/d)2/b),
which can converge to a stationary point if q = d. The primary claims are that VAMO achieves FOSVRG-
like convergence speed with significantly lower computational and, critically, dynamic memory
costs. Experiments on neural networks and LLM finetuning are presented to support these claims.

### Strengths
Novel Hybrid Estimator: The proposed gradient estimator (Eq. 6) is novel. The insight to
use a ZO-based correction term α(ˆ∇fIk (ˆx) − ˆ∇f(ˆx)), which has zero expectation, is technically
sound. This ensures the full estimator vsk is an unbiased estimator of the true current gradient
∇f(xsk), which is an elegant property for the analysis.

2. Dimension-Independent ZO-Hybrid Rate: The theoretical analysis successfully breaks the
dimension-dependency curse of pure ZO methods. Achieving a rate of O(1/T + 1/b) (Corollary
1) that is independent of d is a strong theoretical contribution for a method that incorporates
ZO components.

3. Strong Empirical Performance vs. FO-SGD: The experiments, particularly in Figure 2
(LLM finetuning), show that VAMO (with q = 1) converges significantly faster in terms of both
steps and wall-clock time than FO-SGD. This empirically validates the theoretical advantage of
the O(1/T + ...) rate over FO-SGD’s O(1/√T) rate for reaching a certain precision.

### Weaknesses
Despite its theoretical novelty, the paper’s core claims about its practical advantages are based on a
series of critical, and in some cases contradictory, flaws in the analysis of its computational and memory
costs.

1. Fundamentally Misleading Convergence Claims: The paper repeatedly claims an O(1/T )
rate, equating it with FO-SVRG (e.g., Abstract: "significantly improving over SGD’s... rate";
Conclusion: "achieving convergence performance similar to FO-SVRG"). This is a misrepresentation.
The actual derived rate is O(1/T + 1/b). This is not convergence to a stationary point.
It is linear convergence to a noise ball of size O(1/b). This non-vanishing error term, which
dominates at large T, means VAMO (q = 1) cannot achieve high-precision solutions and may, in
fact, converge to a worse solution than FO-SGD (which does converge to ϵ = 0, albeit slower).
This is a crucial distinction that is glossed over.

2. Contradictory and Factually Incorrect Memory Analysis: This is the paper’s most severe
flaw. The paper is motivated by reducing the high dynamic memory of FO methods (e.g., storing
activations for backpropagation). The inner-loop update (Algorithm 1, Step 7) explicitly requires
a mini-batch FO gradient, ∇fIk (xsk). Computing this FO gradient requires backpropagation
and storing intermediate activations, leading to a dynamic memory cost of O(b ·P|al|). This is
the exact same dynamic memory cost as FO-SGD. The paper’s text (e.g., Section 4.3, Lines 325-
328: "thus do not need to store intermediate results", "dynamic memory only reaches maxl |xl|")
is factually incorrect and fundamentally misunderstands the cost of its own algorithm. The
data in the paper confirm this. Table 3 correctly lists VAMO’s dynamic memory as
Pl max{b ·|al|, |xl|}, identical to FO-SGD. Table 2 empirically shows VAMO’s memory (e.g., 21.46 GB) is
almost identical to FO-SGD’s (20.33 GB). The small increase is expected, as VAMO = FO-SGD
(dynamic) + |x| (snapshot state). The central claim that VAMO is a low-memory algorithm
(relative to FO-SGD) is false.

3. No Asymptotic Computational Advantage for O(1/T ) Convergence: To achieve the true
O(1/T ) convergence (i.e., remove the O(1/b) error), one must use the multi-point variant and set
q = d (Theorem 2).The computational complexity of VAMO (q = d) is O(qnS + (bd + bq)T) =
O(dnS + (bd + bd)T) = O(dnS + bdT ). Therefore, to achieve the same convergence rate as FOSVRG,
VAMO requires the identical asymptotic computational complexity. The paper’s claim
of computational efficiency is only valid when comparing the non-converging q = 1 version to the
converging FO-SVRG, which is an apples-to-oranges comparison.

4. Incorrect Computational Complexity in Table 1: The complexity for VAMO (q = 1) is
listed as O(nS +bdT ). This is incorrect. The inner loop (Step 7) computes two gradients: ∇fIk
(cost O(bd)) and ˆ∇fIk (cost O(bq), or O(b) for q = 1). The correct complexity is O(qnS +(bd+
bq)T). For q = 1, this is O(nS + (bd + b)T), which is O(nS + bdT ) only if d ≫ 1. This O(bqT)
term is missing from the table.

5. Empirical Evidence of Non-Convergence: In Figure 1b, VAMO(q = 1) clearly converges to
a final training loss that is visibly *higher* than that of FO-SGD. This plot empirically confirms
the O(1/b) noise ball limitation. This is a poor result, as it fails to match the solution quality of
the baseline FO-SGD, yet this is not discussed.

### Questions
1. The text in Section 4.3 (Lines 325-328) claims that VAMO does not need to store intermediate
activations. However, Step 7 of Algorithm 1 computes an FO gradient ∇fIk (xsk), which necessitates storing activations. Your own Table 3 and Table 2 confirm that VAMO has the same dynamic memory as FO-SGD. Can you please clarify this fundamental contradiction? Is the premise of the paper’s memory-saving benefits not incorrect?

2. The O(1/T + 1/b) rate converges to a noise ball, not a stationary point. Why is this
misleadingly presented as "similar to FO-SVRG" (which converges to 0) and an improvement
over FO-SGD (which also converges to 0)? Figure 1b seems to confirm that VAMO(q = 1) converges
to a worse solution.

3. The inner loop (Step 7) computes both an FO-grad (cost O(bd)) and a ZO-grad (cost O(bq)).
Why is the O(bqT) term missing from the complexity analysis in Table 1?

4. Given that VAMO(q = d) has the same rate and computation as FO-SVRG, why was this
Comparison (which is the only fair comparison of two O(1/T ) methods) is not included in the
experiments?

5. How sensitive are convergence and stability to the choice of α, μ, and q? Could you provide
empirical ablations or adaptive scheduling strategies, validating these settings across different
model dimensions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The research presents VAMO as a hybrid optimizer that works with big models under memory restrictions. The optimizer uses First-Order (FO) speed together with Zeroth-Order (ZO) efficiency by substituting the expensive SVRG variance-reduction algorithm step with a low-cost ZO gradient estimate. The algorithm provides fast dimension-independent convergence speed that outperforms SGD yet requires memory levels similar to SGD and less than Adam. The empirical results show VAMO runs at a slower pace than Adam but provides an attractive trade-off between performance and memory usage for restricted resource scenarios.

### Strengths
1. VAMO achieves a fast, linear convergence rate of $\mathcal{O}(1/T)$, which is an improvement over the $\mathcal{O}(1/\sqrt{T})$ rate of standard SGD.
2. A key advantage is that its convergence rate is independent of the model's parameter dimension $d$. This allows it to overcome the "curse of dimensionality" that makes purely Zeroth-Order (ZO) methods impractical for large models.
3. This paper provides a strong theoretical guarantee. VAMO's gradient estimator is designed to be unbiased.

### Weaknesses
1. The experimental results demonstrate that VAMO achieves better performance than SGD, but it fails to match the convergence speed and training efficiency of the Adam optimizer during large-scale fine-tuning.
2. The theoretical analysis shows that VAMO's convergence rate, while faster than SGD's, includes an additional error term of $\mathcal{O}(1/b)$ that is not present in the purely First-Order FO-SVRG algorithm.
3. The paper introduces a multi-point variant to minimize the additional error term. The improvement requires additional computational resources, which increase with the number of query points $q$, thus users need to decide between faster convergence and higher processing expenses.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes VAMO, a hybrid variance-reduced optimizer that blends a mini-batch first-order (FO) gradient with a zeroth-order (ZO) SVRG-style correction computed at snapshot points. The key idea is to replace the expensive full-batch FO snapshot gradient with a full-batch ZO estimate and to weight the correction with a mixing coefficient \alpha. The authors prove a  convergence rate of O(1/T+1/b) independent of dimension d. Experiments span a synthetic task, MNIST MLP, and fine-tuning GPT-2 / GPT-2-Medium / RoBERTa-Large on SST-2 and MNLI, with GPU memory comparisons versus FO-SGD/Adagrad/Adam.

### Strengths
1. I like the discussion of memory: the paper argues VAMO’s snapshot uses forward-only ZO passes, so peak dynamic memory resembles ZO-SGD rather than FO-SVRG, and optimizer state is lighter than Adam/Adagrad. Included tables and a clear decomposition (weights / states / dynamics) are helpful.
2. The main bound removes the typical d dependence of ZO methods and improves over FO-SGD’s O(1/\sqrt{T}) in theory.

### Weaknesses
1.ZO methods inherently trade off performance, memory, and wall-clock. The paper treats ZO snapshots as “cheap,” but a full-batch snapshot still requires multiple forward passes per direction over the entire dataset (or many mini-batches). In practice, if memory is the bottleneck then ZO can help. However, runtime can balloon unless the number of directions q is tiny—yet shrinking q raises estimator variance and hurts convergence. Therefore, without an accounting of function evaluations (FEs) and throughput (e.g., tokens/sec), it’s unclear when VAMO is actually preferable to tuned FO baselines (Adam/Lion/Adafactor) or to ZO-SVRG variants with different q.

2. ZO can also be performed when the random direction is Gaussian, i.e., let u(x;\theta) = E_{\delta\sim N(0,\sigma^2 I) f(x+\delta;\theta), then \nabla_x u(x;\theta) = E_{\delta\sim N(0,\sigma^2 I) [\delta/\sigma^2  f(x+\delta;\theta). In high dimension, Gaussian vs. coordinate-wise directions can yield different estimator variance and smoothing bias. Yet this is not compared in the paper.

3. Fine-tuning results present training loss (showing convergence) and memory costs. However, I would expect also the task metrics (accuracy/F1 for SST-2/MNLI) and stability stats (divergence/NaNs) to show the effectiveness of VAMO.

4. SVRG-style methods hinge on the outer-loop frequency S and inner length m. The paper would benefit from an empirical study of how often to recompute ZO snapshots (and with what q) under a fixed compute budget; otherwise, it’s hard to see when VAMO is preferable to carefully tuned FO-SGD/Adam or existing ZO-SVRG variants.

### Questions
see weakenesses.

### Soundness
3

### Presentation
3

### Contribution
2
