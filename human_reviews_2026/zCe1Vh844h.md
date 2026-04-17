# Generalized Fisher-Weighted SVD: Scalable Kronecker-Factored Fisher Approximation for Compressing Large Language Models.

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
The Fisher information is a fundamental concept for characterizing the sensitivity of parameters in neural networks. However, leveraging the full observed Fisher information is too expensive for large models, so most methods rely on simple diagonal approximations. While efficient, this approach ignores parameter correlations, often resulting in reduced performance on downstream tasks. In this work, we mitigate these limitations  and propose Generalized Fisher-Weighted SVD (GFWSVD) — a fully deterministic  post-training LLM compression technique that accounts for both diagonal and off-diagonal elements of the Fisher information matrix, providing a more accurate reflection of parameter importance. To make the method tractable, we introduce a scalable adaptation of the Kronecker-factored approximation algorithm for the observed Fisher information. We demonstrate the effectiveness of our method on LLM compression, showing improvements over existing compression baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Generalized Fisher-Weighted SVD (GFWSVD), a post-training compression method for LLMs. It addresses the limitations of prior work like FWSVD, which uses only diagonal approximations of the Fisher Information Matrix (FIM) and ignores parameter correlations. GFWSVD leverages a Kronecker-factored approximation ($A \otimes B$) of the FIM to capture both row and column dependencies, integrating these factors into a generalized SVD framework. The authors provide a scalable algorithm to compute these factors efficiently and demonstrate empirically that GFWSVD outperforms existing gradient- and activation-based compression baselines (FWSVD, ASVD, SVD-LLM) on BERT and LLaMA-2 models.

### Strengths
1. The paper is well-written and clearly structured.
2. The proposed method GFWSVD is novel and  backed with theoretical grounding and practical evaluations, effectively generalizing FWSVD.

### Weaknesses
1. Evaluations on modern LLMs are lacking. Modern LLMs like Llama 3.1/Qwen 3 are known to be overtrained and harder to compress. It is important to provide evaluations on such models for the proposed method to become practical.
2. The paper can benefit from a more detailed literature review. More recent SVD-based LLM compression approaches can achieve much higher compression ratios on more recent LLMs; for example, BitStack[1] matches quantization results and far surpasses the baselines in the paper. I understand this paper is more analytical, but a section with a more detailed discussion of the practicality of GFWSVD and more recent approaches is needed.

[1] BitStack: Any-Size Compression of Large Language Models in Variable Memory Environments

### Questions
See Weaknesses. I’ll increase the score from 4 to 6/8 if the authors can provide the requested experiments and discussions.

### Soundness
3

### Presentation
4

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
This work proposes Generalized Fisher-Weighted SVD (GFWSVD), a post-training LLM compression technique. The method uses the Fisher information to approximate the Hessian, which is then further approximated using a Kronecker product.
Since leveraging the full Fisher information is computationally expensive for large-scale models, existing approaches typically rely on diagonal approximations. This work tightens that approximation by proposing a more general method. The authors develop theory to motivate their approach, assuming the model's dense weight matrices are drawn from a matrix-variate normal distribution. Empirically, they demonstrate that their method achieves better perplexity and accuracy compared to (some) existing methods.

### Strengths
The strength of this paper is that it generally seems to outperform the baselines in terms of perplexity and accuracy. This can be observed in Tables 3 and 4 of the main paper, along with Table 8 in the appendix. It also tightens the diagonal approximation of the Fisher information used in other works, albeit under the assumption that the weight matrices are drawn from a matrix-variate normal distribution.

### Weaknesses
I think one of the major weaknesses of this paper is the clarity in presentation, in that it is often missing key details to either experiments or is not clear to understand. For example:
- In Theorem 1, what is the "task loss function"? I don't think the "task" is clearly defined. I assume that the authors are referring to the problem in Equation (5), but then this should be referenced.  In fact, is this first condition (or perhaps the second condition) redundant? Based on lines 154-156 of the manuscript, it seems to me that 1 and 3 already imply 2, or 2 and 3 already imply 1. I think the authors can frame this a bit more clearly.
- Table 1 is difficult to follow. Is this the time it takes to perform the decomposition of all of the matrices in the corresponding model (e.g., LLaMa-2-7B) or is it just a few matrices of the model? If it is a few, then which matrices are chosen? What are the dimensions of these matrices? Or is this the time taken to approximate the actual Fisher information matrix? 
- What is the number of batches $D$ used to approximate the Fisher information in your experiments?
- Separating Tables 2 and 3 is a bit confusing to me; it seems to me that you can include Table 2 as a row in Table 3. 

I'm also concerned at the compression rates used in the experiments; as far as I am aware, a 20% compression rate as done in Table 4 is pretty low and existing works compress at a far more aggressive rate and have good performance -- see [1] as an example.

---

[1] "BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference". Changwoo Lee, Soo Min Kwon, Qing Qu, Hun-Seok Kim. NeurIPS 2024.

### Questions
I have listed a few questions in the weaknesses section. Here are just a few more:

- Why is assuming that the dense weight matrices follow a matrix-variate normal distribution a fair assumption? As far as I can tell, this generalized decomposition only holds under this assumption. Can you reference existing works that also make this assumption, or is there any evidence you can provide? 
- In the limitations section, it says that often a constant $\alpha$ is used to ensure positive semidefiniteness. What is the constant used for experiments? Does FWSVD suffer from this issue as well?

I apologize in advance if these questions are already answered in the manuscript and I missed them. If so, I would appreciate it if the authors could kindly point me to the relevant sections.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes GFWSVD: a post-training LLM compression method that uses a Kronecker-factored approximation of the full Fisher Information Matrix (FIM) to perform a generalized weighted SVD on linear layers. The authors prove that FWSVD is a special case of their framework, derive a closed-form 2-factor decomposition (Theorem 1), and provide a rank-1 SVD algorithm to estimate Kronecker factors efficiently. Experiments on BERT/GLUE and LLaMA-2-7B (MMLU, WikiText-2/PTB) show consistent improvements over SVD, ASVD and FWSVD at comparable compression rates.

### Strengths
1. Clear generalization & theory. Establishes MVN–Fisher–GSVD connection and proves FWSVD as a special case; gives a closed-form solution for low-rank factors (Eq. 7, 10)
2. Scalable factor estimation. Rank-1 SVD on a permuted empirical Fisher enables practical Kronecker factorization with matrix–vector products (Alg. 1; complexity discussion + empirical runtimes).

### Weaknesses
1. The paper uses opposite meanings of “Compression Rate” in different sections. Table 2 maps ranks to “compression rate” as removed % (e.g., r=600 ≈ 1%, r=50 ≈ 36%), whereas Table 8 and Fig. 3 (LLM/MMLU) treat it as retained % (e.g., “Compression Rate 99% (r=600)”, x-axis 80–95%, and Full model=100%). This contradicts the BERT side and makes cross-figure/table reading ambiguous. 
2. The LLM side evaluates only LLaMA-2-7B-chat, lacking newer families like Llama-3 and Qwen-2.5/Qwen-3 (ideally multiple sizes) to support claims of architecture- and scale-robustness. The baselines should also include recent SVD-based methods—SVD-LLM v2, Basis Sharing, and Dobi-SVD—under a unified setup, and report a 50%–90% compression-ratio sweep (e.g., 50/60/70/80/90%) to enable apples-to-apples quality-vs-compression comparisons. 
**SVD-LLM v2** https://arxiv.org/abs/2503.12340 
**Basis Sharing** https://openreview.net/pdf?id=gp32jvUquq
 **Dobi-SVD**  https://openreview.net/pdf?id=kws76i5XB8
3. The paper targets better post-training compression via Kronecker-factored FIM and generalized weighted SVD, but it reports quality only and omits end-to-end time-to-truncated-model versus SVD-LLM.
4. **References & in-text citations are not ICLR-compliant**

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

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
This paper introduced the Generalized Fisher Weighted SVD (GFWSVD), a post-training, dual pipeline
compression technique that leverages the Kronecker-factored approximation for the full empirical Fisher
Information Matrix (FIM) to drive optimal compression for dense weight matrices of diverse large
language models. Specifically, the method introduced, at first, a scalable rank-1 Kronecker decomposition
algorithm that reduces FIM factorization cost from $\mathcal O((mn)^3)$ to $\mathcal O(m^3 + n^3)$,
then it proposed an efficient Kronecker decomposition based Singular Value Decomposition for dense
layer compressions. The paper theoretically shows (in Theorem 1) that under MVN + Kronecker
assumptions, this method yields the optimal weighted low-rank approximation in expectation. Empirically
demonstrates improvements over vanilla SVD, ASVD, SVD-LLM and diagonal FWSVD on BERT
(GLUE) and LLaMA-2 (MMLU, perplexity) across a range of compression rates.
Strengths

### Strengths
1. Introduced a generalized, architecture agnostic compression method that applies to any linear (or
Kronecker-structured) layer in an LLM. By factoring the full empirical Fisher Information into two small
sensitivity matrices and plugging them into a weighted SVD, GFWSVD ensures efficient compression
with respect to parameter interactions, while reducing the factorization cost from $\mathcal O((mn)^3)$
to $\mathcal O(m^3 + n^3)$.
2. Theoretical optimality (Theorem 1) follows from the MVN/Laplace approximation around an MLE
solution, where assumptions that mirror those commonly used in second-order optimization (e.g. K-FAC)
and Bayesian posteriors. This alignment with well-understood curvature approximations ensured
GFWSVD both sound and readily applicable in real-world settings.
3. GFWSVD outperforms prior works on recognized baselines and benchmarks across a range of
compression rates.

### Weaknesses
1. While the paper details the theoretical and empirical cost of the offline Kronecker decomposition
(compression) step, there is insufficient experiments for performance on computation side. Since one of
the main appeals of low-rank methods is reduced FLOPs and wall-clock latency at runtime, the absence
of end-to-end benchmarks (e.g. on LLaMA-7B-chat-bf across different ranks) makes it hard to judge realworld
benefit compared to ASVD, SVD-LLM, or Dobi-SVD. Furthermore, the time complexity analysis
was primarily focused on offline model decomposition step, which there is no theoretical analysis for
inference side analysis. Therefore, a thorough theoretical and quantitative analysis for computational
performance is expected for an acceptance (Major concern, will consider raising score if properly
addressed).
2. The GFWSVD exhibits a relatively low compression ceiling, with only about a 40 % reduction at rank
1, whereas pure SVD methods like SVD-LLM can achieve over 60 % reduction under comparable
accuracy constraints when accuracy drop is tolerable. Without strategies to extend beyond this “shallow”
bound (e.g. multi-term Kronecker sums or hybrid pruning), GFWSVD’s standalone compression
advantage appears limited. (If not properly solved, the maximum score will be a weak accept)
3. The evaluation covers ASVD, SVD-LLM, and the original FWSVD, but omits more recent or stronger
methods such as SVD-LLM v2 and Dobi-SVD. Including those would clarify where GFWSVD stands
against the current state of the art and strengthen its performance.

### Questions
1. The paper proposed only a single Kron-term (rank-1) FIM factorization and varies only the model’s
rank $r$ when compressing weights. It remains unclear how choosing different ranks for the Fisher
decomposition itself (or using multiple Kronecker terms) impacts end-task performance—and whether the
Fisher-based weighting truly outperforms unweighted or diagonal-weighted SVD across ranks. An
ablation study over both FIM rank and model rank is expected to clarify how sensitive accuracy is to
those choices and quantify the standalone benefit of the Fisher weighting.
2. GFWSVD also relies heavily on Cholesky factorizations deriving FIM blocks ($A = L_A L_A^\top$,
$B = L_B L_B^\top$). However, it is not uncommon that, in practice, finite-batch Fisher estimates can be
near-singular or even indefinite. The paper does not report on how often Cholesky fails, what damping is
applied, or how numerical issues affect the compressed model’s performance. Therefore, an empirical
analysis of these edge cases and corresponding regularization strategies to tackle the edges are expected
for robustness consideration.
Few other tips:
1. Typo in Appendix D, Table 4: The ASVD entry “–0.03” at 64 % compression appears to report the
change in STS-B score (compressed minus original) rather than the absolute correlation. All other
methods list absolute STS-B values (about 0.7–0.9), so ASVD’s “–0.03” seems to be a transcription error.
2. In Alg 1 step 1: “IF $\leftarrow$ $\frac{|D|}\sum g_i g_i^T$”, where $g_i g_i^T$ is with quadratic
complexity and was never materialized in practice. The author should clarify that in practice full matrix
was never formed but accumulated only $G_iG_i^T$ and $G_i^T G_i$.

### Soundness
3

### Presentation
3

### Contribution
3
