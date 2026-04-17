# FLoRA-NA: Nearly Accurate Aggregation for Federated Low-Rank Adaptation

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
With the rapid emergence of foundation models and the increasing need for fine-tuning across distributed environments, Federated Low-Rank Adaptation (FedLoRA) has recently gained significant attention. Despite its potential, current FedLoRA methods face notable challenges due to inexact updates. Existing approaches have attempted to mitigate this issue, but they often introduce a \emph{local-global generalization gap} and incur \emph{substantial communication overhead}, limiting their scalability and effectiveness.  
To address these limitations, we propose \textbf{F}ederated \textbf{Lo}w-\textbf{R}ank \textbf{A}ggregation with \textbf{N}early \textbf{A}ccurate Estimation (FLoRA-NA). FLoRA-NA leverages the local LoRA matrices on the server to estimate the aggregated matrices $\hat{A}$ and $\hat{B}$, which are then distributed to clients for local updates. This surrogated aggregated matrices minimizes the divergence between ideal $\bar{W} = \sum^{U}_{u=1}B_u A_u$ and practical updates $\hat{W} = \hat{B}\hat{A}$ without adding communication cost beyond vanilla FedLoRA. By doing so, FLoRA-NA achieves communication efficiency and bridges the gap between local personalization and global generalization, addressing a key limitation of prior personalized FedLoRA approaches.
We conduct extensive evaluations across diverse tasks, including natural language understanding, mathematical reasoning, and code-solving ability using various foundation models. Experimental results consistently demonstrate that FLoRA-NA achieves state-of-the-art performance while maintaining low communication overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes Federated Low-Rank Aggregation with Nearly Accurate Estimation (FLoRA-NA). FLoRA-NA leverages the local LoRA matrices on the server to estimate the aggregated matrices \\(\hat{A}\\) and \\(\hat{B}\\), which are then distributed to clients for local updates, to address the aggregation errors in FL with LoRA.

### Strengths
- The proposed FLoRA-NA effectively addresses the aggregation errors of FL with LoRA by approximating the "ideal" aggregation.
- The paper is well-written and easy to understand.
- The extensive experiments validate the effectiveness of the proposed method.

### Weaknesses
- How to solve the optimization problem in Equation (6) is not clearly explained.
- Lines 337-340, "SVD-based methods may introduce numerical inaccuracies due to floating-point precision errors, potentially degrading the overall performance of FedLoRA." It's better to add some performance comparisons in Table 1.

### Questions
- Lines 196-195, "These vectors can be interpreted as transformation vectors that determine how each client’s local LoRA gradients are linearly combined during aggregation." Why can it be considered as a linear combination?
- Why is there such a large discrepancy between the local and global performance of FFA-LoRA? Aren't the local and global models the same? Is it due to the differences in the test data for local and global models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces FLoRA-NA, a “nearly accurate” aggregation method for federated LoRA fine-tuning. The authors argue that existing FedLoRA methods suffer from aggregation errors when averaging low-rank matrices across clients. To mitigate this, FLoRA-NA introduces auxiliary weighting vectors ( P, Q ) to linearly combine local LoRA matrices, claiming to approximate the true aggregated matrix product without additional communication cost. The method is backed by convergence analyses and evaluated on standard NLP, reasoning, and code datasets.

### Strengths
- Empirical thoroughness: The experiments are broad and cover many baselines, including federated LoRA, DoRA, and HiRA variants, on both language and reasoning tasks.


- Nice writing and clear presentation: The narrative is coherent and polished, and the paper is written well.

### Weaknesses
`W1: The core technical idea is not convincing`

The claimed “nearly accurate aggregation” is essentially a weighted averaging of LoRA matrices with learned coefficients ( P, Q ). This approach is neither theoretically optimal nor computationally cheaper compared to well-established matrix approximation methods.
 
If the goal is to minimize $| \bar{B}\bar{A} - \tfrac{1}{U} \sum_u B_u A_u |,$, then taking truncated SVD or other low-rank approximation methods provably yield the best low-rank reconstruction in Frobenius norm. The proposed optimization over two scalar vectors $ P, Q \in \mathbb{R}^U $ is an ad-hoc and suboptimal heuristic, without any proof that it approximates the optimal low-rank structure. 


The paper never explains why this weighting approach should outperform SVD-based aggregation, nor provides any theoretical justification beyond hand-waving about computational efficiency. The argument that “SVD introduces numerical inaccuracies” is very weak, in practice, taking truncated SVD is stable and extremely fast.

`W2: Execution time comparison is misleading`

In Table 1, SVD and Gram-Schmidt are reported as being 10–15 times slower than FLoRA-NA. However, the comparison is unfair: taking truncated SVD matrices would in fact run much faster than full SVD (10s of times faster or so), and this is confirmed in other works like LoRA-GA (https://arxiv.org/abs/2407.05000), without any loss in performance.

Thus, the claim of computational efficiency is unsubstantiated and incorrect, in my opinion.

`W3: Overclaiming novelty`

Most of the claimed advantages (no extra communication, better generalization) stem simply from tuning a linear combination of existing LoRA matrices. This is closer to a reweighted averaging trick than a novel algorithmic contribution. The “nearly accurate” branding oversells what is, in essence, a parameter-weighted variant of FedAvg.

The idea of using truncated SVD as the best inexact approximation has already been established in FedEx-LoRA, which makes the proposed approach in this paper appear redundant.

`W4: Conceptual inconsistency in results`

The paper claims FLoRA-NA achieves “nearly exact” aggregation, yet FedEx-LoRA, which performs exact aggregation, yields lower accuracy (Table 2). This is not explained logically, the authors state it is because “ the residual error is added to the frozen pretrained model, which ensures the accuracy of forward update, but cannot fully participate in gradient updates of A and B matrices.,” but this line is vague and unconvincing.

If the proposed method is truly more “accurate,” it should not outperform an exact method unless the claim itself is overstated.
Could you please clarify this?

`W5: Missing related work`

The paper fails to discuss recent federated low-rank methods such as Ravan (https://arxiv.org/abs/2506.05568) and Fed-SB (https://arxiv.org/abs/2502.15436), which provide a clear communication–accuracy Pareto frontier compared to this and other works - at a quick glance. This omission weakens the positioning of FLoRA-NA as a “state-of-the-art” solution. The authors should include comparisons with these works, or at the very least, discuss them.

---

While the empirical coverage in the work is impressive, the core contribution is mathematically weak, and the claimed superiority over SVD-based methods is not supported by any evidence (theoretical or empirical). The paper’s theoretical and experimental arguments do not convincingly establish that FLoRA-NA offers any principled or practical advantage.

### Questions
Please refer to weaknesses

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper revisits FedLoRA and argues that separately averaging the low-rank matrices 𝐴 and 𝐵 on the server introduces an aggregation bias. The authors propose FLoRA-NA, which learns aggregation weights to approximate the ideal bilinear average and derive a convergence bound including this bias term. Experiments on NLP and reasoning benchmarks show moderate improvements over standard FedLoRA. The idea is intuitive and empirically effective, but conceptually incremental and lacks strong causal or theoretical justification.

### Strengths
The paper clearly identifies a practical limitation of FedLoRA aggregation and formalizes it as an explicit bias term. The proposed solution is simple, communication-efficient, and easy to implement.
Experiments are well-organized and consistently show moderate but stable performance gains across multiple benchmarks.

### Weaknesses
The paper’s main motivation that the standard FedLoRA aggregation rule BA=1/U∑BuAu introduces a harmful aggregation bias is intuitively appealing, but the presented evidence and methodology raise several concerns:

1. Lack of causal proof that aggregation bias harms performance.
The paper defines an error term 𝜌 = ∥𝐵ˉ𝐴ˉ−1/𝑈∑𝐵𝑢𝐴𝑢∥ and incorporates it into a convergence upper bound. However, this analysis merely indicates that model convergence may depend on ρ; it does not prove that a non-zero ρ necessarily degrades performance. No lower-bound analysis, sensitivity test, or monotonicity relation is provided, making the claim that “the bias is inherently harmful” insufficiently supported.

2. Possible confounding with non-IID data heterogeneity.
The reported improvements could stem from alleviating data heterogeneity rather than from the aggregation rule itself. Since non-IID strength is not controlled and there is no oracle baseline (directly computing 1/𝑈∑𝐵𝑢𝐴𝑢), the experiments cannot disentangle aggregation effects from data heterogeneity. To establish a causal link, the authors should fix the data partition and vary only the aggregation operator while measuring the correlation between ρ and performance.

3. Overstated claim that the standard FedLoRA rule is “erroneous.”
The paper repeatedly describes the classical FedLoRA averaging as “erroneous” or “fundamentally flawed.” Yet the analysis only shows algebraic non-equivalence, not that this difference is detrimental under general conditions. Without stronger theoretical or experimental isolation, such a statement appears exaggerated.

4. Limited methodological novelty.
The idea of learning aggregation weights on the server side is far from new. Similar learnable aggregation or re-weighted averaging strategies have been extensively explored in earlier federated optimization, distillation, and low-rank model aggregation works (2018–2021). The proposed approach learning separate coefficients 
𝑃,𝑄 for the A and B matrices is essentially a linear re-weighted averaging scheme adapted to the LoRA setting. Hence, the contribution is largely engineering-oriented reuse of an existing idea rather than a conceptual innovation.

5. Suggested clarifications.
Explicitly quantify how ρ affects optimization error or final convergence (e.g., linear vs. sublinear dependence).
Add an oracle baseline that uses the true average 1/𝑈∑𝐵𝑢𝐴𝑢 to decouple non-IID effects. Discuss how this approach fundamentally differs from prior learnable aggregation methods rather than merely showing empirical improvements.

6. Alternative perspective: directly addressing non-IID heterogeneity may be more fundamental.
The paper attributes performance degradation to inaccurate matrix aggregation, but an equally plausible explanation is that client-specific differences in 𝐴𝑢,𝐵𝑢 caused by non-IID data are the real source of the problem. Methods such as FRLoRA (ICLR 2025), which reduce client drift through residual accumulation and subspace alignment, inherently make client updates more consistent thereby indirectly shrinking ρ without modifying the aggregation rule itself. In this sense, tackling heterogeneity at its root might offer a more principled and effective solution than merely adjusting the aggregation formula.

### Questions
The motivation and theoretical justification remain unconvincing. The claim that the standard FedLoRA aggregation rule is “fundamentally wrong” is overstated and lacks causal evidence. The proposed learnable-weight scheme is not methodologically novel and revisits well-studied ideas under a new name. Moreover, non-IID heterogeneity is not controlled and could fully explain the observed performance gap. In contrast, approaches like FRLoRA that directly mitigate client drift may address the root cause more effectively. Overall, the work appears to be an engineering refinement rather than a conceptual breakthrough.

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
The paper presents FLoRA-NA, a federated low-rank adaptation method designed to improve update accuracy without increasing communication cost. It estimates aggregated LoRA matrices on the server to reduce the divergence between ideal and practical updates, thereby narrowing the local–global generalization gap. Experiments across language understanding, reasoning, and code tasks show that FLoRA-NA achieves state-of-the-art global performance with strong communication efficiency.

### Strengths
1. The paper addresses a well-defined problem in FedLoRA, i.e., the inexact update issue and the resulting local–global generalization gap.
2. Extensive experiments across diverse tasks such as language understanding, reasoning, and code generation demonstrate consistent improvements over strong baselines.
3. The paper provides a clear theoretical analysis.

### Weaknesses
1. The introduced matrices P and Q may increase communication costs, but a detailed analysis is missing.
2. The experiments focus mainly on language-related tasks; evaluations on other modalities, e.g., vision-language, would better demonstrate the method’s generality.
3. The paper does not discuss the limitations of the proposed method, which would be valuable for understanding its potential weaknesses and applicability boundaries.

### Questions
1. The manuscript ignores important implementation details concerning the optimization of the introduced matrices P and Q. In particular, Algorithm 1 does not show how P and Q are updated (e.g., update rules, gradients, local vs. server updates, or whether they are trained jointly with other parameters). Please clarify these points and update Algorithm 1 (or provide a supplement) to include explicit optimization steps.

2. Are the matrices P and Q layer-wise, or are they shared across multiple layers? Please clarify their scope and how they are applied in the model.

3. If P and Q are layer-wise, the additional communication cost introduced by these matrices should not be ignored. Please discuss how this overhead affects overall efficiency and whether it is accounted for in the experiments.

### Soundness
2

### Presentation
3

### Contribution
3
