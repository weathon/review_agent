# Lookahead Unmasking Elicits Reliable Decoding in Diffusion Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Masked Diffusion Models (MDMs) as language models generate by iteratively
unmasking tokens, yet their performance crucially depends on the inference-
time order of unmasking. Prevailing heuristics, such as confidence-based sam-
pling, are myopic: they optimize locally, fail to leverage extra test-time compute,
and let early decoding mistakes cascade. We propose Lookahead Unmasking
(LookUM), which addresses these concerns by reformulating sampling as path
selection over all possible unmasking orders without the need for an external
reward model. Our framework couples (i) a path generator that proposes paths
by sampling from pools of unmasking sets with (ii) a verifier that computes the
uncertainty of the proposed paths and performs importance sampling to subse-
quently select the final paths. Empirically, erroneous unmasking measurably in-
flates sequence-level uncertainty, and our method exploits this to avoid error-prone
trajectories. We validate our framework across six benchmarks, such as mathe-
matics, planning, and coding, and demonstrate consistent performance improve-
ments. LookUM requires only two to three paths to achieve peak performance, demon-
strating remarkably efficient path selection. The consistent improvements on both
LLaDA and post-trained LLaDA 1.5 are particularly striking: base LLaDA with
LookUM rivals the performance of RL-tuned LLaDA 1.5, while LookUM further
enhances LLaDA 1.5 itself—showing that uncertainty-based verification provides
orthogonal benefits to reinforcement learning and underscoring the versatility of
our framework. Code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes **Lookahead Unmasking (LookUM)**, a method to improve inference in masked diffusion models by optimizing the order of token unmasking through a two-step process: a *path generator* that proposes candidate unmasking sets, and a *path verifier* that scores these candidates using uncertainty estimated from one-step-ahead (“lookahead”) predictions. The core idea that I get is that local unmasking errors increase sequence-level uncertainty, and that LookUM can potentially avoid these error-prone trajectories by verifying potential paths before committing to an unmasking order. The paper shows empirical gains across six benchmarks (mathematical reasoning, planning, and code generation) using LLaDA and LLaDA 1.5 models, claiming improved accuracy with minimal computational overhead. The results suggest that uncertainty-based lookahead offers complementary benefits to reinforcement learning–based fine-tuning.

### Strengths
- The authors reformulate unmasking as a path selection problem, offering a conceptually clean framing of inference-time decision-making in diffusion language models.

- They demonstrate strong empirical improvements across multiple reasoning tasks (up to +8 points on MBPP, +4 on GSM8K), even over RL-tuned LLaDA 1.5.

- It's computationally efficient — optimal performance achieved with only 2–3 candidate paths, comparable in cost to classifier-free guidance.

- The paper is clearly written and well-organized, with ablation studies separating the impact of each component (path generator, verifier, sampling scheme).

- Demonstrates complementary improvements to existing reinforcement learning–based optimization, suggesting broader applicability.

### Weaknesses
1. The claimed theoretical motivation for the verifier is superficial. While the paper suggests that uncertainty correlates with path correctness, there is no formal justification (like via KL optimal control or stochastic path planning theory) linking the proposed verifier to optimal decoding behavior.

2. The definition and construction of the “lookahead state” $\tilde{x}_{t-1}$ isn't well put forth. It is unclear whether this state is sampled deterministically (argmax) or stochastically (categorical sampling), or whether such a proxy meaningfully represents the true next-step state distribution.

3. The verifier’s uncertainty signal is evaluated only using entropy or confidence; there is no evidence is given that this correlates causally with improved unmasking fidelity or that it generalizes beyond the tested benchmarks.

4. GSM8K performance values in Table 2 (≈70s) do not match Figure 3 (≈30), suggesting either metric mismatch or reporting inconsistency?

5. Key hyperparameters (e.g., the threshold in Certainty Filtering, Table 4) are missing, and sensitivity analyses for these thresholds are not reported.
6. The biggest weakness is by far the lack of comparison to **Path Planning (P²)** sampling (Peng et al., 2025), which addresses an almost identical problem—selecting optimal unmasking sequences via planning. As a result, this paper lacks significant novelty.

7. The improvement claims depend entirely on *in silico* reasoning accuracy. There are no wet-lab or real-world experimental validations to show whether uncertainty-guided decoding actually leads to more robust or interpretable outcomes beyond benchmark accuracy.

### Questions
1. How is the “lookahead state” $\tilde{x}_{t-1}$ sampled — via argmax, sampling, or another scheme? How sensitive is the verifier’s performance to this choice?  

2. Can the authors provide quantitative evidence (with correlation plots or ablation curves) showing that entropy-based uncertainty correlates with true reasoning correctness?  

3. What is the threshold used in Certainty Filtering (Table 4), and how does performance vary with different threshold values?  

4. How do the authors reconcile the mismatch between GSM8K scores reported in Table 2 and Figure 3?  

5. Why haven't the authors compared LookUM to Path Planning (P2) or other planning-based decoding frameworks? If not, can they justify the omission?  

6. Could a variant of LookUM integrate model-internal signals (attention entropy, gradient magnitudes, etc.) into the verifier?  

7. Beyond benchmark metrics, the authors shoudl perform a wet-lab or grounded validation to test whether uncertainty-based decoding yields outputs that are more interpretable, verifiable, or experimentally meaningful?

With sound responses to these questions, I'd be willing to raise the score to a 4 or even a 6.

### Soundness
2

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
This paper introduces Lookahead Unmasking (LookUM), an inference-time decoding framework for masked diffusion language models (MDLMs). The method reframes unmasking as a path selection problem: at each step, multiple unmasking “paths” are proposed, and a verifier based on sequence-level uncertainty selects the most consistent one. The approach is model-agnostic, requires no fine-tuning, and achieves consistent performance gains on reasoning, coding, and planning benchmarks with LLaDA and LLaDA-1.5.

### Strengths
The paper is well-written, clear, and motivated by a relevant problem — improving diffusion model decoding efficiency.

The formulation of unmasking as path selection is intuitive and offers a unifying framework that could, in principle, incorporate uncertainty, reward, or heuristic guidance.

The experimental setup is extensive, covering multiple reasoning and coding benchmarks and both base and RL-tuned diffusion LMs.

### Weaknesses
#### **1. Unfair comparison due to unequal compute budgets**

The main experimental results (e.g., Table 2 ) compare LookUM — which explicitly samples **multiple paths (2–4 per step)** — against baseline methods that only use **a single sampling trajectory**. This means that LookUM’s inference-time compute is **2–4× higher**, as each path requires a separate forward pass through the model .
The authors claim that “the verifier overhead is negligible,” but the dominant cost in MDLM inference is model evaluation itself, not verifier scoring. Thus, a method using (k) paths incurs roughly (k\times) inference cost. Comparing multi-path results against single-path baselines is **not a fair comparison** of sampling quality per unit compute.

For example, if LookUM achieves higher accuracy on HumanEval or GSM8K using 2 paths, it is effectively performing twice the work. The proper control would be either (a) match compute (e.g., let baselines sample twice), or (b) report performance *per unit of FLOPs* or wall-time. Without this normalization, the empirical improvement is difficult to interpret.

#### **2. Comparison with Baselines**

The conceptual framework of treating unmasking as *path planning or path selection* has already appeared in earlier diffusion-decoding work. Prior studies (e.g., those exploring **path-planning for masked diffusion models** and **
Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions**.  The authors should compare with them.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

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
This paper introduces Lookahead Unmasking (LookUM) for diffusion language models, reframing decoding as path selection over unmasking orders. A path generator samples candidate unmasking sets from a high-certainty pool, and a verifier scores one-step lookahead states using sequence-level uncertainty (avg. negative entropy or confidence), selecting paths via importance sampling (SMC/NIS). LookUM yields consistent gains on several reasoning benchmarks.

### Strengths
- The proposed method is conceptually neat and easy to plug into existing pretrained diffusion language models. It's also a good combination of search algorithms and diffusion language models.
- The proposed method consistently improves performance on several reasoning benchmarks (math, code, sudoku) compared with recent baselines.
- This paper conducts detailed ablation studies on components of the proposed method and explores the integration with external reward models.

### Weaknesses
- While the proposed method shows improvements on benchmarks, the score difference compared to the best baselines is not very large, especially given that the proposed method's computational cost is 2-3$\times$ as much.
- This paper doesn't show or compare measurements on the actual inference cost. That would make the performance-cost trade-off clearer.
- The number of lookahead steps is an important hyperparameter for the proposed method. Throughout the paper it's fixed to be $1$. Why not consider more lookahead steps (e.g., more candidate branches, fewer lookahead steps VS fewer candidate branches, more lookahead steps, under the same computational budget)? What could be the difficulties?

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Masked Diffusion Models (MDMs) train with an any-order objective, allowing multiple possible sampling paths. This paper addresses the problem of finding an optimal unmasking path during inference. Existing heuristic strategies are typically locally greedy and fail to capture sequence-level dependencies. To address this, the authors propose Lookahead Unmasking (LookUM) — a method that uses the average uncertainty of the next step to guide path selection. LookUM reframes decoding as a path selection problem, consisting of two components: a path generator that proposes candidate unmasking paths, and a verifier that scores these paths using sequence-level uncertainty. The paper reports strong empirical results.

### Strengths
- The paper proposes a simple yet effective inference-time approach for identifying optimal unmasking paths without modifying the training process.
- The method is evaluated across multiple benchmarks and achieves strong empirical performance compared to existing baselines.

### Weaknesses
- While LookUM increases inference-time computation, the paper does not clearly quantify this overhead compared to baseline methods. A more detailed analysis of inference time should be included in the paper.
- Another way to improve diffusion model performance under a higher inference-time budget is to increase the number of denoising steps. It would be interesting to compare LookUM against this baseline under a fixed compute or time budget, to better understand its effectiveness.
- The approach performs multiple forward passes for each inference step. Although conceptually simple, this can substantially increase inference time. However, it might not be necessary to perform LookUM during each of the inference steps. It would be useful to explore whether using any method could decide when to apply lookahead unmasking — potentially avoiding unnecessary lookahead steps and reducing the inference cost without significant performance loss.

### Questions
- Would LookUM perform better if the verifier did not average negative entropy over all tokens? Perhaps focusing on a subset of the most uncertain tokens could reduce noise in the uncertainty estimation.
- Is there any intuitive reason of the performance drop as we increase the number of paths (e.g., in MATH500)? How frequently does it occur? One possible explanation is that the estimated uncertainty does not perfectly correlate with the true path quality.

### Soundness
3

### Presentation
3

### Contribution
2
