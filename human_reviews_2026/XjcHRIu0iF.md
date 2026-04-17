# Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Masked diffusion models (MDMs) offer a compelling alternative to autoregres-
sive models (ARMs) for discrete text generation because they enable parallel
token sampling, rather than sequential, left-to-right generation. This means po-
tentially much faster inference. However, effective parallel sampling faces two
competing requirements: (i) simultaneously updated tokens must be conditionally
independent, and (ii) updates should prioritise high-confidence predictions. These
goals conflict because high-confidence predictions often cluster and depend on
each other, opportunities for parallel updates.

We present PUNT, a model-agnostic sampler that reconciles this trade-off. Our
method identifies token dependencies and removes lower-confidence tokens from
conflicting groups. This produces sets of indices for unmasking that satisfy both
independence and confidence criteria. Our approach ensures improved parallel
unmasking through approximate conditional independence testing.

Our experiments show that PUNT delivers a superior trade-off between accuracy
and compute when compared to other strong training-free baselines, especially for
generation of longer sequences. On the IFEval benchmark, it achieves up to 16%
higher accuracy over baseline methods, including sequential generation (one-by-
one). These gains hold across different values of hyperparameters, mitigating the
need for brittle hyperparameter tuning. Moreover, we observe that PUNT induces
an emergent hierarchical generation strategy, where the model first establishes
high-level paragraph structure before local refinement, suggesting a planning-like
generation process that contributes to strong alignment performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PUNT (Parallel Unmasking with Non-influence Tests), a training-free sampling algorithm for Masked Diffusion Models (MDMs) that enables efficient parallel token generation while maintaining output quality. The key idea is to select subsets of masked tokens that are approximately conditionally independent, thereby allowing them to be decoded simultaneously without introducing dependency errors. PUNT implements a divide-and-conquer procedure that identifies such token sets using O(log |M|) model evaluations per iteration, relying on a contextual independence test based on KL divergence between conditional distributions.

### Strengths
1. The paper is built on a very clear and reasonable motivation. It correctly identifies the fundamental conflict in parallel MDM sampling: the desire to unmask high-confidence tokens versus the necessity of conditional independence between simultaneously updated tokens. PUNT's approach of directly testing for and mitigating this "inter-token interference" is a sound and principled alternative to purely confidence-based heuristics.
2. PUNT demonstrates a clear Pareto improvement on long-sequence and instruction-following tasks like IFEval. The observation of an emergent hierarchical generation strategy (Fig. 2) is a significant and interesting finding.
3. The connection between independence stability and attention sparsity offers an intuitive, architecture-aware justification of the method’s assumptions.

### Weaknesses
1. Experimental validation is limited. The current experiments mainly demonstrate advantages on two benchmarks (IFEval and MT-Bench), while the improvement on MT-Bench over the Dilated Sampler is marginal. On short-answer tasks (§4.2), PUNT shows no clear advantage—likely due to higher per-step complexity. The evaluation lacks additional baselines such as APD (Adaptive Parallel Decoding, arXiv:2506.00413) or other few-step DLM planners that could better contextualize PUNT’s trade-offs.
2. Definition 3.2 relies on a single, fixed sequential order ($X^i$ given $X_{<i}$). The algorithm implements this order by sorting all masked tokens $M$ based on their initial confidence (at the start of the step). This static, confidence-based ordering may be suboptimal, as token confidences are likely to change once other tokens are (tentatively) revealed.
3. The actual implementation (Sec 3.3, Alg. 1) appears to use a much stricter test than the sequential one defined in Definition 3.2. It performs a batched test, checking all tokens in the "test" set $S_1$ for dependence on the entire "anchor" set $S_0$. This is stricter than Eq. 2, which only tests a token $r_i$ against preceding tokens $R_{<i}$(31). Does this simplification, made for the sake of parallel computation, lead to over-pruning?

### Questions
1. The paper proposes $\epsilon$ as a fixed hyperparameter. Have the authors explored using a dynamic $\epsilon$? For instance, a schedule that starts with a small, strict $\epsilon$ (to establish the high-level structure, and then increases $\epsilon$ in later steps to be more aggressive and rapidly fill in local details? 
2. How sensitive is PUNT’s performance to the confidence ordering strategy?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents PUNT, a model-agnostic sampler that reconciles the trade-off between speed and quality in MDMs by using approximate conditional independence testing to identify and resolve token dependencies. PUNT delivers a superior trade-off between accuracy and compute, especially for longer sequences, without requiring brittle hyperparameter tuning.

### Strengths
- PUNT is efficient, training-free and dynamically adapts to sequence-specific dependencies.
- PUNT induces an emergent hierarchical generation strategy, suggesting a planning-like process that contributes to its strong performance.

### Weaknesses
- The "independence stability" assumption, which PUNT relies on, is a strong approximation but it is not proven.
- The claim of mitigating "brittle hyperparameter tuning" is insufficiently supported as no sensitivity analysis is provided for $\epsilon$.
- The method underperforms on short-answer tasks where the computational overhead of multiple forward passes per step is not amortized.

### Questions
How was the hyperparameter $\epsilon$ selected? How is the method sensitive to the hyperparameter?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PUNT, a new model-agnostic sampler applied in Masked diffusion models. By making the use of a proposed Contextual Independence Assumption and the corresponding recursion algorithm,  PUNT enables to select the postions from masked positions set M that relatively independent of all |-M| umasked positions for decoding, in terms of a time complexity O(log |M|). By taking contextual independence into consideration, PUNT can efficiently decode multiple tokens in parallel, while keeps its decoding accuracy.

### Strengths
This paper designing PUNT by skillfully utilizing a contextual independence assumption and a recursion algorithm, merging the tests of subcases of one recursion level into one evaluation, which is training-free and efficient. 
There is a point mentioned in section 3.4 that  “assumption3.3 is a direct consequence of the Transformer architecture’s attention mechansim. If the attention from position i to position j is zero, then position j has no direct influence on the representation at position i.” This point of view shows the relationship between sparse attention and the testing of contextual independence. Perhaps this can serve as an inspiration to utilize independent testing to identify sparse locations.

### Weaknesses
I noticed that in the divide-and-conquer process, some tokens are considered as ‘dependent on the anchor’  in a previous evaluation, but soon be considered as ‘independent’ in the next evaluation (such as the token ‘mince’ in Figure 1,left). It seems that only tokens tested to be dependent on the anchors of all evaluations are considered as ‘dependent’ and absent from the current generation, which means tokens in parallel may also be dependent on each other. It seems that PUNT is less capable of separating the dependency on tokens than DILATED when NFE < 100 (Figure 3 left). 

As shown Figure 4, it seems that PUNT fails to exceed DILATED at NFE 400 on MT-Bench, but  there is a lack of sufficient explanation for this phenomenon.

### Questions
As shown in Figure 1, why to take the tokens  that are always dependent on anchors (such as ‘egg’) among all tests, rather than tokens that rely on anchors at least once among all tests (such as ‘mince’), as the final ‘rejected’?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the conflict between speed and quality in Masked Diffusion Models (MDMs). The authors propose **PUNT**, a **training-free sampler** that enables efficient parallel decoding by **explicitly testing for token dependencies**. PUNT identifies and prunes tokens that are not conditionally independent, using a recursive $O(\log|M|)$ **algorithm**. Experiments show PUNT achieves a **state-of-the-art accuracy-compute trade-off**, especially on long-form generation tasks.

### Strengths
1.  **Novel and Effective Algorithm:** PUNT is an **elegant and efficient** $O(\log|M|)$ **solution** to a well-defined problem (confidence vs. independence in parallel sampling). The use of explicit independence testing is a strong contribution.

2.  **Strong Empirical Results:** The method **clearly outperforms strong baselines** on the accuracy-compute Pareto frontier for relevant long-sequence benchmarks (IFEval, MT-Bench).

3.  **Strong Theoretical Justification:** The method is well-grounded with **strong theoretical justification**, particularly by connecting its "Independence Stability" assumption to the properties of Transformer attention. The discovery of an **emergent coarse-to-fine generation strategy** is also a valuable insight.

### Weaknesses
1.  **Limited Scope:** The method's advantages are **diminished on short-sequence tasks** (GSM8K, MBPP), where its $O(\log|M|)$ NFE-per-step overhead is less efficient than simpler samplers.

2.  **Critical Hyperparameter Sensitivity:** The algorithm's effectiveness hinges entirely on the **KL divergence threshold** $\epsilon$, which is not a simple-to-tune parameter but a **fundamental trade-off between speed and quality**. For a task with highly dependent tokens (e.g., code generation), most tokens will have a high $D_{KL}$. The user is forced into an impossible choice:
    - Set $\epsilon$ **low** for *quality*: This respects the dependencies, but will prune nearly all tokens, **collapsing the sampler's speed** to be sequential.
    - Set $\epsilon$ **high** for *speed*: This ignores the dependencies to unmask more tokens, but will **destroy generation quality** by violating the independence assumption.
    This makes $\epsilon$ a critical, task-specific parameter that requires a costly sweep for any new model or domain.

I also think that this parameter is not well ablated in the paper

3.  **Unclear Baseline:** The abstract's claim of outperforming "sequential generation" is confusing, as the `TOPK` baseline in the plots performs poorly, suggesting it's **not a true one-token-at-a-time sequential baseline**.

4.  **Misleading Efficiency Metric:** The paper presents plots against **"Denoising steps"** alongside **NFE**. This "step" metric is misleading, as a single PUNT step is $O(\log|M|)$ more expensive than a baseline step. **NFE is the only meaningful measure of compute**, and the focus on "steps" can obscure the true cost.

### Questions
I wonder if similar results could be explored in image generation pipelines like MaskGIT

### Soundness
4

### Presentation
4

### Contribution
4
