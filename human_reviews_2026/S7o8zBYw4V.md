# DARE:Difficulty-Aware Dynamic Routing for Mixture of Experts

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 4

## Abstract
Sparse Mixture-of-Experts (MoE) architectures have become a foundational approach for efficiently scaling Large Vision-Language Models (LVLMs), as they activate only a subset of parameters for each input. However, the commonly adopted Top-K routing strategy assigns a fixed number of experts to every token, ignoring the natural variation in token complexity. This static allocation often results in suboptimal resource utilization, where simple tokens receive excessive computation and complex tokens are insufficiently processed. While recent dynamic routing methods attempt to address this limitation, they lack principled mechanisms to explicitly guide expert allocation based on token-level difficulty, resulting in suboptimal performance in practice.
In this paper, we propose \textbf{D}ifficulty-\textbf{A}ware Dynamic \textbf{R}outing for Mixture of \textbf{E}xperts (\textbf{DARE}), a novel routing strategy that adapts expert selection according to the complexity of each token. DARE %incorporates
introduces a lightweight predictor that estimates the difficulty of individual tokens based on their log-perplexity as a theoretically grounded proxy, and employs a set of learnable thresholds to dynamically determine the appropriate number of experts to activate. This mechanism enables fine-grained and adaptive allocation of computational resources, allowing the model to devote more capacity to challenging tokens while conserving resources on easier ones. Extensive experiments on standard vision-language benchmarks demonstrate that DARE consistently outperforms both fixed Top-K routing and existing adaptive routing strategies. It achieves superior task performance while simultaneously improving computational efficiency, highlighting the effectiveness and generality of difficulty-aware routing in sparse MoE architectures for large-scale multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DARE, a dynamic routing mechanism for MoE architectures that explicitly models token-level difficulty when allocating experts. Instead of using a fixed number of experts per token (as in Top-K routing), DARE introduces a lightweight difficulty predictor based on log-perplexity to estimate the complexity of each token. A learnable thresholding mechanism then dynamically determines how many experts to activate per token, enabling adaptive and efficient resource allocation. Experiments show that DARE improves accuracy while reducing computation by 39% compared to Top-K routing, outperforming recent dynamic methods.

### Strengths
1. Dynamic routing is a core topic in the study of MoE routing mechanisms.

2. Intuitively, the dynamic routing design of this paper makes sense.

### Weaknesses
1. _Overclaim_. Some prior works have also designed dynamic routing strategies based on token-level difficulty, such as Top-p and DYNMoE. Therefore, Lines 61–62 appear somewhat overclaimed. In addition, although the authors provide empirical comparisons with existing dynamic routing approaches, they do not conceptually clarify the advantages of the proposed prediction-based method over these approaches, raising concerns about its generalization.

2. _Unclear overhead of the difficulty predictor_. The token difficulty is estimated through an additional difficulty predictor. However, the paper does not report the computational overhead introduced by the difficulty predictor.

3. _Insufficient experiments_. The experiments are mainly conducted on MoE-LLaVA, without validation on other MoE models. Moreover, MoE-LLaVA (2023) is not a native MoE model but a dense model upcycled into an MoE variant, and its sparsity ratio (activated experts / total experts) is relatively high, which deviates considerably from current mainstream MoE configurations. Thus, further experiments on recent and representative MoE architectures are needed to validate the effectiveness of the proposed method.

4. The proposed method does not seem to include any design tailored for multimodal scenarios, so it would be helpful to add experiments on language-only tasks.

In conclusion, this work lacks a clear conceptual comparison with existing difficulty-based dynamic routing methods, making it difficult to assess the effectiveness of the proposed approach across different scenarios. Furthermore, the absence of experiments on recent mainstream MoE models further raises concerns about the effectiveness. Therefore, my current rating is Reject.

### Questions
Please refer to Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DARE, a novel dynamic routing strategy for Mixture-of-Experts (MoE) models in the context of Large Vision-Language Models (LVLMs). The core idea is to dynamically allocate a variable number of experts to each token based on its estimated difficulty, which is proxied by log-perplexity. By introducing a lightweight difficulty predictor and an online threshold adaptation mechanism, DARE aims to assign more computational resources to complex tokens and fewer to simple ones, thereby improving both task performance and computational efficiency compared to static Top-K routing and other existing dynamic methods.

### Strengths
1. This paper is well-motivated. It addresses a significant and well-recognized limitation of standard MoE architectures. DARE chooses the logarithmic perplexity as a proxy for difficulty, which is a theoretically well-grounded metric directly related to the model’s own uncertainty and loss function. 
2. This paper conduct extensive experiments across multiple standard vision-language benchmarks and on various LLM backbones.
3. The paper is well-written, logically structured, and easy to follow.

### Weaknesses
1. Dependence on the predefined target distribution \Pi. The online threshold adaptation mechanism relies on a manually predefined target expert assignment distribution \Pi (for example, for Qwen2-1.5B, Π = [0.6, 0.3, 0.09, 0.01]). This makes \Pi a critical, model-specific hyperparameter.
2. Limitation of the Difficulty Proxy: The paper relies solely on log-perplexity as the difficulty measure. While this effectively captures prediction uncertainty, it does not reflect semantic complexity or task relevance. Some low-perplexity tokens may still require deeper reasoning, while high-perplexity tokens could simply be rare but unimportant. The paper would benefit from discussing these limitations and considering alternative or combined proxies—such as integrating perplexity with gradient-based importance or attention scores.
3. The Number of Experts. All experiments in this paper appear to be conducted with only four experts per MoE layer (M = 4). It remains an open and important question how DARE performs and behaves when the number of experts increases. Would using a simple quantile-based threshold still work well, or could it cause some experts to be used less than they should? A discussion or empirical analysis on the scalability of DARE under larger MoE configurations would make the paper more convincing.

### Questions
1. Regarding the target distribution \Pi: Have you experimented with learning this distribution or using a parametric form instead of setting discrete values manually? How sensitive are the results to small perturbations in the \Pi_k values? Providing some intuition on how to set \Pi for a new model would be very helpful for practitioners.
2. Is the architecture of the difficulty predictor (a simple MLP) sufficient? Have you tried more complex structures, such as using attention mechanisms to aggregate contextual information for more accurate difficulty prediction?
3. In the Analysis of Expert Load Balance(Figure 4), DARE demonstrates outstanding performance. Does this balance come at the expense of expert specialization? Could a more quantitative analysis be provided—for example, measuring each expert’s activation preferences when processing different types of tokens (such as nouns, verbs, or domain-specific terms)? This would offer deeper insights into expert behavior under the DARE routing mechanism.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on routing strategies of sparse MoE in vision-language models, noting that a fixed Top-K causes excessive computation for easy tokens and insufficient computation for difficult ones. It therefore proposes DARE: a lightweight MLP first predicts token difficulty from hidden representations (using log perplexity as an interpretable proxy), then maps the predicted difficulty to “the number of experts to activate in this layer” through a set of learnable thresholds, thus dynamically deciding how many experts to use per token. The authors provide an analysis of the negative correlation between difficulty and accuracy, the complete training objective (task CE, difficulty regression, load balancing), and multi-benchmark experiments based on Qwen2-1.5B, Qwen3-1.7B, and StableLM-1.6B. The main conclusion is: on multiple benchmarks, performance is comparable to or better than strong baselines, while reducing the average number of activated experts to about 1.22, and achieving actual gains in inference FLOPs, throughput, and latency.

### Strengths
The most commendable aspect is the closed loop between motivation and evidence: from the correlation observation in Fig. 2, to using log perplexity for difficulty supervision, to mapping difficulty into “number of experts” through threshold stratification—the technical line is clear, interpretable, and consistently reproducible across benchmarks.
In terms of efficiency, the authors not only report a significant reduction in the average number of activated experts, but also demonstrate end-to-end benefits under the same hardware setup: FLOPs/token reduced to 52.25, throughput increased to 75 tok/s, latency shortened to 5.4 s, with memory usage almost unchanged compared to Top-2 MoE-LLaVA.
More detailed visualizations (e.g., layer–task activation counts, load-balancing differences across methods) further aid understanding of routing mechanisms.
It’s worth emphasizing your observation that “simply reducing activated experts can speed up inference with almost no performance drop, especially under fixed-expert + Top-K setups”—this paper’s data shows traces of that: compared to fixed Top-2, DARE’s lower Avg-K yields similar scores, and efficiency gains align with the drop in Avg-K. However, cases like MoE++ and DynMoE show performance degradation when Avg-K ≈ 1, suggesting that “reducing Avg-K” alone isn’t sufficient—difficulty-aware routing still makes a difference.

### Weaknesses
The biggest comparability issue lies in the “number of experts” setting. Using 4 experts may underestimate the potential of dynamic baselines, so conclusions drawn solely from 4-expert setups have limited persuasiveness. For concerns like “should expert-selection methods be tested with more experts,” I agree that replication should at least include the common scale 16 to avoid conclusions tied to low-expert special cases.

The efficiency evaluation chain is also incomplete. The speed comparison only directly measures DARE vs. MoE-LLaVA (Top-2) in memory, FLOPs, throughput, and latency; neither the main paper nor the appendix includes speed results for MoE++,Top-p, ReMoE, or DynMoE, which appear in the performance tables. There are also no controlled experiments under “same Avg-K” or “same number of activated parameters,” leaving the efficiency gain from “difficulty awareness” mixed with that from “fewer activated experts.” Your third comment fully applies here—those baselines should be added with Avg-K alignment.

Statistical robustness still needs improvement. The main result table lacks error bars or repetitions (the efficiency table mentions “means over five independent runs,” but performance tables do not), making a few-to-ten-point gap hard to judge for significance.

### Questions
Please include additional experiments with multiple experts and improve the speed tests. A more thorough response and supplementation of these issues may positively influence my assessment of the paper’s persuasiveness.

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
This paper introduces DARE (Difficulty-Aware Dynamic Routing), a novel routing mechanism for Sparse Mixture-of-Experts (MoE) models. The central thesis is that the conventional Top-K routing strategy is suboptimal as it allocates a fixed computational budget (i.e., a fixed number of experts) to every token, irrespective of its intrinsic complexity. To address this, DARE proposes a dynamic allocation strategy guided by an explicit measure of token-level difficulty. Specifically, the method uses log-perplexity as a proxy for difficulty. A lightweight MLP, termed the "difficulty predictor," is trained alongside the main model to estimate this difficulty score for each token's hidden representation. Based on the predicted score, a variable number of experts $K^*$ is selected for each token by comparing the score against a set of dynamically adjusted thresholds. These thresholds are updated online using an exponential moving average (EMA) on the quantiles of the difficulty distribution within each mini-batch, aiming to match a predefined target distribution of expert counts. The authors evaluate DARE on several vision-language benchmarks, demonstrating superior performance and computational efficiency compared to fixed Top-K routing and other recent dynamic routing strategies.

### Strengths
1.  The paper correctly identifies a key limitation of the standard Top-K routing in MoE models and provides a clear motivation for a more adaptive approach.
2.  Despite methodological concerns, the paper presents compelling empirical results across multiple vision-language benchmarks and three different model backbones. DARE consistently outperforms various baselines, including both static and dynamic routing methods, often while using fewer computational resources.
3. The authors provide a good set of analyses, including ablations on the core components of their method, visualizations of expert allocation patterns, and measurements of inference efficiency.

### Weaknesses
1.  No other potential difficulty signals (e.g., gating entropy, gradient norms, attention-based metrics) are explored or even discussed as alternatives in the main text. This makes the design choice seem arbitrary rather than principled.
2.  DARE introduces a non-trivial amount of new machinery: an MLP predictor with its own learning rate, an auxiliary loss weighted by $\alpha$, a target distribution $\Pi$ that must be specified per model, and an EMA momentum $\gamma$. This complexity is a clear disadvantage compared to the elegance of Top-K routing. The lack of a sensitivity analysis for these new hyperparameters is a glaring omission.

### Questions
1.  Could you provide a more detailed justification for using a separate, supervised difficulty predictor MLP instead of a simpler, unsupervised signal derived directly from the gating network's output distribution (e.g., its entropy)? Did you perform any experiments comparing these alternatives?
2.  The target distribution $\Pi$ appears to be a critical, manually-tuned hyperparameter. How were the specific values for each backbone model (e.g., $\Pi = (0.6, 0.3, 0.09, 0.01)$ for Qwen2-1.5B) determined? Please provide an ablation study showing how performance changes with different settings of $\Pi$, for instance, comparing your chosen "long-tailed" distribution to a uniform one and another differently-skewed one.

### Soundness
2

### Presentation
3

### Contribution
2
