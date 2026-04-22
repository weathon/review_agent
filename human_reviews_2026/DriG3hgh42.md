# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to *9.1*$\times$ speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. *Codes are provided in the supplementary material and will be released publicly on GitHub.*

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces dLLM-Cache, a novel training-free framework designed to accelerate
inference in diffusion LLMs. It proposes a dual-component caching strategy. First, it
employs long-interval caching for the static input prompt, refreshing its features only
infrequently. Second, it uses an adaptive, short-interval caching mechanism for the
dynamically evolving response. This adaptive component is guided by a lightweight &quot;V-
verify&quot; mechanism. The paper demonstrates through extensive experiments on LLaDA and
Dream models that dLLM-Cache can achieve up to a 9.1x speedup, bringing dLLM inference
latency closer to that of autoregressive models without compromising output quality.

### Strengths
1. The paper addresses a critical and widely recognized problem: the high inference
cost of diffusion LLMs. The achieved speedups are substantial and represent a major
step towards making these models practical for real-world applications.

2. The V-verify mechanism is a clever, lightweight, and empirically-grounded solution
for adaptively selecting which tokens to update, avoiding the overhead of more
complex methods.
3. A key strength is that dLLM-Cache does not require any model retraining. This
makes the method easy to adopt.

### Weaknesses
1. The high-level diagrams provide a helpful overview of the caching workflow.
However, the specific structure of the cache, particularly whether it is a single global
entity or maintained on a per-layer basis, is not explicitly detailed.
2. The paper could better justify the necessity of caching four feature types (K, V,
AttnOut, FFNOut), as this is a departure from the more standard KV-caching in
ARMs.
3. It is unclear why some configurations, such as MMLU with LLaDA Instruct, benefit
less from caching than the Base model. An analysis for the variance in speedup
across different benchmarks may be beneficial.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a method for the caching mechanism in the diffusion large language models. The contribution of this paper includes the v-verify method to detect the tokens that show the most changed tokens for updating and then would not use the cache at those tokens. Experimental results show that the proposed method can achieve large inference acceleration compared to LLaDA and Dream.

### Strengths
1. The proposed V-Verify module effectively identifies and selects tokens for caching.
2. The method is well-motivated and demonstrates strong performance in accelerating dLLMs.

### Weaknesses
1. Contribution is limited

The paper claims two main contributions: (i) adopting different update intervals for the prompt and response, and (ii) introducing V-verify to identify the most changed tokens for partial updates. However, the first contribution, also emphasized in the main analysis experiments (Section 3.2), has already been explored in prior works on KV-Cache for diffusion LLMs, such as dKV-Cache (the prefill part) and Fast-dLLM (PrefixCache). The other contribution is the V-verify mechanism, which selects tokens to be cached based on the similarity between their V vectors. While this idea is interesting, it appears too incremental to constitute a sufficient contribution for a full paper.

2. Comparison with AR models requires stronger justification and additional results

The comparison with autoregressive models presented in Table 4 lacks clarity and fairness. The reported acceleration is marginal (47.73 vs 49.86 TPS), and the base model used for comparison is not instruction-tuned for reasoning tasks like GSM8K, making the results less meaningful. For instance, using a more appropriate baseline such as Llama3-8B-Instruct would yield around 79.6% accuracy on GSM8K, significantly higher than the 67.2% reported here. An apple-to-apple comparison is needed.

Furthermore, the procedure for measuring TPS and the experimental setup are unclear. From the FLOPs calculation in the appendix, it seems that diffusion LLMs inherently require more computation than AR models, so it is counterintuitive that they achieve higher speed. Clarification on this discrepancy and a more thorough comparison of speed with ARs is needed. 

In addition, the efficiency evaluation is conducted with only 128 steps, resulting in relatively poor accuracy (67%) compared to the 71% and 78% reported in Table 1. A more comprehensive comparison across different steps is needed.

### Questions
1. In the v-verify stage, since q is not recalculated and only a subset of tokens is available for q, the input x to the attention input 
x only has the hidden states with partial tokens. If we only have partial x/q/k/v vectors, how can the corresponding v vectors for the cached tokens be calculated as a new one?

2. Could you elaborate on why your approach outperforms dKV-Cache and Fast-dLLM? It would be helpful to provide some insights or for this improvement.

3. Is your method sensitive to the number of decoding steps? Given that many recent works support parallel decoding with multiple tokens per step and I think it would be a trend, how would your method perform when applied to those settings?

### Soundness
3

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
This paper introduces dLLM-Cache, an adaptive caching framework designed to accelerate diffusion-based large language models (dLLMs) without retraining. The key idea is to cache intermediate representations during inference, leveraging redundancy in prompts and generated tokens. The proposed cache selection strategy dynamically balances accuracy and efficiency using prompt similarity and token reuse mechanisms. Experiments on various LLM-based diffusion tasks (e.g., LLaDA, DreamFusion) show that dLLM-Cache achieves substantial inference speedup with minimal degradation in performance.

### Strengths
1. The paper is clearly written and well-structured, making the method easy to understand.
2. The adaptive caching mechanism is simple yet effective, requiring no retraining or architectural modification.
3. Comprehensive experiments demonstrate strong empirical improvements across multiple diffusion-based LLMs.
4. Ablation studies support the robustness of the cache selection mechanism.

### Weaknesses
1. The theoretical complexity reduction is not analyzed in sufficient depth. A more formal discussion or derivation of the speed–accuracy trade-off would greatly enhance clarity.
2. Since dLLM-Cache reuses outdated KV pairs, it would strengthen the work to mathematically and empirically analyze the upper bound of the approximation error under different values of K.
3. The limitations of the proposed method, particularly under dynamic or semantically diverse prompts, are not fully explored. A deeper investigation of such edge cases would improve the completeness of the study.
4. Several tables (e.g., Table 1, Table 2, and Table 3) present overlapping or repetitive results, with only minor variations in metrics or experimental settings. These could be merged or reorganized to improve readability and reduce redundancy.
5. The observation that using outdated cache can sometimes improve accuracy is intriguing but insufficiently discussed. A more detailed explanation or hypothesis about why this occurs would make the findings more convincing.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
