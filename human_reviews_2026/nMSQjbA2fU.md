# Data-Free Pruning of Self-Attention Layers in LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Many self-attention sublayers in large language models (LLMs) can be removed with little to no loss. We attribute this to the Attention Suppression Hypothesis: during pre-training, some deep attention layers learn to mute their own contribution, leaving the residual stream and the MLP to carry the representation. 
We propose Gate-Norm, a one-shot, weight-only criterion that ranks attention sublayers by query--key coupling and removes the least coupled ones—requiring no calibration data, no forward passes, no fine-tuning, and no specialized kernels. On 40-layer, 13B-parameter LLaMA models, Gate-Norm prunes the model under a second. Pruning $8$–$16$ attention sublayers yields up to $1.30\times$ higher inference throughput while keeping average zero-shot accuracy within $2\%$ of the unpruned baseline across BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy/Challenge, and OpenBookQA. 
Across these settings, Gate-Norm matches data-driven pruning methods in accuracy while being $\sim\ 1000\times$ faster to score layers, enabling practical, data-free compression of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates structural redundancy in LLMs and proposes a data-free pruning method for self-attention sublayers, motivated by the _Attention Suppression Hypothesis_ — the claim that deeper attention layers in pretrained LLMs often learn to mute their own outputs, leaving the MLP and residual path to carry most representational information.

### Strengths
- Clear theoretical grounding via the Attention Suppression Hypothesis
- The performance–efficiency trade-off is quantitatively superior to prior approaches.
- The paper is clearly written with consistent mathematical notation, well-labeled figures/tables.

### Weaknesses
- Validated only on LLaMA-13B (v1 and v2). It is unclear whether Gate-Norm generalizes to encoder–decoder architectures (e.g., T5), smaller models, or different pretraining objectives.
- While the authors show that $|AttnOut_\ell|/|X_\ell|$ decays with depth, they do not quantify how MLP layers compensate. 
- No confidence intervals or statistical tests are reported.
- No comparison to block-drop + light finetuning; omits instruction-tuned and long-context settings where alternative baselines may behave differently.

### Questions
- Can the Gate-Norm metric detect redundant MLP sublayers, or is it exclusive to attention weights?
- How sensitive is Gate-Norm to reparameterization or scaling?
- Does pruning via Gate-Norm affect downstream finetuning or alignment behaviors like instruction-following tasks?

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
5

### Summary
This paper proposes a data-free pruning method for LLMs. The authors start with an empirical study on LLaMA-13B, deeper attention layers change their hidden states less in both magnitude and direction, revealing an “attention suppression” effect. They argue that deeper layers become redundant, with most representation handled by the residual and MLP parts. Based on this, they define a gate matrix norm (i.e. the Frobenius norm of the query and key weight product) to measure coupling strength. Smaller gate-norm values indicate higher cosine similarity between input and output, meaning less change, so such layers can be safely pruned without major performance loss.

### Strengths
- **Clear and easy to follow.** The paper is generally well structured and readable. The ideas flow logically, and the intuition behind the method is explained in a straightforward way.
- **Simple and data-free approach.** The pruning method doesn’t rely on any additional data or fine-tuning, which makes it simple and relatively easy to implement. The computational cost also seems quite low, even for large models.
- **Reasonably practical and scalable.** The approach looks practical enough to be applied to other large-scale models with only minor changes, showing potential for efficient and hardware-friendly compression in real scenarios.

### Weaknesses
- **Key Concern.** The method is mainly built on the attention suppression phenomenon, which the authors identify through large-scale empirical analysis (e.g., Fig. 3). But this raises a question: if the paper already measures how cosine similarity and norm changes evolve across layers, why not just use those observed metrics directly to decide which layers to prune? In other words, instead of introducing the Gate-Norm as a new proxy, it might be more straightforward to prune layers that already show high similarity and small norm variation in the empirical results. It would be helpful if the authors could explain why Gate-Norm is needed here—does it offer practical advantages (like being faster, more stable, or more generalizable across models), or is it just an indirect way of reflecting the same statistics?
- **Limited empirical validation.** The method relies heavily on the assumption that attention layers exhibit redundancy in deeper parts of the model. However, such redundancy may vary across architectures. The experiments mainly focus on **LLaMA-13B**, with no validation on more recent models such as **LLaMA-2 or LLaMA-3**, which are commonly used in related studies [1–4]. A broader evaluation would help strengthen the generality of the findings.
    
    [1] LLM-Pruner: On the Structural Pruning of Large Language Models. NeurIPS 2023.

    [2] D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models. NeurIPS 2024.

    [3] Skipgpt: Each Token Is One of A Kind. ICML 2025

    [4] Not All Layers of LLMs Are Necessary During Inference. IJCAI 2025.
    
- **Related work coverage.** Since the paper focuses on **attention-layer pruning**, the related-work section could be expanded. It currently cites only one prior work on layer pruning, while several recent studies explore similar or complementary directions. A more detailed discussion would help clarify how this method fits within the broader pruning literature.
- If the authors can address these concerns clearly in the rebuttal, I would be inclined to raise my overall rating.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies redundancy in deep self‑attention layers of large language models (LLMs) and proposes a simple, data‑free method for removing redundant attention sublayers. To operationalize this idea, the paper introduces Gate‑Norm, a weight‑only proxy for measuring an attention sublayer’s token‑mixing strength. Experiments are performed on two LLaMA‑13B checkpoints (v1 and v2). The authors compare Gate‑Norm against block removal (ShortGPT), data‑driven attention pruning (which measures cosine similarity on calibration data), and random removal.

### Strengths
1.  The Gate‑Norm proxy is conceptually simple and easy to implement. It uses only the trained query and key matrices and does not require any calibration data or activation statistics.
2. Experiments demonstrate that pruning 8–16 attention layers yields 1.1–1.3× higher inference throughput while reducing average zero‑shot accuracy by at most ~2 %. The method thus offers a promising depth‑compression strategy for LLM deployment, particularly on devices without GPUs or with strict latency and privacy requirements.

### Weaknesses
1. The experiments focus exclusively on two 13B‑parameter LLaMA models. It remains unclear whether the Attention Suppression phenomenon and the gate‑norm proxy generalize to other model families (e.g., Qwen, Mistral, smaller or larger LLaMA variants)
2. The proposed algorithm either keeps or completely disables an attention sublayer. Finer‑grained options (e.g., partial gating, head pruning) are not explored.
3. The "no fine-tuning" aspect is a strength for one-shot pruning. However, the paper does not explore if a very brief period of fine-tuning (e.g., a few hundred steps) after pruning could fully recover the minor accuracy loss, potentially enabling even more aggressive (e.g., 20+ layer) pruning.

### Questions
See Weaknesses.

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
This paper introduces Gate-Norm, a novel and extremely efficient method for compressing Large Language Models (LLMs) by pruning entire self-attention sublayers. The authors propose the "Attention Suppression Hypothesis," suggesting that during pre-training, some attention layers learn to become functionally inactive, effectively "muting" their contribution. Gate-Norm leverages this insight to create a one-shot, data-free importance score calculated directly from model weights. This allows it to identify and remove redundant attention layers in milliseconds—without needing any calibration data, forward passes, or fine-tuning—achieving significant inference speedups with only a minimal drop in accuracy.

### Strengths
1. The method's greatest strength is its speed and simplicity. Being data-free, one-shot, and running in milliseconds (~1000x faster than alternatives) makes it incredibly practical for on-the-fly, on-device compression without the massive overhead of data-driven approaches.

2. The entire process requires no calibration datasets, no GPUs for the pruning step itself, and no costly post-pruning fine-tuning. This makes the method highly accessible and easy to deploy.

### Weaknesses
1. The method focuses exclusively on attention sublayers. The paper itself notes that MLP layers have twice the parameters and also contribute to runtime, but they are not targeted for pruning by this method.

2. While the results support the hypothesis, the introduction doesn't provide direct evidence for "attention suppression" itself (e.g., by showing near-zero output norms from the targeted layers during inference). The claim rests on the method's success.

3. The criterion for pruning is static and based only on weights. This might incorrectly remove a layer that, while often suppressed, is critical for certain rare but important types of inputs or reasoning paths.

4. This work only discussed the Transformer architecture. Results on MoEs should be included to provide a more conprehensive understanding.

Overall, this paper provides limited contribution compared to prior works. The method is simple and based on hypotheses and heuristics. The result is not surprising, as prior work has discussed the phenomenon that attention layers are more redundant than MLP layers.

### Questions
Same as above.

### Soundness
3

### Presentation
2

### Contribution
1
