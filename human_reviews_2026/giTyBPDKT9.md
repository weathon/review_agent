# LLMBoost: Make Large Language Models Stronger with Boosting

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Ensemble learning of LLMs has emerged as a promising alternative to enhance performance, but existing approaches typically treat models as black boxes, combining the inputs or final outputs while overlooking the rich internal representations and interactions across models.In this work, we introduce LLMBoost, a novel ensemble fine-tuning framework that breaks this barrier by explicitly leveraging intermediate states of LLMs. Inspired by the boosting paradigm, LLMBoost incorporates three key innovations. First, a cross-model attention mechanism enables successor models to access and fuse hidden states from predecessors, facilitating hierarchical error correction and knowledge transfer. Second, a chain training paradigm progressively fine-tunes connected models with an error-suppression objective, ensuring that each model rectifies the mispredictions of its predecessor with minimal additional computation. Third, a near-parallel inference paradigm design pipelines hidden states across models layer by layer, achieving inference efficiency approaching single-model decoding. We further establish the theoretical foundations of LLMBoost, proving that sequential integration guarantees monotonic improvements under bounded correction assumptions. Extensive experiments on commonsense reasoning and arithmetic reasoning tasks demonstrate that LLMBoost consistently boosts accuracy while reducing inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- Proposes **LLMBOOST**, a boosting-style *internal* ensemble: later models attend to earlier models’ hidden states via residual cross-model attention; training uses an error-suppression loss to down-weight a predecessor’s top wrong logit while up-weighting the correct one; inference uses near-parallel pipelining and top-k backward fusion of logits.
- Provides a **theoretical monotonic-improvement** result under a “bounded correction” assumption and a suitable scale $\lambda $.
- On commonsense/arithmetic benchmarks and a Toolchain dataset, reports consistent but modest gains vs VOTE/UNITE/T-copilot; latency improved 47% vs the *sequential* variant (not vs external baselines).

### Strengths
- **Idea:** Ensembling *inside* the network (state sharing) instead of only outputs is interesting.
- **Design details:** Clear components (cross-model attention, error-token forwarding, top-k backward fusion) with concrete formulas and a training algorithm.
- **Theory:** Gives a clean MSE-based guarantee (with assumptions) and interpretable role of the scaling $ \lambda $.
- **Empirics:** Broad tasks, multiple sizes (3B–8B, Qwen/Llama); monotonic gains as $n$ increases (diminishing returns beyond $n=3$).

### Weaknesses
- **Representation compatibility & alignment.** Requires homogeneous model families (matching layer shapes/hidden formats); no cross-family alignment is provided. Current scope does not address heterogeneous LLMs—limiting generality.
- **Efficiency & complexity.** Training/inference require multiple full LLMs with state exchange; results compare sequential vs near-parallel *within* LLMBOOST, but provide no wall-clock/token-cost comparison vs external baselines. Memory/throughput implications (forward+backward across models) are not quantified.
- **Theory scope.** The guarantee hinges on assumptions (successor residual fit; choice of $\lambda $, is MSE-centric, and is not linked to accuracy/error rates; no diagnostics show when assumptions fail.
- **Baseline fairness at equal “effective capacity.”** LLMBOOST fine-tunes multiple models and infers with all of them; comparisons vs a single larger model (≈ total params or total trainable adapters) are missing.

### Questions
1. **Representation alignment.** Since the method is currently inapplicable to heterogeneous families, did you try alignment adapters (e.g., per-layer projections)? If not, can you report a small Qwen-Llama cross-family study with such adapters?
2. **Efficiency/cost.** Please report end-to-end wall-clock, peak memory, and token cost versus UNITE/VOTE/T-copilot, not just sequential vs near-parallel within LLMBOOST. Include per-GPU memory for training $n$ models with LoRA and the overhead of state passing.
3. **Baseline parity by capacity.** Add a comparison to a larger single model (or a single model with equivalent total LoRA params) trained under the same data/budget to address “effective parameter” fairness.
4. **Theory diagnostics.** Can you measure the residual-fit assumption empirically (e.g., residual MSE across tokens/steps) and relate choices of $\lambda$ to accuracy/robustness (not just MSE)?
5. **Cross-family diversity.** Any compatibility analyses (layer-wise stat shifts, Canonical Correlation Analysis of hidden states) explaining why families clash and guiding adapter design? Current experiments only compose same-family models.

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
3

### Summary
This work introduces a boosting-style framework to ensemble LLM models. By leveraging a cascaded parallel structure, their framework enables successor models to benefit from the hidden states of previous models, supporting hierarchical error correction. With a near-parallel inference mechanism it allows the hidden states to stream across models, reducing latency. The idea in this paper is to benefit from a boosting ensemble of networks. Evaluation benchmarks include commonsense and arithmetic reasoning, and tool-chain.

### Strengths
1. The idea of boosting by incorporating previous model correction seems to be relatively novel in the realm of LLMs with cross-model attention fusion. However, I have some concerns about the evaluation discussed in W1.

2. The results indicate that the ensembled model improves performance over the baseline approach by a decent margin on commonsense reasoning and arithmetic reasoning benchmarks.

### Weaknesses
1. Unfair capacity comparison. The ensembles (e.g., 3×8B = 24B total parameters) are not compared against a single 24B model or an equivalently fast system (perhaps a 16B model is faster than the 3-model ensemble with a better performance?), so the true efficiency–accuracy trade-off is unclear. I recommend that the authors include such results and clearly explain the setting for comparison.

2. The evaluated benchmarks in this work include arithmetic and commonsense reasoning. In order to really test the work's applicability, the authors should consider using longer sequence/context benchmarks (see q2).

3. The method currently relies on identical architectures (e.g., all LLaMA-3), limiting the formation of heterogeneous ensembles. Since the paper frames the approach as a self-improving mechanism, it would be valuable to explore whether diverse models -- with domain-specific expertise -- can be combined to enhance task- or domain-adaptive performance.

### Questions
1. Please refer to Weakness 1 and provide some additional comparisons/clarity on the evaluation.

2. Most evaluated tasks focus on short output sequences, such as commonsense and arithmetic reasoning benchmarks. I recommend that the authors include some more complex, longer context length/output length tasks, such as long-form, multi-turn dialogue datasets like MT-Bench and code generation (HumanEval), to better demonstrate the effectiveness in longer contexts.

3. Including evaluations on the GLUE benchmark would better measure general language understanding and demonstrate the method's transferability to broader NLP tasks.

4. Exploring the heterogeneous architectures as discussed in Weakness 3.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes LLMBoost, a boosting-inspired ensemble framework that leverages the internal hidden states of large language models rather than treating models as black boxes that only mix inputs or final outputs. It connects multiple models so that later models can directly attend to and fuse the intermediate representations of earlier ones, enabling hierarchical error correction and knowledge transfer. The authors introduce three components: a cross-model attention mechanism for accessing predecessor hidden states; a chain training procedure that progressively fine-tunes each model with an error-suppression objective to correct its predecessor’s mistakes with minimal extra computation; and a near-parallel inference pipeline that passes hidden states layer by layer across models to achieve efficiency close to single-model decoding.

### Strengths
1. The paper is well-written and easy to follow.
2. The idea of chain training paradigm is novel, interesting and straightforward.

### Weaknesses
1. Given that LLMBoost significantly increases computational cost and GPU memory usage during both training and inference, the performance gains appear relatively modest.
2. LLMBoost introduces a range of hyperparameters, which makes training and inference more complex, and for some of them (e.g., $\beta$ in Equation 6), the paper does not discuss how to set them.
3. The Error-Suppression Objective seems a bit strange to me.  It maximizes the probability margin between the ground-truth token and other tokens, yet a token that differs from the ground truth is not necessarily incorrect. For example, if the ground-truth token is "hello" and the model predicts "hi", aggressively suppressing the probability of "hi" may undermine the model’s ability to generalize.

### Questions
1. Given the multi-fold increase in GPU memory during both training and inference with LLMBoost, I believe it should be compared against a larger single model with a similar memory budget. For example, LLMBoost 2×7B vs. Qwen2.5-14B.
2. During backpropagation, does Cross-Model Attention propagate gradients to the preceding model?
3. Given that the models later in the training pipeline correct more error tokens, do they achieve higher accuracy than the earlier models? During inference, why not assign greater weight to the logits of the later models?
4. In Table 1, for configurations like 2×8B or 2×3B, how is VOTE implemented?
5. A closing square bracket is missing in Equation 6.
6. Some of the notation in Figure 2 appears to be incorrect. For example, $L_l^2$ should be $L_l^1$.

### Soundness
2

### Presentation
2

### Contribution
2
