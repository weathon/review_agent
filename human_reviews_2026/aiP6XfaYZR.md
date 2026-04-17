# What Layers When: Learning to Skip Compute in LLMs with Residual Gates

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
We introduce GateSkip, a simple residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch is equipped with a sigmoid-linear gate that compresses the branch’s output before it re-enters the residual stream. During inference we rank tokens by the gate and skip low-importance ones using a per-layer budget. While early-exit or router-based Mixture-of-Depths models are known to be unstable and need extensive retraining, our smooth, differentiable gates fine-tune stably on top of pretrained models. On long-form reasoning, we save up to 15% compute while retaining >90% of baseline accuracy. On instruction-tuned models we see accuracy gains at full compute and match baseline quality near 50% savings. The learned gates give insight into transformer information flow (e.g., BOS tokens act as anchors), and the method combines easily with quantization, pruning, and self-speculative decoding.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes GateSkip, a lightweight, differentiable gating mechanism added to the residual stream of decoder-only LMs.
The gate module is a linear+sigmoid function that multiplies the output after each attention and MLP branch. Gates are trained with an L2 penalty on top of a pretrained model. At inference, token-wise gate scores are converted to per-layer quantile thresholds to keep a target budget of tokens and skip the rest. Extensive experiments on various benchmarks, previous methods, model sizes, and ablations are conducted.

### Strengths
- The paper is well written. The introduction section clearly states the research problem, issues in current methods, and the contributions.
- The method is simple and effective.
- The experiments are solid. I particularly like the ablation results in Table 5.

### Weaknesses
- The viewpoint “early layers allocate more computation to the beginning-of-sequence token and punctuation, while deeper layers become increasingly selective and focus on content-bearing words” may be overclaimed based on Figures 2 and 4.
- The overhead of HxH gates per branch, including both parameters and computation, may not be negligible.

### Questions
- Did you try non-uniform per-layer budgets (b_l), such as allocating more compute at mid-depth?
- There are many interesting results in the experiments. Could you offer some explanations for these results?
	- The difference between generative and log-likelihood benchmarks: previous methods show large degradation on generative benchmarks, while even random skipping is not too bad on log-likelihood benchmarks (Tables 1–3).
	- GateSkip achieves better performance as saved compute increases in the PIQA-Gen task (Table 4).
- If I skip some tokens at the first layer, it seems that I have no KV cache to copy from. How do you handle this issue?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
GateSkip is a new residual-stream gating method for decoder-only LLMs that enables token-wise layer skipping during inference. It uses smooth, differentiable gates fine-tuned on pretrained models to decide token importance, avoiding the instability of earlier adaptive compute methods. Tests on Llama and Gemma show 15–20% compute savings with over 90% accuracy retention and compatibility with techniques like quantization and pruning. Gate analysis also reveals higher importance for BOS tokens and punctuation.

### Strengths
1. The proposed method achieves a strong compute–accuracy trade-off, reducing computation by up to 15% while maintaining over 90% accuracy.
2. It is compatible with other efficiency techniques, further enhancing its practical value.
3. The authors conducted comprehensive ablation studies to validate their design choices.

### Weaknesses
1. It seems the method is sensitive to architecture choice. The placement of the gate is important. I'd prefer more discussion on why this is the case. 
2. Similarly, this method also is sensitive to the choice of the gate. I wonder, besides MLP, whether the authors have tried other projection methods.  


Presentation:
1. I am not sure why there is an empty line after two paragraphs in the Related Work part (e.g., Line 95~97, 104~105).
2. According to ICLR format for tables, table captions should come before the table. 
3. Line 472: the reference to the appendix broke.

### Questions
1. What is the intuition behind this design?
2. In the Related Work part, the authors mention that layer pruning, token pruning, etc., operate on different axes of efficiency from your work, and thus those are orthogonal. Could you elaborate more on what specifically those axes are and why they are different from yours? It seems your method also helps for layer skipping in the sense of the attention layer or MLP layer.

### Soundness
3

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
The paper introduces GateSkip, a residual-stream gating mechanism that enables token-wise layer skipping in decoder-only LMs. Each Attention/MLP branch outputs is multiplied by a sigmoid(linear) gate; gates are trained jointly with the backbone using LM loss + sparsity penalty, and at inference the per-token gate scores are quantile-thresholded to enforce a layer-wise compute budget (skipped tokens copy hidden states and KV cache upward). 

On Llama/Gemma models, GateSkip claims up to 15–20% compute savings while retaining >90% of baseline accuracy on long-form reasoning, and near-50% savings with matched accuracy on instruction-tuned models; it also composes with quantization, pruning, and self-speculative decoding. Methods, hyperparameters, and seeds are documented (with single-seed runs and bootstrap variance).

### Strengths
### Originality

* Reframes adaptive depth via residual gating (no discrete router during training), then converts scores to hard skips at test time; per-token budgets encourage fine-grained allocation.

### Quality

* Clear mechanism (Eq. 2), objective (Eq. 3–5), token selection (Eq. 6), init, and overhead; ablations cover gate shape, sharing, placement, and skipping granularity.
* Side-by-side comparisons against MoD, CALM, FREE, and static baselines under identical protocols.

### Clarity

* Figures/tables report compute budgets and (stated) seed counts; the experimental setup is centralized in §4.1 with dataset/model details.

### Significance

* Achieves compute–accuracy trade-offs on generative tasks where prior methods degrade; composes with 4-bit quantization and self-speculative decoding.

### Weaknesses
1. Most results are single-seed with bootstrap CIs; for stability claims this is thin. Please add ≥5 seeds for key curves and show error bars.

2. Backbone fine-tuning confounds “accuracy gains at 0%.” The setup jointly fine-tunes the backbone while training gates, so reported accuracy lifts at 0% compute savings likely reflect backbone adaptation. 

3. End-to-end speed is modest/non-monotonic. The latency/throughput table shows limited and non-monotonic gains vs. skip rate; please report FLOPs vs. wall-clock and break out gate overhead. A kernel-optimized implementation (vLLM/custom CUDA) would also strengthen real-world evidence.

4. Positioning vs. strongest baselines. MoD is included, but add a router-tuned MoD on the same frozen backbone and compare stability (loss curves, LR sensitivity) to substantiate the “stable training” claim.

### Questions
1. Isolating gate effect. Please report frozen-backbone results and an equally fine-tuned baseline (no gates) to separate backbone tuning gains from gating.

2. Seed count / CIs. For the main curves (Tables 1–3), can you re-run with ≥5 seeds and include 95% CIs?

3. Wall-clock accounting. What is the per-layer gate FLOPs/latency? Please add a column breaking down FLOPs saved vs. FLOPs added and relate this to Table 10’s latency/throughput.

4. Router baseline stability. Under your setup, how does router-tuned MoD behave (loss curves / collapse rate) vs. GateSkip? A stability plot would support the abstract’s claim.

5. Gating loss + budgets. Why L2 on gate activations (Eq. 4) vs L1/KL? Sensitivity to λS and to the budget decay schedule?

6. Where does it help most? Any breakdown of skips by layer/token type on reasoning tasks (build on your BOS/punctuation analysis) to guide deployments?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GateSkip, a method for enabling token-wise layer skipping in decoder-only transformer language models. GateSkip adds a small linear gate with sigmoid activation to each attention and MLP branch in the transformer. During training, the gates are optimized to remain sparse while preserving language modeling accuracy. At inference time, the gates produce token-level importance scores, allowing the model to skip the least important tokens in each layer. Skipped tokens have their hidden states and key-value cache entries copied upward.

GateSkip is evaluated on Llama and Gemma models and find that it reduces computation by up to 15% on reasoning tasks while retaining over 90% of the original accuracy. On instruction-tuned models, GateSkip improves accuracy even at full compute and matches baseline quality with around 50% compute savings. Analyzing the learned gate values reveals that early layers allocate more compute to the BOS token and punctuation, while deeper layers focus compute on content words.

### Strengths
* **Novel approach**: The paper introduces a novel approach to layer skipping, GateSkip, which is different from existing methods such as Mixture-of-Depths (MoD) and early-exit methods.

* **Smooth and differentiable gates**: The use of smooth and differentiable gates allows for stable training and fine-tuning on top of pre-trained models, which is a significant advantage over existing methods that rely on hard, discrete decisions.

* **Token-level control**: GateSkip provides fine-grained control at both the token and module level, enabling nuanced allocation of compute resources.

* **Compatibility with other efficiency techniques**: The paper shows that GateSkip can be combined seamlessly with other efficiency techniques such as quantization, pruning, and self-speculative decoding.

* **Strong experimental results**: The paper presents strong experimental results, demonstrating that GateSkip can reduce computation by up to 15% while retaining over 90% of the original accuracy on reasoning tasks.

* **Insights into transformer information flow**: The analysis of learned gate values provides interesting insights into the information flow within transformers, such as the allocation of compute resources to different tokens and layers.

### Weaknesses
* **Efficiency**:  Provide more context on the setting for the benchmarking. For the throughput reported numbers in line 447. Do you increase the batch size as you increase the number of tokens skipped ?  How does this method affect the peak memory utilization compared to the baseline when there is no skipping ? Is this for pre-fill or decode ? what is the total number of tokens etc.

* **Analysis**: For the most part I like the analysis in the paper. I would suggest to maybe emphasize some other things more e.g why do we have significantly better performance for this method when using it on instruction tunede models ? Am I right to tell from Figure 1 C) that the performance of GateSkip-3b-Instruct is almost 50% at 0% skipped while Llama-3b-Instruct is only about 35% ? This is a huge difference in quality. What accounts for this ?

* **Writing and Clarity**: The work could benefit from some more polishing and make the work more clear e.g on Line 472: Appendix ??

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
