# GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Fine-tuning Large Language Models (LLMs) with downstream data is often considered time-consuming and expensive. Structured pruning methods are primarily employed to improve the inference efficiency of pre-trained models. Meanwhile, they often require additional time and memory for training, knowledge distillation, structure search, and other strategies, making efficient model fine-tuning challenging to achieve. To simultaneously enhance the training and inference efficiency of downstream task fine-tuning, we introduce GradPruner, which can prune layers of LLMs guided by gradients in the early stages of fine-tuning. GradPruner uses the cumulative gradients of each parameter during the initial phase of fine-tuning to compute the Initial Gradient Information Accumulation Matrix (IGIA-Matrix) to assess the importance of layers and perform pruning. We sparsify the pruned layers based on the IGIA-Matrix and merge them with the remaining layers. Only elements with the same sign are merged to reduce interference from sign variations. We conducted extensive experiments on two LLMs across eight well-known datasets in downstream tasks. Including medical, financial, and general benchmark tasks. The results demonstrate that GradPruner has achieved a parameter reduction of 40% with only a 0.99% decrease in accuracy. Our code is available at https://anonymous.4open.science/r/LLM-GradPrune-436D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
GradPruner uses the IGIA-Matrix computed from the early stages of fine-tuning (within the first 1% of steps) to evaluate the importance of parameters and layers. Layers with low importance scores are pruned, and to further retain performance under high pruning rates, the parameters of pruned layers are merged into the retained layer using a sign-based merging rule. Experiments on two LLMs across eight datasets show that GradPruner achieves approximately 40% parameter reduction with less than 1% average accuracy drop, while also significantly reducing fine-tuning time, inference latency, and memory consumption compared to representative pruning baselines.

### Strengths
* GradPruner jointly improves both training and inference efficiency. Unlike many pruning approaches that focus only on inference speed-ups, GradPruner is explicitly designed to reduce fine-tuning time and memory consumption as well. I believe this aligns well with the practical need for LLMs to quickly adapt to new downstream tasks.
* GradPruner is simple to understand yet conceptually novel. The paper introduces a new gradient-based importance metric computed from less than 1% of the early fine-tuning steps, which effectively reduces the computational cost of pruning. In addition, it seems like the proposed pruning strategy is generally applicable to all transformer-based architectures.
* The experiments are comprehensive. Evaluations are conducted on two representative LLMs across eight datasets, which enhances the reliability of the results. The paper also provides detailed ablation studies that clearly demonstrate the contribution of each component.

### Weaknesses
* Limited theoretical grounding. Lines 201–208 simulate the gradient of W via a matrix multiplication, yet the rationale for this simulation is not theoretically justified. In addition, the sign-based merging rule in Equation (5) is presented as a heuristic with little theoretical explanation. Clearer derivations would strengthen the contribution.
* Concern over the stability of early-step gradients. The method relies on gradient statistics collected within the first 1% of fine-tuning steps to estimate layer importance, yet such early gradients may not always provide a stable or reliable signal. For example, in small or noisy datasets, gradient variance can be high, potentially causing the IGIA-Matrix to mis-rank layer importance and resulting in suboptimal pruning decisions.
* Incomplete efficiency metrics. The paper aims to make layer pruning more efficient and reports average time and parameter reduction. Including FLOPs would provide a more standardized and hardware-agnostic measure of computational efficiency.
* Clarity of the method description could be improved. The paper does not explicitly detail how each transformer sub-module is handled during merging. It appears that each linear sub-module of a pruned layer is merged into the corresponding sub-module of the preceding retained layer, but this should be stated unambiguously. Moreover, Equation (5), which specifies merging based on sign agreement, is difficult to parse. A more thorough explanation and a small illustrative example would aid the reader's understanding.

### Questions
* Can you clarify the exact computation and shape alignment for simulating the gradient of W via LoRA gradients?
* Why is raw summation chosen for IGIA-matrix aggregation across all linear layers, rather than normalization (e.g., mean or norm per parameter or per layer size)? 
* Could the authors elaborate on the layer merging procedure, perhaps with an example?
* Can the approach still be effective with even fewer gradient accumulation steps (e.g., less than 0.02% steps)？

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GradPruner, a gradient-guided structured pruning framework designed to enhance both fine-tuning and inference efficiency for large language models (LLMs). Unlike most existing structured pruning methods that rely on calibration data or additional distillation steps, GradPruner leverages initial gradient information obtained during the first few LoRA fine-tuning steps to estimate parameter and layer importance.

The key innovation lies in computing an Initial Gradient Information Accumulation Matrix (IGIA-Matrix) to quantify layer importance early in training. Based on this metric, GradPruner performs layer-level pruning followed by a layer merging step that sparsifies and merges pruned layers into adjacent retained layers while resolving sign conflicts to reduce destructive interference.

Empirically, GradPruner achieves a 40% reduction in parameters with only 0.99% loss in downstream accuracy, tested on two LLMs (LLaMA3.1-8B and Mistral-7B) across eight diverse benchmarks spanning medical, financial, and general reasoning tasks. It shows consistent performance improvements over strong structured pruning baselines (LLM-Pruner, LaCo, SAT, APT, and MINITRON), and reduces both training and inference time/memory by over 35%.

### Strengths
1. The use of IGIA-Matrix computed from <1% of training steps is original and empirically justified by gradient sensitivity analysis.


2. The proposed sign-consistent merging technique effectively preserves accuracy even under 40% pruning.


3. The authors test across multiple domains, model sizes, and fine-tuning regimes with strong baselines, demonstrating robustness.


4. Substantial reductions in both training and inference costs (~35–40%) while maintaining accuracy are practically valuable.


5. The paper analyzes pruning ratios, merging counts, and alternatives such as kernel pruning and weighted averaging.

### Weaknesses
1. While the empirical gradient correlation study is convincing, the paper lacks a deeper theoretical analysis of why early gradient accumulation correlates with long-term importance, beyond empirical observation.


2. The method assumes access to LoRA gradients and may not generalize to non-LoRA or adapter-free fine-tuning setups.


3. The layer-importance estimation could behave differently for tasks with varying gradient noise; this is not fully explored.

### Questions
1. How sensitive is GradPruner’s performance to the number of initial fine-tuning steps ttt?


2. Have the authors tested GradPruner when using other adapter methods (e.g., QLoRA, DoRA) to verify whether IGIA-Matrix remains stable?


3. In the merging phase, how is the sparsity rate ppp selected? Could adaptive sparsity (learned from IGIA statistics) yield further improvements?


4. Does GradPruner preserve the same inference graph (number of layers) after merging, or does merging affect sequence length/runtime at deployment?


5. Can GradPruner be integrated with post-training quantization or low-rank compression? If so, how does the gradient-based importance interact with quantization noise?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes **GradPruner**, a lora-based gradient-guided **layer pruning + sign-based layer merging** method for efficient fine-tuning and inference. The method (i) computes an **Initial Gradient Information Accumulation (IGIA)** matrix from early-step LoRA gradients to score parameter/layer importance, (ii) prunes layers with low summed IGIA (over linear sublayers), and (iii) **sparsify-then-merge** pruned layers into the preceding retained layer using a **sign-consistency rule**. Experiments on two LLMs (Llama-3.1-8B, Mistral-7B) and eight datasets report ~**40% sparsity** with ~**0.99%** average degradation; the headline comparison (Table 1) is at **40% sparsity**.

### Strengths
- **Simple, pragmatic pipeline.** Early-step gradient accumulation → IGIA-based layer scoring (sum over linear sublayers) → pruning → sign-based merging. The pruning score is clearly defined.
- **Operationally clear merging.** “Top-p% by IGIA then sign-consistent addition into the preceding kept layer” is straightforward to implement and shown with a framework figure.
- **Broad task coverage with efficiency reporting.** Ablations include number of merged layers and sparsity-rate sweeps for the proposed method.

### Weaknesses
### (A) Insufficient theoretical grounding for **merging** (most important)
- **Self-inconsistency between pruning and merging.** The paper emphasizes that **layers differ in importance** and uses **IGIA** to make importance-aware pruning decisions. However, during **merging**, contributions from pruned layers are **added with equal weight** whenever signs match—**without any sensitivity weighting** (e.g., IGIA- or Fisher-based) for either donor or receiver layers. This disconnect undermines the rationale that sensitivity should matter.
- **Cross-layer addition lacks justification.** There is no theoretical argument that **elementwise addition across different Transformer blocks** preserves function or yields bounded error, even after sparsification. A first-order approximation, Hessian/Fisher weighting, or Lipschitz-based stability discussion is missing.
- **Actionable request.** Provide a clear **merging objective** (e.g., minimize a first-order loss surrogate), an **error bound**, or at least **IGIA-weighted** or **Fisher-weighted** merges. Otherwise, the method uses importance for pruning but not for merging.

### (B) Comparisons are narrow in **pruning ratio** and **model scale**
- **Single-ratio comparisons.** The main table compares methods **only at 40% sparsity**. While internal ablations vary sparsity for GradPruner, there is **no multi-ratio cross-method** comparison to show whether the advantage holds when pruning is milder or more aggressive.
- **Limited model scaling.** Results center on **7–8B** models (plus a 3B FT reference). It is unclear whether gains **transfer down** to ~1–2B or **up** to ~14B+ models.
- **Actionable request.** Report **accuracy–sparsity curves** for **multiple baselines**, and include **smaller (≈1.7B)** and **larger (≈14B)** models.

### (C) Training recipe & domain generalization
- The domain-specific fine-tuning often appears to train and test within the **same dataset**. This setup makes it hard to assess **out-of-distribution** robustness within the domain.
- **Actionable request.** Include **cross-dataset** evaluations per domain (e.g., train on one medical QA dataset, test on another; for math-style settings: train on MetaMathQA-40k, evaluate on GSM8K / GSM-Plus) to demonstrate generalization beyond the training distribution.

### (D) Minor Issues / Clarity
- **Equation (2) ambiguity/typo.** The equation seems to multiply **the B-gradient twice**, which is likely a typo and inconsistent with LoRA structure. Please clarify the intended mapping and dimensional compatibility.
- **Citation formatting.** Around **line ~310**, the first paragraph’s citation is not enclosed in parentheses, unlike others; please standardize the style.
- **Small typos.** A few truncated terms (e.g., “IGIA-Matri”) and punctuation/spacing glitches around the merging equation.

### Questions
please see weakness above.

### Soundness
2

### Presentation
1

### Contribution
3
