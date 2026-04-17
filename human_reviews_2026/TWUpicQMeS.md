# FlowMM: Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Traditional KV cache eviction strategies, which discard less critical KV-pairs based on attention scores, often degrade generation quality, causing context loss or hallucinations. Recent efforts shift toward KV merging, merging eviction tokens with retention tokens based on similarity. However, in multimodal scenarios, distributional biases across modality tokens and attentional biases in cross-modal interactions limit its effectiveness. This work introduces FlowMM, an adaptive framework for cross-modal information flow-guided multimodal KV cache merging. FlowMM leverages cross-modal information flow to dynamically apply layer-specific merging strategies, capturing modality-specific patterns while preserving contextual integrity. Furthermore, we introduce a sensitivity-adaptive token matching mechanism that jointly evaluates token similarity and task-critical sensitivity, merging low-risk tokens while safeguarding high-sensitivity ones. Extensive experiments across diverse leading MLLMs show that FlowMM reduces KV cache memory by 80% to 95% and decoding latency by 1.3-1.8×, while maintaining competitive task performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes FlowMM, a framework for cross-modal information flow-guided KV cache merging in multimodal large language models (MLLMs).Instead of using fixed or unimodal cache-compression strategies, FlowMM dynamically analyzes layer-wise cross-modal attention flow and applies layer-specific merging accordingly.It also introduces a sensitivity-adaptive token matching mechanism to preserve task-critical tokens during merging.Experiments on several MLLMs (Qwen2.5-VL, InternVL-2.5, MobileVLM-V2) show that FlowMM can reduce KV cache memory by 80–95% and improve decoding speed by 1.3–1.8×, while maintaining comparable task accuracy.

### Strengths
1. The paper addresses an important and practical problem—efficient inference for MLLMs—where KV cache compression is indeed a bottleneck.
2. The idea of using cross-modal attention flow to guide layer-specific merging is novel and intuitively motivated by observed modality interactions.
3. The design of sensitivity-adaptive token matching is well-motivated and ablation results show its necessity.
4. Experiments cover multiple open-source MLLMs and standard multimodal benchmarks, with clear reporting of latency and memory reduction.
5. Writing and presentation are clear, and the diagrams (e.g., Fig. 1–2) effectively illustrate the differences among methods.

### Weaknesses
1. Limited novelty – While the idea of using information flow to guide merging is novel in formulation, the method builds upon existing merging frameworks (e.g., CaM, LOOK-M) and mainly extends them heuristically to multimodal settings.
2. Need for deeper analysis – The cross-modal interaction ratio and token-sensitivity metrics are empirically defined. The paper lacks theoretical justification or sensitivity analysis explaining why these metrics generalize well across modalities.
3. Missing qualitative analysis – There is no visualization of how information flow changes across layers or examples showing merged vs. unmerged attention distributions. Visualizations of the attention-flow evolution or token-sensitivity distribution would help verify the interpretability claims.
4. Reproducibility concerns. Details such as implementation overhead, merging frequency, or computational cost of attention-flow measurement are not sufficiently discussed.

### Questions
1. How often is the cross-modal flow analysis performed—once per layer during inference, or pre-computed offline? What is the computational overhead compared to full caching?
2. Could the authors visualize how cross-modal attention patterns evolve across layers, and how merging affects them?
3. How sensitive is FlowMM to the chosen merging threshold θ and sensitivity τ across different datasets?
4. Can FlowMM be extended to video-language or audio-language MLLMs with longer temporal context?
5. Would combining FlowMM with existing quantization or pruning techniques (e.g., GEAR, KIVI) lead to additive benefits?

### Soundness
3

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
5

### Summary
The paper proposes FlowMM, a framework for efficient multimodal context inference that reduces the memory footprint of key value caches during decoding. Instead of evicting tokens purely by low attention scores, FlowMM merges tokens in a way that is guided by estimated cross modal information flow. The method adapts merging strategies per layer to capture modality specific patterns and introduces a sensitivity adaptive matching mechanism that considers both token similarity and task sensitivity. Experiments on several modern multimodal large language models report very large KV memory reductions between eighty and ninety five percent and speedups of about one point three to one point eight times while keeping performance competitive.

### Strengths
1. The idea of steering KV merging using estimated cross modal information flow and layer specific policies is a clear conceptual step beyond attention score based eviction or naive similarity based merging.

2. The framing of sensitivity adaptive matching acknowledges that some tokens are risky to merge even if similar, which is a practical insight for long context multimodal prompts where a few critical tokens can dominate correctness.

3. The abstract clearly separates the components of the proposal which are the information flow estimator, layer wise merging policy, and sensitivity aware matching. The motivation is also clearly stated in terms of biases in multimodal attention distributions.

4.  If the reported memory reduction and speedup numbers hold across strong baselines and varied tasks, the impact on serving cost for multimodal assistants could be substantial

### Weaknesses
1. The abstract does not explain how cross modal information flow is computed. Without a well justified estimator and ablations that vary it, reviewers cannot assess whether gains come from the estimator or from generic similarity based merging.
2. The claim of maintaining competitive task performance is broad. It is important to see results across many task types such as OCR heavy tasks, visual reasoning that requires spatial grounding, audio text alignment if applicable, and multi turn dialogues. Single number averages can hide regressions on sensitive tasks.
3. Information flow estimation and sensitivity scoring likely add compute and memory. The paper must quantify this overhead and show wall clock latency and throughput at realistic batch sizes and sequence lengths, not only per token speed.
4. KV caching has a rapidly evolving literature including eviction, merging, and distillation strategies. The contribution will feel weaker if baselines do not include strong recent multimodal aware merging methods, reranking based eviction, and oracle ablations such as perfect sensitivity masks.
5. Aggressive merging can cause context loss and hallucinations that are subtle. The paper should include targeted stress tests for hallucination under long distractor contexts and for modality imbalance such as many more image tokens than text tokens.

### Questions
1. How exactly is cross modal information flow defined and computed?  Is it derived from attention rollout, gradient based attribution, or learned predictors. What training signal is used and is it model specific or model agnostic.

2. What is the computational overhead of estimating information flow at inference time? Can it be amortized across tokens or cached per layer.
3. How is task sensitivity quantified. Is it based on entropy, calibration error, token type heuristics, or a learned risk model? How sensitive are results to the sensitivity threshold?
4. How does FlowMM behave under distribution shift such as longer than trained contexts, images with dense text, or dialogues that interleave many images and text turns?
5. What happens when modalities are more than two such as text, image, and audio. Is the merging policy still layer specific but modality agnostic, or does it require pairwise flow modeling that scales quadratically?

### Soundness
3

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
2

### Summary
This paper presents FlowMM, a novel and timely method for compressing the Key-Value (KV) cache in Multimodal Large Language Models (MLLMs). The core insight is that existing KV cache compression techniques, designed for unimodal text, are suboptimal for MLLMs due to the distinct distribution and interaction patterns of visual and textual tokens across different transformer layers. FlowMM addresses this by (1) dynamically analyzing cross-modal attention flow to apply layer-specific merging strategies and (2) incorporating a sensitivity-adaptive token matching mechanism to preserve task-critical information. The experiments are extensive, demonstrating significant reductions in memory (80-95%) and latency (1.3-1.8x) while maintaining performance close to a full cache, outperforming strong baselines.

### Strengths
1. Motivation and Problem Identification: The paper effectively highlights the limitations of existing KV cache eviction and merging strategies in multimodal contexts, pinpointing "distributional biases" and "attentional biases" as key challenges. This sets a clear and compelling stage for the proposed work.
2. Core Idea: The concept of using cross-modal information flow to guide layer-specific merging is the paper's most significant contribution. The observation that shallow layers are intra-modal and deeper layers are cross-modal is well-supported by preliminary analysis (Figure 3) and provides a principled foundation for the method.
3. Effective Ablation Studies: The ablations successfully validate the importance of both core components: the information flow guidance and the sensitivity-adaptive matching. The results clearly show that removing either component leads to a notable performance drop.
4. Practicality: The fact that FlowMM is a plug-and-play method requiring no fine-tuning is a major practical advantage for easy adoption.

### Weaknesses
1. Clarity of the Merging Operation:
The paper clearly defines how to decide when and what to merge (using ρ^l and sensitivity). However, the exact mechanism of how the merging is performed (the functions f_merge and g_merge in Eq. 5) is somewhat glossed over. A more detailed explanation or a reference to the specific merging function (e.g., weighted averaging based on attention) would be helpful.

2. Definition and Calculation of Sensitivity:
The proposal to use attention scores as a proxy for sensitivity is practical but could be better motivated. The claim that it is a "near-zero-overhead approximation" is valid, but a small experiment or citation showing the correlation between a token's attention score and the performance drop when it is merged would strengthen this design choice.
The relationship between the sensitivity score I (Eq. 8) and the threshold τ (Eq. 10) could be explained more clearly. How is τ chosen? Is it a hyperparameter or dynamically set?

3. Hyperparameter Sensitivity:
The paper identifies an optimal range for the threshold θ (0.2-0.3) via ablation. It would be useful to discuss the robustness of FlowMM to this parameter. How sensitive is the performance to small changes in θ outside this range? Is this value consistent across different models?

### Questions
1. Merging Mechanism: Could you please elaborate on the specific operation used to merge the key and value vectors of two tokens? For example, is it a simple average, a weighted average based on attention or sensitivity scores, or a more complex function?

2. Computational Overhead: While the memory and latency benefits are clear, what is the computational overhead introduced by calculating the cross-modal ratio ρ^l, the token importance I, and the cosine similarity matrix during inference? A brief analysis or discussion of this trade-off would be valuable.

3. Sensitivity Threshold τ: How is the sensitivity threshold τ in Equation (10) determined? Is it a fixed value, or is it adapted based on the distribution of sensitivity scores in the current context?

4. Generalizability: The method relies on analyzing attention patterns. How would FlowMM perform with MLLM architectures that use significantly different attention mechanisms (e.g., sparse attention or other efficiency-focused variants)?

### Soundness
2

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
3

### Summary
This paper proposes FlowMM, a KV cache compression framework for multimodal LLMs that uses cross-modal information flow to guide layer-specific merging strategies. FlowMM measures cross-modal attention ratios per layer and applies intra-modal merging in shallow layers and cross-modal merging in deep layers based on a threshold. It also incorporates sensitivity-adaptive token matching to preserve high-importance tokens. Evaluated on MileBench, FlowMM achieves substantial KV cache memory reduction and decoding speedup while maintaining competitive performance.

### Strengths
- The exposition is clear and the problem is well-motivated, with the method inspired from known insights in multimodal interpretability literature.
- Evaluation on diverse long-context tasks showing strong improvements over baselines in the reported experimental setting.
- Solid ablation studies validating both major components contribute to performance.

### Weaknesses
- Critical concern: LOOK-M baseline performance appears severely degraded compared to original paper. LOOK-M's original paper reports near-lossless performance at similar compression ratios, often matching or exceeding full cache. FlowMM reports massive degradations for LOOK-M at the same settings. This raises serious questions about baseline implementation quality versus genuine failure to generalize to newer architectures. The authors must provide comparison on original LOOK-M models (LLaVA-v1.5, MobileVLM-v2, InternVL-v1.5) and tasks to validate whether gains stem from methodological improvements or suboptimal baseline implementation.
- Narrow evaluation scope limited exclusively to MileBench long-context tasks. Generalizability to standard multimodal benchmarks like VQA, image captioning, or visual reasoning and models larger than 7B (ex. Qwen VL 32B) remains unexplored.
- Limited novelty; the practical contribution largely combines existing kv-cache merging techniques while conceptual novelty of applying it in a layer specific way, follows from prior work in interpretability.
- Weak hyperparameter justification with task-dependent threshold. Optimal values vary across datasets with no principled guidance for selection in practice. Design choices lack grounding beyond empirical tuning.

Overall: This paper presents a well-motivated approach with solid ablations, but the dramatic underperformance of LOOK-M compared to its original paper is a critical concern, along with narrow evaluation limited to long-context tasks and lack of testing on larger models. If the authors can demonstrate conclusively that FlowMM outperforms LOOK-M as well demonstrate generalizability to larger models, the reviewer can lean towards acceptance.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
