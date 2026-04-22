# Beyond Single Models: Mitigating Multimodal Hallucinations via Adaptive Token Ensemble Decoding

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Large Vision-Language Models (LVLMs) have recently achieved impressive results in multimodal tasks such as image captioning and visual question answering. However, they remain prone to **object hallucination**—generating descriptions of nonexistent or misidentified objects. Prior work has partially mitigated this via auxiliary training objectives or external modules, but challenges remain in terms of scalability, adaptability, and model independence. To address these limitations, we propose **A**daptive **T**oken **E**nsemble **D**ecoding (**ATED**), a training-free, token-level ensemble framework that mitigates hallucination by aggregating predictions from multiple LVLMs during inference. ATED dynamically computes uncertainty-based weights for each model, reflecting their reliability at each decoding step. It also integrates diverse decoding paths to improve contextual grounding and semantic consistency. Experiments on standard hallucination detection benchmarks demonstrate that ATED significantly outperforms state-of-the-art methods, reducing hallucination without compromising fluency or relevance. Our findings highlight the benefits of adaptive ensembling and point to a promising direction for improving LVLM robustness in high-stakes applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Adaptive Token Ensemble Decoding (ATED), a training-free framework for mitigating object hallucination in Large Vision-Language Models (LVLMs).
ATED aggregates token-level logits from multiple LVLMs via an uncertainty-guided weighting mechanism that dynamically adjusts each model’s contribution during decoding.
A greedy uncertainty-minimization algorithm (UGO) further refines ensemble weights to balance reliability and computational cost.
Across POPE, CHAIR, and MME benchmarks, ATED consistently outperforms strong baselines, reducing hallucinations without harming fluency.
Ablation studies confirm that adaptive weighting and visual perturbations are key to robustness, while latency experiments show favorable trade-offs between inference speed and caption quality.
Overall, ATED demonstrates a flexible and effective approach to improving LVLM robustness in multimodal reasoning and captioning tasks.

### Strengths
- The paper addresses an important and persistent weakness of LVLMs (object hallucination) with a conceptually simple yet broadly applicable ensemble framework.
- ATED can be integrated with various off-the-shelf LVLMs (InstructBLIP, MiniGPT-4, LLaVA-1.5/Next) without retraining, showing practical scalability.
- The experiments span multiple benchmarks (POPE, CHAIR, MME) and include both quantitative and qualitative analyses, providing convincing empirical support.

### Weaknesses
- The proposed ATED appears conceptually close to prior methods: ensemble decoding (ED) and visual-contrastive decoding (VCD).
The only substantial novelty lies in the Uncertainty-Greedy Optimization (UGO), whose ablation (Table 3) suggests only marginal contribution, making the methodological innovation relatively weak.

- The paper lacks statistical significance analysis or variance reporting, which limits confidence in the reported percentage improvements.

- While ATED claims to be “training-free,” the computation overhead of multi-model forward passes is non-trivial; the paper does not provide detailed latency or cost comparisons under identical hardware constraints.

### Questions
- How many LVLMs were actually ensembled in each experiment, and how does performance scale with the number of models (N = 2 → 3 → 4)?

- How sensitive is ATED to the hyperparameters?

- Can the uncertainty-guided weighting be estimated within a single model using internal heads or adapters, rather than across multiple LVLMs?

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
The paper proposes Adaptive Token Ensemble Decoding (ATED), a training-free, token-level ensemble method to curb object hallucination in LVLMs. ATED aggregates per-step predictions from multiple LVLMs using uncertainty-adaptive weights and fuses diverse decoding paths to strengthen grounding and semantic consistency—without extra training or model-specific modules. On standard hallucination benchmarks, ATED lowers hallucination rates while preserving fluency and task relevance, outperforming prior objectives and post-hoc detectors.

### Strengths
1. The topic is interesting and tries to address an important problem.

2. The paper writing is easy to follow.

### Weaknesses
1. Beyond entropy, can uncertainty be measured with alternative metrics?

2. I’m unclear on the exact decoding procedure. After computing model-specific uncertainty weights, which model (or aggregation) actually drives decoding? Does this operate token-by-token only, or can it decode full sentences? If full sentences, must outputs from all models be forced to match exactly?

3. Please provide efficiency measurements for the entire procedure.

4. I would like deeper analysis—for example, showing whether higher-performing models systematically receive larger weights.

5. Can the proposed method be applied to the Qwen-2.5-VL family?

6. For the benchmark discussion, note that several recent studies [1, 2, 3] address both hallucination and maintain performance (even some improvement) on general scenario. I recommend the authors add some benchmarks like OCRBench, MMMU, MME etc.

[1] Mitigating Object Hallucinations via Sentence-Level Early Intervention.

[2] A topic-level self-correctional approach to mitigate hallucinations in mllms.

[3] Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Adaptive Token Ensemble Decoding (ATED), a training‑free framework that mitigates hallucinations in large vision‑language models by fusing next‑token logits from multiple LVLMs with uncertainty‑guided weights. ATED generates perturbed image variants, applies contrastive decoding, and greedily minimizes entropy to assign importance weights to each model, allowing a trade‑off between inference latency and accuracy. Experiments on POPE, CHAIR and MME demonstrate consistent gains over individual backbones and several plug‑and‑play decoding baselines, with ablations analysing weighting strategies and latency.

### Strengths
(1) Training-free, plug-and-play method that leverages existing LVLMs without retraining; works across several backbones. 
(2) Consistent empirical gains on POPE / CHAIR / MME over strong decoding baselines (VCD, ICD, SID). 
(3) Ablations + latency knob make the method well-diagnosed and practically tunable.

### Weaknesses
(1) The paper compares a multi-model ensemble to single-model baselines; real-world feasibility of running 2–3 LVLMs + perturbations per token is unclear. 

(2) The paper also lacks comparison to 2025 ED / FastED / iTaD / IFCD-style plug-and-play hallucination mitigators, weakening the “significantly outperforms SOTA” claim.

### Questions
(1) Can you provide a fair multi-model baseline (e.g., 3 LVLMs with uniform logit averaging, same perturbations) to isolate the gain from your uncertainty-greedy weighting?

(2) Can you add or report results vs 2025 ED/FastED-type methods on at least one of POPE/CHAIR to strengthen the SOTA claim?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Adaptive Token Ensemble Decoding (ATED), a training-free ensemble framework designed to mitigate multimodal hallucinations in large vision-language models (LVLMs). ATED dynamically fuses token-level logits from multiple LVLMs based on uncertainty-guided weighting, allowing it to leverage the complementary strengths of each model during inference. The method is evaluated on several benchmarks including POPE, CHAIR, and MME, and achieves significant improvements in hallucination reduction without requiring retraining or fine-tuning. The authors further discuss the trade-off between inference latency and accuracy and analyze various ensemble strategies.

### Strengths
1. The paper introduces a clear and well-motivated idea: ensemble decoding at the token level across multiple LVLMs guided by adaptive uncertainty. This fine-grained approach extends ensemble learning into multimodal generation, which is both innovative and practically relevant.
2. ATED does not require additional training, making it broadly applicable across existing LVLMs and compatible with open-source backbones like LLaVA, InstructBLIP, and MiniGPT-4.
3. The paper includes comparisons on multiple benchmarks (POPE, CHAIR, MME) and provides ablation studies showing the contribution of each component, such as uncertainty-guided weighting and greedy optimization.

### Weaknesses
1. The proposed ATED framework requires simultaneous inference across multiple LVLMs, which substantially increases GPU memory usage and deployment cost. Figure 4 also shows that inference latency can increase up to six times compared to standard decoding. In contrast, other training-free approaches such as VCD typically introduce at most a twofold increase in latency. This raises concerns about ATED’s scalability and practicality in real-world applications where efficiency is critical.

2. The experiments mainly compare ATED with training-free decoding strategies (e.g., VCD, ICD, SID, OPERA), but do not include training-based hallucination mitigation methods, such as instruction-tuning or preference optimization. Including such comparisons would better demonstrate ATED’s effectiveness and contributions.

3. While the paper reports improvements on hallucination-related metrics such as POPE, CHAIR, and MME (hallucination subset), it lacks qualitative and quantitative evaluation of overall generation quality (e.g., fluency, coherence, descriptive richness). For instance, benchmarks like RefoMB, LLaVA-Bench, MMStar, and MM-Vet could assess how ATED affects long-form captioning and reasoning under complex visual conditions. Without these results, it remains unclear whether ATED preserves or degrades naturalness in extended outputs.

4. The paper focuses mainly on hallucination benchmarks but does not discuss whether ATED affects general-purpose multimodal understanding. Evaluations on textVQA, DocVQA, InfoVQA, and VQAv2 would help determine whether the ensemble decoding alters the model’s broader reasoning or comprehension abilities. It is important to verify that the hallucination reduction does not come at the cost of decreased general accuracy or robustness on standard multimodal benchmarks.

### Questions
1. Include an analysis of deployment efficiency, especially GPU memory consumption and throughput, to clarify ATED’s practical applicability.
2. Add comparisons with training-based hallucination mitigation methods, such as fine-tuned models using preference alignment or reinforcement learning.
3. Extend the evaluation to long-form generation benchmarks and provide both evaluations of fluency and relevance.
4. Evaluate ATED’s impact on general-purpose performance using standard multimodal benchmarks (e.g., textVQA, DocVQA, VQAv2).

### Soundness
3

### Presentation
3

### Contribution
2
