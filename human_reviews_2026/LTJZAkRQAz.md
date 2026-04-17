# Towards Interpretable Steering  for Hallucination Mitigation  in Large Vision–Language Models

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large vision-language models (LVLMs) have achieved impressive performance in multimodal understanding and generation, yet they remain prone to hallucinations, particularly object hallucinations where entities are described yet do not exist in the input image. Existing mitigation methods often focus on output-level adjustments, while the internal mechanisms driving hallucinations remain poorly understood. In this work, we adopt an internal representation-level perspective by introducing sparse autoencoders (SAEs) to decompose dense visual features into sparse monosemantic neurons for interpreting and steering LVLMs.  Building on prior findings that injecting image noise exacerbates hallucinations, we further investigate how noise perturbations reshape internal representations, revealing that noise alters monosemantic neuron activations, disrupts visual semantics, and induces hallucinations. Furthermore, we show that manipulating specific neurons enables controllable influence over LVLM outputs. Based on these insights, we propose Contrastive Neuron Steering (CNS), which selectively amplifies truth neurons while suppressing perturbation-induced activations to mitigate hallucinations, and further enhances understanding of image-specific features through adaptive neuron constraints and always-on neuron suppression. Extensive experiments and analyses demonstrate that CNS effectively reduces hallucinations. Moreover, our CNS enables interpretable and controllable internal neuron-level interventions, providing both practical mitigation and mechanistic insights into how LVLMs encode and sometimes misrepresent visual information.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the internal organization and dynamics of visual representations in VLMs, focusing on how these representations respond to perturbations and contribute to visual hallucinations. The authors use SAE to interpret and manipulate the internal visual features, showing how image noise alters neuron activations, disrupts semantic structure, and leads to hallucinated outputs. They demonstrate that targeted neuron-level interventions—enhancing or suppressing specific neurons—can effectively control LVLM outputs, and that coordinated multi-neuron modulation is more powerful than single-neuron manipulation. Building on these insights, the authors propose CNS, a method that amplifies meaningful neurons while suppressing perturbation-induced activations, mitigating hallucinations. CNS integrates seamlessly with existing decoding-based approaches and achieves consistent hallucination reduction across multiple benchmarks

### Strengths
* The paper is well written and clearly structured, making the contributions accessible and easy to follow.

* The proposed applications demonstrate convincing performance improvements, highlighting the practical benefits of the methods.

* The analysis section effectively reveals a clear relationship between noise strength, model performance, and neuron change ratio, providing valuable insights into the model’s internal dynamics.

* The approach is relatively lightweight, suggesting that it can be applied efficiently without excessive computational overhead.

### Weaknesses
* The paper claims in the introduction that “By applying SAEs to LVLMs, we decompose dense embeddings into sparse, monosemantic neurons, each corresponding to a single interpretable concept.” However, the evidence supporting this claim appears limited. The only validation provided relies on visualizing the top-16 images with the highest activation values. As the authors themselves acknowledge, feature splitting and feature hedging are well-known limitations of SAEs, which raises doubts about whether the identified neurons are truly monosemantic or disentangled. Without more rigorous quantitative or qualitative evidence (e.g., systematic concept alignment or disentanglement metrics), I suggest that the authors moderate this claim.

* The proposed method introduces several hyperparameters — $\lambda$, top-K, and $\tau$. Although ablation studies are included, the main text does not clearly describe how these parameters were selected. Were they tuned using a validation or training set? Are the same values used consistently across all experiments? Clarifying the hyperparameter selection process is important for reproducibility and fair evaluation.

* Conceptually, the method resembles a denoising procedure applied to internal representations by modifying their sparse neuron decompositions. While this is an interesting approach, it would be valuable to discuss how it compares to more standard robustness techniques — for example, finetuning the model on noisy images to encourage similar representations to their clean counterparts (e.g., by regressing to pseudo ground-truth features, as is often done for adversarial robustness). It remains unclear whether the proposed method offers advantages beyond such established approaches or whether it primarily achieves a similar effect through a more hand-crafted intervention.

* Missing prior work and comparisons to previous approaches that edit internal representations of VLMs: 

[1] Jiang et al., Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations, ICLR 2025

[2] Kaduri et al., What's in the Image? A Deep-Dive into the Vision of Vision Language Models, CVPR 2025

### Questions
* The paper mentions a small group of always-on neurons — are these essentially similar to the “register neurons” described in [3], except that they are extracted from SAE as opposed to MLPs?

* What are the scaling laws in this context? Does training larger SAEs (with appropriately adjusted parameters) lead to better hallucination reduction results?

[3] Jiang et al., Vision Transformers Don't Need Trained Registers, NeurIPS 2025

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
This paper investigates the relationship between internal visual representations and hallucinations in large vision–language models (LVLMs). By introducing sparse autoencoders (SAEs), the authors disentangle dense visual features into interpretable monosemantic neurons and study how perturbations (e.g., noise) reshape them. They find that noise alters neuron activations and disrupts semantic structures, leading to hallucinations. Building on these insights, they propose Contrastive Neuron Steering (CNS), which amplifies truth-consistent neurons while suppressing perturbation-induced ones, achieving improved interpretability and hallucination mitigation across multiple LVLMs.

### Strengths
1. Novel interpretability perspective.
The paper offers an insightful neuron-level analysis of LVLM hallucinations, revealing how perturbations reshape visual semantics. This represents a clear advancement over purely decoding-based methods.

2. Simple yet elegant approach. The method is conceptually intuitive, leveraging noisy–clean contrasts to steer neuron activations and does not require retraining.

3. Complementary to existing methods. CNS can be combined with decoding-based techniques such as VCD or M3ID, showing consistent improvements across multiple LVLMs.

### Weaknesses
1. The evaluation primarily relies on relatively coarse, binary-style metrics (POPE, MME), which capture existence-level hallucinations but not free-form generative ones. Including open-ended benchmarks such as AMBER [1] or MMHal-Bench [2] would better validate the method’s generality and reveal how neuron-level steering behaves under natural caption generation.

2. The neuron-level findings are interesting, but the link between interpretability and performance improvement remains somewhat superficial. It would strengthen the work to include controlled experiments that verify whether manipulating the identified neurons genuinely leads to consistent hallucination reduction across diverse prompts or noise conditions.

3. Experiments are limited to some old LVLMs (LLaVA, InstructBLIP, Qwen-VL). Testing on newer, stronger models such as DeepSeek-VL or LLaVA-OneVision would strengthen claims of generalizability.

[1] Amber: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation, 2023.

[2] Aligning Large Multimodal Models with Factually Augmented RLHF, ACL 2024.

### Questions
See weakness

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
The paper investigates object hallucinations in MLLM by probing the internal visual encoder representations using SAEs. It claims that hallucinations arise from unstable neuron activations under noise and introduces CNS — a method that amplifies neurons responsive to clean images while suppressing those activated by noisy images, along with “always-on neuron suppression.”

### Strengths
Identifies “always-on” neurons that encode global statistics, which is a potential insight for interpretability.

CNS is simple, efficient, and can be applied plug-and-play at inference time.

### Weaknesses
Lack of causal rigor: The paper assumes neuron instability causes hallucination but presents only correlational evidence. No controlled interventions or causal mediation analyses are performed.

Weak empirical validation: More benchmarks are need for method generalizability, such as HallusionBench RLHF-V. Also improvements on the reported benchmarks are marginal (<2%).

Novelty: he novelty is marginal, and the intuition about the correlation between attention and hallucination is nearly common sense since OPERA. CNS is conceptually close to VCD (contrast between distorted vs. clean views) but simply moved to feature space.  Since the insight is not new enough, a more elegant way is need for ICLR

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduce SAEs to interpret and steer the internal visual representations of LVLMs. They find that neuron-level interventions can modulate LVLM outputs for targeted concepts. Then they propose CNS to amplifies meaningful neurons while suppressing perturbation induced activations for hallucination mitigation.

### Strengths
1. This paper tackles visual hallucinations in LVLMs, a important problem that directly impacts the trustworthiness and safety of multimodal AI systems.

2. This paper applys SAE-based interpretability techniques to vision-language models. This is a promising research directions for mechanistic interpretability in multimodal models.

### Weaknesses
1.  Flawed Logic in "Neuron-Level Statistical Analysis".
- It is a well-established phenomenon in representation learning that some neurons naturally encode global features (e.g., color, texture, brightness) that are present across diverse images. The observation seems trivial.
- The authors claim that always-active neurons (appearing in top-20 activations across all images) encode "over-generalized pseudo-global concepts" and are therefore detrimental. However, this claim lacks both theoretical justification and empirical evidence. 

2. Flawed Logic in “Diagnosing Hallucinations with SAEs.” 
- The part fails to establish a causal link with the former part "Neuron-Level Statistical Analysis". 
- The current observation that "perturbing neurons increases hallucinations" is trivial without comparing. 
- The diagnosis methodology lacks critical baselines: Always-active neurons vs. image-specific neurons vs. randomly selected neurons
- No experimental setup is provided for this part. How is hallucination rate measured? No metrics or protocols is described.

3. Limited Experimental Validation and Insufficient Baselines.
- The proposed method shows modest gains across benchmarks. From Table1, Table 2, and Table 4, there is only ~1% improvement on average. Given the added computational cost (generating perturbed images at inference time, SAE encoding/decoding), the improvements do not convincingly demonstrate the method's effectiveness.
- The paper omits several state-of-the-art hallucination mitigation methods. Missing reference and comparisons:
OPERA (Huang et al., 2023), HALC (Chen et al., 2024),  ADHH (Yang et al., 2025)

OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. CVPR 2023

HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding. ICML 2024.

Understanding and Mitigating Hallucinations in Large Vision-Language Models via Modular Attribution and Intervention. ICLR 2025.

### Questions
1. Why always-active neurons (appearing in top-20 activations across all images) encode "over-generalized pseudo-global concepts"? Is there theoretical justification and empirical evidence?

2. What is the causal link between "Neuron-Level Statistical Analysis" and "Diagnosing Hallucinations with SAEs"? What is the experimental setup for this part? Which dataset and model is used? How is hallucination rate measured?

### Soundness
2

### Presentation
2

### Contribution
2
