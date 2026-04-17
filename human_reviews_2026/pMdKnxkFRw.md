# STAR: Speculative Decoding with Searchable Drafting and Target-Aware Refinement for Multimodal Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Speculative decoding (SD) has proven to be an effective technique for accelerating autoregressive generation in large language models (LLMs), however its application to vision-language models (VLMs) remains relatively unexplored. We propose~\textit{STAR}, a novel SD framework designed specifically for fast and efficient decoding in VLMs. STAR leverages a neural architecture search (NAS) framework with target-aware supernet training to automatically identify both the optimal interaction strategy between the draft and target models, and the most suitable draft model architecture for the underlying hardware implementation platform. STAR additionally incorporates adaptive intermediate feature distillation, guided by attention entropy, to enable efficient draft training. Experiments on a range of well-established VLMs, including LLaVA series, Pixtral, and SmolVLM, demonstrate that STAR achieves up to a $3.8\times$ speedup compared to standard decoding approaches and significantly outperforms existing SD baselines in both inference throughput and speculative acceptance length across a wide spectrum of VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes STAR, a speculative decoding method for VLMs, which utilizes a NAS framework to find the optimal draft model configuration. STAR is comprised of a two-phase training strategy and three distinct loss functions. On various tasks, STAR outperforms existing speculative decoding methods.

### Strengths
1. The paper tackles the highly relevant challenge of speculative decoding for VLMs.
2. The paper is clear and well-presented.
3. STAR demonstrates performance gains over existing methods.

### Weaknesses
1. The comparison appears unfair. STAR uses dataset specific draft models selected via an exhaustive search to find the highest speedup for each benchmark. Baselines (e.g., Medusa, EAGLE) likely use general configurations. Was STAR also evaluated using a single, fixed configuration for a fair comparison?

2. The exhaustive search after TPPT is complete introduces a substantial cost (e.g., approximately 12 minutes per 100 mini-batches on MMT-Bench), limiting practical adaptability. Have the authors considered a test-time adaptation method?

3. The work is an effective application of existing techniques (NAS, pruning, distillation), limiting its conceptual novelty.

4. The evaluation lacks comparisons to other recent speculative decoding methods specifically designed for multimodal LLMs, such as DREAM [1] and MASSV [2].

[1] DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding (NeurIPS 2025)  
[2] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (arXiv 2025)

### Questions
Please refer to the Weaknesses section above.

### Soundness
3

### Presentation
3

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
This paper introduces STAR, a speculative decoding method for VLMs. It employs a neural architecture search framework to automatically identify the optimal interaction strategy between the draft and target model and the ideal draft model architecture. Moreover, STAR introduces a intermediate feature distillation for training of draft model. Experimental results show that it achieves a better performance than Ealge-2 etc. on multiple VLMs.

### Strengths
- The proposed integrated training and distillation shows clear speed-up gains over naïve pruning.
- The entropy + delta-entropy metric provides a principled way to select informative layers for distillation/architecture decisions.
- The authors conduct experiments across multiple multimodal benchmarks with

### Weaknesses
- TPPT requires target model forward passes and intermediate feature distillation, the overhead need to be clarified.
- While fine granularity gives little benefit here, optimal settings may differ for other model families or tasks.
- The related work[1] needs to be discussed and compared.

[1] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding, NeurIPS’2025.

### Questions
- How can we monitor acceptance length and draft rejections online to autotune γ and compression ratios per workload?
- What is the expected GPU-hour if the target model grows to 70 B or 110 B parameters? Is there a transfer protocol so that a super-net trained for LLaVA-7 B can warm-start the 13 B variant?
- AIFD uses sum of entropy and ∆-entropy. Why not a weighted sum or a learned gating network? Did the authors try other indicators, e.g., Fisher information or gradient variance?
- Have the authors tried tree-based verification to accept discontinuous spans?

### Soundness
2

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
This paper presents STAR, a speculative decoding framework tailored for vision-language models (VLMs). Unlike prior speculative decoding methods designed for text-only LLMs, STAR addresses multimodal challenges such as visual token redundancy and cross-modal feature alignment. The framework introduces three main components: (1) Searchable Drafting: a neural architecture search (NAS)–based approach to automatically find the optimal draft model structure, pruning ratio, and feature interaction strategy; (2) Target-Aware Refinement: an adaptive intermediate feature distillation method that selects target layers features based on attention entropy and stability for better multimodal supervision; (3) Adaptive Pruning: dynamic visual and textual token pruning guided by the target model’s attention maps.

Through these components, STAR jointly optimizes speed, accuracy, and hardware efficiency. Experiments on multiple VLMs (LLaVA, Pixtral, SmolVLM) and six benchmarks demonstrate up to 3.8× decoding speedup with minimal performance loss, showing both strong system-level integration and transferability across architectures and devices.

### Strengths
- The paper systematically extends speculative decoding from text-only LLMs to multimodal VLMs through a well-structured framework combining searchable drafting, target-aware refinement, and adaptive pruning. This integration demonstrates strong system-level design ability and effectively addresses VLM-specific issues such as visual token redundancy and cross-modal feature alignment. 

- The method shows good transferability, working consistently across multiple VLM architectures (LLaVA, Pixtral, SmolVLM) and diverse multimodal benchmarks (ScienceQA, MMBench, SEED-Bench, MathVista). The experiments are comprehensive and carefully executed, covering throughput, acceptance ratio, and hardware efficiency, which provides solid empirical validation.

### Weaknesses
- The algorithmic novelty is modest. While the proposed STAR framework is well-motivated and demonstrates strong empirical results, its technical novelty appears incremental and compositional rather than conceptual. Each component—Neural Architecture Search (NAS), attention-based intermediate feature distillation, and adaptive token pruning—has been extensively studied in prior literature. STAR primarily reassembles these existing techniques within the context of speculative decoding for VLMs, without introducing a fundamentally new algorithmic mechanism. Consequently, the contribution seems more system-level and application-driven than theoretically or algorithmically innovative. The key value lies in integrating multiple known techniques effectively to address the multimodal bottlenecks in speculative decoding, rather than in proposing a novel computational principle.

- Another concern is that all baselines (SPD, Medusa, Hydra, EAGLE, etc.) were designed for text-only LLMs. Although the authors state they “adapt” these methods to VLMs, the adaptation process is not described in sufficient detail. As a result, it is unclear whether the improvements arise from STAR’s architectural innovations or simply from modality-specific adaptations (e.g., pruning visual tokens, target-aware distillation) that are unavailable to the baselines.

### Questions
The authors should better substantiate STAR’s originality and fairness. 

- To address the limited novelty concern, consider formalize the Target-Aware Refinement beyond a simple attention-entropy heuristic, introduce a multimodal-specific NAS objective rather than reusing standard search schemes, and show ablations quantifying how each module (NAS, refinement, pruning) contributes to both speedup and acceptance ratio. 

- To address the fairness concern, they should clearly describe how LLM baselines (SPD, Hydra, EAGLE, etc.) were adapted to VLMs, or include stronger multimodal variants (e.g., EAGLE + vision token pruning) to ensure comparable settings.

### Soundness
3

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
3

### Summary
STAR accelerates VLM inference through speculative decoding with neural architecture search to jointly optimize draft model configuration (attention head pruning, visual token compression, feature injection) and target model alignment. The framework uses adaptive intermediate feature distillation and two-phase progressive training, achieving up to 3.8× speedup and consistently outperforming existing methods across multiple VLMs and benchmarks.

### Strengths
1. The proposed method achieves consistent improvements over a number of reported baselines and datasets.
2. The paper applies NAS to VLM speculative decoding, which jointly optimizes draft model architecture and target feature alignment through entropy-guided distillation, addressing a previously unexplored optimization space in multimodal acceleration.
3. The method is clearly introduced.

### Weaknesses
1. The connection between motivation and methodology is unclear. The authors stated that the VLMs require more computation than text-only LLMs, but did not explain where the extra computation comes from. For decoder-only VLMs, does the extra compute come from the visual encoder, which is separate from the speculative decoding process studied in this paper?
2. The experimental settings are not described clearly enough. For example, for the data in Table 2, what inference batch size was used, what inference framework was employed, and on what hardware were the measurements taken?
3. Lack of a comparable baseline: [EAGLE-3](https://arxiv.org/abs/2503.01840) is not included.

### Questions
1. Table 3 compares STAR with EAGLE in different GPU settings. What is the performance of STAR where no hardware-aware searching is applied in different GPUs? (or with the same searching result applied to all GPUs)
2. Other questions in the "Weakness" section.

### Soundness
2

### Presentation
3

### Contribution
2
