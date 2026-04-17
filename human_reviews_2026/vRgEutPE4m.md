# EffiVMT: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Recently, breakthroughs in the video diffusion transformer have shown remarkable capabilities in diverse motion generations. As for the motion-transfer task, current methods mainly use two-stage Low-Rank Adaptations (LoRAs) finetuning to obtain better performance. However, existing adaptation-based motion transfer still suffers from **motion inconsistency** and **tuning inefficiency** when applied to large video diffusion transformers. Naive two-stage LoRA tuning struggles to maintain motion consistency between generated and input videos due to the inherent spatial-temporal coupling in the 3D attention operator. In addition, they require time-consuming fine-tuning processes in both stages. To tackle these issues, we propose EffiVMT, an efficient **three-stage** video motion transfer framework that finetunes a powerful video diffusion transformer to synthesize complex motion. In **stage 1**, we propose a spatial-temporal head classification technique to decouple the heads of 3D attention to distinct groups for spatial-appearance and temporal motion processing. We then finetune the spatial heads in the **stage 2**.  In the **stage 3** of temporal head tuning, we design the sparse motion sampling and adaptive RoPE to accelerate the tuning speed. To address the lack of a benchmark for this field, we introduce MotionBench, a comprehensive benchmark comprising diverse motion, including creative camera motion, single object motion, multiple object motion, and complex human motion. We show extensive evaluations on MotionBench to verify the superiority of EffiVMT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes EffiVMT, an efficient framework for video motion transfer based on large pretrained Diffusion Transformers. The core contribution lies in a three-stage decoupled finetuning strategy that classifies attention heads into spatial and temporal types, followed by separate LoRA tuning for each. To further reduce training cost, the authors introduce sparse motion sampling and adaptive Rotary Position Embedding for temporal interpolation. They also present a new evaluation benchmark, MotionBench, covering diverse motion types.

### Strengths
* The paper identifies real inefficiencies in DiT-based motion transfer, coupled attention and slow LoRA finetuning, and proposes an elegant decomposition of spatial and temporal adaptation.
* The classification of attention heads based on attention-map similarity is well-motivated and empirically effective. Sparse motion sampling and adaptive RoPE are practical engineering contributions that improve efficiency without major sacrifices in quality.
* The paper is well-written, structured logically, and the visuals clearly illustrate both motivation and improvements.

### Weaknesses
* Although the engineering contributions are solid, the core idea of decoupled spatial-temporal adaptation and LoRA tuning has been explored in prior UNet-based works. The novelty mainly lies in adapting it to DiTs with additional heuristics
* The experiments rely heavily on Wan 2.1 as the backbone and Human/Animal motion data. It remains unclear how well EffiVMT generalizes to more diverse base models and more diverse scene generation.
* MotionBench is a good step forward but relatively small. It is unclear whether it truly captures the diversity of motion patterns needed for large-scale evaluation.

### Questions
Most of the major concerns are reflected in the weaknesses above, while I still have several minor questions as follows.

* How consistent is the head classification across different pretrained backbones? Is the “spatial vs. temporal” division robust, or highly model-specific?

* What happens if the sparse sampling ratio is pushed further,e.g., using 1/8 or 1/10 of frames? Is there a measurable degradation threshold?

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
5

### Summary
The paper introcudes a method for motion transfer mor the MMDiT-based video diffusion models. Since these 3D attention-based models doesn't have separate spatial and temporal layers, it is not easy to disentangle motion from appearance for the motion transfer task. EffiVMT introduces a pseudo-supervised method to classify each attention head in the 3D attention layer to identify spatial- and temporal-focused heads. Using this separation, they apply generic two-stage tuning-based motion transfer method to disentangle the motion from the source video. While fine-tuning, they also apply adaptive RoPE scaling in order to keep the model's learned temporal geometry while enjoying the short-clip fine-tuning. They also introduce a benchmark for the motion transfer task which is an important contribution to this line of work since there is no established way of quantitatively measuring the motion transfer task.

### Strengths
- The paper pinpoints specific deficiencies in prior video motion transfer methods (spatial-temporal entanglement causing poor motion fidelity, and cumbersome two-stage tuning) and directly targets them. The proposed spatial-temporal decoupling of attention is a logical and well-justified solution that yields clear benefits in motion consistency
- EffiVMT demonstrates state-of-the-art results on all evaluated fronts. It produces videos that better preserve the reference motion than existing methods, and this is reflected in high motion fidelity scores and human rankings
- The authors provide a thorough evaluation, including new metrics and a new benchmark dataset (MotionBench). They evaluate not only automatic metrics (which cover multiple aspects of video quality) but also conduct a user study with 20 participants, which showed a clear preference for the proposed method
- The paper’s presentation aids understanding of the method. The inclusion of algorithm pseudocode and an overview diagram helps in reproducing the approach. The authors also cite relevant works extensively and position their contributions properly in context. Assuming the MotionBench dataset will be released, this work provides both the tools and results for others to replicate and build upon the findings.

### Weaknesses
- The method, while effective, could be viewed as an engineering improvement over existing paradigms rather than a fundamentally new approach. It builds on the known two-stage fine-tuning concept (spatial then temporal LoRA) seen in prior work, and improves it by making the stages more efficient and disentangled.
- The contribution method relies on an assumption that the attention heads can be classified into two classes. The spatial-temporal head classification in Stage 1 is based on a heuristic threshold on attention patterns. There isn’t detailed analysis of how robust this classification is. If certain attention heads were misclassified (spatial vs temporal) for a given video or layer, it might impact performance (e.g., a truly temporal head being left out of motion tuning, or vice versa). The paper does not report if the number of heads classified as temporal vs spatial varies with different videos or if the 50% split (via the chosen threshold) is always optimal. This could be a weak point if the method is sensitive to this setting. A more adaptive or learned approach could potentially improve reliability here, though it’s not explored. 
- The paper lacks an explicit discussion of any limitations or scenarios where EffiVMT might not perform as well. While the results are uniformly strong on MotionBench, it would strengthen the submission to acknowledge possible weaknesses.

### Questions
- How does EffiVMT fundamentally differ from prior two-stage tuning methods (e.g., MotionDirector for U-Net, or DiT-based tuning by Abdal et al.) beyond the head-classification and efficiency tweaks? The method is presented as a three-stage pipeline, but one might see it as a refined version of the known spatial-then-temporal LoRA approach. 
- How sensitive is your spatial-temporal head classification to the chosen cosine similarity threshold ($\alpha$)? The method currently uses a fixed threshold to split heads roughly evenly. 
- Is there any analysis of attention heads' similarity scores? Does the hypothesis that the attention heads belong to either of two classes hold across all layers and attention heads?
- Do you plan to release the MotionBench dataset and your code? This work would have a strong impact on the community, and releasing the benchmark in particular would allow others to validate and compare future methods.
- What are the current limitations or failure modes of EffiVMT? While the results are excellent on the presented scenarios, it would be helpful if the authors can discuss any cases that remain challenging.

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
This paper proposes EffiVMT, an efficient three-stage framework for video motion transfer. This method addresses two key limitations of existing LoRA-based fine-tuning approaches: (1) the coupling of spatial and temporal information in unified 3D attention blocks leads to motion inconsistency; (2) the excessively long token sequence during multi-frame video fine-tuning results in high computational overhead.
To solve these issues, the authors first introduce a spatial-temporal attention head partitioning strategy, which divides attention heads into two categories: those responsible for spatial appearance and those for temporal motion. Based on this decoupling, a two-stage fine-tuning process is designed: the second stage uses single-frame samples to optimize spatial LoRA, while the third stage fine-tunes temporal LoRA through a sparse motion sampling scheme and adaptive Rotary Position Embedding (adaptive RoPE). This reduces the number of frames while maintaining temporal consistency.
In addition, this paper introduces MotionBench, a comprehensive benchmark covering various motion types, to support the standardized evaluation of video motion transfer tasks.

### Strengths
1.Problem formulation is timely and well-motivated: As video DiTs grow in scale, efficient and faithful motion transfer becomes increasingly critical. The paper correctly identifies that naive LoRA adaptation fails to respect the implicit spatial-temporal coupling in 3D attention, leading to appearance leakage and poor motion fidelity—issues not adequately addressed in prior work.

2.The proposed MotionBench covers multiple action categories and styles, which helps to achieve more rigorous and comparable evaluations.

### Weaknesses
1.The approach of visualizing attention maps to separate spatial and temporal attention heads is not particularly novel, as such spatial-temporal decoupling has been explored in prior video Transformer literature. Similarly, the proposed adaptive RoPE—adjusting positional embeddings according to subsampled frame indices—is conceptually similar to techniques commonly used in frame interpolation–based video generation methods, where positional encodings are adapted to variable or non-uniform frame rates.

2.There is a discrepancy between the description in line 472 of the main text and the results reported in Table 2. The experimental configuration for the “w/o Sparse Sampling” ablation in Table 2 is not clearly specified. Moreover, the numerical values in Table 2 appear inconsistent with those reported in Table 5 of the supplementary material. Additionally, the color coding in Table 2 is misleading.

3.The current ablation in Table 2 evaluates the impact of removing each component individually (e.g., w/o STD LoRA, w/o Adaptive RoPE, w/o Sparse Sampling). However, it lacks a full ablation baseline that disables all three proposed components simultaneously.

4.Key recent methods like ReVideo are cited but not evaluated.

### Questions
See weaknesses

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
The paper proposes EffiVMT, a three-stage video motion transfer framework designed to efficiently finetune large video diffusion transformers. EffiVMT introduces a spatial-temporal head classification to decouple 3D attention heads, sequentially finetunes spatial and temporal heads, and leverages sparse motion sampling with adaptive RoPE to accelerate tuning. Experiments demonstrate strong performance, particularly on complex motion scenarios.

### Strengths
* EffiVMT demonstrates strong performance, especially in complex human motion scenarios.
* The paper is well-structured, with a clear explanation of the three-stage pipeline and associated techniques.
* MotionBench provides a useful dataset for standardized evaluation of video motion transfer.

### Weaknesses
* The pseudo ground truths (M_{spatial}, M_{temporal}) based on diagonal proximity are highly heuristic and may not generalize to complex scenes. The fixed α = 1.25 used to classify heads lacks principled justification. No ablation is provided to show robustness to different α values.
* The approach assumes spatial and temporal attention maps are separable, which may not hold in videos with complex spatio-temporal interactions.
* Many design choices (e.g., diagonal maps, α threshold, dual-branch fusion) are empirical, without formal analysis or theoretical grounding. It is unclear under what conditions these design choices improve attention modeling.
* The method depends on fine-tuning, which could limit its applicability to short video clips, and it is unclear whether EffiVMT can efficiently handle longer sequences. Additionally, the paper does not provide an analysis of compute overhead, which may become a significant bottleneck for high-resolution or long videos.

### Questions
* Are temporal and spatial heads effectively separated in practice? Can this be visualized or quantified?
* How sensitive is the model to the pseudo ground truth design (diagonal maps)?
* Does dual attention fusion specifically improve temporal coherence metrics?
* The paper claims to support multi-object motion scenarios, but the examples only include videos with two objects. It remains unclear whether the model can handle more objects (e.g., three or more) and how its performance scales as the number of objects increases.

### Soundness
3

### Presentation
3

### Contribution
2
