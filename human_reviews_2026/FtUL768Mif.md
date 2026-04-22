# Adapted-Language ViT: Empowering Self-Supervised Vision Transformers with LLMs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
The integration of Large Language Model~(LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Adapted-Language Vision Transformers (ALViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. ALViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that ALViT significantly improves performance in various downstream vision tasks, showcasing an effective and efficient way to harness LLM knowledge for visual understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AL-ViT (Adapted Language Vision Transformer), a framework designed to improve multimodal alignment between visual and linguistic representations in large-scale Vision Transformers. The key idea is to co-adapt a ViT and an LLM block through joint MAE-based self-supervision and LoRA-based LLM adaptation, achieving the alignment between modalities.

### Strengths
**Simple Solution:** The training pipeline (MAE pre-training + LoRA adaptation) is well-structured and straightforward.

**Good presentation:** The paper is well-organized and easy to follow.

### Weaknesses
**Incremental Technical Novelty:** The method primarily combines existing techniques — MAE for visual pretraining, LoRA for efficient adaptation, and LLM fusion from LM4Vision — into a joint optimization scheme. There is no fundamentally new algorithmic or theoretical contribution beyond this integration.

**Unclear Mechanistic Insight:** The paper hypothesizes that joint MAE-LoRA training bridges modality mismatch, but provides no formal analysis or theoretical grounding. The attention-entropy study is qualitative and does not establish causality.

**Cost and Gain:** The addition of an LLM block (even partially adapted) introduces large computational overhead and complexity for minimal gain. There is no clear analysis of training cost, FLOPs, or memory increase.

**Weak Motivation for LLM Use:** Since the LLM is frozen and only LoRA-adapted via reconstruction loss (not text supervision), it is unclear whether the language knowledge actually benefits the visual features — especially without textual context or captions.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This author introduces a novel framework called ALVIT, which aims to infuse the semantic knowledge of LLMs into self-supervised Vision Transformers to enhance performance on pure vision tasks. To address the modality mismatch between ViTs (vision-centric) and LLMs (text-centric), the authors propose a co-pretraining strategy. This strategy pre-trains the ViT backbone with a MAE objective, while simultaneously using the same MAE reconstruction loss to train Lora layers within the LLM fusion blocks. This joint optimization guides the ViT to produce LLM-friendly features and also enables the LLM to effectively interpret visual information. Experiments show that ALVIT significantly outperforms strong baselines (including MAE-ViT and LM4Vision) on ImageNet and its variants (such as IN-A, IN-C).

### Strengths
- **Co-Pretraining Strategy**: The core contribution of this paper is its co-pretraining framework. Using the MAE reconstruction loss to simultaneously guide the learning of the ViT backbone and the adaptation of LoRA layers in the LLM blocks is a novel and effective method to resolve the modality mismatch between visual and language representations.

- **Promising Performance**: ALVIT not only achieves performance improvements on ImageNet-1K but, more importantly, demonstrates significant advantages on multiple robustness benchmarks. This strongly proves that the model successfully leverages the knowledge from the LLM to enhance its resilience to out-of-distribution samples.

- **Solid Ablation Studies**: The authors validate the design choices of ALVIT through comprehensive ablation studies.

- **In-depth Mechanism Analysis**: Through a background robustness analysis on ImageNet and visualization of attention entropy, the paper provides profound insights into ALVIT's working mechanism. The analysis indicates that ALVIT exhibits a stronger ability to distinguish between background and foreground, which explains the source of its robustness.

### Weaknesses
- **Insufficient Discussion on MAE as an SSL Paradigm**: The authors chose MAE because its reconstruction loss is suitable for co-training. However, the authors' own analysis (Figure 3(a)) shows that the attention pattern of the MAE ViT/B baseline is "indifferent" to background and foreground regions. Given that other SSL paradigms (like DINO) are known for their strong foreground/background separation capabilities, the paper should further discuss why MAE is the sole or optimal choice for this task and whether other SSL objectives were considered.

- **Lack of Explanation for "Stronger LLMs Not Bringing Gains"**: A surprising finding (Table 3) is that using more advanced LLMs (like Gemma 2 or LLaMA 3.1) did not bring any performance improvement over the older LLaMA 1. This contradicts the intuition that stronger LLMs should provide richer semantic knowledge. The paper reports this phenomenon but fails to provide an in-depth discussion or hypothesis. Does this imply that ALVIT primarily utilizes the general transformer architecture of the LLM blocks rather than their specific, more advanced semantic knowledge?

- **Limited and Modest Gains on Downstream Tasks**: Although the paper includes object detection and instance segmentation results on MS COCO in Appendix B.1, these evaluations are not presented in the main text. Furthermore, while the results show consistent improvements, the gains are relatively modest (e.g., only a +0.5 increase in Bounding Box AP on COCO). To more comprehensively demonstrate that ALVIT is a superior representation learning method, the authors should present evaluations on a broader range of downstream tasks in the main paper.

### Questions
Please see the weakness section above.

### Soundness
3

### Presentation
3

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
This paper introduces Adapted-Language Vision Transformers (ALViT), which integrate frozen LLM blocks into ViTs via a MAE + LoRA self-supervised training scheme. By co-adapting both modules, ALViT achieves higher accuracy than previous LLM-fusion baselines and robustness on ImageNet benchmarks and shows improved background sensitivity via attention-entropy analyses.

### Strengths
1. Clear and novel motivation: extend supervised only LM4Vision to SSL training.
2. Efficient tuning: The author find the frozen LM layer is not suitable for SSL training, and uses LoRA for finetuning without increasing much trainable parameters.
3. Solid empirical performance: although marginal, consistently outperforms MAE-ViT basleine and LM4Vision; robustness improvements are also convincing.  
4. Thorough ablations: analyzes LoRA, parameter count, LLM layers, random initialization, and multiple seeds.  
5. attention entropy visualizations support the hypothesis of improved information filtering and robustness.

### Weaknesses
1. Lack of justification for the training objective:
Masked Image Modeling is a well-established self-supervised learning objective, but it is no longer the most advanced one. Methods such as MoCo, DINO, and iBOT all outperform MAE by a large margin. Why do the authors choose MAE instead of adopting these stronger SSL objectives?

2. Lack of metrics:
The paper mainly evaluates fine-tuning accuracy. However, for SSL models, other important metrics include linear probing accuracy and kNN accuracy. Can ALViT also outperform the MAE baselines on these metrics?

3. Missing baseline:
In Table 3, the authors compare different variants with roughly the same number of trainable parameters. However, an important baseline seems missing: simply increasing the depth of the MAE ViT-B to match the parameter count (for example, using a 13-layer ViT-B). Additionally, the paper primarily focuses on ViT-B-sized models—can ALViT maintain its advantage across larger model scales?

### Questions
1. Why do the authors choose Masked Image Modeling as the training objective, given that more advanced SSL methods such as MoCo, DINO, and iBOT have been shown to outperform MAE by a large margin?

2. Can the authors report additional SSL metrics such as linear probing accuracy and kNN accuracy to evaluate whether ALViT also outperforms MAE baselines on these measures?

3. Can the author provide an additional baseline where the depth of MAE ViT-B is simply increased (e.g., using a 13-layer ViT-B) to match the trainable parameter count of ALViT? 

4. Can the authors verify whether ALViT maintains its performance advantages across different model scales beyond ViT-B?

Since there is no borderline this year, I would still recommend borderline accept at this point for the clear motivation, solid results and experiments.

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
5

### Summary
This paper, building on the foundation of LLM4Vision, introduces a new method for using LLMs to enhance vision-only capabilities. The main change is the addition of the Masked Auto-Encoding (MAE) pre-training task, during which the LLM is fine-tuned via LoRA. Overall, it has achieved a few improvements in image classification tasks. This work has inspirational significance (or heuristic value) for the study of vision-only pre-training paradigms that can simultaneously fine-tune both the visual encoder and the LLM.

### Strengths
1.  The paper is well-written, clearly articulated, and the methodology is relatively easy to follow.
2.  The exploration of applying LLMs within a vision-only pre-training paradigm is insightful.
3.  The feature analysis within the ablation study is well-executed, providing an intuitive visualization of the differences at the feature representation level resulting from the proposed training strategy.

### Weaknesses
1.  This work offers limited novelty built upon the LLM4Vision foundation. The finding that fine-tuning a few parameters on an adapted pre-training task can boost performance is rather straightforward and somewhat anticipated.
2.  Furthermore, judging from the final results, the performance improvement appears to be quite marginal.
3.  Regarding performance validation, the authors have primarily focused on classification tasks. As a general-purpose visual encoder, its effectiveness must be validated on a broader range of vision tasks (e.g., detection, segmentation, visual understanding).
4.  Moreover, the paper lacks direct comparisons with LLM4Vision across this wider set of tasks, making it difficult to fully assess the benefits of the proposed modifications.

### Questions
1.  What is the model's performance on a more diverse set of vision tasks (e.g., object detection, semantic segmentation)? 
2.  Are there more experimental results providing a direct comparison against LLM4Vision under identical settings?

### Soundness
2

### Presentation
3

### Contribution
2
