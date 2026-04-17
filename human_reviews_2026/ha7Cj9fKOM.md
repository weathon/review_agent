# SUMMR: Self-supervised Joint Representation Learning for Symmetric Multimodal Retrieval

- Decision: Reject
- Scores: 4, 6, 6, 2, 4

## Abstract
Existing works on multimodality-to-multimodality (MM2MM) retrieval mainly focus on asymmetric retrieval, where text-image pairs in query and context serve distinct roles. In this work, we address the critical yet underexplored challenge of symmetric retrieval, where queries and contexts are interchangeable. We propose SUMMR, a novel two-stage self-supervised framework that leverages unlabeled web-scale image-text pairs, contrasting previous methods that heavily rely on costly supervised data. Based on the observation that both semantic alignment and discrepancies exist between the two modalities, we first learn a mask to disentangle shared and unique information within each image-text pair, allowing us to align the shared concepts while preserving modality-specific details. Then, we leverage this mask to automatically generate positive and negative samples for self-supervised contrastive learning of the final joint embedding. Complementing this framework, we introduce a novel benchmark featuring high-quality human-annotated positive and hard-negative pairs to evaluate symmetric MM2MM retrieval. On this benchmark, extensive experiments against ten SOTA methods show SUMMR surpasses the strongest supervised VLM by 3.42 points, with over 50x fewer model parameters and a 5x smaller embedding dimension. Code will be available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper identifies multimodality-to-multimododality (MM2MM) retrieval as a largely neglected problem. To eliminate the expensive manual annotation that has stalled progress, the authors propose SUMMR, a two-stage self-supervised framework for this task. It first learns an intersection mask that disentangles shared concepts from modality-specific details inside any unlabeled image-text pair, using distillation-guided global-to-local alignment and adaptive QDA thresholding.  It then exploits the mask to automatically create positives (intersection masked) and hard negatives (difference masked), and trains a compact late-fusion encoder with contrastive learning. Training on 800 k LAION-5B pairs and evaluating on a new human-curated sym-MM2MM benchmark (1 M distractors), SUMMR outperforms some strong supervised models while being >50× smaller and 5× lower-dimensional embeddings.

### Strengths
1. The paper proposes a multimodality-to-multimododality (MM2MM) retrieval task, and build a dataset for the task. 
2. The paper proposes SUMMER, a two-stage self-supervised framework for this task, which outperforms some strong supervised models while being >50× smaller and 5× lower-dimensional embeddings.

### Weaknesses
1. Constructed samples only simulate deletion, not addition or modification, heavy reliance on offline hard-negative mining to cover the gap. Image masking needs semantic segmentation; iterative clustering introduces extra hyper-parameters (t, Kmax, rmax) that may not generalise across domains.
2. SUMMER is only for the MM2MM task proposed in the paper. The MM2MM task might be a interesting task, but it's not that commonly used, nor is it that basic.

### Questions
1. How does performance scale with much larger uncurated web data (>10 M pairs) and larger backbones (>2 B)?
2. Can the mask-generation module be made differentiable end-to-end instead of relying on QDA thresholding?
3. What is the exact computational overhead of segmentation-based masking at inference time?
4. How sensitive are thresholds τV, τL to domain shift, and can they be predicted meta-learned for new corpora?

### Soundness
3

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
4

### Summary
This paper introduces the task of Symmetric Multimodal-to-Multimodal (sym-MM2MM) retrieval and proposes SUMMR, a self-supervised framework. Its contributions are:

*   **Novel Two-Stage Training Paradigm**
    *   Stage 1 learns to disentangle shared and unique semantics from image-text pairs.
    *   Stage 2 leverages this disentanglement to auto-construct samples for contrastive learning.

*   **Dedicated Benchmark Dataset**
    *   Constructed via a generative model-assisted pipeline with human annotation.
    *   Provides a reliable standard for evaluating sym-MM2MM retrieval.

*   **Strong Empirical Performance**
    *   SUMMR outperforms traditional supervised methods that are 50x larger in scale.
    *   It uses a compact model (0.2B parameters) and offers an efficient, annotation-free solution.

### Strengths
The strengths of this paper can be summarized as follows:

1.   **Originality:** This paper demonstrates high originality by being the first to formally define the "symmetric MM2MM retrieval" task. Its core contribution is a highly creative and novel self-supervised framework that leverages semantic disentanglement to generate its own training signal, moving beyond reliance on manually annotated data.

2.   **Quality:** The methodology is rigorously designed and validated, which is evidenced by the comprehensive ablation studies that effectively justify the necessity of each proposed component. The framework's robustness and quality are further underscored by its superior performance across different backbone architectures.

3.   **Clarity:** The paper is exceptionally well-written. Despite the inherent complexity of the proposed two-stage framework, the logical flow remains easy to follow, a feat achieved through well-designed figures and a clear, staged explanation of the pipeline.

4.  **Significance:** This work is of high significance as it directly addresses the critical bottleneck of annotation scarcity in symmetric retrieval scenarios. Furthermore, it introduces a self-supervised disentanglement paradigm that could inspire future research in multimodal representation learning, and the release of the benchmark dataset lays a solid foundation for subsequent work in this new area.

### Weaknesses
The weaknesses of this paper can be summarized as follows:

1.   **Experimental Comparisons:** While the comparisons against existing methods are favorable, the experimental setup lacks a crucial baseline, as the authors omit supervised fine-tuning of strong models (e.g., CLIP or mmE5) on their proposed benchmark. This omission makes it difficult to ascertain the actual performance gain of their self-supervised framework versus a direct supervised approach on the same task.

2.  **Validation of Disentanglement:** The validation for the core innovation of "semantic disentanglement" remains indirect, as it relies solely on downstream retrieval performance and similarity heatmaps. A more direct, quantitative evaluation of the Stage-1 intersection masks is missing, such as measuring their IoU against ground-truth segmentation masks to definitively prove the accurate localization of shared concepts.

3.   **Robustness Analysis:** The proposed two-stage pipeline inherently carries a risk of error propagation, where imperfect disentanglement in Stage-1 could directly compromise the quality of samples generated for Stage-2 training. The paper does not provide any analysis of the framework's robustness to such potentially noisy masks during the second stage.

4.   **Dataset Details:** The specific scale and partitioning details of the constructed benchmark dataset are not fully disclosed, including the exact number of positive/negative pairs in each split. This lack of detail could impact the reproducibility of the reported results and hinders a proper assessment of the model's generalization capability on a larger scale.

### Questions
The main questions for the authors are as follows:

1.   **Hyperparameter Sensitivity and Computational Cost:** It is crucial to understand the sensitivity of the final results to the newly introduced hyperparameters, such as the evolutionary mask schedule and loss weights. Furthermore, providing a rough estimate of the computational cost required for the two-stage training would greatly facilitate a more comprehensive comparison with other methods in terms of efficiency.

2.   **Dataset Details and Generalization:** To ensure full transparency and reproducibility, could the authors release detailed statistics on their benchmark dataset splits, including the exact counts of positive and negative pairs? Additionally, it would be valuable to know if there are plans to further validate SUMMR's generalization on larger-scale datasets or to explore its application to other downstream tasks that could benefit from semantic disentanglement.

3.   **Quantitative Evaluation of Disentanglement:** Moving beyond indirect validation through retrieval metrics, can the authors provide a direct and quantitative assessment of the Stage-1 "intersection masks"? For instance, evaluating the mask quality by calculating metrics like IoU against pixel-level annotations on a standard grounding dataset would offer concrete evidence for their precision in locating shared semantics.

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
3

### Summary
This paper targets the underexplored problem of symmetric multimodality-to-multimodality retrieval, where image-text pairs are interchangeable in both query and database roles. The authors propose SUMMR, a novel two-stage self-supervised framework that disentangles shared vs. modality-specific information within image-text pairs, automatically generating positive and hard-negative samples for contrastive joint embedding learning. A new human-verified benchmark is created to evaluate this task. Results show SUMMR outperforms 10 SOTA supervised multimodal embedding models, including large VLMs, with far fewer parameters and unlabeled training data.

### Strengths
1. New problem definition: Symmetric MM2MM retrieval is clearly motivated and differentiated from asymmetric paradigms.
2. Novel technical design:
Disentanglement of intersection vs. difference information is conceptually strong and well executed.
Evolutionary masking + QDA thresholding is innovative and improves robustness.
3. Self-supervised data construction removes annotation bottlenecks and aligns supervision with task needs.
4 . Benchmark contribution is valuable and fills a critical gap for this task.
5 . Strong empirical results: Surpasses SOTA supervised VLMs by 3.42 points with 50× smaller model and 5× smaller embedding size.
6. Comprehensive ablations show well-justified design choices.
7 . Clear writing and intuitive illustrations (Fig. 1, Fig. 3, Fig. 6 show the workflow well).

### Weaknesses
1. The dataset relies heavily on VLM + LLM + diffusion model augmentations (Fig. 2).
Human verification is mentioned but not quantified (e.g., % filtered, inter-annotator agreement).
2. Retrieval errors in edge semantics (color, orientation, occlusion) remain unclear.
3. Real-world symmetric retrieval scenarios (e-commerce, product catalog) are not benchmarked.
4. Disentangled shared vs. unique regions are shown only in a few examples—more systematic measurements needed (e.g., alignment with object grounding benchmarks).
5. Appears to rely more on strong CLIP-pretrained vision-language interaction; performance gap with DINO/BGE suggests method may inherit modality disparities.
6. Stage 1 requires LoRA tuning and repeated similarity estimation; no training-time comparison vs. baselines.

### Questions
1.	How large is manual annotation involvement in the benchmark?
What percentage of generated pairs were rejected? How many annotators and validation stages?
2.	Can the proposed masking approach handle multiple shared objects or relational semantics (e.g., “boy holding a ball next to dog”)?
3.	Since QDA assumes Gaussian similarity distributions, did you observe any multimodal deviations? If so, how was it handled?
4.	Would SUMMR work if raw captions are noisy (e.g., weak web alt-text)? Any mitigation strategies?
5.	Can the benchmark be released without copyright issues from COCO/LAION/WuKong source data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SUMMR, a self-supervised framework for symmetric multimodal-to-multimodal (MM2MM) retrieval. The key innovation is a two-stage approach that first learns to disentangle shared (intersection) and unique (difference) information between image-text pairs, then uses this disentanglement to automatically generate positive and negative samples for contrastive learning. It introduces a new benchmark for symmetric MM2MM retrieval and claim state-of-the-art results with 50x fewer parameters than supervised baselines.

### Strengths
1.Symmetric and asymmetric multimodal retrieval are clearly distinguished.

2.Self-supervised approach avoids expensive manual annotation, suitable for expansion.

3.Model efficiency is high: Significantly superior to the VLM baseline in terms of parameter count and embedding dimension.

4.Benchmark construction method is transparent: Uses VLM+LLM+SD synthetic data, supplemented by manual verification.

### Weaknesses
·No comparison to simpler baselines: What about standard CLIP with symmetric loss? Or CLIP with random masking?

·Table 2 shows Stage 1 alone achieves 82.2, but Stage 1+2 only improves to 86.5， is this 4.3 point gain worth the complexity?

·Missing analysis of failure cases: When does the QDA assumption break down? The types and causes of failure cases are not analyzed, making it difficult to determine the bottlenecks in methods.

·The training objectives of stage 1 are complex, and there may be conflicts between multiple losses, lacking theoretical or empirical balance analysis.

·Stage 1 requires computing all pairwise local similarities (O(n²) in patch/token count) , not provide FLOPs for Stage 1 vs. Stage 2.

·SUMMR is self-supervised training on symmetric tasks, while the baseline model is supervised training on asymmetric tasks.

·The evolutionary mask schedule ρ = 1 $\rightarrow$ 0 lacks theoretical justification. Why this particular annealing schedule? The paper provides no ablation on different schedules or convergence guarantees.

### Questions
1.SUMMR is self-supervised training on symmetric tasks, while the baseline model is supervised training on asymmetric tasks. Does this 'task misalignment' affect the fairness of comparison?

2.Can the evolution process of the mask during training be visualized? How to ensure that the mask truly captures the semantic 'intersection'?

3.The negative samples constructed currently are generated only through 'mask difference'. Have authors considered introducing ‘semantic conflicts’?

4.Why masking intersection creates semantic equivalence? This seems to assume that remaining information is perfectly complementary across modalities.

5.In Section 4.2, have authors used or compared existing segmentation methods (SAM) when using segmentation for image masking?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study the ([symmetric] multimodal-to-multimodal) MM2MM retrieval problem, where multi-modal queries and contexts have a bidirectional relationship (i.e., contexts retrieved with a query can be used as a query such that the resulting context may be the original query) -- which has some interesting use cases (c.f. Figure 1). To accomplish this task, the authors develop a late-fusion architecture and self-supervised learning strategy (that doesn't require any supervised pairs) to train SUMMR, a MM2MM retrieval system. The two stages of SUMMR are: (1) learning an "intersection mask" from image-text pairs capable of disentangling the shared semantic concepts from the modality specific details from an image-text pair (lines 213-215; Figure 3 -- has many technical details) and (2) leveraging this mask to generate positive and hard-negative examples for contrastive learning (line 215). Additionally, the create a new benchmark (sym-MM2MM) to evaluate the MM2MM task with a pipeline of: (1) using source images from {COCO, LAION, WuKong}, generate a fine-grained description using GPT-4o; (2) use GPT-4o to rewrite the description to create a modified description to create a controlled information gap; and (3) use Stable Diffusion (SD) 3.5 to synthesize a new image from the modified description (resulting in a positive pair and a hard negative pair that are validated with human annotation). Training SUMMR on 800k image-text pairs from LAION-5B, experiments are conducted to demonstrate improvements over supervised encoder-based methods (e.g., CLIP) and supervised VLM-based (e.g., MM-Emded, VLM2Vec, LamRA, UniME, mmE5) with different SUMMR backbones in terms of recall and precision based metrics (noting parameter efficiency). Ablations are performed to demonstrate the relative contribution of different SUMMR components and self-supervised sample construction strategy. Additionally ablations and sensitivity analyses are performed in the appendices.

### Strengths
Strengths of this paper include:
- The MM2MM retrieval task isn't well-studied and has some interesting applications. Additionally, in these applications where query-specific instructions aren't needed, the resulting models can likely be significantly smaller (as encoding-based models are likely sufficient).
- The paper is clear overall and has sufficient architectural details and rationales for design choices that with the code, I believe I could: (1) understand and reproduce many of the results and (2) explore additional directions for improving the model.
- For the most part, the empirical performance is positive and the ablation studies address the key points introduced in the paper.

### Weaknesses
Weaknesses of the paper include:
- The sym-MM2MM benchmark isn't particularly strong. As best as I can tell from the anonymous repository, it is 214 queries (that generate a positive and hard negative match) -- which I actually couldn't find in the paper. Additionally, the sym-MM2MM process includes a rewrite of the caption that is 'worse', making the secondary image a perturbation that doesn't 'improve' the image and is likely to be reflective of biases in GPT-4o or the associated prompt. In the same vein, this obviously heavily relies on strong VLLMs and human annotation. I think the data is likely good enough to support the experiments (even if the confidence intervals are likely larger than implied and is somewhat designed in line with the SUMMR model), but it isn't a very strong independent contribution.
- While a different problem, there isn't any cross-modal comparison with existing datasets. First of all, these would likely be interesting (but I don't believe required). More importantly, without showing this, I don't understand how this work "breaks the deadlock" (line 76) of "the field is stuck" (line 73) for more commonly studied multimodal retrieval settings.
- A bit of a nit, but SUMMR seems pretty specific to image-text settings; this is at least worth discussing.
- What would an "adapted VLM" method for this task look like? If it is a straightforward extension, it is a nice contribution and likely to perform better (albeit I might be missing something).
- I would recommend referencing even more of the results from the Appendices (e.g., sensitivity analyses) as I had more criticisms until I decided to go back and read the appendices.

### Questions
Reframing my 'weaknesses' as questions:
- How might you go about improving sym-MM2MM. What are the limitations and can they be resolved (and do you have plans to do so)?
- What results do you get for cross-modal settings (maybe with constrained instructions, a subset of MMEB)? Even only comparing to supervised encoder-based methods would be interesting -- even if worse, knowing how much would be useful. 
- Is SUMMR easily extended to other modalities such that there could be an evaluation for MMEB-v2?
- What would an "adapted VLM" method for MM2MM look like?

### Soundness
2

### Presentation
3

### Contribution
2
