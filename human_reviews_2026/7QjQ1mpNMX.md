# Large Pretraining Datasets Don't Guarantee Robustness after Fine-Tuning

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Large-scale pretrained models are widely leveraged as foundations for learning new specialized tasks via fine-tuning, with the goal of maintaining the general performance of the model while allowing it to gain new skills. A valuable goal for all such models is robustness: the ability to perform well on out-of-distribution (OOD) tasks. We assess whether fine-tuning preserves the overall robustness of the pretrained model, and observed that models pretrained on large datasets exhibited strong catastrophic forgetting and loss of OOD generalization. To systematically assess robustness preservation in fine-tuned models, we propose the Robustness Inheritance Benchmark (ImageNet-RIB). The benchmark, which can be applied to any pretrained model, consists of a set of related but distinct OOD (downstream) tasks and involves fine-tuning on one of the OOD tasks in the set then testing on the rest. We find that though continual learning methods help, fine-tuning reduces robustness across pretrained models. Surprisingly, models pretrained on the largest and most diverse datasets (e.g., LAION-2B) exhibit both larger robustness losses and lower absolute robustness after fine-tuning on small datasets, relative to models pretrained on smaller datasets. These findings suggest that starting with the strongest foundation model is not necessarily the best approach for performance on specialist tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper conducts a large-scale study effect of finetuning on the robustness capabilities of pretrained vision models. They analyze different finetuning algorithms as well as different pretraining datasets across different sizes and find counterintuitive insights such as larger pretraining datasets (such as LAION-2B) don’t necessarily guarantee better absolute robustness (or its preservation), continual learning algorithms along with robustness finetuning algorithms work well, etc.

### Strengths
1. The investigation of the central question is timely, especially as we are moving towards a regime of blind scaling. I appreciate the author's effort towards running controlled experiments with different variations to pinpoint what leads to (or lack thereof) robustness preservation.
2. The authors clearly define the underlying metrics and experiment setup, and report results across many different pretraining-finetuning combinations. 
3. Overall, the paper is well written and easy to understand.

### Weaknesses
While I like the paper’s overall motivation and effort towards a comprehensive experiment design, I have a few fundamental confusions and critiques for some of the design choices. Please see the questions below for specific comments.

### Questions
1. The robustness improvement metric that only looks at the aggregate differences in accuracy is insufficient for a detailed analysis of robustness preservation, in my opinion, since even with minor aggregate differences, there can be large sample-level variations. Could the authors provide analysis with sample level agreement or invariance rates (between pretrained v/s finetuned model)?

2. Similarly, for the representational analyses subsection, previous works have established that CKA is unsuitable to analyse representational invariances [1]. Could the authors repeat this analysis with STIR instead [1]?

3. As I understand it, regardless of the pretraining dataset used, the authors still fine-tune on ImageNet-1k before conducting further fine-tuning experiments on robustness-related benchmarks. I'm unsure about this design choice, as it introduces potential confounding factors and deviates from the common practice of fine-tuning directly after pretraining. This raises questions about whether some of the analyses, such as the poorer performance of larger models, might be artefacts of this intermediate step. Could the authors elaborate on this? Additionally, do the authors ensure that all models are fine-tuned to convergence on ImageNet-1k? Models pretrained on larger datasets may require more time to converge, so using a fixed number of epochs might be insufficient. Could the authors share the accuracy/loss curves for this intermediate fine-tuning stage?


[1] Measuring Representational Robustness of Neural Networks Through Shared Invariances

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how fine-tuning affects robustness in pretrained models. It finds that models pretrained on larger, more diverse datasets (e.g., LAION-2B) show stronger catastrophic forgetting phenomenon and out-of-distribution performance loss when fine-tuned on smaller datasets, compared to smaller-scale pretrained models. The authors systematically evaluate this by introducing ImageNet-RIB, a benchmark that fine-tunes on one OOD dataset and evaluates on others within a suite of ImageNet-derived datasets. Experiments across ViT and ResNet architectures show that continue learning and robust fine-tuning methods (LwF, WiSE-FT, and model soup) can mitigate but not eliminate this robustness degradation. The paper emphasizes that foundation models pretrained on large-scale datasets do not guarantee OOD robustness after fine-tuning.

### Strengths
- **The topic studied is important and timely** - whether “bigger pretraining” always leads to better fine-tuned robustness. This problem is both practically relevant and valuable in a safety context for understanding the scaling behavior of pretrained models 
- **Proposed benchmark could be useful for the community to evaluate OOD robustness.** The proposed ImageNet-RIB extends existing robustness benchmarks by systematically testing robustness across multiple OOD dataset pairs. The design and description (fine-tune on one and evaluate on the rest) take reproducibility into account. 
- **Experimental setting is well-designed to explore model architectures and pretraining datasets.** The study includes multiple architectures (ViTs, ResNets) and pretraining datasets (ImageNet, LAION, OpenAI CLIP) with augmentations. The finding that large-scale pretraining can hurt robustness after fine-tuning is consistent across different settings, together with ablation studies. 
- **Insightful analysis with CKA.** Probing representation shifts from fine-tuning for different layers leads to a deeper understanding on which layer is more related to catastrophic forgetting.

### Weaknesses
We thank the authors for submitting the paper to ICLR 2026! There are a few weaknesses listed below which I believe can make the paper better. For some points, please also refer to the questions below.
- **Novelty concern.** ImageNet-RIB mainly aggregates existing ImageNet OOD datasets. The conceptual contribution is limited beyond combining them. Also, only evaluating on ImageNet-related datasets is not enough to make a strong argument on OOD robustness with pre-training dataset scale in general.
- **Weak statements about the position of this work in related work.** The related work section reads more like a background rather than a critical positioning of this work relative to prior robustness on transfer-learning studies. In addition to discussing other robust fine-tuning methods, consider including studies on OOD robustness in the pre-training and fine-tuning paradigm, which are closely related to this study. 
- **Limited methodological depth.** Even though the authors mention issues on inaccessible pretraining datasets, continual learning methods are narrowly chosen and replay-based or architecture-based techniques are not explored. Also, more causal or theoretical explanation on why and how large-scale pre-training amplifies forgetting would increase the significance of the work, which mainly contains experimental observations. 
- **Minor issue with presentation.** Figure 1 is very well-designed and informative. However, there are some unreadable tables (i.e., the font of the numbers are very small). I would recommend increasing figure legends (especially for Figure 5) to increase readability. On page 2, the contribution statement is unclear - claims “four-fold” but lists three bullet points.

### Questions
- Would it be possible to measure or control the domain distance between pre-training and fine-tuning datasets (e.g., with CLIP embedding distance)? This would clarify whether robustness decrease stems from data-scale or different domain-shift effects.
- The paper suggests dataset size drives forgetting (Figure 6) but only shows one subset experiment. Running multiple controlled size levels would strengthen this claim. Also, why only focus on ImageNet-related dataset, would results be comprehensive enough to generalize to all other kinds of downstream scenarios? Try some other fine-grained or sketches which are outside of ImageNet OOD domain? 
- How about comparing with prior robustness benchmarks? A direct empirical comparison to Taori et al. (2020) or some other works would contextualize ImageNet-RIB’s novelty. 
- How would the CKA analysis be like if it is plotted over fine-tuning epochs? Would that help show how the forgetting emerges?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This research focuses on the robustness issue (out-of-distribution performance degradation) of fine-tuning. For a more streamlined evaluation, the paper merges various existing out-of-distribution datasets for computer vision and provides several experiments on it. A main claim is that a model trained with richer and larger datasets could have a more severe robustness degradation after fine-tuning, which is supported by two models trained with larger datasets (OpenAI & LAIN-2B).

### Strengths
1. The paper focuses on an important research topic, the robustness issue of fine-tuning. It has become an especially essential domain for understanding large pre-trained models.
2. The paper is well written and easy to follow.

### Weaknesses
Although the paper discusses an important research topic and provides several interesting experiments, there are several key concerns, including novelty and experiments.
1. It is challenging to identify sufficient additional value that this paper contributes to the existing literature. The phenomenon of out-of-distribution (OOD) generalization degradation after fine-tuning is a well-established problem, with many prior works with rigorous theoretical analyses and proposing new mitigation algorithms. Although the paper's primary claim focuses on a potential issue from the use of richer and larger datasets, the evidence provided to support this claim appears insufficient. Despite observing this potential robustness problem with two large models, it is difficult to definitively conclude that this observation is generalizable to other models or datasets, thus limiting the broader applicability of this specific finding. Since the paper's finding is not yet supported by theoretical analysis or insights, more diverse experiments would be helpful for improving its reliability.


2. Experiments
- The experimental scope is limited to computer vision models and datasets. This narrow focus raises questions about the generalizability of the findings to other domains (e.g., LLM) where fine-tuning practices and robustness challenges might differ.
- The main experimental design is to compare various existing robust training methods for out-of-distribution scenarios. While the paper shows that continual learning can improve robustness of fine-tuning, the observed differences in experimental results do not appear to be very significant. Given that there is no variance measure (e.g., standard deviation) for many of the main results, making it difficult to confidently see the true impact and reliability of the reported improvements.

### Questions
The main concerns are explained together with the weakness section.

### Soundness
2

### Presentation
2

### Contribution
1
