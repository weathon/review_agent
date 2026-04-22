# UniRestorer: Universal Image Restoration via Adaptively Estimating Image Degradation at Proper Granularity

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Recently, considerable progress has been made in all-in-one image restoration. Generally, existing methods can be degradation-agnostic or degradation-aware. However, the former are limited in leveraging degradation estimation-based priors, and the latter suffer from the inevitable error in degradation estimation. Consequently, the performance of existing methods has a large gap compared to specific single-task models. In this work, we make a step forward in this topic, and present our UniRestorer with improved restoration performance. Specifically, we perform hierarchical clustering on degradation space, and train a multi-granularity mixture-of-experts (MoE) restoration model. Then, UniRestorer adopts both degradation and granularity estimation to adaptively select an appropriate expert for image restoration. In contrast to existing degradation-agnostic and -aware methods, UniRestorer can leverage degradation estimation to benefit degradation-specific restoration, and use granularity estimation to make the model robust to degradation estimation error. Experimental results show that our UniRestorer outperforms state-of-the-art all-in-one methods by a large margin, and is promising in closing the performance gap to specific single-task models. The code and pre-trained models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
UniRestorer presents a strong universal image restoration framework that achieves impressive performance across single, mixed, OOD, and even real-world degradations. Its combination of multi-granularity degradation representations, hierarchical expert specialization, and uncertainty-aware routing enables both robustness and precision, outperforming prior all-in-one baselines and approaching single-task SOTA results.
However, the core mechanism raises conceptual concerns: the training pipeline is overly complex, and the single-expert activation strategy appears suboptimal and parameter-inefficient. Clarification on these design choices—especially how the MoE routing behaves under mixed degradations—would be crucial to fully assess the method’s contribution.

### Strengths
1. The paper’s fine-grained degradation representation learning is a clear strength. By retraining a DA-CLIP–based extractor with fine-grained textual labels (e.g., light/medium/heavy noise or haze) and contrastive supervision, the authors enable the model to capture not only degradation types but also intensity levels.  Supplementary t-SNE visualizations results show that these representations are separable at both coarse and fine granularity, providing a solid foundation for the subsequent hierarchical clustering and multi-granularity expert design.
2. Demonstrates strong cross-distribution generalization, maintaining stable PSNR/SSIM across single, mixed, OOD, and real-world degradations, with zero-shot gains on unseen types.
3. Uncertainty-aware hierarchical routing adaptively selects coarse or fine experts for robustness or precision, reducing routing ambiguity and representation conflict while matching or surpassing single-task performance.
4. Comprehensive evidence beyond parameter scaling: broad baselines and ablations show gains arise from division of labor and routing, with the LoRA variant retaining near full-model performance.

### Weaknesses
1. The inference scheme activates only a single expert at a time, which limits the expressive power of the MoE and introduces substantial parameter redundancy, as many experts remain unused for each input.
2. When the MoE system encounters mixed degradations, how frequently does the router fall back to the 0-th level (coarse) expert?
If this fallback occurs in most cases, it is unclear why the model significantly outperforms a single Restormer trained directly on mixed degradations. Conversely, if the router instead activates a fine-level expert (e.g., a “rain” or “haze” expert) for a mixed input such as rain-haze, it would contradict the intended degradation clustering principle and could potentially degrade performance. Clarification on the router’s behavior and expert selection under mixed-degradation inputs is needed.
3. The overall training pipeline is overly complex and resource-intensive. It requires first training a degradation extractor, then performing hierarchical clustering, followed by separate training for multiple experts and an additional router stage.
4. The paper does not clarify how data sufficiency is ensured for the fine-granularity experts. Since the hierarchical clustering process recursively divides the training set into smaller subsets, some fine-level clusters may contain only a limited number of samples. It is unclear whether the authors applied any method to avoid underfitting or data imbalance across experts.

### Questions
The second weakness is the main issue that confuses me the most. I would appreciate a clear explanation, and I may consider raising my rating if it is addressed convincingly.

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
4

### Summary
The paper proposes UniRestorer, a universal all-in-one image restoration framework that bridges degradation-agnostic and degradation-aware paradigms. It performs hierarchical clustering on degradation representations to build a multi-granularity mixture-of-experts (MoE) model. Through degradation and granularity estimation, the router adaptively selects fine- or coarse-grained experts, achieving robustness against estimation errors. Experiments on single- and mixed-degradation benchmarks show that UniRestorer surpasses prior all-in-one methods and nearly matches single-task performance

### Strengths
1. This paper introduces a multi-granularity degradation representation that unifies coarse- and fine-grained experts. The idea of granularity estimation to quantify degradation uncertainty and guide routing is novel and intuitive.
2. Comprehensive experiments: covers 7 single-degradation and 11 mixed-degradation settings, plus real-world and unseen tasks.
3. Paper is well-organized and technically detailed with intuitive figures (especially Fig. 3 illustrating routing).
4. The hierarchical MoE design could inspire cross-domain generalization and adaptive restoration architectures.

### Weaknesses
1. While granularity estimation is conceptually convincing, there’s no formal uncertainty-theoretic or probabilistic analysis of its behavior.
2. Each granularity level adds parameters and routing complexity; actual FLOPs and latency comparisons are limited. The LoRA variant helps, but trade-offs between full and LoRA experts could be better quantified.
3. The K-means–based clustering assumes a consistent degradation embedding space; sensitivity to clustering hyperparameters (number of clusters, feature normalization) is not reported.
4. Evaluation primarily uses synthetic degradations; more real-world degradation diversity (e.g., motion blur, ISP artifacts) would strengthen the claim of universality.
5. The effect of routing noise or errors in granularity estimation itself is underexplored; visualization of routing confidence would add interpretability.

### Questions
1. How sensitive is performance to the number of levels (l = 3) or cluster count per level? Could adaptive clustering improve robustness?
2. Can you visualize which experts are selected for different degradations and how granularity affects routing under ambiguous inputs?
3. Could the same hierarchical expert tree generalize to unseen degradation combinations(e.g., low-light + blur + noise) without retraining?
4. In hybrid usage, can user-provided degradation cues override granularity routing? Would it further close the gap with single-task models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
UniRestorer proposes a multi-granularity MoE framework for all-in-one image restoration. It (1) hierarchically clusters degradation space, (2) trains multi-level MoE experts, and (3) uses joint degradation + granularity estimation to route inputs to the most suitable expert. Claims large gains over SOTA all-in-one models and narrows gap to single-task models. Code and models will be released.

### Strengths
Granularity estimation elegantly handles degradation estimation noise — robust and practical.
Hierarchical clustering + MoE scales well across 15+ degradation types.
Large, consistent gains (e.g., +2.1 dB PSNR on mixed test sets).
Comprehensive ablations (granularity levels, routing loss, expert count).
Clean figures: t-SNE of degradation space, expert activation heatmaps.

### Weaknesses
Clustering is offline and static — no online adaptation to new/unseen degradations.
Granularity estimator adds overhead — no inference latency reported (vs. PromptIR, AirNet).
No theoretical justification for hierarchical clustering choice (e.g., why 3 levels?).
Evaluation limited to synthetic degradations — no real-world camera pipeline (e.g., RAW → ISP).
MoE training unstable? No mention of load balancing loss or expert collapse.

### Questions
Report inference FPS on RTX 3090 for 512×512 input — how much slower than PromptIR?
Test on real-world degradations (e.g., DND, SIDD, GoPro real blur).
Ablate dynamic clustering — can granularity be learned end-to-end without offline K-means?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Existing all-in-one restoration schemes are either degradation-agnostic or degradation-aware, leaving a clear performance gap to single-task experts. This paper proposes UniRestorer, which first hierarchically clusters the degradation space and trains a multi-granularity mixture-of-experts (MoE) network. At inference it jointly estimates degradation type and granularity to activate the most suitable expert. Extensive experiments show that UniRestorer significantly surpasses state-of-the-art all-in-one competitors and narrows the gap to dedicated single-task models.

### Strengths
1. The paper proposes UniRestorer, the first framework that simultaneously exploits degradation and granularity estimation to overcome the inherent limitations of both degradation-agnostic and degradation-aware restoration methods.
2. Extensive quantitative and qualitative experiments convincingly demonstrate the superiority of UniRestorer over existing all-in-one baselines and its competitive performance against task-specific models.
3. The idea of granularity-aware expert selection is clearly articulated and technically grounded; it offers a fresh insight that could inspire future work on robust universal image restoration.

### Weaknesses
1. The paper lacks a quantitative analysis of the proposed hierarchical degradation-clustering step.
2. No comparison or discussion is provided against alternative clustering strategies (e.g., the spectral clustering adopted in SEAL).
3. It is unclear whether the Restormer baseline in Table 1 was re-trained under exactly the same degradation protocol and parameter budget; an ablation that removes both degradation and granularity estimation while keeping the backbone capacity fixed would better isolate the gain of the proposed method.

### Questions
Please check the weakness

### Soundness
3

### Presentation
2

### Contribution
2
