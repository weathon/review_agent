# RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 0, 6

## Abstract
Model merging aims to combine task-specific models into a unified model that is capable of multi-tasking, without any computational overhead of re-training. Regression Mean (RegMean), an approach that formulates model merging as a linear regression problem, aims to find the optimal weights for each linear layer in the merge model by minimizing the discrepancy in predictions between the merge and candidate models. RegMean provides a precise closed-form solution for the merging problem; therefore, it offers explainability and computational efficiency. However, RegMean merges each linear layer independently, overlooking how the features and information in the earlier layers propagate through the layers and influence the final prediction in the merge model. In this paper, we introduce RegMean++, a simple yet effective alternative to RegMean, that explicitly incorporates both intra- and cross-layer dependencies between merge models' layers into RegMean's objective. By accounting for these dependencies, RegMean++ better captures the behaviors of the merge model. Extensive experiments demonstrate that RegMean++ consistently outperforms RegMean across diverse settings, including in-domain (ID) and out-of-domain (OOD) generalization, sequential merging, large-scale tasks, and robustness under several types of distribution shifts. Furthermore, RegMean++ achieves competitive or state-of-the-art performance compared to various recent advanced model merging methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of model-merging: combining multiple task-specific models into a unified model without retraining. The authors engage with an existing method called RegMean (Regression Mean) which treats each linear layer in the merge model independently to minimize prediction error. They argue that RegMean neglects cross-layer and intra-layer dependencies (i.e., how features propagate through layers and how earlier layers affect later ones). This paper makes the following contributions:


* A new method called RegMean++ which extends RegMean by explicitly incorporating both intra-layer dependencies and cross-layer dependencies into the regression objective (i.e., modelling how layer outputs and later layers interact).


* Extensive empirical evaluation showing that RegMean++ outperforms RegMean across diverse settings: in-domain (ID) and out-of-domain (OOD) generalization, sequential merging (adding tasks one after another), large-scale tasks, and robustness under various distribution shifts. 

* A demonstration that this methodology competes with or even beats recent advanced model-merging methods (beyond just the baseline RegMean) in some cases.

### Strengths
* The authors identify a meaningful shortcoming in RegMean: the layer-wise independence assumption ignores downstream effects and feature propagation. The articulation of this gap is clear.



* According to the submission summary, the method is evaluated across multiple axes (ID vs. OOD, sequential merging, large-scale tasks, robustness) and consistently outperforms the baseline (RegMean).

### Weaknesses
* While the regression‐mean merging paradigm is attractive, it implicitly assumes linear relationships between the candidate models and the merged model’s weights. The world of deep networks is highly nonlinear and features propagate in complex ways; how valid is the linear regression assumption in deep networks? Extra discussion needed.


* The paper claims to model intra‐ and cross‐layer dependencies, but it may be unclear exactly how much additional modelling is done (e.g., are inter‐layer weights estimated, correlation structures learned?) and how scalable that is to very deep architectures.


* It would strengthen the work to include analyses of when RegMean++ fails, e.g., tasks that are very dissimilar or when layer dependencies are minimal, or whether the benefit is marginal in some cases—and what trade‐offs exist.

### Questions
* Compared with baseline RegMean (which is relatively cheap), how much extra computation or memory cost does RegMean++ incur (both at merging time, and at inference time if relevant)? Are there any deployment concerns?

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
This paper proposes a method called RegMean++ for multi-task model merging, which is built upon the previous RegMean work.

### Strengths
- This paper points out that RegMean considers only single-layer information during merging, while ignoring the role of cross-layer information flow.
- The paper compares the proposed method with existing approaches in terms of accuracy, out-of-distribution generalization, and performance under distribution shift scenarios.

### Weaknesses
- This paper is an improvement based on RegMean, and the main difference lies in the input features; thus, its novelty is limited.
- The paper only validates the method on ViT architectures and simple image classification tasks, lacking verification on LLMs and text generation tasks.
- The experiments do not provide a comparison of time costs among different model merging methods.
- In Table 2, it is unclear why the proposed method outperforms RegMean on OOD data; this conclusion lacks deeper analysis or theoretical explanation.

### Questions
In Table 1, the original paper reports 85.86 for TSV-M and 86.3 for ISO-C, while this paper reports 83.1 and 82.5, respectively. What causes this discrepancy in the reported results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes an improvement over RegMean. However, it lacks clear motivation, presents limited novelty, and provides an insufficiently explained methodology.

### Strengths
Extensive and well-controlled empirical campaign: Main benchmark covers 8 diverse image tasks, 3 model sizes, and 11 baselines. Sustainability under scale tested up to 20 tasks and sequential arrival. Robustness evaluated on seven corruption types and class-imbalanced or ImageNet-OOD samples.

### Weaknesses
1)This paper lacks clear motivation, presents limited novelty, and provides an insufficiently explained methodology.
2)No theoretical grounding for the “corrected” statistics: The manuscript claims RegMean++ “incorporates intra- and cross-layer dependencies” but offers no proof or even informal argument that using merged-model activations minimises a meaningful objective that couples layers (Sec. 3.1–3.2). Consequently, convergence, optimality, or error bounds with respect to the true multi-task risk are absent.
3)Computational cost brushed aside: RegMean++ needs an extra forward pass through the growing merged model for every layer to collect statistics (Algorithm 1, line 5). No wall-clock or FLOP comparison is given; the abstract claims “no computational overhead of re-training” but omits this non-trivial overhead relative to RegMean.
4)Scalability to larger models unaddressed: All experiments use ViT-B/32, B/16, L/14 (≤303 M params). The limitation section itself flags billion-scale models as future work, so current evidence does not support generalisability to LLM or large multimodal scenarios.
Code and checkpoints are not yet released, which limits immediate reproducibility.

### Questions
What is the motivation of this paper, and what are its main contributions?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents RegMean++, an extension of RegMean that augments the merging objective with both intra-layer and cross-layer dependency modeling. The method is evaluated across diverse scenarios—ID and OOD settings, sequential merging, and merging with corrupted data—and is accompanied by a detailed layer-wise importance analysis of the merging process.

### Strengths
[1] The paper is clearly written. The idea of using representations from the merged model—rather than the originals—is a neat and effective trick that substantially boosts merging performance.

[2] The evaluation is comprehensive, covering multiple vision benchmarks and providing thoughtful analyses.

### Weaknesses
[1] The study is confined to vision tasks. Demonstrating results on LLMs—where model merging is widely practised—would significantly strengthen the paper’s impact.

[2] While RegMean++ advances over RegMean, the sequential-merging results should also be compared to state-of-the-art methods tailored for this setting to establish competitiveness.

### Questions
(1) The reported performance for SOTA baselines such as Iso-C [1] and TSV-M[2] appears considerably lower than in their original papers. Is this due to a different set of model checkpoints being used? If so, could you verify whether the same trends hold when evaluating on the exact checkpoints used in those works?

References:

[1] Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A. D., & van de Weijer, J. (2025). No task left behind: Isotropic model merging with common and task-specific subspaces. arXiv preprint arXiv:2502.04959.

[2] Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., & Rodola, E. (2025). Task singular vectors: Reducing task interference in model merging. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 18695-18705).

### Soundness
3

### Presentation
3

### Contribution
3
