# MetaModelSelect: A Lightweight Post-hoc Metamodel for Selective Classification

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Selective classification equips neural networks with a reject option, abstaining on low‑confidence inputs. Most post‑hoc selectors compress the logit vector into a single scalar (e.g., maximum softmax probability or energy), discarding structure in the remaining logits. We introduce MetaModelSelect, a lightweight two‑layer metamodel ($\approx 49$k parameters, <1ms overhead) trained on a frozen backbone to predict per‑example correctness. The metamodel leverages (i) a learnable embedding of the predicted class, (ii) the top‑3 entries of the normalized logit vector $\tilde z = z/\\|z\\|_{p^\*}$, and (iii) a logit‑concentration statistic $h(z) = \\tfrac{1}{C}\\sum_i \\tilde z_i^{\\,2}$. On ImageNet‑1k, Stanford Cars, and the long‑tailed iNaturalist‑2019, MetaModelSelect achieves state‑of‑the‑art risk–coverage with relative AURC reductions of 2.0-4.2\% over tuned MSP, Energy, and MaxLogit-$p$-Norm baselines, without additional data or backbone retraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduced the MetaModelSelect a post-hoc selective classification approach. The main contributions lie in making use of local logits geometry (top-3 p-normalised) and global logit concentration, together with class-specific feature into a single model. MetaModelSelect improves RC curve and achieves AURC between 2.2-3.7% over similar baselines showing empirical gains. The model do not require additional data or backbone retraining.

### Strengths
- Well defined feature design, combining local logit geometry and global concentration with class priors via embedding shows a novel formulation for selective classification.

- The lightweight approach 49k-parameters with < 1ms overhead show the authors concern to efficiency and it is a welcome contribution to the area.

### Weaknesses
- MetaModelSelect result gains are hard to define if they are substantial given the narrow scope. The paper deliberately did not compared against stronger baselines (multiple-pass post-hoc scores) makes it hard to assess absolute progress.  I suggest the authors to include stronger baselines and discuss the performance deltas, in terms of predictions and computational requirements, to clarify absolute progress.

### Questions
- Calibration is a known problem in selective approaches, can the authors elaborate on the calibration procedure.

- How stable is the chosen p-norm across seeds/datasets? Any benefit from learning p end-to-end within the metamodel?

- An important ablation study is top-k sensitivity of the method. I asked myself with k=3 and not 2 or 5? Can the authors provide some small experiment and elaborate on why k=3. 

- Could you make the class embedding input-aware (conditioned on metadata) while staying post-hoc?

- How would MetaModelSelect compare against learning linear combination of the tested baselines(Energy, Max Logit and p_Norm)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The work proposes a new post-hoc approach for selective classification called MetaModelSelect.
MetaModelSelect is a simple two-layer neural network trained on top of pre-existing classifiers that aims to predict per-sample correctness.
Experimental evaluation over three datasets shows (small) improvements.

### Strengths
The main strengths are:

1. The approach is intuitive and straightforward, making it easy to implement in real-life cases
2. The authors perform experiments over two large datasets, i.e., iNaturalist and ImageNet, supporting the usage of their approach on large-scale benchmarks;

### Weaknesses
In my view, these are the main shortcomings of the paper:

*Novelty:* I am not fully convinced by the novelty of the approach. 
1. First, it seems to me that the only modification to ConfidNet [1] is the explicit usage of  predicted class embeddings, top-k local logits and concentration measures,  while in ConfidNet, the authors do not explicitly consider this possibility (but still learn an entire network to predict possible mistakes). I would argue $(i)$ this change is rather incremental compared to ConfidNet; $(ii)$ I would like to see how the proposed approach compares w.r.t. ConfidNet. 
2. Second, the proposed approach seems once again a slight modification to the regression approach proposed in [2] (Theorem 9), where the authors advocate for the usage of a post-hoc trained model. I think the authors should discuss this point in detail.

*Clarity* While the overall idea is clear enough, I think the paper could benefit from a better writing.
1. some choices seem quite arbitrary (e.g., why top 3 logits and not top 4 logits?). The authors should motivate this better.
2. The experimental part is quite confusing. E.g., the authors state *we report accuracy (or error) at coverages C*. I think the authors should be clear on how they evaluate the methods.

*Soundness* I do not understand what the advantage is of considering a pre-defined set of transformations over the logits compared to considering a deeper neural network, which could extract the same information starting from the original logits.

*Relation with previous works* A few benchmarks have been proposed to evaluate existing selective classification methods, i.e., [3] and [4]. In [3], the authors show that there is no clear winner across methods in terms of performance; hence, the results are not particularly surprising when evaluated on only 3 datasets. Moreover, I wonder how the proposed approach works w.r.t. $(1)$ coverage failures (as shown in [3]) and $(2)$ w.r.t. the AUGRC metric proposed in [4].


**References**

[1] - Corbière, C., Thome, N., Bar-Hen, A., Cord, M., & Pérez, P. (2019). Addressing failure prediction by learning model confidence. Advances in neural information processing systems, 32.

[2] - Franc, V., Prusa, D., & Voracek, V. (2023). Optimal strategies for reject option classifiers. Journal of Machine Learning Research, 24(11), 1-49.

[3] - Pugnana, A., Perini, L., Davis, J., & Ruggieri, S. (2024). Deep Neural Network Benchmarks for Selective Classification. Journal of Data-centric Machine Learning Research.

[4] - Traub, Jeremias, Till J. Bungert, Carsten T. Lüth, Michael Baumgartner, Klaus H. Maier-Hein, Lena Maier-Hein, and Paul F. Jäger. "Overcoming common flaws in the evaluation of selective classification systems." Advances in Neural Information Processing Systems 37 (2024): 2323-2347.

### Questions
I have the following questions:

1. please discuss the weaknesses I highlighted
2. I did not understand how the frozen classifier is trained. Are you using the same training set to train both the classifier and the MetaModelSelect? I think this might be prone to overfitting.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new post-hoc method for selective classification via a lightweight metamodel. Specifically, the metamodel is a MLP with two hidden layers. The predicted class, top-k of normalized logits, mean of squares of the normalized logits are integrated as the input for the metamodel, which direct predicts the confidence score of whether the classifier is correct. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is easy to follow. 
2. The proposed post-hoc method can be adapted on different base classifiers. 
3. The metamodel is lightweight, which brings little extra cost.

### Weaknesses
1. The improvements are not significant. Statistical tests are expected. Meanwhile, the statement “2~4% AURC reductions” is not proper. Compared to the simple but effective baseline Softmax, the proposed methods seem to achieve limited improvements.
2. The selection processes and results of (p,d,B) are not provided, which makes the results less convincing.
3. There is no theoretical or in-depth experimental analysis on why such a simple metamodel can work well.

### Questions
1. As the backbone model is CCL-SC, why not use the datasets in its paper?
2. What are the final settings of hyperparameter p and d? Are the results sensitive to the choices?

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
The authors propose to train a small auxiliary model on validation data in order to perform confidence estimation for selective classification. They demonstrate their approach leads to performance improvements on a number of image classification datasets.

### Strengths
- The proposed approach is simple and direct. The inference cost is minimal.
- The proposed approach leads to improved performance on the discussed benchmarks.

### Weaknesses
The paper load for ICLR this year has been large, and so I have not been able to spend as much time as I would like on reviewing. I encourage the authors to correct any errors/misunderstandings I may have with regards to the paper.  

1. **Poor presentation**
    1. Table 1 overflows into the margin.
    1. It is unclear what Figure 2 is trying to illustrate.
    1. Large tables are hard to parse, when the same information could be easily conveyed using RC curves (see [1]).
    1. Before the conclusion the authors make claims about calibration without at any point evaluating calibration error.
    1. Surely the mean of squares measures spread, not concentration.
2. **Lack of knowledge advancement in contribution**
    1. Model design and feature selection is not well motivated -- in fact, in the appendix, it states that 50 different features were tried for effectiveness, a gridsearch, shotgun approach.
    1. There is little explanation for *why* the proposed approach outperforms the baseline. From the reader's perspective they've just trained an auxiliary model on some features and shown that it performs. A better contribution would have an analysis with the conclusion e.g. "we demonstrate the mean of squares is a useful feature *because* it captures epistemic uncertainty in logit vector".
3. **Experimental weaknesses**
    1. Experiments are all based on top of the CCL-SC baseline -- if the approach is post hoc, it should generalise across various pre-training recipes. Besides, many practitioners are likely to not have used CCL-SC for their model. [1] demonstrate that certain training recipes degrade softmax SC -- the meta model approach would be more appealing if it were demonstrated to work generally, regardless of whether the pretraining recipe has degraded the softmax.
    1. The demonstrated absolute performance improvements over softmax are modest.
    1. No experiments on data efficiency (how many samples does the meta model need to perform well?).
 4. **Lack of awareness of the literature**
    1. [2] Propose a similar approach for calibration, but it is not referenced. 
    1. [1] Establish and explain that p-Norm is only effective under certain circumstances, e.g. models trained with label smoothing, and not effective for models trained with vanilla CE and data augmentations. This needs to be considered when including it in an empirical comparison.
    1. [3] encode class-specific (and inter-class) uncertainty into a post-hoc optimisable confidence score, similar to this work, but are not referenced or compared against. 


[1] Xia et al, Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It, ICLR 2025

[2] Tomani et al, Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration, ECCV 2022

[3] Gomes et al, A Data-Driven Measure of Relative Uncertainty for Misclassification Detection, ICLR 2024

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2
