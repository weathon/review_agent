# Conformal Correction for Efficiency May be at Odds with Entropy

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Conformal prediction (CP) provides a comprehensive framework to produce statistically rigorous uncertainty sets for black-box machine learning models. To further improve the efficiency of CP, conformal correction is proposed to fine-tune or wrap the base model with an extra module using a conformal-aware inefficiency loss. In this work, we empirically and theoretically identify a trade-off between the CP efficiency and the entropy of model prediction. We then propose an entropy-constrained conformal correction method, exploring a better Pareto optimum between efficiency and entropy. Extensive experimental results on both computer vision and graph datasets demonstrate the efficacy of the proposed method. For instance, it can significantly improve the efficiency of state-of-the-art CP methods by up to 34.4\%, given an entropy threshold.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows a fundamental trade-off in conformal prediction between efficiency (smaller sets) and prediction entropy (more decisive probabilities), and reframes conformal correction as multi-objective optimization on this Pareto frontier. It introduces EC3, a plug-in adapter that takes base probabilities and is trained with a conformal-aware inefficiency loss while constraining entropy via focal loss and temperature scaling, shrinking sets without inflating uncertainty. A theoretical link between APS expected set size and entropy explains why improving one typically harms the other. Experiments on vision and graph benchmarks indicate EC3 consistently yields better efficiency–entropy trade-offs and improves conditional coverage while preserving target marginal coverage.

### Strengths
1. This paper clearly identifies and formalizes a trade-off between CP efficiency and prediction entropy, with supporting analysis (e.g., APS-based bounds) that explains why the objectives can conflict.
2. This paper persuasively recasts conformal correction as a multi-objective problem with entropy as a first-class constraint, making the practical value obvious.
3. The paper is well organized; figures (especially Pareto plots) effectively illustrate key ideas and make the theory easy to follow.

### Weaknesses
1. The contribution is hard to assess because similar ideas have been explored previously. Please expand the related-work discussion to delineate what is new here and add the relevant prior methods as baselines to the empirical evaluation.
2. This work does not establish formal conformal coverage guarantees, in contrast to standard CP.

[1] Xi H, Huang J, Liu K, et al. Does confidence calibration improve conformal prediction.    
[2] Dabah L, Tirer T. On Temperature Scaling and Conformal Prediction of Deep Classifiers.

### Questions
1. See weaknesses. Please discuss more about related works and experimental baselines to outline the contribution.
2. Please provide more training details: loss trajectories and how the two objectives (efficiency vs. entropy) evolve and interact during optimization.
3. The evaluation shows that, at matched entropy, your method yields smaller prediction sets. Why is this desirable for downstream decision making?
4. Hyperparameters appear important; please expand discussion and provide practical selection guidelines.
5. How does the method perform under distribution shift?

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
The paper introduces Entropy-Constrained Conformal Correction (EC$^3$), an approach to improve the efficiency of non-conformity score “adapters” for downstream conformal prediction. The authors facilitate this by augmenting the inefficiency loss used to train these adapters with focal loss. Lastly, they provide theoretical grounding for adding focal loss and empirical results using EC$^3$ on image classification, node classification, and Q&A tasks.

### Strengths
- The paper includes a strong empirical evaluation, presenting results across several datasets with varying characteristics (i.e., number of classes). 
- The paper provides a clear justification for using **focal loss** in training adapters for Adaptive Prediction Sets (APS) non-conformity scores. It also provides a good theoretical connection between minimizing focal loss and maximizing entropy, leading to smaller prediction set sizes (**Theorem 3**, under the assumption $\mu \geq 0.5$).  
- The paper also provides a **class-conditional version of EC³** to address imbalanced class coverage.

### Weaknesses
- The main weakness of the paper is the lack of comparison with APS using randomization, as introduced in [1]. The authors employ APS **without randomization** in their experiments. In other words, their implementation omits the second red and bolded term in the randomized non-conformity score:
$$
V(x, y; u) = \sum_{i=1}^{y} \hat{\pi}_ {(i)}(x)~\textcolor{red}{\mathbf{- u \hat{\pi}_{(y)}(x) }}
$$
for 
$u \sim U([0,1])$. Theoretically and empirically, randomized APS has been shown to **reduce set size**  (**improve efficiency**) compared to non-randomized APS [2]; thus, it is imperative to include randomized APS when comparing with SOTA methods.  

   - For instance, Cora-ML is reported to have efficiencies of around 4 and 1.85 for the baseline CP and CF-GNN methods in Table 2, respectively, using non-randomized APS. However, Figure 5 in [2] shows efficiencies closer to 1.5 when using the randomized APS approach for both baseline CP and CF-GNN — indicating that randomized APS performs similarly to EC³. (Similar comparisons are available in [2] for the remaining datasets). For this reason, the perceived gains of **EC³** may not be as significant as claimed.

- The paper is missing an efficiency plot/table for the class-conditional $EC^3$. It is important to quantify the efficiency and SSCV trade-off between the two methods.

- The paper references $EC^{3}-1$ and $EC^{3}-2$, but it is not clear what those are referencing.

- Missing definitions for SSCV and WSC, either in the main body or appendix.


References

[1] Y. Romano et al. Classification with valid and adaptive coverage [NeurIPS 202]

[2] P. Maneriker et al. Conformal Prediction: A Theoretical Note and Benchmarking Transductive Node Classification in Graphs [TMLR 2025]

### Questions
See weaknesses. The main weakness/question is a lack of comparison with APS with randomization. If that can be addressed by the authors and the $EC^3$ still provides quality results, I will be happy to adjust my score accordingly. 

L743: "inequation" -> "inequality".

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
This paper investigates the relationship between efficiency (small size of conformal prediction (CP) sets) and entropy (uncertainty in model predictions) in conformal prediction frameworks. The authors show empirically and theoretically that these two quantities are often in conflict: increasing CP efficiency typically raises prediction entropy, reducing confidence in predictions.

To address this, the authors introduce $EC^3$ (Entropy-Constrained Conformal Correction), a new conformal correction method that adds an entropy control term (via focal loss and temperature scaling) to balance efficiency and entropy. They formalize this trade-off as a Pareto frontier and use $EC^3$ to search for better optima.

Experiments on vision (CIFAR-10/100) and graph datasets (Cora-ML, CS, Photos) show that $EC^3$ achieves up to 34% improvement in efficiency while maintaining marginal coverage and improving conditional coverage.

### Strengths
- The paper provides a first rigorous analysis linking CP inefficiency and prediction entropy (Propositions 1-2, Theorem 3).
- It identifies a real tension between compact prediction sets and calibrated uncertainty, which has been largely ignored by prior conformal training work.
- The $EC^3$ objective combines focal loss and inefficiency regularization with entropy control; temperature scaling provides a simple yet effective Pareto traversal mechanism.
- Extensive experiments across multiple architectures and domains; results are consistent and show significant practical gains.
- The paper is clearly written, with good visualizations (e.g., Pareto frontiers) and an accessible discussion of theoretical results.

### Weaknesses
- The analysis focuses on adaptive conformal prediction (APS); extension to other CP variants (e.g., regression or non-adaptive scores) is not discussed.
- The entropy parameter $\gamma$ and the temperature $T$ are hyperparameters tuned via grid search; no principled guidance for choosing them is provided.
- While acknowledged in the Limitations section, empirical degradation in base model accuracy is not quantified or analyzed.
- Some proofs (especially Proposition 2 and Theorem 3) rely on simplifying assumptions (exchangeability, bounded calibration errors) that might not hold in practical non-i.i.d. data settings.
- Baselines are limited to conformal-training-based methods; recent calibration or information-theoretic conformal approaches could strengthen evaluation.

### Questions
- How sensitive is $EC^3$ to the choice of $\gamma$ (entropy weight) and $\beta$ (inefficiency weight)? Can adaptive schedules mitigate tuning difficulty?
- Does $EC^3$ preserve or degrade conditional coverage guarantees beyond empirical results? Are there any theoretical bounds?
- How does the method perform on non-classification tasks (e.g., regression CP or language model calibration)?
- Could temperature scaling alone (without $EC^3$) achieve comparable Pareto improvements with careful tuning?
- Are there insights into how entropy affects human interpretability of conformal sets - e.g., does lower entropy align with better human decision support?

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
2

### Summary
This paper empirically highlights a tradeoff between the entropy of the model predictions and the set size ( efficiency ) of CP. The
 authors provide theoretical justifications of this trade-off for a specific score function and propose a new conformal training algorithms that achieves a more favorable balance between entropy and efficiency. The method, which fine-tunes pertained predictors using the proposed conformal adaptor, rather than, scratch, demonstrates improved efficiency in terms of prediction set size compared to existing conformal training methods such as ConfTr ( Stutz et al. )

### Strengths
1. The investigation of the entropy and CP set size in this particular setting of conformal training is novel to the best of my knowledge. the framing is new and can be valuable for future research. 

2. The theoretical results, though specific to a single score function, are well-motivated and provide interesting intuition.

3. The empirical results show promise in terms of prediction set size minimization relative to prior work. 

4. The authors had valuable practical considerations in mind. Instead of retraining the entire model, the paper utilizes the idea of conformal wrappers to finetune, which reduces the computational cost and improves practicality. this design choice suggests that the authors are mindful of real-world applicability.

### Weaknesses
1. **Limited scope of score functions**: the authors focus exclusively on the APS-type score functions. it remains unclear whether the observed entropy-efficiency tradeoff generalizes to other commonly used conformal scores such as 1-p(y|x), which is standard in split-conformal methods. At minimum, empirical evidence across multiple score functions would significantly strengthen the claims. 

2. Theoretical results are again presented only considering the APS score. While generalizing the analysis to other score functions may be challenging, it would be valuable to at least empirically test whether similar behaviors are observed with alternatives. 

3. I found the evidence insufficient for the observed claimed trade-off. More explanations are needed. Particularly the explanation around lines 070-090 and Figure 1 does not convincingly establish the existence of the proposed tradeoff. It would be important to (i) evaluate whether the same trend holds under full model retraining, (ii) demonstrate the empirical tradeoff using other score functions. Moreover in this paragraph I found the explanation confusing for figure 1. does this figure correspond with only fine-tuning using L_class ? if only utilizing the L_class, then there is no term balancing conformal set size and thus its just standard fine-tuning using cross entropy loss ? I would appreciate clarification from the authors. 

4. the alignment between text and figure is sometimes unclear, making it difficult to interpret how empirical results support the theoretical claims.

### Questions
1. **Line 158**: the authors mentions jointly optimizing L_class and L_efficiency. The figure only corresponds to L_class. ( which is standard cross entropy loss). It is possible to train ( even from scratch) using only L_efficiency to obtain smaller sets at comparable accuracy ( thats what conformal training does.) Can you show the results and figure when only using L_efficiency ? 


2.  Figure 1a vs Line 160: In figure 1a, efficiency initially decreases while accuracy increases, yet the text suggests both increase together. Please clarify this discrepancy or update the figure/text for consistency. 

3. Line 071 (CIFAR-100 results): It is counterintuitive that prediction set size increases after applying conformal correction. Could you clarify this behavior?

### Soundness
2

### Presentation
2

### Contribution
2
