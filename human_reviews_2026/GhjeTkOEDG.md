# CountTRuCoLa: Rule Learning for Explainable Temporal Knowledge Graph Forecasting

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4

## Abstract
We address the task of temporal knowledge graph (TKG) forecasting by introducing a fully interpretable method based on temporal rules. 
Motivated by recent work proposing a strong baseline using recurrent facts, our approach learns four simple types of rules with a confidence function that considers both recency and frequency.
Evaluated on nine datasets, our method achieves performance that is competitive with state-of-the-art models and consistently outperforms the majority of them, while providing fully interpretable predictions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CountTRuCoLa, an explainable rule-based method for forecasting future relationships in temporal knowledge graphs. Inspired by a repetition-based baseline, it uses four simple rule types (e.g., repeating entity-relation patterns, cross-relation links with common entities) and a confidence score that considers how recent and frequent past events were. It combines top rules’ predictions to rank results, and tests on nine datasets show it performs well, avoids memory/time issues with large graphs, runs efficiently on CPUs, and lets users trace predictions to specific rules and past events—proving simple, explainable methods can match complex neural models.

### Strengths
1. The proposed method is fully interpretable, allowing every prediction to be traced back to specific temporal rules and observed facts—an important contribution to transparency in TKG forecasting.

2. Despite its rule-based simplicity and CPU-only implementation, the model achieves competitive accuracy and superior scalability compared to complex neural or reinforcement learning models.

3. The authors conduct extensive experiments on nine datasets, including ablation studies that clearly demonstrate the contribution of different rule types and confidence components.

### Weaknesses
1. The paper mainly extends existing rule-based and recurrence-driven approaches without introducing new ideas in explainability. It focuses on weighting known temporal patterns via confidence functions rather than advancing interpretable modeling, such as complex or compositional rule reasoning.

2. The work lacks thorough ablation and sensitivity analyses of the confidence function. The model relies on parameter tuning across numerous rule instances rather than uncovering consistent or generalizable temporal regularities, weakening its explainability claim.

3. The experimental evaluation is not sufficiently up-to-date. Only one 2024 baseline method, Rec.B, is included, while recent baselines are missing and should be compared.

4. The paper does not report the Hits@1 and Hits@3 results, which are widely used evaluation metrics in TKGR research.

### Questions
See the weakness.

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
This paper presents a fully explainable temporal knowledge graph forecasting method that learns four simple types of temporal rules, leveraging a confidence function based on recency and frequency; extensive evaluation on nine datasets shows that the approach delivers competitive and often superior performance compared to state-of-the-art models, while maintaining interpretability in its predictions.

### Strengths
1. The rule-based approach enables explicit explanations for each prediction.

2. The paper's motivation is clear and the writing is sufficiently detailed. Each parameter in the confidence function has a clear, intuitive interpretation.

3. The experiments are comprehensive, validating the method’s effectiveness on nine datasets.

### Weaknesses
1. Although the paper defines four rule types, the definitions and examples appear to lack consideration of multi-hop rules.

2. The z-rules and f-rules are static and do not incorporate any temporal dynamics, which may be a poor fit for datasets where the frequency distributions of entities or relations change significantly over time

3. In Equation (12), the functional form of $\(f_r\)$ appears ad hoc, lacking derivation from first principles or probabilistic models. Its selection is not justified, and it is unclear why this form is preferable to simpler exponential decay.

4. The paper introduces numerous hyperparameters, both trainable and non-trainable. However, the necessity of so many parameters remains questionable due to a lack of supporting experiments.

5. Although experiments are conducted on multiple datasets, the authors do not report Hits@1 and Hits@3 results. These metrics, particularly Hits@1, can be critically important in some cases.

### Questions
See Weakness

### Soundness
3

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
The paper presents CountTRuCoLa, an interpretable rule-based framework for temporal knowledge graph forecasting. The method learns four families of temporal rules and equips each rule with a parametric temporal confidence that jointly captures recency and frequency of supporting events. At inference, predictions from multiple rules are fused via a truncated, decayed noisy-OR aggregator over the top-H rule confidences. The approach targets single-step extrapolation, runs efficiently on CPU, and ships an explainer that attributes each prediction to concrete rules and their fitted confidence curves. Experiments across a broad suite of benchmarks report competitive or state-of-the-art MRR together with favorable runtime and memory behavior.

### Strengths
1.	Interpretability and transparency – Every prediction can be traced back to concrete symbolic rules and corresponding confidence curves. The use of four rule types enables fine-grained temporal reasoning while retaining human interpretability.
2.	Principled temporal modeling – The confidence function decomposes temporal influence into a recency component and a frequency component, each with distinct, interpretable parameters (α, λ, ϕ, ρ, κ, γ). This formulation clearly captures short- and long-term dynamics.
3.	Sound aggregation mechanism – The top-H decayed noisy-OR aggregator is a reasonable balance between simplicity and partial dependence modeling. Its hyperparameters (H, D) are explicitly defined and empirically validated.
4.	Comprehensive empirical evaluation – Nine datasets, extensive ablation studies, and runtime analyses are provided. CountTRuCoLa achieves strong predictive accuracy on both small and large graphs without GPU reliance.
5.	Explainability tool – The accompanying explanation interface visualizes rule activations, enabling comparison with neural baselines and facilitating post-hoc analysis of temporal dependencies.
6.	Reproducibility and documentation – Code, datasets, and appendices (A.1–A.9) offer complete transparency regarding data collection, hyperparameter tuning, and runtime environments.

### Weaknesses
1.	Lack of formal definition for “explainability” and rule faithfulness.
The paper claims interpretability through rule-level explanations and visualized confidence curves (Sec. 6), yet does not quantitatively measure whether these explanations are faithful to the prediction process. There is no evaluation of how rule confidence aligns with contribution scores or ground-truth relevance. A simple correlation or fidelity metric would help validate that interpretability is not merely descriptive.
2.	Counterfactual and causal reasoning terminology ambiguity.
Although the title and abstract emphasize “reasoning,” the framework is purely observational—confidence estimation depends only on empirical Δ-lag statistics (Sec. 4.2). There is no notion of intervention, counterfactual simulation, or temporal causation. Clarifying the term “reasoning” to mean “pattern-based temporal inference” would avoid conceptual confusion.
3.	Two-stage confidence fitting without sensitivity analysis.
In Eq. (14), the confidence function is decomposed into recency and frequency components fitted sequentially. This procedure lacks quantitative analysis of whether the order of fitting (recency-first) affects the balance between short- and long-term contributions. The absence of a sensitivity or ablation study limits understanding of parameter robustness.
4.	Simplifying assumptions in the noisy-OR aggregation.
The truncated decayed noisy-OR (Sec. 4.3) aggregates top-H rule confidences under an implicit conditional independence assumption. The paper acknowledges this simplification (last paragraph of Sec. 4.3) but does not empirically test its effect—e.g., whether correlated rules inflate probability estimates. A diagnostic example or comparison with a logistic combiner could strengthen this section.
5.	Limited evaluation horizon.
Experiments (Sec. 5) only cover single-step extrapolation. While this setting is common in temporal KGs, the paper does not analyze how rule confidence behaves under longer horizons or sparser timestamps, leaving uncertainty about the model’s stability for multi-step forecasting.

### Questions
1.	Confidence function shape: In Fig. 4, learned confidence curves exhibit various non-monotonic shapes. Could you characterize the conditions under which these curves are monotonic or peaked, and how such shapes affect prediction reliability?
2.	Sensitivity to H and decay parameters: The noisy-OR aggregator uses truncation parameter H and decay D (Sec. 4.3). Have you explored how varying these parameters impacts both MRR and rule interpretability? For instance, does larger H lead to overcounting correlated rules?
3.	Rule family contributions: The paper defines four rule types (xy, c, z, f). Could you report which families contribute most to top-ranked predictions across datasets? An aggregated table of rule coverage or confidence variance would clarify their relative importance.
4.	Temporal granularity: Since datasets such as ICEWS and GDELT have different time resolutions, how does the choice of window W affect the fitted recency decay parameters? Is there evidence that these parameters are transferable across datasets?
5.	Scalability breakdown: Can you provide more detailed timing results—such as per-dataset CPU time for rule extraction and fitting—to better substantiate the “CPU-efficient” claim?

### Soundness
3

### Presentation
3

### Contribution
3
