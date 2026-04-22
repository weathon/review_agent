# Controlling Structured Explanations via Shapley Values

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Structured explanations elucidate complex feature interactions of deep networks, promoting interpretability and accountability. However, existing work primarily focuses on post hoc diagnostic analyses and does not address the fidelity of structured explanations during network training. In contrast, we adopt a Shapley value-based framework to analyze and regulate structured explanations during training. Our analysis shows that valid subexplanation counts in structured explanations of Transformers and CNNs strongly correlate with each model's feature interaction strength. We also adopt a Shapley value-based multi-order interaction regularizer and experimentally demonstrate on the large-scale ImageNet and fine-grained CUB-200 datasets that this regularization allows the model to actively control explanation scale and interpretability during training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to first connect interaction Shapley values to "explanation scale", which is based on the number of explanations of different sizes, a specific prior explanation method.
After making this connection empirically, the paper analyzes a few architectures, particularly CNN and transformer based methods on their Shapley interaction values.
Like prior works, they note that CNNs focus on small details while transformers focus more on compositional structures.
The paper then suggests using previously proposed loss functions to train in tandem with the normal objectives to modify the model behavior.
The paper provides experiments showing they can modify the explanation behavior of models using these loss functions.

### Strengths
- The paper finds an interesting connection between interaction Shapley values and "explanation scale".
- The method does not require computing Shapley values or interaction strength directly.
- The paper demonstrates that it can modify the behavior of models using explanation regularizations.

### Weaknesses
- "Importantly, our goal is not to characterize or optimize the accuracy–explainability tradeoff, but to show how explanation structure can be systematically regulated while maintaining competitive accuracy, leaving a study of such tradeoffs to future work." This seems like an important qualification. But why do we want to regulate explanation structure if not for this type of trade-off? Merely showing that it's possible is not insightful unless it is *useful*.
  - "the value of controllability lies in adapting explanation structure to deployment needs, as prior work shows different structures can enhance robustness, modularity, or usability, and also improve user understanding and counterfactual reasoning". This suggests the potential benefits of controllability. However, none of these benefits are demonstrated. Only qualitative assessment of various explanations of model behavior. I would like to see results that clearly show this controllability enables one or more of these benefits for the task that is considered (when compared to relevant baselines for the task).

- The paper seems to follow Deng et al., 2022 a fair bit. And many of the sections have background and proposed methods interleaved. This makes it more difficult to isolate the contributions of this paper. Could you clearly mark all sections that are background? Ideally, all background would be in one section so that it is clear what is new and what is prior work. For example, though one of the main claims is about controlling explanation scale, the loss functions are from Deng et al. 2022. Thus, it is unclear what is novel here.
  - Overall, I would like a clearer separation. It seems that the main contribution is showing the connection and then using it to modify the model. But the actual loss functions and such are actually from a different paper.

- Computing intereaction strength reliably seems very difficult given that even Shapley values are hard to compute. Essentially, the variance of the estimators can be high given a small number of samples. Could you explain why the number of samples you chose was sufficient for reasonably accurate estimation?

- The correlation between subexplanation count and interaction strength doesn't seem particularly strong. Could you provide Pearson's, Spearman's and Kendall's Tau correlation values?

- "Explanation scale" is not a well-known term and needs to be at least intuitively explained  in the introduction. Also, the idea of "structured explanations" needs at least an example in the introduction as this is not a well-known term. I'm also a little concerned that this explanation type is fairly narrow and thus the impact of the paper is fundamentally based on this explanation method, which is not mainstream.

### Questions
- Is the connection between explanation scale and interaction strength surprising or unexpected? Could you give more discussion on this point, which is one of your main contributions? It seems that this is very natural. Are you merely saying that you give evidence for this or is there something more insightful that I'm missing?

- What are other ways to modify the behavior of a neural network? This seems to be the primary claim but it does not seem to be compared to alternatives in the prior work or other alternatives? It is hard to know if the proposed method is "good" at modifying the behavior if there is no comparisons.
  - For example, how does this compare to Shapley Explanation Networks [Wang et al., 2021]? This seems to be a related work specifically since it is used to control the training of the architecture but was not discussed in related works.

Wang, R., Wang, X., & Inouye, D. (2021, May). Shapley Explanation Networks. In International Conference on Learning Representations.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Shapley-value-based method for providing structured explanation for neural network. The key advantage is to incorporate interaction-based regularization into training. It considers multi-order feature interaction strength for explanation structure, enabling models to be trained for desired reasoning behaviors

### Strengths
- It proposes a novel approach for incorporating interaction interactions during training
- The evaluation was performed across multiple model types and datasets.

### Weaknesses
- Compromised accuracy would be barrier for practical adoption of this method.
- Trade-off introduced by the method, such as stability vs storage, can be hard to tune in practice without rigorous analysis each time.

### Questions
- Can this regularization method enhance model robustness or out-of-distribution generalization?

### Soundness
2

### Presentation
2

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
The paper proposes a method to control the structured explanations obtaining during network training by introducing a framework regularization based on Shapley values. The authors analyze Transformer and CNN architectures using this framework.

### Strengths
- The use of  Shapley values to encode varying levels of feature dependence providing greater control over how the network leverages the features.
- The experiments with ProtoTree that demonstrate  the proposed regularization substantially improve interpretability.

### Weaknesses
- **Motivation for control:** I like the idea of having control over interactions, but I’m not sure why this is necessary. What is the underlying motivation for introducing this control? As you mentioned, if there isn’t a clear notion of an “optimal” level, what problem does this control solve?
- **Discussion on oversimplification:** You mention that explanations can be oversimplified, and I agree that neural networks are inherently complex. However, increasing the complexity of explanations does not necessarily improve interpretability. In fact, one could say that providing more distinct important regions can make interpretability more difficult.
- **Clarification of SAG visualizations:** I’m having trouble interpreting the visualizations in Figure 4 — the highlighted regions (red segments) look almost identical across multiple repetitions. Since SAG explanations appear central to your analysis, it would be helpful to provide more details on how these visualizations should be interpreted.

### Questions
- **Experiment in Table 2:** I found the experiment in Table 2 interesting. Which dataset was used? Do you observe similar trends across other datasets? Also, what is the rationale for focusing on reducing the importance of localized regions? Why aim for sparse explanations? While you propose an additional level of control, could this potentially make interpretation more difficult rather than easier?
- **Patch features:** Since patches are used as features, does the patch size affect the results? Would it be feasible to use features corresponding to human-interpretable concepts instead?
- **Effect of regularization on high-order interactions:** Why does regularizing to reduce high-order interactions help ProtoTree focus on the foreground? What would happen if the regularization were applied in the opposite direction, for instance to reduce the L+ term?
- I also include some questions above.

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
2

### Summary
The paper introduces a Shapley value–based interaction regularization that integrates interpretability objectives into model training. The method enables controllable shifts in reasoning style and explanation scale across architectures such as CNNs and Transformers.

### Strengths
- The paper provides interesting empirical evidence that CNNs exhibit more disjunctive (localized) reasoning, while Transformers demonstrate more compositional (holistic) reasoning. This comparison helps clarify architectural differences in how models integrate features and reason over spatial contexts.
- By incorporating Shapley-based interaction regularizers into the training objective, the paper moves beyond post hoc analysis and offers a method to influence interpretability during learning.
- The experiments show that adjusting the regularization strength can shift models between holistic and localized reasoning, illustrating that the method offers controllable interpretability behavior rather than fixed outcomes.

### Weaknesses
- The paper claims to "establish a theoretical connection" (line 43) between subexplanation counts and Shapley-based interaction strength, but the evidence presented appears to be empirical and very weak.
- The novelty appears limited. The proposed multi-order interaction regularizer completely follows Deng et al. (2022). The main contribution seems to lie in the empirical observation of differences in interaction structures between CNNs and Transformers, and this work looks like a simple combination of Jiang et al. (2024) and Deng et al. (2022).
- The paper could benefit from improved clarity and explanation of figures. For readers less familiar with the specific background, the lack of a comprehensive description makes it difficult to understand how the figures are drawn. This makes it challenging to verify the authors’ claims.
- The paper presents an internal inconsistency regarding the role of interaction-based regularization. In lines 407–409, the authors state that the proposed method merely provides controllability over the model’s reasoning behavior "without implying that any particular style is inherently superior or more interpretable." However, in lines 455–457, the authors claim that their regularization leads to more semantically meaningful representations and improves the accuracy of explanations.

### Questions
- Could the authors clarify the exact meaning of the axes in Figure 2? In addition, I noticed that the ranking of models appears to vary across different orders. Does this variability suggest any limitations in the stability or reliability of the proposed interaction-based evaluation？
- In Eq. (4), the computation of $U_c(k_1, k_2)$ involves an expectation over masked subsets. However, the paper states that the training loss "only involves an additional forward–backward pass on two masked variants of the input." Could you clarify how this expectation is handled during optimization?
- The paper uses different ranges of $k$ across figures (e.g., 0.05–0.5 in Figure 1, 0.1–0.4 in Figure 2, and >0.5 in Figure 3). Could the authors clarify how these $k$ values are chosen in practice?
- I am somewhat confused by the discrepancy: Figure 4 suggests that CNNs mainly focus on localized regions, which would imply stronger interactions at lower $k$. However, in line 300 and Figure 3, the interaction strength for CNNs appears concentrated at higher $k$. Could the authors reconcile these observations or explain whether the definition or scaling of k differs across figures?

### Soundness
3

### Presentation
2

### Contribution
2
