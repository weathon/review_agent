# Balancing Interpretability and Accuracy: Energy-Ensemble Concept Bottleneck Models for Enhanced Concept Inference

- Decision: Reject
- Scores: 5, 3, 5, 5

## Abstract
Concept bottleneck models (CBM) have emerged as a promising solution to address the lack of interpretability in deep learning models. However, recent researches on CBM prioritize task accuracy at the expense of interpretability, weakening their ability to accurately infer key concepts. This work addresses this trade-off by introducing the energy ensemble CBM (EE-CBM). The EE-CBM leverages an energy-based concept encoder to effectively extract concepts, overcoming the information bottleneck common in conventional CBMs. Additionally, a novel energy ensemble gate within the EE-CBM architecture efficiently combines energy and concept probability to further address this bottleneck. Moreover, the EE-CBM employs the maximum mean discrepancy loss to enhance concept discrimination within the concept space and facilitate accurate concept inference. An experimental evaluation on benchmark datasets (CUB-200-2011, TravelingBirds, AwA2, CheXpert, and CelebA) demonstrates that EE-CBM achieve state-of-the-art performance in both concept accuracy and interpretability. This work positions the EE-CBM as a significant advancement in CBM researches, enabling them to effectively balance performance and interpretability for improved model transparency. Our code is available at https://anonymous.4open.science/r/EE-CBM-F48D.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces an Energy Ensemble CBM (EE-CBM) architecture that integrates energy and concept probability through an Energy Ensemble Gate (EEG). This model aims to balance task accuracy with interpretability, address information bottleneck issues in CBMs, and enhance the distinctiveness of concepts through the use of MMD loss. The proposed approach is promising but requires clearer exposition and stronger empirical validation to substantiate its claims.

### Strengths
1. The dual-branch architecture is innovative; one branch focuses on concept extraction while the other calculates concept probability, enhancing the model's interpretability.
2. Incorporation of Maximum Mean Discrepancy (MMD) loss to ensure the concepts learned are orthogonal, which is beneficial for model robustness and interpretability.
3. Demonstrated robustness against datasets with significant background variability, which is crucial for practical applications.

### Weaknesses
1. The mathematical notation, particularly in equations 14, 15, and 16, is poorly presented and leads to confusion. The pseudocode and overall technical exposition need significant improvement for clarity.
2. The literature review in Section 2.1 lacks discussion on concept discrimination, despite relevant studies such as those presented in [1]. This omission is a critical gap, especially given past research on concept orthogonality.
3. The discussion related to label-free approaches in Section 2.1 seems misplaced as it does not pertain directly to supervised Concept Bottleneck Models (CBMs), thus diluting the focus of the related work.
4. Claims about improvements in model performance and quantifiable uncertainty by the concept probability branch in Section 3.1 are not substantiated with empirical evidence, contrasting with findings from related work like in [2] Figures 4, 5, and 6.
5. The authors do not demonstrate a significant improvement over existing methods such as those in [3]. The functionalities described could be achieved with simpler architectures (e.g., x-c-y single branch) suggested by prior works, questioning the novelty of the proposed approach.
6. There is a lack of experiments addressing the trade-off between accuracy and interpretability. The experimental design does not adequately highlight any distinctive advantages of the proposed CBM over conventional approaches.

[1] Chen et al. "Concept Whitening for Interpretable Image Recognition", Nature Machine Intelligence, 2020.

[2] Kim, Eunji, et al. "Probabilistic Concept Bottleneck Models." ICML, 2023.

[3] Xu et al. "Energy-based concept bottleneck models." ICLR, 2024.

### Questions
1. **Clarification on Figure 1(c):** The separation of concept $\mathbf{c}$ and label $\mathbf{y}$ in this figure appears contradictory to the joint optimization described in the Energy Concept Bottleneck Model (ECBM). What does $y'$ represent in this context, and why is it depicted as separate from $\mathbf{c}$?
2. **Model Performance versus Concept Accuracy (Line 65):** The authors state that "While these methods can improve label accuracy, often struggle to infer accurate concepts" (Line 065-066). However, these models can achieve high scores in concept accuracy (referenced in Table 1). This seems paradoxical. Can you explain the apparent discrepancy between these observations.
3. **Balancing Task Accuracy and Interpretability:** The related work section critiques large networks for prioritizing classification over interpretability. However, the main contribution of your work claims to balance these aspects. Could you provide empirical evidence, similar to Figure 1(c) from [1], showing how your model achieves this balance?
4. **Sensitivity to Hyperparameters (Section 3.1, Equation 9):** The authors lack detailed ablation studies on the sensitivity of the proposed method to its hyperparameters for each loss component, particularly $\lambda$. The appendix focuses only on the concept loss hyperparameter $\lambda_c$ for each dataset. The authors present results when the EEG and MMD loss weights are set to zero (Table 3), but lacks comprehensive experimental data on how varying these weights might affect the model's performance for each dataset. Could you provide additional experimental results illustrating the impact of increasing these weights? This would help in understanding the robustness and sensitivity of the model to these specific hyperparameters.
5. **Comparison of MMD Loss and Concept Whitening (from [2]):** Given that Concept Whitening aims to make concepts nearly orthogonal, how does MMD loss compare in terms of effectiveness and efficiency in achieving orthogonality among the learned concepts compared with [2]?
6. **Handling Uncertainty in CheXpert Dataset (Section 4):** How does the model address the uncertainty attributes present in the CheXpert dataset? Are there specific techniques or modifications employed that enhance the model's reliability and accuracy in this context? Given that the CUB dataset also contains uncertainty attributes, can you explain why these were not utilized in your experiments?
7. **Lack of Detail on Coop-CBM (Table 1):** Coop-CBM is mentioned without sufficient introduction or referencing. Could you provide a detailed description and relevant citations to clarify its role and significance to use it as your baseline?
8. **Overall Concept Accuracy Not Calculated in [3]:** The manuscript does not report overall concept accuracy, which could be crucial for assessing the holistic performance of concept predictions across a dataset. Why was this metric omitted, and can it be included to provide a more comprehensive evaluation of the model's interpretability?
9. **Interventions on Bottleneck Concepts (Figure 4):** When discussing interventions in the bottleneck, are these applied to groups of concepts or individual concepts? Clarifying this could help understand the granularity and specific impact of interventions on the model's output.
10. **Source of Concepts and Comparison with Similar Methods (Figure 6):** From which dataset were the five concepts selected for analysis in Figure 6? Can you provide a comparative analysis using tsne visualizations against similar methodologies, such as those in [2] and [4] Figure 5, to highlight the distinctions or improvements offered by your approach?

[1] Zarlenga, Mateo Espinosa, et al. "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off." NeurIPS, 2022.

[2] Chen, Zhi, et al. "Concept Whitening for Interpretable Image Recognition", Nature Machine Intelligence, 2020.

[3] Xu et al. "Energy-based concept bottleneck models." ICLR, 2024.

[4] Kim, Sangwon et al. "EQ-CBM: A Probabilistic Concept Bottleneck with Energy-based Models and Quantized Vectors." ACCV, 2024.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces the Energy Ensemble Concept Bottleneck Model (EE-CBM), which aims to improve the balance between interpretability and accuracy in Concept Bottleneck Models (CBMs). The EE-CBM employs an energy-based concept encoder and integrates concept values and probabilities to enhance concept inference and reduce concept uncertainty. The model is evaluated on multiple benchmark datasets and shows state-of-the-art performance in both concept accuracy and interpretability.

### Strengths
- Innovative Approach: The introduction of the energy-based concept encoder and the energy ensemble gate (EEG) is a novel approach to address the trade-off between accuracy and interpretability.
- Strong Results: The experimental results demonstrate that EE-CBM achieves state-of-the-art performance on multiple datasets, showing significant improvements in both concept and task accuracy.
- Energy based model presentation: although the space does not allow for extensive descriptions, the provided succinct description allows understanding the overall idea and functioning of energy-based models without checking the

### Weaknesses
## Major Issues
- **Related work**: The presentation and comparison with existing work are insufficient. The paper lacks a dedicated related work section, and the existing background section and the comparisons provided in the introduction are inadequate.
  - The CEM is likely misunderstood by the authors; it focuses on making task predictions on concept embeddings, not on using two concept representations.
  - The claim that "EE-CBM resolves uncertainty in concept prediction" was already addressed by ProbCBM.
  - The statements about label-free CBMs are questionable. Concepts in these models are explained by their own semantic meaning and heatmaps can be used to explain concept predictions, in both cases just like in standard CBMs.
- **Method Notation** Many notations are not clear or not sufficient in the method presentation:
  - $C’$: the authors define it as: “concept value $C’$ through FC layers as in conventional CBM models”, but then they say that \phi maps to $R^u$ where “$u$ represents the dimension of the concept”, but then again $K$ concept features $C’$ are mentioned. It appears to be a concept embedding, not the concept values of conventional CBM models.
  - Is $\phi$ a per-concept concept encoder? In that case it should have been defined indicized, $\phi_i$ also in the mapping from $R^d$ to $R^u$.
  - The dimensions of $C$ are not specified.
  - The Concept MMD loss is crucial based on ablation study results, but it is poorly presented with statements like "$\mu$ is a kind of mapping function," which lacks scientific clarity. 
- **Metrics to Sustain Claims**
  - “Breaking the information bottleneck”: it is not sufficient to provide high classification accuracy to show that the proposed model breaks the information bottleneck of standard CBM models. The author should also provide the following metrics:
    - The information plane [1] comparing the methods in terms of mutual information between the concept representations (C) and the input (X) and label (Y).
    - Concept efficiency, to test the model performance when reducing the number of concepts as shown in CEM.
  - The “Concept Importance” experiment does not report the concept importance - commonly measured with metric like CaCE [2] to assess how much a task prediction is important for a given task. Instead, the authors only report some qualitative results with the associated concept predictions: all methods achieving good concept accuracy could report the same results.

[1] Tishby, Naftali, Fernando C. Pereira, and William Bialek. "The information bottleneck method." arXiv preprint physics/0004057 (2000).

[2] Goyal, Y., Feder, A., Shalit, U., & Kim, B. (2019). Explaining classifiers with causal concept effect (cace). arXiv preprint arXiv:1907.07165.

## Minor issues
- **Paper presentation**: The introduction section is not well-written, particularly in the second paragraph when introducing related work. Additionally, the acronym EE-CBM is mentioned without being introduced in the third paragraph, and the use of concept embedding is mentioned without prior introduction in the fifth paragraph. The background section on CBM is poorly written and structured, with CBM data and functional representation placed in the middle of the paragraph. The MMD loss ablation study is inserted in section 4.1 before the ablation studies section 4.2. 
- **Validity of experimental results**: There are doubts about the validity of the experiments 
  - Typically Bool CBM an and Fuzzy-CBM performs much worse than CEM or Prob-CBM while in your experiments they show comparable results on both CheXpert and AwA2 dataset. How do you justify this?
  - Experimental settings for compared methods are missing. The training procedures for the compared methods are not reported, even in the appendix.

### Questions
- What is C’? Is it the concept prediction or concept representation/embeddings?
- What are the results of your model when reducing the number of concepts? Does it still provide high classification accuracy? For example, use CUB with only 10 randomly selected concepts for training and inference. This is necessary to demonstrate that you break the information bottleneck.
- Why are CEM and ECBM completely unresponsive to interventions in Figure 4?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper targets strengthening concept accuracy in CBM. To address this issue, the authors propose EE-CBM, which is an approach based on energy. Specifically, EE-CBM incorporates a concept extraction branch and a concept probability branch and applies MMD as a loss to each concept embedding. Lots of empirical experiments show the effectiveness of EE-CBM.

### Strengths
- The paper is well-structured and clearly written, making it easy to follow.
- The experiments include a wide range of baselines and datasets, providing strong validation for the performance of EE-CBM in improving concept and label accuracy.
- The design of concept extraction and concept probability branches is reasonable.

### Weaknesses
- As the author claimed in the introduction section, CEM is proposed to address the trade-off between accuracy and interpretability, which has the same motivation as EE-CBM. Thus, the motivation in this paper is weak and the authors could offer further discussions about what CEM failed to do in addition to the methodology difference.
- Despite that EE-CBM is devised to enhance concept accuracy, the improvement of concept accuracy is extremely incremental as shown in Table 1 compared to ECBM or Prob-CBM.
- The authors could display some wrongly classified samples and their corresponding concept values of EE-CBM and other approaches.
- It seems that the authors mixed the meanings of model interpretability and concept accuracy, and used these two expressions randomly. However, they are absolutely different, so I strongly suggest the authors add a paragraph to explain the relation between model interpretability and concept accuracy.
- Fig. 3 seems to be incomplete with wrong indices.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Energy Ensemble Concept Bottleneck Models (EE-CBMs). EE-CBMs employ a combination of energy-based concept prediction and traditional concept representation prediction to improve the concept predictive performance of CBMs, thereby improving their generalization and downstream task performance. By learning concept embeddings and gating them with their learn probabilities while incorporating a residual channel from the input, EE-CBMs can achieve high concept and task accuracies across several tasks while being receptive to concept interventions and better generalizing across distribution shifts on their inputs.

### Strengths
Thank you so much for submitting this work! I enjoyed reading this paper, learned a lot from it, and appreciate the time taken to write it up and submit it to ICLR. Below are what I believe are this paper’s main strengths:

1. **[Originality] (Critical)** The idea of introducing an energy-based pathway to concept prediction, on top of a standard concept representation learning pathway, is a clear novel use and extension of ideas in previous concept-based models. As such, I believe this work is certainly novel and may be of potential interest to the rest of the community.
2. **[Significance] (Major)** The paper's main purpose, accurately and interpretably predicting concepts and tasks for CBM architectures, is an important and highly active area of research. If it is proven to work as expected, this work has the potential to be impactful.
3. **[Quality and Clarity] (Minor)** The method is very well explained and written. Moreover, the paper is very well placed within the CBM and XAI literature. I would mark this as a major strength if it weren't for the lack of motivation to explain why energy-based prediction is the best way/approach to achieving this paper's goals.
4. **[Quality and Significance] (Minor)** The method is evaluated across a multiplicity of datasets against several key baselines, where it is shown to outperform existing baselines. Therefore, this work provides large amounts of evidence in favor of the proposed method's effectiveness. I would mark this as a “critical” strength if it weren’t for some major concerns I have regarding how some of the baselines may be evaluated (see below).
5. **[Significance] (Minor)** The paper provides the code and configs needed to reproduce the EE-CBM results in this paper. It is therefore taking the necessary steps to ensure reproducibility; however, it could benefit from also including details/code to reproduce the remaining baselines used during evaluation.

### Weaknesses
In contrast, I believe the following are some of this work’s limitations:

1. **[Quality and Significance] (Critical)** I have some major concerns regarding the fairness of the evaluation against existing baselines. These concerns include (1) the fact that some results for some of the baselines seem to **contradict those seen in previous works** (including the original energy-based CBM and CEMs), without any explanation for the discrepancy and (2) the fact that CelebA, a dataset where EE-CBM seems to be underperforming, is, for some reason **pushed to the appendix without any justification or even mention in the main body**. Moreover, given that there is no mention of how hyperparameters were selected for competing baselines, it is very difficult to judge the fairness of the evaluation, even if one is familiar with those baselines. See below for specific questions on these matters.
2. **[Significance] (Major)** EE-CBM requires several hyperparameters to be selected ($\lambda_c$, $\lambda_y$, $\lambda_e$, $\lambda_\text{mmd}$, $\lambda$ for Langevin dynamics, concept embedding size $u$, etc.) yet no recommendations or ablations are provided to understand how these values affect EE-CBM’s performance and its usability. Moreover, it is unclear how the introduction of MCMC or Lavengin dynamics affects the training times of EE-CBM compared to similar baselines.
3. **[Significance and Clarity] (Major)** The motivation behind using a combination of an energy-based pathway and a concept-representation-learning-based pathway for concept prediction is not entirely clear. I can see that it works and improves things; however, this work could be significantly more impactful if it better motivated the need for such a path and built a clear argument as to why it improves things. See below for specific questions on these matters.

### Questions
Currently, I am a bit borderline with this paper’s decision, given some of my concerns with the fairness of its evaluation. However, I am absolutely happy to be convinced that some or all of my conclusions are wrong and to change my recommendation based on a discussion with the authors. For this, the following questions could help clarify/question some of my concerns:

1. **(Critical)** The intervention results in Figure 4 seem a bit surprising, particularly for CEMs. Are you randomly intervening on the concept embeddings when training CEMs (as indicated in the original CEM paper)? If so, do you have an intuition as to why the interventions in CUB look very different to those seen in the CEM paper, its IntCEM follow-up work [1], the original ECBM paper [2], and other previous works (e.g., [3])? If random interventions are not done during training (i.e., *RandInt* is not used as expected), do you have a sense of how results in Figure 4 and Table 1 change for CEMs when CEM's RandInt is used during its training? If no random interventions are performed during training for CEM, at the very least, I would strongly suggest that this work should make it very clear that the used “CEM” baseline is not the same as the one the original work proposed.
2. **(Critical)** More generally, and more importantly, I am concerned with how fairly other baselines were studied during the evaluation. The fact that CelebA, the one dataset where EE-CBM is underperforming compared to other baselines, is *without justification* pushed to the Appendix and not even discussed in the main body should be reason for concern. Can you please elaborate on why results on this dataset were pushed to the Appendix and why they are not discussed in the main body of the paper?
3. **(Critical)** What were the hyperparameter values tested during training for all baselines? These are all missing and not discussed (only EE-CBM’s *selected* hyperparameters are discussed in Appendix B). Were hyperparameters selected based on best validation accuracy or test accuracy? This is not entirely clear in Appendix B, yet it makes a huge difference in terms of the fairness of the evaluation, and it is necessary for reproducibility.
4. **(Major)** Related to the question above, how does EE-CBM’s performance change as one varies its hyperparameters? Given a large number of hyperparameters this model has ($\lambda_c$, $\lambda_y$, $\lambda_e$, $\lambda_\text{mmd}$, $\lambda$ for Langevin dynamics, concept embedding size $u$, etc.), I believe it is key to have this sort of information somewhere in the paper and, at the very least, a guideline on how to select these values in practice.
5. **(Major)** Could you please elaborate on why introducing the energy-based pathway was useful/needed in the first place? I can definitely see that it helps (which is great!), but I believe the motivation for why such a path was needed in the first place is missing in the paper. Is there any way to frame it to make it immediately clear that an energy-based pathway for concept prediction is needed?
6. **(Major)** Related to the previous question: I have some hesitations about the continuous claim in this work that just because a method uses a higher-capacity model, it is less interpretable (see section 2.1 for examples where this claim is made several times). Regardless of how a model generates an explanation (i.e., whether it does this using a white box model or a highly-parametric complex model), if this explanation is (1) accurate, (2) reflective of the downstream task (i.e., it contains all the necessary information to describe the downstream label), (3) composed by human-understandable units of information (i.e., concepts), and (4) actionable (e.g., you can perform interventions or counterfactuals to see how the final decision changes), then I do not see why it matters whether it was generated by a large complex black box model or a simple white box model. Could you please elaborate on why using complex backbones to predict concepts is worse when all of the goals mentioned above, which are the goals of most, if not all, CBM-like approaches, are satisfied? And in that case, why is this argument not applicable to using a complex backbone for EE-CBM’s $f(\mathbf{x})$ function or a complex MLP for its energy function?
7. **(Major)** What is the intuition behind EE-CBM’s better generalization to background shifts? From the text, it is unclear why, intuitively, this must be the case.
8. **(Major)** Do you have a sense as to how this method would perform when dealing with concept incompleteness in the training set? This is a key factor to consider/evaluate if one is to know how this approach can be used in real-world tasks where concept annotations may not be sufficient to explain the downstream task fully.
9. **(Major)** Are concept uncertainty labels used during training (e.g., in CheXpert)? This seems to be implied when talking about Table 1’s results in Section 4.1. However, it is not explicitly indicated or discussed anywhere in the main body of the paper.
10. **(Minor)** Could you please elaborate on the computational training cost of introducing the energy-based pathway in this model?
11. **(Minor)** What does Figure 5 provide that Table 1’s concept accuracy column does not already provide? I might’ve misunderstood something here but I am not entirely sure what the key message of Figure 5 is, as it is unclear how those examples were selected and how that shows that the model truly “understood” a concept (it could just predict a concept’s value entirely from spurious correlations without having to really understand it).
12. **(Minor)** Why is it claimed that EE-CBM is a “concept scalar model” when, in reality, it still generates a high-dimensional concept representation for each concept that is only afterwards gated by the scalar probability? Am I misunderstanding something here?

### Minor Suggestions and Typos

Whilst reading this work, I found the following potential minor issues/typos which may be helpful when preparing a new version of this manuscript:

1. **(Potential Typo)** In line 48, “CEM is modified CBM networks” should probably be “CEM is a modified CBM network”.
2. **(Potential Typo)** In line 262, “… hidden connections between concepts are learned and representation is improved” should probably be something along the lines of “… hidden connections between concepts are learned and their representation is improved”
3. **(Clarity, IMPORTANT)** Is a sentence missing in line 286? It jumps to equation 10 without any preamble or explanation.
4. **(Nitpicking, notation)** In equation (10), it seems that an upper case $\Sigma$ is used for the summation notation rather than Latex’s standard \sum command (e.g., $\sum_{k=1}^K$).
5. **(Clarity)** When talking about high-dimensional concept representations, using the word “concept” to mean both the actual concept and its representation can complicate the reading (e.g., as in Section 3.1 where “concept” is used to mean a concept’s high dimensional representation and the actual concept). Instead, I would suggest using “concept representation” or “concept embedding” when talking about a specific concept’s high-dimensional representation.

## References

- [1] Espinosa Zarlenga, Mateo, et al. "Learning to Receive Help: Intervention-Aware Concept Embedding Models." NeurIPS (2023).
- [2] Xu, Xinyue, et al. "Energy-based concept bottleneck models: unifying prediction, concept intervention, and conditional interpretations." ICLR (2024).
- [3] Collins, Katherine Maeve, et al. "Human uncertainty in concept-based AI systems." *Proceedings of the 2023 AAAI/ACM Conference on AI, Ethics, and Society*. 2023.

### Soundness
1

### Presentation
3

### Contribution
2
