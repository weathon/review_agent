# Additive Coupling of Liquid Neural Networks and Modern Hopfield Layer for Regression

- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Regression tasks on complex datasets often involve diverse feature interactions, long-range dependencies, and structured patterns that must be recalled across examples for accurate prediction. Conventional models—such as MLPs, tree ensembles, or standard continuous-time networks, struggle to maintain predictions and stability over extended horizons, especially when patterns must be reused. To address these challenges, we introduce a hybrid architecture that couples Liquid Neural Networks (LNNs) with Modern Hopfield Networks (MHNs) using additive fusion. The LNN component delivers input-adaptive continuous dynamics, while the associative memory enables retrieval and correction using previously encountered global structures. This biologically-inspired design preserves adaptability and stability, while leveraging memory-based recall for consistent predictions. On the OpenML-CTR23 regression benchmark, our approach consistently improved performance, with mean and median gains of 10.42\% and 5.37\%.  These results demonstrate the effectiveness of integrating continuous dynamics and content-addressable memory for complex regression scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a hybrid regression framework that combines the adaptive dynamics of Liquid Time-Constant (LTC) networks with the associative memory of Modern Hopfield Networks (MHN). The LTC component models input-dependent continuous-time dynamics, allowing each neuron to adapt its response to varying inputs, while the MHN compensates for LTC’s lack of an explicit memory mechanism by retrieving patterns from a set of stored prototypes. The two components are integrated through an additive coupling scheme that balances dynamic evolution and memory recall, followed by a lightweight MLP regression head. The authors evaluate the approach on the CTR23 benchmark, which includes 34 tabular regression datasets, and report consistent performance gains over both classical (XGBoost and Random Forest) and neural baselines (vanilla LTC).

### Strengths
This paper provides an extensive and well-structured experimental evaluation, demonstrating that the proposed model improves performance on 29 out of the 34 datasets in the CTR23 benchmark. The benchmark itself covers a diverse range of regression tasks, making the evaluation comprehensive. The authors include clear and insightful visualizations, particularly the parity and loss-landscape plots. that effectively illustrate both the strengths and the limitations of their method. Their inclusion of less favorable cases, such as on the Wave Energy dataset, strengthens the credibility of the results by avoiding selective reporting. Finally, the paper’s discussion of limitations is particularly commendable, showing that the authors critically assessed the behavior of their model and acknowledged scenarios where it may not perform as well.

### Weaknesses
While the paper presents a coherent hybrid framework, the overall novelty is somewhat limited, as it primarily combines two existing mechanisms, Liquid Time-Constant networks and Modern Hopfield Networks, in a straightforward additive manner. The methodological contribution lies more in the integration than in introducing new concepts. Empirically, the performance gains are modest, with improvements in RMSE that, while consistent, are relatively small across most datasets. The paper would benefit from a more thorough hyperparameter sensitivity analysis, especially regarding key design choices such as the Hopfield memory size (set to 16) and the retrieval scaling factor (β = 0.25). It remains unclear how these values were selected or how they affect performance and stability. Additionally, there is no discussion of computational cost or efficiency, which is important given the added memory and retrieval operations of the Hopfield component. Comparing training and inference times against baselines would have provided a more complete picture of the trade-offs. Finally, there are minor writing and formatting inconsistencies, including grammatical errors, such as in the second paragraph of Section 4.2, and occasional awkward phrasing, which slightly detract from the overall readability of the paper.

### Questions
1.	What is the common practice for setting hyperparameters for CTR evaluations?
2.	For clarity and consistency, it would be helpful to format Table 2 in the same way as Table 1, for instance, by bolding the best RMSE and underlining the second-best result.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes a hybrid architecture for structurally complex tabular regression data. It uses Liquid Neural Networks (LNNs) to process continuous-state data adaptively. To address the fact that LNNs are inherently local in time and feature space, the authors apply Modern Hopfield Networks (MHNs) for memory retrieval. They then use simple additive coupling to combine these two parts. Evaluation on OpenML-CTR23 shows that each part of the architecture is necessary and effective, as claimed.

### Strengths
1.	The idea is simple and direct: use an LNN for continuous data processing and add an MHN to provide memory retrieval.
2.	The paper presents lemmas establishing boundedness, implying smooth gradients and overall system stability.
3.	The ablation study on the additive LNN+MHN design shows each component works as intended; the MHN indeed supplies memory and is essential.
4.	On OpenML-CTR23, the architecture outperforms competing methods.
5.	The writing is clear and easy to follow.

### Weaknesses
1.	Comparisons to related OpenML-CTR23 work. As noted in the Introduction, this paper also reports results on OpenML-CTR23. Please add direct comparisons to the methods cited there (e.g., ‘Accurate predictions on small data with a tabular foundation model’) or briefly justify why such comparisons are not appropriate.
2.	Compute/efficiency reporting. Please add a small table like params or FLOPs.
3.	In Figure 2, the legend refers to orange tracks, but none are visible.



Noah Hollmann, Samuel Muller, Lennart Purucker, Arjun Krishnakumar, Max K ¨ orfer, Shi Bin Hoo, ¨
Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular
foundation model. Nature, 637(8045):319–326, Jan 2025. ISSN 1476-4687. doi: 10.1038/
s41586-024-08328-6. URL https://doi.org/10.1038/s41586-024-08328-6.

### Questions
1.	Is there a formal reproducibility statement? Any plan to release code and model checkpoints?
2.	Please include an LLM usage statement (per venue policy).
3.	Additional comments: I am not familiar with this topic. The paper is clearly written and, on that basis, I am inclined to give a high rating, but with low confidence. My rating may change during the rebuttal stage.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a hybrid regression model that couples Liquid Time-Constant (LTC) networks with Modern Hopfield Networks (MHNs) through a simple additive fusion mechanism. Tested on the OpenML-CTR23 benchmark of 34 regression datasets, the proposed model achieves consistent gains—averaging a 10.4% reduction in RMSE over baselines, including XGBoost and vanilla LNNs. Theoretical proofs demonstrate bounded, stable dynamics and smoother gradient behavior.

### Strengths
1. Lemmas 1–3 are clearly stated, which makes it easy for me to follow, even for readers who are not specialists in this field.
2. The experiments on the CTR23 datasets demonstrate the robustness of the proposed method and show that it outperforms baselines in most scenarios.
3. The proposed model inherits the strengths of LTC in terms of local-region regression and stability.

### Weaknesses
1. In Lemma 4, the authors assert that the linear combination of gradients can **reduce variance and aid convergence**. However, they should provide either theoretical proof or empirical evidence to support this claim.
2. The authors do not report standard deviations ($\pm$ std) or confidence intervals for their experimental results.
3. A key limitation of continuous-time regression models is **temporal generalization**—i.e., the model is trained on a training set whose time range does not overlap with that of the test/validation set. Otherwise, the task is similar to the standard function fitting on scatter plots. In the conclusion section, the author said they overcame the accumulation of errors over long horizons. Did they refer to the temporal generalization? However, I cannot find a description of how to split the training and test sets to ensure that the time ranges are non-overlapping.

### Questions
1. The authors compare their proposed method with traditional machine-learning regressors (XGBoost, Random Forest, Ridge Regression, etc.). Have you considered comparisons with state-of-the-art neural network–based regression models?

2. When comparing the proposed method with other models, you should evaluate not only the MSE metric, but also report the computational complexity, runtime (wall-clock time), and possibly parameter efficiency.

3. In Figure 2, I cannot find the orange tracks mentioned in the caption. Is this a typo or a missing step to color the curves?

4. In Figure 3, the authors state that the proposed model exhibits broader and smoother basins on the Brazilian Houses dataset. However, the y-axis units differ: the LTC plot uses a scale of 1, while the proposed model’s plot uses $10^4$. Does this mean that LTC is actually smoother in this visualization?

5. The experiments focus on low-dimensional real-world datasets. Have you tested the model on multi-frequency or high-dimensional synthetic examples, such as $y = \sin(1/t)$ or $y = \sum_{i=1}^{100} \cos(t_i)+\sin(t_{101-i})$? It is a challenge for most neural network-based models.

6. The author can explore the temporal generalization of their proposed model.

### Soundness
3

### Presentation
3

### Contribution
2
