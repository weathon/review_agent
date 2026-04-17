# From Regression to Dose–Response: A Framework to predict Activity and $EC50$ for GPCRs

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
ML models have revolutionised structural biology and significantly advanced drug discovery, yet they struggle with predicting the ligand-induced activity of G-protein coupled receptors (GPCRs). GPCRs are membrane proteins acting as cellular "sensors" which trigger a cascade of intracellular processes upon binding a diverse set of molecules. Human GPCRs account for nearly $30$% of targets of approved drugs, and approximately half of them are olfactory receptors (ORs). Beyond their role in smell perception, ORs are increasingly linked to diseases such as obesity, diabetes, asthma, and cancer. The core interest and difficulty in modelling the molecule-induced response of ORs and GPCRs lie in predicting activity and potency (i.e. half maximal effective concentration, $EC_{50}$). In this paper, we propose a new way of modelling these properties. Instead of direct regression on $EC_{50}$ values, we mimic in vitro dose-response assays by sampling binary activity labels for a protein-molecule pair $(s, m)$ at a molecular concentration $c$. Then we design a novel model that learns the activation probability $P(active|s,m,c)$ at any given $c$. Finally, querying the model across concentrations enables fitting a logistic curve, from which both activity (curve maximum) and $EC_{50}$ (inflection point) are derived. On test sets of $1155$ protein-molecule pairs, our framework improves activity prediction by $10$%  over the state-of-the-art. For $EC_{50}$ estimation, it achieves an error of $0.725$ log units, reducing the error by $40$% compared to a regression baseline and surpassing the affinity module of Boltz-2 by 0.35 log units. Notably, our approach effectively identifies novel active scaffolds, demonstrating its potential to replace expensive in vitro primary screening. The proposed framework is protein-agnostic and can be extended to a broad field of drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a novel framework for predicting EC50 of G-protein coupled receptors (GPCRs).  Instead of treating EC50 prediction as a direct regression problem, the authors reformulate it as a series of binary classification tasks that mimic in vitro dose-response experiments. The core idea is to train a model that predicts the activation probability of a protein-molecule pair at a given concentration.

### Strengths
1. The idea that reformulating the EC50 prediction to a series of binary classification tasks is interesting.

2. Experimental results show that the proposed framework outperforms conventional direct regression approaches for EC50 prediction.

### Weaknesses
1. The validation is primarily conducted on the M2OR dataset. Additional protein targets or datasets should be included to assess the generalizability of the framework, such as BindingDB.

2. The study should include benchmarks against more advanced drug–target interaction prediction methods such as DTIAM[1] and GraphBAN[2].

3. The inference process is inherently more computationally expensive than direct regression. To predict the EC50 for a single protein-molecule pair, the model requires multiple queries across different concentration data points. 

4. Lack of deeper insights into why the proposed framework yields better results than direct EC50 prediction. More experiments or case analyses are needed to uncover the underlying mechanisms.

[1]. Lu, Z., Song, G., Zhu, H. et al. DTIAM: a unified framework for predicting drug-target interactions, binding affinities and drug mechanisms. Nat Commun 16, 2548 (2025). https://doi.org/10.1038/s41467-025-57828-0

[2]. Hadipour H, Li Y Y, Sun Y, et al. GraphBAN: An inductive graph-based approach for enhanced prediction of compound-protein interactions[J]. Nature Communications, 2025, 16(1): 2541.

### Questions
1. Is the proposed EC50 prediction framework applicable to any DTI prediction model? It would be interesting to see if the benchmarked models also benefit from this.

2.  Can the framework be directly applied to IC50 prediction tasks? If validated, this would demonstrate the broader utility and robustness of the approach across different bioactivity endpoints.

3. The choice of threshold $\varepsilon_L$ and $\varepsilon_U$ is not sufficiently justified. It would be important to analyze how varying $\varepsilon_L$ and $\varepsilon_U$ affects performance, whether the optimal value differs across datasets, and if adaptive threshold selection could improve generalizability.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework that reframes EC50 prediction from a direct regression task to a series of concentration-dependent classification tasks, mimicking in vitro dose-response assays. The proposed model, ASMI-DR, learns to predict activation probability at a given concentration. By querying the model across a range of concentrations, a curve is fitted to derive both activity and EC50, reportedly achieving state-of-the-art results on the M2OR dataset.

### Strengths
1. The core idea of reformulating EC50 regression as a classification task that mimics biological assays is highly innovative and provides a simple but effective framework for predicting both activity and potency.
2. The paper demonstrates strong performance with a rigorous evaluation across multiple out-of-distribution scenarios and a wide range of baselines. The paper even conducted a quantitative comparison against a real-world in vitro screening workflow, which adequately demonstrates the model's practical value.

### Weaknesses
1. The model imposes strict requirements on its training data, which can limit its practical applicability. The framework relies on high-quality, complete dose-response curve data, which are more expensive to acquire and are scarce in existing databases (e.g., ChEMBL, BindingDB), to generate its training labels.
2. As shown in this paper, the authors admit that the framework fails when attempting to incorporate abundant single-point screening data.

### Questions
1. The sampling margins and soft label distribution in the training algorithm seem empirical. Could you provide a justification for these choices or a sensitivity analysis?
2. Why does the framework fail when handling abundant screening data? Have you considered alternative sampling strategies or loss functions to address this limitation?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This manuscript proposes a framework for predicting EC50 values and activity
of olfactory receptors from molecular graphs and amino acid sequences.
Instead of directly predicting EC50 and activity,
the authors propose to artificially create binary classification tasks for different concentration levels,
mimicking the underlying experiments that are used to obtain these measurements, more closely.

### Strengths
- The idea to use concentration data for activity prediction is interesting.
 - Overall, the paper has a good reading flow and is understandable.
 - Improvements for the EC50 prediction look impressive.

### Weaknesses
- It is not clear how useful it is to add concentration inputs that are not available in practice.
   On line 143, the authors state that they need to produce surrogate values.
   However, since this surrogate data is obtained from the EC50 values,
   this can not add additional information and typically tends to increase noise in the data.
   It might be that this noise has a regularising effect, but it is hard to verify such hypotheses.
   If the data is obtained by performing this kind of experiments,
   I would expect that it is possible to construct a dataset of activity at different concentration values.
   This should allow to test the quality of the surrogate values and/or my regularisation hypothesis.
   Furthermore, a dataset including concentration values should make the proposed model much more powerful 
   since it could directly incorporate this additional information.
 - I do not understand why the standard paradigm needs to be challenged (as stated on line 80).
   There has been evidence that multi-task learning can benefit tasks in drug discovery (e.g. Mayr et al., 2014),
   but I would have expected EC50 values to follow a Gaussian-like distribution,
   in which case regression is a sensible thing to do.
 - From my experiences, graph neural networks bring little to no benefits when working with small molecules.
   Typically, pre-computed fingerprints work just as well or better for most machine learning tasks.
   It seems like the GNN embedding network could be easily replaced by fingerprints.
 - In general, it is not clear how much of the performance comes from the novel paradigm
   and how much performance is due to the network architecture.
   An ablation study, where the SVM, RF or BiLSTM are used in the new paradigm
   might be helpful to provide some insights here.
 - There is no information on how the hyper-parameters were chosen and what alternative architectures were considered.
 - The results in Table&nbsp;1 indicate that this model is not competitive with an SVM from 2020 for activity prediction,
   even though this SVM model is likely much simpler than the proposed multi-block architecture with GNNs and cross-attention.
   This seems to indicate that this model is not going to be useful for practial activity prediction tasks.
   As a result, the claim in the abstract that the model improves state-of-the-art with 10% on activity prediction is completely unwaranted.
 - The baselines for the EC50 prediction seem to be models mainly optimised for activity prediction.
   Therefore, it is unclear how competitive these baselines are.
   Furthermore, it seems reasonable to expect that a multi-task setup enables reasonable EC50 prediction,
   however, no "plain" multi-task models have been included in the comparison.
   Simply adding a second "head" to predict the EC50 values should give a good idea.
   This would allow the SVM, RF and BiLSTM model to be included as baselines in Table&nbsp;2.
 - It is not entirely clear why the model has only been tested on olfactory receptor data.
   The method seems to be general enough to be applicable to other, more common QSAR datasets (e.g. toxicity, ...).

### minor issues
 - The main text contains plain references to figures in the appendix.
   It would be nice to make clear that these figures are not in the main part.
 - It is unclear what the small values in parentheses represent in Tables&nbsp;1 and&nbsp;2.
   Are these standard deviations, confidence intervals, IQR or something else?
 - The bold numbers in table&nbsp;1 do not make much sense.
   How is SVM not the best model in terms of Recall?

### Questions
1. Do you have any results applying this model to real experiment data?
 2. How well does the surrogate data model the real data?
 3. Are there any arguments to motivate the need to challenge the standard approach?
 4. How do other architectures perform when used in the new paradigm?
 5. What architectures were considered and how were the hyper-parameters chosen?
 6. What is the motivation to use graph neural networks?
 7. How well does the model perform when using molecular fingerprints instead of the GNN embeddings?
 8. What does it mean to improve SOTA with 10% if you can not match the performance of an SVM from 2020?
 9. What are the relevant metrics in Table&nbsp;1 and why?
 10. How competitive are the baselines for the EC50 prediction?
 11. How well does a simple multi-task setup work for combined activity and EC50 prediction?
 12. How does this model on more standard QSAR (e.g. toxicity) prediction tasks?
 13. What are the values in parentheses in Tables&nbsp;1 and&nbsp;2?

### Soundness
1

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
4

### Summary
The paper presents a new framework for predicting GPCR activity and EC50 by mimicking in vitro dose-response assays, thereby capturing the concentration-dependent nature of receptor activation. The authors introduce ASMI-DR, a model that estimates activation probabilities across concentrations, fits a logistic curve, and derives both activity and EC50 from it. ASMI-DR achieves strong performance compared to existing baselines and demonstrates robust generalization to novel molecular scaffolds and receptor variants.

### Strengths
1. The core idea of modeling dose–response relationships through this framework is intuitive and well‑motivated.
2. The experiment results look promising. 
3. The method is clearly written and well-delivered.

### Weaknesses
1. Several baselines are not evaluated under the same conditions as the proposed ASMI-DR model. Boltz-2 and the experimental screening baselines are evaluated on restricted subsets or external protocols. This inconsistency weakens the direct comparability and may create confusion about ASMI-DR’s relative performance.
2. The paper does not clearly describe the training setup for certain baselines, particularly GNN-CLS and MAARDTI.
3. Table 1 contains several missing entries (e.g., precision/recall for SVM, AveP for some models) without any accompanying explanation.
4. The authors could discuss related work, “Predicting Dose-Response Curves with Deep Neural Networks.”





Alonso-Campaña, José, et al. “Predicting Dose-Response Curves with Deep Neural Networks.” Proceedings of the 41st International Conference on Machine Learning, 2024, https://proceedings.mlr.press/v235/alonso-campana24a.html.

### Questions
1. Could the authors clarify whether the train/test splits were stratified to maintain consistent ratios of active and inactive protein–molecule pairs? Given the strong class imbalance in the dataset, ensuring similar label distributions across splits is important for fair and unbiased evaluation.

2. Could the authors include early enrichment metrics, such as EF1%, in their evaluation? This would better capture how effectively the model ranks true actives near the top of the prediction list, which is particularly relevant for virtual screening scenarios.

### Soundness
2

### Presentation
3

### Contribution
2
