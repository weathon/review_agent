# Hierarchy Pruning for Unseen Domain Discovery in Predictive Healthcare

- Avg Score: 4.40
- Decision: Reject
- Scores: 2, 4, 6, 6, 4

## Abstract
Healthcare providers often divide patient populations into cohorts based on shared clinical factors, such as medical history, to deliver personalized healthcare services. This idea has also been adopted in clinical prediction models, where it presents a vital challenge: capturing both global and cohort-specific patterns while enabling model generalization to unseen domains. Since cohort boundaries naturally correspond to domain boundaries, addressing this challenge falls under the scope of domain generalization (DG), especially when domain labels are not explicitly available in EHR data. However, regular DG approaches often struggle in clinical settings due to the absence of domain labels and the inherent gap in medical knowledge. Moreover, the rich hierarchical structures embedded in medical ontologies have rarely been explored as a basis for deriving clinically meaningful domain partitions. Hence, we propose UdonCare, a hierarchy-guided method that iteratively divides patients into latent domains and decomposes domain-invariant (label) information from patient data. On two public datasets, MIMIC-III and MIMIC-IV, UdonCare shows superiority over eight baselines across four clinical prediction tasks with substantial domain gaps, highlighting the potential of medical knowledge in guiding clinical DG problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
UdonCare attempts to improve accuracy with various medical predictions by using domain generalization techniques to better predict across a wide range of conditions.

### Strengths
- Evaluations on MIMIC-III and MIMIC-IV are good
- AUROC and AUPRC are really good metrics for model evaluation
- Task selection is also good. Mortality, Readmission, Drug Recommendation, and Diagnosis Prediction are all important.

### Weaknesses
- The list of baselines is very very incomplete. In particular, this paper is critically missing the most important structured EHR data baseline: gradient boosted trees (either through xgboost or lightgbm). I would recommend reject purely for that missing baseline alone as it is such an important baseline in EHR settings. 
- Time splitting is a good approach, but the way it is done in this paper is incorrect. This paper uses as the test set all patients with a last visit after 2017 and excludes them from the training set. This is wrong. The visits for these patients which were before 2017 should be in the training set. The reason for this is because the train test split needs to approximate a realistic train/test deployment in the model. We right now don't know which patients are going to get visits in the future so we can't exclude them from the training set for a deployment right now. So you shouldn't exclude patients with future visits from a model backtesting procedure.
- The base models getting worse than random accuracy on AUROC for readmission prediction is very very suspicious, and I think an indication that something went wrong with your setup. This might be because I didn't understand your target data vs source data distinction. What is your target data and what is your source data? Is it the time split.
- The source and target domains for table 2, your main experiments. I really confusing. What they precisely?

### Questions
I just want to confirm: the "target data" in your experiments section is the 2017 data? And the "source data" is the earlier data? Or am I misunderstanding something?

And this target data is unused for all models (except oracle), including your domain generalization models?

### Soundness
1

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
4

### Summary
This paper investigates how hierarchical medical knowledge can be effectively incorporated into the problem of domain generalization. To this end, the authors introduce UdonCare, a hierarchy-guided framework that iteratively partitions patients into latent domains for domain discovery and disentangles domain-invariant representations from patient data. The proposed framework is empirically evaluated on the MIMIC-III and MIMIC-IV datasets, demonstrating both its effectiveness and computational efficiency.

### Strengths
**S1.** The paper tackles a practically important and clinically meaningful problem by investigating how hierarchical medical ontologies can be incorporated into clinical domain generalization (DG) tasks.

**S2.** The proposed UdonCare framework integrates medical ontologies into an iterative domain discovery process. Its design involves a hierarchy-guided domain discovery algorithm, along with distinct domain and backbone pathways that jointly contribute to disentangling domain-invariant representations from patient data.

**S3.** The experimental evaluation includes comparisons with established baseline models, analyses of the decomposition effectiveness, ablation studies, examinations of training data size effects, and runtime analyses, thereby providing a reasonably thorough empirical validation of the proposed framework.

### Weaknesses
**W1.** The paper motivates its contribution by highlighting the limited attempts to integrate medical knowledge into DG settings, despite existing works on knowledge-driven predictive modeling and clinical DG methods. While this motivation is reasonable, the paper would benefit from a clearer articulation of the technical challenges involved in bridging these two research lines. Specifically, it should elaborate on what makes the incorporation of hierarchical medical knowledge into DG settings non-trivial and how UdonCare effectively addresses these challenges. Without such clarification, the methodological novelty and technical depth of the proposal appear somewhat limited, particularly since UdonCare relies heavily on established techniques in its framework design.

**W2.** The current scope of UdonCare is limited to modeling relational clinical data, such as diseases, procedures, and medications, within binary classification settings. It would strengthen the paper to discuss the framework’s potential generalization to other data modalities and to broader predictive tasks beyond binary classification.

**W3.** The evaluation of UdonCare is conducted solely on the MIMIC-III and MIMIC-IV datasets, which are relatively homogeneous even after removing overlapping time ranges during preprocessing. To more rigorously assess the generalizability of the proposed framework, it would be beneficial to include experiments on additional datasets with greater heterogeneity, such as those adopted in ManyDG or SLDG.

**W4.** Since the primary objective of UdonCare is domain discovery aimed at enhancing the performance of domain generalization tasks, it would be valuable to provide a deeper analysis of the discovered domains. In particular, interpreting these latent domains through the lens of medical ontology knowledge could yield meaningful insights into their clinical relevance and the underlying patterns captured by the framework. Such analysis would not only substantiate the effectiveness of the proposed discovery process but also strengthen the interpretability and practical utility of the proposal.

**W5.** The overall presentation of the paper could be improved in several aspects to enhance readability and coherence:

* **Abstract:** The connection between cohort modeling and DG should be articulated more explicitly to clarify the paper’s conceptual motivation.

* **Organization:** It would aid readers’ comprehension if the related work section, particularly the discussion of existing DG methods, could be placed earlier in the paper.

* **Preliminaries:** A concise yet precise formulation of the DG problem should be introduced in the preliminaries to provide a clear foundation for the subsequent methodology.

* **Methodology:** The methodological exposition could be streamlined by defining the core mechanisms of UdonCare as distinct modules and elaborating on them in a structured, module-wise manner.

* **Notation:** Given the large number of symbols and variables used, including a notation table summarizing the key notations would substantially improve clarity.

### Questions
Beyond W1-W5, I have the following questions for clarification:

**Q1.** The underlying rationale behind the node scoring formulation in Equation (5) should be explained in greater detail.

**Q2.** In Section 4.1, the criteria “$d < 120$” and “$d > 4500$” are mentioned. Could the authors clarify whether this variable $d$ corresponds to the same linear classifier described in Section 4.3? If not, the distinction between the two should be made explicit to avoid confusion.

**Q3.** In the experiment assessing the effectiveness of decomposition (Table 3), it is stated that the first two rows exhibit higher similarities. However, the second row seems to display a notably lower similarity than the first. Additional clarification is needed on this phenomenon and how these results support the conclusions drawn from this experiment.

### Soundness
3

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
3

### Summary
This paper tackles the critical problem of domain generalization (DG) in clinical prediction, where models trained on Electronic Health Records (EHR) from one population or time period fail when applied to another. The authors argue that standard DG methods are ill-suited for healthcare because EHR data lacks the explicit domain labels found in other fields, and methods that simply cluster patient features ignore the rich clinical semantics that define patient cohorts. To solve this, the paper proposes UDONCARE, a novel framework that, for the first time, uses medical ontologies to actively discover latent, clinically meaningful domains. The framework operates in two main steps: first, a hierarchy-pruning algorithm analyzes the disease hierarchy to identify an optimal level of abstraction, merging overly specific leaf nodes (e.g., "congestive heart failure") into more general, robust ancestor nodes (e.g., "cardiovascular disease") based on node scores and a beam search. Overall result are satisfied.

### Strengths
1. The paper's core strength is its novel use of medical ontologies to solve a DG problem. Instead of treating domains as arbitrary, data-driven clusters, it grounds them in established medical knowledge. This makes the discovered domains more interpretable and robust.
2. The hierarchy-pruning algorithm is a non-trivial and well-designed contribution. It correctly identifies that neither leaf-level codes nor root-level categories are optimal, and it provides a principled method to find the best-fitting level of abstraction. The ablation study in Figure 3 confirms this custom pruning method is superior to standard k-Means, hierarchical clustering, or simpler tree pruning.
3. UDONCARE shows consistent and often substantial performance gains over a wide range of baselines on four distinct prediction tasks and two large-scale public datasets.

### Weaknesses
1. The framework's success seems to hinge almost entirely on the ICD-9-CM disease hierarchy. The authors note in the appendix that adding procedure and medication codes yielded "marginal" and "inconsistent" gains. This is a limitation, as it suggests the model may not handle domain shifts caused by factors not captured in the disease hierarchy.
2. The study uses a temporal split (e.g., pre-2017 vs. post-2017) as its proxy for domain shift. While this is a valid and common setup, it is not the only (or even most difficult) type of shift. The paper's introduction mentions generalizing across different hospitals, but this cross-institutional generalization is not tested.
3. The domain discovery algorithm is multi-staged and complex, involving hierarchy-aware node initialization, a specific scoring function, a bottom-up pass, and a rectified beam search. This could create a higher barrier for reproducibility and adoption compared to simpler methods.

### Questions
1. Your domain discovery relies almost exclusively on the ICD-9 disease hierarchy, and you found that adding other codes didn't help. How would UDONCARE handle a domain shift that is uncorrelated with disease codes, such as a change in hospital billing practices, documentation standards, or the introduction of a new EHR system?
2. The pruning algorithm is designed to find an optimal set of ancestor nodes. What did the result look like in practice? Did the model converge on very high-level concepts, mid-level ones, or a mix? How interpretable were these auto-discovered domains?

### Soundness
2

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
In this paper, the authors propose UDONCARE, a framework that leverages medical ontologies to enhance domain generalization in electronic health record (EHR) prediction. Instead of relying on predefined domain labels, UDONCARE automatically discovers clinically meaningful latent domains by pruning hierarchies such as ICD-9 and learning domain-invariant features through mutual learning. Evaluated on MIMIC-III and MIMIC-IV across multiple prediction tasks, the model achieves strong generalization under domain shifts while maintaining computational efficiency. The work provides a novel, knowledge-guided approach for robust healthcare prediction across unseen patient populations.

### Strengths
1. The motivation of this study has been clearly illustrated. Benefitting from that, it becomes reasonable for readers to comprehend the necessity of using both Step 1 and 2 to tackle the problem accordingly.

2. Empirical coverage is broad: four clinically meaningful tasks, two large public datasets, and eight competitive baselines, demonstrating general usability as well as advanced performance,e.g., sizeable AUPRC gains on mortality and readmission have beewhile maintaining computation comparable to prior work.

3. Comprehensive analysis, ablation studies and side studies such as runtime test provide valuable insights regarding characteristics of UDONCARE.

### Weaknesses
1. There are still some hyperparameter studies that could have been done for a better understanding of UDONCARE. For instance, the updating frequency of M (mentioned in Line 269) can be further investigated. 

2. Some technical details can be further elaborated. For instance, how the loss terms in Eq. 9 functions together during the optimization? Per description, I am assuming they are added altogether without any weighting, but please explicitly detail it in Eq. 9.

### Questions
I am personally wondering whether this framework can be expanded to general domain, where features are also represented within an ontology, e.g., e-commerce.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to enhance the performance of predictive healthcare using EHR data by considering domain generalization. Specifically, this work iteratively divides patients into latent domains guided by the medical code (e.g., ICD) hierarchy. Within the hierarchy, each leaf node (i.e., code) can be represented by the embedding either from the model or from the entity name embedding. Then, all other nodes' embedding can be obtained using all its descendants. After that, each node can be assigned a score about coverage, purity, and depth borrowing the idea of information gain. Finally, if a parent node's score is higher than all its children, all children will be removed and if it is lower than all its children, the parent will be removed. Otherwise, the Beam-Search algorithm will be applied to either include or remove the parent using the Silhouette score. Now, given a pruned set of codes, patients can be divided into latent domains using the domain labels under those pruned codes and the component invariant to domain shifts can be obtained for training together with the original prediction loss. The experiments comparing with DG baselines using both MIMIC III and MIMIC IV on four different tasks show consistent improvement of performance.

### Strengths
1) Domain shift is a challenging concern within healthcare and the idea of using medical code hierarchy to guide the patient partition is interesting and sound.

2) The proposed pruning algorithm to include either the parent only or the children only codes in the hierarchy can not only improve the efficiency but also cover the necessary information hidden in the hierarchy.

3) Consistent performance improvement can be observed over multiple healthcare predictive tasks using both MIMIC III and MIMIC IV compared to a comprehensive list of baselines.

### Weaknesses
1) Some technical discussions or explanations are not clear, which makes it hard to follow and verify. For example, the step on the lowest common ancestor was to increase the similarity between certain node pairs (e.g., sharing the same parent). Right? It is not clear how the most similar pair was determined. 

2) In the step of hierarchy pruning, some details are missing. For example, we may have a case that in the lowest level, a parent (e.g., node A) has been removed due to the lower score compared to all it children in the leaves. However, in the next level, the parent's parent (e.g., B is a parent of A) is also having a lower score than all it children. Then, we need to keep A and remove B. How this conflict has been resolved?

3) In the step of domain searching, for each parent and children pair, are we also either including the parent only or the children only? The score is evaluated by checking if the node is pruned, how the separation is done. Only parent nodes are considered for this separation? It is not clear. What kind of features have been used to calculate the Silhouette score?

4) For the learning objective, it seems that two components in (9) are combined together. Is it directly summing them with the KL divergence sharing the same tradeoff parameter? This is also not clear.

### Questions
Questions can be found in the above weakness section.

### Soundness
3

### Presentation
2

### Contribution
3
