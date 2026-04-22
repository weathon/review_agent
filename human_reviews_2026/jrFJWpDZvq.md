# DeepSADR: Deep Transfer Learning with Subsequence Interaction and Adaptive Readout for Cancer Drug Response Prediction

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Cancer treatment efficacy exhibits high inter-patient heterogeneity due to genomic variations. While large-scale in vitro drug response data from cancer cell lines exist, predicting patient drug responses remains challenging due to genomic distribution shifts and the scarcity of clinical response data. Existing transfer learning methods primarily align global genomic features between cell lines and patients. However, they often ignore two critical aspects. First, drug response depends on specific drug substructures and genomic pathways. Second, drug response mechanisms differ in vitro and in vivo settings due to factors such as the immune system and tumor microenvironment. To address these limitations, we propose DeepSADR, a novel deep transfer learning framework for enhanced drug response prediction based on subsequence interaction and adaptive readout. In particular, DeepSADR models drug responses as interpretable bipartite interaction graphs between drug substructures and enriched genomic pathways. Subsequently, a supervised graph autoencoder was designed to capture latent interactions between drugs and gene subsequences within these interaction graphs. In addition, DeepSADR treats the drug response process as a transferable domain. A Set Transformer-based adaptive readout (AR) function learns domain-invariant response representations, enabling effective knowledge transfer from abundant cell line data to scarce patient data. Extensive experiments on clinical patient cohorts demonstrate that DeepSADR significantly outperforms state-of-the-art methods, and ablation experiments have validated the effectiveness of each module.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DeepSADR, a deep transfer learning framework designed to improve drug response prediction for clinical patients. The core idea is to decompose drugs and genomes into subsequences (substructures and functional pathways, respectively), model their interactions in a graph, and use an adaptive readout mechanism for knowledge transfer. The experimental results demonstrate that DeepSADR achieves performance improvements. However, the approach is essentially a combination of existing components, and the training framework itself is not new. In addition, the method relies heavily on predefined prior knowledge (such as fixed functional pathways and drug-specific thresholds), which limits its generalizability.

### Strengths
1. The methodological pipeline is clearly described, and the overall framework is easy to follow.

2. The proposed approach demonstrates substantial performance improvements over existing baselines

### Weaknesses
1. The paper states that "we assigned the genes to 13 functional pathways", but it is unclear how this assignment is performed and why these 13 fixed pathways are predefined. For patients with different diseases, should the pathways be redefined? Since each patient’s pathway profile can differ, these fixed and limited pathways may hinder the model’s generalizability. Moreover, such a strong prior assumption may fail to capture disease-specific biological processes or drug-specific mechanisms, especially across different cancer types.

2. The paper does not clearly describe the data split strategy. Were the baseline models retrained by the authors? Some reported results differ from those in prior literature. For example, WISER reports an AUC of 0.868 for CODE-AE on Fluorouracil, which is inconsistent with the results shown in Table 1.

3. The model appears highly sensitive to threshold parameters $t$, with different thresholds set for each drug. This indicates that the model requires fine-grained hyperparameter tuning for each individual drug. In practice, this is a major limitation as it requires sufficient labeled data for each new drug to determine the optimal threshold, which contradicts the established goal of addressing the scarcity of clinical data.

4. In the ablation study, removing either the AR or SN module leads to a significant performance drop. This suggests that using only one of these modules appears to be ineffective, whereas combining them yields considerable performance gains. This is unusual, and further experiments are needed to elucidate the interactions between these modules.

5. The interpretability validation is insufficient. The interaction heatmap for Temozolomide is shown without deeper analysis of the underlying biological mechanisms, leaving the reader uncertain about the biological relevance of the findings.

6. In Equations (15), (16), and (17), several symbols (e.g., s, E, Y) are not defined, making these formulas difficult to interpret.

### Questions
See weaknesses.

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
The authors present a novel training paradigm that leverages SMILES representations of drugs and enhanced genomic features to effectively recommend drugs for patients. While the approach is interesting and potentially impactful, several key methodological details are insufficiently explained. This lack of clarity makes it difficult to fully understand how the proposed method addresses the limitations and challenges of prior baselines and achieves its reported performance gains.

### Strengths
- The present work aims to bridge the gap between laboratory-based drug studies and computational learning approaches for cancer drug recommendation. 
- The integration of SMILES representations and genomic enrichment for predictive modeling is well justified and conceptually sound. 
- The paper is well written, especially the introduction section.
- The results are encouraging and demonstrate clear improvements over the baseline models.

### Weaknesses
- While the proposed approach is interesting, I have concerns regarding how the method learns a domain-invariant representation between the cell line and patient genomic profiles, given that there is no explicit training objective designed for domain alignment.

- The authors have not clearly specified which datasets are used during each phase of training (e.g., pre-training vs. fine-tuning). This lack of clarity makes it difficult for readers to fully understand how the reported performance improvements are achieved. 

-  The biological interpretability experiment is insufficiently explained. Further clarification is needed to understand the biological insights or validations derived from the proposed model.

-I have several questions regarding the implementation of the method. Please refer to the “Questions” section for detailed queries. If these issues are adequately addressed, I would be open to update my score.

### Questions
- (L 233-235) The necessity of modeling long-term feature dependencies for molecular graphs is unclear. Compared to graphs in other domains, molecular graphs are typically smaller in size. Why, then, is long-term dependency modeling essential in this context? Has an ablation study been conducted to justify this design choice?

- Suggestion : Consider revising Equation (7). Presenting both the cell line and patient genomic profiles together in the same formulation may confuse readers.

- (L 261-269)The interpretability of the subsequence graph remains unclear. Do the associations learned by the model have any biological or scientific validation? Could the approach potentially introduce spurious associations? Given the limited size of the patient genomic dataset, such issues might substantially affect the model’s generalization ability.

- Why not utilize a drug–gene interaction database, such as DGIdb(https://dgidb.org/browse/sources), to construct an interaction graph? 

- Why is mean squared error (MSE) used for a binary prediction task? Wouldn’t the standard cross-entropy loss be more appropriate for categorical outputs?
- Please clarify what is used as the prior probability term  p(z) in Equation (18).

- What is the rationale behind concatenating pre-trained and fine-tuned features in Equation (19)? Is this decision empirically motivated? Also, based on Figure 1, does the concatenation involve cell line and patient genomic profiles? Since these come from different domains with no one-to-one correspondence, I have doubts about its implementation. Please clarify how this concatenation is implemented. let me know if I have misunderstood any part. 

- Is fine-tuning performed on both the patient and cell line datasets? Please clearly specify which datasets belong to the pre-training and fine-tuning stages.
- How is the evaluation conducted? Is it based on k-fold cross-validation, or on the mean performance across multiple independent runs (e.g., five runs)?
- Which feature representations are used for the t-SNE plot in Figure 4—pre-trained, post-trained, or concatenated features?
- Please elaborate on Figure 2. How should the results be interpreted, and how statistically or biologically significant are these findings?

### Soundness
3

### Presentation
3

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
This paper presents DeepSADR, a transfer learning framework for predicting cancer drug responses in patients using cell line data. It models drug-gene relationships as bipartite subsequence interaction graphs and uses a supervised graph autoencoder to learn interpretable latent features. An adaptive readout module based on the Set Transformer enables domain-invariant feature learning between cell lines and patients. Experiments show that DeepSADR outperforms existing methods and provides improved interpretability.

### Strengths
The paper is clearly written and well-structured, with a strong motivation based on the challenge of predicting patient drug responses from scarce clinical data and heterogeneous genomic profiles.
The proposed subsequence interaction graph and adaptive readout mechanism are conceptually sound, capturing fine-grained drug–gene associations while enabling domain-invariant feature learning, making the technical approach meaningful and biologically interpretable.
The experimental setup is comprehensive, including multiple clinical cohorts, extensive ablation studies, and comparisons with state-of-the-art baselines, demonstrating robust performance, good generalization, and practical applicability for personalized cancer treatment prediction.

### Weaknesses
Major: A key weakness is the limited dataset size for several drugs, which may undermine the reliability and generalizability of the model’s predictions. In particular, drugs such as Sorafenib (26 samples) and Cisplatin (40 samples) have very few data points, increasing the risk of overfitting. The paper also does not discuss strategies to mitigate issues arising from small sample sizes or class imbalance, which could affect the robustness of the experimental results.
Minor: In Table 1, for Gemcitabine, DeepSADR achieves an AUPR of 0.702, which is lower than WISER (0.752), yet it is marked in bold. Additionally, Appendix A.1 (Pseudocode and List of Notations) is disorganized and difficult to read. Some recent relevant models, such as PREDICT-AI (https://doi.org/10.1145/3637528.3671652), are missing from the discussion.

### Questions
Please see the weaknesses section for further discussion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors have tackled the problem of drug response prediction in patients, in the face of limited labelled data. They build on previous approaches in this space, and identify a pertinent gap in prior literature – substructures of drugs and genomic pathways can affect the response to treatment. They propose the use of graph autoencoders and a set transformer to obtain adaptive readouts. The method proposed – DeepSADR – is shown to outperform existing methods in this space.

### Strengths
1.	The problem is well motivated and the research gap identified is relevant.
2.	Extensive ablation studies have been conducted to show the importance of each piece in the proposed architecture.
3.	Visual analysis of the most important substructures was also done, which is good for interpretability.

### Weaknesses
My key concern is that the reason for treating "drug response" as a separate domain is unclear. The results do appear to improve but it would be good to add some intuition on why the improvement is seen.

A few baselines seem to be missing, it would be good to include them -  more details in the Questions section below.

### Questions
1.	What is the computation overhead of this method, given that for every (patient, drug) pair, a separate interaction graph must be constructed on the fly? Please provide some intuition about this.
2.	It is unclear how the model explicitly handles the biological differences in drug responses across patients and cell lines.  Is this offloaded to the trainable adaptive readout and predictor parts of the network in fine tuning? 
3.	Please comment on the data itself – are there batch effects? If so, how is it handled? I’m guessing this is already handled in the data from the earlier method, but please include this in the description.
4.	The graph construction is also dependent on the choice of threshold t. Are there any sensitivity experiments done to understand the effect of this value?
5.	The authors propose the use of drug response process as a different domain – the reason and intuition is unclear. 
6.	The model is finally trained in an inductive manner, like some of the prior methods. Please mention this.
7.	In section 4.2, the results are found to be significantly better, please add p values to indicate significance.
8.	Please combine the table 1 in results with the results from the appendix, so that the standard deviation can be seen.
9.	Please include the process of hyperparameter tuning.
10.	Please highlight the limitations of this model.
11.	Please also include baselines like PANCDR[1] and TransDRP[2].
12.	The description of the baselines seems erroneous, please update them 
13.	In table 7, the model without finetuning appears to do better – is there a reason why?
14.	In Table 10, Sorafenib is repeated – was this not present in the earlier set of 5 drugs used for training?

[1] Kim, J., Park, S. H., & Lee, H. (2024). PANCDR: precise medicine prediction using an adversarial network for cancer drug response. Briefings in Bioinformatics, 25(2), bbae088.

[2] Liu, X., & Li, M. (2025, April). Knowledge-Guided Domain Adaptation Model for Transferring Drug Response Prediction from Cell Lines to Patients. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 1, pp. 523-531).

### Soundness
3

### Presentation
3

### Contribution
3
