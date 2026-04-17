# Performance vs interpretability trade-off of hand-crafted and language model features: The case of protein superfamily classification

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
The newfound rise of protein language models (PLMs) that leverage data and compute has introduced an interesting conflict in computational biology: a trade-off between the high predictive performance of non-interpretable features and the scientific insight that can be gained from interpretable, hand-crafted ones. In this work, we highlight and study this conflict via the task of classifying protein domains into their CATH superfamilies. We train one-vs-all linear SVM classifiers for 45 CATH superfamilies, each characterised by significant class imbalance. We address the class imbalance by using a class-balanced loss function and the arithmetic mean (AM) of specificity and sensitivity for evaluation. Our analysis compares nine feature vector types, which are either non-interpretable embeddings from PLMs or interpretable hand-crafted features. The latter includes amino acid composition (AAC), di- and tri-peptide composition (DPC, TPC), and novel sequence-order (2OAAC, 3OAAC) and structure-based features (OCPC, CSIC). Our results demonstrate that PLM-based features achieve superior test AM scores of 90-99\% with low variability, outperforming hand-crafted features by 20-30\%. While PLM features yield high classification accuracy, their lack of interpretability obscures the underlying biological determinants. Conversely, the interpretability of hand-crafted features, despite their relatively low performance, can be leveraged to infer sequence and structural characteristics of CATH superfamilies. The proposed hand-crafted CSIC feature stikes a balance between predictive performance and interpretability, because it overfits to a lesser extent. This can be valuable for downstream applications like investigating protein-related diseases and guiding rational protein design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In their submission, the authors investigate different feature sets for a particular protein classification task, namely protein superfamily classification. The authors carefully introduce three different feature sets: sequence-based (n-gram), structure-based (based on contact pairs derived from protein structure), protein foundation model features (attention or embeddings from ProtBERT). The structure-based feature were proposed by the authors and they stress their broad availability by leveraging AlphaFold structure. They train SVM classifiers on different feature sets in a one vs. all fashion to discriminate most informative feature sets. Protein Language Model embeddings turn out to be most predictive, but not interpretable as the authors point out correctly, closely followed by contact-based features proposed by the authors.

### Strengths
* Pedagogical introduction to the field and to the different feature sets in a way that should be accessible also for people unfamiliar with a computational biology context.
* Newly proposed structural features that are apparently very discriminative (at least for the task at hand) that can be derived from widely available structural information (e.g. from AlphaFold)
* The authors correctly emphasize the tension between the discriminative power of features and their interpretability, which is important for applications in the Natural Sciences
* The authors use a sensible way of fixing the distance threshold for their contact-based features in a data-dependent manner.

### Weaknesses
* Just a single task is considered. More general insights could be drawn if further qualitatively different tasks would be considered e.g. enzyme classification or gene ontology prediction.
* The authors miss MSA-features as important and very powerful category of features.
* No combination of feature sets is considered. It would also be interesting to understand their individual contributions (e.g. using a Shapley formalism) to quantify which features contributes how much and potential overlap between different feature sets.
* There is a strong imbalance between setup and results. The actual results constitute barely one page in the manuscript. The largest part of the paper provides feature definitions (which could to some degree also be provided in the supplementary material). This would free up some space to add more details on experiments and implications.
* No statements about the statistical significance of their findings were made. These could for example be implemented via emprical bootstrapping the performance difference on the test set.
* Without doubt, the proposed structural features are more interpretable than foundation model features but are also not as interpretable as for example the sequence-based features. To turn this into a stronger submission, it would be nice to see some more specific insights from these more interpretable features to get an idea what kind of insights they could enable.

### Questions
* Did the authors explore any other protein language models, e.g. ESM2?
* Can the authors motivate their choice of classification algorithm?
* Was the structural information taken from experimental data or from AlphaFold? It would be interesting to quantify the difference between the two to assess whether the author's scalability argument (by using AlphaFold structures) is valid.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper touches a tradeoff between the interpretability and performance, and their findings were insightful especially to the domains where interpretability is important. They trained LinearSVC (in some cases SGDClassifier) with features either by hand-crafted engineering or PLMs, and compared the results in predicting 45 CATH superfamilies. The authors provided insight into how an interpretable features would achieve a comparable performances in the down streaming tasks.

### Strengths
1. The comparison processes were rigorous and the authors used both sequence and structure based hand-crafted features. 
2. The authors provided novel structure features, that could be useful to the biocomputation community, also the ProtBERT-Attn could also be useful even if it is not interpretable.
3. The authors covered the imbalance problems. The dataset is diverse, covering 45 CATH superfamilies.
4. The authors ran five random splits which made the experiment more rigiorous.
5. Easy to follow and understand the methodology.

### Weaknesses
1. The scope of this paper is my biggest concern. The paper aims to achieve a balance between performance and interpretability of PLMs. However, only the ProtBERT was evaluated. The scope of PML comparison is somehow limited, making me worried about whether their observations/results could be generalized when using other PLMs such as the ESM family or ProtT5. By doing so they can cover more training set of LMs and more dimensions.
2. For TPC and 30AAC, they were using SGDClassifier, how would that compared with the cases when LMs also give high-dimensional features? (Related to Weaknesses #1)
3. In most hand-crafted feature scenarios, the model is overfitting. 
4. Complexity to calculate CSIC.
5. Missing some ablation studies: for example, the authors' conclusions would be more solid if they can ablate the ProtBERT-Attn Feature (16 is the total), or ablating the CSIC itself (for instance, Intervals K)
6. (Minor) Table 5 doesn't have the dimensions for the features. Although table 1 has such information, table 5 would be more readable if the authors could indicate which features are high-dimensional or not. In addition, the order the features in Table 5 could be consistent with Table 1.

### Questions
1. Can you combine multiple features? How will that affect the performances of the LinearSVC/ SGDClassifier.
2. How might your experiments be generalized to non-bio-computation domains?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper discusses the trade-off between manual features and protein language model (PLM) features in protein superfamily classification tasks in terms of prediction performance and interpretability, and proposes a new feature engineering method to balance the two.

### Strengths
1. By employing various feature engineering techniques, this study analyzed the encoding capacity of protein sequences for genetic information.
2.proposed one-vs-all classifiers to predict the CATH homologous superfamily of a protein domain.

### Weaknesses
1.The article's structure and mathematical notation are somewhat disorganized, making it difficult for readers to grasp the core methodology. 
2.The study lacks sufficient experiments, fails to analyze downstream applications, and does not conduct ablation experiments on the method itself.

### Questions
How can the proposed study be integrated with existing LLM-based AI technologies for protein understanding and generation?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the performance versus interpretability trade-off in protein superfamily classification by comparing nine hand-crafted features versus protein language model features (ProtBERT). The authors conclude that while PLMs offer high predictive accuracy, carefully engineered interpretable features like CSIC can balance performance and interpretability.

### Strengths
1. The paper is well-structured and easy to follow.
2. I appreciate the effort in designing the evaluations for class-imbalance protein predictions, which is well-motivated in real life.

### Weaknesses
Major Issues

The experimental setup is far too narrow to support the paper's broad claims. The authors argue there's a fundamental trade-off between performance and interpretability in protein modeling, but they only test this on a single task with a single PLM:

* CATH superfamily classification is a simplified sequence similarity problem. The conclusion cannot be generalized to other tasks like  binding affinity and structure predictions. The paper can't properly claim a "fundamental trade-off" exists across computational biology when it  only looked at one relatively simple classification problem.

* only ProtBERT:  ESM is SOTA on most benchmarks and is what people actually use in practice. Testing against an older, weaker model doesn't tell us much about the real performance gap.

This experiment just shows ProtBERT beats some hand-crafted features on superfamily classification. The conclusions about performance vs. interpretability being a universal trade-off cannot be drawn given this limited evidence.

In addition, the paper claim hand-crafted features are interpretable, but never demonstrate this. Where's the biological insight? What features distinguish different superfamilies? Without showing this, what is the interpretability?

Overall, the execution of the paper lacks the experimental rigor and technical depth expected for a venue like iclr.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
1
