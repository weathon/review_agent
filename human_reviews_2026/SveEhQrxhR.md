# MD-RE: A Multi-Discrimination Framework for Document-Level Relation Extraction with Adaptive Threshold Shifted Loss

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Document-level relation extraction (DocRE) aims to identify relations for an entity pair within a document. Existing methods can be broadly classified into two categories: direct encoding of the entire document or enhancement using extracted evidence sentences. However, the former often introduces noise unrelated to relations, while the latter is heavily dependent on the quality of evidence extraction. Moreover, these DocRE models typically use an adaptive threshold to predict all potential relations for an entity pair. As a result, class imbalance in DocRE often leads the model to learn a high throshold for an entity pair, which in turn causes the model to frequently predict that the entity pair has no relation. To address these issues, we propose a **M**ulti-**D**iscrimination framework (**MD-RE**) that does not rely on evidence sentences. MD-RE employs three discriminators with dynamically adjusted thresholds to independently predict relations, and aggregates their outputs via a weighted fusion strategy. Furthermore, we propose an **A**daptive **T**hreshold **S**hifted **L**oss (**ATSL**), which encourages lower threshold to alleviate the high false negative rate resulting from class imbalance. Experiments on three datasets demonstrate that our MD-RE framework with ATSL achieves new state-of-the-art results. Moreover, ATSL significantly improves the performance of various existing DocRE models. In addition, combining other losses with MD-RE also yields competitive results. Our code is available at https://anonymous.4open.science/r/MD-RE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper points out that for the document-level relation extraction task, existing methods face challenges in identifying the relationships between entities. This paper proposes a Multi-Discrimination framework (MD-RE) for DocRE and incorporates a perceptual loss function to improve the model's performance on the DocRE task.

### Strengths
1. This paper is well-written, with clear logic, and it fully conveys the motivation and methodology of the research.

2. The paper conducts extensive experiments, including comparative experiments on models of different types and scales.

### Weaknesses
1. First, the paper points out that MD-RE will improve the model's performance on the DocRE task. However, MD-RE adopts multiple discriminators, and the additional computational and time costs incurred have not been evaluated. This makes comparisons with previous methods unfair.

2. Second, based on Table 5, the paper argues that the ATSL loss proposed by the authors is superior to other loss functions. Nevertheless, the gain brought by ATSL loss is minimal—this gain may likely come from the added cumbersome computational process, and the effect is not significant.

3. Finally, the experiments presented in Tables 7 and 8 use non-state-of-the-art baselines, which is unreasonable.

### Questions
See Weaknesses.

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
MD-RE effectively addresses the issue of evidence-sentence dependency and reduces document noise through multi-discriminator fusion, showing solid performance across several datasets. ATSL’s threshold bias design mitigates class imbalance, particularly reducing false negatives. However, the paper lacks a direct comparison with multi-stage baseline models, and there is no analysis of how to mitigate ATSL’s increase in false positives. Additionally, MD-RE’s performance in long-document or low-resource scenarios remains untested.

### Strengths
1.The introduction of MD-RE addresses the limitation of evidence-sentence dependency by using multiple discriminators, which enhances the model's ability to handle document noise and supports multi-perspective reasoning.

2.ATSL's threshold bias design allows for flexible decision boundary adjustments, improving performance by reducing false negatives.

3.The extensive experiments demonstrate that MD-RE outperforms state-of-the-art methods across multiple datasets and shows generalization across various baselines.

### Weaknesses
1.No direct comparison with multi-stage or multi-discriminator DocRE models (e.g., hierarchical filtering methods) to clarify MD-RE’s novelty in framework design.

2.MD-RE’s performance on long documents (e.g., >10 sentences) or low-resource settings (e.g., few-shot) is untested, leaving its generalizability to challenging scenarios unclear.

3.The fusion module’s weighted sum strategy lacks ablation against other fusion methods (e.g., attention-based fusion), and the weighting factor α’s tuning rationale across datasets is not explained.

4.The class imbalance mitigation comes at an unacceptable cost of false positives. While ATSL reduces false negatives (FN) from 5833 to 4181, it triples false positives (FP) from 1942 to 4029 (Table 8). The paper does not explain how this FP surge is justified for real-world applications (e.g., knowledge graph construction, where FPs corrupt downstream tasks). This critical trade-off is ignored, undermining ATSL’s practical value.

5.The ATSL loss’s theoretical analysis is incomplete and inconsistent. The paper claims ATSL is "convex" but does not address that the convexity holds only for fixed P_T and N_T—in practice, these sets change dynamically with logits, making the overall loss non-convex. 

6.No analysis of MD-RE’s failure cases beyond the single case study; it is unknown which relation types (e.g., long-tail) or document structures lead to underperformance.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a document-level relation extraction method that integrates a multi-discriminator framework (MD-RE) with an Adaptive Threshold Shifted Loss (ATSL). The main innovation lies in its three-stage discriminator design (recall, coarse, and fine), which jointly addresses noise reduction and class imbalance without relying on external evidence sentences. The authors conducted comprehensive experiments on three benchmark datasets, demonstrating that the proposed approach achieves state-of-the-art performance on multiple metrics, while the ATSL loss exhibits strong generalization ability across different models.

### Strengths
1. The authors accurately identify a key challenge in DocRE — severe class imbalance that causes overly high adaptive thresholds and excessive false negatives — and propose ATSL as a simple yet generalizable solution.
2. By introducing bias terms (λ, β), ATSL mathematically redefines the margin constraints between positive/negative samples and decision thresholds. The convexity proof and Bayes-consistency analysis in the appendix provide a solid theoretical foundation for the method.
3. Extensive experiments across three mainstream datasets (including loss replacement, disabling LNS, and removing discriminators) consistently verify the effectiveness of both MD-RE and ATSL, and confirm the adaptability of ATSL to various baseline models.

### Weaknesses
1. Although Figure 2 outlines the overall framework, the differences among the three discriminators’ decision behaviors are not clearly visualized. For example, what specific error types are filtered by the coarse versus the fine discriminator? The lack of such case studies makes the internal working mechanism less intuitive.
2. The paper notes that the optimal α parameter varies across datasets but does not provide systematic tuning guidelines. This sensitivity could hinder real-world deployment, particularly in low-resource scenarios without a dedicated validation set.
3. LNS selects top-k negative samples based on loss values, yet no comparison with alternative strategies (e.g., random sampling, margin-based mining) is provided, and ablation studies suggest its contribution is marginal.
4. The final fusion module uses fixed weighted averaging (Eq. 7–8). Although ablation studies show its effectiveness, the absence of comparisons with other fusion strategies makes it difficult to judge the optimality of this design.

### Questions
1. What specific types of false positives increase after applying ATSL? Could rule-based post-processing help reduce them?
2. For documents with sparse relational structures, is the full three-stage filtering necessary? It would be valuable to evaluate a subset of DWIE with low relation density to test the method’s generalizability.
3. The rule “directly reject if the recall discriminator predicts NA” — does it occasionally remove true positives? If so, what is the approximate proportion?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of document-level relation extraction (DocRE), where models must determine relations between entity pairs across long texts. Existing DocRE systems often suffer from two issues: (1) High false-negative rates due to extreme class imbalance, causing the model to learn an overly strict relation-prediction threshold. (2) Noise from full-document encoding, which makes it difficult to distinguish subtle positive relations from dominant negative patterns.

The authors propose MD-RE, a Multi-Discrimination framework consisting of three discriminators with different decision thresholds: a recall-oriented discriminator, a coarse one, and a fine one. Instead of relying on sentence-extraction or evidence mining, these discriminators independently evaluate candidate relations from different perspectives and are combined via a weighted fusion mechanism. They introduce Adaptive Threshold Shifted Loss (ATSL), which shifts the learned threshold logits to counteract the high FN rates, and significantly include some good true positive candidates inside the decision boundary.

### Strengths
The multi-discriminatory design introduced by the author(s) is an interesting one. It targets a specific limitation of other methods involving high false negatives. Their recall-based discriminator helps in correcting a significant number of false negatives. 

In particular, the redesigning of Adaptive-Threshold Loss (ATL) seems to be a stronger contribution of the author(s) to me. Because of the class imbalance with too many false negatives in the predictions, they introduce a shifting parameter in the loss calculation which significantly improves the decision mechanism of the discriminators. They call it the ATSL loss. They also provide satisfactory result analysis to represent the strength of this loss in their paper. The improved performance proves that the logits produced from the training follow the precision-recall trend of the individual discriminators.

### Weaknesses
* Though unlike traditional ensemble methods, the discriminators share the encoder, the fusion of the discriminators seems to be a specialized utilization of an ensemble method to me. I expect the author(s) to emphasize on this during the rebuttal.

* The removal of coarse, and fine discriminators show a slight decrease in F1 (still higher than other SOTAs) (Table 6). I would like to see how the method performs without both of them, only with the recall Discriminator.

* The method naturally includes a lot of False Positives in the game (Table 8). The authors did not provide any result analysis using the AUC scores. As a result, I cannot assess how strong the model is to discriminate between a positive and negative relation. I would like to see their result analysis on AUC metrics too.

### Questions
To assess the performance gain of the coarse and fine discriminator more deeply, I would like to see how the recall-only discriminator performs.

Please refer to the suggestions in the ‘Weaknesses’ section.

### Soundness
3

### Presentation
4

### Contribution
2
