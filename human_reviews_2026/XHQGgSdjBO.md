# Hierarchical Cautious Optimization for Semi-Supervised Medical Image Segmentation with Limited Labeled Data

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Semi-supervised learning (SSL) effectively addresses limited labeled data challenges in volumetric medical image segmentation by leveraging both ground-truth labels and pseudo-labels from unlabeled data. However, conventional optimizers treat gradients from labeled and unlabeled sources equally, often leading to either over-trust or over-rejection of pseudo-label signals. We introduce Hierarchical Cautious Optimization (HCO), which establishes a trust hierarchy between gradient sources. HCO computes momentum estimates using only gradients from labeled data and incorporates unlabeled gradients only when they align with this trusted direction. Our approach integrates into existing momentum-based optimizers with minimal implementation effort and computational cost. Evaluations across three datasets demonstrate consistent performance improvements, particularly on a challenging fetal MRI dataset where Dice scores for fetal lungs and liver increased from 0.68 to 0.84 and 0.71 to 0.82, respectively. The consistent gains across optimizers and datasets, combined with minimal implementation overhead, position HCO as a practical enhancement for existing SSL medical segmentation pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of noisy optimization in semi-supervised medical image segmentation, where limited annotations can lead to unreliable pseudo-labels. To mitigate this, the authors propose Hierarchical Cautious Optimization (HCO), an optimizer that establishes a trust hierarchy between labeled and unlabeled gradients. It is a kind of extension of previous Cautious Optimization work. The method is lightweight, generalizable to various optimizers (SGD, Adam), and introduces negligible computational overhead. Experiments across three datasets (Left Atrium MRI, Pancreas CT, and Fetal MRI) and two backbone optimizers demonstrate consistent improvements in Dice and surface metrics, confirming HCO’s effectiveness as a general-purpose optimization strategy for semi-supervised segmentation.

### Strengths
Addressing the noisy optimization problem is new and interesting in semi-supervised medical image segmentation.

The verified scenario is well-suited to the method designs.

The performance is consistently improved across different settings.

The paper is clear and well-written.

### Weaknesses
While the key idea of hierarchical cautious optimization is interesting, the technical novelty beyond prior work (Cautious Optimizers) could be further clarified. The paper may benefit from emphasizing medical-specific considerations, such as how anatomical or imaging characteristics influence gradient noise.

The experimental validation, although multi-dataset, is limited to two backbone optimizers. Given the wide variety of open-source SSL frameworks, additional and extensive comparisons could strengthen the empirical claims.

It would be valuable to extend the evaluation to standard noisy segmentation tasks to demonstrate broader applicability beyond semi-supervised learning.

As foundation models and larger annotated datasets are becoming increasingly available, it would be insightful to analyze HCO’s performance with varying label ratios, including higher-label or fully-supervised regimes augmented with unlabeled data, to assess its scalability and relevance in future medical AI settings.

### Questions
Please check the above weaknesses. Overall, this work can be improved by adding more results to show its generalized designs.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a novel optimization framework, Hierarchical Cautious Optimization (HCO), designed for semi-supervised medical image segmentation, particularly in scenarios with limited labeled data. HCO builds upon the Cautious Optimizer (CO) by introducing a hierarchical mechanism that modulates the learning rate based on the consistency between the gradients of labeled and unlabeled data. The authors propose two variants, Hierarchical Cautious AdamW (HCO-AdamW) and Hierarchical Cautious SGD (HCO-SGD). The paper provides a theoretical convergence guarantee for HCO and demonstrates its effectiveness through extensive experiments on three different medical imaging datasets (Left-Atrium MRI, Pancreas-CT, and Fetal-MRI) and two semi-supervised learning frameworks (BCP and CMT). The results show that HCO consistently outperforms standard optimizers and the original CO, leading to improved segmentation accuracy.

### Strengths
1.  The proposed Hierarchical Cautious Optimization (HCO) framework is a novel and intuitive extension of the Cautious Optimizer (CO).
2.  The paper presents compelling experimental results across three challenging medical imaging datasets and two different semi-supervised learning frameworks, consistently demonstrating the superiority of HCO over standard optimizers and CO.
3.  The inclusion of a convergence proof for HCO strengthens the paper's theoretical foundation.
4.  The paper is well-organized, clearly written, and easy to understand.

### Weaknesses
1.  In Table 3, the results of AdamW_CO are substantially lower than those of the standard semi-supervised baseline, AdamW. Could the authors explain the reason for this? Moreover, is it inadequate to illustrate the effectiveness of the method merely by taking the CO method as the core comparative approach?
2.  While the paper provides implementation details, a more in-depth analysis of the sensitivity of HCO to its hyperparameters would be beneficial.
3.  Nearly all the citation formats of the references in the paper are incorrect. Authors should be acutely aware of the difference between `\citep` and `\citet`. In the ICLR LaTeX template, the `\cite` command specifically represents the inline citation form, which differs from that used in other conferences.
4.  There are some minor formatting errors in the article. For example, Appendix D is empty.

### Questions
In the convergence analysis, you make an assumption about the positive correlation between the overall update direction and the negative true gradient. Can you provide more intuition or empirical evidence to support this assumption?

### Soundness
4

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
5

### Summary
This paper proposes an optimizer framework for semi-supervised medical image segmentation, named Hierarchical Cautious Optimization (HCO). The core idea is to introduce a “trust hierarchy” into momentum-based optimizers: the momentum is computed only from labeled samples, and unlabeled gradients are incorporated into the update only when they are aligned with that trusted direction. The authors claim that this method improves segmentation accuracy across multiple datasets without modifying the model architecture.

### Strengths
1. The method is simple and easy to implement. It can be seamlessly integrated into common momentum optimizers with minimal code modification.
2. Experiments cover multiple datasets, including atrial, pancreas, and fetal MRI, demonstrating cross-modality performance.
3. Writing quality is good. The paper is well-structured, results appear reproducible, and the appendix is complete.

### Weaknesses
1. Lack of innovation
The proposed “trust hierarchy” is essentially a gating mechanism based on gradient sign consistency, highly similar to existing approaches such as Cautious Optimizer (CO) and SignSGD. The only difference is the use of a “labeled momentum” as a reference direction, which is a minor modification and hardly a methodological breakthrough.

2. Weak theoretical contribution
The so-called “convergence proof” heavily relies on assumptions (see Remark 1 and Theorem 1). The key assumption — that the expected direction is correlated with the negative gradient — is not verifiable but merely postulated. There is no rigorous analysis of the unlabeled gradient distribution, making the theoretical part almost unverifiable in demonstrating the claimed mechanism.

3. Limited experimental credibility
The experiments only report mean ± standard deviation without details of statistical significance analysis. Although the improvements are statistically significant, most of them are less than 2% in Dice, which is close to experimental noise. Moreover, the core conclusions depend on a private fetal MRI dataset, which limits reproducibility.

4. Lack of comparisons and ablations
The paper does not provide a systematic comparison with other “direction-filtering” optimizers such as CO, SignSGD, or GradDrop. The ablation study only compares “with vs. without hierarchy,” without analyzing key factors such as gating strength or α parameters.

5. Overstated conceptual framing
The paper repeatedly uses terms like “hierarchy,” “trust,” and “cautious,” but the actual mechanism is a simple sign-based gradient filter. The novelty lies more in terminology than in substance.

### Questions
1. Please include more qualitative visualizations, including both successful and failed cases, ideally showing prediction probability maps or boundary uncertainty distributions.
2. Provide a formula-level comparison table in the appendix with existing “cautious” optimizers.
3. Conduct a statistical analysis of the correlation between labeled and unlabeled gradient directions to support the theoretical assumptions.
4. Validate the method on public datasets to ensure reproducibility.
5. If the main focus is optimization theory, please show cross-domain experiments (non-medical data) to demonstrate generality.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Hierarchical Cautious Optimization (HCO),  a novel optimization framework that establishes a trust hierarchy between labeled and unlabeled gradients. HCO forms momentum using only labeled gradients and incorporates unlabeled gradients only when their directions align with the trusted labeled momentum. This hierarchical filtering aims to mitigate noisy pseudo-label effects such as confirmation bias and co-training collapse. The method requires minimal implementation changes and adds negligible computation.

### Strengths
- HCO is motivated clearly by addressing confirmation bias and co-training collapse in semi-supervised learning.
- HCO proposes a simple yet effective optimization strategy that improves model performance without modifying existing SSL frameworks, and the theoretical analysis in the appendix provides convergence guarantees under certain assumptions.
- The paper is well-written and clearly organized, with consistent notation and explanations.
- Models enhanced with HCO achieve notable performance improvements on several commonly used medical segmentation datasets.

### Weaknesses
- Although the method claims to serve as a plug-and-play module that can be easily integrated into existing SSL pipelines, the experiments only include evaluations on BCP. To my knowledge, several follow-up works in this domain, such as PMT [1] and AD-MT [2], have already explored more advanced SSL paradigms. Demonstrating the effectiveness of HCO on these frameworks would make the paper much more convincing. My overall evaluation of the paper will largely depend on how this issue is addressed.
- Figure 1 is relatively simple and less informative; compared with the textual explanations, it fails to intuitively illustrate how the proposed method works.
- The paper lacks sufficient citations, including the aforementioned works and other more recent advances in semi-supervised medical image segmentation.

References

[1] Gao N, Zhou S, Wang L, et al. PMT: Progressive Mean Teacher via Exploring Temporal Consistency for Semi-supervised Medical Image Segmentation. In: European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 144–160.

[2] Zhao Z, Wang Z, Wang L, et al. Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation. In: European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 227–243.

### Questions
- After integrating HCO into the optimizer, does the optimizer become sensitive to hyperparameters? If so, is this sensitivity consistent with that of the original optimizer, or does it exhibit any new trends? If not, please provide theoretical or empirical justification.
- HCO appears to be a general framework applicable to semi-supervised learning beyond medical image segmentation. Could the authors demonstrate or discuss its potential generalization to other domains? If such evidence is absent, please clarify the reasons.
- The proposed HCO method is remarkably simple and straightforward. Could the authors confirm whether there truly has been no prior related work adopting a similar idea or formulation?

### Soundness
3

### Presentation
3

### Contribution
3
