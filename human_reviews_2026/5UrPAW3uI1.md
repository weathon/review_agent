# FedOpenMatch: Towards Semi-Supervised Federated Learning in Open-Set Environments

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Semi-supervised federated learning (SSFL) has emerged as an effective approach to leverage unlabeled data distributed across multiple data owners for improving model generalization. Existing SSFL methods typically assume that labeled and unlabeled data share the same label space. However, in realistic federated scenarios, unlabeled data often contain categories absent from the labeled set, i.e., outliers, which can severely degrade the performance of SSFL algorithms.
In this paper, we address this under-explored issue, formally propose the open-set semi-supervised federated learning (OSSFL) problem,  and develop the first OSSFL framework, FedOpenMatch. Our method adopts a one-vs-all (OVA) classifier as the outlier detector, equipped with logit adjustment to mitigate inlier-outlier imbalance and a gradient stop mechanism to reduce feature interference between the OVA and inlier classifiers. In addition, we introduce the logit consistency regularization loss, yielding more robust performance.
Extensive experiments on standard benchmarks across diverse data settings demonstrate the effectiveness of FedOpenMatch, which significantly outperforms the baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a method called FedOpenMatch for open-set federated semi-supervised learning. By "open-set", the authors refer to the setting where outlier classes exist in unlabeled datasets, and the model should be able to classify inlier classes right, while detecting outlier classes. To address the problem, FedOpenMatch leverages a two-head structure, with a feature extractor, an inlier classifier and an OVA outlier classifier. The authors introduce several techniques to enhance the performance of FedOpenMatch, such as stop-gradients for the OVA branch, logit consistency instead of probability consistency, and logit adjustment to handle the imbalance issue of OVA classifiers. Experiments are done on three datasets and against multiple baselines, where FedOpenMatch outperforms a variety of baselines, including open-set and closed-set ones.

### Strengths
1. This paper tackles the problem of open-set federated semi-supervised learning, which is of practical value. Being able to detect outlier classes is an important property especially in FL, where the collected data may contain a large amount of noise. 
2. This paper is overall well-written and easy to follow. The organization of this paper is reasonable, and the technical solutions and intuitions are clearly stated together, making it easy to understand. 
3. Experiments of FedOpenMatch is extensive. The authors compare against lots of baselines, including SSFL algorithms and adapted Open-set SSL methods (to the FL setting) with multiple baselines, number of seen classes, etc. Moreover, ablation studies are sufficient and demonstrate the impact of each individual design technique. Some ablation studies even analyze why the proposed techniques help (e.g. analyzing gradient similarity to understand the impact of gradient stop). This makes the paper well-justified.

### Weaknesses
1. One major weakness of this paper is that it is built upon multiple existing ideas. For example, the logit adjustment method is directly taken from Menon et al. 2021, and the weak-strong consistency is modified from existing methods. Overall, this slightly weakens the amount of novel insights of this paper, despite still being a solid paper with new problems tackled. 
2. I am interested in how FedOpenMatch is sensitive to the number of local iterations/epochs. As local clients only have unlabeled data, training on solely unlabeled data for too long may lead to diverging model updates. Therefore, from my understanding, the number of local iterations/epochs is an important parameter to determine in federated SSL. The authors did not provide such analyses though. 
3. Gradient similarity fail to completely explain the accuracy curve. While in Figure 3, the relations between gradient similarity and open-set accuracy is significant, such relations are not so apparent in Figures 4 and 5. For example, in Figure 4, for the early ~250 steps, gradient similarity is below 0.2, yet the accuracies are close. This makes me wonder whether there are other factors that may impact the accuracy, e.g. gradient magnitude.  
4. Minor points. 
    - It seems that in Eqn. 6, the sign of $\omega\log \pi$ is reversed. I checked the original paper and found that the sign before the term should be - instead of +.

### Questions
One minor question about FedOpenMatch is when a sample will be categorized as outlier, e.g. when all OVA classifiers report it as an outlier? I am not highly familiar with open-set semi-supervised learning so some additional backgrounds may help.

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
The paper introduces the open-set semi-supervised federated learning (OSSFL) problem, where clients possess unlabeled data potentially containing unseen classes, and the server maintains a small labeled set. To address this, the authors propose FedOpenMatch, a framework that jointly trains an inlier classifier and a one-vs-all (OVA) outlier detector. The method incorporates gradient stop, logit adjustment, and logit consistency regularization to improve stability and open-set discrimination under federated conditions.

### Strengths
The paper presents a clear problem setting that distinguishes SSFL, OSSL, FOSR, and the OSSFL. By adding the GS, logit adjustment and LCR to a standard architecture, experimental results show FedOpenMatch improves both closed-set and open-set balanced accuracy over federated adaptations of OSSL baselines and standard SSFL methods.

### Weaknesses
While the paper positions as a new OSSFL formulation, its novelty relative to prior open-world SSFL work is limited and largely rooted in scenario framing rather than a new learning principle. The authors should better justify why the label-at-server configuration represents the dominant or more practical case, and clarify how insights generalize across settings. As it stands, the novelty appears to hinge more on scenario framing than on a fundamentally new learning principle.

Additionally, centralized OSSL algorithms appear to be adapted to FL. Their relatively weak and unstable performance raises concern that implementations or hyperparameters under FL are suboptimal. More evidence could be helpful to show these are strong federated instantiations rather than strawmen.

Further, the evaluation setup is narrow, using only ResNet-18, limited Dirichlet heterogeneity, and no covariate or mixed-shift scenarios. Key design details, such as pseudo-label staleness under non-IID drift, are under-analyzed. Finally, grouping all unseen classes into a single "unknown" category obscures variability.

### Questions
N/A

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
This paper defines Open-set Semi-Supervised Federated Learning (OSSFL) (unlabeled client data contain unseen classes) and presents FedOpenMatch, the first framework for this setting. It combines an OVA outlier detector with logit adjustment for imbalance, a gradient-stop to decouple OVA/inlier heads, and logit consistency regularization. Experiments on standard benchmarks show sizable improvements in open-set accuracy (e.g., +14.33% on CIFAR-100).

### Strengths
- First formalization of OSSFL and a tailored framework
- Strong, consistent gains; well-motivated components (OVA + adjustment + gradient-stop + consistency)
- Good figures/tables and taxonomy (SSFL vs. OSSL vs. FOSR vs. OSSFL).
- Addresses a real-world failure mode for SSFL; likely baseline for future work.

### Weaknesses
- Limited analysis of non-IID client shifts (class/feature distribution) on OVA calibration.
- Communication/compute overhead of OVA and consistency losses isn’t profiled; end-to-end systems analysis would help.
- Robustness to extreme open-set ratios and ablations isolating each component could be expanded.

### Questions
- How sensitive is OVA performance to class imbalance and thresholds under client heterogeneity?
- Can gradient-stop harm representation sharing between inlier/OVA? Any alternatives (e.g., orthogonal heads)?
- What happens when unknown prevalence is very high or client pools have disjoint seen sets?

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
The paper introduces **FedOpenMatch**, the first framework for **Open-Set Semi-Supervised Federated Learning (OSSFL)**—a realistic yet previously unexplored setting where unlabeled client data contain **outliers**. Unlike standard SSFL methods that assume closed-set label spaces, FedOpenMatch jointly trains an **inlier classifier** and a **one-vs-all (OVA) outlier detector** to safely leverage open-set unlabeled data.  Extensive experiments on CIFAR-10, CIFAR-100, and SVHN show FedOpenMatch significantly boosts.

### Strengths
The paper demonstrates good **originality** by formally defining a new and realistic problem setting—Open-Set Semi-Supervised Federated Learning (OSSFL)—which bridges the gap between open-set semi-supervised learning and federated learning. The experimental design is also **rigorous**, covering multiple datasets, label scarcity regimes, and heterogeneity levels, with thorough ablation studies validating each component. The writing is clear and well-structured.

### Weaknesses
1. The method assumes the server’s labeled set is perfectly balanced and clean, which may not hold in real-world label-at-server settings; robustness to label noise or skewed class priors is unexamined. 

2. Second, all experiments use ResNet-18 and synthetic Dirichlet splits—evaluations on larger models (e.g., ViTs) may change the conclusion on feature conflicts between the OVA classifier and the inlier classifier.

### Questions
Does your method work with bigger models or more realistic data splits?

All experiments use ResNet-18 and synthetic data splits (Dirichlet). But real-world data may have domain shifts (e.g., different hospitals or cameras), and people now often use Vision Transformers (ViTs).  
- Have you tried FedOpenMatch with a ViT or a more realistic non-IID split (e.g., by domain or semantic group)?  
- Would the “gradient stop” still be needed in those cases, or is it only helpful for ResNet-18?

### Soundness
3

### Presentation
3

### Contribution
3
