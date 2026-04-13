## Human Reviewer 1

### Summary
This paper proposes calibrated deep clustering, a method to calibrate the confidence of deep clustering networks. Specifically, it proposes a separate calibration head alongside the traditional clustering head to optimize the learning and prediction confidence of clusters. This approach leads to better-calibrated models as it ensures that the calibration head consistently achieves lower errors compared to the clustering head. The authors provide a theoretical proof to demonstrate that the calibration error from the calibration head is less than or equal to that of the clustering head, supporting the validity of their method.

### Strengths
1. The authors provide a comprehensive comparison of their method against 14 prior works. They have also tested the method on 6 datasets, which is a sufficient number to validate their approach. The experimental results clearly demonstrate that CDC achieves significantly better performance than previous methods. The experimental design, including both the main experiments and ablation studies, is rigorous and well-justified.
2. The figures are well-designed and effectively highlight the advantages of the proposed approach. Both the presentation of the results and the visuals are compelling and easy to interpret.
3. The authors offer sound theoretical proof showing that the proposed method achieves a lower expected calibration error and penalizes only unreliable regions, adding to the method’s credibility.
4. The motivation behind the work is well-justified. Both the introduction and abstract are clearly written and provide a strong foundation for the paper.

### Weaknesses
1. The authors should reconsider the structure of the paper. In my view, the methods section should first focus on the network and training design, followed by secondary aspects like initialization strategies. While initialization techniques can improve performance and speed up convergence, they are supplementary rather than a primary contribution. Therefore, dedicating excessive space to this, even with theoretical justifications, is unnecessary. Additionally, the experimental results section is too brief. I recommend condensing the methods description and moving a more detailed analysis of results from the appendix into the main body of the paper.

2. The notation is inconsistent and confusing. Several terms are introduced without adequate mathematical explanation. For example, $p_i^{clu}$ is first mentioned in Line 230 as "the prediction of $x_i$ by the clustering head" but lacks a formal mathematical definition. I suggest including an equation, such as defining the clustering head as $g(\theta_{clu},\cdot)$ and expressing $p_i^{clu} = g(\theta_{clu}, x_i)$. A similar issue with $p_i^{cal}$ in Line 235. When introducing the proposed pipeline, the authors should clearly guide the reader through the entire process, from input sample $x$ to the final loss function, in a step-by-step manner in the methods section.

3. The title of Section 3 should be 'Proposed Method'.

Overall, the paper presents promising results and good analysis. However, the presentation of the methodology is lacking, which makes the paper difficult to follow. Revising the organization and clarity of the exposition is neccessary.

### Questions
1. The 'classifier $g(\theta, \cdot)$' is mentioned only in the Notation and Algorithm 1. However its role in the proposed pipeline remains unclear to me. Can you clarify its specific function within the pipeline? Is this term meant to generally refer to both the clustering head and calibration head?
2. Can you provide a detailed algorithm outlining how prediction is done?
3. In Equation 4, is the total loss simply the sum of the two individual losses? Is there a hyperparameter that controls the weighting between them? Are the two losses on similar scales? Additionally, is the use of the negative entropy loss inspired by prior works? It would be helpful to see an ablation study that removes this term and compare the results.
4. How does the algorithm perform in the presence of out-of-distribution (OOD) samples? Specifically, could you provide a Density-confidence plot as in figure 3, when OOD samples are introduced in the dataset similar? One possible way to generate OOD samples is by adding random Gaussian noise to selected samples or by mixing data from irrelevant datasets.

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes a dual head deep clustering model that can effectively calibrate the estimated confidence and the actual accuracy.

### Strengths
This paper proposes a dual head deep clustering model that can effectively calibrate the estimated confidence and the actual accuracy.

### Weaknesses
According to the discussion between the authors and other reviewers, I altered my rating.

### Questions
No

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper considers calibration in the context of deep clustering, more specifically pseudo-label self-training based deep clustering. The authors propose an architecture consisting of a backbone (trained in a self-supervised manner) and two heads, where one is the common clustering head, while the other is a calibration head. The calibration head obtains calibrated confidences by overclustering the data, computing the average prediction for each of the clusters, and then aligning the predictions of the calibration head for members of the clusters to the clusters average prediction. The better calibrated predictions can then be leveraged for the pseudo-label selection. Beyond the calibration head, the authors propose a new initialization strategy for clustering (and calibration) heads that does ensure initialization that resembles the k-means performance on the backbone features as well as a mechanism to dynamically select the threshold for pseudo-label selection for each class. The proposed approach is evaluated on six datasets, obtaining promising performance and lower expected calibration error compared to prior deep clustering baselines.

### Strengths
- The paper is overall well written and the contributions presented in a clear and well-motivated manner. 

- Considering calibration in deep clustering is growing more important as state-of-the-art methods rely increasingly on pseudo-label self-training and has been not been addressed previously to the reviewers knowledge.

- The proposed approaches are intuitive and effective as indicated by the empirical results and the ablation study. 

- Empirical results are supported by theoretic insights and analysis.

### Weaknesses
Hyperparameter selection is challenging in a clustering setting when no labeled data is available. It is therefore beneficial to have approaches that are robust with respect to hyperparameters. Currently, the sensitivity to the added hyperparameters is not discussed and relative specific hyperparameters are defined for different datasets (B, K) or not provided (H). How were these hyperparameters selected and how sensitive is the model to these choices. Do the authors have insights on how these should be chosen in the typical clustering setting where labeled data is unavailable? 

One weakness of the work is that results are currently reported over single runs and without information about variability. Deep clustering approaches tend to be less stable than their supervised counterparts and some more information on the robustness of these results would therefore be beneficial.

The claim in the abstract regarding surpassing state-of-the-art deep clustering methods by 10x in term of expected calibration error should be revised to clarify that this is compared to only some of the baselines and on some of the considered datasets.

### Questions
Could the authors comment on the sensitivity with respect to the hyperparameters as well as on the variability in results over multiple runs?

Should the subscripts be switched in Theorem 1? So $E_F[Conf^{cal}]$ <= $E_F[Conf^{clu}]$ and  $E_T[Conf^{cal}]$ = $E_T[Conf^{clu}]$?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors identify the problem of overconfident predictions in deep clustering methods. To overcome this problem, they propose a dual head design with a dedicated calibration head aside the usual clustering head to improve deep clustering performance and calibration. Further, they propose an initialization strategy for the dual heads that improves stability when transitioning between the pretraining and deep clustering phases.

### Strengths
**Originality**
- The authors identify and investigate the issue of overconfidence in deep clustering methods. Based on their findings they propose a novel Calibrated Deep Clustering (CDC) method.  This is a quite an interesting angle to improve deep clustering methods and connects the field to confidence calibration methods from (semi)-supervised learning. 

**Quality**
- The authors provide extensive experiments across different benchmark data sets and deep clustering methods. Further, they made the extra effort of reimplementing SeCu, SCAN and CC to obtain a unified benchmarking environment increasing the fairness of the comparison. I would highly encourage the authors to release their code with the full benchmarking environment, as this will benefit the deep clustering community. Additionally, the authors provide ablation studies and a hyperparameter analysis for their algorithm CDC.

**Clarity**
- The paper is well written, and all design decisions are clearly motivated

**Significance**
- Their method CDC combines SCAN’s self-labeling procedure with a calibration head.  As self-labeling is widely used and their method does not make specific assumptions about the underlying deep clustering method, I believe this method can impact many deep clustering methods going forward.

### Weaknesses
- Unclear reporting of experimental results.  The authors fail to mention if they report average performance results in Table 2 and other figures, or if they report the results of a single (best) run. As many deep clustering methods have high variability across different random seeds it would be important to report standard deviations as well, e.g., SCAN reports standard deviations of up to 3% for CIFAR-20.  Currently, it is not possible to judge whether the proposed method CDC is stable across multiple runs, because no standard deviations are reported and it is not mentioned if several runs are conducted at all.

- The hyperparameter for mini-clusters K needs to be specified quite differently for each data set ranging from 40 to 1000. Further, K does not seem to be related to the ground truth number of clusters of each data set and the authors do not provide any unsupervised heuristic that can be used to set K. The same holds true for the batch size B.

- The authors do not specify how they tuned their MoCo-v2 backbone. Further, no hyperparameters for the pretraining are reported. This is an important aspect that needs to be reported for reproducibility.

### Questions
- What exactly is reported in Table 2? Are these the best or average results over multiple runs? If yes, how many runs were conducted and what are the standard deviations across runs? See also the corresponding weakness described above.

- Hyperparameter sensitivity analysis w.r.t. K in Figure 8 shows that CDC is stable for different K for CIFAR-20 and STL-10, but it is set quite different for each data set.  Is there a heuristic to set K or does K need tuning w.r.t. the ground truth labels for each data set? See also the corresponding weakness described above.

- Is the MoCo-v2 pretraining used for CC, SCAN and SeCu as well or just for your method? Based on your description it is not entirely clear whether you just use the same ResNet-34 backbone for these methods or if you use MoCo-v2 pretraining for them as well. 

- Please clarify how you conducted the MoCo-v2 pretraining and which hyperparameters you used for each data set.

- Based on my understanding your method could be used as a dedicated self-labeling stage on top of any deep clustering method, similar to how SCAN's self-labeling is currently used to improve clustering performance after the joint deep clustering stage. I would be interested to see how your method would improve, e.g., SeCu after its deep clustering stage. This would further illustrate the general applicability of your method and increase its significance for the deep clustering community.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper addresses the over-confidence problem in deep clustering, where the predicted confidence for a sample significantly exceeds its actual accuracy. The authors propose a novel calibrated deep clustering framework that consists of a dual-head model, including a calibration head and a clustering head. The calibration head corrects the overconfident predictions from the clustering head, aligning estimated confidence with the model's learning status. The clustering head then selects reliable high-confidence samples for pseudo-label self-training. Additionally, the paper introduces a new network initialization strategy that improves training speed and robustness. The proposed model demonstrates significant improvements over state-of-the-art methods, achieving a 10× reduction in expected calibration error and enhanced clustering accuracy, supported by solid theoretical guarantees.

### Strengths
- Novel Approach to Calibration: The introduction of a dual-head model that effectively calibrates confidence estimates addresses a critical gap in deep clustering, enhancing the reliability of predictions.

- Improved Self-Training Mechanism: By leveraging high-confidence samples for pseudo-label self-training, the proposed method improves the quality of the training data, which is crucial for achieving better clustering performance.

- Robust Experimental Validation: The extensive experiments conducted demonstrate the effectiveness of the calibrated deep clustering model, showcasing significant improvements in both expected calibration error and clustering accuracy compared to existing methods, which adds credibility to the proposed approach.

### Weaknesses
- This paper would benefit from a more comprehensive literature review. Specifically, it overlooks some closely related works, such as [a,b], which address the over-confidence issue for standard supervised learning or unsupervised domain adaptation (i.e., unlabeled target data clustering with the help of labeled source data). I recommend that the authors include a discussion of these works and its connections to their research. This will help to contextualize their contributions within the existing body of knowledge.

[a] Revisiting the Calibration of Modern Neural Networks. NeurIPS, 2021.

[b] Transferable Calibration with Lower Bias and Variance in Domain Adaptation. NeurIPS, 2020.

- The authors mention that the calibration head adjusts the overconfident predictions of the clustering head to generate prediction confidence that aligns with the model's learning status. However, in the context of completely unsupervised learning and clustering, it is not clear how the model's learning status is determined without any labeled data. I recommend that the authors provide a more detailed explanation of the methodology used to assess the model's learning state in the absence of supervision. This clarification is essential for understanding the effectiveness of the calibration head and its role in improving prediction confidence.

- The authors state that the clustering head dynamically selects reliable high-confidence samples estimated by the calibration head for pseudo-label self-training. However, it would be beneficial to explore whether insights from existing semi-supervised learning (SSL) methods, such as those discussed in [c], could be incorporated to further enhance the self-training process.

Additionally, a critical analysis of current self-training methods is warranted to determine whether they contribute to poor calibration of predictions. It would be valuable to quantify the extent of this calibration issue and discuss its implications on model performance. By addressing these points, the authors could strengthen their argument and provide a more comprehensive understanding of the calibration dynamics in their approach.

[c] Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning. CVPR, 2022.

- The authors note that existing approaches rely on a fixed threshold to select pseudo labels, which does not account for the model's learning status. This can lead to issues during different training stages: a higher threshold in the early stages may result in selecting fewer and class-imbalanced samples, hindering convergence speed, while in later stages, a fixed threshold may introduce noisy pseudo-labels due to increased predicted confidence.

It would be beneficial for the authors to consider incorporating insights from existing SSL works, such as [d] and its subsequent studies, which have proposed dynamic thresholds and class-wise thresholds. These methods could potentially enhance the adaptability of pseudo-label selection throughout the training process, improving both the quality of the training data and the overall performance of the model. A discussion on how these approaches could be integrated or compared with the proposed methodology would strengthen this work.

[d] FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling. NeurIPS, 2021.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4