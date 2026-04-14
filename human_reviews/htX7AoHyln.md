# GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experiential results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GSBA$^K$, a comprehensive and query-efficient geometric score-based attack specifically designed for the challenging top-K setting. GSBA$^K$ incorporates innovative gradient estimation techniques to identify a better initial boundary point and exploits the geometric properties of decision boundaries to improve both query efficiency and attack adaptability. The top-K setting presents significant difficulties in adversarial attacks, particularly for targeted attacks. The proposed method combines a novel gradient technique with the existing CGBA approach to address the complexities of score-based black-box attacks in the top-K scenario. While the proposed approach is promising, the clarity of the paper’s writing limits the accessibility of the method. For instance, it is unclear what the notation $\zeta _ {\mathbf{x} _ {b_n}, c, z _ i}$ in Eq. (6) represents or why $\mathbf{x} _ {b_n}, c, z _ i$ appear in the subscript. The distinction between the two versions of gradient estimation—one on the decision boundary and the other in the non-adversarial region—warrants clarification. An overview would help provide a broader perspective on why both approaches are necessary and how they contribute differently to the attack method. Most importantly, the paper lacks a theoretical analysis of the accuracy of the gradient estimation (e.g., examining the expected cosine similarity between the estimated gradient and the true gradient).

### Strengths
The top-K setting poses substantial challenges for adversarial attacks, especially in targeted scenarios. The proposed method integrates a novel gradient technique with the existing CGBA approach to tackle the complexities of score-based black-box attacks within the top-K framework. Experimental results demonstrate promising performance.

### Weaknesses
1. Although the ablation study and experiments in the Appendix provide valuable insights, the paper *lacks a theoretical analysis of the gradient estimation accuracy* (e.g., evaluating the expected cosine similarity between the estimated and true gradients). Without this analysis, the quality of the estimated gradient remains uncertain from a theoretical perspective. 

3. Some baseline comparisons appear to be missing. For instance, the QuadAttacK method [1] is not included in the experiments. How does the ASR of GSBA$^K$ compare to QuadAttacK?

3. The writing in Sections 4.1 and 4.2 makes it difficult to understand the proposed approach. For example, the notation $\zeta _ {\mathbf{x} _ {b_n}, c, z _ i}$ in Eq. (6) is unclear, and the reason for including $\mathbf{x} _ {b_n}, c, z _ i$ in the subscript is not well-explained. The symbols used in the formulas are overly complex and lack brevity.

[1] Paniagua, T., Grainger, R., & Wu, T. (2023). QuadAttacK: A Quadratic Programming Approach to Learning Ordered Top-K Adversarial Attacks. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.

### Questions
1. Targeted attacks in the top-K setting for black-box scenarios are highly challenging. A key question is how to ensure that the estimated gradient consistently points toward the narrow adversarial region corresponding to the $K$ target classes. Additionally, how is a high attack success rate (ASR) maintained in such targeted attacks?

2. Could you provide a simplified overview explaining why there are two versions of gradient estimation (one on the decision boundary and the other in the non-adversarial region)? This would help make the rationale for this choice more accessible and easier to understand.

### Soundness
3

### Presentation
3

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
The paper addresses the problem of score-based adversarial attacks on top-K predictions. They propose a novel GSBA^K attack that uses gradient estimation in the non-adversarial region and on the decision boundary to craft adversarial perturbations. The authors discuss both targeted and untargeted settings. The approach is extensively evaluated on multiple datasets and compared to existing methods in the top-K setting.

### Strengths
Originality: the paper proposes a novel and efficient black-box attack that relies on existing approaches to some extent but contains original elements such as novel gradient estimation methods.

Quality: proposed attack is extensively evaluated and compared to existing work. The asttack is tested on both vanilla and robust models (Table 6). Limitations and Social Impact are discussed in the Appendix F.

Reproducibility: the code is provided in the supplementary materials, not carefully checked. 

Clarity: the paper is well structured, the geometrical intuition is illustrated in numerous Figures. 

Significance: the paper seems to introduce valuable improvement into the black-box robustness evaluation field and addresses a challenging and insufficiently investigated field of top-K predictions.

### Weaknesses
Some minor points:

1) Avoid duplication of very long expressions in Equations (7), (9), (12), (13) to improve readability. Consider denoting the nominator as h and use g = h / ||h||_2

3) typo in line 024: experiental

### Questions
1) line 271: “experimentally it is observed that …” - do you have an explanation for this observation? It seems a bit counterintuitive to me. 

2) In the line 396, could you please also state the number of classes in each dataset, especially Pascal VOC? It could be useful to compare the number of target classes K that you use in your attacks with the total number of classes N. Have you investigated extreme cases e. g. when K ~ N/2 i. e. we need to completely shift the prediction scores?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a novel setting in black-box adversarial attacks, specifically focusing on generating top-k adversarial examples with access to the target classifier's class prediction probabilities. While traditional decision-based black-box attacks rely solely on class labels, this work leverages additional information in the form of prediction probabilities to enhance attack effectiveness.
The authors propose a geometry-based framework that generates both targeted and untargeted attacks without requiring a surrogate model. The method operates iteratively: given an input data point and query access to a black-box model, it systematically updates the data point to approach the decision boundary through repeated classifier queries. At the initial boundary point, by exploiting the geometry of the decision boundary, the framework identifies boundary points with reduced perturbation norms.
Comprehensive experiments presented in both the main paper and appendix demonstrate that the additional information provided by prediction probabilities significantly enhances attack effectiveness. While existing methods fail to adequately leverage this information and perform poorly in this novel setting, the proposed approach consistently achieves superior results compared to current alternatives.

### Strengths
**Originality**: This work presents the first systematic investigation of black-box adversarial attacks in the top-k setting with access to prediction probabilities, addressing a previously unexplored practical scenario.

**Quality**: The methodology demonstrates technical soundness, particularly in its approach to gradient approximation. The rationale of approximating gradients in adversarial and non-adversarial regions in different ways for top-k setting makes sense. 

**Clarity**: The paper is mostly clear and well-written. However, there are some missing details which I have asked in the question section. In terms of logical flow, I personally think that the section about estimating the gradients in non-adversarial region should be discussed before boundary region since that is the first step in the attack generation.

**Significance**: The paper makes substantial contributions to adversarial machine learning by introducing a novel attack setting that leverages prediction probability information in top-k predictions. The paper demonstrates how this additional information can be effectively used.

### Weaknesses
There are a few key weaknesses:
1. Lines 270 - 271: experimentally, it is observed that $\zeta\_{\boldsymbol{x}\_{b_n}, c, \boldsymbol{z}\_i}=\zeta\_{\boldsymbol{x}\_{b_n}, -c, \boldsymbol{z}\_i}$. This might be true for some directions, but this cannot be guaranteed. The trained models can have flat surface across a direction. The formula (7) is based on that. I understand that this works in practice, but this does not have a strong/concrete evidence. 
2. One drawback of (5) is that the formulation does not consider relative change i.e., the change in a class prediction from 0.5 → 0.52 is considered the same as 0.02 → 0.04 where one might argue that the latter is better than the former.
3. Similarly, Eq(12) can weigh two directions/vectors equally even if one would overall hurt the performance of majority of classes as compared to the other. For example, consider two vectors z1 and z2. Assume we are considering top-3 attack. Consider the model predicts initially with the target class probabilities shown {….., 0.03,0.04,0.02} (assume the other class probabilities are fixed). Assume 

    z1 changes the predictions to {…., 0.023,0.021,0.32}: reduces the confidence of two classes and increases one.

    z2 changes the predictions to {….., 0.13, 0.14, 0.12}: equally increases the confidence of all classes.

4. Another issue is that the method might be too dependent on the initial boundary point because once it finds a point on the boundary, to find the next boundary point, it requires that the change in the probability should be larger as in Eq(5). It is actually very likely that the confidence would decrease for the top-k predictions and correspondingly might increase for the the rest but still the top k predictions remain top k. For example, {0.3,0.3,0.3,0.1} → {0.28,0.28,0.28,0.16}. This might explain why the method doesn’t perform that well for small norms. 
5. The experiments do not provide confidence intervals. For example, in table 2, it is important to know how much of deviation there is across different target classes (through standard deviation). I would highly encourage the authors to add that in the paper, if possible. 
6. Although there are extensive experiments, the missing experiments are for larger K and for multi-label datasets with larger number of labels e.g., MS-COCO, OpenImages. It would be great to see the performance for larger data sizes in multi-label settings. 
7. Lines 177 - 178: Should be $P(\boldsymbol{x}):[0,1]^{C\_h \times W \times H} \rightarrow [0,1]^C$ since the paper considers probabilities. 
8. Since the work is relevant to multi-label learning in general, it is good to cite and mention recent multi-label attacks e.g., [1,2].

--
1. Mahmood, Hassan, and Ehsan Elhamifar. "Semantic-Aware Multi-Label Adversarial Attacks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
2. Jia, Jinyuan, Wenjie Qu, and Neil Gong. "Multiguard: Provably robust multi-label classification against adversarial examples." Advances in Neural Information Processing Systems 35 (2022): 10150-10163.

### Questions
1. From Table 1, I do see that the performance on targeted attacks drops quickly as the K increases. For example, the best top-2 accuracy of 98.2% to top-4 accuracy of 81.4%. Which makes me question that how well the method would scale for larger multi-label datasets and for larger K e.g., K = 20 for OpenImages dataset? Please provide comments or any analysis.
2. It is not clear how the target labels are selected for targeted attack setting for ImageNet experiments. For PASCAL-VOC, how many experiments are performed for random target class setting in Table 2 and how much do the results vary? Did you observe any influence of choosing specific target classes on the performance?
3. For DCT, what percentage of low frequency components were selected? Have you performed any experiments to show the impact of choosing different number of components?
4. Since you are considering the setting where the sum of probabilities is 1, maximizing the probability of a set of target classes would reduce the probabilities of other classes. In multi-label learning, the labels are correlated and it is difficult to maximize a set of target class probabilities while decreasing the other highly correlated classes. This was observed in one of the recent works [1]. Did you observe something similar? Any comments?

--

1. Mahmood, Hassan, and Ehsan Elhamifar. "Semantic-Aware Multi-Label Adversarial Attacks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

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
4

### Summary
This paper presents a score-based and query-efficient blackbox adversarial attack method. The proposed method is designed based on GSBA, which searches for the adversarial samples close to the classification boundary along a semi-circular path. The merit of the proposed method is more accurate gradient estimation to update incrementally the adversarial sample in the adversarial region and along the classification boundary. This study applies the method to a challenging top-k adversarial attack scenario and demonstrate its superior performance compared to the baselines in an empirical comparison.

### Strengths
This study offers a query-efficient solution to a challenging top-K adversarial attack problem and organise a comprehensive study in both targeted and untargeted attack scenarios for single-label and multi-label learning tasks. The core idea is to boost the blackbox gradient estimator by reweighing different sample points around the current estimate, instead of treating them equally. The weight assigned to each sampled point is defined by measuring the decision confidence variation. The sampled points are considered, only if they cause the decision confidence scores all decrease with respect to the targeted top-K classes towards the adversarial region. This weighing technique better tunes the gradient estimator to be aligned to the goal of adversarial attacks. This design improves the efficiency of attacks, rather than initialising the attack with a random or less attack-optimised gradient estimates.

### Weaknesses
I have the following suggestion and comments over the limitation of this study. 

(1) This paper needs significant improvement to get a clear logic structure. The attack according to Figure.3 is initialised by locating an adversarial sample along the classification boundary, then followed by searching for more closer adversarial samples in the adversarial region along a semi-circular path. Therefore, Section 4.2 should be presented before 4.1. 

(2) Reweighing the sampled point to form a more accurate gradient estimator also reduces the number of sampled points $\{z_{i}\}$ that are useful and have non-zero weights in gradient estimation. As a result, it may produce a better gradient estimation, yet require more sampled points especially at the beginning iterations of the attack. 

(3) Why do you set the number of the queried points as $\lfloor{I_{0}\sqrt{n+1}}\rfloor$ ?

### Questions
1/ Regarding the second bullet point in the comments of weakness, please provide further discussion if the reweighing technique may require to increase the number of queries, since the reweighing step can exclude some sampled points from estimating the gradients. 

2/ Please justify the third bullet point with further discussions.

### Soundness
3

### Presentation
2

### Contribution
3
