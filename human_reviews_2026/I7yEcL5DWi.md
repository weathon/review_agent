# Rethinking Consistent Multi-Label Classification Under Inexact Supervision

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4, 8, 6

## Abstract
Partial multi-label learning and complementary multi-label learning are two popular weakly supervised multi-label classification paradigms that aim to alleviate the high annotation costs of collecting precisely annotated multi-label data. In partial multi-label learning, each instance is annotated with a candidate label set, among which only some labels are relevant; in complementary multi-label learning, each instance is annotated with complementary labels indicating the classes to which the instance does not belong. Existing consistent approaches for the two paradigms either require accurate estimation of the generation process of candidate or complementary labels or assume a uniform distribution to eliminate the estimation problem. However, both conditions are usually difficult to satisfy in real-world scenarios. In this paper, we propose consistent approaches that do not rely on the aforementioned conditions to handle both problems in a unified way. Specifically, we propose two risk estimators based on first- and second-order strategies. Theoretically, we prove consistency w.r.t. two widely used multi-label classification evaluation metrics and derive convergence rates for the estimation errors of the proposed risk estimators. Empirically, extensive experimental results on both real-world and synthetic datasets validate the effectiveness of our proposed approaches against state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a unified framework, COMES, for consistent multi-label classification under inexact supervision, targeting both Partial Multi-Label (PML) and Complementary Multi-Label (CML) learning. The authors aim to overcome the limitations of existing methods, which rely on either estimating the complex label generation process or adopting a uniform distribution assumption. The core contribution is a new data generation assumption—that true negatives are marked as non-candidates with a constant, instance-independent probability—which allows for a new approach. Based on this premise, the paper derives unbiased risk estimators (and their subsequent corrected, consistent versions) for two widely used metrics: the Hamming loss (COMES-HL, a first-order strategy) and the Ranking loss (COMES-RL, a second-order strategy). The authors provide theoretical guarantees for the consistency and estimation error bounds of their proposed estimators. Empirically, the framework is validated on ten benchmark datasets, where it is shown to outperform current state-of-the-art methods.

### Strengths
1.  The paper is well-structured and clearly articulates its approach (COMES) as a unified solution for both PML and CML problems.
2.  It introduces a new data generation assumption that avoids the common pitfalls of transition matrix estimation or uniform distribution assumptions.
3.  The proposed methods are supported by both theoretical guarantees and extensive empirical validation.

### Weaknesses
1. There is a contradiction between the method's motivation and its empirical results. The second-order strategy (COMES-RL), which was introduced specifically to model label correlations, paradoxically performs significantly _worse_ than the first-order strategy (COMES-HL) on datasets with strong label correlations (e.g., CUB and COCO), an inconsistency the authors do not address.
    
2. The paper's claim of proposing "unbiased risk estimators" is misleading. The practical estimators (Eq. 8, 14) used in the algorithm are "corrected" versions that are _biased_ (to avoid overfitting from the original unbiased forms).
    
3. The method's reliance on accurate class-prior estimation ($\pi_j$) is a critical vulnerability.

### Questions
See in weekness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposed the consistent approaches to handle both partial label and complementary multi-label problems in a unified way, with two unbiased risk estimators based on first- and second-order strategies. The theoretical work is elegant and self-contained. In addition, the empirical study largely validates the effectiveness of the proposed approaches.

### Strengths
1. The paper innovatively transforms the PML problem into a negative-unlabeled (NU) learning problem through a carefully derived loss function, providing a clear and elegant theoretical perspective. 
2. The theoretical analysis is rigorous, with well-stated assumptions, formal consistency proofs, and convergence rate derivations. 
3. The empirical evaluation is thorough, covering both standard benchmarks and additional real-world datasets, which enhances the credibility and generality of the proposed method. 
4. The paper is well-motivated and clearly written, with smooth logical flow and sound reasoning connecting problem definition, theory, and empirical validation.

### Weaknesses
1. The derivation in Lemma 1 heavily relies on assumptions, which may be difficult for readers to intuitively grasp. I would suggest providing more intuitive insights or illustrative examples to clarify the underlying rationale of the lemma and its implications for the overall theoretical framework. 
2. The rank loss based on the second-order strategy appears to primarily capture the gap between the ground-truth and non-ground-truth labels, but not the relevance among ground-truth labels themselves. It would be interesting to discuss whether a rank loss can be designed to simultaneously model both the inter-ground-truth relevance and the ground-truth and non-ground-truth gap. 
3. In the experiments on synthetic benchmark datasets, the differences between case-a and case-b are not clearly analyzed. It would strengthen the empirical section if the authors could elaborate on the motivation for setting up these two distinct cases, and provide a detailed discussion of the respective findings and their implications. 
4. Although this paper focuses on weakly supervised multi-label learning, the proposed loss formulations and theoretical derivations seem readily transferable to weakly supervised multi-class settings, such as partial label learning and complementary label learning. A discussion along with potential preliminary experiments on this broader applicability would further highlight the generality and impact of this paper.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the partial/complementary multi-label learning problem. By proposing a consistent multi-label classification
under inexact supervision framework called COMES, this paper designs two risk-consistent estimators w.r.t. two classic multi-class classification losses, i.e., the Hamming loss and the ranking loss. Compared with previous works, the proposed estimators neither require estimating the generation process of candidate or complementary labels nor rely on the uniform-distribution assumption. In the theoretical parts, this paper derives the generalization bounds for the proposed estimators (COMES-HL and COMES-RL). In experiments, the proposed estimators are evaluated on both real-world and synthetic PML benchmark datasets, achieving lower errors and higher average precision compared with previous methods.

### Strengths
- This paper is overall well written and easy to follow.
- The proposed methods are theoretically inspired and proved to be consistent. Although I did not check every detail in the proof, the theoretical results seem sound and reasonable.
- The proposed methods neither require estimating the generation process of candidate or complementary labels nor rely on the uniform-distribution assumption, which is a significant advancement compared to existing methods.

### Weaknesses
**Major**
- In Theorem 2, if the non-negative $\alpha=0$, according to (9), the estimator is inconsistent because the bound becomes independent of $n$. For example, if the classification problem is easy enough so that $g_j(x)$ predicts every $y$ exactly, then theoretically, we have $\pi_j\mathbb{E}\_{p(x|y_j=1)} [\ell(g_j(x),1)]=0$. Nonetheless, I believe that the estimator can still be proved consistent in such a corner case, because the classification problem becomes very easy now. I hope the authors could discuss this case in further detail so that the theoretical results can be more rigorous and complete.
- Similarly, in Theorem 5, it is also possible that $\gamma=0$ for easy classification problems, making the bound independent of $n$.
- Another of my major concerns is about the fair comparison in the experiment section. In Figure 2, the inaccurate class priors affect the performance of COMES. Even for the relatively robust COMES-RL, under a slight noise $\theta=0.1$, the average precision drops unacceptably. For example, on mirflickr, mAP drops from 0.818 to approximately 0.80, and on music_style, mAP drops from 0.732 to below 0.70, which makes COMES-RL perform worse than many of the compared methods in Table 2. Therefore, I wonder if these compared methods also use the true class priors in Table 2. If so, from my perspective, an additional comparison under estimated class priors may be more persuasive.

**Minor**
- In Eq.(8), an absolute value function is used to prevent overfitting. Further explanations are needed to elaborate on how and why this approach works. 
- In Section 3.1, the generation procedure of non-candidate labels is class-dependent instead of instance-dependent. I think this point should be emphasized in the context.
- Line 250. "$\sup\_{g_i \in \mathcal{G}}\\|g\\|_\infty$". Should it be "$\sup\_{g_i \in \mathcal{G}}\\|g_i\\|\_\infty$"?

### Questions
- In Eq. (13), there is no negative loss term to induce overfitting. Why is the flooding regularization technique necessary?
- Why do Eq. (8) and Eq. (14) use different strategies to avoid overfitting?
- Just out of curiosity, why are the proposed methods named first-order and second-order strategies when no derivatives are involved?
- How do the proposed methods perform under estimated class priors?

### Soundness
2

### Presentation
3

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
The paper proposes the COMES framework for partial/complementary multi‑label learning (as the paper points out, these settings are formally equivalent). The paper focuses on deriving two new unbiased losses for Hamming loss (COMES‑HL) and ranking loss (COMES‑RL) under the setting with the assumption that each label has a different but constant (not instance-dependent) probability of being in the candidate set, while being in fact irrelevant for the sample. The authors prove consistency with finite‑sample bounds for both derived losses and perform experiments on six real‑world PML datasets and four synthetic ones, and compare against 5 different algorithms across Hamming loss, ranking loss, one‑error,  coverage, and AP.

### Strengths
- The paper sounds and is easy to read.
- Clear theoretical contribution: unbiased estimators with finite‑sample bounds for both Hamming and ranking loss.
- Useful relaxation of assumptions compared to previous methods.

### Weaknesses
- The biggest weakness of experiments is that they assume the priors are known, but the problem of estimating priors is, in this case, complex, and the problem may not be identifiable in some cases.
- It is not clear what dataset is used for Figures 2 and 3; comparison with baselines could also be added there.
- While new losses relax the assumption on uniform distribution, calling them general is an overstatement for me, as obviously, the assumption on constant p_j is still strong and likely untrue in many real-world cases.
- "This data generation process coincides well with the annotation process of candidate labels. For example, when asking annotators to provide candidate labels for an image dataset, we
can show them an image and a class label and ask them to determine whether the image is irrelevant
to that class. This is often an easier question to answer than directly asking all relevant labels, since
it is less demanding to exclude some obviously irrelevant labels." - depends, if there is a lot of labels, is it really better to list irrelevant ones? Not sure about that, but I recall that datasets used in experiments were actually created using crowdsourcing, and candidate label sets were created by taking the union of all assigned label sets. Are there any datasets created in this manner?
- Experiments lack a good description of datasets and baseline methods (also in the appendix).
"We evaluate against five classical baselines commonly used in PML/CML learning." - Only classical? None of it is SOTA? What about the assumption these methods are using, do they also require priors? Also, what worries me is that from those baselines, the simplest BCE performs the best most of the time. This makes me question baseline choice and correctness of presented experiments.
A comment how assumptions of the method match data annotations of benchmark datasets would be nice. Also, are all the real datasets indeed real? I might be wrong, but I think some of them are actually created synthetically (yeast ones?). Limited information on data splits/repetitions etc.
- NIT: There are multiple metrics in MLC called "coverage", I assume the authors use the minimal ranking coverage metric here because the stated lower is better, but it would be nice to have metrics defined in the appendix.
- NIT: It's not clear from the main text what is Case-a and Case-b in Table 3
- NIT: When I use the code link, for every file I select, it says: The requested file is not found. Basically, the code is not accessible at the moment of writing this review.

### Questions
Please refer to the weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addressed the Partial and Complementary multi-label learning in a unified way. The authors proposed unbiased risk estimators for hamming and ranking loss under inexact supervision, and provided consistency convergence guarantees. In comparison to existing approaches, the proposed method does not rely on accurate estimation of data generation process. They also showed the effectiveness of the framework through empirical results.

### Strengths
Originality: The problem of PML and CML have been addressed separately but this paper provides a unified framework to address the two settings. The authors proposed the method which generalize the label generation process (instead of naively assuming the process) for risk estimators.

Quality: The methodology demonstrates technical soundness. The assumptions and lemmas have extensive proofs and details. Based on the experimental section, the method seems to perform well empirically too.

Clarity: The paper is mainly easy to follow and has proper motivation of the problem setting. 

Significance: The paper makes substantial contributions to the area of multi-label learning. I think the results are significant (the theoretical and experimental, both).

### Weaknesses
1. Although the method is more general than prior work, it depends on a strong assumption used in Lemma 1, which results in the independence of the data generation process from the samples. (Authors mentioned this in the paper as well)
2. I am particularly focused on the result of the Lemma that $p(x|s_j = 0) = p(x|y_j = 0)$. This assumption seems to be a very strong signal for negative labels. This implies a perfect annotation process for 'irrelevant' labels. The proof of the lemma uses another assumption: $p(j \notin Y | x, s_j = 0) =p(j \notin Y | s_j = 0). $, which has not been explained in the paper. 
3. The experimental section could be improved a lot. The authors have done experiments but the rationale behind case A and case B is not mentioned. Why choose this particular method of generating the labels and why not some other method (or different \tau)? 
4. For the effectiveness of the framework, and the impact of the data generation process, there is a need for more experiments. What would be the effect of changing the dataset size for training?

### Questions
1. In figure 2, the experiments are done with different sigmas for class priors. Why not actually modify the flip rate in case A for example, and then evaluate?
2. The average precision of mirflickr stays the same as you increase sigma. Why? 
3. This paper consider data generation process independent from the samples. Shouldn’t MLCL perform better because it is estimating the data generation process from the samples? Can you elaborate? Because I would assume that estimating the data generation process based on the knowledge of the actual samples should be better than independent data generation process (in this case, flipping the negative labels to positive labels with prob 0.9). 
4. Theorem 4 assumes that $l$ is symmetric but in reality, the used loss binary cross entropy, is not symmetric. Am I missing something?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses multi-label classification under inexact supervision task, proposing a unified framework COMES for partial multi-label learning (PML) and complementary multi-label learning (CML). Existing methods require estimating label generation processes or rely on uniform distribution assumptions. The paper assumes that candidate labels are generated by querying whether an instance is irrelevant to each class, and designs first-order and second-order unbiased risk estimators. The first-order strategy decomposes the problem into multiple binary classification problems using Hamming loss, while the second-order strategy takes label correlations into account using ranking loss. Theoretically, the paper proves consistency and derives convergence rates for both Hamming and ranking losses, improving generalization through absolute value wrapping and flooding regularization. Experiments validate the effectiveness on six real-world and four synthetic datasets, with COMES significantly outperforming baselines including CCMN, GDF, CTL, and MLCL in most cases.

### Strengths
1.	The unified perspective on treating PML and CML as equivalent problems is interesting.
2.	This paper not only proves consistency of risk estimators but also derives convergence rates for estimation errors. It establishes generalization bounds by Rademacher complexity analysis, and proves that minimizing the corrected risk estimator can achieve Bayes risk.
3.	This paper covers different types of datasets (images, audio, biological information, etc.) with reasonable settings. Sensitivity analyses evaluate the impact of inaccurate class priors and hyperparameter β, and ablation studies validate the necessity of each module.

### Weaknesses
1.	Algorithm 1 contains an obvious error in line 8 of the pseudocode. The conditional branch "else if using the COMES-HL algorithm then" should be changed to COMES-RL rather than COMES-HL. 
2.	This paper provides a detailed introduction to first-order and second-order strategies and validates their effectiveness through experiments. However, it does not explain how to reasonably select or combine first-order and second-order strategies in practical applications.

### Questions
1.	Is the conditional branch error in Algorithm 1, line 8 a typesetting issue or an actual error in the code? Does this error exist in the algorithm implementation, or is it a description error in the main text?
2.	What is the reason for using different network architectures (MLP and ResNet-50) in experiments? Does this design affect the fairness and comparability of experimental results?
3.	How should first-order and second-order strategies be chosen in practical applications? Has the author considered designing a mechanism to automatically select one strategy or a weighted combination of the two strategies based on dataset characteristics, such as label correlation strength or dataset size?

### Soundness
3

### Presentation
3

### Contribution
3
