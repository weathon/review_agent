# Does Weak-to-strong Generalization Happen under Spurious Correlations?

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We initiate a unified theoretical and algorithmic study of a key problem in weak-to-strong (W2S) generalization: when fine-tuning a strong pre-trained student with pseudolabels from a weaker teacher on a downstream task with spurious correlations, does W2S happen, and how to improve it upon failures? We consider two sources of spurious correlations caused by group imbalance: (i) a weak teacher fine-tuned on group-imbalanced labeled data with a minority group of fraction $\eta_\ell$, and (ii) a group-imbalanced unlabeled set pseudolabeled by the teacher with a minority group of fraction $\eta_u$. Theoretically, a precise characterization of W2S gain at the proportional asymptotic limit shows that W2S always happens with sufficient pseudolabels when $\eta_u = \eta_\ell$ but may fail when $\eta_u \ne \eta_\ell$, where W2S gain diminishes as $(\eta_u - \eta_\ell)^2$ increases. Our theory is corroborated by extensive experiments on various spurious correlation benchmarks and teacher-student pairs. To boost W2S performance upon failures, we further propose a simple, effective algorithmic remedy that retrains the strong student on its high-confidence data subset after W2S fine-tuning. Our algorithm is group-label-free and achieves consistent, substantial improvements over vanilla W2S fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a unified theoretical and algorithmic investigation into the phenomenon of Weak-to-Strong (W2S) generalization under the presence of spurious correlations caused by group imbalance. The authors model two key sources of imbalance: the minority fraction in the labeled data ($\eta_l$) used to train the teacher, and the minority fraction in the unlabeled data ($\eta_u$) used for W2S fine-tuning.

The main theoretical contribution, derived within the proportional asymptotic limit of a linear regression setting, precisely characterizes the W2S gain. The theory predicts that W2S always occurs when $\eta_l = \eta_u$, but critically, W2S generalization can fail when $\eta_l \neq \eta_u$, with the gain diminishing as the squared difference $(\eta_u - \eta_l)^2$ increases.

Based on these findings, the paper proposes Enhanced-W2S, a simple yet effective algorithmic remedy. This method involves post-W2S retraining of the strong student on its high-confidence data subset using the Generalized Cross-Entropy (GCE) loss. The method is entirely group-label-free. Extensive experiments across four spurious correlation benchmarks and ten different teacher-student pairs confirm the theoretical predictions and demonstrate consistent, substantial gains achieved by Enhanced-W2S over vanilla W2S fine-tuning.

### Strengths
1. The paper addresses a highly critical, yet previously unexplored, aspect of W2S generalization: its robustness to real-world biases like spurious correlations. The idea attracts me a lot and I'd like to see the core idea published in the conference. To my best knowledge, this issue has not been investigated in previous W2SG papers, both empirically and theoretically.
2. The key theoretical conclusion that W2S gain increases as the teacher-student similarity decreases is intuitive and aligns with teacher-student disagreement literature (Charikar et al., 2024; Mulgund & Pabbaraju, 2025; Yao et al., 2025; Xu et al., 2025).
3. To my knowledge, the paper includes a thorough review of relevant literature concerning W2S generalization and group robustness.
4. The result appears correct since it can recover Dong et al. (2025), though I have not verified the proof details.

### Weaknesses
## Weaknesses
1. The paper is generally well-written, but it could be significantly improved by providing more explicit and intuitive explanations for every mathematical choice and definition (e.g., the precise geometric meaning of $\left\\|\Xi\right\\|_F^2$ and the role of $p_T$ and $p_S$) alongside the formal derivations. This would greatly enhance the accessibility and understanding of the theoretical model.
2. The paper's theory successfully demonstrates that under certain ideal conditions, such as when the minority group proportions in the labeled and pseudo-labeled data match ($\eta_l = \eta_u$), W2SG is more likely to occur. However, the conclusion doesn't seem to fully elucidate the precise conditions under which W2SG is guaranteed to succeed or fail in the non-ideal scenario where data imbalance persists (i.e., $\eta_l \neq \eta_u$). The theory notes that W2S may fail when $\eta_u \ne \eta_l$. For instance, when $\eta_l \neq \eta_u$, the W2S gain tends to diminish but is not universally negative, suggesting there may be boundary conditions or trade-offs that are not fully explored by the analysis on $(\eta_u - \eta_l)^2$ alone. 
3. The detailed excess risk characterizations rely on the ridgeless linear regression setting and a specific, low-intrinsic-dimension kernel assumption.
4. The Enhanced-W2S method introduces two new hyperparameters ($p$ for subset fraction and $q$ for GCE loss). While tuning ranges are provided, the paper does not thoroughly explore the robustness of the optimal $(p, q)$ pair across different $\eta_l, \eta_u$ settings or the effort required to tune them.

Minor: Line 270: The text above Figure 2 should be revised.

## Future Work
Since the paper quantifies the error, I strongly suggest the authors separately quantify the excess risk for the minority group and the majority group in the theoretical analysis. The current theoretical results (Theorems 1 and 2) characterize the average excess risk $ER_{\eta_t}(f)$ across a test distribution $\mathcal{D}(\eta_t)$, but presenting the excess risk specifically for the majority ($\eta_t=0$) and minority ($\eta_t=1$) test groups would provide deeper insight into how spurious correlation affects group performance.

This work touches closely upon the concept of fairness in machine learning, as spurious correlation fundamentally impacts group robustness. The data imbalance investigated here—specifically the role of the minority group fractions $\eta_l$ and $\eta_u$—bears a striking resemblance to the problem of base rate differences discussed in the fairness literature [1]. Given this relationship, I recommend including a discussion on this intersection. Specifically, exploring the connection between the W2S failure mechanism (when $\eta_l \neq \eta_u$) and established fairness criteria could be a valuable direction for future work in W2S.  If you have time, and can derive the difference in errors between the two groups from your current proof to build a similar theory of fairness (maybe in the appendix), this paper would then connect W2S to both robustness and fairness. This would make it much more interesting, and I'd be able to offer you my stronger support. Please don't hesitate to correct me if I've misunderstood anything. 

[1] The Implicit Fairness Criterion of Unconstrained Learning. ICML 2019.

### Questions
- I infer that achieving parity between the minority group proportions in the labeled ($\eta_l$) and unlabeled ($\eta_u$) datasets is crucial for W2SG. Theoretically, does the primary W2S gain come from the minority or the majority group? In the optimal gain scenario ($\eta_u = \eta_l$), which group primarily contributes to the remaining excess risk?
- Strictly speaking, is data imbalance the sole source of spurious correlation in the setting analyzed by this paper?
- How does group robustness affect fairness?

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
4

### Summary
This paper investigates weak-to-strong generalization in the presence of spurious correlations arising from group imbalance. The authors provide a theoretical analysis demonstrating that the disparity in group imbalance between the weak and strong model training datasets is the key factor controlling the weak-to-strong gain. Their analysis shows that weak-to-strong generalization can fail when the group imbalance distributions differ significantly between the two phases. Furthermore, the paper proposes a novel method called Enhanced-W2S, which is empirically shown to achieve superior weak-to-strong generalization on spurious correlation benchmarks.

### Strengths
* This work addresses the role of spurious correlation in weak-to-strong generalization, a topic that has not been sufficiently explored in prior literature.
* The theoretical framework and results are clearly stated, and the paper provides detailed, insightful discussions on these findings.
* Beyond the theoretical analysis, the authors propose a novel method, Enhanced-W2S, as an effective remedy. The effectiveness of this method is well-supported by extensive experiments on various spurious correlation benchmarks and architectures.

### Weaknesses
* The theoretical framework, particularly the formal statement of Assumption 2, can be hard to follow at first glance. I believe that adding a more intuitive, illustrative description can significantly improve reader understanding.
* Theoretical discussions are provided before explaining some technical terms (See Questions).
* The connection between theoretical insights and the intuition behind the proposed method is not clearly discussed (See Questions).

### Questions
* In main results, the terms $\mathcal{V}_T^{(0)}, \mathcal{V}_T^{(1)},  \mathcal{V}_S^{(0)}, \mathcal{V}_S^{(1)}$ appear. What is the precise meaning of these terms? It seems this term might relate to variance (according to line 228), but there are no explicit discussions on it. It would be helpful if the authors could explicitly discuss the meaning of each term right after the statement of the main theorem.
* In the draft, there is a discussion on why the proposed method can be beneficial in the spurious correlation setting. However, the connection between this intuition and the theoretical insights is vague for me. Could you discuss this point in more detail?

### Soundness
3

### Presentation
3

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
The paper considers a weak-to-strong generalization setup with ridgeless regression under spurious correlations with teacher and student models having fixed sets of features that entangle the core and group features differently. They show that in the appropriate limit of sample sizes and core feature dimension, weak-to-strong generalization happens (i) whenever the spurious correlations are the same and there is enough teacher labeled samples, (ii) when there is a mismatch of spurious correlations between the teacher and the student the the W2S gain decreases with this difference, and (iii) algorithmic improvement to standard W2S training pipeline that improves the W2S student performance when there are spurious correlations.

### Strengths
The main significance of the paper is:
1. Consider regression with spurious correlation in the context of W2S is an interesting way to capture different pre-trainings and capacities of student and teacher models
2. Provides crisp characterization of weak-to-strong generalizations in regression with spurious correlations in the appropriate limit of sample sizes and feature dimensions in terms of mismatch in level of spurious correlation, similarity between student and teacher features, and the number of samples
3. Alternative way of performing W2S training when the spurious correlations don’t align with retraining the student model on high confidence points which improves over the vanilla W2S.

The idea of considering spurious correlation is original. The writing is clear and the mathematical arguments seem sound.

### Weaknesses
1. The main theorem, Theorem 3, does not formally establish weak-to-strong generalizations. It only gives a sufficient condition under which weak-to-strong generalization happens in this case in terms of $p_{T S}$. It would be more interesting to find natural choices of $\phi_S$ and $\phi_T$ for which we can understand $p_{T  S}$ and therefore W2S improvement.
2. Here the W2S improvement, like in other ridgeless regression setups, comes “artifically” from the fact that the teacher is trained with too little samples and thus is not optimally. The student is able to  improve over the teacher because it has sufficient samples, and so the effect of student having better features than the teacher is confounded with the effect of the teacher being suboptimal. 
3. The paper asks whether W2S still happens in data with spurious correlation, but the definition of what makes a teacher “weak” and a student “strong” (Assumption 2) is based on the spurious correlations, i.e. that the weak model entangles the core and group features and the strong model doesn’t. The result could be further strengthened with a more agnostic way of defining weak and strong which implies Assumption 2. That way, we could use this setup to get insights on the mechanisms underpinning W2S generalization.
4. It is unclear how broadly applicable algorithmic insights are to other non-vision related W2S setups.

### Questions
1. Is there a choice of parameters in Assumption 2 (i.e. T and S) under which we can have a theoretical characterization of $p_{T\and S}$?
2. Why is the asymptotic limit taking in this exact form? Is it mostly for the simplicity of analysis of the excess risk in ridgeless regression or is there any intuition behind? Could get a similar result for finite sample size and feature dimensions?
3. Is there a natural agnostic definition of weak and strong that implies Assumption 2?
4. Do you think that the enhanced W2S training would improve performance for the original weak-to-strong setup in Burns et al 2023 (and is it even applicable)?

### Soundness
3

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
4

### Summary
This paper extends the results of Dong et al. (2025) to the setting where there are spurious correlations induced by group imbalance in the training data distribution. The paper derives excess risk bounds for (1) a teacher model trained on a small number of labeled examples (Theorem 1) with a representation that doesn't well-separate the spurious and non-spurious features, and (2) a student model trained on an unlabeled data distribution pseudolabeled by the teacher. The bounds indicate that the weak-to-strong gains are largest when the percentage of the "minority group" data is very close in the teacher and student's training data distributions. Based on this observation, it proposes to do two rounds of weakly-supervised training, where in the second round the student retrains on the subset of examples where it was confident in its label.

### Strengths
- Testable theory for the magnitude of weak-to-strong generalization in the presence of group imbalance / spurious correlations
- Theoretical motivation for retraining on subset of data where the student is most confident
- The above approach outperforms "Vanilla" W2S generalization on several datasets and for several teacher/student pairs.

### Weaknesses
- The analysis seems to rely on the fact that the teacher model is an ERM on the labeled data. In this case, why doesn’t the student also use the small labeled set instead of (or in addition to) using the pseudolabeled data? How would the student's performance bound compare in that case? I’m sure the analysis can be fixed to incur some additive error term that measures the teacher’s excess risk over the ERM, but the same point applies

- Not enough discussion of (or motivation for---why are we interested in W2S in the context of spurious correlations?) the incremental novelty over Dong et al. The paper seems like a targeted extension of Dong et al. to the spurious-correlation regime, but that regime is not well-motivated.

- The confidence loss / retraining idea is similar to the confidence loss in Burns et al and very similar to the iterative confident retraining in COSINE [1], and the idea of confident data selection in weak supervision is old (see e.g., [2] and many others). The experiments are missing "3rd-party" baselines like the Burns confidence loss, and the paper is missing contextualization of this idea in the long literature on weak supervision, self-training, and co-training.

Minor Presentation comments:
- Assumption 2 takes a long time to parse. A figure here might be helpful?
- Section 2.3 can probably be cut or moved to the appendix…we might not need experimental validation of the theory in the exact setting of the Theorem. It would be more interesting to argue why some of the assumptions of the theorem are practically relevant, or to bring some of the content from Appendix E into the main text

[1] Yu et al. "Fine-Tuning Pre-trained Language Model with Weak Supervision: A Contrastive-Regularized Self-Training Approach"

[2] Zhang et al. "CoTrade: Confident co-training with data editing"

### Questions
- If we're doing multiple rounds of training, why not co-train the teacher?
- Why not use the labeled data to train the student?
- "For each experiment on a given dataset, we include all teacher—student pairs whose relative strength (measured by accuracy) remains stable when we vary parameters including $\eta_l$, $\eta_u$, $N$, or $n$” ← What does this mean?

### Soundness
3

### Presentation
2

### Contribution
2
