# Information-Theoretic Criteria for Knowledge Distillation in Multimodal Learning

- Decision: Reject
- Scores: 4, 2, 4, 6, 2, 6

## Abstract
The rapid increase in multimodal data availability has sparked significant interest in cross-modal knowledge distillation (KD) techniques, where richer "teacher" modalities transfer information to weaker "student" modalities during model training to improve performance. However, despite successes across various applications, cross-modal KD does not always result in improved outcomes, primarily due to a limited theoretical understanding that could inform practice. To address this gap, we introduce the Cross-modal Complementarity Hypothesis (CCH): we propose that cross-modal KD is effective when the mutual information between teacher and student representations exceeds the mutual information between the student representation and the labels. We theoretically validate the CCH in a joint Gaussian model and further confirm it empirically across diverse multimodal datasets, including image, text, video, audio, and cancer-related omics data. Our study establishes a novel theoretical framework for understanding cross-modal KD and offers practical guidelines based on the CCH criterion to select optimal teacher modalities for improving the performance of weaker modalities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces the Cross-modal Complementarity Hypothesis (CCH), a theoretical framework for understanding when cross-modal knowledge distillation (KD) improves performance in multimodal learning. The hypothesis posits that KD is effective when the mutual information between teacher and student representations exceeds that between student representations and the labels. The authors validate CCH both theoretically—via a joint Gaussian model—and empirically across diverse datasets, including synthetic data, MNIST/MNIST-M, CMU-MOSEI, and cancer-related omics datasets. They demonstrate that CCH reliably predicts when KD will enhance student performance and offer practical guidance for selecting teacher modalities.

### Strengths
- The paper is generally well-written and well-motivated.
- The organisation of the paper is clear.

### Weaknesses
- MFH already posits that the success of cross-modal KD depends on the degree of label-relevant information present in the teacher modality and its alignment with the student. CCH formalizes this intuition using mutual information inequalities, but the underlying insight appears to be a refinement rather than a fundamentally new idea.

- All experiments use identical architectures for teacher and student models to isolate mutual information effects. This setup may not reflect realistic KD scenarios where teacher and student can differ in capacity and architecture.

- Its a bit confusing to use MNIST and MNIST-M as the teacher and student modality, as they are basically the same modality but from different domain. I think the author should use AV-MNIST instead.

- its a bit confusing to see $I(H_{text}; H_{audio})$. Please specifcy how is this computed.

### Questions
n/a

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces the Cross-modal Complementarity Hypothesis (CCH), proposing that cross-modal knowledge distillation (KD) is effective when the mutual information between teacher and student representations exceeds that between the student and the labels (I(H₁;H₂) > I(H₂;Y)). The authors support this claim through Gaussian-model analysis and extensive empirical validation across synthetic, vision, audio-visual, and biomedical datasets. The work aims to provide a principled, information-theoretic understanding of when cross-modal distillation helps.
However, despite a systematic set of experiments, the theoretical contribution remains shallow and largely intuitive. The proposed criterion essentially reformulates a well-known intuition that KD is useful only when the teacher carries additional label-relevant information that the student lacks, without introducing new theoretical insights or addressing the practical challenges of mutual information estimation.

### Strengths
1. The paper is clearly written and well organized, with a coherent conceptual narrative and a solid empirical structure.
2. It consolidates a range of existing intuitions about when knowledge distillation is effective into a single, measurable information-theoretic framework.
3. The experimental evaluation is diverse and carefully executed, and the accompanying code and datasets are well documented, enhancing empirical transparency and reproducibility.

### Weaknesses
1. The central criterion (I(H₁;H₂) > I(H₂;Y)) essentially restates an intuitive idea that distillation is beneficial only when the teacher provides additional label-relevant information that the student lacks. This condition is conceptually simple and offers limited theoretical novelty beyond rephrasing a well-known intuition in information-theoretic terms.
2. The Gaussian analysis functions more as a didactic toy example than as a rigorous theoretical derivation. It does not uncover any new properties of mutual information, nor does it deepen our understanding of how knowledge distillation operates in nonlinear or high-dimensional settings.
3. The hypothesis describes a correlative rather than a causal relationship, offering no explanation of why or how mutual information alignment leads to improved performance. It provides little insight into the internal mechanisms of representation transfer.
4. The empirical evaluation primarily confirms the hypothesis rather than attempting to challenge or falsify it. The observed correlation between MI gap and performance improvement is largely expected and does not establish theoretical necessity.
5. The proposed condition is difficult to estimate reliably in high-dimensional multimodal models, which limits its practicality as a predictive or diagnostic tool for real-world knowledge distillation systems.

### Questions
1. Could the authors clarify how the proposed CCH framework fundamentally differs from prior approaches that explicitly maximize teacher–student mutual information, such as Variational Information Distillation (VID) or Contrastive KD?
2. Are there scenarios where the condition I(H₁;H₂) > I(H₂;Y) holds but distillation does not yield improvement? Providing such counterexamples could help delineate the boundaries and limitations of the hypothesis.
3. Can the authors establish a clearer theoretical or intuitive link between mutual information and the optimization dynamics of distillation, for example in terms of gradient alignment or feature transfer behavior?
4. How sensitive are the reported empirical results to the specific mutual information estimator used (e.g., MINE, latentMI, KSG) and to variations in data dimensionality?
5. What would be the practical implications or implementations of this criterion in modern large-scale multimodal transformer architectures, where direct MI estimation is computationally infeasible?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the \emph{Cross-modal Complementarity Hypothesis} (CCH), an information-theoretic rule-of-thumb for predicting when cross-modal knowledge distillation will help a student model. The key idea is that distillation is beneficial when the mutual information between teacher and student representations exceeds that between the student representation and the label, offering a purportedly a priori criterion for choosing teacher modalities. The authors provide a sufficiency argument under a linear-Gaussian, small-regularization regime and present empirical evidence across synthetic data and several multimodal domains (e.g., digit variants, sentiment/audio-visual-text, and omics), plus a comparison suggesting the criterion can inform when to add KD on top of fusion.

While conceptually tidy, the theoretical support is narrow: guarantees rely on idealized linear assumptions and asymptotics, leaving unclear applicability to nonlinear deep networks and finite-sample training typical in CVPR settings. The “a priori” test hinges on estimating mutual information between learned representations, which itself requires trained models and high-dimensional MI estimators with nontrivial bias/variance---the paper does not convincingly quantify estimation reliability or decision robustness. Empirically, most benchmarks are relatively forgiving; there is little evidence on harder, vision-centric tasks (e.g., dense prediction, detection, long-tail distributions, or noisy/weak teachers) where the hypothesis would be stress-tested. Practical guidance is also limited: the work says \emph{when} to distill but gives scant insight into \emph{how} (temperature/$\lambda$, layer choice, calibration, or resilience to teacher errors). Finally, the connection between KD gains and shared label-relevant information is not entirely new; without stronger theory beyond the linear case or sharper ablations ruling out confounds (capacity, regularization side effects), the contribution feels incremental. Overall, a clean unifying lens with promising intuition, but presently short of a reliable decision rule for modern multimodal vision systems.

### Strengths
It positions the hypothesis against related work and argues that, despite prior intuitions, no earlier paper stated a concrete MI-based feasibility condition for cross-modal KD.

### Weaknesses
The only formal support is in an idealized linear regression setting under a jointly Gaussian data model with a quadratic KD penalty—far from modern non-linear deep nets used in vision.

The paper must average over 50 runs and shows that absolute MI values vary by estimator, indicating sensitivity that is not quantified with uncertainty or calibration analyses.

The condition relies on mutual information between learned representations H1, H2, which themselves require training to obtain. This blunts the promise of a pre-training decision rule.

The MI toolbox (KSG, MINE, latent-MI) is itself known to be challenging in high-D; the paper’s overview underscores that exact MI is intractable beyond small, known-distribution cases, raising questions about robustness as a gating signal.

### Questions
How sensitive is the CCH decision to finite-sample noise? Can you bound the probability of a wrong decision (sign flip) as a function of sample size and MI-estimator error?

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
This work proposes a mutual information–based quantifiable criterion to determine whether cross-modal knowledge distillation can effectively improve the performance of the student model. It reveals that when the mutual information between the teacher and student modalities exceeds that between the student modality and the labels, cross-modal knowledge distillation is beneficial to the student’s performance. Under the joint Gaussian assumption, the paper theoretically proves the validity conditions of the CCH criterion and derives how knowledge distillation contributes to reducing the student model’s risk. Extensive experiments across multiple benchmarks further validate the effectiveness of the proposed strategy.

### Strengths
1. Analyzing cross-modal knowledge distillation from the perspective of mutual information is an interesting and insightful attempt.

2. The overall writing is clear and easy to follow.

3. The effectiveness of the CCH criterion is validated across diverse datasets, including synthetic data, image datasets (MNIST/MNIST-M), multimodal sentiment analysis (CMU-MOSEI), and cancer multi-omics data (TCGA).

### Weaknesses
1. The CCH criterion relies on estimates of mutual information (MI). However, accurately estimating MI in high-dimensional settings is a well-known challenge and can be computationally expensive. In practical scenarios, particularly during the model design phase, performing an additional and potentially unstable MI estimation to “a priori” decide whether KD will be beneficial may be impractical. This limitation weakens the usefulness of CCH as a true a priori criterion. The paper employs multiple MI estimators (such as latentMI, MINE, and KSG), but the differing results suggest limited robustness.

2. Theorem 1 and its proof are based on very strong assumptions, namely that the data follow a joint Gaussian distribution and both teacher and student models are linear. Although these assumptions help simplify the theoretical analysis, modern deep learning involves highly nonlinear models and complex, non-Gaussian data distributions. It remains unclear to what extent the theoretical guarantees can generalize to real-world cross-modal KD scenarios that rely on deep neural networks.

3. In the classical knowledge distillation setting, the teacher model is typically larger and stronger, while the student is smaller and more efficient. Even if the CCH condition is satisfied, a student with limited capacity may not be able to absorb the complementary information provided by the teacher. It would be helpful for the paper to discuss or empirically analyze how the capacity gap affects the practical value of the CCH criterion.

4. The current work focuses mainly on distillation between two modalities. It would be interesting to explore whether the CCH criterion can be extended to multi-modal interactions involving three or more modalities, where pairwise mutual information relationships and higher-order dependencies become more complex. A discussion or preliminary investigation in this direction could further strengthen the contribution.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Cross-modal Complementarity Hypothesis (CCH), a simple criterion based on mutual information that enables users to a priori determine whether cross-modal knowledge distillation (KD) between teacher and student modalities is likely to be successful.
The work provides a theoretically motivated and practically meaningful attempt to define explicit conditions under which cross-modal KD is feasible.

### Strengths
- The paper presents an interesting and intuitive hypothesis based on comparisons of mutual information among the teacher, student, and label distributions.

 - The idea of formalizing a predictive criterion for when cross-modal KD will succeed is appealing and relevant to multimodal learning research.

- The experimental section is comprehensive and tried to empirically demonstrates the proposed hypothesis across multiple modalities.

### Weaknesses
- The theoretical analysis relies on simplified assumptions, including data modeled as a mixture of Gaussian distributions and the use of a linear regressor without non-linearity. These constraints substantially limit the generality of the theoretical findings. Since the theoretical proofs are derived under such restricted settings, the overall contributions are limited from a theoretical standpoint.
Moreover, the proposed criterion appears naïve and idealized, as it depends on assumptions such as access to optimal weights, which are impractical in real-world scenarios.

- Also, in the synthetic experiments, it remains unclear how the parameterization of the Gaussian mixture can be related to real-world datasets. Providing either a theoretical bridge or empirical demonstration of this connection would strengthen the paper.
Applying varying Gaussian blur to inputs intuitively disrupts cross-modal transfer, but it is not obvious how such manipulation validates Theorem 1. The link between this experimental setup and the theoretical claim should be clarified.

- Although the paper presents extensive empirical validation, the strength of evidence may not meet ICLR standards, given the simplicity of the theoretical setup and the lack of strong empirical grounding in realistic conditions.

### Questions
As noted above, there appears to be a disconnect between Theorem 1 and the experimental results. This point should be clarified, and I encourage the authors to provide further explanation or evidence to bridge this gap.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes the Cross-modal Complementarity Hypothesis (CCH), an information-theoretic criterion for when cross-modal knowledge distillation (KD) helps. In a joint Gaussian setting, the authors show that KD improves a student when the mutual information between teacher and student representations exceeds that between the student and the label. Experiments across synthetic, vision, language, audio/video, and omics datasets, using multiple MI estimators, consistently align with this prediction, with gains when the MI gap is positive and degradations otherwise. The study provides a clear rationale for cross-modal KD and practical guidance on selecting teacher modalities and tuning KD strength.

### Strengths
1. The CCH gives a clear MI gap condition that is intuitive and can be checked with learned representations, turning “should we distill across modalities?” into a verifiable precondition and a practical handle for choosing the teacher modality and KD strength.

2. The paper provides a relatively rich theoretical analysis, which helps to make the imposed conditions more interpretable.

3. The experimental results on both synthetic and real-world tasks further validate the effectiveness of the proposed method.

### Weaknesses
1. The theoretical results are established under the assumption of joint Gaussian distributions. However, it remains unclear whether the sufficiency conditions still hold in non-Gaussian, nonlinear, and multi-class settings, as well as in the presence of practical asymmetries (e.g., heterogeneous architectures, representation mismatch, or domain-adaptive preprocessing). Further validation in these more realistic scenarios would strengthen the contribution.

2. When the student model has insufficient capacity, when excessive regularization is applied, or when the teacher model is overfitted, the behavior of the mutual-information-gap criterion appears to be insufficiently clarified.

3. When there are substantial differences between the teacher and student in terms of architecture, capacity, representation level, or domain preprocessing, it remains unclear whether the proposed method can still maintain reliable predictive performance.

### Questions
Does the proposed method remain effective when there are significant differences between the teacher and student architectures?

### Soundness
3

### Presentation
3

### Contribution
3
