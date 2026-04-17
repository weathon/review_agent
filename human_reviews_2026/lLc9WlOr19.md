# Conformal Mirror Statistics for Model Alignment: Uncertainty Quantification with FDR Control

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Foundation models are increasingly adopted across diverse domains, but their safe deployment requires outputs that align with human interpretation, especially in high-stakes applications. This motivates the need for rigorous uncertainty quantification (UQ) methods to assess alignment reliability. Most existing methods rely on large labeled datasets, limiting their applicability in real-world settings where labeled data is scarce or expensive. In this paper, we introduce Conformal Mirror Statistics (CMS), a novel framework for UQ in model alignment, selecting aligned outputs for unlabeled data with the false discovery rate (FDR) under control. Unlike conventional conformal methods based on $p$-value calibration, CMS generalizes to broader settings without restrictive calibration size requirements. We further establish theoretical guarantees by proving FDR control under weaker data assumptions than existing methods. Empirical results on simulations and a large sepsis cohort from MIMIC-III demonstrate that CMS consistently outperforms conventional methods while reliably identifying aligned outputs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Conformal Mirror Statistics (CMS), a framework aimed at quantifying uncertainty for model alignment with control of the False Discovery Rate (FDR). The core motivation is to address the limitation of existing conformal methods that rely on large labeled calibration datasets. The proposed method constructs a new statistic leveraging both rank and magnitude information, and establishes asymptotic FDR control under what are claimed to be weaker assumptions than prior mirror statistics approaches. An aggregation procedure via multiple data splitting is also introduced to improve stability. The method is empirically evaluated on a clinical decision-making task using the MIMIC-III sepsis cohort.

### Strengths
1.  **Novel Formulation:** The idea of integrating the concept of mirror statistics into the conformal prediction framework is novel and represents an interesting direction for the field of uncertainty quantification.
2.  **Addressing Data Scarcity:** The paper tackles a practically important problem—performing reliable uncertainty quantification when labeled data is scarce, which is highly relevant for real-world applications in domains like healthcare.
3. The presentation is clear.

### Weaknesses
1.  **Core Motivation:** The paper's primary motivation is critically flawed. It incorrectly claims that conventional conformal p-value methods "cannot reject any hypotheses" when the test set size $m$ is much larger than the calibration set size $n$, stating that $\min_j p_j \geq 1/(n+1) > \alpha/m$ prevents rejection. This is a misunderstanding of the Benjamini-Hochberg (BH) procedure. If a sufficient number $k$ of hypotheses have p-values at the minimum level, say $1/(n+1)$, the BH procedure can indeed reject them provided that $k / m * \alpha \geq 1/(n+1)$ for a sufficiently large $k$. Therefore, the central problem the paper claims to solve is mischaracterized. The actual limitation of the standard method in this regime is potentially low *power*, a nuance the paper completely fails to acknowledge or investigate. This severely undermines the justification for the proposed work.
2.  While the paper achieves *asymptotic* FDR control, it does so at the expense of *finite-sample* FDR control, which is a cornerstone of standard conformal inference methods. The paper does not adequately discuss the implications of this trade-off or justify why sacrificing finite-sample guarantees for asymptotic ones is preferable, especially when the problem of the baseline method (complete failure to reject) is incorrect.
3.  The experimental section is weak and fails to provide compelling evidence for the method's utility.
    *   **Lack of Baseline:** The most critical comparison that demonstrating CMS has higher *power* than the standard conformal p-value method under small calibration sets is completely absent.
    *   **Limited Scope:** The evaluation is confined to a single dataset and a single application (sepsis treatment). The robustness of the method across different data distributions, model architectures, and alignment score functions remains unverified.

### Questions
Please refer to the weaknesses. I will raise the score if my questions can be resolved.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel framework to provide UQ with minimal labeled data. In specific, the conformal mirror statistics (CMS) is introduced which evaluates whether the output of a model is reliable or not under the specified FDR control threshold. 

Since it exploits the properties of alignment scores to build the new mirror statistics, it relaxes previous conditions in Dai et al. (2023b). The method is applied to show a reliable alignment selection on a large-scale sepsis cohort from the MIMIC-III dateset and can be further used to analyze LLM-outputs.

### Strengths
The core idea of the paper is to control FDR on black-box model-generated claims with scarce labeled data, where FDR controls that within in the selected subset $\hat{S}$, the proportion of their true alignment scores being smaller than c should be controlled under $\alpha$.

The traditional way is to use the conformal p-values and applying BH procedure to control FDR, which can be impossible if the test size $m$ is larger than the calibration size $n$. The conformal mirror statistics is proposed, which incorporates both rank information and the magnitude differences between predicted scores. The FDR control is established in theory and robust conformal alignment is proposed to reduce the extra variability by random splitting.

The idea is well-motivation and important in the current study.

### Weaknesses
From my reading, I have the following concerns regarding the paper::

* To construct the Conformal Mirror Statistic, the predicted alignment scores $\hat{A}$ must be estimated on $D^{tr}$, which is a subset of the labeled data $D_l$. Given the stated assumption of limited access to labeled data, the estimated alignment score $\hat{A}$ may not be accurate, which could introduce bias into the downstream analysis. For example, the distribution of $\hat{A}_{n+j} - \hat{A}_i$ might not be symmetric conditional on ${A}_i \leq c$.

* In addition, Theorem 1 and its associated assumptions are established based on known scores $A$, which raises further concerns. Do the theoretical results still hold if the estimate of $A$ is poor, or the correlations introduced by the estimated $\hat{A}$ would violate the i.i.d. assumption? I believe the guarantee of FDR control should also be hinged on the quality of the alignment score estimation.

* Although the authors propose a robust conformal alignment procedure, they do not illustrate the magnitude of the efficiency gain compared to a single random split (i.e., a comparison of the power of multiple mirror FDR versus Mirror FDR).

* Further, I found the definition of power to be unconventional. Power should be defined as the probability of detecting an aligned treatment, conditional on the treatment being correct (which should be factual and do not depend on $c$). Therefore, the denominator in the power calculation should be invariant to the choice of $c$. Otherwise, it can lead to counter-intuitive results, such as those in Figure 3, where increasing the value of $c$ leads to both increasing power and decreasing FDR. This would incorrectly imply that a larger $c$ always yields a better result.

### Questions
see weakness

### Soundness
2

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
The paper proposes Conformal Mirror Statistics (CMS), which is a framework for uncertainty quantification in model alignment that selects the aligned outputs for unlabeled data with false discovery rate (FDR) control. Compared with the conventional approaches based on p-value calibration, the proposed method reduces reliance on large labeled data sets. They also propose an aggregation procedure based on e-values to mitigate the instability from random data splits. The proposed method is evaluated on a MIMIC-III sepsis cohort.

### Strengths
The presentation of the paper is clear. The proposed method is relatively novel and relaxes the sample size requirement in existing methods by incorporating the mirror statistic. Both theoretical and numerical results are provided and convincing.

### Weaknesses
No simulation studies are conducted to compare the performance of the proposed approach and the existing approaches; this is really needed.

There seems to be selection bias in the data application by fine-tuning on survivor trajectories. It is not clear how the selection bias is addressed in the analysis, and under what assumptions for selection the analysis is valid or not.

Some additional intuitive and high-level discussion on the assumptions and theory would be helpful for the reader.

### Questions
1.	What does the symbol $\lesssim$ in equation (3) and equation display below (3) mean?

2.	Intuitively, why is Assumption 1 needed? In practice, how would we know if this assumption is satisfied? Remark 1 mentions that for any distribution of g(X), one can apply the transformation to make the distribution symmetric. This seems to rely on G to be known. Does it mean that we need to know the true distribution G to ensure the proposed method works?

3.	In Theorem 1, the condition contains $P(FDP(\tau)\leq \alpha)\to 1$ as $m\to\infty$, and the conclusion contains $FDP(\tau)\leq \alpha) \to 1$ as $m\to\infty$. Does one directly imply the other? How to interpret this result?

4.	Are there any theoretical results on how large n_cal needs to be for existing methods (e.g., Jin & Candes 2023ab and Gui et al. 2024 referenced in the paper)? If there are, how does the sample size requirement in this paper compare with their requirement? Simulation studies would help empirically.

5.	In Section 4.1, it is mentioned that “q corresponds to the refined predictive distribution produced student network fine-tuned on survivor trajectories.” Does this mean that there is selection bias since the training is only based on survivors? How is this bias handled?

6.	In Section 4.2 figures, the FDR curves show a plateau when \alpha is large. What is the interpretation and rationale? Also, for FDR, is smaller always better, or would we like it to be close to the nominal level (but smaller)? From the plot, the FDR can be much lower than alpha, especially when alpha is large. Any thoughts on this and on how the method may be improved?

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
3

### Summary
This paper proposes Conformal Mirror Statistics (CMS) for uncertainty quantification and alignment of black-box models. In particular, building on the “Conformal Alignment” setting of Gui et al. (2024)--where that conformal alignment paper itself builds closely on “Conformal Selection” papers Jin and Candes (2023a, 2023b)--the current paper proposes CMS as an alternative to the conformal p-values used in Gui et al. (2024) for model alignment. Here “model alignment” refers to generating outputs that have some real-valued “alignment score” that exceeds some target threshold $c$, where the alignment score is a function of the model’s output and some expert ground-truth label; however, at test time, the ground-truth labels are not known, and so the goal of alignment in this case is to select the subset of unlabeled model outputs to label as “aligned,” while controlling the false discovery rate (FDR). In addition to CMS, the authors propose an approach to stable aggregation using e-values and the e-BH procedure. Overall, the authors claimed contributions are the following: (1) that CMS introduce mirror statistics to the conformal prediction literature, and that CMS require less reliance on large labeled data; (2) that CMS require weaker assumptions than in prior mirror statistics work; (3) stable aggregation via multi-data-splitting and e-value-based combination; (4) empirical validation on MIMIC-III sepsis data.

### Strengths
The authors study an important problem of how to select aligned model outputs with FDR control, which is studied by Gui et al. (2024); the current paper’s CMS approach seems original/novel, sound, and well motivated (ie, it seems reasonable that leveraging the magnitude information in calibration-set alignment scores should provide benefits relative to simply using rank information). As far as I’m aware the introduction of mirror statistics into conformal inference settings seems novel. The paper is fairly well-written, and assuming that CMS perform as the authors claim in practical experiments, I think this paper would make a significant contribution of interest to both community (ie, conformal and AI broadly).

### Weaknesses
**Clarity:** Despite the paper being overall fairly clear, some parts of the presentation could still be improved. For example, although I believe I understand the intuition for how CMS leverage *magnitude* information from the alignment scores, which seems like it should provide benefits over conformal p-values (which only use rank information), I think some of the language on how CMS “reduce reliance on large labeled data” should be revised for accuracy. Eg, the authors claim that “[t]his innovation eliminates the reliance on large calibration sets while retaining distribution-free validity,” which I think is too strong of a statement, as I first interpreted this as meaning *no* calibration data are required, but I think that the authors actually mean that only *less* calibration data is required than conformal p-value-based methods. 

**Experimental evaluation:** I have some concerns on the experimental evaluation, as follows:
- *Missing power comparison of CMS vs conformal p-value methods:* The proposed CMS methods are compared against prior conformal p-value methods regarding FDR, but I do not see a comparison of CMS vs these baselines regarding power (and have looked for this in the appendix). I think this should be added to see how the proposed methods compare to baselines in this regard.

- *Comparison of CMS vs conformal p-value methods regarding sample size required:* The first main claimed contribution of the paper is that CMS require less calibration data than conformal p-value baselines. This seems accurate (given that CMS leverage magnitude information whereas p-values only relative information), but because this is a central claim, I think it would be good for the authors to demonstrate this empirically too, at least with a simple example.

- *Unexplained CMS FDR violations in Appendix Figure 6:* Figure 6 of the appendix provides further FDR empirical evaluations--ie, seemingly expanding on Figure 4 from the main for more target values of $c$. However, whereas Figure 4 in the main does not have any clear FDR violations of the proposed CMS methods, the first two frames of Figure 6 do appear to have CMS FDR violations, with the proposed methods’ FDR exceeding the diagonal (where empirical and target FDR are equal). These violations do not appear to be mentioned or explained. 

The paper could be improved by addressing the above, improving the clarity of presentation, and if the authors could present additional experimental results on other dataset(s) beyond the one MIMIC dataset currently evaluated on.

### Questions
Why is the conformal p-value defined on only the unaligned calibration set?

Please respond to my comments/ questions in “Weaknesses.” Eg, can the authors provide comparison of CMS and conformal p-value methods regarding power and sample size required? Also, can the authors explain or provide further analysis of the apparent FDR violations of the proposed methods in Figure 6?

Overall I like the ideas in the paper, but I think that some of the presentation and empirical evaluation (as described in Weaknesses) should be improved. If these concerns are addressed I would consider improving my recommendation.

### Soundness
2

### Presentation
2

### Contribution
3
