# SOFTADACLIP: A SMOOTH CLIPPING STRATEGY FOR FAIR AND PRIVATE MODEL TRAINING

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 2, 8, 4

## Abstract
Differential privacy (DP) provides strong protection for sensitive data, but often reduces model performance and fairness, especially for underrepresented groups. One major reason is gradient clipping in DP-SGD, which can disproportionately suppress learning signals for minority subpopulations. Although adaptive clipping can enhance utility, it still relies on uniform hard clipping, which may restrict fairness. To address this, we introduce SoftAdaClip, a differentially private training method that replaces hard clipping with a smooth, tanh-based transformation to preserve relative gradient magnitudes while bounding sensitivity. We evaluate SoftAdaClip on various datasets, including MIMIC-III (clinical text), GOSSIS-eICU (structured healthcare), and Adult Income (tabular data). Our results show that SoftAdaClip reduces subgroup disparities by up to 87% compared to DP-SGD and up to 48% compared to Adaptive-DPSGD, and these reductions in subgroup disparities are statistically significant. These findings underscore the importance of integrating smooth transformations with adaptive mechanisms to achieve fair and private model training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SoftAdaClip, a differentially private (DP) training strategy that replaces the conventional hard clipping in DP-SGD with a smooth tanh-based transformation, integrated into an adaptive clipping framework. The authors argue that hard clipping disproportionately suppresses gradients from underrepresented subgroups, contributing to fairness degradation during DP training. SoftAdaClip aims to preserve relative gradient magnitudes while maintaining sensitivity bounds necessary for DP. Experiments on three datasets (MIMIC-III, GOSSIS-eICU, Adult Income) show reduced subgroup disparities and often improved utility over DP-SGD and Adaptive-DP-SGD. The paper also analyzes clipping behaviors and conducts ablations to separate smoothing from adaptivity.

### Strengths
The paper tackles a meaningful and timely challenge at the intersection of privacy and fairness, where trade-offs are often assumed unavoidable. The proposed method is conceptually simple yet intuitively motivated and appears compatible with standard training infrastructures. The empirical results show notable improvements in subgroup loss disparities across multiple real-world datasets, particularly in healthcare domains where fairness issues can have severe consequences. The statistical significance analysis is appreciated, and the ablation study helps clarify the distinct role of adaptivity. The work would likely be of interest to both privacy researchers and practitioners deploying DP in sensitive domains.

### Weaknesses
Although promising, the novelty is incremental: the method mainly replaces a min() rescaling with a tanh-based one inside an existing adaptive clipping algorithm. The theoretical foundations stop at sensitivity bounding; there is no deeper optimization or fairness analysis (e.g., convergence, bias dynamics, subgroup gradient geometry, compatibility with Rényi-DP accounting). The fairness evaluation relies almost exclusively on loss gaps; no standard fairness metrics like Equal Opportunity or demographic parity gaps appear. The method struggles on low-gradient regimes, requiring manual threshold tuning, which undermines the claim of being a drop-in robust improvement. Presentation needs polishing: key equations and experimental setups feel buried, and figures require clearer labeling and narrative connection. References to prior fairness-aware DP methods are limited in experimental comparison.

### Questions
Are there guarantees that replacing min() with tanh() preserves unbiasedness or improves optimization dynamics? Any theoretical characterization of how gradient direction distortion differs from hard clipping?

In Algorithm 2, how does adaptive update interact with tanh scaling? Could certain C trajectories amplify subgroup divergence?
Privacy accounting. Since tanh depends on gradient norm, is there any subtle impact on the DP accountant or amplification? Please clarify formally.

Why not include widely-used fairness metrics (AUC/F1 gaps per subgroup, equalized odds violations, accuracy parity)? Loss gaps alone may not reflect actual decision-level harms.

Why is DPSGD-Global-Adapt (Esipova et al., 2022) excluded from experiments? If code is unavailable, can closer replication or alternative strong baselines (e.g., subgroup-adaptive clipping techniques) be tested?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors highlight the problem of unfairness when performing private training using methods like DP-SGD, which employ methods like gradient clipping to bound sensitivity for adding differentially private noise. They propose replacing the hard clipping operation with a smooth tanh-based gradient operation, and demonstrate improvements in group fairness after performing differentially private training using their method.

### Strengths
**[S1]** Very well-motivated question, highlighting how clipping in DP-SGD may lead to unfairness (as shown in previous works mentioned in the paper, viz. Esipova et al, Tran et al, Bagdasaryan et al)

**[S2]** Using a tanh-based smooth transformation instead of hard clipping is less aggressive and lossy, still retaining gradient information about groups with large gradients.

**[S3]** The design and description of the proposed method are very clearly and unambiguously done, including justifying the design of the gradient transformation and privacy-preserving clipping threshold updates.

**[S4]** In-depth and frank discussion about the limitations of the work is included, including key points about sensitivity to hyperparameter tuning when small clipping thresholds are adopted.

**[S5]** The use of a Wilcoxon signed-rank test with Bonferroni correction to illustrate the statistical significance of the improvements in loss gap is thoroughly appreciated and reflects good practices in presenting empirical results!

**[S6]** Ablation study on adaptive thresholding is useful and clearly demonstrates the importance of pairing smooth transformation with adaptive thresholding.

### Weaknesses
**[W1]** The authors provide a justification for using different hyperparameters for non-private and private training, which is convincing. However, the only concern I have is that this makes it difficult to fairly assess the true utility loss due to private training.

**[W2]** **Needs significant editing.** The presentation of the work needs significant improvements and editing. For example, there are duplicate paragraphs in page 6 in Section 4.1, where paragraph 1 and the first part of paragraph 2 state the same thing in slightly different language, suggesting redundant text was left over while drafting. There is also a missing appendix reference in the last line right before Section 5.1 ("dataset-specific results are provided in Appendix ??").

**[W3]** **Important missing baselines.** This work only looks at limited baselines, primarily at non-private clipping-based baselines. However, I feel like this work’s contribution cannot be truly assessed without meaningful comparison against important fair DP-SGD baselines like [1] and [2], which employ methods like Langrangian dual based fairness-constrained training, or Esipova et al (the comparison against Esipova et al is very limited, and for a venue like ICLR, it is not appropriate to simply defer a comparison to future work; *you must compare against all important and relevant baselines in your own work*) At the end of the day, while their work proposes improvements to the clipping paradigm used in DP-SGD, it is important to see if it actually contributes an overall fairer way of doing DP-SGD, or if it does not lead to any improvements in fairness when added to/compared against these fair DP-SGD baselines, rendering it functionally redundant. Put another way, while the proposed method might improve upon DP-SGD and Andrew et al., it is unclear if it will actually lead to meaningful improvements (if any at all) in fairness when compared against or integrated into existing sophisticated SoTA fair DP-SGD baselines like [1] or [2]. The absence of results against such baselines presents a weaker assessment. Put another way, this paper shows that the proposed method is fairer than non-fairness-aware baselines, but does it really outperform prominent and existing fair DP-SGD methods, or is it inferior to them/does not provide meaningful improvement in conjunction with them? This is an important question to answer for a venue of this stature.

**[W4]** The experimental settings section is presented poorly. The models used for eICU and Income datasets are not described and are vaguely called the simple and complex models, with actual model descriptions deferred to the appendix, while **the experimental setting section instead spends most of its real estate talking about what related work does** (which is better discussed in related work or in the appendix as additional details, while priotizing mentioning the settings used in **this paper**). Therefore, the experimental setting section does not discuss what it is intended to do: properly and exhaustively describe the experimental settings used in **this** paper. I heavily implore the authors to improve the presentation of this section and include concrete details about **their** settings here (which should take precedence here!) instead of sending them to the appendix (in fact, you can actually send the related work discussion in this section to the appendix, but what has been sent to the appendix should actually be in the main text in this section!), especially for a submission to a venue like this

---

## References

[1] Tran, Khang et al. “FairDP: Achieving Fairness Certification with Differential Privacy.” 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) (2023): 956-976.

[2] Tran, Cuong et al. “Differentially Private and Fair Deep Learning: A Lagrangian Dual Approach.” ArXiv abs/2009.12562 (2020): n. pag.

### Questions
[Q1] Can you please address W1 and make sure that the best possible set of hyperparameters is used for each setting, perhaps via a hyperparameter search, to obtain the best possible non-private utility to compare against?

[Q2] Can you, to the best of your ability, add more comparisons against SoTA baselines for better showcasing the efficacy of your method (as mentioned in W3; please feel free to include any more baselines than those included as well)? I believe this will make your paper significantly more convincing.

[Q3] Pursuant to W4, can you please provide a much better drafted experimental settings section that focuses primarily on what **you** do in this paper, while making sure to exhaustively and unambiguously discuss all the settings/models, etc., used?

In short, I believe this paper has the potential to make a good contribution. However, in its current state, with its experiments, lack of comparison against important SoTA methods (or any in-depth comparison against even just any fair DP-SGD method at all), ambiguity in choice of hyperparameters for private/non-private settings, presentation, etc., it does not inspire strong support from my end. However, I’ll be more than happy to engage with the authors, and if there’s enough improvement, I would be happy to strengthen my support for the paper (contingent upon my concerns being satisfactorily addressed).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces SoftAdaClip, an alternative to (Adaptive) Differentially Private (DP) SGD that replaces hard gradient clipping with a smooth, tanh-based transformation to better preserve relative gradient magnitudes.
Similar to prior work (Adaptive-DPSGD), the method adaptively adjusts the clipping threshold during training.
SoftAdaClip demonstrates improved fairness by reducing subgroup disparities (measured in loss differences) across text and tabular datasets compared to both DP-SGD and Adaptive-DPSGD.

### Strengths
* SoftAdaClip introduces a simple yet elegant/effective modification to Adaptive-DPSGD.
* The paper is very verbose, well-motivated, and easy to follow.
* The evaluation is performed over different data modalities (text and tabular data).
* The results are promising/consistent, showing clear improvement over DP-SGD and Adaptive-DPSGD in terms of the measured fairness metric (loss disparities).

### Weaknesses
## Weaknesses:
* While SoftAdaClip is simple/elegant (which is great), the novelty feels somewhat incremental (the core idea boils down to 1 line of code) may not qualify as a substantial contribution for a top-tier conference.
Overall, the work is promising, but it currently reads more like a strong workshop/early-stage research paper than a full conference paper.
I encourage the authors to continue working on the paper.

* The results reported in the abstract are slightly misleading -- it would be more appropriate to report average improvements over DP-SGD/Adaptive-DPSGD, rather than cherrypicking the best differences.
Additionally, from Table 1, while SoftAdaClip achieves lower loss than Adaptive-DPSGD, but this does not seem to lead to better accuracy/f1.
This is only briefly mentioned on the last page.
Given that accuracy/f1 are more practically important than loss, this deserves more discussion.
Finally, accuracy per subgroup (as in (Bagdasaryan et al., 2019)) is not provided, which will be valuable for understanding fairness performance across subgroups.

* The paper primarily focuses on two DP-SGD alternatives -- Adaptive-DPSGD and DPSGD-Global-Adapt. However, several other important works should be considered or at least be acknowledged (this is not an exhaustive list):
	* tempered sigmoid activations in DP-SGD [1]
	* private and fair classification with pre-traiing [2]
	* FairDP-SGD and FairPATE [3]
	* DP-SGD without clipping [4, 5]

* In Table 4, the presentation of gradient norms seems confusing: it reports Before -> Diff (After) rather than Before -> After (Diff). Moreover, the reported clipping for SoftAdaClip appears larger than for the other methods, which seems inconsistent with the claim in Section 4.1 and Appendix B.3 that SoftAdaClip applies less clipping.
This is somewhat intuitive, since SoftAdaClip could clip some gradients more than hard clipping (as shown in Figure 1).
It would be helpful if the authors clarify this discrepancy. 

* While the paper is easy to follow, the writing is extremely verbose and contains repetitions across sections, which can make it harder to navigate.
	* Section 2 (Related Work) contains repeated information between the first subsection and Section 2.1.
	* Section 2.1 mixes background/preliminaries with related work. It would be clearer to separate these two aspects into distinct sections.
	* Section 4.1 (Gradient Behavior Analysis) discusses specific results (e.g., Table 4) that seem more appropriate for Section 5 (Results).
	* Section 5 (Results) begins by continuing discussion of the experimental setup, which should be fully contained in Section 4.
	* Sections 6 (Limitations) and 7 (Conclusion) take a full page, which could be better utilized by moving some experiments/tables from the Appendix.

## Minor Weaknesses/Comments:
* The variable \epsilon in Algorithm 2 may be confusing, as \epsilon is already used in the DP definition; consider using a different symbol.
* References are inconsistently cited (e.g., \citet vs. \cite/\citep).
* Small typos/punctuation errors -- "It would nevertheless be valuable to It would still be beneficial to," etc.


## References:

[1] Papernot et al., Tempered Sigmoid Activations for Deep Learning with Differential Privacy. In AAAI, 2021

[2] Berrada et al., Unlocking Accuracy and Fairness in Differentially Private Image Classification. 2023

[3] Yaghin et al., Learning with Impartiality to Walk on the Pareto Frontier of Fairness, Privacy, and Utility. In RegML at NeurIPS, 2023

[4] Bethune et al., DP-SGD Without Clipping: The Lipschitz Neural Network Way. In ICLR, 2024

[5] Zhang et al., Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach. In ICLR, 2024

### Questions
## Questions/Suggestions for Improvements:
* Appendix A states that demographic subgroups are balanced, but it is unclear whether the target labels are imblanced/balanced. Can the authors clarify the label distributions?
* How do different imbalance ratios and different values of \epsilon affect SoftAdaClip’s performance?
* Have the authors evaluated performance on smaller subgroups (e.g., >20 groups) as in (Bagdasaryan et al. 2019)?
* Why was the tanh function chosen for smooth clipping? Would other functions (e.g., sigmoid) work as well?
* It is unclear whether the methods uses DP-Adam or DP-SGD as base for SoftAdaClip. From the code, it seems DP-Adam is used -- if so, is it defined in the paper (I only can see DP-SGD)? Are there any differences in performance between DP-Adam and DP-SGD?
* What is the difference between the clipping parameters C and C_0?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
SoftAdaClip proposes a novel DP training method integrating a smooth tanh-based transformation into adaptive clipping. It is proposed to mitigate a disproportionate effect of DP on minority groups in terms of performance.

### Strengths
- tested on 3 real-world datasets: MIMIC-III (clinical text dataset), GOSSIS-1-eICU (structured healthcare dataset), Adult Income (tabular dataset).
- tackles important problem of fairness under DP training

### Weaknesses
- the paper should include experiments demonstrating the efficacy of SoftAdaClip in mitigating fairness disparities in a standard vision task to support general application of the method.

### Questions
How was the smooth function chosen? Why not using a different sigmoidal or smooth bounded function instead of tanh?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper provides a novel approach to building fair and private model training through adaptive clipping  using tanh transformation that preserves magnitudes of gradients. The paper addresses a clear problem of privacy-fairness tradeoff with a unique approach.

### Strengths
- Mechanism design is clean and nicely presented, can be integrated well into the existing libraries and pipelines
- tanh idea is also quite strong, from the literature I know DPSGD always focused on balancing noise multiplier with clipping bound without focusing on how the clipping is performed.
- Empirically there is some evidence of improved fairness with same guarantees

### Weaknesses
Overall, while the paper looks great it lacks experimental evidence of the usefulness of the method, here are a couple of questions:
- I believe the paper should include at least a basic example of applying proposed method to vision problems, even MNIST or CIFAR is enough. It is stated as out-of-scope in limitations, but it will still be very useful to have.
- Additionally unbalancing these vision datasets and demonstrating how far the method can apply.
- I would like to see how different epsilon changes the effect of the method. 
- I am also curious about selection of the noise multiplier (noise/clipping threshold) and how this selection affects the method. 
- Can we have different fairness metrics besides difference in loss? maybe accuracy on test sets? 
- Comparison with related work. The paper only compares with DPSGD and adaptive clipping (Andrew et al) but even adaptive clipping has different settings. It will be helpful to also compare with Bu et al 2024 (automatic clipping).

### Questions
Addressing weakness above will significantly help the paper.

### Soundness
3

### Presentation
3

### Contribution
3
