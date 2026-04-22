# Group Contrastive Learning for Weakly Paired Multimodal Data

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
We present GROOVE, a semi-supervised multimodal representation learning approach for high-content perturbation data where samples across modalities are weakly paired through shared perturbation labels but lack direct correspondence. Our primary contribution is GroupCLIP, a novel group-level contrastive loss that bridges the gap between CLIP for paired cross-modal data and SupCon for uni-modal supervised contrastive learning, addressing a fundamental gap in contrastive learning for weakly-paired settings. We integrate GroupCLIP with an on-the-fly backtranslating autoencoder framework to encourage cross-modally entangled representations while maintaining group-level coherence within a shared latent space. Critically, we introduce a comprehensive combinatorial evaluation framework that systematically assesses representation learners across multiple optimal transport aligners, addressing key limitations in existing evaluation strategies. This framework includes novel simulations that systematically vary shared versus modality-specific perturbation effects enabling principled assessment of method robustness. Our combinatorial benchmarking reveals that there is not yet an aligner that uniformly dominates across settings or modality pairs. Across simulations and two real single-cell genetic perturbation datasets, GROOVE performs on par with or outperforms existing approaches for downstream cross-modal matching and imputation tasks. Our ablation studies demonstrate that GroupCLIP is the key component driving performance gains. These results highlight the importance of leveraging group-level constraints for effective multimodal representation learning in scenarios where only weak pairing is available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GROOVE, a semi-supervised representation learning method for weakly paired multimodal data. The approach integrates an on-the-fly backtranslating autoencoder with a group-level contrastive loss, GroupCLIP, which leverages shared labels to enforce cross-modal consistency within a shared latent space. The authors also propose a combinatorial evaluation framework that tests representation learners against various optimal transport (OT) alignment algorithms. The method is evaluated on cross-modal matching and imputation tasks across simulated and two real single-cell perturbation datasets.

### Strengths
* The paper addresses the important and difficult problem of "weakly paired" multimodal data. This is a common scenario in fields like single-cell biology, where destructive measurement techniques prevent collecting coupled omics data.

* The paper provides a robust evaluation framework, when compared to state of the art. By testing representation learners against various alignment algorithms, the authors decouple these two choices and provide a thorough assessment.

### Weaknesses
* While the problem addressed in this work is important, and the results seem solid and insightful to me, the novelty of the method appears  limited. On the one hand, it's simplicity is a strength, in my opinion, but the proposed methodology is not as preponderant as the effort the authors put in the experimental evaluation.

* Although the evaluation section is solid, in my opinion, it lacks several benchmarks established in the literature (see questions below).

### Questions
* The GROOVE method was mainly evaluated on ATAC, protein, and count data. How does it handle other kinds of modalities, such as images? The study by (Xi, 2024) [1] uses different benchmarks, including CITE-seq Data (NeurIPS 2021 challenge) and PerturbSeq/Single Cell Images, which provide more diversity in modality type. Would extra experiments on the datasets studied in [1] provide stronger empirical evidence of the method's generalizability? Is there time for the authors to assess the performance of the proposed method on such additional datasets?

* The ablation study in Table 5 is central to the claim that GroupCLIP is the key performance driver justifying the overall method performance. However, the "No GroupCLIP" model is effectively just the backtranslation framework, which the authors admit may be suboptimal without a pre-trained encoder. Can you elaborate more on how the ablation support the claim of GroupCLIP being so central to the performance of the proposed method?

* In [1], it is argued that any model with a reconstruction loss is forced to learn modality-specific noise, which is "counterproductive to matching". Could the authors discuss this claim in the context of GROOVE's autoencoder and backtranslation components? Furthermore, could this theoretical conflict explain your paper's own observation that "better matching performance does not guarantee optimal downstream task performance"? For instance, is it possible that the reconstruction-based losses (which the work in [1] argues are bad for matching) are actually necessary or beneficial for the downstream imputation task, thereby creating a trade-off between the two objectives?

[1] Xi, Johnny, et al. "Propensity score alignment of unpaired multimodal data." Neurips (2024)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper tackles multimodal settings where examples from two modalities share only a group/label (e.g., perturbation ID) rather than instance-level pairs, and proposes a group-aware contrastive objective (“GroupCLIP”) that pulls together all samples from the same label across modalities while pushing apart different labels. 
- The method combines this group-contrastive objective with a reconstruction pathway and on-the-fly “backtranslation” between modalities, yielding a shared embedding space that supports both matching and imputation. 
- The paper argues that evaluating only instance-level matching is inadequate for weakly paired data, and introduces a combinatorial evaluation protocol that factors out aligner choice using labeled variants of entropic OT, GW-OT, and COOT.

### Strengths
- The paper addresses a realistic regime where modalities cannot be co-measured on the same cell, making group-level supervision both natural and necessary. 
- The paper is well written and easy to follow
- The evaluation design separates representation learning from alignment by sweeping multiple labeled OT variants, which reduces confounding and yields a more credible comparison across methods. 
- The ablations are clear and show that removing the group-contrastive term degrades performance consistently across metrics, which supports the central claim about the importance of group-aware contrast.

### Weaknesses
- CLIP’s web pairs are often weak and effectively many-to-one (e.g., many different dog images paired with near-identical captions), so large-scale CLIP training already approximates a group-level supervision regime rather than strict instance pairing. Thus it is important for the paper to show clear advantages in regimes where per-instance captions carry little unique information beyond a coarse label. The key claimed difference is that GROOVE does not need any per-instance pairing at all, whereas CLIP still relies on a text associated with each image; however, this difference is only compelling if the method outperforms strong CLIP-style baselines constructed to mimic weak pairs.
- The group-level objective risks collapsing within-label diversity because it contracts all samples of a label toward each other across modalities, which can be harmful when a perturbation has heterogeneous cellular responses; this risk is amplified by balanced per-label batching that repeatedly couples the same aggregate label sets. 
- The backtranslation path adds complexity but appears to contribute less than the group-contrastive term in ablations, which raises questions about whether the extra module is necessary relative to a streamlined group-contrastive-only baseline with stronger decoders. 
- The approach assumes label sets are perfectly aligned across modalities during training, but real datasets often contain partial, missing, or mis-specified label mappings; the paper does not evaluate robustness to label noise or missing labels, which is central to the weakly paired setting it targets. 
- The proposed evaluation leans heavily on OT-based aligners, and while the authors sensibly sweep variants, the method’s ranking changes across datasets, which suggests sensitivity to the aligner choice. The paper does not analyze why particular aligners pair best with GROOVE or how to choose them reliably without oracle tuning.

### Questions
see weaknesses above

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
4

### Summary
This paper addresses an important problem in single-cell biology: learning representations from technically unpaired multi-modal data with shared functional labels such as perturbations (making it “weakly” paired). It introduces GroupCLIP, a novel contrastive loss extension for cross-modal representation alignment based on group-level supervision. This loss is combined with backtranslating autoencoders for higher-quality pseudo-pair generation. In an extensive evaluation with various optimal transport methods for matching, the method shows improved performance in cross-modal matching and imputation on small single-cell perturbation sets compared to baselines.

### Strengths
1) The paper tackles the important and practical challenge of learning from weakly paired multimodal data (where only group labels connect modalities), a common scenario in biological perturbation screens

2) The core contribution, the GroupCLIP loss effectively bridges the gap between cross-modal contrastive learning and uni-modal supervised contrastive learning for this specific weakly paired setting

3) The proposed method, GROOVE, outperforms the most comparable methods on real single-cell data for both matching and imputation

4) The paper includes both an ablation study showing that GroupCLIP is the main driver of the performance, as well as a thorough evaluation framework for all-against-all (learner and OT method) comparison.

5) The paper is generally well-written, clearly motivated, and provides substantial methodological detail (extensive appendices on architecture, sampling, baselines, and simulations)

### Weaknesses
1) Many of the performance differences reported in the simulation results (Table 1 Bary. FOSCTTM, Table 2) appear small and likely not statistically significant given the overlapping standard errors. The authors should provide a statistical test to show significant improvement.

2) The paper doesn't provide any analysis on a) sensitivity to hyperparameters alpha and beta that balance the GroupCLIP and reconstruction/backtranslation losses. It's unclear if the chosen values generalize or require dataset-specific tuning. b) the effectiveness of a balanced undersampling strategy is proposed, there's no analysis showing its effectiveness or the limits of imbalance the method can tolerate.

3) The discussion section focuses primarily on future directions for the community rather than critically analyzing the limitations of the proposed method itself (e.g., potential failure modes, hyperparameter sensitivity, unclear value of backtranslation).

4) The real-world datasets used are relatively small subsets derived after significant feature selection and focusing on specific experimental conditions. While understandable for computational reasons (OT scaling) and somewhat sufficient for demonstration on more homogeneous data, it leaves open the question of whether the method scales and performs well on larger and more heterogeneous datasets that are of interest for real-world screens.

5) The abstract's claim of "consistent outperformance in downstream cross-modal matching and imputation tasks" [lines 023f] is slightly overstated, as GROOVE did not significantly outperform the other methods on the simulations. Also, while GroupCLIP is novel, the overall GROOVE architecture relies heavily on adapting an existing backtranslating autoencoder framework from unsupervised machine translation. The ablation results (Table 5) also suggest the backtranslation component itself adds minimal value over a standard autoencoder in this setup, making the main effective innovation primarily the GroupCLIP loss. But since this is in an applications track and the results are outperforming alternative methods, this might be less important but it should be acknowledged in abstract/discussion.

6) While correctly identifying a gap for weakly paired supervised CLIP, the background could acknowledge existing supervised/semi-supervised CLIP extensions that use few perfect pairs (e.g., S-CLIP, SemiCLIP).

### Questions
1) The ablation description for "Autoencoder only" vs. "No GroupCLIP" could be clearer. Does "Autoencoder only" include GroupCLIP?

2) The results show mostly superior performance when using label-constrained OT methods. Could you briefly discuss why leveraging labels during the OT alignment step provides such a significant boost compared to standard OT, even after label-aware representation learning with GroupCLIP?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes GROOVE, a multimodal representation-learning approach for weakly paired data. In particular, GROOVE addresses challenges in single-cell perturbation analyses, where, while valuable, multimodal analysis with measurements from different modalities but the same cells can be infeasible to obtain. The proposed method combines a novel group-level contrastive loss (GroupCLIP), which leverages shared labels (across perturbations) to enforce consistency across modalities. It also leverages an on-the-fly backtranslating autoencoder to enforce well-mixed shared representations between modalities. The authors benchmark GROOVE against two standard baselines and compare representation learners across multiple optimal transport aligners. Experiments across simulated datasets (with varying degrees of modality sharing) and two real single-cell perturbation datasets show that  GROOVE (with an appropriate OT aligner) can lead to performance improvement in downstream cross-modal matching and imputation tasks. The ablation studies show that GroupCLIP is the key component for the performance gains.

### Strengths
- GroupCLIP is an extension that combines SupCon (using supervised class-labels for contrastive learning) and CLIP (for cross-modal alignment). Technically, it is a straightforward extension, but I find this simplicity a strength rather than a weakness.

- The motivation of the work is clear and addresses an important gap; multimodal methods that can leverage weakly paired data are crucial for biological applications where true paired measurements are experimentally infeasible

- A well-motivated experimental design evaluating different OT aligners.

### Weaknesses
- W1: The contributions of the paper are minimal. Besides GroupCLIP, the second contribution is the backtranslating autoencoder. This, however, doesn’t seem to have any positive effect on GROOVE’s performance.
- W2: The experimental analysis is limited to only two baselines. The authors discuss many more methods in the related work, but it’s unclear how these methods differ from GROOVE and why they weren’t chosen for benchmarking (for instance, Samaran et al, 2024)
- W3: Inconclusive findings wrt design choices of similarity metrics and OT aligners

### Questions
- The ablation study shows that removing GroupCLIP causes the largest performance drop, while backtranslation alone performs similarly to a standard autoencoder. Does this suggest that a simpler architecture employing GroupCLIP with a standard autoencoder would be equally effective? Have you evaluated this at different shared portion settings and real data?
- How does GROOVE scale computationally as the number of perturbations increases? The paper mentions ~20 perturbations, but this number could be much larger. Does the undersampling strategy become inefficient when there are many rare labels? Would this also affect the labeled OT aligners?
- How sensitive is GROOVE to the temperature parameter τ and the loss weights λ?

### Soundness
3

### Presentation
2

### Contribution
2
