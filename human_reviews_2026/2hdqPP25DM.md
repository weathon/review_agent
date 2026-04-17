# Arcueid: Multi-trigger Cloud Shaping for Unified Backdoor Attack Paradigms

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Machine learning have driven breakthroughs in recognition, detection, and generation, yet their increasing ubiquity also exposes them to backdoor attack hazards, threatening the security of real-world AI deployments. Existing backdoor methods, however, remain fragile in adaptive settings for **rigid dependency on a static trigger**, **narrow scope in fixed one-to-one mappings**, or **unrealistic assumptions for levels of access**, thereby failing to scale to dynamic, large-class scenarios under realistic constraints. Therefore, we present ***Arcueid***, a theoretically grounded multi-trigger backdoor framework that **achieves scalable and robust attacks across $M \mapsto M$, $M \mapsto N$, and $M \mapsto 1$ paradigms**. It operates under restrictive settings, **requiring only black-box knowledge and extremely low poisoning budgets**. At its core lies a *Joint Cloud Shaping Multi-trigger Optimization* strategy that simultaneously compacts trigger-induced feature clouds and enforces inter-cloud separation, ensuring stable, non-interfering, and target-consistent decision regions, while decoupling trigger generation from label mapping to enable dynamic reconfiguration of targets and robust transferability across models and datasets. Extensive experiments on multiple datasets and five CNN/transformer architectures show that ***Arcueid*** attains near-perfect average ASR (**>97\%**) across targets in each paradigm with negligible clean accuracy drop (**<5%**) even at poisoning rates of **0.1%**, significantly outperforming SOTA baselines. Moreover, ***Arcueid*** consistently withstands representative pre-/mid-/post-training defenses, exhibits strong stealth with indistinguishable perceptual shifts, and sustains steady resilience across comprehensive ablation studies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Arcueid is a black-box, multi-trigger backdoor framework that optimizes compact, well-separated “trigger clouds” in feature space (via Joint Cloud Shaping) so that multiple triggers can reliably map to arbitrary target labels across models and datasets with extremely low poisoning rates while remaining stealthy and robust to many defenses.

### Strengths
Arcueid introduces a unified and highly transferable multi-trigger backdoor framework that leverages Joint Cloud Shaping to optimize compact and well-separated trigger clusters in feature space, achieving strong attack success rates even under black-box and low-poisoning conditions. It generalizes across models and datasets, maintains minimal impact on clean accuracy, and remains robust against multiple state-of-the-art defenses, demonstrating both theoretical soundness and practical stealth.

### Weaknesses
1. The transferability theory essentially depends on the assumption that the surrogate and target models share a well-aligned representation space — specifically, that there exists a bounded linear mapping \(A\) and a small representation discrepancy \(\delta\) such that the triggered feature clouds in the target model preserve the same margin as those in the surrogate (see Proposition 2). However, this assumption is difficult to verify or guarantee in practice. The surrogate dataset in the paper is explicitly non-IID and small in scale (5k–15k samples), and the surrogate and victim architectures may differ substantially, leading to estimation errors in class centers and large shifts in feature geometry. As a result, the theoretical condition \(L_h\|A\|\delta < \gamma\) may not hold in real scenarios, causing the actual attack success rate (ASR) on the target model to degrade or become unstable.

2. The paper demonstrates Arcueid’s strong resistance to post-training defenses (Figure 7), but the results also reveal that NAD and FTSAM mitigate different aspects of the backdoor. NAD offers partial but unstable reduction of ASR through knowledge distillation, whereas FTSAM notably decreases ASR in single-target (M→1) cases but is ineffective for multi-target attacks. This suggests that applying NAD first to sanitize model representations and then using FTSAM for fine-tuning could substantially lower ASR.

3. The paper does not quantitatively evaluate how similar the surrogate data distribution is to the target one. It would be useful to report a feature-space similarity metric (e.g., CKA or FID) and identify the approximate sample size at which surrogate features become comparable to those of the target model.

### Questions
1. How sensitive is the theoretical guarantee to deviations in representation alignment between the surrogate and target models?  

2. Can the authors empirically quantify the feature-space similarity between the surrogate and victim models (e.g., via CKA, FID, or linear probe accuracy) to validate the assumption used in Proposition 2?  

3. At what surrogate dataset scale do the feature representations become sufficiently aligned for successful transfer, and how consistent is this threshold across different architectures?  

4. Given that the surrogate dataset is non-IID and much smaller (5k–15k samples), how does the attack performance change as the domain shift increases or the sample size decreases?  

5. Compared with highly OOD surrogate datasets (such as using SVHN for CIFAR-10), how can the degree of “OOD-ness” be quantitatively measured? Even if the data distributions do not overlap, can this distance be meaningfully quantified?  

6. If the target model undergoes slight fine-tuning or domain adaptation after deployment, how does the proposed method perform—does the backdoor persist or degrade over time?

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
5

### Summary
This paper introduces Arcueid, a theoretically grounded multi-trigger backdoor attack framework capable of operating under black-box settings and extremely low poisoning budgets. The core idea is the Joint Cloud Shaping Multi-trigger Optimization, which enforces compactness within each trigger-induced feature cloud and separation across triggers in the latent space. This decouples trigger generation from label mapping, enabling flexible M→M, M→N, and M→1 backdoor paradigms. Extensive experiments across multiple datasets (CIFAR-10/100, TinyImageNet) and architectures (ResNet, VGG, ViT) show high attack success rates (>97%) with minimal clean accuracy drop (<5%). The paper also demonstrates robustness to state-of-the-art defenses.

### Strengths
- Comprehensive evaluation on diverse models and datasets, achieving high ASR and stealth even under low poisoning rates.

- The experiments include both dirty label and clean label for the proposed attack.

- This paper provides a theoretical analysis.

### Weaknesses
- Experimental details are missing. Hyperparameters (such as learning rate, epoch, optimizer, etc) should be provided at least in the appendix.

- The threat model is described, but no details are given in the experiments. I couldn't find out what the specific data is used for $D_{sur}$, nor the specific architecture for $f_{sur}$. It is crucial because this paper claims that the proposed attack is effective under a black-box threat model.

- The "Joint Cloud Shaping Multi-trigger Optimization" is a part of Arcueid, so it cannot be two contributions.

### Questions
Thanks for the interesting paper, but I have a few suggestions. I think this paper overclaims on a few points.

- The drop in clean accuracy (< 5%) is not negligible. BadNets has almost 0% drop in clean accuracy.

- The limitation "L1: Rigid Dependency" ignored dynamic attacks [A], M-N attacks[B], Source-Specific and Dynamic-Triggers attacks[C], etc. I suggest that the authors should include more related and well-known works.

- I do not see any difficulties for the BadNets poisoning strategy to achieve M-N, M-M, or M-1 attacks. I'd expect that putting multiple triggers to poison the training data could achieve the same attack performance. The authors should demonstrate whether a simple BadNets approach is effective or not, with both empirical results and some analysis.

I'd suggest that the authors provide evidence (for example, an experiment) to support every statement in the paper, rather than just evaluating the performance of the proposed attack.

[A] Input-Aware Dynamic Backdoor Attack

[B] M-to-n backdoor paradigm: A multi-trigger and multi-target attack to deep learning models

[C] Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Arcueid, a multi-trigger backdoor attack framework that operates across three paradigms (M→M, M→N, M→1) under black-box constraints with extremely low poisoning rates. The core contribution is a "Joint Cloud Shaping Multi-trigger Optimization" strategy that shapes feature representations to create compact, separated trigger-induced clusters.

### Strengths
1. Strong theoretical foundation: The paper provides rigorous theoretical analysis with formal propositions and proofs for feasibility, non-interference, transferability, and stability.

2. Practical threat model: Black-box access with extremely low poisoning budgets addresses realistic constraints

3. Novel technical approach: Decoupling trigger optimization from label mapping is elegant and enables paradigm flexibility. The joint cloud shaping objective is well-motivated.

4. Thorough supplementary materials: Extensive appendix with proofs, additional experiments, and reproducibility details.

### Weaknesses
1. Limited novelty in core technique: While the application is novel, the core optimization (minimizing intra-cluster variance + maximizing inter-cluster separation) is standard in clustering/metric learning. The connection to existing literature (e.g., contrastive learning, metric learning) is not discussed.

2. Computational cost not addressed: No analysis of optimization time for trigger generation, scalability to larger K (number of triggers), comparison of computational overhead vs. baselines

3. Experimental design:

a) Clean-label instability (Table 4): ASRs in clean-label attack are strong on CIFAR-10 (98% ASR) unstable on larger datasets:
- TinyImageNet 200→200: 79.8%±18.3% ASR (unreliable, some runs ~62%)
- CIFAR-100 100→100: 82.3%±10.1% ASR

Paper claims method "scales reliably" but provides no analysis of why variance explodes or how to mitigate it

b) Unfair baseline comparisons (Table 6): Tests baselines at 0.01% PR, while original methods designed for 0.05-1% PR, so 0.01% is likely outside effective operating range. There should be ablations across multiple PRs, justification for 0.01% comparison point.

c) Missing relevant multi-trigger comparisons:
- Marksman (Doan et al., NeurIPS 2022) - cited line 127 but not compared, supports arbitrary target mapping
- One-to-N/N-to-One (Xue et al., 2022) - cited but not compared, directly studies N→1 and 1→N paradigms
- Sleeper Agent (Souri et al., 2022), LIRA (Doan et al., 2021) - cited but not tested

d) Table 1 (main results) lacks any baseline comparisons - difficult to assess relative performance

4. Theory-practice gap (Equation 9):

Trigger optimization assumes fixed $\theta$ but victim training updates $\theta$ on poisoned data. Propositions 1-6 guarantee feasibility for static decision boundaries but provide no analysis of stability when $\theta$ evolves during training.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

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
A multi-trigger backdoor framework is proposed to generate multi-trigger backdoor attacks.  Three key limitations were recognized: single trigger, fixed trigger target mapping, and unrealistic threat scenarios. The proposed method requires black-box knowledge and low poisoning budgets. The key idea is to minimize intra-trigger variance and maximize inter-trigger separation jointly. Comprehensive experiments are conducted on different datasets showing the effectiveness of the proposed method.

### Strengths
Key limitations of existing backdoor attacks have been clearly recognized, where the authors propose Arcueid to overcome those limitations concretely. Comprehensive experiments have been conducted to verify the effectiveness of the proposed method, focusing on pre-, mid-, and post-training defenses. Ablation studies are also extensive and convincing.

### Weaknesses
Adaptive defense is not discussed. The proposed method takes the adaptive backdoor attack into the design, but adaptive defense is not clearly discussed. Arcueid explicitly shapes compact, separable clouds, a defender who is aware of the design can be expected to detect or disrupt it. No attacker-aware evaluations (where the attacker optimizes to minimize detector signals) are reported. It would be great if the authors could provide a more detailed discussion on it. 


Feature space defenses. This work focuses on feature-space manipulation but provides limited comparison with existing feature-space defenses, such as Activation Clustering [a]. Feature space defenses explicitly detect or mitigate latent-space anomalies that Arcueid’s cloud-shaping mechanism may introduce. Without including such representation-level defenses, the evaluation remains incomplete.  It would be great if the authors could discuss it. 

[a] Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.

### Questions
Please provide a discussion on the adaptive and especially feature-space defense.

### Soundness
2

### Presentation
2

### Contribution
2
