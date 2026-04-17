# Self-Supervised Learning with Side Information

- Decision: Reject
- Scores: 4, 8, 4, 6

## Abstract
A core assumption behind many successful self-supervised learning (SSL) methods is that different views of the same input share the information needed for downstream tasks. However, this MultiView assumption can be overly permissive in real-world settings, where task-irrelevant features may persist across views and become entangled with useful signals. Motivated by challenges in colonoscopy—where polyp cues must be isolated from dominant but irrelevant background textures—we present an information-theoretic analysis of this general failure mode in SSL. We further formalize this with our proposed Nuisance-Free MultiView (NF-MV) assumption, which reframes the goal of SSL as learning representations that are sufficient for task-relevant information while being invariant to shared nuisance structure. We theoretically show that such representations yield improved generalization, and derive an idealized objective that balances standard view alignment with a mutual information penalty on nuisance content.

To implement this in practice, we introduce a method that leverages side information—auxiliary data that shares nuisance structure but does not contain any task-relevant signals. The nuisance penalty is then approximated using a Jensen-Shannon divergence between main and side representations, in a way that is tractable and compatible with standard joint embedding architectures.

Experiments on synthetic tasks with spurious correlations and on real-world colonoscopy datasets demonstrate that the proposed method improves generalization for a wide range of SSL methods and architectures by learning the relevant features. These findings highlight the benefits of explicitly modelling what should not be preserved during self-supervised learning, offering a new and practical perspective on the MultiView framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper tackles an important limitation of mainstream self-supervised learning: the Multi-View (MV) assumption treats all features shared across augmented views as useful, even when many are task-irrelevant nuisances. Therefore, it proposes a new Nuisance-Free Multi-View assumption that explicitly excludes nuisance variables. Specifically, it implements this framework using side information—auxiliary data that shares nuisance structure but lacks task-relevant information—and penalize representational overlap using a Jensen-Shannon divergence between main and side representations. However, the theoretical advance is incremental and the experimental breadth is insufficient to assert broad applicability.

### Strengths
1.	It cleanly formalises nuisance in an information-theoretic way and gives a generalisation bound that rewards minimal representations for the task subset, not all MV tasks.
2.	One extra term (JSD) with no additional encoders, or costly MI estimators. 
3.	Writing is good and easy to follow.

### Weaknesses
1.	CIFAR+MNIST is useful, yet the nuisance (CIFAR background) is low-dimensional and visually very distinct from the signal (MNIST digit). It is unclear whether the method survives nuisances that are semantically richer or partially correlated with the label.
2.	The scale-up ability is not discussed, since CIFAR and MNIST only contain 10 classes.
3.	No comparison with other contemporary medical-SSL methods such as M2CRL, FocusMAE, or SepCLR run on identical data splits. Reported numbers mix different backbones and pre-training sets, making fair comparison difficult.
4.	Sensitivity analyses is missing.

### Questions
1.	What happens if the side set is not label-free?
2.	How does γ scale with nuisance dimensionality? How sensitive is γ to different nuisance strengths? 
3.	What if the side set itself contains some task-relevant signal? This method assumes a single encoder that must represent both nuisance and signal to disentangle them. For very high nuisance complexity this could increase the necessary encoder capacity, contradicting the “minimal representation” narrative.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- This paper critiques the standard multi view asusmpting in self-supervisied learning for being too permissive, allowing task-irrelevant nuisance factors, such as dominant background textures in colonoscopy, to persist across views and become entangled with task-relevant signals.
- The authors introduce a Nuisance-Free MultiView (NF-MV) assumption, which reframes the objective of self-supervised learning (SSL) as learning representations sufficient for the task while being explicitly invariant to shared nuisance structure.
- The nuisance penalty is implemented practically by approximating an idealized mutual information objective using the Jensen-Shannon divergence (JSD) between the representations of the main and side data distributions.
- The authors provide a simple, modular, and architecture-agnostic extension to standard Joint Embedding Architectures (JEAs) like Barlow Twins, CorInfoMax, and Masked Siamese Networks (MSN), introducing negligible computational overhead.
- Experiments on controlled Cifar10+MNIST datasets, engineered with spurious correlations, demonstrated that the side information (SI) variants significantly improved performance on generalization tasks compared to standard and naive baselines.

### Strengths
- Strong conceptual contribution: The formalization of the Nuisance-Free MultiView (NF-MV) assumption is a solid conceptual contribution. It provides a new and useful information-theoretic perspective on what SSL should (and should not) be learning, backed by a theoretical argument for improved generalization. The NF-MV assumption provides a principled extension of the standard MultiView framework that explicitly models what representations should not preserve. Theorem 1 formalizes the generalization benefit of learning minimal sufficient representations for the target task rather than all MultiView-induced tasks, adapting the Xu–Raginsky bound to show strictly tighter generalization bounds when `I(Z'; X) < I(Z; X)`.
- Simple and general: The proposed method is elegant in its simplicity. It is not a complex new architecture but a modular penalty term that can be added to existing joint embedding architectures. The paper contrasts this favorably against contrastive analysis methods like SepCLR, which require multiple separate encoders (one for salient features, one for common features), multiple feature spaces, and substantially higher computational and memory costs.
- Strong emperical results: There are two sets of experiments in the paper, on a hybrid synthetic Cifar+MNIST, and on real-world colonoscopy data.
  - The Cifar+MNIST experiments provide clear evidence that the method helps models overcome spurious correlations.
  - On the colonoscopy experiments with PolypsSet, this methods achieves 80.3% F1 on histology classification while matching models trained on an order of magnitude more private data. Clearly demonstrates the value of leveraging typically discarded background frames as side information.

### Weaknesses
- Limited guidance on side information selection: The Nuisance-Free MultiView (NF-MV) assumption strictly requires that the nuisance variable n is independent of the label y, formalized as `I(y;n)=0`. This assumption is restrictive in real-world scenarios, and the paper lists this precise point as a core limitation. The authors provide limited guidance on systematically identifying appropriate side information sources beyond the colonoscopy example. How would practitioners determine whether available auxiliary data satisfies this assumption? How much contamination is tolerable before performance degrades below baseline?
- Coarse mutual information proxy: One weakness is in the practical implementation of the JSD penalty. While the theoretical objective aims to maximize the complex mutual information $\mathbf{I}(\mathbf{z}; B_\alpha)$, the practical implementation computes the JSD between the average softmax outputs of the main and side batches. The authors note this, and it clearly works, but it feels disconnected from the information-theoretic motivation and may not be the most powerful way to enforce separation. The authors provides no empirical validation of approximation quality, such as, how well does $\text{JSD}_\alpha(\mathbf{\bar{z}}_X || \mathbf{\bar{z}}_S)$ correlate with $\mathbf{I}(\mathbf{z}; B)$ estimated via methods like CLUB or InfoNCE? Given that the cited MI estimators (lines 257-262) are dismissed for "high variance and bias" without demonstration, it's unclear whether the simplicity+tractability tradeoff is optimal or whether intermediate approaches might be more effective.
- Minor point on hyperparameter guidance: While the `γ` values differ across methods (reflecting their different loss scales), the paper could provide a heuristic for initial `γ` selection—perhaps as a ratio to the primary SSL loss magnitude. However, the hyperparameter search required is modest and shows predictable trends.

### Questions
- Can you provide more systematic guidance or a principled procedure for identifying valid side information in new domains? What diagnostic tests or metrics could practitioners use to verify the `I(y;n)=0` assumption? (really, just restating my weakness point).
- Could you confirm that the "MSN" row (76.1, avg F1) is your own controlled baseline, using the same ViT-S architecture and pre-trained on the same REAL-Colon dataset as your method, just without the side information? And that the "MSN-SI (ours)" row (80.3, avg F1) is the identical setup plus your JSD penalty? The table formatting doesn't clearly distinguish their baseline from prior work.

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
This paper presents the Nuisance-Free MultiView (NF-MV) framework for self-supervised learning (SSL), which argues that standard MultiView-based SSL methods (e.g., SimCLR, Barlow Twins, MSN) are too permissive because they retain shared but task-irrelevant features. To address this, the authors propose incorporating side information—data that shares nuisance structure but lacks task-relevant signals—and penalizing representational overlap with a Jensen–Shannon divergence (JSD) term. Experiments on synthetic CIFAR+MNIST tasks and colonoscopy datasets show improved generalization.

### Strengths
- Identifies a real and well-documented weakness in existing self-supervised learning (SSL): the MultiView assumption often retains shared nuisance features that hurt downstream generalization.
- Provides a formal, information-theoretic reformulation through the Nuisance-Free MultiView (NF-MV) assumption, which is explicitly stated and mathematically well-defined.
- Works seamlessly with multiple Joint-Embedding Architectures (JEAs): Barlow Twins, CorInfoMax, Masked Siamese Networks, etc.
- Computationally lightweight: the JSD is computed from existing model outputs (softmaigible overhead.

### Weaknesses
- The paper’s central claim, that SSL can be improved by discouraging shared nuisance structure, is not new. The idea of separating task-relevant vs nuisance components dates back to Domain Separation Networks, Contrastive Analysis, and information bottleneck extensions. The NF-MV assumption is a relabeling of those ideas with minor formal changes; it adds no new theoretical insight beyond replacing the nuisance variable n with a side dataset S. The theoretical results (Theorem 1) are direct corollaries of Xu & Raginsky (2017) and do not provide novel analysis or guarantees specific to SSL.
- Method is heuristic and poorly justified theoretically. The proposed JSD penalty is applied to batch-averaged prototype softmax distributions rather than instance-level representations. This design is ad hoc and undermines the information-theoretic motivation, as it does not properly estimate mutual information. The approach maximizes separation (JSD) between main and side data rather than explicitly enforcing invariance to nuisances. The desired disentanglement of task and nuisance subspaces is therefore not guaranteed, and may in fact promote encoding of domain cues rather than removal of them. The method’s behavior depends critically on the quality and purity of the side data, which is not verifiable or robustly handled.
- Weak empirical validation. The synthetic CIFAR10+MNIST setup is a toy problem where the nuisance (background) is obvious and separable, offering limited insight into real-world complexity.
In the colonoscopy experiments, the side data (negative frames) and target data (positive frames) are trivially distinct in pixel distribution. Thus, improvements may stem from easier domain separation rather than true nuisance invariance.
Baselines are unevenly compared: SepCLR is trained for half as many epochs (500 vs 1000), and private datasets used by prior works differ in size and quality.
No other domains (e.g., natural videos, non-medical images) are tested, so generalization is unproven.
- Limited impact and scope. The method’s benefit depends on having large quantities of “nuisance-only” data (e.g., 87% negative frames in REAL-Colon). This is uncommon outside of specific medical-imaging domains.
Without clear theoretical innovation or strong empirical breadth, the contribution is too incremental for a top-tier venue.

### Questions
In open-domain SSL (e.g., ImageNet or LAION pretraining), how can “side information” be realistically defined and obtained? Specifically, what data could serve as a nuisance-only side set when task relevance is unknown—how would you verify its purity, scale it to diverse domains, and avoid suppressing features that later prove useful for downstream tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an extension to the standard MultiView assumption in self-supervised learning (SSL) by introducing the Nuisance-Free MultiView (NF-MV) assumption, which aims to make representations invariant to task-irrelevant (nuisance) factors while preserving task-relevant information. The authors motivate this with information-theoretic analysis, drawing from the Information Bottleneck principle, and implement it practically by leveraging "side information"—auxiliary data that shares nuisance structures but lacks task signals. They approximate a mutual information penalty using a Jensen-Shannon divergence (JSD) between representations from main and side data, making it compatible with joint embedding architectures (JEAs). Experiments on a synthetic Cifar10+MNIST dataset with spurious correlations and real-world colonoscopy data (using REAL-Colon) show improvements in generalization for methods like Barlow Twins, CorInfoMax, and Masked Siamese Networks.

### Strengths
1. The information-theoretic framing is solid and builds nicely on prior work like the MultiView assumption (Sridharan & Kakade, 2008) and extensions of the Information Bottleneck (Chechik & Tishby, 2002)
2.The JSD-based penalty is lightweight, adding negligible compute overhead, and plugs into existing JEAs without needing multiple encoders (unlike contrastive analysis methods like SepCLR). This modularity is appealing for real-world adoption.
3. The writing is clear, with good figures (e.g., Fig. 1 illustrating info overlap). It opens a perspective on explicitly modeling what not to learn in SSL, which could apply beyond medical imaging to domains with persistent nuisances.

### Weaknesses
1. The NF-MV assumption relies on well-defined side information that's nuisance-rich but task-irrelevant. In the colonoscopy case, polyp-negative frames work well, but the paper doesn't deeply explore how to identify or generate such data in less obvious domains. What if side info inadvertently includes subtle task signals?
2. While the synthetic and real experiments are solid, they're somewhat narrow. Only a few JEAs are tested (Barlow Twins, CorInfoMax, MSN), and no comparisons to other debiasing techniques like invariant risk minimization or recent SSL variants (e.g., DINOv2 or MAE adaptations). The colonoscopy results are promising, but use a single dataset for pre-training; cross-dataset validation (e.g., on KVASIR or EndoScene) would strengthen claims.
3. The JSD weighting (α, γ) and side info ratio (e.g., 5-40%) seem tuned per experiment, but ablation details are mostly in appendices. More discussion on robustness to these choices would help reproducibility.
4.How does performance scale with side info quality/quantity?

### Questions
See weak nesss
If my concern is resolved, I am willing to raise it to 8 points; this is a very novel work.

### Soundness
4

### Presentation
4

### Contribution
4
