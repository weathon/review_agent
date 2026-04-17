# Swap-and-Spoil: Untargeted Byzantine Attacks via Class-Consistent View Swaps in Vertical Federated Learning

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Vertical Federated Learning (VFL) secures a highly privacy-preserving multi-party training paradigm in which features are vertically distributed across participants for the same sample space. Security attacks against VFL have gained attention recently, but most discussions revolve around data poisoning attacks such as backdoor attacks. Byzantine attack against a federated learning system can target the main model performance and drop its accuracy with a single adversary participating in the training. While such untargeted Byzantine attacks have been explored in horizontal settings, they still remain underexplored in vertical settings of federated systems. In this paper, we demonstrate how an adversary can mount a successful untargeted Byzantine attack that drives down the global model’s inference-time accuracy. To realize this, we perform a consistent cluster-based swapping in the feature space, creating a persistent and poisoned cross-view association during training. The model internalizes this adversary-induced association and, when evaluated on clean, correctly aligned data, fails dramatically. We also show that, the widely-practiced defenses in VFL fail to detect the attack without degrading the model performance. Through this endeavour, our findings establish untargeted Byzantine attacks as a real, underexplored threat to VFL and motivate the design of robust, VFL-specific defenses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces "Swap-and-Spoil," a novel untargeted Byzantine attack specifically designed for VFL. The main contribution is the attack exploits VFL's structure by implementing a class-consistent view swap among malicious participants. This method effectively corrupts the overall model accuracy without specific targeting.

### Strengths
Unlike traditional data poisoning, the paper introduces a novel Byzantine attack mechanism specific to VFL’s structure: a class-consistent view swap. This approach is highly original. It reveals a new vulnerability in VFL systems by presenting an attack that is stealthy and hard to detect with existing defenses. This makes it an important benchmark for developing more robust defenses and gives it significant practical value. The proposed attack mechanism is clearly explained, and experiments support that it effectively degrades overall model performance.

### Weaknesses
While the effectiveness of the attack is demonstrated, the paper lacks comparisons with random swap attacks and evaluations of resistance against existing robust aggregation algorithms. Figures and tables, especially Table 5, overflow their boundaries and there are numerous grammatical and spelling errors, necessitating thorough revisions. The definition and implementation of the core concept of “class consistency” should be described more clearly, both mathematically and intuitively.

### Questions
Q1: Can the authors define the “class-consistent view swapping” strategy more formally, either mathematically or algorithmically (for example with pseudocode)?
Q2: Does an adversary need access to the true class labels of the training samples to carry out the attack?
Q3: By how much does “class-consistency” improve the attack success rate (i.e., the drop in downstream model accuracy) compared to a simple Random Swap Attack (RSA)?
Q4: Can you empirically demonstrate and quantify how effective the attack remains when a VFL system employs established robust aggregation mechanisms such as Krum, Trimmed Mean, or Median?
Q5: How does the attack’s effectiveness change when the number of participants is much larger (for example N > 10)? 
Q6: How is the attack affected on datasets where features are distributed across clients in a more complex and heterogeneous way (for example, with high inter-feature correlation or imbalanced feature sets)?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates a new class of untargeted Byzantine attacks in Vertical Federated Learning (VFL), an area where prior work has mainly focused on targeted backdoors or label inference. The authors propose a cluster-consistent feature-swapping attack that poisons the joint feature representation during training. The attack operates in two stages: first, the adversary uses limited label information to infer latent class structure via semi-supervised clustering, and second, it performs class-consistent feature swaps between clusters to create persistent cross-view misalignments.

### Strengths
- **Novel attack concept:** Introducing untargeted Byzantine corruption via class-consistent view swapping fills a clear gap between random noise and targeted backdoor attacks in VFL.

- **Well-defined threat model:** The assumption of a passive adversary with partial label access is realistic and consistent with real-world cross-organization data sharing.

### Weaknesses
- **Simplified setting:** Experiments are limited to two-party VFL on relatively simple datasets (MNIST, FashionMNIST, UCI tabular). The attack’s scalability to multi-party or high-dimensional, real-world VFL remains uncertain.

- **Weak defense discussion:** Although new defense gates are mentioned, they are rudimentary and come with severe utility degradation. The paper stops short of providing meaningful defense insights beyond confirming that existing defenses fail.

- **Missing deeper analysis of attack transferability:** It is unclear how robust the cluster-swapping attack remains when the adversary’s clustering accuracy degrades or when auxiliary label availability is further reduced.

- **No ablation on key assumptions:** The 5% labeled auxiliary dataset assumption is strong; the paper doesn’t quantify how attack effectiveness changes with smaller or noisier supervision.

### Questions
Please see the weaknesses.

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
4

### Summary
This paper studies the byzantine attack in VFL setting. Specifically, they target the training phase attack that corrupt the top model (hold by the active cient) such that it  learn spurious correlation of the adversary's view and the clean view, and make wrong inference. They propose a two stage training methods, first is using clustering to predict labels of adversary's features, second stage is to swap features and poison models. The two stage methods are well designed for efficient attack. The empirical evaluation shows it can keep good performance and circumvent the defense. However, the experiment lacks baselines.

### Strengths
1. The attack threat model assumes the passive adversary which is more challengable.
2. The methods are systematical, including two designed steps.

### Weaknesses
1. The first cluster step will alter the embedding distribution and make it misaligned the natural distribution of the honest training embedding, which hurts the model's performance.
2. There are other clustering or shadow model based methods to predict labels. However, the authors did not compare any.
3. The baselines are too less only with the random noise attack. However, in introduction, the authors have mentioned other byzantine attacks works like [1] but they did not compare. 
4. There is no ablation studies, like hoe the number of passive clients will affect the attack, and the effect of model layers.
4. The format of Table is our the margin.

[1] Hijack Vertical Federated Learning Models As One Party

### Questions
1. why "A lower attack accuracy indicates a stronger attack"?
2. How to keep internal consistency in the stage 2, since you have change features to different clusters, which will lead to a different distribution of embedding.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an untargeted Byzantine attack specifically tailored to Vertical Federated Learning (VFL). The attack, "Swap-and-Spoil," consists of two stages: (1) the adversary uses a small auxiliary labeled dataset (5%) to learn latent class structures through semi-supervised clustering, and (2) performs class-consistent view swaps during training, exchanging features across inferred clusters. This manipulation disrupts cross-view feature alignment and induces inference-time accuracy collapse on clean data. Experiments show that Swap-and-Spoil effectively bypasses common VFL defenses without significantly affecting training metrics, revealing a critical gap in current security mechanisms.

### Strengths
- Introduces a new untargeted Byzantine attack tailored to the unique structure of VFL.
- Distinct from trigger-based or gradient-manipulation attacks, the class-consistent swap strategy is stealthy and generalizable.
- Strong experimental validation across both visual (MNIST, CIFAR-10) and tabular (UCI-HAR, Mushroom) datasets.

### Weaknesses
1. A reconstruction-based or embedding-monitoring defense might still detect statistical inconsistencies caused by swapped feature associations.
2. Visualization (e.g., t-SNE/PCA) of embeddings before and after swapping would clarify how the attack alters representation space.
3. The attack depends heavily on the auxiliary labeled data (5%); an ablation study varying this proportion (e.g., 1%, 10%) would improve understanding of feasibility and robustness.
4. The clustering and contrastive learning steps (SimCLR + GMM) may incur high computational cost for a single adversarial client—this should be quantified.
5. Sensitivity analysis on the parameter *k* (for top-*k* farthest swaps) is missing.
6. The method focuses on two-party VFL; discussion on scalability to multi-party settings would strengthen the paper.
7. The paper should clarify the distinction between "untargeted degradation" and the occasional mention of "target class misclassification," as this could confuse readers about the attack objective.

### Questions
Please address the aforementioned weaknesses. Specifically, clarify Swap-and-Spoil’s detectability under reconstruction-based defenses, analyze sensitivity to auxiliary data and k, quantify computational overhead, and elaborate on scalability to multi-party settings.

### Soundness
4

### Presentation
3

### Contribution
3
