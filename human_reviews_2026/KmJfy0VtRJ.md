# Consistency and Unified Semantic Regularization for Generalized Category Discovery

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Generalized Category Discovery (GCD) aims to leverage labeled data to learn clustering-friendly representations for unlabeled data. Among existing approaches, self-supervised contrastive learning (CL) is the most widely adopted, typically optimizing two objectives: $\texttt{consistency}$ and $\texttt{uniformity}$. However, we observe an inherent tension between these objectives—while uniformity encourages a uniform distribution across the feature space, it can conflict with the goal of learning class-discriminative representations. To address this, we propose a two-stage framework that disentangles feature learning from self-contrastive objectives to better capture category concepts and represent auxiliary unlabeled data. In the first stage, the model constructs visual representations anchored to known category prototypes while reinforcing semantic links between labeled classes. The second stage extends this representation space to discover novel categories using a consistency objective combined with specifically designed regularization. Moreover, we introduce a novel $\texttt{Semantic Exploration Energy mechanism}$ to capture shared semantics across categories, thereby mitigating the information loss caused by prototype orthogonalization. The proposed framework—Consistency and Unified Semantic Regularization ($\texttt{CURE}$)—retains the consistency objective and enhances it with semantic energy regularization. Our CURE achieves state-of-the-art performance across multiple benchmarks and significantly alleviates performance imbalance between known and novel classes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies the task of Generalized Category Discovery (GCD). Motivated by the conflicts between consistency and uniformity in self-supervised contrastive learning (CL), this paper proposes a two-stage framework that disentangles feature learning from self-contrastive objectives. The authors further introduce Semantic Exploration Energy
Mechanism to enhance feature representation. Comprehensive experiments validate the superiority of the proposed method.

### Strengths
1. This paper is well-motivated and easy to follow.
2. This paper proposes several novel components, including semantic exploration energy, label-guided concept structure, as well as structure-guided semantic expansion.
3. Comprehensive comparative results and ablations are conducted to validate the method.

### Weaknesses
1. Although the method achieves remarkable performance, it is a little bit complex with several hyper-parameters. The effect of each important parameter should be presented.
2. The method contains two-stage training, each with several components. I was wondering whether the method consumes a lot more memory and training time than conventional GCD methods. The comparison of computational resources and training time should be included.
3. Some references and citations are missing. The paper should cite all the baseline methods for comparison (Table 1) in the references list, i.e., ProtoGCD and PrCAL.

### Questions
Please include experiments and analysis raised in weaknesses.

### Soundness
3

### Presentation
4

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
This paper presents a two-stage framework by disentangling feature learning from self-contrastive objectives to better capture category concepts and represent auxiliary unlabeled data. The proposed method mitigates the adverse effect of uniformity on novel category discovery.

### Strengths
1. New idea to improve the class discriminative representations.
2. The paper is well-organized and can be easily understood.
3. Good results on different benchmark datasets, including CIFAR-10, CIFAR-100, ImageNet-100, CUB-200, Stanford-Cars and Herbarium19.

### Weaknesses
1. The technical novelty of this work seems weak, as most key components are designed by slightly modifying existing modules.
2. The literature review needs to focus more on GCD and explain better the motivations.
3. The main problem to be solved is the adverse effects of representation uniformity induced by CL. Does it mean that this is also the main problem in GCD? I believe CL and GCD have different problems to be solved.

### Questions
See weaknesses

### Soundness
3

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
5

### Summary
The paper tackles Generalized Category Discovery (GCD) and argues that the usual contrastive-learning recipe (consistency + uniformity) is internally conflicted: uniformity pushes features to spread on the hypersphere, which can hurt class-discriminative structure for GCD. To avoid this, the authors propose CURE, a two-stage pipeline:

1. Stage I: use labeled data to build a “semantic topology” of known-class prototypes; instead of enforcing orthogonality, they add a Semantic Exploration Energy (SEE) regularizer that keeps prototypes softly connected, plus a label-guided concept structure to push this structure down to the feature space. 
2. Stage II: run structure-guided semantic expansion — cluster all data, align part of the clusters to known classes via Hungarian matching, treat the rest as novel-class candidates, and then train with a JS-consistency loss, logit-adjusted self-distillation, entropy regularization, and a second-stage semantic energy over the full prototype set. 

This lets them discard CL-style uniformity while still learning a semantically smooth space, and achieves SOTA or near-SOTA results on 7 GCD benchmarks.

### Strengths
1. The paper is easy to follow and the overall pipeline is clearly presented.

2. The motivation is well aligned with the proposed two-stage design.

3. The experimental section is reasonably comprehensive (multiple datasets, several ablations).

### Weaknesses
1. **Core motivation is unverified.** The whole paper rests on the claim that “uniformity may hurt class separation and thus GCD,” but there is no ablation that turns uniformity on/off (or varies its strength) to demonstrate this. Without such evidence, it is unclear whether uniformity is actually the bottleneck in current GCD pipelines.
2. **Missing discussion of closely related ideas.** Prior work such as **hyperGCD**[1] starts from a very similar observation — that learning on a spherical / overly uniform space can be suboptimal for GCD — but this paper does not analyze the connection, differences in geometry, or when the proposed semantic energy is preferable. This weakens the motivation part.
3. **Limited novelty.** Apart from Semantic Exploration Energy (SEE), most components already exist in recent parametric GCD methods.

   * “Label-guided concept structure” is essentially supervised/contrastive alignment on labeled data, which is standard in SimGCD[2], ProtoGCD[3], DebGCD[4], LegoGCD[5], CMS[6], etc.
   * The cluster-to-prototype alignment with Hungarian matching is very close to earlier “pseudo-label → prototype” or “cluster → parametric head” pipelines (e.g., UNO [7] and later GCD variants).
   * “Semantic consistency optimization” is just multi-view consistency, which almost all recent GCD methods use.
   * “Logit-aware self-distillation” and “virtual sampling + entropy regularization” are minor engineering refinements.

     Given this, the paper should make a much more precise novelty claim.
4. **SEE is not empirically validated.** The method claims to “preserve semantic structure,” but no evidence is shown: no before/after prototype–prototype similarity, no qualitative example on a fine-grained dataset (e.g., whether visually close bird species stay close), and no analysis of whether SEE avoids simply shrinking the prototype space. Without such visualization/analysis, it is hard to tell whether SEE is doing what it is supposed to do.
5. **Typos.** In Table 2, CIFAR-100, the entry “85.0 6.4 82.3” is clearly a typo and should be fixed.

[1] Hyperbolic Category Discovery

[2] Parametric classification for generalized category discovery: A baseline study.

[3] ProtoGCD: Unified and Unbiased Prototype Learning for Generalized Category Discovery

[4] DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery

[5] Solving the Catastrophic Forgetting Problem in Generalized Category Discovery
 
[6] Contrastive Mean-Shift Learning for Generalized Category Discovery

[7] A Unified Objective for Novel Class Discovery

### Questions
1. Please show a concrete prototype–prototype similarity matrix before and after applying SEE on a fine-grained dataset, and highlight which semantic relations are actually preserved. Otherwise, the benefit of SEE is speculative.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the problem of Generalized Category Discovery (GCD) by critiquing the widely-used contrastive learning (CL) paradigm. The authors identify a key tension between the uniformity objective of CL, which promotes a uniform feature distribution, and the need for class-structured representations for effective clustering. To resolve this, they propose a two-stage framework named CURE. In the first stage, CURE leverages labeled data to construct a semantically meaningful prototype space, using a novel Semantic Exploration Energy (SEE) regularizer to prevent prototype fragmentation. In the second stage, the framework discovers novel categories by applying consistency constraints (via JS-divergence), self-distillation, and the SEE regularizer to both labeled and unlabeled data. The authors claim this approach abandons the problematic uniformity constraint of CL, leading to improved performance and better balance between known and novel class discovery.

### Strengths
1. This paper argues that the dominant contrastive learning paradigm in Generalized Category Discovery is suboptimal. Specifically, it posits that the uniformity objective of CL, which encourages features to be uniformly distributed, conflicts with the goal of learning class-discriminative, clustered representations needed for GCD.

2. The key claim is that CURE is the first GCD framework to completely discard CL methods for unlabeled data, relying solely on consistency and semantic structuring. This is intended to avoid the "noise" caused by the uniformity objective.

3. The paper correctly identifies that standard supervised learning with one-hot labels tends to enforce prototype orthogonality, which can sever semantic links between classes. This is a real issue that hinders generalization to novel, but semantically related, categories.

### Weaknesses
1. The central claim to be the "first GCD framework that aims to alleviate the impact of uniformity by entirely discarding CL methods" is a major overstatement. The method's core mechanism for learning from unlabeled data is a consistency loss (JS-divergence) between augmentations. This consistency regularization is a foundational principle of self-supervised learning and a key component of many modern CL frameworks (e.g., BYOL, SimSiam), which are precisely the methods that moved away from explicit negative sampling. The paper does not discard CL; it discards the InfoNCE formulation and its associated negative-sampling-based uniformity term. This mischaracterization of the contribution is a fundamental weakness. The work is a reformulation of CL, not a departure from it.

2. The motivation is to create a more "clustering-friendly" representation space by removing the uniformity constraint. The proposed solution replaces this implicit regularization (uniformity from InfoNCE) with a different set of explicit regularizers (SEE, consistency loss, etc.). It is not self-evident that this new combination is inherently more "principled" for clustering, rather than just being a different, empirically effective, set of constraints. 

3. In addition, progress [1] has been made on the uniformity of features in general category discovery, where plug-and-play loss functions are used to discuss the information represented by the covariance matrix. A comparison and discussion with this work should be conducted.

4. Semantic Exploration Energy is effectively a form of prototype graph regularization, encouraging a compact manifold. Similar concepts of regularizing the geometry of the prototype space exist in metric learning and zero-shot learning. The formulation itself is a straightforward application.


[1] Generalized Category Discovery via Token Manifold Capacity Learning. In Arxiv, 2025.

### Questions
1. The authors claim to "entirely discard CL methods". However, the JS-divergence loss on augmented views (L_JS) is a cornerstone of consistency-based self-supervised learning, a major branch of CL. Can the authors clarify this claim? Would it be more accurate to state that the method discards the negative-sampling-based uniformity objective of InfoNCE-style CL, rather than CL as a whole?

2. The two-stage design appears crucial. Stage 1 learns a "semantic topology" from labeled data. How sensitive is the performance of Stage 2 to the quality and nature of the representation learned in Stage 1? For instance, if the labeled classes are not semantically representative of the novel classes (e.g., labeled are all animals, novel are all vehicles), would the structure imposed by SEE in Stage 1 become a harmful prior during the discovery process in Stage 2?

### Soundness
3

### Presentation
2

### Contribution
2
