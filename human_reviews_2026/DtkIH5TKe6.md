# Let OOD Feature Exploring Vast Predefined Classifiers

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6, 6

## Abstract
Real-world out-of-distribution (OOD) data exhibit broad, continually evolving distributions, rendering reliance solely on in-distribution (ID) data insufficient for robust detection. Consequently, methods leveraging auxiliary Outlier Exposure (OE) data have emerged, substantially enhancing generalization by jointly fine-tuning models on ID and large-scale OE data. However, many existing approaches primarily enforce orthogonality between ID and OE features while pushing OE predictions toward near-uniform, low-confidence scores, thus overlooking the controllability of representation geometry. We propose Vast Predefined Classifiers (VPC), which constructs a pre-specified Orthogonal Equiangular Feature Space (OEFS) to explicitly separate ID and OOD representations while capturing the rich variability of OOD features. We employ evidential priors to align ID features with their class-specific Equiangular Basic Vectors (EBVs), thereby preserving ID performance. In parallel, a new VEBV loss encourages OE features to explore the subspace spanned by Vast EBVs (VEBVs), enabling a rich characterization of diverse OOD patterns. This dual optimization, coupled with the prescribed geometric representation space, promotes optimal orthogonality between ID and OOD representations. Furthermore, we introduce the VPC Score, a discriminative metric based on the L2 activation intensity of features over the predefined classifiers. Extensive experiments across diverse OOD settings and training paradigms on benchmarks including CIFAR-10/100 and the ImageNet-1k, demonstrate strong and robust performance, validating VPC's effectiveness. Code is available at https://github.com/eightnight2049/VPC.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Vast Predefined Classifiers for OE-based OOD detection by combining Neural Collapse and Evidential Deep Learning. extensive experiments demonstrate the effectiveness of Vast Predefined Classifiers.

### Strengths
1. This paper is well-written.
2. This paper use 3 performance metrics
3. This paper provides extensive visualization

### Weaknesses
1. lack of novelty: the use of ETF for OOD detection has been explored in [1]. Compared with [1], this paper additionally introduce V vectors, which is a nature extenstion in the setting of OE and also similarly explored in [2]. Eq. (5) is normal learning objective in Evidential Deep Learning without any modification.
2. what is $f_j()$ in $\mathcal{L}_{OC}$?
3. the authors claim that their propsoed VPC score differs from softmax score and energy score in that the latter two are heuristic. However, there is thorectical justification for the VPC score with regards to the Separation between ID and OOD data.
4. Given that this paper focuses on improve the feature discriminative. The mostly reent feature-based OOD detection methods [3-5] should be discussed.
5. Lack of experiments on large-scale datasets (e.g. ImageNet-1k) and transformer-based backbones (e.g. ViT).
6. the authors are encouraged to conduct experiments on the OpenOOD benchmark.

[1]  Distributional Prototype Learning for Out-of-distribution Detection

[2]  Learning Placeholders for Open-Set Recognition

[3]  Conjnorm: Tractable density estimation for out-of-distribution detection

[4]  Out-of-distribution detection based on in-distribution data patterns memorization with modern hopfield energy

[5]  Learning with Mixture of Prototypes for Out-of-Distribution Detection

### Questions
see weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes VPC (Vast Predefined Classifiers), a novel approach for Out-of-Distribution (OOD) detection that explicitly models OOD variability by allocating a large set of predefined optimal classifiers distinct from ID classifiers. Inspired by Neural Collapse and equiangular basis vectors (EBVs), the method aligns the feature space to an Orthogonal Equiangular Feature Space (OEFS) with predefined EBVs/VEBVs. The authors further propose VPC Score, based on the VEBVs, which measures the relative activation intensity across orthogonal subspaces to distinguish ID and OOD samples. The authors demonstrate that VPC achieves state-of-the-art performance on CIFAR-10/100 benchmarks, significantly outperforming existing OE-based methods.

### Strengths
1. Originality: The proposed method VPC can sufficiently address the problems that are brought by traditional OE methods, i.e., incomplete ID/OOD separation and limited representation for OOD samples
2. Quality: The paper is written with clear motivation and logical support, making the proposed method more interpretable and theoretically justifiable.
3. Handling the variability of OOD features: different from previous works, the proposed VPC can jointly separate the ID/OOD features, as well as preserving the diverse OOD features, making it possible for more representation tasks.

### Weaknesses
1. Limited implementation details: the paper lacks sufficient discussion on how the hyperparameters are determined for optimal performance, such as the $\alpha$ and $\beta$ for the VPC score.
2. The experiment setting is limited to CIFAR datasets, which are relatively small. Given the empirical analysis in Table 5, the proposed method works by adopting a significantly larger V value, compared to the number of classes. The performance of VPC remains unclear when the number of ID classes is large (e.g., ImageNet-1K), which may require an even larger V value. This can potentially lead to unaffordable computational overhead or suboptimal OOD performance.

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on the out-of-distribution detection problem. Specifically, this work proposes a geometry-driven framework for OOD detection, that distinguishes ID data from OOD data by embedding to an orthogonal equiangular feature space, which provides an evidential priors that mitigates the instability from outlier exposure data, and then a new activation-based OOD score is introduced which is demonstrated to be effective than traditional softmax and energy scores.

### Strengths
1. The idea of this work that fixing the orthogonal equiangular basis builds on the neural collapse, to the reviewer's knowledge on the area of OOD detection, is new and interesting.
2. The paper has provided comprehensive experimental verification on demonstrating the performance of the proposed method.

### Weaknesses
1. Although the proposal is empirically demonstrated to be effective, there is no theoretical justification that rigorously justify or explain why the newly introduced representation structure can help the OOD separability or generalize to the unseen OOD distributions.
2. The computational cost or overhead seems to be much than previous OOD score based on softmax and energy, especially for large-scale ID dataset.
3. Except for the common benchmark datasets, some large-scale dataset should also be considered in the verification. And more advanced OOD detection methods like ASH or other scoring functions are not considered in the comparison, which should carefully support the claim that current proposal is SOTA.

### Questions
1. What guarantees separability between unseen OOD samples and ID features, as the OOD distribution shifts outside the auxiliary OE coverage, with the orthogonal equiangular construction? 
2. What is the computational complexity for maintaining and updating thousands of supervised classifiers in high dimensionality? Can this approach extend to modern vision transformers or multimodal models? 
3. Is it ensured that the penultimate layer retains the intended geometry under nonlinear activations or the presence of normalization layers? 
4. Neural Collapse is known to predominantly occur asymptotically with cross-entropy, under certain conditions. In the context of fixed EBVs and with extra constraints, what guarantees converge to the intended geometric configuration?

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
4

### Summary
The paper “LET OOD FEATURE EXPLORE VAST PREDEFINED CLASSIFIERS (VPC)” proposes a new framework for out-of-distribution (OOD) detection that focuses on explicitly modeling the geometry of feature representations. Current outlier exposure (OE) methods improve robustness by fine-tuning networks on in-distribution (ID) and auxiliary OOD data, but they largely encourage low-confidence OOD predictions or post-hoc score regularization without directly shaping feature space structure. As a result, separability between ID and OOD representations is often unstable, and OOD features lack sufficient representational diversity.

To address these issues, the authors introduce Vast Predefined Classifiers (VPC), which define an Orthogonal Equiangular Feature Space (OEFS) composed of fixed Equiangular Basic Vectors (EBVs) representing ID classes and a large pool of Vast EBVs (VEBVs) representing an extended OOD subspace. They design several complementary losses that guide feature alignment: an evidential prior–guided Neural Collapse (ENC) loss that stabilizes ID features toward their EBVs, a VEBV loss that encourages OOD features to explore the VEBV subspace, and an orthogonality constraint that prevents mixing of ID and OOD features. The framework introduces the VPC Score, based on the L2 activation intensity difference between the ID and OOD subspaces, serving as a class-agnostic and interpretable OOD detection metric. Experiments on CIFAR-10 and CIFAR-100 using multiple backbones (WideResNet, ResNet, DenseNet) show that VPC achieves state-of-the-art results, outperforming existing baselines such as OE, Energy-OE, DAL, and PFS. Ablation studies confirm that each component—particularly the geometric subspace design and the VEBV loss—contributes to improved performance.

### Strengths
The advantages of this method lie in its geometric interpretability, stable training through evidential priors, scalability for modeling diverse OOD patterns, and class-agnostic scoring that avoids bias from classifier heads. 

Empirical results demonstrate strong robustness and clear separability between ID and OOD features. 

Eq 5 in preventing over-bias toward any single class and promoting stable convergence of ID feature is new to me, sounds interesting.

### Weaknesses
The authors may need to further consider the definition of OOD detection in Sec 3.1. Defining OOD data to be those out of the label space might be more suitable for OOD detection. The adopted definition based on support set may fail to consider the case of covariate shift, which should be considered as ID instead of OOD. 

A paragraph of intuition at the beginning of Sec 3.2 will improve the clarity of the manuscript, otherwise readers may understand what the authors want to do very late in Sec 3.3 – 3.4. The intuition should echo the drawbacks of previous works in OOD diversity and some other things like this. 

Hat m in Eq 4 is not defined, I think it is the embedding feature, right? There are a lot of other notations that are not defined, such as \tau in the same formulation. The authors should carefully check about them. 

The proposed method is somehow like a variance of KNN, could the authors further highlight the improvement over KNN?

Could Eq 6 ensure the diversity of OOD basis vectors? The models tend to map OOD data to the same region in the embedding space even at the very beginning,  optimizing via Eq 6 will exacerbate this case and in this case V=1 is enough, obviously violating the claimed “vast” features in Sec 3.2.

The authors define w in the unit hypersphere, but, from my view, none of the objective in Sec 3 can ensure this constraint. 

As far as I know, many works conducted experiments on ImageNet benchmark or ViT family, such as [1,2]. I do not want to add the rebuttal burden of the authors, as many researchers lack sufficient computational resources. But you can consider it as a suggestion to improve the quality of this research in the future. 

[1] Watermarking for Out-of-distribution Detection

[2] Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources

### Questions
Kindly please see the weakness above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Vast Predefined Classifiers (VPC), a geometry-driven out-of-distribution (OOD) detection framework that leverages auxiliary outlier exposure (OE) data. VPC defines an Orthogonal Equiangular Feature Space (OEFS) with two sets of fixed, equiangular basis vectors:

* EBVs (Equiangular Basic Vectors) for in-distribution (ID) classes,
* VEBVs (Vast EBVs) to span a rich subspace for diverse OOD patterns.

During training, ID features are aligned to EBVs using an evidential prior-guided loss (LENC), while OE features are attracted to the VEBV subspace via a VEBV loss (LVEBV) and kept orthogonal to ID prototypes via an orthogonality constraint (LOC). OOD detection is performed using the VPC Score, based on the L2 activation intensity over the two subspaces. Experiments on CIFAR-10/100 show state-of-the-art performance in both two-stage and one-stage training setups.

### Strengths
The key strengths of the proposed approach are:

* Explicit geometric control: OEFS provides a principled, interpretable structure that cleanly separates ID and OOD representations.

* Scalable OOD modeling: VEBVs offer a flexible, expandable subspace to capture diverse OOD modes, unlike methods that merely push OOD to uniform outputs.

* Strong empirical results: VPC achieves SOTA on standard benchmarks across multiple architectures and training paradigms.

* Robust scoring: The VPC Score is class-agnostic, geometry-based, and avoids pitfalls of softmax-based confidence.

### Weaknesses
The main limitations of the proposed PVC approach are:

* Fixed classifier geometry: Predefined EBVs/VEBVs may limit adaptability to complex or evolving ID/OOD distributions, especially when class semantics shift.

* Computational overhead: Large VEBV sets increase memory and computation (e.g., for projection and loss calculation), though not deeply analyzed. It's not clear what the computational overhead of PVC is since no ablation results are proposed. Showing how PVC scales as a function of the dataset size, or providing some asymptotic bounds with respect to $K + V$ could help understand the applicability of PVC to real-world scenarios.

* Dependence on OE data: Like all OE-based methods, performance relies on the quality and relevance of auxiliary OOD data; may degrade if OE is unrepresentative.

* Limited real-world validation: The experiments are mainly confined to CIFAR benchmarks and lack evaluation on more complex datasets (e.g., ImageNet-scale or domain-shift scenarios).

### Questions
Although PVC achieved SOTA results, the fact that the experiments are only conducted on CIFAR raises a few doubts about the strength of the approach and its broader applicability to real-word scenarios. Could the authors show some results on stronger/larger datasets such as (tiny-)ImageNet or apply their approach on domain-shift scenarios? This is the main limitation of the paper.

Could the authors share some insights on the computational overhead of their approach? While PVC achieved SOTA results, the improvements seem rather marginal. It's important to know how the relative gains fare with the computational needs of the approach.

### Soundness
3

### Presentation
3

### Contribution
2
