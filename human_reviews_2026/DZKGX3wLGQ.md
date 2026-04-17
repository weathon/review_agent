# HyperINR: Ensuring Semantics in Weights with Implicit Function Theorem

- Decision: Reject
- Scores: 0, 2, 6, 4

## Abstract
Implicit Neural Representations (INRs) have demonstrated remarkable capability in representing 2D and 3D data, whose semantics is convincingly captured in the weights of the corresponding neural network. Despite successes in applying INRs to various applications, a precise theoretical explanation for the mechanism of encoding semantics of data into network weights is still missing. 
In this work, we propose a jointly trainable HyperINR model, which learns a hypernetwork to map a learnable low-dimensional latent space to the weight space of an INR.
By employing the classic Implicit Function Theorem, the HyperINR is shown to be a mathematically rigorous framework that ensures the mapping of the semantics of data to the weight latent representations. Extensive experiments of classification tasks on 2D and 3D data confirm the effectiveness of our approach and demonstrate superior performance compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces HyperINR, a method for generating the weights of implicit neural representations (INRs). HyperINR takes an autodecoding approach by assigning each signal a learnable latent code, which is then fed to the hypernetwork to generate the weights of the INR. The latent code is learned during training or inference. A theoretical analysis of HyperINR using the implicit function theorem is also presented. HyperINR is evaluated on fitting INRs for MNIST, FashionMNIST, ModelNet40, ShapeNet10, and ScanNet10 as well as the using the learned representations downstream for classification.

### Strengths
This paper proposes a new method, HyperINR, and some mathematical analysis.

### Weaknesses
**Major Weaknesses**

**(W1) Contribution/novelty**: The novelty of HyperINR is low as there are no significant methodological innovations presented in the paper. The use of hypernetworks for generating INRs has been studied in many previous works, and similarly for autodecoding. The proposed method seems to be very similar to the hypernetwork of SIREN, except using autodecoding. The contribution of the theoretical section is also not clear because the theory is not used to propose any new innovations such as loss functions or a different architecture and its purpose is not clear. 

**(W2) Related works**: Most of the “hypernetworks for INR generation” literature is not cited or discussed. Example works include [1-5]. Furthermore, the section on equivariance and symmetry of INR weights is completely superfluous as there is nothing in the rest of the paper concerning equivariance or symmetry.

**(W3) Baselines**: Related to the above weaknesses, all previous hypernetwork methods are missing as baselines. Furthermore, other methods such as Functa are missing entirely, and INR2Array is missing from Table 2, whereas inr2vec is missing from Table 1. 

**(W4) Empirical evaluation**: The empirical evaluation of this method is also very poor. In addition to missing baselines, discussed above, the quality of the learned INRs is not adequately compared to that of the baseline methods. There are also no qualitative comparisons. On all datasets, the main comparison is only downstream classification accuracy. 

Furthermore, as described below, the accuracies of baseline methods are misreported. With the correct accuracies, HyperINR performs significantly below the inr2vec baseline on classification accuracy for 3D data (Table 2). 

**(W5) Lack of ablation study**: While this method only considers small models, there is no study of how the model scales or how the HyperINR compares at parameter counts similar to that of other methods.

**(W6) Unjustified claims, missing citations, or wrong data**: Many key claims are not justified by either empirical evidence presented in the paper or its supplementary material or by citation of previous works, or the cited claims or data is incorrect. These include:
1. “Achieving clean and unambiguous clusters in weight space remains difficult for previous work” (Line 362)
2. “We report a single representative run without variance because our method is not affected by the typical initialization sensitivity of INR fitting” (Lines 374-376)
3. “While the errors on MNIST and FashionMNIST are slightly higher, they remain lower than those reported in prior work…” (Lines 456-459)
4. In Table 2, the reported result of inr2vec for top-1 classification accuracy on ModelNet40 in this paper is 81.7, but in the original paper, this is the reported result for shape retrieval, not top-1 classification accuracy. The actual reported performance of inr2vec on top-1 classification is 87.0%, significantly better than the reported result for HyperINR. A similar thing happens for ShapeNet10 (reported in this paper 90.6%, actual reported result in the original paper 93.3%) and ScanNet10 (reported in this paper 65.2%, actual reported result in the original paper 72.1%). 

**(W7) Limitations**: As an autodecoding method, this method requires test-time optimization. 

**Minor Weaknesses**
There may have been an error in the formatting of the Tables. 

[1] Chen, Yinbo, and Xiaolong Wang. "Transformers as meta-learners for implicit neural representations." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[2] Kim, Chiheon, et al. "Generalizable implicit neural representations via instance pattern composers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[3] Gu, Jeffrey, Kuan-Chieh Wang, and Serena Yeung. "Generalizable neural fields as partially observed neural processes." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[4] Lee, Doyup, et al. "Locality-aware generalizable implicit neural representation." Advances in Neural Information Processing Systems 36 (2023): 48363-48381.

[5] Gu, Jeffrey, and Serena Yeung-Levy. "Foundation models secretly understand neural network weights: Enhancing hypernetwork architectures with foundation models." arXiv preprint arXiv:2503.00838 (2025).

### Questions
**(Q1)**: What are the ways the theoretical analysis in the paper could be used to improve INR methods?

**(Q2)**: How does HyperINR scale with increasing size?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose HyperINR, a framework for learning latent embeddings of 2D and 3D data that aims to theoretically ensure the embeddings capture the semantic structure of the data. It comprises a hypernetwork and a main implicit neural representation (INR) network: for a given data sample x, the hypernetwork maps the latent embedding of x to the weights of the INR network, which then maps a coordinate to the pixel value. During training, both the hypernetwork and latent embeddings are jointly optimized to minimize reconstruction loss. During the test, only the embeddings for the test set are optimized with the fixed hypernetwork using the same loss. The authors claim that the latent embeddings capture the semantics of the data, showing that a necessary condition for the loss Hessian to be full rank is satisfied. Experiments on classification tasks using latent embeddings show that HyperINR achieves performance comparable to or better than baseline methods.

### Strengths
S1. The proposed method achieves comparable or superior accuracy compared to the baselines.

S2. The paper provides a clear description of the model architecture and training process, with diagrams that improve readability and understanding.

### Weaknesses
W1. The theoretical claims and motivations should be reinforced.

W1.1. The main contribution claimed by the authors is that the proposed method theoretically ensures the embeddings capture the semantics of the data, showing that a necessary condition for the loss Hessian to be full rank is satisfied. However, this claim has several conceptual issues. First, the notion of “semantics of the data” (as well as “local” and “global” semantics) is vague and lacks a formal definition or measurable criterion. Second, even if the loss Hessian were full rank, this does not necessarily imply that the learned embeddings encode semantic or meaningful structures in the data; the connection between the two should be more elaborated. Third, the paper only demonstrates that a necessary condition for the Hessian to be full rank holds, which is insufficient to conclude that the Hessian itself is indeed full rank.

W1.2. The authors point out that existing methods do not guarantee that the same data sample will converge to the same weights, presenting this as a limitation. However, the paper does not clearly explain why this property is problematic in practice or how it affects the learned representations. Moreover, the authors do not provide theoretical or empirical evidence that their proposed method actually ensures such convergence. The authors should better elaborate on the motivation and clarify its practical and theoretical implications.

W2. The experiments should be reinforced.

W2.1. The authors are encouraged to include a state-of-the-art embedding method [1] as a competitor. Note that it achieved higher classification accuracy than HyperINR for the same datasets. In addition, contrary to the authors' claim that labels are integrated into the representation in [1], labels are not used to learn the representation for the test set.

W2.2. The explanation for reporting only a single representative run is not convincing. The authors do not provide sufficient evidence that the proposed method is indeed insensitive to initialization. Moreover, even if this claim were correct, conducting multiple runs would strengthen the paper by substantiating the claim and demonstrating the robustness of the results. The authors are encouraged to include such evaluations, as also related to the issue discussed in W1.2.

W2.3. Reporting the training and inference time as well as memory requirements for all datasets, rather than for a single one, would make the experimental results more convincing and comprehensive.

W3. Considering W1 and W2, the overall contributions remain limited, since the proposed method does not convincingly resolve the issues it aims to tackle and fails to show empirical superiority over a state-of-the-art method.

[1] A. Gielisse and J. van Gemert. "End-to-End Implicit Neural Representations for Classification." CVPR’25

### Questions
Please refer to W1-W3.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes HyperINR, a theoretically grounded framework for implicit neural representations (INRs) that connects data semantics to network weights through the Implicit Function Theorem (IFT). A shared hypernetwork maps low-dimensional latent vectors to INR weights, and the IFT analysis is used to guarantee a local one-to-one mapping between the data space and the latent (weight) space, theoretically ensuring that semantics are preserved in INR weights. Empirical evaluations on 2D and 3D datasets demonstrate meaningful clustering and competitive classification performance using compact architectures and minimal supervision.

### Strengths
**Strong theoretical motivation** 
The IFT-based analysis provides a rigorous mathematical argument for how data semantics can be encoded in weight space, offering a new perspective on “semantics-in-weights.” By grounding semantics preservation in the IFT framework, the work connects classical mathematical theory with modern INR modeling.


**Compact formulation**
The hypernetwork–INR framework is conceptually simple, jointly optimizes latent embeddings and weights, and avoids complex multi-stage training used in prior latent-INR approaches.

### Weaknesses
While the IFT analysis is elegant, the link to equivariance and symmetry in the data and weight spaces is not clearly demonstrated. Logical or empirical evidence showing that the proposed mapping preserves or improves symmetry properties would strengthen the theoretical claim.

The empirical section nicely illustrates clustering and lightweight classification, but it remains limited in scope. I strongly recommend:

- Empirical verification of IFT conditions,  e.g., evaluating the Jacobian/Hessian rank, condition number, and local continuity/uniqueness under perturbations.
- Capacity- and compute-matched baselines (e.g., DWS, NFN, inr2vec, Functa) to ensure fair comparison.
- Scaling to more challenging datasets and reporting rigorous reconstruction metrics (Chamfer, IoU, LPIPS, etc.).
- Comprehensive ablations on latent dimensionality, coordinate sampling density, and hypernetwork capacity.
- Symmetry/equivariance probes,  e.g., rigid transformations in data space and hidden-unit permutations in weight space,  to clarify the connection to the related work on equivariant INR representations.

These extensions would substantially reinforce the experimental evidence for the paper’s theoretical claims and enhance its empirical credibility.

### Questions
Your IFT analysis establishes a local one-to-one mapping z=g(X) between data and the latent (weight) space under full-rank conditions, ensuring that local semantics are preserved. However, this by itself does not imply equivariance in data space (e.g., under rigid transformations or reparameterizations) nor does it address weight-space symmetries (e.g., neuron permutation invariance).

- Does HyperINR prove or enforce any group-equivariance of g with respect to transformations on X?
- How does your framework handle weight permutation symmetries in the INR (i.e., identifiability up to symmetry)?

If such properties are not explicitly enforced, what modifications, such as an equivariant hypernetwork, symmetry-invariant regularization, canonicalization of INR weights, or latent regularization to produce predictable transformations under known data symmetries, could extend HyperINR to provide these guarantees?

Could you include experiments applying known data-space transformations (e.g., rotations or translations) and examine whether the induced changes in the latent z vectors follow a consistent, structure-preserving mapping? This would empirically validate whether the theoretical local bijection also preserves symmetry relationships in practice.

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
5

### Summary
The paper presents HyperINR, a model that leverages Implicit Neural Representations (INRs) for data classification tasks by ensuring the semantics of the data are embedded in the neural network weights. The core idea is to utilize a hypernetwork that maps low-dimensional latent vectors into the weight space of an INR model, creating a more structured way to encode data semantics. The experimental setup focuses on classification tasks, using both 2D image datasets (MNIST, FashionMNIST) and 3D shape datasets (ModelNet40, ShapeNet10, ScanNet10). The results show that HyperINR outperforms existing methods like DWS, NFN, and Inr2Array in terms of classification accuracy on multiple benchmarks.

### Strengths
- HyperINR outperforms or competes favorably with several state-of-the-art INR classification baselines like DWS, NFN, and Inr2Array on both 2D and 3D datasets.

- The paper includes a compelling demonstration of smooth interpolations in latent space, showing that HyperINR generates continuous and semantically meaningful transformations between data points.

### Weaknesses
- I don't really see the role of IFT in this paper. You can have some smooth mapping between the local neighborhood of X and Z. This is natural. But I do not see how it is related to better classification.

- I wonder how significant this work is. As a method jointly trains a conditional INR model for a collection of data, it is essentially another setup of latent-coded INR, but the way of latent code is through a hypernetwork to form the INR weights. I don't see really foundational difference from concatenating latent code to the input (audo-decoder in DeepSDF) or modulation in functa. If the claim is that hypernetwork is a more expressive way then there should be established baselines. 

- Missing baseline of yet another way to jointly train INRs (with MAML meta-learning): End-to-End Implicit Neural Representations for Classification (CVPR 2025)

- Also using this way and jointly train INRs on the whole dataset it can align the weight space dimensions, then it is natural better than weight-space equivariant architectures (e.g. DWSNet, NFN) that is designed for separately trained INRs. I wonder whether comparison with these methods is fair. And the classifier is even not applied on the weight space.

### Questions
Please clarify the role of IFT in the paper. And how do you define "semantics".

And see my other concerns in the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
