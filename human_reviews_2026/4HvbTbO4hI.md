# A Unifying View of Vector, Product and Scalar Quantization: An Information-Theoretic Perspective

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Discrete visual tokenization, predominantly driven by vector, scalar, and product quantization, lacks a unifying conceptual framework that elucidates the impact and tradeoffs of different quantization optimization objectives. In this paper, we propose a unified information-theoretic framework to shed light on these considerations. To do so, we view quantization as information compression and define the information loss (quantization error), compression ratio, and input/output as information-theoretic quantities. Using this framework, we resolve three central open questions: First, we theoretically prove and empirically demonstrate that minimizing quantization error, rather than maximizing codebook utilization, is the paramount optimization objective for ensuring training stability and reconstruction fidelity. Second, we establish two critical fairness conditions for intrinsic algorithm comparison: controlling the latent feature distribution variance and ensuring identical compression ratios. Third, we demonstrate, both theoretically and empirically, that under these conditions, modern vector quantization outperforms scalar and product quantization at minimizing quantization error. Our work provides a foundational reframing of quantization algorithms, resolving conceptual ambiguities and providing the first artifact-free comparison that establishes quantization error minimization as the core optimization criterion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
For discrete autoencoders, this paper demonstrates  that  that minimizing quantization error (not maximizing codebook utilization) is key for stability and fidelity, establishing fairness conditions for algorithm comparison, and demonstrating vector quantization's superiority in minimizing error under these conditions.

### Strengths
The paper is well-written, with clear and accessible content. I think its primary contribution lies in Proposition 1, which points out that minimizing quantization error can maximize codebook utilization (a mainstream approach currently proposed based on intuition and experiments), while the latter cannot guarantee the former.  The paper should develop its analysis and experiments on this aspect. Unfortunately, the authors shifts its focus to theoretical analyses of three quantization methods, which clearly lack innovation as detailed later.

### Weaknesses
1）The title of the paper is overly grandiose and inconsistent with its research content. The paper primarily investigates the impact of three quantization methods for discrete autoencoders. However, the title gives the impression that the paper presents a brand new theoretical analysis method for quantization.

2）The quantization analysis  based on minimizing error (presented in Section 4.4) is a classic information-theoretic approach, and the paper's method and results lack  innovation. Drawn from these results, the conclusion that VQ outperforms the other two methods,  lacks rigor and is very likely to be incorrect for two reasons: first, quantization error is closely related to the actual distribution of the data, which the paper does not study; second, the theoretical bounds provided in the paper are not tight, making performance comparisons based on them unreasonable.

### Questions
My  major concerns are given above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a unified information-theoretic framework for analyzing and comparing Vector Quantization (VQ), Product Quantization (PQ), and Scalar Quantization (SQ). By viewing quantization as an information compression process, it defines key quantities such as information loss (quantization error) and compression ratio, and derives theoretical and empirical conclusions regarding their relationships. The authors demonstrate that minimizing quantization error should be the primary optimization objective. They further establish fairness conditions for comparing different quantization schemes and validate their framework through controlled experiments using the VQ-Transplant model.

### Strengths
1. The unified benchmark and consistent architectural setup for comparing VQ, SQ, and PQ is a practical contribution that clarifies prior inconsistencies.
2. The conclusion that minimizing quantization error (rather than maximizing codebook utilization) is more critical for reconstruction and generation quality is insightful.

### Weaknesses
Some definitions and experimental details are insufficiently explained (see Questions).

### Questions
1. Definition 3 quantifies compression ratio Qr as the ratio between the input and output information quantities, derived from spatial and codebook dimensions. However, from an information-theoretic perspective—as the authors themselves claim—the compression ratio should ideally be defined in terms of entropy rather than quantity counts. Two signals with the same spatial dimensions can differ substantially in entropy. Could the authors justify whether their definition of Qr is reasonable, or discuss an entropy-based measure?
2. Figure 2 shows a linear relationship between quantization error and latent distribution variance. Which quantization scheme (VQ, PQ, or SQ) is used for this analysis? Does this linearity hold consistently across different quantization methods?
3. The experiments employ a pre-trained VQ-oriented model, with subsequent substitution of its quantization module for PQ and SQ. Could this adaptation lead to suboptimal performance for the latter methods due to the encoder-decoder’s alignment with the VQ latent space? Have the authors considered retraining the encoder to avoid this potential bias?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a unifying information-theoretic framework to analyze and compare various quantization methods (VQ, PQ, SQ) used in discrete visual tokenization. The core contributions are threefold: 1) It theoretically and empirically argues that minimizing quantization error (information loss) is a more critical optimization objective than maximizing codebook utilization. 2) It establishes two essential "fairness conditions" for comparing quantization algorithms: identical latent distributions and constant compression ratios. 3) Under these fair conditions, it demonstrates the intrinsic superiority of modern VQ methods over PQ and SQ. The claims are supported by well-designed experiments using a "VQ-Transplant" framework.

### Strengths
1.The proposed information-theoretic framework provides a principled and clear perspective to understand the fundamental trade-offs in quantization, demystifying the relationship between quantization error, codebook utilization, and reconstruction quality.
2.The introduction of two "fairness conditions" is a significant contribution. It addresses a critical gap in prior work where comparisons were often confounded by architectural or training differences. This sets a higher standard for future research in this area.
3.The "VQ-Transplant" experimental design is clever, effectively isolating the performance of the quantization module itself. The strong correlation found between quantization error and reconstruction fidelity (r-FID) provides convincing evidence for the paper's main claim.

### Weaknesses
The paper frames its analysis in information theory, defining quantities based on bit counts and using squared Euclidean distance as "information loss." While Proposition 1 insightfully connects codebook utilization to conditional entropy, the main empirical metric remains MSE. The paper could better articulate the connection between minimizing the theoretical information loss (e.g., H(X|Z)) and the practical objective of minimizing MSE. Is the framework a fundamental new theoretical lens, or primarily a useful reframing of established concepts with information-theoretic terminology?

### Questions
refer to weakness

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
This paper presents a unified information-theoretic framework for analyzing vector, product, and scalar quantization (VQ, PQ, SQ). By viewing quantization as an information compression process, the authors formally define quantities such as information loss, compression ratio, and information capacity. The paper proves that minimizing quantization error (information loss) is a more fundamental optimization objective than maximizing codebook utilization, and introduces two fairness conditions for comparing quantization algorithms: (1) identical latent feature distributions and (2) identical compression ratios. Under these conditions, both theoretical and empirical analyses demonstrate that VQ outperforms PQ and SQ in minimizing information loss.

### Strengths
The proposed information-theoretic formulation provides a rigorous, unifying perspective on quantization algorithms.

The authors formally prove that minimizing quantization error implies full codebook utilization (but not vice versa), and derive scaling laws for optimal VQ/PQ/SQ errors.

The two fairness conditions (controlled latent distributions and compression ratios) are well-motivated and improve reproducibility and validity of empirical evaluations.

### Weaknesses
While the framework is elegant, it mostly systematizes existing methods rather than introducing fundamentally new algorithms.

Some notations (e.g., Q_i,Q_o,Q_r) and assumptions could be clarified for broader readability.

Limited discussion on downstream effects or interpretability benefits for generative models

The paper is not well-organized and difficult to follow, even for readers familiar with quantization and visual representation learning. The presentation contains an excessive number of equations and derivations, while the key insights and conclusions are often buried or insufficiently highlighted.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
