# Thinking in Groups: Permutation Tests Reveal Near-Out-of-Distribution

- Decision: Reject
- Scores: 4, 2, 4, 0, 2

## Abstract
Deep neural networks (DNNs) have the potential to power many biomedical workflows, but training them on truly representative, IID datasets is often infeasible. Most models instead rely on biased or incomplete data, making them prone to out-of-distribution (OoD) inputs that closely resemble in-distribution samples. Such near-OoD cases are harder to detect than standard OOD benchmarks and can cause unreliable—even catastrophic—predictions. Biomedical assays, however, offer a unique opportunity: they often generate multiple correlated measurements per specimen through biological or technical replicates. Exploiting this insight, we introduce Homogeneous OoD (HOoD), a novel OoD detection framework for correlated data. HOoD projects groups of correlated measurements through a trained model and uses permutation-based hypothesis tests to compare them with known subpopulations. Each test yields an interpretable p-value, quantifying how well a group matches a subpopulation. By aggregating these p-values, HOoD reliably identifies OoD groups. In evaluations, HOoD consistently outperforms point-wise and ensemble-based OoD detectors, demonstrating its promise for robust real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a permutation-based hypothesis testing framework for out-of-distribution (OoD) detection. The key idea is that near-OoD cases are much harder to detect than far-OoD cases. To address this challenge, the method leverages permutation tests on latent representations extracted from trained models (e.g., ResNet, Autoencoder, or hybrid architectures).
Instead of evaluating OoDness at the individual sample level, the approach performs multi-response permutation tests on batches of homogeneous samples to assess whether the new batch is exchangeable with any known in-distribution class. This statistical formulation provides a principled, non-parametric test of distributional equivalence, independent of model assumptions. In biomedical assays, where multiple correlated measurements are available per batch, this batch-wise approach can effectively exploit intra-batch dependencies to improve OoD detection sensitivity.

### Strengths
1.	The use of permutation-based hypothesis testing is theoretically well-grounded and naturally consistent with the assumption of a homogeneous batch, where all samples are expected to be drawn from the same underlying distribution. 
2.	The proposed method is particularly well-suited for biomedical assays, where samples are naturally organized into groups (multiple cells from the same specimen or repeated measurements from the same assay).
3.	Unlike most existing approaches, the method focuses on batch-level OoD decisions rather than point-wise detection, a perspective that has received limited attention in the current literature despite its practical relevance in correlated data settings.
4.	The method works on latent features extracted from any trained model making it architecture-agnostic.
5.	The method provides greater interpretability than black-box OoD detectors, as its p-values offer a clear statistical measure of evidence for or against distributional equivalence.

### Weaknesses
1.	The main benchmarks (MNIST and CIFAR-10) are too simple to convincingly demonstrate the superiority or generality of the proposed method.
2.	MRPP is an established statistical method; the novelty of the paper lies primarily in its application to OoD detection within biological assays. The theoretical innovation is therefore modest, focusing more on application than on methodological advancement.
3.	The method is computationally demanding, which may limit its scalability to large datasets or scenarios involving hundreds of classes.
4.	The strong assumption of batch homogeneity may limit robustness, as performance can deteriorate with mixed or heterogeneous batches.

### Questions
The experimental protocol raises a number of questions that warrant further clarification :

1.	The datasets used in the experiments (MNIST, CIFAR-10, and AMRB) are relatively simple, even for point-wise OoD detection. While CIFAR-10 is commonly used, state-of-the-art OoD evaluations often include more challenging benchmarks such as CIFAR-100 or ImageNet. To convincingly demonstrate the method’s scalability and generality, the authors should provide results on at least one larger and more complex dataset. 

2.	The authors note that a single specimen often produces multiple replicates, resulting in highly correlated samples. To better mimic this situation, it would have been interesting to generate batches from a small number of original images, each subjected to data augmentation, rather than using a large number of completely different images. In the current experimental protocol, while all samples within a batch belong to the same class, they are not necessarily correlated, which may provide less realistic intra-batch information compared to batches of augmented replicates.

3.	The batch size is likely an important factor for a batch-wise method; however, in the experiments it appears to be fixed at 100. It would be valuable to discuss how the choice of sample size affects the method’s performance, as well as the broader influence of parameters $N$ (number of reference samples / class) and $M$ (number of samples to be tested) on the results and robustness.

4.	The benchmark involves three categories of methods: permutation-based tests, conformal prediction approaches, and point-wise methods with batch-level aggregation of individual scores. This setup is appropriate and covers the main methodological families for OoD detection. However, the point-wise baseline relies solely on MSP (Maximum Softmax Probability), which is a relatively simple and often outperformed method.I suggest including more competitive and widely used OoD baselines, such as ODIN (Liang et al., 2018), Mahalanobis distance (Lee et al., 2018), or energy-based scores (Liu et al., 2020), to strengthen the experimental comparison and better position the proposed method within the current literature.


5.	The proposed approach relies on the assumption that batches are homogeneous, i.e., that all samples within a batch originate from the same underlying distribution or specimen. This is a strong assumption, which may not always hold in practical scenarios. It would be helpful if the authors could either provide a stronger justification for this assumption—for instance, by relating it to biological settings where multiple replicates of the same specimen are indeed expected (see also point 3)—or include an experimental analysis showing how the method behaves under partially heterogeneous batches. Such results would clarify the robustness and applicability of the approach beyond idealized conditions.

6.	An analysis of the computational complexity and scalability of the method is missing. Since permutation-based tests typically require a large number of randomized assignments, the approach may become computationally expensive as the number of samples or reference classes increases. A discussion or empirical evaluation of runtime, memory requirements, and potential strategies for scaling (e.g., subsampling, ...) would strengthen the paper and clarify its feasibility for large-scale datasets or real-world applications.

The paper proposes an interesting and well-founded approach using permutation-based hypothesis testing for OoD detection. The idea is particularly relevant for biomedical applications, where data often come in homogeneous batches or correlated replicates. The statistical grounding of the method enhances interpretability compared to black-box detectors. However, the paper would benefit from stronger experimental validation, including more complex datasets and competitive baselines. A discussion of scalability and sensitivity to parameters would also strengthen the practical impact. Overall, the work is promising but requires more empirical depth.

### Soundness
2

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
The paper try to formulate a near OOD detection approach.

### Strengths
I find no strength in the paper.

### Weaknesses
1. Clarity and structure: The paper is poorly organized and difficult to follow. The mathematical notation and arguments are inconsistent and at times incorrect. The Introduction lacks focus, it starts by seemingly addressing transfer learning, then diverges into loosely connected real-world examples that do not align with the stated problem or the rest of the paper.

2. Ambiguity in exposition: The paragraph on Page 2, Lines 57–65 is incomprehensible. The authors should restate it with precise definitions, equations, and motivation. As written, it is not possible to understand the intended meaning.

3. Lack of definition for “near-OoD”: The term “near out-of-distribution” appears central to the paper, but there is no rigorous or formal definition. It remains unclear how near-OoD differs quantitatively from standard OoD or domain shift.

4. Definition 1 unclear: Definition 1 introduces symbols without context. The random variable, its domain, dimensionality, and distributional assumptions are all unspecified. The reader cannot tell what this definition contributes to the method or theory.

5. Scalability concerns: Permutation tests are well known to be computationally expensive. The paper does not discuss how the proposed approach would scale to realistic inference scenarios involving millions of samples, nor are there any approximations or efficiency analyses.

6. Undefined classifier (Section 2.1): The symbol M appears without prior definition. It is unclear whether this refers to a base classifier, a meta-model, or an embedding extractor.

Overall, this submission appears premature for peer review. The writing is confusing, the problem formulation lacks rigor, and several key definitions and symbols are missing. As a result, it is not possible to assess the correctness, novelty, or practical relevance of the proposed method. I encourage the authors to substantially revise the paper for clarity, provide formal definitions and scalable implementations, and ensure consistency between the motivation and methodology before resubmission to a major venue.

### Questions
See my above points.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper primarily aims to investigate whether groups of correlated test inputs (biological/technical replicates) can be used to detect near-out-of-distribution (OoD) shifts more reliably than standard point-wise methods, by testing their exchangeability with in-distribution subpopulations via permutation tests.

### Strengths
1. **[Important] An important problem setting.** The paper formalises an overlooked setting where test inputs arrive as *homogeneous* groups (same subpopulation) and recasts OoD detection as K two-sample hypothesis tests between the test group and K in-distribution reference groups. This is model-agnostic to the latent response used.

2. **Competitive empirical evidence.** Across MNIST, CIFAR-10, and a bacteria single-cell dataset (AMRB), HOoD with MRPP/LSP achieves high AUC on batch classification, outperforming MSP, CPP and CP in many cases.

### Weaknesses
1. **[Important] Dependence on the homogeneity prior at test time.** The method seems to assume each test batch is homogeneous (same subpopulation). Real deployments may see mixed batches; the paper does not quantify failure modes when homogeneity is violated.
2. **[Important] Limited evaluation breadth.** Only MNIST, CIFAR-10 and AMRB seem to be used; AMRB is domain-relevant but narrow (5 species, 21 strains). No testing on larger image or non-image biomedical cohorts, and no distribution shift types beyond label-exclusion.
3. **Limited coverage of computational cost and scalability.** Experiments use 3,000 permutations and sample size = 100 per batch; complexity with larger K, longer latents, or bigger batches is not benchmarked. (Table 1 caption.)
4. **Ambiguity on batch construction.** How many replicates are needed, and how sensitive are results to M (batch size)? Beyond the single setting (M≈100), guidance is insufficient.

### Questions
Refer to the "Weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This submission introduces a method of out-of-distribution (OoD)
sample detection in multiclass classifiers. More precisely, it defines
a special case (Homogeneous OoD) where each sample is formed by
several subsamples (e.g. image patches), and proposes to use a
two-sample test comparing this set of subsamples to sets of validation
subsamples from each class.

### Strengths
I am not expert enough in OoD be sure but it is possible that the homogeneous OoD problem defined in this submission was indeed overlooked so far.

### Weaknesses
My main concern is on clarity. The writing and definitions are often
vague, making it sometimes difficult to follow. For example:

- Definition 1 introduces what a homogeneous sample but simply requires
that there exists a mapping between subsamples and groups which
assigns all subsamples to the same group. I don't see in which
situation such a mapping could not be defined.

-The manuscript often refers to latent responses, e.g. "using latent
responses Z(xi ; ϕ) : RD → RL where Z = [Z1 , . . . , ZL ]." but never
defines what they are supposed to be (and in this precise quote, we do
not know what L is supposed to be either and it is never used later).

- The D_I and D_O sets (l. 201) are introduced but then never used
either in the method.

- In the caption of Figure 1, it is said that "k<<K" and I couldn't
  understand why (I don't think this is discussed anywhere).

The current exposition of the method itself is very short (less than
half a page), and I suggest that the authors expand it a little to
improve clarity.

Finally, I have some concern about the significance of the
methodological contribution itself, which is probably too straightforward to justify a
publication in a machine learning conference.

### Questions
The proposed test seems to exploit a single validation
sample for each class, shouldn't we be concerned about the variance
of the result (with respect to the choice of this sample)?

In addition, the decision is taken by thresholding the maximum p-value
over all classes, which doesn't account for the number of
classes. Wouldn't problems with more classes mechanically be more
liberal in their OoD detection?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose an out-of-distribution (OoD) detection framework designed for correlated data. The approach employs permutation-based hypothesis testing to compare unknown samples with known in-distribution (InD) subpopulations. Given a trained model, homogeneous reference samples from InD subpopulations, and an unknown sample, the method performs two-sample hypothesis testing with permutation tests to assess the exchangeability of the unknown sample with each of the reference samples. The proposed method is evaluated on MNIST, CIFAR-10, and the AMRB domain dataset.

### Strengths
This article focuses on the Near-OoD detection problem, which represents a challenging and important area of research.

### Weaknesses
No meaningful baseline comparisons are provided. The writing and presentation could be further improved to enhance clarity and readability. Some concepts are misused, and the use of basic machine learning terminology should be aligned with standard conventions.

### Questions
- Could the authors elaborate on the connection between homogeneity and correlation?
- The method is claimed to be designed for correlated data; however, the application cases do not clearly reflect this scenario. Could the authors explain how these applications involve correlated data and how the proposed method addresses correlation in practice?

### Soundness
2

### Presentation
1

### Contribution
2
