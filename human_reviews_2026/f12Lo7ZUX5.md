# Reverse Distillation: Consistently Scaling Protein Language Model Representations

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 6

## Abstract
Unlike the predictable scaling laws in natural language processing and computer vision, protein language models (PLMs) scale poorly: for many tasks, models within the same family plateau or even decrease in performance, with mid-sized models often outperforming the largest in the family. We introduce Reverse Distillation a principled framework that decomposes large PLM representations into orthogonal subspaces guided by smaller models of the same family. The resulting embeddings have a nested, Matryoshka-style structure: the first $k$ dimensions of a larger model's embedding are exactly the representation from the smaller model. This ensures that larger reverse-distilled models consistently outperform smaller ones. A motivating intuition is that smaller models, constrained by capacity, preferentially encode broadly-shared protein features. Reverse distillation isolates these shared features and orthogonally extracts additional contributions from larger models, preventing interference between the two. On ProteinGym benchmarks, reverse-distilled ESM-2 variants outperform their respective baselines at the same embedding dimensionality, with the reverse-distilled 15 billion parameter model achieving the strongest performance. Our framework is generalizable to any model family where scaling challenges persist. Code and trained models are available at https://github.com/rohitsinghlab/plm_reverse_distillation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tries to investigate why larger Protein Language Models such as ESM-2 fail to exhibit the expected scaling gains.
The authors attribute this to the entanglement of general and specialized representations within large models, which increases the variance of linear probes.
To address the above issue, this paper proposes Reverse Distillation, a linear subspace decomposition method that uses smaller models to define a general feature subspace and extracts orthogonal residuals from larger models to represent specialized knowledge.
Extensive experiments on ProteinGym and BioMap show that RD consistently improves predictive performance.

### Strengths
1. This paper tries to tackle an important issue in large Protein Language Models (PLMs): the unexpected degradation of scaling behavior
2. The proposed Reverse Distillation method is computationally lightweight and purely linear, involving only least-squares fitting and SVD decomposition
3. Despite the method’s simplicity, RD exhibits stable and monotonic performance gains across multiple datasets and scaling levels.

### Weaknesses
1. The paper builds its entire motivation on the “general vs. specialized representation” hypothesis but does not provide a quantitative or qualitative analysis to validate it
2. The proposed Optimal Constrained Approximation theorem only guarantees minimal reconstruction error under a prefix constraint: a standard property of linear least squares combined with SVD. However, this result does not theoretically justify why the assumed decomposition is needed or effective.
3. Since RD performs a chain-wise representation enhancement, a straightforward baseline naturally arises: direct representation fusion across the same model chain. It remains unclear whether RD’s improvement comes from its “distillation mechanism” or simply from aggregating multi-scale features. A comparison against naive fusion, or fusion with simple KD objectives, is essential to establish the method’s actual contribution.
4. The experiments demonstrate improvement within the ESM-2 family, but it is uncertain whether the observed scaling restoration generalizes to other architectures.

### Questions
All my concerns about this paper are stated in the weakness section. Please refer to the weakness section for rebuttal/discussion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the well-known "counterintuitive scaling" problem in Protein Language Models (PLMs) like ESM-2, where larger models often perform worse than medium-sized models on downstream benchmarks.1 The authors hypothesize this is due to "feature entanglement," where larger models mix "universal" features (from small models) with "specialized" features, and this mixture acts as noise for standard linear probes.2 The authors propose "Reverse Distillation" (RD), a novel and elegant post-hoc framework. Instead of compressing, RD uses a smaller model's representation ($H_r$) as a basis and decomposes a larger model's representation ($H_p$) into an orthogonal combination $[H_r, H_{res}]$, where $H_{res}$ captures the new, orthogonal information from the larger model.2 The method is theoretically grounded (Theorem 1) 2 and empirically shown to restore monotonic scaling (i.e., the rd.3B model consistently beats the rd.650M model) on ProteinGym and BioMap benchmarks.2

### Strengths
1. The paper tackles a critical, well-documented problem (PLM scaling failure ) with a highly novel solution. The idea of using smaller models as a basis for post-hoc orthogonal decomposition is elegant and new.
2.  The experiments persuasively demonstrate that RD works. It not only improves baseline performance (e.g., rd.3B > 3B) but, more importantly, it restores monotonic scaling (rd.3B > rd.650M wins 96.4% of the time, vs. 53.6% for the baseline).
3.The BioMap experiment (Table 4) provides strong evidence for the "feature entanglement" hypothesis. RD specifically fixes the scaling failure on "universal" features (like secondary structure) that the paper hypothesized were "drowned out" in larger models.
4. The method is post-hoc, requiring no model retraining. The "Chained" version provides a practical, novel way to create Matryoshka-style nested embeddings from an existing model family.

### Weaknesses
The paper's primary weakness is the exclusion of the ESM-2 15B model.2 The most severe example of scaling failure is the performance degradation from 3B to 15B.2 The paper only demonstrates fixing the 650M-to-3B plateau. Without testing the 15B model, the central claim of "solving" the scaling paradox is incomplete.
The core idea of using orthogonal subspaces to separate/disentangle knowledge, while novel in this application, is conceptually similar to methods in continual learning (e.g., O-LoRA), which should be cited.

### Questions
see weakness

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper improves PLMs through a model distillation process which decomposes large models into smaller sub-models with disentangled residuals. The resulting embeddings also enjoy the Matryoshka property which allows for slices of embeddings to remain informative. These models also recover better scaling properties, allowing for more efficient parameter use. Benchmarking was done on ProteinGym and BioMap, showing good predictive performance as well as scaling with model size.

### Strengths
This paper is quite strong in my opinion. The proposed methods are a clear improvement on current approaches and is a valuable contribution to the protein representation field.

### Weaknesses
* Additional inference time though not prohibitive could still limit adoption.

### Questions
n/a

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses a significant problem in biological foundation models: they scale poorly compared to models in natural language processing. Specifically, larger Protein Language Models (PLMs) in families like ESM-2 often underperform smaller ones on key benchmarks, a phenomenon known as non-monotonic scaling. The authors hypothesize this is because small models capture "universal" features (like secondary structure), while larger models add "specialized" features (like protein-family specific functions). When these features are entangled in a single representation, the specialized features can act as noise, degrading performance on tasks that rely on universal patterns. To solve this, the paper introduces a method that decomposes large protein language model representations into orthogonal subspaces guided by smaller models. On benchmarks like ProteinGym and BioMap, the reverse-distilled ESM-2 models (e.g., rd.650M) broadly outperform their corresponding baselines (e.g., 650M).

### Strengths
1. The paper establishes the central problem: PLMs "scale relatively poorly" , with the ESM-2 family's performance plateauing. The authors' core hypothesis is highly intuitive that this is due to larger models "entangling" universal (low-level) and specialized (high-level) features, which increases variance.

2. This work introduces high-performing and efficient embeddings. The resulting models outperform baselines at the same size. They also feature a "Matryoshka-style" structure, which allows smaller prefixes of a single embedding to be used as valid, lower-dimensional representations, saving computation and storage.

3. The experimental design is comprehensive. The authors test their method on standard, challenging benchmarks, including ProteinGym DMS and BioMap. The inclusion of practical analyses, such as an ablation on the training data size and a measurement of inference overhead, further strengthens the work's quality.

### Weaknesses
1. The authors should at least attempt a reverse distillation of 3B $\rightarrow$ 15B (or the full chain up to 15B). This experiment is critical. If rd.15B outperforms rd.3B, the paper's core thesis is validated. If rd.15B still underperforms, it would suggest the scaling problem is more complex than just feature entanglement, fundamentally weakening the paper's conclusion.
2. This linear-only approach may be restrictive. The paper itself hypothesizes that larger models encode rarer, higher-order phenomena. These complex, higher-order features may not be neatly separable from the universal features via a simple linear projection. The authors' method might only be extracting the linearly predictable component, leaving a "residual" that is still a mix of true novel features and non-linear transformations of universal features.

### Questions
1. The paper hypothesizes that $H_{r}$ captures "universal" features and $H_{res}$ captures "specialized" ones. Beyond downstream task performance, did you conduct any qualitative analysis to verify this? For example, could you use feature attribution or probing to show that $H_{res}$ contains information about specific protein-family motifs or epistatic interactions that are demonstrably absent when probing $H_{r}$?

### Soundness
3

### Presentation
3

### Contribution
3
