# Error Feedback for Muon and Friends

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Recent optimizers like Muon, Scion, and Gluon have pushed the frontier of large-scale deep learning by exploiting layer-wise linear minimization oracles (LMOs) over non-Euclidean norm balls, capturing neural network structure in ways traditional algorithms cannot. Yet, no principled distributed framework exists for these methods, and communication bottlenecks remain unaddressed. The very few distributed variants are heuristic, with no convergence guarantees in sight. We introduce EF21-Muon, the first communication-efficient, non-Euclidean LMO-based optimizer with rigorous convergence guarantees. EF21-Muon supports stochastic gradients, momentum, and bidirectional compression with error feedback–marking the first extension of error feedback beyond the Euclidean setting. It recovers Muon/Scion when compression is off and specific norms are chosen, providing the first efficient distributed implementation of this powerful family. Our theory covers non-Euclidean smooth and the more general $(L_0, L_1)$–smooth setting, matching best-known Euclidean rates and enabling faster convergence under suitable norm choices. We further extend the analysis to layer-wise (generalized) smoothness regimes, capturing the anisotropic structure of deep networks. Experiments on NanoGPT benchmarking EF21-Muon against uncompressed Muon/Scion/Gluon demonstrate up to 7× communication savings with no accuracy degradation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes EF21-Muon, which aims to be a communication-efficient distributed learning framework for the Muon family of optimizers. A key insight is to reformulate the LMO update using the sharp operator which recasts the update as a normalized steepest descent step. As Muon is steadily gaining popularity, this area of research is particularly timely and well-placed. Additionally, it is important to provide a principles framework for Muon, including in the setting of distributed training which suffers from communication inefficiencies.

### Strengths
1. The topic is particularly timely, and addresses the constraints of distributed training which is not typically encountered in centralized settings. 

2. The experiment setup is quite relevant. I believe that NanoGPT on FineWeb is a good choice for optimizer evaluation in the context of model LMs.

### Weaknesses
1. Not much of a weakness, but the authors argue that due to increasing size, all training is distributed. This is true, but most frameworks tend to be parallelism (pipeline, model, etc) rather than the FL-type. 

2. Experiments are conducted on a single model/scale (NanoGPT 124M). Demonstrating the effectiveness of EF21-Muon on other domains (e.g., ViT) or at a larger scale would make the claims of general applicability significantly more robust.

### Questions
Please see weaknesses. Not much questions to add.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents EF21-Muon, a communication-efficient distributed optimizer based on recent LMO optimizers with rigorous convergence guarantees. This method achieves up to 7x communications savings compared to uncompressed baselines.

Furthermore, a few notable insights include:
1. Bidirectional compression with error feedback
2. Support for non-Euclidean geometries through arbitrary norm choices
3. Layer-wise treatment of neural network parameters
4. Theoretical guarantees under many settings

### Strengths
1. Significant theoretical contribution, rigorous convergence analysis for compressed, distributed LMO-based methods in non-Euclidean settings.
2. Addresses a real bottleneck in the distributed training of large models, where Muon is gaining popularity.
3. Comprehensive theory, a very long appendix with many proofs.

### Weaknesses
1. Only one model size was trained, and it is very small by modern standards (120M).
2. Only one evaluation (Loss) was used. It's good practice to include a few downstream evaluations just in case for bold claims.
3. Compression overhead is not clearly addressed, topk requires transmitting indices and not thoroughly analyzed for the distributed setting with varying model architectures. This is especially important in optimization as a general-purpose optimizer should work on different model shapes.
4. Very heavy and dense notation, difficult to read.
5. The algorithm itself is not especially novel, error feedback comes from EF21.

### Questions
1. How does this method scale when applied to bigger models? How does wall-clock time scale as you increase the model size? Is it a constant factor increase or is it quadratic, log, etc?
2. How does EF21-Muon compare to other compressed distributed training variants of AdamW or SGD?
3. The method proposed has a lot of hyperparameters, how sensitive is the training to choices beyond the limited ablations shown?

### Soundness
3

### Presentation
2

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
This work studies a distributed non-Euclidean LMO-based optimization method that generalizes and recovers Muon and Scion in the non-compressed regime. Specifically, the authors introduce EF21-Muon, a unified and communication-efficient algorithm that incorporates stochastic gradients, momentum, and bidirectional compression with error feedback, while encompassing several existing compressed methods as special cases. Furthermore, the authors present comprehensive convergence guarantees for multiple settings, including both deterministic and stochastic cases, as well as for non-Euclidean smooth and generalized non-Euclidean smooth functions, under both layer-wise and joint parameter treatments. Experimental results on nanoGPT under various worker compressors show significant communication savings for the proposed algorithm.

### Strengths
- This work provides rigorous and comprehensive convergence guarantees for a wide range of settings.
- It supports bidirectional compression for the smooth regime.
- The proposed algorithm further supports non-Euclidean contractive compressors, thus enhancing generality.
- The analysis of the algorithms is conducted in non-Euclidean norms.
- The proposed algorithms achieve the state-of-the-art convergence rates.
- The paper is clearly structured and easy to follow.
- The discussion of the contribution of each term in the convergence guarantees is insightful.

### Weaknesses
- The results in the generalized smoothness regime do not include primal compression.
- The experimental results could additionally include some other baseline algorithms for comparison to better contextualize performance of EF21-Muon.
- It may also have been beneficial to include wall-clock time experiments.

### Questions
- Can the authors elaborate a bit on why in the generalized smooth setup the convergence guarantees are established without primal compression? Can these results potentially be extended to that setting as well?
- Could the authors clarify how the combinations of compressors are applied in the experimental setup?

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
This paper introduces a distributed optimizer that brings error‑feedback (EF) to layer‑wise linear minimization oracle (LMO) methods such as Muon/Scion/Gluon, extending EF beyond the Euclidean setting and enabling bidirectional compression with support for stochastic gradients and momentum.

### Strengths
The paper establishes non‑Euclidean EF convergence for LMO‑based methods, and shows how rates recover Euclidean EF21 results when specialized, which is a nontrivial generalization.

On NanoGPT-124M trained on FineWeb, they report up to 7x reduction in worker‑to‑server communication without loss of accuracy.

### Weaknesses
The novelty is rather incremental since EF21-P introduces bidirectional EF with momentum/stochasticity, Gluon introduces non‑Euclidean layer‑wise LMO analyses, and Dion introduces distributed Muon‑style optimizers.

Experiments show that compression reduces uplink bytes but degrades token efficiency and final loss at a fixed 5B budget, so the claim of "no accuracy degradation" is unsupported under equal‑budget comparisons.

The convergence results assume exact LMO steps, yet the implementation uses inexact Newton–Schulz updates. Recent literature emphasizes that this gap matters, but the paper neither analyzes nor bounds the approximation error.

### Questions
Results are on NanoGPT‑124M, but how would the conclusions change for billion‑parameter LLMs and larger clusters?

The main results and plots focus on w2s, but how is the total communication cost including s2w, and what are the gains from s2w compression?

### Soundness
3

### Presentation
3

### Contribution
2
