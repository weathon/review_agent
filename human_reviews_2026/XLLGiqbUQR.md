# Lightweight MSA Design Advances Protein Folding From Evolutionary Embeddings

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Protein structure prediction often hinges on multiple sequence alignments (MSAs), which underperform on low-homology and orphan proteins. We introduce PLAME, a lightweight MSA design framework that leverages evolutionary embeddings from pretrained protein language models to generate MSAs that better support downstream folding. PLAME couples these embeddings with a conservation–diversity loss that balances agreement on conserved positions with coverage of plausible sequence variation. Beyond generation, we develop (i) an MSA selection strategy to filter high-quality candidates and (ii) a sequence-quality metric that is complementary to depth-based measures and predictive of folding gains.
On AlphaFold2 low-homology/orphan benchmarks, PLAME delivers state-of-the-art improvements in structure accuracy (e.g., lDDT/TM-score), with consistent gains when paired with AlphaFold3. Ablations isolate the benefits of the selection strategy, and case studies elucidate how MSA characteristics shape AlphaFold confidence and error modes. Finally, we show PLAME functions as a lightweight adapter, enabling ESMFold to approach AlphaFold2-level accuracy while retaining ESMFold-like inference speed. PLAME thus provides a practical path to high-quality folding for proteins lacking strong evolutionary neighbors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PLAME, a lightweight framework for synthetic multiple sequence alignment (MSA) generation to improve protein structure prediction when few natural homologs are available.
PLAME encodes each query sequence using ESM-2 embeddings and then autoregressively generates virtual MSAs in the embedding space.
The model is trained with a conservation–diversity loss that balances fidelity and entropy, and a filtering module, HiFiAD, selects high-fidelity alignments based on BLOSUM recovery and diversity thresholds.

The generated MSAs are fed into AlphaFold2 and AlphaFold3.
On curated low-homology benchmarks, the method shows moderate improvements in pLDDT, GDT, and TM-scores over baselines such as EvoDiff, MSAGPT, and DHR, while being computationally efficient.
The authors present PLAME as a lightweight adapter narrowing the gap between ESMFold and AF2-level accuracy.

### Strengths
1. Addresses an important limitation: lack of MSAs for low-homology proteins.

1. Introduces a clear conservation–diversity objective and a simple selection module (HiFiAD).

1. Evaluated on multiple structure predictors and standard benchmarks.

1. Offers practical computational efficiency compared with traditional MSA search.

1. Provides some transparency through ablations and limitations discussion.

### Weaknesses
1. Methodological risk — unverified AI-generated data (major).
   PLAME uses embeddings from a pre-trained model (ESM-2) as the only source of evolutionary signal.
   These are AI-generated, unvalidated representations but are treated as if they contained true biological information.
   This undermines methodological soundness and risks propagating training-set biases.

1. Experimental results show that PLAME does not always outperform baselines; in several targets, accuracy decreases.
   This suggests that the ESM-2 embeddings may introduce noise that confuses downstream folding rather than improving it.

1. The authors do not report co-evolutionary metrics (e.g., contact precision, MI) to confirm that synthetic MSAs carry meaningful structure information.

1. Missing confidence intervals, random-seed reporting, and code-release details.

1. Efficiency claims are based on a single example; full end-to-end runtime is not provided.

### Questions
1. How do the authors justify this model-on-model design from a biological perspective? Is it scientifically reliable to use unverified embeddings from one AI model as evolutionary data for another?

1. What evidence demonstrates that ESM-2 embeddings capture genuine evolutionary couplings rather than statistical priors learned from UniRef50?

1. Have you tried replacing ESM-2 with random or untrained protein language models to evaluate whether the observed gains arise from real biological information or simply from the modeling bias of ESM-2?

### Soundness
3

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
2

### Summary
Protein structure prediction heavily relies on evolutionary information from multiple sequence alignment (MSA). However, traditional MSA methods underperform in low-homology and orphan proteins due to insufficient evolutionary signals. This limitation restricts structural biology applications in drug design and functional studies, particularly when dealing with proteins lacking clear evolutionary neighbors. Current approaches face two critical challenges: First, supervised bias, methods trained on existing MSA databases tend to favor highly homologous families, making them unsuitable for low-homology and orphan proteins. Second, weak alignment-folding correlation, the lack of lightweight metrics to directly link MSA characteristics with folding outcomes results in generated targets that may not effectively improve structural accuracy. Additionally, high computational costs and limited generalizability hinder further optimization.
This study proposes a lightweight multiple sequence alignment (MSA) framework, PLAME, which leverages pre-trained protein language models with evolutionary embeddings to generate high-quality MSA. This approach significantly improves structural prediction accuracy for low-homology and orphan proteins.

### Strengths
1.This study addresses MSA design, a crucial aspect of protein structure prediction. The research demonstrates thorough motivation, theoretical analysis, and comprehensive experiments, with the paper presenting a well-founded and complete work.
2.The experimental results shows superior performance on multiple tasks and baseline models.
Weakness

### Weaknesses
1.The proposed combined loss function is well theoretically motivated, however, it still lacks ablation study on different loss functions, since it is a proposed method in your study.
2.Similarly, I wonder the effectiveness of the MSA selection module. It would be better to include more ablation study.

### Questions
Please refer to the weakness part.

### Soundness
3

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
3

### Summary
The paper proposes an MSA design framework to generate MSAs that better support downstream folding, especially low-homology or orphan proteins. The core idea is to produce sequences that trade off conservation (agreeing with strongly conserved positions) and diversity (covering plausible variation), controlled by a conservation-diversity loss operating on PLM embeddings by a base MSA Transformer. Experiment results show that PLAME can produce MSAs that improve downstream folding performance compared to Alphafold2.

### Strengths
1. The paper addresses an important problem of improving MSAs for low homology cases. 

2. The proposed conservation–diversity objective is both intuitive and theoretically justified.

### Weaknesses
1.  The notations are not clear.

  (1) The meaning for different dimensions in $ \mathbf{H}_\mathrm{enc} $ is not given (except $N$ has been introduced before in L135). 

  (2) Eq (5) does not specify which one was encoded from $\mathbf{H}_r$. 

  (3) Which two axes are permuted in $\mathbf{X}_\mathrm{dec} ^\top $ ?

  (4) MSAs were denoted by $ \mathbf{M} $ in L135, while in L238, they were denoted by $M={m_1, \cdots, m_n}$.

2. The rationale of the proposed MSA selection method is not sufficiently justified. The current Section 2.4 describes what HiFiAD does, but not why it makes sense and how it differs from previous selection methods. In addition, the efficacy claim of HiFiAD needs more evidence. Only results with HiFiAD are reported (Table 2). The baseline for only EvoDiff/MSAGPT/DHR is missing.  Details for similarity-based methods (Top/Down-Rec) are missing.

3. The sensitivity of the conservation-diversity tradeoff in Eq. (14) is unknown.

4. Comparison to standard MSA pipelines. The paper should show head-to-head results on the same folding stack when fed (1) raw database MSA, (2) database MSA augmented with PLAME sequences, and (3) PLAME-only MSA for low-homology targets.

### Questions
Please refer to the weakness section for my questions.

### Soundness
3

### Presentation
1

### Contribution
2
