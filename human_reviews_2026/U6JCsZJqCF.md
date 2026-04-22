# Improving ML attacks on LWE with data repetition and stepwise regression

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
ML attacks on learning with errors (LWE) with binary or small secrets only succeed on LWE settings with very simple secrets. For example, they can recover secrets with up to three non-zero bits when models are trained on not-reduced LWE data, and three non-zero bits in the ''cruel region (Nolte et al., 2024)'' when BKZ pre-processing is applied. We show that larger training sets and the use of repeated examples in the training data allow the recovery of denser secrets. We empirically observe a power-law relationship between model based attempts to recover the secrets, dataset size and repeated examples. We introduce a stepwise regression technique to recover the ``cool bits'' of the secret. Overall, these techniques allow for the recovery of denser binary secrets: up to Hamming weight $70$ (and $8$ cruel bits) for dimension $256$ $\log_2 q=20$ and $75$ (and $7$ cruel bits) for dimension $512$ $\log_2 q=41$ (vs $33$ and $63$ Hamming weight and $3$ cruel bits in previous works). We also demonstrate our methods' effectiveness on denser ternary secrets, showing a substantial improvement over prior work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies machine-learning-based attacks on the Learning With Errors (LWE) problem using transformer architectures. It introduces two empirical improvements: data repetition and stepwise regression. The authors find that data repetition follows a power-law scaling (data-repetition law) showing that repeating samples accelerates learning and increases recovery accuracy. Stepwise regression slightly improves the recovery of ternary and binary secrets, though it assumes some knowledge of the difficult bits during training.

### Strengths
The paper is a carefully executed experimental study: it confirms and extends known heuristics (from modular arithmetic tasks) to a cryptanalytic setting. Its key contribution is to quantify and generalize how repeated exposure to the same samples can significantly improve transformer-based attacks, revealing a clear scaling relationship that had not been explicitly measured in prior work.

### Weaknesses
While the paper presents a careful empirical study and introduces useful combinations of known techniques (e.g., data repetition and stepwise regression), the experimental improvements reported, though measurable, appear incremental relative to prior results.

### Questions
Q1: The claim that “LWE is provably hard for large n and not too large q” (line 39) is not supported by a citation, and its intended reference is unclear. It would be helpful if the authors clarified what specific result or prior work they refer to.

Q2: It is unclear how $R\Lambda$ (obtained from BKZ reduction, line 146) relates to the operator R later applied to transform (A, b) into (R A, R b). The authors should clarify the intended dimensions of $\Lambda$, whether $R\Lambda$ is the full reduction matrix or a submatrix, and how R is derived from it to ensure that the transformations are well-defined.

Q3: The paper reports a clear empirical improvement from data repetition and formulates a corresponding scaling law (line 420), but the mechanism driving this effect remains vague. The results suggest that repetition possibly increases the effective signal-to-noise ratio of gradient updates etc. I encourage the authors to clarify or justify this interpretation, either by referencing prior analyses or by providing an explicit discussion of how repetition modifies the learning dynamics in their setting.

Q4: The description of the stepwise regression procedure assumes that the cruel bits of the secret can be recovered, yet it is unclear how these bits are actually obtained or estimated in practice. The paper appears to train the regressor under full supervision, which does not correspond to an actual recovery setting. The authors should clarify whether the cruel bits are recovered iteratively (e.g., by conditioning on already predicted bits), through separate statistical inference, or simply assumed known during training. Making this distinction explicit is essential for evaluating the practical validity of the proposed stepwise regression improvement.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Note I have no prior experience in the field of ML attacks in cryptography (or really anything to do with cryptography) so this review is really just an educated guess.

The paper presents a method to attack LWE, an important post-quantum cryptography mechanism. It demonstrates significant improvements over several recent state-of-the-art benchmarks, recovering secrets with greater Hamming weights and more "cruel bits" than previously possible, with greater reliability. The authors also empirically demonstrate a power-law relationship showing how the number of "model based attempts" needed to find a secret scales with the amount of training data and the level of data repetition.

### Strengths
The paper appears to be very well motivated and clearly written. It presents a method to attack LWE, an important post-quantum cryptography mechanism. It demonstrates significant improvements over several recent state-of-the-art benchmarks, recovering secrets with greater Hamming weights and more "cruel bits" than previously possible, and with greater reliability. Its originality comes from combining disparate techniques: data repetition and stepwise regression. Although I am not an expert in this field, it would appear that this is a significant result. The synthetic data finding suggests that future attacks could be made much more efficient by first pre-training models on massive, inexpensive synthetic datasets.

### Weaknesses
(I am not really qualified to find serious weaknesses in this paper. I used Gemini 2.5 Pro to suggest areas for improvement, a condensed version of which is below. It accords with my understanding of the paper, and does not appear to be disqualifying. If the authors make revisions based on this review, I would suggest considering these points.)

While the paper presents strong results, there are a few areas that could be tightened. The "synthetic data" contribution feels a bit overstated; it's not truly "free" since it requires parameters derived from the costly BKZ reduction in the first place . It's better described as a powerful data amplification technique. Similarly, the "scaling law" analysis is quite narrow, as it's based on just one LWE setting and, for the key repetition finding, a single secret. This makes it hard to generalize as a "law" for LWE. Finally, the new stepwise regression algorithm seems to require knowing the exact number of non-zero "cool bits" ($h_\text{cool}$), which an attacker wouldn't have. The paper would be more convincing if it discussed the impact of estimating this value and how sensitive the attack is to getting that estimate wrong.

### Questions
Do you find any of the "weaknesses" above to be unfair? For those that are legitimate, can you propose changes to address them?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper proposes an improved method for uncovering secrets under the Learning With Errors (LWE) problem. It proposes to train on more data and use a revised model architecture to improve the success rate of recovering the secret. On evaluating on larger, more challenging LWE instances, the proposed method demonstrates significant improvements over prior approaches.

### Strengths
- Targets an important yet challenging problem of recovering secrets under LWE.
- Superior empirical results to prior methods

### Weaknesses
- The technical novelty appears limited and incremental compared to prior work.
- Limited theoretical analysis of why the proposed method works better.
- The presentation could be improved for easier understanding.
- Topic fit into ICLR seems not very strong

### Questions
I am not familiar with using machine learning to attack LWE. From what I understand, this paper proposes a set of engineering techniques to improve the empirical performance of recovering secrets under LWE. These techniques include more training data and slightly revised model architecture. However, these technical contributions seem incremental compared to prior work.

What are the key insights that led to these improvements? This paper is written in a way that is hard to follow, especially for readers not already familiar with prior work. Could you clarify, in simple terms, what are the main reasons that the proposed method outperforms prior approaches?

The evaluations show significant empirical improvements, which is encouraging; however, there is limited theoretical analysis of why the proposed method works better. Would it be possible to provide some theoretical insights or intuitions behind the empirical success?

I am concerned about the fit of this paper's topic into ICLR. From the machine learning perspective, improved data sizes and model tweaks seem like engineering optimizations. While the problem of recovering secrets under LWE is a hard and important problem, it is not clear how readers interested in machine learning would benefit from this work. I would appreciate it if there could be more discussion on this.

### Soundness
3

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
2

### Summary
The paper studies machine learning (ML) attacks on Learning With Errors (LWE) problems, specifically on instances with **binary or ternary sparse secrets**. The authors revisit prior ML-based attacks that struggled to recover more than a few “cruel bits” (hard-to-learn secret entries) and propose two key modifications:  
- (1) training on **very large datasets with repetitions**, and  
- (2) replacing standard regression-based recovery with a **stepwise and dual-stepwise regression scheme**.  

They test their method across several LWE configurations $(n=256,512, log_2 q=12,20,28,41)$ and report substantial improvements. For instance, they recover up to **8 cruel bits** and **Hamming weights up to 70–75**, exceeding prior ML attacks such as SALSA, PICANTE, and the “Cool & Cruel Bits method”. They also empirically observe a scaling relation  
$\ln(A_R) = C_R - \alpha_R \ln D$ between the number of model-based attempts $A_R$ and the amount of distinct data $D$,  
with the exponent $\alpha_R$ increasing as repetition $R$ grows.  
The paper also describes a synthetic-data variant, designed to match the variance of BKZ-reduced samples,  
which reportedly reproduces the BKZ results while avoiding the ~42 million CPU-hours otherwise required.

While I am generally familiar with machine learning methods, I am not an expert in lattice-based cryptography or ML-based cryptanalysis. My assessment focuses on the clarity, methodological soundness, and empirical evidence presented in the main paper.

### Strengths
1. **Clear empirical improvement.** The paper demonstrates that a combination of large-scale data with repetition and the proposed stepwise recovery substantially extends the regime where ML attacks succeed. The jump from 2–3 to 7–8 recoverable cruel bits is a convincing improvement over earlier work.

2. **Good experimental depth.** The authors evaluate on several LWE parameter sets, across both binary and ternary secrets, and document results systematically. The tables are detailed and allow for meaningful comparison to existing attacks.

3. **Interesting finding on repetition.** The role of data repetition is well-isolated and carefully quantified. The empirical scaling law with varying repetition levels is a nice insight into how data reuse can amplify model performance rather than simply causing overfitting.

4. **Practical recovery pipeline.** The proposed (dual) stepwise regression for cool-bit recovery is simple and intuitive.
It replaces the regression stage from 'Cool & Cruel' and appears to contribute to the improved results, although direct ablations are provided only in the appendix.

5. **Synthetic data analysis.** Demonstrating that synthetic data can serve as a stand-in for BKZ-reduced data is valuable from a reproducibility perspective. It makes the method much more accessible for future researchers.

### Weaknesses
1. **Scope of the scaling law.** While Figure 1 shows results for three secrets with different Hamming weights,  
the scaling relation $ln(A_R)=C_R-\alpha_R \ln D$ illustrated in Figure 2 is evaluated only for a single secret ($h = 70$).  
The main text does not report multi-secret fits or confidence intervals for the scaling parameters,  
which would strengthen the claim.

2. **Comparison on equal compute budget.** The work demonstrates improved recovery but does not normalize for total compute (GPU time + BKZ cost) against prior methods. Some efficiency comparison would help contextualize the gains.

3. **Repetition vs. memorization.** While repetition improves recovery, it remains unclear whether the gain comes from learning more meaningful correlations or from repeated exposure to the same examples. 

4. **Stepwise regression sensitivity.** The paper describes switching the regression target once the remaining fraction of 1-entries exceeds 0.5, but does not discuss how this procedure behaves if the true sparsity pattern deviates from that threshold.  
Analyzing robustness or providing an adaptive stopping criterion would clarify the method’s reliability.

5. **Lack of contextualization.** The paper does not discuss when the demonstrated recovery capabilities (e.g., number of cruel bits or Hamming weights achieved) would translate into practical cryptographic impact. A brief impact statement or discussion of real-world relevance would help readers interpret the results.

### Questions
1. I found the scaling-law section interesting, but it was a bit hard to judge how general it is.  
   Could you comment on whether this behavior also appears for other secrets or Hamming weights beyond the example in Figure 2?

2. The role of repetition seems quite central to your results.  Do you have any intuition for why repetition helps so much and where precisely it helps the most?

3. For the stepwise and dual-stepwise regression, it would help me to understand how sensitive the method is to different sparsity levels.  Did you observe cases where it becomes less reliable?

4. Did you try any other regression variants (like ridge or LASSO) before settling on the stepwise approach, or is that left for future work?

5. From a broader perspective, how do your results translate to parameter choices in deployed LWE schemes? Do they suggest specific ranges that might be unsafe under these attacks?

6. At what concrete success threshold (e.g., number of recovered cruel bits, Hamming weight, or % of secret recovered) would you consider this attack a practical threat to an LWE-based scheme in deployment? It would also help if the authors could provide a short *impact statement* that puts the empirical results into context.

### Soundness
3

### Presentation
2

### Contribution
2
