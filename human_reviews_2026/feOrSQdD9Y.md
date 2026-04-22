# Convergence Analysis of Tsetlin Machines under Noise-Free and Noisy Training Conditions: From $2$ Bits to $k$ Bits

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The Tsetlin Machine (TM) is an innovative machine learning algorithm grounded in propositional logic, achieving state-of-the-art performance across a variety of pattern recognition tasks. Prior theoretical work has established convergence results for the 1-bit operator under both noisy and noise-free conditions, and for the 2-bit XOR operator under noise-free conditions. This paper first extends the analysis to the 2-bit AND and OR operators. We show that the TM converges almost surely to the correct 2-bit AND and OR operators under the noise-free training condition, and we identify a distinctive property of the 2-bit OR operator, where a single clause can jointly represent two sub-patterns, in contrast to the XOR operator. We further investigate noisy training scenarios, demonstrating that mislabelled samples prevent exact convergence but still permit efficient learning, whereas irrelevant variables do not prevent almost-sure convergence. Building on the 2-bit analysis, we then generalize the results to the $k$-bit setting ($k>2$), providing a unified theoretical treatment applicable to general scenarios. Together, these findings provide a robust and comprehensive theoretical foundation for analyzing TM convergence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
A theoretical study of the probabilistic training convergence of Tsetlin Machines for basic logical operators with and without noise.

### Strengths
Paper is reasonably clear although the presentation could be improved in parts. For example, a figure would shorten and clarify the setup in Section 2.   It would help to explicitly state that Tables 1 and 2 are transition probabilities to bring along readers who are not already familiar with the material.

### Weaknesses
The paper, plus 30 pages of appendix, is a detailed treatise on the convergence of Tsetlin machines for basic logical operators.  It is impossible to estimate the validity of the theorems without a deep study of the lengthy appendix.

While the results are of theoretical interest,  we cannot arrive at any conclusions on practical significance without experiments on real-world data sets.   In the absence of such experiments, the paper might be better suited for a refereed journal.

### Questions
How about some experimental results on substantive real data sets? 
How can you convince the audience that Tsetlin machines may be practical alternatives to deep neural networks?

### Soundness
3

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
This work provides detailed proofs on the convergence properties of Tsetlin Machines (TMs). Given that existing literature only discussed TMs convergence in 1-bit settings for the Identity/NOT operator and in 2-bit settings for the XOR operator only, it extends the current theoretical understanding of TMs by providing (1) a noise-free (almost surely) convergence proof for the AND operator in a 2-bit setting, (2) a noise-free (almost surely) convergence proof for the OR operator in a 2-bit setting, (3) (almost surely) convergence/recurrence analysis of AND, OR, XOR operators in a 2-bit setting with noisy data (under two random noise types: mislabeled samples and irrelevant input variables).

### Strengths
S1: strong motivation is given for improving the theoretical ground of TMs properties, with a clear identification of the gap (what was already shown in the literature, and what was missing).

S2: self-contained notation for setting up the TM functionalities, consistent notations used throughout proofs.

S3: a complete proof provided for (1), and high level proof sketches provided for (2,3) where complex reasoning involves (proof details are presented in the appendix). Figure 2 helps the reader understand the proof structure.

S4: key findings on the OR operator where it is shown to be able to learn joint sub-patterns, which is novel.

S5: the convergence analysis on TM's behaviour with noisy data is particularly valuable as real-world data is more than often quite noisy.

### Weaknesses
W1: limited scope - this work explores the 2-bit settings, however, it is missing the discussion of how this work may contribute to analysis of TMs behaviours at a larger scale (in the real deployment of TMs, it is often at a much larger scale). Or more generally, missing discussion on how the theoretical improvements relates to practical usage of TMs.

W2: Section 6 seems to be weaker than section 3&4, partially due to the defer of (almost all) proof details to the appendix and only leaving the statements in the main text. The key rationales behind these findings could be partially included to enhance the strengths of these statements.

### Questions
Q1: The definition of the hyper-parameter "T" does not seem to be well presented. It was not clear to the reader what type of values it takes (it was briefly mentioned in one of the proofs that T is an integer, but no formal definition for T is found in the notation section) and what is its range. It would be nice if the authors can present the definition of T better in revisions as it is heavily used in the proofs.

Q2: as discussed in W1, can the authors explains a bit more on how this work influence the practical usage of TMs? Any insights that we can draw from these analysis to be applied to real world scenarios?

### Soundness
3

### Presentation
2

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
This paper studies the convergence behavior of Tsetlin Machines (TM) on basic Boolean operators. It proves almost-sure convergence for AND and OR under noise-free data, clarifies how the resource parameter T is necessary for OR (due to recurrent dynamics without it), and revisits XOR to contrast why clauses must include both literals there. The paper then analyzes learning under two noise models, mislabeled samples (non-convergence / recurrence) and irrelevant input variables (preserving almost-sure convergence with suitable T). Experiments on synthetic setups illustrate the theorems and qualitative behaviors.

### Strengths
1.	The paper has good theoretical novelty: Clear separation of behaviors for AND vs. OR vs. XOR, including why OR needs 𝑇 to exit recurrence, and why XOR forces inclusion of both literals (Type II feedback pressure).

2.	It has rigorous clause/TA-state analyses leading to a unique absorbing state for AND and a family of absorbing conditions for OR. The use of absorbing-state arguments vs. stationary distributions is a neat angle.

### Weaknesses
1.	The theory focuses on 1–2-bit settings. While justified as foundational, readers may want a clear roadmap for extending these techniques to higher-arity clauses and multi-bit operators beyond AND/OR/XOR, or to real datasets with structured features. (Some remarks suggest feasibility, but formal generalizations are not provided.)

2.	Convergence is asymptotic (“infinite time”). Finite-time/sample complexity bounds or rates (even coarse) would materially strengthen the results and their practical import. The empirical section shows good convergence in practice, but theory doesn’t quantify this.

3.	Experiments are primarily synthetic and ablation-style. Comparisons to classical concept learning or bandit baselines learning conjunctions/disjunctions (more related work) would better support the empirical importance.

### Questions
1.	Can you provide high-level bounds (even loose) on expected time to absorption under noise-free AND/OR as a function of m,T,s,N and input distribution?

2.	Which steps in the proofs rely critically on the 2-bit structure? Do you foresee analogous absorbing-state conditions for k-bit conjunctions/disjunctions and for multi-clause compositions? Any obstacles beyond combinatorial explosion?

3.	Can you add guidance on the qualitative effect of s (granularity) and N (TA states per action) on convergence speed/stability?

### Soundness
3

### Presentation
3

### Contribution
3
