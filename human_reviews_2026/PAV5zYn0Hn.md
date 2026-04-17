# Neural Sum-of-Squares: Certifying the Nonnegativity of Polynomials with Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Certifying nonnegativity of polynomials is a well-known NP-hard problem with direct applications spanning non-convex optimization, control, robotics, and beyond. A sufficient condition for nonnegativity is the Sum-of-Squares property, i.e., it can be written as a sum of squares of other polynomials. In practice, however, certifying the SOS criterion remains computationally expensive and often involves solving a Semidefinite Program (SDP), whose dimensionality grows quadratically in the size of the monomial basis of the SOS expression; hence, various methods to reduce the size of the monomial basis have been proposed. In this work, we introduce the first learning-augmented algorithm to certify the SOS criterion. To this end, we train a Transformer model that predicts an almost-minimal monomial basis for a given polynomial, thereby drastically reducing the size of the corresponding SDP. Our overall methodology comprises three key components: efficient training dataset generation of over 100 million SOS polynomials, design and training of the corresponding Transformer architecture, and a systematic fallback mechanism to ensure correct termination, which we analyze theoretically. We validate our approach on over 200 benchmark datasets, achieving speedups of over $100\times$ compared to state-of-the-art solvers and enabling the solution of instances where competing approaches fail. Our findings provide novel insights towards transforming the practical scalability of SOS programming. Code is available at https://github.com/ZIB-IOL/Neural-Sum-of-Squares.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel learning-augmented approach to Sum-of-Squares (SOS) certification, using transformers to predict small monomial bases resulting in large reduction of the corresponding SDP problem. An iterative fallback mechanism is deployed to ensure correctness of the approach. Empirical results demonstrate important speedups, as well as robustness across polynomial structures, supported by experiments on extensive synthetic datasets. The writing is clear, figures are informative, and the method’s motivation, i.e., addressing the core computational bottleneck in SOS programming, is convincing.

### Strengths
The approach is novel, sound, supported by experiments, and demonstrates consistent speedups across different polynomial structures. The writing of the main part is mostly clear, and the overall methodology is both coherent and practically relevant.

### Weaknesses
While the approach is promising, the evaluation is limited to synthetic data, making its generalization to real or benchmark SOS problems unclear. The theoretical analysis offers mainly worst-case guarantees and does not fully characterize how prediction quality influences solver performance. Comparisons with modern sparsity-based SOS solvers are brief.

### Major Points

- The paper relies on several different transformer architectures for different experiment grids, but the reasons behind these architectural choices are not clearly explained. In addition, training appears to require very large datasets, but there is no information about the training costs. As a result, it is difficult to evaluate the practical trade-off between the cost of learning and the runtime improvements obtained during SOS solving.

- Although the paper claims real-world SOS benchmarks are not widely available, there are well-established applications, such as Lyapunov stability certification in control. Testing even a small selection of such problems would greatly strengthen the practical relevance and demonstrate generalization beyond synthetic data.

- Theoretical results give useful worst-case guarantees for the fallback mechanism, but they do not provide any attempt of a principled explanation of when the transformer should produce accurate predictions that yield substantial speedups. At present, the conditions under which the learning generalizes reliably (e.g., to sparse or block-diagonal structure) are supported only by experiments.


### Minor Points

- It would be appropriate to distinguish between a polynomial $p \in \mathbb{R}[x_1,\dots,x_n]$ and its evaluation $p(\mathbf{x}) \in \mathbb{R}$ at some point $\mathbf{x} \in \mathbb{R}^n$.
- Line 106: “…smallest possible basis *for which a given polynomial* $p$ admits an SOS decomposition.”
- Lines 126–129: rephrase for clarity.
- Figures should be placed and discussed immediately after they are introduced. For example, Figure 1 is only referred to at Line 185. Apply this consistently throughout the paper (**especially Section 4**) and appendix for all figures and tables to improve readability.
- Line 210: there should be an intersection with $\mathbf{Z}^n_{\geq 0}$, as the networks likely do not predict rational exponents.
- A lemma typically denotes a nontrivial intermediate theoretical result. Since Lemma 1 is a direct observation from the SOS definition, calling it a remark may probably be more appropriate.
- It is not necessary to include the Newton polytope in the input to your algorithms, as the polytope depends on the true input $p$.
- Line 343: replace *is* by *are*.
- Figure 3 (or Section 4.1) could be improved by reporting, for example, the average gap between the Transformer’s prediction and the minimal necessary basis size.
- Figure 6 seems more appropriate for the main text, where it could help illustrate the concept on your simple toy example.
- Line 1001 (Table 9): k should be in math mode.
- Use $\hat{B}$ consistently for a single meaning (see Lines 1069–1070).
- Algorithm 2: write the concatenation $(B_{\text{cov}}, C)$ properly. Also, use t\in`\{2,...,k\}`  instead of 
$t \in (2, \dots, k)$.
- Line 1511: $|S|$ should be $|S(p)|$.

### Questions
## Questions

1. Why is it not reasonable to pick a single transformer architecture that is not dependent on the data, and that has some specialized features, such as equivariance under variable ordering?

2. You generate non-SOS polynomials by perturbing PSD spectra in order to test the overhead of your approach. Did you also consider the inverse procedure — starting from non-SOS polynomials and projecting or adjusting them into the SOS cone — to obtain more realistic and challenging borderline SOS instances for the training data?

3. Any rationale for why your approach underperforms on small and simple test cases?

4. In Table 2, what is the structure of polynomials used for testing? Averages are taken over how many instances?

5. In Figure 7(ii), why is #$\text{SDPs}$ (y-axis) between 0 and 1?

6. What is $B^{*}$ in Table 1?

7. What is the meaning of the numbers above the bars in Figure 3?

8. Why is $L=2$ used in Experiment 3? By “combined approach,” do you mean first using the greedy approach to ensure the necessary condition from Lemma 1, then using permutation-based expansion with $L=2$?

9. What modifications are necessary for your approach to be applied to general (un)constrained polynomial optimization problems?

10. Which repair mechanism was used in Figure 5(ii)?

11. Line 1563: What is $T$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a learning-augmented algorithm for certifying the nonnegativity of polynomials via the sum-of-squares (SOS) criterion, a classic computational bottleneck in polynomial optimization. The authors leverage a Transformer model to predict a compact monomial basis for a polynomial, reducing the size (and cost) of the associated semidefinite program (SDP) used for SOS verification. They propose a fallback mechanism to ensure correctness for incomplete predictions, provide theoretical analysis on algorithmic efficiency, and empirically validate the method over a very large synthetic dataset and multiple benchmark scenarios, demonstrating substantial speedup over state-of-the-art methods while maintaining correctness guarantees.

### Strengths
1. The paper tackles the persistent computational challenge of SOS certification with a hybrid approach that smartly decomposes basis selection into a learning-driven task, while retaining the robustness of fallback combinatorial methods. The integration of a Transformer model into this traditionally symbolic and combinatorial problem is notably creative.

2. The systematic repair and iterative expansion procedure ensures that, even when the learned model predictions are imperfect, the algorithm retains the worst-case correctness guarantees of classical approaches. The underlying fallback logic is clearly explained and theoretically justified.

3. The empirical evaluation, including Table 1 and Table 2, is broad, comparing against strong baselines such as Newton polytope and full-basis SDP, and modern solvers like SoS.jl and TSSOS. The results are benchmarked across polynomial structure (dense, sparse, low-rank, block-diagonal), variable and degree diversity, and stress test the approach on scalability and distribution shifts.

4. The paper rigorously analyzes the robustness, computational savings, and expansion scheduling, connecting prediction quality to overall efficiency and validating these claims empirically.

5. The paper delivers detailed ablations on design choices such as permutation count and expansion scheduling, showing how performance and computational tradeoffs play out in practice.

### Weaknesses
1. The empirical results rely almost entirely on synthetically generated polynomial instances. The lack of real-world or domain-specific SOS instances is acknowledged as a limitation, but is still significant. This limits the evidence for generalization beyond the synthetic distribution and may mask domain-specific failings.

2. Although Table 1 and other results show strong average-case speedups, there is only brief mention of numerical instability (e.g., SCS used for low-rank cases due to issues with MOSEK). There is little detailed discussion on how the approach fares with poorly conditioned or pathological instances, or the frequency and severity of failed or unreliable SDP solves.

3. While there are useful heatmaps and success rate statistics, the paper stops short of a rigorous exploration of the behavioral regime where the Transformer model fails most catastrophically (e.g., in dense and low-rank polynomials). Further, it remains unclear whether model “hallucinations” or systematic biases exist in certain polynomial structures that could lead to silent failure, even after repair.

4. The algorithm for fallback/expansion and scoring monomials by permutation frequency is mathematically described, but several technical implementation choices and their rationale (e.g., why a specific permutation count $L=4\sim8$ balances computational cost and recall) are only empirically motivated. It would be beneficial to see more theoretical discussion or even bounds on the expected number of necessary expansions under mild error assumptions. Furthermore, the text can be unclear on the relationship between minimum basis selection and coverage repair, especially when the predicted basis is far from minimal.

5. The paper claims generalization to degrees and distributions not seen in training, but the OOD analysis (while a nice touch) remains somewhat superficial and is itself conducted over synthetic shifts. More rigorous OOD characterization (such as sharp ablation on unfamiliar sparsity, more adversarial polynomials, or empirically ill-posed SDPs) is missing.

6. The model is said to predict “near-minimal” bases, but there is no formal guarantee, and the possibility of pathological overextension (and thus loss of computational gains) is not fully explored in the main paper.

7. The repair algorithms are shown to work well overall, but the impact of their greedy design is not fully characterized in failure cases, such as cases with many missing basis elements or in noisy/perturbed input. Table 9 and Table 10 provide some view on combinatoric recovery, but the exhaustive performance of the repair mechanism, especially as the dataset difficulty increases, is left for future work.

### Questions
1. Are there plans or possibilities to evaluate the framework on real-world, domain-specific SOS problems (e.g., systems or control, chemical design, robotics)? Can the authors comment on what barriers exist to acquiring such benchmarks, or how the method would fare on “messier” non-synthetic input?

2. Could the authors expand on the types and frequency of solver instabilities encountered, e.g., when switching from MOSEK to SCS? Are there specific structures or matrix conditions that systematically cause difficulties? What practical guidelines could be recommended?

3. For the few cases where the coverage-repaired basis remains insufficient and a large number of basis expansions are required, what failure patterns emerge? Do these correlate with any specific polynomial properties, and can they be detected a priori?

4. Is there any pathway to either guarantee (or at least probabilistically certify) that the learned basis is minimal, or offer reliable bounds on non-minimality? Are there pathological cases in the data generation or in-the-wild polynomials that routinely “confuse” the model?

5. For the non-SOS (negative) polynomials, what is the practical cost for fallback expansion, especially for high-degree/large-variable settings? Could this be further accelerated with early termination heuristics, and would this threaten guarantees?

6. Have the authors tried scaling beyond 100 variables or polynomials with very large/structured sparsity—are there new bottlenecks for the Transformer or SDP solvers in such regimes?

7. Are there possible theoretical refinements to the expansion schedule or the permutation scoring to further reduce worst-case number of SDPs or average expansion cost?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the challenge of certifying the non-negativity of polynomials efficiently being constructing a minimal basis based on the output of a trained transformer model and subsequently correcting the prediction to form a valid basis. The paper presents a comprehensive set of experiments to demonstrate their claims of drastic speedups for SOS programming.

### Strengths
**Well-Motivated and Interesting Problem**:
The problem of identifying a compact basis is, in the reviewer’s opinion, interesting, and this paper proposes a novel learning-based algorithm to address this challenge.

**Comprehensive Experimental Evaluation**:
The paper presents a range of experiments that provide strong empirical evidence supporting its claims.

**Strong Experimental Results**:
The proposed solution consistently demonstrates substantial speedups over baseline methods.

The authors also provide readable and well-documented code in the supplementary material, which aids reproducibility (although I have not attempted to reproduce the results).

### Weaknesses
**No Insight into the necessity of transformers**:
It is not clear why transformers, in particular, are essential to the proposed method. As presented, the approach appears applicable to any sequence-to-sequence model with a similar encoding scheme. Furthermore, the paper does not provide insights from the learned model itself that could help explain how or why the trained models can predict compact bases efficiently.

**Several missing discussions/references**:
This work does not adequately address the rich body of related work on basis selection in Polynomial Optimization. In particular, the authors should make note of and should consider baselining against the following works:

[1]. Zheng et al - Chordal decomposition in operator-splitting methods for sparse semidefinite programs, 2023

[2] Mason and Papachristodoulou - Chordal sparsity, decomposing SDPs and the Lyapunov equation, 2014

[3] Newton and Papachristodoulou - Sparse polynomial optimisation for neural network verification, 2023

[4] Ahmadi and Hall - Sum of squares basis pursuit with linear and second order cone programming, 2015

[5] Miller et al, Decomposed structured subsets for semidefinite and sum-of-squares optimization, 2022

**Lack of experiments on real-world datasets**: 

While the basis pursuit for SOS verification is promising, this work suffers from a lack of applications to problems of interest that utilize polynomial optimization. Moreover, there is a lack of baselining against methods that don’t use basis search to improve efficiency - for instance, how does the end-to-end performance of the proposed transformer-based approach compare with AnySOS [6]? Can the proposed approach be used for real-time data-driven control that uses SOS programming, such as [7],[8]?

[6] Driggs and Fawzi. AnySOS: An anytime algorithm for sos programming, 2019

[7] Dai and Sznaier. A semi-algebraic optimization approach to data-driven control of continuous-time nonlinear systems, 2021

[8] Strasser and Berberich. Data-driven control of nonlinear systems: Beyond polynomial dynamics. 2021

### Questions
1. Since basis repair is being performed, how would the method perform when initialized with a random starting basis, as opposed to predicting them from a learnt model?

2. How does the performance of the method change across different checkpoints during training?

3. From line 693, it appears that only one epoch of training is required, suggesting very fast convergence. Could the authors provide insights into why this is the case?

4. Why would other sequence-to-sequence models, such as RNNs, LSTMs, or SSMs, not be suitable for this task?

5. Several lemmas (Lemmas 4, 5, and 6) and a theorem (Theorem 8) are presented in the appendix without reference in the main text. Could the authors include a discussion in the main body explaining the significance of these results and how they contribute to the development of the method?

6. In lines 262-263, the authors state: ‘’we exploit the fact that our Transformer is not equivariant with respect to variable orderings.’’ Could the authors provide a brief justification for this claim?

7. The paper claims that ‘’Despite being trained only on degree 12 polynomials, our model can successfully predict bases for degree 20 polynomials by leveraging familiar substructures—though it has not seen complete degree 20 monomials.’’ It remains unclear whether this behavior would extend to settings with a much larger number of variables (say 1000s).

---
Some minor comments/nitpicks that did not affect the decision:

Line 156: Specify that $Q$ is a *symmetric* matrix; this would improve clarity and rigor.

Figure 3 is presented after Figure 4; consider reordering for consistency.

**Looking forward to the discussion period to clarify these questions and strengthen this work.**

### Soundness
3

### Presentation
1

### Contribution
2
