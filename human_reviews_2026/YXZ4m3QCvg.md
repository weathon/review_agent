# Can Test-time Computation Mitigate Reproduction Bias in Neural Symbolic Regression?

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Symbolic regression aims to discover mathematical equations that fit given numerical data. It has been applied in various fields of scientific research, such as producing human-readable expressions that explain physical phenomena. Recently, Neural symbolic regression (NSR) methods that involve Transformers pre-trained on large-scale synthetic datasets have gained attention. While these methods offer advantages such as short inference time, they suffer from low performance, particularly when the number of input variables is large. In this study, we analyze the reasons for this limitation and suggest ways to improve NSR. We first provide a theoretical analysis showing that, under naive inference strategies, Transformers are unable to construct expressions in a compositional manner while verifying their numerical validity. Next, we explore how Transformers generate expressions in practice despite the lack of compositional generalizability. Our empirical analysis shows that the search space of NSR methods are greatly restricted due to reproduction bias, where the majority of generated expressions are merely copied from the training data. We finally examined if tailoring test-time strategies can reduce reproduction bias and improve numerical accuracy. We empirically demonstrate that providing additional information to the model at test time can significantly mitigate reproduction bias. On the other hand, we also found that reducing reproduction bias does not necessarily correlate with improved accuracy. These findings contribute to a deeper understanding of the limitation of NSR approaches and offer a foundation for designing more robust, generalizable symbolic regression methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper examines reproduction bias in Transformer-based symbolic regression, showing that models mainly memorize training expressions instead of composing new ones. It provides a theoretical justification and introduces NSR-GVS, a test-time algorithm that verifies and reuses accurate sub-expressions to reduce bias.

### Strengths
The main strength is that it explicitly demonstrates memorization behavior in neural symbolic regression.

### Weaknesses
The weakness is that these insights are not new in 2025, as several recent studies have reached similar conclusions.

### Questions
1. Since symbolic regression has been proven to be NP-hard [1], it is not surprising that neural symbolic regression struggles to find exact target equations. While the paper provides valuable empirical evidence, the results confirm known limitations rather than revealing new behavior.
2. The related work section is weak. Recent top-tier conferences (2024–2025) have largely moved away from end-to-end neural models like NeSymReS toward iterative or optimization-based symbolic regression methods, which align closely with the concept of “test-time computation” discussed in this paper. However, these recent works are barely acknowledged, suggesting the authors are not fully aware of current trends.
3. The experiments rely only on the Feynman benchmark. Additional tests on black-box datasets from SRBench would help validate the method’s effectiveness and generality.
4. The paper measures reproduction bias syntactically rather than semantically. This means equivalent equations in different algebraic forms may be incorrectly labeled as novel. A semantic-level comparison would provide a more accurate measure of reproduction bias.

[1] Virgolin, Marco, and Solon P. Pissis. "Symbolic Regression is NP-hard." Transactions on Machine Learning Research.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a valuable empirical and theoretical investigation into a critical limitation of Neural Symbolic Regression (NSR) methods: ​reproduction bias, which is the tendency of models to generate expressions nearly identical to those seen during training. The authors compellingly demonstrate that under standard inference, models like NeSymReS struggle to produce expression structures not present in the training data, and this bias correlates with poor performance on such unseen expressions. A key contribution is the exploration of various test-time computation strategies (e.g., TPSR, NSR-gvs) to mitigate this bias. The paper also provides a theoretical argument, based on circuit complexity, suggesting Transformers may lack the inherent capacity for compositional generation guided by numerical data. The empirical findings are robust, showing that while these strategies can reduce reproduction bias, this reduction does not consistently lead to improved numerical accuracy, raising important questions for future research.

### Strengths
**Identification and Empirical Demonstration of a Critical Issue**:​​ The paper's greatest strength is its clear identification and systematic empirical demonstration of reproduction bias in NSR. The experimental design, using "not_included" and "baseline" datasets, effectively isolates the problem and shows its severe impact on generalization.

**Rigorous Exploration of Test-Time Strategies**:​​ The paper goes beyond merely identifying the problem by rigorously evaluating multiple test-time computation strategies (TPSR, NSR-gvs, and their combination). The analysis in Figures 4 and 7, which examines the trade-off between reproduction bias, numerical accuracy, and computational cost, is thorough and provides valuable insights for the community. The finding that bias reduction and accuracy improvement are not directly correlated is a nuanced and important result.

**Theoretical Ambition**:​​ Attempting to ground the limitations of NSR in computational complexity theory is a commendable and ambitious goal. The argument that the "last-token prediction" problem may be too hard for bounded-precision Transformers adds a theoretical layer to the empirical observations.

### Weaknesses
​**Fundamental Flaw in the Definition and Evidence for Reproduction Bias**:​​ The central concept of "reproduction bias" is potentially flawed. The criterion for bias—whether a generated expression tree structure is present in the training data—does not account for ​symbolic equivalence. Many expressions are functionally identical (e.g., x1*(x1+x2)and x1**2 + x1*x2). Therefore, a model generating an equivalent but syntactically different expression should be considered a success, not evidence of bias. The provided evidence may actually indicate that larger models (e.g., transformer4sr trained on 1.5M samples) are betterat finding valid, equivalent expressions, explaining their higher rate of "novel" (i.e., non-identical) generations (6%) compared to a smaller model (NeSymReS on 100K samples, <1%). The paper would be significantly strengthened by evaluating whether generated expressions are functionally equivalent to the ground truth, not just structurally identical.

**Limited Scope and Questionable Generalizability of Findings**:​​ The empirical analysis relies heavily on models trained with relatively small datasets (100K expressions for the main analysis). The performance of state-of-the-art NSR models, which are pre-trained on orders of magnitude more data (e.g., 100M expressions for the official NeSymReS model), is not investigated. It remains an open question whether the severe reproduction bias demonstrated here is merely a symptom of under-training or a fundamental limitation that persists even at scale. Testing on the official pre-trained models is crucial to determine the broader relevance of the findings.

**Theoretical Argument is Misaligned with the Practical Goal of SR**:​​ The theoretical analysis in Section 4, while interesting, may be misleading. The "last-token prediction" task is an artificial construct. The ultimate goal of symbolic regression is to find anyexpression that fits the data well, regardless of its specific token sequence. A model might predict the "wrong" final token according to a specific canonical form but still produce a functionally equivalent and highly accurate expression. Therefore, the inability to solve this specific task does not necessarily prove that Transformers cannot be effective for the broader, more flexible goal of symbolic regression. The argument would be more compelling if it addressed the problem of discovering equivalent expressions.

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper studies why neural symbolic regression (NSR) models often fail to produce novel equations and instead copy training expressions (a "reproduction bias"). Theoretically, under the circuit-complexity assumption TC0 ̸= NC1, and log-precision constraints, the authors prove that for sufficiently large problem sizes no bounded-depth, polynomial-width Transformer can even solve a last-token prediction task (i.e. predict the last leaf node of otherwise complete expression that minimizes  the numeric data), implying difficulty with compositional generation  while accounting for numerical loss.

Empirically, they show that in a 1.5M training dataset, most decoded expressions are copies of the training set and accuracies decreases significantly when the target tree is unseen.

Finally they then test three test-time strategies: larger beam sizes, TPSR (MCTS), and their proposed NSR-gvs (verified subtrees). Methods that inject extra information increase novelty but don’t reliably improve accuracy, while simply enlarging the beam often boosts accuracy without reducing reproduction bias,revealing a tension between escaping memorization and maintaining fit.

### Strengths
- The identified reproduction bias is significant and analyzed soundly from both theoretical and empirical perspective

### Weaknesses
- While the paper explores different strategies and approaches, no proposed solution, as acknowledged by the author in the conclusions, seems to both decrease the reproduction bias and increase accuracy.
- Experiments are carried out with a maximum dataset size of 1.5M equations, whereas established work in the field trains these models with up to 100M. Hence, the conclusions might be biased by this difference in scale.
- The connection between the theoretical and empirical evaluation settings is poorly described.

### Questions
Feel free to address the weakness I have mentioned

### Soundness
3

### Presentation
3

### Contribution
2
