## Human Reviewer 1

### Summary
This work empirically compares Kolmogorov-Arnold Networks with Multi-Layer Perceptron on
learning irregular or noisy functions. The experiment results show that KAN do not always perform
the best.

### Strengths
Experiment codes are provided for reproducibility.

Do provide some insight on what KAN may be good at modeling.

### Weaknesses
The finding is purely empirical.

The paper does not clearly state the experiment setting in the main text.

The experiment does not provide conclusive results.

The experiment only tries to fit relatively simple functions. The result may not be relevant to real-world problems.

### Questions
It is possible to include more challenging problems for comparison? It is well established that MLP can model fairly complicated functions.

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
4

---

## Human Reviewer 2

### Summary
Authors compare the performance of Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptron (MLP) networks on irregular or noisy functions. The author experimentally demonstrated that KAN does not always outperform MLP.

### Strengths
- The author compared KAN and MLP on various irregular and noisy functions and experimentally demonstrated in which cases KAN is worse than MLP.

### Weaknesses
- The author merely compared KAN and MLP experimentally but did not analyze why KAN or MLP performs poorly in certain situations.

- The author experimentally demonstrated that KAN is sometimes inferior to MLP. It would be better to propose a new, improved KAN model to address this.

- There are no experiments on high-dimensional functions. In one dimension, both KAN and MLP are likely to approximate well to some extent, but more experiments are needed to explore how they perform in high-dimensional spaces with irregular points.

- If the experiments are conducted only on univariate functions, many models besides MLP can be compared with KAN. It would be beneficial to include other models commonly used in machine learning in the experiments.

### Questions
I do not have a complete understanding of KAN, but I think KAN appears to be a generalization of projection pursuit regression. Is this correct?

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper conducts a comparative analysis of experiments between MLP and KANs, discussing the outcomes. It challenges the assumption that KANs consistently outperform MLP in modeling mathematical equations, particularly with irregular functions. The experiments involve applying MLP and KAN to various functions—regular, non-differentiable, discontinuous, singular, and coherent oscillation, with and without noise. These functions are single input and single output. Variations include different training sample sizes, iteration counts, and optimizers. The findings demonstrate that KANs do not always surpass MLP.

While this paper serves as a great exploration to KANs and does establish that KANs are not invariably superior to MLP, it falls short by only providing experimental evidence without introducing new theoretical insights or network structures, thus lacking substantial academic contribution.

### Strengths
The structure of the paper is clear and well-organized.
The experimental results are clearly presented.
The experiments validate that KANs are not consistently superior to MLP.

### Weaknesses
The experiments could be designed more targeted. For instance, in the experiments for non-differentiability, both functions feature only a single non-differentiable point. A comparison between functions with single versus multiple non-differentiable points would be more insightful, given the focus on the impact of these points.

The discussion lacks depth. Given the simplicity of both the functions and network structures used, there is potential for a more detailed examination of how parameters are trained and the reasons behind specific outcomes.

The discussion section does not yield any intriguing or unexpected conclusions, nor does it propose any novel theories or structures.

### Questions
Given that the structures of both MLP and KANs are well-known, a deeper analysis of their capabilities and limitations in the related work section would be beneficial. More thorough research could uncover more significant findings. For instance, some limitations of KANs identified in the paper are not due to the Kolmogorov-Arnold Theorem but rather due to B-spline, a critical component in KANs that is not discussed in the paper at all.

### Soundness
3

### Presentation
3

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
In this empirical study, the authors compare the performance of two ad-hoc versions of KAN and MLPs, both with identical parameter counts, in learning ten single-dimensional real functions, with and without added noise.

### Strengths
The work has considered different classes of functions with some common irregularities. The empirical comparison is sound, and extensive plots provided let's the reader to compare the performance of tested KAN and MLPs in each case.

### Weaknesses
The comparison primarily centers on the resulting test accuracy curves; however, it lacks the necessary theoretical justification and fundamental analysis to substantiate the findings.

While the authors have structured the text well, the plots are somewhat cluttered and could be presented more effectively for better clarity. 

Overall, the work appears basic and does not demonstrate the level of novelty typically expected from submissions to ICLR.

### Questions
If authors can provide some theoretical insights backing the observed empirical results in all or some of the function classes tested, it would make the work more promising and considerable for this conference.

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4