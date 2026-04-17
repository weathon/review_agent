# Review

## Summary
This paper proposes a two-stage algorithm for offline change point localization and inference in dynamic multilayer random dot product graphs (D-MRDPGs). The first stage uses seeded binary segmentation with refined CUSUM statistics to generate a coarse set of candidates, while the second stage refines these candidates through low-rank tensor estimation. The authors establish the consistency of the algorithm in estimating the number and locations of change points. They also derive the limiting distributions of the refined estimators under both vanishing and non-vanishing jump regimes. Additionally, a data-driven procedure is developed for constructing confidence intervals. Extensive numerical experiments demonstrate the superior performance and practical utility of the proposed methods compared to existing alternatives.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The problem is well-motivated, and the proposed method is clearly explained.

2. The paper makes significant theoretical contributions by establishing the consistency of the two-stage algorithm in estimating change points and deriving the limiting distributions of the refined estimators under various regimes. These results provide a solid foundation for the proposed methods.

3. The authors conduct extensive numerical experiments on both synthetic and real-world datasets, demonstrating the superior performance of their methods compared to existing alternatives. The results showcase the practical utility and effectiveness of the proposed approach.

## Weaknesses
1. The assumption of mutual independence among the four sequences in Algorithm 1 is a bit strong. While the authors mention that the same two split tensor sequences can be used in practice, it would be helpful to provide more details on how this odd-even splitting approach works and how it compares to using truly independent sequences.

2. The paper could benefit from a more thorough discussion of the computational complexity of the proposed algorithm. While the authors provide some details on the computational cost of each stage, a more comprehensive analysis of the overall computational complexity and how it scales with various parameters would be valuable.

3. The paper could benefit from a more thorough discussion of the limitations of the proposed method. For example, the authors assume that the minimal spacing $\Delta$ between successive change points scales with the time horizon $T$, essentially bounding the number of changes $K$. It would be helpful to provide more details on how the method performs when this assumption is violated and whether there are any potential solutions to address this.

## Questions
1. Could the authors provide more details on how the odd-even splitting approach works in practice and how it compares to using truly independent sequences?

2. Could the authors provide a more comprehensive analysis of the computational complexity of the proposed algorithm and how it scales with various parameters?

3. Could the authors provide more details on how the method performs when the assumption on the minimal spacing $\Delta$ is violated and whether there are any potential solutions to address this?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4