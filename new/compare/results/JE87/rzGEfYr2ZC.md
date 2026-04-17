# Review

## Summary
This paper proposes a novel pruning method for large language models that leverages the Frank-Wolfe (FW) algorithm to optimize the pruning mask. The key contributions are:
1. Formulation of mask selection as a convex program
2. Use of FW algorithm for efficient mask optimization
3. Theoretical guarantees for the quality of the rounded solution
4. Strong empirical results on modern LLMs

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors provide a clear motivation for their approach and do a good job of explaining the methodology.
2. The empirical results are comprehensive, covering multiple architectures and showing consistent improvements over baselines.
3. The theoretical analysis provides valuable insights into the trade-offs between optimization error and thresholding error.

## Weaknesses
1. The paper lacks a detailed analysis of the computational complexity and memory requirements of the proposed method compared to baselines.
2. The discussion of the limitations of the approach is somewhat limited. The authors could provide more insights into potential failure modes or scenarios where the method may not perform well.
3. The paper does not extensively explore the sensitivity of the method to hyperparameters beyond the number of iterations and sample size.

## Questions
1. How does the computational complexity of SparseFW compare to Wanda and RIA in terms of training time and memory usage?
2. What are the main limitations of the Frank-Wolfe algorithm for this problem, and how might they impact the quality of the pruning masks?
3. How sensitive is the method to the choice of calibration data, and what strategies could be employed to improve the robustness of the pruning process?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4