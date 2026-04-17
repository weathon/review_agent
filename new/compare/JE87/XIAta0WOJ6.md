# Review

## Summary
This paper studies the stochastic bilevel optimization problem with a nonconvex upper-level function and a strongly convex lower-level function. The authors propose a class of algorithms called $F^2SA-p$, which builds upon the existing first-order method $F^2SA$ by using $p$th-order finite difference to approximate the hypergradient. The authors establish the SFO complexity of $O(p\epsilon^{-4-2/p})$ for $p$th-order smooth problems, which improves upon the complexity of $O(\epsilon^{-6})$ for the first-order smooth problems achieved by $F^2SA$. The authors also show that the $\Omega(\epsilon^{-4})$ lower bound for single-level optimization can be extended to the bilevel setting under a similar smoothness condition.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The authors provide a new interpretation of $F^2SA$ as using forward difference to approximate the hypergradient. Based on this, they propose a class of algorithms called $F^2SA-p$ that use $p$th-order finite difference for hyper-gradient approximation.
2. The authors establish the SFO complexity of $O(p\epsilon^{-4-2/p})$ for $p$th-order smooth problems, which improves upon the complexity of $O(\epsilon^{-6})$ for the first-order smooth problems achieved by $F^2SA$.
3. The authors show that the $\Omega(\epsilon^{-4})$ lower bound for single-level optimization can be extended to the bilevel setting under a similar smoothness condition.

## Weaknesses
1. The proposed algorithm is an extension of $F^2SA$, and the analysis is also based on similar techniques used in $F^2SA$. This makes the novelty and technical contribution of this work limited.
2. The proposed algorithm requires the gradient of $g$ to compute the finite difference, which is not required in $F^2SA$. This makes the proposed algorithm less practical than $F^2SA$.
3. The experiments are only conducted on a single dataset and a single problem setting, which is not sufficient to demonstrate the effectiveness and efficiency of the proposed algorithm. More experiments on different datasets and problem settings are needed to strengthen the empirical results.

## Questions
1. Can the authors provide more intuition and justification for the proposed algorithm, such as the connection between the $p$th-order finite difference and the hypergradient?
2. Can the authors provide more details on the practical implementation of the proposed algorithm, such as how to compute the finite difference and how to choose the parameters?
3. Can the authors provide more experimental results on different datasets and problem settings to demonstrate the effectiveness and efficiency of the proposed algorithm?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4