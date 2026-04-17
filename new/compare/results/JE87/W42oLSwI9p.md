# Review

## Summary
The paper proposes a one-step diffusion-based approach to solve integer linear programming problems, including a newly proposed iterative integer projection (IIP) layer and an objective-guided sampling with momentum scheme.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper proposes three one-step diffusion-based solvers, i.e., CMILP, SCMILP, and MFILP, for integer linear programming problems.
2. The paper extends binary 0-1 ILP neural solver to the non-binary case for feasible solution prediction.
3. The paper proposes a sampling method based on gradient descent and a momentum-based gradient descent approach to improve the sampling process.

## Weaknesses
1. The paper lacks novelty. The one-step diffusion model is not a new concept and has been explored in previous research. The proposed objective-guided sampling method is also not novel, as it is a common approach in many papers that use diffusion models for constraint satisfaction problems. The paper fails to provide a clear explanation of how their approach differs from existing methods.
2. The paper lacks clarity in its explanation of the proposed methods. The paper fails to provide a clear explanation of how the IIP layer works and how it differs from existing methods. The paper also lacks a detailed explanation of the objective-guided sampling method and how it can ensure constraint satisfaction.
3. The paper lacks a thorough literature review. The paper fails to mention and compare with existing methods that use machine learning techniques for solving integer linear programming problems, such as the following papers: 
    1. "A GNN-guided predict-and-search framework for mixed-integer linear programming"
    2. "GNN&GBDT-guided fast optimizing framework for large-scale integer programming"
    3. "Learning to optimize for mixed-integer non-linear programming with feasibility guarantees"
    4. "A survey for solving mixed integer programming via machine learning"
    5. "Machine learning augmented branch and bound for mixed integer linear programming"
    6. "Learning to remove cuts in integer linear programming"
    7. "Neural network based lagrangian relaxation method for large scale integer programming"
    8. "Unsupervised learning for combinatorial optimization with principled objective relaxation"
    9. "Deep learning for integer programming: A survey"
4. The paper lacks a thorough experimental evaluation. The paper fails to compare their method with existing methods that use machine learning techniques for solving integer linear programming problems. The paper also lacks a detailed analysis of the performance of their method on different types of integer linear programming problems.

## Questions
1. How does the proposed objective-guided sampling method ensure constraint satisfaction?
2. How does the IIP layer differ from existing methods, and how does it handle non-binary integer problems?
3. How does the proposed method compare with existing methods that use machine learning techniques for solving integer linear programming problems?
4. How does the proposed method perform on different types of integer linear programming problems?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4