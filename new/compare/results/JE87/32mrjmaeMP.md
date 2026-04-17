# Review

## Summary
The paper proposes a data-free approach to regularize representation drift in Task Arithmetic, which is a modular and scalable method for adapting foundation models. The authors leverage Kronecker-Factored Approximate Curvature (KFAC) to approximate the generalized Gauss-Newton (GGN) matrix, which is used as a regularizer to encourage weight disentanglement. The proposed method, TAK, improves weight disentanglement without requiring access to external task data, and scales representation drift regularization by aggregating per-task regularizers into a single surrogate, ensuring constant complexity and storage regardless of the number of tasks. The authors demonstrate the effectiveness of TAK in task addition and negation, and show its robustness to task vector rescaling.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper proposes a novel data-free approach to regularize representation drift in Task Arithmetic, which is a significant contribution to the field. The use of KFAC to approximate the GGN matrix is a well-established technique that allows for efficient computation of the regularizer. The proposed method, TAK, improves weight disentanglement without requiring access to external task data, which is a key advantage in many real-world scenarios such as decentralized or privacy-preserving learning. The method also scales well with the number of tasks, making it practical for large-scale applications. The paper is well-written and easy to follow, with clear explanations of the proposed method and its advantages. The authors provide detailed experimental results that demonstrate the effectiveness of TAK in task addition and negation, as well as its robustness to task vector rescaling.

## Weaknesses
The paper has a few weaknesses that should be addressed. First, while the proposed method, TAK, is data-free, it requires pre-computation of the KFAC matrices for each task, which can be computationally expensive. The authors should provide more details on the computational cost of this step and potential ways to reduce it. Second, the paper focuses primarily on task addition and negation, but does not explore other operations such as task ordering or composition, which are also important aspects of Task Arithmetic. The authors should consider extending their analysis to these other operations to provide a more comprehensive evaluation of TAK. Finally, the paper could benefit from a more thorough comparison with existing methods that address cross-task interference, such as those based on attention mechanisms or regularization techniques. The authors should include more baselines in their experiments to demonstrate the superiority of TAK.

## Questions
1. Can you provide more details on the computational cost of pre-computing the KFAC matrices for each task? How does this cost compare to the cost of training the model on each task separately?
2. Have you considered extending your analysis to other operations in Task Arithmetic, such as task ordering or composition? How do you expect TAK to perform in these scenarios?
3. Can you include more baselines in your experiments to demonstrate the superiority of TAK? Specifically, how does TAK compare to other methods that address cross-task interference, such as those based on attention mechanisms or regularization techniques?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4