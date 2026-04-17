# Review

## Summary
The paper proposes a probabilistic framework to infer task composition for meta-learning. The authors propose a generative model that represents tasks as a combination of reusable modules. The model is trained by maximizing the likelihood of the training tasks, and at test time, it infers the task composition that best explains the new task.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
- The paper tackles an important problem in meta-learning, which is to quickly infer a solution to a new task using feedback from previous tasks. 
- The probabilistic framework provides a principled approach to model task composition and separate within-module and across-module dynamics. 
- The paper demonstrates the effectiveness of the proposed approach on synthetic rule learning and motor learning tasks.

## Weaknesses
- The proposed approach assumes that the training and test tasks share a common set of modules, which may not always hold true in practice. 
- The evaluation is limited to synthetic tasks, and it is unclear how well the approach would generalize to more complex and realistic tasks. 
- The proposed approach assumes that the training and test tasks share a common set of modules, which may not always hold true in practice. 
- The paper lacks a comparison with existing meta-learning approaches that also aim to leverage feedback from previous tasks.

## Questions
- How does the proposed approach handle cases where the test task requires a new module that was not encountered during training? 
- Can the approach handle tasks with continuous module parameters, such as in a continuous control problem? 
- How does the proposed approach compare to existing meta-learning approaches in terms of sample efficiency and final performance? 
- How sensitive is the approach to the choice of particle filter used for inference? 
- How well does the approach generalize to more complex and realistic tasks, such as image classification or natural language processing?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4