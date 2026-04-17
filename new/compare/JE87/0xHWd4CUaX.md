# Review

## Summary
This paper proposes a novel RL framework for automated code refactoring. This framework uses contrastive pre-trained code graph embeddings to overcome the limitations of the traditional heuristic-based reward functions. The proposed method is evaluated on three code refactoring datasets and compared with four categories of refactoring approaches. The results show that the proposed method achieves the best balance across all metrics, with particularly strong gains in generalization.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The idea of using contrastive pre-trained code graph embeddings to guide code refactoring is interesting.
2. The proposed method achieves the best balance across all metrics, with particularly strong gains in generalization.

## Weaknesses
1. The writing of the paper needs to be improved. There are many typos, e.g., line 16: "objecting to code quality" should be "opposing code quality"; line 25: "refactorings" should be "refactoring"; line 46: "teacher models for guided exploration" should be "teacher models for guided refactoring"; line 50: "new manner" should be "a new manner"; line 113: "lemon deep learning technologies" should be "learning-based technologies"; line 121: "translate the defect of static approaches" should be "translate the advantage of static approaches"; line 122: "lots of supervision" should be "limited supervision"; line 123: "we take a step forward and learn refactoring aware representations using self-supervised contrastive objectives" should be "we take a step forward and learn refactoring-aware representations using self-supervised contrastive objectives"; line 132: "The standard RL framework models this interaction as a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, γ)" should be "The standard RL framework models this interaction as a Markov Decision Process (MDP) defined by the tuple (S, A, P, R, γ), where S represents the state space (code representations), A denotes the action space (possible refactorings), P describes transition dynamics, R specifies the reward function, and γ is the discount factor"; line 139: "the use of contrastive learning to make effective representation from unlabeled code" should be "the use of contrastive learning to make effective representations from unlabeled code"; line 144: "For code graphs, positive pairs can be generated through structure preserving transformations like variable renaming or statement reordering" should be "For code graphs, positive pairs can be generated through structure-preserving transformations like variable renaming or statement reordering"; line 145: "Ding et al. (2021)" should be "Ding et al. (2021)."
2. The motivation of the paper is not clear. It is unclear why the traditional heuristic-based reward functions are not sufficient to guide code refactoring.
3. The proposed method is not clearly described. For example, it is unclear how the contrastive pre-trained code graph embeddings are used to guide code refactoring.
4. The evaluation metrics are not clearly described. For example, it is unclear how Syntactic Improvement (SI) is calculated.
5. The details of the ablation study are not provided. For example, it is unclear what "w/o contrastive pre-training" means.
6. The cross-language generalization experiment is not clearly described. For example, it is unclear how the proposed method is applied to Python and C++ codebases.

## Questions
1. Why the traditional heuristic-based reward functions are not sufficient to guide code refactoring?
2. How the contrastive pre-trained code graph embeddings are used to guide code refactoring?
3. How Syntactic Improvement (SI) is calculated?
4. What "w/o contrastive pre-training" means?
5. How the proposed method is applied to Python and C++ codebases?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4