# Review

## Summary
The paper introduces MoEP (Modular Expert Paths), a method to add sparsity to language models without increasing the total number of parameters, unlike traditional Mixture-of-Experts (MoE) methods. MoEP combines model parallelism with MoE-style linear projections to enable selective token activation, accelerating model learning. The authors demonstrate that MoEP outperforms the GPT-2 baseline, offering a promising approach for creating compact, sparse models.

## Soundness
1

## Presentation
1

## Contribution
1

## Strengths
The paper's focus on sparsity in language models is timely and relevant, addressing a critical need for more efficient models without increasing parameter counts.

## Weaknesses
1. The paper's structure and flow are difficult to follow, with abrupt transitions and unclear explanations of key components like the role of the linear router and the exact nature of expert selection.
2. The experimental setup is severely limited, with only GPT-2 used as a baseline and evaluations conducted on a small dataset, making it hard to assess the generalizability of MoEP.
3. The paper lacks ablation studies to validate the contributions of different components, leaving the reader uncertain about the impact of various design choices.

## Questions
1. Can the authors provide a more detailed explanation of the linear router and how it selects experts?
2. How does MoEP compare to other sparsity-inducing methods that do not increase the total number of parameters, such as those based on pruning?
3. Would the authors consider adding additional baselines or conducting experiments on larger datasets to validate the robustness of MoEP?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
1

## Confidence
4