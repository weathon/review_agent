# Review

## Summary
The paper proposes a novel framework called TARget-aware Forgetting (TARF) to address the challenge of machine unlearning, which aims to make a trained model forget specific data as if it had never been used during training. The authors introduce new settings that decouple the class label and the target concept, and investigate three problems beyond conventional all-matched forgetting: target mismatch, model mismatch, and data mismatch. The proposed TARF framework consists of annealed forgetting and target-aware retaining, which collaboratively enable target identification and separation. The authors conduct comprehensive experiments on various benchmarks and real-world applications to demonstrate the effectiveness of TARF.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper introduces new settings that decouple the class label and the target concept, which is a novel approach in the field of machine unlearning.
2. The authors provide a systematic analysis of the challenges associated with restrictive unlearning in the presence of mismatched label domains.
3. The proposed TARF framework is a general approach that can handle different types of forgetting tasks, including target mismatch, model mismatch, and data mismatch.

## Weaknesses
1. The paper could benefit from a more detailed discussion of the limitations of the proposed approach and potential directions for future research.
2. The experiments are limited to image classification tasks, and it would be valuable to explore the applicability of TARF to other types of tasks.
3. The paper does not provide a detailed analysis of the computational complexity of the proposed framework, which could be an important consideration for practical applications.

## Questions
1. How does the proposed TARF framework perform on larger and more complex datasets beyond CIFAR-10 and CIFAR-100?
2. Can the authors provide more insights into the choice of hyperparameters in TARF and how they affect the performance of the framework?
3. How does the TARF framework compare to other state-of-the-art machine unlearning techniques in terms of computational efficiency?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4