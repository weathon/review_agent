# Review

## Summary
This paper proposes a novel method to reduce the computational cost of multi-condition diffusion models. The core idea is to replace the full attention between all the conditions and latents with two efficient attention mechanisms, position-aligned attention and keyword-scoped attention. The former restricts the attention between the latents and spatial conditions to aligned spatial positions, while the latter masks out irrelevant regions in the latents for attention with subject conditions. The authors also propose to use an early timestep sampling strategy to accelerate the fine-tuning. Experiments show that the proposed method achieves a significant inference speedup and reduces the VRAM consumption for the attention module.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The proposed method is well-motivated. The attention between latents and conditions is indeed redundant. Restricting attention to more specific regions can both improve efficiency and maintain performance.
2. The proposed position-aligned attention and keyword-scoped attention are simple yet effective. Experimental results show that the proposed method achieves a significant inference speedup and reduces the VRAM consumption for the attention module.
3. The paper is well-written and easy to follow.

## Weaknesses
1. The proposed early-timestep sampling seems to have a limited connection with the position-aligned attention and keyword-scoped attention. It would be better to have more discussion on the relationship between early-timestep sampling and the other two components.
2. The early-timestep sampling is proposed to accelerate the fine-tuning. However, the authors do not provide the training time comparison between the proposed method and the baseline.
3. The proposed method is only evaluated on three tasks, subject-Canny-to-image, subject-depth-to-image, and canny-depth-to-image. It would be better to evaluate the proposed method on more tasks, such as subject-instance-to-image, which is also considered in OminiControl and UniCombine.

## Questions
1. It would be better to provide the training time comparison between the proposed method and the baseline.
2. It would be better to evaluate the proposed method on more tasks, such as subject-instance-to-image.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4