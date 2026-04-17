# Review

## Summary
This paper reveals a priming vulnerability in diffusion language models (DLMs), which stems from their iterative denoising process. Specifically, if an affirmative token for a harmful query appears at an intermediate step, subsequent denoising can be steered toward a harmful response. To address this vulnerability, the authors propose a new safety alignment method called Recovery Alignment (RA), which trains models to generate safe responses from contaminated intermediate states that contain affirmative tokens. The experiments demonstrate that RA significantly mitigates the vulnerability with minimal impact on task performance and improves robustness against conventional jailbreak attacks.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-written and easy to follow.
- The proposed method is simple and effective, with minimal impact on task performance.
- The experiments are comprehensive, including both qualitative and quantitative analyses.

## Weaknesses
- The proposed method is only effective against the priming vulnerability and may not address other types of jailbreak attacks.
- The experimental models are limited to three LLaDA family models. It would be beneficial to include models like MixCoT and other non-LLaDA models.
- The paper lacks a detailed analysis of the training cost associated with the proposed method.

## Questions
- How does the proposed method perform on other models, such as MixCoT and other non-LLaDA models?
- What is the training cost associated with the proposed method, and is it feasible for large-scale applications?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4