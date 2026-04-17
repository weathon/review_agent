# Review

## Summary
This paper proposes a novel method to tackle the overfitting and underfitting problems in sparse-view Gaussian Splatting. The authors propose a depth-and-density guided Gaussian dropout strategy to reduce the Gaussian density near the camera, and a distance-aware fidelity enhancement module to improve the reconstruction quality in distant areas. The authors also introduce a new evaluation metric to quantify the stability of learned Gaussian distributions. Experiments on multiple datasets demonstrate the effectiveness of the proposed method.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The authors propose a novel method to tackle the overfitting and underfitting problems in sparse-view Gaussian Splatting. The proposed method is well-motivated and technically sound.
2. The authors introduce a new evaluation metric to quantify the stability of learned Gaussian distributions, which is a novel and useful contribution.
3. The paper is well-written and easy to follow.

## Weaknesses
1. The authors do not provide a detailed analysis of the computational overhead of the proposed method. It would be helpful to report the training and inference times and compare them with the baseline methods.
2. The authors do not provide a detailed analysis of the sensitivity of the proposed method to the choice of hyperparameters. It would be helpful to report how the performance of the method changes with different choices of hyperparameters.

## Questions
1. The authors use a monocular depth estimator to estimate the depth of the scene. How accurate is the depth estimation, and how does the accuracy of the depth estimation affect the performance of the proposed method?
2. The authors use a depth-thresholding strategy to construct a binary mask to separate the image into near and far regions. How sensitive is the method to the choice of the depth threshold?
3. The authors report the PSNR, SSIM, LPIPS, and AVGE metrics. How do these metrics relate to the visual quality of the rendered images? Are there any other metrics that could be used to evaluate the visual quality of the rendered images?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4