# Review

## Summary
The paper proposes a lossless watermarking scheme for diffusion models. The main idea is to first encode the watermark message into a binary sequence, which is then mixed with random padding and embedded into the latent code of the diffusion model. The authors theoretically prove that the watermarked noise distribution preserves the target prior up to third-order moments. Extensive experiments show that the proposed method achieves better performance than existing watermarking schemes in terms of undetectability, traceability, and computational efficiency.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The proposed method is a lossless watermarking scheme, which does not introduce any distortion to the generated images.
2. The authors theoretically prove that the watermarked noise distribution is statistically indistinguishable from a standard multivariate normal distribution.
3. The proposed method eliminates the need for per-image key management, which improves the computational efficiency and convenience of use.
4. The proposed method is encryption-free, which improves the security and robustness of the watermarking scheme.

## Weaknesses
1. The proposed method is only evaluated on two diffusion models, i.e., SD v1.52 and SD v2.13. It is unclear whether the method is applicable to other diffusion models.
2. The robustness of the proposed method against re-generation and editing attacks is not evaluated.
3. The proposed method is only evaluated on two datasets, i.e., COCO and SDP. It is unclear whether the method is applicable to other datasets.

## Questions
1. Can the proposed method be applied to other diffusion models, such as DiT?
2. How robust is the proposed method against re-generation and editing attacks?
3. Can the proposed method be applied to other datasets, such as LAION?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4