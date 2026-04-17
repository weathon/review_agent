# Review

## Summary
This paper introduces a novel sampling approach for diffusion models, particularly addressing the challenges of generating molecular data. The authors argue that conventional diffusion models struggle with molecular generation due to the highly concentrated nature of molecular data distributions, which makes even minor deviations from valid structures problematic. To tackle this, they propose a method called DIffuse and STeer (DIST), which selectively corrects and filters intermediate distributions to keep sampling trajectories aligned with valid molecular configurations. The authors provide both theoretical analysis and empirical evidence demonstrating that DIST improves the stability and validity of generated molecules while reducing computational costs. The method is shown to be effective across various diffusion-based models, including those using GNNs and Transformers, and is evaluated on standard molecular generation benchmarks such as QM9 and GEOM-Drugs.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper identifies a critical limitation in applying standard diffusion models to molecular data, namely the dense-concentrated structure (DC-structure) of molecular distributions, which makes even minor deviations from valid structures problematic.
2. The proposed DIST method is model-agnostic and can be integrated into various existing diffusion-based models, demonstrating the generality of the approach across different architectures.
3. The paper provides a rigorous theoretical analysis of the DC-structure problem, including formal definitions and proofs that establish the foundations for the proposed method.

## Weaknesses
1. The paper does not adequately address the computational overhead introduced by the additional correction steps, particularly the cost of running multiple reverse inferences for pilot samples. A more detailed analysis of the trade-off between computational cost and performance improvement is needed.
2. The effectiveness of the method relies heavily on the choice of threshold τ, which controls the selection of valid regions. The paper lacks a clear strategy for tuning this parameter, and it is unclear how sensitive the method is to different threshold values.
3. While the paper demonstrates improvements on standard benchmarks, it would be more compelling if it included evaluations on more challenging datasets or real-world applications where the DC-structure problem is more pronounced.

## Questions
1. The paper mentions using a universal threshold τ for filtering batches. How is this universal threshold chosen, and how sensitive is the method to different threshold values?
2. The proposed method requires running multiple reverse inferences for pilot samples, which likely introduces significant computational overhead. Can you provide a more detailed analysis of the computational cost compared to standard sampling methods?
3. How does the method perform on more complex molecular structures or datasets beyond QM9 and GEOM-Drugs? Are there specific types of molecules where the method is particularly effective or less so?
4. The paper mentions that the method can reduce the number of timesteps required. However, it would be helpful to see a more detailed analysis of how the method's performance scales with different numbers of timesteps.
5. How does the proposed method compare with other recent approaches in molecular generation that also address the issue of sampling from high-dimensional, multimodal distributions?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4