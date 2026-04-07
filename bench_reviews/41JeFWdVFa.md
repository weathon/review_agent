## Summary
This paper proposes LDP, a lightweight denoising autoencoder plug-in designed to improve the generalization of single-image super-resolution models to unseen degradations. LDP models the degradation process by conditioning on LR high-frequency components and enforces LR cyclic consistency. It operates in two modes: as an auxiliary training loss and as an inference-time post-processing module for diffusion models via posterior sampling. Experiments across GAN-, transformer-, Mamba-, and diffusion-based SR models show consistent improvements on synthetic and real-world benchmarks.

## Strengths
- **Comprehensive and convincing evaluation across diverse architectures and degradation types.** The paper demonstrates performance gains for four distinct SR model families (FeMaSR, SwinIR, MambaIR, StableSR) across five synthetic degradation categories and three real-world datasets, using both reference and non-reference metrics. Evidence: Tables 3, 4, and 5 show consistent improvements (e.g., StableSR gains up to +2.16 PSNR on hybrid degradations).
- **Effective and non-trivial degradation modeling.** The method is shown not to collapse into simple downsampling, a common failure mode. Evidence: Table 2 shows LDP’s predicted LR images have significantly lower similarity to downsampled SR images than strong baselines like DRN, while Table 1 shows it achieves strong LR prediction metrics across diverse degradations.
- **Practical, flexible, and lightweight design.** With only ~642K parameters, LDP functions as a plug-in training loss and an inference-time correction module, making it widely applicable. Evidence: Sections 3.3 and 4 demonstrate successful application in both fine-tuning and posterior sampling settings, and Table 14 confirms its low memory overhead compared to alternatives.

## Weaknesses
- **Missing direct performance comparison with the most relevant contemporary baseline (Lway).** The paper compares LDP’s training cost to Lway but does not provide a direct comparison of super-resolution performance on the same benchmarks. Since Lway is a directly competing degradation-modeling method for SR generalization, its absence from Tables 3 and 4 undermines the claim that LDP is a superior plug-in. Evidence: Lway is discussed in related work and compared only for training efficiency in Table 14.
- **Insufficient discussion of the computational trade-off for inference-time posterior sampling.** While LDP improves consistency, its use in posterior sampling can incur a substantial inference time penalty (e.g., ~9x slowdown for StableSR in Table 13). The paper buries this analysis in the appendix and does not adequately discuss whether the modest quantitative gains (e.g., often <0.01 in CLIPIQA for UPSR in Table 5) justify this cost in the main limitations. This is a practical concern for deployment.
- **Conceptual reliance on prior work without full ablation of core components.** The core idea of using noise alignment to bridge HR and LR features builds directly on DR2 (Wang et al., 2023b). While the conditional DAE formulation is a useful instantiation, the ablation study (Table 6) does not isolate the contribution of the proposed patch-dependent noise or the necessity of the Degradation Prediction Module versus a simpler conditioning mechanism. This makes it harder to attribute gains specifically to the novel design choices.

## Nice-to-Haves
- **Evaluation on a broader suite of real-world degradation benchmarks** (e.g., NTIRE challenges, RealBlur) to further substantiate claims of robust generalization.
- **Deeper failure mode analysis** beyond the noted texture-artifact trade-off for FeMaSR, such as per-image performance analysis or identifying degradation types where LDP underperforms.
- **Ablation comparing the conditioning signal *yhf* to other potential signals** (e.g., learned embeddings) to more rigorously justify its design.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the diffusion alignment property is not explained.** The paper provides a citation to Wang et al. (2023b) and an intuitive statement in Section 3.1; a full derivation is not required for an empirical paper.
- **Criticism about vague training details.** Hyperparameters for fine-tuning each model are provided in Appendix D, which is sufficient for reproducibility.
- **Criticism about formatting artifacts in equations.** These are parser issues, not paper problems.
- **Suggestion to integrate training and inference modes for a single model.** This is an interesting next step but not required to validate the current contribution.
- **Demand for comparisons to DANv2, DDNM, GRL, or MAN.** The paper’s scope is a plug-in for existing models, not a new SOTA SR method; comparisons to degradation-modeling plug-ins (like Lway) are more relevant than full SR methods.

## Novel Insights
The paper’s primary novel insight is the formulation of a lightweight, conditional denoising autoencoder as a dual-mode plug-in for SR generalization. By using LR high-frequency components as a condition and integrating patch-dependent noise, it provides a practical and efficient way to enforce cycle consistency across diverse SR architectures. While the underlying principles of degradation modeling and consistency are known, the specific instantiation as a tiny, trainable plug-in applicable during both training and inference is a useful engineering insight with demonstrated empirical benefits.

## Suggestions
- Add a direct super-resolution performance comparison between LDP and Lway (the most relevant baseline) on the same synthetic and real-world benchmarks used in Tables 3 and 4 to firmly establish superiority.
- Move the discussion of computational cost for posterior sampling (from Appendix F, Table 13) into the main limitations section, with a clear analysis of the fidelity-versus-speed trade-off.
- Consider a supplementary ablation experiment that trains an LDP variant without patch-dependent noise (i.e., using a global timestep) to quantify the importance of this design choice.