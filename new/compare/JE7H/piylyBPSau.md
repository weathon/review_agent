---
job_id: a733e729-826c-402b-af3d-63fc4799978f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: piylyBPSau.pdf
paper: GENCOGS: Generative Completion-Based 3D Gaussian Splatting for High-Fidelity Few-Shot Novel View Synthesis
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a generative-completion-based 3D Gaussian Splatting method for few-shot novel view synthesis, which is squarely within ICLR’s scope on representation learning, generative models, and vision.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methods, Experiments, Results, Conclusion) are present. The work is technically non-trivial, has substantial experiments with strong baselines, and is written in clear English. While I see several important weaknesses (notably in technical rigor and positioning), they are not of the “fatal / desk-reject” type.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to LLM reviewers, or suspicious invisible content in the provided main text.

---

# Expected Review Outcome:

## Summary

The paper introduces GenCoGS, a 3D Gaussian Splatting framework for few-shot novel view synthesis that explicitly targets scene completion in under-observed regions. It proposes two generative completion-based strategies: (1) a Generative point Cloud completion-based Gaussian Initialization (GCGI) that learns to generate complementary 3D points and filters them to obtain a more complete point cloud for Gaussian initialization; and (2) a Generative pseudo view Completion-based Gaussian Optimization (GCGO) that uses an image-to-video diffusion model, a perturbed camera trajectory, and a generative consistency loss to guide Gaussian optimization with completed pseudo views while attempting to mitigate hallucinations. Extensive experiments on LLFF, DTU, and Shiny show consistent improvements over several strong NeRF-, 3DGS-, and diffusion-based few-shot NVS baselines.

## Strengths

1. **Clear problem focus and motivation.**  
   The paper articulates a concrete limitation of current 3DGS-based few-shot NVS methods: poor scene completion and artifacts in under-observed regions due to over-reliance on sparse training views and simple pseudo-view sampling (Figures 1 and 4). The idea of explicitly addressing both initialization (point cloud completeness) and optimization (pseudo-view hallucination) is a coherent design choice.

2. **Reasonably well-designed generative completion pipeline.**  
   - The GCGI pipeline combines a point-based generator (DGCNN + Transformer + FoldingNet) with a kd-tree-based filtering module to “generate-and-filter” complementary points (Section 3.1, Equations (1)–(8), Figure 3). Conceptually, this is a sensible way to enrich the initial SfM point cloud without fully trusting a generative model.  
   - The GCGO pipeline uses a perturbed camera trajectory (Equation (11)) and an I2V diffusion model conditioned on CLIP features and pseudo views, combined with a spatially adaptive confidence mask and LPIPS-based structure loss for hallucination suppression (Equations (12)–(19), Figure 4). While heuristic, the components are reasonably motivated and tied to concrete failure modes.

3. **Strong and broad empirical results with relevant baselines.**  
   - On LLFF, Table 1 shows GenCoGS outperforming 3DGS-based baselines such as FSGS, DNGaussian, BinoGS, and diffusion-based IPSM, ReconFusion, CAT3D, and ReconX across PSNR / SSIM / LPIPS / AVGE, under 3-, 6-, and 9-view settings. The margins over the strongest GS baseline (often BinoGS) are moderate but consistent (e.g., +0.69 dB PSNR and lower AVGE in the 3-view case).  
   - On DTU, Table 2 and extended Table 7 report sizable gains (up to +2.40 dB PSNR and clear SSIM/LPIPS/AVGE improvements) over both NeRF-based and 3DGS-based few-shot methods.  
   - On the challenging Shiny dataset, Table 3 again shows non-trivial improvements over FreeNeRF, RegNeRF, SparseNeRF, and FSGS.  
   Overall, the quantitative evidence is strong and covers multiple datasets and shot counts.

4. **Ablation studies that meaningfully dissect components.**  
   - Table 4 shows that GCGI and GCGO each contribute to performance, and their combination gives the best metrics on LLFF (3-view).  
   - Table 6 studies the roles of CPG and CPF within GCGI, including a degradation scenario where only 1/4 of SfM points are kept; the complementary generation and filtering consistently recover or exceed baseline quality, indicating some robustness to poor SfM.  
   - Table 5 isolates the impact of trajectory-based sampling vs random sampling and of the generative consistency loss \(\mathcal{L}_{GC}\); the combination yields the best numbers.  
   - Appendix ablations (Figures 9–12, Table 9) probe several hyperparameters (\(\delta_1, \delta_2, \delta_3, A, f, \beta\)), which is useful to understand trade-offs between exploration and hallucination.

5. **Qualitative figures support the claims about scene completion and hallucination.**  
   - Figure 3 visually illustrates how the initial point cloud \(\mathbf{P}_0\), the naive combination with generative points \(\mathbf{P}_c\), and the filtered complete point cloud \(\mathbf{P}_f\) differ; the filtered point cloud indeed appears denser and less noisy than \(\mathbf{P}_c\), aligning with the CPF design.  
   - Figure 6 compares GenCoGS to DNGaussian, BinoGS, and ViewCrafter on LLFF; the highlighted insets show GenCoGS better resolving fine details and reducing floating artifacts, particularly in under-observed background regions.  
   - Figure 7’s ablation views (baseline vs +GCGI vs +GCGO vs full) qualitatively corroborate Table 4, with GCGI improving structure and GCGO filling hollows.  
   - Figure 8 demonstrates the “see-saw” described in the text: larger trajectory perturbation \(A\) yields more coverage but noticeable hallucination, which is mitigated at the chosen \(A=2.0\).

6. **Reproducibility and implementation details.**  
   The paper describes key hyperparameters (e.g., \(k, \delta_1, A, f, \delta_2, \delta_3, \alpha, \beta, m\)), optimization schedule, densification regime, datasets and splits, and evaluation metrics. Appendix A provides preliminaries for 3DGS and I2V diffusion, and Appendix B expands experimental details. Although some engineering detail is still missing (see weaknesses), the core pipeline is reasonably well specified.

## Weaknesses

1. **Limited conceptual novelty relative to recent diffusion-guided and few-shot 3DGS works.**  
   The paper’s main ingredients are: (i) a point-cloud completion network applied to SfM points, (ii) a kd-tree-based distance filter, (iii) diffusion-based pseudo-view generation with perturbed camera trajectories, and (iv) confidence masking + LPIPS regularization. All of these are existing ideas in isolation (point completion, spatial filtering, diffusion-guided 3D, pseudo views), and several recent works already combine diffusion priors with NeRF/3DGS for few-shot NVS (ReconFusion, IPSM, CAT3D, ViewCrafter, ReconX) and also emphasize structural consistency and hallucination reduction.  
   The paper positions itself mainly against FSGS, DNGaussian, BinoGS, and a few diffusion baselines, but it does not convincingly articulate what is *fundamentally* new beyond “using a learned point completion model plus a heuristic filter, and using I2V diffusion for pseudo views plus a heuristic mask”. This comes across more as a fairly strong engineering combination rather than a conceptually sharp advance.

2. **Mathematical formulation in the CPF module has inconsistencies and is under-specified.**  
   - Equation (5) is problematic:  
     \[
     p_{i,k} = k - \min_{p \in (\mathbf{P}_0 \cap t_i)} \|p_i' - p\|,\quad p_i' \in t_i.
     \]  
     Here \(p_{i,k}\) is supposed to denote the \(k\)-th nearest neighbor of \(p_i'\), but the right-hand side subtracts a distance from the integer \(k\), which is nonsensical dimensionally. At best, this seems like a typo; at worst, it obscures the actual neighbor selection. A more plausible formulation would be something like \(p_{i,k} = \arg\min_{p \in (\mathbf{P}_0 \cap t_i)} \|p_i' - p\|\) for each of the \(k\) nearest neighbors. Without a correct equation and explicit indexing, the definition of \(y_i\) in Equation (6) is ambiguous.  
   - Equation (7) defines \(\mu(\mathbf{P}_0)\) as the mean pairwise distance across all point pairs, but this is extremely expensive (\(O(n^2)\)) and not discussed in terms of approximation or subsampling; it is also unclear whether this is recomputed per scene or per batch.  
   - The filtering mask \(M = \mathbf{1}(y \le \delta_1 \cdot \mu(\mathbf{P}_0))\) is fully hand-crafted; there is no learning or adaptation, and there is no analysis of sensitivity beyond a one-dimensional ablation of \(\delta_1\) (Figure 9). This module is central to hallucination mitigation but mathematically shallow and somewhat brittle.  
   These issues reduce confidence in the rigor and robustness of the CPF design; they should be clarified and formalized.

3. **Heavily heuristic and multi-parameter design in GCGO with limited principled justification.**  
   The generative consistency loss \(\mathcal{L}_{GC}\) (Equations (12)–(19)) has several moving parts: an L2 difference map, Gaussian-blur-based local statistics, an adaptive threshold \(T(u,v)\) with coefficient \(\delta_2\), morphological operations with threshold \(\delta_3\), plus VGG-based LPIPS and a reconstruction term with mixing weight \(\lambda\) and top-level weight \(\alpha\), combined again with another weight \(\beta\) in the overall objective (Equation (20)).  
   While the paper does provide ablations on \(\delta_2,\delta_3,\beta\) and reports that performance is not wildly unstable (Figures 10–12), there is no principled analysis of how these masks interact with the diffusion model’s errors, or whether the mask systematically excludes or includes specific semantic regions (e.g., shiny highlights, thin structures). This read as a large collection of post-hoc heuristics that happen to work empirically, rather than a well-founded design. For a generative-completion method claiming to control hallucination, more careful reasoning or statistics about the mask’s behavior would strengthen the contribution.

4. **Dependence on a large pre-trained I2V diffusion model with insufficient analysis of cost and deployment constraints.**  
   Although Appendix C and Table 10 show a rough comparison of training time and memory on LLFF (e.g., 40 minutes and 4 GB for GenCoGS vs 30 minutes / 3 GB for BinoGS and much lighter for plain 3DGS), this comparison is very shallow:  
   - It does not isolate the cost of diffusion sampling vs 3DGS optimization (e.g., number of diffusion steps per pseudo view, number of pseudo views per iteration, etc.).  
   - It is unclear whether the I2V model is used at full resolution and how much GPU memory/time is consumed by it specifically.  
   - There is no discussion of scalability to higher resolutions or larger scenes, nor of whether the method can run on more modest hardware common in research labs.  
   Given that the method’s *core* contribution over existing 3DGS baselines is precisely the addition of a heavy I2V module, the paper should provide a more honest and quantified analysis of the overhead and its practical implications.

5. **Missing and under-discussed recent few-shot GS works.**  
   The related work section focuses on RegNeRF / SparseNeRF / FreeNeRF and a handful of GS-based methods (FSGS, DNGaussian, BinoGS, CoherentGS, IPSM, ReconFusion, CAT3D, etc.), but several very relevant 3DGS few-shot NVS variants are not cited or discussed at all (see “Potentially Missing Related Work” below). Many of these specifically target better scene completion or structural consistency under few views (e.g., self-ensembling, depth-aware GS, structure-consistent matching).  
   The lack of comparison or discussion makes it harder to judge whether GenCoGS is genuinely advancing the state of the art relative to *all* current GS-based few-shot methods, or just relative to the particular subset included in experiments.

6. **Limited analysis of failure cases and ambiguity in under-determined regions.**  
   The paper repeatedly invokes the analogy to human imagination, but in practice, few-shot NVS is under-determined: multiple plausible completions may exist for unseen regions. The method seems to implicitly rely on the diffusion prior’s biases, but there is no discussion or visualization of “plausible but wrong” completions, or of cases where the generative prior contradicts the actual scene geometry/appearance.  
   Figures 5–7 show successful qualitative examples, but they do not show failure modes or mismatched completions. This omission matters because the claim is not only that the method improves metrics, but also that it achieves “accurate and coherent scene completion”. Without counterexamples or a more nuanced discussion, that claim feels overstated.

7. **Ambiguity about the training and supervision of the CPG module.**  
   Section 3.1.1 states that an “end-to-end complementary point generation module” is designed inspired by Yu et al. (2021b), using a DGCNN + Transformer encoder/decoder and FoldingNet. However, it is unclear:  
   - On what data this module is trained (synthetic point clouds? SfM outputs with GT meshes?),  
   - What is the loss function (Chamfer / EMD between ground-truth dense point clouds and generated points?),  
   - Whether it is pre-trained once and frozen, or trained jointly with each scene; if pre-trained, what domains and whether there is any domain gap to LLFF/DTU/Shiny scenes.  
   The Appendix gives a Chamfer-distance comparison in Table 8 but not the training regime of CPG itself. The lack of detail undermines reproducibility and makes it difficult to assess whether the reported gains stem from a well-trained completion model or from ad-hoc heuristics.

8. **Some notational and expository rough edges.**  
   - In Equation (1), the positional encoding \(PE(c_i)\) is added to the feature from DGCNN, but its form is not described (Fourier features? learned embeddings?), which affects implementation.  
   - In Equation (4), \(P_i' = \mathcal{H}(c_i')\) is described as “neighboring points centered at \(c_i'\)”, but neither the number of points per patch nor their spatial arrangement are specified.  
   - In Equation (10), the notation \(\mathbb{E}[z_0 \mid z_t, F_c, I_p]\) inside the denoising function \(p_\theta\) is non-standard; typical diffusion formulations use a noise-prediction or score network, not an explicit conditional expectation. While this can be heuristic, it would be helpful to clarify the actual LDM formulation being used (especially since the Preliminary section in Appendix A describes a different style of objective in Equation (23)).  
   These are not fatal, but they cumulatively raise the barrier for faithful re-implementation.

9. **Ethical and broader impact discussion is absent.**  
   The method leans heavily on a powerful generative prior that can hallucinate content in under-observed regions. While this is standard in NVS, it has implications for downstream uses (e.g., misinterpreting reconstructed scenes as ground truth). The paper does not discuss any such concerns, e.g., how to communicate uncertainty or detect hallucinated content. Given the explicit focus on generative completion, a short discussion here would be appropriate.

## Potentially Missing Related Work

The following directly relevant works on few-shot or sparse-view 3DGS-based NVS appear to be missing from the references and discussion:

1. **Zhao et al., “Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis”, 2024.**  
   - This work addresses overfitting and robustness in few-shot GS via a self-ensembling strategy. It is directly comparable as a 3DGS-based, few-shot NVS method with a different regularization/completion mechanism.  
   - It should be cited and discussed in Section 2 (“Few-shot Novel View Synthesis”), and, if feasible, added as a baseline in Tables 1, 2, and 7; at a minimum, the authors should qualitatively compare the conceptual difference between self-ensembling and generative completion.

2. **Kumar & Vats, “Few-shot Novel View Synthesis using Depth Aware 3D Gaussian Splatting”, 2024.**  
   - This paper introduces depth-aware enhancements for 3DGS in few-shot settings, tackling ambiguity in under-observed regions using depth priors.  
   - It is highly relevant to GenCoGS’s focus on improving scene completion under sparse views and should be added in Section 2 (and possibly as another depth-based baseline in experiments).

3. **Peng et al., “Structure Consistent Gaussian Splatting with Matching Prior for Few-shot Novel View Synthesis”, 2024.**  
   - This method uses matching priors to enforce structure consistency in few-shot GS, very close in spirit to GenCoGS’s goal of reducing structural artifacts and floating Gaussians.  
   - It should be discussed alongside FSGS, DNGaussian, BinoGS, and CoherentGS in Section 2, with a clear comparison of how GenCoGS differs in terms of using generative priors versus matching priors.

4. **Yin et al., “FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training”, 2024.**  
   - FewViewGS addresses sparse-view NVS with matching-based constraints and a specialized multi-stage training scheme for GS.  
   - This is another core 3DGS few-shot paper that should be cited and compared in the related work section, and ideally in experiments if code or metrics are available.

Including and discussing these works would present a more complete picture of the current landscape in few-shot GS and help position GenCoGS’s contributions more precisely.

## Questions

1. **Clarification of Equation (5) and CPF implementation.**  
   - Please clarify the intended mathematical definition of \(p_{i,k}\) in Equation (5). Is this a typo, and should it be something like \(p_{i,k} = \operatorname{NN}_k(p_i', \mathbf{P}_0 \cap t_i)\)?  
   - Practically, how is \(\mu(\mathbf{P}_0)\) computed for large point clouds? Is it approximated via subsampling? A brief algorithmic description or pseudo-code for CPF would be very helpful.

2. **Training regime for the CPG (DGCNN+Transformer+FoldingNet) module.**  
   - On what dataset(s) and supervision signals is CPG trained? Are there ground-truth dense point clouds or meshes used?  
   - Is CPG trained once offline and then frozen, or is there any per-scene adaptation? If it is pre-trained, did you observe performance degradation on scenes whose geometry materially differs from training data?

3. **Quantitative analysis of the confidence mask behavior.**  
   - Can you provide statistics about how much area (percentage of pixels) \(\hat{M}_r\) typically selects per pseudo view, and how that varies over time or across datasets?  
   - Have you observed systematic failure modes where the mask either over-suppresses true structures (e.g., specular highlights in Shiny) or fails to suppress egregious hallucinations? Some representative visualizations of \(M_r\) and \(\hat{M}_r\) overlaid with pseudo views (beyond Figure 4) would be informative.

4. **Detailed computational breakdown of GCGO.**  
   - Could you provide per-scene numbers for the number of pseudo views generated, diffusion steps per view, and the ratio of diffusion time to 3DGS optimization time?  
   - How does performance and runtime change if you reduce the number of pseudo views or steps (e.g., trading off some PSNR for faster training)?

5. **Behavior in extreme sparsity / domain shift.**  
   - Have you tried 1–2 view regimes or scenes with significantly different appearance (e.g., outdoor natural scenes if trained mostly indoors)? How does GenCoGS behave when the diffusion prior is clearly mismatched to the scene distribution?  
   - Any insight into whether GenCoGS “hallucinates” more aggressively in those cases, and whether the CPF and \(\mathcal{L}_{GC}\) still help?

Clarifications and additional analyses on these points could substantially increase confidence in the method’s robustness and conceptual soundness.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The method is empirically strong and reasonably well engineered, but several central components (CPF equation, CPG training, mask design) lack rigorous specification or analysis, and diffusion usage is highly heuristic.

## Presentation Rating

3: good.  
The paper is generally clear, with well-structured sections and informative figures/tables, but there are notational inconsistencies (e.g., Equation (5)), missing training details for key modules, and limited discussion of failure cases.

## Contribution Rating

2: fair.  
The combination of point-cloud completion, generative pseudo-view completion, and heuristic hallucination suppression yields solid performance gains, but the conceptual novelty over recent diffusion- and GS-based few-shot NVS work is moderate, and positioning versus several directly related GS methods is incomplete.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper presents a well-engineered combination of generative point-cloud and view completion around 3DGS that achieves consistently strong quantitative and qualitative results across standard few-shot NVS benchmarks. However, the contribution feels more incremental than the framing suggests, several important recent GS-based few-shot works are missing from the comparison, and there are non-trivial issues in the mathematical formulation and specification of key components. With clearer theory, better justification of heuristics, and more complete positioning, this could become a strong contribution, but in its current form it falls slightly short of the bar I would expect for ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with NeRF/3DGS-based few-shot NVS and diffusion-guided reconstruction, and I carefully checked the core equations and experimental design, though I did not re-implement the method.