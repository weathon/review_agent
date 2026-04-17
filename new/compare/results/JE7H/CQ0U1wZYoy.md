---
job_id: 4d7e1fb1-017d-4ea7-bd7d-6ce6bb17ab4d
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: CQ0U1wZYoy.pdf
paper: Seeing Through the Prism: Compound & Controllable Restoration of Scientific Images
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper presents a conditional diffusion framework with contrastive representation learning for controllable image restoration, evaluated on multiple scientific domains. This squarely fits ICLR topics in generative models, representation learning, and applications to physical/biological sciences.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Methods, Experiments / Results, Discussion, Conclusion) are present and written in clear English. The work proposes a nontrivial method, includes multiple quantitative tables (Tables 1–4) and qualitative figures, and compares against strong baselines. I do not see fatal theoretical or experimental flaws that would justify desk rejection, although there are important issues to raise in the full review.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions aimed at automated reviewers, or suspicious formatting in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces PRISM, a conditional diffusion framework for restoring scientific and environmental images that suffer from compound degradations. The core idea is to fine‑tune a CLIP image encoder with a weighted contrastive objective that organizes degradations and their mixtures into a compositional latent space, then use these embeddings together with text prompts to condition a latent diffusion model for joint and controllable restoration. The authors construct a mixed‑degradation benchmark and evaluate PRISM on synthetic mixtures, zero‑shot real composite distortions, and several downstream scientific tasks, arguing that selective, prompt‑controlled restoration can improve scientific utility over fully automatic “fix everything” pipelines.

## Strengths

1. **Clear focus on compound degradations and controllability, with strong empirical support.**  
   - Table 1 (MDB results) shows that PRISM substantially outperforms both all‑in‑one CNN/Transformer baselines (e.g., PromptIR) and recent diffusion‑based methods (DiffPlugin, MPerceiver, AutoDIR) on mixed degradations: +1.24–2 dB PSNR over the best diffusion baseline and better LPIPS/FID. These are competitive baselines and the gains appear consistent across metrics.  
   - Figure 3 usefully breaks down performance as the number of distortions increases. It shows that when moving from single to four distortions, PRISM trained with composite supervision degrades much less than AutoDIR and the primitive‑trained PRISM variant, directly substantiating the claim that compound‑aware supervision improves robustness to complex mixtures.

2. **Well‑motivated representation learning component with explicit compositional objective.**  
   - The fine‑tuning of CLIP’s image encoder via the weighted contrastive loss in Section 3.2 is conceptually interesting: the Jaccard‑based weights \(w_{jk}\) encode overlap between distortion sets, and the loss is designed so that mixtures lie “between” their primitive distortions in the latent space. This is more principled than simply tagging images with a global “degraded” label and gives a concrete representation‑learning contribution.  
   - The added “quality regularizer” \(\mathcal{L}_{\mathrm{qual}}^{(j)} = \sum_{c\in d^{(j)}} \hat{p}(c \mid e_{\text{clean}})\) is a simple but thoughtful way to keep clean embeddings anchored as distortion‑free, addressing a common failure mode of adapting CLIP to quality/degradation signals.

3. **Compelling evaluation on real, unseen composite degradations.**  
   - Table 2 shows consistent zero‑shot gains on UIEB (underwater), POLED (under‑display camera), and ThapaSet (fluid lensing). PRISM exceeds the strongest diffusion baseline (MPerceiver/AutoDIR) by about 1 dB PSNR on all three datasets and improves SSIM and LPIPS.  
   - Figure 5 qualitatively illustrates a realistic coral reef use case: an automated pass first identifies high‑level distortions (“warp + color + motion blur”), then experts iteratively apply prompts (“unwarp”, “fix coloring”, “unblur”). This example makes the controllability story concrete and suggests the method is not narrowly tuned to synthetic toy settings.

4. **Downstream, task‑centric evaluation that highlights why controllability matters.**  
   - Table 3 is a strong part of the paper: it quantifies how full vs selective restoration affects downstream accuracy or mIoU on remote sensing, camera traps, microscopy, and urban scenes using off‑the‑shelf task models. In camera traps, microscopy, and urban scenes, selective restoration significantly outperforms full automatic restoration (with p‑values < 0.05).  
   - Figure 6 visually connects this to microscopy: super‑resolution alone recovers more clathrin‑coated pits (higher mIoU), while additional denoising suppresses faint, biologically meaningful structures (highlighted by red circles), exactly illustrating the phenomenon discussed in the text. Table 4 then numerically shows that segmentation prefers super‑resolution whereas fluorescence quantification prefers denoising, reinforcing the “no single best restoration” argument.

5. **Good integration of figures with the narrative and clear high‑level pipeline.**  
   - Figure 1 gives a helpful overview of PRISM’s intended usage scenarios (mixed degradation removal, handling unseen composites, controllable prompting) across different domains, which frames the contributions early.  
   - Figure 2 clearly depicts the two‑stage pipeline: (1) the CLIP image encoder is fine‑tuned to organize mixtures vs primitives; (2) the diffusion model consumes both the image embedding and text embedding, and an SCPM block refines the final output. This makes the architecture and conditioning path easy to follow.

6. **Breadth of domains and mixed‑degradation benchmark.**  
   - The synthetic dataset uses 2M images from diverse scientific / environmental sources (ImageNet, Sen12MS, iWildCam, underwater, Cityscapes, BioSR, MRI, astronomy), with up to three degradations per image. Combined with the Mixed Degradations Benchmark (MDB) and the custom Rooftop Cityscapes dataset, this goes beyond typical “rain/haze” benchmarks and gives some credibility to the claim that PRISM is generally applicable across domains.

7. **Reproducibility and implementation context.**  
   - The paper states that all baselines are trained on the same primitive distortions, and that code, datasets, and evaluation protocols are released. The choice of Stable Diffusion v1.5 and CLIP makes it straightforward for other researchers to reproduce or extend the work.

## Weaknesses

1. **Key mathematical component (weighted contrastive loss) is under‑specified and somewhat internally inconsistent with its stated goal.**  
   - The Jaccard‑based weight is defined as  
     \[
     w_{jk} = \exp\left(1 - \frac{|d^{(j)}\cap d^{(k)}|}{|d^{(j)}\cup d^{(k)}|}\right),
     \]
     and used only in the denominator of \(\mathcal{L}_{\text{ctr}}^{(j)}\). For identical distortion sets, the Jaccard index is 1, giving \(w_{jk} = e^{0} = 1\); for disjoint sets it is 0, giving \(w_{jk} = e^{1} \approx 2.72\). Thus dissimilar variants are *upweighted* as negatives. The text around Eq. (contrastive loss, Page 5) says the goal is that “an image degraded by haze+rain should be more similar to haze-only and rain-only images than to unrelated distortions such as noise.”  
     - While giving dissimilar distortions larger negative weight and similar mixtures smaller weight is a plausible heuristic (similar mixtures are less strongly repelled), this subtle logic is never explained, and no alternative like directly *downweighting* or removing negatives that share primitives is considered.  
     - Moreover, the loss is not symmetric: \(w_{jk}\) appears only in the denominator, and there is no corresponding weighting of the positive term. It would help to clarify whether this loss is symmetric in practice, how gradient magnitudes change as overlap varies, and whether alternative formulations (e.g., explicitly pulling mixtures toward the *average* of primitive embeddings) were tested.  
   - More critically, the classifier used in \(\hat{p}(c \mid e_{\text{clean}})\) for \(\mathcal{L}_{\text{qual}}\) is not specified in the main text. The paper never states whether this is a separate head on top of CLIP, what loss is used to train \(\hat{p}\), and whether the same classifier is also used for the automated distortion prediction in Section 3.3. This missing detail is important because \(\mathcal{L}_{\text{qual}}\) is part of the core objective and the automated restoration pipeline depends on accurate multi‑label distortion prediction.

2. **Limited ablation and quantitative analysis of the representation component in the main paper.**  
   - The paper repeatedly attributes gains to “compound‑aware supervision” and “contrastive disentanglement,” but most supporting evidence is pushed to the appendix. In the main text we only see:  
     - Figure 3, which compares PRISM with/without composite training but does not isolate the effect of the Jaccard‑weighted loss vs a simpler contrastive loss or a purely classification‑based training of CLIP.  
     - Figure 4, which shows PSNR for sequential vs composite prompting with three encoder variants (pretrained, primitive‑aware, compound‑aware CLIP). This is useful, but does not show an ablation where the diffusion model is conditioned on naive one‑hot primitive labels or standard CLIP embeddings to isolate how much of the gain is due to the geometric structure rather than just having any degradation‑aware embedding.  
   - Without clearer ablations in the main paper (not just in Appendix E) that systematically switch off: (a) training on composite vs primitive degradations, (b) Jaccard weighting vs unweighted negatives, and (c) the \(\mathcal{L}_{\text{qual}}\) term, it is difficult to assess how critical each design is relative to simpler alternatives.

3. **Synthetic degradation pipeline and domain shift are not analyzed deeply enough.**  
   - The entire MDB training and evaluation is based on synthetic degradations composed up to three at a time, while the downstream tasks and zero‑shot benchmarks involve real, complex, physically grounded distortions. Although Table 2 and the downstream results in Tables 3–4 show promising generalization, there is no quantitative breakdown of how well the synthetic procedure approximates real degradations.  
   - For example, we never see statistics on the distribution of degradation intensities or co‑occurrence patterns (beyond referencing Table 9 in Appendix E). It would be helpful to see how performance on MDB correlates with performance on UIEB / POLED / ThapaSet across models, or whether some distortions are systematically mis‑handled when moved from synthetic to real. This matters because the main claim is that a compositional latent design enables robust zero‑shot generalization, but the bridge from synthetic mixtures to real physics is not fully examined.

4. **Downstream evaluation, while valuable, leaves some important gaps in methodology and interpretation.**  
   - Table 3 uses pretrained task models and reports mean ± std over 3 seeds, but it is unclear what is being re‑seeded (restoration randomness, task model finetuning, or selection of images). More precise description of the resampling protocol is needed to trust the p‑values.  
   - For remote sensing, the authors note that full restoration slightly outperforms selective restoration because “removing only clouds leaves images under‑illuminated and hazy.” However, no concrete example or figure is provided to illustrate a failure of selective restoration, analogous to Figure 6 for microscopy. A qualitative figure for one of the non‑microscopy domains would strengthen the argument that controllability matters *and* that failures are understandable to domain experts.  
   - Restoration variants in BioSR (Table 4) are treated almost like hand‑crafted conditions (“Denoise”, “Super‑Resolution”, “Combined”), but the text does not describe exactly what prompts correspond to these settings and whether their strength/intensity is controlled. For instance, is “Denoise” a prompt like “remove photon noise only” or are parameters adjusted? This affects how generalizable these insights are to other microscopy tasks.

5. **Some baselines and recent related work on diffusion‑based compound restoration are missing.**  
   - The related work section covers AutoDIR, MPerceiver, PatchDiffuser, AllRestorer, OneRestore, etc., but omits several very closely related recent diffusion‑based restoration methods that explicitly address mixed degradations or prompt‑guided restoration:  
     1. **Lyu et al., 2025, “Step-Calibrated Diffusion for Biomedical Optical Image Restoration”** – focuses specifically on biomedical image restoration with diffusion models under complex degradations, which is highly relevant to the microscopy and MRI aspects of this paper. It should be discussed in Section 2.1 and compared conceptually against PRISM’s CLIP‑based conditioning.  
     2. **Yu et al., 2024, “Universal Image Restoration with Text Prompt Diffusion”** – directly tackles universal text‑prompted restoration. It should be compared in Section 2.3 and ideally added as a baseline or at least a qualitative comparison given its similarity to PRISM’s prompt‑guided setup.  
     3. **Luo et al., 2024, “Refusion: Accelerated Diffusion Model for Image Restoration”** – proposes an efficient diffusion‑based restoration model; efficiency and quality trade‑offs are relevant to the “competitive runtime” claims in Section 4.2.1 and Appendix E.  
     4. **Zhang et al., 2025, “Unified Image Restoration and Enhancement: Degradation Calibrated Cycle Reconstruction Diffusion Model”** – another unified diffusion approach for multiple degradations, directly comparable to PRISM’s joint restoration.  
     5. **Wang et al., 2025, “All-in-One Image Restoration via Diffusion Models with Degradation Perception and Semantic Enhancement”** – uses degradation perception and semantic cues, conceptually close to PRISM’s CLIP‑based conditioning.  
     6. **Tao et al., 2025, “Joint Conditional Diffusion Model for Image Restoration with Mixed Degradations”** – explicitly studies mixed degradations using a joint conditional diffusion model; this is arguably the most directly comparable to PRISM’s aim and should be discussed side‑by‑side in Section 2.2 / 2.3.  
   - Omitting these makes the method look more unique than it is; the core novelty is more in the particular CLIP fine‑tuning and the scientific downstream framing than in using conditional diffusion for mixed degradations per se.

6. **Controllability is only partially realized; control axes are coarse and global.**  
   - The paper emphasizes “controllable restoration” but control is essentially binary on a per‑primitive basis: remove or do not remove each degradation type, globally across the image. There is no mechanism in the main framework to control *intensity* of removal (how much denoising or how much dehazing) or to restrict corrections to spatial subregions, which are often critical in scientific applications (e.g., selectively denoising background vs foreground structures in microscopy).  
   - The authors briefly acknowledge this in Section 4.2.1: “extending controllability beyond specifying which distortions to remove to their intensity and spatial extent would enable localized restoration.” However, given how central “control” is to the pitch and to Figures 1, 5, and 6, the absence of any partial or graded control is a significant limitation. At minimum, experiments varying prompt strength or combining positive and negative prompts would make the controllability claim more convincing.

7. **Some inconsistencies and typos that obscure details.**  
   - Table 1 and Table 2 list “MPerceover” in the header and “MPerceiver” in the text (Section 2.3); this kind of inconsistency can cause confusion when trying to reproduce baselines.  
   - The training setup for the automatic distortion predictor MLP (Section 3.3) is not described: loss function, label source (primitive vs mixture), class imbalance handling, and whether it shares weights with the quality regularizer classifier are all missing. This makes the “automated restoration” pathway in Figure 5 more of a sketch than a reproducible component.  
   - Figure 3’s y‑axis and legend are somewhat compressed in the main text; while one can infer that the shaded segments represent different numbers of distortions and the \(\Delta\) above each bar is the PSNR drop from 1 to 4 distortions, a clearer caption would help readers interpret the stacked structure without consulting the appendix.

Overall, none of these are fatal, but they collectively reduce confidence in exactly how and why the method works as claimed and limit clarity about its relationship to the fast‑moving diffusion‑based restoration literature.

## Potentially Missing Related Work

1. **Y. Lyu, S. J. Cha, C. Jiang, “Step-Calibrated Diffusion for Biomedical Optical Image Restoration,” 2025.**  
   - Relevance: Diffusion‑based restoration tailored for biomedical imagery under complex degradations, directly overlapping with the microscopy and MRI use cases here.  
   - Suggested integration: Discuss in Section 2.1 (Restoration in Scientific Domains) as a specialized biomedical diffusion baseline, and comment in Section 4.2.1 on how PRISM’s generalist design compares to a domain‑specialized diffusion model.

2. **B. Yu, Z. Fan, X. Xiang, “Universal Image Restoration with Text Prompt Diffusion,” 2024.**  
   - Relevance: Universal, text‑prompted diffusion model for image restoration, very close to PRISM’s prompt‑guided conditioning.  
   - Suggested integration: Add to Section 2.3 as a key prompt‑guided diffusion baseline, and clarify in Section 3.2 how PRISM’s explicit compositional latent structure differs from this work’s text conditioning.

3. **Z. Luo, Y. Zhang, H. Li, “Refusion: Accelerated Diffusion Model for Image Restoration,” 2024.**  
   - Relevance: Proposes efficiency improvements for diffusion‑based restoration under complex degradations.  
   - Suggested integration: Cite in Section 2.3 as an efficiency‑oriented diffusion baseline and compare runtimes in Appendix E’s latency Table 13, clarifying where PRISM sits in the quality–efficiency trade‑off.

4. **H. Zhang, L. Wang, J. Chen, “Unified Image Restoration and Enhancement: Degradation Calibrated Cycle Reconstruction Diffusion Model,” 2025.**  
   - Relevance: A unified diffusion model that handles multiple restoration and enhancement tasks with explicit degradation calibration, conceptually close to PRISM’s joint handling of multiple distortions.  
   - Suggested integration: Discuss in Section 2.2 / 2.3 as a representative unified diffusion framework; clarify whether PRISM’s compositional CLIP geometry offers benefits over degradation calibration.

5. **Y. Wang, X. Liu, P. Zhao, “All-in-One Image Restoration via Diffusion Models with Degradation Perception and Semantic Enhancement,” 2025.**  
   - Relevance: Uses degradation perception and semantic signals inside a diffusion framework to handle multiple degradations; another strong baseline for all‑in‑one diffusion.  
   - Suggested integration: Include in Section 2.2 / 2.3 and compare its “degradation perception” design to PRISM’s vision‑language‑based latent disentanglement.

6. **W. Tao, J. X. Shi, J. F. Qi, “Joint Conditional Diffusion Model for Image Restoration with Mixed Degradations,” 2025.**  
   - Relevance: Directly targets mixed degradations with a joint conditional diffusion model; arguably the most directly comparable method to PRISM’s mixed‑degradation focus.  
   - Suggested integration: Discuss prominently in Section 2.2 as a main prior on mixed degradation diffusion; explain in Section 4.1 why PRISM’s compositional CLIP embedding might yield better controllability or generalization compared to this joint conditioning.

Other listed works in the web context (e.g., about multimodal reasoning, intermediate distribution shaping, data attribution, interpretability for LLM outputs) are conceptually interesting but more tangential; I do not see them as “directly related” in the same sense as the above diffusion‑restoration methods.

## Questions

1. **On the weighted contrastive loss and its alternatives:**  
   - Can you provide a more formal explanation of how the Jaccard‑based weights \(w_{jk}\) affect gradients and why the chosen exponentiation \(w_{jk} = \exp(1 - J)\) is preferable to simpler choices like \(w_{jk} = 1 + \alpha (1 - J)\) or direct downweighting of overlapping mixtures?  
   - Did you experiment with explicitly pulling compound distortions toward the *average* of their primitive embeddings (e.g., an auxiliary loss \(\|e_{\text{dist}}^{\text{haze+rain}} - \tfrac{1}{2}(e_{\text{dist}}^{\text{haze}} + e_{\text{dist}}^{\text{rain}})\|_2^2\))? How did this compare?

2. **On the distortion classifier and automated restoration pipeline:**  
   - Please describe in detail the architecture and training procedure for \(\hat{p}(c \mid e)\): what loss is used, what labels are assigned to mixtures, and how is class imbalance handled?  
   - Is the classifier used for \(\mathcal{L}_{\text{qual}}\) in Eq. (quality regularizer) the same as the one used by the automated restoration MLP in Section 3.3, or are they separate heads? Clarifying this would help understand how much of the downstream performance depends on multi‑label classification accuracy.

3. **On ablation and comparisons to simpler conditioning baselines:**  
   - Could you provide, either in the rebuttal or final version, a quantitative comparison where the diffusion model is conditioned on: (a) unmodified CLIP image embeddings, (b) purely primitive‑label one‑hot vectors, and (c) your compound‑aware CLIP embeddings, under the same MDB and zero‑shot settings? This would isolate the value of the representation learning component.  
   - Similarly, an ablation disabling \(\mathcal{L}_{\text{qual}}\) would clarify whether anchoring the clean embedding meaningfully affects controllability or just slightly alters the latent geometry.

4. **On controllability granularity:**  
   - Did you attempt to modulate the “strength” of restoration by, for example, scaling or mixing positive and negative prompts (“slightly dehaze”, “strongly dehaze”) or interpolating between prompt embeddings? Any evidence here would make the controllability story stronger and might suggest future extensions toward intensity/spatial control.  

5. **On reproducibility and experimental protocol:**  
   - For the downstream experiments in Table 3 and the microscopy results in Table 4, could you detail the exact evaluation splits, number of images, and seeding protocol used to compute means and p‑values?  
   - For MDB, how many images have exactly 1, 2, and 3 distortions at test time? This would help interpret Figure 3’s breakdown.

Answers and, where possible, small additional ablations could increase my confidence in both the representation component and the practical robustness of the method.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core methodology (contrastive CLIP adaptation + conditional diffusion) is coherent and well matched to the problem, and the experimental evidence across Tables 1–4 and Figures 3–6 broadly supports the main qualitative claims. However, some mathematical and methodological details (weighted contrastive loss behavior, distortion classifier training, ablation coverage) are under‑specified, which prevents a higher soundness rating.

## Presentation Rating

3: good.  
The paper is generally well written and well structured, with informative figures (especially Figures 1, 2, 5, 6) and clear motivation. Some important implementation details and related works are omitted or pushed to the appendix, and there are minor inconsistencies/typos, but overall the exposition is solid.

## Contribution Rating

3: good.  
The paper makes a meaningful contribution by combining a compositional, CLIP‑based latent geometry with diffusion for controllable mixed‑degradation restoration, and by convincingly framing the importance of task‑dependent, selective restoration in scientific domains via Tables 3–4 and Figure 6. While not entirely unique in using conditional diffusion for mixed degradations, the representation design and downstream evaluation are sufficiently substantial for ICLR.

## Overall Rating

8: Accept, good paper (poster).  
The work presents a well‑motivated and technically reasonable approach to compound and controllable restoration, supported by strong quantitative improvements over a wide range of baselines (Tables 1–2) and thoughtful downstream evaluations (Tables 3–4, Figures 5–6) that highlight an under‑explored but important point: more restoration is not always better for scientific tasks. There are gaps in ablation, missing very recent related work, and some under‑specified mathematical and implementation details, but these seem addressable in revision and do not undermine the main claims. I recommend acceptance as a solid, impactful contribution to diffusion‑based restoration and scientific imaging.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion‑based restoration, CLIP adaptation, and multi‑degradation benchmarks, and I carefully checked the main derivations and experimental tables. Some implementation details are necessarily inferred from context or delegated to the appendix, so there is minor residual uncertainty, but not enough to affect the overall evaluation.