## Weaknesses

1. **Methodology for redundant-part detection is heuristic and under-specified.**  
   - The redundant detection pipeline in Section 4.2 hinges on comparing the semantic output of a diffusion inpainting model with the original content and declaring a part redundant if “a significant semantic difference is detected” (described around Eq. (8) and Figure 5).  
   - However, Eq. (8) is mathematically opaque:
     \[
     \{\langle p_k^r, b_k^r\rangle\}_{k=0}^j = \{G(R(p_i^g,b_i^g),p_i^g)\}_{i=0}^n < \tau
     \]
     It conflates sets with inequality, and it is never clearly specified what exactly is being thresholded against \(\tau\) (similarity score? detection probability? IoU?).  
   - The paper does not clearly describe:  
     - how \(p^R\) is computed (is it a textual label from G, a distribution over classes, an embedding similarity score?),  
     - what semantic distance metric is used and how \(\tau\) is chosen or tuned,  
     - how overlapping parts or occlusions are handled (e.g., overlapping arms, hand on face).  
   - Since redundant detection is central to FHAD and to Step I and Step III in HumanCalibrator, this lack of clarity weakens the technical soundness and makes reproduction difficult.

2. **Evaluation of localization quality is indirect and incomplete.**  
   - By design, AIGC Human-Aware 1K annotations for real AIGC videos do **not** include bounding boxes (Section 3 and Figure 3); location quality is instead “evaluated” via repair quality. This means there is no direct quantitative metric for how accurate the predicted bounding boxes \(B^a\) (and redundant \(b^r\)) are on the main AIGC test set.  
   - As a result, detection metrics in Table 1 only measure whether the model correctly predicts **which part type** is abnormal, not **where** it is. For a task explicitly defined as fine-grained detection with bounding boxes (see Task Definition, Eq. (1)), this is a major gap. The only direct localization evaluation is on the synthetic COCO Human-Aware Val (Appendix F), which is less representative and easier.  
   - This limitation should be made much more explicit in Section 5, and a small subset of AIGC Human-Aware 1K with bounding boxes (even if partial) would considerably bolster the claims about fine-grained localization.

3. **Dataset scale and diversity seem limited for such a broad claim.**  
   - AIGC Human-Aware 1K contains only 1,000 frames (Table S3: 649 absent, 158 redundant, 343 normal). This is modest, especially given the variety of possible body poses, clothing, motions, and camera angles in modern AIGC.  
   - All AIGC examples come, as far as the main text states, from the PIKA split of VidProM (Section 3). This raises concerns about generator bias: the anomalies and visual style may be highly specific to this source. The method is only qualitatively shown to generalize to other generators in Figure 9; there is no quantitative evaluation on additional AIGC sources, which would be important to support some of the broader claims (e.g., “across two main-stream domains”).  
   - COCO Human-Aware Val is artificially constructed by masking COCO frames. While useful for controlled absent detection, it does not approximate the often messy visual cues of real generative failures (blending artifacts, subtle deformations, etc.). The absence of a larger and more diverse real-AIGC benchmark limits the strength of the empirical story.

4. **Baselines for the repair stage are thin and somewhat unfair.**  
   - The paper uses only one quantitative baseline for repair quality and visual consistency: a pose-conditioned re-generation that effectively **regenerates the entire image** (Section B.4 and Table 2). This baseline is not optimized for partial inpainting or content preservation, so the extremely poor FID (98.86) and low latent consistency (0.668) are unsurprising.  
   - There is no comparison to more natural baselines for localized repair, such as:  
     - direct use of Stable Diffusion inpainting with coarse prompts but **without** HumanCalibrator’s fine-grained masks,  
     - using a generic segmentation/pose detector to crop and inpaint anomalous regions with heuristics (e.g., remove inconsistent limbs identified by pose).  
   - Without such baselines, it is hard to disentangle how much benefit comes from the FHAD perception modules versus simply using a strong inpainting model with manually curated masks.

5. **Ambiguity and inconsistencies in mathematical notation and loss formulation.**  
   - Eq. (5), the central training objective for AHD, is written as:
     \[
     p(\langle p_i^a,b_i^a \rangle | X_i, I_a) = 
     \prod_{j=1}^{L} p_\theta(x_j \mid X_i, I_{a,<j}, \langle p_i^a,b_i^a\rangle, _{<j}),
     \]
     where \(x_j\) seems to represent tokens of the concatenated instruction and target, but the conditional arguments are syntactically malformed (e.g., the trailing “,\_,<j>” term). It is unclear whether both the part class label and bounding box coordinates are tokenized, and if so how (e.g., normalized numeric tokens, discretized bins, textual phrases like “a hand at [0.3, 0.4, 0.5, 0.6]”).  
   - For Eq. (9) and Eq. (10), the piecewise recursion on \(X_t\) is reasonable, but the indexing of \(\langle p_t, b_t \rangle\) for redundant vs absent parts is confusingly written and could be prone to off-by-one errors in implementation.  
   - These notational issues are not fatal mathematically, but they hinder understanding of the actual learning objective and may mask subtle implementation choices (e.g., bbox tokenization, sampling order) that materially affect performance.

6. **Limited analysis of failure modes and robustness.**  
   - While Appendix G (Figure S5(b)) shows some failure cases, the main text mostly highlights positive examples (Figures 7, 8, 9). There is little analysis of where and why the method breaks down systematically. For example:  
     - performance across variations in pose complexity, occlusions, multiple people, extreme camera angles, or heavy motion blur,  
     - behavior on near-boundary cases (e.g., foreshortened limbs that look “short” but are still anatomically plausible).  
   - Given that AIGC systems evolve rapidly, understanding how brittle the approach is to new artifact patterns would greatly affect its practical significance.

7. **VLM baseline evaluation protocol is somewhat ad hoc and may not be fully convincing.**  
   - For AIGC Human-Aware 1K, the generative VLMs are prompted with fairly specific questions (Appendix B.2), and their outputs are post-processed by GPT‑4o‑mini using a custom prompt (Figure S1). This introduces potential bias from the meta-LLM and from prompt engineering choices.  
   - The main text claims that “even though humans and these VLMs are exposed to similar volumes of normal data, abnormality detection remains markedly easier for human cognition” (Section 5), but that analogy is loose: the underlying training distributions, architectures, and objective functions are quite different. It would be useful to more carefully qualify this statement and more systematically explore prompt variations, temperature, or chain-of-thought strategies before concluding that these models fundamentally lack abnormality perception capability.

8. **Some conceptual mixing of “abnormality detection”, “real-world existence”, and “quality assessment”.**  
   - Section 2 and Figure 2 position FHAD as distinct from AIGC detection and quality assessment, but in practice the notion of “whether the generated human body structure could occur in the real world” straddles both low-level anomaly detection and subjective realism assessment.  
   - For instance, slightly stylized faces or exaggerated limb proportions might technically not be found in real humans but are still visually acceptable; conversely, subtle joint misalignments that are rare but physically plausible may be flagged or ignored. The current taxonomy (head/ear/arm/hand/leg/foot; absent/redundant) does not capture these nuances.  
   - The paper would benefit from more formalization of what is considered “objectively impossible” versus “low likelihood” versus “aesthetic artifact”, perhaps through a discussion of inter-annotator agreement or explicit design choices in AIGC Human-Aware 1K annotation (Section D glosses over how borderline cases were handled quantitatively).

9. **Lack of human evaluation for repair plausibility.**  
   - All repair metrics are automated (CLIP-based scores, FID, latent similarity). While these are useful, they do not necessarily reflect human judgments of anatomical correctness or overall realism.  
   - Given that FHAD is motivated by human perception (“simple for humans, hard for VLMs”), a small-scale user study (e.g., pairwise comparisons between original and repaired images or between HumanCalibrator and baseline repairs on AIGC frames) would significantly strengthen the empirical claims and place the CLIP-based metrics in context.

10. **Reproducibility details are mostly in the appendix and still incomplete.**  
    - Section 5 gives only a high-level description: fine-tuning LLaVA 1.5‑7B for 2 epochs on COCO train with learning rate \(2 \times 10^{-5}\); details of the masking, prompt templates \(I_a\), inpainting prompt templates \(T(\cdot)\), super-resolution model, and thresholding for redundant detection are mostly in Appendix A, and some are still vague (e.g., exact bounding-box expansion margin, selection of \(\tau\)).  
    - For a benchmark-style contribution, a more explicit protocol (including scripts for generating COCO Human-Aware Val, exact text templates, and configuration of GroundingDINO and StableDiffusion2-Inpainting) is essential for future work to build on this; the current description would require substantial reverse engineering.

Taken together, these weaknesses do not undermine the central observation and basic validity of the approach, but they reduce the methodological sharpness and make the work feel more like a strong engineering system paper than a fully polished benchmark + method package that sets a clearly defined and easily reproducible standard.
