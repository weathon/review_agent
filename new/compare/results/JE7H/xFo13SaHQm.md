---
job_id: e7be6a33-7fe1-47bb-84b0-2c7e3236db35
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: xFo13SaHQm.pdf
paper: WithAnyone: Mitigating Copy-Paste Artifacts in Identity-Consistent Image Generation via MultiID Contrastive Training
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on generative diffusion models, identity‑preserving image generation, a new dataset, and an evaluation benchmark, which are squarely within ICLR’s scope (representation learning, generative models, datasets/benchmarks).

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, method, experiments, results, conclusion) are present and reasonably complete. The work is technically non‑trivial, clearly written in English, and supported by substantial experiments; no fatal methodological or theoretical flaws are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any hidden prompts, instructions to LLM reviewers, or other manipulative content in the paper.

---

# Expected Review Outcome:

## Summary

The paper tackles identity‑consistent text‑to‑image generation, focusing on the common failure mode where models “copy‑paste’’ the reference face instead of generating natural variations. The authors introduce MultiID‑2M, a large‑scale multi‑person dataset with paired references, and MultiID‑Bench, a standardized benchmark that measures both identity fidelity and copy‑paste artifacts. Building on these, they propose WithAnyone, a FLUX‑based diffusion model trained with a four‑phase pipeline and two ID‑supervision losses (a GT‑aligned identity loss and a contrastive loss with extended negatives), which achieves strong identity similarity while substantially reducing copy‑paste on the proposed benchmark and several external evaluations.

## Strengths

1. **Clear problem formulation and useful metric.**  
   The paper articulates the “copy‑paste’’ artifact very clearly and formalizes it via the Copy‑Paste metric \(M_{\text{CP}}\) in **Equation (2)**. This metric compares angular distances to the reference and ground truth, normalized by the reference–GT difference. It is conceptually well aligned with the stated goal (favor variation when GT differs from the reference) and, as shown in **Table 1** and **Table 2**, allows disentangling “high similarity because of copying” from “high similarity because of true identity understanding”. The correlation analysis with human judgments in **Table 7** (Appendix H) and the trade‑off plots in **Figure 5** provide convincing evidence that \(M_{\text{CP}}\) captures a meaningful perceptual dimension.

2. **Substantive dataset and benchmark contribution.**  
   MultiID‑2M fills a real gap: large‑scale, multi‑person images with multiple references per ID. The data construction pipeline in **Figure 3** and Section 3 describes a fairly rigorous process (ArcFace clustering, DBSCAN filtering, multi‑name retrieval, matching with cosine threshold, plus several automated filters). **Table 4** and **Figures 13–14** position the dataset quantitatively against prior multi‑ID resources, highlighting size, number of paired group photos, and per‑ID reference counts. MultiID‑Bench is similarly well specified, with a clear split protocol, no identity overlap with training, and explicit definitions for identity blending (**Equation (8)**), CLIP‑based quality metrics, and thresholds for reporting \(M_{\text{CP}}\). This appears quite valuable for the community.

3. **Reasonable architectural design tightly linked to the objective.**  
   WithAnyone’s architecture, summarized in **Figure 4(a)** and detailed in Appendix E, is a principled extension of FLUX with cross‑attention identity adapters. The use of ArcFace tokens restricted to specific spatial regions via attention masks is a clean way to localize identity injection, and the optional SigLIP branch supports controllable transfer of mid‑level attributes such as hairstyle and accessories. The feature‑injection equation (9) in Appendix E is mathematically straightforward and shows how identity tokens modulate the backbone without overriding the generative prior.

4. **Well‑motivated training objectives and pipeline.**  
   The combination of a flow‑matching diffusion loss (**Equation (3)**), GT‑aligned ID loss (**Equation (4)**), and InfoNCE‑style contrastive loss (**Equation (5)**) is coherent. The GT‑aligned trick in Section 5.1 / Appendix E.1 is a nice practical idea: using GT landmarks to align generated faces during training avoids noisy landmark detection on partially denoised samples and enables applying \(\mathcal{L}_{\text{ID}}\) at all timesteps. **Figure 7** directly visualizes the effect of GT‑aligned vs prediction‑aligned landmarks across noise levels, which is a strong, very concrete piece of evidence. The four‑phase training schedule in Section 5.2 (also summarized in **Figure 3**, bottom row, and quantified in **Table 6**) systematically steps from reconstruction to paired tuning and then style tuning; this staged design is arguably one of the main reasons they can reduce copy‑paste without destroying fidelity.

5. **Comprehensive empirical evaluation on many baselines.**  
   The experimental section covers both single‑ID and multi‑ID settings, and includes a wide range of baselines: general customization models, face‑specific methods, and closed APIs like GPT‑4o. **Table 1** (single‑person) shows that WithAnyone matches or surpasses the best ID‑specific baselines on \(\text{Sim(GT)}\) while substantially lowering \(M_{\text{CP}}\); **Table 2** (multi‑person) demonstrates especially strong gains in \(\text{Sim(GT)}\) over existing multi‑ID methods while keeping copy‑paste and identity blending competitive. The additional comparison on OmniContext in **Table 1(b)** reinforces that WithAnyone remains competitive on an independent benchmark focused on VLM‑judged subject consistency and prompt following.

6. **Qualitative evidence that aligns with the quantitative story.**  
   **Figure 2** illustrates the core motivation by comparing real‑image similarity distributions to those of several methods, visually driving home that many existing customization models collapse to near‑duplicate faces. **Figure 6** and **Figure 9** (main paper and appendix) provide multi‑row grids where WithAnyone often succeeds at changing pose/expression while keeping recognizable identity, whereas InstantID, PuLID, and some general models either copy the reference or drift in identity. The galleries in **Figures 10–12**, as well as robustness illustrations in **Figures 15–16**, are strong and consistent: the method appears to handle low‑quality references and previously unseen non‑celebrity identities reasonably well.

7. **User study and metric validation.**  
   The user study setup in Appendix H, with a clear four‑criterion ranking protocol and an interface example in **Figure 20**, is reasonably thorough for a generative paper. **Figure 8** and **Table 7** demonstrate that human rankings prefer WithAnyone across identity similarity, copy‑paste, prompt adherence, and aesthetics, and that the proposed copy‑paste metric correlates moderately with human judgments. This is an important sanity check given that the key contribution lies in redefining what “good identity preservation’’ means.

## Weaknesses

1. **Overstated performance claims and somewhat ambiguous reading of Table 1 / Figure 5.**  
   In Section 6.1 and earlier, the paper states that WithAnyone “achieves the highest face similarity with regard to GT while maintaining a markedly lower copy‑paste score”. However, in **Table 1(a)**, InstantID attains \(\text{Sim(GT)} = 0.464\), slightly higher than WithAnyone’s 0.460 on the single‑person subset, while WithAnyone has much lower \(M_{\text{CP}}\). **Figure 5(a)** also places InstantID slightly to the right of WithAnyone on the x‑axis. A more accurate phrasing would be that WithAnyone achieves *near‑maximal* \(\text{Sim(GT)}\) on the Pareto frontier with significantly reduced copy‑paste, rather than the absolute highest similarity. Although this is a minor numeric discrepancy, it weakens the perception of rigor; the authors should revise language and provide formal Pareto‑front analysis if they want to claim a “break” of the trade‑off.

2. **Copy‑Paste metric formulation needs deeper analysis and limitations.**  
   While the angular‑distance formulation in **Equation (2)** is intuitive, the paper does not analyze its sensitivity to the reliability of face embeddings or to small \(\theta_{tr}\). When the reference and GT are very similar (\(\theta_{tr} \approx 0\)), \(M_{\text{CP}}\) becomes ill‑conditioned even with \(\varepsilon\), and small noise in \(\theta_{gt}\) and \(\theta_{gr}\) can flip the sign. The authors partially mitigate this by only ranking copy‑paste for cases above a similarity threshold (noted under **Table 1** and **Table 2**), but they never quantify how many test cases are filtered out or how robust the metric is under embedding perturbations. Since this metric is a central contribution and used to argue trade‑offs in **Figure 5**, a more rigorous empirical or theoretical analysis (e.g., calibration curves, ablations with different face encoders, reliability across pose/expression) is necessary.

3. **Some mathematical inconsistencies and loose definitions around the contrastive loss.**  
   The InfoNCE loss in **Equation (5)** is written as  
   \[
   \mathcal{L}_{\mathrm{CL}} = -\log \frac{\exp(\cos(\mathbf{g},\mathbf{t})/\tau)}{\sum_{j=1}^M \exp(\cos(\mathbf{g},\mathbf{n}_j))/\tau},
   \]
   which appears to have a misplaced division by \(\tau\) in the denominator (it should be inside the exponent). More importantly, Section 5.1 says this loss “pulls the generated image closer to its reference images” but the positive in Eq. (5) is \(\mathbf{t}\), the GT embedding, not a reference embedding \(\mathbf{r}\). Given the training setup in Phase 3 where reference and target images differ, this mismatch matters: is \(\mathbf{t}\) the GT face in the group photo, a randomly chosen reference of the same identity, or something else? Appendix F.1 describes sampling negatives from the reference bank but leaves the exact batch‑construction and balance between within‑batch vs external negatives quite vague. Since the efficacy of \(\mathcal{L}_{\mathrm{CL}}\) is central to their claim of improved identity learning (see **Table 3**, row “w/o Ext. Neg.” and **Figure 17**), the loss definition and sampling strategy should be cleaned up with precise notation and corrected formula.

4. **Limited ablation depth and some confusing trends.**  
   The ablation in **Table 3** is appreciated but fairly thin relative to the complexity of the training pipeline. For instance:
   - Removing the paired Phase 3 increases \(M_{\text{CP}}\) from 0.161 to 0.239 but *also* *increases* \(\text{Sim(GT)}\) from 0.405 to 0.406. The text in Section 6.3 claims Phase 3 “reduces copy‑paste without diminishing similarity,” which is only marginally true and not discussed. It would be helpful to plot the full frontier (similarity vs \(M_{\text{CP}}\)) for different Phase‑3 mixing ratios to show the trade‑off.
   - “w/o Ext. Neg.” in **Table 3** reduces \(\text{Sim(GT)}\) from 0.405 to 0.368 but also significantly lowers \(M_{\text{CP}}\) from 0.161 to 0.074. This suggests that extended negatives both strengthen identity and push the model toward more reference‑like embeddings, mildly increasing copy‑paste. This subtlety is important conceptually but not unpacked in the discussion.
   - The FFHQ‑only baseline is extremely weak (0.224 \(\text{Sim(GT)}\)) but it conflates data scale, diversity, and architecture/training hyperparameters. Given how central the dataset is, a more controlled ablation (e.g., similar scale subsets of MultiID‑2M vs FFHQ) would better isolate the effect of pairing and multi‑ID structure.

5. **Benchmark choices and potential bias in evaluation metrics.**  
   The evaluation pipe uses ArcFace, AdaFace, and FaceNet, and the authors average their cosine similarities (Appendix F.4, **Table 5**). Yet, ArcFace is also heavily used in dataset construction (clustering, retrieval) and in the training losses (\(\mathcal{L}_{\mathrm{ID}}\) and \(\mathcal{L}_{\mathrm{CL}}\)). Even though the averaging across models partially mitigates this, there is still a risk that the model implicitly optimizes toward the specific embedding geometry of ArcFace. A more neutral, self‑supervised identity metric (e.g., DINOv2‑based) or a second‑order metric like verification accuracy on an independent face dataset would strengthen claims of general identity preservation beyond the training encoder.

6. **Dataset and evaluation scope are narrower than the narrative suggests.**  
   Despite the strong engineering effort, MultiID‑2M only contains publicly known figures filtered via CC‑licensed web photos, and **Figure 13(b)** shows a strong concentration on Chinese and US celebrities. While Appendix C acknowledges this, the main text largely frames it as a broadly diverse dataset. For applications like personal photo customization, one might worry about distribution shifts: the non‑celebrity results in **Figure 16** are promising but purely qualitative and limited in count. A small quantitative evaluation on a held‑out non‑celebrity set would give more confidence that neither the dataset nor the learned model are over‑specialized to celebrity appearance statistics.

7. **Baselines and fairness of comparison could be better documented.**  
   Although many baselines are included, the fairness of their setup is only briefly addressed in Appendix F.1. For example, some models (e.g., InstantID, PuLID) are primarily designed for single‑ID conditioning, and others like OmniGen2 or Qwen‑Image‑Edit are general editing systems not explicitly optimized for multi‑ID. It is unclear whether prompt engineering, reference framing, or number of inference steps were tuned equally for all methods or simply left at defaults. Given the central role of **Table 2** and **Figure 6** for multi‑person performance, a more detailed description of how multi‑ID prompts and reference inputs are formatted per method, and how many tries per sample are allowed, is needed to ensure the comparisons are not accidentally biased in favor of WithAnyone.

8. **User study limitations and lack of statistical analysis in the main text.**  
   The user study uses 10 participants over 230 groups, which is fine, but the main paper only shows aggregate bubble plots in **Figure 8**. There is no direct reporting of inter‑rater reliability, confidence intervals on rankings, or even p‑values for differences between methods. Appendix H provides some correlation stats with machine metrics, but not method‑wise significance tests. Since the user study is used to validate both the model and the copy‑paste metric, it deserves more rigorous statistical treatment (e.g., pairwise Wilcoxon tests across methods per criterion, plus reporting many‑comparison corrections).

9. **Ethical considerations could be more concrete with respect to deployment.**  
   The ethics section makes reasonable generic claims and emphasizes CC‑licensed celebrity data and anonymization. However, WithAnyone is specifically about highly realistic identity cloning, and the paper does not discuss how the released model will enforce non‑celebrity usage policies, or what safeguards are actually implemented in the code (e.g., watermarking, fine‑tuned safety filters, or prompts rejection). Given the dual‑use potential, especially shown by **Figures 10–12** and **16**, some more explicit discussion of practical abuse‑mitigation strategies for an open‑source release would be appropriate for a major venue.

## Potentially Missing Related Work

1. **Yuan, L. “AnyPhoto: Multi-Person Identity Preserving Image Generation with ID Adaptive Modulation on Location Canvas,” 2026.**  
   This work specifically tackles multi‑person identity‑preserving generation with explicit location control and ID‑adaptive modulation, which is conceptually very close to WithAnyone’s goal of controllable multi‑ID synthesis with reduced copy‑paste. It should be cited and discussed in Section 2 (“Multi‑ID Preservation”) and also compared conceptually to the attention‑mask and cross‑attention injection schemes described in Section 5 and Appendix E. If available, adding AnyPhoto to **Table 2** and **Figure 5** would materially strengthen the empirical positioning.

If AnyPhoto or similar works already exist with public benchmarks, the authors should discuss whether MultiID‑Bench overlaps or complements those evaluation protocols.

## Questions

1. **Clarification of contrastive loss positives and negatives.**  
   In **Equation (5)** and the surrounding text, is \(\mathbf{t}\) always the embedding of the ground‑truth face in the group image, or is it any reference image of the same identity? If it is the GT face, how is this consistent with the statement that the loss “pulls the generated image closer to its reference images”? Please provide a precise definition and, if necessary, correct the equation.

2. **Sampling strategy and scale for extended negatives.**  
   Appendix F.1 mentions extending negatives to 4096 by drawing from the reference bank. How exactly are these sampled (uniform over IDs, balanced by frequency, etc.), and how many distinct negative IDs and images per batch are typically used? Have you tried alternative strategies like hard negative mining? A small ablation on the negative‑pool construction would clarify how critical this detail is.

3. **Robustness of \(M_{\text{CP}}\) to embedding noise and small \(\theta_{tr}\).**  
   Can you report how many test pairs are filtered out by the similarity thresholds used for copy‑paste ranking in **Table 1** and **Table 2**? Additionally, could you provide an experiment where you artificially add noise to the ArcFace embeddings or swap the encoder (e.g., ArcFace vs AdaFace vs DINOv2) to show the stability of \(M_{\text{CP}}\) rankings across methods?

4. **Non‑celebrity quantitative evaluation.**  
   Beyond the qualitative examples in **Figure 16**, do you have any small‑scale quantitative test on non‑celebrity identities (e.g., staff volunteers or synthetic “unseen” identity datasets) to confirm that both MultiID‑Bench metrics and WithAnyone’s performance generalize beyond celebrity images?

5. **Fairness of multi‑ID baseline configurations.**  
   For methods like OmniGen2, GPT‑4o, UMO, etc., how is the multi‑ID conditioning specified? Are reference faces provided via image prompts, composite canvases, or text descriptions? Are the same numbers and crops of references used as in WithAnyone? Clarifying this would help interpret **Table 2** and **Figure 6**.

6. **SigLIP interpolation and controllability.**  
   Appendix G and **Figure 18** suggest that adjusting the SigLIP weight trades off \(\text{Sim(Ref)}\) and copy‑paste. Can the authors elaborate on how users would control this at inference time? Is \(\lambda\) exposed as a knob, and does it interact with the text prompt in non‑trivial ways?

Author responses that clarify these points, especially around the contrastive loss, the robustness of the copy‑paste metric, and the fairness of baselines, could substantially increase confidence in the paper.

## Flag For Ethics Review

No ethics review needed.  

(There are ethical considerations around identity cloning, but the paper includes a reasonably thorough ethics section and uses CC‑licensed celebrity data; I do not see a specific red‑flag violation requiring formal ethics escalation.)

## Details Of Ethics Concerns

N/A.

## Soundness Rating
3: good.  
The methodology and experiments are largely sound and well supported, with clear architectural details and a thoughtful training pipeline, though some mathematical definitions (contrastive loss, copy‑paste metric stability) and evaluation biases need clarification.

## Presentation Rating
3: good.  
The paper is generally well written and easy to follow; figures (especially **Figures 3–7**) and tables are informative. Some claims are slightly overstated and a few mathematical definitions need cleanup, but overall clarity is above average.

## Contribution Rating
3: good.  
The combination of a new multi‑ID dataset/benchmark, a copy‑paste metric, and a tailored FLUX‑based model represents a solid, multi‑faceted contribution to identity‑consistent generation, even if individual ingredients are evolutions of existing ideas rather than fundamentally new theory.

## Overall Rating
6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work offers a well‑executed and timely package (dataset, benchmark, model) addressing a real pain point in ID‑consistent generation, with extensive experiments and reasonably convincing qualitative and quantitative evidence. However, some core components (contrastive loss formulation, copy‑paste metric robustness, and nuanced interpretation of ablations) need clearer, more rigorous treatment, and the empirical advantage over the strongest baselines is often a Pareto‑style trade‑off rather than a clear domination. With these caveats addressed, the paper would make a valuable addition to ICLR.

## Reviewer Confidence
4: confident.  
I am familiar with diffusion‑based image generation, personalization, and representation learning, and I carefully examined the math and experiments, though I did not attempt to reimplement the method.