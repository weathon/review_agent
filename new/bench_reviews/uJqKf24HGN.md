Now I have enough calibration data. Let me compose the final review.

## Summary

UniCon proposes a unidirectional information flow paradigm for controlling large-scale diffusion models: instead of bidirectional interaction (where the adapter modifies features that flow back through the diffusion model), UniCon makes information flow only from the frozen diffusion model to the adapter, which then produces the final output directly. This eliminates the need to compute and store gradients for the diffusion model during training, yielding substantial VRAM savings (~50%) and training speedups (~2.3×), with the additional benefit of architecture-agnostic applicability to both U-Net and Transformer-based diffusion models.

## Strengths

- **Clear, impactful architectural insight with practical training efficiency gains**: The observation that detaching the adapter from the diffusion model's backward pass eliminates gradient storage is simple, correct, and highly consequential for scaling to large models. Figure 6 provides detailed per-component breakdowns (weight, activation, gradient, optimizer VRAM; forward/backward time) confirming that gradient storage—nearly half of ControlNet's VRAM—is entirely eliminated, and backward propagation time is nearly halved.

- **Comprehensive performance improvements across architectures and tasks**: Table 2 shows UniCon outperforming ControlNet across all five tasks on both DiT and SD U-Net architectures on nearly all metrics (e.g., DiT Canny SSIM 0.5458 vs. 0.4748; SR FID 20.34 vs. 26.43). UniCon-Half, with comparable parameters to ControlNet, still consistently improves performance, supporting the method's genuine value beyond mere parameter scaling.

- **Architecture-agnostic applicability validated on U-Net and Transformer**: Unlike ControlNet, which was designed around U-Net's encoder-decoder structure, UniCon applies uniformly to both SD (U-Net) and DiT (Transformer). Figure 2 presents parallel designs, addressing a real limitation as diffusion models shift toward Transformer backbones.

- **Thorough ablation study**: Table 1 systematically decomposes the design along three axes (which part to copy in Table 1a, connector type in Table 1b, unidirectional vs. bidirectional flow in Table 1c), and Figure 3 visualizes all five architectural variants tested, together providing solid empirical support for each design choice.

- **Enables training larger adapters within the same resource budget**: Figure 1(d) demonstrates that with the same training resources, UniCon supports 2× adapter parameters, yielding improved controllability (SSIM 0.47→0.55) and generation quality. This is a direct practical consequence of freeing resources otherwise spent on diffusion model gradients.

- **Demonstrated scalability via SUPIR-UniCon on SD3**: Figure 8 shows UniCon successfully applied to SUPIR with the SD3 backbone, illustrating that the approach scales to state-of-the-art large-scale models where ControlNet-based scaling becomes computationally prohibitive.

## Weaknesses

### Fatal
None.

### Major

- **Incomplete efficiency picture — absent inference cost analysis**: The paper frames UniCon as more efficient than ControlNet, but all efficiency claims (VRAM reduction, speedup) concern the *training* phase only (Figure 1, Figure 6, abstract). At inference, UniCon must execute both a full frozen diffusion model *and* a full-sized adapter sequentially, whereas ControlNet only adds a partial encoder copy alongside the original model. The paper never discusses this asymmetry. For deployment-sensitive applications, inference cost matters as much as training cost. Even a brief analysis or acknowledgment would suffice, but its complete absence makes the efficiency narrative one-sided. The paper should report inference latency and peak VRAM for UniCon vs. ControlNet.

- **Overclaimed generality of quality improvement on high-level control tasks**: The paper states that unidirectional flow "substantially enhances performance, improving controllability and generative quality in both high-level and low-level tasks" (Section 4.2). However, for Canny (a high-level control task), the data tells a more nuanced story: in Table 2 (DiT), UniCon improves FID over ControlNet (46.71 vs 51.52), but in Table 1c, UniCon-Full's FID for Canny (55.22) is worse than several ControlNet variants (Skip-Layer at 49.78, Full at 50.17) — suggesting that the unidirectional design may trade generation quality for controllability on some high-level tasks. The paper does not discuss this tradeoff, and the blanket "improving quality in both" claim overstates what the data supports.

- **Misleading framing of the "same adapter parameter size" comparison**: The abstract states UniCon "reduces GPU memory usage by one-third and increases training speed by 2.3 times, while maintaining the same adapter parameter size." This is technically about the comparison in Figure 1(c) where UniCon-Half and ControlNet have similar parameter counts. However, the headline results in Table 2 compare UniCon-Full (~2× parameters) against ControlNet-Encoder, which inflates the apparent gains. UniCon-Half is the fairer apples-to-apples comparison and is only shown as a secondary result for SR tasks in Table 2. The abstract's phrasing invites the reader to associate the efficiency gains with the (unfair) full-parameter comparison.

### Minor

- **Skip-Layer's incompatibility with UniCon is asserted but not analyzed**: Table 1c / Section 4.2 notes that "the skip-layer design compromising the output capability of the copied diffusion model" but provides no quantitative analysis or deeper explanation. If UniCon's core contribution is the unidirectional paradigm, its dependence on having a full-copy adapter (which carries ~2× inference cost) is a meaningful architectural constraint. A brief discussion of why this incompatibility arises would strengthen the paper.

- **SUPIR-UniCon on SD3 lacks quantitative evaluation**: Figure 8 presents only qualitative results for the potentially significant SD3 extension. A quantitative comparison (e.g., PSNR, FID) against standard ControlNet-based SUPIR would strengthen the scalability claim.

- **No variance or statistical significance reported**: The evaluation uses 1,000 test images for FID, which is a relatively small sample for this metric. Reporting confidence intervals or variance across runs would strengthen the reliability of reported improvements, though this is a community-standard practice concern rather than a critical flaw.

### Trivial
None.

## Nice-to-Haves

- Report inference latency and peak VRAM for UniCon vs. ControlNet to complete the efficiency narrative.
- Include UniCon-Half results for all tasks (not just SR) in Table 2 as the primary fair comparison.
- Analyze and discuss when/why the unidirectional design may trade quality for controllability (e.g., on high-level control tasks like Canny).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"T2I-Adapter SR PSNR 18.94 suggests implementation issues"** (Harsh Critic): This is speculation about potential implementation problems in a baseline. The paper's results may simply reflect T2I-Adapter's known limitations for pixel-level SR tasks, and this is not the authors' responsibility to defend a baseline's performance.

- **"FID on 1K images is unreliable"** (Harsh Critic): While FID on small samples has known limitations, 1,000 images is a common evaluation size in the field, and this is more of a generic concern than a specific flaw.

- **"Table naming confusion"** (Harsh Critic): The different meanings of "Full" across tables — while slightly confusing, this is a presentation/notation issue that doesn't affect substance, and the tables have clear captions explaining what each row represents.

- **"UniCon-Half not clearly defined in Table 2"** (Harsh Critic): The paper does describe UniCon-Half in context and the footnote/text provides explanation, even if not perfectly crisp.

- **"The second motivation (limited generative capabilities with fixed parameters) is asserted without evidence"** (Harsh Critic): The paper provides Figure 4 and Section 4.2 ablation results showing that replacing the diffusion decoder leads to quality degradation, which partially supports this motivation. The claim is not wholly unsupported.

- **Strength about "Figure 3(d) vs 3(e) validating necessity of preserving full diffusion model"** (Strength Finder): This is a valid ablation contribution but is more of a sanity check than a core strength — it confirms a design principle rather than introducing a new insight.

## Novel Insights

The reviews surface an interesting tension in UniCon's position in the design space: the method trades bidirectional gradient flow (which enables precise but expensive control) for unidirectional efficiency (cheaper training, potentially costlier inference). This places UniCon in a different niche from ControlNet — best suited for scenarios where training cost is the bottleneck (scaling to larger models) but inference cost is less critical, or where the quality-controllability tradeoff favors controllability (low-level tasks). The paper would benefit from explicitly characterizing this niche rather than claiming universal superiority.

## Suggestions

- Add a subsection or paragraph explicitly analyzing inference cost (latency, VRAM) comparing UniCon and ControlNet, and discuss deployment scenarios where each approach is preferable.
- Present UniCon-Half as the primary comparison in Table 2 for all tasks, with UniCon-Full as an additional "extended capacity" result, to ensure fair parameter-matched evaluation.
- Add a brief discussion characterizing when unidirectional flow helps vs. when it may trade quality for controllability, grounded in the Canny vs SR data.

## Score and Decision

**Calibration anchors:**

- **High band (avg > 7)**: DiffMatch (8.0) — novel conditional diffusion architecture with strong, comprehensive validation across many scenarios; LyJi5ugyJx (9.2) — fundamental improvements in consistency models with large-scale validation; hBGavkf61a/sojpn00o8z (7.25) — novel diffusion-based architectures with solid theoretical+empirical work. UniCon is clearly below these — its contribution is a practical architectural change rather than a fundamental methodological advance, and it has meaningful presentation gaps.

- **Medium band (4-6)**: SaRA (6.25) — training efficiency method for diffusion models, accepted as poster, with some concerns about assumptions but solid empirical backing; C0HDYvGwol (5.60) — adapter module with uneven evaluation across tasks, rejected; MBDH5zyxHM (4.6) — overclaimed adapter approach, rejected; MjhTb4gwFP (5.0) — ControlNet alternative with limited comparison, rejected; CatVTON (6.25) — lightweight adapter with training efficiency, accepted. UniCon has stronger empirical results and a more compelling core insight than most medium-band papers. Its training efficiency gains are real and well-demonstrated. Compared to SaRA (6.25), UniCon has a simpler, cleaner insight but similar concerns about scope of claims. 

- **Low band (< 3)**: ELR-Diffusion (2.5) — efficiency claims for diffusion adapter but missing key comparisons and questionable claims; APCtrl (3.67) — overclaimed ControlNet alternative with limited novelty; VideoDiT (2.5) — missing inference analysis. UniCon is clearly above these — its core insight is sound, well-motivated, and empirically validated, whereas the low-band papers have fundamental flaws.

UniCon sits above most medium-band adapter papers due to its clear architectural insight, comprehensive cross-architecture validation, and well-demonstrated training efficiency gains. However, it falls short of the high band due to the incomplete efficiency narrative (missing inference cost), overclaimed generality, and the parameter-count asymmetry in headline comparisons. Compared to SaRA (6.25) which was accepted, UniCon has a comparably strong efficiency contribution but more honest (if still incomplete) empirical validation. The inference cost omission is a moderate concern that doesn't invalidate the results but does undermine the efficiency framing.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>