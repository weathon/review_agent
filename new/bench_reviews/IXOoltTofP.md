## Summary
The paper introduces 3DAxisPrompt, a visual prompting method that embeds 3D coordinate axes and SAM-generated object contours into multi-view rendered point cloud images to elicit 3D spatial grounding and reasoning in GPT-4o without fine-tuning. The authors conduct a systematic investigation of prompt formats (depth images, tri-view, multi-view, various mark types) and evaluate on indoor localization, route planning, outdoor localization, and robot action prediction across four datasets. The core finding is that explicit geometric scaffolds—particularly 3D axes with ticks/labels and object contours—enable GPT-4o to output 3D coordinates and structured action plans where standard prompting fails.

## Strengths
- **Systematic and revealing investigation of visual prompt formats for 3D tasks.** Figure 7 quantifies that axis ticks are strictly necessary (removing them causes complete failure) and axis labels reduce to-bbx errors by 37%. Figure 2 and Table 1 progressively establish that tri-view succeeds where multi-view fails for 3D grounding, depth images don't help, and 2D contour marks outperform bounding boxes (0.271 vs 0.376 NRMSE to-center). These are concrete, actionable design insights not previously documented.

- **Measurable quantitative improvement across prompt variants.** Table 1 shows a clear progression: from basic Mark (0.333 NRMSE to-center) to Mark+2D contour+CoT (0.219), a 34% improvement. Adding CoT alone yields a 19% gain (0.271→0.219), demonstrating the method's sensitivity to reasoning scaffolds.

- **Evaluation spans four datasets and four distinct 3D task types.** Table 2 covers indoor localization (ScanNet), outdoor localization (nuScenes with vehicles and vegetation), route planning (ScanNet, 79% average success), and robot action prediction (FMB, 72.5%/62.5% grasp/release), demonstrating the method generalizes beyond a single domain or task.

- **Compelling qualitative demonstrations of the capability gap.** Figures 1, 4, and 5 show side-by-side comparisons where standard GPT-4o responds with "I cannot directly analyze 3D models..." while 3DAxisPrompt yields specific 3D coordinates, step-by-step route plans, and structured robot action sequences.

## Weaknesses

### Major

- **The evaluation protocol is too small and uncontrolled to support the paper's quantitative claims.** Section 4.1 explicitly states evaluation was done via the ChatGPT interface ("due to limited and costly GPT-4o API quota") on only 20 randomly selected scenes per dataset. With stochastic MLLM outputs and prompt-sensitive behavior, a single-run evaluation of 20 scenes per task—without variance estimates, temperature controls, or repeated trials—cannot establish reliable quantitative gains. The route planning results (Table 2) report success rates like 80%, 100%, 70% from what appears to be 5 task templates total; the robot action prediction uses "whether the orders can complete the mission" as its criterion. These are suggestive demonstrations, not statistically substantiated benchmark results.

- **The paper does not isolate what 3DAxisPrompt actually contributes.** The method bundles several ingredients: 3D axis, rendered multi-view images, text-formatted point clouds (which Section 3.2 admits are "essential"), SAM-derived masks/contours, marks, CoT, and reference object coordinates. Table 1 only compares variants of this bundled package—all rows include the axis + SAM marks. There is no baseline of axis-only, contours-only, text point cloud-only, or SoM-style 2D prompting on the same renders. Because CoT and reference coordinates materially improve performance (19% gain from CoT alone), the paper cannot support its central claim that the **3D axis** is the key enabler. The observed gains could equally come from the structured text or reasoning scaffolds.

- **The paper overclaims about the demonstrated capability.** The abstract claims the method extends MLLMs to "real-world 3D scenarios" and GPT-4o can "effectively perceive an object's 3D position." But the setup provides an explicit coordinate frame drawn into the scene, object marks/contours, and text-formatted point cloud coordinates. What is demonstrated is that GPT-4o can sometimes use heavily engineered external geometric scaffolds to output plausible coordinates—a narrower claim than "3D grounding and reasoning." The distinction matters: success on localization under this setup can reflect coordinate extraction or interpolation from overlaid rulers rather than genuine spatial reasoning about 3D scenes.

### Minor

- **NRMSE normalization is underspecified.** Equation 3 defines NRMSE with max(x_i) but doesn't clarify whether this is per-axis, per-scene, or globally normalized. For route planning and robot action prediction, "success" is defined operationally (whether navigation reaches destination, whether orders complete a mission) but the judging criteria for textual outputs are not described—it's unclear whether success reflects executable spatial reasoning or merely plausible language.

- **The evaluation sample selection is conceptually confusing.** Section 4.1 states the authors "randomly selected 20 scenes from each test dataset as validation data." Selecting from test data and treating it as validation blurs the standard train/val/test split, and the paper doesn't specify the random seed or stratification procedure, making the protocol non-reproducible.

### Trivial

- The paper uses "3DAxisPrompt" and "3DaxisPrompt" interchangeably throughout, and sometimes "3DAxiesPrompt" (Figure 4 caption, Section 4.3 robot action prediction text), creating minor notation inconsistency.

- Section 3.1 Equation 2 is introduced as the model input formulation but is descriptive rather than explanatory—the method depends on non-trivial preprocessing (axis insertion, view selection, SAM masks) not specified at a reproducible level.

## Nice-to-Haves
- Analyze failure modes by depth, occlusion, and object density to quantify where the method breaks down (the paper mentions these qualitatively but doesn't provide systematic analysis)
- Show error distributions and per-axis errors, not just aggregate NRMSE, to clarify whether errors are dominated by depth or other axes
- Compare against a simple rule-based coordinate extraction baseline from the rendered axis views to establish whether the claimed reasoning contribution exceeds what a geometric parser could achieve
- Test on tasks requiring relational 3D reasoning beyond coordinate reading (e.g., collision-aware path validity, support/containment under occlusion) to better match the paper's framing

## Removed Points
**These points are flagged to be removed, treat them with caution:**

- "The evaluation protocol is too small to establish reliable gains" — KEPT as Major weakness, genuinely substantiated
- "No fair baseline against simpler prompt engineering" — KEPT as Major, verified against Table 1
- "Overclaims 3D grounding from coordinate extraction" — KEPT as Major, verified against Sections 3.2, 4.2, abstract
- "NRMSE underspecified" — KEPT as Minor, verified against Equation 3
- "Missing baseline for SoM-style 2D prompting" — This is valid but the paper doesn't claim to compare to SoM; it's more of a scope expansion request. Moved to Nice-to-Haves.
- "Need larger, pre-declared sample evaluation" — Valid but partially addressed by the paper's own acknowledgment of quota constraints; the issue is kept as Major because 20 scenes/dataset is indeed too small for the claims made.

## Novel Insights
The paper's most genuinely novel observation is that MLLMs like GPT-4o can be induced to operate on 3D scene tasks—localization, planning, action prediction—when provided with explicit external geometric scaffolds (3D axes with ticks/labels, object contours, text-formatted point clouds). The investigation reveals that axis ticks are indispensable (failure without them), tri-view rendering is more effective than arbitrary multi-view for 3D grounding, and 2D contour marks outperform bounding boxes for delineating objects. However, the paper also reveals the limitation of this approach: the model may be performing coordinate extraction from overlaid rulers rather than genuine 3D reasoning, and the method's effectiveness is entangled with CoT and reference point scaffolds. This represents a meaningful but narrower contribution than the paper's framing suggests—a useful prompting technique whose mechanism is not fully isolated.

## Suggestions
1. **Reframe the contribution more narrowly and accurately.** Position the paper as investigating what external geometric scaffolds enable off-the-shelf MLLMs to perform 3D spatial tasks, rather than claiming to "promote 3D grounding and reasoning" (which implies intrinsic capability). The discussion (Section 5) does this better than the abstract—bring that framing forward.

2. **Run repeated trials with controlled decoding settings** (fixed temperature, temperature=0.7 with multiple samples) and report variance. Even 3-5 runs per scene would dramatically strengthen the claims.

3. **Add proper ablations isolating each component:** axis-only, contours-only, text point cloud-only, marks-only, and combinations. This is necessary to support any claim about what the 3D axis contributes versus other scaffolds.

4. **Expand the evaluation sample** beyond 20 scenes per dataset, or at minimum provide a power analysis showing that 20 scenes is sufficient for the observed effect sizes.

5. **Clarify the judging criteria for route planning and robot action prediction.** Define what constitutes "success" operationally (e.g., coordinate accuracy thresholds, executable command validity) rather than relying on subjective assessment of textual responses.

## Score and Decision
I calibrated against several anchor papers:

- **High-scoring papers (7-8):** Papers like TraceVLA (scores 6,6,8,8) had comparable novel prompting ideas but backed them with extensive experiments (137 SimplerEnv configs, 4 real-robot tasks) and proper baselines. This paper is clearly below that bar due to its 20-scene evaluation and missing component ablations.

- **Borderline papers (4-6):** The most relevant anchor is `8ibaVk4mU8` (COARSE CORRESPONDENCES, scores 6,5,3), which also proposes visual prompting for MLLM 3D understanding and has similar scope but was ultimately withdrawn/rejected. Papers in the `VQZCXoteoP` (3,3,5,5) and `w0es2hinsd` (5,5,8,3) range have similar issues: genuinely interesting findings undermined by small-scale evaluation and missing baselines. The paper under review is slightly more polished and has better figures/abl

ations than the lowest-scoring papers, but the scale limitations are comparable.

- **Low-scoring papers (≤3):** Papers like `cPmLjxedbD` (1,1,1,1) and `XWfjugkXzN` (3,1,1) were rejected for essentially no experiments, unclear methodology, or no proposed technique. This paper is well above those—it has a real method, real experiments, and real findings, even if limited.

The paper has genuine contributions and interesting findings, but the evaluation is too small (20 scenes/dataset, ChatGPT interface, no variance) and the component isolation is insufficient (no ablation of axis vs. contours vs. text coordinates) to support acceptance. However, it's also not a reject-level paper—there's real substance here. I place it in the borderline range, leaning toward the lower end due to the evaluation protocol limitations.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>