---
job_id: b2563404-c40e-4f54-98b3-196f79db67ef
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 4Ha2srdhPN.pdf
paper: GRAID: Enhancing Spatial Reasoning of VLMs Through High-Fidelity Data Generation
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely about representation learning for vision–language models, synthetic dataset generation for spatial reasoning, and downstream VQA benchmarks, all well within ICLR’s core topics.

## Minimum Quality
Pass ✅.  
The paper has all required sections (abstract, introduction, related work, method, experiments, results, conclusion), is written in English, presents a concrete framework (GRAID + SPARQ) with algorithms and equations, and includes substantial experimental evaluation against prior work. I do not see fatal methodological flaws or evidence of test leakage that would mandate an immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or attempts to manipulate automated reviewing in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper proposes GRAID, a framework for generating large-scale spatial reasoning VQA datasets using only 2D bounding boxes from object detectors, avoiding single-view 3D reconstruction and caption-based LLM question generation. A key component, SPARQ, uses lightweight predicates to quickly screen images for question applicability before running more expensive realization routines, enabling efficient generation of over 8.5M VQA pairs across BDD100k, NuImages, and Waymo. Human evaluation suggests substantially higher dataset fidelity than a community SpatialVLM dataset, and fine-tuning several VLMs on GRAID improves performance on held-out GRAID question types and multiple external benchmarks such as BLINK, A-OKVQA, RealWorldQA, NaturalBench, and VSR.

## Strengths

1. **Clear problem focus and practical contribution (spatial reasoning via high-fidelity data).**  
   The paper identifies a concrete and important failure mode of current VLMs, namely spatial reasoning, and attacks it through data generation rather than architectural changes. The choice to restrict to qualitative spatial relations in 2D (bounding-box geometry) is well motivated by the error analyses of 3D single-view reconstruction pipelines and by the human study on SpatialVLM’s OpenSpaces (Page 7, Figure 1).

2. **Methodological clarity and explicit algorithms.**  
   The pipeline is well articulated: object detection interface (Page 4, Section 3.1), predicate checks and question realization via SPARQ (Section 3.2), and a concrete algorithm for a representative relation (Algorithm 1, Page 5). For RightOf, the condition \(x_{\min}^{(1)} > x_{\max}^{(2)}\) plus \(\text{IoU}(b_1, b_2)=0\) is explicit and easy to reason about, which is preferable to the opaque heuristics in some previous synthetic VQA works.

3. **Scalability demonstrated and quantitatively characterized.**  
   The paper does not just claim efficiency; it measures it. **Table 3** (Page 20) reports average predicate time vs apply time and hit rates for each question template. For example, LargestAppearance shows a 0.02 ms predicate vs 69.74 ms apply, with a 78.8% predicate→QA hit rate, supporting the claim of up to \(1400\times\) savings for some templates in Section 3.2. This level of instrumentation is uncommon and useful for practitioners considering adopting or extending the system.

4. **Diverse question taxonomy and dataset coverage.**  
   GRAID implements 22 templates across categories: spatial relations, counting, ranking/extrema, localization, and size/aspect. **Figure 2** (Page 6) gives a clear radial breakdown of the 5.3M questions in GRAID-BDD by category and template, making the coverage and distribution of cognitive skills explicit. **Table 2** (Page 7) further shows sizes across datasets and depth / non-depth variants, giving a concrete view of scale and splits.

5. **Careful human evaluation and comparison to prior pipelines.**  
   The human study on OpenSpaces vs GRAID-BDD is a strong point. On OpenSpaces, only 41.6% of questions are considered valid and 57.6% of answers are incorrect (Page 7), whereas for GRAID-BDD without depth, over 95.58% of questions and 93.69% of answers are judged valid, with detailed accounting of unclear vs invalid cases (Page 7–8). This directly supports the central claim that 2D qualitative geometry with conservative thresholds yields much higher-fidelity supervision than single-view 3D metric estimation.

6. **Evidence for concept learning and cross-dataset generalization.**  
   The experiments on RQ1 and RQ2 are thoughtfully designed. For RQ2, fine-tuning Llama 3.2 11B only on six primitive questions (LeftOf, RightOf, HowMany, AreMore, LargestAppearance, IsObjectCentered) and then evaluating on all 22 templates in both GRAID-BDD and unseen GRAID-NuImages shows broad gains. **Figure 3** (Page 9) and **Figure 4 / img-9** (Appendix, Page 24) explicitly visualize before/after accuracy by question type, including many templates not seen during training and a fifth topic (Size & Aspect) unseen in training. This strongly supports the claim that GRAID teaches reusable spatial primitives rather than template memorization.

7. **External benchmark improvements vs a strong baseline and a competing synthetic dataset.**  
   The RQ3 experiments fine-tune four different instruction-tuned VLM families on GRAID-BDD vs on the SpatialVLM-generated OpenSpaces. **Tables 4, 5, and 6** (Pages 21–23) show consistent gains across multiple benchmarks. For example, for Llama 3.2 11B (Table 4), BLINK Overall increases from 25.72% to 42.13% after GRAID, with large jumps on clearly spatial-heavy subtasks: Relative Depth (+41.94%), Visual Correspondence (+23.83%), and Spatial Relation (+35.66%). At the same time, NaturalBench metrics improve slightly or stay stable, suggesting spatial skills are improved without catastrophic degradation elsewhere, in contrast with the severe regression when training on OpenSpaces.

8. **Ablation on where spatial reasoning is learned inside the VLM.**  
   The LoRA ablation in Appendix A.4, together with **img-5–img-8** (Page 25) and **Table 7**, indicates that disallowing language-layer adaptation significantly harms training (higher loss, lower performance), while removing ViT, attention, or MLP adapters has much smaller impact. This is an interesting diagnostic result: spatial skills appear to be encoded and manipulated primarily in the language layers, consistent with recent observations about LVLM representations.

9. **Open, extensible framework orientation.**  
   GRAID is presented explicitly as a template/predicate library that can be instantiated with any detector and extended with new question types. The appendix’s detailed template descriptions (Pages 17–19) clarify the geometric and statistical conditions used to avoid ambiguous questions (buffer margins, area ratios, depth margins), which increases confidence in both extensibility and reproducibility.

## Weaknesses

1. **Limited exploration of failure modes and biases of 2D-only reasoning.**  
   While the paper convincingly argues that 3D single-view pipelines are noisy, it glosses over the systematic errors induced by using only 2D bounding boxes as a surrogate for spatial relations. For example, Algorithm 1 (RightOf, Page 5) uses simple constraints \(x_{\min}^{(1)} > x_{\max}^{(2)}\) and \(\text{IoU}=0\), but does not analyze how often ground-truth semantics disagree with this 2D geometry due to perspective projection (e.g., objects that are diagonally offset or at different depths but overlapping in x-coordinates). The depth-based questions (Closer, Farther, DepthRanking) are treated qualitatively, but the same kind of careful human error breakdown given for SpatialVLM’s metric questions is not provided for these templates. A more systematic, quantitative study of where 2D geometry and human spatial judgments diverge, beyond a single overall validity rate, would significantly strengthen the argument that 2D is “reliably” sufficient.

2. **Human evaluation scope is narrow and not clearly stratified by question type.**  
   The human study on GRAID-BDD covers only 317 VQA pairs (Page 7–8) across 4 annotators, and the sampling procedure is seeded by each evaluator’s name but otherwise unspecified. There is no stratification by question template, category, or difficulty. As a result, it is unclear whether rarer or more subtle templates (e.g., ObjectsInRow, MostClusteredObjects, Quadrants, depth-based questions) exhibit similar fidelity to simpler count and Left/Right relations. In **Figure 2**, the long tail of question types is visible, but the human study might be dominated by the more frequent templates; the paper should at least report per-question-type validity or confidence intervals to substantiate the “>91% human-verified validity” claim by template, not just overall.

3. **Comparisons to related generation frameworks are incomplete or asymmetric.**  
   The paper extensively critiques SpatialVLM’s OpenSpaces dataset, and gives a partial discussion of SpatialRGPT and SpaRE, but the empirical comparison is narrow:

   * For SpatialRGPT, the authors state that masked region queries prevent human evaluation (Page 7), then essentially drop it. This is a bit too convenient, especially given that region-based models are increasingly common. It would be more convincing to design a small, controlled evaluation of region-based QA (with explicit region prompts or Set-of-Mark style references) rather than discarding the dataset.
   * SpaRE is discussed qualitatively (caption reliance, hallucinations) but not evaluated or used as a training baseline, even though it is exactly another “spatial synthetic QA from real images” pipeline. Including SpaRE-trained models or at least a subset of its data in the RQ3 benchmark comparison would provide a more balanced landscape.

   As is, the primary empirical foil is SpatialVLM’s community OpenSpaces dataset, which is acknowledged elsewhere to be relatively weak.

4. **Statistical rigor of external benchmark evaluation is limited.**  
   **Tables 4–6** show many large improvements, but lack any measure of variance or statistical significance. All results appear to be single-run SFT with a single random seed and fixed number of steps, and there is no notion of robustness across seeds or training lengths. For example, in Table 4, Llama+GRAID dramatically improves BLINK Relative Depth and Spatial Relation, but some other subtasks like Forensic Detection and Art Style show negligible gains or even slight regressions. Without error bars or repeated runs, it is hard to disentangle genuine improvements from noise in the fine-tuning process. Given that the training sets used for RQ3 are relatively small (51,546 examples, Appendix A.3) and SFT lasts only 200 steps, the training dynamics may be brittle.

5. **Potential distribution overfitting to driving scenes is not deeply analyzed.**  
   The authors argue that improved performance on indoor/outdoor benchmarks like BLINK and NaturalBench implies that GRAID teaches scene-agnostic spatial concepts (Page 9). However, all synthetic training data are from driving datasets (BDD, NuImages, Waymo) with heavily biased object vocabularies (cars, trucks, traffic signs, etc.). While the authors note that only 10/143 BLINK spatial questions mention “car,” this is anecdotal. The paper does not show robustness to other domains that share little visual overlap (e.g., medical imaging, diagrams, or low-text scenarios). A more explicit analysis, for example performance grouped by object type or by presence/absence of traffic-related objects on downstream benchmarks, would better support the “domain-agnostic spatial primitives” claim.

6. **Some templates and thresholds are quite heuristic and under-justified.**  
   The templates in Appendix A.1 use multiple tunable hyperparameters (area ratios, separation margins, aspect ratio thresholds, variance thresholds for ObjectsInRow, DBSCAN eps proportional to image diagonal, depth margins, etc.). These are motivated qualitatively but not systematically tuned or ablated. For instance, RankLargestK requires “each consecutive pair has a sufficient multiplicative gap,” but the exact ratio is not given in the main text, and there is no analysis of sensitivity. Similarly, the depth templates use a margin_ratio to avoid ambiguous closeness rankings, yet there is no quantitative evaluation of how varying this ratio trades off coverage vs accuracy. From a methodological standpoint, these heuristic design choices are central to dataset quality and should be more carefully explored.

7. **Evaluation setup for RQ1/RQ2 is somewhat underspecified and may conflate template memorization with generalization.**  
   For RQ1, the model is fine-tuned on 10% of GRAID-BDD with unstratified sampling and evaluated on 1000 unstratified examples from GRAID-BDD and 1000 from GRAID-NuImages (Page 8). Because the training and test distributions are unstratified over templates, there is a risk that the held-out set shares many question types and object combinations that closely mirror the training set, especially for frequent questions like HowMany and MoreThanThresholdHowMany. For RQ2, Figure 3 and Figure 4 show large gains on unseen templates, but the paper does not clarify whether the evaluation is balanced across question types or how accuracy is computed over multi-choice vs free-form vs Yes/No questions. Without this information, it is difficult to precisely interpret the reported “+47.5% on BDD and +37.9% on NuImages” style aggregate improvements.

8. **Mathematical / formal clarity gaps around depth-based questions and multi-object geometry.**  
   While Equation-style notation for bounding boxes and probabilities is clear in Section 3.1, the extension to depth is opaque. For Closer/Farther/DepthRanking, the paper states that SAM masks and a monocular depth map are used to “estimate per-class closest depth” and compares them by a margin (Appendix A.1), but there is no formalization of how depth is aggregated over a mask (min, mean, median? how are outliers handled?), nor how margin_ratio is applied. Similarly, multi-object predicates like ObjectsInRow perform “linear regression on centers” and use “normalized vertical residual variance” as a threshold, but the exact functional form is not given. These gaps matter because they determine whether these more complex templates produce unambiguous labels; they deserve at least an equation-level formalization comparable to Algorithm 1.

9. **Related work section misses several very closely aligned recent studies.**  
   The related work focuses on SpatialVLM, SpatialRGPT, SpaRE, and some bounding-box–aware VLMs, but omits several recent works that explicitly target spatial reasoning via synthetic data or curricula:
   * SpatialTraceGen (Huh et al., 2025), which proposes high-fidelity reasoning traces for spatial reasoning distillation, is very close in spirit and should be discussed as an alternative to pure QA-label supervision.
   * SpatialLadder (Zhang et al., 2025), which uses progressive curricula for spatial reasoning in VLMs, is a direct complement to GRAID’s template-based data; a discussion of whether GRAID data could be used in such a curriculum is natural.
   * Levental (2026) on “Can Vision-Language Models See Squares?” and related work on text-recognition-mediated spatial reasoning sheds light on perceptual shortcuts that might interact with GRAID’s bounding-box–only supervision.
   * A more recent SpatialVLM / 3D VLM formulation (e.g., Chen et al., 2026) is not cited, even though the paper positions itself partly as “2D vs 3D” spatial reasoning.  

   These omissions weaken the positioning.

10. **Presentation issues and minor clarity problems.**  
    There are some duplicated sentences (e.g., “Table 1 offers a comparison of the differences…” repeated on Page 2), and a few typos or slightly confusing phrasing (“We conduct a series of fine-tuning experiments to determine how well a VLM can learn spatial reasoning concepts from our data. For all experiments we use Meta Llama-3.2-Vision-Instruct-11B as the base model… We ask the following research questions: - [leftmargin=*] - RQ1…”). The description of the LoRA ablations referring to the plots is a bit terse, and it takes effort to map img-5–img-8 to the text. These are not fatal, but they do detract from clarity.

## Potentially Missing Related Work

1. **Huh et al., “SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation”, 2025.**  
   This work also targets spatial reasoning improvements in VLMs via high-fidelity synthetic supervision, but uses reasoning traces instead of pure QA pairs. It is directly relevant to the “data for spatial reasoning” theme and should be cited and discussed in Section 2, with a comparison of trace-based vs QA-based supervision.

2. **Zhang et al., “SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models”, 2025.**  
   SpatialLadder presents a curriculum-based training strategy for spatial reasoning, which could likely benefit from or be combined with GRAID-generated data. It should be mentioned in the related work on enhancing spatial reasoning capabilities and possibly discussed in Section 5 as a complementary direction.

3. **Levental, “Can Vision-Language Models See Squares? Text-Recognition Mediates Spatial Reasoning Across Three Model Families”, 2026.**  
   This paper studies how VLMs sometimes solve spatial reasoning tasks via text-related shortcuts (e.g., reading labels) rather than genuine geometric understanding. Given that GRAID’s questions and datasets are built on natural images that often include textual content (road signs, etc.), this is highly relevant to potential shortcut behavior and should be discussed in Section 2 or Section 5 when interpreting benchmark gains.

4. **Chen et al., “SpatialVLM: 3D Vision-Language Models”, 2026.**  
   This is a more recent 3D-centric spatial VLM, strengthening the “3D-first” side of the design space that GRAID positions itself against. It would be appropriate to add this citation in the 3D reconstruction subsection of Section 2 and clarify how GRAID’s 2D qualitative geometry compares to this more advanced 3D approach.

## Questions

1. **Per-template validity and error breakdown.**  
   Can you provide per-question-type human validity statistics (or at least by category, e.g., “Spatial Relations vs Ranking vs Localization vs Size & Aspect vs Depth-based”) for GRAID-BDD, similar in granularity to the computational breakdown in Table 3? If space is tight, a summary figure or table in the appendix would suffice. This would greatly help assess where the framework is most and least reliable.

2. **Depth template formalization and quality.**  
   For Closer/Farther/DepthRanking, could you specify mathematically how you convert SAM masks and monocular depth into a scalar depth per instance and per class (e.g., \(d_{\text{class}} = \min_{(x,y)\in \text{mask}} D(x,y)\) or another functional)? Also, have you run a smaller human validation focused specifically on these depth-based templates, analogous to what you did for non-depth questions?

3. **Distribution of training vs evaluation data in RQ2.**  
   In RQ2, Figure 3 shows gains across many templates not seen during training. Can you clarify how the evaluation sets are constructed (e.g., equal number of instances per question type? or unstratified sampling)? Some question types are inherently easier (e.g., HowMany with small counts) than others (ObjectsInRow). A more controlled split would help interpret which skills are actually being transferred.

4. **Robustness across seeds and training schedules.**  
   Have you repeated RQ3 with different SFT seeds or slightly varied training steps? If so, how stable are the improvements on BLINK, NaturalBench, and VSR? Even a small-scale study (e.g., 3 seeds on one model family) would help differentiate systematic gains from training noise.

5. **Interaction with 3D or region-based models.**  
   Do you envision any hybrid use of GRAID with region-based approaches like SpatialRGPT or 3D-centric models (e.g., training with both GRAID QA and 3D traces)? It would be interesting to hear whether you tried, or plan to try, fine-tuning region-aware models on GRAID’s QA pairs to combine benefits of explicit localization with high-fidelity QA.

6. **Heuristic thresholds / hyperparameters.**  
   Can you provide more detail on how thresholds such as area margins, separation distances, DBSCAN eps, and variance tolerances for ObjectsInRow were chosen? Did you perform any sensitivity analysis (even qualitative) to see how these affect dataset size and validity rates?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard autonomous driving datasets with bounding-box annotations and synthetic QA generation; no new sensitive data or human subjects are collected beyond small-scale annotation for validation.

## Soundness Rating

3: good.  
The technical ideas (2D geometric predicates + template realization + SPARQ) are straightforward but correctly implemented, and the experimental methodology is mostly solid, with multiple models and benchmarks. Some aspects (depth-based question formalization, statistical robustness across seeds, and detailed analysis of failure modes) are underdeveloped, preventing an “excellent” rating.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with clear figures (especially Figures 1–4) and tables (1–3, 4–6, 7). A few duplicated sentences, minor typos, and slightly compressed explanations in the ablations and depth templates reduce clarity but do not obstruct understanding.

## Contribution Rating

3: good.  
The contribution is not conceptually deep in a theoretical sense, but the framework and resulting datasets are practically valuable, address a well-identified gap (spatial reasoning), and include convincing evidence of impact on strong existing VLMs and benchmarks. The work should be of interest to both researchers and practitioners focusing on spatial reasoning.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper sits above the bar primarily because it tackles an important and stubborn weakness of VLMs using a clear, implementable framework, backs claims with a substantial dataset and multiple external benchmarks, and provides evidence of genuine concept learning and cross-dataset transfer. However, some methodological components (heuristic thresholds, depth templates) are under-analyzed, the human validation is relatively small and not stratified, and the comparative story is somewhat skewed toward a single weak baseline (OpenSpaces). With stronger analysis of error modes and fuller positioning against very recent related work, this could move toward a stronger accept.

## Reviewer Confidence

4: confident.  
I am familiar with spatial reasoning for VLMs, synthetic VQA generation, and the cited baselines, and I carefully examined the math, algorithms, and reported results. Some missing details (e.g., depth aggregation, threshold selection) leave room for uncertainty, but they do not fundamentally alter my overall assessment.