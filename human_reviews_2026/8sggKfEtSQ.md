# Aligning Vision-Language Models With Human Directional Reference

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Spatial expressions are inherently ambiguous because communicators may adopt different perspectives, making interpretation highly dependent on the chosen frame of reference. Despite recent advances, current vision-language models (VLMs) still struggle to resolve this ambiguity in the absence of a clear reference frame, limiting effective communication between humans and machines. In contrast, humans often overcome this challenge by employing object-centered frames anchored to objects with an intrinsic ‘front’, a property known as frontedness, which determines their orientation and the spatial relationships around them. In this paper, we investigate the feasibility of endowing VLMs with object-centered spatial reasoning abilities, with frontedness as an essential component of the object-centric frame. To this end, we introduce a benchmark of synthetic 3D scenes for systematically evaluating the spatial reasoning of VLMs, and find that they consistently misidentify object orientations and tend to adopt a view-centric perspective. We show that enabling VLMs to perform spatial reasoning from an object-centric perspective achieves better alignment with human behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Spatial expressions are ambiguous due to perspective taking. This paper mainly studies the problem of object-centered spatial reasoning that associate with objects that has an intrinsic front. This paper introduces a new synthetic benchmark to systematically evaluate spatial reasoning of vision-language models (VLMs) and experiments show that VLMs are not good at identify object orientations and use a view centric perspective. They also show that spatial reasoning from an object centric perspective achieves better alignment with human behavior.

### Strengths
1. Well-design figures show clearly of the spatial reasoning problem being studied. The paper writing is also easy to understand with good flows.
2. The ablation results show the importance of both orientation prediction and object centric spatial reasoning.

### Weaknesses
The contribution in this paper is very limited. The COMFORT paper referenced has studied the problem of perspective taking, object frontness, and showed that vision language models (VLMs) adopt a view-centric perspective and cannot flexibly use other perspectives like object-centric perspective (i.e., intrinsic FoR). Therefore the only contribution is showing that fine-tuning VLMs for object-centric spatial reasoning lead to better performance, but it's not very surprising for this finding.

### Questions
1. For human evaluation, participants are from 6 PhD students from an NLP research laboratory. Is this standard for participant recruiting? It lacks of diversity of the participants and PhD students in an NLP background maybe familiar with the spatial reasoning problems and will make the participants not representative for human performance.
2. It's better to bold or underscore the best model performance and second best performance in the tables to make the results easier to compare.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper formalizes object-centered (OC) spatial reasoning through frontedness (intrinsic "front" of objects) and introduces SOFA, a synthetic 3D benchmark that tests (i) object orientation understanding and (ii) spatial relation reasoning with appropriate frame-of-reference (FoR) selection (OC for fronted objects, viewer-centered/VC for non-fronted). 

Baseline VLMs show canonical/front-facing bias and VC bias. After task-specific fine-tuning (LoRA + unfrozen vision encoder) on synthetic relational annotations, two mid-size VLMs approach human-like FoR selection and improve orientation accuracy, with modest transfer to a small real-image slice. The work is conceptually clear and methodologically careful, but several wins appear to stem from in-domain synthetic training aligned tightly with the eval, so the leap beyond the trained domain is still tentative.

### Strengths
- The key insight is making it clear when to use object-centered vs viewer-centered reasoning. This makes the model's competence auditable, not just whether it gets "left/right" correct. That separation is the paper's real conceptual contribution. The paper quantifies two specific biases in modern VLMs: they assume objects are in canonical/front-facing poses and default to viewer-centric perspectives. This makes "VC bias" a measurable phenomenon rather than just anecdotal observation.
- Joint training is synergistic. Training on orientation alone helps with orientation but not object-centered relations. Training on OC relations alone helps with relations and somewhat helps orientation. Training on both together works best, suggesting these skills mutually reinforce each other. The approach uses LoRA on the language model while unfreezing the vision encoder, trained on synthetic data with combined losses for both orientation and OC reasoning. After fine-tuning, models show large improvements on SOFA benchmarks, getting much closer to human performance. The ablations confirm that you really need both types of training together.
- "Real" transferable. On a small MS-COCO subset focused on fronted objects, the fine-tuned models show some transfer to real images, though performance is still well below human level. This suggests the approach has promise beyond synthetic data, even if there's a long way to go.

### Weaknesses
- Data quality: Very obviously, the data suffer from uniform lighting, canonical placements, and 90° rotations do not capture continuous headings, occlusion, or clutter.
- The strongest gains reported indeed stem from fine-tuning on a synthetic dataset (SOFA-train) that is structurally identical to the evaluation setting. In other words: the training and test data share identical rendering pipelines, spatial layouts (4-way axial placements, 90° rotations), and categorical frontedness priors; although the authors exclude test meshes, the category distribution and geometric conventions are nearly identical; the fine-tuned model directly optimizes for the same question templates and answer formats used in evaluation. So **this is largely "train in-domain, test in-domain" fine-tuning success rather than emergent, generalized spatial understanding.**
- Prior work such as SpaRE (https://arxiv.org/abs/2504.20648; ACL 2025) has already shown that targeted synthetic spatial data can boost VLM spatial reasoning, so using more synthetic supervision in a matched domain is not, by itself, novel. Without demonstrating cross-domain generalization (to real images, continuous viewpoints, or unseen object categories), the work risks being seen as an exercise in careful dataset engineering rather than a meaningful advance in spatial cognition or model alignment.

### Questions
### Q1: Are results robust to non-axis-aligned figure placements and continuous headings (not just 0, 90, 180, 270)?

**Action:** Add a continuous-angle split (e.g., every 15°), random radial offsets, and non-planar placements; track Ori/FoR metrics vs angle.

### Q2: Are models simply overfitting the prompt schema or a constrained label set?

**Action:** Swap templates (no "reasoning-then-answer"), change answer format (free-text vs object name), and test robustness to synonyms/extra distractors; include a "no-template" ablation.

### Q3: Does improvement hinge on category overlap (fronted vs non-fronted priors)?

**Action:** Create category-disjoint and appearance-disjoint splits (including novel fronted categories) and report Ori/FoR deltas; increase the real-image slice beyond 280 and include non-fronted real objects.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how VLMs handle ambiguous spatial language, arguing that models should often use an object-centeric frame of reference (FoR) anchored to an object’s intrinsic frontedness rather than defaulting to the viewer perspective. The authors introduce a synthetic 3D benchmark, SOFA, to test two skills: (1) recognizing an object’s facing direction (orientation) when frontedness exists, and (2) choosing the correct frame (object-centric or viewer-centric) when answering spatial-relation questions. They show current VLMs frequently misidentify orientations and overwhelmingly default to the camera’s view; with targeted fine-tuning using SOFA-style annotations, models align much better with human behavior on both tasks.

### Strengths
* The key ideas are grounded in a rich literature in psychology and linguistic that study how linguistic descriptions affect human spatial cognition.

* The writing is clear and coherent, addressing critical cognitive bottleneck for VLMs that should be solved to achieve human-level spatial understanding.

* Provides a systematic analysis based on the new SOFA benchmark, showing that recent VLMs still exhibit viewpoint-related biases in spatial reasoning.

* Beyond proposing a benchmark, the paper demonstrates that fine-tuning VLMs (LLaVA-NeXT and Qwen2.5-VL) on SOFT leads to notable improvements in spatial reasoning.

### Weaknesses
Although the paper introduces a new benchmark SOFA, the technical novelty of both this benchmark and the takeaway from the failure analysis are questionable, in comparison with multiple previous studies. I outline the main concerns below.

* **Missing related work**:

  * **RoboSpatial [1]** makes a similar claim that VLMs struggle to reason in "object-centric" FoR, and present a benchmark to assess and fine-tune VLMs. This work should be cited and a further discussion seems necessary to clarify the novelty of SOFA.

  * **SITE [2]** also includes a "spatial orientation" subset that evaluates how VLMs handle object-centric FoR. Moreover, this benchmark consists of real images rather than synthetic ones. **OmniSpatial [3]** and **ViewSpatial-Bench [4]** also include perspective-taking tasks. Could the author elaborate on the advantage of synthetic setups in relation to such real benchmarks?

* The novelty relative to COMFORT [5] is unclear. Please clarify how this work differs from or extends COMFORT's claims and analysis on the FoR understanding of VLMs.

* Moreover, the statement that **"SOFA reveals limitations in their understanding of orientation and a tendency to adopt the viewer-centric perspective"** may not be a new finding, since the benchmarks listed above (and COMFORT) have extenstively discussed this limitation of VLMs.

* The claim that **"limited consideration of FoR undermines the consistency and reliability of such evaluations"** (line 133) is confusing. For instance, Yin et al. [6] include perspective-taking tasks, so it's unclear which aspect of FoR these previous work are lacking. Could the authors please elaborate more on the core missing aspect of these previous benchmarks?

* **"These approaches remain misaligned with human strategies for resolving FoR-related ambiguity, which rely on an object’s intrinsic orientation rather than inconsistent perspectives."** (line 140-141). This comment is unclear. What does it mean to rely on "inconsistent perspectives"? Is the intention to emphasize that humans can integrate information across multiple perspectives?

* Does the training dataset consist of similar synthetic scenes as in Figure 7? A concerning aspect on the training/evaluation on SOFA is that the overall scene layout would mostly overlap among training and evaluation data. In this case, is there any chance that VLMs memorize the scene patterns? Would there be a more constrained way to show that fine-tuned models are indeed "aligned" to human FoR, rather than memorizing scene structures?

---

[1] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics, Song et al., CVPR 2025

[2] SITE: towards Spatial Intelligence Thorough Evaluation, Wang et al., ICCV 2025

[3] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models, Jia et al., 2025

[4] ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models, Li et al., 2025

[5] Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities, Zhang et al., ICLR 2025

[6] Spatial Mental Modeling from Limited Views, Yin et al., 2025

### Questions
* Is SOFA a fully synthetic dataset? Is there a potential to extend the data synthesis pipeline to generate more realistic scenes? The current state of SOFA seems quite limited with a small number of categories and fixed scene configurations (e.g., layout, lighting, texture).

* In Section 4.3, the data preparation process for real-world scenarios seems quite hand-crafted and small in size. Have the authors considered showing generalization using existing real-image benchmarks (e.g., RoboSpatial [1], SITE [2]) instead?

### Soundness
3

### Presentation
2

### Contribution
2
