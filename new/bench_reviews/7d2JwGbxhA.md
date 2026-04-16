## Summary
This paper proposes OCEBO, a self-distillation framework for object-centric learning that trains from scratch on real-world images by updating the target encoder as an EMA of the object-centric encoder. Its key technical ingredient is cross-view patch filtering, which suppresses unreliable early targets and appears crucial for preventing slot collapse; empirically, the method achieves non-collapsed training on COCO/COCO+ and reaches performance in the vicinity of prior object-centric systems that rely on large pretrained non-object-centric teachers.

## Strengths
- **Targets an important bottleneck in real-world object-centric learning.** The paper is clearly motivated by a real limitation of prior slot-based methods: reliance on frozen pretrained target encoders. The framing is consistent across the paper, and the proposed solution—bootstrapping the teacher from the object-centric encoder itself—is a meaningful conceptual departure from the frozen-target setup.
- **The proposed cross-view patch filtering appears both novel and genuinely necessary.** The motivation in Sec. 3.3 is intuitive, and Table 1(a) strongly supports its importance: removing patch filtering leads to collapse (`d=0.02`) and large drops in FG-ARI/mBO (e.g., MOVi-E FG-ARI from 54.8 to 27.7).
- **The paper demonstrates a real technical achievement: from-scratch training on real images without collapse.** Given the paper’s direct ablations and prior context, avoiding slot collapse under random initialization on COCO-scale real-world data is nontrivial and is likely the paper’s strongest contribution.
- **Ablations are reasonably informative for the core system pieces.** Table 1 covers patch filtering, the object-centric loss, the sharpening stage, and a larger-data regime. In particular, the collapse of the `λ_oc=0` variant and the gains from sharpening/data scale provide useful evidence that the method is not trivially working by virtue of standard SSL alone.
- **The paper is generally candid about limitations of direct SOTA comparison.** Sec. 4.3 explicitly notes that OCEBO is not directly comparable to methods using stronger pretrained encoders and different decoders/training refinements, which is an honest presentation choice.
- **The work is potentially valuable to the community even if some claims are overstated.** Showing that object-centric pretraining from scratch on real data is feasible is itself useful and may open a worthwhile line of work.

## Weaknesses
###: Fatal
None.

### Major:
- **The paper overstates what its scaling experiments establish.** The headline claims in the abstract/introduction/conclusion include “removing the upper bound on performance,” “enabling large-scale pretraining,” and “scales well with dataset size.” In the main paper, the scaling evidence is limited to COCO (~118k) vs COCO+ (~241k), i.e., essentially a 2× data increase (Table 1(d)). That is enough to show positive scaling over this range, but not enough to substantiate stronger claims about removing an upper bound or supporting the broader “object-centric foundation models” narrative. The paper does cite prior saturation around ~16k images, but it does not provide a matched scaling comparison in the main paper to show that OCEBO truly breaks that ceiling under comparable conditions.
- **The central mechanism claim—EMA teacher improvement via injected object-centric inductive bias—is plausible but not directly demonstrated.** Much of the conceptual contribution rests on the hypothesis from the introduction that prior EMA teacher bootstrapping fails because the teacher lacks object-centric inductive biases, and that making the teacher an EMA of the object-centric encoder fixes this. However, the empirical support is indirect: Table 1(b) sets `λ_oc=0`, which changes the training problem substantially rather than isolating “teacher object-centricity,” and Figure 3 is qualitative. There is no controlled experiment that directly compares alternative EMA teacher constructions under otherwise matched settings, nor any analysis of how the teacher representation evolves over training to verify the proposed drift explanation.
- **The state-of-the-art comparison is contextual rather than causal evidence for the proposed mechanism.** In Table 2, OCEBO is compared to systems that differ in multiple ways at once: pretrained encoder strength, decoder family, and additional high-resolution/fine-tuning stages. The authors acknowledge this, which is good, but it limits what Table 2 can support. It shows OCEBO is promising and in the ballpark despite no external pretrained teacher; it does **not** isolate that target-encoder bootstrapping itself is the reason for competitiveness.
- **“Comparable to state of the art” should be stated more carefully.** The results are mixed across metrics, and OCEBO is clearly behind FT-DINOSAUR on several mBO numbers (e.g., MOVi-C 27.3 vs 44.2; EntitySeg 16.0 vs 28.4 in Table 2). The paper’s discussion of FG-ARI/mBO tradeoffs is fair, but the strongest safe conclusion is that OCEBO is competitive given its much weaker pretraining setup—not that it robustly matches the strongest baselines.

### Minor
- **The mask sharpening stage weakens the elegance of the end-to-end bootstrapping story.** Sec. 3.4 introduces an additional 100-epoch stage with a frozen target and an `ℓ2` loss because masks otherwise “sometimes” lack clear boundaries. This is a reasonable engineering fix, but it suggests the main self-distillation objective alone does not fully solve the segmentation-quality problem.
- **Some methodological details are hard to parse from the extracted text, and the notation in Sec. 3.2 appears inconsistent.** In Eqs. 2/4/6, the notation for student vs teacher distributions is confusing in the extracted version (the same symbols appear reused). This may partly reflect extraction issues, so I do not treat it as a strong flaw, but it does reduce clarity of the method as presented here.
- **The collapse metric `d` is useful as a diagnostic but not yet validated as a robust metric.** The paper defines `d` in Sec. 4.2 and uses it consistently in Table 1, but does not show broader evidence that it reliably tracks human-judged collapse beyond the few presented variants.
- **The paper raises an important dataset-dependence issue but does not deeply analyze it.** The ImageNet result in Sec. 4.2 is interesting and supports the claim that suitable multi-object scene data matters. However, this point is only briefly explored despite being highly relevant to the paper’s larger “large-scale pretraining” ambition.
- **Zero-shot evaluation is well motivated, but reporting only a subset of datasets in the main paper limits the empirical picture.** The paper says conclusions are identical on the remaining datasets, which is plausible, but the main text still gives only partial benchmark coverage.

### Trivial
- **A robustness analysis over evaluation slot count would be useful.** Since different slot counts are selected per dataset at evaluation, some sensitivity analysis would help interpret zero-shot robustness, though this is not a core flaw.

## Nice-to-Haves
- Add a more convincing scaling curve with several data regimes, ideally including smaller subsets and, if possible, a matched comparison to prior frozen-target approaches.
- Include a controlled experiment that isolates the role of object-centricity in the EMA teacher, rather than changing multiple ingredients at once.
- Analyze the ImageNet/curation issue more systematically to clarify what data properties are required for successful object-centric pretraining.
- Provide qualitative failure cases and training-evolution visualizations to better support the filtering/collapse-avoidance narrative.
- If feasible, combine OCEBO with stronger decoders or high-resolution refinement to test whether its benefits are orthogonal to existing architectural improvements.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Reproducibility concern based on code/models being released upon acceptance.** Per instructions, I remove this as a reproducibility nitpick rather than a substantive paper weakness.
- **Requests to compare against additional related work not already discussed.** I do not include missing-related-work complaints because I cannot externally verify completeness, and the paper already situates itself against several core object-centric baselines.
- **Criticism that comparisons are unfair because baselines use stronger pretrained encoders.** This asymmetry actually favors the baselines, not the authors’ method; per instruction, such fairness complaints should not be used against the paper.
- **Strong criticism of notation/formatting alone.** The equation notation in the extracted text is somewhat confusing, but pure presentation nitpicks were filtered unless they materially affected interpretability.

## Novel Insights
The most interesting synthesis here is that the paper’s strongest contribution is narrower than its broadest narrative: it convincingly establishes **feasibility** of from-scratch object-centric training on real images via a curriculum-like filtering mechanism, but only **suggests**, rather than proves, the stronger mechanistic story that EMA teacher bootstrapping succeeds specifically because object-centric inductive biases are injected into the teacher. Put differently, the paper already has a publishable core contribution without needing the stronger “upper bound removed / foundation-model path unlocked” framing; tightening the claims around feasibility and early scaling would make the work more convincing.

## Suggestions
- Reframe the main claim more conservatively around **feasibility and promising early scaling**, rather than “removing the upper bound” or establishing large-scale pretraining in a strong sense.
- Add a controlled experiment isolating the proposed mechanism: hold architecture/training fixed and vary only how the teacher is constructed/updated.
- Expand the scaling study beyond two data points and, if possible, include a matched frozen-target baseline under the same regime.
- Strengthen the discussion around Table 2 to emphasize “competitive given no large pretrained teacher” rather than broad comparability to tuned SOTA.
- Analyze the dataset requirement more deeply, since this is likely central to whether object-centric pretraining can truly scale.
- Provide a brief validation of the collapse metric `d`, even if only by correlating it with qualitative collapsed/non-collapsed examples across more runs.

## Score and Decision
**Assessment on key axes:**  
- **Originality:** good; the EMA-bootstrapped object-centric teacher plus patch filtering is a meaningful new angle.  
- **Importance:** good; the question of escaping frozen non-object-centric teachers is important for this area.  
- **Support for claims:** mixed; the paper supports from-scratch feasibility well, but overreaches on scaling and mechanism.  
- **Experimental soundness:** solid but incomplete; key ablations exist, yet the central causal/mechanistic claim is not isolated, and scaling evidence is limited.  
- **Clarity:** generally decent, with some notation/presentation ambiguities in the extracted text.  
- **Community value:** good; even a narrower version of the claim is useful to the object-centric learning community.

**Calibration against human-review anchors:**  
- Compared with **Cycle Consistency Driven Object Discovery** (`f1xnBr4WD6`, scores 8/8/6/5, accepted), this paper is weaker because that work appears to validate claims more comprehensively, including downstream utility, while OCEBO’s strongest claims outrun its evidence.  
- Compared with **On the Transfer of Object-Centric Representation Learning** (`bSq0XGS3kW`, scores 6/5/3/6, accepted), OCEBO feels similarly in the “promising, useful, but not fully nailed down” band; unlike that paper, OCEBO has a clearer technical novelty, but it also overclaims more on scalability/mechanism.  
- Compared with weaker borderline/reject papers such as **Adaptive Slot Attention** (`EaLfdBPlIh`, all 5s, reject/withdrawn) and **CODiT** (`KgN0mo6pLo`, 5/6/3/5, reject), OCEBO has a more substantive and credible core contribution, and its main weaknesses are about claim calibration rather than lack of contribution.

Overall, this strikes me as a **borderline-to-weak accept on substance**, but since I must give a binary recommendation and the claim/evidence gap is material, I lean **Reject** in its current form unless the venue is particularly favorable to promising feasibility papers.

**Score: 5.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>