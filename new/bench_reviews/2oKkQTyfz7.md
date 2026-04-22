Now I have sufficient information to write the final review. Let me synthesize everything.

## Summary

The paper introduces GSA-VLN (General Scene Adaptation for Vision-and-Language Navigation), a new task requiring VLN agents to continuously improve as they execute instructions in persistent environments through long-term memory and parameter adaptation. To evaluate this task, the authors create GSA-R2R, a dataset that significantly expands environmental diversity (150 buildings across 20 types from HM3D/MP3D, including OOD non-residential buildings) and instruction diversity (7 styles including OOD Scene and User instructions generated via a three-stage LLM-orchestrated pipeline). The authors benchmark existing adaptation methods and propose GR-DUET, which maintains a global topological graph across episodes with environment-specific pretraining, achieving the best results across all GSA-R2R splits.

## Strengths

- **Novel and important task formulation**: GSA-VLN addresses a meaningful gap between standard VLN evaluation (one-shot, frozen parameters) and real-world deployment in persistent environments. The formalization with memory bank (Eq. 1), memory access (Eq. 2), and parameter update (Eq. 3) cleanly distinguishes the task from standard VLN and TTA (Section 3.2).

- **Substantially expanded dataset**: GSA-R2R addresses two concrete deficiencies in existing VLN benchmarks: expanding from 29 evaluation scenes (6 types) in R2R to 150 scenes (20 types) including OOD non-residential buildings, and from 1 instruction style to 7 styles. Table 1 provides direct quantitative comparison showing GSA-R2R has the greatest number and diversity of scenes, paths, and instructions among embodied navigation datasets.

- **Insightful benchmarking that reveals failure modes**: The paper demonstrates that TTA methods (TENT, SAR) degrade performance because entropy-based confidence becomes unreliable after error accumulation in sequential decision-making (Section 4.3.2, Table 4), and that existing memory-based methods catastrophically fail due to overly long history embeddings. These are non-obvious findings valuable to the community.

- **Strong empirical improvements from GR-DUET**: The method achieves 11.6% and 8.5% SR improvements over vanilla DUET on Test-R-Basic and Test-N-Basic (Table 4), and up to ~11% SR improvement on Scene/User instruction splits (Tables 5–6). Ablation studies (Tables 7–8) confirm both the pretraining strategy and buffer-based graph construction contribute.

- **Three-stage instruction pipeline**: The speaker → VLM refinement → LLM role-playing pipeline (Section 3.3.2) is principled, with human evaluation (Table 2) confirming ~80% path-instruction alignment and high style distinctiveness (96.1% for Scene).

- **Experimental rigor**: Evaluation with batch size 1 in online manner, three runs with randomly sequenced instructions reporting mean ± standard error, and consistent CLIP-ViT/B-16 features across baselines ensure fair comparisons.

## Weaknesses

### Fatal

None.

### Major

- **No demonstration that agents actually adapt *over time***: The paper's central claim is that GSA-VLN enables agents to "continuously improve as they execute instructions in a specific environment" (Section 3.2, abstract). Yet all reported results (Tables 3–6) are aggregate metrics over all episodes within a split, never showing performance as a function of the number of prior instructions executed in that environment. The most natural evaluation for an "adaptation" task—an episode-order performance curve (e.g., SR for episodes 1–100 vs. 101–200 vs. 201–300)—is entirely absent. Moreover, the very low standard errors across three random episode orderings (e.g., GR-DUET on Test-R-Basic: 69.3 ± 0.2 SR) suggest that episode order barely affects outcomes, which actively undermines the adaptation narrative. This is a significant gap between the framing ("continuously improve over time") and the evidence provided.

- **Map access vs. adaptation conflation**: GR-DUET's primary mechanism is maintaining a global topological graph—essentially giving the agent a growing map. The 11.6% SR improvement over vanilla DUET (Table 4) could be entirely attributed to map access rather than "adaptation." The paper never isolates these factors. A critical control experiment—DUET with an oracle topological map at test time (no adaptation, but map provided)—would distinguish whether the gains come from "adapting" or from "having a map." Without this, the paper's theoretical framing around parameter adaptation (Eq. 3) and memory-based adaptation is not validated by the method itself, which performs *no* parameter updates at evaluation time and only updates the graph.

- **GR-DUET does not implement the task's own formulation for parameter adaptation**: Eq. 3 presents parameter update via unsupervised learning on the memory bank as a core component of GSA-VLN. GR-DUET performs *no* parameter updates at evaluation time—it only updates the topological graph. Meanwhile, the optimization-based baselines (TENT, SAR, BT, MLM, MRC) that do update parameters either fail or provide marginal gains. This creates a tension where the proposed method doesn't implement the task's own theoretical formulation, and the methods that do implement it don't work. The paper acknowledges this indirectly in the conclusion ("we aim to explore more unsupervised learning approaches"), but the disconnect between task formulation and method remains a significant concern.

### Minor

- **Unfair comparison against memory-based baselines**: TourHAMT and OVER-NAV achieve catastrophically low performance (14.9% and 22.3% SR on Test-R-Basic, Table 4). The paper attributes this to "excessively long history embeddings as input, which confuses the model"—which is exactly what happens when methods designed for IVLN's ~6–100 episodes per environment are applied unmodified to GSA-R2R's 600 episodes. No accommodation is made (e.g., history truncation, sliding windows). While this reflects how these methods actually perform out-of-the-box, the authors should acknowledge this asymmetry more explicitly and note that these baselines are being evaluated outside their design regime.

- **Train-test graph quality mismatch not analyzed**: During pretraining, the agent receives the "complete ground truth topological map" (Section 4.1, line 182–183); during evaluation, the graph is built incrementally from potentially failed trajectories. This mismatch is acknowledged but not analyzed—e.g., how much does graph quality degrade, and how does graph completeness correlate with SR? This would strengthen the paper's understanding of its own method.

- **Instruction quality ceiling**: Table 2 shows 23.4% of Scene instructions don't accurately describe their path. This caps achievable SR and makes absolute numbers harder to interpret. The paper discusses this in the human evaluation but could provide more analysis of how this ceiling affects different splits.

### Trivial

- The ablation in Table 8 uses α to represent both "proportion" and "buffer size," which is slightly confusing, though the paper does explain this choice (Section 4.4).

## Nice-to-Haves

- **Episode-order performance curves**: Plot SR as a function of episode number within each environment (binned by execution order). This would directly validate the "continuous improvement" claim and is the single most impactful addition the paper could make.
- **DUET + oracle map baseline**: A DUET variant given an oracle topological map at test time would isolate map access vs. adaptation contributions.
- **Graph quality analysis at evaluation time**: Measuring how accurately the agent-built graph matches ground truth and correlating graph completeness with SR would address the train-test mismatch concern.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that "low standard errors actively contradict the adaptation narrative"**: The low standard errors mean that *across random orderings* the aggregate performance is consistent, but this doesn't directly contradict adaptation—it could mean that all orderings benefit similarly from adaptation on average. However, it does weaken the case that episode order matters, so a related but softened version of this point is kept as a Major weakness.

- **Harsh Critic's claim about the motivating analogy being "misleading"**: The observation that GR-DUET's smallest gains come from instruction style adaptation (Scene: +8.5% vs. Basic: +11.6%) is an interesting analysis point but doesn't make the introduction misleading—the introduction motivates both spatial and stylistic adaptation, and GR-DUET improves on both.

- **Data leakage concerns for EnvDrop speaker (Footnote 2)**: The paper discloses that "We include the evaluation splits in the training of EnvDrop to improve quality" (Footnote 2). This is a disclosure/completeness issue rather than a fundamental flaw in the dataset itself, since the speaker is a tool for instruction generation, not the evaluated agent. This is not a critical concern.

- **ScaleVLN data leakage presentation**: The paper clearly marks ScaleVLN with a † symbol in Table 3 and provides a footnote (Footnote 3) explaining the leakage. The disclosure, while brief, is present. This is a presentation preference rather than a substantive flaw.

- **Demand for "adapted memory baselines" (TourHAMT/OVER-NAV with truncation)**: While it would be nice to see these baselines adapted to the new setting, requiring the authors to redesign competitor methods is unreasonable. The paper already demonstrates what happens with the out-of-the-box implementations and explains why they fail. This is a nice-to-have, not a major flaw.

- **ID/OOD categorization verification**: The claim that HM3D residential buildings may not truly be "in-distribution" because they have different visual characteristics from MP3D is speculative without evidence. The paper's categorization is reasonable (residential = ID since training is on residential), and the performance gaps between R and N splits (Table 3) validate the categorization empirically.

- **Strength Finder's claim about "Figure 1 effectively motivates the task"**: Generic and does not provide specific evidence beyond what's obvious.

## Novel Insights

The paper reveals an important distinction between two adaptation mechanisms in VLN: memory-based (graph/map building) and optimization-based (parameter updates). The empirical finding that the former works well while the latter largely fails in this setting—combined with the specific diagnosis that entropy-based TTA fails because "errors accumulate over time, making entropy measures meaningless after an incorrect step" (Section 4.3.2)—suggests that the VLN community's adaptation paradigm may need to shift from test-time parameter optimization toward structural memory accumulation for persistent-environment settings. However, the paper's own method doesn't fully bridge this gap, as it demonstrates "adaptation" only in the sense of accumulating spatial memory, without showing temporal improvement across episodes.

## Suggestions

- The single most impactful addition: report performance curves across episodes (e.g., SR for first 100 episodes vs. next 100 vs. last 100 within each building). Even if adaptation effects are small, reporting them (or their absence) is essential for the paper's core claim.
- Add a DUET + oracle graph experiment to disentangle the contribution of map access from adaptation.
- In the conclusion or discussion, explicitly acknowledge that GR-DUET demonstrates memory-based spatial accumulation but not temporal improvement or parameter adaptation, and frame the future work accordingly.

## Score and Decision

**Calibration anchors:**

High-scoring papers:
- EQA-MX (avg 8.0, Accept/spotlight): Large-scale dataset with novel multimodal EQA tasks + well-matched VQ-Fusion method — stronger alignment between task formulation and method, comprehensive experiments with no gap between claims and evidence.
- Selective Visual Representations for Embodied AI (avg 7.5, Accept/spotlight): Novel codebook approach with clear experimental validation.

Medium-scoring papers:
- WebCanvas (avg 4.75, Withdrawn): Benchmark dataset with partial evaluation metric — partial coverage of the full task formulation, similar to GSA-VLN's gap between task definition and method.
- CIL with Pre-trained Models (avg 4.25, Withdrawn): Benchmark + method where simple baseline beats SOTA, claimed adaptation not fully explored.
- BeGin graph continual learning (avg 4.0, Withdrawn): Benchmark contribution with no novel method proposed.

Low-scoring papers:
- cPmLjxedbD (avg 1.0, Reject): No contributed techniques, no dataset, no evaluations — far below this paper.
- MrGca1Q7mK (avg 1.5, Withdrawn): Algorithm never implemented — not comparable.

This paper is substantially better than the low-band papers (real dataset, real method, real experiments). Compared to the high-band papers (EQA-MX at 8.0), it falls short because its method doesn't implement its own task formulation (no parameter adaptation, no episode-order analysis). Compared to the medium-band benchmark papers (WebCanvas at 4.75, BeGin at 4.0), this paper is stronger because GSA-R2R is a more substantial dataset contribution with a genuine (if incompletely validated) method and insightful failure mode analysis. The key tension is: this is a solid dataset paper with a method that works but doesn't validate the core adaptation claim. This lands it in the 5–6 range: the dataset contribution is meaningful, but the incomplete validation of the adaptation narrative and the map-vs-adaptation conflation are significant concerns that prevent acceptance without revision.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>