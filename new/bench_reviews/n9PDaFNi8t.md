Now I have sufficient calibration data. Let me carefully synthesize the review.

## Summary

OS-Atlas presents a foundation action model for generalist GUI agents, built on two key innovations: (1) a multi-platform GUI grounding data synthesis toolkit and the largest open-source cross-platform grounding corpus (13M+ elements covering web, desktop, and mobile), and (2) a unified action space that resolves naming conflicts across heterogeneous datasets. The model operates in three modes (Grounding, Action, Agent) and is evaluated across six benchmarks spanning three platforms, demonstrating state-of-the-art grounding accuracy and strong zero-shot OOD performance on step-level agent tasks, frequently outperforming GPT-4o.

## Strengths

- **Substantial data infrastructure contribution**: The cross-platform GUI grounding corpus (2.3M+ screenshots, 13M+ elements across 5 platforms, including desktop data absent from prior work) and the open-source synthesis toolkit represent a significant engineering contribution that will benefit the community beyond this specific model. Table 1 clearly shows the dataset exceeds prior open-source efforts in scale and platform coverage.

- **Strong grounding performance with clear data scaling evidence**: ScreenSpot results (Table 2) show OS-Atlas-Base-7B achieving 82.47% average accuracy, substantially outperforming UGround-7B (74.15%) without a planner. The data scaling analysis (Figure 3) demonstrates a clear positive correlation between grounding data volume and both IoU and accuracy, particularly for web where data is most abundant (7.7M elements → 48% IoU).

- **Effective unified action space**: The ablation in Figure 5 provides empirical evidence that naive multi-task training with conflicting action spaces degrades performance, and the unified action space (reducing from 17 to 10 action types) consistently improves SR and grounding accuracy across all three platforms.

- **Comprehensive evaluation breadth**: Six benchmarks across three platforms under two settings (zero-shot OOD and SFT) provides a thorough assessment. The correction of ScreenSpot (ScreenSpot-V2, 11.32% error rate) is a useful community contribution.

- **Valuable ablation insights**: Figure 4 decisively shows that web-only pre-training fails to generalize to desktop (desktop Text drops from 72.16% to 28.57%) and mobile, validating the importance of cross-platform data—a finding with practical implications for future data collection priorities.

## Weaknesses

### Fatal
None.

### Major

- **Claim of being an "open-source alternative to GPT-4o" is overstated relative to the evidence.** The abstract, contributions (point 4), and conclusion repeatedly frame OS-Atlas as a replacement for commercial VLMs like GPT-4o. However, the only full-episode agent evaluation (Table 3, OSWorld) still uses GPT-4o as the planner—OS-Atlas only replaces the grounding module. The claim rests entirely on step-level offline benchmarks (Tables 4–5), where OS-Atlas outperforms a vanilla GPT-4o that receives no GUI-specific training or unified action space. In the OSWorld end-to-end evaluation, the best OS-Atlas configuration achieves only 14.63% vs human 72.36%. The paper would be better served by framing OS-Atlas as a "strong domain-specialized foundation model that exceeds vanilla GPT-4o on step-level GUI tasks" rather than a general alternative.

- **No end-to-end agent evaluation with OS-Atlas in Agent Mode.** The paper motivates and defines an "Agent Mode" (Figure 1) where OS-Atlas should function as both planner and actor, but never evaluates this configuration on a realistic interactive environment like OSWorld. The grounding mode experiment (§4.3) tests only the grounding component with an external GPT-4o planner, and the agent task experiments (§5) are all step-level offline evaluations. Without at least one full-episode agent experiment, the claim that OS-Atlas serves as a "foundation action model for generalist GUI agents" is not adequately substantiated.

- **GPT-4o-generated instruction grounding labels lack quality validation.** Section 3.2 describes using GPT-4o with Set-of-Mark prompting to generate sub-instructions for the IG data—data that feeds directly into pretraining and whose contribution is analyzed in ablations (Figure 4). No human evaluation, inter-annotator agreement, or even a sampled audit of these labels is reported. If GPT-4o's phrasings systematically differ from real user instructions (e.g., being more verbose, more explicit about element references), the model's IG capability may overfit to GPT-4o's idiosyncrasies rather than genuinely learning instruction following. Given that the paper positions IG data as one of its key data innovations, the absence of any quality analysis is a gap.

### Minor

- **Cross-dataset data contamination is not thoroughly addressed.** While the paper notes removing ScreenSpot/Mind2Web/Omniact overlap from Wave-UI (§3.2 footnote), the agent training uses Mind2Web for fine-tuning while evaluating on GUI-Act-Web and OmniAct-Web, which may share websites or apps. For OS-Atlas-Pro (§5.4), which uses all 7 datasets for fine-tuning, no statement clarifies whether held-out splits are respected. Given that the central narrative is OOD generalization, explicit URL/app-level deduplication analysis would strengthen these claims.

- **Unified action space ablation conflates two effects.** The "w/o unified action" ablation (Figure 5) changes both the label space (17→10 action types) and the conflict resolution simultaneously. A smaller action vocabulary trivially increases Type EM accuracy, so it is unclear whether the benefit comes from resolving naming conflicts or from compressing the label space. A finer-grained ablation (e.g., unifying names without reducing vocabulary, or analyzing per-action performance) would clarify this.

- **OS-Atlas-4B shows concerning degradation in zero-shot OOD on some benchmarks.** On OmniAct-Web (Table 4), OS-Atlas-4B achieves only 46.74% Type accuracy versus GPT-4o's 79.33%—a massive gap that is not discussed. While the 7B model performs well, this suggests the foundation is less robust at smaller scales, which matters for practical deployment. The paper should acknowledge this gap and analyze its causes.

- **No comparison with UGround on agent tasks.** UGround-7B is the strongest prior grounding model (Table 2) and a directly comparable concurrent work, yet is absent from agent task evaluations (Tables 4–5). Including UGround on at least some agent benchmarks would more fully substantiate SOTA claims.

### Trivial
- The paper states "over 13 million GUI elements" in the abstract but Table 1 reports 13.58M; this minor inconsistency could be harmonized.

## Nice-to-Haves

- Failure case analysis with categorized errors (e.g., grounding failures vs. action type prediction failures) would reveal systematic limitations and guide future work.
- Per-category grounding performance breakdown (text vs. icon/widget) in main tables, not just in ablations.
- Cross-platform transfer analysis (e.g., train on web+mobile, test on desktop) to quantify the value of multi-platform data more precisely.
- End-to-end Agent Mode evaluation on OSWorld with OS-Atlas serving as both planner and actor.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Limited architectural novelty" (Neutral Reviewer)**: The paper never claims architectural novelty. Its stated contributions are data, unified action space, and the resulting model performance. Criticizing a paper for not contributing something outside its stated scope is not fair.

- **"Insufficient training details / reproducibility" (Neutral Reviewer)**: The paper refers to Appendix E for training details. Lack of appendix content (which was removed from the PDF excerpt) is not a fair criticism of the paper itself. Hyperparameter and computational cost details are standard appendix material.

- **"Data imbalance across platforms" (Human Finder)**: The paper openly addresses this—the desktop has only 54K screenshots, mobile 285K. The ablation in Figure 4 explicitly shows the consequences of having less data for certain platforms, and Figure 3 shows lower desktop IoU explicitly. This is already addressed.

- **"Very low absolute success rates on OSWorld" (Human Finder)**: The paper is transparent about this in Table 3, showing human performance at 72.36%. Low absolute performance on a challenging benchmark is not a weakness of the paper; it is a characterization of the field's current state.

- **"Fairness of comparisons / backbone advantages" (Human Finder)**: The paper uses two backbones (InternVL-2-4B and Qwen2-VL-7B) and consistently shows OS-Atlas improvements over both respective baselines. The Qwen2-VL-7B backbone having GUI data makes the comparison fair since OS-Atlas must demonstrate incremental improvement over an already strong base. This is not an unfair advantage.

- **"Copyright concerns about crawled web data" (from UGround review analogy)**: This is standard practice in the field and not a specific weakness of this paper.

- **"No variance or confidence intervals" (Spark/Harsh)**: Single-run evaluation is the norm for large-scale VLM evaluations in this venue. Requesting confidence intervals is above the community standard for this type of work.

- **"Requesting confidence intervals" (Ferret-UI 2 review analogy)**: Same reasoning as above—not standard practice for this type of large-scale evaluation.

## Novel Insights

The ablation revealing that web-only grounding data catastrophically fails to generalize to desktop (Text accuracy drops from 72.16% to 28.57%) is an important finding that contradicts the intuition that GUI understanding transfers readily across platforms. This empirically establishes that cross-platform grounding data is not merely beneficial but necessary, providing clear guidance for future data collection efforts: investing in desktop and mobile-specific data collection infrastructure is essential, not optional.

## Suggestions

1. **Temper the "alternative to GPT-4o" framing** to "a strong domain-specialized open-source model that exceeds vanilla GPT-4o prompting on step-level GUI tasks"—this is honest and still impactful.
2. **Add at least one end-to-end Agent Mode evaluation** on an interactive benchmark (e.g., OSWorld with OS-Atlas generating both plans and actions) to substantiate the "foundation for agents" claim.
3. **Audit a sample of GPT-4o-generated instruction grounding data** (even 100 examples rated by humans) and report quality statistics to validate this critical data pipeline.
4. **Discuss the OS-Atlas-4B performance gap on OmniAct-Web** in the zero-shot OOD setting and analyze why action type prediction degrades so sharply.

## Score and Decision

**Calibration comparison:**
- UGround (Oral, avg ~7.75): Similar scope (GUI grounding model, large synthetic data, ScreenSpot eval), but stronger narrative coherence and end-to-end agent evaluation including online settings. OS-Atlas adds desktop data, unified action space, and more platforms but has weaker agent-mode evaluation and overclaims.
- Grounding MLLM in GUI World (Poster, avg 6.0): Weaker results, less multi-platform coverage. OS-Atlas is clearly stronger.
- Aguvis (Reject, avg 5.5): Similar unified-action concept but lacked ablation validation. OS-Atlas has better ablations and stronger data contribution.
- Ferret-UI 2 (Poster, avg ~6.3): Multi-platform support, GPT-4o data annotation, but less thorough ablation analysis. OS-Atlas has comparable strengths and similar data quality concerns.

OS-Atlas sits clearly above the rejected GUI agent papers (Aguvis, OmniParser, AutoGUI) due to its comprehensive data infrastructure, strong empirical results, and meaningful ablations. It sits below UGround (Oral) because UGround has more complete agent evaluation (including online benchmarks), less overclaim, and end-to-end agent demonstrations. The overclaim about being a GPT-4o alternative and the missing end-to-end agent mode evaluation are the main factors preventing a higher score.

OS-Atlas is a solid, impactful systems/data contribution with useful engineering insights for the community. The core contributions (data toolkit, unified action space, ScreenSpot-V2) are real and valuable. The weaknesses are significant but do not invalidate the results.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>