# DP-Nav: Dynamic Exploration Driven by Semantic Region Potential for Zero-shot Visual Navigation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Visual navigation requires the agent to autonomously navigate to a specified goal based on sequential visual perception. A key challenge is to achieve target localization and optimize the path simultaneously. However, most existing frontier-based methods rely on static navigation policies, which update the target frontiers at fixed time intervals to guide the agent's exploration. These approaches cannot dynamically assess potential regions encountered during navigation, thereby preventing timely policy adjustments. Moreover, the presence of multiple frontiers within the same region often leads to repeated exploration of identical regions, further exacerbating path redundancy and inefficiency. To address the above limitations, we propose DP-Nav, a novel dynamic navigation framework driven by the potential of semantic regions. Our approach first identifies distinct semantic regions from sequential visual perception and treats an independent semantic region as a policy unit. Furthermore, we introduce a Scoring-Screening Mechanism (SSM) that evaluates and filters these semantic regions based on their potential utility. Then SSM assigns exploration priorities to different regions, selecting the semantic region with the highest potential value for the agent's subsequent exploration. More significantly, we design a Dynamic Policy Trigger (DPT) module that enables on-demand activation of the SSM, allowing the agent to dynamically adapt its exploration policy in response to environmental changes and perceptual feedback, thereby addressing the rigidity of static policies. Extensive experiments on Object Goal Navigation, Text Goal Navigation, and Instance Image Goal Navigation across Gibson, HM3D, and MP3D datasets demonstrate that DP-Nav achieves SOTA performance and improves path efficiency by about $7\%\sim17\%$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes DP-Nav, a dynamic visual navigation framework driven by semantic region potential. It represents the scene with a “region–junction graph” and aggregates each region via representative views. At the policy level, a Scoring–Screening Mechanism (SSM) fuses VLM semantic scores with path cost, and four Dynamic Policy Triggers (DPT) enable on-demand replanning. Experiments on ObjectNav, TextNav, and InstanceNav in Gibson/HM3D/MP3D report improvements over zero-shot baselines along with ablations.

### Strengths
* **Cross-task and cross-dataset zero-shot comparisons**  
  • On HM3D ObjectNav, DP-Nav outperforms UniGoal by +8% SR and +10.5% SPL.  
  • On HM3D TextNav, it surpasses UniGoal by +5.4% SR and +8.1% SPL.  
  • On InstanceNav, it shows substantial gains over GOAT and PSL.  

* **Relatively complete implementation and appendix details**  
  • The paper specifies the VLM used (Qwen2.5-VL-3B-Instruct) and the step limit / success radius.  
  • The appendix explains prompt usage scenarios (scoring prompts and deep-exploration prompts).

### Weaknesses
* **Insufficient direct validation that VLM scoring reflects “navigability”**  
  • The textual example in Fig. 4 (“Looks like a restroom, give 0.8”) appears to measure semantic visibility rather than reachability.  
  • SSM treats $VLM(V*_r)$ as the core of region potential, yet provides no correlation or calibration curves versus success rate or SPL.  
  • Only one VLM model is used; there is no cross-model consistency or degraded controls (e.g., random or heuristic scores).  

* **Coverage and omissions of hand-crafted triggers are not systematically evaluated**  
  • The four triggers are rule-based; despite gains, there is no quantitative diagnostic for misses and false alarms.  
  • The view-update gate $φ_pug$ being too small induces “action oscillation,” while too large causes “policy lag,” indicating strong hyperparameter sensitivity.  
  • No specific analysis is provided for difficult layouts such as long corridors, loops, dead ends, or repeated visits.  
  • For the Region-Reached trigger that re-invokes the VLM to decide “deep exploration,” the false-trigger rate and backtracking cost are unreported.  

* **Limited interpretability and sensitivity analysis for SSM weighting and screening**  
  • For preference score ( $\mathrm{PS}=\gamma\cdot \mathrm{VLM}+(1-\gamma)e^{-\lambda\cdot \mathrm{Path}}$ ), γ and λ appear only in a table without theoretical or empirical calibration procedure.  
  • The assignment to progressing/backtracking lists depends on threshold sets (ϕ, ζ), but systematic sensitivity curves are missing.  

* **Experimental fairness and statistical rigor need strengthening**  
  • The SOTA claims in Table 1 lack variance, confidence intervals, and multiple random seeds.  
  • Although there are ablations on triggers and SSM, significance testing and stratified comparisons across scenarios are absent.

### Questions
* **Validate and calibrate VLM scoring for navigability**  
  • Construct counterfactual scenes that are semantically visible yet hard to reach, and report correlations and segmented trends between VLM scores and success rate/SPL. Provide an ROC curve for ζ with target AUC above random.  
  • Conduct cross-model consistency experiments (add 1–2 more VLMs) and degraded controls (random scores, simple heuristics) to verify robust ranking. Target an improvement of at least x% over degraded controls, presented as new columns in a Table 1–style summary.  
  • Report sensitivity scans of ζ and performance under OOD conditions (occlusion, camouflage, atypical appearances); provide threshold–performance curves and select a stable operating point.  
  • In Appendix A.11, include full prompts, few-shot examples, failure cases, and statistics of failure types (false positives/negatives) aligned with the subjective text in Fig. 4.  
 
* **Systematically evaluate trigger coverage to reduce omissions and false triggers**  
  • Build benchmark sets for long corridors, loops, dead ends, and repeated visits; report trigger rates, false-trigger rates, and miss rates for each of the four triggers with PR curves or box plots.  
  • Perform fine-grained scans over $φ_pug$ and couple them with different numbers of representative views (RP) to quantify “oscillation” versus “lag,” reporting policy-switch frequency and average path redundancy.  
  • Add a “trigger failure type” column to a Table 3/4–style ablation, quantifying each failure category and its contribution to SR/SPL to verify improved coverage.  

* **Enhance experimental fairness and statistical rigor**  
  • For all entries in Table 1, report mean ± standard deviation with 3–5 random seeds.  

* **Assess VLM output consistency and sampling variance (same environment, different sampling)**  
  • Fix the environment and camera pose; vary the VLM random seed, temperature, or sampling strategy (top-k, top-p). Re-score multiple times and report consistency of region scores and rankings using ICC, Kendall τ, or Spearman ρ. Criterion: τ or ρ ≥ 0.8; if lower, analyze causes.  
  • Apply small pose jitter within the same environment (translation ±5 cm, rotation ±5°), resample representative views, and replicate scoring. Compute per-region score variance and coefficient of variation; plot variance versus jitter magnitude. Criterion: within task tolerance, score variance should not cause frequent flips among the top-k regions.  
  • Test prompt robustness via minor paraphrases and synonym substitutions; compare ranking differences and report the similarity distribution. Criterion: high median similarity with a small interquartile range.  
  • Quantify the effect of consistency on navigation outcomes by comparing the distributions of SR and SPL across repeated scoring-driven plans. Report mean differences and confidence intervals and the maximum observed fluctuation. Criterion: SR/SPL variation remains within a preset bound, for example absolute difference ≤ 1 percentage point.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a method for constructing a topological graph for object-goal navigation. This graph consists of two types of nodes: region nodes and junction nodes, where region nodes represent areas to explore and junction nodes connect region nodes. A Score-Screening Mechanism (SSM) is proposed using LVM to assign scores to each region node and divide them into a processing list and a backtracking list. A Dynamic Policy Trigger (DPT) module is introduced to activate SSM and dynamically drive the agent's exploration. Experiments are conducted on three types of navigation tasks across three datasets, achieving superior performance.

### Strengths
1. This paper identifies two issues in existing frontier-based exploration (FEB) navigation: frontiers are updated at fixed time steps, and the agent may jump inconsistently between different regions due to each frontier being treated individually.

2. The method part is technically sound. Note that it refers to the correctness and feasibility, with a clear illustration of nice-looking figures. 

3. The experiment performance is superior to the recent works on all three tasks across different datasets. In some cases, the SR and SPL improvement is evident by a margin.

### Weaknesses
There are several weaknesses concerning novelty and claim:

1. Sections 3.3.1 to 3.4.3 are more engineering-oriented; that is, it is hard to identify new network architectures, algorithms, or pipelines. This is not questionable with respect to the correctness of the method, but rather concerns the novelty. The authors should highlight the key novelty that is different from previous works.

2. In Related Work (Line 141 - 142), the authors briefly mention that the proposed method is a policy with self-adaptation, which is different from previous graph-based methods. However, it lacks support or evidence for "self-adaptation" in the Method section. The main distinction from previous graph-based methods is not highlighted.

3. The authors mention two key issues in existing FBE methods in the Introduction (Line 49 - 76), but how the proposed method addresses these two issues is not clearly stated. For example, does the DPT module update frontiers at varying time steps? Do we only consider one frontier in each semantic region, or consider all frontiers in the semantic region as a whole for the decision?

4. The authors mention "semantic" regions several times. But in Section 3.3.1, the region node is constructed purely based on geometry with traditional approaches. It is questionable these regions truly contain meaningful semantics.

5. The four triggers described in Section 3.4.4 are not clearly presented. It is hard to follow this section. I regard this as a writing issue. In addition, it is unclear whether others can borrow this idea without using the graph built in Section 3.3.1, as these triggers closely coupled with the graph.

### Questions
1. Can we merge two junction nodes in the topological graph regardless of their distance? In Line 229 - 230, there is a radius threshold to fuse junction nodes, which I think can be removed.

2. Why do we need to differentiate the Processing list and the Backtracking list? We can just explore regions based on the combination of VLM scores and distance in descending order.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes DP-Nav, which performs VLM-guided dynamic exploration on a Region–Junction Graph (RJG). The system comprises four modules: (1) RJG construction from RGB-D by extracting traversable skeletons and identifying region/junction nodes; (2) Representative Perspectives (RP) that store multi-view evidence per region; (3) a Scoring–Screening Mechanism (SSM) that fuses a VLM score and distance-based path cost to split nodes into a Progressing List (PL) and a Backtracking List (BL); (4) a Dynamic Policy Trigger (DPT) with four events (region discovery, perspective update, junction pass, region reached) to activate SSM for on-demand replanning. Experiments on ObjectNav, TextNav and InstanceNav over Gibson/HM3D/MP3D report improvements in SR/SPL.

### Strengths
- The decision unit is lifted from frontier points to the region level.
- Figures are visually appealing and clearly convey the main ideas.
- The writing is clear and easy to follow.

### Weaknesses
- Although the specific implementation details differ slightly, the overall idea appears quite similar to existing approaches that also rely on region-based potential estimation and multi-scale scoring mechanisms. Without a clearer articulation of the conceptual innovation or the insight that motivates this design, the contribution feels incremental.
- Heavy hand-tuning of triggers and thresholds makes the pipeline look like a “strategy stack”; the learning-based adaptivity appears limited. Sec. 3.4.4 defines four trigger types, specifies update rules via formulas, and fixes a trigger priority order—these timings and conflict resolutions are hand-crafted. In SSM, the fusion weights/thresholds (e.g., the near-distance and low-score thresholds and diversity parameters in Eqs. 7–11) are also manually set, with little justification or comparative evidence as to why these choices are necessary.
- RJG construction relies on skeletonization with depth thresholds, traversability masks, and node degree. It is unclear whether this geometric heuristic is stable in challenging cases such as cross-level transitions, stairs, or irregular rooms, and whether it can properly merge multiple entrances of the same room or handle mixed room types when identifying nodes.
- The region potential mainly comes from the VLM score of RP plus a distance term, with no explicit semantic support or updatable pixel-level evidence. Under viewpoint changes, occlusion, or similar-appearance distractors, the score reliability is uncertain. Although the authors use Progressing and Backtracking lists, there seems to be no record of “sufficiently searched but goal not found”, which may still cause repeated exploration of already visited regions.

### Questions
- After reaching a region, how do you define “sufficiently searched so we won’t return”?
- What is the rationale for the fixed priority order of the four triggers? In practice, what are the frequency histograms and average intervals of each trigger?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a dynamic framework based on semantic regions, called DP-Nav to enable semantic navigation in embodied agents. This is in contrast to existing frontier based methods that ignore semantic region information. DP-Nav builds an additional region-junction graph to keep track of semantic regions and the corresponding frontiers. Through experiments the authors show that their method outperforms SOTA methods.

### Strengths
The idea to keep track of semantic regions seems to be useful to efficiently explore the environment. The paper is well written and easy to understand.

### Weaknesses
I find the work presented in this paper to provide useful insights for the community. I didn’t find any significant weaknesses.

### Questions
1. Do you think the region potential can be extended to incorporate textual semantics (e.g., “bathroom” vs. “corridor”)?

Minor comments:
The formatting of some table captions needs to be revised. For example, it’s hard to distinguish the wrapped text from the caption of Table 4. Same for Figure 3.

### Soundness
3

### Presentation
3

### Contribution
3
