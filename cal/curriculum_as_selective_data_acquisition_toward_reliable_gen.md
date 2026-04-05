=== CALIBRATION EXAMPLE 5 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title reflects the paper’s main framing well: curriculum as selective data acquisition in goal-conditioned RL. It is accurate, though somewhat broader than the empirical evidence, which is limited to a small GridWorld with UVFAs.
- The abstract clearly states the problem, method, and headline findings: uniform vs. curriculum goal sampling, UVFAs with PBRS, and improved goal coverage / approximation error / edge-goal success.
- However, some claims are stronger than the evidence in the paper. In particular, “reduce approximation error” is asserted in the abstract, but the paper does not present a clearly quantified approximation-error analysis beyond qualitative statements. Likewise, “toward reliable generalization” and “persistent and open-ended agents” are aspirational relative to the narrow experimental setting.

### Introduction & Motivation
- The motivation is reasonable and well aligned with ICLR interests: curriculum learning, goal-conditioned RL, and distribution shift in training data are all relevant. The link to open-ended learning is timely.
- The gap is framed as “curricula as data acquisition” rather than exploration, which is a defensible perspective. That said, the introduction somewhat overstates novelty: prior curriculum and goal-generation work already implicitly changes the training distribution, so the paper needs to more sharply distinguish its conceptual reframing from existing teacher/student or goal-generation literature.
- Contributions are stated, but they remain somewhat vague. The paper promises analysis of distributional shifts, approximation error, and policy success, yet the concrete novelty over prior work is mostly a simple edge-biased sampling heuristic in GridWorld.
- The introduction slightly over-sells the generality of the conclusion. The evidence supports that a biased goal sampler can help on harder goals in this toy domain; it does not yet establish a “principled mechanism” for persistence or open-ended learning.

### Method / Approach
- The method is described at a high level, but reproducibility is not strong enough for ICLR standards. Key details are missing or underspecified: the GridWorld size/layout, action space, exact termination conditions, how “valid grid cells” are defined, and how the held-out evaluation set is constructed.
- The UVFA setup is basic and understandable, but the paper is conceptually muddled in places. The method says the UVFA outputs \(V(s,g)\) and is trained by mean-squared regression to “pseudo-reward targets,” yet the evaluation says greedy action selection is based on negated returns. It is not fully clear whether the model learns state values, action values, or return-to-go estimates, and how those are used to choose actions at test time.
- The PBRS formulation appears intended to preserve optimality, but the reward equation and target construction are not explained carefully enough to verify correctness. In particular, the paper should clarify whether shaping is applied during data collection, training targets, or both, and how this interacts with discounted returns-to-go.
- The curriculum design is very simple: uniform vs edge-weighted goal sampling. That is acceptable if the paper’s claim is modest, but then the paper should be careful not to generalize too far from this single heuristic. The “weighted curriculum” is mentioned, but its exact sampling distribution is not specified precisely enough to reproduce.
- Failure modes are only lightly acknowledged. For example, edge-biased sampling could simply overfit peripheral goals while hurting central coverage; the method section does not discuss such tradeoffs or how the fixed-size dataset interacts with distributional reweighting.

### Experiments & Results
- The experiments do test the central claim in a limited sense: does curriculum-induced goal reweighting change coverage and success? Yes, at least in GridWorld, with separate reporting for edge vs. interior goals.
- The baselines are minimal but appropriate for a first-pass study: uniform sampling versus curriculum-biased sampling. However, for an ICLR-level paper, the baseline set is too weak to support broader claims. There is no comparison to other curriculum strategies, hindsight-style relabeling, prioritized replay, or adaptive goal samplers that would more convincingly situate the method.
- The biggest issue is statistical weakness. Results are averaged over only three seeds, with no significance tests, confidence intervals beyond standard deviation bars, or formal effect-size analysis. This is a concern because the reported gains are modest and sometimes inconsistent.
- The paper claims improved “approximation error,” but the experiments do not convincingly quantify this. The closest evidence is success rates and a qualitative statement about distributional shifts. If the paper’s conceptual contribution is about data distribution shaping function approximation, it needs direct error metrics over the state–goal space, not just policy success.
- The evaluation horizon sweep is useful, but the reported conclusions are mostly concentrated on \(H=16\). It would be helpful to show whether the curriculum consistently helps across all horizons or only in one setting.
- There is some risk of cherry-picking in the presentation: the text emphasizes edge-goal gains and downplays the cases where performance is comparable or potentially worse on easier goals. Since the core claim is about targeted redistribution of training data, the tradeoff should be shown more systematically.
- The paper does not report learning curves, sample efficiency across training, or robustness to different curriculum strengths. Those would materially affect the conclusions, especially for a claim about selective data acquisition.

### Writing & Clarity
- The main conceptual thread is understandable: curricula change the data distribution, and that can improve approximation where the agent is weak. This is a good and coherent framing.
- That said, several sections remain too ambiguous for a technical paper. The method description leaves open important implementation details, and the relationship between PBRS, pseudo-reward targets, and evaluation is not fully transparent.
- Figures and tables are only partially informative in the extracted text. Table 1 summarizes one horizon and one set of outcomes, but it is too sparse to support the paper’s broader conclusions. The paper would benefit from a clearer presentation of the training distribution shift itself and a more direct visualization of approximation error over the state–goal space.
- The discussion sometimes repeats the central claim without adding new evidence. It would be stronger if it connected specific empirical observations to specific mechanistic interpretations.

### Limitations & Broader Impact
- The paper does acknowledge some limitations: small GridWorld, hand-designed curricula, modest gains, and inconsistent results across seeds. That is good and important.
- However, the limitations are still underdeveloped relative to the claims. The biggest missing limitation is that the evidence does not establish the “selective data acquisition” framing as more than a plausible interpretation of goal reweighting in a toy domain. There is no analysis of whether this remains beneficial when function approximation is harder, when the goal space is continuous, or when exploration itself is the bottleneck rather than data imbalance.
- The broader impact discussion is minimal. Given the paper’s open-ended learning framing, it should be more careful about overstating relevance to “persistent” or “generally capable” agents without evidence.
- Negative societal impact is not a major concern here, but the paper does miss an opportunity to discuss the limits of curriculum shaping in safety-critical settings, where biased data acquisition could entrench blind spots if poorly designed.

### Overall Assessment
This paper has a clear and timely idea: curriculum learning can be interpreted as selective data acquisition that reshapes the training distribution in goal-conditioned RL. That framing is interesting and plausibly useful, and the GridWorld experiments provide some supportive evidence that edge-biased sampling improves success on harder goals. However, under ICLR standards the empirical and methodological support is too thin to sustain the broader claims. The experiments are small, the baselines are limited, statistical evidence is weak, and the key mechanism—improved approximation via distribution shift—is not directly demonstrated with enough rigor. I think the contribution is promising as an exploratory position/empirical note, but not yet at the level of a strong ICLR paper unless the authors substantially tighten the methodology and add more convincing evidence.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies curriculum learning in goal-conditioned reinforcement learning (GCRL) through a data-distribution lens, arguing that curricula act as selective data acquisition by shifting which state-goal pairs are emphasized during training. Using a small deterministic GridWorld with UVFAs and potential-based reward shaping, the authors compare uniform goal sampling to edge-biased curricula and report modest improvements in edge-goal success and approximation quality. The paper’s central claim is conceptual rather than algorithmic: curricula should be understood as a mechanism for reshaping training data distributions, with implications for reliable generalization and open-ended learning.

### Strengths
1. **Clear and timely conceptual framing.** The paper connects curriculum learning to selective data acquisition, which is a useful perspective for GCRL and aligns with ICLR interest in understanding what training distributions do to generalization.
2. **Simple, controlled experimental setup.** The use of deterministic GridWorld, UVFAs, and fixed data collection helps isolate the effect of curriculum-induced distribution shift rather than conflating it with environment stochasticity or architecture changes.
3. **Some evidence of targeted benefit.** The reported results indicate improvements specifically on harder edge goals, e.g., at \(H=16\) edge success rises from \(0.060 \pm 0.055\) to \(0.143 \pm 0.107\), which is consistent with the curriculum hypothesis.
4. **Attempts to assess representation/approximation, not only return.** The paper claims curricula reduce approximation error and alter training distribution, which is a stronger scientific angle than merely reporting performance gains.
5. **Acknowledges limitations.** The discussion and limitations sections are reasonably candid that the gains are modest, the curriculum is hand-designed, and the experiments are restricted to small GridWorlds.

### Weaknesses
1. **Novelty is limited relative to the curriculum-learning literature.** The core intervention is an edge-biased sampling heuristic, which is closely related to prior reverse curricula, automatic goal generation, and teacher-student curricula. The paper’s main novelty is largely a re-interpretation rather than a substantively new algorithm or theory, which is likely below ICLR’s bar unless supported by stronger evidence.
2. **Empirical evidence is too weak for ICLR standards.** Results are based on only three seeds and a toy environment, with modest gains and high variance. For example, overall success improves from \(0.276\) to \(0.297\), which is small in absolute terms, and edge performance has large standard deviations.
3. **No strong ablation or baseline coverage.** The paper compares mainly uniform sampling vs. two hand-crafted curricula. It does not benchmark against standard GCRL baselines or established curriculum methods, nor does it isolate whether gains come from sampling frequency, trajectory quality, or reward-shaping interactions.
4. **The “selective data acquisition” claim is under-supported.** The paper asserts a distributional mechanism, but provides limited quantitative evidence beyond qualitative plots and success rates. It does not rigorously analyze coverage, effective sample reweighting, learning dynamics, or approximation error across the full state-goal space.
5. **Methodological details are incomplete.** Key implementation details needed for reproducibility are missing or unclear, such as the exact grid size, valid goal set, data collection policy behavior under shaping, how “held-out goals” are constructed, and what precise metric is used for “approximation error.”
6. **Potential conceptual inconsistency in the framing.** The paper motivates persistent/open-ended learning, but the experiments are in a fixed finite GridWorld with manual edge weighting. The leap from this toy setting to OEL is mostly rhetorical and not substantiated empirically.
7. **Clarity and presentation are uneven.** Some sections repeat claims, and parts of the discussion overstate the empirical support relative to the results shown. The paper would benefit from a tighter distinction between evidence and interpretation.

### Novelty & Significance
**Novelty:** Moderate-to-low. The re-framing of curriculum learning as selective data acquisition is interesting, but the concrete method is a simple heuristic and the experiments are not sufficiently extensive to establish a strong new scientific result.  
**Clarity:** Fair. The high-level idea is understandable, but several experimental and evaluation details are underspecified, which limits interpretability.  
**Reproducibility:** Fair-to-poor. The setup is simple, but the paper lacks enough detail to confidently reproduce the experiments without additional clarification.  
**Significance:** Limited for ICLR in its current form. The topic is relevant, but the empirical scale, novelty, and rigor seem below the typical acceptance bar for ICLR, which generally expects either a compelling new method, strong theoretical insight, or broad and convincing empirical evidence.

### Suggestions for Improvement
1. **Add stronger baselines and comparisons.** Compare against standard GCRL curriculum methods such as reverse curriculum generation, self-paced/teacher-student approaches, HER-style baselines if applicable, and other goal-sampling heuristics.
2. **Expand the experimental scope.** Test on more than one grid size and at least one nontrivial continuous-control or sparse-reward goal-conditioned benchmark to show the effect is not GridWorld-specific.
3. **Quantify the distributional claim directly.** Measure goal visitation entropy, coverage, reweighting effects, effective sample size, and approximation error across the state-goal space to substantiate the “selective data acquisition” interpretation.
4. **Include more seeds and statistical testing.** Three seeds are too few for ICLR-level confidence. Report confidence intervals, significance tests, and effect sizes over a larger number of runs.
5. **Clarify evaluation protocol.** Specify how held-out goals are chosen, whether the policy is evaluated with the same shaping scheme, and how success and approximation error are computed.
6. **Add ablations on curriculum design.** Test alternative reweighting strengths, interior-biased curricula, random biasing, and versions without PBRS to disentangle the contribution of curriculum from reward shaping.
7. **Tighten the claims.** Soften claims about persistent/open-ended learning unless supported by experiments; present the paper primarily as evidence that curriculum changes data distribution and can improve targeted generalization in GCRL.
8. **Improve the theory/analysis link.** If the selective acquisition framing is central, provide a more formal analysis of why biased goal sampling should reduce approximation error or improve coverage under UVFA training.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add stronger baselines beyond uniform sampling, especially standard GCRL baselines like HER, reverse curriculum, automatic goal generation, and adversarial/teacher-student curricula. At ICLR, a claim that your curriculum is a meaningful mechanism for selective data acquisition is not convincing if it only beats uniform sampling in a toy setting.

2. Add ablations separating curriculum effects from reward shaping and greedy data collection. Right now it is unclear whether gains come from the curriculum, PBRS, or the training/evaluation protocol; without isolating these factors, the central causal claim is not believable.

3. Evaluate on more than one environment family, ideally at least one standard benchmark with nontrivial geometry or stochasticity beyond deterministic GridWorld. ICLR expects evidence that the claim generalizes; a single hand-designed gridworld is too narrow to support “principled mechanism” language.

4. Compare against random reweighting / matched-coverage baselines that control for the amount of edge-goal exposure. Otherwise the improvement could simply be due to seeing harder goals more often, not to curriculum as a selective acquisition strategy.

5. Report learning curves versus final performance, with confidence intervals over enough seeds. With only three seeds and small gains, the paper does not establish that the curriculum effect is stable enough to support the stated generalization claims.

### Deeper Analysis Needed (top 3-5 only)
1. Add a direct analysis of how the sampling distribution changes value approximation error across the full state-goal space. The paper claims curricula improve “selective data acquisition,” but it never quantifies where approximation error drops, where it rises, and whether these shifts explain success.

2. Analyze whether edge-goal gains come from better coverage, better optimization, or simply easier credit assignment under PBRS. Without this decomposition, the paper does not identify the mechanism behind the observed improvements.

3. Provide a difficulty-stratified breakdown beyond the coarse edge/interior split, such as success and value error versus goal distance, corner proximity, or reachability under shortest path length. The current analysis is too coarse to justify the “underachieved goals” interpretation.

4. Quantify statistical significance and effect sizes for the reported improvements. The paper repeatedly emphasizes modest gains, but it does not establish whether these are robust enough to matter versus run-to-run variance.

5. Analyze failure cases where curriculum hurts easy goals or certain seeds. ICLR reviewers will want to know when selective acquisition is harmful, because that directly tests whether the method is a reliable mechanism or just a brittle heuristic.

### Visualizations & Case Studies
1. Add heatmaps over the full grid showing goal sampling frequency, value prediction error, and success rate for both conditions. This would directly reveal whether the method truly reallocates learning toward underachieved regions and whether that reallocation improves those regions.

2. Include per-goal learning trajectories showing how success evolves over training for representative easy, medium, and hard goals. This would expose whether the curriculum accelerates learning or just redistributes performance at convergence.

3. Show side-by-side qualitative rollouts on goals that improve and goals that regress under curriculum. Without concrete examples, it is impossible to see whether the method genuinely changes behavior or only moves aggregate numbers slightly.

4. Visualize the curriculum weights or sampling probabilities over time. The paper claims selective acquisition, but there is no evidence that the curriculum adapts in a way that tracks difficulty rather than merely encoding a hand-crafted edge prior.

### Obvious Next Steps
1. Replace the hand-crafted edge heuristic with an adaptive teacher that selects goals based on online learning progress or uncertainty. That is the most direct next step if the paper wants to argue for curriculum as principled data acquisition rather than a toy manual heuristic.

2. Extend the setup to continuous-control or procedurally generated goal spaces. The core claim is about generalization and persistent learning, and GridWorld is too simple to show that the idea scales.

3. Test whether the same selective acquisition idea improves representation learning under shared encoders or larger function approximators. The current MLP/GridWorld setup is too small to establish whether the effect survives in realistic function approximation regimes.

4. Evaluate interaction with replay/relabeling methods used in modern GCRL. If the paper wants to position curriculum as a data-selection mechanism, it should show how it complements or competes with the dominant data-acquisition tools in the area.

# Final Consolidated Review
## Summary
This paper argues that curriculum learning in goal-conditioned RL should be viewed as a form of selective data acquisition: by biasing goal sampling toward underachieved regions, the training distribution shifts in ways that can improve generalization where the agent is weakest. The empirical evidence is limited to deterministic GridWorld with UVFAs and PBRS, where edge-biased curricula produce modest gains on harder edge goals and only small changes in overall success.

## Strengths
- The paper has a coherent and timely conceptual framing: curriculum as distribution shaping / selective data acquisition is a reasonable lens for goal-conditioned RL, and the authors connect it to persistent and open-ended learning in a way that is easy to understand.
- The experimental setup is intentionally controlled and simple, which does isolate the effect of goal-sampling bias to some extent. The reported numbers do show the expected pattern on hard edge goals, e.g. edge success improves from 0.060 to 0.143 at \(H=16\), which is consistent with the core hypothesis.

## Weaknesses
- The empirical evidence is very thin for the strength of the claims being made. The paper relies on a toy deterministic GridWorld, only three seeds, and modest gains with large variance; this is not enough to support the paper’s broader language about “reliable generalization” or “a pathway toward persistent and open-ended agents.”
- The central mechanistic claim — that curricula improve learning by reshaping the state-goal data distribution and thereby reducing approximation error — is not convincingly demonstrated. The paper states this repeatedly, but does not provide a rigorous approximation-error analysis across the state-goal space, nor a clean decomposition showing why the gains occur.
- The comparison set is too weak. Uniform sampling plus two hand-crafted curricula is not sufficient to establish that this is a meaningful or principled mechanism relative to standard GCRL curriculum methods, goal relabeling, or adaptive teachers.

## Nice-to-Haves
- Add stronger baselines from curriculum and goal-conditioned RL, especially reverse curriculum generation, teacher-student or adaptive goal-sampling methods, and HER-style baselines where appropriate.
- Include direct quantitative analyses of coverage and approximation error over the full state-goal space, rather than inferring mechanism from success rates alone.
- Report learning curves, confidence intervals, and statistical tests over more seeds to clarify whether the observed gains are stable or just run-to-run noise.
- Test at least one additional environment family beyond GridWorld to show the effect is not specific to this hand-designed setting.

## Novel Insights
The most interesting contribution is not an algorithm, but a reframing: curriculum learning is treated as a selective acquisition policy over goal-state pairs, which makes the paper’s argument about distribution shift more principled than a generic “curricula help exploration” story. That said, the experiments only partially validate this lens. The results suggest that biasing training toward difficult edge goals can improve performance in those regions, but the evidence stops short of showing that the curriculum truly induces a favorable redistribution of approximation capacity rather than simply exploiting a small-grid heuristic under PBRS.

## Potentially Missed Related Work
- Reverse Curriculum Generation — highly relevant because it also adapts goal sampling based on difficulty and is a direct comparator for the paper’s framing.
- Teacher-student curriculum learning — relevant as a more principled adaptive sampling paradigm for goal selection.
- Automatic goal generation / self-play curricula — relevant as established approaches to selective task acquisition in RL.
- Prioritized relabeling / replay methods in GCRL — relevant because they also manipulate the training-data distribution and could help situate the paper’s “data acquisition” interpretation.

## Suggestions
- Tighten the claims to match the evidence: present this primarily as a small empirical study supporting a useful lens on curriculum, not as evidence for persistent or open-ended learning.
- Add a direct measurement of where approximation error changes across the grid under uniform vs curriculum sampling, and show whether those changes explain the edge-goal gains.
- Disentangle curriculum effects from PBRS and the greedy data-collection protocol with ablations, so the causal story is clearer.
- If possible, replace the hand-crafted edge heuristic with an adaptive sampler that reacts to learning progress; that would make the “selective data acquisition” framing substantially more convincing.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0, 0.0, 0.0]
Average score: 0.0
Binary outcome: Reject
