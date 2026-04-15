## Summary

Generative Value Learning (GVL) reframes value estimation as a temporal ordering problem: a frozen VLM (Gemini-1.5-Pro) autoregressively predicts task completion percentages for *shuffled* video frames, then reconstructs temporal progress. By breaking the VLM's implicit chronological bias, GVL generates dense, semantically grounded value estimates without any task-specific training. The paper evaluates on 50 OXE datasets and 250 bimanual ALOHA tasks, introduces the Value-Order Correlation (VOC) metric, and demonstrates downstream applications including dataset filtering, success detection, and advantage-weighted regression.

---

## Claims and Support

| Claim | Verdict | Notes |
|---|---|---|
| GVL is a universal value function estimator via shuffled-frame autoregression | **Partially supported** | VOC measures temporal order recovery, not value calibration per se; the paper equates these, which is not automatic |
| Shuffling is essential; chronological prompting yields degenerate monotonic predictions | **Supported** | Fig. 5 (right) and ablation in Sec 4.4 clearly demonstrate this; value curves collapse without shuffling |
| GVL zero-shot produces "effective values" for 300+ tasks | **Partially supported** | OXE results are strong; but ALOHA-250 zero-shot yields median VOC 0.12 and positive on only ~60% of tasks — paper's own text says "promising, but worse than… OXE" |
| Multi-modal in-context learning, including cross-embodiment transfer | **Supported with scope caveat** | Cross-embodiment shown on 13 tasks (ALOHA-13); one-shot ALOHA-250 uses same-task demonstrations — closer to test-time adaptation than general ICL |
| VOC is a predictive evaluation metric for value quality | **Supported as proxy** | The paper explicitly calls it "lightweight, yet predictive"; Table 3 shows VOC correlates with downstream AWR outcomes, but correlation is imperfect (2/7 tasks contradict it) |
| GVL enables success detection and filtered imitation learning | **Supported** | Table 2 and Fig. 6 show GVL-SD consistently outperforms SuccessVQA on 6 simulated tasks |
| GVL values improve real-world policy learning via AWR | **Weakly supported** | 7 tasks, 10 trials, wins 5/7 but loses 2/7; no variance reporting; non-standard advantage formulation unexplained |

---

## Strengths

- **Genuinely clever and non-obvious core idea.** The shuffled-frame prompting trick is elegant: using a harder prediction task (unshuffling) to elicit more faithful semantic reasoning from a VLM is a real insight, not an obvious engineering decision. Fig. 5 (right) concretely validates why this is necessary.
- **Unusually broad real-world evaluation.** 50 OXE datasets + 250 ALOHA tasks vastly exceeds the scale of prior VLM reward/value model papers, which typically evaluate on a handful of simulated environments.
- **Compelling in-context learning results.** The scaling curve from 0→5 in-context examples (Fig. 3 right) is strong, and the cross-embodiment result (human videos improving robot value prediction) is a distinctive capability enabled by long-context VLMs.
- **Practical, immediately deployable applications.** Dataset quality estimation using VOC (Table 1) and success-filtered imitation learning (Fig. 6) provide direct, actionable tools for the robotics community dealing with mixed-quality datasets.
- **Strong ablations validating design choices.** The single-frame ablation (Table 4, VOC = −0.08 vs 0.74) and the no-shuffling ablation (Fig. 5 right) are clean and convincing validations of the two key design decisions.

---

## Weaknesses

### Fatal
*None identified.*

### Major

- **VOC conflation with value quality overstates the evidence.** VOC measures rank correlation with expert timestep order on demonstrations. A model can score highly by producing any monotone ordering — it need not produce semantically calibrated values. The paper motivates GVL with Bellman consistency and "universal value estimation," but no experiment tests Bellman-style consistency, value calibration against annotated progress, or performance on non-monotonic trajectories. The paper itself hedges appropriately in one place ("lightweight, yet predictive") but then promotes VOC as validating "value prediction quality" throughout Section 4. The contribution may be better characterized as a *strong temporal ordering heuristic* rather than a validated universal value function. This distinction matters for downstream AWR, where calibration of value magnitudes, not just ordinal ranking, determines advantage weights.

- **Abstract overclaims uniformly "effective values" across 300+ tasks.** The abstract states GVL can "predict effective values for more than 300 distinct real-world tasks." However, the paper's own ALOHA-250 zero-shot results report a median VOC of 0.12 with positive correlation on only ~60% of trajectories — a weak signal acknowledged in Section 4.1 as "promising, but worse than the performance on the OXE datasets." Framing this as "effective values" without qualification is misleading. A more accurate narrative: strong zero-shot performance on OXE, weak zero-shot on long-horizon bimanual tasks, substantially improved with same-task in-context examples.

- **Image-goal evaluation confound in OXE.** For OXE datasets lacking text annotations, the paper uses "the last frame of the trajectory as the goal specification" — i.e., the terminal frame of the *same trajectory being evaluated*. This materially changes the task from general semantic progress estimation to distance-to-terminal-frame ranking, which is an easier and different problem. Since a substantial fraction of OXE lacks language goals (Fig. 2 right), the aggregated VOC histograms conflate two qualitatively different settings, and the "zero-shot" framing for the image-goal datasets is misleading. The paper does not separate these results or flag the confound prominently.

- **AWR downstream evidence is thin and the formulation is unexplained.** Table 3 reports 7 tasks × 10 trials with no variance, confidence intervals, or multiple seeds. Two of seven tasks (open-drawer, remove-gears) show GVL *hurting* performance. More importantly, Eq. 6 weights transitions by exp(τ(v_{k+1} − v_k)), which is the exponential of a value *difference*, not a standard advantage. Standard AWR uses A(s,a) = Q(s,a) − V(s); the paper's formulation is closer to potential-based reward shaping. This choice is never justified, and no comparison to simpler weighting strategies (e.g., uniform, timestep-based, terminal-proximity) is provided. Without such controls, the modest gains cannot be attributed specifically to value quality.

### Minor

- **One-shot ALOHA results reflect same-task test-time adaptation, not broad ICL generalization.** The largest gains on ALOHA-250 come from one-shot examples collected from the *same task*. This is meaningful but is closer to few-shot adaptation with matched supervision than the "multi-modal in-context learning" framing suggests. Cross-task ICL is mentioned only in the appendix.

- **Cross-embodiment results validated only at limited scale.** Fig. 4 demonstrates cross-embodiment transfer on ALOHA-13 only. Scaling this result to ALOHA-250 or OXE would substantially strengthen the cross-embodiment claim.

- **Success detection baselines are narrow.** GVL-SD is compared only against SuccessVQA and SuccessVQA-CoT, both VLM-based. In a simulation setting, relevant baselines include state-based success detectors, final-frame classifiers, or return-proxy methods. Demonstrating that the *shuffling mechanism* (not just VLM usage) is the key factor is important but not isolated here.

### Trivial

- The no-shuffling ablation in Section 4.4 is demonstrated through the success-detection setting (Fig. 5) rather than directly on the headline OXE or ALOHA value-prediction benchmarks. A matched quantitative comparison on those benchmarks would make the ablation more complete.

---

## Nice-to-Haves

- **Value calibration analysis.** Plotting raw predicted value curves vs. timestep for representative trajectories (linear? sigmoidal? bursty?) would reveal whether GVL produces magnitudes suitable for AWR or just ordinal rankings.
- **Computational cost discussion.** Running Gemini-1.5-Pro autoregressively over 150+ shuffled images per prediction is non-trivial. Characterizing latency and cost per trajectory would clarify the practical deployment scope.
- **Anchor frame sensitivity ablation.** The first frame is fixed as anchor. Ablating alternate anchors (goal image, random frame, no anchor) would strengthen the design justification.
- **More VLM generalization evidence in main text.** Appendix ablations on other VLMs are relevant to the universality claim; a brief main-paper summary would strengthen it.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Neutral reviewer: "Dependence on Proprietary Model Capabilities" as a reproducibility weakness.** The paper explicitly uses Gemini-1.5-Pro and ablates other VLMs in the appendix. Per hard rules, if the paper cites it, it exists; availability/reproducibility concerns rooted in the model being closed-source are reviewer knowledge-gap issues, not author errors. The method can in principle be replicated with any long-context VLM, which the appendix ablations validate.

- **Harsh critic: AWR comparison to trained value functions as a required experiment.** The paper's contribution is a training-free approach; demanding comparison to trained value functions goes partially outside its stated scope (though it would be informative as a nice-to-have for the stronger RL framing).

- **General criticism about "Bellman consistency not formally established."** The paper explicitly states this as an *analogy* and *insight* ("For example, a VLM is unlikely to estimate that a task is 50% completed if it already has a 50% completion prediction in context"), not a formal claim. Criticizing the absence of formal Bellman consistency proofs for an empirical systems paper is scope creep.

---

## Novel Insights

The paper's most genuinely novel observation is that VLMs harbor a strong temporal shortcut bias when shown video in chronological order — producing degenerate monotonic predictions — and that this bias can be neutralized by reframing value prediction as an *ordering/unshuffling* task. This is not just an engineering trick: it reveals something about how large multimodal models encode temporal structure and how that encoding can be exploited constructively. The practical implication — that a frozen, general-purpose VLM can serve as a reasonable zero-shot value estimator for diverse robotics tasks without any task-specific training, and can be improved at test time with cross-embodiment examples — is a meaningful capability that prior work had not demonstrated at this scale.

---

## Suggestions

1. **Narrow abstract/intro claims:** Replace "effective values for more than 300 tasks" with "high-quality value estimates on the majority of OXE datasets and, with in-context examples, on 250 challenging bimanual tasks." Acknowledge the ALOHA zero-shot weakness upfront rather than only in Section 4.1.

2. **Separate text-goal and image-goal evaluations more prominently.** The image-goal setting (last-frame as goal) is methodologically different from text-goal zero-shot estimation. Report these separately in Table 1 and discuss the limitation.

3. **Justify or replace the AWR advantage surrogate (Eq. 6).** Either provide a formal motivation for using v_{k+1} − v_k as an advantage proxy, or compare it against alternative weighting schemes (uniform, timestep-based, return-weighting) on even a subset of tasks.

4. **Add variance/seed reporting to Table 3.** Even reporting standard error across 10 trials would substantially improve the credibility of the AWR results.

5. **Include VOC-calibrated vs. raw-values analysis for AWR.** Showing how VOC score predicts the *magnitude* of advantage weights in practice would bridge the gap between the proxy metric and the downstream application.

---

## Score and Decision

**Originality:** High. The shuffled-frame value estimation framing is non-obvious and genuinely distinct from prior VLM reward model work.

**Importance:** High. Value/reward specification is a genuine bottleneck in robot learning, and a training-free, broadly applicable approach has clear community value.

**Claim support:** Mixed. OXE results are compelling; ALOHA zero-shot is weak; AWR is preliminary and uses a non-standard formulation; the headline metric has noted limitations.

**Experimental soundness:** Moderate. Scale is impressive but several key evaluations are thin (7-task AWR with no variance, narrow success detection baselines, unacknowledged image-goal confound).

**Writing clarity:** Good overall, but abstract/introduction overstate the evidence on the hardest benchmarks.

**Value to the community:** High — both the method and the VOC metric are immediately useful for dataset curation at scale.

The paper presents a genuinely clever and novel contribution evaluated at impressive scale, but the framing exceeds what the evidence strictly supports, the headline metric has unacknowledged limitations for the strongest claims, and the downstream policy-learning evidence is too preliminary to anchor the paper's RL framing. These are correctable issues that do not undermine the core contribution; the paper merits acceptance with revision.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>