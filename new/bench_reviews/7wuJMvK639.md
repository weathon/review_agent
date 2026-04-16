Let me verify the key claims against the paper content provided in the prompt. I need to check:

1. The "no reward design" claim — the paper says "without any reward design or skill primitives" in the abstract. But the paper uses velocity-based rewards and Hasenclever-style tracking rewards. Let me check.

2. The user study details — does the paper specify methodology?

3. The sample efficiency claim about "orders of magnitude fewer interactions"

4. The baseline details for visual tasks

Let me re-read key sections of the paper carefully.

From the paper content provided:

**On "no reward design":** The abstract says "without any reward design or skill primitives." Section 4.1 says "All 5 visual control tasks use a reward function that is proportional to the linear forward velocity, while non-visual tasks reward displacement in any direction." And the tracking agent uses "reward function from Hasenclever et al. (2020)." So there IS reward design, just that the downstream tasks use simple velocity rewards. The claim "without any reward design" is overstated — the tracking reward is itself heavily designed, and the termination condition is a form of reward shaping.

**On the user study:** Figure 6 shows 97.8% preference for Puppeteer. The paper says "We conduct a user study (n = 51) in which humans are shown pairs of short (~10s) clips of policy rollouts from TD-MPC2 and our method, and are asked to provide their preference." No details on: how many clips per method, per task, randomization, blinding, or whether participants could identify which was which. Appendix F is referenced but not included.

**On sample efficiency:** The paper says "several orders of magnitude less interactions to learn new tasks compared to prior work" and claims ≤3M steps for downstream tasks vs MoCapAct's ~150M steps. But the 10M pretraining steps for tracking are not counted.

**On baselines:** The paper says "We only evaluate SAC and DreamerV3 on proprioceptive tasks as they do not achieve any meaningful performance" — this means there's no visual baseline comparison for SAC and DreamerV3 on visual tasks, though there are hierarchical variants.

**On the stairs task:** Puppeteer gets lower reward than TD-MPC2, but the paper attributes this to "reward hacking" by TD-MPC2 (rolling up stairs).

Now let me write the consolidated review.

---

## Summary

Puppeteer is a hierarchical world model for visual whole-body control of a 56-DoF simulated humanoid. A low-level tracking agent is pretrained on MoCap data via TD-MPC2 to follow end-effector reference commands, then frozen and reused across tasks, while a high-level puppeteer agent learns visual tasks by generating these commands via planning. The method produces policies competitive in reward with TD-MPC2 across 8 tasks, while a user study (n=51) shows humans overwhelmingly prefer Puppeteer's motions.

## Strengths

- **Clean and effective hierarchical decomposition.** Using the same TD-MPC2 backbone for both levels and end-effector positions as the command interface is conceptually simple, physically grounded, and enables sharing a single tracking model across all tasks. The design is intuitive: the tracking agent learns physics, the puppeteer learns visuomotor coordination. This simplicity is a genuine strength over prior work that trains thousands of per-clip policies (MoCapAct).

- **Strong empirical results on a challenging domain.** Achieving non-trivial performance on a 56-DoF humanoid with visual observations is genuinely hard — SAC and DreamerV3 fail entirely on these tasks. The method works across 8 diverse locomotion tasks with a single tracking model, demonstrating practical reusability. The zero-shot generalization to 3× gap lengths (Figure 9) is a particularly compelling result.

- **Meaningful naturalness evaluation.** The paper takes naturalness seriously — the user study and quantitative proxies (torso height, episode length) represent a genuine effort to evaluate an important but often-ignored dimension of humanoid control. The 97.8% preference is striking even with caveats.

- **Comprehensive ablations.** Figure 8 carefully dissects the importance of the offline+online data mixture, number of MoCap clips, and planning at both hierarchy levels. The finding that planning at both levels is critical (vs. a model-free high-level policy) is informative.

- **Contribution to the community.** The 8-task benchmark addresses a real gap — no existing benchmark combines 56-DoF humanoids with visual observations, and the authors release code and environments.

## Weaknesses

### Major:

- **The naturalness evaluation is compelling in direction but has significant methodological gaps.** The user study (n=51, 97.8% preference) compares Puppeteer only against TD-MPC2, which has no motion prior whatsoever. This is a low bar for demonstrating naturalness — a fair comparison would also include methods with MoCap-based priors (e.g., AMP, DeepMimic, or motion-prior methods), at least on the proprioceptive tasks they support. The paper claims "unprecedented" naturalness evaluation, but showing that a MoCap-informed method produces more human-like motion than a method with no motion prior is largely expected. Additionally, the study methodology is underspecified: no details on clip selection, randomization, blinding, or inter-rater variance. The paper states participants are "undergraduate and graduate students across multiple universities and disciplines" (Appendix F reference), but does not indicate whether participants were blinded to which method was which or whether cherry-picked clips were shown. The quantitative naturalness proxies in Table 1 are only reported for a single task, and torso height/episode length are weak proxies for human-likeness (a stiff robot can have high torso height). These concerns do not invalidate the finding—Puppeteer clearly produces more natural motions—but they substantially weaken the strength of the "broadly preferred" and "unprecedented" claims.

- **The "no reward design" claim is overstated.** The abstract states the approach works "without any reward design or skill primitives," but in reality: (1) the visual tasks use carefully chosen velocity-proportional rewards; (2) the tracking agent uses the Hasenclever et al. (2020) tracking reward function; (3) non-foot contact termination is a strong implicit reward signal. The paper itself acknowledges that "TD-MPC2 baseline degenerates to highly unrealistic behavior without a contact-based termination condition" (Section 4.1), which concedes that MDP design choices matter significantly. The correct claim is that the method avoids *task-specific* reward shaping and skill primitives, not that it avoids reward design entirely.

- **The sample efficiency comparison to MoCapAct is incomplete.** The paper claims "several orders of magnitude less interactions" (Introduction) and highlights ≤3M downstream steps vs. ~150M for MoCapAct. However, the tracking model pretraining requires 10M steps (12 GPU-days). If counting total environment interactions to produce a controller for a given downstream task, the gap is smaller. The reusability argument (amortize pretraining over many tasks) is valid but is not formally made — the paper does not present a per-task total cost analysis. The "orders of magnitude" language is misleading without this accounting.

- **Task suite is locomotion-centric, limiting claims about "whole-body" control.** All 8 tasks are variants of walking, running, or jumping over/navigating terrain. The paper acknowledges this limitation, but the framing of "whole-body control" implies a broader scope. Tasks like manipulation, reaching, or object interaction would better test the generality of the motion prior and the command interface. On the current tasks, the end-effector command space (head, hands, feet) is naturally suited to locomotion, leaving its expressiveness for other tasks untested.

### Minor:

- **Performance-naturalness trade-off is inadequately analyzed.** On *stairs*, TD-MPC2 achieves higher reward by learning to roll, which the paper dismisses as "reward hacking." But this reveals a real tension: the MoCap prior constrains the policy space, potentially preventing creative but task-efficient solutions. The paper should explicitly characterize this trade-off rather than dismissing one side as undesirable. A small analysis showing what happens when Puppeteer's reward is augmented with a naturalness bonus, or when TD-MPC2 is given the same termination conditions but allowed longer training, would clarify whether Puppeteer is truly matching performance or just making a different trade-off.

- **Visual baselines are sparse.** SAC and DreamerV3 are excluded from visual tasks because they "do not achieve any meaningful performance." While understandable, reporting at least their raw numerical failures (or early learning curves) would help future researchers understand the difficulty landscape of this benchmark.

- **No ablation of the command representation.** End-effector positions as commands is a key design choice, yet no comparison to alternative command spaces (latent embeddings, joint angles, etc.) is provided. The ablation in Figure 8 focuses on planning and data, not on this central interface design.

- **No tracking quality analysis.** The tracking agent is foundational, but there is no quantitative evaluation of how accurately it tracks MoCap references. Since downstream performance is bounded by tracking quality, this gap makes it hard to assess whether the approach has headroom for improvement.

### Trivial:

- **Inference-time computational cost is not discussed.** The hierarchical approach involves planning at both levels with MPPI, which could be expensive. This matters for any practical deployment but is not a core flaw of the research.

## Nice-to-Haves

- Comparison against motion-prior baselines (e.g., AMP-style adversarial reward with TD-MPC2) to isolate whether the naturalness benefit comes from the hierarchical structure or from MoCap data priors specifically.
- Evaluation of tasks beyond locomotion (e.g., manipulation, reaching) to test the generality of the command interface.
- Analysis of OOD commands: how often does the puppeteer produce end-effector targets outside the tracking agent's training distribution, and how does tracking quality degrade?
- Varying k > 1 for temporal abstraction; the paper uses k=1 throughout.
- Reporting naturalness proxies across all 8 tasks, not just one.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Missing comparison with MoCapAct/DeepMimic on proprioceptive tasks*: The paper provides a clear justification for excluding these methods — they don't support visual observations and require orders of magnitude more interactions. A proprioceptive-only comparison would have been informative but is not required; the paper's focus is on visual control, and excluding methods that require privileged state access is reasonable.
- *User study participants are a convenience sample of ML-adjacent students*: This is standard practice for user studies in this research area and does not invalidate the results. The concern about blinding is valid and kept above, but the sample composition itself is not a weakness.
- *Formatting and presentation concerns*: Removed as formatting nitpicks per instructions.
- *Reproducibility concerns about undisclosed hyperparameters*: Removed as minor reproducibility nitpicks; the paper provides appendix references for implementation details and releases code.
- *No sim-to-real evaluation*: This is clearly out of scope for a method paper focused on simulated control. The paper does not make claims about physical robots.
- *Concerns about the benchmark not being "standardized" enough*: As a newly proposed benchmark, this is expected. The paper releases code and environments, which is sufficient for community adoption.

## Novel Insights

The most interesting insight from this work is the empirical finding that planning at both levels of the hierarchy is critical (Figure 8, "Planning low+high" vs. alternatives), and that a MoCap-informed motion prior can simultaneously stabilize learning and regularize behavior toward naturalness in a 56-DoF visual control problem. The tension between reward maximization and naturalness — vividly illustrated by the *stairs* task where TD-MPC2 "rolls" to higher reward — highlights a fundamental challenge for RL in humanoid control that the community should address explicitly rather than treating naturalness as an implicit byproduct of motion priors.

## Suggestions

- Tone down the "no reward design" claim to "no task-specific reward engineering" — this is more accurate and still strong.
- Add at least one proprioceptive baseline that also uses MoCap data (e.g., a TD-MPC2 agent with AMP-style adversary) to isolate the contribution of the hierarchical structure vs. the MoCap prior for naturalness.
- Report naturalness proxies (torso height, episode length) across all tasks, not just *gaps*, and correlate them with human preference data if possible.
- Provide user study details in the main text: number of comparisons, whether participants were blinded, how clips were selected.
- Count total environment interactions (pretraining + downstream) when making sample efficiency claims against MoCapAct.

## Score and Decision

**Calibration comparisons:**
- **TD-MPC2** (8,8,8,8 = spotlight): Same backbone algorithm, but that paper had much broader evaluation (104 tasks), very clean contribution delineation, and is a method paper with less overclaiming.
- **Universal Humanoid Motion Representations** (8,8,8 = spotlight): Similar domain, similar concerns about locomotion-only tasks and missing baselines, but stronger novelty in the representation learning approach and broader evaluation.
- **H-GAP** (8,8,6 = spotlight): Similar hierarchical MoCap-based humanoid control, similar concerns about task diversity, but a fundamentally different offline approach (no online RL needed).
- **MPC²** (5,6,8,6 = poster): Hierarchical model-based control for high-DoF systems, weaker novelty than Puppeteer but also less overclaiming.

Puppeteer is a solid contribution but with real overclaiming issues. The method works well, the hierarchical decomposition is clean, and the naturalness evaluation is a genuine step forward for the field. However, the naturalness claims are stronger than the evidence warrants (comparison only against baselines without motion priors, underspecified user study), the "no reward design" claim is inaccurate, and the task suite is narrower than "whole-body control" implies. The paper is above the acceptance threshold but would be significantly stronger with more tempered claims and a more rigorous naturalness evaluation.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>