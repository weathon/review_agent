## Summary
This paper presents *Puppeteer*, a two-stage hierarchical world-model approach for visual whole-body humanoid control. A single low-level TD-MPC2 tracking model is pretrained on retargeted MoCap motions and then frozen, while a high-level visual TD-MPC2 model learns downstream tasks by outputting end-effector reference commands. Across 8 simulated 56-DoF humanoid tasks, the method achieves task returns roughly comparable to a strong flat TD-MPC2 baseline while producing motions that are much more preferred by human evaluators.

## Strengths
- **Addresses a genuinely hard and important problem.** Visual whole-body control for a 56-DoF humanoid is a challenging setting, and the paper tackles it with high-dimensional actions, image observations, and unstable bipedal dynamics rather than relying on a simplified setup.
- **Clean, practically appealing architecture.** The decomposition into a reusable low-level tracker and a task-specific high-level puppeteer is simple and well motivated. The reuse of a *single* low-level model across tasks is a meaningful practical contribution relative to approaches that require many per-clip or per-skill controllers.
- **Strong evidence that the approach improves motion naturalness relative to a strong end-to-end baseline.** The user study is striking: Figure 6 reports an overwhelming preference for Puppeteer over TD-MPC2, and the qualitative examples in Figure 7 support that claim.
- **Competitive downstream performance despite the naturalness constraint/prior.** Figure 5 shows that Puppeteer is usually close to TD-MPC2 in return across most tasks, which is nontrivial given that the method constrains behavior through a pretrained motion-tracking interface.
- **Useful ablations.** The paper does more than a minimal benchmark comparison: Figure 8 analyzes offline+online pretraining, number of MoCap clips, planning at each level, and high-level pretraining; Figure 9 adds a zero-shot generalization result.
- **Interesting systems/method detail around termination-aware planning.** The added termination head and soft rollout truncation for planning in episodic MDPs is a useful practical extension of TD-MPC2 and appears relevant in this domain, where failure terminations matter.

## Weaknesses
###: Fatal
- None.

### Major:
- **The paper overstates its “without reward design” / “without simplifying assumptions” claim, and this is a real mismatch with the method.** The paper repeatedly claims control “without any reward design,” but Section 3.1 explicitly says: *“We label all transitions using the reward function from Hasenclever et al. (2020).”* Section 4.1 also defines task rewards for downstream tasks: visual tasks use forward-velocity reward and non-visual tasks reward displacement. Likewise, claims of “without any explicit domain knowledge” are too strong given the use of retargeted human MoCap, a hand-chosen end-effector command interface, and a handcrafted tracking reward. The contribution is still meaningful, but the strongest framing is inaccurate; the paper is better described as avoiding **additional naturalness/style reward engineering** rather than avoiding reward design altogether.
- **The broad sample-efficiency / superiority claim over prior humanoid-control pipelines is not fully supported by the presented evidence.** The Introduction and related-work framing claim “several orders of magnitude less interactions” than prior work, especially relative to MoCapAct-style pipelines, but Section 4.1 explicitly says the paper *refrains* from direct comparison to MoCapAct and DeepMimic. Given the differences in tasks, observation modalities, embodiments, and training pipelines, the paper does not establish a controlled apples-to-apples interaction-efficiency advantage. This should be narrowed to a more local claim about the presented benchmark and baselines.
- **The naturalness evidence, while compelling against TD-MPC2, is narrower than the headline claim suggests.** The user study only compares against one baseline, TD-MPC2, and the quantitative “naturalness” proxies in Table 1 are only reported on the *gaps* task and are fairly coarse. Since Puppeteer’s core mechanism includes MoCap-based low-level pretraining while TD-MPC2 is trained end-to-end from scratch, the current evidence most strongly supports: “a hierarchical MoCap-prior approach yields more natural motions than flat TD-MPC2.” That is valuable, but it is a narrower claim than “produces natural and human-like motions” in a broad comparative sense.
- **The paper does not sufficiently isolate which component is responsible for the naturalness gain.** The main causal ambiguity is whether naturalness comes from (i) hierarchy itself, (ii) the MoCap-based motion prior/frozen tracker, (iii) the end-effector command bottleneck, or some combination. The ablations in Figure 8 are useful, but none directly answer this question. This matters because the paper attributes the effect to the hierarchical world-model design, while the most plausible explanation is that the pretrained low-level motion prior is doing much of the work.

### Minor
- **Task diversity is limited, and the paper’s own limitation section effectively acknowledges this.** The 8 tasks are almost entirely locomotion/terrain traversal variants. This is still a meaningful benchmark, but it is weaker evidence for broad claims that the tracker is easily “sharable and generalizable across tasks.” Section 6 already notes that the tasks “primarily evaluate the visio-locomotive capabilities” of the method.
- **The temporal abstraction aspect of the hierarchy is underdeveloped in the experiments.** Section 3.2 presents \(k\) as a key hyperparameter trading off strong motion priors and control granularity, but Section 4.1 sets \(k=1\), meaning both levels act at the same frequency. This does not invalidate the method, since the hierarchy here is still meaningful as an action/interface decomposition, but it leaves one advertised aspect of the design empirically untested.
- **The stairs result exposes an underexplored performance–naturalness tradeoff.** The paper honestly notes that TD-MPC2 achieves better return on *stairs* by learning to “roll” up the stairs. This is a good observation, but the paper could do more to analyze whether Puppeteer’s lower return here reflects a desirable constraint, a limitation of the command space, or a more general cost of the motion prior.
- **Naturalness metrics could be richer and broader.** Table 1 gives only episode length and torso height on one task. The user study is valuable, but stronger objective evaluation across tasks would better support the paper’s emphasis on naturalness.
- **The low-level tracker’s limitations are not characterized in much detail.** Since the whole framework relies on the tracker as a reusable interface, more analysis of tracking accuracy/failure modes and what kinds of commanded motions are or are not representable would strengthen the paper.

### Trivial
- None.

## Nice-to-Haves
- An ablation over \(k>1\) to test genuine temporal abstraction.
- Additional objective motion-quality metrics across several tasks, not just *gaps*.
- A more explicit analysis of command tracking accuracy and end-effector bottleneck effects.
- A deeper discussion of how the pretraining cost should be amortized when making efficiency claims.
- Broader downstream tasks beyond locomotion-heavy terrain traversal.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Lack of real-world / sim-to-real evaluation.** The paper is clearly a simulated RL/control paper built around DMControl/MuJoCo and a new simulated benchmark. Criticizing the absence of real-robot experiments is largely scope creep here rather than a core flaw.
- **Requests for additional external related-work comparisons.** Per instruction, I do not include missing-related-work critiques.
- **Pure formatting/style complaints.** Parser artifacts are explicitly not paper issues, and minor style/figure-text nitpicks are not substantive here.
- **Complaints about availability/release/verification of cited datasets, tools, or code.** Those are disallowed and not valid criticisms under the review rules.
- **Fairness complaints where asymmetry favors the baseline.** For example, comparing the authors’ method against strong flat RL baselines despite the authors using a pretrained low-level tracker is not inherently unfair to the authors; if anything, it sets a strong bar.

## Novel Insights
The strongest way to understand this paper is not as “hierarchical RL beats flat RL,” but as a demonstration that **decoupling task optimization from motion realization through a reusable pretrained tracking world model changes what the reward can exploit**. The empirical story is therefore less about raw return gains and more about *structural regularization of behavior*: by forcing the high-level policy to act through a low-dimensional end-effector interface executed by a MoCap-grounded tracker, the method restricts access to many unnatural high-reward strategies that flat RL readily discovers. This interpretation explains both the strong human preference results and the occasional return deficit on tasks like *stairs*.

## Suggestions
- **Narrow the central claims.** Replace “without reward design” with a precise statement such as “without naturalness-specific reward engineering or hand-designed skill primitives.” Likewise soften “without simplifying assumptions” and broad sample-efficiency claims.
- **Be explicit that the motion prior is central, not incidental.** The paper is strongest when framed as a hierarchical world-model method that leverages a pretrained MoCap-based tracker to induce human-like behavior.
- **Add one ablation isolating source of naturalness.** For example: compare frozen pretrained tracker vs. trainable tracker, or compare the same architecture with/without MoCap-pretrained low level.
- **Expand the analysis of the stairs tradeoff.** This is one of the most interesting parts of the paper and deserves fuller discussion.
- **Report broader objective motion metrics across more tasks.** This would make the naturalness claim less dependent on a single user study and one-task proxies.
- **Clarify efficiency accounting.** Distinguish downstream task sample efficiency from total pipeline cost, including the amortized cost/benefit of low-level pretraining.

In terms of **originality**, the paper is moderately original: the overall decomposition is intuitive, but the specific instantiation as two TD-MPC2 world models with a reusable low-level tracker is a meaningful contribution. In terms of **importance**, the problem is clearly important and timely. On **support for claims**, the main empirical claims about outperforming TD-MPC2 on naturalness are well supported, but several broader framing claims are overstated. On **experimental soundness**, the benchmark and ablations are reasonably strong, though the causal story behind the gains is not fully isolated. **Clarity** is generally good; the paper is well organized and easy to follow. For **community value**, the benchmark, code release, and practical architectural lesson are useful contributions.

## Score and Decision
**Calibration papers used:**
- **H-GAP** (`/home/wg25r/review_agent/human_reviews/LYG6tBlEX0.md`, scores 8/8/6, Accept spotlight): similar humanoid-control setting with motion priors and planning; accepted because it showed a strong, well-supported planning story. The current paper is comparable in ambition and practical relevance, but weaker in claim calibration because some headline claims are overstated and direct comparative support is thinner.
- **Universal Humanoid Motion Representations** (`/home/wg25r/review_agent/human_reviews/OrOd8PxOO2.md`, scores 8/8/8, Accept spotlight): strong motion-prior paper with broad downstream evidence and strong support for human-like motion claims. The current paper is below this anchor because its task diversity and naturalness evaluation breadth are narrower.
- **MPC²** (`/home/wg25r/review_agent/human_reviews/MWHIIWrWWu.md`, scores 5/6/8/6, Accept poster): a hierarchical high-DoF control paper with meaningful but more modest contributions and some underdeveloped comparisons. Puppeteer is stronger than this anchor because it offers clearer empirical value, a stronger benchmark contribution, and compelling human-preference evidence.
- **HuWo** (`/home/wg25r/review_agent/human_reviews/bhUIoQ61pA.md`, scores 6/5/6/3, Reject) and **HumanoidOlympics** (`/home/wg25r/review_agent/human_reviews/pblB72EmrM.md`, scores 6/5/3/5, Reject): useful lower anchors for papers with meaningful ideas but weaker support, narrower benchmarking, or overclaiming. The present submission is above these because it has a cleaner technical story and substantially more convincing empirical evidence.

Overall, this paper is **above the acceptance bar** for me. The main contribution is real and useful, and the naturalness results are compelling. However, it is **not** a spotlight-level paper in its current form because some of the strongest framing claims are not adequately supported and the evaluation does not yet fully disentangle why the method works.

**Score: 7.0 / 10 — Weak Accept**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>