# Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 4

## Abstract
Large language model agents operate in partially observable, long-horizon settings where obtaining supervision remains a major bottleneck. We address this by utilizing a source of supervision overlooked in existing post-training methods: unintended yet successful goals embedded within agent rollouts. Specifically, we introduce Hindsight Supervised Learning (HSL), where an auxiliary LLM reviews each completed trajectory and relabels it with all of the natural-language goals the agent actually achieved. HSL then pairs the trajectory with its relabeled goals and uses these pairs for additional fine-tuning. To mitigate suboptimality in the relabeled data, we propose two learning techniques for HSL, irrelevant-action masking and sample reweighting. Our experiments show that HSL is flexible and compatible with existing post-training pipelines. It improves both SFT and DPO, with larger gains on long-horizon tasks with more diverse goal spaces. Moreover, HSL is sample-efficient: on ALFWorld, it surpasses baselines trained on the full dataset while using only one quarter of the ground-truth demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a sample-efficient learning method that reuses agent trajectories via hindsight experience relabeling. The methods are benchmarked in embodied navigation tasks with good empirical performances. Theoretical performance bounds are provided for the proposed method as well.

### Strengths
The paper is easy to follow, and the introduced algorithms are simple to add on to existing models, making them widely usable. The baselines are chosen well and extensively, while the studies on sample efficiency, ablations, and analysis are also designed well. Empirical performances are promising for the proposed method. (especially section on ensuring the performance gain doesn't only come from using a more powerful LLM)

### Weaknesses
- None of the main results have any statistical significance or uncertain qualification. The tasks from ALFWorld and WebShop should ideally be reran multiple times for every algorithm in order to at least get some standard deviation in score or success rate.
- The novelty of this work is rather limited, as hindsight experience relabeling is a widely used technique across robot learning and reinforcement learning.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Hindsight Supervised Learning (HSL) for (text-only) LLM agents in closed-world environments (i.e., with a structured schema of available goals): in addition to training the policy on expert demonstrations, train on on-policy rollouts labeled on the fly by an auxiliary, highly capable LLM, which relabels rollouts with all goals actually achieved, as judged by the auxiliary LLM. Relabeled trajectories are kept in a FIFO queue that is continually updated, and enter training alongside pre-existing expert demonstrations. Theoretical results provide some motivation for their approach; experiments on ALFWorld and WebShop show meaningful gains on the former and smaller gains on the latter. The paper does not establish results for open-world goal spaces.

### Strengths
Practical, simple idea that composes known ingredients (hindsight relabeling + SFT) into a new, tidy, reusable pipeline for language agents.

Clear empirical signal on ALFWorld; sample-efficiency curves indicate strong wins at lower demo budgets. Notably, on ALFWorld Unseen the method reaches ≈92.5% with 800 demos vs ≈82.8% for DPO with >3,200, indicating real demo-efficiency.

Implementation details (masking + reweighting + on-policy refresh) are intuitive and ablations suggest each adds value.

Theory is tidy (imitation-style discrepancy bound) and aligns with the empirical recipe—even if mostly motivational.

### Weaknesses
The paper has potential to be a nice contribution but has serious methodological shortcomings in its current presentation:

- No variance / seeds (critical). Table results and curves lack seeds, error bars, or confidence intervals. Delta could be within variance. A single cherry-picked seed is an absolute non-starter to get the paper accepted. (Addressing this point alone would change my score from 2 to 4; see questions below).

- Code / reproducibility (critical). Prompts are provided, but there’s no released code or relabeled dataset for third-party checking (and no determinism details). Uploading assets as a ZIP to OpenReview (or to some CDN that is anonymized) for inspection is a totally reasonable academic practice (especially if there are manually labeled results on the paper) that the authors should be expected to satisfy.

- Cost/efficiency underreported (important). The paper states fine-tuning wall-clock but does not clarify whether this includes the compute cost for continual relabeling with a large model, nor report the latter separately. Without this, “data efficiency” is unclear relative to real cost.

- Label-quality evidence is thin (important). The relabeler is validated with a very small human spot-check of only 50 trajectories.

- Heavy dependency on a powerful annotator (important). Relabeling uses a ~70B model for a 1B agent. There’s no sensitivity to smaller relabelers; conclusions may depend on annotator strength.

- The framing of "almost unsupervised" could be misread (important). The method requires ground-truth demonstrations and mixes them in every training step. It does not replace demos; the current evidence shows materially improved use of demos, certainly not demo-free learning. A modification of language here could be in order. The paper also does not compare clearly the compute cost of their method vs that of their baseline (i.e. using expert demonstrations only); not discussing trade-offs between / scaling laws of compute and dataset size limits the reach of their conclusions.

- Ablation confounds in RELABELFAILURE (preferable). Because that variant relabels only failed trajectories, it gets less data as the base improves. This confounds “algorithm” with “data volume.”

- Theory–practice gap (preferable). The bound depends on coverage/optimality constants that are not measured nor bounded experimentally; masking/reweighting are argued heuristically to help them, but this link isn’t empirically probed.

These evidence gaps drive my Soundness=1 and overall score. My take is that this paper is a valuable practical contribution with clear empirical signal on the right kind of tasks, but acceptance should be contingent on addressing the issues above; most notably, statistical significance. These are fixable and would substantially strengthen the paper.

### Questions
I found this paper interesting. Below are some questions/requests that I think that, if addressed, would substantially improve the quality of the paper.

- Seeds/variance: How many seeds per result? Seems only one. Please report mean±std (≥5) for Table 1 and all curves.

- Transparency: Will you release code, relabeling prompts, and the relabeled data?

- Relabeler sensitivity & cost: How does performance change with 3B/8B/70B relabelers? Report relabeling tokens, wall-clock, and total GPU hours per setting.

- Label validation at scale: can you label a substantially larger set of trajectories (e.g. 200) and/or add automatic predicate checks (where supported) to strengthen the statistical significance, and report confusion matrices per goal type?

- Ablation deconfounding: Please re-run a variant of RELABELFAILURE that (i) relabels all trajectories (successes included) and (ii) fixes the relabeled-sample budget across variants to disentangle algorithmic effect from data volume.

- Theory diagnostics: Any proxy measurements over training for coverage (e.g., diversity/goal-type coverage of hindsight set) and hindsight-expert quality (environment predicate agreement) to support the claimed mechanisms?

thank you!

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce an approach Hindsight supervised learning which trains an LLM agent on relabeled data. A relabeler LLM is used to relabel the LLM agent's trajectories with the goals that the LLM actually achieved. The authors provide results showing that by doing so, the agent is able to outperform baseline approaches.

### Strengths
The paper is very easy to read and provides a thorough theoretical and experimental analysis. The results are strong and very promising.

### Weaknesses
The relabeler’s output space is manually constrained with environment-specific templates. It would be helpful to discuss how HSL would scale to domains without predefined goals or how to automatically infer such goal spaces.

The method assumes that any achieved goal is beneficial to learn from but in some situations unintended achievements may not align with user intent or task utility. It would be useful to discuss ways to filter or weight relabeled goals based on relevance to user-defined objectives.

It would be useful to discuss the distinction between successful but unintended goals and partial progress toward intended goals. The method seems to treat both similarly, but their learning value may differ.

### Questions
Could you please discuss the goal space and how this might be defined in environments that don't have clearly predefined goals

Could you discuss how incorrect labels might dealt with and how they impact downstream performance?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Hindsight Supervised Learning (HSL), a post-training method for LLM agents. It leverages the unintended but successful goals achieved during an agent’s rollouts by relabeling trajectories with goals that were actually accomplished. These relabeled examples are then used for fine-tuning, with two key techniques: irrelevant-action masking and demonstration reweighting. Experiments on ALFWorld and WebShop benchmarks show HSL improves both supervised fine-tuning and DPO, achieving higher success rates with fewer expert demonstrations.

### Strengths
- The paper provides somewhat strong empirical validation, showing consistent performance gains and higher sample efficiency across multiple benchmarks.
- There is a theoretical property offered although it is difficult to parse due to presentation issues. I still appreciate the attempt to include a theoretical result, which is rare to see in papers with large foundation models.

### Weaknesses
Below are my comments in the order of appearance in the paper.

1- In addition to hindsight relabeling, the paper should also discuss learning from play data in robotics. In that line of research, it is common to define goals in play data and label the trajectories with those goals.
2- Around line 177-178, I was confused by why the paper uses both a goal state and a language instruction. I understand that it is desirable to have the function $\delta$ because it will be useful for detecting the agent reaching other goals, but then, maybe goal state is all the paper needs? Why also have an instruction? Having both seems redundant from an RL perspective.
3- In line 195, I believe the statement $K \leq T$ implicitly assumes the goals are mutually exclusive so the agent cannot achieve multiple goals at the same time. Perhaps this should be clarified earlier, because in reality an agent can achieve both "picking up a fruit" and "picking up a banana" at the same time, for example (or "closing the drawer" and "putting the mug in the drawer", etc.)
4- "improve agent optimality" is a weird phrase. If something is optimal, it cannot be improved by definition.
5- There is a system prompt that describes the space of valid goals. So the paper needs a definition for each goal. Perhaps something like "this set of states achieves this goal". But if it is a mapping, there would be no need for an LLM. So I assume, what is needed is a natural language description of each goal. This should be part of the problem statement.
6- In the theoretical analysis section, there is this variable $h$ in Equation 1, but it is not defined anywhere. This makes it very difficult to follow the theory. I assume it is history. But then, how can $\tau_{t-1}$ for different values of $t$ can be equal to $h$? The former has varying length depending on $t$ whereas the latter is fixed. Does this mean the probability term inside the summation is nonzero only for a single $t$? If so, writing it as a summation is unnecessarily complex. Perhaps this is not the case, and there is something I miss about $h$. Since it is not defined, I do not know what it can be.
7- The conclusion section should discuss when (for what tasks) this method can and cannot be applied.

### Questions
Please see my comments and clarification questions in the Weaknesses section above.

### Soundness
3

### Presentation
2

### Contribution
3
