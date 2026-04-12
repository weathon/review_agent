## Summary
This paper introduces **ToMBench-Hard**, a 900-example manually curated Theory-of-Mind benchmark spanning six ToM dimensions, and **Social-R1**, an RL framework that combines outcome rewards with a trajectory-level “social thinking” reward model inspired by Social Information Processing theory. Empirically, outcome-only RL on the hard benchmark already improves performance substantially, and the full method further improves results on several social reasoning benchmarks, though the evidence for the added value of the trajectory-level reward is mixed rather than cleanly established.

## Strengths
- **The paper identifies and operationalizes a concrete gap in current social-reasoning evaluation.** ToMBench-Hard is not just another aggregate benchmark: it is explicitly constructed around six ToM dimensions (_Emotion, Desire, Intention, Knowledge, Belief, Non-literal Communication_) and includes adversarial manipulations such as perceptual-access and asymmetric-information variations. The examples in Appendix A.1.3 and Figures 6–7 make the intended failure mode—shortcut reliance instead of genuine perspective reasoning—quite concrete.
- **Outcome-only RL on a hard social reasoning dataset appears genuinely effective.** This is one of the clearest empirical findings in the paper and is well supported by Table 3: for both Qwen3-4B and Qwen3-8B, the `w/o TRM` variant strongly improves over the base models on ToMBench-Hard and several transfer benchmarks. Independent of the more ambitious trajectory-reward claim, the paper makes a credible case that RL with verifiable outcomes can materially strengthen social reasoning when the training set is sufficiently challenging.
- **The trajectory-level reward is psychologically structured rather than generic process supervision.** The reward rubric is not an opaque “reasoning quality” score; it is organized around social cue perception, ToM-consistent interpretation, and concise reasoning, grounded in SIP theory (Section 3.2, Appendix A.2.1). Whether or not the current validation is sufficient, this is a more domain-specific and conceptually motivated process reward design than is typical.
- **The transfer evaluation is broader than the training task.** The model is trained on ToMBench-Hard but evaluated not only on ToMBench/ToMBench-Hard, but also on SocialIQA, EmoBench, MotiveBench, and SimpleToM. This broad evaluation is important because the paper’s central claim is about social reasoning rather than narrow benchmark optimization, and some cross-benchmark gains are indeed large, especially on SimpleToM and EmoBench.
- **The strongest empirical claim is parameter efficiency against some large open baselines, even if the framing should be more careful.** Table 2 does show Social-R1-4B outperforming the reported LLaMA3.1-70B numbers on all listed benchmarks. That is an interesting result in the paper as presented, even though it should not be overinterpreted as a universal superiority claim.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overclaims the necessity and robustness of the trajectory-level reward model (TRM); the ablations show mixed, not consistently additive, gains.**  
  This is the most important issue because the paper’s central novelty is not just RL on hard ToM data, but the claim that “process-level thinking rewards provide additional gains” and that “supervising the reasoning trajectory” is critical. Table 3 does not support this uniformly. For example, on **SimpleToM**, `Social-R1 4B w/o TRM = 0.9718` vs full `Social-R1-4B = 0.9365`; for 8B, `w/o TRM = 0.9741` vs full `0.8963`. On **EmoBench**, the 8B `w/o TRM` result (0.7205) is essentially tied with or slightly below the full model (0.7212), and on some settings the untrained-TRM variants are also surprisingly competitive in the appendix. So while the full method often helps, the paper does **not** establish that the trained social TRM is a consistently beneficial ingredient across tasks. The current narrative is stronger than the evidence.
- **The TRM is only weakly validated as a measure of social reasoning quality.**  
  The reward model is trained from LLM-generated and LLM-scored trajectories: o3 generates initial “gold” trajectories that are manually refined; GPT-4o/Qwen models generate candidates; GPT-5 scores them using the rubric; then a Qwen3-4B reward model is trained on pairwise preferences. This pipeline may be practical, but the paper does not show that the resulting reward correlates with independent human judgments, nor does it test for reward hacking or stylistic confounds. Since the method’s novelty rests on the claim that it supervises “human-like” social reasoning rather than surface-form compliance, this missing validation matters.
- **The benchmark contribution is promising, but construct validation of ToMBench-Hard is still limited.**  
  The paper shows that humans substantially outperform current LLMs and that ToMBench-Hard is harder than ToM-RL. That establishes difficulty, but not fully the stronger claim that the benchmark specifically isolates genuine ToM reasoning rather than a blend of ToM, narrative complexity, annotation artifacts, or linguistic difficulty. The benchmark is manually curated and proportioned across dimensions, which is good, but there are no inter-annotator agreement statistics, no systematic perturbation tests beyond a few qualitative examples, and no more formal analysis of shortcut resistance. For a benchmark positioned as a central contribution, stronger validation would be expected.
- **The empirical evidence does not disentangle whether RL is needed versus simpler supervised exposure to high-quality trajectories.**  
  The paper introduces both curated hard cases and curated/refined reasoning trajectories, then optimizes with RL. However, there is no supervised fine-tuning baseline on the same trajectory data. As a result, the paper cannot cleanly attribute gains to reinforcement learning and trajectory-level reward shaping, as opposed to simply benefiting from better social reasoning exemplars. This is particularly important because the data scale is small enough that a strong SFT control is feasible and highly informative.

### Minor
- **The training scale is small relative to the breadth of the claims.**  
  The policy is trained on 700 training samples for 300 GRPO steps, and the TRM uses a 3k preference dataset derived from 6.3k trajectories. The cross-benchmark gains are encouraging, so this is not by itself evidence of failure, but it does make claims like “genuine, robust, and systematic” enhancement of social intelligence feel overstated. A data-scaling or robustness analysis would help determine whether this is a general method or a highly data-efficient but narrow adaptation.
- **The paper leaves important reward-design questions unanswered.**  
  Equation (1) combines format, outcome, and thinking rewards, and Appendix A.3.1 states all three weights are set to 1.0. But there is no sensitivity analysis over \( \lambda_o \) and \( \lambda_t \), despite the paper’s core claim depending on the marginal value of the thinking reward. Without this, it remains unclear whether the method’s gains are mainly from the outcome reward and hard dataset, with the TRM acting as a weak auxiliary signal.
- **Some result presentation is confusing enough to impede interpretation.**  
  In Table 2, the reported percentage gains do not appear consistently tied to a single baseline convention across rows/benchmarks. Similarly, the mix of “thinking,” “disable thinking,” “+COT,” and task-specific prompting (e.g., `+MS` for SimpleToM) makes some comparisons harder to parse than necessary. This does not invalidate the results, but it does weaken clarity.
- **The benchmark annotation description is thinner than desirable for a new dataset paper.**  
  The main text says samples were “cross-checked independently by three annotators,” while Appendix A.1.2 gives a somewhat different and more detailed annotation workflow involving multiple graduate students and disagreement resolution. This is not a contradiction severe enough to undermine the paper, but the protocol should be unified and quantified more clearly.

### Trivial
- **The framing occasionally overstates what has been demonstrated.**  
  Phrases such as “human-like social intelligence” and “genuine social reasoning” go beyond what the current evidence establishes. The paper shows benchmark improvements on social reasoning tasks; it does not yet convincingly demonstrate human-like reasoning processes.

## Nice-to-Haves
- Add an **SFT baseline** trained on the same refined reasoning trajectories used in the pipeline.
- Report **human correlation** for the TRM on a held-out set of reasoning traces, or at least compare GPT-5 labels with human pairwise preferences.
- Provide a **weight sensitivity study** for \( \lambda_f, \lambda_o, \lambda_t \), especially \( \lambda_t \).
- Include stronger **benchmark validation**, e.g., perturbation-based shortcut tests, item difficulty/error analyses by ToM dimension, or agreement statistics.
- Analyze whether narrow social-RL training causes any **general capability regression** outside the target benchmarks.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the LLaMA3-70B baseline is “likely flawed,” “broken,” or scientifically unsupported because its scores are anomalously low.**  
  The paper reports those results under its evaluation setup, and there is no direct evidence in the submission that the baseline was mis-evaluated. It is fair to say the comparison should be interpreted cautiously and documented more clearly, but not to assert evaluator error without evidence.
- **Complaint that comparing a trained 4B model to an untrained 70B model is inherently unfair.**  
  This is not a valid weakness in this context. Showing that a post-trained small model can beat a much larger pretrained baseline is a standard and meaningful comparison; if anything, the asymmetry favors the baseline on raw model capacity.
- **Claim that the paper says RL for social reasoning is completely unexplored.**  
  The paper does cite prior work such as ToM-RL and describes the area as “under-explored,” which is a reasonable characterization rather than a false claim of total novelty.
- **Criticism based on repository or release-status concerns.**  
  The paper cites and links resources; availability/existence doubts are not valid review points here.
- **Pure formatting/style issues and generic reproducibility nitpicks.**  
  There are some awkward phrasings and table-formatting artifacts in the extracted text, but these are not substantive scientific weaknesses.

## Novel Insights
The paper is strongest when read as making **two separable contributions rather than one unified triumph**: (1) hard, outcome-verifiable social reasoning data is already enough to make RL materially useful in this domain, and (2) process-level social rewards are an interesting but not yet conclusively validated extension. In other words, the current evidence more strongly supports the claim that **benchmark/task design is the primary driver**, while the specialized social TRM remains a promising but only partially substantiated add-on. Reframing the paper this way would make its empirical story both more honest and more compelling.

## Suggestions
- Reframe the main claim: present **outcome-based RL on ToMBench-Hard** as the most solid contribution, and describe the TRM as a **promising but mixed** extension unless stronger evidence is added.
- Add a **matched SFT baseline** on the same reasoning data to isolate the value of RL.
- Validate the TRM against **independent human judgments** and probe whether it rewards social reasoning quality rather than style or rubric mimicry.
- Include a **reward-weight ablation** to show whether \( \lambda_t \) contributes distinct value beyond outcome reward.
- Strengthen ToMBench-Hard with **agreement metrics**, deeper error analysis, and more systematic evidence of shortcut resistance.
- Clarify table comparisons and baseline conventions so that percentage gains and “thinking vs. no-thinking” settings are immediately interpretable.

Overall, the paper has a real idea and some genuinely encouraging results, especially around hard-data RL for social reasoning. But the present version overstates what has been proven about trajectory-level supervision and needs sharper empirical isolation of where the gains actually come from.