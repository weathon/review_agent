# The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM and generalizes beyond detoxification to a helpfulness preference setting, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a framework to audit alignment of LLMs. The authors look at how to estimate the reward of an LLM ("reward inference") with a three-stage auditing process which is called the Alignment Auditor. 

The first step is "making ambiguity explicit" by which the authors mean learning a posterior distribution over reward parameters theta. They put a Gaussian prior over theta and a Bradley-Terry likelihood for pairs. They show that "non-identifiability can be systematically reduced", meaning that the posterior spread of theta (a proxy for non-identifiability) is reduced by making its uncertainty shrink.

Second, they use "uncetainty-aware diagnositcs" by which the authors mean they decompose uncertainty into aleatoric and epistemic and run some tests. The goal of these tests is to show that the inferred reward cannot be trusted. The tests include (a) injecting irrelevant tokens which they argue should increase epistemic uncertainty; and (b) correlate the variance of the reward with the distance from the training distribution.

Third, they show that the "refined objectives", i.e. the inferred reward function (the mean of the posterior reward distribution), when used in RLHF, show a decrease in the percentage of toxic completions on a held-out dataset on risky prompts. 

In brief, the authors turn the problem of bayesian IRL into an auditing process, where the parimary goal is verification.

### Strengths
Great work, it was fun reading it.

1. Clearly formalised, motivated. It's connected to existing literature well and the idea of using reward shrinking as a way to understand if the inferred reward is good, as a tool of auditing, is cool and nice.

2. Very well-written and presented. Thanks.

3. Results are clear.

4. It's nice that you obtain calibration scores and show the decomposition of rewards over different rounds.

### Weaknesses
(To be clear, the weaknesses section is longer than strengths, but I think the paper has more strengths than weaknesses).

I (probably) understand why you chose to frame the paper the way you did. However, I feel this has exposed the paper to a few issues:

1. Your primary claim is that you are an *auditing* framework and you show you find where the rewards have some issues. If this is indeed the primary claim -- we have an *audit framework* to find issues -- you *must* show that other methods of auditing cannot achieve the same thing. For instance, for the toxicity setup (page 6-7), you must show that other ways of auditing LLMs, with the same train-validation split, would not have been able to identify issues in alignment. What do behavioral audits, adversarial audits, uncertainty-based audits miss? Even though your *mechanism* how you audit is different -- this is understood -- the *outcomes* of what you flag or not are not, relative to the alternatives.

2. Relying on retraining to prove usefulness is of course a very expensive procedure. I understand the academic value of this, but the extent to which this will become a practical tool is highly questionable. (Which is alright -- there's value in this either way). 

3. There's a bit of a confusion, it seems, on the framing on "auditing". After you obtain your reward after posterior shrinkage, you continue to optimize the model with the expectation of the posterior. Then, it blurrs the line between "audit" vs "fixing" the problem because now you're re-training the model again, not auditing the original reward.  In other words: you're now no longer just saying *this reward had XX issues*, but also saying *with this new reward, you get XX better properties*. This is fine -- but then this doesn't become an "audit" tool, as it's now a re-training method. I think this is fine and good -- but just it does not square in terms of the framing that this is an auditing mechanism.

I think the biggest part for me is weakness (1). For you to make this a better auditing paper (which I understand it is), you need to do more to compare against other auditing methods and show what are the unique benefits you discover. As it currently stands, the paper largely explores your own method and properties instead of giving a comparative analysis of auditing and recovering issues.

### Questions
1. Is the fact that your estimates are calibrated a purely empirical observation or is there any theory to support why this should/shouldn't be the case? 

2. Any things you tried before that *did not* work and you left out of the paper?

3. Any reason you used PPO instead of GRPO?

4. Are you the first work to look at modeling and contracting the posterior reward distribution or has this been done before?

5. If you could address the three points from the weaknesses, that'd be great too. Especially weakness 1.

I would be mostly convinced to increase my score if:
- (a) You can give a better related work overview on audit procedures both in terms of (i) mechanisms (how audit procedures work) and (ii) outcomes (what audit procedures uncover)
 - (b) show me how you uncover issues that other audit procedures do not. It does not have to be a "better" issue uncovered -- any single dimension along which you discover issues that no other audit framework discovers would be strong evidence for me.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors prescribe a methodology to audit reward scores for RLHF pipeline.

### Strengths
The paper, in general, is well written, and the ideas are presented with clarity. I am not very familiar with audit literature, so I cannot comment on the technical novelty. However, the methodology proposed is sound, and I am convinced of its correctness.

### Weaknesses
I believe the primary cost stems from the computational overhead. Primarily with Stage 2, where one runs Bayesian updates over K iterations to obtain the final reward model for use.
Also, I may be mistaken, but how is one guaranteeing the correctness of \pi_E, the expert?

### Questions
In Figure 6, it appears that after one round, the reward metrics closely track the ground truth. I am assuming the round refers to those in stage 2. In that case, can we get away with doing just a single round? More importantly, the task methodology seems quite intensive. Does it make sense to invest in the time and money, given that we have reward-free alignment techniques? 

How would one extend this audit to a reward signal-free technique, for example, in the case of DPO?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper formalises a LLM auditing framework that relies on Bayesian IRL to infer some ground-truth toxicity labels of completions.

### Strengths
The paper presents a successful experiment to recover the reward (objectives) that was used to finetune an LLM model. 
The experimentation spans a good variety of LLM models and appears robust, though large LLMs are a bit under explored. 
The presented framework is a creative way to verify the level of alignment reached by the LLM during RLHF. 
A clear and an easy to follow presentation. 
LLM alignment is an important and hot topic.

### Weaknesses
The weaknesses in summary:
- The paper would be more clear if there was more emphasis on the auditing capability of the framework.
- Unclear positioning wrt SoTA. 
- Some conclusions would be more solid if supported by more evidence.
- The results would be more sound if a more solid ground truth was used.
- The title’s claim of “refinement” would be clearer with demonstrated refinement rather than validation and partial transfer
- Simplicity of experiments could prevent conclusions significance and generalization (single case and binary one: toxic or not, LLMs < 1b parameters)

In more details, to explain and justify the points above:

- The goal of the paper evolves around the auditing capability of the framework: line 017 "This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification.", line 057 "In this work, we argue that understanding the behavior of LLMs through reward inference should not be approached as a one-shot estimation problem, but as a principled auditing process.". But there is not enough demonstrated auditing capabilities: 
    - No auditing in Stage 1 since we are just learning to distinguish the expert's from the baseline's completions while learning a reward that the expert's maximizes compared to the baseline.
    - On the second Stage, the author adds more iterations and more data to the expert, its scores improve. Also there is new introduced metrics: "Total Uncertainty" and "Epistemic Uncertainty", that makes the model aware of the data points that he would not be able to score (score here is the reward) correctly. But no Auditing capabilities in Stage 2.
    - On the third stage there we verify that the reward that we captured is can be used actually to reduce toxicity. At this stage (3) also it is not auditing.
-lines 105-110 present other BIRL approaches, but none are evaluated in the experimental section --  those would have helped identify the benefits of the proposal. In particular, lines 110-112 did not convince me completely of the novelty of the approach. 
- Considering a toxicity classifier as the ground truth can effect the results of the experiments in many ways.
    - The used toxicity classifier (s-nlp/roberta_toxicity_classifier) have an F1-score of 0.76 (https://huggingface.co/s-nlp/roberta_toxicity_classifier#licensing-information) which is comparable to the results that were reached in the paper.
    - Also there is no discussion in the paper about the limitations and the effects of having a toxicity classifier as ground truth. 
    - Toxicity reduction is arguably only a fraction of general alignment, the authors could discuss how their approach transposes to other alignment objectives.
    - I missed a note regarding the limits of completion-based toxicity detection: I expect the toxicity of answers to heavily depend on the original question! 
    - the paper would be improved by bringing central technical notions (like posterior contraction) closer to their auditing meaning/use. Besides, I would have enjoyed a deeper discussion on the problem of non-identifiability. While the pointed paper (Ng&Russell) clearly illustrates non-identifiability, I believe in learning contexts this non-identifiability is frequent (eg justification of SVM margins) and not always problematic -- provided the inferred model is close enough from the ideal one. 
- In many instances, the paper makes some conclusions with only limited support, for example:
    - In line 377 the author claims "Importantly, sequential Bayesian updates tend to mitigate the effects of reward hacking in comparison to single-round inference (see Figure 5(right)).". The Figure 5 (Right) shows the toxicity reduction in the RLHF process and it compares different rewards on that metric. The "ground truth" does the best, Round 1 does better than the the other Rounds which is not a proof that there was reward hacking at Round 1.
    - In line 473 is the conclusion there is a statement "Third, policy-level validation shows the inferred reward can directly drive PPO,  achieving comparable downstream toxicity reduction compared to the ground truth RLHF alignment." which is questionable, because if we compare the toxicity reduction ratios between the Ground Truth RLHF and the other rounds (Rounds 2-5) in the Figure 5 (Right) we find that the reduced toxicity of the recovered reward is nearly half the reduction by ground truth. 
    - There is not enough numerical metrics and evidence to reinforce the claims based on qualitative results (426 e.g. "With the round-1 (poorly identified) posterior, policy-level alignment with PPO exhibits reward hacking where completions show topic loss, repetition and abrupt cut-offs that suppresses toxic tokens at the expense of coherence and helpfulness (Appendix Tables 1).") or visualization of graphs (353 "while the distribution of inferred rewards in Figure 2(b) reveals a distinct decision boundary between scores assigned to toxic and non-toxic texts.")
- In the title "THE ALIGNMENT AUDITOR: A BAYESIAN FRAMEWORK FOR VERIFYING AND REFINING LLM OBJECTIVES", there is a claim of a "Bayesian framework for ... and refining LLM objectives". The refinement is not demonstrated in the paper. What could have been referred to here is the **validation** of the recovered reward in the third stage. But to get the reward that was used to finetune the baseline LLM in Stage 3 we needed a finetuned version of the same LLM (Expert) first. Meaning the finetuning in Stage 3 is not a refinement of the LLM objectives as inferred to in the title, it is a transfer of **some** of the alignment learning of the expert to another baseline version of itself.

### Questions
- What motivated the use for Roberta toxicity classifier given its reported F1~0.76? And how would the conclusions change if we consider the classifier as a noisy labels rather than ground truth?
- For the claim "Importantly, sequential Bayesian updates tend to mitigate the effects of reward hacking in comparison to single-round inference (see Figure 5(right))."(l.377). What evidence supports this conclusion beyond the Figure 5 (right)?
- How is "comparable" defined in “comparable downstream toxicity reduction” (l.473)? And what evidence supported that claim in the conclusion ?
- How is a "distinct decision boundary" defined in l.353, and what metrics and evidence beyond visualization can be presented to reinforce that claim?
- What concrete audit outputs (e.g. reports like mentioned in l.476) does the framework produce?
- What example can be provided to demonstrate clearly the auditing capabilities of the presented framework?
- What is inferred by "refinement" in the title: "Bayesian framework for … and refining LLM objectives"?
- The experiment demonstrates that the proposed Bayesian auditing method can recover a reward function consistent with the oracle reward, but could the simplicity of this use case limits the significance of the result? Does reproducing oracle RLHF dynamics necessarily imply alignment with human values/preferences? could it be only mere consistency with the proxy signal used during training?
- Any idea of how challenging that would be to go beyond linear model for reward?
- What about moving from the toxicity to helpfulness or even open-ended generation? What challenges would that raise?
- Existing related works seem extensive, is there any way of direct comparison with some of them? that would help grasp the contribution utility of your work.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces an auditing framework that reframes reward inference as a verification process. It uses Bayesian IRL to recover a distribution over possible objectives and supports key auditing capabilities.

### Strengths
This paper presents a framework called Alignment Auditor that structures reward inference into three stages: recovering a distribution over objectives to reduce ambiguity, using uncertainty-aware diagnostics to assess trustworthiness, and training policies using the refined objectives to improve practical utility.

### Weaknesses
The task setting is narrow, focusing only on toxicity detoxification. The ground-truth reward comes from a classifier rather than human preferences, and the reward model is linear over frozen features.

### Questions
The experiments cover only toxicity detoxification. Does the framework apply beyond single-objective safety tasks? How would it handle multi-objective alignment (e.g., helpfulness–harmlessness–honesty) and trade-offs among complex human values?

The method assumes a linear reward over frozen features. Do the findings extend to more realistic, scalable reward architectures (e.g., finetuned, nonlinear, or deep reward models)?

What are the limitations of using a proxy classifier as “ground truth”? Can the authors discuss extensions to real human preference data and how results might change?

Is there evidence or a clear rationale that the auditing procedure scales to modern large models? What are the computational and methodological bottlenecks?

How do the authors ensure that the reported uncertainty reflects genuine reward ambiguity, rather than artifacts from model misspecification, feature freezing, or estimation noise?

### Soundness
3

### Presentation
2

### Contribution
2
