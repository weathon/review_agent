# Unintended Harmful Knowledge Elicitation Issue in Large Reasoning Models and a RL Solution

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The capabilities of large language models (LLMs), particularly large reasoning models (LRMs), are rapidly advancing. This raises concerns about whether LRMs can maintain their safety awareness throughout long-form reasoning. Frustratingly, we identify a prevalent safety issue across LLMs and LRMs, where LRMs can reveal dangerous thoughts, leading to harmful knowledge elicitation when confronting sensitive yet benign topics. For example, when explaining the chemical context of Lewisite, a biological weapon, LRMs analyze its synthesis in their reasoning without recognizing the associated risks. We refer to this issue as the $\textit{unintended elicitation}$ issue. Experiments on our benchmark show that it is a common issue across current LRMs due to their strong multi-step reasoning capabilities. To address this issue, we propose placing LLMs in our synthesized open-ended environments, allowing them to self-search for a safety reasoning pattern to respond responsibly and helpfully. We first design a scalable data synthesis pipeline to generate data that triggers the ``$\textit{unintended elicitation}$'' issue. We further propose a safety-first reward model design, which prioritizes safety while also evaluating the helpfulness of responses and the faithfulness of reasoning. Experiments show that our method improves safety, reduces over-refusal, and maintains strong helpfulness, paving the way for safer deployment in high-stakes domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new dataset and a new RL reward to train language models not to elicit harmful content when reasoning, a failure case in safety alignment that this paper also introduces. From their new dataset and new reward function, they train models that can reason for longer without failing to remain "safe" while improving performance on various safety tasks that involve refusal (refusing to answer harmful prompts) and over-refusal (refusing to answer a harmless prompt) all while maintaining accuracy on common tasks (GSM8k and MMLU).

The dataset created by the authors is generated semi-synthetically by having an LLM generate prompts using sets of seed prompts as examples in their prompt. The seed prompts follow a set of properties such as sensitivity (the prompt may elicit harmful responses), open-endedness (encourages the model to reason / generate a lot of text not necessarily a direct answer) and neutrality (the user didn't request harmful information specifically). These properties help the LLM create additional challenging prompts that may elicit harmful content. The authors then train on this dataset using REINFORCE (RL) with GPT-4 grading each response on the responses safety (-1 if unsafe, weighted reward if safe), if it was faithful and if the response was helpful.  These three weights in addition to the hard -1 penalty for unsafe generations leads them to create a very robust model that can not only refuse harmful requests, but will also reason and answer requests without eliciting harmful material. The authors pair this with significant amounts of ablations to show how each part of their pipeline effects the model.

### Strengths
- **A very well thought-out experiment that is clearly articulated with positive results**. The authors have a clear argument for adding these additional weights during RL paired with fairly convincing ablations that their data, prompt, rewards, etc. are all useful in training and improving the models ability to remain safe over long reasoning generations.  
- **Fairly easy to reproduce** - the methods outlined here suggest aligning on open-ended prompts with a specific reward formulation, this seems fairly achievable in most research settings and the appendix includes fair amounts of documentation on reproducing the authors experiments.

### Weaknesses
1. **Most of the related work is in the appendix**, I don't know if you can push the related works into the appendix like this. I think a better effort should be made to fit the related work in appendix A into the main body of the paper. For reference the main body related works has 7 papers cited but appendix A has a ton and is clearly the "real" related work section. I know if the paper is accepted the page limit goes up to 10 pages, but **this feels rather unfair to the other submissions** in my opinion. But to be absolutely clear where I am at, I do not know if this is "acceptable" so I could be swayed by a compelling argument and no changes.
- Because of this my rating will likely be a 4 until it is fixed, but I do think this paper is great! I just do not want to be unfair to the other submissions. So, fix the related works and give a better effort to fit it in the main body, please!! I think I would minimally be happy with the main body having the 3 sections the related works has in the appendix (but they can be shorter variants I suppose?)
- If you can point to examples where this is acceptable or give a good argument why this should be allowed, I'd also hear that (cutting isn't easy, so I understand this is a big ask).

2. **The method may be over-reliant on a system prompt** - Table 6 concerns me.  Firstly, I may be misunderstanding the "Vanilla" row but I took this to be "Standard RL without the new rewards".  If this is the case, then removing the safety principle seems to really hurt the performance of the model on over-refusal. "This indicates that the prompt is not a static instruction but a critical driver of safe reasoning", I think this may be more related to the roll-outs you generate when training. More analysis on this would be nice, it looks like the reward functions and the dataset are actually very sensitive to the prompts you use during training which is an important detail for those who would want to reproduce these results.  Either more analysis is needed (try training with alterations of the safety principal) or this should be brought to the forefront in the intro. 
- I don't like that this crucial detail is at the very end of the paper, it seems misleading to talk about how the dataset + reward functions increase robustness and decrease over-refusal when really you need this key prompt to get things started in the right direction.

3. **Potentially costly method** I don't think I saw anywhere in the main body a discussion on how using GPT-4 as a reward model hurts training performance (do batches go much slower?) nor anything about monetary cost (this may be a smaller point, but it would be nice to get cost estimates so that other researchers know how much it'd be to retrain this).  Additionally, using different reward models, especially for weaker / opensource models, would help here as well. 


Notes (not weaknesses just callouts):
- Line 255/256 -> "1k4" I think was meant to say "14k"?
- Line 409 -> "At the same time" is used twice (just jarring to read)
- Table 4 -> the capability section has bolded numbers that are not the maximum (the baseline performs better in both cases). Unless I misunderstood what the bold was distinguishing.

### Questions
- Are there any signs of reward hacking? Or any analysis trying to find if the model is hacking the reward? This seems like a big problem in complex reward setups.  
- How does GPT-4 perform on all of these baselines? How do we know we haven't just "distilled" in some weird round-about way the capabilities of GPT-4 to distinguish unsafe content? (I suppose table 3 kind of answers this with the ablations on the rewards themselves).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The manuscript identifies a recurring safety failure in large reasoning models: during open-ended reasoning on sensitive but non-malicious prompts, models may unintentionally generate harmful procedural or technical details. This phenomenon, termed the unintended elicitation issue, emerges when models attempt to be thorough and helpful, revealing dangerous intermediate reasoning steps without recognizing associated risks. To study this, the manuscript constructs a dataset of open-ended prompts synthesized to encourage broad exploration around sensitive topics. Experiments show that unintended elicitation is widespread across contemporary models, including recent reasoning-oriented systems.

To address this problem, the manuscript introduces OpenSafeRL, a reinforcement learning framework that trains models to develop context-aware safety reasoning rather than defaulting to refusal. The method uses a hierarchical reward structure prioritizing safety while also evaluating faithfulness and helpfulness, and incorporates a safety principle prompt to guide reasoning. Results show improved robustness to jailbreak attacks, reduced unintended elicitation, and lower over-refusal rates, while maintaining general task capability.

### Strengths
The manuscript clearly identifies and formalizes a failure mode that is not well captured by prior refusal-focused safety work. By centering on cases where the model’s reasoning process inadvertently crosses into harmful detail, the work highlights an important gap in current safety evaluation standards. The dataset used to elicit this behavior is structured to surface situations where risk emerges gradually during reasoning rather than at the prompt level, which results in a more realistic assessment of how such harm may occur in practice.

The alignment approach is notable in how it prioritizes context-sensitive safety rather than categorical refusal. The reinforcement learning framework does not simply penalize unsafe output, but structures rewards so that the model learns to distinguish safe explanatory detail from unsafe procedural detail, producing measured responses appropriate to the prompt. This approach supports both safety and usefulness, and helps avoid over-refusal, which is a common side effect of existing safety fine-tuning methods.

The experimental evaluation spans multiple models and includes comparisons against standard refusal-based alignment techniques. The improvements are demonstrated across several adversarial prompt settings and reasoning benchmarks, and ablations isolate the contributions of different components in the RL framework. The results show a consistent pattern of reduced harmful content without diminishing performance on general reasoning tasks, suggesting that the method addresses the target failure mode rather than suppressing overall capability.

### Weaknesses
The evaluation relies heavily on an automated harmfulness classifier to assess whether outputs contain unsafe reasoning. Since unintended elicitation often involves gradual escalation rather than explicit directives, the sensitivity of the classifier to subtle forms of harm is critical. If the classifier under-detects nuanced mechanism explanation or conditional procedural guidance, the reported reductions in harmful output may overstate the true level of safety achieved. An assessment of classifier reliability, or partial human validation, would improve confidence in the conclusions.

The work focuses on domains where harmfulness is linked to technical procedural detail. It is less clear how the approach extends to settings where harm is contextual or socially mediated, such as persuasion, planning, or misrepresentation tasks. These forms of unintended harm may not involve explicit stepwise reasoning that can be easily penalized or redirected. Demonstrating generalization to a domain where the criteria for “unsafe detail” are less structurally defined would broaden the applicability of the proposed method.

The approach assumes that the reward model and safety principles are themselves well-calibrated. If either contains systematic blind spots, the RL process may reinforce patterns that appear aligned under the reward model but do not generalize to real-world cases or to adversarially structured prompts. Discussing methods for monitoring or correcting misalignment in the reward signal during training would clarify the robustness of the framework.

### Questions
Does the approach generalize to domains where risk is social or contextual rather than technical?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a previously underexplored vulnerability in large reasoning models (LRMs): the “unintended harmful knowledge elicitation” problem, where models unintentionally reveal unsafe or sensitive information while engaging in multi-step reasoning on benign but risky topics.
To address this, the authors propose OpenSafeRL, a reinforcement-learning alignment method that trains models to reason safely through sensitive content rather than relying on binary refusal.
OpenSafeRL uses a hierarchical safety-first reward model that jointly optimizes for safety, faithfulness, and helpfulness.
Experiments on safety bench, jailbreak attacks, and over-refusal benchmarks demonstrate that the approach reduces harmful outputs while maintaining reasoning capability and lowering over-refusal rates

### Strengths
1. The paper clearly articulates the unintended elicitation phenomenon, distinguishing it from classic jailbreaks or explicit malicious prompts. This is a meaningful conceptual contribution to the safety-alignment literature for reasoning-capable models.

2. The proposed hierarchical reward model elegantly balances safety, faithfulness, and helpfulness. Its design avoids blanket refusals and yields context-aware safety behaviors that generalize across tasks.

3. Results cover safety robustness, over-refusal, and capability, illustrating an appropriate trade-off between safety and performance. Ablations meaningfully analyze reward and data contributions

4. The paper is well-written, with clear examples (e.g., Lewisite case) that illustrate how reasoning chains can expose unsafe content. The visualizations and structured presentation improve readability.

### Weaknesses
1. The experiments are conducted primarily on two medium-sized backbones (Qwen-2.5-7B and LLaMA-3-8B). While the improvements are consistent, it remains uncertain whether the method scales effectively to larger LRMs (e.g., 30B–70B) or frontier reasoning models such as o1 or DeepSeek-R1. Broader model coverage would strengthen generality claims.


2. The hierarchical reward relies on GPT-4o as a reward model; yet, potential biases or inconsistencies in GPT-based judgments may affect reproducibility. Some calibration analysis would be valuable.

3. Reinforcement learning with multiple reward dimensions and GPT-based evaluation may be expensive. Discussion of efficiency or practical feasibility would help readers assess deployment viability.

### Questions
1. The hierarchical reward model introduced in the paper shares conceptual similarities with RLVR approaches, where each sub-objective (safety, faithfulness, helpfulness) can be framed as a verifiable or partially verifiable reward component.  
  Have the authors considered reformulating their method under a verifiable reward learning perspective? For instance, could components like $\textit{faithfulness}$ or $\textit{factual correctness}$ be grounded in deterministic verification signals rather than relying solely on GPT-based judgments? Such integration could improve both interpretability and reproducibility of the reward structure.


 2. The total reward is defined with $\alpha=0.5$, $\beta=0.3$, $\gamma=0.2$.  
  How were these coefficients determined—purely empirical tuning, or a principled multi-objective procedure (e.g., Pareto front search, Lagrangian weighting?)  
  Since the three components can have different scales and gradient variances, does linear weighting risk gradient dominance or unstable training?  



 3. In multi-step reasoning, rewards tend to become sparse or delayed. How does OpenSafeRL ensure stable gradient propagation through long reasoning trajectories?  

4. It would be insightful to analyze how the three reward signals (safety, faithfulness, helpfulness) disagree during training.   Quantifying their divergence or conflict rate could provide a clearer picture of how different alignment objectives interact, and whether specific conflicts (e.g., safety vs. helpfulness) dominate in certain reasoning domains.

5. Since GPT-4o is used for multi-dimensional reward evaluation, how scalable is the framework for large models or long-horizon reasoning tasks?  Could a smaller distilled verifier or heuristic-based verifier maintain comparable safety--helpfulness trade-offs while improving computational efficiency?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the issue on LLMs where the reasoning trace that the model generates, to assist in getting to the final answer, reveals sensitive information unintentionally that wasn’t explicitly queried. This can result either in unsafe LLM outputs, or the refusal of any output as the LLM incorrectly deems the query to be unsafe. To address this, the authors synthesize data containing queries that result in such behaviour and fine-tune models on this data using RL, wherein the reward incentivizes safe outputs.

### Strengths
The paper shows empirical evidence of an important issue: increased capabilities in LLMs due to extended reasoning lead to unintended unsafe consequences in the reasoning process, as well as overly cautious behavior in the final model output. The proposed fine-tuning procedure successfully reduces the latter.

### Weaknesses
The main focus of the work is on *unintended elicitation*. However, **none** empirical analysis explicitly evaluates whether this elicitation in the reasoning process has been mitigated. The results all evaluate the final performance (final answer safety, final answer refusal, final answer capability). The results are thus not indicative of having reduced *unintended elicitation* or affected the reasoning traces in any manner. 
- A reasonable proxy to showcase that would be to show qualitative examples of change in the reasoning process before and after RL training. Or alternatively, using some evaluator to assess the reasoning traces for unintended elicitation.

### Questions
- How is the inclusion of a “wait” token analogous to deeper reasoning? Could the authors clarify how this intervention is done to increase the length of the reasoning trace?
- As per Table 5, the synthesized data and the reward model are effective on different fronts: one for attack robustness, seemingly by increasing over-refusal to fend off attacks, and the other for reduction of over-refusal, thereby becoming more susceptible to attacks. This seems to suggest that just fine-tuning on the synthesized data without any specialized rewards (for example, the vanilla reward of getting the answer right) should show significant improvement. Is that accurate?

### Soundness
3

### Presentation
2

### Contribution
3
