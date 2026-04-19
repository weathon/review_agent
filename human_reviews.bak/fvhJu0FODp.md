# Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 3

## Abstract
Recent breakthroughs in large language models (LLMs) have brought remarkable success in the field of LLM-as-Agent. Nevertheless, a prevalent assumption is that the information processed by LLMs is consistently honest, neglecting the pervasive deceptive or misleading information in human society and AI-generated content. This oversight makes LLMs susceptible to malicious manipulations, potentially resulting in detrimental outcomes. This study utilizes the intricate Avalon game as a testbed to explore LLMs' potential in deceptive environments. Avalon, full of misinformation and requiring sophisticated logic, manifests as a "Game-of-Thoughts". Inspired by the efficacy of humans' recursive thinking and perspective-taking in the Avalon game, we introduce a novel framework, Recursive Contemplation (ReCon), to enhance LLMs' ability to identify and counteract deceptive information. ReCon combines formulation and refinement contemplation processes; formulation contemplation produces initial thoughts and speech, while refinement contemplation further polishes them. Additionally, we incorporate first-order and second-order perspective transitions into these processes respectively. Specifically, the first-order allows an LLM agent to infer others’ mental states, and the second-order involves understanding how others perceive the agent’s mental state. After integrating ReCon with different LLMs, extensive experiment results from the Avalon game indicate its efficacy in aiding LLMs to discern and maneuver around deceptive information without extra fine-tuning and data. Finally, we offer a possible explanation for the efficacy of ReCon and explore the current limitations of LLMs in terms of safety, reasoning, speaking style, and format, potentially furnishing insights for subsequent research.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper explores the challenges LLMs face in processing deceptive information, utilizing the game Avalon for evaluation. It demonstrates that incorporating human-like cognitive patterns improves LLMs' performance in the game. The authors introduce "ReCon", a method that merges "formulation and refinement contemplation" with "first- and second-order perspective transitions" for LLM prompts. In practice, this involves a phase in which the LLM is asked to contemplate internally from a first-order (its own) perspective, and then to contemplate again from a second-order (other players') perspective. This method's efficacy is validated using ChatGPT and Claude in the Avalon game.

### Strengths
The results are interesting, the paper is well written, and the analysis is reasonable. The work is also novel, I believe, and it improves our understanding about how to make LLMs safer.

### Weaknesses
My main comment is that I would like to see how the proposed approach may be applied in case of attempts to extort PII from LLMs or jailbreaking outside of the Avalon environment. It should be fairly straightforward to assess this. Have the authors tried it or can they comment on it at least? 

Other comments:

* The paper argues that ReCon is superior to CoT and its variants that lack either contemplation or perspective transition. I think that the results are pretty convincing, although I want to see some confidence intervals for the estimates in Figure 4. One inconsistency I noted, which the authors also mention, is the differing performance between ReCon without refinement contemplation and ReCon without formulation contemplation when applied to Claude versus ChatGPT. Some insight into this discrepancy would be useful. It might also be beneficial to contrast paired strategies, such as ReCon with and without refinement contemplation, to reinforce the findings. Have the authors tried this approach? 

* Another question is regarding the scalability of LLMs: Would CoT match ReCon's performance when applied to larger LLMs? 

* The number of games used to generate Figure 4 is unclear. It's unclear whether Figure 4's results are derived from the 20 full Avalon games mentioned in the 4.2 Setup, especially since multiple strategies were simultaneously deployed for a single side (good/evil) in this context so that's likely not the case. Clarification on this, along with the inclusion of confidence intervals for the results mentioned above, would provide readers with a clearer understanding of the robustness of the findings.

* The statement "we use ReCon for the good side" seems inconsistent and should likely read "we use CoT for the good side," correct? The paper also contains a few typographical errors that need rectification.

* It would be beneficial for the authors to disclose the specific prompts used for GPT-4 across the various evaluation metrics. Furthermore, it's worth pondering whether the results would remain consistent if humans, rather than CoT, played the game. Lastly, why omitting Claude's performance across all evaluation metrics.

### Questions
Mentioned above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an LLM prompting setup termed "ReCon" consisting of

1. one prompt from the perspective of the agentic-role the LLM is intended to assume, consisting of a description of the scenario, the task description and task-hints (information about the game the prompt is evaluated, what to pay attention to, generalized action recommendations)
2. one prompt to guide the LLM through a THINK-SPEAK pattern similar to ReACT, with nudging language specifically guiding the "THINK" inference towards the goal
3. one prompt restating the role, potentially updated scenario and asking the model to critique the "SPEAK" content (but not the THINK content) from the viewpoint and goals of the other roles, again with specific nudges (similar to Self-Ask)
4. one prompt asking the LLM to act as a critique of a player+role, restating the updated situation and asking the model to offer possible improvements guided by hints

This prompt template is avaluated on the role playing game "Avalon: The Resistance" which is described in detail in the appendix and evaluated by having OpenAI GPT-4 and Anthropic Claude play $n=20$ games of one of four ablations (Recon, Recon wo/ ReACT aka prompt , ReCon w/o Self-ask  aka prompt 3, CoT) and evaluated across 6 metrics based on asking GPT-4 to compare rank the two (ranking primpt not included).
A comparison on LLama 2 was attempted but not completed due to formatting-output issues.

### Strengths
I feel very mean and mean no offense but I cannot point to any particular unconditional strength in the main axes of consideration.
In an effort to highlight positives:

1. The complete paper is very pretty and visually appealing
2. based on the related work the authors are very familiar with the literature.
3. The method *does* win against the baseline in the task proposed

### Weaknesses
1. If the point is to present Avalon as a new benchmark for langauge game playing that as the paper claims "goes beyond a language game and a game of thought", then that benchmark would have to be evaluated with *all* currently relevant baseline methods, as well as studied and discussed in detail to justify the claims: *why* is it a game of thought while e.g. the other games discussed in Serrino et al aren't?
2. If the point is to contribute ReCon, then what is missing is an evaluation on HumanEval, or some other task except the one it appears to be specifically constructed for. E.g., how would one adapt this to a negotiation game, a market game or another situation in which the roles are not as clearly defined etc?

On top of this, the methodology of the evaluation is not strong enough to support anything.
This is currently par for the course and I am sorry that the authors encounter me as a reviewer, but very basic scientific standards currently disqualify most of the established evaluation procedures:

- [recent leaks of basically all large LLM providers system prompts](https://gist.github.com/cedrickchee/9390389d755e574cca24a2b42aaa7d47) show the large possible confounding nature of using blackbox APIs for any form of evaluation, beyond the unknown other methods and filters that might be employed. The authors acknowledge the data and version they use, but this makes replicability impossible, even without the fact that the non-zero temperature and sparse MoE inference at these APIs makes the evaluation indeterministic
- $n=20$ appears very low to me and I would like to see a suitable statistical significance test together with the motivation for choosing and interpreting it (i.e., choosing an appropriate correcting like [this one](https://en.wikipedia.org/wiki/Bonferroni_correction), or justifying a less conservative one)
- I am also confused why the authors didn't use simply use [guidance](https://github.com/guidance-ai/guidance) for their LLamav2 experiments, since they seem otherwise very tuned into the LLM world.  However, Occams *and* Hanlons razor tell me to assume simple unawareness. If the experiments succeed with the help of Guidance, I'd be curious to see the updated results
- That tuned-in-ness brings me to another critique: the described method is basically a specialized application of [Tree of Thoughts](https://arxiv.org/abs/2305.10601). While it might be a concurrent work, the authors also do not adequately put their method into context with ReACT and self-ask as I have done in my summary. What would need to be done is to create a matrix or table comparing all extant/widely used prompt techniques and comparing them while highlighting only *unquestionable* differences
- The room for this could be gained by removing a lot of the unsubstantiated bombast from the introduction and section 2. While I appreciate the aesthetics, I don't think quoting the three body problem, discussing AGIs or making grand claims about CoT not causing the Morgana-role to disclose its thoughts *despite that being in the role description* is suitable for an otherwise quite thinly supported paper. Without more *stats* and *quantifiable information* in the appendix, I also think we do not need to see more individual examples of impressive chats in LLM papers. If e.g. Fig c) shows consistent behaviour, what I would *expect* to see is a section in the appendix detailing 
  1. a definition of hidden thought deception that is operationalizable, maybe even with python code
  2. An experimental setup that evaluates this operationalized definition, including all prompts etc. used
  3. a statistical evaluation
  4. a discussion of possible confounders and alternative explanations
- meanwhile, the authors *don't even give the complete prompt in the appendix*. This is the room where you do not need to worry about brevity dear authors, please just add the original prompts, the prompts used to perform the GPT-4 ranking, basically *everything* that could be used to evaluate, judge and replicate the paper
- finally, these are nitpicks, but it is meant to illustrate why I can't even call the writing clear or pleasant:
    - In the introduction, beyond lacking any justification for claiming a "game of thought"...
    - ...how exactly did your findings (which I assume refer to your method and the experiments) inspire your framework?
    - In the appendix, Park et al. 2023 is not a methodological paper, it is a critical evaluation of deception in Machine learning
    - In Appendix g, the Assasin prompt reads "The evil tema is close to losing; you must guess who Merlin is. losing; you must guess who Merlin is". Is this an LLM artifact or a copying error?
  this is a sampling of the general feeling of sloppiness/lack of attention to detail that permeates the paper.

---

If the paper decides whether to present a new benchmark or a method as its focus, then it can be salvaged:

1. For the method, evaluate on standard benchmarks and add your avalon benchmark as a bonus. Properly compare against ToT and other baselines.U
2. For the benchmark, use more open source models and create a detailed story, supported by suitable experiments, to demonstrate *why* this is a "game of thought" as opposed to  a "language game", and throw in your method as the bonus

For both, use guidance to create a controlled experiment using LLama or the recent mystral etc. finetunes, use proper statistical measurements and report things in a manner that can actually be compared (e.g. the table highlighting differences) and reproduced (all exact prompts!)

### Questions
I tried to be very constructive and complete in my weaknesses section, hence I do not have questions that would meaningfully change my opinion without those issues addressed

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper attempts to study and improve LLMs' ability to produce situationally appropriate responses in a strategic environment requiring deception and the ability to detect deception. LLMs are prompted to play roles in the social-deduction game Avalon. The experimental intervention is a new prompting technique ReCon, which explicitly prompts agents to contemplate the roles of other players based on the game history, and then explicitly prompts them to contemplate how other players will perceive them. The outputs are then evaluated in two ways: (1) by playing complete games of Avalon to see whether ReCon-prompted agents perform better, and (2) having GPT-4 classify outputs as successful or not along six dimensions.

### Strengths
The prompt design is clever and the prompts make explicit ideas that are implicit in human thought processes about social roles.

### Weaknesses
It is hard to see what is generalizable here. LLMs are fundamentally linguistic; any higher-order cognitive processes are emergent properties.  The kind of prompt engineering illustrated here yields no insight into those processes. Instead, it is akin to mumbling different incantations at a mysterious creature to see how it responds. In what sense have the LLMs understood or responded to the ReCon prompts? Unclear. Do they have a usable theory of mind about other agents in the game? Unclear. These linguistic prompt manipulations are inherently fragile. They depend enormously on the current capabilities of the specific LLMs studied; there is no good reason to think that these approaches will transfer to other future LLMs with different training sets, architectures, or fine-tuned guardrails.

The use of ChatGPT to perform multi-dimensional evaluation needs to be validated against human evaluation.

### Questions
none

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies whether LLMs can deceive and act strategically in deceptive environments, using the Avalon game as a testbed. The authors propose a method called Recursive Contemplation (ReCon) to enhance LLMs’ ability to identify and counteract deceptive information, based on theory of mind methods. The authors show that ReCon is able to aid LLMs to discern and maneuver around deceptive information.

### Strengths
1. The paper is well-motivated: studying the truthfulness and deceptive abilities of LLMs is certainly very important.
2. The dataset can be a nice contribution.
3. The paper is well-written.

### Weaknesses
1. Mainly GPT-4 is used for evaluation so I'm not sure how reliable the evaluation is - there should be at least reliable human evaluation.
2. There are only 20 games so not sure how reliable how the results are.
3. 4.2 MULTI-DIMENSIONAL EVALUATION is only done with chatgpt so it's not clear how reliable and reproducible the results will continue to be.
4. I also don't really understand 4.2 MULTI-DIMENSIONAL EVALUATION - why are existing metrics from (Li et al., 2023b; Bang
et al., 2023) being used here? Aren't we trying to study the performance of LLMs on avalon game? Again, I believe human evaluation would be the gold standard here.
5. 4.3 QUALITATIVE ANALYSES is quite interesting, but the results are very anecdotal to be reliable. 'ReCon’s Proficiency in Detecting Misinformation' -> this is quite cool and would be a very nice result, but there are only anecdotes to support it and not a complete quantitiative evaluation. Can an actual dataset be collected with misinfo, true info labeled, and ReCon vs other baselines rigorously benchmarked on it? Similarly for 'The efficacy of ReCon in information', another cool finding, but needs much more extensive comparisons and evaluation to be reliable. 'Lai et al., Werewolf among us: Multimodal resources for modeling persuasion behaviors in social deduction games' could be a useful resource here.

### Questions
1. Explain why the evaluation with GPT-4 makes sense and why human eval is not needed, or provide human eval if necessary.
2. Explain 4.2 MULTI-DIMENSIONAL EVALUATION, including why only chatgpt was used or provide results with open-source LMs, explain the evaluation metrics and why human eval is not needed, or provide human eval if necessary.
3. Additional results on 4.3 QUALITATIVE ANALYSES as appropriate, including whether these can be evaluation rigorously.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
