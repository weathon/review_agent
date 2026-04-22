# LEARNING TO GENERATE FORMALLY VERIFIABLE STEP-BY-STEP LOGIC REASONING VIA STRUCTURED FORMAL INTERMEDIARIES

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Large language models (LLMs) have recently demonstrated impressive performance on complex, multi-step reasoning tasks, especially when post-trained with outcome-rewarded reinforcement learning. However, it has been observed that outcome rewards often overlook flawed intermediate steps, leading to unreliable reasoning steps —even when final answers are correct. To address this unreliable reasoning steps, we propose ProSFI (Process Reward over Structured Formal Intermediates), a novel reward method that enhances reasoning reliability without compromising accuracy. Instead of generating formal proofs directly, which a modest-size (7B) model can rarely do successfully, the model outputs structured intermediate steps aligned with its natural language reasoning. Each step is then verified by a formal prover. Only fully validated reasoning chains receive high rewards. The integration of formal verification guides the model towards generating step-by-step machine-checkable proofs and hence yields more credible final answers. ProSFI offers a simple and effective approach to training trustworthy reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors identify that purely outcome-trained LLM reasoning may be correct but logically unsound.

Authors identify challenges with existing approaches to augmenting LLMs with formal verification
1. models excelling at NL reasoning struggle with formal code generation
2. models excelling at formal code generation struggle with NL inputs

They devise a new framework that prompts the model to generate structured logical information for each reasoning step that can be converted to formal propositions and verified externally. This reduces complexity on the LM generation side and proof construction side (both deal with decomposed problems).

From here, the authors use verification score to provide more granular reward to the model in RL and demonstrate improvements in reasoning soundness on logic tasks and knights and knaves puzzles.

### Strengths
- thorough baseline comparisons with direct proof generation and a RL baseline (outcome only)
- method is well motivated
- solid results in improving reasoning soundness
- compatible with test time scaling with DTV
- proposed reward correlates strongly with soundness

### Weaknesses
- negative impact to accuracy and OOD generalization in accuracy: exchanged for substantial boosts to reasoning soundness, but in OOD settings, the correlation between soundness and accuracy is low, so unclear if the trade off is valuable in this situation where you hope to use soundness as some proxy (per DTV).
- does not evaluate against less symbolic tasks (presumably because evaluation is more difficult, but it would be very interesting to see if this training benefits more realistic tasks)

### Questions
- Is there an interpretation/hypothesis for why PRosFI's accuracy drop from in-domain to out of domain is steeper than the outcome method? I had expected sound reasoning to be more transferable in new settings compared to the potentially incorrect shortcuts learned in outcome-based RL.
- Was there experimentation with penalizing the model more for the correct answer/incorrect reasoning case? It seems some positive reward is still rendered. If the goal is to optimize correctness, does this run slightly counter? Or is this for stability reasons?

### Soundness
3

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
This paper proposes PRoSFI (Process Reward over Structured Formal Intermediates), which is a method for fine-tuning LLMs on logical reasoning tasks. The core idea is to train the LLM to output intermediate reasoning steps in a structured formal language in addition to the final answer (another important observation is that outputting intermediate steps in this manner is easier than generating full formal proofs directly, which almost never succeeds for weaker LMs). These formal intermediate steps are then checked by a deterministic prover. The model is trained with GRPO, where it receives positive "stepwise formal rewards" for each intermediate step that is formally verified.

### Strengths
Flawed or unsound reasoning in LLMs is still an critical issue. The goal of improving the verifiability of intermediate reasoning is important. Towards this end, the approach taken by the paper is reasonable: using structured formal intermediate steps as a bridge (both to improve step by step verification and help weaker LLMs that are unable to produce full proofs that actually align with the CoT reasoning) is a logical and sound design choice for the considered tasks.

### Weaknesses
Using "stepwise formal rewards" is presented as a new paradigm, but it is not: this appears to be a clear instantiation of a process reward model, which is well-established in the alignment literature. Here the process reward is simply the binary signal from the deterministic formal verifier run on each reasoning step. While this is a reasonable reward and approach, the paper is not properly positioned relative to this existing literature. 

I also find the empirical results to be a bit unconvincing. They consistently show that PRoSFI is less accurate than the Outcome-CoT baseline. While the "GPT Soundness" metric does increase, the merit of this trade-off is unclear and not fully justified.

### Questions
The "reward hit rate" of the direct Lean generation is really poor, which is a primary justification for your intermediate approach. This seems true for smaller models, but not so for larger ones. Do you expect the benefits of PRoSFI to scale?

Regarding the Test-Time Scaling (DTV) Comparison: In Section 5.5, you demonstrate that PRoSFI scales well with test-time verification (DTV) while Outcome-CoT does not. You attribute this to Outcome-CoT "lack[ing] the structured outputs required" for formal verification. This seems to preclude the possibility of using other types of verifiers. Why was a different verification method not considered for the Outcome-CoT baseline, such as a separate, learned verifier trained on its natural language reasoning steps (as is common in process supervision literature)?

Re generalizatio: the paper claim "strong logical consistency and reliable generalization" on the Knights and Knaves dataset. However, Table 3 shows that PRoSFI is less accurate than the Outcome-CoT baseline on the primary OOD split (85.31% vs. 89.75%). This mirrors the results from ProverQA. Can you comment on what leads to this drop in performance despite the more reliable generation?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The goal of this paper is to make the intermediate reasoning more reliable in long CoT’s, by using process supervision. Each intermediate step (in the natural language solution) is verified by a formal prover. However, writing formal proofs is not easy even for models that are good in natural language - the paper finds that it is very difficult to generate Lean formalizations that are aligned with the natural language intermediate steps. The ProSFI algorithm, introduced in this paper, introduces another reward function which is based on verification of the intermediate steps. For each intermediate step output by the LM, the model also outputs a JSON/YAML dictionary, which can be converted into a problem that a first-order logic verifier can solve. 

The main experimental setting is as follows. They train and evaluate on ProverQA - a baseline is Qwen2.5-7B-Instruct trained only with an outcome/format reward, similar to Deepseek R1. They evaluate the “soundness” of the responses, i.e. how correct the intermediate responses are, using GPT-OSS-120B, and find that it is low for the outcome reward baseline. On the other hand, ProSFI improves the soundness significantly from the outcome-reward baseline, while preserving the correctness. Additionally, the ProSFI reward is highly correlated with the soundness as evaluated by GPT-OSS-120B (see Figure 3).

The other setting is the Knights and Knaves dataset, previously used in Logic-RL. In this work, both the outcome reward baseline and ProSFI are trained on problems with 3-7 people and tested on problems with 2-8 people. The soundness increases with ProSFI compared to the outcome reward baseline, while the accuracy is mostly similar.

### Strengths
1. The method for using process supervision is very interesting - they use JSON dictionaries to get around the limitations of small LMs when generating formal proofs.
2. The main results on ProverQA seem good.

### Weaknesses
1. The main results of this paper are for synthetic first-order logic tasks. There is a chance that there are additional challenges when trying to generalize to more challenging mathematical tasks.
    1. For example, the structure of the intermediate representations seems to rely on the nature of the task (which is first-order logic), because the intermediate dict specifies the logical rule.
    2. How can this be generalized to other mathematical settings? Will it still be easy to break up into atomic steps?
2. There is still a possibility that there is a tradeoff between rewarding for correctness and rewarding for soundness. This does not seem to be the case in the ProverQA setting, but in the OOD testing setting for Knights and Knaves, with 8 people, the accuracy is worse for ProSFI compared to the outcome reward method.

### Questions
1. Why do you believe the soundness is lower on ProverQA Extra, even with ProSFI?
2. Can it ever be the case that some steps are easy to reason about in natural language, but difficult to prove/verify formally? Have you encountered this?

### Soundness
3

### Presentation
4

### Contribution
3
