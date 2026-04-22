# Large Reasoning Models Learn Better Alignment from Flawed Thinking

- Avg Score: 6.50
- Decision: Reject
- Scores: 6, 4, 8, 8

## Abstract
Large reasoning models (LRMs) “think” by generating structured chain-of-thought (CoT) before producing a final answer, yet they still lack the ability to reason critically about safety alignment and are easily biased when a flawed premise is injected into their thought process. We propose **RECAP** (Robust Safety Alignment via Counter-Aligned Prefilling), a principled reinforcement learning (RL) method for post-training that explicitly teaches models to override flawed reasoning trajectories and reroute to safe and helpful responses. RECAP trains on a mixture of synthetically generated counter-aligned CoT prefills and standard prompts, requires no additional training cost or modifications beyond vanilla reinforcement learning from human feedback (RLHF), and substantially improves safety and jailbreak robustness, reduces overrefusal, and preserves core reasoning capability — all while maintaining inference token budget. Extensive analysis shows that RECAP-trained models engage in self-reflection more frequently and remain robust under adaptive attacks, preserving safety even after repeated attempts to override their reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to train LLMs to recover from flawed premises in its chain of thought (CoT). The motivation is safety alignment. If users can initialize the start of a CoT, this paper shows that it can be used to subvert safety alignment. The solution is to extend RL-based alignment training to include these flawed premises and train the model to recover correctly. This method is a variant of DAPO and called RECAP. RECAP uses both premises that attempt to elicit unaligned behavior and over-refusal, so that the resulting model is safer without significantly higher over-refusal rates. The paper presents a theorem that claims that RECAP is superior to DAPO. Experiments show that LLMs from the Qwen and Llama families trained with RECAP are as safety aligned as those trained on DAPO on standard harmful prompts as good or better at avoiding over-refusal, and as capable on math benchmarks, but are significantly better at avoiding responding to CoT prefill attacks and other jailbreaks.

### Strengths
- The motivation of recovering from flawed premises is an important one and this approach might be of significant interest to researchers beyond those working on safety alignment. I am not aware of other work directly addressing this question.

- The paper is clear and well-written.

- The experiments are thorough, including benchmarks that cover the space of tradeoffs among capability, safety, and over-refusal.

- The additional experiments attempting to break this new method and studying how robust it is makes for a nice addition. Rather than wait for follow up work, the paper gives evidence that this method might withstand some further attempts at jailbreaking.

### Weaknesses
It's unclear from the presentation whether the theoretical analysis is significant or just makes the paper more mathy. The advantage appears to be essentially the difference between the expected advantage on pre-filled samples and a "bounded" error term. But that error seems to be only bounded by assumption (Assumption 2). The appendix argues that this is reasonable because RECAP has access to pre-filled training data whereas DAPO has only clean data. It leaves me with the impression that Theorem 1 works because of very strong assumptions that cannot be verified in practice that effectively beg the question. I would happily increase my score if stronger arguments of the significance of Theorem 1 are presented in discussion, or if the entire section is removed from the paper and the claim of "theoretical" evidence is dropped entirely. The paper is strong already and weak, mathy arguments detract from it.

### Questions
Can at least numeric examples of the bound in Theorem 1 be given. What values of the various quantities are needed to make the bound useful, and is there a way to argue that those are reasonable things to hope for? That would also perhaps strengthen the argument.

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
3

### Summary
This paper presents RECAP, which is an RL–based post-training method designed to improve the safety alignment of LRMs. Recognizing that LRMs are vulnerable to flawed reasoning when exposed to adversarial or biased prompts, RECAP explicitly teaches models to override unsafe reasoning trajectories and reroute toward helpful and aligned responses. The method operates by injecting CoT prefills during training, alongside standard prompts. Empirical results show that RECAP significantly improves safety and jailbreak robustness, reduces overrefusal, and preserves core reasoning capabilities, all while maintaining inference-time efficiency.

### Strengths
(1) Timely and important problem: The paper addresses a highly relevant and impactful problem: safety alignment of LRMs, which is crucial for the broader and responsible deployment of LLMs.

(2) Clarity of presentation: The method is clearly presented and well-articulated, making it easy for readers to follow the core ideas and technical contributions.

### Weaknesses
(1) Strong assumptions in theoretical analysis: While the authors provide theoretical insights to support their algorithm, the main theorem (Theorem 1) relies on strong assumptions. Moreover, the tightness and practical implications of the theoretical bounds remain unclear. Please refer to the corresponding questions for further elaboration. 

(2) Concerns regarding evaluation results: There are some concerns about the evaluation setup and reported results, particularly for the math reasoning task. These concerns could be addressed by responding to the specific questions outlined in the question section.

### Questions
(1) In Section 2.2, could you provide a more concrete or practical example illustrating the scenario: “What happens if a model is forced to continue from another model’s reasoning trace?”

(2) Could you provide an estimation method or approximate values for the nonnegative slack terms \epsilon^A(t) and the clean parity term \xi(t)? These terms are crucial since they directly affect the tightness of the claimed theorem. Presenting estimation methods and representative values would help readers better understand the theoretical insights.

(3) In line 239, you state that this term is typically close to zero because DAPO is not exposed to these states. Does this imply that you assume the model trained under vanilla DAPO will never generate unsafe prefills and then recover from them?

(4) The results are intriguing: RECAP even improves mathematical reasoning capability compared to vanilla DAPO, though the method does not seem to include a specific design for enhancing general reasoning ability. Could the authors also report the standard deviation of the math evaluation results?

(5) Ablation study request: Given the observations from the math task, I recommend adding ablation studies to clarify the contribution of each component:
 (a) Training solely on the 3 K dataset versus solely on the 2 K harmful + over-refusal prompts.
 (b) Re-running experiments with another base model (e.g., LLaMA-8B) to mitigate potential data contamination in DSLLaMA-8B and DSQwen-14B.

### Soundness
2

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
4

### Summary
This paper introduces RECAP (Robust Safety Alignment via Counter-Aligned Prefilling), a reinforcement learning method for training large reasoning models (LRMs) to override flawed CoT trajectories. The approach prefills harmful prompts with unsafe reasoning and benign prompts with overly conservative reasoning during training, forcing models to recover appropriate responses to achieve high rewards. Through experiments on DSLlama-8B and DSQwen-14B, the authors demonstrate improvements in safety, jailbreak robustness, and helpfulness while maintaining math reasoning performance and similar inference-time token budgets.

### Strengths
This paper addresses a critical and timely vulnerability in large reasoning models where simple CoT prefilling can bypass safety alignment, with clear demonstrations showing that LRMs exhibit brittle reasoning when seeded with flawed traces.  The approach is easy to adopt, requiring no additional training cost beyond standard RLHF, and the bidirectional application of the core idea (addressing both overrefusal and safety simultaneously) demonstrates the generality of the method.

The experimental methodology is comprehensive and rigorous. The evaluation spans multiple critical dimensions including direct harmful prompts, sophisticated jailbreaking scenarios (WildJailbreak, Fortress), overrefusal, and math reasoning, using high-quality benchmarks with expert-crafted prompts and instance-specific rubrics. The ablation studies provide valuable practical insights into design choices (prefilling ratio, length, and source), with the key finding that counter-aligned prefills are essential—aligned prefills actually harm performance, confirming the method works through corrective supervision rather than simple exposure. The stress-testing with adaptive attacks (full CoT hijacking and iterative prefill reset) goes beyond standard evaluation to demonstrate that RECAP yields persistent robustness even when adversaries repeatedly attempt to override the model's reasoning, and the behavioral analysis showing increased self-reflection frequency (83.4% vs 59.7%) provides insight into how the method changes model reasoning dynamics.

### Weaknesses
I don't see major flaws with the paper. Here are some minor weaknesses with the work.

First, the theoretical analysis in Section 3.2 and Theorem 1 provides limited insight into why RECAP works. The assumptions (particularly Assumption 3 about DAPO's bounded progress on prefilled samples) essentially encode the conclusion into the premises. The theorem states that RECAP gains advantage "precisely from training on prefilled samples" (line 232), but this is tautological—of course a method trained on prefilled data performs better on prefilled evaluation. The analysis would be stronger if it explained why exposure to counter-aligned reasoning during training leads to more robust safety alignment, beyond simply noting that RECAP sees states that DAPO doesn't.

Second, Section 5.2 claims RECAP-trained models "engage in self-reflection far more often than vanilla RLHF" (line 428), with 83.4% vs 59.7% on StrongReject-Prefill. However, this comparison may conflate two distinct phenomena: (1) models learning to reflect more deeply during reasoning, versus (2) models detecting and correcting prefilled flawed reasoning. The paper doesn't separate these cases. If most self-reflection on safety alignment occurs only when unsafe prefills are present, this would suggest RECAP primarily teaches pattern-matching to detect injected reasoning rather than fundamentally improving reflective reasoning capabilities.

### Questions
- The ablation in Figure 3c shows that prefilling with "aligned traces" (safe reasoning from STAR-1) significantly underperforms no prefilling. Can you explain why safe prefilling harms performance? Is this because models learn to overfit to the prefilled reasoning without developing their own safety reasoning capabilities?
- Section 5.2 reports self-reflection frequencies but doesn't analyze when in the CoT this reflection occurs. Does reflection happen primarily in the first few tokens after a flawed prefill, or is it distributed throughout the reasoning? Understanding this distribution would clarify whether RECAP teaches genuine reflective reasoning or primarily pattern-matching for injected text.
- How does RECAP perform when the flawed prefills come from the policy model itself (self-generated unsafe reasoning) rather than from external weaker models? This would test whether the method relies on distribution mismatch between the policy and prefill source versus truly learning to override flawed reasoning regardless of its origin.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces an RL-based training method designed to improve safety alignment in reasoning models. The method, called RECAP, deliberately trains models on flawed, counter-aligned CoT prefills: harmful prompts are prefilled with unsafe reasoning, and benign prompts with refusal reasoning. This forces models to learn to override flawed trajectories during training. The authors show that RECAP achieves improvements across harmful benchmarks and helpfulness on benign reasoning tasks. Theoretically, the authors prove RECAP achieves higher expected reward than vanilla DAPO under both clean and prefilled evaluation conditions.

### Strengths
- This paper proposes a simple and effective training method to address misalignment in reasoning models. The method simply mix counter-aligned prefills into standard RLHF training, which is immediately deployable and has real-world significance. 
- The method achieves multiple objectives including safety, helpfulness, and reasoning. 
- The paper provides comprehensive experiments and stress testing results to show that RECAP is robust to adversarial attacks. It also provides behavioral analysis showing models engage in self-reflection more often after RECAP. 
- The paper is well-written and easy to follow. The theoretical justification is sound.

### Weaknesses
1. The method depends on the source / quality of synthetic prefills. It is unclear how well the collected prefills dataset generalize to diverse settings in the real-world. How sensitive are the results to the quality / source of the prefills?  
2. Minor issue, but the evaluation of safety could be inflated by using the GPT-4o model as a judge. How aligned is GPT-4o as a judge with human annotators?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
