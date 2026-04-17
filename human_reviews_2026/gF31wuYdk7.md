# Measuring and Mitigating Rapport Bias of Large Language Models under Multi-Agent Social Interactions

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. While prior work has largely focused on conformity bias, we broaden the scope to examine how LLMs build rapport from previous interactions, resist misinformation, and integrate peer input during collaboration, which are key factors for achieving collective intelligence under complex social dynamics. We introduce KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert–novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how rapport, peer action, and self-confidence influence decisions. To mitigate this vulnerability, we evaluate prompting, supervised fine-tuning, and reinforcement learning using Group Relative Policy Optimization (GRPO) across multiple models. Our results show that model size plays a central role in moderating susceptibility to social influence: larger models exhibit stronger resilience and benefit from prompting-based mitigation, whereas smaller models are more vulnerable. For the latter, carefully configured GRPO training improves both robustness and overall performance. Our code and datasets are available at: https://anonymous.4open.science/r/KAIROS-4F71

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes KAIROS, a quiz-style benchmark designed to evaluate LLMs’ behavior under varying conditions of rapport (i.e., historical peer consistency), peer influence, and self-confidence. The authors investigate how these factors affect the robustness of LLMs and assess three technologies (i.e., prompt engineering, SFT and GRPO) in mitigating performance degradation under social stress and overconfidence. Experiments show that GRPO with peer interaction history and outcome-based reward significantly outperforms the other two technologies in maintaining robustness against social perturbations.

### Strengths
1) The benchmark encompasses four categories (i.e., reasoning, knowledge, social, and creativity) and provides fine-grained control over factors such as rapport and peer influence.

2) There is a comprehensive evaluation across model sizes (i.e., from 3B to 120B) and architectures (i.e., Qwen and Llama).

### Weaknesses
This work seems to an extension of the BENCHFORM[1] with many similar settings, such as multiple-choice question-answering format and evaluation metrics. Here I list some of my concerns.

1) The tasks in KAIROS are limited to a multiple-choice question-answering format, which may not fully capture the open-ended or interactive reasoning abilities of LLMs under social contexts. This design seems to over simplify the scenarios where the LLMs would be applied.

2) The design of the peer agent rapport level is simple. For instance, a 25% rapport level merely indicates that all peer agents provide the same answer as the evaluated agent in one of the four prior rounds. This coarse-grained design may fail to reflect the nuanced dynamics of the real interpersonal rapport.

3) The peer-interaction histories are purely symbolic where peer agents only present options without offering rationales or engaging in multi-turn exchanges. It is unclear weather LLMs interpreter these histories as social signal rather than as irrelevant contextual noise. When performing a small-scale replication using the released prompts,  Qwen2.5-3B often dismisses the histories (e.g., ” I do not have enough context or previous Q&A history to determine the correct answer for the problem”) while larger models such as GPT-5 and GPT-oss:120B often ignore the histories without mentioning it, suggesting limited sensitivity to the social stress. 

4) In Line 144, the measure of self-confidence is too absolute: Samples are categorized into high or low confidence by comparing their entropy to the dataset-wise median. This approach enforces a 1:1 split regardless of overall uncertainty distribution. It might be unfair when the overall entropy is high and consequently biases the GRPO training.

5) In Line 186, it seems that the oppose-hard agents and oppose-easy might present a same answer when the correct choice with the highest model-predicted probability.

6) The evaluation does not explore the effect of varying the number of peer agents. Although the size of the social group is randomly sampled from 3 to 6, it lacks a direct experiment on this factor which reflect the intensity of social influence.

7) In Line 318, the authors claim that the training and test sets are form disjoint sources; however, both datasets contain samples derived from BBH and LiveCodeBench. The explanation in Line 635 seems unhelpful for this potential data leakage. 

8) The paper does not include any qualitative case studies to illustrate typical interaction patterns or behavior changes in the models.

[1] Weng, Zhiyuan, Guikun Chen, and Wenguan Wang. "Do as we do, not as you think: the conformity of large language models." arXiv preprint arXiv:2501.13381 (2025).

### Questions
1) In human society, individuals with high self-confidence may still experience self-doubt when consistently opposed to their peers. In constructing simulated interaction history, are the self-confidence levels of the history questions comparable to the current question? Is an LLM with high-confidence on history question changes its correct answer due to the self-doubt? 

2) The hyperparameter $T$ and dropout rate for Monte Carlo estimation are not clearly specified in the paper while the self-confidence estimation might be highly sensitive to them. Can you present additional details or analysis of how these influence the confidence estimates and subsequent results?

3) As mentioned in the Weaknesses section above, the historical peer interactions presented to the LLMs consist solely of answer options, without explicit rationale or dialogue. Is there evidence that the evaluated LLMs interpret these histories as social cues rather than as irrelevant context? Can you provide some cases illustrating how the presence of history influences model responses?

4) The comparison between prompt engineering and GRPO uses completely different system prompt. For example, debate-style prompts can also be used in prompt engineering. Is current comparison a fair judgement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces KAIROS, a benchmark to study how large language models behave under social influence in multi-agent interactions. It shows that LLMs often change correct answers when pressured by peers, especially smaller models. The authors propose mitigation methods, finding that Group Relative Policy Optimization (GRPO) best improves robustness. The work highlights rapport bias as a key vulnerability in socially interactive AI systems.

### Strengths
The authors propose the novel benchmark KAIROS, which evaluates the performance of LLMs under various social contexts. KAIROS is dynamically constructed, where the different evaluation scenarios are based on the responses of the LLM to be evaluated, such as
generating peer agent responses to support or oppose the target LLM’s response. They performed extensive experiments by utilizing multiple training and prompting strategies to test different model architectures using their benchmark. The authors also formulated different training strategies to train socially robust LLMs.

### Weaknesses
The benchmark only quantifies the effect of social influence on LLM responses, but there is no solution to target this issue. Except for a handful outliers, the O-K gap is exacerbated by the different training strategies. The desired outcome is for the gap to be close to zero, which indicates that it can perform well both in isolation and in a social context.

### Questions
The paper only explored multiple-choice questions, but it is easier for the model to answer correctly, as it is just selecting an option versus generating the correct response directly. Then, what would happen if open-ended questions were used? Also, did the authors test what would happen if the answer choices didn’t contain the correct answer and the peer models selected a random one? Would it be able to recognize that the correct answer isn’t an option, or be forced to select one due to peer influence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores how LLMs build and respond to social rapport in multi-agent systems. The authors introduce KAIROS, a benchmark that simulates quiz-style interactions with peers of varying reliability to study rapport bias, or the tendency to trust agreeable partners. They find that larger models resist social influence better, while smaller ones are easily swayed. Among mitigation methods, GRPO with outcome-based rewards and social context offers the best balance of accuracy and robustness. The study also highlights that real progress in multi-agent AI requires improving robustness to social influence, not just raw accuracy.

### Strengths
•	It is an interesting research topic and the KAIROs benchmark can be reused by future study on llm-based MAS.

•	The paper is well structured and the presentation of figures are clear.

•	Section 2 is well established and the design choice is properly justified.

•	The model scope is comprehensive and the inclusion of both SFT and GRPO adds depth to the study.

### Weaknesses
•	Section 2.1 The notations should be clearer. For example, is “y” a single token, or a sequence of generated tokens? If that is a sequence of tokens, how do you compute the predictive distribution? If y is a single token, how does it account responses longer than one token?

•	Section 3.1.3. Given the central role of the reward function, you should elaborate more on both OR and DR with formal definitions in the main body of the paper. The current descriptions are vague.

•	Table 1. You should clarify, for the last two rows, what you are reporting with “± 4.2%” - whether that is standard error, confidence interval, or the range.

•	Section 4.1.  Given the random noise in all the accuracy measures, for all the claims you are making, you should conduct statistical tests to see if the differences are indeed significant. For example, you should test if the improvement in KAIRO accuracy (2.48%) is indeed statistically significant.

•	Table 2. GRPO sometimes causes the model to perform worse than the base condition. This seems to imply that the GRPO is not performing well and there may be issues in the training pipeline.

•	Table 2. You should clarify which row each GRPO/SFT checkpoint we should be comparing against. In other words., which is the base model for GRPO/SFT. This helps readers to judge whether SFT/GRPO indeed improve the accuracy and robustness.

•	Section 4.2. Line 412: “we find that while various training strategies significantly improve both original and protocol accuracy”- it’s your first time using the term “protocol accuracy”. You should properly define it. The same applies to the term “protocol settings”.

•	Section 4.3. This paragraph is very dense and it almost feels like a result dump. It is hard to follow with reading through the entire Appendix D. I think you want to either remove it entirely, or elaborate the setup in length.

### Questions
•	Section 2.1. Why does “oppose-easy” agents generate incorrect answers when the model’s answer is correct, but generates correct answers when the model’s answer is incorrect? What is the underlying rationale? Furthermore, you should clarify the naming convention of the terms “oppose-hard” and “oppose-easy”.

•	Section 2.3. “Large O–K∆ values suggest that an agent correct in isolation may reverse its answer when exposed to peers, undermining system reliability” -> Shouldn’t it be the other way around – large O-K ∆ values mean that the agent is more likely to be correct when exposed to peers, compared to in isolation?

•	Section 4.1 Can you clarify how you derive the value “Original (↑4.93%) and KAIROS (↑1.28%)” and “empowerment improves KAIROS accuracy (↑2.48%)”? Are these values averaged across all models?


•	Does larger O–K∆ mean higher robustness? Or does O–K∆ being closer to zero means higher robustness? The writing in section 2.3 seems to imply the latter, while in section r seems to imply the latter.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies rapport bias—how LLM agents adapt their decisions based on prior interactions and contemporaneous peer signals—in multi-agent systems (MAS). The authors introduce KAIROS, a quiz-style benchmark that varies peer reliability, roles (expert/novice), crowd noise, and adversarial behavior. They define four evaluation dimensions: accuracy, utility (wrong→correct via peers), resistance (correct→correct under pressure), and robustness (accuracy drop from solo to social setting). Mitigation experiments compare prompting, supervised fine-tuning (SFT), and Group Relative Policy Optimization (GRPO) with several reward/prompt schemes and MAS vs non-MAS training contexts. Key findings: (i) model scale improves resilience; (ii) GRPO substantially improves accuracy (solo and KAIROS) but can reduce robustness unless trained with MAS context; (iii) the NS-OR (normal system + outcome reward) configuration gives the best accuracy/robustness trade-off; (iv) confidence-based data filtering boosts accuracy but often harms robustness. The work reframes social bias beyond conformity to include rapport dynamics and provides a controlled setting to measure and (partially) mitigate these effects.

### Strengths
1 Timely problem: Moves beyond single-agent conformity to richer social dynamics (rapport, peer influence, revision under pressure), which is central to practical MAS deployments.  
2 Benchmark design: KAIROS offers fine-grained control over social variables (peer reliability, role asymmetry, adversaries) enabling causal-ish analysis of factors driving susceptibility.  
3 Demonstrates that “better accuracy” can mask fragility under social interference

### Weaknesses
1 It’s unclear how questions are sourced, balanced by domain/difficulty, and de-biased.  
2  The evaluation metrics, such as robutness and utility, are mainly based on the accuracy of the answers from the MAS. However, the accuracy is only one aspect required by users. Other aspects like safety, interpretability, and faithfulness also are important.  
3 The authors introduced a new benchmark, however, the benchmark is composed of multiple existing datasets.

### Questions
Do you use all the datasets mentioned in their original form to build the benchmark? Are the samples fairly distributed?

### Soundness
3

### Presentation
2

### Contribution
2
