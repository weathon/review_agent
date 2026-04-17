# Do Large Language Models Know What They Are Capable Of?

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
We investigate whether large language models (LLMs) can predict whether they will succeed on a given task and whether their predictions improve as they progress through multi-step tasks. We also investigate whether LLMs can learn from in-context experiences to make better decisions about whether to pursue a task in scenarios where failure is costly. All LLMs we tested are overconfident, but most predict their success with better-than-random discriminatory power. We find that newer and larger LLMs generally do not have greater discriminatory power, though Claude models do show such a trend. On multi-step agentic tasks, the overconfidence of several frontier LLMs worsens as they progress through the tasks, and reasoning LLMs perform comparably to or worse than non-reasoning LLMs. With in-context experiences of failure, some but not all LLMs reduce their overconfidence leading to significantly improved decision making, while others do not. Interestingly, all LLMs’ decisions are approximately rational given their estimated probabilities of success, yet their overly-optimistic estimates result in poor decision making. These results suggest that current LLM agents are hindered by their lack of awareness of their own capabilities. We discuss the implications of LLMs' awareness of their capabilities for AI misuse and misalignment risks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the self-awareness of large language models (LLMs), defined as their ability to predict their own success on a given task. The authors conduct three experiments to evaluate this capability: (1) eliciting in-advance confidence estimates on single-step tasks; (2) placing LLMs in a sequential resource-acquisition scenario where they must accept or decline "work contracts" based on their self-assessment, learning from in-context experience; and (3) tracking how confidence estimates evolve at each intermediate step on multi-step tasks.

The core findings are that: LLMs typically do not make more accurate confidence estimates; frontier LLMs successfully learn from past successes and failures  by increasing their risk aversion; and that reasoning LLMs are typically less accurate at predicting their success.

### Strengths
1. The paper tackles the issue of LLM self-awareness, which has significant implications. Understanding whether models "know what they don't know" is fundamental in many AI areas.

2. The evaluation is extensive, covering a wide range of modern LLMs from different families (Llama, GPT, Claude).

3. The paper confirms interesting LLMs behaviours, although not completely unexpected. For example,  in Experiment 2 that improved profitability stems from increased risk aversion

### Weaknesses
1. The central conclusion that LLMs are overconfident is not new, and is extensively documented in prior work, much of which is cited by the authors (e.g., Lin et al., 2022; Tian et al., 2023; Xiong et al., 2024). While the experimental setups are novel, the main takeaway is confirmatory rather than groundbreaking.

2. A significant opportunity seems to have been missed by not exploring methods to address the identified problem. The framework from Experiment 2, which measures expected profit, seems perfectly suited to use as a reward signal. Why did the authors not attempt to use this signal to improve model performance, for instance, through RL fine-tuning? A demonstration of even a simple mitigation strategy on a small-scale LLM would have substantially increased the paper's contribution.

### Questions
- The finding in Experiment 3 that reasoning-enabled models perform no better than their non-reasoning counterparts is surprising. Do the authors have a hypothesis for this? Could it be an artifact of the prompting strategy, the RLHF training, or does it suggest that current reasoning capabilities do not extend to robust self-assessment?

- The paper provides a strong foundation for future work. I would encourage the authors to build on this evaluation by exploring mitigation techniques, as this would be a highly valuable contribution.

### Soundness
3

### Presentation
3

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
The work is a quantitative analysis of the in-advance capability of a language model in determine success/failure outcome of a task. In particular, it studies whether the models can learn from recent successes/failures to make better go/no-go decisions when failure is costly, and update those predictions while working through multi-step, tool-using tasks.

The authors ran three experiments:
* Predicting the probability of success before attempting BigCodeBench Python coding tasks
* A modified setting where models sequentially attempt Python tasks in a “work contracts" setting that resembles a contextual one-armed bandit with an abstain option
* Stepwise confidence on multi-step SWE-Bench Verified, with a 70-tool-call budget

The work then presents several core findings: all models overestimate their success rates, they have better than random guesses, but their guesses have no trend of increasing discriminatory power with capability. Even with in-context learning, many models remain overconfident. Some frontier models improve profit in the bandit setting mainly by abstaining more, not by sharply improving discrimination/calibration. Finally, multi-step dynamics diverge between different models, with reasoning models not necessarily more calibrated than non-reasoning variants.

### Strengths
* The paper is very well written with great flow, good schematics, and no major grammatical or formatting errors
* The paper crystallizes an well-known phenomenon intuitively and anecdotally understood into a rigorous analysis
* Calibration is a very important problem to study and understand

### Weaknesses
* Unfortunately the world of foundation models move incredibly fast and the models tested in this work is already fairly dated. For example, GPT-4.1, Sonnet 3.5, etc. are no longer available. As noted by the authors, some of the behaviors characterized are divergent, and thus likely no longer relevant to the newest class of models (e.g. GPT-5). 
* It would benefit the audience to draw connections between the in-advance setup of the work to calibration in traditional neural network architecture (e.g. [1], among others)

[1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, July). On calibration of modern neural networks. In International conference on machine learning (pp. 1321-1330). PMLR.

### Questions
I don't have more questions beyond what is mentioned in the weakness section.

### Soundness
3

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
This paper investigates whether LLMs can predict whether (and how likely it is that) they can complete a task before performing it. This is tested both for single response cases using the BigCodeBench benchmark and in an agentic setup using the SWE-Bench verified benchmark. The authors find that all of the investigated models (recent GPT, Claude, and LLama models) overestimate the likelihood of completing a task successfully, and while some models can be steered through in-context learning to be more cautious about predicting that they can complete a task, this behaviour generally even persists in that scenario.

### Strengths
* While the research question is relatively simple, the paper evaluates the question thoroughly and considers the problem from multiple angles. And to the best of my knowledge, this paper introduces some new paradigms to evaluate model confidence.
* The paper is very well written, the experimental setup is very clear,  and it connects very well to previous work.
* The paper adequately discusses its limitations.

### Weaknesses
* While I think the task in Experiment 2 with fictional costs is an interesting approach, I was wondering whether this has been validated that LLMs can make such risk-based decisions when they have direct access to the underlying risks. In other words, is there evidence that LLMs can accurately compute the expected reward and base decisions on this implicit calculation? While I don't think this would change results here, since the confidence estimates still seem to be inaccurate anyways, it would be good to establish whether all LLMs can actually do this task since otherwise it may be challenging to derive anything meaningful from the model's decisions in this setup.

### Questions
See the weakness above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work investigates whether LLMs know how likely they will succeed in a given task before attempting the task. The authors conducted three experiments in which LLMs performed confidence estimation in different settings and scenarios. Through the experiments, the study find that LLMs are generally overconfident to the given tasks, but still be able to identify tasks that they are capable of better-than-random. The work also shows that for many LLMs, such 'self-awareness' effect doesn't improve with models' increasing general ability, in-context experiences, and reasoning ability. Overall, the work reveals LLMs' poor self-awareness of capability, which may potentially hinder LLMs' application in high-stakes scenarios.

### Strengths
1. Comprehensive experimental design and clear visualization of the results.
2. Provide empirical evidence of 'self-awareness of ability' across different LLMs with different sizes.

### Weaknesses
1. Need an explicitly defined research gap, motivation, and why the study can fill the gap. The contributions of this work have not been highlighted from previous work.
2. How LLMs know what they are capable of is quantified by self-reported confidence in this study. Auto-regressive LLM models trained by next-token-prediction solve the confidence estimation given tasks in a fundamentally different manner from humans, making it sort of ambiguous to represent 'real confidence' in the tasks merely by reported scores. This could limit the implications of the study.

### Questions
1. What's the difference between in-advance/answer-free confidence estimation and retrospective estimation? Why use such ways of confidence estimation? Should the formers work better in this case? 
2. How is actual success rate in Exp1 calculated? Doing each task for multiple times which produces a success rate for each individual task, or doing each task once that produces an overall success rate for the whole benchmark? 
3. In Exp1, what is the best accuracy of predicting actual success rate with the reported confidence? And worth checking whether rephrasing the confidence estimation prompts can improve the prediction of actual success rate. For example, if asking LLMs to rate in-advance difficulty of each task, rather than the confidence, will the difficulty rating predict success rate well? 
4. In Exp2, how many unique tasks are there in the pool of S and F tasks? Will the repetition of these tasks in the 9-contract sequences affect the results?

### Soundness
2

### Presentation
2

### Contribution
1
