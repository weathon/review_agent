# Training LLM Agents to Empower Humans

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
A truly helpful assistive agent should not only take actions on behalf of a human, but also step out of the way and cede control when there are important decisions to be made. However, current methods for building assistive agents, whether via mimicking expert humans or via RL finetuning on an inferred reward, often encourage agents to complete tasks on their own rather than truly assisting the human attain their objectives. Additionally, these methods often require costly explicit human feedback to provide a training signal. We propose a new approach to tuning assistive language models based on maximizing the human's empowerment, their ability to effect desired changes in the environment. Our empowerment-maximizing method only requires offline text data, providing an unsupervised method for fine-tuning language models to better assist humans. To study the efficacy of our approach, we conducted an 18-person user study comparing our empowerment assistant with a strong baseline. Participants preferred our assistant 78% of the time (p=0.015), with a 31% higher acceptance rate and 38% fewer suggestions. Additionally, we introduce a new environment for evaluating multi-turn code assistance using simulated humans. Using this environment, we show that agents trained with empowerment increase the success rate of a simulated human programmer on challenging coding questions by an average of 192% over an SFT baseline. With this empowerment objective, we provide a framework for useful aligned AI agents at scale using only offline data without the need for any additional human feedback or verifiable rewards.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an LLM decoding method based on [Learning to Assist Humans without Inferring Rewards](https://arxiv.org/pdf/2411.02623), and evaluate it in a synthetic and a user study.

### Strengths
The paper proposes a novel decoding method based on likelihood, and does empirical evaluation of it. The application domain of coding is important.

### Weaknesses
I have concerns with the proposed method and empirical evaluation.
1. The main novelty methodologically over [Learning to Assist Humans without Inferring Rewards](https://arxiv.org/pdf/2411.02623) is their approximation of the mutual information. To me it is quite a coarse approximation to upper bound mutual information by entropy of one of the variables. Fundamentally, empowerment is about some causal influence, not entropy alone. Without the derivation this method looks like a likelihood-based stopping criterion, which does not resemble empowerment.
2. In the simulated human study, the human model is a strictly stronger (in terms of parameters) and the performance gains could just come from the LLM acting as the assistant "staying more passive", and letting the human model do the work. For empowerment, a main question is how the cognitive burden between the model and the human is allocated. If the human does most of the work, then this might lead to better performance. This makes it hard for me to evaluate the evidence.
3. I commend the authors on running a user study, but the main concern from the simulated study (does the model just make the human solve the task) is not addressed: The coding problems that users are shown are quite short, and it might be that the decoding method proposed here leads to humans essentially solving the task.
4. As the method relies on likelihood estimates of the LLM, calibration becomes an issue, which the paper does not adress.

### Questions
- How calibrated are uncertainty estimates of the model?
 - How does your decoding change outputs in terms of length? (Outputs should get shorter if I am getting it correctly, as the end-of-string token will still be output)
 - Can you rule out that the performance gains in the simulated human study comes from the assistant LLM "staying out of the game" / being more passive?
Typo: AH = {{ACCEPT}×L≤KH, {REJECT}×L≤KH, FINISH} doesn't type-check.

### Soundness
2

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
This paper introduces Empower, a new unsupervised fine-tuning method for training language model assistants that aim to empower human users rather than replace them. Instead of optimizing for task completion or explicit feedback, the model learns to maximize the human’s empowerment—the ability of a user’s actions to influence future outcomes. Using a simple logit-threshold algorithm, the assistant learns to generate “obvious” completions (e.g., boilerplate code) while leaving key decisions to the user. In code generation experiments with LiveCodeBench, Empower significantly improves human–model collaboration, achieving higher Pass@1, acceptance rate, and a 192% improvement in simulated human success over supervised fine-tuning baselines. In a 18-participant user study, participants preferred Empower 78% of the time, citing fewer, more useful suggestions.

### Strengths
This paper offers a thoughtful and well-motivated direction for designing more cooperative AI assistants. The central idea—training models to maximize human empowerment rather than simply task completion—is timely. It reframes alignment from “doing the task” to “helping humans do the task better,” which feels like a meaningful step toward more human-centered AI. 

The Empower algorithm itself is simple and scalable, relying only on offline data and language-model likelihoods, which makes it practical compared to methods requiring human feedback or reinforcement learning. Empirically, the results are solid: the method improves both automated coding benchmarks and human–AI collaboration metrics, and the inclusion of an 18-person user study adds credibility beyond simulation.

### Weaknesses
While the paper is well written and empirically convincing, its novelty is somewhat incremental from prior empowerment work in reinforcement learning and assistive robotics. The idea of empowerment—training an agent to maximize the user’s ability to influence the environment—is not new. It has already been explored for many years in reinforcement learning (RL) and human–robot collaboration research. The adaptation to LLMs mainly involves reinterpreting token likelihoods as proxies for empowerment, which may not fully capture the theoretical intent of empowerment-based control. 

Methodologically, the logit-threshold rule feels a bit ad hoc. The paper doesn’t fully explain why this particular thresholding approach should reflect empowerment, or how sensitive the results are to the exact cutoff. It’s an intuitive idea—use token likelihoods to decide where the model should stop generating—but the theoretical link between “high likelihood” and “empowering the human” isn’t very clear. The method works in practice, but it feels more heuristic than principled.

The evaluation, while extensive in the coding domain, is narrowly scoped: all experiments focus on short-horizon code completion, and it's unclear how well the method generalizes to other real-world assistive settings (e.g., writing, planning, dialogue). The simulated “human” setup also relies on large LLMs as stand-ins for humans, which may inflate results.

### Questions
How sensitive are the results to the choice of the logit threshold (η)?

The current setup assumes the base LLM’s likelihood estimates are good proxies for human predictability—does this hold for weaker or domain-shifted models?

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
This paper proposes a new approach to training assistive large language models (LLMs) by maximizing human empowerment. The model is trained to help users reach states where they have greater control and more meaningful choices.
The authors formalize empowerment through mutual information and introduce Empower to select completions whose cumulative likelihood exceeds a threshold.
The proposed approach is empirically validated in both simulated environments (LiveCodeBench) and an 18-participant human user study, demonstrating consistent improvements in performance and user preference.

### Strengths
- Considering LLMs as assistants is an interesting framing. Introducing empowerment as an alignment signal is an interesting and novel idea.
- The method relies only on offline text data, providing a potentially scalable and cost-effective alternative to RLHF or preference-based fine-tuning pipelines.

### Weaknesses
- The environment setup supports only linear text continuation: neither the simulated nor the human users can edit previously written code. This does not reflect how developers actually write programs, where they sometimes design high-level structures first, fill in details later, and revise earlier sections when bugs are found. 
- For the user study, there is no performance comparison with RLHF-trained assistants. Without such baselines, it’s unclear how large the performance gap is and whether the proposed cost savings justify potential losses in performance.

### Questions
- Many existing products (e.g., GitHub Copilot) already perform line-by-line completion with real human users. Have you compared your Empower assistant against such commercial systems, or considered including them in the user study?
- How is the empowerment threshold chosen? Have you analyzed sensitivity to this hyperparameter?

### Soundness
3

### Presentation
2

### Contribution
2
