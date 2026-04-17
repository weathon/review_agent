# Learning to Lie: Adversarial Attacks on Human-AI Teams and LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models---one inspired by literature and the other data-driven---and find that both can effectively harm the human team. Moreover, we find that in this setting while our data-driven model is the most capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions, the model based on principles of cognitive psychology does not lag too far behind. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper conduct a human study on the trust of humans to AIs, and how can AI potentially cheat human. The authors explores psychology-based and neural-network based belief dynamic modelling, and find both methods can effectively harm the human team. Finally, the authors replace human with LLMs and verifies its vulnerability to attacks.

### Strengths
The paper examines how can AI cheat human teams, an important and timely topic. The experiments are insightful and reveals important safety risks. The replacement of human to AIs and find advanced AI models cannot oversight cheating is a plus.

### Weaknesses
1. Clarity. While the experiments are solid, the takeaways and insights for practitioners are not immediately clear. Maybe summarization, highlight and mentioning potential application helps.

2. In the experiment, the authors claims that human will not trust AIs if it is "incorrect on an easy question". However, in subsequent analysis like Section 5.2, the strategic attack on whether the question is hard or not is not mentioned. How is the underlying process of deciding to lie or not? How is the lying answer given? If an wrong answer is given by AI in hard questions, maybe the trust by humans will decrease higher? I wonder if there are such analysis given.

3. Also in Section 5.2, the baseline compares cognitive model, MLP model and no attack. However, in the case without attack, the information provided by AI is a plus to the overall performance. So, I wonder if another baseline is needed for comparing with AI not providing any information at all.

4. Cognitive model seems uneffectivve in predicting the error comparing with equal weights baseline. Are there further experiments to verify the effectiveness of cognitive model? Current version seems highly ineffective.

I apologize that while I work on alignment and conduct experiments on human subjects, I am not familiar with existing works so please clarify if there is any mistake.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose an RL-based framework to control when an AI assistant should behave adversarially versus helpfully while collaborating with a group of three human participants in a sequential decision-making task. The AI attacker first behaves correctly to gain trust, and later selectively provides wrong answers to reduce the team’s overall performance based on the mdp planner. The authors later explore whether LLM agents can replicate the same patterns observed in humans under the attack scenario.

### Strengths
1. The problem of strategically harmful AI advice in human–AI teams is timely and relevant.
2. The task is small and interpretable, so it is easy to see how the attack works.
3. Comparing a psychology-based trust model and an end-to-end learned trust model  is  a reasonable way to study human influence modeling in the designed experiment.

### Weaknesses
1. Limited novelty over prior work [1]. The prior work (IJCAI'23) that already models human–AI interaction as an MDP and uses RL to strategically decide when the AI should behave adversarially to reduce human performance. This paper follows the same paradigm to decide when to deceive as action, and optimize long-term harm with rollouts using MDP. The main difference appears to be moving from a single human to a three-person team when interacting with the AI assistant, which is an incremental extension of the same paradigm.

2. Writing and problem framing are not coherent. Here are a few examples.
a) The introduction begins with a human–AI trust and safety story but ends by adding a new goal, namely checking whether LLM agents can replicate human behavior. This is not motivated by the earlier threat model and reads as disconnected from the main contribution.
b) The paper defines the reward strictly on the current round (the team score difference with and without the adversarial AI) but then claims to use dynamic programming and to “look ahead” several rounds. With a purely one-step reward there is no need to roll out future states, so either the actual objective is cumulative and not written, or the current formulation is incomplete. As written, the reward and the planner do not match. 
c) The core method, UI, and evaluation are built for humans who explicitly allocate influence points. Dropping in LLM agents at the end does not follow from the setup and looks like content padding.

3. Experimental section is underspecified and weakly justified
a) Equal-weight baseline is not clearly defined. It is unclear whether it means uniform influence allocation, or simple averaging.
b) The psychology-based model performs poorly. In the comparison with the learned MLP, the model is close to or worse than the simple baseline, yet this is the model chosen for most of the RL rollouts because it is easy to simulate. This makes the justification of section 4.1 very weak as it does not fit the human behavior very well.

4. Unnatural human-study design
a) Unrealistic task-difficulty selection.
Participants choose the question difficulty themselves each round. It is unclear why a real-world collaborative AI setting would allow a team to self-select task difficulty. Most realistic environments assign tasks or randomize difficulty. Allowing participants to choose introduces potential confounding variables, and the paper does not explain what happens if participants disagree on difficulty. 
b) Artificial “influence” mechanism.
The “influence” manipulation is not naturalistic. Participants see all answers and then must manually allocate a fixed “influence budget” to all four members every round. The team score is then mechanically computed from these allocations. This does not resemble real human-AI collaboration, where people discuss and form a joint decision or the system aggregates input automatically. The attack therefore exploits a contrived UI mechanic rather than demonstrating realistic persuasive or deceptive behavior. As a result, the validity and generalizability of the findings are limited to generalize well.











[1] Strategic adversarial attacks in ai-assisted decision making to reduce human trust and reliance. IJCAI 23

### Questions
see above weakness

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
This paper studies how adversarial AI agents can manipulate human-AI teams using Model-Based Reinforcement Learning. In a trivia game setup, the AI learns to exploit human trust dynamics, reducing team performance. A cognitive and a data-driven MLP method are compared, with the MLP model proving more effective. The study also shows that LLM-based teams display human-like trust behaviors and are similarly vulnerable to attacks.

### Strengths
- The paper is proposing an interesting approach to attack humans and AI collaboration by modeling humans' trust evolution. 
- The proposed method is both evaluated with a human in the loop and with LLM. The evaluation shows the proposed attack is able to reduce the overall team performance.

### Weaknesses
- This paper only evaluated using only 1 type of game. It is unclear how and if the proposed method can be generalized to more complex scenarios where it is non-trivial to get the specific intermediate metric for 
- This work seems to be missing discussions of related works on attacks in multi-agent reinforcement learning literature, which can have a similar setting to this paper.
- There seems to be limited performance comparison to other baseline attack methods, such as the ones that focus on attacking cooperative MARL or even a random selection policy for the AI attacker.

### Questions
- What are the specs of the inputs to the MLP? The paper mentions that "round number, the current performance (c.p.) of the human and AI agents, and a summary of past correct answers" are used as input. How are these encoded and sent to the MLP?
- It seems the MLP is not using language/text information. Would it have higher performance if you also include text information there? The authors claim that the model based on principles of cognitive psychology does not lag too far behind the MLP model, does this still hold if you use more advanced method?
- For the evaluation, how much of the collected human data is used for training and how much is used for evaluation for the accuracy of the human model?
- Do these same approaches work for other games, especially the ones with high variance in the team score?

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
This paper explores how an adversarial AI assistant can intentionally harm decision-making in human-AI teams using a model-based reinforcement learning (MBRL) framework. In a trivia-style game with 25 teams of humans, the AI learns to model human trust evolution and strategically misleads them to degrade performance. Two models, one grounded in cognitive psychology and another data-driven (MLP), serve as the attacker’s internal models. The results show that both attacks successfully reduce team performance, with the data-driven model being more effective. The authors further simulate similar attacks on teams of large language models (LLMs), finding that even advanced models show human-like vulnerabilities to manipulation.

### Strengths
The paper tackles an important and timely topic, studying trust manipulation in human-AI collaboration, through a creative experimental setup.

It integrates psychological theory with reinforcement learning in a clever way, grounding the attack models in cognitive principles.

The experimental design (human-AI trivia teams) is easy to understand and allows for clear quantification of performance degradation.

The inclusion of both human and LLM “team” experiments broadens the relevance of findings to real-world AI safety concerns.

The analysis connects observed behaviors, such as over-reliance on AI or trust collapse, to known human cognitive biases, adding interpretability to technical results.

### Weaknesses
The human study is well executed and valuable given how hard it is to recruit real participants for this kind of setup, but since it focuses only on a short trivia-style game, it’s still unclear how the findings would generalize to other types of collaborative tasks, higher-stakes environments, or longer-term interactions where trust and deception evolve differently.

The setup feels a bit too clean and simplified; it’s a short trivia game where the AI either lies or tells the truth, which doesn’t fully capture how messy and nuanced deception or teamwork can be in real life.

The models assume people update trust in a logical, almost math-like way, but real human trust involves a lot more emotion, tone, and social context, and the validation study focuses more on plausibility than real persuasion or manipulation effects.

The LLM simulation part isn’t explained clearly enough; it’s not totally clear whether these models actually mimic human trust patterns or if they’re just picking up on easy cues from the chat history.

The ethical side feels a bit light; there’s little mention of how participants were debriefed after being deceived, and releasing attack methods like this could backfire without stronger guardrails or usage guidelines.

### Questions
How exactly does the RL attacker balance “lying frequency” with maintaining trust? Does the model explicitly optimize for long-term believability?

Were participants aware that the AI might perform poorly or inconsistently, or was full deception used during the study?

How were the trivia questions selected and verified to avoid bias toward topics the AI might be stronger in?

How were LLM “teams” operationalized? Were they role-playing humans, or simply optimizing point distribution based on chat logs?

If the attacker learns human trust evolution, what prevents it from exploiting individual differences (e.g., cautious vs. trusting participants)?

### Soundness
3

### Presentation
3

### Contribution
4
