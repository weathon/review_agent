# Opponent Shaping in LLM Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 4

## Abstract
Large Language Models (LLMs) are increasingly being deployed as autonomous agents in real-world environments. As these deployments scale, multi-agent interactions become inevitable, making it essential to understand strategic behavior in such systems. A central open question is whether LLM agents, like reinforcement learning agents, can shape the learning dynamics and influence the behavior of others through interaction alone. In this paper, we present the first investigation of opponent shaping (OS) with LLM-based agents. Existing OS algorithms cannot be directly applied to LLMs, as they require higher-order derivatives, face scalability constraints, or depend on architectural components that are absent in transformers. To address this gap, we introduce ShapeLLM, an adaptation of model-free OS methods tailored for transformer-based agents. Using ShapeLLM, we examine whether LLM agents can influence co-players’ learning dynamics across diverse game-theoretic environments. We demonstrate that LLM agents can successfully guide opponents toward exploitable equilibria in competitive games (Iterated Prisoner’s Dilemma, Matching Pennies, and Chicken) and promote coordination and improve collective welfare in cooperative games (Iterated Stag Hunt and a cooperative version of the Prisoner’s Dilemma). Our findings show that LLM agents can both shape and be shaped through interaction, establishing opponent shaping as a key dimension of multi-agent LLM research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces ShapeLLM, a model-agnostic opponent shaping (OS) algorithm that is suitable for LLM architectures. ShapeLLM leverages natural language capabilities by encoding history and context in the repeated play in natural language prompts. It then utilizes a partially observable stochastic game formulation for the policy updation process, making it potentially useful even beyond the context of repeated normal-form games. The effectiveness of the proposed algorithm is demonstrated in a series of 2*2 repeated games, involving both competitive and cooperative interactions. The results demonstrate that in these particular games, ShapeLLM improves the performance of the shaper agent in the competitive games, and steers the interactions towards collaboration in cooperative games.

### Strengths
1. The paper is very well-written, and is very easy to follow and fun to read. The paper covers the related literature, making it easy to understand its contribution to the existing research landscape. It also provides preliminaries for readers who are not entirely familiar with some aspects of the work, making the writing very suitable for the ML community, which is fast-growing and multi-disciplinary by nature.

2. The paper proposes a novel approach that addresses crucial gaps in the existing OS literature. In particular, the proposed approach can be naturally applied to LLM-based agents in various contexts, and as the paper itself suggests, even far beyond those introduced in the experimental setup.

3. The paper reports experimental results using classical game-theoretic setups that capture core elements in strategic behavior that are crucial for OS. This simplicity enables the identification of core behaviors that emerge using the proposed algorithm, making it easier to understand not only *whether* OS occurs, but also *how*.

### Weaknesses
My major concern with the paper (which was also discussed by the authors themselves) is the oversimplicity of the experimental setups, which only include simple two-player two-action games, with no natural language communication. While this simplicity is obviously an advantage (see the *strengths* section), I believe that providing some results in some more complex and realistic environments could have significantly strengthened the contribution of the paper, especially for ML practitioners who might want to apply it to real-world scenarios such as trading, negotiation, persuasion, content generation, etc.

### Questions
I don’t have a specific question here, but I would be interested in the authors’ reflections on how far they believe their approach could generalize to more complex and realistic scenarios. It would also be valuable to hear their thoughts on potential limitations or pitfalls in such extensions, and how these challenges might be addressed.

### Soundness
4

### Presentation
4

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
This work proposed an opponent shaping method (ShaperLLM) and applied it to multi-agent game playing tasks to invesitgate how opponent shaping impacts agents' behaviors. ShaperLLM's algorithm derives from SHAPER by Khan et al., and the authors simplified it and directly optiomized LLMs with Reinforcement Learning. The experiments highlight that LLM agent itself can perform opponent shaping and opponent shaping can promote agents' cooperation.

### Strengths
1. Proposed a method that directly incorporating SHAPER algorithm into the training process of LLMs without external policy network like RNNs.
2. Conducted experiments on various game playing tasks and presents insights on the effectiveness of opponent shaping on multi-agent scenario.

### Weaknesses
1. **Presentation of ShaperLLM's algorithm lacks clarity**: In section 3, the description of two RL rewards (opponent reward and shaper reward) of ShaperLLM is unclear. The two rewards are heavily analyzed in the experiments but their formulas are lacking. Also, some terms for the setting description are a little confusing, like *steps*, *trails*, *rounds*, as they are often interchangeably used in different contexts.
2. **Experiments lack generalizability and scalability**. This work only did experiments on gemma-2-2b-it so their findings may lack generalizability (the authors also ack this in the paper).
3. **Weak baseline**. The work compares ShaperLLM only with *naive agent* that sees only one step of previous interaction history and doesn't compare with other RL methods.

### Questions
1. I would like to see clear description of the **formula definitions** & **optimization objectives** of opponent reward and shaper reward in Section 3 (line 212-215).
2. Somes terms are interchangeably used and confusing. For example, I think $t$ in line 162 means each step, but in later context (line 174-175) means one round. The meaning of $n_{games}$ is not straightforward, does it mean the total number of games? Maybe use one letter like $N$ to replace $n_{games}$ is better. Also in line 174-175 you mentioned *trails* and in line 214-215 you addresses *trail return*, but in experiment setup (Section 4.2) the trail setting is not described; you described *step* instead. What's the relation between *trail* and *step*, could you unify/distinct the terms for the same/different concepts?
3. In line 199, $POMDP(...)$, what's the meaning of $\Omega$ and $\gamma$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates a method for fine-tuning small LLMs for opponent-shaping in iterated matrix games using a Shaper-style algorithm where a naive learner updates their parameters inter-trajectory with PPO and the shaping LLM learns after a seeing a full batch of trajectories, called a Trial. Crucially the LLMs only have available the past action tuple and a histogram summary of the past actions in the episode/Trial.

### Strengths
This is the first paper which finetunes LLMs for opponent-shaping in iterated general-sum matrix games. I found the paper decently clear and well-written, the figures and experiments were all well-motivated. The experiments were sufficient to show what the authors wanted to show (especially with what was likely a constrained compute budget). Passing a summary of past actions as context to the LLM is a clever idea to get around the compute constraints of having to pass very long sequences to the models.

### Weaknesses
Unfortunately the biggest weakness is the lack of novelty. 

The claim in the introduction that past opponent-shaping methods cannot be applied to LLMs is wrong. Advantage alignment ( https://arxiv.org/abs/2406.14662 ) changes the PPO advantages in a way that would also be easy to do for LLMs. And the following paper from earlier this year also implements a meta-episode context with transformers which implements opponent-shaping: https://arxiv.org/pdf/2410.18636 . Past papers don't implement the summarization mechanism for giving context to the model, but this is a trick for saving compute that only works in toy contexts like iterated matrix games, for more complex games one would need to pass the entire past N trajectories as context, which is exactly what Meulemans et. al. do.

The novelty of the paper lies entirely in applying a more limited version (because of the summarization) of the Meulemans et. al. idea to finetuning LLMs on a simpler game than they have.

### Questions
1. Are the Naive Learner parameters reset after each Trial? this is not clear to me from the paper.
2. I'm skeptical of the IPD results in figure 2, the learning does not appear to converge to a Nash equilibrium given that the visitations of DC are by far the most dominant. It seems to me like the naive learner is simply not learning to defect correctly, all past shaping methods on the IPD have the shaping agent converge to tit-for-tat or other reciprocal policies, so it's strange to me that this would converge to always DC
3. Minor quibble in the introduction, but the distinction between an "RL agent" and an "LLM agent" does not really exist, RL is a training procedure that can and does produce LLM agents.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies opponent shaping (OS) in Large Language Model (LLM) agents. It examines whether an LLM can influence another learning agent’s training dynamics and subsequent behavior solely through repeated interaction. The work introduces ShapeLLM, an adaptation of model-free opponent shaping ideas inspired by M-FOS and SHAPER. Because existing OS algorithms rely on higher-order gradients or architectural components not present in transformer models, ShapeLLM instead encodes both within-episode history and across-episode opponent-learning information as structured natural-language context.

Empirically, the authors show that LLM agents trained with ShapeLLM:
- Competitive settings: guide co-players toward exploitable strategies in repeated 2×2 games (Iterated Prisoner’s Dilemma, Matching Pennies, Chicken), yielding higher long-run payoffs.
- Cooperative settings: induce more cooperative play and higher joint payoffs in Iterated Stag Hunt and a cooperative variant of the Prisoner’s Dilemma.

### Strengths
- The work is the **first** formal study of OS in LLM agents. It targets multi-agent LLM interaction concerning how one agent may influence another’s learning dynamics.
- The proposed ShapeLLM adapts model-free OS ideas to transformer-based agents by encoding within-episode history and cross-episode learning context in structured natural-language prompts. 
- Experiments span multiple repeated 2×2 settings: competitive (Iterated Prisoner’s Dilemma, Matching Pennies, Chicken) and cooperative (Iterated Stag Hunt, cooperative IPD). This shows that the shaping procedure functions under varied incentive structures.

### Weaknesses
- Many works already evaluate LLMs and non-LLM agents in competitive or cooperative games. The paper’s contribution is not fully disentangled from these prior settings.
- ShapeLLM encodes within-episode and across-episode information as natural-language context. However, the causal link between this prompt structure and the opponent’s parameter updates is not analyzed. As a result, it is unclear to what extent the observed shaping behavior arises from the LLM’s strategic adaptation versus different prompt design.
- This paper only analyze 2×2 settings, which might limits some claims on competitive setting and collaborative setting.

### Questions
1. Can the authors report ablations on the prompt design? In particular, how does shaping performance vary when altering the amount of intra-episode history versus inter-episode context included?
2. What computational resources (GPU hours, etc) are required to train the shaper under PPO in the reported settings? And what are the estimation if doing larger scale experiments?
3. How do the authors expect ShapeLLM to extend to larger or partially observable environments with richer action spaces (e.g., text-based negotiation or resource allocation)? Are there preliminary results or arguments supporting such transfer?
4. Have alternative base LLMs (e.g., different sizes or pretraining sources) been tested, and if so, how does the choice of model affect shaping behavior? This would help determine whether shaping is model-agnostic.

### Soundness
3

### Presentation
3

### Contribution
2
