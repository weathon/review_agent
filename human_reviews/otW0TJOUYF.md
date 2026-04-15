# Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models

- Decision: Accept (Poster)
- Scores: 5, 8, 8, 6

## Abstract
Multi-agent reinforcement learning (MARL) methods struggle with the non-stationarity of multi-agent systems and fail to adaptively learn online when tested with novel agents. Here, we leverage large language models (LLMs) to create an autonomous agent that can handle these challenges. Our agent, Hypothetical Minds, consists of a cognitively-inspired architecture, featuring modular components for perception, memory, and hierarchical planning over two levels of abstraction. We introduce the Theory of Mind module that scaffolds the high-level planning process by generating hypotheses about other agents' strategies in natural language. It then evaluates and iteratively refines these hypotheses by reinforcing hypotheses that make correct predictions about the other agents' behavior. Hypothetical Minds significantly improves performance over previous LLM-agent and RL baselines on a range of competitive, mixed motive, and collaborative domains in the Melting Pot benchmark, including both dyadic and population-based environments. Additionally, comparisons against LLM-agent baselines and ablations reveal the importance of hypothesis evaluation and refinement for succeeding on complex scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces Hypothetic Minds a new module LLM-based framework for approaching Multi-agent problems across Cooperative, Competitive and Mixed Intention Settings. Their key contribution is in the form a Theory of Mind module which generates hypotheses about strategies followed by other agents and uses these hypotheses to make predictions of next actions. The hypotheses are evaluated  based on the ground truth observations and refined using a Rescorla Wagner update rule. The authors test their framework on 4 games  from the Melting Pot benchmark.

### Strengths
S1. The authors propose a well motivated plug-and-play Theory of Mind module for modular LLM frameworks that uses hypotheses about other agents as natural language representations of latent variables and then utilize and update these hypotheses to select the best actions. 

S2. Their method works across all three settings: Competitive, Cooperative and Mixed intention which is important for developing general purpose multi-agent frameworks. 

S3. Most of the paper is clear and easy to follow.

### Weaknesses
W1. Lack of generalization: The method is only tested in the limited setup of test scenarios of the Melting Pot benchmark. The Theory of Mind module seems to be a general addition that can be extended to any task. Comparison and experimental results in benchmarks which have been previously evaluated with strong LLM-based methods are necessary to demonstrate the generalizability of the framework and the theory of mind module. For example: 
a. Games like Avalon [1], Werewolf [2] for Mixed Intention and Competitive setting
b. Games like the Hanabi challenge [3], ThreeDWorld Multi-Agent Transport [4] (TDW-MAT) and Communicative Watch-And-Help [4] (C-WAH) for cooperative setting. 

W2. Within the Melting Pot benchmark, only a few games have been picked. Does this framework also work effectively on the other setups? Are there any technical limitations restricting the evaluations in the other setups? Even within the collaborative cooking game, only the asymmetric advantages environment tested. However, other maps like Forced Coordination where neither player can complete the task entirely on their own are necessary tests for the ability of this framework. It is insufficient evidence to only evaluate only one game for those three settings. 

W3. Missing ablation studies on top-k, v_thresh and Rescorla Wagner update. While these are hyperparameters, it is important to show how they affect the overall performance as they are part of the Theory of Mind modules which is the key contribution in this paper. 

W4. Concerns about the Hypothesis Refinement step: In the ablation study in Figures 5: MMP + HE is better in 2/4 scenarios (This trend is also seen in Figure 8). This leads to a question about the necessity of the Hypothesis Refinement step. 

W5. While most of the paper is clear, the Ablation studies section with those additional definitions is difficult to follow. Furthermore the conclusion is incomplete and disorganized.

[1] Shi, Zijing, et al. "Cooperation on the Fly: Exploring Language Agents for Ad Hoc Teamwork in the Avalon Game." arXiv preprint arXiv:2312.17515 (2023). 

[2] Xu, Yuzhuang, et al. "Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf." arXiv preprint arXiv:2309.04658 (2023). 

[3] Bard, Noam, et al. "The Hanabi Challenge: A New Frontier for AI Research." Artificial Intelligence 280 (2020): 103216.

[4] Zhang, Hongxin, et al. "Building Cooperative Embodied Agents Modularly with Large Language Models." arXiv preprint arXiv:2307.02485 (2023).

### Questions
Q1. What exactly is Hypothesis Refinement in the ablation studies? Does it mean that in MMP + HE the agent only generates an initial set of hypothesis and does not generate any more hypothesis during the episode? Or is it just that the LLM is not shown the previous hypotheses? 

Q2. What is the rationale for the choice of the Rescorla Wagner update and the hyperparameter 'c'. How does this choice and the value of c impact the convergence of the agent during the episode? Furthermore, how does the choice of top-k influence this?  

Q3. "A hypothesis at time t is generated by asking an LLM to infer another agent’s strategy, conditioned on HM’s memory M of important past observations O" - how does this work in cases where there are more than one opponents/partners? Does the LLM generate a unique hypothesis for each agent? Is the next action prediction based on all or one of these hypothesis? 

Q4. "Each ToM module call uses 3 LLM calls + top k times" This statement is a bit unclear - does this mean there are (3 + K) LLM calls or 3*K LLM calls ? 

Q5. Why are other Collaborative Cooking environments not included in the experiments?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present "Hypothetical Minds" a novel approach to scaffolding LLMs by introducing hypothetical reasoning scenarios that help models better understand and respond to complex queries. They demonstrate that by creating imaginary scenarios and personas, LLMs can improve their reasoning capabilities and provide more nuanced responses across various tasks, including ethical reasoning and creative problem-solving. The work introduces a systematic framework for implementing hypothetical reasoning, supported by extensive empirical evaluation across multiple model architectures and benchmarks.

### Strengths
- The methodological innovation of using hypothetical scenarios as a scaffolding technique is both novel and well-justified, with clear connections to cognitive science literature on human learning. Love to see studies leveraging tools from cog sci to improve AI systems
- The experimental design is comprehensive, testing the approach across different model sizes and architectures, which strengthens the validity of the findings
- The authors provide detailed ablation studies that effectively isolate the impact of different components of their scaffolding approach
- The qualitative analysis of model outputs offers valuable insights into how hypothetical reasoning affects model behavior

### Weaknesses
- The theoretical foundation could be strengthened by more explicitly connecting to existing work on prompt engineering and chain-of-thought reasoning
- The evaluation metrics could be expanded to include more diverse tasks beyond the current set, particularly in domains requiring complex logical reasoning
- I think having more seeds to decrease SEM and confirm statistical significance across all experiments would help strengthen the results
- The scalability analysis of the approach could be more thorough, especially regarding computational overhead and implementation costs
- Consider including a more detailed error analysis to understand when and why the hypothetical scaffolding approach fails

*Note: Very happy to re-evaluate my score if authors address my concerns.*

### Questions
- I'm somewhat surprised not to see Rational Speech Act mentioned. Have you considered extending the ToM module to recursive reasoning about others' ToM?
- How does the performance of the hypothetical scaffolding approach vary with the complexity of the scenario being presented?
- Have the authors considered the potential limitations of using hypothetical scenarios that might introduce unintended biases?
- What is the relationship between the number of hypothetical scenarios used and the model's performance improvement?
- How does this approach compare to other scaffolding techniques in terms of computational efficiency?
- There are a few relevant papers out there, that somewhat similarly have LLMs simulate other agents or people and hypothetical scenarios, that might be worth discussing. E.g., There was a paper with that kind of flavor from Tom Griffiths's lab called "Improving interpersonal communication by simulating audiences with language models"

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
1. The proposed Hypothetical Minds model (HM) is an embodied LLM agent for multi-agent environments. It integrates modular components for perception, memory, and hierarchical planning conditioned on Theory of Mind (ToM) inferences.

2. The Theory of Mind (ToM) module in HM generates, evaluates, and refines hypotheses about other agents’ strategies or goals in natural language. The critical role of hypothesis evaluation and refinement within the ToM Module is identified through ablations and comparisons against LLM-agent baselines.

3. The effectiveness of Hypothetical Minds across multiple multi-agent environments is demonstrated in the Melting Pot benchmark. The agent significantly outperforms LLM-agent and RL baselines in every environment and in a large majority of evaluation scenarios, showcasing its generalizability.

### Strengths
The Theory of Mind Module has been introduced to predict the next behavior of the agent and to make high-level strategy based on best hypothesis.

The proposed Hypothetical Minds agent significantly improved performance over previous LLM-agent and RL baselines on a range of competitive, mixed motive, and collaborative domains including both two-person and population-based environments. A thorough literature survey with a well planned set of experiments and evaluation in the Melting Pot Benchmark have been carried out.

### Weaknesses
1. The roles played by the various latent variables in the Theory of Minds module and the weightages assigned with these variables have not been discussed.

2. How the values of the latent variables are identified is not clear?

3. Lines 033 to 035 in the paper mention that: However, multi-agent reinforcement learning (MARL) methods suffer from
various draw backs, including high sample complexity, poor generalization to agents not seen in training, and limited reasoning capabilities. 

– Citation is necessary for the above claim.

4. Dependence on human subjects to set up scaffolding and prompting for the agent is an important drawback of the proposed system.

### Questions
1. Are Chain of Thought, Tree off Thought and Theory of Mind related ? Please share with examples connecting to your proposal in the paper.

2. What is SAMA in L143.

3. L233 – How the prior probabilities are calculated?

4. L267 -268 - .Each ToM module call uses 3 LLM calls + top k times for behaviour prediction …

Please explain what is done top k times. It is not clear from the above phrasing

5. L342-343 – We directly test our LLM based agent on the evaluation scenarios in four Melting Pot environments (Fig. 3). 

I believe the reference should be Figure 4 instead of 3 right? If so, please correct it

6. Figure 4 . …. 3 seeds are generated per LLM. 

What are the three seeds?

### Soundness
3

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
3

### Summary
The paper introduces "Hypothetical Minds" (HM), an advanced autonomous agent framework leveraging LLMs for multi-agent tasks. HM incorporates a cognitively inspired architecture with distinct modules for perception, memory, and multi-level planning. A crucial innovation is the Theory of Mind (ToM) module, which actively constructs, assesses, and refines hypotheses about other agents’ behaviors in natural language. This hierarchical planning, alongside effective strategy adaptation, enables HM to excel in diverse multi-agent scenarios.

The system is evaluated using the comprehensive Melting Pot benchmark across four different environments, each containing multiple evaluation scenarios that represent competitive, cooperative, and mixed-motive tasks. The results show that Hypothetical Minds consistently outperforms established LLM-agent and RL baselines, demonstrating superior adaptability and generalization.

### Strengths
**Originality**: The paper introduces a novel and sophisticated use of Large Language Models in the context of multi-agent systems by incorporating a Theory of Mind module. This innovative approach allows the agent to not only predict but also adapt to other agents' strategies using natural language. The hypothesis generation and evaluation mechanism represent a significant advancement in the field, setting this work apart from traditional RL and existing LLM-based systems.

**Quality**: The authors have thoroughly validated their approach through comprehensive and rigorous experiments. Evaluations across diverse scenarios in the Melting Pot benchmark demonstrate the robustness and adaptability of the Hypothetical Minds model. The detailed ablation studies provide clear evidence for the effectiveness of the modular components, particularly the hypothesis evaluation and refinement mechanisms.

**Clarity**: The paper presents its methods and findings clearly. The inclusion of detailed empirical results, ablation studies, and illustrative examples helps in comprehending the functioning and impact of the Hypothetical Minds model. The explanation of complex mechanisms such as the Theory of Mind module and hypothesis refinement is done in an accessible manner without compromising on technical depth.

**Significance**: This work addresses critical challenges in the development of autonomous agents, specifically in terms of strategy adaptation, nonstationarity, and generalization across diverse multi-agent environments. The significant performance improvements over state-of-the-art baselines highlight the potential of this approach to influence future research and applications in multi-agent systems. The results show that Hypothetical Minds can be a foundational framework for building more sophisticated and adaptable autonomous agents.

### Weaknesses
**High Computational Cost**: Utilizing state-of-the-art LLMs, particularly GPT-4, incurs significant computational expenses and longer inference times, which can be impractical for some applications. Have you considered conducting experiments on smaller language models and test the effectiveness of the framework?

**Manual Preprocessing for State Representation**: The current method relies on extensive manual preprocessing to convert pixel images into textual map representations. This approach is time-consuming and may not generalize well across various environments.

**Presentation Issues**: The paper would benefit from a more comprehensive introduction to the key problem being addressed. It takes a while for the reader to fully grasp the challenge the model aims to solve. Additionally, there are some formatting issues, such as citations referring to preprint versions (e.g., Wang et al., 2023b; Toolformer) which should be updated to the respective peer-reviewed conference versions if available. Appendix G (Line 1346) contains a broken link that needs fixing. Figures and Tables are not links but rather plain text.

### Questions
- How does the agent determine the reward threshold (Vthr) for validating hypotheses in the Theory of Mind module, and how sensitive is the model’s performance to this threshold?

- Could you provide more details on the specific algorithms or heuristics used in the preprocessing step to generate textual map representations from pixel images?

### Soundness
3

### Presentation
1

### Contribution
3
