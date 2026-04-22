# Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Large language model (LLM) agents exhibit strong mathematical problem-solving abilities and can even solve International Mathematical Olympiad (IMO) level problems with the assistance of formal proof systems. However, due to weak heuristics for auxiliary constructions, AI for geometry problem solving remains dominated by expert models such as AlphaGeometry 2, which rely heavily on large-scale data synthesis and search for both training and evaluation.
In this work, we make the first attempt to build a medalist-level LLM agent for geometry and present InternGeometry. InternGeometry overcomes the heuristic limitations in geometry by iteratively proposing propositions and auxiliary constructions, verifying them with a symbolic engine, and reflecting on the engine's feedback to guide subsequent proposals. A dynamic memory mechanism enables InternGeometry to conduct more than two hundred interactions with the symbolic engine per problem. To further accelerate learning, we introduce Complexity-Boosting Reinforcement Learning (CBRL), which gradually increases the complexity of synthesized problems across training stages.
Built on InternThinker-32B, InternGeometry solves 44 of 50 IMO geometry problems (2000-2024), exceeding the average gold medalist score (40.9), using only 13K training examples, just 0.004% of the data used by AlphaGeometry 2, demonstrating the potential of LLM agents on expert-level geometry tasks. InternGeometry can also propose novel auxiliary constructions for IMO problems that do not appear in human solutions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. The paper introduces InternGeometry, an LLM-based agent that achieves medalist-level performance on International Mathematical Olympiad (IMO) geometry problems. 
2. Unlike previous systems such as AlphaGeometry2, which rely heavily on large-scale data synthesis and heuristic search, InternGeometry integrates long-horizon reasoning, symbolic feedback, and a Complexity-Boosted Reinforcement Learning (CBRL) framework. 
3. This work performs efficient training using only 13K examples (0.004% of AlphaGeometry2’s data).

### Strengths
1. This work introduces a long-horizon agentic reasoning paradigm for geometry, moving beyond static and propose a model toward interactive proof reasoning.
2. The complexity curriculum addresses sparse-reward issues and ensures stable reinforcement learning progression.
3. This work achieves state-of-the-art performance on IMO geometry problems with significantly less training data.

### Weaknesses
1. Does Pass@256 with 200 steps implies ~51K LLM–engine interactions per problem? Does this raise scalability and efficiency concerns. 
2. The paper does not report inference time per problem or single-shot success rate (Pass@1); it will make the paper better if we have this presented so we can assess the model’s true reasoning efficiency.
3. Would be nice if authors can add a baseline between the pretrained InternThinker-32B and the CBRL-trained InternGeometry, this will make it clear on how much improvement gain from CBRL does the proposed method have. 
4. Although case studies are insightful, a deeper analysis of failure cases (the unsolved problems in Table 2) or reasoning trajectories would provide better understanding. Authors can add this qualitative analysis in the draft, which will make the readability better. 
5. Will be great if the paper can mention inference/training compute requirements to reproduce the results. This will help in reproducibility.
6. To justify the InterGeometry's generalization capabilities, authors may want to include an analysis of the cross-domain performance on mathematical and scientific reasoning tasks, which can improve the usefulness of the paper.

### Questions
1. How crucial are the additions of "dynamic diagram adjustment" and "double points" handling for solving the IMO-level problems? Will be good if authors can add this to improve the quality of the paper. 
2. As mentioned in Equation (9) add a brief explanation of how maximizing "absolute advantage" creates a curriculum of moderate difficulty and how the search for ‘k’ is done in each round.
3. The paper builds upon InternThinker-32B as the foundation for InternGeometry model. However, the current version lacks sufficient details about InternThinker-32B’s architecture, training setup, and reasoning capabilities. Hope authors can add this to the draft and improve the paper quality. 
4. Also, please clarify what aspects of InternThinker-32B enable effective long-horizon reasoning and symbolic interaction in this work.

### Soundness
3

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
While LLM agents and agentic workflows have shown strong performance (gold medal level) in International Mathematical Olympiad (IMO), solving geometry problems is still by far the most challenging task, and existing IMO math agents typically have poor performance on such problems. The main reason is that a large subset of geometry problems typically requires multiple trial and error to find the right auxiliaries to add to the current problem diagram to make it solvable. Solutions like AlphaGeometry exist, these solutions typically require an enormous amount of formal data to either train a custom transformer from scratch or perform continual training on the same data but on already pre-trained models. Then, the trained model on formal data would go in a loop with a deductive symbolic engine, where the model proposes constructing a new auxiliary to the problem at hand, and then the symbolic engine would run over and over again to extensively perform deductive search for reaching the proof goal, and upon exhaustion of the engine, the loop continues.

The vast amount of search done on formal deductive engines, and the enormous amount of training data make the existing solutions very inefficient. On the other hand, it is widely known that LLM agents and workflows can reach perfect performance on non-geometry problems in either formal or non-formal settings even when fine-tuned on a very limited amount of training data compared to AlphaGeometry. Thus, this work aims to address this timely question of how LLM agents can be designed to tackle geometry problems as well. They propose introducing a novel tool call for dynamic interaction with their enhanced deductive engine, supporting multiple actions such as building the formal geometric configuration, adding auxiliary constructions, and proposing a proposition to be verified. The dynamic interaction with the engine enables leveraging the already strong high-level natural language planning and reasoning of frontier LLMs for optimized interaction with the engine.

To enable this tool call, they performed cold-start training to teach their base model (InternThinker-32B) how to work with this new tool, and then performed curriculum RL to improve the reasoning with their proposed tool call, and the final solution is InternGeometry which is claimed to perform on par with, or even sometimes outperform the two main baselines: AlphaGeometry and SeedGeometry on the IMO 50 evaluation suite (all geometry problems from 2000-2024 IMO). In contrast to these baselines, InternGeometry is trained on roughly 0.005% of the training data size of other baselines (13k samples, 7k for cold-start, and 6k for RL).

### Strengths
1. The introduction of a new tool call for leveraging strong planning and reasoning of LLMs for dynamic interaction with the formal deductive engine is novel and an exciting idea. The paper did a good job explaining this tool call and motivating it, which makes their high-level approach clear and sound.

2. As mentioned in the summary, much trial and error is typically expected for figuring out the helpful auxiliary constructions for solving the given problem, and this fact would require the proposed agent to deal with the challenge of long-horizon reasoning that demands careful context management of past exploration after each tool call. The paper then proposes to maintain a dynamic memory containing key information from past tool call iterations (e.g., what actions were made in previous turns, what were the outcomes of those actions, plus the current action and current feedback from the deductive engine which contains all successful propositions). The introduction of this dynamic memory is novel and an interesting solution for solving the weak heuristic nature of auxiliary construction in IMO-level geometry problem solving. The ablation also confirms the long-horizon interaction is indeed necessary for obtaining the InternGeometry performance.

3. Only requiring roughly 0.005% of the key baselines' training data (AlphaGeometry and SeedGeometry) to reach comparable performance, and even outperform them on the IMO 50 dataset is an impactful contribution to the LLM math agents community, and this work can be seen as an initial step to harness general-purpose LLMs for solving complex geometry problems.

4. The paper mentions the model, data, and the deductive engine used will be open-sourced, which is of great benefit to the community, and this would make this work reproducible.

### Weaknesses
1. While the RL reward and RL loss are clearly defined, the handling and explanation of the curriculum algorithm lacks clarity, which makes it hard to evaluate the soundness of the proposed curriculum approach. The paper attempts to touch on the theory behind the curriculum algorithm on the surface, and both Theorem 1 and 2 statements are hard to follow and vague, and could be better explained. More importantly, the paper does not explain the CBRL algorithm with sufficient detail and particularly it is not clear how the complexity $\kappa$ is updated in each CBRL round. The paper only briefly mentions the following (line 253): "In practice, in each CBRL round, we sample data conditioned on complexity $\kappa$, perform RL training to the agent, and finally update κ according to learning rate $\alpha$."

2. Data curation for cold-start requires clarity, as the complexity of obtaining and curating such data is actually high. Regarding this, the paper only briefly addresses this around line 262, mentioning that "First, due to the scarcity of data in formal systems, we fine-tuned InternThinker-32B as InternGeometry-Formalizer through expert iteration (Anthony et al., 2017) and then exploit large-scale natural language problem data from diverse sources. This process produced a total of 7K formal problem and solution trajectory pairs, which provide a cold start for InternGeometry." I believe due to the complexity of data curation for this phase, one would want to know the details of how exactly the cold-start trajectory is generated.

### Questions
1. Could you provide a simplified high-level pseudocode for the working implementation of CBRL so the selection of the curriculum complexity $\kappa$ becomes clear, as well as explaining the technical challenges of implementing the CBRL algorithm? This would greatly benefit the soundness and clarity of the RL training done in this paper.

2. Please see Weakness 2, and elaborate more on how exactly the cold-start data is generated?

3. A cost comparison between deploying InternGeometry versus AlphaGeometry or SeedGeometry would be greatly valuable for this work. I understand this might or might not be feasible based on the information available from the prior work, but it would be interesting to compare the number of interaction steps between the deductive engine and the LLM with the prior work. Especially since the LLM usage for proposing propositions might greatly improve the interaction efficiency as well.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces InternGeometry, an LLM agent system with a symbolic geometry solver (InternGeometry-DDAR) that solves math competition geometry problems. It improves the previous SOTA on 50 IMO geometry problems from 43/50 to 44/50, and achieves this with significantly less geometry problem training data than previous methods.

### Strengths
1. InternGeometry is the new SOTA on IMO 50.
2. The fact that the training set of geometry problems is much smaller than in previous approaches is impressive.
3. The analysis about scaling max steps vs. scaling # samples is insightful.

### Weaknesses
1. The paper would benefit from more discussion regarding inference-time costs of the compared methods. For example, listing the model sizes in Table 1 would be helpful, as well as explaining the different search parameters in AlphaGeometry2’s custom beam search. If available, information about the total # of output tokens or wallclock time etc. would also be appreciated.

2. Similarly, I think it would be nice if the paper also discussed training costs and compared them with previous methods. Currently, it mentions the size of the training set, but other information would also be helpful, such as the total # of tokens in the training set and a version of Figure 4 where the x-axis is the number of training tokens.

3. Writing quality is poor:
  - Contextualization within previous work is often missing, e.g., it is not clear what the novelty of CBRL is compared to previous work (curriculum learning for RL, GRPO).
  - There are many grammatical mistakes and unidiomatic uses of English, including in the title and abstract. I recommend using something like ChatGPT to improve the writing.
  - There are missing citations (e.g., the last two paragraphs of the introduction section have no citations)
  - Other miscellaneous issues, e.g., labeling imprecise claims as “theorems” (lines 247-251), potential typos in Eq. (5), and missing explanations (e.g., what’s InternThinker-32B? what’s “split” in Table 2?)

### Questions
1. What is InternThinker-32B? There is no citation, and I couldn’t find any information about it on the Internet.

2. How are Eqs. (5) and (6) different from GRPO? (other than what appears to be typos)

3. How is CBRL different from RL with a curriculum?

4. Should the top-left cell of Table 1 say “AlphaGeometry2” instead of “AlphaGeometry”?

5. The claim that InternGeometry’s test-time scaling budget is “far lower than that of AlphaGeometry2” (lines 311-312) is not clear to me. InternGeometry does pass@256 and uses a larger model (32B instead of 3.3B), so it’s not clear that it uses "far less" computational resources than AlphaGeometry2.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work introduces InternGeometry, a LLM agent designed to solve geometry proof problems at the IMO level. By integrating Complexity-Boosted Reinforcement Learning (CBRL) and dynamic memory mechanisms, the model achieves gold-medalist performance on IMO geometry problems, significantly outperforming strong baselines AlphaGeometry2 and SeedGeometry. Notably, InternGeometry only uses 13K training samples, two orders of magnitude less than AlphaGeometry and SeedGeometry. The experiments are thorough, demonstrating the effectiveness in data efficiency and long-range reasoning.

### Strengths
1. Propose the first LLM agent for IMO-level geometry proving, avoiding the use of specialist models.
2. Propose a dynamic memory mechanism and rejection sampling strategy, enabling up to 200-step interactive reasoning and guiding diverse explorations in interactions.
3. Solid experiments well justify that InternGeometry outperforms current SOTA models, with exceptional data efficiency. Comprehensive ablation studies validate the necessity of key components like long-range interactions, CBRL, and dynamic memory. The case study justifies the model's creative construction capabilities.

### Weaknesses
1. The title is not clear. Since the manuscript focuses on developing plane geometry prover, the title should contain such information.
2. Equation 5 extends beyond the page margin.
3. Experiments on more datasets (e.g., JGEX-AG-231 proposed in AlphaGeometry) and other LLMs (other than InternThinker) can further demonstrate the generalization ability of InternGeometry.
4. Considering that the interactive reasoning requires many steps, analyzing and comparing the computational resources of InternGeometry, AlphaGeometry, and SeedGeometry during reasoning is necessary.

### Questions
1. Line 40-42: Add references to justify LLM agents can obtain medalist-level performance on IMO-level problems.

### Soundness
3

### Presentation
3

### Contribution
3
