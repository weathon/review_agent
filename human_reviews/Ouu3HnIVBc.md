# ADAM: An Embodied Causal Agent in Open-World Environments

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 1

## Abstract
In open-world environments like Minecraft, existing agents face challenges in continuously learning structured knowledge, particularly causality. These challenges stem from the opacity inherent in black-box models and an excessive reliance on prior knowledge during training, which impair their interpretability and generalization capability. To this end, we introduce ADAM, An emboDied causal Agent in Minecraft, which can autonomously navigate the open world, perceive multimodal context, learn causal world knowledge, and tackle complex tasks through lifelong learning. ADAM is empowered by four key components: 1) an interaction module, enabling the agent to execute actions while recording the interaction processes; 2) a causal model module, tasked with constructing an ever-growing causal graph from scratch, which enhances interpretability and reduces reliance on prior knowledge; 3) a controller module, comprising a planner, an actor, and a memory pool, using the learned causal graph to accomplish tasks; 4) a perception module, powered by multimodal large language models, enabling ADAM to perceive like a human player. Extensive experiments show that ADAM constructs a nearly perfect causal graph from scratch, enabling efficient task decomposition and execution with strong interpretability. Notably, in the modified Minecraft game where no prior knowledge is available, ADAM excels with remarkable robustness and generalization capability. ADAM pioneers a novel paradigm that integrates causal methods and embodied agents synergistically. Our project page is at https://opencausalab.github.io/ADAM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Minecraft agent called Adam that relies on a combination of (M)LLM inference and causal discovery. At the core of the method is a causal graph that represents the agent's expertise on the environment logic. The causal graph is proposed by an LLM and each relation in the graph is confirmed/disproved by environment interactions. The proposed method obtains diamonds faster and more reliably than prior methods. Moreover,

### Strengths
- the paper is well structured and clearly written
- the combination of LLM-prompting with CD seems to be quite novel
- the intervention-based refinement of the causal graph seems meaningful and practical

### Weaknesses
- unsupported claims: the authors claim their method has "excellent interpretability" and that their agent "closely aligns with human gameplay;" yet, I cannot find any empirical evidence supporting these claims. 
- a runtime/memory analysis (be it theoretical or empirical) is completely missing, i.e., it is not clear at which cost the claimed SOTA results come
- the results presented in figure 1 are based on a modified causal graph claiming that this removes prior knowledge from LLMs; yet, it is unclear in how far this claim is true; it would be interesting to see an analysis akin to appendix A for the modified environment
- the choice of the acronym Adam is at best unfortunate as it coincides with one of the most influential machine learning papers (https://arxiv.org/abs/1412.6980) and could be mistaken for an attempt to tap unwitting citations as both paper titles start with "Adam: ..."
- insufficient reproducibility due to missing source code

### Questions
- ad interpretability claim: how does the interpretability of the presented method relate to the interpretability of baseline methods, e.g., voyager? isn't the interpretability of the method compromised by the lack of interpretability of the used LLMs?
- the proposed method is quite complex as it consists of 4 modules consisting of several submodules each and the presented ablation studies seem insufficient to rigorously justify such a complicated system; so how was the method designed? how much inspiration was drawn from prior work? 
- Adam is claimed to be a "generalizable framework"; why not back this claim with a complementary application in another environment? 
- what are possible limitations of the method? (the paper does not mention any)
- what do the error bars in tables 1, 2, 3 mean?

### Soundness
1

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
This paper introduces ADAM (An emboDied causal Agent in Minecraft) - an agent architecture that autonomously explores, learns causal world knowledge, and executes complex tasks in Minecraft from multimodal inputs. The system consists of four main components: interaction module, causal model module, controller module, and perception module. The interaction module samples actions and records observations. The causal model module infers causal relationships and constructs causal subgraphs for each action.
The key innovation is integrating causal discovery methods with embodied exploration, enabling the agent to learn accurate causal relations from scratch without relying on prior knowledge.

### Strengths
1. The incorporation of causal discovery methods in a modular framework is a novel in LLM-based embodied exploration, and it does not rely on privileged information unlike prior work
2. The paper demonstrate strong empirical results with well-designed experiments, that led to significantly faster discovery of skills in Minecraft. Performance in modified environments where prior knowledge is invalid did not degrade performance too much demonstrates causal learning is indeed effective. Method also does not require meta-data.
3. The experiments are solid with multiple baselines and includes comprehensive ablation studies as well as detailed analysis of failure cases

### Weaknesses
1. One concern is whether ADAM scales with more complex world and causal graph for intervention-based causal discovery (CD).
2. Interestingly the paper proposes a multimodal agentic framework but all the baselines compared to are text-based frameworks. It would be good to have at least one multi-modal baseline, e.g. [1] as this is also cited by the authors.

[1] Wang, Z., Cai, S., Liu, A., Jin, Y., Hou, J., Zhang, B., ... & Liang, Y. (2023). Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models. arXiv preprint arXiv:2311.05997.

### Questions
See previous

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces an architecture for agents that play the game of Minecraft, based on combining Large Language Models with causal inference. Featuring different modules (e.g., planner, actor, perception), the method is based on inferring causal graphs related to the various crafting dependencies of Minecraft's technology tree, and on enabling an agent to use the knowledge about these dependencies for progressing in the game. The method is evaluated against similar LLM-based methods (albeit using a different action space).

### Strengths
- The method seems to be the first approach that combines casual inference with LLM agents in code-based action spaces, which is a potentially very important direction for future research.
- The method performs quite well for inferring causal graphs on Minecraft, and it seems to provide a way for agents to take advantage of those causal graphs.

### Weaknesses
- The presentation of the method is quite high-level and does not help the reader understand how the method actually works in practice. When the different "modules" are introduced, it is not clear a priori what they actually are. Are they just prompts and specifications to a GPT4 model? If so, it could be beneficial to show one of the prompts earlier, to guide the understanding of the rest of the paper.
- The comparisons in the paper are unclear, due to the choice of a specific action space that is different from the one used in previous work. Indeed, while the paper mostly discusses the "observation space" difference compared to the setting usually employed in reinforcement learning papers, one crucial difference is the one in action space. For instance, Voyager works in an extremely more high-level action space compared to DreamerV3. This is not accurately depicted in the current version of the paper.
- It could be surprising to observe that an off-the-shelf open model is accurately describe an observation to the level of providing enough information for an actor to take the optimal action, especially in an environment that is as visually rich as Minecraft. An ad-hoc evaluation of this specific capability would strengthen the paper.
- The method seems to be highly Minecraft-specific, and the paper does not extensively discuss how it could be generalized to other domains.

### Questions
- Would the method generalize to other environments? If so, what are the assumptions and requirements for the application of the method to a new environment?
- How does the method compare to approaches trained with reinforcement learning? Is it possible to train an agent with reinforcement learning on the same action space that ADAM uses?
- What is the captioning performance of the perception module? What are its failure cases?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper introduces ADAM, an autonomous agent for open-world environments like Minecraft that builds a causal graph from scratch to improve interpretability and performance without relying heavily on pretrained knowledge. Through a combination of interaction, causal reasoning, planning, and multimodal perception, ADAM outperforms existing agents in task success and adaptability, even in modified game settings. ADAM’s approach establishes a new standard for causal reasoning in embodied agents.

### Strengths
1.	The figures in the paper are well-done and enhance clarity, making the content easier to understand.
2.	The experiments conducted in Minecraft show a higher success rate than those achieved by Voyager.

### Weaknesses
1.	The paper appears to be hastily prepared, as it contains numerous typos and minor errors, such as inconsistencies between “Fig.” and “Figure” references and improper usage of quotation marks in Table 4’s caption. I recommend that the authors carefully review and correct these issues.
2.	The major issue lies in the extensive use of pretrained language models that already incorporate substantial knowledge of Minecraft. Since language models may internally form a comprehensive causal graph of the game world, primarily in linguistic form, the proposed additional causal graph construction might be redundant. I suggest that the authors explore scenarios with completely altered world rules in Minecraft to test the validity of models like GPT in such modified environments, perhaps using a setting like “Mars.” Alternatively, they could consider using a language model entirely devoid of Minecraft knowledge, though this may be challenging to achieve.
3.	The agent’s modular design is nearly identical to Voyager, with the primary addition being the causal graph. Experimentally, however, it does not show significant advantages over Voyager, as it does not complete tasks that Voyager was unable to.
4.	The causal graph generated by the model ADAM is quite similar to the hybrid knowledge graph in memory described in [1]. The authors should clarify the differences.
5.	Additionally, the current causal graph is entirely object-centric. In open-world Minecraft, there are many open-ended tasks, such as building and farming, which are not strictly object-centric. This limitation restricts ADAM’s generalization capability in open-ended tasks.
6.	Several relevant works are not cited, including:

	[1] Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks
 
	[2] Mars: Situated Inductive Reasoning in an Open-World Environment, NeurIPS 2024
 
	[3] OmniJARVIS: Unified Vision-Language-Action Tokenization Enables Open-World Instruction Following Agents, NeurIPS 2024

### Questions
See the weakness.

According to the author's reply, they refuse to admit the unfair comparisons they made during the rebuttal stage. I even doubt whether the author carefully reviewed the responses from all reviewers. Therefore, I have decided to change the score to strong reject. 

I strongly recommend that the author carefully review the reviewer's comments and provide a serious response.

### Soundness
2

### Presentation
3

### Contribution
2
