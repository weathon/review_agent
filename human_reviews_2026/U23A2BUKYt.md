# Learning to Orchestrate Agents in Natural Language with the Conductor

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
Powerful large language models (LLMs) from different providers have been expensively trained and finetuned to specialize across varying domains. In this work, we introduce a new kind of Conductor model trained with reinforcement learning to automatically discover powerful coordination strategies among LLMs. Our Conductor learns not only to design targeted communication topologies for effective agent-to-agent collaboration, but also to prompt engineer focused instructions to the LLMs to maximally leverage their individual capabilities.  We show that, by learning optimal coordination strategies over pools of powerful worker LLMs, a 7B Conductor achieves significant performance gains beyond any individual worker, attaining state-of-the-art results in challenging reasoning benchmarks, such as LiveCodeBench and GPQA. By training with randomized agent pools, our conductor effectively adapts to arbitrary sets of open- and closed-source agents, meeting any user requirements. Furthermore, allowing the Conductor to select itself as a worker gives rise to recursive topologies, elevating performance with a new form of dynamic test-time scaling through online iterative adaptation.
More broadly, ours is among the early work demonstrating language model coordination can be unlocked through RL, where powerful coordination strategies emerge naturally in LLMs through pure end-to-end reward maximization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces the conductor, which can automatically divide the task and assign subtasks to different models, and a method to training the conductor with RL. With the conductor, small LLMs surpass large LLMs on multiple tasks.

### Strengths
- The concept of conductor is novel. The workflow is fully automatic and doesn't need human's design.
- The quantitive result is promising.
- The extra analysis is detailed.

### Weaknesses
N/A

### Questions
N/A

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
4

### Summary
The paper presents a major conceptual step: turning a small LLM into a meta-agent that learns to orchestrate larger, specialized LLMs through reinforcement learning. Instead of directly solving problems, the Conductor learns to design agentic workflows—decomposing tasks into subtasks, assigning them to specialized worker models, and defining how agents communicate and share context.
It establishes new SOTA reasoning results on tasks such as LiveCodeBench, GPQA-Diamond, and AIME 2025 and introduces a flexible, extensible framework for autonomous multi-agent coordination via language.

### Strengths
- While prior work (e.g., Mixture-of-Agents [Wang et al., 2024], or Smoothie [Guha et al., 2024]) explores routing or fixed multi-agent topologies, this work chooses to learn such coordination end-to-end using pure reward maximization, without predefined roles or handcrafted scaffolds.
- The idea of recursively calling itself as part of the agentic workflow is interesting and open pathways for further studies.
- The empirical results are strong, beating top frontier models.
- I like the in-domain and out of domain analysis. Really put things in perspectives. It is nice to know that Conductor is not just overfitting on specific task or formats.

### Weaknesses
```One concern I have is whether the benefits from Conductor came from orchestration or agentic workflow planning, or purely prompt engineering?```

The Conductor orchestrates the agentic flow with respective subtasks. And RL is used to train the model to do better at this task which can be roughly divided to two tasks: 1) manage and assign the best match between model and tasks. 2) create better subtask directions/prompts. There is no ablation that fixes 1), that is, use only one model (like GPT-5) and then train Conductor with RL. If Conductor is still doing pretty well, that means benefits from orchestrating is minimal. I would highly suggest doing such an ablation which would make your case a lot stronger. 

```The tasks themselves are not very multi-agent dependent```

Tasks like Math500, AIME or LiveCodeBench are not really suitable tasks for multi-agent. Ideally, a multi-agent task or a task where multi-agent can have some leverage or just single agent are tasks that naturally can be divided to subtasks that are heterogenous. I don't find these tasks to have those properties. Like do you write first part of the math solution and then write the second part using another model? I saw some examples where Conductor ask models to refine or verify. But those are not really something that a model have specialty in. The task selection is a bit off.

```Lack of other benefits```

To continue from last point. A weird task selection would cause the benefit of multi-agent to diminish. Multi-agent frameworks comes in with three big pros -- efficiency, safety and performance. There is no efficiency benefits for this because everything is sequential and one model has to wait for last model to finish (correct me if I misunderstood). This limits the contribution of the Conductor. I would love to see such a pipeline work in a meaningful agentic setup.

### Questions
Typos:
- Line 200 "to its own parent output"

Question:

- How sensitive is the Conductor’s learning to the choice of reward granularity (binary correctness vs. partial credit or verification-based rewards)? Would denser reward shaping improve stability or lead to different emergent strategies? What reward strategies have you guys tried? I am generally really curious about this. 

- What are the different modes of orchestration the Conductor learned? Did you guys do some simple categorization?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a conductor model trained with RL to "orchestrate" between different underlying LLMs to perform work. The conductor attains SOTA on reasoning benchmarks and outperforms all component models. The authors also enable online recursive adaptation of the conductor agent. This unlocks a new form of dynamic test-time compute scaling.

### Strengths
This is an excellent paper. The training setup is sound and the conductor is trained end-to-end with RL. This strategy unlocks new SoTAs on top reasoning benchmarks. The training setup also clearly observes novel strategies from the conductor (e.g., having sub-LLMs validate each other's outputs and plan with each other). There are a number of ablations to prove robustness and follow-ups to results, and many of them have interesting results of their own (e.g., showing the impact of spending more agents on harder tasks). Figures are clean. Examples are thorough.

### Weaknesses
It is unclear if these techniques improve performance compared to other test-time-compute strategies (e.g., pass@k, best-of-N, consensus, or other heavily prompted setups to increase test-time-compute put into solving the same problem). Because the incremental lift from this strategy seems relatively small, this paper would be significantly strengthened by including comparisons to other test-time-compute strategies to show that it's worth it compared to e.g., pass@k/BoN/cons/prompting/etc.

Moreover, it would be useful to understand "cost-normalized" performance; quantitatively, much more latency and cost is incurred from using this strategy and is it worth the performance gained? There is a plot with number of agent calls but latency in terms of time/tokens and cost in terms of actual API pricing or equivalent would be much more interpretable here to baseline the tradeoff needed to get the performance wins.

### Questions
What is the "non-trained" baseline (e.g., BoN/pass@)? What is cost-normalized performance?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes training a small (7B) Conductor model using reinforcement learning (RL) to orchestrate multiple strong LLMs for complex reasoning tasks. The model is trained using GRPO for 200 iterations. The experiments demonstrate that the Conductor achieves better performance on these reasoning tasks compared to any single model. Overall, the experimental results largely support the claim that the Conductor model improves reasoning performance in practice, although the margin of improvement is not always substantial.

### Strengths
- The core idea of using reinforcement learning to train a dedicated task planner/assigner (the "Conductor") is a potentially interesting direction to combine different models’ strengths.
- The experimental results demonstrate that this approach is effective, showing performance gains over individual state-of-the-art models.

### Weaknesses
- **The presentation has significant flaws.** Some expressions are imprecise, lack formal explanations, or are unsuitable for an academic paper. For instance, the GRPO formula in Equation (1) appears to be incorrect, as it seems to be missing the clipping mechanism characteristic of PPO-style algorithms. Also, the paper frequently uses the term "agentic workflows" without providing a formal definition within the context of this study. This term is often over-used, and the authors need to clarify precisely what it entails in their framework. Several other key terms are left undefined or used imprecisely. For example:
    - What do the authors mean by the "latent capability" of LLMs (line 37)?
    - What constitutes the "unconstrained" setting for evaluation (line 267)?
- **The motivation for training a separate Conductor model is not fully convincing given the experimental results.** While Table 1 and Figure 4 show that the Conductor improves upon the best single agent (model), the performance foundation clearly comes from the powerful frontier models it orchestrates (e.g., GPT-5 achieves 90.8 on unseen task AIME25, while Conductor reaches 93.3). The marginal gain seems small. This raises a critical missing baseline: what is the performance of a strong model (like GPT-5 or Gemini 2.5 Pro) when simply prompted to act as the task planner and assigner? Furthermore, the results for other multi-agent baselines in Figure 4 are confusing. Why do established frameworks like MoA and MASRouter show results inferior to a single-agent Gemini 2.5 Pro? Were these baselines tested with the same powerful set of worker models as the Conductor? This needs clarification to fairly assess the Conductor's contribution.

### Questions
- The reward mechanism is described at a high level. Could you elaborate on the credit assignment? Is the final reward applied to the entire sequence of tokens generated by the Conductor? How does this scalar reward effectively train the complex, multi-step workflow generation?
- The premise of the paper is that different models have specialized, complementary skills. Could you provide concrete examples of this from your experiments? For instance, are there cases where a generally "weaker" model (like Qwen3-32B) correctly solves a sub-task that a "stronger" model (like GPT-5) fails at, demonstrating true complementary specialization?
- How much more computational cost (e.g., inference rounds or total tokens) does the Conductor framework introduce compared to the single-model / single-agent scenarios? A clear analysis of the performance-cost trade-off is needed.

### Soundness
3

### Presentation
1

### Contribution
2
