# Training-Free Group Relative Policy Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Reinforcement Learning (RL) has emerged as a pivotal strategy for adapting Large Language Model (LLM) agents to specialized domains and complex tool-use scenarios. However, existing approaches typically instantiate the policy as a parameterized LLM, relying on gradient-based updates such as Group Relative Policy Optimization (GRPO). This paradigm incurs prohibitive computational costs and risks catastrophic forgetting, often making it impractical for resource-constrained scenarios. In this work, we propose a fundamental rethinking of agentic RL by introducing Training-Free Group Relative Policy Optimization (Training-Free GRPO). It instantiates the policy as a frozen LLM paired with a variable experiential context, shifting optimization from the parameter space to the context space. Mirroring the iterative structure of vanilla GRPO, our method replaces gradient descent with multi-epoch RL learning by introspecting on groups of trial-and-error rollouts, where the LLM extracts a semantic group advantage to iteratively refine its problem-solving experiences without parameter updates. Experiments on mathematical reasoning and web search tasks demonstrate that Training-Free GRPO establishes a new Pareto frontier between test-time performance and learning cost. Also, we show that applying our method to a frozen flagship LLM like DeepSeek-V3.1-Terminus using merely 100 training samples yields superior performance to fully fine-tuning a 32B LLM, while slashing learning costs by orders of magnitude from 800 dollars to 8 dollars. It offers a highly effective and accessible pathway for optimizing LLM behaviors in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes Training-free GRPO, a method that keeps model weights fixed while introducing an external experience library to update experiences. In each round, groups of rollouts per query are sampled, and the LLM summarizes winners and losers to extract a natural-language “semantic” advantage. This information is then used to update an external experience library, which is later fed back as a token prior in subsequent API calls. Essentially, the authors introduce a way to build an internal RAG system via prompt engineering and demonstrate that such an internal RAG can improve model performance on AIME and WebWalkerQA.

### Strengths
- The idea is clear and could work under specific model setups.

- The overall paper is well written, with clear motivation and rationale.

- The method is practical and can be replicated by the community.

### Weaknesses
- Please reconsider how you classify your method. Labeling it as "RL" is misleading, given that the model parameters remain frozen and the reward mechanism is not clearly defined. A more appropriate framing would be under prompt-based RAG or inference-time memory optimization. Additionally, since there is no performance comparison with RL baselines, labeling your method as "RL" in all tables is inappropriate. Finally, emphasizing only the training cost savings without comparing performance against GRPO is insufficient and could weaken the overall claim.

- The experience may accumulate over time, eventually exceeding the model's context window, which suggests that this method does not scale well. Moreover, the proposed framework appears unsuitable for training smaller models with limited context and rollout lengths; for example, Qwen-Math with only a 4K context window.

- Only a limited set of baselines is compared, as ReAct is the only method evaluated under the same base model and training configuration. Moreover, the results presentation and comparisons are **improper and unfair** (see Q1 and Q2 for details).

- There is a limitation in the benchmarking setup, as only one benchmark from the math domain (AIME) and one from the web domain are presented. This does not sufficiently support the paper’s claims or demonstrate the effectiveness of the proposed method through empirical evidence. It is recommended to include results on at least two classic math benchmarks, such as **AMC23**, **Minerva**, or **Olympiads**, to strengthen the evaluation. Additionally, regarding the web domain, it would be helpful to clarify whether your method can be applied to more actual, real-world web tasks such as **WebArena** or **WebShop**, as WebWalkerQA is quite limited to reflect the actual web browsing & exploring ability.

- The idea and framework is similar to previous work such as Reflexion and Self-Refine, even though it's discussed.

### Questions
[Q1] Comparison in Table 1 is improper, you are not doing your method on Qwen2.5-32B-Instruct, but why you put those results and compare with your method on DeepSeek-V3.1?

[Q2] Comparison in Table 5 is also unfair, please avoid comparing your method towards the other methods on different base models. Make sure they are trained on the same base model with the same training dataset configuration.

[Q3] How would it work on smaller model? Please report training result over model less than 10B.

Please also addresses the concerns in weakness part.

### Soundness
2

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
The paper proposes Training-free GRPO, an iterative method that builds an experience library by comparing outputs of a base model against each other. Instead of updating model parameters, it refines behavior through accumulated experiential knowledge, offering a lightweight alternative to RL-based fine-tuning. The approach is well-motivated and demonstrates how iterative prompt optimization can achieve competitive results with lower computational cost.

### Strengths
The paper explores the idea of iterative prompt optimization as an alternative to RL-based fine-tuning, aiming to reduce the need for costly online samples. The motivation is clear, and the comparison between parameter-space and context-space optimization is conceptually interesting. The experimental setup spans multiple domains and shows consistent improvements across tasks, demonstrating the method’s practicality. The overall structure and writing make the paper easy to follow, although some implementation details are missing.

### Weaknesses
### Major comments:

- The method appears heavily dependent on the underlying model’s inherent reasoning strength. For instance, results on Qwen2.5-32B-Instruct are missing in Table 1 but appear in Table 4 (WebWalkerQA), where they show minimal improvement. While the authors briefly acknowledge this at the end of Section 3, it should be emphasized further and discussed in more depth.

- From Table 2, access to ground-truth results seems quite important for the improvements. It would benefit readers to explicitly explain how the ground truth is integrated into the offline dataset in Section 2.

- Table 5 can be misleading: since the base models differ, cross-model comparisons are not meaningful. The relevant comparison should be across domains rather than models. Moreover, Training-free GRPO leverages specialized experience libraries, making the claim of “cross-domain generalization” less convincing. A more rigorous evaluation (e.g., testing experience libraries trained on one domain and applied to another) is needed to support such claims.

### Minor comments:

- Since the optimization happens on the prompt level rather than the policy (usually refered to the base model) itself, the name “GRPO” may not be the most precise.

- How does the base model retrieve relevant experiences from the library? In Figure 6 it appears that the full library is appended to the prompt, whereas in Figure 11 each agent trajectory uses specific experiences. Clarifying this mechanism in Section 2 would make the method much clearer.

- Related work is missing prior studies on offline experience or guideline construction, such as:

[1] Fu et al., AutoGuide: Automated Generation and Selection of Context-Aware Guidelines for Large Language Model Agents;

[2] Wang et al., Agent Workflow Memory.

- It’s reasonable that ablations on WebWalkerQA are done on a subset of tasks, but it is unclear why they are performed after only two epochs of experience optimization. This choice should be justified.

### Questions
- When generating $A_{\text{text}_i}$, does the LLM have access to other summaries in the group? From line 194 it seems not, but if so, how can it compute a relative semantic advantage?

- How are “self-generated experiences” produced. Do they follow the same group-comparison pipeline but skip the experience refinement stage?

- In Figure 4, AIME24 and AIME25 show opposite trends in Pass@32 performance across steps. What might explain this behavior?

- What happens if the optimization runs for more than three epochs (e.g., 5 or 10)? Would the experience library become more specialized, or would overfitting occur?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Training-free Group Relative Policy Optimization (Training-free GRPO), a method for enhancing LLM agent performance without updating model parameters. The approach leverages group-based rollouts to distill a natural language "semantic advantage" from successful and failed attempts, which is then used to iteratively build and refine an external "experience library". This library is provided as context (a "token prior") to a frozen LLM, effectively steering its output distribution towards higher-reward behaviors in a data- and compute-efficient manner.

### Strengths
1. The central idea of shifting policy optimization from the parameter space to the context space is a practical concept. It presents a compelling alternative to traditional fine-tuning, directly addressing critical challenges such as prohibitive computational costs, data scarcity, and potential catastrophic forgetting associated with parameter updates.
2. The paper demonstrates the effectiveness of the proposed method with good empirical results on challenging and well-recognized benchmarks for mathematical reasoning (AIME24/25) and web navigation (WebWalkerQA). The application to a capable frozen model (DeepSeek-V3.1) shows significant performance gains, highlighting the practical utility of the approach.
3. The concept is explained with a helpful analogy to vanilla GRPO, and the inclusion of a concrete example in Figure 3 effectively illustrates the core mechanism of generating and applying "semantic advantage". This aids in understanding the intuition behind the method.

### Weaknesses
1. The experimental evaluation suffers from a lack of direct, controlled comparisons, making it difficult to isolate the true contribution of the proposed method. The primary results in Table 1 compare Training-free GRPO on the powerful DeepSeek-V3.1 model against RL baselines fine-tuned on the weaker Qwen2.5-32B model. To make a convincing claim, the paper should have included results for Training-free GRPO on Qwen2.5-32B and, more importantly, a traditional GRPO fine-tuning baseline on DeepSeek-V3.1. Without these crucial control experiments, the observed performance gap could be attributed more to the superior base model than to the merits of the training-free optimization technique.
2. The conceptual framing and terminology used in the paper are potentially misleading. The name "Training-free GRPO" is an oxymoron, as GRPO is fundamentally a training algorithm that updates policy parameters, which this method explicitly avoids. Furthermore, classifying the method as "RL" in Table 1 is questionable. Since the model parameters are frozen and improvement comes from engineering the context, the method is mechanistically closer to iterative prompt optimization or in-context learning rather than reinforcement learning in the traditional sense. This framing creates confusion and potentially overstates the connection to established RL algorithms.
3. The cross-domain transfer analysis in Section 4.1 is not designed as a fair comparison. The experiment shows that specialized fine-tuned models (ReTool, MiroThinker) perform poorly when transferred to a new domain, which is a well-known limitation. However, the Training-free GRPO method is evaluated by providing it with a domain-specific experience library for each task. A more rigorous test of transferability would involve using the experiences learned from the math domain to evaluate performance on the web search domain, which would properly assess how the learned knowledge generalizes, rather than just showing the flexibility of swapping context.
4. The ablation studies could be more comprehensive and the methodology more clearly defined. The paper states that semantic advantage is generated only for groups with "clear winners and losers," but this critical condition is not formally defined. The sensitivity to key hyperparameters, such as the group size $G$, is not explored. Furthermore, the term "reward model" used in Figure 2 is ambiguous; for verifiable tasks like math, this is typically a deterministic reward function, and using the word "model" incorrectly implies a learned, parameterized critic.
5. The important negative result on the weaker QwQ-32B model, where the method underperforms its own ReAct baseline, is a significant finding that is not sufficiently discussed as a core limitation. This suggests that the effectiveness of Training-free GRPO is highly dependent on the advanced reasoning and introspection capabilities of the underlying base model, which should be stated more explicitly as a prerequisite for the method's success.

### Questions
1. The paper states that semantic advantages are generated only for groups with "clear winners and losers" (line 189). Could you please precisely define this condition? Is it based on the variance of rewards, or is there another mechanism at play? How does your method handle the common scenario where all trajectories in a group fail and receive an identical low reward?
2. Could you justify the experimental design choice of not including direct comparisons on the same base model? Specifically, why was a traditional fine-tuning method like GRPO not applied to DeepSeek-V3.1, and why were results for Training-free GRPO on Qwen2.5-32B not reported? Is the observed performance advantage primarily due to the strength of the base model?
3. Could you clarify the conceptual framing of your method? Specifically, please justify the name "Training-free GRPO" given that no policy parameters are updated, and explain why it is classified as "RL" rather than an advanced form of prompt engineering, especially since other iterative methods like ReAct are categorized as "Prompt". Additionally, could you confirm whether the "reward model" is a fixed, deterministic function rather than a learned model?
4. Regarding the cross-domain analysis, would you consider running a more stringent experiment where the experience library learned on the math tasks is directly applied to the web navigation tasks without modification? This would provide a much clearer assessment of the generalizability of the learned "experiential knowledge".
5. The reference section contains several critical errors (e.g., the DeepSeek AI reference cites dates in 2025, the Li et al., 2025a and 2025b citations appear identical). We strongly urge the authors to perform a thorough revision of the bibliography. Could you confirm that these will be corrected?
6. Please describe the differences and connections between your proposed method and the ReasonFlux series of papers.
https://arxiv.org/pdf/2502.06772, https://arxiv.org/pdf/2506.18896, https://arxiv.org/pdf/2506.03136

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces training-free GRPO a method that mimics the traditional gradient updates of GRPO with contextual aggregation of trials.

### Strengths
The paper presents a decent idea with their aggregation mechanism for rationales and incorporates a self judging mechanism with the same LLM that seems to be fairly simple and efficient to implement. The presentation and plots in the paper a nice making it easy to read and understand. Overall I think the papers overall presentation and method are sound.

### Weaknesses
With the strengths being said, I think the paper has significant weaknesses that I think prevent me from recommending its acceptance. First, the marketing of the paper I have fundamental problems with, it is written in a way that makes it seem like through simple pseudo-prompting one can learn new things similar to what could be learnt through RL. I believe this to be simply not true, as if there are problems the model cannot solve (i.e the answer is not in its support) no amount of reasonable prompting (except maybe leaking the answer) should be able to obtain this sample. Now test-time scaling/online ICL does work, but I am reluctant to draw an equivalence between it and RL. 

With that said,  I think the paper suffers from further technical weaknesses. It is a bit weird to be that training-free GRPO is implemented for Deepseek V1 but RL is not implemented for the same model, only for Qwen 32b. Since this is ultimately a psu-prompting work I would expect more complex agentic pipelines being compared, I am unfamiliar with that side of the literature but something akin to [1].

Multiple random seeds and multiple models are also missing. Overall I think the work lacks significant polishing. 


[1] https://arxiv.org/abs/2509.26626

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
