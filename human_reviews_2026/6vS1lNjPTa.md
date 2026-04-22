# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Large language models are powerful generalists, yet solving deep and complex problems such as those of the Humanity’s Last Exam (HLE) remains both conceptually challenging and computationally expensive. We show that small orchestrators managing other models and a variety of tools are able to both push the upper bound of intelligence and improve efficiency in solving difficult agentic tasks.
We introduce ToolOrchestra, a method for training small orchestrators that coordinate the use of intelligent tools. ToolOrchestra makes explicit use of reinforcement learning with outcome-, efficiency-, and user-preference-aware rewards. Using ToolOrchestra, we produce Orchestrator, an 8B model that achieves higher accuracy at lower cost than previous tool-use agents while aligning with user preferences on which tools are to be used for a given query. On HLE, Orchestrator achieves a score of 37.1\%, outperforming GPT-5 (35.1\%) while being 2.5x more efficient. On $\tau^2$-Bench and FRAMES, Orchestrator surpasses GPT-5 by a wide margin while using only about 30\% of the cost. Extensive analysis shows that Orchestrator achieves the best trade-off between performance and cost under multiple metrics, and generalizes robustly to previously unseen tools.  These results demonstrate that composing diverse tools with a lightweight orchestration model is both more efficient and more effective than existing methods, paving the way for practical and scalable tool-augmented reasoning systems. These results demonstrate that orchestrating diverse tools with lightweight agents is not only more efficient, but also more effective, paving the way for practical and scalable tool-augmented reasoning systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
ToolOrchestra trains a compact 8B Orchestrator to coordinate specialized tools and larger LLMs through RL that jointly optimizes correctness, efficiency, and user-preference adherence. It uses a unified tool interface and a synthetic ToolScale dataset to learn multi-turn reasoning and when/how to invoke different tools. On HLE, FRAMES, and τ²-Bench, the Orchestrator outperforms or matches frontier models at a fraction of the cost and generalizes to new tools while following user preferences.

### Strengths
The paper poses orchestration as “small model + tool” collaboration to beat frontier LLMs on accuracy per dollar and latency, and backs it with results on HLE/FRAMES/τ² showing best performance at the lowest cost. The authors trains the Orchestrator to jointly optimize correctness, efficiency (cost/latency), and user-preference alignment, as a more realistic objective than accuracy alone, and demonstrates that the 8B Orchestrator can adapt to new model/tool sets via textual tool descriptions, maintaining top performance under shifted tool menus.

### Weaknesses
(1) A large share of training comes from LLM-generated domains, schemas, tools, and gold action traces (ToolScale), which risks overfitting to stylized, clean environments and brittle tool descriptions rather than messy real-world APIs and noisy user goals.

(2) Although the pipeline claims “verifiable” tasks, they are verified against the same scripted environments the LLM produced; that loop can miss edge cases, security issues, or ambiguous specs that appear in production tools. 

(3) Ecological validity of “GPT-5 can call GPT-5.” The setup explicitly allows an orchestrator to invoke frontier generalist models (e.g., GPT-5/Claude) as tools; while interesting scientifically, it blurs fairness and deployability: real teams often can’t chain premium models for cost, terms-of-use, or data-governance reasons. The paper itself shows prompted models “self-enhancing” (e.g., GPT-5 → GPT-5-mini), which is a peculiar and potentially circular baseline.

(4) 8B isn’t tiny; cost still non-trivial. The “small” orchestrator is 8B parameters and needs training / data collection process.

### Questions
Please explain why 8B model is necessary for training this behavior instead of a smaller 1-3B model. And also GPT-5 'call' GPT-5 is quite awkward, they are not trained to do so, so comparisons are a bit unfair.

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
4

### Summary
This paper introduces ToolOrchestra, a method for training a small model called the Orchestrator to manage other models and tools to solve complex reasoning tasks efficiently. Instead of relying on one huge language model like GPT-5, the Orchestrator learns through reinforcement learning to decide which tools or models to call, balancing accuracy, cost, and user preferences. In experiments, the  8B Orchestrator outperformed GPT-5 on reasoning benchmarks like HLE while being over twice as efficient, showing that smart coordination of multiple tools can be more powerful and economical.

### Strengths
- The idea of orchestrating both tool and model is interesting. Previous works mainly focus on model router / tool selection problem, but seldomly combine them together into consideration, which is quite new and has real world application value.

- The reward design also considers user preference in the tool using problem, which is also very novel and valuable, as this may give user the control to steer the model to more use which tool instead of other ones. This user customization related training paradigm will be of great interest to many.

### Weaknesses
- The reward design writing part in section 3.2 is too crowded and unnecessarily introduce many symbols. Such as for user preference alignment, it’s just the reward’s projection on user preference vector, which is clear just in one sentence.

- The problem should actually be a user-tool-model-orchestrator four-sided decision making problem: the orchestrator needs to consider whether to align with user / call tool / call model / generate itself. The paper can further benefit by putting itself into this broad context of decision making (or decision alignment).

- The data creation process seems very artificial and the tasks seems are enhanced by rounds, which may make the task compounded instead of naturally / inherently difficult. In other words, difficult may stem from the task is complicatedly composed of multiple subtasks, instead of having one that is inherently challenging for orchestrator / LLM to perform reasoning.

- Analysis can be deeper instead of just presenting “good results”. For instance, In table 2 orchestrator’s performance drop 15 when the model pool changes while GPT-5 only drop 5 on HLE. Some stability / error analysis like this will further enhance the paper’s insights.

### Questions
- Will the orchestrator just use the knowledge of itself to answer the question instead of doing delegation? For example, the problem is very simple arithmetic, what will be your orchestrator’s behavior? Is “answer yourself” trained explicitly as an objective?

- In the reward design, how you unify the unit of everything such as cost, latency, etc? Also, how you balance the ratio of cost / correctness, etc. Sometimes correctness is more important than cost, as less money used to get a wrong answer is not preferred. Is this handled in the reward design? Why or why not?

- For the dataset filtering, what LLM is used to test its difficulty / quality? In reality did you discover any bias in the created data? (Such as some questions are too easy and do not need to be offloaded stronger reasoning model to do?)

- I am still wondering why orchestrator calling models other than GPT-5 can even give better performance than GPT-5’s own performance. This is like paradox as other model’s performance cannot reach GPT-5. Therefore, what’s exactly contributing to the success of orchestrator?

- There should be two baselines to compare: one is using off-the-shelf model as orchestrator to coordinate models and other tools, one is using model to directly solve the problem instead of relying on tools. Sometimes we can see the model is better at coordinating others, while other models are good at answer the problem just with basic tools themself instead of relying on other models. This might reflect the model’s inherent confidence in question answering. Why do you think this happens?

- The reasoning model as baseline should be tested. What’s the orchestration performance with low / high reasoning effort? Will extended reasoning benefit in your coordination setting?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ToolOrchestra, a framework based on reinforcement learning (RL) for training small orchestration models that coordinate tool and LLM usage in agentic reasoning. The proposed RL method aims to balance the accuracy, efficiency (cost and latency), and user preference alignment in agentic reasoning. The orchestration model, Orchestrator-8B, is trained on synthetic data generated by a proposed pipeline ToolScale, which produces verifiable multi-turn tool-use trajectories across 10 domains. Results on three complex reasoning benchmarks show that Orchestrator-8B coordinated with heuristic and LLM-based tools outperforms strong baseline models such as GPT-5 while being about 2.5 times more efficient.

### Strengths
- Besides the more commonly studied aspect of reasoning accuracy, ToolOrchestra also casts attention to improving efficiency and user preference alignment in agentic reasoning via RL, which are essential aspects valuable for test-time scaling-up and generalization of agentic systems.
- The idea of using vectorized weights to balance the agentic reasoning accuracy, efficiency and preference of tool usage is interesting and effective.
- The experimental analysis is comprehensive, which covers four different aspects that are well-aligned with the motivations of this work.

### Weaknesses
- The reward function of ToolOrchestra’s RL approach relies on close-sourced GPT-5 as a judge, which requires a relatively large amount of training cost on both money and latency. This reduces the efficiency value of the proposed method, and also limits reproducibility. It would be better to compare the GPT-5 judger with more light-weight and open-sourced reward models and mixing rule-based rewards on applicable tasks.
- The experimental study lacks comparisons to other RL-based tool integration frameworks such as ToRL cited in the paper, and also lacks comparisons to other prompting or fine-tuning baselines focusing on integrating broad toolsets, such as RestGPT [1] and ToolLLaMA [2].

[1] RestGPT: Connecting Large Language Models with Real-World RESTful APIs. Song et al., 2023.

[2] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. Qin et al., 2023.

### Questions
- Any additional experimental results to resolve the above weaknesses?
- The user preference alignment evaluation and analysis in Section 6.4 is interesting. Is there any controllability evaluation or case study to show that agent response could be flexibly altered when different user preferences are given at test-time inference?

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
The paper introduces ToolOrchestra, a method for efficiently training a small LM orchestrator to coordinate both models and tools in agentic tasks. The work frames tool/model usage as a sequential decision process, optimized with RL to account for outcome correctness, efficiency, and user preferences. Extensive benchmarking demonstrates the orchestrator's effectiveness, showing performance improvements over larger models (e.g., GPT-5) at substantially lower computational cost and better adherence to user preferences.

### Strengths
The work addresses a timely challenge about how to efficiently orchestrate diverse model and tools. The system is generally well-motivated from both cognitive and engineering perspectives.
The RL formulation is generally sound with explicit reward design of outcome, efficiency, and preference, considering different aspects comprehensively.
The analysis in Section 6 is thorough and provide interpretability of results, the figures, especially Figure 4, gives concrete insights into the orchestration patterns and further strengthens the depth of the paper.

### Weaknesses
The baselines currently lacks diversity, as there are also recent works aboutt rule-based / logic-based orchestrator, and many other router workers that potentially the paper can compare itself with。
The reward or optimization details lacks justification. There’s no ablation study on reward scaling / instability, etc. Also, how different reward component is exactly unified is under-explained, particularly how to consider them in a unified unit.
The data creation process is very artificial and heavily rely on human priors. The task created is also composed of several components that on purposely make the task difficult instead of each subgoal is inherently difficult itself.
The paper can further benefit from analyzing the robustness to adversarial instruction, tool failures, etc. in order to show the orchestrator’s true robustness. Also the trained model’s safety under distribution shift is critical to consider when deploying your trained orchestrator in the real world scenario.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
