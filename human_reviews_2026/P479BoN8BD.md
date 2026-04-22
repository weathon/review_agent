# CoMind: Towards Community-Driven Agents for Machine Learning Engineering

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Large language model (LLM) agents show promise in automating machine learning (ML) engineering. However, existing agents typically operate in isolation on a given research problem, without engaging with the broader research community, where human researchers often gain insights and contribute by sharing knowledge. To bridge this gap, we introduce MLE-Live, a live evaluation framework designed to assess an agent's ability to communicate with and leverage collective knowledge from a simulated Kaggle research community. Building on this framework, we propose CoMind, an multi-agent system designed to actively integrate external knowledge. 
CoMind employs an iterative parallel exploration mechanism, developing multiple solutions simultaneously to balance exploratory breadth with implementation depth. 
On 75 past Kaggle competitions within our MLE-Live framework, CoMind achieves a 36% medal rate, establishing a new state of the art. Critically, when deployed in eight live, ongoing competitions, CoMind outperforms 92.6% of human competitors on average, placing in the top 5% on three official leaderboards and the top 1\% on one.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CoMind, a multi-agent LLM system for ML engineering that integrates external community knowledge. It also proposes MLE-Live, a framework simulating Kaggle-style community environments with public code and discussions. CoMind uses five specialized agents (Coordinator, Analyzer, Idea Proposer, Coding Agent, Evaluator) to iteratively generate, refine, and evaluate ML solutions. The authors evaluate it on 75 past Kaggle competitions (MLE-Bench) and 8 live competitions. It achieves a 36% medal rate, better than some other open-source baselines (R&D-Agent, ML-Master, AIDE). In live tests, CoMind ranks in the top 10% on average, outperforming the majority of human competitors. Ablation shows strong dependence on public resources, removal causes major drops in submission success and performance. It generates longer, more complex code than baselines, which could mean a focus on exploration but potentially lower efficiency.

### Strengths
The paper's strenghts include the MLE-Live setup with realistic access to public resources. They show strong empirical results on both retrospective and live tasks, with a clear system design, their multi-agent division of roles aligns with human research workflows. The results are there, demonstrating practical success in real Kaggle leaderboards.

The results overall on Kaggle are strong enough that I am leaning towards accept, but I still feel like their approach could be overfitting to some Kaggle property and urgently needs more validation to show its generalisability on another, similar type of challenge (see weaknesses)

### Weaknesses
The weaknesses include the efficiency tradeoff with CoMind producing longer and more complex code, with higher compute/time cost.
I would have liked to see a more thorough analysis with limited tokens, and of runtime efficiency. As such, the scaleability is not really clear to me. It may be too heavily dependent of Kaggle-specific properties, it would have been good to include (even if in a limited manner) some other, similar use case as well, simply to avoid overfitting on some weird Kaggle idiosyncracy. Some of the improvements over baselines are rather modest given the additional architectural complexity.

### Questions
Can you provide more information on the runtime and computational costs per competition (tokens, wall-clock, GPU hours)?

Does CoMind's higher code complexity correlate with actual performance gains or just verbosity?

How would the framework generalize beyond Kaggle-style settings (e.g., real ML research or open-ended tasks)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper CoMind introduces MLE-Live, a benchmark designed to evaluate agents’ ability to leverage shared knowledge in realistic, community-driven Kaggle-style settings, and proposes a multi-agent system that simulates and contributes to such a research community. CoMind comprises five specialized agents—Coordinator, Analyzer, Idea Proposer, Coding Agent, and Evaluator—that iteratively synthesize ideas, implement pipelines, and refine solutions by integrating public discussions and code. Experiments on 75 Kaggle competitions from MLE-Bench show that CoMind achieves a 36% medal rate, surpassing prior systems like Neo and R&D-Agent, and in eight live competitions, it ranks within the top 7.35% on average, outperforming 92.6% of human competitors. The study demonstrates that community-augmented collaboration significantly enhances autonomous ML engineering performance.

### Strengths
- The paper introduces the novel concept of community-driven evaluation and agent collaboration in machine learning engineering. MLE-Live and CoMind simulate social learning and information exchange, a key feature of real-world research. I think this is quite meaningful.
- The paper is well-organized and readable.
- Its strong empirical gains and reproducible open-world setup could influence future benchmarks and frameworks in agentic AI, AutoML, and collaborative reasoning.
- I like the ablation. One downside of the main table is that the base model is all over the place. The ablation helps to put things in perpsective.

Overall I think the paper has its merit. A MLE benchmark that includes discussion, notebooks can be a useful asset for future agentic research.

### Weaknesses
```Gain came from the community data```

First I want to make the point that I believe a MLE-Bench with public resource is good. I have nothing against that. This weakness is mainly concern about the scaffolds proposed. In figure 4 there is the ablation where CoMind w R is much better than CoMind w/o R. CoMind w/o R is roughly equivalent to AIDE with the same resource and backend model. The improvement from scaffold itself is constrained. It would be great to see more evidences that CoMind is a better scaffold than others. 

One evidence I think would be helpful is to actually use the same model (o4-mini) for the main table or at least for a couple top baselines in the main table. This is to eliminate the factors of backend llm. Another thing is to let other agents to have some kind of overall view on the public accessible resources. The current setup in the ablation study has a problem that top voted notebooks may not be the most helpful/effective solution. Having that ablation would greatly help the scaffold's case. 

```More evidences of collaborative reasoning```

What reasoning modes do they workflow commonly show? Does it take the strongest notebook and improve upon it? Making those clear is crucial.

### Questions
- Is the benchmark going to be public for people to use soon?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes MLE-Live, a live evaluation framework that can evaluate AI agents to gather knowledge from a simulated Kaggle research community. The paper then proposes CoMind, a multi-agent system that can combine external knowledge. The CoMind simulates a Kaggle community, including a coordinator, an analyzer, an idea proposer, and an evaluator. The paper compares against multiple SOTA agent frameworks. The paper also includes an ablation study, qualitative and quantitative analysis.

### Strengths
1. The paper provides a live evaluation framework, which contains not only competition, but also discussions and kernels. The paper cut the data before the competition deadline to mitigate data leakage.
2. The CoMind has an analyzer to distill knowledge from community artifacts. The paper also evaluates the eight Live competitions. Apart from quantitive analysis, the paper has some qualitative analysis on task category, winrate, and code complexity. 
3. The paper provides the code and its model. In the appendix, the paper shows training details, additional evaluation details, and detailed case study.

### Weaknesses
1. The evaluation metrics follow the standard MLE-Bench for the main table. The ablation study only reports the win rate. I did not see any metrics directly related to the collaborative nature of agents. The novelty of MLE-Live seems to be limited.
2. Some parts of the paper are not clear. For example, what are the tasks included in the MLE-Live? The paper did not list any statistics or any details of those competitions. The paper also seems to categorize the tasks into three levels. However, it is unclear how those levels are determined, whether through the complexity of tasks or datasets. The external knowledge shared in the MLE-Live is also unclear. What is the average length of a discussion? What are their topics? The main architecture is very high-level and purely prompt-based. The paper also fail to provide cost analysis.
3. Compared to other benchmark papers, the evaluation is pretty limited. For example, in MLE0Dojo https://openreview.net/pdf?id=5W5mFU4oMO, they evaluate eight different backbones and provide task difficulty analysis for different domains. Using code length to evaluate code complexity instead of human annotation seems to be too coarse. The paper also fails to include any error analysis for the failure case.
4. The paper fails to include a use of LLMs section. The evaluated closed source LLMs are also not SOTA>

### Questions
See weakness

### Soundness
3

### Presentation
2

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
This work tackles the problem of automating machine learning engineering. Differently from most existing work, the authors take a community-driven approach and setup where AI agents can perceive real-time information and knowledge from external (human) communities. To this end, they first present their own framework, named MLE-Live. Building on top of MLE-Bench, MLE-Live simulates a live stream of publicly available information, including source codes and discussions (as for Kaggle competitions), namely "community artifacts," which are based on the pieces of information that were uploaded before the competition deadlines. In addition, the authors propose CoMind, a multi-agent framework that leverages such human inputs for machine learning engineering tasks and is made of five sub-agents with different roles (Coordinator, Analyzer, Idea Proposer, Coding Agents, and Evaluator). Evaluating CoMind on MLE-Bench (or MLE-Live) and ongoing Kaggle competitions, the authors present that it outperforms the baselines in most settings.

### Strengths
1. Novelty of the main idea  
The idea of incorporating real-time information from external communities for machine learning engineering agents is somewhat novel and interesting. In addition to machine learning competitions, CoMind has the potential to be leveraged for performing collaborative engineering tasks where human inputs are provided as real-time guidance, and this could be a more important potential application of CoMind.

2. Presentation  
Overall, the manuscript is easy to follow and understand, primarily due to the fact that it contains intuitive figures as well as clear tables. Also, the authors show fair details of the experimental results, including statistics, prompts used, and examples of task completion.

### Weaknesses
1. Comparison across MLE-Live and MLE-Bench  
While it is an interesting idea to take inputs from external communities, the community artifacts may contain too many hints. Although I think it makes sense to utilize the information under fair settings (such as comparisons with other Kaggle participants), comparing CoMind with other baselines on MLE-Bench (Table 1) is unlikely to be an apples-to-apples comparison. I'm aware that there are the results with CoMind but without the resources, but presenting the CoMind results as the primary finding could be misleading.

2. Reliance on and potential vulnerability to information from external communities  
On the other hand, it may also pose a concern that such information from external communities might harm the AI agents and their progress. One of such cases would be inaccurate information; even though the information is from humans, it is still possible that some information is inaccurate or noisy. Another possibility is getting attacks with specific intentions to hinder the participating automated research agents' progress, where such attacks may come from either humans or other AI models. Having defense against them may be important for the proposed approach to be properly launched for real-world competitions or tasks.

3. Minor issues
- I believe the other entities in the MLE-Bench leaderboard have the error range in addition to the scores, whereas the error range is not presented by the authors.

### Questions
1. How did CoMind's scores improve as it iterated during the "ongoing" Kaggle competitions?

### Soundness
2

### Presentation
2

### Contribution
2
