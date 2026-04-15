# Evolve: Evaluating and Optimizing LLMs For Exploration

- Decision: Reject
- Scores: 8, 5, 8, 5

## Abstract
Despite their success in many domains, large language models (LLMs) remain under-studied in scenarios requiring optimal decision-making under uncertainty. This is crucial as many real-world applications, ranging from personalized recommendations to healthcare interventions, demand that LLMs not only predict but also actively learn to make optimal decisions through exploration.
In this work, we measure LLMs' (in)ability to make optimal decisions in bandits, a state-less reinforcement learning setting relevant to many applications. We develop a comprehensive suite of environments that include both context-free and contextual bandits of varying task difficulties to benchmark LLMs' performance. Motivated by the existence of optimal exploration algorithms, we propose efficient ways to integrate this algorithmic knowledge into LLMs: by providing explicit algorithmic guided support during inference; and through knowledge distillation via in-context demonstrations and fine-tuning, using synthetic data generated from these algorithms.
Impressively, these techniques allow us to achieve superior exploration performance with smaller models, surpassing larger models on various tasks. We conducted an extensive ablation study to shed light on the different factors, such as task difficulty and data representations, that influence the efficiency of LLM exploration. Additionally, we provide empirical measurements on the convergence rate of different exploration strategies introduced.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper examines the ability of large language models (LLMs) to perform decision-making tasks. In particular, it is focused on Multi-Armed Bandit (MAB) and Contextual Bandit (CB) problems. The paper introduces BanditBench, a benchmark suite for evaluating large language models in decision-making tasks within bandit environments. It also proposes two approaches to enhance LLM exploration: inference-time algorithmic guided support and algorithmic distillation through in-context demonstrations and fine-tuning using synthetic data generated from optimal algorithms. Results show interesting behavior of LLM-agents in bandit tasks.

### Strengths
- Addresses an important area of LLMs in decision-making tasks: this paper faces a very timely topic. LLM agents are an important research direction that has recently seen a surge in popularity. New research in this area is fundamental in order to better understand the behavior of LLMs when they face decision-making problems under uncertainty. 

- New benchmark: The paper introduces BanditBench, which is a novel benchmark for evaluating LLM exploration abilities. A benchmark in this research area is fundamental. Many papers in this area have different experimental settings. This makes it hard to compare them and for the whole research community to make reliable progress. For this reason, a benchmark on LLM agents is fundamental.

- Empirical evaluation: The paper also conducts comprehensive empirical evaluations and ablation studies on the proposed benchmark. I think that these results are interesting for the research community.

### Weaknesses
- Lack of novelty in some of the contributions: While I believe that BanditBench is a great contribution, the other claim of this paper is: "[...] we propose methods to enhance LLM’s decision-making capability by leveraging optimal algorithms, including algorithmic guided inference-time support and algorithmic distillation approach". The proposed approaches, however, seem to lack of novelty. 
In particular, the technique that the paper calls "Optimal Behavior Fine-Tuning" seems to be exactly what is known in the literature as Behavioral Cloning. "In-Context Few-Shot Demonstration" instead is a sort of in-context behavioral cloning. 

Did not influence the score, but I feel that it may be useful to the readers:
- Related work: In this paper, the authors analyze LLM agents' performance in decision-making and how they deal with uncertainty and exploration. There are some recent papers in this area that feel very relevant: 
  - Controlling Large Language Model Agents with Entropic Activation Steering, Rahn et al., arXiv 2024. This paper investigates exactly the bandit scenario with LLM agents and tries to improve exploration with activation steering using the entropy at the representation level. 
  - On the Importance of Uncertainty in Decision-Making with Large Language Models, Felicioni et al., TMLR 2024. Also this paper studies LLM agents in the (contextual) bandit scenario, but it does it by creating a new final layer on top of the pre-trained LLM and uses various approaches to approximate the Bayesian posterior to implement Thompson Sampling and improve the exploration capabilities of the LLM agent.

### Questions
- Is "Optimal Behavior Fine-Tuning" what is known in the literature as Behavioral Cloning? If so, please change the name in your paper. It can be confusing to a reader 

- Can the applicability of BanditBench be extended to other decision-making scenarios beyond bandit settings? Can you add some discussion about it in the paper (if you find some space, otherwise in the appendix)? I feel like recently LLM agents in more complex domains such as MDPs are very relevant and may be very useful in many real-world applications. Notice however that I believe that a BanditBench is absolutely needed, even if it is a simplified MDP version, because it allows to analyze more carefully the exploration-exploitation trade-off in LLM bandits.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This submission studies the problem of in-context exploration, where an LLM interacts with a bandit environment, and its history of observations and interactions with the environment are given in-context. The LLM agent then decides its next action based on this given context. Two forms of history are considered: raw history, in which the entire history is given in-context and summarized history, where summary statistics are pre-computed and given in-context instead. 

The authors call their framework BanditBench. They consider both stochastic multi-armed bandit and contextual bandit instances. For multi-armed bandits, they consider two action descriptions: choosing between different videos and different clothes. They also consider two reward distributions: Gaussian and Bernoulli. For contextual bandits, they construct their instances from the MovieLens dataset. The MovieLens dataset contains 10,000 real users’ movie ratings. In the constructed contextual bandit instance, the goal is to recommend a personalized movie that the specific user seen at the current round will enjoy. The LLM is given textual features, as well as numerical features taken from a low-rank approximation of each user’s rating matrix as the context in each round.  

The authors propose two mitigations to improve the exploratory behavior of LLMs in bandit tasks. Both methods leverage the behavior of optimal bandit algorithms. For the purposes of this submission, the optimal bandit algorithm considered is UCB for multi-armed bandits and LinUCB for contextual bandits. In inference-time algorithmic guided support (the authors’ first proposed mitigation), the LLM is given the explore/exploit components of UCB/LinUCB at each time step. (E.g. for UCB, this is the empirical average reward and the ‘exploration bonus’ for each arm.) For algorithmic distillation (the authors’ second proposed mitigation), UCB/LinUCB trajectories are given either in-context or via fine-tuning. 

The authors empirically evaluate Gemma-2B, Gemma-9B, Gemini 1.5 Flash, and Gemini 1.5 Pro on 16 multi-armed bandit and 2 contextual bandit tasks. They compare the performance of different models via pariwise win rate. They find that, perhaps surprisingly, few-shot learning boosts Flash’s performance while hurting Pro’s. They also find that fine-tuning significantly improves performance over few-shot learning, and leveraging inference-time support significantly improves performance across all models. Various ablations are also performed.

### Strengths
In-context reinforcement learning is an important and interesting problem, and multi-armed bandits & contextual bandits are an important building block in this direction. The authors propose several mitigations to improve the ability of LLMs to explore in these settings. Moreover, the paper is well-written and the multi-armed bandit experiments are comprehensive.

### Weaknesses
While the multi-armed bandit experiments are thorough, their novelty is somewhat limited as (as the authors point out), Krishnamurthy et al. 2024 study a very similar multi-armed bandit setting. While the multi-armed bandit results in this submission are more comprehensive, their findings are similar to Krishnamurthy et al.

The authors do include contextual bandit experiments (which are not present in Krishnamurthy et al.), but they are less comprehensive than the multi-armed bandit experiments. 

Finally, I am not fully convinced by the authors proposed mitigations. If we give LLMs things which make it easier for them to compute an upper-confidence bound, are we testing the LLMs’ ability to explore, or their ability to implement UCB? One reason why in-context exploration is interesting is because of the complex structure of real-world decision-making tasks. While it is natural to test LLMs’ exploration abilities on simple multi-armed bandit and contextual bandit tasks, we already have optimal algorithms for these domains and so deploying LLMs in such simple settings is not the end goal. Given that UCB is often suboptimal in structured bandit tasks beyond the two studied in this work, do you believe your proposed mitigations will extend to more complicated tasks?

### Questions
See above.

### Soundness
4

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
The authors develop the BanditBench benchmark, which evaluates LLMs' abilities to explore and converge to optimal actions through the multi-armed bandit framework. They comprehensively evaluate the suite of Gemma and Gemini 1.5 models and propose two techniques to boost the LLMs' exploration abilities further.

### Strengths
The paper is well-structured and easy to read. It extends the idea of Krishnamurthy et al. (2024) to contextual bandits, which is an important step for many practical applications.

The LLM evaluation methodology is sound and uses the MovieLens dataset, which I find a good fit for LLM exploration. I especially like the functional interpretation in Section 6, which allows us to compare LLM exploration capabilities to the established bandit algorithms, which clearly shows the LLMs are (unsurprisingly) lagging behind. This gives the paper a much stronger position, not overselling its ideas and showing the areas needed for improvement.

Overall, I think there are a lot of novel ideas, and provided the authors release the source code, the ICLR community can build on this.

---
Krishnamurthy, Akshay, et al. "Can large language models explore in-context?." arXiv preprint arXiv:2403.15371 (2024).

### Weaknesses
In MAB, I would like to see a setting with variable sigma for each action, as the exploration problem for the LLMs might get easier when all of the actions share the same variance.

I find the MovieLens dataset very simplified if the maximum number of actions is set at K=30 (see questions).

### Questions
1. Why is OFT in Figure 2 present only for Gemini 1.5 Flash?
2. Any idea how the LLMs perform in larger action spaces? I can imagine that many real-world applications go well beyond K=30, and any discussion on these scaling laws would be very helpful. This may not be intuitive as we would need to deal with issues such as limited context window and whether LLM can correctly synthesize the information from larger contexts.
3. Based on Figure 5, Gemma models perform terribly in exploration, even with all the techniques introduced in the paper. Do you have any explanation/hypotheses on why this is the case? Is it because of the model sizes?
4. How practical is it to use LLMs for such explicit exploration? If you have explicit actions, it seems easier to use RAG with UCB/Thompson Sampling baked into the external retrieval system, resulting in optimal exploration.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the ability of large language models to perform optimal decision-making under uncertainty through in-context exploration in multi-armed bandit and contextual bandit settings. This work introduces BanditBench, a comprehensive benchmark suite designed to evaluate LLMs in various bandit tasks. They propose two approaches to make use of bandit algorithms: (1) inference-time algorithmic guidance using established algorithms like UCB and (2) algorithmic distillation, where optimal behavior from algorithms is distilled into LLMs through few-shot demonstrations or fine-tuning. They also show the influence of different factors by conducting the ablation experiments.

### Strengths
1. The paper contributes to a relatively underexplored area by focusing on in-context exploration for LLMs in multi-armed bandit and contextual bandit settings. While LLMs are traditionally used for predictive tasks, this work broadens their application to optimal decision-making under uncertainty.
2.  The introduction of BanditBench provides a structured benchmark for evaluating LLMs in decision-making tasks that require exploration and exploitation. 
3. The proposed methods, including inference-time algorithmic guidance and algorithmic distillation, are well-motivated.

### Weaknesses
1. While the use of Summarized History (SH) and Algorithmic Guidance (AG) to enhance the exploration capabilities of LLMs is an intriguing direction, it is important to note that the results in Table 1 indicate that the application of AG in MAB scenarios does not yield consistent improvements and that its performance remains relatively low compared to traditional bandit algorithms (UCB, LinUCB). Additionally, employing AG introduces extra computational overhead. A more detailed discussion of the effects of AG would be beneficial for understanding its role more clearly.
2. The experimental analysis shows mixed results, especially in approaches for knowledge distillation with In-context Demonstration and Optimal Behavior Fine-Tuning for different model sizes and task difficulties. Specifically, in Figure 4, the results across various tasks and methods exhibit oddly similar numerical values (e.g., 0.487, 0.636, 0.267). A deeper investigation into the reasons behind these results could enhance the applicability of the proposed approaches in real-world scenarios.
3. The experiments are primarily focused on two specific domains (clothing and movie recommendations) with relatively small action spaces. It's unclear how well the proposed methods generalize to domains with much larger action spaces (e.g., thousands of items in real-world recommendation systems) or other decision-making problems where exploration could be more challenging due to the size and complexity of the task.

### Questions
Please see the weakness part.
1. Given that the results in Table 1 suggest that the use of Algorithmic Guidance (AG) does not lead to consistent improvements in MAB scenarios, could you provide further insights into the specific conditions under which SH and AG are most effective (especially compared with UCB or LinUCB)? 
2. Since the results in Figure 4 indicate that in-context demonstration performs better in some cases (e.g., Bernoulli Video and Summarized History) while fine-tuning is more effective in others (e.g., Bernoulli Clothes and Raw History), could you provide further analysis to help guide the selection of the most appropriate method in practical applications? Besides, could you clarify the numeric similarities observed in Figure 4?
3. How well do the proposed methods generalize to domains with much larger action spaces, such as real-world recommendation systems that involve thousands of items or more complex decision-making problems where exploration becomes more challenging due to the increased task size and complexity?

### Soundness
2

### Presentation
2

### Contribution
2
