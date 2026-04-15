# Query-Efficient Offline Preference-Based Reinforcement Learning via In-Dataset Exploration

- Decision: Reject
- Scores: 3, 8, 5

## Abstract
Preference-based reinforcement learning has shown great promise in various applications to avoid reward annotations and align better with human intentions. However, obtaining preference feedback can still be expensive or time-consuming, which forms a strong barrier for preference-based RL. In this paper, we propose a novel approach to improve the query efficiency of offline preference-based RL by introducing the concept of in-dataset exploration. In-dataset exploration consists of two key features: weighted trajectory queries and a principled pairwise exploration strategy that balances between pessimism over transitions and optimism over reward functions. We show that such a strategy leads to a provably efficient algorithm that judiciously selects queries to minimize the overall number of queries while ensuring robust performance. We further design an empirical version of our algorithm that tailors the theoretical insights to practical settings. Experiments on various tasks demonstrate that our approach achieves strong performance with significantly fewer queries than state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies improving preference-based RL in offline settings via in-dataset exploration. They proposed an algorithm, called OPRIDE. First, they show that this algorithm achieves statistical efficiency by establishing a bounded suboptimality. Then, they developed a practical version of this algorithm, and experiments show this approach outperforms other methods.

### Strengths
This paper is easy to follow. The description of the algorithm is clear, and the theoretical claims are also clearly stated.

The theoretical part looks sound to me, although I haven't checked the proof in detail.

### Weaknesses
This paper has 10 pages. I am not sure if it violates the paper length limit. Regarding its content, I have the following concerns.

I don't generally understand what "query-efficient" means in this paper, especially in the theoretical part (section 3.1). If I understand it correctly, the authors proposed an algorithm and showed that it achieves good suboptimality upper bound. Then how is small optimality related to "query efficiency"? I would expect "query efficiency" to have something to do with active learning. Otherwise it is simply "statistical efficiency".

Another concern is that there seems to be a big gap between the theoretical algorithm and the practical one (i.e. algorithm 1 vs algorithm 2). All theoretical results are developed for alg 2 while the experiments are conducted for alg 1. I think this is acceptable only when the gap between the two algorithms is small. However, I feel that the gap is large in my current understanding. For instance, alg 2 assumes a query oracle but alg 2 doesn't. Alg 1 uses an optimistic bonus, but no correspondence is found for alg 1. In addition, Alg 1 is heuristic and lacks some explanation. For example, why the loss functions in (14) and (15) are introduced is not explained (eg, why using expectile loss). The intuitive explanation of adding clipped Gaussian noise is not provided, which may be confusing.

### Questions
I didn't understand Tables 1 and 2. I am not sure what these numbers in the tables are. I would suppose they are the output of the learned reward model. However, this confused me even more since comparing the absolute reward value seems meaningless under the Bradley-Terry model (because the distribution will not change under reward shift).

The authors claim to use 10 queries in the experiments. I am not sure how complicated the experiment is, but 10 sounds really small. Could the authors explain more about it?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents OPRIDE, an offline preference-based reinforcement learning (PbRL) algorithm that selects trajectories to query based on a pessimistic-value, optimistic-reward approach. Additionally, OPRIDE uses random weights to augment the offline dataset.

Moreover, this paper presents a mathematical justification for using its pessimistic-value, optimistic-reward approach, setting a likely bound on the suboptimality of the policy value estimation.

Experiments in AntMaze and MetaWorld show OPRIDE beating recent baselines such as Preference Transformer. For instance, across 30 MetaWorld tasks and 5 runs, OPRIDE increases the average reward by 27%.

### Strengths
* As far as I could follow, mathematically robust motivation, with a bound on the suboptimality of the value estimation.
* The evaluation results are impressive, beating Preference Transformer in the MetaWorld tasks.

### Weaknesses
* _W1_: Though the individual sections are really well written, I had trouble following the paper. To mitigate this, I would suggest better delineating the article in the introduction, highlighting key take-aways in each section, adding a summary of Pessimistic Value Iteration (which OPRIDE builds on) to the main paper, and providing more examples. The mathematical analysis is very general (this is a strength of the paper), but providing examples of how it is instantiated for linear MDP with Bradley-Terry preferences would aid in comprehension.
* _W2_: There is still a question whether the same gains observed in MetaWorld and AntMaze apply to other environments such as mujoco-gym and perhaps from human preferences too. (Preference Transformer tested against all these environments)

**[[Post-rebuttal update]]**

The authors revised the manuscript, adding more "sign-posting" and repeating the motivations for each section. As a result the paper is easier to read, thus addressing _W1_. The manuscript remains rather mathematical, but that is the nature of the research.

As for _W2_, the authors conducted additional experiments with mujoco-gym, but OPRIDE surpassing or equalling Preference Transformer (albeit by a lower margin than in Metaworld).

### Questions
* _Q1_: In definition 2, the instantiation of $\phi$ for linear MDP with the Bradley-Terry preference model depends on another $\phi$. Is this a mistake? Or is the definition truly recursive? 
* _Q2_: What is $T$ on the theoretical guarantees section? Do you mean $K$ the number of queries?
* _Q3_: In figure 4, for `push-v2`, both baselines beat OPRIDE at K>15. Why is that? Is there a downside to OPRIDE as the number of queries goes up?

**[[Post-rebuttal update]]**

The authors addressed all questions and updated the manuscript accordingly. See discussion for more details.

**Nitpicks and suggestions**

* When introducing equation 5, explain that SubOpt is the sub-optimality measure, it will help readers later on.
* In equation 6, mention that $o$ is the ground-truth human preference.
* In section 3.2 $N$ is meant to use the number of reward functions, but $N$ was earlier used to indicate the size of the offline dataset.
* In "Answer to Question 1", _tables 3 and 2_, should be _tables 1 and 2_ instead.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose an algorithm for query-efficient offline preference-based RL. In this setting, the dataset of trajectories is fixed and the agent must choose pairs of to submit for a preference. 

First, they assume a linear transition and reward model. Building on top of the PEVI method, they construct a confidence set of reward functions using a projected MLE estimator. Given that confidence set, they construct a set of near-optimal policies. In order to explore maximally, they choose the pair of policies for which there are two reward functions in the confidence set under which these policies achieve maximally different values. They show that the suboptimality of the method is bounded by the sum of two terms: one that is O(1/sqrt(N)) in the dataset size and one that is O(1/sqrt(K)) in the number of comparisons.

Next, they give an implementable algorithm that works with a static offline dataset. It seems that they augment each trajectory L times with L sampled clipped Gaussians. They train N reward functions by a preference transformer method and compute an optimistic reward function using ensemble disagreement. At each round, the agent chooses the trajectories for which the weighted sum of reward predictions has the largest difference between 2 of the ensemble members. Finally, using IQL, they train optimistic V and Q functions and extract a policy. 

They evaluate the implementation on a set of Meta-World manipulation tasks as well as Antmaze. In each they make 10 pairwise queries prior to evaluation the policy performance. They compare against  a handful of recent methods for offline preference-based reward learning. In the experiments, it is clear that this method performs best on the set of manipulation + Antmaze benchmarks. Afterwards, the authors proform studies of the number of queries required to achieve good performance and an ablation where they remove the optimism and the selection method and see how that affects performance.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results. You can incorporate Markdown and Latex into your review. See https://openreview.net/faq.

The problem setting is reasonable and seems like a good way of supervising given large offline dynamics datasets and a hard-to-compute reward function. The theoretical setup seems reasonable and in line with prior work, and it seems like the bounds are interpretable. The theoretical algorithm seems broadly of the right flavor to solve a problem like this. 

Broadly speaking, the experimental section shows good results and some ablations and further empirical study is conducted. I don’t think there are yet many methods that address this problem and there is therefore not a wide list of methods to compare against.

### Weaknesses
I think the exposition and presentation of this paper is poor. I walked away from the first read without a new insight into the nature of preference based learning. I’m sure there is one to be had in this paper but the lack of writing to inform rather than to describe procedures makes it far more difficult to take one away. In particular:
- The motivation for the augmentation with a truncated Gaussian missed me entirely.
- The chain experiment is not described well in the text or the appendix.
- The justification for the use of projected MLE or even the definition of the projected MLE is missing from the method section. 
- There are a substantial number of typos and errors in the writing. 
- It's not clear to me what the counterfactual trajectories are. 
See questions for things I concretely would like to know in order to increase my score.

### Questions
At the end of 3.1 you mention that “querying with an offline dataset can be much more sample efficient when N >> T”. What is T here?
What is “True Reward” in table 1 / 2?
Can you explain why you augment the trajectories with a truncated Gaussian?
Why a truncated Gaussian rather than some other random variable?
Are we sure that an ensemble of reward functions is going to give reasonable uncertainty estimates of the preference model?  What if preferences have high aleatoric uncertainty?
How is the optimization problem in (13) solved?
What do counterfactual trajectories have to do with weighting?
How does this policy error bound compare to e.g. a G-optimal design and then PEVI? The Pacchiano et al result?
What are the steps in the x axis of Figure 3?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
