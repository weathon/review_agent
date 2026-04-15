# Adapting Monte Carlo Tree Search for Generative Flow Network Training

- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
Generative Flow Networks, or GFlowNets, formulate generative modelling in discrete spaces as a sequential decision-making problem. Sampling plays a key role in GFlowNet training, as most algorithms use the learned policy to sample trajectories from the environment. Monte-Carlo Tree Search (MCTS) is a planning algorithm that has successfully been applied to train sequential decision-making models with reinforcement learning (RL). In this work, we leverage known connections between GFlowNets and maximum-entropy RL to adapt MCTS for GFlowNet training. We prove that standard MCTS tree construction processes can be modified to calculate the optimal flows for a GFlowNet, given sufficient samples from the environment. Our results extend to multiple cases of GFN modelling, including terminating-energy and intermediate-energy environments. We investigate practical strategies for employing MCTS as a sampling tool and apply it to different GFN parameterizations and training objectives. Through extensive experiments in a variety of discrete domains, including a language-based reasoning task, we show that our proposed method offers an improvement over standard on-policy sampling.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Authors propose Monte Carlo DAG Search (MCDS) — MCTS-like approach for planning in GFlowNets that can improve training efficiency. It stores a subgraph of the DAG environment with flow estimates for each edge. It is built iteratively and flow estimates are updated using their relation to the GFlowNet rewards and backward policy. A combination of the GFlowNet forward policy and the policy induced by MCDS flow estimates is used as a sampling strategy (denoted by $P_M$) to obtain trajectories for GFlowNet training. Authors claim it to offer improvements over on-policy training, and support their claims by providing experimental evaluation on three environments.

### Strengths
The paper is well-written and easy to follow. I had no problem understanding the motivation, the method and the experiments (apart from some specific details, see Weaknesses). 

GFlowNets have found success in various areas over the years, but the strategies for sampling trajectories for their training have not been extensively explored and studied yet. MCDS is a creative and promising approach that falls in this category.

While the idea of employing MCTS to improve GFlowNets has already been explored in the literature, the proposed approach has a number of crucial differences from the method of [1], as outlined by the authors, so I believe it to be a valid contribution.

References:

[1] Nikita Morozov, Daniil Tiapkin, Sergey Samsonov, Alexey Naumov, Dmitry Vetrov. Improving GFlowNets with Monte Carlo Tree Search. ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling

### Weaknesses
My main concerns lie within the soundness of the experimental evaluation.

Firstly, out of 3 tasks used for evaluation, 2 (hypergrids and factor graphs) environments are toy. By this I mean that the spaces of objects we work with are small enough  ($32^4$ for hypergrid and $5^{12}$ for factor graphs) that it is possible to iterate over all objects in reasonable time, thus the task of training a GFlowNet policy for sampling is artificial. Such environments can offer valuable insight into the performance of a method, but I still strongly advise the authors to include larger environments for the evaluation that are closer to real-world applications. A good example would be molecule generation task introduced in the seminal work [2], which is used as a standard benchmark in many GFlowNet papers.

Secondly, since the authors position the purpose of their approach to be improving trajectory sampling for GFlowNet training, I would like to see some baselines other than the standard on-policy training. E.g. replay buffer approaches [3, 4], and using a mixture of a forward policy and a uniform distribution [5], which were demonstrated to also improve the efficiency of GFlowNet training. As stated by the authors, combining MCDS with such methods could lead to further improvements, but putting some direct comparison to them is also crucial for understanding the efficiency of MCDS.

In addition, I have a concern over which policy is used for evaluation. A standard approach to calculate L1 metric on hypergrids is to use an empirical distribution of terminal states of the last $N$ trajectories sampled for training, and then compare it to the true reward distribution. If this approach is also used here, then the policy being evaluated would be not the GFlowNet forward policy $P_F$, but the combination policy $P_M$. Then, such evaluation would show the improvements coming not only from introducing a different distribution over trajectories for GFlowNet training (motivation stated by the authors), but also from the adjustment to the forward policy introduced by MCDS. Can the authors please provide more details on how the metric is calculated? For factor graph environment I have the same question. 

A positive note here is that I believe that MCDS could generally be used during inference of a trained model to introduce an adjustment to the forward policy (by using $P_M$ instead of $P_F$ during inference), potentially leading to better approximation of the reward distribution. Some experiments on this could further strengthen the paper.

It will also be good to include some runtime measurements to see how much of a computational overhead MCDS adds in comparison to standard on-policy training.

A minor remark, if $Q(s, s')$ in Equation 3 is an entropy-regularized action value function defined consistently with the standard RL notation, the correct way to write it would be $Q(s, s') = R(s, s') + \log \sum_{s''} \exp Q(s', s'')$. On the other hand, if it is defined as an exponent of action value function, I believe there should be no logarithm in the equation on line 187.

There is a typo "benchmarkpr" on line 57.

The paper presents a very interesting and promising approach, but the current quality of experimental evaluation is not enough to recommend acceptance to a conference of this level. However, I believe that with improved and extended experimental evaluation, this work would be a solid contribution to some future venue.

References:

[2] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio. Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. NeurIPS 2021

[3] Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani. Towards Understanding and Improving GFlowNet Training. ICML 2023

[4] Nikhil Vemgal, Elaine Lau, Doina Precup. An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets. ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling 

[5] Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, Yoshua Bengio. GFlowNets and variational inference. ICLR 2023

### Questions
1) Can you please clarify, are the metrics computed over trajectories sampled using $P_F$ or $P_M$ (see Weaknesses)?

2) How are flow estimates $\log F_D$ initialized for the newly added MCDS edges in the hypergrid experiment in Section 4.1 (when there are no neural networks)?

3) Do I understand correctly that MCDS DAG is cleared and rebuilt from scratch every few iterations of training? Could it be beneficial to keep the DAG from previous iterations, instead gradually increasing its size over the course of training?

4) Can the proposed approach be applied when the backward policy is also being trained?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors present a method that combines an MCTS-like algorithm with any GFlowNet training objective to enhance amortized sampler training. This MCTS-type procedure improves the quality of trajectories used during off-policy training, while the training itself can be performed using any existing GFlowNet method. The authors demonstrate the procedure's correctness and validate it through experiments on standard hypergrid and factor graph GFlowNet environments and a language-based reasoning task.

### Strengths
- The authors have proposed a new method based on the Monte-Carlo DAG search that can be combined with any GFlowNet training method, contrasting with a concurring MCTS-based method. In particular, this paper extends the work (Buesing et al. 2020) to the case of non-autoregressive generation, which seems non-trivial to me.
- All the theoretical proofs about correctness look valid to me.

Buesing, L., Heess, N., & Weber, T. (2020, June). Approximate inference in discrete distributions with monte carlo tree search and value functions. In International Conference on Artificial Intelligence and Statistics (pp. 624-634). PMLR.

### Weaknesses
- The theoretical statements seem to be a trivial corollary of the GFlowNet-RL connection by (Tiapkin et al. 2024) and (Deleu et al. 2024), applied to a GFlowNet problem over the subgraph. Thus, I cannot consider the theoretical contribution as a crucial part of this paper.
- Experimental validation: The hypergrid and factor graph environments seem limited in scale, as the final distribution over terminal states is tractable in both, which was used to compute the success metric. The only large-scale problem presented lacks sufficient details for reproducibility. Specifically, there is no information on how MCDS is applied in the language reasoning task. If MCDS is used on underlying structures rather than at the language level, it could be possible to construct an exhaustive or near-exhaustive DAG, simplifying the language reasoning task to an almost trivial problem. To assess whether this is the case, it is essential to know: (a) the size of the MCDS search and (b) the size of the underlying graphs for the reasoning task. The underlying GFlowNet graph appears relatively small, which could make MCTS-like techniques applied to these structures overly effective. Additionally, there are no comparisons with SFT and RLHF baselines for the reasoning task. Comparing only to an inference-level CoT baseline is not enough, as training data is used. It would be more informative to compare the CoT baseline with few-shot examples included in the training set.

### Questions
- Why are the results for TBVar significantly different from those in the original paper (Yu et al., 2024)? The original paper reports an accuracy of 98.41% for 4-step planning, whereas your results show approximately 39.5%. The same issue arises for 6-step planning.
- What is the size of the MCDS search in the factor graph environment?
- What is the reason for using an MCDS and sampling algorithm instead of MCTS and standard RL for the reasoning task? Additionally, what is the final GFlowNet reward function used for this problem? RL methods could address reward maximization directly, so it is unclear why a sampling-based approach is necessary.
- How many training steps or epochs were used for the reasoning task?
- How do you handle race conditions during the parallel construction of the graph by *W* parallel workers? Algorithm 1 outlines the sequential use of workers, yet the text mentions parallel execution (line 233).
- The paper may benefit from referencing existing results on Monte Carlo graph search (e.g., Leurent & Maillard, 2020).


Leurent, E., & Maillard, O. A. (2020, September). Monte-carlo graph search: the value of merging similar states. In Asian Conference on Machine Learning (pp. 577-592). PMLR.

Yu, F., Jiang, L., Kang, H., Hao, S., & Qin, L. (2024). Flow of Reasoning: Efficient Training of LLM Policy with Divergent Thinking. arXiv preprint arXiv:2406.05673.

### Soundness
2

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
3

### Summary
Generative flow networks (GFlowNets) are models for sequentially generating combinatorial objects. GFlowNets are trained by sampling trajectories from the network and optimizing some loss function with respect to the trajectory outputs. This paper proposes a Monte Carlo tree search (MCTS) -based approach to sampling trajectories for training GFlowNets. They apply this approach to different types of GFlowNets trained on different objectives and compare the performance against the conventional on-policy sampling strategy.

### Strengths
**Originality:** Whiles MCTS and GFlowNets are not novel ideas, the novelty arises in the adaptation of MCTS sampling to a domain in which it has not been used, namely, in sampling trajectories for training GFlowNets.

**Significance:** MCTS is a popular, easily-implementable algorithm known to improve sampling efficiency in planning domains and provide statistical guarantees on performance. Having an MCTS-based sampling method for training GFlowNets can be a quick-and-easy approach for improving on the performance of these models that can be adopted by anyone working with GFlowNets.

**Clarity:** Not being familiar with prior work in GFlowNets, it did not take too much effort to get the gist of how they work from this paper. That being said there are some concerns with the clarity (described in the Weaknesses.)

**Quality:** For the most part the approach for adapting MCTS to GFlowNets (which the authors call MCDS) is intuitively sound. The authors also chose a sufficient number of domains (one demonstrative and two complex) for conceptualizing GFlowNets (with and without MCDS) and measuring its performance.

### Weaknesses
While the paper presents an interesting and simple idea, there are a number of concerns with the work. The more concerning issues deal with the experimental evaluation.

1. The results for Hypergrid (Fig 3) do not appear meaningful:

    - The plot states that it shows mean and standard deviation however there no markers, fills or any visual indication of the standard deviation.

    - The standard deviation is not a statistical significance test --- it is a measure of dispersion. Confidence intervals would be an appropriate choice for visualizing statistical significance.

    - Without any statistical testing it is difficult to verify whether the difference in performances are significant. Hence, I believe the results do the support the authors' claims that MCDS improves the performance of GFlowNets over on-policy sampling in Hypergrid.

    - Secondly, inspecting the plots do not show that their is indeed any performance gain over conventional MCTS sampling.

    - Lastly, even if CI plots were produced it is hard to imagine that averaging over a mere 3 seeds would produce results of significance. With 3 seeds the authors may just plot the results over each seed.

2. Some of the issues from (1) similarly exist for the results in Table 1. The standard deviations are reported however this provides no indication of statistical significance.

3. The explanation of the Hypergrid domain in Sec 5.1 did not make any sense and there was no explanation of the performance metric. I had to read the original GFlowNet paper to understand it.

4. The explanation for MCTS in Sec 2.2 describes it such that all states in a MCTS-tree, and consequently MCDS-tree, are unique which is not necessarily true. In fact, if I understand the diagrams in Fig 2, the equivalent trees all will have duplicate states.

5. Additionally, in Fig 2, I cannot understand why the forward policy for MCTS is the way it is. If the reward function for MCTS is using the one shown in Eq 2, then if sampled enough, MCTS should terminate at the bottom left corner and observe a reward. Am I missing something?

### Questions
Line 57: "benchmark" has a typo

Line 157: What is $Q$? It hasn't been defined.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces the Monte Carlo DAG search (MCDS) algorithm, a modified MCTS method for sampling trajectories for GFlowNet training. The paper proves MCDS can find optimal flows for a GFlowNet given exhaustive samples from the environment. The algorithm is evaluated on three discrete domains, and some of the results show improved performance over on-policy sampling.


My recommendation for the article is reject for three main reasons:
1. Poor presentation of background and method.
2. Overclaiming about results of experiments.
3. Poor empirical practices and missing baselines.

### Strengths
1. The idea for MCDS is an interesting combination of ideas from MaxEnt RL and MCTS and the overall method is well motivated.
2. The problem of improving GFlowNets is relevant and well positioned by prior work.
2. Theorem 2 and its proof are novel and presented clearly.
3. The introduction is clear and well-written.

### Weaknesses
1. The background and method sections are poorly presented and are hard to follow. There is too much notation. The text is not coherent.

2. Some claims are not supported by the experiment results, which is exacerbated by poor empirical practices. For instance, line 411: "MCDS improves training compared to on-policy sampling" is supported by the learning curves in Figure 3. However, these curves are averaged over 3 runs, and the shaded regions (which are not visible) are standard deviations, a measure of dispersion and not confidence regarding the mean. There is little evidence that the difference is statistically significant. Standard deviation is not a measure of confidence and 3 seeds are insufficient to make such a general claim. The choice of hyperparameters is not justified. The reported metric is also not justified. The same criticism also applies to line 412: "Larger tree construction budgets providing a bigger improvement".

3. In general, the hyperparameter choices for no experiment are justified, and all claims regarding differences between methods in Hypergrid and Blocksworld are not supported by statistical tests (e.g. confidence measures).

4. While the experiments show comparisons with on-policy sampling, there is no comparison with other baselines that attempt to improve GFlowNet training, such as epsilon uniform exploration, replay buffers, or local search. This limitation is acknowledged in the paper, however, I think their inclusion is required to evaluate MCDS.

5. Minor edits:

  - Line 057: typo "benchmarkpr".
  - Line 325: repeated "performance performance".
  - The reported metrics should be motivated and introduced.

### Questions
1. What does the top plot in Figure 2 show? What do the lines represent? Why is this difference significant?
2. How were the hyperparameters selected? Why?

### Soundness
2

### Presentation
1

### Contribution
3
