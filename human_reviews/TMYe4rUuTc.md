# Evolving Alignment via Asymmetric Self-Play

- Decision: Reject
- Scores: 5, 3, 8

## Abstract
Current RLHF approaches for aligning large language models (LLMs) typically assume a fixed prompt distribution, which is sub-optimal and limits the generalization capabilities for language models. To address this issue, we introduce a general framework that casts alignment as an asymmetric game between two players:  (i) a creator, which strategically generates informative prompt distributions using reward signals, and (ii) a solver, which learns to produce preferred responses on prompts produced by the creator.

This framework of Evolving Alignment via Asymmetric Self-Play (`eva`), results in a simple and efficient approach that can utilize any existing RLHF algorithm. `eva` achieves a new state of the art in widely adopted alignment benchmarks, without the need of any additional human crafted prompts, e.g., it can improve the win rate of finetuned gemma-2-9b-it on Arena-Hard from 51.6% to 60.1% with DPO, from 55.7% to 58.9% with SPPO, from 52.3% to 60.7% with SimPO, and from 54.8% to 60.3% with ORPO, surpassing its 27B version and matching Claude-3-opus. Finally, we show `eva` is effective and robust under various ablation settings. 

We hope `eva` can serve as a scalable and easy-to-use methodology for the research community to build open-ended, robust, and self-improving language agents, that align with human values.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes EVA: Evolving Alignment via A symmetric Self-Play, a framework that that casts alignment as an asymmetric game between two players, a prompt creator and a policy learner. They selects prompts which induce responses with largest reward gaps, and generate similar prompts with those selected prompts. The proposed method improves the baseline by a margin.

### Strengths
This paper is well-written, easy-to-follow and informative; It contains thorough ablation study to show the effectiveness of the proposed method; The empirical evaluation shows much improvement.

### Weaknesses
**Main concerns**

1. The number of iterations in the main results is 2, with only one EVA step in each experiment, which is a little different from what the demonstration in Figure 3 shows. If the EVA step is performed multiple times, would the results be better or worse? What is performance like when you access all data in ultrafeedback?

2. The connection between the minimax regret objective and the algorithm is a somehow vague. The regret concerns the performance gap with the optimal policy. It's not reflected by the informativeness proxy.

**Minor issues**

Line 278: type wrost -> worst

### Questions
The informativeness proxy seems to be similar to the variance of the rewards because they all concern the diversity of the generated responses. However, in lines 393-395, the results shows using variance leads to poor performance. How to interpret this?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Majority of the recent alignment/RLHF pipelines have been designed on static prompt distribution leading to several issues of distribution shift on different prompt distribution. Hence to mitigate this issue, the current work formulates the problem as an asymmetric game where the creator generates novel and informative prompts whereas the solver solves it. This approach improves the performance of existing baselines on SOTA benchmarks.

### Strengths
The reliance of static prompt distribution has inherently limited the capabilities of various alignment designs eventually resulting in issues such as distribution shifts and even jailbreaks can be considered as a special case of the same. Hence, the problem statement is timely and crucial. The paper introduces a creator-solver framework evolving the prompt distribution which enhances the generalization ability of LLMs, allowing them to adapt to more diverse and challenging prompts, which better mirrors real-world applications. An additional advantage lies in the fact, the algorithm can be added on top of current baselines which is crucial rather than building new algorithm from scratch. The empirical results show improvements on existing benchmarks when applied on top of baseline algorithms.

### Weaknesses
1. How is Equation 10 tractable and how is it being solved? Any heuristic of sampling and approximating should result in sub-optimality which is not clear where its accounted.
2. The optimization is over \pi in equation 9 for solving the minimax regret. However, its not absolutely clear how the KL divergence plays a role here and how it is ensured that the response and prompt distributions are close to reference. Without that, the alignment problem is ill-defined. Please provide concrete justifications in theory and empirical results.
3. As described in Algorithm 1, informativeness is evaluated and a prompt subset is created based on current policy estimate and then the policy is updated based on the prompt subset. However, this causes an inter-dependence between the two which leads to nested structure, which is not clearly explained. Specifically, while computing the informativeness score for the prompts, it depends on \theta^*(x_t-1) i.e optimal parameter for the previous distribution. Provide clear explaination on the same. 
4. While iterating, every new prompt distribution will require generating new response, how is the evaluation coming from which reward model?  Is the ground reward available, if not please explain how the preference is obtained and how does it affect suboptimality?
5. Overall, which expression/Theorem guides us in understanding the improvement of prior suboptimality is not clear? Can you please point out/highlight how the current method improves upon the prior suboptimality due to static prompt distribution?
6. Its extremely crucial to show the prompt distribution and demonstrate its perplexity to ensure its not generating some meaningless or irrelevant prompts, since its not very evident on the KL divergence in the prompt space and its relation with the informative measure. Please provide detailed explaination to clarify that.

### Questions
1. Since equation 7, can't be directly solved, and is solved in an asymmetric fashion, then in the solver loop the KL should be over the response distribution and not joint right?
2. How is the KL divergence w.r.t reference policy for the algorithm? Please provide detailed ablation.
3. Whats the reward model availability? Is the true reward model available? 
4. There is a recent line of works on Stacklberg and Bilevel RLHF which deals with the entanglement in a leader-follower setting. Although not specific to updating prompt dist, but can be trivially applied. Provide a detailed comparison with the literature around that [1,2, 3].
5. Can you provide intuitions behind equation 7, on the KL divergence between the joint policy for both prompt and response? Is it even tractable to estimate or approximate this KL 


[1]. Principled Penalty-based Methods for Bilevel Reinforcement Learning and RLHF
[2]. SAIL: Self-Improving Efficient Online Alignment of Large Language Models
[3]. STA-RLHF: Stackelberg Aligned Reinforcement Learning with Human Feedback

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper looks to bring RLHF training past using a fixed prompt distribution. The authors pose the learning problem as an asymmetric game between two players: a prompt creator and solver. The authors then introduce eva, which is a simple prompt evolution approach that can be used in conjunction with existing RLHF algorithms. eva improves the performance of DPO, SimPO, SPPO, and ORPO 
on widely-used benchmarks within the community.

### Strengths
- It is very cool to see a curriculum learning approach leading to improvements in practical RLHF settings with LLMs. 
- I thought the writing quality and presentation of the paper was generally pretty strong throughout with the discourse feeling well structured.
- I appreciated the way that the approach was motivated through discussion of minimax regret in section 3.2. 
- I also found the connection with learning progress highlighted in section 3.4 quite interesting and feel that my concerns about the heuristic nature of the approach are largely addressed through this motivation. 
- The results in Table 1 are very impressive, especially in comparison to human prompts. 
- It is nice to see that the success of eva is very robust to the particular RLHF optimizer used. 
- A nice set of benchmarks and ablations are considered for the experiments. 
- I also thought the robustness with respect to more training iterations was a nice benefit to highlight.

### Weaknesses
- The novelty of the approach is not huge based on the prior literature, although I do believe a very practical contribution is made. 
- The approach is motivated intuitively and includes heuristics i.e. the "Informativeness Proxy" rather than a mathematical derivation of the optimization rule from first principles. 
- Evolving seems important for achieving performance based off table 4, but the particular algorithm chosen is not really described or contrasted with other approaches. This is discussed in the "Prompt synthesis for language models" section of the related work as an orthogonal contribution because other related approaches could be plugged into evolve(·), but it would be very interesting to understand more about how critical this particular choice is for the success of eva. 
- There are some typos that the authors missed such as "DIFFERENT INTUITIVE WYAS" and "Limitations and future directionss".

### Questions
Q1: Could you explain more about the particular choice of evolution algorithm used in your implementation of eva and different potential strengths and weaknesses related to this choice? 

Q2: Do you see empirical evidence of your intuition about learning progress discussed in section 3.4? It seems like some of these claims are directly testable. 

Q3: Could you visualize the curriculum learned in your experiments with eva? It would be very nice to get an intuition for why performance improves and what the heuristic prioritizes over time. 

Minor - Q4: When discussing future directions, the authors write  "(vii) further scaling up w/ million-level data". Can you clarify what this means? Seems like some important context is missing?

### Soundness
3

### Presentation
3

### Contribution
2
