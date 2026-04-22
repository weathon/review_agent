# Value Flows

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
While most reinforcement learning methods today flatten the distribution of future returns to a single scalar value, distributional RL methods exploit the return distribution to provide stronger learning signals and to enable applications in exploration and safe RL.  While the predominant method for estimating the return distribution is by modeling it as a categorical distribution over discrete bins or estimating a finite number of quantiles, such approaches leave unanswered questions about the fine-grained structure of the return distribution and about how to distinguish states with high return uncertainty for decision-making. The key idea in this paper is to use modern, flexible flow-based models to estimate the full future return distributions and identify those states with high return variance. We do so by formulating a new flow-matching objective that generates probability density paths satisfying the distributional Bellman equation. Building upon the learned flow models, we estimate the return uncertainty of distinct states using a new flow derivative ODE. We additionally use this uncertainty information to prioritize learning a more accurate return estimation on certain transitions. We compare our method (Value Flows) with prior methods in the offline and online-to-online settings. Experiments on $37$ state-based and $25$ image-based benchmark tasks demonstrate that Value Flows achieves a $1.3\times$ improvement on average in success rates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Value Flows, which utilizes expressive flow-based model to represent the conditional return random variable. Specifically, Value Flows learns a parameterized vector field model to estimate for the return expectation and variance for full future return distributions. Experiments on 37 state-based and 25 image-based benchmark tasks demonstrate the effectiveness of Value Flows when compared with various baselines in offline and offline-to-online settings.

### Strengths
1.	The motivation of this paper of using flow matching to model complex, multi-modal distributions in the domain of RL is clear. Value Flows deeply integrates flow matching into distributional RL setting to estimate return distribution.
2.	Both the theoretical and experimental parts are strong. For example, Value Flows achieves the best or near-best performance on 9 out of 11 domains.
3.	In Figure 2, they authors provide the example to show Value Flows has better 1-Wasserstein distance when modeling return distribution than both C51 and CODAC.

### Weaknesses
1.	The method section is hard to follow as it is full of symbols and the logic is not explicit. The authors could provide more illustrations to explain the how to connect flow matching to distributional RL.
2.	There are some typos. In Line 065, the format of citations is not correct.
3.	Some details, such as the complete loss function, are omitted in the main paper and provide in the Appendix, making the paper not self-contained.

### Questions
1.	Could you provide more explanations of vector field, such as its illustration in flow matching and its specific role in Value Flow?
2.	Why does Value Flows perform worse than ReBRAC and even IQL in D4RL adroit tasks?
3.	Could the authors compare the compute efficacy of Value Flows with other methods?
4.	If the flow step is very small such as 1 or 2, how could it affect the performance of Value Flows?
5.	Why is the target return vector field introduced in the algorithm of Value Flows?

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
3

### Summary
This paper proposes Value Flows, a reinforcement learning (RL) framework that models the entire return distribution using flow-matching generative models instead of discretized or quantile-based approximations. The method introduces a distributional flow-matching (DFM) objective that enforces the learned density paths to satisfy the distributional Bellman equation. The authors further derive a flow derivative ODE to estimate the variance of the return distribution, which is used as a confidence weight to reweight the flow-matching loss. Value Flows is evaluated on 37 state-based and 25 image-based benchmark tasks, showing a reported 1.3× average improvement in success rates over prior offline and offline-to-online RL baselines (Table 1, Fig. 3).

### Strengths
- Using the flow-matching approach to model the return distribution for each state-action pair is an interesting and intuitively reasonable idea. Flow matching is a more flexible distribution modeling technique that can handle complex distributions, making it more suitable for modeling return distributions in challenging tasks compared to previous distributional RL methods.
- This paper provides a thorough theoretical analysis and derivation of the Value Flows method, and I believe it is a highly effective and reliable approach.
- The experimental section provides substantial evidence supporting the authors’ claims. The return distributions learned by Value Flows are much closer to the ground truth compared to previous distributional RL methods. Moreover, on benchmarks such as OGBench and D4RL, Value Flows demonstrates clear performance advantages over the baselines.
- The writing in this paper is well-organized, and I can easily grasp the information the authors intend to convey.

### Weaknesses
- The paper lacks an analysis of the algorithm’s time and GPU memory consumption. It remains unclear whether flow matching requires more computational time and resources compared to previous methods.
- Some parts of the paper’s demonstrations are not entirely rigorous. For example, in Table 1, categorizing C51 under flow policies seems inappropriate to me.

### Questions
- C51 is an online distributional RL algorithm. When applying it to the offline setting, was any adaptation made—such as introducing conservative measures—or was it run directly under its original configuration?
- How does the algorithm perform on the classic D4RL MuJoCo benchmarks? Does it also show a clear performance advantage there?
- Is the return distribution visualization in Figure 2 based on samples from the in-distribution dataset, or on out-of-distribution samples collected separately in the environment? If it is in-distribution, could you additionally provide experiments with out-of-distribution samples? I believe that it would be more meaningful if the distribution learned by value flows demonstrates generalization.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a method to estimate the return distribution using flow matching. To estimate this distribution, the authors leverage a loss based on the distributional Bellman equation and demonstrate how to integrate it into the flow matching loss. This estimate can then be used to train policies either offline or in an offline-to-online setting. The method is validated on state- and image-based tasks from OGBench and D4RL.

### Strengths
- As far as I know, the approach is novel.
- I think the topic is interesting and relevant to the community.

However, I believe the weaknesses outweigh the positive aspects by a large margin.

### Weaknesses
- The contributions are not clear. The paper presents a loss in the main text and defers the details to the appendix. However, I believe the details are more important than what is stated in the main paper.

### Questions
My main concern is about the DCFM loss. The loss seems ill-posed; in particular, $v=0$ is a solution of Equation (5). This loss is not the one used in the experiments, and the reader needs to refer to Appendix C.2 for a clearer understanding. Indeed, in Appendix C.2, the authors mention that the DCFM loss produced divergent vector fields. To address this issue, the authors proposed adding a regularization term. However, I believe the problem is intrinsic to the loss introduced in the main paper. The "regularization" wBCFM seems more sound than the main loss itself.
I think the paper requires at least a major rewrite to properly highlight this "regularization" in the main paper. Similarly, the policy learning aspect should be explained in more detail in the main paper.

Questions:
- Could you comment on the fact that $v=0$ is a solution of Equation (5)? Why should this loss work despite that?
- You tested different regularization coefficients $\lambda$. Could you also train using only the wBCFM? Does it significantly change the performance?

Looking forward to the rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the inability of prior works in distributional RL to model return distributions due to the approximations made by discretizing the distribution. The paper leverages the recent success of flow models to model the full return distribution and better characterize the uncertainty of the return distribution for better flow matching. The authors ensure that at every step, the return distribution follows the distributional Bellman equation. Doing so, they only need to learn the vector field $v$ which they do using a variant of the distributional flow matching loss (which has the same gradient). They perform experiments on a number of state-based and image based domains in both offline RL and offline-to-online RL.

### Strengths
(1) The authors build on the recent success of flow models to model full return distributions for distributional RL which can be a major move forward from the discrete assumptions in prior works.

(2) The authors derive a way in which they only need to model the vector field and the flow would follow from it. Seems a clean and nice way to avoid complexities still staying within the theoretical bounds. 

(3) The authors propose a flow matching loss that avoids transition dynamics and other complexities. The proposed loss has the same gradient as that of the true loss.

### Weaknesses
(1) While the authors mention that distributional RL has advantages in exploration and safety, the domain chosen by them does not seem to be using these properties of distributional RL. To study the effects of modeling the full distribution and uncertainty, a better domain can be chosen. 

(2) The preliminaries for flow matching should be better. Currently there is no explanation on how $\phi$ is computed (which is relevant in the later sections of the paper). Similarly, since Lemma 2 is crucial for understanding of the method, it should be in the main paper and not in the appendix.

### Questions
(1) Why do you think that aleatoric uncertainty would be better than epistemic?

### Soundness
3

### Presentation
3

### Contribution
3
