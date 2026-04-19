# Learning Large Skillsets in Stochastic Settings with Empowerment

- Decision: Reject
- Scores: 3, 5, 8, 3

## Abstract
General purpose agents need to be able to execute large skillsets in stochastic settings.  Given that the mutual information between skills and states measures the number of distinct skills in a skillset, a compelling objective for learning a diverse skillset is to find the skillset with the largest mutual information between skills and states.  The problem is that the two main unsupervised approaches for maximizing this mutual information objective, Empowerment-based skill learning and Unsupervised Goal-Conditioned Reinforcement Learning, only maximize loose lower bounds on the mutual information, which can impede diverse skillset learning.  We propose a new empowerment objective, Skillset Empowerment, that maximizes a tighter bound on the mutual information between skills and states.  For any proposed skillset, the tighter bound on mutual information is formed by replacing the posterior distribution of the proposed skillset with a variational distribution that is conditioned on the proposed skillset and trained to match the posterior of the proposed skillset.  Maximizing our mutual information lower bound objective is a bandit problem in which actions are skillsets and the rewards are our mutual information objective, and we optimize this bandit problem with a new actor-critic architecture.  We show empirically that our approach is able to learn large abstract skillsets in stochastic domains, including ones with high-dimensional observations, in contrast to existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper studies how to improve the approximation of the empowerment objective for unsupervised skill learning in the domain of Unsupervised Reinforcement Learning (URL).  It highlights limitations in previous approaches due to loose lower-bound approximations and proposes a method to tighten these bounds. The proposed method is primarily evaluated in low-dimensional environments.

### Strengths
1. This paper offers an interesting perspective on revisiting the approximation of the classic unsupervised skill learning objective $I(S;Z)$, 
2. The proposed approach is generalizable and applicable to most existing URL methods that approximate the conditional state distribution $q_\psi(S|Z)$.

### Weaknesses
> ## Regarding motivation and contribution

In Section 2.2 

*"Several prior works have empirically demonstrated this result in which existing empowerment approaches tend to learn skillsets that do not change much from initialization."*

The previous empirical findings can not justify that the issues are caused by the loose lower bound approximation (this paper's focus). Instead, [1][2] consider the issues are implicit for $I(S;Z)$ objective because it can be optimized by skillsets close to initialization. 

**This paper contributes only to the approximation level of the $I(S;Z)$ objective**, so it also suffers the same implicit issue of $I(S;Z)$ as shown in Figure 16 (the torso positions of Ant remain close to initialization)

Furthermore, the focused problem can be considered as $\max_{\psi,\pi} \tilde{I}(\psi,\pi)$, where $\psi$ being the parameter for the variational posterior $q_\psi(Z|S)$. The difference between prior work and this paper is that existing methods update $\pi$ and $\psi$ alternatively in different time steps, while this paper updates $(\pi,\psi)$ together in one single time step. Figure 1 (right) only shows the single-step update of $\pi$ thus could be misleading to leave an impression that existing methods can never learn skillsets such as $\pi_1$. In fact, existing methods that alternatively update $\pi$ and $\psi$ could still learn skillsets similar to $\pi_1$.

In URL research, the notion of diversity is highly related to downstream task adaptation. Skills do not need to cover the entire state space to be diverse, as demonstrated in [8]. A smaller number of skills can still achieve diversity if they occupy distinct regions within the state space, with each skill serving as an effective fine-tuning initialization for downstream tasks within its specific region. Therefore, skill count alone is not an effective measure of skill diversity. Given the significant computational resources involved, the resulting improvements in downstream task adaptation may be minimal.

The overall motivation and contribution are more like findings instead of pushing the boundaries of URL research, making it more appropriate for a workshop.

> ## Regarding methodology

The need to search across the model parameter space makes this approach computationally intensive, even for simple 2D implementations. Using LLM-level computational resources for relatively impractical URL problems is extremely inefficient, especially given that the core contribution of this paper is a shift from alternating updates of $\pi$ and $q_\psi$ to simultaneous updates of $(\pi,q_\psi)$. Additionally, the paper provides no evidence that state of the art URL methods could not achieve similar skill diversity with significantly lower computational demands.

> ## Regarding evaluation

Experiments are mainly performed in simple low-dimensional environments, baselines are old and outdated. The number of skills is not compared to more recent methods such as [1][2][3][4]. It is also unclear how the number of distinct skills is calculated. 
There are no $H(Z), H(Z|S_n)$ figures for VIC to compare the core contribution of tightening the KL divergence bound between the true and variational posterior.

> ## Regarding rigor

The paper frequently discusses "diversity" without a rigorous definition. It seems that in this paper more diversity means more distinct terminating states. If so, $I(S;Z)$ is not a good measure of diversity. Maximizing $I(S;Z)$ can only guarantee that the distribution of $p(S|z)$ has large KL divergence to average state distribution $p(S)$ ([5]), but can not guarantee a large KL-divergence $D_{KL}(p(S|z_1)||p(S|z_2))$ between the conditional state distributions of different skills ([6]).  

As mentioned before, a limited number of skills can also exhibit diversity and achieve comparable downstream task performance with significantly fewer pretraining resources. This requires the terminating states to be not only distinct but also sufficiently distant from one another, Wasserstein distance-based metrics ([6][7]) are better choices to measure diversity in this sense.

It is not clear how the number of distinct skills is evaluated in the experiments. While the metric $\tilde{I}(Z;S_n)$ was mentioned, the exact connection between this mutual information measure and the actual number of skills remains unclear.

> ## Regarding presentation 

Section 1 has overly wordy explanations of the math formula, which makes the reading uncomfortable. Since the relevant formulas appear in Section 2 (e.g., (5)-(7)), it would be clearer and more efficient to introduce these formulas earlier.

Overall, the text is overly wordy without delivering a proportional amount of information. The reading requires considerable focus and energy, yet feels underwhelming in terms of insights gained.
 
> ## References
 

 

 
*[1] Lipschitz-constrained unsupervised skill discovery.*

*[2] Controllability-aware unsupervised skill discovery*

*[3] APS: Active Pretraining with Successor Features*

*[4] Contrastive Intrinsic Control for Unsupervised Skill Discovery* 

*[5] The Information Geometry of Unsupervised Reinforcement Learning*

*[6] Task Adaptation from Skills: Information Geometry, Disentanglement, and New Objectives for Unsupervised Reinforcement Learning*

*[7] Wasserstein Unsupervised Reinforcement Learning*

*[8] Explore, Discover and Learn: Unsupervised Discovery of State-Covering Skills*

### Questions
1. See weaknesses

2. How exactly is the number of distinct skills evaluated? Are $q_\psi(z|S)=1$ for every skill $z$ after traning?

3. How long is the total training time with the computation demands in Appendix B?

4. How sensitive is the training process to hyperparameters?



***
## After detailed discussions
After detailed discussions with the author, I found that I had no misunderstanding from the conceptions to the theories, and I gained a better understanding of the limitations of the implementations.  I can confirm that my rating of 3 and confidence of 5 is reasonable.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces Skillset Empowerment, a new approach aimed at enhancing skill diversity for agents operating in unpredictable environments. By tightening the mutual information bound between skills and states, this method allows agents to learn a wider range of distinct skills more effectively. Experimental results indicate that Skillset Empowerment outperforms prior approaches in complex, stochastic settings.

### Strengths
- Skillset Empowerment presents a fresh perspective on mutual information objectives, tightening the lower bound on mutual information between skills and states. This refined approach represents an original contribution to URL.
- The authors have conducted thorough experiments across multiple environments, which supports the empirical validity of their claims and provides a well-rounded assessment of the method's performance.

### Weaknesses
- Although the method successfully enables agents to learn a large skillset, the paper lacks evaluation on how these learned skills translate to practical, downstream tasks. The Unsupervised Reinforcement Learning Benchmark (URLB) [1] could be an appropriate choice to test this.
- The method assumes access to accurate transition dynamics, which restricts its applicability. This assumption is also problematic because, if the dynamics are known, planning and solving the MDP directly becomes feasible, reducing the need for URL methods.
- The computational demands are high, with individual critics trained for each parameter and multiple variational posteriors required. This increases the resource burden significantly compared to other URL methods, as shown in Appendix B.
- While the paper includes quantitative metrics, it falls short in qualitatively examining the distinctiveness and usefulness of the learned skills. A deeper exploration of these qualitative aspects would provide more insight into the method's practical impact.
- The submission’s structure makes it challenging to follow. Key formulas from Algorithm 1 are located in the appendix, and a large portion of the experimental results is also placed there, leaving only one main results table in the main text.


[1] Laskin M, Yarats D, Liu H, et al. URLB: Unsupervised Reinforcement Learning Benchmark[C]//Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

### Questions
The proposed tighter bound on mutual information is presented as beneficial for skill diversity, but the paper does not provide a formal theorem to support this claim. More rigorous theoretical analysis is necessary.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a new method for mutual information skill learning. The idea is to estimate a tighter mutual information lower-bound, by minimizing the kl divergence between the true skill posterior of the current skillset and the variational posterior of the current skillset, to avoid significant non-stationarity in the traditional unsupervised RL setting, the problem is formulated as an actor-critic bandit problem. The experiments show the advantage of the method over the baselines in stochastic setting.

### Strengths
- The idea of conditioning the posterior on the policy and skill distribution is interesting and formalizing the problem as bandit is novel
- Results are strong and outperform the baselines by a large margin, which shows the failure of most methods in stochastic domains.
- The number of skill learned by the method is much larger than the baselines, which shows that there is a big room for improvement in the problem of unsupervised skill learning.
- The visualization are insightful and shows that the method is really learning something.

### Weaknesses
- The objective for the existing empowerment skill learning methods is not clear to me (equation 6) some method condition only on the current state while some condition on the terminal state, so it is not clear to me how this notation encompasses these methods, can you add specific examples for using different conditioning and how the notation differs between methods?

- The argument of the loose lower bound is not that rigorous, in the paragraph from line 195-199, it does not say why the posterior differs largely from the posterior of the current skill, is it related to how the posterior is trained? can you give a formal proof showing how/why the posteriors of the candidate skill can differ significantly from the current skill?

- The notation is a bit confusing (equation 8 for example, is $\phi$ in the $\log q_{\psi^*}$ the same as the \phi in $\log \phi (z \mid s_0)$? I guess you meant conditioning on the parameters of $\phi$ right?), I recommend to make the notion clearer for example by using a different symbol than $\phi$ in the $\log q_{\psi^*}$ term in equation 8 and clarify what each symbol represents.

- Having |\pi| critics and posteriors seems a very limiting design choice for complex problems in which a more expressive model is usually needed, could you discuss the scalability of your method to more complex problem?

### Questions
- Can also add a visualization of the baselines for qualitative comparison? for example visualizing the entropy of the terminal states distribution in the stochastic four rooms environment.

- Is there any theoretical reason why the skill distribution chosen to be uniform, or is it more of a design choice? how the results will change if you used a different skill distribution? could you discuss this more in the paper?

- Why the results of Ant and Mountain Car are not included in the results table (table 1)? could you include the results ant and mountain car in table 1 and how the results compare to other environments ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the challenge of developing general-purpose agents capable of executing a wide range of skills in stochastic environments. The authors argue that a key objective for learning a diverse set of skills is to maximize the mutual information between skills and states, which quantifies the variety within the skillset. They critique existing unsupervised methods, such as Empowerment-based skill learning and Unsupervised Goal-Conditioned Reinforcement Learning, for only maximizing loose lower bounds on mutual information, which can limit the learning of diverse skillsets. To overcome this, the paper introduces a new empowerment objective termed "Skillset Empowerment," which aims to maximize a tighter bound on the mutual information. This is achieved by replacing the posterior distribution of the proposed skillset with a variational distribution that is conditioned on the skillset and trained to match the posterior. The optimization problem is formulated as a bandit problem, where actions correspond to skillsets and rewards to the mutual information objective. A novel actor-critic architecture is proposed to solve this bandit problem. Empirical results demonstrate the approach's ability to learn large, abstract skillsets in stochastic domains, including those with high-dimensional observations, outperforming current methods.

### Strengths
This paper studies an interesting problem, learning a valuable skill library from a novel perspective.

### Weaknesses
I am very concerned about the results of the experiment. The author only conducted experiments on very simple tasks, and did not conduct experiments on complex tasks [1]. I am afraid it is difficult to verify the effectiveness of the algorithm. In addition, the baseline chosen by the author is very old and needs to be compared with some recent progress [2, 3, 4].

[1] Urlb: Unsupervised reinforcement learning benchmark

[2] Controllability-aware unsupervised skill discovery

[3] APS: Active Pretraining with Successor Features

[4] Contrastive Intrinsic Control for Unsupervised Skill Discovery

### Questions
Can the author visualize some of the skills learned?

### Soundness
2

### Presentation
2

### Contribution
2
