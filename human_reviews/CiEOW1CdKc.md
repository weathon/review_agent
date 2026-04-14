# Latent Wasserstein Adversarial Imitation Learning

- Decision: Reject
- Scores: 5, 5, 5

## Abstract
Imitation Learning (IL) enables agents to mimic expert behavior by learning from demonstrations. However, traditional IL methods require large amounts of medium-to-high-quality demonstrations as well as actions of expert demonstrations, both of which are often unavailable. To address these limitations, we propose LWAIL (Latent Wasserstein Adversarial Imitation Learning), a novel adversarial imitation learning framework that focuses on state-only distribution matching by leveraging the Wasserstein distance computed in a latent space. To obtain a meaningful latent space, our approach includes a pre-training stage, where we employ the Intention Conditioned Value Function (ICVF) model to capture the underlying structure of the state space using randomly generated state-only data. This enhances the policy's understanding of state transitions, enabling the learning process to use only one or a few state-only expert episodes to achieve expert-level performance. Through experiments on multiple MuJoCo environments, we demonstrate that our method outperforms prior Wasserstein-based IL methods and prior adversarial IL methods, achieving better sample efficiency and policy robustness across various tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies the distribution matching idea in imitation learning and considers formulating this idea in a latent space learned through Intention Conditioned Value Function representation. The reward function is then defined in such latent space with a sigmoid transformation to turn it into non-negative and bounded values. The paper demonstrates that such idea can yield better representation of trajectories, which consequently lead to better empirical performance.

### Strengths
* [Originality] The paper proposes to study the distribution matching in a latent space, learned from Intention Conditioned Value Function method, and argues that the latent space can yield a better metric for the distribution matching under Wasserstein distance. This can be inspiring to imitation learning and reinforcement learning
* [Quality and clarity] Overall, the intro and related work are well written, showing a good understanding of the main challenge in imitation learning. The visualization of latent space representation and illustration of a better metric also greatly help readers to understand the paper. 
* [Significance] Considering imitation learning in a well-structured latent space can be more effective than the original trajectory space. The paper can contribute positively to the development of such direction.

### Weaknesses
### Some vague, inconsistent and confusing descriptions of the method
1. “clip threshold $c_0 > 0$ is added to the target action $a’$”. But the following equation applies clip to the random noise. Should it be $clip(a’)$ or $clip(\epsilon)$? 
 2. “the metric $c(s, s’)$ in Eq. (4) is inherently limited to be Euclidean”. There is no $c(s, s’)$ in Eq. (4). Here the metric refers to the 1-Lipschitz constraint? 
3. the reward used by TD3 algorithm is defined as $r(s, s’) = \sigma(-f(s, s’))$ (as described in line 304 and also specified in Line 7 in Algo 1). Why is the latent space not used for reward function? And how does the function $f$ takes $s$ and $s’$ as input to predict the reward while at the same time takes $\phi(s’)$ and $\phi(s)$ (vectors of different dimensions) as input. The reward definition is not even consistent with the optimization in Eq. (9). 
4. some notations are not clear: the summation index t in Eq. (5) starts from 0 and goes to infinity? How to ensure that it is well defined for $\gamma=1$ since $\gamma \in [0, 1]$; what’s the minimization in Eq. (6) over? Also, can authors explain the $\alpha$ in Eq. (6) and why it needs to be set in between 0.5 and 1? 
5. “Euclidean distance … in latent space … capturing the structure of the environment more faithfully”. I don’t know how to interpret this on Fig 2. What’s the structure of the Hopper and HalfCheetah? Also, why is Euclidean distance in this space a more suitable metric? From my understanding of Fig 2, the traj in original space has a better coverage of state values than the traj in the latent space. So the former representation would be better for distribution matching learning? 

### Vagueness in the experiment and lack of important ablations
1. Re the experimental results in Fig 3,  is the reward plotted corresponding to $-f(s, s’)$ or $\sigma(-f(s, s’))$? Why distinctive reward signal is better? It looks from Fig 3b that most rewards in latent space is close to 0 though
2. The reward in LWAIL is defined as the $\sigma(-f)$, which is naturally denser than the original reward function in Maze2D. The paper should compare the performance of TD3 with sigmoid transformation of the spare reward. 
3. In line 416, re “our original method”, do authors mean $f(s, s’)$ or $f(s, s’-s)$?. Further, re “agents without … embedding tend to remain in stable but relatively low-reward states, exhibiting less tendency to engage in riskier explorations”, can authors explain how this phenomenon is related to the ICVF method adopted by the paper? 
4. The paper should provide more ablation studies on the sigmoid transformation of reward function and the embedding difference as input, i.e., $\phi(s’) - \phi(s)$ rather than $\phi(s’)$. It is widely known that in Mujoco control benchmarks transforming rewards to non-negative will have a positive influence on the RL performance. Particularly, in Fig 5, the pseudo reward is always positive as it is the output of a sigmoid function while the ground truth reward may not be. Additionally, a pseudo reward generated by $f(s, s’-s)$ (i.e., without the embedding) should also be reported. Further, in Fig 6, it is unclear what it means for LWAIL without ICVF-learned $\phi$. The paper should explain explicitly how the LWAIL is trained without ICVF and should also report the results on using $f(s, s’-s)$. 

**Overall I found the empirical evaluation appears not quite convincing in its current version and many important details are missing. The inconsistent descriptions in the method part further makes the experiment part harder to understand.**

### Questions
1. what does it mean by “policy’s understanding of state transitions”? 
2. the paper introduces the Kantorovich-Rubinstein duality and then just refers this duality form as Rubinstein dual. Wouldn’t it more accurate to refer it as Kantorovich-Rubinstein duality or KR duality? 
3. why is the reward function in Line 7 Algo 1 computed from $f(s, s’)$? Not $f(\phi(s), \phi(s’) - \phi(s))$? 
4. how is the representation network $\phi$ being used in policy update? There is no mention of $\phi$ from line 5 – 12 in Algo 1. 
5. I found it very hard to grasp the main idea in Section Overcoming Metric Limitations, both intuition-wise and methodology-wise. It would be great if this part can be improved.

### Soundness
3

### Presentation
1

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
The authors present Latent Wasserstein Adversarial Imitation Learning (LWAIL), that uses state-only expert demonstrations and a Wasserstein distance metric computed in a latent space. To achieve this, the method includes a pre-training stage using an Intention Conditioned Value Function (ICVF), which establishes a meaningful latent space. By only requiring a single or limited number of expert state-only trajectories, LWAIL demonstrates competitive performance in imitation learning tasks, as shown in experiments across MuJoCo environments.

### Strengths
- Well-written paper with clear objectives.
- The method is simple to implement and has the potential to be easily applied to various existing GAIL methods.
- The problem and solution are well-defined, and the use of ICVF for pre-training is logically explained, highlighting how it aids in capturing the dynamics of state transitions.

### Weaknesses
- The paper lacks novelty. Both ICVF and WGAIL are existing methods, and state-only imitation learning is not a new topic.
- The authors' contribution lies in combining these two approaches and conducting imitation learning in the latent state space, claiming that it achieves a more robust policy with fewer samples. However, robots in MuJoCo typically exhibit cyclical behavior, and GAIL-based methods generally require only a small number of episodes. Also, the more robust policy has not been validated with specific results.
- The motivation for performing imitation learning in the latent space is insufficiently explained.
- The implementation codes are not provided.
- Lacks a discussion on the limitations of the method.

### Questions
- The tasks in the MuJoCo simulation environment are relatively simple, as they only use state vectors as input. To highlight the necessity of the latent space, would more complex visual imitation tasks be more appropriate? Could you provide related experiments?
- In practice, existing GAIL-based methods could also leverage ICVF to learn in the latent space. Could you provide additional experiments to offer more horizontal comparisons?
- Have the authors explored applying LWAIL in environments with multi-modal state distributions?
- How does LWAIL handle situations where the ICVF-pretrained latent space does not align well with the environment’s true dynamics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a latent Wasserstein adversarial imitation learning (LWAIL) method for achieving expert-level performance with limited state-only expert episodes. The latent space is obtained through a pre-training stage by the Intention Conditioned Value Function (ICVF) model. Experiments on MuJoCo environments demonstrated that LWAIL outperforms prior Wasserstein-based IL methods and prior adversarial IL methods.

### Strengths
1.	The proposed LWAIL method contains pre-training and imitation stages. Starting from an illustrative example showing that Euclidean is not a good distance metric, ICVF with random data is used in the pre-training stage to find a more meaningful embedding space. ICVF-trained embedding provides a more dynamics-aware metric than the vanilla Euclidean distance. Then in the imitation stage, the ICVF-learned embeddings are frozen and LWAIL minimizes the 1-Wasserstein distance between the state-embedding-pair occupancy distributions since it allows for a smoother measure and leverages the underlying geometric property of the state space. 
2.	The learned reward by LWAIL performs equal to or better than TD3 with ground truth reward.

### Weaknesses
1.	This paper claims that the proposed LWAIL can learn more efficient and accurate imitation from limited expert data with only one expert trajectory. The key reasons for this property have not been explained. How do ICVF-trained embeddings contribute to this property?
2.	There are many state embedding methods, however, there is not experimental comparison between ICVF-trained embeddings and other state-of-the-art state embeddings.

### Questions
1.	Figure 6 shows the contribution of ICVF-learned embedding. What is the difference between WDAIL and LWAIL without ICVF embedding?
2.	Figure 2 illustrates the same trajectory in the original state space and the embedding space for Hopper and Halfcheetah. It seems ICVF-trained embedding provides a much more dynamics-aware metric than the vanilla Euclidean distance. Is this observation consistent in more environments?

### Soundness
2

### Presentation
3

### Contribution
2
