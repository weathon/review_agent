# Handling Delay in Real-Time Reinforcement Learning

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
Real-time reinforcement learning (RL) introduces several challenges. First, policies are constrained to a fixed number of actions per second due to hardware limitations. Second, the environment may change while the network is still computing an action, leading to observational delay. The first issue can partly be addressed with pipelining, leading to higher throughput and potentially better policies. However, the second issue remains: if each neuron operates in parallel with an execution time of $\tau$, an $N$-layer feed-forward network experiences observation delay of $\tau N$.
Reducing the number of layers can decrease this delay, but at the cost of the network's expressivity. In this work, we explore the trade-off between minimizing delay and network's expressivity. We present a theoretically motivated solution that leverages temporal skip connections combined with history-augmented observations.  We evaluate several architectures and show that those incorporating temporal skip connections achieve strong performance across various neuron execution times, reinforcement learning algorithms, and environments, including four Mujoco tasks and all MinAtar games. Moreover, we demonstrate parallel neuron computation can accelerate inference by 6-350\% on standard hardware.  Our investigation into temporal skip connections and parallel computations paves the way for more efficient RL agents in real-time setting.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors explain the negative impacts of execution time and observation delay on real-time reinforcement learning performance. They describe a model architecture which includes “temporal skip connections” and “history augmentation”, explaining the theory behind it, and describing its impact on model expressiveness vs minimizing execution delay. Experimentally, they compare this new model architecture to existing architectures in several “game” environments, showing improved performance.

### Strengths
Very interesting paper:
- Great introduction to the problem space.
- Great theoretical motivation section.
- Architecture is well presented & Figure 2 is excellent.

### Weaknesses
Most of the paper speaks to the novel skip-connection architecture with history augmentation. As I understand the paper, some sections measure how existing parallelism approaches improve inference performance. I think these “mundane” sections take focus away from the novel content. Consider moving 1) the abstract sentence “accelerate inference by 6-350% on standard hardware” 2) Figure 1a, and 3) Section 5.4 to the appendix. This will help focus the paper on just the novel content. Apologies if the above is based on a misunderstanding of the paper content.
 
Table 2 would be easier to absorb if it included % deltas.

Erata:
- Typo: number of layes
- Inconsistent speech marks: As ”skip connections”

### Questions
I think the conclusion sentence “These architectures demonstrate robust performance across various environments and neuron execution times” undersells the paper results somewhat. Consider adding a slightly stronger follow-up statement “Experimentally, these architectures often perform better than traditional architectures” or similar.

### Soundness
3

### Presentation
4

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
This paper presents a novel approach to addressing delays in real-time RL caused by neural network computation. The authors introduce a skip connection architecture that enables faster observation-to-action response by creating direct pathways in the network. This architecture is theoretically justified, and empirical results show its effectiveness over baseline models across multiple simulated environments. Through extensive ablation studies, the authors demonstrate that the observed performance improvements are due specifically to the skip connections, demonstrating the potential of this approach for real-time RL applications.

### Strengths
1. The authors introduce skip connections into neural networks to address inference delay in real-time RL. This approach is novel in the context of RL and represents the first application of skip connections specifically to reduce inference delay.
2. Extensive experiments validate the effectiveness of the proposed method. Results show that, with skip connections, agents operating under delay significantly close the performance gap compared to agents without delay and substantially outperform agents with delay but without skip connections.
3. The method is theoretically justified, and the theoretical findings are consistent with the experimental results.
4. The paper is clearly written and well-structured.

### Weaknesses
1. Experimental repetition is inconsistent across settings (10, 5, or 3 times), which could impact result reliability. Standardization or explanation is needed.
2. The neural network used is simple, resulting in minimal inference delay. Testing with more sophisticated models, such as computer vision-based architectures for processing observations, would better demonstrate real-world applicability. Additionally, the authors note that skip connections do not fully mitigate the performance drop from delay in complex tasks, raising the question: is this approach only effective for simple tasks where delay is less critical?
3. Scalability is limited, as larger architectures may encounter GPU memory constraints, reducing the benefits of parallel computation in complex environments.
4. Although targeting real-world delay issues, the method is only tested in simulations, leaving its real-world effectiveness unvalidated.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper mentions two main challenges in real-time reinforcement learning (RL): (1) limited action frequency due to neural network inference time, and (2) observational delay where actions are actually based on previous observations due to significant inference latency. While pipelining the policy network provides a framework to partially address these challenges, it is still not enough. The paper proposes two methods to overcome these challenges:
- Building on policy network pipelining, the paper introduces temporal skip connections which cache hidden representations from previous timestamps to compute the current action within a single layer's execution time. My understanding is that similar to residual connections, this approach can be viewed as an ensemble of multiple actors with different capabilities.
- The paper leverages history augmentation by incorporating previous actions to better maintain the Markovian property in real-time RL setting.
To validate their approach, this paper conducts experiments across various environments. The results demonstrate that when neuron execution time is small, the proposed method successfully accelerates inference without hurting performance. However, with longer neuron execution times or more complex environments, there is still a more noticeable degradation in performance.

### Strengths
- The paper did a good job presenting clear explanations of the motivation, research questions and statements, experimental methodology, and results. The writing is well-structured and easy to follow.
- The discussion of the actor expressive capacity vs observational delay is an interesting and important task in real-time reinforcement learning.
- The proposed methods could be motivated and connected with some theoretical understanding.

### Weaknesses
- The paper could potentially benefit from establishing a stronger connection between the theoretical analysis/motivation and experimental results. Specifically, including a concrete example that demonstrates how incorporating previous actions helps maintain the Markovian property which is broken before, and enhances training performance. This would strengthen the paper's arguments and make them more compelling.

- The proposed methods have suboptimal performance in certain environments, such as in MinAtar games. In my opinion, the paper could benefit from a more detailed/thorough discussion on those scenarios and possibly explore potential solutions, such as enhancing the action network's expressiveness to better handle the rapid changing observations (just a thought, not limit to this). These discussions/explorations would provide very valuable insights into current method's limitations and motivate future research.

- (Minor) The experimental evaluation could be further improved by including a broader range of testing environments, such as (could be a subset of) vanilla pixel-based Atari games. My understanding is that these experiments would further demonstrate the proposed methods are general across even more domains.

### Questions
- It would be helpful to provide clearer motivation and reasoning for how the neural execution times were chosen in different experiments. For example, providing both the environment step time and layer execution time would give readers a better sense of the magnitude of neural execution time in practice.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes to use parallel pipeline computation and temporal skip connections to tackle real-time RL. It presents a formalization of the problem and proposed method as well as empirical results on Mujoco, MinAtar, and Minigrid.

### Strengths
- the paper is well structured and clearly written
- evaluation for both discrete and continuous action spaces
- error bars for quantitative analyses

### Weaknesses
- the novelty of the proposed method is limited to the (rather narrow) problem setting of real-time RL 
- the empirical evaluation is limited to toy tasks (not really real-time RL problems); the paper would greatly benefit from a real-world application (or any application where the real-time constraint arises naturally)
- no reproducibility due to missing code

Especially the combination of the first two points restricts the contribution of this paper as both novelty and significance are severely compromised.

### Questions
- proposition 1 includes a relation between environment stochasticity and observational delay, i.e., the bound breaks down for deterministic environments; how do the empirical results relate to this?

### Soundness
3

### Presentation
3

### Contribution
1
