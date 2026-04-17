# Scalable Option Learning in High-Throughput Environments

- Decision: Reject
- Scores: 6, 4, 6, 6

## Abstract
Hierarchical reinforcement learning (RL) has the potential to enable effective decision-making over long timescales. Existing approaches, while promising, have yet to realize the benefits of large-scale training. In this work, we identify and solve several key challenges in scaling online hierarchical RL to high-throughput environments. We propose Scalable Option Learning (SOL), a highly scalable hierarchical RL algorithm which achieves a ~35x higher throughput compared to existing hierarchical methods. To demonstrate SOL's performance and scalability, we train hierarchical agents using 30 billion frames of experience on the complex game of NetHack, significantly surpassing flat agents and demonstrating positive scaling trends. We also validate SOL on MiniHack and Mujoco environments, showcasing its general applicability. Our code will be open sourced.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Scalable Option Learning (SOL) as a way to scale up Hierarchical Reinforcement Learning to magnitudes more samples. In particular, the paper identifies many scaling challenges related to Hierarchical RL, and proposes valid solutions to these problems to achieve impressive throughput. These include using just a single network for all policies and computing discounted returns for all policies in parallel.

### Strengths
- The paper is effective in its key goal of improving throughput, which has repeatedly shown to be of great value to the RL community.

- Section 5 does a good job of ablating the contributions of the method, specifically demonstrating where the benefits of the method come from.

- Source code release and key parts of the code already being included make the contribution far more valuable as it can be easily used and adapted.

### Weaknesses
My primary concern of this paper is the similarity to the existing method, Agent57 [1]. This method also: 
- Learns multiple policies with different tasks (different amounts of intrinsic reward, discount rates, and probability of random actions), all parameterized by a single network.
- Uses a controller to select which policy to execute
- The single network appends a vector indicating which policy is being executed.
- Can achieve massive throughput (Agent57 reaches ~66,560 frames per second on Atari environments, almost 6 years ago, albeit on different hardware).
Additionally, Agent57 also provides in own intrinsic rewards [2] rather than relying on given rewards. While Agent57 does not use hierarchical RL (its controller selects policies for entire episodes), I think the amount of similarity without any acknowledgment or even a citation is problematic.

While acknowledged by the authors, the heavy reliance on a provided intrinsic reward is still problematic. In particular, many such methods are themselves computationally taxing, meaning the effectiveness of this method without provided intrinsic rewards may be greatly reduced.

While performance graphs currently show uncertainty using two standard errors, I’d prefer to see the use of RLiable’s [3] methods. They proposed using 95% confidence intervals when reporting results, and have been considered the highest standard for reporting results for some time.

### Questions
- I think a diagram of the agent’s network architecture would improve the paper, especially since it is one of the main ways higher throughput is achieved. Would it be possible to provide this in an appendix?

- As mentioned, this method has numerous similarities to Agent57. I would appreciate a detailed comparison of these methods, and exactly what the novelty of your method is in comparison. I would be very interested to see the performance of SOL on the Atari benchmark for direct comparison, although I understand this may not be possible within the given period.

- Given some existing intrinsic reward methods (such as E3B [4] or any reasonable method you wish), what would the throughput of SOL be, and what would the performance be like compared to the handcrafted rewards?

I do believe this paper would be valuable to the RL community if accepted. If you are able to answer all of my questions sufficiently, I will raise my score. If you can do this and show that your method is competitive with existing work on the Atari benchmark and include these results in the final paper, I will raise my score considerably higher.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Scalable Option Learning (SOL), a hierarchical reinforcement learning (RL) algorithm designed to scale online hierarchical RL to high-throughput environments with billions of samples. Building on the options framework, SOL jointly optimizes a controller policy and multiple option policies using actor-critic objectives, assuming access to intrinsic reward functions for each option. It enables ~35-580x higher throughput compared to prior hierarchical methods like HIRO, Option-Critic, and MOC, while retaining 86% of flat PPO's speed.
It is evaluated on challenging environments: NetHack, custom MiniHack tasks, and MuJoCo PointMaze. Results highlight SOL's ability to learn complementary options, adapt option lengths, and scale effectively.

### Strengths
The paper identifies critical bottlenecks in scaling hierarchical RL (e.g., non-parallelizable forward passes and return computations due to dynamic policy switching) and provides elegant, systems-level solutions.

Experiments are comprehensive and challenging. Training on 30 billion steps in NetHack (a sparse, long-horizon, partially observable environment) is impressive and demonstrates positive scaling trends, surpassing flat PPO and hierarchical baselines like HiPPO.

### Weaknesses
1. **Lack of Clarity for Broader Audiences:**
The paper assumes familiarity with hierarchical reinforcement learning (HRL) and high-throughput RL, which may make it challenging for readers less versed in these areas. For instance, it does not clearly delineate scenarios where HRL is advantageous, such as long-horizon tasks with sparse rewards (e.g., NetHack’s dungeon navigation or robotics with delayed feedback) or settings requiring complex behavior coordination (e.g., switching between fighting and healing). Similarly, the relationship between HRL and high-throughput RL is underexplored; while SOL bridges these by enabling parallelized training for HRL, the paper could better explain how HRL’s task decomposition complements high-throughput RL’s scalability to make its contributions more accessible. A brief primer on when and why HRL is needed, and its synergy with large-scale training, would enhance clarity for a wider audience.
2. **Limited Performance Comparisons with Prior HRL Methods:**
While the paper compares SOL’s throughput to prior HRL algorithms (HIRO, Option-Critic, MOC, Hierarchical RLLib), it omits these as performance baselines in key experiments like NetHack and MiniHack, relying instead on flat RL (APPO), SOL-HiPPO, and Motif. This choice is not well-explained, leaving it unclear why established HRL methods are excluded from performance evaluations. The likely reason—computational infeasibility due to their low throughput (55-1,200 steps/second vs. SOL’s 43,000)—is not explicitly stated, nor are challenges like adapting these methods to NetHack’s symbolic inputs. Including MOC only in MuJoCo and not NetHack further muddles the comparison scope. Explicitly addressing these omissions would strengthen the paper’s rigor and clarify SOL’s standing relative to prior HRL work.
3. **Dependence on Predefined Option Rewards:**
SOL’s performance heavily relies on predefined intrinsic rewards for options (e.g., ΔHealth, ΔScore in NetHack), which require domain knowledge to design. Experiments show that without these rewards, SOL-HiPPO (using only task rewards) performs similarly to flat APPO, suggesting that the hierarchical architecture’s benefits are contingent on well-crafted rewards. This raises questions about the architecture’s standalone effectiveness, as the hierarchy alone does not consistently outperform flat RL without intrinsic guidance.

4. **Unclear Scalability Advantage of Dual-Policy Training:**
The paper claims that SOL’s hierarchical approach, involving training both a controller and option policies, is more scalable than traditional single-policy RL, but this is not convincingly justified. Intuitively, training two policies (controller and options) should increase computational complexity compared to a single flat policy (e.g., PPO). While SOL’s innovations—such as a unified neural network, environment wrapper, and parallelized computations—enable high throughput, these appear as engineering optimizations rather than fundamental algorithmic advances.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Scalable Option Learning (SOL), a hierarchical reinforcement learning algorithm that scales to billions of samples in high-throughput environments. Its main contribution is overcoming the system-level bottlenecks that have so far limited the scalability of hierarchical RL compared to flat agents.

### Strengths
The paper is well written and clearly structured, making it easy to follow. The proposed solution is presented in a clear way, and the experimental results appear solid and convincing.

### Weaknesses
1. The paper’s novelty lies primarily in systems-level design. While the throughput improvements are impressive, the core RL algorithmic advances are modest, with most contributions being engineering optimizations rather than new principles for hierarchical learning.

2. The evaluation focuses mainly on NetHack and MiniHack, with only a simple PointMaze task in MuJoCo. Although the proposed solutions are promising and could have broader impact, this potential is not fully demonstrated within the current set of experiments.

3. The framework assumes access to well-defined intrinsic option rewards, which is a strong assumption. In their absence, hierarchical RL provides no clear advantage over flat RL, highlighting a key limitation of the approach.

### Questions
1. In Section 3.3, paragraph Architecture, the one-hot vector $u$ is introduced to determine whether an option policy or the controller policy is active. How is $u$ updated during training and execution? Is it managed solely by the environment wrapper, or is there any learning signal associated with it?

2. IMPALA's V-trace employs an off-policy correction term to account for lag between the actors and the learner in asynchronous settings. In your method, the value objective $V^{\omega}(s_t)$ does not appear to include such a correction . Could you clarify why this choice was made? In practice, is there a noticeable lag between the actor policies and the learner’s most recent version, and if so, how is this handled?

### Soundness
3

### Presentation
4

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
The paper proposes Scalable Option Learning, a hierarchical RL framework optimized for high-throughput distributed training. The method unifies multiple option policies into a single network with masking and parallel advantage computation, enabling efficient batching on GPUs. Experiments on the NetHack Learning Environment and MiniHack claim large throughput gains (35–580x) over prior hierarchical baselines and improved learning performance in long-horizon tasks.

### Strengths
- Well-executed system engineering that integrates hierarchy into scalable RL pipelines.
- Strong empirical demonstration on NetHack.
- Transparent reporting of hyperparameters and ablations.
- The throughput improvements are impressive, even if largely due to implementation efficiency.

### Weaknesses
- The paper mentions option-specific reward signals but does not describe a general way to derive them. If these require manual tuning, applicability beyond benchmark tasks is uncertain.
- The learning formulation largely reuses option-critic ideas, the primary novelty lies in batching and software efficiency.
- The main empirical evidence comes from NetHack. Additional results on diverse continuous-control or multi-task domains would strengthen generality claims.
- The paper includes descriptive visualizations of option usage but no quantitative diversity metrics (mutual information, entropy). It remains unclear whether options represent meaningful temporal abstractions or redundant sub-policies.
- Gains may partly reflect different batching or hardware configurations, compute-normalized comparisons would be more informative.

### Questions
- How are the intrinsic rewards defined in each environment? Are they fixed heuristics or learned?
- How sensitive is SOL’s performance to these reward definitions or scaling factors?
- Are gradient interferences between option heads and the shared trunk observed, and if so, how are they mitigated?
- Can the same design support multi-level hierarchies (options over options)?
- How is throughput measured and normalized (environment frames per second, learner updates per wall-clock second, ...)?

### Soundness
3

### Presentation
4

### Contribution
3
