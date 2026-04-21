# Accelerating Goal-Conditioned Reinforcement Learning Algorithms and Research

- Avg Score: 7.50
- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
Self-supervision has the potential to transform reinforcement learning (RL), paralleling the breakthroughs it has enabled in other areas of machine learning. While self-supervised learning in other domains aims to find patterns in a fixed dataset, self-supervised goal-conditioned reinforcement learning (GCRL) agents discover *new* behaviors by learning from the goals achieved during unstructured interaction with the environment. However, these methods have failed to see similar success, both due to a lack of data from slow environment simulations as well as a lack of stable algorithms. We take a step toward addressing both of these issues by releasing a high-performance codebase and benchmark (`JaxGCRL`) for self-supervised GCRL, enabling researchers to train agents for millions of environment steps in minutes on a single GPU. By utilizing GPU-accelerated replay buffers, environments, and a stable contrastive RL algorithm, we reduce training time by up to $22\times$. Additionally, we assess key design choices in contrastive RL, identifying those that most effectively stabilize and enhance training performance. With this approach, we provide a foundation for future research in self-supervised GCRL, enabling researchers to quickly iterate on new ideas and evaluate them in diverse and challenging environments. Code: [https://anonymous.4open.science/r/JaxGCRL-2316/README.md](https://anonymous.4open.science/r/JaxGCRL-2316/README.md)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce JaxGCRL, a benchmark and framework for evaluating goal-conditioned RL algorithms based on contrastive learning.
They re-implement a number of goal-conditioned tasks from prior literature and evaluate their implementation on it. 

They then evaluate the effect of different losses, more samples, and larger networks on their implementation. They demonstrate that their Jax-based implementation is significantly faster than previous libraries, accelerating future research.

### Strengths
The paper has several significant strengths.

- Although not novel, JAX implementations are to be commended. They improve research iteration speed significantly.
- The paper is very well written. The authors communicate their results clearly and unambiguously.
- The authors evaluate using the inter-quartile mean and bootstrapped confidence intervals. This is more sound than using learning curves etc.
- The authors provide a number of ablations and experiments that explain the performance of their implementation.

### Weaknesses
However, I have a number of issues with this paper, which is why I recommend rejection. 

- The authors claim that their setting is challenging, but do not effectively demonstrate that this is the case. The authors demonstrate that by using a bigger network (1024 layer width and depth of 4) and layer norm, the performance significantly improves. They also run experiments where they train for significantly more interactions. However, as best I can tell (it is not always clear which network is used in which experiment), the authors never run their biggest, highest performing network for 300M steps on all the tasks. The authors do not pitch their work as focussing on sample efficiency, and therefore I am not sure why their evaluation framework should be compelling if the tasks can be solved by scaling up networks and using more samples. If the authors can provide a demonstration that this does not satisfactorily solve their benchmark, **I will raise my score**. However, without this demonstration, I do not believe that the experimental insights and JAX implementation are enough to warrant acceptance.
- I am confused about the experiments concerning the update-to-date ratio (UTD). Given a fixed step budget, doing fewer or more updates is a pure trade-off. You can do fewer, less noisy updates, or do more, noisier updates. This occurs all over RL, for example when choosing the number of parallel environments to use in PPO. I am not sure why a high or low number of updates would be beneficial, or this quantity would be interesting to examine.

I also have a number of more minor points:
- The authors claim that they cannot directly compare brax and mujoco because brax uses a different physics engine, but the MuJoCo physics engine has been available in brax for a while now [1] -- what exactly is the issue here? 
- The discussion of related work on jax-based environments is missing some work. Gymnax [2] and PureJaxRL [3], both were important landmarks in the use of and benefits of JAX in RL and warrant inclusion.
- The authors should probably rephrase line 117, which begins with "In addition to reducing CPU-GPU data-transfer overhead...". While implementing an environment in JAX *does* do this, there are also significant other factors such as JIT compilation and the resulting operator fusion and the ability to use more vectorised environments than a typical CPU + pytorch approach that lead to the significant speedups. 
- A number of the papers listed in the appendix have incorrect citations or are missing authors.
- Line 1032 in the appendix contains a typo (lenght -> length)

### Questions
See weaknesses.

[1] Brax documentation. https://github.com/google/brax?tab=readme-ov-file#one-api-four-pipelines

[2] Gymnax: A JAX-based Reinforcement Learning Library. Robert Lange. https://github.com/RobertTLange/gymnax

[3] Lu, Chris, et al. "Discovered policy optimisation." Advances in Neural Information Processing Systems 35 (2022): 16455-16468. https://github.com/luchris429/purejaxrl

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides a JIT-complied codebase with vectorized environments that can speed up the training and iterating new ideas on goal-conditioned reinforcement learning problems.
In additional, it provides a stable baseline algorithm for the goal-conditioned reinforcement learning problems that's benchmarked in the 8 diverse continuous environments.

### Strengths
* The JaxGCRL codebase is significantly faster than the original codebase. 
* The proposed baseline consistently outperform the counterpart in all 8 environments, demonstrating the stableness from simple to complex environments.
* The performance of different design choice is extensively tested and the result metric is easy to interpret.

### Weaknesses
The paper mentions it leverages the power of GPU-accelerated simulators, but by comparing against the brax training code under https://github.com/google/brax/tree/main/brax/training, there are some similarities for the training code as well, and it's not mentioned in the paper.

### Questions
* In Sec 5.3, why is the contrastive objective only evaluated on part of the 8 environments? Similar question in sec 5.6 for examining different UTD ratios.
* In Fig 1. Are the num_actors same for the JaxGCRL and CRL?
* How do you define if the agent is within goal's proximity?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces JaxGCRL, a codebase that contains environments and a scalable goal-conditioned RL algorithm, all implemented in JAX. This allows researchers to train GC agents much faster than before, making these experiments more accessible. This work also analyses several design decisions of contrastive RL algorithms, enabled by a fast simulator & algorithm implementation.

### Strengths
- Adding even more environments and algorithms to the JAX ecosystem is great, especially for goal-conditioned RL which is lacking in this space.
- The proof-of-concept experiments demonstrate what this library can allow, namely, more thorough investigation of design decisions in goal-conditioned RL
- The writing and motivation is quite clear.

### Weaknesses
- I can't see any major weaknesses, apart from the limited number of environments, although 8 is pretty respectable.

### Questions
- What is your support plan going forward with JaxGCRL, are you planning on adding new environments or algorithms?
- It seems like JaxGCRL is very much focused on brax-type environments, is there a part of goal conditioned RL research that potentially focuses rather on discrete action environments that you are leaving out?
- What about other non-contrastive GCRL algorithms? Are you planning on adding support for those?
	- Relatedly, how easy would it be for someone else to implement a new GCRL algorithm to fit within your framework?
	- And how easy is it to add another goal conditioned environment, based on an existing JAX environment? For instance, minigrid or craftax or xland minigrid, etc?
- In the maze, for instance, can you dynamically, within the JIT, change the maze layout, or does it have to be statically known at compile time?
- Is there an easy way to transfer a JaxGCRL agent to existing environments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new library for goal conditioned reinforcement learning (GCRL). Unlike prior work, their method runs end to end on the GPU, making training faster. They implement 8 environments in JAX, as well a few algorithms and various objectives. Then, they evaluate existing methods across a number of axes, investigating replay ratios, model sizes, and energy functions.

### Strengths
- The library is well-motivated, as speeding up RL leads to better experimentation
- The authors implement many environments and energy functions
- The scale, energy function, update-to-data ratio experiments are interesting and useful for future work on GCRL

### Weaknesses
The library appears like a "one-and-done" sort of thing that will not be maintained after publication. In RL there is already a large graveyard of abandoned RL projects that no longer run and provide no value to the community. Given this fact, I can only review the current state of the library. In its current state, I think the library needs a bit more work before publication. Please see https://docs.cleanrl.dev for an example of what I think a modern RL library should look like.

- There is no documentation, it is unclear:
    - Which approaches are implemented
    - How to use these approaches
    - How to add new models
    - The structure of the codebase
- There are no unit tests, so the correctness of the code (and the ability to maintain the code as time goes on) is unclear
- The train script is solely for the authors, relying on a pre-existing conda environment
- There are no tutorials beyond a single bash command that runs a parameter sweep
- The library relies on wandb, and does not seem to run without a wandb account
- As far as I understand, the authors only implement 3 algorithms, and I would like to see more than three baselines so that we can do proper comparisons

### Questions
- Figure 4 typo: "though DPO policies remain at the goal for a shorter"

### Soundness
3

### Presentation
3

### Contribution
1
