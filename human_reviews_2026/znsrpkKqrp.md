# Option Discovery via Differentiable Neural Decomposition

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Option discovery via neural network decomposition is a promising way of discovering temporally extended actions in reinforcement learning. The challenge is that the number of sub-functions a network encodes grows exponentially with its size, so finding sub-functions that can be useful in downstream tasks is a difficult combinatorial search problem. In this paper, we turn this combinatorial search problem into a differentiable problem by showing that extracting sub-functions from a network is equivalent to learning masks over the neurons of the network. In addition to extracting sub-functions, we can also learn default input parameters to such sub-functions through masks over the inputs. Neuron masks select what to execute; input masks specify how to call it. We evaluate our masking scheme on grid-world problems with binary observations \rev{and on a robotics task with continuous action and observation spaces}, using feedforward and recurrent policies. Our results show that masking can produce sub-functions with default input parameters that improve sample efficiency on downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
# Summary

The paper expands on a prior work by Alikhasi & Lelis 2024 to alleviate two limitations that prevented from applying that prior work in practical setting. The first one is the inability to consider a large portion of the combinatorially large space of sub-trees of so-called "neural trees", a result of decomposing neural networks into chains of if-then-else statements. The second one, the so-called "default input parameters" that encode values of features not present in a different observation from the one the policies were learned for, a must-have to make the policies generalize.

The work is based on a neat observation that masking neurons in the network can allow devising the corresponding sub-tree of the neural tree. Thus, both learning the sub-trees and the default parameters can be done with gradient descent by masking the input observation.


# Significance

This work makes previously suggested idea practically applicable, which makes it significant in my book.

# Soundness

The authors claim that "The process of selecting a subset of options that minimizes the Levin loss is NP-hard in general (Garey & Johnson, 1979)". I am not sure where in the cited book is the mentioned problem, it would be great to refer to it specifically (e.g., page number)


# Novelty

The novelty of the paper is somewhat lessens by this being a follow up work, but the paper does solve a real issue with the previously proposed approach, in a novel way.


# Scholarship

The scolarship seems to be fair, with the related work covering several related directions. 
 
# Clarity

The paper is sufficiently clear, easy to follow, although sometimes feels like parts of the text were rephrased from previous versions by different people. However, I did not find it to be an issue.


# Evaluation and Reproducibility

The evaluation specifies precisely the hypotheses tested, and tests on a variety of environments, compared to other options learning methods (baselines). DiDec shows consistently high performance, compared to the baselines. 


# Minor
1. line 073: the sentence is awkwardly composed, as if only part of it was changed into active/passive voice.
2. line 099: First reference to Figure 1 is on page 2, while the figure itself is on page 5. Consider bringing closer.

### Strengths
1. Neat observation that allows to make existing theoretical idea practically applicable.
2. Sound presentation, motivating example.
3. Extensive experimental evaluation.

### Weaknesses
1. The motivating example can be presented in a more structured way.
2. The complexity result is unclear, is it known from the literature (probably not)? If not, should be presented more formally.

### Questions
1. You say "The process of selecting a subset of options that minimizes the Levin loss is NP-hard in general (Garey & Johnson, 1979)". Is it really a known result? If not, do you have a proof for the claim? Even if it is, could you specify a corresponding optimization problem (e.g., integer program)? Integer programming solvers can deal with quite large problems, maybe a greedy approach is not actually needed?

2. I wonder whether there is a way to make your subtrees/options symbolically representable. Do you have a way to recognize what do the learned options correspond to in each of the tested domains? For instance, do the learned options policies in MiniGrid correspond to moving between rooms? Something else?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Differentiable Dec-Options (DIDEC), a method for discovering reusable, temporally-extended "options" from previously trained neural network policies. The work's primary claim is that it overcomes the exponential combinatorial search problem of its predecessor, Dec-Options, which limited that method to impractically small networks. DIDEC reframes this combinatorial search as a differentiable masking problem, allowing it to scale to larger networks (e.g., 64-256 neurons). A secondary contribution is the concept of "default input parameters," learned via input masking, which allows extracted sub-functions to generalize to new states. The method is evaluated in discrete grid-world domains, where it shows improved sample efficiency over baselines.

### Strengths
1.  The core technical idea of reframing the $3^d$ combinatorial sub-tree search into a differentiable 3-way masking problem is clever.
2.  The insight of "default input parameters" to disentangle a skill from its context is a useful contribution to the sub-problem of skill generalization.
3.  The paper is well-written, with clear motivation and a strong illustrative example in Figure 1.

### Weaknesses
1.  Misleading Contribution (BC vs. RL): The paper's primary flaw is framing itself as an "option discovery" paper. The algorithm is a supervised, brute-force behavior cloning system that clones every possible sub-trajectory of length $z \in [2, z_{max}]$. This has little to do with the elegant goal-driven formulation of the options framework with the underlying option induced SMDP.
2.  Impractical Computational Complexity: The paper claims to solve the scalability problem of Dec-Options but ignores the $O(N \times T \times z_{max} \times E \times C_{train})$ complexity of its own generation step. This is not a "scalable" solution; I ackowledge it's improvements over Dec-Option, but as a work aims to scale this direction up, current complexity of DIDEC is doubtful to be applied to actual RL envs. 
3.  Severely Limited Scope and Insufficient Validation: Validation on grid-worlds alone is not sufficient for ICLR in 2025. If this is a theory paper its acceptable, but as stated below, this work clearly lack of theoretical analysis. Standard benchmarks (MuJoCo, DMC, robotics) that define modern RL research was not tested at all.
4.  Stacked Heuristics: The method is not a principled solution. It relies on a proxy loss (cross-entropy) to generate options and a separate, non-optimal heuristic search (SHC) to select them. This stack of approximations lacks any formal guarantees. How is the convergence of each sub-option is guranteed (since sub-trajectory are segmented brute-forcely by sliding window, trained on trajectory not even sampled from its own)?  How is the optimality guaranteed?
5.  Authors need to work on the concise and accurate math derivations. Most parts of the algo are verbally explained and very hard to follow. It would be much ease for readers if they explain with equations / pseudo code.

### Questions
I acknowledge the improvements over Dec-Option of this work but raised my concerns as in weakness section. I do think the direction Dec-Option proposed is an exciting and novel direction worth to explore. But the claim authors made at the beginning "scaling Dec-Option up" turns to be a disappointment. I'm willing to raise my score if the author can prove their scalability to "real" problem such as locomotion or VLA tasks.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a hierarchical approach (called DIDEC) that reuses the set of learned policies and trajectories from previous tasks to define options for new tasks. This is mainly done by first converting the policies into fixed horizon options, then using them to discover reusable neural sub-functions by treating network decomposition as a *differentiable masking problem*. Building on the prior Dec-Options framework, which exhaustively enumerated neural sub-trees, DIDEC uses gradient-based learning to mask neurons and inputs, thereby identifying sub-functions and their “default parameters” that generalize across tasks. This enables option discovery in large networks and supports transfer and lifelong learning. Experiments across MiniGrid, MiniHack, and ComboGrid environments using A2C and PPO demonstrate that DIDEC improves sample efficiency and task generalization compared to various baselines.

### Strengths
* **Significance and originality:** The work contributes to both hierarchical and lifelong R by proposing a novel way to adapt learned policies for new tasks through learned masks.
* **Methodological depth:** Provides a detailed formulation, algorithms, and training procedure (with pseudocode and appendices). The approach generalizes across RL algorithms (A2C, PPO) and architectures (MLP, LSTM).
* **Experimental validation:** Demonstrates strong sample efficiency and transfer benefits across diverse domains. Particularly impressive is DIDEC’s ability to solve the 5×5 sparse-reward ComboGrid, where all baselines fail to converge.
* **Ablation studies:** Includes informative ablations showing the effects of input-only vs. neuron-only masking, illustrating the versatility of the method.
* **Clarity and completeness:** The paper is well structured and situates the work clearly within related fields of option discovery, masking, and programmatic RL.

### Weaknesses
- **Clarity:** The paper dedicates excessive space to background, delaying the introduction of the main method until Section 5 (page 4). This reduces focus on the novel contributions and experimental insights. This is potentially why the main algorithm (DIDEC) is only fully described in the appendix, which hinders readability and accessibility in the main text. Also the lifelong setup in only first mentioned in the background and the critical dependence on prior trajectories comes even much later. It would have helped if this context appeared in the abstract and introduction.

- **Lack of analysis of learned options:** The paper does not explore what the masked options learn. E.g., their termination sets, which inputs are masked, or how they perform when reused in original tasks after masking.

- **Missing failure analysis:** No discussion of when DIDEC fails, such as tasks with significantly different horizon lengths from prior ones (which may break deterministic termination assumptions), or when offline-learned masks face distribution shift in online fine-tuning (a common problem with offline RL). There is also no analysis of performance trends as the number of prior tasks increases.

- **Potential bias in seed selection:** Using fixed training/testing environment/task configurations (defined by handpicked seeds) without random resampling raises concerns about potential cherry-picking of favorable configurations. Sampling environment/task configuration seeds per run would yield a more robust evaluation.

- **Fairness and runtime concerns:** DIDEC requires an additional offline training phase using trajectories from prior tasks, raising questions of fairness in comparison to baselines and missing runtime comparisons. The experiments are also missing comparisons with other masking-based methods from related work (e.g., weight or input mask learning approaches), which would strengthen claims of novelty.

- **Minor:**
  - Line 87 notation ($π_j j = 1^{i−1}$) is unclear. 
  - Figures (e.g., Figure 9) omit full visualizations for all baselines. 
  - Definition of $d_{obs}$ per environment is missing (is it a flattened grid observation?)?

### Questions
1. Could the authors analyze the learned options (termination sets, input masks, average individual performance) to provide intuition about what behaviors DIDEC extracts? Similarly also analyse option usage during testing and final learned policies.
2. How does performance scale as the number of prior tasks increases?  Does DIDEC saturate or degrade with larger option libraries?
3. What are the failure modes? For example when previous tasks differ significantly in horizon or reward structure?
4. Can the authors assess runtime and computational overhead compared to baselines, given the additional offline mask training phase?
5. How sensitive are results to choice of training/testing seeds? Would randomizing them per run affect DIDEC’s performance consistency?
6. Could the paper include a comparison with prior masking-based transfer approaches to better position DIDEC’s advantages?
7. What steps could mitigate potential distribution shift from offline-learned masks to online adaptation?
8. Please clarify the ambiguous notation (e.g., line 87) and include complete visualization heatmaps for all baselines.
9. Is DIDEC robust to architectures or activations beyond ReLU/tanh, and how would it adapt to continuous input domains?

### Soundness
3

### Presentation
2

### Contribution
3
