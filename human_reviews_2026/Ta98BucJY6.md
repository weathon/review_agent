# DHP: Discrete Hierarchical Planning for Hierarchical Reinforcement Learning Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Hierarchical Reinforcement Learning (HRL) agents often struggle with long-horizon visual planning due to their reliance on error-prone distance metrics. We propose Discrete Hierarchical Planning (DHP), a method that replaces continuous distance estimates with discrete reachability checks to evaluate subgoal feasibility. DHP recursively constructs tree-structured plans by decomposing long-term goals into sequences of simpler subtasks, using a novel advantage estimation strategy that inherently rewards shorter plans and generalizes beyond training depths. In addition, to address the data efficiency challenge, we introduce an exploration strategy that generates targeted training examples for the planning modules without needing expert data. Experiments in 25-room navigation environments demonstrate a 100% success rate (vs. 90% baseline). We also present an offline variant that achieves state-of-the-art results on OGBench benchmarks, with up to 71% absolute gains on giant HumanoidMaze tasks, demonstrating our core contributions are architecture-agnostic. The method also generalizes to momentum-based control tasks and requires only $\log N$ steps for replanning. Theoretical analysis and ablations validate our design choices.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses long-horizon visual planning by proposing a discrete hierarchical planning framework. The authors train an encoder–decoder module to learn a discrete latent space, upon which a high-level policy plans subgoals recursively. The approach builds on prior work of Director for goal-conditioned visual planning.

### Strengths
1. The problem of hierarchical visual planning in long-horizon settings is both relevant for the community.

2. The idea of using a discrete latent space to constrain high-level planning is promising and can potentially improve sample efficiency.

### Weaknesses
1. Limited and inconsistent evaluation.
    - The experiments are restricted to a single environment (25-room navigation). Prior works such as GCP evaluated both 9-room and 25-room variants, as well as FrankaKitchen in LEXA's paper, which are missing here.

    - In RoboYoga, the baselines are not fully reported; only the proposed method’s results are shown, and its performance on RoboYoga–Quadruped does not appear to improve.

    - The paper reports episode rewards, but it is unclear whether these are evaluated using exploration or planning policies. If an explorer is active, the interpretation of "the sharp rise indicates a switch from exploration to planning" becomes confusing. The mechanism of this switch need clarification.

2. Conceptual and presentation clarity.

    - The paper assumes prior familiarity with Director and RSSM, making it difficult for readers not familiar with these models to follow. Section 2.1 should include a concise overview of these components.

    - The notation in the planning section (e.g., $s_0$) is confusing, as it initially suggests the initial state rather than the subsequent state in the planning trajectory.

    - The description of GC-Director is unclear.

3. Insufficient ablations and analysis.

    - The choice of tree depth (D=8) is not justified. Why was 8 chosen when D=3 yields comparable reward and fewer steps?

    - There are no ablations analyzing the impact of removing or modifying the discrete CVAE (e.g., using a continuous version).

    - The paper mentions providing memory of past states as additional input, even though the RSSM already encodes recurrent state, why is this necessary, and what empirical effect does it have?

    - The paper claims GC-Director fails due to task complexity, but LEXA, which lacks explicit planning, performs adequately. This requires more rigorous reasoning or empirical support.

4. Lack of experimental rigor.

    - The final performance in Figure 7(a) is not substantially higher than LEXA, which does not use hierarchical planning.

    - The similarity metric differs from LEXA: the paper uses cosine_max instead of temporal similarity, even though LEXA(Cos) performed worse. why not using the temporal similarity?

    - There is no clear explanation of how SAC is used to optimize the managers and whether this differs from Director’s joint training scheme.

**Minor Suggestions (Not affecting the score)**
1. Improve Figure 1 so that the illustration aligns with its textual description. A well-designed figure should be interpretable without heavily relying on the section text. Consider showing both the training and exploration phases.

2. Consider adding “Visual” to the title to better situate the work in the visual planning literature.

3. Clarify the notation for trajectory states and unify the explanation of algorithms for smoother reading.

### Questions
1. What justifies the “sharp rise” of DHP performance in Figure 7? How does this correspond to switching from exploration to planning?

2. Regarding the CVAE:

    - Is it identical to the one used in Director? If so, what constitutes the novelty of GCSR?

    - Could you show ablations comparing the discrete and continuous variants of the CVAE?
3. How do you choose $\Delta_R$?

4. How can DHP achieve higher episodic rewards than LEXA despite comparable or shorter episode lengths?

5. Why was cosine_max used as the similarity measure rather than the temporal similarity metric used in LEXA?

6. Could you elaborate on SAC’s role in optimizing the high-level policy, given Director trains managers jointly with workers?

7. Please clarify Algorithm 6’s dataset collection process, does it occur sequentially with Algorithm 1, and can these be combined for clarity?

6. Appendix D’s sample trajectories should explicitly indicate generated subgoals for interpretability.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method that replaces continuous distance estimates with discrete reachability checks to evaluate subgoal feasibility. It recursively constructs tree-structured plans by decomposing long-term goals into sequences of simpler subtasks, using an advantage estimation strategy that inherently rewards shorter plans.

It seems clearly presented, and shows a clear improvement on a baseline task although performance improvements seem example dependent.

### Strengths
Reachability (binary) may avoid coupling to brittle distance metrics and naturally handles disconnected regions, as the authors claim.


Contraction property of the return operators.


Successful results on 25-room benchmark and competitive path lengths, where ablations show training with shallow depths still provides advantage of the proposed methods.

### Weaknesses
Easy to understand the flow and contribution of the paper. 

The resulting model performs expertly on the standard 25-room task than the current SOTA approaches, but not on others. 

It would be quite sensitive to model error.  The cosine_max similarity check may judge that two similar-looking states are close even though the underlying configurations differ. 

The paper trains a static-state MLP as an approximation, so the planning can be sensitive to such approximation.

### Questions
How often does imagination mark unreachable subgoals as reachable?  What if the world model quality is low? 

Examples such as maze-like environments with partial observation only? 

It seems that the memory can be a limiting factor for complex problems. What if the memory needs to be truncated?

What are the specific cases where min-child is especially helpful?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The proposed method can be summarized as follows.
Given a task, it bisects the desired trajectory by generating mid-point states. 
Those mid-point states are stored in a binary tree. 
In the experiment, a visual navigation domain with 25 rooms is used, and the tree was tested up to a depth of 3.
The state encoder uses RSSM, and SAC trains the RL policy with policy gradients.
The main paper doesn't show the overall algorithm for training high-level policy/exploration and the lower-level policy/exploration.

### Strengths
Improved performance on the 25-room navigation domain from 90% rate to 100% as shown in Table 1.
The proposed approach improves the variance of the average episode length.
The proposed idea is simple compared with existing HRL methods.
It learns to bisect the initial state and the goal state, using the midpoint state for learning lower-level policy functions.

### Weaknesses
The application of the proposed approach is limited, and it is not clear how the subtask generation returns meaningful subtasks/subgoals.

Most of the related works were developed until 2020, except for a few.
There are many missing HRL approaches from 2020
such as option-based HRLs, neuro-symbolic planning, and RL, or identifying subtasks/subgoals.

Here is a partial list of such approaches (and there are more)
* Reward machines: Exploiting reward function structure in reinforcement learning
* Hierarchical Reinforcement Learning with AI Planning Models
* Reinforcement Learning with Option Machines
* Integrating Symbolic Planning and Hierarchical Reinforcement Learning for Robust Decision-Making
* Learning to represent action values as a hypergraph on the action vertices
* Learning Parameterized Task Structure for Generalization to Unseen Entities
* Fast inference and transfer of compositional task structures for few-shot task generalization
* Unsupervised Task Graph Generation from Instructional Video Transcripts.

### Questions
Q1 How does the trained policy generalize?
What's the impact of permuting/modifying connections all patches of images in the 25 rooms?
How many steps are needed to move from the center of each room to the end via a straight line?

Q2 In Figure 7, all figures show a sharp transition around 3M steps for the HDP configuration.
Could you explain why?

Q3 What is the DHP with the Depth 1 configuration?
Is it dividing the initial-goal state with a mid-point state?

Q4 How frequently does this trajectory bisection happen?

Q5 As the proposed method can traverse the bi-sected tree in a depth-first manner, how does the return estimate from the whole tree give an advantage?

Q6 Does the proposed approach identify re-usable/interpretable sub-goals?

Q7 For state representation learning, what are the requirements for the computational resource/data?
Does it learn online while learning the RL policy?

Q8 What limits the proposed approach from being applied to the problem domains used in the related papers listed above?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a HRL algorithm that learns to propose sub-goals that are based on discrete notion of reachability rather than relying on continuous distance metrics in an embedding space. They propose a tree-structured decomposition for generating intermediate subgoals (with subgoals occurring around half of the time interval between start and desired goal). The method demonstrates improved success rates on long horizon tasks.

### Strengths
- The motivation is clear and the method outlined is mostly clear (see a few clarification questions below)

### Weaknesses
- The training procedure involves quite a few moving components. Especially the need for extensive exploration to ensure the CVAE offers enough coverage to select suitable sub-goals. This indicates a dependence on the base director architecture to be good enough to reach rewarding trajectories from which the explorer can further improve coverage, so might be critically dependent on the task’s reward structure.  
  - The paper could benefit from a clear pseudocode / pictorial view of various stages of training.  
- The predominant evaluation is limited to just a single domain of 25 room navigation. While still informative the paper could benefit from the inclusion of more long horizon environments typically benchmarked in HRL (AntMaze, OGBench tasks).

### Questions
- In section 2.2, are you using multiple CVAEs for different time scales (or tree depth) $Q$? Or are the encoder/decoder networks conditioned on the timescale?
- In section 2.3.2, the reachability reward is based on model-predicted reachability i.e. by simulating the worker policy inside the RSSM. Could you clarify if the planning policy is trained after the worker and RSSM are trained using the exploration policy, or does the training require some special scheduling? How is the threshold parameter $\\Delta_R$ chosen – is it dependent on the maximum depth of the tree?

### Soundness
2

### Presentation
2

### Contribution
2
