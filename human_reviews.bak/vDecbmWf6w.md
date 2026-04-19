# Zero-Shot Offline Imitation Learning via Optimal Transport

- Decision: Reject
- Scores: 3, 6, 8, 6

## Abstract
Zero-shot imitation learning algorithms hold the promise of reproducing unseen behavior from as little as a single demonstration at test time.
Existing practical approaches view the expert demonstration as a sequence of goals, enabling imitation with a high-level goal selector, and a low-level goal-conditioned policy. 
However, this framework can suffer from myopic behavior: the agent's immediate actions towards achieving individual goals may undermine long-term objectives.
We introduce a novel method that mitigates this issue by directly optimizing the occupancy matching objective that is intrinsic to imitation learning. 
We propose to lift a goal-conditioned value function to a distance between occupancies, which are in turn approximated via a learned world model.
The resulting method can learn from offline, suboptimal data, and is capable of non-myopic, zero-shot imitation, as we demonstrate in complex, continuous benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents a method for zero-shot imitation learning with a single demonstration (without action labels or partial trajectories consisting of a sequence of goals). A method called ZILOT that learns to plan and act according to the goals is proposed. Firstly, ZILOT learns a dynamics model (similar to TD-MPC2) using a dataset of transitions which can be sub-optimal. For planning using the demonstration, a non-Markovian method is employed to match the occupancy of the partial demonstration and the state-visitation distribution of the policy. The discrepancy between occupancies is computed using Optimal Transport which requires value functions: V to get reachability of a goal state and W to get the steps taken between goals. The non-Markovian planner uses this discrepancy to select the best action. Experiments conducted on multiple benchmarks show that ZILOT is better at W1 distance between expert and policy occupancies and better at generating trajectories that follow the order of goals in the demonstration.

### Strengths
- The idea of recovering expert behaviors from partial trajectories with subgoals and without action labels is interesting and challenging. 
- The drawback of prior methods that used goal-conditioned policies for this task is discussed well to motivate the proposed method.

### Weaknesses
- Many design choices of the proposed algorithm are not clear. Why does ZILOT use a non-Markovian policy as I believe the expert policies are Markovian? How is the function $\phi$ that maps states to goals defined / learned?
- The experiments are not convincing as the baselines like FB-IL [1] and OTR [2] are missing. It is not clear why the task success rate or the average returns are not used for comparing methods. 
- The limitations of the methods are not discussed. The only limitation presented is in Sec 7 that describes the dependence on a learned dynamics model.

### Questions
1. The paper is motivated by the fact that methods using goal-conditioned policies for zero-shot IL fail (Proposition 1 is used to support this argument). However, ZILOT plans using all the goal states in the demonstration. What if the goal-conditioned policy is given the expert demonstation as context? This way the goal-condition policy might discard the bad states to achieve a goal (shown in Proposition 1). Moreover, this would lead to learning a Markovian policy.
2. The goodness of the proposed method depends on the planning horizon, and the paper discusses that ZILOT can be myopic too without a long planning horizon. This limits the applicability of the method in many real world tasks when planning for a long horizon. Is there a way to mitigate this by estimating the visitations using some form of bootstrapping? 
3. Since the method uses optimal transport and cites OTR [2] in the Related work, I feel it should be added as another baseline that uses the expert states and the sub-optimal dataset to recover a reward function and train a policy over it. 
4. Since sub-optimal data is used to learn the value-functions, can optimal V, W be recovered? If not, it is not highlighted how will this gap affect the final performance? 
5. The abstract talks about zero-shot IL with existing practical methods viewing demonstrations as a sequence of goals. I feel this is not true as FB-IL [1] does zero-shot IL from a single demonstration. Moreover, I feel FB-IL should be a baseline. Although, ZILOT deals with demonstrations with a subset of states, I feel using Eq 8 in [1] should recover the policy with partial demonstrations.

### References
[1] Pirotta et al., Fast Imitation via Behavior Foundation Models. ICLR'24\
[2] Luo et al., Optimal transport for offline imitation learning, ICLR'23

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an approach for zero-shot imitation -- use a single expert trajectory to extract a policy to mimic this task. The general approach is to solve an optimal transport problem (equipped by a goal-reaching cost function) with a model that can be used to estimate the policy's stationary distribution. Across a number of tasks (fetch, halfcheetah, pointmaze), the proposed approach can learn useful skills.

### Strengths
The paper studies a very important problem: the ability to quickly learn new skills and policies from few expert demonstrations is a much-desired property of policies. The method is conceptually simple and well-motivated theoretically.

I found the paper to be easy to read, and the background for all relevant concepts was motivated and well-introduced (within Sec 2-4)

Figure 4 clearly demonstrates the main improvement over the sub-goal approach, where the MPC + CIs overshoot the puck when going to the first subgoal.

The experiments are relatively thorough (both in the main paper, and the Appendix), indicating improvement over other model-based zero-shot imitation methods.

### Weaknesses
I had no major objections to the paper; 

However, it would have been nice to see comparisons to other approaches to zero-shot imitation (e.g. along the line of FB representations. Another axis that could improve the paper would be to see a wider range of downstream tasks (requiring the synthesis of more diverse motions),  perhaps those from (Frans et al)

Minor nit: There is a lot of content in the Appendix, but it is not clearly linked within the main paper. I would encourage the authors to link and discuss this content within the main paper, so that it is not missed by a reader.

### Questions
1. How "in-distribution" are the inference tasks compared to the trajectories used to train TD-MPC? How does the quality of learned policy change we try to imitate more "OOD"  expert behaviors?

### Soundness
3

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
2

### Summary
This paper identifies an existing problem with goal-conditioned imitation learning: an agent that optimized to achieve individual goals can undermine long-term objectives. Instead of learning a goal-conditioned policy, ZILOT learns a dynamics model with a goal-conditioned value function. It uses optimal transport to minimize the divergence between the rollout policy and expert demonstration.

Experiments show that ZILOT imitates demonstration more closely compared with MPC and goal-conditioned policy.

### Strengths
- Identifies an existing problem of goal-conditioned policy learning and proposes a solution based on optimal transport.
- Demonstrate success of ZILOT on a range of locomotion and manipulation tasks and show better alignment with demonstrations.

### Weaknesses
- Lack of explanation and analysis of the myopic behavior in goal-conditioned imitation learning and why ZILOT bypass those limitations.
- Baselines are relatively simple. How does it compare with some diffusion-based methods, e.g. Diffusion Policy?

### Questions
- Instead of conditioning on an intermediate goal, can we condition on a sequence of goals to mitigate myopic behavior?
- Instead of optimal transport to minimize the distance between distributions, how does it compare to diffusion-based methods?

### Soundness
3

### Presentation
2

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
The authors introduce a new method for zero-shot imitation learning based on optimal transport.  
The approach involves combining a modified goal-conditioned TD-MPC2 algorithm with optimal transport. The authors begin by training goal-conditioned value functions V and W, as well as a dynamics model obtained from TD-MPC2. Then they run the OT process using the Sinkhorn algorithm, given V, W, P, and the expert trajectory, to compute the cost matrix and transport matrix.

### Strengths
The method shows example of non-myopic behaviour compared to the baselines. The proposed approach does not depened on availability of actions in the expert trajectory. The proposed approach works as zero-shot "agent correction" policy. Authors evaluated their method on variety of different environments.

### Weaknesses
The proposed method requires both the learned world model and expert demonstrations. 
In practice, the training world model requires a lot of resources and either access to a simulator or a large collection of datasets.

Though, the authors positioned their paper as "zero-shot IL" it is still far away from "fair" zero-shot since it requires some conditions for the method to work:
1. Access to the offline dataset 
2. Trained Transition model
3. Expert trajectory.

### Questions
1. What is the additional computational cost introduced by solving OT problems compared to running just the planners?
2. How does the method scale when there are more than 1 demonstration? Does the method have a problem with multimodal behavior in the bigger data collection?
3. Can the authors apply the method to cross-domain / cross-embodiment settings?
4. What if the expert policy is different from the resulting policy in TD-MPC? I.E can we observe a distributional shift if our expert policy comes from an environment with slightly different dynamics, for example, cross-embodiment (x-magical https://github.com/kevinzakka/x-magical)?
5. Can the current method be altered to remove the model-based approach and replace it with the offline dataset of video collection? I.e why can't we learn value functions from just video data without rewards and actions and then use it in OT?

### Soundness
3

### Presentation
4

### Contribution
3
