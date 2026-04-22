# Training Verifiably Robust Agents Using Set-Based Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Deep reinforcement learning uses neural networks to solve complex control tasks.
However, neural networks are sensitive to input perturbations, which makes their deployment in safety-critical environments challenging and thus their formal verification necessary. This work lifts recent results from formal verification of neural networks to reinforcement learning in continuous state and action spaces. While previous work mainly focuses on adversarial attacks for robust reinforcement learning, we augment reinforcement learning with set-based computing: We enclose all possible outputs for a set of perturbed inputs and compute a gradient set for training, i.e., each possible output has a different gradient. Thereby, we control the size of the propagated sets, yielding favorable worst-case bounds for actions and value functions that enable formal verification across different verification frameworks for up to 9 times larger input perturbations. Our work addresses the gap between state-of-the-art adversarial training methods and formal verification to train verifiably robust agents, making them applicable in safety-critical environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Set-based reinforcement learning (RL) is training procedure where policy is optimized to maximize reward under worst-case adversarial perturbation at each step. The method uses zonotope propagation (“set-based” approach) to compute lower bounds for reward, in analogy with certified training used for image classifiers. In the paper, two such algorithms (SA-SC and SA-PC) are presented. They produce policies that are significantly more robust than previous methods based on heuristic adversarial attacks. However, these set-based RL algorithms also demonstrate a noticeable decrease in both natural reward (without perturbations) and adversarial reward (under attacks), as can be seen in Figure 7. The paper further provides theoretical justification for the proposed algorithms.

### Strengths
This paper presents strong contribution by introducing new policy training algorithm that maximizes worst-case rewards, which is important for safety-critical systems. The experimental evaluation shows clearly that proposed algorithms outperform existing methods in terms of verified rewards on standard dynamical system benchmarks. The proofs included in appendix are brief but correct, and overall the paper is written in satisfactory manner.

### Weaknesses
Despite valuable contribution, this paper has several weaknesses which limit its overall impact. The main issues are following:

1. The term reinforcement learning seems not fully appropriate for described algorithm. The method is designed specifically for control of dynamical systems, and environment setting is never stated explicitly. From examples, it is clear that all benchmarks are deterministic continuous-time systems. Such setup is not typical for reinforcement learning, since it lacks stochasticity which is essential component of RL formulation.

2. The SA-SC algorithm appears unnecessary. Motivation for introducing “exploration noise” in actions (lines 211–214) is unclear. Moreover, experimental results indicate that SA-SC is in most cases outperformed by SA-PC.

3. Section 4 derivation relies on rather arbitrary assumptions about probability distributions. In particular, the claim that expected value over zonotope equals its center is quite restrictive and limits generality of possible distributions.

4. Proposition 4.1 is mathematically correct but its presentation is misleading. More precise statement would be: “There exists a zonotope relaxation of ReLU such that $E_{h_k \sim H_{k}}[h_k] = E_{h_{k-1} \\sim U(\ell_{k-1},u_{k-1})}[L_k(h_{k-1})]$. This zonotope relaxation has $m_k = \ldots$.” The proof holds only for ReLU networks, and this restriction should be written explicitly. Furthermore, the paragraph before Proposition 4.1 creates confusion and even suggests that proposition might be false, probably due to use of different relaxation than the one stated. The proof ends abruptly and does not explain why approximation errors $\underline{d}_k, \overline{d}_k$ are equal to $t_k$, which is necessary for concluding that zonotope center equals expected value of ReLU under uniform input.

5. Figure 7 shows that proposed algorithms achieve lower natural and adversarial rewards, but this observation is not analyzed in text. The same metrics should be reported also for other benchmarks. Certified training methods usually involve trade-off between robustness and natural accuracy, and there can remain significant gap between verified and true worst-case performance. These two aspects require explicit discussion.

6. Colour scheme in Figures 5–11 complicates interpretation. It is difficult to distinguish PA-PC from SA-SC ($\omega=0.5$).

### Questions
The authors may provide comments regarding weaknesses mentioned above.

*List of Typos and Additional Remarks*

- Abstract should begin with “Deep,” because statement “Reinforcement learning uses neural networks” is not correct for all RL methods.
- Replay buffer is denoted as $D$ in Algorithm 1, but as $B$ in text, which causes inconsistency.
- In line 783 symbol used for regression loss differs from that in main text.
- Function $f$ in line 828 should be defined.
- Equality in line 835 appears inaccurate; probably density inside integral must more likely be $U(\ell_{k-1}, u_{k-1})$.

### Soundness
4

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
This paper introduces set-based reinforcement learning for training verifiably robust agents in continuous control tasks. Unlike adversarial training methods, it uses gradient sets computed from entire input perturbation sets rather than single adversarial examples. The approach enables formal verification for up to 9 times larger input perturbations across different verification frameworks.

### Strengths
Significant novel contribution introducing first set-based RL algorithm bridging adversarial training and formal verification. Major theoretical innovation using gradient sets from entire perturbation sets. Results demonstrate up to 9x larger perturbation tolerance compared to state-of-the-art methods while maintaining formal verifiability across multiple verification frameworks (CORA, CROWN-Reach, JuliaReach, NVV). The paper tests across multiple established benchmarks (Navigation Task, 1D/2D Quadrotor, Inverted Pendulum) with proper statistical analysis, ablation studies, and comparisons against multiple baseline methods including recent adversarial training approaches.

### Weaknesses
The Navigation Task incorporates safety through reward penalties rather than explicit constraints during training. The authors acknowledge that formal safety guarantees require separate verification beyond just reward lower bounds, limiting the direct safety claims.

Evaluation focuses primarily on continuous control tasks with relatively low-dimensional state/action spaces. The approach's effectiveness on high-dimensional problems (e.g., image-based RL) or discrete action spaces remains unclear. However, this is a common limitation in this area across majority of approaches.
The method introduces several new hyperparameters that require careful tuning. The paper provides limited guidance on hyperparameter selection across different domains, and the sensitivity analysis is incomplete, potentially limiting practical adoption.

### Questions
How does the computational complexity and memory usage of your set-based training compare to a,b-CROWN's verification approach? Can you provide detailed runtime comparisons on larger networks and discuss the trade-offs between training-time robustness vs. after-training verification?
Does the approach scale to high-dimensional observation spaces (e.g., image-based RL tasks)? How is the scalability compared to ab-crown?


The current work focuses on continuous control; can the set-based approach be extended to discrete action spaces?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a **set-based reinforcement learning (RL)** framework that extends set-based neural network training to the actor–critic setting. The core idea is to **propagate uncertainty sets (zonotopes)** through the policy and value networks to obtain bounds on outputs and gradients. The method defines a **set-based regression loss** for the critic that penalizes both prediction error and set diameter, and a **set-based policy gradient** for the actor that encourages compact output sets. The approach aims to yield RL policies that are **verifiably robust** under input perturbations. Experimental results on navigation and control benchmarks demonstrate improved verified performance and robustness compared to adversarially trained baselines.

### Strengths
**Originality**  
- The work is an interesting and timely attempt to unify **formal verification and reinforcement learning**, using set-based propagation within the training loop.  
- The idea of **gradient sets** and integrating them with actor–critic architectures is novel and potentially impactful.

**Quality**  
- The theoretical exposition (e.g., use of zonotopes, set propagation through affine and activation layers) is mostly sound.  
- The proposed framework connects well to existing verification methods, showing that trained policies can be more amenable to reachability-based certification.

**Clarity**  
- The algorithm is clearly presented, and the basic zonotope operations are described in a straightforward way.  
- The empirical section reports verified robustness metrics under multiple verification toolboxes, which adds credibility.

**Significance**  
- The topic—training verifiably robust RL agents—is important and underexplored. The framework could stimulate further research into **hybrid learning–verification paradigms**.

### Weaknesses
1. **Over-approximation accumulation**  
   - The framework propagates set over-approximations step-by-step in an RL setting, but there is **no theoretical bound** on how the set diameter grows over time. Without such analysis, the reachability sets may become **too conservative** (i.e., trivial or uninformative), limiting practical use.

2. **Goal reachability and certification**  
   - The paper uses over-approximated reachable sets to argue about goal attainment, but **overlap with the goal set does not imply actual reachability**. There is no mechanism ensuring that the system trajectory truly enters the goal region.

3. **Complexity and scalability**  
   - The authors do not analyze the **computational complexity** of set propagation or describe how generator matrices are pruned to maintain tractability. The absence of any **scalability discussion** (time/memory vs. state dimension) is concerning.

4. **Notation and clarity issues**  
   - The same symbol \(\theta\) appears at different lines (e.g., 105 vs. 124) for possibly different objects—this causes confusion.  
   - In Eq. (11), the variable \(t\) appears in \(\nu(s_t, t)\) but is not used later; this seems meaningless unless time-varying noise is considered.

5. **Activation coverage and approximation errors**  
   - Proposition 4.1 only discusses ReLU networks. The **approximation error bounds** for non-ReLU activations are not analyzed.

6. **Reward design justification**  
   - The reward function used at line 418 lacks motivation or theoretical connection to the claimed “verified performance.” The impact of reward shaping on the certification outcome is not studied.

7. **Runtime and scalability reporting**  
   - The paper omits **training runtime**, **state/action dimensions**, and **network sizes**. Table 1 shows only verification time, making it difficult to evaluate the **overall efficiency**.

### Questions
1. **Over-approximation accumulation**  
   - How do you ensure that the reachable sets remain non-trivial as uncertainties propagate across time?  
   - Is there a formal bound on the diameter growth of the sets, e.g., as a function of the Lipschitz constants of dynamics and policy?

2. **Propagation until termination**  
   - Does “termination” refer to a fixed time horizon or a goal-reaching condition?  
   - How is the reachability propagation stopped, and how is goal satisfaction formally defined?

3. **Goal set overlap vs. actual reachability**  
   - Since the approach relies on outer approximations, how can we be sure the agent truly **reaches** the goal set rather than merely overlapping with it?

4. **Parameter notation**  
   - Are the two symbols \(\theta\) (line 105 and line 124) referring to the same network parameters? If not, please rename one (e.g., \(\vartheta\) or \(\Theta\)).

5. **Equation (11)**  
   - What is the purpose of \(t\) in \(\nu(s_t, t)\)? If it is redundant, it should be removed for clarity.

6. **Equation (12)**  
   - Please clarify what \(c\) and \(G\) represent and how they are derived from the actor network’s output enclosure. Are they the zonotope center and generator matrix computed via affine propagation?

7. **Proposition 4.1**  
   - How does the result generalize to **non-ReLU activations** such as \(\tanh\)?  
   - What are the quantitative approximation errors introduced by these nonlinearities?

8. **Complexity and computation**  
   - What is the time complexity of a single forward pass and backward pass in your set-based framework?  
   - Do you apply any **order-reduction heuristics** to control the explosion of zonotope generators?

9. **Reward design**  
   - Why was the specific reward function in line 418 chosen?  
   - Does it have any special properties that make verification easier or bounds tighter?

10. **Runtime and scalability**  
    - Please report wall-clock training time, state/action dimensions, and network sizes for all tasks.  
    - How well does the method scale to high-dimensional or long-horizon RL problems?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a **set-based reinforcement learning (RL)** framework that integrates *set-based neural network training* with the *actor–critic paradigm* to achieve **formally verifiable robustness**. The method introduces *gradient sets* to account for uncertainty propagation through neural networks: each possible output under input perturbations has a corresponding gradient. By minimizing the size of propagated sets during training, the resulting policy and value functions exhibit improved *worst-case guarantees* and can be verified using standard reachability tools such as CORA. The experiments across benchmarks (e.g., 1D/2D Quadrotor, Inverted Pendulum, Navigation Task) demonstrate improved verified performance and robustness compared to adversarially trained agents, extending recent work on set-based training for feed-forward networks.

### Strengths
- The work extends **set-based neural network training** to the reinforcement learning setting, introducing gradient sets into both actor and critic updates.  
- It offers an elegant synthesis of *verification-oriented training* and *policy optimization*, bridging robust RL and formal methods.  
- The proposed set-based regression loss and policy gradient are mathematically grounded in probabilistic reasoning, linking the *likelihood–prior decomposition* with set diameter minimization.  
- The results showing successful verification across multiple tools (CORA, CROWN-Reach, JuliaReach, NNV) indicate cross-framework generality.  
- The structure of the paper is clear and self-contained, with strong visualizations (e.g., Figs. 1, 3, 5–7) and detailed pseudocode (Algorithm 1).

### Weaknesses
1. **Theoretical limitations of over-approximation**  
   - The approach relies on *outer approximations* of reachable sets, yet the paper provides no quantitative analysis of **set overgrowth or error bounds**.  
   - This makes it unclear whether the final reachability sets meaningfully constrain the true behavior, especially over long horizons.

2. **Lack of computational analysis**  
   - While the framework is elegant, there is no complexity or runtime discussion of *set propagation* or *gradient set computation*.  
   - The claim of scalability (e.g., to 2D Quadrotor and Hopper-v2) is qualitative and unsupported by profiling data.

3. **Empirical evidence limited in scope**  
   - The experiments demonstrate robustness but not **training stability** or **verification time trade-offs** in detail.  
   - It remains unclear whether set-based training slows convergence compared to standard or adversarially trained DDPG.

### Questions
1. **On over-approximation accumulation**  
   - How does the set diameter evolve across time steps?  
   - Do you apply zonotope order reduction or pruning to prevent explosion of generator dimensions?

2. **Verification tightness**  
   - Can the authors quantify the difference between the *verified lower bound* and the *empirical return*?  
   - What percentage of trajectories are “verified tight” (i.e., within a small margin)?

3. **Scalability**  
   - What is the computational cost of one training epoch compared to adversarial training (e.g., FGSM-based)?  
   - How does verification time grow with network depth or state dimension?

4. **Reward function and verification metric**  
   - Why was $r(s,a)=w^\top|s-s^*|$ chosen? Would other smooth reward forms (e.g., quadratic penalties) affect the verified performance definition in Eq. (28)?  

5. **Theoretical extension**  
   - Could the framework be adapted for stochastic policies or discrete action spaces?  
   - Is there an underlying connection between your set-based gradient and **Lipschitz regularization** approaches?

6. **Practicality**  
   - Can the proposed method handle **nonlinear system models** beyond the evaluated tasks (e.g., with unmodeled dynamics or contact)?  
   - Have you observed instability due to conservative set updates in high-dimensional tasks?

### Soundness
3

### Presentation
3

### Contribution
2
