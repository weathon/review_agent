# STAN: A Spatio-Temporal Attention Network for Space Debris Multistage Collision Avoidance

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
The rapid expansion of space missions has led to an exponential increase in space debris, posing severe threats to spacecraft. Existing approaches struggle to handle multistage collision risks in cluttered orbital environments, and the use of continuous low-thrust propulsion further complicates avoidance planning. To address these challenges, we propose the Spatio-Temporal Attention Network (**STAN**), which employs novel Spatio-Temporal Attention (**ST-Attention**) layers in place of conventional attention mechanisms. STAN encodes satellite-debris pairs and integrates time and distance into attention weight computation, enabling the model to generate context-aware low-thrust maneuvers. The model is trained using deep reinforcement learning across four representative multistage collision scenarios, jointly optimizing collision probability, fuel consumption, and orbital deviation. Experimental results show that STAN outperforms baseline methods in safety performance, fuel efficiency, and orbit preservation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces **STAN**, a reinforcement‑learning policy for satellite–debris collision avoidance under continuous low‑thrust. The key architectural idea is a Spatio‑Temporal Attention (ST‑Attention) layer that injects a physics‑motivated bias—computed from each object’s distance of closest approach (DCA) and time to closest approach (TCA)—into scaled dot‑product attention over a variable‑sized set of debris. STAN outputs both a thrust vector and a termination signal; training uses PPO across four scenario families (single, strict multistage, probability‑based multistage, complex). The manuscript claims large improvements in reward, fuel, and collision probability versus FC/CNN/LSTM baselines, with ablations for attention and termination.

### Strengths
I think the problem choice is timely and practically important: multistage conjunction handling with **continuous** low-thrust and variable-length inputs is closer to real electric-propulsion operations than impulsive-burn abstractions. The physics-aware inductive bias—folding DCA/TCA into attention—offers an intuitive way to prioritize likely threats without heavy modeling; the mechanism is clearly written with $b_i=-\sum_j\gamma_j\Phi_{ij}$ and $W=\mathrm{softmax}(QK^\top/\sqrt{D}+\lambda B)$. Conceptually, the **termination head** is a good interface for fuel economy in continuous-thrust settings. I also appreciated the scenario taxonomy and visuals, and that some PPO hyperparameters and environment details are spelled out. In terms of significance, a robust policy for variable-$N$ debris could matter for on-board autonomy. And in terms of clarity, the high-level pipeline and figures communicate the intent well, even if some equations/indices need tightening.

### Weaknesses
**1) Termination gating appears inverted and non-differentiable.**  
Equation (7) computes the final action as
$ \Delta v_t^{\text{final}} = H(p_{\text{done}}-0.5)\,[\alpha S(t)+(1-\alpha)a] $,
which *enables* thrust when the model believes the task is “done,” contradicting the prose (“stop thrusting when appropriate”) and discarding the continuous probability via a hard Heaviside. I strongly recommend flipping the logic (e.g., multiply by $1-p_{\text{done}}$ or use a smooth sigmoid gate) and **re-running all experiments**.

**2) ST-Attention bias broadcasting is ambiguous and may be ineffective.**  
You define a per-debris bias $b_i$ from $\Phi_i=(d_i,t_i)$ and then state “each row is identical ($B_{ij}=b_j$)” before adding $\lambda B$ to the logits. If $B_{i*}$ is row-constant, adding the same constant to a row cancels in the row-wise softmax; if it’s column-constant (per-key), the bias is global and not pairwise. Please pin down the tensor shapes and show an ablation that the bias actually changes attention maps and outcomes.

**3) Internal inconsistencies weaken trust.**  
The manuscript alternates between a collision-probability threshold of $0.02$ vs $0.002$; the ablation-section narrative claims rewards “around $750$” in the 10-debris probabilistic setting while the table lists $\approx 276$; scenario size is “up to $116$” debris in one place and $1178$ elsewhere. Some tables also show FC outperforming STAN on reward (and reporting implausibly tiny probabilities) despite the text claiming consistent dominance. Please take a look at these items.

**4) Collision-probability ($P_c$) modeling is under-specified and numerically suspect.**  
The analytic form lacks the covariance definitions $\sigma_R,\sigma_S,\sigma_W,\sigma_{SW}$ and how they are estimated/propagated. Reported values span $10^{-34}$–$10^{-1}$, which looks implausible without a careful uncertainty model and numerically stable evaluation. Safety claims based on $P_c$ are hard to interpret or reproduce.

**5) Baselines do not represent current best set/attention policies (and may be unfair).**  
Given STAN’s set-attention core, comparisons against FC/CNN/LSTM are not enough. Please include capacity-matched **vanilla Transformer/Set Transformer** and consider a simple **model-based** low-thrust planner, with clearly documented padding/masking for permutation robustness.

**6) Physics realism gap.**  
Everything uses a two-body model, yet “complex” scenarios reference real-world (e.g., CelesTrak) data. Over a 1.2-hour window some regimes may be fine, but ignoring J2/drag (especially in LEO) and omitting the TLE→state pipeline (e.g., SGP4) risks mischaracterizing DCA/TCA and $P_c$.

**7) Methodology and statistics need tightening.**  
Results appear single-seed; no mean ± std or clear train/test splits. The “strict multistage” scenario is policy-dependent (training continues until success, then the next collision is injected on the updated trajectory), which risks leakage between training and evaluation. Runtime scaling claims should be backed by matched FLOPs/wall-clock under identical widths/batching.

**8) Reproducibility and clarity gaps.**  
Key actor sizes, number of heads/embedding $D$, normalization/dropout, and values/schedules for $\lambda,\gamma,\alpha$ are missing. Orbit deviation is defined on normalized elements but table captions/values sometimes read like kilometers.

### Questions
Every item raised in the Weaknesses section can be viewed as a question for the authors.
I may well be mistaken on several of these points, and I would sincerely appreciate clarification or correction wherever appropriate.
If the authors can address or resolve even part of these concerns—whether by showing that I misunderstood something or by providing additional detail—it would be very helpful.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes STAN (Spatio-Temporal Attention Network), a reinforcement learning-based policy network combined with attention mechanisms for continuous low-thrust spacecraft collision avoidance in complex, multi-debris orbital environments. The work addresses key limitations in prior methods, including the inability to handle variable numbers of debris, and insufficient integration of physics-informed risk metrics into learning-based control.

### Strengths
ST-Attention design combines self-attention with physically-informed bias to prioritize debris threats effectively. It addresses the gap in standard attention mechanisms that lack domain knowledge.

Architecture - Strengths:
1. Combines learned and domain knowledge through fusing of self-attention for complex interactions with explicit physics-informed bias features, capturing both debris correlations and collision risks.

2. Scalable to arbitrary debris counts through encoding features and using attention allows the model to handle variable numbers of debris objects and multi-threat scenarios.

3. Learnable weighting setup let the network adjust the importance of physical features relative to learned embeddings, improving adaptability to different orbital contexts or debris densities.

### Weaknesses
Architecture - Limitations:

1. Physics-aware attention bias incorporates the distance of closest approach (DCA) and time to closest approach (TCA) as a learnable bias, the model ensures attention is grounded in domain-relevant risk indicators, not just learned embeddings. Model may be more biased towards immediate critical threats than long-range operational settings such as mission fuel consumption.

2. Uniform broadcast of bias across all columns to form 𝑁×𝑁. This assumes the physical importance of debris i affects all pairwise interactions equally, which may ignore interaction-specific relationships between debris pairs (e.g., cross-collision influence).

3. Mean pooling across debris dimension reduces to a single vector 𝑓 and  both heads rely on it. May lose fine-grained per-debris information, limiting the decoder’s ability to make nuanced, individual maneuvers for specific threats.

4. Reward penalizes collision probability but does not explicitly enforce safety constraints. Policy may occasionally select risky maneuvers if they increase cumulative reward. Does not explicitly account for sensor noise or uncertainty in debris position, which could make the reward function misleading in real operational settings.

5. Some terms may overlap in effect, e.g., minimum distance softness (p_s) and collision probability (p_c) are related; summing both may overweight certain safety aspects.

Experiments - Limitations:

1. Current approach considers Two-body orbital dynamics only and it neglects higher-order perturbations such as J2 (Earth oblateness), atmospheric drag, solar radiation pressure, and third-body effects (Moon, Sun).

2. Multistage collisions are simulated progressively, but each scenario cover only a short timeframe (e.g., 1.2 hours in experiments). The simulation is for 1.2-hour mission window and does not test long-term maneuver planning, cumulative collision risk, or fuel optimization over multiple orbital periods. Policy generalization to extended missions or sequential conjunctions is unclear.

3. Spacecraft near-circular orbit, debris initialized along same orbit. Unrealistic debris distribution case, the real debris is in multiple orbital planes, inclinations, and eccentricities.

4. Exact positions and velocities of debris are used. Ignores sensor noise, tracking errors, and uncertainty in debris catalog.

### Questions
The authors can address the experimental limitaions concerns on choosing shorter time frame for experiments, why didn't consider the smaller pertubations in the orbital dynamics, circular initialization of debris orbits and sensor noise in the modelling scenrios.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose STAN, a Spatio-Temporal Attention Network for avoiding space debris with a deep reinforcement learning approach. STAN's architecture adds a novel spatio-temporal attention mechanism that includes the distance to the closest approach and time to the closest approach as an added bias to the computation of the attention weights. Space debris avoidance is formulated as a Markov Decision Process (MDP) and solved with Proximal Policy Optimization (PPO). The model is evaluated on a simulated environment with multiple debris objects and compared to  baselines with different architectures (CNN, MLP and Transformer). The results show that STAN outperforms the baselines in terms of collision avoidance rate and fuel consumption and that the added spatio-temporal attention mechanism outperforms a standard Transformer architecture.

### Strengths
- The paper is well written and easy to follow.
- Nice illustrations help to understand the proposed method and results.
- The proposed spatio-temporal attention mechanism is novel and well motivated.
- The spatio-temporal attention mechanism is shown to improve performance over a standard Transformer architecture.

### Weaknesses
- The architectural innovation is not compared / discussed to other algorithms in the field of space debris collision avoidance. 
- The claims in the introduction from l. 41 to 46 could be supported with references.
- Stick with constant capitalization of figure and section references.
- Define acronyms at first use (Spatio-Temporal Attention (ST-Attention) in l.49) and then continue with the acronym.
- The reward function definition could be made reader-friendly by including the variable names directly in the equation.
- The authors space key seems to be broken, white spaces are missing in l. 177, 180, 182, 242, 329, 340, 351, 356, 342, 390, 392, 393, 447, 448, 449 and 478.

### Questions
- How are hyperparameters selected? How many seeds were used for the experiments? What are the standard deviations?
- Can you share the codebase and the test environment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a solution for multistage collision avoidance. 
The solution utilizes an attention layer with temporal encoding, which incorporates the time to closest approach and the miss distance between the pair of objects.
The model is used as an RL policy for satellite maneuvering. The model was compared with CNN, LSTM, and FC architectures in an undisclosed environment.

### Strengths
# originality

The paper proposes a space-time (miss distance x time to closest approach) encoding added to the representation after dot product attention, applied to RL for collision avoidance maneuvers in lower Earth orbits. The model has two heads, one with the 3D thrust vector and another one with and indicator of maneuver. 

# quality 

The paper illiustrates the architecture of the solution with a clear Fig. 1.

# clarity

The paper is written in clear English

# significance

The problem of complex collision avoidance is important for space sciences and for RL for real-world critical systems

### Weaknesses
# originality
The proposed policy architecture performance in the PPO framework depends on the critic network. This is not covered in the paper or annexes. 

## Physics fusion
In Section 4.2, the paper attributes the b term as a "physical prior". As these numbers (TCA and miss distance) are estimated using statistical assumptions rather than physical laws, it is challenging to consider them more than features or time-space encodings.

# quality 


## Related work

Relevant papers are not considered, nor compared against. For example:
- Kolosa (2019) https://www.proquest.com/openview/320685ea5feac29eb871b0c9f169d002/1?pq-origsite=gscholar&cbl=18750&diss=y
- Miller (2019) https://www.researchgate.net/profile/Richard-Linares/publication/331135625_LOW-THRUST_OPTIMAL_CONTROL_VIA_REINFORCEMENT_LEARNING/links/5c67324b299bf1e3a5abe460/LOW-THRUST-OPTIMAL-CONTROL-VIA-REINFORCEMENT-LEARNING.pdf
- Herrera (2020) https://www.proquest.com/openview/efe02b87a62929000fc02e548eaeee6a/1?pq-origsite=gscholar&cbl=18750&diss=y
- Federici (2021) https://www.researchgate.net/profile/Lorenzo-Federici/publication/353828924_Autonomous_Guidance_for_Cislunar_Orbit_Transfers_via_Reinforcement_Learning/links/612a31bf0360302a00618551/Autonomous-Guidance-for-Cislunar-Orbit-Transfers-via-Reinforcement-Learning.pdf
- Sullivan (2021) https://ieeexplore.ieee.org/abstract/document/9438267
- Bonasera (2022) https://arc.aiaa.org/doi/abs/10.2514/1.G006783
- Dolan (2023) https://proceedings.mlr.press/v211/dolan23a.html
- Lafarge (2023) https://www.sciencedirect.com/science/article/pii/S0094576523002928
- Zhang (2023) https://ieeexplore.ieee.org/abstract/document/10332252
- Qu (2023) https://ieeexplore.ieee.org/abstract/document/10451330
- Holder (2025) https://ojs.aaai.org/index.php/AAAI/article/view/34852
- Kazemi (2024) https://ieeexplore.ieee.org/abstract/document/10611892

and others.

## Paper attributed to ICLR 2025 but in fact withdrawn
In line 132, the manuscript states:


while high-profile works such as the Equivariant Spatio-Temporal Attentive Graph Network (ESTAG) (Wu et al., 2023) and the
Spacetime E(n)-Transformer (SET) (Charles, 2025) advanced equivariant attention mechanisms for physical and geometric systems.


In the References section, (Charles, 2025) indicates an openreview link for ICLR 2025. When following the link, we see the author has withdrawn the paper, with the comment: "I am withdrawing the paper as I do not believe it currently meets the standards of the conference."

The cited paper (Charles, 2025) is high-profile in what sense?

## Results validation

1. Experiments are not run in a publicly available benchmark for orbital dynamics, like Kolosa (2019), Herrera (2020), or others, so the results are not comparable.
1. Other experiments with competing models using RL or optimization for Collision Avoidance are not shown as benchmarks. See above the list of references for relevant papers in the CAM literature. This would provide evidence of the architecture's superiority wrt other published architectures.
1. Comparing against CNN and LSTM is not enough, because the standard for the prediction of quantities related to collision avoidance is the transformer.
1. It is not clear if ablations only remove architectural elements or also features, for example, are the TCA and miss distance included as features when B is ablated?
1. The code is not runable, as it does not use standard or published environments for orbital dynamics, as said before, and the code for the environment is not disclosed.
1. Large scale experiments. Debris comes in general in large clouds of small objects. In Fig 3 iii), we see an increasing number of debris up to 1000, versus training and inference times, but not reward or specific performance metrics.

## Missing ablations

1. Evidence needed for mean pooling (eq. (6))
1. Behavior with a varying number of debris across time, $N_t$
1. Evidence needed for sentence "The self-attention component can
identify complex, non-obvious relationships between threats, while the physics-based bias ensures
that the model remains sensitive to critical, domain-specific risk indicators, improving learning efficiency and final policy performance."
1. Evidence needed for sentence "In practice, we found that the traditional attention mechanism was not effective"



# clarity



## Notation

In figure 1, Satellite state and debris state appears before being defined.
Eq 2. $\Delta v$ is a vector. In the manuscript all vectors are bold except this one. This raises confusion whether this notates speed or velocity. Apparently it is the second. Also in line 190 $u_t$ is a vector and is not in bold.

# significance

## Single agent formulation
The current and future spacefaring is based in constellations. So it is arguable that the single agent setting is not compliant with real deployments. Also, avoidance maneuveres can be motivated by other satellites and not just debris.

## Speed constraint
The speed constraint --- on the norm of the velocity --- entails that the spacecraft can exert omnidirectional thrust. This is not realistic in practice.

### Questions
See Weaknesses

### Soundness
1

### Presentation
2

### Contribution
2
