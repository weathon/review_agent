# SYNAPSE: Simulation Benchmark of Neuro-Adaptive Patient-Specific Evaluation for Episodic Decision-Making

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Recent advances in time-series analysis, treatment outcome prediction, and reinforcement learning (RL) have demonstrated great potential to automate decision-making in healthcare. However, the high stakes nature complicates the deployment of such frameworks in practice, clinically, or in the long term. A major challenge is the absence of realistic benchmark environments that capture the sequential, patient-specific nature of various therapies, which could enable extensive offline testing, evaluation, and model selection prior to clinical adoption. To address this, we introduce the SImulation Benchmark of Neuro-Adaptive Patient-Specific Evaluation (SYNAPSE), in the context of adaptive deep brain stimulation (DBS), a treatment for managing the motor symptoms of Parkinson’s disease (PD). Specifically, SYNAPSE is constructed using real-world data collected from both clinical and at-home studies involving participants undergoing DBS therapy. It enables offline training and evaluation of different treatment strategies, reflecting both short- and long-term effects, as well as treatment outcome prediction capturing participants’ responses to a range of temporal dynamics. Additionally, it allows for the assessment of safety-critical constraints inherent to neurostimulation decision-making. By rigorously validating its realism against clinical data and supporting both short- and long-term decision-making, SYNAPSE offers clear guidance for future DBS policy development, as well as helps identify and address key challenges in advancing truly personalized neurostimulation therapies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper develops a simulation for aDBS for Parkinsons. The approach uses clinical data to train models of human patients. It enables future aDBS policy to be tested in silico before deployment on real patients.

### Strengths
aDBS is a very promising technology to improve the lives of patients. However, there do not currently exist many ways to effectively test new methods for aDBS control before deployment in the patient. This paper addresses this very important problem by using patient derived data  to create an aDBS simulation.

### Weaknesses
Immediate rewards and ihr could be better explained when initially introduced.

More details about the actual modeling need to be included since this is the point of the paper. This shouldnt just be in the appendix. A diagram of the simulation architecture would also be helpful

A discussion of how one would choose which environment to test a new policy for a given patient would be helpful. is there a way to quantify how close a new patient is to each simulation environment to know which would be best to use?

More detail about the heterogeneity of responses would be useful and differences in each environment would be useful

Related works section talking about other data-driven human simulations for medical applications would be useful to better contextualize this work

This is an unusual topic for ICLR. I would recommend briefly discussing "Coprocessor Actor Critic: A Model-Based Reinforcement Learning Approach For Adaptive Brain Stimulation"  in related works as this is the most similar work that has also been published in this community

### Questions
New control strategies be out of distribution of the policies that the simulation was trained on. Is there a way to quantify uncertainty in the simulation?
What does turning off dbs mean? Is this just no stimulation?
Can medication be modeled in the simulation?

### Soundness
2

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
This paper develops a SImulation Benchmark of Neuro-Adaptive PatientSpecific Evaluation (SYNAPSE) for adaptive deep brain stimulation (DBS) in Parkinson’s disease. The paper includes offline training and evaluation of different treatment strategies, reflecting both short- and long-term effects.

### Strengths
1.	The developed simulator is critical to understand the impact of reinforcement learning on DBS and bridge the gap between evaluation and real-world clinical practice. 
2.	The paper includes a diverse set of modern RL algorithms across multiple reward definitions and time horizons.

### Weaknesses
1.	The contribution of this paper needs clarification. The experiment and the developed dataset have significant overlap with the existing work [1]. The paper should explicitly articulate what the new components are provided in the newly proposed benchmark.
2.	The dataset includes only five patients, which limits the generality and makes the term “benchmark” somewhat premature. While the authors acknowledge this limitation, a stronger justification is needed to argue why this sample is sufficient to model variability across the PD population.
3.	While standard RL metrics (e.g., episodic return) are used, there is a lack of discussion on how these results map to clinically meaningful outcomes.
4.	Learning Agent is not defined in Figure 2. Does it refer to an RL agent?

[1] Gao, Qitong, Stephen L. Schmidt, Afsana Chowdhury, Guangyu Feng, Jennifer J. Peters, Katherine Genty, Warren M. Grill, Dennis A. Turner, and Miroslav Pajic. "Offline learning of closed-loop deep brain stimulation controllers for parkinson disease treatment." In Proceedings of the ACM/IEEE 14th International Conference on Cyber-Physical Systems (with CPS-IoT Week 2023), pp. 44-55. 2023.

### Questions
1.	What are the key differences between SYNAPSE and the previous dataset/work [1]?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SYNAPSE, a domain-specific simulator and benchmark for adaptive deep brain stimulation (aDBS) in Parkinson’s disease (PD). The key claim is that SYNAPSE provides patient-specific environments learned from real clinical and at-home data, supporting offline/online RL, off-policy evaluation (OPE), human-feedback modeling, and policy transfer across a small “virtual cohort.” The authors report multiple fidelity metrics (trajectory/statistics alignment) and illustrative RL/OPE experiments across a small but longitudinal dataset.

### Strengths
This is an important problem. Safe RL for closed-loop neurostimulation is impactful and arguably underexplored compared to EHR-style decision making.

I like that the authors model each participant as an environment... this seems like a potentially useful abstraction for transfer and personalization.

The authors include off policy evaluation (OPE), long-horizon outcomes, and a human-feedback component.

The benchmark is built from longitudinal aDBS deployments rather than purely synthetic toy dynamics.

### Weaknesses
The presented results are primarily within-patient... the paper probably needs leave-one-patient-out (LOPO) analyses to show simulator fidelity and RL/OPE behavior on a held-out patient.

Current metrics (e.g. latent space or marginal distribution closeness) are only loosely tied to policy improvement. I would think the paper should evaluate policy ranking agreement, value calibration, and/or regret vs. real logs or prospectively held-out traces.

aDBS has clear safety constraints (energy budgets, ramping/overshoot limits)... I did not see reports of constraint-violation rates or adverse-proxy statistics under learned policies.

Offline RL and OPE comparisons feel thin. In particular, a comparison to prior benchmarks like EpiCare (Hargrave, Spaeth, Grosenick NeurIPS 2024) a recent, broad healthcare RL benchmark with a stronger OPE/offline RL evaluation suite (that seems adaptable here / is synthetic) would situate SYNAPSE more clearly (even if only to show why device-level control needs different methods/metrics). And such prior work on POMDPs should be cited here?

Minor but probably should be corrected: 
- It seems that the author's backronyms spells "SINAPSE", but the authors use "SYNAPSE" or sometimes "SYNAPSES". 
- There's some real potential redundancy between Fig 1 and Fig 2

### Questions
What happens under LOPO? Authors should try training simulators on N−1 patients and report transition/reward fidelity, OPE calibration, and policy performance/regret on the held-out patient.

How do common OPE methods behave under coverage shift?

Which hard constraints are enforced in simulation, and what are the violation rates during training/rollout?

Provide calibration plots and sensitivity to priors/likelihoods; does HF aid or mislead policy selection?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents an environment (SYNAPSE) for accelerating RL research and benchmarking adaptive deep brain stimulation policies for Parkinson's disease.

### Strengths
The paper targets a key Reinforcement Learning problem in medical settings, and solving it would have substantial impact.

### Weaknesses
1. The main weakness is training the transition dynamics on very limited data from a single individual for each environment while claiming them realistic. The simulator can only be accurate for states it has actually observed, leaving many counterfactuals where it will likely produce unreliable outputs. Only a model with foundation-model-level scale and data coverage could approach the fidelity needed for realistic simulation.

2. Insufficient evaluation - Using t-SNE, predicted-action MAE, and reward EMD/AE to assess the transition dynamics is not a meaningful or rational evaluation strategy.

3. Referring to the setup as a Human-involved MDP (HMDP) is redundant, as both $R$ and $R^H$ can simply be components of a single vector-valued reward function within a standard MDP.

4. It is unclear which policies the authors actually trained. They state that all RL policies were trained with DDPG, yet later they compare PPO, A2C, and CQL agents as well.

5. The paper appears confused about the differences between online, offline, and off-policy methods. For example, 
a) in the Clinical Sessions section, the authors write: 'The RL controllers are trained using deep deterministic policy gradient (DDPG) with data collected from three other types of controllers, followed by finetuning with the latest data.' But DDPG is an online algorithm, meaning it learns through interaction with the environment, not from static datasets collected by other controllers. This makes the described training setup inconsistent with how DDPG is intended to operate.
b) The authors claim that the simulator enables off-policy evaluation, but a simulator is not required for that. A simulator enables online training, not off-policy evaluation.

6. In the 'Challenges for ML/RL in DBS' section, the authors claim that existing simulators lack patient-specific nuances seen in vivo. But SYNAPSE’s transition dynamics are trained on data from only five patients, which leads to the same limitation.

### Questions
1. The authors note that existing DBS controllers have been validated through clinical testing. While we understand that those datasets are not public, such data is often available under Data Usage Agreements. Could the authors explore accessing any of these datasets to compare SYNAPSE’s predictions? This would significantly strengthen confidence in the simulator’s validity.
2. In the 'At-Home Sessions' section, the authors state: 'When the participant chose to start a session, one of the three controllers was uniformly randomly chosen to start until the participant chose to end the session.' How safe is random policy selection in this context, and why was this strategy chosen?

### Soundness
1

### Presentation
1

### Contribution
1
