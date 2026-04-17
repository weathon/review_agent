# Guided Policy Optimization under Partial Observability

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Reinforcement Learning (RL) in partially observable environments poses significant challenges due to the complexity of learning under uncertainty. While additional information, such as that available in simulations, can enhance training, effectively leveraging it remains an open problem. To address this, we introduce Guided Policy Optimization (GPO), a framework that co-trains a guider and a learner. The guider takes advantage of privileged information while ensuring alignment with the learner's policy that is primarily trained via imitation learning. We theoretically demonstrate that this learning scheme achieves optimality comparable to direct RL, thereby overcoming key limitations inherent in existing approaches. Empirical evaluations show strong performance of GPO across various tasks, including continuous control with partial observability and noise, and memory-based challenges, significantly outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Guided Policy Optimization (GPO) for POMDPs with privileged information available only during training. GPO co-trains a guider and a learner, coupled by two mechanisms: (i) backtracking/alignment that constrains the guider to remain within a “possibly good” region trackable by the learner, and (ii) optionally adding RL updates to the learner to compensate when imitation lags. Empirically, the paper reports consistent gains over PPO, PPO+BC, A2D/ADVISOR-style teacher–student methods, and other variants across didactic settings, Brax continuous control with partial observability and added noise, and POPGym memory tasks, with ablations for thresholds/clipping and alignment strategies.

### Strengths
1.The paper excels at motivating the work. The "impossibly good" teacher is a memorable concept, and the TigerDoor example  perfectly illustrates the concrete failure mode of naive TSL.

2.Proposition 1  provides a strong theoretical foundation for the framework, giving confidence that the learner's imitation objective is not just a heuristic but is equivalent to a constrained RL update.

### Weaknesses
1.The strongest results rely on GPO-penalty/clip, where the learner uses a hybrid RL+IL objective. Current evidence does not isolate how much of the gain derives from constraining the guider versus adding RL to the learner. Controlled experiments (fix guider, swap learner objective; and vice versa) are needed to separate effects.

2.While the mechanism is novel, the high-level concept of co-training a privileged teacher and a partially observable student is not. The paper does a good job arguing against prior work like A2D in the appendix , but the conceptual leap is more of a significant and highly effective refinement than a completely new paradigm.

3.The failure mode analysis in Section 4.4 is honest but reveals a key weakness. GPO's performance is sensitive to the KL threshold or clip parameter. In difficult memory tasks (e.g., CountRecallHard), a poorly tuned threshold causes GPO to underperform PPO. The authors explain this is because the learner (GRU) fundamentally cannot store the information, making the guider truly inimitable. This is a valid explanation, but it implies GPO does not fully solve the imitation gap so much as manage it via a new, sensitive hyperparameter that may require careful task-specific tuning.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Guided Policy Optimization (GPO) algorithm that jointly trains a teacher and a student policy to mitigate the traditional imitation gap in reinforcement learning (RL). The approach draws an analogy to active learning. Experiments conducted on both didactic and locomotion benchmark tasks demonstrate empirical improvements over standard on-policy RL baselines, particularly in environments with partial or noisy observations for the student.

### Strengths
\+ The idea of co-training a teacher and student policy is intriguing. 

\+ The proposed method is technically solid. 

\+ Experiments seem comprehensive on SOTA locomotion tasks.

### Weaknesses
- The training process involves hyperparameters and seems intricate to tune. 

- Frequent heuristic descriptions: Terms such as *“possibly good region,” “impossibly good,”* and *“inimitable”* require clearer definitions. I would suggest that authors provide a more formal characterization of these concepts.

- A few claims may not be factual: 

  * Authors used TSL to refer to IL and policy distillation "as there is no fundamental distinction between them." (Line 82). However, IL usually refers to the case where no environmental rewards are available during learning. 

  * Crucially, none of the existing methods provides a theoretical guarantee that teacher supervision will actually be beneficial. (Line 119 - 120)

- Fasibility of the core assumptions: (see Q2) The assumption that a learnable teacher model with privileged information can co-train effectively with the student (see Q2) may be restrictive in practice.

- The method is evaluated only with PPO as the underlying RL algorithm, primarily due to its trust-region design (see Q3).

### Questions
**Q1.** How robust is the proposed algorithm to different configurations of the predefined thresholds used during training?

**Q2.** In most IL or TSL settings, the teacher is a fixed pretrained model (e.g., an expert policy or human demonstrator). What realistic scenarios would allow both teacher and student to be learnable models, with the teacher having privileged information?

**Q3**: How applicable would this method be given different backbone RL algorithms that are on-policy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper discusses the learning of RL agents in POMDPs environment with access to a teacher's partial policy that might have used privileged information (modeles as full state observability). The method consists in iterations of teacher learning and guided student policy learning, where the learned student policy is used as guidance for the next step. Experimental results with a toy problem as well as with simulated and challenging environments show the superior performance of the proposed method (often achieving higher cumulative rewards and faster convergence) compared to SoTA approaches.

### Strengths
- Well-written, clear writing
- Relevant problem
- Good empirical support


I can't vouch for its originality though as I'm not well familiar with the literature

### Weaknesses
- Despite Theorem 1, the proposed method is mostly heuristic, and good empirical performance might not replicate to other

### Questions
How does the proposed model compare to the alternatives in terms of computational cost? Having two separate and intertwined learning seems to add significant computational overhead.

The high-level description of the algorithm suggest that the learned policy depends only on the current observation, which would be very suboptimal for a POMDP. Would learning from a sequence of observations and actions work just the same? What policy class was used in the experiments (in terms of input)?

### Soundness
4

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
4

### Summary
This paper introduces Guided Policy Optimization (GPO),  a novel framework for Reinforcement Learning in partially observable environments. GPO co-trains a guider (using privileged information) and a learner (using partial observations and imitation learning) while employing an alignment step that constrains the guider to remain imitable by the learner. This mechanism ensures the guider acts as a "possibly good" teacher, significantly outperforming existing methods on continuous control and memory tasks.

### Strengths
- The idea of the guider's backtracking step is novel and interesting. It interprets the principled solution to the "impossibly good" teacher problem. By actively keeping the privileged guider's policy within the learner's reachable policy region, GPO ensures that the supervision remains beneficial. 
- The implementation of the GPO method is quite clever, with additional state and observation and incorporated into PPO policy improvement. 
- The method is rigorously tested across a diverse set of tasks, including challenging continuous control (Brax) with partial observability and noise, and various memory-intensive tasks (POPGym), showing consistent and significant improvements over strong baselines.

### Weaknesses
- The main concern of the paper is the mismatch between the theory part and the practical implementation. For example, in Section 3.1, the GPO iteration is performed in four steps in sequence, but the actual loss of the 4 steps is combined in one policy improvement step in PPO, which might result in an optimisation issue. Besides, the use of L4 loss is not quite obvious, since the GPO iteration should work even if the learner doesn't have a value function. 
- There are some unclear design choices. For example, the paper focuses on the partially observed setup. In Section 2, the learner's policy depends on the trajectory, while in the following sections, the input is only a one-step observation o. In theory, it doesn't include enough information to solve the POMDP. Besides, the use of the memory-based experiment seems a little unmotivated. The idea should be using memory-based models like RNN to solve PODMP instead of on memory-based tasks, if I understand it correctly? It would be helpful to provide more explanations on these choices. 
- Some additional visualisations might help the understanding of the GPO mechanism. For example, the tunable parameters $\alpha$ during training, and the performance gap between the guider and learner during training.

### Questions
1. In Equations 11 and 12, are the advantage functions of the guider and learner the same since they both use the same trajectories and value functions? 
2. In Section 4.3, the input of the learner is the whole memory, while the input of the guider only keeps the important information, and is it correct? 
3. The paper aims to learn a "possibly good" teacher. Can the same be achieved by carefully tuning the update rate of the PPO+BC method? For example, restricting the teacher update size and increasing update steps for the student. 
4. Line 276, what does the exact backtracking mean?
5. For Equation L1-L4, L1 and L3 are maximisation, and L2 and L4 are minimisation? It would be clearer if they represent the same.

### Soundness
2

### Presentation
2

### Contribution
3
