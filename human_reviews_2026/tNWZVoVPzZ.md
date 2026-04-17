# PAMDP: Interact to Persona Alignment via a Partially Observable Markov Decision Process

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The interaction process of comprehending user-specific nuances and adapting to their preferences represents a pivotal consideration for Persona Large Language Models, as it more authentically mirrors genuine dialogue dynamics than adherence to general human value alignment. In this paper, we conceptualize this ``Interact to Persona Alignment'' challenge as a Partially Observable Markov Decision Process, abbreviated as PAMDP, wherein the user’s dynamically evolving profile through interaction is treated as an unobservable variable to the assistant. Grounded in this formulation, we propose a dual-critic reinforcement learning framework, with a continuous latent space action representing the assistant’s utterance. We evaluate our approach on both offline datasets and the online simulator, ultimately demonstrating its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends the definition of Partially Observed Markov Decision Process (POMDP) to dialogue settings. It then introduces an actor-critic algorithm to learn policy in this setting. The experiments on two datasets show that the proposed method can improve performance compared to simple baselines such as chain of thought and behavior cloning.

### Strengths
1. The presentation of the paper is clear.
2. The proposed method shows performance improvement on two datasets.

### Weaknesses
1. The proposed method relies on the quality of user simulator. If user simulator cannot well simulate real world settings which is often the case then the trained policy may work well with user simulator but may not work well under real settings.
2. It will be better if other aspects of dialogues such as informativeness and helpfulness can be evaluated.
3. The proposed method seems just a simple mapping of notations from POMDP. It seems not very novel or different.
4. More advanced RL-based dialogue improvement framework can be compared as baselines, such as [1].

References:

[1] EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning

### Questions
How different is the new PAMDP, or it is just POMDP with different notations mapped to specific concepts in the dialogue setting?

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
This paper deals with the “Interact to Persona Alignment” problem. The authors conceptualize the issue as a Persona Consistency Markov Decision Process (PAMDP) and build a dual-critic reinforcement learning framework. They also evaluate their approach on both offline datasets and the online datasets with the user simulator, ultimately demonstrating its effectiveness.

### Strengths
1. I find the concept transformation method proposed in this paper highly compelling. LLMs often struggle to adapt to user preferences in real-world applications, making personalization a challenging task. This paper addresses the problem by reformulating the Persona Alignment problem as a Markov Decision Process (PAMDP) and introduces a dual-critic reinforcement learning framework to enable a more streamlined and effective personalization experience.
2. The proposed approach is notably simple (in a positive sense), intuitive, and demonstrates strong practical effectiveness.

### Weaknesses
1. The implementation details, especially in the experimental section, remain insufficient. Specifically, the mapping between the mathematical notations in the formulas and their actual representations in the model is unclear.
2. The use of the ALOE benchmark in offline experiments raises some concerns. The evaluation metrics differ from those in the original ALOE paper, and the reward model employed appears questionable. It is unclear whether it aligns with the original reward function.
3. In online experiments, the evaluation relies heavily on the user simulator. However, the design and reliability of the user simulator are not described in sufficient detail. It remains unclear how the simulator is built and how its quality is ensured.

### Questions
1. While the formalization based on POMDP is elegant, many of the core components lack clear implementation-level descriptions. For instance, what exactly are the hidden state H and the context w in the actual system? How are these represented in practice? Furthermore, why is a double-critic evaluation necessary? 

2. For evaluation, you compare your method against baseline models using win rate, but the original ALOE paper provides its own evaluation metrics. Why not use those as well? Additionally, the proposed method is not directly compared to other personalized methods like BE. This makes it difficult to clearly identify the advantages of your method. A more comprehensive comparison would help clarify its empirical strengths.

3. The proposed method does not seem to construct an explicit user profile on the assistant side for reward computation. Instead, it relies on the quality of the final responses as a reward signal. How can we ensure that the user-related information embedded in the dialogue has a positive effect on model optimization? It is possible that the expected user profile may not be accurate, yet the resulting output is still good—how is this handled in your framework?

4. Regarding the dynamic evaluation in multi-turn dialogues, have you considered the impact of the user simulator’s quality on the results? How is the user simulator initialized based on the user profile? More detailed explanations would help justify the reliability of the online evaluation setting.

5. The citation format should be revised, and the meanings of each symbol should be clearly stated. In addition, the comparative experiments need to be added to the main content instead of the appendix, and the main content should illustrate more about your work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper formulates “Interact-to-Persona Alignment” for LLMs as a partially observable control problem, where the user’s evolving persona is a hidden state variable. It introduces the Persona Alignment MDP (PAMDP) and derives a Bellman equation specialized to this setting, a dual-critic advantage estimator, and shows the estimator is unbiased relative to the standard POMDP advantage. The actor produces a continuous latent action that steers the base LLM’s generation; critics estimate values with and without access to the hidden persona during training. Experiments span offline datasets (ALOE, PrefEval) and an LLM-based online simulator, reporting improvements over prompt/PEFT/FPFT/CoT and asymmetric-A2C baselines.

### Strengths
Treating persona alignment as a partially observable control problem pins down what is observed (dialogue history) vs. hidden (persona) and yields a proper Bellman structure. This makes the problem analyzable and comparable to established POMDP methods, rather than staying at the level of ad-hoc prompting.

Using one critic with history only and another with access to persona during training gives an estimator that the authors argue is unbiased relative to the true POMDP advantage, unlike standard asymmetric A2C. The actor outputs a low-dimensional latent that steers the base LLM, with KL regularization to a behavior prior (from BC).

The paper’s Theorem 3 contrasts its estimator against asymmetric A2C (UAAC/DCRL-style) and argues lower bias. Methodologically it also situates the online study alongside PPDPP’s prompt-programmed dialog planning. Relative to recent personalization datasets (ALOE, PrefEval), the focus here is on treating persona as an unobserved variable updated through interaction.

### Weaknesses
The results rely on LLM-as-judge and LLM-generated rewards. Both the reward used during RL and the offline/online evaluation are produced by the same family model (Qwen2.5-72B-Instruct). The same family of models is involved in generating rewards and judging outputs. No human evaluation is reported.

Cumulative returns being negative makes it unclear how “good” a policy is on an absolute scale. In the online simulator (max 6 turns), all methods—including yours—show negative cumulative returns throughout; your method is best but still ends at about −2.5. Without a calibrated scale or a normalized metric, it is difficult to infer practical significance from “less negative” numbers. Please normalize rewards (e.g., to [0,1] or standardized z-scores), report success@k and step-wise human ratings, and include qualitative trace examples that map scores to perceived usefulness. Also justify the fixed 6-turn horizon and show sensitivity to longer dialogs.  

The online environment stitches together three LLM-prompted modules—Profile-Infer, User Simulator (which selectively reveals parts of the profile each turn), and a Reward Generator that mirrors the offline reward design. While useful for ablations, this pipeline may not reflect real users’ inconsistent, messy disclosures, and it inherits the same judge/reward biases. Please release the simulator prompts/code for scrutiny, stress-test with randomized persona noise and non-staged disclosures, and, if possible, run a small study with consented human participants to verify transfer.  

Theorem 3 shows that the dual-critic advantage estimator is unbiased in expectation given access to the true hidden persona ω; in practice, the full-information critic uses ω while the actor never sees ω, and the system relies on LLM-inferred profiles or dataset metadata that can be noisy or partially wrong. That mismatch (and the function-approximation error in both critics) could re-introduce bias/variance. Please quantify sensitivity to persona noise and critic misspecification.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PAMDP, a novel framework that models the task of aligning a large language model to an individual user's persona as a Partially Observable Markov Decision Process (POMDP). The authors argue that standard alignment techniques optimize for general preferences, failing to capture user-specific nuances that evolve during a conversation. In their PAMDP formulation, the user's dynamic profile is an unobservable variable, and the dialogue history is the observable state. They propose a dual-critic reinforcement learning algorithm to solve this, using an actor that only sees the observable history and two critics that access both observable and unobservable information during training to provide an unbiased advantage estimate. This approach, which uses a continuous latent action space, is validated on both offline datasets and an online simulator, demonstrating

### Strengths
- Originality: The paper's identify the "Interact to Persona Alignment" problem. It is the first work to formally model this problem as a Partially Observable Markov Decision Process (POMDP), which the authors term the Persona Alignment MDP (PAMDP).
- The PAMDP framework is well-justified. The authors formally derive the Bellman equation for their PAMDP (Theorem 1) and propose a novel dual-critic advantage estimator (Theorem 2). Crucially, they provide a mathematical proof (Theorem 3) that their estimator is unbiased, giving the method a strong theoretical foundation.
- The method is validated on both offline datasets (ALOE and PrefEval) using two different base LLMs (Qwen2.5-7B and Llama3-8B) and in an online setting with an LLM-based user simulator.

### Weaknesses
- The initial policy model (being initialized by behavior cloning. It implies the RL method may be only able to refine a policy that is already persona-aware and may not be able to learn a personalized policy from scratch. A experiment from scratch, or non-expert BC initialization, should be provided. 
- The proposed method introduces a dual-critic architecture. This is more computationally expensive than the single-critic baselines or standard fine-tuning methods. The paper does not provide any analysis of this added overhead. 
- The reward is only a scalar score without fine-grained feedback. The authors could explore using a more fine-grained rewards on multiple criteria, such as style, relevance, persona-consistency.

### Questions
- In Sec 4.1, the offline learning  trains on pre-collected datasets. Offline RL is hard due to the distributional shift. The paper uses BC initialization. How do your method mitigate the distributio shift? Will using a BC initaliation be circumventing this problem?
- The reward as well as the evaluation is using the same LLM. Wouldn't that be "circular" evaluation?
- I am not sure what "dynamically evolving profile" means. In Table 8&9, it seems to be only progressively revealed, it is not exactly what "evolving" means I think. I think the dynamically evolving profile should indicates that the user's preference is changed in the interaction, i.e., $p(\omega_{t+1} | \omega_t, h_t, u_t)$.
- Standard TD should be $\hat{A} \triangleq \delta(h,\omega,u)=r(h,\omega,u)+\gamma V(h^{\prime}, \omega)-V(h,\omega)$. Why do you use $V(h^{\prime})$ instead of $V(h^{\prime}, \omega)$.

### Soundness
3

### Presentation
3

### Contribution
3
