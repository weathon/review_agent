# Recurrent model for Sequential reasoning

- Decision: Reject
- Scores: 0, 2, 4, 6, 2, 2, 2, 8

## Abstract
We propose a recurrent architecture designed to extend test-time scaling capabilities to sequential input streams. By interleaving fast, iterative reasoning loops between slow observation updates, our method facilitates dynamic compression of latent representations, where internal states self-organize into stable clusters that persist and evolve alongside the input. This mechanism allows the model to maintain coherent representations over long horizons, significantly improving out-of-distribution generalization in reinforcement learning and algorithmic tasks compared to standard sequential baselines such as LSTM, state space models, and Transformer variants.

Code: https://anonymous.4open.science/r/fastslow-81DB

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper aims to address the challenge of real-time control and planning, where inputs arrive continuously and decisions must be made asynchronously while planning. The authors consider a two time scales process where planning (reasoning) steps are performed faster than input arrival rate. 
They propose a recurrent architecture and an auxiliary memory module.
In their experiments, they compare the proposed architecture against several sequence model architectures on two benchmarks: MiniGrid, and a maze environment.

### Strengths
- The problem of effective "real-time" control (and planning) is interesting and important for the community (and has been studied extensively in the literature).
- The proposed method is original.

### Weaknesses
`W1`: Throughout the paper there are many inaccuracies and lack of precision, suggesting a failure to identify precisely what is the problem to be addressed, what are the relevant baselines and methods in current literature for that particular problem and setting, and what is the appropriate experimental setup for proper evaluation (for showing progress). 
As an example (line 13):
> However, these models are not designed to accept sequential and dynamical input, while many real-world applications as well as the survival of organisms in nature require real-time input to be processed in parallel with the reasoning.

The referred approaches are by definition sequence models, and are designed to accept sequential input, e.g., token sequences. 
If I understand correctly, the authors' intention was that these methods are not design for a rather practical setting where there is an active input stream, with inputs arriving at a given rate, while the model is required to provide outputs asynchronously, as a real-time controller, while planning (reasoning).

Importantly, this problem and setting is very different from that of chain-of-thought (CoT), and in fact I would not consider CoT as part of the most relevant literature.
Moreover, the term "reasoning" is typically used in current literature for language-based planning. Here, there is no use of language modality, and thus "reasoning" is not the appropriate term here, perhaps "Planning" is more appropriate.

I refer the authors to the control and reinforcement learning literature, where such settings were extensively studied.
For example, in autonomous driving [1], algorithms perform concurrent planning and control in real-world settings successfully.

[1] Paden, B., Čáp, M., Yong, S. Z., Yershov, D., & Frazzoli, E. (2016). A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on intelligent vehicles, 1(1), 33-55.




`W2`: The paper contains many unjustified claims and statements.
Example:
> Indeed, a realtime agent, such as a living organism in nature, is required to solve the reasoning task with this deadline constraint.

This is not a trivial claim, please provide supporting evidence (e.g., refer to prior works that validate this claim).





`W3`: It is unclear that effective real-time control can not be achieved by existing methods under various reasonable choices, i.e., that there is a real problem with existing methods.
The authors are responsible for establishing that there is a problem with existing methods in this setting for an important subset of tasks, under certain hardware choices, etc.
Currently, the paper does not convincingly support this claim.

For example, many existing robotic applications operate in real-world setting, in real time, and perform very well [1][2], without the method proposed in this paper. Where is the problem?

[1] Tang, C., Abbatematteo, B., Hu, J., Chandra, R., Martín-Martín, R., & Stone, P. (2025, April). Deep reinforcement learning for robotics: A survey of real-world successes. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 27, pp. 28694-28698).

[2] Paden, B., Čáp, M., Yong, S. Z., Yershov, D., & Frazzoli, E. (2016). A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on intelligent vehicles, 1(1), 33-55.





`W4`:  Poor experimental design. 
The experimental setup compares *sequence model architectures*. It is thus unclear how the choice of sequence model architecture is relevant for addressing the problem at hand, i.e., real-time asynchronous control. 
Real-time response time crucially depends on model size, hardware, architecture choices, and on the difficulty and properties of the specific problem at hand. 
The experiments should compare existing methods for real-time control, rather than sequence model architectures. Model architecture, size, hardware, and other factors should be controlled for any real-time control baseline.

In the related works section the authors referred to existing methods. Why are these methods not included in the comparison?

In addition, it is unclear how real-time control was implemented in the experiments, if at all. What is the observation (input) arrival rate? How each baseline processes and computes outputs for such real-time setting? How is the real-time aspect implemented in the experiments? Why and how this design reflects real-world real-time control settings?

### Questions
`Q1`: Why is the latent (hidden) dimensionality of the baselines different from that of the proposed method? How this affects the results?

`Q2`: How did you choose the hyperparameters in your experiments? Why?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel recurrent reasoning architecture which works on two timescales; a fast recurrent inner loop and a slower observation timescale to handle real-time streaming inputs. In essence, their method performs many recurrent hidden state updates between each new observation which is received. The authors base their framework off of a Kuramoto oscillatory-based mode (AKOrN). The authors evaluate their approach on synthetic maze path prediction tasks as well as several minigrid RL tasks. Their results show good OOD generalization for these tasks compared to other recurrent models such as LSTM, Mamba-2 and Transformer-XL

I think this paper brings forth interesting ideas, but it is not yet up to the standards of an ICLR main conference track submission. The writing is poor, the experimental results, although interesting, lack depth both in the baselines and benchmarks considered. Although the main ideas behind this paper are interesting, I think it still needs some work. See below for suggestions on how to address these weaknesses.

### Strengths
- The architecture proposed by the authors is novel and interesting. It is grounded in observations from the neuroscience/biology literature as well as observations from the LLM reasoning/test time inference literature. 
- I think the idea of scaling "fast inner loop" iterations to improve reasoning is interesting and could lead to improved model design.
- The method proposed in the paper has strong OOD generalization on the considered tasks.

### Weaknesses
**Writing Style of the paper is poor and needs polishing**

- I find the writing style of the paper to be poor; it feels like this paper was written in a rush. The paper has many typos and poorly constructed sentences. See Questions section for a couple examples. I did not write up an exhaustive list.
- I also feel the narrative of the paper lacks structure. What is the problem you are trying to solve? How are you solving it? 
- The paper discusses test time scaling, large reasoning models (LRMs)and continuous CoT a lot as motivations. The authors should make clear that their work is *inspired* by claims from this literature but not a direct extension of it. 

**Experiments lack breadth both on the benchmarks/tasks considered and on the model baselines considered**

- 3 seeds is very few for RL. I find that the results lack statistical significance because of this
- I find the experimental section to lack comparable benchmarks. The authors only evaluate on two, mainly synthetic tasks, and do not evaluate on any known tasks from the recurrent model literature such as tasks from the LongRangeArena.
- The baselines (LSTM, Mamba-2, Transformer-XL) are relevant, but the parameter counts shown in Table 4 indicate major capacity mismatches (ours reported at 1.16M / 1.19M parameters vs baselines up to ≈27M). This raises a concern that improvements might partially come from better inductive bias rather than a fair capacity/compute comparison. A fairer comparison requires (a) matching parameter budgets or (b) showing that the gains persist when baselines are scaled to similar parameter counts and compute budgets.
- Important related system baselines that deal with streaming or latent recurrent state (Perceiver variants, Looped Transformer, Universal Transformer with recurrence, stronger RL baselines) should be included or directly discussed in more depth. It could also be interesting to test S4/S5 state space models as they are known to perform better than Mambas on tasks similar to the "Maze" task considered (I am thinking about PathX and related tasks)
- It seems from the ablations done in Fig. 4 that auxiliary memory has a significant impact on the performance of the author's method on the Maze task. The authors should also evaluate other architectures with augmented memory to control for the effect of their architecture

I want to stress that my point is not that the authors should compare against *all* baselines I am proposing here and that they should have experiments on a large set of benchmarks. I am aware this is a methods paper proposing a novel architecture, and thus scaling of the method/adaptation of the method to many different regimes/tasks is not the main focus here. My point is mainly that I do not find the tasks, nor the baseline models to be carefully chosen to properly illustrate the point the authors make. I believe that strengthening the experimental design of the paper could lead to a much stronger submission.

### Questions
**Typos**
- Line 107 "we propose to approach realtime sequential reasoning task" not a grammatically correct sentence
- Line 162 "We choose the AKOrN model for the recurrent update because for several reasons."
- There are many more instances of grammatically incorrect sentences, but I will not list all of them.

**Experiments**
- Why did the authors use training curves (with steps) in Figure 3 to support their claim of improved model performance? The claim you are making does not hinge in any way on the training dynamics and you do not discuss this in the surrounding text.
- Why did the authors use the same epoch number, learning rate and batch size across all models? Was the hyperparameter tuning done independently for each considered architecture? Are the authors confident that each baseline model was trained using the best hyperparameter configuration?

**Other**
- A key contributions paragraph/bullet point list could be beneficial to readers and help them extract the claims being made by the paper
- A few references to very recent/arXiv works are used as anchors for claims (e.g., Miyato et al., 2025 AKOrN). Ensure that the relationship between those works and your contribution is explicitly clarified: what is new here vs what AKOrN already provided?
- In Section 3, make explicit how observation encoding delays (encoder cost) factor into the real-time deadline (Sec.2). Clarify whether encoders run within the slow tick or concurrently.
- Expand captions of Figures 3–7 with clearer description of plotted quantities (exact metrics, units).
- Increase the number of seeds shown in the Appendix for reproducibility.
- In Table 1 what do the checkmarks and crosses mean quantitatively. This is explained nowhere

### Soundness
1

### Presentation
1

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
This paper presents a method that extends the test-time scaling capability of a recurrent model to consider real-time sequential inputs. The proposed method decomposes the system into fast and slow processes. The slow process takes in the observation, and the fast process reasons while waiting for the observations. This paper adopts a recent Kuramoto oscillatory-based model, AKOrN, to update the states in the fast process. It also introduces auxiliary internal and external memory to retain information in fast and slow time scale. The experiment on the egocentric maze and MiniGrid tasks showed that the proposed method improves the performance and the accuracy in MiniGrid also scales as reasoning iteration increases.

### Strengths
- This method is inspired by biology and is capable of performing real-time sequential reasoning.
- The proposed method can potentially apply to different latent recurrent reasoning models.
- The experiment shows that the proposed method can improve tasks that require sequential inputs.

### Weaknesses
- The main architectural change is the fast and slow recurrent process. How is this idea different from having hierarchical LSTMs where the LSTMs at different levels operate at different time scales?
- The proposed method is mainly based on AKOrN. It is unclear how much of the improvement is from the choice of the update rule. Will other updating rules show the same effect?
- The experiment results in some RL tasks (e.g., MultiRoom) have similar performance to the baseline models. It is hard to conclude the proposed method has better performance.

### Questions
- Why do the internal memory only apply to synthetic tasks and the external memory only apply to RL tasks?
- How many reasoning iterations do both experiments use? How much delay is there when increasing the reasoning iteration? What is a realistic number of reasoning iterations to achieve real-time interactions?

### Soundness
2

### Presentation
3

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
The paper introduces a recurrent reasoning model designed to handle streams of incoming information in $\textbf{real time}$ while operating under a time limit. The approach combines a fast inner-loop reasoning process with a slower sequence of external observations, allowing the system to update its understanding continuously. 

The method builds on the AKOrN framework—a Kuramoto-based recurrent updater—by adapting it to work across two timescales and improving its stability with additional memory components, either internal (such as GRU or LSTM units) or external. 

Experiments on a synthetic maze navigation task and partially observable MiniGrid reinforcement learning environments show that the model performs well on familiar data and generalizes better to new situations. Increasing the number of inner reasoning steps at test time further improves performance, demonstrating both flexibility and robustness compared to standard recurrent and transformer-based baselines.

### Strengths
1. The paper clearly formulates the challenge of real-time sequential reasoning under time constraints, introducing a structured fast/slow update framework that links internal reasoning cycles with observation timescales (Eq. 1; Fig. 1, p. 3).

2. The model extends AKOrN’s hyperspherical latent update mechanism (Eqs. 2–3) to handle streaming inputs and introduces auxiliary memory modules—internal (Eq. 4) and external (Eq. 5)—to maintain stability and temporal continuity (pp. 3–4).

3. The authors provide detailed hyperparameter settings (Tables 2–4, pp. 12–13) and an anonymous code repository, supporting transparency and replicability of results.

### Weaknesses
1. Although the deadline constraint is well‑stated (p. 2), there is no wall‑clock latency analysis or throughput/compute‑budget evaluation per observation step. The MiniGrid simulator allows pausing between frames; it is unclear whether the method still outperforms baselines under fixed latency budgets or when T (inner iterations) is capped to meet r eal‑time constraints.The paper’s central “real‑time” positioning therefore remains conceptual rather than empirically demonstrated.

2. The evidence is limited to a synthetic ego‑centric Maze classification setup with a fixed output length S=60 (p. 5) and grid‑world RL (MiniGrid). Absent are higher‑dimensional perception or control (e.g., image‑rich or continuous control), where the proposed fast/slow coupling and memory continuity might be stress‑tested more severely. As a result, claims of general dynamic real‑world decision‑making are only partially supported by the current benchmarks.

3. There is little qualitative analysis of failure modes (e.g., specific MiniGrid layouts where the agent fails, types of partial observability that break the approach), nor disaggregation by sub‑skills (memory vs. planning vs. exploration). This limits insight into why the model helps and when it fails. (Figures 6–7 summarize means ± std, but not error taxonomy; pp. 8–9).

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Authors propose to adopt so-called reasoning module to the realtime sequential reasoning task, in which the agent is required to engage in reasoning and inference at the same time as in-taking the realtime observation sequence. Toward this goal, the authors incorporate the recurrent module into the coupled system of fast and slow processes, in which the module reasons at the fast time scale by sequentially integrating into its state the sensory-observation that occurs at the slow time scale.

### Strengths
- Authors show their approach in reinforcement learning tasks and synthetic tasks, and show that it performs competitively against other well-known models for sequential processing, such as LSTM, Mamba, and TransformerXL.

- Authors demonstrate that their strategy enables effective test-time scaling of the recurrent module to the time scale of observation, so that an agent makes better decisions by observing more at the test time; That is, the longer the agent is immersed in the given environment, the deeper the reasoning and, hence, the decision making.

- With this test-time scaling, the proposed approach leads to improved decision making on more complex tasks, surpassing LSTM, Mamba, and TransformerXL on Maze and MiniGrid tasks.

### Weaknesses
1) The authors use the term "reasoning," but in the result tables and visualizations of the solved tasks, it appears that they simply generate a sequence of output actions.

2) The authors examined the effectiveness of their approach only on relatively simple maze-solving and MiniGrid tasks; they did not conduct experiments with more complex robotic simulations. The practical usefulness of the developed approach is questionable.

3) Figure 1 is not informative enough. What do the captions in the figure mean? How do the reasoning module and memory module interact?

4) The comparison with basic action generation methods is incomplete; for example, the recent RATE method [1] or the classic Decision Transformer [2] are not mentioned.

[1] Cherepanov, E., Staroverov, A., Yudin, D., Kovalev, A. K., & Panov, A. I. (2023). Recurrent action transformer with memory. arXiv preprint arXiv:2306.09459.

[2] Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34, 15084-15097.

5) The article does not provide information on the inference performance of the developed model, which is critical for fast simulation.

6) The article's formatting should be improved; punctuation marks are missing at the end of the formulas.

### Questions
1) In the section on reinforcement learning, the authors write that their model is trained end-to-end using Proximal Policy Optimization. Why was this choice of RL method made?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies the problem of real-time sequential reasoning: settings where observations arrive as a stream and the agent must keep reasoning while new inputs are arriving, as opposed to ingesting a fixed context and only then iterating. The authors propose a two-timescale architecture that couples a slow observation process (new sensory input every data step \tau) with a fast recurrent reasoning process that can run multiple inner updates (T steps) per observation. Specifically, at each slow data step the observation is encoded into a context vector C_\tau, and a fast recurrent core integrates this new context into its hidden state by iterating a shared update multiple times. The paper also introduces auxiliary memory to stabilise long sequences and prevent representational collapse. The method is evaluated on (i) an ego-centric Maze task, where the model must infer a global path from local views over time, and (ii) MiniGrid RL tasks in a zero-shot OOD setup. In both, the proposed model matches or outperforms LSTM, Mamba-2, and Transformer-XL, especially on OOD mazes and longer-horizon RL variants. The key empirical claim is that success improves as we allow more inner recurrent steps at test time (i.e. test-time scaling works even when inputs are streaming) unlike many previous latent-recurrent approaches that assumed a fixed input window.

### Strengths
- The split between a slow observation process and a fast reasoning loop is intuitive and well explained. 
- The test environments (maze and MiniGrid) are a simple yet reasonable testbed for partial observability and sequential reasoning. 
- The ablation shows that auxiliary memory improves OOD maze performance.

### Weaknesses
- The proposed model performs several internal reasoning steps for each observation, while all baselines do only one. Does this mean it effectively gets more compute per timestep? If so, the performance gains might partly come from having more compute rather than from the architecture itself. 
- The paper uses an internal auxiliary memory for the maze task and an external one for the RL tasks. There’s no general rule for when to use which, which suggests that the design might need to be adjusted for each domain rather than being a single unified approach.
- In the maze task, the baselines are much larger (up to 25–27M parameters) compared to the proposed model (1.1M). Given the small dataset, this could make the baselines more prone to overfitting.

### Questions
- Have you tried training with T reasoning steps and evaluating with T’ > T on the same input sequence?
- What guided the choice between the internal and external memory versions? Could they be combined or selected automatically?
- What happens if the AKOrN module is replaced with a simpler recurrent unit?
- In the maze task, did you explore smaller baselines to check if overfitting explains the gap in OOD performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 7

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses real-time sequential reasoning where a model must reason while observations arrive as a stream, rather than processing a complete sequence first. The authors propose coupling a fast recurrent reasoning module (AKOrN) with slow observational dynamics, using auxiliary memory to maintain continuity during streaming inputs.

### Strengths
- Real-time sequential reasoning is genuinely underexplored. The deadline constraint formalization is precise, and the biological grounding provides solid analogous motivation for the approach.
- Using AKOrN (Kuramoto oscillators on hyperspheres) adds mathematical rigor beyond standard RNNs. The fast-slow timescale separation is architecturally elegant. This method has potential when tested with further experiments and evaluations.

### Weaknesses
- The method shows better test-time scaling on LavaCrossing and MultiRoom but not DoorKey. The authors' explanation (task-dependent benefits) is reasonable but raises a critical question: how do we know a priori which tasks will benefit? Without a principled framework for predicting when this approach helps, practitioners can't know whether to use it. The paper needs either: (a) a theoretical characterization of suitable task types, or (b) extensive empirical analysis across diverse task categories.
- Only toy MiniGrid World environments low dimensional observations and discrete actions are tested. No validation on complex fields like robotics, vision tasks, language, or any realistic sequential decision-making. The strong maze results are promising, but one synthetic task is insufficient to claim this is a general approach for "real-time sequential reasoning."
- The method requires much more computation than baseline through iterative forward passes. However, there is zero analysis of compute cost or scalability. For real-time applications, efficiency is core to the problem definition. The method's ability for "real time" sequential reasoning requires further justification.

### Questions
- The paper cites COCONUT, Looped Transformer, and CTM but doesn't compare against them experimentally. Can you justify for this?
- You use sophisticated Kuramoto dynamics on hyperspheres, but provide no ablations. Can you provide more analysis to show that this is essential?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 8

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors of this submission present an approach to incorporate real-time reasoning in recurrent neural networks. Typically, encoder-decoder sequence models ingest the whole input sequence before processing it to generate the output sequence. Here, the authors motivate parallel processing of partial input sequences while continuing to observe the rest of the sequence. The authors motivate this approach via the requirement of real-time reasoning in reinforcement learning and sequential prediction problems. The recently introduced AKOrN network is adapted to update the internal hidden state at a faster rate relative to the input data clock, and two dedicated fast and slow memory banks are integrated into the adapted AKOrN network that further enhance reasoning performance on synthetic problems like Maze solving.

### Strengths
Strengths:
- The proposed work is a neat extension of recently proposed AKOrN to enhance its sequential reasoning capabilities. The idea of processing inputs as they arrive in a streaming fashion is widely applicable, and the approach here can potentially extend to real-world applications such as video / audio compression with inherently streaming input characteristics.
- Empirical results show the strength of the proposed approach compared to other baselines in both in- and out-of-distribution settings for challenging synthetic reasoning problems. 
- The application to reinforcement learning is also particularly intriguing, I believe the proposed sequential reasoning will be vital to making fast and accurate policies for real world RL.

### Weaknesses
Weaknesses:
- While the empirical results are strong, the tasks used to measure performance gains of real-time reasoning are all toy tasks that don't transfer to performance on naturalistic data. It would be great if the authors could comment on how the proposed approach is relevant to SOTA foundation models which also operate on sequences of input.
- The proposed idea of slow and fast reasoning is not necessarily novel. It is an idea that has been explored in key prior work, e.g. Adaptive computation time for recurrent neural networks, by Graves et al., 2017.

### Questions
NA. Please refer to my review above.

### Soundness
3

### Presentation
3

### Contribution
3
