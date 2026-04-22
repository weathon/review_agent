# Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Reinforcement Learning (RL) has become the _de facto_ standard for tuning LLMs to solve tasks involving reasoning.
However, growing evidence shows that models trained in such way often suffer from a significant loss in diversity.
We argue that this arises because RL implicitly optimizes the "mode-seeking" or "zero-forcing" _Reverse KL_ to a target distribution causing the model to concentrate mass on certain high-probability regions of the target while neglecting others. 
In this work, we instead begin from an explicit target distribution, obtained by filtering out incorrect answers while preserving the relative probabilities of correct ones.
Starting from a pre-trained LLM, we approximate this target distribution using the $\alpha$-divergence family, which unifies prior approaches and enables direct control of the precision–diversity trade-off by interpolating between mode-seeking and mass-covering divergences.
On a Lean theorem-proving benchmark, our method achieves state-of-the-art performance along the coverage–precision Pareto frontier, outperforming all prior methods on the coverage axis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Distributional Matching with Verifiable Rewards (DMVR), a framework for training LLMs on reasoning tasks that explicitly defines a target distribution by filtering incorrect answers while preserving relative probabilities of correct ones. The authors introduce Amari-DPG, which uses Amari's α-divergence family to interpolate between mode-seeking (Reverse KL) and mass-covering (Forward KL) divergences, enabling control over the precision-diversity trade-off. Experiments on LEAN theorem proving demonstrate state-of-the-art results along the coverage-precision Pareto frontier.

### Strengths
- Clear conceptual contribution: The paper provides a unified perspective that clarifies how RLVR methods implicitly optimize Reverse KL to a filtered distribution, explaining diversity loss in a principled way.

- Theoretically grounded: The connection between RLVR and distributional matching (Lemmas 1-2) is well-established, and the use of α-divergences to interpolate between objectives is mathematically sound.

- Comprehensive experimental analysis: The diversity analysis using Simpson index and Shannon entropy, along with problem difficulty transitions, provides valuable insights into model behavior.

### Weaknesses
## Limited Novelty

The core technical contribution is incremental. The paper essentially applies existing f-DPG methods with Amari α-divergences to the RLVR setting. While the unification perspective is useful, the algorithmic novelty is limited:

- The connection between RLVR and Reverse KL was already established by Korbak et al. (2022b)
- f-DPG and KL-DPG are existing methods
- Amari α-divergences are well-known in the literature

The main contribution is demonstrating that α ≈ 1 can outperform standard GRPO, but this feels more like careful hyperparameter tuning than a fundamental advance.

Aligning Language Models with Preferences through f -divergence Minimization.

On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting.

A Distributional Approach to Controlled Text Generation.


## Narrow Experimental Validation

- Single domain: Experiments are limited to LEAN theorem proving. The generalizability to other reasoning tasks (code generation, mathematical problem solving, multi-step reasoning) is unclear.

- Single base model: Only DeepSeek-Prover-V1.5-SFT (7B) is evaluated. Scaling behavior and performance on larger models is unknown.

## Overclaimed Improvements

- It missed a lot of relevant baselines: such as DAPO, Dr. GRPO, GPG, etc.

- Looking at Figure 2, the comparison is confusing. It seemed that Amari-DPG improved pass@1 but significantly harmed the pass@k when k becomes large (e.g., 256, 512). This can be understood as the model loses a lot of answer diversity. 


## Overlook of Relevant Papers

The authors claim in the conclusion section that RLVR does not generate fundamentally new capabilities but instead reweights and amplifies behaviors already present in the base model. However, they failed to cite the very closely related papers that discovered the same phenomenon. 

The invisible leash: Why rlvr may not escape its origin.

### Questions
How does the method perform on other reasoning domains beyond LEAN?

What is the computational overhead compared to GRPO in wall-clock time?

Can you provide statistical significance tests for the main results?

Why not compare to other recent LEAN training methods (e.g., Kimina-prover, InternLM2.5-StepProver)?

How sensitive are results to the ε threshold for Z_c?

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
The paper presents DMVR, a distributional matching view for post training with verifiable rewards. The key idea is to form an explicit target distribution by filtering out incorrect generations with a verifier and then approximate this target with an autoregressive policy. The authors show that standard RL with verifiable rewards is equivalent to minimizing the reverse KL to a softened filtered target, which explains the loss of diversity in many RL runs. They then propose Amari DPG, which minimizes an α divergence to the filtered target and gives direct control over the precision and diversity tradeoff. Following theorem proving, the method traces a Pareto frontier between pass@1 and pass@256 and improves coverage over several GRPO style baselines.

### Strengths
1. The paper formalizes the filtered target distribution, pc, and proves that maximizing the usual RLVR objective equals minimizing the reverse KL to a softened filter. This clarifies why RLVR becomes mode seeking and why diversity collapses in practice. The argument is simple but convincing.
2. The method unifies rejection sampling fine-tuning at one end and RLVR-style training at the other end, and allows smooth interpolation between mass covering and mode-seeking regimes. This aligns with prior distributional-matching insights and provides a knob practitioners can use.
3. The paper also measures tactic and premise diversity and shows how those relate to pass@1 and pass@256, which is informative for understanding diversity effects beyond one score.

### Weaknesses
1. Evaluation is quite narrow. All main experiments are with a single base model family and a 7B scale, which is my biggest concern.
2. The GRPO baselines use β=0 by default and add a single high KL setting. However, recent works show that reward shaping like rewarding the unlikely or optimizing pass@k can rescue coverage. A stronger baseline suite with these variants tuned as carefully as Amari DPG would make the frontier result harder to question. [1] Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening

### Questions
1. How does Amari DPG compare against two recent diversity aware baselines: pass@k training and rewarding the unlikely, when all methods receive similar compute and careful tuning. If combined, do they add or conflict.
2. Several recent papers suggest that RL tends to reweight existing solutions rather than create new ones. Do you see evidence that α near zero brings the model closer to the base distribution in that sense, for example by recovering harder rare solutions that the base already had. Any analysis of overlap of successful trajectories would be helpful.
3. If possible, conduct more experiments on different model families. (I know it may be too much for a rebuttal, and I will not take this into my final rating, just for the paper quality considerations.)

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
3

### Summary
This paper addresses the diversity loss issue in LLMs tuned via Reinforcement Learning by proposing the Distributional Matching with Verifiable Rewards (DMVR)  framework. It constructs an explicit target distribution by filtering out incorrect answers while preserving the relative probabilities of correct ones from a pre-trained base model. To approximate this target, the paper introduces Amari-DPG, which leverages Amari’s α-divergence family to enable controllable trade-offs between precision and diversity.

### Strengths
1. The theoretical analysis is rigorous, clearly attributing the diversity loss in RL-tuned LLMs to the mode-seeking behavior induced by the Reverse KL divergence.

2. The paper is well-structured and addresses a meaningful research question, highlighting the significance of the accuracy–diversity trade-off in RL-aligned LLMs.

### Weaknesses
The main weakness lies in the experimental evaluation.

1. The authors focus on RL algorithms in LLMs, yet RL-specific experiments are limited. It is essential to include baselines covering algorithms such as PPO and REINFORCE.

2. Additionally, evaluations are confined to Lean. It would be valuable to incorporate 1–2 additional tasks (e.g., code generation or natural language mathematical reasoning) to assess the framework’s general applicability beyond formal theorem proving.

3. Why doesn't Lean's testset use mainstream theorem proving benchmark, such as MiniF2F？

If these concerns are adequately addressed, I will consider raising my score.

### Questions
1. Amari-DPG (α=0) uses on-policy sampling. Compared to RS-FT’s base model sampling, how does it perform in terms of sample efficiency (e.g., rejection rate of incorrect samples)?

2. Amari-DPG (α=0.25) shows limited improvement in the efficiency of medium-difficulty problems. Can dynamically adjusting α during training balance the goals of "preserving the solvability of hard problems" and "improving the efficiency of medium-difficulty problems"?

3. In Section 3.2, you argue that Amari’s α-divergence unifies Reverse KL  and Forward KL by interpolating between α→1 and α→0. However, the paper only shows gradient equivalence for these edge cases. Can you provide a formal proof (or more rigorous intuition) that the α-divergence’s interpolation smoothly preserves the "mode-seeking to mass-covering" transition across all $\alpha \in (0,1)$ ?

4. You unify RLVR and RS-FT by showing Amari-DPG recovers both methods at α extremes. However, RLVR uses a KL penalty ($\beta$) to balance reward and proximity to $\pi_{base}$, while Amari-DPG sets $\beta=0$ by default. Theoretically, how does Amari-DPG’s α parameter interact with RLVR’s $\beta$ if both are used together?

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
3

### Summary
Building on prior works on Distributional Matching, the authors consider the target distribution $p_c$, which is essentially the I-projection of the base policy onto the space of distributions that strictly filters out incorrect responses. The authors then argue that the RLVR optimizes the reverse KL to $p_c$, which potentially reduces diversity. In order to interpolate between the reverse and forward KL behaviors, the authors propose **Amari-DPG**, which utilizes Amari's $\alpha$-divergence for the $f$-DPG (Go et al., 2023). The efficacy of this is shown in a formal math dataset with Lean, along with extensive ablations.

### Strengths
- Paper is well-written and easy to follow, with enough background explanation
- A simple approach with meaningful peformance gain
- Extensive ablations

### Weaknesses
- Novelty-wise, at the end of the day, the methodology is precisely f-DPG of Go et al. (2023), with f-divergence replaced with Amari's $\alpha$-divergence. Also, the target distribution $p_c$ is taken from Khalifa et al. (2021). But, I also concur that the simplicity of the proposed method overshadows the "lack" of novelty.
- The writing can be made a bit clearer. As this is a mixture of prior works, a clearer separation of the authors' contributions and prior works in Section 3 would be helpful.
- The computation of the partition function, $Z_c$, is not mentioned explicitly anywhere in the paper. This makes the paper not so self-contained.

**minor suggestion**
- The "original" reference for the alpha-divergence is [1], and thus also referred to as Renyi's alpha-divergence. But, their forms are a bit different, although they are equivalent up to reparametrization; see Appendix B of [2], for instance. Maybe good to mention this just for clarity..!


[1] https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/On-Measures-of-Entropy-and-Information/chapter/On-Measures-of-Entropy-and-Information/bsmsp/1200512181

[2] https://proceedings.mlr.press/v70/li17a.html

### Questions
1. What may be a principled way of choosing $\alpha$ other than doing a grid search of $\alpha$?

2. Does different value of $\alpha$ impact the trainig stability/convergence rate?

3. Albeit with extensive ablations, the authors only consider the formal math task. What other tasks may benefit significantly from the proposed approach?

4. Can one think about "tuning" the $\alpha$ throughout the training? Of course, this means that the partition function would have to be reevaluated, but just a random thought.

5. (Minor) Very recently, there have been some issues regarding floating-point operations (https://x.com/QPHutu/status/1984258808332550245). Can the authors comment on whether this issue(?) applies to their experiments?

6. (Minor) Other than Amari's $\alpha$-divergence, there is another way of interpolating forward and reverse KL via density ratio metrics (DRMs) [1]. Could the authors comment on what would happen if such another interpolation is used?

### Soundness
3

### Presentation
3

### Contribution
3
