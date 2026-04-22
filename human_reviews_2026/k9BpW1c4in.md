# FACM: Flow-Anchored Consistency Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Continuous-time Consistency Models (CMs) promise efficient few-step generation but face significant challenges with training instability. We argue this instability stems from a fundamental conflict: Training the network exclusively on a shortcut objective leads to the catastrophic forgetting of the instantaneous velocity field that defines the flow. Our solution is to explicitly anchor the model in the underlying flow, ensuring high trajectory fidelity during training. We introduce the Flow-Anchored Consistency Model (FACM), where a Flow Matching (FM) task serves as a dynamic anchor for the primary CM shortcut objective. Key to this Flow-Anchoring approach is a novel expanded time interval strategy that unifies optimization for a single model while decoupling the two tasks to ensure stable, architecturally-agnostic training. By distilling a pre-trained LightningDiT model, our method achieves a state-of-the-art FID of 1.32 with two steps (NFE=2) and 1.70 with just one step (NFE=1) on ImageNet 256$\times$256. To address the challenge of scalability, we develop a memory-efficient Chain-JVP that resolves key incompatibilities with FSDP. This method allows us to scale FACM training on a 14B parameter model (Wan 2.2), accelerating its Text-to-Image inference from 2$\times$40 to 2-8 steps. Our code and pretrained models:
https://github.com/ali-vilab/FACM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed FACM, a method for stabilizing and improving training of continuous consistency models. The method consists in combining the continuous consistency objective with a flow matching anchor, to encourage the model to learn the underlying flow, which results in improved performance and stability. The method further introduces a novel time embedding strategy and a JVP computation which allows scaling to models with billions of parameters. The method is validated on image generation benchmarks and text-to-image benchmarks and results in competitive performance, beating other one and few-step generative models.

### Strengths
The method is a simplification of continuous consistency models, by leveraging the features learned with flow-matching training. The novel objective is well motivated and effective, and achieves strong performance. The introduction of the scalable JVP computation is valuable for the research community.

### Weaknesses
The method uses now two forward passes, one of which requires JVP computation, so I am concerned about the training efficiency. Nothing about it is reported in the paper, but I think it is important to mention how slower the model gets when trained with FACM.
While simplified compared to sCM, the method still requires complex formulations and weighting functions, for which no ablation is provided.

As a minor note, the authors claim that the MeanFlow code is not available, while it was already by the time of the submission deadline.

### Questions
- Regarding CFG at sampling, does the model require only one NFE or two NFEs? And if it's the 1 NFE case, is the model only conditioned on the conditioning signal, or does it require extra parameters like in MeanFlow?
- Still regarding CFG training, how slower is the model, given that it requires one additional forward pass for $v_{cond}$? Also there does not seem to be any information regarding the probability of using the null token for the pretrained/online base model. In addition, $t_{low}$ is only mentioned as hyperparameter but never in the text.
- What's the reason behind the cosine similarity loss for the FM objective? I am not familiar with this term being used in the flow/score literature, is this part of the contribution or is it commonly used in other FM methods?
- Are the consistency and FM losses simply summed together? Do you use any weighting term or a percentage of the batch similar to Shortcut Models or MeanFlow?

### Soundness
3

### Presentation
3

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
This paper identifies the cause of training instability problem for continuous-time consistency models and proposes a solution of Flow-Anchored Consistency Models(FACM) to solve it. The key design is the Flow-Anchoring principle, where a Flow Matching objective is used as a stable anchor and enables the model to learn both the flow's velocity field and the shortcut efficiently. Furthermore a memory-efficient Chain-JVP scheme is proposed which enables scalable FACM training up to a 14B parameter model. Experiment shows that the proposed algorithm achieves state-of-the-art results for few-step image generation task on a set of benchmarks including ImageNet 256x256 and CIFAR-10.

### Strengths
1. The analysis on the training instability issue in continuous-time consistency models is clear and convincing
2. The proposed flow-anchoring solution is elegant and effective, and the process of dual-objective training is clearly presented.
3. The achieved experiment result is impressive, where in table 1, FACM@NFE=2 outperforms Multi-NFE baseline of LightningDiT.

### Weaknesses
1. Although the paper shows efficiency in terms of reduced NFE, it would be good to also report the total generation time per image.
2. The individual components of the proposed algorithm including the flow matching loss and consistency model loss are existing work, and the proposed FSDP compatible Chain-JVP scheme still has significant computational overhead

### Questions
1. What is the computational overhead of the proposed Chain-JVP method?
2. What is the computational cost of distilling a 14B model and what is the GPU resources required?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Flow-Anchored Consistency Models (FACM), a new framework for training consistency models that explicitly anchors the learned consistency mapping to an underlying flow-matching objective. The authors argue that existing continuous-time consistency models are unstable because they discard the flow-field information during training, leading to mode collapse or divergence. FACM mitigates this by jointly optimizing a flow-matching loss (to learn the velocity field) and a consistency loss (to learn efficient one-step mapping), decoupled through an expanded time interval strategy that ensures smooth continuity between the two objectives. The paper further introduces Chain-JVP, a memory-efficient Jacobian–vector product computation compatible with large-scale distributed training (e.g., FSDP), enabling the training of models up to 14B parameters. Experiments on CIFAR-10 and ImageNet-256 demonstrate that FACM achieves state-of-the-art one- and few-step generation performance, outperforming MeanFlow and sCM, and scales effectively to large text-to-image models.

### Strengths
1. The paper identifies a clear and well-motivated problem — instability in continuous-time consistency model training — and provides a principled remedy via flow anchoring.
2. The formulation is elegant, combining the benefits of flow matching (stability, theoretical grounding) and consistency models (few-step generation) into a unified loss.
3. The expanded time interval trick is simple yet effective, ensuring a smooth transition between FM and CM regions while avoiding gradient coupling issues.
4. The Chain-JVP technique is a practical and non-trivial engineering contribution, making second-order consistency training feasible at scale.
5. Empirical results are strong: FACM outperforms prior methods such as MeanFlow, sCM, and IMM in one- and two-step generation across multiple datasets.
6. The framework is demonstrated on a 14B-parameter text-to-image model, providing credible evidence of scalability beyond toy datasets.

### Weaknesses
1. The theoretical discussion remains somewhat heuristic. While the “anchoring” intuition is appealing, a more rigorous analysis (e.g., convergence guarantees or stability proofs) would strengthen the contribution.
2. The relationship between FACM and existing flow–consistency hybrids (e.g., MeanFlow, iCM) could be clarified — in particular, what distinguishes the proposed anchoring mechanism from joint FM/CM training used before.
3. Some ablations (e.g., varying the relative weight of FM vs. CM losses) are discussed briefly but lack quantitative depth.
4. The paper does not compare computational costs (training time, FLOPs) against prior CM methods, leaving unclear whether stability comes at significant overhead.

### Questions
1. How sensitive is FACM to the weighting between FM and CM objectives, and does this affect sample quality or convergence speed?
2. Could the authors clarify the precise role of the extended time interval—would a shared time embedding with gating suffice?
3. Does the Chain-JVP approach introduce any bias or approximation error compared to full JVP computation?

### Soundness
3

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
This paper proposes Flow-Anchored Consistency Models (FACM), which incorporates a flow matching objective into consistency model training to stabilize training dynamics. The authors demonstrate that the key to effective joint optimization of flow matching and consistency model training is decoupling of flow velocity prediction and mean velocity prediction.

Specifically, in contrast to MeanFlow and its variants which are trained to predict flow velocity at time $t$ when given time condition $(t,t)$ and mean velocity from time $t$ to $r$ when given time condition $(t,r)$, FACMs are trained to predict flow velocity at $t$ given $2-t$ and mean velocity from time $t$ to $1$ given $t$. In contrast to MeanFlow, whose time conditions are coupled, i.e., $r = t$ for flow velocity, FACM uses decoupled time conditions, i.e., $r \neq t$ for flow velocity. Additional techniques such as interpolation of the shortcut target with the current EMA network output and scalable chain-JVP implementation are provided to further accelerate training.

The authors verify the scalability of FACM on CIFAR-10, ImageNet 256x256, and a Text-to-Image dataset. FACM is shown to consistently out-perform several fast baselines such as IMM, MeanFlow, and sCM.

### Strengths
- **[S1] This paper is original in the aspect that it provides a new perspective into training instability of CMs.** Specifically, the authors hypothesize that CM training instability arises from missing flow velocity supervision, and that one should decouple flow and mean velocity time conditions to mitigate conflict between the two velocities.

- **[S2] This paper is significant in the aspect that it provides a number of techniques for scaling CMs.** The authors provide a number of practical techniques, such as interpolation of shortcut target and current prediction and chain-JVP, which are shown to work on difficult tasks such as ImageNet 256x256 and text-to-image generation.

### Weaknesses
- **[W1] The paper lacks theoretical novelty, in the sense that it is a special case of MeanFlow.** MeanFlow learns a flow map between all time pairs $(t,r)$ for $0 \leq t < r \leq 1$, along with a flow matching loss at $t = r$. FACM is a special instance of MeanFlow where a flow map is learned only for time pairs $(t,1)$ for $0 \leq t \leq 1$, also with flow matching loss at $t = r$, where $r = 2 - c\_{FM}$ if one uses expanded time interval proposed in Section 3.3.2. Under this perspective, FACM may be viewed just as a collection of techniques for improved training of MeanFlow.

- **[W2] The paper is missing key ablations of the proposed techniques.** Under the MeanFlow perspective written in [W1], key techniques proposed in this paper can be categorized into (1) fixing $r = 1$, i.e., training only with time pairs $(t,1)$, (2) expanded time interval, and (3) other extraneous techniques such as shortcut target interpolation, residual clamping, and tuning $\alpha(t)$,$\beta(t)$ in Eq. (11) and (13). However, authors only provide an ablation of (2), so with all techniques intertwined, it is difficult to judge what is the largest contributor to the final generative performance.

- **[W3] The key hypothesis for instability of CMs, claimed by the authors, is unsupported.** At lines 171-175, the authors write "However, without a stable anchor in the underlying flow, the model's output $F_\theta$ quickly begins to drift. ... the derivative term in the identity grows to dominate the ground-truth velocity $v$, effectively diluting its supervisory signal." However, there is no supporting theory or experiment to verify this claim.

### Questions
**[Q1] Can the authors provide ablations mentioned in [W2] starting from a MeanFlow baseline?** In particular, I am curious how generative performance changes if (a) one fixes $r = 1$. I expect we would obtain a stronger few-step model, as it is solving an easier task compared to MeanFlow, which learns flow maps for all pairs $(t,r)$.

**[Q2] Can the authors provide theoretical or experimental evidence of the hypothesis that instability in CM is caused by the lack of instantaneously velocity field supervision?** The authors could, for instance, plot the variance of the derivative term $dF_{\theta^{-}}/dt$ without and with flow matching loss.

**[Q3] What is the purpose the cosine similarity term in Eq. (9)?** I believe this loss is redundant, as a cosine similarity is already implicitly contained in the first flow matching objective. If this term plays a non-trivial role, then it should also be included in the ablations.

### Soundness
3

### Presentation
3

### Contribution
3
