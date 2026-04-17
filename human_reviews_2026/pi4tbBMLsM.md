# Local Reinforcement Learning with Action-Conditioned Root Mean Squared Q-Functions

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
The Forward-Forward (FF) Algorithm is a recently proposed learning procedure for neural networks that employs two forward passes instead of the traditional forward and backward passes used in backpropagation. However, FF remains largely confined to supervised settings, leaving a gap at domains where learning signals can be yielded more naturally such as RL. In this work, inspired by FF's goodness function using layer activity statistics, we introduce Action-conditioned Root mean squared Q-Functions (ARQ), a novel value estimation method that applies a goodness function and action conditioning for local RL using temporal difference learning. Despite its simplicity and biological grounding, our approach achieves superior performance compared to state-of-the-art local backprop-free RL methods in the MinAtar and the DeepMind Control Suite benchmarks, while also outperforming algorithms trained with backpropagation on most tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an extension of the recently developed artificial dopamine framework for local reinforcement learning (RL with no or limited backpropagation). The authors propose to use the action as input to the local networks Q prediction, instead of parameterizing a vector of per-action Q function, as done in AD and related RL works such as DQN> They show the impact of their change in empirical experiments on common small-scale RL benchmarks.

### Strengths
The change to the previous AD setup is small, but impactful (I consider small clear changes a strength!). The experiemnts clearly show that moving the action to the input of the network can yield large performance gains. The writing is clear, and the experiments are reasonably designed.

Overall, I believe the method here has merit, but it's current presentation is a bit too superficial for me to fully endorse acceptance (see below). At the same time I believe there are no reasons the paper should not be accepted.

### Weaknesses
While the authors investigate the impact of the action-conditioning, the additional experiments (e.g. Figure 6) don't serve to enhance the understanding of the concrete change proposed in this paper. Why does moving the action to the input conditioning help compared to a DQN like parameterization? This is currently my main issue prevent me from giving a higher rating to the paper and I would be happy if the authors addressed this in the rebuttal.

The papers usefulness for future authors would be helped with a standard appendix containing per env training curves for all algorithms, as well as a clean diagram of the architecture and an easy reference table for hyperparameters. This is standard in the RL literature, and as such it would help the authors method reach a wider audience.

The performance differences between AD and ARQ in the DMC seem partially insignificant, yet they are highlighted. Please ensure that confidence metrics are accounted for.

There are some missing references and other polishing errors:
- Line 342: Figure ??
- Figure 5: ARQ and ARQ w/o AC are indistinguishable

### Questions
Could other forward-forward architectures such as Sun et al. 2025 also be used to improve AD?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ARQ, a biologically inspired local learning algorithm that extends Hinton’s Forward-Forward paradigm to reinforcement learning. ARQ replaces backpropagation with a forward-only value estimation mechanism using action-conditioned root mean squared Q-functions, enabling local, backprop-free RL. The work looks to extend the Artificial Dopamine framework by eliminating some action-space constraints. The results show some improvements on MinAtar and DM Control Suite benchmarks.

### Strengths
The paper is well-organized and written in a clear, accessible manner. The motivation, extending Hinton’s biologically plausible FF learning to reinforcement learning, is well introduced. Figures effectively illustrate the difference between supervised and local RL setups and the architecture of ARQ versus Artificial Dopamine baseline. 

The experiments are conducted on standard small-scale with consistent settings so comparisons against AD baseline and backprop-based methods like DQN and SAC are appropriate. 

Across both benchmarks, ARQ consistently outperforms AD and can be better than traditional backprop-based methods. 

The visualization of neuron activations provides a nice illustration of how the proposed method might be creating local representations for specific state-action pairs.

The biological plausibility of the approach is discussed.

### Weaknesses
Conceptually, the paper is a pretty straightforward extension of Guan et als 2024 Artificial Dopamine work. The core novelty seems to be in replacing the scalar Q-value with an RMS alternative. 

There is little extension of the AD idea for biologically plausible RL. 

Much of the architectural structure, training pipeline and evaluation setup is directly taken from the AD work

The root-mean-square formulation is introduced heuristically with little theoretical motivation or analysis of properties – which means the paper lacks new theoretical insight or learning principle beyond empirical observation.

The evaluation is restricted to low-dimensional benchmarks. These are useful for proof-of-concept, but they are not sufficient to demonstrate scalability or robustness of the approach. 

The paper repeatedly emphasizes biological plausibility, but the actual learning rule remains a standard TD-style update with neural architectures that use FF’s local updates. There’s no evidence of truly biologically grounded mechanisms so biological plausibility feels somewhat overplayed.

While the ablation studies are appreciated, they are minimal. limited to choice of nonlinearity and action conditioning. There is little exploration of architectural sensitivity (e.g., hidden dimension size, number of layers, effect of top-down connections). Given the absence of theory, a more extensive empirical analysis would be needed to strengthen the paper.

### Questions
While the analogy to the “goodness” function from the Forward-Forward algorithm is intuitive, there’s no mathematical argument explaining why this RMS formulation should lead to better learning stability or generalization. Can you offer insight?

The claimed innovation — using the root mean squared of the hidden vector after mean subtraction — is a relatively minor modification to existing FF or AD approaches. It’s more of a functional reparameterization than a conceptual breakthrough, and the improvement in results could partly stem from increased capacity rather than a new idea.

It remains unclear whether ARQ could handle high-dimensional, visual, or partially observable tasks.

Ablations are limited to choice of nonlinearity and action conditioning. There is little exploration of architectural sensitivity (e.g., hidden dimension size, number of layers, effect of top-down connections and so on). Given the absence of theory, a more extensive empirical analysis would strengthen the paper.

Just 3 seeds must give poor measures of standard error – this just seems wrong… you then go on to bold best performing algo in the tables, without considering statistical significance – even when there are only 3 replicates and stds overlap. Canyou give significance?

Minor things
P7 Figure ?? – latex compile issue
Protect Capitals in bib – like {A}tari

### Soundness
2

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
4

### Summary
This work follows a recent trend of modifying neural network training to operate in a layer-by-layer manner, rather than processing inputs through all layers and updating the network parameters in a single pass. It builds on Artificial Dopamine (AD), which learns Q-values by training each layer separately and then averaging the value estimates from all layers during policy execution. This work modifies the AD architecture by incorporating the action directly into each layer’s input, whereas AD only takes the state as input and outputs multiple action-specific heads—an approach more similar to a DQN-style network.

The authors conducted comprehensive experiments across a diverse set of tasks, demonstrating a consistent performance improvement of their method, particularly over its counterpart, Artificial Dopamine.

### Strengths
The proposed modification represents a solid and reasonable extension to AD, supported by empirical results that show clear improvements over the original method. Additionally, the authors provide comprehensive ablation studies that further validate the effectiveness of their approach.

### Weaknesses
I would suggest that the authors consider running experiments with additional random seeds to more robustly validate the advantages of the proposed method over AD. Using only three seeds—especially when reporting such large performance gaps—may not be sufficiently convincing to support the claim that the proposed approach is fundamentally superior.

Moreover, this line of research is largely driven by heuristic intuitions inspired by analogies between biological and artificial neurons. Given the substantial performance improvements observed, the contribution would be significantly strengthened by providing deeper theoretical insights, particularly in the context of reinforcement learning. For instance, value estimation in the Bellman equation can be viewed as a dynamic programming process that propagates rewards across state–action pairs. Does the proposed method accelerate this propagation, or is there another underlying mechanism that explains the improvements? A clearer theoretical justification—beyond empirical evidence—would help reinforce the contribution and clarify why both the proposed method and its predecessor, AD, achieve such notable gains.

### Questions
A specific question regarding the architectural design:

As far as I understand AD, each layer contains multiple attention matrices, with the number matching the size of the discrete action space. Specifically, for each layer $l$, there is a set of matrices $W_{\text{attn},1}^{[l]}, W_{\text{attn},2}^{[l]}, \ldots, W_{\text{attn},|A|}^{[l]}$, where each is responsible for computing the Q-value corresponding to a particular action.

In this work, the authors argue that they remove this constraint. In particular, according to Equation (13), the network outputs a vector $y_t^l$. However, it is unclear to me how such a vector could be generated using a single attention matrix, assuming $W_{\text{attn}}$ is a two-dimensional matrix and $\mathbf{X}$ and $\mathbf{h}_t^l$ are vectors.

Is it the case that the authors introduce an arbitrary number of attention matrices to produce the vector $\mathbf{y}_t^l$? Could you provide more details on the architectural design so that the innovation at the architecture level becomes clearer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce an ARQ method which does not use the backpropagation update. Experiments on MinAtar benchmarks show ARQ has higher reward than backpropation methods.

### Strengths
1. the idea is novel as it bridges FF and TD learning
2. the result looks promising. Experiments on MinAtar benchmarks show ARQ has higher reward than backpropation methods.

### Weaknesses
1. The author should carefully review the manuscript for consistency and completeness. For example: Line 156: The term “q-function” should be standardized to “Q-function” for consistency throughout the paper. Line 342: The placeholder “Figure ??” needs to be replaced with the correct figure number.

2. Based on Table 2, while ARQ demonstrates strong performance compared to local RL methods and even surpasses backprop-based algorithms in many discrete tasks, its results on continuous control tasks (DeepMind Control Suite) remain slightly below state-of-the-art actor–critic methods

### Questions
1. The authors note that output-layer conditioning is computationally efficient for discrete tasks with low-dimensional action spaces. Could you provide quantitative or qualitative evidence on how ARQ compares in terms of computational efficiency to previous local RL methods (e.g., AD) and backprop-based baselines? For example, does ARQ reduce memory usage, training time, or inference cost?

2. Is it feasible to integrate ARQ into an actor–critic architecture? If so, what are the main challenges or design constraints that would need to be addressed?

3. Could you clarify the conceptual difference between decentralized Q-learning (as in ARQ or multi-agent setups) and distributional Q-learning approaches like D4PG? While distributional Q models the return distribution, does it also imply decentralization, or are these fundamentally distinct paradigms?

4. What are the shades in Figure 5 represents for? is it std or confidence interval? Please be specified in the caption

### Soundness
3

### Presentation
3

### Contribution
3
