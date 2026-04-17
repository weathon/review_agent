# Elmur: External Layer Memory With Up- Date/Rewrite For Long-Horizon Rl Problems

Egor Cherepanov1,2, Alexey K. Kovalev1,2**, Aleksandr I. Panov**1,2 1AXXX, 2MIRIAI
cherepanov@axxx.tech

## Abstract

Real-world robotic agents must act under partial observability and long horizons, where key cues may appear long before they affect decision making. However, most modern approaches rely solely on instantaneous information, without incorporating insights from the past. Standard recurrent or transformer models struggle with retaining and leveraging long-term dependencies: context windows truncate history, while naive memory extensions fail under scale and sparsity. We propose ELMUR (External Layer Memory with Update/Rewrite), a transformer architecture with structured external memory. Each layer maintains memory embeddings, interacts with them via bidirectional cross-attention, and updates them through an Least Recently Used **(LRU)** memory module using replacement or convex blending. ELMUR extends effective horizons up to 100,000 times beyond the attention window and achieves a 100% success rate on a synthetic T-Maze task with corridors up to one million steps. In POPGym, it outperforms baselines on more than half of the tasks. On MIKASA-Robo sparse-reward manipulation tasks with visual observations, it nearly doubles the performance of strong baselines, achieving the best success rate on 21 out of 23 tasks and improving the aggregate success rate across all tasks by about 70% over the previous best baseline. These results demonstrate that structured, layer-local external memory offers a simple and scalable approach to decision making under partial observability. Code and project page: **https://elmur-paper.github.io/**.

## 1 Introduction

Imagine a robot cooking pasta: it stirs once, adds salt, and later adds salt again, repeating until the dish is inedible. The issue is simple: the robot cannot remember if salt was already added, since it dissolves invisibly, nor how much is still in the container. This is a case of partial observability - the world rarely reveals all necessary information. Humans recall past actions effortlessly, but robots lack this ability. Though effective in controlled settings (Kim et al., 2024; Black et al., 2024), robots often fail under partial observability (Fang et al., 2025; Cherepanov et al., 2026a). Standard recurrent (Ouyang et al., 2025) and transformer (Gao et al., 2025) models rely heavily on short observation windows, making them brittle under long-horizon dependencies and sparse signals. This motivates hybrid memory-augmented transformers that explicitly store and retrieve past information (Fang et al., 2025; Shi et al., 2025). Within the Reinforcement Learning (RL) paradigm (Sutton et al., 1998), long-horizon challenges are compounded by sample inefficiency and sparse rewards: real-world exploration is costly and unsafe, while simulation suffers from a sim-to-real gap (Zhang et al., 2025a). Offline RL mitigates this with pre-collected datasets (Levine et al., 2020), but usually assumes dense feedback; reshaping sparse rewards demands domain knowledge and risks bias (Wu et al., 2021; Mu et al., 2024; Wang et al., 2025a). In robotics, delayed feedback makes long-term memory indispensable. A complementary paradigm is Imitation Learning (IL) (Zare et al., 2024), whose simplest form, Behavior Cloning (BC), reduces control to supervised learning on demonstration pairs. Building on this idea, recent Vision-Language-Action (VLA) models (Brohan et al., 2022; Team et al., 2024; Kim et al.,
2024) scale with large datasets, yet their fixed transformer windows (Fang et al., 2025; Shi et al., 2025) leave three challenges: (i) extending context without quadratic cost, (ii) mitigating truncationinduced forgetting, and (iii) retaining task-relevant information across long horizons. This motivates our central question: **how can we equip IL policies with efficient long-term memory to solve** long-horizon, partially observable tasks?

1

![1_image_0.png](1_image_0.png) 

To address these challenges, we introduce **ELMUR** (External Layer Memory with Update/Rewrite), a transformer architecture in which every layer is augmented with a structured external layer memory (Figure 1). ELMUR combines three ingredients: (i) layer-local memory embeddings that persist across segments, (ii) bidirectional token–memory read/write interaction via cross-attention
(mem2tok, tok2mem), and (iii) a Least Recently Used (LRU) update block that refreshes memory through replacement or convex blending, balancing stability and adaptability. This design enables efficient segment-level recurrence and extends retention of task-relevant information up to 100,000× beyond the native attention window, making long-horizon decision making feasible in robotics. We evaluate ELMUR on the synthetic T-Maze (Ni et al., 2023), the robotic MIKASA- Robo (Cherepanov et al., 2026a) suite of sparse-reward manipulation tasks with visual observations, and the diverse POPGym benchmark (Morad et al., 2023a), all designed to test memory under partial observability. ELMUR achieves a 100% success rate on T-Maze corridors up to one million steps, nearly doubles baseline performance on MIKASA-Robo, ranking first on 21 of 23 tasks and increasing the overall success rate across the suite by roughly 70% relative to the strongest prior method, and obtains the top score on 24 of 48 POPGym tasks. These results demonstrate that EL- MUR enables stable retention of task-relevant information, efficient long-term storage, and robust generalization under partial observability. Our contributions are twofold:
- **We propose ELMUR**, a transformer with layer-local external memory, bidirectional tokenmemory cross-attention, and an LRU-based update rule rewriting memory via replacement or convex blending (Section 3). This design extends memory horizons far beyond the attention window.

- **We empirically demonstrate** that ELMUR achieves robust generalization under partial observability across synthetic, robotic, and puzzle/control tasks (Section 5).

- **We provide a theoretical analysis** of LRU-based memory dynamics, establishing formal bounds on forgetting, retention horizons, and stability of memory embeddings (Section 4).

## 2 Background

Many real-world robotic and control tasks involve partial observability, where the agent cannot directly access the true system state (Lauri et al., 2022). This setting is modeled as a partially observable Markov decision process (POMDP), defined as the tuple (S, A, O, T, Z, R, ρ0, γ), with latent state space S, action space A, and observation space O. The transition dynamics are T : S × A → ∆(S), where T(s
′| *s, a*) is the probability of reaching s
′after taking a in s. The observation function Z : *S × A →* ∆(O) specifies Z(o | s
′, a), the probability of observing o after Algorithm 1 ELMUR layer update for segment i at layer ℓ. Inputs are token hidden states h ∈ R 
B×L×d, memory (*m, p*) with m ∈ R
B×M×d, anchors p ∈ Z
B×M, and absolute times t. Outputs are updated hidden states h
′and memory (m′, p′).

// Input embedding (before first layer) - to encode observation 1: h ← ObsEncoder(o)
// Token track - sequence processing and enrichment with information from memory 2: h ← AddNormh + SelfAttention(h; causal mask) 3: Brel ← RelativeBias(t, p) *// bias for adding temporal dependence* 4: h ← AddNormh + CrossAttention(Q=*h, K*=m, V =m; noncausal mask, Bread)
5: h ← AddNormh + TokenFFN(h)
6: h
′ ← h
// Output decoding (after final layer)
7: a ← ActionHead(h
′) // map to action distribution and compute loss
// Memory track 8: Brel ← RelativeBias(p, t) *// reversed bias for write* 9: u ← AddNormm + CrossAttention(Q=m, K=h
′, V =h
′; noncausal mask, Bwrite)
10: u˜ ← AddNormu + MemoryFFN(u)
11: (m′, p′) ← LRU(m, p, *u, t* ˜ )
12: **return** h
′, (m′, p′)
reaching s
′ under action a. The reward function is R : *S × A →* R, the initial state distribution is ρ0 ∈ ∆(S), and γ ∈ (0, 1) is the discount factor. In the special case of full observability, the observation equals the state (ot ≡ st), reducing the POMDP to a Markov decision process (MDP). The optimal policy then depends only on the current state, π
∗(at | st). In the general POMDP case, however, the agent cannot access st directly and must rely on the full history ht = (o0, a0, o1, a1*, . . . , o*t), yielding π
∗(at | ht). A practical alternative is to approximate history with a learned memory state mt = fϕ(mt−1, ot, at−1), πθ : M → ∆(A), πθ(at | mt), where fϕ is a, for instance, recurrent (Hausknecht & Stone, 2015) or memoryaugmented (Parisotto et al., 2020) update rule.

## 3 Method

Many real-world decision-making tasks involve long horizons and partial observability, where key information may appear thousands of steps before it is needed. Standard transformers are limited by a fixed attention window: naive extensions of context length increase cost quadratically, while truncation causes forgetting. Efficient long-term reasoning thus requires a mechanism to store and retrieve task-relevant information across long trajectories. To this end, we propose **ELMUR** (External Layer Memory with Update/Rewrite), a GPT-style (Radford et al., 2019) transformer decoder augmented with structured external memory. Unlike architectures that simply cache hidden states (Dai et al., 2019), ELMUR
equips each layer with its own memory track and explicit read–write operations, enabling persistent storage and selective updating via the LRU memory management.

![2_image_0.png](2_image_0.png)

Figure 2: LRU-based memory management in EL-
MUR. Each layer maintains M memory slots, initialized with random vectors (green). As new segments arrive, tokens write updates into empty slots (purple) by full replacement. Once all slots are filled, the least recently used slot is refreshed via a convex update with parameter λ that blends new content with the previous memory (grey). Anchors below each row indicate the timestep of the most recent update. This scheme ensures bounded capacity while preserving long-horizon information.

ELMUR Overview. As shown in Figure 1, each ELMUR layer has two coupled tracks. The **token** track processes observations into actions, while the **memory track** persists across segments. Both interact through cross-attention: memory shapes token representations, and tokens update memory. Interaction occurs via mem2tok (read) and tok2mem (write) blocks, modulated by relative biases from token timesteps and memory anchors. Trajectories are split into segments, processed sequentially for efficiency and recurrent memory updates. At each segment's end, hidden states update memory, carried forward. LRU memory management fills empty slots first, then refreshes the least used slot by convexly blending old and new information. This bidirectional design provides temporally grounded memory for long-horizon decisions. Algorithm 1 summarizes the method. Segment-Level Recurrence. Feeding infinitely long sequences into a transformer is infeasible, since self-attention scales quadratically. Splitting into shorter segments reduces cost but complicates information flow. Segment-level recurrence addresses this by treating the transformer as an RNN over segments, passing memory from one segment to the next (Dai et al., 2019; Bulatov et al., 2022). In ELMUR, this memory is realized as layer-local external memory instead of cached activations. Each layer maintains memory that is read within the current segment and updated before moving to the next. Formally, with context length L, a trajectory of length T is partitioned into S = ⌈*T /L*⌉ segments Si: h
(i) = TokenTrackSi, sg(mi−1), where h
(i) ∈ R
B×L×d denotes the hidden states of tokens in segment i, computed from the segment Si and the detached memory sg(mi−1) carried from the previous segment.

Token Track. Within each segment Si, observations are encoded into token embeddings x ∈
R 
L×d, where d is the model dimension. The token track models local dependencies and augments them with information from memory m ∈ RM×d. Standard transformers rely on fixed-window selfattention, whereas ELMUR also retrieves information from its external memory via cross-attention, allowing predictions to depend not only on recent tokens but on distant past events stored in memory. Self-attention, equipped with relative positional encodings (Dai et al., 2019) and a causal mask, models local dependencies within the segment:
hsa = AddNorm(x + SelfAttention(x)), (1)
where AddNorm(·) denotes a residual connection followed by normalization. Long-term context is handled by external memory. Tokens' hidden states hsa then query memory via the mem2tok:
hmem2tok = AddNorm(hsa + CrossAttention(Q = hsa*, K, V* = m)). (2)
Here memory embeddings act as keys and values, with a non-causal mask and a relative bias reflecting token-memory temporal distance. Finally, representations are refined with a feed-forward network (FFN). In contrast to popular Decision Transformer (DT) Chen et al. (2021) that employ a standard MLP-based FFN, we adopt a DeepSeek-MoE FFN (Dai et al., 2024), following the design of DeepSeek-V3 (Liu et al., 2024a). Mixture-of-Experts (MoE) improve parameter efficiency and specialization by routing tokens to a sparse set of experts, scaling capacity without proportional compute. This design enables expressive updates while keeping inference efficient:
h = AddNorm(hmem2tok + FFN(hmem2tok)). (3)
The resulting hidden states are then passed to the action head, applied only after the final layer.

Training is supervised, minimizing the error between predicted and demonstrated actions, using mean squared loss for continuous spaces and cross-entropy for discrete ones. The loss backpropagates through the entire network to update model parameters. Memory Track. Reading from memory is not enough for long-horizon reasoning; the model must also write new information. Without an explicit write path, past events would be forgotten or cached inefficiently. The memory track addresses this by allowing tokens to update persistent memory, retaining salient information while overwriting less useful content.

Each layer maintains its own memory embeddings m ∈ RM×d. After processing a segment, token states update memory through the tok2mem block:
mtok2mem = AddNorm(m + CrossAttention(Q = m*, K, V* = h)). (4)
As in mem2tok, a non-causal mask is applied, but the relative bias is reversed to favor temporally aligned memory embeddings. Updates are then refined by a FFN with residual connection:
mnew = AddNorm(mtok2mem + FFN(mtok2mem)), (5)
analogous to the token track, the FFN uses a DeepSeek-MoE block instead of a standard MLP.

Finally, mnew is merged with existing slots via the LRU rule (Figure 2, Figure 2), filling empty slots first and otherwise refreshing the least recently used by convex blending. This keeps memory bounded yet consistently updated with relevant information. Relative Bias. When memory extends across multiple segments, absolute indices become ambiguous: the same token position may correspond to different points in the trajectory. To resolve this, the model requires a signal that encodes relative distances between tokens and memory entries. ELMUR provides this signal through a learned relative bias added to cross-attention logits:

$$\operatorname{Attn}(\mathbf{Q},\mathbf{K})={\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_{h}}}}+\mathbf{B}_{\mathrm{rel}}.$$
$$(6)$$
√dh+ Brel. (6)
The bias Brel is derived from pairwise offsets ∆ = ±(t−p) between a token position t and a memory anchor p (the last update time of a slot). Offsets are clamped to [−Dmax+1, Dmax−1], where Dmax is the maximum relative distance supported by the bias table. These clamped values are shifted into
[0, 2Dmax−2] and used to index a learnable embedding table E ∈ R
(2Dmax−1)×H, where H is the number of attention heads. Each offset corresponds to a per-head embedding E[∆] ∈ R
H, and stacking these indices produces

so produces  $\mathbf{B}_{\text{rel}}=\begin{cases}\mathbf{E}[t-p]\in\mathbb{R}^{B\times H\times L\times M},\text{mem2}\texttt{tok}\texttt{(read)}\\ \mathbf{E}[p-t]\in\mathbb{R}^{B\times H\times M\times L},\text{tok2}\texttt{mem}\texttt{(write)}.\end{cases}$
$$(T)$$
In the read path (mem2tok), the bias prioritizes retrieval from temporally close memory embeddings while keeping distant ones accessible. In the write path (tok2mem), offsets are reversed, guiding updates toward memory embeddings aligned with the writing tokens. Both directions draw from the same embedding table E but can learn distinct patterns. By relying on relative rather than absolute timestep, ELMUR ensures consistent and coherent memory interactions across long horizons. Memory Management with LRU. External memory must remain bounded: storing every token is infeasible, while naive truncation risks catastrophic forgetting. A principled policy is needed to decide which slots to refresh or preserve as new content arrives. ELMUR employs a Least Recently Used (LRU) block (Figure 2, Figure 2) that manages M slots per layer, each holding a vector and an anchor (its last update time). By always updating the least recently used slot, the block ensures bounded capacity while retaining context.

At training start, **initialization** samples embeddings from N (0, σ2I) and marks them empty. While empty slots remain, **full replacement** inserts new vectors directly. Once all slots are filled, the block switches to **convex update**, blending the oldest slot with new content:
mi+1 j = λ mi+1 new + (1 − λ) mij, (8)
where λ ∈ [0, 1] is a tunable hyperparameter that controls the balance between overwriting and retention. By adjusting λ, one can choose whether memory favors fast plasticity (larger λ) or longterm stability (smaller λ). This policy uses memory capacity fully before overwriting and applies gradual blending thereafter, enabling bounded yet persistent long-horizon memory.

Algorithm 2 LRU update for layer memory. Inputs: current memory (*m, p*) (may be uninitialized), candidate updates u˜, newest segment time t, blend λ∈[0, 1],
init scale σ. Output: updated memory (m′, p′).

// Initialization (cold start)
1: if *m, p* uninitialized **then** 2: m ← N (0, σ2I) *// initial slots* 3: p ← −1 *// sentinel anchors* 4: **end if**
// Choose write index 5: empty ← (p < 0) 6: if any empty **then** 7: j
⋆ ← first(empty) *// use first empty slot* 8: α ← 1 *// full replacement* 9: **else**
10: j
⋆ ← arg minj pj *// least recently used* 11: α ← λ *// convex blend* 12: **end if**
// Integrate 13: blend ← α u˜j
⋆ + (1 − α) mj
⋆
14: m′ ← m; m′j
⋆ ← blend 15: p
′ ← p; p
′j
⋆ ← t 16: **return** (m′, p′)

$$\mathbf{\partial}\cdot(1-\lambda)\,\mathbf{m}_{j}^{i},$$

By combining token-level processing with an explicit memory system, ELMUR offers three core advantages: (i) relative-bias cross-attention provides temporally grounded read-write access, (ii) the LRU-based manager ensures bounded capacity while remaining adaptive, and (iii) segment-level recurrence enables scalable learning over long horizons.

## 4 Theoretical Analysis

Understanding the retention properties of ELMUR's memory is crucial for characterizing its ability to handle long-horizon dependencies. In this section, we analyze how information is preserved or forgotten under the LRU update mechanism. We derive bounds on memory retention and effective horizons, and connect these results to the empirical behaviors observed in long-horizon tasks. At the core of ELMUR's memory module is the convex update rule with blending factor λ ∈ [0, 1]. Let fix a memory embedding j at segment index i. If this memory embedding is selected for update with new content mi+1 new , the rule (Figure 2) is mi+1 j = λ mi+1 new + (1 − λ) mij
, while all other memory embeddings n ̸= j remain unchanged: mi+1 n = min. If the memory embedding was empty, the update reduces to full replacement mi+1 j = mi+1 new .

Proposition 1 (Exponential Forgetting). After k overwrites of memory embedding j, the content evolves as

$${\bf m}_{j}^{\,i+k}=(1-\lambda)^{k}\,{\bf m}_{j}^{\,i}+\sum_{u=1}^{k}\lambda(1-\lambda)^{k-u}\,{\bf m}_{\mathrm{new}}^{\,i+u},$$
$$(9)$$

new , (9)
where mi+u new denotes the write at update i+u (see Appendix A.1 for a full derivation). Consequently, the coefficient of the initial content mi j after k overwrites is (1 − λ)
k, and the contribution of the write performed τ updates earlier is λ(1 − λ)
τ−1.

Corollary (Half-life). The number of overwrites k0.5 after which the contribution of mi jhalves is k0.5 =
ln(1/2)
ln(1−λ) =ln 2
− ln(1−λ) ∼ln 2 λ
, as λ → 0. Thus, smaller λ extends retention, while larger λ accelerates overwriting. Effective horizon in environment steps. Since only one memory embedding is updated per segment of length L, a memory is overwritten once every M segments in expectation. The *effective* retention horizon H(ϵ) thus quantifies how many environment steps a stored contribution remains influential before its weight decays below a negligible threshold ϵ, i.e., H(ϵ) = M · L ·ln(ϵ)
ln(1−λ)
. In particular, the half-life in environment steps is H0.5 = M ·L·ln 2
− *ln(1*−λ) ∼ M ·L·
ln 2 λ
, as λ → 0.

Unlike models where all memory is updated at every step (like RNNs), ELMUR's LRU policy ensures (i) memory embeddings not selected for overwrite retain their content exactly until replacement, and (ii) once selected, their contributions decay exponentially with rate λ. This produces a retention horizon that scales linearly with both the number of memory embeddings M and the segment length L, providing a conservative lower bound. In practice, effective horizons are often much longer (Figure 3). Proposition 2 (Memory Boundedness). A natural question is whether repeated convex updates could cause memory values to grow without limit. We show that, under standard bounded-input assumptions, the norm of every memory embedding remains uniformly bounded throughout training and inference. Suppose that every new write is norm-bounded, ∥mt new∥ ≤ C for some constant C > 0, and the initial memory satisfies ∥m0 j∥ ≤ C. Then for all segments i and slots j, it holds that
∥mi j
∥ ≤ C. Since each update is a convex combination of the previous and a bounded new values, the memory embedding always remains inside the closed ball of radius C. This guarantees stability of activations even across arbitrarily long trajectories. See Appendix A.2 for the detailed proof.

## 5 Experiments

We evaluate ELMUR on synthetic (Ni et al., 2023) tasks, 48 POPGym puzzle/control tasks (Morad et al., 2023a), and robotic manipulation (Cherepanov et al., 2026a), all designed to test memory under partial observability. Our study is guided by the following research questions (RQs):
1. RQ1: Does ELMUR retain information across horizons far beyond its attention window? 2. RQ2: How well does ELMUR generalize to shorter and longer sequences? 3. RQ3: Is ELMUR effective on manipulation tasks with visual observations? 4. RQ4: How consistent is ELMUR across puzzles, control, and robotics tasks? 5. RQ5: What is the impact of components of ELMUR on its memorization?

## 5.1 Benchmarks And Baselines

We evaluate ELMUR on three benchmarks designed to isolate memory (Appendix, Figure 7). The T-Maze requires recalling an early cue after traversing a long corridor with sparse rewards. The MIKASA-Robo suite provides robotic tabletop tasks with RGB observations and continuous actions, including color-recall (RememberColor) and delayed reversal (TakeItBack). Finally, POPGym offers a diverse collection of partially observable puzzles and control environments for evaluating general memory use. Detailed descriptions can be found in Appendix A.4.

We compare against baselines spanning sequence models and offline RL for long-horizon tasks.

We include transformers - Decision Transformer (DT) (Chen et al., 2021) and Recurrent Action Transformer with Memory (RATE) (Cherepanov et al., 2026c) - as representative architectures for memory-augmented policy learning. We also evaluate DMamba (Ota, 2024), a state-space model with efficient recurrence, as a recent alternative to attention. For IL/offline RL, we use Behavior Cloning (BC) via MLP as the simplest supervised baseline, Conservative Q-Learning (CQL) (Kumar et al., 2020) as a strong offline RL method, and Diffusion Policy (DP) (Chi et al., 2023) as a stateof-the-art generative policy. Together, these span transformer, state-space, and offline/generative approaches, providing a competitive reference set for evaluation. We do not compare with online RL baselines, since they assume interactive data collection with exploration, yielding incomparable training budgets. Likewise, we omit real-robot experiments to avoid confounds such as latency, resets, and safety constraints, focusing instead on controlled, reproducible studies. Experimental Setup. For RQ1, we test T-Maze cue retention by training with short contexts
(L=10, S=3) and evaluating on corridors up to 106steps. For RQ2, we train on 7 T-Maze lengths distributions (9–900) and validate on 11 shorter/longer ones (9–9600) to assess interpolation and extrapolation. For RQ3, we use MIKASA-Robo tasks, training by imitation from expert demonstrations and evaluating zero-shot. For RQ4, we compare T-Maze, POPGym-48, and MIKASA-Robo to test robustness across synthetic puzzles, control, and robotics. For RQ5, we ablate RememberColor3-v0, varying M, λ, σ, and (*L, S*), and remove relative bias, LRU, and per-layer memory to measure component contributions. Evaluation Protocol. Unless stated otherwise, each model is trained with three (four for T-Maze) independent runs (different initialization). For each run we evaluate on 100 episodes with distinct environment seeds and compute the run mean. We then report the grand mean ± standard error of the mean (SEM) across the three run means. For per-task leaderboards (e.g., POPGym-48) we apply this protocol per task and aggregate as specified in the benchmark. Training Details and Hardware. All models are trained from scratch under the same data budgets and preprocessing. We use segment-level recurrence with detached memory between segments; losses are applied on each processed segment. Optimizers, schedulers, and hyperparameters follow the task-specific configuration table in Appendix, Table 7. All experiments were run on a single NVIDIA A100 (80 GB) per job. Training/evaluation code paths, seeds, and environment versions are fixed across methods for reproducibility.

![7_image_0.png](7_image_0.png)

## 5.2 Results

We evaluate ELMUR on T-Maze, MIKASA- Robo, and POPGym, addressing RQ1–RQ5 on retention, generalization, manipulation, crossdomain robustness, and ablations.

![7_image_1.png](7_image_1.png) 
Figure 4: **Generalization of ELMUR across** T-Maze lengths. Each cell shows success rate
(mean ± standard error) for training vs. validation lengths. ELMUR transfers perfectly: models trained on shorter sequences retain 100% success up to 9600 steps. Training lengths were split into three equal segments.

RQ1: Retention beyond attention. To test memory retention, we train on T-Maze corridors of length T while restricting the context size to *L < T*, forcing the model to solve tasks where the cue must be preserved beyond the native attention span. At validation, we evaluate on much longer corridors - up to one million steps - without increasing L, thereby probing memory retention far beyond the training horizon. ELMUR achieves 100% success even under this extreme extrapolation (Figure 3), implying retention horizons nearly 100,000× larger than the attention window (L=10 with only S=3 segments used during training). RQ2: Generalization across sequence lengths. We train ELMUR on T-Maze with short contexts (3 to 300 steps) and then evaluate across 11 validation lengths ranging from 9 to 9600 steps. The model transfers seamlessly in both directions: it solves tasks shorter than those seen during training without overfitting to a fixed scale, and it also extrapolates to sequences orders of magnitude longer.

As shown in Figure 4, ELMUR maintains 100% success across all train/test pairs, demonstrating robust generalization beyond the training horizon. RQ3: Manipulation with visual observations. Results in Table 1 indicate that EL- MUR achieves higher success rates than other baselines on the MIKASA-Robo tasks. In

Table 1: Success rates (mean ± standard error) on MIKASA-Robo tasks, averaged over 3 runs with 100 evaluation seeds. ELMUR outperforms baselines, showing stronger memory in manipulation. See results for all 32 MIKASA-Robo tasks in Appendix, Table 8

.

Task RATE DT BC-MLP CQL-MLP DP ELMUR (ours) RememberColor3-v0 0.65±0.04 0.01±0.01 0.27±0.03 0.29±0.01 0.32±0.01 0.89±**0.07** RememberColor5-v0 0.13±0.03 0.07±0.05 0.12±0.01 0.15±0.02 0.10±0.02 0.19±**0.03** RememberColor9-v0 0.09±0.02 0.01±0.01 0.12±0.02 0.15±0.01 0.17±0.01 0.23±**0.02** TakeItBack-v0 0.42±0.24 0.08±0.04 0.33±0.10 0.04±0.01 0.05±0.02 0.78±**0.03**

TakeItBack-v0, it obtains 0.78±0.03 compared to 0.42±0.24 for the next-best model, and in RememberColor[3,5,9]-v0 its performance remains stable as the number of distractors increases. Overall, ELMUR shows more reliable performance under visual interference in manipulation tasks with pixel inputs.

| Table 2: Aggregated returns on 48 POPGym tasks. RATE DT Rand. BC-MLP BC-LSTM ELMUR All (48) 9.5 5.8 -12.2 -6.8 9.0 10.4 Puzzle (33) 0.45 -3.5 -14.6 -11.9 -0.2 1.2 Reactive (15) 9.1 9.3 2.3 5.1 9.1 9.2   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

RQ4: Robustness across domains. Across synthetic (T-Maze), control/puzzle (48 POPGym),
and robotic (MIKASA-Robo) benchmarks, ELMUR consistently outperforms baselines, generalizing across diverse modalities, actions, and rewards. On POPGym, it achieves the best overall score (10.4), with the largest gains on memory puzzles (1.2 vs. 0.45 for RATE; DT and BC-LSTM score below zero), showing the importance of explicit memory for long-term dependencies. On reactive tasks, ELMUR stays competitive without sacrificing puzzle performance, ranking first on 24 of 48 tasks (full results in Table 5). Figure 5 shows consistent per-task gains over DT, especially on memory-intensive puzzles. Improved retention comes with little overhead: on T-Maze, ELMUR has 2.1M parameters (vs. 1.7M for RATE, 1.8M for DT) yet runs faster per step (6.8±0.5 ms) than RATE (7.2±0.3 ms) and DT (10.7±0.1 ms). Efficiency stems from (i) a short attention window with longterm context handled by bounded memory, so complexity depends on memory size not sequence length, and (ii) MoE feed-forward layers, which raise capacity without proportional compute. Thus, explicit memory is both effective and efficient for long-horizon RL.

Table 3: Ablation study results.

RQ5: Ablation Study. We ablate ELMUR's memory design on RememberColor3-v0 (Figure 6, Table 3). Unless noted, models use *per-layer* memory, relative-bias token–memory crossattention, and LRU-based updates; *shared memory* denotes embeddings shared across layers. In Figure 6 (b–d) the LRU factor is fixed to λ = 0 to isolate other effects. Results average three runs of 20 episodes. Performance scales with memory size M: when M ≥ N (the number of segments needed), success is nearperfect; when *M < N*, accuracy drops sharply, especially near M ≈ N (Figure 6, c–d). Intermediate blending (λ ≈ 0.4–0.6) is unstable (Figure 6, a), while larger initialization σ mitigates collapse (Figure 6, b). Finer recurrence (shorter segments, larger N) stresses capacity unless M scales accordingly. Component ablations confirm that capacity and LRU dominate. Removing LRU leaves stale entries, and removing both LRU and relative bias prevents effective retrieval. Relative bias gives modest gains, while shared memory degrades performance, underscoring the value of layer-local design. Finally, replacing MoE-FFN with MLP-FFN preserves accuracy while improving computational efficiency. To confirm that memory mechanisms do not harm performance on fully observable MDPs, we evaluated all models on the simple control task CartPole-v1 (Towers et al., 2024). ELMUR, RATE,
RMT, TrXL, BC-MLP, BC-LSTM, and CQL all achieved the maximum return of 500 ± 0, showing that adding memory does not break performance in standard MDP settings. See additional results on more competitive D4RL benchmark (Todorov et al., 2012) with MDP tasks in the Appendix, Section A.5, Table 4.

Setting Score Baseline ELMUR 1.00 ± 0.00 Shared memory 0.45 ± 0.03 No rel. bias 0.95 ± 0.05 No LRU 0.43 ± 0.22 No rel. bias; No LRU 0.22 ± 0.11 MoE → MLP 1.00 ± 0.00

![8_image_0.png](8_image_0.png) 
Figure 5: ELMUR compared to DT on all 48 POPGym tasks. Each model was trained with three independent runs, validated over 100 episodes each. Bars show the mean performance with 95% confidence intervals computed over these three means. ELMUR achieves consistent improvements over DT, with the largest gains on memory-intensive puzzles.

![9_image_0.png](9_image_0.png)

## 6 Related Work

Manipulation. Transformer approaches to robotic manipulation can be broadly categorized by their underlying design principles. *Perception-centric visuomotor transformers* focus on multi-view or 3D perception to improve near fully observable control (Shridhar et al., 2023; Goyal et al., 2024). Sequence/skill modeling distills demonstrations into reusable action chunks but remains bottlenecked by limited context (Huang et al., 2023; Kobayashi et al., 2025). Planning/value-augmented transformers integrate transformers with planning or value learning for closed-loop control under finite context (Zhang et al., 2025b; Hu et al., 2025). *Alternative backbones* adopt state-space models or diffusion for efficiency, but without persistent memory (Liu et al., 2024b; Chi et al., 2023). Scaling to VLA broadens task coverage with language but still suffers from fixed horizons, with some remedies via summarization, feature banks, or hierarchy (Zitkovich et al., 2023; Team et al., 2024; Kim et al., 2024; Fang et al., 2025; Shi et al., 2025). ELMUR differs by training as a standard IL transformer while removing the context bottleneck through structured, layer-local external memory. Memory. Efforts to extend sequence models to long horizons take several forms. Implicit recurrence and state-space models compress history in hidden dynamics, offering efficiency but little control over forgetting (Beck et al., 2024; Gu & Dao, 2023). External memory with learned access provides addressable storage but complicates optimization (Graves et al., 2016; Santoro et al., 2016). *Transformer context extension* retains history via caches or auxiliary slots but keeps memory peripheral (Dai et al., 2019). In RL, memory is often implemented through *episodic buffers* for salient events (Lampinen et al., 2021) or *sequence-model adaptations* that retrofit transformers for recurrence (Parisotto et al., 2020; Cherepanov et al., 2026b). Architectures vary in integration: RATE (Cherepanov et al., 2026c) concatenates memory with tokens, Memformer (Wu et al., 2020) uses global slots, and Block-Recurrent Transformers (Hutchins et al., 2022) recycle hidden states.

ELMUR instead gives each layer an external memory with dedicated mem2tok/tok2mem crossattention and LRU updates, yielding bounded memory for long-horizon tasks (Appendix A.6).

## 7 Conclusion

We introduced ELMUR, a transformer architecture with layer-local external memory, bidirectional token–memory cross-attention, and an LRU-based update rule. Unlike prior methods, ELMUR
integrates explicit memory into every layer, achieving retention horizons up to 100,000× beyond the native attention window. Our analysis establishes formal guarantees on half-life and boundedness under convex blending, and experiments on T-Maze, 48 POPGym tasks, and MIKASA-Robo demonstrate consistent improvements over strong baselines, underscoring reliable credit assignment under partial observability. We envision ELMUR as a simple and extensible framework for longhorizon decision-making with scalable memory in sequential control.

## Reproducibility Statement

We have taken several measures to ensure the reproducibility of our results. **Model details:** A complete description of the ELMUR architecture, including pseudocode for the layer update and the LRU-based memory module, is provided in Section 3, Algorithm 1, and Figure 2. Theoretical results: All assumptions and formal proofs of our propositions on exponential forgetting, half-life, and boundedness are presented in Section 4 and detailed in Appendix A.1–A.2. **Experimental setup:** Benchmarks, training procedures, and evaluation protocols are described in Section 5, with additional specifications (hyperparameters, dataset preprocessing, random seeds, and hardware setup) reported in Appendix 7 and A.4. **Baselines:** All baselines are implemented from open-source libraries or faithfully re-implemented with hyperparameters matched to their original publications, as described in Section 5 and Appendix A.6. **Code and data:** An anonymous repository with the implementation of ELMUR, training scripts, and configuration files is provided in the supplementary material. Together, these resources enable full replication of both our theoretical analysis and empirical findings.

## References

Herve Abdi and Lynne J Williams. Principal component analysis. ´ Wiley interdisciplinary reviews:
computational statistics, 2(4):433–459, 2010.

Amir Hosein Khas Ahmadi. *Memory-based graph networks*. University of Toronto (Canada), 2020. Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, ¨
Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended ¨ long short-term memory. *Advances in Neural Information Processing Systems*, 37:107547– 107603, 2024.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. *arXiv* preprint arXiv:2501.00663, 2024.

Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, ˜
Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. *arXiv preprint arXiv:2503.14734*, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi 0: A vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. *arXiv preprint arXiv:2212.06817*, 2022.

Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:11079–11091, 2022.

Yevgen Chebotar, Quan Vuong, Karol Hausman, Fei Xia, Yao Lu, Alex Irpan, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, et al. Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions. In *Conference on Robot Learning*, pp. 3909–3928. PMLR, 2023.

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. *Advances in neural information processing systems*, 34:15084–15097, 2021.

Egor Cherepanov, Nikita Kachaev, Alexey Kovalev, and Aleksandr Panov. Memory, benchmark &
robots: A benchmark for solving complex tasks with reinforcement learning. In *The Fourteenth* International Conference on Learning Representations, 2026a. URL https://openreview. net/forum?id=9cLPurIZMj.

Egor Cherepanov, Nikita Kachaev, Artem Zholus, Alexey Kovalev, and Aleksandr Panov. Unraveling the complexity of memory in RL agents: an approach for classification and evaluation. In *The Fourteenth International Conference on Learning Representations*, 2026b. URL https://openreview.net/forum?id=lJKdOYFF5W.

Egor Cherepanov, Aleksei Staroverov, Alexey Kovalev, and Aleksandr Panov. Recurrent action transformer with memory. In The Fourteenth International Conference on Learning Representations, 2026c. URL https://openreview.net/forum?id=kByN4v0M3e.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023.

Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

Murtaza Dalal, Ajay Mandlekar, Caelan Garrett, Ankur Handa, Ruslan Salakhutdinov, and Dieter Fox. Imitating task and motion planning with visuomotor transformers. arXiv preprint arXiv:2305.16309, 2023.

Siyu Ding, Junyuan Shang, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Erniedoc: A retrospective long-document modeling transformer. *arXiv preprint arXiv:2012.15688*, 2020.

Kevin Esslinger, Robert Platt, and Christopher Amato. Deep transformer q-networks for partially observable reinforcement learning. *arXiv preprint arXiv:2206.01078*, 2022.

Haoquan Fang, Markus Grotz, Wilbert Pumacay, Yi Ru Wang, Dieter Fox, Ranjay Krishna, and Jiafei Duan. Sam2act: Integrating visual foundation model with a memory architecture for robotic manipulation. *arXiv preprint arXiv:2501.18564*, 2025.

Niklas Funk, Julen Urain, Joao Carvalho, Vignesh Prasad, Georgia Chalvatzaki, and Jan Peters.

Actionflow: Equivariant, accurate, and efficient policies with spatially symmetric flow matching. arXiv preprint arXiv:2409.04576, 2024.

Kai Gao, Fan Wang, Erica Aduh, Dylan Randle, and Jane Shi. Must: Multi-head skill transformer for long-horizon dexterous manipulation with skill progress. *arXiv preprint arXiv:2502.02753*, 2025.

Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, and Dieter Fox. Rvt: Robotic view transformer for 3d object manipulation. In *Conference on Robot Learning*, pp. 694–710. PMLR, 2023.

Ankit Goyal, Valts Blukis, Jie Xu, Yijie Guo, Yu-Wei Chao, and Dieter Fox. Rvt-2: Learning precise manipulation from few demonstrations. *arXiv preprint arXiv:2406.08545*, 2024.

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.

Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-
Barwinska, Sergio Gomez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John P. Agapiou, Adria Puigdom ` enech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, ` Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. *Nature*, 538:471–476, 2016. URL https://api.semanticscholar.org/CorpusID:205251479.

Jake Grigsby, Justin Sasek, Samyak Parajuli, Daniel Adebi, Amy Zhang, and Yuke Zhu. Amago-2:
Breaking the multi-task barrier in meta-reinforcement learning with transformers. *Advances in* Neural Information Processing Systems, 37:87473–87508, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´
state spaces. *arXiv preprint arXiv:2111.00396*, 2021.

ByungOk Han, Jaehong Kim, and Jinhyeok Jang. A dual process vla: Efficient robotic manipulation leveraging vlm. *arXiv preprint arXiv:2410.15549*, 2024.