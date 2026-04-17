# Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping for Deeper Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has shown significant promise for enhancing the reasoning capabilities of large language models (LLMs). However, prevailing algorithms like GRPO broadcast a uniform advantage signal across all tokens in a sequence. This coarse-grained approach overlooks the pivotal role of uncertain, high-stakes decisions during reasoning, leading to inefficient exploration and the well-documented problem of entropy collapse. To address this, we introduce $\textbf{U}$n$\textbf{C}$ertainty-aware $\textbf{A}$dvantage $\textbf{S}$haping ($\textbf{UCAS}$), a model-free method that refines credit assignment by leveraging the model's internal uncertainty signals. UCAS operates in two stages: it first modulates the response-level advantage using the model's overall self-confidence, and then applies a token-level penalty based on raw logit certainty. This dual mechanism encourages exploration of high-uncertainty paths that yield correct answers while penalizing overconfident yet erroneous reasoning, effectively balancing the exploration-exploitation trade-off. Extensive experiments on five mathematical reasoning benchmarks show that UCAS significantly outperforms strong RLVR baselines across multiple model scales, including 1.5B and 7B. Our analysis confirms that UCAS not only achieves higher rewards but also promotes greater reasoning diversity and successfully mitigates entropy collapse.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UCAS (UnCertainty-Aware Advantage Shaping), a method to improve Reinforcement Learning with Verifiable Rewards (RLVR) for large language models (LLMs). The key idea is to shape the advantage signal using model uncertainty: (1) modulating the sequence-level advantage by the model’s self-confidence, and (2) penalizing token-level certainty derived from logits. The goal is to prevent entropy collapse and encourage exploration. The authors report improvements over GRPO and DAPO on several mathematical reasoning benchmarks (AIME24, MATH-500, AMC, Minerva, OlympiadBench) using Qwen2.5-Math-1.5B and 7B.

### Strengths
1. The paper tackles a real problem in RLVR: entropy collapse and insufficient exploration.

2. The proposed uncertainty-aware shaping is conceptually simple, easy to implement, and compatible with existing RLVR frameworks.

3. The presentation is clear, and the motivation is easy to follow.

### Weaknesses
###  Limited Novelty and Theoretical Justification

The core contributions lack sufficient novelty. The paper essentially applies existing uncertainty quantification techniques (self-confidence from prior work, raw logit values) to weight advantages differently. The two-stage mechanism is straightforward:

- Response-level: exponential weighting based on normalized confidence (Eq. 7)
- Token-level: min-max normalized logit penalty (Eq. 8)

Neither component introduces fundamentally new concepts. The exponential weighting scheme is ad-hoc without theoretical grounding for why this specific functional form is optimal. Why exponential rather than linear, polynomial, or other monotonic functions? The paper provides no principled justification beyond empirical performance.

### Weak conceptual contribution.

The paper presents UCAS as a new framework, but it does not introduce any fundamentally new algorithmic principle beyond applying confidence-based scaling to the GRPO advantage. Similar uncertainty-aware or entropy-regularized approaches already exist (e.g., semantic entropy regularization, variance-aware advantage estimation, or KTAE). The novelty over prior work is therefore marginal.

# Writing and Presentation Issues

The paper oversells the contribution. Terms like "unlocking exploration" and "deeper reasoning" are not substantiated by the actual improvements shown

The "entropy collapse" narrative is emphasized throughout, but Figure 3 shows UCAS entropy actually drops initially before recovering—this deserves more analysis

Some claims lack support: "encourages exploration of high-uncertainty paths that yield correct answers" (lines 25-26)—but the method equally amplifies penalties for wrong answers with high uncertainty

### Questions
How do results change with different α and β values?

Why does entropy initially drop before recovering (Figure 3)?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitations of prevailing algorithms such as GRPO in RLVR, which broadcast a uniform advantage signal across all tokens in a sequence. This coarse-grained approach overlooks the pivotal role of uncertain, high-stakes decisions during reasoning, leading to inefficient exploration and the well-documented problem of entropy collapse. To tackle this, the authors introduce UnCertainty-aware Advantage Shaping (UCAS), a model-free method that refines credit assignment by leveraging the model’s internal uncertainty signals. UCAS operates in two stages: it first modulates the response-level advantage using the model’s overall self-confidence, and then applies a token-level penalty based on raw logit certainty.

### Strengths
1.  The topic of this paper is important, as it addresses both the exploration-exploitation trade-off in LLM reasoning and the problem that GRPO broadcast a uniform advantage signal across all tokens in a sequence. 
2.  The authors conduct comparisons with up-to-date baselines and report various metrics, including pass@k.
3.  The proposed method is intuitive and makes sense to me.

### Weaknesses
1.  The proposed method is mostly based on heuristics; more theoretical analysis should be included.
2.  The idea of response-level and token-level advantage shaping using uncertainty/confidence has already been proposed in works such as Seed-GRPO and Entropy Advantage shaping. I encourage the authors to elaborate more on the unique contributions of their work.
3.  The proposed method may work for GRPO, where the advantage is uniform for all tokens in a trajectory. However, it is unclear how it would apply to PPO.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes UCAS, an uncertainty aware advantage shaping method for RLVR training of reasoning LLMs. The idea is to reshape the learning signal in two stages. First, at the response level, the method scales the group normalized advantage by a self confidence score computed as average KL to uniform. Second, at the token level, it subtracts a certainty penalty based on the raw logits for the chosen tokens. Experiments on five math benchmarks and two model sizes show consistent pass@1 gains over GRPO, DAPO, and other recent RLVR variants, with longer reasoning chains and recovery of generation entropy as training proceeds.

### Strengths
1. Uncertainty aware shaping that mixes a response level self confidence weight with a token level logit penalty is simple, well motivated, and easy to implement inside existing GRPO or DAPO code. It does not add a new model or a verifier, unlike many PRM based approaches. The method is explained clearly with a compact formula and an algorithm box.
2. On AIME24, MATH 500, AMC, Minerva, and OlympiadBench, UCAS wins on both 1.5B and 7B Qwen math models, with gains over DAPO, KTAE, and Oat Zero. The ablation in Table 2 shows both response level and token level parts contribute, and their combination is best.
3. The work tackles the widely reported entropy collapse in RLVR and shows recovery of generation entropy during training together with longer responses.

### Weaknesses
1. All experiments are in math with verifiable final answers. It is unclear if the same shaping works for code unit tests or symbolic tasks, and especially for non binary or dense reward settings.
2. The response level confidence is KL to uniform and the token level proxy is raw logits. Both are known to be imperfect confidence measures and can be miscalibrated.
3. The training uses 16 rollouts per prompt and drops KL and entropy regularizers in some baselines. The paper should include a sensitivity study for alpha and beta, and report results when baselines are tuned with their recommended regularization settings. Otherwise, part of the gain may come from different regularization or longer responses.
4. KTAE [1] also produces token level advantages without extra models. Recent entropy induced advantage methods also reshape the advantage. The novelty margin would be clearer with side by side plots of entropy and pass@k against those methods under the same compute. [1] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning

### Questions
See weaknesses.

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
This paper proposes UCAS, a method to improve RLVR for reasoning LLMs by addressing the coarse credit assignment problem in GRPO. Instead of broadcasting a uniform advantage across all tokens, UCAS reshapes the learning signal using the model’s intrinsic uncertainty at two levels: (1) a response-level modulation that amplifies rewards for correct but uncertain trajectories and penalizes overconfident wrong ones, and (2) a token-level certainty penalty that discourages local overconfidence to prevent entropy collapse. Experiments on five mathematical reasoning benchmarks show consistent and significant improvements across both 1.5B and 7B Qwen-Math models, leading to deeper reasoning chains, higher rewards, and better exploration diversity.

### Strengths
1. The paper is easy to read and well structured, making the method and motivation intuitive to follow. Improving exploration in RL for reasoning LLMs is a very active and timely topic, and this method provides a valuable and well-motivated contribution in that direction.

2. The method is lightweight, model-free, and compatible with existing RLVR pipelines, which makes it practically useful for scaling to larger models.

3. The experiments are comprehensive, comparing against a wide range of strong RLVR and reasoning baselines with clear and consistent performance gains.

### Weaknesses
1. The theoretical foundation of UCAS is mostly intuitive and the paper doesn’t formally analyze why the two-stage shaping leads to more stable optimization or guarantees improved exploration.

2. From Table 2, the token-level certainty component seems to contribute little or even slightly hurt performance in some cases, suggesting its effect is weaker or less stable than the response-level shaping.

3. The paper lacks ablation or sensitivity analysis for the two key hyperparameters \alpha and \beta, which directly control the strength of response-level modulation and token-level penalty. It is unclear how stable UCAS is to different settings of these values.

### Questions
1. How do you choose the hypeparameters and how they affect the performance?

2. Since UCAS is conceptually orthogonal to the underlying policy optimization algorithm, have the authors tested it with other GRPO-family methods (e.g., DAPO or GSPO)? Demonstrating consistent gains across multiple RLVR optimizers would strengthen the claim that UCAS provides generalizable advantage shaping rather than optimizer-specific benefits.

### Soundness
4

### Presentation
4

### Contribution
3
