# Predicting the Order of Upcoming Tokens Improves Language Modeling

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
Multi-Token Prediction (MTP) has been proposed as an auxiliary objective to improve next-token prediction (NTP) in language model training but shows inconsistent improvements, underperforming in standard NLP benchmarks. We found MTP's exact future token prediction to be too difficult as an auxiliary loss. Instead, we propose Token Order Prediction (TOP), which trains models to order upcoming tokens by their proximity using a learning-to-rank loss. TOP requires only a single additional unembedding layer compared to MTP's multiple transformer layers. We pretrain models of 340M, 1.8B, and 7B parameters using NTP, MTP, DeepSeek MTP and TOP objectives. The results of eight standard NLP benchmarks show that TOP overall outperforms NTP, MTP, and DeepSeek MTP even at scale. On the synthetic star graph task, TOP enables pathfinding on graphs where NTP and MTP fail.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces TOP, a new auxiliary training objective for language models that aims to enhance next-token prediction. TOP works by training models to rank upcoming tokens based on their proximity in the sequence. Unlike MTP, TOP uses only a single extra unembedding layer and relaxes the prediction task to ordering within a window. The authors pretrain transformer models of 340M, 1.8B, and 7B parameters on FineWeb-Edu using NTP, MTP, and TOP, evaluating them on eight standard NLP benchmarks and a synthetic star graph pathfinding task. Results show TOP outperforming NTP and MTP overall, especially at larger scales.

### Strengths
1. The idea of replacing exact multi-token prediction with a ranking-based objective is interesting and well-motivated. Compared to MTP, TOP's task relaxation to ordering makes it a more scalable auxiliary loss. 
2. The TOP achieves consistent empirical gains, where experiments across eight diverse benchmarks and three model sizes are performed, and TOP improves over the NTP and MTP baselines.
3. TOP adds just one unembedding layer compared to MTP's multiple transformer heads, and scales better than MTP on non-coding tasks.

### Weaknesses
1. First, the presentation of this paper needs to be improved. In general, this paper introduces an intuitively effective and simple idea, i.e., using token order prediction to replace the much harder exact multiple token prediction. But the current draft makes it difficult to follow. For example, Fig 1 can be improved to illustrate how the proposed TOP works; Section 4 Method, instead of pure psudocode, the authors should also explain the working process of TOP; Figure 2, what do different color curves mean? I guess they indicate different positions, but it is better to be marked on the figure.
2. The empirical evaluation is narrow. The model is trained on limited pre-training budget, where 104B tokens are used for all sizes of models. For example, according to the scaling law, for 7B model, it is roughly 15 % of the compute used for comparable 7B models in the literature, and it is unclear whether TOP is still advantageous after full-scale training (e.g., 1T tokens).
3.  Except for the synthetic star graph, it would be good to provide other planning-intensive benchmarks results, which can better validate the task generalizability of TOP.

### Questions
1. Why was the window size set eqaul to the sequence length? How does downstream performance and training cost vary when changing the window size $W$?”
2. For the star graph task, MTP-4 succeeds on easier graphs but fails on harder ones. How about trying MTP with more heads?
3. What does the second row "NTP Loss" mean in Table 2? I think each column corresponds to NTP, MTP, and TOP loss respectively.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Token Order Prediction (TOP), a novel auxiliary training objective for autoregressive language models that replaces the exact multi-token prediction (MTP) with a learning-to-rank formulation. TOP trains the model to rank all vocabulary tokens by their proximity to the current position within a configurable window. The authors implement TOP using a single additional linear layer and a ListNet-style ranking loss, making it far more parameter-efficient than MTP.

### Strengths
This paper is well structured and highly motivated. Replacing token identity prediction with ordinal proximity is a clever relaxation that retains lookahead signal while reducing optimization hardness.

TOP requires only one extra linear layer, in contrast to MTP's per-token transformer heads.

The evaluation is comprehensive, spanning multiple model sizes, diverse NLP benchmarks.

### Weaknesses
The observation that TOP achieves higher NTP training loss yet better downstream performance is interesting but underexplored. The authors hypothesize regularization but provide no ablation (e.g., varying TOP loss weight, early stopping comparisons) to confirm this.

The TOP target assigns scores to all vocabulary tokens, most of which do not appear in the window. This may create a highly sparse and noisy supervision signal.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Token Order Prediction (TOP) as a new auxiliary loss for large language model pretraining. Instead of predicting the exact identities of future tokens as in Multi-Token Prediction (MTP), TOP trains the model to rank future tokens by proximity using a listwise ranking loss (ListNet). The authors train models of 340M, 1.8B, and 7B parameters with NTP, MTP, and TOP objectives and evaluate them on eight NLP benchmarks as well as a synthetic star-graph pathfinding task. TOP reportedly outperforms both NTP and MTP while being simpler and more parameter-efficient.

### Strengths
1) Interesting reformulation of auxiliary objectives — The idea of relaxing MTP’s difficult prediction target into a ranking-based proximity task is conceptually appealing.

2) Simple and lightweight implementation — TOP only requires an additional unembedding layer and is compatible with existing transformer architectures.

3) Broad empirical evaluation — The experiments cover multiple model scales and both standard and synthetic benchmarks.

### Weaknesses
W1) **Figure 2 lacks clarity**

The figure is supposed to show that predicting farther tokens is harder, but there’s no legend or label for each position (t+1, t+2, …).
It’s unclear which curve corresponds to which distance, so the trend the authors claim isn’t visually evident.

A similar plot for the TOP objective would also help illustrate whether it really leads to smoother or easier training.

W2) **The claim that TOP is “easier” isn’t well supported**

The paper keeps describing TOP as an easier task than MTP but never shows evidence—no convergence plots, gradient analysis, or theoretical argument.

Without that, the claim feels more like intuition than proof.

W3) **Hyperparameter setup seems too shallow to trust**

Each model size only lists one fixed set of hyperparameters, with no sign of tuning or validation.
It’s unclear whether these settings are optimal or even comparable across NTP, MTP, and TOP.
This makes it hard to tell if the reported gains come from the new objective itself or just from a configuration that happens to work better for TOP.

W4) **Missing ablations and analysis**

There’s no variation of window size, loss weight, or ranking depth, and no look into how TOP affects learned representations.
Without these, it’s difficult to understand what the model is actually learning from the objective.

### Questions
Q1) **Combining MTP and TOP**
Have you tried using MTP and TOP together or at different training stages?
For example, applying TOP early and MTP later might be complementary.

Q2) **Speculative decoding and improvements**
TOP performs worse than MTP in self-speculative decoding (Table 4).
Is this a limitation of TOP or a mismatch with decoding efficiency?
Any ideas on improving it e.g., refining the ranking loss or combining TOP signals at inference?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Token Order Prediction (TOP), an auxiliary training objective for LM that predicts the relative order and proximity of upcoming tokens rather than their exact values. The key insight is that Multi-Token Prediction (MTP), which attempts to predict exact future tokens, is too difficult. TOP instead uses a learning-to-rank loss where the target is a proximity-weighted bag-of-words vector: tokens appearing at distance d within a window W receive scores of W-d, with farther tokens scored as 0. This creates a soft probability distribution emphasizing near-future tokens. Architecturally, TOP adds only a single unembedding head (vs. MTP's multiple transformer layers). Experiments on 340M, 1.8B, and 7B parameter models show TOP consistently outperforms both NTP-only and NTP+MTP baselines across 8 standard NLP benchmarks, with gains increasing at larger scales. On a synthetic star graph pathfinding task designed to test look-ahead reasoning, TOP achieves 100% accuracy while MTP and NTP models fail on complex configurations. The paper also evaluates self-speculative decoding, where MTP performs better, but concludes TOP is a promising direction for improving LLM pretraining.

### Strengths
1. **Novel approach**: The paper proposes TOP as a theoretically sound middle ground between NTP (too myopic) and MTP (too difficult). The connection to learning-to-rank is elegant.

2. **Strong empirical results on specific tasks**: TOP consistently outperforms both baselines across most tasks, with particularly impressive results on the star graph task (100% accuracy where others fail) demonstrating genuine improvement in look-ahead reasoning capabilities.

### Weaknesses
1. **Missing critical baselines and ablations, while over-emphasizing obvious observations**:
   - Motivation section (page 3): Approximately two-thirds of a page is devoted to explaining Figure 2, which shows that MTP loss increases with prediction distance. This is an intuitive and expected result that does not warrant such extensive discussion.
   - MTP baseline incomplete: The paper only compares against Meta's MTP variant (multiple linear heads for MTP) but ignores DeepSeek's MTP architecture (sequential auxiliary transformer layers approach), which has become the dominant MTP design in recent LLM pretraining. This is a significant omission given DeepSeek-V3's prominence.
   - Window size ablation missing: The choice of window size W is a key hyperparameter for TOP, yet no ablation study investigates how performance varies with W. What is the sensitivity? 

2. **Experimental setup raises questions about generalizability**:
   - Insufficient training data: Models are trained on only 100B tokens (a subset of FineWeb-Edu). For context, typical convergence for 7B models requires 1-2T tokens. Since TOP is an auxiliary loss, it's unclear whether the observed benefits persist in fully-converged models or only help in the under-trained regime. The paper should either use the full FineWeb-Edu dataset or explicitly discuss this limitation.
   - Task diversity limited: Table 2 focuses heavily on multiple-choice classification tasks. Given that the MTP usually emphasizes "look-ahead" reasoning, the paper should evaluate on code generation benchmarks (e.g., HumanEval, MBPP), where planning ahead is crucial. The star graph task is synthetic and may not reflect real-world look-ahead benefits.
   - Statistical rigor: Table 2 reports only single-run point estimates without error bars or confidence intervals. Many of the benchmarks (especially QA tasks) are known to have high variance. Including mean ± std across multiple seeds would strengthen the claims. Additionally, the absence of MMLU (a standard robust benchmark) is notable.

3. **Presentation quality issues**:
   - Figure 2: The color legend is missing, making it unclear which line corresponds to which future token prediction (t+1, t+2, etc.).
   - Mathematical notation: Vectors are not typeset in boldface (e.g., y, s should be **y**, **s**), making it difficult to distinguish vectors from scalars throughout the paper.
   - Algorithm 1: Critical symbols lack clear definitions. For example, `next[v]` (line 179) and the output `y` (line 178) are not explained in the caption or main text. The main text simply states "please refer to the pseudocode" without walking through the logic, forcing readers to reverse-engineer the algorithm.

### Questions
Regarding the model parameters comparison between TOP and MTP, a more detailed analysis should be presented. For example, the 7B model needs 4096*32000 extract parameters for TOP. while for MTP, it equivalent to how many additional token predictions?

### Soundness
3

### Presentation
1

### Contribution
2
