# PosS: Position Specialist Generates Better Draft for Speculative Decoding

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Speculative decoding accelerates Large Language Model (LLM) inference by using a small draft model to predict multiple tokens, and a large target model to verify these tokens in parallel. Recent studies leverage the hidden state of the target model to enhance draft model prediction accuracy. However, existing methods suffer from the degrading quality of draft token predictions at later positions, due to error accumulation in draft model generated features. In this paper, we propose Position Specialists (PosS), which consist of multiple position-specialized draft layers to generate tokens at assigned position(s). Position specialists greatly improve token acceptance rate at later positions per drafting round, as each specialist only needs to focus on handling a certain level of draft model feature deviation. Experiment results on six datasets demonstrate that \textbf{\method} effectively improves over baselines on average acceptance length and speed-up ratio.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses error accumulation in speculative decoding, a phenomenon where the quality of draft tokens progressively degrades at later positions in a sequence. To mitigate this, they introduce Position Specialists (PosS), a novel approach that replaces a single draft model with multiple, position-specialized draft layers. Each specialist is explicitly trained to predict the token for a specific set of positions. Extensive experiments on Llama-2-13B and Llama-3-8B models across six datasets demonstrate that PosS outperforms baseline methods like EAGLE-3, achieving an approximately 10% higher speed-up ratio.

### Strengths
1. Clear Motivation: The paper clearly identifies the degradation of draft quality over sequence length. The proposed solution of using specialized models for different positions is intuitive.
2. Useful Metric: The introduction of the "position-wise acceptance rate" (pos-acc) is a valuable contribution.
3. Strong Baseline: The inclusion of strong baselines like EAGLE-3 and HASS makes the comparison reasonable.

### Weaknesses
1. Missing Comparison to Parameter-Matched Models: The paper fails to compare PosS against a single draft model with an equivalent parameter count (e.g., using an MoE layer, which makes the single draft model become multiple experts, and can also route between different experts dynamically instead of a pre-defined order in PosS). This makes it impossible to determine if the benefits come from "specialization" or simply from increased model capacity.
2. High Deployment Complexity: Maintaining independent KV-Caches for each specialist and switching between them introduces significant overhead. This complexity poses challenges for deployment, especially within modern serving frameworks.
3. Unfavorable Cost-Benefit Ratio: The performance gains are often marginal and inconsistent, as seen in Table-2, the L2-13B model, where PosS (3.34x) offers negligible improvement over HASS (3.33x). Such limited benefits do not justify the substantial increase in system complexity and overhead.

### Questions
1. In the appendix, the ablation studies in Tables 3 and 4 show that PosS-2 almost always outperforms PosS-3. Why were the experiments in the main text conducted with PosS-3 instead?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper propose Position Specialists(PosS), which consist of multiple position-specialized draft layers to generate tokens at assigned position. They also introduce position-wise to measure the conditional probability of accepting the i-th token given the acceptance of its preceding (i-1)-th token. Experimental results show the effectiveness of the proposed method.

### Strengths
1. This paper is technically sound and easy to understand. 
2. The experimental results show the effectiveness of the proposed method.

### Weaknesses
1. What is the difference between PosS and Medusa, which also uses different heads for different token positions? It seems that PosS-1 is exactly the same as Medusa. 
2. Eq.4 seems degenerate to P(A_k),  why don’t you describe Eq.4 like that? 
3. What is the performance when changing the architecture of PosS-1 to Medusa and leaving other things unchanged?

### Questions
See weaknesses above. The difference to Medusa is important.

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
5

### Summary
This paper introduces Position Specialists (PoSS), a novel architectural framework designed to enhance the accuracy and efficiency of speculative decoding for Large Language Models. The core problem addressed is the rapid degradation of draft prediction quality at later positions, which is caused by the accumulation of feature deviation between the draft model's generated features and the target model's features. PoSS resolves this by replacing the single draft layer with multiple position-specialized draft layers, each trained to handle the expected level of feature deviation at its assigned position range.

### Strengths
- The introduction of the position-wise acceptance rate (pos-acc) metric provides a crucial analytical tool for diagnosing and comparing the efficiency of different speculative decoding methods at a granular level.

- The Position Specialists (PoSS) architecture is a novel and intuitive approach that effectively mitigates the fundamental challenge of accumulated feature deviation by distributing the prediction task across multiple specialized draft layers.

### Weaknesses
1. The performance gain achieved by the proposed method is difficult to solely attribute to the "Position Specialists" architecture, as opposed to the "HASS-like" recursive feature alignment training strategy (which PoSS utilizes). Specifically, the incremental improvement of PosS-3(E2) over the HASS baseline is marginal (an average of only 2.1% on L3-8B and 0.3% on L2-13B, based on the speedup ratio from Table 2 at t=0). This small margin suggests that the performance lift might primarily stem from the training approach common to both HASS and PosS.

2. This work shares conceptual similarity with an existing approach[1] which also emphasizes the differentiated treatment of draft tokens at different positions. A clear discussion detailing the differences and experimental results between the proposed POSS method and the aforementioned work is better.

[1] Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding

### Questions
- Following weakness 1, to definitively demonstrate the effectiveness of the proposed Position Specialists architecture, the authors should design an ablation study that isolates its contribution from the HASS training method. A critical experiment would involve training the EAGLE-3 architecture (i.e., the single draft layer) using the same training data and total steps as PosS-3(E3), but without the position specialists. Comparing this result directly against PosS-3(E3) would clearly indicate the incremental value of the PoSS architecture itself. If you solve this core problem, I'm willing to raise my score.

- How is the Key-Value (KV) cache managed under the PoSS framework? Since different specialist layers  are used to predict tokens at different positions, the features generated by these layers are distinct. This prevents the standard sharing and reusing of KV caches that is typical in single-model decoding. Could the authors quantify the potential overhead introduced by managing non-shared KV caches and discuss whether this processing time significantly impacts the overall speedup?

- The authors should provide performance results using industry-standard, high-performance serving frameworks such as vllm or sglang. Providing this data would significantly increase the practical relevance and utility of the proposed method for real-world deployment.

### Soundness
3

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
3

### Summary
This paper introduces a new framework named Position Specialists (PosS) to address the issue of degrading acceptance rates as the speculative step increases. The authors introduce a novel metric "pos-acc" to quantify and analyze this problem. The proposed method demonstrates significant improvements in both average acceptance length and speed-up ratios over strong baselines.

### Strengths
1. The core motivation is clear and compelling. The paper accurately identifies a critical and practical problem in existing methods: the "accumulated feature deviation" and the limited generalization capability of a single draft model. The paper is well-written, with a logical flow and clear exposition.
2. The design of PosS is elegant and intuitive, employing different "specialist" models to handle tasks of varying difficulty (i.e., different levels of feature deviation). The effectiveness of the approach is strongly supported by comprehensive experiments.

### Weaknesses
1. The paper mentions "tree-draft" in Section 6.2 but does not sufficiently clarify its relationship with the experiments. It's unclear whether "draft depth" in Sections 6.2 and 6.3 refers to the depth of the tree or simply the length of a linear draft sequence. The core idea of PosS is to improve the quality of a single draft path. The paper should discuss PosS's orthogonality with parallel verification methods like tree-drafting.
2. Compared to a single draft model, POSS requires training and storing multiple specialist models. While the paper argues in Appendix C that the inference-time memory overhead is acceptable (approx. 218M per specialist), the training process becomes more complex (e.g., requiring initialization from a pre-trained EAGLE model and subsequent staged training). It would strengthen the paper to briefly discuss this trade-off and its impact on training time and computational resources.
3. The specialist allocation strategy (PosS-n) is straightforward, but its optimality is unexplored. A brief discussion on potential alternative or more adaptive allocation strategies would strengthen the paper.

### Questions
1. It accurately identifies a key problem in speculative decoding and proposes an innovative, effective, and well-reasoned solution. The experiments are solid, the analysis is thorough, and the conclusions are well-supported.

### Soundness
3

### Presentation
3

### Contribution
3
