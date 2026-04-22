# SILA: Enhancing Long-Context Retrieval Capability of Linear Attention via Selective Ignoring

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Linear attention models have recently emerged as computationally efficient alternatives to Transformers.
Despite competitive performance on general commonsense tasks, they still struggle to match Transformers on long-context retrieval tasks.
In this work, we re-examine linear attention models from the perspective of memory writing.
We propose that enabling linear attention models to learn selective ignoring provides a promising approach to addressing long-context retrieval tasks under fixed memory capacity.
Guided by this principle, we demonstrate how to interpret and intervene in the behavior of linear attention models, thereby revealing the true retrieval capabilities of popular models.
Informed by these observations, we introduce Selective Ignoring Linear Attention (SILA), which incorporates a redesigned memory architecture and a weighted loss training strategy to encourage selective memory writing.
SILA exhibits remarkable long-context retrieval capabilities, achieving 20$\times$ context length extrapolation on the Passkey Retrieval task, and demonstrating superior memory utilization efficiency on the Needle-in-a-Haystack benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SILA, which enhances linear attention retrieval through improvements in both recurrence modeling and training objectives. Its core idea is to allow the model to selectively disregard irrelevant tokens.

The authors begin with an analysis of the NIAH task and establish a more reliable evaluation setting. Motivated by the need for selective memory writing, SILA incorporates several architectural innovations: it decouples the “read” and “write” operations (i.e., read-before-write) and introduces state-dependent gates. In addition, the authors employ a progressive selective weighted loss, based on a pretrained transformer, to improve the efficiency of long-context training.

Experimental results under the Transformer-to-RNN paradigm demonstrate that SILA improves the retrieval capability of linear attention models while maintaining competitive performance on general short-context modeling.

### Strengths
1. The motivation and proposed approach of this work are well-reasoned and coherent. The paper identifies a key deficiency of linear attention models under NIAH-style tasks and demonstrates that this issue can be alleviated through selective ignoring.

2. The analysis of the NIAH benchmark is logical and convincing, ensuring the reliability of the experimental evaluation.

3. The improvements in both modeling and training are essential and directly address the need for selective ignoring.

### Weaknesses
1. Some of the conclusions are not thoroughly or fairly validated. For example, the passkey retrieval extrapolation results of Qwen3 and GatedDelta are not shown; since the transferred Qwen3 checkpoints already support 32k context without ntk scaling, further clarification is needed here. In addition, in Table 5, only SILA is a transferred model, making it difficult to fairly compare the effectiveness of the proposed method with other selective writing approaches such as LongMamba.

2. The paper lacks ablation studies and detailed analyses for the proposed improvements. It remains unclear how each component—such as the training strategy and the two architectural modifications—specifically contributes to retrieval performance or NIAH results.

3. The experimental setting is overly constrained. The model scale (0.6B), training length (1k/4k), and evaluation scenario (Transformer-to-RNN transfer) are all highly specialized. Under such a narrow setup, it is difficult to disentangle how much of the observed gain arises from the proposed method itself rather than from others like parameter increases or tuning. Moreover, there are concerns regarding the scalability of the approach.

### Questions
1. In Table 1, the in-context retrieval capability appears correlated with training data scale. Since Qwen3 was trained on 36T high-quality tokens while other linear models were significantly undertrained, would a Transformer trained on a comparable corpus (e.g., hundreds of billions of tokens) still demonstrate similarly robust NIAH-Word performance?

2. Compared to linear attention baselines such as GatedDeltaNet, how do SILA’s computational and parameter costs change? What are the shapes of $W_{\gamma}$ and $W_{\beta}$? 

3. What is the rationale for the chosen functional form of the forget gate $\phi(x)$ on the negative axis in Eq.10, which seems relatively uncommon?

4. As the memory-dependent gates introduce nonlinearity in the recurrence via the hidden state, does this hurt parallelism or training efficiency, particularly for long-context sequences? From an implementation standpoint, is it necessary to materialize all intermediate states across time steps?

5. Regarding the weighted loss, how were the threshold and scaling factor determined? Were these hyperparameters tuned through systematic sweeps?

6. What are the NIAH results for SILA (standard loss)? Only commonsense reasoning results are reported.

7. To support the claim that the improvement stems from selective ignoring, it would be helpful to visualize and analyze the gate values (${\gamma}_t$ and ${\beta}_t$) and compare them against those of GatedDeltaNet.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper re-frames long‑context retrieval in linear attention models as a memory‑writing problem and argues that strong performance often comes from specialized digit‑token shortcuts rather than general associative recall. The authors show instability in standard NIAH evaluation and recommend sample‑level haystack shuffling and also propose SILA, a linear‑attention variant that decouples recall from writing and introduces memory‑dependent gates.

### Strengths
1. The paper convincingly shows that linear models often “win” on NIAH by preferentially writing digits rather than learning general retrieval. This is an important finding. 
2. Two concrete fixes—sample‑level shuffling and NIAH‑Word—expose the over‑reliance on digits. This yields an evaluation setup other works can adopt immediately.   
3. SILA’s read‑before‑write decoupling  allows using the current token locally without committing it to memory; memory‑dependent gates use retrieved state to decide writing/forgetting.

### Weaknesses
1. Teacher‑dependence & compute overhead not quantified. The weighted‑loss pipeline requires per‑token attention from a reference Transformer. The paper does not report added training FLOPs/throughput or wall‑clock vs. a standard linear‑attention pretrain.  
2. SILA performs an extra memory read for gating/recall each step. Please report inference speed and memory footprint vs. comparable linear baselines. 
3. Unfair comparisons: SILA‑0.6B is initialized from Qwen3‑0.6B and trained on 15B FineWeb‑Edu tokens with bespoke weighting, whereas many baselines are off‑the‑shelf with different data/state sizes.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SILA (Selective Ignoring Linear Attention), an architecture designed to enhance the long-context retrieval ability of linear attention models. SILA introduces (1) a memory-dependent gating mechanism to selectively write information into memory and (2) a weighted-loss scheme to emphasize important tokens during training. Experiments show strong gains in synthetic and benchmark long-context retrieval tasks, demonstrating better generalization to 10–100× longer contexts than training.

### Strengths
- Well-motivated and relevant problem: The paper tackles a key limitation of efficient attention models—poor long-range retrieval—highly relevant to long-context LLM research.

- Clear and technically novel mechanism: The selective ignoring gate and weighted-loss supervision are simple yet effective extensions that yield interpretable and consistent improvements.

- Comprehensive empirical validation: The paper provides clear component-wise ablations and visual analyses showing how the gating improves selective memory use.

### Weaknesses
- Limited scalability to current LLMs: SILA requires replacing the attention mechanism and retraining from scratch, making it impractical for integration into existing large pretrained models (e.g., GPT-series).

- Benchmark and scale limitations: Evaluations are restricted to mid-sized (0.6B) models and synthetic retrieval tasks, lacking validation on realistic large-scale or multi-domain benchmarks.

- Incomplete comparison: The paper does not benchmark against other recent selective-memory or efficient-attention architectures.

### Questions
see weeknesses.

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
2

### Summary
Motivated by the limitations of existing linear attention methods that fail to perform robust long-context retrieval due to the fixed memory capacity, the paper proposes a novel linear attention method that learns to selectively ignore irrelevant tokens for long-context retrieval tasks and attend to important instruction tokens. The proposed method, Selective Ignoring Linear Attention (SILA), shows improvement on needle-in-the-haystack tasks with better memory utilization efficiency compared with prior methods.

### Strengths
1. The paper is motivated by the limitations of existing linear attention methods on robust long-context retrieval tasks, with well-designed controlled experiments, and the proposed method is designed to solve the identified issues of prior methods.

2. The paper conducts extensive experiments and compares the proposed methods over several strong baselines from the literature. Given similar training token sizes and model sizes, models trained with the proposed method outperform strong baselines, such as RWKV7 and Gated DeltaNet, by a large margin, especially on NIAH-word, demonstrating better robustness. 

3. The paper further demonstrates the efficiency of the proposed method compared with other linear attention methods and also shows the general reasoning capabilities of the method, adding empirical strengths to the method.

### Weaknesses
1. The evaluation of long-context retrieval is limited to NIAH and its variants. The proposed method is only compared with baselines on NIAH-1, NIAH-2, and NIAH-Word in Table 2. On the passkey retrieval task, SILA is not compared with any other baseline; on the in-context recall task from MAD-Lab benchmark, SILA is only compared with baselines in the setting using 2-layer shallow models instead of the 0.6 B model. The empirical strength of SILA on long-context retrieval needs to be further validated by comparisons with other methods on more benchmarks, such as MAD, Multi-query associative calls (MQAR), and RegBench. 

2. There lacks analysis of different components of SILA. There is no ablation of the loss design or gate design of SILA, and it is unclear which part of the proposed method contributes most to the empirical improvements on NIAH. 

3. Particularly, there is no analysis of how SILA addresses the memorization issues of other linear attention methods on NIAH: Is there evidence for SILA models not biasing towards digit tokens and performing general retrieval tasks, beyond the results on NIAH-Word? Is there evidence for SILA models correctly attending to the instruction tokens for the strong instruction variants of NIAH?


That being said, I'm willing to adjust the scores if the authors can provide more analysis results, especially the ones mentioned above.

### Questions
1. In Table 2, are the NIAH-1 and NIAH-2 datasets with or without sample-level shuffling, following findings from Figure 1(b)?

### Soundness
3

### Presentation
3

### Contribution
2
