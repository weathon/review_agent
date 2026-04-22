# Latent Reasoning via Sentence Embedding Prediction

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 4, 6, 4, 4

## Abstract
Autoregressive language models (LMs) generate one token at a time, yet human reasoning operates over higher-level abstractions - sentences, propositions, and concepts. This contrast raises a central question- Can LMs likewise learn to reason over structured semantic units rather than raw token sequences? In this work, we investigate whether pretrained LMs can be lifted into such abstract reasoning spaces by building on their learned representations. We present a framework that adapts a pretrained token-level LM to operate in sentence space by autoregressively predicting continuous embeddings of next sentences. We explore two embedding paradigms inspired by classical representation learning: 1) semantic embeddings, learned via autoencoding to preserve surface meaning; and 2) contextual embeddings, trained via next-sentence prediction to encode anticipatory structure. We evaluate both under two inference regimes: Discretized, which decodes each predicted embedding into text before re-encoding; and Continuous, which reasons entirely in embedding space for improved efficiency. Across four domains - mathematics, logic, commonsense, and planning - contextual embeddings under continuous inference show competitive performance with Chain-of-Thought (CoT) while reducing inference-time FLOPs on average by half. We also present early signs of scalability and modular adaptation. Finally, to visualize latent trajectories, we introduce SentenceLens, a diagnostic tool that decodes intermediate model states into interpretable sentences. Together, our results indicate that pretrained LMs can effectively transition to abstract, structured reasoning within latent embedding spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework that elevates pretrained language models from token-level generation to sentence-level reasoning. Instead of predicting tokens, the model autoregressively predicts continuous embeddings of the next sentence, supporting two embedding paradigms—semantic and contextual. The authors further design two inference modes: discretized and continuous. Experiments show that contextual embeddings under continuous inference achieve competitive performance with CoT reasoning while halving inference FLOPs.

### Strengths
1-The paper takes a new step beyond token-level CoT reasoning by framing reasoning as prediction over sentence embeddings.

2-The experiments span mathematical, logical, commonsense, and planning tasks, showing consistent findings and careful analysis of efficiency and robustness.

### Weaknesses
1-Although the idea is intriguing, the improvements over CoT are modest and not statistically analyzed; performance is often close to, but not consistently better than, token-level baselines.

2-All experiments are limited to sub-1B GPT-2 models; the claimed scalability to larger models is only hypothesized.

3-SentenceLens is interesting but lacks quantitative validation (e.g., measuring interpretability gains or correlation with reasoning correctness).

4-The paper contains many technical details and citations but could be clearer.

5-The reported performance gains may largely stem from fine-tuning effects rather than genuine enhancements in reasoning capability. The proposed method’s contribution to reasoning improvement is therefore not convincingly demonstrated.

6-Several comparative algorithms and components are mentioned, yet the paper lacks rigorous ablation or empirical evidence showing their individual effectiveness or contribution to the final results.

7-As stated on page 5, it remains unclear whether the observed reasoning ability truly originates from the latent model itself, or from the pretrained transformer’s inherited capability.

8-While SentenceLens offers interpretability, it requires an additional decoding module and increases computational complexity. Thus, the interpretability improvement comes at the cost of additional model overhead rather than being an inherent property of the latent reasoning framework.

### Questions
1-Can the proposed latent reasoning generalize to open-ended generation tasks (e.g., summarization or dialogue) rather than step-bounded reasoning?

2-How does the approach compare to recent reasoning compression methods?

### Soundness
4

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
3

### Summary
This study addresses the issues of inefficient computation and suboptimal reasoning granularity encountered by LLMs when generating step-by-step reasoning chains on a token-by-token basis. To this end, the authors introduce a novel framework designed to lift the abstract reasoning capabilities of pre-trained language models from the token level to the sentence level. The central concept of this framework is to have the model autoregressively predict the continuous embedding vectors of the next sentence, instead of generating raw tokens. Specifically, the work develops two types of embedding vectors—semantic embeddings and contextual embeddings—and investigates both discretized and continuous inference modes. Experimental findings demonstrate that, under the setting of “contextual embeddings with continuous inference”, the proposed method achieves performance comparable to CoT, while simultaneously reducing the inference computational cost by half.

### Strengths
1. The paper is clearly written, and the figures are well-designed and easy to follow.
2. The attempt to perform auto-regressive decoding at the sentence level is innovative.

### Weaknesses
1. I understand the authors' experimental scope was limited to GPT-2-sized models due to computational constraints. To strengthen the experimental evidence, I suggest the authors broaden their evaluation to include more recent models, such as Llama-3.2-1B-Instruct, Qwen3-0.6B, and Qwen3-1.7B.
2. I am puzzled by the mere 2x speedup achieved for sentence-level auto-regressive decoding. Given the degree of parallelism involved, a more significant gain (e.g., 10x) would be expected intuitively. I would appreciate a detailed explanation or analysis from the authors to clarify why the speedup was only twofold.
3. Could the authors clarify what the bolded figures in Table 2 specifically signify? My interpretation is that they are intended to highlight the best-performing method within each dataset. However, based on the data presented, it appears the best method has been incorrectly marked across all four datasets. The authors should verify and correct these annotations to ensure the table's accuracy.

### Questions
Please see the weakness.

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
3

### Summary
the paper proposes DynaGNN,      a a framework that adapts GNN architectures to handle temporal changes in graph structure through a meta-learning approach with edge-aware attention mechanisms. The key contribution is enabling GNNs to dynamically adjust their parameters based on evolving graph topology without ful  retraining.

### Strengths
I appreciate the comprehensive evaluation across multiple dynamic graph benchmarks and the practical applicability to real-world scenarios like social networks and traffic prediction. The edge-aware attention mechanism is a nice touch that effectively captures local topology changes, and the meta-learning framework provides good theoretical grounding for adaptation.

### Weaknesses
The computational overhead isn't thoroughly analyzed, which concerns me for large-scale deployment. I also think the paper oversells the novelty a bit since similar meta-learning approaches exist in the literature, and the comparison with some recent temporal GNN methods like EvolveGCN seems incomplete.

### Questions
How does the method scale to graphs with millions of nodes?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new paradigm for language model reasoning, called Sentence-level Latent Reasoning. Unlike traditional token-level autoregressive generation, this approach enables the model to perform prediction and reasoning in the sentence embedding space, thereby achieving a more abstract and efficient reasoning process.

### Strengths
1. The paper introduces a novel idea of performing autoregressive reasoning in the sentence embedding space, elevating the reasoning hierarchy of language models from the token level to the sentence level.
2. The proposed SentenceLens enables decoding intermediate latent states into natural language sentences, making the model’s “thought process” interpretable and analyzable.
3. By reasoning directly in the continuous embedding space, computational efficiency is significantly improved (1.5–2.5× speedup).

### Weaknesses
1. Although the paper proposes the concept of “sentence-level reasoning,” it does not sufficiently justify why sentence-level embeddings can necessarily capture the logical structures required for reasoning that token-level embeddings cannot.
2. The comparison is quite limited: it only benchmarks against one latent reasoning model (Coconut), and shows no significant advantage except on the Blocksworld task (where the performance gap is unusually large, raising concerns about possible evaluation bias). Additionally, computational efficiency is not compared with Coconut.

### Questions
1. Does your method offer any computational efficiency advantage compared to Coconut? If so, by how much?
2. Why do you introduce an encoder during training? Why are its parameters shared with the decoder?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework for performing latent reasoning in the sentence embedding space. The authors repurpose a pretrained language model (GPT-2) and construct two types of sentence-level embeddings (semantic and contextual) to map natural language into a continuous latent space, enabling autoregressive generation within that space. The paper further introduces a visualization tool, *SentenceLens*, to analyze the model’s “latent reasoning trajectories” in embedding space.

### Strengths
- The experiments span diverse reasoning domains (mathematical, logical, commonsense, and planning), demonstrating the general applicability of the framework.

- The paper provides an interpretability tool (*SentenceLens*) for analyzing latent reasoning trajectories.

- Despite the complex design, the authors’ exposition is overall clear and logically structured.

### Weaknesses
- The InfoNCE loss ratio and λ parameter for CTX-C are not specified, making the experiments hard to reproduce.

- The scope of fine-tuning (which components are updated) is unclear, and gradient flow is not described.

- No comparisons are made against strong latent-reasoning baselines such as CoCoMix, CoDi, or Token Assorted.

- Table 2 reports only a single inference mode; it should separately present accuracy and FLOPs across different reasoning modes.

- The empirical improvements over coconut and cot are relatively minor and do not convincingly demonstrate the claimed efficiency–performance trade-off.

### Questions
- Is $h_N$ the final-layer hidden state or a multi-layer aggregation?

- What value of λ is used for CTX-C? Have you performed a λ-sweep analysis?

- How does *SentenceLens* decode embeddings originating from different encoders?

- What is the exact architecture of the M-projection head? Is it used during both training and inference?
 How does the Continuous mode avoid semantic drift without the M head?

- Which components are frozen during fine-tuning? Do the encoder and decoder receive gradient updates?

- Why are CoCoMix, CoDi, and Token Assorted excluded from the baseline comparisons?

### Soundness
2

### Presentation
3

### Contribution
2
