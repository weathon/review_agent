# SinkTrack: Attention Sink based Context Anchoring for Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4, 6

## Abstract
Large language models (LLMs) suffer from hallucination and context forgetting. Prior studies suggest that attention drift is a primary cause of these problems, where LLMs' focus shifts towards newly generated tokens and away from the initial input context. To address this, we make use of a related, intrinsic characteristic of LLMs: attention sink – the tendency to consistently allocate high attention to the very first token (i.e., ⟨BOS⟩) of a sequence. Concretely, we propose an advanced context anchoring method, SINKTRACK, which treats ⟨BOS⟩ as an information anchor and injects key contextual features (such as those derived from the input image or instruction) into its representation. As such, LLM remains anchored to the initial input context throughout the entire generation process. SINKTRACK is training-free, plug-and-play, and introduces negligible inference overhead. Experiments demonstrate that SINKTRACK mitigates hallucination and context forgetting across both textual (e.g., +18.9% on QuAC with Llama3.1-8B-Instruct) and multi-modal (e.g., +23.0% on M3CoT with Qwen2.5-VL-7B-Instruct) tasks. Its consistent gains across different architectures and scales underscore the robustness and generalizability. We also analyze its underlying working mechanism from the perspective of information delivery. Our source code is available at anonymous GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SINKTRACK, a training-free, dual-track cross-attention mechanism, solves the LLM problem of "forgetting" the original input by turning the naturally stable first token into an active information anchor that constantly and adaptively retrieves the most important details from the context.

### Strengths
1. The paper presents novel solutions to mitigate context forgetting and hallucination in LLMs by actively leveraging the model's intrinsic "attention sink" tendency towards the initial <bos> token.
2. The quality of the work is evident through the systematic exploration of context anchoring methods, such as soft injection and SINKTRACK, which effectively enhances context coherence. The empirical results on relevant datasets confirm the method’s efficacy and significance as a solid step forward in improving LLM reliability over long contexts.
3. This paper excels in clarity. The methodology is simple and well-motivated, and the overall organization of the paper is clear and easy to follow.

### Weaknesses
1. Sticking all the important context onto just the first token might create an information bottleneck for really long documents. The author needs to consider the impact of this.

2. The 'CoT' baseline is too simple, making the new method's performance improvements look bigger than they might actually be. The author needs to compare with more advanced methods.

3. It is critical to include experiments demonstrating that SINKTRACK does not hurt the model's general abilities, such as common sense reasoning and instruction following, in standard short-context benchmarks.

### Questions
1. Why is the $L_2$ norm used for Information Gain in Equation 5 instead of cosine similarity or another metric?

2. In "soft injection," does the "information vector" refer to the vector corresponding to the image information?

3. Does soft injection perform fusion on the $\langle\text{bos}\rangle$ hidden state before the key-value projection, and how is the $\alpha$ coefficient set?

4. Why does the CoT (Chain-of-Thought) method show a decrease in performance compared to the baseline in Table 2?

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
This paper proposes a test-time method to mitigate reduce hallucination in MLLMs, and context forgetting on both LLMs and MLLMs. It uses soft injection to fuse information into LLM's computational flow.

### Strengths
1. The paper is generally well-written and easy to follow.
2. The authors conducted comprehensive experiments on multiple tasks and multiple LLMs to validate their idea.

### Weaknesses
1.	Methodological clarity:
Some details in the method section remain unclear. For example, it is not explicitly stated how f_{\text{info}} is obtained. From Figure 3, it appears that f_{\text{info}} results from mean pooling over the encoder outputs; however, the architecture details of the Vision/LLM encoder are missing. The authors also mention that this work focuses only on decoder-only LLMs, which further raises questions about how the encoder component is integrated. In addition, an analysis of the extra computational overhead introduced by this module would be helpful.
2.	Experimental setup:
In Table 2, it is unclear why Chain-of-Thought (CoT) prompting performs worse than Direct Prompting, particularly on text-only tasks. Moreover, the current baselines seem rather limited. Including stronger or more diverse baselines would better demonstrate the effectiveness and generalizability of the proposed method.
3. Definition ambiguity:
The distinction between vertical and horizontal information flow is not clearly justified. It is unclear whether this formulation is newly introduced by the authors or adapted from prior work. Providing clearer definitions and theoretical grounding would strengthen the analysis.
4. Potential information loss and generalizability:
The proposed approach compresses all information into a single token, which may still lead to information loss. Although the reported results on the two tasks are promising, the generalizability of this information compression strategy remains uncertain. Further empirical validation on broader or more diverse tasks would make the claims more convincing.

### Questions
see weakness

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
2

### Summary
This paper tackles hallucination and context forgetting caused by attention drift. The authors observe that, during generation, LLMs gradually shift their focus toward newly generated tokens and away from the initial context. Inspired by attention sink, they propose SinkTrack, a training-free, inference-time enhancement that treats the BOS token as an information anchor and continually injects key contextual features (e.g., image or instruction embeddings) into its representation. Experiments on both text-only and multi-modal QA show that SinkTrack suppresses drift-induced hallucinations and forgetting, outperforming both direct inference and chain-of-thought reasoning.

### Strengths
1. The paper highlights how attention drift triggers hallucination and context forgetting, which is a key insight for both long-context LLMs and long-context multi-modal LLMs.  
2. The authors introduce a novel, training-free inference enhancement that alleviates the above issues with negligible overhead.  
3. The effectiveness is demonstrated across the Qwen, Gemma, MiniCPM, and Llama families, accompanied by visualization analyses and discussions.

### Weaknesses
1. The evaluation benchmarks are somewhat outdated. While my limited knowledge of vision-related tasks and hallucination-related discussion prevents me from assessing those aspects, relying solely on QuAC and SQuAD to evaluate SinkTrack’s long-context improvements is clearly insufficient. The authors should first justify why QuAC is representative and then add results on standard long-context benchmarks such as LongBench [1], L-Eval [2], NIAH [3], and RULER [4], or more recent long-term conversation benchmarks like LongMemEval [5].
2. The authors compare SinkTrack only against direct query and Chain-of-Thought, without comparison with the methods cited in Related Work. While three main streams of prior work are reviewed, their empirical relationship to SinkTrack has not been examined, such as an experimental comparison with a retrieval-augmented method. Including such comparisons would greatly strengthen the cohesion of the paper and the soundness of the method.
3. The evaluation does not state the context lengths used, nor does it examine how attention drift varies as prompt length grows. Because SinkTrack itself relies on attention-based calculation, I am worried about whether the attention for SinkTrack could also drift when the prompt becomes very long. 

[1] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding https://arxiv.org/abs/2308.14508

[2] L-Eval: Instituting Standardized Evaluation for Long Context Language Models https://arxiv.org/abs/2307.11088

[3] Needle In A Haystack - Pressure Testing LLMs https://github.com/gkamradt/LLMTest_NeedleInAHaystack

[4] RULER: What's the Real Context Size of Your Long-Context Language Models? https://arxiv.org/abs/2404.06654

[5] LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory https://arxiv.org/abs/2410.10813

### Questions
See Weaknesses

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
3

### Summary
This paper proposes a training-free method to enable context anchoring (reducing hallucination) for language models, the key idea is to leverage the attention sink phenomena and fuse the hidden representation of context into the sink token to encourage more attention to the context.

Experiments are conducted on three LLMs and four VLMs, showing improved performance over baseline (direct inference and CoT inference).

### Strengths
* The paper is written relatively clearly.
* The idea of leveraging attention sink to encourage context anchoring is interesting.

### Weaknesses
* The baseline is limited: Using attention sink to enhance context anchoring is nice, yet this paper does not compare to prior training-free methods to mitigate halluciation. To name a few: [ACT(ICML 2024)](https://arxiv.org/pdf/2406.15765), [VCD(CVPR 2024)](https://arxiv.org/pdf/2311.16922), [OPERA(CVPR 2024)](https://arxiv.org/pdf/2311.17911), [SID(ICLR 2025)](https://arxiv.org/pdf/2408.02032), [DAC(ICML 2025)](https://arxiv.org/abs/2502.01969).

* It is nice that the paper evaluates on both image and textual benchmarks, yet i found the choice of textual benchmark a bit unconventional. If the idea is to mitigate hallucination for long context, evaluation on well-studied benchmarks such as [LongBench](https://arxiv.org/abs/2308.14508) would be appropriate.

### Questions
* IIUC, the SINKTRACK method lets the <BOS> token attend to the rest of the tokens (instead of replacing the K,V of its own with the mean pooling of the rest of the token). If so, that should be illustrated clearly in Figure 3 (which shows the mean pooling operation).
* I would suggest the author to present their full methods more clearly. Currently the paper is written to illustrate the "failed" attempt first (hard injection, soft injection, mean pooling), which makes it a bit hard to understand the actual proposed method. These comparison can be included in the ablation study.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SinkTrack, a training-free, plug-and-play mechanism that enhances large language models (LLMs) by addressing two key issues: hallucination and context forgetting. The authors identify attention drift—the model’s tendency to focus increasingly on recent tokens during generation. They exploit an opposing, intrinsic behavior called the attention sink, where the first token (⟨BOS⟩) consistently receives high attention throughout decoding. SinkTrack injects key contextual information into the ⟨BOS⟩ token’s representation, and introduces a dual-track attention mechanism: one track performs adaptive cross-attention for ⟨BOS⟩, while the other maintains standard causal self-attention for other tokens. This allows ⟨BOS⟩ to query relevant information dynamically, counteracting information decay during long-context generation. The experimental results show its promise. SinkTrack requires no training, introduces negligible inference overhead, and is compatible across architectures and modalities. The authors also provide interpretability analyses of information flow, showing how the ⟨BOS⟩ token propagates injected information vertically (across layers) and horizontally (across tokens).

### Strengths
SinkTrack is training-free, plug-and-play, and requires only a one-time injection at inference.

The dual-track design elegantly balances adaptation (via BOS cross-attention) and model integrity (via standard causal flow), preserving pretrained representations while enhancing context retention.

Evaluations span six datasets, covering both text and vision-language reasoning.

### Weaknesses
The approach is empirically strong but lacks formal justification for why attention anchoring should improve global consistency.

There’s no discussion of convergence, gradient flow, or formal guarantees that information propagation remains stable as sequence length grows.

The early versions of SinkTrack use mean-pooling for contextual information compression, which can cause information loss in very long contexts.

While the benchmarks are diverse, all are QA or reasoning tasks. The generality claim would be stronger with results on open-ended generation, summarization, or code generation.

The writing is dense in places and could be more accessible. For instance, the transitions between hard, soft, and dual-track injection could be improved.

### Questions
Would you please provide more justification for attention anchoring, and its convergence properties?

### Soundness
2

### Presentation
3

### Contribution
2
