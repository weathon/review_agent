# Revisiting Long-context Modeling from Context Denoising Perspective

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Long-context models (LCMs) have demonstrated great potential in processing long sequences, facilitating many real-world applications. The success of LCMs can be attributed to their ability to locate implicit critical information within the context for further prediction. However, recent research reveals that LCMs are often susceptible to contextual noise, i.e., irrelevant tokens, that can mislead model attention. In this paper, we conduct a fine-grained analysis of the context noise and propose an effective metric, the Integrated Gradient (IG) score, to detect and quantify the noise information within the context. Our findings reveal that even simple mitigation of detected context noise can substantially boost the model's attention on critical tokens and benefit subsequent predictions. Building on this insight, we propose Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. Extensive experiments across four tasks, under both context window scaling and long-context alignment settings, demonstrate the superiority of CDT. Notably, when trained with CDT, an open-source 8B model can achieve performance (50.92) comparable to GPT-4o (51.00).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Context Denoising Training (CDT), a straightforward yet effective training strategy that improves attention on critical tokens while reinforcing their influence on model predictions. The authors first propose the Fact Retrieval (FR) score and Integrated Gradient (IG) score to identify and quantify context noise, demonstrating that existing long-context methods fail to distinguish critical from irrelevant information, degrading performance. They show that simple mitigation of detected context noise can substantially boost the model’s attention on critical tokens and benefit subsequent predictions.

Building on this insight, CDT subtracts the corresponding gradients to manipulate the irrelevant token embeddings, thus suppressing the context noise. This allows the model to focus on critical tokens and strengthens the causal link between them and the final output. Across LongBench, RULER, BABILong, LongPPL, and other benchmarks, CDT consistently boosts long-context performance.

### Strengths
1. This paper conducts extensive experiments across multiple settings and long-context benchmarks, thoroughly demonstrating the effectiveness of CDT over multiple baselines.
2. The introduction of the FR score, IG score, and identifier $\mathcal{I}(x_i)$ provides valuable insights into how long-context LLMs comprehend long contexts.
3. Beyond presenting an effective long-context training paradigm, the authors offer in-depth discussions and visualized analyses that solidify the credibility of their approach.

### Weaknesses
See Questions.

### Questions
1. The paper introduces the FR score, IG score, and an identifier. However, only the identifier is actually used in CDT. The FR and IG scores serve only as diagnostic tools because they require prior knowledge of the four token types in the context. I would appreciate it if the authors could clarify these two distinct contributions more clearly.
2. It is unclear why low-frequency words should be singled out. No prior work is cited to justify this taxonomy. A simpler partition into supporting, interference, and irrelevant tokens appears sufficient, and the proposed identifier itself only decides whether a token is irrelevant.
3. I find it confusing that authors compare CE, LongCE, and CDT with YaRN. YaRN is not a training paradigm. It is merely an interpolation method like NTK. Likewise, comparing CDT at inference time with FlexPrefill and XAttention is also confusing. These techniques aim at enhancing efficiency, not performance.
4. As a work focused on long-context training, CDT lacks citations to early efforts in this line[1-3]. In addition, the strategy of identifying critical tokens to improve training or inference should be clearly differentiated from similar techniques originally developed in short-input or long-output scenarios[4-5].
5. The authors note in the appendix that the improvement brought by our method on complex reasoning tasks is not as significant as that on other tasks. Since CDT also identifies critical tokens to enhance reasoning, could the high-entropy token approach[5] be combined to further boost performance?
6. Typo: irrelevant instead of irrevelant in Figure 3a

[1] Effective Long-Context Scaling of Foundation Models https://arxiv.org/abs/2309.16039

[2] Long Context is Not Long at All: A Prospector of Long-Dependency Data for Large Language Models https://arxiv.org/abs/2405.17915

[3] LongWanjuan: Towards Systematic Measurement for Long Text Quality https://arxiv.org/abs/2402.13583

[4] RHO-1: Not All Tokens Are What You Need https://arxiv.org/abs/2404.07965

[5] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning https://arxiv.org/abs/2506.01939

### Soundness
4

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
4

### Summary
This paper identifies a key weakness in long-context models (LCMs): their susceptibility to contextual noise—irrelevant tokens that distract the model from critical information. The authors propose a novel training strategy called Context Denoising Training (CDT), which improves a model’s ability to focus on salient tokens and strengthen their influence on predictions.

### Strengths
1. The Integrated Gradient (IG) score, which more accurately identifies important tokens compared to attention-based methods.

2. Manual noise suppression experiments: Showing that reducing noise in input embeddings boosts attention on critical tokens by ~10×.

3. The CDT training strategy: A lightweight, online method that detects and suppresses noisy tokens during training, improving model focus without heavy computational overhead.

4.Extensive evaluation: Across 4 task types (real-world, synthetic, language modeling, reasoning) and multiple models, CDT consistently outperforms baselines and even enables an 8B model to nearly match GPT-4o on LongBench-E.

### Weaknesses
1. Is there any relevant literature or experimental evidence supporting the statement in line 759 that "performance gains typically exhibit diminishing returns with increased token budgets"?

2. In lines 768–769, it is stated that "LongCE achieves a 13-point gain per 1B tokens versus ProLong’s 0.3-point gain per 1B tokens." How was the 13-point gain calculated?

3. Why are the results for NExtLong-512K-Instruct and ProLong-512K-Instruct shown in Table 1, but not included in Table 2?

### Questions
See Weaknesses.

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
4

### Summary
the paper address the issue of Long-context language models (LCMs) struggle with contextual noise—when irrelevant information in lengthy contexts overwhelms critical tokens, leading to some issues. The paper propose some measurement.

### Strengths
* Tackles a crucial challenge in long-context language models: contextual noise overwhelming critical information
* High practical relevance for real-world applications (RAG, document QA, long-context reasoning)
* Problem is timely given the trend toward longer context windows in LLMs

### Weaknesses
1. The paper proposes an indirect approach: compute IG scores externally, identify critical tokens, perturb embeddings, then train. However, they provide no compelling explanation for why this is superior to simply training the model to learn token importance directly through standard optimization.

2. The preliminary study (Section 3) relies entirely on synthetic tasks with manually injected noise, which provides no evidence that the observed attention patterns occur in real-world scenarios.

3. The proposed method pre-selects critical tokens for training sequences, which seems to circumvent the model's own ability to learn token importance organically during training. Since intuitive learning suggests independent discovery of salient tokens, the authors should clarify why this indirect training approach outperforms direct training.

### Questions
see weakness.

### Soundness
2

### Presentation
3

### Contribution
2
