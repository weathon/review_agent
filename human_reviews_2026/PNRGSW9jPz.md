# EvA: An Evidence-First Audio Understanding Paradigm for LALMs

- Decision: Reject
- Scores: 8, 2, 4, 2, 8

## Abstract
While Large Audio Language Models (LALMs) have demonstrated remarkable capabilities in audio understanding tasks, their performance degrades sharply in complex acoustic scenes, revealing a fundamental limitation in their perceptual grounding. In this work, we first identify a critical failure mode that exposes this limitation: state-of-the-art LALMs paradoxically struggle more with simple evidence-extraction tasks than with complex reasoning ones. We diagnose this as a breakdown in acoustic evidence grounding, a problem rooted in systemic information loss during feature encoding and fusion. To address this, we introduce EvA (Evidence-First Audio), a new paradigm that prioritizes maximizing the fidelity of acoustic evidence. EvA's dual-encoder architecture combines Whisper with CED-Base, a ViT-based general audio encoder, and pioneers a structure-preserving, two-stage fusion process. First, it enriches evidence by hierarchically aggregating multi-level features from within the CED-Base encoder. Second, it integrates this representation with Whisper's output via a time-aligned, inject-and-add mechanism that guarantees perfect temporal integrity. To facilitate training for this paradigm, we co-develop EvA-Perception, a large-scale open-source dataset with high-temporal-precision annotations. Our resulting model establishes a new open-source state-of-the-art on multiple challenging benchmarks, including MMAU, MMAR, and MMSU. Crucially, EvA achieves its most significant gains on perception-heavy subsets, validating our hypothesis that addressing the evidence bottleneck is key to unlocking the next level of audio understanding.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper diagnoses a critical "evidence bottleneck" in Large Audio Language Models, arguing that their primary limitation is poor perceptual grounding rather than flawed reasoning. To address this, the paper introduces Evidence-First Audio, a novel paradigm built on a dual-encoder architecture (Whisper and CED-Base) and a non-compressive, two-stage fusion mechanism that hierarchically aggregates multi-level acoustic features while preserveing temporal fidelity. The paper also develops EvA-Perception, a large-scale dataset with high-temporal-precision annotations to facilitate training. The resulting model achieves new state-of-the-art performance on the MMAU, MMAR, and MMSU benchmarks, with the most significant gains on perception-heavy tasks, thereby validating their evidence-first hypothesis. The EvA-Perception dataset and EvA model will be released.

### Strengths
1.The paper presents an insightful diagnosis of a critical yet overlooked limitation in existing Large Audio Language Models. This diagnosis is well-supported by both theoretical arguments  and comprehensive experimental validation. 

2.To address this limitation, the paper introduces a novel dual-stream architecture and a purpose-built dataset, EvA-Perception, which collectively achieve state-of-the-art performance across multiple challenging benchmarks. 

3.The commitment to open-sourcing the models and the newly created dataset significantly enhances the reproducibility and potential impact of the work.

### Weaknesses
1.The paper proposes a sophisticated architectural design within the CED-path, yet the ablation studies do not fully justify this complexity. More granular ablations would strengthen the paper's design claims and provide clearer insights for future work.

2.While the paper is generally well-written, the introduction section could be improved. The core concept of "evidence" is central to the paper's thesis, yet it is used extensively without a concise, upfront definition, creating an initial barrier to understanding.

3. The description of the fusion mechanism as "lossless" (L214, L477) appears to be an overstatement.

### Questions
1.The term "evidence" is foundational to the paper's narrative and contributions. Could the authors provide a concrete definition of this concept?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work addresses the issue of information loss in evidence encoding for large audio language models. They approach it by combining multiple feature encoders, using hierarchical aggregation, time-aligned feature combination. They also contributes a dataset with high quality temporal annotations.

### Strengths
I appreciate the effort went into building the model and dataset. I encourage the authors to keep improving the work through technical innovation

### Weaknesses
1. Author name is revealed as the folder name in supplementary material

2. to show that model perception is lagging behind reasoning, instead of showing absolute scores of perception and reasoning as figure 1, one should show the gap between model and human performance, i.e. to show that the gap between model and human is bigger on perception

3. the encoder "CED-Base" is never introduced in the paper - the full model name is never shown, the architecture is never explained, even the original paper is never cited anywhere. 

4. I strongly oppose using information theory to level up a paper when it doesn't actually add anything to the work. Section 3.1 and 3.2 are not rigorous and even flawed, not helpful in explaining their approach, and is a waste of space. What they are trying to say is that using more audio encoders can provide more information, which is commonsense and do not need any theoretical motivation. 

why the math is not rigorous or even flawed:

4.1. Lack of a Well-Formed Probabilistic Model (whether implicit of not)
Although Section~3 defines
$$
Z:\text{ground-truth acoustic evidence}, \quad
X:\text{raw waveform}, \quad
H:\text{encoder hidden}, \quad
O:\text{final representation}, \quad
Y:\text{output text},
$$
the paper never specifies a joint distribution $p(Z,X)$.
Are $Z$ latent causes that generate $X$, or deterministic annotations extracted from it?
If $Z=f(X)$, then $I(Z;X)=H(Z)$ trivially and the Data Processing Inequality (DPI) adds nothing.
If $Z$ is latent, its distribution must be defined before mutual information can be evaluated.
Thus, every $I(Z;\cdot)$ term remains symbolic rather than quantitative.

4.2. Misuse of the Data Processing Inequality
The paper treats
$$
Z \rightarrow X \xrightarrow{E} H \xrightarrow{P} O \xrightarrow{\pi} Y
$$
as a Markov chain and directly applies
$$
I(Z;Y) \le I(Z;O) \le I(Z;H) \le I(Z;X).
$$
However, $E, P,$ and $\pi$ are deterministic neural networks whose parameters depend on the training data.
Once parameters are learned, the conditional independence required by DPI no longer holds.

4.3. No Operational Link to Performance
The claimed ``information ceiling’’ never connects $I(Z;Y)$ to measurable task metrics such as WER or accuracy.
A rigorous bound would invoke Fano’s inequality or rate–distortion theory to relate mutual information to the achievable Bayes risk.

4.4. Tautological Lemma in Section 3.2
The so-called ``Complementary Evidence Advantage’’ states:
$$
I(Z;O_1, O_2)
  = I(Z;O_1) + I(Z;O_2 \mid O_1)
  \ge I(Z;O_1),
$$
which is merely the chain rule for mutual information.
It does not establish that a dual-path model can achieve higher $I(Z;Y)$;
it only restates that adding variables cannot decrease mutual information.

4.5. Unsupported ``Strict Superiority’’
Proposition 1 claims a strict gain if $I(Z;O_2 \mid O_1) > 0$,
but this follows trivially from the lemma and is not empirically verified.

4.6. Unproven ``$Z$-Sufficient Fusion’’
The authors later require a fusion function $F$ satisfying
$$
I(Z;F(O_1, O_2)) = I(Z;O_1, O_2),
$$
yet their proposed frequency-pooled, gated fusion is many-to-one and clearly non-invertible.
No argument or estimator is offered to demonstrate that it preserves $Z$-information.

5. They claim to contribute a dataset, but there is no innovation in their approach because it just apply other (M)LLMs to extract and aggregate information (which is already used in many audio LLM works). Plus there is no example of the constructed dataset.

### Questions
no questions

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EvA (Evidence-first Audio-language model), a novel architecture designed to enhance the evidence-extraction capability of audio-language reasoning models. Unlike traditional end-to-end LALMs that directly generate answers from audio embeddings, EvA explicitly divides the reasoning process into two stages: (1) Evidence-extraction encoding audio segments with dual-encoder to avoid being hampered by fusion strategy; (2) Answer Generation—where a large language model performs reasoning and generates responses based on the extracted evidence and textual context. The authors argue that reasoning based on verifiable acoustic evidence can reduce the model's "hallucination" phenomenon and improve interpretability.
EvA is implemented as an extension of the Kimi-Audio-7B framework, incorporating a Time-Aware Alignment and Inject-and-Add Fusion mechanism. By fusing features from Whisper and CED while minimizing the loss of acoustic evidence during fusion, the model achieves enhanced performance. The paper also introduces a new dataset, EvA-Perception, which features high-temporal-precision annotations. Finally, the model achieves state-of-the-art results on multiple benchmarks including MMAU, MMAR, and MMSU.

### Strengths
The authors claim that EvA effectively mitigates the “evidence bottleneck” by increasing the amount of acoustic information available to the LLM without retraining encoders.

Evidence:
- Empirical results (Table 2): EvA surpasses strong baselines such as Kimi-Audio, Qwen2.5-Omni, and R1-AQA across all benchmarks.
- Ablation results (Table 3): show that adding the CED aggregator and alignment yields significant improvements in both AudioCaps CLAP metrics and benchmark perception accuracy.
- Qualitative examples (Fig. 3): demonstrate that EvA-generated captions capture fine-grained temporal and tonal details better than baselines.
- **Theoretical analysis** (Section 3): formally proves that dual-path fusion provides strictly higher information capacity than single-path models (via mutual information inequalities).

Overall, this method has Excellent theoretical–empirical consistency, Clear diagnosis of a real architectural weakness in existing LALMs, Strong experimental gains without retraining encoders.

### Weaknesses
1. Task coverage narrow (mainly perception, English-only).

2. The authors' dual-encoder shares similar architectural ideas with that in SALOMNN. However, there is a lack of experimental comparison. While the experiments include comparisons between the dual-encoder and single-encoder, they fail to demonstrate that their proposed Aggregation and Fusion strategies outperform the Window-level Q-Former used in SALMONN.

3. Meanwhile, there is a lack of theoretical proof regarding the advantages of their strategy over Q-Former.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

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
This paper introduces a promising dataset; however, it is unclear whether it will be publicly released. Since the main contribution lies in the dataset itself, the overall significance of the paper is limited. The experiments are not comprehensive, with insufficient results and analysis, and the LALM training approach is fairly standard with limited novelty.

### Strengths
The paper provides a valuable dataset that could benefit future research on LALM training and evaluation.

### Weaknesses
The proposed weak-to-strong and mixed-to-strong strategies with SFT and GRPO are standard practices in current literature and therefore lack novelty.

The experimental evaluation is not comprehensive, as it includes only three benchmarks. Exploring additional LALM benchmarks would help strengthen your claims.

The paper contains some redundant analysis that could be streamlined. I recommend focusing on deeper technical insights and introducing more concrete methodological novelties to strengthen the contribution.

The overall writing could be improved, particularly by providing a clearer introduction and a more structured related work section.

While the paper proposes an interesting dataset, the overall contribution is not substantial enough to meet the standards of ICLR. The methods and analyses presented are relatively incremental, and the paper would benefit from stronger technical innovations or deeper theoretical insights.

### Questions
As this paper seeks to advance large audio-language models through a newly proposed dataset AudioMCQ, could the authors clarify how frequently the dataset has been utilized in the domain and provide evidence that it is well-defined and meaningful? Have you released the dataset and codes, or do you plan to make them publicly available?

Is there any human annotation involved to verify the quality of the dataset? How do you ensure the correctness of the outputs generated by other models?

Could the authors elaborate on the potential research impact of this dataset and how it advances the field?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies a critical limitation in existing Large Audio Language Models (LALMs), which the authors term the evidence bottleneck. They show that the primary failure mode of state-of-the-art LALMs in complex acoustic scenes is not a deficiency in high-level reasoning, but rather a fundamental breakdown in perceptual grounding caused by information loss during audio encoding and fusion.To address this, the authors propose EvA, a new evidence-first paradigm. The core of EvA is a dual-encoder architecture that uses a speech-centric Whisper encoder and a generalist ViT-based audio encoder (CED-Base). The key novelty lies in its information-preserving fusion mechanism, which involves hierarchical aggregation of multi-level features from the generalist encoder and a time-aware inject-and-add fusion that aligns the generalist audio features to the Whisper timeline without temporal compression. To facilitate training, the authors also developed EvA-Perception, a new large-scale, open-source dataset with high-temporal-precision annotations designed to improve evidence-grounded training. The resulting EvA model is shown to set a new open-source state-of-the-art on the MMAU, MMAR, and MMSU benchmarks, with the most significant performance gains observed on perception-centric subsets, thereby validating the paper's central hypothesis.

### Strengths
1. Clear Problem Formulation and Motivation: The paper effectively identifies a critical problem in existing LALMs, positing that performance limitations stem from incomplete acoustic evidence. This hypothesis is well-supported by both empirical data in Figure 1 and the theoretical argument in Lemma 1. This clear problem diagnosis naturally motivates the proposed method: augmenting the audio encoder with a CED module to provide richer acoustic information.
2. Novel Architecture and Insight: The authors argue that previous information fusion methods are inherently lossy. In response, they propose a novel Aggregator designed to preserve information across different frequency bands and hierarchical layers. Furthermore, the inject-and-add strategy is a clever approach that effectively integrates semantic information while simplifying the training process.
3. Strong and Well-Analyzed Empirical Results: The paper demonstrates significant performance improvements, achieving state-of-the-art results across three challenging and diverse benchmarks. Crucially, the authors go beyond reporting overall scores by breaking down performance into Perception and Reasoning categories. The results convincingly show that EvA yields the largest gains in the Perception category while also improving Reasoning performance, validating the core hypothesis of the paper.
4. Significant Contribution to the Community: This work offers more than just a new perspective on audio representation; it also contributes several valuable datasets under the EvA. The creation of a large-scale, high-quality dataset with fine-grained temporal annotations is a substantial contribution in its own right and will be a valuable resource for future research.

### Weaknesses
1. Potentially Unfair Baseline Comparison: The comparison in Table 2 may lack fairness for two reasons. First, it appears the baseline models were not fine-tuned on the newly introduced EvA datasets, making it difficult to disentangle the performance gains from the novel architecture versus the new training data. Second, the paper notes its model is English-only, whereas many of the baseline models are multilingual. This linguistic mismatch could be another source of unfairness in the comparison.
2. Lack of Ablation Studies for the Aggregator Module: The paper's ablation studies primarily focus on the overall impact of the CED path. However, there are no detailed experiments to validate the specific design choices within the CED Aggregator itself. The individual contributions of the frequency-pooled gate and the cross-layer fusion mechanism have not been separately investigated, leaving the optimality of the Aggregator's design unsubstantiated.

### Questions
1. What is the detailed rationale behind using a frequency-pooled gate? What are the specific challenges (e.g., computational complexity, feature space mismatch) associated with retaining and fusing the full, unpooled 2D time-frequency feature map?
2. How significant is the empirical contribution of the cross-layer fusion mechanism? The paper hypothesizes that intermediate encoder layers provide richer, low-level information that is lost in the final layer. Is there direct empirical evidence from ablation studies to support this design choice and quantify its benefit over a simpler fusion that uses only the final layer's features?

### Soundness
4

### Presentation
4

### Contribution
4
