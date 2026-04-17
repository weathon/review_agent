# STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Spoken Language Models (SLMs) are designed to take speech inputs and produce
spoken responses. However, current SLMs lack the ability to perform an internal,
unspoken thinking process before responding. In contrast, humans typically engage
in complex mental reasoning internally, enabling them to communicate ideas clearly
and concisely. Thus, integrating an unspoken thought process into SLMs is highly
desirable. While naively generating a complete chain-of-thought (CoT) reasoning
before starting to talk can enable thinking for SLMs, this induces additional latency
for the speech response, as the CoT reasoning can be arbitrarily long. To solve
this issue, we propose STITCH, a novel generation method that alternates between
the generation of unspoken reasoning chunks and spoken response chunks. Since
the audio duration of a chunk of spoken response is much longer than the time to
generate the tokens in a chunk of spoken response, we use the remaining free time
to generate the unspoken reasoning tokens. When a chunk of audio is played to the
user, the model continues to generate the next unspoken reasoning chunk, achieving
simultaneous thinking and talking. Remarkably, STITCH matches the latency
of baselines that cannot generate unspoken CoT by design while outperforming
those baselines by 15% on math reasoning datasets; STITCH also performs equally
well on non-reasoning datasets as those baseline models.
Some animations and demonstrations are on the project page: https://d223302.github.io/STITCH.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces STITCH, a Spoken Language Model (SLM) that performs internal reasoning alongside speech generation, theoretically without increasing interaction latency. STITCH generates speech chunk-by-chunk, with each chunk containing reasoning tokens, text tokens, and speech tokens. Experimental results demonstrate that this method improves reasoning performance compared to baseline approaches.

### Strengths
- The idea of chunk-wise reasoning alongside speech generation is novel.
- The investigation into optimal reasoning chunk length and the use of reasoning from other models adds further novelties.
- Experimental results clearly show the effectiveness of the proposed method.

### Weaknesses
- The paper does not compare STITCH against thinker-talker models. The substantial drop in reasoning performance observed in interleaved models (relative to text-only LLMs) often stems from the interleaved LLM's need to generate both text and speech tokens. In contrast, the thinker component in thinker-talker models generates only text, addressing this limitation. Although the authors mention that thinker-talker models are harder to fine-tune—which may imply challenges in adapting STITCH to such architectures—it would be fairer to include thinker-talker models in the comparisons.
- Evaluation is conducted solely on text outputs. While this is somewhat acceptable, it would be more robust to also evaluate directly on speech outputs, since the performance gap between text and speech evaluations can depend on the backbone LLM’s text-speech alignment capabilities.
- There is an inconsistency in the introduction section. Lines 084-087 state that STITCH-S cannot generate reasoning by design, but it actually generates text reasoning—it simply does not generate text reasoning in the first chunk. Am I missing something here?
- Statistical analysis should be performed to assess whether there are significant differences in responses among TBS, STITCH-R, STITCH-S, etc.

### Questions
N/A.

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
3

### Summary
This study introduces STITCH, a method designed to incorporate reasoning into speech by interleaving it within the existing Speech-Language Model (SLM) framework—specifically building upon GLM-4-Voice, one of the text–speech interleaving architectures (e.g., GLM-4-Voice, VITA-Audio). The authors propose two variants, -R and -S, depending on whether reasoning appears at the very beginning or after the first text–speech chunk.
The proposed approach is optimized based on the A100 + vLLM configuration. Compared to the baseline method, in which GLM-4-Voice performs “thinking” before speaking, STITCH demonstrates comparable performance on mathematical reasoning tasks and achieves performance on general conversational tasks that is similar to the original GLM-4-Voice.

### Strengths
**[S1]** The paper is written clearly and is easy to follow. The operational mechanism of the proposed method is straightforward to understand.

**[S2]** Notably, the model exhibits comparable results on mathematical tasks compared to approaches that explicitly perform reasoning with reduced latency.

### Weaknesses
While the study presents an interesting direction, the scope of its **novelty and generalizability appears somewhat limited**. The following points are offered as considerations rather than criticisms:

**[W1]** The reported optimization is based on the **A100 + vLLM** setting, which may limit the applicability of the results. It remains uncertain whether the proposed approach would generalize well to limited hardwares, larger models, or alternative architectures, such as those that jointly optimize a separate decoder (or talker) during inference rather than interleaving, or models that predict codecs (e.g., *moshi*). In addition, the interleaving method applied in this study appears to extend the existing text–speech interleaving approach, already performed in models such as GLM-4-Voice and VITA-Audio, to the reasoning component, without introducing additional architectural or methodological considerations specific to reasoning itself.

- **[Q1]** To what extent do the reported gains persist across different hardware budgets, model scales, and architectural variants (e.g., non-interleaving pipelines with a joint talker/decoder, or codec-predictive systems such as moshi)?

**[W2]** The dataset used comprises general dialogue, mathematical, and knowledge-intensive questions generated through **GPT-4o**. It is possible that employing reasoning-specialized models, such as **Qwen3-235B** or **DeepSeek**, for path construction might have yielded broader improvements, particularly in the non-reasoning tasks presented in Table 1. The comparable performance to the baseline GLM-4-Voice on general tasks therefore leaves some ambiguity regarding how the reasoning path contributes in those contexts.

- **[Q2]** Does the choice of path-generating model (e.g., GPT-4o vs. reasoning-oriented models) materially affect downstream performance, especially on non-reasoning tasks (Table 1)?

**[W3]** The paper does not provide an analysis or guarantee as to whether the reasoning path consistently concludes earlier in the 100 / 13 / 26 configuration, nor whether reasoning reliably precedes the corresponding text segments. Given that human reasoning typically precedes linguistic expression, a more explicit discussion or examination of this alignment could further strengthen the study.

- **[Q3-1]** Does reasoning precede and complete before corresponding text/speech segments, and does early termination of reasoning path occur in the 100 / 13 / 26 configuration?

- **[Q3-2]** If the reasoning segment tends to finish much earlier than the corresponding text, could the authors examine (1) whether adjusting the reasoning chunk length during training so that it depletes at a pace similar to the text segment, and (2) independently, enforcing a structure where reasoning always precedes the textual response (to prevent the model from answering first and reasoning afterward), would lead to different experimental results?

### Questions
See weaknesses

### Soundness
3

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
The paper proposes STITCH, a generation method that alternates between unspoken reasoning chunks and spoken response chunks. It aims to significantly reduce the latency between reasoning generation and spoken response. Experimental results on both reasoning (e.g., math) and non-reasoning datasets, such as TriviaQA, show that STITCH performs on par with or better than baselines that either reason before speaking or do not perform reasoning at all.

### Strengths
- The authors introduce three ways of integrating reasoning into spoken language models: Thinking Before Speech (TBS), Simultaneous Thinking and Talking with Reasoning First (STITCH-R), and Simultaneous Thinking and Talking with Speaking First (STITCH-S). The methodology is clearly described, and Figure 2 effectively visualizes the differences between these methods.
- The analysis is good. The authors report performance while varying the length of reasoning chunks during inference and analyze the number of tokens used in the reasoning process. They also conduct experiments using different reasoning models to study how reasoning quality affects spoken responses.
- Experiments are conducted on both reasoning-oriented and non-reasoning datasets, making the experiments comprehensive.

### Weaknesses
- Based on the performance tables (Table 1-(a) and 1-(b)), there is no clear winner that consistently outperforms all other baselines. On math datasets, STITCH-R and STITCH-S show mixed performance across models, sometimes performing significantly worse than TBS (e.g., TBS 64.94, STITCH-R 58.70, STITCH-S 56.72). Similarly, on non-reasoning datasets, STITCH-R and STITCH-S perform inconsistently relative to other baselines. The paper does not clearly explain the reasons behind these trends.
- Following the previous point, there are no clear guidelines on when to use STITCH-R versus STITCH-S.
- The main motivation for reducing interaction latency is to improve user experience while maintaining response quality. However, the paper does not include any user-centric evaluation (e.g., human preference, perceived responsiveness, or satisfaction).
- It is unclear how STITCH handles cases where reasoning takes longer than speech generation. Would the model introduce pauses until reasoning is completed? Such pauses might make the users feel more inconvenient to use it.

### Questions
In STITCH-S, reasoning occurs after the response. In this case, the reasoning could be merely post-hoc justification. Why does this variant outperform baselines that reason before speaking or models without reasoning?

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
3

### Summary
This paper introduces STITCH, a new framework that allows Spoken Language Models (SLMs) to “think” while they are “talking.” Current SLMs can only generate speech responses directly, without any internal reasoning before speaking. In contrast, humans silently think before responding aloud. STITCH mimics this behavior by dividing the reasoning and response generation into small “chunks.”
Instead of generating a long chain of reasoning first, STITCH alternates between reasoning chunks and speech chunks, so the model can keep thinking while speaking. The paper shows that STITCH improves reasoning performance by 15–20% especially on math tasks compared to baseline models, while keeping response latency nearly the same. It performs equally well on non-reasoning tasks, showing that “thinking” doesn’t harm general speech quality.

### Strengths
- The idea is novel — enabling SLMs to think internally while speaking. The chunked reasoning design, STITCH-R and STITCH-S is creative and practical for real-time systems.
- Overall the paper is very well-written

### Weaknesses
- The experiments mainly focus on math reasoning datasets like GSM8K and SVAMP. It would be valuable to test STITCH on more diverse reasoning domains such as commonsense, dialogue reasoning, or multi-hop factual reasoning.

### Questions
- The paper compares only within GLM-4-Voice variants. Is it possible to see how STITCH performs when applied to other open-source SLMs like Qwen2.5-Omni or Baichuan-Audio to prove generalizability?
- How does STITCH behave under actual streaming conditions where speech synthesis latency varies?

### Soundness
3

### Presentation
3

### Contribution
3
