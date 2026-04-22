# Understanding Conformal Factuality for RAG-based LLMs: Novel Metrics and Systematic Insights

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Large language models (LLMs) are powerful generic generative models which can often generate responses that are plausible but not grounded in factual reality usually referred to as “hallucinations”. This is a challenge for using LLMs in applications that require answers that are factually correct. Two approaches have emerged as promising ways to mitigate this issue in the literature: (i) Conformal factuality filtering framework that provides statistical guarantee on the factual accuracy of claims in the final output, but cannot mitigate the hallucinations in the response generation and (ii) retrieval-augmented generation (RAG), which utilizes trusted knowledge bases as reference to guide the generation of response with the aim of reducing hallucinations but does not offer statistical guarantees. In this work, we unite these two approaches by integrating a conformal factuality framework with RAG and systematically study their performance to understand their strengths and limitations. We investigate the role of different key components: reference for generation and scoring functions, sensitivity to calibration data, LLM model capacity, reasoning and robustness to distractors. We propose three new metrics: \emph{non-empty rate}, \emph{non-vacuous empirical factuality},  and  \emph{sufficient correctness} to address limitations of standard factuality measures that fail to meaningfully capture usefulness of the output. Our experiments are comprehensive spanning three datasets (FActScore, MATH, and Natural Questions) and multiple model families and sizes. Our results show the importance of designing scoring functions and highlight trade-offs between correctness and informativeness that standard metrics fail to capture. Together, our findings provide insights that are practically useful and sheds light on the importance of re-thinking LLM factuality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors investigate a conformal factuality framework with RAG: the performance and the limitations. Three  metrics are introduced for standard factuality limitation.  According to the experiments on three datasets (FActScore, MATH, and NQ), importance of designing scoring functions is concluded.

### Strengths
1. The paper unifies two complementary paradigms (RAG + CP) and offers a principled framework for combining grounding and statistical guarantees. Comparing to previous work, a comprehensive understanding of strenghts and limitations is explored.

2. Comprehensive empirical design: evaluates across three diverse datasets (FActScore, MATH, Natural Questions) and multiple model families (Qwen3, SmolLM2, Llama-3.x), enabling cross-domain and cross-architecture study.

3. Analysis on robustness of scoring function: distribution shift and distractor. And some other effect is done: access to reference, prompt strategy, modeling scaling.

### Weaknesses
1. Limited novelty in the algorithmic contribution: It shows systematic analysis rather than a new algorithmic integration of RAG and CP.

2. In the study, score function is needed. Entailmment-based scores and LLM-based scores are mentioned. Reference is usually needed. The limitation add the challenging of real-word (noisy retrieval or insufficient information). In addition, although some comparison is done for the two family  scorers, it is not comprehensive: e.g,, larger model (>8B), other architecture (MOE) etc. And it is unclear why Entailmment-based scores are better.

3. The metrics: Lot of metrics do not change a lot. It seems only "Power" metric have a higher variance for different settings. There is no big variance for the proposed metrics (Non-empty Rate, Non-vacuous Empirical Factuality, Correctness, Sufficient Correctness). The value of the proposed metrics are unclear.

### Questions
1. Practical deployment:
In real applications (e.g., QA systems), how would users set the target factuality level 1−α? Is there guidance on selecting α to balance factuality and informativeness?

2. Entailment vs confidence scorers:
Why do entailment-based scorers sometimes outperform larger LLM-based ones? Is it due to explicit textual grounding or better calibration properties?

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
4

### Summary
This paper studies RAG with conformal factuality filtering, a framework designed to provide statistical guarantees on output factuality. The authors conduct a systematic analysis across diverse datasets (FactScore, MATH, NQ) and model families(Qwen3 0.6B, 4B, 8B; Llama-3.2 1B&3B; SmlLM2), investigating critical components such as scoring function design, the role of references, model capacity, and robustness to distractors. Key findings indicate that while providing references consistently improves generation and scoring, current conformal filtering methods are highly sensitive to distribution shifts and adversarial distractors, highlighting the need for more robust scoring functions to ensure both conformal factuality are followed.

### Strengths
1. I appreciate the systematic studies on conformal filtering in a practical RAG setup (Section 5), which matches much more with the realistic setting, ie distraction happens, and the limitation of calibration set's collection. This result reveals the practical challenges of conformal filtering's guarantee where imperfect retrieval doesn't exist.

2. They presents a good coverage of evaluation to carefully consider the weakness of Empirical Factuality, and supplement Non-empty Rate, Non-vacuous Empirical Factuality.

### Weaknesses
1.  There are couples of statement is unclear in this paper and several inconsistency within the paper's description. More specifically:
1.1. It's unclear on how the calibration set and claims are extracted (Line 97-98)
1.2. The results (except Figure 2), doesn't use Sufficient Correctness (SC) at all, but it were claimed as a novel evaluation metrics contribution
1.3. Line 190 claims gpt-oss-20b is used, but no any results related to this model.

2. The coverage of the model are very small size in general. Given nowadays the most powerful Proprietary LLMs are faithfully strong when reference is provided, how the conformal factuality filtering will work is a much worth study direction. Given the limited novelty of the paper, it's hard to credit the paper enough if the study coverages are without stronger and deeper understanding about the frontier models.

3. Could diver a bit deeper more on the practical side for the calibration cost: Conformal methods often require expensive calibration; the paper does not discuss computational or data efficiency trade-offs.

4. Missing human evaluation: While factual metrics are well-defined, it would helpful to further validate the alignment with human judgement.

### Questions
See weakness for questions.

Some additional suggestions:
1. Line 080: "strenghts" => "strength"

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the integration of Retrieval-Augmented Generation (RAG) and the Conformal Factuality (CP) filtering framework to mitigate LLM hallucinations. The authors found that aggressive filtering to maintain factuality often leads to a high rate of empty or vacuous outputs. To evaluate this challenge, they introduce three new metrics—Non-empty Rate, Non-vacuous Empirical Factuality, and Sufficient Correctness—which explicitly capture the balance between correctness and informativeness (utility). Key results show that references significantly boost generation quality, lightweight entailment-based scoring functions are often more efficient and effective than large LLM confidence scorers, and the entire system is highly sensitive to distribution shifts and distractors in the data.

### Strengths
- The study is done in various datasets and LLMs families. 
- The experiments show insightful findings like critical flaw in standard measures which fail to penalize empty or useless outputs, and vulnerability to distribution shifts and adversarial distractors.

### Weaknesses
- The paper contribution is limited. While the systematic analysis is appreciated, the core contribution is largely incremental, relying on the mechanistic combination of two established techniques.
- The study's findings are heavily focused on the trade-offs within the combined method (e.g., comparing different scorers or prompt designs for the combined system). However, a comprehensive analysis requires more robust comparison against strong standalone baselines under the exact same circumstances.

### Questions
Your robustness tests show a core conflict: forcing high accuracy (EF/NvEF) makes the output unhelpfully empty (low NR). For a practical RAG system that needs to be both trustworthy and useful, what specific combination of your new metrics (NvEF, NR, and SC) should a practitioner target? Is there a principled way to choose between a very safe but less informative answer (e.g., 90% correctness, 50% coverage) and a more complete answer that is slightly riskier (80% correctness, 80% coverage)?

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
This paper systematically investigates the integration of Conformal Factuality (CF) filtering with Retrieval-Augmented Generation (RAG) to mitigate Large Language Model (LLM) hallucinations while providing statistical guarantees. The authors identify a critical limitation in existing CF metrics, where empty responses are considered perfectly factual. To address this, they introduce three new metrics: Non-empty Rate (NR), Non-vacuous Empirical Factuality (NvEF), and Sufficient Correctness (SC), which better balance factuality with informativeness.

The study conducts comprehensive experiments across multiple datasets (FActScore, MATH, Natural Questions) and model families (Qwen3, Llama-3.2, SmolLM2).

### Strengths
- The introduction of new metrics for measuring factuality makes sense.
- As an empirical study paper, it's good to consider multiple model families and benchmarks.

### Weaknesses
- While titled as "RAG-based," much of the core setup assumes an "oracle retriever" that always provides relevant references (Section 2), which is a bit unrealistic. The quality of the reference should be critical as well - beyond simple binary existence in Section 3.

- After reading the paper, I find it hard to connect the insights the authors obtained to real-world settings and their practical implications. For example, how we apply the findings to challenging agentic scenarios for designing effective information-seeking agents (based on RAG or Internet Search)?

- Why stronger models of larger sizes are not included in the study? For example, we have Qwen3-30B and its thinking variants, which are widely used in more practical agentic settings.

- Unfinished sentence in line 172.

- Margin issue of Figure 4 and Figure 8.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3
