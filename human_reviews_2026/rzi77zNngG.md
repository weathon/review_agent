# Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Unlearning in large language models (LLMs) is critical for regulatory compliance and for building ethical generative AI systems that avoid producing private, toxic, illegal, or copyrighted content. Despite rapid progress, in this work we show that \textit{almost all} existing unlearning methods fail to achieve true forgetting in practice. Specifically, while evaluations of these `unlearned' models under deterministic (greedy) decoding often suggest successful knowledge removal using standard benchmarks (as has been done in the literature), we show that sensitive information reliably resurfaces when models are sampled with standard probabilistic decoding. To rigorously capture this vulnerability, we introduce \texttt{leak@$k$}, a new meta-evaluation metric that quantifies the likelihood of forgotten knowledge reappearing when generating $k$ samples from the model under realistic decoding strategies. Using three widely adopted benchmarks, TOFU, MUSE, and WMDP, we conduct the first large-scale, systematic study of unlearning reliability using our newly defined \texttt{leak@$k$} metric. Our findings demonstrate that knowledge leakage persists across methods and tasks, underscoring that current state-of-the-art unlearning techniques provide only limited forgetting and highlighting the urgent need for more robust approaches to LLM unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper argues that current LLM unlearning methods only appear effective when evaluated using greedy (deterministic) decoding, but actually fail under probabilistic decoding (e.g., using temperature and top-p sampling). To expose this hidden leakage, the authors propose Leak@k, a meta-metric that measures the likelihood that sensitive or "forgotten" information reappears in at least one of k generated responses. They conduct a large-scale evaluation across three major unlearning benchmarks (TOFU, MUSE, WMDP) and show that as the number of generations increases, leakage rises significantly across all existing unlearning methods, revealing that current techniques do not guarantee reliable forgetting in practical usage settings.

### Strengths
1. The motivation for introducing a new metric for unlearning evaluation is well-founded.

2. The paper is well-written and easy to follow.

3. The experimental section is comprehensive.

### Weaknesses
1. The proposed metric Leak@K requires the unlearned LLM to generate multiple responses for each prompt, which significantly increases the computational cost of evaluation.

2. The idea of using probabilistic decoding to evaluate model memorization is not new, as it has been explored in prior works [1][2].

3. The overall contribution of the paper appears limited, as it primarily introduces a new evaluation metric for unlearning without proposing a corresponding unlearning method or broader framework.



[1] Quantifying Memorization Across Neural Language Models. ICLR 2023.
[2] A Probabilistic Perspective on Unlearning and Alignment for Large Language Models. ICLR 2025.

### Questions
please see weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission points out a critical problem that existing large language model (LLMs) unlearning methods do not really make LLMs forget the knowledge. The greedy coding setting does not really reflect the unlearning methods’ performance in real-world settings. This paper proposes a metric named leak@k to facilitate the evaluations of LLM unleaning.

### Strengths
This paper has the following strengths:
- This paper is clearly written and easy to read.
- This paper identifies a practical and important limitation in evaluating large language model unlearning.
- The proposed metric demonstrates effectiveness in facilitating LLM unlearning.

### Weaknesses
This paper has the following weaknesses:
- The evaluated models and methods are limited.
- There is a lack of insight into why these unlearning methods fail to perform effectively outside the greedy setting.

### Questions
I have the following questions/suggestions:
- The reference formatting appears to be incorrect. It would be better for citations to follow the Name (Year) or (Name, Year) styles as appropriate to the context, rather than using only the Name (Year) format.
- Evaluate on more model sizes like 13B scale.

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
4

### Summary
The paper proposes leak@k, a novel meta-evaluation metric designed to measure the LLM unlearning efficacy by quantifying how often unwanted knowledge appears when generating under probabilistic decoding. Using this metric, the authors conduct a large-scale evaluation of unlearning efficacy across three widely used LLM unlearning benchmarks: TOFU, MUSE, and WMDP. The analysis reveals that, although existing unlearning methods appear effective under deterministic greedy decoding, sensitive information frequently resurfaces when standard sampling strategies are used. These findings demonstrate that current unlearning techniques commonly fail to achieve true forgetting in practice, highlighting the need for more reliable and principled approaches to LLM unlearning.

### Strengths
- [S1] **Interesting and relevant direction.** The paper addresses a timely and important issue in LLM research, the reliability of unlearning, and provides a new perspective by re-examining prior work under probabilistic decoding. This direction is valuable given the growing importance of safe and compliant model deployment.
- [S2] **Comprehensive experimental coverage.** The authors perform extensive empirical studies across multiple established benchmarks (TOFU, MUSE, and WMDP), offering a fairly broad assessment of existing unlearning techniques under different decoding conditions.
- [S3] **Clarity and structure.** The paper is generally well-written and easy to follow. The motivation, setup, and presentation of results are clear and logically connected.

### Weaknesses
- [W1] **Misalignment with the goal of unlearning.** The paper’s overall analysis appears misaligned with the conventional goal of unlearning, which is to make the unlearned model approximate the retain model, rather than simply avoid producing the correct answer. Accordingly, the Retrain curve in Figure 1 should be interpreted as a performance upper bound that effective unlearning methods should aim to approach, yet this interpretation is never discussed. In several cases (e.g., Tables 1, 7, and 8), even the retrained model generates the same “forgotten” answer, possibly due to hallucination or because the information legitimately exists in the retained data. Penalizing the unlearned model for such behavior seems conceptually inconsistent. This ambiguity raises a fundamental question of whether leak@k truly measures unlearning efficacy or merely captures aspects of refusal and alignment. Without a clearer definition of what constitutes a valid “leak,” the evaluation risks conflating unlearning failures with general response suppression or alignment artifacts.
- [W2] **Weak theoretical justification.** The theoretical section contributes little to supporting the main claims. The proof that the proposed estimator is unbiased is mathematically correct but does not meaningfully strengthen the argument for why leak@k is an appropriate or informative metric for unlearning. Unless the paper first establishes why leak@k captures the core desiderata of forgetting, the theoretical analysis feels disconnected from the empirical motivation. Furthermore, the remark that pass@k is a special case of leak@k is not clearly explained in terms of its practical significance for unlearning evaluation.
- [W3] **Metric choice and dependency on data format.** The evaluation relies on different similarity metrics (e.g., entailment score for TOFU, ROUGE for MUSE-News, mixed rubric for WMDP), and this choice appears strongly dependent on the specific format of the benchmark. This raises concerns about comparability and potential bias: for example, how short or long should answers be to justify using ROUGE vs. entailment score? What happens when the forget set includes paraphrased or reformatted variants of the same information? This dependence suggests that the evaluation pipeline may require dataset-specific manual tuning, reducing its generalizability and introducing potential evaluation bias.
- [W4] **Decoding parameter sensitivity.** The results show that increasing temperature and top-p generally increases leakage, but the paper does not clarify what decoding configurations are considered “standard” or “representative” for evaluating successful unlearning. For instance, is $T = 1.0$ chosen as an upper bound for theoretical reasons or empirical convenience? High temperature may also introduce randomness that confounds the distinction between resurfacing knowledge and random noise. Without clearer guidance or principled justification for which parameter setup is proper for evaluation, it is hard to interpret what constitutes a fair or meaningful evaluation under leak@k.

### Questions
- [Q1] The TOFU benchmark already measures the probability of generating paraphrased answers from the forget set, which implicitly accounts for uncertainty without relying on greedy decoding. How does leak@k differ conceptually from this evaluation protocol, and does it truly capture a distinct failure mode?
- [Q2] There are several instances where citations are formatted inconsistently (e.g., only the year is enclosed in parentheses). Please ensure that citations consistently include both author and year when appropriate.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper systematically demonstrates a critical vulnerability in current LLM unlearning techniques, arguing that their perceived effectiveness is an artifact of flawed evaluation protocols based on deterministic (greedy) decoding. The authors posit that these methods fail to achieve robust forgetting in practical scenarios utilizing probabilistic sampling, introducing a novel meta-metric, $leak@k$, to quantify the expected worst-case information leakage ($S_j$) across $k$ generations. Through a large-scale empirical study on standard benchmarks (TOFU, MUSE-News, WMDP), they show that $leak@k$ scores for state-of-the-art unlearning methods rise sharply with $k$, proving that sensitive information reliably resurfaces and concluding that current techniques offer merely an "illusion of forgetting" rather than true knowledge removal.

### Strengths
1. The paper is well-written; its logical structure and clear organization make the core arguments easy to follow and understand. Its primary and most significant contribution lies in the clear and timely identification of a critical vulnerability in current LLM unlearning evaluation. The authors rightly argue that the field's widespread reliance on deterministic (greedy) decoding fosters a misleading "illusion of forgetting," which obscures substantial risks present in real-world probabilistic sampling scenarios. I really thank the authors who tackle this important limitation of the existing LLM unlearning benchmark. Recognizing and articulating this gap is a major contribution, as it challenges the validity of prior claims and sets a new, more rigorous standard for evaluating robust unlearning.

2. To formally address this evaluation gap, the paper introduces $leak@k$, a novel and principled meta-metric that is both original and technically sound. By formalizing the evaluation as the expected worst-case leakage over $k$ samples, $leak@k := \mathbb{E}[\max_{1 \le j \le k} S_j]$, the authors provide a flexible and well-defined tool for the community. The quality of this contribution is further reinforced by the development of a low-variance unbiased estimator, demonstrating a high degree of statistical rigor.

3. The paper's empirical validation is of exceptional quality, marked by its comprehensive scope and clarity. The authors conduct a large-scale, systematic study across three diverse benchmarks (TOFU, MUSE-News, WMDP) and a full suite of state-of-the-art unlearning methods. The results are presented unambiguously and provide compelling, data-driven evidence that all evaluated techniques consistently fail under this more realistic probabilistic assessment, thereby strongly supporting the paper's central message.

### Weaknesses
While I strongly agree with the paper's major contribution, the following weaknesses make it difficult to evaluate the work more highly:

1. While the paper is generally well-structured and the prose is clear, the inconsistent use of citation commands (e.g., \cite vs. \citep) detracts from its professional polish. A thorough revision to ensure appropriate and consistent citation formatting throughout the manuscript is necessary to improve overall readability.

2. A significant weakness is the omission of several recent and highly relevant LLM unlearning analysis/algorithm papers, including numerous works from ICLR 2025 [1, 2, 3, 4]. I believe the authors should discuss with the existing LLM unlearning papers that pointed out the limitations in LLM unlearning [1,2]. Also, to make a compelling case, the paper's claims must be validated against a more diverse state of the art. For instance, methods like [3] and [4] reported exceptional performance on the TOFU benchmark, and [4] specifically demonstrated that using LoRA for unlearning offers significant advantages in both successful unlearning results and cost-efficiency. It is essential to investigate whether the probabilistic leakage issue identified by the authors persists even in these more advanced and robust unlearning techniques.

3. The paper's focus on the worst-case score (maximum leakage) is well-justified for safety and privacy. However, the analysis misses an opportunity to provide a more complete picture of the failure mode by also reporting the average score on the forget set (i.e., $\mathbb{E}[S_j]$) under probabilistic decoding. Presenting this average leakage alongside the $leak@k$ (worst-case) score would offer a more nuanced diagnostic, allowing a distinction between models that leak catastrophically but rarely versus those that leak more moderately but consistently. This additional statistic would enrich the understanding of how different unlearning methods fail.

[1] Catastrophic Failure of LLM Unlearning via Quantization, ICLR 2025

[2] Towards Effective Evaluations and Comparisons for LLM Unlearning Methods, ICLR 2025

[3] LLM Unlearning via Loss Adjustment with Only Forget Data, ICLR 2025

[4] Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs, ICLR 2025

### Questions
I have summarized my main weaknesses/questions in the Weaknesses section, so please refer to that content. Below are additional questions:

1. The proposed evaluation metric, which requires $n=200$ samples per prompt for a stable $\hat{leak@k}$ estimate, is necessarily more computationally expensive than the greedy-decoding baseline. Could the authors provide a quantitative analysis of this evaluation cost overhead? Discussing this practical trade-off would significantly strengthen the paper's assessment of the metric's viability for large-scale benchmarking.

2. The extensive experiments compellingly demonstrate that all evaluated unlearning methods remain vulnerable to probabilistic leakage. Based on the insights gained from designing $leak@k$ and observing these failure modes, what advice or intuition can the authors offer for designing future unlearning algorithms? Specifically, what properties should a truly robust algorithm possess to succeed under this more realistic evaluation? A discussion on this would be highly valuable for guiding future research in the field.

### Soundness
2

### Presentation
3

### Contribution
3
