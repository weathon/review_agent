# Evaluating Cross-Modal Reasoning Ability and Problem Characteristics with Multimodal Item Response Theory

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Multimodal Large Language Models (MLLMs) have recently emerged as general architectures capable of reasoning over diverse modalities.
Benchmarks for MLLMs should measure their ability for cross‑modal integration. However, current benchmarks are filled with shortcut questions, which can be solved using only single modality, and thereby yielding unreliable rankings. For example, in vision-language cases, we can find the correct answer without either the image or the text. These low-quality questions unnecessarily increase the size and computational requirements of benchmarks. We introduce a multi-modal and multidimensional item response theory framework (M3IRT) that extends classical IRT by decomposing both model ability and item difficulty into image‑only, text‑only, and cross‑modal components. M3IRT estimates cross‑modal ability of MLLMs and each question’s cross‑modal difficulty, enabling compact, high‑quality subsets that better reflect multimodal reasoning. Across 24 VLMs on three benchmarks, M3IRT prioritizes genuinely cross‑modal questions over shortcuts and preserves ranking fidelity even when 50\% of items are artificially generated low‑quality questions, thereby reducing evaluation cost while improving reliability. M3IRT thus offers a practical tool for assessing cross‑modal reasoning and refining multimodal benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a key deficiency in current benchmarks for Multimodal Large Language Models (MLLMs): the prevalence of "shortcut questions" that can be answered using only a single modality, such as text or image alone. This flaw leads to unreliable model rankings and unnecessarily increases evaluation costs due to large benchmark sizes. To resolve this, the authors propose a novel framework called Multimodal and Multidimensional Item Response Theory (M³-IRT). This approach extends classical Item Response Theory (IRT) by decomposing both the model's ability and the question's difficulty into three latent components: image-only, text-only, and cross-modal integration.

This decomposition allows the framework to precisely identify high-quality questions that genuinely require cross-modal reasoning while filtering out low-quality shortcuts. The method was validated through experiments on 24 Vision-Language Models (VLMs) across three benchmarks: MMMU, MATHVISTA, and SEED-BENCH. The results demonstrate that M³-IRT is robust against contamination, maintaining ranking fidelity even when benchmarks contain up to 50% artificially generated low-quality questions. Furthermore, the framework enables the creation of compact, high-quality benchmark subsets that can accurately reproduce the full ranking using only a small fraction of the questions, thereby significantly reducing computational overhead while improving evaluation reliability.

### Strengths
1. The paper's primary strength lies in its novel framework, M³-IRT, which moves beyond simple accuracy scores to offer a more granular and interpretable evaluation.

2. The framework enables the creation of compact, high-quality benchmark subsets that significantly reduce the computational cost of evaluation.

3. The paper effectively demonstrates the method's resilience to the low-quality and "shortcut" questions that are common in existing benchmarks.

### Weaknesses
1.  The decomposition of a model's ability, theta ($\theta$), is a heuristic and oversimplified mathematical model. It operates on the assumption that the distinct abilities (e.g., image, text, cross-modal) are independent and linearly additive. However, a model's actual reasoning ability is complex. For instance, a model might leverage textual concepts to guide its attention within an image, creating a synergistic effect where cross-modal ability mutually enhances the image ability. Such complex, non-linear interactions cannot be captured by the current linear additive framework.

2.  Given that numerous parameters are optimized via SGD, the stability of $\theta$ across benchmarks of varying scales has not been adequately validated. A critical question remains unanswered: if the number of questions were to substantially increase or decrease, would a model's ability score exhibit significant fluctuations? This lack of demonstrated robustness raises concerns about the reliability of the scores when the evaluation context changes.

3.  The model's ability ($\theta$), question difficulty ($b$), and discrimination ($a$) are all optimized simultaneously. Consequently, the difficulty of a question is inherently influenced by the abilities of the models in the test cohort. This renders all metrics as **relative scores** within the context of a specific group (24 VLMs in this paper), rather than absolute measures. If the composition or average capability of the VLM cohort were to change, the perceived difficulty of the questions would also shift, raising concerns about the potential for instability in the evaluation framework.

### Questions
1. The $\theta_{base}$ in Supp. Table 2: Mathvista shows that GPT-4o's base is 0.00 and Llama-3.2-11B is 0.10, which is weird. Can authors explain this phenomenon?

2. Given that the evaluation scores are relative to the tested models, how do you address the potential instability of these scores as the model evolves?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper discusses the assessment of Multimodal Large Language Models (MLLMs). The authors observe that some multimodal problems in conventional datasets or benchmarks can be solved using only one modality. This makes current evaluation efforts of Multimodal Large Language Models less efficient and less reliable. The authors then propose their method, M3-IRT, which estimates the cross‑modal ability of MLLMs and each question’s cross‑modal difficulty, enabling compact, high‑quality subsets that better reflect multimodal reasoning.

### Strengths
1. This paper studies a very important problem in Multimodal Large Language Models, and the authors' observations about the weaknesses of existing evaluation efforts are reasonable.
2. This paper provides detailed descriptions of their evaluation framework. There are a lot of figures to visualize the results of evaluation.

### Weaknesses
1. The problem that this paper has pointed out (prior evaluation efforts use multimodal problems that can be solved with only one of the modalities) has been studied previously. This paper did not provide proper reference to these prior works. For example, MMEvalPro [1] is a recently proposed dataset with manually labeled questions to mitigate the problem.
2. The results are mostly displayed in the figures, while I expect more accurate numbers to be displayed in tables. While figures are effective for visualizing high-level trends and making qualitative comparisons, they are insufficient for a rigorous and reproducible scientific paper. It is difficult, and in some cases impossible, for the reader to extract the precise performance metrics, which is essential for understanding the exact magnitude of the reported improvements and for future comparative analysis by other researchers.
3. This paper should include a more comprehensive discussion of related works in MLLM evaluation, including [2-3].

[1] MMEvalPro: Calibrating Multimodal Benchmarks Towards Trustworthy and Efficient Evaluation
[2] Multifaceted Evaluation of Audio-Visual Capability for MLLMs: Effectiveness, Efficiency, Generalizability and Robustness
[3] Revisiting multi-modal llm evaluation

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To address unreliable rankings and high costs in MLLM evaluation caused by shortcut questions answerable from a single modality, this paper introduces the M³-IRT and M²-IRT frameworks. These extend classical Item Response Theory (IRT) by decomposing both model abilities and question difficulties into image-only, text-only, and cross-modal components. This allows for quantifying a model's cross-modal reasoning and an item's cross-modal demand. Experiments show the framework effectively prioritizes genuine cross-modal questions, faithfully reproducing full-benchmark rankings with as little as a 10% subset. It remains robust even when 50% of the benchmark is contaminated with low-quality items, significantly reducing evaluation costs while enhancing reliability.

### Strengths
1. The paper makes a contribution by addressing the critical challenge of "shortcut questions" in multimodal benchmarks. It provides a systematic solution that enhances the reliability of evaluations while simultaneously reducing computational costs.
2. The paper effectively targets two major pain points in current multimodal evaluation: unreliable model rankings caused by shortcut question contamination, and the high computational cost associated with large-scale benchmarks. The proposed M³-IRT framework offers an efficient and effective solution to both issues.
3. By decomposing model abilities into image-only, text-only, and cross-modal components, it offers valuable and interpretable insights into the specific strengths and weaknesses of different MLLMs, moving beyond a single, monolithic accuracy score.

### Weaknesses
1. The model's core assumption of linear decomposition for abilities and difficulties might oversimplify the complex, potentially non-linear interactions that occur during cross-modal reasoning. 
2. The interpretability of the estimated parameters such as cross-modal difficulty is derived purely from model performance patterns and lacks external validation against human cognitive judgments of what constitutes a cross-modal task.
3. It is noted that the paper generates artificial low-quality questions through full swapping of images or text, which effectively simulates extreme scenarios involving complete modal mismatch. That said, it may be worth considering whether this approach fully covers the more prevalent and subtle shortcut features commonly observed in real-world multimodal benchmarks. In practical contexts, shortcuts often exhibit characteristics of concealment and diversity. Given that the framework’s ability to filter such subtle shortcuts has not yet been evaluated, it might be valuable to further explore its generalizability when applied to real-world benchmarks.

### Questions
Have you tested any non-linear variants? If not, how sensitive are your conclusions to the linearity assumption?

In addition, beyond swapping-based corruption, does your filter still identify low-quality items under subtler artifacts?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors proposed M^3-IRT, a multi-modal and multidimensional item response theory framework to evaluate cross-modal reasoning ability of Multimodal Large Models (MLLMs). Multiple experiments were conducted with several VLMs to show that M^3-IRT could reduce evaluation cost while improving reliability.

### Strengths
1. Both the motivation of decomposing model ability and item difficulty and the proposed M^3-IRT framework make sense and are technically sound to me. Even though it builds upon the classical IRT framework, the extension to MLLMs makes lots of sense and could provide more nuanced evaluation and improved reliability with marginal cost.
2. Extensive experiments were conducted to justify the effectiveness and efficiency of the proposed framework.
3. Writing is good and easy to follow.

### Weaknesses
1. Multiple hyper-parameters were introduced by the proposed model. It would be better to provide more discussion on: a. how accurate the estimation of these parameters based on the method in section 4.4? b. sensitivity of the hyper-parameters; c. how many data are needed? d. cost of the estimation.
2. The proposed framework mainly filters the items to be tested rather than help curate new dataset. While evaluation might be costly, we only need to run it once for every new model. With the lowering  trends of inference cost of LLMs, I am not sure whether the gain on reducing some eval examples would outweigh the cost of train a separate M^3-IRT model.

### Questions
Please refer to the weakness section and provide more justifications accordingly.

### Soundness
3

### Presentation
3

### Contribution
3
