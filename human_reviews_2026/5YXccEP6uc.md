# SciTS: Scientific Time Series Understanding and Generation with LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
The scientific reasoning ability of large language models (LLMs) has recently attracted significant attention. Time series, as a fundamental modality in scientific data, presents unique challenges that are often overlooked in current multimodal LLMs, which either encode numerical sequences as text or convert them into images. Such approaches may be insufficient for comprehensive scientific time series understanding and generation. Existing unified time series models typically specialise in either forecasting or analysis, and their effectiveness on non-periodic, heterogeneous scientific signals remains unclear. To address these gaps, we introduce SciTS, a benchmark spanning 12 scientific domains and 43 tasks, with over 50k+ instances, both univariate and multivariate signals ranging from $10^0$ to $10^7$ in length and up to 10~MHz in frequency. We benchmark 17 models, including text-only LLMs, multimodal LLMs, and unified time series models, and find that general-purpose LLMs exhibit stronger generalisability than specialised time series models, while representing time series as text or images limits their performance due to excessively long sequences and loss of numerical precision, respectively. We then introduce TimeOmni, a working example to explore insights into how LLMs can be extended to handle scientific time series while remaining compatible with general-purpose LLM training. This work fills a gap in both dedicated benchmarks and illustrative frameworks for scientific time series, paving the way for LLMs to understand and generate complex temporal scientific data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SciTS, a large-scale benchmark for evaluating the ability of LLMs to understand and generate scientific time series across 12 domains, 43 tasks, and over 52k instances. It further proposes TimeOmni, a modular framework that integrates explicit temporal encoding, router-based patch experts, and patch reprogramming for unifying diverse scientific signals within LLM architectures.

### Strengths
1. SciTS fills a gap by offering a unified high-quality evaluation suite for scientific time series
2. The benchmark includes 17 baselines across modalities with consistent metrics, which is good for cross-domain generalization and model behavior.

### Weaknesses
1. TimeOmni uses fine-tuning while others are evaluated purely zero-shot, which would impact strict comparability. An ablation without fine-tuning would strengthen the claim.
2. No ablation for key modules, like router, patch reprogramming, and expert families.
3. It remains unclear how TimeOmni handles very long or high-frequency sequences beyond benchmark scales.
4. Benchmark provenance and overlap with LLM pre-training corpora are not fully documented.

### Questions
1. How is normalization handled across domains with very different frequency/time scales?
2. Were all non-TimeOmni baselines strictly zero-shot? If so, could you include a variant of TimeOmni under the same condition?
3. How stable and interpretable is the router-based patch expert mechanism during inference?
4. Do you plan to release detailed data provenance and licensing information to ensure benchmark sustainability?
5. Could the framework extend to spatio-temporal or higher-dimensional scientific data (e.g., radar or 3D simulations)?

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
The paper collects a comprehensive set of scientific related multimodal time series datasets, spanning various tasks including forecasting, classification, anomaly detection, etc. It benchmarks text-only LLMs, multimodal LLMs, and unified time series models and addresses the limitations of existing approaches by its proposed TimeOmni. TimeOmni fine-tunes an LLM by aligning time series representations and text representations, with separate output heads for different tasks. Extensive experiments across the benchmark show that TimeOmni achieves strong performance relative to all three categories of existing methods.

### Strengths
1. The paper contributes a large-scale multimodal time series benchmark covering diverse tasks and domains.

2. The paper conducted broad evaluation comparing both text-only LLMs, multimodal LLMs, and unified time series models.

3. The authors propose a new model that aligns time series and text, and show through extensive experiments that the model outperforms all three types of existing methods.

### Weaknesses
1. Is TimeOmni trained and evaluated per domain, or is it trained jointly across all domains and tasks? The compared models, even for those open-source models with the same scale, are evaluated in a zero-shot setting, so it seems unfair for those models as TimeOmni has been fine-tuned on the target domains. I may have missed this part, but have the authors tried any out-of-domain testing?

2. The current design does not support dynamic-length generation, which limits flexibility and scalability compared to sequence decoders that can have variable-length outputs.

3. Apart from per-domain performance, are there any results on per-task performance? For example, how does TimeOmni compare with current time series models on the forecasting task? Practitioners may not care about having a general model that can handle all the tasks but more on the performance for a specific task of their interest.

### Questions
1. Have the authors explored RL training after supervised fine-tuning?

2. For multivariate time series with dimensions up to 58 in Neuroscience, would the current representations make the input length very long? 

3. How to compute the scores of a task if a part of samples fail this task? For example, did the authors impute zeros to compute MAPE for those failed samples?

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
The paper titled “SciTS: Scientific Time Series Understanding and Generation with LLMs”  introduces SciTS, a new benchmark for scientific time series understanding and generation, spanning a number of disciplines, tasks and data with diverse signal characteristics. The authors benchmark 17 models, revealing that general-purpose LLMs often show better generalization than specialized time series models, but their performance is limited when time series are converted to text (due to long sequences) or images (due to loss of precision). To address these challenges, the paper proposes TimeOmni, an LLM-based framework that explicitly models temporal dynamics and supports both time series understanding and generation, achieving the top rank on the challenging SciTS benchmark.

### Strengths
The benchmark provided is pretty exhaustive for scientific time series. It consists of 52,056 instances spanning 43 domain-specific tasks across 12 scientific disciplines. It has has diversity types of data including both univariate and multivariate. It also includes various task types such as anomaly detection, classification, multiple-choice question answering (MCQ), event localization, forecasting, imputation, and synthesis

The proposed TimeOmni demonstrates its effectiveness by achieving the highest overall ranking on the challenging SciTS benchmark, underscoring the advantage of its approach.

### Weaknesses
The paper is primarily a benchmark contribution (SciTS), with the proposed TimeOmni methodology being a secondary, and arguably incremental, focus. While the benchmark is vast, this heavy reliance on data creation means the paper's core scientific novelty may be perceived as low, as the methodological innovation is not groundbreaking.

The creation of the SciTS benchmark largely involves the collection and curation of existing open-source datasets and data from scientific domain websites, alongside some numerical simulation methods. While the combination, annotation, and unifying under a prompt-based format are novel contributions , the underlying raw time series signals themselves are largely drawn from pre-existing sources.

The proposed TimeOmni framework is an adaptation of existing LLM componentsFor instance, it uses a Patch Reprogramming module which is a concept previously presented in Time-LLM (Jin et al., 2024),  the router and patch family are basically selective resizing. 

The title, "SCITS: SCIENTIFIC TIME SERIES UNDERSTANDING AND GENERATION WITH LLMS," is misleading. The title should use the word benchmark because that is what it is. Potential suggestion:
"SCITS: A NEW BENCHMARK FOR SCIENTIFIC TIME SERIES UNDERSTANDING”

### Questions
na

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SCITS (Scientific Time Series Understanding and Generation) — a comprehensive benchmark encompassing 12 scientific disciplines, 43 tasks, and over 50,000 instances — designed to evaluate the capability of large language models (LLMs) in processing scientific time series data. The authors observe that existing multimodal LLMs either encode time series as text (leading to excessive sequence lengths) or as images (losing numerical precision). To bridge this gap, the paper proposes TimeOmni, an LLM-compatible framework that explicitly models temporal dynamics through patch experts and adaptive routing, enabling both understanding and generation of time series while remaining compatible with general LLM training. Extensive benchmarking across 17 models—including GPT-5, Gemini-2.5, Qwen3, and Moirai—demonstrates that TimeOmni consistently outperforms both general LLMs and specialised time-series models across most scientific domains.

### Strengths
1. The introduction of SCITS represents a major step forward for LLM-based scientific data understanding. The benchmark’s diversity—covering astronomy, neuroscience, meteorology, physiology, and more—significantly broadens the scope beyond traditional forecasting or anomaly detection benchmarks.
2. The dual emphasis on understanding (classification, QA, anomaly detection) and generation (forecasting, imputation, synthesis) is novel, establishing SCITS as arguably the first unified evaluation for both reasoning and signal generation.
3. The manuscript is clearly written, logically structured, and well-illustrated.
4. TimeOmni not only achieves full task coverage but also top-1 average ranking across nearly all disciplines, demonstrating both robustness and generality.

### Weaknesses
1. The paper briefly mentions recent works like Time-LLM and ChatTS, but deeper discussion of conceptual differences and computational efficiency trade-offs would help position TimeOmni more clearly in the landscape.
2. The paper evaluates models in a zero-shot setting only. While this fairly tests generalisation, it leaves open the question of how much performance could improve with lightweight adaptation, e.g., LoRA or task-specific finetuning.
3. Although the patch-expert routing and reprogramming modules are key innovations, the paper does not include ablations isolating their contributions. Quantifying improvements from each would strengthen the architectural claims.

### Questions
1. Given that the router dynamically selects patch experts, what is the computational overhead compared to standard LLM inference? Is the training cost comparable to multimodal extensions (e.g., audio or vision encoders)?
2. How does TimeOmni perform under few-shot or domain-specific fine-tuning? Could small-scale adaptation bridge the performance gap between open-source and closed-source LLMs?
3. Could the authors provide quantitative ablation results on the router and patch-reprogramming components to confirm their necessity?

### Soundness
3

### Presentation
3

### Contribution
3
