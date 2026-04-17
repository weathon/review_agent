# XFacta: Contemporary, Real-World Dataset and Evaluation for Multimodal Misinformation Detection with Multimodal LLMs

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
The rapid spread of multimodal misinformation on social media calls for more effective and robust detection methods.
Recent advances leveraging multimodal large language models (MLLMs) have shown the potential in addressing this challenge. 
However, it remains unclear exactly where the bottleneck of existing approaches lies (evidence retrieval v.s. reasoning), hindering the further advances in this field. 
On the dataset side, existing benchmarks either contain outdated events, leading to evaluation bias due to discrepancies with contemporary social media scenarios as MLLMs can simply memorize these events, or artificially synthetic, failing to reflect real-world misinformation patterns.
Additionally, it lacks comprehensive analyses of MLLM-based model design strategies.
To address these issues, we introduce XFacta, a contemporary, real-world dataset that is better suited for evaluating MLLM-based detectors. We systematically evaluate various MLLM-based misinformation detection strategies, assessing models across different architectures and scales, as well as benchmarking against existing detection methods.
Building on these analyses, we further enable a semi-automatic detection-in-the-loop framework that continuously updates XFacta with new content to maintain its contemporary relevance.
Our analysis provides valuable insights and practices for advancing the field of multimodal misinformation detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces XFACTA, a contemporary, real-world dataset for evaluating multimodal misinformation detection with multimodal large language models (MLLMs). Unlike existing datasets which are outdated, synthetic, or biased. XFACTA collects 2,400 real posts (1,200 real + 1,200 fake) from X/Twitter, spanning January 2024 to April 2025. The authors benchmark closed-source (GPT-4o, Gemini-2.0-Flash) and open-source (Qwen-VL) MLLMs under different evidence retrieval and reasoning strategies, comparing their results with prior detectors such as MMFakeBench, LEMMA, and Sniffer.

### Strengths
1. The dataset intentionally focuses on recent events (post-2024), avoiding overlap with pre-training corpora of MLLMs like GPT-4o and Gemini. This makes XFACTA a valuable benchmark for testing real-time misinformation detection, where memorization bias is minimized.
2. The paper systematically investigates multiple dimensions, i.e., evidence retrieval strategies, reasoning paradigms (CoT, multi-step reasoning, etc.), and model architectures, offering insights into where MLLMs struggle (retrieval vs. reasoning bottlenecks).

### Weaknesses
1. Unclear labeling process for the three misinformation types. The labeling of Deepfakes, Image Out-of-Context, and Text Misleading relies on “flagging posts by journalists or X Community Notes,” but the paper does not specify who performed the final verification or how inter-annotator reliability was measured.
2. While XFACTA’s realism is commendable, 2,400 samples (1,200 fake) is relatively small compared to datasets. Such a small dataset may not adequately represent the diversity of misinformation strategies on social media.
3. Prompt sensitivity and fair comparison issues. The performance gap in Table 7 between GPT-4o (ours: 88.8%) and other GPT-4o-based systems (e.g., LEMMA 77.3%, MMFakeBench 68.2%) may stem partly from prompt-format differences, not purely methodological superiority. A sensitivity analysis of prompts quality is necessary to ensure that the improvements are reproducible rather than prompt-dependent.
4. Limited depth in experimental validation. Although multiple pretrained models are tested, the study lacks fine-tuning experiments or ablation analyses.
5. Unclear statistics on overlapping or multi-label posts. The paper states that “each fake post may be assigned one or more error-type labels,” but the extent of label overlap is not quantified.

### Questions
6. Unclear statistics on overlapping or multi-label posts. The paper states that “each fake post may be assigned one or more error-type labels,” but the extent of label overlap is not quantified.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces the XFacta dataset, which evaluates the abilities of MLLMs to fact-check posts on social media containing both text and image. The dataset is collected from X/Twitter using community notes and reputable sources to categorize the posts into real and fake. The posts are then further categorized into "deepfakes", "image out of context", and "text misleading". An analysis of several fact-checking strategies is also presented.

### Strengths
1. Evaluating MLLMs on their ability to detect misinformation is an extremely important problem. Specifically, the multimodal aspect of misinformation is a challenging problem to solve and will require excellent multimodal reasoning from the MLLM.

2. The dataset presented is likely to be useful for practitioners seeking to construct AI systems for flagging misinformation online.

### Weaknesses
1. There are several other similar benchmarks for evaluating capabilities of MLLM to detect misinformation (cited in the paper itself). The authors argue that the dataset consists of contemporary posts unlike other benchmarks. In my opinion, this is insufficient because the presented benchmark may also get outdated soon in a couple of years. It would be better to construct a continuously evolving benchmark for this (similar to LiveBench/LiveCodeBench) .

2. It would also be good to include the URLs for the retrieved evidence (E_i, E_t) for each post in the dataset as this would enhance reproducibility + insulate the benchmark from future events that may change the veracity of the posts in question.

3. The evaluation of the different strategies for detecting misinformation is a bit perplexing. In particular, it is not clear why the strategies cannot be utilized simultaneously. For example, an MLLM agent may use start off using strategy 1, then if it's inconclusive it may move on to other strategies. Being constrained to use one strategy seems artificial. Also, it seems that the performance of the strategies is somewhat data dependent, it should be possible to construct datasets where a particular strategy is favored. I do not quite understand the usefulness of the insights obtained from this analysis.

### Questions
Please address the weakness above

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces XFACTA, a contemporary, real-world dataset for multimodal misinformation detection, composed of social media posts to avoid temporal bias and model memorization. XFACTA enables a rigorous analysis of MLLM-based detection methods, clarifying the relative impact of evidence retrieval versus reasoning. The study evaluates various MLLM architectures and existing approaches, providing key insights into effective design strategies. Furthermore, a semi-automatic detection-in-the-loop framework is proposed to continuously update the dataset, ensuring long-term relevance. XFACTA and the accompanying analysis serve as a robust benchmark for advancing multimodal misinformation detection research.

### Strengths
1 Introduces XFACTA, a novel dataset with real-world misinformation samples from X, avoiding temporal leakage and memorization bias common in outdated benchmarks.

2 Provides a clear disentanglement of the roles of evidence retrieval and reasoning in detection performance, offering actionable insights into model bottlenecks.

3 Proposes a semi-automatic detection-in-the-loop framework that continuously updates the dataset with new verified cases, ensuring long-term relevance and scalability.

### Weaknesses
1 While the dataset is curated with recent and real-world examples, its reliance on Twitter (X) as the primary source follows a well-established practice in prior work. The advancement over existing datasets is incremental.

2 The proposed semi-automatic detection-in-the-loop framework lacks a comprehensive methodological description. A clearer overview of the full detection pipeline, particularly with a schematic illustration or step-by-step workflow, would improve transparency and reproducibility.

3 The evaluation does not include direct comparison with human fact-checkers. As a result, it remains unclear how closely the system’s reasoning performance approaches human-level judgment in complex multimodal misinformation scenarios.

### Questions
1 Given that many existing multimodal misinformation datasets are also sourced from Twitter (X), what specific characteristics or design choices in XFACTA differentiate it from prior efforts, and how do these contribute to measurable improvements in evaluating MLLM-based detectors?

2 Could the authors provide a more detailed overview of the detection-in-the-loop pipeline, ideally with a schematic diagram or algorithmic description, to clarify the interaction between automated detection, human verification, and data update processes?

3 How does the system’s performance compare to that of human fact-checkers on the same multimodal misinformation tasks, particularly in cases requiring nuanced reasoning across text and image modalities?

### Soundness
2

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
2

### Summary
The paper introduces XFACTA, a contemporary, real-world multimodal misinformation dataset sourced from X/Twitter spanning Jan 2024–Apr 2025. Building on this dataset, the authors disentangle two core bottlenecks: evidence retrieval vs. reasoning. They design eight retrieval strategies plus two evidence post-processing steps (domain filtering, evidence extraction), and systematically compare multiple MLLMs across reasoning paradigms (CoT, prompt ensemble, self-consistency, multi-step). They also present a detection-in-the-loop data expansion pipeline and validate OOD generalization on timely Snopes cases.

### Strengths
1. The dataset’s contemporaneity (2024–2025) addresses memorization bias in current LLMs, ensuring fair evaluation beyond training knowledge leakage.
2. he authors use optimal transport alignment and topic matching to reduce distributional bias between real and fake samples, producing a credible and well-balanced corpus.
3. The dual-axis study (evidence retrieval vs. reasoning) provides a structured framework that clarifies where current MLLMs fail,  representing a crucial step toward principled detector design..

### Weaknesses
1. The process and consistency of manual review and evidence annotation are described briefly; inter-annotator agreement metrics and annotation protocols are missing.
2. The authors emphasize contemporaneity but do not systematically quantify how model performance degrades on older vs. newer misinformation, which would strengthen their central claim.
3. The annotations and the retrieved evidence may share the same sources (e.g., journalists’ debunking pages). Without strict separation between training and evaluation, this overlap can create shortcuts. The paper does not describe systematic deduplication and domain isolation procedures.
4. The paper aligns image distributions using optimal transport, but it lacks ablations with no alignment or alternative alignment methods to demonstrate the net contribution of this step to fairness.

### Questions
1. How was annotation quality controlled?  Were multiple annotators involved, and if so, what was the inter-annotator agreement (e.g., Cohen’s κ)?
2. Have you performed semantic deduplication across retweets/aggregator posts and near-duplicate image removal?  How do you ensure there is no leakage between Dev and Test?
3. How much fairness improvement is contributed by image distribution alignment (via optimal transport) ?  Have you conducted ablations or comparisons with alternative alignment methods ?

### Soundness
3

### Presentation
3

### Contribution
3
