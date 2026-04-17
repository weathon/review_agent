# Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Fully open multimodal large language models (MLLMs) currently lag behind proprietary counterparts, primarily due to a significant gap in data quality for supervised fine-tuning (SFT). 
Existing open-source datasets are often plagued by widespread noise and a critical deficit in complex reasoning data, such as Chain-of-Thought (CoT), which hinders the development of advanced model capabilities.
Addressing these challenges, 
our work makes three primary contributions.
First, we introduce Honey-Data-15M, a new SFT dataset comprising approximately 15 million QA pairs, processed through multiple cleaning techniques and enhanced with a novel dual-level (short and long) CoT enrichment strategy.
Second, we introduce HoneyPipe, the data curation pipeline, and its underlying framework DataStudio, providing the community with a transparent and adaptable methodology for data curation that moves beyond static dataset releases. 
Finally, to validate our dataset and pipeline, we train Bee-8B, an 8B model on Honey-Data-15M. Experiments show that Bee-8B establishes a new state-of-the-art (SOTA) for fully open MLLMs, achieving performance that is competitive with, and in some cases surpasses, recent 
semi-open models such as InternVL3.5-8B. A comprehensive ablation study further dissects the impact of our data curation process, revealing that each stage provides significant performance gains across a wide range of benchmarks.
Our work delivers to the community a suite of foundational resources, including: the Honey-Data-15M corpus; the full-stack suite comprising HoneyPipe and DataStudio; training recipes; an evaluation harness; and the model weights. This effort demonstrates that a principled focus on data quality is a key pathway to developing fully open MLLMs that are highly competitive with their semi-open counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper introduces Bee, a long-context evaluation corpus focused on improving benchmark reliability and diagnostic granularity for long-context LLMs. To reduce any artifacts or contaminations from using synthetic or web-mined long-context datasets, they provide human-written and length-controlled contexts.This corpus covers four task types and also fine-grained position control (of beginning, middle, and end).
- The authors evaluate 15 open and closed LLMs on Bee and find that models generally retain strong short-context ability but degrade beyond 64K tokens. Also, it confirms the “lost-in-the-middle” phenomenon showing middle-context reasoning is worst and LLMs hallucinate under dense distractor settings, even when retrieval accuracy remains high.

### Strengths
- In terms of originality, the authors introduce a long-context corpus named Bee, combining human-authored text, controlled length scaling and multiple evidence positions. Bee also seems to provide a strong diagnostic resource for evaluating context fidelity in long-context settings. In terms of domain the dataset covers, since it includes multiple domains, including policy, law, science, manuals, it would serve as a value source in the long-context research.
- The data construction pipeline is clearly explained and includes human validation at every stage—authors filter GPT-generated distractors and manually verify evidence spans.

### Weaknesses
- While the authors emphasize “human-authored and contamination-free,” the process for verifying that test instances aren’t part of model pretraining data is not sufficiently detailed. No automatic overlap detection (e.g., MinHash or embedding-based retrieval) against known training corpora is reported to mitigate this and also the “human written” claim might not fully guarantee model-agnostic contamination, especially given the high proportion of UN and government reports in the pretraining datasets.
- Although the Bee corpus supports entity tracking and multi-hop reasoning, most experiments laid in the main paper seems to focus on factual recall and summarization accuracy while limited analysis of higher-order reasoning. This makes the contribution weaker and Bee is rather more shown to be retrieval-heavy than reasoning-intensive benchmark.
- The evaluation has rather heavy reliance on LLM-as-a-judge, which might introduce biases since the same model being evaluated also serves as evaluator.
- Also, in terms of reporting the LLM-as-judge performance, there is no inter-model agreement check or human correlation study. Although the authors seem to partially mitigate this by including ROUGE-L and BLEU, these metrics are weak at measuring factual alignment for long-form outputs.

### Questions
- The context range (8K-200K) for the Bee corpus is motivated by “scaling to extreme contexts” according to the authors but is there any theoretical justifications for this? Is 200K a representative threshold for real tasks?
- How many human annotators were used per instance? Also, what was the inter-annotator agreement or validation accuracy before filtering?
- Have the authors tested whether models fine-tuned on Bee generalize to other long-context corpora publicly availably, and conversely, how well do Bee-trained models perform on out-of-domain texts? (Or any future experimental plans?)
- [Minor] I’m not sure if the current primary area or the “datasets and benchmarks” area might be a better fit for the paper.

### Soundness
2

### Presentation
2

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
The authors tackle the problem of quality in SFT datasets, where previous contributions often have noise in the form of a lack of instruction following, factual inaccuracies, etc., for multimodal LLMs. The authors propose a data collection pipeline, contribute the data collected from the same, and train a model on the data.

## Honey 15M and HoneyPipe

The 15M dataset is curated using the proposed (automated) HoneyPipe pipeline. Stage 1 - the authors collect 24M image-text pairs from publicly available datasets and deduplicate them (using image hashing and text hashing for instructions - both approximate techniques). With the final image-instruction-response tuples, they assign a domain label (unclear how many tuples exist / how manual tagging was done for a million scale). Stage 2 - heuristics-based filtering with rules and using a powerful model for logical consistency filtering. Stage 3 - (partially applicable to the collected tuples) CoT is promoted and ensured by removing phrases discouraging it, and LLM as Judge to compare the new CoT and the original ground truth response. Stage 4 - Long CoT generated with different models. The authors end with the total Honey 15M. Figure 3  has a breakdown by categories.

## Bee-8B 

Authors train a Qwen3-8B + SigLip2 model. The training gradually moves from warmup to alignment to SFT to post-training RL.

# Results

The trained model performs really well across benchmarks, with Math & Reasoning gaining a lot of performance. Ablations suggest curation helped with math-based tasks and did not help specific performance in MMBench (with some loss). No-CoT also seems to give strong performance on DocVQA.

### Strengths
1. Solid contributions, clear presentation, and clean analysis from the authors; I really appreciate the details in the appendix.
2. The gains in performance and ablations clearly suggest the strengths of the data, especially with math/reasoning style benchmarks.

### Weaknesses
1. Lack of release details: it is unclear what the authors are releasing publicly, given that a lot of this work has strongly benefitted from public work. I strongly encourage authors to clarify this here and also encourage/ask if the authors can release checkpoints for each of the different stages - releasing these checkpoints would strongly improve the contributions beyond the final model/data and improve open science beyond improving model performance.
2. [Minor] Lack of reproducibility and ethics statements. The authors are strongly encouraged to write these statements.

### Questions
/ Please look at the questions in the weaknesses and summary above on unclear/missing details. I'm willing to bump my score to 8 once authors provide clear details.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the performance gap between fully open multimodal large language models (MLLMs) and their proprietary counterparts, attributing this gap primarily to a deficit in high-quality Supervised Fine-Tuning (SFT) data. The authors argue existing open datasets are noisy and lack complex reasoning data, such as Chain-of-Thought (CoT).

They make a  new, large-scale SFT dataset of approximately 15 million QA pairs, open source the data platform, and train an 8B model. Authors also benchmark Bee-8B on a wide array of tasks, claiming it establishes a new state-of-the-art (SOTA) for fully open MLLMs and achieves performance competitive with semi-open models , particularly on reasoning-heavy benchmarks like MMMU , MMStar , and CharXiv. Ablation studies are provided to demonstrate that both the noise filtering and the CoT enrichment stages contribute significantly to these performance gains

### Strengths
The paper's methodology for data curation is extensive, and the five-stage training recipe for the validation model is thorough. The ablation studies convincingly isolate the benefits of data filtering and CoT enrichment which supports the central thesis.

Paper is also well-written, clearly structured, and logically organized. The core thesis is stated upfront and defended throughout.

The primary bottleneck for open-source MLLM development is widely recognized as access to high-quality, large-scale SFT data. This paper delivers not only a 15-million-sample dataset specifically designed to address this (with a principled dual-CoT approach) but also the entire pipeline to create it.

### Weaknesses
- Both short‑ and long‑CoT enrichment hinge on model‑generated content; “fidelity verification” uses a Qwen‑family judge. The paper acknowledges Qwen3‑32B as the evaluator for some tasks (e.g., DocVQA), while Bee‑8B is trained on Qwen3‑8B, raising potential brand/self‑preference risks and evaluation coupling (same family, same stylistic priors). Empirically, LLM judges exhibit position, verbosity, and self‑enhancement biases; they correlate with humans but not uniformly across domains. A careful judge‑robustness analysis (randomized order, multiple independent judges, human spot‑checks, reporting agreement) is needed to reinforce the claims. For instance, the yourbench paper employs an ensemble of judges in order to reduce self-preference and increase reliability. A similar method, with multiple judges, can be employed to further strengthen the claims

- Although many benchmarks have automatic metrics or references, the paper also “enables LLM‑based judging on DocVQA” using a specific setup. Given known instabilities of LLM‑as‑judge (prompt sensitivity, order effects), stronger reporting (prompt template, seed settings, inter‑judge agreement, and correlation to human labels on a held‑out slice) is warranted.
- Given the breadth of sources and the emphasis on chart/document benchmarks, the paper should report automatic overlap/de‑dup statistics (hashes, text similarity, synthetic perturbed matching) between Honey‑Data‑15M and every evaluation set (MMMU/MMMU‑Pro, MMStar, LogicVista, DynaMath, CharXiv, OCRBench, etc.). Without this, some gains could come from near‑duplicates or stylistic priming. (The chosen benchmarks are widely used and public, which increases leakage risk.). If the authors have indeed already performed this, then they should report this.

### Questions
- How sensitive are your results to the choice of LLM judge? Please report metrics with (at least) two families of judges
- Where applicable, please report some CI / confidence intervals for the figures in Table 2
- For greater adoption in the community, for each source category in Figure 3, please provide a table with license terms, known restrictions (e.g., non‑commercial), and PII removal strategy. For OCRed textbooks/receipts and K‑12 materials, what steps ensure copyright compliance?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the issue of open-source multimodal models (MLLMs) lagging due to a lack of high-quality data. It introduces a high-quality fine-tuning dataset containing 15 million QA pairs. This dataset has undergone noise filtering and was enhanced using a "dual-level CoT" strategy. The paper also opens up the transparent data curation pipeline, HoneyPipe.

### Strengths
* Excellent performance, comprehensive, and open-sourced.

* A transparent and reproducible data curation pipeline.

### Weaknesses
* The description "meticulously cleaned of noise" is somewhat of an overstatement. The method relies on strong models, or LLM-as-a-Judge, meaning its effectiveness is still constrained by these models' capabilities.

* Experiments may be designed to validate the performance of the model-based filtering operator (Qwen2.5-VL-72B).

* Regarding Stage 1 (Data Aggregation), it appears to be an "exploration" of the novelty, diversity, and coverage of existing datasets. However, it lacks sufficient "novelty," as it is primarily a large-scale aggregation of existing, public community datasets (e.g., LLaVA, PixMo). The domain classification is coarse and relies on manual inspection, creating a significant engineering burden for reproducibility.

* Regarding Stage 3 & 4, the process seems more like "utilizing" existing information to construct questions & answers. A good utilization process should be able to discover difficult problems. In this process, it's necessary to correct the noise introduced during the exploration phase. However, this paper directly uses an LLM to judge if the newly generated CoT answer is consistent with the original one. This can cause high-quality, correct CoT samples to be filtered out simply because they "conflict with the original precise answer".

* The "long CoT" generation relies on "top proprietary MLLMs". A "good" utilization process should actively identify which problems are "difficult" and worth deep mining. In this paper, Stage 4 (Long CoT) largely passively receives samples that failed Stage 3 verification. This feels more like a "remedy" than a principled "hard-case discovery" mechanism.

### Questions
Besides massive data and data cleaning, what other factors contribute to the model's performance?

### Soundness
3

### Presentation
4

### Contribution
3
