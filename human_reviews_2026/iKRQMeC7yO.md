# AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The application of large language models (LLMs) in the medical field has garnered significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. To bridge this gap, we introduce AnesSuite, the first comprehensive dataset suite specifically designed for anesthesiology reasoning in LLMs. The suite features AnesBench, an evaluation benchmark tailored to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2).  Alongside this benchmark, the suite includes three training datasets that provide an infrastructure for continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards (RLVR). Leveraging this suite, we develop Morpheus, the first baseline model collection for anesthesiology reasoning.  Despite undergoing limited training with SFT and group relative policy optimization (GRPO), Morpheus not only achieves substantial improvements in anesthesiology that rival larger-scale models, but also demonstrates enhanced reasoning capabilities across general medical and broad-domain benchmarks. Furthermore, through comprehensive evaluations and experiments, we analyze the key factors influencing anesthesiology reasoning performance, including model characteristics, training strategies and training data. Both AnesSuite and Morpheus will be open-sourced at https://github.com/MiliLab/AnesSuite.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a comprehensive suite for analyzing, training, and evaluating LLMs’ capabilities in the domain of anesthesiology. The benchmark results demonstrate that current models, especially open-weight solutions may not perform perfectly on the benchmark. And both Qwen2.5 models and Morpheus would improve after being trained on the provided datasets.

### Strengths
1. The first suite for benchmarking, training, and analyzing failure patterns in the particular domain of anesthesiology
2. Comprehensive analysis and vivid figures demonstrating model performance gaps and some insights on the benchmark

### Weaknesses
1. The benchmark is not challenging enough. Overall the best open-weight model could solve 82% of the problems (DeepSeek-R1) and even a smaller Qwen3-14B could solve most of them (70%). These observations indicate that these problems in the benchmark are not challenging enough, especially considering the form of MCQA. One solution may be to refactor the MCQA questions into the free-end generation task that asks the model to answer the question without providing options.
2. Training data effectiveness is not properly demonstrated: table 4 presents models trained on the proposed training set. However, this type of training-testing paradigm falls back to the in-distribution evaluation. A better way to validate the training data would be mixing them with other post-processing data and test the model or test the trained model on other OOD medical tasks.

### Questions
How would the model trained on the proposed dedicated data perform on other medical tasks or even general tasks?

Overall, I appreciate the effort of constructing this suite for such a specific domain, it would potentially be a better paper with further clarifications and improvement. I would consider increasing my score once most of my concerns are addressed.

### Soundness
4

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
4

### Summary
The paper introduces a comprehensive suite of artifacts for the medical domain of anesthesiology, including benchmarks, a specialized corpus, and a anesthesiology-aligned model. The authors detail their end-to-end pipeline, covering pre-training data collection, benchmark creation, and the continuous training of a base model on the resources. The proposed benchmark is used to benchmark current SoTA medical LLMs, as well as open-weight and frontier models.

### Strengths
- The authors have conducted a substantial amount of work, delivering all the necessary sources and artifacts for the medical domain of anesthesiology, a field that is currently underserved by existing resources.

- The paper presents a solid methodology for assessing both contamination and originality across all proposed resources, ensuring the integrity of their contributions.

### Weaknesses
**[W1]** While the authors acknowledge that anesthesiology is underrepresented in current medical NLP resources and provide a setup that uses both reasoning and factual knowedge, they do not offer a clear, tangible motivation for how these resources will benefit medical-aligned LLMs overall (i.e. examples illustrating how these resources could be applied in real-world scenarios).


**[W2]** The methodology for benchmark creation and human verification is insufficiently detailed. Although their hybrid approach (LLMs + human verification) is reasonable, the paper does not specify:
- The number of human annotators involved per sample
- The instructions or rubric used for verification
- Annotator agreement rates
- How quality is ensured for data not verified by humans
These missing details make it difficult to assess the reliability and validity of the benchmarks.

**[W3]** The paper does not discuss how alignment in the anesthesiology domain affects performance on other downstream medical tasks (e.g., MedQA) or general tasks (e.g., MMLU). 

**[W4]** The pre-training and SFT data are likely to be used in continuous pretraining of medical-aligned models that need to perform well across multiple medical domains. However, the authors do not provide a realistic CPT scenario demonstrating how their data would be used in such a context. Even a hypothetical example would help clarify the potential impact and application of their resources.

**[W5]** The use of anesthesiology-related documents from Fineweb for ontinuous pretraining involves replaying existing data rather than introducing new resources. As a result, this aspect of the contribution is less substantial compared to the creation of the main benchmark.

### Questions
Please refer to the Weaknesses section above.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AnesSuite, a dataset and benchmark collection designed to test and improve LLM reasoning in anesthesiology. AnesBench provides evaluation tasks that cover factual knowledge and complex decision making in English and Chinese. The authors also release three training datasets that target continued pretraining, supervised fine tuning, and reinforcement learning with verifiable rewards. Using these resources, they build a baseline model named Morpheus and run a broad set of experiments against strong public models. The analysis examines how language transfer, training data choices, and training methods affect performance on anesthesiology reasoning.

### Strengths
1.  The study makes a clear and timely observation that evaluation should reflect realistic clinical settings, and it identifies contextual complexity as a common bottleneck for reasoning across tasks.

2. The three-tier benchmark design and the linked training resources form a coherent pipeline, and the Morpheus baselines with ablations provide useful evidence on which choices lead to gains.

### Weaknesses
1. I appreciate the careful build of a bilingual, cognitively tiered benchmark and the broad empirical sweep, yet the main novelty lies in the dataset suite rather than a method or algorithm that is central to ICLR. The focus on anesthesiology is narrow for a broad ML venue and is also narrow even within medical or clinical tracks, which limits general use.

2. The dataset and benchmark draw mainly on public exams and textbooks and are text only. They do not include images, multimodal clinical data, or real clinical or PHI data. Existing resources already support assessment of medical knowledge, such as MedQA, MedMCQA, and PubMedQA. Clinically grounded datasets that capture real decision-making and multimodal signals are more directly relevant to patient care. The contribution is limited to that, not a direct link to clinically actionable use.


3. I am concerned about data quality. On AnesBench, SFT on the proposed dataset does not beat the Qwen2.5-Instruct baselines: Morpheus 7B with SFT only averages 54 percent compared with 59 percent for the 7B baseline, and Morpheus 14B with SFT only averages 57 percent compared with 64 percent for the 14B baseline. Most of the gains appear after the GRPO stage, to 63 percent and 69 percent, and that is modest. SFT slightly improves the English subset but drops the Chinese subset by a large margin, which points to issues with data quality, data balance, or objective alignment. Train and test sit in the same domain due to vertical fine-tuning. One would usually expect larger SFT gains in that case, so the results are underwhelming.

### Questions
Please see the question about weakness above.

### Soundness
3

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
2

### Summary
The paper introduces AnesSuite, a suite aimed at assessing and improving anesthesiology reasoning in LLMs. It comprises:
(1) AnesBench — a bilingual (EN/ZH) multiple‑choice benchmark (7,972 items) stratified by cognitive demand into System 1 (factual), System 1.x (hybrid), and System 2 (complex decision making). 
(2) AnesCorpus — ~2.4M anesthesia‑related documents (1.8M EN + 0.6M ZH) for continued pretraining (CPT), filtered from FineWeb/Chinese‑FineWeb with keyword rules and a two‑stage decontamination (n‑gram screen + LCS>64 removal) to reduce overlap with the benchmark.
(3) AnesQA — 20,713 English QA pairs produced via a two‑model LLM pipeline (Llama‑3.3‑70B to generate questions; Qwen‑2.5‑72B to filter/answer), with manual spot‑checks and regex filtering to excise 119 flawed items.
(4) AnesR1 — 10,287 verifiable MCQs (EN+ZH) with chain‑of‑thought (CoT) traces generated by DeepSeek‑R1 and kept via rejection sampling only when the final choice is correct. Using these resources, the authors train Morpheus‑7B/14B (Qwen‑2.5 inits) with SFT + GRPO; they report improvements over the instruction‑tuned bases and parity with larger baselines on AnesBench.

### Strengths
- First-of-it-kind domain‑specific reasoning suite for anesthesiology that separates fast recall from complex decision‑making (System 1→2) and offers training corpora aligned with CPT/SFT/RLVR stages. Prior work either subsumed anesthesiology under broad medical categories or focused on factual recall; this suite targets decision‑making explicitly;

- Bilingual design with explicit experiments on language transfer. The ablation shows CPT on English‑heavy corpora can help EN while hurting ZH, highlighting data‑language balance as a first‑order concern in medical domains;

- Reasoning‑centric baselines (Morpheus) showing that limited RL with verifiable rewards (GRPO) plus CoT data can close the gap to much larger models on this domain.

### Weaknesses
1. External validity of System‑2 constructed from exam/textbook‑style abstractions rather than real perioperative trajectories; this may under‑represent key anesthetic complexities (e.g., dynamic vitals, ventilator settings, lab time‑series). 

2. Leakage analysis scope: the permutation‑confidence heuristic over 500 items is informative but indirect; it cannot rule out contamination in proprietary models, nor leakage via near‑duplicates beyond LCS>64.

3. GRPO is described at a high level; the exact verifiable reward signal (e.g., answer‑only vs. additional structure) and stability/variance aren’t deeply analyzed.

### Questions
1. For AnesR1, did clinicians evaluate a subset of DeepSeek‑R1 chains for medical correctness beyond the final answer? Any evidence that RL on these chains avoids entrenching subtle misconceptions?  

2. What exact reward(s) were used in GRPO—answer correctness only, or also intermediate checks (e.g., vital‑sign logic)? How robust are results to reward misspecification?

### Soundness
2

### Presentation
2

### Contribution
2
