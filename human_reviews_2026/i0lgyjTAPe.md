# ReasonMap: Fine-Grained Annotation of Mathematical Definition and Theorems in Long CoT Reasoning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Annotating mathematical knowledge is a fundamental prerequisite for structured knowledge acquisition from large-scale mathematical solutions. However, existing datasets lack the fine-grained annotations of theorems and definitions at scale. This makes it challenging to assess how well modern mathematical reasoning models understand and apply specific knowledge within their complex chain-of-thought (CoT) outputs.
In this paper, we propose a new task: Automatic Annotation of Mathematical Definitions and Theorems (AAMDT), which aims to extract Mathematical Definitions and Theorems (MDTs) from long CoT reasoning. To tackle this, we introduce ReasonMap, a novel two-stage training framework. The first stage, Foundational Model Training, builds a broadly capable model by performing Supervised Fine-Tuning on a hybrid corpus of both concise human annotations and LLM-augmented long CoT data. The second stage, High-Fidelity Alignment, then refines this model using Direct Preference Optimization to ensure the final output is both precise and reliable.
Comprehensive experiments show ReasonMap consistently outperforms strong baselines on the AAMDT task, especially in long CoT scenarios. Crucially, we validate the quality of our annotations by demonstrating that the extracted MDTs can directly enhance the mathematical reasoning performance of downstream large language models. Our work offers a scalable solution for the automatic annotation of mathematical corpora, significantly reducing the reliance on manual labeling. The
github repository can be found at: \url{https://anonymous.4open.science/r/ReasonMap-6A83}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a method for annotating mathematical solutions with the definitions and theorems used within them. To this end, the authors define a new task, which they call AAMDT (Annotation of Axioms, Definitions, and Mathematical Theorems), and develop ReasonMap, a multi-stage pipeline to train models for this task. This pipeline is composed of 2 SFT stages - guided by human annotations, and LLM-generated solutions, followed by a DPO alignment stage. The authors demonstrate that each stage of the pipeline significantly improves performance on the AAMDT task for both in-distribution and out-of-distribution data. Their results show that fine-tuned small open-source models can achieve performance comparable to some commercial models. Finally, they show that the extracted definitions and theorems can be used as hints to marginally improve the problem-solving accuracy of models on the MATH500 benchmark and an internal test set.

### Strengths
1. The paper is generally clear and well-written.
2. The authors provide all necessary materials to ensure the reproducibility of their work.
3. The proposed idea is interesting and can serve as a good way to extract structured information from mathematical reasoning for downstream or retrieval-based tasks.
4. The main experiments and ablation studies effectively validate the pipeline's design and demonstrate its significant impact on performance.

### Weaknesses
1. The motivation for claiming the need for categorically-diverse problem sets (Section 2.1) is not well-supported by the cited literature. The paper claims that prior work has shown the benefits of diversity, but LIMO [1] filters problems based on difficulty, not topic diversity, and Light-R1 [2] only briefly mentions category filtering without providing evidence of its impact. To strengthen the paper's motivation, the authors should provide direct evidence that diversity in mathematical categories and concepts is beneficial for training or cite work that explicitly isolates this as a significant factor.

2. The paper repeatedly argues that existing alternatives like human annotation or proprietary model APIs are too expensive. However, it fails to provide a comparative cost analysis for the proposed multi-stage training pipeline. With the availability of powerful and inexpensive generalist models, the claim that training a specialized model is more cost-effective is not sufficiently justified. This criticism also applies to the argument that Omni-Math's [3] annotation process is expensive, despite its use of the cost-efficient GPT-4o model.

3. The evaluation of the AAMDT task may be flawed. The fine-tuned models could be learning the specific annotation format of the ground-truth data, which would naturally lead to higher string-matching scores. This creates an unfair comparison with proprietary models that have not been exposed to this specific format, potentially measuring format adherence rather than true MDT extraction capability.

4. The models used in the evaluation are significantly outdated, with one of the base models and propriatery models having been superseeded by at least 1 generation of LLMs. In particular, GPT-4o-mini has been superseeded by GPT-4.1-mini, and most recently GPT-5-mini (and even potentially GPT-5-nano in terms of performance), DeepSeek V3 - by V3.1 (or even R1 if we include reasoning models), and Qwen-2.5-7B by Qwen3-8B and Qwen3-4B-2507. This design choice is not explained and undermines the reliability of the results, especially in light of the paper's strong claim that "even powerful models struggle with the precision required for this task" (L381). This is a particular concern given that the costs for the newer GPT models are similar to GPT-4o-mini.

5. The authors attempt to validate the usability of MDTs in downstream tasks by using them as hints for model solving. However, the improvement there is only marginal, especially given that the MDTs presented significantly relevant information from the **ground-truth solution**, but diluted. This also makes the setting unrealistic, as one is expected to have access to the actual solution to apply the method.

### Questions
1. Could you provide an approximate cost comparison between running the ReasonMap pipeline (training+inference) and using proprietary LLM APIs for different workload sizes? This would help substantiate the claim of cost-efficiency. Using public pricing for a comparable GPU node (e.g., from a cloud provider) would be a reasonable proxy.

2. The paper should provide evidence validating the use of GPT-4o-mini as an LLM-as-a-judge. How does its evaluation accuracy compare against human-annotated or otherwise validated ground-truth matchings? For example, can it correctly identify non-trivial semantic equivalences, such as the relationship between the Power of a Point theorem and the Secant-Tangent theorem? The current prompt seems to favor simple semantic similarity, which may lead to missing valid (or adding invalid) matches.

3. Given that the OOD evaluation only compares MDTs against the ground-truth domain, what metrics does the evaluation use? If counting a proportion of the MDTs that can be appropriated to the domain, that is not only highly subjective (and not validated, similar to Q2), but it is also unclear whether valid MDTs may come from a related but unlisted domain.

4. Furthermore, can the authors provide some information about the reliability and consistency of the MDTs? The example in A.9 paints a picture of what a sample output may look like, but one could subjectively also refer to a definition of a "homomorphism", "generator", etc., which are missing from the example.

5. Given that the reward function in Section 3.2.2 is granular and automatically calculable, it seems well-suited for methods like GRPO [4], which can leverage detailed reward signals. What was the rationale for choosing DPO, which uses a simpler binary preference signal, over a more granular reward optimization method?

6. Can the authors re-evaluate their method using more current open-source and proprietary models? If time permits, applying the ReasonMap pipeline to a newer base model would also significantly strengthen the paper's claims about its effectiveness.

7. Could you validate the downstream utility of MDTs using a more realistic baseline? For instance, by providing hints derived from a model's own (potentially incorrect) generated solution, rather than leaking information from the ground-truth solution, or by comparing to any existing methods that use hint generation?

8. How was the DeepMath test set in 4.5 constructed?

9. The authors should address any concerns mentioned in the **Weaknesses** section that have not been touched upon as part of the aforementioned questions.

## Current rating

My recommendation for this work is a score of **2: Reject**. While the execution is solid, it is unclear how useful this work can be in further research and applications. In particular, the motivation aspect in the current seems weak, with unclear references to related work and an unconvincing demonstration of downstream or training utility. These issues give the impression that the proposed task is artificial and leaves its practical feasibility unproven. The paper's relevance is also diminished by its use of models that are no longer at the forefront of the field. I am willing to raise my score if the authors can satisfactorily address these major concerns during the rebuttal/discussion phase.

### References

[1] Ye, Yixin, et al. "Limo: Less is more for reasoning." arXiv preprint arXiv:2502.03387 (2025).

[2] Wen, Liang, et al. "Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond." arXiv preprint arXiv:2503.10460 (2025).

[3] Gao, Bofei, et al. "Omni-math: A universal olympiad level mathematic benchmark for large language models." arXiv preprint arXiv:2410.07985 (2024).

[4] Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new task, "Automatic Annotation of Mathematical Definitions and Theorems" (AAMDT), which aims to extract Mathematical Definitions and Theorems (MDTs) in a fine-grained manner from long Chain-of-Thought (CoT) reasoning processes. To address this task, the authors have designed a two-stage training framework named ReasonMap. The first stage, "Base Model Training," constructs a model with foundational knowledge and long-text generalization capabilities through Supervised Fine-Tuning (SFT) on both human-curated and LLM-augmented long CoT data. The second stage, "High-Fidelity Alignment," further refines the model using Direct Preference Optimization (DPO) to enhance the precision and reliability of the output. Experimental results show that ReasonMap significantly outperforms several strong baselines on the AAMDT task, especially in long CoT scenarios. Additionally, the authors validate the effectiveness of the extracted MDTs through a downstream math problem-solving task, demonstrating that they can serve as in-context prompts to enhance the reasoning abilities of other large language models.

### Strengths
1.  **Novel and Important Task:** The proposed AAMDT task addresses a key pain point in the current field of large model mathematical reasoning. Fine-grained knowledge annotation is crucial for understanding models' reasoning processes, conducting interpretability analysis, and constructing higher-quality training data. This work is highly pioneering and has significant potential for application.
    
2.  **Comprehensive Experiments:** The authors have conducted a series of thorough experiments to validate their method's effectiveness. Not only did they compare their method against multiple baselines on in-distribution (ID) and out-of-distribution (OOD) test sets, but they also demonstrated the necessity of each component in the framework (e.g., sequential training, definition-only samples) through detailed ablation studies. Furthermore, the downstream task validation directly demonstrates the practical value of the extracted MDTs.

### Weaknesses
1.  **Insufficient Representativeness of Models in Downstream Task Validation:** This is my primary concern. In Section 4.5, the authors use Qwen2.5-7B and Llama3.1-8B as base models to validate the improvement in problem-solving abilities from MDTs. Although these are excellent open-source models, they are not the current state-of-the-art Large Reasoning Models (LRMs). The performance gains observed on these models (around 1-2% on MATH500 and 1-2% on DeepMath), which have limited reasoning capabilities themselves, may not fully reflect the value of MDTs. It is possible that the reasoning capabilities of these models themselves are the bottleneck, limiting the potential impact of the MDTs. To more convincingly demonstrate the effectiveness of MDTs, I would be very eager to see experimental results on stronger, recognized expert models in mathematical reasoning. If the results are positive, I would be willing to raise my score.
    
2.  **Coarse-Grained OOD Evaluation Metric:** The authors use "domain relevance" as a proxy metric for OOD evaluation. While this is a viable compromise in the absence of fine-grained labels, its validity is debatable. This metric can only determine if an extracted MDT belongs to the broad domain of the problem but cannot measure its **effectiveness** or **non-triviality**. For example, in a calculus proof involving multiple complex theorems (like the Intermediate Value Theorem or L'Hôpital's Rule), if the model extracts overly basic and broad labels such as "Definition: Set" or "Definition: Real Number," it would be considered correct under the current OOD evaluation because it is indeed related to the domain of mathematics. However, such "correct but uninformative" labels are of little help in understanding the proof or aiding reasoning, and may even introduce noise due to their redundancy (and this situation is indeed prevalent in your publicly released dataset).
    
3.  **Potential Misalignment in Long CoT Generation:** In constructing the D_long dataset, the authors instruct the model to "pretend it has not seen the original proof" and generate a new long CoT based on a brief human-written proof outline. This process presents a subtle challenge. Based on my experience and testing, the model is likely to fall into one of two extremes: a) **Path Deviation:** The model fails to strictly follow the core logic of the original proof, introducing new theorems or steps not present in the original. This creates a factual misalignment between the generated long CoT (A_l) and the original MDT labels (L), resulting in noisy data. b) **Over-Imitation:** The model simply expands and "colloquializes" each step of the short proof without genuinely exhibiting a complex, exploratory CoT process. This undermines the original purpose of using this data for "long CoT generalization training." While this does not significantly undermine the paper's persuasiveness, I hope the authors can elaborate more on the limitations of this process.

### Questions
1.  Regarding the downstream task validation, have you considered conducting experiments on stronger reasoning models (e.g., GPT-4o, DeepSeek-V3)? Are the performance gains from MDTs still significant on these models? This is crucial for assessing the true potential of this work.
    
2.  Regarding the OOD evaluation metric, are the authors aware of the potential issues caused by its coarse granularity (i.e., rewarding overly broad or trivial MDTs)? Could you comment on the frequency of this situation in your tests? Have you considered supplementary evaluation methods, such as manual evaluation of a small sample to measure the **informativeness** or **utility** of the extracted MDTs?
    
3.  When generating the long-chain reasoning data (D_long), how did you ensure that the generated content remained logically consistent with the original proof outline while avoiding simple paraphrasing, thereby ensuring the data's high quality and alignment? Was there any data cleaning or filtering process involved?

### Soundness
3

### Presentation
4

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
The paper introduces the Automatic Annotation of Mathematical Definitions and Theorems (AAMDT) task, aimed at identifying fine-grained mathematical definitions and theorems within large language models’ (LLMs) chain-of-thought (CoT) reasoning. To tackle this, the authors propose ReasonMap, a two-stage framework. In Stage 1 (Foundational Model Training), a hybrid dataset combining concise human-written proofs and long LLM-augmented CoTs is used to teach both accuracy and long-context understanding. Stage 2 (High-Fidelity Alignment) refines this model using Direct Preference Optimization (DPO) to reduce hallucinations and formatting inconsistencies. Experimental results across in-distribution and out-of-distribution benchmarks show that ReasonMap significantly outperforms strong baselines such as GPT-4o-mini and DeepSeek-V3, especially for complex reasoning tasks. The extracted annotations (Mathematical Definitions and Theorems, MDTs) also improve the performance of other reasoning models when injected as contextual hints. This work provides a scalable, cost-effective way to generate high-quality, fine-grained mathematical annotations and enhances interpretability and downstream reasoning quality in LLMs.

### Strengths
1. The novel task definition (AAMDT) fills a crucial gap in mathematical reasoning datasets.
2. Two-stage framework effectively combines supervised fine-tuning and preference alignment.
3. Empirical validation is thorough—covering in-distribution, out-of-distribution, and downstream reasoning impact. Clear scalability advantage, reducing reliance on manual annotation.

### Weaknesses
1. The paper claims to annotate `fine-grained` Mathematical Definitions and Theorems (MDTs) but does not clearly formalize what level of granularity qualifies as fine-grained. Without a standardized taxonomy or ontology (e.g., linking MDTs to known mathematical knowledge graphs), reproducibility and consistency across datasets remain unclear.
2. The long CoT augmentation relies on DeepSeek-R1 generations. This introduces an uncontrolled source of noise and style bias. No explicit data filtering or quality assurance metrics are reported for these synthetic CoTs.
3. The baselines (GPT-4o-mini, DeepSeek-V3, Qwen, Llama) are used in few-shot prompting setups, not fully fine-tuned under the same conditions. This weakens the claim that ReasonMap significantly outperforms other methods. I think the improvements could stem from different tuning or dataset exposure rather than intrinsic superiority.
4. The F1 and accuracy metrics depend on another LLM’s judgement matching of MDTs. Any human evaluation or inter-annotator agreement study could be provided?
5. The rejected examples in DPO training are generated by the same model $\pi$ long, filtered only by a scalar threshold. This can lead to homogeneous (i.e., the bad examples tend to look very similar to the good ones), limiting the robustness of alignment and making the model overfit to its own prior mistakes. If the model only learns from its own style of mistakes, it won’t see enough diverse or surprising wrong answers.

### Questions
As my understanding, the paper says "it annotates mathematical definitions and theorems (MDTs) used in reasoning chains", it assumes that each problem has one correct set of theorems and definitions used in the solution (ground truth). But in mathematics, there are often multiple valid ways to solve the same problem — you might use a different theorem or definition to reach the same correct answer. So when ReasonMap compares its generated annotations against that single “ground truth” list, it might mark an answer as wrong just because it used a different but still correct theorem.

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
ReasonMap approaches an elicitation-based (‘fine-grained’) extraction of mathematical definitions and theorems (MDTs) from long chain-of-thought solutions via a two-stage pipeline: foundational SFT on more concise proofs plus LLM-augmented short to long CoTs, followed by DPO with negatives selection. On long-CoT tests it outperforms baselines and shows OOD gains. The proposed solution has practical implications in the area of Mathematical Reasoning, but there are some questions wrt the depth of analysis and communication of the contribution (articulated below). The proposed approach looks promising: addressing some of the limitations may substantially improve the long term impact.

### Strengths
- Clearly articulated/described two-stage training. 

- Relevant empirical results with mid-size models beat larger baselines on long-CoT extraction by notable F1 margins. Gains persist on OOD sets.

- Overall downstream utility and applicability.

### Weaknesses
- Lack of a better articulation of the underlying principle.

- The paper lacks a clear structural model on the assumptions behind mathematical reasoning. 

- Variability in reasoning structure across datasets (styles, domains) is neither analyzed nor controlled, limiting external validity

- Baseline selection is opaque (inclusion/exclusion criteria).

- Coverage of related work is incomplete and does not adequately position the contribution relative to math (and more general) CoT supervision. One example: https://arxiv.org/pdf/2502.12616

- Lack of more extensive qualitative analyses.

### Questions
* Qualitative analysis/Interpretation.

- What are the dominant false-positive/false-negative patterns (e.g., confusing lemmas with definitions, inference drift, spurious generic statements)?

- How do errors differ across your proposed solution and the baselines? What are the side effects?

* Experimental design.

- What is the formal selection (inclusion and exclusion) for the baselines? Can you elicit them more formally?

- How does the system handle adversarial or misleading CoTs (negations, swapped order, aliasing of named theorems, step omissions, LaTeX variants, informal paraphrases)?

- How is the variability within LLM family and out-family?

Mechanism.

- What are the underlying assumptions behind your selected datasets (these are your proxies for mathematical reasoning)?

- Can you state the key underlying principles of the underlying mechanism that the proposed approach aims to achieve? What is being hypothesized as a representational effect (apart from the task scores)?

### Soundness
2

### Presentation
3

### Contribution
2
