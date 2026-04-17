# WebWatcher: Breaking New Frontiers of Vision-Language Deep Research Agent

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Web agents such as deep research have demonstrated superhuman cognitive abilities, capable of solving highly challenging information-seeking problems. However,
most research remains largely text-centric, overlooking visual information in the
real world. This makes multimodal deep research highly challenging, as such
agents require much stronger perceptual, logical, and knowledge-based reasoning
abilities, as well as proficiency in more sophisticated tools. To address this limitation, we introduce WebWatcher, a multimodal agent for deep research with joint
reasoning ability across both visual and textual modalities. It uses high-quality
synthetic trajectories for efficient cold start training, utilizes various tools for deep
reasoning, and further enhances generalization through reinforcement learning. To
better evaluate the capabilities of multimodal agents, we propose BrowseComp-VL,
a benchmark with the style of BrowseComp that requires complex information
retrieval involving both visual and textual information. Experimental results show
that WebWatcher outperforms the prompt-based workflow and open-source agents
on HLE and BrowseComp-VL, and demonstrates its perception, multimodal reasoning, and searching capabilities across the other three benchmarks, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the limitation of existing text-centric deep research agents by introducing WebWatcher, a multimodal (vision-language) agent designed for complex information-seeking tasks. WebWatcher integrates visual-language reasoning with multi-tool interaction (e.g., Web Image Search, OCR, Code Interpreter) and uses a two-stage training pipeline: (1) cold-start supervised fine-tuning (SFT) on high-quality synthetic tool-use trajectories, and (2) reinforcement learning via Group-Relative Policy Optimization (GRPO) to enhance generalization.

To evaluate multimodal agents, the authors propose BrowseComp-VL, a benchmark extending the text-only BrowseComp to vision-language tasks. It includes obfuscated, multi-step queries across 5 domains (e.g., Natural Science, Entertainment) and two difficulty levels, requiring cross-modal reasoning and tool planning.

Experimental results show WebWatcher outperforms proprietary baselines (e.g., GPT-4o, Gemini-2.5-flash) and open-source agents on four high-difficulty benchmarks (HLE, LiveVQA, BrowseComp-VL, MMSearch) and performs competitively on the perception-focused SimpleVQA, demonstrating its versatility in both knowledge-intensive and visual reasoning tasks.

### Strengths
1. The paper fills a critical gap in existing deep research agents, which are largely text-bound. By integrating vision-language reasoning with flexible multi-tool use, WebWatcher moves beyond template-driven multimodal pipelines (e.g., OCR-only visual agents) to enable adaptive, complex problem-solving. 

2. The introduction of BrowseComp-VL addresses the lack of benchmarks for multi-step, obfuscated vision-language tasks—unlike existing VQA datasets (e.g., SimpleVQA) that focus on shallow perception.

3. Evaluations span 5 diverse benchmarks, comparing WebWatcher to 3 types of baselines (direct inference, prompt workflows, reasoning models) to isolate the impact of multimodal reasoning and tool use. Ablations (e.g., cold-start vs. instruct initialization, Pass@k analysis) validate key design choices.

### Weaknesses
1. The data construction methodology is inherited from webdancer and websailor v1. Browsecomp-vl is one of the few benchmarks that require multimodal capabilities. However, this dataset has a significant weakness: the incorporation of multimodality merely involves replacing entities in unimodal questions with their visual representations. As a result, the problem solver only needs to disambiguate the entities referred to by the images, while the remainder of the process is essentially no different from solving unimodal information retrieval problems, such as the original browsecomp.

2. The paper attributes WebWatcher’s success to "enhanced visual-language reasoning," but it does not isolate the impact of individual visual components.

### Questions
See the weaknesses

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Existing vision-language agents struggle with complex, multi-step research tasks that require integrating visual and textual information from the web. 
This paper introduces WebWatcher, a multimodal agent that uses a novel data generation pipeline and a hybrid training strategy to enhance its reasoning and tool-use capabilities. 
WebWatcher is trained on high-quality, synthesized trajectories and further refined with reinforcement learning, enabling it to flexibly use tools like web search, OCR, and a code interpreter. 
The agent's superiority is demonstrated through strong performance on several challenging VQA benchmarks, including the newly proposed BrowseComp-VL, where it outperforms existing open-source and proprietary systems.

### Strengths
1. The paper details a structured, multi-stage pipeline to create complex Vision-Question Answering (VQA) pairs from text-based sources. This process begins by generating multi-hop textual questions from hyperlink graphs to ensure reasoning depth. It then grounds these questions in authentic web images and transforms them into VQA format. To maintain high quality, the pipeline incorporates a two-stage filtering process using "Selector" and "Examiner" models to validate contextual alignment and visual plausibility.
2. The research employs and validates a two-phase training approach that uses Supervised Fine-Tuning (SFT) as a "cold start" before applying Reinforcement Learning (RL). The paper provides an analysis demonstrating that this SFT stage, which uses pre-generated high-quality trajectories, is critical for successful training. Experimental results show that an agent without the SFT cold start fails to achieve meaningful performance during RL training, whereas the SFT-initialized agent shows significant initial scores and subsequent improvement.

### Weaknesses
1.  The abstract claims the model "outperforms or matches proprietary baselines" across four VQA benchmarks. [cite_start]However, on the HLE benchmark, the model's overall average accuracy (13.6%) is slightly lower than proprietary reasoning models like Gemini-2.5-Pro (15.8%) and o4-mini (16.0%). The claim of superiority should be qualified to specify the benchmarks where this holds true, as it is not universal across all tested environments
2. The training methodology includes a trajectory filtering rule that removes any trajectory with fewer than three tool calls. This could introduce an inductive bias that favors longer, more complex reasoning chains. The paper lacks an ablation study to determine if this bias leads to redundant or inefficient tool usage on tasks that do not inherently require multi-step interactions.
3. The difficulty of the BrowseComp-VL benchmark is increased at Level 2 through "obfuscated entities and attributes". This method of "fuzzing" questions may introduce confounding variables, conflating the challenge of multi-modal reasoning with that of linguistic ambiguity and retrieval noise. The paper does not provide a human baseline performance or a detailed error analysis to disentangle these factors.

[Minor]
1. The evaluation of answer correctness relies on the "LLM-as-Judges" approach. This methodology is subject to potential biases, especially if the judge model shares an architectural family with the models being tested. The paper does not present results on inter-rater reliability with human experts or robustness checks using different judge models to validate the evaluation framework.
2. The Pass@k analysis demonstrates that performance on HLE improves significantly with more sampling, rising to 41.9% at k=32. However, the paper fails to quantify the inference cost and latency associated with this multi-rollout strategy. Without a cost-benefit analysis, the practical viability of achieving these higher scores in real-world applications remains unevaluated.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents **WebWatcher**, a multimodal deep-research agent designed to combine vision-language reasoning with dynamic tool use for complex information-seeking tasks. The system integrates five external tools—web text search, web image search, page visiting, OCR, and code interpreter—and learns tool-augmented reasoning through a two-stage pipeline: (1) **automated trajectory generation** for supervised cold-start training and (2) **group-relative policy optimization (GRPO)** reinforcement learning for fine-tuning. To evaluate multimodal research ability, the authors introduce **BrowseComp-VL**, an extension of the BrowseComp benchmark into visual domains, requiring cross-modal retrieval and multi-step reasoning. Experiments on **five challenging benchmarks** (HLE, LiveVQA, BrowseComp-VL, MMSearch, and SimpleVQA) show that WebWatcher-32B consistently outperforms both open-source and proprietary reasoning agents under comparable model sizes.

### Strengths
– **Well-motivated and timely contribution:** The paper clearly identifies a missing dimension in current deep-research agents—robust multimodal reasoning that jointly leverages textual and visual information.

– **Benchmark creation:** BrowseComp-VL fills a notable evaluation gap by introducing visually grounded, obfuscated, and multi-hop reasoning tasks. The construction process (Levels 1–2, entity masking, selector–examiner filtering) is rigorous and convincing.

– **Comprehensive experiments:** Results on five datasets demonstrate consistent superiority of WebWatcher over both direct-inference LMMs (e.g., GPT-4o, Gemini 2.5) and workflow baselines. The performance scaling from 7B → 32B models is clearly shown.

– **Clarity and completeness:** The paper is well written, with careful mathematical formalization, clean figures, and transparent dataset statistics.

### Weaknesses
– **Limited novelty in learning algorithms:** The overall architecture builds on established paradigms (ReAct for trajectory structure, GRPO for RL optimization). Innovation lies in the integration rather than in a fundamentally new learning mechanism.

– **Cost and efficiency reporting:** The paper does not specify training compute (GPU hours, wall-clock time) or inference latency. Quantitative comparisons with other open-source agents (e.g., WebDancer, WebSailor, WebShaper) would clarify efficiency and resource footprint.

– **Scalability risks:** The pipeline depends on GPT-4o for trajectory annotation, which may limit reproducibility or increase cost at scale; discussion of potential automation or open-source substitutes would be valuable.

### Questions
Please see the weakness above.
1. How many tool calls or reasoning steps typically occur before convergence, and do you encounter diminishing returns beyond a certain depth?
2. Do you encounter issues with the shared memory or context growing unboundedly during long reasoning sequences? If so, how is this mitigated, and what is the computational or memory cost of maintaining such context?
3. It would also be helpful if the authors could provide qualitative visualizations comparing WebWatcher’s reasoning paths to those of baseline systems to highlight its hierarchical and multimodal advantages.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces WebWatcher, a multimodal deep research web agent designed to perform complex reasoning across both visual and textual information. It combines large language models with multiple external tools to handle information-seeking tasks that require cross-modal understanding and planning. The authors propose a new benchmark, BrowseComp-VL, extending previous text-based benchmarks into the visual domain. They construct synthetic multimodal question-answering dataset and use a two-stage training process: SFT and GRPO. Experiments show that WebWatcher achieves competitive performance on several benchmarks including HLE, LiveVQA, and MMSearch.

### Strengths
* The paper addresses an important area—multimodal deep research—by trajectory data creation and two-stage training process including SFT and RL.
* The proposed automated trajectory generation pipeline offers a scalable way to construct training samples for multi-model deep research.

### Weaknesses
* Although the paper mentions that the proposed benchmark was verified by PhD-level experts at line 144, it does not provide details on the verification process or quantitative reliability measures (e.g., Cohen’s κ), making the evaluation reliability insufficient.
* Several key baselines are missing, such as o3 and GPT-4.1, which limits the completeness and fairness of the performance comparison.

### Questions
* What are the performance results of o3 and GPT-4.1 in Table 1 and Table 2?
* Please provide more details about the manual verification process during the benchmark construction phase, including how the PhD-level experts conducted validation and whether any quantitative reliability metrics (e.g., inter-rater agreement) were reported.

### Soundness
2

### Presentation
3

### Contribution
2
