# Understanding LoRA As Knowledge Memory: An Empirical Analysis

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Continuous knowledge updating in Large Language Models (LLMs) is a critical challenge. While existing methods like Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL) offer solutions, they are constrained by retrieval quality and context length. Departing from the conventional view of LLM memory that relies on context, this work highlights a novel parametric approach via Low-Rank Adaptation (LoRA). Although a few studies have hinted at this potential, LoRA's mechanics and optimal usage as a memory component remain largely unexplored. To bridge this gap, we conduct the first systematic and comprehensive empirical study of LoRA-based knowledge memory. Our analysis spans multiple dimensions, including the fundamental memory characteristics of LoRA, how to optimize a single LoRA, the possibilities of combining multiple LoRAs, and its synergy with existing methods in complex scenarios. Ultimately, this paper presents the first systematic framework for LoRA-based memory, offering foundational insights and actionable guidelines to future research and application.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work presents the first systematic study of using LoRA as knowledge memory. Comprehensive exmperiments are conducted, with corresponding empirial analysis. This study reveals the potential of LoRA (single or multi-) to serve as a standalone incremental knowledge update module or a complementary module of RAG or ICL, once configured properly. Synthetic datasets are built to investigate the influences on information capacity, data diversity, and model scalability for single LoRA. 
The findings on data-parameter efficiency motivates the authors to develop mult-LoRA systems, which show competitive performance with good routing or merging strategies on both synthetic and real-world datasets, although lower than ICL and RAG. Further experiments show ICL and RAG combined with LoRA knowledge modules can boost the performance, especially, ICL + multi-LoRA provides the most promising results. Thus, multi-LoRA provides a possibility to serve as a supplemantary module to ICL, so that the combination can effcientive equip an out-of-date model with up-to-date knowledge.

### Strengths
1. The idea is novel, which may opens a new door to solve the problem that LLMs lacking up-to-date knowledge. 
2. As the initial trial of this idea, a good amount of experiments are conducted to study the major aspects of the problem, which helps illustrate the performance landscape under typical use cases.
3. Synthetic datasets are built in a way to minimize the possiblity of knowledge contamination.
4. Clear take away messages are delivered.

### Weaknesses
1. To further demonstrate the efficacy of the proposed method, experiments under more realistic settings are needed. For example, news of different categories, such as politics, sports, and finance, are updated incrementally every day. Before and after the LLM's training end date, the things going on are mostly related logically, while the datasets used in the experiments are mostly standalone and only in 1 special category and are not in the time-incremental fashion. Time dependency and topic diversity may change the landscape. 
2. According to the analysis, multi-LoRA has many advantages over single LoRA, especially with external support. But reliable and high-performing merging or routing methods are not fully explored. 
3. In Table 1, the results in one dataset for one LLM family are not sufficient to draw a general conclusion on which single method or combination of methods work the best. Experiments on more models and datasets should be provided. 
4. Regardless of 3, ICL + mullti-LoRA seems the best performing combination. However, it also combines the disadvantages of both: ICL is slower than RAG, and multi-LoRA is slower than single. 
5. The code is not yet released.

As the initial work of this idea, I can accept imperfect answers for 1 and 2. But sufficient justifications to 3 and 4 should be provided.

### Questions
1. "Threshold" is mentioned in Fig.2 and also in the appendix, but not formally defined. Please provide the definition and motivation of setting this threshold. 
2. The lora ranks of PhoneBook experiment is provide in line 1272--1273. But this critical param is not provided in any other settings, especially not for the benefit in time section. 
3. In fig.5's 1st panel, any reason ICL is much better than the single-LoRA? 
4. In fig.5's 2nd panel, RAG (text embedding) is said to underperform single-LoRA, but its bar is higher.

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
4

### Summary
The paper offers a systematic and comprehensive empirical study of Low-Rank Adaptation (LoRA) modules as "knowledge memory" for large language models (LLMs). Departing from conventional memory-augmentation approaches like RAG or ICL, the authors analyze LoRA’s scalability, finite capacity, parameter efficiency, optimal single-module usage, strategies for combining/merging multiple LoRAs, and interactions with external context for complex multi-hop scenarios. The work includes custom benchmarks (PhoneBook, PaperQA), extensive experiments on capacity scaling, data engineering, merging techniques, and computational efficiency, culminating in practical guidelines for deploying LoRA as a modular, efficient, and complementary memory solution alongside existing retrieval or in-context strategies.

### Strengths
- Experiments are thorough and well-controlled, including custom datasets (PhoneBook, PaperQA) designed specifically to measure symbolic memorization, generalization, and internalization of new knowledge. The methodology is transparent, and setup/hyperparameters are clearly explained.
- The manuscript is well-written and organized, making the takeaways easily accessible for future readers.

### Weaknesses
- The paper repeatedly claims to be “the first systematic” analysis of LoRA as knowledge memory, but prior work has already explored modular LoRAs, merging, and parametric memory (e.g., PRAG [1], DyRAG [2],  LoRA soups [3], LoRAHub [4]) in closely related scopes; the current framing doesn’t clearly delineate what is truly new beyond the chosen evaluation mix and efficiency metric.
- Evaluation relies heavily on synthetic or simplified setups. Two of the core testbeds—PhoneBook and the CF slices—measure keyed recall or local factual “edits” that may not reflect messy, open-domain knowledge integration or multi-document reasoning encountered in practice. The strict EM on PhoneBook and efficacy on small CF subsets may overstate real-world utility (Sec. 3; Fig. 1–2).
- LLM-as-judge (PaperQA) introduces circularity and bias. PaperQA’s scoring uses GPT-4.1 on a 0–10 scale; this risks format overfitting (since QA is also the training target) and inherits judge bias without human calibration or inter-rater checks (Sec. 4; PaperQA description). No agreement statistics, rubric validation, or robustness to prompt variance are provided.
- Base-model scaling experiment is under-controlled. All Qwen sizes are trained with identical hyperparameters and raw data (Fig. 4 left), which may be sub-optimal for some sizes and confound the non-monotonic pattern. Per-size LR/step tuning or compute-matched comparisons would be more informative.


[1] Parametric Retrieval Augmented Generation. 

[2] Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancement

[3] LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks

[4] LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition

### Questions
- Will more advanced LoRA techniques, such as DoRA and PiSSA, be more effective?
- Can the authors delineate precisely what’s new vs. prior multi-LoRA/merging/modular-memory work, and add a comparison table?
- In Fig. 1/7, the paper shows rank-dependent saturation. Could you provide per-layer/per-module breakdowns (e.g., attention vs. MLP adapters) to clarify where capacity limits arise?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a systematic empirical study of Low‑Rank Adaptation (LoRA) framed as a parametric knowledge memory for large language models. The authors characterize single‑module properties (capacity, saturation, and parameter efficiency) across synthetic and factual benchmarks, and introduce two new evaluation sets (PhoneBook and PaperQA). They then explore data‑engineering strategies (QA, summaries, rewrites) to maximize information density, and analyze how base model scale and generator quality affect LoRA’s internalization. Moving to multi‑module designs, the work evaluates routing and merging trade‑offs, comparing oracle vs. embedding‑based routing and several merging schemes, and studies hybridization with ICL/RAG on long, multi‑hop narratives. The empirical narrative is careful and broad: it foregrounds capacity/efficiency curves, the impact of synthetic formats, and the system‑level costs of routing/merging, while demonstrating that hybrid parametric–non‑parametric systems often yield the most robust performance.

### Strengths
+ The empirical design is comprehensive and systematic: controlled rank–capacity sweeps, capacity saturation and efficiency analyses, ablations on synthetic data formats and generator quality, and careful comparisons of routing/merging strategies against oracles and standard baselines (ICL/RAG). The study uses multiple datasets spanning synthetic, counterfactual, and long‑form multi‑hop settings, supporting robust conclusions.


+ The paper is well structured, moving from single‑LoRA properties to multi‑LoRA systems and hybrids. Figures and appendices clearly present experimental setups, prompts, and hyperparameters, facilitating reproducibility and interpretability of results.


+ By deriving actionable design principles, e.g., efficiency‑optimal rank regimes, the value of data‑format curricula, and the role of sophisticated merging, this work provides practical guidance for building updatable, efficient memory systems. Its hybrid findings (LoRA + ICL/RAG) offer a pragmatic pathway for real‑world deployment.

### Weaknesses
- Since the main contribution of this work is a comprehensive analysis of LoRA as a parametric memory, the authors should explicitly explain the contribution of the related work that they have mentioned in the introduction section (L54-63) and explain their differences to those works and also the contributions they have provided (one by one) in the introduction section. For an empirical study it is important to clearly point out the differences to emphasize the originality and more importantly contributions of the study. 

- The omission of several contemporary LoRA‐related memory works weakens the authors’ assertion that their study is the first systematic framework for LoRA‐based knowledge memory. While the authors cite some key early efforts  failing to acknowledge other closely related or contemporaneous papers makes it appear as though the authors are either unaware of or downplay existing analyses. This gap can undermine the perceived novelty of their contribution and lead readers to question how their findings build on or differ from prior work. Since this work is an intensive empirical study, it is crucial for the authors to have a strong and solid related works study. Following I have listed some of the missing works: 

   * LoRA-Augmented Generation (LAG) for Knowledge-Intensive Tasks – Fleshman & Van Durme (2025). 

  * How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM? – Pletenev et al. (Findings of NAACL 2025). 

  * WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models – Wang et al. (NeurIPS 2024). 

  * LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models – Liang et al. (CVPR 2025). 

  Acknowledging these works will clarify that while the paper is a significant and comprehensive empirical synthesis, the concepts of LoRA‐based knowledge modules, parametric RAG and capacity analysis already have preliminary treatments in the literature.

  * LoRA Without Regret – Schulman (2025). (I consider this one a concurrent work, just added it here to encourage the authors to consider it in their future studies)

 * The core capacity/efficiency sweeps (rank vs. capacity/efficiency, saturation curves) and much of the routing/merging analysis are conducted mainly on Llama‑3.1. These are the paper’s backbone “final” insights, but they are not fully replicated on a second family.

 * There is no study showing the effect of adding same number of parameters to the base model vs. adding the same number of parameters via singe or multi LoRA. This is important to study specially for the multi-LoRA scenario where routing among different LoRAs somehow mimics an MoE setup, hence, to have an apple-to-apple comparison the approximate same number of parameters should be considered. 

 * In the multi LoRA experiments comparison to single LoRA it is not clear that what is the combined number of parameters in the multi LoRA setup vs. the single LoRA

 * No study on done on more general tasks and capabilities of LLMs (on general benchmarks)

 * The submission lacks experiments into how LoRA-based parametric memory, multi-LoRA, RAG, and ICL affect hallucination tendencies, the authors could incorporate explicit hallucination evaluations alongside their existing knowledge-recall and efficacy metrics.

### Questions
* L28: why do you have citation for right after the "Large Language Models" at the start of the sentence? Is it because of a LLM hallucination in injecting citations in the sentences? Please check this section out. 

 * In Q3 it is not clear that how number of memorized tokens are calculated.

### Soundness
3

### Presentation
2

### Contribution
3
