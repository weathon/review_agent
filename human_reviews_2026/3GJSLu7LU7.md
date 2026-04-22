# EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Retrieval-Augmented Generation (RAG) has advanced open-domain question answering by incorporating external information into model reasoning. However, effectively leveraging external information to enhance reasoning presents the following challenges: (1) low signal-to-noise ratio, where answer-supportive external information is diluted by irrelevant material, and (2) error accumulation, which arises in multi-hop reasoning when incomplete or misleading information is incorporated. To address these challenges, we introduce EviNote-RAG, a framework that follows a retrieve–note–answer workflow. Instead of reasoning directly over raw external information, the model first produces Supportive-Evidence Notes (SENs), which concisely preserve answer-critical information and explicitly mark key and uncertainty information to improve accuracy. We further design an entailment-based Evidence Quality Reward (EQR) to ensure that SENs are logically sufficient to derive the final answer, thereby enhancing SENs' quality. Experiments on both in-domain and out-of-domain QA benchmarks show that EviNote-RAG achieves state-of-the-art performance, improving answer accuracy, training stability, robustness, and efficiency. In particular, it yields relative F1 gains of 20% on HotpotQA (+0.093), 40% on Bamboogle (+0.151), and 91% on 2Wiki (+0.256), benefiting from improvements in the reasoning process.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EviNote-RAG, a reinforcement learning-based retrieval-augmented generation (RAG) framework aimed at enhancing multi-hop open-domain question answering (QA). EviNote-RAG interposes a support-evidence note-taking phase between retrieval and answer generation, encouraging models to distill concise, answer-focused, and evidence-annotated summaries of retrieved content. The core novelty centers on the use of Supportive-Evidence Notes (SENs)—explicitly annotated for key and uncertain information—and the incorporation of an entailment-based Evidence Quality Reward (EQR) that guides the model to produce notes sufficient to logically entail the correct answer. Empirical evaluation on a broad suite of in-domain and out-of-domain QA benchmarks demonstrates state-of-the-art performance, validating both the stability and accuracy gains attributed to EviNote-RAG.

### Strengths
Well-motivated problem and principled pipeline: The paper systematically identifies the key weaknesses of existing RAG paradigms, especially the propagation of noise and lack of intermediate reasoning supervision, then builds a retriever-note-answer workflow to address them.

Innovative evidence selection strategy: Introducing Supportive-Evidence Notes (SENs), with both key and uncertain information annotations, is a thoughtful human-inspired abstraction for focusing the model’s reasoning and reducing distraction from irrelevant retrievals.

Entailment-based reward shaping: The design and deployment of the Evidence Quality Reward (EQR), using a lightweight NLI entailment judge to validate that extracted notes logically entail the ground-truth answer, provide a meaningful intermediate training signal beyond answer correctness.

Clarity and visualization: Figures such as Figure 1 (illustration of EviNote-RAG vs. baselines) and Figure 2 (system pipeline overview) provide clear, accessible visualizations of the technical contributions. The workflow and note-taking mechanism are well illustrated.

Robust empirical evaluation: Extensive benchmark evaluations (see Table 1, Table 2, and others) show consistent relative performance gains over strong agentic, prompt-based, and RL baselines—including on challenging out-of-domain settings where prior state-of-the-art models degrade.

### Weaknesses
Marginal Incremental Novelty Over Recent RL-based RAGs: While the addition of SENs and EQR is a principled extension, the framework rests on a well-trodden foundation of RL-based RAG (see e.g., existing agentic RL paradigms outlined in Section 2.2 and Table 1). The work is arguably more evolutionary than revolutionary—i.e., integrating recent ideas on summarization, evidence selection, and NLI rewards rather than inventing fundamentally new techniques. For example, capturing intermediate notes and using entailment-based signals have both been foreshadowed in prior art.

Limited explicit analysis of failure modes and generalization: Despite promising out-of-domain results, diagnostics are overwhelmingly positive. A more critical analysis of where EviNote-RAG fails, especially in harder multi-hop or ambiguous queries, and how SEN/EQR affect error types, would improve scientific insight and transparency. Are there cases where evidence abstraction discards subtle but crucial signals?

Scalability and efficiency unstated/insufficiently quantified: While token- and latency-level efficiency improvements are illustrated (e.g., Figure 3c/d, Figure 7), computational overhead from the entailment judge and dynamic summarization are discussed only in passing (Section 3.3, 4.4). More granular reporting on training/inference time per query compared to simpler baselines, as well as the resource requirements (in terms of deployed hardware and model size tradeoffs), would strengthen practical impact claims.

Reward sparsity and reinforcement learning dynamics: The reward structure, while intuitive, remains somewhat ad hoc. The thresholds and weights (reward of “1+R_EQR” for correct answer, “0.1+R_EQR” for format compliance, etc.) lack direct ablation or sensitivity analysis. The effect of tuning these hyperparameters (Section 3.4) on stability and optimization remains largely unexplored.

Ambiguity in evidence annotation labeling: The process for evidence annotation (i.e., what counts as “key” or “uncertain” and how the model learns to distinguish this) is not as formally defined or empirically substantiated as it could be. Is annotation accuracy monitored? Could this be made more rigorous?

### Questions
How would your system fare when transplanted to other RAG settings/tasks? The experimental scope is QA-centric. Do you expect SEN+EQR to generalize to dialogue systems, knowledge-grounded text generation, or fact verification? Please discuss any early empirical findings or expected constraints.

Can you provide more detailed ablations on reward hyperparameters and evidence annotation labeling? Specifically, how sensitive is performance/stability to the $1+R_{EQR}$ and $0.1+R_{EQR}$ choices, and what guidance can you provide on threshold selection? Is evidence labeling accuracy measured, and do label errors propagate?

What are the computational tradeoffs compared to non-SEN or non-EQR variants? Please quantitatively compare resource requirements, training/inference latency, and throughput to the main agentic RL baseline, separately for training and at inference.

Could you provide more detailed error analysis? Your results and case studies are almost uniformly positive. What are common scenarios where SEN+EQR fails—e.g., when key evidence is subtle or distributed, or when retrieval outputs highly conflicting signals?

Are your improvements statistically significant? Please provide standard deviations, confidence intervals, or significance tests for Table 1/Table 2 results.

What is the overhead imposed by the entailment judge? Is it feasible for real-time or low-latency QA deployments?

### Soundness
3

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
4

### Summary
The paper proposes EviNote-RAG, a retrieval-augmented generation framework that restructures the retrieve -> answer pipeline into retrieve -> note -> answer pipeline.
After retrieval, the model first writes Supportive-Evidence Notes (SENs) that distill only answer-critical content and explicitly annotate key versus uncertain information.
The final answer is then produced conditioned on these generated notes. 
To improve the usefulness of SENs, the authors introduce an Evidence Quality Reward (EQR): a lightweight NLI entailment scores with a small language model, to judge whether the final SEN entails the gold answer. This score is used to shape reinforcement-learning updates using GRPO. 

Experiments are conducted on Qwen-2.5, covering in-domain NQ/HotpotQA and out-of-domain PopQA, TriviaQA, 2WikiMQA, Musique, and Bamboogle
This paper reports better F1/EM than strong prompt- and RL-based RAG baselines and shows training-stability plots and ablations isolating SEN and EQR.

### Strengths
1. This paper addresses the potential issue of noise and error accumulation in RAG. 
The authors introduce sens (supportive-evidence notes) on retrieved content as intermediate results to serve as reliable and denoised information, thereby supporting more robust reasoning. (These sens capture effective information from noisy text in a relatively structured format, like knowledge base triplets.)

2. The paper incorporates notes into the RL training process as an intermediate signal. 
Specifically, the authors employ a small NLI model to determine whether the reasoning in the notes constitutes a correct logical entailment. This signal is then used as a component of the reward signal, thereby enriching the outcome reward to some extent.

### Weaknesses
1. Lack of contributions. The evidence-supporting notes proposed in this paper essentially convert textual information into structured data, leveraging structured information for verification to assist reasoning. However, this approach is not novel, as similar techniques have been employed in prior multi-hop RAG studies, as well as in some KBQA works.

2. Concerns about sen rewards. This paper proposes using intermediate notes as reward signals in reinforcement learning, which could partially alleviate the model's overreliance on outcome-based rewards.
Nevertheless, the effectiveness of employing a small NLI model to evaluate such signals remains questionable.
Additionally, while incorporating intermediate rewards in multi-hop QA has become relatively common in current research, the rewards obtained through such methods remain outcome-oriented and are applied to the entire sequence without fine-grained credit assignment to attribute advantages to specific intermediate steps. 
Consequently, the effectiveness of this reward mechanism is debatable.

3. The experiments were conducted on only 500 samples, resulting in high variance. Whether the proposed method significantly outperforms the baseline approaches requires more thorough validation. 
The experimental setup should be aligned with prior work to ensure a fair comparison (both train and inference)

### Questions
my questions are listed in weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes EviNote-RAG, an agentic RAG framework that adopts a retrieve-note-answer pipeline. Specifically, after retrieving a set of documents, EviNote-RAG dynamically introduces a summarisation phrase, generateing supportive-evidence notes from the retrieved documents.  The overall framework is trained using reinforcement learning with a specially designed evidence quality reward. This reward leverages a NLI model to evaluate whether the generated notes entail the correct answer, thereby encouraging the generation of high-quality notes.

### Strengths
1. The paper proposes an agentic RAG framework EviNote-RAG, which dynamically generates supportive-evidence notes to help the LLM better understand the key information within retrieved documents. 
2. Experimental results on seven datasets show that the proposed EviNote-RAG outperforms existing RL-based agentic RAG such as Search-R1 and R1-Searcher. 
3. The paper conducts extensive ablation studies to validate the effectiveness of key components in the proposed framework.

### Weaknesses
1. The prompt used to instruct the LLM to perform retrieval and generate supportive-evidence notes is missing from the paper. 
2. Some important training details are missing from the paper. For example, it is unclear what training data were used and whether the proposed model underwent a supervised fine-tuning (SFT) stage. This lack of information raises reproducibility concerns, especially since the code is not provided. 
3. The paper memtions that supportive-evidence note generation is optional after each retrieval, but it lacks an in-depth analysis of when and why the model chooses to trigger note generation. An analysis of this decision-making process would help better understand the behavior of the proposed framework and the practical benefits of incorporating the note generation mechanism. 
4. The paper mentions that EviNote-RAG improves efficiency (line 469). However, the paper lack efficiency analysis compared with baselines. Intuitively, compared with Search-R1 or R1-Searcher, I assume the proposed EviNote-RAG is less efficient since it requires an additional note generation phrase.

### Questions
Please see the questions in the Weaknesses section.

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
4

### Summary
This paper mainly improves Search-R1 by adding one \<summary\> action to produce *Supportive-Evidence Notes (SENs)*. The reward strategy encourages the model to taking notes (\<summary\>) that are useful to answer the question (supporting the claim $h$, Eq 1). Strong baselines including Search-R1 are compared with and extensive ablation studies were conducted to study the effects of proposed improvements.

### Strengths
1. Extensive experiments, including comparisons with strong baselines and comprehensive ablation studies, were conducted to demonstrate the effectiveness of the proposed method.
2. To prove the method's generalization, evaluation was performed on both in-domain and out-of-domain datasets.
3. The paper presents a clear analysis of the experimental results.

### Weaknesses
This work improves Search-R1 by training the LLM to summarize useful content from retrieved documents into Supportive-Evidence Notes (SENs). However, several aspects of the methodology require clarification:

1. Reward Computation: The exact calculation method for the reward (Eq. 1) is unclear.
2. Motivation for Eq. 1: Given that the LLM itself can identify whether SEN$_{\text{last}}$ supports the answer to a question, the necessity of the proposed method (Eq. 1) is not well articulated.
3. Figure 3(a) illustrates that the proposed method exhibits superior stability compared to the baseline (Search-R1). However, the paper does not provide an explanation for this increased stability.
4. Missing Details:
**Reward Design:** Regarding the scoring mechanism for the \<Summary\> tag content: are both uncertainty and highlighting required to achieve a score?
**In-Domain/Out-of-Domain Datasets:** In Line 310, "In-Domain Performance... aligned with the training data" suggests that for out-of-domain evaluation, the training and testing datasets are different. However, there is no description provided for the specific datasets used in the "Out-of-Domain" scenario.

### Questions
1. Could you please elaborate on the implementation details of Equation 1?
2. In lines 198–211, the LLM is already capable of judging whether the Supporting Evidence Notes (SEN) support the answer for a given question. Why was an LLM-as-a-judge approach not used directly (e.g., prompting the LLM to determine if the ground-truth answer is present in SEN$_{\text{last}}$ to score the EQR? What specific benefits does Equation 1 offer over an LLM-as-a-judge method?
3. Regarding line 224, could you provide more detail on the criteria used to judge whether the note-taking condition holds?
4. For Equation 2, what is the assigned reward if the response format and the answer are both correct, but the note-taking condition is not met?

### Soundness
3

### Presentation
3

### Contribution
2
