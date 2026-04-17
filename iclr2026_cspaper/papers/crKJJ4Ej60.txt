# Copy-Paste To Mitigate Large Language Model Hallucinations

Yongchao Long1,2 Yingying Zhang3 Xianbin Wen1 Xian Wu3,† **Yuxi Zhou**1,†
Shenda Hong2,†
1Department of Computer Science, Tianjin University of Technology, Tianjin, China 2National Institute of Health Data Science, Peking University, Beijing, China 3Tencent Jarvis Lab, Shenzhen, China
†Corresponding author

## Abstract

While Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to generate contextually grounded responses, contextual faithfulness remains challenging as LLMs may not consistently trust provided context, leading to hallucinations that undermine reliability. We observe an inverse correlation between response copying degree and context-unfaithful hallucinations on RAGTruth, suggesting higher copying degrees reduce hallucinations by fostering genuine contextual belief. We propose **Copy-Paste**, a generation paradigm that directly embeds contextual fragments to ensure faithfulness, and instantiate it through CopyPasteLLM via two-stage high-copying preference training. We design three prompting methods to enhance copying degree, demonstrating that high-copying responses achieve superior contextual faithfulness and hallucination control. These approaches enable a fully automated pipeline that transforms generated responses into high-copying preference data for training CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best performance in both counterfactual and original contexts, remarkably with 12.2% to 24.5% accuracy improvements on FaithEval over the best baseline, while requiring only 365 training samples—*1/50th* of baseline data. To elucidate Copy- PasteLLM's effectiveness, we propose the Context-Parameter Copying Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates reliance on internal parametric knowledge rather than external knowledge during generation. All codes are available at https://github.com/longyongchao/ CopyPasteLLM

## 1 Introduction

Large language models (LLMs) have brought revolutionary breakthroughs to natural language processing (Annepaka & Pakray, 2025; Qin et al., 2024), while retrieval-augmented generation (RAG) further empowers LLMs with grounded external knowledge capabilities (Fan et al., 2024; Zhao et al., 2024). However, LLMs inevitably suffer from knowledge conflicts (Xu et al., 2024) —when internal parametric knowledge conflicts with external contextual knowledge, LLMs may favor internal parametric knowledge, leading to contextual faithfulness hallucinations (Bi et al., 2024; Ming et al., 2025; Niu et al., 2024). Such hallucinations are particularly critical in knowledge-intensive domains (Vishwanath et al., 2024) like rare disease medical consultations (Reese et al., 2025), where clinicians may lack systematic knowledge reserves (Zhang et al., 2022) to judge whether model responses are faithful to contexts, while patient communities often rely on self-consultation or LLM
queries without professional medical supervision (Busch et al., 2025; Aydin et al., 2025). Chen
& Shu (2024); Zhang et al. (2025c) shows LLM-generated content is more deceptive than humanwritten content. Without clear attributability, faithfulness hallucinations pose potential risks to clinical decisions and patient behaviors (Kim et al., 2025). Current research primarily follows two directions in enhancing the reliability of LLMs: (i) generation with citations, where models produce responses accompanied by attributable citations (Wu et al., 2025; Abolghasemi et al., 2025; Ji et al., 2025; Press et al., 2024; Song et al., 2025), and (ii) improving contextual faithfulness through techniques such as prompting strategies (Zhou et al., 2023; Zhang et al., 2025a), constrained decoding (Shi et al., 2024; T.y.s.s et al., 2025; Liu et al., 2025), or fine-tuning (Bi et al., 2025; Huang et al., 2025b; Si et al., 2025; Li et al., 2025a). However, the former struggles to ensure consistency between the generated content and its cited sources, while the latter typically lacks mechanisms for explicit attribution. Consequently, achieving both faithfulness and verifiable attribution remains a critical and unresolved challenge.

![1_image_0.png](1_image_0.png)

To address these challenges, we propose an intuitive solution: rather than having models reinterpret retrieved content, we advocate for directly quoting original sentences. This copy-paste generation strategy embeds key contextual fragments directly, avoiding secondary knowledge processing and potentially reducing paraphrasing hallucination risks. Importantly, copied content itself serves as direct evidence of faithfulness without requiring additional verifiable attribution mechanism. This approach is motivated by our observation of an inverse correlation between copying degree and hallucination density on the RAGTruth dataset (Figure 1), leading us to hypothesize that high copying degrees may help mitigate hallucination problems. Specifically, we formally propose Copy-Paste as a generation paradigm that leverages high-copying degree as an operational proxy for contextual faithfulness through a two-stage pipeline that internalizes surface-level copying behavior into model-level contextual trust. The first stage generates high-copying responses through hard and soft constraints to enhance copying degree. The second stage (**CopyPasteLLM**) applies direct preference optimization (Rafailov et al., 2023) training to internalize the high-copying preferences from the first stage into the LLM's contextual faithfulness. Experimental results demonstrate that CopyPasteLLM, trained on only 365 high-copying samples, outperforms strongest baselines by 12.2%-24.5% on FaithEval. Additionally, we propose the Context-Parameter Copying Capturing algorithm, which enables fine-grained analysis of knowledge source reliance throughout the entire Chain-of-Thought reasoning process, rather than merely examining final short answers. The algorithm captures contextual versus parametric knowledge usage at each token position, providing novel insights into how models dynamically balance different knowledge sources during sequential reasoning. Mechanistic analysis reveals CopyPasteLLM maintains similar contextual knowledge representations as the base model while recalibrating internal confidence in parametric knowledge, thereby enhancing contextual trust.

## 2 Preliminaries 2.1 Problem Formulation

Task Given a query Q and a context C, the model generates an answer A. In high-stakes domains such as medicine, the faithfulness of the generated answer to the context is of paramount importance. While conventional RAG research often emphasizes abstractive generation and semantic relevance, our focus in this work is a specialized task that we term **Copy-Paste**. The goal of Copy-Paste is to maximize the reuse of lexical units from the context C in the final answer A, thereby ensuring high contextual faithfulness and minimizing hallucination. Formally, the task can be defined as: (Q, C) 7→ A. Quantification Following Grusky et al. (2018), we quantify the response copying degree from context with two metrics:

$$\kappa=\frac{1}{|A|}\sum_{f\in\mathcal{F}}|f|,\quad\delta=\frac{1}{|A|}\sum_{f\in\mathcal{F}}|f|^{2}$$

$$(1)$$
2(1)
where F is the set of copy fragments computed by copy fragment detection algorithm (detailed at Appendix I), |·| denotes sequence length. **Copy Coverage (**κ): the fraction of answer tokens that are covered by some copy fragment, reflecting the overall degree of lexical reuse. **Copy Density (**δ):
a length-sensitive variant that emphasizes longer copied fragments, capturing whether the answer tends to copy long spans verbatim rather than isolated words. Balance While maximizing copy-paste is central to our formulation, an effective answer A should also remain relevant to the query Q and be linguistically fluent. Specifically, we measure query relevance using embedding-based similarity, and fluency via perplexity. Thus, the Copy-Paste task can be viewed as optimizing a trade-off among faithfulness, **query relevance**, and **fluency**. Unlike extractive summarization (Zhang et al., 2023), Copy-Paste is query-aware and ensures fluent, context-faithful answers.

## 2.2 Motivating Observation On Ragtruth

To validate the intuition that high copying degrees may reduce hallucination, we conducted a preliminary analysis on the RAGTruth QA subset Niu et al. (2024), which contains 839 context-dependent questions. Each question includes responses from 6 different models with word-level contextual faithfulness hallucination annotations, enabling precise quantification of hallucination density per model. We computed copy coverage (κ) and copy density (δ) for each model's responses across the dataset, then visualized the relationship using two-dimensional kernel density estimation with copy coverage (x-axis) and copy density (y-axis). The analysis reveals a clear pattern: density kernels positioned toward the upper-right region (indicating higher copying coverage and density) correspond to lower hallucination density across models (Figure 1).

## 3 Methodology

Our approach consists of two sequential stages: (1) constructing high-copying candidate responses through Copy-Paste-Prompting methods, and (2) training CopyPasteLLM through automated preference data construction that internalizes a preference for contextual evidence. Figure 2 illustrates the complete pipeline. To verify that the learned policy truly reallocates reliance from parametric priors to context, we additionally introduce an interpretability tool, Context-Parameter Copying Capturing.

![3_image_0.png](3_image_0.png)

![3_image_1.png](3_image_1.png)

## 0. Response From Cp-Prompting And Baselines 3.1 Copy-Paste-Prompting: Constructing High-Copying Responses

```
!
B.
A.
C. [1]
             [1] [1]

```

$ \#
We operationalize the Copy-Paste objective through three complementary prompting paradigms that progressively relax constraints while preserving lexical fidelity to the context. CP-Order implements a strict extractive regime: it first selects context sentences relevant to the query and then directly reorders them into a coherent answer. This hard constraint intentionally forgoes abstractive paraphrasing, which suppresses the model's tendency to resolve conflicts using parametric priors. The method excels when answers can be composed from a small set of highly informative sentences but tends to sacrifice fluency when discourse connectives are missing. (See L.1.1 & L.1.2 for prompts)
CP-Link maintains the same extractive core but allows the model to generate short transitions between copied spans. These transitions are not intended to introduce new facts; instead, they serve as discourse glue to restore local coherence after sentence reordering. Empirically, this limited generative freedom improves readability while preserving the high-copying signature that anchors the answer to source text. (See L.1.1 & L.1.3 for prompts) In contrast, CP-Refine adopts a soft-constraint, iterative refinement process with a writer–reviewer loop. The writer proposes an answer given the query and context; the reviewer provides verbal feedback focused on copying degree, contextual faithfulness, query relevance, and fluency; the writer then revises the answer until a composite copy score exceeds a threshold. This procedure treats copying as a target state that is continually optimized rather than a fixed structural constraint. As shown by our experiments (See Table 2), CP-Refine achieves a better balance among faithfulness, readability, and relevance (See L.1.4 for prompts). Algorithm 1 in Appendix summarizes the unified procedure, which we use to produce diverse yet consistently high-copying candidates for downstream preference construction.

## 3.2 Copypastellm: Internalizing Contextual Trust From High-Copying Preferences

Copy-Paste-Prompting supplies not only single responses but a structured spectrum of behaviors—from strictly extractive to softly refined. CopyPasteLLM converts this spectrum into explicit preferences that can be internalized by a policy through direct preference optimization. Our pipeline begins by generating six types of candidates for each query–context pair: conventional abstractive baselines (Base, Attributed, Citations) and three Copy-Paste variants (CP-Order, CP-Link, CP- Refine). We then perform multi-criteria filtering that simultaneously enforces contextual faithfulness (AlignScore, MiniCheck), copying strength (κ, δ), query relevance (embedding similarity), and fluency (perplexity). This step ensures the retained set covers a high-quality front of the faithfulness–fluency–relevance trade space rather than merely maximizing copying. The remaining candidates are ranked by an Elo-style LLM-as-Judge tournament that diagnoses two major hallucination modes—Twist and Causal—so the final preference reflects error severity, not only stylistic quality. A key nuance arises when gold answers are available: we append the correct answer to the top Copy-Paste candidate to transform faithful reasoning into a definitive conclusion, while appending incorrect answers to the other Copy-Paste candidates to create informative negative pairs. This labeling strategy focuses learning on trusting context while disentangling reasoning traces from final decisions. The resulting dataset yields roughly five preference pairs per sample, enabling data-efficient DPO training that teaches the model to prefer high-copying, context-grounded responses even when they conflict with parametric priors. Algorithm 2 in Appendix formalizes the procedure.

## 3.3 Context-Parameter Copying Capturing

Context-Parameter Copying Capturing provides a principled, token-level probe of knowledge usage during generation. The method executes two runs for each query: with context and without context. At each decoding step in Chain-of-Thought mode, it collects the top-K candidate tokens with their probabilities and hidden states. Tokens that appear in the provided context are taken as contextual knowledge, whereas tokens that are preferred in the context-free run serve as proxies for parametric knowledge. Algorithm 4 specifies the full procedure. Conceptually, this procedure is inspired by Knowledge Token Capturing (KTC) (Bi et al., 2024).

Unlike KTC, which primarily analyzes short final answers, our Context-Parameter Copying Capturing extends the analysis to the entire Chain-of-Thought response trajectory, enabling sequential, position-aware assessment of contextual versus parametric reliance.

## 4 Experiment

Our Copy-Paste approach is a two-stage framework where Copy-Paste-Prompting generates highcopying preference data, and CopyPasteLLM learns contextual faithfulness from this data. To validate our complete pipeline, we conduct comprehensive experiments addressing three key research questions:
- RQ1: Do Copy-Paste-Prompting methods effectively enhance contextual faithfulness and mitigate RAG hallucinations through high-copying response generation?

- RQ2: Does training with high-copying responses from Copy-Paste-Prompting as DPO
preference trajectories enable CopyPasteLLM to genuinely trust contextual knowledge—even when it is counterfactual?

- RQ3: What are the underlying mechanisms of CopyPasteLLM's contextual belief? We will interpret this by analyzing logits and hidden states.

## 4.1 Two-Stage Framework Validation

Experimental setup is detailed in Appendix B. 4.1.1 STAGE 1: COPY-PASTE-PROMPTING AS PREFERENCE DATA GENERATOR (RQ1) In the first stage, we evaluate whether our prompting methods can effectively generate responses with high-copying and improved contextual faithfulness. The baselines here represent different response generation paradigms that will serve as rejected responses in our CopyPasteLLM training. Our primary objectives are to: (1) validate that Copy-Paste-Prompting methods achieve superior

Model Method **Training**

Size

FaithEval ConFiQA-QA ConFiQA-MR ConFiQA-MC

Acc Hit Acc Hit Acc Hit Acc Hit

Llama

-

38B

Context-DPO (Bi et al., 2025) 18,000 80.2 36.7 88.9T96.1T88.4T85.8T92.1T80.9T

Attributed (Zhou et al., 2023) - 67.1 34.2 51.5 91.4 53.3 71.5 37.3 53.6 CoCoLex (T.y.s.s et al., 2025) - 69.2 17.9 48.5 37.4 53.9 14.8 36.1 15.5 Canoe (Si et al., 2025) 10,000 71.4 34.0 64.3 93.2 66.6 **83.8** 64.5 73.7 ParamMute (Huang et al., 2025b) 32,580 68.5 22.5 74.4 82.2 75.5 72.4 81.4 70.2 CopyPasteLLM (Ours) 365 **92.8 37.2 83.6 96.7 80.9** 83.4 **86.8 75.9**

Mis

tral7B

-v

0.

2

Context-DPO (Bi et al., 2025) 18,000 77.1 33.8 84.8T94.8T81.3T85.3T80.4T80.8T

Attributed (Zhou et al., 2023) - 65.6 32.0 56.6 84.4 29.2 69.8 39.0 57.4

CoCoLex (T.y.s.s et al., 2025) - 65.3 35.4 57.3 50.8 41.8 33.5 32.5 33.7

CopyPasteLLM (Ours) 365 **89.3 41.8 84.4 95.0 80.8 90.8 82.5 86.3**

Llama3.

1

-

8BAttributed (Zhou et al., 2023) - 65.5 32.0 49.9 88.4 39.8 69.2 15.5 52.6

CoCoLex (T.y.s.s et al., 2025) - 68.1 36.2 48.5 57.3 40.4 38.4 13.5 37.2 CopyPasteLLM (Ours) 365 **92.6 41.0 72.4 90.1 75.4 84.8 83.5 79.9**

contextual faithfulness through explicit copying mechanisms, and (2) generate high-quality preferred responses for subsequent DPO training. A comprehensive comparison with state-of-the-art methods will be presented in the next stage after DPO training. Table 2: Performance comparison of Copy-Paste-Prompting against baselines across models and datasets. Methods with colored backgrounds are our proposed Copy-Paste-Prompting. **Bold** indicates the best performance, underlined indicates the second-best performance. *Faith.*: Faithfulness
(*M.C.*: MiniCheck, *A.S.*: AlignScore), *Hallu.*: Hallucination, *Flu.*: Fluency. Our experimental results demonstrate that Copy-Paste-Prompting methods consistently outperform baselines across all evaluation metrics (Table 2). **(1) CP-Refine** excels in hallucination reduction (best in 3/4 models, 14/24 top scores) and contextual faithfulness (+10.9% to 19.1% over baselines) while maintaining fluency—achieving best perplexity in Q-72B/D-V3 and second-best in M- 7B/L-8B, suggesting advanced models better handle high-copying constraints. **(2) CP-Order** leads contextual faithfulness (14/24 top scores) with second-best hallucination performance but notably poorer fluency. **(3) CP-Link** shows modest improvements, excelling only in contextual faithfulness

| Method                        | RAGTruth                        | FaithEval                       | PubmedQA                        | AVERAGE            |      |        |        |      |                    |
|-------------------------------|---------------------------------|---------------------------------|---------------------------------|--------------------|------|--------|--------|------|--------------------|
| Faith.                        | Hallu.                          | Flu.                            | Faith.                          | Hallu.             | Flu. | Faith. | Hallu. | Flu. | Faith. Hallu. Flu. |
| M.C. A.S. Twist Causal        | M.C. A.S. Twist Causal          | M.C. A.S. Twist Causal          |                                 |                    |      |        |        |      |                    |
| Mistral-7B-Instruct-v0.2 (7B) |                                 |                                 |                                 |                    |      |        |        |      |                    |
| Attributed                    | 69.58 63.43 1506.9 1494.5 19.54 | 88.28 90.67 1527.1 1513.7 37.32 | 75.49 77.90 1464.7 1450.4 23.53 | 77.56 1492.9 26.80 |      |        |        |      |                    |
| Citations                     | 57.82 49.39 1472.5 1475.7 14.41 | 73.50 74.25 1392.1 1416.2 27.98 | 55.79 52.35 1415.9 1370.0 13.93 | 60.52 1423.7 18.77 |      |        |        |      |                    |
| CP-Link                       | 89.39 75.45 1518.9 1519.5 73.33 | 93.41 92.44 1510.9 1521.9 49.40 | 96.50 88.52 1518.4 1580.7 35.57 | 89.29 1528.4 52.77 |      |        |        |      |                    |
| CP-Order                      | 91.25 71.98 1467.9 1472.4 65.62 | 94.89 92.27 1522.6 1501.5 43.74 | 93.18 82.35 1528.3 1559.1 32.65 | 87.65 1508.6 47.34 |      |        |        |      |                    |
| CP-Refine                     | 82.18 74.56 1533.8 1537.9 18.46 | 92.85 94.68 1547.4 1546.7 26.63 | 91.52 88.21 1572.7 1539.7 17.79 | 87.33 1546.4 20.96 |      |        |        |      |                    |
| Llama-3.1-8B-Instruct (8B)    |                                 |                                 |                                 |                    |      |        |        |      |                    |
| Attributed                    | 57.02 65.29 1526.3 1554.3 26.22 | 85.22 85.65 1516.5 1536.9 330.8 | 71.10 60.01 1530.0 1553.1 47.36 | 70.72 1536.2 134.8 |      |        |        |      |                    |
| Citations                     | 64.27 72.81 1428.5 1574.4 16.78 | 88.81 86.80 1486.2 1555.6 39.65 | 78.56 73.03 1403.4 1463.4 19.11 | 77.38 1485.3 25.18 |      |        |        |      |                    |
| CP-Link                       | 70.58 78.83 1401.1 1328.3 17.83 | 91.54 89.23 1456.2 1366.3 24.09 | 80.74 80.79 1396.4 1371.1 19.65 | 81.95 1386.6 20.52 |      |        |        |      |                    |
| CP-Order                      | 75.30 94.81 1498.4 1498.0 26.35 | 95.44 98.12 1523.2 1541.2 33.46 | 87.07 97.62 1633.6 1559.1 27.83 | 91.39 1542.3 29.21 |      |        |        |      |                    |
| CP-Refine                     | 77.30 88.52 1645.7 1545.0 17.75 | 94.40 93.71 1517.9 1500.1 26.99 | 87.29 91.19 1536.5 1553.2 18.64 | 88.74 1549.7 21.13 |      |        |        |      |                    |
| Qwen2.5-72B-Instruct (72B)    |                                 |                                 |                                 |                    |      |        |        |      |                    |
| Attributed                    | 57.00 62.23 1504.5 1525.5 19.68 | 85.74 83.03 1537.3 1490.0 293.8 | 77.99 69.25 1509.9 1441.5 33.42 | 72.54 1501.5 115.6 |      |        |        |      |                    |
| Citations                     | 74.32 77.52 1455.5 1498.0 18.61 | 90.98 88.30 1456.5 1476.7 34.67 | 82.01 76.62 1358.8 1413.6 22.89 | 81.63 1443.2 25.39 |      |        |        |      |                    |
| CP-Link                       | 75.75 85.37 1446.3 1363.2 27.47 | 92.88 92.00 1443.5 1424.2 39.55 | 86.21 88.58 1527.9 1489.2 33.43 | 86.80 1449.1 33.48 |      |        |        |      |                    |
| CP-Order                      | 76.32 94.60 1509.2 1589.6 30.56 | 95.78 98.16 1539.3 1579.7 38.11 | 87.85 97.52 1546.8 1575.9 35.26 | 91.71 1556.8 34.65 |      |        |        |      |                    |
| CP-Refine                     | 78.14 90.88 1584.6 1523.7 20.12 | 94.72 95.48 1523.4 1529.4 27.65 | 88.88 95.04 1556.7 1579.9 20.29 | 90.52 1549.6 22.69 |      |        |        |      |                    |
| DeepSeek-V3-0324 (671B)       |                                 |                                 |                                 |                    |      |        |        |      |                    |
| Attributed                    | 56.42 59.60 1417.1 1449.1 27.52 | 86.90 83.46 1524.3 1535.0 63.27 | 75.56 69.24 1449.2 1487.9 36.88 | 71.86 1477.1 42.56 |      |        |        |      |                    |
| Citations                     | 62.32 64.45 1510.8 1565.6 34.63 | 87.38 85.69 1463.0 1477.0 36.09 | 75.93 71.85 1460.4 1387.5 23.27 | 74.60 1477.4 31.33 |      |        |        |      |                    |
| CP-Link                       | 70.59 72.54 1382.9 1360.3 34.19 | 92.60 88.08 1489.1 1374.8 35.55 | 81.56 77.67 1380.9 1351.1 28.54 | 80.51 1389.9 32.76 |      |        |        |      |                    |
| CP-Order                      | 75.53 92.87 1579.4 1555.2 59.11 | 95.23 97.79 1569.9 1548.1 34.30 | 87.20 97.38 1561.8 1621.7 27.56 | 91.00 1572.7 40.32 |      |        |        |      |                    |
| CP-Refine                     | 77.14 90.02 1609.8 1569.7 22.57 | 94.45 93.06 1453.7 1565.2 33.84 | 87.39 91.05 1647.7 1651.7 21.91 | 88.85 1583.0 26.11 |      |        |        |      |                    |

with even worse fluency than CP-Order, indicating hard constraints limit generative capabilities. (4) We observe **strong hallucination-faithfulness correlation**: in 18/24 scenarios (75%), optimal hallucination performance coincides with best contextual faithfulness. We hypothesize that the superior contextual faithfulness of Copy-Paste-Prompting stems from high-copying in responses. Copy- Paste-Prompting achieves significantly higher copying degree than the two baselines (see Appendix Figure 5). Additionally, we compare query relevance between the three Copy-Paste-Prompting methods and the strongest baseline in Appendix Figure 6, demonstrating that Copy-Paste-Refine can address queries while maintaining high copying rates through soft constraints.

## 4.1.2 Stage 2: Copypastellm (Rq2)

Table 3: Accuracy in non-counterfactual settings. PubMedQA is evaluated on artificial subset 20,000 samples (none used for CopyPasteLLM training, see Appendix Table 4). ConFiQA uses Original context and Original answers.

| Mistral-7B-v0.2     | Llama-3-8B   | Llama-3.1-8B      |         |                   |         |                   |       |    |
|---------------------|--------------|-------------------|---------|-------------------|---------|-------------------|-------|----|
| Method              | AVG          |                   |         |                   |         |                   |       |    |
| PubMed              | ConFiQA      | PubMed            | ConFiQA | PubMed            | ConFiQA |                   |       |    |
| QA                  | QA           | QA                |         |                   |         |                   |       |    |
| QA                  | MR           | MC                | QA      | MR                | MC      | QA                | MR    | MC |
| Base                | 88.60        | 96.22 71.20 72.27 | 97.3    | 98.02 93.00 91.02 | 98.15   | 97.93 89.48 89.97 | 90.26 |    |
| CopyPasteLLM (Ours) | 91.40        | 97.43 91.87 91.20 | 97.5    | 99.30 97.17 96.27 | 97.67   | 99.02 94.95 94.92 | 95.73 |    |

CopyPasteLLM demonstrates remarkable efficiency by achieving superior performance in counterfactual scenarios using only 365 query-context pairs as input to construct preference data through our automated pipeline—a base data requirement that is 50× smaller than the strongest baseline Context-DPO (18,000 samples) and significantly more efficient than other fine-tuning methods such as Canoe (10,000) and ParamMute (32,580). As shown in Table 1, on the FaithEval counterfactual subset, CopyPasteLLM surpasses the strongest baselines by substantial margins: 12.6, 12.2, and 24.5 percentage points across Llama-3-8B, Mistral-7B-v0.2, and Llama-3.1-8B respectively, achieving a peak accuracy of 92.8% on Llama-3-8B—remarkably outperforming GPT-4o's reported 47.5% on this challenging subset (see Appendix Table 6). Additionally, CopyPasteLLM
consistently achieves the highest Hit Rate across all models, despite the inherent difficulty of exact matching in FaithEval's lengthy gold standard answers. On ConFiQA's three counterfactual subsets, CopyPasteLLM maintains superior performance in unseen settings compared to recent fine-tuning baselines and copy-guided decoding method CoCoLex, with particularly notable results on Mistral7B-v0.2 where it outperforms even Context-DPO trained on ConFiQA on the most challenging Multi-Conflict subset. In non-counterfactual scenarios, CopyPasteLLM maintains exceptional contextual faithfulness while demonstrating significant improvements over base models (Table 3). On relatively straightforward datasets—PubMedQA and ConFiQA-QA—the method achieves modest but consistent improvements, with average accuracy gains of 1.01% (from 96.04% to 97.05%). More importantly, on the more challenging ConFiQA-MR and ConFiQA-MC subsets, CopyPasteLLM delivers substantial performance gains, improving average accuracy from 84.49% to 94.37%, with the most dramatic improvement of 20.67% observed on Mistral-7B-v0.2 for the MR subset. These results demonstrate that CopyPasteLLM's enhanced contextual trust, achieved without introducing additional parametric knowledge through LoRA training, leads to significant improvements in knowledge-intensive question answering accuracy. For fine-grained analysis across conflict complexity, knowledge domains, and reasoning ambiguity, see Appendix E; for response length and copying behavior analysis, see Appendix F; for ablation studies and training dynamics, see Appendix G.

## 4.2 Interpretable Analysis Of Copypastellm (Rq3)

We propose the Context-Parameter Copying Capturing (Algorithm 4), which is designed to capture the degree to which the model copies contextual or parametric knowledge during token generation.

Specifically, in CoT reasoning mode, our method monitors the model's internal representations by analyzing the top-K token logits (ranked by probability) and corresponding hidden states at each generation step, thereby quantifying the model's reliance on external context versus internal parametric knowledge. This algorithm extends the Knowledge Token Capturing (Bi et al., 2024) to sequential analysis, enabling comprehensive evaluation of model responses during CoT reasoning.

![7_image_0.png](7_image_0.png) 

![7_image_1.png](7_image_1.png)

We first analyze the logits output power of CopyPasteLLM and its base models across three datasets at each generation step, considering both the magnitude and frequency of logits at specific response positions, as illustrated in Figure 3. To ensure fair comparison by providing base with longer token generation opportunities, we filtered out samples where CopyPasteLLM responses exceeded base response lengths, with complete dataset statistics shown in Appendix Figure 13. Our analysis reveals three key observations: (1) In CoT with context task, Both base and CopyPasteLLM demonstrate higher reliance on contextual knowledge than parametric knowledge. (2) However, CopyPasteLLM exhibits significantly stronger contextual knowledge utilization compared to base, while showing reduced reliance on parametric knowledge. (3) From a positional perspective, Copy- PasteLLM achieves peak contextual knowledge utilization earlier in the response generation process than base. Collectively, these findings suggest that CopyPasteLLM not only demonstrates stronger but also earlier contextual engagement compared to base, indicating enhanced contextual trust and willingness to *believe* the provided context. We further employ UMAP dimensionality reduction to analyze the captured hidden states distributions, as shown in Figure 4. Our visualization reveals two striking patterns: (1) Base models exhibit minimal distinction between contextual and parametric knowledge semantic representations (1st column), whereas CopyPasteLLM demonstrates relatively clear separation between these two knowledge types (2nd column). (2) More intriguingly, contextual knowledge representations in CopyPasteLLM remain nearly co-distributed with those in base models (3rd column), while their parametric knowledge distributions differ substantially (4th column). Based on these observations, we infer that CopyPasteLLM fundamentally recalibrates the model's internal confidence in parametric knowledge without compromising its contextual processing capabilities. This selective parametric knowledge suppression, rather than contextual knowledge enhancement, enables CopyPasteLLM to achieve superior contextual faithfulness by strategically reducing competition from internal parametric knowledge during generation. For a theoretical interpretation of this mechanism through attention dynamics and entropy reduction, see Appendix A.

## 5 Related Work

While Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language models in external knowledge (Fan et al., 2024; Zhao et al., 2024), ensuring contextual faithfulness remains an open challenge. LLMs often exhibit a tendency to rely on their pretrained parametric knowledge rather than adhering to the provided context, resulting in responses that may contradict or ignore retrieved evidence (Niu et al., 2024; Bi et al., 2024; Ming et al., 2025).

This contextual unfaithfulness poses significant concerns in critical applications such as healthcare (Vishwanath et al., 2024; Kim et al., 2025), where accuracy and reliability are paramount. Existing research has systematically studied this phenomenon from evaluation and mechanistic perspectives. Evaluation studies construct synthetic scenarios revealing LLMs' propensity to favor internal knowledge over external evidence (Xu et al., 2024; Li et al., 2025b; Joren et al., 2025; Goyal et al., 2025). Mechanistic analyses identify attention heads (Wu et al., 2024; Huang et al., 2025a), FFNs (Sun et al., 2024) and logit distributions (Bi et al., 2024) that respectively process external and internal knowledge sources. Solutions to improve contextual faithfulness include generation with citations (Gao et al., 2023; Press et al., 2024; Song et al., 2025; Wu et al., 2025), prompt engineering (Zhou et al., 2023; Zhang et al., 2025a), decoding methods (Shi et al., 2024; T.y.s.s et al., 2025; Liu et al., 2025) and finetuning (Bi et al., 2025; Si et al., 2025; Li et al., 2025a; Huang et al., 2025b). While generation with citations methods may lack content-source consistency and other approaches often provide limited attribution mechanisms, our Copy-Paste paradigm targets both challenges simultaneously: it enhances contextual faithfulness through direct lexical reuse from source text while inherently providing transparent attribution, and internalizes this copying behavior into genuine model-level contextual trust through preference optimization.

## 6 Conclusion

We propose Copy-Paste, a generation paradigm that directly embeds contextual fragments into responses to mitigate faithfulness hallucinations in RAG systems. Based on the observed inverse correlation between copying degree and hallucination density, we instantiate this paradigm through a two-stage framework: Copy-Paste-Prompting methods first generate high-copying responses, then preference optimization internalizes contextual trust into CopyPasteLLM. CopyPasteLLM achieves remarkable data efficiency, delivering 12.2%-24.5% improvements on FaithEval using only 365 training samples—50× smaller than existing baselines. Our Context-Parameter Copying Capturing analysis reveals that effectiveness stems from recalibrating parametric knowledge confidence rather than enhancing contextual representations. The copy-paste paradigm provides an elegant solution to RAG attribution challenges, where copied content serves as inherent faithfulness evidence without requiring additional verification mechanisms. We discuss limitations and future directions in Appendix K.

## 7 Ethics Statement

This work addresses the critical challenge of contextual faithfulness in large language models, particularly in high-stakes domains such as healthcare. While our CopyPasteLLM approach aims to reduce hallucinations by promoting direct copying from provided context, we acknowledge potential risks: over-reliance on copied content may lead to verbatim reproduction of potentially biased or incorrect source material. The method's effectiveness depends on the quality and accuracy of the provided context, and users should exercise caution when applying this approach in sensitive applications. We encourage responsible deployment with appropriate human oversight and validation mechanisms.

## 8 Reproducibility Statement

To ensure reproducibility, we provide the following: (1) All experimental details and hyperparameters are documented in the appendix. (2) We use publicly available datasets (FaithEval, ConFiQA, PubMedQA, RAGTruth) with standard evaluation protocols (see Appendix B). (3) Model training details, including DPO hyperparameters (see Appendix D) and preference data construction procedures (see Algorithm 1 and 2). (4) The Context-Parameter Copying Capturing algorithm is fully described in Algorithm 4. (5) All prompting templates for Copy-Paste- Prompting methods are provided in Appendix L. The complete implementation is available at https://github.com/longyongchao/CopyPasteLLM.

## Acknowledgments

We sincerely thank the anonymous reviewers and area chairs for their insightful comments and constructive suggestions that helped improve this paper. This work was supported by the National Natural Science Foundation of China (62102008, 62172018, 62202332, 62376197, 62020106004, 92048301), the CCF-Tencent Rhino-Bird Open Research Fund (CCF-Tencent RAGR20250108), the CCF-Zhipu Large Model Innovation Fund (CCF-Zhipu202414), the Tianjin Science and Technology Program (23JCYBJC00360), the Key Research and Development Program of Shaanxi Province (2023-ZDLGY-48), the Tianchi Elite Youth Doctoral Program (CZ002701, CZ002707), the PKU- OPPO Fund (BO202301, BO202503), the Research Project of Peking University in the State Key Laboratory of Vascular Homeostasis and Remodeling (2025-SKLVHR-YCTS-02), and the Beijing Municipal Science and Technology Commission (Z251100000725008).

## References

Amin Abolghasemi, Leif Azzopardi, Seyyed Hadi Hashemi, Maarten de Rijke, and Suzan Verberne. Evaluation of attribution bias in generator-aware retrieval-augmented large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 21105–21124, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-2565. doi: 10.18653/v1/2025.findings-acl.1087. URL https://aclanthology.org/2025. findings-acl.1087/.

Yadagiri Annepaka and Partha Pakray. Large language models: a survey of their development, capabilities, and applications. Knowledge and Information Systems, 67(3):2967–3022, 2025.

Serhat Aydin, Mert Karabacak, Victoria Vlachos, and Konstantinos Margetis. Navigating the potential and pitfalls of large language models in patient-centered medication guidance and selfdecision support. Frontiers in Medicine, 12:1527864, 2025.

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Junfeng Fang, Hongcheng Gao, Shiyu Ni, and Xueqi Cheng. Is factuality enhancement a free lunch for LLMs? Better factuality can lead to worse context-faithfulness. In ICLR 2025, October 2024.

Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen Huang, Lingrui Mei, Junfeng Fang, Zehao Li, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang, and Shenghua Liu. Context-DPO: Aligning language models for context-faithfulness. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 10280–10300, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025. findings-acl.536. URL https://aclanthology.org/2025.findings-acl.536/.

Felix Busch, Lena Hoffmann, Christopher Rueger, Elon HC van Dijk, Rawen Kader, Esteban Ortiz-
Prado, Marcus R Makowski, Luca Saba, Martin Hadamitzky, Jakob Nikolas Kather, et al. Current applications and challenges in large language models for patient care: a systematic review. Communications Medicine, 5(1):26, 2025.

Canyu Chen and Kai Shu. Can LLM-generated misinformation be detected? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=ccxD4mtkTU.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 6491–6501, 2024.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 6465–6488, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 398. URL https://aclanthology.org/2023.emnlp-main.398/.

Sachin Goyal, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Context-parametric inversion: Why instruction finetuning may not actually improve context reliance. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=SPS6HzVzyt.

Max Grusky, Mor Naaman, and Yoav Artzi. Newsroom: A dataset of 1.3 million summaries with diverse extractive strategies. In Marilyn Walker, Heng Ji, and Amanda Stent (eds.),
NAACL-HLT 2018, pp. 708–719. Association for Computational Linguistics, 2018. doi: 10.18653/v1/N18-1065. URL https://aclanthology.org/N18-1065/.

Lei Huang, Xiaocheng Feng, Weitao Ma, Yuchun Fan, Xiachong Feng, Yangfan Ye, Weihong Zhong, Yuxuan Gu, Baoxin Wang, Dayong Wu, Guoping Hu, and Bing Qin. Improving contextual faithfulness of large language models via retrieval heads-induced optimization. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16896–16913, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.826. URL https: //aclanthology.org/2025.acl-long.826/.

Pengcheng Huang, Zhenghao Liu, Yukun Yan, Haiyan Zhao, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong Xiao, Ge Yu, and Chenyan Xiong. Parammute: Suppressing knowledgecritical ffns for faithful retrieval-augmented generation. In Proceedings of the 39th International Conference on Neural Information Processing Systems, NIPS '25, Red Hook, NY, USA, 2025b. Curran Associates Inc. URL https://neurips.cc/virtual/2025/poster/119254.

Bin Ji, Huijun Liu, Mingzhe Du, Shasha Li, Xiaodong Liu, Jun Ma, Jie Yu, and See-Kiong Ng.

Towards verifiable text generation with generative agent. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 24230–24238, 2025.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA: A
dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL https://aclanthology.org/ D19-1259/.

Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian.

Sufficient context: A new lens on retrieval augmented generation systems. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=Jjr2Odj8DJ.

Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Mingyu Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, Rodrigo Gameiro, et al. Medical hallucinations in foundation models and their impact on healthcare. arXiv preprint arXiv:2503.05777, 2025.

Kun Li, Tianhua Zhang, Yunxiang Li, Hongyin Luo, Abdalla Mohamed Salama Sayed Moustafa, Xixin Wu, James R. Glass, and Helen M. Meng. Generate, discriminate, evolve: Enhancing context faithfulness via fine-grained sentence-level self-evolution. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 17091–17105, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025. findings-acl.878. URL https://aclanthology.org/2025.findings-acl.878/.

Yuepei Li, Kang Zhou, Qiao Qiao, Bach Nguyen, Qing Wang, and Qi Li. Investigating context faithfulness in large language models: The roles of memory strength and evidence style. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of ACL 2025, pp. 4789–4807, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-256-5.

Zhining Liu, Rana Ali Amjad, Ravinarayana Adkathimar, Tianxin Wei, and Hanghang Tong. Self-
Elicit: Your language model secretly knows where is the relevant evidence. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9153–9173, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.448. URL https://aclanthology. org/2025.acl-long.448/.

Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong, and Shafiq Joty. FaithEval: Can your language model stay faithful to context, even if "the moon is made of marshmallows". In ICLR 2025, 2025. URL https://openreview.net/forum?

id=UeVx6L59fg.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), ACL 2024, pp. 10862–10878. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.

585. URL https://aclanthology.org/2024.acl-long.585/.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.