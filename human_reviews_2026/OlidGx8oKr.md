# Incentive-Aligned Multi-Source LLM Summaries

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source’s stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source’s incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Truthful Text Summarization (TTS), a multi-document text summarization framework that promotes peer-agreement among the sources on the facts in the summary and scores each source's stance on each atomic claim from the summary. In this way, TTS achieves summary faithfulness in noisy and adversarial conditions (where some documents are manipulative or noisy).

The authors provide a substantial mathematical framework of peer prediction applied to summarization under incentives.
They conduct experiments on two datasets, NaturalQuestions and ClashEval (300-document subsets) which they synthetically augment to contain faithful and unfaithful sources for each query. They use Gemini-2.5-flash as the LLM backbone and compare TTS to 3 baseline methods: Initial Summary (multi-doc summary), Majority Prompt (an LLM summary that only includes majority claims), and Majority Claims (majority claims are used to obtain a re-summary). In automatic evaluation, TTS strongly outperforms all the baselines in terms of Precision, Recall, F1 Score and Accuracy (calculated via an LLM-as-a-Judge).

### Strengths
* a novel technique is introduced that frames summarization with incentive as peer prediction
* a substantial theoretical framework is presented for informative corroboration among multiple sources
* the technique outperforms a series of baselines on NaturalQuestions and ClashEval synthetically adapted for the task

### Weaknesses
* evaluation datasets, although originally containing real-world data, heavily rely on data synthesis - both truthful sources (paraphrases) and untruthful ones are synthesized using an LLM. For evaluating a technique that claims to be robust against strategic misinformation / manipulation attacks, this does not seem to be enough
* evaluation would also be much more complete with human judgements which could expose what rather brittle P, R, F1 & LLMaaJ (a bit less so) cannot capture. ROUGE / BLEU for fluency evaluation are questionable

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes TTS, a mechanism-design-based framework that incentivizes LLMs to prefer truthful over adversarial or uninformative sources during multi-source summarization. The core idea is to apply LOO peer-prediction scoring mechanism that assigns reliability scores to individual sources and rewards summaries consistent with high-reliability evidence. By reframing summarization as an incentive alignment problem rather than a post-hoc filtering task, the paper aims to structurally encourage honesty in LLM-generated summaries.

### Strengths
- Reframes summarization robustness as an incentive-alignment problem, introducing a peer-prediction mechanism rarely explored in LLM summarization.
- Demonstrates measurable suppression of uninformative and adversarial sources, yielding large factual-accuracy improvements.
- Provides empirical evidence that truthful reporting tends to emerge as the dominant strategy, as dishonest stances consistently reduce expected rewards.

### Weaknesses
- Although these assumptions (A1–A3) are standard in peer-prediction theory, they seem overly restrictive and arguably unnecessary for LLM summarization. Real-world sources are correlated through shared training data and retrieval biases, making the independence assumptions theoretically elegant but empirically weak.
- TTS requires each source to generate, cross-evaluate, and aggregate stances in a leave-one-out manner, leading to a quadratic number of model calls. The paper provides no runtime or scalability analysis, raising doubts about its practical feasibility for large-scale use.
- The evaluation relies mainly on internal ablations rather than strong external baselines. While TTS is positioned as conceptually distinct from retrieval-robust RAG systems, both aim to improve factual reliability under noisy evidence. Direct or qualitative comparisons with recent methods would better clarify TTS’s relative robustness and contribution.

### Questions
- What happens when all sources are unreliable? Does the system abstain or hallucinate?
- Does the incentive-based scoring scale to multi-hop or long-context summarization?
- Can TTS be integrated into the retrieval stage to detect adversarial documents earlier and promote a richer set of high-quality, reliable sources from the start?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the important challenge of building LLM-based RAG summarization systems that are incentive-robust. They propose TTS, a mechanism for decomposing summaries into claims and using hold-one-out evaluate of each source to determine whether it scores highly enough under a peer-consistency-based reliability score to remain in the retrieved context.
The authors build a sophisticated theoretical framework modeling source claims, player strategic incentives, and the mechanisms for whether a claim is reported in order to show that the approach incentivizes truth-telling from all sources. 
The approach is evaluated over 300 QA pairs from two datasets (NQ and ClashEval) for which the authors construct synthetic scenarios in which sources have mixed truthfulness.

### Strengths
S1. The proposed algorithm addresses an important, understudied problem of incentive alignment and is of interested to researchers working on creating adversary-robust RAG approaches.

S2. The authors provide substantial valuable theoretical backing to show that the proposed approach disincentivizes false reporting. The extent of the analysis is comendable. 

S3. Experiments model multiple interesting scenarios, including coordinated and uncoordinated adversarial behavior. In these scenarios, the results seem convincing.

### Weaknesses
W1. The experiments include only 300 datapoints for 2 datasets; this is substantially less than a typical robust evaluation. The paper could benefit from larger datasets and at least one more, noticeably different source/domain for which RAG models are typically used. The scenarios constructed from these datasets are also synthetically generated and not necessarily reflective of real world RAG QA. The Sources (reliable, deceptive, and adversarial) are all LLM generated and may have artifacts that are easier to detect (i.e. less subtle) than real ones.

W2. The approach is substantially slower and more computationally intensive than typical RAG -- it requires generating stance classifications for every (claims, document) in the cross-product of claims/documents in all retrieved sources except a claims's original source. (The claims also need to be extracted, but this can be pre-computed in a RAG index). The paper could benefit from a real-world case study. 

W3. The writing can often be dense and inaccessible. While the framework is interesting, it reads as aggressively over-mathematized. The authors could benefit from providing simple, prosaic, direct summaries of components of the theoretical framework before diving into some of the equations, some of which might be better appendixed. Some of the wording is also generally difficult to digest; the paragraph that conveys some of the contributions P80-86, is jargon-heavy and quite difficult to understand even after multiple passes by a native speaker with a PhD in the field. e.g. these lines are hard to read: L.83-85 "Signals are embedded in prose: [...] formalize implementability and an equivalence to the standard signal–report model."

### Questions
Q1. Does source selection that disincentivizes a source from having a unique claim end up hurting downstream QA because it decreases diversity in the retrieved context?

Q2. The theoretical analysis makes the modeling assumption that all claims are exchangeable, but is this ever true in practice? Some kinds of sources are likely avoid claims about controversial/uncertain topics and are more likely to report claims that are newsworthy. Therefore it seems that $Pr(Q_{ik} = 1 \mid T_i)$ cannot be boiled down to a constant, it is dependent on a claim's type (e.g. "controversial", "speculative", etc) that is not exchangeable.

### Soundness
3

### Presentation
3

### Contribution
3
