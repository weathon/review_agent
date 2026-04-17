# Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevant Assessment for IR Benchmarks

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Information retrieval (IR) evaluation remains challenging due to incomplete IR benchmark datasets that contain unlabeled relevant chunks. While LLMs and LLM-human hybrid strategies reduce costly human effort, they remain prone to LLM overconfidence and ineffective AI-to-human escalation. To address this, we propose DREAM, a multi-round debate-based relevance assessment framework with LLM agents, built on opposing initial stances and iterative reciprocal critique. Through our agreement-based debate, it yields more accurate labeling for certain cases and more reliable AI-to-human escalation for uncertain ones, achieving 95.2% labeling accuracy with only 3.5% human involvement. Using DREAM, we build BRIDGE, a refined benchmark that mitigates evaluation bias and enables fairer retriever comparison by uncovering 29,824 missing relevant chunks. We then re-benchmark IR systems and extend evaluation to RAG, showing that unaddressed holes not only distort retriever rankings but also drive retrieval–generation misalignment. Code and data will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper aims to address missing relevance annotations (“holes”) in IR benchmarks that distort evaluation and hinder RAG alignment. It proposes DREAM, a multi-agent debate framework where two LLMs take opposing stances (Relevant vs. Irrelevant) and iteratively critique each other until reaching consensus or escalating to humans. Using this method, the authors rebuild benchmarks into BRIDGE, filling large annotation gaps and revealing that many retrieval improvements were previously underestimated. The results show DREAM effectively automates reliable labeling and that completing these “holes” leads to more accurate and faithful IR and RAG evaluations.

### Strengths
1. The paper highlights an often-overlooked issue — that retrieval performance itself has been misjudged because of annotation holes in IR benchmarks. This brings a fresh perspective to understanding the mismatch between retrieval and generation performance.

2. I really like how the multi-agent debate setup helps counter problems like single-model overconfidence or failed confidence-based escalation, where easy cases get escalated and truly hard ones slip through.

### Weaknesses
One thing I wonder is whether the framework considers multi-hop reasoning cases — where the answer can only be inferred by combining multiple chunks. For example, if one passage says “Jack’s mother is Jane” and another says “Jane’s husband is Bob,” neither chunk alone directly answers “Who is Jack’s father?”, but together they clearly provide the evidence. It’d be interesting to see how DREAM handles or could be extended to capture this kind of compositional relevance.

### Questions
1. By default, the debate runs with two rounds of alternating arguments (R=2). I’m curious whether the authors explored what happens with more rounds — for instance, does extending the debate yield deeper reasoning or just redundancy? Some analysis on how performance scales with R could offer useful insights into the debate dynamics.

2. Since both agents (m₁ and m₂) are initialized from the same Llama3.3-70B-Instruct model, I wonder if there’s any risk of “shared bias” or reinforcement from their common pretraining knowledge, even though they take opposing stances. Would using models from different sources (e.g., different architectures or training data) introduce useful diversity, or might it hurt the framework’s stability and its strong human-level accuracy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The focus of the paper is on IR evaluation, where the authors propose an LLM multi-round debate solution. The solution is motivated by the fact that current IR benchmarks have "holes", i.e. data that is not labelled as relevant but might be relevant. Past work has used LLMs for relevance assessment but used single judge, which suffers from overconfidence. The authors point out that this hurts the performance of hybrid systems that send uncertain cases to humans: to do this well, models must be calibrated/ambiguity-aware. 

The paper introduces DREAM to use adversarial debate to decide when to escalate to human judgment. The debate is based on conflicting stances, only escalating when agents agree. They introduce a benchmark based on existing datasets, using DREAM to reduce evaluation holes.

The paper finds that DREAM reduces the escalation ratio while improving accuracy on the evaluation set. Moreover, DREAM can be used to refine real-world IR datasets and fill holes, lowering the human cost by only human-annotating uncertain cases. The refined real-world IR data is used to evaluate systems for RAG, with improvements to a RAG model.

### Strengths
- Strong intrinsic results: the paper shows that DREAM greatly reduces the amount of human annotation needed for the same accuracy
- Downstream utility: the paper demonstrates the utility of DREAM on augmenting a real IR dataset, where human annotation cost would be high with baseline approaches. 
- Human evaluation: the evaluation set is vetted by human expert annotators
- Baselines: the method compares to LLM-only and confidence-based baselines that support its claims.

### Weaknesses
- No discussion of latency cost: compared to LLM-as-judge baselines, this debate approach is much more expensive, requiring multiple model calls across multiple rounds. While the method saves human cost, the trade-off here should be discussed. 
- Novelty/major missing related work section: the paper is missing most related work on multi-agent debate, including https://arxiv.org/abs/2309.13007, https://arxiv.org/abs/2305.14325, https://arxiv.org/abs/2504.13079 which have covered stances, divergent thinking, and evidence conflict. The application of debate to IR evaluation seems novel but the debate methods themselves less so. 
- Dataset size: the size of the evaluation dataset seems quite small compared to the original datasets (BEIR and RobustQA) which have tens of thousands of examples. 
- This paper would benefit from interaction with the extensive literature on selective prediction, which has the same motivation (e.g. https://aclanthology.org/2022.findings-acl.158/, https://arxiv.org/abs/2303.16857, https://arxiv.org/abs/2402.15610) 
- Comparison to LARA would be easier to understand as a plot with different confidence thresholds/escalation ratios on the x axis

### Questions
- Order-dependence: LLMs are often prone to sycophancy and there's a documented trend of models picking whichever answer goes first. Does it matter which agent goes first? It would be good to have an ablation showing stance order doesn't matter. 
- How do you evaluate consensus? Are models instructed to format answers in an extractable way? 
- What was the reason for human disagreement on human eval? 
- Have you thought about distillation? Past work, e.g. https://arxiv.org/abs/2402.01620 distills multi-agent interactions into a single model. In principle, you could distill the better classifications from the DREAM debate pipeline into a single judge model that at test time produces a single decision. 

I'm willing to increase my score if these weaknesses/questions can be addressed.

Other minor points:
- L174: the opponents -> the opponent's
- L092-094: the common explanation...: this needs a citation. I believe another more common explanation for this is that the models are ignoring their contextual knowledge in favor of the parametric knowledge (https://aclanthology.org/2021.emnlp-main.565/, https://aclanthology.org/2022.emnlp-main.146/, https://aclanthology.org/2023.findings-emnlp.968/).

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DREAM, a multi‑agent debate framework for completing missing answer‑aware relevance annotations in IR benchmarks. Specifically, two LLM agents are initialized with opposing stances, debate for up to two rounds, and auto‑label a query–chunk pair upon consensus; persistent disagreements are escalated to human annotators together with the debate history to aid adjudication. DREAM targets accurate, low‑cost query–chunk relevance labeling and reports higher accuracy than single‑agent LLM‑as‑judge and confidence‑based escalation baselines.

Using DREAM, the authors construct BRIDGE, a refined benchmark built from subsets of BEIR and RobustQA that substantially reduces unlabeled relevant “holes.” BRIDGE adds 29,824 previously unlabeled relevant chunks—a 428% increase over the originally annotated 6,976 gold chunks—bringing the total to 36,800. The paper shows that such holes introduce systematic evaluation bias that can underestimate retriever effectiveness, and it introduces RAGAlign to measure retrieval–generation alignment in RAG systems. Results indicate stronger retrieval–generation alignment on BRIDGE than on the original benchmarks, and ablations on the expert‑annotated subset support the choice of two debate rounds and the utility of providing debate history for AI–human synergy.

### Strengths
1. The paper identifies and addresses a critical bottleneck in existing IR benchmarks: the incompleteness (i.e., holes) of current relevance-annotated datasets leading to unreliable retrieval performance evaluation of the RAG system.
2. The proposed multi-agent debate annotation pipeline is intuitive and clearly explained. 
3. The constructed BRIDGE benchmark uncovers 4 times previously unlabeled relevant chunks compared to the original gold chunks with relatively low cost, which should be meaningful for the IR community.

### Weaknesses
1. **Homogeneous, 2-agent setting for multi-agent debate**. In DREAM, both agents are Llama-3.3-70B-Instruct with temperature=0. Although this is a standard minimal setting for multi-agent debate, what if we use more than 2 agents or heterogeneous-model (i.e., different LLMs for each agent) to increase the diversity during debating? Will this bring more accurate annotation?
2. **Agreement treated as reliability without analyzing "wrong-but-agree"**. There is discussion of persistence of agreement across debate rounds in section A.5. However, there are no analysis to quantify the rate of spurious consensus (cases where agents agree yet the consensus contradicts expert labels).
3. **RAGAlign analysis tied to a single generator configuration**. The RAGAlign@K analysis in section 5.2 only focus on Llama3.3-8B-Instruct. It is not shown whether the observed retrieval–generation alignment gains on BRIDGE hold for other LLM generators. Also, there seems to have a generator version discrepancy between Figure 3 (Llama‑3.3‑8B) and Appendix I.4 (Llama‑3.1‑8B).
4. **Will the more accurate chuck annotation help generation performance?** While RAGAlign improves, the paper does not report whether denser, more accurate chunk annotations translate into better end‑task generation metrics under the same generator and prompts.

### Questions
Please see weaknesses.

Another following up question considering W.1 and W.2 together: 
1. If you increase debate diversity—by adding more agents, using heterogeneous base models, or allowing non‑zero temperatures—does the rate of spurious consensus decrease, and does end‑to‑end accuracy improve at comparable escalation cost?

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
The paper proposes DREAM, a debate-based framework for completing missing annotations in information retrieval (IR) benchmarks. It introduces a multi-agent relevance assessment process where two LLMs with opposing stances critique each other to reach agreement, using disagreement as a natural signal for human escalation. Building on DREAM, the authors create BRIDGE, a refined IR benchmark that fills missing relevance labels and enables fairer evaluation of retrieval and retrieval-augmented generation (RAG) systems.

### Strengths
- The paper is well-written and easy to follow
- The method itself is intuitive and fitting for the problem itself and seems to outperform baselines and work well empirically.

### Weaknesses
- Could also compare with this work and related works which use multi-agent debate to improve performance of RAG systems and contrast with these: https://arxiv.org/abs/2504.13079, https://arxiv.org/abs/2501.00332 
- Could clarify on how the quality of the benchmark is impacted by the choice of number of agents and which model families these models come from.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
