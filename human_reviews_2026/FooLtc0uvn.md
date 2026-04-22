# KALE: Enhancing Knowledge Manipulation in Large Language Models via Knowledge-aware Learning

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Despite the impressive performance of large language models (LLMs) pretrained on vast knowledge corpora, advancing  their knowledge manipulation performance—the ability to effectively **recall, reason, and transfer relevant knowledge**—still remains challenging. 
Existing methods mainly leverage supervised fine-tuning (SFT) to enable LLMs to recall task-relevant knowledge by continuing the training process on labeled datasets. However, we observe that LLMs fine-tuned via SFT still occasionally exhibit the *known\&incorrect* phenomenon, where LLMs explicitly possess the relevant knowledge of a given question but cannot effectively manipulate it to answer correctly. To address this challenge, we propose KALE—a novel post-training framework that leverages knowledge graphs (KGs) to generate high-quality relevant rationales and enhance the knowledge manipulation ability via **K**nowledge-**A**ware **LE**arning. Specifically, KALE **first** proposes a **K**nowledge-**I**nduced (KI) data synthesis method to generate high-quality data rationales, i.e., a textual reasoning process from each question to correct answer through external KGs. **Then** KALE proposes a **K**nowledge-**A**ware (KA) fine-tuning paradigm to enhance the knowledge manipulation ability of LLMs. Extensive experiments on **eight** popular benchmarks across **six** different LLM backbones demonstrate the effectiveness of KALE, leading to an accuracy improvement of up to 11.72\% and an average of 4.18\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the “known & incorrect” failure in LLMs—models hold the right facts but fail to use them—and proposes KALE, a post-training framework that strengthens knowledge manipulation (recall, reasoning, transfer).  KALE has two parts: Knowledge-Induced data synthesis that extracts multi-hop reasoning paths from external knowledge graphs and uses them to generate high-quality textual rationales, and Knowledge-Aware fine-tuning that aligns the model’s token distributions *with* and *without* these rationales by minimizing the KL divergence, encouraging the model to internalize rationale information so it can retrieve and apply relevant knowledge even when no rationale is provided at test time. Across eight benchmarks and six backbones, KALE consistently outperforms strong baselines, underscoring more reliable knowledge use.

### Strengths
1. This paper formalizes the “known & incorrect” gap and empirically shows this failure mode remains common after SFT, which shows a clear problem framing and strong motivation.
2. The paper uses external KGs to extract multi-hop reasoning paths → generate textual rationales (KI), then minimize KL divergence between outputs with/without rationales for knowledge-aware fine-tuning (KA), so the model can retrieve relevant knowledge even when no rationale is provided at inference. The pipeline is coherent and goal-aligned.
3. The main experiment results and ablation studies consolidate the effectiveness and scalability of the proposed frameworks.

### Weaknesses
1. All experiments fine-tune on each benchmark’s training set separately. This setup resembles “task-specific adaptation” rather than evaluating cross-task generalization.
2. While the authors emphasize no extra inference-time cost, training includes:
(1) path extraction from large KGs (still requires full preprocessing, though faster than BFS),
(2) GPT-4o calls for rationale generation (API cost and reproducibility issues), and
(3) KL-based consistency training (dual forward passes).
Combined, these are likely much heavier than inference-time methods.

### Questions
1. How does performance degrade as KG coverage/quality drops (e.g., ablate edges, introduce noise)? Any robustness to wrong or conflicting triples, and do you weight paths by confidence?
2. When no full path connects question to answer, what fraction of training pairs fall back to partial paths, and how does that affect accuracy?
3. If you inject KI-style rationales at inference (without KA training), how much do base and SFT models improve, and can base + rationales ever surpass KALE?

### Soundness
4

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
Large Language Models (LLMs) often struggle with "knowledge manipulation," failing to answer questions correctly even when they possess the necessary information, a phenomenon known as "known & incorrect." This paper proposes KALE, a post-training framework that uses Knowledge Graphs (KGs) to generate data rationales, creating structured reasoning paths for Q&A pairs.

### Strengths
1. The paper's problem definition is clear and significant. The "known&incorrect" phenomenon is a key pain point in LLM research. The authors clearly articulate this problem with illustrative cases, providing a strong motivation.

2. The paper proposes a novel data generation framework (KI) that combines KG paths with LLM generation. This offers a systematic method for creating high-quality reasoning data with a clear logical basis.

3. The authors conducted extensive experiments on 8 benchmarks and 6 different LLM backbones.

### Weaknesses
1. The attribution of efficacy for the KI synthesis stage, a core contribution in this paper, is severely confounded. The process is critically dependent on a powerful, SOTA proprietary model (GPT-4o) to "translate" KG paths into "high-quality" rationales. This makes it difficult to discern if the performance gains stem from the KALE framework's superiority or simply from distilling a stronger "teacher" model. The authors' own results in Appendix Q (Table 18) amplify this concern: using a weaker rationale generator (Llama3 70B), KALE's performance on key benchmarks (AbsR, Common, MMLU, BBH) falls below the $KALE_{w/o~KI}$ ablation baseline (Table 2). This strongly suggests KALE's success relies heavily on the external teacher's capability, not its framework's generalizability.

2. The novelty of the second core innovation—Knowledge-Aware fine-tuning—is limited. The method's use of KL divergence to align model (no rationale) and teacher (with rationale) distributions is a mature technique in knowledge and self-distillation. For instance, recent work[1] has employed nearly identical KL-divergence SFT for similar motivations. The paper lacks a sufficient comparison and differentiation from this prior art.

3. The experimental design completely omits a mainstream and powerful alternative: Outcome-Based Reinforcement Learning. This RL approach, which hypothesizes that rewarding final outcomes is sufficient for implicit reasoning, circumvents KALE's central challenge: the "lack of high-quality textual reasoning data." A discussion and empirical comparison with RL methods is strongly suggested.

[1] Efficient Knowledge Injection in LLMs via Self-Distillation.

### Questions
See above

### Soundness
2

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
3

### Summary
This paper proposes KALE, a two-stage framework to improve knowledge manipulation in large language models.
- First, a Knowledge-Induced (KI) data generation step uses external knowledge graphs (e.g., Wikidata) to extract multi-hop reasoning paths and generate rationales via GPT-4o.
- Second, a Knowledge-Aware (KA) learning paradigm minimizes the KL divergence between distributions of models trained with and without rationales, encouraging internalization of explicit reasoning.
- Experiments on multiple knowledge-intensive benchmarks (MMLU, RACE, ARC, BBH, AbsR) show consistent accuracy improvements.

### Strengths
1. The method elegantly integrates external structured knowledge with rationale-based learning, addressing the “known-but-incorrect” problem in LLMs.

2. The proposed model consistently improves across model scales (7B–32B), showing stable generality.

### Weaknesses
1. No evaluation on open-ended generation or general abilities. The paper focuses solely on accuracy in knowledge tasks and does not verify whether KALE harms general fluency or creativity after fine-tuning.

2. No comparison with modern reasoning or “thinking-style” models.
Baselines (ToG, StructGPT, GraphRAG) are early models and do not include current SOTA models like DeepSeek-R1, Qwen2.5-Think, or Llama3 thinking model. Hence, the claimed “SOTA” results may be overstated.

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes knowledge-aware learning (KALE), which consists of two key components: knowledge-induced data synthesis (KI) to generate high-quality relational data, and knowledge-aware fine-tuning (KA) to enable large language models (LLMs) to better manipulate task-relevant knowledge. For KI, the method finds multiple reasoning paths that connect a question entity to an answer entity using the A* algorithm, where a heuristic function is derived from anchor entities. These rationales are expected to provide high-quality textual reasoning data that bridges the question and answer. For KA, the approach minimizes the divergence between the generative distributions with and without the KI-based rationales. Experimental results on various benchmarks demonstrate that KALE outperforms several baseline models, validating its effectiveness.

### Strengths
1.	The proposed knowledge-aware learning (KALE) framework, which integrates knowledge-induced data synthesis (KI) and knowledge-aware fine-tuning (KA), is novel and interesting. The rationale extraction process is well designed to enhance efficiency through the use of anchor entities and a three-step BFS strategy.
2.	The experimental results demonstrate that the proposed KALE framework achieves notable performance improvements, and the ablation studies clearly illustrate the individual effects of KI and KA.
3.	The presentation is clear, well-structured, and generally easy to follow, with good overall organization.

### Weaknesses
1.	The extracted rationales are regarded as a form of Chain-of-Thought (CoT), but it remains unclear why KA is formulated based on the KL divergence between the generative distributions with and without rationales. What is the motivation for using KL divergence, compared to more conventional fine-tuning approaches that jointly generate both rationales and answers under an autoregressive loss?
2.	From a data augmentation perspective, it is unclear why the generated dataset is relatively large compared to those of other baseline methods. How does the size of the automatically augmented dataset compare quantitatively to other approaches?
3.	The proposed rationale extraction relies on named entity recognition (NER) and a graph-based search using the A* algorithm. However, GPT-generated rationales could also be used for training on question–answer pairs. Why does the model exhibit improved performance on test questions for which such rationales are not available? It also remains unclear how the model performs when rationales for question entities are absent in the training data.

### Questions
1.	It remains less convincing that the proposed rationale extraction process is truly necessary. Could simpler or alternative search-based methods address this problem as effectively?

### Soundness
2

### Presentation
3

### Contribution
3
