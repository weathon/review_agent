# PIKA: Expert-Level Synthetic Datasets for Post-Training Alignment from Scratch

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone for aligning large language models (LLMs). However, its effectiveness critically depends on high-quality instruction data. Most existing high-quality alignment datasets are either private or require costly human annotation, which hinders reproducibility and scalability. Even with the emergence of Reinforcement Learning from AI Feedback (RLAIF), concerns about data quality remain. Moreover, it is still unclear to the open-source community how much data is actually required to fine-tune a base model into a strong instruction-following model. 
Current state-of-the-art approaches typically rely on over 300k examples even at the supervised fine-tuning (SFT) stage, yet they still underperform compared to proprietary models, leaving substantial barriers for academic and resource-constrained communities.
To address this gap, we introduce PiKa, a data-efficient family of expert-level alignment datasets. 
In particular, the PiKa-SFT dataset uses only 30k SFT examples—an order of magnitude fewer than the SoTA dataset Magpie.
Through extensive evaluations by fine-tuning Llama-3-8B-Base on PiKa and other public instruction following datasets,
we show that PiKa-SFT alone outperforms models trained on much larger datasets. Remarkably, on two widely used alignment benchmarks, AlpacaEval 2.0 and Arena-Hard, PiKa-SFT fine-tuning surpasses the official Llama-3-8B-Instruct model -- which was trained on over 10M proprietary examples. We further extend our study by training the Qwen2.5 series (0.5B–7B) on PiKa-SFT, consistently outperforming their official instruction-tuned counterparts.  In addition, we curate 30k high-quality preference optimization examples, which further improve alignment performance when applied after SFT initialization. These findings demonstrate that high-quality alignment can be achieved with significantly reduced data, providing a practical and scalable path for advancing open-source LLM alignment research. Our code and data will be available at https://anonymous.4open.science/r/PiKa.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a highly impactful finding that directly challenges the "scale-is-all-you-need" paradigm for LLM alignment. The core contribution is PIKA, a remarkably small (only 30k SFT samples) yet exceptionally high-quality synthetic dataset.

### Strengths
PIKA Dataset Construction: The paper proposes a clean, three-step pipeline (Figure 1) to generate expert-level data16:
(1) Expert-Level Prompt Generation: It uses a persona-based approach, sampling complex, expert personas (e.g., from biology, law) from PersonaHub and prompting GPT-4O to generate knowledge-intensive instructions.
(2) Multi-Path Response Generation: For each instruction, the LLM generates $k$ candidate responses18.
(3) Reward-Model-Guided Selection: A strong reward model (RM) is used to score all responses. The PiKa-SFT dataset retains only the (instruction, highest-scoring response) pair.

### Weaknesses
PIKA's "expert-level" quality is not created from scratch; it is distilled. The framework relies on GPT-4O to generate high-quality instructions and a state-of-the-art Reward Model to filter and select the best responses . This means PIKA's success is fundamentally dependent on the capabilities of these more powerful models, which were themselves trained on massive-scale data.

This process introduces strong, unanalyzed stylistic biases. For example, the dataset's average response length is massive (5,365 tokens), suggesting the pipeline strongly rewards verbosity. The paper does not analyze whether this "longer is better" bias is truly optimal or just an artifact of the specific reward model chosen.

### Questions
plz check the weakness

### Soundness
2

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
3

### Summary
The authors introduce PiKa, a synthetic data suite for post-training alignment. It explores data generation means by persona-driven prompts and filtered by a reward model. The contributions come in two parts: (1) PiKa-SFT with ~30K instruction-response pairs, and (2) a matching preference dataset ~30K chosen vs rejected responses for DPO tuning. The result shows the models tuned on PiKa beat those trained on much larger public datasets on AlpacaEval 2.0 and Arena-Hard.

### Strengths
1. With just 30K SFT pairs, PiKa beats much larger public datasets.

2. The generation scheme is simple, reproducible.

### Weaknesses
1. DPO pairs are constructed by taking top vs bottom responses under a single reward model. Models may learn the reward model rather than robust human preferences.

2. Headline claims hinge on AlpacaEval 2.0 and Arena-Hard; broader real-world tasks like safety, long-context aren’t reported hear.

### Questions
Q1: The baselines include multi-turn conversational datasets (e.g., ShareGPT, WildChat), if I am understanding correctly, Pika seems to be more single-turn and expert-focused. Could the performance gains be partly attributed to a closer domain match between Pika's data style and the evaluation benchmarks (AlpacaEval, Arena-Hard), rather than inherent superiority? How would a Pika-trained model perform on a multi-turn conversational benchmark? It would be better if authors could discuss more about data type differences in both train and test sets.

Q2: The paper uses GPT-4o for instruction generation and as the "LLM-as-judge" for benchmark evaluation. Could this create a potential bias where the model is being evaluated by the same system that helped create its training data? 

Q3: Pika's responses are an order of magnitude longer than those in other datasets. Given the known bias of LLM judges towards longer responses, to what degree can the superior win rates be attributed to response length rather than fundamental improvements in instruction-following quality?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
PiKa is a compact, expert-level synthetic dataset for aligning LLMs that dramatically improves data efficiency in instruction tuning. Built using persona-driven instruction generation and reward-model filtering with GPT-4o, PiKa produces 30k high-difficulty, domain-rich instruction-response pairs and 30k preference examples. Models fine-tuned on PiKa outperform those trained on datasets over ten times larger, even surpassing proprietary instruction-tuned models like Llama-3-8B-Instruct on key alignment benchmarks. This shows that careful synthesis of expert-level data can replace massive corpora.

### Strengths
- The proposed pipeline is effective at improving two popular models in terms of alignment while minimally impact other capabilities.
- The methodology as well as data curation process are well presented and easy to follow.
- The clarity and structure of the paper are commendable.

### Weaknesses
```The method's performance is struggling```
Constructing synthetic data for alignment purposes is not new. The paper introduces another new way of constructing data. However, the evidence from the performance improvements are not strong enough to justify a publication. Magpie came out over a year ago and since then there has been plenty of work that's able to improve AlpacaEval and ArenaHard to new heights. For example, Wang et al. [1] improves llama3-8b to have 57.23 for AlpacaEval 2 and 48.3 for Arena-Hard with <70k data samples. This is much higher than Pika with 32.82 on AlpacaEval2 and 33.5 on Arenahard. The novelty of this data construction pipeline is hard to justify if it struggles to beat works in 2025. I seriously doubt the scalability of such methods, unless authors can provide other evidences.

[1] Improving Model Alignment Through Collective Intelligence of Open-Source LLMS

```Why does difficulty help is not clear```
The link between dataset “difficulty” and alignment improvement could be further substantiated with ablation or causal evidence rather than correlation.

```Does persona actually help?```
While persona-based generation is compelling, the paper could provide more analysis of persona diversity and domain distribution, as these may critically influence performance. It is also unclear how helpful are those persona in terms of quality. It seems like an ablation without persona may be needed.

### Questions
- Could the authors provide more insight into the cost-efficiency tradeoff—for example, how expensive it was to generate, score, and filter 30k examples compared to traditional human-annotated datasets?
- Would the PiKa approach scale effectively to larger datasets, or do the authors expect diminishing returns due to redundancy or computational overhead?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces the PiKa dataset for SFT training, which is constructed through an automated pipeline. Various statistics of the dataset are analyzed, and experiments are conducted on multiple base models to demonstrate the effectiveness of the dataset.

### Strengths
- The paper is well written and easy to follow.

- Detailed statistics of various datasets are analyzed, and fine-tuning on the proposed dataset achieves better performance than previous ones.

### Weaknesses
My main concern lies in the contribution. The proposed construction pipeline is widely used in current synthetic data generation. Could the authors better highlight the unique contributions or distinctive characteristics of the PiKa dataset?

### Questions
Please refer the the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
