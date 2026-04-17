# WirelessMathLM: Teaching Mathematical Reasoning for LLMs in Wireless Communications with Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
Large language models excel at general mathematical reasoning but fail catastrophically on specialized technical mathematics. In wireless communications, where problems require precise manipulation of information-theoretic bounds, optimization constraints, and signal processing formulations, even state-of-the-art models struggle to achieve competent performance. 
We present \textbf{WirelessMathBench-XL}, the first training-scale benchmark for wireless mathematics, comprising 4,027 expert-validated problems from 970 state-of-the-art papers.
To validate dataset quality, we train a family of models called \textbf{WirelessMathLM} (0.5B, 3B, 7B) using Group Relative Policy Optimization with binary verification rewards. WirelessMathLM-7B achieves 39.5\% accuracy, approaching GPT-4o (40.4\%) while using approximately 100 times fewer parameters than DeepSeek-R1 (671B, 57.4\%), with dramatic improvements across all scales (0.5B: +11\%, 3B: +103\%, 7B: +81\%). 
Our controlled experiments reveal three findings that challenge prevailing assumptions about domain specialization. First, verification-based reinforcement learning outperforms supervised fine-tuning by +23\% relative (25.1\% vs 20.4\%) on identical data, showing that exploration against deterministic verifiers enables learning beyond labeled examples. Second, specialized training strengthens foundational capabilities: models gain +8.4 points across five general mathematical benchmarks and improve on knowledge, reasoning, and programming tasks without regression. Third, performance improvements distribute uniformly across 20 wireless subdomains regardless of training prevalence, demonstrating generalizable principle learning rather than pattern memorization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces WirelessMathBench‑XL, a benchmark of 4,027 problems from 970 papers covering wireless‑communications mathematics (optimization, information theory, signal processing, etc.). Problems come in three tiers—MCQ, progressive fill‑in‑the‑blank (25/50/75%), and full equation completion. On top of the dataset, the authors train compact models GRPO with binary, verifiable rewards, without supervised warm‑start. Surprisingly, GRPO on this domain‑specific dataset improves general math performance.

### Strengths
1. The field lacks specialized math datasets for wireless; the paper addresses this with a scalable, verifiable benchmark that aligns with realistic derivations and notations used in comms papers. The pipeline and rubric are detailed, with examples and QA procedures.
2. Leveraging verifiable correctness unlocks RL without costly human feedback. The hierarchical reward (format + answer verification; semantic equivalence for hard expressions) and simple, reproducible training recipe are well specified.
3. The 7B model approaches GPT‑4o on the new benchmark and GRPO delivers substantial lifts over base models across 0.5B/3B/7B.
4. The observation that domain‑specific GRPO improves MATH/Minerva/OlympiadBench/AMC/AIME (average +8.4) is interesting and contrary to standard “forgetting” concerns.

### Weaknesses
1. Reliance on LLMs for QA and evaluation introduces bias. Although experts do a second pass, the first‑stage filtering uses GPT‑4o, and semantic equivalence for complex expressions uses GPT‑4.1‑mini. This can bias both dataset selection and scores toward models similar to the evaluator.
2. Limited analysis on why transfer emerges. The paper notes positive transfer but offers limited causal evidence. Ablations that hold training tokens constant while varying domain specificity (how many domain-specific math questions are included in the training) would strengthen the claim.

### Questions
See my main concerns above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes WirelessMathLM, training compact LLMs (0.5B–7B) for wireless-communications math using verification-based reinforcement learning without massive scale or extensive supervision. They used GPT-4o and DeepSeek-R1 to construct WirelessMathBench-XL from the arXiv papers , a comprehensive benchmark of 4,027 problems from 970 papers. They trained models directly from the base checkpoints by Group Relative Policy Optimization (GRPO) with binary verification rewards, without supervised warm-start. The 7B model achieves 39.5% accuracy on WirelessMathBenchXL, approaching GPT-4o (40.4%) while using ≈100× fewer parameters than DeepSeek-R1 (671B, 57.4%).

### Strengths
1.	The WirelessMathLM 7B model achieves 39.5% accuracy on WirelessMathBenchXL, approaching GPT-4o (40.4%) while using ≈100× fewer parameters than DeepSeek-R1 (671B, 57.4%). And WirelessMathBench-XL, the training model and the GRPO training framework have been publicly released.
2.	The paper demonstrate that verification alone enables efficient domain specialization. GRPO training from base models, without supervised warm-start or human feedback. This challenges the assumption that reinforcement learning requires extensive pre-training.
3.	The paper show that specialized training develops transferable mathematical reasoning, suggesting that learning domain-specific mathematics strengthens fundamental capabilities.

### Weaknesses
1.	Despite the gains, 39.5% overall accuracy may still be too low for high-stakes engineering use; reporting task-wise reliability and confidence intervals would strengthen claims.
2.	The WirelessMathBench-XL construction uses GPT-4o filtering and DeepSeek-R1 extraction, which could bias style/notation and favor certain models

### Questions
1.	Could you report pre-vs-post domain-RL performance on broad, general-purpose benchmarks (e.g., knowledge, commonsense, reading comprehension, math outside wireless, programming) to show there is no regression in foundational abilities?
2.	How does reliance on GPT-4.1-mini for semantic equivalence checking affect grading validity and reproducibility, and can you quantify bias/error rates or provide an open verifier to reduce dependence on a proprietary model?
3. Topic coverage is skewed (e.g., deep learning and convex optimization dominate);  please report per-subdomain performance and discuss rebalancing to avoid overfitting to prevalent areas.

### Soundness
3

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
3

### Summary
1. Builds WirelessMathLM, adapting 0.5B–7B LMs to wireless-communications mathematics via GRPO using verifiable rewards only (no SFT / no human feedback). Reported gains are large: 7B goes from 21.9%→39.5% on WirelessMathBench-XL, “approaching GPT-4o (40.4%)” while using far fewer params than DeepSeek-R1 (671B) (57.4%)
2. Introduces WirelessMathBench-XL: 4,027 problems from 970 papers, with a three-tier format (MCQ, progressive fill-in at 25/50/75%, and full eqn completion), plus a dual-layer QA process (LLM screening + expert review; 78% pass rate at ≥3/5) 
3. Main results: on their test set, 7B GRPO = 39.5% overall; per-type gains esp. on fill-in (14.3%→37.0%); comprehensive baseline table includes proprietary and OSS models (DeepSeek-R1, Qwen-Math, Llama-3.3, etc.)

### Strengths
1. Simple, reproducible-ish training recipe: GRPO with binary verification is appealing, avoids SFT/human annotations, and the training details are reasonably specified (objective, α/β/G, hardware, epochs) 
2. Timely & valuable problem: formal, verifiable math in wireless comms is genuinely under-served; a curated benchmark at real scale is useful to both comms and reasoning communities. The pipeline description (Fig. 2) is clear and covers crawling, extraction, question generation and QA

### Weaknesses
1. Designing and collecting such a domain-specific dataset is a contribution, but I am not sure if posttraining it on 3B/7B models is a great contribution.
2. I know there are some issues with Qwen-3 base model where its math training datasets could be polluted, but have you tried to posttrain on top of Qwen-3?
3. Train a strong SFT on the same training set; compare to GRPO at equal token/computation budgets. This will isolate “verification-only RL” value.

### Questions
Several baselines are instruction-tuned (e.g., Qwen-Math), while authors fine-tune base models with RL. A natural SFT baseline on WirelessMathBench-XL (and DPO/ORPO) is missing; this weakens the claim that “verification alone enables efficient specialization.”

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To advance LLMs' ability to address mathematical problems in wireless communication, this paper introduces WirelessMathBench-XL, a benchmark dataset containing 4,027 problems collected from 970 papers through paper collection, mathematical extraction, and quality assurance. This dataset supports three tasks: multi-choice question-answering, fill-in-the-blank, and full equation completion. This paper further trains WirelessMathLM on these tasks using the GRPO scheme to enhance wireless mathematical reasoning performance. Evaluations on WirelessMathBench-XL demonstrate the superior performance of WirelessMathLM (with small LLM base models) compared to models trained without the GRPO scheme.

### Strengths
There are two main strengths in this work: 1. A new benchmark.  The paper assembles a broad-scope corpus and constructs a comprehensive, high-quality benchmark for wireless mathematics. The breadth and curation appear solid and relevant to the community.
Positive experiment results.  2. This paper demonstrates that GRPO fine-tuning can yield significant improvements over the base model, indicating the method’s effectiveness for this domain. It also shows improvements on general math benchmarks, suggesting that domain-specific training can strengthen fundamental mathematical abilities beyond the target domain.

### Weaknesses
There are a few weaknesses in this work. 1. Novelty/positioning. The overall pipeline and training method appear to reuse existing approaches, primarily transferring them to the wireless domain. As written, the contribution reads more like solid engineering than methodological innovation. Please clarify the novel elements (e.g., benchmark design, verification tooling, reward shaping) and position them against prior work.  2.  According to Section 2, the pipeline uses GPT‑4o and DeepSeek‑R1 to generate and use an LLM for semantic checking. This introduces potential selection and evaluation bias, weakening the “verifiable ground truth” claim. 3. This work showcases the effectiveness of GRPO; other fintuing methods (e.g., SFT/LoRA on the same corpus) should also be investigated, and detailed ablation studies should also be presented. 4. There are several writing issues and typos.

### Questions
1. During the paper filtering process, how is the relevance score assigned to each paper? Is this score automatically generated by the LLM or explicitly calculated based on predefined metrics?
2. Given that the human effort is still required to review each question, what advantages does the LLM-based automated evaluation provide?  Moreover, it is encouraged to develop a fully end-to-end, automated evaluation framework without human intervention. 
3. Compared to general mathematical reasoning problems, what are the main challenges in training LLMs on wireless mathematical problems?

### Soundness
2

### Presentation
2

### Contribution
2
