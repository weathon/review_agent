# AICrypto: A Comprehensive Benchmark for Evaluating Cryptography Capabilities of Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities across a variety of domains. However, their applications in cryptography, which serves as a foundational pillar of cybersecurity, remain largely unexplored. To address this gap, we propose \textbf{AICrypto}, the first comprehensive benchmark designed to evaluate the cryptography capabilities of LLMs. The benchmark comprises 135 multiple-choice questions, 150 capture-the-flag (CTF) challenges, and 30 proof problems, covering a broad range of skills from factual memorization to vulnerability exploitation and formal reasoning. All tasks are carefully reviewed or constructed by cryptography experts to ensure correctness and rigor. To support automated evaluation of CTF challenges, we design an agent-based framework. We introduce strong human expert performance baselines for comparison across all task types. Our evaluation of 17 leading LLMs reveals that state-of-the-art models match or even surpass human experts in memorizing cryptographic concepts, exploiting common vulnerabilities, and routine proofs. However, our case studies reveal that they still lack a deep understanding of abstract mathematical concepts and struggle with tasks that require multi-step reasoning and dynamic analysis. We hope this work could provide insights for future research on LLMs in cryptographic applications. Our code and dataset are available at https://anonymous.4open.science/r/aicrypto-iclr-BDE4/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AICrypto, a cryptography‑focused benchmark intended to comprehensively evaluate LLM capabilities across (i) 135 MCQs, (ii) 150 CTF challenges spanning nine crypto categories, and (iii) 18 proof problems drawn from three university exams. It also proposes an agentic evaluation harness for the CTFs and reports results for 17 leading LLMs. The key finding is that top models match or beat humans on MCQs, approach human performance on routine proofs, but lag substantially on CTFs that demand dynamic reasoning and program analysis.

### Strengths
- MCQs probe factual knowledge; proofs test formal reasoning; and CTFs stress real‑world exploitation skills.
- Mitigating Data Contamination. The paper documents rewriting/verification for MCQs and expert authorship for proofs, with manual quality checks, to avoid data contamination of the benchmark.

### Weaknesses
1. Helper scripts change the original CTF question. For many CTFs, the benchmark injects helper.py/.sage, which meaningfully lowers parsing/IO friction relative to human play and could inflate LLM success on certain categories. An ablation “with vs. without helper” would quantify this effect.
2. Proof scoring is manual and single‑pass. While the authors explain why LLM‑as‑grader is unreliable, the paper does not discuss inter‑rater agreement or rubric calibration across graders. Given that subtle logic gaps are a core finding, reporting inter-annotator metrics would bolster credibility. Moreover, manual grading makes running on the Proof subsection not scalable.
3. Limited # of Questions. Each subsection contains only a limited number of questions per sub-category. For example, DLP (under CTF) just contains 10 questions, while there are just 18 Proof Questions (all sub-categories combined), making the results not significant. Reporting the significance of the results or adding more questions would greatly improve the reliability of the benchmark.
4. Saturation on MCQ subset. All LLMs perform exceptionally well on the MCQ subset, with 11 LLMs either outperforming or matching human performance.

### Questions
Minor clarity/typo issues:
- “provider insights” in the conclusion, line 485
- “consistently general models” line 318
- Fig 4 instead of Fig 5 (line 196)
- Fig. 11 caption text is incorrect

See weaknesses above

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
3

### Summary
The paper introduces AICrypto, a comprehensive benchmark for evaluating LLMs' cryptographic capabilities across three task types: 135 multiple-choice questions testing factual knowledge, 150 CTF challenges requiring practical exploitation skills, and 18 proof problems assessing formal reasoning. The authors evaluate 17 state-of-the-art LLMs using an agent-based framework for CTF challenges and expert grading for proofs. Results show that leading models match or exceed human experts on conceptual knowledge and routine proofs but significantly underperform on practical CTF challenges requiring multi-step reasoning and dynamic analysis.

### Strengths
1. Comprehensive benchmark design: Three complementary task types provide holistic evaluation of cryptographic competence

2. Rigorous curation process: Expert involvement ensures task quality and prevents data contamination

3. Strong empirical evaluation: 17 models evaluated with human expert baselines for comparison

4. Practical insights: Clear identification of model limitations in numerical computation and multi-step reasoning

5. Reproducibility: Code and dataset made publicly available

### Weaknesses
1. Agent framework limitations: The simple agent framework may handicap model performance on CTFs. More sophisticated planning or tool-use strategies aren't explored.

2. Scalability issues: Manual proof grading is unsustainable. The attempted automated grading failure needs addressing for practical benchmark use.

3. Missing analysis: No investigation of whether specific model architectures or training approaches correlate with better cryptographic reasoning.

### Questions
1. How sensitive are CTF results to the agent framework choice? Have you tested more sophisticated approaches like ReAct or tree-search based planning?

2. Could formal verification tools (Coq, Lean) be integrated to enable automated proof checking rather than manual grading?

3. The paper mentions some models timing out on polynomial GCD computations (e.g., Figure 18). Could you quantify computational complexity thresholds where models consistently fail?

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
This paper presents AICrypto, a benchmark focusing on evaluating large language models (LLMs) on cryptographic tasks. It consists of three types of problems, namely, multiple-choice questions, capture-the-flag, and proof problems. An evaluation on 18 LLMs suggests that these LLMs perform well on multiple-choice questions but still struggle on the other two challenging tasks.

### Strengths
- The paper is well-written and easy to follow.

- The presented benchmark covers a range of different cryptographic tasks.

- The evaluation covers a decent amount of recent LLMs, including Gemini 2.5 Pro and Claude Sonnet 4.

### Weaknesses
- The dataset size is very small: The dataset contains 35 MCQs, 150 CTFs, and 18 proofs, and then in total fewer than 200 questions. It is unclear whether any conclusions drawn from 200 examples are statistically significant, especially given that there are lots of factors in LLM evaluation (model, prompt, etc) that can easily affect the performance.
 
- Usefulness is limited due to human-in-the-loop evaluation: MCQs are fully automated but most models have already achieved superhuman performance. The proof problems are interesting and seem to be useful for measuring progress in AI cryptography. However, the evaluation currently requires human supervision, which is extremely expensive.

- Data coverage seems to be poor: Almost all questions are collected from university exams or public cryptography challenges. While valuable, these questions often do not cover many real-world scenarios and are biased towards academic topics.

### Questions
See the weakness points above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a benchmark for evaluating LLMs on cryptography-related tasks, spanning multiple-choice questions on cryptographic concepts, CTF challenges, and proof generation. For CTF tasks, the authors report pass@k and employ an agentic workflow. For proof generation, human experts assess model outputs. The results highlight concrete limitations of current LLMs, including weak mathematical understanding.

### Strengths
- The data construction pipeline is solid.
- The experimental evaluation is comprehensive and rigorous.
- The agentic workflow proves effective for CTF tasks.

### Weaknesses
1. Data contamination safeguards are time-sensitive. To mitigate contamination, the authors include only challenges and exams from 2023 onward (line 156). However, given we are nearing 2026, this approach is unlikely to ensure contamination-free evaluation. These anti-contamination measures are highly time-sensitive and fragile.
2. Too few test samples per task type. Several categories have very limited instances (e.g., Proof Problems/Signatures contains only one item), which undermines the reliability and statistical significance of the conclusions.
3. A subset of failure modes stems from incorrect mathematical computation. Consider enabling a callable calculator tool to isolate reasoning errors from arithmetic mistakes.
4. What are the differences compared with CryptoBench?

### Questions
Line 196: It should reference Figure 5, not Figure 4.

### Soundness
1

### Presentation
2

### Contribution
2
