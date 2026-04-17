# RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline

- Decision: Reject
- Scores: 4, 6, 8, 4

## Abstract
If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RECAP, an agentic framework designed to extract memorized training data from Large Language Models (LLMs), addressing concerns about the illegal distribution of copyrighted material. The core contribution lies in its Jailbreaker component, which circumvents model alignment, and a Feedback Agent that creates an iterative extraction loop. The method is evaluated on a new benchmark, EchoTrace, comprising copyrighted books and scientific papers, and is accompanied by a detailed ablation study. While the paper tackles a significant and timely problem, the contributions feel incremental. The motivation for a new benchmark is unclear, and the experimental results are not fully compelling due to the choice of closed-source models and a lack of comparison on established benchmarks.

### Strengths
Originality: The primary originality lies in the iterative, agent-based design, specifically the Feedback Agent that refines extraction attempts based on previous failures. The Jailbreaker component, while sharing similarities with existing divergence attacks, is leading to the performance improvement.

Quality: The paper includes a thorough ablation study that analyzes the effectiveness and efficiency of RECAP's components, which is a valuable contribution for practitioners and future research.

Clarity: The paper is generally well-written and the RECAP framework is described clearly.

Significance: The problem of copyright infringement via LLM memorization is a critical issue for the AI community. Providing tools to audit and measure this risk is of high significance.

### Weaknesses
Novelty of Core Component: The application of the Jailbreaker, a core novelty, appears similar to divergent attacks proposed in prior work (e.g., Nasr et al., 2023), which also force models to deviate from their alignment. The paper would be strengthened by a clearer distinction and discussion of how this component differs from or builds upon existing attack paradigms.

Benchmark Justification and Scope: The creation of the EchoTrace benchmark is not sufficiently motivated. The field has well-established benchmarks for training data extraction (e.g., the Model Extraction Benchmark, the-stack-smol used by Wang et al., 2024). The choice to create a new one requires justification, especially since its composition (only 35 books and 20 papers, with short 40-token target sequences) may limit the generalizability of the findings. An ablation on the length of the extractable data is notably absent.

Experimental Setup on Closed-Source Models: The evaluation relies heavily on closed-source models (e.g., GPT-4.1, Gemini-2.5-Pro, Claude-3.7) for which the exact training data composition is unknown. This introduces a fundamental validity issue, as it is impossible to confirm whether the EchoTrace sources were actually in the model's training set. The choice of these models over open-weight alternatives is not adequately justified and weakens the evidence for RECAP's efficacy.

Comparative Evaluation: The results are less compelling due to the lack of comparison on established benchmarks used by contemporary work. For instance, comparing against Dynamic Soft Prompting (Wang et al., 2024) on the same test sets (e.g., the-stack-smol) would provide a more direct and convincing performance assessment.

### Questions
1. What was the specific motivation for creating the new EchoTrace benchmark instead of using or extending well-established datasets like the Pile, the Stack, or its derivatives? Were there specific limitations in these existing benchmarks that EchoTrace aims to address?

2. Could the authors extend their experimental results to include the test sets used by Wang et al., 2024 for Dynamic Soft Prompting? This would allow for a more direct and fair comparison with a closely related state-of-the-art method.

3. The paper mainly focuses on closed-source models. What was the reasoning behind this choice? Could the authors also evaluate RECAP on open-source models with known training data (e.g., GPT-Neo, Pythia) to conclusively verify that the extracted text was indeed part of the training corpus? This would significantly strengthen the validity of the claims.

4. There is a wrong citation of RedPajama (194).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the critical challenge of verifying memorized training data in Large Language Models (LLMs) when training data inspection is unavailable. It proposes RECAP, an agentic pipeline featuring a feedback-driven iterative loop and a jailbreaking module, designed to elicit and verify verbatim memorized content from LLMs.Additionally, the paper introduces EchoTrace, a novel benchmark encompassing 35 full-length books and 20 arXiv research papers, totaling over 70,000 40-token passages for evaluation. Experimental results demonstrate RECAP's superiority: it achieves an average ROUGE-L score of 0.46 for copyrighted content across four model families, improves GPT-4.1’s copyrighted text extraction ROUGE-L from 0.38 to 0.47, and ensures no significant contamination from non-training data. The work also includes cost optimization and ethical considerations to avoid copyright misuse.

### Strengths
1.The feedback loop (via Feedback Agent) iteratively refines extractions without injecting excessive external information, reducing false positives.
2.The jailbreaking module effectively circumvents alignment-induced refusals, addressing a key limitation of prior methods like Prefix-Probing and Dynamic Soft Prompting (DSP).The hybrid memorization score filtering balances extraction quality and cost efficiency, addressing the practicality of iterative pipelines.
3.The experiments are comprehensive and well-designed:Cover multiple models and domains, ensuring generalizability. Include ablation studies and cost analysis, providing actionable insights for real-world use.

### Weaknesses
1.The benchmark overrepresents popular works for both public domain and copyrighted categories. This may overestimate RECAP's performance on less mainstream, rarely scraped texts—critical for assessing real-world applicability.
2.Non-training data is limited to 5 books released in 2025, with no diversity in genre or timeframes. This makes it hard to validate RECAP's robustness against false positives across varied non-training scenarios.
3.The jailbreaking module relies on a single static hand-crafted prompt, which the authors acknowledge may fail as LLMs’alignment updates advance. No comparison with dynamic jailbreaking methods is provided, leaving unclear whether static prompts are optimal or just a convenient choice.
4.The module’s effectiveness is only measured by refusal rates, not by whether jailbroken outputs introduce noise or semantic distortions, which could undermine extraction reliability.

### Questions
1.EchoTrace’s focus on popular works may overestimate RECAP's performance. Do you plan to expand the benchmark to include non-mainstream texts and validate RECAP's effectiveness on these?
2. The jailbreaking module uses a static prompt. Have you tested dynamic jailbreaking methods and compared their success rate, extraction quality, and robustness to future LLM alignment updates?
3.RECAP is not evaluated on open-source LLMs. Do you expect similar performance on these models, or would the pipeline require modifications ?
4.The paper omits comparisons with 2024–2025 state-of-the-art methods . Do you plan to replenish these comparisons, and if so, what preliminary insights can you share about RECAP's relative performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
1. The paper tackles the task of extracting memorized data from LLM pretraining corpora to provide verifiable evidence of what models have seen during training.
2. It addresses two key challenges: (i) modern alignment safeguards that cause models to refuse reproducing even public-domain content, and (ii) the limited recall of single-iteration prompting methods that fail to elicit complete memorized passages.
3. The proposed agentic RECAP pipeline, achieves the highest ROUGE-L scores across all tested models and datasets.

### Strengths
1. The EchoTrace dataset is a valuable resource for future work. It covers diverse text types (public-domain, copyrighted, and unseen books). The segmentation and event summaries make it easy to test new elicitation or membership-inference methods.
2. The proposed RECAP method directly addresses both identified challenges. The results are clear and statistically grounded, showing strong and consistent improvements across four major model families.

### Weaknesses
In Prefix-Prompting baselines, longer or more detailed prefixes can sometimes lead to stronger verbatim reproduction. It would strengthen the paper if the authors analyzed whether prompt length differences contribute to RECAP’s performance gains.

### Questions
Could RECAP framework regulate prompt length within its agentic loop?

### Soundness
4

### Presentation
4

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
The work proposes a method called RECAP to test or verify whether a large language model (LLM) has memorized specific text data during training—particularly potentially copyrighted material, such as full books. To enable this verification, we construct a new dataset comprising public domain works, copyrighted books, and non-training new books, categorized by their likelihood of being memorized, and further integrate research papers into the evaluation.

### Strengths
- The dataset used is comprehensive enough to effectively demonstrate that the proposed method can successfully guide the LLM to reveal memorized content.
    
- The experiments are thorough and examine the method from multiple angles, including interesting phenomena like how popular or “welcome” certain memorized content tends to be.

### Weaknesses
- I’m really unsure about the writing style of this work. The method described in the main text feels more like a high-level idea,  very rough and vague. Almost all the actual details are buried in the appendix. I’m not sure if this kind of writing is acceptable, but honestly, it made it super hard for me to understand the method clearly,  so hard that I couldn’t even judge whether the approach is reliable or not. I think the appendix should only support the main text, not carry most of the technical details.
    
    - Specifically, the five steps in the RECAP method require very careful reading to fully grasp. Figure 2 is just a functional overview, even with the main text, it’s still confusing because of some unclear keyword usage. Maybe adding a concrete example directly into the Figure 2 flowchart would help a lot. Also, the four diagrams in Figure 2 — the left side seems to show related data, but then suddenly on the right, BERT and ELMo appear. Are those meant to represent language models (I think maybe it’s Parrot BERT?)? The logic connecting them isn’t clearly explained in the figure.
        
- I’m also not convinced that using ROUGE-L is the right way to measure an LLM’s ability to reproduce text. ROUGE-L looks at the longest common subsequence between two texts. it allows skipping words. But when we’re talking about copyright, we usually care about whether the model can reproduce the text *exactly*, or at least reproduce a long enough chunk to count as infringement [1]. If that’s the case, maybe we should try using Word Error Rate, like in speech recognition tasks?
    
    - Also, how do we interpret the ROUGE-L scores? Intuitively, a higher score means stronger evidence that the LLM remembers the book content. But since there’s a FEEDBACK AGENT involved, how can we tell whether the result comes from the model’s actual memory — or just from being guided by the agent?
        
- And one last thing, why doesn’t Table 1 show the DSP + Jailbreak results like Table 2 does?
    

[1] Copyright violations and large language models

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
3
