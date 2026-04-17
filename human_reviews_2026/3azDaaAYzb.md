# Beyond Safe Answers: A Benchmark for Evaluating True Risk Awareness in Large Reasoning Models

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Existing safety evaluations primarily assess response-level safety, leaving
reasoning-level risks unmeasured. Despite the remarkable proficiency of Large
Reasoning Models (LRMs) in handling complex reasoning tasks, their reliabil-
ity in safety-critical scenarios remains uncertain. We identify Superficial Safety
Alignment (SSA): a phenomenon where models produce superficially safe outputs
while internal reasoning processes fail to genuinely detect and mitigate underly-
ing risks, creating a dangerous illusion of safety and rendering systems prone to
catastrophic failure under minor perturbations. To systematically investigate SSA,
we introduce Beyond Safe Answers (BSA), a novel benchmark comprising 2,000
challenging instances organized into three distinct SSA scenarios and spanning
nine risk categories, each meticulously annotated with risk rationales. We evaluate
23 state-of-the-art LRMs demonstrate the difficulty of this benchmark, with the
best model reaching 54.57% accuracy on risk-rationale identification. Current
benchmarks are largely blind to this latent risk; to our knowledge, BSA is the
first benchmark designed to systematically diagnose SSA. We further explore the
efficacy of safety rules, specialized fine-tuning on safety reasoning data, and diverse
decoding strategies in mitigating SSA. Our work aims for verifiably robust safety
reasoning in LRMs, moving beyond mere superficial compliance and enabling
practitioners to evaluate and improve safety-reasoning fidelity with measurable
evidence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Superficial Safety Alignment (SSA), a phenomenon where Large Reasoning Models (LRMs) produce superficially safe answers even when their internal reasoning fails to genuinely identify or mitigate underlying risks. To address this "illusion of safety," the authors present the Beyond Safe Answers (BSA) benchmark, which contains 2,000 challenging instances designed to evaluate the fidelity of safety reasoning rather than just final outputs. An evaluation of 23 state-of-the-art LRMs on BSA reveals a significant gap between response-level safety and reasoning-level accuracy. The study also investigates mitigation strategies, finding that explicit safety rules and fine-tuning with high-quality reasoning data can help improve genuine risk awareness.

### Strengths
1. The paper identifies a novel and critical problem of superficial safety alignment (SSA). It moves beyond existing evaluations, which primarily check the safety of the final response, to instead probe the safety and correctness of the internal reasoning process. 
2. The authors developed the Beyond Safe Answers (BSA) benchmark as a concrete tool to diagnose this newly defined problem. The benchmark is comprehensive, comprising 2,000 instances across nine risk categories and three specific SSA failure scenarios.
3. The paper evaluates 23 state-of-the-art Large Reasoning Models (LRMs) using the new benchmark, demonstrating that SSA is a widespread issue. Furthermore, the paper investigates existing mitigation strategies, analyzing the impact of safety rules and fine-tuning, which provides actionable insights.

### Weaknesses
1. SSA is an important concept studied in the paper, but its definition is in the appendix. This makes the main content incomplete and more difficult for readers to understand. 
2. The difference between Risk Omission and Cognitive Shortcut is not clear. Does the difference lie in the presence of multi-risk queries?
3. It is not clear why combining Over Sensitivity (where the ground truth should be a direct answer) with the other two safety issues. The author does not justify that. There are some disadvantages of doing so, for instance, the Safe@k metrics may not reflect the safety alignment of the model because a high number may be caused by a high over-refusal rate. 
4. The judge for Over Sensitivity is a bit unclear and questionable (line 891). Since they are all essentially benign queries, why is there a risk_summary, and why should the model identify the Genuine risks?

### Questions
1. The two sentences between lines 256 and 261 are repetitive. 
2. It seems that Claude provides the summarized reasoning process in their API, which may affect the evaluation results [1]. This may also apply to other close LRMs. The authors should double-check and clearly mention this in the paper. 

[1] https://docs.claude.com/en/docs/build-with-claude/extended-thinking#summarized-thinking

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors develop the BSA benchmark which can be used to annotate safety in LRMs. The authors develop this benchmark using new and existing sources and create ground truth annotations using a hybrid approach of human annotators and LLMs-as-judges. They show that leading models do not do very well on this benchmark, and it remains an open challenge.

### Strengths
The authors do a good job of motivating safety with respect to these models and the example in Figure 1 was clear. 

The hybrid annotation approach was nicely validated and made the LLM-as-judge approach more likely to yield high quality data.

### Weaknesses
Cognitive Shortcut is a term that already has been used extensively in the literature. I recommend the authors choose a new term that will be less confusing and not pollute the literature. 

The alarmist language needs to be toned down for an academic publication: “alarming extent” “crucial insights” “vital tool”.

More detail is needed in related work rather than just vague mentions to work. E.g., “Anthropic (10) showed Claude 3 Opus varies behaviors under evaluation”

The paper is hard to read and introduces a lot of new jargon that seems related to deep concepts in psychology but doesn’t really explain these terms. The terms risk anthropomorphization of LLMs. 

Figure 2 has many unexplained acronyms. It would be better to show more examples of the kinds of failure modes that this dataset contains. Two of the categories OS and RO are only given about 1 sentence of description in the main text and these are the core conceptual ideas.

The fact that a simple additional prompt (derived from known best practices) can drastically affect performance on this benchmark undermines its broader usefulness and ability to detect safety problems.

### Questions
Why does it have to make everything explicit in its observable chain of thought to have considered it. Significant reasoning is happening in weight space, and it’s unclear what reasoning models are really doing with the COT. 

What are some examples of over-sensitivity or risk omission? These seem to be ignored in the early part of the paper, only to be highlighted later. 

Is this “reasoning incompetence” or is it just a multi-turn jailbreak packed into a single prompt? From the example, it seems to be the key to understanding what is happening here. 

Please compare this benchmark to related work that uses multiple turns to discuss highly related phenomena. There should be a section on these alternatives in the related work. Basically, the key novelty, it seems, is a single-turn benchmark that studies many of the same safety issues discovered in multi-turn settings. 

Safety Alignment Tax should be cited. It seems that all the results point to this benchmark getting solved as models get better (better prompts, better base models, better fine-tuning all improve performance). What capability does this benchmark pick out that goes above and beyond general capabilities? 

How much of the benchmark is novel vs. derived from existing data-sets?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Superficial Safety Alignment (SSA), a phenomenon where LRMs produce seemingly safe responses despite flawed or incomplete internal reasoning.
To diagnose SSA, the authors propose the Beyond Safe Answers (BSA) benchmark with new metrics (Safe@k, Think@k) and evaluate various models across multiple safety dimensions.
The study reveals a consistent gap between reasoning correctness and response safety, providing insights into the reliability limits of reasoning-based safety alignment.

### Strengths
- The proposed formulation of Superficial Safety Alignment (SSA) is interesting and highlights an underexplored aspect of LRM safety. This perspective has clear practical relevance, as it draws attention to reasoning-level safety failures that are easily overlooked in standard output-based evaluations.
- The evaluation protocol is well-structured and methodologically sound. It provides a clear operationalization of SSA through metrics such as Safe@k and Think@k, enabling systematic diagnosis of reasoning-level safety inconsistencies.
- The authors release a large and well-annotated dataset (BSA), which can serve as a useful resource for future work in reasoning-based safety evaluation.
- The experimental coverage is extensive, spanning diverse LLMs, which lends credibility to the reported findings and demonstrates the consistency and generality of the SSA phenomenon across model families and scales.

### Weaknesses
### Limited contribution and narrow scope

While the identified problem of Superficial Safety Alignment (SSA) is interesting and relevant, it only addresses a limited subset of safety risks in LRMs—specifically, cases where the reasoning is unsafe but the response appears safe. However, this represents only one aspect of the broader safety landscape in LRMs. Moreover, the idea of evaluating the quality of reasoning chains has already been extensively explored in recent literature. The proposed work can thus be seen as an incremental extension of reasoning-chain quality assessment from a safety perspective, without introducing substantial conceptual or methodological novelty. The construction of the dataset and evaluation protocol also does not appear to involve major technical challenges. Additionally, the proposed methods (e.g., integrating safety policies, fine-tuning on STAR-1) primarily test existing techniques rather than presenting a novel or effective solution.

---

### Insufficient coverage of related works

The related work section should include prior studies that evaluate reasoning-chain quality in LRMs, since the core of this paper lies precisely in measuring reasoning quality from a safety perspective (e.g., Think@k metric).

---

### Terminological confusion

The term Superficial Safety Alignment overlaps with terminology already used in earlier alignment research, which may lead to confusion. Since the concept here is specifically tailored to reasoning-capable models, a more distinctive and precise naming would improve clarity.

[1] Superficial Safety Alignment Hypothesis, arxiv 2024

---

### Limited novelty of risk taxonomy

While the risk taxonomy in the BSA benchmark is well structured, most categories—except Cognitive Shortcut—are not new. This diminishes the overall novelty of the proposed benchmark and the distinctiveness of the problem formulation.

---

### Lack of cross-benchmark validation

Experiments are conducted only on the BSA benchmark. The authors should have evaluated models on existing safety benchmarks (e.g., StrongReject, WildJailbreak) as well. If the same SSA phenomena are well observed on standard benchmarks, the necessity and distinct contribution of BSA would be questionable.

--- 

### Formatting issue

The reference citation format deviates from the conventional ICLR style and should be corrected for consistency.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates an important phenomenon: models produce superficially safe outputs while internal reasoning processes harmful content, and summary this as SSA (Superficial Safety Alignment (SSA)). The authors introduce a novel benchmark: Beyond Safe Answers (BSA) with over 2000 instances and report results on 23 reasoning models. The evaluation results show significant emergency to improve reasoning models' internal safety as their think process is also open to the public (except some close-source reasoning models like o3).

### Strengths
1. The investigated phenomenon is very important. While some reasoning models do not open-source their thinking process (like o1), others (like r1) expose the full thinking content to the user. This makes the Superficial Safety Alignment (SSA) much serious.

2. The authors systematically summarizes this phenomenon and provides a complete framework to evaluate it, providing a benchmark for future improvements to this issue.

3. The evaluation metric fully considers the cost of human resource evaluation and reports (1) Safe@1  and  Think@1 And (2) Safe@k  and  Think@k In addition, the fairness of LLM as a judge was also reported. This makes the evaluation results convincing.

4. The authors provide a simple method to deal with this issue that finetunes the reasoning models with high-quality data.

### Weaknesses
1. The presentation could be improved. Line 127-132 has some space that making this page appear somewhat empty.

2. How many GPU hours, total tokens, and dollar cost does one evaluation pipeline consume?

### Questions
Please see the weakness.

### Soundness
4

### Presentation
2

### Contribution
4
