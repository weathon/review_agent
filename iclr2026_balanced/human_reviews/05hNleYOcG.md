## Human Reviewer 1

### Summary
This paper introduces PLAGUE, a modular, memory-augmented multi-round jailbreak framework that coordinates a three-stage Planner–Primer–Finisher pipeline, achieving state-of-the-art attack-success rates on several mainstream LLMs.

### Strengths
1. Comprehensive experimental coverage. The authors conduct cross model, cross category (all 10 HarmBench classes) multi round attacks on five mainstream commercial and open source LLMs, including closed source heavyweights such as o3 and Opus 4.1, yielding highly credible results.
2. Rubric-based feedback outperforms binary signals. A fine-grained 10-point scale scored on four dimensions (Relevance, Practicality, Detail, Compliance) is integrated with backtracking and reflection, giving finer control over the attack trajectory.

### Weaknesses
1. The “lifelong learning” mechanism is oversold; it is only a static retrieval pool.
So-called lifelong learning merely appends a successful attack strategy to a vector base once; there is no online update, forgetting mechanism, policy evolution, or learning from negative samples. It is conceptually misused and far from genuine lifelong-learning techniques such as continual learning or catastrophic-forgetting mitigation.

2. The claimed modularity lacks universal validation.
Although advertised as plug-and-play, only the replacement of GOAT/Crescendo/ActorBreaker is tested. The authors never demonstrate how an arbitrary new module (e.g., a user-designed Planner) would be integrated, specify the interface contract, or show failure cases. Figure 1 also reveals tight coupling (e.g., Primer relies on Planner’s output format), raising doubts about extensibility.

3. Evaluation metrics are one-sided; the diversity–success trade-off is ignored.
Only ASR improvement is reported, yet Figure 4 shows that introducing the ActorBreaker Planner raises diversity by 15 % while ASR drops. Attack cost (e.g., manual screening overhead), cross-model transferability (success-rate drop), and human-perceived stealth (ease of user detection) are never analyzed.

4. Technical contribution is incremental; the work is more engineering tuning than principled innovation.
The Planner + Primer + Finisher pipeline is essentially an optimized assembly of Crescendo (gradual lure), ActorBreaker (plan generation), and GOAT (strategy pool). Key tweaks such as the 0.7 threshold and vector retrieval are heuristic and lack theoretical grounding or causal attribution (insufficient ablation).

5. Computational cost and latency are unreported, leaving practicality in question.
Although Table 5 counts LLM calls, the authors provide no end-to-end latency (embedding retrieval, summarization, parallel LLM invocations). In real-time red-team settings, six or more API calls plus repeated scoring may exceed the response window of production safety systems, making the threat model unrealistic.

### Questions
None.

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 2

### Summary
**NOTE: This paper violates the conference formatting guidelines by substantially reducing the page margins to fit more content. I would recommend a desk rejection due to this severe format violation. Nevertheless, I provide my technical evaluation below and defer the final desk-rejection decision to the AC and PC.**


PLAGUE is a plug-and-play, lifelong-learning framework for generating modular multi-turn jailbreaks against black-box LLMs: it builds an n-step plan by retrieving successful past strategies (Planner), escalates context with benign-seeming intermediate prompts (Primer), and then executes the final exploit (Finisher), while using rubriced reflection, backtracking, and a memory of successful strategies to adapt over time. Evaluated on the HarmBench benchmark, PLAGUE outperforms prior multi-turn and single-turn methods, achieving ASRs such as 81.4% on OpenAI o3, 67.3% on Claude Opus 4.1, and up to 97.8% on Deepseek-R1, while remaining computationally efficient within a six-turn budget; the authors note ethical risks but argue the framework aids systematic vulnerability evaluation and defense development.

### Strengths
- The modular design of PLAGUE is neat.

- PLAGUE introduces a unique embedding-based memory system, enabling it to learn from past interactions and adapt over time to new goals and contexts.

### Weaknesses
- The paper’s scope is limited by its exclusive focus on developing attackers without accompanying defensive methods. While PLAGUE advances the study of multi-turn jailbreaks, it offers no systematic exploration of countermeasures or co-evolving defenses. As a result, the work demonstrates how to break safety mechanisms effectively but provides little insight into how to strengthen or adapt them, narrowing its overall contribution to LLM safety research.

- This works misses crucial recent works that introduced performant advances in multi-turn jailbreaks, e.g., https://arxiv.org/abs/2504.13203, https://arxiv.org/abs/2410.10700, https://arxiv.org/abs/2502.19820 which are shown to be substantially better than Crescendo, the baselines included in this paper. In particular, this work shares strong similarities to https://arxiv.org/abs/2504.13203, which also includes planners, optimizers, and intermediate verifiers. Thus it's really important to discuss and compare to these methods.

### Questions
In addition to the weakness:

- How does PLAGUE compare to wider range of multi-turn red-teaming methods?

- To serve realistic red-teaming needs for broadly revealing LLM vulnerability, it's crucial that an automatic jailbreak or red-team method to be able to discover a wide range of successful attacks. Is PLAGUE capable of identifying multiple diverse attacks given the same seed harmful query? Could you quantify such ability?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
This paper introduces PLAGUE, a multi-stage framework for the automated generation of multi-turn jailbreak attacks against Large Language Models (LLMs). The framework decomposes the attack process into three distinct phases: a Planner, a Primer for context-building, and a Finisher for the final attack. The core design aims to enhance the success rate, diversity, and adaptability of multi-turn attacks through a plug-and-play modular architecture combined with a lifelong learning memory mechanism.

### Strengths
- Systematic Problem Decomposition: A commendable aspect of this work is its attempt to bring a structured and systematic description to the complex and often ad-hoc process of multi-turn attacks. Decomposing the attack into planning, preparation, and execution phases provides a clear workflow for analyzing and designing such attacks.

- Impressive Empirical Results: The method's effectiveness is well-demonstrated, particularly on models known for their strong safety alignment, such as Claude Opus and OpenAI's o3. The data indicates that the system is highly effective in practice.

### Weaknesses
- Limited Novelty: Upon closer inspection, the paper's core claimed innovations appear rather weak. First, the "Primer" stage, whose central idea is to "progressively guide the conversation context with a series of seemingly harmless questions," is practically the definition of any sophisticated multi-turn attack, not a novel contribution. The "lifelong learning" component is essentially a Retrieval-Augmented Generation (RAG) system using vector embeddings to fetch similar strategies from past successes—a common practice in the Agent research domain. Finally, the reflection mechanism, which uses a separate LLM (the Rubric Scorer) to score and provide feedback on generated content, is conceptually identical to the core idea behind agentic reflection frameworks like Reflexion.

- Lack of Deeper Insight: Although the paper successfully jailbreaks the models, it fails to provide deeper insights into the fundamental nature of these LLM security vulnerabilities. It presents an effective attack method but doesn't answer why this method is effective. The lifelong learning module merely reuses similar attack patterns mechanically, without distilling more generalizable principles or patterns from them. For an academic paper, we expect not just a powerful tool, but also a profound understanding of the problem itself.

### Questions
In conclusion, this paper leans heavily towards an engineering-focused integration of techniques, presenting a well-constructed and empirically successful multi-turn attack system. However, its original methodological contributions are quite limited, as it primarily integrates and applies existing ideas. This style feels somewhat misaligned with the research-oriented focus of the ICLR community.
Therefore, I am initially leaning towards a negative rating. My final recommendation will, however, take into account the perspectives of the other reviewers.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces PLAGUE, a plug-and-play framework for designing multi-turn jailbreak attacks on large language models (LLMs). Inspired by lifelong-learning and agentic architectures, PLAGUE divides the attack process into three stages — Planner, Primer, and Finisher — enabling adaptable and modular multi-turn red-teaming. The framework supports integration with prior attacks like GOAT, Crescendo, and ActorBreaker, and achieves significant improvements in attack success rates (ASR) across top-tier models. It also incorporates reflection, memory-based retrieval, and rubric-based evaluation to enhance contextual adaptation.

### Strengths
- A novel multi-step or multi-agent style plug-and-play architecture elegantly decomposes the multi-turn attack into interpretable stages

- Rigorous evaluation and analyes on Harmbench using various backbones and metrics.  

- High empirical performance: signifixantly outperforms both single- and multi-turn attack baselines in ASR 

- Lifelong learning insight with  retrieval-based memory for strategy reuse and adaptation in red timing context 
- Well defined methodology

### Weaknesses
- The main limitation of this paper is that many important related works are missing: X-Teaming (COLM 2025): https://openreview.net/pdf?id=gKfj7Jb1kj, Pandora: (ICLR workshop 2024) https://openreview.net/pdf?id=9o06ugFxIj, Foot-In-The-Floor: https://arxiv.org/pdf/2502.19820, and so on. These methods are also similar. 

- Limited novelty in algorithmic components: The phases (planning, reflection, feedback) heavily rely on established agentic principles (e.g., Reflexion, AutoDAN-Turbo, GOAT), combining rather than innovating core algorithms.

- Setting K=2 appears to me that you are considering up to two turns? Is it so? The scores with just k=1 is very low than they were reported in xteaming.

- Writing, evaluation are very confusing. You mentioned SRE and N-ASR were being used interchangeably, which mean you will be reporting either one 

- No defense-side evaluation: The paper lacks a systematic analysis of how PLAGUE insights could improve model safety — essential for a balanced ICLR contribution.

- Evaluation uses HarmBench only

### Questions
Setting K=2 appears to me that you are considering up to two turns? Is it so?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
4