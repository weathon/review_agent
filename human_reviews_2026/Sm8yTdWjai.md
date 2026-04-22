# RedDebate: Safer Responses Through Multi-Agent Red Teaming Debates

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
We introduce RedDebate, a novel multi-agent debate framework that provides the foundation for Large Language Models (LLMs) to identify and mitigate their own unsafe behaviors. Existing AI safety approaches often rely on costly human evaluation or isolated single-model assessment, both constrained by scalability and prone to oversight failures. RedDebate employs collaborative argumentation among multiple LLMs across diverse debate scenarios, enabling them to critically evaluate one another’s reasoning and systematically uncover unsafe failure modes through fully automated red-teaming. We further integrate distinct long-term memory modules that preserve safety-relevant insights from debate interactions and leverage them during subsequent inference, facilitating continuous refinement of model behavior. Empirical evaluation on safety benchmarks across a diverse set of models demonstrates that RedDebate substantially reduces unsafe outputs. While debate alone allows LLMs to refine their behavior, the addition of memory modules yields further significant reductions. To the best of our knowledge, RedDebate is the first fully automated framework to unify multi-agent debate and red-teaming to progressively enhance LLM safety without human intervention.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces RedDebate, a MAS debate framework designed to identify and mitigate unsafe behaviors. RedDebate employs fully automated red-teaming to uncover unsafe patterns and uses long-term memory to preserve these insights, leveraging them in subsequent inference. Experiments demonstrate the effectiveness of RedDebate.

### Strengths
- The concept of a fully automated MAS safety enhancement framework is very interesting, and I believe this is a highly practical research direction.

- The authors have conducted very detailed experiments, designing various debate strategies and memory modules. They also performed ablation studies on various hyperparameters to evaluate the effectiveness of RedDebate.

### Weaknesses
- The paper lacks a discussion on the additional time overhead introduced by RedDebate. Although this point is mentioned in the limitations section, I believe it is still necessary to measure and present the time and computational resources consumed, as efficiency and cost are critical in many real-world scenarios.

- The paper lacks comparison with stronger baselines. The authors only compare RedDebate with Self-Critique, while potentially overlooking other work with similar objectives, such as [1] and [2].

[1] arxiv.org/abs/2305.14325

[2] arxiv.org/abs/2305.19118

### Questions
- I am curious whether larger-scale commercial models (e.g., gpt, claude) could benefit from this framework. Or would they simply refuse to answer harmful questions and fail to correct other agents' responses, similar to the behavior of gpt-oss as shown in Appendix B?

- What is the intended attack/defense scenario? Is RedDebate meant to be a pre-processing or training stage for a MAS, where its safety is enhanced through many rounds of debate before being deployed on real-world tasks? Or is the debate itself the end task?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces RedDebate, a multi-agent debate framework for LLM behavioral safety. The authors tested 3 main types of debating mechanisms: peer refinement, devil-angel, and Socratic - Socratic worked best among those. They also tested different mechanisms of long-term memory, which was integrated into multi-agent debate, and showed that incorporating LTM substantially improves behavioral safety. Among the LTM mechanisms, the guardrail approach was overall the most effective.

### Strengths
The idea of applying multi-agent debating is interesting and intuitive. The authors also thoroughly explored different mechanisms of long-term memory to augment debating, which is a novel approach. This paper is clearly written. The appendices provide helpful information that supplement the main text, such as the capability evaluation before vs after safety training and human validation of LlamaGuard.

### Weaknesses
I have a few major concerns about the evaluation, which prevent me from fully understanding the significance of the contribution. I'm open to raising my score if these concerns can be addressed during rebuttal.

1. The current selection of benchmarks doesn't enable robust evaluation on the effectiveness of RedDebate. While HarmBench is a widely adopted benchmark in the safety literature, it's relatively small (with a few hundred examples) and potentially overfit by recent models. CoSafe doesn't seem to be an informative benchmark since the baseline error rate is already very low (7-8%) with little room for meaningful improvement. Including benchmarks that are bigger, more recent, and more able to distinguish different models/methods will provide substantially more information about how effective RedDebate is. nvidia/Aegis-AI-Content-Safety-Dataset-2.0 might be a good resource for this.

2. The evaluation in Table 2 seems unfair for the Self-Critique baseline. Self-Critique outperforms SReD without LTM on HarmBench by a large margin - only when LTM mechanisms are included does SReD beat Self-Critique. My interpretation of this is that debating may not be more effective than self-reflection for mitigating safety, and self-reflection is potentially cheaper since it doesn't require multiple models, which undermines the contribution of RedDebate.

### Questions
1. Line 298: the term "agreement rate" is misleading when the metric quantifies the switch rate from unsafe to safe rather than agreement. Switching from unsafe to safe doesn't necessarily indicate agreement, and agreement doesn't always lead to unsafe to safe switches. Consider using a more accurate metric name.

2. Any insights on why Self-Critique consistently outperforms SReD on HarmBench but underperforms on CoSafe?

3. Table 2: Could you equip Self-Critique with LTM? I wonder if that would outperform SReD + LTM, at least on HarmBench

4. How do different evaluation metrics change over early rounds? Table 7 only shows rounds 3-5, but how about rounds 1-2?

5. What is the inference, computational, and time costs of the various debate methods evaluated in the main text? Is this framework realistic to be deployed at inference time?

### Soundness
1

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
This paper introduces RedDebate, a novel framework designed to enhance LLM safety by automating the red-teaming process. Addressing the scalability limitations of human evaluation and the inherent blind spots of single-agent self-correction, the authors propose a multi-agent system where LLM agents collaboratively debate adversarial or unsafe prompts. This structured argumentation allows agents to critically evaluate one another's reasoning, systematically uncover unsafe failure modes, and iteratively refine their own responses. The paper's primary contributions are the fully automated framework itself, which unifies multi-agent debate with red-teaming; the exploration of different debate strategies (such as Socratic and Devil-Angel) to effectively elicit and correct unsafe behavior; and the integration of distinct long-term memory modules (including textual, parametric, and guardrail-based) that enable agents to learn persistently from previously identified failures. Empirical evaluations on safety benchmarks demonstrate that the RedDebate framework significantly reduces unsafe outputs without human intervention, with the guardrail-based memory approach yielding the most substantial safety improvements.

### Strengths
1. The paper addresses the well-motivated and critical problem of LLM safety, tackling the clear scalability and reliability limitations of existing human-led or single-agent evaluation methods.

2. The manuscript is well-written, clearly articulating the proposed "RedDebate" framework, the experimental setup, and the subsequent analysis of the results.

3. The work provides a great, novel perspective on AI safety by framing it as a learning problem solved through multi-agent interaction and, most notably, by integrating different long-term memory modules (textual, parametric, and guardrail-based) to ensure persistent safety improvements.

### Weaknesses
1. The technical novelty of the framework is somewhat limited, as it primarily integrates and applies existing concepts (multi-agent systems, red-teaming, and memory) rather than introducing entirely new techniques.

2. The paper lacks sufficient baseline comparisons. While it includes a "Self-Critique" baseline, it would be strengthened by comparisons against other contemporary automated red-teaming or multi-agent debate frameworks.

3. The evaluation is limited in its scope, focusing on a specific set of smaller-scale open-source models and two standard safety datasets. The findings' generalizability to larger, state-of-the-art models remains unclear.

4. While some ablation studies are present (primarily in the appendix), the paper lacks a comprehensive ablation on the different components to clearly isolate their individual impact (e.g., the precise contribution of specific debate strategies versus the long-term memory).

### Questions
N/A

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
This work proposes RedDebate, the first fully automated framework integrating **Multi-Agent Debate** with **Red Teaming** to enhance LLM safety without human intervention. Through collaborative debate among model agents, it systematically exposes and corrects potentially unsafe behaviors, significantly outperforming traditional single-agent self-critique or manual red teaming approaches.

### Strengths
1. First to propose and systematically implement the combination of "multi-agent debate + automated red teaming" for LLM safety alignment.
2. Creative introduction of a **long- and short-term memory mechanism** to continuously accumulate safety experience, with experiments conducted on multiple memory variants.
3. Compares multiple debate strategies and validates the corresponding effectiveness on standard safety benchmarks such as **HarmBench** and **CoSafe**.

### Weaknesses
1. Multi-agent debate and memory updates significantly increase inference cost (e.g., debate agents generate 1.3× more tokens per round than Self-Critique). While the authors argue that safety gains justify the cost, the approach may not be applicable in resource-constrained scenarios.
2. Although the primary focus of this work is on improving safety, the experimental design might allow an overly "safe" agent to dominate the debate, potentially leading the entire debate process toward overly cautious conclusions. *Note: This is just a concern, not a confirmed issue.*

**Minor Comments**
Overall visual presentation (figures, diagrams) could be further improved.

### Questions
1. Has the framework been tested with agents of differing safety tendencies (e.g., pairing high-risk models with conservative models) to verify robustness in heterogeneous model settings?
2. Have you considered possible approaches to reduce computation and deployment costs?
3. How is safety ensured during the debate process? For example, a high-risk model may reveal harmful details during discussion — how is this controlled or prevented?

I'm willing to increase my score.

### Soundness
4

### Presentation
2

### Contribution
4
