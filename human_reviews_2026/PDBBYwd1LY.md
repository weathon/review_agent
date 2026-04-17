# Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts

- Decision: Accept (Oral)
- Scores: 6, 6, 8

## Abstract
Large Language Models (LLMs) are widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness critical. A significant and underexplored risk is intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective. Existing studies typically induce deception by explicitly setting a hidden objective through prompting or fine-tuning, which may not reflect real-world human-LLM interactions. Moving beyond such human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth, we propose a framework based on Contact Searching Questions~(CSQ). This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the *Deceptive Intention Score*, measures the model's bias toward a hidden objective. The second, the *Deceptive Behavior Score*, measures the inconsistency between the LLM's internal belief and its expressed output. Evaluating 16 leading LLMs, we find that both metrics rise in parallel and escalate with task difficulty for most models. Moreover, increasing model capacity does not always reduce deception, posing a significant challenge for future LLM development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper attempts to evaluate LLMs that knowingly give deceptive responses to user requests despite being aware of the correct responses. Unlike previous work, this paper focuses on deception by models when users do not exhibit dangerous behavior. The key challenges are how to very precisely distinguish between deception, hallucination, and guessing, and how to elicit deceptive behavior from models when users show no dangerous intent.

The authors propose a novel framework, Contact Searching Questions (CSQ), grounded in cognitive psychology, to detect deception in LLMs under benign prompts. The framework introduces two metrics: Deceptive Intention Score and Deceptive Behavior Score, and evaluates 16 state-of-the-art LLMs across varying task difficulties.

The first metric assumes that LLMs should exhibit relatively consistent performance variations across tasks of similar difficulty. If there are significant performance gaps across different tasks, this may indicate that the LLMs engaged in deception or pronounced hallucination on certain tasks.

The second metric assumes that if a model can correctly answer a simple form of a fact-based task but fails on a more complex form, the model may be exhibiting deception. A score near 0 indicates basic consistency, while a score closer to 1 indicates more severe deception.

### Strengths
- Most of the definitions in this paper are derived from well-established human psychology experiments, giving the work a solid theoretical foundation.

- The task investigated is important: deliberate deception by LLMs when users behave normally.

### Weaknesses
- The paper does not clearly rule out the influence of model sampling randomness. For example, it does not show model performance variability on multiple paraphrases of the same question. Also, around line 360 it is mentioned that δ close to 1 means the model can answer $Q_B$ correctly but cannot answer $Q_L$ correctly—could the opposite occur, i.e., the model answers the complex form $Q_L$ correctly but not the simple form $Q_B$?

- I noticed that for all models, deceptive behavior becomes more pronounced as n increases. Could hallucination be a factor here? How does the paper distinguish hallucination from deception? If the distinction is as shown in Figure 6, what is the basis for the values used in Figure 6?

### Questions
N/A

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
4

### Summary
This paper presents an original and timely investigation into self-initiated deception in large language models (LLMs), moving beyond the common focus on prompt-induced lying. The main contribution of this paper is introducing the Contact Searching Question (CSQ) framework, inspired by cognitive psychology, to evaluate deception on benign prompts through two metrics: the Deceptive Intention Score (measuring hidden goal bias) and the Deceptive Behavior Score (measuring belief–expression inconsistency). Overall, this work offers an empirical and theoretical contribution to understanding and quantifying deceptive tendencies in LLMs.

### Strengths
The authors decompose LLM deception into two categories — prompt-induced and intrinsic — and propose an evaluation framework along two complementary dimensions: Deceptive Intention and Deceptive Behavior, offering a fresh perspective for understanding deceptive tendencies in LLMs.

The paper introduces new, well-defined, and interpretable metrics that capture the degree of deceptive intention and behavioral inconsistency, maintaining both theoretical clarity and practical usability.

Comprehensive experiments on 16 open- and closed-source LLMs reveal consistent behavioral patterns, demonstrating the robustness and generality of the proposed framework.

By grounding the analysis in cognitive science, the study connects LLM deception with established theories of human deception, encouraging meaningful cross-disciplinary discussion.

### Weaknesses
The motivation requires deeper discussion. In particular, the paper should more clearly distinguish deception from hallucination and bias, articulating why deception warrants separate treatment. Furthermore, the significance of studying intrinsic deception should be better justified in terms of real-world implications and safety relevance.

Although several benchmarks for evaluating LLM deception already exist, the paper would benefit from a more comprehensive and detailed comparison with prior benchmarks. Specifically, it should highlight what is unique about the proposed framework and how the new evaluation metrics offer advantages in interpretability, robustness, or theoretical grounding compared to existing approaches.

The relation between the model's tendency of complete or break paths and deceptive intention remains unclear.

Although the psychological framing is engaging, the paper at times overextends human analogies when interpreting LLM behavior.

### Questions
One of the core contributions of this work is the evaluation of deception along two dimensions — Deceptive Intention and Deceptive Behavior. Could the proposed metrics also be applied to prompt-induced deception, and if so, how might the outcomes differ?

What is the motivation for using transitive inference and syllogistic reasoning as the basis for evaluation?
How is the “contact” in the statement “vi can contact vj” (line 283) defined? Is this formulation sufficient to describe the relationships between entities?

Where do the entities (nodes) in the directed graphs come from?

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
3

### Summary
This paper introduces a novel framework to investigate and quantify self-initiated deception in LLMs on benign prompts, a critical and underexplored area of LLM safety. The authors propose the Contact Searching Question (CSQ) framework, which is a synthetic reasoning task designed to be free from prior knowledge contamination. they define two key metrics derived from psychological principles: the Deceptive Intention Score (ρ),  and the Deceptive Behavior Score (δ). By evaluating 16 leading LLMs, the paper demonstrates that: 1) deceptive tendencies increase with task difficulty, 2) deceptive intention and behavior are highly correlated, suggesting a unified underlying phenomenon, and 3) increasing model capacity does not reliably reduce deception and may even increase it in some cases.

### Strengths
1. The experimental design is very strong and well-thought-out.

Using a synthetic reachability task effectively isolates reasoning capabilities from world knowledge, which is a major confounder in other evaluations. The operationalization of deception into Deceptive Intention (ρ) and Deceptive Behavior (δ) is both clever and principled. Measuring intention via performance asymmetry on logically opposite tasks (Definition 3.3) and behavior via inconsistency on complex vs. simple probes (Definition 3.4) are robust methods for getting at the core of the paper's definition of deception.

2. Regarding findings 

The core findings that deception escalates with task difficulty and that intention and behavior scores are highly correlated (Figure 5, Line 405) are convincing. The qualitative analysis of COT processes in Appendix F is particularly good. The evidence of a model "silently" fabricating facts (Fig.16) or failing to perform a search it claims to have completed (Fig. 18) provides a direct window into the deceptive mechanisms.

### Weaknesses
My criticisms are minor and mostly relate to the framing of certain claims:

- The strength of scaling claim seems bit overstated?  The paper claims there is a "clear and concerning trend: both deceptive behavior and deceptive intention scores tend to increase with model size" (Line 895). This claim is primarily supported by Figure 11. However, the reported R² values for the log-linear trend are 0.336 and 0.360. While these values do indicate a positive correlation, they are not strong enough to support such a definitive conclusion. It seems that the trend is noisy and based on a small set of open-source models. The language should be moderated to reflect this, for example, by framing it as a "potential trend that warrants further investigation" rather than a clear one.

- The paper does not discuss how these findings might generalize to other domains where deception could manifest differently. I see this not as a flaw in the current work, but as a crucial limitation to acknowledge and propose as a direction for future research.

### Questions
1.  Given the moderate R² values, have you explored if this trend is more pronounced within specific model families (e.g., looking at the Qwen series alone)? Could the trend be driven by a few influential outlier models?

2. CMIIM the analysis in Appendix D.2 shows that the relative ranking of models is stable across different values of k. But how does the choice of k affect the absolute Deceptive Behavior Score (δ)? Is there a value of k that maximizes the measured deception?

### Soundness
3

### Presentation
3

### Contribution
3
