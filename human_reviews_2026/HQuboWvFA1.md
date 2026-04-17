# Monitoring Decomposition Attacks with Lightweight Sequential Monitors

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
As LLMs become more agentic, a critical risk emerges: attackers can \emph{decompose} harmful goals into stateful, benign subtasks that trick LLM agents into executing them without realizing the harmful intent in the same context. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent. We therefore propose adding an external monitor that observes the conversation at a higher level. To facilitate our study on monitoring decomposition attacks, we curate the largest and most diverse dataset, DecomposedHarm, with 4,634 tasks that can be assigned to LLM agents, including general agent tasks, text-to-image, and question-answering tasks, where each task has a benignly decomposed version. We verify our datasets by testing them on frontier models and show an 87\% attack success rate on average on GPT-4o. To defend in real‐time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each sub‑prompt. We show that a carefully prompt-engineered lightweight monitor hits a 93\% defense success rate—outperforming strong baselines such as Llama-Guard-4 and o3-mini, while cutting costs by 90\% and latency by 50\%. Additionally, we show that even under adversarial pressure, combining decomposition attacks with massive random task injection and automated red teaming, our lightweight sequential monitors remain robust. Our findings suggest that guarding against stateful decomposition attacks is "surprisingly easy" with lightweight sequential monitors, enabling safety in real-world LLM agent deployment where expensive solutions are impractical.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper shows that well aligned models such as GPT-4o and Llama-Guard are vulnerable to decomposition attacks, and are effective across different settings. It proposes that lightweight models can cumulatively monitor the queries of decomposition attacks as they progress, and through careful prompt engineering, outperform much stronger baselines in terms of detection rates and cost. The paper also introduces an extensive new dataset for decomposition attacks, where each task is broken down seemingly benign subtasks.

### Strengths
1. The paper is timely, as this problem is a growing concern.
2. The solution is simple, cheap and effective, beating far more expensive zero-shot baselines.
4. The evaluation is extensive, the metrics used are appropriate and the results are convincing.
3. The dataset will prove very useful to the area going forward as a benchmark.

### Weaknesses
1. The defense is reliant on an engineered, static prompt to control detection behavior, which raises concerns about adaptive attacks. While the PAIR baseline does give an adaptive attack (where subtasks are made more benign while maintaining semantics), it doesn't include the system prompt as part of the input, which can lead to a suboptimal objective for the adversary. 

2. The dataset could more details and descriptions regarding the diversity of tasks (subcategories), length of decomposed prompts, whether the subtasks are independent of each other, etc.

### Questions
1. How well does the sequential monitor framework perform when the ICL prompt is included as part of PAIR's input? 

2. Are there scenarios where prompts can be decomposed into independent subtasks? Adversaries could make singular independent queries to avoid providing a cumulative prompt history to the monitor.

### Soundness
3

### Presentation
2

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
The paper examines decomposition attacks against LLM agents. These involve breaking down malicious intents into subtasks that are safe when taken in isolation and can bypass safety filters. 

The paper presents DecomposedHarm, a dataset of 4634 human-annotated agent tasks for conducting and evaluating such attacks, and finds that they are up to 87% successful against GPT-4o. 

The authors also develop a defense using sequential monitoring of LLMs which can detect these attacks with up to 93% success and with low additional latency.

### Strengths
1. The research problem holds significant value, as attacks decomposed through simple prompts remain undetectable by LLMs. This issue exhibits universality, scalability, and urgency, while the novel dataset DecomposedHarm demonstrates high research value.
2. The author innovatively introduces a defense method achieving a maximum defense success rate of 93%, while maintaining robustness under adversarial conditions.
3. DecomposedHarm is an extensive and diverse dataset for studying decomposition attacks, providing clear splits (Table 4) and strong empirical visualization (Figure 2).
4. The authors provides solid quantitative analyses (Figures 2&5, Tables 1–3), consistently showing decomposition sharply reduces refusal rates and generalizes across models and interaction modes.

### Weaknesses
1. Applying o3-mini, GPT-4o, and Claude-3.7-Sonnet model can reach the highest F1 scores, but the costing problem is also concerning, especially examing cumulative context.
2. The authors only applies in-context learning (prompt engineering) methods to improve the sequential monitors. 
3. The ability of defense in the method does not stem from the robustness of the algorithm itself, but rather from the accidental effect of the prompt wording.
4. Although the decomposition attack (Fig. 9) demonstrated in the paper is effective, it is limited to non-adaptive scenarios and single pipelines.

### Questions
1. How much does monitoring performance drop if the best-performing ICL or CoT prompts are replaced with simpler, less-engineered prompts? Can the authors quantify the risk that monitoring is reliant on specific prompts? 
2. If a single attack prompt can be decomposed into many sub-questions, achieving high F1 could bring significant token consumption. How do the authors address this scalability issue?
3. Please report the distribution of harmful metric percentages (harmful metric / total subtask length) and stratify performance reporting by this percentile in test evaluations. Fail to do so may introduce bias in assessments due to the tendency to place harmful steps in advantageous positions.
4. Please evaluate the framework under varying benign-subtask injection rates by fixing the fraction of benign subtasks (e.g., 25%, 50%, and 75% ,benign subtasks / total subtasks). For each setting report F1, cost per task, and average latency. No need for generating new tasks, just pick the tasks that satisfies corresponding ratio already had.

If the main concerns have been addressed, I am considering raising my score.

### Soundness
2

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
3

### Summary
The paper looks at decomposition attacks in the context of LLM agents. These are attacks where a harmful task is broken up into a sequence of benign steps that are cumulatively harmful. The authors produce datasets that capture this category of jailbreaks and then introduce sequential monitors that can detect decomposition attacks. They demonstrate that prompt optimization of a sequential monitor can block decomposition attacks more effectively than standard filters.

## Contributions

 * A novel dataset of decomposition attacks
 * A sequential monitoring method for filtering
 * Results that indicate their sequential monitoring is feasible with reasonably small monitoring models

### Strengths
Overall, the paper deals with an important area of concern: decomposition attacks are an effective attack and mitigations are an important area for further study. As a result, they are studying a problem of interest to the community.

The dataset contribution is the most valuable from my perspective, although prior work has defined and demonstrated decomposition attacks, I believe there is no existing dataset of decomposition attacks that allows for comprehensive evaluation or study.

Separately, the approach described for defense seems reasonable, and the authors do a good job of balancing concerns for the effectiveness of monitors with the feasibility of deployment. 

In summary, the paper studies an important problem and provides a reasonable defense proposal. The evaluations indicate that this is a potentially promising direction for the development of new guardrails.

### Weaknesses
The primary weakness of the paper is a heavy level of overclaiming. The introduction and title suggest that it is "surprisingly easy" to defend against these attacks. However, I do not believe the author's evaluations are sufficient to justify such a claim. There are a few specific issues with the framing:

 * The authors study a variant of the problem where the decomposition attack occurs within a single context. This is easily bypassed if the attacker can carry state over from one context to another. For example, they describe an image generation task to generate a potentially racist propaganda image that violates guidelines. However, it seems like it would be straightforward to split these requests into several contexts and avoid the defense mechanisms proposed. While this is briefly mentioned in the limitations section, it deserves further discussion and motivation for why this particular attack setting is appropriate for study.

 * While the authors do some study of adaptive attacks, it is not clear that these are representative of how a capable attacker would behave. In particular, 10 attempts by GPT-4.1 is a valuable measurement, but far below the level of resources that a capable attacker could apply to such a system. 

 * Most of the comparisons show performance against o3-mini as an expensive evaluator; they don't show that performance is good enough from an objective risk minimization perspective. As a result, the evidence the authors actually have supports that lightweight sequential classifiers can be optimized through prompt optimization to perform similarly to more expensive classifiers. This is a different and weaker claim. 

 * It seems possible that there is information leakage from the similarity in how the data is generated. There's an attempt to account for this by using in-context examples from one task and exploring how well it transfers to another. I'm not sure what justifies this approach. Similarly, the fact that the examples from the train/validation set are programmatically selected by the same process as the test set means that this approach risks undercounting vulnerability. 

## Clarity about evaluations

I also have some questions about the evaluation setup overall. 

 * First, in comparison to GPT-4o and o3-mini as a reference,  why are the prompts unoptimized? It seems to me that it would be appropriate to compare to an optimized prompt for the reference models as well. This provides an effective limit on performance. By comparing against an unoptimized prompt for the more expensive models, you are stacking the deck in your favor. This makes the analysis of the results misleading: the optimized sequential monitor matches the performance of a more expensive, but unoptimized model. 

 * Second, it's unclear why the authors are comparing 4.1-nano with 4o-mini. This is a bit strange because they are different model sizes and I would expect 4.1 to be more effective as a guardrail model. This raises some questions about the overall choices of which models to compare and why they were chosen over alternatives.

### Questions
* How did you evaluate the quality of your LLM-judge for harmfulness?
 * How did you select which models to use for your experiments? E.g., why is 4o-mini being compared with 4.1-nano?
 * Is there a way to evaluate your performance against a better-resourced adversary?
 * Can you explain why these defenses are interesting, given that we can/should expect decomposition attacks to be executed across different contexts?
 * Can you clearly and concisely articulate how the evidence in your paper supports the claim that defending against decomposition attacks is 'surprisingly easy'?

### Soundness
2

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
3

### Summary
The submission produces a dataset ("DecomposedHarm") of highly-effective LLM jailbreaks that rely on decomposing a harmful query into multiple benign-seeming subtasks. Careful experiments show that LLM-based sequential monitors (classifiers of the subtask/prompt sequence) can accurately identify when these subtasks will begin to induce a harmful output. To address the financial cost and latency of such LLM-based monitors, prompt engineering is used to enhance the performance of lightweight LLM sequential monitors (beyond the performances of more sophisticated and expensive models). Notably, even when additional adversarial strategies are added to the decomposition attack, the lightweight sequential monitors robustly discriminate between benign and harmful subtask sequences.

### Strengths
**Originality** The work introduces a new dataset (DecomposedHarm), and a simple, practical method for addressing decomposition attacks.

**Quality** DecomposedHarm addresses a variety of potential jailbreak settings, from image generation to agentic LLM tasks. Comparisons include strong baselines like Llama Guard, which is greatly outperformed. Adversarial evaluation provides additional evidence of the proposed method's benefits. Decompositions are verified by human reviewers. The limitations are thorough and helpful. 

**Clarity** The paper is very well written, with clear figures and tables.

**Significance** The submission addresses an urgent problem of practical importance. The dataset DecomposedHarm will facilitate future research in this area.

### Weaknesses
The submission has no significant weaknesses. 

In the limitations section, perhaps emphasize that decomposition is just one of many attack strategies, and decomposition could potentially play a role in building composite attacks (e.g. with genetic algorithms) stronger than those explored in the submission.

### Questions
1. Line 178: why are the harmful indices the last indices? Couldn't the image become harmful before the last subtask?
2. Did you employ any checks for redundancy in the LLM generated examples that populate DecomposedHarm? A little redundancy is okay, but not if you have overlap between your validation and test sets. 
3. Adding a newer model (Gemini 2.5, Sonnet 4.5, and/or GPT 5) as a reference in Table 1 could boost the relevance of the analysis and clarify what the frontier is. Relatedly, a newer compact model (Haiku?) could be a strong candidate for prompt-engineering. 
4. In Table 1, GPT 4o without any optimization appears cost effective and performant. It would be interesting to see how the references perform with a prompt (latency and F1).
5. Table 1: The GPT 4o mini F1 does not seem to match Table 6’s. 
6. Line 351: there seems to be a typo here – is “o3-mini” supposed to be “GPT 4o”?

### Soundness
4

### Presentation
4

### Contribution
3
