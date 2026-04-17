# Eliciting Harmful Capabilities by Fine-Tuning on Safeguarded Outputs

- Decision: Accept (Poster)
- Scores: 4, 2, 8

## Abstract
Model developers implement safeguards in frontier models to prevent misuse, for example, by employing classifiers to filter dangerous outputs. In this work, we demonstrate that even robustly safeguarded models can be used to elicit harmful capabilities in open-source models through \textit{elicitation attacks}. Our elicitation attacks consist of three stages: 
(i) constructing prompts in adjacent domains to a target harmful task that do not request dangerous information; (ii) obtaining responses to these prompts from safeguarded frontier models;
(iii) fine-tuning open-source models on these prompt-output pairs. Since the requested prompts cannot be used to directly cause harm, they are not refused by frontier model safeguards. We evaluate these elicitation attacks within the domain of hazardous chemical synthesis and processing, and demonstrate that our attacks recover approximately 40\% of the capability gap between the base open-source model and an unrestricted frontier model. We then show that the efficacy of elicitation attacks scales with the capability of the frontier model and the amount of generated fine-tuning data. Our work demonstrates the challenge of mitigating ecosystem level risks with output-level safeguards.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates elicitation attacks, a novel method of circumventing safeguards on frontier models. Instead of directly extracting harmful knowledge, the authors use frontier models to generate harmless but adjacent outputs, and then fine-tune open-source models on these outputs. The experiments, conducted primarily on chemical synthesis and processing tasks, show that fine-tuned open-source models regain up to ~40% of the performance gap relative to unguarded frontier models.

### Strengths
1. The paper identifies a less explored but realistic pathway for adversaries: leveraging benign outputs to indirectly reconstruct harmful capabilities.
2. The authors evaluate the proposed method across multiple open-source models.

### Weaknesses
1. The work convincingly shows uplift in chemical synthesis, but it is not entirely clear how an adversary would scale this approach to more complex or diverse malicious domains in practice, such as cybersecurity, disinformation, and bioterrorism. The paper would benefit from a stronger justification that such attacks are not just proof-of-concept but present a practical risk.

2. The paper argues that rubric evaluation based on keyword matching is an unreliable measure, and introduces structured comparison evaluation as a more fine-grained alternative. However, this method still primarily measures similarity to reference answers rather than the ability to successfully complete harmful tasks. As a result, the evaluation outcomes are heavily influenced by the quality and biases of the frontier models used to generate those references. Consequently, it remains unclear how much of the reported “40% performance gap recovery” actually translates into genuine security threats.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces elicitation attacks, which aim to elicit harmful capabilities from open-source models. This is achieved by fine-tuning open-source models on benign but adjacent outputs from safeguarded frontier models. Using dangerous chemical synthesis and processing as a case study, the authors show that these elicitation attacks can recover up to ~40% of the performance gap between the original open-source model and the unsafeguarded frontier system.

### Strengths
1. This work explores a new and interesting security risk, whether open-source models can pick up harmful skills.
2. The attack method is simple but effective.
3. The results are consistent across different open-source models and measuring metrics.

### Weaknesses
1. Lack of mechanistic insight: The paper doesn’t explain why fine-tuning on benign, adjacent chemical tasks enables a weak model to gain harmful capabilities.
2. Limited scalability to new and more complex harmful knowledge: The approach relies on fixed, manually designed benign queries, while frontier models will continue to advance and acquire more sophisticated or creative harmful capabilities. However, assuming there is a fixed ground truth for these benign questions, then the benign prompts and outputs remain static, which prevents open-source models from learning genuinely new or more complex harmful knowledge.
3. Style learning or technical learning: Based on the above two weaknesses, it remains unclear why the observed improvement occurs. I suspect that the improved performance may reflect imitation of the frontier model’s style rather than genuine understanding of harmful chemical synthesis. The evaluation might therefore capture stylistic similarity instead of true technical competence.
4. Poor transferability: The method is only tested on chemical weapon tasks, which rely on strong domain-specific prior knowledge and extensive manual design. It remains unclear whether the same approach would work for other harmful domains, since the paper never defines or automates what qualifies as “adjacent” benign prompts. For example, if an attacker wanted to learn how to build a bomb, what would the corresponding adjacent benign questions be?
5. Unclear evaluation procedure: The evaluation method is difficult to understand and interpret. In the provided example (Figure 1), the judgment that “125–135 °C causes decomposition” seems to rely on the evaluator model’s own reasoning rather than on a reference solution. Which of R1 or R2 represents the reference solution? If m is the structured comparison function, could you clarify how m(W) is calculated, as shown in Figure 1?

### Questions
Please refer to the questions in the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes a new elicitation attack paradigm, showing that benign content generated by frontier models can be used to finetune de-safeguarded open source models, which then exhibit harmful domain performance uplift. The authors validate this on chemical synthesis and processing tasks. 

The attack first constructs harmless <prompt, output> pairs by (1) prompts derived from benign compounds in “Compounds” database and then frontier models generate answers, or (2) using relevant topic-based prompting to frontier models that bypass the Constitutional Classifier. Then these harmless pairs are used to fine-tune the open source abliterated models via QLoRA. To quantify the quality of answers,  the authors employ rubric scores and additionally propose the structured comparison. It uses frontier models to generate references and a separate frontier model as a judge to compare tested outputs against references along weighted subgoals. The reliability of this structured comparison is further evaluated by human evaluation, deliberate error detection, and ground truth rating comparison.

Experiments show harmful uplift across multiple open-source models. Ablations examine task type, frontier capability, data scale, and training-domain adjacency.

### Strengths
- The motivation of this paper is clear and the intuition is straightforward.
- The discussions from the angles of harmless-to-harmful generalization and ecosystem attacks (i.e., even if both the input and output of the frontier are filtered, a malicious player can still use benign output to train another unguarded open-source model, circumventing guarding strategies at the “system level”) are very insightful to the community. Thanks to the authors for this interesting work!
- This paper is technically sound with comprehensive and systematic evaluations. Dataset collection is also high-quality and thorough.

### Weaknesses
- Limited algorithmic novelty, as the attack applies standard SFT/QLoRA on novel data and the evaluation design is regular. The main contribution is problem formulation, data construction, and evaluation framing, rather than a new attack technique.
- Results are conducted on abliterated open-source models. It remains unclear whether common safeguard techniques would partially persist under this attack.
- The role of the frontier model in this attack paradigm is unclear.
    - There is no discussion on why only using the <prompt, output> pairs generated by frontier models can have the attack success. Specifically, what are the fundamental differences between LibreChem baseline and the method? If using a language model to rewrite the original LibreChem dataset into <prompt, output> template, how is the performance? I suggest that the authors address this during rebuttal, as currently, it’s hard to tell whether APGR gain comes from frontier quality, instructional formatting, task targeting, or all of the above.
    - Similarly, why newer/stronger frontier models can yield larger harmful uplift would help to understand this question.
- The authors acknowledge key limitations of structured comparison, and it does affect the evaluation fairness. In the current setup, it is still unclear how much of APGR comes from true procedural correctness versus format/length/style alignment to the references.
- It is appropriate to avoid publishing any sensitive content; however, the paper could still provide aggregate, non-sensitive information of the testing benchmark (e.g., numbers of samples, high-level category balance, source provenance, etc.) to support claims of coverage and fairness.
- Compared to the breadth of the attack analysis, the paper offers rather limited discussion from the defensive perspective.

### Questions
See above weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4
