# Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs

- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Rote learning is a memorization technique based on repetition. Many researchers argue that rote learning hinders generalization because it encourages verbatim memorization rather than deeper understanding. This concern extends even to factual knowledge, which inevitably requires a certain degree of memorization.
In this work, we challenge this view and demonstrate that large language models (LLMs) can, in fact, generalize over rote memorized data. We introduce a two-phase “memorize-then-generalize” framework, where the model first rote memorizes factual subject-object associations using a synthetic semantically meaningless key token and then learns to generalize by fine-tuning on a small set of semantically meaningful prompts. Extensive experiments over 8 LLMs show that the models can reinterpret rote memorized data through the semantically meaningful prompts, as evidenced by the emergence of structured, semantically aligned latent representations between the key token and the semantically meaningful prompts.
This surprising finding opens the door to both effective and efficient knowledge injection as well as possible risks of repurposing the memorized data for malicious usage.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a "memorize-then-generalize" framework in factual learning for large language models (LLMs). The framework involves two phases: first, the model rote memorizes factual subject-object associations using a synthetic, semantically meaningless key token, for example, "Gene Finley [X] Cody Ross", in which [X] serves as a placeholder for the relation. In the second phase, the model is fine-tuned with semantically meaningful prompts, such as "Who is Gene Finley's mother?", effectively assigning meaning to the key token. The authors find that this approach allows the model to generalize to subject-object pairs not seen in Phase-2, handle diverse prompt formulations, and extend to other languages. The authors analyze the internal representations of the model and find that generalization is reflected in structural shifts in representations. The paper also discusses both positive applications, such as efficient knowledge injection and improved reasoning tasks, and potential risks, such as misuse by adversaries to manipulate the meanings of rote memorized data.

### Strengths
- The paper's presentation is clear and well-structured, making it easy to follow the problem formulation, proposed solution, and empirical evaluation. The results are presented in a coherent manner, and the figures and tables effectively illustrate the key findings.
- Given the settings, the experiments are executed well. After reading various results in the paper, I do not have major concerns or confusions about the validity of the experimental results and the conclusions drawn from them.
- The proposed "memorize-then-generalize" framework is an interesting approach to do factual learning in LLMs, and the findings about the model's ability to generalize from rote memorized data are novel and contribute to our understanding of LLMs.

### Weaknesses
- The experimental evaluation often lacks comparisons with other relevant baselines. In most of the experiments presented in the paper, the authors are analyzing the performance of their proposed method under different settings or configurations, but they do not compare their method with other existing methods or baselines. Although the analysis is itself interesting, it is difficult to assess the practical utility of the proposed framework without such comparisons.
- The practical impact of this work is limited. The proposed framework relies on very structured data (subject-relation-object triplets), which may not be representative of the more complex and unstructured data that LLMs typically encounter in real-world applications. It is unclear how this framework would contribute to improving the performance of LLMs on more general tasks beyond structured factual learning. The discussion on potential applications in Section 6 still falls in this realm.

Overall, I think this paper is well-executed and presents interesting findings, so I would not block its acceptance. However, the lack of comparisons with other baselines and the limited practical impact of the work prevent me from giving a more positive assessment.

### Questions
- Feel free to address any of the weaknesses above.
- How are "unrelated" relations constructed in your experiments? For example, if "Gene Finley [X] Cody Ross" means "Gene Finley is the mother of Cody Ross", do you test the accuracies of prompts like (1) "Cody Ross is the mother of Gene Finley" and (2) "Gene Finley is the father of Cody Ross"?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper propose rote learning, which trains LLMs to verbatim-memorize subject–object facts can still support downstream generalization if followed by a light, semantically meaningful fine-tuning stage.
Rote learning has a two-phase procedure: 
1) Memorize facts using a non-semantic key token (e.g., `A [X] B`).
2) Fine-tune on a small set of meaningful prompts so the model learns to answer natural queries (e.g., “Who is A’s mother?”). 
The authors evaluate rote-learning across 8 LLMs and report that models can reinterpret memorized associations to align with semantic prompts.

Analyses show that the synthetic key token is the anchor that gets remapped.

### Strengths
- The paper tackles an important topic of knowledge injection and manipulation
- The idea is simple and effective and seems to be a promising direction for light-weight knowledge injection
- The experiments are comprehensive and convincing; the three generalization scheme setup is appropriate and the analyses are illuminating

### Weaknesses
One obstacle for rote-learning to become practical is that training on synthetic, non-semantic token might degrade model performance on other tasks especially when it's trained for 20 epochs (despite improvement on knowledge injection). The authors already tested the unrelted knowledge in the analysis. It would be good to see if this method is non-disruptive over a wider range of domains beyond knowledge.

### Questions
- To what extent does it break and overfitting starts to happen?
- The result which shows that reversal reasoning can work is a bit surprising to me because this is one of the main problems reported in many recent papers. I wonder how would this compare with other baselines in your setup?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper is motivated by the assumption that rote memorization hinders generalization in large language models. The authors propose a two-phase memorize-then-generalize framework: 1) force the model to memorize subject–object associations 2) then fine-tuning on a few semantically meaningful prompts. They show that the model can reinterpret the key token and generalize to unseen facts, unseen prompts, and even new languages.

### Strengths
- The experimental design is simple and clear, enabling controlled analysis of memory and generalization dynamics.
- Results are reported across multiple models and evaluation types.

### Weaknesses
- The authors mentioned in paper, "To the best of our knowledge, this is the first work to systematically show that LLMs are able to generalize from memorized data", is clearly an overclaim. Many papers that the authors cite in their paper already demonstrate phenomena where generalization emerges after extensive memorization. I think the authors need to have a better understanding of the current work about the generalization emerge from memorization.

- The paper observes limited generalization in its specific setup and then extrapolates to "LLMs can generalize from memorized data" in general. This ignores scope limitations and external validity. 

- The baselines include only SFT and ICL, omitting stronger and conceptually closer comparisons such as replay, model editing, etc.

- The proposed two-phase “memorize-then-generalize” framework is essentially a reformulation of existing paradigms such as fine-tuning, continual learning with replay, or model editing followed by adaptation. The authors need to cite relevant paper instead of claiming this is a novelty of the paper.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the conventional view that rote memorization (verbatim learning through repetition) is harmful to an LLM's ability to generalize. The authors argue that LLMs can, in fact, generalize effectively from data they have rote-memorized. They introduce a two-phase "memorize-then-generalize" framework to demonstrate this. The key finding is that the model learns to reinterpret the [X] token as meaning "mother of" and applies this semantic understanding to all other associations it memorized in Phase 1. This generalization extends to unseen prompts, different facts, and even other languages. The analysis shows that during Phase 2, the model's internal representation of the key token aligns with the representations of the meaningful prompts.

### Strengths
1. The paper presents a more efficient and effective method for knowledge injection than standard SFT or ICL .
2. The decision to use a fully synthetic dataset with fictional entities is a major strength. This eliminates the confounding variable of pre-existing knowledge and ensures that the model is learning the facts entirely from the training, making the findings on knowledge acquisition highly reliable.
3. The findings are shown to be consistent across 8 different LLMs from 4 model families, including both base and instruction-tuned models. This strongly suggests the "memorize-then-generalize" phenomenon is a fundamental property of these architectures, not an artifact of a single model.

### Weaknesses
1. A primary goal of knowledge injection is to update or correct existing, incorrect facts stored in a model's parameters. The paper's framework is not tested in this more realistic and challenging scenario. It's unclear if the "memorize-then-generalize" method would be effective, or perhaps even detrimental, when the rote-learned fact (e.g., Paris [X] Germany) conflicts with strong pre-trained knowledge.

### Questions
1. How does the "memorize-then-generalize" framework perform when the new fact (e.g., Subject [Y] New_Object) conflicts with a fact the model already "knows" (e.g., Subject [pre-trained relation] Old_Object)? Does the Phase 1 rote memorization of the new association successfully override the old one, or does the pre-trained knowledge interfere?

### Soundness
3

### Presentation
3

### Contribution
3
