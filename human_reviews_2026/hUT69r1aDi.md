# Hybrid Models for Natural Language Reasoning: The Case of Syllogistic Logic

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 4, 2, 2

## Abstract
Despite the remarkable progress in neural models, their ability to generalize—a cornerstone for applications like logical reasoning—remains a critical challenge. We delineate two fundamental aspects of this ability: compositionality, the capacity to abstract atomic logical rules underlying complex inferences, and recursiveness, the aptitude to build intricate representations through iterative application of inference rules. In the literature, these two aspects are often confounded together under the umbrella term of generalization. To sharpen this distinction, we investigated the logical generalization capabilities of pre-trained large language models (LLMs) using the syllogistic fragment as a benchmark for natural language reasoning. Though simple, this fragment provides a foundational yet expressive subset of formal logic that supports controlled evaluation of essential reasoning abilities. Our findings reveal a significant disparity: while LLMs demonstrate reasonable proficiency in recursiveness, they struggle with compositionality. To overcome these limitations and establish a reliable logical prover, we propose a hybrid architecture integrating symbolic reasoning with neural computation. This synergistic interaction enables robust and efficient inference—neural components accelerate processing, while symbolic reasoning ensures completeness. Our experiments show that high efficiency is preserved even with relatively small neural components.
As part of our proposed methodology, this analysis gives a rationale and highlights the potential of hybrid models to effectively address key generalization barriers in neural reasoning systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores an intriguing question: whether language models can effectively generalize in syllogistic logic reasoning tasks, particularly in terms of compositionality and recursiveness. The study finds that while language models perform reasonably well on recursive generalization, their ability to generalize compositionally remains limited. To overcome these limitations, the authors propose a hybrid model designed to address key generalization barriers in neural reasoning systems. I appreciate the methodology adopted in this paper --- the controlled experiments on language models reflect a rigorous design.

### Strengths
1. The paper addresses an intellectually engaging research question.

2. The paper adopts a rigorous and well-structured methodology to explore it.

### Weaknesses
1. If the paper could clearly articulate the representative significance of syllogistic logic for reasoning, as well as its potential implications for extending to more complex forms of reasoning, the overall presentation would be further improved.

2. Perhaps pretraining is a more suitable approach than SFT, since it remains uncertain whether existing pretrained models retain knowledge related to syllogistic reasoning [1]. For example, if the pretraining data contains A→B, B→C, A→C, and your constructed dataset contains swa→cdf, cdf→yur, swa→yur, they may not overlap verbatim, but the underlying logical structure is similar. I believe it is important to reflect on whether such structural generalization might influence performance in ways not fully accounted for.

[1] Physics of language models: Part 2.1, grade-school math and the hidden reasoning process. ICLR 2025. https://arxiv.org/abs/2407.20311

### Questions
See Weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes the short comings of LLMs in extrapolation performances in the syllogistic reasoning task, and propose to use LLMs as an assistant to a symbolic prover to reduce the number of steps explored.

### Strengths
1. The paper performed detailed analysis on a synthetically created syllogistic reasoning task, and exposed the weakness of the current LLMs.
2. The proposed hybrid model can reduce the number of steps need for the solver to find the proof.
3. The paper is written well and easy to follow.

### Weaknesses
1. While the paper proposed to have a hybrid model that use the LLMs as assistant to a symbolic solver, it doesn't propose anything to resolve the issue in LLM itself. The paper will also be stronger if the authors can provide analysis on why LLMs cannot extrapolate well in the syllogistic reasoning task.

2.I also want to ask if using LLMs as an assistant to the symbolic solver will actually incur more computational cost than just use the symbolic server alone, given LLM inference can be expensive.

### Questions
see weakness.

### Soundness
3

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
5

### Summary
This paper evaluated the generalisation capabilities of Large Language Models (LLMs) in natural language reasoning, focusing specifically on syllogistic reasoning.

This work focused on two important topics in logical reasoning: compositionality and recursiveness and showed that while LLMs perform reasonably well on recursiveness, they exhibit significant difficulty with compositionality.

Authors propose a hybrid architecture that integrates symbolic reasoning with neural computation.

### Strengths
Authors use syllogistic logic to explore LLMs' reasoning abilities, and carefully experiment with two fundamental reasoning abilities: composition and recursion.

### Weaknesses
To examine whether LLMs can do composition and recursion of syllogism, authors seem to assume that LLMs can do classic syllogistic reasoning. However, recent research already show that LLMs even struggle with single-step syllogistic reasoning. 

Tiwalayo Eisape, MH Tessler, Ishita Dasgupta, Fei Sha, Sjoerd van Steenkiste, and Tal Linzen. A
systematic comparison of syllogistic reasoning in humans and language models. In NAACL, 2024.

Andrew K Lampinen, Ishita Dasgupta, Stephanie C Y Chan, Hannah R Sheahan, Antonia Creswell,
Dharshan Kumaran, James L McClelland, and Felix Hill. Language models, like humans, show
content effects on reasoning tasks. PNAS Nexus, 3(7), 2024.

Magdalena Wysocka, Danilo Carvalho, Oskar Wysocki, Marco Valentino, and Andre Freitas. SylloBio-NLI: Evaluating large language models on biomedical syllogistic reasoning. ArXiv:2410.14399, 2025.

Authors propose neurosymbolic approach. This is not new. The update-to-date method is to develop novel extendable neural architecture that can achieve symbolic-level syllogistic reasoning. 

Tiansi Dong, Mateja Jamnik, and Pietro Li`o. Neural Reasoning for Sure Through Constructing
Explainable Models. In AAAI, 2025.

### Questions
no

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates the logical generalization capabilities of LLMs by distinguishing between two faculties: compositionality and recursiveness. Using syllogistic logic as a controlled benchmark, the authors conclude that LLMs exhibit reasonable recursiveness but struggle with compositionality. To address this limitation, the paper proposes a hybrid neuro-symbolic architecture that integrates neural computation (as a "Neural Assistant") with a formal symbolic prover.

### Strengths
1. A key strength is the paper's conceptual distinction between compositionality and recursiveness . This framing helps the research community delve into a more detailed and nuanced analysis of generalization in neural models.

2. The paper is self-contained. While the significance or novelty of the individual components may be debatable, the work clearly identifies a specific problem (poor compositionality) and proposes a complete, functioning solution (the hybrid model) to address it.

3. The use of synthetic data and pseudowords is a methodologically sound practice. It effectively isolates logical form from content bias, which is crucial given that modern LLMs are prone to memorization.

### Weaknesses
1. The paper tests FLAN-T5-base and GPT-40-mini. While the results suggest that the compositionality gap is a structural problem, the claim that "scaling to larger models alone may not be sufficient" is not fully proven without testing against today's largest frontier models. 

2. The proposed hybrid method, which uses a neural module to reduce the search space for a symbolic prover , is not entirely novel. The core idea is similar to prior work (e.g., Neural Logic Machine) and can be seen as a straightforward application of LLMs to a specific logical domain. Given the paper's core finding that LLMs struggle with compositionality, a more compelling and original contribution would have been a novel method to solve this compositionality problem directly, rather than bypassing it with a hybrid system.

3. I have concerns about the interpretation of the experimental results. The paper defines compositionality as deconstructing complex structures into simpler components and recursiveness as combining simple structures into complex ones. This framing suggests that compositionality is essentially being treated as the reverse process of recursiveness, which is analogous to other known limitations of transformers, such as the "Reversal Curse" [1]. Moreover, recursiveness is an "easy-to-difficult" generalization (training on simple/short chains, testing on complex/long ones) , while compositionality is a "difficult-to-easy" generalization (training on complex/long chains, testing on simple/short ones). Given this setup, the finding that compositionality is more challenging seems straightforward and expected. The authors should provide a more in-depth justification for why this distinction is insightful.


Overall, I have reservations about this paper's core claims, both in its experimental conclusions and its proposed method. I look forward to the authors' response to the weaknesses raised and will reconsider my score based on their answers.

[1] The Reversal Curse: LLMs trained on “A is B” fail to learn “B is A”

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
