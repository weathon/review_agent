# Review

## Summary
This paper investigates whether LLMs can develop human-like semantic systems that efficiently compress meaning, focusing on color categorization as a test case. The authors replicate human studies using LLMs and introduce a new method called Iterated In-Context Language Learning (IICLL) to simulate cultural evolution of language. Their key finding is that while many LLMs struggle to align with human color naming systems, some frontier models like Gemini 2.0 demonstrate human-like efficiency in categorization, as evaluated using the Information Bottleneck (IB) framework. The paper contributes to understanding how semantic efficiency emerges in AI models and whether they can develop human-aligned linguistic structures without direct training for this purpose.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper combines insights from cognitive science and information theory through the Information Bottleneck (IB) framework to evaluate semantic efficiency in LLMs. This interdisciplinary approach provides a unique and rigorous theoretical foundation for studying how LLMs might develop human-like semantic systems.
2. The authors conduct a comprehensive analysis across 39 models, varying in size, training stages, and modality (text-based and multimodal), which strengthens the generalizability of their findings. The detailed comparison highlights the performance differences and reveals how factors like model size and instruction tuning impact IB efficiency.
3. The introduction of IICLL as a method to simulate cultural transmission of language is a significant contribution. It allows for studying how LLMs might evolve language structures over generations, providing valuable insights into the developmental process of linguistic systems in AI.

## Weaknesses
1. The paper primarily focuses on color categorization, which, while important, might not fully represent the complexities of human language and cognition. The findings might not generalize well to other semantic domains that involve more abstract and nuanced concepts.
2. The study relies on prompt-based interactions with LLMs, which may not fully capture the complexities of human language use and cultural transmission. The models' performance could be influenced by the specific prompts used, potentially limiting the robustness of the findings.
3. The paper acknowledges that the origins of the observed efficiency bias in LLMs are unclear and that investigating this further is an important direction for future work. This suggests that there may be underlying mechanisms or properties of the training data, instruction tuning, or model size that contribute to the observed behavior, but these are not fully understood. This lack of clarity could limit the interpretability and replicability of the results.

## Questions
1. How well do the findings generalize beyond color categorization? Can the observed human-like semantic efficiency in LLMs be replicated in other semantic domains that involve more abstract concepts?
2. What are the underlying mechanisms that drive the emergence of human-like categorization systems in LLMs? Is it solely due to the model architecture, the training data, or something else?
3. How robust are the findings to variations in prompts and interaction schemes? Could the performance of LLMs change significantly if different prompt engineering techniques were used?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4