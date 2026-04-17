# Video models are zero-shot learners and reasoners

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today’s generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn’t explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo’s emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the zero-shot performance of Veo 2 and Veo 3 using a variety of different tasks. The evaluations are grouped into four categories, namely perception, modeling, manipulation, and reasoning. The results indicate emergent capabilities in these tasks, even though they were presumably not trained on them. The success in solving these tasks is compared to the success of Large Language Models (LLMs), and it is argued that, just like LLMs, general vision foundation models will replace specialized models.

### Strengths
1. Vision foundation models are a highly relevant and important topic, and investigating their abilities and shortcomings is of great interest to the scientific community.
2. The proposed tasks across the four categories (Perception, Modeling, Manipulation, Reasoning) are chosen well and help categorize how the model performs in different situations.
3. Both qualitative and quantitative evaluations are conducted, and the qualitative evaluation is extensive and covers a wide variety of different tasks.
4. A project page is included, making it possible for the reader to look at the videos directly, which is not possible in a PDF.

### Weaknesses
My main concern is that the paper contains several claims that are not adequately supported or discussed. The supporting discussion and evidence should be expanded, or the claims moderated accordingly. Specifically, I am referring to the following statements:

1. "Just like LLMs replaced task-specific NLP models". That statement seems debatable. LLMs are very powerful, but the claim that they fully replaced task-specific models requires further discussion or evidence. Saying that task-specific NLP models have been entirely replaced is incorrect [1,2].
2. "Once they become sufficiently cheap and reliable". Is it reasonable to think foundation models could be as cheap and reliable as specialized models that focus, e.g., on edge detection or object grounding? Or is the reasoning behind this claim different? For example, one might argue that such models might never be cheaper in direct comparison, but cheap enough that the small cost is insignificant. In either case, this requires further discussion.
3. "If NLP is a guide, the same trend will play out in vision". A discussion on the similarities and differences between NLP and vision would be a helpful contribution to this paper. What are the similarities between NLP and vision that would indicate that the same pattern will play out? On the other hand, what are the differences and challenges faced by vision that might be challenging for vision foundation models?
4. The chain-of-frame concept is interesting but could be discussed in more detail. It seems a plausible explanation for visual reasoning performance; however, its effectiveness should be further validated, e.g., through comparison with models that generate the final image directly instead of through frame-by-frame generation. The comparison in Section 4.6 to Nano Banana is a good indicator, but more discussion and different tasks would be beneficial. 

[1] Pecher, Branislav, Ivan Srba, and Maria Bielikova. "Comparing specialised small and general large language models on text classification: 100 labelled samples to achieve break-even performance." *arXiv preprint arXiv:2402.12819* (2024).

[2] Xu, Zongzhe, et al. "Specialized Foundation Models Struggle to Beat Supervised Baselines." *The Thirteenth International Conference on Learning Representations* (2025).

### Questions
1. Do I understand correctly that the authors manually evaluated most of the model's outputs? If so, was it always straightforward to determine whether the model’s output should be considered a success or a failure? I would expect many cases where the model’s response is partially correct but not entirely accurate. If such ambiguities occurred, how were they handled?
2. In Figure 56, the example does not appear to be solved correctly (glass 5 should fill up, but then the pipe is blocked in the third image). Is this on purpose, and was this counted as a correct example?

### Soundness
2

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
3

### Summary
This paper demonstrates through extensive experimentation that modern generative video models (specifically Veo 3) possess remarkable zero-shot capabilities across a wide spectrum of vision tasks—from perception to reasoning—suggesting their potential to evolve into general-purpose vision foundation models, much like LLMs in NLP. The key evidence is the substantial performance leap from Veo 2 to Veo 3, indicating rapid progress in this domain.

### Strengths
1.	The paper demonstrates remarkable originality by systematically exploring and validating the provocative thesis that video generation models are emergent, zero-shot generalist vision models, a novel conceptual contribution to the field.
2.	The quality of the empirical evidence is exceptionally high, supported by a massive-scale evaluation across 62 qualitative and 7 quantitative tasks, which makes the core argument both convincing and a significant benchmark for future work.
3.	The work possesses substantial significance as it provides compelling evidence for an impending paradigm shift in computer vision, suggesting a move away from specialized models towards unified foundation models, similar to the transformation witnessed in NLP.

### Weaknesses
1.	While the paper compellingly demonstrates the emergent zero-shot capabilities of video models, it remains a capability showcase and lacks analysis of the underlying mechanisms. A discussion on whether these abilities stem from scale, data, or the video generation objective itself would provide crucial academic depth.
2.	The generational improvement from Veo 2 to Veo 3 is clear, but the claim of being a general-purpose vision foundation model requires benchmarking against other state-of-the-art models (e.g., Sora, Dreamix) to properly establish its standing in the field.
3.	Using best-frame and pass@k (k ≤ 10) inflates results via repeated sampling and selection. This does not reflect realistic usage. The paper should report compute budgets, sampling settings, and variance, and include pass@k vs k and accuracy vs cost curves to support fair comparison.
4.	While the paper demonstrates Veo 3's performance on various tasks, it does not analyze how the model makes its decisions. Understanding the reasoning process in visual reasoning tasks is crucial. Did the authors explore how Veo 3 performs reasoning, especially on complex tasks? Without explainability, the work lacks depth, making it difficult to assess the model’s true potential.
5.	The compelling argument for video models as general-purpose vision foundations is critically undermined by the lack of discussion on practical viability. The claim of potentially replacing efficient, specialized models hinges not only on capability but also on computational efficiency. We strongly recommend the authors address this by including a discussion on the inference cost and latency observed in this study (e.g., via API pricing and generation times), and to explore pathways for future efficiency improvements (e.g., via model distillation or quantization). Without this, the "replacement" thesis lacks credibility and practical guidance for the community.

### Questions
review the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores whether modern video generation models, like Veo3, have zero-shot generalization and reasoning capabilities akin to those of LLMs for langauges. After evaluating on 62 qualitative and 7 quantitative tasks, the authors argue that video models can solve a diverse range of vision tasks, including perception, modeling, manipulation, and reasoning, in zero-shot. Therefore, the paper positions Veo 3 as a potential “foundation model” for vision, analogous to how GPT3 in NLP. The paper is extremely well written, engaging, and comprehensive.

### Strengths
- Clarity and structure: The paper is exceptionally clear and easy to follow. The hierarchy of abilities (perception → modeling → manipulation → reasoning) is intuitive and pedagogically effective.
- Breadth of evaluation: The authors test Veo 3 on an impressively wide range of tasks, demonstrating both creativity and thoroughness.
- Novel perspective: The analogy comparing the NLP trajectory from task-specific models to LLMs is both thought-provoking and pertinent to the vision community.

### Weaknesses
- Lack of standardized automatic metrics: While 7 quantitative tasks are included, most evaluations rely on human eyes instead of automatic metrics. This is problematic for claims of “zero-shot reasoning.” More automatic and reproducible metrics (PNSR, SSIM, LPIPS, etc.) for different downstream tasks respectively would strengthen the argument.
- Limited style and modality diversity: Most examples appear to use realistic-style image. It is unclear whether Veo 3’s performance generalizes to other styles (cartoon, sketch, abstract, etc.) or to non-realistic renderings. Since the claim is about general vision reasoning, this stylistic limitation weakens the universality of the conclusion.
- Overgeneralized framing: The title and abstract make sweeping claims (“video models are zero-shot reasoners”) that exceed the evidence presented. While current experiments demonstrate some results, they lack solid metrics to substantiate these claims.

### Questions
- Could you please provide the total number of unique images? There is a statistical analysis in Appendix B, but Figures 10, 11, and so on show the same image for various downstream tasks. Will this introduce biases into the model, such as making it more susceptible to performing tasks with this specific image?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper argues that large-scale generative video models are on a trajectory to become general-purpose foundation models for machine vision, much as LLMs did for NLP. The authors' core claim is that these models, trained on simple generative primitives at scale, exhibit emergent zero-shot capabilities for tasks they were not explicitly trained on. The methodology consists of prompting the publicly available Veo 3 model  (and its predecessor, Veo 2) with an initial image and a text instruction. The authors then analyze the model's performance on a massive qualitative benchmark of 62 tasks and a focused quantitative benchmark of 7 tasks. The primary finding is that Veo 3 demonstrates broad zero-shot abilities across this stack, substantially outperforming Veo 2 and proving competitive with strong, specialized image models (like Nano Banana) on several tasks. The authors propose the "Chain-of-Frames (CoF)" concept, analogous to Chain-of-Thought, to explain how frame-by-frame generation enables spatio-temporal reasoning.

### Strengths
- This paper is well-written and easy to follow.
- Comprehensive evaluation of 62 tasks covering multiple aspects. 
- Showing strong results, in a zero-shot manner, and covering multiple aspects.

### Weaknesses
The major weakness is that the analysis is specifically conducted under the Veo model, which has the following limitations: 
- Due to a lack of training and designing details, it is impossible to disentangle the single (or even multiple) factors that enable the strong performance of veo model. In other words, it is not surprising that large-scale video generation models show strong results for zero-shot video generation; a few studies also show that these models yield superior performance on non-generation tasks.   
- It's hard to verify whether the conclusion is applicable to Veo only or any SOTA video generation models (e.g., SORA) 

Some minor weaknesses: 
- The success rate for the 62 qualitative tasks was "determined by the authors", which is highly subjective; 
- Also, the claim of zero-shot is also suspicious. Without knowing the training data of veo model, it's hard to detect whether some of the tasks are actually "zero-shot".

### Questions
- The prompt rewriter is the most significant confound. What would the results be if the prompt rewriter is turned off? 


---
Justification:   
This paper presents a fascinating and comprehensive empirical study of the emergent zero-shot capabilities of the Veo 3 video. The breadth of the 62-task qualitative analysis is impressive, and the "Chain-of-Frames" concept is a valuable contribution to how we think about visual reasoning. The clear demonstration of scaling-driven improvement from Veo 2 to Veo 3 supports the central thesis.

However, the paper suffers from two major flaws that make it a poor fit for ICLR:   
- it is fundamentally a technical report analyzing a closed-source commercial product; it offers no algorithmic novelty or reproducible research contribution.
- Its methodology is confounded by the use of a black-box LLM prompt rewriter, making it impossible to attribute the "reasoning" abilities solely to the video model. The "zero-shot" claims are also weakened by the admission of extensive, expert-level prompt engineering.

### Soundness
2

### Presentation
3

### Contribution
2
