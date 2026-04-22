# Visual Abstract Thinking: Enhancing Multimodal Reasoning via Visual Abstraction

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Images usually convey richer detail than text, but often include redundant information, which potentially downgrades multimodal reasoning performance. When faced with lengthy or complex messages, humans tend to employ abstract thinking to convert them into simple and concise abstracts. Inspired by this cognitive strategy, we introduce \textbf{V}isual \textbf{A}bstract \textbf{T}hinking (\textbf{VAT}), a novel paradigm that prompts Multimodal Large Language Models (MLLMs) with visual abstract instead of explicit verbal thoughts or elaborate guidance, permitting a more efficient visual reasoning mechanism via concentrated perception. 
VAT encourages models to focus on more essential visual elements, concepts and structural features by undermining redundant information compared with explicit thinking methods, such as Chain-of-thought (CoT) and tool-using approaches, that increase the complexity of reasoning process via inserting verbose intermediate steps and external knowledge.
Experimental results show that VAT consistently empowers different MLLMs in visual perception and reasoning tasks. VAT achieves an average gain of $2.21\%$ over GPT-5 baseline, surpassing the gain of CoT, demonstrating that VAT better enhances multimodal task performance of MLLMs. Additionally, VAT spends fewer tokens while achieving higher performance. 
These findings highlight the effectiveness of visual abstract thinking and encourage further exploration of more diverse reasoning paradigms from the perspective of human cognition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Visual Abstract Thinking (VAT), a novel reasoning paradigm for multimodal large language models (MLLMs). Instead of relying on explicit textual reasoning such as Chain-of-Thought (CoT) or tool-based intermediate steps, VAT introduces visual abstracts that highlight essential visual elements while suppressing redundant information. Experiments demonstrate that VAT consistently improves visual reasoning performance across different model scales and reasoning modes, with particularly strong gains on spatial and relational reasoning tasks.

### Strengths
The paper presents a thorough experimental study across multiple models, reasoning paradigms, and task types. The ablation analyses are detailed and well-structured, offering meaningful insights.
The overall narrative is well-organized and easy to follow. 
And the reported results demonstrate that VAT achieves both higher accuracy and lower computational cost.

### Weaknesses
While the results are strong overall, the paper would benefit from a more explicit discussion of failure cases or negative results. For instance, Table 1 indicates certain tasks where VAT leads to performance drops. A deeper analysis of why abstraction harms those tasks  would make the work more balanced and informative.

### Questions
1. Beyond the benchmarks used in the paper, I am curious how VAT would perform on tasks that are already abstract or sketch-like in nature—for example, geometry reasoning or diagram understanding tasks where the input is sparse and symbolic rather than visually detailed. Since such tasks already emphasize structural abstraction, would VAT still provide benefits, or could it potentially remove critical geometric cues and thus degrade reasoning?

2. In Table 1, some tasks show performance drops under VAT while CoT yields gains. Combined with the prompt formulation in Appendix B.1, where the instruction enforces “you must fully utilize the provided visual abstracts,” I wonder if a more flexible prompting scheme could help. Specifically, have the authors considered allowing the model to autonomously decide whether or not to use the visual abstract (e.g., “you may use the visual abstract if needed”) and then analyze the resulting task-wise performance differences?

### Soundness
3

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
4

### Summary
This paper proposes Visual Abstract Thinking (VAT), which generates an abstract representation of an original image and uses it together with the original image as input to an MLLM model. This approach allows the model to focus more on the essential visual elements, concepts, and structural features. The authors verify and discuss the effectiveness of VAT through experiments and analysis.

### Strengths
From the experimental results, VAT is shown to be helpful for certain types of multi-modal reasoning (coarse-grained tasks). I also appreciate the ablation analyses presented by the authors in session 4 and 5, including but not limited to the studies on the impact of providing different proportions of sketches, the influence of abstract style combinations and selections, as well as the efficiency.

### Weaknesses
1. Although the authors discuss the differences between Visual Tool-Using Agents and VAT, VAT can still be treated as a special tool-using case.The generation of the abstract representation of an original image relies on other deep learning models rather than being produced by the model itself. These deep learning models should therefore be considered as the model’s tools. This limits the novelty of the approach, making it more like providing the agent with a specialized tool and corresponding result. Considering that different types of tasks may benefit from different abstract styles, making the model to autonomously select and invoke the appropriate styles from different model (abstraction representation generation tool), could potentially yield even better results.

2. As the authors mention in Section 5.1, VAT is particularly helpful for coarse-grained tasks, but its performance gains are limited for the tasks that require fine-grained perception. This indicates that the VAT approach is not sufficiently general, and instead appears to be a task-specific design tailored for particular types of problems.

### Questions
1. What is the relationship between VAT and tool-using agents;

2. Do you consider researching ways for the model to autonomously generate visual abstraction representations to enhance its own visual reasoning ability, rather than relying on external tools.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Visual Abstract Thinking (VAT), a new paradigm designed to improve how MLLMs reason about images. The authors argue that current methods, like CoT, rely on inefficient and distracting verbose text descriptions. VAT replaces this textual reasoning by instead prompting the model with a simplified "visual abstract" of the image, such as a sketch or contour map. This approach helps the MLLM focus on essential visual structures, leading to more accurate and efficient reasoning. The paper's main contributions are demonstrating that VAT achieves higher accuracy than standard CoT and GPT-5 baselines while simultaneously being more cost-effective, requiring significantly fewer tokens and less runtime.

### Strengths
1. The proposed approach is simple yet effective.
2. The method demonstrates broad applicability across various MLLMs, including both proprietary and open-source models.
3. The ablation study on VAT's underlying mechanisms is convincing and provides valuable insights.

### Weaknesses
1. This method relies on external image to sketch tool/models, which is neither context aware, nor task relevant.
2. The 'Visual Abstract Thinking' (VAT) nomenclature could be misleading. Given that the visual abstraction is provided as input rather than generated by the model, 'Visual Abstract Prompting' might be a more accurate descriptor.
3. The analysis lacks certain ablation studies and baseline comparisons:
- An ablation study is needed to show how different sketching methods affect performance.
- A comparison against strong tool-use baselines for perception-intensive tasks. (e.g., Set-of-Mark [1], thinking with images) are missing, considering VAT's advantage of concentrated perception.

Reference:

[1] Yang, Jianwei, et al. "Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v." arXiv preprint arXiv:2310.11441 (2023).

### Questions
1.Could the authors clarify the 'Rand-GT' condition in Figure 4? Its relationship to 'All-GT' is unclear from the figure.

2. There appear to be numerical inconsistencies in the reported gains in Table 1 (e.g., for Qwen2.5-VL+VAT on the ActiView and TextVQA-GT benchmarks). Please carefully verify all reported results.

### Soundness
3

### Presentation
3

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
This paper introduces Visual Abstract Thinking (VAT), a novel multimodal reasoning paradigm. It involves feeding both the original image and an abstract visual representation (e.g., contour maps, sketches) to Multimodal Large Language Models. The aim is to guide models in focusing on critical structures and spatial relationships while reducing interference from redundant details. The study empirically demonstrates that VAT enhances MLLM performance across various visual reasoning tasks, including perception, relational reasoning, and spatial reasoning, offering advantages in computational cost and inference time compared to conventional methods.

### Strengths
1. The paper introduces Visual Abstract Thinking , a new approach that re-imagines how MLLMs process visual information.   By reducing runtime and token consumption compared to existing explicit thinking methods, it offers a more cost-effective approach to multimodal tasks.

2. VAT consistently achieves empirical improvements in various visual reasoning tasks, particularly those focused on object-centric perception, relational reasoning, and spatial understanding. These consistent gains across several benchmarks highlight its effectiveness within its evaluated scope.

### Weaknesses
1. While VAT claims to reduce cognitive load through visual abstraction, introducing a second image input increases the model’s visual payload. The paper should clarify how these abstracts genuinely reduce redundancy rather than simply adding another modality. A quantitative analysis such as mutual information between original and abstract inputs, or a correlation between abstraction fidelity and task performance would better substantiate the claim that VAT acts as a filter, not an augmenter.

2. VAT performs well on structure-sensitive tasks, but its reliance on simplified visual representations (e.g., edges, silhouettes) risks omitting critical cues such as color, texture, or fine-grained details. This raises legitimate concerns about generalizability to tasks like color counting, material identification, or emotion recognition, where such information is diagnostic. In these cases, the abstract may introduce noise rather than clarity, potentially harming performance rather than helping it.

3. The paper treats visual abstraction as a black-box input. To better understand VAT’s mechanism, it would be valuable to examine how the model internally weights or attends to the abstract versus the original image particularly in failure cases. Do attention maps focus on salient regions in the sketch? Is the abstract guiding structural reasoning, or is the effect driven largely by prompting? Such analysis would distinguish between a true shift in reasoning behavior and a superficial prompt effect.

4. The inspiration from human abstract thinking is compelling, but the link between VAT’s design and perceptual cognition remains implicit. Even a speculative discussion connecting the chosen abstractions to principles like Gestalt grouping, visual saliency, or cognitive load theory could help position VAT not merely as an engineering trick, but as a step toward cognitively plausible multimodal reasoning.

### Questions
1. The paper notes that VAT’s benefits are limited, and sometimes even detrimental, for smaller models (e.g., Qwen2.5-VL-7B) compared to larger ones. This suggests that VAT might not be a universally beneficial technique, but rather a "large model privilege" or a method whose efficacy scales with model capacity. Does this imply that smaller models require a different, perhaps even more drastically simplified, form of visual abstraction to avoid overburdening their limited representational capacity? 

2. All experiments in the paper are conducted on static image benchmarks. Applying VAT to dynamic scenarios, such as video understanding or sequential reasoning, presents additional challenges regarding temporal consistency of abstractions. Should abstracts generated for video frames maintain coherence across time, or should they adapt dynamically?

### Soundness
2

### Presentation
3

### Contribution
3
