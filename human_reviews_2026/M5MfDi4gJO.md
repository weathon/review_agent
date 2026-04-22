# Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 6

## Abstract
Large Language Models (LLMs) have shown remarkable success, and their multimodal expansions (MLLMs) further unlock capabilities spanning images, videos, and other modalities beyond text. However, despite this shift, prompt optimization approaches, designed to reduce the burden of manual prompt crafting while maximizing performance, remain confined to text, ultimately limiting the full potential of MLLMs. Motivated by this gap, we introduce the new problem of multimodal prompt optimization, which expands the prior definition of prompt optimization to the multimodal space defined by the pairs of textual and non-textual prompts. To tackle this problem, we then propose the Multimodal Prompt Optimizer (MPO), a unified framework that not only performs the joint optimization of multimodal prompts through alignment-preserving updates but also guides the selection process of candidate prompts by leveraging earlier evaluations as priors in a Bayesian-based selection strategy. Through extensive experiments across diverse modalities that go beyond text, such as images, videos, and even molecules, we demonstrate that MPO outperforms leading text-only optimization methods, establishing multimodal prompt optimization as a crucial step to realizing the potential of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Multimodal Prompt Optimizer (MPO), which jointly refines textual and non-textual prompts through alignment-preserving updates and employs a Bayesian UCB strategy for efficient candidate selection. Experiments across images, videos, and molecules show that MPO outperforms text-only baselines, highlighting multimodal prompt optimization as a key step toward advancing MLLMs.

### Strengths
- The paper expands the prompt optimization space beyond text to incorporate multiple modalities, which represents an interesting and promising research direction.
- MPO shows strong performance across diverse modalities (images, videos, molecules), demonstrating its effectiveness.
- The paper is clearly written and easy to follow.

### Weaknesses
Although the paper claims that MPO is not an Instance-Specific Prompting and Optimization method but rather aims to discover a single, reusable prompt that enhances performance across an entire task, the evaluation still relies on fine-grained dataset partitioning and reducing the number of classes to control task difficulty. This design choice appears to weaken the claimed generalization ability of the proposed method, suggesting that its effectiveness in more complex or diverse scenarios remains to be validated.

### Questions
- Is there a clear order in which the three Exploration Operators designed in this paper are called, and how are they combined? Figure 6 shows that there doesn't appear to be a clear order in which operators are called, and the combinations are arbitrary. The original paper mentions that these operators systematically expand, refine, and recombine non-textual prompts. Could you explain this systematic approach in more detail?
- In the Appendix (lines 665–670), the authors mention that for certain datasets, they further selected groups containing three or four distinct species to maintain a balanced level of difficulty. However, this strategy effectively reduces the overall task complexity, as the generated multimodal prompts only need to distinguish among a small number of categories. It is recommended that the authors further discuss whether the proposed method can still maintain good performance and stability when the number of categories increases significantly. 
- The generation of non-textual multimodal prompts in this paper appears to rely heavily on high-performance image generation and editing models. It is suggested that the authors clarify the extent to which their method depends on the specific capabilities of these models. If alternative image generation or editing models were used, would the proposed approach still maintain its effectiveness and robustness? If the authors can provide convincing explanations or additional evidence regarding these questions, I would be willing to consider increasing the score.

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
3

### Summary
This paper targets the prompt optimization of MLLMs. Unlike prior approaches that only optimize text prompts, the authors incorporate information from another modality into the prompt.
To optimize the multi-modal prompt, the authors propose MPO which consists of two key components: alignment-preserving exploration of multimodal prompt space and prompt selection with prior-inherited Bayesian UCB.

### Strengths
1. The idea is good, which revolutionizes the conventional prompt structure for MLLMs.
2. The paper is well-written, with clear motivation and challenges.
3. The experiments are thorough.

### Weaknesses
1. In the first paragraph of section 3.2, the authors mentioned that "a naive approach that independently updates textual and non-textual components risks producing misaligned prompts". Have the authors tested this naive approach (is it Random Image Prompt in Figure 4? If yes, what are the real performance values instead of performance gains?).
2. When identifying the failure set, how is your multimodal prompt initialized? As it contains both text and image, how are they selected and organized?
3. Another concern is the efficiency: extra models are included to refine the multi-modal prompt. Compared to the baseline methods which optimize textual prompt only, what is the time (or computation steps) used by MPO to complete one iteration?

### Questions
See weaknesses.

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
3

### Summary
This paper introduces Multimodal Prompt Optimization (MPO), a new framework that extends traditional text-only prompt optimization to multimodal large language models (MLLMs) by jointly optimizing textual and non-textual components of prompts. Specifically, the method combines alignment-preserving exploration, which updates text and image prompts coherently based on failure-driven feedback, with a prior-inherited Bayesian UCB selection strategy that leverages parent–child prompt priors to improve sample efficiency. Experiments across image, video, and molecular tasks show that MPO outperforms strong text-only baselines while significantly reducing evaluation cost, suggesting that multimodal prompt optimization can better unlock the reasoning potential of MLLMs.

### Strengths
1.The combination of alignment-preserving exploration with prior-inherited Bayesian UCB is technically sound and elegantly balances multimodal consistency with sample efficiency.

2.The paper is clearly written and well-structured, with intuitive figures and pseudocode that align closely with the algorithmic flow, accompanied by informative ablations and visual analyses.

### Weaknesses
1.The paper frames “multimodal prompt optimization” as discrete text + generated visual prototypes, which risks conflation with learnable soft-prompting (e.g., MaPLe) and needs clearer terminology and positioning.

2.The notion of “parent prompt” and its construction (especially for multi-parent mix) is not formalized.

3.Including powerful image/video generators injects generator priors and extra compute, so gains may stem from tooling rather than the method itself.

4.Efficiency is measured via evaluation counts rather than real cost, omitting overheads from generation and long-context prompting.

5.Cross-modal alignment relies on a single metric (e.g., DSG).

### Questions
1.Which posterior quantile is used for Bayes-UCB, and how do your theoretical conditions differ from standard Bayes-UCB assumptions?

2.If you replace generated prototypes with retrieved in-distribution examples or OOD random images, how do performance and alignment change?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on multimodal prompt optimization for Multimodal Large Language Models (MLLMs), addressing two critical gaps in existing text-only Automatic Prompt Optimization (APO): underutilization of MLLMs’ multimodal capabilities, and inherent challenges (cross-modal inconsistency, sparse high-quality candidates) in multimodal prompt spaces.

Key contributions:  
1. Formalizes the multimodal prompt optimization problem, defining optimal prompts as text-nontext pairs \((t,m)\) that maximize MLLM performance on target tasks.  
2. Proposes the **MPO framework**:  
   - *Alignment-Preserving Exploration* (via cohesive backpropagation, joint multimodal update, and 3 complementary operators) ensures text-image semantic consistency;  
   - *Prior-Inherited Bayesian UCB* leverages parent-child prompt performance correlation to solve cold-start, reducing evaluation budget by 42%-70%.  
3. Validates on 10 datasets across 3 modalities (image/video/molecule), outperforming text-only APO (e.g., +8.6% accuracy on CUB-200-2011).  

MPO fills the multimodal APO gap for MLLMs, with strong cross-model/multimodal generalization and practical efficiency.

### Strengths
- **Originality**: Breaks the text-only limitation of existing APO methods, first formalizing the multimodal prompt optimization problem (defining prompts as text-nontext pairs \((t,m)\)). It also creatively improves Bayesian UCB by leveraging parent-child prompt performance correlation to solve cold-start, extending bandit-based selection to multimodal scenarios innovatively.  
- **Quality**: Conducts rigorous validation—covering 10 datasets across 3 modalities (image/video/molecule), cross-model tests (Qwen2.5-VL, Gemma3), and ablation studies (verifying alignment mechanisms/operators’ necessity). Results are reliable and generalize well.  
- **Clarity**: Clearly presents problem formulation, MPO’s two core components (with formulas and flowcharts), and experimental design. The logical structure is straightforward, enabling easy understanding of the framework.  
- **Significance**: Fills the gap of multimodal prompt optimization for MLLMs, reduces evaluation budget by 42%-70% for practicality, and provides a foundational framework for future multimodal prompt research.

### Weaknesses
1. **Lack of Validation on Mainstream MLLM General Benchmarks, Limiting Evidence of Universal Adaptability**  
The current experiments rely solely on custom task-specific datasets (e.g., CUB-200-2011, PlantVillage) and fail to validate MPO on widely recognized MLLM multimodal benchmarks, leaving its ability to enhance MLLMs’ general capabilities unsubstantiated. Specifically:  
- It omits **static multimodal foundational benchmarks** (e.g., MME, MMBench), which focus on core perceptual capabilities of MLLMs (e.g., image-text matching, attribute recognition)—there is no evidence that MPO can optimize prompt performance for these fundamental tasks.  
- It excludes **long-video dynamic modality benchmarks** (e.g., VideoMME), which require handling temporal alignment between long-sequence videos and text (e.g., locating specific clips, understanding temporal logic). The paper’s existing alignment mechanism, designed for static images, remains untested for such long-video scenarios, casting doubt on its effectiveness.  
- It neglects **cross-modal complex reasoning benchmarks** (e.g., ScienceQA, MathVista, Geometry3k)—benchmarks that demand MLLMs integrate multimodal information to solve logical reasoning or mathematical problems. Since the paper only validates classification/prediction tasks, there is no proof that MPO-optimized prompts can improve complex reasoning performance, leaving MPO’s adaptability to general MLLM scenarios unconfirmed.  

2. **Unaddressed Implicit Deployment Costs, Missing Cost Comparison with Text-only APO**  
While the paper emphasizes that the prior-inherited Bayesian UCB reduces evaluation budget by 42%–70%, it overlooks the significant computational overhead of generating multimodal candidates. Modal-specific generators (e.g., GPT-Image, video editing models) used for image/long-video clip generation incur much higher costs than text prompt generation: for instance, single-image generation takes 2–5 seconds (vs. 0.1 seconds for text generation), and diffusion-based image generators require 3–5 times more GPU memory than text models. Critically, the paper fails to compare the **total deployment cost of MPO** (including iterative multimodal generation overhead) with that of text-only APO. If the implicit costs of multimodal generation offset or even exceed the saved evaluation budget, MPO’s practical utility in real-world deployment would be severely undermined—this key cost trade-off is entirely unaddressed.

### Questions
Same as the section of **Weaknesses**.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper defines the new problem of multimodal prompt optimization and introduces MPO, a framework that jointly optimizes textual and non-textual prompts for multimodal large language models. The method combines alignment-preserving exploration across modalities with a prior-inherited Bayesian UCB strategy for efficient prompt selection. Experiments on diverse modalities (images, videos, and molecules) show consistent improvements over existing text-only prompt optimization baselines.

### Strengths
(1) The idea of extending prompt optimization beyond text is timely and relevant, filling a gap in the growing MLLM literature.

(2) The paper is technically clear, well-written, and supported by convincing experiments on multiple modalities and model backbones.

(3) The proposed Bayesian prior mechanism for efficient search adds a nice practical touch that improves optimization stability.

### Weaknesses
(1) The conceptual jump from text-only to multimodal prompt optimization is natural but not as novel as it’s presented; many parts resemble standard multimodal conditioning or input co-optimization.

(2) The analysis lacks stronger insights into why multimodal prompts help; results show improvement but don’t probe interpretability, modality interactions, or failure cases.

### Questions
(1) How much of the gain actually comes from the added non-textual modality rather than the optimization process itself? A controlled text-only ablation using the same search procedure would clarify this.

(2) How robust is the “alignment-preserving” exploration? When the generated visual component drifts semantically, does the optimization recover or collapse?

(3) The Bayesian UCB prior sounds appealing, but how sensitive is the performance to the prior strength? Could it bias the search toward mediocre parents if the correlation assumption breaks?

(4) Since the framework depends on GPT-based visual and molecular generators, how reproducible is this pipeline for researchers without access to those models?

### Soundness
3

### Presentation
3

### Contribution
3
