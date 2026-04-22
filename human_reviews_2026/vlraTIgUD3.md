# InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
The ability for AI agents to "think with images" requires a sophisticated blend of reasoning and perception.
However, current open multimodal agents still largely fall short on the reasoning aspect crucial for real-world tasks like analyzing documents with dense charts/diagrams and navigating maps.
To address this gap, we introduce O3-bench, a new benchmark designed to evaluate multimodal reasoning with interleaved attention to visual details.
O3-bench features challenging problems that require agents to piece together subtle visual information from distinct image areas through multi-step reasoning.
The problems are highly challenging even for frontier systems like OpenAI o3, which only obtains 40.8\% accuracy on O3-bench.
To make progress, we propose InSight-o3, a multi-agent framework consisting of a visual reasoning agent (vReasoner) and a visual search agent (vSearcher) for which we introduce the task of generalized visual search---locating relational, fuzzy, or conceptual regions described in free-form language, beyond just simple objects or figures in natural images.
We then present a multimodal LLM purpose-trained for this task via reinforcement learning.
As a plug-and-play agent, our vSearcher empowers frontier multimodal models (as vReasoners), significantly improving their performance on a wide range of benchmarks.
This marks a concrete step towards powerful o3-like open systems.
Our code and dataset can be found at https://github.com/m-Just/InSight-o3.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel multi-agent framework that decomposes multimodal reasoning into two agents: a visual reasoning agent (vReasoner) that performs CoT reasoning and provides answers, and a visual search agent (vSearcher) that outputs coordinates given a description. Given an input, the vReasoner processes the answer and, when needed, calls the vSearcher to find a relevant visual region. The authors also built InSight-o3-vS using a novel RL training pipeline and presented the O3-Bench to evaluate complex visual reasoning tasks.

### Strengths
1. I think decomposing mulitmodal reasoning model into two agents can be efficient without the need of training the whole model, especially with strong closed source models we cannot train. 
2. Training pipeline and constructing collage data for vSearcher is interesting and novel.
3. O3-Bench can be beneficial to the community.
4. Nice addiotional observation report from the hands-on experience.

### Weaknesses
1. **Reliance on Closed-Source Models**: The experiments rely heavily on closed-source models (like GPT-5-mini), which are difficult to use for reproduction or further training. As a result, training the vSearcher based on a closed model lacks scalability and reproducibility for the wider research community. I think experiments with open-source models can improve this paper significantly
2. **Lack of Experiments**: While the vSearcher is an interesting approach, its validity would be more convincing if it were tested more rigorously.
3. **Limited Benchmark Coverage**: The O3-Bench has a relatively small scale (185 images, 318 QA samples). This is a limitation, especially since the chart images are sourced from the existing MME-RealWorld dataset.

### Questions
1. Could you provide results for the vSearcher using different model sizes, or paired with open-source vReasoner models (such as Qwen2.5-VL or InternVL3)?
2. The paper states that the O3-BENCH chart images are from MME-RealWorld. Could you clarify in more detail how the tasks (i.e., the questions) for this domain differ from the original MME-RealWorld chart domain?
3. Does the number of layouts in the O3-Bench data affect model performance? (For example, is there a correlation between a higher layout count and lower model accuracy?)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to augment visual-language reasoning models with the capability to perform a visual search function. In particular, it trains an additional visual search module that predicts the bounding box coordinate related to a given description (e.g., instructions derived from the input question), and feeds the retrieved image patches to pretrained reasoning models. A new multi-modal reasoning dataset is also presented, which focuses on reasoning on complex charts and digital maps. Experimental results show that the method improves the performance of several state-of-the-art reasoning models across different datasets.

### Strengths
(1) It is a promising direction to improve multi-modal reasoning models by incorporating external tools (i.e., the visual search module in this case), which could provide both performance boost and enhanced transparency.

(2) The use of collages for visual search training reduces the reliance on large-scale naturalistic data.

(3) The paper develops a new evaluation benchmark for multi-modal reasoning, and can benefit the development of subsequent models.

(4) The proposed method shows generalizability across several models and datasets.

### Weaknesses
(1)  It is not a new idea to combine VLMs with external tools (e.g., some compositional reasoning models [ref1, ref2] already explore the tool usage with reinforcement learning). The paper experiments with a single tool (i.e., visual search framed as a visual grounding task), while solving real-life problems could require diverse abilities. It is unclear whether visual search (especially when trained independently) can help generalize reasoning across different scenarios.

(2) One advantage of having an additional visual search agent is to provide a transparent interface of the decision-making process. Nevertheless, the paper only reports the accuracy on datasets, without any analysis of how the improvement is achieved with visual search.

(3) Looking at Table 1, it appears that the proposed method only shows consistent improvement on models from the GPT family, and can have negative effects on the best-performing Gemini model. Please justify the inconsistency in performance.

(4) The proposed visual search module is trained on synthetic collages created by stitching together different images. Such a paradigm ignores the contextual relationship between different regions within a visual scene, and also introduces boundary artifacts. Since the search agent is essentially trained on a visual grounding task, I wonder how it will perform when training on naturalistic grounding datasets.

(5) The new dataset contains a very limited set of stimuli (~350 images), making it difficult to be used for training or comprehensive evaluation.

[ref1] ViperGPT: Visual Inference via Python Execution for Reasoning. CVPR, 2023.

[ref2] HYDRA:AHyper Agent for Dynamic Compositional Visual Reasoning. ECCV, 2024.

### Questions
(1) What are the advantages of visual search over other types of external tools?

(2) How does the visual search agent help the reasoning agent? Please provide in-depth analysis of the decision-making process.

(3) Why does the model only improve the GPT models but lead to worse performance on Gemini? Is it related to the use of GPT-nano for evaluation?

(4) Please justify how the proposed training paradigm could accommodate the artifacts in synthetic data.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limitations of current multimodal agents in visual reasoning by introducing a new benchmark, o3-bench, which requires models to integrate fine-grained information from multiple image regions while performing complex reasoning. To tackle this challenge, the researchers developed the InSight-o3 multi-agent framework, focusing on its vSearcher module with generalized visual search capabilities. Trained via reinforcement learning, vSearcher can locate conceptual visual regions based on natural language instructions and, as a plug-and-play component, significantly enhances the performance of state-of-the-art models across multiple benchmarks.

### Strengths
1. The paper clearly identifies a specific weakness in current multimodal agents—their inability to perform complex reasoning that requires integrating fine-grained visual details. It presents a challenging benchmark (o3-bench) designed explicitly to measure this underdeveloped capability.

2. The proposed InSight-o3 framework presents a sophisticated multi-agent architecture that decomposes the complex problem into specialized sub-tasks (reasoning and search).

### Weaknesses
The claim that the framework's search steps decrease with increasing resolution is not sufficiently supported, as the reported variations across resolutions are minimal. This suggests the search pattern may be overly reliant on the characteristics of the training data, raising concerns about its scalability and effectiveness in real-world, multi-step search-and-reasoning tasks involving high-resolution images.

The framework's performance on powerful yet tool-agnostic models like GPT-4o and Gemini-2.5 (as seen in VisualProbe-Hard and MME-RW-Lite results) is suboptimal. This indicates a strong dependency on models pre-equipped with tool-calling capabilities trained with multi-turn RL, highlighting a limitation in its general applicability. Further optimization is required to improve its stability and performance across a wider range of model architectures.

The ablation study on hybrid RL training reveals a trade-off: while combining static and dynamic RL improves performance on the proprietary O3-bench, it results in higher per-step inference latency without a clear balance of their respective advantages (static RL's speed vs. dynamic RL's adaptability). Furthermore, the performance gains appear primarily concentrated on O3-bench, questioning the hybrid method's generalization to broader tasks.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of using an LLM reasoning with a specialist LLM visual reasoner which can be called as a tool. The paper makes two contributions: (1) a new, complex, yet quite small dataset of  questions which requires multi-modal reasoning. Half of these are complex questions about maps, as well illustrated in Fig 1, requiring zooming in and reading the legends of the maps. (2) optimize a special visual searcher (i.e. visual LLM) in the context of a strong LLM-based reasoner. The second contribution seems the largest. They formulate the visual reasoning problem as a cooperative task between a visual reasoner (a pre-trained LLM) and a visual searcher (a visual LLM which is called as a tool by the visual reasoner and optimized with RL for the task at hand). Since optimization in a cooperative setting is really hard, the paper chooses to fix the visual reasoner to GPT-5-mini-2025-08-07  and optimize the visual searcher (i.e.  Qw2n2.5-VL-Instruct in this paper).

On standard academic benchmarks such as V*-Bench, VisualProbe, and MME-RealWorld results demonstrate that they can adding their now-trained visual searcher to several GPT variants and Gemini Flash and obtain significant improvements on top of using those models without their visual searcher. The fact that is works on many visual reasoners shows generalization.  Additionally, their best result with GPT-5-mini and Gemini-2.5-Flash outperforms the state-of-the-art on the majority of datasets. Note that while Mini-O3 seems to be stronger, AFAIK this appeared on ArXiv only recently and should be counted as contemporary work.

### Strengths
* This paper decouples visual reasoning from visual search, leading to a modular system which is more understandable and whose components can be trained independently.
* Results are state-of-the-art.
* Good improvement using Gemini as a visual reasoner suggests that even though their visual search model was optimized on GPT-5-mini, it generalizes to multiple visual reasoners.
* Ablation shows good improvement of RL fine-tuning and training efficiency benefit of using their static RL setup in conjunction with the dynamic RL setup.
* New and complex visual reasoning dataset.

### Weaknesses
* The new dataset is rather small and limited in domain.
* Minor: Table 1 is a bit confusing: the bottom part seems the most important while the top part is hardly discussed and only used for context; I would suggest to show the bottom part either on top or maybe better shown separately as Figure 1.
* Minor: it is unclear which tools are used by Qwen2.5-VL. Anything other than 'crop'?
* Minor: specialization of agents and sub-agents in agentic frameworks has been shown to work in prior art. Examples are [Socratic Models, Zeng et al., ICLR'23] and [HAMMR, Castrejon et al., NeurIPS workshop 2024]. Would be good to cite some of these works.

### Questions
I don't really have specific questions.

### Soundness
3

### Presentation
3

### Contribution
3
