# Imagine a City: CityGenAgent for Procedural 3D City Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The automated generation of interactive 3D cities is a critical challenge with broad applications in autonomous driving, virtual reality, and embodied intelligence. While recent advances in generative models and procedural techniques have improved the realism and scalability of city generation, existing methods often struggle with high-fidelity asset creation, controllability, and manipulation. In this work, we present CityGenAgent, a natural language-driven framework based on large language models (LLMs) for hierarchical procedural generation of high-quality 3D cities. Our approach introduces two core programs—$\textbf{Block Program}$ and $\textbf{Building Program}$—which decompose city generation into interpretable and editable components. $\textbf{BlockGen}$ and $\textbf{BuildingGen}$ are trained to generate and execute these programs. We design Spatial Alignment Reward to enhance spatial reasoning and Visual Consistency Reward to bridge the gap between textual program descriptions and their 3D visual realizations. 
Additionally, benefiting from the use of programs and the model's generalization capabilities, our framework allows users to manipulate the results via natural language. Comprehensive evaluations show that CityGenAgent achieves impressive semantic alignment and higher visual quality, establishing a stronger foundation for broad applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CityGenAgent, a novel method for generating 3D cities through a natural language-based framework. The core contribution lies in developing domain-specific LLMs trained explicitly for city generation. The authors introduce two specialized programs—the Block Program for generating city blocks and the Building Program for describing individual buildings. They also construct a domain dataset for SFT and design multiple tailored rewards for RL training. Experimental results demonstrate that the proposed approach can produce high-quality 3D city layouts and exhibits strong controllability and manipulation capabilities.

### Strengths
1. Fine-tuning a pre-trained LLM for city generation is novel and interesting.
2. The design of the reward functions and the overall training objective is reasonable.
3. The proposed method achieves promising visual and structural results.

### Weaknesses
1. The abstract could be improved for clarity and conciseness. The current version is somewhat disorganized, making it difficult to grasp the core contribution. Based on the rest of the paper, the main novelty lies in developing domain-specific agents trained with SFT and RL, distinguishing this work from previous studies. The authors are encouraged to highlight this aspect more clearly in the abstract.
2. The process by which textual building descriptions are converted into 3D models is unclear. It would be helpful to specify whether the 3D assets are generated using a 3D generative model (e.g., Hunyuan3D) or retrieved from a pre-existing asset library.
3. The generation procedure for the training dataset is insufficiently described. Moreover, the authors do not commit to releasing the source code or dataset, and these materials are not provided as supplementary files. This lack of transparency makes it difficult to reproduce the proposed approach.
4. As a 3D generation study, the visual results are somewhat limited. Including video demonstrations would help validate the 3D spatial consistency and realism of the generated cities.
5. The authors could further explore alternative RL optimization strategies, such as DPO and GRPO, for further exploration.

### Questions
Will the authors release the source code and datasets?

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
This paper introduces CityGenAgent, a framework that integrates LLMs with hierarchical procedural generation techniques to create high-quality, interactive 3D city environments from natural language prompts. The core approach decomposes the complex city generation task into a structured hierarchy of specialized agents. The system seeks to improve controllability, fidelity, and manipulation capabilities of city generation.

### Strengths
The paper proposes to integrate LLMs to guide hierarchical procedural generation in the 3D domain, effectively addressing the traditional challenge of translating abstract intent into concrete geometric parameters. The results show the model can produce better 3D city scenes compared with existing methods.

### Weaknesses
1. The model efficiency is a critical weakness for a framework targeting real-world applications like large-scale city generation, which demand high efficiency. Given the reliance on commercial LLM APIs, the token consumption for a complete, complex city generation is likely prohibitive. The paper fails to provide a quantitative scaling analysis (e.g., generation time vs. area/asset count) or propose concrete technical solutions (beyond simple statements) to mitigate the LLM-related computational overhead.

2. The experimental validation focuses on the effect of certain rewards but critically omits the technical justification for the multi-agent hierarchy itself. Without an ablation study quantifying the marginal performance and token-efficiency gain of each agent division, the multi-agent design choice remains an arbitrary architectural decision rather than a proven necessity.

### Questions
1. What's the model used to create the asset?

2. Given the acknowledged limitation of inference time for large scenes, please provide quantitative scaling data: Generation time (in seconds or minutes) vs. scene complexity (measured by total area or asset count) for both CityGenAgent and baselines.

3. Will the performance of CityGenAgent be dependent on the reasoning and instruction-following capability of the underlying LLM? Have the authors experimented with different models?

4. It's encouraged to include some demos or videos to illustrate the generated 3D city scenes.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents a natural language-driven framework named CityGenAgent for hierarchical generation of high-quality 3D city models. The framework is based on large language models and decomposes the city generation process into two core programs, Block Program and Building Program, which are executed by the BlockGen and BuildingGen modules respectively. The paper designs spatial alignment reward and visual consistency reward to enhance spatial reasoning and visual fidelity, and supports interactive operations through natural language.

### Strengths
1. The paper introduces a novel hierarchical procedural generation paradigm via two domain-specific languages (Block Program and Building Program), creatively combining LLMs with structured programs as editable proxies.

2. The technical execution is robust, with clear decomposition into BlockGen/BuildingGen modules, SFT+PPO training pipeline, and reward designs grounded in computable metrics, demonstrating better semantic alignment and visual fidelity over mentioned baselines.

3. The paper is well-structured with intuitive figures, precise DSL definitions, and appendices for prompts and datasets.

4. This advances embodied AI and simulation by providing a scalable, controllable foundation for 3D urban worlds, potentially inspiring similar program-based agents in other procedural domains

### Weaknesses
1. The framework in Section 3.1 decomposes cities using Block Program and Building Program as editable DSL intermediates. However, no comparison is provided with scene graph-based 3D scene generation methods, such as in terms of layout fidelity, editing efficiency, or scalability to multi-block cities.

2. In Section 3.2.1, Block-Gen (SFT) is described as enabling the LLM to generate valid Block Programs that adhere to the schema, including non-self-intersecting polygons and required fields. However, no compliance metrics, such as program parse success rate after natural language conversion, field completeness ratio, or polygon validity, are reported, leaving the structural integrity of the DSL outputs unquantified prior to 3D execution.

3. In Section 3.2.2, Block-Gen(PPO) is introduced to improve spatial reasoning over SFT via Spatial Alignment Reward, claiming better handling of complex layouts. Yet, no ablation study compares SFT-only versus SFT+PPO on PSA or Collision Rate, making it unclear how large the performance gap is and whether PPO is necessary for core gains or merely a non-essential optimization.

4. The reliance on synthetic data generated by GPT-4o for SFT/PPO (Appendix D) limits real-world generalization; while post-processing removes low-quality samples, the dataset may not capture diverse urban styles (e.g., non-Western architectures) or corner cases like irregular terrains or imbalanced building attributes.

### Questions
1. For Spatial Alignment Reward, why choose AABBs over exact polygon overlap? Share ablation results on reward sensitivity to this approximation, and what if switching to exact methods.

2. In program execution (Section 3.4), how are assets for Building Program components sourced or synthesized when no match exists? Provide failure cases handling and success rates from your 50 evaluation prompts.

3. The intro mentions embodied intelligence applications—have you tested any integration with simulators? what adaptations are needed?

4. Manipulation (Section 3.5) claims RL enables generalization—please quantify this with metrics like edit success rate, coherence preservation, and multi-step edit chains.

### Soundness
3

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
4

### Summary
This paper, Imagine a City: CityGenAgent for Procedural 3D City Generation, proposes a natural language-driven framework, CityGenAgent, which leverages large language models (LLMs) for the hierarchical procedural generation of 3D cities. The core contribution is using an agent-based system to iteratively refine and generate complex 3D scenes based on descriptive text input, aiming to improve controllability and fidelity in automated 3D content creation.

### Strengths
1. The task of procedural, language-guided 3D city generation is interesting, addresses a crucial challenge in 3D content creation, and is of interest to the community.

2. The paper is generally well-organized and the approach is presented clearly, making the high-level idea easy for the reader to follow and understand.

### Weaknesses
1. The technical contribution appears somewhat limited. While the paper focuses heavily on the scene description and iterative refinement driven by the LLM, there is a distinct lack of detailed description regarding the actual procedural generation and manipulation of the 3D assets (e.g., buildings, road networks, and underlying geometric operations). For a 3D generation work, the mechanisms for handling 3D assets should be a core component, yet these sections lack sufficient technical detail.

2. The experimental validation is not enough to scientifically support the claims. The majority of the results presented are qualitative demonstrations, which, while visually appealing, are difficult to compare scientifically and objectively. The most crucial quantitative results are confined primarily to a single table (presumably Table 1). The quantitative evaluation is conducted on a very small sample size (only 50 samples), which significantly undermines the statistical persuasiveness and generalizability of the reported results. The comparison against existing methods appears sparse, making it difficult to properly contextualize the performance and novelty of the proposed CityGenAgent.

3. The paper does not include any discussion or data regarding the generation efficiency, computational cost, or inference time, which are critical factors for a procedural generation system, especially one leveraging large models.

### Questions
1. Please provide a more complete and detailed breakdown of the 3D generation process. Specifically, elaborate on how the refined scene descriptions are translated into concrete 3D asset instantiation, placement, and manipulation operations. This is essential for readers to understand and potentially replicate the method.

2. The experimental section requires substantial expansion. This must include: Expanding the size of the quantitative evaluation dataset significantly to ensure statistical validity;  Including more competitive and relevant baseline methods for comprehensive performance comparison; Providing quantitative analysis on the computational costs, such as inference time and memory usage, particularly in relation to the size and complexity of the generated scene.

3. To better illustrate the value of the proposed iterative agent approach, please provide a detailed experimental analysis showing the progressive improvement over the training or generation iterations (e.g., convergence curves or metric improvements over steps). This would help demonstrate why the iterative description refinement is necessary and effective.

### Soundness
3

### Presentation
2

### Contribution
3
