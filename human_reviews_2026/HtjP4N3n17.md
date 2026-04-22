# SIDiffAgent: Self-Improving Diffusion Agent

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Text-to-image diffusion models have revolutionized generative AI, enabling high-quality and photorealistic image synthesis. However, their practical deployment remains hindered by several limitations: sensitivity to prompt phrasing, ambiguity in semantic interpretation (e.g., ``mouse" as animal vs. a computer peripheral), artifacts such as distorted anatomy, and the need for carefully engineered input prompts. Existing methods often require additional training and offer limited controllability, restricting their adaptability in real-world applications. We introduce Self-Improving Diffusion Agent (SIDiffAgent), a training-free agentic framework that leverages the Qwen family of models (Qwen-VL, Qwen-Image, Qwen-Edit, Qwen-Embedding) to address these challenges.SIDiffAgent autonomously manages prompt engineering, detects and corrects poor generations, and performs fine-grained artifact removal, yielding more reliable and consistent outputs. It further incorporates iterative self-improvement by storing a memory of previous experiences in a database. This database of past experiences is then used to inject prompt-based guidance at each stage of the agentic pipeline. SIDiffAgent achieved an average VQA score of 0.884 on GenAIBench, significantly outperforming open-source, proprietary models and agentic methods. We will publicly release our code upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a training-free, multi-agent framework that enhances text-to-image diffusion models by integrating prompt refinement, adaptive negative prompting, image evaluation, and image editing. The system includes three coordinated agents: (1) a generation orchestrator, (2) an evaluation agent, and (3) a guidance agent that collaborate to iteratively enhance image quality and text alignment by learning from past successes and/or failures. Experiments on GenAIBench and DrawBench demonstrate that SIDiffAgent achieves state-of-the-art results, outperforming both proprietary models (e.g., Imagen) and open-source models (e.g., Stable Diffusion).

### Strengths
1. This paper presents a clear and modular agent design that operates entirely at inference time, allowing different components to be easily swapped for newer/stronger models without training/finetuning.
2. This paper shows strong empirical gains with stepwise ablations that clearly demonstrate the contribution of each module to the end performance.

### Weaknesses
1. The method is heavily dependent on the Qwen ecosystem, raising questions about its portability to other model families. It is also unclear whether components drawn from different ecosystems (e.g., mixing a non-Qwen image model with Qwen-based evaluation or editing modules) would integrate effectively.
2. There is no fair comparison in terms of computational cost. The agentic workflow clearly requires significantly more computation per image due to repeated generation, evaluation, and editing steps. Without reporting runtime, memory consumption, or FLOPs, the reader cannot assess the practical efficiency of the method. The paper should include a quality-compute tradeoff analysis to clarify the extent of the overhead introduced by this multi-stage process.

### Questions
1. Were the agentic workflow and baselines evaluated on metrics for aesthetic appeal or image fidelity? If not, can the authors clarify why other evaluation metrics were not considered?

### Soundness
3

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
This paper introduces SIDiffAgent, a training-free, multi-agent framework designed to enhance text-to-image diffusion models. The framework coordinates multiple Qwen-based sub-agents—each specializing in tasks such as intent interpretation, prompt refinement,  adaptive negative prompting, generation, evaluation, and guidance—to collaboratively improve semantic alignment and visual fidelity. A key contribution is the use of Qwen-Image-Edit for structured local editing, which enables targeted corrections while avoiding the randomness inherent in full-image regeneration. Furthermore, the system incorporates a self-improving memory mechanism that learns from the successes and failures of its submodules to guide future generation processes. Extensive experiments on GenAI-Bench and DrawBench demonstrate that SIDiffAgent surpasses the performance of existing open-source and several proprietary models.

### Strengths
1.**Originality of the Proposed Memory System:**  The introduction of  Theory-of-Mind-inspired self-improving memory system is novel and represents a direction that has been rarely explored in diffusion models.

2.**Intuitive and Effective Framework:** The proposed generate → evaluate → edit paradigm is intuitive, simple, and empirically effective, which I find to be one of the most convincing aspects of this work.

### Weaknesses
1.**Limited Novelty:** The proposed local editing after image generation is intuitively effective but lacks clear academic novelty or theoretical depth.
Its practical applicability is also limited, since Qwen-Image/Edit itself is computationally expensive, making the overall framework inefficient for real-world deployment.

2.**Insufficient Ablation Studies:** The paper lacks key ablations to isolate the contribution of each major component, particularly the memory module and local editing mechanism.
Without quantitative evidence of their individual impact, the reliability of the memory design is weakened.

3.**Insufficient Generalization Analysis:** The generalization capability of the proposed framework is not convincingly demonstrated.
All experiments are conducted exclusively on the Qwen family of models, while no evaluation is provided on other comparable backbones such as Flux-Dev or Kontext, which limits the framework’s claimed general applicability.

4.**Lack of Efficiency and Multi-Dimensional Evaluation:** The paper does not provide any analysis of runtime efficiency or computational overhead. Moreover, it omits evaluations on other critical text-to-image dimensions—such as object counting, color consistency, and spatial accuracy—as used in benchmarks like GenEval, which weakens the overall evaluation reliability.

### Questions
I have summarized the issues and questions in the weaknesses above. To summarize.
1. Provide more detailed ablation and effectiveness/limitation analyses for the memory module and the local editing mechanism, along with additional qualitative explanations.

2. Please include experimental results on GenEval, covering key dimensions such as object counting, color consistency, and spatial alignment.

3. Please add evaluation results on other diffusion backbones, such as Flux-Dev and Kontext, to better demonstrate the framework’s generalization capability. 

4. Provide an latency analysis, to clarify the practical feasibility of the proposed method.

I hope the questions and weaknesses raised can be addressed. I will reconsider my score accordingly.

### Soundness
3

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
4

### Summary
The authors present a plug-ang-play multi-agent pipelines to improve diffusion generation. It basically have a set of generation agents (including prompt optimizers and generators), an evaluation agent that can determine if we need to do further editing after the initial generation, and a guidance agent that maintain a database storing prior succesful and failure generation results. The mluti-agent pipeline improves the performance on the base Qwen-Image model and outperforms multiple single-round open-source and proprietary generation models.

### Strengths
1. The paper is clear and easy to follow.
2. The plug-and-play multi-agent system improves the performance a lot on GenAI-bench and the DrawBench.
3. The idea of introducing databases for image correction is interesting.

### Weaknesses
1. Novelty and Baselines
	- While the paper is technically sound with performance gain, the idea of plug-and-play pipelines for diffusion generation is not a new idea. For instance, [1, 2] proves that using LLM object planning can already fix negative prompts and improve prompting a lot in a multi-round fashion. [3] then extends this idea with VLM modules, which is close to the paper's agentic setting already. I believe that these papers worth discussions and even be the baseline in Table 1. The reviewer understands it can be hard to do apple-to-apple comparison, but as they're all important prior plug-and-play methods, being a line in a Table is important.
	- Probably the only part that makes the paper novel is the database. However, the authors did not ablate on the design choice on this. However, the authors fixed it to top-5 retrieval and 200 trajectories. It's also unclear how the integration of this database improves the performance. 

2. Experimental Setup
	- As this work involves multiple components, it would be interesting to see how robust the system is if we change at least one component. It would be interseting to see if we can pair a non-Qwen generator to other Qwen models and vice versa at least to prove the generalizability. 
	- The multi-round, multi-agent pipeline would incur a lot of computational overhead. While the author mentioned it in the limitation section, the reviewer believes this should be discussed and mentioned in the main paper at least. Also, it would be good to analyze failure cases of this multi-agent collaboration scenario, perhaps following [4] (the authors have already mentioned this paper in the paper).

[1] Lian, Long, et al. "Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models." TLDR.

[2] Wu, Tsung-Han, et al. "Self-correcting llm-controlled diffusion models." CVPR 2024.

[3] Wang, Zhenyu, et al. "Genartist: Multimodal llm as an agent for unified image generation and editing." NeurIPS 2024.

[4] Cemri, Mert, et al. "Why do multi-agent llm systems fail?." arXiv 2025.

### Questions
Please read the weakness part. Additionally, some citations in the paper doesn't seem to be correct in terms of the format (L179-L180).

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
3

### Summary
This paper proposes SIDiffAgent, a training-free multi-agent framework designed to improve text-to-image diffusion models through autonomous prompt engineering, adaptive error correction, and iterative self-improvement. SIDiffAgent coordinates a set of that are specialized in creativity analysis, intention parsing, prompt refinement, adaptive negative prompting, image generation, aesthetic and alignment evaluation, and memory-driven guidance.
The system leverages the Qwen family of models and a Theory-of-Mind–inspired feedback mechanism. Experiments on GenAI-Bench and DrawBench show improvements over open-source and proprietary baselines. The system demonstrates improved text-image alignment, artifact mitigation, and compositional reasoning without training.

### Strengths
- SIDiffAgent improves iteratively at inference via trajectory memory. The design extends prior works such as T2I-Copilot by introducing a multi-layer agent hierarchy and adaptive negative prompt generation. 
- The empirical performance gains of SIDiffAgent over open-source and proprietary baselines seems promising.
- Implementation details (such as the prompts of the sub agents, algorithms and hyperparameters) are provided in the appendix.

### Weaknesses
- Marginal algorithmic novelty:
The paper’s contribution lies primarily in system integration rather than introducing a fundamentally new self-adapting optimization algorithm for diffusion models.
- System complexity and reproducibility:
The multi-agent framework involves many interdependent agents and prompts. The training-free claim is valid, but the inference-time across multiple agent calls can be computationally heavy and difficult to replicate. 
- Evaluation: The claiming of perceptual alignment improvements are made qualitatively without human preference study, and there is no analysis of latency or cost trade-offs between the introduced computational cost versus the gains in the performance.

### Questions
- What is the average inference time per generation cycle compared to T2I-Copilot or plain Qwen-Image?
- Can you provide more abaltion study isolateing the effect of negative prompting, and memory guidance, e.g. what fraction of the performance gain can be attributed specifically to the guidance agent versus adaptive negative prompting?
- Have you done any evaluation on the retrieval quality of the guidance agent, e.g. is the top-5 good enough for each new prompt?
- Have you tested the generalization of SIDiffAgent’s memory beyond prompts seen in Episode 1 (e.g., on an unseen dataset)?

### Soundness
3

### Presentation
3

### Contribution
3
