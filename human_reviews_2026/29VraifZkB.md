# ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation via Modular Multimodal Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Automating the transformation of user interface (UI) designs into front-end code holds significant promise for accelerating software development and democratizing design workflows. While multimodal large language models (MLLMs) can translate images to code, they often fail on complex UIs, struggling to unify visual perception, layout planning, and code synthesis within a single monolithic model, which leads to frequent perception and planning errors. To address this, we propose ScreenCoder, a modular multi-agent framework that decomposes the task into three interpretable stages: grounding, planning, and generation. By assigning these distinct responsibilities to specialized agents, our framework achieves significantly higher robustness and fidelity than end-to-end approaches. Furthermore, ScreenCoder serves as a scalable data engine, enabling us to generate high-quality image-code pairs. We use this data to fine-tune open-source MLLM via a dual-stage pipeline of supervised fine-tuning and reinforcement learning, demonstrating substantial gains in its UI generation capabilities. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in layout accuracy, structural coherence, and code correctness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents ScreenCoder, a 3 step agentic framework for generating front-end code from UI screenshots or design mockups. Based on common failure points, Screencoder are spliced into three stages, a vlm based detection stage, a rule-based planning stage, and a llm based code-generation stage. The authors also contribute a dataset and a benchmark, and show a dual‑stage post‑training pipeline that improves an open‑source vlm on the task.

### Strengths
* Breaking the problem into perceptual (vision) and logical (planning, coding) stages is a sensible and interpretable approach. 
* The method achieves high performance across multiple metrics and show clear gains over both open baselines and earlier systems. 
* By generating image-code pairs and a curated test benchmark , the authors contribute valuable datasets.

### Weaknesses
* From a research perspective, ScreenCoder is a blend of existing techniques rather than a fundamentally novel invention. DCGen, LayoutCoder and UICopilot have all used hierarchical generation heuristics, the main difference is leveraging a pretrained VLM for component detection and adding RL fine-tuning, While these yield better results, the conceptual novelty is limited
* The Planning Agent relies on fixed rules and “front-end engineering priors” (hardcoded grid templates, etc.). These heuristics, while pragmatic, feel hand-crafted and dataset-specific. It’s unclear how well they generalize to arbitrary UIs
* missing baselines: LayoutCoder

### Questions
What are the common failure cases for ScreenCoder?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper deals with transforming websites screenshots and design sketches into front end code the problems seems to be interesting in a way that in the recent times web design has been integrated more than ever into our daily lives.

### Strengths
1. The works seems to be well motivated and worked upon with basics which is good to see

2.The results look promising both qualitatively and quantitatively 

3. The paper is well written and easy to follow

### Weaknesses
1. I do not see a discussion on the competitive works relating to screenbench.

2. Some previous instances and citations in section 3 would be useful to support the claims

3. Some analysis on why certain metric is good or bad can be useful

4. What about some failure case analysis

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ScreenCoder, a modular multi-agent framework for transforming UI screenshots or design sketches into front-end HTML/CSS. The pipeline decomposes the task into three stages—grounding, planning, and generation. The authors also introduce the Screen-10K training set and ScreenBench and report automatic and human evaluations, which show improvements over baselines as well as gains from a dual-stage (SFT and RL) post-training pipeline.

### Strengths
- New dataset and benchmark: The paper presents Screen-10K (curated from 50k webpages into 10k clean pairs to stabilize training) and ScreenBench (1,000 contemporary websites emphasizing complex nested layouts).
- The paper is generally well written.

### Weaknesses
- There are already some datasets on UI code generation, which the paper did not discuss/compare with. For example:
Gui et al., VISION2UI: A Real-World Dataset with Layout for Code Generation from UI Designs, https://arxiv.org/abs/2404.06369v1, April 2024.

Hugo Laurenccon, L’eo Tronchon, and Victor Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset. 2024. URL https://api.semanticscholar.org/CorpusID:268385510. 

Especially, the VISION2UI dataset is also evaluated using MLLMs.

Furthermore, there are many UI to code generation approaches, such as Uicopilot. The paper did not compare with these related approaches.  

- This paper presents ScreenCoder as a superior alternative to monolithic, end-to-end models, but the experiments are missing a crucial comparison: a standard MLLM simply fine-tuned on the new Screen-10K dataset. This makes it difficult to tell if the performance gains come from the powerful new dataset or from the complex agentic architecture itself.

- This paper claims improvements in “code correctness”, but evaluation is almost entirely visual. This paper puts a strong emphasis on achieving pixel-perfect replication and high visual fidelity, but the evaluation did not touch on the quality of the generated code. 

- It is not clear why the three agents can solve the recurring failures of MLLMs.

- Human study uses n=6 annotators/participants with limited trials and no inter-rater agreement metrics reported (Appendix A), affecting reliability of the evaluation.

- More technical details should be provided. For example, Reward is −MSE between screenshot and rendered page (Eq. (3)–(4)), but the renderer, resolution normalization, anti-aliasing, and determinism are not specified. GRPO hyperparameters are not described. It is unclear how credit assignment is stabilized given a non-differentiable render step—no discussion of variance reduction beyond GRPO (Sec. 5), impacting training stability.

- Algorithm 1's rule "if l contains subdivisions then ..." is informal—criteria for "contains subdivisions" are not defined, reducing algorithmic completeness.

- Public availability of ScreenBench is not specified, limiting external verification.

### Questions
- How is this work compared with the related work listed in Weaknesses section?

- Why can the three agents solve the recurring failures of MLLMs?

- To disentangle the effects of the agentic architecture from the Screen-10K dataset, could you provide results for a crucial baseline where the same base model (Qwen2.5-VL) is fine-tuned end-to-end on Screen-10K? This would help clarify the true architectural contribution.

- Given the paper's claim of achieving "code correctness", have you considered evaluating the generated code with non-visual metrics that are important to developers?

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
The paper proposes ScreenCoder, a modular, three-agent pipeline (grounding → planning → generation) for visual/design-to-code generation, targeting both screenshots and low-fidelity sketches. On top of the pipeline, the authors use it as a data engine to construct Screen-10K and a new evaluation benchmark ScreenBench, and they further do SFT + RL with a pixel-similarity reward to improve an open MLLM. Experiments on ScreenBench and Design2Code show noticeable gains over strong MLLM baselines, and the qualitative figures are compelling.

### Strengths
+ Clear problem framing and modularization. The paper gives a concrete analysis of two failure modes of current MLLMs on UI-to-code (perception vs. planning) and maps them 1:1 to the three agents, which makes the overall story quite interpretable and also explains why “one big model” often fails in practice. The method section is readable and the pipeline could plausibly be reimplemented.
+ Stronger evaluation setting. Introducing ScreenBench (1k, more contemporary, more structurally complex) is useful, because many existing sets are small/dated and over-emphasize text reproduction; reporting both agentic and finetuned variants is also helpful to separate “inference-time pipeline” vs “model-level gains”.
+ Empirical gains over a broad set of baselines. On the reported metrics, the agentic ScreenCoder is consistently at the top or second-best vs. both open and closed models, which is non-trivial given the strong baselines (GPT-4o, Gemini-2.5, Qwen2.5-VL, Websight, DCGen).

### Weaknesses
- Limited novelty relative to existing agentic image-to-code pipelines. The main idea—splitting the visual-to-code process into grounding, layout planning, and code generation, each handled by an MLLM—is conceptually similar to several recent agentic or divide-and-conquer image-to-code systems [1, 2]. Those works have already argued that monolithic MLLMs conflate perception and layout reasoning and proposed staged workflows for webpage reconstruction. The contribution currently appears to rest on using three agents and turning the pipeline into a data generator. The inclusion of an RL-based post-training step is interesting, yet since inference still relies on prompt-based agents, the paper should clarify why a prompt-driven controller is preferable to a trained (e.g., RL-finetuned) agent for the task.

- The “pixel-perfect” claim is under-supported, especially for sketches. Fig.1 visually suggests that both high-fidelity screenshots and low-fidelity design sketches can be turned into “pixel-perfect” pages. But Sec. 4.4 explains that the final fidelity step relies on UIED to crop visual assets from the original screenshot and map them back. It remains unclear how missing high-resolution assets are obtained when the input is a low-res or line-style sketch like in Fig.1: from where are the missing high-res assets obtained, and what happens when the background is already composited with many elements so that cropping a clean patch is impossible?

- Data-engine contribution is not convincingly evidenced. Sec. 5 presents Screen-10K as a key contribution: the agent is used to generate cleaner, more aligned image–code pairs; these are then used for SFT and RL, and the model improves. However, the current experiments never show the crucial comparison: a model trained/fine-tuned on real human/web data only vs. a model trained on Screen-10K (or Screen-10K + a small amount of real data). What we see is only an internal ablation (+SFT → +RL) within the synthetic dataset, so it is impossible to tell whether synthetic/agent-generated data is actually better than well-filtered real webpages, which is the central motivation of Sec. 5. This makes the conclusion “our method generates more effective training data than real data” too strong; at best the current results show that given this particular synthetic distribution, RL on a pixel-similarity reward helps on two benchmarks.

References:
[1] Divide-and-conquer: Generating ui code from screenshots
[2] LaTCoder: Converting Webpage Design to Code with Layout-as-Thought

### Questions
1.When the input screenshot or sketch contains a complex background with many overlapping elements, how does the system extract clean, high-quality visual assets to achieve the claimed “pixel-perfect” reconstruction?

2.Can the authors include additional experiments to directly compare models trained on Screen-10K with those trained on real or mixed web data, in order to validate the claim that the generated synthetic data are more effective?

### Soundness
2

### Presentation
3

### Contribution
2
