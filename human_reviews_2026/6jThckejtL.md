# JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Generative models often treat continuous data and discrete events as separate processes, creating a gap in modeling complex systems where they interact synchronously. To bridge this gap, we introduce $\textbf{JointDiff}$, a novel diffusion framework designed to unify these two processes by simultaneously generating continuous spatio-temporal data and synchronous discrete events. We demonstrate its efficacy in the sports domain by simultaneously modeling multi-agent trajectories and key possession events. This joint modeling is validated with non-controllable generation and two novel controllable generation scenarios: $\textit{weak-possessor-guidance}$, which offers flexible semantic control over game dynamics through a simple list of intended ball possessors, and $\textit{text-guidance}$, which enables fine-grained, language-driven generation. To enable the conditioning with these guidance signals, we introduce $\textbf{CrossGuid}$, an effective conditioning operation for multi-agent domains. We also share a new unified sports benchmark enhanced with textual descriptions for soccer and football datasets. JointDiff achieves state-of-the-art performance, demonstrating that joint modeling is crucial for building realistic and controllable generative models for interactive systems. [Project](https://guillem-cf.github.io/JointDiff/)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper describes generating multi-agent trajectories in a sports scene, ensuring that the spatio-temporal trajectories of players and the ball are plausible. Motivated by the success of the recent diffusion-based generative framework, the paper proposes combining a discrete representation (ball possession) with a continuous representation (spatial coordinates of individual trajectories). Additionally, the framework can provide additional guidance of user intent, composed of text guidance or weak processor guidance. The results are presented with soccer and football datasets.

### Strengths
- The work proposes an important yet challenging problem, which generates controllable multi-agent trajectories. Theoretically, the framework combines the continuous and discrete representations in a consistent way and can be extended to many other large-scale control systems.

### Weaknesses
- Figures are hard to interpret. All the texts are too small, and the terminology is not clearly defined. Things to be in a clear layout with large enough text whose sizes are comparable to the main paper.

- The exposition in Section 4.3 is weaker compared to other parts of the paper. It is hard to see how different components are put together, and the limited clarity of the figure does not help in understanding the structure. I suggest adding a simplified version of Figure 4 in the main paper.

### Questions
Minor comment
- temporal-evolving -> temporally evolving (line 051)
- Why do you call the works non-generative in line 092?

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
4

### Summary
The paper propose Jointdiff, a diffusion-based framework tailored for trajectory generation in ball games. Jointdiff includes social-temporal block that models continuous multi-agent trajectory and discrete events in a single denoising process. Controllable generation is enabled via Crossguid, which injects external guidance signal represented by . The guidance could be weak-possessor-guidance or text descriptions. Jointdiff is evaluated with three types of ball games-- NBA, NFL and Bundesliga. Results shows Jointdiff achieves SOTA with respect to future prediction and imputation.

### Strengths
1. Principled diffusion-based joint modeling of continuous and discrete modalities. 
2. Controllable scene generation enable interesting applications, for example, it could simulates the outcome of different tactics, to make better decisions.
3. Evaluation across three types of sports
4. Supplementary materials includes many details about text guidance dataset generation, which could be useful for constructing text dataset for other use cases.
5. Model architecture graph is very clear.

### Weaknesses
1. The main paper presents a broadly applicable diffusion architecture but offers limited methodological novelty; the contribution reads more application-driven than algorithmic. Key elements that could strengthen the perceived innovation—e.g., human evaluation design, text-dataset construction, and other data curation details—are largely relegated to the supplementary material. As a result, the main paper feels incremental and under-substantiated. The authors should write more these components into the main text, clearly describe what is genuinely new versus adapted, and include ablations that isolate architectural contributions from application engineering.

2. The aligned training method of text and trajectory is not clear.

3. The related-work discussion is thin and misses several foundational lines (e.g., NRI, dNRI, GRIN)[1,2] and key domain papers (e.g., TacticAI)[3], which weakens the paper’s positioning and novelty claim; at minimum, the authors should (i) contextualize how JointDiff differs conceptually, and (ii) add targeted comparisons/ablations or a rationale for incompatibilities.

[1Kipf, T., Fetaya, E., Wang, K. C., Welling, M., & Zemel, R. (2018, July). Neural relational inference for interacting systems. In International conference on machine learning (pp. 2688-2697). Pmlr.

[2]Li, L., Yao, J., Wenliang, L., He, T., Xiao, T., Yan, J., ... & Zhang, Z. (2021). Grin: Generative relation and intention network for multi-agent trajectory prediction. Advances in Neural Information Processing Systems, 34, 27107-27118.

[3]Wang, Z., Veličković, P., Hennes, D., Tomašev, N., Prince, L., Kaisers, M., ... & Tuyls, K. (2024). TacticAI: an AI assistant for football tactics. Nature communications, 15(1), 1906.

### Questions
1. How exactly is text guidance trained? Please specify the text encoder (e.g., frozen vs. finetuned), how its embeddings are injected (e.g., CrossGuid placement, keys/values/queries), and whether you use classifier-free guidance during training (drop-condition rate, loss formulation).
2. Do you align text to trajectories/events with any auxiliary objectives (e.g., contrastive losses) or just condition the denoiser? Any ablations on encoder choice and guidance strength?
3. Please report the full compute profile: GPU model/count, batch size.
4. How controllable is the generation process? How is the sampling efficiency?
5. How to deal with uncertainties in future generation? e.g., their might be multiple possible future scenarios given the same historical trajectory?
6. Can the model be used to review the middle of a match? For example, a tactical switch or a player substitution?
7. In your opinion, is there a foundation model that can model the trajectories of all ball games?

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
3

### Summary
The paper proposes JointDiff, a diffusion-based generative model for multi-agent trajectories in sports. This method jointly models the continuous trajectories and discrete events in multi-agent systems, enhancing the generation quality. The authors also introduce two new controllable generation tasks: weak-possessor-guidance (WPG), which conditions on an ordered list of possessors, and "text-guidance," which conditions on natural language descriptions. To facilitate this, they propose an architectural module, CrossGuid, a cross-attention mechanism that injects the guidance signal into the diffusion model's denoising network. The authors validate their model on multiple datasets and release a new sports benchmark with enhanced text annotations. JointDiff demonstrates state-of-the-art results on completion/imputation tasks and demonstrating effective control in their new tasks. A human study is also provided, which shows a perceptual preference for the joint model's generations.

### Strengths
1. JointDiff is the first work to introduce joint continuos&discrete diffusion n modeling to time-evolving multi-agent systems. The authors also proposed new guidance methods for controllable generation and corresponding architecture (CrossGuid). In addition, a new sport benchmark is introduced. 

2. The evaluation on sport datasets is comprehensive. Three public sports datasets, NBA, NFL, and BundesLiga are covered, along with two tasks, completion and controllable generation. Human study is also provided.

### Weaknesses
1. A major concern is about the novelty of the proposed method. As stated in the related work, a couple of prior work in other domains have introduced joint diffusion modeling of continuous and discrete data. Can the authors further illustrate how this work distinguish with them in addition to different task settings?

2. The CrossGuid module seems like a standard cross-attention block engineered for the sport multi-agent tasks.

### Questions
1. What is the main contribution beyond the two prediction heads design and the cross-attention gating? Why is this different from prior joint continuous–discrete diffusion work?

2. How does the model ensure consistency during generation? Since it learns to predict $Y_0$ and $E_0$ from two separate heads, is it possible for it to generate a final $(Y_0, E_0)$ pair that is inconsistent? Does this happen in practice?

### Soundness
2

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
This paper introduces JointDiff, a diffusion framework for multi-agent trajectory generation in sports. The framework is designed to unify continuous and discrete diffusion processes to simultaneously generate continuous spatiotemporal trajectories and discrete events (e.g., possession of ball). The paper presents two controllable generation scenarios (weak-possessor-guidance and text-guidance). The paper reports strong performance on scene-level metrics.

### Strengths
The paper's core hypothesis—that joint modeling of continuous trajectories and discrete events is beneficial—is validated on the domains of interest, as evaluated by scene-level metrics (Table 1).  The paper also successfully demonstrates two forms of semantic control (WPG and text-guidance -- shown in Table 2).  The paper also contributes a new text-annotated benchmark for the NFL and BundesLiga datasets, which enables the text-guidance task.

### Weaknesses
The authors claim this is the "first to introduce this joint modeling paradigm to the domain of time-evolving multi-agent systems". This claim, while true in its most narrow interpretation, overlooks a substantial and directly relevant body of un-cited prior art in adjacent fields like robotics and autonomous driving, where joint discrete-continuous diffusion models are already an active area of research for dynamic planning and generation.  (The authors do cite MotionDiffuser.)

I would strongly encourage the authors to engage with the broader literature in these adjacent fields:
https://arxiv.org/abs/2509.21983
https://sites.google.com/view/dimsam-tamp
https://generative-skill-chaining.github.io/
https://arxiv.org/abs/2509.20109
https://arxiv.org/abs/2405.15677
https://arxiv.org/abs/2402.11502
My goal in listing these papers is not to say that any of them are doing exactly what the authors are doing.  But in characterizing the specific novelty of this paper, it is important to situate it in this broader context.  

The paper's architecture appears potentially over-engineered, consisting of a complex stack of modules (Temporal Mamba, Social Transformers, and CrossGuid) built upon the U2Diff architecture. The authors list limitations such as the "architecture requires events to share the same spatio-temporal structure as the trajectory data", but to me this type of limitation is a by-product of the engineering design.

### Questions
I would like the authors to consider the broader landscape of related work, and then clarify what they view as the core novelty of this paper.

### Soundness
3

### Presentation
3

### Contribution
2
