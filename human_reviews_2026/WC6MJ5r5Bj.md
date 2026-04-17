# ReCAPA: Hierarchical Predictive Correction to Mitigate Cascading Failures

- Decision: Accept (Poster)
- Scores: 4, 8, 8

## Abstract
Vision–Language–Action (VLA) systems follow instructions to execute multi-step tasks in multimodal environments. Recent VLA approaches typically rely on post-hoc correction mechanisms or operate under fixed task decompositions and alignment schemes. However, once an intermediate step is mis-specified, local errors propagate through subsequent steps and eventually accumulate into cascading failures. To mitigate this compounding effect, we propose Predictive Alignment and Planning Architecture (ReCAPA), a framework that uses prediction and contrast to adjust deviations across three levels: actions, subgoals, and trajectories. Semantic alignment is enforced at all levels using a Sinkhorn-based module and a Score-field module. The predictive correction and alignment, jointly updates the action-generator in the training phase, enabling it to adjust fine-grained steps to remain aligned with the overall intent. We further introduce two new metrics to quantify error propagation and recovery processes in tasks, capturing how mistakes spread and fade over long-horizon execution.  Experiments show that ReCAPA achieves competitive results on embodied agent benchmarks such as VisualAgentBench, MineDojo, and AI2-THOR, outperforming strong proprietary and open-source Large Language Model (LLM) baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ReCAPA (Reflective Contrastive Alignment and Planning Architecture), a hierarchical predictive correction framework designed to mitigate cascading failures in long-horizon reasoning for vision–language–action (VLA) agents. Unlike prior methods that rely on fixed task decomposition or post-hoc correction, ReCAPA proactively anticipates and corrects deviations across three hierarchical levels—actions, subgoals, and trajectories—using predictive contrastive learning and prompt–trajectory alignment modules based on Sinkhorn optimal transport and score-field gradients. The authors also introduce two diagnostic metrics, Error Propagation Rate (EPR) and Propagation Attenuation Coefficient (PAC), to evaluate how errors accumulate and dissipate during execution. Experiments on VisualAgentBench, MineDojo, and MAP-THOR demonstrate that ReCAPA achieves higher success rates and superior robustness compared to strong LLM baselines, effectively reducing error propagation in multi-step embodied tasks.

### Strengths
The paper tackles an important and underexplored issue—cascading failures in long-horizon reasoning—and proposes a technically sound hierarchical correction framework. The introduction of predictive alignment across multiple levels and the new diagnostic metrics (EPR and PAC) provide useful analytical tools for evaluating robustness in embodied agents. Experiments are comprehensive, spanning several major benchmarks, and the reported improvements are consistent across tasks.

### Weaknesses
1. Conceptual novelty is moderate. The proposed hierarchical predictive correction (HPCC) framework largely builds upon existing ideas in self-reflective planning (e.g., Reflexion, ReAct, AdaPlanner) and hierarchical alignment (e.g., HiP, TrajPrompt). The distinction between ReCAPA’s “predictive correction” and prior feedback-based reflection mechanisms is not sharply articulated. The paper would benefit from a clearer theoretical or algorithmic differentiation beyond combining multi-level prediction and Sinkhorn-based alignment.

2. Many components—such as how cross-level corrective gradients interact with the execution network or how the LLM’s decomposition is integrated during inference—are described only qualitatively. Key design choices (e.g., the dimensionality of embeddings, predictor architectures, and training stability) are omitted, making the method hard to reproduce or verify.

3. The paper emphasizes “hierarchical correction,” yet no qualitative rollout visualizations or case studies are shown to illustrate how errors are detected and corrected in practice. Examples demonstrating the evolution of alignment or prediction across levels would make the mechanism more convincing.

4. The new EPR and PAC metrics are interesting but lack correlation studies with task performance or user interpretability. It remains unclear whether improvements in these metrics genuinely indicate better planning robustness or simply reflect model-specific biases.

5. Writing and organization issues. The text is dense and occasionally repetitive, with unclear boundaries between related work and method sections. Some equations (e.g., Eq. 1–4) are introduced abruptly without sufficient intuition. The overall presentation would benefit from clearer motivation and more concise explanation of the technical pipeline.

### Questions
The proposed EPR and PAC metrics are intriguing, but how sensitive are they to task complexity or trajectory length? Have the authors verified that improvements in these metrics correlate with human-judged robustness or actual task success rates?

I also notice some typo of writing:

1. The appendix title: “Appendix C.3 ABLATION ON PREDICTION AND” seems incomplete.

2. The first paragraph of Related Work has repeat sentence.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ReCAPA, a predictive correction framework designed to mitigate cascading failures in multi-step reasoning in the context of VLA models. The main ides is that small errors in subgoal or action specification can compound over time, degrading overall performance.
ReCAPA addresses this by applying predictive correction mechanisms at multiple levels such as actions, subgoals, and trajectories with the aim of anticipating and correcting deviations before they propagate.
The proposed method demonstrates strong performance, reportedly outperforming both proprietary and open-source large language models on benchmark tasks.

### Strengths
- The paper is clear, well-structured, and easy to follow, with an interesting and well-motivated idea.
- The method is explained clearly, with a logical progression from motivation to implementation.

### Weaknesses
- The discussion of limitations lack details. While two limitations are mentioned, the proposed mitigation strategies are not empirically validated, which reads as somewhat unbalanced.
- The statistical significance of results is unclear. Although the authors mention using three random seeds on VisualAgentBench (Fig. 4), it is not evident how consistent or significant the improvements are across other experiments.
- The reproducibility of the results is limited: the authors do not provide the code, and the training details are not discussed.
- The ablation studies are limited.

### Questions
- Why did the authors not further investigate the proposed mitigation strategies for ReCAPA’s limitations?
- Could the authors include or discuss ablation analyses to clarify the contribution of each module?
- Can the authors elaborate on the statistical significance of their results beyond the limited VisualAgentBench trials?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper focuses on the problem of proactive error prediction and corrections for VLA models. The method, Reflective Contractive Alignment and Planning Architecture (ReCAPA) incorporates a bottom-up approach for deviation detection, and a top-down approach for deviation correction. ReCAPA aims to tackle the challenges of both short-term action error correction as well as long-term plan adherence. The hierarchical framework breaks down an embodied task trajectory into action, subgoal, and trajectory levels. At high level, the model aims to achieve prompt-trajectory distributional alignment through the Sinkhorn-based module. At the action and subgoal level, the model aims to achieve fine-grained prediction and execution alignment through the Score-fieldSong module. The model goes through 2 training stages: a pre-training to align state-action sequences, and a joint training of hierarchical model through a combined loss of Optimal Transport between the task prompt embedding and the overall trajectory distribution, a denoting local objective to pull fine-grained actions and subgoals closer to the prompt semantic intent, and a contrastive corrective loss between each adjacent levels (e.g.. action-subgoal, subgoal-trajectory). The paper also proposed two new evaluation methods, error propagation rage (EPR), and propagation attenuation coefficient (PAC) to explicitly measure the error propagation and cascading degree during the long horizon process. The paper conducted experiments on 3 benchmarks, and compared aganist numerous open sourced and proprietary LLMs, and demonstrated improved overall success rate as well as some EPR and PAC curves. The paper also conducted ablation studies to evaluate the importance of each component.

### Strengths
- The paper is well motivated and well structured. It studies the problem of action deviation from both short term mistake correction and longer term task alignment two objectives.
- The paper proposed an interesting method to decompose the trajectories into hierarchical 3 levels, and align the predicted states with prompts for deviation detection and correction.
- The paper offers sufficient and concise explanations behind the method, loss function designs, and acknowledge the key limitations in the conclusion section while offering future directions. 
- The paper conducted thorough experimentations against multiple benchmarks and baseline models, offering sufficient evidence to support the advantages of the proposed method.

### Weaknesses
1. Table 1: it would be helpful to explain how 'transport rate, coverage, and balance' these three metrics are calculated
2. Questions below

### Questions
1. When an inconsistencies arises at state $S_i$, with action $a_i$, ($S_i \rightarrow a_i$), and the model learns and updates its weight, does it learn a 'correction action' from the current state ($S_i \rightarrow a_i \rightarrow a_j \rightarrow S_j$) or a hopefully better action $a_j$ from the same state $S_i$, ($S_i \rightarrow a_j$)?

2. Section 3.3.2: the prompt embeddings $p$ -- are these the same prompts as in Section 3.3.1 $v$ for the overall task, or are they action/substep specific prompts? 

3. Ln 257: how to generate negative state-action sequences from GPT-4o-mini? 

4. If a model performs consistently well throughout a task, will the PAC score be high or low? 
3. Ln 114-115: duplicates

### Soundness
3

### Presentation
3

### Contribution
4
