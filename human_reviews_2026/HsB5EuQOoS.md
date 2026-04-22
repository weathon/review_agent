# Procedural Mistake Detection via Action Effect Modeling

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Mistake detection in procedural tasks is essential for building intelligent systems that support learning and task execution. Existing approaches primarily analyze how an action is performed, while overlooking what it produces, i.e., the \textbf{action effect}. Yet many errors manifest not in the execution itself but in the resulting outcome, such as an unintended object state or incorrect spatial arrangement. To address this gap, we propose Action Effect Modeling (AEM), a unified framework that jointly captures action execution and its outcomes through a probabilistic formulation. AEM first identifies the outcome of an action by selecting the most informative effect frame based on semantic relevance and visual quality. It then extracts complementary cues from visual grounding and symbolic scene graphs, aligning them in a shared latent space to form robust effect-aware representations. To detect mistakes, we further design a prompt-based detector that incorporates task-specific prompts and aligns each action segment with its intended execution semantics. Our approach achieves state-of-the-art performance on the EgoPER and CaptainCook4D benchmarks under the challenging one-class classification (OCC) setting. These results demonstrate that modeling both execution and outcome yields more reliable mistake detection, and highlight the potential of effect-aware representations to benefit a broader range of downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Procedural Mistake Detection is important for supporting learning and task execution. This paper proposed Action Effect Modeling (AEM) to capture not only the action execution but also its outcome.
The AEM module consists of:
1) Effect frame sampling: Select the most informative effect frame based on semantic score and quality score.
2) Multimodal knowledge extraction: By leveraging the foundation model, Grounding DINO, for object detection to get the object states and relations. Then, use GPT-4o to generate the scene graph and decompose into a state subgraph and a relation subgraph. This process generate extracts the complementary cues.
3) Effect-aware learning: To avoid querying multimodal large language models during inference, a learnable effect token (EFT) captures task-specific action effects through self-attention so that during inference time, the framework depends only on this learned token.

AEM achieves state of the art performance in 2 benchmarks, showcasing the importance of modeling both execution and outcome in mistake detection.

### Strengths
The main strength in this paper is to provide a in-depth ablation studies to show that 1) AEM is effective and generalizable, 2) spatial relationships provide more consistent and discriminative cues for action correctness, 3) Visual features are more effective, 4) cross-modal alignment is important, 5) Simple last frame selection is not good enough, need to incorporate semantic relevance and visual clarity, 6) expressive action segment representations are important downstream mistake detection.

This ablation study is very thorough, and unlike the ablation studies in most paper, this paper yields new interesting "general" insights that may be transferable to other domains beyond action effect modeling. Additionally, the implementation details are clear in the paper including models and compute budgets. 

The figures are well-made for understanding the method.

### Weaknesses
1. The mistake detection result is determined by a predefined threshold. This poses a major problem, if you change the predefined threshold, the results will be greatly affected? On the other hand, if you try different values of the threshold to maximize the performance, it is essentially "tuning" on the test set.
2. The modular approach, despite effectiveness, it may introduce problems like longer processing times, extra engineering, additional training.
3. No baselines of strong end-to-end reasoning models like OpenAI o3, o4 and GPT-5.
4. The idea of modeling both action and effect is not very novel as it has been studied in many earlier papers with different focuses (multimodal action effect prediction, state change in videos, action-conditioned world models) [1,2,3].

References:
[1] Dagan et al. Learning the Effects of Physical Actions in a Multi-modal Environment. 2023.
[2] Souček et al. Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos. 2022
[3] Hu et al. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations. 2024

### Questions
1. What is the predefined threshold used, as mentioned in the Weakness 1?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript presents Action Effect Modeling (AEM), a probabilistic framework that jointly captures the execution of an action and its resulting environmental state, arguing that many procedural errors are only detectable in the outcome rather than the motion itself. AEM first selects an effect‑rich frame using semantic relevance and visual quality cues, then fuses visual grounding with symbolic scene‑graph embeddings into a shared latent space to form an effect‑aware representation; a prompt‑based one‑class classifier is subsequently applied to detect mistakes. Evaluated on EgoPER and CaptainCook4D, AEM outperforms existing methods under the one‑class setting, demonstrating that integrating action outcomes yields more reliable error detection and provides a useful representation for downstream tasks.

### Strengths
* The paper identifies a genuine gap in current procedural‑mistake detection: most methods examine only how an action is performed, ignoring the _outcome_ that actually indicates an error.
* Explicitly modeling the effect of an action is a natural extension of procedural AI and can be transferred to domains such as industrial assembly, robotics, or medical procedure guidance.
* By jointly modeling execution and outcome through the proposed Action Effect Modeling (AEM) framework, the authors offer a conceptually novel solution that blends visual grounding, symbolic scene‑graph reasoning, and prompt‑based one‑class detection.
* The learnable effect token distills effect information into the action representation without requiring external LLMs at inference time, keeping the system lightweight.
* The prompt‑based detector aligns each action segment with a task‑specific textual prompt, explicitly modeling temporal dynamics and semantic context.
* AEM’s two‑branch multimodal supervision (visual + symbolic) is elegant and leverages large models to extract complementary cues.
* Reported AUC/EDA improvements  are convincing, and ablations support the claimed benefits of effect‑aware representations and cross‑modal alignment.
* Extensive ablation studies (effect‑frame sampling, state vs relation supervision, visual vs textual signals, cross‑modal alignment, dynamic fusion, prompt tuning) provide strong evidence for each design choice.

### Weaknesses
* In the overall training objective (Eq. (10) $L = L^{seg} + L^{eff} + L^{CL} + L^{det}$), the authors simply sum the four loss terms without any weighting or scaling factors. I am unsure whether the authors verified that these terms are in comparable magnitude; if not, a large loss could dominate the optimization and bias the learned representations. A brief ablation or sensitivity study showing the relative scales of the losses (or the inclusion of trainable weighting coefficients) would help the reader understand how the optimization balances execution modeling, effect learning, cross‑modal alignment, and detection.
* The manuscript does not provide a systematic hyper‑parameter study. None of the hyper‑parameters (learning rate, batch size, loss weights, temperature, etc.) are tuned or validated on a validation set. A brief ablation or sensitivity analysis, especially on the relative loss weights in Eq. (10) and on the temperature coefficient in the contrastive objectives, would strengthen confidence that the proposed method is robust and not overly sensitive to arbitrary hyper‑parameter choices.
* Heavy reliance on external large models for supervision is not quantified; downstream failure rates of the grounding or scene‑graph modules could propagate to the effect token but are not analyzed.
* No statistical significance tests or confidence intervals are reported, so the robustness of the reported gains is unclear.
* Evaluation is limited to two egocentric cooking datasets; generalization to other procedural domains remains untested, one of which is cited by the authors (Assembly101).
* Minor typographical and formatting inconsistencies—such as missing spaces and content overflowing in Table 2—reduce the document’s overall readability.

### Questions
Please refer to the section on weaknesses.

### Soundness
3

### Presentation
2

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
The paper studied procedural mistake detection and incorporates the effect of actions alongside their execution, which seems to be original. To do this, it introduces Action Effect Modeling (AEM), a probabilistic framework that jointly models action execution, and the resulting outcomes (the hardest part of the paper to understand), and automatic frame selection. Evaluations on egocentric video benchmarks demonstrate that AEM outperforms prior state-of-the-art methods in both action and outcome-based mistake recognition. The ablation study is thorough, assessing multiple important modeling questions.

### Strengths
- S1 The main methodological contribution that incorporates both the action execution and its effect into a probabilistic framework is original in the procedural mistake detection space. 

- S2 The paper builds on the growing literature in procedural mistake detection, enabling measured comparisons by using common pipeline steps from recent works.

- S3 The paper includes two contemporary and popular procedural mistake detection benchmarks in its evaluation.  Generally the proposed work achieves quick strong improvements over the state of the art.  The experiments also include ablation analysis over many key factors in the modeling, demosntrating robust performance.

- S4 The effect frame sampling is very interesting.  It's a shame the paper had to rush through its discussion.  It seems that this notion of "bayesian" frame selection may have a broader potential in problems like this.

### Weaknesses
- W1  The prompt based detector is not well explained, not does it seem to be properly analyzed in the results.  Yet, it is claimed to be a primary contribution.  Is this alignment not needed in general for procedural mistake detection methods?  This part of the paper is very unclear.  And, considering its importance in the overall paper, this significanlty detracts from the quality of the paper.
  
- W2 The paper seems heavily dependent on GPT4o for numerous functionality.  Notwithstanding the fact that GPT4o is a closed model and hence hard to directly analyze, one of the key points of the paper is the relational graph that is 

- W3 The paper is wrought with exposition terseness, small mistakes, and nonuniform level of description across various parts, making it harder to follow than necessary.
  - W3a `action effects` seems to be defined in various ways throughout the paper, always implicitly.  It would be very helpful to more clearly define action effect.  Furthermore, how many action effects are studied?  Is it just the two that are ablated?  Do each of these have a set of features associated with them?  This part of the paper is far from declarative and seriously distracts from the contributions.
  - W3b Isn't the first term in equation 1 $P(\hat{y}|\mathbf{X},e_i,f_e)$?  Any notion of independent of the effect frame is not obvious.
  - W3c The paper has some unfortunate grammatical errors.  EG L077 "from vision-language model" ?  from a? models?  Typographical errors.  EG L430 "clarityfurther". And, there are also some seemingly incorrect sentence structures.  L126 "However, mistake detection in procedural videos is more goal-oriented" ... mistake-detection is more goal oriented or the problem of procedural video modeling is more goal oriented? 

- W4 Given the dependence on numerous pretrained (and sometimes closed) models, it is unfortunate the paper does not analyze the generalizability and robustness to alternatives.  This weakens the impact of the work.

### Questions
- Q1 What text-visual embedding model is used for the effect frame sampling?
- Q2 How are the models actually learned?  The complexity of the marginal and sampling across frames of the video seems like the joint learning (which seems to be proposed) is very "big."  What is happening here really?
- Q3 Until section 4.4 it is not clear that the action segmentation method is also learned (or fine-tuned) as part of the process.  This is unfortunate, since the action segmentation does seem to impact the results.  One wonders how the other methods would perform using these segments (i.e., is the gain from better segments or is it from action effect modeling).

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes AEM for procedural mistake detection, modeling both action execution and effects through effect frame selection, multimodal features (visual and symbolic), and a contrastive prompt detector. It achieves better on EgoPER and CaptainCook4D under OCC.

### Strengths
Strong empirical results on two benchmarks, with ablations validating key components.

### Weaknesses
1. Novelty is limited, as the integration of scene graphs and VLMs builds on existing work (e.g., Hurst et al., 2024) and offers little beyond fusion.
2. Benchmarks are narrow (only two datasets) and lack evaluation across diverse domains, such as assembly or medical procedures, as mentioned in the introduction.
3. The prompt-based detector seems straightforward, and gains over baselines like ProtoMD are modest in some metrics.
4. The methodology sections are dense, with equations (e.g., Eq. 1) that need more intuitive explanations.
5. Some sections (e.g., the methodology) could benefit from additional pseudocode or flowcharts to improve clarity.

### Questions
1. How does AEM handle actions with delayed or invisible effects (e.g., internal changes in objects)? Could AEM be extended to online detection without full segment access?
2. Could you compare computational overhead to simpler execution-only methods?
3. Why only two effect descriptors (states and relations)? Are there plans to extend to more?
4. How robust is the effect of frame sampling to noisy or occluded videos?
5. What is the impact of VLM choice (e.g., alternatives to Hurst et al., 2024) on performance?

### Soundness
3

### Presentation
3

### Contribution
2
