# CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Despite significant progress, multimodal large language models continue to struggle with visual mathematical problem solving.
Some recent works recognize that visual perception is a bottleneck in visual mathematical reasoning, but their solutions are limited to improving the extraction and interpretation of visual inputs.
Notably, they all ignore the key issue of whether the extracted visual cues are faithfully integrated and properly utilized in subsequent reasoning.
Motivated by this, we present CogFlow, a novel cognitive-inspired three-stage framework that incorporates a knowledge internalization stage, explicitly simulating the hierarchical flow of human reasoning: perception$\Rightarrow$internalization$\Rightarrow$reasoning.
In line with this hierarchical flow, we holistically enhance all its stages.
We devise Synergistic Visual Rewards to boost perception capabilities in parametric and semantic spaces, jointly improving visual information extraction from symbols and diagrams.
To guarantee faithful integration of extracted visual cues into subsequent reasoning, we introduce a Knowledge Internalization Reward model in the internalization stage, bridging perception and reasoning.
Moreover, we design a Visual-Gated Policy Optimization algorithm to further enforce the reasoning is grounded with the visual knowledge, preventing models seeking shortcuts that appear coherent but are visually ungrounded reasoning chains.
Moreover, we contribute a new dataset MathCog for model training, which contains samples with over 120K high-quality perception-reasoning aligned annotations.
Comprehensive experiments and analysis on commonly used visual mathematical reasoning benchmarks validate the superiority of the proposed CogFlow.
Project page: https://shchen233.github.io/cogflow.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose CogFlow, a VLM framework that represents images through geometric primitives before performing reasoning, and introduce MathCog, a new dataset annotated with these primitive-level representations. In CogFlow, visual primitives (e.g., points, lines, circles) are learned to be both geometrically accurate (correct coordinates) and semantically consistent with the original image. During training, primitive predictions with low visual score are regenerated using a thresholding mechanism, termed VGPO. At inference time, only similarity-based supervision is applied since ground-truth primitives are unavailable. To further enhance reasoning accuracy, the authors train a logical error reward that identifies inconsistencies in reasoning steps, which is then used to fine-tune the model to improve consistency.

### Strengths
1. **Strong results.** Achieves significant performance gains over comparable open-source baselines, showing consistent improvements across multiple benchmarks. Notably, the proposed method narrows the gap with, and in some cases surpasses, the performance of leading closed-source models. 
2. **Clear presentation.** The writing is clear and organized, with helpful figures. Despite the many nuances of the proposed approach, the motivation and details are clearly presented (though some additional details may be missing, see weaknesses).
3. **Substantive contributions.** Introducing a new reasoning framework around improving VLM perception (CogFlow) and a complementary dataset (MathCog) to support the framework. This method and dataset, which results in strong perfomance, is a valuable addition to the field.

### Weaknesses
1. **Insufficient dataset details and transparency.** The description of the MathCog dataset (in the appendix) lacks important specifics about the data collection and annotation process. It remains unclear exactly what tasks human annotators performed, the scale of human effort involved, and the associated cost or time investment. Additionally, the process for generating negative samples should be more explicitly explained (e.g. prompts used). 
2. **Ambiguity in source of performance improvements.**
It is difficult to disentangle the contribution of the CogFlow framework from that of the MathCog dataset. Since CogFlow is trained using MathCog, the performance gains might partially stem from the dataset's scope rather than the model's method. A useful comparison would be to evaluate other methods when trained on MathCog to better understand how much improvement arises from the dataset versus the proposed architecture. While Table 4 (with Table 1/2) gives some sense of this improvement, it is unclear how much improvement other methods would achieve given the same dataset.
3. **Misalignment with cognitive science framing.**
While the paper draws inspiration from cognitive science concepts, the mapping between these ideas and the proposed components is not entirely convincing. For instance, the "watching" tokens more closely resemble internalization (structured encoding of visual input) than perception. Similarly, the Visual Assessment Reward (VAR) appears to operate across multiple stages of cognition, including perception (error type 1) and reasoning (error types 3 and 4), rather than targeting a specific cognitive process. This partial misalignment makes the method more confusing, though the link can still provide good motivation.
4. **Missing experiments.** Missing comparison with SophiaVL-R1 [1], which may have stronger performance. Missing ablations for individual SVR components (VPR and VSR) (or alpha hyperparameter L248), as well as evaluations of how good FG-CLIP is for similarity reward.

**Suggestions:**

1. Spelling (L26 "grated"->"gated", L27 "that produce appear"->"that appear"). Fig 3 "Visual Similarity Reward" -> "Visual Semantic Reward" and Visual Parameter Reward -> parameterized.
2. Lots of acronyms (SVR, VAR, VGPO, VPR, VSR) makes method a bit confusing, and some acronyms aren't very descriptive (e.g. synergistic visual reward).
3. Equation 4 does not follow the algorithm as described. As described (L292), k=2,3 is not evaluated if k=1 passes the threshold, but equation 4 implies 2 and 3 are generated and checked (maximized). 
4. Equation 3 should include preferred trajectory (s+) in the denominator sum, otherwise it's not softmax. Also Softmax-DPO should be better cited L272 [2]
5. Figure 2 is unclear whether the supervised fine-tuning (SFT) stage uses the "watching" and new "thinking" annotations, as the yellow/green outline suggests watching/thinking are only used during RL.
6. Expand one-sentence figure/table captions.

[1] Fan, Kaixuan, et al. "SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward." arXiv preprint arXiv:2505.17018 (2025).

[2] Chen, Yuxin, et al. "On softmax direct preference optimization for recommendation." Advances in Neural Information Processing Systems 37 (2024): 27463-27489.

### Questions
1. In Figure 7, why does the Baseline+SVRs setting show limited reduction in perception errors, with most gains coming from reducing reasoning errors?
2. In Table 4, do all evaluations use the visual gate during inference? What happens if the model is trained without VGPO but still uses the gate at inference?
Do all evaluations in Table 4 use visual gate at inference? And what is the impact of training without VGPO but using gate at inference?
3. In Figure 2, what do the easy/medium/hard categories refer to?

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
The paper presents CogFlow, a cognitively inspired framework for visual mathematical reasoning that models the process through three stages: perception, internalization, and reasoning. It introduces stage-specific visual rewards to strengthen the coupling between visual understanding and symbolic reasoning. A new dataset, MATHCOG, is constructed to support training and evaluation. Experiments on FlowVerse, MathVerse, and MathVista show consistent improvements, demonstrating the effectiveness of multi-stage perceptual guidance.

### Strengths
1. The paper presents a conceptually clear and cognitively inspired framework that divides multimodal reasoning into perception, internalization, and reasoning stages, effectively bridging visual understanding and symbolic inference.

2. The paper demonstrates a well-organized methodological design, where the layered reward mechanisms are technically coherent and systematically integrated across different reasoning stages.

3. The paper show consistent improvements on three visual mathematical benchmarks, and contributes a new dataset, MATHCOG, that holds potential value for future research.

### Weaknesses
1. The paper mainly reports results on MathVista, where the gains are substantial, but lacks systematic comparisons with other recent benchmarks such as MathVision, We-Math, and DynaMath. This limits the generality of the claimed performance advantage.

2. The paper evaluates only a single model scale (Qwen2.5-VL-7B) without testing different sizes or architectures, making it difficult to assess the scalability and broader applicability of the proposed framework.

3. The paper provides limited analysis of why the multi-stage reward structure improves reasoning performance. The overall explanation remains empirical, and deeper insights into the mechanism behind these gains would strengthen the work's interpretability.

### Questions
The author should include additional experimental results, particularly on other recent benchmarks such as MathVision, We-Math, and DynaMath. Since the performance improvement on MathVista is exceptionally large, results on these datasets are important to verify the generality and robustness of the proposed method.

### Soundness
3

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
The paper argues that the existing vision-language reasoners lack high-quality perception and their reasoning drifts away from their perception in their chain-of-thought. To address these issues, the paper proposes a Cogflow framework that encourages the model to internalize the perceptual primitives before it starts solving the problem. Specifically, the paper proposes visual rewards to ensure that perceptual accuracy in parameter space and layout consistency. This acts as a gate to prevent low-quality generations from unrolling and training rewards for RL training. Further, the paper proposes visual anchored rewards which encourages the model to internalize the perception in its chain of thoughts through synthetically generated negative pairs for softmax-DPO. Finally, they propose visual-gated policy optimization (VGPO) as a global optimization of the model policy. As a result, the model starts performing very well on the FlowVerse and MathVerse datasets. The paper also studies the impact of different design choices in ablation studies.

### Strengths
1. The paper addresses an important problem of the lack of perceptual understanding and reasoning drift in VLMs. 

2. The proposed method is quite comprehensive where each component ensures that the model learns to perceive images accurately at the primitive level and the model is encouraged to utilize perception via additional rewards. 

3. The experimental results suggest that the method works very well on the FlowVerse, MathVerse, and MathVista dataset. The ablation studies showcase the usefulness of different components on final performance.

### Weaknesses
1. Since the paper has too many moving parts, the paper could be written better. Despite decent experience in VL reasoning, I was finding it hard to keep up with a dense introduction with many jargons. The authors should think about how to portray their story in a more simplified manner.

2. The proposed method to get the primitives for shapes (e.g., Circle) does not seem scalable. How will you get primitives for shapes where the actual dimensions are absent like many natural scenes or even synthetic scenes (like CLEVR dataset)? There is very little information about collecting “watching” data in the main text. Overall, the solution is more tailored towards geometry than general-purpose visual reasoning. It is critical to acknowledge that early in the paper to set the expectations right. 

3. The evaluation is somewhat limited. It would be nicer to show performance on more evaluation datasets such as LogicVista, MathVision, We-Math, MMMU-Pro, HallusionBench. Currently, my hunch is that the method will suffer on very complex scenes from LogicVista because there are many images within an image and getting those many primitives might break the model.

### Questions
mentioned above

### Soundness
4

### Presentation
2

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
This paper proposes a novel cognitive-inspired three-stage framework (perception-internalization-reasoning) to enable faithful integration and proper utilization of visual signals for visual mathmatical reasoning.  The authors proposes synergistic vision reward (SVR) consisting of visual parameterized reward and the visual semantic reward and incoporate the reward in the original GRPO algorithm to devise the VRPO algorithm. The author proposes a new visual anchor reward and curates constrastive training examples to mitigate five common types of unfaithful ultilization of visual signals.

### Strengths
+ The paper proposes a new three-step framework for faithful visual mathematical reasoning.
+ It constructs a new dataset MATHCOG featured with three subset with each subset for a stage respectively.
+ Experiments on FlowVerse dataset and MathVerse dataset show substaintial improvement compared with baseline methods.

### Weaknesses
+ Some important details are missing. For example, how to construct the contrastive pairs in MATHCOG-VAR dataset? How to incorporate the five common types of reasoning error? How is the visual anchor reward $R_{VAR}$ implemented? Is it a reward in VRPO algorithm or is it a training stage between SFT and VRPO? How iis the correctness and format reward $R_{IR}$ implemented? How to compute the visual parameterized reward in the parameter space if the predicted primitive and the ground-truth primitive is not within the same class? (i.e., the ground truth is an ellipse while the predicted primitive is a circle). 

+ As in Eq4, the generation of perception output is repeated $k$ times until the SVR reward is large than a fixed threshold. It is eqivalent to best-of-k sampling if the threshold is not met in early attempts. Therefore, it is doubtful whether the comparison with baseline method is fair or not.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3
