# Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 6

## Abstract
While Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in image and video understanding, their ability to comprehend the physical world has become an increasingly important research focus. Despite their improvements, current MLLMs struggle significantly with high-level physics reasoning. 
In this work, we investigate the first step of physical reasoning, i.e., **intuitive physics understanding**, revealing substantial limitations in understanding the dynamics of continuum objects. 
To isolate and evaluate this specific capability, we introduce two fundamental benchmark tasks: Next Frame Selection (NFS) and Temporal Coherence Verification (TCV). Our experiments demonstrate that even state-of-the-art MLLMs perform poorly on these foundational tasks. 
To address this limitation, we propose Scene Dynamic Field (SDF), a concise approach that leverages physics simulators within a multi-task fine-tuning framework. 
SDF substantially improves performance, achieving up to $20.7\%$ gains on fluid tasks while showing strong generalization to unseen physical domains. This work not only highlights a critical gap in current MLLMs but also presents a promising cost-efficient approach for developing more physically grounded MLLMs. Our code and data are available at https://github.com/andylinx/Scene-Dynamic-Field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical limitation in MLLMs: their poor understanding of intuitive physics, particularly for continuum objects like fluids. The authors introduce two diagnostic tasks—Next Frame Selection (NFS) and Temporal Coherence Verification (TCV)—to systematically evaluate low-level physical perception. They reveal that even state-of-the-art MLLMs perform poorly on these tasks, often near random baselines. To bridge this gap, the authors propose Scene Dynamic Field (SDF), a method that leverages physics simulators to generate visual prompts representing motion dynamics. Through multi-task fine-tuning, SDF significantly improves model performance and generalizes well to unseen physical domains like cloth and smoke.

### Strengths
1. The paper is interesting for addressing the problem of poor low-level physical perception by proposing practical physics simulators for visual prompting to improve performance.

2. The paper is well-organized and accessible, with clear explanations of both the problem and the solution.

### Weaknesses
1. Limited Scope of Transfer Experiments: While the transfer experiments across continuum domains (cloth, sand, smoke, plasticine) are promising, the scope remains limited. The paper would be significantly strengthened by evaluating the method's generalization to other fundamental physical phenomena, such as rigid-body dynamics, collisions, or optical effects. 

2. Experimental Design and Baselines: To provide a more comprehensive performance comparison, we suggest expanding Table 1 to include additional baseline models. Specifically, it would be informative to compare against larger models with more parameters.

3. Dependence on Synthetic Data and Real-World Applicability: The SDF method's reliance on synthetic data from simulators is a potential limitation, as such data may not fully capture the complexity and noise of real-world physical systems. The paper could be improved by discussing the feasibility and potential challenges of applying this method to real-world physics problems. For instance, how would the SDF approach perform with real sensor data that is often incomplete or noisy?

4. Analysis of MLLM Limitations: The paper identifies that MLLMs struggle with low-level dynamics but does not deeply investigate the root cause. A more thorough analysis is needed to determine whether these failures stem from architectural limitations (e.g., an inductive bias towards high-level semantics) or a bias in the pre-training data (e.g., a lack of low-level physical reasoning examples). Uncovering this would provide valuable insight for future research.

### Questions
See weaknesses. I am willing to discuss with the authors

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes two low level tests for intuitive physics in MLLMs, Next Frame Selection and Temporal Coherence Verification, focused mainly on fluids. It introduces Scene Dynamic Field, a simulator derived motion visualization used as a visual prompt within a multi task fine tuning scheme, and reports sizable gains on the proposed tests with some transfer to cloth, sand, smoke, and plasticine.

### Strengths
- Clear problem decomposition toward low level dynamics rather than high level QA
- Simple intermediate representation that is easy to plug into existing MLLMs
- Ablations on stride, prompts, model scale, and expert vs self distilled data
- Some transfer beyond fluids and an attention analysis that supports the claim that SDF shifts focus to earlier frames

### Weaknesses
- The weakest point is that there is no comparison to strong motion baselines. SDF is a velocity magnitude style visual prompt, which is conceptually close to optical flow magnitude, flow stacks, dynamic images, or even simple frame differencing. Without head to head baselines under the same training data and budget, the gains could come from adding any explicit motion cue rather than from SDF itself. This leaves the central claim unproven.

- Benchmarks are author designed and multiple choice, so improvements may reflect distractor design rather than genuine physics understanding

- Absolute accuracy remains low, so practical impact is unclear

- Limited evaluation beyond fluids for rigid body scenes or causal reasoning tasks, so the title and claims feel broader than what is shown
- Distractor pruning uses SigLIP embeddings which are related to encoders used by evaluated models, creating a risk of bias

My recommendation is reject. The idea is interesting and the empirical gains are clear on the authors benchmark, but the evaluation misses strong motion baselines, relies on potentially biased distractor construction, and the absolute performance and scope do not yet support the paper’s broad claims.

### Questions
- How does SDF compare to simple optical flow overlays, grayscale motion magnitude, or event frame stacks when training with the same budget?

- Are results robust when distractors are generated with a feature space disjoint from any model under test?

- Can you evaluate on external physics benchmarks without re curating the data to validate generality?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper focuses on the task of fluid and physical scene understanding. It makes two primary contributions:

1. The authors build a comprehensive benchmark for physical reasoning from videos, introducing two key tasks — Next Frame Selection (NFS) and Temporal Coherence Verification (TCV). The dataset is constructed from multiple sources, including ContPhy, PhysBench, and real-world video clips, to cover both synthetic and natural dynamics.

2. The paper proposes the Scene Dynamic Field, a framework that leverages physical simulation principles and integrates Chain-of-Thought (CoT) reasoning with explicit physics modeling. This approach enables multimodal models to reason about scene dynamics beyond static appearance, bridging the gap between perception and physical understanding.

### Strengths
The paper tackles an important and challenging problem — physical and fluid scene understanding — which is a key step toward enabling models to reason about real-world dynamics.

The proposed explicit physical reasoning task is interesting and well-motivated, encouraging models to go beyond visual perception and engage in physically grounded understanding.

The paper is generally well-written and structured, with a clear motivation and logical flow from problem definition to methodology and experiments.

### Weaknesses
1. Limited physical understanding of only fluid dynamics. The proposed framework primarily focuses on liquid and fluid dynamics, which narrows the generality of the method. Extending the benchmark and tasks to cover a broader range of physical interactions — such as rigid-body motion, collision, or elastic deformation — would strengthen the overall contribution.

2. Few missing references to physical reasoning VQA. The paper could benefit from citing and discussing more recent physical understanding and VQA-related research, such as Comphy (https://arxiv.org/abs/2205.01089) and DynSuperCLEVR (https://arxiv.org/abs/2406.00622), which address the VQA for physical reasoning.

### Questions
1.  Data Modality and Encoder Usage:
What is the exact data format of the SDF samples used for training? From Figure 3, the SDF appears to be an RGB-like image generated through velocity-to-color mapping. Could the authors clarify whether these SDF images are indeed three-channel RGB inputs, and if they are processed by the same image encoder as the regular video frames during training?

2. Ablation on the SDF Step in CoT Fine-Tuning. In the SDF-guided Chain-of-Thought fine-tuning process, the framework first predicts the SDF representation and then predicts the next frame based on that SDF. What would happen if we retain the same training pipeline but skip the explicit SDF step, i.e., directly fine-tune the model with simplified CoT instructions to predict the RGB frame without generating the SDF?
This comparison would help clarify how much the explicit SDF stage contributes to physical reasoning, especially since the SDF image appears visually similar to the original frame except for color-coded fluid regions (as shown in Figure 3). Such an ablation could reveal whether the SDF acts mainly as a visual prompt or as a truly distinct physical representation.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper try to addresses the gap in MLLMs understanding of intuitive physics, particularly for continuum objects like fluids. The authors first introduce two low-level benchmark tasks, next frame selection and temporal coherence verification, to demonstrate that current MLLMs perform poorly at perceiving physical dynamics. To solve this, they propose SDF, an intermediate representation generated by physics simulators that visually encodes motion (e.g., velocity as color intensity). Through a multi-task fine-tuning strategy, their SDF-enhanced model achieves substantial gains on fluid tasks and shows strong generalization to unseen physical domains like cloth and sand.

### Strengths
- The motivation of the paper correctly focuses on a simpler, core problem, which is just perceiving physical motion, separating it from complex, high-level reasoning.

- Using a visual map (SDF) from a physics simulator to train the model works. This helps the model learn the idea of dynamics, letting it generalize from fluids to unseen materials like cloth or sand.

- The authors built a strong benchmark by mixing simulated data with real-world videos, and they ran thorough experiments to validate their approach.

- The paper is well presented with clear logic.

### Weaknesses
- The SDF representation is very basic, encoding just the projected velocity magnitude into one color channel. It's questionable if this simple map captures enough information for complex interactions, or if it needs to include more data like 3D vector direction or using optical flow (which avoids physics simulation). A discussion comparing the performance, advantages, and disadvantages of these different representations would greatly strengthen the analysis.

- The pipeline is complex. It requires a full-parameter fine-tuning process, data from simulators, and knowledge distilled from expert models, which is computationally expensive and hard to scale.

### Questions
See in the weekness.

### Soundness
3

### Presentation
3

### Contribution
3
