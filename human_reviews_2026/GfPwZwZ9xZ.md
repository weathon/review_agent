# VLASim: World Modelling via VLM-Directed Abstraction and Simulation from a Single Image

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Generative video models, a leading approach to world modeling, face fundamental limitations. They often violate physical and logical rules, lack interactivity, and operate as opaque black boxes ill-suited for building structured, queryable worlds. To overcome these challenges, we propose a new paradigm focused on distilling an image caption pair into a tractable, abstract representation optimized for simulation. We introduce VLASim, a framework where a Vision-Language Model (VLM) acts as an intelligent agent to orchestrate this process. The VLM autonomously constructs a grounded (2D or 3D) scene representation by selecting from a suite of vision tools, and co-dependently chooses a compatible physics simulator (e.g., rigid body, fluid) to act upon it. Furthermore, VLASim can infer latent dynamics from the static scene to predict plausible future states. Our experiments show that this combination of intelligent abstraction and adaptive simulation results in a versatile world model capable of producing higher-quality simulations across a wider range of dynamic scenarios than prior approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes VLASim, which utilize a VLM as an intelligent agent for distilling a single image into attractable, abstract representation optimized for simulation. It is made of intelligent abstraction, adaptive simulation, and inferred dynamics. Experiments show that VLASim avoids physical artifacts of pixel-prediction models and excels at tasks requiring precise, rule-based reasoning, which provides high-quality simulations across a wider range of dynamic scenarios.

### Strengths
1. Paper is easy-to-follow. 
2. It is a critical problem of physical interaction in the simulation of world model and build an explicit, structured world representation, in which VLM acts to construct a grounded representation.
3. The idea of combine VLM and real-world simulator is new with explicit representations.

### Weaknesses
1. As real scenes are abstracted by simulator, it may be hard to contain all details in the real world. Tiny details may contain influences for real world applications, which may be ignored by the simulator.
2. It seems some foundation models like Gemini Perception, VGGT are utilized for abstraction. This should be illustrated in method overview for clearer expression. Generation Prompt in Figure2 is too general with overlooked details.
3.  It is hard to understand Figure 9. Experiments are too rough and the details of some values cannot be seen. Besides, only comparing with video generation models is not enough considering the mechanism used in your approaches. It seems VLASim is worse than Wan2.2 in Fluid Dynamics and Thermodynamics. More comparisons with simulators are required. 
4. Lack ablations to verify the effectiveness.
5. More visualizations and examples are required to show your design.

Minor issue:
Incorrect citation format at line 37-38 in abstract.

### Questions
1. Why is VLASim worse than Wan2.2 in Fluid Dynamics and Thermodynamics? Please explain the results.

2. The ablations are missing.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the problems of physical logic violations and lack of interactivity in generative video models for world modeling. It also points out the limitations of 3D scene reconstruction methods, which struggle to adapt to abstract 2D environments and cannot support physical simulation. The paper proposes the VLASim framework to address these challenges.

VLASim uses a visual language model as its core intelligent agent. It autonomously selects visual tools to construct abstract representations of 2D or 3D scenes, matches them with compatible physical simulators, and infers potential dynamics from static scenes, achieving structured world modeling.

Experiments show that VLASim performs excellently in the PhysicsIQ benchmark's physical phenomenon simulation and Conway's Game of Life rule reasoning tasks, generating physically reasonable and logically accurate simulation results, outperforming mainstream generative video models.

### Strengths
This paper abandons the approach of directly predicting pixels using generative video models, and shifts to a new paradigm of VLM-guided abstraction + adaptive simulation, fundamentally solving the core problems of physical violations and lack of interactivity in pixel-space models. It also overcomes the limitations of 3D reconstruction methods, which cannot adapt to abstract scenes and lack physical simulation capabilities.

VLASim can autonomously adapt to both 2D and 3D scenes. The generated structured programs allow users to modify actions to explore diverse future states. Compared to methods with fixed simulation types, it has a wider range of applicable scenarios and higher interactive value.

### Weaknesses
As a composite system, VLASim's simulation quality heavily depends on the output of perception tools such as segmentation and 3D reconstruction. If these tools misidentify object shapes or 3D positions, VLM will generate semantically incorrect world programs, and VLM currently lacks a mechanism to correct these errors.

The inference time for generating a complete world program from a single image and text prompt is approximately 10 minutes, far exceeding the inference speed of mainstream generative video models, making it difficult to meet the application requirements of real-time or low-latency scenarios.

Comparisons with Veo3 only include a limited number of examples; a full-scale evaluation was not conducted due to cost considerations. Furthermore, PhysicsIQ's IoU metrics cannot fully capture physical plausibility (such as non-physical phenomena like object fusion), potentially underestimating VLASim's advantages over baseline models.

Existing experiments primarily focus on single-physical-phenomenon scenarios (such as single rigid bodies or single fluids), failing to demonstrate VLASim's performance in complex multi-object interaction scenarios (such as nested rigid body collisions or multi-fluid mixing), resulting in insufficient verification of its generalization capabilities.

### Questions
Regarding simulation issues caused by errors in upstream perception tools, are there plans to design a "tool output verification" mechanism for VLM in the future? For example, could VLM determine the rationality of tool outputs through cross-validation with multiple tools (such as matching segmentation results with 3D point clouds), or adjust the abstraction strategy when tools malfunction?

Is the main bottleneck of the 10-minute inference time the process of VLM generating the world program, or the computational time consumed by perception tools such as 3D reconstruction and segmentation? Are there specific optimization schemes (such as model quantization and tool acceleration) to reduce latency?

VLASim performs excellently in scenarios with clear rules, such as Conway's Game of Life, but how can the model ensure the correctness of simulation logic in scenarios with ambiguous rules (such as non-Newtonian fluid flow)? Is it necessary to introduce domain knowledge or additional rule inputs?

The paper mentions that users can modify actions to explore diverse futures. Are there limitations to the currently supported "custom actions" (such as only supporting the application of simple forces, or being able to adjust physical parameters such as the friction coefficient)? Are there plans to support interaction methods that describe actions in natural language in the future?

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
This paper proposes VLASim, a novel world modeling paradigm that shifts from black-box video prediction to explicit, simulation-based program synthesis . VLASim uses a Vision-Language Model (VLM) as an orchestrating agent that, from a single image and text description, generates an executable Python program representing the world . The VLM intelligently selects from a suite of perception tools (e.g., segmentation, 3D reconstruction) to build a tractable 2D or 3D abstract scene representation . It then adaptively chooses a compatible physics engine (e.g., rigid body, fluid, or 2D logic) and writes the simulator code to act upon this representation . The framework includes a VLM-based critic loop for self-correction . Experiments show that this approach produces more physically plausible and logically sound simulations (e.g., in Conway's Game of Life) than state-of-the-art video generation models .

### Strengths
1. The paper's core contribution is a significant conceptual shift from implicit pixel prediction to explicit, interpretable world program synthesis . This is a major strength, as the resulting model is structured, queryable, and interactive by design, overcoming limitations of opaque video models.

2. The VLM-driven approach is highly versatile. It can intelligently select the appropriate dimensionality (2D vs. 3D) and the correct physics (e.g., rigid body, fluid, or even abstract logic) for a given scene, rather than being a one-size-fits-all model.

3. The framework truly shines in rule-based, deterministic environments. The experiment on Conway's Game of Life, where VLASim achieves a perfect F1 score by inferring the rules while the SOTA video model fails, is a powerful demonstration of this paradigm's advantage.

### Weaknesses
1. The system is critically dependent on the accuracy of its upstream perception toolbox . The paper admits that failures in segmentation or 3D estimation lead to semantically incorrect simulations. The ablation "No API" confirms this is a single point of failure, as the VLM alone cannot ground its reasoning .

2. The claim of modeling from a "single image" is slightly overstated, as the VLM relies heavily on the text prompt (e.g., "The platform rotates clockwise...")  to infer all dynamics. This suggests the VLM is not inferring latent dynamics from the static image itself, but rather translating the text description into a simulation.

3. The examples shown are relatively simple (a few objects) . It is unclear how this agent-based, multi-step synthesis process scales to complex, real-world scenes. The 10-minute inference time to generate a single world program is also high.

4. The paper's main quantitative results on the PhysicsIQ benchmark show VLASim is only "on par" with the open-source baseline Wan 2.2. The authors should qualitatively argue that the benchmark's metrics are flawed and fail to capture the physical plausibility  where VLASim excels . This is a weak empirical position.

### Questions
1. If you provide an ambiguous image (e.g., the dominos) with no text description, what latent dynamics, if any, does the VLM infer? Can it predict that the dominos will fall, or is the text prompt doing all the work of describing the dynamics?

2. The paper states the VLM "has no mechanism to question or correct a faulty tool output". Can the critic-refinement loop correct perception errors (e.g., a failed segmentation) or only code errors?

3. The 10-minute inference time is very high. What is the primary bottleneck?

4. How does the VLM "choose" the compatible physics simulator? Is this an explicit reasoning step (e.g., "This scene needs a rigid body solver"), or does it simply write Python code using libraries like PyBullet because they were listed as "Available libraries" in the prompt?

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
The paper introduces a new pipeline, named VLASim that leverages VLM for world modeling. By autonomously selecting multiple vision tools, the VLM agent constructs the scenes with explicit 2D/3D representations and generates Python program for world simulation. Thanks to the tractable abstract representation, the proposed pipeline can simulate more plausible futures with user-defined actions. It is benchmarked against some existing video generation models on PhysicsIQ and Conway's Game of Life.

### Strengths
S1) The paper presents a brand-new pipeline, which is novel compared to the common practice for world modeling. I believe its technique will provide some inspirations to the community.

S2) The effectiveness and robustness of the proposed framework are demonstrated in several scenarios.

S3) The writing is easy to follow.

### Weaknesses
W1) The paper argues that previous works are typically limited to simple transformations. However, existing world models for decision making (e.g., DriveDreamer and IRASim) do provide fine-grained action controls. On the other hand, the simulation pipeline in this paper doesn't show its richness in action controls and seems to be still limited in transformations.

W2) My main concern is the abstract representation may lose scalability in representing physics and other real-world properties. The proposed pipeline completely discard texture and background information and exclude visual metrics in its experiments. However, I do believe a scalable general and comprehensive representation is necessary for scalable real-world applications.

W3) Apparently, there is a huge sim2real gap in the simulation results. Also, from the supplementary, the supported scenarios are relatively simple and mostly static tabletop cases.

W4) The highly modular pipeline may also introduce accumulated errors compared to end-to-end models, which is also one of my concerns.

### Questions
Q1) The idea of using VLMs and programs for world simulation reminds me of the Genesis project [1] and its related works based on physics engines. Can you discuss the difference?

---

[1] https://genesis-embodied-ai.github.io/

### Soundness
2

### Presentation
3

### Contribution
2
