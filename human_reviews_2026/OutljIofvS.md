# RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
The pursuit of robot generalists, instructable agents capable of performing diverse tasks across diverse environments, demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. As policies expand in scope and complexity, these barriers only intensify, since defining ``success'' in robotics often hinges on nuanced human judgments of execution quality. We introduce RobotArenaInf, a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons.
To measure robustness, we systematically perturb simulated environments along multiple axes, including textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world-trained robot manipulation policies, addressing a critical missing capability in today’s robotics landscape. Benchmark website at \href{https://robotarenainf.github.io}{\texttt{robotarenainf.github.io}}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a fairly comprehensive benchmarking system for evaluating VLA models for robotics. The contributions of this work are:

- A benchmarking protocol for robotics in simulation. This includes VLM-based task evaluation and preference modeling.
- An automated video to simulation pipeline.
- Extensive evaluation of many VLA models.
- Evaluation results across in-distribution and out-of-distribution cases.

### Strengths
- This work provides a powerful framework for the robotics community to develop on top of. This kind of work can really make an impact on scaling the data necessary for VLA-style models to succeed.
- The breadth of this work is quite remarkable, from automating video-to-simulation to evaluation across many of the open-source VLAs to extensive evaluation results.
- This work fills in significant gaps in an end-to-end pipeline that would make robot training closer to the approaches taken for training LLMs.
- The presentation of this work is well done, with thoughtful and descriptive figures and clearly robust evaluation.

### Weaknesses
- Overall, the work is quite strong, but there could be some clarifications that could strengthen the claims in the paper.
- On the claims of comprehensive evaluation, the authors should comment on the evaluation of diverse language instructions from users, as this is a reasonable failure mode of VLAs (e.g. https://arxiv.org/pdf/2411.18676)
- The authors should make it clear what aspects of the pipeline are truly new vs. aggregations of existing approaches. For instance, how original is the approach for real-to-sim? The work doesn’t need to be completely novel, but it would have been nice to see the authors comment on how this approach draws from prior work.

### Questions
- There doesn’t seem to be a discussion on the evaluation of diverse language instructions from users in this work. Can the authors comment on this?
- Related works:
    - Could the authors discuss other works that similarly provide a real to sim pipeline, such as https://arxiv.org/pdf/2403.03949? I think it would be helpful to understand how much of this work is really an aggregation of existing approaches vs. novel contributions that the reader can take away.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a scalable benchmarking framework for evaluating real-world–trained vision-language-action (VLA) robot policies. Traditional real-world testing is costly, slow, and unsafe, while existing simulation benchmarks lack the realism to assess models grounded in real demonstrations. The proposed method bridges this gap by automatically converting real robot videos into simulated digital twins using advances in vision-language modeling, 2D-to-3D generation, and differentiable rendering. Within these simulations, robot policies are evaluated through both automated vision-language-model scoring and large-scale human preference judgments collected online. The framework also introduces systematic environment perturbations to test robustness, enabling reproducible, human-aligned, and scalable evaluation of general-purpose robot manipulation agents.

### Strengths
1. The paper introduces a comprehensive and automated pipeline that seamlessly bridges real-world robot data and simulation, enabling scalable and reproducible evaluation of vision-language-action models.
2. It conducts the largest cross-lab evaluation to date, providing unprecedented insights into the generalization capabilities and limitations of current generalist robot policies under diverse distribution shifts.
3. The paper is well-written, presenting a complex technical system with conceptual clarity and a logical narrative that is easy to follow.

### Weaknesses
My main concerns lie in the proposed real2sim pipeline, including:

1. The 3D assets are primarily from some 3D generation models, which may produce meshes with different shape with real objects, or have implausible collisions. Moreover, the physical parameters are given by some LLMs, which can also result in some implausible physical movements in simulation.
2. Inpainting background makes the camera viewpoint in this evaluation process remain the same as, or close to, that in the original video. However, we often require other camera or multiple cameras for policy inference, e.g., the wrist/first/third view cameras.
3. The pipeline requires system identification of robots, which can cause much extra data collection and human efforts.

### Questions
1. I am curious about the evaluation results of the pi series VLA models on the proposed benchmark.
2. Can you provide more detailed information of your sim-real alignment experiments? For example, what are the controllers and control frequencies of the simulation and real environments, since controller type and frequency can sometimes cause different motions in simulation and real world?

### Soundness
2

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
2

### Summary
In this work, the authors introduce a real-to-sim translation framework named RobotArena to tackle the challenges in real-world robot policy benchmarking with the following two aspects: 1) an automated pipeline for converting real-world videos into simulated environments via VLM-based scene understanding and differentiable rendering. 2) an evaluation protocol combining VLM-based progress scoring and crowdsourced human preference feedback. The authors conduct extensive experimental results on hundreds of scenarios to demonstrate the effectiveness of proposed framework on VLA-based policies.

### Strengths
1. The automated real-to-sim translation pipeline that the paper introduces is innovative.
2. The hybrid assessment method integrating VLM-based scoring with human preference feedback is comprehensive.

### Weaknesses
1. The simulation environment cannot accurately reproduce fine-grained physical interactions (e.g., plug insertion, deformable object manipulation), limiting evaluation fidelity for precision tasks.
2. The multi-stage pipeline may accumulate errors, but the paper lacks quantitative analysis of error propagation across stages.
3. The benchmark primarily focuses on static-camera, table-top manipulation tasks from datasets like Bridge and DROID, lacking coverage of dynamic scenarios, mobile navigation, or multi-object interactions.

### Questions
Regarding the real-to-sim translation pipeline design: 
1. Although the fully automated reality-to-simulation translation pipeline is novel, could the authors provide more details on how error propagation across these stages is quantified? For instance, how do inaccuracies in VLM-based segmentation or 3D mesh generation impact the final simulation fidelity?

Regarding the evaluation methodology and potential biases:
1. The hybrid evaluation combining VLM scoring and human preferences is compelling. Yet, how do the authors mitigate biases inherent in both methods? 

Regarding the experimental design and generalization claims: 
1. The large-scale experiments reveal significant cross-dataset performance drops. Could the authors discuss whether the observed failures are due to visual domain shifts or task complexity differences? 
2. How might the benchmark be extended to include dynamic scenarios to better assess true generalization?

### Soundness
3

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
4

### Summary
This paper introduces a scalable framework for evaluating VLA policies in robotics. It automatically converts real-world robot demonstration videos into simulated environments using advances in VLMs, 2D-to-3D generative modeling, and differentiable rendering. The benchmark assesses robot policies through automated VLM-based scoring and human preference judgments, enabling large-scale, reproducible evaluations without manual setup. Experiments reveal limited cross-dataset generalization and robustness in current models, while confirming consistent performance rankings.

### Strengths
- The pipeline that transforms real-world robot demonstration videos into simulated environments is very nice and generally useful.
- The study conducts, according to the authors, the most extensive evaluation of generalist robot policies to date.

### Weaknesses
Real-to-sim pipeline is very nice. But for example, how is it better than testing VLAs on a bunch of different simulation environments? That would also make sure to include some out-of-distribution domains, wouldn't it? And why is this work framed as a policy evaluation work? It looks like a real-to-sim method, and it deserves credit for that contribution (more for it than for policy evaluation, because real-to-sim is more general).

I think the paper has a good potential, but the following improvements (or addressing these comments) would make it much stronger:
- Wrist camera is a common input, but the current work is missing them as acknowledged by the paper.
- Reliance on generative models may introduce reconstruction artifacts and physics inaccuracies.
- Benchmark scope limited to manipulation tasks: no locomotion, mobile, or multi-agent settings.
- Evaluation depends on a specific VLM backbone (Gemini). This harms accessibility. It would be nice to have open-source variants that are more accessible to research labs.
- Domain perturbations are hand-designed and may not capture natural distribution shifts.

### Questions
Please see the first paragraph that I wrote above in the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
4
