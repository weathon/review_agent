# LeRobot:  An Open-Source Library for End-to-End Robot Learning

- Decision: Accept (Poster)
- Scores: 8, 8, 6

## Abstract
Robotics is undergoing a significant transformation powered by advances in high-level control techniques based on machine learning, giving rise to the field of robot learning.
Recent progress in robot learning has been accelerated by the increasing availability of affordable teleoperation systems, large-scale openly available datasets, and scalable learning-based methods.
However, development in the field of robot learning is often slowed by fragmented, closed-source tools designed to only address specific sub-components within the robotics stack.
In this paper, we present lerobot, an open-source library that integrates across the entire robotics stack, from low-level middleware communication for motor controls to large-scale dataset collection, storage and streaming.
The library is designed with a strong focus on real-world robotics, supporting accessible hardware platforms while remaining extensible to new embodiments.
It also supports efficient implementations for various state-of-the-art robot learning algorithms from multiple prominent paradigms, as well as a generalized asynchronous inference stack.
Unlike traditional pipelines which heavily rely on hand-crafted techniques, lerobot emphasizes scalable learning approaches that improve directly with more data and compute.
Designed for accessibility, scalability, and openness, lerobot lowers the barrier to entry for researchers and practitioners to robotics while providing a platform for reproducible, state-of-the-art robot learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
LeRobot is an open-source, end-to-end robot-learning library from a consistent Python middleware for diverse, low-cost and humanoid/mobile robots to standardized data tooling and scalable policy deployment. It introduces the multimodal LeRobotDataset to ease large-scale collection and reuse of teleoperation data, and an optimized, decoupled inference stack that separates action prediction from execution for robust real-time control. The library ships clean PyTorch implementations of state-of-the-art methods across RL and BC (e.g., ACT, Diffusion Policy, VQ-BET, π₀, SmolVLA) with emphasize on accessibility, reproducibility, and openness. I have no doubt that it presents a useful framework for the community.

### Strengths
1. Many policy learning algorithms, standardized end-to-end robot learning dataset schema, tooling, and policy deployment in one framework
2. Real-robot coverage with low-cost focus for accessibility and democratization; resource transparency
3. Strong evidence of community uptake

### Weaknesses
The paper’s writing currently falls short of top-tier conference standards, particularly in the precision and clarity of its claims. In addition, it lacks a discussion of reproduced results from the implemented algorithms, making it difficult to assess the reliability of the framework. My detailed concerns are outlined below.

### Questions
1. **Scope of the “entire robotics stack.”** The claim of covering the “entire robotics stack” is an overstatement. The library primarily targets robot learning for manipulation tasks and omits major subfields such as SLAM, sim-to-real transfer, and broader perception and control systems. The authors should moderate this claim to accurately reflect the covered scope.
2. **Ambiguity in “explicit” vs. “implicit” models.** The terminology used to distinguish explicit and implicit models is confusing. The paper conflates learning-based methods with implicit models, whereas both explicit (e.g., flow models) and implicit (e.g., energy-based) formulations exist in machine learning. Clarifying what each abstraction encompasses, and specifying how methods that explicitly model the world or action distributions (with or without data) fit into this taxonomy, would improve clarity.
3. **Clarity on compounding errors.** The paper states that classical, modular methods (labeled as explicit models) suffer from compounding errors, but so do monolithic, data-driven policies (labeled as implicit models). It is unclear how the monolithic networks mitigates this issue. The authors should clarify whether their choice genuinely addresses compounding errors or simply shifts where they occur.
4. **Definition of middleware.** The role and scope of “middleware” are not clearly defined. A brief explanation in the main text of what components or functionalities are included under this term would improve clarity for readers.
5. **Language and grammar issues.** The paper would benefit from a grammar and spell check. Some (nit-pick) examples include:
    - “aboard robots” → “onboard robots” (more standard usage)
    - “parallely to low-level control” → “in parallel with low-level control loops”
    - “the the” → “the”
    - “coexistance” → “coexistence”
6. **Reproducibility of supported methods.** It is unclear whether the methods implemented in *LeRobot* reproduce the results reported in the original papers? If not, how large is the gap? Providing a clear mapping (e.g., method X on benchmark Y) and where to verify such reproduction within the repository would help assess reliability of the implemented algorithms.
7. **Extensibility and community contribution.** Including brief documentation or a section describing how to add new robots, learning methods, and simulation environments would make the framework more approachable for community contributors.
8. **Future directions.** A short discussion on planned extensions—such as supporting additional teleoperation devices, robot types, and learning methods—would help situate the library’s development trajectory. The README may be a suitable place for such a roadmap.

### Soundness
3

### Presentation
1

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces LeRobot, an open-source library for robot learning that provides an end-to-end stack for scalable robotics research. It features unified robot integration, standardized datasets, optimized inference, and efficient, reusable state-of-the-art robot learning algorithms. This effort addresses fragmentation in the field, where researchers typically develop tools for their own use with specific robot platforms, data formats, and learning algorithms. By providing a unified and accessible framework, the library reduces the entry barrier and accelerates progress in robot learning through improved accessibility, scalability, and reproducibility.

### Strengths
- The paper addresses the important problem of entry barriers in robotics and embodied AI research. A unified platform and interface for hardware, data collection, and learning would benefit many researchers in the field of robot learning.

- The paper identifies key challenges and roadblocks to progress in robot learning, which motivates the proposed library.

- Current downloads and usage of the platform already demonstrate the value and need for such a unified platform. The statistics on model and dataset downloads also provide insights into the community’s interests and needs over time.

### Weaknesses
- Lack of discussion of what tasks or scenarios may not be suitable for the current LeRobot library. Would all researchers in robot learning benefit from the current LeRobot library?
- The paper would benefit from more discussion on the integration of simulated environments, as many researchers start with toy simulations. What is the procedure for adding custom environments to the ecosystem?
- Broken citation on line 266.

### Questions
- How does LeRobot compare to ROS? What factors might make LeRobot more successful than ROS as an open-source robotics platform?
- How is the correctness of the implemented algorithms ensured, and how is reproducibility maintained?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a library for end-to-end robot learning. It addresses a very pervasive and relevant problem of a typically fragmented robotics stack, for which having simple and unified tooling would greatly enhance productivity and rampup time for beginners. It has various subcomponents that address different parts of the stack: hardware integration, local and streaming datasets in a shared format, async inference for action chunking based models, RL and IL constructs, simulation integrations etc. All of these together are an excellent starting point for someone looking to conduct research and development on low-cost readily available robots in a non-production setting.

On balance, I am slightly leaning towards accepting the paper as it would provide a reference to people using the tooling to conduct research and for publications. However, I do feel that the core functionality of the library has several avenues for improvement (some iterative and others fundamental).

### Strengths
- This is clearly a significant contribution that will help a lot of additional people easily take up robotics and experiment on well-integrated low cost robots. Someone who would previously have been blocked on a part of a complicated stack can use the abstracted tooling of the library to get started quicker and with lesser frustration.

- The library supports a lot of popular models and algorithms that have been shown to work on real robots in a variety of settings.

- The library contains support for asynchronous inference (i.e. that used in modern models that use action chunking and related methods). The functionality to use a hybrid cloud-robot inference is also very useful.

- The dataset tooling seems very useful, especially the structure which helps handle larger scale datasets with millions of trajectories.

- I find it positive that the paper explicitly lists the limitations of the library. This is absolutely understandable since it is in active development.

### Weaknesses
- While the overall library seems excellent as a starting point for someone learning about robotics or who wants to get started with minimal friction with a toy problem, it seems pretty raw for something like production usage. The parallel I would like to draw here is of pytorch, which was suited for both simplicity and production robustness.

- Local porting of large datasets seems very slow compared to what should be possible. E.g. 7+ days of processing time for DROID should be able to be sped up significantly by better parallel processing and systems engineering.

- I feel like LeRobot overindexes on what is ‘popular’ in the current robotics/ML types of models and algorithms (e.g. RL and imitation learning), but lacks some essential functionality that a user would likely need to resort to a third party library for. An example of this is optimized motion planners or IK solvers that could run in either real or simulation environments, which could give ground truth data to train particular e2e models.

- Simulation is often a critical component of end-to-end robot learning. I would have liked to see additional support for additional simulation frameworks (e.g. IsaacSim or Genesis). This to me, would abstract away a lot of the friction a typical user needs to undergo to setup the simulation environment itself. Note: MetaWorld and Libero seem impractical for a lot of real world tasks compared to something more comprehensive.

- A crucial element of deploying robotic systems in the real world is a robust safety layer. In my opinion, the deployment portion of the library should treat safety as the most important principle and contain abstractions that allow for this to be realized (e.g. add protective stops based on certain criteria). While these are typically also implemented in the firmware of the robot itself (e.g. as protective stops or estops), having an added layer on top that can focus on the specific logical safety considerations would be helpful when directly coupled with (for example) the RobotClient / PolicyServer

- (not relevant to the assessment of the paper) It was impossible to not know where the work came from, given that the library is already very popular in the online robotics/ML community.

### Questions
- Do you have any detailed benchmarks for the throughput of LeRobotDataset and StreamingLeRobotDataset as opposed to something like MCAP?

- Do you plan to add GELLO teleop support?

### Soundness
3

### Presentation
3

### Contribution
4
