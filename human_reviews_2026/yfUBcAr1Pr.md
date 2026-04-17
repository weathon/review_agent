# VITA-E: A Dual-Model Framework for Real-Time, Interruptible, and Concurrent Human-Robot Interaction

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
Current Vision-Language-Action (VLA) models are often constrained by a rigid, static interaction paradigm, limiting their ability to handle real-time user interruptions or perform concurrent tasks such as speaking while acting. This hinders seamless human-robot collaboration, resulting in an inflexible and unresponsive user experience. To address these limitations, we introduce VITA-E, a novel dual-model framework designed to enable flexible and robust human-robot interaction in real-time. The core of our approach is a dual-model architecture where two parallel VLA instances operate as an **Active Model** and a **Listening Model**, allowing one to instantly intervene in the other. We further propose a ``model-as-controller'' paradigm, where we fine-tune the VLM to generate special tokens that serve as direct system-level commands, coupling the model's reasoning with the system's behavior. Experiments conducted on a physical humanoid robot demonstrate that VITA-E can reliably handle complex interactive scenarios. Our framework is compatible with various dual-system VLA models, achieving a 100\% success rate on emergency stops and speech interruptions while also successfully performing concurrent speech and action. This represents a significant step towards more natural and capable robotic assistants.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores interruptible and concurrent human–robot interaction using a dual-model framework, called VITA-E. Specifically, two VLMs operate as an “active model” and a “listening model”, allowing one to intervene in the other, thereby enabling interruption while the robot is executing controls. The VLM can also produce speech commands and emit special control tokens to realize a model-as-controller paradigm. The experiments show that the framework enables interruption during robot control.

### Strengths
1) The work studies a significant and practical challenge in HRI and generalist robotics: enabling interruptions and concurrent speech–control behavior during manipulation tasks.

2) The dual-VLM system is reasonable, and the overall interaction modes and response modes are clear and useful.

3) Real-time interruption and concurrency are crucial for HRI. The real-robot experiments demonstrate effectiveness for interruption, concurrency, and emergency stop.

### Weaknesses
1) The paper claims related works lack interruption during robot action, leading to limited baselines for interactive tasks. It would be useful to provide simple baselines—for example, a single-VLM system that runs continuously: If the human issues a new command when the action expert is controlling, VLM immediately sends the new command to the action expert (i.e., one fast “active VLM” that always responds with the latest command).

2) The method appears highly dependent on the VITA model, and the methodology for enabling interruption seems inspired by VITA, which may reduce the novelty. Please highlight the key differences beyond fine-tuning action experts.

3) Because the approach runs two VLMs, it is important to report memory and time usage and compare to baselines. If the dual setup reduces speed, it may hinder applicability.

4) Figure 4 shows the proposed method underperforms the GR00T baseline, especially on LIBERO-10. It is important to analyze why. This is a weakness.

### Questions
1) Please provide baselines for interactive tasks (e.g., a single-VLM polling/time-sliced baseline).
2) Clarify the contribution over VITA—what is fundamentally new here?
3) Report time and memory costs, and compare to baselines.
4) Why does the method underperform GR00T on LIBERO-10? Any ways to improve it (e.g., training data, partial fine-tuning, or control-token robustness)?

Please see the [Weakness] section for more details.

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
3

### Summary
This paper presents VITA-E, a dual framework designed to enable flexible, real-time human-robot interaction in VLA. The core innovation is a parallel architecture where two VLA instances operate as an "Active Model" and a "Listening Model," allowing one to instantly intervene in the other. The framework enables several interaction modes. Experiments on Fourier GR2 demonstrate 100% success rates on emergency stops and speech interruptions, with 93.3% success on task switching.

### Strengths
- The paper targets on an important limitation in current VLAs, unabling to handle real-time interactions such as interruptions.

### Weaknesses
- While the research problem is interesting, the individual modules are not particularly innovative.
- As for the listening model, what is the difference between equipping traditional VLAs with an audio encoder/translator? For example, when receiving audio inputs, perhaps use an audio transcription model to translate the audio into some textual instructions for VLAs. This could at least be a baseline.

### Questions
- If we only tune the listening model, will the performance in Figure 4 drop as well?
- More training details should be provided, such as computational cost comparisons.
- Assume the person says a long sentence. Will the model react during this sentence? Or just wait until it is finished.

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
4

### Summary
This paper introduces VITA-E, which is a framework for VLA systems that aims to improve interaction between the human and robot. VITA-E addresses a very relevant problem in current VLA systems, where the user is unable to interrupt or speak to the robot while it is executing an action. This is an actively studied problem, with the paper citing similar work like [RACER](https://arxiv.org/abs/2409.14674) and [Switch-VLA](https://arxiv.org/abs/2506.03574). What sets VITA-E apart is that it places heavier focus on interactivity with the robot by allowing asynchronous conversation and action.

VITA-E adapts the previous work of [VITA](https://arxiv.org/abs/2408.05211) to robotic tasks. The VITA-E architecture uses two instances of the VLA during execution, where one model acts as a "listener" and the other as an "actor". The authors train the VLA with an augmented dataset to output a set of control tokens that signal different behaviors in the system. This approach enhances interactivity with the system, as the listener can converse with the user, switch tasks, or stop execution while it is executing its current task. 

The authors evaluate VITA-E on an extensive set of experiments using both simulation and real world scenarios. Results show that VITA-E improves interaction with the robot while also staying competitive in model performance on tasks.

### Strengths
+ **Originality**: The paper frames the problem of interactivity in a VLA system in an original and important manner. The authors apply the existing work of VITA to the different domain of human-robot interaction.

+ **Quality**: The paper is well structured; has many clear and visually appealing diagrams. Experiments and results sufficiently show value of the architecture.

+ **Clarity**: The presentation of ideas is well thought out and easy to understand.  

+ **Significance**: The paper highlights how current VLA architectures lack flexible interaction with humans, and provides an innovative solution to the problem of robot interactivity during task execution.

### Weaknesses
+ The conclusion and future work section aptly addresses the weaknesses of the approach.
+ The dual model architecture requires ~2x more compute than a single model.
+ Interruption of tasks and task switching is not very flexible due to it currently relying on the safe retraction to neutral state.
+ There is a loss of model performance from fine-tuning, but as the authors mentioned this is because they had to freeze the base VLM.

### Questions
+ On page 4, you refer to the "Listening" state as "Hearing" state. I'd suggest sticking with a single term to describe this state throughout the paper.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
VITA-E proposes a real-time VLA framework that makes robot interaction interruptible, concurrent, and safe. It runs two VLA instances in parallel—an Active model that executes and a Listening model that can preempt—while the VLM serves as a controller by emitting special tokens (e.g., [ACT], [RES], [INST], [HALT], [END]) that drive a simple state machine for speaking vs. acting. Training reformats embodied data so the model learns to produce these control tokens for manipulation and safety. On a physical humanoid platform, VITA-E achieves fluent talk-while-act (~2.26 s voice latency), 100% success for speech interruption and emergency stop, and 93.3% success for task switching—demonstrating a practical, general recipe for fluid, interruptible human-robot collaboration.

### Strengths
The paper is genuinely original in how it reframes embodied interaction around preemption and safety, combining a dual-model architecture with a lightweight control-token interface to remove the “turn-based” limitation common in VLAs. Its quality is supported by thoughtfully chosen real-robot evaluations that directly test concurrency, interruption, task switching, and emergency stop, plus ablations that show the control-aware fine-tuning—not just prompting—is what yields reliability. The writing is clear and reproducible: the state machine, token semantics, and data formatting are specified precisely, with figures that make the listen/act transitions easy to follow. In significance, the work matters because it elevates interruptibility and safe handoffs to first-class capabilities, offering a model-agnostic interaction layer that others can adopt across stacks; this has immediate relevance for deployable HRI in homes, assistive settings, and light industry, even if downstream perception or control modules evolve.

### Weaknesses
1. Model design: The dual-model architecture introduces a trade-off between responsiveness and computational cost. The authors acknowledge this limitation, and the framework’s generalizability to broader robot stacks has not been demonstrated.

2. Evaluation scope: The experimental evaluation is narrow, especially regarding interaction performance. Concurrency results are reported qualitatively, while other metrics are based on 30-trial averages. Baselines are largely omitted due to mismatched capabilities, leaving comparative advantages insufficiently analyzed.

3. Performance: On manipulation benchmarks, the proposed method is not state-of-the-art, which may limit its appeal to researchers focused on maximizing task success.

4. Task switching: Reported failures arise from the VLM misclassifying new directives as dialogue, revealing brittleness in the control-token decision boundary under distribution shifts.

5. Training and generalization: The pipeline relies on synthetic token injection (e.g., simulated “Stop!” events) instead of real interruption data, raising questions about generalization and safety in real-world scenarios.

6. Unaddressed challenges: Core capabilities such as handling long-horizon, multi-stage tasks and ensuring smooth transitions remain future work. Consequently, while the interaction layer is promising, it is not yet a turnkey solution for complex deployments.

### Questions
1. What is the approximate time it takes from giving a command to stopping the action mentioned in the paper? The time is the same for different tasks？

2. Whether the success rate of emergency stop and voice interruption can be maintained on different real machines, and whether the efficiency can be maintained？

### Soundness
3

### Presentation
2

### Contribution
3
