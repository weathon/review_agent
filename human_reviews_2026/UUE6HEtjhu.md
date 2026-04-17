# AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Vision-language-action (VLA) models have shown promise as generalist robotic policies by jointly leveraging visual, linguistic, and proprioceptive modalities to generate action trajectories. While recent benchmarks have advanced VLA research in domestic tasks, professional science-oriented domains remain underexplored. We introduce AutoBio, a simulation framework and benchmark designed to evaluate robotic automation in biology laboratory environments—an application domain that combines structured protocols with demanding precision and multimodal interaction. AutoBio extends existing simulation capabilities through a pipeline for digitizing real-world laboratory instruments, specialized physics plugins for mechanisms ubiquitous in laboratory workflows, and a rendering stack that support dynamic instrument interfaces and transparent materials through physically based rendering. Our benchmark comprises biologically grounded tasks spanning three difficulty levels, enabling standardized evaluation of language-guided robotic manipulation in experimental protocols. We provide infrastructure for demonstration generation and seamless integration with VLA models. Baseline evaluations with SOTA VLA models reveal significant gaps in precision manipulation, visual reasoning, and instruction following in scientific workflows. By releasing AutoBio, we aim to catalyze research on generalist robotic systems for complex, high-precision, and multimodal professional environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AutoBio, a simulation framework and benchmark aimed at evaluating vision-language-action (VLA) models for biology lab automation. Beyond standard rigid-body interaction, AutoBio adds (i) a digitization pipeline to turn real instruments into simulation-ready assets, (ii) MuJoCo physics plug-ins for lab-specific mechanisms, and (iii) a dual rendering stack (fast MuJoCo + Blender PBR bridge) with reactive instrument UIs. The benchmark contains 16 tasks across three difficulty levels; the experiments train/evaluate π0, π0.5, and RDT on 9 tasks, plus DP/ACT imitation baselines. Results: near-ceiling performance on “easy” tasks; sharp drops on medium/hard tasks involving precision screw motions, UI following, liquid-level reasoning, and rotor symmetry.

### Strengths
1. Well-Motivated and Novel Domain: The paper successfully identifies a clear and important gap in current robotics research. Moving from generalist domestic tasks to specialized, high-precision professional domains like a biology lab is a logical and necessary next step for the field. The unique challenges of this domain (precision, transparency, complex tool-use, protocol-following) are well-articulated.
2. New Simulation Features: The authors engineered new capabilities to meet the domain's demands, like an asset digitization pipeline, custom physics plugins.
3. Rigorous Benchmark and Clear Results: The benchmark tasks are well-designed and thoughtfully scaffolded from Easy to Hard. The experimental results are stark and unambiguous: SOTA VLA models fail significantly as task complexity increases. This demonstrates that AutoBio is a challenging and valuable benchmark that is not "solved" and will be effective at driving future research.

### Weaknesses
1. Lack of full, long-horizon protocol tasks: The introduction emphasizes "Long-horizon workflows"  as a key challenge. However, the benchmark consists of 16 discrete, relatively short-horizon tasks. The "long-horizon" experiment in the appendix (B.4) is weak, merely concatenating two easy tasks (open/close lid). A true test would be a multi-stage protocol combining several different primitives (e.g., pick tube, unscrew, aspirate, transfer to a new tube, place in centrifuge, operate panel).
2. Evaluation uses 9 of 16 tasks: the main results cover a subset (3 per level). It’s unclear whether conclusions generalize across all 16 tasks; several compelling tasks in Appendix B.2 (e.g., vortex mixing, multiple centrifuge variants) are left unreported. A fuller sweep or cross-task generalization metric would strengthen the case.
3. Analysis of VLA failures: The analysis that imitation learning fails on high-precision tasks due to "compounding errors"  is correct but not particularly novel. The procedural nature of the demonstrations  (i.e., "perfect" kinematic paths) is a known confounding factor for IL, as models struggle to mimic this non-human data. The paper would be stronger if it discussed this limitation or compared performance with teleoperated human data.
4. No sim-to-real validation: the work is framed as a stepping stone to lab automation, yet no real-robot evaluation is presented. Given the Blender PBR bridge, a small real-robot demo (even on easy tasks) would substantially improve credibility.

### Questions
1. The "Operate thermal mixer panel" task requires reading a reactive display. The analysis notes that models struggle, in part due to low-resolution inputs obscuring the numbers. Was the "basic" OpenGL renderer or the "advanced" photorealistic Blender renderer used to generate the visual data for training the VLA models?
2. When training a single policy across all 9 (or 16) tasks, does performance degrade vs per-task finetuning?

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
This paper presents AutoBio, a robotics benchmark in simulation evaluating robotic agents' ability to perform lab tasks/experiments important for biology research. It is motivated by increasing AI-based automation of scientific research and the importance and difficulty of lab research. The key contributions are a biology lab simulation environment, including implementations of simulated phenomena that are crucial for lab biology research; a benchmark of biological tasks; and systematic evaluation of VLAs and simpler imitation learning baselines. 

The paper discusses the simulation implementation: first, a process for digitizing assets, particularly lab equipment. Next, physics implementations to augment existing simulation capability: thread mechanism, detent mechanism, eccentric mechanism, and quasi-static liquid. For rendering, the benchmark uses basic rendering from MuJoCo, contributes advanced rendering blending MuJoCo's simulation state with blender's rendering pipeline, and contributes a reactive user interface. The benchmark itself includes randomized scene initialization, procedural demo generation, and task evaluation via predefined status checks on various pieces of state information. Tasks come in three difficulty levels, increasing in difficulty of vision and manipulation precision as well as task requirement/language instruction variation and abstraction/complexity. 

The experiments are conducted by fine-tuning VLAs (pi_0, pi_0.5, and RDT) on either 100 or 20 demo trajectories. Non-VLA imitation learning baselines are trained from scratch on the data. Results show that RDT is more consistent on easy tasks, while pi_0 and pi_0.5 are better in challenging scenarios. No significant advantage is seen from pi_0.5 over pi_0 on medium- to hard-level tasks, which the paper ascribes to high-level complexity and reasoning requirements. The paper also presents failure analysis: failures on easy tasks are usually due to gripper slippage, whereas medium and hard tasks are due to compounding precision manipulation issues; the paper suggests that this indicates a need for algorithms with stronger closed-loop learning abilities, including RL. Failures on these tasks are also due to language understanding limitations, and partial observability. These are exacerbated by low input resolutions, a limitation of the benchmark. Imitation learning baselines are able to match performance on a single task when they are trained on that task alone, but not in the multi-task setting.

### Strengths
### Quality
- Design makes sense and is well-explained.
- Task difficulty clearly matters and is differentiating, which is key - I find this to be the hallmark of a good benchmark 
- Failure analysis is really promising (see weaknesses for thoughts on how to improve, but great addition) 
- Experiment suite is very good! I especially appreciate the holdout experiments, both because I feel it's important to fundamental ML science, and because I suspect this is a domain where there will be plenty of unseen info for a long time. 
- Simulation additions are clearly valuable and significant 
### Clarity
- Incredibly well-written paper overall! Well-structured and very easy to follow. 
- Visual figures are really useful 
### Originality and significance 
- From what I understand, there are bio-lab benchmarks, but this one's design is unique and well-proven 
- Very important problem and this paper presents a benchmark that took considerable work in various aspects, so no concerns about significance.

### Weaknesses
### Quality
- Some simulation features presented do already exist, though I suspect not with the fidelity/design choices needed here, meaning they do not take away from the contributions of this paper. However, they would be worth adding to RW: transparency and liquids are simulated in multiple simulators including OmniGibson (BEHAVIOR-1K, Li et al. 2022), and a threading mechanism is used in TRANSIC (Jiang et al., 2024). 
- Failure analysis needs more detail. Would benefit from examples of failure modes, to convince the reader that the explanations are correct 
- Rendering issues leading to low-quality visual observations is a failure of the benchmark, not the method, and should be attributed accordingly 
- Some analysis/stats on the quality of the simulation would be valuable, though I think it's less central than better exposure of the method results 
### Clarity 
- Results are presented in simple tables. This makes them very hard to follow, especially for a benchmark where all the statistics are important. It would really help to have charts. This should include direct comparison between the baselines and VLAs, comparison between similar tasks, and direct comparison between difficulty levels, as in my opinion this is the most salient comparison toward judging the quality of the benchmark
- Overall, results section just needs more structure. The figures I talked about are a big part of it, but also a claims-driven structure - right now it's quite an info dump, which is hard to follow. 

The reason for the 6 is mainly that I want the results to be presented better. Given that, I would recommend a higher score.

### Questions
- What is the difference between "what" and "why" in Fig 1? 
- Questions about grounding failure analysis, as listed in Weaknesses
- Where do the procedural policies for demo generation come from?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a high-fidelity simulator targeting the biology lab automation domain for robotics. This domain has particular characteristics, like the preponderance of specialized machines (e.g. thermal mixers, centrifuges), clear / colored liquids, and transparent/translucent material (e.g. test tubes) which present difficulties for off-the-shelf robotic simulators like MuJoCo and its base OpenGL renderer. To solve this, they introduce a pipeline which digitizes real lab instruments through a 3D Gaussian Splat representation, adds  MuJoCo plugins for specialized physics (e.g. thread, detent, and eccentric mechanisms, liquid deformations) and more realistic rendering for transparent liquids through Blender physically-based rendering (PBR) and dynamically loading texture maps to make simulated machine control panels and displays dynamic.

In addition, given this simulator, the authors then develop a benchmark targeting 16 biologically-grounded tasks across three difficulty levels (easy, medium, hard) to systematically evaluate language-guided robotic manipulation in lab protocols. They then build infrastructure to generate 100 demonstration trajectories and then carefully evaluate state-of-the-art VLAs and smaller imitation learning baselines after fine-tuning them on the demonstrations.

### Strengths
Overall, the paper provides clear motivation for a focused effort on targeting simulation for the biology lab use-case, a significant area of both research and industrial importance and therefore has the potential for high impact.

The work is a high-quality, well-executed set of improvements targeting the precise difficulties of simulation in this domain. In particular, the improvements to base Mujoco (both physics and rendering) are extremely relevant to the domain. In particular, interacting with the control panels / displays of specialized machines is relatively rare in robotic simulation, and dynamically loading texture maps to achieve dynamic feedback is an example of a small but highly important infrastructural addition.

The primary originality of the work lies in the comprehensiveness and thoughtfulness of new techniques targeting the domain itself. While each technique (e.g. adding fluid modeling) is not extremely novel, the breadth of these which target domain-specific simulation issues demonstrates creativity.

Finally, the careful set of benchmark tasks and baseline evaluation of VLAs / imitation learning policies is likely to be highly impactful, demonstrating the clear gap between current SOTA methods on this benchmark. The analysis of these results is thorough.

### Weaknesses
Especially given that most of the work is aimed at achieving stronger realism, the paper would be strongly improved by any real-world experiments demonstrating that the methods result in transfer onto real robotic hardware.

A significant aspect of the domain is executing longer-term procedures, and much of the work is motivated by the long task horizons. However, the main text has minimal emphasis on this aspect in the experiments / tasks, and the long-horizon task in the appendix is simply concatenation of short trajectories.

The simulation is also focused on static manipulation, which may preclude true autonomous execution of multiple steps in more realistic lab environments given the lack of mobility.

The evaluation trained separate models for each task and evaluated tasks independently, while VLAs are motivated by cross-task transfer and language generalization; evaluation of a VLA trained and tested across multiple tasks could be informative.

### Questions
As you mention, low input resolutions exacerbate instruction following. Is this an issue with the task setup in the "Operate thermal mixer panel" which actually prevents the task from being fully executed?

Can you expand on the physical realism of the "aspirate with pipette" task? Presumably some level of tactile or force feedback is necessary for working with pipettes in the real world in order to control the amount of liquid aspirated.

Is there any value in providing instrument readings as an observation to the policies directly? Presumably some lab devices are IoT-enabled, and it could be a reasonable test of whether perception is fundamentally limiting policy performance or not.

Do you expect increasing the number of demonstrations to significantly affect performance on the medium/hard tasks? Further investigation on data scaling effects (perhaps for a subset of the tasks) could be highly informative.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes AutoBio, a simulation benchmark designed specifically for laboratory tasks and environments. The authors provide a laboratory equipment asset generation pipeline, relevant physics plugins, rendering which supports e.g. transparent materials, and a data generation pipeline. The authors also benchmark VLA and IL baselines on tasks of varying levels of difficulty.

### Strengths
- Useful task suite in an area of robotic manipulation with few realistic benchmarks
- Careful consideration of physics, rendering, assets, etc in the context of biology tasks
- VLA and IL baselines are relevant and highlight weaknesses in more complex tasks
- The presentation is clear and contributions well-explained

### Weaknesses
- Seeing as the realistic assets, physics, and rendering are a central focus, validation on a real robot setup (even on the simpler tasks) would support claims of realism
- The paper notes VLAs may perform well as multi-task agents in the discussion, however this setting is not evaluated

### Questions
- Are benchmarks on simulation performance available (e.g. simulation speed) to gauge evaluation speed, and potentially the applicability of online learning methods?
- The authors note the potential multitask capability of the VLA models. Did authors run multitask experiments on the VLA and IL models?
- Are there future plans to provide additional tasks (e.g. longer horizon, more subtasks, etc), leveraging the same physics and rendering framework?

### Soundness
4

### Presentation
3

### Contribution
3
