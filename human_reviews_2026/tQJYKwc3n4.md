# RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 10, 6, 6, 6

## Abstract
Recent advances in robot learning have accelerated progress toward generalist robots that can perform everyday tasks in human environments. Yet it remains difficult to gauge how close we are to this vision. The field lacks a reproducible, large-scale benchmark for systematic evaluation. To fill this gap, we present RoboCasa365, a comprehensive simulation benchmark for household mobile manipulation. Built on the RoboCasa platform, RoboCasa365 introduces 365 everyday tasks across 2,500 diverse kitchen environments, with over 600 hours of human demonstration data and over 1600 hours of synthetically generated demonstration data---making it one of the most diverse and large-scale resources for studying generalist policies. RoboCasa365 is designed to support systematic evaluations for different problem settings, including multi-task learning, robot foundation model training, and lifelong learning. We conduct extensive experiments on this benchmark with state-of-the-art methods and analyze the impacts of task diversity, dataset scale, and environment variation on generalization. Our results provide new insights into what factors most strongly affect the performance of generalist robots and inform strategies for future progress in the field.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper introduces RoboCasa365, a large-scale simulation framework designed for training and benchmarking generalist robot policies for everyday household tasks. Building upon the existing RoboCasa platform, the authors make significant contributions by massively scaling up the content and creating a structured evaluation suite. The framework includes 2,500 diverse kitchen environments derived from real-world home layouts, a comprehensive set of 365 everyday tasks (65 atomic and 300 composite), and over 2,200 hours of robot interaction data (both human-collected and synthetically generated via MimicGen). The work establishes benchmarks for three critical research areas: massively multi-task training, foundation model training (pre-training and fine-tuning), and lifelong learning. Through extensive experiments, the authors evaluate several state-of-the-art models (Diffusion Policy, π₀, and GR00T N1.5), demonstrating the utility of the benchmark for analyzing the impact of data scale, task diversity, and pre-training on policy generalization.

### Strengths
- The primary strength of this work is the sheer scale and diversity of the simulation environment and accompanying datasets. The introduction of 2,500 unique kitchen scenes modeled after real-world homes and 365 distinct tasks represents a monumental step up from prior simulation benchmarks, which are often limited to a few scenes or a narrow set of tasks. This scale is critical for training and rigorously evaluating the generalization capabilities of high-capacity foundation models.
- The authors have thoughtfully structured the entire framework to facilitate rigorous scientific inquiry. The explicit separation of assets, scenes, and tasks into "pre-training" and "post-training" splits  is a key design choice that enables controlled studies on generalization, data efficiency, and transfer learning, which are central questions in the field.
- The paper provides not just a dataset, but a full suite of defined benchmarks for multi-task learning, foundation model training, and lifelong learning. This provides a ready-to-use toolkit for researchers to compare methods reproducibly and systematically analyze model performance across different learning paradigms.
- The significant expansion of interactable and articulated appliances (from 20 to 456 instances across 12 categories)  is a substantial contribution. This allows for the creation of more complex, realistic, and long-horizon tasks  that better reflect the challenges of real-world household robotics, moving beyond simple pick-and-place scenarios.

### Weaknesses
While acknowledging the sim-to-real gap as a limitation, the paper lacks any empirical grounding or discussion of how its simulation-based success metrics might correlate with real-world performance. This is a critical omission for a benchmark intended to gauge progress towards deployable generalist robots. Recent work such as STAR-Gen[1]  proposes a detailed taxonomy for evaluating real-world generalization across distinct visual, semantic, and behavioral axes. The binary task success metric used in RoboCasa365 is coarse and may not be predictive of a policy's performance on these fine-grained real-world challenges. For instance, a policy might succeed across many simulated scenes but still be brittle to specific real-world semantic or behavioral perturbations not adequately modeled in simulation (e.g., changes in object mass or nuanced language instructions).

[1] Gao et al. A Taxonomy for Evaluating Generalist Robot Policies.

### Questions
The multi-task evaluation includes GR00T N1.5, a model primarily developed for humanoid robots. While its strong performance is interesting, could you discuss the implications of evaluating a humanoid-centric foundation model on a mobile manipulator task set? Does this suggest a high degree of embodiment-agnostic knowledge transfer, or are there specific architectural features (e.g., in its vision-language processing) that are particularly well-suited to the benchmark's challenges, independent of the final embodiment?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors presented RoboCasa365, a large-scale simulation framework for benchmarking generalist robot policies, and offer large pre-training and post-training datasets as well. It provides 2500 kitchen environments with 2000 hours of simulated robot data.

### Strengths
- The authors provide a valuable resource for the community to benchmark generalist robot policies: there is really high diversity in simulated scenes, and a large amount of demonstrations for pre-training and post-training
- There are a large number (365) of tasks spanning across 50 different categories. There is a large number of scenes, with 50 different layouts and each have 50 different styles. This is a huge improvement over prior work.
- The authors also provide a good tool to benchmark lifelong learning, and the dataset includes tasks with various numbers of subtasks.
- The authors provide good experiments to demonstrate the use of the benchmark and the dataset

### Weaknesses
Overall this is a good paper and a great resource for the community. However, I think the experiments section is still a bit lacking, and misses a good opportunity to provide more insights into the dataset collected by the authors.

- The writing in the experiments section is somewhat unclear: (1) It is unclear what the evaluation protocol is and how different models are compared (aka do you report success rate? Are there partial success? etc.). It is also unclear what the numbers mean in Table 1: is it success rate? (2) There lack discussion in the main text on the difference between the three models being compared, and lacks discussion on why GR00T N1.5 may be performing the best.  (3) Table 2 doesn’t bold the best numbers 
- In section 4.2 (specifically Figure 5 and Table 2), in would be interesting to see comparison with: finetune GR00NT 1.5 public checkpoint on a mix of pre-training + post-training data, but does not first do pre-training and then do fine-tuning, and instead just do “pre-training” over the entire data mix.
- It is unclear what checkpoints the authors chose to evaluate for the experiments, and how they chose it. It is unclear how long “pre-training” is and how many epochs over the dataset it’s taken.
- In Section 4.4, the authors observed that Human300 outperforms Human300+MG60, and they hypothesized that it is because “adding MimicGen data dilutes the contribution of other human datasets”. However, there is not support for this hypothesis. One experiment to verify this hypothesis would be to downweight the MG60 data in pre-training and compare. This is very important as a huge chunk of the dataset is composed of MimicGen data, and it would be important to know how useful this data is (or whether it is useful at all). If it is not useful, then the actual dataset would be much less than the 2000h claimed in the paper, and would need to adjust the writing to reflect that.
- Related work section lacks full discussion on benchmarks for generalist robots outside of simulation, such as [1][2][3] and live competitions like [4] [5]

[1] Train offline, test online: A real robot learning benchmark
[2] Homerobot: Open-vocabulary mobile manipulation
[3] AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World
[4] The DARPA robotics challenge finals: Results and perspectives
[5] Analysis and observations from the first amazon picking challenge

### Questions
See my main points in the weakness section.
Other small questions:
- There is no discussion of the underlying simulation framework and how it compares to other simulation frameworks: how fast is it? Is GPU rendering supported?
- Can you say more about how the MG60 dataset is generated and how diverse it is?
- How optimal are the human teleoperated demonstration data?

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
The authors propose a large scale imitation learning benchmark for household manipulation tasks, for studying pretraining and postraining of large scale generalist policy architectures. There are 2500 distinct kitchens, and 365 (some mobile) manipulation tasks, ranging from atomic to compositional.  The authors collect 100 demos for each pretraining task (300 out of 365), and also use the MimicGen framework to generate an additional 10k synthetic demonstrations for each task. The authors evaluate generalist policies in zero-shot task generalization after pretraining, improvement from pretraining and posttraining, and lifelong learning. They find that GROOT is the best for generalization, lifelong learning is still challenging, and that pretraining data composition matters over pure quantity.

### Strengths
The benchmark seems quite comprehensive and the experiments are sensible and  systematic.

One very interesting result is that synthetic demos do not help at all. In the pretraining data study, they show that just using the human demos is good enough. I would like to see additional explanation in why MimicGen is not helpful, since MimicGen itself reports positive gains in using synthetic demos. 

The paper is easy to read.

### Weaknesses
This is not really a substantial weakness, but the experiments and results are very straightforward and almost boring. It would be nice to see more interesting or qualitative phenomena, such as why synthetic demos are harmful, why GROOT outperforms other generalist policies (i.e. is hierarchy helpful), seeing if LoRA finetuning affects performance, both in 4.2 and 4.3, etc. Another suggestion is to stratify the tasks, like contact rich tasks, mobile manipulation tasks, etc. and see performance profiles across different tasks.

### Questions
See weaknesses and questions above.

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
This work introduces RoboCasa365, a simulation benchmark for household robotic manipulation tasks. The benchmark is an extension of RoboCasa, and introduces 365 everyday tasks across 2500 diverse kitchen environments. The authors use this benchmark to evaluate multi-task learning, foundation policy model training, and lifelong learning. Extensive experiment results on the proposed benchmark provide insights into how task diversity, dataset scale, and environment variation influence generalization in robot manipulation.

### Strengths
1. The additional assets and scenes significantly increases the task diversity of the original RoboCasa benchmark. The simulated kitchen environments have highly realistic and diverse layouts; and the authors ensure the task activities are from a diverse set of categories.

2. The paper presents systematic experiments to study key factors in training generalist robot policies. The authors have dedicated effort into implementing various policy training baselines and compared results on both seen and unseen tasks. 

3. Good presentation quality and easy-to-follow writing. Especially sections 3 and 4 are well-organized and provide a good amount of high-level and low-level details about both the dataset statistics and experiment settings.

### Weaknesses
1. Lack of real world evaluations. Although a diverse simulation task suite provides many opportunities for studying algorithms, without a real world digital twin or policy transfer evaluation, the value of the benchmark is limited and it's unclear if any assumptions or implementation bugs in simulation environments would hinder transfer to the real world.

2. Lack of qualitative results. The submission did not include any supplementary materials. For policy learning, it would have been much clearer to show videos of the successful policy rollouts and failure modes. For the task scene designs, it would also improve clarity if the authors can provide more renderings from more camera views into the scenes. 
 
3. In Table 4, the authors claimed "Comparing the Human50 and Human300 settings, we see that increasing the number of pretraining tasks enables a significant improvement in downstream posttraining" -- but the numbers between 2nd and 3rd columns are not really "significantly" different, e.g. 41.0 vs 41.2, or 26.2 vs 28.7. If increasing the number of pretraining tasks from 50 to 300, which is a big change, still yields very minimal performance gains, it makes it questionable whether the training and testing tasks are setup to have enough diversity and/or correlation.

### Questions
1. How are the robot arm controllers implemented, and what's the control frequency? What's the action space -- is it end effector poses and gripper open/close?

2. How is the task conditioning implemented in multi-task policy training? The paper mentioned it's language-conditioned, but is there more investigations on how to embed the language inputs or alternative ways to do task conditioning.

### Soundness
2

### Presentation
2

### Contribution
2
