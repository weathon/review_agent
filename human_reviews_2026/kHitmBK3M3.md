# Assembly-R1: 3D Assembly Reasoning via RL-based Vision Language Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Part assembly requires agents to possess precise spatial interpretation and multi-step structural reasoning. While large Vision-Language Models (VLMs) have shown promising capabilities in general Visual Question Answering (VQA), existing benchmarks inadequately reflect the complexities inherent in assembly reasoning. To bridge this gap, we introduce FurniBench, an assembly-specific VQA benchmark, coupled with FurniQA, a large-scale dataset targeting part recognition, connectivity reasoning, step planning, etc. Using Qwen2-VL-2B-Instruct as a base model, with 39.1% accuracy on FurniBench, we first establish a supervised fine-tuning (SFT) baseline, which highlights both the benefits and the limitations of SFT in this domain. Building on this, we propose Assembly-R1, a model trained via Group Relative Policy Optimization (GRPO). With enhanced reasoning capabilities, Assembly-R1 achieves an accuracy of 73.6%, outperforming other baselines on FurniBench by a large margin. Furthermore, the consistent gain on zero-shot Out-of-Domain (OOD) spatial understanding and Embodied AI benchmarks indicates that Assembly-R1 acquires transferable spatial skills applicable to broader Embodied AI scenarios. This work establishes FurniBench as a critical resource for both diagnosing deficits in current VLMs and teaching the fundamental structural reasoning required for downstream applications. We will release the dataset and code upon acceptance of this paper.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FurniBench, a benchmark for Visual Question Answering (VQA) in 3D assembly tasks, along with FurniQA, a dataset containing 1.6M QA pairs derived from the IKEA ASM dataset. The authors establish a supervised fine-tuning (SFT) baseline (Assembly-V1) and propose Assembly-R1, trained using Group Relative Policy Optimization (GRPO).

### Strengths
1. The motivation for creating a benchmark in 3D assembly reasoning is valid, as general VQA benchmarks often lack the required complexity in structural and procedural understanding.
2. The approaches to reduce bias and subjectivity in dataset construction are reasonable. 
3. The GRPO-based Assembly-R1 shows significant improvements over both the base model (+32.6%) and SFT baseline (+7.6%).
4. The out-of-domain evaluation on CVBench (Table 2) demonstrates the generalization ability brought by RL.

### Weaknesses
1. While FurniBench claims to assess 3D structural and spatial reasoning, the tasks seem to primarily focus on 2D part recognition and sequence inference. Key 3D spatial challenges like reasoning about occlusion, perspective changes, or spatial relationships are not explicitly evaluated. Many tasks could potentially be solved using 2D visual cues and logic, without requiring 3D understanding.
2. The "Part Connectivity" category shows minimal improvement (28.0% → 32.0%), suggesting the model may still struggle with 3D structural relationships.
3. Only one base model architecture (Qwen2-VL-2B) is fine-tuned. It would be valuable to see results on other architectures to demonstrate the general applicability. Furthermore, the paper does not compare with other reasoning enhancement techniques, making it difficult to assess its relative advantages.

### Questions
1. Why is the depth information in the IKEA ASM dataset not utilized in dataset construction?
2. Could you provide more detailed error analysis, particularly for the "Part Connectivity" tasks where improvement is minimal?
3. Given Weakness 1, could you identify a subset of questions that are most indicative of 3D spatial reasoning (i.e., those least solvable by 2D cues alone)? Reporting the model's performance specifically on this challenging subset would provide a clearer measure of its true 3D capabilities
4. The dataset construction mentions that the questions are "manually calibrated." Could you provide more details on this process, such as the calibration procedure and the total annotation time involved?

### Soundness
2

### Presentation
2

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
1. This paper introduces FurniBench, a benchmark for 3D part-assembly visual question answering (VQA), and its associated dataset FurniQA, which contains 1.6 M QA pairs generated from the IKEA-ASM dataset.

2. It also proposes Assembly-R1, a reasoning-enhanced Vision-Language Model (VLM) trained using Group Relative Policy Optimization (GRPO), inspired by DeepSeek-R1, to improve structural and spatial reasoning.

3. Compared with the base model (Qwen2-VL-2B-Instruct, 39.1 %) and an SFT baseline (Assembly-V1, 64.9 %), the GRPO-based model reaches 71.7 % accuracy on FurniBench and 63.5 % on CVBench, showing that reinforcement learning can strengthen reasoning generalization.

### Strengths
1. Novel benchmark contribution: FurniBench fills an under-explored gap in 3D part-assembly reasoning for VQA, offering a structured taxonomy (part recognition, connectivity, and assembly reasoning).

2. Methodological clarity: The paper provides clear formulation of supervised (SFT) and RL (GRPO) stages and the trainig pipelines, and quantitative ablations demonstrate measurable gains, showing how the .

3. Empirical completeness: Experiments include both in-domain and out-of-domain evaluations (on CV-BENCH), along with reward-hacking analysis, adding credibility to the claims that “SFT memorizes, RL generalizes.”

### Weaknesses
1. While the results of baseline models (e.g. Gemini 2.5 Pro, GPT-4o,) are shown in the Figure 3, it would be clearer and more informative to include the numerical baseline performance in Table 1 for easy reference. This would help readers quickly grasp the comparative results on the new benchmark. In addition, the accuracy curve plot (Figure 3) currently provides limited insight and could be moved to the appendix to make space for more comprehensive quantitative tables in the main text.

2. Lack of human evaluation or qualitative validation: No human study or inter-annotator validation is provided to verify the realism or difficulty of the generated QA pairs.

### Questions
1. Benchmark Clarity and Statistics. The benchmark design section is quite detailed, but it would be helpful to include statistical visualizations (e.g., distributions of task types, question difficulty levels, or part categories). Could the authors provide these quantitative breakdowns to better illustrate the dataset’s novelty and coverage?

2. How do the authors believe the CoT process contributes to part recognition rather than simply restating the answer?
For example, in Figure 4(c), the reasoning step “\<think\> The label Z corresponds to the table legs. \</think\>” appears to directly repeat the answer rather than providing additional reasoning.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper makes a well-executed application of VLMs and RL post-training on the 3D assembly tasks. The authors first formulate the task well, and then introduce a corresponding benchmark FurniBench, together with a large-scale dataset (FurniQA) for this task. To proceed, the authors perform SFT and GRPO based on pretrained Qwen2-VL-2B, resulting in two model variants: Assembly-V1 and Assembly-R1. The evaluation results demonstrate that GRPO can significantly enhance model performance, establishing a solid baseline for future research in 3D assembly tasks.

### Strengths
- Applying frontier methods (VLMs and RL post-training) to the 3D assembly task is a good practice to promote the development of AI-related areas
- This paper delivers a complete story, including problem formulation, benchmark design, dataset construction, model training, and experiments. The overall frame is clean and sound.
- Good analysis of OOD results and reward hacking, which manifests the advantages and more insights regarding the RL post-training of Assembly-R1

### Weaknesses
- Limited experimental results, particularly limited baselines for comparison. This makes it infeasible to assess the contribution and superiority of this paper. It turns out to be a story that we do something and the results are here. More potential ablation studies into the method are desired. For example, given the effectiveness of GRPO, how would different reward designs affect the model performance?
- I also have some concerns regarding the impact of this paper. With limited novelty, this work is more like establishing a solid baseline for future research on this task. How could the 3D assembly task relate to other domains with more popularity, e.g., embodied AI and robotics?

### Questions
- Why not continue RL post-training based on Assembly-V1? It seems the two model variants are independent (see Figure 3), which is not a common practice.
- In the analysis of reward hacking (Figure 5), what helps eliminate the phenomenon of reward hacking? The authors explained that “the reward design does not favor long reasoning“. But I think it is insufficient to account for the significant decrease in reward hacking since there is no explicit penalty for this. In other words, high reward hacking may not conflict with high accuracy.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Assembly-R1, a reinforcement learning–enhanced vision-language model (VLM) designed for 3D assembly reasoning tasks. To evaluate model performance in this domain, the authors construct FurniBench, an assembly-specific benchmark, and FurniQA, a dataset of ~1.6 million VQA pairs covering tasks such as part recognition, connectivity reasoning, and next-step inference.
Using Qwen2-VL-2B-Instruct as the base model, the authors first train a supervised fine-tuned version (Assembly-V1) and then apply Group Relative Policy Optimization (GRPO) with simple format + accuracy rewards to produce Assembly-R1, which achieves 71.7 % accuracy on FurniBench versus 64.9 % (SFT) and 39.1 % (base). The RL-trained model also shows slightly better out-of-domain performance on CVBench (63.5 % vs 62.4 %) and demonstrates more coherent chain-of-thought reasoning.

### Strengths
- Timely and relevant problem. The paper addresses 3D structural reasoning for assembly tasks, an underexplored but important domain bridging robotics, visual reasoning, and multimodal understanding.

- Dataset contribution. The creation of FurniBench and FurniQA is valuable. The dataset is large, well-organized, and includes thoughtful bias-mitigation strategies (dynamic part tagging, multiple valid answers, randomized options).

- Clear connection to recent RL-for-reasoning work. The use of GRPO builds on DeepSeek-R1-style frameworks, demonstrating that such techniques can transfer to 3D spatial reasoning.

### Weaknesses
- Limited novelty. The conceptual contribution is small. The method is a direct application of GRPO with two basic rewards (format + accuracy) to a new dataset. There is no significant methodological innovation—no new RL algorithm, architecture, or reward-design insight.

- Narrow empirical scope. Evaluation is confined to a single base model (Qwen2-VL-2B) and a synthetic IKEA-style dataset. No validation is shown on more complex 3D or real-world tasks (e.g., ShapeNet, ScanNet, or robotics demonstrations).

- Marginal out-of-domain gain. The OOD improvement on CVBench is only +1 %. This small gain weakens the claim that “RL generalizes while SFT memorizes.”

- Simplistic reward design. The reward formulation (+1 for format, +1 for correct answer) is too coarse to claim genuine reasoning improvement. There’s no evidence that RL truly learns 3D structural relationships rather than better formatting behavior.

- Missing ablations. The paper lacks sensitivity analyses for reward scaling, sampling temperature, or GRPO hyper-parameters. It also doesn’t isolate whether improvements come from GRPO itself or just more fine-tuning.

- Evaluation metric limitations. Accuracy on multiple-choice questions is a weak proxy for 3D reasoning. Human evaluation or geometric-consistency metrics (e.g., spatial error, collision rate) would provide stronger evidence.

- Dataset and task design risk overfitting. Since FurniQA questions are rule-generated from the same IKEA-ASM dataset, it is unclear how much semantic diversity truly exists. The model may learn annotation heuristics rather than 3D reasoning.

### Questions
- How do you ensure that GRPO improves genuine reasoning rather than simply response formatting?

- Why limit training to 15 k samples from FurniQA given that 1.6 M are available? Does performance saturate beyond this size?

- Did you compare Assembly-R1 with other RL methods (e.g., DPO, RLAIF) or analyze stability under different reward weights?

- Can the trained model handle open-ended free-text answers beyond multiple-choice formats?

- How reproducible are the results? The appendix lists configurations, but is code or dataset access planned before acceptance?

### Soundness
3

### Presentation
2

### Contribution
2
