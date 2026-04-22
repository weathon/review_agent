# Hybrid Training for Vision-Language-Action Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). 
As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. 
However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while  enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Hybrid Training (HyT), a framework that lets VLAs learn from CoT reasoning for better performance while allowing inference without generating thoughts. HyT flexibly enables direct action prediction, thought generation, or instruction following, and shows strong results in both simulation and real-world experiments.

### Strengths
1. Performance-speed breakthrough: HyT resolves the key “reasoning vs. reflex” trade-off in VLAs, matching the success rates of CoT methods while maintaining high inference speed (e.g., 3 Hz vs. 1 Hz for ECoT)—a major step toward practical deployment.

2. Robust empirical results: Extensive experiments show HyT’s clear advantages: strong data-scaling on ClevrSkills, state-of-the-art results on LIBERO (especially “Goal” and “Long” tasks), and real-world robot transfer. Its 54% OOD success vs. 29% baseline highlights superior generalization.

3. Versatile and interpretable model: A single HyT-trained model supports fast “act,” interpretable “think,” and guided “follow” modes—allowing deployment, analysis, and human-robot collaboration without retraining.

### Weaknesses
1.  HyT requires CoT or reasoning traces for training—using oracle thoughts on ClevrSkills and Gemma-2 9B-generated plans on LIBERO. This reliance on costly, curated reasoning data limits scalability and makes adaptation to new domains harder than learning directly from demonstrations.

2. While “act” mode matches “think” mode performance, the paper doesn’t test harder, long-horizon, or abstract tasks where explicit reasoning might still be needed. Without such analysis, the boundaries of HyT’s internalized reasoning remain uncertain.

### Questions
All of my qeustions are listed in the weakness section. If my concerns are well addressed, I will raise my rating.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Hybrid Training (HyT), a method that unifies thinking and acting behaviors in a single vision-language-action model via a learnable modality variable controlling “think/act/follow” modes. HyT trains one model to generate both reasoning traces and executable actions, using a Monte-Carlo sampling scheme to mix modalities during training. The method is evaluated across simulated (ClevrSkills, LIBERO) and real-robot (xArm6) environments, demonstrating competitive task success and faster inference compared to full chain-of-thought reasoning.

### Strengths
1. HyT uses a modality variable to avoid maintaining separate reasoning and acting modules, simplifying deployment and enabling run-time mode control (think/act/follow) without model-switching.
2. HyT shows good performance on the ClevrSkills benchmark, suggesting HyT can learn complex, compositional behaviors required there. And robotic experiments on an xArm6 support practical applicability beyond simulation.

### Weaknesses
1. In Figure 4, reported task success and efficiency gains relative to ECoT are small; it remains unclear whether HyT achieves comparable performance at lower latency or merely approximates ECoT under similar compute.
2. How does HyT compare to the π₀.₅ model (https://github.com/Physical-Intelligence/openpi) and OneTwoVLA (https://arxiv.org/pdf/2505.11917)? Can the BOA/BOR control mechanism (as in OneTwoVLA) achieve similar adaptive switching without retraining?

### Questions
See Weaknesses.

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
This paper introduces Hybrid Training (HyT), a framework for training Vision-Language-Action models that enables learning from chain-of-thought (CoT) reasoning traces while maintaining fast inference speeds. The key insight is that VLAs can internalize knowledge from thought supervision during training without needing to generate thoughts at test time. HyT trains a single model to conditionally predict three distributions based on a modality variable: direct actions ("act"), thoughts then actions ("think"), and following provided instructions ("follow"). The method uses Monte Carlo sampling during training for the three modes. Experiments on ClevrSkills, LIBERO, and real-world tasks demonstrate that HyT achieves performance gains similar to ECoT while maintaining standard VLA inference speeds (3Hz vs 1Hz for ECoT).

### Strengths
- The hypothesis that models can internalize CoT reasoning without explicit generation is compelling and well-articulated through the System 1/System 2 cognition analogy.
- HyT outperforms competitive baselines including MolmoAct (86.6% → 93.7%) and π0-FAST (85.5% → 93.7%) on LIBERO, while maintaining 3× faster inference than ECoT.
- The paper includes extensive experiments across simulated (ClevrSkills, LIBERO) and real-world environments with ablations on data scaling.
- The ability to maintain ECoT-level performance at standard VLA inference speeds addresses a critical deployment constraint for robotic systems.

### Weaknesses
- The loss coefficients (wa:0.25, wτ:0.5, wf:0.25) appear arbitrary without ablation
- The Monte Carlo sampling approach is not compared to direct weighted loss computation
- Individual contributions of Lfollow and Lthink to overall performance are not analyzed
- HiRobot with oracle thoughts often outperforms HyT with oracle, suggesting the "follow" mode may not be optimally implemented. 
- The paper lacks systematic investigation of which task characteristics benefit most from internalized CoT training versus standard VLA training.
- Unclear if all baselines use the same VLM backbone (PaliGemma-2)
- No inference speed comparison with π0-FAST (or other baselines)
- Compatibility with other VLA architectures not explored

### Questions
- Why does HiRobot+Oracle outperform HyT+Oracle in Figure 4?
- How were the loss coefficients (0.25, 0.5, 0.25) determined? What is the sensitivity to these values?
- What is the ablation comparing Monte Carlo sampling versus direct weighted loss computation? Does the sampling variance help or is it just for computational efficiency?
- Can you provide ablations showing individual contributions of Lthink and Lfollow? What happens with just Lact + Lthink?
- What are the specific inference speeds compared to π0-FAST, and can HyT be combined with their efficient tokenization approach?
- In what types of tasks (complexity, horizon length, reasoning requirements) does HyT show the largest improvements over standard VLA training?
- Do all compared baselines (Table in LIBERO results) use the same PaliGemma-2 backbone for fair comparison?

### Soundness
3

### Presentation
4

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
This paper proposes that the benefits of chain-of-thought (CoT) reasoning in VLAs can be reaped without the very slow inference times that generating reasoning chains at test-time requires. Namely, they propose a method called HyT (hybrid training) in which they train the VLA to exhibit multiple modes of behaviors, either predicting actions directly, predicting a reasoning chain and then actions, or predicting actions from an oracle human-provided reasoning chain, with these modes selectable by configuring a modality variable. This hybrid training scheme allows the model to effectively learn from the chain-of-thought data at train time, not having to actually generate these reasoning chains at test-time to still get equally good performance. The authors test their approach against a standard VLA, a reasoning VLA, and a hierarchal VLA on the ClevrSkills simulated benchmark and find that their approach performs the best across multiple dataset sizes. On the Libero simulated benchmark, there approach, adapted off of the OFT-VLA model, performs the best when compared to many prior VLAs on Libero. And finally on eight tasks in the real-world, their model exhibits better performance than a standard VLA while requiring the same inference time (~3 Hz).

### Strengths
(1) The authors tackle a very relevant problem in robotics, and arrive at a nice result in that the benefits of reasoning trace generation can be obtained without having to generate expensive reasoning chains at test time. The proposed hybrid training strategy is general and permits different manners of either conditioning on or not conditioning on reasoning at test time.

(2) The experimental results are reasonably through, and include real-world results, which is nice. In particular, the authors’ method scores the highest on the Libero simulated benchmark when compared to other fine-tuned VLAs.

(3) The paper is clear and easy to read.

### Weaknesses
(1) The analysis of the results could be improved. In particular, it remains unclear what the specific reason is that reasoning trace generation can be avoided at test-time, but yet the benefits of reasoning can still be obtained. In their discussion section the authors wrote that “from analyzing the HyT framework, our understanding is that learning to generate CoT and learning to predict actions from CoT improves the agent’s understanding of the environment” —> could you expand on what it mean to improve the understanding of the environment? I.e., is it that the model can obtain better internal representations if trained with reasoning, or can follow language better, or something else?

(2) Reasoning is perhaps the most appealing when the task involves some complex decision making/logical inference before picking what atomic action to take. It seems that the authors missed trying their approach out on these types of tasks. Indeed from the discussion: “we do not exclude that thoughts generation might be useful at test-time in more complex settings, but this would require evaluating on tasks that require advanced reasoning capabilities, which are currently sparse and rarely adopted in the robotics literature”. It would have been interesting to see whether the hybrid training scheme can bring benefits to these types of tasks as well, or if reasoning at test-time becomes strictly necessary.

### Questions
(1) For the three different modes, the authors chose weights 0.25 for act, 0.5 for think, and 0.25 for follow. Did they ablate different values of these weights? Generally how important is a good selection of these weights to model performance? Which of the three modes matter the most/least?

(2) HiRobot + Oracle seems to be doing the best, as shown in Figure 4. Am I correct in assuming that the main difference between HiRobot and the proposed approach is that two separate PaliGemma models are used? What leads to the improved performance? More capacity? Less feature sharing?

(3) Why do you think ECoT performed worse than the proposed approach? They are very similar, and it would seem that ECoT should perform slightly better since it generates reasoning at test-time?

(4) Why aren’t any of the ClevrSkills baselines compared against in Libero? E.g., ECoT or HiRobot?

(5) For the real-world tasks, what do the reasoning traces look like and how were they generated? Were the types of reasoning traces generated such that reasoning might be particularly useful in the OOD setting?

(6) There appears to be substantial overlap with a contemporaneous, peer-reviewed publication ([1], accepted Aug 1, 2025). Per the ICLR 2026 reviewer FAQ, I will not penalize the paper for missing a citation or comparison to contemporaneous work. However, as written, the submission does not clearly articulate contributions beyond that work. It would be great if the authors could clarify what is genuinely new here (theory, method, or empirical evidence) beyond this prior work.

[1] Chen, W., Belkhale, S., Mirchandani, S., Mees, O., Driess, D., Pertsch, K., & Levine, S. (2025). Training Strategies for Efficient Embodied Reasoning. arXiv preprint arXiv:2505.08243.

### Soundness
2

### Presentation
3

### Contribution
2
