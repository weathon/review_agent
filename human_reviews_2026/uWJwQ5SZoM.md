# Robust Fine-tuning of Vision-Language-Action Robot Policies via Parameter Merging

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
Generalist robot policies, trained on large and diverse datasets, have demonstrated the ability to generalize across a wide spectrum of behaviors, enabling a single policy to act in varied real-world environments. However, they still fall short on new tasks not covered in the training data. When finetuned on limited demonstrations of a new task, these policies often overfit to the specific demonstrations---not only losing their prior abilities to solve a wide variety of generalist tasks but also failing to generalize within the new task itself. In this work, we aim to develop a method that preserves the generalization capabilities of the generalist policy during finetuning, allowing a single policy to robustly incorporate a new skill into its repertoire. Our goal is a single policy that both learns to generalize to variations of the new task and retains the broad competencies gained from pretraining. We show that this can be achieved through a simple yet effective strategy: interpolating the weights of a finetuned model with that of the pretrained model. We show, across extensive simulated and real-world experiments, that such model merging produces a single model that inherits the generalist abilities of the base model and learns to solve the new task robustly, outperforming both the pretrained and finetuned model on out-of-distribution variations of the new task. Moreover, we show that model merging enables continual acquisition of new skills in a lifelong learning setting, without sacrificing previously learned generalist abilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical challenge in robot learning: finetuning large, generalist Vision-Language-Action (VLA) policies on new tasks with limited demonstration data often leads to catastrophic forgetting of prior skills and poor generalization on the new task itself. The authors propose a surprisingly simple yet effective method called RETAIN (Robust finE-tuning with parameter mergINg). The core idea is to linearly interpolate the weights of the pretrained generalist policy and the policy after it has been finetuned on a small dataset for the new task. The authors evaluate RETAIN across both simulated (LIBERO) and real-world (DROID) environments, using a pretrained π₀-style policy. Their evaluation framework rigorously distinguishes between three settings: 1) in-distribution (ID) performance on the new task, 2) OOD performance on variations of the new task, and 3) performance on the original "generalist" tasks. The results demonstrate that RETAIN significantly improves OOD generalization on the new task compared to standard finetuning methods (Task-FT, Co-FT, LoRA) while successfully retaining the policy's original generalist capabilities. The paper further analyzes modality-specific merging, finding that merging the language model parameters is often most critical , and shows initial promise for using RETAIN in a sequential, continual learning setting.

### Strengths
- The proposed method is exceptionally simple to implement, requiring only a weighted average of two model checkpoints. This lack of complexity is a major advantage for practical application, as it incurs no additional training or inference cost and can be easily integrated into existing workflows.
- The paper tackles a highly relevant and pressing issue. As the field moves towards large foundation models, the ability to efficiently and robustly adapt them to new, specific tasks with minimal data is a key bottleneck for real-world deployment. The paper's motivation is strong and clearly articulated with empirical evidence of overfitting in standard methods (Figure 2).
- The experimental design is a significant strength. The use of both simulation (LIBERO) and real-world robots (DROID) provides compelling evidence. The explicit differentiation between ID, OOD, and Generalist performance allows for a nuanced analysis that directly supports the paper's core claims about improving robustness and retaining prior knowledge. This evaluation structure is a practical application of the principles advocated by more formal taxonomies like STAR-Gen.
- The finding that the LLM backbone parameters are the most influential for merging  provides a deeper insight into the functioning of VLAs. It suggests that the generalizable, abstract knowledge is heavily encoded in the language-centric weights, which aligns with the broader understanding that VLM pretraining is a primary source of semantic reasoning.

### Weaknesses
- While the paper demonstrates that parameter merging works, it offers limited insight into why it works so effectively in the non-convex loss landscape of deep neural networks. The approach is presented as an empirical observation. A more rigorous discussion, potentially drawing from literature on model soups, loss landscapes, and mode connectivity in NLP/CV, would significantly strengthen the paper. For instance, does merging find a wider, flatter minimum between the pretrained and finetuned solutions, which is known to correspond to better generalization? This lack of theoretical grounding leaves the method feeling more like a "trick" than a principled technique.
- The merging coefficient $\alpha$ is the method's key hyperparameter, and its selection is not adequately addressed. The authors state they swept over values to find the best-performing ones for their experiments. This is not practical for a real-world user who may not have the resources to perform extensive OOD evaluation to tune $\alpha$ for every new task. How should a user select $\alpha$ without access to a diverse OOD validation set? The paper does not propose a heuristic or a low-cost method for estimating a good value. The optimal $\alpha$ values differ significantly between the LIBERO (often 0.8-0.9) and DROID (0.5-0.75) experiments. The paper attributes lower OOD improvement in LIBERO to a less capable base model, but does not explain why this would lead to a different optimal merging ratio. This suggests that the optimal $\alpha$ may depend on the pretrained model's quality, the finetuning dataset size, and the task similarity, making it difficult to set a priori.
- The paper claims that RETAIN "enables continual acquisition of new skills in a lifelong learning setting". However, the supporting experiment involves only two sequential tasks. This is insufficient evidence for robust continual learning.

### Questions
- The paper reports success rates but lacks a qualitative analysis of failure cases. When the RETAIN policy fails, how does its failure mode differ from the base policy or the fully finetuned policy? Does it produce more conservative, "safer" failures by reverting to the generalist model's behavior, or does it produce novel, unpredictable failures stemming from the weight interpolation? This analysis would provide valuable insight into the method's reliability.
- How does the method perform after 5, 10, or N tasks? Does performance on earlier tasks degrade as more skills are merged? The proposed iterative merging process ($\tilde{\theta}_{n}=(1-\alpha)\cdot\tilde{\theta}_{n-1}+\alpha\cdot\theta_{ft,n}$) might dilute the knowledge of the original pretrained model and early tasks over time.
- A key finding from the STAR-Gen benchmark was that VLAs struggle significantly with semantic generalization (e.g., novel instructions for the same behavior). Does RETAIN improve robustness to linguistic variations, or does it primarily bolster visual robustness? How does the method handle changes to unobservable physical properties (e.g., mass, friction), another axis in STAR-Gen?
- Does a larger or more diverse finetuning dataset generally permit a higher $\alpha$ (leaning more towards the specialist model), whereas a smaller, narrower dataset benefits from a lower $\alpha$ to better preserve the generalist priors?
- Your choice of linear interpolation is compelling due to its simplicity. I am curious whether you experimented with or considered other parameter merging techniques from the broader deep learning literature. For example, methods like task arithmetic (where task vectors are computed and added/subtracted) have shown success in NLP. Was linear interpolation the first and most obvious choice, or did you find it superior to other, perhaps more complex, merging schemes for VLA policies?

### Soundness
3

### Presentation
4

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
This work focuses on VLA fine-tuning, and proposes model merging to induce robustness and mitigate forgetting. Standard fine-tuning on reasonably-sized single-task datasets (i.e. ~100 demonstrations) are known to improve performance on the specific tasks, but degrade it elsewhere (as confirmed by the authors in Figure 2); standard solutions (such as co-training or LORA) tend to only marginally alleviate this issue, or require access to the pre-training dataset. The authors propose a simple linear interpolation strategy between pre-trained an fine-tuned weights, which may optionally specialize to the three sub-networks (vision encoder, language backbone and action head). This technique can be applied in a single fine-tuning round, or over multiple rounds in a continual learning setting. Authors claim that model merging preserves the generalist performance of the VLA, while also boosting OOD performance on the fine-tuning tasks. Their claims are supported by experiments spanning real (DROID) and simulated (LIBERO) environments. The authors include a straightforward ablation of the interpolation parameter $\alpha$.

### Strengths
- Experimental evaluation is extensive, involving both simulated and real-world experiments. Moreover the experiments are described in great detail.
- The problem tackled by this work is timely has clear practical interest.
- This paper is written in a clear and understandable way.
- The evaluation of different merging coefficient for different components is interesting on its own.

### Weaknesses
- The main weakness of this work is that it is not entirely clear whether the overfitting issue arises due to sub-optimal hyperparameter choices. Figure 2 shows that performance of baselines such as Co-FT fluctuates and decays for a large number of gradient steps: at 500 gradient steps both ID and OOD performance on the new task is high, and generalist performance has not decayed. The supplementary material details that the number of gradient steps was chosen for each algorithm/environment combination, but the tuning process is not clear.
- The performance gains from model merging are often minor, and the method does not consistently beat baselines such as Co-FT (e.g. in continual learning settings). More importantly, these performance gains might be due to precise tuning of $\alpha$, which is set on a per-task basis. It is unclear how well the method would perform out of the box, without extensive tuning of $\alpha$, which might be prohibitive in real-world settings.

A few minor issues will be detailed in the following section.

### Questions
### Minor issues
- Model merging is not an entirely novel idea, as the authors acknowledge in the related works. I would suggest adding a couple of references in the introduction as well, which does not point to previous work in model merging. This is not a major weakness, as the authors acknowledge existing works, and focus on empirical evaluations in a multi-modal setting, which is to the best of my knowledge under investigated.
- Line 143: dataset tuples should be $(s_t^{(i)}, a_t^{(i)}, T^{i})$ to be consistent with the notation in Equation 1.
- Line 196 is later contradicted: the model does not improve monotonically the longer it is trained, even ID.
- "mugs" is not formatted correctly in the caption of Table 14.

### Questions
- Did the authors consider regularization to the pre-trained model (i.e. distillation) as an option to prevent model drift? This is for instance done to prevent response collapse when fine-tuning LLMs.
- What do error bars represent in Figures 6 and 7?

### Conclusion
Overall, the paper makes an interesting contribution to an open problem, and I currently recommend acceptance. My main concerns revolve around the experimental settings, and the modest gain in performance with respect to Co-FT. These concerns are outweighed by the fact that (a) experiments are well described, and thus different training regimes can be explored in future work and (b) the strongest baselines require access to pre-training data, while model merging does not. Nonetheless, clarifying whether the training regimes considered are already designed to avoid as much overfitting as possible would improve this work's significance.

### Soundness
3

### Presentation
4

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
This paper addresses the overfitting of generalist robot policies during small-scale dataset finetuning, a process that compromises pre-trained knowledge and inhibits generalization to OOD tasks. The proposed method, RETAIN, mitigates this by merging the parameter spaces of the pre-trained and finetuned policies via linear interpolation. Experimental results indicate that RETAIN achieves robust generalization to novel tasks while preserving broad, general capabilities. The paper also provides a comparative analysis against prior strategies, offering specific insights into weight merging with different parameter groups.

### Strengths
- The paper addresses a critical and practical problem in robotics: how to adapt generalist policies to new tasks with limited data without overfitting or forgetting.
- The proposed method, RETAIN, is simple and efficient, relying on parameter interpolation with no added inference cost.
- The authors provide a valuable comparative analysis against several prior finetuning strategies, offering significant insights to the research community.
- The evaluation protocol is clear and effectively highlights the core problem.

### Weaknesses
1. The selection of the $\alpha$ merging coefficient is unclear and appears to be highly task-dependent (from Appendix A.5). The paper does not offer a methodology or heuristic for tuning $\alpha$ based on specific task characteristics.
2. The insightful analysis on modality-specific merging (Sec 6.5) seems limited to simulation experiments. It is unclear if the conclusion that only LLM parameters need merging holds for the more complex, real-world DROID tasks.
3. The continual learning claims (Sec 6.6) are partially undermined by an unexplained anomaly in Figure 10, where the co-FT baseline achieves surprisingly high OOD performance on the second task, which the authors acknowledge but do not investigate.

### Questions
- For weakness 1: What is the recommended procedure for selecting $\alpha$ in a practical? Can $\alpha$ be estimated from the data?
- For weakness 2: Was the modality-specific merging analysis (i.e., merging only LLM parameters) validated on the real-world DROID tasks? Does the same finding hold?
- For weakness 3: Do the authors have a hypothesis for the anomalous co-FT result in the continual learning experiment (Fig 10)? Could this be due to a specific relationship between the $\textit{plates}$ and $\textit{whiteboard}$ tasks?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a very simple strategy for finetuning of policies by merging model parameters of pretrained models and finetuned models. The work presents empirical results on Droid robot setup and Libero simulations and find a surprising effectivity of this method in achieving task performance in the fine tuned task while retaining performance on the original (generalists) tasks as well as generalization to variations of the fine-tuning task.

### Strengths
- simple method
- extensive experiments

### Weaknesses
- no deeper understanding
- same unclear results and overly strong claims
- questionable hyperparameter selection

Because the method is so simple and not really clear, from a fundamental perspective, why it should have the presented favorable properties, extra care has to be taken in conducting the experiments and interpreting the results. See my questions below.

### Questions
I know that your paper is not trying to give a deeper understanding why the simple model merging does the right thing. Nevertheless, I would like to know if you have analyzed the following:
- we do a linear interpolation in weight space. 
- if the fine-tuning training path in weight space would be more or less linear, then simply training with a lower learning rate or less steps should have the same effect. 
Here the actual questions:
Q1a: Have you ablated for learning rate / number of updates steps. 
Q1b: I think a non-linearity check for the gradient path would be interesting: compare the vector of $\theta_0$ to $\theta_{n/2}$ with $\theta_{n/2}$ to $\theta_{n}$ in terms of collinearity. 

Q2: The results in the paper appear cherry picked in comparison to all results, e.g. presented in Fig. 12.
For those tasks, Co-FT and RETAIN-Co-FT are very similar here (sometimes one or the other wins). 
I suggest creating a more balanced representation in the main paper and tone down your claims. 

Q3: Sec. 6.5: How was the $\alpha$ parameter selected?  I understood that $\alpha$ (I guess the same for all modalities) is selected based on the best OOD test performance. This is against the Machine Learning practice! Never select hyperparameters on the test set! Maybe I am mistaken here, please clarify.

Q4: Which $\alpha$ parameter did you use for the DROID experiments? How was it selected?
Also, did you use different $\alpha$s for the modalities in the experiments, and how did you select this weighting?

Q5: How does alpha depend on training parameters (how many updates) etc?

Q6: Sec 6.6: What would happen if you finetune on both / all sequential tasks at once. As you do Co-FT, you anyway have all the data available, so I think that would be a good comparison / upper baseline. 

Q7: for the sequential learning, what would you expect if one finetunes the base model on both tasks and then merges both together, as a weighted sum. This would allow parallel improvement and sounds like an interesting use case. 

Comments:
- You have one or three additional hyperparameters (see Q4), and depending on their selection, I am not so surprised to see that you can find a setting that works well in your experiments, the question is: will these hp settings also work for a new environment?

Related work: also a paper that looks into finetuning from few demonstrations:
Active Fine-Tuning of Multi-Task Policies, Bagatella et al
This paper does not do model merging but learned output merging and might be interesting to related to.

### Soundness
2

### Presentation
4

### Contribution
3
