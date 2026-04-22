# MolEditRL: Structure-Preserving Molecular Editing via Discrete Diffusion and Reinforcement Learning

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2, 6

## Abstract
Molecular editing aims to modify a given molecule to optimize desired chemical properties while preserving structural similarity. However, current approaches typically rely on string-based or continuous representations, which fail to adequately capture the discrete, graph-structured nature of molecules, resulting in limited structural fidelity and poor controllability. In this paper, we propose MolEditRL, a molecular editing framework that explicitly integrates structural constraints with precise property optimization. Specifically, MolEditRL consists of two stages: (1) a discrete graph diffusion model pretrained to reconstruct target molecules conditioned on source structures and natural language instructions; (2) an editing-aware reinforcement learning fine-tuning stage that further enhances property alignment and structural preservation by explicitly optimizing editing decisions under graph constraints. For comprehensive evaluation, we construct MolEdit-Instruct, the largest and most property-rich molecular editing dataset, comprising 3 million diverse examples spanning single- and multi-property tasks across 10 chemical attributes. Experimental results demonstrate that MolEditRL significantly outperforms state-of-the-art methods in both property optimization accuracy and structural fidelity, achieving a 74% improvement in editing success rate while using 98% fewer parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces MolEditRL, a model that modifies input molecules to optimized some desired chemical property, which is provided by the user as a text prompt. The MolEditRL framework works by combining a graph diffusion model with a reinforcement learning training strategy, with rewards coming from an oracle (e.g. RDKit) for the chemical property of interest. 
The paper also proposes a novel dataset, MolEdit, tailored for benchmarking molecular editing models with natural language instructions.
The method is benchmarked on MolEdit against baselines from the literature, and shows consistently state of the art performance across several tasks and evaluation metrics.

### Strengths
- The paper is quite well-written, as it clearly states the motivation and the high level idea of the method. Even though I am not familiar with the literature, I could easily understand most of the paper. 
- The empirical results shown in the paper are very impressive, as consistently MolEditRL shows state-of-the-art performance across several tasks and evaluation metrics. 
- The paper shows results not only on their new MolEdit dataset, but also on the C-MuMOInstruct dataset from the literature. Even here, MolEditRL is most of the time the leading model.

### Weaknesses
- Some details about the use of an oracle as a reward are unclear to me. It is unclear whether the model would work for chemical properties for which no oracle is available, or if the oracle can be noisy (e.g. a ML model for function/property prediction)
- Related to the point above, it is unclear to me how the oracle for the reward is chosen. It seems like the oracle should depend on the textual instruction S, but there are no details on how it is selected. 
- A few chemical properties are reported. However, it is unclear how these properties were selected for benchmarking, and if the method shows the same performance on other kinds of properties. 

Minor points:
- In Table 1 some negative vspace seem to have been used too aggressively. 
- In Figure 4a, the choice of using a line plot is questionable, since the x axis is models. A bar plot like  FIg. 5b would have been more appropriate
- The authors should use the extra page to present some results on the generalization to unseen properties and on datasets other than their own, as I think it would strengthen the paper.

### Questions
- Please clarify the issues raised on the oracle. 
- Why are you using textual instructions rather than a categorical variable (e.g. one-hot encoding for supported properties) for conditioning the diffusion?

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
3

### Summary
This paper proposes **MolEditRL**, a framework combining discrete graph-diffusion and reinforcement learning (RL) for structure-aware molecular editing. It first uses a discrete diffusion model to learn to reconstruct molecules given editing instructions and source molecules, and then fine-tunes via an RL objective with property-based rewards and a KL-regularizer to enforce structural fidelity.
Experiments cover a new editing dataset (“MolEdit-Instruct”), single- and multi-property editing, and comparisons to state-of-the-art baselines, showing substantial improvements in editing success, scaffold preservation, and parameter efficiency.

### Strengths
The reviewer acknowledges the following contributions of the paper:

 **Novelty and Technical Contribution** 

- The integration of discrete diffusion and reinforcement learning for molecular editing (not just generation) is an original and well-motivated application.

- The KL-regularized RL fine-tuning introduces a principled mechanism to balance property optimization with structural preservation, addressing the classic trade-off between exploration and validity.

- A particularly notable technical insight is how the authors address the high variance problem of rewards that appear only at the final step of the diffusion trajectory. By rewriting the reverse transition using x_{0}-parameterization and approximating gradients through a reward-weighted cross-entropy loss, the algorithm shows a more stable and computationally efficient estimator.

**Dataset Contribution**

The newly released MolEdit-Instruct dataset fills a key gap for large-scale, instruction-based molecular editing tasks and is a valuable contribution to the research community.

**Breadth of Experiments**

- The authors conduct comprehensive experiments covering single-property, multi-property, and cross-dataset generalization scenarios (both with the proposed dataset and C-MuMOInstruct (appendix)). The methods are compared against a variety of baselines, e.g., Drug-assist.

- The obtained results show gains in both editing success rate and structural similarity to the source molecule.

### Weaknesses
**Incremental Nature of Some Components**:

- While the integration of discrete diffusion and RL is novel, both components themselves (diffusion for graph generation and RL for property optimization) have been individually explored before in molecular design. The novelty thus lies more in **their combination and adaptation** rather than in a fundamentally new paradigm.

- In the meantime, the Reviewer found another paper [1] shares highly similar ideas with this work, i.e., using diffusion to generate a new updated molecule and then regularized by reinforcement by the PPO method. Though it should not count this work as identical [1] because this one was released after the ICLR deadline. Instead, the reviewer suggests that the author include this paper in the related works and discuss it.

**Limited Discussion on Computational Cost and Scalability**

- The paper reports efficiency in terms of parameter count but provides limited analysis of training/inference cost, scaling to larger biomolecules, or real-time editability.

- Reinforcement learning fine-tuning may still be computationally expensive due to oracle calls and multi-step diffusion.

**Ablation Gaps on Graph-Specific Components**

- Although the authors conduct extensive ablations, there is no explicit comparison showing the benefit of the structure-aware attention mechanism versus a standard transformer without graph bias. The structural-aware self-attention is emphasized as one of the selling points in the paper; thus, a deeper analysis of this property is important.

**Clarity and Accessibility**

- The methodology section is dense and mathematically detailed, which could challenge readers from a non-ML background. A higher-level intuition or illustrative example could improve accessibility.

- It would also be beneficial to discuss potential limitations, such as failure cases (e.g., when instructions are ambiguous or contradict structural constraints).

[1] Kaech, Benno, et al. "Refine Drugs, Don't Complete Them: Uniform-Source Discrete Flows for Fragment-Based Drug Discovery." Arxiv 2025

### Questions
Q1. In section 3.4, equation (8), how can the function $r(.)$ be implemented? Is this an oracle by the RDKit tool?

Q2. In section 4.5, the effect of different RL training schemes is discussed. What does this statement mean: "while GDPO leverages **x0-parameterization** to optimize ***only the final output*** " and "MolEditRL introduces KL-regularized optimization ***over the entire diffusion process*** "? 

Does this mean that GDPO is also having the reward for the final step? Then how is GDPO different from MolEditRL in this case, given that both methods apply the same x0-parameterization techniques described in Equations (12-14)?

Other questions: please check my weakness section above regarding running time and the missing ablation study on the structure-aware module inside the transformer module.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a molecular editing framework based on discrete diffusion and reinforcement learning. Specifically, authors design a structure-preserving attention mechanism with a bias term encoding adjacency information, to preserve the structural integrity of the template molecule during editing. The molecular generation process is implemented as a pre-trained discrete diffusion model, which is then fine-tuned with editing success and validity rewards on pair molecular edit data. The framework is then tested on several molecular design tasks involving multiple properties and outperforms the baselines on all metrics.

### Strengths
- The proposed structure-aware attention mechanism offers a potential solution to the complex problem of conditioning the molecular generation with structural constraints.

- The joint encoding of both instruction text and molecular structure achieves good alignment between text description and the molecular space.

- The performance of the proposed framework significantly outperforms all baselines by several tasks and metrics. Also, detailed visualizations and interpretations of the generative results as well as ablation studies are provided.

- The authors also provide a comprehensive dataset for molecular edit paired covering a wide range of properties.

### Weaknesses
- The RL setting relies on ready-to-use and fast property calculations, so generalizability to rare or unseen properties is limited (especially for protein binding).

- Some specific questions on methods and clarity. See Questions

### Questions
- Eq3: in the early denoising steps, as $E_{tgt}$ is inaccurate and noisy, how to prevent it from misinforming the denoising of the atoms?

- Since the language used for the editing instruction is well-structured and only the property type (and potential numbers) matter, how much does the language model backbone really contribute?

- Is the structure-preserving behavior more achieved by the structure-aware attention, or the explicit KL divergence objective during RL? I.e. what are the raw similarity values in the ablation study results?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces MolEditRL, a framework for structure-preserving molecular editing. The approach is designed to modify a source molecule based on a natural language instruction to optimize for desired chemical properties while maintaining high structural similarity. MolEditRL consists of a two-stage process: first, a discrete graph diffusion model is pretrained to reconstruct target molecules conditioned on both source structures and text prompts. Second, this model is fine-tuned using reinforcement learning (RL) with a KL-regularized objective to explicitly maximize property-related rewards. To facilitate this research, the authors constructed a new large-scale dataset, MolEdit-Instruct. The experimental results show that MolEditRL outperforms several baselines, including large language models, in both editing accuracy and structural fidelity.

### Strengths
1. The problem of controlled, structure-aware molecular editing is highly relevant to drug discovery. The paper’s core approach of combining a discrete graph diffusion model with a full-trajectory RL fine-tuning process is compelling. By operating on graph representations, it sidesteps the syntactic instability and representational ambiguity issues inherent in string-based methods, representing a step towards more reliable molecular design.
2. The construction of the MolEdit-Instruct dataset is a valuable contribution to the community. The experimental evaluation is extensive, comparing the proposed method against a strong and relevant set of recent baselines on a diverse range of single- and multi-property tasks.
3. The authors show clear motivation of the design choices, particularly the shift towards graph-based representations and the integration of RL for precise optimization.

### Weaknesses
1. The method for encoding the natural language instruction is not sufficiently detailed in the main text. Section 3.1 states that instruction tokens, source atoms, and target atoms are embedded and concatenated into a unified sequence. However, it is unclear how these instruction tokens are specifically embedded and integrated.
2. A well-known failure mode for generative models fine-tuned with RL for molecular optimization is "oracle hacking," where the model learns to generate molecules that achieve high scores from the property predictor but are chemically invalid, unstable, or non-synthesizable. While the KL-regularization term is likely intended to mitigate this by constraining the policy to stay close to the pretrained distribution, the paper does not explicitly discuss this critical issue or demonstrate how its method avoids it. Furthermore, while the paper includes REINVENT4 as a baseline, it misses a broader comparison with other established RL-based molecular optimization frameworks (e.g., MolDQN, GCPN) that could provide a more direct assessment of its RL component.
3. The experiments do not showcase scenarios that leverage the true power and flexibility of natural language, such as describing complex, localized edits or conveying chemical intuition. Without such examples, the language-conditioning aspect feels underexploited, and its contribution to the model's performance is unclear. The current property editing description is too simple.
4. The performance of language-conditioned models is often highly sensitive to the specific phrasing of the input prompt. The paper presents a single, fixed prompt template for each task (Table 6) but provides no analysis or ablation study on how variations in these prompts might affect editing outcomes. This is an omission, as it leaves the robustness of the instruction-following capability unevaluated.
5. The framework's success hinges on the quality of the distribution learned during the first pre-training stage. A high-quality pretrained model should provide a strong prior for the RL fine-tuning, ensuring that the search for high-reward molecules begins from a space of valid and realistic structures. However, the paper does not provide any direct, quantitative evaluation of the stage-one diffusion model (e.g., reconstruction accuracy, validity, and FCD on a held-out set before any RL fine-tuning). This makes it difficult to assess whether the strong final performance is due to a superior pretrained base model or solely the effectiveness of the RL stage.

### Questions
1. Oracle Hacking. Could you please explicitly discuss the problem of "oracle hacking" and elaborate on how the KL-regularization in your RL objective functions maintains chemical realism and prevents the model from generating non-synthesizable molecules? 
2. Justifying Natural Language. Can you provide a more compelling justification for using natural language instructions over structured inputs for the current set of tasks? Better yet, could you demonstrate the model's capabilities on more complex editing tasks where the flexibility of natural language is genuinely advantageous?
3. Evaluating Pre-training. Could you provide standalone evaluation metrics for the discrete diffusion model after the pre-training stage is complete (but before RL)? Metrics like validity, reconstruction accuracy, and FCD against the ground-truth distribution would help disentangle the contributions of the two training stages.
4. Instruction Encoding Details. Could you please clarify the precise mechanism for embedding the instruction tokens and integrating them with the graph representations in the model's input?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents MolEditRL, a framework for structure-preserving molecular editing that integrates discrete graph diffusion with reinforcement learning. The model first performs discrete diffusion pretraining conditioned on the source molecule and natural-language instruction, then applies RL fine-tuning with property-based rewards and KL regularization to balance optimization and structural fidelity. The authors also introduce MolEdit-Instruct, a large-scale dataset for instruction-based molecular editing. Experiments demonstrate substantial gains over existing approaches in both property optimization and structural similarity metrics with fewer parameters.

### Strengths
- Clear writing and strong presentation. The paper is well-structured with clearly outlined motivation, method, and evaluation.
- Technical soundness. The integration of discrete diffusion with RL fine-tuning is implemented carefully and supported by strong experimental performance.
- Comprehensive evaluation. Extensive experiments across single and multi-property editing tasks, with multiple structure-based similarity metrics and FCD, provide convincing evidence.
- Strong empirical results. The method consistently surpasses large language model baselines in both accuracy and structural fidelity while being significantly more parameter-efficient.

### Weaknesses
- Missing efficiency analysis. The paper highlights parameter efficiency but does not report oracle-query efficiency compared to baselines. Quantitative analysis of efficiency would strengthen practical claims.
- Limited novelty in the diffusion–RL combination. Similar hybrid ideas have appeared in related areas. The primary contribution lies in adapting within the molecular editing context rather than fundamentally extending these techniques, making the innovation more technical than conceptual.

### Questions
1. Hyperparameter sensitivity in RL fine-tuning. Hyperparameter sensitivity in RL fine-tuning. Did the authors conduct ablations on key hyperparameters such as $\beta$, policy-update stride, top-k during sampling, and the 0.2 partial-success reward? How did the authors determine the final values?
2. How are train/test splits constructed to avoid data leakage?
3. I am curious about the inference efficiency of the diffusion model. Could the authors compare the generation efficiency of the proposed method with that of other models?
4. The equation 5 appears to mis-type a $q$.

### Soundness
3

### Presentation
3

### Contribution
3
