# Twin Evolution with Meta Preference Optimization for Semi-Supervised Learning of Large Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, yet their adaptation to specific downstream tasks remains challenging due to limited labeled data. Although post-training methods (e.g., SFT, DPO) have proven effective, they face significant limitations due to the scarcity of labeled data. In this paper, we present TwinEvol, a framework that treats downstream task training and evaluation as complementary, co-evolving submodules. TwinEvol introduces an evaluation agent that co-evolves with the main model; this agent is not a static external module but rather self-iterates and evolves through continuous interaction with the generation LLM after iterative calibration. The agent facilitates more nuanced assessment during downstream adaptation, incorporating hard negative mining and meta-preference optimization to achieve comprehensive feedback and efficient knowledge transfer. Through an iterative twin evolution process, the framework establishes a self-reinforcing cycle that effectively propagates knowledge from labeled to unlabeled data while maintaining task alignment. Experiments across various downstream tasks demonstrate that TwinEvol achieves superior performance compared to existing methods. Our code is available at https://anonymous.4open.science/r/TwinEvol/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an approach to semi-supervised training whereby the main LLM is iteratively co-trained with an evaluation agent which provides quality feedback signals for an RL style training. The key idea is two fold: 1) On the labeled examples, model responses are generated and compared with the annotated response. The pairs where the annotation is ranked higher than the model response are used for training the evaluative agent. 2) On the unlabeled examples, K model responses are generated and classified as consistent/inconsistent. SFT is performed on the consistent responses. The evaluative agent generates pseudo-preferences over all pairs of model responses, which are then used for RL using the MetaPO algorithm.  Results are reported using three foundation models: Llama 3.1 8B, Llama 3.2 3B and Gemma 2-9B. on both general purpose and domain benchmarks. The baselines include a number of semi-supervised approaches. The proposed method consistently improves upon the baselines. Ablation studies show that each component in the proposed model plays an important role.

### Strengths
* Proposes an approach for semi-supervised learning by iteratively co-training the LLM and an evaluation agent which can provide supervision for RL.
* Shows that the proposed method improves upon several semi-supervised baselines.
* Proposes an extension to the SimPO algorithm (Meng et al. 2024) which uses pseudo-preferences over multiple pairs of responses, and discusses its theoretical properties.

### Weaknesses
* The algorithm for identifying consistent vs inconsistent responses is not discussed clearly with examples. This is a crucial component of the proposed algorithm.
* The paper suffers from lack of clarity in multiple places. See questions below.

### Questions
* L103: It would be good to compare the proposed method (metaPO) to GRPO (Shao et al. 2024 : https://arxiv.org/pdf/2402.03300)
* L153: What is the algorithm for identifying consistent vs inconsistent responses? It would be useful to give examples of consistent vs inconsistent responses.
* L157: 'We take inconsistent preference annotations as training data for the evaluative agent ' - From the paper, it looks like there is a procedure to classify responses as consistent / inconsistent. However, it is not clear what inconsistent preference annotation means. Could you give an example?
* L179: 'Use the LLM to judge the consistency': Is that done in the paper? If so, what LLM is used for this purpose? Please add details. 
* L214: Please discuss how metaPO differs from SimPO. Is SimPO a special case of metaPO?
* L406: "Table 1 shows that SFT and SemiEvol approaches yield substantial improvements by effectively leveraging both labeled and unlabeled data" - Isn't SFT using only labeled data? 
* L408: "Post-training techniques show marginal improvements or degradation due to distribution misalign- ment and insufficient capability enhancement." - What are the post-techniques that this sentence is referring to?
* L1052: Please give examples of predefined rules, the consistency validation procedure and examples of consistent vs inconsistent responses.

Typo:
* L061 'while' -> 'with'

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TwinEvol, a framework for semi-supervised learning of LLMs designed to address the challenge of limited labeled data. The core of TwinEvol is a dual-branch architecture where a generative LLM and an "Evaluative Agent" co-evolve. This process creates a self-reinforcing cycle that transfers knowledge from a small amount of labeled data to a larger pool of unlabeled data. Key components of this framework include Hard Negative Mining and a MetaPO algorithm. The authors provide theoretical analyses of their approach and empirical results across several benchmarks, demonstrating that TwinEvol outperforms existing methods.

### Strengths
1. The concept of a co-evolving generative LLM and Evaluative Agent is a creative and promising approach to semi-supervised learning. This "twin evolution" creates a dynamic and self-improving system which embodies the idea of self-play and DPO, expected to be able to continuously enhance its capabilities.

2. The paper presents a comprehensive set of experiments on various benchmarks, including MMLU, MMLU-Pro, ARC, FPB, USMLE, PubMedQA, and ConvFinQA. The results consistently show that TwinEvol outperforms a range of baseline methods, including vanilla SFT, self-evolution approaches, and domain adaptation techniques. The improvement is also consistent across different model architectures and scales (Llama3.1-8B, Gemma2-9B, and Llama3.2-3B).

### Weaknesses
1. Complexity of the Framework: The TwinEvol framework, with its dual-branch architecture, co-evolutionary loop, and multiple components (Hard Negative Mining, MetaPO), is considerably more complex than standard fine-tuning methods. The practical overhead of implementing and tuning such a system could be a barrier to adoption.

2. The most critical concern is if the proposed "twin evolution" mechanism is really effective. On all benchmarks, TwinEvol outperforms baseline SemiEvol, and even the naive SFT, only by very small margin. Considering the numerous techniques implemented above, and results in ablation study (Table 3), the reviewer question that the "twin evolution" mechanism itself brings no improvement. The marginal performance gain is due to some technique implementations only. For example, it seems that consistency checking is very effective itself comparing V1 and V2. The reviewer wonder if the performance of TwinEvol would degrade a. lot without CONS, or  other baselines can be improved simply with CONS.

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a training framework in which the LLM and the evaluator are trained alternatingly. The goal is semi-supervised training for leveraging both unlabelled and labelled data. The EvaluationAgent is trained on labelled examples on which it disagrees with the reference. The generative LLM is trained on labelled data first, and then samples output on unlabelled data and uses the signal from the EvaluationAgent.

### Strengths
- The high-level setup makes intuitive sense
- Quite consistent gains across multiple models and datasets (but see caveat below)

### Weaknesses
I feel that the substance of this work is harder to extract than it needs to due to a quite extensive use of terms like knowledge/evolutionary flywheel, co-evolutionary loop/framework, information infusion/propagation/injection in contexts where it's not helpful - a more concise and sober description would help clarity.

At the same time, there are a couple of points that stay fairly vague. For example, the way D_incon and D_con are split is not described sufficiently (L187), although it seems to be a crucial part of the approach.

IIUC the baselines in Table 1 are of similar size, but they are not based on exactly the same LLM. This would mean that the only apples-to-apples comparison is against vanilla and SFT - adding more sophisticated baselines to Table 2 would strengthen the results.

### Questions
- Can you add stronger baselines to Table 2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents and effort to develop a joint evolution framework, termed as TwinEvol, where an evaluation agent and the base LLM are jointly optimized. The authors use a mixture of supervised and semi-supervised data to perform the joint optimization. 

The work provides certain theoretical guarantees, although they largely reflect SimPO and other DPO guarantees. 

On the evaluation front, the authors provide a comparison with SFT and other methods and highlight some of the improvements achieved.

In summary, paper does point to interesting methods to jointly develop an evaluation agent along with the LLM fine-tuning. However, the evaluation and the impact are somewhat unclear.

### Strengths
Proposing the LLM fine-tuning as a joint optimization framework which also involves the learning of an evaluation agent. 

Establishing a theoretical framework to analyze the MetaPO in conjunction with other works based on preference optimization.

Evaluation on multiple tasks and models.

### Weaknesses
The major issue with the paper is the establishment of the impact and the unfair comparison that the authors have made in the evaluations. 

  i. As part of the TwinEvol, the labeled data is used for regular SFT. However, the proposed framework also uses unlabeled data. The amount of unlabeled data used in their setting is not specified either, while this setup is refereed to a prior work. In Table 2, the performance comparison is provided only for the SFT (which uses supervised data and is also part of the proposal). As the TwinEvol also requires additional unsupervised data, this comparison with SFT is unfair.  

Ii. A related concern is in the strategy proposed to utilize the unsupervised data. In Section 3.3, the authors mention the division of the unsupervised pile into a consistent set and an inconsistent set. The authors mention, rather vaguely, the use of “answer comparisons”, “LLM to judge the consistency”. What LLM was used for this purpose and how. The augmentation of this additional “LLM as judge” data could also improve the SFT and other semi-supervised strategies. What additional value does MetaPO bring. The description of using unsupervised data is rather shallow and leaves many questions unanswered.

Iii. The prior work citations and comparisons do not dwell deeply into various related efforts in the field of semi-supervised fine-tuning. For example, efforts related to self-alignment with LLM as a judge [“Meta Rewarding Language Models …” by Wu et al. 2024], Semi-supervised reward modeling [He et al 2024], or self-rewarding language models [SRLM, ICML 2024] are already available. The authors do not compare with these prior efforts, nor is there reference to other similar works on semi-supervised modeling in preference optimization. Without these comparisons and benchmarking, the impact and value addition with the proposal is unclear. Further, the positioning of the work with respect to prior art is also well established.

Iv. The computational complexity discussion leaves a lot to be desired. Does the comparison with SFT and proposal (Table 4) also include the cost of the SFT for the TwinEvol, as the proposal requires SFT as the starting point. Further, how does this compare with other approaches like SemiEvol which utilize unsupervised data. 

v. The writing has various typos and errors, which warrants a thorough proof-reading. Further, various notations are introduced without fully describing the components. In this regard, it appears that the paper writing was performed in haste, leaving it somewhat unpolished.

### Questions
— Careful benchmarking with other prior efforts that utilize semi-supervised learning frameworks, where LLM-as-a-judge is employed.
— Clear description of the efforts in utilizing the semi-supervised data and how the LLM-as-a-judge is utilized in the proposal.
— Appropriate computational considerations which bring out the exact requirements of the proposal. 
— Positioning the work in a refined manner to understand the value addition of the work.   
— Considerations in creating the Evaluation Agent and ablations around the errors made by the evaluation agent itself. For example, the paper does not dwell on how accurate the evaluation agent alone is, and what impact this has on the rest of the LLM training pipeline. 
-- Thorough rewriting to improve the descriptive quality of the draft.

### Soundness
3

### Presentation
2

### Contribution
3
