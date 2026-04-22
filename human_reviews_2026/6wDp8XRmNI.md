# EMFuse: Energy-based Model Fusion for Decision Making

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Model fusion has emerged as a promising research direction, offering a resource-efficient paradigm that leverages existing pre-trained models to circumvent the need for training from scratch. In this work, we investigate the fusion of models specifically adapted for decision-making tasks. This challenge divides into two distinct, yet related subproblems: the direct fusion of models that act as policy and the fusion of dynamics models that subsequently induce a policy. We suggest that these seemingly divergent subproblems can be unified through the lens of energy-based models (EBMs), which parameterizes a conditional distribution via an energy function where lower energy implies higher probability. Our framework, \textbf{EMFuse}, provides this convergence by leveraging the concept of energy as a common currency for fusion. For direct fusion of policies, such as those in language models, the output distribution is commonly softmax (Boltzmann), which essentially defines the negative logarithmic probability as an energy function. For dynamics models, existing works often train a set of models on the same dataset to obtain robust uncertainty estimation; such an ensemble approach leads to an exponential explosion in computational complexity when it comes to dynamics fusion across multiple sets of models. To overcome this, we introduce the Any-step Dynamics Energy-based Transition Model (ADETM), a novel architecture that performs efficient single-model-per-dataset uncertainty estimation with its energy-based backbone, thereby avoiding this computational explosion. Our EMFuse framework surpasses other baselines by 0.34\% to 6.63\% on single/cross domain discrete decision-making benchmarks, and achieved an extra 2.3 to 7.4 normalized points on average in D4RL MuJoCo continuous-control scenarios. Our code is available at https://github.com/LAMDA-RL/EMFuse.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified framework called EMFuse, aimed at solving the problem of model fusion in decision-making. The authors argue that whether it's directly fusing policy models or fusing dynamics models, both can be uniformly addressed through the lens of Energy-Based Models. To address the computational explosion problem caused by using ensembles when fusing dynamics models, the paper introduces an architecture called ADETM. This architecture leverages variable-length action histories within a single model to estimate uncertainty, thereby avoiding the need to ensemble multiple models.Furthermore, the paper proposes the EMSelect algorithm as an alternative to EMFuse. Instead of generating a consensus distribution, it uses a pairwise EMFuse as a reference at each decision step to select the single expert model with the minimum KL divergence from this reference distribution.In the experiments, the authors evaluated EMFuse and EMSelect on LLM tasks and D4RL MuJoCo control tasks. The results show that this method outperforms several existing training-free fusion baselines.

### Strengths
1. This paper provides a conceptually unified framework, treating the two seemingly different problems of policy fusion and dynamics model fusion as an additive combination of energy functions through the EBM perspective.
2. The authors identify the "combinatorial explosion" problem that traditional ensemble methods introduce when fusing dynamics models. They propose the ADETM architecture to address this. By borrowing ideas from ADMPO and combining them with EBMs, ADETM achieves effective uncertainty estimation using only a single model.
3. The paper's experiments cover two distinct decision-making domains: LLM-based discrete decisions and D4RL-based continuous control. It compares the proposed methods against several representative training-free fusion methods like Model Soup, RegMean, and PackLLM.
4.  By comparing EMFuse and EMSelect, the paper reveals that EMFuse seeks a conservative, robust consensus solution, whereas EMSelect chooses an "optimal" expert at each step. This discussion deepens the understanding of the nature of model fusion.

### Weaknesses
1. The experimental results show that EMSelect outperforms EMFuse in multiple benchmarks. This raises a key question: If "selecting an optimal expert at each step" is more effective than "fusing all experts," does this, to some extent, weaken the "consensus" strategy advocated by EMFuse? The paper attributes this to a trade-off but fails to deeply analyze the root cause of this phenomenon.
2. A core assumption of this framework is that all models to be fused must operate on a shared vocabulary or state-action space. The authors acknowledge this, treating vocabulary mapping as a "viable engineering problem." However, this assumption is a very strong limitation in practical applications, especially when fusing models from different sources.
3. Most experiments in the paper use simple uniform weights. The authors attempted entropy-based dynamic weight adjustment but concluded the effect was "not statistically significant." Why did entropy-based weighting fail?

### Questions
1. Can the authors elaborate further on the scenarios in which the "consensus" strategy of EMFuse would be superior to the "selection" strategy of EMSelect?
2. Given that the entropy-based weight adjustment was not effective, have the authors considered other dynamic weighting schemes?
3. Beyond citing existing work on vocabulary mapping, could the authors discuss the potential negative impacts on the EMFuse framework's performance when forcibly aligning the representation spaces of different models? After mapping, can the "shape" and "scale" of the energy functions still retain their original semantics to ensure the effectiveness of the fusion?

### Soundness
2

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
This paper proposes an energy-based decision model fusion framework, EMFuse. It unifies policy fusion (e.g., for LLM) and dynamic model fusion (e.g., in offline reinforcement learning) through  additive energy composition. The authors also propose EMSelect, a KL-logic-based expert selection method, and ADETM, a single-model architecture that estimates uncertainty without ensemble requirements. Experiments on LLM and D4RL benchmarks demonstrate that EMFuse significantly improves performance compared to training-free baseline methods such as Model Soup and RegMean.

### Strengths
1. This paper presents a well-structured and theoretically consistent framework called EMFuse, which unifies policy fusion and dynamics model fusion under the lens of energy-based modeling.
2. By leveraging the additive property of energy functions, the paper provides an intuitive explanation of model fusion and naturally connects it to classical formulations like LogOP and PoE.
3. The  proposed ADETM model avoids the high computational cost of ensemble methods while achieving single-model uncertainty estimation, which makes it quite practical.
4. The experiments cover both large language model tasks and offline reinforcement learning benchmarks, showing the framework’s versatility and effectiveness across different decision-making scenarios.

### Weaknesses
1. The theoretical discussion in this paper is relatively shallow. Although the authors explain the relationship between EMFuse and LogOP/PoE, they do not provide in-depth mathematical analysis or systematic ablations on fusion stability, the influence of λ, or the effect of uncertainty modeling.
2. The main idea is conceptually close to existing energy-addition or PoE/LogOP-based fusion methods.
3. The paper does not provide sufficient analysis of the independent roles of EMFuse, EMSelect, and ADETM. More detailed ablation studies would help clarify how each module contributes to the overall performance improvement.

### Questions
1. This paper assumes that multiple expert distributions can be fused through energy addition. However, if the experts differ significantly or even conflict, could the product of energies lead to overly sharp or degenerate fused distributions?
2. EMSelect performs per-step expert selection using KL divergence but lacks temporal consistency constraints. If the selected expert changes at every step, could this cause unstable behavior or disrupt contextual coherence during decision making?
3. The authors claim that ADETM has an “any-step” property that enhances modeling capability, but in practice it seems to mainly extend the history window. Does ADETM truly offer a representational advantage over existing autoregressive or recurrent ETMs, or is it more of a structural extension rather than a conceptual innovation?
4. The paper always uses uniform fusion weights λ, but different experts may have varying confidence or training quality. Is this fixed weighting scheme reasonable, or would it be better to consider dynamic or uncertainty-based weighting strategies?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates a challenge about resource-efficient model fusion in decision-making tasks. By considering both fusion of models to act as policy and fusion of dynamic models to induce a policy simultaneously, this paper introduces a framework called EMFuse and an architecture called ADETM to tackle the challenges. The EMFuse shows superior performance in multiple benchmarks including subject-mix, finance-suite and D4RL, compared to three modern training-free baselines.

### Strengths
- The paper is well-written and easy to follow. Visual presentations are clear.
- The motivation is clearly described and the designed framework and algorithms seem clearly aligned with the investigated problems
- Though I’m not an expert in the energy-efficient domain, the proposed methods seem interesting and well tackled the problem.
- The proposed approaches achieve superior performance in various benchmarks, and in terms of various evaluation metrics  
- Limitations are thoroughly discussed

### Weaknesses
I don't have a lot background knowledge in energy-based models, the proposed approaches in general seem sound to me and work well in the benchmarks. But given one of the claimed motivations is the heavy computational complexity for existing dynamic models, I'm expecting if further analysis on related evaluations compared to existing dynamic models

### Questions
Please see my weaknesses above. Additionally, I might miss something in related work, I'm curious how do those evaluation labels defined and the values calculated in table 2, i.e., harmless, helpful, agriculture medication, philosophy?

### Soundness
3

### Presentation
4

### Contribution
3
