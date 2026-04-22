# SPUS: A Lightweight and Parameter-Efficient Foundation Model for PDEs

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
We introduce Small PDE U-Net Solver (SPUS), a compact and efficient foundation model (FM) designed as a unified neural operator for solving a wide range of partial differential equations (PDEs). Unlike existing state-of-the-art PDE FMs—primarily based on large complex transformer architectures with high computational and parameter overhead—SPUS leverages a lightweight residual U-Net-based architecture that has been largely underexplored as a foundation model architecture in this domain. To enable effective learning in this minimalist framework, we utilize a simple yet powerful auto-regressive pretraining strategy which closely replicates the behavior of numerical solvers to learn the underlying physics. SPUS is pretrained on a diverse set of fluid dynamics PDEs and evaluated across 6 challenging unseen downstream PDEs spanning various physical systems. Experimental results demonstrate that SPUS using residual U-Net based architecture achieves state-of-the-art generalization on these downstream tasks while requiring significantly fewer parameters and minimal fine-tuning data, highlighting its potential as a highly parameter-efficient FM for solving diverse PDE systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents SPUS, a  compact and efficient foundation model (FM) designed as a unified neural operator for solving a wide range of partial differential equations (PDEs). It adopts a U-Net architecture which is underexplored as a foundation model backbone in the neural PDE solver community. It utilizes an auto-regressive strategy which predicts the entire trajectory based on the initial condition. Experiments demonstrate that SPUS using residual U-Net based architecture achieves state-of-the-art generalization on these downstream tasks while requiring significantly fewer parameters and minimal fine-tuning data.

### Strengths
- The paper is overall well-written and easy to follow. The problem setup and the experiment design are clear.
- The experiment results show that the model is capable of generalizing to unseen initial conditions, unseen equations and scales well with respect to dataset size. The selected two baselines are representative.
- The U-Net architecture is underexplored in the scientific machine learning community. This paper demonstrates the capability of U-Net, which does not have the artifact issues of ViT and ViT-like models.

### Weaknesses
- Formatting issues. Equation (1) does not seem to be properly aligned or wrapped; Line 215 contains only a single $d$.
- Limiting the input to a single initial condition makes the model unable to utilize temporal information. For example, the model cannot simultaneously predicts $t_{0.5}$ and $t_1$ based on $t_0$, while DPOT can change the temporal interval of the input trajectories, and POSEIDON can change the input $t$. This limits the model's capability as a foundation model.
- From my perspective, model-efficiency is not a must for a foundation model, as therefore cannot be considered as an advantage. Moreover, apart from data scalability, model parameter scalability is also important, which could be a disadvantage of U-Net based models.

### Questions
- What is the parameter count of the adapters?
- Can the model generalize to different input resolutions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces SPUS (Small PDE U-Net Solver) — a lightweight (36 M parameter) residual U-Net–based foundation model (FM) for partial differential equations (PDEs).
Unlike prior large transformer-based PDE FMs such as POSEIDON, DPOT, and PROSE-FD, SPUS adopts a simple convolutional encoder–decoder architecture and trains it with an autoregressive (AR) next-step prediction objective, mimicking the behavior of numerical solvers.
The model is pretrained on several compressible Euler PDEs from the PDE-GYM suite and fine-tuned on six unseen downstream PDEs (Euler, Navier–Stokes, and wave equations).

### Strengths
Demonstrates that convolutional architectures remain competitive for PDE foundation modeling, despite recent transformer dominance.

Provides quantitative comparison to state-of-the-art FMs (POSEIDON, DPOT).

Uses challenging, publicly available PDE-GYM benchmarks and tests autoregressive training in a realistic temporal-prediction setup.

Paper is clear; experimental pipeline (pretrain → finetune → rollout) is logically organized.

### Weaknesses
- Lack of genuine novelty and conceptual contribution

The proposed approach offers no clear methodological innovation. The authors simply adopt a standard residual U-Net architecture, train it with a well-known autoregressive (AR) next-step prediction strategy, and evaluate it on existing PDE-GYM datasets. Both the architecture and the training scheme have been extensively studied in the PDE-learning literature. As a result, the paper mainly constitutes a combination of established components rather than a new idea or modeling paradigm. In addition, the overall presentation, including figures and terminology, closely mirrors the POSEIDON paper, while providing less comprehensive experimental analysis.

- Loss of temporal generality compared to POSEIDON

The POSEIDON model is trained in an all-to-all fashion with continuous-in-time embeddings, enabling the model to be queried at any arbitrary time. In contrast, the AR formulation adopted by SPUS predicts only the next timestep given the current one, thus losing the ability to interpolate or query arbitrary intermediate times. This restricts the model’s flexibility and makes it unsuitable for applications that require continuous-time prediction.

- Limited applicability to time-independent PDEs

Because SPUS relies entirely on an autoregressive temporal formulation, it cannot be directly evaluated on steady-state or time-independent PDEs. Supporting such tasks would require additional architectural or procedural engineering (e.g., removing temporal conditioning or introducing pseudo-time variables). The paper does not discuss how such cases would be handled, which limits the generality of the claimed “foundation model” status.

- Incomplete and potentially biased downstream evaluation

Although the authors use the PDE-GYM datasets originally introduced in POSEIDON, they report results on only 6 out of 15 downstream tasks available from that benchmark. The criteria for selecting these tasks are not discussed, and the chosen subset coincides with cases where convolutional architectures are known to perform well. This selective evaluation raises concerns that the tasks were cherry-picked to support the paper’s narrative, rather than representing a fair or comprehensive test of generalization.

- Unfair comparison with POSEIDON due to inference mode

The most critical methodological flaw is the evaluation protocol for POSEIDON. In its original paper (Appendix D.6.2), POSEIDON demonstrates that autoregressive (AR) rollouts yield substantially better performance than direct one-shot predictions for long trajectories (the exact setup used in SPUS). However, in this paper, POSEIDON is evaluated only in direct mode, which is known to degrade accuracy for long-horizon rollouts. Since SPUS performs autoregressive prediction by design, this creates a fundamental evaluation mismatch that favors SPUS.

- Missing comparison to smaller baseline variants

The paper claims substantial parameter efficiency, yet comparisons are made only against large versions of the baselines (POSEIDON-B = 158 M, DPOT-M = 122 M parameters). Both papers also provide smaller variants, POSEIDON-T (21 M) and DPOT-S (30 M), that achieve performance comparable to their base versions. Since SPUS (36 M) is much closer in scale to these lighter models, a fair evaluation of parameter efficiency must include them. Without this, the main empirical claim remains unsubstantiated.

- Unclear evaluation protocol and dataset alignment

It is not specified at which timesteps each model is evaluated. Some downstream trajectories contain 21 timesteps, others only 15.

- Inconsistent or unspecified loss functions during fine-tuning

The paper does not state which loss functions (e.g., MSE, relative L1, or hybrid losses) were used for fine-tuning DPOT, POSEIDON, and SPUS. Since the evaluation metric is MSE, using different training losses can lead to misleading cross-model comparisons. If SPUS was trained directly with MSE while the baselines used relative losses (as in their original works), the reported results may unfairly advantage SPUS.

- Absence of AR evaluation for POSEIDON in Appendix results

Figure A.1 in the appendix reports “error growth over time” but only for direct evaluation of POSEIDON. Since AR rollouts are known to substantially reduce long-term error accumulation, this comparison provides little insight into the true relative performance. An additional plot showing AR-based POSEIDON results would be critical for fairness. Moreover, it would be valuable to understand how POSEIDON would perform if it were fine-tuned using the same autoregressive procedure as SPUS. The paper does not clarify whether such a training regime would yield comparable or potentially improved results for POSEIDON.

- No analysis of scaling behavior

A key feature of any foundation model is scaling performance with model size. The paper provides no experiment or discussion on how SPUS behaves when scaled up or down. Without such evidence, it is difficult to assess whether the proposed model exhibits the characteristic scaling trends expected of an FM.

### Questions
- How stable is the autoregressive rollout of SPUS for very long horizons (e.g., >50 time steps)? Do the authors observe systematic error accumulation or qualitative drift, and how does it compare to transformer-based models?

- Given that SPUS is trained only to predict discrete next steps, how would the model behave if queried at intermediate time steps ? Would this be even possible in current settings?

- Since SPUS’s design is inherently temporal, how do the authors envision adapting it to stationary or steady-state PDEs where no temporal evolution exists?

- Based on their training experience, do the authors expect SPUS performance to improve with increasing model size and (pretraining) dataset size?

The authors should also review the Weaknesses section for additional (implicit) questions and points raised.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SPUS, a lightweight and parameter-efficient foundation model for solving a broad range of PDE systems. SPUS adopts a residual U-Net architecture with only 36M parameters. It is pretrained autoregressively to mimic numerical solvers and fine-tuned on unseen PDE systems.

### Strengths
1. The design is novel and simple, which demonstrates that a lightweight U-Net can serve as an PDE foundation model, challenging the transformer-dominant paradigm.

2. The model achieves competitive results with only one-third the parameters of existing FMs.

3. The model successfully transfers from compressible Euler to incompressible Navier–Stokes and wave equations.

### Weaknesses
1. The paper lacks theoretical or mechanistic analysis explaining why a residual U-Net architecture generalizes across diverse PDE families, beyond the observed empirical results.

2. The contribution of individual design components (e.g., residual blocks, autoregressive training, adapters) remains unclear without ablation studies.

3. The experiments are limited to 2D PDEs at fixed resolution. It is not evident whether SPUS can scale to 3D domains or higher spatial resolutions.

4. Including comparisons with smaller non-transformer baselines (e.g., CNNs, FNOs, or unpretrained U-Nets) would strengthen the claim of parameter efficiency and architectural simplicity.

### Questions
1. Could the authors provide more theoretical insights into why a residual U-Net architecture can generalize effectively across PDE families with distinct dynamics?

2. Have the authors conducted any ablation analyses to isolate the effect of key design components on the final performance?

3. How well does SPUS scale to 3D PDEs or higher spatial resolutions? Are there architectural or computational bottlenecks that would limit such extensions?

4. Would the authors consider adding smaller CNN-based or operator-based baselines to better substantiate the claim of parameter efficiency?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SPUS, the Small PDE U-Net Solver, which aims to explore the U-Net performance in the PDE foundation model. This paper includes extensive experiments to verify its questions, such as whether SPUS generalize to unseen initial conditions or equations or not, as well as the dataset scalability. In the input-one-frame-rollout-inference setting, SPUS surpassed DPOT and Poseidon.

### Strengths
-	It is good to see the investigation of U-Net performance in PDE solving.

-	The authors provide detailed and well-organized experiments.

### Weaknesses
Despite the above strengths, this paper contains some unfair experiments that may render the experimental results meaningless.

### (1) Unfair comparison.

As the authors list in section 3, there are three different forecasting settings. Especially, the correct setting in DPOT is based on several past observations to predict the future. I do not think the setting of repeating ts_0 is a correct usage of DPOT.

Also, according to the statement in “even though DPOT was pretrained on operators of both compressible and incompressible NS equations”, I think the authors directly use the pre-trained models provided by DPOT and did not align the pre-training data with SPUS. This is also quite unfair to baselines, since both pre-training and evaluation data are from PDEgym. 

### (2) Limited novelty.

Although I acknowledge that the authors attempt to rethink the previous architecture of the PDE foundation model, I cannot appreciate the novelty of this paper since this is just an experiment of pre-training a U-Net with PDE data.

All the analyses are just visualizations or quantitative results. I do not think this paper elaborates on why the U-Net works and why it works in a more parameter-efficient way than Transformers.

### (3) About the scalability experiments.

It is common sense that Transformers usually present log-log scalability, which involves both parameter and dataset aspects and can be rigorously tested based on extensive scaling experiments. However, from Table 2, I cannot justify the scalability of SPUS. I think the authors should follow this paper [1] for further experiments. In my opinion, I do not believe that U-Net has good scalability.

[1] Scaling Laws for Neural Language Models. Tech report OpenAI 2020.

[2] Training Compute-Optimal Large Language Models. Tech report DeepMind 2022.

### (4) Missing relative work.

Actually, DPOT and Poseidon are not state-of-the-art foundation models. Please compare with the following work [3].

[3] Unisolver: PDE-Conditional Transformers Are Universal PDE Solvers, ICML 2025.

### (5) Limitation in irregular geometries.

The current design is limited to regular geometries. I think the authors should discuss this in the limitations section.

### Questions
Do the authors adopt the same pre-training data for all the compared baselines?

### Soundness
2

### Presentation
2

### Contribution
1
