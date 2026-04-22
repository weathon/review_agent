# Aegis: Automated Error Generation and Attribution for Multi-Agent Systems

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Large language model based multi-agent systems (MAS) have unlocked significant advancements in tackling complex problems, but their increasing capability introduces a structural fragility that makes them difficult to debug. A key obstacle to improving their reliability is the severe scarcity of large-scale, diverse datasets for error attribution, as existing resources rely on costly and unscalable manual annotation. To address this bottleneck, we introduce *Aegis*, a novel framework for **A**utomated **e**rror **g**eneration and attr**i**bution for multi-agent **s**ystems. *Aegis* constructs a large dataset of **9,533** trajectories with annotated faulty agents and error modes, covering diverse MAS architectures and task domains. This is achieved using a LLM-based manipulator that can adaptively inject context-aware errors into successful execution trajectories. Leveraging fine-grained labels and the structured arrangement of positive-negative sample pairs, *Aegis* supports three different learning paradigms: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning.  We develop learning methods for each paradigm.  Comprehensive experiments show that trained models consistently achieve substantial improvements in error attribution. Notably, several of our fine-tuned LLMs demonstrate performance competitive with or superior to proprietary models an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work features two major contributions for failure attribution of multi-agent system. The first is a large-scale dataset of annotated failure trajectories (failed agents and error modes), obtained by manipulating successful trajectories. The second is unique training methods (reinforcement learning and contrastive learning) which are made possible thanks to the characteristics of the constructed dataset. Experiments on the test set of the proposed dataset and the OOD Who&When dataset demonstrate the effectiveness of the training data and methods, with SFT being the most effective on average. There are many extra analyses and observations presented, which are of potential value for future works.

### Strengths
1. Manipulating successful trajectories is a reasonable and scalable way to create data for failure attribution. This is evidenced by that the resulting dataset has 9000+ samples, which is of significantly larger size than e.g. Who&When dataset.
2. The proposed method of data generation is also controlled enough that it can provide fine-grained information for training methods beyond SFT.
3. Having results on OOD/unseen dataset (Who&When) makes the evaluation more comprehensive and convincing.

### Weaknesses
I don't see major weaknesses in the submission.

### Questions
1. When validating generated trajectories (i.e., seeing whether the intervention really induces a failure), is there a concrete like rule-based function for that, or llm-as-a-judge is employed? If it's the latter case, how accurate the ground-truth labels are (how much noise is there)?
2. Why GRPO consistently performs worse than SFT (e.g., Qwen3-8B-Thinking + GRPO is worse than Qwen3-8B-Non-Thinking + SFT)? Any theory or thoughts?

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
5

### Summary
This paper introduce Aegis, a frame to automatically generate erroneous trajectory for multi-agent system. It's used to construct a large dataset of trajectory with error annotation and modes. The author use the generated dataset to fine-tune LLM to do error attribution with three different learning paradigms. Experiment results show the finetuned model improves on the task of error attribution of multi-agent systems.

### Strengths
1. The data generation pipeline seems effective and the constructed dataset could be useful for the community
2. The results show that the constructed data can significantly improve open-source models capability of error attribution
3. The proposed DCL is interesting and looks promising even though it's still lag behind LLMs

### Weaknesses
1. The design of the error mode taxonomy is critical, but the authors didn't discuss and reveal how it is done in the main body of the paper
2. The data generation pipeline will result in real positive trajectories and synthetic negative trajectories, this may not align with real world cases where errors are real and diverse

### Questions
In addition to the two strategies, using an underperforming LLM for certain steps may be a good way to generate realistic errors but it's not guaranteed to produce error 

Also, how does the model performance on aegis-bench correlated with who&when? It would be better to do a correlation analysis to see if aegis-bench is positively correlated with who&when

For DCL, it may be better to use better text encoders, eg, qwen3-embedding for better performance; also it would be good to list the number of parameter of DCL to show it's much smaller compared to LLM

### Soundness
2

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
5

### Summary
This paper presents AEGIS, an automated framework for generating large-scale error attribution data in multi-agent systems built from LLMs. By injecting controlled, context-aware faults into correct trajectories, AEGIS creates over 9,500 annotated failure cases. These are used to train models to improve attribution performance across both synthetic and real-world datasets.

### Strengths
1. The work presents a scalable and principled pipeline for generating faulty MAS trajectories with programmatic ground-truth labels. This is a significant contribution, as data scarcity has long been a limiting factor for MAS error analysis.
2. AEGIS supports supervised, RL-based, and contrastive learning methods. The design shows thoughtful alignment between the structure of synthetic data and training objectives, which is rarely done this thoroughly in MAS.
3. The authors benchmark a wide array of models across multiple axes and test both in-domain and OOD generalization. Fine-tuned models on AEGIS surpass many proprietary LLMs in accuracy, showing the power of programmatically generated data for enabling high-fidelity diagnostics. This makes the empirical evaluation strong and convincing.

### Weaknesses
1. While the paper proposes a novel Disentangled Contrastive Learning approach, the details of how training pairs are generated remain underspecified. It is unclear how a single correct trajectory is transformed into contrastive pairs and how these are selected. A walkthrough or algorithmic example showing how one trajectory is transformed into training instances for DCL would greatly enhance clarity and reproducibility.
2. AEGIS generates incorrect trajectories by injecting errors into correct ones, but many real-world MAS failures, such as those in Who&When, originate in naturally erroneous decision paths. This creates a potential misalignment between how the data is constructed and how errors manifest in the wild. While the authors demonstrate strong generalization, the conceptual gap between error injection (correct to incorrect) and naturally flawed trajectories (originally incorrect) is not addressed. This weakens the connection to real-world deployment contexts and could affect performance on systems where errors are emergent rather than injected.
3. While the results are strong, the paper would benefit from qualitative examples of common failure modes that the best models still struggle with.

### Questions
1. The paper evaluates on the Who&When benchmark, where trajectories can exceed 50 steps. However the model has a maximum input length of 8192 tokens during training. The paper does not clarify how such lengthy interactions are handled. Are steps truncated, summarized, or skipped? 
2. Are the injected errors randomly selected, or is there a balancing mechanism to ensure coverage across error modes? Could the model become biased toward more frequently injected errors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to address the bottleneck of data scarcity in training error-attribution systems for LLM-based multi-agent systems' execution. The authors introduce Aegis, a framework to programatically generate large-scale dataset of annotated error trajectories. Aegies starts with collecting successful, error-free baseline trajectories from a pre-selected set of MAS, then applies targeted LLM-based interventions to the baseline trajectories (by prompt injection and response corruption), and then validates that the system continuing to execute from the corrupted state indeed fails to complete the task, filtering out the ones that still do not lead to a failure. The authors create 9533 annotated error trajectories across 6 MAS with Aegis, and train models using SFT, RL and CL.

### Strengths
* Addresses an important problem of data scarcity in training models for MAS error identification.
* Provides a simple intervention to utilize previous successful trajectories.
* Presents a dataset of 9,500+ annotated trajectories, and validate its usefulness by showing OOD generalization in training with this data.

### Weaknesses
- Since the technique starts with successful trajectories, it is unclear how this will be applied to MAS built for domains with very high failure rates.
- Due to the dataset being built by corrupting successful trajectories, the dataset will miss out entirely on representing real-world failure modes of the MAS being represented in the dataset.
- Even though an intervention is made in the MAS trajectory to corrupt it, the downstream failure reason could be different from the intervention.

### Questions
1. Can the authors comment on how it could be ensured that an error introduced under taxonomy error category X, actually leads to observed failure mode X, and not a different failure mode?
2. Can the authors discuss how Aegis handles representing real-world failure modes of MAS, since it starts from successful trajectories?
3. The studies performed in the paper show improvements in the ability to detect failure modes in a MAS trajectory (through 3 different learning modes), however, it is unclear how this can be utilized to improve the performance of the MAS itself. Could the authors comment on that?

### Soundness
3

### Presentation
2

### Contribution
2
