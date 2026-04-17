# IMPACT: Industrial Machine Perception via Acoustic Cognitive Transformer

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Industrial acoustic signals encode machine state, yet prevailing data-driven approaches are task-specific supervised pipelines that generalize poorly beyond their design conditions. Progress is further limited by the scarcity of large-scale datasets and pretrained models tailored to active shop floor audio. To address this, we introduce DINOS (Diverse INdustrial Operation Sounds), a dataset of 74,149 recordings totaling over 1,093 hours collected from active manufacturing lines across diverse processes and operating regimes. We also provide IMPACT(Industrial Machine Perception via Acoustic Cognitive Transformer), a reference model pretrained on DINOS to standardize evaluation. Our benchmark is structured in four machine-specific steps: (1) baseline discrimination, (2) moderate operational complexity, (3) scalability to unseen equipment, and (4) domain shift and sensor modality adaptation. Across tasks, models pretrained or fine-tuned on DINOS consistently outperform general-purpose audio models, demonstrating the value of domain-specific pretraining for industrial acoustic perception.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new dataset DINOS, consisting of 74149 acoustic samples collected from active manufacturing lines. The authors then proposed a pretraining method, IMPACT, which is conceptually similar to EAT. The authors evaluated its performance on 27 downstream tasks using DINOS.

### Strengths
1. The collection of DINOS is an earnest effort. DINOS consists of the signals collected from both a microphone and a stethoscope, and covers various types of equipment.
2. The authors evaluated the performance of various off-the-shelf pretrained models on DINOS.

### Weaknesses
1. The evaluation is critically insufficient and cannot show the superiority of IMPACT. The authors did not apply other pretraining methods (e.g., AudioMAE) on DINOS. They only evaluated the off-the-shelf pretrained models (e.g., a model pretrained using AudioMAE method on other acoustic datasets) on DINOS. Since IMPACT is a pretraining method, if the authors want to show the superiority of IMPACT, they need to **pretrain** IMPACT and other pretraining methods (e.g., AudioMAE) **on the same datasets**.
2. The proposed pretraining method is conceptually not sufficiently novel. Its similarity to EAT is also acknowledged by the authors.
3. Therefore, if the majority of the contributions lie in the introduction of DINOS, then this paper might be below the bar of ICLR. It might be more suitable to submit this paper to a venue specialized in industrial sensing or a venue offering dataset tracks.
4. The presentation could be improved overall.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
## An open-access dataset of industrial operation sounds

- Proposed DINOS, a dataset with over ~1000 hours of recordings from active manufacturing lines.
- Proposes IMPACT, a reference baseline models trained on the DINOS dataset.

### Strengths
- Paper is well written (except some minor grammatical errors. Authors, please recheck for missing spaces and punctuation.)
- A comprehensive benchmarking setup, with distinct pretraining and downstream benchmarking sets is provided.
- Limited availability of public, large-scale corpora is a major pain point in manufacturing and floor monitoring, so the dataset could indeed prove invaluable to the community.
- Evaluation, to the extent done in the paper, is good.

### Weaknesses
- Based on the results alone, it is hard to say how useful the proposed dataset is over the publicly available DCASE2025 Challenge Task 2 dataset for pretraining.

### Questions
1. Is there an overlap between the pretraining set for DINOS and DCASE2025 Challenge Task 2? 
2. Why is your paper titled after the model, and not the dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes DINOS (Diverse INdustrial Operation Sounds), a large-scale dataset for understanding Industrial acoustic signals at a large scale. The paper also trained a self-supervised baseline model on the data IMPACT (Industrial Machine Perception via Acoustic Cognitive Transformer).

### Strengths
The paper is unique and interesting. The paper is written well and contains detailed experiments. The self-supervised model IMPACT, trained on the proposed data, achieves the best performance across the majority of the tasks.

### Weaknesses
The paper has limited novelty. The primary contribution of the paper is the dataset; the IMPACT model is based on a well-known existing self-supervised model, EAT.

### Questions
What is the number of parameters across various models in Table 4? Does the Impact model work better because it is a larger model, or due to pretraining on DINOS?

### Soundness
2

### Presentation
2

### Contribution
2
