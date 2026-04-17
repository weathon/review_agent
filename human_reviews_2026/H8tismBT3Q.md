# AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Large language models (LLMs) have significantly advanced protein representation learning. However, their capacity to interpret and design antibodies through natural language remains limited. To address this challenge, we present AFD-Instruction, the first large-scale instruction dataset with functional annotations tailored to antibodies. This dataset encompasses two key components: antibody understanding, which infers functional attributes directly from sequences, and antibody design, which enables de novo sequence generation under functional constraints. These components provide explicit sequence-function alignment and support antibody design guided by natural language instructions. Extensive instruction-tuning experiments on general-purpose LLMs demonstrate that AFD-Instruction consistently improves performance across diverse antibody-related tasks. By linking antibody sequences with textual descriptions of function, AFD-Instruction establishes a new foundation for advancing antibody modeling and accelerating therapeutic discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AFD-Instruction, the first large-scale instruction dataset that links antibody sequences with natural-language functional annotations. The dataset is built through a multi-agent pipeline combining extraction, mechanistic reasoning, and expert verification. It supports two domains: antibody understanding and function-guided antibody design. The authors fine-tune open-source foundation models (Qwen, LLaMA) and demonstrate substantial gains over both protein-language and general-purpose LLMs in understanding and design metrics.

### Strengths
1. High-quality domain focus. Antibody-specific dataset aligning sequences, functional text, and design tasks, and it is the first of its kind.
2. Comprehensive pipeline. Multi-agent extraction and self-questioning expansion show thoughtful integration of automation and expert verification.
3. Rich evaluation. Covers both understanding and design tasks, with structural and energetic validation using tFold + Rosetta.
4. Substantial empirical gains. Instruction-tuned QwenAB/LLaMAB outperforms general and domain-specific baselines across multiple metrics.
5. Ethical considerations (dual-use awareness, controlled access) are helpful and detailed.

### Weaknesses
### W1 Limited methodological novelty.
The dataset-generation procedure largely extends patterns from Mol-Instructions and Evola (instruction synthesis via LLM prompting + self-questioning). The multi-agent framing repackages established extract–verify–summarize steps; conceptual contribution is modest beyond domain adaptation.
### W2 Small biological diversity, large linguistic inflation.
The dataset derives 430 K instructions from only 4.3 K antibodies, meaning linguistic diversity far exceeds biological diversity. It's unclear whether model improvements stem from true functional grounding or synthetic linguistic variety.
### W3 Absence of mechanistic generalization analysis.
The paper shows performance gains but not whether models generalize to unseen antigen classes or novel mechanisms. Without a cross-target or cross-epitope evaluation, "understanding" may still be surface-level pattern matching.
### W4 Evaluation focuses on text similarity rather than scientific correctness.
BLEU/ROUGE metrics reward lexical overlap but not mechanistic accuracy. Expert or structure-aware evaluation (e.g., contact-map consistency, mutation-effect prediction) would be more meaningful.
### W5 Limited ablation and uncertainty quantification.
No breakdown of how each module (Extractor, Mechanism, Function, Self-Questioning) affects data quality. Single-run results in Table 1 and Table 2 without error bars obscure robustness.
### W6 Weak discussion of relation to prior instruction datasets.
Although Table 5 lists comparisons, the paper underplays key distinctions from Evola, which already support large-scale protein instruction tuning, and doesn't clearly justify why antibody specificity requires a new corpus instead of targeted filtering from Evola.

### Questions
If the author could provide clarification to the points mentioned in the weakness section and the following questions, I am happy to consider raising the score.
1. Can models trained on AFD-Instruction transfer to unseen antigen families or novel antibody mechanisms (e.g., pH-dependent binding, conformational epitopes)? How do they perform under cross-antigen split evaluation?
2. How much do the multi-agent refinement, template expansion, and expert QC respectively improve accuracy or diversity of annotations?
3. Given only 4 K unique antibodies, how do you ensure the 430 K instructions are not redundant paraphrases? Did you analyze instruction uniqueness or semantic clustering?
4. Beyond domain restriction, what conceptual advances justify AFD-Instruction as a new dataset family rather than a subset or extension of Evola?
5. Would it be feasible to include domain-expert evaluation of generated mechanism descriptions, or at least automated structure-text alignment (e.g., verifying residue mentions against structural contacts)?

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
This work introduces the first large-scale instruction dataset specifically designed for antibodies, linking their sequences with concise functional descriptions to enable large language models to both interpret and design antibodies. By curating over 430,000 verified instruction–response pairs from structural databases and literature through a multi-agent extraction and self-questioning system, the dataset captures detailed annotations related to targets, mechanisms, and functions. Fine-tuning general LLMs such as LLaMA and Qwen on this resource produces specialized models—LLaMAB and QwenAB—that significantly outperform existing protein and general-purpose models in antibody classification, functional reasoning, and sequence generation tasks. Additionally, the models generate biologically plausible antibody sequences that demonstrate enhanced stability, binding affinity, and structural accuracy. Overall, AFD-Instruction bridges the gap between natural language guidance and molecular design, providing a new foundation for interpretable and goal-directed antibody discovery.

### Strengths
1 The paper introduces AFD-Instruction, the first large-scale instruction dataset for antibodies that pairs antibody sequences with structured natural language functional descriptions. This resource fills a key gap left by previous sequence-only datasets lacking functional or semantic supervision, enabling models to learn the mapping between sequence and function.

2 The authors propose a multi-agent pipeline—comprising Mr. Extractor, Dr. Mechanism, and Prof. Function—and combine self-questioning and template-based generation to produce high-quality instruction data. A rigorous quality control process involving both automated validation and expert review ensures biological reliability and factual consistency across annotations.

3 Models fine-tuned on AFD-Instruction, such as LLaMAB and QwenAB, achieve state-of-the-art results in antibody classification, functional reasoning, and design tasks. Their generated antibodies exhibit greater structural stability and stronger binding affinity compared with existing protein language models, demonstrating high potential for antibody understanding and rational antibody design in therapeutic discovery.

### Weaknesses
1 Although the AFD-Instruction dataset is large and comprehensive, its functional annotations mainly rely on literature-derived and database-extracted descriptions. These primarily cover common mechanisms such as neutralization, blocking, and binding-site recognition, but lack more fine-grained or dynamic functional information such as epitope escape, affinity modulation, or immune regulation.

2 The paper only validates its approach using two model families, Qwen and LLaMA, which limits the generalizability of the results across a broader range of large language model architectures. Moreover, both employed models have relatively modest parameter sizes. It is recommended to include experiments involving a wider variety of model types and parameter scales to more comprehensively evaluate the robustness and scalability of the proposed method.

3 In terms of consistency, the authors implemented a hybrid quality control protocol that combines automated semantic checks with limited expert review, but since only a small subset of data was manually verified, subtle semantic inconsistencies may persist.

### Questions
Same as weakness.

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
4

### Summary
The paper addresses a key challenge in bioinformatics: while Large Language Models (LLMs) are powerful, their ability to interpret and design antibodies using natural language instructions is limited. To solve this, the authors introduce AFD-Instruction, the first large-scale instruction dataset specifically for antibodies, which links sequences to detailed functional annotations. The authors fine-tuned general LLMs (Qwen2-7B and LLaMA-8B) on AFD-Instruction to create new models named QwenAB and LLaMAB. In understanding tasks, these models achieved SOTA results, significantly outperforming both general-purpose LLMs. In design tasks, the models generated antibody sequences with greater plausibility (lower perplexity) and higher fidelity to natural sequences (higher SRR).

### Strengths
1. While instruction tuning for general proteins exists (e.g., InstructProtein, Mol-Instructions) , this paper correctly identifies antibodies as a unique and challenging class that existing resources do not adequately cover.

### Weaknesses
1. The models (QwenAB, LLaMAB) show massive performance gains on classification tasks (e.g., in Table 1, 87.81% on Binding for QwenAB vs. 46.59% for GPT-4o). This dramatic improvement suggests the model might be learning spurious correlations or "template-fitting" rather than generalizable biological reasoning. The Understanding evaluation needs a more challenging, held-out test set to prove generalization.
2. The dataset is built by sampling 4,305 antibody entries from PDB and SAbDab. These databases are inherently biased towards structurally-characterized and crystallized antibodies. The paper mentions using MMseqs2 to mitigate data imbalance, but this addresses sequence redundancy, not the foundational structural and functional bias of the source data.
3. Table 3 compares models on metrics like Perplexity, SRR, and various structural scores. The AFD-tuned models (LLaMAB) perform well. However, this experiment compares models trained on different data and for different tasks. It shows that LLaMAB is a "good" antibody sequence generator in general, but it does not prove that it successfully followed the natural language functional instruction.

### Questions
1. Could you evaluate your model's performance on a temporal hold-out set (i.e., antibodies and papers published after your data collection cutoff)? This would provide much stronger evidence of true generalization.

### Soundness
3

### Presentation
3

### Contribution
2
