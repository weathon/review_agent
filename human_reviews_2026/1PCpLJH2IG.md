# Curriculum-Guided Layer Scaling for Language Model Pretraining

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
As the cost of pretraining large language models grows, there is continued interest in strategies to improve learning efficiency during this core training stage. Motivated by cognitive development, where humans gradually build knowledge as their brains mature, we propose Curriculum-Guided Layer Scaling (CGLS), a framework for compute-efficient pretraining that synchronizes increasing data difficulty with model growth through progressive layer stacking (i.e. gradually adding layers during training). At the 100M parameter scale, using a curriculum transitioning from synthetic short stories to general web data, CGLS outperforms baseline methods on the question-answering benchmarks PIQA and ARC. Pretraining at the 1.2B scale, we stratify the DataComp-LM corpus with a DistilBERT-based classifier and progress from general text to highly technical or specialized content. Our results show that progressively increasing model depth alongside sample difficulty leads to better generalization and zero-shot performance on various downstream benchmarks. Altogether, our findings demonstrate that CGLS unlocks the potential of progressive stacking, offering a simple yet effective strategy for improving generalization on knowledge-intensive and reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose the use of layer scaling combined with curriculum learning to improve the efficiency of LLM pretraining. Layer scaling progressive increases the model size, while curriculum learning progressively increases the difficulties of the samples. Their empirical results show that the combination outperforms naive layer scaling.

### Strengths
1. The research direction of progressively increasing the model size is interesting and worth exploring.

### Weaknesses
1. Overall, the novelty and insight provided by this paper is limited. Various settings used in the experiments seem arbitrary which leads to concern about scalability.
2. The comparison and the improvement over the baseline seem not to be very solid. The setup for the curriculum learning baseline is not clearly specified, while there are various stronger baselines in this category in the literature. For 700M and 2.5B token setting, the improvement over the curriculum learning baseline is slight, while it is unclear how much tuning is conducted for each of the methods. For 20B token setting, comparison with curriculum learning baseline  is not shown.

### Questions
1. The combination of layer scaling and curriculum learning also increases the design complexity. How sensitive are the proposed methods to these settings (When and how many layers are scaled at a time / How large and how much training each curriculum be)? 
2. Do you have results on larger models? This seems to be important as various settings in the experiment seem arbitrary which leads to concern about scalability.

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
4

### Summary
This paper proposes Curriculum-Guided Layer Scaling (CGLS), a novel pretraining framework for language models that synchronizes progressive model growth (layer stacking) with a data curriculum of increasing complexity. The core idea is inspired by cognitive development and aims to improve computational efficiency and downstream performance. The authors conduct experiments at multiple scales (100M and 1.2B parameters) and show that CGLS outperforms several baselines, including standard training, curriculum-only, and layer-scaling-only approaches, on various reasoning and knowledge-intensive benchmarks.

### Strengths
1. The core design of the paper is quite interesting and can provide some inspiration for the field.

2. Write clearly and easily understandable.

### Weaknesses
1. The downstream evaluation focuses heavily on multiple-choice QA tasks (PIQA, ARC). While these are standard, they don't fully capture the breadth of LLM capabilities. The evaluation would be strengthened by including math benchmarks (e.g., GSM8K, AIME 2024, AIME 2025…), and coding benchmarks (e.g., Humaneval…).

2 . The experiments only reach 1B parameters and are confined to the Llama architecture, failing to demonstrate scalability to modern 7B+ models or generalization across diverse architectures (e.g., Qwen…l).

3. No evaluation on dedicated reasoning benchmarks and basic reasoning models is provided.

### Questions
Please see the weaknesses.

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
3

### Summary
The paper introduces Curriculum-Guided Layer Scaling (CGLS), a pretraining paradigm for large language models (LLMs) that synchronizes model growth (via progressive layer stacking) with data curriculum learning. Specifically, at each training stage, several new transformer layers are added and trained in isolation while earlier layers are frozen. The entire model is then unfrozen and fine-tuned on a dataset of higher complexity, with data difficulty measured using a DistilBERT-based classifier or intrinsic dataset structure.

The authors conduct experiments at multiple scales: GPT-2-Small (124M parameters), LLaMA-3.2-1B (1.2B parameters), and a Chinchilla-optimal 1B–20B token setup, as well as a domain-shift setting from general text to code. Across these settings, CGLS consistently outperforms compute-matched baselines, including progressive stacking (MIDAS), layer-scaling-only, and curriculum-only methods. The approach yields improved reasoning and generalization performance on benchmarks such as PIQA, ARC-Easy, and HellaSwag.

### Strengths
1. The integration of curriculum learning and progressive layer scaling into a unified pretraining framework is novel for LLM pretraining. The approach is conceptually sound and aligns well with cognitive principles of gradual learning.

2. The experiments are thorough and well-validated across multiple model scales, with replication and ablation studies supporting the claims. Results show consistent and meaningful gains on reasoning-focused benchmarks such as PIQA and ARC, demonstrating improved efficiency without additional computational cost.

3. The paper is clearly written, well-organized, and easy to understand.

### Weaknesses
1. Most experiments focus on reasoning and question-answering benchmarks. Broader capabilities such as dialogue generation, summarization, or open-ended tasks are not evaluated, leaving the generality of the approach less explored.
2. The effectiveness of CGLS relies on the accuracy of the DistilBERT-based difficulty classifier for data stratification, which may not generalize well to other languages, domains, or modalities.
3. The method is only tested on text-based pretraining; no experiments are conducted on multimodal or speech data, which limits evidence of its broader applicability.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
