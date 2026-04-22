# Caduceus: MoE-enhanced Foundation Models Unifying Biological and Natural Language

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Multi-modality pre-training on protein sequences with textual descriptions has enabled general-purpose protein language models. However, as the property descriptions span heterogeneous domains, we observe a severe *data interference phenomenon*: distinct protein residues often target domain-specific annotations, revealing partially inconsistent functional mechanisms across sources, which substantially leads to degraded performance. This paper addresses this overlooked issue with a novel *Mixture of LoRA Experts (MoLE)* architecture, by efficiently fusing the knowledge across diverse property domains. Concretely, we introduce **Caduceus**, a family of MoE-enhanced foundation models built with a hierarchical pre-training paradigm to jointly integrate biological and natural language. Employing a property-guided gating router that assigns domain-specific protein tokens to different experts, the dual-granularity alignment approach reconciles signals across diverse functional mechanisms. To extend generalization beyond particular tasks, we further incorporate a multi-task instruction tuning phase, enabling robust protein parsing and natural language question answering. Extensive experiments on 17 mainstream benchmarks demonstrate that Caduceus mitigates the intrinsic data interference and consistently delivers the optimal performance. The instruction-tuned Caduceus-Instruct provides precise protein elucidation, significantly surpassing GPT-5, DeepSeek-V3, and Galactica-30B. All the models, source codes, and collected corpus will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work propose that current multimodal pre-training combining protein sequences and diverse textual attribute descriptions suffers from severe data interference—inconsistencies in knowledge mechanisms across different attribute domains lead to degraded model performance. To address this, the authors propose Caduceus based on MoE enhancements.  Experiments are conducted to validate the proposed method.

### Strengths
1. The motivation of this work is clear.
2. The problem identified is important: it clearly points out the data interference caused by textual descriptions of multi-attribute proteins, which has been overlooked in previous research on multimodal proteins.
2. This work is well-written and easy to follow.

### Weaknesses
1. The proposed method is very simple and lacks of novelty, applying MoE-Lora to Text-protein multimodal LLM.
2. The improvement in Table 1 might be due to the addition of property desc during pre-training. 
3. Several of the baselines compared were pure sequence models such as GPT-5 and DeepSeek, while the baseline chosen for the QA task was relatively weak. Models like GPT inherently lack the ability to process biological sequences, so the better performance is understandable.

### Questions
1. The analysis of the moe part seems inadequate; I think at least an ablation analysis of the number of experts and the rank of lora should be performed.
2. Comparison with state-of-the-art methods: While comparisons have been made with many methods, a more refined comparison can be made with other recent excellent protein-text multimodal models rather than  pure sequence LLM, under the exact same settings to highlight MoLE’s unique advantages in addressing data interference.

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
4

### Summary
The paper introduces Caduceus, a multimodal protein model that links biological sequences with natural language. It leverages on the Mixture of LoRA Experts framework, using a property-guided router to send protein tokens to specialized experts and reduce interference between biological domains. Training has two stages: dual-granularity alignment, which connects protein and text representations at both global and local levels, and instruction tuning, which enables question answering with a language model decoder. Caduceus achieves strong results on multiple protein and text benchmarks, showing that domain-specific routing and multimodal training improve performance and interpretability.

### Strengths
- Clear motivation: Identifies and quantifies the “data interference” problem across protein property domains.

- Effective adaptation: Extends the Mixture of LoRA Experts [1] with a biologically informed gating router that routes protein tokens by property.

- Dual-granularity learning: Combines global and local protein-text alignment to capture both sequence-level and residue-level meaning.

- Hierarchical pipeline: Two-stage training (alignment, instruction tuning) connects representation learning with natural language generative modeling.

- Good empirical results: Achieves state-of-the-art performance across 15 benchmarks with solid ablations.

- Clarity and presentation: Well-structured paper with intuitive figures and transparent methodology.

[1] Wu et al., Mixture of LoRA Experts, ICLR 2024, https://arxiv.org/pdf/2404.13628

### Weaknesses
- Limited novelty: Core MoLE mechanism is based on prior work (e.g., [1], as already cited in this paper), so the main technical innovation of this paper seems to be the domain adaptation and the pipeline. I would suggest authors clarify and distinguish the main technical innovation of this paper given the prior work. 

- Concern about data overlap: Pretraining and evaluation datasets may share entries, risking leakage. I would suggest the authors discuss how they ensure there is not data leakage

- Claim issue: In the conclusion, line 478, it is writen that "we propose the Mixture of LoRA Experts (MoLE) to effectively...". This may confuse some readers into thinking its a claim of introducing MoLE itself, while MoLE has already been introduced and studied before. The authors may want to rewrite this to avoid potential confusions/concerns. 

- [Minor] Concern related to fair comparison: Instruction-tuned model is compared to zero-shot general LLMs. While this is a reasonable baseline, this may not be a fair comparison.

- [Minor] Gap in validation: It would be interesting to see causal or mechanistic tests (e.g., in silico mutagenesis, binding effect studies).

[1] Wu et al., Mixture of LoRA Experts, ICLR 2024, https://arxiv.org/pdf/2404.13628

### Questions
1. Can the authors clarify and distinguish their main technical innovation given the prior work?

2. Can the authors provide a discussion on how they ensure there is not data leakage?

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
4

### Summary
This paper notices that existing protein language models struggle to deal with proteins' textual descriptions in heterogeneous domains. To mitigate this problem, this paper proposes a mixture-of-LoRA method to fuse knowledge of different domains. Specifically, this paper proposes a hierarchical pretraining method and property-guided gating router. The proposed multi-task instruction tuning also shows effectiveness on benchmark datasets.

### Strengths
1. Overall, the paper is well written, with figures as visual illustrations. The Introduction section clearly explains the motivation behind the method. It also makes a comparison to existing methods and identifies their drawbacks.

2. Using mixture-of-LoRA experts for multi-modal protein language modeling is novel to me, and experiments also show the effectiveness of the proposed method.

3. Experiments are conducted on multiple benchmark datasets. Both quantitative and qualitative tasks are conducted to comprehensively show the effectiveness of the proposed method.

### Weaknesses
1. Usually when we do experiments, we encourage authors to repeat the same experimental setting multiple times and report both mean and standard deviation. However, this paper shows mean but not stddev, which is difficult for readers to judge how significantly the proposed method outperforms baselines.

2. Though this paper proposes an interesting method, it misses to mention and compare to a highly related existing work [1], which uses retrieval-augmented method to integrate knowledge graph with textual descriptions into proteins' amino acid sequences for a multi-modal representation learning.

[1] Zhang, J., Zhang, D. C., Liang, S., Li, Z., Ying, R., & Shao, J. Retrieval-Augmented Language Model for Knowledge-aware Protein Encoding. In Forty-second International Conference on Machine Learning.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents Caduceus, a multimodal foundation model that unifies large language models and protein language models. Traditional approaches to fusing natural language and protein language domains often suffer from data interference issues. The authors attempt to mitigate this by adopting a mixture of experts (MoE) architecture, where the gating mechanism tries to distinguish between language tokens and protein tokens. Details of the alignment and instruction tuning processes are provided for the development of Caduceus. Experimental results demonstrate the effectiveness of Caduceus across both natural language and life science domains.

### Strengths
- The paper is well written and well motivated overall.  
- Unifying life science and natural language data modalities is an important research direction to pursue.  
- The proposed Caduceus method is clearly presented and easy to follow.  
- The performance of Caduceus appears to be promising.

### Weaknesses
- The development process of Caduceus seems to heavily leverage the QA data described in Section 4.1, which makes it challenging for the model to scale.  
- It is not stated in the paper whether the developed QA dataset will be publicly released for future research, which I strongly encourage the authors to do.  
- When scaling the model size from 650M to 3B, the accuracy gains appear to be marginal, so I am uncertain about the scaling behavior of Caduceus.

### Questions
- DNA sequence models have started to emerge recently, possibly as alternatives to protein language models. For instance, the recent release of AlphaGenome attempts to use DNA sequences to perform downstream tasks directly. I wonder whether Caduceus can be extended to also support DNA sequences.

### Soundness
3

### Presentation
3

### Contribution
3
