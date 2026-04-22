# StochasTok: Improving Fine-Grained Subword Understanding in LLMs

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still struggle disproportionally with seemingly simple subword-level tasks, like counting the number of 'r's in 'strawberry'. A key factor behind these failures is tokenization, which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper, we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to ‘see’ their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs’ downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok’s simplicity allows seamless integration at any stage of the training pipeline, and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes STOCHASTOK, a simple stochastic tokenization scheme that preserves the original vocabulary and introduces *training-time only* reversible splits of existing tokens to expose subword/character structure. The approach aims to improve tokenization invariance without changing model architecture or deployment-time tokenization. Empirically, it delivers consistent gains on curated subword “language games” and cross-tokenizer multi-digit addition while keeping standard LM metrics roughly unchanged on small/medium models; qualitative analyses (e.g., embedding alignment across segmentations) support the intuition. The idea is elegant and easy to integrate (especially for CPT), but evidence on larger models and real-world tasks is currently limited.

### Strengths
1. **Simplicity & compatibility**: Fixed vocabulary, tokenizer-agnostic, and *train-time only* noise makes it a low-friction drop-in (particularly for CPT) with minimal code churn.  
2. **Clear signal on subword skills**: Reproducible improvements on subword-aware tasks and cross-tokenizer arithmetic transfer, with no obvious small-scale regressions on general LM metrics.  
3. **Practical deployment story**: Inference remains deterministic, avoiding train–test segmentation mismatch common to methods that alter the tokenizer itself.  
4. **Plausible mechanism**: Training with multiple valid segmentations encourages segmentation-invariant internal features; analyses suggest layer-wise convergence toward shared representations.

### Weaknesses
1. **External validity at scale**: No results on ≥4B/7B models; claims of robustness and easy integration would be stronger with short-budget CPT evidence at those scales.  
2. **Evaluation scope**: “Real” math (e.g., GSM8K/MATH) and broader tasks (MMLU/BBH/code/RC) are missing; current gains mostly establish tokenization invariance rather than end-task improvements.  
3. **Compute/cost parity**: No end-to-end cost curves (length inflation, throughput, VRAM, wall-time) or budget-matched comparisons, making Pareto efficiency unclear.

### Questions
1. **Scale-up sanity check**: Can you run short-budget CPT on one **4B** and one **7B** model (e.g., 3–10k steps, \(p \in \{0.05, 0.1\}\)) and report subword tasks, a small **MMLU** slice (no-regression), and **throughput/length/VRAM**?  
2. **“Real” math**: On **GSM8K** (dev is fine), compare deterministic vs **BPE-dropout** vs **STOCHASTOK** under **equal compute**, including training curves and final accuracy; also test cross-tokenizer transfer at this scale.

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
4

### Summary
The paper proposes a new stochastic tokenization approach called StochasTok. StochasTok allows for flexible, alternative tokenizations of the same token. Results on various datasets including language games, pre-training small-scale models (50-275M parameters) offer large performance gains.

### Strengths
* The paper is well written and easy to read.
* Tokenization is a really important and often neglected area of LMs.
* The method proposed is simple and works well
* The paper offers a comprehensive analysis and discussion of experiments and results.

### Weaknesses
* Really minor (without trying to be "Reviewer 2"): the models tested are really small for 2025 standards. There is a risk that the gains might not generalize to larger scales.

### Questions
- Perhaps experimenting with slightly larger models might offer a clearer picture if the gains from your method are similar in larger LMs (e.g. ~1B).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces StochasTok, a novel stochastic tokenization method designed to address a key limitation of large language models: their poor performance on subword-level reasoning tasks. Standard tokenizers treat words as opaque symbols, obscuring the internal character structure and making tasks like counting letters or performing multi-digit arithmetic surprisingly difficult for even state-of-the-art models.

The proposed method operates as a lightweight post-processing step that randomly splits tokens into smaller, valid subtokens from the existing vocabulary, thereby exposing the model to the morphological composition of words during training. The authors demonstrate through extensive experimentation that StochasTok significantly enhances subword reasoning capabilities. Their approach consistently and substantially outperforms strong baselines in both pretraining and finetuning scenarios, achieving near-perfect accuracy on language game benchmarks like LangGame and CUTE, and enabling models to rapidly learn complex tasks such as multi-digit addition.

### Strengths
1. The proposed StochasTok approach offers an elegantly simple yet effective solution to the fundamental limitation of subword understanding in LLMs. Its implementation as either a pretraining enhancement or a lightweight finetuning step makes it highly practical and accessible.
2. The authors provide comprehensive validation across multiple domains - from language games (LangGame, CUTE) to mathematical reasoning (multi-digit addition) - demonstrating the method's versatility and robust performance gains.
3. The paper provides some insights into the method's internal mechanisms through embedding visualizations

### Weaknesses
See questions.

### Questions
1. I acknowledge the contributions of this work. However, from my understanding, StochasTok appears to be a special case of BPE-dropout? Is it true that for every tokenization generated by StochasTok, there exists an equivalent BPE-dropout tokenization? If so, could the authors provide an intuitive explanation for why StochasTok performs so much better than BPE-dropout in the experiments? Are there key differences in how the stochasticity is applied or how the model learns from these variations that lead to the significant performance gap? Could the authors provide a more detailed comparative analysis between the two?
2. The paper states that "In BPE, intermediate tokens not present in the final tokenized training dataset are removed from the vocabulary, meaning BPE-dropout can produce tokens outside the original vocabulary", which I'm not sure. If true, how does it encode user inputs during inference, given this vocabulary mismatch?
3. In Figure 1, the "no pretraining" model shows a rapid increase in both training and validation accuracy very early in training, followed by a sudden decrease. What is the authors' explanation for this phenomenon?
4. In Section 5, which focuses on multi-digit addition, what was the range of digits for the numbers used in the training and validation sets?
5. Could you please report performance on individual subtasks of both LangGame and CUTE (e.g., "Inverse Spelling", "Char Deletion")? This would provide clearer insights into the model's capabilities.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a new tokenization method STOCHASTOK to improve subword-level understanding of language models. The method is simple, efficient, and compatible to existing tokenization methods. Experiments show that pretraining models with STOCHASTOK improves model perofrmance on language game tasks and multi-digit addition. The paper also demonstrates that applying STOCHASTOK during fine-tuning can enhance the subword understanding of already pretrained models.

### Strengths
1. **Clear writing:** the paper is well-structured, clearly written, and easy to follow
2. **Method simplicity and effectiveness:** the proposed method is simple, efficient, and shows promising performance improvements across tasks. 
3. **Comprehensive experiments:** the experiments cover multiple training settings and analyses, and consistenly demonstrate the method's advantages.

### Weaknesses
1. **Limited model scale**: the experiments are conducted only on 50M-parameter models and GPT-2. While these show reasonable performance on simple language tasks (BLIMP & ARC), it remains unlearn whether the improvements generalize to larger models. It would be valuable to see results on larger (1B/7B) models. I understand such experiments are complex and do NOT expect them for the rebuttal. 

2. **Limited evaluation scope**: the current evaluations focus mainly on artificial tasks such as word games and digit addition. It would strengthen the paper to provide more results and discussions on  improvements on more realistic tasks.

### Questions
1. The experiments only reports results for STOCHASTOK with p<=0.1, is this an intentional design choice? How does performance change with higher p?

2. Does STOCHASTOK improves model robustness to spelling errors or out-of-vocabulary words?

### Soundness
3

### Presentation
3

### Contribution
3
