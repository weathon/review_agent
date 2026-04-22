# Following the Navigation: Enhancing Small Language Models Contextual Reasoning with LLM Guidance

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, excel in contextual reasoning by leveraging extensive world knowledge and deep contextual understanding. However, their high computational costs limit deployment in resource-constrained settings. Conversely, small language models (SLMs) are more computationally efficient but often struggle with contextual reasoning due to limited parameter capacity and challenges like catastrophic forgetting. Existing enhancement methods for SLMs—such as knowledge distillation and data synthesis—still depend on additional training and face inherent limitations. To address this, we propose Navigation, a novel training-free framework that improves SLMs’ contextual reasoning by distilling LLM-derived contextual processing expertise into generalizable navigation templates. These templates, stored in a scalable Navigation database, guide SLMs through a three-stage process—Generation, Utilization, and Update—to locate and process critical information within complex contexts. Experiments demonstrate that our approach yields an average 10.7\% accuracy gain with a template count equivalent to no more than 2.1\% of the dataset size, enabling models such as Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct  to outperform GPT-3.5-Turbo on diverse contextual reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an approach called Navigation to distill LLMs for SLMs for better performance without additional training. The method can be decomposed into three stages, i.e., Generation, Utilization, and Update, and yield better performance.

### Strengths
- Navigation serves as a good simulator for using knowledge from LLMs to help SLMs boost performance.
- The authors provide detailed experimental results on four SLMs to prove the effectiveness.

### Weaknesses
- For the method, the authors lack some important details. 1. During Guidance Generalization, is the same test set where SLMs are evaluated on used? Is it only the questions correctly answered by LLMs used, or are all the questions used for guidance generalization? Is the guidance consisting of only the template or the relevant reasoning part? Now it seems that the LLMs have everything for the given question, and the SLMs should just follow. Also, there is some potential for data leakage where LLMs may provide the final answers. 2. For Navigation Utilization, is it a multi-turn process where the SLMs first fill in the template, then get the answer, or is it an e2e process? 3. For Navigation Update, how do you ensure that no answer or clues are included in the template? Did you do some experiments or check some cases? 
- For the experiments, why do SFT-models achieve lower performance than CoT-only methods? More implementation details should be included. Since the authors use LoRA for Fine-tuning, full SFT should also be included for fairer comparison. Table 7 presents that as the threshold of similarity increases, the number of LLM calls increases as well. Why does this happen? Is it due to the increase in the number of Navigation updates? If so, the comparison with CoT or SFT is not fair since it relies mostly on LLMs rather than on SLMs.
- Though improving the performance of SLMs, the method of using LLMs to write a sketch for SLMs to follow is widely researched, and the limited domain of benchmarks constrains its generalization to other domains like Math or Code.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes "Navigation," a training-free framework designed to enhance the contextual reasoning capabilities of Small Language Models (SLMs). The core idea is to use an LLM to distill its contextual processing expertise into generalizable templates. These templates are stored in a scalable database and retrieved during inference via semantic similarity. This retrieved template then guides the SLM to locate and process critical information within complex contexts, enabling improved reasoning without requiring additional model training.

### Strengths
* Clear Positioning and Baselines: The paper provides a clear discussion and comparison against related methods. It effectively situates the proposed training-free approach against training-based methods like knowledge distillation, supervised fine-tuning, and other LLM-guidance techniques like in-context learning. 

* Significant Empirical Performance: The empirical results are strong and demonstrate the method's effectiveness. The 'Navigation' framework yields significant performance gains across multiple contextual reasoning benchmarks (MuSR, StrategyQA, and HotpotQA).

### Weaknesses
* Missing Key Related Work: A significant weakness is the omission of highly relevant papers that also explore the use of templates to guide smaller models. The authors must discuss and differentiate their work. Given the strong conceptual overlap, a clear distinction from these works, and potentially comparative experiments, is necessary to properly evaluate the novelty and contribution of the paper.

[A] Yang, Ling, et al. "Supercorrect: Advancing small llm reasoning with thought template distillation and self-correction." ICLR 2025

[B] Yang, Ling, et al. "Reasonflux: Hierarchical llm reasoning via scaling thought templates." arXiv preprint arXiv:2502.06772 (2025). 


* Clarity of Cost Analysis: The cost analysis in Section 4.3 and Appendix D.2  is difficult to follow. For example, the methods (navigation, cot, SLEICL, full LLM) can be compared in the inference cost (latency, gflops).

### Questions
* Limitations of Similarity-Based Matching: The framework's reliance on semantic similarity  for template matching may be a limitation. While shown to be effective for the tested narrative and QA tasks, it is unclear how this matching strategy would generalize to more abstract or creative problems (e.g., mathematical proofs). In such domains, the semantic similarity of the problem description may not correlate well with the required reasoning template. 

* Need for Stronger Matching Evaluation: The evaluation of the template matching component could be strengthened. The current ablation study (Section 4.4) demonstrates the importance of the Generation and Update modules  but does not directly measure the accuracy of the similarity-based retrieval itself. A comparison against an "oracle" baseline (e.g., iterating through all available templates to find the one that produces the best outcome for a given problem) would provide a clearer measure of the matching mechanism's efficacy and show how much performance is being lost by the retrieval step.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new way to improve the contextual reasoning capabilities of small language models by giving them a navigation template - a guidance on how to reason - that is generated by a larger network. They present a system that dynamically generates new such templates for novel types of problems and show that this significantly improves the models performance on some task.

### Strengths
- It is a simple yet cool idea that could be practically relevant and result in real-world cost savings.
- The paper is decently well written and mostly easy to follow.

### Weaknesses
I like the idea of this paper and it could probably be of great practical use. Scientifically its impact is medium and not particularly novel or groundbreaking. The evaluations are not broad enough to fully support the generality of the approach and its actual advantage and generalisation to wider tasks. 

**Major**

- **W1 Missing methodological details:**
    - Eq. 1) How do you compute the semantic similarity? Is this a separate encoder model? Which one? If it is the model itself, do you do average pooling of activations to be able to do cosine similarity?
        - In 4.3 it then becomes clear that you actually use sentence transformers. This should be made clear before (“we use sequence embedding models to compute …”). Otherwise the reader is left confused, as you write L199ff “the smaller model identifies the most relevant navigation template by matching the current task scenario to those stored in the Navigation database,” which implies you use the model itself to compute those similarities.
    - On what kind of data do you SFT? You mention that your dataset (MuSR) only contains 250 samples per task; that seems far too little to effectively SFT a model.
- **W2 Model and baseline choice:** Why don’t you compare to a small reasoning model? Or a distillation (e.g. there are R1 distillations of many smaller models out there; for Llama 3 8B you could directly compare your method to the R1 distill). The comparison to GPT-3.5 feels not fully informative, as it is a relatively old model. How about comparing to a state-of-the-art model with ~70B params? Or something like gpt-5-mini? This would make the claim of the paper much stronger and more relevant. Your approach is black-box, so you should be able to run it as well with something like gpt-5-nano + Navigation by gpt-5-high. If you can show clear benefits here, without significantly exploding costs, it would make a strong case for your method to be employed in practice.
- **W3 Better cost analysis:** Since one of your main arguments is cost/latency improvements, you should evaluate the additional cost of your methods more thoroughly. Optimally you would have some claim like “for X% decrease in tok/s we get Y% improvement.” While you do analyze cost, you only focus on the similarity retrieval, which I find interesting because I would think it is negligible in comparison to the generation time. More interesting is:
    - Additional cost of running the large model?
    - Additional cost from an increased context length / how the templates affect the generation length. This holds for the other baselines as well.
- **W4 Evaluation coverage:** The current evaluation seems limited. You claim that your method improves “SLMs’ contextual reasoning” but only evaluate on a small set of (small to tiny) datasets. Why don’t you evaluate on one of the many existing reasoning benchmarks? I would expect this method could also help with more advanced tasks.


**Minor**

- **w5** You refer multiple times to dataset size in the introduction and in particular to a percentage of that size, yet it is unclear what this dataset size is. Which dataset do you mean?
- **w6** L230: That sentence is a bit confusing; e.g. “refine” → “refines”.
- **w7** L682: You repeat the Brown et al. citation and there is something off with the punctuation.
- **w8** Table 1: I would make clearer that the enhancement is done only with DeepSeek-R1, as this is a bit confusing.
- **w9** Section 4.3 completely lacks an introduction and a connection to what it is discussing. Without knowing that these models are sentence-transformer models, it is very confusing to read. You should introduce that you are evaluating the cost of the similarity search presented earlier.
- **w10** Personal preference: I find it slightly confusing to have table captions above the table and figure captions below. Up to you, no negative effect on scores from this.

**Expectation management: ** If you are able to answer all questions, add the missing methodological details, and address the other major concerns, I will consider raising to a score of 6. I will only consider a higher score if you are able to show a significantly extended evaluation of your method that underlines the real practical use on diverse datasets compared to small modern SOTA models, and in particular evaluate the cost of applying your method better.

### Questions
- **Q1** I am a bit confused by the choice of models. They all seem a bit outdated. In particular, no Gemma 3, Qwen 2.5 instead of 3. Also, you mainly add GPT-3.5-Turbo to show that the small models can outperform it, right?
- **Q2** Am I right that for MuSR you have 3 categories and your process creates 8 templates?
- **Q3** L414: “replacing the generated navigation template with a standard prompt” — which standard prompt? I don’t fully understand the setting “w/o Navigation Generation”.

### Soundness
2

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
This paper addresses the limitations of small language models in handling complex contextual reasoning tasks due to limited capacity and catastrophic forgetting. It introduces a novel training-free framework, Navigation, which enhances small language models  by leveraging generalizable reasoning templates distilled from LLMs The Navigation framework operates in three stages. Experimental validation demonstrates that this method significantly improves small language models' performance.

### Strengths
- The paper introduces a novel training-free framework called Navigation, which enhances small language models (SLMs) by leveraging structured guidance templates distilled from larger models. Navigation operates through a dynamic, three-stage approach that systematically guides SLMs to locate and use critical contextual information. This innovative approach improves contextual reasoning significantly without additional fine-tuning or training data.

- Clearly quantifies computational costs and demonstrates that navigation template retrieval and updating incur minimal latency and resource use, making the method practical for deployment on resource-constrained devices

- Demonstrates consistent and significant accuracy improvements across diverse contextual reasoning benchmarks, showing gains up. This validation supports the effectiveness and generalizability of the approach.

### Weaknesses
- Only a single LLM is used for template generation, leaving unclear whether the method generalizes across different large model architectures or instruction styles.

- Comparisons are restricted to CoT, SLEICL, and SFT. Other strong distillation and retrieval enhanced baselines are not considered, reducing the fairness and breadth of evaluation.

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
