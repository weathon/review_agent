# Empirical Evaluation of Knowledge Distillation from Transformers to Subquadratic Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Knowledge distillation is a widely used technique for compressing large language models (LLMs), in which a smaller student model is trained to mimic a larger teacher model. Typically, both the teacher and student models are Transformer-based architectures, leveraging softmax attention for sequence modeling. However, the quadratic complexity of self-attention during inference remains a significant bottleneck, motivating the exploration of subquadratic alternatives such as structured state-space models (SSMs), linear attention, and recurrent architectures.
In this work, we systematically evaluate the transferability of knowledge distillation from a Transformer teacher model to eight subquadratic student architectures. Our study investigates which subquadratic model can most effectively approximate the teacher model's learned representations through knowledge distillation, and how different architectural design choices influence the training dynamics. We further investigate the impact of initialization strategies, such as matrix mixing and query-key-value (QKV) copying, on the adaptation process. Our empirical results on multiple NLP benchmarks provide insights into the trade-offs between efficiency and performance, highlighting key factors for successful knowledge transfer to subquadratic architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an empirical study of distilling a Transformer teacher model into a variety of sub-quadratic (linear, recurrent, SSM-type) student architectures. It compares different student designs under a common distillation protocol, explores different initialization/alignment strategies (e.g., attention‐matrix alignment, QKV copy, hidden‐state alignment), and evaluates on a suite of zero-shot and long-context language-modeling tasks. Their findings indicate that xLSTM achieves the best overall results, and that hidden-state alignment is the most reliable strategy for improving transfer.

### Strengths
- The paper tackles an important problem, transferring knowledge from Transformer models to more efficient subquadratic architectures.

- The authors conduct extensive experiments across eight subquadratic architectures.

- The paper explores multiple alignment and initialization strategies (matrix mixing, QKV copying, hidden-state alignment) and even includes diagnostic analyses like head diversity and teacher–student overlap.

### Weaknesses
- Limited novelty: The core KD framework (cross-entropy + KL divergence) and the alignment terms largely build on existing methods like MOHAWK. 

- The writing can be improved for clarity and completeness. There are several citation inconsistencies, the paper lacks a dedicated appendix section explaining LLM usage, and the methodology section could be improved.

- Although the paper evaluates several subquadratic architectures, it remains unclear how results generalize to larger models, more diverse datasets, or complex downstream tasks like reasoning, summarization, or instruction-following.

### Questions
- See weaknesses section

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper is about distilling from a LLM into inference-optimized student architectures, designed to mitigate the high computational and runtime cost of quadratic attention in the regular transformer. The paper includes a detailed survey of alternative neural models, and then presents a detailed evaluation of their efficacy in a distillation setting.

Overall the paper lives up to its title, with all its contributions being empirical, however the experimental eval was underwhelming (particularly in terms of model capacities) and didn't surface many exciting findings. See weaknesses, below for elaboration.

Overall it's not clear whether practitioners will get a lot out of the paper in its current state.

### Strengths
The paper is clearly presented, and surveys and empirically compares a wide range of architectures. 

The empirical setup is good, with well chosen evaluations and benchmarks. I liked the inclusion of long-context evaluations, this was the most interesting result which clearly differentiated the various techniques.

The range of techniques for distillation, around KL, attention alignment, hidden state alignment, was very thorough.

### Weaknesses
The experiments were limited to rather toy models (360M parameters). Some work in appendix F moved to larger settings, but the results show an increasing gap in performance with scale, perhaps motivating the authors not to foreground these results. This is acknowledged in line 475, but the hypothesis that matrix mixing will be more effective at larger scales needs to be demonstrated (indeed App F seems to be showing the opposite).

There were no runtime measurements or compute estimates reported (modulo some cursory results in App G, not covering all methods). As this is the crux of the use of sub-quadratic attention alternatives, this seems like a big oversight. I'd like to see detailed accuracy vs speed tradeoffs, including other techniques for increased runtime efficiency, e.g., quantization, pruning, early exit etc.

### Questions
Table 1: S_t is not used anywhere in the equations in the paper, so it's unclear how to align the table with the prose. Please can you clarify.

150: Several of the models referenced in this paragraph, e.g., TransNormer, are not in Table 1. Why are they left off?

189: "[applying KD] avoid[s] the need for expensive pretraining"; sure there's a reason to use KD if we already have LLMs that are unwieldy for inference settings, however the use of these architectures in training directly is also an important experimental setting. For many models, they would be a lot cheaper to train than a full quadratic transformer. 

245: why was a 360M teacher chosen? This is tiny in the context of modern LLMs, for a result to be meaningful I'd expect to see some evaluations of 7B sized models, at least for the headline results.

330: These methods can recover 90-95% of the accuracy of the teacher; can you benchmark this against pruning and quantization methods, which also exhibit an accuracy/speed tradeoff.

405: Can you add Llama-llama to Table 4? It would be interesting to see how the mass on teacher/student varies. Or are these the results reported for the top row, "Smo...(Teacher)"? I'm puzzled how these measures can be applied to a single model, as they require comparison of a teacher and student. What data was used to estimate these values? The test sets?

QKV copying was a bit confusing, was the teacher representations in the KV cache used as a training target in initialising the student parameters? If so, how?

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
This paper is concerned with cross-architecture distillation, specifically distillation from softmax-attention LMs to linear-attention LMs. It investigates whether linear-attention LMs can be properly distilled from softmax-attention LMs, and how design choices should be made to make the best of the distillation.

### Strengths
1. This paper might be first several work that concerns about cross-architecture distillation, specifically distillation from softmax-attention LMs to linear-attention LMs. The observations and conclusions drawn in this paper might be inspiring for intrigued audience.
2. The results offer clear, practical guidance for practitioners. The strong performance of xLSTM/GLA/MetaLA and the identified superiority of hidden-state alignment are directly applicable to future model efficiency efforts.

### Weaknesses
1. The study is confined to mid-sized models (~360M parameters) and a 3B-token dataset. While this is a reasonable scope, it leaves open the question of how these findings generalize to larger-scale models (e.g., 7B+ parameters) trained on massive, trillion-token corpora. The effectiveness of techniques like matrix mixing might change with scale.
2. The work primarily focuses on replacing the attention module while keeping other components (MLP, embeddings) fixed. It does not explore whether the optimal subquadratic student might require co-design of other components (e.g., different normalization schemes) for peak performance.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper performs a large-scale empirical study of knowledge distillation from a Transformer teacher to eight subquadratic student architectures, evaluating alignment strategies such as QKV copying, attention matrix mixing, and hidden-state alignment. Using both short- and long-context NLP benchmarks, the study finds that architectures with dynamic memory (e.g., xLSTM, GLA, MetaLA) best recover teacher behaviors, while kernel-only linear attention struggles. Hidden-state alignment is the most effective KD technique, whereas QKV copying and attention matrix mixing show limited gains. The contribution is primarily empirical and diagnostic, offering valuable insights into cross-architecture KD, though it lacks new algorithms and deeper theoretical analysis.

### Strengths
This paper conducts a valuable and large-scale empirical study on distilling Transformer knowledge into subquadratic architectures, addressing an important and under-explored research question. Its originality lies primarily in systematic problem formulation and comparative execution rather than new methods, offering a useful recombination of ideas applied to an emerging model class. The experiments are broad and well-organized, and the writing clearly contextualizes prior work, though causal explanations and controlled comparisons could be stronger. Despite being mainly empirical, the findings provide practical guidance for efficient model design, making the work a meaningful, though incremental, contribution to the field

### Weaknesses
The main limitations lie in insufficient experimental controls, incomplete validation of efficiency claims, and limited mechanistic analysis. It remains unclear whether the observed gains stem from architectural inductive bias or from scale differences. The distillation setup also lacks strong KD baselines (e.g., MiniLLM, Born-Again KD), making it difficult to assess the relative effectiveness of the proposed alignment pipeline. While the paper reports head-overlap and attention mass metrics, it provides limited explanation for why certain architectures (e.g., xLSTM, GLA) transfer more effectively, leaving representation-level mechanisms underexplored. Finally, all findings rely on a single Transformer teacher (SmolLM-360M), leaving teacher-specific bias untested and the generality of conclusions across other model families (e.g., Mistral, Qwen, OPT) unclear.

### Questions
1. Distillation baselines: Why are strong KD baselines (e.g., TinyBERT, MiniLLM, feature-level KD, progressive KD) omitted? Could these be included for a more competitive comparison?

2. Efficiency validation: Some components introduce quadratic complexity. Can latency/FLOPs/memory profiling be provided to verify the claimed efficiency benefits?

3. Generalizability of conclusions: Experiments use a single Transformer teacher. Have other large teachers (e.g.Mistral, Qwen, OPT) been tested to ensure claims are not teacher-specific?

4. Mechanistic understanding: Can representation-level analyses (e.g., CKA, attention pattern comparison, memory ablation) be added to explain why the method works, not only that it works?

### Soundness
2

### Presentation
2

### Contribution
2
