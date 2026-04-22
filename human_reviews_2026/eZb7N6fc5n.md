# CoT-MT$^3$ : CoT-Guided Meta Test-Time Training for Multimodal Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Multimodal Models (LMMs) have achieved remarkable results across various tasks, but they still face challenges in complex multimodal reasoning that is typically performed via chain-of-thought (CoT).
Recent studies also start to explore the retrieval-augmented few-shot setting to alleviate this problem.
However, existing methods still lack tailored retrieval strategy and effective utilization of demonstrations in complex reasoning scenarios, resulting in limited reasoning improvements.
In this paper, we introduce a novel framework, termed CoT-Guided Meta Test-Time Training (CoT-MT$^3$), to enhance LMMs' few-shot multimodal reasoning ability by employing a CoT-guided Weighted Retrieval (CWR) strategy and a Meta Test-Time Training (MT$^3$) paradigm. 
To provide more relevant demonstrations, CWR employs a retrieval-specific CoT to highlight key information and deep reasoning of the test query for problem-solving. 
Retrieval is then performed based on the weighted similarity of both the original query and the derived CoT cues.
Moreover, to fully leverage retrieved demonstrations, MT$^3$ introduces a context-based meta-learning paradigm by constructing multiple training samples per query with varying context sizes and combinations using few-shot demonstrations.
Experiments across three benchmarks show that our CoT-MT$^3$ achieves a significant relative improvement of up to 4.82\% on MathVerse and 8.38\% on We-Math in the 4-shot setting.
Notably, we also observe that our CoT-MT$^3$ demonstrates exceptional robustness across different context sizes, highlighting its effectiveness and generalization to few-shot reasoning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CoT-MT3, a retrieval-augmented test-time training framework for improving few-shot multimodal reasoning in Large Multimodal Models (LMMs). The method integrates two components: (1) CoT-guided Weighted Retrieval (CWR), which incorporates structured retrieval-specific chain-of-thought (CoT) reasoning into the retrieval process; and (2) Meta Test-Time Training (MT3), which introduces a context-based meta-learning paradigm that trains LMMs using combinations of few-shot demonstrations with varying context sizes. Experiments on three multimodal math reasoning benchmarks—MathVerse, MathVista, and We-Math—show consistent performance gains over existing baselines. The framework demonstrates robustness across multiple few-shot settings and is claimed to be generalizable across diverse reasoning scenarios.

### Strengths
The idea of generating a retrieval-specific Chain-of-Thought (CoT) to improve the quality of retrieved examples makes a lot of sense. It goes beyond just ``question similarity'' and helps the model focus on reasoning-relevant content during retrieval.

The method is evaluated on three different multimodal math reasoning benchmarks (MathVerse, MathVista, and We-Math), across 2/4/6-shot settings. The improvements over strong baselines like TTT-ICL and QBICL are consistent. While the gains are not massive in every case, they’re steady and meaningful.

The paper includes detailed ablation studies on key design choices—such as the weight parameter $w$ in retrieval, the context size $k$ in MT3, and comparisons among different TTT strategies. This shows the authors have carefully examined what parts of the method actually contribute.

Few-shot multimodal reasoning—especially under test-time training constraints—is still an open and under-researched area. This paper proposes a fairly practical and modular framework that addresses it directly, and can likely be applied to related settings with minor modifications.

### Weaknesses
The paper describes MT3 as a ``meta-learning paradigm,'' but it doesn’t really align with the standard definition of meta-learning. There is no inner/outer loop, no bi-level optimization, and no explicit task adaptation process. What the authors actually implement is more of a prompt diversification strategy, where the model is trained with varied context sizes. This is closer to robust context mixing rather than genuine meta-learning. The terminology could easily mislead readers expecting MAML-like or Reptile-style formulations.

The approach heavily depends on the quality of the retrieval-specific Chain-of-Thought (CoT) generated before retrieval. However, no analysis is provided for scenarios where the CoT is flawed or misleading. For instance, if the CoT incorrectly identifies relevant theorems or focuses on the wrong reasoning steps, how does that affect retrieval and final prediction? Some robustness evaluation or failure-case analysis would make the method more convincing.

While the CoT-guided weighted retrieval (CWR) is well-motivated, the paper provides minimal insight into how the weighted combination of question-based and reasoning-based similarity actually impacts retrieval outcomes. The ablation on the weight parameter $w$ is helpful but not sufficient; qualitative examples or analysis of diversity versus redundancy among retrieved demonstrations would add depth. Additionally, the retriever (Vista) is treated as a black box—its suitability for this specific task is not discussed.

CoT-MT3 performs multiple heavy operations at test time: CoT generation, retrieval from a large corpus, sampling of multiple prompt combinations, and fine-tuning with LoRA. Despite this, there is no mention of computational cost, latency, or memory footprint. Without runtime comparisons against baselines (e.g., ICL or TTT-NN), it is hard to judge the practical efficiency of the method.

Although MT3 aims to improve robustness by varying context sizes, it still performs gradient-based updates on a small number of retrieved examples. There is no evidence showing whether the model avoids overfitting or simply memorizes spurious patterns from the demonstrations. A comparison against simpler regularization strategies (e.g., dropout, prompt shuffling) could help clarify this.

Several important hyperparameters and architectural details are fixed without justification. For example, the LoRA configuration (rank 8, $\alpha{=}16$) is not ablated, and the similarity function $\text{sim}(f(x), f(y))$ is not formally defined—are image and text embeddings equally weighted, or are they learned separately? These omissions weaken reproducibility and interpretability.

All experiments are focused on multimodal mathematical reasoning tasks. Since the retrieval-specific CoT is explicitly math-oriented (extracting theorems, listing knowns, etc.), it is unclear whether the same method would work for general multimodal reasoning (e.g., VQA, diagram understanding). A small-scale test on a non-math domain would greatly strengthen the generalization claim.

The paper states that code will be released upon acceptance, but for a complex method combining retrieval, CoT generation, and fine-tuning, reproducibility is difficult without pseudo-code or algorithmic steps. Including a concise algorithm block (for CWR and MT3) would make the contribution clearer and easier to reproduce.

### Questions
You mention that a retrieval-specific CoT is produced for each test query, but it’s not fully clear whether this is generated by the same LMM used for final prediction, or a separate frozen module. Is it generated in a single pass? Is there any sampling or filtering involved? Also, how sensitive is the final performance to the quality of this CoT?

For both question-based and reasoning-based similarity, what exactly is the feature encoder $f(\cdot)$? Is it Vista, Qwen2-VL, or another model? Are the embeddings frozen or fine-tuned? Are text and image features handled jointly or separately?

Why fix LoRA rank to 8 and $\alpha = 16$? It would be useful to know whether these settings are optimal. Did you experiment with other ranks or values of $\alpha$? Would full fine-tuning yield similar gains? An ablation here would help understand where the gains are coming from.


Since your method involves CoT generation, retrieval, context sampling, and LoRA-based adaptation, how long does it take per test instance? Can you provide average latency or throughput numbers, compared to baselines like TTT-ICL or QBICL? How robust is CoT-MT3 when retrieval is poor?
If the top-$m$ retrieved demos are irrelevant or noisy, does MT3 still help? Or does it overfit to bad demonstrations? Did you try testing under retrieval noise or random demo injection?


The CoT prompting template is clearly math-specific (e.g., extract knowns, theorems, etc.). Would your method generalize to other multimodal reasoning domains like VQA, commonsense, or procedural understanding? What adaptations would be needed? How large is the retrieval index and how is it built? You mention using MultiMath-300K for retrieval—do you use the entire corpus? What kind of retriever is used (e.g., FAISS, Vista, dense transformer)? What’s the average retrieval time and memory usage?

Why not include pseudo-code or algorithm blocks? Figure 2 is helpful, but a pseudocode version of MT3 and CWR would really improve clarity and reproducibility. Any plans to add that in the appendix? Did you try disabling reasoning similarity (i.e., $w = 1.0$)? It would be helpful to know how much the reasoning part contributes to retrieval. A comparison between question-only ($w = 1.0$) and your mixed setup would make this more concrete.

\textit{If you’re able to address these questions with clarity and detail, especially around the retrieval/CoT generation process and efficiency, I’m open to increasing my score.}

### Soundness
3

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
This paper proposed a retrieval-augmented framework to improve few-shot multimodal reasoning at test time. It has two pieces: (1) CoT-Guided Weighted Retrieval (CWR), which first prompts the model to produce a retrieval-specific CoT and then retrieves demonstrations using a weighted combination of query similarity and reasoning-similarity; (2) Meta Test-Time Training (MT3), which fine-tunes the model on multiple few-shot variants (varying context sizes and combinations) constructed from the retrieved demos to improve robustness.

### Strengths
* CoT-Guided Retrieval. The paper proposed a novel CoT-guided retrieval approach that enhances relevance compared to standard query-document matching.
* Retrieval-Specific CoT Prompt. The authors designed a retrieval-specific CoT prompt to extract key reasoning information, addressing the limitation of conventional CoT prompts that only focus on the question itself.
* Thorough ablation studies. The paper conducts thorough ablation studies to disentangle the contributions of different components, including the retrieval weight w and context size k. Results show that both CWR and MT³ bring significant gains, the optimal performance achieved at a retrieval weight of $w = 0.7$, a practical recommendation of setting $k = \lfloor m/2 \rfloor$.

### Weaknesses
* Limited Evaluation Breadth. While the paper provides extensive experiments on multi-modal math reasoning tasks, it does not evaluate the generalizability of CoT-MT³ to other multi-modal tasks. Whether CoT-MT³ can transfer effectively to broader multi-modal settings remains an open question.
* Gains Are Uneven and Sometimes Marginal. As shown in Table 1, CoT-MT³ does not consistently outperform all baselines. For instance, it is not the best-performing method under the TD (2-shot) setting. Moreover, the improvements are relatively small in several configurations, such as TD (6-shot), TL (4-shot), VI (4-shot), and the overall 2-shot and 4-shot results.
* Lack of Statistical Significance. In cases where the performance differences are minor (as in Table 1), the paper should report statistical significance to confirm that the observed gains are attributable to CoT-MT³ rather than random variation.
* Compute Cost and Practicality of Test-Time Fine-Tuning. The method involves fine-tuning per test query with multiple prompt variants, which may limit its applicability in real-time or interactive scenarios. The paper lacks a discussion on latency and computational cost, which are essential for assessing the practicality of the approach.

### Questions
1.	What is the average MT³ time consumption per query?
2.	Are the choices of $w = 0.7$ and $k = \lfloor m/2 \rfloor$ universally optimal across different datasets?
3.	Are the observed gains statistically significant when the improvements are marginal?
4.	Can CoT-MT³ be applied to other types of multi-modal tasks beyond science/math domains?

### Soundness
2

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
4

### Summary
The paper proposes CoT-MT3, a framework for few-shot multimodal reasoning that combines CoT-guided weighted retrieval (CWR) and meta test-time training (MT3). CWR enhances retrieval relevance by generating a structured, retrieval-specific chain-of-thought to capture key problem information and reasoning semantics. MT3 improves robustness by meta-training the model on diverse few-shot contexts of varying sizes during test time. The method demonstrates consistent gains across multiple multimodal mathematical reasoning benchmarks and exhibits robustness to context length variations. The experimental design is thorough and the ablation studies are informative.

### Strengths
1. The paper has good experimental design, technical implementation, and result analysis. It conducts evaluations on three mathematical benchmarks, covering different modality dependence levels and diagnostic metrics.
2. The writting in this paper is clear, which makes the method easy to understand and reproduce.

### Weaknesses
1. It lacks direct evaluation of retrieval quality. The paper only reports downstream reasoning accuracy but never directly measures whether the CoT-Guided Weighted Retrieval (CWR) actually retrieves more "reasoning-aligned" examples.  
2. It over-relies on math-specific structures, which limits its generalization claims. The retrieval-specific Chain-of-Thought (CoT) prompt (Figure 3) is explicitly designed for mathematics. Therefore, the paper’s claim of advancing "multimodal reasoning" is at risk of overgeneralization.
3. The core idea of "retrieval-specific CoT", i.e., prompting the model to first generate a preliminary solution plan (identifying key infomation, theorems, and reasoning steps) and then using this plan to retrieve demonstrations with similar reasoning paths, has been explored in existing ICL literature[1]. Therefore, this strategy indeed  lacks significant novelty. Furthermore, the paper fails to provide an ablation study on the four distinct components of its retrieval-specific prompt (Figure 3), making it unclear which of these components are critical for the retrieval improvement and whether the complex prompt design is fully justified.

[1] In-context learning with iterative demonstration selection.

### Questions
1. The paper uses a fixed weight w = 0.7 across all benchmarks and shot settings (Section 4.1). Is this value robust across different domains and model scales? 
2. The retrieval-specific Chain-of-Thought (CoT) prompt (Figure 3) is explicitly designed for mathematical reasoning ("Identify relevant mathematical theorems"). Is this method still effective in non-mathematical domains?  
3. The paper reports downstream reasoning accuracy but does not directly evaluate retrieval relevance. What proportion of the top-m retrieved examples are reasoning-aligned (use the same theorems or strategies)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes CoT-MT^3, an approach which targets to improve LLMs' multimodal reasoning capabilities in retrieval augmented few-shot learning settings. The proposed methodology combines CoT and test-time scaling. First, the proposed methodology implements a CoT-guided strategy to retrieve relevant examples instead of retrieving them conditioned directly on question query only. The second stage, meta test-time scaling, finetunes the large multimodal model (LLM) on a small set of training data which consists of retrieved few-shot examples with different number of demonstrations. The proposed approach is tested on 3 distinct mathematical benchmarks where CoT-MT^3 significantly outperforms previous approaches. The ablation studies cover investigating the contribution of different components, and some hyperparameters.

### Strengths
- Simple yet effective approach. Implementing CoT-MT^3 should be appropriately easy for an average researcher. However, it significantly outperforms previous methodologies.
- The paper will focuses on a currently active and impactful direction.
- Good presentation. The paper is well written, easy to understand, and it contains good self-explanatory figures.

### Weaknesses
- The evaluations only cover multimodal mathematical reasoning benchmarks. Nonetheless, the title and abstract are misleading in this sense. There are experiments on GQA dataset, yet represented in the appendix. It'd be good to push this part to the main text, even include more reasoning tasks with multimodal signals. Another option is to extend limitations section.
- The experiments only cover one vision-language model with 7B parameters (`Qwen2-VL-7B`). It would be interesting to see the significance of model scale (both larger and smaller) and different families (e.g., `Qwen2.5-VL`, `Idefics3`)
- The ablation studies lack an important study, where one could combine question-based querying (QB) with the proposed meta test-time training approach. Fig. 4 illustrates that that MT^3's contribution is much more larger than CoT-based retrieval.
- There are no ablation studies examining the chain-of-thought prompt.

### Questions
- Do you think that the approach could benefit from a per-sample adaptive `w` argument, where `w` takes a different value depending on the question?
- Could you also please share the exact prompt used for GQA evalautions? I think it should be different than the prompt used for other benchmarks, as the prompt Fig. 3 starts with *You are a mathematics expert. I will now provide you with a multimodal math problem.*.

### Soundness
2

### Presentation
4

### Contribution
2
