# Massive Editing for Large Language Models Based on Dynamic Weight Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Knowledge Editing (KE) is a field that studies how to modify some knowledge in Large Language Models (LLMs) at a low cost (compared to pre-training). Currently, performing large-scale edits on LLMs while ensuring the Reliability, Generality, and Locality metrics of the edits remain a challenge. This paper proposes a Massive editing approach for LLMs based on dynamic weight Generation (MeG). Our MeG involves attaching a dynamic weight neuron to specific layers of the LLMs and using a diffusion model to conditionally generate the weights of this neuron based on the input query required for the knowledge. This allows the use of adding a single dynamic weight neuron to achieve the goal of large-scale knowledge editing. Experiments show that our MeG can significantly improve the performance of large-scale KE in terms of Reliability, Generality, and Locality metrics compared to existing knowledge editing methods, particularly with a high percentage point increase in the absolute value index for the Locality metric, demonstrating the advantages of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of large-scale knowledge editing (KE) in LLMs, where many new facts must be incorporated without full retraining. Specifically, the authors propose Massive Editing via dynamic weight Generation (MeG), a novel approach that appends a single dynamic neuron to a selected feedforward layer of the model and uses a conditional diffusion model to generate the neuron’s weights for each query. Experiments on two KE benchmarks and three LLMs show that MeG substantially outperforms prior methods in the key metrics of Reliability, Generality, and Locality.

### Strengths
- The paper is in general well-written, with a logical flow that makes the complex methodology easy to follow and understand.
- The idea of generating a weight vector for a new neuron using a diffusion model conditioned on a text query is novel in the KE context.
- The method yields strong performance across multiple models and datasets.

### Weaknesses
- The proposed method employs a hand-tuned entropy threshold ( \epsilon ) in the familiarity network to determine whether a query is relevant, which is an important step in the overall approach. However, the authors do not provide justification for the chosen value of this hyperparameter. A more detailed discussion on how this value was selected and an analysis of the model’s sensitivity to this parameter would be valuable.

- The authors claim that the generation steps can be compressed to only 50 using fast sampling techniques. What exactly is this “fast sampling technique”? Is there rigorous theoretical support for this compression process, or is the number of steps (the ‘50’ here) defined purely empirically? If it is empirically determined, how can you ensure that it is globally effective across different datasets or scenarios? How can you guarantee that compressing the generation steps still achieves performance comparable to that of the original larger number of steps and how do you know that compressing the denoising process to exactly 50 steps is the minimal number required to maintain comparable performance? Does this imply that for each new dataset, you need to iteratively test different numbers of steps from, say, 1000 down to 10, to find the optimal value?

- The efficiency of the diffusion model is somewhat concerning, and I am still unclear about the fast sampling technique mentioned as a solution to this issue. Please provide more detailed explanations regarding this method. Additionally, could you clarify the total computational cost required to prepare MeG for a new batch of edits (e.g., the time needed to collect weights and train the diffusion model)? How does this cost scale with the number of edits?

- MeG adds only one neuron per query at a specific layer. However, is a single neuron sufficient to encode all possible knowledge updates? It is possible that for more complex types of knowledge, for instance, multi-hop reasoning in question-answering tasks (e.g., the MQuAKE dataset [1]) or long-form generation (e.g., the LEME dataset [2]), a single neuron may be inadequate. The authors could further discuss the representational limitations of using a single neuron, or evaluate the proposed method on additional benchmark datasets to assess its broader applicability.

- Could you justify the choice of DiT as the generative model for weight generation? For example, why use DiT instead of alternative generative models such as GANs or VAEs?

[1] Zhong, Zexuan, et al. "MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions." The 2023 Conference on Empirical Methods in Natural Language Processing.
[2] Rosati, Domenic, et al. "Long-form evaluation of model editing." NAACL-HLT. 2024.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
### Summary

This paper proposes a novel framework, MeG, to address performance degradation in massive-scale knowledge editing. The authors identify that existing methods, which modify the model's internal weights, suffer from limited capacity and cumulative interference, leading to poor locality as the number of edits increases. MeG introduces a new paradigm by attaching a single dynamic neuron to a specific FFN layer rather than altering existing parameters. The weights for this neuron are generated conditionally by a pre-trained diffusion model based on the input query. To protect unrelated knowledge, a "Familiarity Network" acts as a router, classifying queries by entropy; irrelevant queries bypass the dynamic neuron (using a "zero neuron"), thus preventing any interference. Experiments on multiple LLMs with up to 10,000 edits show that MeG significantly outperforms baselines, especially in maintaining locality and general model performance.



### Advantages

* The core mechanism of using a diffusion model to generate dynamic neuron weights, rather than permanently modifying internal weights, is an innovative approach to knowledge editing.
* The empirical results demonstrate better scalability and an improvement in preserving locality on 10,000-edit benchmarks, where most baseline methods show degradation.



### Weakness and Questions

* The method's effectiveness hinges on a "Familiarity Network" trained to distinguish edited queries from irrelevant ones based on output entropy, which may not be a reliable proxy for all unseen, non-malicious queries. Could the authors conduct an experiment testing the Locality score on a curated set of unseen but correct facts (which the base model knew) to verify that the entropy-based router does not misclassify them as "irrelevant" and cause the model to fail on them?

* The offline training process for the diffusion model requires a "Knowledge-Weight Collection" step, where optimal neuron weights must be pre-calculated for every single edit in the massive dataset, a process that seems computationally expensive. Would it be possible to provide an analysis of the total offline pre-computation time (including both Knowledge-Weight Collection and diffusion model training) required for the 10,000-edit scenario and compare it to the setup time for baselines like MEMIT?

* The proposed method requires running a 50-step diffusion model for every single relevant query at inference time to generate its unique neuron, which appears to introduce a substantial and non-trivial latency. Could the authors perform an experiment measuring the inference throughput (e.g., queries per second on a batch of edited prompts) and compare it directly to the throughput of a baseline method like MEMIT, which only requires a standard forward pass?

### Strengths
Please see above.

### Weaknesses
Please see above.

### Questions
Please see above.

### Soundness
3

### Presentation
2

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
This paper introduces MeG, a method for massive knowledge editing in large language models (LLMs) by dynamically generating the weights of a single neuron using a diffusion model, conditioned on the editing query. MeG attaches this neuron to specific layers of the LLM at inference time, allowing large-scale knowledge updates while minimizing interference with unrelated knowledge. Additional mechanisms, such as contrastive representation learning and a familiarity network, are incorporated to improve generalization and locality. Extensive experiments are conducted on two benchmarks (ZsRE, COUNTERFACT) and across three LLMs, comparing MeG to leading baselines.

### Strengths
- The paper departs from typical weight-modification-heavy or neuron-expansion approaches. The dynamic generation of a single neuron per query, via a diffusion model, is a creative application of generative modeling tailored for knowledge editing. This removes the static storage overhead of previous neural expansion methods and decouples editing capacity from LLM size.
- The empirical validation, as shown in Table 1 and Table 2, covers different LLMs and datasets at large edit scales, with MeG consistently outperforming baselines, notably excelling on the Locality metric. Table 3 further shows preservation of general abilities on outside benchmarks.
- The methodology is well-formulated mathematically, with explicit loss functions (e.g., InfoNCE for TE, velocity formulation for diffusion objective) and precise inference pipeline descriptions, as shown in Figure 2.

### Weaknesses
- While the method is empirically validated, the theoretical underpinning regarding the maximum knowledge capacity, expressivity, and tradeoffs of allocating only one neuron per edit is weak. For example, in Section 4, while Equation (velocity prediction) and the overall diffusion objective are clearly described, no guarantees or fundamental analysis explain why or when this approach might fail, especially as the number or diversity of edits grows. Are there queries for which a single-neuron intervention is insufficiently expressive? An analysis about intrinsic editability bounds is missing.
- The paper claims scale-invariant interference (see Figure 1 and Section 6.1), but as multiple edits may potentially lead to similar or correlated neurons generated for queries with overlapping semantics, there is risk of implicit interference via the diffusion model's conditioning. There is a lack of quantitative or qualitative assessment of such “semantic collisions”—e.g., how does accuracy/locality behave when editing batches of near-duplicate or contradictory knowledge?
- The paper lacks several important baselines in knowledge editing, particularly AlphaEdit [1] and RLEdit [2], which should be included or at least discussed for a fair comparison.

[1] AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

[2] Reinforced Lifelong Editing for Language Models

### Questions
Same as Weaknesses

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
3

### Summary
The paper proposes MeG, a large-scale knowledge editing (KE) framework for LLMs that attaches a single dynamic neuron to selected FFN layers and uses a diffusion-based weight generator, which is conditioned on an InfoNCE-tuned text encoder and gated by an entropy-based familiarity network to synthesize neuron weights per query on the fly, aiming to support tens of thousands of edits while preserving reliability, generality, and locality. Experiments on ZsRE and COUNTERFACT with Phi-2, GPT-J, and Llama-3 seem to show improvements over multiple baselines.

### Strengths
+ It is indeed interesting to see the paper reframes large-scale KE as conditional weight generation via diffusion for a single attachable neuron, which could address interference accumulation and capacity saturation in inner-weight editing and extra-neuron methods in a principled manner.

+ The authors have conducted extensive experiments with multiple backbone LLMs on multiple datasets, which demonstrate the effectiveness of the proposed method compared with the baselines.

### Weaknesses
- The method relies on a tuned text encoder, a K-way familiarity network with entropy thresholding, and a DiT-based generator, which is complex compared to the baselines.

- The usage of the diffusion module seems not very well justified. I'm curious if we could directly learn a simpler mapping function between the original neuron and the new neuron instead of relying on the potentially unstable diffusion process.

### Questions
Please refer to my summary of weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
