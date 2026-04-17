# SeVA: Learning to Ask Discriminative Queries for Fine-Grained Visual Recognition

- Decision: Reject
- Scores: 4, 4, 6, 2, 6

## Abstract
Fine-grained visual recognition (FGVR) aims to distinguish categories based on subtle, localized cues. Recent methods use vision–language models to ask questions for visual hints, but typically rely on fixed templates that yield static attributions rather than adaptive, informative queries. This limits their ability to reveal discriminative features critical to fine-grained categorization. In this work, we ask a key question: how can we ask better questions that are context aware, targeted, and dynamically guide visual reasoning? We propose the Anchored Self-Questioning Vision Agent (SeVA), an iterative reasoning framework that combines a visual–question-answering model with two large language models acting as a Questioner and a Reasoner. Rather than extracting surface-level attributions, SeVA begins with a coarse prediction and then actively interrogates the image by generating discriminative, context-sensitive sub-questions. A Verifier highlights relevant regions, and the Reasoner integrates accumulated evidence to refine the prediction over multiple rounds. To ensure stable and effective interaction between these components, SeVA introduces two complementary types of semantic anchors: (i) explicit anchors from prior category names that guide early attention, and (ii) implicit anchors from previous predictions that provide a language-based gradient for progressive reasoning. Experiments on standard FGVR benchmarks demonstrate the importance of asking good questions, enabling SeVA to outperform state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an anchored self-questioning vision agent  (SeVA) for fine-grained visual recognition by emulating the human experts' capability of hypothesis-driven and iterative reasoning. Its questioner-verifier-reasoner architecture can identify the subtle visual differences without relying on labeled data. Besides, the explicit and implicit semantic anchors are designed to ground the reasoning process. The authors conduct extensive experiments on several benchmarks and ablation studies to show the effectiveness of their methods.

### Strengths
1. The proposed self-questioning vision agent seems to make sense.
2. Qualitative visualizations can illustrate the step-by-step reasoning process (Figure 5).
3. The comparison of benchmarks seems to be quite thorough.

### Weaknesses
1. The questioner-reasoner mechanism is not novel enough and has been widely utilized in some work [1][2].
2. Two large language models acting as a questioner and a reasoner induce a high computational cost, which is not discussed.
3. The authors' expression attitude is very casual. Some of the figures and tables (such as Figure 7 and Figure 9) were directly taken from screenshots, which are not clear and greatly affect the viewing experience.
4. The authors merely display some successful qualitative examples, lacking the analysis of failure cases.
5. There is a lack of comparison with existing models that are also reasoning-based, e.g., Visual-RFT, Vision-R1, R1-VL, VLM-R1.

[1] Large Language Models are Better Reasoners with Self-Verification, EMNLP 23.

[2] IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models, EMNLP 23.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes an agent workflow combining a multimodal large language model and two large language model (three roles) to improve the accuracy of fine-grained classification tasks.

### Strengths
S1. I think the method idea is great, very inspiring. The process of human experts performing fine-grained classification is indeed worth learning from.
S2. The writing is quite fluent, and the method description is clear.

### Weaknesses
W1. The experimental setup appears outdated. The Multimodal Large Models (MLLMs) used in the evaluation (MiniGPT4, Qwen-VL, BLIP-2) represent work from two to three years ago. There is now a substantial performance gap between these earlier models and the current state-of-the-art. While it is acceptable for the authors to validate their method on these established models, it is crucial to also demonstrate that the proposed method's effectiveness is maintained when integrated with more recent, powerful MLLMs to prove its continued relevance and robustness.
W2. The motivation for the task framing is insufficient. Fine-grained visual classification is a very mature field where numerous specialized deep learning methods have already achieved accuracies exceeding 90% on standard benchmarks. These methods typically do not rely on the extensive, general-purpose data that MLLMs require and are significantly smaller in terms of model parameters. In this context, what is the justification for employing massive MLLMs for this task? The paper needs to better articulate the significance of this approach. Furthermore, the experiments are notably missing comparisons against these high-performing, "traditional" specialist methods. Including such baselines is essential to properly contextualize the performance and trade-offs (e.g., accuracy vs. model size and data dependency) of the proposed method.

### Questions
Apart from the Weakness section, there are also the following contents.
Q1. How is the integrity of the reasoning path guaranteed? Taking the "Seagull" case in Figure 5 as an example, there is a 5-step reasoning process. Are the questions at these five steps generated entirely by the Large Language Model? If so, what mechanisms are in place to ensure the correctness and relevance of these questions? Specifically, how do you prevent the model from generating logically flawed or irrelevant queries that could lead the reasoning process astray?
Q2. Comparison with General-Purpose Agentic Workflows. There is a considerable amount of existing work on training-free, multi-agent workflows (e.g., "team of experts" models, critic-reviewer paradigms). Could you please add a discussion comparing the design of your framework against some of these classic workflow architectures? While your workflow is specifically designed for the fine-grained classification task, an experimental comparison against these more general-purpose workflows is crucial. Such a comparison would serve to demonstrate the specific value and advantages of your specialized design over a generic, off-the-shelf agentic approach.

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
5

### Summary
Recent FGVR methods use vision-language models to ask questions for visual hints, but typically rely on fixed templates that yield static attributions rather than adaptive, informative queries. The paper proposes an iterative reasoning framework that combines a visual-question-answering model with two large language models acting as a Questioner and a Reasoner. It introduces semantic anchors to guide early attention and provide a structured trajectory for progressive reasoning. Experiments on multiple FGVR benchmarks demonstrate that asking the right questions can achieve superior performance.

### Strengths
1. It is well-motivated to ask better questions that can guide the model to focus on the context-aware discriminative features.
2. It achieves strong results on fine-grained classification benchmarks, demonstrating the effectiveness of the proposed method.
3. It is generally well-written and easy to follow.

### Weaknesses
1. When using the same VQA model (Blip2) and LLM (Qwen-L-7B), the proposed approach lags behind FineR, which lacks analysis.
2. There lacks a complete example for explaining the whole procedure, which should be added for better understanding.
3. Since the proposed framework comprises iterative reasoning, the inference time should be checked.
4. In page 8, Fig. 6 should be modified to Fig. 4.

### Questions
1. How will the incorrect intermediate predictions influence the final answer? For example, if the initial category is incorrect, will the performance drop? Can you show any failure cases and do some detailed analysis?
2. The paper claims that the Verifier highlights relevant regions, can you provide the visualization results that show the attention regions of each sub-question?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to orchestrate a VLM for finegrained visual recognition. In particular, it creates a multi-round iterative process where first a VLM tries to guess a finegrained category from an image. Then a VLM is asked to generate sub-questions about the key-characteristics of that category. These sub-questions are then verified with a VLM. Then these answers are given to a reasoner which assesses whether the initial guessed category is coherent with the answers of these subquestions. If not, more iterations are done. To give more clues to the categories, the authors propose to retrieve the top-K closest categories (in text embedding) which matches the query image embedding; these are called explicit anchors. This enables the sub-questions to be targeted at discriminating semantically similar categories. Finally, the ‘implicit anchors’ keep track scores between the anchors and the original query image but I did not understand from this paragraph how this would work. The scores do change over time somehow.

Finally, in Section 3.5 it looks like the whole process was for annotating training data, since they introduce a predictor based on CLIP embeddings for the test set. This was quite surprising to me since it seems like the whole procedure does fine-grained recognition so I do not see why it would not be applicable to the test set directly. Moreover, if the best test-time classifier is based on CLIP, I am not certain why the training data cannot be annotated with a CLIP-like procedure (since only 3 images per label will not get the label (Section 4: dataset). 

Anyway, an ablation study demonstrates that all of their components improve performance. The benchmark results show that their method is better than FineR while it improves over vanilla VLMS (MiniGPT4, QwenVL, BLIP-2).

[A] Towards Universal Image Embeddings: A Large-Scale Dataset and Challenge for Generic Image Representations, Ypsilantis et al. ICCV’23

### Strengths
* Method outperforms FineR.
* Method outperforms several VLMs when the task is given directly.

### Weaknesses
Major
* I am very confused about the experimental setup. Why does it make sense to have an elaborate process to predict labels on a training dataset, only to use a CLIP-based classifier at test time? Why not use this process also at training time? 
* The paper claims only a limited amount of labels are being used. I strongly disagree with this claim since the VLMs used in this paper are trained on tons and tons of data, possibly including the very datasets which this paper is using. So I think the ‘annotation free’ claim (Fig 2.) is invalid, which breaks a core motivation of the paper.
* Results seem very low compared to the state-of-the-art in finegrained recognition (even though the setting maybe slightly different). For example, in [A] several methods are compared on CARS196. The DINO-v2 embedding (where DINO is trained without any annotated labels) yields 79.5 recall@1 which is basically equivalent to 1-KNN accuracy. CLIP (also used in this paper) yields 82.2 recall@1. In contrast, the highest accuracy in this paper is 64. So the resulting classifier for the whole process seems sub-standard, which suggests that the practical relevance of this paper is rather limited.
* The goal of the paper is not clearly stated. I think the goal is to create a large training set given partially annotated data, which is then used to obtain a CLIP-based kNN classifier. If I am correct, this becomes only clear after section 3.5 did not correspond at all to the assumptions I had upon reading the abstract and introduction.

Medium
* I do not understand the experimental setup. Half of the category names are unavailable. But does this mean that it is known which images belong to the same class? 
* Is the experimental setup equivalent to FineR? Choosing which 3 images per class do not have a label will likely make a huge difference.
* Is the ‘zero-shot’ experiment just using a CLIP k-NN classifier given the fully annotated (and given) training set?

Minor
* Page 2: I do not understand why the LLM-based reasoning is effective in reducing inference time. I would say it is much more expensive than the typical use of embeddings.
* Sec. 3. - overview: why ‘prior category names’? Will they change during the process?
* The ‘iterative reflection reasoning’ was vague. Why is this reflection? An example would help.
* Sec. 4.1 ‘the baseline model’ is left undefined here. At this point it cannot be understood by the reader.

### Questions
Please explain why the core problem addressed in this paper is a valid one.

Please explain why there are such big differences on CARS196 w.r.t. the results in the universal embedding paper ([A] Towards Universal Image Embeddings: A Large-Scale Dataset and Challenge for Generic Image Representations, Ypsilantis et al. ICCV’23)

Please explain why the 'annotation' as presented in this paper cannot be applied to the test set. Or show quantitative results.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents SeVA, an agent-based FGVR system that iteratively refines predictions via anchored questioning with VQA and LLMs. Results show improvements over SOTA on datasets like CUB and Cars.

### Strengths
1. Dynamic queries of this method improve over static ones in FineR.
2. Visualizations (e.g., Figure 1) effectively illustrate concepts.

### Weaknesses
1. Novelty is limited. It builds incrementally on FineR (Liu et al., 2024) without fundamental advances in FGVR. In addition, there is some overlap with recent vocabulary-free FGVR (ICCV 2025), reducing originality.
2. Potential high computational cost from multi-round LLM interactions, not fully addressed.
3. Benchmarks are standard but lack challenging settings like noisy data or cross-domain transfer.

### Questions
1. What is the average inference time per image, and how does it compare to FineR?
2. Could SeVA extend to video-based FGVR?
3. How does performance hold in few-shot FGVR scenarios?
4. Sensitivity to LLM choices (e.g., Qwen vs. GPT variants)?
5. Comparison to non-LLM iterative methods?

### Soundness
3

### Presentation
3

### Contribution
3
