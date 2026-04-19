# Revisiting In-context Learning Inference Circuit in Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
In-context Learning (ICL) is an emerging few-shot learning paradigm on Language Models (LMs) with inner mechanisms un-explored. There are already existing works describing the inner processing of ICL, while they struggle to capture all the inference phenomena in large language models. Therefore, this paper proposes a comprehensive circuit to model the inference dynamics and try to explain the observed phenomena of ICL. In detail, we divide ICL inference into 3 major operations: (1) Input Text Encode: LMs encode every input text (in the demonstrations and queries) into linear representation in the hidden states with sufficient information to solve ICL tasks. (2) Semantics Merge: LMs merge the encoded representations of demonstrations with their corresponding label tokens to produce joint representations of labels and demonstrations. (3) Feature Retrieval and Copy: LMs search the joint representations of demonstrations similar to the query representation on a task subspace, and copy the searched representations into the query. Then, language model heads capture these copied label representations to a certain extent and decode them into predicted labels. Through careful measurements, the proposed inference circuit successfully captures and unifies many fragmented phenomena observed during the ICL process, making it a comprehensive and practical explanation of the ICL inference process. Moreover, ablation analysis by disabling the proposed steps seriously damages the ICL performance, suggesting the proposed inference circuit is a dominating mechanism. Additionally, we confirm and list some bypass mechanisms that solve ICL tasks in parallel with the proposed circuit.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a three-stage ICL circuit hypothesis and provides thorough empirical examinations of the existence and significance of these stages. Within this circuit framework, many phenomena are explained, such as how Forerunner Tokens encode input text representations and the bias in input text encoding towards position. These findings present intriguing insights.

### Strengths
- **Originality:** To my knowledge, the three-stage circuit proposed by the authors is a novel contribution.
- **Quality:** The hypothesis put forward is reasonable, and the experiments are thorough with a well-crafted methodology.
- **Clarity:** The arguments and evidence presented in the paper are clear, and the experimental descriptions are appropriately detailed.
- **Significance:** Currently, ICL is one of the most important applications in the LLM field, and understanding the mechanisms behind ICL will greatly aid in enhancing its performance.

### Weaknesses
The three-stage ICL framework appears to have implicit applicability conditions, which I believe should be clarified.

For example, in Fig. 1 on page 2, a few-shot scenario with $ k=2 $ is presented, which indeed fits the three-stage ICL circuit framework. However, in a zero-shot scenario ($ k=0 $), step 1 may still exist, but steps 2 and 3 would not be applicable. In a few-shot scenario with $ k=1 $, steps 1 and 2 might still apply, but step 3 cannot exist.

Therefore, the framework proposed in this paper should be limited to discussions of scenarios where $ k \geq 2 $. A related question arises: if the focus is restricted to this scenario, what potential issues might emerge?

Furthermore, if we condition on $ k \geq C $ (where $ C $ is a fixed value), could this value vary depending on the problem type? For instance, in tasks like SST-2 and SST-5, which have different label set sizes, might the value of $ C $ differ across these scenarios?

### Questions
The three-stage framework proposed in the paper is quite interesting. The questions I have raised are mentioned in the weaknesses section. Here, I would like to know what inspired you to propose this framework. Each part of the framework consists of very specific ideas—were they derived from repeated trial and error, or were they inspired by something else? Alternatively, are they improvements on a significant prior work?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the mechanisms within large language models (LLMs) that enable in-context learning (ICL) tasks, breaking down the process into three distinct stages: summarization, semantic merging, and feature retrieval/copying. The study employs a variety of experiments across multiple LLMs to validate its findings. Overall, this paper presents valuable insights that can contribute to the field of LLM research, particularly within the ICL community.

### Strengths
1. The findings in this paper are clearly explained. Experimental results and visualizations enhance readability and help the audience follow the study's progression easily.

2. This work is well-connected to existing ICL research, with discussions that compare its findings to prior studies on ICL explainability and demonstration selection.

3. The study uses multiple LLMs, strengthening the generalizability of the findings across different model architectures.

4. The insights provided are thought-provoking and have potential practical implications for ICL applications.

### Weaknesses
1. [Section 3.1] The authors used mutual nearest-neighbor kernel alignment to evaluate LLMs' summarization abilities. However, the term “summarize” lacks clarity. Does it refer to encoding capabilities similar to those in BGE?

2. [Section 3.1] Additionally, the kernel alignment metric may not be sufficiently robust, as alignment scores in Figure 2 range only from 0.25 to 0.35, which is not significant enough. Consequently, the finding on “summarization” may hold only to a limited extent.

3. [Section 4.1 – Copying from Text Feature to Label Token] It is unclear whether the copying mechanism is applied solely to label tokens or if it extends to other tokens within the input. Using results from other tokens as a baseline could provide a more nuanced understanding of the copying process.

4. [Figure 5, Right] After layer 40, the classification accuracy drops significantly. The authors did not investigate potential reasons for this decline. Could it be due to the gradual degradation of copied information?

5. The experimental setup in Section 5.1 is insufficiently detailed. For instance, how many attention heads are disconnected at each layer? Additionally, the experiments lack certain baselines, such as randomly disconnecting some attention heads to observe the impact on model performance.

Despite these questions and weaknesses, I believe this paper still offers meaningful insights.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
4

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
The authors propose a three-step inference circuit to capture the in-context learning (ICL) process in large language models (LLMs):

1. Summarize: Each input (both demonstration and query) is encoded into linear representations in the model's hidden states.

2. Semantics Merge: The encoded representation of each demonstration is merged with its label, creating a joint representation for the label and demonstration.

3. Feature Retrieval and Copy: The model retrieves and copies the label representation most similar to the query's representation, using this merged representation to predict the query's label.

This circuit explains various ICL phenomena, such as position bias, robustness to noisy labels, and demonstration saturation. Ablation studies show that removing steps in this process significantly reduces performance, supporting the dominance of this circuit in ICL. The paper also identifies some bypass mechanisms that operate in parallel, assisting the inference circuit through residual connections.

### Strengths
1. The authors proposed to use the mutual nearest-neighbor kernel alignment of the intermediate representations of LLMs and sentence embeddings produced by another pre-training model to assess the quality of these representations. This method is novel.

2. Extensive analysis has been performed on all three steps of the proposed framework. Possible explanations have also been provided for many phenomena.

3. The experiments are performed with real-world LLMs and datasets, which makes the insights more likely to be useful in practice.

4. The paper is well-written and easy to follow.

### Weaknesses
1. The majority of the analysis is based on associations without verifying their strength and whether those effects are causal. For example, Figure 2 right does not look significant enough for me. The peaks highlighted in Figure 5 also look pretty noisy to me.

2. The causal evidence that the authors provided in the ablation study only shows the effect of deleting the hypothesized important components in ICL. What if unimportant components are deleted? Would they have a similar effect? Only if the unimportant components have a significantly weaker effect on ICL performance, can we draw a causal conclusion that the proposed three-step process dominates ICL.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to explain the mechanisms behind in-context learning (ICL) using the inference circuit framework.

According to the authors, the ICL process consists of three internal steps:
1. Summarize: Large language models (LLMs) encode each demonstration within its corresponding forerunner token,  $s_i$ .
2. Semantics Merge: The semantics of each demonstration and its label are combined into the representation of the label  $y_i$ .
3. Feature Retrieval and Copy: LLMs rely on the accumulated labels  $y_{1:k}$  to respond accurately to the query  $s_q$ , yielding the most appropriate answer.

Each step is empirically validated using methods such as kernel alignment and embedding comparisons. The authors also seek to align their findings with those of prior research, reinforcing the credibility of the arguments presented in this work.

### Strengths
- Attempts to explain the inner workings of ICL, based on reasonable assumptions and investigative tools.
- The findings align with previous work, encouraging readers to accept the claims presented in the paper.
- Visualized results help readers quickly grasp the core concepts and findings of the paper.

### Weaknesses
- While the proposed framework is logical and reasonable, it remains challenging to argue definitively that the core mechanism of ICL follows the assumptions presented in the paper. As noted in Section 5.2, there are exceptions that do not align well with the proposed framework, raising concerns that the explanations may be superficial and fail to capture the essence of ICL. This is understandable, as fully explaining the inner workings of neural networks is inherently difficult, if not nearly impossible.
- I am somewhat unclear about the core novelty of this paper. As I understand it, the primary contribution seems to be the attempt to apply the existing inference circuit framework to the ICL of specific LLMs, including LLaMA 3. In Section 2.1, I did not find explanations that clarify why the procedure conducted in this paper is particularly innovative, compelling, or novel. More comprehensive comparisons with prior work employing induction or inference circuits to illustrate the inner workings of ICL would be helpful to underscore the merits and uniqueness of this study.
- In Section 3.1, sentence embeddings generated by an external encoder (BGE M3) are compared to hidden representations computed by an LLM. Since these two representations come from different models, without any modification or fine-tuning, there is a risk that their vector spaces are not aligned or compatible. This raises concerns about whether this experiment is sufficiently reasonable.

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
