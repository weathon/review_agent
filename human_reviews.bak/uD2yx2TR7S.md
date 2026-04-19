# Vocabulary-Defined Semantics: Latent Space Clustering for Improving In-Context Learning

- Decision: Reject
- Scores: 3, 1, 5, 5

## Abstract
In-context learning enables language models (LM) to adapt to downstream data or tasks by incorporating few samples as demonstrations within the prompts. It offers strong performance without the expense of fine-tuning.
However, due to the context-length restriction, the demonstrations occupy only a small proportion of usable samples. The limitation exacerbates the difficulty of optimization, since the performance of in-context learning can be unstable depending on the quality, format, or order of demonstrations.
Prior work, such as Knn Prompting, index samples based on the similarities of logits at the output-side, in addition to the regular retrieval operation at the input-side.
They improve in-context learning by leveraging the core ability of next-token prediction, rather than relying solely on the emergent capacity to make analogies.
Despite this, the hard-to-optimize issue of in-context learning still exists. In our view, it stems from the process of selecting demonstrations. To address this, we propose complementing in-context learning with an additional clustering operation, making full use of all usable samples.
We propose a novel approach ``vocabulary-defined semantics''.
Grounded in LM vocabulary, which is the label space of model outputs, the proposed approach computes semantically equivalent latent representations for output labels. Then, taking the representations as centroids, a clustering operation is performed to align the semantic properties between the language model and the downstream data/tasks.
Based on extensive experiments across diverse textual understanding datasets and multiple models, our approach outperforms the state-of-the-art in terms of effectiveness and efficiency. On average, it achieves $3\%-49\%$ improvements via the clustering module, while requiring only half of the computation time via the similarity-based logits computation.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Summary:

This work proposes a new demonstrations retrieval method in in-context learning by utilizing the latent space of language models.

Contributions:

The proposed method can outperform traditional random in-context learning and the other work named KP.

### Strengths
1. The proposed method can outperform traditional random in-context learning and the other work named KP.
2. This work explores the possibilities of utilizing the latent space of language models to better understand the downstream tasks.

### Weaknesses
1. The experiment section is quite insufficient. To begin with, it is widely recognized that numerous methods exist to enhance the quality of demonstration retrieval in in-context learning. However, these methods have not been evaluated or listed for comparative purposes. Additionally, the selection of a weak baseline—traditional in-context learning with randomly retrieved demonstrations—raises concerns about the robustness of the findings. The absence of comparative experiments leads to apprehension regarding potential overclaims made by the authors.

2. The related work section is inadequately developed. As previously noted, there exists a substantial body of literature addressing demonstration retrieval in in-context learning that has not been referenced or discussed within this section.

3. Furthermore, it is important to highlight that all datasets employed in this study pertain exclusively to classification tasks, with no inclusion of text-generation tasks. This significant limitation should be explicitly acknowledged in the paper.

4. Last but not least, the writing of this paper can be quite aweful. The authors fail to explain the new terminology which confuses me a lot. Additionally, the references section does not meet academic standards, as it appears to rely solely on citations from Semantic Scholar.

### Questions
1. I doubt whether your method can help Gemma2 so much. The poor results of Gemma2 of KP and ICL with randomly selected demonstrations have implied that it even fails to understand the task. Under this condition, I have no idea why the latent space of Gemma2 can help.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
1

### Summary
This paper proposes a 'latent space clustering' method to improve ICL performance, which is named as VDS (Vocabulary-Defined Semantics). With the incorporation of learnable parameters, this method is reported to achieve significant improvements over vanilla ICL and KNN prompting. A comparison with other PEFT techniques is also presented. While the performance does not surpass IA in all respects, VDS demonstrates notable advantages in terms of reduced training and inference costs. In the conclusion section, the authors suggest that their method "outperforms the state-of-the-art," with additional benefits in computational efficiency and storage requirements.

### Strengths
1. This paper studies In-context Learning, which is an important problem.
2. The authors claim that their method "outperforms the state-of-the-art," while showing advantages in both the computation cost and storage cost.
3. The authors have succeeded in making their work intellectually stimulating, encouraging readers to exercise their analytical skills to decode the presented concepts.

### Weaknesses
1. VDS requires learnable parameters like IA$^3$ but shows inferior performance in all datasets.
2. This method, thanks to the author's description, is sophisticated to understand and implement.

### Questions
Necessary justification should be provided to explain why this "VDS" method should presented in an ineffable manner.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a novel in-context learning method by replacing the indexing operation of KNN prompting with a clustering operation to align the semantic properties between the language model and the downstream data/tasks.  The model constructs semantic bases to represent the label space of model output and introduces similarity-based logits to quantify the semantic gap of the language model with downstream data. Furthermore, the approach innovates by introducing a centroid-known clustering module to optimize the logits, mitigating the semantic gap.

### Strengths
Adopting similarity-based semantic logits with the help of semantic bases to quantify the semantic gap is interesting.

### Weaknesses
1. The authors only compare the proposed method with two baselines in the experiment. However, many in-context learning methods have been proposed in recent two years. The baselines are insufficient.

2. The necessity of similarity-based logits is not particularly convincing. It can be observed in Tab.6 that the improvements brought by sm are limitied.

3. The authors claim that the proposed model is dependent on the quality of latent representations, but this characteristic has not been validated through relevant experiments. 
4. The ablations performs in the paper are unclear. Although some component-level ablations on the model have been implemented, it's unclear what the take-away message from ablations is.

### Questions
1.The results in Table 3 show that the improvement of the proposed method compared to ICL and KP is significant on the CARER, TREC, and WebSS datasets. Can the authors further analyze the reasons for this?

2. I wonder whether the similarity-based logit computation or the clustering module contributes more to the final performance.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper addresses instability issue in in-context learning (ICL) method for language models due to inconsistent quality, format, or order of prompts. It proposes a novel method called "vocabulary-defined semantics" to enhance ICL by clustering semantically equivalent latent representations based on model output labels. This approach aims to optimize demonstration selection and align semantic properties between language models and downstream tasks. Experiments show that the proposed method outperforms state-of-the-art approaches, achieving 3% to 49% improvement in performance while reducing computation time by half.

### Strengths
1. Paper Presentation: The paper is clearly structured and systematically presents the proposed method, making it easy for readers to follow the motivation, methodology, and contributions. The discussion of related work and limitations of existing methods helps set up the context effectively, while the proposed solution is logically introduced.

2. Promising Experiment Results in Classification Task: The experimental setup is comprehensive, utilizing eight diverse text understanding datasets for classification tasks, which demonstrates the robustness and generalizability of the approach. The performance improvement ranges from 3% to 49%, highlights the effectiveness of the method. The comparisons include both effectiveness and efficiency, showing significant performance gains while reducing computation time.

### Weaknesses
1. Limited Evaluation Scope: The method is only evaluated in text classification tasks, despite in-context learning (ICL) being widely used in text generation tasks, such as question answering (QA). Evaluating the approach on these broader tasks could provide a more comprehensive understanding of its effectiveness.

2. Need for More Baseline Comparisons: The paper could benefit from including additional baseline methods, especially considering that ICL methods have been widely studied with different approaches, such as self-reflective retrieval or feedback-augmented ICL.

3. Ablation Study Lacks Analysis of Semantic Basis: The ablation study does not analyze the choice of the semantic basis. It would be beneficial to explore whether using a larger or smaller basis impacts the performance, as this could provide insights into the scalability and flexibility of the proposed method.

### Questions
1. Are there any limitations in applying this method to text generation tasks?
2. How can we determine the appropriate semantic basis, or are there any insights into how the choice of semantic basis affects performance?

### Soundness
2

### Presentation
3

### Contribution
2
