# Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
With the rapid development of large language models (LLMs), identifying efficient strategies for training such large-scale systems has become increasingly critical. Although LLMs have achieved remarkable success across diverse applications, the necessity of maintaining full dense matrices during pre-training has been questioned, giving rise to parameter-efficient sparse pre-training methods which retains parameter-efficiency in both training and inference. These methods can be further divided into connectivity sparse training and spectral sparse training, with dynamic connectivity sparse training and low-rank factorization emerging as representative approaches for the two branches.
However, a unified framework that effectively combines the strengths of both has yet to be established. In this work, we observe that the cancellation effect between the sparse and low-rank branches may limit the expressivity of the model, manifesting as output conflicts when the two components are combined. To address this issue, we first quantify the cancellation effect using the overlap cancellation ratio (OCR) and then propose a novel scheme that integrates dynamic sparse training with low-rank training, introducing a simple yet effective **alignment loss** to mitigate the disagreement between the two branches and promote better collaboration. We validate this scheme by combining a representative dynamic sparse training method, CHTs, with low-rank training, resulting in a new parameter-efficient training approach termed **CHTsL**. The method is evaluated on LLaMA60M and LLaMA130M using the OpenWebText and C4 datasets, where only 10%, 20%, and 30% of the parameters are preserved compared to dense training. Experimental results demonstrate that our proposed scheme effectively alleviates the cancellation effect, especially in the Q and K matrices of the attention layers, and improves training stability and performance compared to the naive combination of sparse and low-rank components. Additionally, the new scheme enables CHTsL to consistently outperform other parameter-efficient sparse training methods under the same parameter budget, achieving performance closest to that of dense training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores a combination of LoRA, activation and dynamic sparse connections to improve pre-training efficiency. It attempts to unify CoLA and CHTs approaches through an alignment loss, aiming to maintain performance while reducing parameter usage.

### Strengths
1. The idea of integrating CoLA with dynamic sparse connections is conceptually reasonable and consistent with recent trends in parameter-efficient pre-training. The use of alignment loss to bridge different sparsity paradigms shows an effort toward unified optimization.

2. The experiments show solid performance across different sparsity levels, demonstrating the stability and robustness of the proposed approach.

3. The paper is well-written and clearly structured, allowing readers to easily follow the methodology and experimental design.

### Weaknesses
1. The proposed method mainly unifies CoLA and CHTs via an alignment loss, which appears to be an incremental combination and is somewhat limited in novelty.
2. The evaluation is somewhat limited. It would be beneficial to include downstream task benchmarks such as HellaSwag and COPA, similar to what was done in CHTS, to better demonstrate generalization capability and practical value.

### Questions
None.

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
5

### Summary
This work proposed CHTSL, an approach that unifies connectivity sparse training and spectral sparse training. They introduces an alignment loss mitigate the disagreement between the two branches and promote better collaboration. Experiments on small-scale LLama validate the efficacy of their approach.

### Strengths
- The paper is written and organized well.
- The idea of alignment loss is reasonable and inspiring.

### Weaknesses
- Lack of literature review. The paper discussed with pruning works, accounting for connectivity sparse training, differentiating against spectral sparse (low-rank) training. But in my opinion, structured pruning actually results in low-rank for the full model. Since the structured pruned models would have many columns or rows zero-out. The paper needs to discuss with structured-pruning-aware works, such as Only-Train-Once.

Only Train Once: A One-Shot Neural Network Training And Pruning Framework

- Lack of sufficient numerical experiments. 

  - The numerical results are conducted under small-scale LLMs. It would be better to conduct over larger-scale LLMs to show the generality. 
  
  -  Besides SiLU, it would be better to show over other activations.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

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
This paper studies the cancellation effect between connectivity-sparse and spectral-sparse branches in dynamic sparse training and introduces an alignment loss. The idea is simple and practical, and experiments show clear gains over baselines.

### Strengths
- This paper identifies the "cancellation effect" and proposes the OCR metric, which provides a valuable quantitative perspective on hybrid sparse training.

- The proposed method is simple and easy to implement.

### Weaknesses
- The alignment loss is conceptually orthogonal to any combination of dynamic sparsity and low-rank training, yet experiments are limited to the CHTs + low-rank setup. Testing additional combinations would be necessary to confirm its generality.

- OCR captures output-level discrepancies but does not fully demonstrate whether alignment mitigates gradient-level conflicts between branches. A more comprehensive analysis at the gradient level is recommended.

- The paper lacks practical efficiency evaluations such as inference memory and throughput.

### Questions
- Is there a correlation between OCR and global cosine similarity? Since OCR measures element-wise sign inconsistency, it may be influenced by local fluctuations rather than true directional cancellation. Such analysis could clarify OCR’s distinct role.

- How does the model's performance differ when the alignment loss is applied only to the Q/K layers compared to applying it across Q/K/V/O or FFN layers?

- Do larger models (e.g., LLaMA-7B) exhibit similar cancellation patterns, and does alignment maintain its effectiveness at scale?

### Soundness
3

### Presentation
3

### Contribution
3
