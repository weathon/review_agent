# On the Adversarial Robustness of Discrete Image Tokenizers

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Discrete image tokenizers encode visual inputs in a sequence of tokens from a finite vocabulary. Pre-trained tokenizers, typically trained together with a decoder for image reconstruction, are an increasingly popular alternative to CLIP image encoders for multimodal systems, including encoder-only, encoder-decoder and decoder-only models. However, unlike CLIP encoders, their vulnerability to adversarial attacks has not been explored. Ours being the first work studying this topic, we first formulate attacks that aim to perturb the features extracted by discrete tokenizers, and thus change the extracted tokens. Since the attacks target only the image encoding, they are computationally efficient, agnostic of the downstream application, and effective on classification, multimodal retrieval and captioning tasks. Second, to defend against this vulnerability, inspired by recent work on robust CLIP encoders, we fine-tune popular tokenizers with unsupervised adversarial training, while keeping all other components frozen. While unsupervised and task-agnostic, our approach significantly improves robustness to both unsupervised and end-to-end attacks. Unlike standard supervised adversarial training, our method can generalize well to unseen tasks and data while also being able to directly leverage any amount of unlabeled images. Overall, our work demonstrates that the resistance of image tokenizers to adversarial attacks strongly impacts the robustness in downstream tasks, and presents an important step in developing generalizable and safe multimodal foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the adversarial robustness of discrete image tokenizers, an area noted as underexplored. The authors first propose an unsupervised adversarial attack targeting these tokenizers. Second, based on this attack, the authors propose an unsupervised adversarial fine-tuning strategy to enhance tokenizer robustness.

### Strengths
- This is the first work to systematically study the adversarial robustness of discrete image tokenizers.
- The experimental results demonstrate the effectiveness of the proposed unsupervised adversarial fine-tuning.
- The defense method only requires fine-tuning the tokenizer's encoder, while keeping the much larger downstream components like LLMs frozen.

### Weaknesses
- As shown in Figure 1, the unsupervised attack performs comparably to, or even weaker than, standard end-to-end supervised attacks, especially at small $\epsilon$ values. The authors do not demonstrate that this attack uncovers new vulnerabilities that supervised attacks miss. I didn't see the motivation. Is the only reason unsupervison?
The authors claim the attack is 'unsupervised' and 'task-agnostic'. However, this property is not unique to tokenizer models. Unsupervised attacks on the embedding space of other VLMs like CLIP are also feasible. It is therefore unclear why this 'unsupervised' property is particularly relevant to tokenizers.

- The proposed unsupervised adversarial fine-tuning strategy also appears to lack methodological novelty. The core idea is to minimize the distance between the embeddings of perturbed samples and the embeddings of the original samples (from the original tokenizer) . This is a well-established paradigm in robust representation learning, bearing a strong resemblance to the core idea of works like TRADES [1]. The proposed loss in this paper is almost identical to the robustness part of TRADES. The paper seems to apply a known framework to a new model (tokenizers) without sufficiently discussing its connections and differences from prior work.

[1] Zhang, Hongyang, et al. "Theoretically principled trade-off between robustness and accuracy." International conference on machine learning. PMLR, 2019.

- The paper's core focus is on discrete image tokenizers. However, both the proposed attack and defense operate in the continuous embedding space ($h_i$) before quantization. While the paper mentions the quantization step and its non-differentiable nature, it does not actually exploit or target this discretization process with a unique attack or defense strategy. The current methods seem applicable to any encoder-based VLM (like CLIP) without modification, which weakens the paper's specific contribution to "discrete tokenizers."

### Questions
- Given the results in Figure 1, can the authors further clarify why it was necessary to propose a new unsupervised attack (Eq. 1)  instead of just using known, stronger supervised attacks for adversarial training? If the goal was simply to enable unsupervised training, why not adapt existing unsupervised attacks on embedding spaces (like those for CLIP )?

- Can the authors elaborate on the differences between their training objective and TRADES [1], or other unsupervised/self-supervised AT methods? What is the core methodological innovation of this work?

- Did the authors attempt to directly attack the discrete token indices ($q_i$), perhaps by using Gumbel-Softmax or other gradient estimation techniques to bypass the non-differentiability? Would an attack on the discrete space expose different vulnerabilities than the continuous embedding space attack?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper systematically investigates the adversarial robustness of discrete image tokenizers used in multimodal systems. It introduces an unsupervised embedding-space attack that maximizes the distance between original and perturbed images in the pre-quantization feature space, effectively altering codebook indices without relying on labels or downstream supervision. Building on this attack objective, the authors further perform adversarial fine-tuning by updating only the tokenizer encoder while keeping downstream components frozen, thereby enhancing robustness under ℓ∞-bounded perturbations for models such as FuseLIP and UniTok-MLLM. Comprehensive experiments demonstrate the accuracy–robustness trade-off and cross-task transferability across classification, retrieval, and vision–language generation tasks. The results demonstrates that unsupervised tokenizer-level robustness training can generalize across diverse downstream tasks, achieving both efficiency and task-agnostic applicability, in contrast to traditional end-to-end adversarial training.

### Strengths
- **Problem Motivation and Scope:**  
  The paper is well-motivated, addressing a critical yet underexplored vulnerability in multimodal foundation models. The focus on discrete image tokenizers—now ubiquitous in modern vision and vision-language pipelines—is both timely and highly relevant.  

- **General, Task-Agnostic Defense:**  
  The work introduces an unsupervised adversarial fine-tuning strategy that operates entirely at the tokenizer level and requires only unlabeled images. The resulting robust tokenizers can be seamlessly integrated into architectures such as FuseLIP and UniTok-MLLM, providing broad robustness improvements. Extensive quantitative evaluations cover classification, retrieval, and VQA tasks, while qualitative analyses demonstrate improved resistance to targeted captioning attacks.  

- **Empirical Rigor and Diversity:**  
  The paper provides strong empirical support, including multi-dataset evaluations, ablation studies, and visualizations of adversarial effects. The qualitative analyses (e.g., Figures 3 and 4) compellingly illustrate both targeted and untargeted safety vulnerabilities—such as policy breaches in captioning—and the corresponding effectiveness of the proposed defense.  

- **Clarity and Writing Quality:**  
  The paper is clearly written and well-structured. The motivation and problem formulation are easy to follow, and the experimental results are presented concisely with well-designed figures and tables highlighting key findings. The background and references are appropriate, making the work accessible even to readers less familiar with adversarial robustness research.

### Weaknesses
1. **Minor Grammatical and Typographical Errors:**
   - “captioninig” → “captioning” (Section 4.2, “VQA and captioninig tasks”)  
   - “severaly degraded” → “severely degraded” (Table 4 discussion, “the resulting clean performance is several[y] degraded”)  
   - “unsupervsied” → “unsupervised” (Discussion section, “improve robustness against unsupervsied and end-to-end supervised attacks”)  
   - “imputs” → “inputs” (Related work section, “extend masked modeling losses to visual imputs”)  

2. **No Explicit Runtime or Efficiency Analysis:**  
   The rationale for attacking/fine-tuning at the tokenizer level is partly computational, suggesting notable savings compared to full model adversarial training. However, the paper provides no quantitative runtime or resource comparison, leaving readers uncertain about the true efficiency benefits.

### Questions
**Questions for the Authors:**

1. **Efficiency Comparison:**  
   Since one of the motivations for tokenizer-level adversarial fine-tuning is computational efficiency, could the authors provide quantitative evidence—such as training time, GPU hours, or memory usage—comparing this approach with full end-to-end adversarial training? This would clarify the practical magnitude of the claimed efficiency gains.

2. **Comparison to Other Discrete Adversarial Methods:**  
   Have the authors run, or could they run for the final version, a controlled comparison with other discrete adversarial or robustness-enhancing methods—such as **Discrete Adversarial Training (DAT)** or **manifold regularization** adapted to modern tokenizers? What trade-offs (e.g., computational cost, clean accuracy degradation, robustness generalization) would the authors expect in such a comparison?

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
This paper is the first to study the adversarial robustness of discrete image tokenizers used in multimodal foundation models.
It proposes unsupervised embedding-space attacks that are efficient and task-agnostic, and uses them for unsupervised adversarial fine-tuning to improve tokenizer robustness.
Experiments on classification, retrieval, VQA, and captioning show large robustness gains without harming clean accuracy, with good generalization to unseen tasks and clear safety benefits.

### Strengths
- Novel problem definition: First dedicated work on adversarial robustness for discrete image tokenizers, an important but overlooked component in multimodal systems.
- Well-motivated and efficient method: Attack design is simple, label-free, and computationally less expensive compared to end-to-end attacks.
- Strong empirical coverage: Extensive experiments across diverse downstream tasks and datasets (classification, retrieval, VQA, captioning) validate both attacks and defenses.

### Weaknesses
- The attack and defense methods employed are effective but relatively classical (APGD, standard adversarial training)
- The study currently focuses on a small set of tokenizer architectures (TiTok, UniTok). Exploring a wider variety of tokenizer designs — such as different quantization schemes (VQ vs FSQ), codebook sizes, number of tokens, or hybrid architectures — would help assess whether the proposed defense generalizes across structural variations and reveal design factors influencing robustness.

**Minor Comments**
- Figures could explicitly state attack parameters for clarity.

### Questions
1. Could the proposed robustness improvements hold against newer or structurally different attacks beyond ℓp-norm bounded perturbations?
2. Would your tokenizer-level robustness improvement generalize to structurally different tokenizers, including varied quantization techniques and token lengths?

### Soundness
2

### Presentation
3

### Contribution
3
