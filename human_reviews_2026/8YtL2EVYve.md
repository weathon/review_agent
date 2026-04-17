# Allusive Adversarial Examples via Latent Space in Multimodal Large Language Models

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Multimodal large language models (MLLMs) generate text by conditioning on heterogeneous inputs such as images and text. We present allusive adversarial examples, a new class of attacks that imperceptibly encode target instructions into non-textual modalities. Unlike prior adversarial examples, these attacks manipulate model outputs without altering the textual instruction. To construct them, we introduce a practical learning framework that leverages cross-modal alignment and exploits the shared latent space of MLLMs. Empirical evaluation on LLaVA, InternVL, Qwen-VL, and Gemma demonstrates that our method produces efficient and effective adversarial examples, uncovering a critical security risk in multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new class of attacks on Multimodal Large Language Models (MLLMs) that inject hidden instructions into non-text modalities (e.g., images) via latent space alignment, without modifying textual inputs.

### Strengths
- Well written.
- The proposed method is interesting.

### Weaknesses
- This paper assumes all target instructions are order-agnostic, but this is not universally true. 
- This paper does not test high-risk instructions (e.g., "Generate phishing text ", "Misclassify stop signs as speed limits ") that would demonstrate real-world threat.
- No analysis of instruction length is provided.
- This paper claims allusive examples are "hard to detect ", but no evidence supports this.

### Questions
- Please see "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- This paper introduces a novel multimodal jailbreaking attack, 
   - suggestive adversarial samples. 
- By covertly encoding target instructions into non-text modalities, 
   - this attack manipulates the output of MLLMs without altering the textual instructions, 
   - revealing potential security risks in multimodal models.

### Strengths
- Introduces a novel concept, suggestive adversarial samples, focusing on implicitly embedding instructions in the latent space of non-text modalities, thereby filling a research gap on covert adversarial attacks against multimodal models.

- Develops a rigorous theoretical framework that formalizes the core conditions of such samples as theorems and provides mathematical proofs, closing the loop from definition to derivation to implementation.

- Features comprehensive experiments: the method is validated on 13 LVLMs and 8 classes of target instructions, and robustness across architectural differences and conflict scenarios is investigated, demonstrating broad generalizability.

- Proposes an efficient generation approach that only optimizes the image encoder and projection layers to reduce computational cost, making it compatible with existing models and highly practical for engineering deployment.

### Weaknesses
- The paper advocates using the cross-modal alignment mechanism for adversarial attacks but fails to thoroughly analyze the potential vulnerabilities in this alignment mechanism.

- The central claim of the paper is the success of suggestive adversarial samples, yet the experimental results show almost no improvement compared to non-suggestive adversarial samples. Moreover, it does not clearly explain why suggestive adversarial samples are superior to non-suggestive ones.

- The paper conducts tests in four scenarios, but the "specific word insertion" scenario actually overlaps with the semantic category insertion scenario, which lacks rigor.

- The readability is poor, as the paper introduces many unnecessary mathematical symbols. For instance, the experiment only involves two modalities, but the definitions are extended to infinite cases.

### Questions
- The practical gradient-based algorithm for efficiently generating these adversarial examples focuses on minimizing perturbations in the non-text modalities. Is this the main insight of your paper?

- Is only a contiguous block of image patch-wise tokens modified in the proposed method?

- Are gradients computed only for that patch-wise tokens rather than for the entire image?

- How is the start index $j$  of the block chosen? Is it fixed, random, or optimized?

- The “partial-token, partial-backprop” trick is crucial for keeping the visual change imperceptible. Was this trick first proposed by your team? Have other researchers used a similar approach before?

> Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning

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
This paper proposes a new class of attacks on multimodal large language models (MLLMs), termed Allusive Adversarial Examples (AAEs). Unlike traditional adversarial attacks that alter visible input content, AAEs imperceptibly inject latent instructions into non-text modalities (e.g., images) by exploiting the shared latent alignment space in MLLMs. The authors theoretically formalize the notion of order-agnostic hidden instructions and derive sufficient conditions for such adversarial allusions to occur. They further propose an optimization-based method to construct these examples efficiently, introducing a gradient-descent algorithm that modifies image embeddings to mimic target textual instruction vectors. Experiments across 13 state-of-the-art LVLMs (e.g., LLaVA, InternVL, Qwen-VL, Gemma) demonstrate that AAEs can reliably induce specific behaviors—such as modifying verbosity, injecting words, or enforcing refusals—without explicit text manipulation. The results suggest that these attacks are both effective and stealthy, raising concerns about latent vulnerabilities in multimodal systems.

### Strengths
- The concept of “allusive” adversarial examples that operate through latent multimodal alignment rather than explicit surface perturbations is genuinely new and thought-provoking. 
- The experiments encompass a wide range of popular open-source LVLMs, improving generality and credibility of the findings.
- Figures illustrating latent perturbation effects and qualitative outputs (e.g., “nice” example in Fig. 7) make the attack mechanism intuitively understandable.

### Weaknesses
- While the authors claim that a “subsequence of image embeddings overlaps with textual instruction vectors,” they do not provide quantitative visualization (e.g., cosine similarity heatmaps, embedding trajectory analyses) to directly confirm that latent alignment.
- The experiments primarily compare non-allusive (visible) vs allusive attacks, but do not benchmark against prior adversarial defense or detection methods for MLLMs. Hence, it is unclear how severe these attacks are relative to known perturbation-based methods.
- The “order-agnostic” property is central to the theory but is not measured or verified experimentally. The paper assumes that LVLMs process hidden instructions invariantly to order, which may not universally hold.
- The study assumes access to gradients through the image encoder and projection modules, which is unrealistic threat model for most deployed MLLMs, which are closed-source and API-based.
- The formal sections (Def. 1–3, Thm. 2–4) occupy large portions of the paper but contribute little to actionable understanding of the real vulnerability surface or countermeasures.
- Claims that perturbations are “imperceptible” are unsubstantiated; there is no user study or perceptual metric to ensure visual indistinguishability.

### Questions
- How robust are these allusive adversarial examples under input transformations such as compression, cropping, or resizing? Does the attack survive such perturbations?
- Can the authors provide quantitative evidence (e.g., embedding cosine similarity plots) demonstrating that the injected latent subsequences indeed align with target textual embeddings?
- How realistic is the gradient access assumption? Could the same effect be approximated through black-box optimization using only model outputs?
- What are the potential defense directions to mitigate such attacks?
- For “imperceptible” claims, can the authors report PSNR/SSIM metrics or show visual difference maps between original and adversarial images?

### Soundness
3

### Presentation
4

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
The article focuses on cross-modal alignment for constructing adversarial examples in multimodal models by leveraging features in the aligned feature space. An optimization framework and extensive experiments further demonstrate the effectiveness of the proposed approach.

### Strengths
1. Proposes a new class of adversarial attacks leveraging cross-modal latent alignment, exposing vulnerabilities beyond single-modality perturbations.
2. Provides rigorous definitions of allusive adversarial examples, including order-agnostic instructions and feasibility conditions.
3. The paper presents a clear and well-organized structure with experiments showing strong ReI on multiple MLLMs and intuitive illustrative cases.

### Weaknesses
1. The current description only mentions order-agnosticism with respect to textual instructions, without clarifying whether allusive adversarial examples affect image interpretation.

2. The explanation of how adversarial strength is controlled is not sufficiently specific.

3. The experiments on the generalization ability of the proposed adversarial examples are insufficient.

4. The application scenarios of these allusive adversarial examples need further discussion.

### Questions
1. Do allusive adversarial examples function as instructions inserted in image content, and does this impair image interpretation despite claimed order-agnosticism?

2. How is adversarial strength controlled—by adjusting the visual segment length $(j:j+l-1)$ in Eq. (1), or the instruction length $I_t$?

3. How does the generalization ability of your adversarial examples compare to the baseline gradient-based method? From a user perspective, is there a noticeable difference in how "benign" the examples appear between the two methods?

4. What are the intended application scenarios for the four target behaviors, given their relatively low harmfulness? What is the computational cost of injecting the same instruction into a single example?

### Soundness
3

### Presentation
3

### Contribution
2
