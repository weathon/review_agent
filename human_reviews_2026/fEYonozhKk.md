# AP-OOD: Attention Pooling for Out-of- Distribution Detection

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 8

## Abstract
Out-of-distribution (OOD) detection, which maps high-dimensional data into a scalar OOD score, is critical for the reliable deployment of machine learning models. A key challenge in recent research is how to effectively leverage and aggregate token embeddings from language models to obtain the OOD score. In this work, we propose AP-OOD, a novel OOD detection method for natural language that goes beyond simple average-based aggregation by exploiting token-level information. AP-OOD is a semi-supervised approach that flexibly interpolates between unsupervised and supervised settings, enabling the use of limited auxiliary outlier data. Empirically, AP-OOD sets a new state of the art in OOD detection for text: in the unsupervised setting, it reduces the FPR95 (false positive rate at 95% true positives) from 27.77% to 5.91% on XSUM summarization, and from 75.19% to 68.13% on WMT15 En–Fr translation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AP-OOD, an iteration on OOD detection using Mahalanobis distance with token embeddings [1]. In particular, by replaces mean pooling of token embeddings with learnable attention pooling to improve OOD detection in text models. The authors provide rigorous theoretical background as well as empirical evaluations to showcase the validity of AP-OOD. Additionally, the paper presents the methodology in both its supervised formulation (leveraging auxiliary data) and unsupervised formulation.

[1] Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J Liu. Out-of-distribution detection and selective generation for conditional language models. In The Eleventh International Conference on Learning Representations, 2023.

### Strengths
The reviewer notes the following strengths:
- The paper presents a very clear motivation, with clear justifications on why its nessacary to leverage token-level information beyond mean-pooling.
- The paper is backed with sound theoretical background which helps frame the entire work and adds valuable contribution to the OOD detection field.
- The empirical analysis showcases meaningful improvements upon the baseline, with indications that the methodology can be applicable beyond the base text modality (given audio classification evaluations).

### Weaknesses
The reviewer notes the following weakness:
- Although the improvement is measurable, the incorporation of learnable attention pooling into Mahalanobis distance is not conceptually groundbreaking and may hinder the novelty of the overall work.
- The paper presents few intuitive explanations for overall observations leading to difficulties in finding the novelty of the work.
- The lack of experiments on larger, more realistic language models, potentially makes the work less applicable in real-world settings.

### Questions
The reviewer would like to encourage the authors to revisit the presentation of the overall work. In particular, much of the background can be attributed to Ren et al's paper and could be instead used to visit some of the ablations experiments presented in the appendix [1]. Additionally, more focus on the novelty of the underlying work whether that be the theoretical or algorithmic contribution would help improve the overall quality of the paper.

[1] Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J Liu. Out-of-distribution detection and selective generation for conditional language models. In The Eleventh International Conference on Learning Representations, 2023.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces AP-OOD, a novel method for out-of-distribution (OOD) detection in natural language tasks that leverages token-level information instead of simple average-based embedding aggregation. Operating in a semi-supervised framework, AP-OOD flexibly combines unsupervised and supervised approaches by using limited auxiliary outlier data.

### Strengths
- The motivation for using standard Gaussian-based averaging is well explained.
- The paper presents an interesting and original idea.
- The authors thoughtfully explore multiple use cases to demonstrate the method’s applicability.

### Weaknesses
- The paper suffers from clarity and presentation issues. The introduction does not sufficiently articulate the limitations of existing methods or explain how the proposed approach addresses them. The methodology section is overly condensed and lacks precision, with several symbols insufficiently defined. Moreover, the authors do not reference the accompanying pseudocode. Overall, the manuscript appears hastily written and insufficiently polished.

- Although the approach is presented as semi-supervised, the paper merely demonstrates a transition between supervised and unsupervised scenarios without providing a clear formulation or justification for the semi-supervised setting. Additionally, the proposed supervised out-of-distribution scenario appears limited in practical applicability, as it does not reflect realistic deployment conditions.

- The experimental setup is restricted to a simplistic baseline where embeddings are averaged across sequences, effectively assuming a Gaussian distribution. This assumption is unnecessarily restrictive, as the embedding distribution could be more accurately modeled using mixture models or more expressive probabilistic frameworks such as normalizing flows.

- The concatenation of Z-embeddings (Algorithm 1, line 8) is likely infeasible for larger datasets due to significant memory constraints, which raises concerns about the scalability of the proposed approach.

- The proposed model assumes access to the training data during OOD detection, which severely limits its applicability in realistic or privacy-sensitive scenarios where such access is unavailable.

- The experimental comparison is incomplete. The authors evaluate primarily against standard baselines while omitting several important domain-specific OOD detection methods, such as:

Directed Sparsification – Yiyou Sun and Yixuan Li (2022), DICE: Leveraging Sparsification for Out-of-Distribution Detection.

Virtual-logit Matching – Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang (2022), VIM: Out-of-Distribution Detection with Virtual-Logit Matching (NeurIPS).

GradNorm – Rui Huang, Andrew Geng, and Yixuan Li (2021), On the Importance of Gradients for Detecting Distributional Shifts in the Wild (NeurIPS).

### Questions
Please refer to the weakness section.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors address the problem of out-of-distribution (OOD) detection in natural language processing, highlighting the limitation of mean pooling approaches that lose token-level information. To overcome this, the authors propose AP-OOD, a novel OOD detection method that uses attention pooling to assign higher weights to informative tokens within a sequence. The method builds upon the Mahalanobis distance and introduces attention pooling to better utilize token-level information during OOD scoring. AP-OOD can operate in unsupervised, semi-supervised, and supervised settings and demonstrates improvements over existing baselines across various tasks, including summarization, translation, and audio data.

### Strengths
1. This paper moves beyond prior OOD detection approaches that focused on summarizing the entire sentence through mean pooling, introducing a new method that leverages token-level information via attention pooling.

2. By using a toy experiment to intuitively illustrate why attention pooling outperforms mean pooling, the paper clearly and convincingly conveys the core idea of the proposed method.

### Weaknesses
1. The model requires extensive grid search over β, M, T, and λ, which significantly increases computational cost. Since AP-OOD relies on an attention pooling mechanism, each configuration must be trained separately, making the search process expensive. This can limit scalability and practicality when applied to large datasets.

2. Although the paper claims that the unsupervised setting does not use AUX data for training, it still leverages AUX samples for hyperparameter selection. Moreover, the model’s performance appears highly sensitive to hyperparameter choices. This effectively makes the selection process semi-supervised and may introduce bias if the AUX distribution differs from the OOD distribution encountered at test time. As a result, the chosen model may not generalize reliably across unseen OOD scenarios.

### Questions
1. Attention weights indicate where the model is “looking,” but they do not necessarily reflect the tokens that causally contribute to the OOD decision. Could the authors empirically verify whether higher attention values actually correspond to greater influence on the final OOD score? Could the authors show if tokens with higher attention are indeed more important for the model’s OOD detection?

2. In Section A.2, the paper states that the Mahalanobis distance and the proposed decomposition are “equivalent”.  However, this statement seems mathematically ambiguous. It is unclear whether this equivalence truly holds for arbitrary linearly independent w_j. As currently written, the derivation seems to assume equivalence without sufficient justification. Could the authors explain this step in more detail?

3. The authors set the number of parameters in AP-OOD to match that of the Mahalanobis baseline for a fair comparison. However, it would be interesting to see how much further the performance could improve if the model were allowed to use a larger capacity.

4. AP-OOD employs multiple heads and multiple queries per head to capture diverse token-level informations. However, it remains unclear whether different heads and queries actually learn distinct OOD-related patterns or if they are largely redundant. Could the authors provide qualitative or quantitative evidence showing the contribution of each head to OOD detection? Such analysis would help clarify whether the attention structure learns meaningful diversity rather than merely increasing model capacity.

5. The authors mention that they ensure a fair comparison by matching the number of parameters between AP-OOD and the Mahalanobis baseline. However, this fairness applies only to model size, not necessarily to computational complexity or inference cost. It would be helpful to clarify how much additional computational overhead AP-OOD introduces in practice and whether this extra cost is justified by the observed performance improvements.

Please answer the questions in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper points out the limitations of mean pooling for sequence representations and proposes Attention Pooling (AP-OOD) to compute OOD scores while preserving token-level information. The method uses attention (i) within each sequence and (ii) corpus-wide to estimate representative embeddings, then computes a distance/score between the input sequence’s token embeddings and the pooled corpus statistics.

During training, the backbone LM is frozen and only the AP-OOD module is trained; the approach applies seamlessly from unsupervised (ID-only) to semi-supervised settings using AUX data. The evaluation covers summarization (PEGASUS-LARGE trained on XSUM), translation (Transformer-base trained on WMT15 En–Fr), and audio (MIMII-DG), and reports both input-OOD (encoder embeddings) and output-OOD (decoder embeddings).

In experiment results, the paper shows improved AUROC and reduced FPR@95TPR, with especially large gains over embedding-based alternatives on summarization and audio.

### Strengths
- Replaces mean pooling with a learned attention-pooling formulation for OOD scoring; the paper clearly decomposes Mahalanobis into attention pooling and illustrates the failure mode of mean pooling with an intuitive toy example.

- Broad empirical coverage across two NLG tasks and one audio task; AP-OOD improves AUROC/FPR95 in unsupervised, semi-supervised, and supervised regimes, with consistent AUX scaling curves. The audio experiment demonstrates modality generality (MIMII-DG).

### Weaknesses
- (W1) Scope limited to task specific backbone models: 
The models used (PEGASUS-LARGE or Transformer-base) do not consider modern LLMs (e.g., Llama, Qwen, Phi, etc.). Even if one assumes task-specific language models, today’s practice allows fine-tuning LLMs or few-shot (in-context) learning. Defining ID vs. OOD simply as “training data vs. domain-shifted datasets” can be somewhat unrealistic for contemporary LLM usage.

- (W2) Does not directly address hallucination:
Hallucination is a central issue in language generation. While OOD detection is related to hallucination mitigation, the paper offers no empirical evidence or discussion of how AP-OOD might help reduce hallucinations.

### Questions
**Please provide responses to the reviews mentioned in the Weaknesses section.**

**Additional questions**

(Q1) Application to modern LLMs & hallucination mitigation:
- Why wasn’t AP-OOD validated on advanced open-source LLMs such as Llama, Qwen, or Phi?

(Q2) Input-/Output-OOD for unsafe or malicious prompts:
- (Q2-1) How should AUX be constructed to enable reliable detection of unsafe prompts (e.g., jailbreaks or prompts that solicit illegal activity)?
- (Q2-2) Based on the method and the reported results, do you expect AP-OOD to be effective for blocking malicious inputs (and, by extension, for detecting unsafe outputs)? If so, under what conditions?

### Soundness
3

### Presentation
3

### Contribution
3
