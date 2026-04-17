# Tampering Detection for Pre-trained Encoders Using Fingerprintwins

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Encoder-as-a-Service (EaaS) enables pre-trained encoders to be shared across tasks, reducing cost but introducing integrity risks when models are modified without notice. Detecting tampering is difficult under a strict black-box setting, where the encoder is hidden within unknown pipelines and only application outputs are observable. Existing fingerprinting methods fail under these conditions as they require model predictions or task-specific information.

We present a novel fingerprinting framework for black-box encoder verification, \emph{grounded in a theoretical insight that larger embedding divergence increases the likelihood of downstream output differences}. Building on this principle, we construct \emph{fingerprint twins}—paired inputs that produce nearly identical embeddings on an intact encoder but diverge sharply after tampering. We simulate realistic changes using \emph{importance-aware perturbations} and optimize twins to maximize KL divergence while constraining perturbations within an $\epsilon$-ball for natural appearance. Experiments across datasets and encoder types demonstrate reliable, task-agnostic detection with negligible impact on utility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents a novel method of discovering if pre-trained general encoder weights were changed after the pre-training. It is a hard problem, since during inference the encoder is hidden and an unknown downstream pipeline follows the encoding. The authors propose an fingerprinting scheme based on twined inputs that helps to discover such changes (adversarial or benign). The method is tested with importance-aware perturbations used to mimic adversarial weights tampering.

### Strengths
In general I have a very positive view on this work: 
- it is well-organized and the main idea is clearly presented.
- the work shows interesting insights in small scale like ablation study on subtle tampering or additional control experiment with noisy samples.
- the problem of tampering detection seems interesting.

### Weaknesses
Unfortunately, I have big concerns connected to the experimental setup (W1/W4) and questions about the usefulness of the threat model proposed (W2) and how the tampering mimicking generalize (W3). Those are followed by other issues (W5). 

W1. Resnets pretrained on CIFAR-10 are not up to the task of being a general purpose pre-trained encoders even for small tasks used in this work.  Evaluation should use bigger encoders pre-trained using at least Imagenet to be used as a general purpose for given downstream tasks. (only partial Tab7 results). Most of the experiments cannot be considered real-world use cases (line 445).  


W2. I wonder what are the possible use cases behind proposed/ used threat model. It seems odd to me that non-adversarial party has white access to the model at first - fingerprinting generation, and then only black box during verification. This is an important point as it could change the applicability of the method. 

W3. Applicability of the proposed method and how it generalizes. Is mimicking via I-A perturbation enough to support your claims? E.g. could you explain if small fine-tuning or adding an adapter layer of the encoder would be considered tampering? (Explain in detail experiments like tab 4d.) What is allowed in the downstream pipeline and what is not?

W4. Tab 1 results: some tampered encoders have better performance than untampered ones, some have much worse - to my understanding it shows that the evaluation setup is lacking (see W1). Could you comment on that?

W5. Others:
- Could you write more about limitations of the proposed method? 
- Tested tampering attacks are not described or motivated.

### Questions
In general: weaknesses, especially W1. 

Ad W2. 
- Could you elaborate on the use cases? 
- How to understand the assumption of trusted third party widespread integrity checks unknowingly to adversaries? 
- Could you elaborate on how general is the proposed method? In particular assumptions (lines 162-163) using importance-aware perturbation and the motivation for this in previous works? 
- The models utility for specific downstream task could be improved by fine-tuning, would it be detected as tampering (which would mean that your method hinder the utility)?

Ad W2/W4 
- What would happen if the attacker would be aware of the fingerprinting used?

### Soundness
1

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
This paper addresses the integrity verification challenge of pre-trained encoders in the Encoder-as-a-Service (EaaS) paradigm under strict black-box settings—where only application outputs are observable, and internal model access or downstream pipeline knowledge is unavailable. Existing fingerprinting methods fail here as they rely on model predictions or task-specific information.
For any Lipschitz-continuous downstream function, larger divergence in encoder embeddings increases the likelihood of observable differences in application outputs. Building on this, the authors propose fingerprint twins—paired inputs optimized to produce nearly identical embeddings on intact encoders but sharply divergent embeddings after tampering. To simulate realistic tampering (e.g., backdoor injection, INT8 quantization), they design importance-aware perturbations that target low-importance weights (estimated via layer-wise normalized gradients under contrastive loss), ensuring tampered encoders retain normal performance while inducing detectable embedding shifts.

### Strengths
1-It addresses a practical, under-solved challenge in EaaS—black-box encoder integrity—where existing methods (relying on model predictions) are inapplicable. This aligns with real-world deployment needs (e.g., cloud-based encoder services) and avoids "solution looking for a problem" pitfalls.

2-Proposition 1 (linking embedding divergence to downstream output differences via Lipschitz continuity) provides a rigorous basis for fingerprint twin design, distinguishing the work from heuristic-based methods. The mathematical proof ensures the approach’s generality across downstream pipelines.

3-The combination of fingerprint twins and importance-aware perturbations is novel: Fingerprint twins solve the "downstream-agnostic" problem by focusing on embedding consistency, not task-specific outputs; Importance-aware perturbations mimic real attacker behavior (preserving performance while tampering), ensuring experiments reflect realistic threats.

### Weaknesses
1-The paper only evaluates vision encoders (ResNet-18 pre-trained via SimCLR/ImageNet). EaaS includes language (e.g., BERT) and multimodal (e.g., CLIP) encoders, which have distinct embedding spaces and tampering vectors (e.g., token-level backdoors in language models). Without validation on non-vision encoders, the method’s generalizability to EaaS as a whole is unproven.

2-Experiments only test static tampering (e.g., one-time backdoor injection, quantization). In practice, encoders may undergo dynamic, incremental changes (e.g., continuous fine-tuning for domain adaptation). The paper does not evaluate whether fingerprint twins retain detection ability over time or if retraining is required, limiting its applicability to long-running EaaS deployments.

3-For large-scale EaaS (e.g., cloud services with 1000+ encoders), the method’s fingerprint generation cost is unclear. The paper mentions generating 1000 twin pairs per encoder, but does not report time/compute overhead. Without scalability data, it is unknown if the method is feasible for large deployments.

### Questions
Please see Weaknesses.

### Soundness
3

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
This paper tackles the problem of detecting tampering in pre-trained encoders deployed in black-box settings, where only downstream task outputs are accessible. 
The authors propose fingerprint twins, paired inputs that produce nearly identical embeddings on an intact encoder but diverge after tampering. The method simulates realistic model manipulations via importance-aware weight perturbations and optimizes twins to maximize KL divergence while maintaining visual fidelity. 
Extensive experiments across datasets, architectures, and attack types demonstrate that the proposed fingerprints can reliably distinguish tampered from untampered encoders with only a small number of queries, achieving strong transferability across downstream tasks and minimal impact on baseline utility .

### Strengths
1. Clear motivation and problem setup for strict black-box encoder integrity verification, covering a realistic Encoder-as-a-Service scenario.

2. Novel twin-based fingerprinting mechanism with theoretical connection to Lipschitz downstream transformations.

3. Strong empirical results across datasets (CIFAR-10/100, STL-10, GTSRB, SVHN) and tampering types (backdoor, DRUPE, quantization), including ablations exploring perturbation magnitude and subtle tampering.

4. Well-designed hypothesis-testing framework enabling low query complexity and downstream-agnostic evaluation.

### Weaknesses
1. Clarity of consistency-rate computation. 
The paper heavily relies on the “consistency rate” metric for verification, but the operational definition and exact decision procedure could be more transparent. For instance, how consistency-rate is computed under arbitrary downstream tasks could be explained more formally.

2. Scalability concerns.
Although the method is tested on multiple datasets and even includes ImageNet-pretrained models, scalability to large-scale or higher-resolution encoders (e.g., ViT, 512×512 vision models) is unclear.

3. Overhead concerns. 
The approach involves white-box access and iterative optimization for fingerprint generation; computational cost may become significant for foundation-model-scale encoders and there seems no discussion on this.

4. Perturbation simulation choices.
Manipulation simulation currently relies primarily on importance-aware weight perturbations. 
It would be helpful to see analysis of more subtle real-world tampering, e.g., small-scale fine-tuning, channel-wise scaling, or feature-space transformations.
It is unclear how sensitive the method is to the specific perturbation design, and whether alternative (or more subtle) tampering regimes would produce high-quality twins with similar discriminative power.

5. Limited discussion of encoder-level defenses.
The paper would benefit from a deeper positioning against prior work on encoder-level security and robustness. 
Several recent studies have the integrity or robustness of pretrained encoders[1][2][3]. 
Acknowledging these efforts and clarifying how this method differs, particularly in terms of threat model, deployment setting, and guarantees, would strengthen the narrative and highlight the contribution of fingerprint twins for black-box integrity verification.

[1] Zheng et al, SSL-cleanse: Trojan detection and mitigation in self-supervised learning. ECCV'24

[2] Feng et al., Detecting Backdoors in Pre-trained Encoders. CVPR'23

[3] Bansal et al., CleanCLIP: Mitigating Data Poisoning Attacks in Multimodal Contrastive Learning. ICCV'23

### Questions
Please respond to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
