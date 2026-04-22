# Learning an Image Editing Model without Image Editing Pairs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Recent image editing models have achieved impressive results while following natural language editing instructions,  but they rely on supervised fine-tuning with large datasets of input-target pairs. This is a critical bottleneck, as such naturally occurring pairs are hard to curate at scale. Current workarounds use synthetic training pairs that leverage the zero-shot capabilities of existing models. However, this can propagate and magnify the artifacts of the pretrained model into the final trained model. In this work, we present a new training paradigm that eliminates the need for paired data entirely. Our approach directly optimizes a few-step diffusion model by unrolling it during training and leveraging feedback from vision-language models (VLMs).  For each input and editing instruction, the VLM evaluates if an edit follows the instruction and preserves unchanged content, providing direct gradients for end-to-end optimization. To ensure visual fidelity, we incorporate distribution matching loss (DMD), which constrains generated images to remain within the image manifold learned by pretrained models. We evaluate our method on standard benchmarks and include an extensive ablation study.  Without any paired data, our method performs on par with various image editing diffusion models trained on extensive supervised paired data, under the few-step setting. Given the same VLM as the reward model, we also outperform RL-based techniques like Flow-GRPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes NP-Edit, a training paradigm for instruction-following image editing that removes the need for paired before/after supervision. The core idea is to unroll a few-step diffusion generator during training and optimize it end-to-end with differentiable feedback from a vision–language model (VLM) that answers templated Yes/No questions about (i) whether an edit was executed and (ii) whether identity/context was preserved; a distribution-matching distillation (DMD) loss to a text-to-image teacher constrains realism. The method trains a lightweight 2B-parameter few-step editor (4 steps by default), with results reported on GEdit-Bench and DreamBooth customization showing competitive performance with larger supervised baselines under few-step sampling, plus ablations showing the complementary roles of VLM feedback and DMD.

### Strengths
* **Originality.** The paper is, to my knowledge, the first to train a general instruction-following image editor using **gradient** feedback from a VLM (rather than scalar RL rewards or hand-crafted objectives), coupled with DMD to keep edits on the teacher manifold; this is a non-trivial and timely contribution in the post-training space.  
* **Quality.** The technical formulation is clear: a two-step unroll to produce cleaner intermediate states for reliable VLM judgments; a binary-logit loss over Yes/No tokens; and a KL-motivated DMD term implemented with an auxiliary velocity predictor. Ablations convincingly show that removing either VLM loss or DMD substantially degrades performance.    
* **Clarity.** The paper supplies an end-to-end algorithm box, precise templates for VLM prompts/questions, and concrete training hyperparameters (optimizer, learning rate schedule, warm-up, weighting, sampling times), all of which aid reproducibility.  
* **Significance.** Competitive results with a 2B few-step editor against much larger systems under few-step sampling, and solid customization performance, suggest a practical path toward data-efficient editing without paired supervision. The explicit focus on few steps is also valuable for latency-sensitive applications.

### Weaknesses
* **Evaluation dependence on automated judges.** Most quantitative evaluation relies on VIEScore (GPT-4o-based) and related automated metrics; no human perceptual study is presented. This raises concerns about metric bias and circularity (a VLM both trains and evaluates), especially for nuanced fidelity judgments. Adding a user study or double-blind human pairwise comparisons would strengthen claims. 
* **VLM supervision reliability and calibration.** The paper shows VLM judgments are unreliable on noisy/blurry intermediates—motivating few-step training—but broader issues remain: sensitivity to prompt wording and model-specific biases. More analysis of calibration or robustness (e.g., multiple VLM judges, temperature/decoding settings) would be helpful.  
* **Resource/engineering overhead.** Keeping a VLM in GPU memory is noted as a limitation; concrete throughput/memory numbers for typical hardware (and at training vs inference) are not reported, which makes it harder to assess deployability.

### Questions
* **Judge diversity.** Beyond LLaVA-OneVision-7B, what happens if you ensemble multiple VLMs (or mix families) as teachers during training? The scaling table shows benefits from stronger VLMs; an ensemble could mitigate single-judge biases—did you attempt this?  
* **Human evaluation.** Can you provide a small-scale user study (e.g., 100 instructions × 3 systems, pairwise preferences for edit correctness and fidelity) to corroborate VIEScore-based gains and address potential metric bias? 
* **Failure modes by edit type.** The ablations suggest that Removal and certain style edits are harder. What specific misbehaviors occur (e.g., partial removal, texture spillover), and can targeted question templates or auxiliary losses mitigate them?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces NP-Edit, a new framework for training image editing diffusion models without any paired supervision. Instead of relying on input–output image pairs or synthetic datasets, NP-Edit leverages direct gradient feedback from Vision–Language Models (VLMs) to guide model optimization. The VLM evaluates whether an edit instruction is followed and whether unchanged regions are preserved, providing differentiable supervision. To maintain realism, the method also employs a Distribution Matching Distillation (DMD) loss to align generated images with the manifold of a pretrained diffusion model. The proposed framework achieves performance on par with supervised image editing diffusion models, despite requiring no paired data. The paper includes extensive experiments and ablation studies examining the influence of VLM backbone choice, dataset diversity, and loss formulation.

### Strengths
1. The paper addresses a well-recognized bottleneck in image editing, dependence on expensive, hard-to-scale paired datasets. The motivation is timely and relevant, given the increasing reliance on generative models in open-world applications.

2. The idea of replacing supervised pairs with gradient-based supervision from a VLM is conceptually elegant. This formulation could generalize beyond image editing to other multimodal generation tasks.

3. Competitive empirical performance under the few-step diffusion setting has been demonstrated.

4. The author also includes solid ablation studies and well-motivated analyses.

### Weaknesses
1. Since supervision is entirely derived from VLM feedback, the model’s performance is directly tied to the accuracy, bias, and robustness of the chosen VLM. Any systematic bias (e.g., cultural or aesthetic preferences) or failure to interpret nuanced edit instructions could propagate into the trained model.

2. Unrolling the few-step diffusion model during training and backpropagating through the VLM introduces potentially high computational and memory costs. The paper lacks a detailed analysis of training efficiency or scalability to high-resolution datasets.

### Questions
Overall, this submission makes a conceptual and practical contribution to the field of text-guided image editing by eliminating dependence on paired training data through the use of VLM-based differentiable supervision. The approach is both innovative and impactful, offering a scalable alternative to traditional supervised paradigms. The authors are suggested to address the above weaknesses to further strengthen the work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims at eliminating the image pairs which are requested by contemporary in-context DiT model, which are expensive to curate and scale. Given only a reference image and an instruction, the NPEdit model is supervised by the VLM's feedback signals. The authors further designed VLM-based editing loss and DMD loss. Rich experiments are conducted and NPEdit achieves state-of-the-art results under few-step setting.

### Strengths
1. The topic of eliminating the requirements of image pairs  is very important for In-context DiTs in image editing task. Because image-text pair are easy and cheap to collect and scale up. Instead, image pairs are expensive and rare. 

2. The model proposes VLM-editing loss and DMD loss, which seems novel. 

3. The results surprise me because according to my personal experience, training in-context DiTs with  image-text pairs usually leads to copy-paste problem. Yet the results on image customization task demonstrate that NPEdit could alter the spatial orientation and location quite well, which is interesting.

### Weaknesses
1. Inadequate experiments on non-rigid editing, which is refered to as 'action' in the paper. NPEdit only demonstrates one non-rigid editing, 'a person waves a hand', which is not persuasive. In a previous benchmark TEdBench[1], Imagic + Imagen[1]  and Forgedit + SD 1.4 [2] could conduct some hard non-rigid instructions on this benchmark, for example,  let the dog sit, jump, or  let a bird spread its wings, let a bird looking backward, close an open book etc.

2. Facial identity preservation does not seem good. VLMs are generally not good at measuring facial similarity, far worse than specific face recognition models like arcface. Using VLMs' judgements as supervision signals may not be accurate. The facial identities in the editing results of NPEdit further confirm this doubt. 

3. The paper did not explain very clearly how to obtain the editing result fed to VLM during training. It seems like an inversion method. This step is very important in the training process and should be elaborated further. 


Reference:

[1] Imagic: Text-Based Real Image Editing with Diffusion Models

[2] Forgedit: Text Guided Image Editing via Learning and Forgetting

### Questions
I  notice that there is  only one  non-rigid editing instruction ( which is referred as 'action') in the paper, " a person waves a hand". How about other non-rigid instructions? In TEDBench[1] editing benchmark,  Imagic[1] and Forgedit[2] could conduct various non-rigid editing, for example, let the dog sit/jump, or  let a bird spread its wings, let a bird look backward, close an open book etc. 

I am willing to adjust my rating if  the authors could  address my concerns.

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
4

### Summary
The paper proposes NP-Edit, a novel framework for training image editing diffusion models without any paired image data. Instead of relying on costly or synthetic edit supervision, NP-Edit trains a few-step diffusion model using direct gradient feedback from a Vision-Language Model (VLM), which evaluates whether an edit follows a natural language instruction while preserving unchanged content. In addition, the method incorporates Distribution Matching Distillation (DMD) to align edited outputs in the manifold of the pretrained T2I model. Experimental results show that NP-Edit achieves editing quality competitive with state-of-the-art methods..

### Strengths
S1. In the Abstract and Introduction section (Sec. 1.), the motivation for training an image editing model using the set of unpaired images are clearly explained. The paper is built on solid and strong motivation and requiredness of the task.

S2. The proposed components (loss functions) are plausible and details of VLM usage is clearly released in the Appendix. The idea of implementing unpaired training setup using the prior knowledge of VLM and manifold constraint is a novel approach.

S3. Related to S2, each loss function plays a critical role for performance improvement, as explained in the Ablation study section (Table 3). The paper clearly addresses the requiredness of each proposed loss term.

S4. The proposed method shows strong qualitative results against baselines, including image customization tasks. Quantitative results also reported, which shows the outstanding performance on GEdit-Bench dataset and on-part evaluation result on customization task.

### Weaknesses
W1. The method relies on the Vision Language Model (VLM) to judge whether the edited results of the model are correct. If the VLM misinterprets the image or instruction, the training signal becomes noisy, which can negatively impact convergence and editing accuracy. A more detailed analysis on VLM feedback dependency is required. How does the bias of VLM affect the model training procedure and overall performance? Is the Distribution Matching Distillation (DMD) enough to alleviate the bias of VLM?

W2. I was wondering if the method is applicable to more difficult tasks, such as 1) object addition or duplication, 2) object moving, and 3) enlarging or shrinking the object size. Extensive experiments with additional tasks is required.

W3. The method does not rely on supervision with paired before-to-after images. However, due to the absence of paired supervision, the edited output may drift from the reference object's identity, or introduce unwanted structural changes. Is there any analysis of the corresponding phenomenon? If the phenomenon that I mentioned is not crucial, please logically explain which part of the method alleviates this.

W4. Quantitative analysis on computation overhead is required. How much time is required to train the model and edit a single image?

### Questions
Please check the weakness section. It would also strengthen the paper to include a comparison with the recently released I2I model *Nanobanana*.

### Soundness
4

### Presentation
3

### Contribution
3
