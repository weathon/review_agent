# Attack as Defense: Run-time Backdoor Implantation for Image Content Protection

- Decision: Reject
- Scores: 6, 5, 5, 8, 5

## Abstract
As generative models achieve great success, tampering and modifying the sensitive image contents (i.e., human faces, artist signatures, commercial logos, etc.) have induced a significant threat with social impact. 
The backdoor attack is a method that implants vulnerabilities in a target model, which can be activated through a trigger.
In this work, we innovatively prevent the abuse of image content modification by implanting the backdoor into image-editing models. Once the protected sensitive content on an image is modified by an editing model, the backdoor will be triggered, making the editing fail. 
Unlike traditional backdoor attacks that use data poisoning, to enable protection on individual images and eliminate the need for model training, we developed the first framework for run-time backdoor implantation, which is both time- and resource- efficient. We generate imperceptible perturbations on the images to inject the backdoor and define the protected area as the only backdoor trigger. Editing other unprotected insensitive areas will not trigger the backdoor, which minimizes the negative impact on legal image modifications. Evaluations with state-of-the-art image editing models show that our protective method can increase the CLIP-FID of generated images from 12.72 to 39.91, or reduce the SSIM from 0.503 to 0.167 when subjected to malicious editing. At the same time, our method exhibits minimal impact on benign editing, which demonstrates the efficacy of our proposed framework. The proposed run-time backdoor can also achieve effective protection on the latest diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel run-time backdoor implantation framework for image editing models, aimed at protecting sensitive content such as watermarks, faces, and logos from unauthorized modifications. Unlike traditional backdoor attacks that require model training or data poisoning, this approach applies protective noise directly to images, activating a backdoor when sensitive regions are edited. This design enables targeted protection while allowing benign modifications in non-sensitive areas to proceed.

The framework’s region-aware trigger mechanism offers selective and minimal interference, enhancing protection precision. Through evaluations on models like LaMa, MAT, and LDM, the method shows significant success in resisting malicious edits, reducing structural similarity and preserving visual quality on authorized edits. Key contributions include a new trigger mechanism based on protected areas, a set of optimized loss functions to improve robustness, and efficient run-time protection without altering model training.

While the approach is innovative and effective, improvements could be made in baseline comparisons, the interpretability of protective noise generation, and broader testing across generative model types. Addressing these areas would further strengthen the paper's applicability and impact.

### Strengths
1. The proposed “run-time backdoor implantation” framework is an innovative and practical approach. Unlike traditional backdoor attacks that rely on model training modifications, this method achieves protection during inference via protective noise. This design aligns well with the need for efficient resource use and ease of deployment, making it valuable both academically and practically.
2. The paper demonstrates strong experimental results across multiple datasets and models (such as LaMa and MAT), particularly in protecting sensitive content like watermarks and brand logos. This effectiveness in preventing unauthorized modifications is especially relevant, given the rising importance of security in generative models. 
3. This mechanism is an excellent feature that ensures only modifications to sensitive regions will trigger the backdoor, allowing benign edits in non-sensitive areas to proceed unaffected. This region-based trigger design not only enhances protection accuracy but also minimizes interference with legal modifications, providing a useful approach for fine-grained content protection.

### Weaknesses
1. While the paper provides some insight into loss functions (such as implant loss and incomplete-trigger loss), it could offer more detailed explanations of how the protective noise is generated and how the backdoor mechanism behaves under varying edit conditions. Additional information on these aspects would increase transparency and reproducibility.
2. While the “hide loss” helps prevent activation in non-trigger areas, testing additional edge cases (e.g., repeated edits or complex mask shapes in real applications) would strengthen confidence in the method’s robustness for real-world usage.
3.The paper does not include direct comparisons to existing baseline methods. Even if there are no direct benchmarks for this specific approach, comparisons with alternative anti-tampering techniques could offer a clearer perspective on the strengths and weaknesses of this framework.

### Questions
1. Although the impact of noise intensity on protection is discussed, further ablation experiments focusing on the contributions of each component (e.g., implant loss, incomplete-trigger loss, hide loss) would offer a clearer understanding of each module’s impact on model robustness and protection. 
2. The paper mainly evaluates performance on LaMa, MAT, and LDM inpainting models, but it’s unclear how well the method generalizes to other types of editing models. Testing with more diverse generative models, such as other visual editing or text-driven generation models, could improve its generalizability.
3. Some key concepts, such as the specific role of protective noise and the execution of the incomplete trigger, could benefit from additional clarification. Expanding on these details would help readers more intuitively understand the method’s logic.
4. Current visualizations illustrate some results, but a broader set of scenarios showcasing the diversity and robustness of the protection could enhance clarity. Adding more visual comparisons across different editing contexts would provide a stronger demonstration of the protection’s effectiveness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces embedding backdoors into images as a form of protection. When users attempt to edit the protected (backdoored) region, significant image quality degradation occurs. To achieve this, the paper designs a new optimization objective for leveraging a pre-trained inpainting model without the need of entire retraining. Experimental results demonstrate its capability to resist image manipulation to a certain extent.

### Strengths
- The paper proposes to use embedded backdoor to achieve editing protection for images.
- A joint objective function is designed to achieve the embedding process of backdoor.
- Experiments are conducted on data from various scenarios, showing that the proposed method can effectively resist editing.

### Weaknesses
Threat Model: I believe there exists a significant gap between the scenario set by the authors and real-world scenarios.
- Firstly, the authors seem to assume that the inpainting model accessed by both the user and defender is the same (please correct me if I misunderstood). If the user adopts an inpainting model different from the one used by the defender to perform the backdoor, does this mean that the protective method proposed in this paper will become ineffective? In other words, the transferability of protective backdoors (handling unknown inpainting models) and scenarios where the inpainting model is a black box should be discussed.
- Secondly, this paper only involves inpainting methods based on learning (such as inpainting models based on GANs or Diffusion) and lacks an evaluation of traditional image editing techniques. For instance, what results would occur when using software like Photoshop to edit regions protected by the method proposed?

Related Work:
- While the proposed method is based on the concept of backdoors, it is similar to image immunization in active defense strategies. Therefore, I believe that including a comparative analysis of the strengths and weaknesses between the proposed method and active defense algorithms (such as [1*, 2*]) would enhance the quality of this paper. Furthermore, in the experimental section, these active defense algorithms could be used as baselines for comparison.

[1*] “Editguard: Versatile image watermarking for tamper localization and copyright protection” in CVPR’24.

[2*] “Draw: Defending camera-shooted raw against image manipulation” in CVPR’23.

Experiments:
- In Sec. 4.2, although the proposed method leads to significant quality degradation when editing the protected region, its impact on areas outside the protected region cannot be overlooked. As shown in the last row of Table 1, for the "watermark" data in the "w/o" case, there can be a PSNR difference of up to 5.64 or 4.69 (in LaMa and MAT, respectively). For other types of data, there are also PSNR differences ranging from 1.21 to 1.91. This substantial difference contradicts the author's statement of being "minimal" (line 416). Therefore, I believe that the proposed method lacks practical feasibility.
- There is a lack of robustness experiments against post-processing. While the authors mention in line 519 that the method can resist operations like cropping and scaling, I could not find relevant results supporting this claim, including in Appendix D. This absence of evidence regarding the robustness of the method against post-processing operations is a significant limitation that should be addressed.

Presentation: This document contains many careless writing errors and lacks proofreading (e.g. inappropriate use of left brackets), especially in the appendix.
- Line 148: “Previous works( ?”
- Line 809: “figure ??”
- Line 832: “LaM( Suvorov et al. (2022))a”
- Line 841: In the appendix, still state “refer to the appendix”
- Line 850: “the figure, …” missing figure number.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel run-time backdoor implantation framework that utilizes protective noise to safeguard sensitive image content in image-editing models, especially against unauthorized modifications. Unlike traditional backdoor techniques, which require training-phase interventions, this approach operates purely during inference, embedding region-specific noise as a backdoor trigger that activates only when sensitive areas are edited. The framework is validated across multiple inpainting models, showing effective resistance to tampering with minimal impact on legitimate editing operations.

### Strengths
1. The use of backdoor mechanisms for defensive purposes is novel and reframes backdoor techniques from a security risk to a protective tool, especially in content-sensitive applications.
2. As the method operates without model retraining, it enables the protection of sensitive areas across various pre-trained models with reduced computational requirements, making it feasible for large-scale applications.

### Weaknesses
1. The effectiveness of the protective noise may be limited to the specific model for which it is designed, raising concerns about adaptability. The noise generated for one model might not be effective for another due to differences in model architecture and processing.
2. The study does not address whether protection remains effective if an attacker reprocesses the edited image through the same or different models. Multiple edits could potentially bypass the initial protection, which might limit the framework's resilience in real-world applications.

### Questions
1. To improve adaptability, the authors could investigate methods to generalize protective noise across different models or make the protective noise more universally applicable.
2. A thorough evaluation of the framework’s resilience to re-editing or re-input scenarios, where the modified image is fed back into the model, would strengthen the practical applicability. Demonstrating protection persistence across multiple editing cycles would address potential bypasses.

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
4

### Summary
This paper presents a runtime backdoor implantation method that protects sensitive image regions from unauthorized edits. By adding subtle protective noise during inference, the approach activates a backdoor only when targeted areas are modified, preventing tampering without retraining the model. Experimental results show effective defense against malicious editing, offering a practical solution for safeguarding image content in various applications.

### Strengths
1. Resource-Efficient Without Retraining: The proposed runtime backdoor injection method adds protective noise during inference, eliminating the need for model retraining and reducing resource consumption while preserving the model structure.
2. Region-Aware Trigger, Ensures Legitimate Edits: This method activates the backdoor protection only when specific protected regions are edited, allowing legitimate edits to proceed unaffected and optimizing the loss function to minimize impact on non-trigger regions.
3. Cross-Model Transferability: The approach shows strong transfer performance across various image editing models (e.g., LaMa, MAT, LDM), maintaining effectiveness even with data augmentation techniques like flipping and cropping, demonstrating robustness in diverse applications.

### Weaknesses
1. Trade-off between perturbation visibility and anti-editing effectiveness: While increasing the strength of the protective noise improves anti-editing performance, it also introduces noticeable pixel-level changes, which can degrade the visual quality of the image. This trade-off suggests that ensuring strong protection may be challenging without compromising user experience due to the visibility of the perturbation.
2. Dependence on specific trigger regions: The effectiveness of this method relies on designated trigger regions, meaning the backdoor protection only activates when these specific areas are edited. If malicious editing avoids these regions, the protection mechanism may not be effective, limiting the flexibility and applicability of the approach. The region-trigger mechanism also provides attackers with the possibility to evade detection.
3. Limited applicability to diffusion models: Although the method performs well in convolutional and Transformer models, it is less effective in diffusion models (e.g., LDM), especially when the perturbation strength is low, as the protection mechanism may fail. The denoising process in diffusion models tends to neutralize the protective noise, reducing the effectiveness of the approach in these types of models. It is recommended to increase the experiments on other types of generative models (such as GAN or VAE) to further verify the versatility of the method. At the same time, conduct more in-depth experiments on the diffusion model to optimize the perturbation or trigger mechanism and enhance the protection effect.

### Questions
1. Will the size of the entire perturbation region affect the effectiveness of the protection? What is the interaction between the noise magnitude and the trigger region? Additional experiments are needed to clarify this.
2. How robust is the proposed solution to various image enhancements (such as brightness adjustment, rotation, and scaling)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a method for image content protection by implanting run-time backdoors in image-editing models and triggering the backdoors when sensitive content on an image is modified by an image-editing model. The triggered backdoor will then stop the editing from happening. The backdoors are injected via imperceptible perturbations on the images and the protected sensitive area in an image can be defined to trigger the backdoor. Experiments are performed on several images and evaluated using standard image similarity metrics.

### Strengths
The method of implanting run-time backdoors to prevent editing of sensitive image content is interesting. 

The method, technique and experiments are also presented well.

### Weaknesses
It is not clear from the paper how robust the method is on image distortions and perturbations such as compression, smoothening, even before editing of sensitive image content is performed. In a realistic scenario, an image can go through such distortions. It is also not clear from the paper, how the method will respond to global image transformations such as blur.

### Questions
What are the limitations of the proposed approach? If an adversary is aware of this approach, will the adversary be able to evade?

### Soundness
3

### Presentation
3

### Contribution
3
